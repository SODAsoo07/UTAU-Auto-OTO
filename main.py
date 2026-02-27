"""
UTAU Auto OTO - 한국어/일본어 UTAU 음원 자동 OTO 생성기
윈도우 독립 실행형 GUI 프로그램 (CustomTkinter 기반)
"""

import os
import sys
import subprocess as sp
import threading
import traceback
import datetime
import logging
from tkinter import filedialog, messagebox
import tkinter as tk
import json

try:
    import customtkinter as ctk
except ImportError:
    print("customtkinter 모듈이 필요합니다. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'customtkinter'])
    import customtkinter as ctk

from core.lab_generator import generate_labs, generate_dictionary, verify_dict_lab_match
from core.oto_generator import (
    generate_oto,
    DEFAULT_PARAMS,
    train_kr_autotune_profile,
    save_kr_autotune_profile,
    apply_kr_autotune_profile_to_oto,
)
from core.ja_lab_generator import generate_ja_labs, generate_ja_dictionary
from core.ja_oto_generator import (
    generate_ja_oto,
    JA_DEFAULT_PARAMS,
    train_ja_autotune_profile,
    save_ja_autotune_profile,
    apply_ja_autotune_profile_to_oto,
)
from core.oto_validator import validate_oto_timing
from core.mfa_runner import find_mfa_executable, check_mfa_model, download_mfa_model, run_mfa_align, patch_mfa_korean_support

# ==============================================================================
# 로깅 설정
# ==============================================================================

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.datetime.now().strftime('log_%Y%m%d_%H%M%S.txt')
LOG_PATH = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 앱 상수
# ==============================================================================

APP_NAME = "UTAU Auto OTO Generator"
APP_VERSION = "2.0.0"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 750

LANGUAGE_OPTIONS = [
    "한국어 (CVC/CVVC/연속음/연단음 자동 매핑)",
    "日本語 (CVVC/연속음/연단음 자동 매핑)"
]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # MFA 경로
        self.mfa_path = find_mfa_executable()
        
        # OpenUtau 호환 에일리어스 생성 여부
        self.openutau_var = ctk.BooleanVar(value=False)
        self.gen_missing_vowels_var = ctk.BooleanVar(value=True)
        self.no_base_oto_var = ctk.BooleanVar(value=False)
        
        # 언어 선택
        self.language_var = ctk.StringVar(value=LANGUAGE_OPTIONS[0])

        # 작업 상태
        self.is_running = False

        # 사용자 옵션
        self.custom_phoneme_var = ctk.StringVar(value="")  # 커스텀 매핑 파일 경로
        self.alias_suffix_var = ctk.StringVar(value="")    # 에일리어스 접미사 (예: C4)

        self._build_ui()
        self._load_config()

        logger.info(f"{APP_NAME} v{APP_VERSION} 시작")
        if self.mfa_path:
            logger.info(f"MFA 경로: {self.mfa_path}")
        else:
            logger.warning("MFA를 찾을 수 없습니다.")

    def _build_ui(self):
        # ── 상단: 경로 설정 영역 ──
        path_frame = ctk.CTkFrame(self)
        self.auto_format_var = ctk.StringVar(value="자동 감지 (권장)")

        # 경로 프레임
        path_frame = ctk.CTkFrame(self)
        path_frame.pack(fill="x", padx=15, pady=10)

        # ── 상단: 버전 & 언어 ──
        lang_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        lang_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(lang_frame, text=f"Auto OTO {APP_VERSION}", font=("", 16, "bold"), text_color="#64B5F6").pack(side="left")
        
        self.lang_var = ctk.StringVar(value="Korean (한국어)") # Changed from self.language_var to self.lang_var
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_frame, values=["Korean (한국어)", "Japanese (日本語)"], variable=self.lang_var,
            command=self._on_language_change, width=150 # Changed command to _on_language_change
        )
        self.lang_dropdown.pack(side="left", padx=10)
        self.lang_info_label = ctk.CTkLabel(lang_frame, text="", text_color="gray")
        self.lang_info_label.pack(side="left", padx=10)

        # 특수발음 커스텀 맵 (Option)
        row0 = ctk.CTkFrame(path_frame, fg_color="transparent")
        row0.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row0, text="특수 발음 (Option):", width=120, anchor="w").pack(side="left")
        self.custom_entry = ctk.CTkEntry(row0, placeholder_text="커스텀 매핑 규칙 파일 (.txt) (선택)", textvariable=self.custom_phoneme_var)
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row0, text="찾아보기", width=90, command=lambda: self._browse_file(self.custom_entry, [("Text 파일", "*.txt")])).pack(side="right")

        # 생성 에일리어스 접미사 (Option)
        row0b = ctk.CTkFrame(path_frame, fg_color="transparent")
        row0b.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row0b, text="접미사 (Option):", width=120, anchor="w").pack(side="left")
        self.suffix_entry = ctk.CTkEntry(row0b, placeholder_text="예: C4 (모든 에일리어스 끝에 _C4 형태로 부여)", textvariable=self.alias_suffix_var)
        self.suffix_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))

        # WAV 폴더
        row1 = ctk.CTkFrame(path_frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row1, text="WAV 폴더:", width=120, anchor="w").pack(side="left")
        self.wav_entry = ctk.CTkEntry(row1, placeholder_text="WAV 파일이 있는 폴더 경로")
        self.wav_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row1, text="찾아보기", width=90, command=lambda: self._browse_folder(self.wav_entry)).pack(side="right")

        # 템플릿 OTO (선택)
        row2 = ctk.CTkFrame(path_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row2, text="템플릿 OTO:", width=120, anchor="w").pack(side="left")
        self.tpl_entry = ctk.CTkEntry(row2, placeholder_text="선택 사항 (없을 시 자동 생성 포맷으로 파일명/라벨 기반 생성)")
        self.tpl_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        self.tpl_browse_btn = ctk.CTkButton(
            row2,
            text="찾아보기",
            width=90,
            command=lambda: self._browse_file(self.tpl_entry, [("OTO 파일", "*.ini")])
        )
        self.tpl_browse_btn.pack(side="right")

        row2b = ctk.CTkFrame(path_frame, fg_color="transparent")
        row2b.pack(fill="x", padx=10, pady=(0, 3))
        ctk.CTkLabel(row2b, text="", width=120).pack(side="left")
        self.no_base_oto_checkbox = ctk.CTkCheckBox(
            row2b,
            text="'템플릿 OTO 없음' (CVVC/VCV OpenUtau 호환 에일리어스 자동 생성)",
            variable=self.no_base_oto_var,
            command=self._on_no_base_oto_toggle,
            text_color="#A5D6A7",
        )
        self.no_base_oto_checkbox.pack(side="left", padx=(5, 0))

        # 형식 지정 (템플릿 유무와 무관하게 선택 가능, 자동 감지 선택 시 판별)
        row_format = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_format.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_format, text="형식 지정:", width=120, anchor="w").pack(side="left")
        
        self.auto_format_var = ctk.StringVar(value="자동 감지 (권장)")
        FORMAT_OPTIONS = ["자동 감지 (권장)", "CVC/연단음", "CVVC", "VCV (연속음)"]
        self.format_dropdown = ctk.CTkOptionMenu(
            row_format, values=FORMAT_OPTIONS, variable=self.auto_format_var,
            width=200, command=self._save_config
        )
        self.format_dropdown.pack(side="left", padx=(5, 5))
        ctk.CTkLabel(row_format, text="(※ 선택 시 템플릿이 있어도 해당 형식으로 처리, 미선택 시 자동 감지)", text_color="gray").pack(side="left", padx=10)

        # 출력 경로
        row3 = ctk.CTkFrame(path_frame, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=(3, 10))
        ctk.CTkLabel(row3, text="출력 경로:", width=120, anchor="w").pack(side="left")
        self.out_entry = ctk.CTkEntry(row3, placeholder_text="생성된 oto.ini 저장 경로")
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row3, text="저장", width=90, command=lambda: self._browser_save(self.out_entry, [("OTO 파일", "*.ini")])).pack(side="right")

        # ── 중간: 탭 영역 ──
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=5)

        self._build_pipeline_tab()
        self._build_params_tab()
        self._build_profile_tune_tab()
        self._build_log_tab()

        # ── 하단: 상태 바 & 버튼 ──
        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=15, pady=(5, 15))

        self.status_label = ctk.CTkLabel(bottom, text="대기 중", anchor="w", text_color="gray")
        self.status_label.pack(side="left", padx=10)

        self.run_btn = ctk.CTkButton(bottom, text="▶ 전체 실행", font=("", 14, "bold"),
                                      width=150, height=40, command=self._run_full_pipeline)
        self.run_btn.pack(side="right", padx=10)

        self.report_btn = ctk.CTkButton(bottom, text="🐛 오류 제보", width=120, height=40,
                                         fg_color="#FF6B6B", hover_color="#EE5A5A",
                                         command=self._export_error_report)
        self.report_btn.pack(side="right", padx=5)

    def _build_pipeline_tab(self):
        tab = self.tabview.add("🚀 파이프라인")

        # ── MFA 상태 표시 영역 ──
        mfa_frame = ctk.CTkFrame(tab, border_width=1, border_color="#444")
        mfa_frame.pack(fill="x", padx=10, pady=(5, 10))
        mfa_inner = ctk.CTkFrame(mfa_frame, fg_color="transparent")
        mfa_inner.pack(fill="x", padx=10, pady=8)

        if self.mfa_path:
            self.mfa_status_label = ctk.CTkLabel(mfa_inner, text="✅ MFA 설치됨", font=("", 13, "bold"), text_color="#66BB6A")
        else:
            self.mfa_status_label = ctk.CTkLabel(mfa_inner, text="❌ MFA 미설치", font=("", 13, "bold"), text_color="#FF6B6B")
        self.mfa_status_label.pack(side="left")

        self.mfa_install_btn = ctk.CTkButton(
            mfa_inner, text="⬇ MFA 자동 설치", width=150,
            fg_color="#FFA726", hover_color="#FB8C00", text_color="black",
            command=self._run_mfa_setup
        )
        if self.mfa_path:
            self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
        self.mfa_install_btn.pack(side="right")

        # ── 파이프라인 단계들 ──
        steps = [
            ("1️⃣ Lab 파일 생성", "WAV 파일명을 기반으로 Lab 파일을 자동으로 생성합니다.", self._run_lab_gen),
            ("2️⃣ 사전(Dictionary) 생성", "Lab 파일로부터 MFA용 음소 사전 파일을 만듭니다.", self._run_dict_gen),
            ("3️⃣ MFA 음성 정렬", "Montreal Forced Aligner를 실행하여 TextGrid를 만듭니다.", self._run_mfa),
            ("4️⃣ OTO.ini 자동 생성", "TextGrid와 튜닝 파라미터로 최종 OTO를 계산합니다.", self._run_oto_gen),
        ]

        for title, desc, cmd in steps:
            frame = ctk.CTkFrame(tab)
            frame.pack(fill="x", padx=10, pady=5)
            left = ctk.CTkFrame(frame, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            ctk.CTkLabel(left, text=title, font=("", 14, "bold"), anchor="w").pack(anchor="w")
            ctk.CTkLabel(left, text=desc, text_color="gray", anchor="w", wraplength=500).pack(anchor="w")
            
            # 4단계인 경우에만 OpenUtau 옵션 체크박스 추가
            if "OTO" in title:
                opt_frame = ctk.CTkFrame(left, fg_color="transparent")
                opt_frame.pack(fill="x", pady=(5, 0))
                ctk.CTkCheckBox(
                    opt_frame, 
                    text="OpenUtau Phonemizer 호환 에일리어스 동시 생성 (k, t, p 종성 규칙 및 - CV 연음 등 복제 추가)",
                    text_color="#90CAF9",
                    variable=self.openutau_var,
                    command=self._save_config
                ).pack(anchor="w")
                
                self.gen_missing_vowels_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text="템플릿에 없더라도 기본 단모음(a, e, i, o, u...) 에일리어스 자동 생성",
                    text_color="#A5D6A7",
                    variable=self.gen_missing_vowels_var,
                    command=self._save_config
                )
                self.gen_missing_vowels_checkbox.pack(anchor="w", pady=(5, 0))
                
            ctk.CTkButton(frame, text="실행", width=80, command=cmd).pack(side="right", padx=10)

    def _build_params_tab(self):
        tab = self.tabview.add("🎛️ 튜닝 파라미터")

        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.param_vars = {}

        param_groups = [
            ("일반 VC 파라미터", [
                ('VC_CONSONANT_RATIO', 'VC 자음 사용 비율', 0.1, 1.0, 0.05),
                ('VC_VOWEL_START', 'VC 모음 시작 위치', 0.1, 1.0, 0.05),
                ('VC_PRE_OFFSET', 'VC 선행발성 오프셋 (ms)', 0, 50, 1),
                ('VC_OVL_RATIO', 'VC 오버랩 비율', 0.1, 1.0, 0.05),
            ]),
            ("일반 CV 파라미터", [
                ('CV_PRE_RATIO', 'CV 선행발성 비율', 0.1, 1.0, 0.05),
                ('CV_OVL_RATIO', 'CV 오버랩 비율', 0.1, 1.0, 0.05),
            ]),
            ("이중모음 CV 파라미터", [
                ('DIPHTHONG_CV_PRE_RATIO', '이중모음 CV 선행발성 비율', 0.1, 1.0, 0.05),
                ('DIPHTHONG_CV_CONSONANT_RATIO', '이중모음 CV 고정 비율', 0.1, 1.0, 0.05),
            ]),
            ("이중모음 VC 파라미터", [
                ('DIPHTHONG_VC_VOWEL_START', '이중모음 VC 모음 시작', 0.1, 1.0, 0.05),
                ('DIPHTHONG_VC_CONSONANT', '이중모음 VC 자음 비율', 0.1, 1.0, 0.05),
            ]),
        ]

        for group_name, params in param_groups:
            ctk.CTkLabel(scroll, text=group_name, font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            for key, label, min_val, max_val, step in params:
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", padx=15, pady=2)

                default = DEFAULT_PARAMS.get(key, 0.5)
                var = ctk.DoubleVar(value=default)
                self.param_vars[key] = var

                ctk.CTkLabel(row, text=label, width=250, anchor="w").pack(side="left")

                val_label = ctk.CTkLabel(row, text=f"{default:.2f}", width=50)
                val_label.pack(side="right", padx=(5, 0))

                slider = ctk.CTkSlider(row, from_=min_val, to=max_val,
                                        number_of_steps=int((max_val - min_val) / step),
                                        variable=var,
                                        command=lambda v, lbl=val_label: lbl.configure(text=f"{v:.2f}"))
                slider.pack(side="right", fill="x", expand=True, padx=5)

    def _build_log_tab(self):
        tab = self.tabview.add("📋 로그")

        self.log_text = ctk.CTkTextbox(tab, font=("Consolas", 12), state="normal")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(btn_frame, text="로그 지우기", width=100, command=self._clear_log).pack(side="left")
        ctk.CTkButton(btn_frame, text="로그 파일 열기", width=120,
                       command=lambda: os.startfile(LOG_PATH) if os.path.exists(LOG_PATH) else None).pack(side="left", padx=5)

    def _build_profile_tune_tab(self):
        tab = self.tabview.add("🧩 프로파일 미세조정")

        self.tune_auto_oto_var = ctk.StringVar(value="")
        self.tune_manual_oto_var = ctk.StringVar(value="")
        self.tune_profile_out_var = ctk.StringVar(value="")
        self.tune_apply_target_var = ctk.StringVar(value="")

        container = ctk.CTkFrame(tab)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        desc = (
            "자동 생성 OTO와 수동 보정 OTO(일부만 있어도 가능)를 매칭해 "
            "델타 프로파일을 학습한 뒤, 선택한 OTO에 미세조정을 적용합니다."
        )
        ctk.CTkLabel(container, text=desc, text_color="gray", wraplength=760, justify="left").pack(
            fill="x", padx=10, pady=(8, 12)
        )

        def _row(parent, label, var, browse_cmd):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(row, text=label, width=170, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True, padx=(5, 5))
            ctk.CTkButton(row, text="찾기", width=90, command=browse_cmd).pack(side="right")
            return ent

        _row(
            container,
            "자동 OTO (입력):",
            self.tune_auto_oto_var,
            lambda: self._browse_file_by_var(self.tune_auto_oto_var, [("OTO 파일", "*.ini"), ("All", "*.*")]),
        )
        _row(
            container,
            "수동 OTO (참조):",
            self.tune_manual_oto_var,
            lambda: self._browse_file_by_var(self.tune_manual_oto_var, [("OTO 파일", "*.ini"), ("All", "*.*")]),
        )
        _row(
            container,
            "프로파일 저장 경로:",
            self.tune_profile_out_var,
            lambda: self._browse_save_by_var(self.tune_profile_out_var, [("JSON 파일", "*.json"), ("All", "*.*")], ".json"),
        )
        _row(
            container,
            "적용 대상 OTO:",
            self.tune_apply_target_var,
            lambda: self._browse_file_by_var(self.tune_apply_target_var, [("OTO 파일", "*.ini"), ("All", "*.*")]),
        )

        tip = (
            "팁: 적용 대상 OTO를 비워두면 자동 OTO 입력 파일에 바로 적용합니다. "
            "프로파일 경로를 비우면 자동 OTO와 같은 폴더에 저장합니다."
        )
        ctk.CTkLabel(container, text=tip, text_color="#9E9E9E", wraplength=760, justify="left").pack(
            fill="x", padx=10, pady=(8, 12)
        )

        btn_row = ctk.CTkFrame(container, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkButton(
            btn_row,
            text="프로파일 학습 + 적용",
            height=38,
            font=("", 14, "bold"),
            command=self._run_profile_finetune,
        ).pack(side="right")

    # ── 유틸리티 ──

    def _browse_folder(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_file(self, entry, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browser_save(self, entry, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".ini")
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_file_by_var(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_save_by_var(self, var, filetypes, defext):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=defext)
        if path:
            var.set(path)

    def _get_language(self):
        """현재 선택된 언어를 'korean' 또는 'japanese'로 반환"""
        sel = self.lang_var.get() # Changed from self.language_var to self.lang_var
        if 'Japanese' in sel:
            return 'japanese'
        return 'korean'

    def _on_language_change(self, value):
        if "Korean" in value:
            self.lang_info_label.configure(text="한국어 음소(a, k, ga 등) 자동 파싱 및 생성")
            self.gen_missing_vowels_checkbox.configure(state="normal")
        else:
            self.lang_info_label.configure(text="일본어 음소(a, k, ka 등) 자동 파싱 및 생성")
            self.gen_missing_vowels_checkbox.configure(state="normal")
        self._save_config()

    def _on_no_base_oto_toggle(self):
        no_base = bool(self.no_base_oto_var.get())
        if no_base:
            self.tpl_entry.configure(state="disabled")
            if hasattr(self, "tpl_browse_btn"):
                self.tpl_browse_btn.configure(state="disabled")
        else:
            self.tpl_entry.configure(state="normal")
            if hasattr(self, "tpl_browse_btn"):
                self.tpl_browse_btn.configure(state="normal")
        self._save_config()

    def _append_log(self, msg):
        """스레드 안전한 로그 출력"""
        def _do():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(0, _do)

    def _set_status(self, msg):
        def _do():
            self.status_label.configure(text=msg)
        self.after(0, _do)

    def _run_auto_validation(self, wav_dir, tg_folder, out_path):
        """OTO 생성 직후 자동 검증 (파일명/리스트 + mel + TextGrid 결합)"""
        self._append_log("🧪 OTO 자동 검증 시작...")
        summary = validate_oto_timing(
            wav_dir=wav_dir,
            tg_folder=tg_folder,
            oto_path=out_path,
            language=self._get_language(),
            callback=self._append_log,
        )
        err_count = summary.get("errors", 0)
        warn_count = summary.get("warnings", 0)
        if err_count > 0:
            self._append_log(f"⚠ 자동 검증 결과: error {err_count}, warning {warn_count}")
        else:
            self._append_log(f"✅ 자동 검증 결과: warning {warn_count} (error 0)")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _set_running(self, running):
        self.is_running = running
        def _do():
            state = "disabled" if running else "normal"
            self.run_btn.configure(state=state)
        self.after(0, _do)

    def _get_params(self):
        """현재 슬라이더 값으로 파라미터 딕셔너리 생성"""
        return {key: var.get() for key, var in self.param_vars.items()}

    # ── MFA 설치 (GUI 내장) ──

    def _run_mfa_setup(self):
        """GUI 안에서 MFA 포터블 환경을 자동 설치합니다."""
        def task():
            self._set_running(True)
            self._set_status("⬇ MFA 자동 설치 중... (10~20분 소요)")
            try:
                import shutil
                portable_env_dir = os.path.join(APP_DIR, '.env')
                public_root = os.environ.get('PUBLIC', r'C:\Users\Public')
                fallback_env_dir = os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env')
                env_dir = portable_env_dir
                if any(ord(ch) > 127 for ch in portable_env_dir):
                    env_dir = fallback_env_dir
                    self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어 MFA 환경을 공용 폴더에 설치합니다.")
                    self._append_log(f"   대체 설치 경로: {env_dir}")
                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                installer = os.path.join(APP_DIR, 'Miniconda3-latest-Windows-x86_64.exe')

                # 이미 설치 확인
                if os.path.exists(mfa_exe):
                    self._append_log("✅ MFA가 이미 설치되어 있습니다!")
                    self.mfa_path = mfa_exe
                    self._update_mfa_status(True)
                    self._set_status("✅ MFA 준비 완료")
                    return

                system_conda = shutil.which('conda')

                if system_conda:
                    self._append_log(f"🔍 시스템에 설치된 Conda 발견: {system_conda}")
                    self._append_log("   Miniconda 다운로드를 건너뛰고 기존 Conda를 활용하여 환경을 구성합니다.")
                    self._append_log("[1/2] 🔧 MFA 전용 로컬 환경 생성 및 설치 중... (5~10분, 용량이 큽니다)")
                    
                    cmd = [system_conda, 'create', '-y', '-p', env_dir, '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama']
                    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace')
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    
                    if process.returncode != 0:
                        self._append_log("❌ MFA 설치 실패")
                        return

                    self._append_log("[2/2] 📦 추가 의존성 모듈 설치 중...")
                    # conda run을 사용하여 해당 환경 내에서 pip 실행 보장
                    sp.run([system_conda, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                    self._append_log("✅ MFA 시스템 구성 완료!")
                
                else:
                    self._append_log("🔍 시스템 Conda를 찾을 수 없습니다. 자체 Miniconda 포터블 환경을 구축합니다.")
                    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
                    # Step 1: Miniconda 다운로드
                    if not os.path.exists(conda_exe):
                        if not os.path.exists(installer):
                            self._append_log("[1/3] 📥 Miniconda 다운로드 중... (약 80MB)")
                            url = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
                            ps_cmd = (
                                f'[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; '
                                f"Invoke-WebRequest -Uri '{url}' -OutFile '{installer}'"
                            )
                            result = sp.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True)
                            if result.returncode != 0:
                                self._append_log(f"❌ 다운로드 실패: {result.stderr}")
                                return
                        self._append_log("✅ Miniconda 다운로드 완료!")

                        # Step 2: 포터블 설치
                        self._append_log("[2/3] 📦 Miniconda 포터블 설치 중... (2~5분)")
                        self._append_log(f"   설치 경로: {env_dir}")
                        # Miniconda(NSIS)는 /D= 경로를 raw command-line에서 파싱한다.
                        # subprocess(list)는 공백 경로를 자동 인용하면서 /D가 무시될 수 있어,
                        # 직접 command-line 문자열로 실행한다.
                        if os.path.isdir(env_dir) and not os.path.exists(conda_exe):
                            try:
                                shutil.rmtree(env_dir)
                                self._append_log("   이전 실패 흔적(.env 폴더)을 정리하고 재시도합니다.")
                            except Exception as cleanup_error:
                                self._append_log(f"❌ 기존 .env 폴더 정리 실패: {cleanup_error}")
                                return

                        install_cmd = (
                            f'"{installer}" /InstallationType=JustMe /RegisterPython=0 '
                            f'/AddToPath=0 /S /D={env_dir}'
                        )
                        result = sp.run(
                            install_cmd,
                            capture_output=True, text=True, timeout=1200
                        )
                        if result.returncode != 0 or not os.path.exists(conda_exe):
                            # /D 경로가 무시된 경우 기본 경로에 설치되었는지 보정 탐지
                            user_home = os.path.expanduser('~')
                            fallback_conda_candidates = [
                                os.path.join(user_home, 'miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(user_home, 'Miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Miniconda3', 'Scripts', 'conda.exe'),
                            ]
                            detected_conda = next((p for p in fallback_conda_candidates if p and os.path.exists(p)), None)
                            if detected_conda:
                                conda_exe = detected_conda
                                env_dir = os.path.dirname(os.path.dirname(conda_exe))
                                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                                self._append_log("⚠ 지정 경로에서 Conda를 찾지 못했지만 기본 설치 경로를 감지했습니다.")
                                self._append_log(f"   감지된 Conda: {conda_exe}")
                            else:
                                self._append_log(f"❌ Miniconda 설치 실패 (code={result.returncode})")
                                if result.stdout and result.stdout.strip():
                                    self._append_log(f"   stdout: {result.stdout.strip()[:500]}")
                                if result.stderr and result.stderr.strip():
                                    self._append_log(f"   stderr: {result.stderr.strip()[:500]}")
                                return
                        self._append_log("✅ Miniconda 설치 완료!")

                    # Step 3: MFA 설치
                    self._append_log("[3/3] 🔧 MFA 설치 중... (5~10분, 용량이 큽니다)")
                    # HTTP 000 에러를 막기 위해 --override-channels 로 conda-forge 만 강제 사용
                    process = sp.Popen(
                        [conda_exe, 'install', '-y', '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama'],
                        stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace'
                    )
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    if process.returncode != 0:
                        self._append_log("❌ MFA 설치 실패")
                        return

                    self._append_log("✅ Conda 패키지 설치 완료!")
                    
                    self._append_log("[3.5/4] 📦 추가 의존성 모듈 설치 중...")
                    sp.run([conda_exe, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                # Step 4: 한국어 모델
                self._append_log("[마지막] 🌐 한국어 음향 모델 다운로드 중... (1~2분)")
                process = sp.Popen(
                    [mfa_exe, 'model', 'download', 'acoustic', 'korean_mfa', '--ignore_cache'],
                    stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace'
                )
                for line in process.stdout:
                    stripped = line.strip()
                    if stripped:
                        self._append_log(stripped)
                process.wait()

                # 설치파일 정리
                if os.path.exists(installer):
                    os.remove(installer)

                self.mfa_path = mfa_exe
                self._update_mfa_status(True)
                self._append_log("")
                self._append_log("🎉 MFA 설치가 모두 완료되었습니다!")
                self._append_log("   이제 '3️⃣ MFA 음성 정렬' 버튼을 사용할 수 있습니다.")
                self._set_status("✅ MFA 설치 완료!")

            except Exception as e:
                self._handle_error("MFA 설치", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _update_mfa_status(self, installed):
        """MFA 상태 UI를 업데이트합니다."""
        def _do():
            if installed:
                self.mfa_status_label.configure(text="✅ MFA 설치됨", text_color="#66BB6A")
                self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                self.mfa_status_label.configure(text="❌ MFA 미설치", text_color="#FF6B6B")
                self.mfa_install_btn.configure(text="⬇ MFA 자동 설치", state="normal", fg_color="#FFA726")
        self.after(0, _do)

    # ── 개별 실행 ──

    def _run_in_thread(self, func):
        if self.is_running:
            messagebox.showwarning("실행 중", "이미 작업이 진행 중입니다. 완료 후 다시 시도해 주세요.")
            return
        self.tabview.set("📋 로그")
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1️⃣ Lab 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                if self._get_language() == 'japanese':
                    count, total, errors = generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, total, errors = generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠️ {e}")
                self._set_status(f"✅ Lab 생성 완료 ({count}/{total})")
            except Exception as e:
                self._handle_error("Lab 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("2️⃣ 사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                lang = self._get_language()
                if lang == 'japanese':
                    dict_filename = "japanese_dict.txt"
                else:
                    dict_filename = "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                
                if lang == 'japanese':
                    count, entries, errors = generate_ja_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, entries, errors = generate_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠️ {e}")
                self._append_log(f"📘 사전 저장 경로: {dict_path}")
                self._set_status(f"✅ 사전 생성 완료 ({entries}개 항목)")
            except Exception as e:
                self._handle_error("사전 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_mfa(self):
        def task():
            self._set_running(True)
            self._set_status("3️⃣ MFA 음성 정렬 중... (시간이 걸릴 수 있습니다)")
            try:
                if not self.mfa_path:
                    self._append_log("❌ MFA 실행 파일을 찾을 수 없습니다!")
                    self._append_log("   💡 MFA를 설치하거나, 프로그램 폴더에 .env/ 포터블 환경을 배치해 주세요.")
                    self._set_status("❌ MFA 미설치")
                    return

                wav_dir = self.wav_entry.get()
                lang = self._get_language()
                dict_filename = "japanese_dict.txt" if lang == 'japanese' else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                output_dir = os.path.join(wav_dir, "textgrids")

                has_model, msg = check_mfa_model(self.mfa_path, language=lang)
                self._append_log(msg)
                if not has_model:
                    download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)

                success, err = run_mfa_align(self.mfa_path, wav_dir, dict_path, output_dir, language=lang, callback=self._append_log)
                if success:
                    self._set_status("✅ MFA 정렬 완료")
                else:
                    self._append_log(f"❌ MFA 실패: {err}")
                    self._set_status("❌ MFA 실패")
            except Exception as e:
                self._handle_error("MFA 정렬", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_oto_gen(self):
        def task():
            self._set_running(True)
            self._set_status("4️⃣ OTO.ini 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()

                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return
                if not out_path:
                    self._append_log("❌ 출력 경로를 입력해 주세요.")
                    return

                tg_folder = os.path.join(wav_dir, "textgrids")
                if not os.path.exists(tg_folder):
                    self._append_log("❌ TextGrid 폴더를 찾을 수 없습니다. (3단계 생략 시 오류 발생 가능)")
                
                # 템플릿은 없어도 무방함
                
                params = self._get_params()
                gen_ou = self.openutau_var.get()
                gen_missing = self.gen_missing_vowels_var.get()
                auto_format = self.auto_format_var.get()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                alias_suffix = self.alias_suffix_var.get().strip()
                if self.no_base_oto_var.get():
                    self._append_log("ℹ '베이스 OTO 없음' 선택: 템플릿 없이 OpenUtau 호환 자동 에일리어스 생성 모드로 실행합니다.")

                lang = self._get_language()
                if lang == 'japanese':
                    processed, total, errors = generate_ja_oto(
                        tg_folder, tpl_path, out_path,
                        params=None,
                        generate_openutau=gen_ou,
                        gen_missing_vowels=gen_missing,
                        auto_format=auto_format,
                        custom_phonemes_path=custom_phonemes_path,
                        alias_suffix=alias_suffix,
                        callback=self._append_log
                    )
                else:
                    processed, total, errors = generate_oto(
                        tg_folder, tpl_path, out_path,
                        params,
                        gen_ou,
                        gen_missing,
                        auto_format=auto_format,
                        custom_phonemes_path=custom_phonemes_path,
                        alias_suffix=alias_suffix,
                        callback=self._append_log
                    )
                self._run_auto_validation(wav_dir, tg_folder, out_path)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠️ {e}")
                    self._set_status(f"⚠ OTO 생성 완료(경고 {len(errors)}건) ({processed}/{total})")
                else:
                    self._set_status(f"✅ OTO 생성 완료 ({processed}/{total})")
            except Exception as e:
                self._handle_error("OTO 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_profile_finetune(self):
        def task():
            self._set_running(True)
            self._set_status("🧩 프로파일 미세조정 진행 중...")
            try:
                auto_oto = self.tune_auto_oto_var.get().strip()
                manual_oto = self.tune_manual_oto_var.get().strip()
                profile_out = self.tune_profile_out_var.get().strip()
                apply_target = self.tune_apply_target_var.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if not auto_oto or not os.path.exists(auto_oto):
                    self._append_log("❌ 자동 OTO 입력 파일 경로가 비어있거나 파일이 없습니다.")
                    return
                if not manual_oto or not os.path.exists(manual_oto):
                    self._append_log("❌ 수동 OTO 참조 파일 경로가 비어있거나 파일이 없습니다.")
                    return

                if not profile_out:
                    base_dir = os.path.dirname(os.path.abspath(auto_oto))
                    if self._get_language() == "japanese":
                        profile_out = os.path.join(base_dir, ".ja_user_autotune_profile.json")
                    else:
                        profile_out = os.path.join(base_dir, ".kr_user_autotune_profile.json")
                    self.tune_profile_out_var.set(profile_out)

                if not apply_target:
                    apply_target = auto_oto
                    self.tune_apply_target_var.set(apply_target)
                if not os.path.exists(apply_target):
                    self._append_log(f"❌ 적용 대상 OTO 파일을 찾을 수 없습니다: {apply_target}")
                    return

                lang = self._get_language()
                self._append_log(f"🧪 미세조정 학습 시작 ({'일본어' if lang == 'japanese' else '한국어'})")
                self._append_log(f"   자동 OTO: {auto_oto}")
                self._append_log(f"   수동 OTO: {manual_oto}")

                if lang == "japanese":
                    profile = train_ja_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠️ 학습 가능한 매칭 샘플이 부족합니다. (최소 8쌍 이상 권장)")
                        return
                    if not save_ja_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_ja_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )
                else:
                    profile = train_kr_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠️ 학습 가능한 매칭 샘플이 부족합니다. (최소 8쌍 이상 권장)")
                        return
                    if not save_kr_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_kr_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )

                pairs = int(profile.get("matched_pairs", 0))
                buckets = len((profile.get("buckets") or {}))
                self._append_log(f"✅ 프로파일 저장 완료: {profile_out}")
                self._append_log(f"✅ 학습 결과: matched_pairs={pairs}, buckets={buckets}")
                self._append_log(f"✅ 적용 완료: {changed} lines adjusted ({apply_target})")
                self._set_status(f"✅ 미세조정 완료 ({changed} lines)")
            except Exception as e:
                self._handle_error("프로파일 미세조정", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_full_pipeline(self):
        """전체 파이프라인을 순서대로 실행"""
        def task():
            self._set_running(True)
            try:
                # Step 1: Lab
                self._set_status("1/4 - Lab 파일 생성 중...")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                lang = self._get_language()
                if lang == 'japanese':
                    generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)

                # Step 2: Dictionary
                self._set_status("2/4 - 사전 파일 생성 중...")
                dict_filename = "japanese_dict.txt" if lang == 'japanese' else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                if lang == 'japanese':
                    generate_ja_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    generate_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)

                # Step 3: MFA
                if self.mfa_path:
                    self._set_status("3/4 - MFA 음성 정렬 중... (시간 소요)")
                    output_dir = os.path.join(wav_dir, "textgrids")
                    has_model, _ = check_mfa_model(self.mfa_path, language=lang)
                    if not has_model:
                        download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)
                    success, err = run_mfa_align(self.mfa_path, wav_dir, dict_path, output_dir, language=lang, callback=self._append_log)
                    if not success:
                        self._append_log(f"❌ MFA 실패: {err}")
                        self._append_log("   MFA 없이 계속 진행합니다 (기존 TextGrid 사용)")
                else:
                    self._append_log("⚠️ MFA가 설치되어 있지 않습니다. TextGrid를 직접 준비해 주세요.")

                # Step 4: OTO
                self._set_status("4/4 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()
                if out_path: # tpl_path는 이제 필수가 아님
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없음' 선택: 템플릿 없이 OpenUtau 호환 자동 에일리어스 생성 모드로 실행합니다.")

                    if lang == 'japanese':
                        generate_ja_oto(
                            tg_folder, tpl_path, out_path,
                            params=None,
                            generate_openutau=gen_ou,
                            gen_missing_vowels=gen_missing,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=self._append_log
                        )
                    else:
                        generate_oto(
                            tg_folder, tpl_path, out_path,
                            params,
                            gen_ou,
                            gen_missing,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=self._append_log
                        )
                    self._run_auto_validation(wav_dir, tg_folder, out_path)
                else:
                    self._append_log("⚠️ 출력 경로가 비어있어 OTO 생성을 건너뜁니다.")

                self._set_status("🎉 전체 파이프라인 완료!")
                self._append_log("\n" + "=" * 50)
                self._append_log("🎉 모든 작업이 완료되었습니다!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error("전체 파이프라인", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    # ── 에러 핸들링 & 제보 ──

    def _handle_error(self, step_name, exception):
        """에러를 로그에 기록하고 사용자에게 친절하게 안내"""
        tb = traceback.format_exc()
        logger.error(f"[{step_name}] {exception}\n{tb}")
        self._append_log(f"\n{'='*50}")
        self._append_log(f"❌ [{step_name}] 에서 오류가 발생했습니다!")
        self._append_log(f"   오류 내용: {exception}")
        self._append_log(f"   💡 '오류 제보' 버튼을 눌러 로그 파일을 개발자에게 보내주세요.")
        self._append_log(f"{'='*50}\n")
        self._set_status(f"❌ 에러 발생: {step_name}")

    def _export_error_report(self):
        """오류 제보용 로그 파일을 생성하여 사용자가 쉽게 공유할 수 있도록 합니다."""
        report_path = os.path.join(APP_DIR, f"error_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"=== UTAU Auto OTO 오류 보고서 ===\n")
                f.write(f"프로그램 버전: {APP_VERSION}\n")
                f.write(f"시간: {datetime.datetime.now().isoformat()}\n")
                f.write(f"OS: {sys.platform}\n")
                f.write(f"Python: {sys.version}\n")
                f.write(f"MFA 경로: {self.mfa_path or '미설치'}\n")
                f.write(f"\n--- 사용자 설정 ---\n")
                f.write(f"WAV 폴더: {self.wav_entry.get()}\n")
                f.write(f"템플릿 OTO: {self.tpl_entry.get()}\n")
                f.write(f"베이스 OTO 없음: {self.no_base_oto_var.get()}\n")
                f.write(f"출력 경로: {self.out_entry.get()}\n")
                f.write(f"\n--- 파라미터 ---\n")
                for key, var in self.param_vars.items():
                    f.write(f"{key}: {var.get()}\n")
                f.write(f"\n--- 로그 내용 ---\n")
                f.write(self.log_text.get("1.0", "end"))

                # 최근 로그 파일 내용도 첨부
                if os.path.exists(LOG_PATH):
                    f.write(f"\n--- 상세 로그 파일 ({LOG_PATH}) ---\n")
                    with open(LOG_PATH, 'r', encoding='utf-8') as lf:
                        f.write(lf.read())

            # 보고서 폴더 열기
            if sys.platform == 'win32':
                os.startfile(os.path.dirname(report_path))

            messagebox.showinfo(
                "오류 보고서 생성 완료",
                f"오류 보고서가 생성되었습니다!\n\n"
                f"📄 파일 위치:\n{report_path}\n\n"
                f"이 파일을 개발자에게 보내주시면\n"
                f"문제를 빠르게 해결할 수 있습니다.\n\n"
                f"💡 디스코드나 이메일로 이 파일을 첨부해 주세요!"
            )
        except Exception as e:
            messagebox.showerror("오류", f"보고서 생성 실패: {e}")

    # ── 설정 관리 ──

    def _save_config(self):
        """현재 설정을 config.json 파일에 저장합니다."""
        params = self._get_params()
        config = {
            "wav_dir": self.wav_entry.get(),
            "tpl_path": self.tpl_entry.get(),
            "out_path": self.out_entry.get(),
            "custom_phonemes": self.custom_phoneme_var.get(),
            "alias_suffix": self.alias_suffix_var.get(),
            "openutau_compatible": self.openutau_var.get(),
            "gen_missing_vowels": self.gen_missing_vowels_var.get(),
            "no_base_oto": self.no_base_oto_var.get(),
            "language": self.lang_var.get(),
            "auto_format": self.auto_format_var.get(),
            "tune_auto_oto": self.tune_auto_oto_var.get() if hasattr(self, "tune_auto_oto_var") else "",
            "tune_manual_oto": self.tune_manual_oto_var.get() if hasattr(self, "tune_manual_oto_var") else "",
            "tune_profile_out": self.tune_profile_out_var.get() if hasattr(self, "tune_profile_out_var") else "",
            "tune_apply_target": self.tune_apply_target_var.get() if hasattr(self, "tune_apply_target_var") else "",
            "params": params
        }
        config_path = os.path.join(APP_DIR, 'config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")

    def _load_config(self):
        """config.json 파일에서 설정을 불러옵니다."""
        config_path = os.path.join(APP_DIR, 'config.json')
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'wav_dir' in config: self.wav_entry.insert(0, config['wav_dir'])
            self.tpl_entry.delete(0, "end")
            self.tpl_entry.insert(0, config.get('tpl_path', '')) # Changed from last_tpl
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, config.get('out_path', '')) # Changed from last_out
            
            saved_auto = config.get('auto_format', '자동 감지 (권장)')
            auto_map = {
                "CVC (단독음)": "CVC/연단음",
                "CVVC (기본)": "CVVC",
                "VCV (연속음)": "VCV (연속음)",
            }
            saved_auto = auto_map.get(saved_auto, saved_auto)
            valid_auto_formats = {"자동 감지 (권장)", "CVC/연단음", "CVVC", "VCV (연속음)"}
            if saved_auto in valid_auto_formats:
                self.auto_format_var.set(saved_auto)
            
            if "custom_phonemes" in config:
                self.custom_phoneme_var.set(config.get("custom_phonemes", ""))
                self.custom_entry.delete(0, "end")
                self.custom_entry.insert(0, self.custom_phoneme_var.get())
            if "alias_suffix" in config:
                self.alias_suffix_var.set(config.get("alias_suffix", ""))
                self.suffix_entry.delete(0, "end")
                self.suffix_entry.insert(0, self.alias_suffix_var.get())
            if "openutau_compatible" in config:
                self.openutau_var.set(config["openutau_compatible"])
            if "gen_missing_vowels" in config:
                self.gen_missing_vowels_var.set(config["gen_missing_vowels"])
            if "no_base_oto" in config:
                self.no_base_oto_var.set(config["no_base_oto"])
            if "language" in config:
                self.lang_var.set(config["language"]) # Changed from self.language_var
            if "auto_format" in config:
                saved_auto = config["auto_format"]
                auto_map = {
                    "CVC (단독음)": "CVC/연단음",
                    "CVVC (기본)": "CVVC",
                    "VCV (연속음)": "VCV (연속음)",
                }
                saved_auto = auto_map.get(saved_auto, saved_auto)
                if saved_auto in {"자동 감지 (권장)", "CVC/연단음", "CVVC", "VCV (연속음)"}:
                    self.auto_format_var.set(saved_auto)

            if hasattr(self, "tune_auto_oto_var"):
                self.tune_auto_oto_var.set(config.get("tune_auto_oto", ""))
            if hasattr(self, "tune_manual_oto_var"):
                self.tune_manual_oto_var.set(config.get("tune_manual_oto", ""))
            if hasattr(self, "tune_profile_out_var"):
                self.tune_profile_out_var.set(config.get("tune_profile_out", ""))
            if hasattr(self, "tune_apply_target_var"):
                self.tune_apply_target_var.set(config.get("tune_apply_target", ""))
            
            self._on_language_change(self.lang_var.get()) # Changed from self.language_var
            self._on_no_base_oto_toggle()
            
            if 'params' in config:
                params = config['params']
                for key, val in params.items():
                    if key in self.param_vars:
                        self.param_vars[key].set(val)
        except Exception as e:
            logger.error(f"설정 불러오기 실패: {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
