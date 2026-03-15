# -*- coding: utf-8 -*-
import json
import os

import tkinter as tk
import customtkinter as ctk

from core.format_type_utils import normalize_auto_format_value
from ui.theme_tokens import LANGUAGE_DROPDOWN_THEME, LANGUAGE_NOTICE_THEME, PALETTE


class LayoutMixin:
    def _build_ui(self):
        self.auto_format_var = ctk.StringVar(value="자동 감지 (권장)")

        main_body = tk.PanedWindow(
            self,
            orient="horizontal",
            sashwidth=6,
            background=PALETTE.panel_bg,
            bd=0,
            relief="flat",
        )
        main_body.pack(fill="both", expand=True)

        left_wrap = ctk.CTkFrame(main_body, fg_color="transparent")
        right_wrap = ctk.CTkFrame(main_body, fg_color="transparent")
        left_panel = ctk.CTkFrame(left_wrap, fg_color="transparent")
        right_panel = ctk.CTkFrame(right_wrap, fg_color="transparent")
        left_panel.pack(fill="both", expand=True, padx=(15, 8), pady=(10, 5))
        right_panel.pack(fill="both", expand=True, padx=(8, 15), pady=(10, 5))
        main_body.add(left_wrap, minsize=520)
        main_body.add(right_wrap, minsize=520)

        def _place_sash():
            try:
                total = max(self.winfo_width(), 1200)
                left_ratio = 1.5
                right_ratio = 0.8
                denom = left_ratio + right_ratio
                left_width = max(520, int(total * (left_ratio / denom)))
                right_width = max(520, total - left_width)
                if left_width + right_width < total:
                    left_width = total - right_width
                main_body.sash_place(0, left_width, 0)
            except Exception:
                pass

        self.after(50, _place_sash)

        path_frame = ctk.CTkFrame(
            left_panel,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
            corner_radius=8,
        )
        path_frame.pack(fill="x", padx=0, pady=(0, 8))

        def _style_primary_button(widget):
            widget.configure(
                fg_color=PALETTE.primary_button_bg,
                hover_color=PALETTE.primary_button_hover,
                text_color=PALETTE.primary_button_text,
            )

        def _style_blue_menu(widget):
            widget.configure(
                fg_color=PALETTE.menu_bg,
                button_color=PALETTE.menu_button,
                button_hover_color=PALETTE.menu_button_hover,
                dropdown_fg_color=PALETTE.menu_dropdown_bg,
                dropdown_hover_color=PALETTE.menu_dropdown_hover,
                dropdown_text_color=PALETTE.menu_text,
                text_color=PALETTE.menu_text,
            )

        header_row = ctk.CTkFrame(path_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            header_row,
            text=f"Auto OTO {self.app_version}",
            font=("", 16, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(side="left")

        self.lang_notice_label = ctk.CTkLabel(
            path_frame,
            text="",
            corner_radius=8,
            fg_color=LANGUAGE_NOTICE_THEME["japanese"]["fg_color"],
            text_color=LANGUAGE_NOTICE_THEME["japanese"]["text_color"],
            anchor="w",
            justify="left",
            padx=12,
            pady=8,
        )
        self.lang_notice_label.pack(fill="x", padx=12, pady=(0, 10))

        form_body = ctk.CTkFrame(path_frame, fg_color="transparent")
        form_body.pack(fill="x", padx=12, pady=(0, 6))

        def build_form_row(parent):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=4)
            return row

        def build_left_label(parent, text, width=115):
            return ctk.CTkLabel(parent, text=text, width=width, anchor="w")

        lang_row = build_form_row(form_body)
        build_left_label(lang_row, "언어", width=115).pack(side="left")
        self.lang_var = ctk.StringVar(value="Korean (한국어)")
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_row,
            values=["Korean (한국어)", "Japanese (日本語)"],
            variable=self.lang_var,
            command=self._on_language_change,
            width=200,
        )
        self.lang_dropdown.configure(
            fg_color=PALETTE.lang_dropdown_default_bg,
            button_color=PALETTE.lang_dropdown_default_button,
            button_hover_color=PALETTE.lang_dropdown_default_hover,
            dropdown_fg_color=PALETTE.lang_dropdown_dropdown_bg,
            dropdown_hover_color=PALETTE.lang_dropdown_dropdown_hover,
            dropdown_text_color=PALETTE.menu_text,
            text_color=PALETTE.lang_dropdown_text,
        )
        self.lang_dropdown.pack(side="left", padx=(6, 12))
        self.lang_info_label = ctk.CTkLabel(lang_row, text="", text_color=PALETTE.neutral_text)
        self.lang_info_label.pack(side="left", fill="x", expand=True)

        row1 = build_form_row(form_body)
        build_left_label(row1, "WAV 폴더:").pack(side="left")
        self.wav_entry = ctk.CTkEntry(row1, placeholder_text="WAV 파일이 있는 폴더 경로")
        self.wav_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.wav_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        wav_browse_btn = ctk.CTkButton(row1, text="찾아보기", width=90, command=lambda: self._browse_folder(self.wav_entry))
        _style_primary_button(wav_browse_btn)
        wav_browse_btn.pack(side="right")

        row2 = build_form_row(form_body)
        build_left_label(row2, "템플릿 OTO:").pack(side="left")
        self.tpl_entry = ctk.CTkEntry(row2, placeholder_text="선택 사항 (없을 시 파일명/라벨 기반 자동 생성)")
        self.tpl_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.tpl_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        self.tpl_browse_btn = ctk.CTkButton(
            row2,
            text="찾아보기",
            width=90,
            command=lambda: self._browse_file(self.tpl_entry, [("OTO 파일", "*.ini")]),
        )
        _style_primary_button(self.tpl_browse_btn)
        self.tpl_browse_btn.pack(side="right")

        row2b = build_form_row(form_body)
        ctk.CTkLabel(row2b, text="", width=115).pack(side="left")
        self.no_base_oto_checkbox = ctk.CTkCheckBox(
            row2b,
            text="템플릿 OTO 없음 (OpenUtau 호환 에일리어스 자동 생성)",
            variable=self.no_base_oto_var,
            command=self._on_no_base_oto_toggle,
            text_color=PALETTE.success_text,
        )
        self.no_base_oto_checkbox.pack(side="left", padx=(6, 0))

        row3 = build_form_row(form_body)
        build_left_label(row3, "출력 경로:").pack(side="left")
        self.out_entry = ctk.CTkEntry(row3, placeholder_text="생성된 oto.ini 저장 경로")
        self.out_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        out_save_btn = ctk.CTkButton(
            row3,
            text="저장",
            width=90,
            command=lambda: self._browser_save(self.out_entry, [("OTO 파일", "*.ini")]),
        )
        _style_primary_button(out_save_btn)
        out_save_btn.pack(side="right")

        row_format = build_form_row(form_body)
        build_left_label(row_format, "형식 지정:").pack(side="left")
        format_options = self._get_auto_format_options("korean")
        self.format_dropdown = ctk.CTkOptionMenu(
            row_format,
            values=format_options,
            variable=self.auto_format_var,
            width=190,
            command=self._on_format_change,
        )
        _style_blue_menu(self.format_dropdown)
        self.format_dropdown.pack(side="left", padx=(6, 8))
        ctk.CTkLabel(
            row_format,
            text="(템플릿 유무와 무관하게 우선 적용)",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)
        row_format_extra = build_form_row(form_body)
        build_left_label(row_format_extra, "JP 에일리어스:").pack(side="left")
        self.ja_alias_style_menu = ctk.CTkOptionMenu(
            row_format_extra,
            values=["원본 그대로", "히라가나", "로마자"],
            variable=self.ja_alias_style_var,
            width=190,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.ja_alias_style_menu)
        self.ja_alias_style_menu.pack(side="left", padx=(6, 8))
        ctk.CTkLabel(
            row_format_extra,
            text="(일본어 OTO 적용)",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)




        row_align = build_form_row(form_body)

        self.row_aligner = ctk.CTkFrame(row_align, fg_color="transparent")
        self.row_aligner.pack(side="left", fill="x", expand=True, padx=(0, 10))
        build_left_label(self.row_aligner, "정렬 엔진:").pack(side="left")
        self.aligner_menu = ctk.CTkOptionMenu(
            self.row_aligner,
            values=["MFA", "No-MFA (Experimental)"],
            variable=self.aligner_var,
            width=190,
            command=self._on_aligner_change,
        )
        _style_blue_menu(self.aligner_menu)
        self.aligner_menu.pack(side="left", padx=(6, 8))
        self.aligner_help_label = ctk.CTkLabel(
            self.row_aligner,
            text="(기본은 MFA입니다. 필요 시 자동 설치됩니다.)",
            text_color=PALETTE.neutral_text,
        )
        self.aligner_help_label.pack(side="left", fill="x", expand=True)
        row_align_extra = build_form_row(form_body)
        build_left_label(row_align_extra, "MFA 정렬 프로필:").pack(side="left")
        self.mfa_align_profile_menu = ctk.CTkOptionMenu(
            row_align_extra,
            values=["기본", "정밀", "빠름"],
            variable=self.mfa_align_profile_var,
            width=190,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.mfa_align_profile_menu)
        self.mfa_align_profile_menu.pack(side="left", padx=(6, 8))
        ctk.CTkLabel(
            row_align_extra,
            text="(기본=정확도 균형)",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)





        self.row_aligner_advanced = build_form_row(form_body)
        ctk.CTkLabel(
            self.row_aligner_advanced,
            text="고급 정렬 옵션은 '고급 설정' 탭에서 켤 수 있습니다.",
            text_color=PALETTE.neutral_text,
            anchor="w",
        ).pack(side="left", padx=(121, 0))

        advanced_row = build_form_row(path_frame)
        self.advanced_toggle_btn = ctk.CTkButton(
            advanced_row,
            text="▶ 고급 옵션 (특수 발음/접미사)",
            width=260,
            fg_color=PALETTE.advanced_toggle_bg,
            hover_color=PALETTE.advanced_toggle_hover,
            text_color=PALETTE.advanced_toggle_text,
            border_width=1,
            border_color=PALETTE.advanced_toggle_border,
            command=self._toggle_advanced_options,
        )
        self.advanced_toggle_btn.pack(side="right", padx=(0, 12), pady=(0, 4))
        self.advanced_hint_label = ctk.CTkLabel(
            path_frame,
            text="고급 설정 탭은 고급 설정 탭에서 볼 수 있습니다.",
            text_color=PALETTE.hint_text,
            anchor="e",
        )
        self.advanced_hint_label.pack(fill="x", padx=12, pady=(0, 6))

        self.advanced_options_frame = ctk.CTkFrame(path_frame, fg_color="transparent")

        row0 = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0, text="특수 발음 (선택):", width=120, anchor="w").pack(side="left")
        self.custom_entry = ctk.CTkEntry(row0, placeholder_text="커스텀 매핑 규칙 파일 (.txt)", textvariable=self.custom_phoneme_var)
        self.custom_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        custom_browse_btn = ctk.CTkButton(row0, text="찾아보기", width=90, command=lambda: self._browse_file(self.custom_entry, [("Text 파일", "*.txt")]))
        _style_primary_button(custom_browse_btn)
        custom_browse_btn.pack(side="right")

        row0b = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0b.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0b, text="접미사 (선택):", width=120, anchor="w").pack(side="left")
        self.suffix_entry = ctk.CTkEntry(row0b, placeholder_text="예: C4 (모든 에일리어스 끝에 _C4 형태로 부여)", textvariable=self.alias_suffix_var)
        self.suffix_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.suffix_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))

        self._toggle_advanced_options(force=False)

        self.pipeline_action_host = ctk.CTkFrame(
            left_panel,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
            corner_radius=8,
        )
        self.pipeline_action_host.pack(side="bottom", fill="x", padx=0, pady=(8, 0))

        self.tabview = ctk.CTkTabview(right_panel)
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_pipeline_tab()
        self._build_params_tab()
        self._build_profile_tune_tab()
        self._build_advanced_settings_tab()
        self._build_log_tab()
        self.tabview.set("로그")

        bottom = ctk.CTkFrame(self)
        bottom.pack(side="bottom", fill="x", padx=15, pady=(5, 15))

        status_group = ctk.CTkFrame(bottom, fg_color="transparent")
        status_group.pack(side="left", fill="x", expand=True, padx=(10, 6))

        self.status_label = ctk.CTkLabel(status_group, text="대기 중", anchor="w", text_color=PALETTE.neutral_text)
        self.status_label.pack(fill="x")

        progress_row = ctk.CTkFrame(status_group, fg_color="transparent")
        progress_row.pack(fill="x", pady=(4, 0))
        self.progress_bar = ctk.CTkProgressBar(progress_row, height=12)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_label = ctk.CTkLabel(progress_row, text="0%", width=42, anchor="e", text_color=PALETTE.neutral_text)
        self.progress_label.pack(side="left", padx=(8, 0))

        self.run_btn = ctk.CTkButton(bottom, text="▶ 전체 실행", font=("", 14, "bold"), width=150, height=40, command=self._run_full_pipeline)
        self.run_btn.pack(side="right", padx=10)

        self.report_btn = ctk.CTkButton(
            bottom,
            text="🐛 오류 제보",
            width=120,
            height=40,
            fg_color=PALETTE.danger_button_bg,
            hover_color=PALETTE.danger_button_hover,
            command=self._export_error_report,
        )
        self.report_btn.pack(side="right", padx=5)
        self._sync_aligner_ui()

    def _get_language(self):
        """UI 언어 선택값을 내부 코드('korean'/'japanese')로 정규화합니다."""
        sel = str(self.lang_var.get() or "")
        low = sel.strip().lower()
        if (
            "japanese" in low
            or "日本" in sel
            or "にほん" in sel
            or low in {"ja", "jp"}
        ):
            return "japanese"
        if (
            "korean" in low
            or "한국" in sel
            or "조선" in sel
            or low in {"ko", "kr"}
        ):
            return "korean"
        return "korean"

    def _get_auto_format_options(self, language=None):
        lang = language or self._get_language()
        if lang == "korean":
            return ["자동 감지 (권장)", "CV/연단음", "CVC (한국어 전용)", "CVVC", "VCV (연속음)"]
        return ["자동 감지 (권장)", "CV/연단음", "CVVC", "VCV (연속음)"]

    def _set_auto_format_from_code(self, format_code, language=None):
        lang = language or self._get_language()
        values = self._get_auto_format_options(lang)
        if hasattr(self, "format_dropdown"):
            self.format_dropdown.configure(values=values)
        label_map = {
            "": "자동 감지 (권장)",
            "cv": "CV/연단음",
            "cvc": "CVC (한국어 전용)",
            "cvvc": "CVVC",
            "vcv": "VCV (연속음)",
        }
        label = label_map.get(str(format_code or "").strip().lower(), "자동 감지 (권장)")
        if label not in values:
            label = "CV/연단음" if label == "CVC (한국어 전용)" and lang != "korean" else "자동 감지 (권장)"
        self.auto_format_var.set(label)

    def _on_language_change(self, value):
        lang = self._get_language()
        if lang == "korean":
            self.lang_info_label.configure(text="한국어 단위(a, k, ga 등) 에일리어스를 기준으로 생성합니다.")
            self.lang_notice_label.configure(
                text="현재 언어: 한국어\nLab 생성, 사전 생성, 정렬, OTO 계산이 모두 한국어 규칙으로 진행됩니다.",
                fg_color=LANGUAGE_NOTICE_THEME["korean"]["fg_color"],
                text_color=LANGUAGE_NOTICE_THEME["korean"]["text_color"],
            )
            try:
                self.lang_dropdown.configure(
                    fg_color=LANGUAGE_DROPDOWN_THEME["korean"]["fg_color"],
                    button_color=LANGUAGE_DROPDOWN_THEME["korean"]["button_color"],
                    button_hover_color=LANGUAGE_DROPDOWN_THEME["korean"]["button_hover_color"],
                )
            except Exception:
                pass
            self.gen_missing_vowels_checkbox.configure(state="normal")
            if hasattr(self, "ja_alias_style_menu"):
                self.ja_alias_style_menu.configure(state="disabled")
        else:
            self.lang_info_label.configure(text="일본어 단위(a, k, ka 등) 에일리어스를 기준으로 생성합니다.")
            self.lang_notice_label.configure(
                text="현재 언어: 일본어\n특수 발음 기호가 섞인 파일은 Lab 생성 전에 언어 선택을 다시 확인하세요.",
                fg_color=LANGUAGE_NOTICE_THEME["japanese"]["fg_color"],
                text_color=LANGUAGE_NOTICE_THEME["japanese"]["text_color"],
            )
            try:
                self.lang_dropdown.configure(
                    fg_color=LANGUAGE_DROPDOWN_THEME["japanese"]["fg_color"],
                    button_color=LANGUAGE_DROPDOWN_THEME["japanese"]["button_color"],
                    button_hover_color=LANGUAGE_DROPDOWN_THEME["japanese"]["button_hover_color"],
                )
            except Exception:
                pass
            self.gen_missing_vowels_checkbox.configure(state="normal")
            if hasattr(self, "ja_alias_style_menu"):
                self.ja_alias_style_menu.configure(state="normal")
        current_code = normalize_auto_format_value(self._get_language(), self.auto_format_var.get())
        self._set_auto_format_from_code(current_code, self._get_language())
        if hasattr(self, "_apply_recommended_ml_model_defaults"):
            self._apply_recommended_ml_model_defaults()
        self._save_config()
        self._refresh_ml_backend_status()
    def _get_ja_alias_style_code(self):
        style = self.ja_alias_style_var.get().strip()
        if style == "히라가나":
            return "hiragana"
        if style == "로마자":
            return "romaji"
        return "original"

    def _get_mfa_align_profile_code(self):
        profile = str(self.mfa_align_profile_var.get() if hasattr(self, "mfa_align_profile_var") else "").strip()
        if profile in {"빠름 (저사양 추천)", "fast"}:
            return "fast"
        if profile in {"정확도 우선", "accurate", "accurate_adapted", "speaker_adapted"}:
            return "accurate"
        if profile in {"기본", "default", "정확도 우선 (기본)"}:
            return "default"
        return "default"

    def _on_aligner_change(self, _value=None):
        self._sync_aligner_ui()
        self._save_config()

    def _on_advanced_aligner_toggle(self):
        self._sync_aligner_ui()
        self._save_config()

    def _sync_aligner_ui(self):
        options = ["MFA", "No-MFA (Experimental)"]
        current = str(self.aligner_var.get() if hasattr(self, "aligner_var") else "MFA").strip()
        if current not in options:
            current = "MFA"
        if hasattr(self, "aligner_var"):
            self.aligner_var.set(current)
        if hasattr(self, "aligner_menu"):
            self.aligner_menu.configure(values=options)
            try:
                self.aligner_menu.set(current)
            except Exception:
                pass
        use_no_mfa = current == "No-MFA (Experimental)"
        if hasattr(self, "mfa_align_profile_menu"):
            self.mfa_align_profile_menu.configure(state="disabled" if use_no_mfa else "normal")
        if hasattr(self, "aligner_help_label"):
            if use_no_mfa:
                self.aligner_help_label.configure(text="(No-MFA experimental path. Boundary model quality is critical.)")
            else:
                self.aligner_help_label.configure(text="(기본은 MFA입니다. 정렬 버튼을 누르면 필요 시 자동 설치됩니다.)")
        if hasattr(self, "align_step_title_label") and hasattr(self, "align_step_desc_label"):
            if use_no_mfa:
                self.align_step_title_label.configure(text="3. 음성 경계 추정 (No-MFA)")
                self.align_step_desc_label.configure(text="MFA/TextGrid 생성을 건너뜁니다. No-MFA 경계 모델 기반 경로를 사용합니다.")
            else:
                self.align_step_title_label.configure(text="3. 음성 정렬 (MFA)")
                self.align_step_desc_label.configure(text="MFA로 TextGrid를 생성합니다. MFA가 없으면 자동 설치 후 계속 진행합니다.")

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

    def _on_format_change(self, _value=None):
        self._save_config()
        self._refresh_ml_backend_status()

    def _on_ml_backend_change(self, _value=None):
        self._save_config()
        self._refresh_ml_backend_status()

    def _on_ml_backend_detail_toggle(self):
        self._refresh_ml_backend_status()

    def _on_ml_model_root_change(self, _value=None):
        self._save_config()
        self._refresh_ml_backend_status()

    def _refresh_ml_backend_status(self):
        if not hasattr(self, "ml_coupled_status_label"):
            return
        try:
            from core.format_type_utils import normalize_auto_format_value
            from core.oto_ml_refiner import _resolve_backend_model_dir
        except Exception:
            return

        lang = self._get_language() if hasattr(self, "_get_language") else "korean"
        fmt = ""
        if hasattr(self, "auto_format_var"):
            fmt = normalize_auto_format_value(lang, self.auto_format_var.get())
        fmt_display = fmt or "auto"
        routed_fmt = fmt or "general"
        env_key = "UTOA_JA_OTO_ML_DIR" if lang == "japanese" else "UTOA_KR_OTO_ML_DIR"
        override = ""
        if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
            override = str(self.ml_model_root_ja_var.get() or "").strip()
        if lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
            override = str(self.ml_model_root_kr_var.get() or "").strip()

        prev_env = os.environ.get(env_key)
        try:
            if override:
                os.environ[env_key] = override
            else:
                os.environ.pop(env_key, None)
            ensemble_dir = _resolve_backend_model_dir(lang, routed_fmt, backend="ensemble_v1")
            lightgbm_dir = _resolve_backend_model_dir(lang, routed_fmt, backend="lightgbm")
        finally:
            if prev_env is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev_env

        def _status_icon(model_dir: str) -> str:
            return "OK" if model_dir else "--"

        selected = ""
        if hasattr(self, "ml_coupled_backend_var"):
            selected = str(self.ml_coupled_backend_var.get() or "auto").strip().lower()
        selected = {
            "ensemble_v1": "ensemble",
        }.get(selected, selected or "auto")

        text = (
            f"현재 포맷: {fmt_display} | ensemble {_status_icon(ensemble_dir)} / "
            f"lightgbm {_status_icon(lightgbm_dir)}"
        )
        if selected:
            text += f" | 선택: {selected}"
        if override:
            text += " | 사용자 경로 사용"
        self.ml_coupled_status_label.configure(text=text)

        if not hasattr(self, "ml_coupled_status_detail_label"):
            return
        show_detail = False
        if hasattr(self, "ml_coupled_status_detail_var"):
            try:
                show_detail = bool(self.ml_coupled_status_detail_var.get())
            except Exception:
                show_detail = False
        if not show_detail:
            self.ml_coupled_status_detail_label.configure(text="")
            return

        def _read_meta(model_dir: str) -> dict:
            if not model_dir:
                return {}
            meta_path = os.path.join(model_dir, "model_meta.json")
            if not os.path.isfile(meta_path):
                return {}
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}

        def _fmt_line(label: str, model_dir: str, meta: dict) -> str:
            if not model_dir:
                return f"{label}: (없음)"
            version = str(meta.get("model_version", "") or meta.get("version", "") or "unknown")
            backend = str(meta.get("backend", "") or "unknown")
            created = str(meta.get("created_at", "") or "unknown")
            return f"{label}: {model_dir} | backend={backend} | version={version} | created_at={created}"

        detail_lines = []
        if override:
            detail_lines.append(f"custom_root: {override}")
        detail_lines += [
            _fmt_line("ensemble", ensemble_dir or "", _read_meta(ensemble_dir) if ensemble_dir else {}),
            _fmt_line("lightgbm", lightgbm_dir or "", _read_meta(lightgbm_dir) if lightgbm_dir else {}),
        ]
        self.ml_coupled_status_detail_label.configure(text="\n".join(detail_lines))

    def _toggle_advanced_options(self, force=None):
        if force is None:
            self.advanced_options_expanded = not self.advanced_options_expanded
        else:
            self.advanced_options_expanded = bool(force)

        if self.advanced_options_expanded:
            self.advanced_toggle_btn.configure(text="▼ 고급 옵션 (특수 발음/접미사)")
            self.advanced_options_frame.pack(fill="x", padx=10, pady=(0, 3))
        else:
            self.advanced_toggle_btn.configure(text="▶ 고급 옵션 (특수 발음/접미사)")
            self.advanced_options_frame.pack_forget()

    def _get_params(self):
        """현재 슬라이더 값으로 파라미터 딕셔너리 생성"""
        return {key: var.get() for key, var in self.param_vars.items()}

    # ── MFA 설치 (GUI 내장) ──
