# -*- coding: utf-8 -*-
import json
import os

import tkinter as tk
import customtkinter as ctk

from core.format_type_utils import normalize_auto_format_value
from core.pipeline_status import normalize_aligner_name
from ui.theme_tokens import (
    LANGUAGE_DROPDOWN_THEME,
    LANGUAGE_NOTICE_THEME,
    PALETTE,
    THEME_PROFILE_OPTIONS,
    apply_theme_profile,
    get_theme_appearance_mode,
)
from ui.i18n import t

EN_CVVC_UI_ENABLED = False
NO_MFA_REMAP_LABEL = "베이스 OTO 재매핑 + 보정"
HSMM_OTO_LABEL = "HSMM OTO"
TAB_ORDER = ("파이프라인", "고급 설정", "파라미터 조정", "로그", "상세 로그", "크레딧")
DEVELOPER_ONLY_TABS = frozenset({"파라미터 조정", "상세 로그"})


class LayoutMixin:
    def _build_ui(self):
        self.configure(fg_color=PALETTE.panel_bg)
        try:
            pending_job = getattr(self, "_lazy_prewarm_job", None)
            if pending_job:
                self.after_cancel(pending_job)
        except Exception:
            pass
        self._lazy_prewarm_job = None
        self._lazy_prewarm_queue = []
        self._tab_build_pending = set()
        if not hasattr(self, "auto_format_var"):
            self.auto_format_var = ctk.StringVar(value="자동 감지 (권장)")
        self._ensure_param_vars()

        main_body = tk.PanedWindow(
            self,
            orient="horizontal",
            sashwidth=6,
            background=PALETTE.panel_bg,
            bd=0,
            relief="flat",
        )
        main_body.pack(fill="both", expand=True)

        left_wrap = ctk.CTkFrame(main_body, fg_color=PALETTE.panel_bg)
        right_wrap = ctk.CTkFrame(main_body, fg_color=PALETTE.panel_bg)
        left_panel = ctk.CTkFrame(left_wrap, fg_color=PALETTE.panel_bg)
        right_panel = ctk.CTkFrame(right_wrap, fg_color=PALETTE.panel_bg)
        left_panel.pack(fill="both", expand=True, padx=(15, 8), pady=(10, 5))
        right_panel.pack(fill="both", expand=True, padx=(8, 15), pady=(10, 5))
        left_scroll = ctk.CTkScrollableFrame(left_panel, fg_color=PALETTE.panel_bg)
        left_scroll.pack(fill="both", expand=True, padx=0, pady=0)
        main_body.add(left_wrap, minsize=340)
        main_body.add(right_wrap, minsize=320)

        def _place_sash(_event=None):
            try:
                total = max(self.winfo_width(), 900)
                left_ratio = 1.0
                right_ratio = 1.0
                denom = left_ratio + right_ratio
                left_width = max(340, int(total * (left_ratio / denom)))
                right_width = max(320, total - left_width)
                if left_width + right_width < total:
                    left_width = total - right_width
                main_body.sash_place(0, left_width, 0)
            except Exception:
                pass

        self.after(50, _place_sash)
        try:
            main_body.bind("<Configure>", _place_sash, add="+")
        except Exception:
            pass

        path_frame = ctk.CTkFrame(
            left_scroll,
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

        # Machine-translation notice — shown only when UI language is English or Japanese
        from ui.i18n import get_language as _get_ui_lang_now
        _mt_notice_texts = {
            "en": "Some translations are machine-generated and may contain errors.",
            "ja": "一部の翻訳は機械翻訳であり、誤りが含まれる場合があります。",
        }
        _mt_text = _mt_notice_texts.get(_get_ui_lang_now(), "")
        if _mt_text:
            ctk.CTkLabel(
                header_row,
                text=f"  ⚠ {_mt_text}",
                font=("", 11),
                text_color=PALETTE.neutral_text,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

        # UI language selector (left panel top)
        from ui.i18n import get_language as _get_ui_lang
        ui_lang_top_row = ctk.CTkFrame(path_frame, fg_color="transparent")
        ui_lang_top_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(ui_lang_top_row, text=t("UI 언어 (UI Language"), width=115, anchor="w").pack(side="left")
        if not hasattr(self, "ui_lang_var"):
            _cur_lang = _get_ui_lang()
            _ui_lang_display = "English" if _cur_lang == "en" else ("日本語" if _cur_lang == "ja" else "한국어")
            self.ui_lang_var = ctk.StringVar(value=_ui_lang_display)
        self.ui_lang_dropdown = ctk.CTkOptionMenu(
            ui_lang_top_row,
            values=["한국어", "English", "日本語"],
            variable=self.ui_lang_var,
            command=self._on_ui_language_change,
            width=200,
        )
        self.ui_lang_dropdown.pack(side="left", padx=(6, 12))
        self._ui_lang_hint_label = ctk.CTkLabel(ui_lang_top_row, text="", text_color=PALETTE.neutral_text)
        self._ui_lang_hint_label.pack(side="left")

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
        build_left_label(lang_row, t("언어"), width=115).pack(side="left")
        if not hasattr(self, "lang_var"):
            self.lang_var = ctk.StringVar(value="Korean (한국어)")
        language_values = ["Korean (한국어)", "Japanese (日本語)"]
        if str(getattr(self, "release_channel", "stable")).strip().lower() == "preview":
            language_values.append("English (Preview CVVC)")
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_row,
            values=language_values,
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
        build_left_label(row1, t("WAV 폴더:")).pack(side="left")
        self.wav_entry = ctk.CTkEntry(row1, placeholder_text=t("WAV 파일이 있는 폴더 경로"))
        self.wav_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.wav_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        wav_browse_btn = ctk.CTkButton(row1, text=t("찾아보기"), width=90, command=lambda: self._browse_folder(self.wav_entry))
        _style_primary_button(wav_browse_btn)
        wav_browse_btn.pack(side="right")

        row1b = build_form_row(form_body)
        ctk.CTkLabel(row1b, text="", width=115).pack(side="left")
        self.recursive_voicebank_scan_checkbox = ctk.CTkCheckBox(
            row1b,
            text=t("하위 폴더 자동 탐색(배치 처리)"),
            variable=self.recursive_voicebank_scan_var,
            command=self._save_config,
        )
        self.recursive_voicebank_scan_checkbox.pack(side="left", padx=(6, 0))

        row2 = build_form_row(form_body)
        build_left_label(row2, t("템플릿 OTO:")).pack(side="left")
        self.tpl_entry = ctk.CTkEntry(row2, placeholder_text=t("선택 사항 (없을 시 파일명/라벨 기반 자동 생성)"))
        self.tpl_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.tpl_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        self.tpl_browse_btn = ctk.CTkButton(
            row2,
            text=t("찾아보기"),
            width=90,
            command=lambda: self._browse_file(self.tpl_entry, [("OTO 파일", "*.ini")]),
        )
        _style_primary_button(self.tpl_browse_btn)
        self.tpl_browse_btn.pack(side="right")

        row2b = build_form_row(form_body)
        ctk.CTkLabel(row2b, text="", width=115).pack(side="left")
        self.no_base_oto_checkbox = ctk.CTkCheckBox(
            row2b,
            text=t("템플릿 OTO 없음 (OpenUtau 호환 에일리어스 자동 생성)"),
            variable=self.no_base_oto_var,
            command=self._on_no_base_oto_toggle,
            text_color=PALETTE.success_text,
        )
        self.no_base_oto_checkbox.pack(side="left", padx=(6, 0))

        row3 = build_form_row(form_body)
        build_left_label(row3, t("출력 경로:")).pack(side="left")
        self.out_entry = ctk.CTkEntry(row3, placeholder_text=t("생성된 oto.ini 저장 경로"))
        self.out_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        out_save_btn = ctk.CTkButton(
            row3,
            text=t("저장"),
            width=90,
            command=lambda: self._browser_save(self.out_entry, [("OTO 파일", "*.ini")]),
        )
        _style_primary_button(out_save_btn)
        out_save_btn.pack(side="right")

        row_suffix = build_form_row(form_body)
        build_left_label(row_suffix, t("접미사:")).pack(side="left")
        self.suffix_entry = ctk.CTkEntry(
            row_suffix,
            placeholder_text=t("선택 사항: 예: C4 (모든 에일리어스 끝에 _C4 형태로 부여)"),
            textvariable=self.alias_suffix_var,
        )
        self.suffix_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.suffix_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        suffix_apply_btn = ctk.CTkButton(
            row_suffix,
            text=t("적용"),
            width=90,
            command=self._apply_suffix_to_generated_oto,
        )
        _style_primary_button(suffix_apply_btn)
        suffix_apply_btn.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            row_suffix,
            text=t("(생성된 oto에 접미사 적용)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")

        row_boundary_model = build_form_row(form_body)
        build_left_label(row_boundary_model, t("경계 모델:")).pack(side="left")
        self.use_boundary_model_checkbox = ctk.CTkCheckBox(
            row_boundary_model,
            text=t("학습된 경계 모델 사용"),
            variable=self.use_boundary_model_var,
            command=self._on_use_boundary_model_toggle,
        )
        self.use_boundary_model_checkbox.pack(side="left", padx=(6, 8))
        self.boundary_model_entry = ctk.CTkEntry(
            row_boundary_model,
            placeholder_text=t("체크포인트(.pt) 경로 — 비우면 규칙 기반 사용"),
            textvariable=self.boundary_model_path_var,
        )
        self.boundary_model_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.boundary_model_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        boundary_browse_btn = ctk.CTkButton(
            row_boundary_model,
            text=t("찾아보기"),
            width=90,
            command=lambda: self._browse_file_by_var(self.boundary_model_path_var, [("PyTorch 체크포인트", "*.pt")]),
        )
        _style_primary_button(boundary_browse_btn)
        boundary_browse_btn.pack(side="left")

        row_format = build_form_row(form_body)
        build_left_label(row_format, t("형식 지정:")).pack(side="left")
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
            text=t("(템플릿 유무와 무관하게 우선 적용)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)
        self.ja_alias_row = build_form_row(form_body)
        build_left_label(self.ja_alias_row, t("JP 에일리어스:")).pack(side="left")
        self.ja_alias_style_menu = ctk.CTkOptionMenu(
            self.ja_alias_row,
            values=["원본 그대로", "히라가나", "로마자"],
            variable=self.ja_alias_style_var,
            width=190,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.ja_alias_style_menu)
        self.ja_alias_style_menu.pack(side="left", padx=(6, 8))
        self.ja_alias_hint_label = ctk.CTkLabel(
            self.ja_alias_row,
            text=t("(일본어 OTO 적용)"),
            text_color=PALETTE.neutral_text,
        )
        self.ja_alias_hint_label.pack(side="left", fill="x", expand=True)

        if EN_CVVC_UI_ENABLED:
            self.en_cvvc_row = build_form_row(form_body)
            build_left_label(self.en_cvvc_row, "EN CVVC", width=115).pack(side="left")

            self.en_cvvc_pack_menu = ctk.CTkOptionMenu(
                self.en_cvvc_row,
                values=["LITE", "FULL_GA"],
                variable=self.en_cvvc_pack_var,
                width=95,
                command=lambda _v: self._save_config(),
            )
            _style_blue_menu(self.en_cvvc_pack_menu)
            self.en_cvvc_pack_menu.pack(side="left", padx=(6, 6))

            self.en_cvvc_beat_menu = ctk.CTkOptionMenu(
                self.en_cvvc_row,
                values=["8-beat", "4-beat"],
                variable=self.en_cvvc_beat_var,
                width=95,
                command=lambda _v: self._save_config(),
            )
            _style_blue_menu(self.en_cvvc_beat_menu)
            self.en_cvvc_beat_menu.pack(side="left", padx=(0, 6))

            self.en_cvvc_preset_menu = ctk.CTkOptionMenu(
                self.en_cvvc_row,
                values=["Core", "Core+", "All+Alt"],
                variable=self.en_cvvc_preset_var,
                width=110,
                command=lambda _v: self._save_config(),
            )
            _style_blue_menu(self.en_cvvc_preset_menu)
            self.en_cvvc_preset_menu.pack(side="left", padx=(0, 8))

            self.en_cvvc_list_fallback_checkbox = ctk.CTkCheckBox(
                self.en_cvvc_row,
                text=t("List-only 합성(실험)"),
                variable=self.en_cvvc_list_fallback_var,
                command=self._save_config,
                width=140,
            )
            self.en_cvvc_list_fallback_checkbox.pack(side="left", padx=(0, 8))

            self.en_cvvc_hint_label = ctk.CTkLabel(
                self.en_cvvc_row,
                text=t("(Preview: 기본 OTO + 옵션 시 list-only(vv/cc/alt) 합성)"),
                text_color=PALETTE.neutral_text,
            )
            self.en_cvvc_hint_label.pack(side="left", fill="x", expand=True)
        else:
            self.en_cvvc_row = None
            self.en_cvvc_pack_menu = None
            self.en_cvvc_beat_menu = None
            self.en_cvvc_preset_menu = None
            self.en_cvvc_list_fallback_checkbox = None
            self.en_cvvc_hint_label = None




        row_align = build_form_row(form_body)

        self.row_aligner = ctk.CTkFrame(row_align, fg_color="transparent")
        self.row_aligner.pack(side="left", fill="x", expand=True, padx=(0, 10))
        build_left_label(self.row_aligner, t("정렬 엔진:")).pack(side="left")
        self.aligner_menu = ctk.CTkOptionMenu(
            self.row_aligner,
            values=["WFL", HSMM_OTO_LABEL, "MFA (레거시)"],
            variable=self.aligner_var,
            width=190,
            command=self._on_aligner_change,
        )
        _style_blue_menu(self.aligner_menu)
        self.aligner_menu.pack(side="left", padx=(6, 8))
        self.aligner_help_label = ctk.CTkLabel(
            self.row_aligner,
            text=t("(기본은 WFL입니다. WFL/MFA는 선택한 경우에만 환경·모델 준비 여부를 확인하며, 준비되지 않으면 자동 폴백합니다.)"),
            text_color=PALETTE.neutral_text,
        )
        self.aligner_help_label.pack(side="left", fill="x", expand=True)
        self.row_no_mfa_oto_mode = build_form_row(form_body)
        build_left_label(self.row_no_mfa_oto_mode, t("No-MFA 생성:")).pack(side="left")
        self.no_mfa_oto_mode_menu = ctk.CTkOptionMenu(
            self.row_no_mfa_oto_mode,
            values=[
                NO_MFA_REMAP_LABEL,
            ],
            variable=self.no_mfa_oto_mode_var,
            width=280,
            command=lambda _v: self._on_no_mfa_oto_mode_change(),
        )
        _style_blue_menu(self.no_mfa_oto_mode_menu)
        self.no_mfa_oto_mode_menu.pack(side="left", padx=(6, 8))
        self.no_mfa_oto_mode_hint_label = ctk.CTkLabel(
            self.row_no_mfa_oto_mode,
            text=t("(베이스 OTO 필수, TextGrid 없이 음향 후보 점수로 보정)"),
            text_color=PALETTE.neutral_text,
        )
        self.no_mfa_oto_mode_hint_label.pack(side="left", fill="x", expand=True)
        # Boundary scorer checkpoint picker. Discovered model files appear as
        # additional options; "자동 (auto)" defers to the resolver's default
        # (mtime-newest non-experimental). Hidden together with the device row
        # under the same developer-mode + CRNN-engine gate (see _sync_aligner_ui).
        self.row_oto_crnn_scorer_model = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_scorer_model, t("Boundary scorer:")).pack(side="left")
        scorer_choices = (
            self._oto_crnn_model_choice_options()
            if hasattr(self, "_oto_crnn_model_choice_options")
            else ["자동 (auto)"]
        )
        if hasattr(self, "_set_oto_crnn_model_choice_from_code"):
            initial_code = ""
            if hasattr(self, "oto_crnn_model_choice_var"):
                try:
                    initial_code = self.oto_crnn_model_choice_var.get()
                except Exception:
                    initial_code = ""
            self._set_oto_crnn_model_choice_from_code(initial_code or "auto")
        self.oto_crnn_scorer_model_menu = ctk.CTkOptionMenu(
            self.row_oto_crnn_scorer_model,
            values=scorer_choices,
            variable=self.oto_crnn_model_choice_var,
            width=460,
            command=lambda _v: self._on_oto_crnn_model_choice_change(_v),
        )
        _style_blue_menu(self.oto_crnn_scorer_model_menu)
        self.oto_crnn_scorer_model_menu.pack(side="left", padx=(6, 8))
        self.oto_crnn_scorer_model_hint = ctk.CTkLabel(
            self.row_oto_crnn_scorer_model,
            text=t("(자동=기본 모델, 다른 .pt 선택 시 추론에 즉시 반영)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_crnn_scorer_model_hint.pack(side="left", fill="x", expand=True)
        self.row_oto_crnn_scorer_model.pack_forget()
        self.row_oto_crnn_engine = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_engine, t("CRNN Mode:")).pack(side="left")
        self.oto_crnn_engine_menu = ctk.CTkOptionMenu(
            self.row_oto_crnn_engine,
            values=["Stage1 heuristic only", "Boundary decoder + optional corrections"],
            variable=self.oto_crnn_engine_var,
            width=280,
            command=lambda _v: self._on_oto_crnn_engine_change(),
        )
        _style_blue_menu(self.oto_crnn_engine_menu)
        self.oto_crnn_engine_menu.pack(side="left", padx=(6, 8))
        self.oto_crnn_engine_hint = ctk.CTkLabel(
            self.row_oto_crnn_engine,
            text=t("(Stage1 = Boundary scorer + deterministic OTO heuristics only)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_crnn_engine_hint.pack(side="left", fill="x", expand=True)
        self.row_oto_crnn_engine.pack_forget()
        self.row_oto_stage2_model = build_form_row(form_body)
        self.oto_stage2_enable_checkbox = ctk.CTkCheckBox(
            self.row_oto_stage2_model,
            text=t("Stage2 OTO Assigner"),
            variable=self.oto_stage2_enable_var,
            command=lambda: self._on_oto_stage2_setting_change(),
            checkbox_width=18,
            checkbox_height=18,
        )
        self.oto_stage2_enable_checkbox.pack(side="left", padx=(0, 8))
        stage2_choices = (
            self._oto_stage2_model_choice_options()
            if hasattr(self, "_oto_stage2_model_choice_options")
            else ["자동 (auto)"]
        )
        self.oto_stage2_model_menu = ctk.CTkOptionMenu(
            self.row_oto_stage2_model,
            values=stage2_choices,
            variable=self.oto_stage2_model_choice_var,
            width=330,
            command=lambda _v: self._on_oto_stage2_model_choice_change(_v),
        )
        _style_blue_menu(self.oto_stage2_model_menu)
        self.oto_stage2_model_menu.pack(side="left", padx=(6, 8))
        self.oto_stage2_model_hint = ctk.CTkLabel(
            self.row_oto_stage2_model,
            text=t("(선택 시 Boundary Decoder 뒤에서 2단계 OTO anchor를 재지정)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_stage2_model_hint.pack(side="left", fill="x", expand=True)
        self.row_oto_stage2_model.pack_forget()
        self.row_oto_crnn_model = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_model, t("CRNN Device:")).pack(side="left")
        self.oto_crnn_device_menu = ctk.CTkOptionMenu(
            self.row_oto_crnn_model,
            values=["auto", "cuda", "cpu"],
            variable=self.oto_crnn_device_var,
            width=80,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.oto_crnn_device_menu)
        self.oto_crnn_device_menu.pack(side="left", padx=(6, 8))
        self.oto_crnn_model_hint_label = ctk.CTkLabel(
            self.row_oto_crnn_model,
            text=t("(추론 디바이스. auto = GPU 가용 시 GPU 사용)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_crnn_model_hint_label.pack(side="left", fill="x", expand=True)
        self.row_oto_crnn_model.pack_forget()
        self.row_oto_crnn_special_aliases = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_special_aliases, t("특수 에일리어스:")).pack(side="left")
        self.oto_crnn_special_aliases_entry = ctk.CTkEntry(
            self.row_oto_crnn_special_aliases,
            textvariable=self.oto_crnn_special_aliases_var,
            placeholder_text="쉼표로 구분 (예: Sp, br, cl)",
        )
        self.oto_crnn_special_aliases_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.oto_crnn_special_aliases_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.row_oto_crnn_special_aliases.pack_forget()
        self.row_align_extra = build_form_row(form_body)
        build_left_label(self.row_align_extra, t("MFA 정렬 프로필:")).pack(side="left")
        self.mfa_align_profile_menu = ctk.CTkOptionMenu(
            self.row_align_extra,
            values=["기본", "정밀", "정밀 + 화자 적응", "빠름"],
            variable=self.mfa_align_profile_var,
            width=220,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.mfa_align_profile_menu)
        self.mfa_align_profile_menu.pack(side="left", padx=(6, 8))
        ctk.CTkLabel(
            self.row_align_extra,
            text=t("(기본=정확도 균형)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)





        self.row_aligner_advanced = build_form_row(form_body)
        ctk.CTkLabel(
            self.row_aligner_advanced,
            text=t("정렬 고급 옵션은 자동으로 조정됩니다."),
            text_color=PALETTE.neutral_text,
            anchor="w",
        ).pack(side="left", padx=(121, 0))

        advanced_row = build_form_row(path_frame)
        self.advanced_toggle_btn = ctk.CTkButton(
            advanced_row,
            text=t("▶ 추가 옵션 (특수 발음)"),
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
            text=t("필요한 경우에만 사용하세요."),
            text_color=PALETTE.hint_text,
            anchor="e",
        )
        self.advanced_hint_label.pack(fill="x", padx=12, pady=(0, 6))

        self.advanced_options_frame = ctk.CTkFrame(path_frame, fg_color="transparent")

        row0 = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0, text=t("특수 발음 (선택):"), width=120, anchor="w").pack(side="left")
        self.custom_entry = ctk.CTkEntry(row0, placeholder_text=t("커스텀 매핑 규칙 파일 (.txt)"), textvariable=self.custom_phoneme_var)
        self.custom_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        custom_browse_btn = ctk.CTkButton(row0, text=t("찾아보기"), width=90, command=lambda: self._browse_file(self.custom_entry, [("Text 파일", "*.txt")]))
        _style_primary_button(custom_browse_btn)
        custom_browse_btn.pack(side="right")

        self._toggle_advanced_options(force=False)

        self.pipeline_action_host = ctk.CTkFrame(
            left_scroll,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
            corner_radius=8,
        )
        self.pipeline_action_host.pack(side="top", fill="x", padx=0, pady=(8, 0))

        tab_header = ctk.CTkFrame(right_panel, fg_color=PALETTE.panel_bg)
        tab_header.pack(fill="x", padx=0, pady=(0, 4))
        self.dev_mode_btn = ctk.CTkButton(
            tab_header,
            text=t("고급 보기"),
            width=110,
            height=24,
            corner_radius=6,
            font=("", 11),
            fg_color="#C2CFDF",
            hover_color="#B1C0D4",
            text_color="#2A3A50",
            command=self._toggle_developer_mode,
        )
        self.dev_mode_btn.pack(side="left", padx=(2, 0))

        self.theme_profile_menu = ctk.CTkOptionMenu(
            tab_header,
            values=list(THEME_PROFILE_OPTIONS),
            variable=self.ui_theme_var,
            width=120,
            command=self._on_theme_profile_change,
        )
        _style_blue_menu(self.theme_profile_menu)
        self.theme_profile_menu.pack(side="left", padx=(8, 0))

        self.tabview = ctk.CTkTabview(
            right_panel,
            fg_color=PALETTE.panel_bg,
            segmented_button_fg_color=PALETTE.advanced_toggle_bg,
            segmented_button_selected_color=PALETTE.primary_button_bg,
            segmented_button_selected_hover_color=PALETTE.primary_button_hover,
            segmented_button_unselected_color=PALETTE.advanced_toggle_bg,
            segmented_button_unselected_hover_color=PALETTE.advanced_toggle_hover,
            text_color=PALETTE.neutral_text,
            command=self._on_tabview_change,
        )
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        self._lazy_tab_builders = {
            "파이프라인": self._build_pipeline_tab,
            "고급 설정": self._build_advanced_settings_tab,
            "파라미터 조정": self._build_params_tab,
            "로그": self._build_log_tab,
            "상세 로그": self._build_detail_log_tab,
            "크레딧": self._build_credits_tab,
        }
        self._built_tabs = set()
        for tab_name in TAB_ORDER:
            if tab_name not in self._lazy_tab_builders or not self._tab_allowed_by_mode(tab_name):
                continue
            try:
                self._get_or_add_tab(tab_name)
            except Exception:
                pass
        for tab_name in ("파이프라인",):
            self._ensure_tab_built(tab_name)
        self.tabview.set("파이프라인")
        self._on_tabview_change()
        self._sync_developer_mode_ui()
        self._schedule_lazy_tab_prewarm()

        bottom = ctk.CTkFrame(self, fg_color=PALETTE.panel_bg)
        bottom.pack(side="bottom", fill="x", padx=15, pady=(5, 15))
        self.bottom_bar_frame = bottom

        status_group = ctk.CTkFrame(bottom, fg_color="transparent")
        status_group.pack(side="left", fill="x", expand=True, padx=(10, 6))
        self.bottom_status_group = status_group

        self.status_label = ctk.CTkLabel(status_group, text=t("대기 중"), anchor="w", text_color=PALETTE.neutral_text)
        self.status_label.pack(fill="x")

        progress_row = ctk.CTkFrame(status_group, fg_color="transparent")
        progress_row.pack(fill="x", pady=(4, 0))
        self.progress_bar = ctk.CTkProgressBar(progress_row, height=12)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_label = ctk.CTkLabel(progress_row, text="0%", width=42, anchor="e", text_color=PALETTE.neutral_text)
        self.progress_label.pack(side="left", padx=(8, 0))

        actions_group = ctk.CTkFrame(bottom, fg_color="transparent")
        actions_group.pack(side="right", padx=(6, 0))
        self.bottom_actions_group = actions_group

        self.run_btn = ctk.CTkButton(
            actions_group,
            text=t("▶ 전체 실행"),
            font=("", 14, "bold"),
            width=150,
            height=40,
            command=self._run_full_pipeline,
        )
        self.run_btn.pack(side="right", padx=10)

        self.report_btn = ctk.CTkButton(
            actions_group,
            text=t("🐛 제보 리포트 복사"),
            width=120,
            height=40,
            fg_color=PALETTE.danger_button_bg,
            hover_color=PALETTE.danger_button_hover,
            command=self._copy_quick_error_report,
        )
        self.report_btn.pack(side="right", padx=5)
        if hasattr(self, "_update_responsive_bottom_layout"):
            try:
                self._update_responsive_bottom_layout()
            except Exception:
                pass
        self._sync_aligner_ui()
        self._sync_ja_alias_controls()
        self._sync_en_cvvc_controls()
        self._sync_base_oto_requirement_ui()
        self._sync_cmpx_route_lock()

    def _on_theme_profile_change(self, _value=None):
        selected = str(
            _value
            if _value is not None
            else (self.ui_theme_var.get() if hasattr(self, "ui_theme_var") else "")
        ).strip()
        self._apply_ui_theme_profile(selected, persist=True, rebuild=True)

    def _apply_ui_theme_profile(self, profile_name, *, persist=True, rebuild=False):
        selected = apply_theme_profile(profile_name)
        if hasattr(self, "ui_theme_var"):
            self.ui_theme_var.set(selected)
        try:
            ctk.set_appearance_mode(get_theme_appearance_mode(selected))
        except Exception:
            pass
        if persist:
            self._save_config()
        if rebuild:
            self._rebuild_ui_for_theme_change()

    def _snapshot_ui_state_for_theme_change(self):
        def _safe_var_get(name, default=""):
            var = getattr(self, name, None)
            if var is None:
                return default
            try:
                return var.get()
            except Exception:
                return default

        def _safe_entry_get(name, default=""):
            widget = getattr(self, name, None)
            if widget is None:
                return default
            try:
                return widget.get()
            except Exception:
                return default

        return {
            "wav_dir": _safe_entry_get("wav_entry", ""),
            "tpl_path": _safe_entry_get("tpl_entry", ""),
            "out_path": _safe_entry_get("out_entry", ""),
            "lang": _safe_var_get("lang_var", "Korean (한국어)"),
            "format": _safe_var_get("auto_format_var", "자동 감지 (권장)"),
        }

    def _restore_ui_state_after_theme_change(self, state):
        if not isinstance(state, dict):
            return
        try:
            if hasattr(self, "wav_entry"):
                self.wav_entry.delete(0, "end")
                self.wav_entry.insert(0, str(state.get("wav_dir", "") or ""))
            if hasattr(self, "tpl_entry"):
                self.tpl_entry.delete(0, "end")
                self.tpl_entry.insert(0, str(state.get("tpl_path", "") or ""))
            if hasattr(self, "out_entry"):
                self.out_entry.delete(0, "end")
                self.out_entry.insert(0, str(state.get("out_path", "") or ""))
        except Exception:
            pass

        if hasattr(self, "lang_var"):
            try:
                self.lang_var.set(str(state.get("lang", self.lang_var.get()) or self.lang_var.get()))
            except Exception:
                pass
        if hasattr(self, "auto_format_var"):
            try:
                self.auto_format_var.set(str(state.get("format", self.auto_format_var.get()) or self.auto_format_var.get()))
            except Exception:
                pass

        if hasattr(self, "_sync_language_notice"):
            try:
                self._sync_language_notice()
            except Exception:
                pass
        if hasattr(self, "_sync_ja_alias_controls"):
            self._sync_ja_alias_controls()
        if hasattr(self, "_sync_en_cvvc_controls"):
            self._sync_en_cvvc_controls()
        if hasattr(self, "_sync_base_oto_requirement_ui"):
            self._sync_base_oto_requirement_ui()
        if hasattr(self, "_sync_cmpx_route_lock"):
            self._sync_cmpx_route_lock()
        if hasattr(self, "_on_language_change"):
            try:
                self._on_language_change(self.lang_var.get())
            except Exception:
                pass

    def _on_tabview_change(self):
        current = ""
        if hasattr(self, "tabview"):
            try:
                current = str(self.tabview.get() or "").strip()
            except Exception:
                current = ""
        if current:
            self._request_tab_build(current)

    def _developer_mode_enabled(self) -> bool:
        if not hasattr(self, "developer_mode_enabled_var"):
            return False
        try:
            return bool(self.developer_mode_enabled_var.get())
        except Exception:
            return False

    def _tab_allowed_by_mode(self, tab_name: str) -> bool:
        name = str(tab_name or "").strip()
        return name not in DEVELOPER_ONLY_TABS or self._developer_mode_enabled()

    def _sync_secondary_tab_visibility(self, enabled: bool) -> None:
        tabview = getattr(self, "tabview", None)
        if tabview is None:
            return
        try:
            visible = set(getattr(tabview, "_tab_dict", {}).keys())
        except Exception:
            return

        if enabled:
            for tab_name in TAB_ORDER:
                if tab_name not in DEVELOPER_ONLY_TABS or tab_name in visible:
                    continue
                target_index = sum(1 for prior in TAB_ORDER[: TAB_ORDER.index(tab_name)] if prior in visible)
                try:
                    tabview.insert(target_index, tab_name)
                    visible.add(tab_name)
                except Exception:
                    continue
            return

        try:
            current = str(tabview.get() or "").strip()
        except Exception:
            current = ""
        if current in DEVELOPER_ONLY_TABS:
            try:
                tabview.set("파이프라인")
            except Exception:
                pass

        built_tabs = getattr(self, "_built_tabs", None)
        pending = getattr(self, "_tab_build_pending", None)
        queue = list(getattr(self, "_lazy_prewarm_queue", []) or [])
        for tab_name in DEVELOPER_ONLY_TABS:
            if tab_name not in visible:
                continue
            try:
                tabview.delete(tab_name)
            except Exception:
                continue
            if isinstance(built_tabs, set):
                built_tabs.discard(tab_name)
            if isinstance(pending, set):
                pending.discard(tab_name)
            queue = [name for name in queue if name != tab_name]
            if tab_name == "상세 로그":
                self.detail_log_text = None
        self._lazy_prewarm_queue = queue

    def _request_tab_build(self, tab_name):
        name = str(tab_name or "").strip()
        if not name or not self._tab_allowed_by_mode(name):
            return
        if name in (getattr(self, "_built_tabs", set()) or set()):
            return
        pending = getattr(self, "_tab_build_pending", None)
        if not isinstance(pending, set):
            pending = set()
            self._tab_build_pending = pending
        if name in pending:
            return
        pending.add(name)

        def _build_once():
            try:
                self._ensure_tab_built(name)
            finally:
                try:
                    pending.discard(name)
                except Exception:
                    pass

        try:
            self.after_idle(_build_once)
        except Exception:
            _build_once()

    def _ensure_tab_built(self, tab_name):
        name = str(tab_name or "").strip()
        if not name or not self._tab_allowed_by_mode(name):
            return
        built_tabs = getattr(self, "_built_tabs", None)
        if not isinstance(built_tabs, set):
            built_tabs = set()
            self._built_tabs = built_tabs
        if name in built_tabs:
            return
        builders = getattr(self, "_lazy_tab_builders", None) or {}
        builder = builders.get(name)
        if builder is None:
            return
        builder()
        built_tabs.add(name)

        if name == "파이프라인" and hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()
        if name == "고급 설정":
            if hasattr(self, "_sync_developer_mode_ui"):
                self._sync_developer_mode_ui()
            if hasattr(self, "_sync_consistency_toggle_label"):
                self._sync_consistency_toggle_label()
            if hasattr(self, "_sync_vc_correction_toggle"):
                self._sync_vc_correction_toggle()
            if hasattr(self, "_sync_cvn_correction_toggle"):
                self._sync_cvn_correction_toggle()
            if hasattr(self, "_sync_ml_correction_ui"):
                self._sync_ml_correction_ui()
            if hasattr(self, "_sync_mapping_supervised_ui"):
                self._sync_mapping_supervised_ui()
            if hasattr(self, "_sync_ml_e2e_controls"):
                self._sync_ml_e2e_controls()
            if hasattr(self, "_sync_advanced_tuning_slider_controls"):
                self._sync_advanced_tuning_slider_controls()
            if hasattr(self, "_refresh_ml_backend_status"):
                self._refresh_ml_backend_status()

    def _schedule_lazy_tab_prewarm(self):
        builders = getattr(self, "_lazy_tab_builders", None) or {}
        built_tabs = getattr(self, "_built_tabs", None) or set()
        preferred_order = ("로그", "고급 설정", "파라미터 조정", "상세 로그", "크레딧", "파이프라인")
        queue = [
            name
            for name in preferred_order
            if name in builders and name not in built_tabs and self._tab_allowed_by_mode(name)
        ]
        for name in builders.keys():
            if name not in built_tabs and name not in queue and self._tab_allowed_by_mode(name):
                queue.append(name)
        self._lazy_prewarm_queue = queue
        if not queue:
            return

        def _kickoff():
            self._lazy_prewarm_job = None
            self._run_lazy_tab_prewarm_step()

        try:
            self._lazy_prewarm_job = self.after(180, _kickoff)
        except Exception:
            _kickoff()

    def _run_lazy_tab_prewarm_step(self):
        queue = list(getattr(self, "_lazy_prewarm_queue", []) or [])
        if not queue:
            self._lazy_prewarm_job = None
            return
        name = queue.pop(0)
        self._lazy_prewarm_queue = queue
        self._request_tab_build(name)
        if not queue:
            self._lazy_prewarm_job = None
            return
        try:
            self._lazy_prewarm_job = self.after(120, self._run_lazy_tab_prewarm_step)
        except Exception:
            self._run_lazy_tab_prewarm_step()

    def _rebuild_ui_for_theme_change(self):
        if bool(getattr(self, "is_running", False)):
            if hasattr(self, "_append_log"):
                self._append_log("[UI] 작업 중에는 테마를 변경할 수 없습니다.")
            return
        if bool(getattr(self, "_theme_rebuild_pending", False)):
            return
        self._theme_rebuild_pending = True
        self._suppress_ui_callback_errors = True

        snapshot = self._snapshot_ui_state_for_theme_change()

        def _do_rebuild():
            try:
                for child in list(self.winfo_children()):
                    try:
                        child.destroy()
                    except Exception:
                        pass
                self._build_ui()
                # Theme switch path restores only required state instead of full config
                # reload to avoid textvariable callback races and speed up redraw.
                self._restore_ui_state_after_theme_change(snapshot)
            finally:
                self._theme_rebuild_pending = False
                try:
                    self.after(600, lambda: setattr(self, "_suppress_ui_callback_errors", False))
                except Exception:
                    self._suppress_ui_callback_errors = False

        try:
            self.after_idle(_do_rebuild)
        except Exception:
            _do_rebuild()

    def _is_preview_channel(self):
        return str(getattr(self, "release_channel", "stable")).strip().lower() == "preview"

    def _sync_ja_alias_controls(self):
        row = getattr(self, "ja_alias_row", None)
        if row is None:
            return
        is_japanese = self._get_language() == "japanese"
        try:
            if is_japanese:
                if not row.winfo_ismapped():
                    row.pack(fill="x", pady=4)
            else:
                row.pack_forget()
        except Exception:
            pass
        menu = getattr(self, "ja_alias_style_menu", None)
        if menu is not None:
            try:
                menu.configure(state="normal" if is_japanese else "disabled")
            except Exception:
                pass

    def _sync_en_cvvc_controls(self):
        row = getattr(self, "en_cvvc_row", None)
        if row is None:
            return
        show = EN_CVVC_UI_ENABLED and self._is_preview_channel() and self._get_language() == "english"
        is_english = self._get_language() == "english"
        try:
            if show:
                if not row.winfo_ismapped():
                    row.pack(fill="x", pady=4)
            else:
                row.pack_forget()
        except Exception:
            pass
        state = "normal" if is_english else "disabled"
        for widget_name in (
            "en_cvvc_pack_menu",
            "en_cvvc_beat_menu",
            "en_cvvc_preset_menu",
            "en_cvvc_list_fallback_checkbox",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except Exception:
                pass
        hint = getattr(self, "en_cvvc_hint_label", None)
        if hint is not None:
            try:
                if not show:
                    hint.configure(text=t("(영어 선택 시 표시되는 Preview 전용 설정)"))
                else:
                    hint.configure(text=t("(Preview: 기본 OTO + 옵션 시 list-only(vv/cc/alt) 합성)"))
            except Exception:
                pass

    def _get_language(self):
        """UI 언어 선택값을 내부 코드('korean'/'japanese'/'english')로 정규화합니다."""
        sel = str(self.lang_var.get() or "")
        low = sel.strip().lower()
        if "english" in low or "영어" in sel or low in {"en", "eng"}:
            return "english" if self._is_preview_channel() else "korean"
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
            values = [
                "자동 감지 (권장)",
                "CV/연단음",
                "CVC (한국어 전용)",
                "COC (한국어 CVC 파생형)",
                "CVVC",
                "VCV (연속음)",
            ]
            if self._is_preview_channel():
                values.append("C+V (템플릿 전용)")
                values.append("CMPX (프리뷰)")
            return values
        if lang == "english":
            return ["CVVC"]
        return ["자동 감지 (권장)", "CV/연단음", "CVVC", "VCV (연속음)"]

    def _set_auto_format_from_code(self, format_code, language=None):
        lang = language or self._get_language()
        values = self._get_auto_format_options(lang)
        if hasattr(self, "format_dropdown"):
            self.format_dropdown.configure(values=values)
        if lang == "english":
            self.auto_format_var.set("CVVC")
            return
        label_map = {
            "": "자동 감지 (권장)",
            "cv": "CV/연단음",
            "c_plus_v": "C+V (템플릿 전용)",
            "cvc": "CVC (한국어 전용)",
            "coc": "COC (한국어 파생형)",
            "cvvc": "CVVC",
            "vcv": "VCV (연속음)",
            "cmpx": "CMPX (프리뷰)",
        }
        label = label_map.get(str(format_code or "").strip().lower(), "자동 감지 (권장)")
        if label not in values:
            if label in {"CVC (한국어 전용)", "COC (한국어 파생형)", "C+V (템플릿 전용)"} and lang != "korean":
                label = "CV/연단음"
            elif label == "C+V (템플릿 전용)" and not self._is_preview_channel():
                label = "CV/연단음"
            else:
                label = "자동 감지 (권장)"
        self.auto_format_var.set(label)

    def _on_ui_language_change(self, value: str) -> None:
        from ui.i18n import set_language as _set_lang
        normalized = str(value or "").strip()
        lowered = normalized.lower()
        if lowered in {"english", "en"}:
            lang_code = "en"
        elif lowered in {"japanese", "ja"} or ("日" in normalized):
            lang_code = "ja"
        else:
            lang_code = "ko"
        _set_lang(lang_code)
        if hasattr(self, "_save_config"):
            self._save_config()
        if hasattr(self, "_ui_lang_hint_label"):
            from ui.i18n import t as _t
            self._ui_lang_hint_label.configure(text=_t("재시작 후 적용됩니다."))

    def _on_language_change(self, value):
        lang = self._get_language()
        if lang == "korean":
            self.lang_info_label.configure(text=t("한국어 단위(a, k, ga 등) 에일리어스를 기준으로 생성합니다."))
            self.lang_notice_label.configure(
                text=(
                    "현재 언어: 한국어\n"
                    "Lab 생성, 사전 생성, 정렬, OTO 계산이 모두 한국어 규칙으로 진행됩니다.\n"
                    "기본 인코딩: UTF-8"
                ),
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
        elif lang == "japanese":
            self.lang_info_label.configure(text=t("일본어 단위(a, k, ka 등) 에일리어스를 기준으로 생성합니다."))
            self.lang_notice_label.configure(
                text=(
                    "현재 언어: 일본어\n"
                    "특수 발음 기호가 섞인 파일은 Lab 생성 전에 언어 선택을 다시 확인하세요.\n"
                    "기본 인코딩: UTF-8"
                ),
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
        else:
            self.lang_info_label.configure(text=t("영어 CVVC base OTO를 불러와 생성합니다. (프리뷰 전용)"))
            self.lang_notice_label.configure(
                text=(
                    "현재 언어: 영어(프리뷰)\n"
                    "Lab/사전/MFA를 건너뛰고, 선택한 EN CVVC 리스트의 base OTO를 조합해 생성합니다.\n"
                    "list-only(vv/cc/alt) 섹션은 옵션으로 합성 생성할 수 있습니다."
                ),
                fg_color=LANGUAGE_NOTICE_THEME["english"]["fg_color"],
                text_color=LANGUAGE_NOTICE_THEME["english"]["text_color"],
            )
            try:
                self.lang_dropdown.configure(
                    fg_color=LANGUAGE_DROPDOWN_THEME["english"]["fg_color"],
                    button_color=LANGUAGE_DROPDOWN_THEME["english"]["button_color"],
                    button_hover_color=LANGUAGE_DROPDOWN_THEME["english"]["button_hover_color"],
                )
            except Exception:
                pass
            self.gen_missing_vowels_checkbox.configure(state="disabled")
            if hasattr(self, "ja_alias_style_menu"):
                self.ja_alias_style_menu.configure(state="disabled")
            if hasattr(self, "aligner_var"):
                self.aligner_var.set(HSMM_OTO_LABEL)
        current_code = normalize_auto_format_value(self._get_language(), self.auto_format_var.get())
        self._set_auto_format_from_code(current_code, self._get_language())
        if hasattr(self, "_apply_recommended_ml_model_defaults"):
            self._apply_recommended_ml_model_defaults()
        if hasattr(self, "_sync_consistency_toggle_label"):
            self._sync_consistency_toggle_label()
        self._sync_ja_alias_controls()
        self._sync_en_cvvc_controls()
        self._sync_base_oto_requirement_ui()
        self._sync_cmpx_route_lock()
        self._sync_aligner_ui()
        self._save_config()
        self._refresh_ml_backend_status()

    def _sync_consistency_toggle_label(self):
        checkbox = getattr(self, "kr_continuity_enable_checkbox", None)
        if checkbox is None:
            return
        lang = self._get_language() if hasattr(self, "_get_language") else "korean"
        if lang == "japanese":
            text = t("일관성 보정 사용")
        elif lang == "korean":
            text = t("연속성 보정 사용")
        else:
            text = t("연속성/파일 일관성 보정 사용")
        try:
            checkbox.configure(text=text)
        except Exception:
            pass
    def _requires_base_oto_for_current_mode(self):
        lang = self._get_language()
        if lang == "english":
            return True
        if lang == "korean":
            fmt = normalize_auto_format_value(lang, self.auto_format_var.get()) if hasattr(self, "auto_format_var") else ""
            if fmt in {"cmpx", "c_plus_v"}:
                return True
        return False

    def _sync_base_oto_requirement_ui(self):
        requires_base = self._requires_base_oto_for_current_mode()
        if requires_base and hasattr(self, "no_base_oto_var"):
            self.no_base_oto_var.set(False)
        if hasattr(self, "no_base_oto_checkbox"):
            label = "템플릿 OTO 없음 (OpenUtau 호환 에일리어스 자동 생성)"
            if requires_base:
                label = "템플릿 OTO 필수 (현재 모드)"
            try:
                self.no_base_oto_checkbox.configure(
                    state="disabled" if requires_base else "normal",
                    text=label,
                )
            except Exception:
                pass
        if hasattr(self, "tpl_entry"):
            try:
                self.tpl_entry.configure(state="normal")
            except Exception:
                pass
        if hasattr(self, "tpl_browse_btn"):
            try:
                self.tpl_browse_btn.configure(state="normal")
            except Exception:
                pass
        if not requires_base:
            no_base = bool(self.no_base_oto_var.get()) if hasattr(self, "no_base_oto_var") else False
            if hasattr(self, "tpl_entry"):
                try:
                    self.tpl_entry.configure(state="disabled" if no_base else "normal")
                except Exception:
                    pass
            if hasattr(self, "tpl_browse_btn"):
                try:
                    self.tpl_browse_btn.configure(state="disabled" if no_base else "normal")
                except Exception:
                    pass

    def _sync_cmpx_route_lock(self):
        is_kr_template_only = False
        try:
            lang = self._get_language()
            fmt = normalize_auto_format_value(lang, self.auto_format_var.get()) if hasattr(self, "auto_format_var") else ""
            if lang == "korean" and fmt == "c_plus_v" and not self._is_preview_channel():
                try:
                    self.auto_format_var.set("CV/연단음")
                    fmt = "cv"
                except Exception:
                    fmt = "cv"
            is_kr_template_only = (lang == "korean" and fmt in {"cmpx", "c_plus_v"})
        except Exception:
            is_kr_template_only = False
        if is_kr_template_only and hasattr(self, "ml_route_var"):
            try:
                if hasattr(self, "_set_ml_route_from_code"):
                    self._set_ml_route_from_code("nomfa")
                else:
                    self.ml_route_var.set("No-MFA")
            except Exception:
                pass
        if hasattr(self, "ml_route_menu"):
            try:
                self.ml_route_menu.configure(state="disabled" if is_kr_template_only else "normal")
            except Exception:
                pass
    def _get_ja_alias_style_code(self):
        style = self.ja_alias_style_var.get().strip()
        if style == "히라가나":
            return "hiragana"
        if style == "로마자":
            return "romaji"
        return "original"

    @staticmethod
    def _normalize_no_mfa_oto_mode_code(value):
        raw = str(value or "").strip().lower()
        if raw in {"crnn", "oto_crnn", "oto-crnn", "crnn_oto", "crnn-oto"}:
            return "remap"
        if raw in {"remap", "base_remap", "base"}:
            return "remap"
        text = str(value or "").strip()
        if text == "CRNN OTO 예측기(실험)":
            return "remap"
        if text in {"베이스 OTO 재매핑 + 보정", NO_MFA_REMAP_LABEL}:
            return "remap"
        return "remap"

    @staticmethod
    def _normalize_oto_crnn_engine_code(value):
        raw = str(value or "").strip().lower()
        if raw in {
            "stage1",
            "stage1_heuristic",
            "stage1-heuristic",
            "stage1 heuristic only",
            "heuristic",
            "heuristic_only",
            "boundary heuristic",
        }:
            return "stage1_heuristic"
        # Legacy direct CRNN path is removed from UI routing.
        return "boundary_decoder"

    def _set_oto_crnn_engine_from_code(self, code):
        normalized = self._normalize_oto_crnn_engine_code(code)
        label = (
            "Stage1 heuristic only"
            if normalized == "stage1_heuristic"
            else "Boundary decoder + optional corrections"
        )
        if hasattr(self, "oto_crnn_engine_var"):
            try:
                self.oto_crnn_engine_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_oto_crnn_engine_code(self):
        if not hasattr(self, "oto_crnn_engine_var"):
            return "boundary_decoder"
        try:
            current = self.oto_crnn_engine_var.get()
        except Exception:
            current = ""
        return self._set_oto_crnn_engine_from_code(current)

    def _on_oto_crnn_engine_change(self):
        self._set_oto_crnn_engine_from_code(
            self.oto_crnn_engine_var.get() if hasattr(self, "oto_crnn_engine_var") else ""
        )
        self._save_config()

    # --- Boundary scorer model choice --------------------------------
    # The dropdown label is "자동 (auto)" or a friendly label produced by
    # `list_available_boundary_scorer_models`. The persisted code is either
    # "auto" or the model basename. The resolver in
    # `core.coarse_crnn.oto_predictor_generator` accepts a basename and
    # searches the standard roots, so we never store an absolute path.

    _OTO_CRNN_MODEL_CHOICE_AUTO_LABEL = "자동 (auto)"

    def _list_oto_crnn_model_choices(self) -> list[dict[str, object]]:
        try:
            from core.coarse_crnn.oto_predictor_generator import (
                list_available_boundary_scorer_models,
            )
        except Exception:
            return []
        try:
            return list_available_boundary_scorer_models()
        except Exception:
            return []

    def _oto_crnn_model_choice_options(self) -> list[str]:
        labels = [self._OTO_CRNN_MODEL_CHOICE_AUTO_LABEL]
        for item in self._list_oto_crnn_model_choices():
            label = str(item.get("label") or item.get("name") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _oto_crnn_label_to_code(self, label: object) -> str:
        text = str(label or "").strip()
        if not text or text == self._OTO_CRNN_MODEL_CHOICE_AUTO_LABEL or text.lower() == "auto":
            return "auto"
        for item in self._list_oto_crnn_model_choices():
            if str(item.get("label") or "") == text:
                return str(item.get("name") or "auto")
            if str(item.get("name") or "") == text:
                return str(item.get("name") or "auto")
        # Unknown label (e.g. stale config) → fall back to auto-resolve.
        return "auto"

    def _oto_crnn_code_to_label(self, code: object) -> str:
        text = str(code or "").strip()
        if not text or text.lower() == "auto":
            return self._OTO_CRNN_MODEL_CHOICE_AUTO_LABEL
        for item in self._list_oto_crnn_model_choices():
            if str(item.get("name") or "") == text:
                return str(item.get("label") or item.get("name") or text)
        return self._OTO_CRNN_MODEL_CHOICE_AUTO_LABEL

    def _set_oto_crnn_model_choice_from_code(self, code) -> str:
        label = self._oto_crnn_code_to_label(code)
        if hasattr(self, "oto_crnn_model_choice_var"):
            try:
                self.oto_crnn_model_choice_var.set(label)
            except Exception:
                pass
        return self._oto_crnn_label_to_code(label)

    def _get_oto_crnn_model_choice_code(self) -> str:
        if not hasattr(self, "oto_crnn_model_choice_var"):
            return "auto"
        try:
            current = self.oto_crnn_model_choice_var.get()
        except Exception:
            current = ""
        return self._oto_crnn_label_to_code(current)

    def _on_oto_crnn_model_choice_change(self, _value=None):
        code = self._get_oto_crnn_model_choice_code()
        if hasattr(self, "_append_log"):
            try:
                self._append_log(f"[CRNN-OTO] boundary scorer model = {code}")
            except Exception:
                pass
        if hasattr(self, "_save_config"):
            self._save_config()

    _OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL = "자동 (auto)"

    def _list_oto_stage2_model_choices(self) -> list[dict[str, object]]:
        try:
            from core.coarse_crnn.oto_predictor_generator import (
                list_available_stage2_oto_models,
            )
        except Exception:
            return []
        try:
            return list_available_stage2_oto_models()
        except Exception:
            return []

    def _oto_stage2_model_choice_options(self) -> list[str]:
        labels = [self._OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL]
        for item in self._list_oto_stage2_model_choices():
            label = str(item.get("label") or item.get("name") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _oto_stage2_label_to_code(self, label: object) -> str:
        text = str(label or "").strip()
        if not text or text == self._OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL or text.lower() == "auto":
            return "auto"
        for item in self._list_oto_stage2_model_choices():
            if str(item.get("label") or "") == text:
                return str(item.get("name") or "auto")
            if str(item.get("name") or "") == text:
                return str(item.get("name") or "auto")
        return "auto"

    def _oto_stage2_code_to_label(self, code: object) -> str:
        text = str(code or "").strip()
        if not text or text.lower() == "auto":
            return self._OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL
        for item in self._list_oto_stage2_model_choices():
            if str(item.get("name") or "") == text:
                return str(item.get("label") or item.get("name") or text)
        return self._OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL

    def _set_oto_stage2_model_choice_from_code(self, code) -> str:
        label = self._oto_stage2_code_to_label(code)
        if hasattr(self, "oto_stage2_model_choice_var"):
            try:
                self.oto_stage2_model_choice_var.set(label)
            except Exception:
                pass
        return self._oto_stage2_label_to_code(label)

    def _get_oto_stage2_model_choice_code(self) -> str:
        if not hasattr(self, "oto_stage2_model_choice_var"):
            return "auto"
        try:
            current = self.oto_stage2_model_choice_var.get()
        except Exception:
            current = ""
        return self._oto_stage2_label_to_code(current)

    def _on_oto_stage2_model_choice_change(self, _value=None):
        code = self._get_oto_stage2_model_choice_code()
        if hasattr(self, "_append_log"):
            try:
                self._append_log(f"[CRNN-OTO] stage2 model = {code}")
            except Exception:
                pass
        if hasattr(self, "_save_config"):
            self._save_config()

    def _on_oto_stage2_setting_change(self):
        if hasattr(self, "_append_log"):
            try:
                enabled = bool(self.oto_stage2_enable_var.get()) if hasattr(self, "oto_stage2_enable_var") else False
                self._append_log(f"[CRNN-OTO] Stage2 OTO Assigner {'ON' if enabled else 'OFF'}")
            except Exception:
                pass
        if hasattr(self, "_save_config"):
            self._save_config()

    def _get_selected_phoneme_boundary_model_path(self) -> str:
        models = getattr(self, "_phoneme_boundary_models", []) or []
        try:
            selected = str(self.phoneme_boundary_model_var.get() if hasattr(self, "phoneme_boundary_model_var") else "").strip()
        except Exception:
            selected = ""
        for item in models:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "").strip()
            name = str(item.get("name") or "").strip()
            if selected and selected in {label, name}:
                path = str(item.get("path") or "").strip()
                return path if os.path.isfile(path) else ""
        return ""

    def _set_no_mfa_oto_mode_from_code(self, code):
        normalized = self._normalize_no_mfa_oto_mode_code(code)
        label = NO_MFA_REMAP_LABEL
        if hasattr(self, "no_mfa_oto_mode_var"):
            try:
                self.no_mfa_oto_mode_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_no_mfa_oto_mode_code(self):
        if not hasattr(self, "no_mfa_oto_mode_var"):
            return "remap"
        try:
            current = self.no_mfa_oto_mode_var.get()
        except Exception:
            current = ""
        normalized = self._set_no_mfa_oto_mode_from_code(current)
        return normalized

    def _on_no_mfa_oto_mode_change(self):
        self._set_no_mfa_oto_mode_from_code(
            self.no_mfa_oto_mode_var.get() if hasattr(self, "no_mfa_oto_mode_var") else ""
        )
        self._sync_aligner_ui()
        self._save_config()

    def _get_mfa_align_profile_code(self):
        profile = str(self.mfa_align_profile_var.get() if hasattr(self, "mfa_align_profile_var") else "").strip()
        if profile in {"빠름", "빠름 (저사양 추천)", "fast"}:
            return "fast"
        if profile in {
            "정밀 + 화자 적응",
            "정확도 우선 + 화자 적응",
            "accurate_adapted",
            "speaker_adapted",
            "speaker_adaptation",
        }:
            return "accurate_adapted"
        if profile in {"정밀", "정확도 우선", "정확도 우선 (정밀)", "accurate"}:
            return "accurate"
        if profile in {"기본", "default", "정확도 우선 (기본)"}:
            return "default"
        return "default"

    def _on_aligner_change(self, _value=None):
        self._sync_aligner_ui()
        try:
            selected = normalize_aligner_name(
                self.aligner_var.get() if hasattr(self, "aligner_var") else "",
                default="hsmm_oto",
            )
        except Exception:
            selected = "hsmm_oto"
        self._mfa_explicitly_selected = selected == "mfa"
        if selected == "mfa" and hasattr(self, "_prompt_mfa_install_for_explicit_selection"):
            self._prompt_mfa_install_for_explicit_selection()
        self._save_config()

    def _on_ml_route_change(self, _value=None):
        if hasattr(self, "_get_ml_route_code"):
            self._get_ml_route_code()
        self._sync_aligner_ui()
        self._save_config()

    def _on_advanced_aligner_toggle(self):
        self._sync_aligner_ui()
        self._save_config()

    def _sync_aligner_ui(self):
        developer_enabled = (
            bool(self.developer_mode_enabled_var.get())
            if hasattr(self, "developer_mode_enabled_var")
            else False
        )
        # WFL-ASR is the default/primary aligner. normalize_aligner_name("WFL")
        # routes to the "wfl" engine; readiness (external torch env + pretrained
        # model) is checked at run time and falls back if not configured.
        # MFA is kept as a legacy option; the heuristic sequence aligner is moved
        # behind developer mode now that WFL is the primary non-MFA engine.
        options = ["WFL", HSMM_OTO_LABEL, "MFA (레거시)"]
        if developer_enabled:
            options = options + ["전용(시퀀스)"]
        lang = self._get_language()
        current = str(self.aligner_var.get() if hasattr(self, "aligner_var") else HSMM_OTO_LABEL).strip()
        fmt = normalize_auto_format_value(lang, self.auto_format_var.get()) if hasattr(self, "auto_format_var") else ""
        is_kr_template_only = (lang == "korean" and fmt in {"cmpx", "c_plus_v"})
        forced_no_mfa = bool(lang == "english" or is_kr_template_only)
        if current in {"No-MFA", "No-MFA (Experimental)"}:
            current = HSMM_OTO_LABEL
        if forced_no_mfa:
            current = HSMM_OTO_LABEL
        if current not in options:
            current = HSMM_OTO_LABEL
        if hasattr(self, "aligner_var"):
            self.aligner_var.set(current)
        if hasattr(self, "aligner_menu"):
            self.aligner_menu.configure(values=options)
            try:
                self.aligner_menu.set(current)
            except Exception:
                pass
        # Show engine-specific action buttons contextually: MFA diagnose/install
        # only when MFA (legacy) is selected; WFL status only when WFL is selected.
        current_engine = normalize_aligner_name(current, default="hsmm_oto")
        for _attr, _show in (
            ("mfa_repair_btn", current_engine == "mfa"),
            ("mfa_install_btn", current_engine == "mfa"),
            ("wfl_status_btn", current_engine == "wfl"),
            ("wfl_model_btn", current_engine == "wfl"),
        ):
            _btn = getattr(self, _attr, None)
            if _btn is not None:
                try:
                    _btn.grid() if _show else _btn.grid_remove()
                except Exception:
                    pass
        use_hsmm_oto = current == HSMM_OTO_LABEL
        use_no_mfa = forced_no_mfa
        use_sequence = current == "전용(시퀀스)"
        use_coarse_crnn = normalize_aligner_name(current, default="hsmm_oto") == "coarse_crnn"
        is_cmpx_preview = (lang == "korean" and fmt == "cmpx")
        is_c_plus_v_mode = (lang == "korean" and fmt == "c_plus_v")
        limit_ml_routes_for_no_mfa = use_no_mfa and not (
            lang == "english" or is_kr_template_only
        )
        current_route_code = (
            self._get_ml_route_code()
            if hasattr(self, "_get_ml_route_code")
            else str(self.ml_route_var.get() if hasattr(self, "ml_route_var") else "auto")
        )
        if is_kr_template_only:
            current_route_code = "nomfa"
        elif limit_ml_routes_for_no_mfa and current_route_code not in {"auto", "nomfa"}:
            current_route_code = "auto"

        route_values = (
            self._get_ml_route_option_labels(no_mfa_only=limit_ml_routes_for_no_mfa)
            if hasattr(self, "_get_ml_route_option_labels")
            else ["자동(자동 라우팅)", "No-MFA", "v1", "v2", "E2E 하이브리드(실험)"]
        )
        route_label = (
            self._ml_route_label_from_code(current_route_code)
            if hasattr(self, "_ml_route_label_from_code")
            else str(current_route_code)
        )
        if hasattr(self, "_set_ml_route_from_code"):
            self._set_ml_route_from_code(current_route_code)
        if hasattr(self, "ml_route_menu"):
            try:
                self.ml_route_menu.configure(
                    values=route_values,
                    state="disabled" if is_kr_template_only else "normal",
                )
                self.ml_route_menu.set(route_label)
            except Exception:
                pass

        no_mfa_mode_code = self._get_no_mfa_oto_mode_code()
        crnn_engine_code = (
            self._get_oto_crnn_engine_code()
            if hasattr(self, "_get_oto_crnn_engine_code")
            else "boundary_decoder"
        )
        if no_mfa_mode_code != "remap":
            no_mfa_mode_code = self._set_no_mfa_oto_mode_from_code("remap")
        if no_mfa_mode_code == "crnn":
            no_mfa_mode_code = self._set_no_mfa_oto_mode_from_code("remap")
        no_mfa_values = [NO_MFA_REMAP_LABEL]
        if hasattr(self, "no_mfa_oto_mode_menu"):
            try:
                self.no_mfa_oto_mode_menu.configure(values=no_mfa_values)
                self.no_mfa_oto_mode_menu.set(NO_MFA_REMAP_LABEL)
            except Exception:
                pass
        if hasattr(self, "oto_crnn_engine_menu"):
            try:
                engine_values = ["Stage1 heuristic only", "Boundary decoder + optional corrections"]
                self.oto_crnn_engine_menu.configure(
                    values=engine_values
                )
                self.oto_crnn_engine_menu.set(
                    "Stage1 heuristic only"
                    if crnn_engine_code == "stage1_heuristic"
                    else "Boundary decoder + optional corrections"
                )
            except Exception:
                pass
        no_mfa_mode_desc = NO_MFA_REMAP_LABEL
        if hasattr(self, "mfa_align_profile_menu"):
            self.mfa_align_profile_menu.configure(
                state="disabled" if (use_no_mfa or use_sequence or use_coarse_crnn or use_hsmm_oto) else "normal"
            )
        show_no_mfa_mode_row = use_no_mfa and not (
            lang == "english" or is_kr_template_only
        )
        if hasattr(self, "row_no_mfa_oto_mode") and self.row_no_mfa_oto_mode is not None:
            try:
                if show_no_mfa_mode_row:
                    if not self.row_no_mfa_oto_mode.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_no_mfa_oto_mode.pack(**pack_kwargs)
                else:
                    self.row_no_mfa_oto_mode.pack_forget()
            except Exception:
                pass
        show_crnn_model_row = False
        if hasattr(self, "row_oto_crnn_scorer_model") and self.row_oto_crnn_scorer_model is not None:
            try:
                if show_crnn_model_row:
                    if hasattr(self, "oto_crnn_scorer_model_menu") and hasattr(self, "_oto_crnn_model_choice_options"):
                        try:
                            choices = self._oto_crnn_model_choice_options()
                            self.oto_crnn_scorer_model_menu.configure(values=choices)
                            if hasattr(self, "oto_crnn_model_choice_var"):
                                current_label = ""
                                try:
                                    current_label = self.oto_crnn_model_choice_var.get()
                                except Exception:
                                    current_label = ""
                                if current_label not in choices:
                                    self._set_oto_crnn_model_choice_from_code("auto")
                        except Exception:
                            pass
                    if not self.row_oto_crnn_scorer_model.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_oto_crnn_scorer_model.pack(**pack_kwargs)
                else:
                    self.row_oto_crnn_scorer_model.pack_forget()
            except Exception:
                pass
        if hasattr(self, "row_oto_crnn_engine") and self.row_oto_crnn_engine is not None:
            try:
                if show_crnn_model_row:
                    if not self.row_oto_crnn_engine.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_oto_crnn_engine.pack(**pack_kwargs)
                else:
                    self.row_oto_crnn_engine.pack_forget()
            except Exception:
                pass
        if hasattr(self, "row_oto_crnn_model") and self.row_oto_crnn_model is not None:
            try:
                if show_crnn_model_row:
                    if not self.row_oto_crnn_model.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_oto_crnn_model.pack(**pack_kwargs)
                else:
                    self.row_oto_crnn_model.pack_forget()
            except Exception:
                pass
        if hasattr(self, "row_oto_stage2_model") and self.row_oto_stage2_model is not None:
            try:
                if show_crnn_model_row and crnn_engine_code != "stage1_heuristic":
                    if hasattr(self, "oto_stage2_model_menu") and hasattr(self, "_oto_stage2_model_choice_options"):
                        try:
                            choices = self._oto_stage2_model_choice_options()
                            self.oto_stage2_model_menu.configure(values=choices)
                            if hasattr(self, "oto_stage2_model_choice_var"):
                                current_label = ""
                                try:
                                    current_label = self.oto_stage2_model_choice_var.get()
                                except Exception:
                                    current_label = ""
                                if current_label not in choices:
                                    self._set_oto_stage2_model_choice_from_code("auto")
                        except Exception:
                            pass
                    if not self.row_oto_stage2_model.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_oto_stage2_model.pack(**pack_kwargs)
                else:
                    self.row_oto_stage2_model.pack_forget()
            except Exception:
                pass
        if hasattr(self, "row_oto_crnn_special_aliases") and self.row_oto_crnn_special_aliases is not None:
            try:
                if show_crnn_model_row:
                    if not self.row_oto_crnn_special_aliases.winfo_ismapped():
                        pack_kwargs = {"fill": "x", "pady": 4}
                        if hasattr(self, "row_align_extra") and self.row_align_extra is not None:
                            pack_kwargs["before"] = self.row_align_extra
                        self.row_oto_crnn_special_aliases.pack(**pack_kwargs)
                else:
                    self.row_oto_crnn_special_aliases.pack_forget()
            except Exception:
                pass
        if hasattr(self, "aligner_help_label"):
            if use_no_mfa:
                if lang == "english":
                    self.aligner_help_label.configure(text=t("(영어 Preview CVVC 모드에서는 정렬을 사용하지 않습니다.)"))
                elif lang == "korean" and fmt == "cmpx":
                    self.aligner_help_label.configure(text=t("(CMPX Preview 모드에서는 정렬을 사용하지 않습니다.)"))
                elif is_c_plus_v_mode:
                    self.aligner_help_label.configure(text=t("(한국어 C+V 모드는 템플릿 기반 생성으로 정렬을 사용하지 않습니다.)"))
                else:
                    self.aligner_help_label.configure(
                        text=f"(No-MFA 생성 방식: {no_mfa_mode_desc})"
                    )
            elif use_sequence:
                self.aligner_help_label.configure(text=t("(시퀀스 라벨 기반 전용 aligner baseline을 사용합니다.)"))
            elif use_hsmm_oto:
                self.aligner_help_label.configure(text=t("(TextGrid 없이 파일명 순서 기반 HSMM OTO를 생성합니다.)"))
            elif use_coarse_crnn:
                self.aligner_help_label.configure(text=t("(CRNN OTO 직접 예측을 사용합니다. TextGrid 정렬 단계는 건너뜁니다.)"))
            else:
                self.aligner_help_label.configure(text=t("(MFA 정렬을 사용합니다. MFA가 없으면 설치 여부를 먼저 확인합니다.)"))
        if hasattr(self, "pipeline_step_align_btn") and self.pipeline_step_align_btn is not None:
            try:
                if use_no_mfa or use_hsmm_oto:
                    self.pipeline_step_align_btn.configure(
                        state="disabled",
                        fg_color="#8E98A6",
                        hover_color="#8E98A6",
                        text_color="#E7ECF2",
                    )
                else:
                    self.pipeline_step_align_btn.configure(
                        state="normal",
                        fg_color=PALETTE.primary_button_bg,
                        hover_color=PALETTE.primary_button_hover,
                        text_color=PALETTE.primary_button_text,
                    )
            except Exception:
                pass
        if hasattr(self, "align_step_title_label") and hasattr(self, "align_step_desc_label"):
            if use_no_mfa:
                if lang == "english":
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (영어 Preview)"))
                    self.align_step_desc_label.configure(text=t("영어 CVVC Preview 모드는 Lab/사전/MFA 없이 base OTO 목록으로 생성합니다."))
                elif lang == "korean" and fmt == "cmpx":
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (CMPX Preview)"))
                    self.align_step_desc_label.configure(text=t("한국어 CMPX Preview 모드는 Lab/사전/MFA 없이 base OTO를 WAV에 재매핑해 생성합니다."))
                elif is_c_plus_v_mode:
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (한국어 C+V)"))
                    self.align_step_desc_label.configure(text=t("한국어 C+V 모드는 템플릿 OTO를 WAV에 재매핑하는 방식으로 생성합니다."))
                else:
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (No-MFA)"))
                    self.align_step_desc_label.configure(
                        text=t("MFA 정렬 없이 진행합니다. 베이스 OTO를 WAV에 재매핑하고 저신뢰 라인은 음향 후보 점수로 보정합니다.")
                    )
            else:
                if use_sequence:
                    self.align_step_title_label.configure(text=t("2. 음성 정렬 (전용 시퀀스)"))
                    self.align_step_desc_label.configure(text=t("frame-hop 시퀀스 라벨 기반으로 TextGrid를 생성합니다. 실패 시 MFA fallback을 사용합니다."))
                elif use_hsmm_oto:
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (HSMM OTO)"))
                    self.align_step_desc_label.configure(text=t("OTO 생성 단계에서 파일명 슬롯 기반 HSMM 디코더로 oto.ini를 생성합니다."))
                elif use_coarse_crnn:
                    self.align_step_title_label.configure(text=t("2. 정렬 단계 건너뜀 (CRNN OTO)"))
                    self.align_step_desc_label.configure(text=t("OTO 생성 단계에서 CRNN 직접 예측 모델로 oto.ini를 생성합니다."))
                else:
                    self.align_step_title_label.configure(text=t("2. 음성 정렬"))
                    self.align_step_desc_label.configure(text=t("MFA로 TextGrid를 생성합니다. MFA가 없으면 설치 여부를 먼저 확인합니다."))

    def _toggle_developer_mode(self):
        if not hasattr(self, "developer_mode_enabled_var"):
            return
        self.developer_mode_enabled_var.set(not bool(self.developer_mode_enabled_var.get()))
        self._sync_developer_mode_ui()
        self._save_config()

    def _sync_developer_mode_ui(self):
        enabled = self._developer_mode_enabled()
        if hasattr(self, "dev_mode_btn"):
            self.dev_mode_btn.configure(
                text=t("기본 보기") if enabled else t("고급 보기"),
                fg_color="#7E91AD" if enabled else "#C2CFDF",
                text_color=PALETTE.primary_button_text if enabled else "#2A3A50",
            )
        self._sync_secondary_tab_visibility(enabled)
        if hasattr(self, "advanced_developer_frame") and self.advanced_developer_frame is not None:
            if enabled:
                try:
                    if not self.advanced_developer_frame.winfo_ismapped():
                        self.advanced_developer_frame.pack(fill="x", padx=0, pady=(0, 0))
                except Exception:
                    pass
            else:
                try:
                    self.advanced_developer_frame.pack_forget()
                except Exception:
                    pass
        if hasattr(self, "boundary_smoke_btn") and self.boundary_smoke_btn is not None:
            try:
                is_running = bool(getattr(self, "is_running", False))
                self.boundary_smoke_btn.configure(state="normal" if (enabled and not is_running) else "disabled")
            except Exception:
                pass
        if hasattr(self, "boundary_smoke_hint_label") and self.boundary_smoke_hint_label is not None:
            try:
                self.boundary_smoke_hint_label.configure(
                    text_color=PALETTE.hint_text if enabled else "#AEB7C6"
                )
            except Exception:
                pass
        if hasattr(self, "phoneme_boundary_visualize_btn") and self.phoneme_boundary_visualize_btn is not None:
            try:
                is_running = bool(getattr(self, "is_running", False))
                self.phoneme_boundary_visualize_btn.configure(state="normal" if (enabled and not is_running) else "disabled")
            except Exception:
                pass
        if hasattr(self, "phoneme_boundary_model_menu") and self.phoneme_boundary_model_menu is not None:
            try:
                self.phoneme_boundary_model_menu.configure(state="normal" if enabled else "disabled")
            except Exception:
                pass
        if hasattr(self, "phoneme_boundary_visualize_hint") and self.phoneme_boundary_visualize_hint is not None:
            try:
                self.phoneme_boundary_visualize_hint.configure(
                    text_color=PALETTE.hint_text if enabled else "#AEB7C6"
                )
            except Exception:
                pass
        detail_frames = getattr(self, "vc_neighbor_detail_frames", [])
        for frame in detail_frames:
            if frame is None:
                continue
            if enabled:
                try:
                    if not frame.winfo_ismapped():
                        frame.pack(fill="x", padx=10, pady=(2, 8))
                except Exception:
                    pass
            else:
                try:
                    frame.pack_forget()
                except Exception:
                    pass
        for widget in getattr(self, "vc_neighbor_detail_controls", []):
            if widget is None:
                continue
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except Exception:
                pass
        if hasattr(self, "_sync_ml_correction_ui"):
            self._sync_ml_correction_ui()
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()

    def _set_suboption_container_enabled(self, container, enabled: bool):
        if container is None:
            return
        muted_text_color = "#AEB7C6"
        default_text_color = PALETTE.neutral_text
        stack = [container]
        visited = set()
        while stack:
            widget = stack.pop()
            if widget is None:
                continue
            marker = id(widget)
            if marker in visited:
                continue
            visited.add(marker)
            try:
                children = list(widget.winfo_children())
            except Exception:
                children = []
            stack.extend(children)
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except Exception:
                pass
            try:
                current_text_color = widget.cget("text_color")
            except Exception:
                continue
            try:
                if not hasattr(widget, "_utoa_enabled_text_color"):
                    base_color = current_text_color
                    if base_color in (None, "", "transparent"):
                        base_color = default_text_color
                    setattr(widget, "_utoa_enabled_text_color", base_color)
                base_color = getattr(widget, "_utoa_enabled_text_color", default_text_color)
                widget.configure(text_color=base_color if enabled else muted_text_color)
            except Exception:
                pass

    def _sync_ml_correction_ui(self):
        enabled = (
            bool(self.enable_ml_correction_var.get())
            if hasattr(self, "enable_ml_correction_var")
            else True
        )
        for container in (
            getattr(self, "advanced_dev_runtime_controls_frame", None),
            getattr(self, "advanced_ml_section_frame", None),
            getattr(self, "advanced_ml_public_frame", None),
        ):
            self._set_suboption_container_enabled(container, enabled)

    def _on_enable_ml_correction_toggle(self):
        if hasattr(self, "_sync_ml_correction_ui"):
            self._sync_ml_correction_ui()
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()
        self._save_config()

    def _lightgbm_correction_disabled(self) -> bool:
        if not hasattr(self, "disable_lightgbm_correction_var"):
            return False
        try:
            return bool(self.disable_lightgbm_correction_var.get())
        except Exception:
            return False

    def _should_apply_hsmm_lightgbm(self, enable_ml_correction: bool) -> bool:
        return bool(enable_ml_correction) and not self._lightgbm_correction_disabled()

    def _sync_mapping_supervised_ui(self):
        enabled = (
            bool(self.mapping_supervised_enable_var.get())
            if hasattr(self, "mapping_supervised_enable_var")
            else True
        )
        ml_master_enabled = (
            bool(self.enable_ml_correction_var.get())
            if hasattr(self, "enable_ml_correction_var")
            else True
        )
        enabled = bool(enabled and ml_master_enabled)
        for frame in getattr(self, "mapping_supervised_dependent_frames", []):
            self._set_suboption_container_enabled(frame, enabled)

    def _on_mapping_supervised_toggle(self):
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        self._save_config()

    def _sync_ml_e2e_controls(self):
        e2e_enabled = (
            bool(self.ml_e2e_enable_var.get())
            if hasattr(self, "ml_e2e_enable_var")
            else False
        )
        ml_master_enabled = (
            bool(self.enable_ml_correction_var.get())
            if hasattr(self, "enable_ml_correction_var")
            else True
        )
        enabled = bool(e2e_enabled and ml_master_enabled)
        for frame in getattr(self, "ml_e2e_dependent_frames", []):
            self._set_suboption_container_enabled(frame, enabled)

    def _on_ml_e2e_toggle(self):
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()
        self._save_config()

    def _sync_vc_correction_toggle(self):
        if not hasattr(self, "vc_correction_enable_var"):
            return
        continuity_enabled = (
            bool(self.kr_continuity_enable_var.get())
            if hasattr(self, "kr_continuity_enable_var")
            else True
        )
        kr_enabled = (
            bool(self.kr_vc_neighbor_enable_var.get())
            if hasattr(self, "kr_vc_neighbor_enable_var")
            else True
        )
        ja_enabled = (
            bool(self.ja_vc_neighbor_enable_var.get())
            if hasattr(self, "ja_vc_neighbor_enable_var")
            else True
        )
        self.vc_correction_enable_var.set(bool(continuity_enabled and kr_enabled and ja_enabled))
        developer_enabled = (
            bool(self.developer_mode_enabled_var.get())
            if hasattr(self, "developer_mode_enabled_var")
            else False
        )
        frame_by_lang = getattr(self, "vc_neighbor_detail_frame_by_lang", None)
        if isinstance(frame_by_lang, dict) and frame_by_lang:
            self._set_suboption_container_enabled(frame_by_lang.get("kr"), bool(developer_enabled and kr_enabled))
            self._set_suboption_container_enabled(frame_by_lang.get("ja"), bool(developer_enabled and ja_enabled))
        else:
            for frame in getattr(self, "vc_neighbor_detail_frames", []):
                self._set_suboption_container_enabled(frame, developer_enabled)
        if hasattr(self, "_sync_cvn_correction_toggle"):
            self._sync_cvn_correction_toggle()

    def _sync_cvn_correction_toggle(self):
        enabled = (
            bool(self.cvn_correction_enable_var.get())
            if hasattr(self, "cvn_correction_enable_var")
            else True
        )
        checkbox = getattr(self, "cvn_low_conf_only_checkbox", None)
        if checkbox is not None:
            self._set_suboption_container_enabled(checkbox, enabled)
        if not enabled and hasattr(self, "cvn_low_conf_only_var"):
            try:
                self.cvn_low_conf_only_var.set(False)
            except Exception:
                pass

    def _on_cvn_correction_toggle(self):
        if hasattr(self, "_sync_cvn_correction_toggle"):
            self._sync_cvn_correction_toggle()
        self._save_config()

    def _on_vc_neighbor_language_toggle(self):
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        self._save_config()

    def _on_vc_correction_toggle(self):
        enabled = bool(self.vc_correction_enable_var.get()) if hasattr(self, "vc_correction_enable_var") else True
        if hasattr(self, "kr_continuity_enable_var"):
            self.kr_continuity_enable_var.set(enabled)
        if hasattr(self, "kr_vc_neighbor_enable_var"):
            self.kr_vc_neighbor_enable_var.set(enabled)
        if hasattr(self, "ja_vc_neighbor_enable_var"):
            self.ja_vc_neighbor_enable_var.set(enabled)
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        self._save_config()

    def _on_use_boundary_model_toggle(self):
        enabled = bool(self.use_boundary_model_var.get())
        if enabled:
            path = str(self.boundary_model_path_var.get() or "").strip()
            if not path or not os.path.isfile(path):
                self._append_log("⚠ 학습된 경계 모델 경로가 없거나 잘못되었습니다. 규칙 기반으로 동작합니다.")
            else:
                self._append_log(f"ℹ 학습된 경계 모델 사용: {os.path.basename(path)}")
        else:
            self._append_log("ℹ 경계 모델 끔 — 규칙 기반(world_v1) 인코더 사용")
        self._save_config()

    def _on_no_base_oto_toggle(self):
        if self._requires_base_oto_for_current_mode():
            self.no_base_oto_var.set(False)
            self.tpl_entry.configure(state="normal")
            if hasattr(self, "tpl_browse_btn"):
                self.tpl_browse_btn.configure(state="normal")
            self._save_config()
            return
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
        if hasattr(self, "_apply_format_consistency_recommendation"):
            self._apply_format_consistency_recommendation(save_config=False, write_log=False)
        self._sync_base_oto_requirement_ui()
        self._sync_cmpx_route_lock()
        self._sync_aligner_ui()
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
        if lang == "english":
            self.ml_coupled_status_label.configure(text=t("현재 포맷: cvvc | 영어 Preview 모드에서는 ML 보정을 사용하지 않습니다."))
            if hasattr(self, "ml_coupled_status_detail_label"):
                self.ml_coupled_status_detail_label.configure(text="")
            return
        fmt = ""
        if hasattr(self, "auto_format_var"):
            fmt = normalize_auto_format_value(lang, self.auto_format_var.get())
        if lang == "english":
            self.ml_coupled_status_label.configure(text=t("현재 포맷: cvvc | 영어 Preview 모드에서는 ML 보정을 사용하지 않습니다."))
            if hasattr(self, "ml_coupled_status_detail_label"):
                self.ml_coupled_status_detail_label.configure(text="")
            return
        if lang == "korean" and fmt in {"cmpx", "c_plus_v"}:
            mode_label = "cmpx" if fmt == "cmpx" else "c_plus_v"
            self.ml_coupled_status_label.configure(
                text=f"현재 포맷: {mode_label} | 템플릿 전용 모드에서는 ML 보정을 사용하지 않습니다."
            )
            if hasattr(self, "ml_coupled_status_detail_label"):
                self.ml_coupled_status_detail_label.configure(text="")
            return
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
            self.advanced_toggle_btn.configure(text=t("▼ 고급 옵션 (특수 발음)"))
            self.advanced_options_frame.pack(fill="x", padx=10, pady=(0, 3))
        else:
            self.advanced_toggle_btn.configure(text=t("▶ 고급 옵션 (특수 발음)"))
            self.advanced_options_frame.pack_forget()

    def _get_params(self):
        """현재 슬라이더 값으로 파라미터 딕셔너리 생성"""
        if hasattr(self, "param_vars") and self.param_vars:
            return {key: var.get() for key, var in self.param_vars.items()}
        return None

    # ── MFA 설치 (GUI 내장) ──
