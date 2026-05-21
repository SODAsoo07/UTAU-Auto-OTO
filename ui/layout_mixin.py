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
NO_MFA_REMAP_LABEL = "Base OTO remap (legacy fallback)"
NO_MFA_SSL_SLOT_LABEL = "MFA-Free auto + fallback"


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
            self.auto_format_var = ctk.StringVar(value="・尖徐 ・川ｧ (・護棗)")

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

        # Machine-translation notice 窶・shown only when UI language is English or Japanese
        from ui.i18n import get_language as _get_ui_lang_now
        _mt_notice_texts = {
            "en": "Some translations are machine-generated and may contain errors.",
            "ja": "Some translations are machine-generated and may contain errors.",
        }
        _mt_text = _mt_notice_texts.get(_get_ui_lang_now(), "")
        if _mt_text:
            ctk.CTkLabel(
                header_row,
                text=f"  笞 {_mt_text}",
                font=("", 11),
                text_color=PALETTE.neutral_text,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

        # UI language selector (left panel top)
        from ui.i18n import get_language as _get_ui_lang
        ui_lang_top_row = ctk.CTkFrame(path_frame, fg_color="transparent")
        ui_lang_top_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(ui_lang_top_row, text=t("UI Language"), width=115, anchor="w").pack(side="left")
        if not hasattr(self, "ui_lang_var"):
            _cur_lang = _get_ui_lang()
            _ui_lang_display = "English" if _cur_lang == "en" else ("Japanese" if _cur_lang == "ja" else "Korean")
            self.ui_lang_var = ctk.StringVar(value=_ui_lang_display)
        self.ui_lang_dropdown = ctk.CTkOptionMenu(
            ui_lang_top_row,
            values=["Korean", "English", "Japanese"],
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
        build_left_label(lang_row, t("・ｸ・ｴ"), width=115).pack(side="left")
        if not hasattr(self, "lang_var"):
            self.lang_var = ctk.StringVar(value="Korean (﨑懋ｵｭ・ｴ)")
        language_values = ["Korean (﨑懋ｵｭ・ｴ)", "Japanese (譌･譛ｬ隱・"]
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
        build_left_label(row1, t("WAV 尞ｴ・・")).pack(side="left")
        self.wav_entry = ctk.CTkEntry(row1, placeholder_text=t("Select a WAV folder"))
        self.wav_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.wav_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        wav_browse_btn = ctk.CTkButton(row1, text=t("・ｾ・・ｳｴ・ｰ"), width=90, command=lambda: self._browse_folder(self.wav_entry))
        _style_primary_button(wav_browse_btn)
        wav_browse_btn.pack(side="right")

        row1b = build_form_row(form_body)
        ctk.CTkLabel(row1b, text="", width=115).pack(side="left")
        self.recursive_voicebank_scan_checkbox = ctk.CTkCheckBox(
            row1b,
            text=t("﨑們怱 尞ｴ・・・尖徐 夋川ラ(・ｰ・・・俯ｦｬ)"),
            variable=self.recursive_voicebank_scan_var,
            command=self._save_config,
        )
        self.recursive_voicebank_scan_checkbox.pack(side="left", padx=(6, 0))

        row2 = build_form_row(form_body)
        build_left_label(row2, t("奛懦伯・ｿ OTO:")).pack(side="left")
        self.tpl_entry = ctk.CTkEntry(row2, placeholder_text=t("・夋・・ｬ﨑ｭ (・・揆 ・・甯護攵・・・ｼ・ｨ ・ｰ・・・尖徐 ・晧┳)"))
        self.tpl_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.tpl_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        self.tpl_browse_btn = ctk.CTkButton(
            row2,
            text=t("・ｾ・・ｳｴ・ｰ"),
            width=90,
            command=lambda: self._browse_file(self.tpl_entry, [("OTO 甯護攵", "*.ini")]),
        )
        _style_primary_button(self.tpl_browse_btn)
        self.tpl_browse_btn.pack(side="right")

        row2b = build_form_row(form_body)
        ctk.CTkLabel(row2b, text="", width=115).pack(side="left")
        self.no_base_oto_checkbox = ctk.CTkCheckBox(
            row2b,
            text=t("奛懦伯・ｿ OTO ・・搆 (OpenUtau 嶸ｸ嶹・・川攵・ｬ・ｴ・､ ・尖徐 ・晧┳)"),
            variable=self.no_base_oto_var,
            command=self._on_no_base_oto_toggle,
            text_color=PALETTE.success_text,
        )
        self.no_base_oto_checkbox.pack(side="left", padx=(6, 0))

        row3 = build_form_row(form_body)
        build_left_label(row3, t("・罹･ ・ｽ・・")).pack(side="left")
        self.out_entry = ctk.CTkEntry(row3, placeholder_text=t("Output oto.ini path"))
        self.out_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        out_save_btn = ctk.CTkButton(
            row3,
            text=t("・・･"),
            width=90,
            command=lambda: self._browser_save(self.out_entry, [("OTO 甯護攵", "*.ini")]),
        )
        _style_primary_button(out_save_btn)
        out_save_btn.pack(side="right")

        row_suffix = build_form_row(form_body)
        build_left_label(row_suffix, t("・瀧ｯｸ・ｬ:")).pack(side="left")
        self.suffix_entry = ctk.CTkEntry(
            row_suffix,
            placeholder_text=t("・夋・・ｬ﨑ｭ: ・・ C4 (・ｨ・ ・川攵・ｬ・ｴ・､ ・晧乱 _C4 嶸倣・・・・・ｬ)"),
            textvariable=self.alias_suffix_var,
        )
        self.suffix_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.suffix_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))
        ctk.CTkLabel(
            row_suffix,
            text=t("(・罹･ alias ・瀧ｯｸ・ｬ)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")

        row_format = build_form_row(form_body)
        build_left_label(row_format, t("嶸菩享 ・・・")).pack(side="left")
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
            text=t("(奛懦伯・ｿ ・・ｴ・ ・ｴ・﨑俾ｲ・・ｰ・ ・・圸)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)
        self.ja_alias_row = build_form_row(form_body)
        build_left_label(self.ja_alias_row, t("JP ・川攵・ｬ・ｴ・､:")).pack(side="left")
        self.ja_alias_style_menu = ctk.CTkOptionMenu(
            self.ja_alias_row,
            values=["auto", "hiragana", "romaji"],
            variable=self.ja_alias_style_var,
            width=190,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.ja_alias_style_menu)
        self.ja_alias_style_menu.pack(side="left", padx=(6, 8))
        self.ja_alias_hint_label = ctk.CTkLabel(
            self.ja_alias_row,
            text=t("(・ｼ・ｸ・ｴ OTO ・・圸)"),
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
                text=t("List-only 﨑ｩ・ｱ(・､嵭・"),
                variable=self.en_cvvc_list_fallback_var,
                command=self._save_config,
                width=140,
            )
            self.en_cvvc_list_fallback_checkbox.pack(side="left", padx=(0, 8))

            self.en_cvvc_hint_label = ctk.CTkLabel(
                self.en_cvvc_row,
                text=t("(Preview: ・ｰ・ｸ OTO + ・ｵ・・・・list-only(vv/cc/alt) 﨑ｩ・ｱ)"),
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
        build_left_label(self.row_aligner, t("・簿ｬ ・肥ｧ・")).pack(side="left")
        self.aligner_menu = ctk.CTkOptionMenu(
            self.row_aligner,
            values=["MFA", "・・圸(・懦・､)"],
            variable=self.aligner_var,
            width=190,
            command=self._on_aligner_change,
        )
        _style_blue_menu(self.aligner_menu)
        self.aligner_menu.pack(side="left", padx=(6, 8))
        self.aligner_help_label = ctk.CTkLabel(
            self.row_aligner,
            text=t("(・ｰ・ｸ・ MFA・・笈・､. 﨑・囈 ・・・尖徐 ・､・俯姓・壱共.)"),
            text_color=PALETTE.neutral_text,
        )
        self.aligner_help_label.pack(side="left", fill="x", expand=True)
        self.row_no_mfa_oto_mode = build_form_row(form_body)
        build_left_label(self.row_no_mfa_oto_mode, t("No-MFA ・晧┳:")).pack(side="left")
        self.no_mfa_oto_mode_menu = ctk.CTkOptionMenu(
            self.row_no_mfa_oto_mode,
            values=[
                NO_MFA_SSL_SLOT_LABEL,
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
            text=t("(기본: MFA-Free 모델 기반 자동 지정, 낮은 신뢰도는 자동 fallback)"),
            text_color=PALETTE.neutral_text,
        )
        self.no_mfa_oto_mode_hint_label.pack(side="left", fill="x", expand=True)
        # Boundary scorer checkpoint picker. Discovered model files appear as
        # additional options; "・尖徐 (auto)" defers to the resolver's default
        # (mtime-newest non-experimental). Hidden together with the device row
        # under the same developer-mode + retired-model gate (see _sync_aligner_ui).
        self.row_oto_crnn_scorer_model = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_scorer_model, t("Boundary scorer:")).pack(side="left")
        scorer_choices = (
            self._oto_crnn_model_choice_options()
            if hasattr(self, "_oto_crnn_model_choice_options")
            else ["・尖徐 (auto)"]
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
            text=t("(・尖徐=・ｰ・ｸ ・ｨ・ｸ, ・､・ｸ .pt ・夋・・・・罷｡・・・餓亨 ・們・)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_crnn_scorer_model_hint.pack(side="left", fill="x", expand=True)
        self.row_oto_crnn_scorer_model.pack_forget()
        self.row_oto_crnn_engine = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_engine, t("Retired model mode:")).pack(side="left")
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
            else ["・尖徐 (auto)"]
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
            text=t("(・夋・・・Boundary Decoder ・､・川・ 2・ｨ・・OTO anchor・ｼ ・ｬ・・・"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_stage2_model_hint.pack(side="left", fill="x", expand=True)
        self.row_oto_stage2_model.pack_forget()
        self.row_oto_crnn_model = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_model, t("Retired model device:")).pack(side="left")
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
            text=t("(・罷｡ ・罷ｰ肥擽・､. auto = GPU ・・ｩ ・・GPU ・ｬ・ｩ)"),
            text_color=PALETTE.neutral_text,
        )
        self.oto_crnn_model_hint_label.pack(side="left", fill="x", expand=True)
        self.row_oto_crnn_model.pack_forget()
        self.row_oto_crnn_special_aliases = build_form_row(form_body)
        build_left_label(self.row_oto_crnn_special_aliases, t("孖ｹ・・・川攵・ｬ・ｴ・､:")).pack(side="left")
        self.oto_crnn_special_aliases_entry = ctk.CTkEntry(
            self.row_oto_crnn_special_aliases,
            textvariable=self.oto_crnn_special_aliases_var,
            placeholder_text="・ｼ岺罹｡・・ｬ・・(・・ Sp, br, cl)",
        )
        self.oto_crnn_special_aliases_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.oto_crnn_special_aliases_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.row_oto_crnn_special_aliases.pack_forget()
        self.row_align_extra = build_form_row(form_body)
        build_left_label(self.row_align_extra, t("MFA ・簿ｬ 嵓・｡懦符:")).pack(side="left")
        self.mfa_align_profile_menu = ctk.CTkOptionMenu(
            self.row_align_extra,
            values=["accurate", "fast", "fast + fallback", "normal"],
            variable=self.mfa_align_profile_var,
            width=220,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(self.mfa_align_profile_menu)
        self.mfa_align_profile_menu.pack(side="left", padx=(6, 8))
        ctk.CTkLabel(
            self.row_align_extra,
            text=t("(・ｰ・ｸ=・倣剳・・・嶸・"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", fill="x", expand=True)





        self.row_aligner_advanced = build_form_row(form_body)
        ctk.CTkLabel(
            self.row_aligner_advanced,
            text=t("・簿ｬ ・・・・ｵ・們捩 ・尖徐・ｼ・・・ｰ・簿姓・壱共."),
            text_color=PALETTE.neutral_text,
            anchor="w",
        ).pack(side="left", padx=(121, 0))

        advanced_row = build_form_row(path_frame)
        self.advanced_toggle_btn = ctk.CTkButton(
            advanced_row,
            text=t("笆ｶ ・緋ｰ ・ｵ・・(孖ｹ・・・懍搆)"),
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
            text=t("﨑・囈﨑・・ｽ・ｰ・尖ｧ・・ｬ・ｩ﨑們┷・・"),
            text_color=PALETTE.hint_text,
            anchor="e",
        )
        self.advanced_hint_label.pack(fill="x", padx=12, pady=(0, 6))

        self.advanced_options_frame = ctk.CTkFrame(path_frame, fg_color="transparent")

        row0 = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0, text=t("孖ｹ・・・懍搆 (・夋・:"), width=120, anchor="w").pack(side="left")
        self.custom_entry = ctk.CTkEntry(row0, placeholder_text=t("・､・､奛 ・､﨑・・懍ｹ・甯護攵 (.txt)"), textvariable=self.custom_phoneme_var)
        self.custom_entry.configure(fg_color=PALETTE.input_bg, border_color=PALETTE.input_border)
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        custom_browse_btn = ctk.CTkButton(row0, text=t("・ｾ・・ｳｴ・ｰ"), width=90, command=lambda: self._browse_file(self.custom_entry, [("Text 甯護攵", "*.txt")]))
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
            text=t("・罹ｰ懍梵 ・､・・OFF"),
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
            "甯護擽嵓・攵・ｸ": self._build_pipeline_tab,
            "Advanced": self._build_advanced_settings_tab,
            "Parameters": self._build_params_tab,
            "・懋ｷｸ": self._build_log_tab,
            "・・┷ ・懋ｷｸ": self._build_detail_log_tab,
            "增ｬ・壱肌": self._build_credits_tab,
        }
        self._built_tabs = set()
        # Keep all tab headers visible from startup, but build content lazily.
        for tab_name in self._lazy_tab_builders.keys():
            try:
                self._get_or_add_tab(tab_name)
            except Exception:
                pass
        for tab_name in ("甯護擽嵓・攵・ｸ",):
            self._ensure_tab_built(tab_name)
        self.tabview.set("甯護擽嵓・攵・ｸ")
        self._on_tabview_change()
        self._schedule_lazy_tab_prewarm()
        self._sync_developer_mode_ui()

        bottom = ctk.CTkFrame(self, fg_color=PALETTE.panel_bg)
        bottom.pack(side="bottom", fill="x", padx=15, pady=(5, 15))
        self.bottom_bar_frame = bottom

        status_group = ctk.CTkFrame(bottom, fg_color="transparent")
        status_group.pack(side="left", fill="x", expand=True, padx=(10, 6))
        self.bottom_status_group = status_group

        self.status_label = ctk.CTkLabel(status_group, text=t("Ready"), anchor="w", text_color=PALETTE.neutral_text)
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
            text=t("Run"),
            font=("", 14, "bold"),
            width=150,
            height=40,
            command=self._run_full_pipeline,
        )
        self.run_btn.pack(side="right", padx=10)

        self.report_btn = ctk.CTkButton(
            actions_group,
            text=t("菅 ・罹ｳｴ ・ｬ尞ｬ孖ｸ ・ｵ・ｬ"),
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
            "lang": _safe_var_get("lang_var", "Korean (﨑懋ｵｭ・ｴ)"),
            "format": _safe_var_get("auto_format_var", "・尖徐 ・川ｧ (・護棗)"),
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

    def _request_tab_build(self, tab_name):
        name = str(tab_name or "").strip()
        if not name:
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
        if not name:
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

        if name == "甯護擽嵓・攵・ｸ" and hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()
        if name == "Advanced":
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
        preferred_order = tuple(builders.keys())
        queue = [name for name in preferred_order if name in builders and name not in built_tabs]
        for name in builders.keys():
            if name not in built_tabs and name not in queue:
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
                self._append_log("[UI] ・卓羅 ・卓乱・・奛誤ｧ壱･ｼ ・・ｽ﨑 ・・・・慣・壱共.")
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
                    hint.configure(text=t("(・・牟 ・夋・・・岺懍亨・俯株 Preview ・・圸 ・､・・"))
                else:
                    hint.configure(text=t("(Preview: ・ｰ・ｸ OTO + ・ｵ・・・・list-only(vv/cc/alt) 﨑ｩ・ｱ)"))
            except Exception:
                pass

    def _get_language(self):
        """UI ・ｸ・ｴ ・夋晝ｰ廷揆 ・ｴ・ ・罷糖('korean'/'japanese'/'english')・・・母ｷ懦剩﨑ｩ・壱共."""
        sel = str(self.lang_var.get() or "")
        low = sel.strip().lower()
        if "english" in low or "・・牟" in sel or low in {"en", "eng"}:
            return "english" if self._is_preview_channel() else "korean"
        if (
            "japanese" in low
            or "譌･譛ｬ" in sel
            or "japanese" in sel
            or low in {"ja", "jp"}
        ):
            return "japanese"
        if (
            "korean" in low
            or "﨑懋ｵｭ" in sel
            or "・ｰ・" in sel
            or low in {"ko", "kr"}
        ):
            return "korean"
        return "korean"

    def _get_auto_format_options(self, language=None):
        lang = language or self._get_language()
        if lang == "korean":
            values = [
                "・尖徐 ・川ｧ (・護棗)",
                "CV",
                "CVC (﨑懋ｵｭ・ｴ ・・圸)",
                "COC (﨑懋ｵｭ・ｴ CVC 甯護・嶸・",
                "CVVC",
                "VCV (・ｰ・作搆)",
            ]
            if self._is_preview_channel():
                values.append("C+V (奛懦伯・ｿ ・・圸)")
                values.append("CMPX (嵓・ｦｬ・ｰ)")
            return values
        if lang == "english":
            return ["CVVC"]
        return ["auto", "CV", "CVVC", "VCV"]

    def _set_auto_format_from_code(self, format_code, language=None):
        lang = language or self._get_language()
        values = self._get_auto_format_options(lang)
        if hasattr(self, "format_dropdown"):
            self.format_dropdown.configure(values=values)
        if lang == "english":
            self.auto_format_var.set("CVVC")
            return
        label_map = {
            "": "・尖徐 ・川ｧ (・護棗)",
            "cv": "CV",
            "c_plus_v": "C+V (奛懦伯・ｿ ・・圸)",
            "cvc": "CVC (﨑懋ｵｭ・ｴ ・・圸)",
            "coc": "COC (﨑懋ｵｭ・ｴ 甯護・嶸・",
            "cvvc": "CVVC",
            "vcv": "VCV (・ｰ・作搆)",
            "cmpx": "CMPX (嵓・ｦｬ・ｰ)",
        }
        label = label_map.get(str(format_code or "").strip().lower(), "・尖徐 ・川ｧ (・護棗)")
        if label not in values:
            if label in {"CVC (﨑懋ｵｭ・ｴ ・・圸)", "COC (﨑懋ｵｭ・ｴ 甯護・嶸・", "C+V (奛懦伯・ｿ ・・圸)"} and lang != "korean":
                label = "CV"
            elif label == "C+V (奛懦伯・ｿ ・・圸)" and not self._is_preview_channel():
                label = "CV"
            else:
                label = "・尖徐 ・川ｧ (・護棗)"
        self.auto_format_var.set(label)

    def _on_ui_language_change(self, value: str) -> None:
        from ui.i18n import set_language as _set_lang
        normalized = str(value or "").strip()
        lowered = normalized.lower()
        if lowered in {"english", "en"}:
            lang_code = "en"
        elif lowered in {"japanese", "ja"} or ("譌･" in normalized):
            lang_code = "ja"
        else:
            lang_code = "ko"
        _set_lang(lang_code)
        if hasattr(self, "_save_config"):
            self._save_config()
        if hasattr(self, "_ui_lang_hint_label"):
            from ui.i18n import t as _t
            self._ui_lang_hint_label.configure(text=_t("・ｬ・懍梠 弡・・・圸・ｩ・壱共."))

    def _on_language_change(self, value):
        lang = self._get_language()
        if lang == "korean":
            self.lang_info_label.configure(text=t("﨑懋ｵｭ・ｴ ・ｨ・・a, k, ga ・ｱ) ・川攵・ｬ・ｴ・､・ｼ ・ｰ・・ｼ・・・晧┳﨑ｩ・壱共."))
            self.lang_notice_label.configure(
                text=(
                    "嶸・椪 ・ｸ・ｴ: 﨑懋ｵｭ・ｴ\n"
                    "Lab ・晧┳, ・ｬ・・・晧┳, ・簿ｬ, OTO ・・げ・ｴ ・ｨ・・﨑懋ｵｭ・ｴ ・懍ｹ呷愍・・・・哩・ｩ・壱共.\n"
                    "・ｰ・ｸ ・ｸ・罷畠: UTF-8"
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
            self.lang_info_label.configure(text=t("・ｼ・ｸ・ｴ ・ｨ・・a, k, ka ・ｱ) ・川攵・ｬ・ｴ・､・ｼ ・ｰ・・ｼ・・・晧┳﨑ｩ・壱共."))
            self.lang_notice_label.configure(
                text=(
                    "嶸・椪 ・ｸ・ｴ: ・ｼ・ｸ・ｴ\n"
                    "孖ｹ・・・懍搆 ・ｰ嶸ｸ・ ・樌攤 甯護攵・ Lab ・晧┳ ・・乱 ・ｸ・ｴ ・夋晧揆 ・､・・嶹菩攤﨑們┷・・\n"
                    "・ｰ・ｸ ・ｸ・罷畠: UTF-8"
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
            self.lang_info_label.configure(text=t("・・牟 CVVC base OTO・ｼ ・壱洳・ ・晧┳﨑ｩ・壱共. (嵓・ｦｬ・ｰ ・・圸)"))
            self.lang_notice_label.configure(
                text=(
                    "嶸・椪 ・ｸ・ｴ: ・・牟(嵓・ｦｬ・ｰ)\n"
                    "Lab/・ｬ・・MFA・ｼ ・ｴ・壱峅・, ・夋晨復 EN CVVC ・ｬ・､孖ｸ・・base OTO・ｼ ・ｰ﨑ｩ﨑ｴ ・晧┳﨑ｩ・壱共.\n"
                    "list-only(vv/cc/alt) ・ｹ・們捩 ・ｵ・們愍・・﨑ｩ・ｱ ・晧┳﨑 ・・・溢慣・壱共."
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
                self.aligner_var.set("MFA")
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
            text = t("・ｼ・・ｱ ・ｴ・・・ｬ・ｩ")
        elif lang == "korean":
            text = t("・ｰ・作┳ ・ｴ・・・ｬ・ｩ")
        else:
            text = t("・ｰ・作┳/甯護攵 ・ｼ・・ｱ ・ｴ・・・ｬ・ｩ")
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
            label = "奛懦伯・ｿ OTO ・・搆 (OpenUtau 嶸ｸ嶹・・川攵・ｬ・ｴ・､ ・尖徐 ・晧┳)"
            if requires_base:
                label = "奛懦伯・ｿ OTO 﨑・・ (嶸・椪 ・ｨ・・"
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
                    self.auto_format_var.set("CV")
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
        if style == "hiragana":
            return "hiragana"
        if style == "・罹ｧ溢梵":
            return "romaji"
        return "original"

    @staticmethod
    def _normalize_no_mfa_oto_mode_code(value):
        raw = str(value or "").strip().lower()
        if raw in {"crnn", "oto_crnn", "oto-crnn", "crnn_oto", "crnn-oto"}:
            return "remap"
        if raw in {"mfa_free", "mfa-free", "ssl_slot", "ssl-slot", "mfa_free_ssl_slot"}:
            return "mfa_free_ssl_slot"
        if raw in {"remap", "base_remap", "base"}:
            return "remap"
        text = str(value or "").strip()
        if text == NO_MFA_SSL_SLOT_LABEL:
            return "mfa_free_ssl_slot"
        if text == "Retired OTO predictor":
            return "remap"
        if text == NO_MFA_REMAP_LABEL:
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
        # Legacy direct model path is removed from UI routing.
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
    # The dropdown label is "・尖徐 (auto)" or a friendly label produced by
    # Retired model controls are kept as inert config compatibility helpers.

    _RETIRED_OTO_MODEL_CHOICE_AUTO_LABEL = "・尖徐 (auto)"

    def _list_oto_crnn_model_choices(self) -> list[dict[str, object]]:
        return []

    def _oto_crnn_model_choice_options(self) -> list[str]:
        labels = [self._RETIRED_OTO_MODEL_CHOICE_AUTO_LABEL]
        for item in self._list_oto_crnn_model_choices():
            label = str(item.get("label") or item.get("name") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _oto_crnn_label_to_code(self, label: object) -> str:
        text = str(label or "").strip()
        if not text or text == self._RETIRED_OTO_MODEL_CHOICE_AUTO_LABEL or text.lower() == "auto":
            return "auto"
        for item in self._list_oto_crnn_model_choices():
            if str(item.get("label") or "") == text:
                return str(item.get("name") or "auto")
            if str(item.get("name") or "") == text:
                return str(item.get("name") or "auto")
        # Unknown label (e.g. stale config) 竊・fall back to auto-resolve.
        return "auto"

    def _oto_crnn_code_to_label(self, code: object) -> str:
        text = str(code or "").strip()
        if not text or text.lower() == "auto":
            return self._RETIRED_OTO_MODEL_CHOICE_AUTO_LABEL
        for item in self._list_oto_crnn_model_choices():
            if str(item.get("name") or "") == text:
                return str(item.get("label") or item.get("name") or text)
        return self._RETIRED_OTO_MODEL_CHOICE_AUTO_LABEL

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
                self._append_log(f"[retired-oto] boundary scorer model = {code}")
            except Exception:
                pass
        if hasattr(self, "_save_config"):
            self._save_config()

    _OTO_STAGE2_MODEL_CHOICE_AUTO_LABEL = "・尖徐 (auto)"

    def _list_oto_stage2_model_choices(self) -> list[dict[str, object]]:
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
                self._append_log(f"[retired-oto] stage2 model = {code}")
            except Exception:
                pass
        if hasattr(self, "_save_config"):
            self._save_config()

    def _on_oto_stage2_setting_change(self):
        if hasattr(self, "_append_log"):
            try:
                enabled = bool(self.oto_stage2_enable_var.get()) if hasattr(self, "oto_stage2_enable_var") else False
                self._append_log(f"[retired-oto] Stage2 OTO Assigner {'ON' if enabled else 'OFF'}")
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
        label = NO_MFA_SSL_SLOT_LABEL if normalized == "mfa_free_ssl_slot" else NO_MFA_REMAP_LABEL
        if hasattr(self, "no_mfa_oto_mode_var"):
            try:
                self.no_mfa_oto_mode_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_no_mfa_oto_mode_code(self):
        if not hasattr(self, "no_mfa_oto_mode_var"):
            return "mfa_free_ssl_slot"
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
        if profile in {"fast", "normal"}:
            return "fast"
        if profile in {
            "・簿ｰ + 嶹肥梵 ・・搗",
            "・倣剳・・・ｰ・ + 嶹肥梵 ・・搗",
            "accurate_adapted",
            "speaker_adapted",
            "speaker_adaptation",
        }:
            return "accurate_adapted"
        if profile in {"・簿ｰ", "・倣剳・・・ｰ・", "・倣剳・・・ｰ・ (・簿ｰ)", "accurate"}:
            return "accurate"
        if profile in {"・ｰ・ｸ", "default", "・倣剳・・・ｰ・ (・ｰ・ｸ)"}:
            return "default"
        return "default"

    def _on_aligner_change(self, _value=None):
        self._sync_aligner_ui()
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
        options = ["MFA", "・・圸(・懦・､)"]
        lang = self._get_language()
        current = str(self.aligner_var.get() if hasattr(self, "aligner_var") else "MFA").strip()
        fmt = normalize_auto_format_value(lang, self.auto_format_var.get()) if hasattr(self, "auto_format_var") else ""
        is_kr_template_only = (lang == "korean" and fmt in {"cmpx", "c_plus_v"})
        forced_no_mfa = bool(lang == "english" or is_kr_template_only)
        if current in {"No-MFA", "No-MFA (Experimental)"}:
            current = "MFA"
        if forced_no_mfa:
            current = "MFA"
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
        use_no_mfa = forced_no_mfa
        use_sequence = current == "・・圸(・懦・､)"
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
            else ["・尖徐(・尖徐 ・ｼ・ｰ甯・", "No-MFA", "v1", "v2", "E2E 﨑們擽・誤ｦｬ・・・､嵭・"]
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
        if no_mfa_mode_code == "crnn":
            no_mfa_mode_code = self._set_no_mfa_oto_mode_from_code("mfa_free_ssl_slot")
        no_mfa_values = [NO_MFA_SSL_SLOT_LABEL, NO_MFA_REMAP_LABEL]
        if hasattr(self, "no_mfa_oto_mode_menu"):
            try:
                self.no_mfa_oto_mode_menu.configure(values=no_mfa_values)
                self.no_mfa_oto_mode_menu.set(
                    NO_MFA_SSL_SLOT_LABEL if no_mfa_mode_code == "mfa_free_ssl_slot" else NO_MFA_REMAP_LABEL
                )
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
        no_mfa_mode_desc = (
            NO_MFA_SSL_SLOT_LABEL
            if no_mfa_mode_code == "mfa_free_ssl_slot"
            else NO_MFA_REMAP_LABEL
        )
        if hasattr(self, "mfa_align_profile_menu"):
            self.mfa_align_profile_menu.configure(
                state="disabled" if (use_no_mfa or use_sequence) else "normal"
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
                    self.aligner_help_label.configure(text=t("(・・牟 Preview CVVC ・ｨ・懍乱・罹株 ・簿ｬ・・・ｬ・ｩ﨑們ｧ ・喜慣・壱共.)"))
                elif lang == "korean" and fmt == "cmpx":
                    self.aligner_help_label.configure(text=t("(CMPX Preview ・ｨ・懍乱・罹株 ・簿ｬ・・・ｬ・ｩ﨑們ｧ ・喜慣・壱共.)"))
                elif is_c_plus_v_mode:
                    self.aligner_help_label.configure(text=t("(﨑懋ｵｭ・ｴ C+V ・ｨ・罹株 奛懦伯・ｿ ・ｰ・・・晧┳・ｼ・・・簿ｬ・・・ｬ・ｩ﨑們ｧ ・喜慣・壱共.)"))
                else:
                    self.aligner_help_label.configure(
                        text=f"(No-MFA ・晧┳ ・ｩ・・ {no_mfa_mode_desc})"
                    )
            elif use_sequence:
                self.aligner_help_label.configure(text=t("(・懦・､ ・ｼ・ｨ ・ｰ・・・・圸 aligner baseline・・・ｬ・ｩ﨑ｩ・壱共.)"))
            else:
                self.aligner_help_label.configure(text=t("(・ｰ・ｸ・ MFA・・笈・､. ・簿ｬ ・・款・・・・･ｴ・ｴ 﨑・囈 ・・・尖徐 ・､・俯姓・壱共.)"))
        if hasattr(self, "pipeline_step_align_btn") and self.pipeline_step_align_btn is not None:
            try:
                if use_no_mfa:
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
                    self.align_step_title_label.configure(text=t("2. ・簿ｬ ・ｨ・・・ｴ・壱怙 (・・牟 Preview)"))
                    self.align_step_desc_label.configure(text=t("・・牟 CVVC Preview ・ｨ・罹株 Lab/・ｬ・・MFA ・・擽 base OTO ・ｩ・晧愍・・・晧┳﨑ｩ・壱共."))
                elif lang == "korean" and fmt == "cmpx":
                    self.align_step_title_label.configure(text=t("2. ・簿ｬ ・ｨ・・・ｴ・壱怙 (CMPX Preview)"))
                    self.align_step_desc_label.configure(text=t("﨑懋ｵｭ・ｴ CMPX Preview ・ｨ・罹株 Lab/・ｬ・・MFA ・・擽 base OTO・ｼ WAV・・・ｬ・､﨑啄紛 ・晧┳﨑ｩ・壱共."))
                elif is_c_plus_v_mode:
                    self.align_step_title_label.configure(text=t("2. ・簿ｬ ・ｨ・・・ｴ・壱怙 (﨑懋ｵｭ・ｴ C+V)"))
                    self.align_step_desc_label.configure(text=t("﨑懋ｵｭ・ｴ C+V ・ｨ・罹株 奛懦伯・ｿ OTO・ｼ WAV・・・ｬ・､﨑啄葺・・・ｩ・晧愍・・・晧┳﨑ｩ・壱共."))
                else:
                    self.align_step_title_label.configure(text=t("2. ・簿ｬ ・ｨ・・・ｴ・壱怙 (No-MFA)"))
                    self.align_step_desc_label.configure(
                        text=t("MFA ・簿ｬ ・・擽 ・・哩﨑ｩ・壱共. ・・ｴ・､ OTO・ｼ WAV・・・ｬ・､﨑啄葺・ ・・・ｰ ・ｼ・ｸ・ ・醐箕 弡・ｳｴ ・川・・・・ｴ・倣鮒・壱共.")
                    )
            else:
                if use_sequence:
                    self.align_step_title_label.configure(text=t("2. ・護┳ ・簿ｬ (・・圸 ・懦・､)"))
                    self.align_step_desc_label.configure(text=t("frame-hop ・懦・､ ・ｼ・ｨ ・ｰ・們愍・・TextGrid・ｼ ・晧┳﨑ｩ・壱共. ・､甯ｨ ・・MFA fallback・・・ｬ・ｩ﨑ｩ・壱共."))
                else:
                    self.align_step_title_label.configure(text=t("2. ・護┳ ・簿ｬ"))
                    self.align_step_desc_label.configure(text=t("MFA・・TextGrid・ｼ ・晧┳﨑ｩ・壱共. MFA・ ・・愍・ｴ ・尖徐 ・､・・弡・・・・ ・・哩﨑ｩ・壱共."))

    def _toggle_developer_mode(self):
        if not hasattr(self, "developer_mode_enabled_var"):
            return
        self.developer_mode_enabled_var.set(not bool(self.developer_mode_enabled_var.get()))
        self._sync_developer_mode_ui()
        self._save_config()

    def _sync_developer_mode_ui(self):
        enabled = bool(self.developer_mode_enabled_var.get()) if hasattr(self, "developer_mode_enabled_var") else False
        if hasattr(self, "dev_mode_btn"):
            self.dev_mode_btn.configure(
                text=t("・罹ｰ懍梵 ・､・・ON") if enabled else "・罹ｰ懍梵 ・､・・OFF",
                fg_color="#7E91AD" if enabled else "#C2CFDF",
                text_color=PALETTE.primary_button_text if enabled else "#2A3A50",
            )
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
        if hasattr(self, "mfa_free_oto_preview_btn") and self.mfa_free_oto_preview_btn is not None:
            try:
                is_running = bool(getattr(self, "is_running", False))
                self.mfa_free_oto_preview_btn.configure(state="normal" if (enabled and not is_running) else "disabled")
            except Exception:
                pass
        if hasattr(self, "mfa_free_oto_preview_hint") and self.mfa_free_oto_preview_hint is not None:
            try:
                self.mfa_free_oto_preview_hint.configure(
                    text_color=PALETTE.hint_text if enabled else "#AEB7C6"
                )
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
            self.ml_coupled_status_label.configure(text=t("嶸・椪 尞ｬ・ｷ: cvvc | ・・牟 Preview ・ｨ・懍乱・罹株 ML ・ｴ・菩揆 ・ｬ・ｩ﨑們ｧ ・喜慣・壱共."))
            if hasattr(self, "ml_coupled_status_detail_label"):
                self.ml_coupled_status_detail_label.configure(text="")
            return
        fmt = ""
        if hasattr(self, "auto_format_var"):
            fmt = normalize_auto_format_value(lang, self.auto_format_var.get())
        if lang == "english":
            self.ml_coupled_status_label.configure(text=t("嶸・椪 尞ｬ・ｷ: cvvc | ・・牟 Preview ・ｨ・懍乱・罹株 ML ・ｴ・菩揆 ・ｬ・ｩ﨑們ｧ ・喜慣・壱共."))
            if hasattr(self, "ml_coupled_status_detail_label"):
                self.ml_coupled_status_detail_label.configure(text="")
            return
        if lang == "korean" and fmt in {"cmpx", "c_plus_v"}:
            mode_label = "cmpx" if fmt == "cmpx" else "c_plus_v"
            self.ml_coupled_status_label.configure(
                text=f"嶸・椪 尞ｬ・ｷ: {mode_label} | 奛懦伯・ｿ ・・圸 ・ｨ・懍乱・罹株 ML ・ｴ・菩揆 ・ｬ・ｩ﨑們ｧ ・喜慣・壱共."
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
            f"嶸・椪 尞ｬ・ｷ: {fmt_display} | ensemble {_status_icon(ensemble_dir)} / "
            f"lightgbm {_status_icon(lightgbm_dir)}"
        )
        if selected:
            text += f" | ・夋・ {selected}"
        if override:
            text += " | ・ｬ・ｩ・・・ｽ・・・ｬ・ｩ"
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
                return f"{label}: (・・搆)"
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
            self.advanced_toggle_btn.configure(text=t("笆ｼ ・・・・ｵ・・(孖ｹ・・・懍搆)"))
            self.advanced_options_frame.pack(fill="x", padx=10, pady=(0, 3))
        else:
            self.advanced_toggle_btn.configure(text=t("笆ｶ ・・・・ｵ・・(孖ｹ・・・懍搆)"))
            self.advanced_options_frame.pack_forget()

    def _get_params(self):
        """嶸・椪 ・ｬ・ｼ・ｴ・・・廷愍・・甯誤攵・ｸ奓ｰ ・菩・・壱ｦｬ ・晧┳"""
        if hasattr(self, "param_vars") and self.param_vars:
            return {key: var.get() for key, var in self.param_vars.items()}
        return None

    # 笏笏 MFA ・､・・(GUI ・ｴ・･) 笏笏



