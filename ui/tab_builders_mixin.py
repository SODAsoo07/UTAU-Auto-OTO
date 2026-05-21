# -*- coding: utf-8 -*-
import os

import customtkinter as ctk

from core.oto_generator import DEFAULT_PARAMS
from ui.theme_tokens import PALETTE, TAB_OVERLAY_THEME
from ui.ui_builders import (
    ui_checkbox,
    ui_help,
    ui_inline_checkbox,
    ui_row,
    ui_section,
)
from ui.ui_declarative import build_declarative
from ui.ui_layout_defs import (
    ADVANCED_ML_SECTION_SUBTITLE,
    ADVANCED_ML_SECTION_TITLE,
    ADVANCED_WEAK_BOUNDARY_HELP,
    ADVANCED_WEAK_BOUNDARY_OPTIONS,
    PIPELINE_MODEL_QUICK_HELP,
)
from ui.i18n import t


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


def _style_primary_button(widget):
    widget.configure(
        fg_color=PALETTE.primary_button_bg,
        hover_color=PALETTE.primary_button_hover,
        text_color=PALETTE.primary_button_text,
    )


class TabBuildersMixin:
    def _get_or_add_tab(self, tab_name: str):
        try:
            return self.tabview.tab(tab_name)
        except Exception:
            return self.tabview.add(tab_name)

    def _tab_overlay_color(self, tab_name: str) -> str:
        return TAB_OVERLAY_THEME.get(tab_name, PALETTE.panel_bg)

    def _bind_wraplength_to_container(self, container, widgets, *, padding=40, min_wrap=240):
        targets = [w for w in (widgets or []) if w is not None]
        if not targets or container is None:
            return

        def _update(_event=None):
            try:
                width = int(container.winfo_width())
            except Exception:
                return
            wrap = max(int(min_wrap), width - int(padding))
            for widget in targets:
                try:
                    widget.configure(wraplength=wrap)
                except Exception:
                    continue

        try:
            container.bind("<Configure>", _update, add="+")
        except Exception:
            pass
        try:
            container.after(0, _update)
        except Exception:
            pass

    def _ui_checkbox(self, *args, **kwargs):
        return ui_checkbox(*args, **kwargs)

    def _ui_help(self, *args, **kwargs):
        return ui_help(*args, **kwargs)

    def _ui_section(self, *args, **kwargs):
        return ui_section(*args, **kwargs)

    def _ui_row(self, *args, **kwargs):
        return ui_row(*args, **kwargs)

    def _ui_inline_checkbox(self, *args, **kwargs):
        return ui_inline_checkbox(*args, **kwargs)

    def _style_primary_button(self, widget):
        _style_primary_button(widget)

    def _style_blue_menu(self, widget):
        _style_blue_menu(widget)

    def _layout_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "ui_layout.json")

    def _build_pipeline_tab(self):
        tab_name = "甯護擽嵓・攵・ｸ"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        root = ctk.CTkFrame(tab, fg_color=overlay)
        root.pack(fill="both", expand=True, padx=5, pady=5)

        content = ctk.CTkScrollableFrame(root, fg_color=overlay)
        content.pack(side="top", fill="both", expand=True, padx=0, pady=0)
        build_declarative(
            self,
            content,
            layout_path=self._layout_path(),
            tab_key="pipeline",
            context={
                "overlay": overlay,
                "root": root,
                "content": content,
                "tab": tab,
            },
        )
        if hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()

    def _pipeline_steps(self):
        return [
            (
                "lab_dict",
                "1. Lab+・ｬ・・・晧┳",
                "WAV 甯護攵・川・ Lab・・・晧┳﨑俾ｳ, ・ｴ・ｴ・・・懍搆 ・ｬ・・甯護攵・護ｧ 﨑・・溢乱 ・晧┳﨑ｩ・壱共.",
                self._run_lab_dict_gen,
            ),
            (
                "align",
                "2. ・護┳ ・簿ｬ",
                "MFA・・TextGrid・ｼ ・晧┳﨑ｩ・壱共. MFA・ ・・愍・ｴ ・尖徐 ・､・・弡・・・・ ・・哩﨑ｩ・壱共.",
                self._run_mfa,
            ),
            (
                "oto",
                "3. OTO.ini ・晧┳",
                "TextGrid ・ｰ・們愍・・OTO 甯誤攵・ｸ奓ｰ・ｼ ・・げ﨑ｴ ・・･﨑ｩ・壱共.",
                self._run_oto_gen,
            ),
        ]

    def _slot_pipeline_action_panel(self, parent, layout_root, _node):
        overlay = layout_root.get("_overlay", PALETTE.panel_bg)
        ctx = layout_root.get("_context", {}) if isinstance(layout_root, dict) else {}
        root = ctx.get("root", parent)

        action_parent = getattr(self, "pipeline_action_host", root)
        if action_parent is root:
            action_panel = ctk.CTkFrame(root, fg_color=overlay)
            action_panel.pack(side="bottom", fill="x", padx=0, pady=(4, 0))
        else:
            for child in action_parent.winfo_children():
                child.destroy()
            action_panel = ctk.CTkFrame(
                action_parent,
                fg_color="transparent",
            )
            action_panel.pack(fill="x", padx=10, pady=8)
        action_panel.grid_columnconfigure(0, weight=1)
        action_panel.grid_columnconfigure(1, weight=1)

        left_actions = ctk.CTkFrame(action_panel, fg_color="transparent")
        left_actions.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=6)

        right_actions = ctk.CTkFrame(action_panel, fg_color="transparent")
        right_actions.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=6)

        ctk.CTkLabel(
            left_actions,
            text=t("Option"),
            font=("", 13, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", pady=(0, 4))

        mfa_btn_row = ctk.CTkFrame(left_actions, fg_color="transparent")
        mfa_btn_row.pack(anchor="w")
        mfa_btn_row.grid_columnconfigure(0, weight=1)
        mfa_btn_row.grid_columnconfigure(1, weight=1)

        self.mfa_repair_btn = ctk.CTkButton(
            mfa_btn_row,
            text=t("剥 MFA ・・卿/・ｵ・ｬ"),
            width=108,
            fg_color="#B0BEC5",
            hover_color="#90A4AE",
            text_color="black",
            command=self._run_mfa_diagnose_repair,
        )
        self.mfa_repair_btn.grid(row=0, column=0, padx=(0, 6), pady=1, sticky="w")

        self.mfa_install_btn = ctk.CTkButton(
            mfa_btn_row,
            text=t("Option"),
            width=108,
            fg_color="#FFA726",
            hover_color="#FB8C00",
            text_color="black",
            command=self._run_mfa_setup,
        )
        mfa_ready = bool(getattr(self, "_mfa_ui_ready", False))
        if getattr(self, "_mfa_install_in_progress", False):
            self.mfa_install_btn.configure(text=t("肌 ・､・・・・.."), state="disabled", fg_color="#B0BEC5")
        elif getattr(self, "_mfa_path_probe_pending", False):
            self.mfa_install_btn.configure(text=t("嶹菩攤 ・・.."), state="disabled", fg_color="#B0BEC5")
        elif mfa_ready:
            self.mfa_install_btn.configure(text=t("Done"), state="disabled", fg_color="#388E3C")
        self.mfa_install_btn.grid(row=0, column=1, padx=(6, 0), pady=1, sticky="w")


        ctk.CTkLabel(
            right_actions,
            text=t("Option"),
            font=("", 13, "bold"),
            text_color=PALETTE.neutral_text,
        ).pack(anchor="w", pady=(0, 4))

        steps_grid = ctk.CTkFrame(right_actions, fg_color="transparent")
        steps_grid.pack(anchor="w")
        steps_grid.grid_columnconfigure(0, weight=1)

        steps = self._pipeline_steps()
        self.pipeline_step_buttons = {}
        self.pipeline_step_align_btn = None
        for idx, (step_key, title, _desc, cmd) in enumerate(steps):
            short = title.split(".", 1)[-1].strip()
            step_btn = ctk.CTkButton(
                steps_grid,
                text=short,
                width=190,
                command=cmd,
            )
            self._style_primary_button(step_btn)
            step_btn.grid(row=idx * 2, column=0, padx=(0, 8), pady=1, sticky="w")
            if idx < len(steps) - 1:
                ctk.CTkLabel(
                    steps_grid,
                    # retired corrupted UI text removed
                    font=("", 14, "bold"),
                    text_color=PALETTE.hint_text,
                    anchor="w",
                ).grid(row=(idx * 2) + 1, column=0, padx=(82, 0), pady=(0, 1), sticky="w")
            self.pipeline_step_buttons[step_key] = step_btn
            if step_key == "align":
                self.pipeline_step_align_btn = step_btn

        if hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()
        if hasattr(self, "_sync_developer_mode_ui"):
            self._sync_developer_mode_ui()

        return action_panel

    def _slot_pipeline_mfa_status(self, parent, _layout_root, _node):
        mfa_inner = ctk.CTkFrame(parent, fg_color="transparent")
        mfa_inner.pack(fill="x", padx=10, pady=8)

        status_group = ctk.CTkFrame(mfa_inner, fg_color="transparent")
        status_group.pack(side="left")

        mfa_ready = bool(getattr(self, "_mfa_ui_ready", False))
        if getattr(self, "_mfa_install_in_progress", False):
            self.mfa_status_label = ctk.CTkLabel(
                status_group, text=t("肌 MFA ・､・・・・.."), font=("", 13, "bold"), text_color="#C27803"
            )
        elif getattr(self, "_mfa_path_probe_pending", False):
            self.mfa_status_label = ctk.CTkLabel(
                status_group, text=t("竢ｳ MFA 嶹菩攤 ・・.."), font=("", 13, "bold"), text_color=PALETTE.hint_text
            )
        elif mfa_ready:
            self.mfa_status_label = ctk.CTkLabel(
                status_group, text=t("笨・MFA ・､・俯勢"), font=("", 13, "bold"), text_color="#4F8F61"
            )
        else:
            self.mfa_status_label = ctk.CTkLabel(
                text=t("Option"),
            )
        self.mfa_status_label.pack(side="left")
        return mfa_inner

    def _slot_pipeline_intro(self, parent, _layout_root, _node):
        intro_label = ctk.CTkLabel(
            parent,
            text=(
                "・們搆 ・､・俯ｘ 'MFA ・・卿/・ｵ・ｬ'・・嶹俾ｲｽ ・ｬ・ｱ・ｼ 嶸・椪 ・ｸ・ｴ ・ｨ・ｸ ・､・ｴ・罹糖 ・誤ｬｸ・・10~20・・・ｴ・・・ｸ・ｴ ・・・溢慣・壱共.\n"
                "・ｸ・懋ｰ ・晝ｸｰ・ｴ ・ｼ・ 'MFA ・・卿/・ｵ・ｬ'・ｼ ・誤洳 ・尖徐 ・ｵ・ｬ・ｼ ・罹巡﨑・・､ ・簿ｬ・・・､・・・､嵂駕葺・ｸ・・"
            ),
            text_color=PALETTE.neutral_text,
            wraplength=760,
            justify="left",
        )
        intro_label.pack(fill="x", padx=20, pady=(0, 8))
        self._bind_wraplength_to_container(parent, [intro_label], padding=48, min_wrap=280)
        return intro_label

    def _slot_pipeline_steps(self, parent, layout_root, _node):
        overlay = layout_root.get("_overlay", PALETTE.panel_bg)
        for step_key, title, desc, _cmd in self._pipeline_steps():
            frame = ctk.CTkFrame(parent, fg_color=overlay, border_width=1, border_color=PALETTE.panel_border)
            frame.pack(fill="x", padx=10, pady=4)
            left = ctk.CTkFrame(frame, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True, padx=10, pady=6)
            title_label = ctk.CTkLabel(left, text=title, font=("", 14, "bold"), anchor="w")
            title_label.pack(anchor="w")
            desc_label = ctk.CTkLabel(left, text=desc, text_color=PALETTE.neutral_text, anchor="w", wraplength=500)
            desc_label.pack(anchor="w")
            self._bind_wraplength_to_container(frame, [desc_label], padding=300, min_wrap=220)
            if step_key == "align":
                self.align_step_title_label = title_label
                self.align_step_desc_label = desc_label

            if "OTO" in title:
                opt_frame = ctk.CTkFrame(left, fg_color="transparent")
                opt_frame.pack(fill="x", pady=(5, 0))
                ctk.CTkCheckBox(
                    opt_frame,
                    text=t("OpenUtau 嶸ｸ嶹・・・巡 ・川攵・ｬ・ｴ・､ ・尖徐 ・晧┳"),
                    text_color="#5E7E95",
                    variable=self.openutau_var,
                    command=self._save_config,
                ).pack(anchor="w")

                self.gen_missing_vowels_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text=t("・・攷・・・ｨ・・・ｨ・護龍(VV) ・川攵・ｬ・ｴ・､ ・ｴ・・・晧┳"),
                    text_color="#64866F",
                    variable=self.gen_missing_vowels_var,
                    command=self._save_config,
                )
                self.gen_missing_vowels_checkbox.pack(anchor="w", pady=(5, 0))
                self.gen_dash_alias_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text=t("・ｴ・・・ｴ・ｸ '-' ・川攵・ｬ・ｴ・､ ・晧┳"),
                    text_color=PALETTE.neutral_text,
                    variable=self.gen_dash_alias_var,
                    command=self._save_config,
                )
                self.gen_dash_alias_checkbox.pack(anchor="w", pady=(5, 0))

                ctk.CTkLabel(
                    opt_frame,
                    text=t("ML ・ｴ・菩捩 ・尖徐・ｼ・・・・圸・俯ｩｰ ・ｨ・ｸ・ｴ ・・愍・ｴ ・尖徐・ｼ・・・ｴ・壱怐・壱共."),
                    text_color=PALETTE.hint_text,
                ).pack(anchor="w", pady=(6, 0))
        return None

    def _build_params_tab(self):
        tab_name = "Parameters"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        scroll = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        summary = ctk.CTkLabel(
            scroll,
            text=t("甯誤攵・ｸ奓ｰ ・ｰ・・夋ｭ・川・ OTO ・ｴ・・・・・・ｼ ・ｰ・倣腹 ・・・溢慣・壱共. ・ｰ・ｸ・・・ｬ・ｩ・・・護棗﨑ｩ・壱共."),
            text_color=PALETTE.neutral_text,
            justify="left",
            wraplength=760,
        )
        summary.pack(fill="x", padx=10, pady=(8, 8))
        self._bind_wraplength_to_container(scroll, [summary], padding=40, min_wrap=280)

        self.param_vars = {}
        param_groups = [
            (
                "・ｰ・ｸ VC 甯誤攵・ｸ奓ｰ",
                [
                    ("VC_CONSONANT_RATIO", "VC ・川搆 ・ｬ・・・・惠", 0.1, 1.0, 0.05),
                    ("VC_VOWEL_START", "VC ・ｨ・・・懍梠 ・・惠", 0.1, 1.0, 0.05),
                    ("VC_PRE_OFFSET", "VC ・嵂雅ｰ懍搆 ・､嵓・・ (ms)", 0, 50, 1),
                    ("VC_OVL_RATIO", "VC ・､・・棠 ・・惠", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "・ｰ・ｸ CV 甯誤攵・ｸ奓ｰ",
                [
                    ("CV_PRE_RATIO", "CV ・嵂雅ｰ懍搆 ・・惠", 0.1, 1.0, 0.05),
                    ("CV_OVL_RATIO", "CV ・､・・棠 ・・惠", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "・ｴ・瀧ｪｨ・・CV 甯誤攵・ｸ奓ｰ",
                [
                    ("DIPHTHONG_CV_PRE_RATIO", "・ｴ・瀧ｪｨ・・CV ・嵂雅ｰ懍搆 ・・惠", 0.1, 1.0, 0.05),
                    ("DIPHTHONG_CV_CONSONANT_RATIO", "・ｴ・瀧ｪｨ・・CV ・川搆 ・・惠", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "・ｴ・瀧ｪｨ・・VC 甯誤攵・ｸ奓ｰ",
                [
                    ("DIPHTHONG_VC_VOWEL_START", "・ｴ・瀧ｪｨ・・VC ・ｨ・・・懍梠", 0.1, 1.0, 0.05),
                    ("DIPHTHONG_VC_CONSONANT", "・ｴ・瀧ｪｨ・・VC ・川搆 ・・惠", 0.1, 1.0, 0.05),
                ],
            ),
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
                slider = ctk.CTkSlider(
                    row,
                    from_=min_val,
                    to=max_val,
                    number_of_steps=int((max_val - min_val) / step),
                    variable=var,
                    command=lambda v, lbl=val_label: lbl.configure(text=f"{v:.2f}"),
                )
                slider.pack(side="right", fill="x", expand=True, padx=5)

    def _build_log_tab(self):
        tab_name = "・懋ｷｸ"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        self.log_text = ctk.CTkTextbox(
            tab,
            font=("Consolas", 12),
            state="normal",
            fg_color=PALETTE.input_bg,
            text_color=PALETTE.neutral_text,
            border_color=PALETTE.panel_border,
            border_width=1,
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        if hasattr(self, "_flush_ui_log_buffer"):
            self._flush_ui_log_buffer()

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)
        clear_btn = ctk.CTkButton(btn_frame, text=t("・懋ｷｸ ・・ｰ・ｰ"), width=100, command=self._clear_log)
        _style_primary_button(clear_btn)
        clear_btn.pack(side="left")
        open_btn = ctk.CTkButton(
            btn_frame,
            text=t("・懋ｷｸ 甯護攵 ・ｴ・ｰ"),
            width=120,
            command=lambda: os.startfile(self.log_path) if os.path.exists(self.log_path) else None,
        )
        _style_primary_button(open_btn)
        open_btn.pack(side="left", padx=5)

    def _build_detail_log_tab(self):
        tab_name = "・・┷ ・懋ｷｸ"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)

        desc = ctk.CTkLabel(
            tab,
            text=t("CMD/・罹ｸ醐売・懍┷・､ ・尖ｳｸ ・懋ｷｸ・ｼ ・懋ｰ・・・ｼ・・岺懍亨﨑ｩ・壱共. (・・﨑・・・詐 嵓誤洳・・"),
            text_color=PALETTE.neutral_text,
            anchor="w",
        )
        desc.pack(fill="x", padx=8, pady=(6, 0))

        self.detail_log_text = ctk.CTkTextbox(
            tab,
            font=("Consolas", 11),
            state="normal",
            fg_color=PALETTE.input_bg,
            text_color=PALETTE.neutral_text,
            border_color=PALETTE.panel_border,
            border_width=1,
        )
        self.detail_log_text.pack(fill="both", expand=True, padx=5, pady=5)
        if hasattr(self, "_flush_detail_log_buffer"):
            self._flush_detail_log_buffer()

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)
        clear_btn = ctk.CTkButton(
            btn_frame,
            text=t("・・┷ ・懋ｷｸ ・・ｰ・ｰ"),
            width=120,
            command=lambda: self.detail_log_text.delete("1.0", "end"),
        )
        _style_primary_button(clear_btn)
        clear_btn.pack(side="left")
        open_btn = ctk.CTkButton(
            btn_frame,
            text=t("・懋ｷｸ 甯護攵 ・ｴ・ｰ"),
            width=120,
            command=lambda: os.startfile(self.log_path) if os.path.exists(self.log_path) else None,
        )
        _style_primary_button(open_btn)
        open_btn.pack(side="left", padx=5)

    def _build_credits_tab(self):
        tab_name = "增ｬ・壱肌"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)

        scroll = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            scroll,
            text=t("-・ｴ・・・ｨ・ｸ ・ｰ・ｴ奓ｰ ・懋ｳｵ・・・・它 ・ｼ・ ・・豆"),
            font=("", 15, "bold"),
            text_color=PALETTE.header_accent,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(6, 10))

        credit_lines = [
            "・ｴ・罷ｹ・Kamebit",
            "22 twotwosibi",
            "・呰ム・ｬ/・ｵ・・攪 ・護┳・懋ｳｵ・・infinityecho00&Anonymous Voice Provider",
            "・壱ｹ・2xbin",
            "・們亨・・Lyusia",
            "HUEY",
            "・瀧Η Ironic_SP",
            "・ｼ・ｬ・ｰ RARIU",
            "・罹舶嶸ｸ neohyajach",
            "・俯ｯｸ・・Namini",
            "・ｸ・・ｸ Nowano",
            "・､・一ｰ・AngaeSil1115",
            "・ｬ奝 ・ｼ・俾ｸｰ Sato Yanagi",
            "嶸懍┳ Comet",
            "・・ざ BUTCHER_TUNING",
            "寬懍ｴ・ianharuesaone",
            "・､・ DAU_Multiverse",
            "・瀧Η Blue",
        ]

        for line in credit_lines:
            ctk.CTkLabel(
                scroll,
                text=line,
                text_color=PALETTE.neutral_text,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=1)

        ctk.CTkLabel(
            scroll,
            text=t("・川ぎ﨑ｩ・壱共!"),
            font=("", 14, "bold"),
            text_color=PALETTE.success_text,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(12, 6))

    def _build_advanced_settings_tab_legacy(self):
        tab_name = "Parameters"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        container = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        self._ensure_advanced_setting_vars()

        adv_intro_label = ctk.CTkLabel(
            container,
            text=t("・・・・､・菩捩 ・罹ｰ・孖罹享・ｩ 﨑ｭ・ｩ・・笈・､. ・､・ｬ・ｩ・川・・・・ｰ・ｸ・・・・ 弡・﨑・囈﨑・・ｽ・ｰ・・・ｰ・倣葺・ｸ・・"),
            text_color=PALETTE.neutral_text,
            wraplength=760,
            justify="left",
        )
        adv_intro_label.pack(fill="x", padx=10, pady=(8, 12))
        self._bind_wraplength_to_container(container, [adv_intro_label], padding=40, min_wrap=280)
        self.advanced_tuning_slider_bindings = {}

        basic_toggle_frame = ctk.CTkFrame(
            container,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        basic_toggle_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.advanced_basic_frame = basic_toggle_frame
        ctk.CTkLabel(
            basic_toggle_frame,
            text=t("Option"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("ML ・ｴ・・・ｬ・ｩ"),
            text_color="#9A8250",
            variable=self.enable_ml_correction_var,
            command=(
                self._on_enable_ml_correction_toggle
                if hasattr(self, "_on_enable_ml_correction_toggle")
                else self._save_config
            ),
        )
        self.enable_ml_correction_checkbox.pack(anchor="w", padx=12, pady=(0, 4))

        self.kr_continuity_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・ｰ・作┳/甯護攵 ・ｼ・・ｱ ・ｴ・・・ｬ・ｩ"),
            text_color="#64866F",
            variable=self.vc_correction_enable_var,
            command=self._on_vc_correction_toggle,
        )
        self.kr_continuity_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        continuity_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("嶸菩享 ・・ｽ ・・・肥ｲ懋ｰ廷擽 ・尖徐 ・・圸・ｩ・壱共. 﨑・囈﨑・・ｽ・ｰ ・・椈 ・罹ｰ懍梵 ・ｬ・ｼ・ｴ・肥乱・・・ｸ・・庭ｧ・・ｸ・ｸ ・ｰ・倣葺・ｸ・・"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        continuity_help_label.pack(anchor="w", padx=34, pady=(0, 6))
        self._bind_wraplength_to_container(basic_toggle_frame, [continuity_help_label], padding=64, min_wrap=260)

        post_frame = ctk.CTkFrame(
            container,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        post_frame.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(
            post_frame,
            text=t("Option"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cont_row = ctk.CTkFrame(post_frame, fg_color="transparent")
        cont_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            cont_row,
            text=t("・ｰ・作┳ offset ・ｴ・・・・復 (ms)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        cont_entry = ctk.CTkEntry(
            cont_row,
            width=90,
            textvariable=self.kr_continuity_max_offset_adj_var,
            placeholder_text=t("・ｰ・ｸ・・180)"),
        )
        cont_entry.pack(side="left", padx=(10, 8))
        cont_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cont_row,
            text=t("增ｬ・・・｡・・・・・・ｰ・作┳ ・ｴ・菩擽 ・倣紛・・, ・滝ｲ・・｡・・・・・・ｴ・菩擽 ・ｽ﨑ｴ・瀧笈・､."),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        bank_preset_frame = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        bank_preset_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(
            bank_preset_frame,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        bank_preset_help_label = ctk.CTkLabel(
            bank_preset_frame,
            text=t("soft/・ｼ・・・・〓・ｩ 嵓・ｦｬ・駆揆 﨑・・溢乱 ・・圸﨑ｩ・壱共. ・ｼ・戦・・ ・ｴ・菩揆 ・倣葺・・ ・ｼ・們捩 ・溢菩・愍・・・呷梠﨑ｩ・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=730,
            justify="left",
        )
        bank_preset_help_label.pack(anchor="w", pady=(0, 6))
        self._bind_wraplength_to_container(bank_preset_frame, [bank_preset_help_label], padding=30, min_wrap=240)

        preset_grid = ctk.CTkFrame(bank_preset_frame, fg_color="transparent")
        preset_grid.pack(anchor="w")
        preset_grid.grid_columnconfigure(0, weight=1)
        preset_grid.grid_columnconfigure(1, weight=1)

        self.voicebank_preset_buttons = {}

        btn_soft_agg = ctk.CTkButton(
            preset_grid,
            text=t("soft ・・〓 ﾂｷ ・ｼ・戦・"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "aggressive"),
        )
        _style_primary_button(btn_soft_agg)
        btn_soft_agg.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_soft_cons = ctk.CTkButton(
            preset_grid,
            text=t("Option"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "conservative"),
        )
        _style_primary_button(btn_soft_cons)
        btn_soft_cons.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="w")

        btn_normal_agg = ctk.CTkButton(
            preset_grid,
            text=t("・ｼ・・・・〓 ﾂｷ ・ｼ・戦・"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "aggressive"),
        )
        _style_primary_button(btn_normal_agg)
        btn_normal_agg.grid(row=1, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_normal_cons = ctk.CTkButton(
            preset_grid,
            text=t("Option"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "conservative"),
        )
        _style_primary_button(btn_normal_cons)
        btn_normal_cons.grid(row=1, column=1, padx=(0, 8), pady=2, sticky="w")

        self.voicebank_preset_buttons = {
            "soft:aggressive": btn_soft_agg,
            "soft:conservative": btn_soft_cons,
            "normal:aggressive": btn_normal_agg,
            "normal:conservative": btn_normal_cons,
        }
        if hasattr(self, "_sync_voicebank_preset_button_styles"):
            self._sync_voicebank_preset_button_styles()

        self.soft_bank_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("Option"),
            text_color="#6F819A",
            variable=self.soft_bank_mode_var,
            command=self._save_config,
        )
        self.soft_bank_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        soft_bank_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・ｨ・誤ｦｬ・ ・珠擽 ・樌擽・ｰ・・・懍┳・ｴ ・ｽ﨑・・護寳・川・ ・懍搆 ・・攷・ｼ ・ｵ・ｱ ・､・・懍揆 ・・攵 ・・・罹株 ・ｵ・們桿・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        soft_bank_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [soft_bank_help_label], padding=64, min_wrap=260)

        self.low_rms_gain_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・・ｼ・ｨ WAV ・尖徐 ・晨少/・母ｷ懦剩 (RMS ・ｰ・・ ・尖ｳｸ ・ｴ・ｴ)"),
            text_color="#6F819A",
            variable=self.low_rms_gain_enable_var,
            command=self._save_config,
        )
        self.low_rms_gain_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_rms_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・簿ｬ ・ｨ・・乱・罹ｧ・・・亨 ・卓羅 ・ｵ・ｬ・ｸ・・・ｼ・ｨ ・ｴ・菩揆 ・・圸﨑ｩ・壱共. ・尖ｳｸ WAV 甯護攵・ ・壱劇 ・・ｽ﨑們ｧ ・喜慣・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        low_rms_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [low_rms_help_label], padding=64, min_wrap=260)

        self.weak_voice_assist_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・懍搆・ｴ 彧尖ｦｰ ・護寳・・・肥ｲ・(・尖ｪｨ ・ｬ・・・倣剩)"),
            text_color="#6F819A",
            variable=self.weak_voice_assist_enable_var,
            command=(
                self._on_weak_voice_assist_toggle
                if hasattr(self, "_on_weak_voice_assist_toggle")
                else self._save_config
            ),
        )
        self.weak_voice_assist_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        weak_voice_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・・名・ｰ・ｼ・俯涵 ・､・ｬ・ｰ・・・川搆/・ｨ・・・ｬ・・擽 彧尖ｦｰ ・護寳・川・ ・簿ｬ ・倣剳・・･ｼ ・廷擽・ｰ ・・復 ・肥ｲ・・ｵ・們桿・壱共. (・尖ｳｸ WAV ・ｴ・ｴ)"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        weak_voice_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [weak_voice_help_label], padding=64, min_wrap=260)

        weak_voice_strength_row = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        weak_voice_strength_row.pack(fill="x", padx=34, pady=(0, 10))
        self.weak_voice_strength_title_label = ctk.CTkLabel(
            weak_voice_strength_row,
            text=t("・尖ｪｨ ・・・・簿巡"),
            width=130,
            anchor="w",
            text_color=PALETTE.neutral_text,
        )
        self.weak_voice_strength_title_label.pack(side="left")
        try:
            _raw_strength = (
                str(self.weak_voice_assist_strength_var.get() or "")
                if hasattr(self, "weak_voice_assist_strength_var")
                else ""
            ).strip()
            _strength_current = float(_raw_strength) if _raw_strength else 0.35
        except Exception:
            _strength_current = 0.35
        _strength_current = max(0.0, min(1.0, _strength_current))
        weak_voice_strength_dvar = ctk.DoubleVar(value=_strength_current)
        weak_voice_strength_value_label = ctk.CTkLabel(
            weak_voice_strength_row,
            text=f"{_strength_current:.2f}",
            width=46,
        )
        weak_voice_strength_value_label.pack(side="right", padx=(6, 0))
        self.weak_voice_strength_value_label = weak_voice_strength_value_label

        def _on_weak_voice_strength_change(value):
            try:
                val = max(0.0, min(1.0, float(value)))
            except Exception:
                return
            weak_voice_strength_value_label.configure(text=f"{val:.2f}")
            if hasattr(self, "weak_voice_assist_strength_var"):
                self.weak_voice_assist_strength_var.set(f"{val:.3f}".rstrip("0").rstrip("."))
            self._save_config()

        weak_voice_strength_slider = ctk.CTkSlider(
            weak_voice_strength_row,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=weak_voice_strength_dvar,
            command=_on_weak_voice_strength_change,
        )
        weak_voice_strength_slider.pack(side="right", fill="x", expand=True, padx=8)
        self.weak_voice_strength_slider = weak_voice_strength_slider
        self.advanced_tuning_slider_bindings["UTOA_MFA_WEAK_VOICE_PREEMPH_MIX"] = {
            "var": getattr(self, "weak_voice_assist_strength_var", None),
            "dvar": weak_voice_strength_dvar,
            "label": weak_voice_strength_value_label,
            "fmt": "{:.2f}",
            "default": 0.35,
            "min": 0.0,
            "max": 1.0,
        }
        if hasattr(self, "_sync_weak_voice_assist_controls"):
            self._sync_weak_voice_assist_controls()

        mapping_strict_row = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        mapping_strict_row.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            mapping_strict_row,
            text=t("・護・・､・､﨑・・ｨ・ｨ"),
            text_color=PALETTE.neutral_text,
            width=130,
            anchor="w",
        ).pack(side="left")
        mapping_strict_menu = ctk.CTkOptionMenu(
            mapping_strict_row,
            values=(
                self._get_mapping_strict_mode_option_labels()
                if hasattr(self, "_get_mapping_strict_mode_option_labels")
                else ["・・・・・ｲｩ", "・・胸德・・・ｲｩ(・・攷 嵂餓捩 尞ｴ・ｱ)", "off"]
            ),
            variable=self.mapping_strict_mode_var,
            width=260,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(mapping_strict_menu)
        mapping_strict_menu.pack(side="left", padx=(10, 0))
        mapping_strict_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・護溢揆 寀・ｴ ・・･・ｱ・・・・擽・・・・, ・尖徐 ・､・菩乱・・・､墲ｵ・俯株 ・川攵・ｬ・ｴ・､・ ・們牟・ ・・・溢慣・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        mapping_strict_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self.low_conf_force_lock_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
            variable=self.low_conf_force_lock_mode_var,
            command=self._save_config,
        )
        self.low_conf_force_lock_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_conf_force_lock_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・簿ｬ ・・ｰ・ ・ｮ・ ・ｬ・・乱・罹株 弡・ｳｴ ・戦売・ｼ ・賀ｳ ・溢メ ・・ｹ們乱 ・・倣鮒・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        low_conf_force_lock_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(
            basic_toggle_frame,
            [mapping_strict_help_label, low_conf_force_lock_help_label],
            padding=64,
            min_wrap=260,
        )

        weak_boundary_row = self._ui_row(basic_toggle_frame, padx=12, pady=(0, 2))
        for option in ADVANCED_WEAK_BOUNDARY_OPTIONS:
            var = getattr(self, option["var"])
            self._ui_inline_checkbox(
                weak_boundary_row,
                text=option["text"],
                variable=var,
                command=self._save_config,
                padx=option.get("padx", (0, 0)),
            )
        weak_boundary_help = self._ui_help(basic_toggle_frame, ADVANCED_WEAK_BOUNDARY_HELP)
        self._bind_wraplength_to_container(basic_toggle_frame, [weak_boundary_help], padding=64, min_wrap=260)

        dev_container = ctk.CTkFrame(container, fg_color="transparent")
        dev_container.pack(fill="x", padx=0, pady=(0, 0))
        self.advanced_developer_frame = dev_container

        ml_frame = self._ui_section(
            dev_container,
            ADVANCED_ML_SECTION_TITLE,
            ADVANCED_ML_SECTION_SUBTITLE,
        )
        self.advanced_ml_section_frame = ml_frame
        if not hasattr(self, "mapping_supervised_enable_var"):
            self.mapping_supervised_enable_var = ctk.BooleanVar(value=True)

        self.mapping_supervised_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("・､﨑・・ｨ・ｸ(・ｨ・ｰ ・肥ｽ罷畠 ・ｬ・､・肥牟) ・ｬ・ｩ"),
            text_color=PALETTE.neutral_text,
            variable=self.mapping_supervised_enable_var,
            command=(
                self._on_mapping_supervised_toggle
                if hasattr(self, "_on_mapping_supervised_toggle")
                else self._save_config
            ),
        )
        self.mapping_supervised_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))

        mapping_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        mapping_mode_row.pack(anchor="w", padx=12, pady=(0, 4), fill="x")
        ctk.CTkLabel(
            mapping_mode_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        mapping_mode_menu = ctk.CTkOptionMenu(
            mapping_mode_row,
            values=(
                self._get_mapping_supervised_mode_option_labels()
                if hasattr(self, "_get_mapping_supervised_mode_option_labels")
                else ["・尖徐(・護棗)", "尞ｬ・ｷ ・ｰ・", "・・溜 ・ｰ・"]
            ),
            variable=self.mapping_supervised_mode_var,
            width=220,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(mapping_mode_menu)
        mapping_mode_menu.pack(side="left", padx=(10, 0))
        if not hasattr(self, "cv_order_prior_enable_var"):
            self.cv_order_prior_enable_var = ctk.BooleanVar(value=True)
        if not hasattr(self, "cv_order_prior_strength_var"):
            self.cv_order_prior_strength_var = ctk.StringVar(value="")

        cv_order_prior_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        cv_order_prior_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        self.cv_order_prior_enable_checkbox = ctk.CTkCheckBox(
            cv_order_prior_row,
            text=t("CV 甯護攵 ・懍・ ・ｰ・・・､﨑・・ｰ﨑ｩ ・ｬ・ｩ"),
            text_color=PALETTE.neutral_text,
            variable=self.cv_order_prior_enable_var,
            command=self._save_config,
        )
        self.cv_order_prior_enable_checkbox.pack(side="left")
        ctk.CTkLabel(
            cv_order_prior_row,
            text=t("・簿巡"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(12, 4))
        cv_order_prior_strength_entry = ctk.CTkEntry(
            cv_order_prior_row,
            width=72,
            textvariable=self.cv_order_prior_strength_var,
            placeholder_text=t("auto"),
        )
        cv_order_prior_strength_entry.pack(side="left")
        cv_order_prior_strength_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cv_order_prior_row,
            text=t("(0.0~1.0, ・廷揆・俯｡・・懍・ prior ・倣剩)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(8, 0))
        self.mapping_supervised_dependent_frames = [mapping_mode_row, cv_order_prior_row]

        selector_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        selector_mode_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            selector_mode_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        self.ml_selector_mode_segment = ctk.CTkSegmentedButton(
            selector_mode_row,
            values=["auto", "additive"],
            variable=self.ml_selector_mode_var,
            command=lambda _value: self._save_config(),
        )
        self.ml_selector_mode_segment.pack(side="left", padx=(10, 0))

        route_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        route_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            route_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        route_menu = ctk.CTkOptionMenu(
            route_row,
            values=(
                self._get_ml_route_option_labels(no_mfa_only=False)
                if hasattr(self, "_get_ml_route_option_labels")
                else ["・尖徐(・尖徐 ・ｼ・ｰ甯・", "No-MFA", "v1", "v2", "E2E 﨑們擽・誤ｦｬ・・・､嵭・"]
            ),
            variable=self.ml_route_var,
            width=180,
            command=self._on_ml_route_change if hasattr(self, "_on_ml_route_change") else (lambda _v: self._save_config()),
        )
        self.ml_route_menu = route_menu
        _style_blue_menu(route_menu)
        route_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            route_row,
            text=t("Option"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        coupled_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("・・OTO ・ｴ・・・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_coupled_enable_var,
            command=self._save_config,
        )
        coupled_enable_checkbox.pack(anchor="w", padx=12, pady=(8, 0))

        coupled_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        coupled_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            coupled_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        coupled_conf_entry = ctk.CTkEntry(
            coupled_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_var,
            placeholder_text=t("auto"),
        )
        coupled_conf_entry.pack(side="left", padx=(10, 8))
        coupled_conf_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            coupled_row,
            text="device",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(6, 4))
        coupled_device_menu = ctk.CTkOptionMenu(
            coupled_row,
            values=["auto", "cpu", "cuda"],
            variable=self.ml_coupled_device_var,
            width=90,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(coupled_device_menu)
        coupled_device_menu.pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(
            coupled_row,
            text=t("Strict ・懍平"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_strict_constraint_var,
            command=self._save_config,
        ).pack(side="left")

        model_conf_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkCheckBox(
            model_conf_row,
            text=t("Model-aware min_conf (model_meta) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_coupled_min_conf_use_model_meta_var,
            command=self._save_config,
        ).pack(side="left")
        ctk.CTkLabel(
            model_conf_row,
            text="offset",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(10, 4))
        model_conf_offset_entry = ctk.CTkEntry(
            model_conf_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_model_offset_var,
            placeholder_text="0.00",
        )
        model_conf_offset_entry.pack(side="left", padx=(0, 8))
        model_conf_offset_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            model_conf_row,
            text="(range: -0.30 ~ +0.30)",
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        model_conf_grid = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_grid.pack(anchor="w", padx=12, pady=(4, 0), fill="x")

        def _add_min_conf_field(parent, label, var, placeholder):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(side="left", padx=(0, 10))
            ctk.CTkLabel(row, text=label, text_color=PALETTE.neutral_text).pack(side="left")
            ent = ctk.CTkEntry(row, width=78, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", padx=(6, 0))
            ent.bind("<FocusOut>", lambda _e: self._save_config())
            return ent

        kr_row = ctk.CTkFrame(model_conf_grid, fg_color="transparent")
        kr_row.pack(anchor="w", fill="x", pady=(0, 2))
        ctk.CTkLabel(kr_row, text="KR", text_color="#5F8C87", width=26, anchor="w").pack(side="left")
        _add_min_conf_field(kr_row, "CV", self.ml_coupled_min_conf_kr_cv_var, "0.55")
        _add_min_conf_field(kr_row, "CVC", self.ml_coupled_min_conf_kr_cvc_var, "0.75")
        _add_min_conf_field(kr_row, "CVVC", self.ml_coupled_min_conf_kr_cvvc_var, "0.78")
        _add_min_conf_field(kr_row, "VCV", self.ml_coupled_min_conf_kr_vcv_var, "0.72")

        ja_row = ctk.CTkFrame(model_conf_grid, fg_color="transparent")
        ja_row.pack(anchor="w", fill="x", pady=(0, 2))
        ctk.CTkLabel(ja_row, text="JA", text_color="#5F8C87", width=26, anchor="w").pack(side="left")
        _add_min_conf_field(ja_row, "CV", self.ml_coupled_min_conf_ja_cv_var, "0.40")
        _add_min_conf_field(ja_row, "CVC", self.ml_coupled_min_conf_ja_cvc_var, "0.50")
        _add_min_conf_field(ja_row, "CVVC", self.ml_coupled_min_conf_ja_cvvc_var, "0.72")
        _add_min_conf_field(ja_row, "VCV", self.ml_coupled_min_conf_ja_vcv_var, "0.65")

        backend_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        backend_row.pack(anchor="w", padx=12, pady=(6, 8), fill="x")
        ctk.CTkLabel(
            backend_row,
            text="Coupled backend",
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        backend_menu = ctk.CTkOptionMenu(
            backend_row,
            values=["auto", "ensemble"],
            variable=self.ml_coupled_backend_var,
            width=90,
            command=lambda _v: self._on_ml_backend_change(),
        )
        _style_blue_menu(backend_menu)
        backend_menu.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            backend_row,
            text=t("(auto=ensemble ・ｰ・)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(8, 0))

        batch_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Batch inference (coupled v2) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_batch_inference_enable_var,
            command=self._save_config,
        )
        batch_enable_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        batch_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        batch_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            batch_row,
            text="Batch size",
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        batch_size_entry = ctk.CTkEntry(
            batch_row,
            width=90,
            textvariable=self.ml_batch_inference_size_var,
            placeholder_text="256",
        )
        batch_size_entry.pack(side="left", padx=(10, 8))
        batch_size_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            batch_row,
            text=t("(・護棗 128~512, ・懍・ 32)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        legacy_fallback_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("v1 fallback(lightgbm) ・ｴ・ｸ ・ｬ・ｩ"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_legacy_fallback_enable_var,
            command=self._save_config,
        )
        legacy_fallback_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        hybrid_routing_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("﨑們擽・誤ｦｬ・・・ｼ・ｰ甯・・・卓ｹ・・護擽孖ｸ) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_hybrid_routing_enable_var,
            command=self._save_config,
        )
        hybrid_routing_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_toggle_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Option"),
            text_color="#5F8C87",
            variable=self.ml_e2e_enable_var,
            command=(
                self._on_ml_e2e_toggle
                if hasattr(self, "_on_ml_e2e_toggle")
                else self._save_config
            ),
        )
        e2e_toggle_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        e2e_mode_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        ctk.CTkLabel(
            e2e_mode_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        e2e_mode_menu = ctk.CTkOptionMenu(
            e2e_mode_row,
            values=["hybrid", "legacy_only", "e2e_only"],
            variable=self.ml_e2e_mode_var,
            width=120,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(e2e_mode_menu)
        e2e_mode_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            e2e_mode_row,
            text=t("(・護棗: hybrid)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        e2e_threshold_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        e2e_threshold_row.pack(anchor="w", padx=12, pady=(2, 4), fill="x")
        ctk.CTkLabel(e2e_threshold_row, text="T_low", text_color=PALETTE.neutral_text).pack(side="left")
        e2e_t_low_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_t_low_var,
            placeholder_text="0.52",
        )
        e2e_t_low_entry.pack(side="left", padx=(6, 8))
        e2e_t_low_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(e2e_threshold_row, text="T_high", text_color=PALETTE.neutral_text).pack(side="left", padx=(2, 0))
        e2e_t_high_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_t_high_var,
            placeholder_text="0.72",
        )
        e2e_t_high_entry.pack(side="left", padx=(6, 8))
        e2e_t_high_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(e2e_threshold_row, text="blend", text_color=PALETTE.neutral_text).pack(side="left", padx=(2, 0))
        e2e_blend_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_blend_alpha_var,
            placeholder_text="0.60",
        )
        e2e_blend_entry.pack(side="left", padx=(6, 8))
        e2e_blend_entry.bind("<FocusOut>", lambda _e: self._save_config())
        self.ml_e2e_dependent_frames = [e2e_mode_row, e2e_threshold_row]

        gamma_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        gamma_row.pack(anchor="w", padx=12, pady=(2, 6), fill="x")
        ctk.CTkLabel(
            gamma_row,
            text="Anchor mel gamma",
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        gamma_entry = ctk.CTkEntry(
            gamma_row,
            width=80,
            textvariable=self.ml_anchor_mel_gamma_var,
            placeholder_text="1.0",
        )
        gamma_entry.pack(side="left", padx=(10, 8))
        gamma_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            gamma_row,
            text="(>1.0: stronger shrink on low mel reliability)",
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        detail_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        detail_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        if not hasattr(self, "ml_coupled_status_detail_var"):
            self.ml_coupled_status_detail_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            detail_row,
            text=t("・川┷德・・ｴ・ｰ (・ｽ・・・・・・晧┳・ｼ)"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_status_detail_var,
            command=lambda: self._on_ml_backend_detail_toggle(),
        ).pack(side="left")

        def _model_root_row(label, var):
            row = ctk.CTkFrame(ml_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(6, 0))
            ctk.CTkLabel(row, text=label, width=150, anchor="w", text_color=PALETTE.neutral_text).pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=var, width=360)
            ent.pack(side="left", fill="x", expand=True, padx=(5, 6))
            ent.bind("<FocusOut>", lambda _e: self._on_ml_model_root_change())
            btn_row = ctk.CTkFrame(row, fg_color="transparent")
            btn_row.pack(side="right", padx=(6, 0))
            btn_width = 90
            browse_btn = ctk.CTkButton(
                btn_row,
                text=t("・ｾ・・ｳｴ・ｰ"),
                width=btn_width,
                command=lambda v=var: (
                    self._browse_folder_by_var(v, initial_dir=self._preferred_ml_model_browse_dir(v)),
                    self._on_ml_model_root_change(),
                ),
            )
            _style_primary_button(browse_btn)
            browse_btn.pack(side="right")
            open_btn = ctk.CTkButton(
                btn_row,
                text=t("・ｴ・ｰ"),
                width=btn_width,
                command=lambda v=var: os.startfile(str(v.get()).strip()) if os.path.isdir(str(v.get()).strip()) else None,
            )
            _style_primary_button(open_btn)
            open_btn.pack(side="right", padx=(0, 6))
            return ent

        if hasattr(self, "ml_model_root_kr_var"):
            _model_root_row("・ｨ・ｸ ・ｽ・・(﨑懋ｵｭ・ｴ)", self.ml_model_root_kr_var)
        if hasattr(self, "ml_model_root_ja_var"):
            _model_root_row("・ｨ・ｸ ・ｽ・・(・ｼ・ｸ・ｴ)", self.ml_model_root_ja_var)
        ctk.CTkLabel(
            ml_frame,
            text=t("・ｽ・罹株 ・ｨ・ｸ 尞ｴ・・・ｴ・・・model_meta.json) ・尖株 ・・怱 ・ｨ孖ｸ・ｼ ・・倣腹 ・・・溢慣・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        self.ml_coupled_status_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color=PALETTE.hint_text,
            wraplength=720,
            justify="left",
        )
        self.ml_coupled_status_label.pack(anchor="w", padx=12, pady=(4, 8))
        self.ml_coupled_status_detail_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color=PALETTE.hint_text,
            wraplength=760,
            justify="left",
        )
        self.ml_coupled_status_detail_label.pack(anchor="w", padx=12, pady=(0, 8))
        if hasattr(self, "_refresh_ml_backend_status"):
            self._refresh_ml_backend_status()
        if hasattr(self, "_sync_ml_correction_ui"):
            self._sync_ml_correction_ui()
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()

        vc_frame = self._ui_section(
            dev_container,
            "Option",
            "・ｰ・ｸ UI・川・・・・ｴ・・・ｬ・ｩ ・ｬ・・・・夋晨鮒・壱共. ・ｸ・ ・ｬ・ｼ・ｴ・罷株 ・罹ｰ懍梵 ・､・・ON・ｼ ・誤ｧ・嶹懍┳嶹罷姓・壱共.",
        )

        self.vc_neighbor_detail_controls = []
        self.vc_neighbor_detail_frames = []

        def _on_env_slider_change(value, var, label_widget, fmt):
            try:
                val = float(value)
            except Exception:
                return
            label_widget.configure(text=fmt.format(val))
            var.set(f"{val:.3f}".rstrip("0").rstrip("."))
            self._save_config()

        def _add_env_slider(parent, label, env_key, var, *, min_val, max_val, step, default_val, fmt):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=2)
            ctk.CTkLabel(row, text=label, width=230, anchor="w").pack(side="left")
            try:
                raw = str(var.get() or "").strip()
                current = float(raw) if raw else float(default_val)
            except Exception:
                current = float(default_val)
            dvar = ctk.DoubleVar(value=current)
            val_label = ctk.CTkLabel(row, text=fmt.format(current), width=60)
            val_label.pack(side="right", padx=(6, 0))
            slider = ctk.CTkSlider(
                row,
                from_=min_val,
                to=max_val,
                number_of_steps=int(round((max_val - min_val) / step)),
                variable=dvar,
                command=lambda v, lbl=val_label: _on_env_slider_change(v, var, lbl, fmt),
            )
            slider.pack(side="right", fill="x", expand=True, padx=8)
            ctk.CTkLabel(
                row,
                text=f"嶹俾ｲｽ・・・ {env_key}",
                text_color=PALETTE.hint_text,
                anchor="w",
            ).pack(side="left", padx=(6, 0))
            self.vc_neighbor_detail_controls.append(slider)
            self.advanced_tuning_slider_bindings[str(env_key)] = {
                "var": var,
                "dvar": dvar,
                "label": val_label,
                "fmt": fmt,
                "default": float(default_val),
                "min": float(min_val),
                "max": float(max_val),
            }
            return slider

        kr_box = ctk.CTkFrame(vc_frame)
        kr_box.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(kr_box, text=t("﨑懋ｵｭ・ｴ (KR)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            kr_box,
            text=t("VC ・ｴ・・・ｴ・・・ｬ・ｩ"),
            variable=self.kr_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        kr_detail = ctk.CTkFrame(kr_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(kr_detail)
        _add_env_slider(
            kr_detail,
            "・ｴ・・・簿巡(Blend)",
            "UTOA_KR_VC_NEIGHBOR_BLEND",
            self.kr_vc_neighbor_blend_var,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            default_val=0.35,
            fmt="{:.2f}",
        )
        _add_env_slider(
            kr_detail,
            "・罹劇 ・ｴ・・ms)",
            "UTOA_KR_VC_NEIGHBOR_MAX_SHIFT",
            self.kr_vc_neighbor_max_shift_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=45.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・ｴ・・・ｷ・､嵓・・ｰ・ ・ｬ・(ms)",
            "UTOA_KR_VC_NEIGHBOR_LEAD_MS",
            self.kr_vc_neighbor_lead_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=6.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・､・・・､嵓・・ ・ｰ・ ・ｬ・(ms)",
            "UTOA_KR_VC_NEIGHBOR_TAIL_MS",
            self.kr_vc_neighbor_tail_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=8.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・懍・ VC ・ｸ・ｴ(ms)",
            "UTOA_KR_VC_NEIGHBOR_MIN_LEN",
            self.kr_vc_neighbor_min_len_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=35.0,
            fmt="{:.0f}",
        )

        ja_box = ctk.CTkFrame(vc_frame)
        ja_box.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(ja_box, text=t("・ｼ・ｸ・ｴ (JA)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            ja_box,
            text=t("VC ・ｴ・・・ｴ・・・ｬ・ｩ"),
            variable=self.ja_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        ja_detail = ctk.CTkFrame(ja_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(ja_detail)
        _add_env_slider(
            ja_detail,
            "・ｴ・・・簿巡(Blend)",
            "UTOA_JA_VC_NEIGHBOR_BLEND",
            self.ja_vc_neighbor_blend_var,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            default_val=0.35,
            fmt="{:.2f}",
        )
        _add_env_slider(
            ja_detail,
            "・罹劇 ・ｴ・・ms)",
            "UTOA_JA_VC_NEIGHBOR_MAX_SHIFT",
            self.ja_vc_neighbor_max_shift_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=45.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・ｴ・・・ｷ・､嵓・・ｰ・ ・ｬ・(ms)",
            "UTOA_JA_VC_NEIGHBOR_LEAD_MS",
            self.ja_vc_neighbor_lead_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=6.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・､・・・､嵓・・ ・ｰ・ ・ｬ・(ms)",
            "UTOA_JA_VC_NEIGHBOR_TAIL_MS",
            self.ja_vc_neighbor_tail_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=8.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・懍・ VC ・ｸ・ｴ(ms)",
            "UTOA_JA_VC_NEIGHBOR_MIN_LEN",
            self.ja_vc_neighbor_min_len_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=35.0,
            fmt="{:.0f}",
        )
        self.vc_neighbor_detail_frame_by_lang = {
            "kr": kr_detail,
            "ja": ja_detail,
        }

        aligner_frame = ctk.CTkFrame(
            dev_container,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        aligner_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            aligner_frame,
            text=t("・・・・簿ｬ ・肥ｧ・・ｵ・們捩 嶸・椪 ・懋ｳｵ﨑們ｧ ・喜慣・壱共."),
            text_color=PALETTE.neutral_text,
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(10, 10))
        mfa_free_preview_row = ctk.CTkFrame(aligner_frame, fg_color="transparent")
        mfa_free_preview_row.pack(fill="x", padx=12, pady=(0, 10))
        self.mfa_free_oto_preview_btn = ctk.CTkButton(
            mfa_free_preview_row,
            text=t("MFA-Free SSL ・ｬ・ｯ ・ｴ・啄┣ 奛護侃孖ｸ"),
            width=230,
            height=28,
            command=lambda: self._run_mfa_free_oto_preview_from_ui(),
        )
        _style_primary_button(self.mfa_free_oto_preview_btn)
        self.mfa_free_oto_preview_btn.pack(side="left", padx=(0, 10))
        self.mfa_free_oto_preview_hint = ctk.CTkLabel(
            mfa_free_preview_row,
            text=t("嶸・椪 WAV/奛懦伯・ｿ ・ｰ・・ｼ・・preview oto.ini, anchor JSON, overlay HTML・・・晧┳﨑ｩ・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=620,
            justify="left",
        )
        self.mfa_free_oto_preview_hint.pack(side="left", fill="x", expand=True)

        dev_reset_row = ctk.CTkFrame(dev_container, fg_color="transparent")
        dev_reset_row.pack(fill="x", padx=10, pady=(8, 10))
        reset_btn = ctk.CTkButton(
            dev_reset_row,
            text=t("Option"),
            width=170,
            height=28,
            fg_color=PALETTE.danger_button_bg,
            hover_color=PALETTE.danger_button_hover,
            text_color=PALETTE.primary_button_text,
            command=lambda: self._reset_developer_settings_defaults(),
        )
        reset_btn.pack(side="left")

        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_advanced_tuning_slider_controls"):
            self._sync_advanced_tuning_slider_controls()

    def _build_profile_tune_tab(self):
        # VC ・ｴ・母ｰ・・ｰ・・UI・・・・・・､・・夋ｭ・ｼ・・奝ｵ﨑ｩ・們来・ｵ・壱共.
        return

    def _ensure_advanced_setting_vars(self) -> None:
        if hasattr(self, "_advanced_setting_vars_ready"):
            return
        self._advanced_setting_vars_ready = True
        defaults = {
            "enable_ml_correction_var": ("bool", True),
            "vc_correction_enable_var": ("bool", True),
            "kr_continuity_enable_var": ("bool", True),
            "cvn_correction_enable_var": ("bool", False),
            "cvn_low_conf_only_var": ("bool", False),
            "mapping_supervised_enable_var": ("bool", True),
            "cv_order_prior_enable_var": ("bool", True),
            "kr_vc_neighbor_enable_var": ("bool", True),
            "ja_vc_neighbor_enable_var": ("bool", True),
            "soft_bank_mode_var": ("bool", False),
            "low_rms_gain_enable_var": ("bool", False),
            "weak_voice_assist_enable_var": ("bool", True),
            "weak_boundary_reduce_missing_var": ("bool", False),
            "weak_boundary_block_mismap_var": ("bool", False),
            "ml_coupled_enable_var": ("bool", True),
            "ml_coupled_min_conf_use_model_meta_var": ("bool", True),
            "ml_coupled_strict_constraint_var": ("bool", True),
            "ml_batch_inference_enable_var": ("bool", True),
            "ml_legacy_fallback_enable_var": ("bool", True),
            "ml_hybrid_routing_enable_var": ("bool", True),
            "ml_e2e_enable_var": ("bool", False),
            "low_conf_force_lock_mode_var": ("bool", False),
            "kr_continuity_max_offset_adj_var": ("str", ""),
            "weak_voice_assist_strength_var": ("str", ""),
            "mapping_strict_mode_var": ("str", "・・胸德・・・ｲｩ(・・攷 嵂餓捩 尞ｴ・ｱ)"),
            "mapping_supervised_mode_var": ("str", "・尖徐(・護棗)"),
            "cv_order_prior_strength_var": ("str", ""),
            "kr_mapping_confidence_threshold_var": ("str", ""),
            "ml_route_var": ("str", "・尖徐(・尖徐 ・ｼ・ｰ甯・"),
            "ml_selector_mode_var": ("str", "+・・駕┣"),
            "ml_e2e_mode_var": ("str", "hybrid"),
            "ml_e2e_t_low_var": ("str", ""),
            "ml_e2e_t_high_var": ("str", ""),
            "ml_e2e_blend_alpha_var": ("str", ""),
            "ml_coupled_min_conf_var": ("str", ""),
            "ml_coupled_min_conf_model_offset_var": ("str", ""),
            "ml_coupled_min_conf_kr_cv_var": ("str", ""),
            "ml_coupled_min_conf_kr_cvc_var": ("str", ""),
            "ml_coupled_min_conf_kr_cvvc_var": ("str", ""),
            "ml_coupled_min_conf_kr_vcv_var": ("str", ""),
            "ml_coupled_min_conf_ja_cv_var": ("str", ""),
            "ml_coupled_min_conf_ja_cvc_var": ("str", ""),
            "ml_coupled_min_conf_ja_cvvc_var": ("str", ""),
            "ml_coupled_min_conf_ja_vcv_var": ("str", ""),
            "ml_coupled_device_var": ("str", "auto"),
            "ml_coupled_backend_var": ("str", "auto"),
            "ml_batch_inference_size_var": ("str", "256"),
            "ml_anchor_mel_gamma_var": ("str", "1.2"),
            "kr_vc_neighbor_blend_var": ("str", ""),
            "kr_vc_neighbor_max_shift_var": ("str", ""),
            "kr_vc_neighbor_lead_ms_var": ("str", ""),
            "kr_vc_neighbor_tail_ms_var": ("str", ""),
            "kr_vc_neighbor_min_len_var": ("str", ""),
            "ja_vc_neighbor_blend_var": ("str", ""),
            "ja_vc_neighbor_max_shift_var": ("str", ""),
            "ja_vc_neighbor_lead_ms_var": ("str", ""),
            "ja_vc_neighbor_tail_ms_var": ("str", ""),
            "ja_vc_neighbor_min_len_var": ("str", ""),
        }
        for name, (kind, default) in defaults.items():
            if hasattr(self, name):
                continue
            if kind == "bool":
                setattr(self, name, ctk.BooleanVar(value=bool(default)))
            else:
                setattr(self, name, ctk.StringVar(value=str(default)))

    def _get_or_create_advanced_dev_container(self, parent):
        frame = getattr(self, "advanced_developer_frame", None)
        if frame is not None:
            try:
                if frame.winfo_exists():
                    return frame
            except Exception:
                pass
        dev_container = ctk.CTkFrame(parent, fg_color="transparent")
        dev_container.pack(fill="x", padx=0, pady=(0, 0))
        self.advanced_developer_frame = dev_container
        return dev_container

    def _ensure_advanced_dev_runtime_controls(self, parent):
        frame = getattr(self, "advanced_dev_runtime_controls_frame", None)
        if frame is not None:
            try:
                if frame.winfo_exists():
                    return frame
            except Exception:
                pass

        frame = ctk.CTkFrame(
            parent,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        frame.pack(fill="x", padx=10, pady=(0, 8))
        self.advanced_dev_runtime_controls_frame = frame

        ctk.CTkLabel(
            frame,
            text=t("・､嵂・・ｱ・罷糖"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.pack(anchor="w", padx=12, pady=(0, 4), fill="x")
        ctk.CTkLabel(top_row, text="Coupled backend", text_color=PALETTE.neutral_text).pack(side="left")

        backend_value = str(self.ml_coupled_backend_var.get() if hasattr(self, "ml_coupled_backend_var") else "auto").strip().lower()
        backend_value = {"ensemble_v1": "ensemble"}.get(backend_value, backend_value)
        if backend_value not in {"auto", "ensemble"}:
            backend_value = "auto"
        if hasattr(self, "ml_coupled_backend_var"):
            self.ml_coupled_backend_var.set(backend_value)

        backend_menu = ctk.CTkOptionMenu(
            top_row,
            values=["auto", "ensemble"],
            variable=self.ml_coupled_backend_var,
            width=110,
            command=lambda _v: self._on_ml_backend_change(),
        )
        self._style_blue_menu(backend_menu)
        backend_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            top_row,
            text=t("(auto=ensemble ・ｰ・)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        return frame

    def _build_advanced_settings_tab(self):
        tab_name = "Advanced"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        container = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        self._ensure_advanced_setting_vars()
        self.advanced_tuning_slider_bindings = {}

        build_declarative(
            self,
            container,
            layout_path=self._layout_path(),
            tab_key="advanced_settings",
            context={
                "overlay": overlay,
                "root": tab,
                "content": container,
                "tab": tab,
            },
        )
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_ml_correction_ui"):
            self._sync_ml_correction_ui()
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()
        if hasattr(self, "_sync_advanced_tuning_slider_controls"):
            self._sync_advanced_tuning_slider_controls()

    def _slot_advanced_basic_toggles(self, parent, _layout_root, _node):
        basic_toggle_frame = ctk.CTkFrame(
            parent,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        basic_toggle_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.advanced_basic_frame = basic_toggle_frame
        ctk.CTkLabel(
            basic_toggle_frame,
            text=t("Option"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("ML ・ｴ・・・ｬ・ｩ"),
            text_color="#9A8250",
            variable=self.enable_ml_correction_var,
            command=(
                self._on_enable_ml_correction_toggle
                if hasattr(self, "_on_enable_ml_correction_toggle")
                else self._save_config
            ),
        )
        self.enable_ml_correction_checkbox.pack(anchor="w", padx=12, pady=(0, 4))

        self.kr_continuity_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・ｰ・作┳/甯護攵 ・ｼ・・ｱ ・ｴ・・・ｬ・ｩ"),
            text_color="#64866F",
            variable=self.vc_correction_enable_var,
            command=self._on_vc_correction_toggle,
        )
        self.kr_continuity_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        continuity_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("嶸菩享 ・・ｽ ・・・肥ｲ懋ｰ廷擽 ・尖徐 ・・圸・ｩ・壱共. 﨑・囈﨑・・ｽ・ｰ ・・椈 ・罹ｰ懍梵 ・ｬ・ｼ・ｴ・肥乱・・・ｸ・・庭ｧ・・ｸ・ｸ ・ｰ・倣葺・ｸ・・"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        continuity_help_label.pack(anchor="w", padx=34, pady=(0, 6))
        self._bind_wraplength_to_container(basic_toggle_frame, [continuity_help_label], padding=64, min_wrap=260)

        post_frame = ctk.CTkFrame(
            parent,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        post_frame.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(
            post_frame,
            text=t("Option"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cont_row = ctk.CTkFrame(post_frame, fg_color="transparent")
        cont_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            cont_row,
            text=t("・ｰ・作┳ offset ・ｴ・・・・復 (ms)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        cont_entry = ctk.CTkEntry(
            cont_row,
            width=90,
            textvariable=self.kr_continuity_max_offset_adj_var,
            placeholder_text=t("・ｰ・ｸ・・180)"),
        )
        cont_entry.pack(side="left", padx=(10, 8))
        cont_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cont_row,
            text=t("增ｬ・・・｡・・・・・・ｰ・作┳ ・ｴ・菩擽 ・倣紛・・, ・滝ｲ・・｡・・・・・・ｴ・菩擽 ・ｽ﨑ｴ・瀧笈・､."),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        bank_preset_frame = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        bank_preset_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(
            bank_preset_frame,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        bank_preset_help_label = ctk.CTkLabel(
            bank_preset_frame,
            text=t("soft/・ｼ・・・・〓・ｩ 嵓・ｦｬ・駆揆 﨑・・溢乱 ・・圸﨑ｩ・壱共. ・ｼ・戦・・ ・ｴ・菩揆 ・倣葺・・ ・ｼ・們捩 ・溢菩・愍・・・呷梠﨑ｩ・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=730,
            justify="left",
        )
        bank_preset_help_label.pack(anchor="w", pady=(0, 6))
        self._bind_wraplength_to_container(bank_preset_frame, [bank_preset_help_label], padding=30, min_wrap=240)

        preset_grid = ctk.CTkFrame(bank_preset_frame, fg_color="transparent")
        preset_grid.pack(anchor="w")
        preset_grid.grid_columnconfigure(0, weight=1)
        preset_grid.grid_columnconfigure(1, weight=1)

        self.voicebank_preset_buttons = {}

        btn_soft_agg = ctk.CTkButton(
            preset_grid,
            text=t("soft ・・〓 ﾂｷ ・ｼ・戦・"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "aggressive"),
        )
        self._style_primary_button(btn_soft_agg)
        btn_soft_agg.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_soft_cons = ctk.CTkButton(
            preset_grid,
            text=t("Option"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "conservative"),
        )
        self._style_primary_button(btn_soft_cons)
        btn_soft_cons.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="w")

        btn_normal_agg = ctk.CTkButton(
            preset_grid,
            text=t("・ｼ・・・・〓 ﾂｷ ・ｼ・戦・"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "aggressive"),
        )
        self._style_primary_button(btn_normal_agg)
        btn_normal_agg.grid(row=1, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_normal_cons = ctk.CTkButton(
            preset_grid,
            text=t("Option"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "conservative"),
        )
        self._style_primary_button(btn_normal_cons)
        btn_normal_cons.grid(row=1, column=1, padx=(0, 8), pady=2, sticky="w")

        self.voicebank_preset_buttons = {
            "soft:aggressive": btn_soft_agg,
            "soft:conservative": btn_soft_cons,
            "normal:aggressive": btn_normal_agg,
            "normal:conservative": btn_normal_cons,
        }
        if hasattr(self, "_sync_voicebank_preset_button_styles"):
            self._sync_voicebank_preset_button_styles()

        def _sync_pronunciation_missing_reduce_state():
            enabled = False
            if hasattr(self, "soft_bank_mode_var"):
                enabled = bool(self.soft_bank_mode_var.get()) or enabled
            if hasattr(self, "weak_voice_assist_enable_var"):
                enabled = bool(self.weak_voice_assist_enable_var.get()) or enabled
            if hasattr(self, "soft_bank_mode_var"):
                self.soft_bank_mode_var.set(enabled)
            if hasattr(self, "weak_voice_assist_enable_var"):
                self.weak_voice_assist_enable_var.set(enabled)

        def _on_pronunciation_missing_reduce_toggle():
            enabled = bool(self.soft_bank_mode_var.get()) if hasattr(self, "soft_bank_mode_var") else False
            if hasattr(self, "weak_voice_assist_enable_var"):
                self.weak_voice_assist_enable_var.set(enabled)
            if hasattr(self, "_on_weak_voice_assist_toggle"):
                self._on_weak_voice_assist_toggle()
            else:
                if hasattr(self, "_sync_weak_voice_assist_controls"):
                    self._sync_weak_voice_assist_controls()
                self._save_config()

        _sync_pronunciation_missing_reduce_state()
        self.soft_bank_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・懍搆 ・・攷 ・・擽・ｰ"),
            text_color="#6F819A",
            variable=self.soft_bank_mode_var,
            command=_on_pronunciation_missing_reduce_toggle,
        )
        self.soft_bank_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        soft_bank_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・ｨ・誤ｦｬ/・ｽ﨑・・懍┳・川・ ・・攷ﾂｷ・ｵ・ｱ ・､・・懍揆 ・・桿・壱共. (soft ・・〓 + ・尖ｪｨ ・・・・ｴ・・﨑ｨ・・・・圸)"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        soft_bank_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [soft_bank_help_label], padding=64, min_wrap=260)

        self.low_rms_gain_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("・・ｼ・ｨ WAV ・尖徐 ・晨少/・母ｷ懦剩 (RMS ・ｰ・・ ・尖ｳｸ ・ｴ・ｴ)"),
            text_color="#6F819A",
            variable=self.low_rms_gain_enable_var,
            command=self._save_config,
        )
        self.low_rms_gain_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_rms_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・簿ｬ ・ｨ・・乱・罹ｧ・・・亨 ・卓羅 ・ｵ・ｬ・ｸ・・・ｼ・ｨ ・ｴ・菩揆 ・・圸﨑ｩ・壱共. ・尖ｳｸ WAV 甯護攵・ ・壱劇 ・・ｽ﨑們ｧ ・喜慣・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        low_rms_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [low_rms_help_label], padding=64, min_wrap=260)

        weak_voice_strength_row = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        weak_voice_strength_row.pack(fill="x", padx=34, pady=(0, 10))
        self.weak_voice_strength_title_label = ctk.CTkLabel(
            weak_voice_strength_row,
            text=t("・尖ｪｨ ・・・・簿巡"),
            width=130,
            anchor="w",
            text_color=PALETTE.neutral_text,
        )
        self.weak_voice_strength_title_label.pack(side="left")
        try:
            _raw_strength = (
                str(self.weak_voice_assist_strength_var.get() or "")
                if hasattr(self, "weak_voice_assist_strength_var")
                else ""
            ).strip()
            _strength_current = float(_raw_strength) if _raw_strength else 0.35
        except Exception:
            _strength_current = 0.35
        _strength_current = max(0.0, min(1.0, _strength_current))
        weak_voice_strength_dvar = ctk.DoubleVar(value=_strength_current)
        weak_voice_strength_value_label = ctk.CTkLabel(
            weak_voice_strength_row,
            text=f"{_strength_current:.2f}",
            width=46,
        )
        weak_voice_strength_value_label.pack(side="right", padx=(6, 0))
        self.weak_voice_strength_value_label = weak_voice_strength_value_label

        def _on_weak_voice_strength_change(value):
            try:
                val = max(0.0, min(1.0, float(value)))
            except Exception:
                return
            weak_voice_strength_value_label.configure(text=f"{val:.2f}")
            if hasattr(self, "weak_voice_assist_strength_var"):
                self.weak_voice_assist_strength_var.set(f"{val:.3f}".rstrip("0").rstrip("."))
            self._save_config()

        weak_voice_strength_slider = ctk.CTkSlider(
            weak_voice_strength_row,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=weak_voice_strength_dvar,
            command=_on_weak_voice_strength_change,
        )
        weak_voice_strength_slider.pack(side="right", fill="x", expand=True, padx=8)
        self.weak_voice_strength_slider = weak_voice_strength_slider
        self.advanced_tuning_slider_bindings["UTOA_MFA_WEAK_VOICE_PREEMPH_MIX"] = {
            "var": getattr(self, "weak_voice_assist_strength_var", None),
            "dvar": weak_voice_strength_dvar,
            "label": weak_voice_strength_value_label,
            "fmt": "{:.2f}",
            "default": 0.35,
            "min": 0.0,
            "max": 1.0,
        }
        if hasattr(self, "_sync_weak_voice_assist_controls"):
            self._sync_weak_voice_assist_controls()

        mapping_strict_row = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        mapping_strict_row.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            mapping_strict_row,
            text=t("・護・・､・､﨑・・ｨ・ｨ"),
            text_color=PALETTE.neutral_text,
            width=130,
            anchor="w",
        ).pack(side="left")
        mapping_strict_menu = ctk.CTkOptionMenu(
            mapping_strict_row,
            values=(
                self._get_mapping_strict_mode_option_labels()
                if hasattr(self, "_get_mapping_strict_mode_option_labels")
                else ["・・・・・ｲｩ", "・・胸德・・・ｲｩ(・・攷 嵂餓捩 尞ｴ・ｱ)", "off"]
            ),
            variable=self.mapping_strict_mode_var,
            width=260,
            command=lambda _v: self._save_config(),
        )
        self._style_blue_menu(mapping_strict_menu)
        mapping_strict_menu.pack(side="left", padx=(10, 0))
        mapping_strict_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・護溢揆 寀・ｴ ・・･・ｱ・・・・擽・・・・, ・尖徐 ・､・菩乱・・・､墲ｵ・俯株 ・川攵・ｬ・ｴ・､・ ・們牟・ ・・・溢慣・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        mapping_strict_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self.low_conf_force_lock_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
            variable=self.low_conf_force_lock_mode_var,
            command=self._save_config,
        )
        self.low_conf_force_lock_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_conf_force_lock_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("・簿ｬ ・・ｰ・ ・ｮ・ ・ｬ・・乱・罹株 弡・ｳｴ ・戦売・ｼ ・賀ｳ ・溢メ ・・ｹ們乱 ・・倣鮒・壱共."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        low_conf_force_lock_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(
            basic_toggle_frame,
            [mapping_strict_help_label, low_conf_force_lock_help_label],
            padding=64,
            min_wrap=260,
        )

        weak_boundary_row = self._ui_row(basic_toggle_frame, padx=12, pady=(0, 2))
        for option in ADVANCED_WEAK_BOUNDARY_OPTIONS:
            var = getattr(self, option["var"])
            self._ui_inline_checkbox(
                weak_boundary_row,
                text=option["text"],
                variable=var,
                command=self._save_config,
                padx=option.get("padx", (0, 0)),
            )
        weak_boundary_help = self._ui_help(basic_toggle_frame, ADVANCED_WEAK_BOUNDARY_HELP)
        self._bind_wraplength_to_container(basic_toggle_frame, [weak_boundary_help], padding=64, min_wrap=260)
        return basic_toggle_frame

    def _slot_advanced_ml_section(self, parent, _layout_root, _node):
        public_ml_frame = self._ui_section(
            parent,
            "Option",
            "・ｰ・ｸ ・ｴ・・・ｨ・懍凰 ・・OTO ・ｴ・菩揆 ・・ｴ・・・ｰ・逸鮒・壱共.",
        )
        self.advanced_ml_public_frame = public_ml_frame

        selector_row = ctk.CTkFrame(public_ml_frame, fg_color="transparent")
        selector_row.pack(anchor="w", padx=12, pady=(2, 6), fill="x")
        ctk.CTkLabel(
            selector_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")

        mode_code = "selector"
        if hasattr(self, "_normalize_ml_selector_mode"):
            mode_code = self._normalize_ml_selector_mode(
                self.ml_selector_mode_var.get() if hasattr(self, "ml_selector_mode_var") else "selector"
            )
        mode_label = "delta" if mode_code == "delta" else "additive"
        if hasattr(self, "ml_selector_mode_var"):
            self.ml_selector_mode_var.set(mode_label)

        self.ml_selector_mode_segment = ctk.CTkSegmentedButton(
            selector_row,
            values=["auto", "additive"],
            variable=self.ml_selector_mode_var,
            command=lambda _value: self._save_config(),
        )
        self.ml_selector_mode_segment.pack(side="left", padx=(10, 0))

        selector_help = ctk.CTkLabel(
            public_ml_frame,
            text=t("Option"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=720,
        )
        selector_help.pack(anchor="w", padx=34, pady=(0, 6))
        self._bind_wraplength_to_container(public_ml_frame, [selector_help], padding=64, min_wrap=240)

        self.ml_coupled_enable_checkbox = ctk.CTkCheckBox(
            public_ml_frame,
            text=t("・・OTO ・ｴ・・・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_coupled_enable_var,
            command=self._save_config,
        )
        self.ml_coupled_enable_checkbox.pack(anchor="w", padx=12, pady=(4, 8))

        dev_container = self._get_or_create_advanced_dev_container(parent)
        self._ensure_advanced_dev_runtime_controls(dev_container)

        ml_frame = self._ui_section(
            dev_container,
            ADVANCED_ML_SECTION_TITLE,
            ADVANCED_ML_SECTION_SUBTITLE,
        )
        self.advanced_ml_section_frame = ml_frame
        if not hasattr(self, "mapping_supervised_enable_var"):
            self.mapping_supervised_enable_var = ctk.BooleanVar(value=True)

        self.mapping_supervised_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Syllable 弡・ｳｴ・・・ｬ・ｩ (・簿ｬ ・ｰ・・・・・ｵ・､)"),
            text_color="#5F8C87",
            variable=self.mapping_supervised_enable_var,
            command=(
                self._on_mapping_supervised_toggle
                if hasattr(self, "_on_mapping_supervised_toggle")
                else self._save_config
            ),
        )
        self.mapping_supervised_enable_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        mapping_supervised_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        mapping_supervised_row.pack(anchor="w", padx=12, pady=(4, 8), fill="x")
        ctk.CTkLabel(
            mapping_supervised_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        self.mapping_supervised_mode_menu = ctk.CTkOptionMenu(
            mapping_supervised_row,
            values=(
                self._get_mapping_supervised_mode_option_labels()
                if hasattr(self, "_get_mapping_supervised_mode_option_labels")
                else ["・尖徐(・護棗)", "・・・ｵ・､ ・卓峡", "mel+aic", "mel+align"]
            ),
            variable=self.mapping_supervised_mode_var,
            width=200,
            command=lambda _v: self._save_config(),
        )
        self._style_blue_menu(self.mapping_supervised_mode_menu)
        self.mapping_supervised_mode_menu.pack(side="left", padx=(10, 0))

        cv_order_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        cv_order_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        self.cv_order_prior_enable_checkbox = ctk.CTkCheckBox(
            cv_order_row,
            text=t("CV ・懍・ prior ・ｴ・・・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.cv_order_prior_enable_var,
            command=self._save_config,
        )
        self.cv_order_prior_enable_checkbox.pack(side="left")
        ctk.CTkLabel(
            cv_order_row,
            text=t("・簿巡"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(10, 4))
        cv_prior_entry = ctk.CTkEntry(
            cv_order_row,
            width=70,
            textvariable=self.cv_order_prior_strength_var,
            placeholder_text=t("auto"),
        )
        cv_prior_entry.pack(side="left", padx=(0, 6))
        cv_prior_entry.bind("<FocusOut>", lambda _e: self._save_config())

        threshold_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        threshold_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            threshold_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        threshold_entry = ctk.CTkEntry(
            threshold_row,
            width=70,
            textvariable=self.kr_mapping_confidence_threshold_var,
            placeholder_text=t("auto"),
        )
        threshold_entry.pack(side="left", padx=(10, 6))
        threshold_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            threshold_row,
            text=t("・廷擽 ・廷揆・俯｡・弡・ｳｴ・・・・圸 ・肥怱・ｼ ・・棘・壱共."),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))
        self.mapping_supervised_dependent_frames = [
            mapping_supervised_row,
            cv_order_row,
            threshold_row,
        ]

        route_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        route_row.pack(anchor="w", padx=12, pady=(4, 8), fill="x")
        ctk.CTkLabel(
            route_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        route_menu = ctk.CTkOptionMenu(
            route_row,
            values=(
                self._get_ml_route_option_labels()
                if hasattr(self, "_get_ml_route_option_labels")
                else ["・尖徐(・尖徐 ・ｼ・ｰ甯・", "No-MFA", "v1", "v2", "E2E 﨑們擽・誤ｦｬ・・・､嵭・"]
            ),
            variable=self.ml_route_var,
            width=180,
            command=self._on_ml_route_change if hasattr(self, "_on_ml_route_change") else (lambda _v: self._save_config()),
        )
        self.ml_route_menu = route_menu
        self._style_blue_menu(route_menu)
        route_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            route_row,
            text=t("Option"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        coupled_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        coupled_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            coupled_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        coupled_conf_entry = ctk.CTkEntry(
            coupled_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_var,
            placeholder_text=t("auto"),
        )
        coupled_conf_entry.pack(side="left", padx=(10, 8))
        coupled_conf_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            coupled_row,
            text="device",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(6, 4))
        coupled_device_menu = ctk.CTkOptionMenu(
            coupled_row,
            values=["auto", "cpu", "cuda"],
            variable=self.ml_coupled_device_var,
            width=90,
            command=lambda _v: self._save_config(),
        )
        self._style_blue_menu(coupled_device_menu)
        coupled_device_menu.pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(
            coupled_row,
            text=t("Strict ・懍平"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_strict_constraint_var,
            command=self._save_config,
        ).pack(side="left")

        model_conf_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkCheckBox(
            model_conf_row,
            text=t("Model-aware min_conf (model_meta) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_coupled_min_conf_use_model_meta_var,
            command=self._save_config,
        ).pack(side="left")
        ctk.CTkLabel(
            model_conf_row,
            text="offset",
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(10, 4))
        model_conf_offset_entry = ctk.CTkEntry(
            model_conf_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_model_offset_var,
            placeholder_text="0.00",
        )
        model_conf_offset_entry.pack(side="left", padx=(0, 8))
        model_conf_offset_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            model_conf_row,
            text="(range: -0.30 ~ +0.30)",
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        model_conf_grid = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_grid.pack(anchor="w", padx=12, pady=(4, 0), fill="x")

        def _add_min_conf_field(parent, label, var, placeholder):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(side="left", padx=(0, 10))
            ctk.CTkLabel(row, text=label, text_color=PALETTE.neutral_text).pack(side="left")
            ent = ctk.CTkEntry(row, width=78, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", padx=(6, 0))
            ent.bind("<FocusOut>", lambda _e: self._save_config())
            return ent

        kr_row = ctk.CTkFrame(model_conf_grid, fg_color="transparent")
        kr_row.pack(anchor="w", fill="x", pady=(0, 2))
        ctk.CTkLabel(kr_row, text="KR", text_color="#5F8C87", width=26, anchor="w").pack(side="left")
        _add_min_conf_field(kr_row, "CV", self.ml_coupled_min_conf_kr_cv_var, "0.55")
        _add_min_conf_field(kr_row, "CVC", self.ml_coupled_min_conf_kr_cvc_var, "0.75")
        _add_min_conf_field(kr_row, "CVVC", self.ml_coupled_min_conf_kr_cvvc_var, "0.78")
        _add_min_conf_field(kr_row, "VCV", self.ml_coupled_min_conf_kr_vcv_var, "0.72")

        ja_row = ctk.CTkFrame(model_conf_grid, fg_color="transparent")
        ja_row.pack(anchor="w", fill="x", pady=(0, 2))
        ctk.CTkLabel(ja_row, text="JA", text_color="#5F8C87", width=26, anchor="w").pack(side="left")
        _add_min_conf_field(ja_row, "CV", self.ml_coupled_min_conf_ja_cv_var, "0.40")
        _add_min_conf_field(ja_row, "CVC", self.ml_coupled_min_conf_ja_cvc_var, "0.50")
        _add_min_conf_field(ja_row, "CVVC", self.ml_coupled_min_conf_ja_cvvc_var, "0.72")
        _add_min_conf_field(ja_row, "VCV", self.ml_coupled_min_conf_ja_vcv_var, "0.65")

        batch_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Batch inference (coupled v2) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_batch_inference_enable_var,
            command=self._save_config,
        )
        batch_enable_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        batch_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        batch_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            batch_row,
            text="Batch size",
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        batch_size_entry = ctk.CTkEntry(
            batch_row,
            width=90,
            textvariable=self.ml_batch_inference_size_var,
            placeholder_text="256",
        )
        batch_size_entry.pack(side="left", padx=(10, 8))
        batch_size_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            batch_row,
            text=t("(・護棗 128~512, ・懍・ 32)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        legacy_fallback_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("v1 fallback(lightgbm) ・ｴ・ｸ ・ｬ・ｩ"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_legacy_fallback_enable_var,
            command=self._save_config,
        )
        legacy_fallback_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        hybrid_routing_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("﨑們擽・誤ｦｬ・・・ｼ・ｰ甯・・・卓ｹ・・護擽孖ｸ) ・ｬ・ｩ"),
            text_color="#5F8C87",
            variable=self.ml_hybrid_routing_enable_var,
            command=self._save_config,
        )
        hybrid_routing_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_toggle_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Option"),
            text_color="#5F8C87",
            variable=self.ml_e2e_enable_var,
            command=(
                self._on_ml_e2e_toggle
                if hasattr(self, "_on_ml_e2e_toggle")
                else self._save_config
            ),
        )
        e2e_toggle_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        e2e_mode_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        ctk.CTkLabel(
            e2e_mode_row,
            text=t("Option"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        e2e_mode_menu = ctk.CTkOptionMenu(
            e2e_mode_row,
            values=["hybrid", "legacy_only", "e2e_only"],
            variable=self.ml_e2e_mode_var,
            width=120,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(e2e_mode_menu)
        e2e_mode_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            e2e_mode_row,
            text=t("(・護棗: hybrid)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        e2e_threshold_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        e2e_threshold_row.pack(anchor="w", padx=12, pady=(2, 4), fill="x")
        ctk.CTkLabel(e2e_threshold_row, text="T_low", text_color=PALETTE.neutral_text).pack(side="left")
        e2e_t_low_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_t_low_var,
            placeholder_text="0.52",
        )
        e2e_t_low_entry.pack(side="left", padx=(6, 8))
        e2e_t_low_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(e2e_threshold_row, text="T_high", text_color=PALETTE.neutral_text).pack(side="left", padx=(2, 0))
        e2e_t_high_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_t_high_var,
            placeholder_text="0.72",
        )
        e2e_t_high_entry.pack(side="left", padx=(6, 8))
        e2e_t_high_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(e2e_threshold_row, text="blend", text_color=PALETTE.neutral_text).pack(side="left", padx=(2, 0))
        e2e_blend_entry = ctk.CTkEntry(
            e2e_threshold_row,
            width=68,
            textvariable=self.ml_e2e_blend_alpha_var,
            placeholder_text="0.60",
        )
        e2e_blend_entry.pack(side="left", padx=(6, 8))
        e2e_blend_entry.bind("<FocusOut>", lambda _e: self._save_config())
        self.ml_e2e_dependent_frames = [e2e_mode_row, e2e_threshold_row]

        gamma_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        gamma_row.pack(anchor="w", padx=12, pady=(2, 6), fill="x")
        ctk.CTkLabel(
            gamma_row,
            text="Anchor mel gamma",
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        gamma_entry = ctk.CTkEntry(
            gamma_row,
            width=80,
            textvariable=self.ml_anchor_mel_gamma_var,
            placeholder_text="1.0",
        )
        gamma_entry.pack(side="left", padx=(10, 8))
        gamma_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            gamma_row,
            text="(>1.0: stronger shrink on low mel reliability)",
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        detail_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        detail_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        if not hasattr(self, "ml_coupled_status_detail_var"):
            self.ml_coupled_status_detail_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            detail_row,
            text=t("・川┷德・・ｴ・ｰ (・ｽ・・・・・・晧┳・ｼ)"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_status_detail_var,
            command=lambda: self._on_ml_backend_detail_toggle(),
        ).pack(side="left")

        def _model_root_row(label, var):
            row = ctk.CTkFrame(ml_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(6, 0))
            ctk.CTkLabel(row, text=label, width=150, anchor="w", text_color=PALETTE.neutral_text).pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=var, width=360)
            ent.pack(side="left", fill="x", expand=True, padx=(5, 6))
            ent.bind("<FocusOut>", lambda _e: self._on_ml_model_root_change())
            btn_row = ctk.CTkFrame(row, fg_color="transparent")
            btn_row.pack(side="right", padx=(6, 0))
            btn_width = 90
            browse_btn = ctk.CTkButton(
                btn_row,
                text=t("・ｾ・・ｳｴ・ｰ"),
                width=btn_width,
                command=lambda v=var: (
                    self._browse_folder_by_var(v, initial_dir=self._preferred_ml_model_browse_dir(v)),
                    self._on_ml_model_root_change(),
                ),
            )
            self._style_primary_button(browse_btn)
            browse_btn.pack(side="right")
            open_btn = ctk.CTkButton(
                btn_row,
                text=t("・ｴ・ｰ"),
                width=btn_width,
                command=lambda v=var: os.startfile(str(v.get()).strip()) if os.path.isdir(str(v.get()).strip()) else None,
            )
            self._style_primary_button(open_btn)
            open_btn.pack(side="right", padx=(0, 6))
            return ent

        if hasattr(self, "ml_model_root_kr_var"):
            _model_root_row("・ｨ・ｸ ・ｽ・・(﨑懋ｵｭ・ｴ)", self.ml_model_root_kr_var)
        if hasattr(self, "ml_model_root_ja_var"):
            _model_root_row("・ｨ・ｸ ・ｽ・・(・ｼ・ｸ・ｴ)", self.ml_model_root_ja_var)
        ctk.CTkLabel(
            ml_frame,
            text=t("・ｽ・罹株 ・ｨ・ｸ 尞ｴ・・・ｴ・・・model_meta.json) ・尖株 ・・怱 ・ｨ孖ｸ・ｼ ・・倣腹 ・・・溢慣・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        self.ml_coupled_status_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color=PALETTE.hint_text,
            wraplength=720,
            justify="left",
        )
        self.ml_coupled_status_label.pack(anchor="w", padx=12, pady=(4, 8))
        self.ml_coupled_status_detail_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color=PALETTE.hint_text,
            wraplength=760,
            justify="left",
        )
        self.ml_coupled_status_detail_label.pack(anchor="w", padx=12, pady=(0, 8))
        if hasattr(self, "_refresh_ml_backend_status"):
            self._refresh_ml_backend_status()
        if hasattr(self, "_sync_ml_correction_ui"):
            self._sync_ml_correction_ui()
        if hasattr(self, "_sync_mapping_supervised_ui"):
            self._sync_mapping_supervised_ui()
        if hasattr(self, "_sync_ml_e2e_controls"):
            self._sync_ml_e2e_controls()
        return ml_frame

    def _slot_advanced_vc_section(self, parent, _layout_root, _node):
        dev_container = self._get_or_create_advanced_dev_container(parent)

        vc_frame = self._ui_section(
            dev_container,
            "Option",
            "・ｰ・ｸ UI・川・・・・ｴ・・・ｬ・ｩ ・ｬ・・・・夋晨鮒・壱共. ・ｸ・ ・ｬ・ｼ・ｴ・罷株 ・罹ｰ懍梵 ・､・・ON・ｼ ・誤ｧ・嶹懍┳嶹罷姓・壱共.",
        )

        self.vc_neighbor_detail_controls = []
        self.vc_neighbor_detail_frames = []

        def _on_env_slider_change(value, var, label_widget, fmt):
            try:
                val = float(value)
            except Exception:
                return
            label_widget.configure(text=fmt.format(val))
            var.set(f"{val:.3f}".rstrip("0").rstrip("."))
            self._save_config()

        def _add_env_slider(parent, label, env_key, var, *, min_val, max_val, step, default_val, fmt):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=2)
            ctk.CTkLabel(row, text=label, width=230, anchor="w").pack(side="left")
            try:
                raw = str(var.get() or "").strip()
                current = float(raw) if raw else float(default_val)
            except Exception:
                current = float(default_val)
            dvar = ctk.DoubleVar(value=current)
            val_label = ctk.CTkLabel(row, text=fmt.format(current), width=60)
            val_label.pack(side="right", padx=(6, 0))
            slider = ctk.CTkSlider(
                row,
                from_=min_val,
                to=max_val,
                number_of_steps=int(round((max_val - min_val) / step)),
                variable=dvar,
                command=lambda v, lbl=val_label: _on_env_slider_change(v, var, lbl, fmt),
            )
            slider.pack(side="right", fill="x", expand=True, padx=8)
            ctk.CTkLabel(
                row,
                text=f"嶹俾ｲｽ・・・ {env_key}",
                text_color=PALETTE.hint_text,
                anchor="w",
            ).pack(side="left", padx=(6, 0))
            self.vc_neighbor_detail_controls.append(slider)
            self.advanced_tuning_slider_bindings[str(env_key)] = {
                "var": var,
                "dvar": dvar,
                "label": val_label,
                "fmt": fmt,
                "default": float(default_val),
                "min": float(min_val),
                "max": float(max_val),
            }
            return slider

        kr_box = ctk.CTkFrame(vc_frame)
        kr_box.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(kr_box, text=t("﨑懋ｵｭ・ｴ (KR)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            kr_box,
            text=t("VC ・ｴ・・・ｴ・・・ｬ・ｩ"),
            variable=self.kr_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        kr_detail = ctk.CTkFrame(kr_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(kr_detail)
        _add_env_slider(
            kr_detail,
            "・ｴ・・・簿巡(Blend)",
            "UTOA_KR_VC_NEIGHBOR_BLEND",
            self.kr_vc_neighbor_blend_var,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            default_val=0.35,
            fmt="{:.2f}",
        )
        _add_env_slider(
            kr_detail,
            "・罹劇 ・ｴ・・ms)",
            "UTOA_KR_VC_NEIGHBOR_MAX_SHIFT",
            self.kr_vc_neighbor_max_shift_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=45.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・ｴ・・・ｷ・､嵓・・ｰ・ ・ｬ・(ms)",
            "UTOA_KR_VC_NEIGHBOR_LEAD_MS",
            self.kr_vc_neighbor_lead_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=6.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・､・・・､嵓・・ ・ｰ・ ・ｬ・(ms)",
            "UTOA_KR_VC_NEIGHBOR_TAIL_MS",
            self.kr_vc_neighbor_tail_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=8.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            kr_detail,
            "・懍・ VC ・ｸ・ｴ(ms)",
            "UTOA_KR_VC_NEIGHBOR_MIN_LEN",
            self.kr_vc_neighbor_min_len_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=35.0,
            fmt="{:.0f}",
        )

        ja_box = ctk.CTkFrame(vc_frame)
        ja_box.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(ja_box, text=t("・ｼ・ｸ・ｴ (JA)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            ja_box,
            text=t("VC ・ｴ・・・ｴ・・・ｬ・ｩ"),
            variable=self.ja_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        ja_detail = ctk.CTkFrame(ja_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(ja_detail)
        _add_env_slider(
            ja_detail,
            "・ｴ・・・簿巡(Blend)",
            "UTOA_JA_VC_NEIGHBOR_BLEND",
            self.ja_vc_neighbor_blend_var,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            default_val=0.35,
            fmt="{:.2f}",
        )
        _add_env_slider(
            ja_detail,
            "・罹劇 ・ｴ・・ms)",
            "UTOA_JA_VC_NEIGHBOR_MAX_SHIFT",
            self.ja_vc_neighbor_max_shift_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=45.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・ｴ・・・ｷ・､嵓・・ｰ・ ・ｬ・(ms)",
            "UTOA_JA_VC_NEIGHBOR_LEAD_MS",
            self.ja_vc_neighbor_lead_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=6.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・､・・・､嵓・・ ・ｰ・ ・ｬ・(ms)",
            "UTOA_JA_VC_NEIGHBOR_TAIL_MS",
            self.ja_vc_neighbor_tail_ms_var,
            min_val=0.0,
            max_val=80.0,
            step=1.0,
            default_val=8.0,
            fmt="{:.0f}",
        )
        _add_env_slider(
            ja_detail,
            "・懍・ VC ・ｸ・ｴ(ms)",
            "UTOA_JA_VC_NEIGHBOR_MIN_LEN",
            self.ja_vc_neighbor_min_len_var,
            min_val=0.0,
            max_val=200.0,
            step=1.0,
            default_val=35.0,
            fmt="{:.0f}",
        )
        self.vc_neighbor_detail_frame_by_lang = {
            "kr": kr_detail,
            "ja": ja_detail,
        }

        aligner_frame = ctk.CTkFrame(
            dev_container,
            fg_color=PALETTE.panel_bg,
            border_width=1,
            border_color=PALETTE.panel_border,
        )
        aligner_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            aligner_frame,
            text=t("・・・・簿ｬ ・肥ｧ・・ｵ・們捩 嶸・椪 ・懋ｳｵ﨑們ｧ ・喜慣・壱共."),
            text_color=PALETTE.neutral_text,
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(10, 10))
        mfa_free_preview_row = ctk.CTkFrame(aligner_frame, fg_color="transparent")
        mfa_free_preview_row.pack(fill="x", padx=12, pady=(0, 10))
        self.mfa_free_oto_preview_btn = ctk.CTkButton(
            mfa_free_preview_row,
            text=t("MFA-Free SSL ・ｬ・ｯ ・ｴ・啄┣ 奛護侃孖ｸ"),
            width=230,
            height=28,
            command=lambda: self._run_mfa_free_oto_preview_from_ui(),
        )
        _style_primary_button(self.mfa_free_oto_preview_btn)
        self.mfa_free_oto_preview_btn.pack(side="left", padx=(0, 10))
        self.mfa_free_oto_preview_hint = ctk.CTkLabel(
            mfa_free_preview_row,
            text=t("嶸・椪 WAV/奛懦伯・ｿ ・ｰ・・ｼ・・preview oto.ini, anchor JSON, overlay HTML・・・晧┳﨑ｩ・壱共."),
            text_color=PALETTE.hint_text,
            wraplength=620,
            justify="left",
        )
        self.mfa_free_oto_preview_hint.pack(side="left", fill="x", expand=True)

        dev_reset_row = ctk.CTkFrame(dev_container, fg_color="transparent")
        dev_reset_row.pack(fill="x", padx=10, pady=(8, 10))
        reset_btn = ctk.CTkButton(
            dev_reset_row,
            text=t("Option"),
            width=170,
            height=28,
            fg_color=PALETTE.danger_button_bg,
            hover_color=PALETTE.danger_button_hover,
            text_color=PALETTE.primary_button_text,
            command=lambda: self._reset_developer_settings_defaults(),
        )
        reset_btn.pack(side="left")
        return vc_frame


