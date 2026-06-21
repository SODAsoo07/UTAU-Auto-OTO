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


PARAM_GROUPS = (
    (
        "기본 VC 파라미터",
        (
            ("VC_CONSONANT_RATIO", "VC 자음 구간 비율", 0.1, 1.0, 0.05),
            ("VC_VOWEL_START", "VC 모음 시작 비율", 0.1, 1.0, 0.05),
            ("VC_PRE_OFFSET", "VC 선행발음 오프셋 (ms)", 0, 50, 1),
            ("VC_OVL_RATIO", "VC 오버랩 비율", 0.1, 1.0, 0.05),
        ),
    ),
    (
        "기본 CV 파라미터",
        (
            ("CV_PRE_RATIO", "CV 선행발음 비율", 0.1, 1.0, 0.05),
            ("CV_OVL_RATIO", "CV 오버랩 비율", 0.1, 1.0, 0.05),
        ),
    ),
    (
        "이중모음 CV 파라미터",
        (
            ("DIPHTHONG_CV_PRE_RATIO", "이중모음 CV 선행발음 비율", 0.1, 1.0, 0.05),
            ("DIPHTHONG_CV_CONSONANT_RATIO", "이중모음 CV 자음 비율", 0.1, 1.0, 0.05),
        ),
    ),
    (
        "이중모음 VC 파라미터",
        (
            ("DIPHTHONG_VC_VOWEL_START", "이중모음 VC 모음 시작", 0.1, 1.0, 0.05),
            ("DIPHTHONG_VC_CONSONANT", "이중모음 VC 자음 비율", 0.1, 1.0, 0.05),
        ),
    ),
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


def _style_primary_button(widget):
    widget.configure(
        fg_color=PALETTE.primary_button_bg,
        hover_color=PALETTE.primary_button_hover,
        text_color=PALETTE.primary_button_text,
    )


class TabBuildersMixin:
    def _ensure_param_vars(self) -> None:
        existing = getattr(self, "param_vars", None)
        if not isinstance(existing, dict):
            existing = {}
            self.param_vars = existing
        for _group_name, params in PARAM_GROUPS:
            for key, _label, _min_val, _max_val, _step in params:
                if key not in existing:
                    existing[key] = ctk.DoubleVar(value=DEFAULT_PARAMS.get(key, 0.5))

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
        tab_name = "파이프라인"
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
                "1. Lab+사전 생성",
                "WAV 파일에서 Lab을 생성하고, 이어서 발음 사전 파일까지 한 번에 생성합니다.",
                self._run_lab_dict_gen,
            ),
            (
                "align",
                "2. 정렬 단계",
                "기본 HSMM OTO에서는 건너뜁니다. WFL 엔진 선택 시에만 TextGrid를 생성합니다.",
                self._run_mfa,
            ),
            (
                "oto",
                "3. OTO.ini 생성",
                "HSMM OTO 또는 TextGrid 기반으로 OTO 파라미터를 계산해 저장합니다.",
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
            text=t("실행"),
            font=("", 13, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", pady=(0, 4))

        aligner_btn_row = ctk.CTkFrame(left_actions, fg_color="transparent")
        aligner_btn_row.pack(anchor="w")
        aligner_btn_row.grid_columnconfigure(0, weight=1)

        self.wfl_status_btn = ctk.CTkButton(
            aligner_btn_row,
            text=t("🔍 WFL 상태 확인"),
            width=120,
            fg_color="#90CAF9",
            hover_color="#64B5F6",
            text_color="black",
            command=self._run_wfl_status_check,
        )
        self.wfl_status_btn.grid(row=0, column=0, padx=(0, 6), pady=1, sticky="w")

        self.wfl_model_btn = ctk.CTkButton(
            aligner_btn_row,
            text=t("⬇ WFL 모델 받기"),
            width=120,
            fg_color="#FFA726",
            hover_color="#FB8C00",
            text_color="black",
            command=self._run_wfl_model_download,
        )
        self.wfl_model_btn.grid(row=0, column=1, padx=(6, 0), pady=1, sticky="w")

        ctk.CTkLabel(
            right_actions,
            text=t("순서대로 실행"),
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
                    text="↓",
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
        self.mfa_status_label = None
        return None

    def _slot_pipeline_intro(self, parent, _layout_root, _node):
        intro_label = ctk.CTkLabel(
            parent,
            text=(
                "HSMM OTO는 외부 도구 설치 없이 바로 실행됩니다.\n"
                "WFL 엔진을 사용하려면 WFL 모델을 먼저 다운로드하세요."
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
                    text=t("OpenUtau 호환 별도 에일리어스 자동 생성"),
                    text_color="#5E7E95",
                    variable=self.openutau_var,
                    command=self._save_config,
                ).pack(anchor="w")

                self.gen_missing_vowels_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text=t("누락된 모음/모음열(VV) 에일리어스 보완 생성"),
                    text_color="#64866F",
                    variable=self.gen_missing_vowels_var,
                    command=self._save_config,
                )
                self.gen_missing_vowels_checkbox.pack(anchor="w", pady=(5, 0))
                self.gen_dash_alias_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text=t("어두/어미 '-' 에일리어스 생성"),
                    text_color=PALETTE.neutral_text,
                    variable=self.gen_dash_alias_var,
                    command=self._save_config,
                )
                self.gen_dash_alias_checkbox.pack(anchor="w", pady=(5, 0))

                ctk.CTkLabel(
                    opt_frame,
                    text=t("ML 보정은 자동으로 적용되며 모델이 없으면 자동으로 건너뜁니다."),
                    text_color=PALETTE.hint_text,
                ).pack(anchor="w", pady=(6, 0))
        return None

    def _build_params_tab(self):
        tab_name = "파라미터 조정"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        scroll = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        summary = ctk.CTkLabel(
            scroll,
            text=t("파라미터 조정 탭에서 OTO 보정 계수를 조정할 수 있습니다. 기본값 사용을 권장합니다."),
            text_color=PALETTE.neutral_text,
            justify="left",
            wraplength=760,
        )
        summary.pack(fill="x", padx=10, pady=(8, 8))
        self._bind_wraplength_to_container(scroll, [summary], padding=40, min_wrap=280)

        self._ensure_param_vars()

        for group_name, params in PARAM_GROUPS:
            ctk.CTkLabel(scroll, text=group_name, font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            for key, label, min_val, max_val, step in params:
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", padx=15, pady=2)
                var = self.param_vars[key]
                current = float(var.get())
                ctk.CTkLabel(row, text=label, width=250, anchor="w").pack(side="left")
                val_label = ctk.CTkLabel(row, text=f"{current:.2f}", width=50)
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
        tab_name = "로그"
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
        clear_btn = ctk.CTkButton(btn_frame, text=t("로그 지우기"), width=100, command=self._clear_log)
        _style_primary_button(clear_btn)
        clear_btn.pack(side="left")
        open_btn = ctk.CTkButton(
            btn_frame,
            text=t("로그 파일 열기"),
            width=120,
            command=lambda: os.startfile(self.log_path) if os.path.exists(self.log_path) else None,
        )
        _style_primary_button(open_btn)
        open_btn.pack(side="left", padx=5)

    def _build_detail_log_tab(self):
        tab_name = "상세 로그"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)

        desc = ctk.CTkLabel(
            tab,
            text=t("CMD/서브프로세스 원본 로그를 시간순으로 표시합니다. (저부하 버퍼 플러시)"),
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
            text=t("상세 로그 지우기"),
            width=120,
            command=lambda: self.detail_log_text.delete("1.0", "end"),
        )
        _style_primary_button(clear_btn)
        clear_btn.pack(side="left")
        open_btn = ctk.CTkButton(
            btn_frame,
            text=t("로그 파일 열기"),
            width=120,
            command=lambda: os.startfile(self.log_path) if os.path.exists(self.log_path) else None,
        )
        _style_primary_button(open_btn)
        open_btn.pack(side="left", padx=5)

    def _build_credits_tab(self):
        tab_name = "크레딧"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)

        scroll = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            scroll,
            text=t("-보정 모델 데이터 제공에 도움 주신 분들"),
            font=("", 15, "bold"),
            text_color=PALETTE.header_accent,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(6, 10))

        credit_lines = [
            "카메비 Kamebit",
            "22 twotwosibi",
            "촙타러/익명의 음성제공자 infinityecho00&Anonymous Voice Provider",
            "빈빈 2xbin",
            "류시아 Lyusia",
            "HUEY",
            "집냥 Ironic_SP",
            "라리우 RARIU",
            "서딕호 neohyajach",
            "나미니 Namini",
            "노와노 Nowano",
            "실안개 AngaeSil1115",
            "사토 야나기 Sato Yanagi",
            "혜성 Comet",
            "도살 BUTCHER_TUNING",
            "펜촉 ianharuesaone",
            "다유 DAU_Multiverse",
            "집냥 Blue",
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
            text=t("감사합니다!"),
            font=("", 14, "bold"),
            text_color=PALETTE.success_text,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(12, 6))

    def _build_advanced_settings_tab_legacy(self):
        tab_name = "고급 설정"
        tab = self._get_or_add_tab(tab_name)
        overlay = self._tab_overlay_color(tab_name)
        tab.configure(fg_color=overlay)
        container = ctk.CTkScrollableFrame(tab, fg_color=overlay)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        self._ensure_advanced_setting_vars()

        adv_intro_label = ctk.CTkLabel(
            container,
            text=t("고급 설정은 개발/튜닝용 항목입니다. 실사용에서는 기본값 유지 후 필요한 경우만 조정하세요."),
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
            text=t("핵심 옵션"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("ML 보정 사용"),
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
            text=t("연속성/파일 일관성 보정 사용"),
            text_color="#64866F",
            variable=self.vc_correction_enable_var,
            command=self._on_vc_correction_toggle,
        )
        self.kr_continuity_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        continuity_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("형식 변경 시 추천값이 자동 적용됩니다. 필요한 경우 아래 개발자 슬라이더에서 세부값만 미세 조정하세요."),
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
            text=t("후처리(연속성) 옵션"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cont_row = ctk.CTkFrame(post_frame, fg_color="transparent")
        cont_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            cont_row,
            text=t("연속성 offset 보정 상한 (ms)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        cont_entry = ctk.CTkEntry(
            cont_row,
            width=90,
            textvariable=self.kr_continuity_max_offset_adj_var,
            placeholder_text=t("기본값(180)"),
        )
        cont_entry.pack(side="left", padx=(10, 8))
        cont_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cont_row,
            text=t("크게 잡을수록 연속성 보정이 강해지고, 작게 잡을수록 보정이 약해집니다."),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        bank_preset_frame = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        bank_preset_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(
            bank_preset_frame,
            text=t("보이스뱅크 튜닝 프리셋"),
            text_color=PALETTE.neutral_text,
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        bank_preset_help_label = ctk.CTkLabel(
            bank_preset_frame,
            text=t("soft/일반 뱅크용 프리셋을 한 번에 적용합니다. 민감형은 보정을 강하게, 일반은 안정적으로 동작합니다."),
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
            text=t("soft 뱅크 · 민감형"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "aggressive"),
        )
        _style_primary_button(btn_soft_agg)
        btn_soft_agg.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_soft_cons = ctk.CTkButton(
            preset_grid,
            text=t("soft 뱅크 · 일반"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "conservative"),
        )
        _style_primary_button(btn_soft_cons)
        btn_soft_cons.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="w")

        btn_normal_agg = ctk.CTkButton(
            preset_grid,
            text=t("일반 뱅크 · 민감형"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "aggressive"),
        )
        _style_primary_button(btn_normal_agg)
        btn_normal_agg.grid(row=1, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_normal_cons = ctk.CTkButton(
            preset_grid,
            text=t("일반 뱅크 · 일반"),
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
            text=t("발음 누락 줄이기 추천"),
            text_color="#6F819A",
            variable=self.soft_bank_mode_var,
            command=self._save_config,
        )
        self.soft_bank_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        soft_bank_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("숨소리가 많이 섞이거나 발성이 약한 음원에서 발음 누락과 공백 오검출을 줄일 때 켜는 옵션입니다."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        soft_bank_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [soft_bank_help_label], padding=64, min_wrap=260)

        self.low_rms_gain_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("저볼륨 WAV 자동 증폭/정규화 (RMS 기반, 원본 보존)"),
            text_color="#6F819A",
            variable=self.low_rms_gain_enable_var,
            command=self._save_config,
        )
        self.low_rms_gain_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_rms_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("정렬 단계에서만 임시 작업 복사본에 볼륨 보정을 적용합니다. 원본 WAV 파일은 절대 변경하지 않습니다."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        low_rms_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [low_rms_help_label], padding=64, min_wrap=260)

        self.weak_voice_assist_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("발음이 흐린 음원에 추천 (자모 구분 강화)"),
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
            text=t("웅얼거림처럼 들리거나 자음/모음 구분이 흐린 음원에서 정렬 정확도를 높이기 위한 추천 옵션입니다. (원본 WAV 보존)"),
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
            text=t("자모 대비 강도"),
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
            text=t("음절 오매핑 차단"),
            text_color=PALETTE.neutral_text,
            width=130,
            anchor="w",
        ).pack(side="left")
        mapping_strict_menu = ctk.CTkOptionMenu(
            mapping_strict_row,
            values=(
                self._get_mapping_strict_mode_option_labels()
                if hasattr(self, "_get_mapping_strict_mode_option_labels")
                else ["완전 엄격", "적당히 엄격(누락 행은 폴백)", "off"]
            ),
            variable=self.mapping_strict_mode_var,
            width=260,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(mapping_strict_menu)
        mapping_strict_menu.pack(side="left", padx=(10, 0))
        mapping_strict_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("음절을 틀릴 가능성을 줄이는 대신, 자동 설정에서 스킵되는 에일리어스가 늘어날 수 있습니다."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        mapping_strict_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self.low_conf_force_lock_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("저신뢰 구간 강제 고정 모드"),
            text_color=PALETTE.neutral_text,
            variable=self.low_conf_force_lock_mode_var,
            command=self._save_config,
        )
        self.low_conf_force_lock_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_conf_force_lock_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("정렬 신뢰가 낮은 구간에서는 후보 점프를 막고 예상 위치에 고정합니다."),
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

        self.cvn_correction_enable_checkbox = self._ui_checkbox(
            ml_frame,
            text=t("자음 구분기(C/V) 보정 사용"),
            text_color="#5F8C87",
            variable=self.cvn_correction_enable_var,
            command=(
                self._on_cvn_correction_toggle
                if hasattr(self, "_on_cvn_correction_toggle")
                else self._save_config
            ),
        )

        self.cvn_low_conf_only_checkbox = self._ui_checkbox(
            ml_frame,
            text=t("정렬 저신뢰 샘플에만 C/V 후보정 적용"),
            text_color=PALETTE.neutral_text,
            variable=self.cvn_low_conf_only_var,
            command=self._save_config,
            padx=34,
            pady=(0, 4),
        )

        self.mapping_supervised_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("매핑 모델(단조 디코딩 리스코어) 사용"),
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
            text=t("매핑 모델 모드"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        mapping_mode_menu = ctk.CTkOptionMenu(
            mapping_mode_row,
            values=(
                self._get_mapping_supervised_mode_option_labels()
                if hasattr(self, "_get_mapping_supervised_mode_option_labels")
                else ["자동(권장)", "포맷 우선", "전역 우선"]
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
            text=t("CV 파일 순서 기반 매핑 결합 사용"),
            text_color=PALETTE.neutral_text,
            variable=self.cv_order_prior_enable_var,
            command=self._save_config,
        )
        self.cv_order_prior_enable_checkbox.pack(side="left")
        ctk.CTkLabel(
            cv_order_prior_row,
            text=t("강도"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(12, 4))
        cv_order_prior_strength_entry = ctk.CTkEntry(
            cv_order_prior_row,
            width=72,
            textvariable=self.cv_order_prior_strength_var,
            placeholder_text=t("기본값"),
        )
        cv_order_prior_strength_entry.pack(side="left")
        cv_order_prior_strength_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cv_order_prior_row,
            text=t("(0.0~1.0, 높을수록 순서 prior 강화)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(8, 0))
        self.mapping_supervised_dependent_frames = [mapping_mode_row, cv_order_prior_row]

        selector_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        selector_mode_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            selector_mode_row,
            text=t("ML 보정 모드"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        self.ml_selector_mode_segment = ctk.CTkSegmentedButton(
            selector_mode_row,
            values=["상대적 보정", "+셀렉터"],
            variable=self.ml_selector_mode_var,
            command=lambda _value: self._save_config(),
        )
        self.ml_selector_mode_segment.pack(side="left", padx=(10, 0))

        route_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        route_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            route_row,
            text=t("ML 방식"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        route_menu = ctk.CTkOptionMenu(
            route_row,
            values=(
                self._get_ml_route_option_labels(no_mfa_only=False)
                if hasattr(self, "_get_ml_route_option_labels")
                else ["자동(자동 라우팅)", "No-MFA", "v1", "v2", "E2E 하이브리드(실험)"]
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
            text=t("자동=환경 기반 라우팅, No-MFA=정렬 없이 보정, v1=기존, v2=확장, E2E=실험"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        coupled_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("멜+OTO 보정 사용"),
            text_color="#5F8C87",
            variable=self.ml_coupled_enable_var,
            command=self._save_config,
        )
        coupled_enable_checkbox.pack(anchor="w", padx=12, pady=(8, 0))

        coupled_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        coupled_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            coupled_row,
            text=t("Coupled 최소 신뢰도"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        coupled_conf_entry = ctk.CTkEntry(
            coupled_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_var,
            placeholder_text=t("기본값"),
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
            text=t("Strict 제약"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_strict_constraint_var,
            command=self._save_config,
        ).pack(side="left")

        model_conf_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkCheckBox(
            model_conf_row,
            text=t("Model-aware min_conf (model_meta) 사용"),
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
            text=t("(auto=ensemble 우선)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(8, 0))

        batch_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Batch inference (coupled v2) 사용"),
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
            text=t("(권장 128~512, 최소 32)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        legacy_fallback_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("v1 fallback(lightgbm) 체인 사용"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_legacy_fallback_enable_var,
            command=self._save_config,
        )
        legacy_fallback_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        hybrid_routing_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("하이브리드 라우팅(가중치 게이트) 사용"),
            text_color="#5F8C87",
            variable=self.ml_hybrid_routing_enable_var,
            command=self._save_config,
        )
        hybrid_routing_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_toggle_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("E2E 하이브리드(실험) 활성화"),
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
            text=t("E2E 모드"),
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
            text=t("(권장: hybrid)"),
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
            text=t("자세히 보기 (경로/버전/생성일)"),
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
                text=t("찾아보기"),
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
                text=t("열기"),
                width=btn_width,
                command=lambda v=var: os.startfile(str(v.get()).strip()) if os.path.isdir(str(v.get()).strip()) else None,
            )
            _style_primary_button(open_btn)
            open_btn.pack(side="right", padx=(0, 6))
            return ent

        if hasattr(self, "ml_model_root_kr_var"):
            _model_root_row("모델 경로 (한국어)", self.ml_model_root_kr_var)
        if hasattr(self, "ml_model_root_ja_var"):
            _model_root_row("모델 경로 (일본어)", self.ml_model_root_ja_var)
        ctk.CTkLabel(
            ml_frame,
            text=t("경로는 모델 폴더(내부에 model_meta.json) 또는 상위 루트를 지정할 수 있습니다."),
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
            "VC 보정",
            "기본 UI에서는 보정 사용 여부만 선택합니다. 세부 슬라이더는 개발자 설정 ON일 때만 활성화됩니다.",
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
                text=f"환경변수: {env_key}",
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
        ctk.CTkLabel(kr_box, text=t("한국어 (KR)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            kr_box,
            text=t("VC 이웃 보정 사용"),
            variable=self.kr_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        kr_detail = ctk.CTkFrame(kr_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(kr_detail)
        _add_env_slider(
            kr_detail,
            "보정 강도(Blend)",
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
            "최대 이동(ms)",
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
            "이전 컷오프 기준 여유(ms)",
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
            "다음 오프셋 기준 여유(ms)",
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
            "최소 VC 길이(ms)",
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
        ctk.CTkLabel(ja_box, text=t("일본어 (JA)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            ja_box,
            text=t("VC 이웃 보정 사용"),
            variable=self.ja_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        ja_detail = ctk.CTkFrame(ja_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(ja_detail)
        _add_env_slider(
            ja_detail,
            "보정 강도(Blend)",
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
            "최대 이동(ms)",
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
            "이전 컷오프 기준 여유(ms)",
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
            "다음 오프셋 기준 여유(ms)",
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
            "최소 VC 길이(ms)",
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

        dev_reset_row = ctk.CTkFrame(dev_container, fg_color="transparent")
        dev_reset_row.pack(fill="x", padx=10, pady=(8, 10))
        reset_btn = ctk.CTkButton(
            dev_reset_row,
            text=t("개발자 설정 초기화"),
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
        # VC 보정값 조절 UI는 고급 설정 탭으로 통합되었습니다.
        return

    def _ensure_advanced_setting_vars(self) -> None:
        if hasattr(self, "_advanced_setting_vars_ready"):
            return
        self._advanced_setting_vars_ready = True
        defaults = {
            "enable_ml_correction_var": ("bool", True),
            "disable_lightgbm_correction_var": ("bool", True),
            "vc_correction_enable_var": ("bool", True),
            "kr_continuity_enable_var": ("bool", True),
            "cvn_correction_enable_var": ("bool", True),
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
            "mapping_strict_mode_var": ("str", "적당히 엄격(누락 행은 폴백)"),
            "mapping_supervised_mode_var": ("str", "자동(권장)"),
            "cv_order_prior_strength_var": ("str", ""),
            "kr_mapping_confidence_threshold_var": ("str", ""),
            "ml_route_var": ("str", "자동(자동 라우팅)"),
            "ml_selector_mode_var": ("str", "+셀렉터"),
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
            text=t("실행 백엔드"),
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
            text=t("(auto=ensemble 우선)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(2, 0))

        return frame

    def _build_advanced_settings_tab(self):
        tab_name = "고급 설정"
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
            text=t("핵심 옵션"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("ML 보정 사용"),
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
            text=t("연속성/파일 일관성 보정 사용"),
            text_color="#64866F",
            variable=self.vc_correction_enable_var,
            command=self._on_vc_correction_toggle,
        )
        self.kr_continuity_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        continuity_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("형식 변경 시 추천값이 자동 적용됩니다. 필요한 경우 아래 개발자 슬라이더에서 세부값만 미세 조정하세요."),
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
            text=t("후처리(연속성) 옵션"),
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cont_row = ctk.CTkFrame(post_frame, fg_color="transparent")
        cont_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            cont_row,
            text=t("연속성 offset 보정 상한 (ms)"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        cont_entry = ctk.CTkEntry(
            cont_row,
            width=90,
            textvariable=self.kr_continuity_max_offset_adj_var,
            placeholder_text=t("기본값(180)"),
        )
        cont_entry.pack(side="left", padx=(10, 8))
        cont_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cont_row,
            text=t("크게 잡을수록 연속성 보정이 강해지고, 작게 잡을수록 보정이 약해집니다."),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        bank_preset_frame = ctk.CTkFrame(basic_toggle_frame, fg_color="transparent")
        bank_preset_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(
            bank_preset_frame,
            text=t("보이스뱅크 튜닝 프리셋"),
            text_color=PALETTE.neutral_text,
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        bank_preset_help_label = ctk.CTkLabel(
            bank_preset_frame,
            text=t("soft/일반 뱅크용 프리셋을 한 번에 적용합니다. 민감형은 보정을 강하게, 일반은 안정적으로 동작합니다."),
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
            text=t("soft 뱅크 · 민감형"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "aggressive"),
        )
        self._style_primary_button(btn_soft_agg)
        btn_soft_agg.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_soft_cons = ctk.CTkButton(
            preset_grid,
            text=t("soft 뱅크 · 일반"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("soft", "conservative"),
        )
        self._style_primary_button(btn_soft_cons)
        btn_soft_cons.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="w")

        btn_normal_agg = ctk.CTkButton(
            preset_grid,
            text=t("일반 뱅크 · 민감형"),
            width=170,
            command=lambda: self._on_voicebank_preset_button("normal", "aggressive"),
        )
        self._style_primary_button(btn_normal_agg)
        btn_normal_agg.grid(row=1, column=0, padx=(0, 8), pady=2, sticky="w")

        btn_normal_cons = ctk.CTkButton(
            preset_grid,
            text=t("일반 뱅크 · 일반"),
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
            text=t("발음 누락 줄이기"),
            text_color="#6F819A",
            variable=self.soft_bank_mode_var,
            command=_on_pronunciation_missing_reduce_toggle,
        )
        self.soft_bank_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        soft_bank_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("숨소리/약한 발성에서 누락·공백 오검출을 줄입니다. (soft 뱅크 + 자모 대비 보정 함께 적용)"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        soft_bank_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self._bind_wraplength_to_container(basic_toggle_frame, [soft_bank_help_label], padding=64, min_wrap=260)

        self.low_rms_gain_enable_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("저볼륨 WAV 자동 증폭/정규화 (RMS 기반, 원본 보존)"),
            text_color="#6F819A",
            variable=self.low_rms_gain_enable_var,
            command=self._save_config,
        )
        self.low_rms_gain_enable_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_rms_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("정렬 단계에서만 임시 작업 복사본에 볼륨 보정을 적용합니다. 원본 WAV 파일은 절대 변경하지 않습니다."),
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
            text=t("자모 대비 강도"),
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
            text=t("음절 오매핑 차단"),
            text_color=PALETTE.neutral_text,
            width=130,
            anchor="w",
        ).pack(side="left")
        mapping_strict_menu = ctk.CTkOptionMenu(
            mapping_strict_row,
            values=(
                self._get_mapping_strict_mode_option_labels()
                if hasattr(self, "_get_mapping_strict_mode_option_labels")
                else ["완전 엄격", "적당히 엄격(누락 행은 폴백)", "off"]
            ),
            variable=self.mapping_strict_mode_var,
            width=260,
            command=lambda _v: self._save_config(),
        )
        self._style_blue_menu(mapping_strict_menu)
        mapping_strict_menu.pack(side="left", padx=(10, 0))
        mapping_strict_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("음절을 틀릴 가능성을 줄이는 대신, 자동 설정에서 스킵되는 에일리어스가 늘어날 수 있습니다."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=730,
        )
        mapping_strict_help_label.pack(anchor="w", padx=34, pady=(0, 8))
        self.low_conf_force_lock_mode_checkbox = ctk.CTkCheckBox(
            basic_toggle_frame,
            text=t("저신뢰 구간 강제 고정 모드"),
            text_color=PALETTE.neutral_text,
            variable=self.low_conf_force_lock_mode_var,
            command=self._save_config,
        )
        self.low_conf_force_lock_mode_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        low_conf_force_lock_help_label = ctk.CTkLabel(
            basic_toggle_frame,
            text=t("정렬 신뢰가 낮은 구간에서는 후보 점프를 막고 예상 위치에 고정합니다."),
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
            "ML 보정 옵션",
            "기본 보정 모드와 멜+OTO 보정을 빠르게 조절합니다.",
        )
        self.advanced_ml_public_frame = public_ml_frame

        selector_row = ctk.CTkFrame(public_ml_frame, fg_color="transparent")
        selector_row.pack(anchor="w", padx=12, pady=(2, 6), fill="x")
        ctk.CTkLabel(
            selector_row,
            text=t("ML 보정 모드"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")

        mode_code = "selector"
        if hasattr(self, "_normalize_ml_selector_mode"):
            mode_code = self._normalize_ml_selector_mode(
                self.ml_selector_mode_var.get() if hasattr(self, "ml_selector_mode_var") else "selector"
            )
        mode_label = "상대적 보정" if mode_code == "delta" else "+셀렉터"
        if hasattr(self, "ml_selector_mode_var"):
            self.ml_selector_mode_var.set(mode_label)

        self.ml_selector_mode_segment = ctk.CTkSegmentedButton(
            selector_row,
            values=["상대적 보정", "+셀렉터"],
            variable=self.ml_selector_mode_var,
            command=lambda _value: self._save_config(),
        )
        self.ml_selector_mode_segment.pack(side="left", padx=(10, 0))

        selector_help = ctk.CTkLabel(
            public_ml_frame,
            text=t("상대적 보정=델타 기반, +셀렉터=후보 비교로 안정 보정"),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=720,
        )
        selector_help.pack(anchor="w", padx=34, pady=(0, 6))
        self._bind_wraplength_to_container(public_ml_frame, [selector_help], padding=64, min_wrap=240)

        self.ml_coupled_enable_checkbox = ctk.CTkCheckBox(
            public_ml_frame,
            text=t("멜+OTO 보정 사용"),
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

        self.disable_lightgbm_correction_checkbox = self._ui_checkbox(
            ml_frame,
            text=t("HSMM LightGBM 후처리 비활성화"),
            text_color="#9A6250",
            variable=self.disable_lightgbm_correction_var,
            command=self._save_config,
        )
        self.disable_lightgbm_correction_checkbox.pack(anchor="w", padx=12, pady=(2, 4))
        lightgbm_disable_help = ctk.CTkLabel(
            ml_frame,
            text=t("ON이면 HSMM OTO 생성 후 LightGBM 보정 단계를 건너뜁니다."),
            text_color=PALETTE.hint_text,
            justify="left",
            wraplength=720,
        )
        lightgbm_disable_help.pack(anchor="w", padx=34, pady=(0, 6))
        self._bind_wraplength_to_container(ml_frame, [lightgbm_disable_help], padding=64, min_wrap=240)

        self.cvn_correction_enable_checkbox = self._ui_checkbox(
            ml_frame,
            text=t("자음 구분기(C/V) 보정 사용"),
            text_color="#5F8C87",
            variable=self.cvn_correction_enable_var,
            command=(
                self._on_cvn_correction_toggle
                if hasattr(self, "_on_cvn_correction_toggle")
                else self._save_config
            ),
        )

        self.cvn_low_conf_only_checkbox = self._ui_checkbox(
            ml_frame,
            text=t("정렬 저신뢰 샘플에만 C/V 후보정 적용"),
            text_color=PALETTE.neutral_text,
            variable=self.cvn_low_conf_only_var,
            command=self._save_config,
            padx=34,
            pady=(0, 4),
        )

        self.mapping_supervised_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("Syllable 후보정 사용 (정렬 기반 멜 앵커)"),
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
            text=t("Syllable 후보정 모드"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        self.mapping_supervised_mode_menu = ctk.CTkOptionMenu(
            mapping_supervised_row,
            values=(
                self._get_mapping_supervised_mode_option_labels()
                if hasattr(self, "_get_mapping_supervised_mode_option_labels")
                else ["자동(권장)", "멜 앵커 중심", "mel+aic", "mel+align"]
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
            text=t("CV 순서 prior 보정 사용"),
            text_color="#5F8C87",
            variable=self.cv_order_prior_enable_var,
            command=self._save_config,
        )
        self.cv_order_prior_enable_checkbox.pack(side="left")
        ctk.CTkLabel(
            cv_order_row,
            text=t("강도"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left", padx=(10, 4))
        cv_prior_entry = ctk.CTkEntry(
            cv_order_row,
            width=70,
            textvariable=self.cv_order_prior_strength_var,
            placeholder_text=t("기본값"),
        )
        cv_prior_entry.pack(side="left", padx=(0, 6))
        cv_prior_entry.bind("<FocusOut>", lambda _e: self._save_config())

        threshold_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        threshold_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            threshold_row,
            text=t("Syllable 후보정 임계값"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        threshold_entry = ctk.CTkEntry(
            threshold_row,
            width=70,
            textvariable=self.kr_mapping_confidence_threshold_var,
            placeholder_text=t("기본값"),
        )
        threshold_entry.pack(side="left", padx=(10, 6))
        threshold_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            threshold_row,
            text=t("값이 높을수록 후보정 적용 범위를 좁힙니다."),
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
            text=t("ML 라우팅"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        route_menu = ctk.CTkOptionMenu(
            route_row,
            values=(
                self._get_ml_route_option_labels()
                if hasattr(self, "_get_ml_route_option_labels")
                else ["자동(자동 라우팅)", "No-MFA", "v1", "v2", "E2E 하이브리드(실험)"]
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
            text=t("자동=환경 기반 라우팅, No-MFA=정렬 없이 보정, v1=기존, v2=확장, E2E=실험"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        coupled_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        coupled_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            coupled_row,
            text=t("Coupled 최소 신뢰도"),
            text_color=PALETTE.neutral_text,
        ).pack(side="left")
        coupled_conf_entry = ctk.CTkEntry(
            coupled_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_var,
            placeholder_text=t("기본값"),
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
            text=t("Strict 제약"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_coupled_strict_constraint_var,
            command=self._save_config,
        ).pack(side="left")

        model_conf_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        model_conf_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkCheckBox(
            model_conf_row,
            text=t("Model-aware min_conf (model_meta) 사용"),
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
            text=t("Batch inference (coupled v2) 사용"),
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
            text=t("(권장 128~512, 최소 32)"),
            text_color=PALETTE.hint_text,
        ).pack(side="left", padx=(4, 0))

        legacy_fallback_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("v1 fallback(lightgbm) 체인 사용"),
            text_color=PALETTE.neutral_text,
            variable=self.ml_legacy_fallback_enable_var,
            command=self._save_config,
        )
        legacy_fallback_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        hybrid_routing_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("하이브리드 라우팅(가중치 게이트) 사용"),
            text_color="#5F8C87",
            variable=self.ml_hybrid_routing_enable_var,
            command=self._save_config,
        )
        hybrid_routing_checkbox.pack(anchor="w", padx=12, pady=(4, 0))

        e2e_toggle_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text=t("E2E 하이브리드(실험) 활성화"),
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
            text=t("E2E 모드"),
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
            text=t("(권장: hybrid)"),
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
            text=t("자세히 보기 (경로/버전/생성일)"),
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
                text=t("찾아보기"),
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
                text=t("열기"),
                width=btn_width,
                command=lambda v=var: os.startfile(str(v.get()).strip()) if os.path.isdir(str(v.get()).strip()) else None,
            )
            self._style_primary_button(open_btn)
            open_btn.pack(side="right", padx=(0, 6))
            return ent

        if hasattr(self, "ml_model_root_kr_var"):
            _model_root_row("모델 경로 (한국어)", self.ml_model_root_kr_var)
        if hasattr(self, "ml_model_root_ja_var"):
            _model_root_row("모델 경로 (일본어)", self.ml_model_root_ja_var)
        ctk.CTkLabel(
            ml_frame,
            text=t("경로는 모델 폴더(내부에 model_meta.json) 또는 상위 루트를 지정할 수 있습니다."),
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
            "VC 보정",
            "기본 UI에서는 보정 사용 여부만 선택합니다. 세부 슬라이더는 개발자 설정 ON일 때만 활성화됩니다.",
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
                text=f"환경변수: {env_key}",
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
        ctk.CTkLabel(kr_box, text=t("한국어 (KR)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            kr_box,
            text=t("VC 이웃 보정 사용"),
            variable=self.kr_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        kr_detail = ctk.CTkFrame(kr_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(kr_detail)
        _add_env_slider(
            kr_detail,
            "보정 강도(Blend)",
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
            "최대 이동(ms)",
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
            "이전 컷오프 기준 여유(ms)",
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
            "다음 오프셋 기준 여유(ms)",
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
            "최소 VC 길이(ms)",
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
        ctk.CTkLabel(ja_box, text=t("일본어 (JA)"), font=("", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkCheckBox(
            ja_box,
            text=t("VC 이웃 보정 사용"),
            variable=self.ja_vc_neighbor_enable_var,
            command=self._on_vc_neighbor_language_toggle,
        ).pack(anchor="w", padx=15, pady=(0, 4))
        ja_detail = ctk.CTkFrame(ja_box, fg_color="transparent")
        self.vc_neighbor_detail_frames.append(ja_detail)
        _add_env_slider(
            ja_detail,
            "보정 강도(Blend)",
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
            "최대 이동(ms)",
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
            "이전 컷오프 기준 여유(ms)",
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
            "다음 오프셋 기준 여유(ms)",
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
            "최소 VC 길이(ms)",
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

        dev_reset_row = ctk.CTkFrame(dev_container, fg_color="transparent")
        dev_reset_row.pack(fill="x", padx=10, pady=(8, 10))
        reset_btn = ctk.CTkButton(
            dev_reset_row,
            text=t("개발자 설정 초기화"),
            width=170,
            height=28,
            fg_color=PALETTE.danger_button_bg,
            hover_color=PALETTE.danger_button_hover,
            text_color=PALETTE.primary_button_text,
            command=lambda: self._reset_developer_settings_defaults(),
        )
        reset_btn.pack(side="left")
        return vc_frame


