# -*- coding: utf-8 -*-
import os

import customtkinter as ctk

from core.oto_generator import DEFAULT_PARAMS
from ui.theme_tokens import PALETTE


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


class TabBuildersMixin:
    def _build_pipeline_tab(self):
        tab = self.tabview.add("파이프라인")
        root = ctk.CTkFrame(tab)
        root.pack(fill="both", expand=True, padx=5, pady=5)

        content = ctk.CTkScrollableFrame(root)
        content.pack(side="top", fill="both", expand=True, padx=0, pady=0)

        action_parent = getattr(self, "pipeline_action_host", root)
        if action_parent is root:
            action_panel = ctk.CTkFrame(root)
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

        mfa_frame = ctk.CTkFrame(content, border_width=1, border_color="#444")
        mfa_frame.pack(fill="x", padx=10, pady=(5, 10))
        mfa_inner = ctk.CTkFrame(mfa_frame, fg_color="transparent")
        mfa_inner.pack(fill="x", padx=10, pady=8)

        status_group = ctk.CTkFrame(mfa_inner, fg_color="transparent")
        status_group.pack(side="left")

        if self.mfa_path:
            self.mfa_status_label = ctk.CTkLabel(
                status_group, text="✅ MFA 설치됨", font=("", 13, "bold"), text_color="#66BB6A"
            )
        else:
            self.mfa_status_label = ctk.CTkLabel(
                status_group, text="❌ MFA 미설치", font=("", 13, "bold"), text_color="#FF6B6B"
            )
        self.mfa_status_label.pack(side="left")

        ctk.CTkLabel(
            content,
            text=(
                "처음 설치나 'MFA 진단/복구'는 환경 구성과 현재 언어 모델 다운로드 때문에 10~20분 이상 걸릴 수 있습니다.\n"
                "문제가 생기면 먼저 'MFA 진단/복구'를 눌러 자동 복구를 시도한 뒤 정렬을 다시 실행하세요."
            ),
            text_color="gray",
            wraplength=760,
            justify="left",
        ).pack(fill="x", padx=20, pady=(0, 8))

        steps = [
            ("1. Lab 생성", "WAV 파일에서 라벨(Lab) 파일을 생성합니다.", self._run_lab_gen),
            ("2. 사전(Dictionary) 생성", "Lab 기반으로 발음 사전 파일을 생성합니다.", self._run_dict_gen),
            ("3. 음성 정렬 (MFA)", "MFA로 TextGrid를 생성합니다. MFA가 없으면 자동 설치 후 계속 진행합니다.", self._run_mfa),
            ("4. OTO.ini 생성", "TextGrid 기반으로 OTO 파라미터를 계산해 저장합니다.", self._run_oto_gen),
        ]

        ctk.CTkLabel(
            left_actions,
            text="실행",
            font=("", 13, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", pady=(0, 4))

        mfa_btn_row = ctk.CTkFrame(left_actions, fg_color="transparent")
        mfa_btn_row.pack(anchor="w")
        mfa_btn_row.grid_columnconfigure(0, weight=1)
        mfa_btn_row.grid_columnconfigure(1, weight=1)

        self.mfa_repair_btn = ctk.CTkButton(
            mfa_btn_row,
            text="🔍 MFA 진단/복구",
            width=160,
            fg_color="#B0BEC5",
            hover_color="#90A4AE",
            text_color="black",
            command=self._run_mfa_diagnose_repair,
        )
        self.mfa_repair_btn.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="w")

        self.mfa_install_btn = ctk.CTkButton(
            mfa_btn_row,
            text="⬇ MFA 원클릭 설치",
            width=160,
            fg_color="#FFA726",
            hover_color="#FB8C00",
            text_color="black",
            command=self._run_mfa_setup,
        )
        if self.mfa_path:
            self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
        self.mfa_install_btn.grid(row=0, column=1, padx=(6, 0), pady=2, sticky="w")

        ctk.CTkLabel(
            right_actions,
            text="파이프라인 단계",
            font=("", 13, "bold"),
            text_color=PALETTE.neutral_text,
        ).pack(anchor="w", pady=(0, 4))

        steps_grid = ctk.CTkFrame(right_actions, fg_color="transparent")
        steps_grid.pack(anchor="w")
        steps_grid.grid_columnconfigure(0, weight=1)
        steps_grid.grid_columnconfigure(1, weight=1)

        for idx, (title, _desc, cmd) in enumerate(steps):
            short = title.split(".", 1)[-1].strip()
            ctk.CTkButton(
                steps_grid,
                text=short,
                width=160,
                command=cmd,
            ).grid(row=idx // 2, column=idx % 2, padx=(0, 8), pady=2, sticky="w")

        for title, desc, cmd in steps:
            frame = ctk.CTkFrame(content)
            frame.pack(fill="x", padx=10, pady=5)
            left = ctk.CTkFrame(frame, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            title_label = ctk.CTkLabel(left, text=title, font=("", 14, "bold"), anchor="w")
            title_label.pack(anchor="w")
            desc_label = ctk.CTkLabel(left, text=desc, text_color="gray", anchor="w", wraplength=500)
            desc_label.pack(anchor="w")
            if title.startswith("3. 음성 정렬"):
                self.align_step_title_label = title_label
                self.align_step_desc_label = desc_label

            if "OTO" in title:
                opt_frame = ctk.CTkFrame(left, fg_color="transparent")
                opt_frame.pack(fill="x", pady=(5, 0))
                ctk.CTkCheckBox(
                    opt_frame,
                    text="OpenUtau 호환 별도 에일리어스 자동 생성",
                    text_color="#90CAF9",
                    variable=self.openutau_var,
                    command=self._save_config,
                ).pack(anchor="w")

                self.gen_missing_vowels_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text="누락된 모음/모음열(VV) 에일리어스 보완 생성",
                    text_color="#A5D6A7",
                    variable=self.gen_missing_vowels_var,
                    command=self._save_config,
                )
                self.gen_missing_vowels_checkbox.pack(anchor="w", pady=(5, 0))

                ctk.CTkLabel(
                    opt_frame,
                    text="ML 보정 옵션은 '고급 설정' 탭에서 변경할 수 있습니다.",
                    text_color="#9E9E9E",
                ).pack(anchor="w", pady=(6, 0))
        if hasattr(self, "_sync_aligner_ui"):
            self._sync_aligner_ui()

    def _build_params_tab(self):
        tab = self.tabview.add("파라미터")
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.param_vars = {}
        param_groups = [
            (
                "기본 VC 파라미터",
                [
                    ("VC_CONSONANT_RATIO", "VC 자음 구간 비율", 0.1, 1.0, 0.05),
                    ("VC_VOWEL_START", "VC 모음 시작 비율", 0.1, 1.0, 0.05),
                    ("VC_PRE_OFFSET", "VC 선행발음 오프셋 (ms)", 0, 50, 1),
                    ("VC_OVL_RATIO", "VC 오버랩 비율", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "기본 CV 파라미터",
                [
                    ("CV_PRE_RATIO", "CV 선행발음 비율", 0.1, 1.0, 0.05),
                    ("CV_OVL_RATIO", "CV 오버랩 비율", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "이중모음 CV 파라미터",
                [
                    ("DIPHTHONG_CV_PRE_RATIO", "이중모음 CV 선행발음 비율", 0.1, 1.0, 0.05),
                    ("DIPHTHONG_CV_CONSONANT_RATIO", "이중모음 CV 자음 비율", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "이중모음 VC 파라미터",
                [
                    ("DIPHTHONG_VC_VOWEL_START", "이중모음 VC 모음 시작", 0.1, 1.0, 0.05),
                    ("DIPHTHONG_VC_CONSONANT", "이중모음 VC 자음 비율", 0.1, 1.0, 0.05),
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
        tab = self.tabview.add("로그")
        self.log_text = ctk.CTkTextbox(tab, font=("Consolas", 12), state="normal")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(btn_frame, text="로그 지우기", width=100, command=self._clear_log).pack(side="left")
        ctk.CTkButton(
            btn_frame,
            text="로그 파일 열기",
            width=120,
            command=lambda: os.startfile(self.log_path) if os.path.exists(self.log_path) else None,
        ).pack(side="left", padx=5)

    def _build_advanced_settings_tab(self):
        tab = self.tabview.add("고급 설정")
        container = ctk.CTkScrollableFrame(tab)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            container,
            text="고급 설정은 기본 사용에 필요하지 않습니다. 초보자는 MFA만 사용하는 것을 권장합니다.",
            text_color="gray",
            wraplength=760,
            justify="left",
        ).pack(fill="x", padx=10, pady=(8, 12))

        ml_frame = ctk.CTkFrame(container)
        ml_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            ml_frame,
            text="ML 보정 옵션",
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text="ML 보정 사용 (기본 ON)",
            text_color="#FFD54F",
            variable=self.enable_ml_correction_var,
            command=self._save_config,
        )
        self.enable_ml_correction_checkbox.pack(anchor="w", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            ml_frame,
            text="OFF 시 ML 보정(legacy/autofree) 단계를 건너뜁니다.",
            text_color="#9E9E9E",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        selector_mode_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        selector_mode_row.pack(anchor="w", padx=12, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            selector_mode_row,
            text="ML 보정 모드",
            text_color="#B0BEC5",
        ).pack(side="left")
        self.ml_selector_mode_segment = ctk.CTkSegmentedButton(
            selector_mode_row,
            values=["기본 정책", "델타만", "델타+셀렉터"],
            variable=self.ml_selector_mode_var,
            command=lambda _value: self._save_config(),
        )
        self.ml_selector_mode_segment.pack(side="left", padx=(10, 0))

        route_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        route_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            route_row,
            text="ML route",
            text_color="#B0BEC5",
        ).pack(side="left")
        route_menu = ctk.CTkOptionMenu(
            route_row,
            values=["legacy", "autofree_v1"],
            variable=self.ml_route_var,
            width=140,
            command=lambda _v: self._save_config(),
        )
        _style_blue_menu(route_menu)
        route_menu.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(
            route_row,
            text="(legacy / autofree_v1)",
            text_color="#9E9E9E",
        ).pack(side="left", padx=(4, 0))

        coupled_enable_checkbox = ctk.CTkCheckBox(
            ml_frame,
            text="Coupled mel+oto 보정 사용",
            text_color="#80CBC4",
            variable=self.ml_coupled_enable_var,
            command=self._save_config,
        )
        coupled_enable_checkbox.pack(anchor="w", padx=12, pady=(8, 0))

        coupled_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        coupled_row.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
        ctk.CTkLabel(
            coupled_row,
            text="Coupled 최소 신뢰도",
            text_color="#B0BEC5",
        ).pack(side="left")
        coupled_conf_entry = ctk.CTkEntry(
            coupled_row,
            width=70,
            textvariable=self.ml_coupled_min_conf_var,
            placeholder_text="기본값",
        )
        coupled_conf_entry.pack(side="left", padx=(10, 8))
        coupled_conf_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            coupled_row,
            text="device",
            text_color="#B0BEC5",
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
            text="Strict 제약",
            text_color="#B0BEC5",
            variable=self.ml_coupled_strict_constraint_var,
            command=self._save_config,
        ).pack(side="left")

        backend_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        backend_row.pack(anchor="w", padx=12, pady=(6, 8), fill="x")
        ctk.CTkLabel(
            backend_row,
            text="Coupled backend",
            text_color="#B0BEC5",
        ).pack(side="left")
        backend_menu = ctk.CTkOptionMenu(
            backend_row,
            values=["auto", "ensemble", "v1", "v2"],
            variable=self.ml_coupled_backend_var,
            width=90,
            command=lambda _v: self._on_ml_backend_change(),
        )
        _style_blue_menu(backend_menu)
        backend_menu.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            backend_row,
            text="(auto=ensemble->v2->v1, v2=mel raw encoder)",
            text_color="#9E9E9E",
        ).pack(side="left", padx=(8, 0))

        gamma_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        gamma_row.pack(anchor="w", padx=12, pady=(2, 6), fill="x")
        ctk.CTkLabel(
            gamma_row,
            text="Anchor mel gamma",
            text_color="#B0BEC5",
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
            text_color="#9E9E9E",
        ).pack(side="left", padx=(4, 0))

        detail_row = ctk.CTkFrame(ml_frame, fg_color="transparent")
        detail_row.pack(anchor="w", padx=12, pady=(2, 0), fill="x")
        if not hasattr(self, "ml_coupled_status_detail_var"):
            self.ml_coupled_status_detail_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            detail_row,
            text="자세히 보기 (경로/버전/생성일)",
            text_color="#B0BEC5",
            variable=self.ml_coupled_status_detail_var,
            command=lambda: self._on_ml_backend_detail_toggle(),
        ).pack(side="left")

        def _model_root_row(label, var):
            row = ctk.CTkFrame(ml_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(6, 0))
            ctk.CTkLabel(row, text=label, width=150, anchor="w", text_color="#B0BEC5").pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=var, width=360)
            ent.pack(side="left", fill="x", expand=True, padx=(5, 6))
            ent.bind("<FocusOut>", lambda _e: self._on_ml_model_root_change())
            btn_row = ctk.CTkFrame(row, fg_color="transparent")
            btn_row.pack(side="right", padx=(6, 0))
            btn_width = 90
            ctk.CTkButton(
                btn_row,
                text="찾아보기",
                width=btn_width,
                command=lambda v=var: (
                    self._browse_folder_by_var(v, initial_dir=self._preferred_ml_model_browse_dir(v)),
                    self._on_ml_model_root_change(),
                ),
            ).pack(side="right")
            ctk.CTkButton(
                btn_row,
                text="열기",
                width=btn_width,
                command=lambda v=var: os.startfile(str(v.get()).strip()) if os.path.isdir(str(v.get()).strip()) else None,
            ).pack(side="right", padx=(0, 6))
            return ent

        if hasattr(self, "ml_model_root_kr_var"):
            _model_root_row("모델 경로 (한국어)", self.ml_model_root_kr_var)
        if hasattr(self, "ml_model_root_ja_var"):
            _model_root_row("모델 경로 (일본어)", self.ml_model_root_ja_var)
        ctk.CTkLabel(
            ml_frame,
            text="경로는 모델 폴더(내부에 model_meta.json) 또는 상위 루트를 지정할 수 있습니다.",
            text_color="#9E9E9E",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        self.ml_coupled_status_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color="#9E9E9E",
            wraplength=720,
            justify="left",
        )
        self.ml_coupled_status_label.pack(anchor="w", padx=12, pady=(4, 8))
        self.ml_coupled_status_detail_label = ctk.CTkLabel(
            ml_frame,
            text="",
            text_color="#9E9E9E",
            wraplength=760,
            justify="left",
        )
        self.ml_coupled_status_detail_label.pack(anchor="w", padx=12, pady=(0, 8))
        if hasattr(self, "_refresh_ml_backend_status"):
            self._refresh_ml_backend_status()

        post_frame = ctk.CTkFrame(container)
        post_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            post_frame,
            text="후처리(연속성) 옵션",
            font=("", 14, "bold"),
            text_color=PALETTE.header_accent,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cont_row = ctk.CTkFrame(post_frame, fg_color="transparent")
        cont_row.pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        ctk.CTkLabel(
            cont_row,
            text="연속성 offset 보정 상한 (ms)",
            text_color="#B0BEC5",
        ).pack(side="left")
        cont_entry = ctk.CTkEntry(
            cont_row,
            width=90,
            textvariable=self.kr_continuity_max_offset_adj_var,
            placeholder_text="기본값(180)",
        )
        cont_entry.pack(side="left", padx=(10, 8))
        cont_entry.bind("<FocusOut>", lambda _e: self._save_config())
        ctk.CTkLabel(
            cont_row,
            text="크게 잡을수록 연속성 보정이 강해지고, 작게 잡을수록 보정이 약해집니다.",
            text_color="#9E9E9E",
        ).pack(side="left", padx=(4, 0))

        aligner_frame = ctk.CTkFrame(container)
        aligner_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            aligner_frame,
            text="고급 정렬 엔진 옵션은 현재 제공하지 않습니다.",
            text_color="gray",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(10, 10))

    def _build_profile_tune_tab(self):
        tab = self.tabview.add("🎯 프로파일 미세 조정")
        self.tune_auto_oto_var = ctk.StringVar(value="")
        self.tune_manual_oto_var = ctk.StringVar(value="")
        self.tune_profile_out_var = ctk.StringVar(value="")
        self.tune_apply_target_var = ctk.StringVar(value="")

        container = ctk.CTkScrollableFrame(tab)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        desc = "자동 생성 OTO와 수동 OTO를 비교해 보정 프로파일을 만들고, 다른 OTO에 적용할 수 있습니다."
        ctk.CTkLabel(container, text=desc, text_color="gray", wraplength=760, justify="left").pack(
            fill="x", padx=10, pady=(8, 12)
        )

        def _row(parent, label, var, browse_cmd):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(row, text=label, width=170, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True, padx=(5, 5))
            ctk.CTkButton(row, text="찾아보기", width=90, command=browse_cmd).pack(side="right")
            return ent

        _row(
            container,
            "자동 OTO (.ini):",
            self.tune_auto_oto_var,
            lambda: self._browse_file_by_var(self.tune_auto_oto_var, [("OTO 파일", "*.ini"), ("All", "*.*")]),
        )
        _row(
            container,
            "수동 OTO (.ini):",
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

        tip = "프로파일 생성만 할 경우 적용 대상 OTO는 비워도 됩니다. 적용 대상을 지정하면 바로 보정 적용까지 수행합니다."
        ctk.CTkLabel(container, text=tip, text_color="#9E9E9E", wraplength=760, justify="left").pack(
            fill="x", padx=10, pady=(8, 12)
        )

        btn_row = ctk.CTkFrame(container, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkButton(
            btn_row,
            text="프로파일 생성 + 적용",
            height=38,
            font=("", 14, "bold"),
            command=self._run_profile_finetune,
        ).pack(side="right")
