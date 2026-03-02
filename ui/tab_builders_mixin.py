import os

import customtkinter as ctk

from core.oto_generator import DEFAULT_PARAMS


class TabBuildersMixin:
    def _build_pipeline_tab(self):
        tab = self.tabview.add("🚀 파이프라인")
        content = ctk.CTkScrollableFrame(tab)
        content.pack(fill="both", expand=True, padx=5, pady=5)

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

        self.sofa_status_label = ctk.CTkLabel(status_group, text="  |  ", text_color="gray")
        self.sofa_status_label.pack(side="left")
        if self._is_sofa_installed():
            self.sofa_status_label.configure(text="✅ SOFA 설치됨", text_color="#66BB6A")
        else:
            self.sofa_status_label.configure(text="❌ SOFA 미설치", text_color="#FF6B6B")

        self.mfa_install_btn = ctk.CTkButton(
            mfa_inner,
            text="⬇ MFA 자동 설치",
            width=150,
            fg_color="#FFA726",
            hover_color="#FB8C00",
            text_color="black",
            command=self._run_mfa_setup,
        )
        if self.mfa_path:
            self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
        self.mfa_install_btn.pack(side="right")

        self.sofa_install_btn = ctk.CTkButton(
            mfa_inner,
            text="⬇ SOFA 자동 설치",
            width=150,
            fg_color="#42A5F5",
            hover_color="#1E88E5",
            text_color="black",
            command=self._run_sofa_setup,
        )
        if self._is_sofa_installed():
            self.sofa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
        self.sofa_install_btn.pack(side="right", padx=(0, 8))

        steps = [
            ("1️⃣ Lab 파일 생성", "WAV 파일명을 기반으로 Lab 파일을 자동으로 생성합니다.", self._run_lab_gen),
            ("2️⃣ 사전(Dictionary) 생성", "Lab 파일로부터 정렬용 음소 사전 파일을 만듭니다.", self._run_dict_gen),
            ("3️⃣ 음성 정렬 (MFA/SOFA)", "선택한 정렬 엔진(MFA 또는 SOFA)으로 TextGrid를 생성합니다.", self._run_mfa),
            ("4️⃣ OTO.ini 자동 생성", "TextGrid와 튜닝 파라미터로 최종 OTO를 계산합니다.", self._run_oto_gen),
        ]

        for title, desc, cmd in steps:
            frame = ctk.CTkFrame(content)
            frame.pack(fill="x", padx=10, pady=5)
            left = ctk.CTkFrame(frame, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            ctk.CTkLabel(left, text=title, font=("", 14, "bold"), anchor="w").pack(anchor="w")
            ctk.CTkLabel(left, text=desc, text_color="gray", anchor="w", wraplength=500).pack(anchor="w")

            if "OTO" in title:
                opt_frame = ctk.CTkFrame(left, fg_color="transparent")
                opt_frame.pack(fill="x", pady=(5, 0))
                ctk.CTkCheckBox(
                    opt_frame,
                    text="OpenUtau Phonemizer 호환 에일리어스 동시 생성 (k, t, p 종성 규칙 및 - CV 연음 등 복제 추가)",
                    text_color="#90CAF9",
                    variable=self.openutau_var,
                    command=self._save_config,
                ).pack(anchor="w")

                self.gen_missing_vowels_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text="템플릿에 없더라도 기본 단모음(a, e, i, o, u...) 에일리어스 자동 생성",
                    text_color="#A5D6A7",
                    variable=self.gen_missing_vowels_var,
                    command=self._save_config,
                )
                self.gen_missing_vowels_checkbox.pack(anchor="w", pady=(5, 0))

                self.enable_ml_correction_checkbox = ctk.CTkCheckBox(
                    opt_frame,
                    text="OTO ML 수치 보정 적용 (끄면 규칙 기반 결과만 생성)",
                    text_color="#FFD54F",
                    variable=self.enable_ml_correction_var,
                    command=self._save_config,
                )
                self.enable_ml_correction_checkbox.pack(anchor="w", pady=(5, 0))

            ctk.CTkButton(frame, text="실행", width=80, command=cmd).pack(side="right", padx=10)

    def _build_params_tab(self):
        tab = self.tabview.add("🎛️ 튜닝 파라미터")
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.param_vars = {}
        param_groups = [
            (
                "일반 VC 파라미터",
                [
                    ("VC_CONSONANT_RATIO", "VC 자음 사용 비율", 0.1, 1.0, 0.05),
                    ("VC_VOWEL_START", "VC 모음 시작 위치", 0.1, 1.0, 0.05),
                    ("VC_PRE_OFFSET", "VC 선행발성 오프셋 (ms)", 0, 50, 1),
                    ("VC_OVL_RATIO", "VC 오버랩 비율", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "일반 CV 파라미터",
                [
                    ("CV_PRE_RATIO", "CV 선행발성 비율", 0.1, 1.0, 0.05),
                    ("CV_OVL_RATIO", "CV 오버랩 비율", 0.1, 1.0, 0.05),
                ],
            ),
            (
                "이중모음 CV 파라미터",
                [
                    ("DIPHTHONG_CV_PRE_RATIO", "이중모음 CV 선행발성 비율", 0.1, 1.0, 0.05),
                    ("DIPHTHONG_CV_CONSONANT_RATIO", "이중모음 CV 고정 비율", 0.1, 1.0, 0.05),
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
        tab = self.tabview.add("📋 로그")
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

    def _build_profile_tune_tab(self):
        tab = self.tabview.add("🧩 프로파일 미세조정")
        self.tune_auto_oto_var = ctk.StringVar(value="")
        self.tune_manual_oto_var = ctk.StringVar(value="")
        self.tune_profile_out_var = ctk.StringVar(value="")
        self.tune_apply_target_var = ctk.StringVar(value="")

        container = ctk.CTkScrollableFrame(tab)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        desc = "자동 생성 OTO와 수동 보정 OTO(일부만 있어도 가능)를 매칭해 델타 프로파일을 학습한 뒤, 선택한 OTO에 미세조정을 적용합니다."
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

        tip = "팁: 적용 대상 OTO를 비워두면 자동 OTO 입력 파일에 바로 적용합니다. 프로파일 경로를 비우면 자동 OTO와 같은 폴더에 저장합니다."
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
