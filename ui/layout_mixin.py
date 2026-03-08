import customtkinter as ctk


class LayoutMixin:
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
        ctk.CTkLabel(lang_frame, text=f"Auto OTO {self.app_version}", font=("", 16, "bold"), text_color="#64B5F6").pack(side="left")
        
        self.lang_var = ctk.StringVar(value="Korean (한국어)") # Changed from self.language_var to self.lang_var
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_frame, values=["Korean (한국어)", "Japanese (日本語)"], variable=self.lang_var,
            command=self._on_language_change, width=150 # Changed command to _on_language_change
        )
        self.lang_dropdown.pack(side="left", padx=10)
        self.lang_info_label = ctk.CTkLabel(lang_frame, text="", text_color="gray")
        self.lang_info_label.pack(side="left", padx=10)

        # 고급 옵션(특수 발음/접미사) - 기본 접힘
        self.advanced_toggle_btn = ctk.CTkButton(
            path_frame,
            text="▶ 고급 옵션 (특수 발음/접미사)",
            width=260,
            fg_color="transparent",
            border_width=1,
            command=self._toggle_advanced_options,
        )
        self.advanced_toggle_btn.pack(anchor="w", padx=10, pady=(3, 2))

        self.advanced_options_frame = ctk.CTkFrame(path_frame, fg_color="transparent")

        row0 = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0, text="특수 발음 (Option):", width=120, anchor="w").pack(side="left")
        self.custom_entry = ctk.CTkEntry(row0, placeholder_text="커스텀 매핑 규칙 파일 (.txt) (선택)", textvariable=self.custom_phoneme_var)
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row0, text="찾아보기", width=90, command=lambda: self._browse_file(self.custom_entry, [("Text 파일", "*.txt")])).pack(side="right")

        row0b = ctk.CTkFrame(self.advanced_options_frame, fg_color="transparent")
        row0b.pack(fill="x", padx=0, pady=3)
        ctk.CTkLabel(row0b, text="접미사 (Option):", width=120, anchor="w").pack(side="left")
        self.suffix_entry = ctk.CTkEntry(row0b, placeholder_text="예: C4 (모든 에일리어스 끝에 _C4 형태로 부여)", textvariable=self.alias_suffix_var)
        self.suffix_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))

        self._toggle_advanced_options(force=False)

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
        FORMAT_OPTIONS = ["자동 감지 (권장)", "CV/연단음", "CVVC", "VCV (연속음)"]
        self.format_dropdown = ctk.CTkOptionMenu(
            row_format, values=FORMAT_OPTIONS, variable=self.auto_format_var,
            width=200, command=self._save_config
        )
        self.format_dropdown.pack(side="left", padx=(5, 5))
        ctk.CTkLabel(row_format, text="(※ 선택 시 템플릿이 있어도 해당 형식으로 처리, 미선택 시 자동 감지)", text_color="gray").pack(side="left", padx=10)

        row_alias_style = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_alias_style.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_alias_style, text="JP 에일리어스:", width=120, anchor="w").pack(side="left")
        self.ja_alias_style_menu = ctk.CTkOptionMenu(
            row_alias_style,
            values=["원본 그대로", "히라가나", "로마자"],
            variable=self.ja_alias_style_var,
            width=200,
            command=lambda _v: self._save_config(),
        )
        self.ja_alias_style_menu.pack(side="left", padx=(5, 5))
        ctk.CTkLabel(
            row_alias_style,
            text="(일본어 OTO 생성 시 적용됩니다)",
            text_color="gray",
        ).pack(side="left", padx=10)

        row_aligner = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_aligner.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_aligner, text="정렬 엔진:", width=120, anchor="w").pack(side="left")
        self.aligner_menu = ctk.CTkOptionMenu(
            row_aligner,
            values=["MFA", "SOFA"],
            variable=self.aligner_var,
            width=200,
            command=lambda _v: self._save_config(),
        )
        self.aligner_menu.pack(side="left", padx=(5, 5))
        ctk.CTkLabel(
            row_aligner,
            text="(MFA/SOFA 중 하나를 선택해 TextGrid를 생성합니다.)",
            text_color="gray",
        ).pack(side="left", padx=10)

        row_mfa_profile = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_mfa_profile.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_mfa_profile, text="MFA 정렬 프로필", width=120, anchor="w").pack(side="left")
        self.mfa_align_profile_menu = ctk.CTkOptionMenu(
            row_mfa_profile,
            values=["정확도 우선 (기본)", "빠름 (저사양 추천)"],
            variable=self.mfa_align_profile_var,
            width=200,
            command=lambda _v: self._save_config(),
        )
        self.mfa_align_profile_menu.pack(side="left", padx=(5, 5))
        ctk.CTkLabel(
            row_mfa_profile,
            text="(기본은 기존 설정 유지, 느린 환경에서는 빠름 권장)",
            text_color="gray",
        ).pack(side="left", padx=10)

        row_sofa_ckpt = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_sofa_ckpt.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_sofa_ckpt, text="SOFA 체크포인트:", width=120, anchor="w").pack(side="left")
        self.sofa_ckpt_entry = ctk.CTkEntry(
            row_sofa_ckpt,
            placeholder_text="SOFA 모델 체크포인트 파일 (.ckpt)",
            textvariable=self.sofa_ckpt_var,
        )
        self.sofa_ckpt_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(
            row_sofa_ckpt,
            text="자동 다운로드",
            width=110,
            command=self._download_sofa_model_for_current_language,
        ).pack(side="right", padx=(0, 5))
        ctk.CTkButton(
            row_sofa_ckpt,
            text="찾아보기",
            width=90,
            command=lambda: self._browse_file_by_var(
                self.sofa_ckpt_var, [("Checkpoint", "*.ckpt"), ("All", "*.*")]
            ),
        ).pack(side="right")

        row_sofa_dict = ctk.CTkFrame(path_frame, fg_color="transparent")
        row_sofa_dict.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_sofa_dict, text="SOFA 사전(dict.txt):", width=120, anchor="w").pack(side="left")
        self.sofa_dict_entry = ctk.CTkEntry(
            row_sofa_dict,
            placeholder_text="SOFA dictionary 파일 경로",
            textvariable=self.sofa_dict_var,
        )
        self.sofa_dict_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(
            row_sofa_dict,
            text="찾아보기",
            width=90,
            command=lambda: self._browse_file_by_var(
                self.sofa_dict_var, [("Dictionary", "*.txt"), ("All", "*.*")]
            ),
        ).pack(side="right")

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
            if hasattr(self, "ja_alias_style_menu"):
                self.ja_alias_style_menu.configure(state="disabled")
        else:
            self.lang_info_label.configure(text="일본어 음소(a, k, ka 등) 자동 파싱 및 생성")
            self.gen_missing_vowels_checkbox.configure(state="normal")
            if hasattr(self, "ja_alias_style_menu"):
                self.ja_alias_style_menu.configure(state="normal")
        self._save_config()

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
        return "accurate"

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

