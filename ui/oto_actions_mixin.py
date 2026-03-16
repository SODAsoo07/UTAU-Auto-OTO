import os

from core.ja_oto_generator import generate_ja_oto
from core.oto_generator import generate_oto


class OtoActionsMixin:
    def _run_oto_gen(self):
        def task():
            self._set_running(True)
            self._set_status("4단계 OTO.ini 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()

                if not wav_dir:
                    self._append_log("오류: WAV 폴더를 입력해 주세요.")
                    return
                if not out_path:
                    self._append_log("오류: 출력 경로를 입력해 주세요.")
                    return

                cleanup_snapshot = self._snapshot_output_tree_for_cleanup(out_path)
                tg_folder = os.path.join(wav_dir, "textgrids")
                if not os.path.exists(tg_folder):
                    self._append_log("경고: textgrids 폴더가 없습니다. 3단계 정렬/라벨 생성을 먼저 실행하세요.")

                params = self._get_params()
                gen_ou = self.openutau_var.get()
                gen_missing = self.gen_missing_vowels_var.get()
                gen_dash_alias = (
                    self.gen_dash_alias_var.get()
                    if hasattr(self, "gen_dash_alias_var")
                    else True
                )
                enable_ml_correction = self.enable_ml_correction_var.get()
                auto_format = self.auto_format_var.get()
                lang = self._get_language()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                alias_suffix = self.alias_suffix_var.get().strip()
                ja_alias_style = self._get_ja_alias_style_code()
                ja_words_fallback = (
                    self.ja_mapping_words_fallback_enabled_var.get()
                    if hasattr(self, "ja_mapping_words_fallback_enabled_var")
                    else True
                )
                ja_spn_threshold = (
                    self.ja_mapping_spn_ratio_threshold_var.get()
                    if hasattr(self, "ja_mapping_spn_ratio_threshold_var")
                    else 0.35
                )
                ja_min_vowel_ratio = (
                    self.ja_mapping_min_vowel_phone_ratio_var.get()
                    if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var")
                    else 0.5
                )
                ja_debug_reason = (
                    self.ja_mapping_debug_reason_logging_var.get()
                    if hasattr(self, "ja_mapping_debug_reason_logging_var")
                    else True
                )

                kr_anchor_profile_path = (
                    self.kr_anchor_profile_path_var.get().strip()
                    if hasattr(self, "kr_anchor_profile_path_var")
                    else ""
                )
                kr_conf_raw = (
                    self.kr_mapping_confidence_threshold_var.get()
                    if hasattr(self, "kr_mapping_confidence_threshold_var")
                    else ""
                )
                kr_jump_default = (
                    self.kr_mapping_max_index_jump_default_var.get()
                    if hasattr(self, "kr_mapping_max_index_jump_default_var")
                    else 1
                )
                kr_jump_hi = (
                    self.kr_mapping_max_index_jump_high_conf_var.get()
                    if hasattr(self, "kr_mapping_max_index_jump_high_conf_var")
                    else 2
                )
                kr_continuity_max_offset_adj_raw = (
                    self.kr_continuity_max_offset_adj_var.get()
                    if hasattr(self, "kr_continuity_max_offset_adj_var")
                    else ""
                )
                kr_vc_neighbor_enable = (
                    bool(self.kr_vc_neighbor_enable_var.get())
                    if hasattr(self, "kr_vc_neighbor_enable_var")
                    else True
                )
                ja_vc_neighbor_enable = (
                    bool(self.ja_vc_neighbor_enable_var.get())
                    if hasattr(self, "ja_vc_neighbor_enable_var")
                    else True
                )
                kr_vc_neighbor_blend_raw = (
                    self.kr_vc_neighbor_blend_var.get()
                    if hasattr(self, "kr_vc_neighbor_blend_var")
                    else ""
                )
                kr_vc_neighbor_max_shift_raw = (
                    self.kr_vc_neighbor_max_shift_var.get()
                    if hasattr(self, "kr_vc_neighbor_max_shift_var")
                    else ""
                )
                kr_vc_neighbor_lead_ms_raw = (
                    self.kr_vc_neighbor_lead_ms_var.get()
                    if hasattr(self, "kr_vc_neighbor_lead_ms_var")
                    else ""
                )
                kr_vc_neighbor_tail_ms_raw = (
                    self.kr_vc_neighbor_tail_ms_var.get()
                    if hasattr(self, "kr_vc_neighbor_tail_ms_var")
                    else ""
                )
                kr_vc_neighbor_min_len_raw = (
                    self.kr_vc_neighbor_min_len_var.get()
                    if hasattr(self, "kr_vc_neighbor_min_len_var")
                    else ""
                )
                ja_vc_neighbor_blend_raw = (
                    self.ja_vc_neighbor_blend_var.get()
                    if hasattr(self, "ja_vc_neighbor_blend_var")
                    else ""
                )
                ja_vc_neighbor_max_shift_raw = (
                    self.ja_vc_neighbor_max_shift_var.get()
                    if hasattr(self, "ja_vc_neighbor_max_shift_var")
                    else ""
                )
                ja_vc_neighbor_lead_ms_raw = (
                    self.ja_vc_neighbor_lead_ms_var.get()
                    if hasattr(self, "ja_vc_neighbor_lead_ms_var")
                    else ""
                )
                ja_vc_neighbor_tail_ms_raw = (
                    self.ja_vc_neighbor_tail_ms_var.get()
                    if hasattr(self, "ja_vc_neighbor_tail_ms_var")
                    else ""
                )
                ja_vc_neighbor_min_len_raw = (
                    self.ja_vc_neighbor_min_len_var.get()
                    if hasattr(self, "ja_vc_neighbor_min_len_var")
                    else ""
                )
                ml_anchor_mel_gamma_raw = (
                    self.ml_anchor_mel_gamma_var.get()
                    if hasattr(self, "ml_anchor_mel_gamma_var")
                    else ""
                )
                ml_mel_weight_mode = (
                    self.ml_mel_weight_mode_var.get()
                    if hasattr(self, "ml_mel_weight_mode_var")
                    else "auto"
                )
                ml_same_lang_only = (
                    self.ml_same_language_borrow_only_var.get()
                    if hasattr(self, "ml_same_language_borrow_only_var")
                    else True
                )
                ml_selector_mode = (
                    self.ml_selector_mode_var.get()
                    if hasattr(self, "ml_selector_mode_var")
                    else "기본 정책"
                )
                ml_coupled_enable = (
                    self.ml_coupled_enable_var.get()
                    if hasattr(self, "ml_coupled_enable_var")
                    else True
                )
                ml_coupled_min_conf_raw = (
                    self.ml_coupled_min_conf_var.get()
                    if hasattr(self, "ml_coupled_min_conf_var")
                    else ""
                )
                ml_coupled_min_conf_use_model_meta = (
                    self.ml_coupled_min_conf_use_model_meta_var.get()
                    if hasattr(self, "ml_coupled_min_conf_use_model_meta_var")
                    else True
                )
                ml_coupled_min_conf_model_offset_raw = (
                    self.ml_coupled_min_conf_model_offset_var.get()
                    if hasattr(self, "ml_coupled_min_conf_model_offset_var")
                    else ""
                )
                ml_coupled_min_conf_format_raw = {
                    "UTOA_ML_COUPLED_MIN_CONF_KR_CV": (
                        self.ml_coupled_min_conf_kr_cv_var.get()
                        if hasattr(self, "ml_coupled_min_conf_kr_cv_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_KR_CVC": (
                        self.ml_coupled_min_conf_kr_cvc_var.get()
                        if hasattr(self, "ml_coupled_min_conf_kr_cvc_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_KR_CVVC": (
                        self.ml_coupled_min_conf_kr_cvvc_var.get()
                        if hasattr(self, "ml_coupled_min_conf_kr_cvvc_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_KR_VCV": (
                        self.ml_coupled_min_conf_kr_vcv_var.get()
                        if hasattr(self, "ml_coupled_min_conf_kr_vcv_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_JA_CV": (
                        self.ml_coupled_min_conf_ja_cv_var.get()
                        if hasattr(self, "ml_coupled_min_conf_ja_cv_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_JA_CVC": (
                        self.ml_coupled_min_conf_ja_cvc_var.get()
                        if hasattr(self, "ml_coupled_min_conf_ja_cvc_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_JA_CVVC": (
                        self.ml_coupled_min_conf_ja_cvvc_var.get()
                        if hasattr(self, "ml_coupled_min_conf_ja_cvvc_var")
                        else ""
                    ),
                    "UTOA_ML_COUPLED_MIN_CONF_JA_VCV": (
                        self.ml_coupled_min_conf_ja_vcv_var.get()
                        if hasattr(self, "ml_coupled_min_conf_ja_vcv_var")
                        else ""
                    ),
                }
                ml_coupled_device = (
                    str(self.ml_coupled_device_var.get()).strip().lower()
                    if hasattr(self, "ml_coupled_device_var")
                    else "auto"
                )
                ml_coupled_backend = (
                    str(self.ml_coupled_backend_var.get()).strip().lower()
                    if hasattr(self, "ml_coupled_backend_var")
                    else "auto"
                )
                ml_route = (
                    str(self.ml_route_var.get()).strip().lower()
                    if hasattr(self, "ml_route_var")
                    else "autofree_v1"
                )
                ml_model_root = ""
                if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
                    ml_model_root = str(self.ml_model_root_ja_var.get() or "").strip()
                if lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
                    ml_model_root = str(self.ml_model_root_kr_var.get() or "").strip()
                ml_coupled_strict_constraint = (
                    self.ml_coupled_strict_constraint_var.get()
                    if hasattr(self, "ml_coupled_strict_constraint_var")
                    else False
                )
                ml_batch_inference_enable = (
                    self.ml_batch_inference_enable_var.get()
                    if hasattr(self, "ml_batch_inference_enable_var")
                    else True
                )
                ml_batch_inference_size_raw = (
                    self.ml_batch_inference_size_var.get()
                    if hasattr(self, "ml_batch_inference_size_var")
                    else "256"
                )
                ml_legacy_fallback_enable = (
                    self.ml_legacy_fallback_enable_var.get()
                    if hasattr(self, "ml_legacy_fallback_enable_var")
                    else False
                )

                def _parse_optional_threshold(raw_value, lo=0.0, hi=1.0):
                    text = str(raw_value or "").strip()
                    if not text:
                        return None
                    try:
                        return max(float(lo), min(float(hi), float(text)))
                    except Exception:
                        return None

                def _parse_optional_float(raw_value, lo=0.0, hi=None):
                    text = str(raw_value or "").strip()
                    if not text:
                        return None
                    try:
                        value = max(float(lo), float(text))
                        if hi is not None:
                            value = min(float(hi), value)
                        return value
                    except Exception:
                        return None

                kr_conf_threshold = _parse_optional_threshold(kr_conf_raw, lo=0.0, hi=1.0)
                ml_coupled_min_conf = _parse_optional_threshold(ml_coupled_min_conf_raw, lo=0.0, hi=1.0)
                ml_coupled_min_conf_model_offset = _parse_optional_float(
                    ml_coupled_min_conf_model_offset_raw,
                    lo=-0.30,
                    hi=0.30,
                )
                ml_coupled_min_conf_format = {
                    key: _parse_optional_threshold(raw_val, lo=0.0, hi=1.0)
                    for key, raw_val in ml_coupled_min_conf_format_raw.items()
                }
                kr_continuity_max_offset_adj = _parse_optional_float(kr_continuity_max_offset_adj_raw, lo=0.0, hi=2000.0)
                kr_vc_neighbor_blend = _parse_optional_float(kr_vc_neighbor_blend_raw, lo=0.0, hi=1.0)
                kr_vc_neighbor_max_shift = _parse_optional_float(kr_vc_neighbor_max_shift_raw, lo=0.0, hi=300.0)
                kr_vc_neighbor_lead_ms = _parse_optional_float(kr_vc_neighbor_lead_ms_raw, lo=0.0, hi=200.0)
                kr_vc_neighbor_tail_ms = _parse_optional_float(kr_vc_neighbor_tail_ms_raw, lo=0.0, hi=200.0)
                kr_vc_neighbor_min_len = _parse_optional_float(kr_vc_neighbor_min_len_raw, lo=0.0, hi=400.0)
                ja_vc_neighbor_blend = _parse_optional_float(ja_vc_neighbor_blend_raw, lo=0.0, hi=1.0)
                ja_vc_neighbor_max_shift = _parse_optional_float(ja_vc_neighbor_max_shift_raw, lo=0.0, hi=300.0)
                ja_vc_neighbor_lead_ms = _parse_optional_float(ja_vc_neighbor_lead_ms_raw, lo=0.0, hi=200.0)
                ja_vc_neighbor_tail_ms = _parse_optional_float(ja_vc_neighbor_tail_ms_raw, lo=0.0, hi=200.0)
                ja_vc_neighbor_min_len = _parse_optional_float(ja_vc_neighbor_min_len_raw, lo=0.0, hi=400.0)
                ml_anchor_mel_gamma = _parse_optional_float(ml_anchor_mel_gamma_raw, lo=0.1, hi=10.0)
                try:
                    ml_batch_inference_size = max(32, min(4096, int(float(str(ml_batch_inference_size_raw).strip() or "256"))))
                except Exception:
                    ml_batch_inference_size = 256

                os.environ["UTOA_ML_SAME_LANGUAGE_BORROW_ONLY"] = "1" if ml_same_lang_only else "0"
                selector_mode_code = self._apply_ml_selector_runtime_mode(ml_selector_mode)
                if ml_coupled_device not in {"auto", "cpu", "cuda"}:
                    ml_coupled_device = "auto"
                ml_coupled_backend = {
                    "ensemble_v1": "ensemble",
                }.get(ml_coupled_backend, ml_coupled_backend)
                if ml_coupled_backend not in {"auto", "ensemble"}:
                    ml_coupled_backend = "auto"
                ensemble_enabled = ml_coupled_backend in {"auto", "ensemble"}
                if kr_conf_threshold is None:
                    os.environ.pop("UTOA_KR_MAPPING_CONF_THRESHOLD", None)
                else:
                    os.environ["UTOA_KR_MAPPING_CONF_THRESHOLD"] = str(float(kr_conf_threshold))
                if kr_continuity_max_offset_adj is None:
                    os.environ.pop("UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ", None)
                else:
                    os.environ["UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ"] = str(float(kr_continuity_max_offset_adj))
                os.environ["UTOA_KR_VC_NEIGHBOR_ENABLE"] = "1" if kr_vc_neighbor_enable else "0"
                os.environ["UTOA_JA_VC_NEIGHBOR_ENABLE"] = "1" if ja_vc_neighbor_enable else "0"
                if kr_vc_neighbor_blend is None:
                    os.environ.pop("UTOA_KR_VC_NEIGHBOR_BLEND", None)
                else:
                    os.environ["UTOA_KR_VC_NEIGHBOR_BLEND"] = str(float(kr_vc_neighbor_blend))
                if kr_vc_neighbor_max_shift is None:
                    os.environ.pop("UTOA_KR_VC_NEIGHBOR_MAX_SHIFT", None)
                else:
                    os.environ["UTOA_KR_VC_NEIGHBOR_MAX_SHIFT"] = str(float(kr_vc_neighbor_max_shift))
                if kr_vc_neighbor_lead_ms is None:
                    os.environ.pop("UTOA_KR_VC_NEIGHBOR_LEAD_MS", None)
                else:
                    os.environ["UTOA_KR_VC_NEIGHBOR_LEAD_MS"] = str(float(kr_vc_neighbor_lead_ms))
                if kr_vc_neighbor_tail_ms is None:
                    os.environ.pop("UTOA_KR_VC_NEIGHBOR_TAIL_MS", None)
                else:
                    os.environ["UTOA_KR_VC_NEIGHBOR_TAIL_MS"] = str(float(kr_vc_neighbor_tail_ms))
                if kr_vc_neighbor_min_len is None:
                    os.environ.pop("UTOA_KR_VC_NEIGHBOR_MIN_LEN", None)
                else:
                    os.environ["UTOA_KR_VC_NEIGHBOR_MIN_LEN"] = str(float(kr_vc_neighbor_min_len))
                if ja_vc_neighbor_blend is None:
                    os.environ.pop("UTOA_JA_VC_NEIGHBOR_BLEND", None)
                else:
                    os.environ["UTOA_JA_VC_NEIGHBOR_BLEND"] = str(float(ja_vc_neighbor_blend))
                if ja_vc_neighbor_max_shift is None:
                    os.environ.pop("UTOA_JA_VC_NEIGHBOR_MAX_SHIFT", None)
                else:
                    os.environ["UTOA_JA_VC_NEIGHBOR_MAX_SHIFT"] = str(float(ja_vc_neighbor_max_shift))
                if ja_vc_neighbor_lead_ms is None:
                    os.environ.pop("UTOA_JA_VC_NEIGHBOR_LEAD_MS", None)
                else:
                    os.environ["UTOA_JA_VC_NEIGHBOR_LEAD_MS"] = str(float(ja_vc_neighbor_lead_ms))
                if ja_vc_neighbor_tail_ms is None:
                    os.environ.pop("UTOA_JA_VC_NEIGHBOR_TAIL_MS", None)
                else:
                    os.environ["UTOA_JA_VC_NEIGHBOR_TAIL_MS"] = str(float(ja_vc_neighbor_tail_ms))
                if ja_vc_neighbor_min_len is None:
                    os.environ.pop("UTOA_JA_VC_NEIGHBOR_MIN_LEN", None)
                else:
                    os.environ["UTOA_JA_VC_NEIGHBOR_MIN_LEN"] = str(float(ja_vc_neighbor_min_len))
                if ml_anchor_mel_gamma is None:
                    os.environ.pop("UTOA_ML_ANCHOR_MEL_GAMMA", None)
                else:
                    os.environ["UTOA_ML_ANCHOR_MEL_GAMMA"] = str(float(ml_anchor_mel_gamma))
                mel_weight_mode_code = self._apply_mel_weight_runtime_mode(ml_mel_weight_mode)
                os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_DEFAULT"] = str(int(kr_jump_default))
                os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_HIGH_CONF"] = str(int(kr_jump_hi))
                os.environ["UTOA_ML_COUPLED_ENABLE"] = "1" if ml_coupled_enable else "0"
                os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "1" if ensemble_enabled else "0"
                if ml_coupled_min_conf is None:
                    os.environ.pop("UTOA_ML_COUPLED_MIN_CONF", None)
                else:
                    os.environ["UTOA_ML_COUPLED_MIN_CONF"] = str(float(ml_coupled_min_conf))
                os.environ["UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META"] = (
                    "1" if bool(ml_coupled_min_conf_use_model_meta) else "0"
                )
                if ml_coupled_min_conf_model_offset is None:
                    os.environ.pop("UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET", None)
                else:
                    os.environ["UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET"] = str(float(ml_coupled_min_conf_model_offset))
                for env_key, env_val in ml_coupled_min_conf_format.items():
                    if env_val is None:
                        os.environ.pop(env_key, None)
                    else:
                        os.environ[env_key] = str(float(env_val))
                os.environ["UTOA_ML_COUPLED_DEVICE"] = str(ml_coupled_device)
                coupled_backend_env = {"auto": "auto", "ensemble": "auto"}.get(
                    ml_coupled_backend,
                    "auto",
                )
                os.environ["UTOA_ML_COUPLED_BACKEND"] = str(coupled_backend_env)
                os.environ["UTOA_ML_COUPLED_STRICT_CONSTRAINT"] = "1" if ml_coupled_strict_constraint else "0"
                os.environ["UTOA_ML_BATCH_INFERENCE_ENABLE"] = "1" if bool(ml_batch_inference_enable) else "0"
                os.environ["UTOA_ML_BATCH_INFERENCE_SIZE"] = str(int(ml_batch_inference_size))
                os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "1" if bool(ml_legacy_fallback_enable) else "0"
                os.environ["UTOA_ML_ROUTE"] = "autofree_v1" if ml_route == "autofree_v1" else "legacy"
                os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "1" if ml_route == "autofree_v1" else "0"
                os.environ["UTOA_ENABLE_HEAD_TAIL_DASH_ALIAS"] = "1" if bool(gen_dash_alias) else "0"
                if lang == "japanese":
                    if ml_model_root:
                        os.environ["UTOA_JA_OTO_ML_DIR"] = ml_model_root
                        os.environ["UTOA_JA_AUTOFREE_ML_DIR"] = ml_model_root
                    else:
                        os.environ.pop("UTOA_JA_OTO_ML_DIR", None)
                        os.environ.pop("UTOA_JA_AUTOFREE_ML_DIR", None)
                else:
                    if ml_model_root:
                        os.environ["UTOA_KR_OTO_ML_DIR"] = ml_model_root
                        os.environ["UTOA_KR_AUTOFREE_ML_DIR"] = ml_model_root
                    else:
                        os.environ.pop("UTOA_KR_OTO_ML_DIR", None)
                        os.environ.pop("UTOA_KR_AUTOFREE_ML_DIR", None)
                if kr_anchor_profile_path:
                    os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = kr_anchor_profile_path
                else:
                    os.environ.pop("UTOA_KR_ANCHOR_PROFILE_PATH", None)

                self._append_log(
                    f"[OTO-ML] 실행 옵션: ml={'ON' if enable_ml_correction else 'OFF'}, selector={self._describe_ml_selector_mode(selector_mode_code)}"
                )
                self._append_log(f"[OTO-ML] route={os.environ.get('UTOA_ML_ROUTE', 'autofree_v1')}")
                self._append_log(
                    f"[OTO-ML] mel_weight_mode={self._describe_mel_weight_mode(mel_weight_mode_code)}"
                )
                min_conf_display = (
                    f"{float(ml_coupled_min_conf):.2f}"
                    if ml_coupled_min_conf is not None
                    else "default(0.42, model-aware)"
                )
                self._append_log(
                    f"[OTO-ML] coupled={'ON' if ml_coupled_enable else 'OFF'}, backend={ml_coupled_backend}, ensemble={'ON' if ensemble_enabled else 'OFF'}, min_conf={min_conf_display}, device={ml_coupled_device}, strict={'ON' if ml_coupled_strict_constraint else 'OFF'}"
                )
                if bool(ml_coupled_min_conf_use_model_meta):
                    if ml_coupled_min_conf_model_offset is None:
                        self._append_log("[OTO-ML] min_conf policy: model-aware(meta) + per-format override")
                    else:
                        self._append_log(
                            f"[OTO-ML] min_conf policy: model-aware(meta, offset={float(ml_coupled_min_conf_model_offset):+.3f}) + per-format override"
                        )
                self._append_log(
                    f"[OTO-ML] runtime: batch={'ON' if bool(ml_batch_inference_enable) else 'OFF'}({int(ml_batch_inference_size)}), legacy_fallback={'ON' if bool(ml_legacy_fallback_enable) else 'OFF'}"
                )
                if ml_route == "autofree_v1":
                    self._append_log("[OTO-ML] route policy: coupled primary + autofree auxiliary(residual)")
                kr_conf_display = f"{float(kr_conf_threshold):.2f}" if kr_conf_threshold is not None else "default(by format)"
                self._append_log(f"[KR-MAP] confidence_threshold={kr_conf_display}")
                if self.no_base_oto_var.get():
                    self._append_log("설정: '기본 OTO 없이 생성' 사용 중입니다.")
                if lang == "japanese":
                    self._append_log(f"설정: 일본어 에일리어스 스타일 = {self.ja_alias_style_var.get()}")
                    processed, total, errors = generate_ja_oto(
                        tg_folder,
                        tpl_path,
                        out_path,
                        params=None,
                        generate_openutau=gen_ou,
                        gen_missing_vowels=gen_missing,
                        enable_ml_correction=enable_ml_correction,
                        alias_style=ja_alias_style,
                        ja_mapping_words_fallback_enabled=bool(ja_words_fallback),
                        ja_mapping_spn_ratio_threshold=float(ja_spn_threshold),
                        ja_mapping_min_vowel_phone_ratio=float(ja_min_vowel_ratio),
                        ja_mapping_debug_reason_logging=bool(ja_debug_reason),
                        auto_format=auto_format,
                        custom_phonemes_path=custom_phonemes_path,
                        alias_suffix=alias_suffix,
                        callback=self._append_log,
                    )
                else:
                    processed, total, errors = generate_oto(
                        tg_folder,
                        tpl_path,
                        out_path,
                        params,
                        gen_ou,
                        gen_missing,
                        enable_ml_correction=enable_ml_correction,
                        auto_format=auto_format,
                        custom_phonemes_path=custom_phonemes_path,
                        alias_suffix=alias_suffix,
                        kr_anchor_profile_path=kr_anchor_profile_path,
                        kr_mapping_confidence_threshold=kr_conf_threshold,
                        kr_mapping_max_index_jump_default=int(kr_jump_default),
                        kr_mapping_max_index_jump_high_conf=int(kr_jump_hi),
                        callback=self._append_log,
                    )

                self._run_auto_validation(wav_dir, tg_folder, out_path)
                if not errors:
                    self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)
                if errors:
                    for e in errors:
                        self._append_log(f"  - {e}")
                    self._set_status(f"오류: OTO 생성 실패 {len(errors)}건 ({processed}/{total})")
                else:
                    self._set_status(f"완료: OTO 생성 성공 ({processed}/{total})")
            except Exception as e:
                self._handle_error("OTO 생성", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
