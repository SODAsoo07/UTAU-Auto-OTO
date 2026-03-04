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

                tg_folder = os.path.join(wav_dir, "textgrids")
                if not os.path.exists(tg_folder):
                    self._append_log("경고: textgrids 폴더가 없습니다. 3단계 정렬/라벨 생성을 먼저 실행하세요.")

                params = self._get_params()
                gen_ou = self.openutau_var.get()
                gen_missing = self.gen_missing_vowels_var.get()
                enable_ml_correction = self.enable_ml_correction_var.get()
                enable_pytorch_bridge = self.enable_pytorch_bridge_var.get()
                auto_format = self.auto_format_var.get()
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
                kr_conf_threshold = (
                    self.kr_mapping_confidence_threshold_var.get()
                    if hasattr(self, "kr_mapping_confidence_threshold_var")
                    else 0.60
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
                ml_same_lang_only = (
                    self.ml_same_language_borrow_only_var.get()
                    if hasattr(self, "ml_same_language_borrow_only_var")
                    else True
                )
                ml_use_pseudo_labels = (
                    self.ml_use_pseudo_labels_var.get()
                    if hasattr(self, "ml_use_pseudo_labels_var")
                    else True
                )
                ml_pseudo_weight_high = (
                    self.ml_pseudo_weight_high_var.get()
                    if hasattr(self, "ml_pseudo_weight_high_var")
                    else 0.7
                )
                ml_pseudo_weight_mid = (
                    self.ml_pseudo_weight_mid_var.get()
                    if hasattr(self, "ml_pseudo_weight_mid_var")
                    else 0.4
                )

                os.environ["UTOA_ML_SAME_LANGUAGE_BORROW_ONLY"] = "1" if ml_same_lang_only else "0"
                os.environ["UTOA_ML_USE_PSEUDO_LABELS"] = "1" if ml_use_pseudo_labels else "0"
                os.environ["UTOA_ML_PSEUDO_WEIGHT_HIGH"] = str(float(ml_pseudo_weight_high))
                os.environ["UTOA_ML_PSEUDO_WEIGHT_MID"] = str(float(ml_pseudo_weight_mid))
                os.environ["UTOA_KR_MAPPING_CONF_THRESHOLD"] = str(float(kr_conf_threshold))
                os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_DEFAULT"] = str(int(kr_jump_default))
                os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_HIGH_CONF"] = str(int(kr_jump_hi))
                if kr_anchor_profile_path:
                    os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = kr_anchor_profile_path
                else:
                    os.environ.pop("UTOA_KR_ANCHOR_PROFILE_PATH", None)

                self._append_log(
                    f"[OTO-ML] 실행 옵션: ml={'ON' if enable_ml_correction else 'OFF'}, "
                    f"pytorch_bridge={'ON' if enable_pytorch_bridge else 'OFF'}"
                )
                if self.no_base_oto_var.get():
                    self._append_log("설정: '기본 OTO 없이 생성' 사용 중입니다.")

                lang = self._get_language()
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
                        enable_pytorch_bridge=enable_pytorch_bridge,
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
                        enable_pytorch_bridge=enable_pytorch_bridge,
                        auto_format=auto_format,
                        custom_phonemes_path=custom_phonemes_path,
                        alias_suffix=alias_suffix,
                        kr_anchor_profile_path=kr_anchor_profile_path,
                        kr_mapping_confidence_threshold=float(kr_conf_threshold),
                        kr_mapping_max_index_jump_default=int(kr_jump_default),
                        kr_mapping_max_index_jump_high_conf=int(kr_jump_hi),
                        callback=self._append_log,
                    )

                self._run_auto_validation(wav_dir, tg_folder, out_path)
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
