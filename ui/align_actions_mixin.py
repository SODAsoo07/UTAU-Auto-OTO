import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.mfa_runner import check_mfa_model, download_mfa_model
from core.pipeline_status import normalize_aligner_name


class AlignActionsMixin:
    def _run_mfa(self):
        def task():
            self._set_running(True)
            self._set_status("3. Alignment running...")
            try:
                wav_dir = self.wav_entry.get()
                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 정렬 단계를 건너뜁니다.")
                    self._set_status("Alignment skipped (English Preview CVVC)")
                    return

                dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                output_dir = os.path.join(wav_dir, "textgrids")
                if hasattr(self, "_validate_alignment_input_files"):
                    if not self._validate_alignment_input_files(wav_dir, dict_path):
                        return

                primary_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                    default="mfa",
                )
                mfa_profile = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "accurate"
                )
                self._append_log(
                    f"ℹ 정렬 실행 시작: engine={primary_engine}, profile={mfa_profile}"
                )

                if primary_engine == "mfa":
                    if hasattr(self, "_ensure_mfa_ready_for_language"):
                        if not self._ensure_mfa_ready_for_language(lang):
                            self._set_status("MFA not ready")
                            return
                    elif self.mfa_path:
                        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
                        self._append_log(msg)
                        if not has_model and not download_mfa_model(
                            self.mfa_path, language=lang, callback=self._append_log
                        ):
                            self._set_status("MFA model missing")
                            return
                elif primary_engine == "ctc":
                    self._append_log("ℹ CTC 엔진 선택: torchaudio MMS 백엔드로 실행합니다.")

                if primary_engine == "mfa":
                    self._append_log(f"MFA profile: {mfa_profile}")
                elif primary_engine == "ctc":
                    self._append_log("Alignment engine: CTC (MMS)")
                else:
                    self._append_log("Alignment engine: none (MFA bypass)")
                if hasattr(self, "_apply_advanced_tuning_envs"):
                    self._apply_advanced_tuning_envs()

                result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary_engine,
                    fallback_aligner="",
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=mfa_profile,
                    callback=self._append_log,
                )
                if bool(result.get("ok", False)):
                    used_engine = str(result.get("used_engine", "") or "mfa").upper()
                    self._append_log(f"✅ 정렬 완료: engine={used_engine}, output={output_dir}")
                    self._set_status(f"{used_engine} alignment complete")
                else:
                    err = str(result.get("message", "") or "alignment failed")
                    code = str(result.get("code", "") or "")
                    self._append_log(f"Alignment failed: {err} ({code})")
                    if (
                        primary_engine == "ctc"
                        and hasattr(self, "_is_ctc_runtime_missing_error")
                        and self._is_ctc_runtime_missing_error(code, err)
                    ):
                        recovered = False
                        if hasattr(self, "_run_setup_ctc_script_fallback"):
                            recovered = bool(self._run_setup_ctc_script_fallback(reason="align_runtime_missing"))
                        if recovered:
                            self._append_log("ℹ CTC 런타임 복구 완료. 정렬을 다시 시도해 주세요.")
                        elif hasattr(self, "_notify_ctc_runtime_missing"):
                            self._notify_ctc_runtime_missing(detail=err)
                    if hasattr(self, "_is_lab_or_dict_missing_alignment_error") and hasattr(self, "_notify_lab_or_dict_missing"):
                        if self._is_lab_or_dict_missing_alignment_error(code, err):
                            self._notify_lab_or_dict_missing(wav_dir, dict_path)
                    if (
                        primary_engine == "mfa"
                        and hasattr(self, "_schedule_alignment_failure_mfa_followup")
                    ):
                        self._schedule_alignment_failure_mfa_followup(
                            language=lang,
                            align_code=code,
                            align_message=err,
                        )
                    self._set_status("Alignment failed")
            except Exception as e:
                self._handle_error("Alignment", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
