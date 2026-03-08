import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.mfa_runner import check_mfa_model, download_mfa_model
from core.sofa_runner import get_default_sofa_repo_dir


class AlignActionsMixin:
    def _run_mfa(self):
        def task():
            self._set_running(True)
            self._set_status("3️⃣ 음성 정렬 중... (시간이 걸릴 수 있습니다)")
            try:
                wav_dir = self.wav_entry.get()
                lang = self._get_language()
                dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                output_dir = os.path.join(wav_dir, "textgrids")
                primary = "sofa" if self.aligner_var.get() == "SOFA" else "mfa"
                fallback = "mfa" if primary == "sofa" else ""

                if primary == "sofa":
                    runtime_getter = getattr(self, "_get_sofa_runtime_kwargs", None)
                    if callable(runtime_getter):
                        sofa_kwargs = runtime_getter(lang)
                    else:
                        sofa_kwargs = {
                            "sofa_repo_dir": (
                                get_default_sofa_repo_dir("utau_kr_v1")
                                if lang == "korean"
                                else get_default_sofa_repo_dir()
                            ),
                            "mode": "force",
                            "g2p": "Dictionary",
                            "ap_detector": "LoudnessSpectralcentroidAPDetector",
                            "ap_detector_config": "",
                            "save_confidence": lang == "korean",
                            "out_formats": "TextGrid",
                            "extra_infer_args": "",
                            "two_pass_retry": lang == "korean",
                            "two_pass_retry_mode": "match",
                            "confidence_threshold": 0.55,
                            "low_confidence_max_files": 0,
                        }
                    ckpt = self.sofa_ckpt_var.get().strip()
                    ckpt = self._ensure_sofa_model_ready(lang)
                    sdic = self.sofa_dict_var.get().strip() or dict_path
                    if not self.sofa_dict_var.get().strip():
                        self._after_safe(lambda p=sdic: self.sofa_dict_var.set(p))
                        self._append_log(f"ℹ SOFA 사전이 비어 있어 현재 생성 사전을 사용합니다: {sdic}")
                else:
                    sofa_kwargs = {}
                    ckpt = ""
                    sdic = dict_path

                if primary == "mfa":
                    if hasattr(self, "_ensure_mfa_ready_for_language"):
                        if not self._ensure_mfa_ready_for_language(lang):
                            self._set_status("❌ MFA 설치/준비 실패")
                            return
                    elif self.mfa_path:
                        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
                        self._append_log(msg)
                        if not has_model and not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                            self._set_status("❌ MFA 모델 준비 실패")
                            return

                if primary == "mfa":
                    mfa_profile = (
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    )
                    self._append_log(f"ℹ MFA 정렬 프로필: {mfa_profile}")
                else:
                    self._append_log(f"ℹ SOFA 전용 Python: {self.sofa_python_var.get().strip()}")
                    self._append_log(
                        "ℹ SOFA 실행 옵션: "
                        f"repo={sofa_kwargs.get('sofa_repo_dir')}, "
                        f"mode={sofa_kwargs.get('mode')}, "
                        f"2-pass={'ON' if sofa_kwargs.get('two_pass_retry') else 'OFF'}"
                    )

                result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary,
                    fallback_aligner=fallback,
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=(
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    ),
                    sofa_kwargs={
                        **sofa_kwargs,
                        "ckpt_path": ckpt,
                        "dictionary_path": sdic,
                        "sofa_python": self.sofa_python_var.get().strip(),
                    },
                    callback=self._append_log,
                )
                if bool(result.get("ok", False)):
                    used_engine = str(result.get("used_engine", "") or primary).upper()
                    if result.get("fallback_used"):
                        self._append_log(f"ℹ 정렬 fallback 적용: {result.get('fallback_path', '')}")
                    self._set_status(f"✅ {used_engine} 정렬 완료")
                else:
                    err = str(result.get("message", "") or "정렬 실패")
                    code = str(result.get("code", "") or "")
                    show_advanced = bool(self.show_advanced_aligner_var.get()) if hasattr(self, "show_advanced_aligner_var") else False
                    self._append_log(f"❌ 정렬 실패: {err} ({code})")
                    if primary == "mfa" and show_advanced:
                        self._notify_mfa_failure_suggest_sofa(lang, err)
                    self._set_status("❌ 정렬 실패")
            except Exception as e:
                self._handle_error("음성 정렬", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
