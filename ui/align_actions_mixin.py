import os

from core.mfa_runner import check_mfa_model, download_mfa_model, run_mfa_align
from core.sofa_runner import get_default_sofa_repo_dir, run_sofa_align


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

                if self.aligner_var.get() == "SOFA":
                    ckpt = self._ensure_sofa_model_ready(lang)
                    sdic = self.sofa_dict_var.get().strip() or dict_path
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
                    if not self.sofa_dict_var.get().strip():
                        self._after_safe(lambda p=sdic: self.sofa_dict_var.set(p))
                        self._append_log(f"ℹ SOFA 사전이 비어 있어 현재 생성 사전을 사용합니다: {sdic}")
                    if not ckpt or not sdic:
                        self._append_log("❌ SOFA 사용 시 체크포인트(.ckpt)와 사전(dict.txt) 경로가 필요합니다.")
                        self._set_status("❌ SOFA 설정 누락")
                        return

                    self._append_log("🔀 정렬 엔진: SOFA")
                    self._append_log(f"ℹ SOFA 전용 Python: {self.sofa_python_var.get().strip()}")
                    self._append_log(
                        "ℹ SOFA 실행 옵션: "
                        f"repo={sofa_kwargs.get('sofa_repo_dir')}, "
                        f"mode={sofa_kwargs.get('mode')}, "
                        f"2-pass={'ON' if sofa_kwargs.get('two_pass_retry') else 'OFF'}"
                    )
                    success, err = run_sofa_align(
                        wav_folder=wav_dir,
                        output_folder=output_dir,
                        ckpt_path=ckpt,
                        dictionary_path=sdic,
                        mfa_path=self.mfa_path or "",
                        sofa_python=self.sofa_python_var.get().strip(),
                        callback=self._append_log,
                        **sofa_kwargs,
                    )
                    if success:
                        self._set_status("✅ SOFA 정렬 완료")
                    else:
                        self._append_log(f"❌ SOFA 실패: {err}")
                        self._set_status("❌ SOFA 실패")
                else:
                    self._append_log("🔀 정렬 엔진: MFA(권장)")
                    if not self.mfa_path:
                        self._append_log("❌ MFA 실행 파일을 찾을 수 없습니다!")
                        self._append_log("   💡 MFA를 설치하거나, 프로그램 폴더에 .env/ 포터블 환경을 배치해 주세요.")
                        self._set_status("❌ MFA 미설치")
                        return

                    has_model, msg = check_mfa_model(self.mfa_path, language=lang)
                    self._append_log(msg)
                    if not has_model:
                        download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)

                    success, err = run_mfa_align(
                        self.mfa_path,
                        wav_dir,
                        dict_path,
                        output_dir,
                        language=lang,
                        callback=self._append_log,
                    )
                    if success:
                        self._set_status("✅ MFA 정렬 완료")
                    else:
                        self._append_log(f"❌ MFA 실패: {err}")
                        self._notify_mfa_failure_suggest_sofa(lang, err)
                        self._set_status("❌ MFA 실패")
            except Exception as e:
                self._handle_error("음성 정렬", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
