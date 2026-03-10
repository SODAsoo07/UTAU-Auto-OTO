import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.mfa_runner import check_mfa_model, download_mfa_model


class AlignActionsMixin:
    def _run_mfa(self):
        def task():
            self._set_running(True)
            self._set_status("3. Alignment running...")
            try:
                wav_dir = self.wav_entry.get()
                lang = self._get_language()
                dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                output_dir = os.path.join(wav_dir, "textgrids")

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

                mfa_profile = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "accurate"
                )
                self._append_log(f"MFA profile: {mfa_profile}")

                result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner="mfa",
                    fallback_aligner="",
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=mfa_profile,
                    callback=self._append_log,
                )
                if bool(result.get("ok", False)):
                    used_engine = str(result.get("used_engine", "") or "mfa").upper()
                    self._set_status(f"{used_engine} alignment complete")
                else:
                    err = str(result.get("message", "") or "alignment failed")
                    code = str(result.get("code", "") or "")
                    self._append_log(f"Alignment failed: {err} ({code})")
                    self._set_status("Alignment failed")
            except Exception as e:
                self._handle_error("Alignment", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
