import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.domino_runner import check_domino_ready
from core.mfa_runner import check_mfa_model, download_mfa_model
from core.pipeline_status import OK, has_textgrid_files, normalize_aligner_name


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
                overwrite_existing_tg = False
                has_existing_tg = has_textgrid_files(output_dir)
                if has_textgrid_files(output_dir):
                    overwrite_existing_tg = bool(
                        self._ask_yes_no_dialog(
                            "기존 TextGrid 감지",
                            "기존 TextGrid 파일이 있습니다.\n정렬 결과를 덮어쓰시겠습니까?\n\n"
                            "예: 기존 TextGrid 삭제 후 다시 정렬\n"
                            "아니오: 기존 결과 유지(정렬 건너뜀)",
                            default=False,
                        )
                    )
                    if overwrite_existing_tg:
                        self._append_log("ℹ 기존 TextGrid를 덮어쓰도록 정렬을 다시 실행합니다.")
                    else:
                        self._append_log("ℹ 기존 TextGrid를 유지합니다. 정렬 단계는 건너뜁니다.")
                use_existing_tg_only = bool(has_existing_tg and not overwrite_existing_tg)
                primary_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                    default="mfa",
                )
                fallback_engine = "mfa" if primary_engine == "domino" else ""

                if use_existing_tg_only:
                    mfa_profile = (
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    )
                    mfa_runtime_options = (
                        self._build_mfa_runtime_options()
                        if hasattr(self, "_build_mfa_runtime_options")
                        else {"constrained_mode": "off", "recursive_mfa": True}
                    )
                    result = run_alignment_with_fallback(
                        language=lang,
                        wav_folder=wav_dir,
                        dictionary_path=dict_path,
                        output_folder=output_dir,
                        primary_aligner=primary_engine,
                        fallback_aligner=fallback_engine,
                        mfa_path=self.mfa_path or "",
                        mfa_align_profile=mfa_profile,
                        mfa_runtime_options=mfa_runtime_options,
                        overwrite_existing_textgrids=False,
                        callback=self._append_log,
                    )
                    if bool(result.get("ok", False)):
                        self._set_status("기존 TextGrid 유지")
                    else:
                        self._set_status("Alignment failed")
                    return

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
                elif primary_engine == "domino":
                    if lang != "japanese":
                        self._append_log("Domino (JP) aligns Japanese only. Switch language to Japanese.")
                        self._set_status("Domino supports Japanese only")
                        return
                    domino_ready = check_domino_ready(language=lang, wav_folder=wav_dir, callback=self._append_log)
                    if str(domino_ready.get("code", "")).upper() != OK:
                        self._append_log(
                            "[Domino] not ready. Preparing MFA fallback path if possible..."
                        )
                        if hasattr(self, "_ensure_mfa_ready_for_language"):
                            if not self._ensure_mfa_ready_for_language(lang):
                                self._set_status("Domino/MFA not ready")
                                return

                mfa_profile = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "accurate"
                )
                if primary_engine == "mfa":
                    self._append_log(f"MFA profile: {mfa_profile}")
                else:
                    self._append_log("Alignment engine: Domino (JP) with MFA fallback")
                mfa_runtime_options = (
                    self._build_mfa_runtime_options()
                    if hasattr(self, "_build_mfa_runtime_options")
                    else {"constrained_mode": "off", "recursive_mfa": True}
                )
                strict_label = (
                    self._describe_mapping_strict_mode(mfa_runtime_options.get("constrained_mode", "off"))
                    if hasattr(self, "_describe_mapping_strict_mode")
                    else str(mfa_runtime_options.get("constrained_mode", "off"))
                )
                self._append_log(f"MFA constrained mode: {strict_label}")

                result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary_engine,
                    fallback_aligner=fallback_engine,
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=mfa_profile,
                    mfa_runtime_options=mfa_runtime_options,
                    overwrite_existing_textgrids=overwrite_existing_tg,
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
