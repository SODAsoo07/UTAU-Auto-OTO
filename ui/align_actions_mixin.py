import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.format_type_utils import normalize_auto_format_value
from core.mfa_runner import check_mfa_model, download_mfa_model
from core.pipeline_status import normalize_aligner_name
from ui.i18n import t


class AlignActionsMixin:
    def _run_mfa(self):
        active_worker = getattr(self, "_active_worker_thread", None)
        if bool(getattr(self, "is_running", False)) and active_worker is not None and active_worker.is_alive():
            self._set_status(t("다른 작업 진행 중"))
            self._append_log("ℹ 다른 작업이 진행 중이라 정렬을 시작하지 않습니다.")
            return

        wav_dir_for_prompt = str(self.wav_entry.get() or "").strip()
        overwrite_existing_textgrids = False
        lang_for_prompt = self._get_language() if hasattr(self, "_get_language") else ""
        if (
            wav_dir_for_prompt
            and lang_for_prompt in {"korean", "japanese"}
            and hasattr(self, "_confirm_language_script_mismatch")
        ):
            if not self._confirm_language_script_mismatch(
                lang_for_prompt,
                wav_dir_for_prompt,
                stage_name=t("정렬"),
            ):
                self._set_status(t("정렬 취소됨"))
                return
        primary_engine_for_prompt = normalize_aligner_name(
            self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
            default="mfa",
        )
        if (
            wav_dir_for_prompt
            and primary_engine_for_prompt not in {"none", "coarse_crnn"}
            and lang_for_prompt != "english"
        ):
            textgrid_dir_for_prompt = os.path.join(wav_dir_for_prompt, "textgrids")
            if os.path.isdir(textgrid_dir_for_prompt):
                ask_fn = getattr(self, "_ask_yes_no_dialog_sync", None)
                if callable(ask_fn):
                    proceed = ask_fn(
                        t("기존 TextGrid 덮어쓰기 확인"),
                        (
                            "textgrids 폴더에 기존 정렬 결과가 있습니다.\n"
                            "기존 폴더를 덮어씌워 재생성하겠습니까?"
                        ),
                        default=False,
                    )
                else:
                    proceed = False
                if not proceed:
                    self._append_log("ℹ 사용자가 기존 textgrids 덮어쓰기를 취소했습니다. 정렬을 중단합니다.")
                    self._set_status(t("정렬 취소됨"))
                    return
                overwrite_existing_textgrids = True

        def task():
            self._set_running(True)
            self._set_status("3. Alignment running...")
            try:
                wav_dir = self.wav_entry.get()
                lang = self._get_language()
                selected_format = normalize_auto_format_value(
                    lang,
                    self.auto_format_var.get() if hasattr(self, "auto_format_var") else "",
                )
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 정렬 단계를 건너뜁니다.")
                    self._set_status("Alignment skipped (English Preview CVVC)")
                    return

                dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                output_dir = os.path.join(wav_dir, "textgrids")
                primary_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                    default="mfa",
                )
                if primary_engine == "coarse_crnn":
                    primary_engine = "mfa"
                if hasattr(self, "_validate_alignment_input_files"):
                    needs_lab_dict = primary_engine == "mfa"
                    if needs_lab_dict and (not self._validate_alignment_input_files(wav_dir, dict_path)):
                        return

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
                elif primary_engine == "sequence":
                    self._append_log("ℹ 전용 시퀀스 aligner 엔진 선택: frame-hop 라벨 기반 정렬을 실행합니다.")
                elif primary_engine == "coarse_crnn":
                    self._append_log("ℹ CRNN aligner 엔진 선택: SIL/C/V 경계 정렬을 실행합니다.")

                if primary_engine == "mfa":
                    self._append_log(f"MFA profile: {mfa_profile}")
                elif primary_engine == "sequence":
                    self._append_log("Alignment engine: Dedicated Sequence")
                elif primary_engine == "coarse_crnn":
                    self._append_log("Alignment engine: CRNN (experimental)")
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
                    format_hint=selected_format,
                    overwrite_existing_textgrids=bool(overwrite_existing_textgrids),
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

