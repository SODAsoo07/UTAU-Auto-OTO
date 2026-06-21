import os

from core.alignment_pipeline import run_alignment_with_fallback
from core.format_type_utils import normalize_auto_format_value
from core.pipeline_status import normalize_aligner_name
from ui.i18n import t


class AlignActionsMixin:
    def _run_wfl_status_check(self):
        """Report whether the WFL-ASR engine (external torch env + pretrained
        model) is configured/ready for the current language, and the resolved
        paths.  Invoked from the 'WFL 상태 확인' button."""
        try:
            from core.wfl_runner import (
                _resolve_wfl_model_dir,
                _resolve_wfl_python,
                _resolve_wfl_repo,
                check_wfl_ready,
            )
        except Exception as exc:
            self._append_log(f"❌ WFL 모듈을 불러오지 못했습니다: {exc}")
            return

        lang = self._get_language() if hasattr(self, "_get_language") else "japanese"
        py = _resolve_wfl_python()
        repo = _resolve_wfl_repo()
        model = _resolve_wfl_model_dir(lang)
        self._append_log("ℹ WFL 상태 확인")
        self._append_log(f"   python : {py or '(미설정)'}")
        self._append_log(f"   repo   : {repo or '(미설정)'}")
        self._append_log(f"   model  : {model or '(미설정)'}  [{lang}]")
        report = check_wfl_ready(language=lang, wav_folder="")
        ready = bool(report.get("ready", False))
        if ready:
            self._append_log("✅ WFL 준비 완료. 정렬에 사용할 수 있습니다.")
            self._set_status("WFL ready")
        else:
            self._append_log(f"⚠ WFL 미준비: {report.get('message', '')}")
            if str(report.get("code", "")) == "ALIGN_MODEL_MISSING":
                self._append_log("   ⬇ 'WFL 모델 받기' 버튼으로 사전훈련 모델을 자동 설치할 수 있습니다.")
            self._append_log(
                "   환경변수 UTOA_WFL_PYTHON / UTOA_WFL_REPO / "
                "UTOA_WFL_MODEL_DIR_<LANG> 로 직접 지정할 수도 있습니다. 미준비 시 정렬은 HSMM OTO로 폴백합니다."
            )
            self._set_status("WFL not ready (fallback: HSMM OTO)")

    def _run_wfl_model_download(self):
        """Download + install the pretrained WFL model for the current language
        into <app-root>/models/wfl/<lang>. Runs in a worker thread."""
        lang = self._get_language() if hasattr(self, "_get_language") else "japanese"
        try:
            from core.wfl_runner import download_wfl_model
        except Exception as exc:
            self._append_log(f"❌ WFL 모듈을 불러오지 못했습니다: {exc}")
            return

        def task():
            self._set_running(True)
            self._set_status(f"⏳ WFL 모델 다운로드 중... ({lang})")
            try:
                ok, msg = download_wfl_model(lang, callback=self._append_log)
                if ok:
                    self._append_log(f"✅ {msg}")
                    self._set_status("✅ WFL 모델 설치 완료")
                else:
                    self._append_log(f"⚠ {msg}")
                    self._set_status("⚪ WFL 모델 설치 미완료")
            except Exception as e:
                self._handle_error("WFL 모델 다운로드", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

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
            self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
            default="hsmm_oto",
        )
        if (
            wav_dir_for_prompt
            and primary_engine_for_prompt in {"wfl", "sequence"}
            and lang_for_prompt != "english"
        ):
            prompt_dirs = [wav_dir_for_prompt]
            try:
                if self._is_recursive_voicebank_scan_enabled():
                    targets, _batch_enabled = self._resolve_voicebank_batch_targets_for_ui(wav_dir_for_prompt)
                    if targets:
                        prompt_dirs = [target.wav_dir for target in targets]
            except Exception:
                prompt_dirs = [wav_dir_for_prompt]
            has_existing_textgrids = False
            for prompt_dir in prompt_dirs:
                textgrid_dir_for_prompt = os.path.join(prompt_dir, "textgrids")
                if os.path.isdir(textgrid_dir_for_prompt):
                    has_existing_textgrids = True
                    break
            if has_existing_textgrids:
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

                primary_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
                    default="hsmm_oto",
                )
                if primary_engine == "hsmm_oto":
                    self._append_log("ℹ HSMM OTO 모드에서는 별도 TextGrid 정렬 단계를 건너뜁니다.")
                    self._set_status("Alignment skipped (HSMM OTO)")
                    return
                if hasattr(self, "_resolve_voicebank_batch_targets_for_ui"):
                    targets, batch_scan_enabled = self._resolve_voicebank_batch_targets_for_ui(wav_dir)
                    if not targets:
                        return
                    if hasattr(self, "_append_voicebank_batch_summary"):
                        self._append_voicebank_batch_summary(
                            targets,
                            batch_scan_enabled=batch_scan_enabled,
                            stage_name="정렬",
                        )
                else:
                    targets = [
                        type(
                            "_Target",
                            (),
                            {
                                "wav_dir": os.path.abspath(wav_dir),
                                "index": 0,
                                "total": 1,
                            },
                        )()
                    ]

                self._append_log(
                    f"ℹ 정렬 실행 시작: engine={primary_engine}"
                )

                if primary_engine == "sequence":
                    self._append_log("ℹ 전용 시퀀스 aligner 엔진 선택: frame-hop 라벨 기반 정렬을 실행합니다.")
                elif primary_engine == "wfl":
                    self._append_log("ℹ WFL-ASR 엔진 선택: 준비되지 않으면 HSMM OTO로 자동 폴백합니다.")
                if primary_engine == "sequence":
                    self._append_log("Alignment engine: Dedicated Sequence")
                elif primary_engine == "wfl":
                    self._append_log("Alignment engine: WFL-ASR")
                else:
                    self._append_log(f"Alignment engine: {primary_engine}")
                if hasattr(self, "_apply_advanced_tuning_envs"):
                    self._apply_advanced_tuning_envs()

                success_count = 0
                failed_count = 0
                for target in targets:
                    target_wav_dir = target.wav_dir
                    prefix = f"[Batch {target.index + 1}/{target.total}] " if target.total > 1 else ""
                    if target.total > 1:
                        self._append_log(f"{prefix}대상 폴더: {target_wav_dir}")
                    dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                    dict_path = os.path.join(target_wav_dir, dict_filename)
                    output_dir = os.path.join(target_wav_dir, "textgrids")
                    if hasattr(self, "_validate_alignment_input_files"):
                        needs_lab_dict = primary_engine in ("wfl",)
                        if needs_lab_dict and (not self._validate_alignment_input_files(target_wav_dir, dict_path)):
                            failed_count += 1
                            continue

                    result = run_alignment_with_fallback(
                        language=lang,
                        wav_folder=target_wav_dir,
                        dictionary_path=dict_path,
                        output_folder=output_dir,
                        primary_aligner=primary_engine,
                        fallback_aligner=("hsmm_oto" if primary_engine == "wfl" else ""),
                        format_hint=selected_format,
                        overwrite_existing_textgrids=bool(overwrite_existing_textgrids),
                        callback=self._append_log,
                    )
                    if bool(result.get("ok", False)):
                        used_engine = str(result.get("used_engine", "") or "unknown").upper()
                        self._append_log(f"{prefix}✅ 정렬 완료: engine={used_engine}, output={output_dir}")
                        success_count += 1
                    else:
                        err = str(result.get("message", "") or "alignment failed")
                        code = str(result.get("code", "") or "")
                        self._append_log(f"{prefix}Alignment failed: {err} ({code})")
                        if hasattr(self, "_is_lab_or_dict_missing_alignment_error") and hasattr(self, "_notify_lab_or_dict_missing"):
                            if self._is_lab_or_dict_missing_alignment_error(code, err):
                                self._notify_lab_or_dict_missing(target_wav_dir, dict_path)
                        failed_count += 1
                if failed_count:
                    self._set_status(f"Alignment partial/failed ({success_count}/{len(targets)})")
                else:
                    self._set_status(f"Alignment complete ({success_count}/{len(targets)})")
            except Exception as e:
                self._handle_error("Alignment", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

