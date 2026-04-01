import os
import subprocess
import sys
from datetime import datetime

from core.alignment_pipeline import run_alignment_with_fallback
from core.mfa_runner import check_mfa_model, download_mfa_model
from core.pipeline_status import normalize_aligner_name


class AlignActionsMixin:
    def _run_mfa(self):
        active_worker = getattr(self, "_active_worker_thread", None)
        if bool(getattr(self, "is_running", False)) and active_worker is not None and active_worker.is_alive():
            self._set_status("다른 작업 진행 중")
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
                stage_name="정렬",
            ):
                self._set_status("정렬 취소됨")
                return
        primary_engine_for_prompt = normalize_aligner_name(
            self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
            default="mfa",
        )
        if wav_dir_for_prompt and primary_engine_for_prompt != "none" and lang_for_prompt != "english":
            textgrid_dir_for_prompt = os.path.join(wav_dir_for_prompt, "textgrids")
            if os.path.isdir(textgrid_dir_for_prompt):
                ask_fn = getattr(self, "_ask_yes_no_dialog_sync", None)
                if callable(ask_fn):
                    proceed = ask_fn(
                        "기존 TextGrid 덮어쓰기 확인",
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
                    self._set_status("정렬 취소됨")
                    return
                overwrite_existing_textgrids = True

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
                primary_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                    default="mfa",
                )
                if hasattr(self, "_validate_alignment_input_files"):
                    needs_lab_dict = primary_engine in {"mfa", "ctc"}
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
                elif primary_engine == "ctc":
                    self._append_log("ℹ CTC 엔진 선택: torchaudio MMS 백엔드로 실행합니다.")
                elif primary_engine == "sequence":
                    self._append_log("ℹ 전용 시퀀스 aligner 엔진 선택: frame-hop 라벨 기반 정렬을 실행합니다.")

                if primary_engine == "mfa":
                    self._append_log(f"MFA profile: {mfa_profile}")
                elif primary_engine == "ctc":
                    self._append_log("Alignment engine: CTC (MMS)")
                elif primary_engine == "sequence":
                    self._append_log("Alignment engine: Dedicated Sequence")
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

    def _run_alignment_compare_visual(self):
        active_worker = getattr(self, "_active_worker_thread", None)
        if bool(getattr(self, "is_running", False)) and active_worker is not None and active_worker.is_alive():
            self._set_status("다른 작업 진행 중")
            self._append_log("ℹ 다른 작업이 진행 중이라 정렬 비교를 시작하지 않습니다.")
            return

        wav_dir = str(self.wav_entry.get() or "").strip()
        if not wav_dir or not os.path.isdir(wav_dir):
            self._set_status("정렬 비교 실행 불가")
            self._append_log("❌ 정렬 비교 실패: WAV 폴더 경로가 비어 있거나 유효하지 않습니다.")
            return

        lang = self._get_language() if hasattr(self, "_get_language") else "korean"
        if str(lang).strip().lower() not in {"korean", "japanese"}:
            self._set_status("정렬 비교 미지원")
            self._append_log("❌ 정렬 비교는 한국어/일본어 보이스뱅크에서만 지원합니다.")
            return

        def task():
            self._set_running(True)
            self._set_status("정렬 비교 실행 중...")
            try:
                script_path = os.path.join(self.app_dir if hasattr(self, "app_dir") else os.getcwd(), "scripts", "compare_alignment_visual.py")
                if not os.path.isfile(script_path):
                    self._append_log(f"❌ 비교 스크립트를 찾을 수 없습니다: {script_path}")
                    self._set_status("정렬 비교 실패")
                    return

                profile_code = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "default"
                )
                writable_root = (
                    str(getattr(self, "writable_data_dir", "") or "").strip()
                    or str(getattr(self, "app_data_dir", "") or "").strip()
                    or (self.app_dir if hasattr(self, "app_dir") else os.getcwd())
                )
                temp_root = os.path.join(writable_root, "temp")
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_root = os.path.join(
                    temp_root,
                    f"alignment_compare_ui_{os.path.basename(os.path.abspath(wav_dir))}_{stamp}",
                )
                os.makedirs(out_root, exist_ok=True)

                cmd = [
                    sys.executable,
                    script_path,
                    "--voicebank",
                    wav_dir,
                    "--language",
                    str(lang).strip().lower(),
                    "--target",
                    "both",
                    "--max-plots",
                    "12",
                    "--mfa-profile",
                    str(profile_code or "default"),
                    "--out-root",
                    out_root,
                ]
                self._append_log("ℹ 정렬 비교 실행: MFA 기준으로 Sequence/CTC 결과를 동시에 비교합니다.")
                self._append_log(f"ℹ 비교 출력 경로: {out_root}")

                env = dict(os.environ)
                env["PYTHONIOENCODING"] = "utf-8"
                process = subprocess.Popen(
                    cmd,
                    cwd=self.app_dir if hasattr(self, "app_dir") else os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                if process.stdout is not None:
                    for raw in process.stdout:
                        line = str(raw or "").rstrip()
                        if line:
                            self._append_log(f"[COMPARE] {line}")
                rc = int(process.wait())
                if rc != 0:
                    self._append_log(f"❌ 정렬 비교 실행 실패 (exit={rc})")
                    self._set_status("정렬 비교 실패")
                    return

                summary_md = os.path.join(out_root, "summary.md")
                summary_json = os.path.join(out_root, "summary.json")
                if os.path.isfile(summary_md):
                    self._append_log(f"✅ 정렬 비교 완료: {summary_md}")
                else:
                    self._append_log("⚠ 정렬 비교는 완료됐지만 summary.md를 찾지 못했습니다.")
                if os.path.isfile(summary_json):
                    self._append_log(f"ℹ 기계 판독 요약: {summary_json}")

                def _open_outputs():
                    try:
                        if os.path.isfile(summary_md):
                            os.startfile(summary_md)
                        elif os.path.isdir(out_root):
                            os.startfile(out_root)
                    except Exception:
                        pass

                if hasattr(self, "_after_safe"):
                    self._after_safe(_open_outputs)
                else:
                    _open_outputs()
                self._set_status("정렬 비교 완료")
            except Exception as e:
                self._handle_error("정렬 비교 리포트", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
