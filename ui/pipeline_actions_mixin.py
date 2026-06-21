import os
import sys
import locale
import subprocess as sp
import json
import re
import shutil
import time
import threading

from core.alignment_pipeline import run_alignment_with_fallback
from core.en_cvvc_builder import generate_en_cvvc_oto
from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.kr_cmpx_preview_builder import (
    generate_kr_cmpx_preview_oto,
    resolve_kr_cmpx_preview_source_oto,
)
from core.ja_oto_generator import generate_ja_oto
from core.lab_generator import generate_dictionary, generate_labs
from core.no_mfa_oto_builder import (
    generate_no_mfa_auto_oto,
    get_last_no_mfa_runtime_meta,
    resolve_no_mfa_source_oto,
)
from core.mfa_free_oto.review_generation import generate_hsmm_oto_review
from core.oto_file_utils import detect_oto_text_encoding, reencode_oto_file
from core.oto_generator import generate_oto
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues
from core.error_codes import format_error_with_recovery
from core.format_type_utils import normalize_auto_format_value
from core.distribution_guard import is_training_paths_enabled
from core.cuda_runtime_bootstrap import (
    collect_cuda_runtime_diagnosis,
    install_torch_cuda_runtime,
)
from core.generation.mapping_runtime import (
    format_mapping_reason_schema_summary,
    format_mapping_summary,
)
from ui.i18n import t
from ui.voicebank_batch import (
    VoicebankBatchTarget,
    resolve_voicebank_batch_targets,
)



class PipelineActionsMixin:
    def _is_recursive_voicebank_scan_enabled(self) -> bool:
        try:
            return bool(self.recursive_voicebank_scan_var.get()) if hasattr(self, "recursive_voicebank_scan_var") else False
        except Exception:
            return False

    def _resolve_voicebank_batch_targets_for_ui(
        self,
        root_wav_dir: str,
        base_out_path: str = "",
    ) -> tuple[list[VoicebankBatchTarget], bool]:
        batch_scan_enabled = self._is_recursive_voicebank_scan_enabled()
        targets = resolve_voicebank_batch_targets(
            root_wav_dir,
            base_out_path,
            batch_scan_enabled=batch_scan_enabled,
        )
        if batch_scan_enabled and not targets:
            self._append_log("❌ 하위 폴더에서 WAV 파일이 있는 보이스 폴더를 찾지 못했습니다.")
            self._set_status("❌ 배치 대상 없음")
        elif (not batch_scan_enabled) and not targets:
            self._append_log(f"❌ WAV 폴더가 존재하지 않습니다: {root_wav_dir}")
            self._set_status("❌ WAV 경로 오류")
        return targets, batch_scan_enabled

    def _append_voicebank_batch_summary(
        self,
        targets: list[VoicebankBatchTarget],
        *,
        batch_scan_enabled: bool,
        stage_name: str,
    ) -> None:
        if not batch_scan_enabled:
            return
        if len(targets) > 1:
            self._append_log(f"ℹ {stage_name}: 하위 WAV 폴더 {len(targets)}개를 pitch별로 순차 처리합니다.")
        elif targets:
            self._append_log(f"ℹ {stage_name}: 하위 WAV 폴더 1개를 처리합니다.")

    def _run_hsmm_oto_preview_generation(
        self,
        *,
        wav_dir: str,
        out_path: str,
        source_oto_path: str = "",
        language: str = "japanese",
        format_type: str = "CV",
        apply_lightgbm: bool = False,
        lightgbm_policy: str = "auto",
        lightgbm_model_dir: str = "",
        callback=None,
    ):
        base_dir = (
            str(getattr(self, "writable_data_dir", "") or "").strip()
            or str(getattr(self, "app_data_dir", "") or "").strip()
            or str(getattr(self, "app_dir", "") or "").strip()
            or os.getcwd()
        )
        bank_name = os.path.basename(os.path.normpath(wav_dir)) or "voicebank"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        preview_dir = os.path.abspath(
            os.path.join(base_dir, "ml_workspace", "mfa_free_oto", "ui_hsmm_oto", f"{bank_name}_{timestamp}")
        )
        os.makedirs(preview_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", bank_name).strip("._") or "voicebank"
        policy = str(lightgbm_policy or "auto").strip().lower() or "auto"
        model_dir = str(lightgbm_model_dir or "").strip()
        device = ""
        if hasattr(self, "oto_crnn_device_var"):
            try:
                device = str(self.oto_crnn_device_var.get() or "").strip().lower()
            except Exception:
                device = ""
        if device not in {"cpu", "cuda"}:
            device = ""

        base_oto = (
            resolve_no_mfa_source_oto(
                wav_dir=wav_dir,
                source_hint=source_oto_path,
            )
            if str(source_oto_path or "").strip()
            else ""
        )
        if callback:
            try:
                callback("[HSMM OTO] progress 0/1")
            except Exception:
                pass
        if base_oto:
            self._append_log(f"[HSMM OTO] base oto alias list: {base_oto}")
            self._append_log("[HSMM OTO] base oto timing ignored; HSMM/acoustic events assign parameters.")
        else:
            self._append_log("[HSMM OTO] no base oto; filename row-plan and acoustic events are used.")
        if bool(apply_lightgbm):
            self._append_log(f"[HSMM OTO] LightGBM postprocess requested (policy={policy}).")
        self._append_log(f"[HSMM OTO] input wav dir: {os.path.abspath(wav_dir)}")
        self._append_log(f"[HSMM OTO] preview artifacts: {preview_dir}")
        self._append_log("[HSMM OTO] stage 1/4: prepare decoder workspace")
        self._append_log("[HSMM OTO] runtime helper: core.mfa_free_oto.review_generation")
        self._append_log("[HSMM OTO] stage 2/4: run HSMM/acoustic decoder")

        def _review_callback(message):
            text = str(message or "").strip()
            if text:
                self._append_log(f"[HSMM OTO] {text}")

        try:
            preview_summary = generate_hsmm_oto_review(
                wav_dir=os.path.abspath(wav_dir),
                out_dir=preview_dir,
                name=safe_name,
                template_oto=base_oto,
                language=language,
                format_type=format_type,
                alias_type="auto",
                encoder="acoustic_world_v1",
                device=device,
                apply_lightgbm=bool(apply_lightgbm),
                lightgbm_policy=policy,
                lightgbm_model_dir=os.path.abspath(model_dir) if model_dir else "",
                callback=_review_callback,
            )
        except Exception as exc:
            self._append_log(f"[HSMM OTO] decoder failed: {exc}")
            return 0, 0, [f"HSMM OTO generation failed: {exc}"]

        if "review_split_counts" not in preview_summary and isinstance(preview_summary.get("split_counts"), dict):
            preview_summary["review_split_counts"] = preview_summary.get("split_counts")
        if "review_split_output_paths" not in preview_summary and isinstance(preview_summary.get("split_output_paths"), dict):
            preview_summary["review_split_output_paths"] = preview_summary.get("split_output_paths")
        if "oto_rows" not in preview_summary and preview_summary.get("processed") is not None:
            preview_summary["oto_rows"] = preview_summary.get("processed")

        self._append_log(f"[HSMM OTO] decoder finished: ok={bool(preview_summary.get('ok'))}")
        if not bool(preview_summary.get("ok")):
            errors = [str(item) for item in list(preview_summary.get("errors", []) or []) if str(item).strip()]
            if not errors:
                errors = ["HSMM OTO generation failed"]
            return int(preview_summary.get("processed") or 0), int(preview_summary.get("total") or 0), errors

        self._append_log("[HSMM OTO] stage 3/4: copy generated oto")
        generated_path = str(preview_summary.get("generated_oto_path") or "").strip()
        out_path_abs = os.path.abspath(out_path)
        if generated_path and os.path.isfile(generated_path):
            out_dir = os.path.dirname(out_path_abs)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if os.path.abspath(generated_path) != out_path_abs:
                shutil.copyfile(generated_path, out_path_abs)

        row_count = 0
        try:
            with open(out_path_abs, "r", encoding="utf-8-sig") as handle:
                row_count = sum(1 for line in handle if line.strip() and "=" in line)
        except Exception:
            row_count = 0
        row_count = int(preview_summary.get("processed") or preview_summary.get("oto_rows") or row_count or 0)
        total_count = int(preview_summary.get("total") or row_count or 0)
        if callback:
            try:
                callback(f"[HSMM OTO] progress {row_count}/{total_count or row_count or 1}")
            except Exception:
                pass
        self._append_log(f"[HSMM OTO] preview oto: {out_path_abs}")
        if generated_path:
            self._append_log(f"[HSMM OTO] generated source: {generated_path}")
        self._append_log("[HSMM OTO] stage 4/4: summarize review split")
        self._log_hsmm_oto_preview_split_summary(preview_summary)
        return row_count, total_count or row_count, []

    def _parse_hsmm_oto_preview_summary(self, stdout_lines):
        text = "\n".join(str(line or "") for line in (stdout_lines or []) if str(line or "").strip())
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        if "review_split_counts" not in payload and isinstance(payload.get("split_counts"), dict):
            payload["review_split_counts"] = payload.get("split_counts")
        if "review_split_output_paths" not in payload and isinstance(payload.get("split_output_paths"), dict):
            payload["review_split_output_paths"] = payload.get("split_output_paths")
        if "oto_rows" not in payload and payload.get("processed") is not None:
            payload["oto_rows"] = payload.get("processed")
        return payload

    def _parse_mfa_free_preview_summary(self, stdout_lines):
        return self._parse_hsmm_oto_preview_summary(stdout_lines)

    def _log_hsmm_oto_preview_split_summary(self, payload):
        if not isinstance(payload, dict):
            return
        counts = payload.get("review_split_counts")
        if isinstance(counts, dict) and counts:
            self._append_log(
                "[HSMM OTO] review split counts: "
                f"fix_required={int(counts.get('fix_required') or 0)}, "
                f"attention_only={int(counts.get('attention_only') or 0)}, "
                f"clean={int(counts.get('clean') or 0)}, "
                f"total={int(counts.get('total') or 0)}"
            )
        if bool(payload.get("guard_failed")):
            reasons = [
                str(item)
                for item in list(payload.get("review_split_guard_reasons", []) or [])
                if str(item).strip()
            ]
            if reasons:
                self._append_log(f"[HSMM OTO] review split guard: manual review required ({', '.join(reasons)})")
        paths = payload.get("review_split_output_paths")
        if isinstance(paths, dict) and paths:
            for key in (
                "fix_required",
                "attention_only",
                "clean",
                "review_all",
                "validation_jsonl",
                "summary",
                "review_session",
            ):
                value = str(paths.get(key) or "").strip()
                if value:
                    self._append_log(f"[HSMM OTO] {key}: {value}")
        row_provenance_path = str(payload.get("row_provenance_path") or "").strip()
        if row_provenance_path:
            self._append_log(f"[HSMM OTO] row_provenance: {row_provenance_path}")
        lightgbm = payload.get("lightgbm_postprocess")
        if isinstance(lightgbm, dict) and lightgbm.get("enabled"):
            status = str(lightgbm.get("status") or "unknown")
            changed = int(lightgbm.get("changed") or 0)
            reason = str(lightgbm.get("reason") or "").strip()
            suffix = f", reason={reason}" if reason else ""
            self._append_log(f"[HSMM OTO] LightGBM postprocess: status={status}, changed={changed}{suffix}")
            pre_path = str(lightgbm.get("pre_lightgbm_path") or "").strip()
            if pre_path:
                self._append_log(f"[HSMM OTO] pre-LightGBM oto: {pre_path}")

    def _log_mfa_free_preview_split_summary(self, payload):
        return self._log_hsmm_oto_preview_split_summary(payload)

    @staticmethod
    def _has_any_lab_files(wav_dir: str) -> bool:
        root = str(wav_dir or "").strip()
        if not root or not os.path.isdir(root):
            return False
        for _cur, _dirs, files in os.walk(root):
            if any(str(name).lower().endswith(".lab") for name in files):
                return True
        return False

    def _notify_lab_or_dict_missing(self, wav_dir: str, dict_path: str) -> None:
        guide = "Lab 파일 또는 딕셔너리 파일이 없습니다. 왼쪽 탭에서 Lab+사전 생성 버튼을 클릭해주세요."
        self._append_log(f"❌ {guide}")
        if not self._has_any_lab_files(wav_dir):
            self._append_log(f"   - Lab 파일 미존재: {wav_dir}")
        if not (dict_path and os.path.isfile(dict_path)):
            self._append_log(f"   - 딕셔너리 파일 미존재: {dict_path}")
        self._after_safe(
            lambda: self._show_copyable_alert(
                title=t("정렬 입력 파일 누락"),
                message=guide,
                alert_key="align_lab_dict_missing",
            )
        )

    def _validate_alignment_input_files(self, wav_dir: str, dict_path: str) -> bool:
        has_lab = self._has_any_lab_files(wav_dir)
        has_dict = bool(dict_path and os.path.isfile(dict_path))
        if has_lab and has_dict:
            return True
        self._notify_lab_or_dict_missing(wav_dir, dict_path)
        self._set_status(f"❌ {t('정렬 입력 파일 누락')}")
        return False

    @staticmethod
    def _is_lab_or_dict_missing_alignment_error(code: str, message: str) -> bool:
        c = str(code or "").strip().upper()
        text = str(message or "")
        lowered = text.lower()
        if c == "ALIGN_DICT_MISSING":
            return True
        if "dictionary not found" in lowered:
            return True
        if "textgrid" in lowered and ("not found" in lowered or "missing" in lowered):
            return True
        if "lab" in lowered and ("not found" in lowered or "missing" in lowered):
            return True
        return False

    def _read_runtime_var(self, var_name, default=None):
        var = getattr(self, var_name, None)
        if var is None:
            return default
        try:
            return var.get()
        except Exception:
            return default

    def _to_bool(self, value, default=False):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return default

    def _subprocess_no_window_kwargs(self):
        if os.name != "nt":
            return {}
        kwargs = {}
        try:
            startupinfo = sp.STARTUPINFO()
            startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            kwargs["startupinfo"] = startupinfo
        except Exception:
            pass
        try:
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | int(
                getattr(sp, "CREATE_NO_WINDOW", 0)
            )
        except Exception:
            pass
        return kwargs

    def _run_subprocess_hidden(self, args, **kwargs):
        for key, value in self._subprocess_no_window_kwargs().items():
            kwargs.setdefault(key, value)
        return sp.run(args, **kwargs)

    def _popen_subprocess_hidden(self, args, **kwargs):
        for key, value in self._subprocess_no_window_kwargs().items():
            kwargs.setdefault(key, value)
        return sp.Popen(args, **kwargs)

    def _preferred_subprocess_encoding(self):
        try:
            return locale.getpreferredencoding(False) or "utf-8"
        except Exception:
            return "utf-8"

    def _subprocess_decode_candidates(self):
        candidates = []
        for enc in (
            "utf-8-sig",
            "utf-8",
            self._preferred_subprocess_encoding(),
            getattr(locale, "getencoding", lambda: "")() or "",
            "cp932",
            "cp949",
            "mbcs",
        ):
            enc = str(enc or "").strip()
            if enc and enc not in candidates:
                candidates.append(enc)
        return candidates

    def _score_decoded_subprocess_text(self, text):
        score = 0
        for ch in str(text or ""):
            code = ord(ch)
            if ch == "\ufffd":
                score -= 20
            elif 0x20 <= code <= 0x7E or ch in "\r\n\t":
                score += 1
            elif 0xAC00 <= code <= 0xD7A3:
                score += 4
            elif 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
                score += 3
            elif 0xFF61 <= code <= 0xFF9F:
                score -= 6
            elif code < 0x20:
                score -= 10
        return score

    def _decode_subprocess_output(self, data):
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        raw = bytes(data)
        best_text = ""
        best_score = None
        for enc in self._subprocess_decode_candidates():
            try:
                decoded = raw.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
            score = self._score_decoded_subprocess_text(decoded)
            if best_score is None or score > best_score:
                best_text = decoded
                best_score = score
        if best_score is not None:
            return best_text
        return raw.decode("utf-8", errors="replace")

    def _iter_decoded_stdout_lines(self, process):
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return
        for raw_line in stdout:
            line = self._decode_subprocess_output(raw_line).strip()
            if line:
                yield line

    def _cuda_startup_check_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".cuda_startup_check_state.json")

    def _load_cuda_startup_check_state(self):
        path = self._cuda_startup_check_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_cuda_startup_check_state(self, **updates):
        state = self._load_cuda_startup_check_state()
        state.update(updates or {})
        path = self._cuda_startup_check_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _schedule_startup_cuda_runtime_check(self):
        enabled = str(os.environ.get("UTOA_ENABLE_STARTUP_CUDA_RUNTIME_CHECK", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return
        if str(os.environ.get("UTOA_DISABLE_STARTUP_CUDA_RUNTIME_CHECK", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_cuda_runtime_check_scheduled", False):
            return
        state = self._load_cuda_startup_check_state()
        if bool(state.get("first_check_done", False)):
            return

        def _kickoff():
            if getattr(self, "_is_closing", False):
                return
            thread = threading.Thread(
                target=self._run_startup_cuda_runtime_check_task,
                daemon=True,
            )
            thread.start()

        self._startup_cuda_runtime_check_scheduled = True
        self._after_safe(_kickoff, delay_ms=1800)

    def _run_startup_cuda_runtime_check_task(self):
        now = time.time()
        self._save_cuda_startup_check_state(
            last_attempt_ts=now,
            last_result="running",
        )
        try:
            diagnosis = collect_cuda_runtime_diagnosis()
            has_nvidia = bool(diagnosis.get("nvidia_gpu_present"))
            has_cuda = bool(diagnosis.get("torch_cuda_available"))
            gpu_name = str((diagnosis.get("nvidia_gpu_names") or [""])[0] or "").strip()
            nvidia_cuda_ver = str(diagnosis.get("nvidia_cuda_version", "") or "").strip()
            torch_ver = str(diagnosis.get("torch_version", "") or "").strip()
            torch_cuda_ver = str(diagnosis.get("torch_cuda_version", "") or "").strip()
            direct_cmd = str(diagnosis.get("recommended_install_command_text", "") or "").strip()

            if (not has_nvidia) or has_cuda:
                self._save_cuda_startup_check_state(
                    first_check_done=True,
                    last_result="ready" if has_cuda else "skip_no_nvidia",
                    last_checked_ts=time.time(),
                    nvidia_gpu_present=has_nvidia,
                    torch_cuda_available=has_cuda,
                )
                return

            app_dir = getattr(self, "app_dir", "") or os.getcwd()
            setup_script_path = os.path.join(app_dir, "setup_mfa.bat")
            setup_cmd = "setup_mfa.bat --with-ml --with-cuda"
            setup_cmd_line = setup_cmd if os.path.isfile(setup_script_path) else "(setup_mfa.bat 파일을 찾지 못함)"
            title = "CUDA 런타임 자동 설치"
            message = (
                "NVIDIA GPU가 감지되었지만 현재 Python 런타임에서 CUDA를 사용할 수 없습니다.\n\n"
                f"- GPU: {gpu_name or '(미확인)'}\n"
                f"- nvidia-smi CUDA 버전: {nvidia_cuda_ver or '(미확인)'}\n"
                f"- torch 버전: {torch_ver or '(미설치 또는 import 실패)'}\n"
                f"- torch CUDA 빌드: {torch_cuda_ver or '(없음)'}\n\n"
                "지금 네트워크로 CUDA용 PyTorch를 자동 설치할까요?\n\n"
                f"- 자동 설치 실패 시 수동 명령: {direct_cmd or '(권장 pip 명령 생성 실패)'}\n"
                f"- 대체 경로: {setup_cmd_line}"
            )
            self._append_log(
                "⚠ NVIDIA GPU는 감지되었지만 torch CUDA 런타임이 비활성 상태입니다. "
                "초기 1회 자동 설치 선택 팝업을 표시합니다."
            )

            wants_auto_install = False
            if hasattr(self, "_ask_yes_no_dialog_sync"):
                wants_auto_install = bool(
                    self._ask_yes_no_dialog_sync(
                        title=title,
                        message=message,
                        default=False,
                    )
                )

            post_diag = diagnosis
            install_result = {}
            state_result = "needs_cuda_runtime"
            if wants_auto_install:
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 시작합니다.")
                command_args = diagnosis.get("recommended_install_command") or []
                python_exe = ""
                if isinstance(command_args, (list, tuple)) and command_args:
                    python_exe = str(command_args[0] or "").strip()
                install_result = install_torch_cuda_runtime(
                    python_exe=python_exe,
                    cuda_version=nvidia_cuda_ver,
                    timeout_sec=3600,
                )
                post_diag = collect_cuda_runtime_diagnosis(python_exe=python_exe)
                install_ok = bool(install_result.get("success", False))
                runtime_ok = bool(post_diag.get("torch_cuda_available", False))
                if install_ok and runtime_ok:
                    state_result = "auto_install_success"
                    self._append_log("✅ CUDA 런타임 자동 설치가 완료되었습니다.")
                else:
                    state_result = "auto_install_failed"
                    self._append_log("❌ CUDA 런타임 자동 설치에 실패했습니다.")
                    tail = str(install_result.get("stderr_tail", "") or install_result.get("stdout_tail", "") or "").strip()
                    if tail:
                        self._append_log(f"   설치 로그 요약: {tail[-240:]}")
                    self._after_safe(
                        lambda cmd=direct_cmd, setup=setup_cmd_line, t=tail: self._show_copyable_alert(
                            title="CUDA 런타임 자동 설치 실패",
                            message=(
                                "CUDA 자동 설치가 실패했습니다.\n\n"
                                "아래 방법으로 수동 설치 후 앱을 다시 실행해 주세요.\n"
                                f"1) {cmd or '(권장 pip 명령 생성 실패)'}\n"
                                f"2) {setup}\n\n"
                                f"마지막 오류:\n{(t or '(로그 없음)')[-600:]}"
                            ),
                            alert_key="cuda_runtime_auto_install_failed",
                            link_url="https://pytorch.org/get-started/locally/",
                            link_label="PyTorch 설치 가이드",
                        )
                    )
            else:
                state_result = "user_declined_install"
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 건너뛰었습니다.")

            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result=state_result,
                last_checked_ts=time.time(),
                nvidia_gpu_present=has_nvidia,
                torch_cuda_available=bool(post_diag.get("torch_cuda_available", False)),
                nvidia_gpu_name=gpu_name,
                nvidia_cuda_version=nvidia_cuda_ver,
                torch_version=torch_ver,
                torch_cuda_version=str(post_diag.get("torch_cuda_version", "") or torch_cuda_ver),
                recommended_command=direct_cmd,
                setup_command=setup_cmd if os.path.isfile(setup_script_path) else "",
                auto_install_attempted=bool(wants_auto_install),
                auto_install_success=bool(state_result == "auto_install_success"),
                install_returncode=int(install_result.get("returncode", -1)) if install_result else -1,
            )
        except Exception as exc:
            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result="error",
                last_error=str(exc),
                last_error_ts=time.time(),
            )

    def _run_lab_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1~2단계 - Lab+사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 Lab/사전 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ Lab+사전 단계 건너뜀 (영어 Preview)")
                    return
                if hasattr(self, "_confirm_language_script_mismatch"):
                    if not self._confirm_language_script_mismatch(lang, wav_dir, stage_name="Lab+사전 생성"):
                        self._set_status("취소됨: 언어 설정 확인")
                        return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                targets, batch_scan_enabled = self._resolve_voicebank_batch_targets_for_ui(wav_dir)
                if not targets:
                    return
                self._append_voicebank_batch_summary(
                    targets,
                    batch_scan_enabled=batch_scan_enabled,
                    stage_name="Lab+사전 생성",
                )

                total_lab_count = 0
                total_lab_files = 0
                total_entries = 0
                for target in targets:
                    prefix = f"[Batch {target.index + 1}/{target.total}] " if target.total > 1 else ""
                    target_wav_dir = target.wav_dir
                    if target.total > 1:
                        self._append_log(f"{prefix}대상 폴더: {target_wav_dir}")
                    if lang == "japanese":
                        lab_count, lab_total, lab_errors = generate_ja_labs(
                            target_wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    else:
                        lab_count, lab_total, lab_errors = generate_labs(
                            target_wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    if lab_errors:
                        for err in lab_errors:
                            self._append_log(f"  ⚠ {err}")
                    self._append_log(f"{prefix}🧪 Lab 생성 완료 ({lab_count}/{lab_total})")
                    total_lab_count += int(lab_count or 0)
                    total_lab_files += int(lab_total or 0)

                    dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                    dict_path = os.path.join(target_wav_dir, dict_filename)
                    if lang == "japanese":
                        _count, entries, dict_errors = generate_ja_dictionary(
                            target_wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    else:
                        _count, entries, dict_errors = generate_dictionary(
                            target_wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    if dict_errors:
                        for err in dict_errors:
                            self._append_log(f"  ⚠ {err}")
                    total_entries += int(entries or 0)
                    self._append_log(f"{prefix}📝 사전 저장 경로: {dict_path}")
                self._set_status(
                    f"✅ Lab+사전 생성 완료 (Lab {total_lab_count}/{total_lab_files}, Dict {total_entries}개 항목)"
                )
            except Exception as e:
                self._handle_error("Lab+사전 생성", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1단계 - Lab 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 Lab 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ Lab 단계 건너뜀 (영어 Preview)")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                targets, batch_scan_enabled = self._resolve_voicebank_batch_targets_for_ui(wav_dir)
                if not targets:
                    return
                self._append_voicebank_batch_summary(
                    targets,
                    batch_scan_enabled=batch_scan_enabled,
                    stage_name="Lab 생성",
                )

                count = 0
                total = 0
                for target in targets:
                    prefix = f"[Batch {target.index + 1}/{target.total}] " if target.total > 1 else ""
                    if target.total > 1:
                        self._append_log(f"{prefix}대상 폴더: {target.wav_dir}")
                    if lang == "japanese":
                        c, t, errors = generate_ja_labs(
                            target.wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    else:
                        c, t, errors = generate_labs(
                            target.wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    count += int(c or 0)
                    total += int(t or 0)
                    if errors:
                        for e in errors:
                            self._append_log(f"  ⚠ {e}")
                    self._append_log(f"{prefix}Lab 생성 완료 ({c}/{t})")
                self._set_status(f"✅ Lab 생성 완료 ({count}/{total})")
            except Exception as e:
                self._handle_error("Lab 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("2단계 - 사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 사전 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ 사전 단계 건너뜀 (영어 Preview)")
                    return
                if hasattr(self, "_confirm_language_script_mismatch"):
                    if not self._confirm_language_script_mismatch(lang, wav_dir, stage_name=t("사전 생성")):
                        self._set_status("취소됨: 언어 설정 확인")
                        return
                targets, batch_scan_enabled = self._resolve_voicebank_batch_targets_for_ui(wav_dir)
                if not targets:
                    return
                self._append_voicebank_batch_summary(
                    targets,
                    batch_scan_enabled=batch_scan_enabled,
                    stage_name="사전 생성",
                )

                total_entries = 0
                for target in targets:
                    prefix = f"[Batch {target.index + 1}/{target.total}] " if target.total > 1 else ""
                    if target.total > 1:
                        self._append_log(f"{prefix}대상 폴더: {target.wav_dir}")
                    if lang == 'japanese':
                        dict_filename = "japanese_dict.txt"
                    else:
                        dict_filename = "korean_dict.txt"
                    dict_path = os.path.join(target.wav_dir, dict_filename)

                    if lang == 'japanese':
                        _count, entries, errors = generate_ja_dictionary(
                            target.wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    else:
                        _count, entries, errors = generate_dictionary(
                            target.wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=self._append_log,
                        )
                    if errors:
                        for e in errors:
                            self._append_log(f"  ⚠ {e}")
                    total_entries += int(entries or 0)
                    self._append_log(f"{prefix}📝 사전 저장 경로: {dict_path}")
                self._set_status(f"✅ 사전 생성 완료 ({total_entries}개 항목)")
            except Exception as e:
                self._handle_error(t("사전 생성"), e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_profile_finetune(self):
        def task():
            self._set_running(True)
            self._set_status("⚙ 프로파일 미세 조정 중...")
            try:
                if not is_training_paths_enabled():
                    self._append_log("⚠ 배포 빌드에서는 프로파일 훈련/미세 조정 기능이 비활성화되어 있습니다.")
                    self._set_status("⚠ 배포 빌드에서는 프로파일 훈련 기능을 사용할 수 없습니다.")
                    return

                auto_oto = self.tune_auto_oto_var.get().strip()
                manual_oto = self.tune_manual_oto_var.get().strip()
                profile_out = self.tune_profile_out_var.get().strip()
                apply_target = self.tune_apply_target_var.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if not auto_oto or not os.path.exists(auto_oto):
                    self._append_log("❌ 자동 생성 OTO 파일 경로가 올바르지 않습니다.")
                    return
                if not manual_oto or not os.path.exists(manual_oto):
                    self._append_log("❌ 수동 보정 OTO 파일 경로가 올바르지 않습니다.")
                    return

                if not profile_out:
                    base_dir = os.path.dirname(os.path.abspath(auto_oto))
                    if self._get_language() == "japanese":
                        profile_out = os.path.join(base_dir, ".ja_user_autotune_profile.json")
                    else:
                        profile_out = os.path.join(base_dir, ".kr_user_autotune_profile.json")
                    self.tune_profile_out_var.set(profile_out)

                if not apply_target:
                    apply_target = auto_oto
                    self.tune_apply_target_var.set(apply_target)
                if not os.path.exists(apply_target):
                    self._append_log(f"❌ 적용 대상 OTO 파일이 존재하지 않습니다: {apply_target}")
                    return

                lang = self._get_language()
                self._append_log(f"📝 프로파일 학습 시작 ({'일본어' if lang == 'japanese' else '한국어'})")
                self._append_log(f"   자동 OTO: {auto_oto}")
                self._append_log(f"   수동 OTO: {manual_oto}")

                if lang == "japanese":
                    from core.ja_oto_autotune import (
                        apply_ja_autotune_profile_to_oto,
                        save_ja_autotune_profile,
                        train_ja_autotune_profile,
                    )
                    profile = train_ja_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠ 학습 가능한 매칭 데이터가 부족합니다. (최소 8개 이상 권장)")
                        return
                    if not save_ja_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_ja_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )
                else:
                    from core.kr_oto_file_ops import (
                        apply_kr_autotune_profile_to_oto,
                        save_kr_autotune_profile,
                        train_kr_autotune_profile,
                    )
                    profile = train_kr_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠ 학습 가능한 매칭 데이터가 부족합니다. (최소 8개 이상 권장)")
                        return
                    if not save_kr_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_kr_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )

                pairs = int(profile.get("matched_pairs", 0))
                buckets = len((profile.get("buckets") or {}))
                self._append_log(f"✅ 프로파일 저장 완료: {profile_out}")
                self._append_log(f"✅ 학습 통계: matched_pairs={pairs}, buckets={buckets}")
                self._append_log(f"✅ 적용 완료: {changed} lines adjusted ({apply_target})")
                self._set_status(f"✅ 미세 조정 완료 ({changed} lines)")
            except Exception as e:
                self._handle_error("프로파일 기반 미세 조정", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_phoneme_boundary_smoke(self):
        """실보이스뱅크 경계 검출 스모크(매니페스트→학습→평가)를 UI에서 바로 실행합니다."""

        def task():
            self._set_running(True)
            self._set_status("경계 스모크 테스트 준비 중...")
            try:
                if not is_training_paths_enabled():
                    self._append_log("⚠ 배포 빌드에서는 경계 스모크 테스트 기능이 비활성화되어 있습니다.")
                    self._set_status("⚠ 배포 빌드에서는 경계 스모크 테스트를 사용할 수 없습니다.")
                    return

                if hasattr(self, "developer_mode_enabled_var") and not bool(self.developer_mode_enabled_var.get()):
                    self._append_log("⚠ 경계 스모크 테스트는 개발자 모드에서만 실행할 수 있습니다.")
                    self._set_status("⚠ 개발자 모드를 먼저 켜 주세요.")
                    return

                wav_dir = str(self.wav_entry.get() if hasattr(self, "wav_entry") else "").strip()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    self._set_status("❌ WAV 경로 누락")
                    return
                voicebank_dir = os.path.abspath(wav_dir)
                if not os.path.isdir(voicebank_dir):
                    self._append_log(f"❌ WAV 폴더가 존재하지 않습니다: {voicebank_dir}")
                    self._set_status("❌ WAV 경로 오류")
                    return

                source_oto = ""
                tpl_hint = str(self.tpl_entry.get() if hasattr(self, "tpl_entry") else "").strip()
                if tpl_hint and os.path.isfile(tpl_hint) and str(tpl_hint).lower().endswith(".ini"):
                    source_oto = os.path.abspath(tpl_hint)
                else:
                    candidate = os.path.join(voicebank_dir, "oto.ini")
                    if os.path.isfile(candidate):
                        source_oto = os.path.abspath(candidate)

                bank_name = os.path.basename(os.path.normpath(voicebank_dir)) or "voicebank"
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                smoke_rel_dir = os.path.join(
                    "ml_workspace", "phoneme_boundary", "ui_smoke", f"{bank_name}_{timestamp}"
                )
                base_dir = (
                    str(getattr(self, "writable_data_dir", "") or "").strip()
                    or str(getattr(self, "app_data_dir", "") or "").strip()
                    or str(getattr(self, "app_dir", "") or "").strip()
                    or os.getcwd()
                )
                out_dir = os.path.abspath(os.path.join(base_dir, smoke_rel_dir))
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except PermissionError:
                    fallback_base = (
                        str(os.environ.get("LOCALAPPDATA", "") or "").strip()
                        or str(os.environ.get("TEMP", "") or "").strip()
                        or os.getcwd()
                    )
                    out_dir = os.path.abspath(
                        os.path.join(fallback_base, "UTAU_Auto_OTO_v3", smoke_rel_dir)
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    self._append_log(f"[Boundary Smoke] 출력 경로 권한 문제로 폴백 경로를 사용합니다: {out_dir}")

                module_args = [
                    "-m",
                    "ml.scripts.phoneme_boundary.smoke_voicebank",
                    "--voicebank-dir",
                    voicebank_dir,
                    "--out-dir",
                    out_dir,
                    "--max-rows",
                    "80",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--max-frames",
                    "240",
                    "--n-mels",
                    "16",
                    "--hidden",
                    "8",
                    "--conv-channels",
                    "8",
                    "--num-workers",
                    "0",
                    "--device",
                    "auto",
                ]
                if source_oto:
                    module_args.extend(["--source-oto", source_oto])

                runner_candidates = []
                if str(getattr(sys, "executable", "") or "").strip():
                    runner_candidates.append([str(sys.executable)])
                runner_candidates.append(["py", "-3.11"])
                runner_candidates.append(["python"])

                last_return_code = None
                launch_errors = []
                ran = False
                for runner in runner_candidates:
                    cmd = list(runner) + module_args
                    pretty_cmd = " ".join(
                        f'"{part}"' if (" " in str(part) or "\t" in str(part)) else str(part) for part in cmd
                    )
                    self._append_log(f"[Boundary Smoke] 실행: {pretty_cmd}")
                    try:
                        process = self._popen_subprocess_hidden(
                            cmd,
                            cwd=str(getattr(self, "app_dir", "") or os.getcwd()),
                            stdout=sp.PIPE,
                            stderr=sp.STDOUT,
                            text=False,
                        )
                    except FileNotFoundError as e:
                        launch_errors.append(f"{runner[0]}: {e}")
                        continue
                    except Exception as e:
                        launch_errors.append(f"{runner[0]}: {e}")
                        continue

                    ran = True
                    for line in self._iter_decoded_stdout_lines(process):
                        self._append_log(f"[Boundary Smoke] {line}")
                    process.wait()
                    last_return_code = int(process.returncode)
                    if last_return_code == 0:
                        break
                    self._append_log(
                        f"[Boundary Smoke] 실행 실패(code={last_return_code}). 다음 Python 런너를 시도합니다."
                    )

                if not ran:
                    self._append_log("❌ 경계 스모크 테스트를 시작할 Python 런너를 찾지 못했습니다.")
                    for detail in launch_errors:
                        self._append_log(f"   - {detail}")
                    self._set_status("❌ Python 런너 없음")
                    return

                if last_return_code is None or int(last_return_code) != 0:
                    self._append_log(f"❌ 경계 스모크 테스트 실패 (code={last_return_code})")
                    self._append_log(f"   출력 폴더: {out_dir}")
                    self._set_status("❌ 경계 스모크 실패")
                    return

                summary_path = os.path.join(out_dir, "summary.json")
                self._append_log(f"✅ 경계 스모크 테스트 완료")
                self._append_log(f"   결과 폴더: {out_dir}")
                if os.path.isfile(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        eval_summary = payload.get("eval_summary") or {}
                        rows = int(eval_summary.get("rows", 0) or 0)
                        mae_ms = float(eval_summary.get("mae_ms", 0.0) or 0.0)
                        p90_ms = float(eval_summary.get("p90_ms", 0.0) or 0.0)
                        by_label = eval_summary.get("by_label") or {}
                        vowel_onset = by_label.get("vowel_onset") or {}
                        vowel_nucleus = by_label.get("vowel_nucleus") or {}
                        self._append_log(
                            f"[Boundary Smoke] rows={rows}, MAE={mae_ms:.2f}ms, P90={p90_ms:.2f}ms, "
                            f"vowel_onset_MAE={float(vowel_onset.get('mae_ms', 0.0) or 0.0):.2f}ms, "
                            f"vowel_nucleus_MAE={float(vowel_nucleus.get('mae_ms', 0.0) or 0.0):.2f}ms"
                        )
                    except Exception as e:
                        self._append_log(f"⚠ summary.json 파싱 실패: {e}")
                self._set_status("✅ 경계 스모크 테스트 완료")
            except Exception as e:
                self._handle_error("경계 스모크 테스트", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_phoneme_boundary_visualize(self):
        """현재 WAV 폴더에 대해 선택한 PhonemeBoundary 모델로 예측 JSON+PNG를 생성합니다."""

        def task():
            self._set_running(True)
            self._set_status("PhonemeBoundary 시각화 준비 중...")
            try:
                if hasattr(self, "developer_mode_enabled_var") and not bool(self.developer_mode_enabled_var.get()):
                    self._append_log("⚠ PhonemeBoundary 시각화는 개발자 모드에서만 실행할 수 있습니다.")
                    self._set_status("⚠ 개발자 모드를 먼저 켜 주세요.")
                    return

                wav_dir = str(self.wav_entry.get() if hasattr(self, "wav_entry") else "").strip()
                if not wav_dir or not os.path.isdir(wav_dir):
                    self._append_log("❌ WAV 폴더를 먼저 지정해 주세요.")
                    self._set_status("❌ WAV 경로 누락")
                    return
                wav_dir = os.path.abspath(wav_dir)

                models = getattr(self, "_phoneme_boundary_models", []) or []
                selected_label = str(self.phoneme_boundary_model_var.get() if hasattr(self, "phoneme_boundary_model_var") else "").strip()
                chosen = next((m for m in models if m.get("label") == selected_label), None)
                if chosen is None:
                    self._append_log("❌ 사용 가능한 PhonemeBoundary 모델이 없습니다. ml_workspace/models/phoneme_boundary/*.pt 를 확인해 주세요.")
                    self._set_status("❌ 모델 없음")
                    return
                model_path = str(chosen.get("path") or "").strip()

                bank_name = os.path.basename(os.path.normpath(wav_dir)) or "voicebank"
                model_stem = os.path.splitext(os.path.basename(model_path))[0]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rel = os.path.join("ml_workspace", "phoneme_boundary", "ui_visualize", f"{bank_name}_{model_stem}_{timestamp}")
                base_dir = (
                    str(getattr(self, "writable_data_dir", "") or "").strip()
                    or str(getattr(self, "app_data_dir", "") or "").strip()
                    or str(getattr(self, "app_dir", "") or "").strip()
                    or os.getcwd()
                )
                out_dir = os.path.abspath(os.path.join(base_dir, rel))
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except PermissionError:
                    fallback = str(os.environ.get("LOCALAPPDATA", "") or os.environ.get("TEMP", "") or os.getcwd())
                    out_dir = os.path.abspath(os.path.join(fallback, "UTAU_Auto_OTO_v3", rel))
                    os.makedirs(out_dir, exist_ok=True)

                runner_candidates = []
                if str(getattr(sys, "executable", "") or "").strip():
                    runner_candidates.append([str(sys.executable)])
                runner_candidates.append(["py", "-3.11"])
                runner_candidates.append(["python"])

                def _run(module_args, tag):
                    last_code = None
                    for runner in runner_candidates:
                        cmd = list(runner) + module_args
                        pretty = " ".join(f'"{p}"' if " " in str(p) else str(p) for p in cmd)
                        self._append_log(f"[{tag}] 실행: {pretty}")
                        try:
                            proc = self._popen_subprocess_hidden(
                                cmd,
                                cwd=str(getattr(self, "app_dir", "") or os.getcwd()),
                                stdout=sp.PIPE, stderr=sp.STDOUT, text=False,
                            )
                        except FileNotFoundError as e:
                            self._append_log(f"[{tag}] 런너 실패({runner[0]}): {e}")
                            continue
                        for line in self._iter_decoded_stdout_lines(proc):
                            self._append_log(f"[{tag}] {line}")
                        proc.wait()
                        last_code = int(proc.returncode)
                        if last_code == 0:
                            return 0
                        self._append_log(f"[{tag}] 실행 실패(code={last_code}), 다음 런너 시도")
                    return last_code if last_code is not None else -1

                predict_args = [
                    "-m", "scripts.dev.predict_phoneme_boundary_dir",
                    "--model", model_path,
                    "--wav-dir", wav_dir,
                    "--out-dir", out_dir,
                    "--device", "auto",
                ]
                rc = _run(predict_args, "PB Predict")
                if rc != 0:
                    self._append_log(f"❌ 예측 단계 실패 (code={rc})")
                    self._set_status("❌ 예측 실패")
                    return

                viz_args = [
                    "-m", "scripts.dev.visualize_phoneme_boundary",
                    "--batch", out_dir,
                    "--out-dir", out_dir,
                ]
                rc = _run(viz_args, "PB Visualize")
                if rc != 0:
                    self._append_log(f"⚠ 시각화 단계 실패 (code={rc}). JSON은 출력 폴더에 남아 있습니다.")
                    self._set_status("⚠ 시각화 실패 (JSON만 생성)")
                else:
                    self._append_log(f"✅ 완료. 결과 폴더: {out_dir}")
                    self._set_status("✅ PhonemeBoundary 시각화 완료")
                try:
                    os.startfile(out_dir)  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                self._handle_error("PhonemeBoundary 시각화", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_full_pipeline_batch_targets(
        self,
        targets: list[VoicebankBatchTarget],
        *,
        lang: str,
        selected_format: str,
        base_tpl_path: str,
        aligner_engine: str,
        no_mfa_mode_code: str,
        custom_phonemes_path: str,
    ) -> tuple[int, int]:
        success_count = 0
        failed_count = 0
        gen_dash_alias = self.gen_dash_alias_var.get() if hasattr(self, "gen_dash_alias_var") else True
        enable_ml_correction = self.enable_ml_correction_var.get()
        auto_format = self.auto_format_var.get()
        alias_suffix = self.alias_suffix_var.get().strip()
        auto_policy = self._apply_auto_ml_policy_env(
            lang,
            auto_format,
            enable_ml_default=enable_ml_correction,
            gen_dash_alias=gen_dash_alias,
        )
        enable_ml_correction = bool(auto_policy.get("enable_ml"))
        self._apply_advanced_tuning_envs()
        boundary_model_path = (
            self._apply_boundary_model_env() if hasattr(self, "_apply_boundary_model_env") else ""
        )
        self._append_log(
            f"ℹ 경계 인코더: {'학습 모델 ' + os.path.basename(boundary_model_path) if boundary_model_path else '규칙 기반(world_v1)'}"
        )
        self._append_log(
            f"[OTO-ML] auto policy: ml={'ON' if enable_ml_correction else 'OFF'}, "
            f"route={auto_policy.get('route')}, coupled={'ON' if auto_policy.get('has_coupled') else 'OFF'}, "
            f"hybrid={'ON' if auto_policy.get('hybrid_routing') else 'OFF'}"
        )

        # 일본어: 베이스 OTO가 Shift-JIS(cp932)면 생성 결과도 같은 인코딩으로 보존한다.
        preserve_out_encoding = ""
        base_oto_for_encoding = "" if self.no_base_oto_var.get() else str(base_tpl_path or "").strip()
        if lang == "japanese" and base_oto_for_encoding and os.path.isfile(base_oto_for_encoding):
            try:
                base_encoding = detect_oto_text_encoding(base_oto_for_encoding)
            except Exception:
                base_encoding = "utf-8"
            if str(base_encoding or "").strip().lower() not in {"utf-8", "utf_8", "utf-8-sig", "utf_8_sig"}:
                preserve_out_encoding = base_encoding
                self._append_log(f"ℹ 베이스 OTO 인코딩 감지: {base_encoding} → 생성 OTO 인코딩 보존")

        for target in targets:
            prefix = f"[Batch {target.index + 1}/{target.total}] " if target.total > 1 else ""
            target_wav_dir = target.wav_dir
            target_out_path = target.out_path
            target_tpl_path = "" if self.no_base_oto_var.get() else str(base_tpl_path or "").strip()
            textgrid_dir = os.path.join(target_wav_dir, "textgrids")
            self._append_log(f"{prefix}대상 폴더: {target_wav_dir}")
            self._append_log(f"{prefix}출력 경로: {target_out_path}")

            def _set_target_progress(local_ratio: float) -> None:
                try:
                    local = max(0.0, min(1.0, float(local_ratio)))
                except Exception:
                    local = 0.0
                self._set_progress((float(target.index) + local) / float(max(1, target.total)))

            def _make_target_callback(stage_start: float, stage_span: float):
                def _cb(msg):
                    text = str(msg or "")
                    self._append_log(f"{prefix}{text}")
                    ratio = self._parse_progress_ratio_from_status(text) if hasattr(self, "_parse_progress_ratio_from_status") else None
                    if ratio is not None:
                        _set_target_progress(stage_start + stage_span * float(ratio))
                return _cb

            cleanup_snapshot = self._snapshot_output_tree_for_cleanup(target_out_path)
            processed = 0
            total = 0
            errors: list[str] = []
            ok = False
            try:
                if lang == "english":
                    errors = ["영어 Preview CVVC는 배치 전체 실행에서 지원하지 않습니다."]
                elif lang == "korean" and selected_format == "cmpx":
                    source_oto = resolve_kr_cmpx_preview_source_oto(
                        wav_dir=target_wav_dir,
                        source_hint=target_tpl_path,
                    )
                    if not source_oto:
                        errors = ["한국어 CMPX Preview용 베이스 OTO를 찾지 못했습니다."]
                    else:
                        self._set_status(f"{prefix}KR CMPX Preview OTO 생성 중...")
                        processed, total, errors = generate_kr_cmpx_preview_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=source_oto,
                            alias_suffix=alias_suffix,
                            callback=_make_target_callback(0.20, 0.65),
                        )
                elif lang == "korean" and selected_format == "c_plus_v":
                    source_oto = resolve_no_mfa_source_oto(
                        wav_dir=target_wav_dir,
                        source_hint=target_tpl_path,
                    )
                    if not source_oto:
                        errors = ["한국어 C+V 모드용 베이스 OTO를 찾지 못했습니다."]
                    else:
                        self._set_status(f"{prefix}KR C+V OTO 생성 중...")
                        processed, total, errors = generate_no_mfa_auto_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=source_oto,
                            alias_suffix=alias_suffix,
                            language=lang,
                            stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                            generation_mode="remap",
                            callback=_make_target_callback(0.20, 0.65),
                        )
                elif aligner_engine == "hsmm_oto":
                    hsmm_source_oto = ""
                    if not bool(self.no_base_oto_var.get()):
                        hsmm_source_oto = resolve_no_mfa_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=target_tpl_path,
                        )
                    apply_hsmm_lightgbm = (
                        self._should_apply_hsmm_lightgbm(enable_ml_correction)
                        if hasattr(self, "_should_apply_hsmm_lightgbm")
                        else bool(enable_ml_correction)
                    )
                    self._append_log(
                        f"{prefix}HSMM OTO 시작: language={lang}, format={selected_format}, "
                        f"lightgbm={'ON' if apply_hsmm_lightgbm else 'OFF'}"
                    )
                    if bool(enable_ml_correction) and not apply_hsmm_lightgbm:
                        self._append_log(f"{prefix}[HSMM] LightGBM postprocess disabled by developer setting.")
                    if hsmm_source_oto:
                        self._append_log(f"{prefix}[HSMM] base oto: {hsmm_source_oto}")
                    else:
                        self._append_log(f"{prefix}[HSMM] no base oto: filename slots will be used.")
                    self._set_status(f"{prefix}HSMM OTO 생성 중...")
                    _set_target_progress(0.20)
                    processed, total, errors = self._run_hsmm_oto_preview_generation(
                        wav_dir=target_wav_dir,
                        out_path=target_out_path,
                        source_oto_path=hsmm_source_oto,
                        language=lang,
                        format_type=selected_format,
                        apply_lightgbm=apply_hsmm_lightgbm,
                        callback=_make_target_callback(0.20, 0.65),
                    )
                    self._append_log(f"{prefix}HSMM OTO 완료: processed={processed}/{total}, errors={len(errors or [])}")
                elif aligner_engine == "none":
                    if bool(self.no_base_oto_var.get()):
                        errors = ["No-MFA 자동설정 모드에서는 베이스 OTO가 필요합니다."]
                    else:
                        source_oto = resolve_no_mfa_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=target_tpl_path,
                        )
                        if not source_oto:
                            errors = ["No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다."]
                        else:
                            self._set_status(f"{prefix}No-MFA OTO 생성 중...")
                            processed, total, errors = generate_no_mfa_auto_oto(
                                wav_dir=target_wav_dir,
                                out_path=target_out_path,
                                source_oto_path=source_oto,
                                alias_suffix=alias_suffix,
                                language=lang,
                                stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                                generation_mode=no_mfa_mode_code,
                                callback=_make_target_callback(0.20, 0.65),
                            )
                else:
                    self._set_status(f"{prefix}Lab 파일 생성 중...")
                    if lang == "japanese":
                        lab_count, lab_total, lab_errors = generate_ja_labs(
                            target_wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=_make_target_callback(0.02, 0.18),
                        )
                    else:
                        lab_count, lab_total, lab_errors = generate_labs(
                            target_wav_dir,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=_make_target_callback(0.02, 0.18),
                        )
                    if lab_errors:
                        for err in lab_errors:
                            self._append_log(f"{prefix}⚠ {err}")
                    self._append_log(f"{prefix}Lab 생성 완료 ({lab_count}/{lab_total})")
                    _set_target_progress(0.22)

                    dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                    dict_path = os.path.join(target_wav_dir, dict_filename)
                    self._set_status(f"{prefix}사전 파일 생성 중...")
                    if lang == "japanese":
                        generate_ja_dictionary(
                            target_wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=_make_target_callback(0.22, 0.12),
                        )
                    else:
                        generate_dictionary(
                            target_wav_dir,
                            dict_path,
                            custom_phonemes_path=custom_phonemes_path,
                            callback=_make_target_callback(0.22, 0.12),
                        )
                    _set_target_progress(0.36)

                    primary_engine = normalize_aligner_name(aligner_engine, default="wfl")
                    if primary_engine == "wfl":
                        if not self._validate_alignment_input_files(target_wav_dir, dict_path):
                            errors = ["정렬 입력 파일 점검 실패"]
                    if not errors:
                        self._set_status(f"{prefix}정렬 실행 중...")
                        align_result = run_alignment_with_fallback(
                            language=lang,
                            wav_folder=target_wav_dir,
                            dictionary_path=dict_path,
                            output_folder=textgrid_dir,
                            primary_aligner=primary_engine,
                            fallback_aligner=("hsmm_oto" if primary_engine == "wfl" else ""),
                            format_hint=selected_format,
                            callback=_make_target_callback(0.36, 0.24),
                        )
                        if not bool(align_result.get("ok", False)):
                            self._append_log(
                                f"{prefix}⚠ 정렬 실패 상태로 OTO 생성을 계속합니다: {align_result.get('message', '')}"
                            )
                    _set_target_progress(0.62)

                    if not errors:
                        preflight = collect_runtime_preflight_issues(
                            language=lang,
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            aligner=self.aligner_var.get(),
                            textgrid_dir=textgrid_dir,
                            tpl_path=target_tpl_path,
                            no_mfa_oto_mode=no_mfa_mode_code,
                            no_base_oto=bool(self.no_base_oto_var.get()),
                            custom_phonemes_path=custom_phonemes_path,
                            require_output=True,
                        )
                        error_records = list(preflight.get("error_records") or [])
                        if error_records:
                            for item in error_records:
                                self._append_log(f"{prefix}❌ {item.get('display')}")
                            errors = ["사전 점검 실패"]

                    if not errors:
                        self._set_status(f"{prefix}OTO 생성 중...")
                        gen_ou = self.openutau_var.get()
                        gen_missing = self.gen_missing_vowels_var.get()
                        if lang == "japanese":
                            processed, total, errors = generate_ja_oto(
                                textgrid_dir,
                                target_tpl_path,
                                target_out_path,
                                params=None,
                                generate_openutau=gen_ou,
                                gen_missing_vowels=gen_missing,
                                enable_ml_correction=enable_ml_correction,
                                alias_style=self._get_ja_alias_style_code(),
                                ja_mapping_words_fallback_enabled=bool(
                                    self.ja_mapping_words_fallback_enabled_var.get()
                                    if hasattr(self, "ja_mapping_words_fallback_enabled_var")
                                    else True
                                ),
                                ja_mapping_spn_ratio_threshold=float(
                                    self.ja_mapping_spn_ratio_threshold_var.get()
                                    if hasattr(self, "ja_mapping_spn_ratio_threshold_var")
                                    else 0.35
                                ),
                                ja_mapping_min_vowel_phone_ratio=float(
                                    self.ja_mapping_min_vowel_phone_ratio_var.get()
                                    if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var")
                                    else 0.5
                                ),
                                ja_mapping_debug_reason_logging=bool(
                                    self.ja_mapping_debug_reason_logging_var.get()
                                    if hasattr(self, "ja_mapping_debug_reason_logging_var")
                                    else True
                                ),
                                auto_format=auto_format,
                                custom_phonemes_path=custom_phonemes_path,
                                alias_suffix=alias_suffix,
                                callback=_make_target_callback(0.62, 0.26),
                            )
                        else:
                            processed, total, errors = generate_oto(
                                textgrid_dir,
                                target_tpl_path,
                                target_out_path,
                                self._get_params(),
                                gen_ou,
                                gen_missing,
                                enable_ml_correction=enable_ml_correction,
                                auto_format=auto_format,
                                custom_phonemes_path=custom_phonemes_path,
                                alias_suffix=alias_suffix,
                                callback=_make_target_callback(0.62, 0.26),
                            )

                if errors:
                    self._append_log(f"{prefix}❌ 실패: {len(errors)}건")
                    for err in errors:
                        self._append_log(f"{prefix}  - {err}")
                elif not os.path.isfile(target_out_path):
                    errors = [f"OTO 파일 저장 실패: {target_out_path}"]
                    self._append_log(f"{prefix}❌ {errors[0]}")
                else:
                    _set_target_progress(0.90)
                    self._set_status(f"{prefix}OTO 자동 검증 중...")
                    self._run_auto_validation(
                        target_wav_dir,
                        textgrid_dir,
                        target_out_path,
                        callback=_make_target_callback(0.90, 0.08),
                    )
                    self._cleanup_generated_output_artifacts(target_out_path, snapshot=cleanup_snapshot)
                    if preserve_out_encoding:
                        try:
                            changed, used = reencode_oto_file(target_out_path, preserve_out_encoding)
                            if changed:
                                self._append_log(
                                    f"{prefix}ℹ 생성 OTO 인코딩 보존: {os.path.basename(target_out_path)} → {used}"
                                )
                        except Exception as enc_exc:
                            self._append_log(f"{prefix}⚠ OTO 인코딩 보존 실패({preserve_out_encoding}): {enc_exc}")
                    _set_target_progress(1.0)
                    self._append_log(f"{prefix}✅ 완료: {target_out_path} ({processed}/{total})")
                    ok = True
            except Exception as exc:
                self._append_log(f"{prefix}❌ 배치 처리 실패: {exc}")

            if ok:
                success_count += 1
            else:
                failed_count += 1

        if failed_count:
            self._set_status(f"완료(부분 성공): 배치 전체 실행 {success_count}/{len(targets)}")
        else:
            self._set_status(f"✅ 배치 전체 실행 완료 ({success_count}/{len(targets)})")
        return success_count, failed_count

    def _run_full_pipeline(self):
        """Lab 생성 → 정렬 → OTO 생성 → 검증 순서로 전체 파이프라인을 실행합니다."""
        def task():
            self._set_running(True)
            stage_order = ["lab", "dict", "align", "oto", "validate"]
            stage_weights = {
                "lab": 0.22,
                "dict": 0.12,
                "align": 0.28,
                "oto": 0.28,
                "validate": 0.10,
            }
            stage_progress = {name: 0.0 for name in stage_order}

            def _safe_ratio(value):
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    return 0.0

            def _overall_progress():
                return sum(stage_weights[name] * stage_progress.get(name, 0.0) for name in stage_order)

            def _set_stage_progress(stage, ratio):
                ratio_clamped = _safe_ratio(ratio)
                prev = stage_progress.get(stage, 0.0)
                if ratio_clamped < prev:
                    ratio_clamped = prev
                stage_progress[stage] = ratio_clamped
                self._set_progress(_overall_progress())

            def _extract_progress_ratio(msg):
                text = str(msg or "")
                ratio = self._parse_progress_ratio_from_status(text) if hasattr(self, "_parse_progress_ratio_from_status") else None
                if ratio is not None:
                    return ratio
                m = re.search(r"(?:files?|rows?|items?)\s*=\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
                if m:
                    total = int(m.group(2))
                    if total > 0:
                        return max(0.0, min(1.0, float(int(m.group(1))) / float(total)))
                return None

            def _make_stage_callback(stage):
                def _cb(msg):
                    self._append_log(msg)
                    ratio = _extract_progress_ratio(msg)
                    if ratio is not None:
                        _set_stage_progress(stage, ratio)
                return _cb

            try:
                # Step 1: Lab
                _set_stage_progress("lab", 0.02)
                self._set_status("1/5 - Lab 파일 생성 중...")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return
                out_path = self.out_entry.get().strip()
                cleanup_snapshot = self._snapshot_output_tree_for_cleanup(out_path) if out_path else None
                lang = self._get_language()
                self._append_log(
                    f"ℹ 현재 언어: {'일본어' if lang == 'japanese' else '한국어' if lang == 'korean' else '영어'}"
                )
                if lang in {"korean", "japanese"} and hasattr(self, "_confirm_language_script_mismatch"):
                    if not self._confirm_language_script_mismatch(lang, wav_dir, stage_name=t("전체 파이프라인")):
                        self._set_status("취소됨: 언어 설정 확인")
                        return
                selected_format = normalize_auto_format_value(
                    lang,
                    self.auto_format_var.get() if hasattr(self, "auto_format_var") else "",
                )
                textgrid_dir = os.path.join(wav_dir, "textgrids")
                has_textgrid = False
                if os.path.isdir(textgrid_dir):
                    try:
                        has_textgrid = any(str(name).lower().endswith(".textgrid") for name in os.listdir(textgrid_dir))
                    except Exception:
                        has_textgrid = False
                tpl_path_preflight = "" if self.no_base_oto_var.get() else self.tpl_entry.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                aligner_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
                    default="hsmm_oto",
                )
                hsmm_oto_mode = aligner_engine == "hsmm_oto"
                no_mfa_auto_mode = (
                    aligner_engine == "none"
                    and lang != "english"
                    and not (lang == "korean" and selected_format in {"cmpx", "c_plus_v"})
                )
                no_mfa_mode_code = (
                    self._get_no_mfa_oto_mode_code()
                    if hasattr(self, "_get_no_mfa_oto_mode_code")
                    else "remap"
                )
                if no_mfa_mode_code != "remap":
                    no_mfa_mode_code = "remap"
                no_mfa_mode_text = "베이스 OTO 재매핑 + 보정"
                no_mfa_source_oto = ""
                source_oto_required_for_no_mfa = True
                if self._is_recursive_voicebank_scan_enabled():
                    batch_targets, batch_scan_enabled = self._resolve_voicebank_batch_targets_for_ui(
                        wav_dir,
                        out_path,
                    )
                    if not batch_targets:
                        return
                    self._append_voicebank_batch_summary(
                        batch_targets,
                        batch_scan_enabled=batch_scan_enabled,
                        stage_name="전체 실행",
                    )
                    self._run_full_pipeline_batch_targets(
                        batch_targets,
                        lang=lang,
                        selected_format=selected_format,
                        base_tpl_path=tpl_path_preflight,
                        aligner_engine=aligner_engine,
                        no_mfa_mode_code=no_mfa_mode_code,
                        custom_phonemes_path=custom_phonemes_path,
                    )
                    return
                if no_mfa_auto_mode:
                    if source_oto_required_for_no_mfa and bool(self.no_base_oto_var.get()):
                        self._append_log("❌ No-MFA 자동설정 모드에서는 베이스 OTO(템플릿 ini)가 필요합니다.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    if source_oto_required_for_no_mfa or not bool(self.no_base_oto_var.get()):
                        no_mfa_source_oto = resolve_no_mfa_source_oto(
                            wav_dir=wav_dir,
                            source_hint=tpl_path_preflight,
                        )
                    if source_oto_required_for_no_mfa and not no_mfa_source_oto:
                        self._append_log("❌ No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다.")
                        self._append_log("   템플릿 OTO 경로에 baseoto.ini 또는 oto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    if no_mfa_source_oto:
                        tpl_path_preflight = no_mfa_source_oto

                if lang == "english":
                    if not getattr(self, "_is_preview_channel", lambda: False)():
                        self._append_log("❌ 영어 CVVC 모드는 Preview 채널에서만 사용할 수 있습니다.")
                        self._set_status("❌ Preview 전용 기능")
                        return
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    if self.no_base_oto_var.get() or not tpl_path_preflight or not os.path.isfile(tpl_path_preflight):
                        self._append_log("❌ 영어 Preview CVVC 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                        self._append_log("   템플릿 OTO 경로에 base oto.ini 또는 oto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return

                    self._append_log("ℹ 영어 Preview CVVC 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log(f"[EN-CVVC] base oto: {tpl_path_preflight}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    en_pack = self.en_cvvc_pack_var.get() if hasattr(self, "en_cvvc_pack_var") else "LITE"
                    en_beat = self.en_cvvc_beat_var.get() if hasattr(self, "en_cvvc_beat_var") else "8-beat"
                    en_preset = self.en_cvvc_preset_var.get() if hasattr(self, "en_cvvc_preset_var") else "Core"
                    en_list_fallback = (
                        bool(self.en_cvvc_list_fallback_var.get())
                        if hasattr(self, "en_cvvc_list_fallback_var")
                        else True
                    )
                    self._append_log(
                        f"[EN-CVVC] pack={en_pack} beat={en_beat} preset={en_preset} "
                        f"list_only_synth={'ON' if en_list_fallback else 'OFF'}"
                    )

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - EN CVVC OTO 생성 중...")
                    _processed, _total, oto_errors = generate_en_cvvc_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        pack=en_pack,
                        beat=en_beat,
                        preset=en_preset,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        include_list_only_synthesis=en_list_fallback,
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if lang == "korean" and selected_format == "cmpx":
                    if not getattr(self, "_is_preview_channel", lambda: False)():
                        self._append_log("❌ 한국어 CMPX 모드는 Preview 채널에서만 사용할 수 있습니다.")
                        self._set_status("❌ Preview 전용 기능")
                        return
                    if hasattr(self, "ml_route_var"):
                        try:
                            current_route = (
                                self._get_ml_route_code()
                                if hasattr(self, "_get_ml_route_code")
                                else str(self.ml_route_var.get() or "").strip().lower()
                            )
                        except Exception:
                            current_route = ""
                        if current_route != "nomfa":
                            try:
                                if hasattr(self, "_set_ml_route_from_code"):
                                    self._set_ml_route_from_code("nomfa")
                                else:
                                    self.ml_route_var.set("No-MFA")
                            except Exception:
                                pass
                            self._append_log("ℹ CMPX 모드 제한: ML route를 No-MFA로 고정합니다.")
                    os.environ["UTOA_ML_ROUTE"] = "autofree_v1"
                    os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "1"
                    os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "0"
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    if self.no_base_oto_var.get():
                        self._append_log("❌ 한국어 CMPX Preview 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    source_oto = resolve_kr_cmpx_preview_source_oto(
                        wav_dir=wav_dir,
                        source_hint=tpl_path_preflight,
                    )
                    if not source_oto:
                        self._append_log("❌ 한국어 CMPX Preview용 베이스 OTO를 찾지 못했습니다.")
                        self._append_log("   템플릿 OTO에 비교용 oto.ini 또는 baseoto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return

                    self._append_log("ℹ 한국어 CMPX Preview 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log(f"[KR-CMPX] base oto: {source_oto}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - KR CMPX Preview OTO 생성 중...")
                    _processed, _total, oto_errors = generate_kr_cmpx_preview_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=source_oto,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if lang == "korean" and selected_format == "c_plus_v":
                    if not getattr(self, "_is_preview_channel", lambda: False)():
                        self._append_log("❌ 한국어 C+V 모드는 Preview 채널에서만 사용할 수 있습니다.")
                        self._set_status("❌ Preview 전용 기능")
                        return
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    if self.no_base_oto_var.get():
                        self._append_log("❌ 한국어 C+V 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    source_oto = resolve_no_mfa_source_oto(
                        wav_dir=wav_dir,
                        source_hint=tpl_path_preflight,
                    )
                    if not source_oto:
                        self._append_log("❌ 한국어 C+V 모드용 베이스 OTO를 찾지 못했습니다.")
                        self._append_log("   템플릿 OTO에 baseoto.ini 또는 oto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return

                    self._append_log("ℹ 한국어 C+V 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log("ℹ 베이스 OTO가 0값(별칭 전용)이어도 remap 보정으로 파라미터를 자동 추정합니다.")
                    self._append_log(f"[KR-C+V] base oto: {source_oto}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - KR C+V OTO 생성 중...")
                    _processed, _total, oto_errors = generate_no_mfa_auto_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=source_oto,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        language=lang,
                        stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                        generation_mode="remap",
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if hsmm_oto_mode:
                    hsmm_source_oto = ""
                    if not bool(self.no_base_oto_var.get()):
                        hsmm_source_oto = resolve_no_mfa_source_oto(
                            wav_dir=wav_dir,
                            source_hint=tpl_path_preflight,
                        )
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    self._append_log("ℹ HSMM OTO 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log("ℹ 파일명 슬롯 순서와 음향 이벤트를 사용하며 source oto timing은 사용하지 않습니다.")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - HSMM OTO 생성 중...")
                    apply_hsmm_lightgbm = (
                        self._should_apply_hsmm_lightgbm(enable_ml_correction)
                        if hasattr(self, "_should_apply_hsmm_lightgbm")
                        else bool(enable_ml_correction)
                    )
                    if bool(enable_ml_correction) and not apply_hsmm_lightgbm:
                        self._append_log("[HSMM] LightGBM postprocess disabled by developer setting.")
                    _processed, _total, oto_errors = self._run_hsmm_oto_preview_generation(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=hsmm_source_oto,
                        language=lang,
                        format_type=selected_format,
                        apply_lightgbm=apply_hsmm_lightgbm,
                        callback=_make_stage_callback("oto"),
                    )
                    if oto_errors:
                        self._append_log(f"❌ OTO 생성 실패: {len(oto_errors)}건")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                        self._set_status(f"❌ OTO 생성 실패 ({_processed}/{_total})")
                        return
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                preflight = collect_runtime_preflight_issues(
                    language=lang,
                    wav_dir=wav_dir,
                    out_path=out_path,
                    aligner=self.aligner_var.get(),
                    textgrid_dir=textgrid_dir,
                    tpl_path=tpl_path_preflight,
                    no_mfa_oto_mode=no_mfa_mode_code,
                    no_base_oto=bool(self.no_base_oto_var.get()),
                    custom_phonemes_path=custom_phonemes_path,
                    require_output=False,
                )
                warning_records = list(preflight.get("warning_records") or [])
                error_records = list(preflight.get("error_records") or [])
                for item in warning_records:
                    self._append_log(f"⚠ {item.get('display')}")
                if error_records:
                    for item in error_records:
                        self._append_log(f"❌ {item.get('display')}")
                    first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                    self._set_status(f"❌ 사전 점검 실패 ({first_code})")
                    return

                if no_mfa_auto_mode:
                    self._append_log("ℹ No-MFA 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    if has_textgrid:
                        self._append_log("ℹ TextGrid가 있어도 No-MFA 선택 시에는 선택한 No-MFA 생성 방식으로 진행합니다.")
                    self._append_log(f"ℹ No-MFA 생성 방식: {no_mfa_mode_text}")
                    if no_mfa_source_oto:
                        self._append_log(f"[No-MFA] base oto: {no_mfa_source_oto}")
                    else:
                        self._append_log("[No-MFA] source oto not used.")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - No-MFA 자동설정 OTO 생성 중...")
                    _processed, _total, oto_errors = generate_no_mfa_auto_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=no_mfa_source_oto,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        language=lang,
                        stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                        generation_mode=no_mfa_mode_code,
                        callback=_make_stage_callback("oto"),
                    )
                    if oto_errors:
                        self._append_log(f"❌ OTO 생성 실패: {len(oto_errors)}건")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                        self._set_status(f"❌ OTO 생성 실패 ({_processed}/{_total})")
                        return
                    runtime_meta = get_last_no_mfa_runtime_meta()
                    confidence = float(runtime_meta.get("confidence", 0.0) or 0.0)
                    fallback_hint = str(runtime_meta.get("fallback_hint", "") or "")
                    fallback_used = bool(runtime_meta.get("fallback_used", False))
                    if runtime_meta:
                        self._append_log(
                            f"[No-MFA] confidence={confidence:.2f} "
                            f"fallback_hint={fallback_hint or '-'} fallback_used={'yes' if fallback_used else 'no'}"
                        )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if lang == 'japanese':
                    lab_count, lab_total, _lab_errors = generate_ja_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("lab"),
                    )
                else:
                    lab_count, lab_total, _lab_errors = generate_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("lab"),
                    )
                if lab_total:
                    _set_stage_progress("lab", float(lab_count) / float(lab_total))
                _set_stage_progress("lab", 1.0)

                # Step 2: Dictionary
                _set_stage_progress("dict", 0.05)
                self._set_status("2/5 - 사전 파일 생성 중...")
                dict_filename = "japanese_dict.txt" if lang == 'japanese' else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                if lang == 'japanese':
                    generate_ja_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("dict"),
                    )
                else:
                    generate_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("dict"),
                    )
                _set_stage_progress("dict", 1.0)

                # Step 3: Alignment
                output_dir = os.path.join(wav_dir, "textgrids")
                align_ok = False
                align_err = ""
                align_engine = self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO"
                primary_engine = normalize_aligner_name(align_engine, default="hsmm_oto")
                fallback_engine = ""
                _set_stage_progress("align", 0.05)
                if primary_engine in {"none", "hsmm_oto"}:
                    self._set_status("3/5 - 정렬 건너뛰기(no-MFA)")
                elif primary_engine == "sequence":
                    self._set_status("3/5 - 전용 시퀀스 정렬 준비 중...")
                elif primary_engine == "wfl":
                    # WFL runs in its own external env; no MFA readiness gate.
                    # Falls back to HSMM OTO when WFL is not ready/configured.
                    self._set_status("3/5 - WFL-ASR 정렬 준비 중...")
                    fallback_engine = "hsmm_oto"
                    if not self._validate_alignment_input_files(wav_dir, dict_path):
                        return
                if primary_engine == "sequence":
                    self._append_log("ℹ 정렬 엔진: 전용 시퀀스 baseline")
                elif primary_engine == "wfl":
                    self._append_log("ℹ 정렬 엔진: WFL-ASR (미준비 시 HSMM OTO 폴백)")
                else:
                    self._append_log(f"ℹ 정렬 엔진: {primary_engine}")
                if hasattr(self, "_apply_advanced_tuning_envs"):
                    self._apply_advanced_tuning_envs()

                align_result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary_engine,
                    fallback_aligner=fallback_engine,
                    format_hint=selected_format,
                    callback=_make_stage_callback("align"),
                )
                align_ok = bool(align_result.get("ok", False))
                align_err = str(align_result.get("message", "") or "")
                if align_result.get("fallback_used"):
                    self._append_log(f"ℹ 정렬 fallback 경로: {align_result.get('fallback_path', '')}")
                if primary_engine == "none":
                    has_textgrid = False
                    if os.path.isdir(output_dir):
                        for _name in os.listdir(output_dir):
                            if str(_name).lower().endswith(".textgrid"):
                                has_textgrid = True
                                break
                    if not has_textgrid:
                        self._append_log("❌ No-MFA 모드는 OTO 생성을 위해 TextGrid 입력이 필요합니다.")
                        self._append_log("   대안: 정렬 엔진을 MFA로 변경하거나 textgrids 폴더에 TextGrid를 미리 준비하세요.")
                        self._set_status("❌ No-MFA 입력 부족 (TextGrid 필요)")
                        return

                if not align_ok:
                    align_code = str(align_result.get("code", "ALIGN_RUN_FAILED"))
                    self._append_log(f"⚠ {format_error_with_recovery(align_code, align_err)}")
                    if self._is_lab_or_dict_missing_alignment_error(align_code, align_err):
                        self._notify_lab_or_dict_missing(wav_dir, dict_path)
                    self._append_log("⚠ 정렬 실패 상태로 다음 단계를 진행합니다.")
                _set_stage_progress("align", 1.0)

                # Step 4: OTO
                _set_stage_progress("oto", 0.03)
                self._set_status("4/5 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                if out_path:
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    gen_dash_alias = (
                        self.gen_dash_alias_var.get()
                        if hasattr(self, "gen_dash_alias_var")
                        else True
                    )
                    enable_ml_correction = self.enable_ml_correction_var.get()
                    # 공개 배포에서는 복잡한 ML 파라미터 대신 자동 정책을 사용합니다.
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
                    ml_route = (
                        str(self.ml_route_var.get()).strip().lower()
                        if hasattr(self, "ml_route_var")
                        else "autofree_v1"
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
                    ml_model_root = ""
                    if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
                        ml_model_root = str(self.ml_model_root_ja_var.get() or "").strip()
                    if lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
                        ml_model_root = str(self.ml_model_root_kr_var.get() or "").strip()
                    # (legacy tuning values removed)
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    ja_words_fallback = self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True
                    ja_spn_threshold = self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35
                    ja_min_vowel_ratio = self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5
                    ja_debug_reason = self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True
                    auto_policy = self._apply_auto_ml_policy_env(
                        lang,
                        auto_format,
                        enable_ml_default=enable_ml_correction,
                        gen_dash_alias=gen_dash_alias,
                    )
                    enable_ml_correction = bool(auto_policy.get("enable_ml"))
                    self._apply_advanced_tuning_envs()
                    self._append_log(
                        f"[OTO-ML] auto policy: ml={'ON' if enable_ml_correction else 'OFF'}, "
                        f"route={auto_policy.get('route')}, coupled={'ON' if auto_policy.get('has_coupled') else 'OFF'}, "
                        f"hybrid={'ON' if auto_policy.get('hybrid_routing') else 'OFF'}"
                    )
                    if not enable_ml_correction:
                        self._append_log("[OTO-ML] ML 모델이 없어 보정을 건너뜁니다.")
                    route_code = str(auto_policy.get("route", "") or "").strip().lower()
                    if route_code in {"nomfa", "v2"}:
                        self._append_log(f"[OTO-ML] route policy: {route_code} (coupled primary + autofree auxiliary)")
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없이 생성'이 활성화되어 OpenUtau 스타일로 생성합니다.")

                    oto_errors = []
                    generation_runtime_report = {}
                    if lang == 'japanese':
                        self._append_log(f"ℹ 일본어 별칭 스타일: {self.ja_alias_style_var.get()}")
                        _processed, _total, oto_errors = generate_ja_oto(
                            tg_folder, tpl_path, out_path,
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
                            callback=_make_stage_callback("oto"),
                            runtime_report=generation_runtime_report,
                        )
                    else:
                        _processed, _total, oto_errors = generate_oto(
                            tg_folder, tpl_path, out_path,
                            params,
                            gen_ou,
                            gen_missing,
                            enable_ml_correction=enable_ml_correction,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=_make_stage_callback("oto"),
                            runtime_report=generation_runtime_report,
                        )
                    if isinstance(generation_runtime_report, dict):
                        gen_code = str(generation_runtime_report.get("code", "") or "")
                        gen_msg = str(generation_runtime_report.get("message", "") or "")
                        gen_processed = int(generation_runtime_report.get("processed", _processed) or _processed)
                        gen_total = int(generation_runtime_report.get("total", _total) or _total)
                        self._append_log(
                            f"[Generate] code={gen_code or '-'} "
                            f"processed={gen_processed}/{gen_total} "
                            f"message={gen_msg or '-'}"
                        )
                        gen_mapping = (
                            generation_runtime_report.get("mapping")
                            if isinstance(generation_runtime_report.get("mapping"), dict)
                            else {}
                        )
                        if gen_mapping:
                            self._append_log(
                                "[Generate][Mapping] " + format_mapping_summary(gen_mapping)
                            )
                            self._append_log(
                                "[Generate][MappingReason] "
                                + format_mapping_reason_schema_summary(gen_mapping)
                            )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, tg_folder, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있어 자동 정리를 건너뜁니다. ({len(oto_errors)}건)")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)
                else:
                    self._append_log("⚠ 출력 경로가 없어 OTO 생성을 건너뜁니다.")
                    _set_stage_progress("oto", 1.0)
                    _set_stage_progress("validate", 1.0)

                self._set_status("✅ 전체 파이프라인 완료")
                self._append_log("\n" + "=" * 50)
                self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error(t("전체 파이프라인"), e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)
