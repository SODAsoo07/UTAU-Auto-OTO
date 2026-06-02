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
from core.mfa_runner import (
    ALERT_MSVC_REQUIRED,
    MFA_PORTABLE_PYTHON_VERSION,
    check_mfa_model,
    diagnose_mfa_runtime,
    download_mfa_model,
    ensure_mfa_python_packaging_stack,
    ensure_japanese_support,
    ensure_korean_support,
    get_default_mfa_conda_root,
    get_default_mfa_env_dir,
    get_default_mfa_micromamba_exe,
    get_default_mfa_micromamba_root,
    get_mfa_env_python_version,
    find_mfa_executable,
    mfa_env_requires_python_downgrade,
)
from core.oto_generator import generate_oto
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues
from core.error_codes import format_error_with_recovery
from core.format_type_utils import normalize_auto_format_value
from core.distribution_guard import is_training_paths_enabled
from core.runtime_paths import resolve_setup_mfa_script_path
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
    def _is_mfa_module_missing_error(code: str, message: str) -> bool:
        c = str(code or "").strip().upper()
        text = str(message or "")
        lowered = text.lower()
        if c == "ALIGN_EXEC_MISSING" and ("mfa runtime module is missing" in lowered or "mfa_module_missing" in lowered):
            return True
        if "no module named" in lowered and "montreal_forced_aligner" in lowered:
            return True
        if "montreal_forced_aligner.command_line.mfa" in lowered:
            return True
        return False

    def _notify_mfa_module_missing(self, language: str, detail: str = "") -> None:
        lang = str(language or "korean").strip().lower()
        lang_label = "일본어" if lang == "japanese" else "한국어"
        guide = self._build_setup_mfa_recovery_guide(language=lang)
        tail = str(detail or "").strip()
        if len(tail) > 420:
            tail = tail[-420:]
        message = (
            f"{lang_label} MFA 실행 모듈이 현재 환경에서 누락되었거나 손상되었습니다.\n"
            "정렬을 계속 진행하려면 MFA 복구를 먼저 실행해 주세요.\n\n"
            f"{guide}"
        )
        if tail:
            message += f"\n\n[오류 요약]\n{tail}"
        self._after_safe(
            lambda msg=message, l=lang: self._show_copyable_alert(
                title="MFA 모듈 누락 감지",
                message=msg,
                alert_key=f"mfa_module_missing_{l}",
            )
        )

    def _notify_korean_dependency_degraded(self):
        self._append_log(ALERT_MSVC_REQUIRED)
        self._append_log(
            "⚠ 일부 의존성 설치에 실패했습니다. 사용은 가능하지만 정확도에 영향이 있을 수 있으니 C++ 툴을 설치해주세요."
        )
        self._append_log("설치 링크: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

    def _notify_long_install_time(self, target="MFA"):
        title = "MFA 초기 설치 안내"
        message = (
            "처음 MFA를 설치하거나 진단/복구를 실행할 때는 환경 구성, Python 패키지 복구, 현재 언어 모델 다운로드 때문에 시간이 오래 걸릴 수 있습니다.\n\n"
            "환경과 네트워크 속도에 따라 보통 10~20분, 경우에 따라 그 이상 걸릴 수 있습니다.\n"
            "설치 중에는 프로그램을 종료하지 말고 기다려 주세요.\n\n"
            "권장 순서:\n"
            "1) 'MFA 진단/복구' 버튼으로 현재 상태 점검\n"
            "2) 필요한 복구와 모델 다운로드 자동 진행\n"
            "3) 완료 후 정렬 다시 시도"
        )
        alert_key = "install_time_mfa"
        self._append_log(f"ℹ {title}: 설치에 시간이 오래 걸릴 수 있습니다.")
        self._after_safe(
            lambda: self._show_copyable_alert(
                title=title,
                message=message,
                alert_key=alert_key,
            )
        )

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

    def _resolve_setup_mfa_script_path(self):
        return resolve_setup_mfa_script_path(
            app_dir=str(getattr(self, "app_dir", "") or ""),
            app_data_dir=str(getattr(self, "app_data_dir", "") or ""),
            writable_data_dir=str(getattr(self, "writable_data_dir", "") or ""),
            frozen=bool(getattr(sys, "frozen", False)),
            executable_path=str(getattr(sys, "executable", "") or ""),
        )

    def _run_setup_mfa_script_fallback(self, language="korean", reason="") -> bool:
        script_path = self._resolve_setup_mfa_script_path()
        if not script_path:
            self._append_log("⚠ setup_mfa.bat을 찾지 못해 배치 폴백 복구를 건너뜁니다.")
            return False

        lang = str(language or "korean").strip().lower()
        self._append_log("⚠ 내부 MFA 복구가 실패해 setup_mfa.bat 폴백 복구를 시도합니다.")
        if reason:
            self._append_log(f"   사유: {reason}")

        cmd = [script_path, "--recovery", "--non-interactive", "--language", lang]
        env = os.environ.copy()
        try:
            env_dir = get_default_mfa_env_dir()
            runtime_root = os.path.dirname(env_dir)
            if runtime_root:
                env["UTOA_MFA_SHARED_ROOT"] = runtime_root
        except Exception:
            pass
        self._append_log(f"   실행: {' '.join(cmd)}")

        try:
            process = self._popen_subprocess_hidden(
                cmd,
                cwd=os.path.dirname(script_path) or None,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=False,
                env=env,
            )
        except Exception as e:
            self._append_log(f"❌ setup_mfa.bat 실행 실패: {e}")
            return False

        if process.stdout is not None:
            for line in self._iter_decoded_stdout_lines(process):
                self._append_log(line)
        process.wait()
        if process.returncode != 0:
            self._append_log(f"❌ setup_mfa.bat 폴백 복구 실패 (code={process.returncode})")
            return False

        resolved = find_mfa_executable() or ""
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved

        report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
        if report.get("ready"):
            self._append_log("✅ setup_mfa.bat 폴백 복구 완료")
            return True
        self._append_log("❌ setup_mfa.bat 실행은 완료됐지만 MFA 상태가 아직 준비되지 않았습니다.")
        return False

    def _build_setup_mfa_recovery_guide(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        if lang not in {"korean", "japanese"}:
            lang = "korean"
        script_path = self._resolve_setup_mfa_script_path()
        if script_path:
            command = f'cmd /c ""{script_path}" --recovery --non-interactive --language {lang}"'
            return (
                "설치 프로그램에 동봉된 setup_mfa.bat을 직접 실행해 추가 복구를 진행해 주세요.\n"
                f"- 파일: {script_path}\n"
                "- 실행 명령:\n"
                f"{command}"
            )
        return (
            "설치 프로그램에 동봉된 setup_mfa.bat을 직접 실행해 추가 복구를 진행해 주세요.\n"
            "- 실행 예시:\n"
            f"cmd /c \"\"%LOCALAPPDATA%\\UTAU_Auto_OTO_v3\\setup_mfa.bat\" --recovery --non-interactive --language {lang}\"\n"
            "또는\n"
            f"cmd /c \"\"%LOCALAPPDATA%\\UTAU_Auto_OTO\\setup_mfa.bat\" --recovery --non-interactive --language {lang}\""
        )

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

    def _install_mfa_runtime(self, language="korean"):
        import shutil
        import tempfile

        lang = str(language or "korean").strip().lower()
        app_dir = getattr(self, "app_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_dir = get_default_mfa_env_dir()
        legacy_conda_root = get_default_mfa_conda_root()
        micromamba_root = get_default_mfa_micromamba_root()
        runtime_root = os.path.dirname(env_dir)

        def _resolve_micromamba_exe_path(root: str) -> str:
            candidates = [
                os.path.join(root, "Library", "bin", "micromamba.exe"),
                os.path.join(root, "bin", "micromamba.exe"),
                os.path.join(root, "micromamba.exe"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return candidates[0]

        def _ensure_writable_dir(path: str) -> tuple[bool, str]:
            try:
                os.makedirs(path, exist_ok=True)
                fd, probe = tempfile.mkstemp(prefix=".utoa_write_probe_", dir=path)
                os.close(fd)
                os.remove(probe)
                return True, ""
            except Exception as e:
                return False, str(e)

        root_ok, root_err = _ensure_writable_dir(runtime_root)
        if not root_ok:
            local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
            if local_app_data:
                fallback_root = os.path.join(local_app_data, "UTAU_Auto_OTO_v3")
            else:
                fallback_root = os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3")
            fallback_ok, fallback_err = _ensure_writable_dir(fallback_root)
            if fallback_ok:
                self._append_log(f"⚠ 기본 MFA 경로 쓰기 실패: {runtime_root} ({root_err})")
                self._append_log(f"ℹ LOCALAPPDATA 경로로 전환: {fallback_root}")
                runtime_root = fallback_root
                env_dir = os.path.join(runtime_root, ".env")
                legacy_conda_root = os.path.join(runtime_root, "miniconda")
                micromamba_root = os.path.join(runtime_root, "micromamba")
            else:
                self._append_log(f"❌ MFA 런타임 경로를 준비하지 못했습니다: {runtime_root}")
                if root_err:
                    self._append_log(f"   원인: {root_err}")
                self._append_log(f"❌ LOCALAPPDATA 대체 경로도 실패: {fallback_root}")
                if fallback_err:
                    self._append_log(f"   원인: {fallback_err}")
                return False

        micromamba_exe = _resolve_micromamba_exe_path(micromamba_root)
        if any(ord(ch) > 127 for ch in app_dir):
            self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어도 MFA 환경은 공용 폴더를 사용합니다.")
        self._append_log(f"ℹ MFA 공용 환경 경로: {env_dir}")
        self._append_log(f"ℹ MFA Micromamba 경로: {micromamba_root}")
        mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
        micromamba_archive = os.path.join(runtime_root, 'micromamba-win-64-latest.tar.bz2')
        vc_redist_exe = os.path.join(runtime_root, 'vc_redist.x64.exe')
        vc_redist_url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        vc_runtime_markers = [
            os.path.join(system_root, "System32", "msvcp140.dll"),
            os.path.join(system_root, "System32", "msvcp140_1.dll"),
            os.path.join(system_root, "System32", "vcruntime140.dll"),
            os.path.join(system_root, "System32", "vcruntime140_1.dll"),
        ]
        archive_url = 'https://micro.mamba.pm/api/micromamba/win-64/latest'
        direct_exe_url = 'https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64'

        def _looks_like_tls_error(text: str) -> bool:
            msg = str(text or "").lower()
            return (
                "trust relationship for the ssl/tls secure channel" in msg
                or "ssl/tls secure channel" in msg
                or "certificate" in msg
                or "tls" in msg
            )

        def _looks_like_vc_runtime_missing(text: str) -> bool:
            msg = str(text or "").lower()
            return (
                "msvcp140.dll" in msg
                or "msvcp140_1.dll" in msg
                or "vcruntime140.dll" in msg
                or "vcruntime140_1.dll" in msg
                or "side-by-side configuration is incorrect" in msg
            )

        def _attempt_powershell_download(url: str, out_path: str) -> tuple[bool, str]:
            ps_cmd = (
                f"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
                f"Invoke-WebRequest -Uri '{url}' -OutFile '{out_path}'"
            )
            try:
                result = self._run_subprocess_hidden(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                return False, str(e)
            ok = result.returncode == 0 and os.path.exists(out_path)
            msg = (result.stderr or result.stdout or "").strip()
            return ok, msg

        def _attempt_curl_download(url: str, out_path: str) -> tuple[bool, str]:
            try:
                result = self._run_subprocess_hidden(
                    ['curl.exe', '-L', '--fail', '--retry', '2', '--retry-delay', '2', '-o', out_path, url],
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return False, "curl.exe not found"
            except Exception as e:
                return False, str(e)
            ok = result.returncode == 0 and os.path.exists(out_path)
            msg = (result.stderr or result.stdout or "").strip()
            return ok, msg

        def _vc_runtime_ready() -> bool:
            return all(os.path.exists(path) for path in vc_runtime_markers)

        def _ensure_vc_runtime() -> bool:
            if _vc_runtime_ready():
                return True
            self._append_log("🔧 Microsoft VC++ 런타임 점검/복구 중...")
            if not os.path.exists(vc_redist_exe):
                self._append_log("ℹ VC++ 재배포 패키지 다운로드 중...")
                ok, msg = _attempt_powershell_download(vc_redist_url, vc_redist_exe)
                if not ok:
                    if msg:
                        self._append_log(f"⚠ VC++ 런타임 다운로드 1차 실패(PowerShell): {msg}")
                    ok, msg = _attempt_curl_download(vc_redist_url, vc_redist_exe)
                if not ok:
                    self._append_log(f"❌ VC++ 런타임 다운로드 실패: {msg}")
                    if _looks_like_tls_error(msg):
                        self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 후 다시 시도하세요.")
                    self._append_log("   수동 설치 링크: https://aka.ms/vs/17/release/vc_redist.x64.exe")
                    return False
            for args, label in (
                (['/install', '/quiet', '/norestart'], "silent"),
                (['/install', '/passive', '/norestart'], "passive"),
            ):
                try:
                    result = self._run_subprocess_hidden(
                        [vc_redist_exe, *args],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                except Exception as e:
                    self._append_log(f"⚠ VC++ 런타임 설치 실행 실패({label}): {e}")
                    continue
                if result.returncode in {0, 1638, 3010} and _vc_runtime_ready():
                    self._append_log("✅ VC++ 런타임 준비 완료")
                    return True
                out = (result.stderr or result.stdout or "").strip()
                if out:
                    self._append_log(f"⚠ VC++ 런타임 설치 로그({label}): {out[:500]}")
            if _vc_runtime_ready():
                self._append_log("✅ VC++ 런타임 준비 완료")
                return True
            self._append_log("⚠ VC++ 런타임 자동 복구에 실패했습니다.")
            self._append_log("   수동 설치 후 다시 시도: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False

        def _remove_env_dir():
            if not os.path.isdir(env_dir):
                return True
            self._append_log(f"🧹 기존 MFA 환경 정리 중: {env_dir}")
            try:
                shutil.rmtree(env_dir)
                return True
            except Exception as e:
                self._append_log(f"❌ 기존 MFA 환경 정리 실패: {e}")
                return False

        def _remove_legacy_conda_root():
            if not os.path.isdir(legacy_conda_root):
                return True
            try:
                shutil.rmtree(legacy_conda_root)
                self._append_log(f"🧹 기존 Miniconda 흔적 정리: {legacy_conda_root}")
                return True
            except Exception as e:
                self._append_log(f"⚠ 기존 Miniconda 폴더 정리 실패: {e}")
                return False

        def _download_micromamba():
            if os.path.exists(micromamba_archive):
                self._append_log("ℹ Micromamba 아카이브가 이미 있어 재사용합니다.")
                return True
            if os.path.exists(micromamba_exe):
                self._append_log("ℹ Micromamba 실행 파일이 이미 있어 재사용합니다.")
                return True
            try:
                os.makedirs(os.path.dirname(micromamba_archive), exist_ok=True)
            except Exception as e:
                self._append_log(f"❌ Micromamba 다운로드 폴더 생성 실패: {e}")
                self._append_log(f"   경로: {os.path.dirname(micromamba_archive)}")
                return False
            self._set_status("⬇ Micromamba 다운로드 중...")
            self._append_log("[1/3] ⬇ Micromamba 다운로드 중... (약 15MB)")

            ok, msg = _attempt_powershell_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 2차 실패(curl): {msg}")

            self._append_log("ℹ 기본 URL 실패. GitHub 미러로 재시도합니다.")
            os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)

            ok, msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 미러 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            self._append_log(f"❌ Micromamba 다운로드 실패: {msg}")
            if _looks_like_tls_error(msg):
                self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 또는 수동 파일 배치를 진행하세요.")
            return False

        def _extract_micromamba():
            if os.path.exists(micromamba_exe):
                self._append_log("✅ Micromamba 실행 파일이 이미 있습니다.")
                return True
            self._set_status("📦 Micromamba 압축 해제 중...")
            self._append_log("[2/3] 📦 Micromamba 압축 해제 중...")
            try:
                if os.path.isdir(micromamba_root):
                    shutil.rmtree(micromamba_root)
                os.makedirs(micromamba_root, exist_ok=True)
                result = self._run_subprocess_hidden(
                    ['tar', '-xjf', micromamba_archive, '-C', micromamba_root],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    self._append_log(f"⚠ Micromamba 압축 해제 실패: {result.stderr or result.stdout}")
                    self._append_log("ℹ 압축 해제 없이 단일 micromamba.exe 경로로 재시도합니다.")
                    os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)
                    ok, _msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        ok, _msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        self._append_log(f"❌ Micromamba 실행 파일 준비 실패: {_msg}")
                        return False
                    return True
                resolved = get_default_mfa_micromamba_exe()
                if not os.path.exists(resolved):
                    self._append_log("❌ Micromamba 실행 파일을 찾지 못했습니다.")
                    return False
                return True
            except Exception as e:
                self._append_log(f"❌ Micromamba 준비 실패: {e}")
                return False

        def _run_micromamba(cmd, step_label, retry_on_vc_runtime=True):
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = micromamba_root
            try:
                process = self._popen_subprocess_hidden(
                    [micromamba_exe, *cmd],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=False,
                    env=env,
                )
            except OSError as e:
                err = str(e)
                if retry_on_vc_runtime and _looks_like_vc_runtime_missing(err):
                    self._append_log("⚠ VC++ 런타임 누락으로 Micromamba 실행에 실패했습니다. 자동 복구 후 재시도합니다.")
                    if _ensure_vc_runtime():
                        return _run_micromamba(cmd, step_label, retry_on_vc_runtime=False)
                self._append_log(f"❌ Micromamba 실행 실패: {err}")
                return False
            self._append_log(step_label)
            for line in self._iter_decoded_stdout_lines(process):
                self._append_log(line)
            process.wait()
            return process.returncode == 0

        def _refresh_mfa_path():
            nonlocal mfa_exe
            resolved = find_mfa_executable()
            if resolved and os.path.exists(resolved):
                mfa_exe = resolved
                self.mfa_path = resolved
                self._append_log(f"ℹ 감지된 MFA 실행 파일: {resolved}")
                return True
            fallback_candidates = [
                os.path.join(env_dir, 'python.exe'),
            ]
            python_exe = fallback_candidates[0]
            if os.path.exists(micromamba_exe):
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'set "CONDA_PREFIX={env_dir}"\r\n')
                        wf.write(
                            f'set "PATH={env_dir};{env_dir}\\Library\\mingw-w64\\bin;'
                            f'{env_dir}\\Library\\usr\\bin;{env_dir}\\Library\\bin;'
                            f'{env_dir}\\Scripts;{env_dir}\\bin;%PATH%"\r\n'
                        )
                        wf.write(f'"{env_dir}\\python.exe" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 배치 래퍼 생성 실패: {e}")
            if os.path.exists(python_exe):
                probe = self._run_subprocess_hidden(
                    [
                        python_exe,
                        '-c',
                        "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') else 1)",
                    ],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    return False
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'"{python_exe}" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 래퍼 생성 실패: {e}")
            return False

        if not os.path.exists(mfa_exe):
            _refresh_mfa_path()

        def _remove_env_dir():
            if not os.path.isdir(env_dir):
                return True
            self._append_log(f"🧹 기존 MFA 환경 정리 중: {env_dir}")
            try:
                shutil.rmtree(env_dir)
                return True
            except Exception as e:
                self._append_log(f"❌ 기존 MFA 환경 정리 실패: {e}")
                return False

        def _remove_legacy_conda_root():
            if not os.path.isdir(legacy_conda_root):
                return True
            try:
                shutil.rmtree(legacy_conda_root)
                self._append_log(f"🧹 기존 Miniconda 흔적 정리: {legacy_conda_root}")
                return True
            except Exception as e:
                self._append_log(f"⚠ 기존 Miniconda 폴더 정리 실패: {e}")
                return False

        def _download_micromamba():
            if os.path.exists(micromamba_archive):
                self._append_log("ℹ Micromamba 아카이브가 이미 있어 재사용합니다.")
                return True
            if os.path.exists(micromamba_exe):
                self._append_log("ℹ Micromamba 실행 파일이 이미 있어 재사용합니다.")
                return True
            self._set_status("⬇ Micromamba 다운로드 중...")
            self._append_log("[1/3] ⬇ Micromamba 다운로드 중... (약 15MB)")

            ok, msg = _attempt_powershell_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 2차 실패(curl): {msg}")

            self._append_log("ℹ 기본 URL 실패. GitHub 미러로 재시도합니다.")
            os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)

            ok, msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 미러 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            self._append_log(f"❌ Micromamba 다운로드 실패: {msg}")
            if _looks_like_tls_error(msg):
                self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 또는 수동 파일 배치를 진행하세요.")
            return False

        def _extract_micromamba():
            if os.path.exists(micromamba_exe):
                self._append_log("✅ Micromamba 실행 파일이 이미 있습니다.")
                return True
            self._set_status("📦 Micromamba 압축 해제 중...")
            self._append_log("[2/3] 📦 Micromamba 압축 해제 중...")
            try:
                if os.path.isdir(micromamba_root):
                    shutil.rmtree(micromamba_root)
                os.makedirs(micromamba_root, exist_ok=True)
                result = sp.run(
                    ['tar', '-xjf', micromamba_archive, '-C', micromamba_root],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    self._append_log(f"⚠ Micromamba 압축 해제 실패: {result.stderr or result.stdout}")
                    self._append_log("ℹ 압축 해제 없이 단일 micromamba.exe 경로로 재시도합니다.")
                    os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)
                    ok, _msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        ok, _msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        self._append_log(f"❌ Micromamba 실행 파일 준비 실패: {_msg}")
                        return False
                    return True
                resolved = get_default_mfa_micromamba_exe()
                if not os.path.exists(resolved):
                    self._append_log("❌ Micromamba 실행 파일을 찾지 못했습니다.")
                    return False
                return True
            except Exception as e:
                self._append_log(f"❌ Micromamba 준비 실패: {e}")
                return False

        def _run_micromamba(cmd, step_label, retry_on_vc_runtime=True):
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = micromamba_root
            try:
                process = self._popen_subprocess_hidden(
                    [micromamba_exe, *cmd],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=False,
                    env=env,
                )
            except OSError as e:
                err = str(e)
                if retry_on_vc_runtime and _looks_like_vc_runtime_missing(err):
                    self._append_log("⚠ VC++ 런타임 누락으로 Micromamba 실행에 실패했습니다. 자동 복구 후 재시도합니다.")
                    if _ensure_vc_runtime():
                        return _run_micromamba(cmd, step_label, retry_on_vc_runtime=False)
                self._append_log(f"❌ Micromamba 실행 실패: {err}")
                return False
            self._append_log(step_label)
            for line in self._iter_decoded_stdout_lines(process):
                self._append_log(line)
            process.wait()
            return process.returncode == 0

        def _refresh_mfa_path():
            nonlocal mfa_exe
            resolved = find_mfa_executable()
            if resolved and os.path.exists(resolved):
                mfa_exe = resolved
                self.mfa_path = resolved
                self._append_log(f"ℹ 감지된 MFA 실행 파일: {resolved}")
                return True
            fallback_candidates = [
                os.path.join(env_dir, 'python.exe'),
            ]
            python_exe = fallback_candidates[0]
            if os.path.exists(micromamba_exe):
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'set "CONDA_PREFIX={env_dir}"\r\n')
                        wf.write(
                            f'set "PATH={env_dir};{env_dir}\\Library\\mingw-w64\\bin;'
                            f'{env_dir}\\Library\\usr\\bin;{env_dir}\\Library\\bin;'
                            f'{env_dir}\\Scripts;{env_dir}\\bin;%PATH%"\r\n'
                        )
                        wf.write(f'"{env_dir}\\python.exe" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 배치 래퍼 생성 실패: {e}")
            if os.path.exists(python_exe):
                probe = self._run_subprocess_hidden(
                    [
                        python_exe,
                        '-c',
                        "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') else 1)",
                    ],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    return False
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'"{python_exe}" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 래퍼 생성 실패: {e}")
            return False

        if not os.path.exists(mfa_exe):
            _refresh_mfa_path()

        if os.path.exists(mfa_exe):
            py_ver = get_mfa_env_python_version(mfa_exe)
            if mfa_env_requires_python_downgrade(mfa_exe):
                self._append_log(
                    f"⚠ 현재 MFA 환경 Python {py_ver or '(unknown)'} 은/는 "
                    f"Windows MFA 의존성과 호환되지 않아 Python {MFA_PORTABLE_PYTHON_VERSION} 기준으로 다시 구성합니다."
                )
                if not _remove_env_dir():
                    return False
                self.mfa_path = ""
            else:
                self._append_log("✅ MFA 실행 환경이 이미 있습니다.")
                if py_ver:
                    self._append_log(f"ℹ MFA Python 버전: {py_ver}")
                self.mfa_path = mfa_exe

        if not os.path.exists(mfa_exe):
            self._append_log("🔍 경량 Micromamba 기반으로 MFA 환경을 준비합니다.")
            if not _ensure_vc_runtime():
                self._append_log("⚠ VC++ 런타임 자동 복구가 완료되지 않았습니다. 설치가 실패하면 수동 설치 후 다시 시도하세요.")
            _remove_legacy_conda_root()
            if not _download_micromamba():
                return False
            if not _extract_micromamba():
                return False
            if os.path.isdir(env_dir) and not os.path.exists(mfa_exe):
                if not _remove_env_dir():
                    return False
            self._set_status("🔧 MFA 환경 설치 중...")
            if not _run_micromamba(
                [
                    'create',
                    '-y',
                    '-r',
                    micromamba_root,
                    '-p',
                    env_dir,
                    '-c',
                    'conda-forge',
                    f'python={MFA_PORTABLE_PYTHON_VERSION}',
                    'montreal-forced-aligner',
                    'colorama',
                ],
                "[3/3] 🔧 MFA 설치 중... (3~10분)",
            ):
                self._append_log("❌ MFA 설치 실패")
                return False
            self._append_log("✅ Micromamba 기반 MFA 환경 생성 완료 (후속 점검 진행)")
            if not _refresh_mfa_path():
                self._append_log("❌ MFA 실행 파일을 찾지 못했습니다.")
                return False

        def _create_portable_mfa_env(step_label: str) -> bool:
            return _run_micromamba(
                [
                    'create',
                    '-y',
                    '-r',
                    micromamba_root,
                    '-p',
                    env_dir,
                    '-c',
                    'conda-forge',
                    f'python={MFA_PORTABLE_PYTHON_VERSION}',
                    'montreal-forced-aligner',
                    'colorama',
                ],
                step_label,
            )

        self._set_status("🔧 MFA Python 도구 점검 중...")
        packaging_ok = ensure_mfa_python_packaging_stack(self.mfa_path or mfa_exe, callback=self._append_log)
        if not packaging_ok:
            self._append_log("⚠ MFA Python 패키지 도구 복구 실패. 환경을 재구성해 1회 재시도합니다.")
            if not _remove_env_dir():
                self._append_log("❌ MFA 환경 재구성 전 기존 env 삭제 실패")
                return False
            self.mfa_path = ""
            if not _download_micromamba():
                return False
            if not _extract_micromamba():
                return False
            self._set_status("🔧 MFA 환경 재구성 중...")
            if not _create_portable_mfa_env("[재시도] 🔧 MFA 환경 재구성 중... (3~10분)"):
                self._append_log("❌ MFA 환경 재구성 실패")
                return False
            if not _refresh_mfa_path():
                self._append_log("❌ 재구성 후 MFA 실행 파일을 찾지 못했습니다.")
                return False
            self._set_status("🔧 MFA Python 도구 재점검 중...")
            packaging_ok = ensure_mfa_python_packaging_stack(self.mfa_path or mfa_exe, callback=self._append_log)

        if not packaging_ok:
            self._append_log("❌ MFA Python 패키지 도구 준비 실패")
            return False

        if lang == "korean":
            self._set_status("🔧 한국어 MFA 의존성 점검 중...")
            if not ensure_korean_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 한국어 MFA 의존성 준비 실패")
                return False
            dep_report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if "korean_deps_degraded" in list(dep_report.get("issues", []) or []):
                self._notify_korean_dependency_degraded()
        elif lang == "japanese":
            self._set_status("🔧 일본어 MFA 의존성 점검 중...")
            if not ensure_japanese_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 일본어 MFA 의존성 준비 실패")
                return False

        self._set_status("⬇ MFA 모델 다운로드 중...")
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("[추가 단계] ⬇ 현재 언어용 MFA 음향 모델 다운로드 중...")
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                return False

        if os.path.exists(micromamba_archive):
            try:
                os.remove(micromamba_archive)
            except OSError:
                pass

        self._update_mfa_status(True)
        self._append_log("")
        self._append_log("✅ MFA 설치가 완료되었습니다!")
        self._append_log("   이제 정렬을 바로 계속 진행합니다.")
        return True

    def _format_mfa_diagnosis_summary(self, report):
        checks = dict((report or {}).get("checks", {}) or {})
        issues = list((report or {}).get("issues", []) or [])
        lang = "일본어" if str((report or {}).get("language", "")).lower() == "japanese" else "한국어"
        lang_support_ok = checks.get("language_support")
        lang_support_mode = str(checks.get("language_support_mode", "") or "").strip().lower()
        if lang_support_ok is None:
            lang_support_text = "미확인"
        elif lang_support_ok and lang_support_mode == "fallback":
            lang_support_text = "OK (degraded fallback)"
        else:
            lang_support_text = "OK" if lang_support_ok else "복구 필요"
        return "\n".join([
            f"언어: {lang}",
            f"MFA 실행 파일: {'OK' if checks.get('mfa_executable') else '없음'}",
            f"Python 실행 파일: {'OK' if checks.get('python_exe') else '없음'}",
            f"pip 실행 파일: {'OK' if checks.get('pip_exe') else '없음'}",
            f"Python 버전: {checks.get('python_version') or '(확인 불가)'}",
            f"재구성 필요: {'예' if checks.get('python_rebuild_required') else '아니오'}",
            f"패키지 도구(pip/setuptools/wheel): {('OK' if checks.get('packaging_stack') else '복구 필요') if checks.get('packaging_stack') is not None else '미확인'}",
            f"{lang} 의존성: {lang_support_text}",
            f"{lang} MFA 모델: {'OK' if checks.get('model_ready') else '다운로드 필요'}",
            f"이슈 코드: {', '.join(issues) if issues else '(없음)'}",
            f"최종 준비 상태: {'준비 완료' if (report or {}).get('ready') else '추가 복구 필요'}",
        ])

    def _confirm_mfa_install_action(self, language="korean", reason="install_required") -> bool:
        lang = str(language or "korean").strip().lower()
        lang_label = "일본어" if lang == "japanese" else "한국어"
        reason_map = {
            "missing_runtime": "MFA 실행 환경이 없습니다.",
            "python_rebuild": "MFA Python 버전 재구성이 필요합니다.",
            "model_download": f"{lang_label} MFA 모델 다운로드가 필요합니다.",
            "startup_auto_repair": "초기 자동 점검에서 복구 필요 상태가 감지되었습니다.",
            "install_required": "MFA 설치/복구가 필요합니다.",
        }
        reason_text = reason_map.get(str(reason or "").strip().lower(), reason_map["install_required"])
        title = "MFA 설치/복구 확인"
        message = (
            f"{reason_text}\n\n"
            "진단은 완료되었고, 지금부터 실제 설치/복구를 진행하려고 합니다.\n"
            "설치를 시작할까요?"
        )
        self._append_log(f"ℹ MFA 진단 결과: {reason_text}")
        self._append_log("ℹ 설치 전 사용자 확인 팝업을 표시합니다.")
        approved = False
        if hasattr(self, "_ask_yes_no_dialog_sync"):
            approved = bool(
                self._ask_yes_no_dialog_sync(
                    title=title,
                    message=message,
                    default=False,
                )
            )
        if approved:
            self._append_log("ℹ 사용자 확인: MFA 설치/복구를 진행합니다.")
        else:
            self._append_log("ℹ 사용자 선택: MFA 설치/복구를 진행하지 않고 진단만 유지합니다.")
        return approved

    def _repair_existing_mfa_runtime(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        resolved = self.mfa_path or find_mfa_executable() or ""
        if not resolved or not os.path.exists(resolved):
            self._append_log("❌ 복구할 MFA 환경을 찾지 못했습니다.")
            return False
        self.mfa_path = resolved

        self._set_status("🔧 MFA Python 도구 점검 중...")
        if not ensure_mfa_python_packaging_stack(self.mfa_path, callback=self._append_log):
            self._append_log("❌ MFA Python 패키지 도구 복구 실패")
            return False

        if lang == "korean":
            self._set_status("🔧 한국어 MFA 의존성 점검 중...")
            if not ensure_korean_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 한국어 MFA 의존성 준비 실패")
                return False
            dep_report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if "korean_deps_degraded" in list(dep_report.get("issues", []) or []):
                self._notify_korean_dependency_degraded()
        elif lang == "japanese":
            self._set_status("🔧 일본어 MFA 의존성 점검 중...")
            if not ensure_japanese_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 일본어 MFA 의존성 준비 실패")
                return False

        self._set_status("🔧 MFA 모델 점검 중...")
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("ℹ 현재 언어용 MFA 모델이 없어 자동 다운로드를 시작합니다.")
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                return False
        return True

    def _ensure_mfa_ready_for_language(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        self._last_mfa_install_declined = False
        resolved = self.mfa_path if (self.mfa_path and os.path.exists(self.mfa_path)) else (find_mfa_executable() or "")
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved
        if not self.mfa_path or not os.path.exists(self.mfa_path):
            self._append_log("ℹ MFA 실행 환경이 없어 설치/복구 확인이 필요합니다.")
            if not self._confirm_mfa_install_action(language=lang, reason="missing_runtime"):
                self._mfa_ready_cache_ok = False
                self._last_mfa_install_declined = True
                return False
            self._notify_long_install_time("MFA")
            install_ok = False
            self._set_mfa_install_progress_state(True)
            try:
                install_ok = self._install_mfa_runtime(language=lang)
            finally:
                self._set_mfa_install_progress_state(False)
            if install_ok:
                cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
                self._mfa_ready_cache_key = cache_key
                self._mfa_ready_cache_ok = True
                return True
            self._mfa_ready_cache_ok = False
            return self._run_setup_mfa_script_fallback(language=lang, reason="auto_install_failed")
        soft_rebuild_gate = str(os.environ.get("UTOA_MFA_SOFT_REBUILD", "1") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if mfa_env_requires_python_downgrade(self.mfa_path):
            py_ver = get_mfa_env_python_version(self.mfa_path)
            if soft_rebuild_gate:
                self._append_log(
                    f"⚠ MFA 환경 Python {py_ver or '(unknown)'} 감지: 재구성 권장 상태지만 soft 정책으로 현재 환경으로 먼저 실행을 시도합니다."
                )
            else:
                self._append_log(
                    f"⚠ 현재 MFA 환경 Python {py_ver or '(unknown)'} 은/는 "
                    f"Windows MFA 의존성과 호환되지 않아 Python {MFA_PORTABLE_PYTHON_VERSION} 기준으로 다시 구성합니다."
                )
                self.mfa_path = ""
                self._mfa_ready_cache_ok = False
                if not self._confirm_mfa_install_action(language=lang, reason="python_rebuild"):
                    self._last_mfa_install_declined = True
                    return False
                install_ok = False
                self._set_mfa_install_progress_state(True)
                try:
                    install_ok = self._install_mfa_runtime(language=lang)
                finally:
                    self._set_mfa_install_progress_state(False)
                if install_ok:
                    cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
                    self._mfa_ready_cache_key = cache_key
                    self._mfa_ready_cache_ok = True
                    return True
                return self._run_setup_mfa_script_fallback(language=lang, reason="python_rebuild_install_failed")

        # Runtime path exists: keep per-run check lightweight.
        cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
        if (
            bool(getattr(self, "_mfa_ready_cache_ok", False))
            and str(getattr(self, "_mfa_ready_cache_key", "") or "") == cache_key
        ):
            return True
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("ℹ 현재 언어용 MFA 모델이 없어 자동 다운로드를 시작합니다.")
            if not self._confirm_mfa_install_action(language=lang, reason="model_download"):
                self._mfa_ready_cache_ok = False
                self._last_mfa_install_declined = True
                return False
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                self._mfa_ready_cache_ok = False
                return False
        self._mfa_ready_cache_key = cache_key
        self._mfa_ready_cache_ok = True
        return True

    def _prompt_mfa_install_for_explicit_selection(self):
        if bool(getattr(self, "_mfa_selection_install_prompt_seen", False)):
            return
        self._mfa_selection_install_prompt_seen = True
        lang = self._get_language() if hasattr(self, "_get_language") else "korean"
        resolved = self.mfa_path if (self.mfa_path and os.path.exists(self.mfa_path)) else (find_mfa_executable() or "")
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved
        try:
            report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
        except Exception:
            report = {"ready": False}
        if bool(report.get("ready", False)):
            if hasattr(self, "_update_mfa_status"):
                self._update_mfa_status(True)
            return
        reason = "install_required"
        checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
        if not checks.get("mfa_executable"):
            reason = "missing_runtime"
        elif checks.get("python_rebuild_required"):
            reason = "python_rebuild"
        elif not checks.get("model_ready", True):
            reason = "model_download"
        if not self._confirm_mfa_install_action(language=lang, reason=reason):
            return
        self._run_mfa_diagnose_repair()

    def _mfa_startup_repair_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".mfa_startup_repair_state.json")

    def _load_mfa_startup_repair_state(self):
        path = self._mfa_startup_repair_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_mfa_startup_repair_state(self, **updates):
        state = self._load_mfa_startup_repair_state()
        state.update(updates or {})
        path = self._mfa_startup_repair_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _check_startup_ml_runtime_ready(self, mfa_report=None):
        report = mfa_report if isinstance(mfa_report, dict) else {}
        env_dir = str(report.get("env_dir", "") or "").strip()
        python_exe = os.path.join(env_dir, "python.exe") if env_dir else ""
        result = {
            "ready": False,
            "python_exe": python_exe,
            "missing_modules": [],
            "detail": "",
        }
        if not python_exe or not os.path.isfile(python_exe):
            result["detail"] = "MFA Python 실행 파일을 찾지 못했습니다."
            return result

        probe_code = (
            "import json\n"
            "mods=['pandas','lightgbm','onnxruntime']\n"
            "missing=[]\n"
            "for m in mods:\n"
            "    try:\n"
            "        __import__(m)\n"
            "    except Exception as e:\n"
            "        missing.append({'module':m,'error':f'{type(e).__name__}: {e}'})\n"
            "print(json.dumps({'missing':missing}, ensure_ascii=False))\n"
            "raise SystemExit(1 if missing else 0)\n"
        )
        probe_env = os.environ.copy()
        # Prevent host/runtime Python vars from leaking into the MFA env probe.
        for leaked_key in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONEXECUTABLE",
            "__PYVENV_LAUNCHER__",
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
        ):
            probe_env.pop(leaked_key, None)
        probe_env["PYTHONNOUSERSITE"] = "1"
        probe_env["PYTHONUTF8"] = "1"
        probe_env["PYTHONIOENCODING"] = "utf-8"
        path_parts = [
            env_dir,
            os.path.join(env_dir, "Scripts"),
            os.path.join(env_dir, "Library", "bin"),
            os.path.join(env_dir, "Library", "usr", "bin"),
            os.path.join(env_dir, "Library", "mingw-w64", "bin"),
        ]
        existing_path = str(probe_env.get("PATH", "") or "")
        probe_env["PATH"] = os.pathsep.join(path_parts + ([existing_path] if existing_path else []))
        try:
            run = self._run_subprocess_hidden(
                [python_exe, "-I", "-c", probe_code],
                capture_output=True,
                text=False,
                timeout=180,
                env=probe_env,
                cwd=env_dir or None,
            )
        except Exception as exc:
            result["detail"] = f"ML 런타임 점검 실행 실패: {exc}"
            return result

        stdout_text = self._decode_subprocess_output(getattr(run, "stdout", b"") or b"").strip()
        stderr_text = self._decode_subprocess_output(getattr(run, "stderr", b"") or b"").strip()
        payload = {}
        for raw in (stdout_text, stderr_text):
            if not raw:
                continue
            for line in reversed(raw.splitlines()):
                line = str(line or "").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    payload = parsed
                    break
            if payload:
                break

        missing_items = list(payload.get("missing", []) or [])
        missing_modules = []
        detail_tokens = []
        for item in missing_items:
            if not isinstance(item, dict):
                continue
            mod = str(item.get("module", "") or "").strip()
            err = str(item.get("error", "") or "").strip()
            if mod:
                missing_modules.append(mod)
                if err:
                    detail_tokens.append(f"{mod}: {err}")
        result["missing_modules"] = missing_modules
        result["ready"] = bool(run.returncode == 0 and not missing_modules)
        if result["ready"]:
            return result
        if detail_tokens:
            result["detail"] = " | ".join(detail_tokens)
        else:
            tail = (stderr_text or stdout_text or "").strip()
            result["detail"] = tail[-400:] if tail else "ML 런타임 모듈 import 확인에 실패했습니다."
        return result

    def _run_setup_mfa_install_with_ml(self):
        script_path = self._resolve_setup_mfa_script_path()
        if not script_path:
            self._append_log("⚠ setup_mfa.bat을 찾지 못해 MFA+ML 자동 설치를 진행할 수 없습니다.")
            return False

        cmd = [script_path, "--non-interactive", "--install", "--with-ml"]
        env = os.environ.copy()
        for leaked_key in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONEXECUTABLE",
            "__PYVENV_LAUNCHER__",
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
        ):
            env.pop(leaked_key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            runtime_root = os.path.dirname(get_default_mfa_env_dir())
        except Exception:
            runtime_root = ""
        if runtime_root:
            env["UTOA_MFA_SHARED_ROOT"] = runtime_root
            cmd.extend(["--runtime-root", runtime_root])

        self._append_log("ℹ startup 점검: setup_mfa.bat로 MFA+ML 설치/복구를 진행합니다.")
        self._append_log(f"   실행: {' '.join(cmd)}")
        try:
            process = self._popen_subprocess_hidden(
                cmd,
                cwd=os.path.dirname(script_path) or None,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=False,
                env=env,
            )
        except Exception as exc:
            self._append_log(f"❌ setup_mfa.bat 실행 실패: {exc}")
            return False

        if process.stdout is not None:
            for line in self._iter_decoded_stdout_lines(process):
                self._append_log(line)
        process.wait()
        if process.returncode != 0:
            self._append_log(f"❌ setup_mfa.bat MFA+ML 설치 실패 (code={process.returncode})")
            return False

        resolved = find_mfa_executable() or ""
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved
        return True

    def _schedule_startup_mfa_auto_repair(self):
        if str(os.environ.get("UTOA_ENABLE_STARTUP_MFA_AUTO_REPAIR", "")).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        if str(os.environ.get("UTOA_DISABLE_STARTUP_MFA_AUTO_REPAIR", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_mfa_auto_repair_scheduled", False):
            return
        try:
            if normalize_aligner_name(self.aligner_var.get(), default="hsmm_oto") != "mfa":
                return
        except Exception:
            return

        state = self._load_mfa_startup_repair_state()
        # Run startup MFA+ML check only once per install/runtime.
        if bool(state.get("first_check_done", False)):
            return
        if float(state.get("last_attempt_ts", 0.0) or 0.0) > 0:
            return

        def _kickoff():
            if self.is_running:
                return
            self._run_in_thread(self._run_startup_mfa_auto_repair_task)

        self._startup_mfa_auto_repair_scheduled = True
        self._after_safe(_kickoff, delay_ms=1200)

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

    def _run_startup_mfa_auto_repair_task(self):
        self._set_running(True)
        lang = self._get_language()
        now = time.time()
        self._save_mfa_startup_repair_state(
            last_attempt_ts=now,
            last_result="running",
            last_language=lang,
        )
        try:
            self._set_status("⏳ 조금만 기다려주세요, 프로그램 필수 요소들을 설치하고 점검하는 중입니다.")
            self._append_log("🔍 초기 설치 점검: MFA+ML 상태를 자동 진단합니다.")
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved

            before = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            ml_before = self._check_startup_ml_runtime_ready(before)

            if before.get("ready") and ml_before.get("ready"):
                if "korean_deps_degraded" in list(before.get("issues", []) or []):
                    self._notify_korean_dependency_degraded()
                self._append_log("✅ 초기 MFA+ML 상태 점검 완료 (복구 불필요)")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA+ML 점검 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    last_ml_missing_modules=[],
                    first_check_done=True,
                )
                return

            self._append_log("⚠ 초기 설치 점검에서 MFA 또는 ML 런타임 이상을 감지했습니다.")
            self._append_log("ℹ 자동복구는 진단 후 사용자 확인을 받은 경우에만 설치를 진행합니다.")
            self._append_log(self._format_mfa_diagnosis_summary(before))
            if not ml_before.get("ready"):
                missing_ml = ", ".join(list(ml_before.get("missing_modules", []) or [])) or "pandas/lightgbm/onnxruntime"
                self._append_log(f"ℹ ML 런타임 점검 결과: 복구 필요 ({missing_ml})")
                detail = str(ml_before.get("detail", "") or "").strip()
                if detail:
                    self._append_log(f"   상세: {detail}")

            missing_targets = []
            if not before.get("ready"):
                missing_targets.append("MFA 런타임")
            if not ml_before.get("ready"):
                missing_targets.append("ML 런타임")
            missing_label = ", ".join(missing_targets) or "MFA/ML 런타임"
            confirm_title = "초기 설치 자동 복구 확인"
            confirm_message = (
                f"초기 점검 결과 {missing_label} 복구가 필요합니다.\n\n"
                "지금 setup_mfa.bat을 실행해 자동 설치/복구를 진행할까요?\n"
                "(MFA + ML 패키지 설치 포함)"
            )
            approved = False
            if hasattr(self, "_ask_yes_no_dialog_sync"):
                approved = bool(
                    self._ask_yes_no_dialog_sync(
                        title=confirm_title,
                        message=confirm_message,
                        default=False,
                    )
                )
            if not approved:
                self._append_log("ℹ 사용자 선택: startup MFA+ML 자동 설치를 건너뛰었습니다.")
                self._set_status("⚠ MFA/ML 추가 복구 필요")
                issues = list(before.get("issues", []) or [])
                self._save_mfa_startup_repair_state(
                    last_result="user_declined",
                    last_issues=issues,
                    last_ml_missing_modules=list(ml_before.get("missing_modules", []) or []),
                    last_failure_ts=time.time(),
                    first_check_done=True,
                )
                return

            self._append_log("ℹ 사용자 확인: startup MFA+ML 자동 설치/복구를 진행합니다.")
            self._notify_long_install_time("MFA+ML")
            install_ok = False
            self._set_mfa_install_progress_state(True)
            try:
                install_ok = self._run_setup_mfa_install_with_ml()
            finally:
                self._set_mfa_install_progress_state(False)

            after = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            ml_after = self._check_startup_ml_runtime_ready(after)
            final_ready = bool(after.get("ready")) and bool(ml_after.get("ready"))

            if install_ok and final_ready:
                self._append_log("✅ 초기 MFA+ML 자동 복구 완료")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA+ML 복구 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    last_ml_missing_modules=[],
                    first_check_done=True,
                )
            else:
                issues = list(after.get("issues", []) or [])
                self._append_log("⚠ 초기 MFA+ML 자동 복구가 완료되지 않았습니다. 'MFA 진단/복구' 버튼으로 재시도하세요.")
                detail = str(ml_after.get("detail", "") or "").strip()
                if detail and not ml_after.get("ready"):
                    self._append_log(f"⚠ ML 런타임 점검 상세: {detail}")
                recovery_guide = self._build_setup_mfa_recovery_guide(language=lang)
                self._after_safe(
                    lambda guide=recovery_guide: self._show_copyable_alert(
                        title="MFA/ML 자동 복구 추가 안내",
                        message=(
                            "초기 자동 복구가 완료되지 않았습니다.\n"
                            "앱 내부의 'MFA 진단/복구'를 다시 시도하거나 아래 방법으로 수동 복구를 진행해 주세요.\n\n"
                            f"{guide}\n"
                            "추가로 ML 런타임이 누락된 경우 setup_mfa.bat --with-ml 옵션으로 재실행해 주세요."
                        ),
                        alert_key="mfa_ml_startup_repair_needs_attention",
                    )
                )
                self._set_status("⚠ MFA/ML 추가 복구 필요")
                self._save_mfa_startup_repair_state(
                    last_result="failed",
                    last_issues=issues,
                    last_ml_missing_modules=list(ml_after.get("missing_modules", []) or []),
                    last_failure_ts=time.time(),
                    first_check_done=True,
                )
        except Exception as e:
            self._save_mfa_startup_repair_state(
                last_result="error",
                last_error=str(e),
                last_error_ts=time.time(),
                first_check_done=True,
            )
            self._handle_error("초기 MFA/ML 자동 복구", e)
        finally:
            self._save_mfa_startup_repair_state(
                first_check_done=True,
                last_checked_ts=time.time(),
            )
            self._set_running(False)

    def _schedule_alignment_failure_mfa_followup(self, language="korean", align_code="", align_message=""):
        if getattr(self, "_mfa_alignment_failure_followup_running", False):
            return
        self._mfa_alignment_failure_followup_running = True
        thread = threading.Thread(
            target=self._run_alignment_failure_mfa_followup_task,
            args=(language, align_code, align_message),
            daemon=True,
        )
        thread.start()

    def _run_alignment_failure_mfa_followup_task(self, language="korean", align_code="", align_message=""):
        lang = str(language or "korean").strip().lower()
        try:
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved
            report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            summary = self._format_mfa_diagnosis_summary(report)
            issues = list(report.get("issues", []) or [])
            self._append_log("🔎 정렬 실패 후 MFA 자동 점검 결과")
            self._append_log(summary)
            if "korean_deps_degraded" in issues:
                self._notify_korean_dependency_degraded()

            ready = bool(report.get("ready", False))
            issue_token = "_".join(sorted(str(i) for i in issues))[:80] or "none"
            if ready:
                message = (
                    "정렬 실패 후 MFA 자동 점검 결과, 런타임 자체는 준비 상태입니다.\n"
                    "이번 실패는 입력 데이터(발음/사전/오디오) 또는 현재 정렬 프로필 영향일 가능성이 큽니다.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "권장:\n"
                    "1) 사전/라벨 파일 재생성\n"
                    "2) MFA 프로필을 '기본' 또는 '빠름'으로 변경 후 재시도\n"
                    "3) 계속 실패하면 'MFA 진단/복구' 버튼 실행\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}"
                )
                alert_key = f"mfa_after_align_fail_ready_{lang}_{issue_token}"
            else:
                recovery_guide = self._build_setup_mfa_recovery_guide(language=lang)
                message = (
                    "정렬 실패 후 MFA 자동 점검에서 추가 복구가 필요하다고 판단되었습니다.\n"
                    "앱의 'MFA 진단/복구' 버튼을 눌러 복구를 진행해 주세요.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}\n\n"
                    "[수동 복구]\n"
                    f"{recovery_guide}"
                )
                alert_key = f"mfa_after_align_fail_repair_{lang}_{issue_token}"

            self._after_safe(
                lambda msg=message, key=alert_key: self._show_copyable_alert(
                    title="MFA 정렬 실패 후 점검 안내",
                    message=msg,
                    alert_key=key,
                )
            )
        except Exception as exc:
            self._append_log(f"⚠ 정렬 실패 후 MFA 자동 점검 중 오류: {exc}")
        finally:
            self._mfa_alignment_failure_followup_running = False

    def _run_mfa_diagnose_repair(self):
        self._notify_long_install_time("MFA")

        def task():
            self._set_running(True)
            lang = self._get_language()
            lang_label = "일본어" if lang == "japanese" else "한국어"
            self._set_status("🔍 MFA 진단/복구 중... (시간이 걸릴 수 있습니다)")
            try:
                resolved = self.mfa_path or find_mfa_executable() or ""
                if resolved and os.path.exists(resolved):
                    self.mfa_path = resolved
                before = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
                self._append_log("🔎 MFA 진단 시작")
                self._append_log(self._format_mfa_diagnosis_summary(before))

                ok = False
                if not before.get("checks", {}).get("mfa_executable") or before.get("checks", {}).get("python_rebuild_required"):
                    self._append_log("ℹ MFA 환경이 없거나 재구성이 필요해 새로 설치합니다.")
                    self._set_mfa_install_progress_state(True)
                    try:
                        ok = self._install_mfa_runtime(language=lang)
                    finally:
                        self._set_mfa_install_progress_state(False)
                else:
                    ok = self._repair_existing_mfa_runtime(language=lang)

                after = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
                needs_attention = not bool(after.get("ready", False))
                recovery_guide = self._build_setup_mfa_recovery_guide(language=lang) if needs_attention else ""
                summary = (
                    "[진단 전]\n"
                    f"{self._format_mfa_diagnosis_summary(before)}\n\n"
                    "[진단 후]\n"
                    f"{self._format_mfa_diagnosis_summary(after)}"
                )
                if needs_attention:
                    summary = (
                        f"{summary}\n\n"
                        "[추가 수동 복구]\n"
                        f"{recovery_guide}"
                    )
                self._after_safe(lambda s=summary, ready=after.get("ready", False), ll=lang_label: self._show_copyable_alert(
                    title=f"MFA 진단/복구 결과 ({ll})",
                    message=s,
                    alert_key=f"mfa_diag_repair_{lang}_{'ok' if ready else 'needs_attention'}",
                ))
                if ok and after.get("ready"):
                    self._update_mfa_status(True)
                    self._set_status("✅ MFA 진단/복구 완료")
                else:
                    self._set_status("❌ MFA 진단/복구 미완료")
            except Exception as e:
                self._handle_error("MFA 진단/복구", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_mfa_setup(self):
        """GUI 안에서 MFA 포터블 환경을 자동 설치합니다."""
        self._notify_long_install_time("MFA")
        def task():
            self._set_running(True)
            self._set_status("⏳ 조금만 기다려주세요, 프로그램 필수 요소들을 설치하고 점검하는 중입니다.")
            try:
                self._set_mfa_install_progress_state(True)
                try:
                    install_ok = self._install_mfa_runtime(language=self._get_language())
                finally:
                    self._set_mfa_install_progress_state(False)
                if install_ok:
                    self._set_status("✅ MFA 설치 완료")
                else:
                    self._set_status("⚪ MFA 설치 미완료")

            except Exception as e:
                self._handle_error("MFA 설치", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _set_mfa_install_progress_state(self, installing):
        self._mfa_install_in_progress = bool(installing)
        if not installing:
            installed_now = False
            try:
                resolved = self.mfa_path or find_mfa_executable() or ""
                if resolved and os.path.exists(resolved):
                    self.mfa_path = resolved
                report = diagnose_mfa_runtime(self.mfa_path or "", language=self._get_language())
                installed_now = bool(report.get("ready", False))
            except Exception:
                installed_now = bool(self.mfa_path and os.path.exists(str(self.mfa_path)))
            self._update_mfa_status(installed_now)
            return

        def _do():
            status_label = getattr(self, "mfa_status_label", None)
            install_btn = getattr(self, "mfa_install_btn", None)
            if status_label is None or install_btn is None:
                return
            status_label.configure(text="🔧 MFA 설치 중...", text_color="#C27803")
            install_btn.configure(text="🔧 설치 중...", state="disabled", fg_color="#B0BEC5")

        self._after_safe(_do)

    def _update_mfa_status(self, installed):
        """MFA 설치 상태를 UI에 반영합니다."""
        self._mfa_install_in_progress = False
        self._mfa_ui_ready = bool(installed)
        def _do():
            status_label = getattr(self, "mfa_status_label", None)
            install_btn = getattr(self, "mfa_install_btn", None)
            if status_label is None or install_btn is None:
                return
            if installed:
                status_label.configure(text="✅ MFA 설치됨", text_color="#4F8F61")
                install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                status_label.configure(text="⚪ MFA 미완료", text_color="#90A4AE")
                install_btn.configure(text="⬇ MFA 원클릭 설치", state="normal", fg_color="#FFA726")
        self._after_safe(_do)

    # --------------------------------------------------------------------------

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
        self._append_log(
            f"[OTO-ML] auto policy: ml={'ON' if enable_ml_correction else 'OFF'}, "
            f"route={auto_policy.get('route')}, coupled={'ON' if auto_policy.get('has_coupled') else 'OFF'}, "
            f"hybrid={'ON' if auto_policy.get('hybrid_routing') else 'OFF'}"
        )

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

                    primary_engine = normalize_aligner_name(aligner_engine, default="mfa")
                    if primary_engine == "mfa":
                        if not self._ensure_mfa_ready_for_language(lang):
                            errors = ["MFA 설치/모델 준비 실패"]
                        elif not self._validate_alignment_input_files(target_wav_dir, dict_path):
                            errors = ["정렬 입력 파일 점검 실패"]
                    if not errors:
                        self._set_status(f"{prefix}정렬 실행 중...")
                        align_result = run_alignment_with_fallback(
                            language=lang,
                            wav_folder=target_wav_dir,
                            dictionary_path=dict_path,
                            output_folder=textgrid_dir,
                            primary_aligner=primary_engine,
                            fallback_aligner="",
                            mfa_path=self.mfa_path or "",
                            mfa_align_profile=(
                                self._get_mfa_align_profile_code()
                                if hasattr(self, "_get_mfa_align_profile_code")
                                else "accurate"
                            ),
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
                else:
                    self._set_status("3/5 - MFA 정렬 준비 중...")
                    if not self._ensure_mfa_ready_for_language(lang):
                        self._append_log("❌ MFA 설치/모델 준비 실패")
                        self._set_status("❌ MFA 설치/모델 준비 실패")
                        return
                    if not self._validate_alignment_input_files(wav_dir, dict_path):
                        return
                mfa_profile = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "accurate"
                )
                if primary_engine == "mfa":
                    self._append_log(f"ℹ MFA 정렬 프로필: {mfa_profile}")
                elif primary_engine == "sequence":
                    self._append_log("ℹ 정렬 엔진: 전용 시퀀스 baseline")
                else:
                    self._append_log("ℹ 정렬 엔진: none (MFA 비사용)")
                if hasattr(self, "_apply_advanced_tuning_envs"):
                    self._apply_advanced_tuning_envs()

                align_result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary_engine,
                    fallback_aligner=fallback_engine,
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=(
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    ),
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
                    if self._is_mfa_module_missing_error(align_code, align_err):
                        self._append_log("❌ MFA 모듈 누락이 감지되어 정렬이 실패했습니다. 'MFA 진단/복구'를 실행해 주세요.")
                        self._notify_mfa_module_missing(language=lang, detail=align_err)
                    if self._is_lab_or_dict_missing_alignment_error(align_code, align_err):
                        self._notify_lab_or_dict_missing(wav_dir, dict_path)
                    if primary_engine == "mfa":
                        self._schedule_alignment_failure_mfa_followup(
                            language=lang,
                            align_code=align_code,
                            align_message=align_err,
                        )
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
