import datetime
import glob
import json
import logging
import os
import re
import shutil
import sys
import threading
import traceback
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.format_type_utils import normalize_auto_format_value
from core.log_events import classify_log_message, log_with_event
from core.mfa_runner import ALERT_MFA_PERMISSION_DENIED, ALERT_MSVC_REQUIRED
from core.oto_validator import validate_oto_timing


class FileDialogMixin:
    def _browse_folder(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_file(self, entry, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browser_save(self, entry, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".ini")
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_file_by_var(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _ml_model_repo_root(self):
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        installed_root = os.path.join(base_dir, "models_installed", "oto_ml")
        if os.path.isdir(installed_root):
            return installed_root
        return os.path.join(base_dir, "ML_models")

    def _ml_model_language_root(self, language):
        return os.path.join(self._ml_model_repo_root(), str(language or "").strip().lower())

    def _ml_model_language_for_var(self, var):
        if hasattr(self, "ml_model_root_ja_var") and var is self.ml_model_root_ja_var:
            return "japanese"
        return "korean"

    def _preferred_ml_model_browse_dir(self, var):
        language = self._ml_model_language_for_var(var)
        current = str(var.get() or "").strip()
        if os.path.isdir(current):
            return current

        format_code = ""
        if hasattr(self, "_get_language") and hasattr(self, "auto_format_var") and self._get_language() == language:
            try:
                format_code = normalize_auto_format_value(language, self.auto_format_var.get()) or ""
            except Exception:
                format_code = ""

        lang_root = self._ml_model_language_root(language)
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy_root = os.path.join(base_dir, "ML_models", language)
        fmt_root = os.path.join(lang_root, format_code) if format_code else ""
        candidates = []
        if fmt_root:
            candidates.extend(
                [
                    os.path.join(fmt_root, "v1_ensemble"),
                    fmt_root,
                ]
            )
        candidates.extend([lang_root, legacy_root, self._ml_model_repo_root()])
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return current or lang_root or self._ml_model_repo_root()

    def _recommended_ml_model_root(self, language):
        lang_root = self._ml_model_language_root(language)
        if os.path.isdir(lang_root):
            return lang_root
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy_root = os.path.join(base_dir, "ML_models", language)
        if os.path.isdir(legacy_root):
            return legacy_root
        repo_root = self._ml_model_repo_root()
        if os.path.isdir(repo_root):
            return repo_root
        return ""

    def _language_has_ensemble_bundle(self, language):
        lang_root = self._ml_model_language_root(language)
        if not os.path.isdir(lang_root):
            return False
        pattern = os.path.join(lang_root, "*", "v1_ensemble", "model_meta.json")
        return any(os.path.isfile(path) for path in glob.glob(pattern))

    def _apply_recommended_ml_model_defaults(self):
        language = self._get_language() if hasattr(self, "_get_language") else "korean"
        if language == "japanese" and hasattr(self, "ml_model_root_ja_var"):
            target_var = self.ml_model_root_ja_var
        elif hasattr(self, "ml_model_root_kr_var"):
            target_var = self.ml_model_root_kr_var
        else:
            target_var = None

        if target_var is not None:
            current_root = str(target_var.get() or "").strip()
            if not current_root or not os.path.isdir(current_root):
                recommended_root = self._recommended_ml_model_root(language)
                if recommended_root:
                    target_var.set(recommended_root)

        if hasattr(self, "ml_coupled_backend_var"):
            backend = str(self.ml_coupled_backend_var.get() or "auto").strip().lower()
            if backend in {"", "auto"} and self._language_has_ensemble_bundle(language):
                self.ml_coupled_backend_var.set("ensemble")

    def _browse_folder_by_var(self, var, initial_dir=""):
        path = filedialog.askdirectory(initialdir=initial_dir or self._preferred_ml_model_browse_dir(var))
        if path:
            var.set(path)

    def _browse_save_by_var(self, var, filetypes, defext):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=defext)
        if path:
            var.set(path)


class AppRuntimeMixin:
    _MSVC_BUILD_TOOLS_URL = "https://visualstudio.microsoft.com/visual-cpp-build-tools/"

    def _normalize_ui_message(self, msg: str) -> str:
        original = str(msg or "")
        text = original
        if not text:
            return text

        # 제어문자/깨진 영역을 최대한 정리해 UI 로그 가독성을 유지한다.
        text = re.sub(r"[\uF8F0-\uF8FF]", " ", text)
        text = re.sub(r"[\uFF61-\uFF9F]", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        if len(text) <= 2:
            if "MFA" in original:
                return "MFA 작업 진행 중..."
            if "WhisperX" in original or "whisperx" in original:
                return "WhisperX 작업 진행 중..."
            if "OTO" in original:
                return "OTO 작업 진행 중..."
            if "Lab" in original or "lab" in original:
                return "Lab 작업 진행 중..."
            if "dict" in original.lower() or "사전" in original:
                return "사전 작업 진행 중..."
        return text

    def _normalize_ml_selector_mode(self, value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"delta", "delta_only", "delta only", "델타만"}:
            return "delta"
        if mode in {"selector", "delta+selector", "delta + selector", "델타+셀렉터"}:
            return "selector"
        return "policy"

    def _describe_ml_selector_mode(self, value: str) -> str:
        mode = self._normalize_ml_selector_mode(value)
        if mode == "delta":
            return "델타만"
        if mode == "selector":
            return "델타+셀렉터"
        return "기본 정책"

    def _apply_ml_selector_runtime_mode(self, value: str) -> str:
        mode = self._normalize_ml_selector_mode(value)
        if mode == "delta":
            os.environ["UTOA_DISABLE_OTO_SELECTOR"] = "1"
            os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
        elif mode == "selector":
            os.environ["UTOA_FORCE_OTO_SELECTOR"] = "1"
            os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
        else:
            os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
            os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
        return mode

    def _after_safe(self, callback, delay_ms=0):
        if self._is_closing:
            return
        try:
            self.after(delay_ms, callback)
        except tk.TclError:
            pass

    def _force_exit_now(self):
        try:
            logging.shutdown()
        except Exception:
            pass
        os._exit(0)

    def _on_close_request(self):
        if self._is_closing:
            return
        self._is_closing = True
        try:
            self._save_config()
        except Exception:
            pass

        # 기본은 정상 종료를 우선합니다.
        # onefile(frozen)에서는 강제 종료가 _MEI 정리 경고를 유발할 수 있으므로 제한합니다.
        killer = None
        if self.is_running and not getattr(sys, "frozen", False):
            killer = threading.Timer(2.5, self._force_exit_now)
            killer.daemon = True
            killer.start()
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            if killer is not None:
                self._force_exit_now()

    def _log_to_file(self, msg, level=logging.INFO):
        try:
            logger = getattr(self, "logger", logging.getLogger(__name__))
            event_log_path = getattr(self, "event_log_path", "")
            log_with_event(logger, event_log_path, str(msg), level=level)
        except Exception:
            pass

    def _should_show_ui_log(self, msg):
        if msg in (ALERT_MSVC_REQUIRED, ALERT_MFA_PERMISSION_DENIED):
            return True
        return bool(classify_log_message(str(msg or "")).get("ui_visible"))

    def _append_log(self, msg, log_to_file=True):
        msg = self._normalize_ui_message(str(msg))
        now = time.monotonic()
        last_msg = getattr(self, "_last_log_msg", "")
        last_ts = float(getattr(self, "_last_log_ts", 0.0) or 0.0)
        # 여러 계층(callback + logger + task print)에서 동일 메시지가 짧은 시간 내
        # 중복 유입되는 경우 1회만 기록한다.
        if msg == last_msg and (now - last_ts) < 0.35:
            return
        self._last_log_msg = msg
        self._last_log_ts = now

        if log_to_file:
            self._log_to_file(msg)
        if msg == ALERT_MSVC_REQUIRED or self._looks_like_msvc_requirement_message(msg):
            self._after_safe(
                self._show_msvc_required_alert
            )
            return
        if msg == ALERT_MFA_PERMISSION_DENIED:
            self._after_safe(
                lambda: self._show_copyable_alert(
                    title="MFA 실행 권한 오류",
                    message=(
                        "MFA 내부 실행 파일(compute-mfcc-feats) 실행 권한이 없어 정렬이 실패할 수 있습니다. (WinError 5)\n\n"
                        "확인 항목:\n"
                        "- 백신/Defender 격리 또는 실행 차단 여부\n"
                        "- Controlled Folder Access/AppLocker 정책\n"
                        "- 프로젝트를 Desktop 외 경로로 이동 후 재설치"
                    ),
                    alert_key="mfa_permission_denied",
                )
            )
            return

        record = classify_log_message(str(msg or ""))
        if not record.get("ui_visible"):
            return

        def _do():
            self.log_text.insert("end", record["message"] + "\n")
            self.log_text.see("end")

        self._after_safe(_do)

    def _looks_like_msvc_requirement_message(self, msg):
        text = str(msg or "")
        lowered = text.lower()
        return (
            ("microsoft visual c++ 14.0" in lowered or "c++ build tools" in lowered)
            and ("필요" in text or "required" in lowered)
        )

    def _show_msvc_required_alert(self):
        self._show_copyable_alert(
            title="MFA 의존성 설치 안내",
            message=(
                "MFA 한국어 의존성 설치 중 Microsoft Visual C++ 14.0+가 필요합니다.\n\n"
                "설치 링크:\n"
                f"{self._MSVC_BUILD_TOOLS_URL}\n\n"
                "권장:\n"
                "1) C++ Build Tools 설치\n"
                "2) 터미널 재시작\n"
                "3) MFA 설치/정렬 재시도"
            ),
            alert_key="msvc_required",
            link_url=self._MSVC_BUILD_TOOLS_URL,
            link_label="설치 링크 열기",
        )

    def _show_copyable_alert(self, title, message, alert_key=None, link_url="", link_label="링크 열기"):
        if self._is_closing:
            return
        if alert_key and alert_key in self._shown_alert_keys:
            return
        if alert_key:
            self._shown_alert_keys.add(alert_key)

        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("700x330")
        win.minsize(520, 260)
        win.transient(self)
        win.grab_set()

        frame = ctk.CTkFrame(win)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(frame, text=title, font=("", 16, "bold"), anchor="w").pack(
            fill="x", padx=10, pady=(10, 6)
        )

        text = ctk.CTkTextbox(frame, wrap="word", font=("Consolas", 12))
        text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        text.insert("1.0", message)
        text.configure(state="disabled")

        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.pack(fill="x", padx=10, pady=(0, 10))

        def _copy_text():
            win.clipboard_clear()
            win.clipboard_append(message)
            win.update()

        def _copy_link():
            if not link_url:
                return
            win.clipboard_clear()
            win.clipboard_append(link_url)
            win.update()

        def _open_link():
            if not link_url:
                return
            try:
                webbrowser.open(link_url)
            except Exception:
                try:
                    os.startfile(link_url)
                except Exception:
                    pass

        ctk.CTkButton(btns, text="복사", width=90, command=_copy_text).pack(side="right")
        if link_url:
            ctk.CTkButton(btns, text="링크 복사", width=100, command=_copy_link).pack(side="right", padx=(0, 8))
            ctk.CTkButton(btns, text=link_label, width=120, command=_open_link).pack(side="right", padx=(0, 8))
        ctk.CTkButton(btns, text="닫기", width=90, command=win.destroy).pack(side="right", padx=(0, 8))

    def _set_status(self, msg):
        msg = self._normalize_ui_message(str(msg))
        color = self._status_color_for_message(msg)

        def _do():
            self.status_label.configure(text=msg, text_color=color)
            ratio = self._parse_progress_ratio_from_status(msg)
            if ratio is None:
                if msg.strip().startswith("✅") or ("완료" in msg and not self.is_running):
                    ratio = 1.0
                elif msg.strip().startswith("❌") and not self.is_running:
                    ratio = 0.0
            if ratio is not None:
                self._set_progress(ratio)

        self._after_safe(_do)

    def _parse_progress_ratio_from_status(self, msg):
        text = str(msg or "")
        pair = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", text)
        if pair:
            cur = int(pair.group(1))
            total = int(pair.group(2))
            if total > 0:
                return max(0.0, min(1.0, float(cur) / float(total)))
        pct = re.search(r"(?<!\d)(\d{1,3})\s*%", text)
        if pct:
            val = int(pct.group(1))
            return max(0.0, min(1.0, float(val) / 100.0))
        return None

    def _set_progress(self, ratio):
        if not hasattr(self, "progress_bar"):
            return
        try:
            value = max(0.0, min(1.0, float(ratio)))
        except Exception:
            return
        self.progress_bar.set(value)
        if hasattr(self, "progress_label"):
            self.progress_label.configure(text=f"{int(round(value * 100.0))}%")

    def _status_color_for_message(self, msg):
        text = str(msg or "")
        lowered = text.lower()

        if self.is_running:
            # 작업 중 상태는 눈에 띄도록 밝은 색 사용
            return "#FFE082"
        if text.strip().startswith("❌") or "error" in lowered or "실패" in text or "오류" in text:
            return "#FF6B6B"
        if text.strip().startswith("⚠") or "warning" in lowered or "경고" in text:
            return "#FFB74D"
        if text.strip().startswith("✅") or "success" in lowered or "완료" in text:
            return "#66BB6A"
        return "gray"

    def _run_auto_validation(self, wav_dir, tg_folder, out_path):
        self._append_log("🧪 OTO 자동 검증 시작...")
        summary = validate_oto_timing(
            wav_dir=wav_dir,
            tg_folder=tg_folder,
            oto_path=out_path,
            language=self._get_language(),
            callback=lambda msg: self._append_log(msg, log_to_file=True),
        )
        err_count = summary.get("errors", 0)
        warn_count = summary.get("warnings", 0)
        if err_count > 0:
            self._append_log(f"⚠ 자동 검증 결과: error {err_count}, warning {warn_count}")
        else:
            self._append_log(f"✅ 자동 검증 결과: warning {warn_count} (error 0)")

    def _path_is_within(self, path, parent):
        try:
            abs_path = os.path.normcase(os.path.abspath(path or ""))
            abs_parent = os.path.normcase(os.path.abspath(parent or ""))
            if not abs_path or not abs_parent:
                return False
            return os.path.commonpath([abs_path, abs_parent]) == abs_parent
        except Exception:
            return False

    def _snapshot_output_tree_for_cleanup(self, out_path):
        out_file = os.path.abspath(str(out_path or "").strip())
        if not out_file:
            return None
        out_dir = os.path.dirname(out_file) or os.getcwd()
        snapshot = {"root": os.path.normcase(out_dir), "files": set(), "dirs": set()}
        if not os.path.isdir(out_dir):
            return snapshot
        for cur_root, _dir_names, file_names in os.walk(out_dir):
            cur_abs = os.path.normcase(os.path.abspath(cur_root))
            snapshot["dirs"].add(cur_abs)
            for name in file_names:
                fpath = os.path.normcase(os.path.abspath(os.path.join(cur_root, name)))
                snapshot["files"].add(fpath)
        return snapshot

    def _is_generated_oto_artifact_file(self, file_name, file_path_norm, keep_file_norm, snapshot_files, snapshot_provided):
        low = str(file_name or "").strip().lower()
        if not low:
            return False
        if file_path_norm == keep_file_norm:
            return False
        if not snapshot_provided:
            # 안전을 위해 스냅샷이 없으면 자동 삭제를 수행하지 않는다.
            return False
        # out_dir가 새로 생성된 경우(snapshot_files가 비어 있음)에도 새 파일로 간주한다.
        is_new_in_run = (not snapshot_files) or (file_path_norm not in snapshot_files)
        if not is_new_in_run:
            return False
        # 자동 생성 부산물 OTO 계열만 정리한다.
        return (
            low == "oto.ini"
            or (low.startswith("oto.") and low.endswith(".ini"))
            or (low.startswith("oto_") and low.endswith(".ini"))
        )

    def _cleanup_generated_output_artifacts(self, out_path, snapshot=None):
        out_file = os.path.abspath(str(out_path or "").strip())
        if not out_file:
            return {"removed_files": 0, "removed_dirs": 0, "failed": 0}
        out_dir = os.path.dirname(out_file) or os.getcwd()
        if not os.path.isdir(out_dir):
            return {"removed_files": 0, "removed_dirs": 0, "failed": 0}

        keep_file = os.path.normcase(out_file)
        snapshot_files = set()
        snapshot_dirs = set()
        snapshot_provided = isinstance(snapshot, dict)
        if snapshot_provided:
            snapshot_files = set(snapshot.get("files") or [])
            snapshot_dirs = set(snapshot.get("dirs") or [])

        removed_files = 0
        removed_dirs = 0
        failed = 0

        for cur_root, _dir_names, file_names in os.walk(out_dir):
            for name in file_names:
                fpath = os.path.abspath(os.path.join(cur_root, name))
                norm = os.path.normcase(fpath)
                if norm == keep_file:
                    continue
                if not self._is_generated_oto_artifact_file(
                    name,
                    norm,
                    keep_file,
                    snapshot_files,
                    snapshot_provided,
                ):
                    continue
                try:
                    os.remove(fpath)
                    removed_files += 1
                except Exception:
                    failed += 1

        textgrids_dir = os.path.abspath(os.path.join(out_dir, "textgrids"))
        # Keep generated TextGrid files for inspection/reuse.
        _ = textgrids_dir

        for cur_root, dir_names, _file_names in os.walk(out_dir, topdown=False):
            for name in dir_names:
                dpath = os.path.abspath(os.path.join(cur_root, name))
                dnorm = os.path.normcase(dpath)
                if snapshot_dirs and dnorm in snapshot_dirs:
                    continue
                if self._path_is_within(out_file, dpath):
                    continue
                try:
                    os.rmdir(dpath)
                    removed_dirs += 1
                except OSError:
                    continue
                except Exception:
                    failed += 1

        if removed_files > 0 or removed_dirs > 0:
            self._append_log(f"🧹 자동 정리 완료: files={removed_files}, dirs={removed_dirs}")
        if failed > 0:
            self._append_log(f"⚠ 자동 정리 실패 항목: {failed}")
        return {"removed_files": removed_files, "removed_dirs": removed_dirs, "failed": failed}

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _set_running(self, running):
        self.is_running = running

        def _do():
            state = "disabled" if running else "normal"
            self.run_btn.configure(state=state)
            if hasattr(self, "status_label"):
                current_text = self.status_label.cget("text")
                self.status_label.configure(text_color=self._status_color_for_message(current_text))
                if running:
                    self._set_progress(0.0)
                else:
                    ratio = self._parse_progress_ratio_from_status(current_text)
                    if ratio is None:
                        if str(current_text).strip().startswith("✅") or "완료" in str(current_text):
                            ratio = 1.0
                        elif str(current_text).strip().startswith("❌"):
                            ratio = 0.0
                    if ratio is not None:
                        self._set_progress(ratio)

        self._after_safe(_do)

    def _run_in_thread(self, func):
        if self.is_running:
            messagebox.showwarning("실행 중", "이미 작업이 진행 중입니다. 완료 후 다시 시도해 주세요.")
            return
        # 탭 이름이 변경되어도 로그 탭으로 안전하게 이동
        switched = False
        for tab_name in ("로그", "📋 로그", "📝 로그"):
            try:
                self.tabview.set(tab_name)
                switched = True
                break
            except Exception:
                continue
        if not switched:
            try:
                tab_dict = getattr(self.tabview, "_tab_dict", {})
                if tab_dict:
                    self.tabview.set(next(iter(tab_dict.keys())))
            except Exception:
                pass
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def _handle_error(self, step_name, exception):
        tb = traceback.format_exc()
        self.logger.error(f"[{step_name}] {exception}\n{tb}")
        self._append_log(f"\n{'=' * 50}")
        self._append_log(f"❌ [{step_name}] 에서 오류가 발생했습니다!")
        self._append_log(f"   오류 내용: {exception}")
        self._append_log("   💡 '오류 제보' 버튼을 눌러 로그 파일을 개발자에게 보내주세요.")
        self._append_log(f"{'=' * 50}\n")
        self._set_status(f"❌ 에러 발생: {step_name}")

    def _export_error_report(self):
        report_path = os.path.join(
            self.app_dir,
            f"error_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== UTAU Auto OTO 오류 보고서 ===\n")
                f.write(f"프로그램 버전: {self.app_version}\n")
                f.write(f"시간: {datetime.datetime.now().isoformat()}\n")
                f.write(f"OS: {sys.platform}\n")
                f.write(f"Python: {sys.version}\n")
                f.write(f"MFA 경로: {self.mfa_path or '미설치'}\n")
                f.write(f"정렬 엔진: {self.aligner_var.get()}\n")
                if hasattr(self, "mfa_align_profile_var"):
                    f.write(f"MFA 정렬 프로필: {self.mfa_align_profile_var.get()}\n")
                f.write(f"구조화 이벤트 로그: {getattr(self, 'event_log_path', '')}\n")
                f.write("\n--- 사용자 설정 ---\n")
                f.write(f"WAV 폴더: {self.wav_entry.get()}\n")
                f.write(f"템플릿 OTO: {self.tpl_entry.get()}\n")
                f.write(f"베이스 OTO 없음: {self.no_base_oto_var.get()}\n")
                f.write(f"출력 경로: {self.out_entry.get()}\n")
                f.write("\n--- 파라미터 ---\n")
                for key, var in self.param_vars.items():
                    f.write(f"{key}: {var.get()}\n")
                f.write("\n--- 로그 내용 ---\n")
                f.write(self.log_text.get("1.0", "end"))

                if os.path.exists(self.log_path):
                    f.write(f"\n--- 상세 로그 파일 ({self.log_path}) ---\n")
                    with open(self.log_path, "r", encoding="utf-8") as lf:
                        f.write(lf.read())
                event_log_path = getattr(self, "event_log_path", "")
                if event_log_path and os.path.exists(event_log_path):
                    f.write(f"\n--- 구조화 이벤트 로그 ({event_log_path}) ---\n")
                    with open(event_log_path, "r", encoding="utf-8") as ef:
                        f.write(ef.read())

            if sys.platform == "win32":
                os.startfile(os.path.dirname(report_path))

            messagebox.showinfo(
                "오류 보고서 생성 완료",
                f"오류 보고서가 생성되었습니다!\n\n"
                f"파일 위치:\n{report_path}\n\n"
                f"이 파일을 개발자에게 보내주시면\n"
                f"문제를 빠르게 해결할 수 있습니다.",
            )
        except Exception as e:
            messagebox.showerror("오류", f"보고서 생성 실패: {e}")


class ConfigMixin:
    def _save_config(self, *_args):
        params = self._get_params()
        config = {
            "wav_dir": self.wav_entry.get(),
            "tpl_path": self.tpl_entry.get(),
            "out_path": self.out_entry.get(),
            "custom_phonemes": self.custom_phoneme_var.get(),
            "alias_suffix": self.alias_suffix_var.get(),
            "openutau_compatible": self.openutau_var.get(),
            "gen_missing_vowels": self.gen_missing_vowels_var.get(),
            "no_base_oto": self.no_base_oto_var.get(),
            "enable_ml_correction": self.enable_ml_correction_var.get() if hasattr(self, "enable_ml_correction_var") else True,
            "ml_route": self.ml_route_var.get() if hasattr(self, "ml_route_var") else "legacy",
            "ml_selector_mode": self.ml_selector_mode_var.get() if hasattr(self, "ml_selector_mode_var") else "기본 정책",
            "ml_coupled_enable": self.ml_coupled_enable_var.get() if hasattr(self, "ml_coupled_enable_var") else True,
            "ml_coupled_min_conf": self.ml_coupled_min_conf_var.get() if hasattr(self, "ml_coupled_min_conf_var") else "",
            "ml_coupled_device": self.ml_coupled_device_var.get() if hasattr(self, "ml_coupled_device_var") else "auto",
            "ml_coupled_backend": self.ml_coupled_backend_var.get() if hasattr(self, "ml_coupled_backend_var") else "auto",
            "ml_coupled_strict_constraint": self.ml_coupled_strict_constraint_var.get() if hasattr(self, "ml_coupled_strict_constraint_var") else False,
            "ml_anchor_mel_gamma": self.ml_anchor_mel_gamma_var.get() if hasattr(self, "ml_anchor_mel_gamma_var") else "",
            "ml_model_root_kr": self.ml_model_root_kr_var.get() if hasattr(self, "ml_model_root_kr_var") else "",
            "ml_model_root_ja": self.ml_model_root_ja_var.get() if hasattr(self, "ml_model_root_ja_var") else "",
            "ja_mapping_words_fallback_enabled": self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True,
            "ja_mapping_spn_ratio_threshold": self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35,
            "ja_mapping_min_vowel_phone_ratio": self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5,
            "ja_mapping_debug_reason_logging": self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True,
            "kr_anchor_profile_path": self.kr_anchor_profile_path_var.get() if hasattr(self, "kr_anchor_profile_path_var") else "",
            "kr_mapping_confidence_threshold": self.kr_mapping_confidence_threshold_var.get() if hasattr(self, "kr_mapping_confidence_threshold_var") else "",
            "kr_mapping_max_index_jump_default": self.kr_mapping_max_index_jump_default_var.get() if hasattr(self, "kr_mapping_max_index_jump_default_var") else 1,
            "kr_mapping_max_index_jump_high_conf": self.kr_mapping_max_index_jump_high_conf_var.get() if hasattr(self, "kr_mapping_max_index_jump_high_conf_var") else 2,
            "kr_continuity_max_offset_adj": self.kr_continuity_max_offset_adj_var.get() if hasattr(self, "kr_continuity_max_offset_adj_var") else "",
            "ml_same_language_borrow_only": self.ml_same_language_borrow_only_var.get() if hasattr(self, "ml_same_language_borrow_only_var") else True,
            "language": self.lang_var.get(),
            "auto_format": self.auto_format_var.get(),
            "ja_alias_style": self.ja_alias_style_var.get(),
            "show_advanced_aligner": self.show_advanced_aligner_var.get() if hasattr(self, "show_advanced_aligner_var") else False,
            "aligner": self.aligner_var.get(),
            "mfa_align_profile": self.mfa_align_profile_var.get() if hasattr(self, "mfa_align_profile_var") else "기본",
            "whisperx_profile": self.whisperx_profile_var.get() if hasattr(self, "whisperx_profile_var") else "balanced",
            "whisperx_device": self.whisperx_device_var.get() if hasattr(self, "whisperx_device_var") else "auto",
            "whisperx_compute_type": self.whisperx_compute_type_var.get() if hasattr(self, "whisperx_compute_type_var") else "int8",
            "whisperx_batch_size": self.whisperx_batch_size_var.get() if hasattr(self, "whisperx_batch_size_var") else 8,
            "whisperx_align_model": self.whisperx_align_model_var.get() if hasattr(self, "whisperx_align_model_var") else "",
            "whisperx_cleanup_intermediate": self.whisperx_cleanup_intermediate_var.get() if hasattr(self, "whisperx_cleanup_intermediate_var") else True,
            "whisperx_save_debug_json": self.whisperx_save_debug_json_var.get() if hasattr(self, "whisperx_save_debug_json_var") else False,
            "tune_auto_oto": self.tune_auto_oto_var.get() if hasattr(self, "tune_auto_oto_var") else "",
            "tune_manual_oto": self.tune_manual_oto_var.get() if hasattr(self, "tune_manual_oto_var") else "",
            "tune_profile_out": self.tune_profile_out_var.get() if hasattr(self, "tune_profile_out_var") else "",
            "tune_apply_target": self.tune_apply_target_var.get() if hasattr(self, "tune_apply_target_var") else "",
            "params": params,
        }
        config_path = os.path.join(self.app_dir, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {e}")

    def _load_config(self):
        config_path = os.path.join(self.app_dir, "config.json")
        if not os.path.exists(config_path):
            if hasattr(self, "_apply_recommended_ml_model_defaults"):
                self._apply_recommended_ml_model_defaults()
            if hasattr(self, "_refresh_ml_backend_status"):
                self._refresh_ml_backend_status()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "wav_dir" in config:
                self.wav_entry.insert(0, config["wav_dir"])
            self.tpl_entry.delete(0, "end")
            self.tpl_entry.insert(0, config.get("tpl_path", ""))
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, config.get("out_path", ""))

            if "openutau_compatible" in config and hasattr(self, "openutau_var"):
                self.openutau_var.set(bool(config.get("openutau_compatible", False)))
            if "gen_missing_vowels" in config and hasattr(self, "gen_missing_vowels_var"):
                self.gen_missing_vowels_var.set(bool(config.get("gen_missing_vowels", True)))
            if "no_base_oto" in config and hasattr(self, "no_base_oto_var"):
                self.no_base_oto_var.set(bool(config.get("no_base_oto", False)))
            if hasattr(self, "enable_ml_correction_var"):
                self.enable_ml_correction_var.set(bool(config.get("enable_ml_correction", True)))
            if "ml_route" in config and hasattr(self, "ml_route_var"):
                saved_route = str(config.get("ml_route", "legacy") or "legacy").strip().lower()
                if saved_route in {"legacy", "autofree_v1"}:
                    self.ml_route_var.set(saved_route)

            if "language" in config and hasattr(self, "lang_var"):
                saved_language = str(config.get("language", "") or "").strip()
                if "Japanese" in saved_language:
                    self.lang_var.set("Japanese (日本語)")
                elif "Korean" in saved_language:
                    self.lang_var.set("Korean (한국어)")

            lang = self._get_language() if hasattr(self, "_get_language") else "korean"
            saved_auto = config.get("auto_format", "자동 감지 (권장)")
            saved_auto_code = normalize_auto_format_value(lang, saved_auto)
            if hasattr(self, "_set_auto_format_from_code"):
                self._set_auto_format_from_code(saved_auto_code, lang)
            if "ja_alias_style" in config:
                saved_style = config.get("ja_alias_style", "원본 그대로")
                if saved_style in {"원본 그대로", "히라가나", "로마자"}:
                    self.ja_alias_style_var.set(saved_style)
            if "aligner" in config and hasattr(self, "aligner_var"):
                self.aligner_var.set("MFA")
            if hasattr(self, "show_advanced_aligner_var"):
                self.show_advanced_aligner_var.set(False)
            if "mfa_align_profile" in config and hasattr(self, "mfa_align_profile_var"):
                saved_profile = str(config.get("mfa_align_profile", "기본") or "").strip()
                legacy_to_current = {
                    "정확도 우선 (기본)": "기본",
                    "default": "기본",
                    "accurate": "정확도 우선",
                    "accurate_adapted": "정확도 우선",
                    "speaker_adapted": "정확도 우선",
                    "fast": "빠름 (저사양 추천)",
                }
                saved_profile = legacy_to_current.get(saved_profile, saved_profile)
                if saved_profile in {"기본", "정확도 우선", "빠름 (저사양 추천)"}:
                    self.mfa_align_profile_var.set(saved_profile)
            if "whisperx_profile" in config and hasattr(self, "whisperx_profile_var"):
                profile = str(config.get("whisperx_profile", "balanced") or "balanced").strip().lower()
                if profile in {"low_load", "balanced", "high_accuracy"}:
                    self.whisperx_profile_var.set(profile)
            if "whisperx_device" in config and hasattr(self, "whisperx_device_var"):
                device = str(config.get("whisperx_device", "auto") or "auto").strip().lower()
                if device in {"auto", "cpu", "cuda"}:
                    self.whisperx_device_var.set(device)
            if "whisperx_compute_type" in config and hasattr(self, "whisperx_compute_type_var"):
                self.whisperx_compute_type_var.set(str(config.get("whisperx_compute_type", "int8") or "int8"))
            if "whisperx_batch_size" in config and hasattr(self, "whisperx_batch_size_var"):
                try:
                    self.whisperx_batch_size_var.set(max(1, int(config.get("whisperx_batch_size", 8))))
                except Exception:
                    pass
            if "whisperx_align_model" in config and hasattr(self, "whisperx_align_model_var"):
                self.whisperx_align_model_var.set(str(config.get("whisperx_align_model", "") or ""))
            if "whisperx_cleanup_intermediate" in config and hasattr(self, "whisperx_cleanup_intermediate_var"):
                self.whisperx_cleanup_intermediate_var.set(bool(config.get("whisperx_cleanup_intermediate", True)))
            if "whisperx_save_debug_json" in config and hasattr(self, "whisperx_save_debug_json_var"):
                self.whisperx_save_debug_json_var.set(bool(config.get("whisperx_save_debug_json", False)))
            if "ml_selector_mode" in config and hasattr(self, "ml_selector_mode_var"):
                saved_selector_mode = str(config.get("ml_selector_mode", "기본 정책") or "").strip()
                if saved_selector_mode in {"기본 정책", "델타만", "델타+셀렉터"}:
                    self.ml_selector_mode_var.set(saved_selector_mode)
            if "ml_coupled_enable" in config and hasattr(self, "ml_coupled_enable_var"):
                self.ml_coupled_enable_var.set(bool(config.get("ml_coupled_enable", True)))
            if "ml_coupled_min_conf" in config and hasattr(self, "ml_coupled_min_conf_var"):
                raw_conf = config.get("ml_coupled_min_conf", "")
                if raw_conf is None:
                    self.ml_coupled_min_conf_var.set("")
                else:
                    txt = str(raw_conf).strip()
                    if not txt:
                        self.ml_coupled_min_conf_var.set("")
                    else:
                        try:
                            conf = max(0.0, min(1.0, float(txt)))
                            if abs(conf - 0.55) <= 1e-9:
                                self.ml_coupled_min_conf_var.set("")
                            else:
                                self.ml_coupled_min_conf_var.set(f"{conf:.2f}".rstrip("0").rstrip("."))
                        except Exception:
                            self.ml_coupled_min_conf_var.set("")
            if "kr_mapping_confidence_threshold" in config and hasattr(self, "kr_mapping_confidence_threshold_var"):
                raw_conf = config.get("kr_mapping_confidence_threshold", "")
                if raw_conf is None:
                    self.kr_mapping_confidence_threshold_var.set("")
                else:
                    txt = str(raw_conf).strip()
                    if not txt:
                        self.kr_mapping_confidence_threshold_var.set("")
                    else:
                        try:
                            conf = max(0.0, min(1.0, float(txt)))
                            if abs(conf - 0.60) <= 1e-9:
                                self.kr_mapping_confidence_threshold_var.set("")
                            else:
                                self.kr_mapping_confidence_threshold_var.set(
                                    f"{conf:.2f}".rstrip("0").rstrip(".")
                                )
                        except Exception:
                            self.kr_mapping_confidence_threshold_var.set("")
            if "kr_continuity_max_offset_adj" in config and hasattr(self, "kr_continuity_max_offset_adj_var"):
                raw_val = config.get("kr_continuity_max_offset_adj", "")
                if raw_val is None:
                    self.kr_continuity_max_offset_adj_var.set("")
                else:
                    txt = str(raw_val).strip()
                    if not txt:
                        self.kr_continuity_max_offset_adj_var.set("")
                    else:
                        try:
                            val = max(0.0, float(txt))
                            if abs(val - 180.0) <= 1e-9:
                                self.kr_continuity_max_offset_adj_var.set("")
                            else:
                                self.kr_continuity_max_offset_adj_var.set(
                                    f"{val:.2f}".rstrip("0").rstrip(".")
                                )
                        except Exception:
                            self.kr_continuity_max_offset_adj_var.set("")
            if "ml_anchor_mel_gamma" in config and hasattr(self, "ml_anchor_mel_gamma_var"):
                raw_val = config.get("ml_anchor_mel_gamma", "")
                if raw_val is None:
                    self.ml_anchor_mel_gamma_var.set("")
                else:
                    txt = str(raw_val).strip()
                    if not txt:
                        self.ml_anchor_mel_gamma_var.set("")
                    else:
                        try:
                            val = max(0.1, float(txt))
                            if abs(val - 1.0) <= 1e-9:
                                self.ml_anchor_mel_gamma_var.set("")
                            else:
                                self.ml_anchor_mel_gamma_var.set(
                                    f"{val:.2f}".rstrip("0").rstrip(".")
                                )
                        except Exception:
                            self.ml_anchor_mel_gamma_var.set("")
            if "ml_coupled_device" in config and hasattr(self, "ml_coupled_device_var"):
                device = str(config.get("ml_coupled_device", "auto") or "auto").strip().lower()
                if device in {"auto", "cpu", "cuda"}:
                    self.ml_coupled_device_var.set(device)
            if "ml_coupled_backend" in config and hasattr(self, "ml_coupled_backend_var"):
                backend = str(config.get("ml_coupled_backend", "auto") or "auto").strip().lower()
                backend = {
                    "ensemble_v1": "ensemble",
                }.get(backend, backend)
                if backend in {"auto", "ensemble"}:
                    self.ml_coupled_backend_var.set(backend)
            if "ml_coupled_strict_constraint" in config and hasattr(self, "ml_coupled_strict_constraint_var"):
                self.ml_coupled_strict_constraint_var.set(bool(config.get("ml_coupled_strict_constraint", False)))
            if "ml_model_root_kr" in config and hasattr(self, "ml_model_root_kr_var"):
                self.ml_model_root_kr_var.set(str(config.get("ml_model_root_kr", "") or ""))
            if "ml_model_root_ja" in config and hasattr(self, "ml_model_root_ja_var"):
                self.ml_model_root_ja_var.set(str(config.get("ml_model_root_ja", "") or ""))

            if hasattr(self, "tune_auto_oto_var"):
                self.tune_auto_oto_var.set(config.get("tune_auto_oto", ""))
            if hasattr(self, "tune_manual_oto_var"):
                self.tune_manual_oto_var.set(config.get("tune_manual_oto", ""))
            if hasattr(self, "tune_profile_out_var"):
                self.tune_profile_out_var.set(config.get("tune_profile_out", ""))
            if hasattr(self, "tune_apply_target_var"):
                self.tune_apply_target_var.set(config.get("tune_apply_target", ""))

            if hasattr(self, "_apply_recommended_ml_model_defaults"):
                self._apply_recommended_ml_model_defaults()
            self._on_language_change(self.lang_var.get())
            self._on_no_base_oto_toggle()
            if hasattr(self, "_sync_aligner_ui"):
                self._sync_aligner_ui()

            if "params" in config:
                params = config["params"]
                for key, val in params.items():
                    if key in self.param_vars:
                        self.param_vars[key].set(val)
        except Exception as e:
            self.logger.error(f"설정 불러오기 실패: {e}")
