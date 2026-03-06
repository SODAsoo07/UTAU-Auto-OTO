import datetime
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
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.log_events import classify_log_message, log_with_event
from core.mfa_runner import ALERT_MFA_PERMISSION_DENIED, ALERT_MSVC_REQUIRED
from core.oto_validator import validate_oto_timing
from core.sofa_runner import get_sofa_env_python


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

    def _browse_save_by_var(self, var, filetypes, defext):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=defext)
        if path:
            var.set(path)


class AppRuntimeMixin:
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
            if "SOFA" in original:
                return "SOFA 작업 진행 중..."
            if "WhisperX" in original or "whisperx" in original:
                return "WhisperX 작업 진행 중..."
            if "OTO" in original:
                return "OTO 작업 진행 중..."
            if "Lab" in original or "lab" in original:
                return "Lab 작업 진행 중..."
            if "dict" in original.lower() or "사전" in original:
                return "사전 작업 진행 중..."
        return text

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
        if msg == ALERT_MSVC_REQUIRED:
            self._after_safe(
                lambda: self._show_copyable_alert(
                    title="MFA 의존성 설치 안내",
                    message=(
                        "MFA 한국어 의존성 설치 중 Microsoft Visual C++ 14.0+가 필요합니다.\n\n"
                        "설치 링크:\n"
                        "https://visualstudio.microsoft.com/visual-cpp-build-tools/\n\n"
                        "권장:\n"
                        "1) C++ Build Tools 설치\n"
                        "2) 터미널 재시작\n"
                        "3) MFA 설치/정렬 재시도"
                    ),
                    alert_key="msvc_required",
                )
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

    def _show_copyable_alert(self, title, message, alert_key=None):
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

        ctk.CTkButton(btns, text="복사", width=90, command=_copy_text).pack(side="right")
        ctk.CTkButton(btns, text="닫기", width=90, command=win.destroy).pack(side="right", padx=(0, 8))

    def _set_status(self, msg):
        msg = self._normalize_ui_message(str(msg))
        color = self._status_color_for_message(msg)

        def _do():
            self.status_label.configure(text=msg, text_color=color)

        self._after_safe(_do)

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
        if isinstance(snapshot, dict):
            snapshot_files = set(snapshot.get("files") or [])
            snapshot_dirs = set(snapshot.get("dirs") or [])

        known_names = {
            ".ja_oto_autotune_profile.json",
            "japanese_dict.txt",
            "korean_dict.txt",
        }
        removed_files = 0
        removed_dirs = 0
        failed = 0

        for cur_root, _dir_names, file_names in os.walk(out_dir):
            for name in file_names:
                fpath = os.path.abspath(os.path.join(cur_root, name))
                norm = os.path.normcase(fpath)
                if norm == keep_file:
                    continue
                low = name.lower()
                is_known_intermediate = (
                    low.endswith(".lab")
                    or low.endswith(".textgrid")
                    or low in known_names
                )
                is_new_in_run = bool(snapshot_files) and norm not in snapshot_files
                if not is_known_intermediate and not is_new_in_run:
                    continue
                try:
                    os.remove(fpath)
                    removed_files += 1
                except Exception:
                    failed += 1

        textgrids_dir = os.path.abspath(os.path.join(out_dir, "textgrids"))
        if os.path.isdir(textgrids_dir) and not self._path_is_within(out_file, textgrids_dir):
            try:
                shutil.rmtree(textgrids_dir)
                removed_dirs += 1
            except Exception:
                failed += 1

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
                f.write(f"SOFA Python: {self.sofa_python_var.get()}\n")
                f.write(f"구조화 이벤트 로그: {getattr(self, 'event_log_path', '')}\n")
                f.write("\n--- 사용자 설정 ---\n")
                f.write(f"WAV 폴더: {self.wav_entry.get()}\n")
                f.write(f"템플릿 OTO: {self.tpl_entry.get()}\n")
                f.write(f"SOFA 체크포인트: {self.sofa_ckpt_var.get()}\n")
                f.write(f"SOFA 사전: {self.sofa_dict_var.get()}\n")
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
            "ja_mapping_words_fallback_enabled": self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True,
            "ja_mapping_spn_ratio_threshold": self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35,
            "ja_mapping_min_vowel_phone_ratio": self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5,
            "ja_mapping_debug_reason_logging": self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True,
            "kr_anchor_profile_path": self.kr_anchor_profile_path_var.get() if hasattr(self, "kr_anchor_profile_path_var") else "",
            "kr_mapping_confidence_threshold": self.kr_mapping_confidence_threshold_var.get() if hasattr(self, "kr_mapping_confidence_threshold_var") else 0.60,
            "kr_mapping_max_index_jump_default": self.kr_mapping_max_index_jump_default_var.get() if hasattr(self, "kr_mapping_max_index_jump_default_var") else 1,
            "kr_mapping_max_index_jump_high_conf": self.kr_mapping_max_index_jump_high_conf_var.get() if hasattr(self, "kr_mapping_max_index_jump_high_conf_var") else 2,
            "ml_same_language_borrow_only": self.ml_same_language_borrow_only_var.get() if hasattr(self, "ml_same_language_borrow_only_var") else True,
            "ml_use_pseudo_labels": self.ml_use_pseudo_labels_var.get() if hasattr(self, "ml_use_pseudo_labels_var") else True,
            "ml_pseudo_weight_high": self.ml_pseudo_weight_high_var.get() if hasattr(self, "ml_pseudo_weight_high_var") else 0.7,
            "ml_pseudo_weight_mid": self.ml_pseudo_weight_mid_var.get() if hasattr(self, "ml_pseudo_weight_mid_var") else 0.4,
            "language": self.lang_var.get(),
            "auto_format": self.auto_format_var.get(),
            "ja_alias_style": self.ja_alias_style_var.get(),
            "aligner": self.aligner_var.get(),
            "mfa_align_profile": self.mfa_align_profile_var.get() if hasattr(self, "mfa_align_profile_var") else "정확도 우선 (기본)",
            "sofa_ckpt": self.sofa_ckpt_var.get(),
            "sofa_dict": self.sofa_dict_var.get(),
            "sofa_python": self.sofa_python_var.get(),
            "sofa_repo_dir": self.sofa_repo_dir_var.get() if hasattr(self, "sofa_repo_dir_var") else "",
            "sofa_mode": self.sofa_mode_var.get() if hasattr(self, "sofa_mode_var") else "force",
            "sofa_g2p": self.sofa_g2p_var.get() if hasattr(self, "sofa_g2p_var") else "Dictionary",
            "sofa_ap_detector": self.sofa_ap_detector_var.get() if hasattr(self, "sofa_ap_detector_var") else "LoudnessSpectralcentroidAPDetector",
            "sofa_ap_detector_config": self.sofa_ap_detector_config_var.get() if hasattr(self, "sofa_ap_detector_config_var") else "",
            "sofa_save_confidence": self.sofa_save_confidence_var.get() if hasattr(self, "sofa_save_confidence_var") else True,
            "sofa_out_formats": self.sofa_out_formats_var.get() if hasattr(self, "sofa_out_formats_var") else "TextGrid",
            "sofa_extra_infer_args": self.sofa_extra_infer_args_var.get() if hasattr(self, "sofa_extra_infer_args_var") else "",
            "sofa_two_pass_retry": self.sofa_two_pass_retry_var.get() if hasattr(self, "sofa_two_pass_retry_var") else True,
            "sofa_two_pass_retry_mode": self.sofa_two_pass_retry_mode_var.get() if hasattr(self, "sofa_two_pass_retry_mode_var") else "match",
            "sofa_confidence_threshold": self.sofa_confidence_threshold_var.get() if hasattr(self, "sofa_confidence_threshold_var") else 0.55,
            "sofa_low_confidence_max_files": self.sofa_low_confidence_max_files_var.get() if hasattr(self, "sofa_low_confidence_max_files_var") else 0,
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

            saved_auto = config.get("auto_format", "자동 감지 (권장)")
            auto_map = {
                "CVC (단독음)": "CVC/연단음",
                "CVC (연단음)": "CVC/연단음",
                "CVVC (기본)": "CVVC",
                "VCV (연속음)": "VCV (연속음)",
            }
            saved_auto = auto_map.get(saved_auto, saved_auto)
            valid_auto_formats = {"자동 감지 (권장)", "CVC/연단음", "CVVC", "VCV (연속음)"}
            if saved_auto in valid_auto_formats:
                self.auto_format_var.set(saved_auto)

            if "custom_phonemes" in config:
                self.custom_phoneme_var.set(config.get("custom_phonemes", ""))
                self.custom_entry.delete(0, "end")
                self.custom_entry.insert(0, self.custom_phoneme_var.get())
            if "alias_suffix" in config:
                self.alias_suffix_var.set(config.get("alias_suffix", ""))
                self.suffix_entry.delete(0, "end")
                self.suffix_entry.insert(0, self.alias_suffix_var.get())
            if "openutau_compatible" in config:
                self.openutau_var.set(config["openutau_compatible"])
            if "gen_missing_vowels" in config:
                self.gen_missing_vowels_var.set(config["gen_missing_vowels"])
            if "no_base_oto" in config:
                self.no_base_oto_var.set(config["no_base_oto"])
            if "enable_ml_correction" in config and hasattr(self, "enable_ml_correction_var"):
                self.enable_ml_correction_var.set(config["enable_ml_correction"])
            if "ja_mapping_words_fallback_enabled" in config and hasattr(self, "ja_mapping_words_fallback_enabled_var"):
                self.ja_mapping_words_fallback_enabled_var.set(bool(config["ja_mapping_words_fallback_enabled"]))
            if "ja_mapping_spn_ratio_threshold" in config and hasattr(self, "ja_mapping_spn_ratio_threshold_var"):
                try:
                    self.ja_mapping_spn_ratio_threshold_var.set(float(config["ja_mapping_spn_ratio_threshold"]))
                except Exception:
                    pass
            if "ja_mapping_min_vowel_phone_ratio" in config and hasattr(self, "ja_mapping_min_vowel_phone_ratio_var"):
                try:
                    self.ja_mapping_min_vowel_phone_ratio_var.set(float(config["ja_mapping_min_vowel_phone_ratio"]))
                except Exception:
                    pass
            if "ja_mapping_debug_reason_logging" in config and hasattr(self, "ja_mapping_debug_reason_logging_var"):
                self.ja_mapping_debug_reason_logging_var.set(bool(config["ja_mapping_debug_reason_logging"]))
            if "kr_anchor_profile_path" in config and hasattr(self, "kr_anchor_profile_path_var"):
                self.kr_anchor_profile_path_var.set(str(config.get("kr_anchor_profile_path", "") or ""))
            if "kr_mapping_confidence_threshold" in config and hasattr(self, "kr_mapping_confidence_threshold_var"):
                try:
                    self.kr_mapping_confidence_threshold_var.set(float(config["kr_mapping_confidence_threshold"]))
                except Exception:
                    pass
            if "kr_mapping_max_index_jump_default" in config and hasattr(self, "kr_mapping_max_index_jump_default_var"):
                try:
                    self.kr_mapping_max_index_jump_default_var.set(int(float(config["kr_mapping_max_index_jump_default"])))
                except Exception:
                    pass
            if "kr_mapping_max_index_jump_high_conf" in config and hasattr(self, "kr_mapping_max_index_jump_high_conf_var"):
                try:
                    self.kr_mapping_max_index_jump_high_conf_var.set(int(float(config["kr_mapping_max_index_jump_high_conf"])))
                except Exception:
                    pass
            if "ml_same_language_borrow_only" in config and hasattr(self, "ml_same_language_borrow_only_var"):
                self.ml_same_language_borrow_only_var.set(bool(config["ml_same_language_borrow_only"]))
            if "ml_use_pseudo_labels" in config and hasattr(self, "ml_use_pseudo_labels_var"):
                self.ml_use_pseudo_labels_var.set(bool(config["ml_use_pseudo_labels"]))
            if "ml_pseudo_weight_high" in config and hasattr(self, "ml_pseudo_weight_high_var"):
                try:
                    self.ml_pseudo_weight_high_var.set(float(config["ml_pseudo_weight_high"]))
                except Exception:
                    pass
            if "ml_pseudo_weight_mid" in config and hasattr(self, "ml_pseudo_weight_mid_var"):
                try:
                    self.ml_pseudo_weight_mid_var.set(float(config["ml_pseudo_weight_mid"]))
                except Exception:
                    pass
            if "language" in config:
                self.lang_var.set(config["language"])
            if "auto_format" in config:
                saved_auto = config["auto_format"]
                auto_map = {
                    "CVC (단독음)": "CVC/연단음",
                    "CVC (연단음)": "CVC/연단음",
                    "CVVC (기본)": "CVVC",
                    "VCV (연속음)": "VCV (연속음)",
                }
                saved_auto = auto_map.get(saved_auto, saved_auto)
                if saved_auto in {"자동 감지 (권장)", "CVC/연단음", "CVVC", "VCV (연속음)"}:
                    self.auto_format_var.set(saved_auto)
            if "ja_alias_style" in config:
                saved_style = config.get("ja_alias_style", "원본 그대로")
                if saved_style in {"원본 그대로", "히라가나", "로마자"}:
                    self.ja_alias_style_var.set(saved_style)
            if "aligner" in config:
                saved_aligner = config.get("aligner", "MFA")
                if saved_aligner in {"MFA", "SOFA"}:
                    self.aligner_var.set(saved_aligner)
            if "mfa_align_profile" in config and hasattr(self, "mfa_align_profile_var"):
                saved_profile = str(config.get("mfa_align_profile", "정확도 우선 (기본)") or "").strip()
                if saved_profile in {"정확도 우선 (기본)", "빠름 (저사양 추천)", "accurate", "fast"}:
                    if saved_profile == "fast":
                        saved_profile = "빠름 (저사양 추천)"
                    elif saved_profile == "accurate":
                        saved_profile = "정확도 우선 (기본)"
                    self.mfa_align_profile_var.set(saved_profile)
            if "sofa_ckpt" in config:
                self.sofa_ckpt_var.set(config.get("sofa_ckpt", ""))
            if "sofa_dict" in config:
                self.sofa_dict_var.set(config.get("sofa_dict", ""))
            if "sofa_python" in config:
                self.sofa_python_var.set(config.get("sofa_python", get_sofa_env_python()))
            if "sofa_repo_dir" in config and hasattr(self, "sofa_repo_dir_var"):
                self.sofa_repo_dir_var.set(config.get("sofa_repo_dir", ""))
            if "sofa_mode" in config and hasattr(self, "sofa_mode_var"):
                self.sofa_mode_var.set(config.get("sofa_mode", "force"))
            if "sofa_g2p" in config and hasattr(self, "sofa_g2p_var"):
                self.sofa_g2p_var.set(config.get("sofa_g2p", "Dictionary"))
            if "sofa_ap_detector" in config and hasattr(self, "sofa_ap_detector_var"):
                self.sofa_ap_detector_var.set(
                    config.get("sofa_ap_detector", "LoudnessSpectralcentroidAPDetector")
                )
            if "sofa_ap_detector_config" in config and hasattr(self, "sofa_ap_detector_config_var"):
                self.sofa_ap_detector_config_var.set(config.get("sofa_ap_detector_config", ""))
            if "sofa_save_confidence" in config and hasattr(self, "sofa_save_confidence_var"):
                self.sofa_save_confidence_var.set(config.get("sofa_save_confidence", True))
            if "sofa_out_formats" in config and hasattr(self, "sofa_out_formats_var"):
                self.sofa_out_formats_var.set(config.get("sofa_out_formats", "TextGrid"))
            if "sofa_extra_infer_args" in config and hasattr(self, "sofa_extra_infer_args_var"):
                self.sofa_extra_infer_args_var.set(config.get("sofa_extra_infer_args", ""))
            if "sofa_two_pass_retry" in config and hasattr(self, "sofa_two_pass_retry_var"):
                self.sofa_two_pass_retry_var.set(config.get("sofa_two_pass_retry", True))
            if "sofa_two_pass_retry_mode" in config and hasattr(self, "sofa_two_pass_retry_mode_var"):
                self.sofa_two_pass_retry_mode_var.set(config.get("sofa_two_pass_retry_mode", "match"))
            if "sofa_confidence_threshold" in config and hasattr(self, "sofa_confidence_threshold_var"):
                self.sofa_confidence_threshold_var.set(config.get("sofa_confidence_threshold", 0.55))
            if "sofa_low_confidence_max_files" in config and hasattr(self, "sofa_low_confidence_max_files_var"):
                self.sofa_low_confidence_max_files_var.set(config.get("sofa_low_confidence_max_files", 0))
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

            if hasattr(self, "tune_auto_oto_var"):
                self.tune_auto_oto_var.set(config.get("tune_auto_oto", ""))
            if hasattr(self, "tune_manual_oto_var"):
                self.tune_manual_oto_var.set(config.get("tune_manual_oto", ""))
            if hasattr(self, "tune_profile_out_var"):
                self.tune_profile_out_var.set(config.get("tune_profile_out", ""))
            if hasattr(self, "tune_apply_target_var"):
                self.tune_apply_target_var.set(config.get("tune_apply_target", ""))

            self._on_language_change(self.lang_var.get())
            self._on_no_base_oto_toggle()

            if "params" in config:
                params = config["params"]
                for key, val in params.items():
                    if key in self.param_vars:
                        self.param_vars[key].set(val)
        except Exception as e:
            self.logger.error(f"설정 불러오기 실패: {e}")
