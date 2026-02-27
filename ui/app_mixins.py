import datetime
import json
import logging
import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

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

    def _append_log(self, msg):
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

        def _do():
            self.log_text.insert("end", msg + "\n")
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
            callback=self._append_log,
        )
        err_count = summary.get("errors", 0)
        warn_count = summary.get("warnings", 0)
        if err_count > 0:
            self._append_log(f"⚠ 자동 검증 결과: error {err_count}, warning {warn_count}")
        else:
            self._append_log(f"✅ 자동 검증 결과: warning {warn_count} (error 0)")

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
        self.tabview.set("📋 로그")
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
                f.write(f"SOFA Python: {self.sofa_python_var.get()}\n")
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
    def _save_config(self):
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
            "language": self.lang_var.get(),
            "auto_format": self.auto_format_var.get(),
            "ja_alias_style": self.ja_alias_style_var.get(),
            "aligner": self.aligner_var.get(),
            "sofa_ckpt": self.sofa_ckpt_var.get(),
            "sofa_dict": self.sofa_dict_var.get(),
            "sofa_python": self.sofa_python_var.get(),
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
            if "sofa_ckpt" in config:
                self.sofa_ckpt_var.set(config.get("sofa_ckpt", ""))
            if "sofa_dict" in config:
                self.sofa_dict_var.set(config.get("sofa_dict", ""))
            if "sofa_python" in config:
                self.sofa_python_var.set(config.get("sofa_python", get_sofa_env_python()))

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
