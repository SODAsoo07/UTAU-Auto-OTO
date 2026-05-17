import datetime
import glob
import json
import logging
import os
import platform
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
from core.runtime_paths import resolve_setup_mfa_script_path
from ui.theme_tokens import DEFAULT_THEME_PROFILE, PALETTE, normalize_theme_profile_name
from ui.i18n import t

_HANGUL_CHAR_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]")
_KANA_CHAR_RE = re.compile(r"[\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC\u30FD-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]")


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
        external_root = os.path.join(base_dir, "models")
        installed_root = os.path.join(base_dir, "models_installed", "oto_ml")
        noml_auto_root = os.path.join(base_dir, "ML_models_noml_auto")
        legacy_root = os.path.join(base_dir, "ML_models")

        for candidate in (external_root, noml_auto_root, installed_root, legacy_root):
            if self._dir_has_model_meta(candidate):
                return candidate
        for candidate in (external_root, noml_auto_root, installed_root, legacy_root):
            if os.path.isdir(candidate):
                return candidate
        return legacy_root

    def _ml_model_language_root(self, language):
        lang = str(language or "").strip().lower()
        repo_root = self._ml_model_repo_root()
        direct = os.path.join(repo_root, lang)
        if os.path.isdir(direct):
            return direct
        nested = os.path.join(repo_root, "oto_ml", lang)
        if os.path.isdir(nested):
            return nested
        return direct

    def _ml_model_language_for_var(self, var):
        if hasattr(self, "ml_model_root_ja_var") and var is self.ml_model_root_ja_var:
            return "japanese"
        return "korean"

    def _is_legacy_ml_language_root(self, path, language):
        raw = str(path or "").strip()
        if not raw:
            return False
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy_lang_root = os.path.abspath(os.path.join(base_dir, "ML_models", str(language or "").strip().lower()))
        target = os.path.abspath(raw)
        try:
            common = os.path.commonpath([target, legacy_lang_root])
        except Exception:
            return False
        return common == legacy_lang_root

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
        noml_root = os.path.join(base_dir, "ML_models_noml_auto", str(language or "").strip().lower())
        if os.path.isdir(noml_root):
            return noml_root
        legacy_root = os.path.join(base_dir, "ML_models", language)
        if os.path.isdir(legacy_root):
            return legacy_root
        repo_root = self._ml_model_repo_root()
        if os.path.isdir(repo_root):
            return repo_root
        return ""

    def _recommended_e2e_model_root(self):
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        primary_root = os.path.join(base_dir, "models", "oto_e2e")
        fallback_root = os.path.join(base_dir, "assets", "models", "oto_e2e")

        for candidate in (primary_root, fallback_root):
            if self._dir_has_model_meta(candidate):
                return candidate
        for candidate in (primary_root, fallback_root):
            if os.path.isdir(candidate):
                return candidate
        return ""

    def _clear_e2e_model_envs(self):
        for key in (
            "UTOA_E2E_MODEL_DIR",
            "UTOA_E2E_ALIAS_MODEL_DIR",
            "UTOA_KR_E2E_MODEL_DIR",
            "UTOA_JA_E2E_MODEL_DIR",
            "UTOA_KR_E2E_ALIAS_MODEL_DIR",
            "UTOA_JA_E2E_ALIAS_MODEL_DIR",
        ):
            os.environ.pop(key, None)

    def _apply_e2e_model_env(self, language):
        self._clear_e2e_model_envs()
        lang = str(language or "").strip().lower()
        if lang not in {"korean", "japanese"}:
            return {"model_root": "", "alias_root": "", "has_models": False}

        model_root = str(self._recommended_e2e_model_root() or "").strip()
        if not model_root or not os.path.isdir(model_root):
            return {"model_root": "", "alias_root": "", "has_models": False}

        os.environ["UTOA_E2E_MODEL_DIR"] = model_root

        alias_root = os.path.join(model_root, lang, "by_alias")
        if os.path.isdir(alias_root):
            os.environ["UTOA_E2E_ALIAS_MODEL_DIR"] = alias_root
            alias_key = "UTOA_JA_E2E_ALIAS_MODEL_DIR" if lang == "japanese" else "UTOA_KR_E2E_ALIAS_MODEL_DIR"
            os.environ[alias_key] = alias_root
        else:
            alias_root = ""

        return {
            "model_root": model_root,
            "alias_root": alias_root,
            "has_models": bool(self._dir_has_model_meta(model_root)),
        }

    def _language_has_ensemble_bundle(self, language):
        lang_root = self._ml_model_language_root(language)
        if not os.path.isdir(lang_root):
            return False
        pattern = os.path.join(lang_root, "*", "v1_ensemble", "model_meta.json")
        return any(os.path.isfile(path) for path in glob.glob(pattern))

    def _apply_recommended_ml_model_defaults(self):
        current_language = self._get_language() if hasattr(self, "_get_language") else "korean"
        base_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets = []
        if hasattr(self, "ml_model_root_kr_var"):
            targets.append(("korean", self.ml_model_root_kr_var))
        if hasattr(self, "ml_model_root_ja_var"):
            targets.append(("japanese", self.ml_model_root_ja_var))

        for language, target_var in targets:
            current_root = str(target_var.get() or "").strip()
            recommended_root = str(self._recommended_ml_model_root(language) or "").strip()
            if not recommended_root:
                continue
            use_recommended = False
            models_lang_root = os.path.abspath(os.path.join(base_dir, "models", str(language or "").strip().lower()))
            recommended_abs = os.path.abspath(recommended_root)
            current_abs = os.path.abspath(current_root) if current_root else ""
            if not current_root or not os.path.isdir(current_root):
                use_recommended = True
            elif recommended_abs == models_lang_root and self._dir_has_model_meta(models_lang_root):
                if current_abs != models_lang_root:
                    use_recommended = True
            elif "ML_models_noml_auto" in os.path.abspath(recommended_root):
                if self._is_legacy_ml_language_root(current_root, language):
                    use_recommended = True
            if use_recommended:
                target_var.set(recommended_root)

        if hasattr(self, "ml_coupled_backend_var"):
            backend = str(self.ml_coupled_backend_var.get() or "auto").strip().lower()
            if backend in {"", "auto"} and self._language_has_ensemble_bundle(current_language):
                self.ml_coupled_backend_var.set("ensemble")

    def _effective_ml_model_root(self, language):
        lang = str(language or "").strip().lower()
        configured = ""
        if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
            configured = str(self.ml_model_root_ja_var.get() or "").strip()
        elif lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
            configured = str(self.ml_model_root_kr_var.get() or "").strip()
        if configured and self._dir_has_model_meta(configured):
            return os.path.abspath(configured), "configured"
        recommended = str(self._recommended_ml_model_root(lang) or "").strip()
        if recommended and self._dir_has_model_meta(recommended):
            return os.path.abspath(recommended), "recommended"
        env_key = "UTOA_JA_OTO_ML_DIR" if lang == "japanese" else "UTOA_KR_OTO_ML_DIR"
        env_root = str(os.environ.get(env_key, "") or "").strip()
        if env_root and self._dir_has_model_meta(env_root):
            return os.path.abspath(env_root), "env"
        return (configured or recommended or env_root or ""), "missing"

    def _browse_folder_by_var(self, var, initial_dir=""):
        path = filedialog.askdirectory(initialdir=initial_dir or self._preferred_ml_model_browse_dir(var))
        if path:
            var.set(path)

    def _dir_has_model_meta(self, path):
        raw = str(path or "").strip()
        if not raw or not os.path.isdir(raw):
            return False
        try:
            for _root, _dirs, files in os.walk(raw):
                if "model_meta.json" in files:
                    return True
        except Exception:
            return False
        return False

    def _detect_language_model_root(self, selected_root, language):
        lang = str(language or "").strip().lower()
        base = str(selected_root or "").strip()
        if not base:
            return ""
        base_abs = os.path.abspath(base)

        candidates = [
            os.path.join(base, lang),
            os.path.join(base, "oto_ml", lang),
            os.path.join(base, "ML_models", lang),
            os.path.join(base, "models", lang),
            os.path.join(base, "models", "oto_ml", lang),
        ]
        for candidate in candidates:
            if self._dir_has_model_meta(candidate):
                return candidate

        # Allow selecting a language root directly (e.g. .../models/korean)
        leaf = os.path.basename(os.path.normpath(base)).strip().lower()
        if leaf == lang and self._dir_has_model_meta(base):
            return base

        # Recursive fallback: allow selecting a parent folder that contains model trees.
        # Pick the highest (shortest) language root, preferring non-autofree trees.
        scored = []
        try:
            for root, _dirs, files in os.walk(base_abs):
                if "model_meta.json" not in files:
                    continue
                norm_parts = os.path.normpath(root).split(os.sep)
                lower_parts = [part.lower() for part in norm_parts]
                for idx, part in enumerate(lower_parts):
                    if part != lang:
                        continue
                    cand = os.sep.join(norm_parts[: idx + 1])
                    cand_abs = os.path.abspath(cand)
                    if not cand_abs.lower().startswith(base_abs.lower()):
                        continue
                    if not self._dir_has_model_meta(cand_abs):
                        continue
                    in_autofree = "autofree" in lower_parts[:idx]
                    priority = 1 if not in_autofree else 2
                    depth = cand_abs.count(os.sep)
                    scored.append((priority, depth, len(cand_abs), cand_abs))
        except Exception:
            scored = []

        if scored:
            scored.sort()
            return scored[0][3]
        return ""

    def _browse_and_apply_downloaded_model_root(self):
        initial_dir = self._ml_model_repo_root() if hasattr(self, "_ml_model_repo_root") else ""
        selected = filedialog.askdirectory(initialdir=initial_dir)
        if not selected:
            return

        kr_root = self._detect_language_model_root(selected, "korean")
        ja_root = self._detect_language_model_root(selected, "japanese")

        if hasattr(self, "ml_model_root_kr_var") and kr_root:
            self.ml_model_root_kr_var.set(kr_root)
        if hasattr(self, "ml_model_root_ja_var") and ja_root:
            self.ml_model_root_ja_var.set(ja_root)

        if hasattr(self, "_on_ml_model_root_change"):
            self._on_ml_model_root_change()

        try:
            if hasattr(self, "_append_log"):
                self._append_log(
                    "[OTO-ML] 모델 루트를 적용했습니다: "
                    f"KR={kr_root or '(없음)'} / JA={ja_root or '(없음)'}"
                )
        except Exception:
            pass

        if not kr_root and not ja_root:
            messagebox.showwarning(
                "모델 경로 확인",
                "선택한 폴더에서 model_meta.json을 찾지 못했습니다.\n"
                "예: models/korean 또는 models/oto_ml/korean 경로를 선택해 주세요.",
            )
        elif (not kr_root) or (not ja_root):
            messagebox.showinfo(
                "모델 경로 일부 적용",
                "한쪽 언어 모델만 찾았습니다.\n"
                f"한국어: {kr_root or '없음'}\n일본어: {ja_root or '없음'}",
            )

    def _browse_save_by_var(self, var, filetypes, defext):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=defext)
        if path:
            var.set(path)


class AppRuntimeMixin:
    _MSVC_BUILD_TOOLS_URL = "https://visualstudio.microsoft.com/visual-cpp-build-tools/"

    def _path_mask_roots(self):
        roots = []
        seen = set()

        def _add_root(label, value):
            raw = str(value or "").strip()
            if not raw:
                return
            try:
                normalized = os.path.normcase(os.path.abspath(raw))
            except Exception:
                return
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            roots.append((label, normalized))

        _add_root("app", getattr(self, "app_dir", ""))
        _add_root("data", getattr(self, "writable_data_dir", ""))
        _add_root("data", getattr(self, "app_data_dir", ""))
        try:
            _add_root("cwd", os.getcwd())
        except Exception:
            pass
        _add_root("user", os.environ.get("USERPROFILE", ""))
        _add_root("localappdata", os.environ.get("LOCALAPPDATA", ""))
        _add_root("temp", os.environ.get("TEMP", ""))
        _add_root("tmp", os.environ.get("TMP", ""))
        return roots

    def _sanitize_path_value(self, value):
        text = str(value or "").strip()
        if not text:
            return ""
        expanded = os.path.expandvars(os.path.expanduser(text))
        looks_abs = bool(re.match(r"^[A-Za-z]:[\\/]", expanded)) or expanded.startswith("\\\\")
        if not looks_abs:
            return text.replace("\\", "/")

        try:
            normalized = os.path.normcase(os.path.abspath(expanded))
        except Exception:
            normalized = expanded

        for label, root in self._path_mask_roots():
            root_cmp = root.rstrip("\\/")
            if normalized == root_cmp:
                return f"<{label}>"
            prefix = root_cmp + os.sep
            if normalized.startswith(prefix):
                try:
                    rel = os.path.relpath(expanded, root_cmp)
                except Exception:
                    rel = expanded[len(prefix):]
                rel = str(rel or "").replace("\\", "/").lstrip("./")
                return f"<{label}>/{rel}" if rel else f"<{label}>"

        tail = re.split(r"[\\/]+", expanded.strip("\\/"))
        tail = [part for part in tail if part and not re.match(r"^[A-Za-z]:$", part)]
        if not tail:
            return "<path>"
        short_tail = "/".join(tail[-3:])
        return f"<path>/{short_tail}"

    def _mask_sensitive_text(self, text):
        masked = str(text or "")
        if not masked:
            return masked

        def _replace_path(match):
            return self._sanitize_path_value(match.group(0))

        masked = re.sub(r"(?<![\w/])(?:[A-Za-z]:\\|[A-Za-z]:/)[^\s\"'<>|]+", _replace_path, masked)
        masked = re.sub(r"\\\\[^\s\"'<>|]+", _replace_path, masked)

        usernames = set()
        for value in (
            os.environ.get("USERNAME", ""),
            os.path.basename(os.environ.get("USERPROFILE", "") or ""),
            os.path.basename(os.environ.get("HOME", "") or ""),
        ):
            token = str(value or "").strip()
            if token:
                usernames.add(token)

        masked = re.sub(r"(?i)(\\\\users\\\\)([^\\\\/\r\n]+)", r"\1***", masked)
        masked = re.sub(r"(?i)(/users/)([^/\r\n]+)", r"\1***", masked)
        for username in sorted(usernames, key=len, reverse=True):
            masked = re.sub(re.escape(username), "***", masked, flags=re.IGNORECASE)
        return masked

    def _normalize_ui_message(self, msg: str) -> str:
        original = self._mask_sensitive_text(str(msg or ""))
        text = original
        if not text:
            return text

        text = re.sub(r"[\uF8F0-\uF8FF]", " ", text)
        text = re.sub(r"[\uFF61-\uFF9F]", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        def _looks_like_mojibake(s: str) -> bool:
            if not s:
                return False
            markers = ("繝", "荳", "逕", "蛯", "笞", "隨", "邃", "", "甯", "嶹", "・ｻ", "ｽ")
            marker_hits = sum(s.count(m) for m in markers)
            replacement_hits = s.count("\ufffd")
            broken_halfwidth = len(re.findall(r"[ｦ-ﾟ]", s))
            return marker_hits >= 2 or replacement_hits >= 1 or broken_halfwidth >= 6

        if len(text) <= 2 or _looks_like_mojibake(text) or _looks_like_mojibake(original):
            lowered = original.lower()
            if "mfa" in lowered:
                return "MFA task in progress..."
            if "whisperx" in lowered:
                return "WhisperX task in progress..."
            if "oto" in lowered:
                return "OTO task in progress..."
            if "lab" in lowered:
                return "Lab generation in progress..."
            if "dict" in lowered or "dictionary" in lowered:
                return "Dictionary processing in progress..."
            return "Processing..."
        return text

    def _install_global_exception_hooks(self):
        if getattr(self, "_global_exception_hooks_installed", False):
            return
        self._global_exception_hooks_installed = True

        def _tk_exception_handler(exc_type, exc_value, exc_traceback):
            if isinstance(exc_value, KeyboardInterrupt):
                return
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            if self._is_transient_tk_callback_error(exc_value, tb):
                self.logger.warning(
                    "[UI Callback] transient TclError suppressed during rebuild: %s",
                    self._mask_sensitive_text(str(exc_value or "")),
                )
                return
            self._handle_error("UI Callback", exc_value, traceback_text=tb, auto_popup=True)

        self.report_callback_exception = _tk_exception_handler

        previous_thread_hook = getattr(threading, "excepthook", None)

        def _thread_exception_handler(args):
            try:
                if args is None or args.exc_type is KeyboardInterrupt:
                    return
                tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
                if self._is_transient_tk_callback_error(args.exc_value, tb):
                    self.logger.warning(
                        "[Thread Callback] transient TclError suppressed during rebuild: %s",
                        self._mask_sensitive_text(str(args.exc_value or "")),
                    )
                    return
                thread_name = getattr(getattr(args, "thread", None), "name", "") or "worker"
                self._after_safe(
                    lambda step=f"Thread:{thread_name}", err=args.exc_value, trace=tb: self._handle_error(
                        step,
                        err,
                        traceback_text=trace,
                        auto_popup=True,
                    )
                )
            finally:
                if callable(previous_thread_hook):
                    try:
                        previous_thread_hook(args)
                    except Exception:
                        pass

        if callable(previous_thread_hook):
            threading.excepthook = _thread_exception_handler

    def _is_transient_tk_callback_error(self, exception, traceback_text=""):
        if not isinstance(exception, tk.TclError):
            return False
        msg = str(exception or "").strip().lower()
        during_theme_rebuild = bool(getattr(self, "_theme_rebuild_pending", False)) or bool(
            getattr(self, "_suppress_ui_callback_errors", False)
        )
        transient_tokens = (
            "invalid command name",
            "application has been destroyed",
            "can't invoke",
            "bad window path name",
        )
        if any(token in msg for token in transient_tokens):
            return True
        if during_theme_rebuild and "image" in msg and "doesn't exist" in msg:
            return True
        if during_theme_rebuild and ("theme" in msg or "appearance" in msg):
            return True
        if during_theme_rebuild and "tclerror" in str(traceback_text or "").lower():
            return True
        return False

    def _should_show_error_popup(self, step_name: str, exception_text: str) -> bool:
        now = float(time.monotonic())
        signature = f"{str(step_name or '').strip()}::{str(exception_text or '').strip()}"
        last_signature = str(getattr(self, "_last_error_popup_signature", "") or "")
        last_time = float(getattr(self, "_last_error_popup_time", 0.0) or 0.0)
        if signature and signature == last_signature and (now - last_time) < 2.0:
            repeat_count = int(getattr(self, "_last_error_popup_repeat_count", 0) or 0) + 1
            self._last_error_popup_repeat_count = repeat_count
            if repeat_count == 1:
                try:
                    self._append_log("[UI] Repeated identical error popups suppressed.")
                except Exception:
                    pass
            return False
        self._last_error_popup_signature = signature
        self._last_error_popup_time = now
        self._last_error_popup_repeat_count = 0
        return True

    def _normalize_ml_selector_mode(self, value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode in {
            "delta",
            "delta_only",
            "delta only",
            "delta-only",
            "델타",
            "델타만",
            "상대적 보정",
            "상대적보정",
            "상대 보정",
        }:
            return "delta"
        if mode in {
            "selector",
            "delta+selector",
            "delta + selector",
            "델타+셀렉터",
            "델타 + 셀렉터",
            "셀렉터",
            "+셀렉터",
            "플러스셀렉터",
            "플러스 셀렉터",
        }:
            return "selector"
        if mode in {"기본 정책", "기본정책", "policy"}:
            return "policy"
        return "policy"

    def _describe_ml_selector_mode(self, value: str) -> str:
        mode = self._normalize_ml_selector_mode(value)
        if mode == "delta":
            return "delta"
        if mode == "selector":
            return "delta+selector"
        return "policy"

    def _apply_ml_selector_runtime_mode(self, value: str) -> str:
        mode = self._normalize_ml_selector_mode(value)
        if mode == "delta":
            os.environ["UTOA_DISABLE_OTO_SELECTOR"] = "1"
            os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
            os.environ["UTOA_ML_SELECTOR_ENABLE"] = "0"
        elif mode == "selector":
            os.environ["UTOA_FORCE_OTO_SELECTOR"] = "1"
            os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
            os.environ["UTOA_ML_SELECTOR_ENABLE"] = "1"
        else:
            os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
            os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
            # policy mode delegates to runtime default (currently selector OFF).
            os.environ.pop("UTOA_ML_SELECTOR_ENABLE", None)
        return mode

    def _resolve_auto_format_for_policy(self, language: str, auto_format_value: str) -> str:
        lang = str(language or "korean").strip().lower()
        try:
            normalized = normalize_auto_format_value(lang, auto_format_value)
        except Exception:
            normalized = ""
        fmt = str(normalized or "").strip().lower()
        if not fmt:
            return "general"
        if "auto" in fmt or "・尖徐" in fmt:
            return "general"
        return fmt

    def _clear_ml_override_envs(self) -> None:
        for key in list(os.environ.keys()):
            if key.startswith("UTOA_ML_COUPLED_MIN_CONF_"):
                os.environ.pop(key, None)
        for key in (
            "UTOA_ML_COUPLED_MIN_CONF",
            "UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET",
            "UTOA_ML_GATED_ENSEMBLE_ENABLE",
            "UTOA_ML_ANCHOR_MEL_GAMMA",
            "UTOA_E2E_ENABLE",
            "UTOA_E2E_MODE",
            "UTOA_E2E_T_LOW",
            "UTOA_E2E_T_HIGH",
            "UTOA_E2E_BLEND_ALPHA",
            "UTOA_E2E_MODEL_DIR",
            "UTOA_E2E_ALIAS_MODEL_DIR",
            "UTOA_KR_E2E_MODEL_DIR",
            "UTOA_JA_E2E_MODEL_DIR",
            "UTOA_KR_E2E_ALIAS_MODEL_DIR",
            "UTOA_JA_E2E_ALIAS_MODEL_DIR",
        ):
            os.environ.pop(key, None)

    def _clear_advanced_tuning_envs(self) -> None:
        for key in (
            "UTOA_KR_MAPPING_CONF_THRESHOLD",
            "UTOA_KR_MAPPING_ONLY_ENABLE",
            "UTOA_KR_CONTINUITY_ENABLE",
            "UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ",
            "UTOA_KR_VC_NEIGHBOR_ENABLE",
            "UTOA_KR_VC_NEIGHBOR_BLEND",
            "UTOA_KR_VC_NEIGHBOR_MAX_SHIFT",
            "UTOA_KR_VC_NEIGHBOR_LEAD_MS",
            "UTOA_KR_VC_NEIGHBOR_TAIL_MS",
            "UTOA_KR_VC_NEIGHBOR_MIN_LEN",
            "UTOA_JA_VC_NEIGHBOR_ENABLE",
            "UTOA_JA_VC_NEIGHBOR_BLEND",
            "UTOA_JA_VC_NEIGHBOR_MAX_SHIFT",
            "UTOA_JA_VC_NEIGHBOR_LEAD_MS",
            "UTOA_JA_VC_NEIGHBOR_TAIL_MS",
            "UTOA_JA_VC_NEIGHBOR_MIN_LEN",
            "UTOA_CVN_CORRECTION_ENABLE",
            "UTOA_CVN_LOW_CONF_ONLY",
            "UTOA_CVN_C_THRESHOLD",
            "UTOA_MAPPING_SUPERVISED_ENABLE",
            "UTOA_MAPPING_SUPERVISED_MODE",
            "UTOA_CV_ORDER_PRIOR_ENABLE",
            "UTOA_CV_ORDER_PRIOR_STRENGTH",
            "UTOA_SOFT_BANK_MODE",
            "UTOA_MFA_SOFT_BANK_MODE",
            "UTOA_MEL_SOUND_DB_MARGIN",
            "UTOA_MEL_SOUND_ENERGY_MIN",
            "UTOA_MEL_HARD_SILENCE_ENERGY_MAX",
            "UTOA_MEL_SOFT_SILENCE_ENERGY_MAX",
            "UTOA_MEL_WEAK_F2_MAX",
            "UTOA_MEL_WEAK_F3_MAX",
            "UTOA_MEL_WEAK_HIGH_MAX",
            "UTOA_MEL_ONSET_SLOPE_BASE",
            "UTOA_MEL_ONSET_ENERGY_BASE",
            "UTOA_MEL_ONSET_DB_MARGIN_BASE",
            "UTOA_MEL_ONSET_ENERGY_SIBILANT",
            "UTOA_MEL_ONSET_DB_MARGIN_SIBILANT",
            "UTOA_MFA_LOW_RMS_GAIN_ENABLE",
            "UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE",
            "UTOA_MFA_WEAK_VOICE_PREEMPH_MIX",
            "UTOA_SYLLABLE_STRICT_MODE",
            "UTOA_LOW_CONF_FORCE_LOCK_MODE",
            "UTOA_WEAK_BOUNDARY_MISSING_REDUCTION_ENABLE",
            "UTOA_WEAK_BOUNDARY_ANTIMISMAP_ENABLE",
        ):
            os.environ.pop(key, None)

    def _apply_advanced_tuning_envs(self) -> None:
        self._clear_advanced_tuning_envs()

        def _parse_float(raw_value):
            txt = str(raw_value or "").strip()
            if not txt:
                return None
            try:
                return float(txt)
            except Exception:
                return None

        def _set_float_env(var_name: str, env_name: str, *, min_value=None, max_value=None):
            if not hasattr(self, var_name):
                return
            try:
                raw = getattr(self, var_name).get()
            except Exception:
                return
            val = _parse_float(raw)
            if val is None:
                return
            if min_value is not None:
                val = max(float(min_value), val)
            if max_value is not None:
                val = min(float(max_value), val)
            os.environ[env_name] = f"{val:.3f}".rstrip("0").rstrip(".")

        file_consistency_enabled = (
            bool(self.vc_correction_enable_var.get())
            if hasattr(self, "vc_correction_enable_var")
            else True
        )
        continuity_enabled = file_consistency_enabled and (
            bool(self.kr_continuity_enable_var.get())
            if hasattr(self, "kr_continuity_enable_var")
            else True
        )
        os.environ["UTOA_KR_CONTINUITY_ENABLE"] = "1" if continuity_enabled else "0"
        if continuity_enabled:
            _set_float_env(
                "kr_continuity_max_offset_adj_var",
                "UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ",
                min_value=0.0,
            )

        kr_vc_enabled = file_consistency_enabled and (
            bool(self.kr_vc_neighbor_enable_var.get())
            if hasattr(self, "kr_vc_neighbor_enable_var")
            else True
        )
        ja_vc_enabled = file_consistency_enabled and (
            bool(self.ja_vc_neighbor_enable_var.get())
            if hasattr(self, "ja_vc_neighbor_enable_var")
            else True
        )
        os.environ["UTOA_KR_VC_NEIGHBOR_ENABLE"] = "1" if kr_vc_enabled else "0"
        os.environ["UTOA_JA_VC_NEIGHBOR_ENABLE"] = "1" if ja_vc_enabled else "0"
        cvn_correction_enabled = (
            bool(self.cvn_correction_enable_var.get())
            if hasattr(self, "cvn_correction_enable_var")
            else True
        )
        os.environ["UTOA_CVN_CORRECTION_ENABLE"] = "1" if cvn_correction_enabled else "0"
        cvn_low_conf_only = (
            bool(self.cvn_low_conf_only_var.get())
            if hasattr(self, "cvn_low_conf_only_var")
            else False
        )
        os.environ["UTOA_CVN_LOW_CONF_ONLY"] = "1" if cvn_low_conf_only else "0"
        os.environ["UTOA_CVN_C_THRESHOLD"] = "0.4"
        mapping_supervised_enabled = (
            bool(self.mapping_supervised_enable_var.get())
            if hasattr(self, "mapping_supervised_enable_var")
            else True
        )
        os.environ["UTOA_MAPPING_SUPERVISED_ENABLE"] = "1" if mapping_supervised_enabled else "0"
        mapping_mode_code = (
            self._get_mapping_supervised_mode_code()
            if hasattr(self, "_get_mapping_supervised_mode_code")
            else "auto"
        )
        os.environ["UTOA_MAPPING_SUPERVISED_MODE"] = str(mapping_mode_code or "auto").strip().lower()
        cv_order_prior_enabled = (
            bool(self.cv_order_prior_enable_var.get())
            if hasattr(self, "cv_order_prior_enable_var")
            else True
        )
        os.environ["UTOA_CV_ORDER_PRIOR_ENABLE"] = "1" if cv_order_prior_enabled else "0"
        _set_float_env(
            "cv_order_prior_strength_var",
            "UTOA_CV_ORDER_PRIOR_STRENGTH",
            min_value=0.0,
            max_value=1.0,
        )

        _set_float_env("kr_vc_neighbor_blend_var", "UTOA_KR_VC_NEIGHBOR_BLEND", min_value=0.0, max_value=1.0)
        _set_float_env("kr_vc_neighbor_max_shift_var", "UTOA_KR_VC_NEIGHBOR_MAX_SHIFT", min_value=0.0)
        _set_float_env("kr_vc_neighbor_lead_ms_var", "UTOA_KR_VC_NEIGHBOR_LEAD_MS", min_value=0.0)
        _set_float_env("kr_vc_neighbor_tail_ms_var", "UTOA_KR_VC_NEIGHBOR_TAIL_MS", min_value=0.0)
        _set_float_env("kr_vc_neighbor_min_len_var", "UTOA_KR_VC_NEIGHBOR_MIN_LEN", min_value=0.0)

        _set_float_env("ja_vc_neighbor_blend_var", "UTOA_JA_VC_NEIGHBOR_BLEND", min_value=0.0, max_value=1.0)
        _set_float_env("ja_vc_neighbor_max_shift_var", "UTOA_JA_VC_NEIGHBOR_MAX_SHIFT", min_value=0.0)
        _set_float_env("ja_vc_neighbor_lead_ms_var", "UTOA_JA_VC_NEIGHBOR_LEAD_MS", min_value=0.0)
        _set_float_env("ja_vc_neighbor_tail_ms_var", "UTOA_JA_VC_NEIGHBOR_TAIL_MS", min_value=0.0)
        _set_float_env("ja_vc_neighbor_min_len_var", "UTOA_JA_VC_NEIGHBOR_MIN_LEN", min_value=0.0)

        soft_bank_mode = (
            bool(self.soft_bank_mode_var.get())
            if hasattr(self, "soft_bank_mode_var")
            else False
        )
        if soft_bank_mode:
            os.environ["UTOA_SOFT_BANK_MODE"] = "1"
            os.environ["UTOA_MFA_SOFT_BANK_MODE"] = "1"
            # Low-level/noisy bank mode:
            # Relax onset/sound thresholds so weak voiced segments are less likely
            # to be treated as silence.
            os.environ["UTOA_MEL_SOUND_DB_MARGIN"] = "1.15"
            os.environ["UTOA_MEL_SOUND_ENERGY_MIN"] = "0.10"
            os.environ["UTOA_MEL_HARD_SILENCE_ENERGY_MAX"] = "0.11"
            os.environ["UTOA_MEL_SOFT_SILENCE_ENERGY_MAX"] = "0.14"
            os.environ["UTOA_MEL_WEAK_F2_MAX"] = "0.04"
            os.environ["UTOA_MEL_WEAK_F3_MAX"] = "0.04"
            os.environ["UTOA_MEL_WEAK_HIGH_MAX"] = "0.085"
            os.environ["UTOA_MEL_ONSET_SLOPE_BASE"] = "0.012"
            os.environ["UTOA_MEL_ONSET_ENERGY_BASE"] = "0.10"
            os.environ["UTOA_MEL_ONSET_DB_MARGIN_BASE"] = "-2.6"
            os.environ["UTOA_MEL_ONSET_ENERGY_SIBILANT"] = "0.095"
            os.environ["UTOA_MEL_ONSET_DB_MARGIN_SIBILANT"] = "-4.2"

        low_rms_gain_enabled = (
            bool(self.low_rms_gain_enable_var.get())
            if hasattr(self, "low_rms_gain_enable_var")
            else True
        )
        os.environ["UTOA_MFA_LOW_RMS_GAIN_ENABLE"] = "1" if low_rms_gain_enabled else "0"
        weak_voice_assist_enabled = (
            bool(self.weak_voice_assist_enable_var.get())
            if hasattr(self, "weak_voice_assist_enable_var")
            else True
        )
        os.environ["UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE"] = "1" if weak_voice_assist_enabled else "0"
        if weak_voice_assist_enabled:
            _set_float_env(
                "weak_voice_assist_strength_var",
                "UTOA_MFA_WEAK_VOICE_PREEMPH_MIX",
                min_value=0.0,
                max_value=1.0,
            )
        strict_mode_code = (
            self._get_mapping_strict_mode_code()
            if hasattr(self, "_get_mapping_strict_mode_code")
            else "soft"
        )
        os.environ["UTOA_SYLLABLE_STRICT_MODE"] = str(strict_mode_code or "soft").strip().lower()
        low_conf_force_lock_mode = (
            bool(self.low_conf_force_lock_mode_var.get())
            if hasattr(self, "low_conf_force_lock_mode_var")
            else False
        )
        os.environ["UTOA_LOW_CONF_FORCE_LOCK_MODE"] = "1" if low_conf_force_lock_mode else "0"
        weak_missing_reduce = (
            bool(self.weak_boundary_reduce_missing_var.get())
            if hasattr(self, "weak_boundary_reduce_missing_var")
            else False
        )
        weak_block_mismap = (
            bool(self.weak_boundary_block_mismap_var.get())
            if hasattr(self, "weak_boundary_block_mismap_var")
            else False
        )
        os.environ["UTOA_WEAK_BOUNDARY_MISSING_REDUCTION_ENABLE"] = "1" if weak_missing_reduce else "0"
        os.environ["UTOA_WEAK_BOUNDARY_ANTIMISMAP_ENABLE"] = "1" if weak_block_mismap else "0"

    def _sync_advanced_tuning_slider_controls(self) -> None:
        bindings = getattr(self, "advanced_tuning_slider_bindings", None)
        if not isinstance(bindings, dict):
            return
        for _, meta in bindings.items():
            if not isinstance(meta, dict):
                continue
            var = meta.get("var")
            dvar = meta.get("dvar")
            label = meta.get("label")
            fmt = str(meta.get("fmt", "{:.2f}") or "{:.2f}")
            try:
                default_val = float(meta.get("default", 0.0))
            except Exception:
                default_val = 0.0
            try:
                min_val = float(meta.get("min", default_val))
            except Exception:
                min_val = default_val
            try:
                max_val = float(meta.get("max", default_val))
            except Exception:
                max_val = default_val
            if max_val < min_val:
                min_val, max_val = max_val, min_val

            current = default_val
            if var is not None:
                try:
                    raw = str(var.get() or "").strip()
                except Exception:
                    raw = ""
                if raw:
                    try:
                        current = float(raw)
                    except Exception:
                        current = default_val
            current = max(min_val, min(max_val, current))

            if dvar is not None:
                try:
                    dvar.set(current)
                except Exception:
                    pass
            if label is not None:
                try:
                    label.configure(text=fmt.format(current))
                except Exception:
                    pass

    @staticmethod
    def _get_format_consistency_recommendation(language: str, format_code: str) -> dict:
        lang = str(language or "korean").strip().lower()
        fmt = str(format_code or "").strip().lower()
        if not fmt:
            fmt = "general"

        common_enable = {
            "vc_correction_enable_var": True,
            "kr_continuity_enable_var": True,
            "kr_vc_neighbor_enable_var": True,
            "ja_vc_neighbor_enable_var": True,
        }
        profiles = {
            "korean": {
                "general": {
                    "kr_continuity_max_offset_adj_var": "170",
                    "kr_vc_neighbor_blend_var": "0.32",
                    "kr_vc_neighbor_max_shift_var": "40",
                    "kr_vc_neighbor_lead_ms_var": "5",
                    "kr_vc_neighbor_tail_ms_var": "7",
                    "kr_vc_neighbor_min_len_var": "38",
                    "ja_vc_neighbor_blend_var": "0.35",
                    "ja_vc_neighbor_max_shift_var": "45",
                    "ja_vc_neighbor_lead_ms_var": "6",
                    "ja_vc_neighbor_tail_ms_var": "8",
                    "ja_vc_neighbor_min_len_var": "35",
                },
                "cv": {
                    "kr_continuity_max_offset_adj_var": "140",
                    "kr_vc_neighbor_blend_var": "0.24",
                    "kr_vc_neighbor_max_shift_var": "30",
                    "kr_vc_neighbor_lead_ms_var": "4",
                    "kr_vc_neighbor_tail_ms_var": "6",
                    "kr_vc_neighbor_min_len_var": "45",
                    "ja_vc_neighbor_blend_var": "0.28",
                    "ja_vc_neighbor_max_shift_var": "34",
                    "ja_vc_neighbor_lead_ms_var": "5",
                    "ja_vc_neighbor_tail_ms_var": "6",
                    "ja_vc_neighbor_min_len_var": "42",
                },
                "cvc": {
                    "kr_continuity_max_offset_adj_var": "175",
                    "kr_vc_neighbor_blend_var": "0.34",
                    "kr_vc_neighbor_max_shift_var": "42",
                    "kr_vc_neighbor_lead_ms_var": "6",
                    "kr_vc_neighbor_tail_ms_var": "8",
                    "kr_vc_neighbor_min_len_var": "34",
                    "ja_vc_neighbor_blend_var": "0.32",
                    "ja_vc_neighbor_max_shift_var": "40",
                    "ja_vc_neighbor_lead_ms_var": "5",
                    "ja_vc_neighbor_tail_ms_var": "7",
                    "ja_vc_neighbor_min_len_var": "36",
                },
                "cvvc": {
                    "kr_continuity_max_offset_adj_var": "200",
                    "kr_vc_neighbor_blend_var": "0.44",
                    "kr_vc_neighbor_max_shift_var": "58",
                    "kr_vc_neighbor_lead_ms_var": "8",
                    "kr_vc_neighbor_tail_ms_var": "10",
                    "kr_vc_neighbor_min_len_var": "26",
                    "ja_vc_neighbor_blend_var": "0.42",
                    "ja_vc_neighbor_max_shift_var": "56",
                    "ja_vc_neighbor_lead_ms_var": "7",
                    "ja_vc_neighbor_tail_ms_var": "9",
                    "ja_vc_neighbor_min_len_var": "28",
                },
                "vcv": {
                    "kr_continuity_max_offset_adj_var": "190",
                    "kr_vc_neighbor_blend_var": "0.38",
                    "kr_vc_neighbor_max_shift_var": "50",
                    "kr_vc_neighbor_lead_ms_var": "7",
                    "kr_vc_neighbor_tail_ms_var": "9",
                    "kr_vc_neighbor_min_len_var": "30",
                    "ja_vc_neighbor_blend_var": "0.34",
                    "ja_vc_neighbor_max_shift_var": "44",
                    "ja_vc_neighbor_lead_ms_var": "6",
                    "ja_vc_neighbor_tail_ms_var": "8",
                    "ja_vc_neighbor_min_len_var": "34",
                },
                "c_plus_v": {
                    "kr_continuity_max_offset_adj_var": "120",
                    "kr_vc_neighbor_blend_var": "0.18",
                    "kr_vc_neighbor_max_shift_var": "24",
                    "kr_vc_neighbor_lead_ms_var": "4",
                    "kr_vc_neighbor_tail_ms_var": "5",
                    "kr_vc_neighbor_min_len_var": "48",
                    "ja_vc_neighbor_blend_var": "0.22",
                    "ja_vc_neighbor_max_shift_var": "28",
                    "ja_vc_neighbor_lead_ms_var": "4",
                    "ja_vc_neighbor_tail_ms_var": "6",
                    "ja_vc_neighbor_min_len_var": "46",
                },
                "cmpx": {
                    "kr_continuity_max_offset_adj_var": "150",
                    "kr_vc_neighbor_blend_var": "0.25",
                    "kr_vc_neighbor_max_shift_var": "34",
                    "kr_vc_neighbor_lead_ms_var": "5",
                    "kr_vc_neighbor_tail_ms_var": "6",
                    "kr_vc_neighbor_min_len_var": "42",
                    "ja_vc_neighbor_blend_var": "0.28",
                    "ja_vc_neighbor_max_shift_var": "34",
                    "ja_vc_neighbor_lead_ms_var": "5",
                    "ja_vc_neighbor_tail_ms_var": "6",
                    "ja_vc_neighbor_min_len_var": "40",
                },
            },
            "japanese": {
                "general": {
                    "kr_continuity_max_offset_adj_var": "170",
                    "kr_vc_neighbor_blend_var": "0.32",
                    "kr_vc_neighbor_max_shift_var": "40",
                    "kr_vc_neighbor_lead_ms_var": "5",
                    "kr_vc_neighbor_tail_ms_var": "7",
                    "kr_vc_neighbor_min_len_var": "38",
                    "ja_vc_neighbor_blend_var": "0.30",
                    "ja_vc_neighbor_max_shift_var": "38",
                    "ja_vc_neighbor_lead_ms_var": "5",
                    "ja_vc_neighbor_tail_ms_var": "7",
                    "ja_vc_neighbor_min_len_var": "36",
                },
                "cv": {
                    "kr_continuity_max_offset_adj_var": "150",
                    "kr_vc_neighbor_blend_var": "0.26",
                    "kr_vc_neighbor_max_shift_var": "32",
                    "kr_vc_neighbor_lead_ms_var": "4",
                    "kr_vc_neighbor_tail_ms_var": "6",
                    "kr_vc_neighbor_min_len_var": "44",
                    "ja_vc_neighbor_blend_var": "0.24",
                    "ja_vc_neighbor_max_shift_var": "30",
                    "ja_vc_neighbor_lead_ms_var": "4",
                    "ja_vc_neighbor_tail_ms_var": "6",
                    "ja_vc_neighbor_min_len_var": "44",
                },
                "cvvc": {
                    "kr_continuity_max_offset_adj_var": "185",
                    "kr_vc_neighbor_blend_var": "0.36",
                    "kr_vc_neighbor_max_shift_var": "48",
                    "kr_vc_neighbor_lead_ms_var": "6",
                    "kr_vc_neighbor_tail_ms_var": "8",
                    "kr_vc_neighbor_min_len_var": "32",
                    "ja_vc_neighbor_blend_var": "0.42",
                    "ja_vc_neighbor_max_shift_var": "56",
                    "ja_vc_neighbor_lead_ms_var": "7",
                    "ja_vc_neighbor_tail_ms_var": "9",
                    "ja_vc_neighbor_min_len_var": "28",
                },
                "vcv": {
                    "kr_continuity_max_offset_adj_var": "170",
                    "kr_vc_neighbor_blend_var": "0.31",
                    "kr_vc_neighbor_max_shift_var": "40",
                    "kr_vc_neighbor_lead_ms_var": "5",
                    "kr_vc_neighbor_tail_ms_var": "7",
                    "kr_vc_neighbor_min_len_var": "36",
                    "ja_vc_neighbor_blend_var": "0.34",
                    "ja_vc_neighbor_max_shift_var": "44",
                    "ja_vc_neighbor_lead_ms_var": "6",
                    "ja_vc_neighbor_tail_ms_var": "8",
                    "ja_vc_neighbor_min_len_var": "34",
                },
            },
        }
        selected = profiles.get(lang, {}).get(fmt) or profiles.get(lang, {}).get("general")
        if not isinstance(selected, dict):
            return {}
        result = dict(common_enable)
        result.update(selected)
        return result

    def _apply_format_consistency_recommendation(self, *, save_config: bool = False, write_log: bool = False) -> None:
        if not hasattr(self, "_get_language") or not hasattr(self, "auto_format_var"):
            return
        language = self._get_language()
        if str(language or "").strip().lower() == "english":
            return
        try:
            format_code = normalize_auto_format_value(language, self.auto_format_var.get())
        except Exception:
            format_code = ""
        recommended = self._get_format_consistency_recommendation(language, format_code)
        if not recommended:
            return
        for attr, value in recommended.items():
            if not hasattr(self, attr):
                continue
            try:
                getattr(self, attr).set(value)
            except Exception:
                continue
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_advanced_tuning_slider_controls"):
            self._sync_advanced_tuning_slider_controls()
        if save_config:
            self._save_config()
        if write_log and hasattr(self, "_append_log"):
            fmt_txt = str(format_code or "auto").strip().lower() or "auto"
            self._append_log(f"[권장값] 형식별 연속성/파일 일관성 보정값 적용: lang={language}, fmt={fmt_txt}")

    def _sync_weak_voice_assist_controls(self) -> None:
        enabled = (
            bool(self.weak_voice_assist_enable_var.get())
            if hasattr(self, "weak_voice_assist_enable_var")
            else True
        )
        slider_state = "normal" if enabled else "disabled"
        title_color = PALETTE.neutral_text if enabled else PALETTE.hint_text
        value_color = PALETTE.neutral_text if enabled else PALETTE.hint_text
        if hasattr(self, "weak_voice_strength_slider"):
            try:
                self.weak_voice_strength_slider.configure(state=slider_state)
            except Exception:
                pass
        if hasattr(self, "weak_voice_strength_title_label"):
            try:
                self.weak_voice_strength_title_label.configure(text_color=title_color)
            except Exception:
                pass
        if hasattr(self, "weak_voice_strength_value_label"):
            try:
                self.weak_voice_strength_value_label.configure(text_color=value_color)
            except Exception:
                pass

    def _on_weak_voice_assist_toggle(self) -> None:
        self._sync_weak_voice_assist_controls()
        self._save_config()

    def _on_voicebank_preset_button(self, bank_type: str, aggressiveness: str) -> None:
        key = f"{str(bank_type or '').strip().lower()}:{str(aggressiveness or '').strip().lower()}"
        self._selected_voicebank_preset_key = key
        self._apply_voicebank_tuning_preset(bank_type, aggressiveness)
        self._sync_voicebank_preset_button_styles()

    def _sync_voicebank_preset_button_styles(self) -> None:
        buttons = getattr(self, "voicebank_preset_buttons", None)
        if not isinstance(buttons, dict):
            return
        selected_key = str(getattr(self, "_selected_voicebank_preset_key", "") or "").strip().lower()
        for key, btn in buttons.items():
            if btn is None:
                continue
            try:
                if str(key).strip().lower() == selected_key:
                    btn.configure(
                        fg_color=PALETTE.menu_button,
                        hover_color=PALETTE.menu_button_hover,
                        text_color=PALETTE.menu_text,
                    )
                else:
                    btn.configure(
                        fg_color=PALETTE.primary_button_bg,
                        hover_color=PALETTE.primary_button_hover,
                        text_color=PALETTE.primary_button_text,
                    )
            except Exception:
                continue

    def _apply_voicebank_tuning_preset(self, bank_type: str, aggressiveness: str) -> None:
        bank = str(bank_type or "").strip().lower()
        mode = str(aggressiveness or "").strip().lower()
        presets = {
            ("soft", "aggressive"): {
                "label": "soft voice - aggressive",
                "values": {
                    "kr_continuity_enable_var": True,
                    "kr_continuity_max_offset_adj_var": "240",
                    "vc_correction_enable_var": True,
                    "kr_vc_neighbor_enable_var": True,
                    "ja_vc_neighbor_enable_var": True,
                    "kr_vc_neighbor_blend_var": "0.72",
                    "kr_vc_neighbor_max_shift_var": "85",
                    "kr_vc_neighbor_lead_ms_var": "12",
                    "kr_vc_neighbor_tail_ms_var": "14",
                    "kr_vc_neighbor_min_len_var": "20",
                    "ja_vc_neighbor_blend_var": "0.72",
                    "ja_vc_neighbor_max_shift_var": "85",
                    "ja_vc_neighbor_lead_ms_var": "12",
                    "ja_vc_neighbor_tail_ms_var": "14",
                    "ja_vc_neighbor_min_len_var": "20",
                    "ml_anchor_mel_gamma_var": "0.95",
                    "soft_bank_mode_var": True,
                },
            },
            ("soft", "conservative"): {
                "label": "soft voice - conservative",
                "values": {
                    "kr_continuity_enable_var": True,
                    "kr_continuity_max_offset_adj_var": "170",
                    "vc_correction_enable_var": True,
                    "kr_vc_neighbor_enable_var": True,
                    "ja_vc_neighbor_enable_var": True,
                    "kr_vc_neighbor_blend_var": "0.42",
                    "kr_vc_neighbor_max_shift_var": "55",
                    "kr_vc_neighbor_lead_ms_var": "7",
                    "kr_vc_neighbor_tail_ms_var": "9",
                    "kr_vc_neighbor_min_len_var": "30",
                    "ja_vc_neighbor_blend_var": "0.42",
                    "ja_vc_neighbor_max_shift_var": "55",
                    "ja_vc_neighbor_lead_ms_var": "7",
                    "ja_vc_neighbor_tail_ms_var": "9",
                    "ja_vc_neighbor_min_len_var": "30",
                    "ml_anchor_mel_gamma_var": "1.28",
                    "soft_bank_mode_var": True,
                },
            },
            ("normal", "aggressive"): {
                "label": "normal voice - aggressive",
                "values": {
                    "kr_continuity_enable_var": True,
                    "kr_continuity_max_offset_adj_var": "210",
                    "vc_correction_enable_var": True,
                    "kr_vc_neighbor_enable_var": True,
                    "ja_vc_neighbor_enable_var": True,
                    "kr_vc_neighbor_blend_var": "0.58",
                    "kr_vc_neighbor_max_shift_var": "68",
                    "kr_vc_neighbor_lead_ms_var": "8",
                    "kr_vc_neighbor_tail_ms_var": "10",
                    "kr_vc_neighbor_min_len_var": "26",
                    "ja_vc_neighbor_blend_var": "0.58",
                    "ja_vc_neighbor_max_shift_var": "68",
                    "ja_vc_neighbor_lead_ms_var": "8",
                    "ja_vc_neighbor_tail_ms_var": "10",
                    "ja_vc_neighbor_min_len_var": "26",
                    "ml_anchor_mel_gamma_var": "1.05",
                    "soft_bank_mode_var": False,
                },
            },
            ("normal", "conservative"): {
                "label": "normal voice - conservative",
                "values": {
                    "kr_continuity_enable_var": True,
                    "kr_continuity_max_offset_adj_var": "150",
                    "vc_correction_enable_var": True,
                    "kr_vc_neighbor_enable_var": True,
                    "ja_vc_neighbor_enable_var": True,
                    "kr_vc_neighbor_blend_var": "0.30",
                    "kr_vc_neighbor_max_shift_var": "40",
                    "kr_vc_neighbor_lead_ms_var": "5",
                    "kr_vc_neighbor_tail_ms_var": "7",
                    "kr_vc_neighbor_min_len_var": "40",
                    "ja_vc_neighbor_blend_var": "0.30",
                    "ja_vc_neighbor_max_shift_var": "40",
                    "ja_vc_neighbor_lead_ms_var": "5",
                    "ja_vc_neighbor_tail_ms_var": "7",
                    "ja_vc_neighbor_min_len_var": "40",
                    "ml_anchor_mel_gamma_var": "1.25",
                    "soft_bank_mode_var": False,
                },
            },
        }
        selected = presets.get((bank, mode))
        if not selected:
            return
        values = selected.get("values", {})
        for attr, value in values.items():
            if not hasattr(self, attr):
                continue
            try:
                getattr(self, attr).set(value)
            except Exception:
                continue
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_cvn_correction_toggle"):
            self._sync_cvn_correction_toggle()
        if hasattr(self, "_sync_advanced_tuning_slider_controls"):
            self._sync_advanced_tuning_slider_controls()
        if hasattr(self, "_sync_weak_voice_assist_controls"):
            self._sync_weak_voice_assist_controls()
        self._save_config()
        if hasattr(self, "_append_log"):
            self._append_log(f"[프리셋] 적용 완료: {selected.get('label', '')}")

    def _reset_developer_settings_defaults(self) -> None:
        defaults = {
            "developer_mode_enabled_var": False,
            "aligner_var": "MFA",
            "no_mfa_oto_mode_var": "베이스 OTO 재매핑 + 보정",
            "oto_crnn_model_path_var": "",
            "oto_crnn_engine_var": "boundary_decoder",
            "oto_crnn_device_var": "auto",
            "oto_crnn_special_aliases_var": "",
            "enable_ml_correction_var": True,
            "ml_route_var": "자동(자동 라우팅)",
            "ml_selector_mode_var": "+셀렉터",
            "ml_coupled_enable_var": True,
            "ml_coupled_min_conf_var": "",
            "ml_coupled_min_conf_use_model_meta_var": True,
            "ml_coupled_min_conf_model_offset_var": "",
            "ml_coupled_min_conf_kr_cv_var": "",
            "ml_coupled_min_conf_kr_cvc_var": "",
            "ml_coupled_min_conf_kr_cvvc_var": "",
            "ml_coupled_min_conf_kr_vcv_var": "",
            "ml_coupled_min_conf_ja_cv_var": "",
            "ml_coupled_min_conf_ja_cvc_var": "",
            "ml_coupled_min_conf_ja_cvvc_var": "",
            "ml_coupled_min_conf_ja_vcv_var": "",
            "ml_coupled_device_var": "auto",
            "ml_coupled_backend_var": "auto",
            "ml_coupled_strict_constraint_var": True,
            "ml_batch_inference_enable_var": True,
            "ml_batch_inference_size_var": "256",
            "ml_legacy_fallback_enable_var": True,
            "ml_hybrid_routing_enable_var": True,
            "ml_e2e_enable_var": False,
            "ml_e2e_mode_var": "hybrid",
            "ml_e2e_t_low_var": "",
            "ml_e2e_t_high_var": "",
            "ml_e2e_blend_alpha_var": "",
            "ml_anchor_mel_gamma_var": "1.2",
            "kr_mapping_confidence_threshold_var": "",
            "kr_continuity_enable_var": True,
            "kr_continuity_max_offset_adj_var": "",
            "vc_correction_enable_var": True,
            "cvn_correction_enable_var": True,
            "cvn_low_conf_only_var": False,
            "mapping_supervised_enable_var": True,
            "mapping_supervised_mode_var": "자동(권장)",
            "cv_order_prior_enable_var": True,
            "cv_order_prior_strength_var": "",
            "kr_vc_neighbor_enable_var": True,
            "kr_vc_neighbor_blend_var": "",
            "kr_vc_neighbor_max_shift_var": "",
            "kr_vc_neighbor_lead_ms_var": "",
            "kr_vc_neighbor_tail_ms_var": "",
            "kr_vc_neighbor_min_len_var": "",
            "ja_vc_neighbor_enable_var": True,
            "ja_vc_neighbor_blend_var": "",
            "ja_vc_neighbor_max_shift_var": "",
            "ja_vc_neighbor_lead_ms_var": "",
            "ja_vc_neighbor_tail_ms_var": "",
            "ja_vc_neighbor_min_len_var": "",
            "soft_bank_mode_var": False,
            "weak_voice_assist_enable_var": True,
            "weak_voice_assist_strength_var": "",
            "mapping_strict_mode_var": "적당히 엄격(누락 행은 폴백)",
            "low_conf_force_lock_mode_var": False,
            "weak_boundary_reduce_missing_var": False,
            "weak_boundary_block_mismap_var": False,
        }
        for attr, value in defaults.items():
            if not hasattr(self, attr):
                continue
            try:
                getattr(self, attr).set(value)
            except Exception:
                continue

        if hasattr(self, "_apply_recommended_ml_model_defaults"):
            self._apply_recommended_ml_model_defaults()
        self._clear_ml_override_envs()
        self._clear_advanced_tuning_envs()
        self._save_config()
        if hasattr(self, "_sync_vc_correction_toggle"):
            self._sync_vc_correction_toggle()
        if hasattr(self, "_sync_cvn_correction_toggle"):
            self._sync_cvn_correction_toggle()
        if hasattr(self, "_sync_advanced_tuning_slider_controls"):
            self._sync_advanced_tuning_slider_controls()
        if hasattr(self, "_sync_weak_voice_assist_controls"):
            self._sync_weak_voice_assist_controls()
        if hasattr(self, "_sync_developer_mode_ui"):
            self._sync_developer_mode_ui()
        if hasattr(self, "_refresh_ml_backend_status"):
            self._refresh_ml_backend_status()
        self._append_log("개발자 설정을 기본값으로 초기화했습니다.")

    def _reset_all_settings_defaults(self):
        ask_fn = getattr(self, "_ask_yes_no_dialog_sync", None)
        if callable(ask_fn):
            confirm = bool(
                ask_fn(
                    "전체 초기화",
                    "모든 설정을 기본값으로 초기화할까요?",
                    default=False,
                )
            )
        else:
            confirm = bool(
                messagebox.askyesno(
                    "전체 초기화",
                    "모든 설정을 기본값으로 초기화할까요?",
                )
            )
        if not confirm:
            return False

        keep_paths = True
        if callable(ask_fn):
            keep_paths = bool(
                ask_fn(
                    "경로 유지",
                    "현재 경로(wav/base oto/output oto)를 유지할까요?",
                    default=True,
                )
            )
        else:
            keep_paths = bool(
                messagebox.askyesno(
                    "경로 유지",
                    "현재 경로(wav/base oto/output oto)를 유지할까요?",
                    default="yes",
                )
            )

        if hasattr(self, "_reset_developer_settings_defaults"):
            try:
                self._reset_developer_settings_defaults()
            except Exception:
                pass

        if hasattr(self, "openutau_var"):
            try:
                self.openutau_var.set(False)
            except Exception:
                pass

        if not keep_paths:
            for entry_name in ("wav_entry", "tpl_entry", "out_entry"):
                entry = getattr(self, entry_name, None)
                if entry is None:
                    continue
                try:
                    entry.configure(state="normal")
                except Exception:
                    pass
                try:
                    entry.delete(0, "end")
                except Exception:
                    pass

        if hasattr(self, "_save_config"):
            try:
                self._save_config()
            except Exception:
                pass
        if hasattr(self, "_append_log"):
            try:
                if keep_paths:
                    self._append_log("전체 설정을 기본값으로 초기화했습니다. 경로는 유지했습니다.")
                else:
                    self._append_log("전체 설정을 기본값으로 초기화했습니다. 경로를 함께 초기화했습니다.")
            except Exception:
                pass
        return True

    @staticmethod
    def _normalize_mapping_strict_mode_code(value: str) -> str:
        raw = str(value or "").strip().lower()
        compact = raw.replace(" ", "")
        if raw in {
            "strict",
            "full",
            "full_strict",
            "hard",
        } or compact in {"완전엄격", "완전엄격모드"}:
            return "strict"
        if raw in {
            "off",
            "none",
            "disable",
            "disabled",
            "0",
            "false",
        } or compact in {"끄기", "비활성", "미사용"}:
            return "off"
        if raw in {
            "soft",
            "moderate",
            "balanced",
        } or compact in {"적당히엄격", "적당히엄격모드", "적당히엄격(누락행은폴백)"}:
            return "soft"
        if "완전엄격" in compact:
            return "strict"
        if "적당히엄격" in compact:
            return "soft"
        return "soft"

    @staticmethod
    def _normalize_mapping_supervised_mode_code(value: str) -> str:
        raw = str(value or "").strip().lower()
        compact = raw.replace(" ", "").replace("_", "").replace("-", "")
        if raw in {"global_first", "global-first", "global", "전역우선"} or "전역우선" in compact:
            return "global_first"
        if raw in {"format_first", "format-first", "format", "포맷우선"} or "포맷우선" in compact:
            return "format_first"
        return "auto"

    @staticmethod
    def _mapping_supervised_mode_label_from_code(code: str) -> str:
        normalized = AppRuntimeMixin._normalize_mapping_supervised_mode_code(code)
        table = {
            "auto": "자동(권장)",
            "format_first": "포맷 우선",
            "global_first": "전역 우선",
        }
        return table.get(normalized, "자동(권장)")

    def _set_mapping_supervised_mode_from_code(self, code: str) -> str:
        normalized = self._normalize_mapping_supervised_mode_code(code)
        label = self._mapping_supervised_mode_label_from_code(normalized)
        if hasattr(self, "mapping_supervised_mode_var"):
            try:
                self.mapping_supervised_mode_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_mapping_supervised_mode_code(self) -> str:
        if not hasattr(self, "mapping_supervised_mode_var"):
            return "auto"
        try:
            raw = self.mapping_supervised_mode_var.get()
        except Exception:
            return "auto"
        normalized = self._normalize_mapping_supervised_mode_code(raw)
        normalized_label = self._mapping_supervised_mode_label_from_code(normalized)
        if str(raw or "").strip() != normalized_label:
            try:
                self.mapping_supervised_mode_var.set(normalized_label)
            except Exception:
                pass
        return normalized

    def _get_mapping_supervised_mode_option_labels(self) -> list[str]:
        return [
            self._mapping_supervised_mode_label_from_code("auto"),
            self._mapping_supervised_mode_label_from_code("format_first"),
            self._mapping_supervised_mode_label_from_code("global_first"),
        ]

    @staticmethod
    def _mapping_strict_mode_label_from_code(code: str) -> str:
        normalized = AppRuntimeMixin._normalize_mapping_strict_mode_code(code)
        label_map = {
            "strict": "완전 엄격",
            "soft": "적당히 엄격(누락 행은 폴백)",
            "off": "off",
        }
        return label_map.get(normalized, "적당히 엄격(누락 행은 폴백)")

    def _set_mapping_strict_mode_from_code(self, code: str) -> str:
        normalized = self._normalize_mapping_strict_mode_code(code)
        label = self._mapping_strict_mode_label_from_code(normalized)
        if hasattr(self, "mapping_strict_mode_var"):
            try:
                self.mapping_strict_mode_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_mapping_strict_mode_code(self) -> str:
        if not hasattr(self, "mapping_strict_mode_var"):
            return "soft"
        try:
            raw = self.mapping_strict_mode_var.get()
        except Exception:
            raw = ""
        normalized = self._normalize_mapping_strict_mode_code(raw)
        normalized_label = self._mapping_strict_mode_label_from_code(normalized)
        try:
            if str(raw or "").strip() != normalized_label:
                self.mapping_strict_mode_var.set(normalized_label)
        except Exception:
            pass
        return normalized

    def _get_mapping_strict_mode_option_labels(self) -> list[str]:
        return [
            self._mapping_strict_mode_label_from_code("strict"),
            self._mapping_strict_mode_label_from_code("soft"),
            self._mapping_strict_mode_label_from_code("off"),
        ]

    @staticmethod
    def _normalize_ml_route_code(value: str) -> str:
        raw = str(value or "").strip().lower()
        if "e2e" in raw and ("hybrid" in raw or "하이브리드" in raw):
            return "e2e_hybrid"
        if raw in {
            "",
            "auto",
            "automatic",
            "policy",
        }:
            return "auto"
        if raw in {
            "no-mfa",
            "no_mfa",
            "nomfa",
            "no mfa",
            "autofree",
            "autofree_v1",
            "route_b",
            "b",
        }:
            return "nomfa"
        if raw in {"legacy", "v1", "route_a", "a"}:
            return "v1"
        if raw in {"v2", "route_v2"}:
            return "v2"
        if raw in {
            "e2e",
            "e2e_hybrid",
            "hybrid",
            "hybrid_e2e",
            "oto_e2e",
            "route_e2e",
            "route_c",
            "c",
        }:
            return "e2e_hybrid"
        return "auto"

    @staticmethod
    def _ml_route_label_from_code(code: str) -> str:
        normalized = AppRuntimeMixin._normalize_ml_route_code(code)
        label_map = {
            "auto": "auto",
            "nomfa": "No-MFA",
            "v1": "v1",
            "v2": "v2",
            "e2e_hybrid": "E2E 하이브리드(실험)",
        }
        return label_map.get(normalized, "auto")

    def _set_ml_route_from_code(self, code: str) -> str:
        normalized = self._normalize_ml_route_code(code)
        label = self._ml_route_label_from_code(normalized)
        if hasattr(self, "ml_route_var"):
            try:
                self.ml_route_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_ml_route_code(self) -> str:
        if not hasattr(self, "ml_route_var"):
            return "auto"
        try:
            raw = self.ml_route_var.get()
        except Exception:
            raw = ""
        normalized = self._normalize_ml_route_code(raw)
        normalized_label = self._ml_route_label_from_code(normalized)
        try:
            if str(raw or "").strip() != normalized_label:
                self.ml_route_var.set(normalized_label)
        except Exception:
            pass
        return normalized

    def _get_ml_route_option_labels(self, *, no_mfa_only: bool = False) -> list[str]:
        codes = ["auto", "nomfa"] if no_mfa_only else ["auto", "nomfa", "v1", "v2", "e2e_hybrid"]
        return [self._ml_route_label_from_code(code) for code in codes]

    @staticmethod
    def _map_ml_route_code_to_internal(route_code: str) -> str:
        normalized = AppRuntimeMixin._normalize_ml_route_code(route_code)
        if normalized == "e2e_hybrid":
            return "e2e_hybrid"
        if normalized in {"nomfa", "v2"}:
            return "autofree_v1"
        return "legacy"

    @staticmethod
    def _normalize_e2e_mode_code(value: str) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"legacy_only", "legacy", "legacy-only"}:
            return "legacy_only"
        if raw in {"e2e_only", "e2e", "e2e-only"}:
            return "e2e_only"
        return "hybrid"

    @staticmethod
    def _e2e_mode_label_from_code(code: str) -> str:
        normalized = AppRuntimeMixin._normalize_e2e_mode_code(code)
        label_map = {
            "hybrid": "hybrid",
            "legacy_only": "legacy_only",
            "e2e_only": "e2e_only",
        }
        return label_map.get(normalized, "hybrid")

    def _set_e2e_mode_from_code(self, code: str) -> str:
        normalized = self._normalize_e2e_mode_code(code)
        label = self._e2e_mode_label_from_code(normalized)
        if hasattr(self, "ml_e2e_mode_var"):
            try:
                self.ml_e2e_mode_var.set(label)
            except Exception:
                pass
        return normalized

    def _get_e2e_mode_code(self) -> str:
        if not hasattr(self, "ml_e2e_mode_var"):
            return "hybrid"
        try:
            raw = self.ml_e2e_mode_var.get()
        except Exception:
            raw = ""
        normalized = self._normalize_e2e_mode_code(raw)
        normalized_label = self._e2e_mode_label_from_code(normalized)
        try:
            if str(raw or "").strip() != normalized_label:
                self.ml_e2e_mode_var.set(normalized_label)
        except Exception:
            pass
        return normalized

    def _apply_auto_ml_policy_env(
        self,
        language: str,
        auto_format_value: str,
        *,
        enable_ml_default: bool = True,
        gen_dash_alias: bool = True,
    ) -> dict:
        lang = str(language or "korean").strip().lower()
        fmt = self._resolve_auto_format_for_policy(lang, auto_format_value)
        route_selected = self._get_ml_route_code() if hasattr(self, "_get_ml_route_code") else "auto"
        e2e_mode = self._get_e2e_mode_code() if hasattr(self, "_get_e2e_mode_code") else "hybrid"
        e2e_toggle_enabled = (
            bool(self.ml_e2e_enable_var.get())
            if hasattr(self, "ml_e2e_enable_var")
            else False
        )
        developer_mode_enabled = (
            bool(self.developer_mode_enabled_var.get())
            if hasattr(self, "developer_mode_enabled_var")
            else False
        )
        e2e_requested = bool(e2e_toggle_enabled and developer_mode_enabled)

        def _clamp01_text(value: object, default: float) -> str:
            try:
                numeric = float(str(value).strip())
            except Exception:
                numeric = float(default)
            numeric = max(0.0, min(1.0, numeric))
            return f"{numeric:.4f}".rstrip("0").rstrip(".")

        e2e_t_low = _clamp01_text(
            self.ml_e2e_t_low_var.get() if hasattr(self, "ml_e2e_t_low_var") else "",
            0.52,
        )
        e2e_t_high = _clamp01_text(
            self.ml_e2e_t_high_var.get() if hasattr(self, "ml_e2e_t_high_var") else "",
            0.72,
        )
        e2e_blend_alpha = _clamp01_text(
            self.ml_e2e_blend_alpha_var.get() if hasattr(self, "ml_e2e_blend_alpha_var") else "",
            0.60,
        )

        selector_mode = (
            self.ml_selector_mode_var.get()
            if hasattr(self, "ml_selector_mode_var")
            else "+셀렉터"
        )

        if lang not in {"korean", "japanese"}:
            for key in (
                "UTOA_KR_OTO_ML_DIR",
                "UTOA_KR_AUTOFREE_ML_DIR",
                "UTOA_JA_OTO_ML_DIR",
                "UTOA_JA_AUTOFREE_ML_DIR",
            ):
                os.environ.pop(key, None)
            if hasattr(self, "_clear_e2e_model_envs"):
                self._clear_e2e_model_envs()

            self._apply_ml_selector_runtime_mode("policy")
            self._clear_ml_override_envs()

            os.environ["UTOA_ML_COUPLED_ENABLE"] = "0"
            os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "0"
            os.environ["UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META"] = "1"
            os.environ["UTOA_ML_COUPLED_DEVICE"] = "auto"
            os.environ["UTOA_ML_COUPLED_BACKEND"] = "auto"
            os.environ["UTOA_ML_COUPLED_ONNX_ENABLE"] = "1"
            os.environ["UTOA_ML_BATCH_INFERENCE_ENABLE"] = "0"
            os.environ["UTOA_ML_BATCH_INFERENCE_SIZE"] = "256"
            os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "0"
            os.environ["UTOA_ML_GATED_ENSEMBLE_ENABLE"] = "0"
            os.environ.setdefault("UTOA_MEL_WEIGHT_MODE", "mel_first")
            os.environ["UTOA_ML_ROUTE"] = "disabled"
            os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "0"
            os.environ["UTOA_ENABLE_HEAD_TAIL_DASH_ALIAS"] = "1" if bool(gen_dash_alias) else "0"
            os.environ["UTOA_DISABLE_OTO_ML"] = "1"
            os.environ["UTOA_E2E_ENABLE"] = "0"
            os.environ["UTOA_E2E_MODE"] = "legacy_only"
            os.environ["UTOA_E2E_T_LOW"] = e2e_t_low
            os.environ["UTOA_E2E_T_HIGH"] = e2e_t_high
            os.environ["UTOA_E2E_BLEND_ALPHA"] = e2e_blend_alpha

            return {
                "language": lang,
                "format_type": fmt,
                "model_root": "",
                "enable_ml": False,
                "route": "disabled",
                "route_selected": route_selected,
                "route_internal": "disabled",
                "has_coupled": False,
                "has_ensemble": False,
                "has_autofree": False,
            }

        selector_mode = (
            self.ml_selector_mode_var.get()
            if hasattr(self, "ml_selector_mode_var")
            else "+셀렉터"
        )

        if lang not in {"korean", "japanese"}:
            for key in (
                "UTOA_KR_OTO_ML_DIR",
                "UTOA_KR_AUTOFREE_ML_DIR",
                "UTOA_JA_OTO_ML_DIR",
                "UTOA_JA_AUTOFREE_ML_DIR",
            ):
                os.environ.pop(key, None)
            if hasattr(self, "_clear_e2e_model_envs"):
                self._clear_e2e_model_envs()

            self._apply_ml_selector_runtime_mode("policy")
            self._clear_ml_override_envs()

            os.environ["UTOA_ML_COUPLED_ENABLE"] = "0"
            os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "0"
            os.environ["UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META"] = "1"
            os.environ["UTOA_ML_COUPLED_DEVICE"] = "auto"
            os.environ["UTOA_ML_COUPLED_BACKEND"] = "auto"
            os.environ["UTOA_ML_BATCH_INFERENCE_ENABLE"] = "0"
            os.environ["UTOA_ML_BATCH_INFERENCE_SIZE"] = "256"
            os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "0"
            os.environ["UTOA_ML_GATED_ENSEMBLE_ENABLE"] = "0"
            os.environ.setdefault("UTOA_MEL_WEIGHT_MODE", "mel_first")
            os.environ["UTOA_ML_ROUTE"] = "disabled"
            os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "0"
            os.environ["UTOA_ENABLE_HEAD_TAIL_DASH_ALIAS"] = "1" if bool(gen_dash_alias) else "0"
            os.environ["UTOA_DISABLE_OTO_ML"] = "1"
            os.environ["UTOA_E2E_ENABLE"] = "0"
            os.environ["UTOA_E2E_MODE"] = "legacy_only"
            os.environ["UTOA_E2E_T_LOW"] = e2e_t_low
            os.environ["UTOA_E2E_T_HIGH"] = e2e_t_high
            os.environ["UTOA_E2E_BLEND_ALPHA"] = e2e_blend_alpha

            return {
                "language": lang,
                "format_type": fmt,
                "model_root": "",
                "enable_ml": False,
                "route": "disabled",
                "route_selected": route_selected,
                "route_internal": "disabled",
                "has_coupled": False,
                "has_ensemble": False,
                "has_autofree": False,
            }

        model_root = ""
        if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
            model_root = str(self.ml_model_root_ja_var.get() or "").strip()
        elif lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
            model_root = str(self.ml_model_root_kr_var.get() or "").strip()
        if not model_root and hasattr(self, "_recommended_ml_model_root"):
            try:
                model_root = str(self._recommended_ml_model_root(lang) or "").strip()
            except Exception:
                model_root = ""

        if lang == "japanese":
            if model_root:
                os.environ["UTOA_JA_OTO_ML_DIR"] = model_root
                os.environ["UTOA_JA_AUTOFREE_ML_DIR"] = model_root
            else:
                os.environ.pop("UTOA_JA_OTO_ML_DIR", None)
                os.environ.pop("UTOA_JA_AUTOFREE_ML_DIR", None)
        else:
            if model_root:
                os.environ["UTOA_KR_OTO_ML_DIR"] = model_root
                os.environ["UTOA_KR_AUTOFREE_ML_DIR"] = model_root
            else:
                os.environ.pop("UTOA_KR_OTO_ML_DIR", None)
                os.environ.pop("UTOA_KR_AUTOFREE_ML_DIR", None)

        e2e_env_info = self._apply_e2e_model_env(lang) if hasattr(self, "_apply_e2e_model_env") else {
            "model_root": "",
            "alias_root": "",
            "has_models": False,
        }
        e2e_model_root = str(e2e_env_info.get("model_root", "") or "")
        e2e_alias_root = str(e2e_env_info.get("alias_root", "") or "")

        has_ensemble = False
        has_coupled = False
        try:
            from core.oto_ml_refiner import _resolve_ensemble_model_dir, _resolve_lightgbm_model_dir, _resolve_model_dir

            ensemble_dir = _resolve_ensemble_model_dir(lang, fmt)
            lightgbm_dir = _resolve_lightgbm_model_dir(lang, fmt)
            primary_dir = _resolve_model_dir(lang, fmt)
            has_ensemble = bool(ensemble_dir)
            has_coupled = bool(primary_dir or ensemble_dir or lightgbm_dir)
        except Exception:
            has_ensemble = False
            has_coupled = False

        has_autofree = False
        try:
            from core.oto_ml_autofree_runtime import _resolve_autofree_model_dir

            has_autofree = bool(_resolve_autofree_model_dir(lang, fmt))
        except Exception:
            has_autofree = False

        has_e2e_model = bool(e2e_env_info.get("has_models", False))
        e2e_model_dir = ""
        try:
            from core.oto_e2e_runtime import resolve_e2e_alias_model_root, resolve_e2e_model_dir

            e2e_model_dir = str(resolve_e2e_model_dir(lang, fmt) or "").strip()
            if not e2e_alias_root:
                e2e_alias_root = str(resolve_e2e_alias_model_root(lang, fmt) or "").strip()
            has_e2e_model = bool(e2e_model_dir) or bool(e2e_alias_root and self._dir_has_model_meta(e2e_alias_root))
        except Exception:
            has_e2e_model = bool(has_e2e_model)

        hybrid_routing_enabled = (
            bool(self.ml_hybrid_routing_enable_var.get())
            if hasattr(self, "ml_hybrid_routing_enable_var")
            else True
        )
        aligner_raw = (
            str(self.aligner_var.get() if hasattr(self, "aligner_var") else "MFA")
            .strip()
            .lower()
        )
        aligner_no_mfa = aligner_raw in {"no-mfa", "no_mfa", "none", "none-mfa", "nomfa", "no mfa"}
        no_mfa_generation_mode = bool(
            aligner_no_mfa
            and lang in {"korean", "japanese"}
            and not (lang == "korean" and fmt == "cmpx")
        )
        if no_mfa_generation_mode and route_selected in {"v1", "v2", "e2e_hybrid"}:
            route_selected = "auto"
            if hasattr(self, "_set_ml_route_from_code"):
                self._set_ml_route_from_code("auto")

        coupled_enabled = (
            bool(self.ml_coupled_enable_var.get())
            if hasattr(self, "ml_coupled_enable_var")
            else True
        )
        enable_ml = bool(enable_ml_default and has_coupled and coupled_enabled)
        e2e_route_allowed = bool(enable_ml and e2e_requested and fmt == "cv" and has_e2e_model)
        if route_selected == "auto":
            if e2e_route_allowed:
                route = "e2e_hybrid"
            elif no_mfa_generation_mode and has_autofree:
                route = "nomfa"
            elif has_autofree:
                route = "v2"
            else:
                route = "v1"
        elif route_selected == "nomfa":
            route = "nomfa" if has_autofree else "v1"
        elif route_selected == "v2":
            route = "v2" if has_autofree else "v1"
        elif route_selected == "e2e_hybrid":
            route = "e2e_hybrid" if e2e_route_allowed else ("v2" if has_autofree else "v1")
        else:
            route = "v1"
        route_internal = self._map_ml_route_code_to_internal(route)
        e2e_runtime_enabled = bool(route_internal == "e2e_hybrid" and e2e_route_allowed)
        aux_enabled = bool(enable_ml and route_internal == "autofree_v1" and has_autofree)

        self._apply_ml_selector_runtime_mode(selector_mode)
        self._clear_ml_override_envs()
        if hasattr(self, "_apply_e2e_model_env"):
            self._apply_e2e_model_env(lang)

        os.environ["UTOA_ML_COUPLED_ENABLE"] = "1" if enable_ml else "0"
        os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "1" if (enable_ml and has_ensemble) else "0"
        os.environ["UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META"] = "1"
        os.environ["UTOA_ML_COUPLED_DEVICE"] = "auto"
        os.environ["UTOA_ML_COUPLED_BACKEND"] = "auto"
        os.environ["UTOA_ML_COUPLED_ONNX_ENABLE"] = "1"
        os.environ["UTOA_ML_BATCH_INFERENCE_ENABLE"] = "1" if enable_ml else "0"
        os.environ["UTOA_ML_BATCH_INFERENCE_SIZE"] = "256"
        legacy_fallback_enabled = (
            bool(self.ml_legacy_fallback_enable_var.get())
            if hasattr(self, "ml_legacy_fallback_enable_var")
            else True
        )
        os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = (
            "1" if (enable_ml and legacy_fallback_enabled) else "0"
        )
        os.environ["UTOA_ML_GATED_ENSEMBLE_ENABLE"] = "1" if (enable_ml and hybrid_routing_enabled) else "0"
        os.environ.setdefault("UTOA_MEL_WEIGHT_MODE", "mel_first")
        os.environ["UTOA_ML_ROUTE"] = route_internal
        os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "1" if aux_enabled else "0"
        os.environ["UTOA_E2E_ENABLE"] = "1" if e2e_runtime_enabled else "0"
        os.environ["UTOA_E2E_MODE"] = e2e_mode if e2e_runtime_enabled else "legacy_only"
        os.environ["UTOA_E2E_T_LOW"] = e2e_t_low
        os.environ["UTOA_E2E_T_HIGH"] = e2e_t_high
        os.environ["UTOA_E2E_BLEND_ALPHA"] = e2e_blend_alpha
        os.environ["UTOA_ENABLE_HEAD_TAIL_DASH_ALIAS"] = "1" if bool(gen_dash_alias) else "0"
        if enable_ml:
            os.environ.pop("UTOA_DISABLE_OTO_ML", None)
        else:
            os.environ["UTOA_DISABLE_OTO_ML"] = "1"

        return {
            "language": lang,
            "format_type": fmt,
            "model_root": model_root,
            "e2e_model_root": e2e_model_root,
            "e2e_model_dir": e2e_model_dir,
            "enable_ml": enable_ml,
            "route": route,
            "route_selected": route_selected,
            "route_internal": route_internal,
            "has_coupled": has_coupled,
            "has_ensemble": has_ensemble,
            "has_autofree": has_autofree,
            "has_e2e_model": has_e2e_model,
            "hybrid_routing": bool(enable_ml and hybrid_routing_enabled),
            "e2e_enabled": bool(e2e_runtime_enabled),
            "e2e_mode": str(e2e_mode),
        }

    def _after_safe(self, callback, delay_ms=0):
        if self._is_closing:
            return
        try:
            self.after(delay_ms, callback)
        except tk.TclError:
            pass

    def _compute_adaptive_widget_scale(self, width: int, height: int) -> float:
        base_w = 1280.0
        base_h = 760.0
        try:
            w = max(1.0, float(width))
            h = max(1.0, float(height))
        except Exception:
            return 1.0
        raw = min(w / base_w, h / base_h)
        if raw >= 1.08:
            target = 1.06
        elif raw >= 1.0:
            target = 1.0 + ((raw - 1.0) * 0.5)
        elif raw >= 0.92:
            target = 0.94 + ((raw - 0.92) * 0.75)
        else:
            target = max(0.90, raw * 0.98)
        if raw < 1.0:
            target *= 1.15
        return max(0.90, min(1.08, float(target)))

    def _update_responsive_bottom_layout(self, width: int | None = None):
        bottom = getattr(self, "bottom_bar_frame", None)
        status_group = getattr(self, "bottom_status_group", None)
        actions_group = getattr(self, "bottom_actions_group", None)
        if bottom is None or status_group is None or actions_group is None:
            return
        try:
            ui_width = int(width) if width is not None else int(self.winfo_width())
        except Exception:
            return

        compact = ui_width < 1040
        mode = "compact" if compact else "wide"
        if getattr(self, "_bottom_layout_mode", None) == mode:
            return
        self._bottom_layout_mode = mode

        try:
            status_group.pack_forget()
            actions_group.pack_forget()
        except Exception:
            pass

        if compact:
            status_group.pack(side="top", fill="x", expand=True, padx=(10, 6), pady=(0, 4))
            actions_group.pack(side="top", fill="x", padx=(10, 6), pady=(0, 0))
        else:
            status_group.pack(side="left", fill="x", expand=True, padx=(10, 6))
            actions_group.pack(side="right", padx=(6, 0))

        run_btn = getattr(self, "run_btn", None)
        report_btn = getattr(self, "report_btn", None)
        if run_btn is not None:
            try:
                run_btn.pack_forget()
            except Exception:
                pass
        if report_btn is not None:
            try:
                report_btn.pack_forget()
            except Exception:
                pass

        run_width = 140 if compact else 150
        run_height = 36 if compact else 40
        report_width = 115 if compact else 120
        report_height = 36 if compact else 40
        if run_btn is not None:
            try:
                run_btn.configure(width=run_width, height=run_height, font=("", 13 if compact else 14, "bold"))
            except Exception:
                pass
            run_btn.pack(side="right", padx=(8, 0))
        if report_btn is not None:
            try:
                report_btn.configure(width=report_width, height=report_height)
            except Exception:
                pass
            report_btn.pack(side="right", padx=(6, 0))

    def _apply_adaptive_ui_scaling(self):
        self._adaptive_scale_job = None
        try:
            width = int(self.winfo_width())
            height = int(self.winfo_height())
        except Exception:
            return
        if hasattr(self, "_update_responsive_bottom_layout"):
            self._update_responsive_bottom_layout(width)
        target = self._compute_adaptive_widget_scale(width, height)
        current = float(getattr(self, "_adaptive_widget_scale", 1.0) or 1.0)
        if abs(target - current) < 0.02:
            return
        try:
            ctk.set_widget_scaling(target)
            self._adaptive_widget_scale = target
        except Exception:
            pass

    def _schedule_adaptive_ui_scaling(self, _event=None):
        if self._is_closing:
            return
        job = getattr(self, "_adaptive_scale_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        try:
            self._adaptive_scale_job = self.after(120, self._apply_adaptive_ui_scaling)
        except Exception:
            self._adaptive_scale_job = None

    def _install_adaptive_ui_scaling(self):
        if getattr(self, "_adaptive_scale_installed", False):
            return
        self._adaptive_scale_installed = True
        self._adaptive_widget_scale = 1.0
        self._bottom_layout_mode = None
        self._adaptive_scale_job = None
        try:
            self.bind("<Configure>", self._schedule_adaptive_ui_scaling, add="+")
        except Exception:
            pass
        self._after_safe(self._apply_adaptive_ui_scaling, delay_ms=160)

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

        # ・ｰ・ｸ・ ・菩メ ・・｣誤･ｼ ・ｰ・﨑ｩ・壱共.
        # onefile(frozen)・川・・・・菩・・・｣語ｰ _MEI ・簿ｦｬ ・ｽ・・ｼ ・・懦腹 ・・・溢愍・・・・懦復﨑ｩ・壱共.
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

    def _append_detail_log(self, msg):
        entry = str(msg or "").rstrip()
        if not entry:
            return
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {entry}"
        buf = getattr(self, "_detail_log_buffer", None)
        if buf is None:
            buf = []
            self._detail_log_buffer = buf
        buf.append(line)
        max_buffer = int(getattr(self, "_detail_log_buffer_limit", 2000) or 2000)
        if len(buf) > max_buffer:
            del buf[: len(buf) - max_buffer]
        if not getattr(self, "_detail_log_flush_pending", False):
            self._detail_log_flush_pending = True
            self._after_safe(self._flush_detail_log_buffer, delay_ms=120)

    def _flush_detail_log_buffer(self):
        self._detail_log_flush_pending = False
        widget = getattr(self, "detail_log_text", None)
        if widget is None:
            return
        lines = list(getattr(self, "_detail_log_buffer", []) or [])
        if not lines:
            return
        self._detail_log_buffer = []
        try:
            widget.insert("end", "\n".join(lines) + "\n")
            widget.see("end")
            max_lines = int(getattr(self, "_detail_log_max_lines", 6000) or 6000)
            current_line = int(float(str(widget.index("end-1c")).split(".")[0]))
            if current_line > max_lines:
                trim_lines = current_line - max_lines
                widget.delete("1.0", f"{trim_lines + 1}.0")
        except Exception:
            pass

    def _flush_ui_log_buffer(self):
        self._ui_log_flush_pending = False
        widget = getattr(self, "log_text", None)
        if widget is None:
            return
        lines = list(getattr(self, "_ui_log_buffer", []) or [])
        if not lines:
            return
        self._ui_log_buffer = []
        try:
            widget.insert("end", "\n".join(lines) + "\n")
            widget.see("end")
            max_lines = int(getattr(self, "_ui_log_max_lines", 3000) or 3000)
            current_line = int(float(str(widget.index("end-1c")).split(".")[0]))
            if current_line > max_lines:
                trim_lines = current_line - max_lines
                widget.delete("1.0", f"{trim_lines + 1}.0")
        except Exception:
            pass

    def _insert_ui_log_line(self, msg):
        record = classify_log_message(str(msg or ""))
        if not record.get("ui_visible"):
            return

        line = str(record.get("message", "") or "").rstrip()
        if not line:
            return
        buf = getattr(self, "_ui_log_buffer", None)
        if buf is None:
            buf = []
            self._ui_log_buffer = buf
        buf.append(line)
        max_buffer = int(getattr(self, "_ui_log_buffer_limit", 2000) or 2000)
        if len(buf) > max_buffer:
            del buf[: len(buf) - max_buffer]

        if getattr(self, "log_text", None) is None:
            return

        if not getattr(self, "_ui_log_flush_pending", False):
            self._ui_log_flush_pending = True
            self._after_safe(self._flush_ui_log_buffer, delay_ms=120)

    def _append_log(self, msg, log_to_file=True):
        raw_msg = self._mask_sensitive_text(str(msg or ""))
        self._append_detail_log(raw_msg)
        msg = self._normalize_ui_message(raw_msg)
        link_line = ""
        if msg == ALERT_MSVC_REQUIRED:
            msg = (
                "일부 한국어 의존성 설치에 실패했습니다. "
                "Windows에서는 Microsoft C++ Build Tools가 필요할 수 있습니다."
            )
            link_line = f"다운로드 링크: {self._MSVC_BUILD_TOOLS_URL}"
        elif self._looks_like_msvc_requirement_message(msg):
            link_line = f"다운로드 링크: {self._MSVC_BUILD_TOOLS_URL}"
        now = time.monotonic()
        last_msg = getattr(self, "_last_log_msg", "")
        last_ts = float(getattr(self, "_last_log_ts", 0.0) or 0.0)
        # ・ｬ・ｬ ・・ｸｵ(callback + logger + task print)・川・ ・呷攵 ・肥亨・・ ・ｧ・ ・懋ｰ・・ｴ
        # ・瀧ｳｵ ・・・据・・・ｽ・ｰ 1巐誤ｧ・・ｰ・晨復・､.
        if msg == last_msg and (now - last_ts) < 0.35:
            return
        self._last_log_msg = msg
        self._last_log_ts = now

        if log_to_file:
            self._log_to_file(msg)
            if link_line:
                self._log_to_file(link_line)
        if msg == ALERT_MSVC_REQUIRED or self._looks_like_msvc_requirement_message(msg):
            self._after_safe(
                self._show_msvc_required_alert
            )
        if msg == ALERT_MFA_PERMISSION_DENIED:
            self._after_safe(
                lambda: self._show_copyable_alert(
                    title="MFA permission error",
                    message=(
                        "MFA internal executable (compute-mfcc-feats) may fail due to permission issues (WinError 5).\n\n"
                        "Checklist:\n"
                        "- Check antivirus/Defender exclusion for the MFA runtime path\n"
                        "- Check Controlled Folder Access/AppLocker policies\n"
                        "- Try running the app once with administrator privileges"
                    ),
                    alert_key="mfa_permission_denied",
                )
            )
        self._insert_ui_log_line(msg)
        if link_line:
            self._insert_ui_log_line(link_line)

    def _looks_like_msvc_requirement_message(self, msg):
        text = str(msg or "")
        lowered = text.lower()
        return (
            ("microsoft visual c++ 14.0" in lowered or "c++ build tools" in lowered)
            and ("필수" in text or "required" in lowered)
        )

    def _resolve_setup_mfa_script_path(self):
        return resolve_setup_mfa_script_path(
            app_dir=str(getattr(self, "app_dir", "") or ""),
            app_data_dir=str(getattr(self, "app_data_dir", "") or ""),
            writable_data_dir=str(getattr(self, "writable_data_dir", "") or ""),
            frozen=bool(getattr(sys, "frozen", False)),
            executable_path=str(getattr(sys, "executable", "") or ""),
        )

    def _build_setup_mfa_recovery_guide(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        if lang not in {"korean", "japanese"}:
            lang = "korean"
        script_path = self._resolve_setup_mfa_script_path()
        if script_path:
            command = f'cmd /c ""{script_path}" --recovery --non-interactive --language {lang}"'
            return (
                "자동 복구를 위해 setup_mfa.bat를 수동 실행해 주세요.\n"
                f"- 스크립트 경로: {script_path}\n"
                "- 실행 명령:\n"
                f"{command}"
            )
        return (
            "자동 복구를 위해 setup_mfa.bat를 수동 실행해 주세요.\n"
            "- 실행 명령:\n"
            f"cmd /c \"\"%LOCALAPPDATA%\\UTAU_Auto_OTO_v3\\setup_mfa.bat\" --recovery --non-interactive --language {lang}\"\n"
            "또는\n"
            f"cmd /c \"\"%LOCALAPPDATA%\\UTAU_Auto_OTO\\setup_mfa.bat\" --recovery --non-interactive --language {lang}\""
        )

    def _show_msvc_required_alert(self):
        setup_guide = self._build_setup_mfa_recovery_guide()
        self._show_copyable_alert(
            title="\ud55c\uad6d\uc5b4 \uc758\uc874\uc131 \uc548\ub0b4",
            message=(
                "\uc77c\ubd80 \uc758\uc874\uc131 \uc124\uce58\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4. "
                "\uc0ac\uc6a9\uc740 \uac00\ub2a5\ud558\uc9c0\ub9cc \uc815\ud655\ub3c4\uc5d0 \uc601\ud5a5\uc774 \uc788\uc744 \uc218 \uc788\uc73c\ub2c8 "
                "C++ \ud234\uc744 \uc124\uce58\ud574\uc8fc\uc138\uc694.\n\n"
                "\uc124\uce58 \ub9c1\ud06c:\n"
                f"{self._MSVC_BUILD_TOOLS_URL}\n\n"
                "\uad8c\uc7a5:\n"
                "1) C++ Build Tools \uc124\uce58\n"
                "2) \ud130\ubbf8\ub110 \uc7ac\uc2dc\uc791\n"
                "3) MFA \uc124\uce58/\uc815\ub82c \uc7ac\uc2dc\ub3c4\n\n"
                f"{setup_guide}"
            ),
            alert_key="msvc_required",
            link_url=self._MSVC_BUILD_TOOLS_URL,
            link_label="\uc124\uce58 \ud398\uc774\uc9c0 \uc5f4\uae30",
        )

    def _show_copyable_alert(self, title, message, alert_key=None, link_url="", link_label="링크 열기"):
        if self._is_closing:
            return
        if alert_key and alert_key in self._shown_alert_keys:
            return
        if alert_key:
            self._shown_alert_keys.add(alert_key)

        safe_title = self._normalize_ui_message(str(title or "알림"))
        safe_message = self._mask_sensitive_text(str(message or ""))
        win = tk.Toplevel(self)
        win.title(safe_title)
        win.geometry("700x330")
        win.minsize(520, 260)
        win.transient(self)
        win.grab_set()

        frame = ctk.CTkFrame(win)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(frame, text=safe_title, font=("", 16, "bold"), anchor="w").pack(
            fill="x", padx=10, pady=(10, 6)
        )

        text = ctk.CTkTextbox(frame, wrap="word", font=("Consolas", 12))
        text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        text.insert("1.0", safe_message)
        text.configure(state="disabled")

        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.pack(fill="x", padx=10, pady=(0, 10))

        def _copy_text():
            win.clipboard_clear()
            win.clipboard_append(safe_message)
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

        ctk.CTkButton(btns, text=t("복사"), width=90, command=_copy_text).pack(side="right")
        if link_url:
            ctk.CTkButton(btns, text=t("링크 복사"), width=100, command=_copy_link).pack(side="right", padx=(0, 8))
            ctk.CTkButton(btns, text=link_label, width=120, command=_open_link).pack(side="right", padx=(0, 8))
        ctk.CTkButton(btns, text=t("닫기"), width=90, command=win.destroy).pack(side="right", padx=(0, 8))

    def _ask_yes_no_dialog_sync(self, title: str, message: str, default: bool = False) -> bool:
        safe_title = self._normalize_ui_message(str(title or "안내"))
        safe_message = self._mask_sensitive_text(str(message or ""))

        # Main-thread callers must show the dialog immediately.
        # Waiting on an Event here would block Tk's event loop and freeze the app.
        if threading.current_thread() is threading.main_thread():
            try:
                return bool(messagebox.askyesno(safe_title, safe_message))
            except Exception:
                return bool(default)

        result = {"value": bool(default)}
        done = threading.Event()

        def _prompt():
            try:
                result["value"] = bool(messagebox.askyesno(safe_title, safe_message))
            except Exception:
                result["value"] = bool(default)
            finally:
                done.set()

        self._after_safe(_prompt)
        done.wait(timeout=1200.0)
        return bool(result.get("value", default))

    def _scan_voicebank_script_hints(self, wav_dir: str, sample_limit: int = 4000) -> dict:
        summary = {
            "scanned": 0,
            "hangul_count": 0,
            "kana_count": 0,
            "hangul_examples": [],
            "kana_examples": [],
        }
        root = str(wav_dir or "").strip()
        if not root or not os.path.isdir(root):
            return summary

        exts = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".m4a", ".lab"}
        max_examples = 4
        limit = max(200, int(sample_limit or 0))
        seen_hangul = set()
        seen_kana = set()

        for dirpath, _dirnames, filenames in os.walk(root):
            for file_name in filenames:
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in exts:
                    continue
                stem = os.path.splitext(file_name)[0]
                if not stem:
                    continue
                summary["scanned"] += 1
                has_hangul = bool(_HANGUL_CHAR_RE.search(stem))
                has_kana = bool(_KANA_CHAR_RE.search(stem))
                if has_hangul:
                    summary["hangul_count"] += 1
                    if len(summary["hangul_examples"]) < max_examples and stem not in seen_hangul:
                        summary["hangul_examples"].append(stem)
                        seen_hangul.add(stem)
                if has_kana:
                    summary["kana_count"] += 1
                    if len(summary["kana_examples"]) < max_examples and stem not in seen_kana:
                        summary["kana_examples"].append(stem)
                        seen_kana.add(stem)
                if summary["scanned"] >= limit:
                    return summary
        return summary

    def _confirm_language_script_mismatch(self, language: str, wav_dir: str, stage_name: str) -> bool:
        lang = str(language or "").strip().lower()
        if lang not in {"korean", "japanese"}:
            return True
        scan = self._scan_voicebank_script_hints(wav_dir)
        scanned = int(scan.get("scanned", 0) or 0)
        if scanned <= 0:
            return True

        mismatch = (lang == "korean" and int(scan.get("kana_count", 0) or 0) > 0) or (
            lang == "japanese" and int(scan.get("hangul_count", 0) or 0) > 0
        )
        if not mismatch:
            return True

        lang_text = t("한국어") if lang == "korean" else "일본어"
        mismatch_text = t("히라가나/가타카나") if lang == "korean" else "한글"
        examples = scan.get("kana_examples", []) if lang == "korean" else scan.get("hangul_examples", [])
        example_text = ", ".join(examples[:3]) if examples else "(예시 없음)"
        message = (
            f"{stage_name} 전에 언어 불일치가 감지되었습니다.\n\n"
            f"- 현재 설정 언어: {lang_text}\n"
            f"- 감지된 문자: {mismatch_text}\n"
            f"- 샘플 스캔: {scanned}개 파일\n"
            f"- 예시 파일명: {example_text}\n\n"
            "이 상태로 진행하면 사전/정렬/OTO 결과가 어긋날 수 있습니다.\n"
            "계속 진행하려면 [확인], 중단하려면 [취소]를 누르세요."
        )
        ask_fn = getattr(self, "_ask_yes_no_dialog_sync", None)
        proceed = bool(ask_fn("언어 설정 확인", message, default=False)) if callable(ask_fn) else False
        if not proceed:
            self._append_log(
                f"ℹ 사용자가 언어 불일치 경고에서 취소했습니다. stage={stage_name}, lang={lang}"
            )
        return proceed

    def _set_status(self, msg):
        msg = self._normalize_ui_message(str(msg))
        color = self._status_color_for_message(msg)

        def _do():
            self.status_label.configure(text=msg, text_color=color)
            ratio = self._parse_progress_ratio_from_status(msg)
            if ratio is None:
                lower = msg.strip().lower()
                if ("완료" in msg or "success" in lower or "complete" in lower) and not self.is_running:
                    ratio = 1.0
                elif ("오류" in msg or "error" in lower or "failed" in lower) and not self.is_running:
                    ratio = 0.0
            if ratio is not None:
                self._set_progress(ratio)

        self._after_safe(_do)

    def _log_startup_ml_model_roots_once(self):
        if bool(getattr(self, "_ml_model_root_startup_logged", False)):
            return
        if not hasattr(self, "_effective_ml_model_root"):
            return
        kr_root, kr_src = self._effective_ml_model_root("korean")
        ja_root, ja_src = self._effective_ml_model_root("japanese")
        kr_txt = kr_root if kr_root else "(none)"
        ja_txt = ja_root if ja_root else "(none)"
        self._append_log(
            f"[OTO-ML] startup model roots | KR[{kr_src}]={kr_txt} | JA[{ja_src}]={ja_txt}",
            log_to_file=True,
        )
        self._ml_model_root_startup_logged = True

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
        keyword_ratio = self._infer_progress_ratio_from_keywords(text)
        if keyword_ratio is not None:
            return keyword_ratio
        return None

    def _infer_progress_ratio_from_keywords(self, msg):
        text = str(msg or "")
        lowered = text.lower()
        stage_pairs = {
            "1/5": 0.12,
            "2/5": 0.28,
            "3/5": 0.48,
            "4/5": 0.72,
            "5/5": 0.90,
        }
        for token, ratio in stage_pairs.items():
            if token in lowered:
                return ratio
        stage_text = {
            "1단계": 0.12,
            "2단계": 0.28,
            "3단계": 0.48,
            "4단계": 0.72,
            "5단계": 0.90,
        }
        for token, ratio in stage_text.items():
            if token in text:
                return ratio
        keyword_map_lower = [
            ("checking mfa prerequisites", 0.08),
            ("creating micromamba environment", 0.12),
            ("installing micromamba", 0.20),
            ("installing mfa", 0.40),
            ("repairing python packaging tools", 0.55),
            ("installing korean dependencies", 0.68),
            ("downloading mfa model", 0.80),
            ("extracting mfa model", 0.85),
            ("generating oto", 0.92),
            ("completed successfully", 1.0),
        ]
        for token, ratio in keyword_map_lower:
            if token in lowered:
                return ratio
        keyword_map_text = [
            ("MFA 필수 구성요소 점검", 0.08),
            ("Micromamba 환경 생성", 0.12),
            ("Micromamba 설치", 0.20),
            ("MFA 설치", 0.40),
            ("Python 패키징 도구 복구", 0.55),
            ("한국어 의존성 설치", 0.68),
            ("MFA 모델 다운로드", 0.80),
            ("MFA 모델 압축 해제", 0.85),
            ("OTO 생성", 0.92),
            ("완료", 1.0),
        ]
        for token, ratio in keyword_map_text:
            if token in text:
                return ratio
        if ("완료" in text or "success" in lowered or "complete" in lowered) and not getattr(self, "is_running", False):
            return 1.0
        return None

    def _set_progress(self, ratio):
        if not hasattr(self, "progress_bar"):
            return
        try:
            value = max(0.0, min(1.0, float(ratio)))
        except Exception:
            return
        if threading.current_thread() is not threading.main_thread():
            self._after_safe(lambda v=value: self._set_progress(v))
            return
        if bool(getattr(self, "is_running", False)):
            peak = float(getattr(self, "_progress_peak_ratio", 0.0) or 0.0)
            if value < peak:
                value = peak
            else:
                self._progress_peak_ratio = value
        else:
            self._progress_peak_ratio = value
        self.progress_bar.set(value)
        if hasattr(self, "progress_label"):
            self.progress_label.configure(text=f"{int(round(value * 100.0))}%")

    def _status_color_for_message(self, msg):
        text = str(msg or "")
        lowered = text.lower()

        if self.is_running:
            # running state
            return "#FFE082"
        if ("오류" in text) or ("error" in lowered) or ("failed" in lowered):
            return "#FF6B6B"
        if ("경고" in text) or ("warning" in lowered):
            return "#FFB74D"
        if ("완료" in text) or ("성공" in text) or ("success" in lowered):
            return "#66BB6A"
        return "#FFFFFF"

    def _run_auto_validation(self, wav_dir, tg_folder, out_path, callback=None):
        log_cb = callback if callable(callback) else (lambda msg: self._append_log(msg, log_to_file=True))
        out_file = str(out_path or "").strip()
        if not out_file:
            log_cb("기준 OTO 경로가 없어 자동 검증을 건너뜁니다.")
            return {"errors": 0, "warnings": 0, "sampled_wavs": 0, "checked_wavs_total": 0, "truncated_issues": 0, "skipped": True}
        if not os.path.isfile(out_file):
            log_cb(f"기준 OTO 파일을 찾을 수 없어 자동 검증을 건너뜁니다: {out_file}")
            return {"errors": 0, "warnings": 0, "sampled_wavs": 0, "checked_wavs_total": 0, "truncated_issues": 0, "skipped": True}

        os.environ.setdefault("UTOA_OTO_AUTO_VALIDATE_FAST", "1")
        os.environ.setdefault("UTOA_OTO_AUTO_VALIDATE_MAX_FILES", "180")
        os.environ.setdefault("UTOA_OTO_AUTO_VALIDATE_MAX_ISSUES", "1200")
        os.environ.setdefault("UTOA_OTO_AUTO_VALIDATE_USE_TEXTGRID", "0")
        log_cb(
            "OTO 자동 검증 시작 "
            f"(fast={os.environ.get('UTOA_OTO_AUTO_VALIDATE_FAST', '1')}, "
            f"max_files={os.environ.get('UTOA_OTO_AUTO_VALIDATE_MAX_FILES', '180')})"
        )
        summary = validate_oto_timing(
            wav_dir=wav_dir,
            tg_folder=tg_folder,
            oto_path=out_file,
            language=self._get_language(),
            callback=lambda msg: log_cb(msg),
        )
        err_count = int(summary.get("errors", 0) or 0)
        warn_count = int(summary.get("warnings", 0) or 0)
        sampled_wavs = int(summary.get("sampled_wavs", 0) or 0)
        checked_wavs_total = int(summary.get("checked_wavs_total", 0) or 0)
        truncated_issues = int(summary.get("truncated_issues", 0) or 0)
        if err_count > 0:
            log_cb(
                f"⚠ OTO 검증 결과: error {err_count}, warning {warn_count} "
                f"(files={sampled_wavs}/{checked_wavs_total}, truncated={truncated_issues})"
            )
        else:
            log_cb(
                f"✅ OTO 검증 결과: warning {warn_count} (error 0) "
                f"(files={sampled_wavs}/{checked_wavs_total}, truncated={truncated_issues})"
            )
        return summary

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
            # ・溢・揆 ・・紛 ・､・・・・ｴ ・・愍・ｴ ・尖徐 ・ｭ・罹･ｼ ・倆哩﨑們ｧ ・危株・､.
            return False
        # out_dir・ ・壱｡・・晧┳・・・ｽ・ｰ(snapshot_files・ ・・牟 ・溢搆)・尖巡 ・・甯護攵・・・・｣ｼ﨑罹共.
        is_new_in_run = (not snapshot_files) or (file_path_norm not in snapshot_files)
        if not is_new_in_run:
            return False
        # ・尖徐 ・晧┳ ・・ｰ・ｼ OTO ・・龍・・・簿ｦｬ﨑罹共.
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
            self._append_log(f"임시 정리 완료: files={removed_files}, dirs={removed_dirs}")
        if failed > 0:
            self._append_log(f"임시 정리 실패 항목: {failed}")
        return {"removed_files": removed_files, "removed_dirs": removed_dirs, "failed": failed}

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
            # ・溢・揆 ・・紛 ・､・・・・ｴ ・・愍・ｴ ・尖徐 ・ｭ・罹･ｼ ・倆哩﨑們ｧ ・危株・､.
            return False
        # out_dir・ ・壱｡・・晧┳・・・ｽ・ｰ(snapshot_files・ ・・牟 ・溢搆)・尖巡 ・・甯護攵・・・・｣ｼ﨑罹共.
        is_new_in_run = (not snapshot_files) or (file_path_norm not in snapshot_files)
        if not is_new_in_run:
            return False
        # ・尖徐 ・晧┳ ・・ｰ・ｼ OTO ・・龍・・・簿ｦｬ﨑罹共.
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
            self._append_log(f"임시 정리 완료: files={removed_files}, dirs={removed_dirs}")
        if failed > 0:
            self._append_log(f"임시 정리 실패 항목: {failed}")
        return {"removed_files": removed_files, "removed_dirs": removed_dirs, "failed": failed}

    def _clear_log(self):
        log_widget = getattr(self, "log_text", None)
        if log_widget is not None:
            try:
                log_widget.delete("1.0", "end")
            except Exception:
                pass
        self._ui_log_buffer = []
        self._ui_log_flush_pending = False
        detail_widget = getattr(self, "detail_log_text", None)
        if detail_widget is not None:
            try:
                detail_widget.delete("1.0", "end")
            except Exception:
                pass
        self._detail_log_buffer = []
        self._detail_log_flush_pending = False

    def _set_running(self, running):
        self.is_running = running

        def _do():
            state = "disabled" if running else "normal"
            self.run_btn.configure(state=state)
            if hasattr(self, "status_label"):
                current_text = self.status_label.cget("text")
                self.status_label.configure(text_color=self._status_color_for_message(current_text))
                if running:
                    self._progress_peak_ratio = 0.0
                    self._set_progress(0.0)
                else:
                    ratio = self._parse_progress_ratio_from_status(current_text)
                    if ratio is None:
                        lowered = str(current_text).strip().lower()
                        if ("완료" in str(current_text)) or ("성공" in str(current_text)) or ("success" in lowered):
                            ratio = 1.0
                        elif ("오류" in str(current_text)) or ("실패" in str(current_text)) or ("error" in lowered):
                            ratio = 0.0
                    if ratio is not None:
                        self._set_progress(ratio)

        self._after_safe(_do)

    def _run_in_thread(self, func):
        active_worker = getattr(self, "_active_worker_thread", None)
        if self.is_running:
            if active_worker is not None and active_worker.is_alive():
                messagebox.showwarning("안내", "작업이 이미 실행 중입니다. 완료 후 다시 시도해 주세요.")
                return
            # Stale running flag recovery: previous worker ended but flag remained set.
            self._set_running(False)
            self._append_log("ℹ 내부 상태 복구: 남아 있던 작업 플래그를 자동 해제했습니다.")
        # Try to switch to log tab even if the label changed by theme/localization.
        switched = False
        for tab_name in ("로그", "진행 로그", "상세 로그", "Log"):
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
        def _thread_runner():
            try:
                func()
            finally:
                try:
                    if getattr(self, "_active_worker_thread", None) is threading.current_thread():
                        self._active_worker_thread = None
                except Exception:
                    pass
                # Safety net: if a worker exits while running flag is still set, clear it.
                try:
                    if bool(getattr(self, "is_running", False)):
                        self._after_safe(lambda: self._set_running(False))
                except Exception:
                    pass

        thread = threading.Thread(target=_thread_runner, daemon=True)
        self._active_worker_thread = thread
        thread.start()

    def _safe_clipboard_copy(self, text):
        payload = self._mask_sensitive_text(str(text or ""))
        if not payload:
            return False
        try:
            self.clipboard_clear()
            self.clipboard_append(payload)
            self.update_idletasks()
            return True
        except Exception:
            return False

    def _collect_recent_ui_log_tail(self, max_lines=120, max_chars=8000):
        try:
            raw = str(self.log_text.get("1.0", "end") or "").strip()
        except Exception:
            raw = ""
        if not raw:
            return "(ui log unavailable)"
        lines = raw.splitlines()
        tail = lines[-int(max_lines):] if max_lines > 0 else lines
        text = "\n".join(tail).strip()
        if len(text) > int(max_chars):
            text = text[-int(max_chars):]
        return self._mask_sensitive_text(text or "(ui log empty)")

    def _build_quick_error_report(self, step_name, exception_text, traceback_text=""):
        now = datetime.datetime.now().isoformat()
        language = self._get_language() if hasattr(self, "_get_language") else "unknown"
        auto_format = ""
        try:
            if hasattr(self, "auto_format_var"):
                auto_format = str(self.auto_format_var.get() or "").strip()
        except Exception:
            auto_format = ""
        aligner = ""
        try:
            if hasattr(self, "aligner_var"):
                aligner = str(self.aligner_var.get() or "").strip()
        except Exception:
            aligner = ""
        safe_step = self._mask_sensitive_text(str(step_name or ""))
        safe_error = self._mask_sensitive_text(str(exception_text or ""))
        report_lines = [
            "=== UTAU Auto OTO Quick Report ===",
            f"time: {now}",
            f"app_version: {getattr(self, 'app_version', '')}",
            f"release_channel: {getattr(self, 'release_channel', '')}",
            f"os: {platform.platform()}",
            f"python: {sys.version.split()[0]}",
            f"step: {safe_step}",
            f"error: {safe_error}",
            f"language: {language}",
            f"format: {auto_format}",
            f"aligner: {aligner}",
            f"wav_dir: {self._sanitize_path_value(self.wav_entry.get() if hasattr(self, 'wav_entry') else '')}",
            f"template_oto: {self._sanitize_path_value(self.tpl_entry.get() if hasattr(self, 'tpl_entry') else '')}",
            f"output_oto: {self._sanitize_path_value(self.out_entry.get() if hasattr(self, 'out_entry') else '')}",
            f"log_path: {self._sanitize_path_value(getattr(self, 'log_path', ''))}",
            f"event_log_path: {self._sanitize_path_value(getattr(self, 'event_log_path', ''))}",
            "",
            "[UI Log Tail]",
            self._collect_recent_ui_log_tail(),
        ]
        tb = self._mask_sensitive_text(str(traceback_text or "").strip())
        if tb:
            report_lines.extend(["", "[Traceback]", tb])
        return "\n".join(report_lines).strip() + "\n"

    def _copy_quick_error_report(self):
        step_name = str(getattr(self, "_last_error_step", "manual_report") or "manual_report")
        err_msg = str(getattr(self, "_last_error_message", "(no captured exception)") or "(no captured exception)")
        tb = str(getattr(self, "_last_error_traceback", "") or "")
        report_text = self._build_quick_error_report(step_name, err_msg, tb)
        copied = self._safe_clipboard_copy(report_text)
        if copied:
            self._append_log("빠른 오류 리포트를 클립보드에 복사했습니다.")
            self._show_copyable_alert(
                title="빠른 오류 리포트",
                message=report_text,
                alert_key=None,
            )
        else:
            self._show_copyable_alert(
                title="빠른 오류 리포트",
                message=report_text,
                alert_key=None,
            )

    def _handle_error(self, step_name, exception, traceback_text=None, auto_popup=True):
        if traceback_text is None:
            traceback_text = traceback.format_exc()
        safe_step = self._mask_sensitive_text(str(step_name or "unknown"))
        tb = self._mask_sensitive_text(str(traceback_text or "").strip())
        exc_text = self._mask_sensitive_text(str(exception or ""))
        self._last_error_step = safe_step
        self._last_error_message = exc_text
        self._last_error_traceback = tb
        self.logger.error(f"[{safe_step}] {exc_text}\n{tb}")
        self._append_log(f"\n{'=' * 50}")
        self._append_log(f"오류 발생 [{safe_step}]")
        self._append_log(f"   메시지: {exc_text}")
        report_text = self._build_quick_error_report(safe_step, exc_text, tb)
        copied = self._safe_clipboard_copy(report_text)
        if copied:
            self._append_log("   빠른 리포트를 클립보드에 복사했습니다.")
        else:
            self._append_log("   리포트 자동 복사에 실패했습니다. 아래 내용을 수동으로 복사해 주세요.")
        self._append_log(f"{'=' * 50}\n")
        self._set_status(f"오류 발생: {safe_step}")
        if auto_popup and self._should_show_error_popup(safe_step, exc_text):
            self._after_safe(
                lambda payload=report_text: self._show_copyable_alert(
                    title="오류 리포트",
                    message=payload,
                    alert_key=None,
                )
            )

    def _export_error_report(self):
        writable_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or self.app_dir
        )
        report_path = os.path.join(
            writable_dir,
            f"error_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        try:
            step_name = str(getattr(self, "_last_error_step", "manual_report") or "manual_report")
            err_msg = str(getattr(self, "_last_error_message", "(no captured exception)") or "(no captured exception)")
            tb = str(getattr(self, "_last_error_traceback", "") or "")
            quick_report = self._build_quick_error_report(step_name, err_msg, tb)
            safe_report_path = self._sanitize_path_value(report_path)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(quick_report)
                f.write("\n--- Full UI Log ---\n")
                f.write(self._mask_sensitive_text(str(self.log_text.get("1.0", "end") or "")))
                if os.path.exists(self.log_path):
                    f.write(f"\n--- File Log ({self._sanitize_path_value(self.log_path)}) ---\n")
                    with open(self.log_path, "r", encoding="utf-8", errors="replace") as lf:
                        f.write(self._mask_sensitive_text(lf.read()))
                event_log_path = getattr(self, "event_log_path", "")
                if event_log_path and os.path.exists(event_log_path):
                    f.write(f"\n--- Event Log ({self._sanitize_path_value(event_log_path)}) ---\n")
                    with open(event_log_path, "r", encoding="utf-8", errors="replace") as ef:
                        f.write(self._mask_sensitive_text(ef.read()))

            copied = self._safe_clipboard_copy(quick_report)
            if copied:
                self._append_log("빠른 오류 리포트를 클립보드에 복사했습니다.")
            if sys.platform == "win32":
                os.startfile(os.path.dirname(report_path))

            messagebox.showinfo(
                "오류 리포트",
                (
                    f"오류 리포트를 저장했습니다.\n\n저장 위치:\n{safe_report_path}\n\n"
                    + (
                        "빠른 리포트는 클립보드에도 복사되었습니다.\n필요 시 그대로 붙여넣어 공유해 주세요."
                        if copied
                        else "클립보드 자동 복사에 실패했습니다. 저장된 파일 내용을 공유해 주세요."
                    )
                ),
            )
        except Exception as e:
            messagebox.showerror("오류", f"오류 리포트 저장에 실패했습니다: {self._mask_sensitive_text(str(e))}")

class ConfigMixin:
    def _config_path_candidates(self):
        primary_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or ""
        )
        app_dir = getattr(self, "app_dir", "")
        candidates = [
            os.path.join(primary_dir, "config.json") if primary_dir else "",
            os.path.join(app_dir, "config.json") if app_dir else "",
        ]
        dedup = []
        seen = set()
        for path in candidates:
            norm = os.path.normcase(os.path.abspath(str(path or "")))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            dedup.append(path)
        return dedup

    def _primary_config_path(self):
        candidates = self._config_path_candidates()
        if candidates:
            return candidates[0]
        fallback_dir = getattr(self, "app_dir", "") or os.getcwd()
        return os.path.join(fallback_dir, "config.json")

    def _save_config(self, *_args):
        params = self._get_params()
        from ui.i18n import get_language as _get_ui_lang
        config = {
            "ui_language": _get_ui_lang(),
            "ui_theme_profile": (
                normalize_theme_profile_name(self.ui_theme_var.get(), default=DEFAULT_THEME_PROFILE)
                if hasattr(self, "ui_theme_var")
                else DEFAULT_THEME_PROFILE
            ),
            "voicebank_tuning_preset_key": str(getattr(self, "_selected_voicebank_preset_key", "") or ""),
            "wav_dir": self.wav_entry.get(),
            "tpl_path": self.tpl_entry.get(),
            "out_path": self.out_entry.get(),
            "custom_phonemes": self.custom_phoneme_var.get(),
            "alias_suffix": self.alias_suffix_var.get(),
            "openutau_compatible": self.openutau_var.get(),
            "gen_missing_vowels": self.gen_missing_vowels_var.get(),
            "gen_dash_alias": self.gen_dash_alias_var.get() if hasattr(self, "gen_dash_alias_var") else True,
            "no_base_oto": self.no_base_oto_var.get(),
            "no_mfa_oto_mode": (
                self._get_no_mfa_oto_mode_code()
                if hasattr(self, "_get_no_mfa_oto_mode_code")
                else "remap"
            ),
            "recursive_voicebank_scan": self.recursive_voicebank_scan_var.get() if hasattr(self, "recursive_voicebank_scan_var") else False,
            "enable_ml_correction": self.enable_ml_correction_var.get() if hasattr(self, "enable_ml_correction_var") else True,
            "ml_route": self._get_ml_route_code() if hasattr(self, "_get_ml_route_code") else "auto",
            "ml_selector_mode": self.ml_selector_mode_var.get() if hasattr(self, "ml_selector_mode_var") else "+셀렉터",
            "ml_coupled_enable": self.ml_coupled_enable_var.get() if hasattr(self, "ml_coupled_enable_var") else True,
            "ml_coupled_min_conf": self.ml_coupled_min_conf_var.get() if hasattr(self, "ml_coupled_min_conf_var") else "",
            "ml_coupled_min_conf_use_model_meta": self.ml_coupled_min_conf_use_model_meta_var.get() if hasattr(self, "ml_coupled_min_conf_use_model_meta_var") else True,
            "ml_coupled_min_conf_model_offset": self.ml_coupled_min_conf_model_offset_var.get() if hasattr(self, "ml_coupled_min_conf_model_offset_var") else "",
            "ml_coupled_min_conf_kr_cv": self.ml_coupled_min_conf_kr_cv_var.get() if hasattr(self, "ml_coupled_min_conf_kr_cv_var") else "",
            "ml_coupled_min_conf_kr_cvc": self.ml_coupled_min_conf_kr_cvc_var.get() if hasattr(self, "ml_coupled_min_conf_kr_cvc_var") else "",
            "ml_coupled_min_conf_kr_cvvc": self.ml_coupled_min_conf_kr_cvvc_var.get() if hasattr(self, "ml_coupled_min_conf_kr_cvvc_var") else "",
            "ml_coupled_min_conf_kr_vcv": self.ml_coupled_min_conf_kr_vcv_var.get() if hasattr(self, "ml_coupled_min_conf_kr_vcv_var") else "",
            "ml_coupled_min_conf_ja_cv": self.ml_coupled_min_conf_ja_cv_var.get() if hasattr(self, "ml_coupled_min_conf_ja_cv_var") else "",
            "ml_coupled_min_conf_ja_cvc": self.ml_coupled_min_conf_ja_cvc_var.get() if hasattr(self, "ml_coupled_min_conf_ja_cvc_var") else "",
            "ml_coupled_min_conf_ja_cvvc": self.ml_coupled_min_conf_ja_cvvc_var.get() if hasattr(self, "ml_coupled_min_conf_ja_cvvc_var") else "",
            "ml_coupled_min_conf_ja_vcv": self.ml_coupled_min_conf_ja_vcv_var.get() if hasattr(self, "ml_coupled_min_conf_ja_vcv_var") else "",
            "ml_coupled_device": self.ml_coupled_device_var.get() if hasattr(self, "ml_coupled_device_var") else "auto",
            "ml_coupled_backend": self.ml_coupled_backend_var.get() if hasattr(self, "ml_coupled_backend_var") else "auto",
            "ml_coupled_strict_constraint": self.ml_coupled_strict_constraint_var.get() if hasattr(self, "ml_coupled_strict_constraint_var") else False,
            "ml_batch_inference_enable": self.ml_batch_inference_enable_var.get() if hasattr(self, "ml_batch_inference_enable_var") else True,
            "ml_batch_inference_size": self.ml_batch_inference_size_var.get() if hasattr(self, "ml_batch_inference_size_var") else "256",
            "ml_legacy_fallback_enable": self.ml_legacy_fallback_enable_var.get() if hasattr(self, "ml_legacy_fallback_enable_var") else False,
            "ml_hybrid_routing_enable": self.ml_hybrid_routing_enable_var.get() if hasattr(self, "ml_hybrid_routing_enable_var") else True,
            "ml_e2e_enable": self.ml_e2e_enable_var.get() if hasattr(self, "ml_e2e_enable_var") else False,
            "ml_e2e_mode": self._get_e2e_mode_code() if hasattr(self, "_get_e2e_mode_code") else "hybrid",
            "ml_e2e_t_low": self.ml_e2e_t_low_var.get() if hasattr(self, "ml_e2e_t_low_var") else "",
            "ml_e2e_t_high": self.ml_e2e_t_high_var.get() if hasattr(self, "ml_e2e_t_high_var") else "",
            "ml_e2e_blend_alpha": self.ml_e2e_blend_alpha_var.get() if hasattr(self, "ml_e2e_blend_alpha_var") else "",
            "ml_anchor_mel_gamma": self.ml_anchor_mel_gamma_var.get() if hasattr(self, "ml_anchor_mel_gamma_var") else "",
            "ml_model_root_kr": self.ml_model_root_kr_var.get() if hasattr(self, "ml_model_root_kr_var") else "",
            "ml_model_root_ja": self.ml_model_root_ja_var.get() if hasattr(self, "ml_model_root_ja_var") else "",
            "oto_crnn_model_path": "",
            "oto_crnn_engine": (
                self._get_oto_crnn_engine_code()
                if hasattr(self, "_get_oto_crnn_engine_code")
                else "boundary_decoder"
            ),
            "oto_crnn_device": self.oto_crnn_device_var.get() if hasattr(self, "oto_crnn_device_var") else "auto",
            "oto_crnn_special_aliases": self.oto_crnn_special_aliases_var.get() if hasattr(self, "oto_crnn_special_aliases_var") else "",
            "cvn_correction_enable": self.cvn_correction_enable_var.get() if hasattr(self, "cvn_correction_enable_var") else True,
            "cvn_low_conf_only": self.cvn_low_conf_only_var.get() if hasattr(self, "cvn_low_conf_only_var") else False,
            "mapping_supervised_enable": self.mapping_supervised_enable_var.get() if hasattr(self, "mapping_supervised_enable_var") else True,
            "mapping_supervised_mode": (
                self._get_mapping_supervised_mode_code()
                if hasattr(self, "_get_mapping_supervised_mode_code")
                else "auto"
            ),
            "cv_order_prior_enable": self.cv_order_prior_enable_var.get() if hasattr(self, "cv_order_prior_enable_var") else True,
            "cv_order_prior_strength": self.cv_order_prior_strength_var.get() if hasattr(self, "cv_order_prior_strength_var") else "",
            "vc_correction_enable": self.vc_correction_enable_var.get() if hasattr(self, "vc_correction_enable_var") else True,
            "kr_vc_neighbor_enable": self.kr_vc_neighbor_enable_var.get() if hasattr(self, "kr_vc_neighbor_enable_var") else True,
            "kr_vc_neighbor_blend": self.kr_vc_neighbor_blend_var.get() if hasattr(self, "kr_vc_neighbor_blend_var") else "",
            "kr_vc_neighbor_max_shift": self.kr_vc_neighbor_max_shift_var.get() if hasattr(self, "kr_vc_neighbor_max_shift_var") else "",
            "kr_vc_neighbor_lead_ms": self.kr_vc_neighbor_lead_ms_var.get() if hasattr(self, "kr_vc_neighbor_lead_ms_var") else "",
            "kr_vc_neighbor_tail_ms": self.kr_vc_neighbor_tail_ms_var.get() if hasattr(self, "kr_vc_neighbor_tail_ms_var") else "",
            "kr_vc_neighbor_min_len": self.kr_vc_neighbor_min_len_var.get() if hasattr(self, "kr_vc_neighbor_min_len_var") else "",
            "ja_vc_neighbor_enable": self.ja_vc_neighbor_enable_var.get() if hasattr(self, "ja_vc_neighbor_enable_var") else True,
            "ja_vc_neighbor_blend": self.ja_vc_neighbor_blend_var.get() if hasattr(self, "ja_vc_neighbor_blend_var") else "",
            "ja_vc_neighbor_max_shift": self.ja_vc_neighbor_max_shift_var.get() if hasattr(self, "ja_vc_neighbor_max_shift_var") else "",
            "ja_vc_neighbor_lead_ms": self.ja_vc_neighbor_lead_ms_var.get() if hasattr(self, "ja_vc_neighbor_lead_ms_var") else "",
            "ja_vc_neighbor_tail_ms": self.ja_vc_neighbor_tail_ms_var.get() if hasattr(self, "ja_vc_neighbor_tail_ms_var") else "",
            "ja_vc_neighbor_min_len": self.ja_vc_neighbor_min_len_var.get() if hasattr(self, "ja_vc_neighbor_min_len_var") else "",
            "soft_bank_mode": self.soft_bank_mode_var.get() if hasattr(self, "soft_bank_mode_var") else False,
            "low_rms_gain_enable": self.low_rms_gain_enable_var.get() if hasattr(self, "low_rms_gain_enable_var") else True,
            "weak_voice_assist_enable": self.weak_voice_assist_enable_var.get() if hasattr(self, "weak_voice_assist_enable_var") else True,
            "weak_voice_assist_strength": self.weak_voice_assist_strength_var.get() if hasattr(self, "weak_voice_assist_strength_var") else "",
            "mapping_strict_mode": (
                self._get_mapping_strict_mode_code()
                if hasattr(self, "_get_mapping_strict_mode_code")
                else "soft"
            ),
            "low_conf_force_lock_mode": self.low_conf_force_lock_mode_var.get() if hasattr(self, "low_conf_force_lock_mode_var") else False,
            "weak_boundary_reduce_missing": self.weak_boundary_reduce_missing_var.get() if hasattr(self, "weak_boundary_reduce_missing_var") else False,
            "weak_boundary_block_mismap": self.weak_boundary_block_mismap_var.get() if hasattr(self, "weak_boundary_block_mismap_var") else False,
            "ja_mapping_words_fallback_enabled": self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True,
            "ja_mapping_spn_ratio_threshold": self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35,
            "ja_mapping_min_vowel_phone_ratio": self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5,
            "ja_mapping_debug_reason_logging": self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True,
            "kr_anchor_profile_path": self.kr_anchor_profile_path_var.get() if hasattr(self, "kr_anchor_profile_path_var") else "",
            "kr_mapping_confidence_threshold": self.kr_mapping_confidence_threshold_var.get() if hasattr(self, "kr_mapping_confidence_threshold_var") else "",
            "kr_mapping_max_index_jump_default": self.kr_mapping_max_index_jump_default_var.get() if hasattr(self, "kr_mapping_max_index_jump_default_var") else 1,
            "kr_mapping_max_index_jump_high_conf": self.kr_mapping_max_index_jump_high_conf_var.get() if hasattr(self, "kr_mapping_max_index_jump_high_conf_var") else 2,
            "kr_continuity_enable": self.kr_continuity_enable_var.get() if hasattr(self, "kr_continuity_enable_var") else True,
            "kr_continuity_max_offset_adj": self.kr_continuity_max_offset_adj_var.get() if hasattr(self, "kr_continuity_max_offset_adj_var") else "",
            "ml_same_language_borrow_only": self.ml_same_language_borrow_only_var.get() if hasattr(self, "ml_same_language_borrow_only_var") else True,
            "language": self.lang_var.get(),
            "auto_format": self.auto_format_var.get(),
            "ja_alias_style": self.ja_alias_style_var.get(),
            "en_cvvc_pack": self.en_cvvc_pack_var.get() if hasattr(self, "en_cvvc_pack_var") else "LITE",
            "en_cvvc_beat": self.en_cvvc_beat_var.get() if hasattr(self, "en_cvvc_beat_var") else "8-beat",
            "en_cvvc_preset": self.en_cvvc_preset_var.get() if hasattr(self, "en_cvvc_preset_var") else "Core",
            "en_cvvc_list_fallback": self.en_cvvc_list_fallback_var.get() if hasattr(self, "en_cvvc_list_fallback_var") else True,
            "show_advanced_aligner": self.show_advanced_aligner_var.get() if hasattr(self, "show_advanced_aligner_var") else False,
            "developer_mode_enabled": self.developer_mode_enabled_var.get() if hasattr(self, "developer_mode_enabled_var") else False,
            "aligner": self.aligner_var.get(),
            "mfa_align_profile": self.mfa_align_profile_var.get() if hasattr(self, "mfa_align_profile_var") else "\uae30\ubcf8",
            "mfa_align_profile_code": self._get_mfa_align_profile_code() if hasattr(self, "_get_mfa_align_profile_code") else "default",
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
        config_path = self._primary_config_path()
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {e}")

    def _load_config(self):
        config_path = ""
        for candidate in self._config_path_candidates():
            if os.path.exists(candidate):
                config_path = candidate
                break
        if not config_path:
            if hasattr(self, "_apply_recommended_ml_model_defaults"):
                self._apply_recommended_ml_model_defaults()
            if hasattr(self, "_refresh_ml_backend_status"):
                self._refresh_ml_backend_status()
            if hasattr(self, "_log_startup_ml_model_roots_once"):
                self._log_startup_ml_model_roots_once()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "ui_language" in config:
                from ui.i18n import set_language as _set_ui_lang
                _set_ui_lang(str(config.get("ui_language", "ko") or "ko"))

            if "ui_theme_profile" in config and hasattr(self, "ui_theme_var") and hasattr(self, "_apply_ui_theme_profile"):
                saved_theme = normalize_theme_profile_name(
                    config.get("ui_theme_profile", DEFAULT_THEME_PROFILE),
                    default=DEFAULT_THEME_PROFILE,
                )
                current_theme = str(self.ui_theme_var.get() or "").strip()
                if current_theme != saved_theme:
                    self._apply_ui_theme_profile(saved_theme, persist=False, rebuild=True)
                    return
                self.ui_theme_var.set(saved_theme)
            saved_preset_key = str(config.get("voicebank_tuning_preset_key", "") or "").strip().lower()
            if saved_preset_key in {
                "soft:aggressive",
                "soft:conservative",
                "normal:aggressive",
                "normal:conservative",
            }:
                self._selected_voicebank_preset_key = saved_preset_key
            else:
                self._selected_voicebank_preset_key = ""
            if hasattr(self, "_sync_voicebank_preset_button_styles"):
                self._sync_voicebank_preset_button_styles()

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
            if "gen_dash_alias" in config and hasattr(self, "gen_dash_alias_var"):
                self.gen_dash_alias_var.set(bool(config.get("gen_dash_alias", True)))
            if "no_base_oto" in config and hasattr(self, "no_base_oto_var"):
                self.no_base_oto_var.set(bool(config.get("no_base_oto", False)))
            if "no_mfa_oto_mode" in config and hasattr(self, "no_mfa_oto_mode_var"):
                if hasattr(self, "_set_no_mfa_oto_mode_from_code"):
                    self._set_no_mfa_oto_mode_from_code(config.get("no_mfa_oto_mode", "remap"))
                else:
                    self.no_mfa_oto_mode_var.set(str(config.get("no_mfa_oto_mode", "remap") or "remap"))
            if "soft_bank_mode" in config and hasattr(self, "soft_bank_mode_var"):
                self.soft_bank_mode_var.set(bool(config.get("soft_bank_mode", False)))
            if "low_rms_gain_enable" in config and hasattr(self, "low_rms_gain_enable_var"):
                self.low_rms_gain_enable_var.set(bool(config.get("low_rms_gain_enable", True)))
            if "weak_voice_assist_enable" in config and hasattr(self, "weak_voice_assist_enable_var"):
                self.weak_voice_assist_enable_var.set(bool(config.get("weak_voice_assist_enable", True)))
            if "weak_voice_assist_strength" in config and hasattr(self, "weak_voice_assist_strength_var"):
                self.weak_voice_assist_strength_var.set(str(config.get("weak_voice_assist_strength", "") or ""))
            if "mapping_strict_mode" in config and hasattr(self, "mapping_strict_mode_var"):
                if hasattr(self, "_set_mapping_strict_mode_from_code"):
                    self._set_mapping_strict_mode_from_code(config.get("mapping_strict_mode", "soft"))
                else:
                    self.mapping_strict_mode_var.set(str(config.get("mapping_strict_mode", "soft") or "soft"))
            if "low_conf_force_lock_mode" in config and hasattr(self, "low_conf_force_lock_mode_var"):
                self.low_conf_force_lock_mode_var.set(bool(config.get("low_conf_force_lock_mode", False)))
            if "weak_boundary_reduce_missing" in config and hasattr(self, "weak_boundary_reduce_missing_var"):
                self.weak_boundary_reduce_missing_var.set(bool(config.get("weak_boundary_reduce_missing", False)))
            if "weak_boundary_block_mismap" in config and hasattr(self, "weak_boundary_block_mismap_var"):
                self.weak_boundary_block_mismap_var.set(bool(config.get("weak_boundary_block_mismap", False)))
            if "recursive_voicebank_scan" in config and hasattr(self, "recursive_voicebank_scan_var"):
                self.recursive_voicebank_scan_var.set(bool(config.get("recursive_voicebank_scan", False)))
            if hasattr(self, "enable_ml_correction_var"):
                self.enable_ml_correction_var.set(bool(config.get("enable_ml_correction", True)))
            if hasattr(self, "cvn_correction_enable_var"):
                self.cvn_correction_enable_var.set(bool(config.get("cvn_correction_enable", True)))
            if hasattr(self, "cvn_low_conf_only_var"):
                self.cvn_low_conf_only_var.set(bool(config.get("cvn_low_conf_only", False)))
            if hasattr(self, "mapping_supervised_enable_var"):
                self.mapping_supervised_enable_var.set(bool(config.get("mapping_supervised_enable", True)))
            if hasattr(self, "mapping_supervised_mode_var"):
                if hasattr(self, "_set_mapping_supervised_mode_from_code"):
                    self._set_mapping_supervised_mode_from_code(config.get("mapping_supervised_mode", "auto"))
                else:
                    self.mapping_supervised_mode_var.set(str(config.get("mapping_supervised_mode", "auto") or "auto"))
            if hasattr(self, "cv_order_prior_enable_var"):
                self.cv_order_prior_enable_var.set(bool(config.get("cv_order_prior_enable", True)))
            if hasattr(self, "cv_order_prior_strength_var"):
                self.cv_order_prior_strength_var.set(str(config.get("cv_order_prior_strength", "") or ""))
            if "ml_route" in config and hasattr(self, "ml_route_var"):
                saved_route = self._normalize_ml_route_code(config.get("ml_route", "auto"))
                if hasattr(self, "_set_ml_route_from_code"):
                    self._set_ml_route_from_code(saved_route)
                else:
                    self.ml_route_var.set(saved_route)

            if "language" in config and hasattr(self, "lang_var"):
                saved_language = str(config.get("language", "") or "").strip()
                saved_language_lower = saved_language.lower()
                if "english" in saved_language_lower:
                    if str(getattr(self, "release_channel", "stable")).strip().lower() == "preview":
                        self.lang_var.set("English (Preview CVVC)")
                    else:
                        self.lang_var.set("한국어 (CV/연단음/CVVC/연속음 자동 매핑)")
                elif "japanese" in saved_language_lower:
                    self.lang_var.set("日本語 (CV/연단음/CVVC/연속음 자동 매핑)")
                elif "korean" in saved_language_lower:
                    self.lang_var.set("한국어 (CV/연단음/CVVC/연속음 자동 매핑)")

            if "en_cvvc_pack" in config and hasattr(self, "en_cvvc_pack_var"):
                pack = str(config.get("en_cvvc_pack", "LITE") or "LITE").strip().upper()
                if pack not in {"LITE", "FULL_GA"}:
                    pack = "LITE"
                self.en_cvvc_pack_var.set(pack)
            if "en_cvvc_beat" in config and hasattr(self, "en_cvvc_beat_var"):
                beat_raw = str(config.get("en_cvvc_beat", "8-beat") or "8-beat").strip().lower()
                self.en_cvvc_beat_var.set("4-beat" if beat_raw.startswith("4") else "8-beat")
            if "en_cvvc_preset" in config and hasattr(self, "en_cvvc_preset_var"):
                preset_raw = str(config.get("en_cvvc_preset", "Core") or "Core").strip().lower()
                if preset_raw in {"core+", "core_plus", "plus", "recommended"}:
                    self.en_cvvc_preset_var.set("Core+")
                elif preset_raw in {"all", "all+alt", "all_with_alt", "full"}:
                    self.en_cvvc_preset_var.set("All+Alt")
                else:
                    self.en_cvvc_preset_var.set("Core")
            if "en_cvvc_list_fallback" in config and hasattr(self, "en_cvvc_list_fallback_var"):
                self.en_cvvc_list_fallback_var.set(bool(config.get("en_cvvc_list_fallback", True)))

            lang = self._get_language() if hasattr(self, "_get_language") else "korean"
            saved_auto = config.get("auto_format", "CVVC")
            saved_auto_code = normalize_auto_format_value(lang, saved_auto)
            if hasattr(self, "_set_auto_format_from_code"):
                self._set_auto_format_from_code(saved_auto_code, lang)
            if "ja_alias_style" in config:
                saved_style = str(config.get("ja_alias_style", "원본 그대로") or "원본 그대로").strip()
                if saved_style in {"원본 그대로", "히라가나", "로마자"}:
                    self.ja_alias_style_var.set(saved_style)
            if "aligner" in config and hasattr(self, "aligner_var"):
                try:
                    from core.pipeline_status import normalize_aligner_name

                    saved_aligner = normalize_aligner_name(config.get("aligner", "mfa"), default="mfa")
                except Exception:
                    saved_aligner = "mfa"
                aligner_label_map = {
                    "none": "MFA",
                    "sequence": "전용(시퀀스)",
                    "coarse_crnn": "CRNN(실험적)",
                    "mfa": "MFA",
                }
                self.aligner_var.set(aligner_label_map.get(saved_aligner, "MFA"))
            if "developer_mode_enabled" in config and hasattr(self, "developer_mode_enabled_var"):
                allow_persist_dev = str(os.environ.get("UTOA_ALLOW_PERSISTENT_DEVELOPER_MODE", "")).strip().lower() in {
                    "1", "true", "yes", "on"
                }
                release_channel = str(getattr(self, "release_channel", "") or "").strip().lower()
                force_default_off = release_channel in {"stable", "preview"}
                saved_dev_mode = bool(config.get("developer_mode_enabled", False))
                self.developer_mode_enabled_var.set(
                    saved_dev_mode if (allow_persist_dev or not force_default_off) else False
                )
            if hasattr(self, "show_advanced_aligner_var"):
                self.show_advanced_aligner_var.set(False)
            if hasattr(self, "mfa_align_profile_var"):
                saved_profile_raw = str(
                    config.get(
                        "mfa_align_profile_code",
                        config.get("mfa_align_profile", "\uae30\ubcf8"),
                    )
                    or ""
                ).strip()
                compact = saved_profile_raw.lower().replace(" ", "")
                if compact in {
                    "\ube60\ub984".replace(" ", ""),
                    "\ube60\ub984(\uc800\uc0ac\uc591\ucd94\ucc9c)".replace(" ", ""),
                    "fast",
                    "quick",
                    "lite",
                    "speed",
                }:
                    self.mfa_align_profile_var.set("\ube60\ub984")
                elif compact in {
                    "\uc815\ubc00+\ud654\uc790\uc801\uc751".replace(" ", ""),
                    "\uc815\ud655\ub3c4\uc6b0\uc120+\ud654\uc790\uc801\uc751".replace(" ", ""),
                    "accurate_adapted",
                    "speaker_adapted",
                    "speaker_adaptation",
                    "adapt",
                    "speaker",
                }:
                    self.mfa_align_profile_var.set("\uc815\ubc00 + \ud654\uc790 \uc801\uc751")
                elif compact in {
                    "\uc815\ubc00".replace(" ", ""),
                    "\uc815\ud655\ub3c4\uc6b0\uc120".replace(" ", ""),
                    "accurate",
                }:
                    self.mfa_align_profile_var.set("\uc815\ubc00")
                else:
                    self.mfa_align_profile_var.set("\uae30\ubcf8")
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
                saved_selector_mode = str(config.get("ml_selector_mode", "selector") or "").strip()
                normalized_mode = self._normalize_ml_selector_mode(saved_selector_mode)
                label_map = {
                    "policy": "+셀렉터",
                    "delta": "상대적 보정",
                    "selector": "+셀렉터",
                }
                self.ml_selector_mode_var.set(label_map.get(normalized_mode, "+셀렉터"))
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
                            if abs(conf - 0.42) <= 1e-9:
                                self.ml_coupled_min_conf_var.set("")
                            else:
                                self.ml_coupled_min_conf_var.set(f"{conf:.2f}".rstrip("0").rstrip("."))
                        except Exception:
                            self.ml_coupled_min_conf_var.set("")
            if "ml_coupled_min_conf_use_model_meta" in config and hasattr(self, "ml_coupled_min_conf_use_model_meta_var"):
                self.ml_coupled_min_conf_use_model_meta_var.set(bool(config.get("ml_coupled_min_conf_use_model_meta", True)))
            if "ml_coupled_min_conf_model_offset" in config and hasattr(self, "ml_coupled_min_conf_model_offset_var"):
                raw_offset = config.get("ml_coupled_min_conf_model_offset", "")
                if raw_offset is None:
                    self.ml_coupled_min_conf_model_offset_var.set("")
                else:
                    txt = str(raw_offset).strip()
                    if not txt:
                        self.ml_coupled_min_conf_model_offset_var.set("")
                    else:
                        try:
                            val = max(-0.3, min(0.3, float(txt)))
                            if abs(val) <= 1e-9:
                                self.ml_coupled_min_conf_model_offset_var.set("")
                            else:
                                self.ml_coupled_min_conf_model_offset_var.set(
                                    f"{val:.3f}".rstrip("0").rstrip(".")
                                )
                        except Exception:
                            self.ml_coupled_min_conf_model_offset_var.set("")

            threshold_vars = [
                ("ml_coupled_min_conf_kr_cv", "ml_coupled_min_conf_kr_cv_var"),
                ("ml_coupled_min_conf_kr_cvc", "ml_coupled_min_conf_kr_cvc_var"),
                ("ml_coupled_min_conf_kr_cvvc", "ml_coupled_min_conf_kr_cvvc_var"),
                ("ml_coupled_min_conf_kr_vcv", "ml_coupled_min_conf_kr_vcv_var"),
                ("ml_coupled_min_conf_ja_cv", "ml_coupled_min_conf_ja_cv_var"),
                ("ml_coupled_min_conf_ja_cvc", "ml_coupled_min_conf_ja_cvc_var"),
                ("ml_coupled_min_conf_ja_cvvc", "ml_coupled_min_conf_ja_cvvc_var"),
                ("ml_coupled_min_conf_ja_vcv", "ml_coupled_min_conf_ja_vcv_var"),
            ]
            for conf_key, var_name in threshold_vars:
                if conf_key not in config or not hasattr(self, var_name):
                    continue
                var = getattr(self, var_name)
                raw_conf = config.get(conf_key, "")
                if raw_conf is None:
                    var.set("")
                    continue
                txt = str(raw_conf).strip()
                if not txt:
                    var.set("")
                    continue
                try:
                    conf = max(0.0, min(1.0, float(txt)))
                    var.set(f"{conf:.3f}".rstrip("0").rstrip("."))
                except Exception:
                    var.set("")
            if "ml_batch_inference_enable" in config and hasattr(self, "ml_batch_inference_enable_var"):
                self.ml_batch_inference_enable_var.set(bool(config.get("ml_batch_inference_enable", True)))
            if "ml_batch_inference_size" in config and hasattr(self, "ml_batch_inference_size_var"):
                raw_size = str(config.get("ml_batch_inference_size", "256") or "").strip()
                if not raw_size:
                    self.ml_batch_inference_size_var.set("256")
                else:
                    try:
                        val = max(32, min(4096, int(float(raw_size))))
                        self.ml_batch_inference_size_var.set(str(val))
                    except Exception:
                        self.ml_batch_inference_size_var.set("256")
            if "ml_legacy_fallback_enable" in config and hasattr(self, "ml_legacy_fallback_enable_var"):
                self.ml_legacy_fallback_enable_var.set(bool(config.get("ml_legacy_fallback_enable", True)))
            if "ml_hybrid_routing_enable" in config and hasattr(self, "ml_hybrid_routing_enable_var"):
                self.ml_hybrid_routing_enable_var.set(bool(config.get("ml_hybrid_routing_enable", True)))
            if "ml_e2e_enable" in config and hasattr(self, "ml_e2e_enable_var"):
                self.ml_e2e_enable_var.set(bool(config.get("ml_e2e_enable", False)))
            if "ml_e2e_mode" in config and hasattr(self, "ml_e2e_mode_var"):
                if hasattr(self, "_set_e2e_mode_from_code"):
                    self._set_e2e_mode_from_code(config.get("ml_e2e_mode", "hybrid"))
                else:
                    self.ml_e2e_mode_var.set(str(config.get("ml_e2e_mode", "hybrid") or "hybrid"))
            if "ml_e2e_t_low" in config and hasattr(self, "ml_e2e_t_low_var"):
                raw_val = config.get("ml_e2e_t_low", "")
                if raw_val is None:
                    self.ml_e2e_t_low_var.set("")
                else:
                    txt = str(raw_val).strip()
                    if not txt:
                        self.ml_e2e_t_low_var.set("")
                    else:
                        try:
                            val = max(0.0, min(1.0, float(txt)))
                            if abs(val - 0.52) <= 1e-9:
                                self.ml_e2e_t_low_var.set("")
                            else:
                                self.ml_e2e_t_low_var.set(f"{val:.3f}".rstrip("0").rstrip("."))
                        except Exception:
                            self.ml_e2e_t_low_var.set("")
            if "ml_e2e_t_high" in config and hasattr(self, "ml_e2e_t_high_var"):
                raw_val = config.get("ml_e2e_t_high", "")
                if raw_val is None:
                    self.ml_e2e_t_high_var.set("")
                else:
                    txt = str(raw_val).strip()
                    if not txt:
                        self.ml_e2e_t_high_var.set("")
                    else:
                        try:
                            val = max(0.0, min(1.0, float(txt)))
                            if abs(val - 0.72) <= 1e-9:
                                self.ml_e2e_t_high_var.set("")
                            else:
                                self.ml_e2e_t_high_var.set(f"{val:.3f}".rstrip("0").rstrip("."))
                        except Exception:
                            self.ml_e2e_t_high_var.set("")
            if "ml_e2e_blend_alpha" in config and hasattr(self, "ml_e2e_blend_alpha_var"):
                raw_val = config.get("ml_e2e_blend_alpha", "")
                if raw_val is None:
                    self.ml_e2e_blend_alpha_var.set("")
                else:
                    txt = str(raw_val).strip()
                    if not txt:
                        self.ml_e2e_blend_alpha_var.set("")
                    else:
                        try:
                            val = max(0.0, min(1.0, float(txt)))
                            if abs(val - 0.60) <= 1e-9:
                                self.ml_e2e_blend_alpha_var.set("")
                            else:
                                self.ml_e2e_blend_alpha_var.set(f"{val:.3f}".rstrip("0").rstrip("."))
                        except Exception:
                            self.ml_e2e_blend_alpha_var.set("")
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
            if "kr_continuity_enable" in config and hasattr(self, "kr_continuity_enable_var"):
                self.kr_continuity_enable_var.set(bool(config.get("kr_continuity_enable", True)))
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
            if "kr_vc_neighbor_enable" in config and hasattr(self, "kr_vc_neighbor_enable_var"):
                self.kr_vc_neighbor_enable_var.set(bool(config.get("kr_vc_neighbor_enable", True)))
            if "ja_vc_neighbor_enable" in config and hasattr(self, "ja_vc_neighbor_enable_var"):
                self.ja_vc_neighbor_enable_var.set(bool(config.get("ja_vc_neighbor_enable", True)))
            if "vc_correction_enable" in config:
                master_enabled = bool(config.get("vc_correction_enable", True))
                if hasattr(self, "vc_correction_enable_var"):
                    self.vc_correction_enable_var.set(master_enabled)
                if hasattr(self, "kr_continuity_enable_var"):
                    self.kr_continuity_enable_var.set(master_enabled)
                if hasattr(self, "kr_vc_neighbor_enable_var"):
                    self.kr_vc_neighbor_enable_var.set(master_enabled)
                if hasattr(self, "ja_vc_neighbor_enable_var"):
                    self.ja_vc_neighbor_enable_var.set(master_enabled)
            vc_neighbor_defaults = {
                "kr_vc_neighbor_blend": (self.kr_vc_neighbor_blend_var, 0.35),
                "kr_vc_neighbor_max_shift": (self.kr_vc_neighbor_max_shift_var, 45.0),
                "kr_vc_neighbor_lead_ms": (self.kr_vc_neighbor_lead_ms_var, 6.0),
                "kr_vc_neighbor_tail_ms": (self.kr_vc_neighbor_tail_ms_var, 8.0),
                "kr_vc_neighbor_min_len": (self.kr_vc_neighbor_min_len_var, 35.0),
                "ja_vc_neighbor_blend": (self.ja_vc_neighbor_blend_var, 0.35),
                "ja_vc_neighbor_max_shift": (self.ja_vc_neighbor_max_shift_var, 45.0),
                "ja_vc_neighbor_lead_ms": (self.ja_vc_neighbor_lead_ms_var, 6.0),
                "ja_vc_neighbor_tail_ms": (self.ja_vc_neighbor_tail_ms_var, 8.0),
                "ja_vc_neighbor_min_len": (self.ja_vc_neighbor_min_len_var, 35.0),
            }
            for key, (var, default_val) in vc_neighbor_defaults.items():
                if key not in config:
                    continue
                raw_val = config.get(key, "")
                if raw_val is None:
                    var.set("")
                    continue
                txt = str(raw_val).strip()
                if not txt:
                    var.set("")
                    continue
                try:
                    val = float(txt)
                    if abs(val - float(default_val)) <= 1e-9:
                        var.set("")
                    else:
                        var.set(f"{val:.3f}".rstrip("0").rstrip("."))
                except Exception:
                    var.set("")
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
                self.ml_coupled_strict_constraint_var.set(bool(config.get("ml_coupled_strict_constraint", True)))
            if "ml_model_root_kr" in config and hasattr(self, "ml_model_root_kr_var"):
                self.ml_model_root_kr_var.set(str(config.get("ml_model_root_kr", "") or ""))
            if "ml_model_root_ja" in config and hasattr(self, "ml_model_root_ja_var"):
                self.ml_model_root_ja_var.set(str(config.get("ml_model_root_ja", "") or ""))
            if "oto_crnn_model_path" in config and hasattr(self, "oto_crnn_model_path_var"):
                self.oto_crnn_model_path_var.set("")
            if "oto_crnn_engine" in config and hasattr(self, "oto_crnn_engine_var"):
                if hasattr(self, "_set_oto_crnn_engine_from_code"):
                    self._set_oto_crnn_engine_from_code(config.get("oto_crnn_engine", "boundary_decoder"))
                else:
                    self.oto_crnn_engine_var.set(str(config.get("oto_crnn_engine", "boundary_decoder") or "boundary_decoder"))
            if "oto_crnn_device" in config and hasattr(self, "oto_crnn_device_var"):
                device = str(config.get("oto_crnn_device", "auto") or "auto").strip().lower()
                self.oto_crnn_device_var.set(device if device in {"auto", "cpu", "cuda"} else "auto")
            if "oto_crnn_special_aliases" in config and hasattr(self, "oto_crnn_special_aliases_var"):
                self.oto_crnn_special_aliases_var.set(str(config.get("oto_crnn_special_aliases", "") or ""))

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
            if hasattr(self, "_sync_developer_mode_ui"):
                self._sync_developer_mode_ui()
            if hasattr(self, "_sync_vc_correction_toggle"):
                self._sync_vc_correction_toggle()
            if hasattr(self, "_sync_cvn_correction_toggle"):
                self._sync_cvn_correction_toggle()
            if hasattr(self, "_sync_advanced_tuning_slider_controls"):
                self._sync_advanced_tuning_slider_controls()
            if hasattr(self, "_sync_weak_voice_assist_controls"):
                self._sync_weak_voice_assist_controls()
            if hasattr(self, "_log_startup_ml_model_roots_once"):
                self._log_startup_ml_model_roots_once()

            if "params" in config and hasattr(self, "param_vars") and isinstance(self.param_vars, dict):
                params = config["params"]
                for key, val in params.items():
                    if key in self.param_vars:
                        self.param_vars[key].set(val)
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {e}")


