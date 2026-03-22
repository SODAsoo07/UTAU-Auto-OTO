"""
MFA (Montreal Forced Aligner) 실행 모듈
- 로컬 또는 포터블 Conda 환경에서 MFA 실행
- 실시간 로그 스트리밍
"""

import os
import sys
import subprocess
import re
import logging
import shutil
import hashlib
import tempfile
import locale
import math
import wave
import audioop
from typing import Sequence

from core.pipeline_status import (
    ALIGN_EXEC_MISSING,
    ALIGN_MODEL_MISSING,
    OK,
    make_runtime_report,
)

logger = logging.getLogger(__name__)

ALERT_MSVC_REQUIRED = "__ALERT__MSVC_REQUIRED__"
ALERT_MFA_PERMISSION_DENIED = "__ALERT__MFA_PERMISSION_DENIED__"
MSVC_REQUIRED_TEXT = "microsoft visual c++ 14.0 or greater is required"
MFA_PORTABLE_PYTHON_VERSION = "3.10"
KOREAN_WHEEL_DIRNAME = "mfa_ko_wheels"
_MFA_SINGLE_SPEAKER_FLAG_CACHE = {}
_MFA_SPEAKER_ADAPT_FLAG_CACHE = {}
_MFA_BREATH_WORD_RE = re.compile(r"(?i)^breath\d*$")
_PACKAGING_STACK_IMPORT_EXPR = (
    "import importlib.util as _u; import pip; import wheel; "
    "import sys; sys.exit(0 if _u.find_spec('setuptools') else 1)"
)
_KOREAN_NATIVE_RUNTIME_CHECK_EXPR = (
    "ok=False\n"
    "try:\n"
    "    from mecab import MeCab as _PyMecab\n"
    "    _tok = _PyMecab()\n"
    "    _out = _tok.parse('테스트')\n"
    "    ok = isinstance(_out, list)\n"
    "except Exception:\n"
    "    ok = False\n"
    "if not ok:\n"
    "    try:\n"
    "        import MeCab as _MeCabMod\n"
    "        _tagger = _MeCabMod.Tagger()\n"
    "        _tagger.parse('테스트')\n"
    "        ok = True\n"
    "    except Exception:\n"
    "        ok = False\n"
    "import sys\n"
    "sys.exit(0 if ok else 1)\n"
)
_KOREAN_TOKENIZER_IMPORT_CHECK_EXPR = (
    "import jamo\n"
    "ok=False\n"
    "try:\n"
    "    from mecab import MeCab as _PyMecab\n"
    "    _tok = _PyMecab()\n"
    "    _out = _tok.parse('테스트')\n"
    "    ok = isinstance(_out, list)\n"
    "except Exception:\n"
    "    ok = False\n"
    "if not ok:\n"
    "    try:\n"
    "        import MeCab as _MeCabMod\n"
    "        _tagger = _MeCabMod.Tagger()\n"
    "        _tagger.parse('테스트')\n"
    "        ok = True\n"
    "    except Exception:\n"
    "        ok = False\n"
    "if not ok:\n"
    "    try:\n"
    "        import eunjeon\n"
    "        _tok = eunjeon.Mecab()\n"
    "        _m = _tok.morphs('테스트')\n"
    "        ok = isinstance(_m, list)\n"
    "    except Exception:\n"
    "        ok = False\n"
    "import sys\n"
    "sys.exit(0 if ok else 1)\n"
)

MFA_ALIGN_PROFILE_PRESETS = {
    # Stable default profile (legacy accurate behavior).
    "default": {
        "clean": True,
        "fine_tune": True,
        "textgrid_cleanup": True,
        "beam": 1000,
        "retry_beam": 4000,
        "num_jobs": 1,
        "speaker_adaptation": False,
    },
    # Accuracy-first profile with speaker adaptation when supported by MFA.
    "accurate": {
        "clean": True,
        "fine_tune": True,
        "textgrid_cleanup": True,
        "beam": 1400,
        "retry_beam": 5600,
        "num_jobs": 1,
        "speaker_adaptation": True,
    },
    # Low-load profile for slower hardware.
    "fast": {
        "clean": True,
        "fine_tune": False,
        "textgrid_cleanup": True,
        "beam": 320,
        "retry_beam": 960,
        "num_jobs": 1,
        "speaker_adaptation": False,
    },
}


def _preferred_subprocess_encoding():
    try:
        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        return "utf-8"


def _subprocess_decode_candidates() -> list[str]:
    candidates: list[str] = []
    for enc in (
        "utf-8",
        "cp949",
        "cp932",
        "mbcs",
        _preferred_subprocess_encoding(),
        getattr(locale, "getencoding", lambda: "")() or "",
    ):
        enc = str(enc or "").strip()
        if enc and enc not in candidates:
            candidates.append(enc)
    return candidates


def _score_decoded_subprocess_text(text: str) -> int:
    score = 0
    for ch in str(text or ""):
        code = ord(ch)
        if ch == "\ufffd":
            score -= 20
        elif 0x20 <= code <= 0x7E or ch in "\r\n\t":
            score += 1
        elif 0xAC00 <= code <= 0xD7A3:  # Hangul syllables
            score += 4
        elif 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:  # Kana/CJK
            score += 3
        elif 0xFF61 <= code <= 0xFF9F:  # Halfwidth katakana mojibake hotspot
            score -= 6
        elif code < 0x20:
            score -= 10
        else:
            score += 0
    return score


def _decode_subprocess_output(data) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    raw = bytes(data)
    best_text = ""
    best_score = None
    for enc in _subprocess_decode_candidates():
        try:
            decoded = raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
        score = _score_decoded_subprocess_text(decoded)
        if best_score is None or score > best_score:
            best_text = decoded
            best_score = score
    if best_score is not None:
        return best_text
    return raw.decode("utf-8", errors="replace")


def _subprocess_window_kwargs() -> dict:
    """Hide helper console windows on Windows GUI runs."""
    if os.name != "nt":
        return {}
    kwargs: dict = {}
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    try:
        kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    except Exception:
        pass
    return kwargs


def _run_subprocess_text(args: Sequence[str], **kwargs):
    window_kwargs = _subprocess_window_kwargs()
    for key, value in window_kwargs.items():
        kwargs.setdefault(key, value)
    completed = subprocess.run(args, capture_output=True, text=False, **kwargs)
    completed.stdout = _decode_subprocess_output(getattr(completed, "stdout", b""))
    completed.stderr = _decode_subprocess_output(getattr(completed, "stderr", b""))
    return completed


def _popen_subprocess(args: Sequence[str], **kwargs):
    window_kwargs = _subprocess_window_kwargs()
    for key, value in window_kwargs.items():
        kwargs.setdefault(key, value)
    return subprocess.Popen(args, **kwargs)


def _contains_non_ascii(text):
    try:
        return any(ord(ch) > 127 for ch in str(text or ""))
    except Exception:
        return False


def _path_requires_ascii_fallback(path: str) -> bool:
    value = str(path or "")
    if not value:
        return False
    if _contains_non_ascii(value):
        return True
    # cmd.exe launcher wrappers are fragile with these characters in root paths.
    return any(ch in value for ch in "!&|<>()^")


def _default_mfa_root_dir(mfa_path="", per_process: bool = False):
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    elif mfa_path:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(app_dir, ".mfa_root_ascii")
    if per_process:
        root = f"{root}_p{os.getpid()}"
    os.makedirs(root, exist_ok=True)
    return root


def _seed_mfa_pretrained_models(dst_root: str, src_root: str) -> None:
    """Ensure per-process MFA root contains pretrained_models copied from shared root."""
    if not dst_root or not src_root:
        return
    try:
        dst_root = os.path.abspath(dst_root)
        src_root = os.path.abspath(src_root)
        if dst_root == src_root:
            return
        src_models = os.path.join(src_root, "pretrained_models")
        dst_models = os.path.join(dst_root, "pretrained_models")
        if not os.path.isdir(src_models):
            return
        dst_acoustic = os.path.join(dst_models, "acoustic")
        if os.path.isdir(dst_models) and os.path.isdir(dst_acoustic):
            return
        os.makedirs(dst_root, exist_ok=True)
        shutil.copytree(src_models, dst_models, dirs_exist_ok=True)
    except Exception as exc:
        logger.warning(f"[MFA] Failed to seed pretrained_models: {exc}")


def _resolve_env_python_exe(env_dir: str) -> str:
    candidates = [
        os.path.join(env_dir, "python.exe"),
        os.path.join(env_dir, "Scripts", "python.exe"),
        os.path.join(env_dir, "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def _is_env_scripts_path(path: str) -> bool:
    norm = str(path or "").replace("/", "\\").lower()
    return "\\scripts\\" in norm


def _candidate_shared_mfa_roots() -> list[str]:
    roots: list[str] = []
    env_override = str(os.environ.get("UTOA_MFA_SHARED_ROOT", "") or "").strip()
    if env_override:
        roots.append(env_override)
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        roots.append(os.path.join(local_app_data, "UTAU_Auto_OTO_v3"))
    else:
        roots.append(os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3"))
    public_base = str(os.environ.get("PUBLIC", "") or "").strip()
    if not public_base:
        system_drive = str(os.environ.get("SystemDrive", "C:") or "C:").strip() or "C:"
        public_base = os.path.join(system_drive, "Users", "Public")
    public_root = os.path.join(public_base, "UTAU_Auto_OTO_v3")
    roots.append(public_root)

    dedup: list[str] = []
    seen = set()
    for root in roots:
        normalized = os.path.normcase(os.path.abspath(str(root or "")))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(root)
    return dedup


def _is_writable_directory(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".utoa_write_probe_", dir=path)
        os.close(fd)
        os.remove(probe)
        return True
    except Exception:
        return False


def _resolve_shared_mfa_root() -> str:
    candidates = _candidate_shared_mfa_roots()
    env_override = str(os.environ.get("UTOA_MFA_SHARED_ROOT", "") or "").strip()
    if not env_override and len(candidates) >= 2:
        local_candidate = candidates[0]
        if _path_requires_ascii_fallback(local_candidate):
            logger.info(
                "[MFA] LOCALAPPDATA runtime path contains non-ASCII/shell-sensitive characters; "
                "preferring ASCII-safe shared root."
            )
            candidates = candidates[1:] + [local_candidate]
    for root in candidates:
        if _is_writable_directory(root):
            return root
    return candidates[0] if candidates else os.path.join(os.path.expanduser("~"), "UTAU_Auto_OTO_v3")


def _candidate_korean_wheel_dirs(mfa_path: str = "") -> list[str]:
    candidates: list[str] = []
    env_override = str(os.environ.get("UTOA_KO_WHEEL_DIR", "") or "").strip()
    if env_override:
        candidates.append(env_override)

    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.extend(
            [
                os.path.join(exe_dir, KOREAN_WHEEL_DIRNAME),
                os.path.join(os.path.dirname(exe_dir), KOREAN_WHEEL_DIRNAME),
            ]
        )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend(
        [
            os.path.join(repo_root, KOREAN_WHEEL_DIRNAME),
            os.path.join(repo_root, "build_assets", KOREAN_WHEEL_DIRNAME),
        ]
    )

    if _is_env_scripts_path(mfa_path):
        try:
            env_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
            runtime_root = os.path.dirname(env_dir)
            candidates.append(os.path.join(runtime_root, KOREAN_WHEEL_DIRNAME))
        except Exception:
            pass

    dedup: list[str] = []
    seen = set()
    for path in candidates:
        norm = os.path.normcase(os.path.abspath(str(path or "")))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        dedup.append(path)
    return dedup


def _resolve_korean_wheel_dir(mfa_path: str = "") -> str:
    for path in _candidate_korean_wheel_dirs(mfa_path):
        if not os.path.isdir(path):
            continue
        try:
            if any(name.lower().endswith(".whl") for name in os.listdir(path)):
                return os.path.abspath(path)
        except Exception:
            continue
    return ""


def get_default_mfa_env_dir():
    return os.path.join(_resolve_shared_mfa_root(), ".env")


def get_default_mfa_conda_root():
    return os.path.join(_resolve_shared_mfa_root(), "miniconda")


def get_default_mfa_micromamba_root():
    return os.path.join(_resolve_shared_mfa_root(), "micromamba")


def get_default_mfa_micromamba_exe():
    root = get_default_mfa_micromamba_root()
    candidates = [
        os.path.join(root, "Library", "bin", "micromamba.exe"),
        os.path.join(root, "bin", "micromamba.exe"),
        os.path.join(root, "micromamba.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _candidate_mfa_executable_paths():
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shared_env_dirs = [os.path.join(root, ".env") for root in _candidate_shared_mfa_roots()]
    candidates = [
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.exe'),
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.bat'),
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.cmd'),
        os.path.join(app_dir, '.env', 'bin', 'mfa'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.exe'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.bat'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.cmd'),
    ]
    for shared_env_dir in shared_env_dirs:
        candidates.extend(
            [
                os.path.join(shared_env_dir, 'Scripts', 'mfa.exe'),
                os.path.join(shared_env_dir, 'Scripts', 'mfa.bat'),
                os.path.join(shared_env_dir, 'Scripts', 'mfa.cmd'),
                os.path.join(shared_env_dir, 'bin', 'mfa'),
            ]
        )
    seen = set()
    unique = []
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_mfa_launcher_failure_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    failure_markers = (
        "failed to create process",
        "unable to create process",
        "fatal error in launcher",
        "could not import runpy",
        "no python at",
        "is not recognized as an internal or external command",
    )
    return any(marker in lowered for marker in failure_markers)


def _probe_mfa_launcher(mfa_path: str) -> tuple[bool, str]:
    path = os.path.abspath(str(mfa_path or "").strip())
    if not path or not os.path.exists(path):
        return False, "mfa launcher missing"
    try:
        env = _get_conda_env(path)
        result = _run_subprocess_text(
            [path, "--help"],
            env=env,
            timeout=30,
        )
        combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        if _is_mfa_launcher_failure_text(combined):
            return False, combined
        if result.returncode == 0:
            return True, combined
        lowered = combined.lower()
        if "usage" in lowered and "mfa" in lowered:
            return True, combined
        return False, combined
    except subprocess.TimeoutExpired as exc:
        # Some sandbox/AV environments can make the launcher help probe very slow.
        # Fall back to python module probe before marking launcher as broken.
        msg = str(exc or "").strip() or "launcher probe timed out"
        lowered = path.lower()
        if "\\scripts\\" in lowered:
            env_dir = os.path.dirname(os.path.dirname(path))
            python_exe = _resolve_env_python_exe(env_dir)
            if python_exe and os.path.exists(python_exe):
                env = _get_conda_env(path)
                probe_cmds = [
                    [python_exe, "-m", "montreal_forced_aligner.command_line.mfa", "--help"],
                    [python_exe, "-m", "montreal_forced_aligner", "--help"],
                ]
                for cmd in probe_cmds:
                    try:
                        py_res = _run_subprocess_text(cmd, env=env, timeout=25)
                    except Exception:
                        continue
                    combined = f"{py_res.stdout or ''}\n{py_res.stderr or ''}".strip()
                    if _is_mfa_launcher_failure_text(combined):
                        continue
                    if py_res.returncode == 0:
                        return True, f"{msg}; python-module probe ok"
                    low = combined.lower()
                    if "usage" in low and "mfa" in low:
                        return True, f"{msg}; python-module usage probe ok"
        return False, msg
    except Exception as exc:
        msg = str(exc or "").strip()
        return False, msg or "launcher probe failed"


def _ensure_mfa_batch_wrapper(env_dir: str) -> str:
    if sys.platform != "win32":
        return ""
    root = os.path.abspath(str(env_dir or "").strip())
    if not root:
        return ""
    python_exe = _resolve_env_python_exe(root)
    if not python_exe or not os.path.exists(python_exe):
        return ""

    wrapper_path = os.path.join(root, "Scripts", "mfa.bat")
    try:
        os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
        with open(wrapper_path, "w", encoding="utf-8", newline="") as wf:
            wf.write("@echo off\r\n")
            # Keep wrapper ASCII-only and resolve env path relatively at runtime.
            # This avoids launcher breakage on non-ASCII install roots.
            wf.write('set "SCRIPT_DIR=%~dp0"\r\n')
            wf.write('set "ENV_DIR=%SCRIPT_DIR%.."\r\n')
            wf.write('for %%I in ("%ENV_DIR%") do set "ENV_DIR=%%~fI"\r\n')
            wf.write('set "CONDA_PREFIX=%ENV_DIR%"\r\n')
            wf.write(
                'set "PATH=%ENV_DIR%;%ENV_DIR%\\Library\\mingw-w64\\bin;'
                '%ENV_DIR%\\Library\\usr\\bin;%ENV_DIR%\\Library\\bin;'
                '%ENV_DIR%\\Scripts;%ENV_DIR%\\bin;%PATH%"\r\n'
            )
            wf.write('set "MFA_SCRIPT_PATH=%SCRIPT_DIR%mfa-script.py"\r\n')
            wf.write('set "MFA_ALT_SCRIPT_PATH=%SCRIPT_DIR%mfa.py"\r\n')
            wf.write('set "ENV_PY=%ENV_DIR%\\python.exe"\r\n')
            wf.write('if not exist "%ENV_PY%" set "ENV_PY=%ENV_DIR%\\Scripts\\python.exe"\r\n')
            wf.write('if not exist "%ENV_PY%" set "ENV_PY=%ENV_DIR%\\bin\\python"\r\n')
            wf.write("set \"_UTOA_MFA_EXIT=1\"\r\n")
            wf.write('if exist "%MFA_SCRIPT_PATH%" (\r\n')
            wf.write('  "%ENV_PY%" "%MFA_SCRIPT_PATH%" %*\r\n')
            wf.write('  set "_UTOA_MFA_EXIT=%ERRORLEVEL%"\r\n')
            wf.write(') else if exist "%MFA_ALT_SCRIPT_PATH%" (\r\n')
            wf.write('  "%ENV_PY%" "%MFA_ALT_SCRIPT_PATH%" %*\r\n')
            wf.write('  set "_UTOA_MFA_EXIT=%ERRORLEVEL%"\r\n')
            wf.write(')\r\n')
            wf.write("if not \"%_UTOA_MFA_EXIT%\"==\"0\" (\r\n")
            wf.write('  "%ENV_PY%" -m montreal_forced_aligner.command_line.mfa %*\r\n')
            wf.write("  set \"_UTOA_MFA_EXIT=%ERRORLEVEL%\"\r\n")
            wf.write(")\r\n")
            wf.write("if not \"%_UTOA_MFA_EXIT%\"==\"0\" (\r\n")
            wf.write('  "%ENV_PY%" -m montreal_forced_aligner %*\r\n')
            wf.write("  set \"_UTOA_MFA_EXIT=%ERRORLEVEL%\"\r\n")
            wf.write(")\r\n")
            wf.write("exit /b %_UTOA_MFA_EXIT%\r\n")
        return wrapper_path
    except Exception:
        return ""


def resolve_working_mfa_executable(mfa_path: str, callback=None) -> str:
    path = os.path.abspath(str(mfa_path or "").strip())
    if not path or not os.path.exists(path):
        return path

    def _emit(msg: str):
        logger.info(msg)
        if callback:
            callback(msg)

    healthy, detail = _probe_mfa_launcher(path)
    if detail:
        _emit(f"[MFA] launcher health check failed: {detail[:280]}")
    if healthy:
        lowered = path.lower()
        # If launcher health was recovered only via python module probe,
        # prefer the local batch wrapper to avoid fragile native launcher paths.
        if (
            sys.platform == "win32"
            and lowered.endswith("mfa.exe")
            and "\\scripts\\" in lowered
            and "python-module probe ok" in str(detail or "").lower()
        ):
            env_dir = os.path.dirname(os.path.dirname(path))
            wrapper = _ensure_mfa_batch_wrapper(env_dir)
            if wrapper and os.path.exists(wrapper):
                _emit(f"[MFA] launcher fallback switched to python wrapper: {wrapper}")
                return wrapper
        return path

    lowered = path.lower()
    if sys.platform == "win32" and lowered.endswith("mfa.exe") and "\\scripts\\" in lowered:
        env_dir = os.path.dirname(os.path.dirname(path))
        wrapper = _ensure_mfa_batch_wrapper(env_dir)
        if wrapper and os.path.exists(wrapper):
            wrapper_ok, wrapper_detail = _probe_mfa_launcher(wrapper)
            if wrapper_ok:
                _emit(f"[MFA] mfa.exe launcher fallback enabled: {wrapper}")
                return wrapper
            if wrapper_detail:
                _emit(f"[MFA] mfa.bat launcher check failed: {wrapper_detail[:280]}")

    return path


def _link_or_copy(src, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _parse_float_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _resolve_low_rms_gain_config() -> dict:
    enabled = _parse_bool_env("UTOA_MFA_LOW_RMS_GAIN_ENABLE", True)
    threshold_db = _parse_float_env("UTOA_MFA_LOW_RMS_THRESHOLD_DB", -24.0)
    target_db = _parse_float_env("UTOA_MFA_LOW_RMS_TARGET_DB", -16.0)
    max_gain_db = _parse_float_env("UTOA_MFA_LOW_RMS_MAX_GAIN_DB", 18.0)
    peak_ceiling_db = _parse_float_env("UTOA_MFA_LOW_RMS_PEAK_CEILING_DB", -1.0)
    weak_voice_assist_enabled = _parse_bool_env("UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE", True)
    weak_voice_trigger_db = _parse_float_env("UTOA_MFA_WEAK_VOICE_TRIGGER_DB", -23.0)
    weak_voice_alpha = _parse_float_env("UTOA_MFA_WEAK_VOICE_PREEMPH_ALPHA", 0.92)
    weak_voice_mix = _parse_float_env("UTOA_MFA_WEAK_VOICE_PREEMPH_MIX", 0.35)

    threshold_db = max(-80.0, min(0.0, threshold_db))
    target_db = max(-40.0, min(-1.0, target_db))
    max_gain_db = max(0.0, min(24.0, max_gain_db))
    peak_ceiling_db = max(-12.0, min(-0.1, peak_ceiling_db))
    weak_voice_trigger_db = max(-80.0, min(-6.0, weak_voice_trigger_db))
    weak_voice_alpha = max(0.0, min(0.99, weak_voice_alpha))
    weak_voice_mix = max(0.0, min(1.0, weak_voice_mix))

    return {
        "enabled": bool(enabled),
        "threshold_db": float(threshold_db),
        "target_db": float(target_db),
        "max_gain_db": float(max_gain_db),
        "peak_ceiling_db": float(peak_ceiling_db),
        "weak_voice_assist_enabled": bool(weak_voice_assist_enabled),
        "weak_voice_trigger_db": float(weak_voice_trigger_db),
        "weak_voice_alpha": float(weak_voice_alpha),
        "weak_voice_mix": float(weak_voice_mix),
    }


def _dbfs(value: float, floor: float = 1e-8) -> float:
    return 20.0 * math.log10(max(float(value), float(floor)))


def _pcm_full_scale(sampwidth: int) -> float:
    if sampwidth <= 1:
        return 127.0
    return float((1 << ((int(sampwidth) * 8) - 1)) - 1)


def _apply_pcm_gain(raw: bytes, sampwidth: int, gain_lin: float) -> bytes:
    gain = float(gain_lin)
    if abs(gain - 1.0) < 1e-6:
        return raw
    if sampwidth == 1:
        centered = audioop.bias(raw, 1, -128)
        scaled = audioop.mul(centered, 1, gain)
        return audioop.bias(scaled, 1, 128)
    return audioop.mul(raw, int(sampwidth), gain)


def _pcm_sample_bounds(sampwidth: int) -> tuple[int, int]:
    if sampwidth == 1:
        return -128, 127
    bits = int(sampwidth) * 8
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    return lo, hi


def _apply_preemphasis_pcm(raw: bytes, sampwidth: int, n_channels: int, alpha: float, mix: float) -> tuple[bytes, bool]:
    if not raw or sampwidth <= 0 or n_channels <= 0:
        return raw, False
    frame_bytes = int(sampwidth) * int(n_channels)
    if frame_bytes <= 0 or len(raw) % frame_bytes != 0:
        return raw, False
    lo, hi = _pcm_sample_bounds(sampwidth)
    prev = [0.0] * int(n_channels)
    out = bytearray()
    try:
        mv = memoryview(raw)
        for frame_start in range(0, len(raw), frame_bytes):
            for ch in range(int(n_channels)):
                pos = frame_start + (ch * int(sampwidth))
                if sampwidth == 1:
                    x = int(mv[pos]) - 128
                else:
                    x = int.from_bytes(bytes(mv[pos:pos + int(sampwidth)]), "little", signed=True)
                y = float(x) - (float(alpha) * prev[ch])
                prev[ch] = float(x)
                z = ((1.0 - float(mix)) * float(x)) + (float(mix) * y)
                iv = int(round(z))
                if iv < lo:
                    iv = lo
                elif iv > hi:
                    iv = hi
                if sampwidth == 1:
                    out.append(iv + 128)
                else:
                    out.extend(int(iv).to_bytes(int(sampwidth), "little", signed=True))
        return bytes(out), True
    except Exception:
        return raw, False


def _copy_wav_with_low_rms_gain(src: str, dst: str, cfg: dict) -> dict:
    result = {
        "ok": False,
        "boosted": False,
        "weak_assist": False,
        "reason": "",
        "rms_db_before": None,
        "gain_db_applied": 0.0,
    }
    try:
        src_abs = os.path.normcase(os.path.abspath(str(src or "")))
        dst_abs = os.path.normcase(os.path.abspath(str(dst or "")))
    except Exception:
        src_abs = str(src or "")
        dst_abs = str(dst or "")
    if src_abs and dst_abs and src_abs == dst_abs:
        # Safety guard: never write gain-processed audio back to original source path.
        result["ok"] = True
        result["reason"] = "src_equals_dst_blocked"
        return result
    try:
        with wave.open(src, "rb") as rf:
            params = rf.getparams()
            n_channels = int(params.nchannels or 0)
            sampwidth = int(params.sampwidth or 0)
            comptype = str(params.comptype or "NONE").upper()
            raw = rf.readframes(int(params.nframes or 0))
    except Exception:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "wav_read_failed_passthrough"
        return result

    if comptype != "NONE" or sampwidth <= 0 or sampwidth > 4 or n_channels <= 0:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "unsupported_wav_passthrough"
        return result
    if not raw:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "empty_wav_passthrough"
        return result

    try:
        if n_channels == 1:
            mono = raw
        elif n_channels == 2:
            mono = audioop.tomono(raw, sampwidth, 0.5, 0.5)
        else:
            _link_or_copy(src, dst)
            result["ok"] = True
            result["reason"] = "multichannel_passthrough"
            return result
        rms = float(audioop.rms(mono, sampwidth))
    except Exception:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "rms_probe_failed_passthrough"
        return result

    if rms <= 0.0:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "silent_passthrough"
        return result

    full_scale = _pcm_full_scale(sampwidth)
    rms_db = _dbfs(rms / full_scale)
    result["rms_db_before"] = float(rms_db)

    threshold_db = float(cfg.get("threshold_db", -24.0))
    target_db = float(cfg.get("target_db", -16.0))
    max_gain_db = float(cfg.get("max_gain_db", 18.0))
    peak_ceiling_db = float(cfg.get("peak_ceiling_db", -1.0))
    weak_voice_assist_enabled = bool(cfg.get("weak_voice_assist_enabled", True))
    weak_voice_trigger_db = float(cfg.get("weak_voice_trigger_db", -23.0))
    weak_voice_alpha = float(cfg.get("weak_voice_alpha", 0.92))
    weak_voice_mix = float(cfg.get("weak_voice_mix", 0.35))

    if rms_db >= threshold_db:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "above_threshold_passthrough"
        return result

    gain_db = min(max_gain_db, max(0.0, target_db - rms_db))
    if gain_db <= 1e-4:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "no_gain_needed_passthrough"
        return result

    try:
        processed = _apply_pcm_gain(raw, sampwidth, 10.0 ** (gain_db / 20.0))
        if weak_voice_assist_enabled and rms_db <= weak_voice_trigger_db:
            processed, assisted = _apply_preemphasis_pcm(
                processed,
                sampwidth,
                n_channels,
                alpha=weak_voice_alpha,
                mix=weak_voice_mix,
            )
            if assisted:
                result["weak_assist"] = True
        peak = float(audioop.max(processed, sampwidth))
        peak_db = _dbfs(peak / full_scale) if peak > 0 else -120.0
        if peak_db > peak_ceiling_db:
            reduce_db = peak_ceiling_db - peak_db
            processed = _apply_pcm_gain(processed, sampwidth, 10.0 ** (reduce_db / 20.0))
            gain_db += reduce_db

        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with wave.open(dst, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(processed)
        result["ok"] = True
        result["boosted"] = True
        result["reason"] = "boosted"
        result["gain_db_applied"] = float(gain_db)
        return result
    except Exception:
        _link_or_copy(src, dst)
        result["ok"] = True
        result["reason"] = "gain_apply_failed_passthrough"
        return result


def _prepare_ascii_safe_alignment_workspace(wav_folder, dict_path, output_folder, *, wav_gain_config=None):
    token_src = "|".join([
        os.path.abspath(wav_folder or ""),
        os.path.abspath(dict_path or ""),
        os.path.abspath(output_folder or ""),
    ])
    token = hashlib.sha1(token_src.encode("utf-8", errors="replace")).hexdigest()[:12]
    base = os.path.join(tempfile.gettempdir(), "utoa_mfa_ascii", token)
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    corpus_dir = os.path.join(base, "corpus")
    dict_dir = os.path.join(base, "dict")
    out_dir = os.path.join(base, "out")
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    wav_gain_enabled = bool((wav_gain_config or {}).get("enabled", False))
    wav_gain_stats = {
        "enabled": wav_gain_enabled,
        "scanned": 0,
        "boosted": 0,
        "weak_assist": 0,
        "passthrough": 0,
    }

    for fn in os.listdir(wav_folder):
        low = fn.lower()
        src = os.path.join(wav_folder, fn)
        if not os.path.isfile(src):
            continue
        if low.endswith(".wav"):
            dst = os.path.join(corpus_dir, fn)
            if wav_gain_enabled:
                wav_gain_stats["scanned"] += 1
                outcome = _copy_wav_with_low_rms_gain(src, dst, wav_gain_config or {})
                if outcome.get("boosted"):
                    wav_gain_stats["boosted"] += 1
                else:
                    wav_gain_stats["passthrough"] += 1
                if outcome.get("weak_assist"):
                    wav_gain_stats["weak_assist"] += 1
            else:
                _link_or_copy(src, dst)
        elif low.endswith(".lab") or low.endswith(".txt"):
            if os.path.isfile(src):
                _link_or_copy(src, os.path.join(corpus_dir, fn))

    ext = os.path.splitext(dict_path)[1] or ".txt"
    safe_dict_path = os.path.join(dict_dir, f"dictionary{ext}")
    shutil.copy2(dict_path, safe_dict_path)
    return {
        "base": base,
        "corpus_dir": corpus_dir,
        "dict_path": safe_dict_path,
        "output_dir": out_dir,
        "wav_gain_stats": wav_gain_stats,
    }


def _copy_back_textgrids(safe_output_dir, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    copied = 0
    for dp, dns, fns in os.walk(safe_output_dir):
        rel = os.path.relpath(dp, safe_output_dir)
        dst_dir = output_folder if rel == "." else os.path.join(output_folder, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in fns:
            if not fn.lower().endswith(".textgrid"):
                continue
            shutil.copy2(os.path.join(dp, fn), os.path.join(dst_dir, fn))
            copied += 1
    return copied


def _stderr_has_msvc_requirement(text):
    if not text:
        return False
    lowered = text.lower()
    return (
        MSVC_REQUIRED_TEXT in lowered
        or "visual-cpp-build-tools" in lowered
        or "microsoft c++ build tools" in lowered
    )


def _emit_msvc_required_notice(callback, log_fn):
    if callback:
        callback(ALERT_MSVC_REQUIRED)
    log_fn("⚠ 일부 의존성 설치에 실패했습니다. 사용은 가능하지만 정확도에 영향이 있을 수 있으니 C++ 툴을 설치해주세요.")
    log_fn("   설치 링크: https://visualstudio.microsoft.com/visual-cpp-build-tools/")


def mfa_python_version_requires_downgrade(version_text: str) -> bool:
    text = str(version_text or "").strip()
    if not text:
        return False
    match = re.match(r"^\s*(\d+)\.(\d+)", text)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2))
    return major > 3 or (major == 3 and minor >= 13)


def get_mfa_env_python_version(mfa_path: str) -> str:
    if not mfa_path:
        return ""
    env_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
    python_exe = _resolve_env_python_exe(env_dir)
    if not python_exe:
        return ""
    try:
        env = _get_conda_env(mfa_path)
        result = _run_subprocess_text(
            [
                python_exe,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
            ],
            env=env,
            timeout=20,
        )
        if result.returncode != 0:
            return ""
        return str(result.stdout or "").strip().splitlines()[-1].strip()
    except Exception:
        return ""


def mfa_env_requires_python_downgrade(mfa_path: str) -> bool:
    return mfa_python_version_requires_downgrade(get_mfa_env_python_version(mfa_path))


def _preflight_compute_mfcc(mfa_path, callback=None):
    """MFA 정렬 시작 전에 compute-mfcc-feats 실행 가능 여부를 점검합니다."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        return False, "MFA 실행 파일 경로가 비어 있습니다."

    env = _get_conda_env(mfa_path)
    candidates = []
    if sys.platform == 'win32' and _is_env_scripts_path(mfa_path):
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        candidates.append(os.path.join(env_dir, 'Library', 'bin', 'compute-mfcc-feats.exe'))
        candidates.append('compute-mfcc-feats.exe')
    candidates.append('compute-mfcc-feats')

    last_not_found = None
    for candidate in candidates:
        try:
            # Windows + Python 3.13 조합에서는 확장자 없는 실행명 검색이 실패할 수 있다.
            _run_subprocess_text(
                [candidate, '--help'],
                timeout=15,
                env=env,
            )
            return True, ""
        except FileNotFoundError as e:
            last_not_found = e
            continue
        except PermissionError as e:
            if callback:
                callback(ALERT_MFA_PERMISSION_DENIED)
            err = (
                "compute-mfcc-feats 실행 권한이 없어 MFA 정렬을 시작할 수 없습니다. "
                "(WinError 5: Access denied)"
            )
            log(f"❌ {err}")
            log("   보안 프로그램/권한 정책/파일 차단 여부를 확인해 주세요.")
            return False, f"{err}: {e}"
        except Exception as e:
            err = f"compute-mfcc-feats 사전 점검 중 오류: {e}"
            log(f"❌ {err}")
            return False, err

    err = "compute-mfcc-feats를 찾지 못했습니다. MFA 환경이 손상되었을 수 있습니다."
    log(f"❌ {err}")
    if last_not_found:
        return False, f"{err}: {last_not_found}"
    return False, err


def _validate_alignment_dictionary(dict_path: str, callback=None):
    """Validate MFA dictionary rows before native graph compilation."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not dict_path or not os.path.isfile(dict_path):
        return False, f"Dictionary not found: {dict_path}"

    try:
        with open(dict_path, "rb") as f:
            raw = f.read()
        text = _decode_subprocess_output(raw)
    except Exception as e:
        return False, f"Failed to read dictionary: {e}"

    bad_lines = []
    total = 0
    for idx, line in enumerate(str(text or "").splitlines(), start=1):
        row = str(line or "").strip()
        if not row or row.startswith("#"):
            continue
        total += 1
        parts = row.split()
        # Expected: <word> <phone1> [phone2 ...]
        if len(parts) < 2 or "\ufffd" in row:
            bad_lines.append((idx, row))

    if bad_lines:
        samples = "; ".join(f"{ln}:{txt[:40]}" for ln, txt in bad_lines[:5])
        err = (
            f"Dictionary malformed rows detected ({len(bad_lines)}/{max(total, 1)}). "
            f"examples={samples}"
        )
        log(err)
        return False, err
    return True, ""


def _sanitize_alignment_dictionary_for_mfa(dict_path: str, callback=None):
    """Patch known crash-prone entries in dictionary before MFA graph compile."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not dict_path or not os.path.isfile(dict_path):
        return False, f"Dictionary not found: {dict_path}"
    try:
        with open(dict_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception as e:
        return False, f"Failed to read dictionary for sanitize: {e}"

    changed = 0
    out = []
    for line in lines:
        row = str(line or "").strip()
        if not row or row.startswith("#"):
            out.append(line)
            continue
        parts = row.split()
        if len(parts) >= 2 and _MFA_BREATH_WORD_RE.fullmatch(parts[0]) and parts[1].lower() in {"sil", "pau", "cl"}:
            out.append(f"{parts[0]} spn")
            changed += 1
            continue
        out.append(line)

    if changed <= 0:
        return True, ""
    try:
        with open(dict_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out).rstrip() + "\n")
    except Exception as e:
        return False, f"Failed to write sanitized dictionary: {e}"
    log(f"[MFA] Dictionary sanitize applied: breath->spn ({changed} rows)")
    return True, ""

def _get_conda_env(mfa_path):
    """
    Windows 환경에서 Conda 활성화 없이 mfa.exe를 직접 호출할 때 
    DLL 로드 에러(코드 3228369023)가 발생하는 것을 막기 위해 환경 변수에 PATH를 주입합니다.
    """
    env = os.environ.copy()
    # Avoid inheriting parent interpreter settings that can break env-local python imports.
    for key in (
        'PYTHONHOME',
        'PYTHONPATH',
        'PYTHONSTARTUP',
        'PYTHONUSERBASE',
        'VIRTUAL_ENV',
        '__PYVENV_LAUNCHER__',
    ):
        env.pop(key, None)
    env.setdefault('PYTHONNOUSERSITE', '1')
    if sys.platform == 'win32' and _is_env_scripts_path(mfa_path):
        mfa_path = os.path.abspath(mfa_path)
        env_dir = os.path.abspath(os.path.dirname(os.path.dirname(mfa_path)))
        site_packages = os.path.join(env_dir, 'Lib', 'site-packages')
        eunjeon_data = os.path.join(site_packages, 'eunjeon', 'data')
        mecabrc = os.path.join(site_packages, 'mecabrc')
        new_paths = [
            env_dir,
            os.path.join(env_dir, 'Library', 'mingw-w64', 'bin'),
            os.path.join(env_dir, 'Library', 'usr', 'bin'),
            os.path.join(env_dir, 'Library', 'bin'),
            os.path.join(env_dir, 'Scripts'),
            os.path.join(env_dir, 'bin'),
        ]
        if os.path.isdir(site_packages):
            new_paths.append(site_packages)
        if os.path.isdir(eunjeon_data):
            new_paths.append(eunjeon_data)
        current_path = env.get('PATH', '')
        env['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path
        env['CONDA_PREFIX'] = env_dir
        if os.path.exists(mecabrc):
            env.setdefault('MECABRC', mecabrc)
    mode = str(os.environ.get("UTOA_MFA_ROOT_DIR_MODE", "")).strip().lower()
    per_process = mode in {"per_process", "per-process", "process", "proc"}
    mfa_root = _default_mfa_root_dir(mfa_path, per_process=per_process)
    env.setdefault('MFA_ROOT_DIR', mfa_root)
    if per_process:
        shared_root = _default_mfa_root_dir(mfa_path, per_process=False)
        _seed_mfa_pretrained_models(mfa_root, shared_root)
    return env


def _check_env_imports(python_exe: str, env: dict, import_expr: str):
    res = _run_subprocess_text([python_exe, '-c', import_expr], env=env)
    if res.returncode == 0:
        return True, ''
    detail = (res.stderr or res.stdout or '').strip()
    return False, detail


def _check_packaging_stack(python_exe: str, env: dict):
    return _check_env_imports(python_exe, env, _PACKAGING_STACK_IMPORT_EXPR)


def ensure_mfa_python_packaging_stack(mfa_path, callback=None):
    """
    Ensure pip/setuptools/wheel are available inside the MFA env.
    This is a prerequisite for language-specific dependency repair on native Windows.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or not _is_env_scripts_path(mfa_path):
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
    if not os.path.exists(python_exe):
        return False

    env = _get_conda_env(mfa_path)
    ok, _detail = _check_packaging_stack(python_exe, env)
    if ok:
        return True

    log('[MFA] Python 패키지 도구(pip/setuptools/wheel) 복구 중...')
    repair_cmds = []
    if os.path.exists(conda_exe):
        repair_cmds.append([
            conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir,
            'pip', 'setuptools', 'wheel'
        ])
    system_conda = shutil.which('conda')
    if system_conda:
        repair_cmds.append([
            system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir,
            'pip', 'setuptools', 'wheel'
        ])
    repair_cmds.extend([
        [python_exe, '-m', 'ensurepip', '--upgrade'],
        [python_exe, '-m', 'pip', 'install', '--upgrade', 'setuptools', 'wheel'],
    ])
    if os.path.exists(pip_exe):
        repair_cmds.append([pip_exe, 'install', '--upgrade', 'setuptools', 'wheel'])

    last_err = ""
    for repair_cmd in repair_cmds:
        log(f"   -> repair cmd: {' '.join(repair_cmd)}")
        result = _run_subprocess_text(repair_cmd, env=env)
        if result.returncode != 0:
            err_txt = (result.stderr or result.stdout or '').strip()
            if err_txt:
                log(f"   [warn] repair failed: {err_txt[:500]}")
                last_err = err_txt
            continue
        ok, detail = _check_packaging_stack(python_exe, env)
        if ok:
            return True
        if detail:
            last_err = detail
            log(f"   [warn] import check failed after repair: {detail[:500]}")

    log('[MFA] Failed to restore pip/setuptools/wheel')
    if last_err:
        log(f"   last error: {last_err[:500]}")
    return False


def diagnose_mfa_runtime(mfa_path="", language='korean', callback=None):
    """
    Read-only runtime diagnosis for the current MFA environment.
    Returns a dict that the UI or smoke scripts can summarize.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    lang = str(language or 'korean').strip().lower()
    resolved = mfa_path or find_mfa_executable() or ""
    resolved = resolve_working_mfa_executable(resolved, callback=callback) if resolved else ""
    report = {
        "language": lang,
        "mfa_path": resolved,
        "env_dir": "",
        "ready": False,
        "issues": [],
        "checks": {
            "mfa_executable": False,
            "python_exe": False,
            "pip_exe": False,
            "python_version": "",
            "python_rebuild_required": False,
            "packaging_stack": None,
            "language_support": None,
            "language_support_mode": "",
            "model_ready": False,
        },
    }

    if not resolved or not os.path.exists(resolved):
        report["issues"].append("mfa_missing")
        log("[MFA] 진단: MFA 실행 파일을 찾지 못했습니다.")
        return report

    launcher_ok, launcher_detail = _probe_mfa_launcher(resolved)
    report["checks"]["mfa_executable"] = bool(launcher_ok)
    if not launcher_ok:
        report["issues"].append("mfa_launcher_broken")
        if launcher_detail:
            log(f"[MFA] 진단: launcher broken - {launcher_detail[:280]}")
        return report
    if _is_env_scripts_path(resolved):
        env_dir = os.path.dirname(os.path.dirname(resolved))
        python_exe = _resolve_env_python_exe(env_dir)
        pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
        report["env_dir"] = env_dir
        report["checks"]["python_exe"] = os.path.exists(python_exe)
        report["checks"]["pip_exe"] = os.path.exists(pip_exe)
        py_ver = get_mfa_env_python_version(resolved)
        report["checks"]["python_version"] = py_ver
        rebuild = mfa_python_version_requires_downgrade(py_ver)
        report["checks"]["python_rebuild_required"] = rebuild
        if rebuild:
            report["issues"].append("python_rebuild_required")
        if report["checks"]["python_exe"]:
            env = _get_conda_env(resolved)
            ok, _detail = _check_packaging_stack(python_exe, env)
            report["checks"]["packaging_stack"] = ok
            if not ok:
                report["issues"].append("packaging_stack_missing")
            if lang == 'korean':
                lang_ok, _detail = _check_env_imports(
                    python_exe,
                    env,
                    _KOREAN_NATIVE_RUNTIME_CHECK_EXPR,
                )
                lang_mode = "native" if lang_ok else ""
                if not lang_ok:
                    legacy_ok, _legacy_detail = _check_env_imports(
                        python_exe,
                        env,
                        "import jamo; import eunjeon; tok=eunjeon.Mecab(); tok.morphs('테스트')",
                    )
                    if legacy_ok:
                        lang_ok = True
                        lang_mode = "legacy"
                    fallback_ok, _fb_detail = _check_env_imports(
                        python_exe,
                        env,
                        'from mecab import MeCab; tok=MeCab(); tok.parse("테스트")',
                    )
                    if not lang_ok and fallback_ok:
                        lang_ok = True
                        lang_mode = "fallback"
                        report["issues"].append("korean_deps_degraded")
                    if not lang_ok:
                        report["issues"].append(f"{lang}_deps_missing")
                report["checks"]["language_support"] = lang_ok
                report["checks"]["language_support_mode"] = lang_mode
            else:
                lang_ok, _detail = _check_env_imports(python_exe, env, 'import spacy; import sudachipy; import sudachidict_core')
                report["checks"]["language_support"] = lang_ok
                report["checks"]["language_support_mode"] = "native" if lang_ok else ""
                if not lang_ok:
                    report["issues"].append(f"{lang}_deps_missing")

    has_model, _msg = check_mfa_model(resolved, language=lang)
    report["checks"]["model_ready"] = bool(has_model)
    if not has_model:
        report["issues"].append("model_missing")

    report["ready"] = (
        report["checks"]["mfa_executable"]
        and report["checks"]["python_rebuild_required"] is False
        and report["checks"]["packaging_stack"] in {True, None}
        and report["checks"]["language_support"] in {True, None}
        and report["checks"]["model_ready"]
    )
    return report


def _resolve_single_speaker_flag(mfa_path, env=None):
    """
    MFA 버전에 따라 단일 화자 옵션 표기가 다를 수 있어(--single-speaker / --single_speaker)
    help 출력을 보고 지원되는 표기를 선택합니다.
    """
    key = os.path.abspath(mfa_path or "")
    cached = _MFA_SINGLE_SPEAKER_FLAG_CACHE.get(key)
    if cached:
        return cached

    candidates = ["--single-speaker", "--single_speaker"]
    try:
        res = _run_subprocess_text(
            [mfa_path, "align", "--help"],
            timeout=20,
            env=env,
        )
        help_text = (
            f"{_decode_subprocess_output(res.stdout)}\n"
            f"{_decode_subprocess_output(res.stderr)}"
        ).lower()
        for flag in candidates:
            if flag in help_text:
                _MFA_SINGLE_SPEAKER_FLAG_CACHE[key] = flag
                return flag
    except Exception:
        pass

    # 기본값은 요청에 맞춰 하이픈 표기 우선
    _MFA_SINGLE_SPEAKER_FLAG_CACHE[key] = "--single-speaker"
    return "--single-speaker"


def _resolve_speaker_adaptation_flag(mfa_path, env=None):
    key = os.path.abspath(mfa_path or "")
    cached = _MFA_SPEAKER_ADAPT_FLAG_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = [
        "--uses_speaker_adaptation",
        "--uses-speaker-adaptation",
        "--speaker_adaptation",
        "--speaker-adaptation",
    ]
    try:
        res = _run_subprocess_text(
            [mfa_path, "align", "--help"],
            timeout=20,
            env=env,
        )
        help_text = (
            f"{_decode_subprocess_output(res.stdout)}\n"
            f"{_decode_subprocess_output(res.stderr)}"
        ).lower()
        for flag in candidates:
            if flag in help_text:
                _MFA_SPEAKER_ADAPT_FLAG_CACHE[key] = flag
                return flag
    except Exception:
        pass

    _MFA_SPEAKER_ADAPT_FLAG_CACHE[key] = ""
    return ""


def _normalize_mfa_align_profile(profile):
    p = str(profile or "").strip().lower()
    if p in {"fast", "quick", "lite", "speed"}:
        return "fast"
    if p in {"accurate", "accuracy", "acc", "adapted", "speaker_adapted", "speaker_adaptation"}:
        return "accurate"
    if p in {"default", "basic", "base", "legacy", ""}:
        return "default"
    return "default"


def _resolve_mfa_align_options(align_profile):
    profile = _normalize_mfa_align_profile(align_profile)
    opts = dict(MFA_ALIGN_PROFILE_PRESETS.get(profile, MFA_ALIGN_PROFILE_PRESETS["default"]))

    # Optional env override for advanced users.
    env_jobs = str(os.environ.get("UTOA_MFA_NUM_JOBS", "")).strip()
    if env_jobs:
        try:
            j = int(float(env_jobs))
            if j >= 1:
                opts["num_jobs"] = j
        except Exception:
            pass

    soft_mode = str(os.environ.get("UTOA_MFA_SOFT_BANK_MODE", "")).strip().lower() in {
        "1", "true", "yes", "on", "y"
    }
    if soft_mode:
        base_beam = int(opts.get("beam", 1000) or 1000)
        base_retry = int(opts.get("retry_beam", 4000) or 4000)
        opts["beam"] = max(base_beam, int(round(base_beam * 1.35)))
        opts["retry_beam"] = max(base_retry, int(round(base_retry * 1.45)))
        opts["fine_tune"] = True
        if profile != "fast":
            opts["speaker_adaptation"] = True
        opts["soft_bank_mode"] = True
    return profile, opts


def find_mfa_executable():
    """
    시스템에 설치된 MFA 실행 파일을 탐색합니다.
    포터블 환경 -> Conda 환경 -> 시스템 PATH 순서로 검색합니다.
    
    Returns:
        MFA 실행 파일 경로 또는 None
    """
    # 1. 공유/레거시 포터블 환경
    for p in _candidate_mfa_executable_paths():
        if not os.path.exists(p):
            continue
        resolved = resolve_working_mfa_executable(p)
        ok, _detail = _probe_mfa_launcher(resolved)
        if ok:
            logger.info(f"포터블 MFA 발견: {resolved}")
            return resolved

    # 2. 시스템 PATH
    mfa_path = shutil.which('mfa')
    if mfa_path:
        resolved = resolve_working_mfa_executable(mfa_path)
        ok, _detail = _probe_mfa_launcher(resolved)
        if ok:
            logger.info(f"시스템 MFA 발견: {resolved}")
            return resolved

    # 3. Conda 환경 기본 경로
    conda_paths = [
        os.path.expanduser('~/miniconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/anaconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/miniconda3/Scripts/mfa.exe'),
    ]
    for p in conda_paths:
        if not os.path.exists(p):
            continue
        resolved = resolve_working_mfa_executable(p)
        ok, _detail = _probe_mfa_launcher(resolved)
        if ok:
            logger.info(f"Conda MFA 발견: {resolved}")
            return resolved

    return None


def check_mfa_model(mfa_path, language='korean'):
    """
    MFA 음향 모델이 다운로드되어 있는지 확인합니다.
    
    Args:
        mfa_path: MFA 실행 파일 경로
        language: 'korean' 또는 'japanese'
    
    Returns:
        (설치 여부: bool, 메시지: str)
    """
    if not mfa_path:
        return False, "MFA 실행 파일을 찾을 수 없습니다."
    mfa_path = resolve_working_mfa_executable(mfa_path)
    if not mfa_path or not os.path.exists(mfa_path):
        return False, "MFA 실행 파일을 찾을 수 없습니다."
    launcher_ok, launcher_detail = _probe_mfa_launcher(mfa_path)
    if not launcher_ok:
        detail = (launcher_detail or "").strip()
        if detail:
            return False, f"MFA launcher 오류: {detail[:220]}"
        return False, "MFA launcher 오류가 발생했습니다."

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '일본어' if language == 'japanese' else '한국어'

    try:
        env = _get_conda_env(mfa_path)
        result = _run_subprocess_text(
            [mfa_path, 'model', 'list', 'acoustic'],
            timeout=30,
            env=env,
        )
        combined = f"{_decode_subprocess_output(result.stdout)}\n{_decode_subprocess_output(result.stderr)}"
        if _is_mfa_launcher_failure_text(combined):
            return False, "MFA launcher 오류(프로세스 생성 실패)로 모델 상태를 확인할 수 없습니다."
        stdout_text = _decode_subprocess_output(result.stdout)
        if model_name in stdout_text:
            return True, f"{lang_label} MFA 모델이 설치되어 있습니다."
        else:
            return False, f"{lang_label} MFA 모델이 설치되어 있지 않습니다. 다운로드가 필요합니다."
    except Exception as e:
        return False, f"MFA 모델 확인 실패: {e}"


def check_mfa_ready(language='korean', mfa_path=''):
    resolved_mfa = mfa_path or find_mfa_executable()
    if not resolved_mfa or not os.path.exists(resolved_mfa):
        return make_runtime_report(
            "align",
            ALIGN_EXEC_MISSING,
            "MFA 실행 파일을 찾을 수 없습니다.",
            engine="mfa",
            language=str(language or "korean").strip().lower(),
            mfa_path=str(resolved_mfa or ""),
            ready=False,
        )

    has_model, msg = check_mfa_model(resolved_mfa, language=language)
    if not has_model:
        return make_runtime_report(
            "align",
            ALIGN_MODEL_MISSING,
            msg or "MFA 모델을 찾을 수 없습니다.",
            engine="mfa",
            language=str(language or "korean").strip().lower(),
            mfa_path=str(resolved_mfa or ""),
            ready=False,
        )

    return make_runtime_report(
        "align",
        OK,
        "MFA 정렬 준비 완료",
        engine="mfa",
        language=str(language or "korean").strip().lower(),
        mfa_path=str(resolved_mfa or ""),
        ready=True,
    )


def ensure_korean_support(mfa_path, callback=None):
    """
    Ensure Korean MFA tokenizer dependencies are available:
    - bundled offline wheels (when present)
    - python-mecab-ko (native MeCab binding)
    - jamo
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False
    if not _is_env_scripts_path(mfa_path):
        # System MFA path; skip env-local auto install here.
        return True
    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    if not os.path.exists(python_exe):
        return False
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log('[MFA] Failed to prepare base Python packaging tools before Korean dependency install')
        return False
    pkg_check_cmd = [python_exe, '-c', _PACKAGING_STACK_IMPORT_EXPR]
    native_tokenizer_check_cmd = [
        python_exe,
        '-c',
        _KOREAN_NATIVE_RUNTIME_CHECK_EXPR,
    ]
    check_cmd = [
        python_exe,
        '-c',
        _KOREAN_TOKENIZER_IMPORT_CHECK_EXPR,
    ]
    fallback_runtime_check_cmd = [
        python_exe,
        '-c',
        (
            "ok=False\n"
            "try:\n"
            "    from mecab import MeCab\n"
            "    tok = MeCab()\n"
            "    out = tok.parse('테스트 fallback')\n"
            "    ok = isinstance(out, list)\n"
            "except Exception:\n"
            "    ok = False\n"
            "import sys\n"
            "sys.exit(0 if ok else 1)\n"
        ),
    ]
    try:
        env = _get_conda_env(mfa_path)
        allow_degraded = str(os.environ.get("UTOA_KO_DEGRADED_ALLOW", "1")).strip().lower() not in {"0", "false", "no", "off"}
        degraded_notice_emitted = False
        msvc_notice_emitted = False

        def _emit_degraded_notice_once(require_msvc: bool = False):
            nonlocal degraded_notice_emitted, msvc_notice_emitted
            if require_msvc and not msvc_notice_emitted:
                _emit_msvc_required_notice(callback, log)
                msvc_notice_emitted = True
                degraded_notice_emitted = True
                return
            if degraded_notice_emitted:
                return
            log('[MFA] Korean tokenizer running in degraded fallback mode (native tokenizer unavailable).')
            degraded_notice_emitted = True

        def _ensure_mecab_dictionary():
            site_packages = os.path.join(env_dir, 'Lib', 'site-packages')
            mecabrc = os.path.join(site_packages, 'mecabrc')
            dic_src = os.path.join(site_packages, 'mecab_ko_dic', 'dictionary')
            dic_dst = os.path.join(site_packages, 'mecab-ko-dic')
            if os.path.exists(dic_dst):
                return
            if not (os.path.exists(mecabrc) and os.path.isdir(dic_src)):
                return
            try:
                os.makedirs(dic_dst, exist_ok=True)
                for name in os.listdir(dic_src):
                    src = os.path.join(dic_src, name)
                    dst = os.path.join(dic_dst, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            except Exception as e:
                log(f"[MFA] Failed to seed mecab-ko-dic: {e}")

        def _ensure_mecab_shim():
            site_packages = os.path.join(env_dir, 'Lib', 'site-packages')
            mecab_dir = os.path.join(site_packages, 'mecab')
            mecab_init = os.path.join(mecab_dir, '__init__.py')
            try:
                if not os.path.isdir(mecab_dir):
                    os.makedirs(mecab_dir, exist_ok=True)
                    content = ""
                else:
                    with open(mecab_init, 'r', encoding='utf-8') as f:
                        content = f.read()
                # Keep shim idempotent while allowing migration from buggy V2.
                if 'UTOA_MECAB_SHIM_V3' in content:
                    return
                content = re.sub(r"\n?# UTOA_MECAB_SHIM_V2[\s\S]*$", "", content)
                shim = """

# UTOA_MECAB_SHIM_V3
import re as _utoa_re
try:
    import MeCab as _MeCabMod
except Exception:
    _MeCabMod = None

class _UtoaMecabNode:
    def __init__(self, surface, pos):
        self.surface = surface
        self.pos = pos

def _utoa_simple_tokenize(text):
    text = str(text or "")
    tokens = _utoa_re.findall(r"[가-힣]+|[A-Za-z]+|[0-9]+|[^\s]", text)
    return tokens or [text]

class MeCab:
    def __init__(self):
        self._native = None
        if _MeCabMod is not None:
            try:
                self._native = _MeCabMod.Tagger()
            except Exception:
                self._native = None

    def parse(self, text):
        if self._native is not None:
            node = self._native.parseToNode(text)
            items = []
            while node is not None:
                surface = getattr(node, "surface", "") or ""
                if surface:
                    feature = getattr(node, "feature", "") or ""
                    pos = feature.split(",", 1)[0] if feature else ""
                    items.append(_UtoaMecabNode(surface, pos))
                node = getattr(node, "next", None)
            return items
        # Pure-Python fallback tokenizer for environments without native mecab/eunjeon.
        return [_UtoaMecabNode(tok, "NNG") for tok in _utoa_simple_tokenize(text)]
"""
                with open(mecab_init, 'w', encoding='utf-8') as f:
                    f.write(content + shim)
            except Exception as e:
                log(f"[MFA] Failed to patch mecab shim: {e}")

        def _ensure_jamo_shim():
            site_packages = os.path.join(env_dir, 'Lib', 'site-packages')
            jamo_dir = os.path.join(site_packages, 'jamo')
            jamo_init = os.path.join(jamo_dir, '__init__.py')
            # Do not overwrite a real jamo package.
            if os.path.isfile(jamo_init):
                try:
                    with open(jamo_init, 'r', encoding='utf-8') as f:
                        existing = f.read()
                    if "UTOA_JAMO_SHIM_V1" in existing:
                        return
                    if "def hangul_to_jamo" in existing or "class Jamo" in existing:
                        return
                except Exception:
                    return
            try:
                os.makedirs(jamo_dir, exist_ok=True)
                shim = (
                    "# UTOA_JAMO_SHIM_V1\n"
                    "def hangul_to_jamo(text):\n"
                    "    txt = '' if text is None else str(text)\n"
                    "    for ch in txt:\n"
                    "        yield ch\n\n"
                    "def h2j(text):\n"
                    "    return ''.join(list(hangul_to_jamo(text)))\n\n"
                    "def j2hcj(text):\n"
                    "    return '' if text is None else str(text)\n\n"
                    "def is_jamo(_ch):\n"
                    "    return False\n"
                )
                with open(jamo_init, 'w', encoding='utf-8') as f:
                    f.write(shim)
                log('[MFA] jamo shim injected for degraded Korean fallback mode.')
            except Exception as e:
                log(f"[MFA] Failed to create jamo shim: {e}")

        def _looks_like_pyexpat_dll_issue(msg):
            s = (msg or '').lower()
            return ('pyexpat' in s and 'dll load failed' in s) or ('libexpat' in s and 'not found' in s)

        def _try_repair_pyexpat():
            conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
            cmds = []
            if os.path.exists(conda_exe):
                cmds.append([conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir, 'libexpat'])
            system_conda = shutil.which('conda')
            if system_conda:
                cmds.append([system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir, 'libexpat'])
            for cmd in cmds:
                log(f"   -> repair cmd: {' '.join(cmd)}")
                res = _run_subprocess_text(cmd, env=env)
                if res.returncode == 0:
                    return True
            return False

        def _check_imports():
            res = _run_subprocess_text(check_cmd, env=env)
            if res.returncode == 0:
                return True, ''
            detail = (res.stderr or res.stdout or '').strip()
            return False, detail

        def _ensure_packaging_stack():
            res = _run_subprocess_text(pkg_check_cmd, env=env)
            if res.returncode == 0:
                return True
            log('[MFA] pip/setuptools/wheel 상태를 복구합니다')
            install_cmds = [
                [python_exe, '-m', 'pip', 'install', '--upgrade', 'setuptools', 'wheel'],
            ]
            if os.path.exists(pip_exe):
                install_cmds.append([pip_exe, 'install', '--upgrade', 'setuptools', 'wheel'])
            conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
            if os.path.exists(conda_exe):
                install_cmds.append([
                    conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                    'pip', 'setuptools', 'wheel'
                ])
            system_conda = shutil.which('conda')
            if system_conda:
                install_cmds.append([
                    system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                    'pip', 'setuptools', 'wheel'
                ])
            for install_cmd in install_cmds:
                log(f"   -> repair cmd: {' '.join(install_cmd)}")
                result = _run_subprocess_text(install_cmd, env=env)
                if result.returncode != 0:
                    err_txt = (result.stderr or result.stdout or '').strip()
                    if err_txt:
                        log(f"   [warn] repair failed: {err_txt[:500]}")
                    continue
                verify = _run_subprocess_text(pkg_check_cmd, env=env)
                if verify.returncode == 0:
                    return True
            return False

        if not _ensure_packaging_stack():
            log('[MFA] Failed to restore pip/setuptools/wheel')
            return False

        _ensure_mecab_dictionary()
        _ensure_mecab_shim()
        if allow_degraded:
            _ensure_jamo_shim()
        ok, detail = _check_imports()
        if (not ok) and _looks_like_pyexpat_dll_issue(detail):
            log('[MFA] Detected pyexpat/libexpat DLL issue; trying repair...')
            if _try_repair_pyexpat():
                ok, detail = _check_imports()
        if ok:
            native_check = _run_subprocess_text(native_tokenizer_check_cmd, env=env)
            if native_check.returncode != 0:
                log('[MFA] Korean tokenizer running in degraded fallback mode (native MeCab not available).')
                _emit_degraded_notice_once(require_msvc=False)
            return True
        conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
        system_conda = shutil.which('conda')
        install_plans: list[tuple[str, list[list[str]]]] = []

        wheel_dir = _resolve_korean_wheel_dir(mfa_path)
        if wheel_dir:
            log(f"[MFA] Found bundled Korean wheels: {wheel_dir}")
            offline_cmds: list[list[str]] = [
                [
                    python_exe, '-m', 'pip', 'install', '--upgrade',
                    '--no-index', '--find-links', wheel_dir,
                    'python-mecab-ko', 'jamo', 'python-mecab-ko-dic',
                ]
            ]
            if os.path.exists(pip_exe):
                offline_cmds.append(
                    [
                        pip_exe, 'install', '--upgrade',
                        '--no-index', '--find-links', wheel_dir,
                        'python-mecab-ko', 'jamo', 'python-mecab-ko-dic',
                    ]
                )
            if system_conda:
                offline_cmds.append(
                    [
                        system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip', 'install',
                        '--upgrade', '--no-index', '--find-links', wheel_dir,
                        'python-mecab-ko', 'jamo', 'python-mecab-ko-dic',
                    ]
                )
            install_plans.append(('bundled Korean wheels (offline)', offline_cmds))

        primary_cmds: list[list[str]] = []
        if os.path.exists(conda_exe):
            primary_cmds.append([
                conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                '-c', 'conda-forge', '--override-channels',
                'python-mecab-ko', 'jamo'
            ])
        if system_conda:
            primary_cmds.append([
                system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                '-c', 'conda-forge', '--override-channels',
                'python-mecab-ko', 'jamo'
            ])
        primary_cmds.append([
            python_exe, '-m', 'pip', 'install', '--upgrade', '--only-binary=:all:',
            'python-mecab-ko', 'jamo'
        ])
        if os.path.exists(pip_exe):
            primary_cmds.append([
                pip_exe, 'install', '--upgrade', '--only-binary=:all:', 'python-mecab-ko', 'jamo'
            ])
        if system_conda:
            primary_cmds.append([
                system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip', 'install',
                '--upgrade', '--only-binary=:all:', 'python-mecab-ko', 'jamo'
            ])
        install_plans.append(('python-mecab-ko, jamo (binary preferred)', primary_cmds))

        last_err = detail
        for dep_label, install_cmds in install_plans:
            log(f"[MFA] Installing Korean tokenizer deps: {dep_label}")
            for install_cmd in install_cmds:
                log(f"   -> cmd: {' '.join(install_cmd)}")
                result = _run_subprocess_text(install_cmd, env=env)
                if result.returncode != 0:
                    err_txt = (result.stderr or result.stdout or '').strip()
                    if err_txt:
                        log(f"   [warn] install failed: {err_txt[:500]}")
                    if _stderr_has_msvc_requirement(err_txt):
                        _emit_degraded_notice_once(require_msvc=True)
                    last_err = err_txt or last_err
                    continue
                if not _ensure_packaging_stack():
                    last_err = 'pip/setuptools/wheel repair failed after install'
                    continue
                _ensure_mecab_dictionary()
                _ensure_mecab_shim()
                if allow_degraded:
                    _ensure_jamo_shim()
                ok, detail = _check_imports()
                if (not ok) and _looks_like_pyexpat_dll_issue(detail):
                    log('[MFA] Detected pyexpat/libexpat DLL issue after install; trying repair...')
                    if _try_repair_pyexpat():
                        ok, detail = _check_imports()
                if ok:
                    log('[MFA] Korean tokenizer deps are ready')
                    native_check = _run_subprocess_text(native_tokenizer_check_cmd, env=env)
                    if native_check.returncode != 0:
                        log('[MFA] Korean tokenizer running in degraded fallback mode (native MeCab not available).')
                        _emit_degraded_notice_once(require_msvc=False)
                    return True
                if detail:
                    log(f"   [warn] import check failed after install: {detail[:500]}")
                    last_err = detail
        _ensure_mecab_shim()
        if allow_degraded:
            _ensure_jamo_shim()
        fallback_runtime = _run_subprocess_text(fallback_runtime_check_cmd, env=env)
        if fallback_runtime.returncode == 0 and allow_degraded:
            log('[MFA] Korean tokenizer deps fallback enabled; continuing with reduced-accuracy mode.')
            _emit_degraded_notice_once(require_msvc=_stderr_has_msvc_requirement(last_err))
            return True
        log('[MFA] Failed to prepare Korean tokenizer deps (python-mecab-ko/jamo)')
        if _stderr_has_msvc_requirement(last_err):
            _emit_msvc_required_notice(callback, log)
        else:
            log('[MFA] Native Korean tokenizer dependencies are unavailable; check wheel/network compatibility.')
        if last_err:
            log(f"   last error: {last_err[:500]}")
        return False
    except Exception as e:
        log(f"[MFA] Korean dependency setup error: {e}")
        return False

def ensure_japanese_support(mfa_path, callback=None):
    """
    MFA 일본어 정렬에 필요한 spacy/sudachipy/sudachidict-core가 있는지 확인하고,
    누락 시 자동 설치를 시도합니다.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or not _is_env_scripts_path(mfa_path):
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')

    if not os.path.exists(python_exe):
        return True
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log("⚠️ MFA Python 패키지 도구 복구에 실패해 일본어 의존성 설치를 계속할 수 없습니다.")
        return False

    check_cmd = [python_exe, '-c', 'import spacy; import sudachipy; import sudachidict_core']
    try:
        env = _get_conda_env(mfa_path)
        result = _run_subprocess_text(check_cmd, env=env)
        if result.returncode == 0:
            return True

        log("📦 MFA 일본어 토크나이저 의존성(spacy, sudachipy, sudachidict-core) 설치/확인 중...")

        install_cmd = None
        if os.path.exists(conda_exe):
            install_cmd = [
                conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                '-c', 'conda-forge', '--override-channels',
                'spacy', 'sudachipy', 'sudachidict-core'
            ]
        else:
            system_conda = shutil.which('conda')
            if system_conda:
                install_cmd = [
                    system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir,
                    '-c', 'conda-forge', '--override-channels',
                    'spacy', 'sudachipy', 'sudachidict-core'
                ]
            elif os.path.exists(pip_exe):
                install_cmd = [pip_exe, 'install', 'spacy', 'sudachipy', 'sudachidict-core']

        if not install_cmd:
            log("⚠️ 일본어 의존성 자동 설치 경로를 찾지 못했습니다.")
            return False

        log(f"   -> 실행 명령어: {' '.join(install_cmd)}")
        install_result = _run_subprocess_text(install_cmd, env=env)
        if install_result.returncode != 0:
            if install_result.stderr:
                log(f"   ⚠️ 설치 stderr: {install_result.stderr[:500]}")
            if install_result.stdout:
                log(f"   ⚠️ 설치 stdout: {install_result.stdout[:500]}")
            if os.path.exists(pip_exe):
                pip_cmd = [pip_exe, 'install', 'spacy', 'sudachipy', 'sudachidict-core']
                log(f"   -> 대체 설치 명령어(pip): {' '.join(pip_cmd)}")
                pip_result = _run_subprocess_text(pip_cmd, env=env)
                if pip_result.returncode != 0:
                    if pip_result.stderr:
                        log(f"   ⚠️ pip stderr: {pip_result.stderr[:500]}")
                    if pip_result.stdout:
                        log(f"   ⚠️ pip stdout: {pip_result.stdout[:500]}")
                    return False
            else:
                return False

        verify = _run_subprocess_text(check_cmd, env=env)
        if verify.returncode == 0:
            log("✅ 일본어 토크나이저 의존성 설치 확인 완료")
            return True

        log("⚠️ 일본어 의존성 설치 후에도 import 검증에 실패했습니다.")
        if verify.stderr:
            log(f"   상세 stderr: {verify.stderr[:500]}")
        return False
    except Exception as e:
        log(f"⚠️ 일본어 의존성 자동 확인/설치 중 오류 발생: {e}")
        return False


def download_mfa_model(mfa_path, language='korean', callback=None):
    """Download MFA acoustic model for selected language."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        log('MFA executable not found.')
        return False
    mfa_path = resolve_working_mfa_executable(mfa_path, callback=callback)
    if not mfa_path or not os.path.exists(mfa_path):
        log('MFA executable not found.')
        return False
    launcher_ok, launcher_detail = _probe_mfa_launcher(mfa_path)
    if not launcher_ok:
        log(f"[MFA] launcher is not healthy: {launcher_detail[:280] if launcher_detail else 'unknown error'}")
        return False
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            log('Failed to prepare Korean dependencies (python-mecab-ko/jamo).')
            return False
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            log('Failed to prepare Japanese dependencies (spacy, sudachipy, sudachidict-core).')
            return False
    log(f'Downloading {lang_label} MFA model...')
    try:
        env = _get_conda_env(mfa_path)
        process = _popen_subprocess(
            [mfa_path, 'model', 'download', 'acoustic', model_name, '--ignore_cache'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )
        launcher_failed = False
        if process.stdout:
            for raw_line in iter(process.stdout.readline, b""):
                line = _decode_subprocess_output(raw_line).strip()
                if line:
                    log(line)
                    if _is_mfa_launcher_failure_text(line):
                        launcher_failed = True
        process.wait()
        if launcher_failed:
            log('[MFA] Model download failed due to launcher process creation error.')
            return False
        if process.returncode == 0:
            log(f'{lang_label} MFA model download completed.')
            return True
        log(f'Model download failed (code: {process.returncode})')
        return False
    except Exception as e:
        log(f'Model download error: {e}')
        return False
def run_mfa_align(
    mfa_path,
    wav_folder,
    dict_path,
    output_folder,
    language='korean',
    callback=None,
    align_profile='accurate',
):
    """Run MFA forced alignment."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False, 'MFA executable not found.'
    mfa_path = resolve_working_mfa_executable(mfa_path, callback=callback)
    if not mfa_path or not os.path.exists(mfa_path):
        return False, 'MFA executable not found.'
    launcher_ok, launcher_detail = _probe_mfa_launcher(mfa_path)
    if not launcher_ok:
        if launcher_detail:
            return False, f'MFA launcher error: {launcher_detail[:220]}'
        return False, 'MFA launcher error'
    if not os.path.exists(wav_folder):
        return False, f'WAV folder not found: {wav_folder}'
    if not os.path.exists(dict_path):
        return False, f'Dictionary not found: {dict_path}'
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    log(f'Checking MFA prerequisites... ({lang_label})')
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            err = 'Missing Korean tokenizer dependencies (python-mecab-ko/jamo).'
            log(err)
            return False, err
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            err = 'Missing Japanese tokenizer dependencies.'
            log(err)
            return False, err
    os.makedirs(output_folder, exist_ok=True)
    env = _get_conda_env(mfa_path)
    ok, preflight_err = _preflight_compute_mfcc(mfa_path, callback=callback)
    if not ok:
        return False, preflight_err
    work_wav_folder = wav_folder
    work_dict_path = dict_path
    work_output_folder = output_folder
    safe_workspace = None
    low_rms_gain_cfg = _resolve_low_rms_gain_config()
    low_rms_gain_enabled = bool(low_rms_gain_cfg.get("enabled", False))
    needs_ascii_safe_workspace = any(_contains_non_ascii(p) for p in (wav_folder, dict_path, output_folder))
    if needs_ascii_safe_workspace or low_rms_gain_enabled:
        try:
            safe_workspace = _prepare_ascii_safe_alignment_workspace(
                wav_folder,
                dict_path,
                output_folder,
                wav_gain_config=low_rms_gain_cfg,
            )
            work_wav_folder = safe_workspace["corpus_dir"]
            work_dict_path = safe_workspace["dict_path"]
            work_output_folder = safe_workspace["output_dir"]
            if needs_ascii_safe_workspace and low_rms_gain_enabled:
                log(f"[MFA] Using staging workspace (non-ASCII path + low-RMS gain): {safe_workspace['base']}")
            elif needs_ascii_safe_workspace:
                log(f"[MFA] Non-ASCII path detected, using staging workspace: {safe_workspace['base']}")
            else:
                log(f"[MFA] Using staging workspace for low-RMS gain: {safe_workspace['base']}")
            if low_rms_gain_enabled:
                stats = dict(safe_workspace.get("wav_gain_stats") or {})
                scanned = int(stats.get("scanned", 0) or 0)
                boosted = int(stats.get("boosted", 0) or 0)
                passthrough = int(stats.get("passthrough", 0) or 0)
                log(
                    "[MFA] Low-RMS staging summary: "
                    f"boosted={boosted}/{scanned}, weak_assist={int(stats.get('weak_assist', 0) or 0)}, "
                    f"passthrough={passthrough}, "
                    f"threshold={low_rms_gain_cfg.get('threshold_db', -24.0)}dB, "
                    f"target={low_rms_gain_cfg.get('target_db', -16.0)}dB, "
                    f"max_gain={low_rms_gain_cfg.get('max_gain_db', 18.0)}dB, "
                    f"weak_trigger={low_rms_gain_cfg.get('weak_voice_trigger_db', -23.0)}dB"
                )
        except Exception as e:
            err = f"Failed to prepare MFA staging workspace: {e}"
            log(err)
            return False, err
    dict_sanitize_ok, dict_sanitize_err = _sanitize_alignment_dictionary_for_mfa(work_dict_path, callback=callback)
    if not dict_sanitize_ok:
        return False, dict_sanitize_err
    dict_ok, dict_err = _validate_alignment_dictionary(work_dict_path, callback=callback)
    if not dict_ok:
        return False, dict_err
    single_speaker_flag = _resolve_single_speaker_flag(mfa_path, env=env)
    resolved_profile, align_opts = _resolve_mfa_align_options(align_profile)
    cmd = [
        mfa_path, 'align',
        work_wav_folder, work_dict_path, model_name, work_output_folder,
        single_speaker_flag,
    ]
    speaker_adapt_enabled = bool(align_opts.get("speaker_adaptation", False))
    if speaker_adapt_enabled:
        adapt_flag = _resolve_speaker_adaptation_flag(mfa_path, env=env)
        if adapt_flag:
            cmd.append(adapt_flag)
        else:
            speaker_adapt_enabled = False
            log("[MFA] Speaker adaptation flag not supported by this MFA version; skipped.")
    if align_opts.get("clean", True):
        cmd.append("--clean")
    if align_opts.get("fine_tune", False):
        cmd.append("--fine_tune")
    if align_opts.get("textgrid_cleanup", True):
        cmd.append("--textgrid_cleanup")
    cmd.extend([
        "--beam", str(int(align_opts.get("beam", 1000))),
        "--retry_beam", str(int(align_opts.get("retry_beam", 4000))),
        "--num_jobs", str(int(align_opts.get("num_jobs", 1))),
    ])
    log(
        "Starting MFA alignment... "
        f"({single_speaker_flag}, profile={resolved_profile}, "
        f"speaker_adapt={'on' if speaker_adapt_enabled else 'off'}, "
        f"fine_tune={'on' if align_opts.get('fine_tune') else 'off'}, "
        f"beam={align_opts.get('beam')}, retry_beam={align_opts.get('retry_beam')}, "
        f"num_jobs={align_opts.get('num_jobs')}, "
        f"soft_bank={'on' if align_opts.get('soft_bank_mode') else 'off'})"
    )
    try:
        process = _popen_subprocess(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )
        tail_lines = []
        launcher_failed = False
        if process.stdout:
            for raw_line in iter(process.stdout.readline, b""):
                stripped = _decode_subprocess_output(raw_line).strip()
                if stripped:
                    log(stripped)
                    if _is_mfa_launcher_failure_text(stripped):
                        launcher_failed = True
                    tail_lines.append(stripped)
                    if len(tail_lines) > 120:
                        tail_lines.pop(0)
        process.wait()
        if launcher_failed:
            err = 'MFA launcher failed to create process. Rebuild or repair MFA launcher.'
            log(err)
            return False, err
        if process.returncode == 0:
            if safe_workspace is not None:
                copied = _copy_back_textgrids(work_output_folder, output_folder)
                log(f"[MFA] Copied back {copied} TextGrid files from staging workspace.")
            log('MFA alignment completed successfully.')
            return True, ''
        joined_tail = '\n'.join(tail_lines[-40:])
        lowered_tail = joined_tail.lower()
        if (
            'please install korean support' in lowered_tail
            or ('importerror' in lowered_tail and 'eunjeon' in lowered_tail and 'jamo' in lowered_tail)
            or ('importerror' in lowered_tail and 'mecab' in lowered_tail and 'jamo' in lowered_tail)
        ):
            err = 'Korean tokenizer dependencies are missing in MFA env. (python-mecab-ko/jamo)'
            log(err)
            return False, err
        err = f'MFA alignment failed (code: {process.returncode})'
        if tail_lines:
            err += f' | tail: {tail_lines[-1][:180]}'
        if process.returncode in {3221225477, -1073741819}:
            err += (
                " | hint: access_violation(0xC0000005), likely native kaldi crash "
                "(env binary mismatch or corpus/lexicon edge-case), retry with lower-load profile(fast/default)"
            )
        log(err)
        return False, err
    except FileNotFoundError:
        err = 'MFA executable not found. Check MFA installation.'
        log(err)
        return False, err
    except Exception as e:
        err = f'Unexpected MFA error: {e}'
        log(err)
        return False, err
    finally:
        if safe_workspace is not None:
            base_dir = str(safe_workspace.get("base", "") or "").strip()
            if base_dir:
                try:
                    shutil.rmtree(base_dir, ignore_errors=True)
                    log(f"[MFA] Cleaned staging workspace: {base_dir}")
                except Exception as cleanup_exc:
                    log(f"[MFA] Failed to clean staging workspace: {cleanup_exc}")

def patch_mfa_korean_support(mfa_path, callback=None):
    """
    Keep MFA Korean tokenizer files in python-mecab-ko-friendly state.
    This also reverts older local patches that force eunjeon.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if sys.platform != 'win32' or not mfa_path or not _is_env_scripts_path(mfa_path):
        return True
        
    try:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        site_packages = os.path.join(env_dir, 'Lib', 'site-packages', 'montreal_forced_aligner')
        
        spacy_py = os.path.join(site_packages, 'tokenization', 'spacy.py')
        korean_py = os.path.join(site_packages, 'tokenization', 'korean.py')

        def _ensure_writable_copy(path: str):
            # Conda/micromamba often hardlinks site-packages to pkgs cache.
            # Break hardlinks before patching to avoid corrupting the package cache.
            if not os.path.exists(path):
                return
            try:
                st = os.stat(path)
                if getattr(st, "st_nlink", 1) > 1:
                    tmp = f"{path}.utoa_tmp"
                    shutil.copy2(path, tmp)
                    os.replace(tmp, path)
            except Exception:
                return

        def _file_empty(path: str) -> bool:
            try:
                return os.path.exists(path) and os.path.getsize(path) == 0
            except Exception:
                return False
        
        # 1) spacy.py: keep error/help text and import checks aligned with python-mecab-ko.
        if os.path.exists(spacy_py):
            _ensure_writable_copy(spacy_py)
            if _file_empty(spacy_py):
                log("⚠️ spacy.py is empty. Reinstall MFA package before patching.")
                return False
            with open(spacy_py, 'r', encoding='utf-8') as f:
                content = f.read()
            original = content
            content = content.replace("pip install eunjeon jamo", "pip install python-mecab-ko jamo")
            if "import eunjeon" in content and "import mecab" not in content:
                content = content.replace("import eunjeon", "import mecab")
            if content != original:
                with open(spacy_py, 'w', encoding='utf-8') as f:
                    f.write(content)
                log("   [Patch] spacy.py Korean dependency hints restored to python-mecab-ko")

        # 2) korean.py: remove old EunjeonWrapper injections and restore MeCab path.
        if os.path.exists(korean_py):
            _ensure_writable_copy(korean_py)
            if _file_empty(korean_py):
                log("⚠️ korean.py is empty. Reinstall MFA package before patching.")
                return False
            with open(korean_py, 'r', encoding='utf-8') as f:
                content = f.read()
            original = content
            content = content.replace("from eunjeon import Mecab as MeCab", "from mecab import MeCab")
            content = content.replace("self.tokenizer = EunjeonWrapper()", "self.tokenizer = MeCab()")
            if "class EunjeonWrapper" in content and "class KoreanTokenizer:" in content:
                content = re.sub(
                    r"\nclass EunjeonNode:.*?\nclass KoreanTokenizer:",
                    "\nclass KoreanTokenizer:",
                    content,
                    flags=re.S,
                )
            content = re.sub(r'try:\s+try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
            content = re.sub(r'try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
            if content != original:
                with open(korean_py, 'w', encoding='utf-8') as f:
                    f.write(content)
                log("   [Patch] korean.py tokenizer wiring restored to MeCab-first")
                
        return True
    except Exception as e:
        log(f"⚠️ MFA 한국어 패치 중 오류 발생: {e}")
        return False





