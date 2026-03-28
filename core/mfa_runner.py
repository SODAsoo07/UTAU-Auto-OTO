"""
MFA (Montreal Forced Aligner) runtime helpers.
- Locate MFA executable in portable/runtime/Conda environments.
- Validate and repair dependency/runtime issues before alignment.
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
import time
import wave
import importlib.util
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

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
_MFA_SINGLE_SPEAKER_FLAG_CACHE = {}
_MFA_SPEAKER_ADAPT_FLAG_CACHE = {}
_MFA_BREATH_WORD_RE = re.compile(r"(?i)^breath\d*$")
_MFA_PERMISSION_SOFT_RETRY_CACHE: Dict[str, int] = {}

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
    # Prefer UTF-8 first to prevent locale-dependent mojibake in logs.
    candidates: list[str] = []
    for enc in (
        "utf-8-sig",
        "utf-8",
        _preferred_subprocess_encoding(),
        getattr(locale, "getencoding", lambda: "")() or "",
        "cp949",
        "cp932",
        "mbcs",
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

    # Hard-fix UTF-8 first. If it succeeds, do not attempt locale fallback.
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass

    for enc in _subprocess_decode_candidates():
        if enc in {"utf-8", "utf-8-sig"}:
            continue
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue

    return raw.decode("utf-8", errors="replace")

def _run_subprocess_text(args: Sequence[str], **kwargs):
    completed = subprocess.run(args, capture_output=True, text=False, **kwargs)
    completed.stdout = _decode_subprocess_output(getattr(completed, "stdout", b""))
    completed.stderr = _decode_subprocess_output(getattr(completed, "stderr", b""))
    return completed


def _safe_env_subprocess_cwd(env_dir: str) -> Optional[str]:
    path = str(env_dir or "").strip()
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        return abs_path
    return None


def _contains_non_ascii(text):
    try:
        return any(ord(ch) > 127 for ch in str(text or ""))
    except Exception:
        return False


def _default_mfa_root_dir(mfa_path="", per_process: bool = False):
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    elif mfa_path:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
        # mfa_path often points to "<runtime>\\.env\\Scripts\\mfa.exe".
        # Store MFA_ROOT_DIR under runtime root, not inside .env, to prevent
        # cache bloat in the virtual environment directory.
        base_name = os.path.basename(os.path.normpath(app_dir)).strip().lower()
        if base_name in {".env", "env"}:
            app_dir = os.path.dirname(app_dir)
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(app_dir, ".mfa_root_ascii")
    if sys.platform == "win32" and (
        _contains_non_ascii(root) or "!" in root or "%" in root
    ):
        public_root = str(os.environ.get("PUBLIC", r"C:\Users\Public") or r"C:\Users\Public").strip()
        root = os.path.join(public_root, "UTAU_Auto_OTO_v3", ".mfa_root_ascii")
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


def _candidate_mfa_runtime_roots() -> List[str]:
    roots: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        norm = os.path.normcase(os.path.abspath(str(path or "")))
        if not norm or norm in seen:
            return
        seen.add(norm)
        roots.append(os.path.abspath(path))

    shared_root = str(os.environ.get("UTOA_MFA_SHARED_ROOT", "") or "").strip()
    explicit_runtime_root = str(os.environ.get("UTOA_RUNTIME_ROOT", "") or "").strip()
    if explicit_runtime_root:
        _add(explicit_runtime_root)

    if shared_root:
        _add(shared_root)

    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        parent_dir = os.path.dirname(app_dir)
        # Portable release layout:
        # <payload_root>\UTAU_Auto_OTO\UTAU_Auto_OTO.exe
        # setup_mfa/runtime env are created in <payload_root>\.env.
        # Prefer payload_root first to avoid selecting stale bundled env.
        if os.path.basename(app_dir).lower() in {"utau_auto_oto", "auto_oto"}:
            _add(parent_dir)
            _add(app_dir)
        else:
            _add(app_dir)
            _add(parent_dir)
    else:
        source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _add(source_root)

    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        _add(os.path.join(local_app_data, "UTAU_Auto_OTO_v3"))
        _add(os.path.join(local_app_data, "UTAU_Auto_OTO"))
    else:
        _add(os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3"))
        _add(os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO"))

    public_root = str(os.environ.get("PUBLIC", r"C:\Users\Public") or r"C:\Users\Public").strip()
    _add(os.path.join(public_root, "UTAU_Auto_OTO_v3"))
    return roots


def _resolve_default_mfa_runtime_root() -> str:
    candidates = _candidate_mfa_runtime_roots()
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            fd, probe = tempfile.mkstemp(prefix=".utoa_mfa_probe_", dir=candidate)
            os.close(fd)
            os.remove(probe)
            return candidate
        except Exception:
            continue
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0] if candidates else os.path.join(r"C:\Users\Public", "UTAU_Auto_OTO_v3")


def get_default_mfa_env_dir():
    return os.path.join(_resolve_default_mfa_runtime_root(), ".env")


def get_default_mfa_conda_root():
    return os.path.join(_resolve_default_mfa_runtime_root(), "miniconda")


def get_default_mfa_micromamba_root():
    return os.path.join(_resolve_default_mfa_runtime_root(), "micromamba")


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
    parent_dir = os.path.dirname(app_dir)
    shared_env_dir = get_default_mfa_env_dir()
    env_candidates = [shared_env_dir]
    # Prefer payload-root env in portable frozen layout.
    if getattr(sys, 'frozen', False) and os.path.basename(app_dir).lower() in {"utau_auto_oto", "auto_oto"}:
        env_candidates.extend([
            os.path.join(parent_dir, '.env'),
            os.path.join(parent_dir, 'env'),
            os.path.join(app_dir, '.env'),
            os.path.join(app_dir, 'env'),
        ])
    else:
        env_candidates.extend([
            os.path.join(app_dir, '.env'),
            os.path.join(app_dir, 'env'),
            os.path.join(parent_dir, '.env'),
            os.path.join(parent_dir, 'env'),
        ])
    candidates = []
    for env_dir in env_candidates:
        if not env_dir:
            continue
        candidates.extend([
            os.path.join(env_dir, 'Scripts', 'mfa.exe'),
            os.path.join(env_dir, 'Scripts', 'mfa.bat'),
            os.path.join(env_dir, 'Scripts', 'mfa.cmd'),
            os.path.join(env_dir, 'bin', 'mfa'),
        ])
    seen = set()
    unique = []
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _link_or_copy(src, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _mfa_audio_preprocess_required() -> bool:
    low_rms = _env_bool("UTOA_MFA_LOW_RMS_GAIN_ENABLE", default=False)
    weak = _env_bool("UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE", default=False)
    return bool(low_rms or weak)


def _read_wav_float(path: str) -> tuple[np.ndarray, dict]:
    with wave.open(path, "rb") as wf:
        channels = int(wf.getnchannels() or 1)
        sampwidth = int(wf.getsampwidth() or 2)
        framerate = int(wf.getframerate() or 44100)
        comptype = str(wf.getcomptype() or "NONE")
        compname = str(wf.getcompname() or "not compressed")
        nframes = int(wf.getnframes() or 0)
        raw = wf.readframes(nframes)

    if nframes <= 0:
        meta = {
            "channels": channels,
            "sampwidth": sampwidth,
            "framerate": framerate,
            "comptype": comptype,
            "compname": compname,
        }
        return np.zeros((0, max(1, channels)), dtype=np.float32), meta

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        data_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data_i32 = (
            data_u8[:, 0].astype(np.int32)
            | (data_u8[:, 1].astype(np.int32) << 8)
            | (data_u8[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        data_i32 = (data_i32 ^ sign_bit) - sign_bit
        data = data_i32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        usable = (data.shape[0] // channels) * channels
        data = data[:usable].reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)

    meta = {
        "channels": channels,
        "sampwidth": sampwidth,
        "framerate": framerate,
        "comptype": comptype,
        "compname": compname,
    }
    return np.clip(data.astype(np.float32), -1.0, 1.0), meta


def _write_wav_float(path: str, audio: np.ndarray, meta: dict) -> None:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = np.clip(arr, -1.0, 1.0)

    channels = int(meta.get("channels", 1) or 1)
    sampwidth = int(meta.get("sampwidth", 2) or 2)
    framerate = int(meta.get("framerate", 44100) or 44100)
    comptype = str(meta.get("comptype", "NONE") or "NONE")
    compname = str(meta.get("compname", "not compressed") or "not compressed")

    if arr.shape[1] != channels:
        if channels <= 1:
            arr = np.mean(arr, axis=1, keepdims=True)
            channels = 1
        else:
            arr = np.tile(np.mean(arr, axis=1, keepdims=True), (1, channels))

    flat = arr.reshape(-1)
    if sampwidth == 1:
        pcm = np.clip(np.round((flat * 128.0) + 128.0), 0, 255).astype(np.uint8).tobytes()
    elif sampwidth == 2:
        pcm = np.clip(np.round(flat * 32768.0), -32768, 32767).astype(np.int16).tobytes()
    elif sampwidth == 3:
        ints = np.clip(np.round(flat * float(1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
        packed = np.empty((ints.size, 3), dtype=np.uint8)
        packed[:, 0] = (ints & 0xFF).astype(np.uint8)
        packed[:, 1] = ((ints >> 8) & 0xFF).astype(np.uint8)
        packed[:, 2] = ((ints >> 16) & 0xFF).astype(np.uint8)
        pcm = packed.tobytes()
    elif sampwidth == 4:
        pcm = np.clip(np.round(flat * float(1 << 31)), -(1 << 31), (1 << 31) - 1).astype(np.int32).tobytes()
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(int(sampwidth))
        wf.setframerate(int(framerate))
        try:
            wf.setcomptype(comptype, compname)
        except Exception:
            pass
        wf.writeframes(pcm)


def _preprocess_alignment_wav(src: str, dst: str) -> tuple[bool, str]:
    low_rms = _env_bool("UTOA_MFA_LOW_RMS_GAIN_ENABLE", default=False)
    weak_assist = _env_bool("UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE", default=False)
    if not (low_rms or weak_assist):
        _link_or_copy(src, dst)
        return True, "copied"

    try:
        audio, meta = _read_wav_float(src)
    except Exception as exc:
        _link_or_copy(src, dst)
        return False, f"read failed: {exc}"

    if audio.size <= 0:
        try:
            _write_wav_float(dst, audio, meta)
            return True, "empty"
        except Exception as exc:
            _link_or_copy(src, dst)
            return False, f"write empty failed: {exc}"

    x = np.asarray(audio, dtype=np.float32)
    mono = np.mean(x, axis=1)
    base_rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))

    if low_rms and base_rms > 1e-8:
        target_rms = _env_float("UTOA_MFA_LOW_RMS_TARGET", 0.11)
        target_rms = max(0.02, min(0.30, target_rms))
        max_gain_db = _env_float("UTOA_MFA_LOW_RMS_MAX_GAIN_DB", 14.0)
        max_gain = 10.0 ** (max(0.0, min(30.0, max_gain_db)) / 20.0)
        if base_rms < target_rms:
            gain = min(target_rms / max(base_rms, 1e-8), max_gain)
            x = x * float(gain)

    if weak_assist:
        mix = _env_float("UTOA_MFA_WEAK_VOICE_PREEMPH_MIX", 0.35)
        mix = max(0.0, min(1.0, mix))
        coeff = _env_float("UTOA_MFA_WEAK_VOICE_PREEMPH_COEFF", 0.97)
        coeff = max(0.0, min(0.99, coeff))
        if mix > 1e-6:
            pre = x.copy()
            pre[1:, :] = x[1:, :] - (coeff * x[:-1, :])
            pre[0, :] = x[0, :]
            x = ((1.0 - mix) * x) + (mix * pre)
            # Keep RMS stable after spectral tilt.
            out_rms = float(np.sqrt(np.mean(np.square(np.mean(x, axis=1)), dtype=np.float64)))
            if out_rms > 1e-8 and base_rms > 1e-8:
                x = x * float(base_rms / out_rms)

    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    try:
        _write_wav_float(dst, x, meta)
        return True, "processed"
    except Exception as exc:
        _link_or_copy(src, dst)
        return False, f"write failed: {exc}"


def _prepare_ascii_safe_alignment_workspace(
    wav_folder,
    dict_path,
    output_folder,
    temp_root: str = "",
    callback=None,
):
    token_src = "|".join([
        os.path.abspath(wav_folder or ""),
        os.path.abspath(dict_path or ""),
        os.path.abspath(output_folder or ""),
        "prep=1" if _mfa_audio_preprocess_required() else "prep=0",
    ])
    token = hashlib.sha1(token_src.encode("utf-8", errors="replace")).hexdigest()[:12]
    root_dir = str(temp_root or tempfile.gettempdir())
    base = os.path.join(root_dir, "utoa_mfa_ascii", token)
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    corpus_dir = os.path.join(base, "corpus")
    dict_dir = os.path.join(base, "dict")
    out_dir = os.path.join(base, "out")
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    preprocess_enabled = _mfa_audio_preprocess_required()
    wav_total = 0
    wav_processed = 0
    wav_failed = 0
    for fn in os.listdir(wav_folder):
        low = fn.lower()
        if low.endswith(".wav") or low.endswith(".lab") or low.endswith(".txt"):
            src = os.path.join(wav_folder, fn)
            if os.path.isfile(src):
                dst = os.path.join(corpus_dir, fn)
                if low.endswith(".wav") and preprocess_enabled:
                    wav_total += 1
                    ok, reason = _preprocess_alignment_wav(src, dst)
                    if ok and reason == "processed":
                        wav_processed += 1
                    elif not ok:
                        wav_failed += 1
                        if callback:
                            try:
                                callback(f"[MFA][AudioPrep] fallback copy: {fn} ({reason})")
                            except Exception:
                                pass
                else:
                    _link_or_copy(src, dst)

    ext = os.path.splitext(dict_path)[1] or ".txt"
    safe_dict_path = os.path.join(dict_dir, f"dictionary{ext}")
    shutil.copy2(dict_path, safe_dict_path)
    return {
        "base": base,
        "corpus_dir": corpus_dir,
        "dict_path": safe_dict_path,
        "output_dir": out_dir,
        "audio_preprocess_enabled": preprocess_enabled,
        "audio_preprocess_total": wav_total,
        "audio_preprocess_applied": wav_processed,
        "audio_preprocess_failed": wav_failed,
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
    log_fn("오류: Microsoft Visual C++ 14.0+ (C++ Build Tools)가 필요합니다.")
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
        pip_common_args = ['--no-cache-dir', '--disable-pip-version-check', '--retries', '5', '--timeout', '120']
        trusted_hosts = [
            '--trusted-host', 'pypi.org',
            '--trusted-host', 'files.pythonhosted.org',
            '--trusted-host', 'pypi.python.org',
        ]

        def _looks_like_tls_error(msg):
            text = (msg or '').lower()
            return (
                'ssl' in text and ('certificate' in text or 'secure channel' in text or 'tls' in text)
            ) or ('certificate verify failed' in text)
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
    """MFA 정렬 전에 compute-mfcc-feats 실행 가능 여부를 점검한다."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        return False, "MFA 실행 파일을 찾을 수 없습니다."

    env = _get_conda_env(mfa_path)
    candidates = []
    if sys.platform == 'win32' and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        candidates.append(os.path.join(env_dir, 'Library', 'bin', 'compute-mfcc-feats.exe'))
        candidates.append('compute-mfcc-feats.exe')
    candidates.append('compute-mfcc-feats')

    strict_preflight_gate = str(os.environ.get("UTOA_STRICT_MFA_PREFLIGHT", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    soft_permission_gate = str(os.environ.get("UTOA_MFA_SOFT_PERMISSION", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        soft_permission_retry_limit = max(0, int(str(os.environ.get("UTOA_MFA_SOFT_PERMISSION_RETRY_LIMIT", "1") or "1")))
    except Exception:
        soft_permission_retry_limit = 1
    last_not_found = None
    for candidate in candidates:
        try:
            # Windows 환경에서는 실행 파일 존재/권한을 우선 점검한다.
            subprocess.run(
                [candidate, '--help'],
                capture_output=True,
                text=False,
                timeout=15,
                env=env,
            )
            cache_key = os.path.normcase(os.path.abspath(str(mfa_path or "")))
            if cache_key in _MFA_PERMISSION_SOFT_RETRY_CACHE:
                _MFA_PERMISSION_SOFT_RETRY_CACHE.pop(cache_key, None)
            return True, ""
        except FileNotFoundError as e:
            last_not_found = e
            continue
        except PermissionError as e:
            if callback:
                callback(ALERT_MFA_PERMISSION_DENIED)
            err = (
                "compute-mfcc-feats 실행 권한이 없어 MFA 정렬을 진행할 수 없습니다. "
                "(WinError 5: Access denied)"
            )
            log(f"오류: {err}")
            log("   백신/보안 정책 또는 폴더 권한을 확인한 뒤 다시 시도해 주세요.")
            if not strict_preflight_gate and soft_permission_gate:
                cache_key = os.path.normcase(os.path.abspath(str(mfa_path or "")))
                retry_count = int(_MFA_PERMISSION_SOFT_RETRY_CACHE.get(cache_key, 0)) + 1
                _MFA_PERMISSION_SOFT_RETRY_CACHE[cache_key] = retry_count
                if retry_count <= soft_permission_retry_limit:
                    log(
                        f"[MFA] preflight soft-gate: 권한 오류를 경고로 전환하고 정렬 실행을 시도합니다. "
                        f"(soft retry {retry_count}/{soft_permission_retry_limit})"
                    )
                    return True, ""
                log(
                    f"[MFA] preflight soft-gate limit exceeded: 동일 권한 오류가 반복되어 하드 실패로 전환합니다. "
                    f"(limit={soft_permission_retry_limit})"
                )
            return False, f"{err}: {e}"
        except Exception as e:
            err = f"compute-mfcc-feats 점검 중 예외가 발생했습니다: {e}"
            log(f"오류: {err}")
            if strict_preflight_gate:
                return False, err
            log("[MFA] preflight soft-gate: 점검 예외를 경고로 전환하고 정렬 실행을 시도합니다.")
            return True, ""

    err = "compute-mfcc-feats를 찾지 못했습니다. MFA 환경 복구 또는 재설치를 시도해 주세요."
    log(f"오류: {err}")
    if not strict_preflight_gate:
        log("[MFA] preflight soft-gate: compute-mfcc-feats 미탐지 상태로 정렬 실행을 시도합니다.")
        return True, ""
    if last_not_found:
        return False, f"{err}: {last_not_found}"
    return False, err

def _validate_alignment_dictionary(dict_path: str, callback=None, soft: bool = False):
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
    valid_lines = []
    total = 0
    for idx, line in enumerate(str(text or "").splitlines(), start=1):
        row = str(line or "").strip()
        if not row or row.startswith("#"):
            valid_lines.append(line)
            continue
        total += 1
        parts = row.split()
        # Expected: <word> <phone1> [phone2 ...]
        if len(parts) < 2 or "\ufffd" in row:
            bad_lines.append((idx, row))
            continue
        valid_lines.append(line)

    if bad_lines:
        valid_count = max(total - len(bad_lines), 0)
        samples = "; ".join(f"{ln}:{txt[:40]}" for ln, txt in bad_lines[:5])
        err = (
            f"Dictionary malformed rows detected ({len(bad_lines)}/{max(total, 1)}). "
            f"examples={samples}"
        )
        if (not soft) or valid_count <= 0:
            log(err)
            if valid_count <= 0:
                log("[MFA] 해결 방법: 사전을 다시 생성하거나, 각 행이 '<word> <phone...>' 형식인지 확인해 주세요.")
            return False, err
        try:
            backup_path = f"{dict_path}.bak"
            if not os.path.exists(backup_path):
                shutil.copy2(dict_path, backup_path)
                log(f"[MFA] dictionary backup saved: {backup_path}")
            with open(dict_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(valid_lines).rstrip() + "\n")
            log(f"[MFA] soft dictionary validation applied: dropped={len(bad_lines)}/{max(total, 1)} rows")
            log(f"[MFA] dropped row samples: {samples}")
            log("[MFA] 해결 방법: Lab/사전 생성 단계를 다시 실행하거나 인코딩(UTF-8) 및 발음 열 형식을 점검해 주세요.")
            return True, f"Dictionary had malformed rows; dropped {len(bad_lines)} row(s) in soft mode."
        except Exception as e:
            log(f"{err} | soft repair failed: {e}")
            return False, f"{err} | soft repair failed: {e}"
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
    Build subprocess environment for MFA runtime.
    On Windows, prepend env-specific binary directories to PATH to avoid DLL load errors.
    """
    env = os.environ.copy()
    # Frozen launcher state can leak host Python vars into subprocesses and
    # break ensurepip/pip module resolution inside the MFA env.
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
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if sys.platform == 'win32' and mfa_path and 'Scripts' in mfa_path:
        mfa_path = os.path.abspath(mfa_path)
        env_dir = os.path.abspath(os.path.dirname(os.path.dirname(mfa_path)))
        site_packages = os.path.join(env_dir, 'Lib', 'site-packages')
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


def _korean_tokenizer_import_expr() -> str:
    # Korean path is ready when jamo is importable and one mecab backend is importable.
    # Accept both python-mecab-ko ("mecab") and mecab-python3 ("MeCab") module styles.
    return (
        "import sys\n"
        "import jamo\n"
        "ok=False\n"
        "try:\n"
        "    from mecab import MeCab\n"
        "    _m=MeCab()\n"
        "    _items=_m.parse('가')\n"
        "    ok=bool(_items)\n"
        "except Exception:\n"
        "    try:\n"
        "        from mecab import Tagger\n"
        "        _t=Tagger()\n"
        "        _n=_t.parseToNode('가')\n"
        "        ok=(_n is not None)\n"
        "    except Exception:\n"
        "        try:\n"
        "            import MeCab\n"
        "            _t=MeCab.Tagger()\n"
        "            _n=_t.parseToNode('가')\n"
        "            ok=(_n is not None)\n"
        "        except Exception:\n"
        "            try:\n"
        "                import mecab_ko\n"
        "                ok=True\n"
        "            except Exception:\n"
        "                ok=False\n"
        "sys.exit(0 if ok else 1)\n"
    )


def _resolve_get_pip_script() -> str:
    env_override = str(os.environ.get("UTOA_GET_PIP_PATH", "") or "").strip()
    if env_override and os.path.isfile(env_override):
        return env_override

    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meipass_dir = str(getattr(sys, "_MEIPASS", "") or "").strip()

    candidates = [
        os.path.join(base_dir, "assets", "bootstrap", "get-pip.py"),
        os.path.join(base_dir, "UTAU_Auto_OTO", "assets", "bootstrap", "get-pip.py"),
        os.path.join(base_dir, "get-pip.py"),
        os.path.join(os.getcwd(), "assets", "bootstrap", "get-pip.py"),
        os.path.join(os.getcwd(), "get-pip.py"),
    ]
    if meipass_dir:
        candidates.extend([
            os.path.join(meipass_dir, "assets", "bootstrap", "get-pip.py"),
            os.path.join(meipass_dir, "get-pip.py"),
        ])

    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def ensure_mfa_python_packaging_stack(mfa_path, callback=None):
    """
    Ensure pip/setuptools/pkg_resources/wheel are available inside the MFA env.
    This is a prerequisite for language-specific dependency repair on native Windows.
    """

    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or "Scripts" not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, "Scripts", "pip.exe")
    conda_exe = os.path.join(env_dir, "Scripts", "conda.exe")
    if not os.path.exists(python_exe):
        return False

    env = _get_conda_env(mfa_path)

    def _check_stack_ready() -> tuple[bool, str]:
        ok, detail = _check_env_imports(
            python_exe,
            env,
            "import setuptools; import pip; import wheel",
        )
        return bool(ok), detail or ""

    ok, detail = _check_stack_ready()
    if ok:
        return True

    pip_common_args = [
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--retries",
        "5",
        "--timeout",
        "120",
    ]
    trusted_hosts = [
        "--trusted-host", "pypi.org",
        "--trusted-host", "files.pythonhosted.org",
        "--trusted-host", "pypi.python.org",
    ]

    index_args = []
    index_url = str(os.environ.get("UTOA_PIP_INDEX_URL", "") or "").strip()
    extra_index_url = str(os.environ.get("UTOA_PIP_EXTRA_INDEX_URL", "") or "").strip()
    if index_url:
        index_args.extend(["--index-url", index_url])
    if extra_index_url:
        index_args.extend(["--extra-index-url", extra_index_url])

    def _looks_like_tls_error(text):
        msg = (text or "").lower()
        return (
            ("ssl" in msg and ("certificate" in msg or "secure channel" in msg or "tls" in msg))
            or ("certificate verify failed" in msg)
            or ("trust relationship" in msg)
        )

    def _looks_like_connectivity_error(text):
        msg = (text or "").lower()
        return (
            "timed out" in msg
            or "connection reset" in msg
            or "name or service not known" in msg
            or "temporary failure in name resolution" in msg
            or "nodename nor servname provided" in msg
            or "failed to establish a new connection" in msg
        )

    def _run_repair(cmd, env_override=None):
        log(f"   -> repair cmd: {' '.join(cmd)}")
        result = _run_subprocess_text(cmd, env=(env_override or env))
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()
            if tail:
                log(f"[MFA] repair failed: {tail[:500]}")
        return result

    def _download_online_get_pip() -> str:
        if str(os.environ.get("UTOA_DISABLE_ONLINE_GET_PIP", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
            return ""
        url_candidates = []
        env_url = str(os.environ.get("UTOA_GET_PIP_URL", "") or "").strip()
        if env_url:
            url_candidates.append(env_url)
        url_candidates.append("https://bootstrap.pypa.io/get-pip.py")

        for url in url_candidates:
            try:
                with urllib.request.urlopen(url, timeout=90) as resp:
                    payload = resp.read()
                if not payload or len(payload) < 8192:
                    continue
                target = os.path.join(tempfile.gettempdir(), f"utoa_get_pip_{int(time.time())}.py")
                with open(target, "wb") as fp:
                    fp.write(payload)
                if os.path.isfile(target):
                    log(f"[MFA] Downloaded online get-pip.py: {target}")
                    return target
            except Exception as e:
                log(f"[MFA] online get-pip download failed: {url} ({e})")
                continue
        return ""

    log("[MFA] Python packaging tools (pip/setuptools/wheel) repair started...")

    saw_tls_error = False
    saw_connectivity_error = False

    pip_targets = ["pip", "setuptools<81", "wheel"]
    setuptools_wheel_targets = ["setuptools<81", "wheel"]

    is_conda_managed_env = bool(
        os.path.isdir(os.path.join(env_dir, "conda-meta")) or os.path.exists(conda_exe)
    )
    stage_cmds = []
    if not is_conda_managed_env:
        stage_cmds.extend([
            [python_exe, "-m", "ensurepip", "--upgrade"],
            [python_exe, "-m", "ensurepip", "--default-pip"],
        ])
    else:
        log("[MFA] Conda-managed Python detected; skipping ensurepip stage.")

    stage_cmds.append(
        [python_exe, "-m", "pip", "install", "--upgrade", *pip_common_args, *index_args, *pip_targets]
    )
    if os.path.exists(pip_exe):
        # pip.exe cannot reliably self-upgrade pip on Windows.
        stage_cmds.append([pip_exe, "install", "--upgrade", *pip_common_args, *index_args, *setuptools_wheel_targets])
    if os.path.exists(conda_exe):
        stage_cmds.append([
            conda_exe,
            "install",
            "-y",
            "--solver",
            "classic",
            "-p",
            env_dir,
            "pip",
            "setuptools",
            "wheel",
        ])
    system_conda = shutil.which("conda")
    if system_conda:
        stage_cmds.append([
            system_conda,
            "install",
            "-y",
            "--solver",
            "classic",
            "-p",
            env_dir,
            "pip",
            "setuptools",
            "wheel",
        ])

    for repair_cmd in stage_cmds:
        result = _run_repair(repair_cmd)
        if result.returncode != 0:
            err_txt = (result.stderr or result.stdout or "").strip()
            saw_tls_error = saw_tls_error or _looks_like_tls_error(err_txt)
            saw_connectivity_error = saw_connectivity_error or _looks_like_connectivity_error(err_txt)
            continue
        ok, detail = _check_stack_ready()
        if ok:
            return True
        if detail:
            log(f"[MFA] packaging import still failing: {detail[:500]}")

    if saw_tls_error or saw_connectivity_error:
        tls_cmds = [
            [python_exe, "-m", "pip", "install", "--upgrade", *pip_common_args, *index_args, *trusted_hosts, *pip_targets],
        ]
        if os.path.exists(pip_exe):
            tls_cmds.append([pip_exe, "install", "--upgrade", *pip_common_args, *index_args, *trusted_hosts, *setuptools_wheel_targets])
        for repair_cmd in tls_cmds:
            result = _run_repair(repair_cmd)
            if result.returncode != 0:
                continue
            ok, detail = _check_stack_ready()
            if ok:
                return True
            if detail:
                log(f"[MFA] packaging import still failing: {detail[:500]}")

    force_cmds = [
        [python_exe, "-m", "pip", "install", "--upgrade", "--force-reinstall", *pip_common_args, *index_args, *trusted_hosts, *pip_targets],
    ]
    if os.path.exists(pip_exe):
        force_cmds.append([pip_exe, "install", "--upgrade", "--force-reinstall", *pip_common_args, *index_args, *trusted_hosts, *setuptools_wheel_targets])
    for repair_cmd in force_cmds:
        result = _run_repair(repair_cmd)
        if result.returncode != 0:
            continue
        ok, detail = _check_stack_ready()
        if ok:
            return True
        if detail:
            log(f"[MFA] packaging import still failing: {detail[:500]}")

    get_pip_path = _resolve_get_pip_script()
    if not get_pip_path:
        get_pip_path = _download_online_get_pip()

    if get_pip_path:
        log(f"[MFA] Falling back to get-pip.py: {get_pip_path}")
        env_boot = dict(env or {})
        env_boot.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env_boot.setdefault("PIP_NO_CACHE_DIR", "1")
        env_boot.setdefault("PIP_DEFAULT_TIMEOUT", "120")
        env_boot.setdefault("PIP_RETRIES", "5")

        get_pip_cmds = [
            [
                python_exe,
                get_pip_path,
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--retries",
                "5",
                "--timeout",
                "120",
                "pip",
                "setuptools<81",
                "wheel",
            ],
            [
                python_exe,
                get_pip_path,
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--retries",
                "5",
                "--timeout",
                "120",
                *trusted_hosts,
                "pip",
                "setuptools<81",
                "wheel",
            ],
        ]

        for repair_cmd in get_pip_cmds:
            result = _run_repair(repair_cmd, env_override=env_boot)
            if result.returncode != 0:
                continue
            post_cmd = [
                python_exe,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                *pip_common_args,
                *index_args,
                *trusted_hosts,
                "pip",
                "setuptools<81",
                "wheel",
            ]
            _run_repair(post_cmd, env_override=env_boot)
            ok, detail = _check_stack_ready()
            if ok:
                return True
            if detail:
                log(f"[MFA] packaging import still failing: {detail[:500]}")
    else:
        log("[MFA] Bundled/online get-pip.py could not be resolved.")

    if detail:
        log(f"[MFA] packaging import failed after repairs: {detail[:500]}")
    log("[MFA] Failed to restore pip/setuptools/wheel")
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
            "model_ready": False,
        },
    }

    if not resolved or not os.path.exists(resolved):
        report["issues"].append("mfa_missing")
        log("[MFA] MFA executable was not found.")
        return report

    report["checks"]["mfa_executable"] = True
    if 'Scripts' in resolved:
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
            ok, _detail = _check_env_imports(python_exe, env, 'import setuptools; import pip; import wheel')
            report["checks"]["packaging_stack"] = ok
            if not ok:
                report["issues"].append("packaging_stack_missing")
            import_expr = (
                _korean_tokenizer_import_expr()
                if lang == 'korean'
                else 'import spacy; import sudachipy; import sudachidict_core'
            )
            lang_ok, _detail = _check_env_imports(python_exe, env, import_expr)
            report["checks"]["language_support"] = lang_ok
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
    Detect which single-speaker flag is supported by current MFA version.
    Uses `mfa align --help` and picks `--single-speaker` or `--single_speaker`.
    """
    key = os.path.abspath(mfa_path or "")
    cached = _MFA_SINGLE_SPEAKER_FLAG_CACHE.get(key)
    if cached:
        return cached

    candidates = ["--single-speaker", "--single_speaker"]
    try:
        res = subprocess.run(
            [mfa_path, "align", "--help"],
            capture_output=True,
            text=False,
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

    # Safe fallback for old MFA versions when help probing fails.
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
        res = subprocess.run(
            [mfa_path, "align", "--help"],
            capture_output=True,
            text=False,
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
    return profile, opts


def find_mfa_executable():
    """
    현재 실행 환경에서 MFA 실행 파일 경로를 탐색한다.
    검색 순서: 후보 경로 -> PATH -> Conda 후보 경로

    Returns:
        MFA 실행 파일 경로 또는 None
    """
    # 1) Portable/로컬 후보 경로
    for p in _candidate_mfa_executable_paths():
        if os.path.exists(p):
            logger.info(f"기본 후보 경로에서 MFA 실행 파일 발견: {p}")
            return p

    # 2) PATH
    mfa_path = shutil.which('mfa')
    if mfa_path:
        logger.info(f"PATH에서 MFA 실행 파일 발견: {mfa_path}")
        return mfa_path

    # 3) Conda 후보 경로
    conda_paths = [
        os.path.expanduser('~/miniconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/anaconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/miniconda3/Scripts/mfa.exe'),
    ]
    for p in conda_paths:
        if os.path.exists(p):
            logger.info(f"Conda 후보 경로에서 MFA 실행 파일 발견: {p}")
            return p

    return None

def check_mfa_model(mfa_path, language='korean'):
    """
    Check whether MFA acoustic model is available for selected language.

    Args:
        mfa_path: MFA executable path
        language: 'korean' or 'japanese'

    Returns:
        (is_ready: bool, message: str)
    """
    if not mfa_path:
        return False, "MFA executable not found."

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'

    try:
        env = _get_conda_env(mfa_path)
        result = subprocess.run(
            [mfa_path, 'model', 'list', 'acoustic'],
            capture_output=True, text=False, timeout=30, env=env
        )
        stdout_text = _decode_subprocess_output(result.stdout)
        stderr_text = _decode_subprocess_output(result.stderr)
        combined_text = f"{stdout_text}\n{stderr_text}"
        if model_name in combined_text:
            return True, f"{lang_label} MFA model is available."
        if _has_local_acoustic_model_artifact(mfa_path, model_name, env=env):
            return True, f"{lang_label} MFA model is available (local artifact found)."
        if result.returncode != 0:
            return False, (
                f"{lang_label} MFA model check failed (code={result.returncode}). "
                "Please re-download the model."
            )
        return False, f"{lang_label} MFA model is missing. Please run model download."
    except Exception as e:
        return False, f"MFA model check failed: {e}"


def _candidate_mfa_root_dirs(mfa_path: str, env: Optional[dict] = None) -> List[str]:
    roots: List[str] = []

    def _add(path: str):
        p = os.path.abspath(str(path or "").strip())
        if p and p not in roots:
            roots.append(p)

    try:
        env_root = str((env or {}).get("MFA_ROOT_DIR", "")).strip()
        if env_root:
            _add(env_root)
    except Exception:
        pass

    try:
        _add(_default_mfa_root_dir(mfa_path, per_process=False))
    except Exception:
        pass

    try:
        for runtime_root in _candidate_mfa_runtime_roots():
            _add(os.path.join(runtime_root, ".mfa_root_ascii"))
    except Exception:
        pass

    try:
        if mfa_path:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
            if os.path.isdir(app_dir):
                for name in os.listdir(app_dir):
                    if name.startswith(".mfa_root_ascii_p"):
                        _add(os.path.join(app_dir, name))
    except Exception:
        pass

    # Legacy shared MFA root (still checked for compatibility).
    _add(os.path.expanduser("~/Documents/MFA"))
    return roots


def _find_local_acoustic_model_artifact(mfa_path: str, model_name: str, env: Optional[dict] = None) -> str:
    if not model_name:
        return ""
    for root in _candidate_mfa_root_dirs(mfa_path, env=env):
        acoustic_dir = os.path.join(root, "pretrained_models", "acoustic")
        candidates = [
            os.path.join(acoustic_dir, model_name),
            os.path.join(acoustic_dir, f"{model_name}.zip"),
            os.path.join(acoustic_dir, f"{model_name}.yaml"),
            os.path.join(acoustic_dir, f"{model_name}.yml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return ""


def _has_local_acoustic_model_artifact(mfa_path: str, model_name: str, env: Optional[dict] = None) -> bool:
    return bool(_find_local_acoustic_model_artifact(mfa_path, model_name, env=env))


def _resolve_acoustic_model_reference(mfa_path: str, model_name: str, env: Optional[dict] = None) -> str:
    if not mfa_path or not model_name:
        return model_name
    try:
        result = _run_subprocess_text([mfa_path, "model", "list", "acoustic"], env=env)
        combined = f"{result.stdout}\n{result.stderr}"
        if model_name in combined:
            return model_name
    except Exception:
        pass
    local_artifact = _find_local_acoustic_model_artifact(mfa_path, model_name, env=env)
    if local_artifact:
        return local_artifact
    return model_name


def check_mfa_ready(language='korean', mfa_path=''):
    resolved_mfa = mfa_path or find_mfa_executable()
    if not resolved_mfa or not os.path.exists(resolved_mfa):
        return make_runtime_report(
            "align",
            ALIGN_EXEC_MISSING,
            "MFA executable not found.",
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
            msg or "MFA model is missing.",
            engine="mfa",
            language=str(language or "korean").strip().lower(),
            mfa_path=str(resolved_mfa or ""),
            ready=False,
        )

    return make_runtime_report(
        "align",
        OK,
        "MFA runtime is ready.",
        engine="mfa",
        language=str(language or "korean").strip().lower(),
        mfa_path=str(resolved_mfa or ""),
        ready=True,
    )


def ensure_korean_support(mfa_path, callback=None):
    """
    Ensure Korean MFA tokenizer dependencies are available:
    - jamo
    - python-mecab-ko or mecab-python3
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False
    if 'Scripts' not in mfa_path:
        # System MFA path; skip env-local auto install here.
        return True
    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    safe_cwd = _safe_env_subprocess_cwd(env_dir)
    pip_common_args = [
        '--no-cache-dir',
        '--disable-pip-version-check',
        '--retries',
        '5',
        '--timeout',
        '120',
    ]
    trusted_hosts = [
        '--trusted-host', 'pypi.org',
        '--trusted-host', 'files.pythonhosted.org',
        '--trusted-host', 'pypi.python.org',
    ]
    index_args = []
    index_url = str(os.environ.get("UTOA_PIP_INDEX_URL", "") or "").strip()
    extra_index_url = str(os.environ.get("UTOA_PIP_EXTRA_INDEX_URL", "") or "").strip()
    if index_url:
        index_args.extend(['--index-url', index_url])
    if extra_index_url:
        index_args.extend(['--extra-index-url', extra_index_url])
    if not os.path.exists(python_exe):
        return False
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log('[MFA] Failed to prepare base Python packaging tools before Korean dependency install')
        return False
    pkg_check_cmd = [python_exe, '-c', 'import setuptools, pip, wheel']
    check_cmd = [python_exe, '-c', _korean_tokenizer_import_expr()]
    try:
        env = _get_conda_env(mfa_path)

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
                if not os.path.isfile(mecab_init):
                    return
                with open(mecab_init, 'r', encoding='utf-8') as f:
                    content = f.read()
                marker = "\n# UTOA_MECAB_SHIM"
                if marker not in content:
                    return
                repaired = content.split(marker, 1)[0].rstrip() + "\n"
                if repaired != content:
                    with open(mecab_init, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(repaired)
                    log('[MFA] Removed legacy mecab shim from mecab package (__init__.py).')
            except Exception as e:
                log(f"[MFA] Failed to patch mecab shim: {e}")

        def _looks_like_pyexpat_dll_issue(msg):
            s = (msg or '').lower()
            return ('pyexpat' in s and 'dll load failed' in s) or ('libexpat' in s and 'not found' in s)

        def _looks_like_tls_error(msg):
            s = (msg or '').lower()
            return (
                ('ssl' in s and ('certificate' in s or 'secure channel' in s or 'tls' in s))
                or ('certificate verify failed' in s)
                or ('trust relationship' in s)
            )

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
                res = _run_subprocess_text(cmd, env=env, cwd=safe_cwd)
                if res.returncode == 0:
                    return True
            return False

        def _check_imports():
            res = _run_subprocess_text(check_cmd, env=env, cwd=safe_cwd)
            if res.returncode == 0:
                return True, ''
            detail = (res.stderr or res.stdout or '').strip()
            return False, detail

        def _ensure_pkg_resources():
            res = _run_subprocess_text(pkg_check_cmd, env=env, cwd=safe_cwd)
            if res.returncode == 0:
                return True
            err_detail = (res.stderr or res.stdout or "").strip()
            if err_detail:
                log(f"[MFA] pkg_resources import failed: {err_detail[:500]}")
            log('[MFA] pkg_resources is missing; repairing setuptools first')
            install_cmds = [
                [python_exe, '-m', 'pip', 'install', '--upgrade', 'setuptools<81'],
            ]
            if os.path.exists(pip_exe):
                install_cmds.append([pip_exe, 'install', '--upgrade', 'setuptools<81'])
            conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
            if os.path.exists(conda_exe):
                install_cmds.append([conda_exe, 'install', '-y', '--solver', 'classic', '-p', env_dir, 'setuptools'])
            system_conda = shutil.which('conda')
            if system_conda:
                install_cmds.append([system_conda, 'install', '-y', '--solver', 'classic', '-p', env_dir, 'setuptools'])
            for install_cmd in install_cmds:
                log(f"   -> repair cmd: {' '.join(install_cmd)}")
                result = _run_subprocess_text(install_cmd, env=env, cwd=safe_cwd)
                if result.returncode != 0:
                    continue
                verify = _run_subprocess_text(pkg_check_cmd, env=env, cwd=safe_cwd)
                if verify.returncode == 0:
                    return True
            return False

        if not _ensure_pkg_resources():
            log('[MFA] Failed to restore pkg_resources/setuptools')
            return False

        _ensure_mecab_dictionary()
        _ensure_mecab_shim()
        ok, detail = _check_imports()
        if (not ok) and _looks_like_pyexpat_dll_issue(detail):
            log('[MFA] Detected pyexpat/libexpat DLL issue; trying repair...')
            if _try_repair_pyexpat():
                ok, detail = _check_imports()
        if ok:
            patch_mfa_korean_support(mfa_path, callback)
            return True

        def _run_uninstall_stage(packages):
            pkgs = [str(p).strip() for p in packages if str(p).strip()]
            if not pkgs:
                return True
            args = ['uninstall', '-y', *pkgs]
            cmds = [[python_exe, '-m', 'pip', *args]]
            if os.path.exists(pip_exe):
                cmds.append([pip_exe, *args])
            for uninstall_cmd in cmds:
                log(f"   -> cleanup cmd: {' '.join(uninstall_cmd)}")
                result = _run_subprocess_text(uninstall_cmd, env=env, cwd=safe_cwd)
                if result.returncode == 0:
                    return True
            return False

        def _run_install_stage(packages, *, pip_extra_args=None):
            extra_args = list(pip_extra_args or [])
            stage_last_err = ''

            def _build_cmds(extra):
                args = ['install', '--upgrade', *pip_common_args, *index_args, *extra, *packages]
                cmds = [[python_exe, '-m', 'pip', *args]]
                if os.path.exists(pip_exe):
                    cmds.append([pip_exe, *args])
                system_conda = shutil.which('conda')
                if system_conda:
                    cmds.append([
                        system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip',
                        *args
                    ])
                return cmds

            for pass_idx in range(2):
                use_trusted = pass_idx == 1
                trusted_extra = trusted_hosts if use_trusted else []
                stage_cmds = _build_cmds(extra_args + trusted_extra)
                saw_tls = False
                for install_cmd in stage_cmds:
                    log(f"   -> cmd: {' '.join(install_cmd)}")
                    result = _run_subprocess_text(install_cmd, env=env, cwd=safe_cwd)
                    if result.returncode != 0:
                        err_txt = (result.stderr or result.stdout or '').strip()
                        if err_txt:
                            log(f"   [warn] install failed: {err_txt[:500]}")
                        stage_last_err = err_txt or stage_last_err
                        if _looks_like_tls_error(err_txt):
                            saw_tls = True
                        continue
                    if not _ensure_pkg_resources():
                        stage_last_err = 'pkg_resources/setuptools repair failed after install'
                        continue
                    ok_local, detail_local = _check_imports()
                    if (not ok_local) and _looks_like_pyexpat_dll_issue(detail_local):
                        log('[MFA] Detected pyexpat/libexpat DLL issue after install; trying repair...')
                        if _try_repair_pyexpat():
                            ok_local, detail_local = _check_imports()
                    if ok_local:
                        return True, stage_last_err
                    if detail_local:
                        log(f"   [warn] import check failed after install: {detail_local[:500]}")
                        stage_last_err = detail_local
                if not saw_tls:
                    break

            return False, stage_last_err

        last_err = detail

        # Stage 1: install jamo first (pure Python).
        log('[MFA] Installing Korean tokenizer deps: jamo')
        ok, stage_err = _run_install_stage(['jamo'])
        if stage_err:
            last_err = stage_err
        if ok:
            log('[MFA] Korean tokenizer deps are ready (jamo + mecab backend)')
            patch_mfa_korean_support(mfa_path, callback)
            return True

        # Stage 2: python-mecab-ko wheels only (avoid source-build toolchain on Windows).
        log('[MFA] Installing Korean tokenizer deps: python-mecab-ko + dictionary')
        _run_uninstall_stage(['mecab-python3'])
        ok, stage_err = _run_install_stage(
            ['python-mecab-ko', 'python-mecab-ko-dic'],
            pip_extra_args=['--only-binary=:all:', '--force-reinstall'],
        )
        if stage_err:
            last_err = stage_err
        if ok:
            log('[MFA] Korean tokenizer deps are ready')
            patch_mfa_korean_support(mfa_path, callback)
            return True

        # Stage 3: fallback to mecab-python3 wheels only.
        log('[MFA] Installing Korean tokenizer deps fallback: mecab-python3')
        _run_uninstall_stage(['python-mecab-ko', 'python-mecab-ko-dic'])
        ok, stage_err = _run_install_stage(
            ['mecab-python3'],
            pip_extra_args=['--only-binary=:all:'],
        )
        if stage_err:
            last_err = stage_err
        if ok:
            log('[MFA] Korean tokenizer deps are ready')
            patch_mfa_korean_support(mfa_path, callback)
            return True

        log('[MFA] Failed to prepare Korean tokenizer deps (jamo, python-mecab-ko, mecab-python3)')
        if last_err:
            log(f"   last error: {last_err[:500]}")
        return False
    except Exception as e:
        log(f"[MFA] Korean dependency setup error: {e}")
        return False

def ensure_japanese_support(mfa_path, callback=None):
    """
    Prepare Japanese tokenizer/runtime dependencies for MFA.
    Installs spacy/sudachipy/sudachidict-core and verifies imports.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        log("[MFA] Japanese dependency check skipped: mfa_path is empty.")
        return False
    if 'Scripts' not in mfa_path:
        log("[MFA] Japanese dependency check is uncertain: non-portable MFA path (Scripts missing).")
        return False

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
    safe_cwd = _safe_env_subprocess_cwd(env_dir)
    pip_common_args = [
        '--no-cache-dir',
        '--disable-pip-version-check',
        '--retries',
        '5',
        '--timeout',
        '120',
    ]
    trusted_hosts = [
        '--trusted-host', 'pypi.org',
        '--trusted-host', 'files.pythonhosted.org',
        '--trusted-host', 'pypi.python.org',
    ]
    index_args = []
    index_url = str(os.environ.get("UTOA_PIP_INDEX_URL", "") or "").strip()
    extra_index_url = str(os.environ.get("UTOA_PIP_EXTRA_INDEX_URL", "") or "").strip()
    if index_url:
        index_args.extend(['--index-url', index_url])
    if extra_index_url:
        index_args.extend(['--extra-index-url', extra_index_url])

    if not os.path.exists(python_exe):
        log("[MFA] Japanese dependency check is uncertain: env python.exe not found.")
        return False
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log("[MFA] Failed to prepare Python packaging tools before Japanese dependency install.")
        return False

    check_cmd = [python_exe, '-c', 'import spacy; import sudachipy; import sudachidict_core']
    try:
        env = _get_conda_env(mfa_path)

        def _looks_like_tls_error(msg):
            s = (msg or '').lower()
            return (
                ('ssl' in s and ('certificate' in s or 'secure channel' in s or 'tls' in s))
                or ('certificate verify failed' in s)
                or ('trust relationship' in s)
            )

        def _run_pip_install(packages):
            last_err = ""
            for pass_idx in range(2):
                use_trusted = pass_idx == 1
                extra = trusted_hosts if use_trusted else []
                cmds = [
                    [python_exe, '-m', 'pip', 'install', '--upgrade', *pip_common_args, *index_args, *extra, *packages]
                ]
                if os.path.exists(pip_exe):
                    cmds.append([pip_exe, 'install', '--upgrade', *pip_common_args, *index_args, *extra, *packages])
                saw_tls = False
                for pip_cmd in cmds:
                    log(f"   -> fallback pip cmd: {' '.join(pip_cmd)}")
                    pip_result = _run_subprocess_text(pip_cmd, env=env, cwd=safe_cwd)
                    if pip_result.returncode == 0:
                        return True, last_err
                    err_txt = (pip_result.stderr or pip_result.stdout or '').strip()
                    if err_txt:
                        log(f"   pip stderr/stdout: {err_txt[:500]}")
                    last_err = err_txt or last_err
                    if _looks_like_tls_error(err_txt):
                        saw_tls = True
                if not saw_tls:
                    break
            return False, last_err

        result = _run_subprocess_text(check_cmd, env=env, cwd=safe_cwd)
        if result.returncode == 0:
            return True

        log("[MFA] Installing Japanese tokenizer dependencies (spacy, sudachipy, sudachidict-core)...")

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
            log("[MFA] No conda/pip installer is available for Japanese dependency setup.")
            return False

        log(f"   -> install cmd: {' '.join(install_cmd)}")
        install_result = _run_subprocess_text(install_cmd, env=env, cwd=safe_cwd)
        if install_result.returncode != 0:
            if install_result.stderr:
                log(f"   install stderr: {install_result.stderr[:500]}")
            if install_result.stdout:
                log(f"   install stdout: {install_result.stdout[:500]}")
            pip_ok, pip_err = _run_pip_install(['spacy', 'sudachipy', 'sudachidict-core'])
            if not pip_ok:
                if pip_err:
                    log(f"   pip fallback failed: {pip_err[:500]}")
                return False

        verify = _run_subprocess_text(check_cmd, env=env, cwd=safe_cwd)
        if verify.returncode == 0:
            log("[MFA] Japanese tokenizer dependencies are ready.")
            return True

        log("[MFA] Japanese dependency install completed, but import verification still failed.")
        if verify.stderr:
            log(f"   verify stderr: {verify.stderr[:500]}")
        return False
    except Exception as e:
        log(f"[MFA] Japanese dependency setup error: {e}")
        return False


def download_mfa_model(mfa_path, language='korean', callback=None):
    """Download MFA acoustic model for selected language."""
    interval_raw = str(os.environ.get("UTOA_MFA_MODEL_LOG_INTERVAL_SEC", "30") or "").strip()
    try:
        interval_sec = float(interval_raw)
    except Exception:
        interval_sec = 30.0
    interval_sec = max(5.0, min(600.0, interval_sec))

    last_emit = {"ts": 0.0, "pending": ""}

    def _emit_throttled(msg, force=False):
        if not callback:
            return
        now = time.monotonic()
        if force or (now - last_emit["ts"]) >= interval_sec:
            last_emit["ts"] = now
            last_emit["pending"] = ""
            callback(msg)
        else:
            last_emit["pending"] = msg

    def _flush_pending():
        pending = str(last_emit.get("pending", "") or "").strip()
        if not pending or not callback:
            return
        last_emit["pending"] = ""
        last_emit["ts"] = time.monotonic()
        callback(pending)

    def log(msg, stream=False):
        logger.info(msg)
        if callback:
            _emit_throttled(msg, force=not stream)
    if not mfa_path:
        log('MFA executable not found.')
        return False
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'

    has_model, msg = check_mfa_model(mfa_path, language=language)
    if has_model:
        if msg:
            log(msg)
        return True

    # Dependency preparation is best-effort: continue model download even when it fails.
    # Import verification runs again after download in the regular readiness checks.
    try:
        if language == 'korean':
            if not ensure_korean_support(mfa_path, callback):
                log('[MFA] Korean dependency prepare failed, but model download will continue.')
        elif language == 'japanese':
            if not ensure_japanese_support(mfa_path, callback):
                log('[MFA] Japanese dependency prepare failed, but model download will continue.')
    except Exception as dep_exc:
        log(f'[MFA] Dependency prepare check raised error, but model download will continue: {dep_exc}')

    log(f'Downloading {lang_label} MFA model...')
    try:
        env = _get_conda_env(mfa_path)
        attempts = [
            [mfa_path, 'model', 'download', 'acoustic', model_name, '--ignore_cache'],
            [mfa_path, 'model', 'download', 'acoustic', model_name],
        ]
        for idx, cmd in enumerate(attempts, start=1):
            log(f"[MFA] model download attempt {idx}/{len(attempts)}: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                env=env,
            )
            seen_lines: List[str] = []
            if process.stdout:
                for raw_line in iter(process.stdout.readline, b""):
                    line = _decode_subprocess_output(raw_line).strip()
                    if line:
                        seen_lines.append(line)
                        log(line, stream=True)
            process.wait()
            _flush_pending()
            if process.returncode == 0:
                log(f'{lang_label} MFA model download completed.')
                return True

            has_model, _msg = check_mfa_model(mfa_path, language=language)
            if has_model:
                log(f'{lang_label} MFA model is present despite non-zero downloader exit code. Continuing.')
                return True

            tail = " | ".join(seen_lines[-6:])[:500]
            if tail:
                log(f"[MFA] download attempt {idx} failed (code={process.returncode}): {tail}")
            else:
                log(f"[MFA] download attempt {idx} failed (code={process.returncode})")

        log('Model download failed after retries.')
        return False
    except Exception as e:
        log(f'Model download error: {e}')
        return False


def _normalize_alignment_strict_mode(value) -> str:
    text = str(value or "").strip().lower()
    compact = text.replace(" ", "").replace("-", "_")
    if "strict mode" in text:
        return "strict"
    if ("moderate mode" in text or "balanced mode" in text or "soft strict" in text):
        return "moderate"
    if compact in {
        "strict",
        "full_strict",
        "hard",
        "hard_strict",
    }:
        return "strict"
    if compact in {
        "moderate",
        "medium",
        "balanced",
        "soft_strict",
        "fallback",
    }:
        return "moderate"
    return "off"


def _resolve_mfa_runtime_options(
    *,
    resolved_profile: str,
    runtime_options: Optional[Dict[str, object]] = None,
    callback=None,
) -> Dict[str, object]:
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    raw = dict(runtime_options or {})
    constrained_mode = _normalize_alignment_strict_mode(raw.get("constrained_mode", "off"))
    recursive_requested = bool(raw.get("recursive_mfa", True))
    recursive_enabled = recursive_requested and constrained_mode in {"strict", "moderate"}
    recursive_skip_reason = ""
    if resolved_profile == "fast":
        if recursive_enabled:
            recursive_skip_reason = "fast profile"
        recursive_enabled = False
    elif resolved_profile == "default" and constrained_mode != "strict":
        # Keep default profile responsive unless user explicitly requests strict behavior.
        if recursive_enabled:
            recursive_skip_reason = "default profile"
        recursive_enabled = False

    chunk_size_default = 72 if resolved_profile == "accurate" else 96
    chunk_size = chunk_size_default
    raw_chunk_size = str(raw.get("recursive_chunk_size", "") or "").strip()
    if raw_chunk_size:
        try:
            chunk_size = max(12, min(240, int(float(raw_chunk_size))))
        except Exception:
            chunk_size = chunk_size_default

    max_depth = 8
    raw_max_depth = str(raw.get("recursive_max_depth", "") or "").strip()
    if raw_max_depth:
        try:
            max_depth = max(2, min(12, int(float(raw_max_depth))))
        except Exception:
            max_depth = 8

    beam_scale = 1.0
    if constrained_mode == "strict":
        beam_scale = 0.72
    elif constrained_mode == "moderate":
        beam_scale = 0.84

    if constrained_mode in {"strict", "moderate"}:
        log(
            f"[MFA][constrained] mode={constrained_mode}, "
            f"recursive={'on' if recursive_enabled else 'off'}, "
            f"profile={resolved_profile}"
        )
        if recursive_skip_reason:
            log(f"[MFA][recursive] skipped by profile policy ({recursive_skip_reason}).")

    return {
        "constrained_mode": constrained_mode,
        "recursive_mfa": recursive_enabled,
        "recursive_chunk_size": int(chunk_size),
        "recursive_max_depth": int(max_depth),
        "beam_scale": float(beam_scale),
    }


def _collect_alignment_token_sequence(corpus_dir: str) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    if not os.path.isdir(corpus_dir):
        return tokens
    for filename in sorted(os.listdir(corpus_dir)):
        low = filename.lower()
        if not (low.endswith(".lab") or low.endswith(".txt")):
            continue
        path = os.path.join(corpus_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        for tok in str(text or "").replace("\n", " ").replace("\t", " ").split():
            token = str(tok or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def _build_constrained_dictionary(
    base_dict_path: str,
    corpus_dir: str,
    *,
    mode: str = "moderate",
    callback=None,
) -> Tuple[bool, str, str, Dict[str, int]]:
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    stats = {
        "token_total": 0,
        "selected": 0,
        "missing": 0,
        "duplicates": 0,
    }
    mode_code = _normalize_alignment_strict_mode(mode)
    tokens = _collect_alignment_token_sequence(corpus_dir)
    stats["token_total"] = len(tokens)
    if not tokens:
        return False, "No lab tokens found for constrained dictionary build.", "", stats

    entries: Dict[str, str] = {}
    duplicate_count = 0
    try:
        with open(base_dict_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        return False, f"Failed to read dictionary: {exc}", "", stats

    for raw_line in lines:
        stripped = str(raw_line or "").strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        pron = " ".join(parts[1:])
        if key in entries:
            duplicate_count += 1
            continue
        entries[key] = pron
    stats["duplicates"] = int(duplicate_count)

    selected_lines: List[str] = []
    missing_tokens: List[str] = []
    for token in tokens:
        pron = entries.get(token)
        if not pron:
            missing_tokens.append(token)
            continue
        selected_lines.append(f"{token} {pron}")

    stats["selected"] = len(selected_lines)
    stats["missing"] = len(missing_tokens)
    if not selected_lines:
        return False, "Constrained dictionary became empty.", "", stats

    dict_dir = os.path.dirname(os.path.abspath(base_dict_path))
    dict_ext = os.path.splitext(base_dict_path)[1] or ".txt"
    constrained_path = os.path.join(dict_dir, f"dictionary_constrained{dict_ext}")
    try:
        with open(constrained_path, "w", encoding="utf-8", newline="\n") as f:
            for line in selected_lines:
                f.write(f"{line}\n")
    except Exception as exc:
        return False, f"Failed to write constrained dictionary: {exc}", "", stats

    log(
        f"[MFA][constrained] dictionary selected={stats['selected']}/{stats['token_total']} "
        f"missing={stats['missing']} duplicate_skipped={stats['duplicates']}"
    )

    if missing_tokens:
        preview = ", ".join(missing_tokens[:10])
        if mode_code == "strict":
            return (
                False,
                f"Strict constrained dictionary missing {len(missing_tokens)} tokens: {preview}",
                constrained_path,
                stats,
            )
        log(
            f"[MFA][constrained] moderate mode missing tokens={len(missing_tokens)} "
            f"(sample: {preview})"
        )

    return True, "", constrained_path, stats


def _collect_alignment_units(corpus_dir: str) -> List[Dict[str, str]]:
    units: List[Dict[str, str]] = []
    if not os.path.isdir(corpus_dir):
        return units
    wav_map: Dict[str, str] = {}
    for filename in os.listdir(corpus_dir):
        if not filename.lower().endswith(".wav"):
            continue
        path = os.path.join(corpus_dir, filename)
        if not os.path.isfile(path):
            continue
        stem = os.path.splitext(filename)[0]
        wav_map[stem] = path
    for stem in sorted(wav_map.keys()):
        lab_path = os.path.join(corpus_dir, f"{stem}.lab")
        txt_path = os.path.join(corpus_dir, f"{stem}.txt")
        label_path = ""
        if os.path.isfile(lab_path):
            label_path = lab_path
        elif os.path.isfile(txt_path):
            label_path = txt_path
        if not label_path:
            continue
        units.append(
            {
                "stem": stem,
                "wav": wav_map[stem],
                "label": label_path,
            }
        )
    return units


def _build_mfa_align_command(
    *,
    mfa_path: str,
    corpus_dir: str,
    dict_path: str,
    model_name: str,
    output_dir: str,
    single_speaker_flag: str,
    align_opts: Dict[str, object],
    profile_label: str,
    env=None,
    callback=None,
) -> Tuple[List[str], bool]:
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    cmd = [
        mfa_path,
        "align",
        corpus_dir,
        dict_path,
        model_name,
        output_dir,
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

    cmd.extend(
        [
            "--beam",
            str(int(align_opts.get("beam", 1000))),
            "--retry_beam",
            str(int(align_opts.get("retry_beam", 4000))),
            "--num_jobs",
            str(int(align_opts.get("num_jobs", 1))),
        ]
    )
    return cmd, speaker_adapt_enabled


def _run_mfa_align_command(
    *,
    mfa_path: str,
    corpus_dir: str,
    dict_path: str,
    model_name: str,
    output_dir: str,
    single_speaker_flag: str,
    align_opts: Dict[str, object],
    profile_label: str,
    env=None,
    callback=None,
    pass_tag: str = "",
) -> Tuple[bool, str, List[str]]:
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    cmd, speaker_adapt_enabled = _build_mfa_align_command(
        mfa_path=mfa_path,
        corpus_dir=corpus_dir,
        dict_path=dict_path,
        model_name=model_name,
        output_dir=output_dir,
        single_speaker_flag=single_speaker_flag,
        align_opts=align_opts,
        profile_label=profile_label,
        env=env,
        callback=callback,
    )
    tag = f" [{pass_tag}]" if pass_tag else ""
    log(
        "Starting MFA alignment"
        f"{tag}... ({single_speaker_flag}, profile={profile_label}, "
        f"speaker_adapt={'on' if speaker_adapt_enabled else 'off'}, "
        f"fine_tune={'on' if align_opts.get('fine_tune') else 'off'}, "
        f"beam={align_opts.get('beam')}, retry_beam={align_opts.get('retry_beam')}, "
        f"num_jobs={align_opts.get('num_jobs')})"
    )

    tail_lines: List[str] = []
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )
        if process.stdout:
            for raw_line in iter(process.stdout.readline, b""):
                stripped = _decode_subprocess_output(raw_line).strip()
                if not stripped:
                    continue
                log(stripped)
                tail_lines.append(stripped)
                if len(tail_lines) > 120:
                    tail_lines.pop(0)
        process.wait()
    except FileNotFoundError:
        return False, "MFA executable not found. Check MFA installation.", tail_lines
    except Exception as exc:
        return False, f"Unexpected MFA error: {exc}", tail_lines

    if process.returncode == 0:
        return True, "", tail_lines

    joined_tail = "\n".join(tail_lines[-40:]).lower()
    if (
        "please install korean support" in joined_tail
        or ("importerror" in joined_tail and "jamo" in joined_tail and "mecab" in joined_tail)
    ):
        return False, "Korean tokenizer dependencies (jamo + mecab backend) are missing in MFA env.", tail_lines

    err = f"MFA alignment failed (code: {process.returncode})"
    if tail_lines:
        err += f" | tail: {tail_lines[-1][:180]}"
    if process.returncode in {3221225477, -1073741819}:
        err += (
            " | hint: access_violation(0xC0000005), likely native kaldi crash "
            "(env binary mismatch or corpus/lexicon edge-case), retry with lower-load profile(fast/default)"
        )
    return False, err, tail_lines


def _run_recursive_mfa_align(
    *,
    mfa_path: str,
    corpus_dir: str,
    dict_path: str,
    model_name: str,
    output_dir: str,
    single_speaker_flag: str,
    align_opts: Dict[str, object],
    profile_label: str,
    env=None,
    callback=None,
    chunk_size: int = 96,
    max_depth: int = 8,
) -> Tuple[bool, str, int, List[str]]:
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    units = _collect_alignment_units(corpus_dir)
    if not units:
        return False, "No wav/lab pairs found for recursive MFA.", 0, []

    merged_count = 0

    def _split_and_align(sub_units: List[Dict[str, str]], depth: int) -> Tuple[bool, List[str]]:
        nonlocal merged_count
        if not sub_units:
            return True, []

        if len(sub_units) > int(chunk_size) and depth < int(max_depth):
            mid = max(1, len(sub_units) // 2)
            left_ok, left_fail = _split_and_align(sub_units[:mid], depth + 1)
            right_ok, right_fail = _split_and_align(sub_units[mid:], depth + 1)
            return left_ok and right_ok, left_fail + right_fail

        tmp_root = tempfile.mkdtemp(prefix="utoa_mfa_recursive_")
        try:
            sub_corpus = os.path.join(tmp_root, "corpus")
            sub_out = os.path.join(tmp_root, "out")
            os.makedirs(sub_corpus, exist_ok=True)
            os.makedirs(sub_out, exist_ok=True)
            for unit in sub_units:
                wav_dst = os.path.join(sub_corpus, os.path.basename(unit["wav"]))
                label_dst = os.path.join(sub_corpus, os.path.basename(unit["label"]))
                _link_or_copy(unit["wav"], wav_dst)
                _link_or_copy(unit["label"], label_dst)

            ok, _err, _tail = _run_mfa_align_command(
                mfa_path=mfa_path,
                corpus_dir=sub_corpus,
                dict_path=dict_path,
                model_name=model_name,
                output_dir=sub_out,
                single_speaker_flag=single_speaker_flag,
                align_opts=align_opts,
                profile_label=profile_label,
                env=env,
                callback=callback,
                pass_tag=f"recursive depth={depth} size={len(sub_units)}",
            )
            if ok:
                merged_count += _copy_back_textgrids(sub_out, output_dir)
                return True, []

            if len(sub_units) <= 1 or depth >= int(max_depth):
                failed = [str(unit.get("stem", "")) for unit in sub_units]
                return False, failed

            mid = max(1, len(sub_units) // 2)
            left_ok, left_fail = _split_and_align(sub_units[:mid], depth + 1)
            right_ok, right_fail = _split_and_align(sub_units[mid:], depth + 1)
            return left_ok and right_ok, left_fail + right_fail
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    log(
        f"[MFA][recursive] start chunk_size={chunk_size}, max_depth={max_depth}, "
        f"items={len(units)}"
    )
    ok, failed = _split_and_align(units, 0)
    if ok:
        log(f"[MFA][recursive] complete merged_textgrids={merged_count}")
        return True, "", merged_count, []
    preview = ", ".join(failed[:15])
    err = f"Recursive MFA failed for {len(failed)} item(s): {preview}"
    log(f"[MFA][recursive] {err}")
    return False, err, merged_count, failed


def run_mfa_align(
    mfa_path,
    wav_folder,
    dict_path,
    output_folder,
    language='korean',
    callback=None,
    align_profile='accurate',
    runtime_options: Optional[Dict[str, object]] = None,
):
    """Run MFA forced alignment."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False, 'MFA executable not found.'
    if not os.path.exists(wav_folder):
        return False, f'WAV folder not found: {wav_folder}'
    if not os.path.exists(dict_path):
        return False, f'Dictionary not found: {dict_path}'
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    log(f'Checking MFA prerequisites... ({lang_label})')
    strict_tokenizer_gate = str(os.environ.get("UTOA_STRICT_TOKENIZER_GATE", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            err = 'Missing Korean tokenizer dependencies (jamo + mecab backend).'
            if strict_tokenizer_gate:
                log(err)
                return False, err
            log(f"[MFA] {err} Proceeding because strict tokenizer gate is disabled.")
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            err = 'Japanese tokenizer readiness is uncertain or dependencies are missing.'
            if strict_tokenizer_gate:
                log(err)
                return False, err
            log(f"[MFA] {err} Proceeding because strict tokenizer gate is disabled.")
    os.makedirs(output_folder, exist_ok=True)
    env = _get_conda_env(mfa_path)
    model_reference = _resolve_acoustic_model_reference(mfa_path, model_name, env=env)
    if model_reference != model_name:
        log(f"[MFA] Using local acoustic model artifact: {model_reference}")
    ok, preflight_err = _preflight_compute_mfcc(mfa_path, callback=callback)
    if not ok:
        return False, preflight_err
    work_wav_folder = wav_folder
    work_dict_path = dict_path
    work_output_folder = output_folder
    safe_workspace = None
    needs_audio_prep = _mfa_audio_preprocess_required()
    needs_ascii_safe = any(_contains_non_ascii(p) for p in (wav_folder, dict_path, output_folder))
    if needs_ascii_safe or needs_audio_prep:
        try:
            safe_workspace = _prepare_ascii_safe_alignment_workspace(
                wav_folder,
                dict_path,
                output_folder,
                callback=callback,
            )
            work_wav_folder = safe_workspace["corpus_dir"]
            work_dict_path = safe_workspace["dict_path"]
            work_output_folder = safe_workspace["output_dir"]
            reason_tokens = []
            if needs_ascii_safe:
                reason_tokens.append("ascii-safe")
            if needs_audio_prep:
                reason_tokens.append("audio-prep")
            reason_text = "+".join(reason_tokens) if reason_tokens else "workspace"
            log(f"[MFA] Using temporary alignment workspace ({reason_text}): {safe_workspace['base']}")
            if safe_workspace.get("audio_preprocess_enabled"):
                log(
                    "[MFA][AudioPrep] applied="
                    f"{int(safe_workspace.get('audio_preprocess_applied', 0))}/"
                    f"{int(safe_workspace.get('audio_preprocess_total', 0))}, "
                    f"fallback_copy={int(safe_workspace.get('audio_preprocess_failed', 0))}"
                )
        except Exception as e:
            log(f"[MFA] Temporary workspace primary prepare failed: {e}")
            try:
                retry_root = tempfile.mkdtemp(prefix="utoa_mfa_retry_")
                safe_workspace = _prepare_ascii_safe_alignment_workspace(
                    wav_folder,
                    dict_path,
                    output_folder,
                    temp_root=retry_root,
                    callback=callback,
                )
                work_wav_folder = safe_workspace["corpus_dir"]
                work_dict_path = safe_workspace["dict_path"]
                work_output_folder = safe_workspace["output_dir"]
                log(f"[MFA] Temporary workspace retry succeeded: {safe_workspace['base']}")
                if safe_workspace.get("audio_preprocess_enabled"):
                    log(
                        "[MFA][AudioPrep] applied="
                        f"{int(safe_workspace.get('audio_preprocess_applied', 0))}/"
                        f"{int(safe_workspace.get('audio_preprocess_total', 0))}, "
                        f"fallback_copy={int(safe_workspace.get('audio_preprocess_failed', 0))}"
                    )
            except Exception as retry_exc:
                err = f"Failed to prepare temporary MFA workspace: {retry_exc}"
                log(err)
                return False, err
    dict_sanitize_ok, dict_sanitize_err = _sanitize_alignment_dictionary_for_mfa(work_dict_path, callback=callback)
    if not dict_sanitize_ok:
        return False, dict_sanitize_err
    strict_dict_gate = str(os.environ.get("UTOA_STRICT_DICT_VALIDATION", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dict_ok, dict_err = _validate_alignment_dictionary(
        work_dict_path,
        callback=callback,
        soft=(not strict_dict_gate),
    )
    if not dict_ok:
        return False, dict_err
    single_speaker_flag = _resolve_single_speaker_flag(mfa_path, env=env)
    resolved_profile, align_opts = _resolve_mfa_align_options(align_profile)
    runtime_ctx = _resolve_mfa_runtime_options(
        resolved_profile=resolved_profile,
        runtime_options=runtime_options,
        callback=callback,
    )
    constrained_mode = str(runtime_ctx.get("constrained_mode", "off"))
    recursive_enabled = bool(runtime_ctx.get("recursive_mfa", False))
    recursive_chunk_size = int(runtime_ctx.get("recursive_chunk_size", 96))
    recursive_max_depth = int(runtime_ctx.get("recursive_max_depth", 8))
    beam_scale = float(runtime_ctx.get("beam_scale", 1.0))
    if constrained_mode in {"strict", "moderate"} and beam_scale > 0:
        base_beam = int(align_opts.get("beam", 1000))
        base_retry = int(align_opts.get("retry_beam", 4000))
        align_opts["beam"] = max(120, int(round(base_beam * beam_scale)))
        align_opts["retry_beam"] = max(320, int(round(base_retry * beam_scale)))
        log(
            f"[MFA][constrained] beam {base_beam}->{align_opts['beam']}, "
            f"retry_beam {base_retry}->{align_opts['retry_beam']}"
        )

    attempt_dict_path = work_dict_path
    constrained_active = False
    if constrained_mode in {"strict", "moderate"}:
        c_ok, c_err, constrained_dict_path, _c_stats = _build_constrained_dictionary(
            work_dict_path,
            work_wav_folder,
            mode=constrained_mode,
            callback=callback,
        )
        if c_ok and constrained_dict_path:
            constrained_active = True
            attempt_dict_path = constrained_dict_path
            c_sanitize_ok, c_sanitize_err = _sanitize_alignment_dictionary_for_mfa(
                attempt_dict_path,
                callback=callback,
            )
            if not c_sanitize_ok:
                if constrained_mode == "strict":
                    return False, c_sanitize_err
                log(f"[MFA][constrained] sanitize failed, fallback to base dictionary: {c_sanitize_err}")
                constrained_active = False
                attempt_dict_path = work_dict_path
            c_dict_ok, c_dict_err = _validate_alignment_dictionary(
                attempt_dict_path,
                callback=callback,
                soft=(not strict_dict_gate),
            )
            if not c_dict_ok:
                if constrained_mode == "strict":
                    return False, c_dict_err
                log(f"[MFA][constrained] validation failed, fallback to base dictionary: {c_dict_err}")
                constrained_active = False
                attempt_dict_path = work_dict_path
        else:
            if constrained_mode == "strict":
                return False, c_err or "Strict constrained mode failed."
            log(f"[MFA][constrained] disabled in moderate mode: {c_err}")

    def _run_once_with_optional_recursive(current_dict_path: str, pass_tag: str, allow_recursive: bool):
        ok_run, err_run, _tail = _run_mfa_align_command(
            mfa_path=mfa_path,
            corpus_dir=work_wav_folder,
            dict_path=current_dict_path,
            model_name=model_reference,
            output_dir=work_output_folder,
            single_speaker_flag=single_speaker_flag,
            align_opts=align_opts,
            profile_label=resolved_profile,
            env=env,
            callback=callback,
            pass_tag=pass_tag,
        )
        if ok_run:
            return True, ""
        if not allow_recursive:
            return False, err_run
        log("[MFA][recursive] primary alignment failed; starting segmented fallback.")
        rec_ok, rec_err, _merged_count, _failed = _run_recursive_mfa_align(
            mfa_path=mfa_path,
            corpus_dir=work_wav_folder,
            dict_path=current_dict_path,
            model_name=model_reference,
            output_dir=work_output_folder,
            single_speaker_flag=single_speaker_flag,
            align_opts=align_opts,
            profile_label=resolved_profile,
            env=env,
            callback=callback,
            chunk_size=recursive_chunk_size,
            max_depth=recursive_max_depth,
        )
        if rec_ok:
            return True, ""
        return False, f"{err_run} | recursive: {rec_err}"

    ok, err = _run_once_with_optional_recursive(
        attempt_dict_path,
        "constrained" if constrained_active else "primary",
        recursive_enabled,
    )
    if (not ok) and constrained_active and constrained_mode == "moderate":
        log("[MFA][constrained] moderate fallback: retry with base dictionary.")
        ok, err = _run_once_with_optional_recursive(
            work_dict_path,
            "moderate-fallback",
            False,
        )

    if ok:
        if safe_workspace is not None:
            copied = _copy_back_textgrids(work_output_folder, output_folder)
            log(f"[MFA] Copied back {copied} TextGrid files from ASCII-safe workspace.")
        log('MFA alignment completed successfully.')
        return True, ''

    log(str(err or "MFA alignment failed"))
    return False, str(err or "MFA alignment failed")

def patch_mfa_korean_support(mfa_path, callback=None):
    """
    Patch MFA Korean tokenization modules for mecab-only backend usage.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if sys.platform != "win32" or not mfa_path or "Scripts" not in mfa_path:
        return True

    try:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        tokenization_dir = os.path.join(
            env_dir,
            "Lib",
            "site-packages",
            "montreal_forced_aligner",
            "tokenization",
        )
        spacy_py = os.path.join(tokenization_dir, "spacy.py")
        korean_py = os.path.join(tokenization_dir, "korean.py")

        def _ensure_writable_copy(path: str) -> None:
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

        def _read_file(path: str) -> Optional[str]:
            if not os.path.exists(path):
                return None
            try:
                if os.path.getsize(path) == 0:
                    return None
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None

        def _write_if_changed(path: str, old: str, new: str, label: str) -> None:
            if new == old:
                return
            _ensure_writable_copy(path)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)
            log(f"[MFA] Patched {label} for mecab compatibility.")

        spacy_content = _read_file(spacy_py)
        if spacy_content:
            spacy_new = spacy_content
            spacy_new = spacy_new.replace(
                "pip install python-mecab-ko jamo",
                "pip install python-mecab-ko jamo mecab-python3",
            )
            _write_if_changed(spacy_py, spacy_content, spacy_new, "spacy.py")

        korean_content = _read_file(korean_py)
        if korean_content:
            korean_new = korean_content

            compat_block = """
# UTOA_KOREAN_IMPORT_COMPAT
try:
    import jamo
    _utoa_has_backend = False

    try:
        from mecab import MeCab as _NativeMeCab
        MeCab = _NativeMeCab
        _utoa_has_backend = True
    except Exception:
        try:
            from mecab import Tagger as _MecabTagger

            class _UtoaMecabNode:
                def __init__(self, surface, pos):
                    self.surface = surface
                    self.pos = pos

            class MeCab:
                def __init__(self):
                    self._tagger = _MecabTagger()

                def parse(self, text):
                    node = self._tagger.parseToNode(text)
                    items = []
                    while node is not None:
                        surface = getattr(node, "surface", "") or ""
                        if surface:
                            feature = getattr(node, "feature", "") or ""
                            if not isinstance(feature, str):
                                feature = str(feature)
                            pos = feature.split(",", 1)[0] if feature else ""
                            items.append(_UtoaMecabNode(surface, pos))
                        node = getattr(node, "next", None)
                    return items

            _utoa_has_backend = True
        except Exception:
            try:
                import MeCab as _MeCabModule

                class _UtoaMecabNode:
                    def __init__(self, surface, pos):
                        self.surface = surface
                        self.pos = pos

                class MeCab:
                    def __init__(self):
                        self._tagger = _MeCabModule.Tagger()

                    def parse(self, text):
                        node = self._tagger.parseToNode(text)
                        items = []
                        while node is not None:
                            surface = getattr(node, "surface", "") or ""
                            if surface:
                                feature = getattr(node, "feature", "") or ""
                                if not isinstance(feature, str):
                                    feature = str(feature)
                                pos = feature.split(",", 1)[0] if feature else ""
                                items.append(_UtoaMecabNode(surface, pos))
                            node = getattr(node, "next", None)
                        return items

                _utoa_has_backend = True
            except Exception:
                MeCab = None

    KO_AVAILABLE = _utoa_has_backend
except (ImportError, ModuleNotFoundError):
    KO_AVAILABLE = False
    MeCab = None
    jamo = None
"""
            if "UTOA_KOREAN_IMPORT_COMPAT" not in korean_new:
                korean_new = re.sub(
                    r"try:\s*[\s\S]*?KO_AVAILABLE\s*=\s*False\s*[\s\S]*?jamo\s*=\s*None\s*",
                    compat_block + "\n",
                    korean_new,
                    count=1,
                )
            korean_new = re.sub(
                r"self\.tokenizer\s*=\s*\w+Wrapper\(\)",
                "self.tokenizer = MeCab()",
                korean_new,
            )

            _write_if_changed(korean_py, korean_content, korean_new, "korean.py")

        return True
    except Exception as e:
        log(f"[MFA] Korean patch error: {e}")
        return False
