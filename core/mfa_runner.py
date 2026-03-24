"""
MFA (Montreal Forced Aligner) ・､嵂・・ｨ・・
- ・懍ｻｬ ・尖株 尞ｬ奓ｰ・・Conda 嶹俾ｲｽ・川・ MFA ・､嵂・
- ・､・懋ｰ・・懋ｷｸ ・､孖ｸ・ｬ・・
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
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


def _run_subprocess_text(args: Sequence[str], **kwargs):
    completed = subprocess.run(args, capture_output=True, text=False, **kwargs)
    completed.stdout = _decode_subprocess_output(getattr(completed, "stdout", b""))
    completed.stderr = _decode_subprocess_output(getattr(completed, "stderr", b""))
    return completed


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
        _add(app_dir)
        if os.path.basename(app_dir).lower() in {"utau_auto_oto", "auto_oto"}:
            _add(os.path.dirname(app_dir))
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
    env_candidates = [
        os.path.join(app_dir, '.env'),
        os.path.join(app_dir, 'env'),
        os.path.join(parent_dir, '.env'),
        os.path.join(parent_dir, 'env'),
        shared_env_dir,
    ]
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


def _prepare_ascii_safe_alignment_workspace(wav_folder, dict_path, output_folder):
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

    for fn in os.listdir(wav_folder):
        low = fn.lower()
        if low.endswith(".wav") or low.endswith(".lab") or low.endswith(".txt"):
            src = os.path.join(wav_folder, fn)
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
    log_fn("笞 Microsoft Visual C++ 14.0+ (C++ Build Tools)・ 﨑・囈﨑ｩ・壱共.")
    log_fn("   ・､・・・・〓: https://visualstudio.microsoft.com/visual-cpp-build-tools/")


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
    """MFA ・簿ｬ ・懍梠 ・・乱 compute-mfcc-feats ・､嵂・・・･ ・ｬ・・ｼ ・専ｲ﨑ｩ・壱共."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        return False, "MFA ・､嵂・甯護攵 ・ｽ・懋ｰ ・・牟 ・溢慣・壱共."

    env = _get_conda_env(mfa_path)
    candidates = []
    if sys.platform == 'win32' and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        candidates.append(os.path.join(env_dir, 'Library', 'bin', 'compute-mfcc-feats.exe'))
        candidates.append('compute-mfcc-feats.exe')
    candidates.append('compute-mfcc-feats')

    last_not_found = None
    for candidate in candidates:
        try:
            # Windows + Python 3.13 ・ｰ﨑ｩ・川・・・嶹菩棗・・・・株 ・､嵂雅ｪ・・・餓擽 ・､甯ｨ﨑 ・・・壱共.
            subprocess.run(
                [candidate, '--help'],
                capture_output=True,
                text=False,
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
                "compute-mfcc-feats ・､嵂・・醐復・ｴ ・・牟 MFA ・簿ｬ・・・懍梠﨑 ・・・・慣・壱共. "
                "(WinError 5: Access denied)"
            )
            log(f"笶・{err}")
            log("   ・ｴ・・嵓・｡懋ｷｸ・ｨ/・醐復 ・菩ｱ・甯護攵 ・ｨ・ｨ ・ｬ・・ｼ 嶹菩攤﨑ｴ ・ｼ・ｸ・・")
            return False, f"{err}: {e}"
        except Exception as e:
            err = f"compute-mfcc-feats ・ｬ・・・専ｲ ・・・､・・ {e}"
            log(f"笶・{err}")
            return False, err

    err = "compute-mfcc-feats・ｼ ・ｾ・ ・ｻ嵂溢慣・壱共. MFA 嶹俾ｲｽ・ｴ ・川メ・們来・・・・・溢慣・壱共."
    log(f"笶・{err}")
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
    Windows 嶹俾ｲｽ・川・ Conda 嶹懍┳嶹・・・擽 mfa.exe・ｼ ・・・嶸ｸ・懦腹 ・・
    DLL ・罹糖 ・尖洳(・罷糖 3228369023)・ ・懍・﨑俯株 ・・揆 ・賀ｸｰ ・・紛 嶹俾ｲｽ ・・們乱 PATH・ｼ ・ｼ・・鮒・壱共.
    """
    env = os.environ.copy()
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
        "    ok=True\n"
        "except Exception:\n"
        "    try:\n"
        "        import MeCab\n"
        "        ok=True\n"
        "    except Exception:\n"
        "        try:\n"
        "            import mecab_ko\n"
        "            ok=True\n"
        "        except Exception:\n"
        "            ok=False\n"
        "sys.exit(0 if ok else 1)\n"
    )


def ensure_mfa_python_packaging_stack(mfa_path, callback=None):
    """
    Ensure pip/setuptools/pkg_resources/wheel are available inside the MFA env.
    This is a prerequisite for language-specific dependency repair on native Windows.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or 'Scripts' not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
    if not os.path.exists(python_exe):
        return False

    env = _get_conda_env(mfa_path)
    ok, _detail = _check_env_imports(python_exe, env, 'import pip; import pkg_resources; import wheel')
    if ok:
        return True

    log('[MFA] Python 甯ｨ墲､・ ・・ｵｬ(pip/setuptools/wheel) ・ｵ・ｬ ・・..')
    repair_cmds = [
        [python_exe, '-m', 'ensurepip', '--upgrade'],
        [python_exe, '-m', 'pip', 'install', '--upgrade', 'setuptools<81', 'wheel'],
    ]
    if os.path.exists(pip_exe):
        repair_cmds.append([pip_exe, 'install', '--upgrade', 'setuptools<81', 'wheel'])
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

    for repair_cmd in repair_cmds:
        log(f"   -> repair cmd: {' '.join(repair_cmd)}")
        result = _run_subprocess_text(repair_cmd, env=env)
        if result.returncode != 0:
            continue
        ok, _detail = _check_env_imports(python_exe, env, 'import pip; import pkg_resources; import wheel')
        if ok:
            return True

    log('[MFA] Failed to restore pip/setuptools/pkg_resources/wheel')
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
        log("[MFA] ・・卿: MFA ・､嵂・甯護攵・・・ｾ・ ・ｻ嵂溢慣・壱共.")
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
            ok, _detail = _check_env_imports(python_exe, env, 'import pip; import pkg_resources; import wheel')
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
    MFA ・・・乱 ・ｰ・ｼ ・ｨ・ｼ 嶹肥梵 ・ｵ・・岺懋ｸｰ・ ・､・ｼ ・・・溢牟(--single-speaker / --single_speaker)
    help ・罹･・・・ｴ・ ・・尖据・・岺懋ｸｰ・ｼ ・夋晨鮒・壱共.
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

    # ・ｰ・ｸ・廷捩 ・肥ｲｭ・・・樌ｶｰ 﨑們擽嵓・岺懋ｸｰ ・ｰ・
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
    ・懍侃奛懍乱 ・､・俯頗 MFA ・､嵂・甯護攵・・夋川ラ﨑ｩ・壱共.
    尞ｬ奓ｰ・・嶹俾ｲｽ -> Conda 嶹俾ｲｽ -> ・懍侃奛・PATH ・懍・・・・・駕鮒・壱共.
    
    Returns:
        MFA ・､嵂・甯護攵 ・ｽ・・・尖株 None
    """
    # 1. ・ｵ・/・一ｱｰ・・尞ｬ奓ｰ・・嶹俾ｲｽ
    for p in _candidate_mfa_executable_paths():
        if os.path.exists(p):
            logger.info(f"尞ｬ奓ｰ・・MFA ・懋ｲｬ: {p}")
            return p

    # 2. ・懍侃奛・PATH
    mfa_path = shutil.which('mfa')
    if mfa_path:
        logger.info(f"・懍侃奛・MFA ・懋ｲｬ: {mfa_path}")
        return mfa_path

    # 3. Conda 嶹俾ｲｽ ・ｰ・ｸ ・ｽ・・
    conda_paths = [
        os.path.expanduser('~/miniconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/anaconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/miniconda3/Scripts/mfa.exe'),
    ]
    for p in conda_paths:
        if os.path.exists(p):
            logger.info(f"Conda MFA ・懋ｲｬ: {p}")
            return p

    return None


def check_mfa_model(mfa_path, language='korean'):
    """
    MFA ・醐箕 ・ｨ・ｸ・ｴ ・､・ｴ・罹糖・們牟 ・壱株・ 嶹菩攤﨑ｩ・壱共.
    
    Args:
        mfa_path: MFA ・､嵂・甯護攵 ・ｽ・・
        language: 'korean' ・尖株 'japanese'
    
    Returns:
        (・､・・・ｬ・: bool, ・肥亨・: str)
    """
    if not mfa_path:
        return False, "MFA ・､嵂・甯護攵・・・ｾ・・・・・・慣・壱共."

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '・ｼ・ｸ・ｴ' if language == 'japanese' else '﨑懋ｵｭ・ｴ'

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
            return True, f"{lang_label} MFA ・ｨ・ｸ・ｴ ・､・俯据・ｴ ・溢慣・壱共."
        if _has_local_acoustic_model_artifact(mfa_path, model_name, env=env):
            return True, f"{lang_label} MFA ・ｨ・ｸ ・懍ｻｬ ・・恐甯ｩ孖ｸ・ｼ 嶹菩攤嵂溢慣・壱共."
        if result.returncode != 0:
            return False, (
                f"{lang_label} MFA ・ｨ・ｸ 嶹菩攤 ・・ｹ・ｴ ・､甯ｨ嵂溢慣・壱共(code={result.returncode}). "
                "・ｨ・ｸ ・､・ｴ・罹糖・ 﨑・囈﨑ｩ・壱共."
            )
        return False, f"{lang_label} MFA ・ｨ・ｸ・ｴ ・､・俯据・ｴ ・溢ｧ ・喜慣・壱共. ・､・ｴ・罹糖・ 﨑・囈﨑ｩ・壱共."
    except Exception as e:
        return False, f"MFA ・ｨ・ｸ 嶹菩攤 ・､甯ｨ: {e}"


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

    # MFA ・ｰ・ｸ ・ｨ孖ｸ(・一ｱｰ・・・・嶹菩攤
    _add(os.path.expanduser("~/Documents/MFA"))
    return roots


def _has_local_acoustic_model_artifact(mfa_path: str, model_name: str, env: Optional[dict] = None) -> bool:
    if not model_name:
        return False
    for root in _candidate_mfa_root_dirs(mfa_path, env=env):
        acoustic_dir = os.path.join(root, "pretrained_models", "acoustic")
        candidates = [
            os.path.join(acoustic_dir, model_name),
            os.path.join(acoustic_dir, f"{model_name}.zip"),
            os.path.join(acoustic_dir, f"{model_name}.yaml"),
            os.path.join(acoustic_dir, f"{model_name}.yml"),
            os.path.join(acoustic_dir, f"{model_name}.meta"),
        ]
        if any(os.path.exists(path) for path in candidates):
            return True
    return False


def check_mfa_ready(language='korean', mfa_path=''):
    resolved_mfa = mfa_path or find_mfa_executable()
    if not resolved_mfa or not os.path.exists(resolved_mfa):
        return make_runtime_report(
            "align",
            ALIGN_EXEC_MISSING,
            "MFA ・､嵂・甯護攵・・・ｾ・・・・・・慣・壱共.",
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
            msg or "MFA ・ｨ・ｸ・・・ｾ・・・・・・慣・壱共.",
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
    if not os.path.exists(python_exe):
        return False
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log('[MFA] Failed to prepare base Python packaging tools before Korean dependency install')
        return False
    pkg_check_cmd = [python_exe, '-c', 'import pkg_resources']
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
                if not os.path.isdir(mecab_dir):
                    os.makedirs(mecab_dir, exist_ok=True)
                    content = ""
                else:
                    with open(mecab_init, 'r', encoding='utf-8') as f:
                        content = f.read()
                if 'UTOA_MECAB_SHIM' in content:
                    return
                shim = """

# UTOA_MECAB_SHIM
import MeCab as _MeCabMod
class _UtoaMecabNode:
    def __init__(self, surface, pos):
        self.surface = surface
        self.pos = pos

class MeCab:
    def __init__(self):
        self._tagger = _MeCabMod.Tagger()

    def parse(self, text):
        node = self._tagger.parseToNode(text)
        items = []
        while node is not None:
            surface = getattr(node, "surface", "") or ""
            if surface:
                feature = getattr(node, "feature", "") or ""
                pos = feature.split(",", 1)[0] if feature else ""
                items.append(_UtoaMecabNode(surface, pos))
            node = getattr(node, "next", None)
        return items
"""
                with open(mecab_init, 'w', encoding='utf-8') as f:
                    f.write(content + shim)
            except Exception as e:
                log(f"[MFA] Failed to patch mecab shim: {e}")

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

        def _ensure_pkg_resources():
            res = _run_subprocess_text(pkg_check_cmd, env=env)
            if res.returncode == 0:
                return True
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
                result = _run_subprocess_text(install_cmd, env=env)
                if result.returncode != 0:
                    continue
                verify = _run_subprocess_text(pkg_check_cmd, env=env)
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
        def _run_install_stage(packages, *, pip_extra_args=None):
            extra_args = list(pip_extra_args or [])
            stage_cmds = [[python_exe, '-m', 'pip', 'install', '--upgrade', *extra_args, *packages]]
            if os.path.exists(pip_exe):
                stage_cmds.append([pip_exe, 'install', '--upgrade', *extra_args, *packages])
            system_conda = shutil.which('conda')
            if system_conda:
                stage_cmds.append([
                    system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip', 'install',
                    '--upgrade', *extra_args, *packages
                ])
            stage_last_err = ''
            for install_cmd in stage_cmds:
                log(f"   -> cmd: {' '.join(install_cmd)}")
                result = _run_subprocess_text(install_cmd, env=env)
                if result.returncode != 0:
                    err_txt = (result.stderr or result.stdout or '').strip()
                    if err_txt:
                        log(f"   [warn] install failed: {err_txt[:500]}")
                    stage_last_err = err_txt or stage_last_err
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
        ok, stage_err = _run_install_stage(
            ['python-mecab-ko', 'python-mecab-ko-dic'],
            pip_extra_args=['--only-binary=:all:'],
        )
        if stage_err:
            last_err = stage_err
        if ok:
            log('[MFA] Korean tokenizer deps are ready')
            patch_mfa_korean_support(mfa_path, callback)
            return True

        # Stage 3: fallback to mecab-python3 wheels only.
        log('[MFA] Installing Korean tokenizer deps fallback: mecab-python3')
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
    MFA ・ｼ・ｸ・ｴ ・簿ｬ・・﨑・囈﨑・spacy/sudachipy/sudachidict-core・ ・壱株・ 嶹菩攤﨑俾ｳ,
    ・・攷 ・・・尖徐 ・､・俯･ｼ ・罹巡﨑ｩ・壱共.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or 'Scripts' not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = _resolve_env_python_exe(env_dir)
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')

    if not os.path.exists(python_exe):
        return True
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log("笞・・MFA Python 甯ｨ墲､・ ・・ｵｬ ・ｵ・ｬ・・・､甯ｨ﨑ｴ ・ｼ・ｸ・ｴ ・們｡ｴ・ｱ ・､・俯･ｼ ・・・﨑 ・・・・慣・壱共.")
        return False

    check_cmd = [python_exe, '-c', 'import spacy; import sudachipy; import sudachidict_core']
    try:
        env = _get_conda_env(mfa_path)
        result = _run_subprocess_text(check_cmd, env=env)
        if result.returncode == 0:
            return True

        log("逃 MFA ・ｼ・ｸ・ｴ 奝增ｬ・們擽・ ・們｡ｴ・ｱ(spacy, sudachipy, sudachidict-core) ・､・・嶹菩攤 ・・..")

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
            log("笞・・・ｼ・ｸ・ｴ ・們｡ｴ・ｱ ・尖徐 ・､・・・ｽ・罹･ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
            return False

        log(f"   -> ・､嵂・・・ｹ・ｴ: {' '.join(install_cmd)}")
        install_result = _run_subprocess_text(install_cmd, env=env)
        if install_result.returncode != 0:
            if install_result.stderr:
                log(f"   笞・・・､・・stderr: {install_result.stderr[:500]}")
            if install_result.stdout:
                log(f"   笞・・・､・・stdout: {install_result.stdout[:500]}")
            if os.path.exists(pip_exe):
                pip_cmd = [pip_exe, 'install', 'spacy', 'sudachipy', 'sudachidict-core']
                log(f"   -> ・・ｴ ・､・・・・ｹ・ｴ(pip): {' '.join(pip_cmd)}")
                pip_result = _run_subprocess_text(pip_cmd, env=env)
                if pip_result.returncode != 0:
                    if pip_result.stderr:
                        log(f"   笞・・pip stderr: {pip_result.stderr[:500]}")
                    if pip_result.stdout:
                        log(f"   笞・・pip stdout: {pip_result.stdout[:500]}")
                    return False
            else:
                return False

        verify = _run_subprocess_text(check_cmd, env=env)
        if verify.returncode == 0:
            log("[MFA] Japanese tokenizer dependencies are ready.")
            return True

        log("笞・・・ｼ・ｸ・ｴ ・們｡ｴ・ｱ ・､・・弡・乱・・import ・・晧乱 ・､甯ｨ嵂溢慣・壱共.")
        if verify.stderr:
            log(f"   ・・┷ stderr: {verify.stderr[:500]}")
        return False
    except Exception as e:
        log(f"笞・・・ｼ・ｸ・ｴ ・們｡ｴ・ｱ ・尖徐 嶹菩攤/・､・・・・・､・・・懍・: {e}")
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
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'

    has_model, msg = check_mfa_model(mfa_path, language=language)
    if has_model:
        if msg:
            log(msg)
        return True

    # ・ｨ・ｸ ・､・ｴ・罹糖・・・ｸ・ｴ ・們｡ｴ・ｱ ・・・凰 ・・ｦｬ﨑ｴ ・俯ｦｬ﨑ｩ・壱共.
    # (・們｡ｴ・ｱ import ・､・俾ｰ ・溢牟・・・ｨ・ｸ ・川ｲｴ ・､・ｴ・罹糖・・・・･﨑・・ｽ・ｰ・ ・溢搆)
    try:
        if language == 'korean':
            if not ensure_korean_support(mfa_path, callback):
                log('笞 Korean dependency prepare failed, but model download will continue.')
        elif language == 'japanese':
            if not ensure_japanese_support(mfa_path, callback):
                log('笞 Japanese dependency prepare failed, but model download will continue.')
    except Exception as dep_exc:
        log(f'笞 Dependency prepare check raised error, but model download will continue: {dep_exc}')

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
                        log(line)
            process.wait()
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
        "・・・淀・ｩ",
        "・・Ю・・ｲｩ",
    }:
        return "strict"
    if compact in {
        "moderate",
        "medium",
        "balanced",
        "soft_strict",
        "fallback",
        "・・胸德溢淀・ｩ",
        "・・胸德・・・ｲｩ",
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
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            err = 'Missing Korean tokenizer dependencies (jamo + mecab backend).'
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
    if any(_contains_non_ascii(p) for p in (wav_folder, dict_path, output_folder)):
        try:
            safe_workspace = _prepare_ascii_safe_alignment_workspace(wav_folder, dict_path, output_folder)
            work_wav_folder = safe_workspace["corpus_dir"]
            work_dict_path = safe_workspace["dict_path"]
            work_output_folder = safe_workspace["output_dir"]
            log(f"[MFA] Non-ASCII path detected, using ASCII-safe workspace: {safe_workspace['base']}")
        except Exception as e:
            err = f"Failed to prepare ASCII-safe MFA workspace: {e}"
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
            c_dict_ok, c_dict_err = _validate_alignment_dictionary(attempt_dict_path, callback=callback)
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
            model_name=model_name,
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
            model_name=model_name,
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
            )
            _write_if_changed(spacy_py, spacy_content, spacy_new, "spacy.py")

        korean_content = _read_file(korean_py)
        if korean_content:
            korean_new = korean_content

            # Normalize old try/except import blocks to a single mecab import.
            korean_new = re.sub(
                r"try:\s*from\s+\w+\s+import\s+\w+\s+as\s+MeCab\s*except\s+Exception:\s*from\s+\w+\s+import\s+\w+\s+as\s+MeCab",
                "from mecab import MeCab",
                korean_new,
                count=1,
                flags=re.MULTILINE,
            )
            korean_new = re.sub(
                r"from\s+\w+\s+import\s+\w+\s+as\s+MeCab",
                "from mecab import MeCab",
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
