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
_MFA_SINGLE_SPEAKER_FLAG_CACHE = {}
_MFA_SPEAKER_ADAPT_FLAG_CACHE = {}

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


def _default_mfa_root_dir(mfa_path=""):
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    elif mfa_path:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(mfa_path)))
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(app_dir, ".mfa_root_ascii")
    os.makedirs(root, exist_ok=True)
    return root


def get_default_mfa_env_dir():
    public_root = os.environ.get("PUBLIC", r"C:\Users\Public")
    return os.path.join(public_root, "UTAU_Auto_OTO_v3", ".env")


def get_default_mfa_conda_root():
    public_root = os.environ.get("PUBLIC", r"C:\Users\Public")
    return os.path.join(public_root, "UTAU_Auto_OTO_v3", "miniconda")


def get_default_mfa_micromamba_root():
    public_root = os.environ.get("PUBLIC", r"C:\Users\Public")
    return os.path.join(public_root, "UTAU_Auto_OTO_v3", "micromamba")


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
    shared_env_dir = get_default_mfa_env_dir()
    candidates = [
        os.path.join(shared_env_dir, 'Scripts', 'mfa.exe'),
        os.path.join(shared_env_dir, 'Scripts', 'mfa.bat'),
        os.path.join(shared_env_dir, 'Scripts', 'mfa.cmd'),
        os.path.join(shared_env_dir, 'bin', 'mfa'),
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.exe'),
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.bat'),
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.cmd'),
        os.path.join(app_dir, '.env', 'bin', 'mfa'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.exe'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.bat'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.cmd'),
    ]
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
    log_fn("⚠ Microsoft Visual C++ 14.0+ (C++ Build Tools)가 필요합니다.")
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
    candidates = [
        os.path.join(env_dir, "python.exe"),
        os.path.join(env_dir, "bin", "python"),
    ]
    python_exe = next((p for p in candidates if os.path.exists(p)), "")
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
    if sys.platform == 'win32' and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        candidates.append(os.path.join(env_dir, 'Library', 'bin', 'compute-mfcc-feats.exe'))
        candidates.append('compute-mfcc-feats.exe')
    candidates.append('compute-mfcc-feats')

    last_not_found = None
    for candidate in candidates:
        try:
            # Windows + Python 3.13 조합에서는 확장자 없는 실행명 검색이 실패할 수 있다.
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

def _get_conda_env(mfa_path):
    """
    Windows 환경에서 Conda 활성화 없이 mfa.exe를 직접 호출할 때 
    DLL 로드 에러(코드 3228369023)가 발생하는 것을 막기 위해 환경 변수에 PATH를 주입합니다.
    """
    env = os.environ.copy()
    if sys.platform == 'win32' and mfa_path and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        new_paths = [
            env_dir,
            os.path.join(env_dir, 'Library', 'mingw-w64', 'bin'),
            os.path.join(env_dir, 'Library', 'usr', 'bin'),
            os.path.join(env_dir, 'Library', 'bin'),
            os.path.join(env_dir, 'Scripts'),
            os.path.join(env_dir, 'bin'),
        ]
        current_path = env.get('PATH', '')
        env['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path
        env['CONDA_PREFIX'] = env_dir
    env.setdefault('MFA_ROOT_DIR', _default_mfa_root_dir(mfa_path))
    return env


def _check_env_imports(python_exe: str, env: dict, import_expr: str):
    res = _run_subprocess_text([python_exe, '-c', import_expr], env=env)
    if res.returncode == 0:
        return True, ''
    detail = (res.stderr or res.stdout or '').strip()
    return False, detail


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
    python_exe = os.path.join(env_dir, 'python.exe')
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
    if not os.path.exists(python_exe):
        return False

    env = _get_conda_env(mfa_path)
    ok, _detail = _check_env_imports(python_exe, env, 'import pip; import pkg_resources; import wheel')
    if ok:
        return True

    log('[MFA] Python 패키지 도구(pip/setuptools/wheel) 복구 중...')
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
        log("[MFA] 진단: MFA 실행 파일을 찾지 못했습니다.")
        return report

    report["checks"]["mfa_executable"] = True
    if 'Scripts' in resolved:
        env_dir = os.path.dirname(os.path.dirname(resolved))
        python_exe = os.path.join(env_dir, 'python.exe')
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
                'import eunjeon; import jamo'
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
    MFA 버전에 따라 단일 화자 옵션 표기가 다를 수 있어(--single-speaker / --single_speaker)
    help 출력을 보고 지원되는 표기를 선택합니다.
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
    시스템에 설치된 MFA 실행 파일을 탐색합니다.
    포터블 환경 -> Conda 환경 -> 시스템 PATH 순서로 검색합니다.
    
    Returns:
        MFA 실행 파일 경로 또는 None
    """
    # 1. 공유/레거시 포터블 환경
    for p in _candidate_mfa_executable_paths():
        if os.path.exists(p):
            logger.info(f"포터블 MFA 발견: {p}")
            return p

    # 2. 시스템 PATH
    mfa_path = shutil.which('mfa')
    if mfa_path:
        logger.info(f"시스템 MFA 발견: {mfa_path}")
        return mfa_path

    # 3. Conda 환경 기본 경로
    conda_paths = [
        os.path.expanduser('~/miniconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/anaconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/miniconda3/Scripts/mfa.exe'),
    ]
    for p in conda_paths:
        if os.path.exists(p):
            logger.info(f"Conda MFA 발견: {p}")
            return p

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

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '일본어' if language == 'japanese' else '한국어'

    try:
        env = _get_conda_env(mfa_path)
        result = subprocess.run(
            [mfa_path, 'model', 'list', 'acoustic'],
            capture_output=True, text=False, timeout=30, env=env
        )
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
    - eunjeon
    - jamo
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
    python_exe = os.path.join(env_dir, 'python.exe')
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    if not os.path.exists(python_exe):
        return False
    if not ensure_mfa_python_packaging_stack(mfa_path, callback=callback):
        log('[MFA] Failed to prepare base Python packaging tools before Korean dependency install')
        return False
    pkg_check_cmd = [python_exe, '-c', 'import pkg_resources']
    check_cmd = [python_exe, '-c', 'import eunjeon; import jamo']
    try:
        env = _get_conda_env(mfa_path)

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

        ok, detail = _check_imports()
        if (not ok) and _looks_like_pyexpat_dll_issue(detail):
            log('[MFA] Detected pyexpat/libexpat DLL issue; trying repair...')
            if _try_repair_pyexpat():
                ok, detail = _check_imports()
        if ok:
            patch_mfa_korean_support(mfa_path, callback)
            return True
        log('[MFA] Installing Korean tokenizer deps: eunjeon, jamo')
        install_cmds = [
            [python_exe, '-m', 'pip', 'install', '--upgrade', 'eunjeon', 'jamo'],
        ]
        if os.path.exists(pip_exe):
            install_cmds.append([pip_exe, 'install', '--upgrade', 'eunjeon', 'jamo'])
        system_conda = shutil.which('conda')
        if system_conda:
            install_cmds.append([
                system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip', 'install',
                '--upgrade', 'eunjeon', 'jamo'
            ])
        last_err = detail
        for install_cmd in install_cmds:
            log(f"   -> cmd: {' '.join(install_cmd)}")
            result = _run_subprocess_text(install_cmd, env=env)
            if result.returncode != 0:
                err_txt = (result.stderr or result.stdout or '').strip()
                if err_txt:
                    log(f"   [warn] install failed: {err_txt[:500]}")
                if _stderr_has_msvc_requirement(result.stderr):
                    _emit_msvc_required_notice(callback, log)
                last_err = err_txt or last_err
                continue
            if not _ensure_pkg_resources():
                last_err = 'pkg_resources/setuptools repair failed after install'
                continue
            ok, detail = _check_imports()
            if (not ok) and _looks_like_pyexpat_dll_issue(detail):
                log('[MFA] Detected pyexpat/libexpat DLL issue after install; trying repair...')
                if _try_repair_pyexpat():
                    ok, detail = _check_imports()
            if ok:
                log('[MFA] Korean tokenizer deps are ready')
                patch_mfa_korean_support(mfa_path, callback)
                return True
            if detail:
                log(f"   [warn] import check failed after install: {detail[:500]}")
                last_err = detail
        log('[MFA] Failed to prepare Korean tokenizer deps (eunjeon, jamo)')
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

    if not mfa_path or 'Scripts' not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = os.path.join(env_dir, 'python.exe')
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
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            log('Failed to prepare Korean dependencies (eunjeon, jamo).')
            return False
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            log('Failed to prepare Japanese dependencies (spacy, sudachipy, sudachidict-core).')
            return False
    log(f'Downloading {lang_label} MFA model...')
    try:
        env = _get_conda_env(mfa_path)
        process = subprocess.Popen(
            [mfa_path, 'model', 'download', 'acoustic', model_name, '--ignore_cache'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )
        if process.stdout:
            for raw_line in iter(process.stdout.readline, b""):
                line = _decode_subprocess_output(raw_line).strip()
                if line:
                    log(line)
        process.wait()
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
    if not os.path.exists(wav_folder):
        return False, f'WAV folder not found: {wav_folder}'
    if not os.path.exists(dict_path):
        return False, f'Dictionary not found: {dict_path}'
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    log(f'Checking MFA prerequisites... ({lang_label})')
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            err = 'Missing Korean tokenizer dependencies (eunjeon, jamo).'
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
        f"num_jobs={align_opts.get('num_jobs')})"
    )
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )
        tail_lines = []
        if process.stdout:
            for raw_line in iter(process.stdout.readline, b""):
                stripped = _decode_subprocess_output(raw_line).strip()
                if stripped:
                    log(stripped)
                    tail_lines.append(stripped)
                    if len(tail_lines) > 120:
                        tail_lines.pop(0)
        process.wait()
        if process.returncode == 0:
            if safe_workspace is not None:
                copied = _copy_back_textgrids(work_output_folder, output_folder)
                log(f"[MFA] Copied back {copied} TextGrid files from ASCII-safe workspace.")
            log('MFA alignment completed successfully.')
            return True, ''
        joined_tail = '\n'.join(tail_lines[-40:])
        lowered_tail = joined_tail.lower()
        if (
            'please install korean support' in lowered_tail
            or ('importerror' in lowered_tail and 'eunjeon' in lowered_tail and 'jamo' in lowered_tail)
        ):
            err = 'Korean dependencies (eunjeon, jamo) are missing in MFA env.'
            log(err)
            return False, err
        err = f'MFA alignment failed (code: {process.returncode})'
        if tail_lines:
            err += f' | tail: {tail_lines[-1][:180]}'
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

def patch_mfa_korean_support(mfa_path, callback=None):
    """
    Windows 환경에서 python-mecab-ko는 C++ 빌드 툴이 없어 설치가 실패합니다.
    대신 윈도우용 사전 컴파일된 eunjeon(mecab-ko 포크)를 설치한 뒤, 
    MFA 내부 소스코드(spacy.py, korean.py)가 eunjeon을 참조하도록 강제로 패치합니다.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if sys.platform != 'win32' or not mfa_path or 'Scripts' not in mfa_path:
        return True
        
    try:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        site_packages = os.path.join(env_dir, 'Lib', 'site-packages', 'montreal_forced_aligner')
        
        spacy_py = os.path.join(site_packages, 'tokenization', 'spacy.py')
        korean_py = os.path.join(site_packages, 'tokenization', 'korean.py')
        
        # 1. spacy.py 패치: 'mecab' 대신 'eunjeon'을 체크하도록 수정
        if os.path.exists(spacy_py):
            with open(spacy_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # (1) 에러 메시지 수정 (eunjeon 안내 포함)
            if 'pip install python-mecab-ko jamo' in content:
                content = content.replace("pip install python-mecab-ko jamo", "pip install eunjeon jamo")
            
            # (2) 가용성 체크 로직 수정: 'from mecab import' 나 'import mecab'을 eunjeon으로 우회
            if "import mecab" in content and "import eunjeon" not in content:
                content = content.replace("import mecab", "import eunjeon")
                
            with open(spacy_py, 'w', encoding='utf-8') as f:
                f.write(content)
            log("   [Patch] spacy.py 가용성 체크 수정 완료")

        # 2. korean.py 패치: KO_AVAILABLE를 True로 만들고 Eunjeon 래퍼 주입
        if os.path.exists(korean_py):
            with open(korean_py, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # (1) KO_AVAILABLE 결정 로직 수정 (eunjeon이 있으면 True가 되도록)
            # 중복 패치 방지 및 손상된 파일 복구 로직 추가
            if 'EunjeonWrapper' not in content:
                # 이미 잘못된 패치가 되어있는 경우 (중첩 try) 복구 시도
                content = re.sub(r'try:\s+try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
                content = re.sub(r'try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
                
                # 정석적인 4분할 시퀀스로 교체 (정확한 인덴트 유지)
                if '    from mecab import MeCab' in content:
                    content = content.replace(
                        '    from mecab import MeCab', 
                        '    try:\n        from mecab import MeCab\n    except:\n        from eunjeon import Mecab as MeCab'
                    )
                elif 'from mecab import MeCab' in content:
                    # 인덴트가 없는 경우 (가능성은 낮지만 방어용)
                    content = content.replace(
                        'from mecab import MeCab', 
                        'try:\n    from mecab import MeCab\nexcept:\n    from eunjeon import Mecab as MeCab'
                    )
                
            wrapper_code = '''
class EunjeonNode:
    def __init__(self, surface, pos):
        self.surface = surface
        self.pos = pos

class EunjeonWrapper:
    def __init__(self):
        from eunjeon import Mecab
        self.mecab = Mecab()
        
    def parse(self, text):
        return [EunjeonNode(w, p) for w, p in self.mecab.pos(text)]
'''
            if 'class EunjeonWrapper' not in content:
                # Add wrapper class after imports
                content = content.replace('class KoreanTokenizer:', wrapper_code + '\nclass KoreanTokenizer:')
                # Replace the tokenizer instantiation
                if 'self.tokenizer = MeCab()' in content:
                    content = content.replace("self.tokenizer = MeCab()", "self.tokenizer = EunjeonWrapper()")
                
            with open(korean_py, 'w', encoding='utf-8') as f:
                f.write(content)
            log("   [Patch] korean.py 참조 수정 (KO_AVAILABLE 및 Eunjeon 래퍼) 완료")
                
        return True
    except Exception as e:
        log(f"⚠️ MFA 한국어 패치 중 오류 발생: {e}")
        return False





