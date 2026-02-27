"""
SOFA (Singing-Oriented Forced Aligner) 연동 모듈
- 동일 가상환경(.env)에서 SOFA 의존성 설치
- SOFA infer.py 실행 후 TextGrid 결과를 프로젝트 출력 폴더로 수집
"""

import os
import sys
import glob
import shutil
import logging
import subprocess
import json
import zipfile
import urllib.request
import re
import tempfile

logger = logging.getLogger(__name__)

SOFA_REPO_URL = "https://github.com/qiuqiao/SOFA.git"
SOFA_MODEL_RELEASES = {
    "japanese": ("ariikamusic", "SOFA_Models", "akm_ja_v001"),
    "korean": ("colstone", "SOFA_Models", "KOR-V0.01b"),
}
SOFA_RELEASE_LINKS = {
    "japanese": "https://github.com/ariikamusic/SOFA_Models/releases/tag/akm_ja_v001",
    "korean": "https://github.com/colstone/SOFA_Models/releases/tag/KOR-V0.01b",
}
FFMPEG_RELEASE_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def _subprocess_ui_kwargs():
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs


def get_sofa_release_link(language):
    return SOFA_RELEASE_LINKS.get(language, SOFA_RELEASE_LINKS["korean"])


def get_default_sofa_model_root():
    return os.path.join(os.path.expanduser("~"), "SOFA_Models")


def _app_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runtime_base_dir():
    # PyInstaller onefile 실행 시 번들 데이터는 _MEIPASS 아래에 풀립니다.
    return getattr(sys, "_MEIPASS", _app_dir())


def _sofa_tool_dir():
    return os.path.join(_app_dir(), ".sofa", "tools")


def _runtime_ffmpeg_bin_dir():
    return os.path.join(_sofa_tool_dir(), "ffmpeg", "bin")


def _ensure_runtime_ffmpeg_bin(callback=None):
    """
    DLL 누락 팝업을 피하기 위해 SOFA 전용 ffmpeg를 로컬에 준비합니다.
    """
    if sys.platform != "win32":
        return ""

    ffmpeg_bin = _runtime_ffmpeg_bin_dir()
    ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe")
    ffprobe_exe = os.path.join(ffmpeg_bin, "ffprobe.exe")
    if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
        return ffmpeg_bin

    tool_dir = _sofa_tool_dir()
    os.makedirs(tool_dir, exist_ok=True)
    tmp_zip = os.path.join(tool_dir, "ffmpeg_release_essentials.zip")
    tmp_extract = tempfile.mkdtemp(prefix="ffmpeg_extract_", dir=tool_dir)
    target_root = os.path.join(tool_dir, "ffmpeg")

    try:
        if callback:
            callback("SOFA용 FFmpeg 다운로드 중...")
        with urllib.request.urlopen(FFMPEG_RELEASE_ZIP_URL, timeout=180) as resp:
            with open(tmp_zip, "wb") as f:
                f.write(resp.read())

        if callback:
            callback("SOFA용 FFmpeg 압축 해제 중...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract)

        source_bin = ""
        for root, dirs, _ in os.walk(tmp_extract):
            if "bin" in dirs and os.path.exists(os.path.join(root, "bin", "ffmpeg.exe")):
                source_bin = os.path.join(root, "bin")
                break

        if not source_bin:
            raise RuntimeError("ffmpeg.exe를 찾지 못했습니다.")

        if os.path.exists(target_root):
            shutil.rmtree(target_root, ignore_errors=True)
        os.makedirs(target_root, exist_ok=True)
        shutil.copytree(source_bin, ffmpeg_bin)

        if callback:
            callback(f"SOFA용 FFmpeg 준비 완료: {ffmpeg_bin}")
        return ffmpeg_bin
    except Exception as e:
        if callback:
            callback(f"SOFA용 FFmpeg 준비 실패: {e}")
        return ""
    finally:
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
        shutil.rmtree(tmp_extract, ignore_errors=True)


def _get_bundled_ffmpeg_bin_dirs(extra_dirs=None):
    dirs = []
    extra = [d for d in (extra_dirs or []) if d]
    candidates = extra + [
        os.path.join(_runtime_base_dir(), "ffmpeg", "bin"),
        os.path.join(_app_dir(), "ffmpeg", "bin"),
        # 개발 실행(소스 트리)에서는 build_assets의 정적 ffmpeg를 우선 활용
        os.path.join(_app_dir(), "build_assets", "ffmpeg", "bin"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "ffmpeg.exe")):
            dirs.append(d)
    # 순서 유지 중복 제거
    seen = set()
    unique = []
    for d in dirs:
        key = os.path.normcase(os.path.normpath(d))
        if key not in seen:
            unique.append(d)
            seen.add(key)
    return unique


def _prepend_ffmpeg_to_env_path(env, callback=None, extra_dirs=None):
    ffmpeg_bins = _get_bundled_ffmpeg_bin_dirs(extra_dirs=extra_dirs)
    if not ffmpeg_bins:
        return env
    old_path = env.get("PATH", "")
    parts = [p for p in old_path.split(os.pathsep) if p]
    part_keys = {os.path.normcase(os.path.normpath(p)) for p in parts}
    for d in reversed(ffmpeg_bins):
        key = os.path.normcase(os.path.normpath(d))
        if key not in part_keys:
            parts.insert(0, d)
            part_keys.add(key)
    env["PATH"] = os.pathsep.join(parts)
    ffmpeg_exe = os.path.join(ffmpeg_bins[0], "ffmpeg.exe")
    # 일부 라이브러리는 PATH 대신 명시적 ffmpeg 경로 환경변수를 참조합니다.
    env["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    env["FFMPEG_BINARY"] = ffmpeg_exe
    env["FFMPEG_EXE"] = ffmpeg_exe
    if callback:
        callback(f"ℹ FFmpeg 경로 적용: {ffmpeg_bins[0]}")
    return env


def get_sofa_env_dir():
    return os.path.join(_app_dir(), ".env_sofa")


def get_sofa_env_python():
    if sys.platform == "win32":
        return os.path.join(get_sofa_env_dir(), "Scripts", "python.exe")
    return os.path.join(get_sofa_env_dir(), "bin", "python")


def _resolve_venv_builder_cmd():
    """
    venv 생성용 Python 실행 커맨드를 찾습니다.
    PyInstaller onefile에서는 sys.executable(앱 exe) 재실행을 피해야 합니다.
    """
    candidates = []

    if sys.platform == "win32":
        py_launcher = shutil.which("py")
        if py_launcher:
            candidates.append([py_launcher, "-3"])

    cur = sys.executable
    if cur and os.path.exists(cur) and "python" in os.path.basename(cur).lower():
        candidates.append([cur])

    base = getattr(sys, "_base_executable", "")
    if base and os.path.exists(base) and "python" in os.path.basename(base).lower():
        candidates.append([base])

    path_python = shutil.which("python")
    if path_python:
        candidates.append([path_python])

    seen = set()
    deduped = []
    for cmd in candidates:
        key = tuple(cmd)
        if key not in seen:
            deduped.append(cmd)
            seen.add(key)

    test_code = "import sys; print(sys.executable)"
    for cmd in deduped:
        try:
            proc = subprocess.run(
                cmd + ["-c", test_code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_subprocess_ui_kwargs(),
            )
            if proc.returncode != 0:
                continue
            lines = [x.strip() for x in (proc.stdout or "").splitlines() if x.strip()]
            if not lines:
                continue
            resolved_exe = os.path.abspath(lines[-1])
            app_exe = os.path.abspath(sys.executable)
            if getattr(sys, "frozen", False) and resolved_exe == app_exe:
                continue
            return cmd
        except Exception:
            continue
    return None


def _resolve_python_exe(sofa_python="", mfa_path=""):
    # 0) 사용자 지정 경로 우선
    if sofa_python and os.path.exists(sofa_python):
        return sofa_python

    # 1) SOFA 전용 env
    py = get_sofa_env_python()
    if os.path.exists(py):
        return py

    # 2) MFA env fallback
    if mfa_path and "Scripts" in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        py = os.path.join(env_dir, "python.exe")
        if os.path.exists(py):
            return py

    # 3) 프로젝트 .env
    py = os.path.join(_app_dir(), ".env", "python.exe")
    if os.path.exists(py):
        return py

    # 4) frozen 환경에서는 앱 exe 재실행을 피하고 시스템 python 우선
    if getattr(sys, "frozen", False):
        path_python = shutil.which("python")
        if path_python:
            return path_python
        base = getattr(sys, "_base_executable", "")
        if base and os.path.exists(base) and "python" in os.path.basename(base).lower():
            return base

    # 5) 최종 fallback
    return sys.executable


def ensure_sofa_env(callback=None):
    """
    SOFA 전용 가상환경(.env_sofa)을 준비합니다.
    """
    py = get_sofa_env_python()
    if os.path.exists(py):
        return True, py, ""

    env_dir = get_sofa_env_dir()
    os.makedirs(os.path.dirname(env_dir), exist_ok=True)
    if callback:
        callback(f"📦 SOFA 전용 가상환경 생성 중: {env_dir}")

    venv_builder = _resolve_venv_builder_cmd()
    if not venv_builder:
        return False, "", (
            "SOFA 전용 가상환경 생성용 Python을 찾지 못했습니다. "
            "Python 3.x를 설치하거나 PATH에 python/py를 등록해 주세요."
        )
    if callback:
        callback(f"ℹ venv 생성 인터프리터: {' '.join(venv_builder)}")

    try:
        proc = subprocess.run(
            venv_builder + ["-m", "venv", env_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_subprocess_ui_kwargs(),
        )
        if proc.returncode != 0:
            return False, "", f"SOFA 전용 가상환경 생성 실패: {proc.stderr or proc.stdout}"
    except Exception as e:
        return False, "", f"SOFA 전용 가상환경 생성 중 예외: {e}"

    if not os.path.exists(py):
        return False, "", "SOFA 전용 가상환경 생성 후 python.exe를 찾지 못했습니다."

    # pip 최신화는 실패해도 계속 진행
    try:
        subprocess.run(
            [py, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_subprocess_ui_kwargs(),
        )
    except Exception:
        pass
    return True, py, ""


def _run(cmd, callback=None, cwd=None, env=None):
    if callback:
        callback(f"   -> 실행: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_ui_kwargs(),
    )
    output = []
    if proc.stdout:
        for line in proc.stdout:
            line = line.strip()
            if line:
                output.append(line)
                if callback:
                    callback(line)
    proc.wait()
    return proc.returncode, "\n".join(output)


def _check_imports(py, modules):
    code = "; ".join(f"import {m}" for m in modules)
    proc = subprocess.run(
        [py, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_ui_kwargs(),
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout or "").strip()


def _get_torch_pair_versions(py):
    code = (
        "import torch, torchaudio; "
        "print(getattr(torch, '__version__', '')); "
        "print(getattr(torchaudio, '__version__', ''))"
    )
    proc = subprocess.run(
        [py, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_ui_kwargs(),
    )
    if proc.returncode != 0:
        return "", ""
    lines = [x.strip() for x in (proc.stdout or "").splitlines() if x.strip()]
    if len(lines) < 2:
        return "", ""
    return lines[0], lines[1]


def is_sofa_ready(sofa_python="", mfa_path=""):
    """
    SOFA 실행 준비 완료 여부를 판별합니다.
    - SOFA 코드(infer.py) 존재
    - 핵심 런타임 의존성 import 가능
    """
    py = _resolve_python_exe(sofa_python=sofa_python, mfa_path=mfa_path)
    infer_py = os.path.join(_app_dir(), ".sofa", "SOFA", "infer.py")
    if not os.path.exists(py) or not os.path.exists(infer_py):
        return False, "python 또는 infer.py가 없습니다."

    required_modules = ["click", "requests", "torch", "torchaudio"]
    ok, err = _check_imports(py, required_modules)
    if not ok:
        return False, err
    return True, ""


def _ensure_sofa_runtime_deps(py, callback=None):
    """
    SOFA 실행 시 자주 누락되는 런타임 의존성(torch/torchaudio)을 보강합니다.
    """
    ok, _ = _check_imports(py, ["torch", "torchaudio"])
    if ok:
        tv, tav = _get_torch_pair_versions(py)
        # torchcodec 의존 이슈를 피하기 위해 SOFA에서는 검증된 쌍으로 고정
        if tv.startswith("2.5.1") and tav.startswith("2.5.1"):
            return True, ""
        if callback:
            callback(f"ℹ torch/torchaudio 버전 보정: {tv or '?'} / {tav or '?'} -> 2.5.1/2.5.1")

    if callback:
        callback("📦 SOFA 런타임 의존성(torch, torchaudio) 설치 중...")
    # CPU 휠 인덱스를 우선 사용
    cpu_cmd = [
        py, "-m", "pip", "install",
        "torch==2.5.1", "torchaudio==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]
    rc, out = _run(cpu_cmd, callback=callback)
    if rc != 0:
        # 일반 PyPI fallback
        fallback = [py, "-m", "pip", "install", "torch", "torchaudio"]
        rc2, out2 = _run(fallback, callback=callback)
        if rc2 != 0:
            return False, f"torch/torchaudio 설치 실패\n{(out2 or out)[-1000:]}"

    ok2, err2 = _check_imports(py, ["torch", "torchaudio"])
    if not ok2:
        return False, f"torch/torchaudio import 확인 실패: {err2}"
    return True, ""


def _install_compatible_torchaudio_pair(py, callback=None):
    """
    TorchCodec 의존 이슈를 피하기 위해 torch/torchaudio 호환 쌍으로 재설치합니다.
    """
    if callback:
        callback("📦 torch/torchaudio 호환 버전 재설치 중...")
    _run([py, "-m", "pip", "uninstall", "-y", "torchcodec"], callback=callback)
    _run([py, "-m", "pip", "uninstall", "-y", "torchaudio", "torch"], callback=callback)
    rc, out = _run(
        [
            py, "-m", "pip", "install",
            "torch==2.5.1", "torchaudio==2.5.1",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        callback=callback,
    )
    if rc != 0:
        return False, f"호환 버전 재설치 실패\n{out[-800:] if out else ''}"
    ok2, err2 = _check_imports(py, ["torch", "torchaudio"])
    if not ok2:
        return False, f"재설치 후 import 확인 실패: {err2}"
    return True, ""


def _download_file(url, out_path, callback=None):
    if callback:
        callback(f"⬇ 다운로드: {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "UTAU-Auto-OTO/2.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(out_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    if callback:
        callback(f"✅ 저장: {out_path}")


def _load_dictionary_keys(dict_path):
    keys = set()
    if not dict_path or not os.path.exists(dict_path):
        return keys
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            key = line.split("\t", 1)[0].strip()
            if key:
                keys.add(key)
    return keys


def _read_text_auto(path):
    for enc in ("utf-8", "utf-8-sig", "cp949", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def _greedy_split_by_dictionary(text, dict_keys):
    if not text or not dict_keys:
        return None
    vocab = [k for k in dict_keys if k and " " not in k]
    if not vocab:
        return None
    vocab_set = set(vocab)
    max_len = max(len(k) for k in vocab)
    out = []
    i = 0
    n = len(text)
    while i < n:
        match = None
        span = min(max_len, n - i)
        while span > 0:
            cand = text[i : i + span]
            if cand in vocab_set:
                match = cand
                break
            span -= 1
        if not match:
            return None
        out.append(match)
        i += len(match)
    return out


def _normalize_lab_for_sofa(text, dict_keys):
    # SOFA DictionaryG2P는 공백 구분 토큰을 dictionary key로 조회한다.
    # 공백/인코딩이 불안정한 lab를 dictionary 기준으로 재토큰화한다.
    if not text:
        return ""
    normalized_ws = re.sub(r"\s+", " ", text).strip()
    if not normalized_ws:
        return ""

    tokens = [t for t in normalized_ws.split(" ") if t]
    if tokens and all(t in dict_keys for t in tokens):
        return " ".join(tokens)

    compact = "".join(tokens)
    if compact in dict_keys:
        return compact

    split_tokens = _greedy_split_by_dictionary(compact, dict_keys)
    if split_tokens:
        return " ".join(split_tokens)

    return " ".join(tokens)


def find_sofa_ckpt(language, search_root=None):
    """
    사용자 모델 폴더에서 언어별 SOFA ckpt를 탐색합니다.
    """
    root = search_root or get_default_sofa_model_root()
    if not os.path.exists(root):
        return ""
    lang = (language or "").lower()
    tokens = {
        "korean": ["kor", "korean", "ko"],
        "japanese": ["ja", "japanese", "jp"],
    }.get(lang, [])

    cands = sorted(glob.glob(os.path.join(root, "**", "*.ckpt"), recursive=True))
    if not cands:
        return ""
    for p in cands:
        low = p.lower()
        if any(t in low for t in tokens):
            return p
    return cands[0]


def download_default_sofa_model(language, target_root=None, callback=None):
    """
    GitHub release에서 언어별 기본 SOFA 모델을 자동 다운로드합니다.
    """
    lang = (language or "").lower()
    if lang not in SOFA_MODEL_RELEASES:
        return False, "", f"지원하지 않는 언어: {language}"

    owner, repo, tag = SOFA_MODEL_RELEASES[lang]
    root = target_root or get_default_sofa_model_root()
    lang_dir = os.path.join(root, lang)
    os.makedirs(lang_dir, exist_ok=True)

    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    meta_path = os.path.join(lang_dir, f"{repo}_{tag}.json")
    try:
        _download_file(api_url, meta_path, callback=callback)
        data = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception as e:
        return False, "", f"릴리즈 메타데이터 조회 실패: {e}"
    finally:
        try:
            os.remove(meta_path)
        except Exception:
            pass

    assets = data.get("assets") or []
    if not assets:
        return False, "", "릴리즈에 다운로드 가능한 assets가 없습니다."

    def _pick_asset():
        # 우선순위: .ckpt > .zip > 기타
        for ext in (".ckpt", ".zip"):
            for a in assets:
                name = (a.get("name") or "").lower()
                if name.endswith(ext):
                    return a
        return assets[0]

    asset = _pick_asset()
    asset_name = asset.get("name") or f"{repo}_{tag}.bin"
    dl_url = asset.get("browser_download_url")
    if not dl_url:
        return False, "", "asset 다운로드 URL이 없습니다."

    out_asset = os.path.join(lang_dir, asset_name)
    try:
        _download_file(dl_url, out_asset, callback=callback)
    except Exception as e:
        return False, "", f"모델 파일 다운로드 실패: {e}"

    # zip이면 자동 압축 해제 후 ckpt 탐색
    if out_asset.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(out_asset, "r") as zf:
                zf.extractall(lang_dir)
            if callback:
                callback(f"📦 압축 해제 완료: {out_asset}")
            ckpt = find_sofa_ckpt(lang, search_root=lang_dir)
            if ckpt:
                return True, ckpt, ""
            return True, out_asset, ""
        except Exception as e:
            return False, "", f"압축 해제 실패: {e}"

    if out_asset.lower().endswith(".ckpt"):
        return True, out_asset, ""

    ckpt = find_sofa_ckpt(lang, search_root=lang_dir)
    if ckpt:
        return True, ckpt, ""
    return True, out_asset, ""


def ensure_sofa_support(mfa_path="", sofa_python="", callback=None):
    """
    SOFA 실행에 필요한 코드/의존성을 현재 환경에 준비합니다.
    """
    sofa_root = os.path.join(_app_dir(), ".sofa", "SOFA")
    os.makedirs(os.path.dirname(sofa_root), exist_ok=True)

    ok_env, py, env_err = ensure_sofa_env(callback=callback)
    if not ok_env:
        # 전용 env 생성 실패 시 기존 경로 fallback
        if callback:
            callback(f"⚠ SOFA 전용 env 준비 실패, 기존 Python으로 시도합니다: {env_err}")
        py = _resolve_python_exe(sofa_python=sofa_python, mfa_path=mfa_path)
    if callback:
        callback(f"🔍 SOFA 환경 점검: {py}")

    if sys.platform == "win32" and not shutil.which("git"):
        return False, "git 실행 파일을 찾지 못했습니다. Git for Windows 설치가 필요합니다."

    if not os.path.exists(os.path.join(sofa_root, "infer.py")):
        if callback:
            callback("📦 SOFA 저장소를 다운로드합니다...")
        rc, _ = _run(
            ["git", "clone", "--depth", "1", SOFA_REPO_URL, sofa_root],
            callback=callback,
        )
        if rc != 0:
            return False, "SOFA 저장소 다운로드에 실패했습니다."

    req = os.path.join(sofa_root, "requirements.txt")
    if not os.path.exists(req):
        return False, f"SOFA requirements.txt를 찾을 수 없습니다: {req}"

    if callback:
        callback("📦 SOFA 의존성 설치 중...")
        callback("ℹ SOFA README 기준 Python 3.8 권장입니다.")
    rc, out = _run([py, "-m", "pip", "install", "-r", req], callback=callback, cwd=sofa_root)
    if rc != 0:
        return False, f"SOFA 의존성 설치 실패\n{out[-800:] if out else ''}"

    dep_ok, dep_err = _ensure_sofa_runtime_deps(py, callback=callback)
    if not dep_ok:
        return False, dep_err

    return True, ""


def _prepare_sofa_segments(wav_folder, dictionary_path, callback=None):
    """
    SOFA 입력 포맷(segments/singer/*.wav, *.lab)에 맞춰 작업 폴더를 구성합니다.
    """
    seg_root = os.path.join(wav_folder, "_sofa_segments")
    singer_dir = os.path.join(seg_root, "singer1")
    if os.path.exists(seg_root):
        shutil.rmtree(seg_root, ignore_errors=True)
    os.makedirs(singer_dir, exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(wav_folder, "*.wav")))
    dict_keys = _load_dictionary_keys(dictionary_path)
    copied = 0
    skipped = 0
    normalized_count = 0
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        lab = os.path.join(wav_folder, base + ".lab")
        if not os.path.exists(lab):
            skipped += 1
            continue
        shutil.copy2(w, os.path.join(singer_dir, os.path.basename(w)))
        dst_lab = os.path.join(singer_dir, base + ".lab")
        raw = _read_text_auto(lab)
        norm = _normalize_lab_for_sofa(raw, dict_keys)
        with open(dst_lab, "w", encoding="utf-8") as f:
            f.write(norm)
        if norm != raw.strip():
            normalized_count += 1
        copied += 1

    if callback:
        callback(
            f"📁 SOFA 입력 세그먼트 준비: {copied}개 "
            f"(lab 없음 스킵 {skipped}개, lab 정규화 {normalized_count}개)"
        )
    return seg_root, copied


def _collect_textgrids(seg_root, output_folder, callback=None):
    os.makedirs(output_folder, exist_ok=True)
    patterns = [
        os.path.join(seg_root, "**", "TextGrid", "*.TextGrid"),
        os.path.join(seg_root, "**", "textgrid", "*.TextGrid"),
    ]
    tg_files = []
    for p in patterns:
        tg_files.extend(glob.glob(p, recursive=True))
    tg_files = sorted(set(tg_files))

    copied = 0
    for tg in tg_files:
        dst = os.path.join(output_folder, os.path.basename(tg))
        shutil.copy2(tg, dst)
        copied += 1
    if callback:
        callback(f"📂 SOFA TextGrid 수집 완료: {copied}개 -> {output_folder}")
    return copied


def run_sofa_align(
    wav_folder,
    output_folder,
    ckpt_path,
    dictionary_path,
    mfa_path="",
    sofa_python="",
    callback=None,
):
    """
    SOFA 정렬 실행 후 TextGrid를 output_folder로 모읍니다.
    """
    if not os.path.exists(wav_folder):
        return False, f"WAV 폴더를 찾을 수 없습니다: {wav_folder}"
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False, f"SOFA 체크포인트(.ckpt) 파일을 찾을 수 없습니다: {ckpt_path}"
    if not dictionary_path or not os.path.exists(dictionary_path):
        return False, f"SOFA 사전 파일을 찾을 수 없습니다: {dictionary_path}"

    ok, err = ensure_sofa_support(
        mfa_path=mfa_path,
        sofa_python=sofa_python,
        callback=callback,
    )
    if not ok:
        return False, err

    py = _resolve_python_exe(sofa_python=sofa_python, mfa_path=mfa_path)
    sofa_root = os.path.join(_app_dir(), ".sofa", "SOFA")
    infer_py = os.path.join(sofa_root, "infer.py")
    if not os.path.exists(infer_py):
        return False, "SOFA infer.py를 찾을 수 없습니다."

    seg_root, count = _prepare_sofa_segments(
        wav_folder,
        dictionary_path=dictionary_path,
        callback=callback,
    )
    if count == 0:
        return False, "SOFA 입력용 wav/lab 쌍이 없습니다. 먼저 Lab 생성 단계를 실행해 주세요."

    if callback:
        callback("🚀 SOFA 정렬 시작...")
    run_env = os.environ.copy()
    run_env["PYTHONUTF8"] = "1"
    run_env["PYTHONIOENCODING"] = "utf-8"
    runtime_ffmpeg_bin = _ensure_runtime_ffmpeg_bin(callback=callback)
    extra_bins = [runtime_ffmpeg_bin] if runtime_ffmpeg_bin else []
    run_env = _prepend_ffmpeg_to_env_path(run_env, callback=callback, extra_dirs=extra_bins)
    cmd = [
        py, "-X", "utf8", infer_py,
        "--ckpt", ckpt_path,
        "--folder", seg_root,
        "--dictionary", dictionary_path,
        "--out_formats", "TextGrid",
    ]
    rc, out = _run(cmd, callback=callback, cwd=sofa_root, env=run_env)
    if rc != 0 and (
        "TorchCodec is required for load_with_torchcodec" in (out or "")
        or "Could not load libtorchcodec" in (out or "")
    ):
        if callback:
            callback("⚠ TorchCodec/FFmpeg 호환 오류 감지: torch/torchaudio 호환 쌍으로 1회 재시도합니다.")
        dep_ok, dep_err = _install_compatible_torchaudio_pair(py, callback=callback)
        if not dep_ok:
            return False, dep_err
        rc, out = _run(cmd, callback=callback, cwd=sofa_root, env=run_env)
    if rc != 0:
        return False, f"SOFA 정렬 실패 (코드: {rc})\n{out[-1000:] if out else ''}"

    copied = _collect_textgrids(seg_root, output_folder, callback=callback)
    if copied <= 0:
        return False, "SOFA 정렬은 완료됐지만 TextGrid 결과를 찾지 못했습니다."
    return True, ""

