import argparse
import datetime
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_ASSET_DIR = os.path.join(APP_DIR, "build_assets")
FFMPEG_DIR = os.path.join(BUILD_ASSET_DIR, "ffmpeg")
FFMPEG_BIN_DIR = os.path.join(FFMPEG_DIR, "bin")
FFMPEG_RELEASE_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
REQUIRED_FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe")
REQUIRED_MSVC_RUNTIME_DLLS = ("msvcp140.dll", "msvcp140_1.dll")
MICROMAMBA_EXE_URL = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"

DEFAULT_APP_NAME = "UTAU_Auto_OTO"
DEFAULT_CHANNEL = "stable"
SUPPORTED_CHANNELS = ("stable", "preview")
CHANNEL_ALIASES = {"default": "stable"}
SUPPORTED_CHANNEL_INPUTS = ("stable", "preview", "default")
PREVIEW_REQUIREMENTS_FILE = "requirements-preview.txt"
PYDOMINO_PACKAGE = "pydomino"
DEFAULT_BACKEND = "nuitka"
SUPPORTED_BACKENDS = ("nuitka", "pyinstaller")

RELEASE_DIR_PREFIX = "UTAU_Auto_OTO_Release"

EXCLUDED_MODULES = [
    "torch",
    "torchaudio",
    "torchvision",
    "ml",
    "librosa",
]
EXCLUDED_TRAINING_MODULES = [
    "core.oto_ml.coupled.training",
]
RUNTIME_DATA_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "assets", "models", "oto_ml"), "assets/models/oto_ml"),
    (os.path.join(APP_DIR, "assets", "bootstrap", "get-pip.py"), "assets/bootstrap"),
    (os.path.join(APP_DIR, "ml", "configs"), "ml/configs"),
    (os.path.join(APP_DIR, "config.json"), "."),
]
RELEASE_AUX_FILES = [
    os.path.join(APP_DIR, "setup_mfa.bat"),
    os.path.join(APP_DIR, "requirements.txt"),
    os.path.join(APP_DIR, "requirements-ml.txt"),
    os.path.join(APP_DIR, "scripts", "runtime_recovery.ps1"),
    os.path.join(APP_DIR, "scripts", "startup_diagnose.ps1"),
    os.path.join(APP_DIR, "release_assets", "먼저 실행.txt"),
    os.path.join(APP_DIR, "release_assets", "설치_도우미.bat"),
    os.path.join(APP_DIR, "scripts", "startup_diagnose.bat"),
]
APP_ICON_CANDIDATES = [
    os.path.join(APP_DIR, "release_assets", "AutoOTO-icon.ico"),
    os.path.join(APP_DIR, "AutoOTO-icon.ico"),
]

args = argparse.Namespace(
    onefile=False,
    allow_unsafe_onefile=False,
    name=DEFAULT_APP_NAME,
    channel=DEFAULT_CHANNEL,
    channels="",
    backend=DEFAULT_BACKEND,
    skip_deps=False,
)

EXPECTED_BUILD_PYTHON = (3, 10)


def _configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _assert_build_python_version(expected=EXPECTED_BUILD_PYTHON):
    major, minor = expected
    if os.environ.get("UTOA_ALLOW_NON_310_BUILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("[WARN] UTOA_ALLOW_NON_310_BUILD enabled; skipping build Python version check.")
        return
    if (sys.version_info.major, sys.version_info.minor) != (major, minor):
        raise SystemExit(
            f"Build Python must be {major}.{minor}. "
            f"Current={sys.version_info.major}.{sys.version_info.minor} "
            f"({sys.executable})."
        )


def _normalize_channel(channel: str) -> str:
    normalized = (channel or "").strip().lower()
    return CHANNEL_ALIASES.get(normalized, normalized)


def _parse_channels(channel: str, channels: str) -> list[str]:
    raw_values = []
    if str(channels or "").strip():
        raw_values = [part.strip() for part in str(channels).split(",")]
    elif str(channel or "").strip():
        raw_values = [str(channel).strip()]
    resolved = []
    seen = set()
    for raw in raw_values:
        if not raw:
            continue
        normalized = _normalize_channel(raw)
        if normalized not in SUPPORTED_CHANNELS:
            raise SystemExit(
                f"Invalid channel: {raw} (supported: {', '.join(SUPPORTED_CHANNEL_INPUTS)})"
            )
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        resolved.append(DEFAULT_CHANNEL)
    return resolved


def _ensure_ffmpeg_bin():
    ffmpeg_exe = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    ffprobe_exe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")
    if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        print(f"FFmpeg reuse: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR

    os.makedirs(BUILD_ASSET_DIR, exist_ok=True)
    tmp_zip = os.path.join(BUILD_ASSET_DIR, "ffmpeg_release_essentials.zip")
    tmp_extract = tempfile.mkdtemp(prefix="ffmpeg_extract_", dir=BUILD_ASSET_DIR)
    try:
        print("Downloading FFmpeg (Windows shared build)...")
        with urllib.request.urlopen(FFMPEG_RELEASE_ZIP_URL, timeout=180) as resp:
            with open(tmp_zip, "wb") as f:
                f.write(resp.read())

        print("Extracting FFmpeg archive...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract)

        source_bin = ""
        for root, dirs, _ in os.walk(tmp_extract):
            if "bin" in dirs and os.path.exists(os.path.join(root, "bin", "ffmpeg.exe")):
                source_bin = os.path.join(root, "bin")
                break
        if not source_bin:
            raise RuntimeError("ffmpeg.exe was not found in archive.")

        if os.path.exists(FFMPEG_DIR):
            shutil.rmtree(FFMPEG_DIR)
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        shutil.copytree(source_bin, FFMPEG_BIN_DIR)
        print(f"FFmpeg prepared: {FFMPEG_BIN_DIR}")
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        return FFMPEG_BIN_DIR
    finally:
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
        shutil.rmtree(tmp_extract, ignore_errors=True)


def _validate_ffmpeg_bin(ffmpeg_bin):
    missing = []
    for file_name in REQUIRED_FFMPEG_BINARIES:
        full_path = os.path.join(ffmpeg_bin, file_name)
        if (not os.path.isfile(full_path)) or os.path.getsize(full_path) <= 0:
            missing.append(full_path)
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(f"Required FFmpeg runtime files are missing:\n{lines}")


def _ensure_micromamba_exe():
    """
    Download micromamba.exe for portable builds so setup_mfa.bat can skip tar/bzip2 extraction.
    Best-effort: warn and continue if download fails.
    """
    if os.name != "nt":
        return False
    target_dir = os.path.join(BUILD_ASSET_DIR, "micromamba")
    target_path = os.path.join(target_dir, "micromamba.exe")
    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        print(f"Micromamba reuse: {target_path}")
        return True
    os.makedirs(target_dir, exist_ok=True)
    try:
        print("Downloading micromamba.exe for portable bundle...")
        with urllib.request.urlopen(MICROMAMBA_EXE_URL, timeout=120) as resp:
            payload = resp.read()
        if not payload or len(payload) < 1024 * 512:
            raise RuntimeError("Downloaded micromamba.exe payload is too small.")
        with open(target_path, "wb") as f:
            f.write(payload)
        print(f"Micromamba prepared: {target_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to download micromamba.exe: {exc}")
        return False


def _iter_ffmpeg_runtime_files(ffmpeg_bin):
    if not os.path.isdir(ffmpeg_bin):
        return []
    files = []
    for name in sorted(os.listdir(ffmpeg_bin)):
        src = os.path.join(ffmpeg_bin, name)
        if not os.path.isfile(src):
            continue
        low = name.lower()
        if low.endswith(".exe") or low.endswith(".dll"):
            files.append((src, name))
    return files


def _iter_msvc_runtime_files():
    """
    Collect MSVC runtime DLLs for app-local bundling.
    This avoids target-machine startup failures when VC++ redistributable is absent.
    """
    candidates = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    if windir:
        candidates.append(os.path.join(windir, "System32"))
    candidates.append(os.path.join(sys.base_prefix, "DLLs"))
    candidates.append(sys.base_prefix)

    search_roots = []
    seen = set()
    for root in candidates:
        norm = os.path.normcase(os.path.abspath(str(root or "")))
        if not root or norm in seen or not os.path.isdir(root):
            continue
        seen.add(norm)
        search_roots.append(root)

    if not search_roots:
        print("[WARN] MSVC runtime search roots were not found.")
        return []

    found = []
    found_names = set()
    missing_required = []
    for dll_name in REQUIRED_MSVC_RUNTIME_DLLS:
        located_path = ""
        for root in search_roots:
            probe = os.path.join(root, dll_name)
            if os.path.isfile(probe):
                located_path = probe
                break
        if located_path:
            found.append((located_path, dll_name))
            found_names.add(dll_name.lower())
        else:
            missing_required.append(dll_name)

    if found:
        names = ", ".join(name for _, name in found)
        print(f"[INFO] Bundling MSVC runtime DLLs: {names}")
    if missing_required:
        missing = ", ".join(missing_required)
        print(
            "[WARN] Required MSVC runtime DLLs were not found on build machine: "
            f"{missing}. Target machine may require VC++ Redistributable."
        )
    return found


def _has_preview_channel(target_channels) -> bool:
    return any(str(ch).strip().lower() == "preview" for ch in (target_channels or []))


def _is_module_available(module_name: str) -> bool:
    if not module_name:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False

def _parse_args():
    parser = argparse.ArgumentParser(description="Build UTAU Auto OTO distributables.")
    parser.add_argument("--onefile", action="store_true", help="Build onefile executable (unsafe/experimental).")
    parser.add_argument(
        "--allow-unsafe-onefile",
        action="store_true",
        help="Acknowledge onefile runtime risks and allow --onefile build.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_APP_NAME,
        help="Executable name.",
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        choices=SUPPORTED_CHANNEL_INPUTS,
        help="Release channel tag (stable/default or preview).",
    )
    parser.add_argument(
        "--channels",
        default="",
        help="Comma-separated release channels to package in one build (e.g. stable,preview).",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=SUPPORTED_BACKENDS,
        help="Build backend (nuitka or pyinstaller).",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation step (useful in CI when already installed).",
    )
    return parser.parse_args()


def _detect_app_version():
    main_path = os.path.join(APP_DIR, "main.py")
    if not os.path.exists(main_path):
        return "0.0.0"
    try:
        with open(main_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return "0.0.0"


def _write_release_channel_metadata(target_path, app_name, app_version, channel):
    payload = {
        "app_name": app_name,
        "app_version": app_version,
        "channel": channel,
        "built_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_release_dir(channel):
    return os.path.join(APP_DIR, f"{RELEASE_DIR_PREFIX}_{channel}")


def _on_rmtree_error(func, path, exc_info):
    """
    Windows can leave read-only bits on copied directories/files.
    Try removing read-only and retry the failing operation.
    """
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    func(path)


def _clear_readonly_recursive(root_path):
    if not os.path.exists(root_path):
        return
    for dirpath, _, filenames in os.walk(root_path):
        try:
            os.chmod(dirpath, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            try:
                os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass


def _safe_rmtree(path, retries=3):
    if not os.path.exists(path):
        return
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=_on_rmtree_error)
            return
        except PermissionError as e:
            last_err = e
            _clear_readonly_recursive(path)
            if attempt < retries:
                time.sleep(0.4 * attempt)
        except OSError as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.4 * attempt)
    raise RuntimeError(f"Failed to remove directory after retries: {path}\nCause: {last_err}")


def _iter_runtime_data_entries():
    entries = []
    for src, dst in RUNTIME_DATA_PATHS:
        if os.path.exists(src):
            entries.append((src, dst))
        else:
            print(f"[WARN] Runtime data missing (skip): {src}")
    return entries


def _resolve_app_icon_path():
    for candidate in APP_ICON_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _build_pyinstaller_args(app_name, ffmpeg_bin, app_icon_path="", onefile=False, include_domino_module=False):
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    pyinstaller_args = [
        "main.py",
        f"--name={app_name}",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        f"--add-data={ctk_path};customtkinter/",
        f"--add-data={ffmpeg_bin};ffmpeg/bin",
        "--hidden-import=textgrid",
        "--hidden-import=customtkinter",
        "--hidden-import=onnxruntime",
    ]
    for src, name in _iter_msvc_runtime_files():
        pyinstaller_args.append(f"--add-binary={src};.")
    if include_domino_module:
        pyinstaller_args.append("--hidden-import=pydomino")
    for src, dst in _iter_runtime_data_entries():
        pyinstaller_args.append(f"--add-data={src};{dst}")
    if app_icon_path:
        pyinstaller_args.append(f"--icon={app_icon_path}")
    for module_name in EXCLUDED_MODULES + EXCLUDED_TRAINING_MODULES:
        pyinstaller_args.append(f"--exclude-module={module_name}")
    return pyinstaller_args

def _run_pyinstaller_build(app_name, ffmpeg_bin, app_icon_path="", onefile=False, include_domino_module=False):
    print("Loading PyInstaller...")
    import PyInstaller.__main__

    pyinstaller_args = _build_pyinstaller_args(
        app_name=app_name,
        ffmpeg_bin=ffmpeg_bin,
        app_icon_path=app_icon_path,
        onefile=onefile,
        include_domino_module=include_domino_module,
    )
    PyInstaller.__main__.run(pyinstaller_args)

    if onefile:
        exe_path = os.path.join(APP_DIR, "dist", f"{app_name}.exe")
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Built executable not found: {exe_path}")
        return exe_path

    dist_dir = os.path.join(APP_DIR, "dist", app_name)
    if not os.path.isdir(dist_dir):
        raise FileNotFoundError(f"Built app directory not found: {dist_dir}")
    return dist_dir

def _run_nuitka_build(app_name, ffmpeg_bin, app_icon_path="", onefile=False, include_domino_module=False):
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    output_root = os.path.join(APP_DIR, "dist_nuitka")
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    cpu_jobs = os.cpu_count() or 1
    cpu_jobs = max(1, int(cpu_jobs))
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "main.py",
        "--onefile" if onefile else "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=tk-inter",
        "--windows-console-mode=disable",
        f"--output-dir={output_root}",
        f"--output-filename={app_name}.exe",
        "--include-module=textgrid",
        "--include-package=customtkinter",
        "--include-package=onnxruntime",
        "--remove-output",
        f"--jobs={cpu_jobs}",
        f"--nofollow-import-to={','.join(EXCLUDED_MODULES + EXCLUDED_TRAINING_MODULES)}",
    ]
    if include_domino_module:
        cmd.append("--include-package=pydomino")

    _validate_ffmpeg_bin(ffmpeg_bin)
    include_entries = [(ctk_path, "customtkinter")]
    include_entries.extend(_iter_runtime_data_entries())
    for src, dst in include_entries:
        if os.path.isdir(src):
            cmd.append(f"--include-data-dir={src}={dst}")
        elif os.path.isfile(src):
            file_name = os.path.basename(src)
            target_path = f"{dst}/{file_name}" if str(dst).strip() not in {"", "."} else file_name
            cmd.append(f"--include-data-files={src}={target_path}")
        else:
            print(f"[WARN] Runtime include source missing (skip): {src}")
    ffmpeg_runtime_files = _iter_ffmpeg_runtime_files(ffmpeg_bin)
    for src, name in ffmpeg_runtime_files:
        cmd.append(f"--include-data-files={src}=ffmpeg/bin/{name}")
    for src, name in _iter_msvc_runtime_files():
        cmd.append(f"--include-data-files={src}={name}")
    if app_icon_path:
        cmd.append(f"--windows-icon-from-ico={app_icon_path}")

    print("Running Nuitka build command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    if onefile:
        expected = os.path.join(output_root, f"{app_name}.exe")
        if os.path.isfile(expected):
            return expected
        exe_candidates = [
            os.path.join(output_root, n)
            for n in os.listdir(output_root)
            if n.lower().endswith(".exe")
        ]
        if len(exe_candidates) == 1:
            return exe_candidates[0]
        raise FileNotFoundError(f"Nuitka onefile executable not found in: {output_root}")

    expected_dist = os.path.join(output_root, f"{app_name}.dist")
    if os.path.isdir(expected_dist):
        _validate_packaged_ffmpeg(expected_dist)
        return expected_dist

    dist_candidates = [
        os.path.join(output_root, n)
        for n in os.listdir(output_root)
        if n.lower().endswith(".dist") and os.path.isdir(os.path.join(output_root, n))
    ]
    if len(dist_candidates) == 1:
        _validate_packaged_ffmpeg(dist_candidates[0])
        return dist_candidates[0]

    raise FileNotFoundError(f"Nuitka standalone directory not found in: {output_root}")

def _validate_packaged_ffmpeg(dist_dir):
    ffmpeg_bin = os.path.join(dist_dir, "ffmpeg", "bin")
    missing = []
    for file_name in REQUIRED_FFMPEG_BINARIES:
        full_path = os.path.join(ffmpeg_bin, file_name)
        if (not os.path.isfile(full_path)) or os.path.getsize(full_path) <= 0:
            missing.append(full_path)
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(f"Nuitka output is missing required FFmpeg files:\n{lines}")


def _resolve_release_executable_path(release_dir, app_name, onefile=False):
    if onefile:
        preferred = os.path.join(release_dir, f"{app_name}.exe")
        if os.path.isfile(preferred):
            return preferred
        exe_candidates = [
            os.path.join(release_dir, name)
            for name in os.listdir(release_dir)
            if str(name).lower().endswith(".exe")
            and os.path.isfile(os.path.join(release_dir, name))
        ]
        if len(exe_candidates) == 1:
            return exe_candidates[0]
        return ""

    app_dir = os.path.join(release_dir, app_name)
    preferred = os.path.join(app_dir, f"{app_name}.exe")
    if os.path.isfile(preferred):
        return preferred
    if os.path.isdir(app_dir):
        exe_candidates = [
            os.path.join(app_dir, name)
            for name in os.listdir(app_dir)
            if str(name).lower().endswith(".exe")
            and os.path.isfile(os.path.join(app_dir, name))
        ]
        if len(exe_candidates) == 1:
            return exe_candidates[0]
    return ""


def _create_windows_shortcut(shortcut_path, target_path, working_dir="", description=""):
    if os.name != "nt":
        return False
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Shortcut target not found: {target_path}")

    def _ps_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    target_abs = os.path.abspath(target_path)
    shortcut_abs = os.path.abspath(shortcut_path)
    working_abs = os.path.abspath(working_dir or os.path.dirname(target_abs))
    os.makedirs(os.path.dirname(shortcut_abs), exist_ok=True)
    if os.path.exists(shortcut_abs):
        os.remove(shortcut_abs)

    ps_script = "\n".join(
        [
            "$wsh = New-Object -ComObject WScript.Shell",
            f"$shortcut = $wsh.CreateShortcut({_ps_quote(shortcut_abs)})",
            f"$shortcut.TargetPath = {_ps_quote(target_abs)}",
            f"$shortcut.WorkingDirectory = {_ps_quote(working_abs)}",
            f"$shortcut.IconLocation = {_ps_quote(f'{target_abs},0')}",
            f"$shortcut.Description = {_ps_quote(description or 'Launch app')}",
            "$shortcut.Save()",
        ]
    )
    subprocess.check_call(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ]
    )
    if not os.path.isfile(shortcut_abs):
        raise RuntimeError(f"Failed to create shortcut: {shortcut_abs}")
    return True


def _write_portable_launcher_cmd(release_dir, app_name, target_path):
    """
    Create a path-relocatable launcher script in the release root.
    This is robust across different machines/path roots unlike build-time .lnk files.
    """
    target_abs = os.path.abspath(target_path)
    release_abs = os.path.abspath(release_dir)
    try:
        rel_target = os.path.relpath(target_abs, release_abs)
    except Exception:
        rel_target = os.path.basename(target_abs)
    rel_target = str(rel_target).replace("/", "\\")

    launcher_name = f"Launch_{app_name}.cmd"
    launcher_path = os.path.join(release_abs, launcher_name)
    lines = [
        "@echo off",
        "setlocal EnableExtensions DisableDelayedExpansion",
        "set \"ROOT=%~dp0\"",
        f"set \"TARGET=%ROOT%{rel_target}\"",
        "if not exist \"%TARGET%\" (",
        "  echo [FAILED] App executable not found:",
        "  echo          %TARGET%",
        "  pause",
        "  exit /b 1",
        ")",
        "start \"\" \"%TARGET%\" %*",
        "exit /b %ERRORLEVEL%",
        "",
    ]
    with open(launcher_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\r\n".join(lines))
    return launcher_path


def _copy_release_outputs(app_name, channel, app_version, built_artifact_path, onefile=False):
    release_dir = _get_release_dir(channel)
    if os.path.exists(release_dir):
        _safe_rmtree(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    _write_release_channel_metadata(
        os.path.join(release_dir, "release_channel.json"),
        app_name=app_name,
        app_version=app_version,
        channel=channel,
    )

    if onefile:
        if not os.path.isfile(built_artifact_path):
            raise FileNotFoundError(f"Built executable not found: {built_artifact_path}")
        shutil.copy(built_artifact_path, release_dir)
        print(f"   -> copied: {built_artifact_path}")
    else:
        if not os.path.isdir(built_artifact_path):
            raise FileNotFoundError(f"Built app directory not found: {built_artifact_path}")
        target_dir = os.path.join(release_dir, app_name)
        shutil.copytree(built_artifact_path, target_dir)
        print(f"   -> copied directory: {built_artifact_path}")
        _write_release_channel_metadata(
            os.path.join(target_dir, "release_channel.json"),
            app_name=app_name,
            app_version=app_version,
            channel=channel,
        )

    for extra_path in RELEASE_AUX_FILES:
        if os.path.exists(extra_path):
            shutil.copy(extra_path, release_dir)
            print(f"   -> copied: {os.path.basename(extra_path)}")
    micromamba_src = os.path.join(BUILD_ASSET_DIR, "micromamba", "micromamba.exe")
    if os.path.exists(micromamba_src):
        shutil.copy(micromamba_src, os.path.join(release_dir, "micromamba.exe"))
        print("   -> copied: micromamba.exe")

    if os.name == "nt":
        release_exe = _resolve_release_executable_path(release_dir, app_name, onefile=onefile)
        if not release_exe:
            raise RuntimeError(
                f"Failed to locate release executable for shortcut creation: release_dir={release_dir}, app={app_name}"
            )
        launcher_path = _write_portable_launcher_cmd(
            release_dir=release_dir,
            app_name=app_name,
            target_path=release_exe,
        )
        print(f"   -> created portable launcher: {os.path.basename(launcher_path)}")

        shortcut_path = os.path.join(release_dir, f"{app_name}.lnk")
        create_lnk = str(os.environ.get("UTOA_CREATE_BUILD_SHORTCUT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if create_lnk:
            _create_windows_shortcut(
                shortcut_path=shortcut_path,
                target_path=release_exe,
                working_dir=os.path.dirname(release_exe),
                description=f"Launch {app_name}",
            )
            print(f"   -> created shortcut: {os.path.basename(shortcut_path)}")
        else:
            print("   -> skipped .lnk creation (portable-safe default).")
            print("      set UTOA_CREATE_BUILD_SHORTCUT=1 to keep build-time .lnk output.")
    return release_dir


def _install_build_dependencies(backend, target_channels=None):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    req_candidates = [
        os.path.join(APP_DIR, "requirements.txt"),
        os.path.join(APP_DIR, "requirements-ml.txt"),
    ]
    for req_path in req_candidates:
        if os.path.exists(req_path):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

    if backend == "nuitka":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka", "ordered-set", "zstandard"])
    elif backend == "pyinstaller":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    if not _has_preview_channel(target_channels):
        print("[INFO] Preview channel not selected; skipping pydomino dependency install.")
        return

    preview_req = os.path.join(APP_DIR, PREVIEW_REQUIREMENTS_FILE)
    if not os.path.isfile(preview_req):
        print(
            f"[WARN] Preview channel selected but {PREVIEW_REQUIREMENTS_FILE} was not found. "
            "Skipping pydomino install."
        )
        return

    print("[INFO] Preview channel selected; installing preview-only dependencies (pydomino)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-r",
            preview_req,
        ]
    )

def main():
    _configure_console_encoding()
    global args
    parsed_args = _parse_args()
    if parsed_args is not None:
        args = parsed_args

    target_channels = _parse_channels(args.channel, args.channels)
    args.channel = target_channels[0]
    include_domino_module = _has_preview_channel(target_channels)

    _assert_build_python_version()

    if args.onefile and not args.allow_unsafe_onefile:
        raise SystemExit(
            "onefile builds are disabled by default. Use --allow-unsafe-onefile to proceed."
        )
    if args.allow_unsafe_onefile and not args.onefile:
        print("[INFO] --allow-unsafe-onefile is ignored when --onefile is not used.")

    os.chdir(APP_DIR)
    app_version = _detect_app_version()
    mode_text = "onefile" if args.onefile else "onedir"

    channel_text = ",".join(target_channels)
    print(f"[INFO] channels={channel_text}, backend={args.backend}, version={app_version}")
    if args.skip_deps:
        print("[1/5] Skipping dependency install (--skip-deps).")
    else:
        print("[1/5] Installing build dependencies...")
        _install_build_dependencies(args.backend, target_channels=target_channels)

    if include_domino_module and not _is_module_available(PYDOMINO_PACKAGE):
        print("[WARN] Preview channel selected but pydomino is unavailable in this Python env.")
        print("[WARN] Domino runtime module will not be bundled in this build.")
        include_domino_module = False

    print("[2/5] Preparing FFmpeg runtime...")
    ffmpeg_bin = _ensure_ffmpeg_bin()
    _ensure_micromamba_exe()
    app_icon_path = _resolve_app_icon_path()
    if app_icon_path:
        print(f"[INFO] app_icon={app_icon_path}")
    else:
        print("[WARN] app icon not found; executable will use default icon.")

    print(f"[3/5] Building app with {args.backend}...")
    if args.backend == "nuitka":
        built_artifact = _run_nuitka_build(args.name, ffmpeg_bin, app_icon_path=app_icon_path, onefile=args.onefile, include_domino_module=include_domino_module)
    else:
        built_artifact = _run_pyinstaller_build(args.name, ffmpeg_bin, app_icon_path=app_icon_path, onefile=args.onefile, include_domino_module=include_domino_module)

    print(f"[4/5] Packaging release folder ({mode_text})...")
    release_dirs = []
    total_channels = len(target_channels)
    for idx, channel in enumerate(target_channels, start=1):
        print(f"[4/5] Packaging channel {idx}/{total_channels}: {channel}")
        release_dir = _copy_release_outputs(
            args.name,
            channel=channel,
            app_version=app_version,
            built_artifact_path=built_artifact,
            onefile=args.onefile,
        )
        release_dirs.append(release_dir)

    print("[5/5] Build complete.")
    for release_dir in release_dirs:
        print(f"[DONE] release_dir={release_dir}")


if __name__ == "__main__":
    main()
