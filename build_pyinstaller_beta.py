import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_ASSET_DIR = os.path.join(APP_DIR, "build_assets")
FFMPEG_DIR = os.path.join(BUILD_ASSET_DIR, "ffmpeg")
FFMPEG_BIN_DIR = os.path.join(FFMPEG_DIR, "bin")
FFMPEG_RELEASE_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
REQUIRED_FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe")

DEFAULT_APP_NAME = "UTAU_Auto_OTO"
DEFAULT_CHANNEL = "preview"
SUPPORTED_CHANNELS = ("stable", "preview")
RELEASE_DIR_PREFIX = "UTAU_Auto_OTO_Release_pyinstaller_beta"

APP_ICON_CANDIDATES = [
    os.path.join(APP_DIR, "release_assets", "AutoOTO-icon.ico"),
    os.path.join(APP_DIR, "AutoOTO-icon.ico"),
]

RUNTIME_DATA_BASE_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "ml", "configs"), "ml/configs"),
]

EXCLUDED_MODULES = [
    "torch",
    "torchaudio",
    "torchvision",
    "ml",
    "core.ja_oto_autotune",
    "core.oto_ml.coupled.training",
]

RELEASE_COPY_FILES = [
    "setup_mfa.bat",
    "setup_mfa_beta.bat",
    "requirements.txt",
    "requirements-ml.txt",
    "USER_OPERATION_GUIDE.md",
    "ERROR_CODES.md",
]

MASKED_TRAINING_DATA_ROOTS_YAML = """japanese:
  cv:
    - './data/japanese/cv'
  vcv:
    - './data/japanese/vcv'
  cvvc:
    - './data/japanese/cvvc'

korean:
  cv:
    - './data/korean/cv'
  cvc:
    - './data/korean/cvc'
  cvvc:
    - './data/korean/cvvc'
  vcv:
    - './data/korean/vcv'

notes:
  recursive_oto_search: true
  recursive_wav_search: true
  strip_pitch_suffix_for_matching: true
  strip_pitch_suffix_examples:
    - '_C4'
    - '_D4'
    - '-A4'
    - ' F4'
    - '_G4'
    - '#'
"""

MASKED_DATASET_BUILD_DEFAULT_YAML = """feature_version: v1
matching_key:
  - wav_norm
  - alias_norm
  - occurrence_index
split_policy: voicebank_group
workspace_root: ./workspace
"""


def _configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a beta release package using PyInstaller only. "
            "This script does not modify or depend on Nuitka-specific build flow."
        )
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_APP_NAME,
        help="Executable/app folder name.",
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        choices=SUPPORTED_CHANNELS,
        help="Release channel metadata value. Use preview for beta distribution.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build onefile executable (experimental, slower startup).",
    )
    parser.add_argument(
        "--allow-unsafe-onefile",
        action="store_true",
        help="Acknowledge onefile risks and allow --onefile build.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation if already prepared in the current Python environment.",
    )
    parser.add_argument(
        "--skip-ml-deps",
        action="store_true",
        help="Skip requirements-ml.txt during build dependency installation.",
    )
    parser.add_argument(
        "--clean-release-dir",
        action="store_true",
        help="Delete existing beta release directory before packaging.",
    )
    parser.add_argument(
        "--skip-ml-model-bundle",
        action="store_true",
        help="Do not bundle runtime ML model folders into the release payload.",
    )
    return parser.parse_args()


def _normalize_channel(channel):
    normalized = str(channel or DEFAULT_CHANNEL).strip().lower()
    if normalized in SUPPORTED_CHANNELS:
        return normalized
    return DEFAULT_CHANNEL


def _detect_app_version():
    main_path = os.path.join(APP_DIR, "main.py")
    if not os.path.exists(main_path):
        return "0.0.0"
    try:
        with open(main_path, "r", encoding="utf-8") as f:
            text = f.read()
        match = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1).strip()
    except OSError:
        pass
    return "0.0.0"


def _write_release_channel_metadata(target_path, app_name, app_version, channel):
    payload = {
        "app_name": app_name,
        "app_version": app_version,
        "channel": channel,
        "built_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "build_backend": "pyinstaller",
        "build_profile": "beta",
    }
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_checked(cmd, *, title):
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"[RUN] {title}")
    print(f"      {printable}")
    subprocess.check_call(cmd)


def _install_build_dependencies(skip_ml=False):
    print("[INFO] Installing build dependencies for PyInstaller beta package...")
    _run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], title="Upgrade pip toolchain")

    req_files = ["requirements.txt"]
    if not skip_ml:
        req_files.append("requirements-ml.txt")

    for req_name in req_files:
        req_path = os.path.join(APP_DIR, req_name)
        if not os.path.isfile(req_path):
            print(f"[WARN] Missing dependency file (skip): {req_path}")
            continue
        _run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "-r", req_path], title=f"Install {req_name}")

    _run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller", "pyinstaller-hooks-contrib"], title="Install PyInstaller")


def _validate_ffmpeg_bin(ffmpeg_bin):
    missing = []
    for file_name in REQUIRED_FFMPEG_BINARIES:
        file_path = os.path.join(ffmpeg_bin, file_name)
        if (not os.path.isfile(file_path)) or os.path.getsize(file_path) <= 0:
            missing.append(file_path)
    if missing:
        lines = "\n".join(f"  - {line}" for line in missing)
        raise RuntimeError(f"Required FFmpeg runtime files are missing:\n{lines}")


def _ensure_ffmpeg_bin():
    ffmpeg_exe = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    ffprobe_exe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")
    if os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe):
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        print(f"[OK] Reusing local FFmpeg runtime: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR

    os.makedirs(BUILD_ASSET_DIR, exist_ok=True)
    archive_path = os.path.join(BUILD_ASSET_DIR, "ffmpeg_release_essentials.zip")
    tmp_extract = tempfile.mkdtemp(prefix="ffmpeg_extract_", dir=BUILD_ASSET_DIR)
    try:
        print("[INFO] Downloading FFmpeg runtime archive...")
        with urllib.request.urlopen(FFMPEG_RELEASE_ZIP_URL, timeout=240) as resp:
            with open(archive_path, "wb") as f:
                f.write(resp.read())

        print("[INFO] Extracting FFmpeg runtime archive...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_extract)

        source_bin = ""
        for root, dirs, _ in os.walk(tmp_extract):
            if "bin" in dirs:
                maybe_bin = os.path.join(root, "bin")
                if os.path.isfile(os.path.join(maybe_bin, "ffmpeg.exe")):
                    source_bin = maybe_bin
                    break
        if not source_bin:
            raise RuntimeError("ffmpeg.exe was not found in the downloaded archive.")

        if os.path.exists(FFMPEG_DIR):
            shutil.rmtree(FFMPEG_DIR)
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        shutil.copytree(source_bin, FFMPEG_BIN_DIR)
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        print(f"[OK] FFmpeg prepared: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR
    finally:
        if os.path.isfile(archive_path):
            try:
                os.remove(archive_path)
            except OSError:
                pass
        shutil.rmtree(tmp_extract, ignore_errors=True)


def _iter_runtime_data_entries(channel=DEFAULT_CHANNEL, skip_ml_model_bundle=False):
    entries = []
    for src, dst in RUNTIME_DATA_BASE_PATHS:
        if os.path.exists(src):
            entries.append((src, dst))
        else:
            print(f"[WARN] Runtime data missing (skip): {src}")

    if skip_ml_model_bundle:
        print("[INFO] Runtime ML model bundle step skipped (--skip-ml-model-bundle).")
        return entries

    def _latest_model_meta_mtime(model_root):
        newest = 0.0
        try:
            for root, _dirs, files in os.walk(model_root):
                if "model_meta.json" not in files:
                    continue
                meta_path = os.path.join(root, "model_meta.json")
                try:
                    newest = max(newest, os.path.getmtime(meta_path))
                except OSError:
                    continue
        except Exception:
            newest = 0.0
        if newest > 0.0:
            return newest
        try:
            return float(os.path.getmtime(model_root))
        except OSError:
            return 0.0

    model_override = str(os.environ.get("UTOA_BUILD_MODEL_DIR", "") or "").strip()
    model_candidates = []
    if model_override:
        if os.path.isabs(model_override):
            model_candidates.append(model_override)
        else:
            model_candidates.append(os.path.join(APP_DIR, model_override))

    channel_norm = _normalize_channel(channel)
    model_candidates.extend(
        [
            os.path.join(APP_DIR, "ML_models_latest"),
            os.path.join(APP_DIR, f"ML_models_{channel_norm}"),
            os.path.join(APP_DIR, "ML_models"),
        ]
    )

    selected_model_source = ""
    existing_candidates = [(idx, path) for idx, path in enumerate(model_candidates) if os.path.isdir(path)]
    if existing_candidates:
        if model_override:
            selected_model_source = existing_candidates[0][1]
        else:
            ranked = [
                (_latest_model_meta_mtime(path), -idx, path)
                for idx, path in existing_candidates
            ]
            ranked.sort(reverse=True)
            selected_model_source = ranked[0][2]

    if selected_model_source:
        entries.append((selected_model_source, "ML_models"))
        print(f"[INFO] Runtime ML models source: {selected_model_source} -> ML_models")
    else:
        print("[WARN] Runtime ML model directory not found (ML_models*).")

    legacy_asset_models = os.path.join(APP_DIR, "assets", "models", "oto_ml")
    if os.path.isdir(legacy_asset_models):
        entries.append((legacy_asset_models, "assets/models/oto_ml"))
    else:
        print(f"[WARN] Runtime data missing (skip): {legacy_asset_models}")
    return entries


def _resolve_app_icon_path():
    for candidate in APP_ICON_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _build_pyinstaller_args(
    app_name,
    ffmpeg_bin,
    app_icon_path="",
    onefile=False,
    channel=DEFAULT_CHANNEL,
    skip_ml_model_bundle=False,
):
    import customtkinter

    customtkinter_path = os.path.dirname(customtkinter.__file__)
    args = [
        "main.py",
        f"--name={app_name}",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        f"--add-data={customtkinter_path};customtkinter/",
        f"--add-data={ffmpeg_bin};ffmpeg/bin",
        "--hidden-import=textgrid",
        "--hidden-import=customtkinter",
    ]
    for src, dst in _iter_runtime_data_entries(
        channel=channel,
        skip_ml_model_bundle=skip_ml_model_bundle,
    ):
        args.append(f"--add-data={src};{dst}")
    if app_icon_path:
        args.append(f"--icon={app_icon_path}")
    for module_name in EXCLUDED_MODULES:
        args.append(f"--exclude-module={module_name}")
    return args


def _run_pyinstaller_build(
    app_name,
    ffmpeg_bin,
    app_icon_path="",
    onefile=False,
    channel=DEFAULT_CHANNEL,
    skip_ml_model_bundle=False,
):
    print("[INFO] Loading PyInstaller...")
    import PyInstaller.__main__

    pyinstaller_args = _build_pyinstaller_args(
        app_name=app_name,
        ffmpeg_bin=ffmpeg_bin,
        app_icon_path=app_icon_path,
        onefile=onefile,
        channel=channel,
        skip_ml_model_bundle=skip_ml_model_bundle,
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


def _copy_file_if_exists(src_path, dst_dir):
    if not os.path.isfile(src_path):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, dst_dir)
    return True


def _copy_release_assets_dir(release_dir):
    src = os.path.join(APP_DIR, "release_assets")
    dst = os.path.join(release_dir, "release_assets")
    if not os.path.isdir(src):
        print(f"[WARN] release_assets folder not found: {src}")
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("[INFO] Copied release_assets folder.")


def _mask_sensitive_release_configs(base_dir):
    targets = [
        (
            os.path.join(base_dir, "ml", "configs", "training_data_roots.yaml"),
            MASKED_TRAINING_DATA_ROOTS_YAML,
        ),
        (
            os.path.join(base_dir, "ml", "configs", "dataset_build_default.yaml"),
            MASKED_DATASET_BUILD_DEFAULT_YAML,
        ),
    ]
    for path, content in targets:
        if not os.path.isfile(path):
            print(f"[WARN] Config not found for masking (skip): {path}")
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"[INFO] Masked sensitive config path: {path}")


def _get_release_dir(channel):
    return os.path.join(APP_DIR, f"{RELEASE_DIR_PREFIX}_{channel}")


def _copy_release_outputs(app_name, channel, app_version, built_artifact_path, onefile=False, clean_release_dir=False):
    release_dir = _get_release_dir(channel)
    if clean_release_dir and os.path.isdir(release_dir):
        shutil.rmtree(release_dir)
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
        print(f"[INFO] Copied onefile artifact: {built_artifact_path}")
    else:
        if not os.path.isdir(built_artifact_path):
            raise FileNotFoundError(f"Built app directory not found: {built_artifact_path}")
        target_dir = os.path.join(release_dir, app_name)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(built_artifact_path, target_dir)
        print(f"[INFO] Copied app directory: {built_artifact_path}")
        _write_release_channel_metadata(
            os.path.join(target_dir, "release_channel.json"),
            app_name=app_name,
            app_version=app_version,
            channel=channel,
        )
        _mask_sensitive_release_configs(target_dir)

    for file_name in RELEASE_COPY_FILES:
        src = os.path.join(APP_DIR, file_name)
        copied = _copy_file_if_exists(src, release_dir)
        if copied:
            print(f"[INFO] Copied: {file_name}")
        else:
            print(f"[WARN] Missing file (skip): {file_name}")

    _copy_release_assets_dir(release_dir)
    return release_dir


def main():
    _configure_console_encoding()
    args = _parse_args()

    if args.onefile and not args.allow_unsafe_onefile:
        raise SystemExit(
            "onefile build is disabled by default. Use --allow-unsafe-onefile to continue."
        )
    if args.allow_unsafe_onefile and not args.onefile:
        print("[INFO] --allow-unsafe-onefile is ignored without --onefile.")

    os.chdir(APP_DIR)
    app_version = _detect_app_version()
    mode_text = "onefile" if args.onefile else "onedir"

    print("====================================================")
    print(" UTAU Auto OTO - PyInstaller Beta Build")
    print("====================================================")
    print(f"[INFO] version={app_version}, channel={args.channel}, mode={mode_text}")
    print(f"[INFO] app_name={args.name}")

    if args.skip_deps:
        print("[1/5] Skipping dependency installation (--skip-deps).")
    else:
        print("[1/5] Installing build dependencies...")
        _install_build_dependencies(skip_ml=args.skip_ml_deps)

    print("[2/5] Preparing FFmpeg runtime...")
    ffmpeg_bin = _ensure_ffmpeg_bin()
    app_icon_path = _resolve_app_icon_path()
    if app_icon_path:
        print(f"[INFO] app_icon={app_icon_path}")
    else:
        print("[WARN] App icon not found. Default icon will be used.")

    print("[3/5] Building with PyInstaller...")
    built_artifact = _run_pyinstaller_build(
        args.name,
        ffmpeg_bin,
        app_icon_path=app_icon_path,
        onefile=args.onefile,
        channel=args.channel,
        skip_ml_model_bundle=args.skip_ml_model_bundle,
    )

    print("[4/5] Packaging beta release folder...")
    release_dir = _copy_release_outputs(
        args.name,
        channel=args.channel,
        app_version=app_version,
        built_artifact_path=built_artifact,
        onefile=args.onefile,
        clean_release_dir=args.clean_release_dir,
    )

    print("[5/5] Completed.")
    print(f"[DONE] release_dir={release_dir}")
    print(f"[DONE] installer_script={os.path.join(release_dir, 'setup_mfa_beta.bat')}")


if __name__ == "__main__":
    main()
