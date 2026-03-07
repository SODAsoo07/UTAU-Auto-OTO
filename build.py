import argparse
import os
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
DEFAULT_APP_NAME = "UTAU_Auto_OTO"
EXCLUDED_MODULES = [
    "torch",
    "torchaudio",
    "torchvision",
    "ml",
]
RUNTIME_DATA_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "assets", "models", "oto_ml"), "assets/models/oto_ml"),
    (os.path.join(APP_DIR, "ml", "configs"), "ml/configs"),
    (os.path.join(APP_DIR, "config.json"), "."),
]
args = argparse.Namespace(onefile=False, allow_unsafe_onefile=False, name=DEFAULT_APP_NAME)


def _configure_console_encoding():
    # GitHub Actions Windows(cp1252) 환경에서 이모지/한글 로그 출력 시 인코딩 오류 방지
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _ensure_ffmpeg_bin():
    ffmpeg_exe = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    ffprobe_exe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")
    if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
        print(f"✅ FFmpeg 바이너리 재사용: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR

    os.makedirs(BUILD_ASSET_DIR, exist_ok=True)
    tmp_zip = os.path.join(BUILD_ASSET_DIR, "ffmpeg_release_essentials.zip")
    tmp_extract = tempfile.mkdtemp(prefix="ffmpeg_extract_", dir=BUILD_ASSET_DIR)
    try:
        print("📦 FFmpeg(Windows shared build) 다운로드 중...")
        with urllib.request.urlopen(FFMPEG_RELEASE_ZIP_URL, timeout=180) as resp:
            with open(tmp_zip, "wb") as f:
                f.write(resp.read())

        print("📦 FFmpeg 압축 해제 중...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract)

        source_bin = ""
        for root, dirs, _ in os.walk(tmp_extract):
            if "bin" in dirs and os.path.exists(os.path.join(root, "bin", "ffmpeg.exe")):
                source_bin = os.path.join(root, "bin")
                break
        if not source_bin:
            raise RuntimeError("압축 내부에서 ffmpeg.exe를 찾지 못했습니다.")

        if os.path.exists(FFMPEG_DIR):
            shutil.rmtree(FFMPEG_DIR)
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        shutil.copytree(source_bin, FFMPEG_BIN_DIR)
        print(f"✅ FFmpeg 준비 완료: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR
    finally:
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
        shutil.rmtree(tmp_extract, ignore_errors=True)


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
        help="PyInstaller app name.",
    )
    return parser.parse_args()


def _iter_runtime_data_entries():
    entries = []
    for src, dst in RUNTIME_DATA_PATHS:
        if os.path.exists(src):
            entries.append((src, dst))
        else:
            print(f"⚠ 런타임 데이터 누락(건너뜀): {src}")
    return entries


def _build_pyinstaller_args(app_name, ffmpeg_bin, onefile=False):
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    args = [
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
    ]
    for src, dst in _iter_runtime_data_entries():
        args.append(f"--add-data={src};{dst}")
    for module_name in EXCLUDED_MODULES:
        args.append(f"--exclude-module={module_name}")
    return args


def _copy_release_outputs(app_name, onefile=False):
    release_dir = os.path.join(APP_DIR, "UTAU_Auto_OTO_Release")
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    if onefile:
        exe_path = os.path.join(APP_DIR, "dist", f"{app_name}.exe")
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"빌드 결과 exe를 찾지 못했습니다: {exe_path}")
        shutil.copy(exe_path, release_dir)
        print(f"   -> 복사 완료: {exe_path}")
    else:
        dist_dir = os.path.join(APP_DIR, "dist", app_name)
        if not os.path.isdir(dist_dir):
            raise FileNotFoundError(f"빌드 결과 폴더를 찾지 못했습니다: {dist_dir}")
        target_dir = os.path.join(release_dir, app_name)
        shutil.copytree(dist_dir, target_dir)
        print(f"   -> 폴더 복사 완료: {dist_dir}")

    setup_path = os.path.join(APP_DIR, "setup_mfa.bat")
    if os.path.exists(setup_path):
        shutil.copy(setup_path, release_dir)
        print("   -> 복사 완료: setup_mfa.bat")
    return release_dir


def _install_build_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    req_candidates = [
        os.path.join(APP_DIR, "requirements.txt"),
        os.path.join(APP_DIR, "requirements-ml.txt"),
    ]
    for req_path in req_candidates:
        if os.path.exists(req_path):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])


def main():
    _configure_console_encoding()
    global args
    parsed_args = _parse_args()
    if parsed_args is not None:
        args = parsed_args
    if args.onefile and not args.allow_unsafe_onefile:
        raise SystemExit(
            "onefile 빌드는 기본 비활성화되어 있습니다. "
            "SOFA/모델 캐시 경로 이슈를 이해한 경우에만 --allow-unsafe-onefile을 함께 사용하세요."
        )
    if args.allow_unsafe_onefile and not args.onefile:
        print("ℹ --allow-unsafe-onefile 은 --onefile 미사용 시 무시됩니다.")

    os.chdir(APP_DIR)
    print("🚀 [1/5] 빌드 의존성 설치 중...")
    _install_build_dependencies()

    print("🚀 [2/5] FFmpeg 바이너리 준비 중...")
    ffmpeg_bin = _ensure_ffmpeg_bin()

    print("🚀 [3/5] PyInstaller 모듈 로딩 중...")
    import PyInstaller.__main__

    mode_text = "onefile" if args.onefile else "onedir"
    print(f"🚀 [4/5] {args.name} 빌드 중... (mode={mode_text}, variant=lightgbm-only)")
    pyinstaller_args = _build_pyinstaller_args(
        app_name=args.name,
        ffmpeg_bin=ffmpeg_bin,
        onefile=args.onefile,
    )
    PyInstaller.__main__.run(pyinstaller_args)

    print("🚀 [5/5] 배포 폴더 구성 중...")
    release_dir = _copy_release_outputs(args.name, onefile=args.onefile)

    print(f"\n✅ 빌드 완료: {release_dir}")
    if args.onefile:
        print("⚠ onefile 배포: 실행 경로/캐시 특성상 SOFA/모델 설치가 불안정할 수 있습니다.")
    else:
        print("ℹ onedir 배포는 시작 속도가 빠르며, 사용자용 배포에 권장됩니다.")
    print("ℹ 이 브랜치 빌드는 LightGBM 전용입니다.")


if __name__ == "__main__":
    main()
