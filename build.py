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


def main():
    _configure_console_encoding()
    os.chdir(APP_DIR)
    print("🚀 [1/5] 빌드 의존성 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "customtkinter", "textgrid", "numpy"])

    print("🚀 [2/5] FFmpeg 바이너리 준비 중...")
    ffmpeg_bin = _ensure_ffmpeg_bin()

    print("🚀 [3/5] PyInstaller 모듈 로딩 중...")
    import PyInstaller.__main__
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    print("🚀 [4/5] UTAU_Auto_OTO.exe 빌드 중...")
    PyInstaller.__main__.run([
        "main.py",
        "--name=UTAU_Auto_OTO",
        "--windowed",
        "--onefile",
        "--noconfirm",
        "--clean",
        f"--add-data={ctk_path};customtkinter/",
        f"--add-data={ffmpeg_bin};ffmpeg/bin",
        "--hidden-import=textgrid",
        "--hidden-import=customtkinter",
    ])

    print("🚀 [5/5] 배포 폴더 구성 중...")
    release_dir = os.path.join(APP_DIR, "UTAU_Auto_OTO_Release")
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    exe_path = os.path.join(APP_DIR, "dist", "UTAU_Auto_OTO.exe")
    if os.path.exists(exe_path):
        shutil.copy(exe_path, release_dir)
        print(f"   -> 복사 완료: {exe_path}")
    else:
        raise FileNotFoundError("빌드 결과 exe를 찾지 못했습니다.")

    setup_path = os.path.join(APP_DIR, "setup_mfa.bat")
    if os.path.exists(setup_path):
        shutil.copy(setup_path, release_dir)
        print("   -> 복사 완료: setup_mfa.bat")

    print(f"\n✅ 빌드 완료: {release_dir}")
    print("ℹ FFmpeg는 exe 내부에 포함되어 별도 설치 없이 배포 가능합니다.")


if __name__ == "__main__":
    main()
