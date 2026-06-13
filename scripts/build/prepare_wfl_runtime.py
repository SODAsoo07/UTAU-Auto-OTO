from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PYTHON_VERSION = "3.11.9"
PYTHON_ARCHIVE = f"python-{PYTHON_VERSION}-embed-amd64.zip"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{PYTHON_ARCHIVE}"
TORCH_WHEEL = (
    "https://download.pytorch.org/whl/cpu/"
    "torch-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl"
)
TORCHAUDIO_WHEEL = (
    "https://download.pytorch.org/whl/cpu/"
    "torchaudio-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl"
)


def _default_poc_root(repo_root: Path) -> Path:
    configured = os.environ.get("UTOA_WFL_POC_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (repo_root.parent.parent / "WFL_PoC").resolve()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    print(f"[WFL bundle] downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "AutoOTO-WFL-Build"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[WFL bundle]", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _copy_runtime_source(source_repo: Path, target_repo: Path) -> None:
    required = ("infer.py", "model.py", "utils.py")
    missing = [name for name in required if not (source_repo / name).is_file()]
    if missing:
        raise FileNotFoundError(f"WFL source files missing under {source_repo}: {missing}")
    target_repo.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source_repo / name, target_repo / name)
    source_encoder = source_repo / "encoder"
    if not source_encoder.is_dir():
        raise FileNotFoundError(f"WFL Whisper encoder missing: {source_encoder}")
    shutil.copytree(source_encoder, target_repo / "encoder", dirs_exist_ok=True)


def _copy_model(source: Path, destination: Path) -> None:
    required = ("config.yaml", "last.ckpt", "phonemes.txt", "langs.txt", "lang_phonemes.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"WFL model files missing under {source}: {missing}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in required:
        shutil.copy2(source / name, destination / name)


def _configure_embedded_python_path(python_root: Path) -> None:
    pth_path = python_root / "python311._pth"
    pth_text = pth_path.read_text(encoding="utf-8")
    pth_text = pth_text.replace("#import site", "import site")
    if "Lib/site-packages" not in pth_text:
        pth_text += "\nLib/site-packages\n"
    if "../WFL-ASR" not in pth_text:
        pth_text += "../WFL-ASR\n"
    pth_path.write_text(pth_text, encoding="utf-8")


def _prepare_python(runtime_root: Path, cache_root: Path, *, force: bool) -> None:
    python_root = runtime_root / "python"
    python_exe = python_root / "python.exe"
    marker = python_root / ".autooto_wfl_ready"
    if marker.is_file() and python_exe.is_file() and not force:
        _configure_embedded_python_path(python_root)
        return
    if python_root.exists():
        shutil.rmtree(python_root)
    archive = cache_root / PYTHON_ARCHIVE
    _download(PYTHON_URL, archive)
    python_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(python_root)
    _configure_embedded_python_path(python_root)

    get_pip = cache_root / "get-pip.py"
    _download("https://bootstrap.pypa.io/get-pip.py", get_pip)
    _run([str(python_exe), str(get_pip), "--no-warn-script-location"])
    _run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "numpy<2",
            "transformers==4.41.2",
            "PyYAML==6.0.2",
            "click==8.1.8",
            "soundfile==0.13.1",
            "matplotlib==3.9.4",
        ]
    )
    _run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            TORCH_WHEEL,
            TORCHAUDIO_WHEEL,
        ]
    )
    _run(
        [
            str(python_exe),
            "-c",
            "import torch, torchaudio, transformers, yaml, click, soundfile, numpy, matplotlib",
        ]
    )
    marker.write_text(f"python={PYTHON_VERSION}\ntorch=2.2.2\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the portable WFL inference runtime.")
    parser.add_argument("--output", default="")
    parser.add_argument("--poc-root", default="")
    parser.add_argument("--force-python", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve() if args.output else repo_root / "build_assets" / "wfl_runtime"
    poc_root = Path(args.poc_root).resolve() if args.poc_root else _default_poc_root(repo_root)
    source_repo = poc_root / "WFL-ASR"
    source_ja = poc_root / "model_jp" / "japanese"
    source_ko = poc_root / "model_ko" / "korean"

    output.mkdir(parents=True, exist_ok=True)
    _prepare_python(output, repo_root / "build_assets" / "downloads", force=args.force_python)
    _copy_runtime_source(source_repo, output / "WFL-ASR")
    _copy_model(source_ja, output / "models" / "japanese")
    _copy_model(source_ko, output / "models" / "korean")
    print(f"[WFL bundle] ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
