"""Package the WFL runtime into a .7z archive for distribution.

Usage:
    python scripts/build/package_wfl_runtime.py [--output path/to/wfl_runtime.7z]

Prerequisites:
    - build_assets/wfl_runtime/ must already be prepared (run prepare_wfl_runtime.py first)
    - 7-Zip must be installed

The output archive contains python/ and WFL-ASR/ (no models — those are
downloaded separately via the app UI).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_7zip() -> str:
    env = os.environ.get("UTOA_7ZIP_PATH", "").strip()
    if env and os.path.isfile(env):
        return env
    for name in ("7z", "7za"):
        found = shutil.which(name)
        if found:
            return found
    for cand in (
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if os.path.isfile(cand):
            return cand
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Package WFL runtime as .7z for distribution.")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-models", action="store_true",
                        help="Include models/ in the archive (not recommended for size)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runtime_dir = repo_root / "build_assets" / "wfl_runtime"

    if not (runtime_dir / "python" / "python.exe").is_file():
        print(f"[ERROR] WFL runtime not prepared: {runtime_dir}")
        print("        Run prepare_wfl_runtime.py first.")
        return 1

    seven_zip = _find_7zip()
    if not seven_zip:
        print("[ERROR] 7-Zip not found. Install 7-Zip or set UTOA_7ZIP_PATH.")
        return 1

    output = Path(args.output) if args.output else repo_root / "dist" / "wfl_runtime.7z"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    includes = ["python", "WFL-ASR"]
    if args.include_models:
        includes.append("models")

    # Verify directories exist
    for d in includes:
        p = runtime_dir / d
        if not p.is_dir():
            print(f"[ERROR] Expected directory not found: {p}")
            return 1

    cmd = [seven_zip, "a", "-t7z", "-mx=5", "-mmt=on", str(output)] + includes
    print(f"[INFO] cwd={runtime_dir}")
    print(f"[INFO] {subprocess.list2cmdline(cmd)}")
    result = subprocess.run(cmd, cwd=str(runtime_dir))
    if result.returncode != 0:
        print(f"[ERROR] 7-Zip exited with code {result.returncode}")
        return 1

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"[DONE] {output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
