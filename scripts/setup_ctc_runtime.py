from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from typing import Dict, Tuple


def _check_module(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, str(getattr(mod, "__version__", "") or "ok")
    except Exception as exc:
        return False, str(exc)


def _run_pip(args: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[CTC-SETUP] run: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode == 0


def _runtime_report() -> Dict[str, Tuple[bool, str]]:
    report = {
        "torch": _check_module("torch"),
        "torchaudio": _check_module("torchaudio"),
        "textgrid": _check_module("textgrid"),
    }
    return report


def _print_report(report: Dict[str, Tuple[bool, str]]) -> None:
    for name, (ok, detail) in report.items():
        print(f"[CTC-SETUP] {name}: {'OK' if ok else 'MISSING'} ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup/check runtime dependencies for CTC alignment backend.")
    parser.add_argument("--install", action="store_true", help="Install missing dependencies automatically.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Prefer CPU wheels for torch/torchaudio (Windows/Linux CPU setup).",
    )
    args = parser.parse_args()

    report = _runtime_report()
    _print_report(report)

    missing = [name for name, (ok, _detail) in report.items() if not ok]
    if not missing:
        print("[CTC-SETUP] runtime ready.")
        return 0

    if not args.install:
        print("[CTC-SETUP] missing modules detected. Re-run with --install to install automatically.")
        return 1

    ok = True
    if ("torch" in missing) or ("torchaudio" in missing):
        if args.cpu:
            ok = ok and _run_pip(
                [
                    "install",
                    "torch==2.5.1+cpu",
                    "torchaudio==2.5.1+cpu",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
        else:
            ok = ok and _run_pip(["install", "torch", "torchaudio"])

    if "textgrid" in missing:
        ok = ok and _run_pip(["install", "textgrid"])

    if not ok:
        print("[CTC-SETUP] install command failed.")
        return 2

    print("[CTC-SETUP] re-checking...")
    final_report = _runtime_report()
    _print_report(final_report)
    if all(ok for ok, _detail in final_report.values()):
        print("[CTC-SETUP] runtime ready.")
        return 0

    print("[CTC-SETUP] runtime still incomplete.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

