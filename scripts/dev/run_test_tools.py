from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = {
    "alignment.build_folder": REPO_ROOT / "tests/scripts/alignment/build_alignment_test_folder.py",
    "alignment.compare_visual": REPO_ROOT / "tests/scripts/alignment/compare_alignment_visual.py",
    "alignment.visualize_tg_oto": REPO_ROOT / "tests/scripts/alignment/visualize_textgrid_oto_overlay.py",
    "benchmark.sequence_residual": REPO_ROOT / "tests/scripts/benchmark/benchmark_sequence_residual_datasets.py",
    "smoke.sandbox": REPO_ROOT / "tests/scripts/smoke/sandbox_smoke_check.ps1",
    "training.alignment.export_textgrid": REPO_ROOT / "tests/scripts/training/alignment/export_textgrid_to_sinsy_lab.py",
    "training.alignment.preprocess_sinsy": REPO_ROOT / "tests/scripts/training/alignment/preprocess_sinsy_labels_for_sequence_training.py",
    "training.alignment.split_match_manifest": REPO_ROOT / "tests/scripts/training/alignment/split_filename_lab_match_manifest.py",
    "training.alignment.train_profile": REPO_ROOT / "tests/scripts/training/alignment/train_sequence_aligner_profile_from_sinsy.py",
    "training.data.auto_classify": REPO_ROOT / "tests/scripts/training/data/auto_classify_training_data.py",
    "training.data.generate_shared_profiles": REPO_ROOT / "tests/scripts/training/data/generate_shared_oto_profiles.py",
    "training.coupled.train_ensemble_bundle": REPO_ROOT / "tests/scripts/training/coupled/train_ensemble_bundle.py",
    "training.sequence.build_residual_dataset": REPO_ROOT / "tests/scripts/training/sequence/build_sequence_residual_dataset.py",
    "training.sequence.preprocess_oto_cv": REPO_ROOT / "tests/scripts/training/sequence/preprocess_oto_cv_for_sequence_training.py",
    "training.sequence.train_residual_baseline": REPO_ROOT / "tests/scripts/training/sequence/train_sequence_residual_baseline.py",
    "evaluate.coupled.export_onnx": REPO_ROOT / "tests/scripts/evaluate/coupled/export_coupled_onnx.py",
}


def _resolve_tool(name: str) -> Path:
    key = str(name or "").strip()
    if key not in TOOLS:
        available = ", ".join(sorted(TOOLS))
        raise SystemExit(f"Unknown tool: {key}\nAvailable: {available}")
    path = TOOLS[key]
    if not path.is_file():
        raise SystemExit(f"Tool script not found: {path}")
    return path


def _build_command(script_path: Path, script_args: list[str]) -> list[str]:
    suffix = script_path.suffix.lower()
    if suffix == ".py":
        return [sys.executable, str(script_path), *script_args]
    if suffix == ".ps1":
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            *script_args,
        ]
    raise SystemExit(f"Unsupported script type: {script_path}")


def _cmd_list() -> int:
    print("Available test tools:")
    for key in sorted(TOOLS):
        rel = TOOLS[key].relative_to(REPO_ROOT).as_posix()
        print(f"  - {key}: {rel}")
    return 0


def _cmd_path(tool: str) -> int:
    path = _resolve_tool(tool)
    print(path)
    return 0


def _cmd_run(tool: str, script_args: list[str]) -> int:
    script_path = _resolve_tool(tool)
    cmd = _build_command(script_path, script_args)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified runner for tests/scripts utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available test tools.")

    path_parser = subparsers.add_parser("path", help="Print absolute path for one test tool.")
    path_parser.add_argument("tool", help="Tool key. Use 'list' to inspect keys.")

    run_parser = subparsers.add_parser("run", help="Run one test tool.")
    run_parser.add_argument("tool", help="Tool key. Use 'list' to inspect keys.")
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the tool script.")

    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "path":
        return _cmd_path(args.tool)
    if args.command == "run":
        forwarded = list(args.args or [])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return _cmd_run(args.tool, forwarded)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
