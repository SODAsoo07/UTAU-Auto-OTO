"""Evaluate Korean VCV HSMM OTO generation against staged manual OTO files.

The script deliberately lives under ``scripts/evaluate``: it invokes the
runtime generator as a subprocess, but no runtime module imports evaluation
code. Results are checkpointed after every voicebank folder so a long run can
be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from scripts.dev.diff_oto import build_report


def _oto_row_count(path: Path) -> int:
    return sum(1 for line in read_text_with_fallback(path).splitlines() if parse_oto_line(line))


def discover_targets(
    dataset_root: Path,
    *,
    min_wavs: int,
    min_rows: int,
    include_banks: set[str],
) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for oto_path in sorted(dataset_root.rglob("oto.ini"), key=lambda path: str(path).lower()):
        relative = oto_path.relative_to(dataset_root)
        bank = relative.parts[0]
        if include_banks and bank not in include_banks:
            continue
        wav_count = sum(1 for _path in oto_path.parent.glob("*.wav"))
        row_count = _oto_row_count(oto_path)
        if wav_count < min_wavs or row_count < min_rows:
            continue
        targets.append(
            {
                "bank": bank,
                "relative_oto": relative.as_posix(),
                "oto_path": oto_path,
                "wav_dir": oto_path.parent,
                "wav_count": wav_count,
                "row_count": row_count,
            }
        )
    return targets


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_.-")
    return cleaned or "voicebank"


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(value) for value in values]
    if not vals:
        return {"n": 0, "MAE_ms": 0.0, "abs_p90_ms": 0.0}
    absolute = sorted(abs(value) for value in vals)
    return {
        "n": len(vals),
        "MAE_ms": round(sum(absolute) / len(absolute), 1),
        "abs_p90_ms": round(absolute[int(0.9 * (len(absolute) - 1))], 1),
    }


def _absolute_landmark_metrics(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    def delta(row: dict, parameter: str) -> float:
        gold = row["gold"]
        pred = row["pred"]
        return (float(pred["offset"]) + float(pred[parameter])) - (
            float(gold["offset"]) + float(gold[parameter])
        )

    return {
        "pre_abs": _stats(delta(row, "preutterance") for row in rows),
        "overlap_abs": _stats(delta(row, "overlap") for row in rows),
        "consonant_abs": _stats(delta(row, "consonant") for row in rows),
    }


def _write_results(out_dir: Path, results: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate.json"
    json_path.write_text(json.dumps({"targets": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = (
        "bank",
        "relative_oto",
        "wav_count",
        "rows_matched",
        "only_in_gold",
        "only_in_pred",
        "offset_mae_ms",
        "offset_p90_ms",
        "pre_abs_mae_ms",
        "consonant_abs_mae_ms",
        "overlap_abs_mae_ms",
        "cutoff_end_abs_mae_ms",
        "offset_over_500ms",
        "offset_over_1000ms",
        "manual_zero_anchor_rows",
        "trusted_rows",
        "trusted_offset_mae_ms",
        "trusted_offset_p90_ms",
        "trusted_pre_abs_mae_ms",
        "trusted_consonant_abs_mae_ms",
        "trusted_overlap_abs_mae_ms",
        "trusted_offset_over_500ms",
        "trusted_offset_over_1000ms",
    )
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _evaluate_target(target: dict[str, object], out_dir: Path, *, resume: bool) -> dict:
    relative_oto = str(target["relative_oto"])
    target_dir = out_dir / _slug(relative_oto.removesuffix("/oto.ini"))
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = "hsmm"
    generated_path = target_dir / f"{stem}.generated.ini"
    log_path = target_dir / "generate.log"

    if not (resume and generated_path.is_file()):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "dev" / "mfa_free_oto_review.py"),
            "generate",
            "--wav-dir",
            str(target["wav_dir"]),
            "--out-dir",
            str(target_dir),
            "--name",
            stem,
            "--template-oto",
            str(target["oto_path"]),
            "--language",
            "korean",
            "--format-type",
            "vcv",
            "--alias-type",
            "auto",
            "--encoder",
            "acoustic",
            "--use-hsmm-decoder",
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        log_path.write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"generation failed for {relative_oto}; see {log_path}")

    report, rows = build_report(
        gold_path=str(target["oto_path"]),
        pred_path=str(generated_path),
        cutoff_end_mode="compatible",
        wav_dir=str(target["wav_dir"]),
    )
    landmarks = _absolute_landmark_metrics(rows)
    offset_residuals = [abs(float(row["delta"]["offset"])) for row in rows]
    def has_zero_anchor(row: dict) -> bool:
        return all(
            abs(float(row["gold"][key])) <= 1e-3
            for key in ("offset", "preutterance", "overlap", "consonant")
        )

    trusted_rows = [row for row in rows if not has_zero_anchor(row)]
    trusted_offset_residuals = [abs(float(row["delta"]["offset"])) for row in trusted_rows]
    trusted_offset = _stats(float(row["delta"]["offset"]) for row in trusted_rows)
    trusted_landmarks = _absolute_landmark_metrics(trusted_rows)
    manual_zero_anchor_rows = len(rows) - len(trusted_rows)
    result = {
        "bank": target["bank"],
        "relative_oto": relative_oto,
        "wav_count": target["wav_count"],
        "rows_gold": report["rows_gold"],
        "rows_pred": report["rows_pred"],
        "rows_matched": report["rows_matched"],
        "only_in_gold": report["only_in_gold"],
        "only_in_pred": report["only_in_pred"],
        "offset_mae_ms": report["by_param"]["offset"]["MAE_ms"],
        "offset_p90_ms": report["by_param"]["offset"]["abs_p90_ms"],
        "pre_abs_mae_ms": landmarks["pre_abs"]["MAE_ms"],
        "consonant_abs_mae_ms": landmarks["consonant_abs"]["MAE_ms"],
        "overlap_abs_mae_ms": landmarks["overlap_abs"]["MAE_ms"],
        "cutoff_end_abs_mae_ms": report["by_param"]["cutoff_end_abs"]["MAE_ms"],
        "offset_over_500ms": sum(value > 500.0 for value in offset_residuals),
        "offset_over_1000ms": sum(value > 1000.0 for value in offset_residuals),
        "manual_zero_anchor_rows": manual_zero_anchor_rows,
        "trusted_rows": len(trusted_rows),
        "trusted_offset_mae_ms": trusted_offset["MAE_ms"],
        "trusted_offset_p90_ms": trusted_offset["abs_p90_ms"],
        "trusted_pre_abs_mae_ms": trusted_landmarks["pre_abs"]["MAE_ms"],
        "trusted_consonant_abs_mae_ms": trusted_landmarks["consonant_abs"]["MAE_ms"],
        "trusted_overlap_abs_mae_ms": trusted_landmarks["overlap_abs"]["MAE_ms"],
        "trusted_offset_over_500ms": sum(value > 500.0 for value in trusted_offset_residuals),
        "trusted_offset_over_1000ms": sum(value > 1000.0 for value in trusted_offset_residuals),
        "artifact_dir": str(target_dir.resolve()),
    }
    (target_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    worst_rows = sorted(rows, key=lambda row: abs(float(row["delta"]["offset"])), reverse=True)[:100]
    (target_dir / "worst_offset_rows.json").write_text(
        json.dumps(worst_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    trusted_worst_rows = sorted(
        trusted_rows,
        key=lambda row: abs(float(row["delta"]["offset"])),
        reverse=True,
    )[:100]
    (target_dir / "trusted_worst_offset_rows.json").write_text(
        json.dumps(trusted_worst_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    # Windows launchers can inherit a legacy console codec (for example cp932)
    # even when dataset paths contain Korean or corrupted legacy bank names.
    # Progress output must never abort an otherwise valid evaluation run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset_staged" / "korean" / "vcv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-wavs", type=int, default=100)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--include-bank", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    targets = discover_targets(
        dataset_root,
        min_wavs=max(1, int(args.min_wavs)),
        min_rows=max(1, int(args.min_rows)),
        include_banks={str(value) for value in args.include_bank},
    )
    if not targets:
        raise SystemExit("No Korean VCV targets matched the requested filters.")

    results: list[dict] = []
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target['relative_oto']}", flush=True)
        results.append(_evaluate_target(target, out_dir, resume=bool(args.resume)))
        _write_results(out_dir, results)
    print(f"Wrote {out_dir / 'aggregate.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
