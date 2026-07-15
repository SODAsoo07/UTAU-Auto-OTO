"""Classify row-level Korean VCV OTO improvements and regressions.

Both input directories must be outputs from ``evaluate_korean_vcv_hsmm.py``.
The classifier joins rows against the same staged manual OTO, excludes manual
zero anchors, and records filename structure, alias role, target-index changes,
error direction, and provenance warnings for practical fallback design.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dev.diff_oto import build_report


def _load_targets(directory: Path) -> dict[str, dict]:
    payload = json.loads((directory / "aggregate.json").read_text(encoding="utf-8-sig"))
    return {str(row["relative_oto"]): row for row in payload["targets"]}


def _provenance_by_key(path: Path) -> dict[tuple[str, str, int], dict]:
    rows: dict[tuple[str, str, int], dict] = {}
    occurrences: Counter[tuple[str, str]] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        base_key = (str(row.get("wav", "")).lower(), str(row.get("alias", "")))
        occurrence = occurrences[base_key]
        occurrences[base_key] += 1
        rows[(*base_key, occurrence)] = row
    return rows


def _matched_rows(gold: Path, generated: Path) -> dict[tuple[str, str, int], dict]:
    _report, rows = build_report(
        gold_path=str(gold),
        pred_path=str(generated),
        cutoff_end_mode="compatible",
        wav_dir=str(gold.parent),
    )
    return {
        (str(row["wav"]).lower(), str(row["alias"]), int(row["occurrence"])): row
        for row in rows
        if not all(
            abs(float(row["gold"][name])) <= 1e-3
            for name in ("offset", "preutterance", "overlap", "consonant")
        )
    }


def _filename_structure(wav: str) -> str:
    if "+" in wav:
        return "plus"
    if "'" in wav:
        return "apostrophe"
    return "continuous"


def _outcome(baseline_error: float, candidate_error: float) -> str:
    if baseline_error <= 500.0 < candidate_error:
        return "cross_bad_500"
    if candidate_error <= 500.0 < baseline_error:
        return "cross_good_500"
    if candidate_error < baseline_error - 1e-6:
        return "improved"
    if candidate_error > baseline_error + 1e-6:
        return "worsened"
    return "same"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "bank",
        "relative_oto",
        "wav",
        "alias",
        "occurrence",
        "filename_structure",
        "alias_role",
        "outcome",
        "baseline_error_ms",
        "candidate_error_ms",
        "error_change_ms",
        "candidate_direction",
        "baseline_target_phone_index",
        "candidate_target_phone_index",
        "target_changed",
        "candidate_warnings",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "dataset_staged" / "korean" / "vcv",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    out_dir = (args.out_dir or (candidate_dir / "regression_analysis")).resolve()
    baseline_targets = _load_targets(baseline_dir)
    candidate_targets = _load_targets(candidate_dir)
    relative_paths = sorted(set(baseline_targets) & set(candidate_targets))
    if not relative_paths:
        raise SystemExit("No common relative_oto targets were found.")

    classified: list[dict] = []
    for index, relative_oto in enumerate(relative_paths, start=1):
        candidate = candidate_targets[relative_oto]
        baseline = baseline_targets[relative_oto]
        print(f"[{index}/{len(relative_paths)}] {relative_oto}", flush=True)
        gold = dataset_root / relative_oto
        baseline_artifact = Path(str(baseline["artifact_dir"]))
        candidate_artifact = Path(str(candidate["artifact_dir"]))
        baseline_rows = _matched_rows(gold, baseline_artifact / "hsmm.generated.ini")
        candidate_rows = _matched_rows(gold, candidate_artifact / "hsmm.generated.ini")
        baseline_provenance = _provenance_by_key(baseline_artifact / "hsmm.row_provenance.jsonl")
        candidate_provenance = _provenance_by_key(candidate_artifact / "hsmm.row_provenance.jsonl")
        if baseline_rows.keys() != candidate_rows.keys():
            raise RuntimeError(f"matched row keys differ for {relative_oto}")

        for key, candidate_row in candidate_rows.items():
            baseline_row = baseline_rows[key]
            baseline_error = abs(float(baseline_row["delta"]["offset"]))
            candidate_delta = float(candidate_row["delta"]["offset"])
            candidate_error = abs(candidate_delta)
            baseline_prov = baseline_provenance.get(key, {})
            candidate_prov = candidate_provenance.get(key, {})
            baseline_target = (baseline_prov.get("row_plan") or {}).get("expected_phone_index")
            candidate_target = (candidate_prov.get("row_plan") or {}).get("expected_phone_index")
            classified.append(
                {
                    "bank": candidate["bank"],
                    "relative_oto": relative_oto,
                    "wav": key[0],
                    "alias": key[1],
                    "occurrence": key[2],
                    "filename_structure": _filename_structure(key[0]),
                    "alias_role": candidate_prov.get("role", "unknown"),
                    "outcome": _outcome(baseline_error, candidate_error),
                    "baseline_error_ms": round(baseline_error, 3),
                    "candidate_error_ms": round(candidate_error, 3),
                    "error_change_ms": round(candidate_error - baseline_error, 3),
                    "candidate_direction": "late" if candidate_delta > 0.0 else "early",
                    "baseline_target_phone_index": baseline_target,
                    "candidate_target_phone_index": candidate_target,
                    "target_changed": baseline_target != candidate_target,
                    "candidate_warnings": "|".join(str(value) for value in candidate_prov.get("warnings", [])),
                }
            )

    group_counts: defaultdict[tuple[str, str, str, bool], Counter[str]] = defaultdict(Counter)
    bank_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in classified:
        group_key = (
            str(row["filename_structure"]),
            str(row["alias_role"]),
            str(row["candidate_direction"]),
            bool(row["target_changed"]),
        )
        group_counts[group_key][str(row["outcome"])] += 1
        bank_counts[str(row["bank"])][str(row["outcome"])] += 1

    regression_rows = [row for row in classified if row["outcome"] == "cross_bad_500"]
    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "trusted_rows": len(classified),
        "outcomes": dict(Counter(str(row["outcome"]) for row in classified)),
        "banks": {bank: dict(counts) for bank, counts in sorted(bank_counts.items())},
        "groups": [
            {
                "filename_structure": key[0],
                "alias_role": key[1],
                "candidate_direction": key[2],
                "target_changed": key[3],
                **dict(counts),
            }
            for key, counts in sorted(group_counts.items())
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "classified_rows.csv", classified)
    _write_csv(out_dir / "cross_bad_500_rows.csv", regression_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
