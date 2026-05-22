from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MANIFEST = r"ml_workspace\manual_oto_anchor\manual_oto_anchor_manifest.jsonl"
DEFAULT_OUT_DIR = r"ml_workspace\manual_oto_anchor\splits"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _include_row(row: dict[str, object], *, include_flagged: bool, include_roles: set[str]) -> bool:
    if not bool(row.get("usable_as_target")):
        return False
    quality = str(row.get("label_quality", "") or "")
    if quality == "flagged_manual" and not include_flagged:
        return False
    role = str(row.get("alias_role", "") or "").strip().lower()
    if include_roles and role not in include_roles:
        return False
    if role in {"invalid", "other"}:
        return False
    return True


def split_manual_oto_anchor_manifest(
    *,
    manifest_path: Path,
    out_dir: Path,
    seed: int = 20260522,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    include_flagged: bool = True,
    roles: str = "",
) -> dict[str, object]:
    rows = _load_jsonl(manifest_path)
    include_roles = {item.strip().lower() for item in str(roles or "").split(",") if item.strip()}
    filtered = [row for row in rows if _include_row(row, include_flagged=include_flagged, include_roles=include_roles)]

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        key = "|".join(
            [
                str(row.get("language", "") or ""),
                str(row.get("format_type", "") or ""),
                str(row.get("voicebank_id", "") or ""),
            ]
        )
        grouped[key].append(row)

    bank_keys = sorted(grouped)
    rng = random.Random(int(seed))
    rng.shuffle(bank_keys)
    total = sum(len(grouped[key]) for key in bank_keys)
    target_test = int(round(total * max(0.0, float(test_ratio))))
    target_val = int(round(total * max(0.0, float(val_ratio))))

    assignments: dict[str, str] = {}
    test_count = 0
    val_count = 0
    for key in bank_keys:
        count = len(grouped[key])
        if test_count < target_test:
            assignments[key] = "test"
            test_count += count
        elif val_count < target_val:
            assignments[key] = "val"
            val_count += count
        else:
            assignments[key] = "train"

    split_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        split = assignments[key]
        split_rows[split].extend(grouped[key])

    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / "manual_oto_anchor_filtered.jsonl"
    train_path = out_dir / "manual_oto_anchor_train.jsonl"
    val_path = out_dir / "manual_oto_anchor_val.jsonl"
    test_path = out_dir / "manual_oto_anchor_test.jsonl"
    _write_jsonl(filtered_path, filtered)
    _write_jsonl(train_path, split_rows["train"])
    _write_jsonl(val_path, split_rows["val"])
    _write_jsonl(test_path, split_rows["test"])

    def summarize(rows_for_split: list[dict[str, object]]) -> dict[str, object]:
        roles_counter = Counter(str(row.get("alias_role", "") or "") for row in rows_for_split)
        quality_counter = Counter(str(row.get("label_quality", "") or "") for row in rows_for_split)
        format_role = Counter(
            "|".join(
                [
                    str(row.get("language", "") or ""),
                    str(row.get("format_type", "") or ""),
                    str(row.get("alias_role", "") or ""),
                ]
            )
            for row in rows_for_split
        )
        return {
            "rows": len(rows_for_split),
            "banks": len(
                {
                    "|".join(
                        [
                            str(row.get("language", "") or ""),
                            str(row.get("format_type", "") or ""),
                            str(row.get("voicebank_id", "") or ""),
                        ]
                    )
                    for row in rows_for_split
                }
            ),
            "role_counts": dict(roles_counter),
            "quality_counts": dict(quality_counter),
            "format_role_counts": dict(format_role),
        }

    summary = {
        "source_manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "input_rows": len(rows),
        "filtered_rows": len(filtered),
        "dropped_rows": len(rows) - len(filtered),
        "include_flagged": bool(include_flagged),
        "include_roles": sorted(include_roles),
        "split_paths": {
            "filtered": str(filtered_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "splits": {key: summarize(value) for key, value in split_rows.items()},
    }
    summary_path = out_dir / "manual_oto_anchor_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter and bank-heldout split manual OTO anchor manifest rows.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--include-flagged", action="store_true", default=True)
    parser.add_argument("--exclude-flagged", action="store_false", dest="include_flagged")
    parser.add_argument("--roles", default="", help="Optional comma-separated alias_role allowlist.")
    args = parser.parse_args()
    summary = split_manual_oto_anchor_manifest(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        include_flagged=bool(args.include_flagged),
        roles=str(args.roles or ""),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
