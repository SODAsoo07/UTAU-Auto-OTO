from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.model_context.filename import (
    canonicalize_order_tokens,
    edit_distance,
    filename_expected_tokens,
    read_lab_tokens,
)


_DEFAULT_IGNORE_LABELS = {
    "sp",
    "sil",
    "pau",
    "spn",
    "ap",
    "axh",
    "exh",
    "br",
    "bre",
    "breath",
    "vf",
}

@dataclass
class PairRow:
    base: str
    wav_path: str
    lab_path: str


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _iter_pairs(source: str) -> List[PairRow]:
    root = os.path.abspath(source)
    wav_map: Dict[str, str] = {}
    lab_map: Dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            low = name.lower()
            if not (low.endswith(".wav") or low.endswith(".lab")):
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root)
            key = _norm(os.path.splitext(rel)[0])
            if low.endswith(".wav"):
                wav_map[key] = abs_path
            else:
                lab_map[key] = abs_path
    keys = sorted(set(wav_map.keys()) & set(lab_map.keys()))
    rows: List[PairRow] = []
    for key in keys:
        base = os.path.splitext(os.path.basename(wav_map[key]))[0]
        rows.append(PairRow(base=base, wav_path=wav_map[key], lab_path=lab_map[key]))
    return rows


def _write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split wav/lab pairs into filename-match and mismatch manifests for alignment training."
    )
    parser.add_argument("--source", required=True, help="Root folder with wav/lab pairs")
    parser.add_argument("--out-root", default="", help="Output folder")
    parser.add_argument(
        "--ignore-labels",
        default=",".join(sorted(_DEFAULT_IGNORE_LABELS)),
        help="CSV labels ignored in token matching",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
        help="normalized edit distance <= threshold => match",
    )
    parser.add_argument(
        "--strict-threshold",
        type=float,
        default=0.15,
        help="normalized edit distance <= threshold => near_match",
    )
    args = parser.parse_args()

    source = os.path.abspath(str(args.source or "").strip())
    if not os.path.isdir(source):
        raise SystemExit(f"source not found: {source}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        os.path.abspath(str(args.out_root))
        if str(args.out_root or "").strip()
        else os.path.join(os.getcwd(), "dataset_workspace", "filename_lab_split", f"{os.path.basename(source)}_{stamp}")
    )
    os.makedirs(out_root, exist_ok=True)

    ignore_labels = {_norm(x) for x in str(args.ignore_labels or "").split(",") if _norm(x)}
    pairs = _iter_pairs(source)

    all_rows: List[Dict[str, object]] = []
    match_rows: List[Dict[str, object]] = []
    mismatch_rows: List[Dict[str, object]] = []
    boundary_only_rows: List[Dict[str, object]] = []

    for pair in pairs:
        observed_raw = read_lab_tokens(pair.lab_path)
        expected_raw = filename_expected_tokens(pair.base, language="ja")
        observed = canonicalize_order_tokens(observed_raw, ignore_labels=ignore_labels)
        expected = canonicalize_order_tokens(expected_raw, ignore_labels=ignore_labels)

        dist = edit_distance(expected, observed)
        denom = max(1, len(expected), len(observed))
        norm_dist = float(dist) / float(denom)
        gap = abs(len(expected) - len(observed))

        is_match = bool(norm_dist <= float(args.match_threshold))
        if dist == 0 and len(expected) == len(observed):
            level = "exact"
        elif norm_dist <= float(args.strict_threshold):
            level = "near_match"
        elif is_match:
            level = "loose_match"
        else:
            level = "mismatch"

        row: Dict[str, object] = {
            "base": pair.base,
            "wav_path": pair.wav_path,
            "lab_path": pair.lab_path,
            "expected_tokens": expected,
            "observed_tokens": observed,
            "expected_len": int(len(expected)),
            "observed_len": int(len(observed)),
            "edit_distance": int(dist),
            "normalized_distance": float(round(norm_dist, 6)),
            "token_count_gap": int(gap),
            "match_flag": bool(is_match),
            "match_level": level,
            "boundary_only_recommended": True,
        }
        all_rows.append(row)
        boundary_only_rows.append(row)
        if is_match:
            match_rows.append(row)
        else:
            mismatch_rows.append(row)

    all_rows = sorted(all_rows, key=lambda x: (bool(x["match_flag"]), float(x["normalized_distance"])))
    match_rows = sorted(match_rows, key=lambda x: float(x["normalized_distance"]))
    mismatch_rows = sorted(mismatch_rows, key=lambda x: float(x["normalized_distance"]), reverse=True)

    all_path = os.path.join(out_root, "manifest_all.jsonl")
    match_path = os.path.join(out_root, "manifest_match.jsonl")
    mismatch_path = os.path.join(out_root, "manifest_mismatch.jsonl")
    boundary_only_path = os.path.join(out_root, "manifest_boundary_only.jsonl")

    _write_jsonl(all_path, all_rows)
    _write_jsonl(match_path, match_rows)
    _write_jsonl(mismatch_path, mismatch_rows)
    _write_jsonl(boundary_only_path, boundary_only_rows)

    summary = {
        "source": source,
        "out_root": out_root,
        "total_pairs": int(len(all_rows)),
        "match_pairs": int(len(match_rows)),
        "mismatch_pairs": int(len(mismatch_rows)),
        "match_rate": float(round((len(match_rows) / len(all_rows)) if all_rows else 0.0, 6)),
        "ignore_labels": sorted(ignore_labels),
        "match_threshold": float(args.match_threshold),
        "strict_threshold": float(args.strict_threshold),
        "outputs": {
            "all": all_path,
            "match": match_path,
            "mismatch": mismatch_path,
            "boundary_only": boundary_only_path,
        },
    }
    _write_json(os.path.join(out_root, "summary.json"), summary)

    print(f"[DONE] total_pairs={len(all_rows)}")
    print(f"[DONE] match_pairs={len(match_rows)} mismatch_pairs={len(mismatch_rows)}")
    print(f"[DONE] out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
