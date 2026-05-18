from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.ja_lab_generator import parse_ja_filename, split_ja_romaji_syllable


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

_ROMAJI_CANON = {
    "shi": "sh",
    "si": "sh",
    "chi": "ch",
    "ti": "ch",
    "ji": "j",
    "zi": "j",
    "tsu": "ts",
    "tu": "ts",
    "fu": "f",
    "hu": "f",
    "nn": "n",
    "xn": "n",
    "ltsu": "q",
    "ltu": "q",
    "xtsu": "q",
    "xtu": "q",
    "cl": "q",
}


@dataclass
class PairRow:
    base: str
    wav_path: str
    lab_path: str


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _read_lab_tokens(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            line = str(raw or "").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            label = _norm(" ".join(parts[2:]))
            if label:
                out.append(label)
    return out


def _explode_onset(onset: str) -> List[str]:
    o = _norm(onset)
    if not o:
        return []
    split_map = {
        "ky": ["k", "y"],
        "gy": ["g", "y"],
        "ny": ["n", "y"],
        "hy": ["h", "y"],
        "my": ["m", "y"],
        "ry": ["r", "y"],
        "by": ["b", "y"],
        "py": ["p", "y"],
        "dy": ["d", "y"],
        "ty": ["t", "y"],
    }
    if o in split_map:
        return list(split_map[o])
    return [_ROMAJI_CANON.get(o, o)]


def _filename_expected_tokens(base: str) -> List[str]:
    tokens: List[str] = []
    for syllable in parse_ja_filename(base):
        s = _norm(syllable)
        if not s or s in {"r", "br", "bre", "breath"}:
            continue
        onset, vowel = split_ja_romaji_syllable(s)
        onset = _norm(onset)
        vowel = _norm(vowel)
        if onset:
            tokens.extend(_explode_onset(onset))
        if vowel:
            tokens.append(_ROMAJI_CANON.get(vowel, vowel))
        if (not onset) and (not vowel):
            tokens.append(_ROMAJI_CANON.get(s, s))
    return [t for t in tokens if t]


def _canonicalize_tokens(tokens: Iterable[str], *, ignore_labels: set[str]) -> List[str]:
    out: List[str] = []
    for raw in tokens:
        t = _norm(raw)
        if (not t) or (t in ignore_labels):
            continue
        c = _ROMAJI_CANON.get(t, t)
        if c in ignore_labels:
            continue
        out.append(c)
    return out


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return int(m)
    if m == 0:
        return int(n)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            cost = 0 if ai == bj else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[n][m])


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
        observed_raw = _read_lab_tokens(pair.lab_path)
        expected_raw = _filename_expected_tokens(pair.base)
        observed = _canonicalize_tokens(observed_raw, ignore_labels=ignore_labels)
        expected = _canonicalize_tokens(expected_raw, ignore_labels=ignore_labels)

        dist = _edit_distance(expected, observed)
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
