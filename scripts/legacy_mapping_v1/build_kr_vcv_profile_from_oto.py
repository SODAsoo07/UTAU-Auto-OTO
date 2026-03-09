from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for this script. Please install pyyaml.") from exc

from core.oto_ml_features import parse_oto_rows
from core.oto_generator import classify_alias


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    data = sorted(float(v) for v in values)
    pos = (len(data) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _seed_entry() -> Dict[str, float]:
    return {
        "pre_window_before_ms": 8.0,
        "pre_window_after_ms": 6.0,
        "pre_floor_ms": 40.0,
        "ovl_gap_min_ms": 10.0,
        "ovl_gap_max_ms": 24.0,
        "ovl_gap_target_ms": 16.0,
        "cons_gap_min_ms": 50.0,
        "cons_gap_max_ms": 150.0,
        "cons_gap_target_ms": 100.0,
        "cut_gap_min_ms": 28.0,
        "cut_gap_max_ms": 120.0,
        "cut_gap_target_ms": 70.0,
        "cut_to_next_onset_allow_ms": 8.0,
        "cut_to_next_vowel_allow_ms": 2.0,
    }


def _build_entry(pre_vals: List[float], cons_gap_vals: List[float], cut_gap_vals: List[float], ovl_gap_vals: List[float], alias_type: str) -> Dict[str, float]:
    out = _seed_entry()
    if not pre_vals:
        return out

    pre_q25 = _quantile(pre_vals, 0.25)
    cons_q25 = _quantile(cons_gap_vals, 0.25)
    cons_q50 = _quantile(cons_gap_vals, 0.50)
    cons_q75 = _quantile(cons_gap_vals, 0.75)
    cut_q25 = _quantile(cut_gap_vals, 0.25)
    cut_q50 = _quantile(cut_gap_vals, 0.50)
    cut_q75 = _quantile(cut_gap_vals, 0.75)
    ovl_q25 = _quantile(ovl_gap_vals, 0.25)
    ovl_q50 = _quantile(ovl_gap_vals, 0.50)
    ovl_q75 = _quantile(ovl_gap_vals, 0.75)

    out["pre_floor_ms"] = round(_clamp(pre_q25 * 0.42, 18.0, 76.0), 2)
    out["ovl_gap_min_ms"] = round(_clamp(ovl_q25 * 0.88, 3.0, 40.0), 2)
    out["ovl_gap_target_ms"] = round(_clamp(ovl_q50, 4.0, 60.0), 2)
    out["ovl_gap_max_ms"] = round(_clamp(ovl_q75 * 1.18, out["ovl_gap_target_ms"] + 2.0, 84.0), 2)

    out["cons_gap_min_ms"] = round(_clamp(cons_q25 * 0.88, 10.0, 220.0), 2)
    out["cons_gap_target_ms"] = round(_clamp(cons_q50, out["cons_gap_min_ms"] + 6.0, 260.0), 2)
    out["cons_gap_max_ms"] = round(_clamp(cons_q75 * 1.18, out["cons_gap_target_ms"] + 8.0, 320.0), 2)

    out["cut_gap_min_ms"] = round(_clamp(cut_q25 * 0.85, 8.0, 240.0), 2)
    out["cut_gap_target_ms"] = round(_clamp(cut_q50, out["cut_gap_min_ms"] + 4.0, 320.0), 2)
    out["cut_gap_max_ms"] = round(_clamp(cut_q75 * 1.20, out["cut_gap_target_ms"] + 8.0, 420.0), 2)

    if alias_type == "vc":
        out["cut_to_next_onset_allow_ms"] = 4.0
        out["cut_to_next_vowel_allow_ms"] = -2.0
    elif alias_type == "vv":
        out["cut_to_next_onset_allow_ms"] = 6.0
        out["cut_to_next_vowel_allow_ms"] = 2.0
    elif alias_type == "vcv":
        out["cut_to_next_onset_allow_ms"] = 8.0
        out["cut_to_next_vowel_allow_ms"] = 2.0
    elif alias_type in {"cv", "cv_head"}:
        out["cut_to_next_onset_allow_ms"] = 7.0
        out["cut_to_next_vowel_allow_ms"] = 4.0

    return out


def _iter_training_roots(roots_yaml: str, language: str, format_type: str) -> List[str]:
    with open(roots_yaml, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    raw = payload.get(str(language).strip().lower(), {}).get(str(format_type).strip().lower(), []) or []
    roots = []
    for item in raw:
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(item).strip())))
        if path and os.path.isdir(path):
            roots.append(path)
    return roots


def _discover_oto_files(roots: Sequence[str], min_rows: int = 8) -> List[str]:
    """
    Recursively discover usable OTOs.
    - include: oto.ini / boto.ini
    - exclude: empty top-level placeholder oto.ini (rows < min_rows)
    """
    candidates: List[Tuple[str, int, int]] = []
    name_priority = {"boto.ini": 0, "oto.ini": 1}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                lowered = filename.lower()
                if lowered not in {"oto.ini", "boto.ini"}:
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    rows = parse_oto_rows(path, language="korean")
                    row_count = len(rows)
                except Exception:
                    continue
                if row_count < int(min_rows):
                    continue
                depth = len(os.path.relpath(path, root).split(os.sep))
                candidates.append((path, name_priority.get(lowered, 9), depth))
    # Prefer boto.ini first; otherwise stable by path depth then name.
    candidates.sort(key=lambda x: (x[1], x[2], x[0].lower()))
    return [p for p, _prio, _depth in candidates]


def _accumulate_rows(rows: Iterable[Dict[str, object]], agg: Dict[str, Dict[str, List[float]]]) -> None:
    for row in rows:
        alias = str(row.get("alias") or "")
        alias_type = str(classify_alias(alias, {})).strip().lower()
        if alias_type not in {"cv", "cv_head", "vc", "vv", "vcv"}:
            continue
        pre = max(float(row.get("pre", 0.0) or 0.0), 0.0)
        cons = max(float(row.get("cons", 0.0) or 0.0), 0.0)
        ovl = max(float(row.get("ovl", 0.0) or 0.0), 0.0)
        cutoff_abs = abs(float(row.get("cutoff", 0.0) or 0.0))
        cons_gap = max(cons - pre, 2.0)
        cut_gap = max(cutoff_abs - cons, 2.0)
        ovl_gap = max(pre - ovl, 0.0)

        bucket = agg.setdefault(alias_type, {"pre": [], "cons_gap": [], "cut_gap": [], "ovl_gap": []})
        bucket["pre"].append(pre)
        bucket["cons_gap"].append(cons_gap)
        bucket["cut_gap"].append(cut_gap)
        bucket["ovl_gap"].append(ovl_gap)


def build_profile_from_otos(oto_paths: Sequence[str]) -> Dict[str, object]:
    agg: Dict[str, Dict[str, List[float]]] = {}
    source_otos: List[str] = []
    skipped_otos: List[str] = []
    for oto_path in oto_paths:
        try:
            rows = parse_oto_rows(oto_path, language="korean")
        except Exception:
            skipped_otos.append(os.path.abspath(oto_path))
            continue
        if not rows:
            skipped_otos.append(os.path.abspath(oto_path))
            continue
        _accumulate_rows(rows, agg)
        source_otos.append(os.path.abspath(oto_path))

    profile_entries = {}
    for alias_type in ("cv", "cv_head", "vc", "vv", "vcv"):
        vals = agg.get(alias_type, {})
        profile_entries[alias_type] = _build_entry(
            vals.get("pre", []),
            vals.get("cons_gap", []),
            vals.get("cut_gap", []),
            vals.get("ovl_gap", []),
            alias_type,
        )

    sample_counts = {k: len(v.get("pre", [])) for k, v in agg.items()}
    return {
        "version": 1,
        "mode": "rhythm_stable",
        "source_otos": source_otos,
        "source_count": len(source_otos),
        "skipped_otos": skipped_otos,
        "skipped_count": len(skipped_otos),
        "sample_counts": sample_counts,
        "korean": {"vcv": profile_entries},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build KR VCV anchor profile YAML from reference OTO.")
    ap.add_argument("--oto", action="append", default=[], help="Reference OTO path (repeatable)")
    ap.add_argument(
        "--roots-yaml",
        default="",
        help="Training roots YAML path. When set, auto-discovers oto.ini/boto.ini under language/format roots.",
    )
    ap.add_argument("--lang", default="korean", help="Language key for roots-yaml discovery (default: korean)")
    ap.add_argument("--format", default="vcv", help="Format key for roots-yaml discovery (default: vcv)")
    ap.add_argument("--min-rows", type=int, default=8, help="Minimum OTO rows to keep discovered files (default: 8)")
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "ml", "configs", "kr_vcv_anchor_profile.yaml"),
        help="Output YAML path",
    )
    args = ap.parse_args()

    oto_inputs: List[str] = [os.path.abspath(p) for p in (args.oto or []) if str(p).strip()]
    if args.roots_yaml:
        roots = _iter_training_roots(args.roots_yaml, args.lang, args.format)
        discovered = _discover_oto_files(roots, min_rows=int(args.min_rows))
        oto_inputs.extend(discovered)

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for p in oto_inputs:
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(os.path.abspath(p))

    if not deduped:
        raise RuntimeError("No OTO inputs found. Use --oto or --roots-yaml.")

    payload = build_profile_from_otos(deduped)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    counts = payload.get("sample_counts", {})
    print("saved:", out_path)
    print("source_count:", payload.get("source_count", 0))
    print("skipped_count:", payload.get("skipped_count", 0))
    print("sample_counts:", counts)
    if counts:
        print("median_samples:", statistics.median(counts.values()))


if __name__ == "__main__":
    main()
