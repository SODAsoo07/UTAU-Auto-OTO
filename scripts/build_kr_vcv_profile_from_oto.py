from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    pre_q50 = _quantile(pre_vals, 0.50)
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


def build_profile_from_oto(oto_path: str) -> Dict[str, object]:
    rows = parse_oto_rows(oto_path, language="korean")
    agg: Dict[str, Dict[str, List[float]]] = {}
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
        "source_oto": os.path.abspath(oto_path),
        "sample_counts": sample_counts,
        "korean": {"vcv": profile_entries},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build KR VCV anchor profile YAML from reference OTO.")
    ap.add_argument("--oto", required=True, help="Reference OTO path")
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "ml", "configs", "kr_vcv_anchor_profile.yaml"),
        help="Output YAML path",
    )
    args = ap.parse_args()

    payload = build_profile_from_oto(args.oto)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    counts = payload.get("sample_counts", {})
    print("saved:", out_path)
    print("sample_counts:", counts)
    if counts:
        print("median_samples:", statistics.median(counts.values()))


if __name__ == "__main__":
    main()
