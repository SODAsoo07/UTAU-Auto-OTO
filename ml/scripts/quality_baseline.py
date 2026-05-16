"""
OTO-ML Quality Baseline — PR-01 quality measurement script.

Measures three categories of quality signal on CVVC/CVC OTO files:

1. **Cutoff violation rate** (per alias type + aggregate CI gate)
   - Violation: consonant marker >= abs(cutoff), i.e. cons extends past the sample end
   - Reported per alias type; aggregate gate threshold is 10%

2. **Weighted deviation score** (VC rows, by coda type)
   - Reference margins: stop=16ms, nasal=26ms, liquid=42ms, other=28ms
   - Deviation = max(0, reference_margin - cut_gap)
   - Soft scoring: score how far each row falls short of the margin
   - Not a hard binary gate — just a quality heatmap

3. **Voiceless tight-rate** (CV/VC rows)
   - Tight = (cons - pre) < 18ms, meaning the consonant window is too short
   - If --feature-json is provided, filtered to rows where voiced_conf >= 0.58
   - Without feature data: all CV/VC rows are included

Usage examples:
    # Measure a single OTO file
    python ml/scripts/quality_baseline.py --oto path/to/oto.ini --language korean --format-type cvvc

    # Compare ML-refined vs rule baseline
    python ml/scripts/quality_baseline.py \\
        --oto ml_refined/oto.ini --baseline rule_only/oto.ini \\
        --language korean --format-type cvvc --report report.json

    # Supply feature cache for accurate tight-rate
    python ml/scripts/quality_baseline.py \\
        --oto output/oto.ini --feature-json feature_cache.json \\
        --language korean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import classify_alias_type, parse_oto_rows
from core.runtime.runtime_encoding import bootstrap_utf8_runtime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Consonant-type reference margins for cut_gap (abs(cutoff) - cons).
# Used as soft deviation reference, not hard gates.
_CODA_MARGINS: Dict[str, float] = {
    "stop": 16.0,
    "nasal": 26.0,
    "liquid": 42.0,
    "other": 28.0,
    "none": 28.0,
}

# Alias types that are subject to cutoff violation measurement
_CUTOFF_TARGET_ALIAS_TYPES = {"cv", "cv_head", "vc", "vv", "mono", "vcv"}

# Alias types for voiceless tight-rate (cons - pre gap)
_TIGHT_RATE_ALIAS_TYPES = {"cv", "cv_head", "vc", "vv"}

# Minimum cons-pre gap below which a row is considered "tight"
_TIGHT_THRESHOLD_MS = 18.0

# Voiced confidence threshold for tight-rate filtering (when feature data is available)
_VOICED_CONF_THRESHOLD = 0.58

# Aggregate cutoff violation rate that triggers the CI gate
_GATE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Alias / coda extraction helpers
# ---------------------------------------------------------------------------

def _classify_row(language: str, row: Dict[str, object]) -> Tuple[str, str]:
    """Return (alias_type, coda_type) for a parsed OTO row."""
    alias = str(row.get("alias", "") or "")
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    if not alias_type:
        alias_type = classify_alias_type(language, alias)
    coda_type = _infer_coda_type(language, alias, alias_type)
    return alias_type, coda_type


def _infer_coda_type(language: str, alias: str, alias_type: str) -> str:
    """Infer coda consonant class from an alias string without full feature extraction."""
    if language == "japanese":
        # Japanese CVVC VC entries: "a k", "i t", "u p" etc. — last token is the consonant
        parts = alias.strip().split()
        if len(parts) >= 2 and alias_type in {"vc", "vv"}:
            cons = parts[-1].lower().strip("-").strip("_")
            if cons in {"k", "g", "t", "d", "p", "b"}:
                return "stop"
            if cons in {"m", "n", "ng", "ny"}:
                return "nasal"
            if cons in {"r", "l"}:
                return "liquid"
        return "other"
    else:
        # Korean: extract coda from alias syllable structure
        try:
            from core.oto_ml_features import _extract_kr_structure
            info = _extract_kr_structure(alias, alias_type)
            return str(info.get("coda_type", "other") or "other")
        except Exception:
            pass
        return "other"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _cut_gap(row: Dict[str, object]) -> float:
    """Return abs(cutoff) - cons (positive = room; negative = violation)."""
    cutoff = float(row.get("cutoff", 0.0))
    cons = float(row.get("cons", 0.0))
    return abs(cutoff) - cons


def _cons_pre_gap(row: Dict[str, object]) -> float:
    """Return cons - pre (phonetic window width)."""
    cons = float(row.get("cons", 0.0))
    pre = float(row.get("pre", 0.0))
    return cons - pre


class MetricsAccumulator:
    """Accumulates per-row quality measurements."""

    def __init__(self) -> None:
        # Cutoff violation tracking
        self.cutoff_total: Dict[str, int] = {}
        self.cutoff_violations: Dict[str, int] = {}
        # Weighted deviation (VC rows, by coda type)
        self.deviation_total: Dict[str, int] = {}
        self.deviation_sum: Dict[str, float] = {}
        # Tight-rate tracking
        self.tight_eligible: int = 0
        self.tight_count: int = 0

    def add_row(
        self,
        alias_type: str,
        coda_type: str,
        gap: float,
        cons_pre_gap: float,
        voiced_conf: Optional[float],
    ) -> None:
        if alias_type in _CUTOFF_TARGET_ALIAS_TYPES:
            self.cutoff_total[alias_type] = self.cutoff_total.get(alias_type, 0) + 1
            if gap <= 0.0:
                self.cutoff_violations[alias_type] = self.cutoff_violations.get(alias_type, 0) + 1

        if alias_type in {"vc", "vv"}:
            margin = _CODA_MARGINS.get(coda_type, _CODA_MARGINS["other"])
            deviation = max(0.0, margin - gap)
            self.deviation_total[coda_type] = self.deviation_total.get(coda_type, 0) + 1
            self.deviation_sum[coda_type] = self.deviation_sum.get(coda_type, 0.0) + deviation

        if alias_type in _TIGHT_RATE_ALIAS_TYPES:
            # Filter by voiced_conf if data is available
            if voiced_conf is None or voiced_conf >= _VOICED_CONF_THRESHOLD:
                self.tight_eligible += 1
                if cons_pre_gap < _TIGHT_THRESHOLD_MS:
                    self.tight_count += 1

    def cutoff_violation_rate(self, alias_type: str) -> Optional[float]:
        total = self.cutoff_total.get(alias_type, 0)
        if total == 0:
            return None
        return self.cutoff_violations.get(alias_type, 0) / total

    def aggregate_cutoff_violation_rate(self) -> Optional[float]:
        total = sum(self.cutoff_total.values())
        violations = sum(self.cutoff_violations.values())
        if total == 0:
            return None
        return violations / total

    def mean_deviation(self, coda_type: str) -> Optional[float]:
        n = self.deviation_total.get(coda_type, 0)
        if n == 0:
            return None
        return self.deviation_sum[coda_type] / n

    def tight_rate(self) -> Optional[float]:
        if self.tight_eligible == 0:
            return None
        return self.tight_count / self.tight_eligible

    def to_dict(self, voiced_conf_available: bool = False) -> Dict[str, object]:
        per_type_rates: Dict[str, object] = {}
        for at in sorted(self.cutoff_total):
            rate = self.cutoff_violation_rate(at)
            per_type_rates[at] = {
                "total": self.cutoff_total[at],
                "violations": self.cutoff_violations.get(at, 0),
                "violation_rate": round(rate, 4) if rate is not None else None,
                "gate_ok": (rate is not None and rate <= _GATE_THRESHOLD),
            }

        deviation_by_coda: Dict[str, object] = {}
        for coda in sorted(self.deviation_total):
            mean_dev = self.mean_deviation(coda)
            deviation_by_coda[coda] = {
                "total_vc_rows": self.deviation_total[coda],
                "mean_deviation_ms": round(mean_dev, 2) if mean_dev is not None else None,
                "reference_margin_ms": _CODA_MARGINS.get(coda, 28.0),
            }

        agg_rate = self.aggregate_cutoff_violation_rate()
        tr = self.tight_rate()
        return {
            "cutoff_violations": {
                "per_alias_type": per_type_rates,
                "aggregate": {
                    "total": sum(self.cutoff_total.values()),
                    "violations": sum(self.cutoff_violations.values()),
                    "violation_rate": round(agg_rate, 4) if agg_rate is not None else None,
                    "gate_threshold": _GATE_THRESHOLD,
                    "gate_pass": (agg_rate is not None and agg_rate <= _GATE_THRESHOLD),
                },
            },
            "weighted_deviation": {
                "by_coda_type": deviation_by_coda,
                "note": "Deviation = max(0, reference_margin - cut_gap). Higher = worse.",
            },
            "voiceless_tight_rate": {
                "eligible_rows": self.tight_eligible,
                "tight_rows": self.tight_count,
                "rate": round(tr, 4) if tr is not None else None,
                "tight_threshold_ms": _TIGHT_THRESHOLD_MS,
                "voiced_conf_filter_applied": voiced_conf_available,
                "voiced_conf_threshold": _VOICED_CONF_THRESHOLD if voiced_conf_available else None,
            },
        }


# ---------------------------------------------------------------------------
# Feature cache loading
# ---------------------------------------------------------------------------

def _load_feature_conf_map(feature_json: str) -> Dict[str, float]:
    """Load voiced_conf from a JSON feature cache, keyed by source_row_id or alias."""
    if not feature_json or not os.path.exists(feature_json):
        return {}
    try:
        with open(feature_json, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
        else:
            return {}
        result: Dict[str, float] = {}
        for r in rows:
            conf = r.get("syllable_mel_voiced_conf") or r.get("voiced_conf") or 0.0
            key = r.get("source_row_id") or r.get("alias") or ""
            if key:
                result[str(key)] = float(conf)
        return result
    except Exception:
        return {}


def _lookup_voiced_conf(
    row: Dict[str, object], conf_map: Dict[str, float]
) -> Optional[float]:
    if not conf_map:
        return None
    key = str(row.get("source_row_id") or row.get("alias") or "")
    return conf_map.get(key)


# ---------------------------------------------------------------------------
# OTO file scanning
# ---------------------------------------------------------------------------

def _find_oto_file(bank_dir: str) -> str:
    for name in ("oto.ini", "oto.txt"):
        path = os.path.join(bank_dir, name)
        if os.path.exists(path):
            return path
    return ""


def _measure_oto(
    oto_path: str,
    language: str,
    format_type: str,
    conf_map: Dict[str, float],
) -> Tuple[MetricsAccumulator, List[Dict[str, object]]]:
    rows = parse_oto_rows(oto_path, language=language)
    acc = MetricsAccumulator()
    row_details: List[Dict[str, object]] = []

    for row in rows:
        alias_type, coda_type = _classify_row(language, row)
        gap = _cut_gap(row)
        cp_gap = _cons_pre_gap(row)
        voiced_conf = _lookup_voiced_conf(row, conf_map)

        acc.add_row(alias_type, coda_type, gap, cp_gap, voiced_conf)
        row_details.append({
            "alias": str(row.get("alias", "")),
            "alias_type": alias_type,
            "coda_type": coda_type,
            "offset": row.get("offset"),
            "cons": row.get("cons"),
            "cutoff": row.get("cutoff"),
            "pre": row.get("pre"),
            "cut_gap": round(gap, 2),
            "cons_pre_gap": round(cp_gap, 2),
            "voiced_conf": round(voiced_conf, 3) if voiced_conf is not None else None,
            "cutoff_violation": gap <= 0.0 and alias_type in _CUTOFF_TARGET_ALIAS_TYPES,
            "tight": (
                cp_gap < _TIGHT_THRESHOLD_MS and alias_type in _TIGHT_RATE_ALIAS_TYPES
                and (voiced_conf is None or voiced_conf >= _VOICED_CONF_THRESHOLD)
            ),
        })

    return acc, row_details


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _gate_symbol(ok: Optional[bool]) -> str:
    if ok is None:
        return "  ?"
    return " OK" if ok else "FAIL"


def _print_summary(label: str, summary: Dict[str, object], indent: int = 0) -> None:
    pad = " " * indent
    print(f"\n{pad}{'=' * 52}")
    print(f"{pad}  {label}")
    print(f"{pad}{'=' * 52}")

    cv = summary["cutoff_violations"]
    agg = cv["aggregate"]
    gate_ok = agg.get("gate_pass")
    agg_rate = agg.get("violation_rate")
    agg_pct = f"{agg_rate * 100:.1f}%" if agg_rate is not None else "n/a"
    gate_label = _gate_symbol(gate_ok)
    print(f"\n{pad}  Cutoff violations  [gate {agg['gate_threshold'] * 100:.0f}%] — [{gate_label}] {agg_pct}")
    print(f"{pad}  {'Alias type':<12} {'Total':>6} {'Violations':>10} {'Rate':>7} {'Gate':>5}")
    print(f"{pad}  {'-' * 46}")
    for at, vals in sorted(cv["per_alias_type"].items()):
        rate_val = vals.get("violation_rate")
        pct = f"{rate_val * 100:.1f}%" if rate_val is not None else "n/a"
        per_ok = vals.get("gate_ok")
        per_gate = _gate_symbol(per_ok)
        print(
            f"{pad}  {at:<12} {vals['total']:>6} {vals['violations']:>10} "
            f"{pct:>7} {per_gate:>5}"
        )

    wd = summary["weighted_deviation"]
    print(f"\n{pad}  Weighted deviation (VC rows, by coda type)")
    print(f"{pad}  {'Coda':<10} {'Rows':>6} {'Mean dev':>9} {'Ref margin':>11}")
    print(f"{pad}  {'-' * 40}")
    for coda, vals in sorted(wd["by_coda_type"].items()):
        mean_dev = vals.get("mean_deviation_ms")
        ref = vals["reference_margin_ms"]
        dev_str = f"{mean_dev:.1f}ms" if mean_dev is not None else "n/a"
        print(f"{pad}  {coda:<10} {vals['total_vc_rows']:>6} {dev_str:>9} {ref:.0f}ms{''!s:>6}")

    tr = summary["voiceless_tight_rate"]
    rate_val = tr.get("rate")
    rate_str = f"{rate_val * 100:.1f}%" if rate_val is not None else "n/a"
    conf_note = (
        f" (voiced_conf ≥ {tr['voiced_conf_threshold']:.2f} filter applied)"
        if tr["voiced_conf_filter_applied"]
        else " (no voiced_conf filter — all CV/VC rows included)"
    )
    print(
        f"\n{pad}  Voiceless tight-rate  (cons-pre < {tr['tight_threshold_ms']:.0f}ms)"
    )
    print(f"{pad}  {rate_str}  ({tr['tight_rows']}/{tr['eligible_rows']} rows){conf_note}")


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_summaries(
    baseline_summary: Dict[str, object], refined_summary: Dict[str, object]
) -> Dict[str, object]:
    """Compute per-metric deltas between baseline and ML-refined."""
    def _safe_delta(a, b):
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 4)

    base_cv = baseline_summary["cutoff_violations"]
    ref_cv = refined_summary["cutoff_violations"]

    per_type_deltas: Dict[str, object] = {}
    for at in set(base_cv["per_alias_type"]) | set(ref_cv["per_alias_type"]):
        bv = base_cv["per_alias_type"].get(at, {})
        rv = ref_cv["per_alias_type"].get(at, {})
        per_type_deltas[at] = _safe_delta(bv.get("violation_rate"), rv.get("violation_rate"))

    base_wd = baseline_summary["weighted_deviation"]["by_coda_type"]
    ref_wd = refined_summary["weighted_deviation"]["by_coda_type"]
    dev_deltas: Dict[str, object] = {}
    for coda in set(base_wd) | set(ref_wd):
        bv = base_wd.get(coda, {})
        rv = ref_wd.get(coda, {})
        dev_deltas[coda] = _safe_delta(bv.get("mean_deviation_ms"), rv.get("mean_deviation_ms"))

    return {
        "cutoff_violation_rate_delta": {
            "aggregate": _safe_delta(
                base_cv["aggregate"].get("violation_rate"),
                ref_cv["aggregate"].get("violation_rate"),
            ),
            "per_alias_type": per_type_deltas,
        },
        "mean_deviation_delta_ms": dev_deltas,
        "tight_rate_delta": _safe_delta(
            baseline_summary["voiceless_tight_rate"].get("rate"),
            refined_summary["voiceless_tight_rate"].get("rate"),
        ),
        "note": "Negative delta = improvement (fewer violations / lower deviation / lower tight-rate).",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(
        description="OTO-ML quality baseline measurement (PR-01).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--oto", default="", help="OTO file to measure (ML-refined output).")
    ap.add_argument("--baseline", default="", help="Rule-only baseline OTO for comparison.")
    ap.add_argument("--bank-dir", default="", help="Voicebank directory (auto-finds oto.ini/oto.txt).")
    ap.add_argument("--baseline-dir", default="", help="Baseline voicebank directory.")
    ap.add_argument("--language", default="korean", help="korean / japanese  (default: korean)")
    ap.add_argument("--format-type", default="cvvc", help="cv / cvc / cvvc  (default: cvvc)")
    ap.add_argument(
        "--feature-json",
        default="",
        help="Optional JSON feature cache containing voiced_conf per row.",
    )
    ap.add_argument("--report", default="", help="Write JSON report to this path.")
    ap.add_argument("--rows", action="store_true", help="Include per-row detail in JSON report.")
    args = ap.parse_args()

    language = str(args.language or "korean").strip().lower()
    format_type = str(args.format_type or "cvvc").strip().lower()

    # Resolve OTO paths
    oto_path = str(args.oto or "").strip()
    if not oto_path and args.bank_dir:
        oto_path = _find_oto_file(args.bank_dir)
    baseline_path = str(args.baseline or "").strip()
    if not baseline_path and args.baseline_dir:
        baseline_path = _find_oto_file(args.baseline_dir)

    if not oto_path:
        ap.error("Provide --oto or --bank-dir pointing to an OTO file.")
    if not os.path.exists(oto_path):
        ap.error(f"OTO file not found: {oto_path}")

    conf_map = _load_feature_conf_map(args.feature_json)
    voiced_conf_available = bool(conf_map)

    # Measure primary OTO
    acc, row_details = _measure_oto(oto_path, language, format_type, conf_map)
    refined_summary = acc.to_dict(voiced_conf_available=voiced_conf_available)

    report: Dict[str, object] = {
        "language": language,
        "format_type": format_type,
        "oto_path": oto_path,
        "baseline_path": baseline_path or None,
        "voiced_conf_filter": voiced_conf_available,
        "refined": refined_summary,
    }
    if args.rows:
        report["rows"] = row_details

    # Measure baseline if provided
    if baseline_path and os.path.exists(baseline_path):
        base_acc, base_rows = _measure_oto(baseline_path, language, format_type, conf_map)
        baseline_summary = base_acc.to_dict(voiced_conf_available=voiced_conf_available)
        report["baseline"] = baseline_summary
        report["comparison"] = _compare_summaries(baseline_summary, refined_summary)
        if args.rows:
            report["baseline_rows"] = base_rows

    # Print summaries
    _print_summary("ML-Refined OTO  —  " + os.path.basename(oto_path), refined_summary)
    if "baseline" in report:
        _print_summary(
            "Baseline OTO  —  " + os.path.basename(baseline_path), report["baseline"]
        )
        comp = report["comparison"]
        agg_delta = comp["cutoff_violation_rate_delta"].get("aggregate")
        tight_delta = comp.get("tight_rate_delta")
        print("\n  Delta (ML - baseline):")
        agg_str = (
            f"{agg_delta * 100:+.1f}%" if agg_delta is not None else "n/a"
        )
        tight_str = (
            f"{tight_delta * 100:+.1f}%" if tight_delta is not None else "n/a"
        )
        print(f"    Aggregate cutoff violation rate: {agg_str}")
        print(f"    Tight-rate:                      {tight_str}")
        for coda, delta_ms in sorted(comp["mean_deviation_delta_ms"].items()):
            d_str = f"{delta_ms:+.1f}ms" if delta_ms is not None else "n/a"
            print(f"    Mean deviation [{coda}]:        {d_str}")

    # Gate check
    agg_rate = refined_summary["cutoff_violations"]["aggregate"]["violation_rate"]
    gate_pass = refined_summary["cutoff_violations"]["aggregate"]["gate_pass"]
    if agg_rate is not None:
        gate_label = "PASS" if gate_pass else "FAIL"
        print(f"\n  CI Gate [{_GATE_THRESHOLD * 100:.0f}% aggregate threshold]: {gate_label}")

    # Write report
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Report written: {args.report}")

    # Exit with non-zero if gate fails
    if agg_rate is not None and not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
