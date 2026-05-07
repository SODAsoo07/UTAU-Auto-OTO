from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.coarse_crnn.oto_fallback_policy import FallbackPolicy, decide_fallback, signals_from_prediction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid-search fallback policy thresholds from OTO evaluation JSON (analysis-only)."
    )
    parser.add_argument("--eval-json", required=True, help="Path to evaluate_oto.py output JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    parser.add_argument("--bad-error-ms", type=float, default=120.0)
    parser.add_argument("--good-error-ms", type=float, default=50.0)
    parser.add_argument("--false-fallback-penalty", type=float, default=0.65)
    args = parser.parse_args()

    payload = _read_json(args.eval_json)
    rows = list(payload.get("files") or [])
    if not rows:
        raise ValueError(f"no rows in evaluation JSON: {args.eval_json}")

    best = _search(
        rows,
        bad_error_ms=float(args.bad_error_ms),
        good_error_ms=float(args.good_error_ms),
        false_fallback_penalty=float(args.false_fallback_penalty),
    )
    out = {
        "eval_json": str(Path(args.eval_json).resolve()),
        "row_count": len(rows),
        "bad_error_ms": float(args.bad_error_ms),
        "good_error_ms": float(args.good_error_ms),
        "false_fallback_penalty": float(args.false_fallback_penalty),
        "best": best,
        "recommended_env": _recommend_env(best),
    }
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _search(rows: list[dict[str, Any]], *, bad_error_ms: float, good_error_ms: float, false_fallback_penalty: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    conf_values = [0.28, 0.32, 0.36, 0.40, 0.44]
    min_anchor_values = [0.22, 0.26, 0.30, 0.34]
    span_values = [0.08, 0.10, 0.12, 0.16]
    score_values = [0.50, 0.60, 0.70, 0.80]
    blend_values = [0.45, 0.55, 0.65]
    for conf_max in conf_values:
        for min_anchor_max in min_anchor_values:
            for span_max in span_values:
                for score_threshold in score_values:
                    for blend in blend_values:
                        policy = FallbackPolicy(
                            enabled=True,
                            score_threshold=float(score_threshold),
                            source_blend=float(blend),
                            conf_max=float(conf_max),
                            min_anchor_conf_max=float(min_anchor_max),
                            conf_span_max=float(span_max),
                        )
                        metrics = _evaluate_policy(rows, policy, bad_error_ms=bad_error_ms, good_error_ms=good_error_ms)
                        objective = float(metrics["bad_capture_rate"]) - (float(false_fallback_penalty) * float(metrics["good_false_fallback_rate"]))
                        candidate = {
                            "objective": objective,
                            "policy": asdict(policy),
                            "metrics": metrics,
                        }
                        if best is None or float(candidate["objective"]) > float(best["objective"]):
                            best = candidate
    return dict(best or {})


def _evaluate_policy(rows: list[dict[str, Any]], policy: FallbackPolicy, *, bad_error_ms: float, good_error_ms: float) -> dict[str, float]:
    total = len(rows)
    fallback_count = 0
    bad_count = 0
    good_count = 0
    bad_hit = 0
    good_false = 0
    for row in rows:
        pre_err = float(((row.get("param_abs_errors_ms") or {}).get("preutterance")) or 0.0)
        is_bad = pre_err >= float(bad_error_ms)
        is_good = pre_err <= float(good_error_ms)
        if is_bad:
            bad_count += 1
        if is_good:
            good_count += 1
        signals = signals_from_prediction(
            confidence=float(row.get("confidence", 0.0) or 0.0),
            heatmap_confidence=None,
            alias_type=row.get("alias_type", ""),
            alias_role=row.get("alias_role", ""),
            transition_type=row.get("transition_type", ""),
            row_ratio=float(row.get("row_ratio_in_wav", 0.0) or 0.0),
            is_special=bool(row.get("is_special", False)),
        )
        decision = decide_fallback(signals, policy)
        if decision.apply:
            fallback_count += 1
            if is_bad:
                bad_hit += 1
            if is_good:
                good_false += 1
    return {
        "fallback_rate": float(fallback_count) / float(max(1, total)),
        "bad_capture_rate": float(bad_hit) / float(max(1, bad_count)),
        "good_false_fallback_rate": float(good_false) / float(max(1, good_count)),
        "bad_count": float(bad_count),
        "good_count": float(good_count),
    }


def _recommend_env(best: dict[str, Any]) -> dict[str, str]:
    policy = dict(best.get("policy") or {})
    if not policy:
        return {}
    return {
        "UTOA_OTO_CRNN_FALLBACK_ENABLE": "1",
        "UTOA_OTO_CRNN_FALLBACK_SCORE_THRESHOLD": f"{float(policy.get('score_threshold', 0.65)):.3f}",
        "UTOA_OTO_CRNN_FALLBACK_SOURCE_BLEND": f"{float(policy.get('source_blend', 0.60)):.3f}",
        "UTOA_OTO_CRNN_FALLBACK_CONF_MAX": f"{float(policy.get('conf_max', 0.38)):.3f}",
        "UTOA_OTO_CRNN_FALLBACK_MIN_ANCHOR_CONF_MAX": f"{float(policy.get('min_anchor_conf_max', 0.30)):.3f}",
        "UTOA_OTO_CRNN_FALLBACK_CONF_SPAN_MAX": f"{float(policy.get('conf_span_max', 0.12)):.3f}",
    }


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
