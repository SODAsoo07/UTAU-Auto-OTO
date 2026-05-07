from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.coarse_crnn.oto_confidence_calibration import ConfidenceCalibration, apply_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit simple confidence calibration from OTO eval JSON.")
    parser.add_argument("--eval-json", required=True, help="Path to evaluate_oto.py output JSON.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--good-threshold-ms", type=float, default=50.0)
    parser.add_argument("--bad-threshold-ms", type=float, default=120.0)
    args = parser.parse_args()

    payload = _read_json(args.eval_json)
    rows = list(payload.get("files") or [])
    if not rows:
        raise ValueError(f"no files rows in evaluation JSON: {args.eval_json}")

    best = _fit(rows, good_ms=float(args.good_threshold_ms), bad_ms=float(args.bad_threshold_ms))
    result = {
        "eval_json": str(Path(args.eval_json).resolve()),
        "row_count": len(rows),
        "good_threshold_ms": float(args.good_threshold_ms),
        "bad_threshold_ms": float(args.bad_threshold_ms),
        "best": best,
        "recommended_env": {
            "UTOA_OTO_CRNN_CONF_CALIB_ENABLE": "1",
            "UTOA_OTO_CRNN_CONF_CALIB_SCALE": f"{float(best['scale']):.6f}",
            "UTOA_OTO_CRNN_CONF_CALIB_BIAS": f"{float(best['bias']):.6f}",
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


def _fit(rows: list[dict[str, Any]], *, good_ms: float, bad_ms: float) -> dict[str, float]:
    best: dict[str, float] | None = None
    for scale in (0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0):
        for bias in (-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0):
            calib = ConfidenceCalibration(enabled=True, scale=float(scale), bias=float(bias))
            metrics = _score(rows, calib, good_ms=good_ms, bad_ms=bad_ms)
            objective = float(metrics["good_mean"] - metrics["bad_mean"])
            candidate = {
                "scale": float(scale),
                "bias": float(bias),
                "objective": float(objective),
                **metrics,
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
    return dict(best or {})


def _score(rows: list[dict[str, Any]], calib: ConfidenceCalibration, *, good_ms: float, bad_ms: float) -> dict[str, float]:
    good: list[float] = []
    bad: list[float] = []
    for row in rows:
        base_conf = float(row.get("confidence", 0.0) or 0.0)
        conf = apply_calibration(base_conf, calib)
        pre_err = float(((row.get("param_abs_errors_ms") or {}).get("preutterance")) or 0.0)
        if pre_err <= float(good_ms):
            good.append(conf)
        elif pre_err >= float(bad_ms):
            bad.append(conf)
    good_mean = float(sum(good) / len(good)) if good else 0.0
    bad_mean = float(sum(bad) / len(bad)) if bad else 0.0
    return {
        "good_mean": good_mean,
        "bad_mean": bad_mean,
        "good_count": float(len(good)),
        "bad_count": float(len(bad)),
    }


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
