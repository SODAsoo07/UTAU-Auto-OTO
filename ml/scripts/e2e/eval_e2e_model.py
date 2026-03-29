from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_e2e_runtime import build_e2e_runtime_context, predict_e2e_params
from core.oto_e2e_selector import E2ESelectorConfig, select_e2e_or_legacy


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_alias_group(alias_type: object) -> str:
    alias = str(alias_type or "").strip().lower()
    if alias in {"cv", "cv_head", "mono"}:
        return "cv"
    if alias == "vc":
        return "vc"
    if alias in {"vv", "vcv"}:
        return "vv"
    return "other"


def _load_rows(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _mae(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(abs(v) for v in values) / float(len(values)))


def _violation_rate(params_rows: List[tuple[float, float, float, float, float]]) -> float:
    if not params_rows:
        return 0.0
    bad = 0
    for off, cons, cutoff, pre, _ovl in params_rows:
        _ = float(off)
        c = float(cons)
        cut = float(cutoff)
        p = float(pre)
        # OTO practical constraints:
        # consonant/pre are local values (not absolute offset-based),
        # pre should not exceed consonant, cutoff usually negative.
        if c < 0.0 or p < 0.0 or p > c or cut >= 0.0:
            bad += 1
    return float(bad / float(len(params_rows)))


def _resolve_manual_cutoff_abs_for_eval(row: Dict[str, object]) -> Tuple[float, str]:
    manual_offset = _safe_float(row.get("manual_offset", 0.0), 0.0)
    manual_cutoff = _safe_float(row.get("manual_cutoff", 0.0), 0.0)

    existing_abs = _safe_float(row.get("manual_cutoff_abs", 0.0), 0.0)
    existing_mode = str(row.get("manual_cutoff_mode", "") or "").strip().lower()
    if existing_abs > (manual_offset + 2.0):
        return float(existing_abs), (existing_mode or "stored")

    # Standard negative cutoff: relative tail distance from offset.
    if manual_cutoff < 0.0:
        return float(manual_offset + abs(manual_cutoff)), "negative_relative"

    base_cut_abs = _safe_float(row.get("base_cutoff_abs", 0.0), manual_offset + 40.0)
    wav_ms = _safe_float(row.get("wav_duration_ms", 0.0), 0.0)
    cutoff_abs_raw = abs(float(manual_cutoff))

    # Positive cutoff can appear as relative/absolute/end-referenced legacy styles.
    candidates = [
        ("positive_relative", float(manual_offset + cutoff_abs_raw)),
        ("positive_absolute", float(cutoff_abs_raw)),
    ]
    if wav_ms > 0.0:
        candidates.append(
            ("positive_from_wav_end", float(manual_offset + max(wav_ms - cutoff_abs_raw, 0.0)))
        )

    scored: List[Tuple[float, str, float]] = []
    for mode, cand in candidates:
        score = abs(cand - base_cut_abs)
        if wav_ms > 0.0:
            if cand > (wav_ms + 80.0):
                score += 2000.0 + (cand - wav_ms)
            if cand < (manual_offset + 4.0):
                score += 800.0 + ((manual_offset + 4.0) - cand)
        scored.append((score, mode, cand))
    scored.sort(key=lambda x: x[0])
    chosen_mode, chosen_abs = scored[0][1], float(scored[0][2])

    if wav_ms > 0.0:
        chosen_abs = min(chosen_abs, max(0.0, wav_ms - 2.0))
    if chosen_abs <= (manual_offset + 2.0):
        # Fallback: keep evaluation stable even for malformed rows.
        return float(manual_offset + max(2.1, cutoff_abs_raw)), "fallback_invalid_cutoff"
    return chosen_abs, chosen_mode


def _manual_cutoff_canonical(row: Dict[str, object]) -> Tuple[float, str]:
    manual_offset = _safe_float(row.get("manual_offset", 0.0), 0.0)
    cutoff_abs, mode = _resolve_manual_cutoff_abs_for_eval(row)
    return float(-(cutoff_abs - manual_offset)), mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate E2E model on dataset CSV.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--language", required=True, choices=["korean", "japanese"])
    parser.add_argument("--format-type", default="cv")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "legacy_only", "e2e_only"])
    parser.add_argument("--t-low", type=float, default=0.52)
    parser.add_argument("--t-high", type=float, default=0.72)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--cv-only", default="1")
    parser.add_argument("--alias-group", choices=["all", "cv", "vc", "vv", "other"], default="all")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    rows = _load_rows(os.path.abspath(args.dataset))
    alias_group = str(args.alias_group or "all").strip().lower()
    if alias_group != "all":
        rows = [r for r in rows if _normalize_alias_group(r.get("alias_type", "")) == alias_group]
    elif str(args.cv_only).strip() not in {"0", "false", "False"}:
        rows = [r for r in rows if str(r.get("alias_type", "") or "").strip().lower() in {"cv", "cv_head"}]

    if not rows:
        raise RuntimeError("No rows to evaluate.")

    ctx = None
    if str(args.mode).strip().lower() != "legacy_only":
        if args.language == "japanese":
            os.environ["UTOA_JA_E2E_MODEL_DIR"] = os.path.abspath(args.model_dir)
            os.environ["UTOA_JA_E2E_ALIAS_MODEL_DIR"] = os.path.abspath(args.model_dir)
        else:
            os.environ["UTOA_KR_E2E_MODEL_DIR"] = os.path.abspath(args.model_dir)
            os.environ["UTOA_KR_E2E_ALIAS_MODEL_DIR"] = os.path.abspath(args.model_dir)
        os.environ["UTOA_E2E_ENABLE"] = "1"
        os.environ["UTOA_E2E_MODE"] = str(args.mode)
        os.environ["UTOA_E2E_T_LOW"] = str(args.t_low)
        os.environ["UTOA_E2E_T_HIGH"] = str(args.t_high)
        os.environ["UTOA_E2E_BLEND_ALPHA"] = str(args.alpha)

        ctx = build_e2e_runtime_context(args.language, args.format_type, "e2e_hybrid")
        if not bool(getattr(ctx, "active", False)):
            raise RuntimeError(f"E2E runtime inactive: {getattr(ctx, 'code', '')} {getattr(ctx, 'message', '')}")

    cfg = E2ESelectorConfig(mode=str(args.mode), t_low=float(args.t_low), t_high=float(args.t_high), blend_alpha=float(args.alpha))

    mae_acc = {"offset": [], "cons": [], "cutoff": [], "pre": [], "ovl": []}
    mae_cutoff_raw_acc: List[float] = []
    cutoff_mode_counts: Dict[str, int] = {}
    cutoff_mode_mae_acc: Dict[str, List[float]] = {}
    pred_params_rows: List[tuple[float, float, float, float, float]] = []
    selected_counts: Dict[str, int] = {}
    conf_values: List[float] = []
    predict_failed_count = 0
    predict_failed_codes: Dict[str, int] = {}

    for row in rows:
        legacy_offset = _safe_float(row.get("base_offset", 0.0), 0.0)
        legacy_cons = _safe_float(row.get("base_cons", 0.0), 0.0)
        legacy_cut_abs = _safe_float(row.get("base_cutoff_abs", 0.0), legacy_offset + 10.0)
        legacy_cutoff = -(legacy_cut_abs - legacy_offset)
        legacy_pre = _safe_float(row.get("base_pre", 0.0), 0.0)
        legacy_ovl = _safe_float(row.get("base_ovl", 0.0), 0.0)
        legacy_params = (legacy_offset, legacy_cons, legacy_cutoff, legacy_pre, legacy_ovl)

        if str(args.mode).strip().lower() == "legacy_only":
            out = legacy_params
            selected_counts["legacy"] = int(selected_counts.get("legacy", 0)) + 1
        else:
            pred = predict_e2e_params(ctx, row)
            if not bool(pred.get("ok", False)):
                out = legacy_params
                selected_counts["legacy"] = int(selected_counts.get("legacy", 0)) + 1
                predict_failed_count += 1
                fail_code = str(pred.get("code", "predict_failed") or "predict_failed")
                predict_failed_codes[fail_code] = int(predict_failed_codes.get(fail_code, 0)) + 1
            else:
                e2e_params = pred.get("params")
                if not isinstance(e2e_params, tuple) or len(e2e_params) != 5:
                    out = legacy_params
                    selected_counts["legacy"] = int(selected_counts.get("legacy", 0)) + 1
                    predict_failed_count += 1
                    predict_failed_codes["invalid_params"] = int(predict_failed_codes.get("invalid_params", 0)) + 1
                else:
                    sel = select_e2e_or_legacy(
                        legacy_params=legacy_params,
                        e2e_params=e2e_params,
                        score=float(pred.get("confidence", 0.0) or 0.0),
                        config=cfg,
                        validate_fn=None,
                        alias_type=str(row.get("alias_type", "") or ""),
                    )
                    out = sel.params
                    selected_counts[sel.selected] = int(selected_counts.get(sel.selected, 0)) + 1
                    conf_values.append(float(pred.get("confidence", 0.0) or 0.0))

        manual_cutoff_canonical, cutoff_mode = _manual_cutoff_canonical(row)
        cutoff_mode_counts[cutoff_mode] = int(cutoff_mode_counts.get(cutoff_mode, 0)) + 1

        manual = (
            _safe_float(row.get("manual_offset", 0.0), 0.0),
            _safe_float(row.get("manual_cons", 0.0), 0.0),
            float(manual_cutoff_canonical),
            _safe_float(row.get("manual_pre", 0.0), 0.0),
            _safe_float(row.get("manual_ovl", 0.0), 0.0),
        )
        manual_cutoff_raw = _safe_float(row.get("manual_cutoff", 0.0), 0.0)
        mae_acc["offset"].append(out[0] - manual[0])
        mae_acc["cons"].append(out[1] - manual[1])
        mae_acc["cutoff"].append(out[2] - manual[2])
        mae_cutoff_raw_acc.append(out[2] - manual_cutoff_raw)
        mae_acc["pre"].append(out[3] - manual[3])
        mae_acc["ovl"].append(out[4] - manual[4])
        mode_errs = cutoff_mode_mae_acc.setdefault(cutoff_mode, [])
        mode_errs.append(out[2] - manual[2])
        pred_params_rows.append(out)

    result = {
        "rows_eval": int(len(pred_params_rows)),
        "mode": str(args.mode),
        "language": str(args.language),
        "format_type": str(args.format_type),
        "alias_group": alias_group,
        "mae": {k: _mae(v) for k, v in mae_acc.items()},
        "cutoff_metric": "canonical",
        "mae_cutoff_raw_legacy": _mae(mae_cutoff_raw_acc),
        "cutoff_mode_counts": cutoff_mode_counts,
        "mae_cutoff_by_mode": {k: _mae(v) for k, v in cutoff_mode_mae_acc.items()},
        "constraint_violation_rate": _violation_rate(pred_params_rows),
        "selected_counts": selected_counts,
        "avg_confidence": (sum(conf_values) / float(len(conf_values))) if conf_values else 0.0,
        "predict_failed_count": int(predict_failed_count),
        "predict_failed_codes": predict_failed_codes,
        "model_dir": os.path.abspath(args.model_dir),
    }

    out_json = str(args.out_json or "").strip()
    if not out_json:
        out_json = os.path.splitext(os.path.abspath(args.dataset))[0] + ".e2e_eval.json"
    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[E2E-Eval] rows={result['rows_eval']} out={out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
