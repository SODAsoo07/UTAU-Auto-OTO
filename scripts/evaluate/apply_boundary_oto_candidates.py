"""Apply gated phoneme-boundary OTO offset candidates for evaluation.

This script is intentionally evaluation-only. Runtime-safe candidate logic
lives in ``core.generation.common.oto_boundary_candidates``; this CLI adds
gold OTO before/after reporting for model and gate validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.generation.common.oto_alias_family import alias_family
from core.generation.common.oto_boundary_candidates import (
    DEFAULT_ROLE_LABELS,
    CandidateConfig,
    apply_boundary_candidates,
    config_to_json,
    load_score_maps_for_oto,
    parse_float_map,
    parse_role_labels,
    read_oto_lines,
    write_oto_lines,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Generated/predicted oto.ini to adjust.")
    parser.add_argument("--wav-dir", required=True, help="Directory containing wav files referenced by --pred.")
    parser.add_argument("--model", required=True, help="Phoneme-boundary checkpoint path.")
    parser.add_argument("--out", default="", help="Adjusted oto.ini output path. If omitted, only writes a report.")
    parser.add_argument("--report", default="", help="JSON report output path. Defaults to stdout only.")
    parser.add_argument("--gold", default="", help="Optional trusted gold oto.ini for before/after diff.")
    parser.add_argument(
        "--cutoff-end-mode",
        choices=("raw", "relative", "utau", "compatible"),
        default="raw",
        help="Cutoff interpretation passed to scripts.dev.diff_oto when --gold is provided.",
    )
    parser.add_argument("--language", default="japanese", help="Metadata only; candidate policy is role-label based.")
    parser.add_argument("--format-type", default="cvvc", help="Metadata only; candidate policy is role-label based.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--search-radius-ms", type=float, default=120.0)
    parser.add_argument("--max-shift-ms", type=float, default=60.0)
    parser.add_argument("--min-shift-ms", type=float, default=8.0)
    parser.add_argument("--min-score", type=float, default=0.18)
    parser.add_argument("--min-score-margin", type=float, default=0.035)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--min-new-offset-ms", type=float, default=0.0)
    parser.add_argument("--center-penalty-per-ms", type=float, default=0.0)
    parser.add_argument("--keep-edge-peaks", action="store_true", help="Allow peaks selected at the search-window edge.")
    parser.add_argument("--enable-vv", action="store_true", help="Also allow VV rows to use vowel_onset candidates.")
    parser.add_argument(
        "--role-labels",
        default="",
        help="Comma-separated family:label mapping. Overrides the default cv/vc policy.",
    )
    parser.add_argument(
        "--role-max-shift-ms",
        default="",
        help="Comma-separated family:max_shift_ms overrides, e.g. cv:60,vc:50.",
    )
    parser.add_argument(
        "--role-min-score-margin",
        default="",
        help="Comma-separated family:min_score_margin overrides, e.g. vc:0.05.",
    )
    parser.add_argument("--enable-neighbor-guard", action="store_true")
    parser.add_argument("--neighbor-guard-min-gap-ms", type=float, default=10.0)
    parser.add_argument("--disable-acoustic-heuristics", action="store_true")
    parser.add_argument("--heuristic-weight", type=float, default=0.28)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = read_oto_lines(args.pred)
    config = CandidateConfig(
        role_labels=parse_role_labels(args.role_labels, enable_vv=bool(args.enable_vv)),
        search_radius_ms=float(args.search_radius_ms),
        max_shift_ms=float(args.max_shift_ms),
        min_shift_ms=float(args.min_shift_ms),
        min_score=float(args.min_score),
        min_score_margin=float(args.min_score_margin),
        min_quality=float(args.min_quality),
        min_new_offset_ms=float(args.min_new_offset_ms),
        skip_edge_peaks=not bool(args.keep_edge_peaks),
        center_penalty_per_ms=float(args.center_penalty_per_ms),
        role_max_shift_ms=parse_float_map(args.role_max_shift_ms),
        role_min_score_margin=parse_float_map(args.role_min_score_margin),
        neighbor_guard_enabled=bool(args.enable_neighbor_guard),
        neighbor_guard_min_gap_ms=float(args.neighbor_guard_min_gap_ms),
    )
    score_maps = load_score_maps_for_oto(
        rows,
        wav_dir=args.wav_dir,
        model_path=args.model,
        device=args.device,
        use_acoustic_heuristics=not bool(args.disable_acoustic_heuristics),
        heuristic_weight=float(args.heuristic_weight),
    )
    out_lines, report = apply_boundary_candidates(rows, score_maps, config)
    report["config"] = config_to_json(config)
    report["inputs"] = {
        "pred": os.path.abspath(args.pred),
        "wav_dir": os.path.abspath(args.wav_dir),
        "model": os.path.abspath(args.model),
        "gold": os.path.abspath(args.gold) if args.gold else "",
        "language": str(args.language),
        "format_type": str(args.format_type),
    }
    report["score_maps"] = {
        "requested_wavs": len({row.wav for row in rows if row.parsed and row.wav}),
        "loaded_wavs": len(score_maps),
    }

    adjusted_path = args.out
    if adjusted_path:
        write_oto_lines(adjusted_path, out_lines)

    if args.gold:
        report["diff"] = _build_before_after_diff(
            gold_path=args.gold,
            pred_path=args.pred,
            adjusted_path=adjusted_path,
            adjusted_lines=out_lines,
            wav_dir=args.wav_dir,
            cutoff_end_mode=args.cutoff_end_mode,
        )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    _print_text(payload)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    return 0


def _build_before_after_diff(
    *,
    gold_path: str,
    pred_path: str,
    adjusted_path: str,
    adjusted_lines: list[str],
    wav_dir: str,
    cutoff_end_mode: str,
) -> dict[str, Any]:
    from scripts.dev.diff_oto import build_report

    if adjusted_path:
        path_for_diff = adjusted_path
        before, _before_rows = build_report(
            gold_path=gold_path,
            pred_path=pred_path,
            cutoff_end_mode=cutoff_end_mode,
            wav_dir=wav_dir,
        )
        after, _after_rows = build_report(
            gold_path=gold_path,
            pred_path=path_for_diff,
            cutoff_end_mode=cutoff_end_mode,
            wav_dir=wav_dir,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="boundary_oto_candidates_") as tmp_dir:
            path_for_diff = os.path.join(tmp_dir, "adjusted.ini")
            write_oto_lines(path_for_diff, adjusted_lines)
            before, _before_rows = build_report(
                gold_path=gold_path,
                pred_path=pred_path,
                cutoff_end_mode=cutoff_end_mode,
                wav_dir=wav_dir,
            )
            after, _after_rows = build_report(
                gold_path=gold_path,
                pred_path=path_for_diff,
                cutoff_end_mode=cutoff_end_mode,
                wav_dir=wav_dir,
            )
    anchor_before = _anchor_abs_report(gold_path, pred_path)
    anchor_after = _anchor_abs_report(gold_path, path_for_diff)
    return {
        "before": before,
        "after": after,
        "delta": _diff_delta(before, after),
        "anchor_abs": {
            "before": anchor_before,
            "after": anchor_after,
            "delta": _anchor_abs_delta(anchor_before, anchor_after),
        },
    }


def _anchor_abs_report(gold_path: str, pred_path: str) -> dict[str, Any]:
    from scripts.dev.diff_oto import load

    gold = load(gold_path)
    pred = load(pred_path)
    common = sorted(set(gold.keys()) & set(pred.keys()))
    by_family: dict[str, list[float]] = defaultdict(list)
    values: list[float] = []
    for key in common:
        family = alias_family(key[1])
        gold_anchor = float(gold[key].get("offset", 0.0)) + float(gold[key].get("preutterance", 0.0))
        pred_anchor = float(pred[key].get("offset", 0.0)) + float(pred[key].get("preutterance", 0.0))
        delta = pred_anchor - gold_anchor
        values.append(delta)
        by_family[family].append(delta)
    return {
        "rows_matched": len(common),
        "pre_abs": _residual_stats(values),
        "by_alias_family": {
            family: {"n": len(items), "pre_abs": _residual_stats(items)} for family, items in sorted(by_family.items())
        },
    }


def _anchor_abs_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pre_abs": _metric_delta(before.get("pre_abs"), after.get("pre_abs")),
        "by_alias_family": {},
    }
    before_family = before.get("by_alias_family") if isinstance(before.get("by_alias_family"), Mapping) else {}
    after_family = after.get("by_alias_family") if isinstance(after.get("by_alias_family"), Mapping) else {}
    for family in sorted(set(before_family) | set(after_family)):
        b_payload = before_family.get(family) if isinstance(before_family.get(family), Mapping) else {}
        a_payload = after_family.get(family) if isinstance(after_family.get(family), Mapping) else {}
        out["by_alias_family"][family] = {
            "n": int(a_payload.get("n") or b_payload.get("n") or 0),
            "pre_abs": _metric_delta(b_payload.get("pre_abs"), a_payload.get("pre_abs")),
        }
    return out


def _diff_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_param": {}, "by_alias_family": {}}
    before_params = before.get("by_param") if isinstance(before.get("by_param"), Mapping) else {}
    after_params = after.get("by_param") if isinstance(after.get("by_param"), Mapping) else {}
    for param in sorted(set(before_params) | set(after_params)):
        out["by_param"][param] = _metric_delta(before_params.get(param), after_params.get(param))
    before_family = before.get("by_alias_family") if isinstance(before.get("by_alias_family"), Mapping) else {}
    after_family = after.get("by_alias_family") if isinstance(after.get("by_alias_family"), Mapping) else {}
    for family in sorted(set(before_family) | set(after_family)):
        b_payload = before_family.get(family) if isinstance(before_family.get(family), Mapping) else {}
        a_payload = after_family.get(family) if isinstance(after_family.get(family), Mapping) else {}
        b_params = b_payload.get("by_param") if isinstance(b_payload.get("by_param"), Mapping) else {}
        a_params = a_payload.get("by_param") if isinstance(a_payload.get("by_param"), Mapping) else {}
        out["by_alias_family"][family] = {
            "n": int(a_payload.get("n") or b_payload.get("n") or 0),
            "by_param": {
                param: _metric_delta(b_params.get(param), a_params.get(param))
                for param in sorted(set(b_params) | set(a_params))
            },
        }
    return out


def _metric_delta(before: Any, after: Any) -> dict[str, float]:
    b = before if isinstance(before, Mapping) else {}
    a = after if isinstance(after, Mapping) else {}
    keys = ("MAE_ms", "abs_median_ms", "abs_p90_ms", "mean_delta_ms")
    return {
        f"{key}_delta": round(float(a.get(key, 0.0) or 0.0) - float(b.get(key, 0.0) or 0.0), 3)
        for key in keys
    }


def _residual_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    raw = [float(value) for value in values]
    abs_values = sorted(abs(value) for value in raw)
    return {
        "n": len(raw),
        "mean_delta_ms": round(sum(raw) / len(raw), 1),
        "abs_median_ms": round(abs_values[len(abs_values) // 2], 1),
        "abs_p90_ms": round(abs_values[int(0.9 * (len(abs_values) - 1))], 1),
        "MAE_ms": round(sum(abs_values) / len(abs_values), 1),
    }


def _print_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
