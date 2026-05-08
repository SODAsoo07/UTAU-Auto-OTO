"""Explore multi-signal fallback trigger combinations on an evaluate_oto JSON.

Plan B from [[CRNN-정확도-개선-방안-v2]] 0-order: when the confidence head
is collapsed, monotonic recalibration cannot recover separation. Replace
the confidence-only trigger with combinations of activity-based signals
(``syllable_shift_flag``, ``fn_voiced_miss``, ``hard_failure``,
``heatmap``, ``predicted_error_ms``).

This script consumes the per-row ``files`` array of evaluate_oto.py and
runs each candidate trigger from
``core.coarse_crnn.oto_fallback_policy.default_trigger_grid()`` over a set
of slices (``language/format/role`` patterns). For each (slice, trigger)
pair it reports:
- ``fallback_rate``: how often the trigger fires
- ``bad_recall``: of the truly bad rows, how many are caught
- ``precision`` and ``good_keep_rate``
- ``effective_*``: anchor metrics assuming a perfect fallback overrides
  triggered rows with zero error (upper bound, same accounting as the
  legacy single-trigger analyzer).

The "best" trigger inside each slice is picked by the v2 0-order intent:
maximise ``bad_recall``, then ``good_keep_rate``, then minimise
``fallback_rate``. The script never modifies inference; it only writes
JSON + Markdown reports for human selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from core.coarse_crnn.oto_fallback_policy import (
    default_trigger_grid,
    evaluate_policy_grid,
)


def _slice_rows(
    files: list[dict[str, Any]],
    *,
    language: str,
    format_type: str,
    alias_role: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in files:
        if language and str(row.get("language", "") or "") != language:
            continue
        if format_type and str(row.get("format_type", "") or "") != format_type:
            continue
        if alias_role and str(row.get("alias_role", "") or "") != alias_role:
            continue
        out.append(row)
    return out


def _parse_slice(raw: str) -> tuple[str, str, str]:
    parts = str(raw or "").split("/")
    while len(parts) < 3:
        parts.append("*")
    return (
        "" if parts[0] == "*" else parts[0],
        "" if parts[1] == "*" else parts[1],
        "" if parts[2] == "*" else parts[2],
    )


def _slice_label(language: str, format_type: str, alias_role: str) -> str:
    return f"{language or '*'}/{format_type or '*'}/{alias_role or '*'}"


def _render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# OTO CRNN multi-signal fallback exploration (Plan B)",
        "",
        f"- Source eval: `{result.get('source_eval', '')}`",
        f"- Total rows: {result.get('total_rows', 0)}",
        f"- bad_pre_ms: {result.get('bad_pre_ms', 0)}",
        f"- bad_offset_ms: {result.get('bad_offset_ms', 0)}",
        f"- bad_cutoff_ms: {result.get('bad_cutoff_ms', 0)}",
        "",
    ]
    for slice_block in result.get("slices", []):
        label = slice_block.get("slice", "?")
        rows = slice_block.get("rows", 0)
        bad_rate = slice_block.get("bad_rate", 0.0)
        lines.extend([
            f"## `{label}` ({rows} rows, bad_rate {bad_rate:.4f})",
            "",
            "| trigger | fallback | bad_recall | precision | good_keep | eff_pre_acc50 | eff_pre_mae | eff_offset | eff_cutoff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cand in slice_block.get("candidates", []):
            lines.append(
                "| `{name}` | {fallback_rate:.4f} | {bad_recall:.4f} | {precision:.4f} | "
                "{good_keep_rate:.4f} | {effective_preutterance_acc_50ms:.4f} | "
                "{effective_preutterance_mae_ms:.2f} | {effective_offset_mae_ms:.2f} | "
                "{effective_cutoff_abs_mae_ms:.2f} |".format(**cand)
            )
        best = slice_block.get("best")
        if best:
            lines.extend([
                "",
                f"**Best (Plan B priority)**: `{best['name']}`. "
                f"bad_recall {best['bad_recall']:.4f}, "
                f"good_keep {best['good_keep_rate']:.4f}, "
                f"fallback_rate {best['fallback_rate']:.4f}, "
                f"effective_pre_acc50 {best['effective_preutterance_acc_50ms']:.4f}.",
                "",
            ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore multi-signal fallback trigger combinations.")
    parser.add_argument("--eval", required=True, help="Path to evaluate_oto.py output JSON.")
    parser.add_argument("--out", required=True, help="Path to write the exploration result JSON.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown summary path.")
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Slice in 'language/format/role' form, '*' wildcard. Repeatable. "
             "Default covers global, korean/cvvc/*, japanese/cvvc/*, korean/cvvc/{vc,vv,other}, korean/cvc/*.",
    )
    parser.add_argument("--bad-pre-ms", type=float, default=50.0)
    parser.add_argument("--bad-offset-ms", type=float, default=700.0)
    parser.add_argument("--bad-cutoff-ms", type=float, default=900.0)
    parser.add_argument(
        "--bad-mode",
        choices=("errors_only", "strict"),
        default="errors_only",
        help="errors_only counts a row bad ONLY by ms thresholds (Plan B default; "
             "avoids self-reference when fn_voiced_miss / hard_failure are used as triggers). "
             "strict matches analyze_oto_fallback_candidates.py legacy.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=0,
        help="If >0, only keep the top-N candidates per slice in the output. 0 keeps all.",
    )
    args = parser.parse_args(argv)

    eval_path = os.path.abspath(args.eval)
    if not os.path.isfile(eval_path):
        print(f"[fallback-explore] eval JSON not found: {eval_path}", file=sys.stderr)
        return 2
    with open(eval_path, encoding="utf-8") as fh:
        eval_data = json.load(fh)
    files = eval_data.get("files") or []
    if not isinstance(files, list) or not files:
        print("[fallback-explore] eval JSON has no per-row 'files' array", file=sys.stderr)
        return 2

    raw_slices = list(args.slice) if args.slice else [
        "*/*/*",
        "korean/*/*",
        "japanese/*/*",
        "korean/cvvc/*",
        "japanese/cvvc/*",
        "korean/cvvc/vc",
        "korean/cvvc/vv",
        "korean/cvvc/other",
        "korean/cvc/*",
    ]

    grid = default_trigger_grid()
    out: dict[str, Any] = {
        "source_eval": eval_path,
        "total_rows": len(files),
        "bad_pre_ms": float(args.bad_pre_ms),
        "bad_offset_ms": float(args.bad_offset_ms),
        "bad_cutoff_ms": float(args.bad_cutoff_ms),
        "bad_mode": str(args.bad_mode),
        "trigger_grid": [t.name for t in grid],
        "slices": [],
    }

    for raw in raw_slices:
        lang, fmt, role = _parse_slice(raw)
        rows = _slice_rows(files, language=lang, format_type=fmt, alias_role=role)
        if not rows:
            continue
        candidates = evaluate_policy_grid(
            rows,
            grid=grid,
            bad_pre_ms=float(args.bad_pre_ms),
            bad_offset_ms=float(args.bad_offset_ms),
            bad_cutoff_ms=float(args.bad_cutoff_ms),
            bad_mode=str(args.bad_mode),
        )
        if int(args.top_candidates) > 0:
            candidates = candidates[: int(args.top_candidates)]
        # Bad rate from any single candidate (they all see the same rows).
        from core.coarse_crnn.oto_fallback_policy import _is_bad_row  # type: ignore
        bad_count = sum(
            1 for r in rows
            if _is_bad_row(
                r,
                bad_pre_ms=float(args.bad_pre_ms),
                bad_offset_ms=float(args.bad_offset_ms),
                bad_cutoff_ms=float(args.bad_cutoff_ms),
                mode=str(args.bad_mode),
            )
        )
        bad_rate = float(bad_count) / float(len(rows)) if rows else 0.0
        slice_block = {
            "slice": _slice_label(lang, fmt, role),
            "language": lang or "*",
            "format_type": fmt or "*",
            "alias_role": role or "*",
            "rows": len(rows),
            "bad_rate": bad_rate,
            "candidates": candidates,
            "best": candidates[0] if candidates else None,
        }
        out["slices"].append(slice_block)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    if args.markdown_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown_out)), exist_ok=True)
        with open(args.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(_render_markdown(out))

    # Compact stdout summary: best per slice.
    print(json.dumps(
        {
            "source_eval": eval_path,
            "out": os.path.abspath(args.out),
            "best_per_slice": [
                {
                    "slice": s["slice"],
                    "rows": s["rows"],
                    "bad_rate": s["bad_rate"],
                    "best": s.get("best"),
                }
                for s in out["slices"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
