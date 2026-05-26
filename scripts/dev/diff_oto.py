"""Diff two oto.ini files row-by-row (matched by wav+alias occurrence).
Reports per-(wav,alias,occurrence) offset/preutterance/overlap/consonant/cutoff deltas,
plus per-parameter aggregates (mean/median/MAE). Treats `--gold` as truth."""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.generation.common.oto_alias_family import alias_family


def load(path):
    text = read_text_with_fallback(path)
    parsed_rows = []
    for line in text.splitlines():
        parsed = parse_oto_line(line)
        if not parsed: continue
        base_key = (str(parsed["wav"]).strip().lower(), str(parsed["alias"]).strip())
        parsed_rows.append((base_key, parsed))

    base_counts = Counter(base_key for base_key, _parsed in parsed_rows)
    occurrence_counts = defaultdict(int)
    rows = {}
    for base_key, parsed in parsed_rows:
        occurrence = occurrence_counts[base_key]
        occurrence_counts[base_key] += 1
        key = (*base_key, occurrence)
        rows[key] = {
            "offset": float(parsed["offset"]),
            "consonant": float(parsed["cons"]),
            "cutoff": float(parsed["cutoff"]),
            "preutterance": float(parsed["pre"]),
            "overlap": float(parsed["ovl"]),
            "occurrence": occurrence,
            "duplicate_group_size": int(base_counts[base_key]),
        }
    return rows


def _wav_duration_ms(wav_dir: str, wav_name: str) -> float:
    if not wav_dir or not wav_name:
        return 0.0
    path = os.path.join(wav_dir, wav_name)
    if not os.path.isfile(path):
        return 0.0
    try:
        with wave.open(path, "rb") as handle:
            rate = max(1, int(handle.getframerate()))
            frames = max(0, int(handle.getnframes()))
            return 1000.0 * frames / float(rate)
    except Exception:
        return 0.0


def _print_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _cutoff_end_candidates(row: dict, duration_ms: float, mode: str) -> list[tuple[str, float]]:
    """Return possible absolute cutoff/end positions for one OTO row.

    `relative` follows the local AutoOTO generated-output convention used by
    current bootstrap rows: a negative field is stored as an offset-relative
    active length, but with a negative sign for UTAU-friendly output.

    `utau` follows the canonical OTO right-blank convention used by the manual
    anchor tooling: negative values are measured from the wav end, while
    positive values are offset-relative.

    `compatible` is an evaluation-only diagnostic for mixed legacy banks. It
    does not claim the generated OTO will be interpreted that way by UTAU.
    """
    offset = float(row["offset"])
    cutoff = float(row["cutoff"])
    duration = max(0.0, float(duration_ms or 0.0))
    mode = str(mode or "raw").strip().lower()
    candidates: list[tuple[str, float]] = []

    if mode == "relative":
        if cutoff < 0.0:
            candidates.append(("negative_relative_to_offset", offset + abs(cutoff)))
        else:
            candidates.append(("positive_relative_to_offset", offset + cutoff))
    elif mode == "utau":
        if cutoff < 0.0 and duration > 0.0:
            candidates.append(("negative_right_blank_from_end", duration + cutoff))
        elif cutoff < 0.0:
            candidates.append(("negative_relative_to_offset_fallback", offset + abs(cutoff)))
        else:
            candidates.append(("positive_relative_to_offset", offset + cutoff))
    elif mode == "compatible":
        if cutoff < 0.0:
            candidates.append(("negative_relative_to_offset", offset + abs(cutoff)))
            if duration > 0.0:
                candidates.append(("negative_right_blank_from_end", duration + cutoff))
        else:
            candidates.append(("positive_relative_to_offset", offset + cutoff))
            candidates.append(("positive_absolute_end", cutoff))
            if duration > 0.0:
                candidates.append(("positive_right_blank_from_end", duration - cutoff))
    else:
        raise ValueError(f"unknown cutoff end mode: {mode}")

    valid: list[tuple[str, float]] = []
    for label, value in candidates:
        end = float(value)
        if end < offset + 1.0:
            continue
        if duration > 0.0 and end > duration + 50.0:
            continue
        valid.append((label, end))
    return valid or candidates


def _cutoff_end_delta(
    gold_row: dict,
    pred_row: dict,
    *,
    duration_ms: float,
    mode: str,
) -> tuple[float, dict]:
    gold_candidates = _cutoff_end_candidates(gold_row, duration_ms, mode)
    pred_candidates = _cutoff_end_candidates(pred_row, duration_ms, mode)
    best = min(
        (
            (pred_end - gold_end, gold_label, pred_label, gold_end, pred_end)
            for gold_label, gold_end in gold_candidates
            for pred_label, pred_end in pred_candidates
        ),
        key=lambda item: abs(float(item[0])),
    )
    delta, gold_mode, pred_mode, gold_end, pred_end = best
    return float(delta), {
        "gold_cutoff_end_mode": gold_mode,
        "pred_cutoff_end_mode": pred_mode,
        "gold_cutoff_end_abs": float(gold_end),
        "pred_cutoff_end_abs": float(pred_end),
    }


def build_report(
    *,
    gold_path: str,
    pred_path: str,
    cutoff_end_mode: str = "raw",
    wav_dir: str = "",
) -> tuple[dict, list[dict]]:
    gold = load(gold_path); pred = load(pred_path)
    gk = set(gold.keys()); pk = set(pred.keys())
    common = sorted(gk & pk)
    only_gold = sorted(gk - pk); only_pred = sorted(pk - gk)

    params = ["offset", "preutterance", "overlap", "consonant", "cutoff"]
    diffs = defaultdict(list)
    diffs_by_family = defaultdict(lambda: defaultdict(list))
    cutoff_end_modes = defaultdict(int)
    per_row = []
    include_cutoff_end = str(cutoff_end_mode or "raw").strip().lower() != "raw"
    for key in common:
        g, p = gold[key], pred[key]
        d = {param: p[param] - g[param] for param in params}
        family = alias_family(key[1])
        row_extra = {}
        if include_cutoff_end:
            duration_ms = _wav_duration_ms(wav_dir, key[0])
            cutoff_delta, cutoff_info = _cutoff_end_delta(
                g,
                p,
                duration_ms=duration_ms,
                mode=cutoff_end_mode,
            )
            d["cutoff_end_abs"] = cutoff_delta
            row_extra["duration_ms"] = duration_ms
            row_extra["cutoff_end"] = cutoff_info
            cutoff_end_modes[
                f"{cutoff_info['gold_cutoff_end_mode']}->{cutoff_info['pred_cutoff_end_mode']}"
            ] += 1
        for param, v in d.items():
            diffs[param].append(v)
            diffs_by_family[family][param].append(v)
        row_payload = {
            "wav": key[0],
            "alias": key[1],
            "alias_family": family,
            "occurrence": int(key[2]),
            "gold": g,
            "pred": p,
            "delta": d,
        }
        row_payload.update(row_extra)
        per_row.append(row_payload)

    def stats(vals):
        if not vals: return {}
        absv = [abs(v) for v in vals]
        s = sorted(absv)
        return {
            "n": len(vals),
            "mean_delta_ms": round(sum(vals)/len(vals), 1),
            "abs_median_ms": round(s[len(s)//2], 1),
            "abs_p90_ms": round(s[int(0.9*(len(s)-1))], 1),
            "MAE_ms": round(sum(absv)/len(absv), 1),
        }

    report_params = [*params, "cutoff_end_abs"] if include_cutoff_end else params
    duplicate_groups_gold = sum(1 for row in gold.values() if int(row.get("occurrence", 0)) == 0 and int(row.get("duplicate_group_size", 1)) > 1)
    duplicate_groups_pred = sum(1 for row in pred.values() if int(row.get("occurrence", 0)) == 0 and int(row.get("duplicate_group_size", 1)) > 1)
    report = {
        "gold": os.path.abspath(gold_path),
        "pred": os.path.abspath(pred_path),
        "rows_gold": len(gold), "rows_pred": len(pred),
        "rows_matched": len(common),
        "only_in_gold": len(only_gold), "only_in_pred": len(only_pred),
        "duplicate_key_groups_gold": duplicate_groups_gold,
        "duplicate_key_groups_pred": duplicate_groups_pred,
        "by_param": {param: stats(diffs[param]) for param in report_params},
        "by_alias_family": {
            family: {
                "n": int(max((len(values) for values in param_map.values()), default=0)),
                "by_param": {param: stats(param_map[param]) for param in report_params},
            }
            for family, param_map in sorted(diffs_by_family.items())
        },
    }
    if include_cutoff_end:
        report["cutoff_end_mode"] = str(cutoff_end_mode)
        report["wav_dir"] = os.path.abspath(wav_dir) if wav_dir else ""
        report["cutoff_end_mode_pairs"] = dict(sorted(cutoff_end_modes.items()))
    return report, per_row


def build_output_report(report: dict, per_row: list[dict], *, per_row_limit: int = 60) -> dict:
    payload = dict(report)
    payload["per_row_truncated"] = list(per_row[: max(0, int(per_row_limit))])
    payload["per_row_truncated_count"] = len(payload["per_row_truncated"])
    payload["per_row_total"] = len(per_row)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, help="Reference oto.ini")
    ap.add_argument("--pred", required=True, help="Predicted oto.ini to compare against gold")
    ap.add_argument("--wav-dir", default="", help="Directory containing wav files for cutoff end-position comparison")
    ap.add_argument(
        "--cutoff-end-mode",
        choices=("raw", "relative", "utau", "compatible"),
        default="raw",
        help=(
            "Add cutoff_end_abs metric. relative treats negative cutoff as offset-relative active length; "
            "utau treats negative cutoff as right blank from wav end; compatible picks the closest valid "
            "mixed legacy interpretation for diagnostics."
        ),
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wav_dir = args.wav_dir or os.path.dirname(os.path.abspath(args.gold))
    report, per_row = build_report(
        gold_path=args.gold,
        pred_path=args.pred,
        cutoff_end_mode=args.cutoff_end_mode,
        wav_dir=wav_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    _print_text(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        output_report = build_output_report(report, per_row)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(output_report, ensure_ascii=False, indent=2))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
