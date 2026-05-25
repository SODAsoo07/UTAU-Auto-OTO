from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_context.oto_params import wav_duration_ms  # noqa: E402
from core.generation.common.oto_file_utils import read_text_with_fallback  # noqa: E402


ANCHORS = ("offset", "overlap", "preutterance", "fixed_end", "cutoff")
LANDMARK_KEYS = (
    "vowel_onset_ms",
    "stable_vowel_start_ms",
    "vowel_nucleus_ms",
    "vowel_offset_ms",
    "transition_tail_ms",
    "transition_valley_ms",
    "left_vowel_nucleus_ms",
    "right_vowel_onset_ms",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export manual-vs-generated residuals for manual_oto_anchor landmark calibration."
    )
    parser.add_argument("--anchors-json", required=True, help="manual_oto_anchor preview anchors.json")
    parser.add_argument("--generated-oto", required=True, help="Generated preview oto.ini")
    parser.add_argument("--reference-oto", required=True, help="Manual/reference oto.ini")
    parser.add_argument("--wav-dir", required=True, help="Directory containing referenced WAV files")
    parser.add_argument("--out-dir", required=True, help="Directory for residual_rows.csv/summary/report")
    args = parser.parse_args()

    anchors_json = Path(args.anchors_json)
    generated_oto = Path(args.generated_oto)
    reference_oto = Path(args.reference_oto)
    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_anchor_records(anchors_json)
    generated_rows = _load_oto(generated_oto, wav_dir=wav_dir)
    reference_rows = _load_oto(reference_oto, wav_dir=wav_dir)
    residual_rows = _build_residual_rows(generated_rows, reference_rows, records)
    summary = _summarize(residual_rows, generated_rows=generated_rows, reference_rows=reference_rows)

    _write_csv(out_dir / "residual_rows.csv", residual_rows)
    (out_dir / "residual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "residual_report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_text(path: Path) -> str:
    return read_text_with_fallback(str(path))


def _load_anchor_records(path: Path) -> dict[tuple[str, str], deque[dict[str, object]]]:
    payload = json.loads(_read_text(path))
    out: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(deque)
    for wav_record in payload.get("rows", []) if isinstance(payload, Mapping) else []:
        if not isinstance(wav_record, Mapping):
            continue
        wav = str(wav_record.get("wav") or "")
        for row in wav_record.get("rows", []) or []:
            if not isinstance(row, Mapping):
                continue
            alias = str(row.get("alias") or "")
            out[(wav.lower(), alias)].append(dict(row))
    return out


def _load_oto(path: Path, *, wav_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    duration_cache: dict[str, float] = {}
    for line_no, raw_line in enumerate(_read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        wav_name, rest = line.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 6:
            continue
        alias = parts[0]
        try:
            offset, consonant, cutoff, preutterance, overlap = [float(value or 0.0) for value in parts[1:6]]
        except Exception:
            continue
        key = wav_name.lower()
        if key not in duration_cache:
            duration_cache[key] = wav_duration_ms(str(wav_dir / wav_name), missing_value=0.0)
        duration_ms = float(duration_cache[key])
        cutoff_abs = duration_ms + cutoff if cutoff < 0.0 else offset + cutoff
        rows.append(
            {
                "line_no": int(line_no),
                "wav": wav_name,
                "alias": alias,
                "duration_ms": duration_ms,
                "offset_abs": offset,
                "overlap_abs": offset + overlap,
                "preutterance_abs": offset + preutterance,
                "fixed_end_abs": offset + consonant,
                "cutoff_abs": cutoff_abs,
                "offset": offset,
                "consonant": consonant,
                "cutoff": cutoff,
                "preutterance": preutterance,
                "overlap": overlap,
            }
        )
    return rows


def _build_residual_rows(
    generated_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    anchor_records: dict[tuple[str, str], deque[dict[str, object]]],
) -> list[dict[str, object]]:
    ref_by_key: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(deque)
    for row in reference_rows:
        ref_by_key[(str(row["wav"]).lower(), str(row["alias"]))].append(row)
    out: list[dict[str, object]] = []
    for generated in generated_rows:
        key = (str(generated["wav"]).lower(), str(generated["alias"]))
        if not ref_by_key[key]:
            continue
        reference = ref_by_key[key].popleft()
        anchor_record = anchor_records[key].popleft() if anchor_records[key] else {}
        landmarks = anchor_record.get("acoustic_landmarks") if isinstance(anchor_record, Mapping) else {}
        if not isinstance(landmarks, Mapping):
            landmarks = {}
        role = str(anchor_record.get("alias_role") or "") if isinstance(anchor_record, Mapping) else ""
        source = str(landmarks.get("source") or "")
        row: dict[str, object] = {
            "wav": generated["wav"],
            "alias": generated["alias"],
            "role": role,
            "landmark_source": source,
            "landmark_confidence": _float_or_blank(landmarks.get("confidence")),
            "landmark_warnings": "|".join(str(item) for item in landmarks.get("warnings", []) or []),
        }
        for anchor in ANCHORS:
            gen_abs = float(generated[f"{anchor}_abs"])
            ref_abs = float(reference[f"{anchor}_abs"])
            row[f"{anchor}_generated_abs_ms"] = gen_abs
            row[f"{anchor}_reference_abs_ms"] = ref_abs
            row[f"{anchor}_signed_error_ms"] = gen_abs - ref_abs
            row[f"{anchor}_error_ms"] = abs(gen_abs - ref_abs)
        for key_name in LANDMARK_KEYS:
            landmark = _float_or_none(landmarks.get(key_name))
            row[key_name] = "" if landmark is None else landmark
            if landmark is not None:
                row[f"manual_pre_minus_{key_name}"] = float(reference["preutterance_abs"]) - landmark
                row[f"manual_fixed_minus_{key_name}"] = float(reference["fixed_end_abs"]) - landmark
                row[f"manual_cutoff_minus_{key_name}"] = float(reference["cutoff_abs"]) - landmark
            else:
                row[f"manual_pre_minus_{key_name}"] = ""
                row[f"manual_fixed_minus_{key_name}"] = ""
                row[f"manual_cutoff_minus_{key_name}"] = ""
        out.append(row)
    return out


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _float_or_blank(value: object) -> float | str:
    out = _float_or_none(value)
    return "" if out is None else out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict[str, object]],
    *,
    generated_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "matched_rows": len(rows),
        "generated_rows": len(generated_rows),
        "reference_rows": len(reference_rows),
        "anchor_errors": {anchor: _metric(_numeric_values(row.get(f"{anchor}_error_ms") for row in rows)) for anchor in ANCHORS},
        "primary_max": _metric(
            max(float(row.get(f"{anchor}_error_ms", 0.0) or 0.0) for anchor in ANCHORS)
            for row in rows
        ),
        "likely_slot_shift": _likely_slot_shift_summary(rows),
        "failure_breakdown": _failure_breakdown(rows),
        "by_role": {},
        "by_landmark_source": {},
        "residuals": {},
    }
    for role in sorted({str(row.get("role") or "unknown") for row in rows}):
        group = [row for row in rows if str(row.get("role") or "unknown") == role]
        summary["by_role"][role] = {
            "rows": len(group),
            "preutterance": _metric(_numeric_values(row.get("preutterance_error_ms") for row in group)),
            "primary_max": _metric(
                max(float(row.get(f"{anchor}_error_ms", 0.0) or 0.0) for anchor in ANCHORS)
                for row in group
            ),
        }
    for source in sorted({str(row.get("landmark_source") or "none") for row in rows}):
        group = [row for row in rows if str(row.get("landmark_source") or "none") == source]
        summary["by_landmark_source"][source] = {
            "rows": len(group),
            "preutterance": _metric(_numeric_values(row.get("preutterance_error_ms") for row in group)),
        }
    for key_name in LANDMARK_KEYS:
        residual_key = f"manual_pre_minus_{key_name}"
        values = _numeric_values(row.get(residual_key) for row in rows)
        if values:
            summary["residuals"][residual_key] = _metric(values)
    return summary


def _numeric_values(values: Iterable[object]) -> list[float]:
    out: list[float] = []
    for value in values:
        item = _float_or_none(value)
        if item is not None:
            out.append(float(item))
    return out


def _metric(values: Iterable[float]) -> dict[str, object]:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        "median_ms": float(np.median(arr)),
        "median_abs_ms": float(np.median(abs_arr)),
        "p90_abs_ms": float(np.percentile(abs_arr, 90)),
        "mean_ms": float(np.mean(arr)),
        "mean_abs_ms": float(np.mean(abs_arr)),
        "pass_le10_rate": float(np.mean(abs_arr <= 10.0)),
        "minor_10_30_rate": float(np.mean((abs_arr > 10.0) & (abs_arr <= 30.0))),
        "ordinary_30_60_rate": float(np.mean((abs_arr > 30.0) & (abs_arr < 60.0))),
        "review_ge60_rate": float(np.mean(abs_arr >= 60.0)),
        "warning_60_80_rate": float(np.mean((abs_arr > 60.0) & (abs_arr <= 80.0))),
        "fail_gt_80ms_rate": float(np.mean(abs_arr > 80.0)),
        "pass_le10": int(np.sum(abs_arr <= 10.0)),
        "minor_10_30": int(np.sum((abs_arr > 10.0) & (abs_arr <= 30.0))),
        "ordinary_30_60": int(np.sum((abs_arr > 30.0) & (abs_arr < 60.0))),
        "warning_60_80": int(np.sum((abs_arr > 60.0) & (abs_arr <= 80.0))),
        "fail_gt_80": int(np.sum(abs_arr > 80.0)),
    }


def _likely_slot_shift_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_wav: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_wav[str(row.get("wav") or "")].append(row)
    total = 0
    one_step = 0
    multi_step = 0
    exact_or_near = 0
    for wav_rows in by_wav.values():
        ref_times = sorted(
            {
                float(row.get("preutterance_reference_abs_ms", 0.0) or 0.0)
                for row in wav_rows
            }
        )
        gaps = [right - left for left, right in zip(ref_times, ref_times[1:]) if (right - left) >= 120.0]
        if not gaps:
            continue
        gap = float(np.median(np.asarray(gaps, dtype=np.float64)))
        if gap <= 0.0:
            continue
        for row in wav_rows:
            signed = _float_or_none(row.get("preutterance_signed_error_ms"))
            if signed is None:
                continue
            total += 1
            if abs(signed) <= 80.0:
                exact_or_near += 1
                continue
            slot_delta = int(round(float(signed) / gap))
            if abs(slot_delta) == 1 and abs(abs(float(signed)) - gap) <= max(90.0, 0.36 * gap):
                one_step += 1
            elif abs(slot_delta) >= 2:
                multi_step += 1
    return {
        "total": int(total),
        "exact_or_near": int(exact_or_near),
        "one_step_shift": int(one_step),
        "multi_step_shift": int(multi_step),
        "one_step_shift_rate": float(one_step / total) if total else 0.0,
        "multi_step_shift_rate": float(multi_step / total) if total else 0.0,
    }


def _failure_breakdown(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"top_primary_fail_gt80": []}
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        primary = max(float(row.get(f"{anchor}_error_ms", 0.0) or 0.0) for anchor in ANCHORS)
        if primary <= 80.0:
            continue
        warnings = str(row.get("landmark_warnings") or "none").split("|")
        warning = warnings[0] if warnings and warnings[0] else "none"
        key = (str(row.get("role") or "unknown"), str(row.get("landmark_source") or "none"), warning)
        groups[key].append(float(primary))
    ranked = sorted(groups.items(), key=lambda item: (len(item[1]), float(np.median(item[1]))), reverse=True)
    for (role, source, warning), values in ranked[:30]:
        metric = _metric(values)
        out["top_primary_fail_gt80"].append(
            {
                "role": role,
                "landmark_source": source,
                "warning": warning,
                "rows": int(len(values)),
                "median_abs_ms": float(metric.get("median_abs_ms", 0.0)),
                "p90_abs_ms": float(metric.get("p90_abs_ms", 0.0)),
            }
        )
    return out


def _render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# manual_oto_anchor residual calibration report",
        "",
        f"matched_rows={summary.get('matched_rows')} generated_rows={summary.get('generated_rows')} reference_rows={summary.get('reference_rows')}",
        "",
        "## Anchor Errors",
    ]
    anchor_errors = summary.get("anchor_errors") if isinstance(summary.get("anchor_errors"), Mapping) else {}
    for anchor in ANCHORS:
        metric = anchor_errors.get(anchor) if isinstance(anchor_errors, Mapping) else None
        if not isinstance(metric, Mapping) or not metric.get("count"):
            continue
        lines.append(
            f"- {anchor}: median_abs={float(metric.get('median_abs_ms', 0.0)):.2f}ms "
            f"p90_abs={float(metric.get('p90_abs_ms', 0.0)):.2f}ms "
            f"pass_le10={float(metric.get('pass_le10_rate', 0.0)) * 100.0:.2f}% "
            f"fail_gt_80={float(metric.get('fail_gt_80ms_rate', 0.0)) * 100.0:.2f}%"
        )
    primary = summary.get("primary_max") if isinstance(summary.get("primary_max"), Mapping) else {}
    if primary:
        lines.append(
            f"- primary_max: median_abs={float(primary.get('median_abs_ms', 0.0)):.2f}ms "
            f"p90_abs={float(primary.get('p90_abs_ms', 0.0)):.2f}ms "
            f"fail_gt_80={float(primary.get('fail_gt_80ms_rate', 0.0)) * 100.0:.2f}%"
        )
    slot_shift = summary.get("likely_slot_shift") if isinstance(summary.get("likely_slot_shift"), Mapping) else {}
    if slot_shift:
        lines.extend(
            [
                "",
                "## Likely Slot Shift",
                f"- one_step={int(slot_shift.get('one_step_shift', 0) or 0)}/{int(slot_shift.get('total', 0) or 0)} "
                f"({float(slot_shift.get('one_step_shift_rate', 0.0) or 0.0) * 100.0:.2f}%)",
                f"- multi_step={int(slot_shift.get('multi_step_shift', 0) or 0)}/{int(slot_shift.get('total', 0) or 0)} "
                f"({float(slot_shift.get('multi_step_shift_rate', 0.0) or 0.0) * 100.0:.2f}%)",
            ]
        )
    lines.extend(["", "## Landmark Residuals"])
    residuals = summary.get("residuals") if isinstance(summary.get("residuals"), Mapping) else {}
    for key, metric in sorted(residuals.items()):
        if not isinstance(metric, Mapping) or not metric.get("count"):
            continue
        lines.append(
            f"- {key}: median={float(metric.get('median_ms', 0.0)):.2f}ms "
            f"median_abs={float(metric.get('median_abs_ms', 0.0)):.2f}ms "
            f"p90_abs={float(metric.get('p90_abs_ms', 0.0)):.2f}ms"
        )
    breakdown = summary.get("failure_breakdown") if isinstance(summary.get("failure_breakdown"), Mapping) else {}
    top = breakdown.get("top_primary_fail_gt80") if isinstance(breakdown.get("top_primary_fail_gt80"), list) else []
    if top:
        lines.extend(["", "## Top Primary >80ms Failure Groups"])
        for item in top[:12]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- role={item.get('role')} source={item.get('landmark_source')} warning={item.get('warning')} "
                f"rows={item.get('rows')} median_abs={float(item.get('median_abs_ms', 0.0)):.2f}ms "
                f"p90_abs={float(item.get('p90_abs_ms', 0.0)):.2f}ms"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
