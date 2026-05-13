from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from core.coarse_crnn.oto_evaluate import _analyze_activity_profile, _is_hard_failure
from core.coarse_crnn.oto_targets import resolve_cutoff_abs_ms
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


_SILENCE_TOKENS = {"r", "br", "pau", "sil", "rest"}


@dataclass(frozen=True)
class OtoRow:
    wav: str
    alias: str
    line_index: int
    offset: float
    cons: float
    cutoff: float
    pre: float
    ovl: float

    def pre_abs(self) -> float:
        return float(self.offset) + max(0.0, float(self.pre))

    def con_abs(self) -> float:
        return float(self.offset) + max(0.0, float(self.cons))

    def cutoff_abs(self, duration_ms: float) -> float:
        return float(
            resolve_cutoff_abs_ms(
                offset_ms=float(self.offset),
                cutoff=float(self.cutoff),
                duration_ms=max(1.0, float(duration_ms)),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B compare generated oto.ini files with practical error buckets: "
            "leading-silence concentration, one-step shift, and mis-mapping."
        )
    )
    parser.add_argument("--a-oto", required=True)
    parser.add_argument("--b-oto", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--source-oto", default="", help="Optional base/reference oto.ini for row-index mapping checks.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--front-margin-ms", type=float, default=90.0)
    parser.add_argument("--mapping-match-max-ms", type=float, default=120.0)
    parser.add_argument("--mapping-mismatch-min-ms", type=float, default=120.0)
    parser.add_argument("--mapping-severe-abs-ms", type=float, default=260.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a_rows = _read_oto_rows(str(args.a_oto))
    b_rows = _read_oto_rows(str(args.b_oto))
    src_rows = _read_oto_rows(str(args.source_oto)) if str(args.source_oto or "").strip() else {}

    profile_cache: dict[str, dict[str, float]] = {}
    a_eval = _evaluate_system(
        system_name="A",
        rows_by_wav=a_rows,
        source_rows_by_wav=src_rows,
        wav_dir=str(args.wav_dir),
        sample_rate=int(args.sample_rate),
        front_margin_ms=float(args.front_margin_ms),
        mapping_match_max_ms=float(args.mapping_match_max_ms),
        mapping_mismatch_min_ms=float(args.mapping_mismatch_min_ms),
        mapping_severe_abs_ms=float(args.mapping_severe_abs_ms),
        profile_cache=profile_cache,
    )
    b_eval = _evaluate_system(
        system_name="B",
        rows_by_wav=b_rows,
        source_rows_by_wav=src_rows,
        wav_dir=str(args.wav_dir),
        sample_rate=int(args.sample_rate),
        front_margin_ms=float(args.front_margin_ms),
        mapping_match_max_ms=float(args.mapping_match_max_ms),
        mapping_mismatch_min_ms=float(args.mapping_mismatch_min_ms),
        mapping_severe_abs_ms=float(args.mapping_severe_abs_ms),
        profile_cache=profile_cache,
    )

    report = _build_report(
        a_eval=a_eval,
        b_eval=b_eval,
        a_path=str(args.a_oto),
        b_path=str(args.b_oto),
        source_path=str(args.source_oto or ""),
        top=max(1, int(args.top)),
    )
    os.makedirs(os.path.dirname(os.path.abspath(str(args.out))), exist_ok=True)
    with open(str(args.out), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    return 0


def _read_oto_rows(path: str) -> dict[str, list[OtoRow]]:
    out: dict[str, list[OtoRow]] = {}
    if not path or not os.path.isfile(path):
        return out
    for raw in read_text_with_fallback(path).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav = os.path.basename(str(parsed.get("wav", "") or "").replace("\\", "/")).strip().lower()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav or not alias:
            continue
        bucket = out.setdefault(wav, [])
        bucket.append(
            OtoRow(
                wav=wav,
                alias=alias,
                line_index=int(len(bucket)),
                offset=float(parsed.get("offset", 0.0) or 0.0),
                cons=float(parsed.get("cons", 0.0) or 0.0),
                cutoff=float(parsed.get("cutoff", 0.0) or 0.0),
                pre=float(parsed.get("pre", 0.0) or 0.0),
                ovl=float(parsed.get("ovl", 0.0) or 0.0),
            )
        )
    return out


def _evaluate_system(
    *,
    system_name: str,
    rows_by_wav: dict[str, list[OtoRow]],
    source_rows_by_wav: dict[str, list[OtoRow]],
    wav_dir: str,
    sample_rate: int,
    front_margin_ms: float,
    mapping_match_max_ms: float,
    mapping_mismatch_min_ms: float,
    mapping_severe_abs_ms: float,
    profile_cache: dict[str, dict[str, float]],
) -> dict[str, Any]:
    per_wav: list[dict[str, Any]] = []
    totals = {
        "rows_total": 0,
        "rows_non_silence": 0,
        "front_silence_count": 0,
        "hard_failure_count": 0,
        "sequence_anomaly_count": 0,
        "one_step_shift_count": 0,
        "mis_mapping_count": 0,
    }
    mapped_supported_rows = 0

    for wav, rows in sorted(rows_by_wav.items()):
        wav_path = os.path.join(str(wav_dir), wav).replace("/", os.sep)
        profile = _analyze_activity_profile(
            wav_path,
            sample_rate=int(sample_rate),
            cache=profile_cache,
        )
        duration_ms = float(profile.get("duration_ms", 0.0) or 0.0)
        active_start_ms = float(profile.get("active_start_ms", 0.0) or 0.0)
        src_rows = source_rows_by_wav.get(wav, [])
        src_pre_by_index = [r.pre_abs() for r in src_rows if not _is_blank_or_silence(r.alias)]

        wav_counts = {
            "rows_total": len(rows),
            "rows_non_silence": 0,
            "front_silence_count": 0,
            "hard_failure_count": 0,
            "sequence_anomaly_count": 0,
            "one_step_shift_count": 0,
            "mis_mapping_count": 0,
            "mapping_supported_rows": 0,
        }
        row_details: list[dict[str, Any]] = []
        non_sil_idx = -1
        for row in rows:
            is_sil = _is_blank_or_silence(row.alias)
            if is_sil:
                row_details.append(
                    {
                        "alias": row.alias,
                        "line_index": row.line_index,
                        "silence_like": True,
                    }
                )
                continue

            non_sil_idx += 1
            wav_counts["rows_non_silence"] += 1
            totals["rows_non_silence"] += 1
            pre_abs = row.pre_abs()
            con_abs = row.con_abs()
            mid_abs = (pre_abs + con_abs) * 0.5
            cutoff_abs = row.cutoff_abs(duration_ms=max(1.0, duration_ms))
            anchors = {
                "offset": float(row.offset),
                "overlap": float(row.offset + max(0.0, row.ovl)),
                "preutterance": pre_abs,
                "consonant": con_abs,
                "cutoff": cutoff_abs,
            }
            hard_fail, hard_reason = _is_hard_failure(
                anchors,
                duration_ms=max(1.0, duration_ms),
                activity_profile=profile,
            )
            front = bool(mid_abs < (active_start_ms - float(front_margin_ms)))

            one_step = False
            mis_map = False
            nearest_idx = -1
            nearest_delta_ms = None
            own_delta_ms = None
            if src_pre_by_index and non_sil_idx < len(src_pre_by_index):
                wav_counts["mapping_supported_rows"] += 1
                mapped_supported_rows += 1
                own_delta_ms = abs(float(pre_abs) - float(src_pre_by_index[non_sil_idx]))
                nearest_idx, nearest_delta_ms = _nearest_index(src_pre_by_index, float(pre_abs))
                idx_gap = abs(int(nearest_idx) - int(non_sil_idx))
                nearest_ok = float(nearest_delta_ms) <= float(mapping_match_max_ms)
                own_bad = float(own_delta_ms) >= float(mapping_mismatch_min_ms)
                if idx_gap == 1 and own_bad and nearest_ok:
                    one_step = True
                if (idx_gap >= 1 and own_bad and nearest_ok) or float(own_delta_ms) >= float(mapping_severe_abs_ms):
                    mis_map = True

            wav_counts["front_silence_count"] += 1 if front else 0
            wav_counts["hard_failure_count"] += 1 if hard_fail else 0
            wav_counts["one_step_shift_count"] += 1 if one_step else 0
            wav_counts["mis_mapping_count"] += 1 if mis_map else 0
            totals["front_silence_count"] += 1 if front else 0
            totals["hard_failure_count"] += 1 if hard_fail else 0
            totals["one_step_shift_count"] += 1 if one_step else 0
            totals["mis_mapping_count"] += 1 if mis_map else 0

            row_details.append(
                {
                    "alias": row.alias,
                    "line_index": row.line_index,
                    "silence_like": False,
                    "pre_abs_ms": pre_abs,
                    "con_abs_ms": con_abs,
                    "mid_abs_ms": mid_abs,
                    "active_start_ms": active_start_ms,
                    "front_silence": front,
                    "hard_failure": bool(hard_fail),
                    "hard_failure_reason": str(hard_reason),
                    "one_step_shift": bool(one_step),
                    "mis_mapping": bool(mis_map),
                    "mapping_nearest_index": int(nearest_idx),
                    "mapping_nearest_delta_ms": nearest_delta_ms,
                    "mapping_own_index": int(non_sil_idx),
                    "mapping_own_delta_ms": own_delta_ms,
                }
            )

        totals["rows_total"] += len(rows)
        if wav_counts["rows_non_silence"] <= 0:
            continue
        per_wav.append(
            {
                "wav": wav,
                "wav_path": os.path.abspath(wav_path),
                "duration_ms": duration_ms,
                "active_start_ms": active_start_ms,
                "counts": wav_counts,
                "rates": {
                    "front_silence_rate": _rate(wav_counts["front_silence_count"], wav_counts["rows_non_silence"]),
                    "hard_failure_rate": _rate(wav_counts["hard_failure_count"], wav_counts["rows_non_silence"]),
                    "sequence_anomaly_rate": 0.0,
                    "one_step_shift_rate": _rate(wav_counts["one_step_shift_count"], wav_counts["mapping_supported_rows"]),
                    "mis_mapping_rate": _rate(wav_counts["mis_mapping_count"], wav_counts["mapping_supported_rows"]),
                },
                "rows": row_details,
            }
        )
        _apply_sequence_anomaly_counts(
            wav_payload=per_wav[-1],
            totals=totals,
        )

    overall = {
        "system": system_name,
        "counts": {**totals, "mapping_supported_rows": int(mapped_supported_rows)},
        "rates": {
            "front_silence_rate": _rate(totals["front_silence_count"], totals["rows_non_silence"]),
            "hard_failure_rate": _rate(totals["hard_failure_count"], totals["rows_non_silence"]),
            "sequence_anomaly_rate": _rate(totals["sequence_anomaly_count"], totals["rows_non_silence"]),
            "one_step_shift_rate": _rate(totals["one_step_shift_count"], mapped_supported_rows),
            "mis_mapping_rate": _rate(totals["mis_mapping_count"], mapped_supported_rows),
        },
    }
    return {"overall": overall, "per_wav": per_wav}


def _build_report(
    *,
    a_eval: dict[str, Any],
    b_eval: dict[str, Any],
    a_path: str,
    b_path: str,
    source_path: str,
    top: int,
) -> dict[str, Any]:
    metric_keys = (
        "front_silence_rate",
        "hard_failure_rate",
        "sequence_anomaly_rate",
        "one_step_shift_rate",
        "mis_mapping_rate",
    )
    delta = {
        key: float((b_eval["overall"]["rates"] or {}).get(key, 0.0) or 0.0)
        - float((a_eval["overall"]["rates"] or {}).get(key, 0.0) or 0.0)
        for key in metric_keys
    }
    by_wav = _merge_wav_delta(a_eval.get("per_wav") or [], b_eval.get("per_wav") or [])
    return {
        "a_path": os.path.abspath(a_path),
        "b_path": os.path.abspath(b_path),
        "source_path": os.path.abspath(source_path) if source_path else "",
        "definition": {
            "front_silence": "mid(preutterance,consonant) before active_start - front_margin_ms",
            "sequence_anomaly": "non-silence row-to-row preutterance step is backward or excessively large vs median step",
            "one_step_shift": "nearest source-row index differs by exactly 1 with own-row delta large and nearest-row delta small",
            "mis_mapping": "row maps to non-own source row or has severe own-row preutterance delta",
        },
        "a": a_eval,
        "b": b_eval,
        "delta_b_minus_a": delta,
        "worst_wavs_a": _top_wavs(a_eval.get("per_wav") or [], top=top),
        "worst_wavs_b": _top_wavs(b_eval.get("per_wav") or [], top=top),
        "wav_delta_b_minus_a": by_wav[:top],
    }


def _merge_wav_delta(a_wavs: list[dict[str, Any]], b_wavs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_map = {str(item.get("wav", "") or ""): item for item in a_wavs}
    b_map = {str(item.get("wav", "") or ""): item for item in b_wavs}
    out: list[dict[str, Any]] = []
    for wav in sorted(set(a_map.keys()) | set(b_map.keys())):
        a = a_map.get(wav) or {}
        b = b_map.get(wav) or {}
        a_rates = dict(a.get("rates") or {})
        b_rates = dict(b.get("rates") or {})
        out.append(
            {
                "wav": wav,
                "delta_front_silence_rate": float(b_rates.get("front_silence_rate", 0.0) or 0.0)
                - float(a_rates.get("front_silence_rate", 0.0) or 0.0),
                "delta_hard_failure_rate": float(b_rates.get("hard_failure_rate", 0.0) or 0.0)
                - float(a_rates.get("hard_failure_rate", 0.0) or 0.0),
                "delta_one_step_shift_rate": float(b_rates.get("one_step_shift_rate", 0.0) or 0.0)
                - float(a_rates.get("one_step_shift_rate", 0.0) or 0.0),
                "delta_mis_mapping_rate": float(b_rates.get("mis_mapping_rate", 0.0) or 0.0)
                - float(a_rates.get("mis_mapping_rate", 0.0) or 0.0),
                "delta_sequence_anomaly_rate": float(b_rates.get("sequence_anomaly_rate", 0.0) or 0.0)
                - float(a_rates.get("sequence_anomaly_rate", 0.0) or 0.0),
                "a_rows_non_silence": int(((a.get("counts") or {}).get("rows_non_silence", 0) or 0)),
                "b_rows_non_silence": int(((b.get("counts") or {}).get("rows_non_silence", 0) or 0)),
            }
        )
    out.sort(
        key=lambda item: (
            abs(float(item.get("delta_mis_mapping_rate", 0.0) or 0.0))
            + abs(float(item.get("delta_one_step_shift_rate", 0.0) or 0.0))
            + abs(float(item.get("delta_sequence_anomaly_rate", 0.0) or 0.0))
            + abs(float(item.get("delta_front_silence_rate", 0.0) or 0.0))
        ),
        reverse=True,
    )
    return out


def _top_wavs(items: list[dict[str, Any]], *, top: int) -> list[dict[str, Any]]:
    rows = list(items)
    rows.sort(
        key=lambda item: (
            float((item.get("rates") or {}).get("mis_mapping_rate", 0.0) or 0.0),
            float((item.get("rates") or {}).get("one_step_shift_rate", 0.0) or 0.0),
            float((item.get("rates") or {}).get("sequence_anomaly_rate", 0.0) or 0.0),
            float((item.get("rates") or {}).get("front_silence_rate", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows[: max(1, int(top))]


def _nearest_index(values: list[float], target: float) -> tuple[int, float]:
    best_idx = -1
    best_delta = None
    for idx, value in enumerate(values):
        delta = abs(float(target) - float(value))
        if best_delta is None or delta < best_delta:
            best_idx = idx
            best_delta = delta
    return int(best_idx), float(best_delta if best_delta is not None else 0.0)


def _is_blank_or_silence(alias: str) -> bool:
    text = str(alias or "").strip()
    if not text or text == "-":
        return True
    text = text.lower()
    if text.startswith("-"):
        text = text[1:].strip()
    if not text:
        return True
    parts = text.split()
    return bool(parts) and all(part in _SILENCE_TOKENS for part in parts)


def _rate(numer: int, denom: int) -> float:
    if int(denom) <= 0:
        return 0.0
    return float(int(numer)) / float(int(denom))


def _apply_sequence_anomaly_counts(*, wav_payload: dict[str, Any], totals: dict[str, int]) -> None:
    rows = list(wav_payload.get("rows") or [])
    pre_values: list[float] = []
    row_indices: list[int] = []
    for idx, row in enumerate(rows):
        if bool(row.get("silence_like", False)):
            continue
        pre = row.get("pre_abs_ms")
        if pre is None:
            continue
        pre_values.append(float(pre))
        row_indices.append(idx)
    if len(pre_values) <= 1:
        wav_payload.setdefault("counts", {})["sequence_anomaly_count"] = 0
        wav_payload.setdefault("rates", {})["sequence_anomaly_rate"] = 0.0
        return

    steps = [pre_values[i] - pre_values[i - 1] for i in range(1, len(pre_values))]
    steps_sorted = sorted(steps)
    median_step = float(steps_sorted[len(steps_sorted) // 2])
    max_step = max(160.0, median_step * 3.2)
    min_step = -35.0
    anomaly_count = 0
    for i, step in enumerate(steps):
        bad = bool(step < min_step or step > max_step)
        row_idx = row_indices[i + 1]
        rows[row_idx]["sequence_anomaly"] = bool(bad)
        rows[row_idx]["sequence_step_ms"] = float(step)
        rows[row_idx]["sequence_step_median_ms"] = float(median_step)
        if bad:
            anomaly_count += 1

    wav_payload["rows"] = rows
    wav_payload.setdefault("counts", {})["sequence_anomaly_count"] = int(anomaly_count)
    wav_payload.setdefault("rates", {})["sequence_anomaly_rate"] = _rate(
        int(anomaly_count),
        int((wav_payload.get("counts") or {}).get("rows_non_silence", 0) or 0),
    )
    totals["sequence_anomaly_count"] += int(anomaly_count)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_path": report.get("a_path"),
        "b_path": report.get("b_path"),
        "source_path": report.get("source_path"),
        "a_rates": (((report.get("a") or {}).get("overall") or {}).get("rates") or {}),
        "b_rates": (((report.get("b") or {}).get("overall") or {}).get("rates") or {}),
        "delta_b_minus_a": report.get("delta_b_minus_a"),
        "top_wav_delta": (report.get("wav_delta_b_minus_a") or [])[:10],
    }


if __name__ == "__main__":
    raise SystemExit(main())
