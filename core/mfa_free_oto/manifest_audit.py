from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .features import extract_features
from .htk_lab import OTHER_LABELS
from .kana import parse_kana_text
from .manifest import load_manifest_jsonl, write_manifest_jsonl
from .slot_viterbi import expected_cv_slots_from_phones
from .types import FRAME_LABELS


HARD_FLAG_PREFIXES = (
    "frame_span_out_of_duration",
    "event_out_of_duration",
    "segment_time_invalid",
    "wav_missing",
    "feature_error",
)


@dataclass(frozen=True)
class ManifestAuditConfig:
    duration_tolerance_ms: float = 25.0
    high_energy_silence_ratio: float = 0.45
    voiced_silence_threshold: float = 0.35
    weak_vowel_rms_ratio: float = 0.25


def audit_manifest_path(
    manifest_path: str | Path,
    *,
    clean_out: str | Path | None = None,
    config: ManifestAuditConfig | None = None,
) -> dict:
    rows = load_manifest_jsonl(manifest_path, require_manual=True, require_labels=True)
    report = audit_manifest_rows(rows, manifest_path=str(manifest_path), config=config)
    if clean_out:
        clean_rows = [row for row, row_report in zip(rows, report["row_reports"]) if row_report["clean_trainable"]]
        write_manifest_jsonl(clean_out, clean_rows)
        report["clean_manifest"] = str(Path(clean_out))
        report["clean_rows"] = len(clean_rows)
    return report


def audit_manifest_rows(
    rows: Sequence[dict],
    *,
    manifest_path: str | None = None,
    config: ManifestAuditConfig | None = None,
) -> dict:
    cfg = config or ManifestAuditConfig()
    label_durations: dict[str, float] = {label: 0.0 for label in FRAME_LABELS}
    label_segments: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    dropped_phone_counts: Counter[str] = Counter()
    phone_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    row_reports: list[dict] = []

    for row in rows:
        row_report = audit_row(row, config=cfg)
        row_reports.append(row_report)
        for label, value in row_report["label_duration_ms"].items():
            label_durations[label] += float(value)
        label_segments.update(row_report["label_segment_counts"])
        event_counts.update(row_report["event_counts"])
        dropped_phone_counts.update(row_report["dropped_phone_counts"])
        phone_counts.update(row_report["phone_counts"])
        flag_counts.update(_flag_key(flag) for flag in row_report["flags"])

    total_duration = sum(label_durations.values())
    clean_rows = sum(1 for report in row_reports if report["clean_trainable"])
    slot_eligible_rows = sum(1 for report in row_reports if report["slot_metric_eligible"])
    flagged_rows = sum(1 for report in row_reports if report["flags"])
    hard_flag_rows = sum(1 for report in row_reports if report["hard_flags"])
    return {
        "manifest_path": manifest_path,
        "rows": len(rows),
        "clean_rows": clean_rows,
        "flagged_rows": flagged_rows,
        "hard_flag_rows": hard_flag_rows,
        "slot_metric_eligible_rows": slot_eligible_rows,
        "label_duration_ms": {label: round(value, 3) for label, value in label_durations.items()},
        "label_duration_ratio": {
            label: (float(value) / total_duration if total_duration > 0.0 else 0.0)
            for label, value in label_durations.items()
        },
        "label_segment_counts": dict(label_segments),
        "event_counts": dict(event_counts),
        "dropped_phone_counts": dict(dropped_phone_counts),
        "phone_counts": dict(phone_counts),
        "flag_counts": dict(flag_counts),
        "row_reports": row_reports,
    }


def audit_row(row: dict, *, config: ManifestAuditConfig | None = None) -> dict:
    cfg = config or ManifestAuditConfig()
    row_id = str(row.get("row_id") or row.get("wav_name") or row.get("wav_path"))
    duration_ms = float(row.get("duration_ms") or 0.0)
    flags: list[str] = []
    label_duration: dict[str, float] = {label: 0.0 for label in FRAME_LABELS}
    label_segments: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    phone_counts: Counter[str] = Counter()
    dropped_phone_counts: Counter[str] = Counter()

    for dropped in row.get("dropped_lab_segments") or []:
        dropped_phone_counts.update([str(dropped.get("phone") or "").lower()])

    for segment in row.get("frame_labels") or []:
        label = str(segment.get("label") or "")
        start_ms = _float(segment.get("start_ms"))
        end_ms = _float(segment.get("end_ms"))
        phone = str(segment.get("phone") or "").strip()
        if phone:
            phone_counts.update([phone.lower()])
        if label in label_duration:
            label_segments.update([label])
            label_duration[label] += max(0.0, min(end_ms, duration_ms) - max(0.0, start_ms))
        if start_ms < 0.0 or end_ms <= start_ms:
            flags.append(f"segment_time_invalid:{label}:{start_ms:.1f}-{end_ms:.1f}")
        if end_ms > duration_ms + cfg.duration_tolerance_ms:
            flags.append(f"frame_span_out_of_duration:{label}:{end_ms:.1f}>{duration_ms:.1f}")

    for event in row.get("events") or []:
        label = str(event.get("label") or "")
        time_ms = _float(event.get("time_ms"))
        event_counts.update([label])
        if time_ms > duration_ms + cfg.duration_tolerance_ms:
            flags.append(f"event_out_of_duration:{label}:{time_ms:.1f}>{duration_ms:.1f}")

    acoustic_flags, acoustic_summary = _audit_row_acoustics(row, config=cfg)
    flags.extend(acoustic_flags)

    expected_phones = _expected_phones(row)
    filename_phones = infer_filename_phone_sequence(str(row.get("wav_name") or row.get("row_id") or ""))
    filename_match_ratio = _ordered_match_ratio(_strip_trailing_other(expected_phones), filename_phones)
    slot_count = len(expected_cv_slots_from_phones(expected_phones))
    cv_count = int(event_counts.get("cv_boundary", 0))
    if slot_count and not filename_phones:
        flags.append("filename_phone_unparsed")
    elif slot_count and filename_match_ratio < 0.9:
        flags.append(f"filename_slot_mismatch:{filename_match_ratio:.3f}")
    if slot_count and cv_count == 0:
        flags.append("missing_cv_boundary")

    hard_flags = [flag for flag in flags if _is_hard_flag(flag)]
    slot_metric_eligible = bool(slot_count and filename_match_ratio >= 0.9 and not hard_flags)
    return {
        "row_id": row_id,
        "wav_name": row.get("wav_name"),
        "dataset_id": row.get("dataset_id"),
        "duration_ms": duration_ms,
        "expected_phone_count": len(expected_phones),
        "filename_phone_count": len(filename_phones),
        "filename_match_ratio": filename_match_ratio,
        "cv_boundary_count": cv_count,
        "slot_count": slot_count,
        "slot_metric_eligible": slot_metric_eligible,
        "clean_trainable": not hard_flags,
        "flags": flags,
        "hard_flags": hard_flags,
        "label_duration_ms": {label: round(value, 3) for label, value in label_duration.items()},
        "label_segment_counts": dict(label_segments),
        "event_counts": dict(event_counts),
        "phone_counts": dict(phone_counts),
        "dropped_phone_counts": dict(dropped_phone_counts),
        "acoustic_summary": acoustic_summary,
    }


def infer_filename_phone_sequence(name: str) -> list[str]:
    stem = Path(name).stem.lower()
    if not stem:
        return []
    if not stem.isascii():
        return _parse_kana_stem(stem)
    tokens = [token for token in stem.replace("_", "-").split("-") if token]
    phones: list[str] = []
    for token in tokens:
        parsed = _parse_romaji_token(token)
        if parsed is None:
            return []
        phones.extend(parsed)
    return phones


_KANA_PHONE_TABLE = {
    "あ": ["a"], "い": ["i"], "う": ["u"], "え": ["e"], "お": ["o"],
    "ん": ["n"], "ン": ["n"],
    "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "u"], "け": ["k", "e"], "こ": ["k", "o"],
    "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "u"], "げ": ["g", "e"], "ご": ["g", "o"],
    "さ": ["s", "a"], "し": ["sh", "i"], "す": ["s", "u"], "せ": ["s", "e"], "そ": ["s", "o"],
    "ざ": ["z", "a"], "じ": ["j", "i"], "ず": ["z", "u"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
    "た": ["t", "a"], "ち": ["ch", "i"], "つ": ["ts", "u"], "て": ["t", "e"], "と": ["t", "o"],
    "だ": ["d", "a"], "ぢ": ["j", "i"], "づ": ["z", "u"], "で": ["d", "e"], "ど": ["d", "o"],
    "な": ["n", "a"], "に": ["n", "i"], "ぬ": ["n", "u"], "ね": ["n", "e"], "の": ["n", "o"],
    "は": ["h", "a"], "ひ": ["h", "i"], "ふ": ["f", "u"], "へ": ["h", "e"], "ほ": ["h", "o"],
    "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "u"], "べ": ["b", "e"], "ぼ": ["b", "o"],
    "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "u"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],
    "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "u"], "め": ["m", "e"], "も": ["m", "o"],
    "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
    "ら": ["r", "a"], "り": ["r", "i"], "る": ["r", "u"], "れ": ["r", "e"], "ろ": ["r", "o"],
    "わ": ["w", "a"], "を": ["w", "o"],
    "ア": ["a"], "イ": ["i"], "ウ": ["u"], "エ": ["e"], "オ": ["o"],
}

_KANA_YOON_TABLE = {
    "ゃ": ("y", "a"), "ゅ": ("y", "u"), "ょ": ("y", "o"),
    "ャ": ("y", "a"), "ュ": ("y", "u"), "ョ": ("y", "o"),
}


def _parse_kana_stem(stem: str) -> list[str]:
    return parse_kana_text(stem)


def _audit_row_acoustics(row: dict, *, config: ManifestAuditConfig) -> tuple[list[str], dict]:
    wav_path = Path(str(row.get("wav_path") or ""))
    if not wav_path.is_file():
        return ["wav_missing"], {}
    try:
        batch = extract_features(wav_path, encoder="acoustic-aux")
    except Exception as exc:
        return [f"feature_error:{exc}"], {}
    rms = np.asarray(batch.acoustic_scores.get("rms", []), dtype=np.float32)
    voicing = np.asarray(batch.acoustic_scores.get("voicing", []), dtype=np.float32)
    times = np.asarray(batch.times_ms, dtype=np.float32)
    if rms.size == 0 or times.size == 0:
        return ["feature_error:empty_acoustic_scores"], {}

    per_label_rms: dict[str, list[float]] = defaultdict(list)
    per_label_voicing: dict[str, list[float]] = defaultdict(list)
    flags: list[str] = []
    for segment in row.get("frame_labels") or []:
        label = str(segment.get("label") or "")
        start_ms = _float(segment.get("start_ms"))
        end_ms = _float(segment.get("end_ms"))
        mask = (times >= start_ms) & (times < end_ms)
        if not np.any(mask):
            continue
        mean_rms = float(np.mean(rms[mask]))
        mean_voicing = float(np.mean(voicing[mask])) if voicing.size else 0.0
        per_label_rms[label].append(mean_rms)
        per_label_voicing[label].append(mean_voicing)

    voiced_rms_values = per_label_rms.get("consonant", []) + per_label_rms.get("vowel", [])
    voiced_reference = float(np.median(np.asarray(voiced_rms_values, dtype=np.float32))) if voiced_rms_values else float(np.percentile(rms, 75.0))
    vowel_reference = float(np.median(np.asarray(per_label_rms.get("vowel", []), dtype=np.float32))) if per_label_rms.get("vowel") else voiced_reference

    for idx, segment in enumerate(row.get("frame_labels") or []):
        label = str(segment.get("label") or "")
        start_ms = _float(segment.get("start_ms"))
        end_ms = _float(segment.get("end_ms"))
        mask = (times >= start_ms) & (times < end_ms)
        if not np.any(mask):
            continue
        mean_rms = float(np.mean(rms[mask]))
        mean_voicing = float(np.mean(voicing[mask])) if voicing.size else 0.0
        if label == "silence" and voiced_reference > 0.0:
            if mean_rms >= voiced_reference * config.high_energy_silence_ratio:
                flags.append(f"high_energy_silence:{idx}:{mean_rms:.6f}")
            if mean_voicing >= config.voiced_silence_threshold:
                flags.append(f"voiced_silence:{idx}:{mean_voicing:.3f}")
        if label == "vowel" and vowel_reference > 0.0 and mean_rms <= vowel_reference * config.weak_vowel_rms_ratio:
            flags.append(f"low_energy_vowel:{idx}:{mean_rms:.6f}")

    summary = {
        "rms_p10": float(np.percentile(rms, 10.0)),
        "rms_p50": float(np.percentile(rms, 50.0)),
        "rms_p90": float(np.percentile(rms, 90.0)),
        "voicing_p50": float(np.percentile(voicing, 50.0)) if voicing.size else 0.0,
        "voiced_reference_rms": voiced_reference,
    }
    return flags, summary


def _parse_romaji_token(token: str) -> list[str] | None:
    if token in {"pau", "sil", "sp", "ap"}:
        return []
    if token in OTHER_LABELS or token in {"vf"}:
        return [token]
    vowels = {"a", "i", "u", "e", "o", "n"}
    if token in vowels:
        return [token]
    phones: list[str] = []
    pos = 0
    consonant_clusters = ("ky", "gy", "sh", "ch", "ts", "ny", "hy", "by", "py", "my", "ry", "jy")
    while pos < len(token):
        if token[pos] in vowels:
            phones.append(token[pos])
            pos += 1
            continue
        cluster = next((item for item in consonant_clusters if token.startswith(item, pos)), None)
        if cluster:
            phones.extend(list(cluster))
            pos += len(cluster)
            continue
        char = token[pos]
        if char.isalpha():
            phones.append(char)
            pos += 1
            continue
        return None
    return phones


def _expected_phones(row: dict) -> list[str]:
    phones: list[str] = []
    for item in row.get("expected_phones") or []:
        phone = str(item.get("phone") if isinstance(item, dict) else item).strip().lower()
        if phone and phone not in {"ap", "sp", "sil", "pau"}:
            phones.append(phone)
    return phones


def _strip_trailing_other(phones: Sequence[str]) -> list[str]:
    out = list(phones)
    while out and out[-1].lower() in OTHER_LABELS:
        out.pop()
    return out


def _ordered_match_ratio(expected: Sequence[str], filename: Sequence[str]) -> float:
    if not expected or not filename:
        return 0.0
    expected_list = [item.lower() for item in expected]
    filename_list = [item.lower() for item in filename]
    limit = min(len(expected_list), len(filename_list))
    matches = sum(1 for idx in range(limit) if expected_list[idx] == filename_list[idx])
    penalty = abs(len(expected_list) - len(filename_list))
    return matches / max(1, limit + penalty)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_hard_flag(flag: str) -> bool:
    return any(flag.startswith(prefix) for prefix in HARD_FLAG_PREFIXES)


def _flag_key(flag: str) -> str:
    return flag.split(":", 1)[0]


def top_flagged_rows(report: dict, *, limit: int = 50) -> list[dict]:
    flagged = [row for row in report.get("row_reports", []) if row.get("flags")]
    flagged.sort(key=lambda row: (len(row.get("hard_flags") or []), len(row.get("flags") or [])), reverse=True)
    return flagged[:limit]
