from __future__ import annotations

import json
import hashlib
import math
import os
import wave
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from ..model_context.timebase import TIMEBASE_SCHEMA_VERSION, TimebaseContract, timebase_from_mapping
from .row_plan import FilenameSlot, build_filename_slots
from .types import EVENT_LABELS, FRAME_LABELS, FramePosterior


ACOUSTIC_EVIDENCE_SCHEMA_VERSION = 1
ACOUSTIC_EVIDENCE_EXTRACTOR_VERSION = "acoustic_evidence_pack_v1"
ACOUSTIC_EVIDENCE_DECODER_VERSION = "runtime_event_prior_v1"

_ACOUSTIC_TRACKS = (
    "transition_likelihood",
    "voicing",
    "nucleus_likelihood",
    "sonorant_onset_likelihood",
    "silence_likelihood",
    "world_nucleus",
    "world_voicing",
    "world_periodicity",
    "world_spectral_stability",
)

_REQUIRED_TIMEBASE_SUMMARY_FIELDS = (
    "sample_rate",
    "analysis_sample_rate",
    "window_size_ms",
    "frame_time_anchor",
    "feature_latency_ms",
    "padding_policy",
    "resample_method",
    "rounding",
)


def build_acoustic_evidence_pack(
    posterior: FramePosterior,
    *,
    wav_name: str = "",
    expected_phones: Iterable[str] = (),
    filename_slots: Iterable[object] = (),
    runtime_events: Iterable[Mapping[str, object]] = (),
    candidate_limit_per_label: int = 6,
    include_frame_tracks: bool = True,
) -> dict[str, object]:
    posterior_metadata = dict(posterior.metadata or {})
    wav_path = os.path.abspath(str(posterior.wav_path or "")) if posterior.wav_path else ""
    times = [_round_ms(value) for value in posterior.times_ms]
    frame_shift = _frame_shift_ms(times)
    duration = _duration_ms(times, frame_shift_ms=frame_shift)
    timebase_contract = _timebase_contract_for_evidence(
        posterior,
        wav_path=wav_path,
        frame_shift_ms=frame_shift,
    )
    timebase_contract_payload = timebase_contract.to_dict()
    timebase_fingerprint = timebase_contract.cache_fingerprint()
    class_tracks = _selected_tracks(posterior.class_probs, FRAME_LABELS, len(times))
    event_tracks = _selected_tracks(posterior.event_scores, EVENT_LABELS, len(times))
    acoustic_tracks = _selected_tracks(posterior.acoustic_scores, _ACOUSTIC_TRACKS, len(times))
    candidates: list[dict[str, object]] = []
    for label in EVENT_LABELS:
        candidates.extend(
            _event_candidates(
                label,
                event_tracks.get(label, ()),
                times,
                frame_shift_ms=frame_shift,
                class_tracks=class_tracks,
                acoustic_tracks=acoustic_tracks,
                limit=candidate_limit_per_label,
            )
        )
    candidates.extend(
        _event_candidates(
            "sonorant_constriction",
            acoustic_tracks.get("sonorant_onset_likelihood", ()),
            times,
            frame_shift_ms=frame_shift,
            class_tracks=class_tracks,
            acoustic_tracks=acoustic_tracks,
            limit=max(1, candidate_limit_per_label // 2),
        )
    )
    candidates.extend(
        _event_candidates(
            "vv_boundary",
            _vv_boundary_track(class_tracks, acoustic_tracks, len(times)),
            times,
            frame_shift_ms=frame_shift,
            class_tracks=class_tracks,
            acoustic_tracks=acoustic_tracks,
            limit=max(1, candidate_limit_per_label // 2),
        )
    )
    payload: dict[str, object] = {
        "schema_version": ACOUSTIC_EVIDENCE_SCHEMA_VERSION,
        "wav": str(wav_name or os.path.basename(str(posterior.wav_path or ""))),
        "wav_path": wav_path,
        "wav_sha256": _sha256_file_if_available(wav_path),
        "extractor_version": str(
            posterior_metadata.get("extractor_version")
            or posterior_metadata.get("feature_extractor_version")
            or ACOUSTIC_EVIDENCE_EXTRACTOR_VERSION
        ),
        "decoder_version": str(posterior_metadata.get("decoder_version") or ACOUSTIC_EVIDENCE_DECODER_VERSION),
        "timebase_fingerprint": timebase_fingerprint,
        "expected_phones": [str(phone) for phone in expected_phones],
        "filename_slots": [_json_safe(slot) for slot in filename_slots],
        "timebase": {
            "frame_count": len(times),
            "frame_shift_ms": _round_ms(frame_shift),
            "sample_rate": int(timebase_contract.sample_rate),
            "analysis_sample_rate": int(timebase_contract.analysis_sample_rate),
            "window_size_ms": _round_ms(timebase_contract.window_size_ms),
            "frame_time_anchor": str(timebase_contract.frame_time_anchor),
            "feature_latency_ms": _round_ms(timebase_contract.feature_latency_ms),
            "padding_policy": str(timebase_contract.padding_policy),
            "resample_method": str(timebase_contract.resample_method),
            "rounding": str(timebase_contract.rounding),
            "contract": timebase_contract_payload,
            "fingerprint": timebase_fingerprint,
            "start_ms": times[0] if times else 0.0,
            "duration_ms": _round_ms(duration),
        },
        "reliability": _reliability(
            posterior,
            class_tracks=class_tracks,
            event_tracks=event_tracks,
            acoustic_tracks=acoustic_tracks,
            frame_count=len(times),
        ),
        "event_candidates": candidates,
        "event_candidate_summary": summarize_event_candidate_reliability(candidates),
        "runtime_events": [_json_safe(event) for event in runtime_events],
    }
    if include_frame_tracks:
        payload["frame_posterior"] = {
            "times_ms": times,
            "class_probs": class_tracks,
            "event_scores": event_tracks,
            "acoustic_scores": acoustic_tracks,
        }
    return payload


def write_evidence_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def read_evidence_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    evidence_path = Path(path)
    if not evidence_path.is_file():
        return ()
    rows: list[dict[str, object]] = []
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            if isinstance(data, Mapping):
                rows.append(dict(data))
                continue
            raise ValueError(f"line {line_no}: evidence row is not a JSON object")
    return tuple(rows)


def validate_acoustic_evidence_pack(payload: Mapping[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        schema_version = int(payload.get("schema_version", 0) or 0)
    except Exception:
        schema_version = 0
    if schema_version != ACOUSTIC_EVIDENCE_SCHEMA_VERSION:
        errors.append("evidence.schema_version_unsupported")
    extractor_version = str(payload.get("extractor_version", "") or "").strip()
    decoder_version = str(payload.get("decoder_version", "") or "").strip()
    if not extractor_version:
        errors.append("evidence.extractor_version_missing")
    elif extractor_version != ACOUSTIC_EVIDENCE_EXTRACTOR_VERSION:
        errors.append("evidence.extractor_version_unsupported")
    if not decoder_version:
        errors.append("evidence.decoder_version_missing")
    elif decoder_version != ACOUSTIC_EVIDENCE_DECODER_VERSION:
        errors.append("evidence.decoder_version_unsupported")
    if not str(payload.get("wav", "") or payload.get("wav_path", "") or "").strip():
        errors.append("evidence.wav_missing")
    wav_path = str(payload.get("wav_path", "") or "").strip()
    reported_wav_sha = str(payload.get("wav_sha256", "") or "").strip().lower()
    if reported_wav_sha and len(reported_wav_sha) != 64:
        errors.append("evidence.wav_sha256_invalid")
    if wav_path:
        wav_file = Path(wav_path)
        if wav_file.is_file():
            if not reported_wav_sha:
                errors.append("evidence.wav_sha256_missing")
            else:
                current_sha = _sha256_file_if_available(wav_file).strip().lower()
                if not current_sha:
                    errors.append("evidence.wav_sha256_unavailable")
                elif current_sha != reported_wav_sha:
                    errors.append("evidence.wav_sha256_mismatch")

    timebase = payload.get("timebase")
    frame_count = 0
    if not isinstance(timebase, Mapping):
        errors.append("evidence.timebase_missing")
    else:
        try:
            frame_count = int(timebase.get("frame_count", 0) or 0)
        except Exception:
            frame_count = 0
        if frame_count <= 0:
            errors.append("evidence.timebase.frame_count_invalid")
        if _safe_float(timebase.get("frame_shift_ms", 0.0)) <= 0.0:
            errors.append("evidence.timebase.frame_shift_ms_invalid")
        if _safe_float(timebase.get("duration_ms", 0.0)) <= 0.0:
            errors.append("evidence.timebase.duration_ms_invalid")
        for field in _REQUIRED_TIMEBASE_SUMMARY_FIELDS:
            if field not in timebase:
                errors.append(f"evidence.timebase.{field}_missing")
        if _safe_int(timebase.get("sample_rate", 0)) <= 0:
            errors.append("evidence.timebase.sample_rate_invalid")
        if _safe_int(timebase.get("analysis_sample_rate", 0)) <= 0:
            errors.append("evidence.timebase.analysis_sample_rate_invalid")
        if _safe_float(timebase.get("window_size_ms", 0.0)) <= 0.0:
            errors.append("evidence.timebase.window_size_ms_invalid")
        if _norm_timebase_text(timebase.get("frame_time_anchor")) not in {"left", "center", "right"}:
            errors.append("evidence.timebase.frame_time_anchor_invalid")
        if not _is_finite_number(timebase.get("feature_latency_ms")) or _safe_float(
            timebase.get("feature_latency_ms", 0.0)
        ) < 0.0:
            errors.append("evidence.timebase.feature_latency_ms_invalid")
        if not _norm_timebase_text(timebase.get("padding_policy")):
            errors.append("evidence.timebase.padding_policy_invalid")
        if not _norm_timebase_text(timebase.get("rounding")):
            errors.append("evidence.timebase.rounding_invalid")
        errors.extend(_validate_timebase_contract(timebase, payload))

    frame = payload.get("frame_posterior")
    if not isinstance(frame, Mapping):
        errors.append("evidence.frame_posterior_missing")
        frame = {}

    times, time_errors = _validated_numeric_sequence(frame.get("times_ms", ()), field="frame_posterior.times_ms")
    errors.extend(time_errors)
    if not times:
        errors.append("frame_posterior.times_ms_empty")
    if frame_count > 0 and len(times) != frame_count:
        errors.append("frame_posterior.times_ms_frame_count_mismatch")
    if any(times[idx] <= times[idx - 1] for idx in range(1, len(times))):
        errors.append("frame_posterior.times_ms_not_strictly_increasing")

    effective_frame_count = frame_count if frame_count > 0 else len(times)
    errors.extend(
        _validate_track_mapping(
            frame.get("class_probs", {}),
            FRAME_LABELS,
            frame_count=effective_frame_count,
            field="frame_posterior.class_probs",
            require_all=True,
        )
    )
    errors.extend(
        _validate_track_mapping(
            frame.get("event_scores", {}),
            EVENT_LABELS,
            frame_count=effective_frame_count,
            field="frame_posterior.event_scores",
            require_all=True,
        )
    )
    errors.extend(
        _validate_track_mapping(
            frame.get("acoustic_scores", {}),
            _ACOUSTIC_TRACKS,
            frame_count=effective_frame_count,
            field="frame_posterior.acoustic_scores",
            require_all=False,
        )
    )

    reliability = payload.get("reliability")
    if not isinstance(reliability, Mapping):
        errors.append("evidence.reliability_missing")
    elif frame_count > 0:
        try:
            reliability_frame_count = int(reliability.get("frame_count", 0) or 0)
        except Exception:
            reliability_frame_count = 0
        if reliability_frame_count != frame_count:
            errors.append("evidence.reliability.frame_count_mismatch")

    for field in ("expected_phones", "filename_slots", "event_candidates", "runtime_events"):
        value = payload.get(field, ())
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            errors.append(f"evidence.{field}_invalid")
    return tuple(dict.fromkeys(errors))


def summarize_event_candidate_reliability(
    payload_or_candidates: Mapping[str, object] | Sequence[object],
) -> dict[str, object]:
    if isinstance(payload_or_candidates, Mapping):
        raw_candidates = payload_or_candidates.get("event_candidates", ())
    else:
        raw_candidates = payload_or_candidates
    candidates = [dict(item or {}) for item in list(raw_candidates or ()) if isinstance(item, Mapping)]
    flag_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    type_flag_counts: dict[str, int] = {}
    margins: list[float] = []
    feature_agreements: list[float] = []
    ood_values: list[float] = []
    for candidate in candidates:
        event_type = str(candidate.get("event_type", "") or "unknown")
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        margins.append(_safe_float(candidate.get("margin", 0.0)))
        feature_agreements.append(_safe_float(candidate.get("feature_agreement", 0.0)))
        ood_values.append(_safe_float(candidate.get("ood", 0.0)))
        for flag in list(candidate.get("flags", ()) or ()):
            name = str(flag or "")
            if not name:
                continue
            flag_counts[name] = flag_counts.get(name, 0) + 1
            type_flag_key = f"{event_type}_{name}"
            type_flag_counts[type_flag_key] = type_flag_counts.get(type_flag_key, 0) + 1
    return {
        "event_candidate_count": int(len(candidates)),
        "event_candidate_flagged_count": int(sum(1 for item in candidates if list(item.get("flags", ()) or ()))),
        "event_candidate_min_margin": _round_score(min(margins) if margins else 0.0),
        "event_candidate_min_feature_agreement": _round_score(min(feature_agreements) if feature_agreements else 0.0),
        "event_candidate_max_ood": _round_score(max(ood_values) if ood_values else 0.0),
        "event_candidate_type_counts": type_counts,
        "event_candidate_flag_counts": flag_counts,
        "event_candidate_type_flag_counts": type_flag_counts,
    }


def frame_posterior_from_evidence_pack(payload: Mapping[str, object]) -> FramePosterior:
    frame = payload.get("frame_posterior")
    if not isinstance(frame, Mapping):
        raise ValueError("evidence pack does not contain frame_posterior")
    reliability = dict(payload.get("reliability", {}) or {}) if isinstance(payload.get("reliability"), Mapping) else {}
    metadata = {
        "encoder": str(reliability.get("encoder", "")),
        "rule_based": bool(reliability.get("rule_based", False)),
        "acoustic_feature_set": str(reliability.get("acoustic_feature_set", "")),
        "replayed_from_evidence": True,
        "evidence_schema_version": int(payload.get("schema_version", ACOUSTIC_EVIDENCE_SCHEMA_VERSION) or 1),
        "evidence_extractor_version": str(payload.get("extractor_version", "") or ""),
        "evidence_decoder_version": str(payload.get("decoder_version", "") or ""),
        "evidence_wav_sha256": str(payload.get("wav_sha256", "") or ""),
        "evidence_timebase_fingerprint": str(payload.get("timebase_fingerprint", "") or ""),
    }
    timebase = payload.get("timebase")
    if isinstance(timebase, Mapping):
        metadata["evidence_timebase"] = dict(timebase)
    return FramePosterior(
        wav_path=str(payload.get("wav_path", "") or payload.get("wav", "")),
        times_ms=_float_list(frame.get("times_ms", ())),
        class_probs=_tracks_from_payload(frame.get("class_probs", {}), FRAME_LABELS),
        event_scores=_tracks_from_payload(frame.get("event_scores", {}), EVENT_LABELS),
        acoustic_scores=_tracks_from_payload(frame.get("acoustic_scores", {}), _ACOUSTIC_TRACKS),
        metadata=metadata,
    )


def filename_slots_from_evidence_pack(
    payload: Mapping[str, object],
    *,
    language: str = "",
    format_type: str = "",
) -> tuple[FilenameSlot, ...]:
    wav = str(payload.get("wav", "") or os.path.basename(str(payload.get("wav_path", "") or "")))
    raw_slots = payload.get("filename_slots")
    slots: list[FilenameSlot] = []
    if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)):
        for item in raw_slots:
            if not isinstance(item, Mapping):
                continue
            slot = _filename_slot_from_mapping(item, default_wav=wav)
            if slot is not None:
                slots.append(slot)
    if slots:
        return tuple(sorted(slots, key=lambda item: item.slot_index))
    return build_filename_slots(wav, language=language, format_type=format_type)


def expected_phones_from_evidence_pack(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw = payload.get("expected_phones", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(item) for item in raw if str(item).strip())


def runtime_events_from_evidence_pack(payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw = payload.get("runtime_events", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    events: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, Mapping):
            events.append(dict(item))
    return tuple(events)


def candidate_priors_from_evidence_pack(
    payload: Mapping[str, object],
    slots: Sequence[FilenameSlot],
    *,
    min_score: float = 0.35,
    min_margin: float = 0.05,
    min_feature_agreement: float = 0.35,
) -> tuple[dict[str, object], ...]:
    raw_candidates = payload.get("event_candidates", ())
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return ()
    candidates = [
        dict(item or {})
        for item in raw_candidates
        if isinstance(item, Mapping)
        and _candidate_is_reliable(
            item,
            min_score=min_score,
            min_margin=min_margin,
            min_feature_agreement=min_feature_agreement,
        )
    ]
    if not candidates or not slots:
        return ()
    duration = _payload_duration_ms(payload)
    out: list[dict[str, object]] = []
    out.extend(
        _candidate_priors_for_targets(
            candidates,
            event_type="phone_change",
            prior_label="phone_change",
            targets=[
                _candidate_target(slot, expected_phone_index=slot.phone_start_index, expected_ms=_slot_start_ms(slot, slots, duration))
                for slot in slots
                if slot.onset_phones
            ],
            duration_ms=duration,
        )
    )
    out.extend(
        _candidate_priors_for_targets(
            candidates,
            event_type="cv_boundary",
            prior_label="cv_boundary",
            targets=[
                _candidate_target(
                    slot,
                    expected_phone_index=slot.vowel_phone_index,
                    expected_ms=_slot_vowel_start_ms(slot, slots, duration),
                )
                for slot in slots
                if slot.onset_phones
            ],
            duration_ms=duration,
        )
    )
    out.extend(
        _candidate_priors_for_targets(
            candidates,
            event_type="vowel_nucleus",
            prior_label="vowel_nucleus",
            targets=[
                _candidate_target(
                    slot,
                    expected_phone_index=slot.vowel_phone_index,
                    expected_ms=_slot_nucleus_ms(slot, slots, duration),
                )
                for slot in slots
            ],
            duration_ms=duration,
        )
    )
    out.extend(
        _candidate_priors_for_targets(
            candidates,
            event_type="vv_boundary",
            prior_label="vv_boundary",
            targets=[
                _candidate_target(slot, expected_phone_index=slot.vowel_phone_index, expected_ms=_slot_start_ms(slot, slots, duration))
                for slot in slots
                if not slot.onset_phones and slot.slot_index > 0
            ],
            duration_ms=duration,
        )
    )
    out.extend(
        _candidate_priors_for_targets(
            candidates,
            event_type="sonorant_constriction",
            prior_label="sonorant_constriction",
            targets=[
                _candidate_target(slot, expected_phone_index=slot.phone_start_index, expected_ms=_slot_start_ms(slot, slots, duration))
                for slot in slots
                if _slot_has_sonorant_onset(slot)
            ],
            duration_ms=duration,
        )
    )
    return tuple(out)


def _event_candidates(
    event_type: str,
    values: Sequence[float],
    times: Sequence[float],
    *,
    frame_shift_ms: float,
    class_tracks: Mapping[str, Sequence[float]],
    acoustic_tracks: Mapping[str, Sequence[float]],
    limit: int,
) -> list[dict[str, object]]:
    scores = [_safe_float(value) for value in values]
    if not scores or not times:
        return []
    peaks = _peak_indices(scores, limit=max(1, int(limit)))
    if not peaks:
        return []
    ordered_scores = sorted((scores[idx] for idx in peaks), reverse=True)
    out: list[dict[str, object]] = []
    for rank, idx in enumerate(peaks):
        score = scores[idx]
        next_score = ordered_scores[rank + 1] if rank + 1 < len(ordered_scores) else _second_best_excluding(scores, idx)
        margin = max(0.0, score - next_score)
        width = _candidate_width_ms(scores, idx, threshold=score * 0.65, frame_shift_ms=frame_shift_ms)
        feature_agreement = _feature_agreement(event_type, idx, class_tracks=class_tracks, acoustic_tracks=acoustic_tracks)
        sharpness = _sharpness(scores, idx)
        ood = 1.0 - max(0.0, min(1.0, feature_agreement))
        flags: list[str] = []
        if margin < 0.05:
            flags.append("low_margin")
        if width > 85.0:
            flags.append("wide_candidate")
        if feature_agreement < 0.35:
            flags.append("low_feature_agreement")
        if event_type in {"vv_boundary", "sonorant_constriction"} and margin < 0.10:
            flags.append(f"ambiguous_{event_type}")
        out.append(
            {
                "event_type": str(event_type),
                "frame_index": int(idx),
                "time_ms": _round_ms(times[idx] if idx < len(times) else 0.0),
                "score": _round_score(score),
                "margin": _round_score(margin),
                "width_ms": _round_ms(width),
                "feature_agreement": _round_score(feature_agreement),
                "sharpness": _round_score(sharpness),
                "ood": _round_score(ood),
                "flags": flags,
            }
        )
    return out


def _candidate_is_reliable(
    candidate: Mapping[str, object],
    *,
    min_score: float,
    min_margin: float,
    min_feature_agreement: float,
) -> bool:
    flags = {str(flag) for flag in list(candidate.get("flags", ()) or ())}
    if "low_margin" in flags or "low_feature_agreement" in flags or any(flag.startswith("ambiguous_") for flag in flags):
        return False
    return (
        _safe_float(candidate.get("score", 0.0)) >= float(min_score)
        and _safe_float(candidate.get("margin", 0.0)) >= float(min_margin)
        and _safe_float(candidate.get("feature_agreement", 0.0)) >= float(min_feature_agreement)
    )


def _candidate_target(slot: FilenameSlot, *, expected_phone_index: int, expected_ms: float) -> dict[str, object]:
    return {
        "slot_index": int(slot.slot_index),
        "expected_phone": str(slot.vowel_phone if expected_phone_index == slot.vowel_phone_index else (slot.onset_phones[0] if slot.onset_phones else slot.vowel_phone)),
        "expected_phone_index": int(expected_phone_index),
        "expected_ms": float(expected_ms),
    }


def _candidate_priors_for_targets(
    candidates: Sequence[Mapping[str, object]],
    *,
    event_type: str,
    prior_label: str,
    targets: Sequence[Mapping[str, object]],
    duration_ms: float,
) -> list[dict[str, object]]:
    pool = [dict(candidate) for candidate in candidates if str(candidate.get("event_type", "")) == str(event_type)]
    if not pool or not targets:
        return []
    used: set[int] = set()
    out: list[dict[str, object]] = []
    max_distance = max(60.0, float(duration_ms or 0.0) / float(max(2, len(targets) * 2)))
    for target in sorted(targets, key=lambda item: float(item.get("expected_ms", 0.0) or 0.0)):
        expected = _safe_float(target.get("expected_ms", 0.0))
        best_idx = -1
        best_cost = float("inf")
        for idx, candidate in enumerate(pool):
            if idx in used:
                continue
            distance = abs(_safe_float(candidate.get("time_ms", 0.0)) - expected)
            cost = distance - (25.0 * _safe_float(candidate.get("score", 0.0)))
            if cost < best_cost:
                best_idx = idx
                best_cost = cost
        if best_idx < 0:
            continue
        selected = pool[best_idx]
        if abs(_safe_float(selected.get("time_ms", 0.0)) - expected) > max_distance:
            continue
        used.add(best_idx)
        out.append(
            {
                "label": str(prior_label),
                "selected_time_ms": _round_ms(selected.get("time_ms", 0.0)),
                "score": _round_score(selected.get("score", 0.0)),
                "expected_phone": str(target.get("expected_phone", "")),
                "expected_phone_index": _safe_int(target.get("expected_phone_index", -1), default=-1),
                "slot_index": _safe_int(target.get("slot_index", -1), default=-1),
                "frame_index": _safe_int(selected.get("frame_index", -1), default=-1),
                "source": "evidence_candidate",
                "event_type": str(event_type),
                "candidate_margin": _round_score(selected.get("margin", 0.0)),
                "candidate_feature_agreement": _round_score(selected.get("feature_agreement", 0.0)),
            }
        )
    return out


def _payload_duration_ms(payload: Mapping[str, object]) -> float:
    timebase = payload.get("timebase")
    if isinstance(timebase, Mapping):
        duration = _safe_float(timebase.get("duration_ms", 0.0))
        if duration > 0.0:
            return duration
    frame = payload.get("frame_posterior")
    if isinstance(frame, Mapping):
        times = _float_list(frame.get("times_ms", ()))
        return _duration_ms(times, frame_shift_ms=_frame_shift_ms(times))
    return 0.0


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _slot_start_ms(slot: FilenameSlot, slots: Sequence[FilenameSlot], duration_ms: float) -> float:
    slot_count = max(1, len(slots))
    return float(duration_ms or 0.0) * float(max(0, int(slot.slot_index))) / float(slot_count)


def _slot_vowel_start_ms(slot: FilenameSlot, slots: Sequence[FilenameSlot], duration_ms: float) -> float:
    slot_width = float(duration_ms or 0.0) / float(max(1, len(slots)))
    return _slot_start_ms(slot, slots, duration_ms) + (0.35 * slot_width if slot.onset_phones else 0.0)


def _slot_nucleus_ms(slot: FilenameSlot, slots: Sequence[FilenameSlot], duration_ms: float) -> float:
    slot_width = float(duration_ms or 0.0) / float(max(1, len(slots)))
    return _slot_start_ms(slot, slots, duration_ms) + (0.58 * slot_width)


def _slot_has_sonorant_onset(slot: FilenameSlot) -> bool:
    sonorants = {"m", "n", "ng", "ny", "r", "l", "w", "y", "j"}
    return any(str(phone).lower() in sonorants for phone in slot.onset_phones)


def _selected_tracks(
    mapping: Mapping[str, Sequence[float]],
    names: Iterable[str],
    frame_count: int,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name in names:
        values = mapping.get(str(name))
        if values is None:
            continue
        normalized = [_round_score(value) for value in list(values)[:frame_count]]
        if len(normalized) < frame_count:
            normalized.extend([0.0] * (frame_count - len(normalized)))
        out[str(name)] = normalized
    return out


def _vv_boundary_track(
    class_tracks: Mapping[str, Sequence[float]],
    acoustic_tracks: Mapping[str, Sequence[float]],
    frame_count: int,
) -> list[float]:
    transition = list(acoustic_tracks.get("transition_likelihood", ()))
    vowel = list(class_tracks.get("vowel", ()))
    out: list[float] = []
    for idx in range(frame_count):
        tr = _track_value(transition, idx)
        vv = _track_value(vowel, idx)
        out.append(_round_score(max(0.0, min(1.0, (0.60 * tr) + (0.40 * vv)))))
    return out


def _reliability(
    posterior: FramePosterior,
    *,
    class_tracks: Mapping[str, Sequence[float]],
    event_tracks: Mapping[str, Sequence[float]],
    acoustic_tracks: Mapping[str, Sequence[float]],
    frame_count: int,
) -> dict[str, object]:
    metadata = dict(posterior.metadata or {})
    has_world = any(str(key).startswith("world_") for key in acoustic_tracks)
    avg_event_peak = 0.0
    if event_tracks:
        peaks = [max((_safe_float(value) for value in values), default=0.0) for values in event_tracks.values()]
        avg_event_peak = sum(peaks) / float(max(1, len(peaks)))
    return {
        "frame_count": int(frame_count),
        "rule_based": bool(metadata.get("rule_based")),
        "encoder": str(metadata.get("encoder", "")),
        "acoustic_feature_set": str(metadata.get("acoustic_feature_set", "")),
        "extractor_version": str(
            metadata.get("extractor_version")
            or metadata.get("feature_extractor_version")
            or ACOUSTIC_EVIDENCE_EXTRACTOR_VERSION
        ),
        "decoder_version": str(metadata.get("decoder_version") or ACOUSTIC_EVIDENCE_DECODER_VERSION),
        "has_world_features": bool(has_world),
        "avg_event_peak": _round_score(avg_event_peak),
        "avg_vowel_prob": _round_score(_mean(class_tracks.get("vowel", ()))),
        "avg_consonant_prob": _round_score(_mean(class_tracks.get("consonant", ()))),
        "avg_silence_prob": _round_score(_mean(class_tracks.get("silence", ()))),
        "track_count": int(len(class_tracks) + len(event_tracks) + len(acoustic_tracks)),
    }


def _timebase_contract_for_evidence(
    posterior: FramePosterior,
    *,
    wav_path: str,
    frame_shift_ms: float,
) -> TimebaseContract:
    metadata = dict(posterior.metadata or {})
    sample_rate = _metadata_int(metadata, ("sample_rate", "source_sample_rate"), 0)
    if sample_rate <= 0:
        sample_rate = _wav_sample_rate_if_available(wav_path)
    analysis_sample_rate = _metadata_int(
        metadata,
        ("analysis_sample_rate", "feature_sample_rate", "target_sample_rate"),
        sample_rate,
    )
    if analysis_sample_rate <= 0:
        analysis_sample_rate = sample_rate
    return TimebaseContract(
        sample_rate=int(max(0, sample_rate)),
        analysis_sample_rate=int(max(0, analysis_sample_rate)),
        frame_shift_ms=float(frame_shift_ms),
        window_size_ms=_metadata_float(metadata, ("window_size_ms", "frame_ms"), 25.0),
        frame_time_anchor=str(metadata.get("frame_time_anchor", "") or "left"),
        feature_latency_ms=_metadata_float(metadata, ("feature_latency_ms", "latency_ms"), 0.0),
        padding_policy=str(metadata.get("padding_policy", "") or "reflect"),
        resample_method=str(metadata.get("resample_method", "") or metadata.get("resample", "") or ""),
        rounding=str(metadata.get("rounding", "") or "round_half_up_0.01ms"),
    )


def _validate_timebase_contract(timebase: Mapping[str, object], payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    raw_contract = timebase.get("contract")
    if not isinstance(raw_contract, Mapping):
        return ["evidence.timebase.contract_missing"]
    raw_schema_version = raw_contract.get("schema_version")
    if raw_schema_version is None or str(raw_schema_version).strip() == "":
        errors.append("evidence.timebase.contract_schema_version_missing")
    else:
        try:
            contract_schema_version = int(raw_schema_version)
        except Exception:
            errors.append("evidence.timebase.contract_schema_version_invalid")
            contract_schema_version = int(TIMEBASE_SCHEMA_VERSION)
        if contract_schema_version != int(TIMEBASE_SCHEMA_VERSION):
            errors.append("evidence.timebase.contract_schema_version_unsupported")
    contract = timebase_from_mapping(raw_contract)
    actual_fingerprint = contract.cache_fingerprint()
    fingerprint = str(timebase.get("fingerprint", "") or "").strip().lower()
    if not fingerprint:
        errors.append("evidence.timebase.fingerprint_missing")
    elif len(fingerprint) != 40:
        errors.append("evidence.timebase.fingerprint_invalid")
    elif fingerprint != actual_fingerprint:
        errors.append("evidence.timebase.fingerprint_mismatch")
    top_level_fingerprint = str(payload.get("timebase_fingerprint", "") or "").strip().lower()
    if not top_level_fingerprint:
        errors.append("evidence.timebase_fingerprint_missing")
    elif len(top_level_fingerprint) != 40:
        errors.append("evidence.timebase_fingerprint_invalid")
    elif fingerprint and top_level_fingerprint != fingerprint:
        errors.append("evidence.timebase_fingerprint_mismatch")
    if abs(_safe_float(timebase.get("frame_shift_ms", 0.0)) - float(contract.frame_shift_ms)) > 0.001:
        errors.append("evidence.timebase.contract_frame_shift_mismatch")
    if _safe_int(timebase.get("sample_rate", contract.sample_rate)) != int(contract.sample_rate):
        errors.append("evidence.timebase.contract_sample_rate_mismatch")
    if _safe_int(timebase.get("analysis_sample_rate", contract.analysis_sample_rate)) != int(
        contract.analysis_sample_rate
    ):
        errors.append("evidence.timebase.contract_analysis_sample_rate_mismatch")
    if abs(_safe_float(timebase.get("window_size_ms", 0.0)) - float(contract.window_size_ms)) > 0.001:
        errors.append("evidence.timebase.contract_window_size_mismatch")
    if _norm_timebase_text(timebase.get("frame_time_anchor")) != _norm_timebase_text(contract.frame_time_anchor):
        errors.append("evidence.timebase.contract_frame_time_anchor_mismatch")
    if abs(_safe_float(timebase.get("feature_latency_ms", 0.0)) - float(contract.feature_latency_ms)) > 0.001:
        errors.append("evidence.timebase.contract_feature_latency_mismatch")
    if _norm_timebase_text(timebase.get("padding_policy")) != _norm_timebase_text(contract.padding_policy):
        errors.append("evidence.timebase.contract_padding_policy_mismatch")
    if _norm_timebase_text(timebase.get("resample_method")) != _norm_timebase_text(contract.resample_method):
        errors.append("evidence.timebase.contract_resample_method_mismatch")
    if _norm_timebase_text(timebase.get("rounding")) != _norm_timebase_text(contract.rounding):
        errors.append("evidence.timebase.contract_rounding_mismatch")
    return errors


def _norm_timebase_text(value: object) -> str:
    return str(value or "").strip().lower()


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _metadata_int(metadata: Mapping[str, object], names: Sequence[str], default: int) -> int:
    for name in names:
        try:
            value = int(metadata.get(name, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return int(default)


def _metadata_float(metadata: Mapping[str, object], names: Sequence[str], default: float) -> float:
    for name in names:
        try:
            value = float(metadata.get(name, 0.0) or 0.0)
        except Exception:
            value = 0.0
        if value > 0.0:
            return value
    return float(default)


def _wav_sample_rate_if_available(path: str | Path) -> int:
    try:
        wav_path = Path(path)
        if not wav_path.is_file():
            return 0
        with wave.open(str(wav_path), "rb") as handle:
            return int(handle.getframerate())
    except Exception:
        return 0


def _sha256_file_if_available(path: str | Path) -> str:
    try:
        file_path = Path(path)
        if not file_path.is_file():
            return ""
        hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _peak_indices(values: Sequence[float], *, limit: int) -> list[int]:
    peaks: list[int] = []
    for idx, score in enumerate(values):
        left = values[idx - 1] if idx > 0 else float("-inf")
        right = values[idx + 1] if idx + 1 < len(values) else float("-inf")
        if score >= left and score >= right and score > 0.0:
            peaks.append(idx)
    if not peaks:
        peaks = [max(range(len(values)), key=lambda item: values[item])] if values else []
    peaks.sort(key=lambda item: (float(values[item]), -item), reverse=True)
    return peaks[: max(0, int(limit))]


def _second_best_excluding(values: Sequence[float], excluded_idx: int) -> float:
    best = 0.0
    for idx, value in enumerate(values):
        if idx == excluded_idx:
            continue
        best = max(best, _safe_float(value))
    return best


def _candidate_width_ms(values: Sequence[float], idx: int, *, threshold: float, frame_shift_ms: float) -> float:
    if not values:
        return 0.0
    lo = int(idx)
    hi = int(idx)
    while lo - 1 >= 0 and _safe_float(values[lo - 1]) >= threshold:
        lo -= 1
    while hi + 1 < len(values) and _safe_float(values[hi + 1]) >= threshold:
        hi += 1
    return float(hi - lo + 1) * max(0.001, float(frame_shift_ms))


def _feature_agreement(
    event_type: str,
    idx: int,
    *,
    class_tracks: Mapping[str, Sequence[float]],
    acoustic_tracks: Mapping[str, Sequence[float]],
) -> float:
    if event_type == "vowel_nucleus":
        values = (
            _track_value(class_tracks.get("vowel", ()), idx),
            _track_value(acoustic_tracks.get("voicing", ()), idx),
            max(
                _track_value(acoustic_tracks.get("nucleus_likelihood", ()), idx),
                _track_value(acoustic_tracks.get("world_nucleus", ()), idx),
            ),
        )
    elif event_type == "sonorant_constriction":
        values = (
            _track_value(acoustic_tracks.get("sonorant_onset_likelihood", ()), idx),
            _track_value(acoustic_tracks.get("voicing", ()), idx),
            1.0 - _track_value(class_tracks.get("silence", ()), idx),
        )
    else:
        values = (
            _track_value(acoustic_tracks.get("transition_likelihood", ()), idx),
            1.0 - _track_value(class_tracks.get("silence", ()), idx),
            max(
                _track_value(class_tracks.get("consonant", ()), idx),
                _track_value(class_tracks.get("vowel", ()), idx),
            ),
        )
    return max(0.0, min(1.0, _mean(values)))


def _sharpness(values: Sequence[float], idx: int) -> float:
    if not values:
        return 0.0
    lo = max(0, int(idx) - 3)
    hi = min(len(values), int(idx) + 4)
    local = list(values[lo:hi])
    if not local:
        return 0.0
    return max(0.0, _safe_float(values[idx]) - _mean(local))


def _frame_shift_ms(times: Sequence[float]) -> float:
    diffs = [
        max(0.0, _safe_float(times[idx]) - _safe_float(times[idx - 1]))
        for idx in range(1, len(times))
        if _safe_float(times[idx]) > _safe_float(times[idx - 1])
    ]
    return float(median(diffs)) if diffs else 5.0


def _duration_ms(times: Sequence[float], *, frame_shift_ms: float) -> float:
    if not times:
        return 0.0
    return _safe_float(times[-1]) + max(0.001, float(frame_shift_ms))


def _track_value(values: Sequence[float], idx: int) -> float:
    if idx < 0 or idx >= len(values):
        return 0.0
    return _safe_float(values[idx])


def _tracks_from_payload(payload: object, names: Iterable[str]) -> dict[str, list[float]]:
    if not isinstance(payload, Mapping):
        return {str(name): [] for name in names}
    out: dict[str, list[float]] = {}
    for name in names:
        values = payload.get(str(name), ())
        out[str(name)] = _float_list(values)
    return out


def _float_list(values: object) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [_safe_float(value) for value in values]


def _validated_numeric_sequence(values: object, *, field: str) -> tuple[list[float], list[str]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [], [f"{field}_invalid"]
    out: list[float] = []
    errors: list[str] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            errors.append(f"{field}_non_numeric")
            continue
        if not math.isfinite(number):
            errors.append(f"{field}_non_finite")
            continue
        out.append(number)
    return out, errors


def _validate_track_mapping(
    payload: object,
    names: Iterable[str],
    *,
    frame_count: int,
    field: str,
    require_all: bool,
) -> list[str]:
    if not isinstance(payload, Mapping):
        return [f"{field}_missing"]
    errors: list[str] = []
    present = 0
    for name in names:
        key = str(name)
        if key not in payload:
            if require_all:
                errors.append(f"{field}.missing:{key}")
            continue
        values, value_errors = _validated_numeric_sequence(payload.get(key, ()), field=f"{field}.{key}")
        errors.extend(value_errors)
        if frame_count > 0 and len(values) != frame_count:
            errors.append(f"{field}.length_mismatch:{key}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{field}.range_invalid:{key}")
        if values:
            present += 1
    if not require_all and present == 0:
        errors.append(f"{field}_missing")
    return errors


def _filename_slot_from_mapping(payload: Mapping[str, object], *, default_wav: str) -> FilenameSlot | None:
    try:
        return FilenameSlot(
            wav=str(payload.get("wav", "") or default_wav),
            slot_index=int(payload.get("slot_index", 0) or 0),
            token=str(payload.get("token", "")),
            onset=str(payload.get("onset", "")),
            vowel=str(payload.get("vowel", "")),
            onset_phones=tuple(str(item) for item in payload.get("onset_phones", ()) or ()),
            vowel_phone=str(payload.get("vowel_phone", "")),
            phone_start_index=int(payload.get("phone_start_index", 0) or 0),
            vowel_phone_index=int(payload.get("vowel_phone_index", 0) or 0),
            warnings=tuple(str(item) for item in payload.get("warnings", ()) or ()),
            schema_version=int(payload.get("schema_version", 1) or 1),
        )
    except Exception:
        return None


def _mean(values: Iterable[float]) -> float:
    items = [_safe_float(value) for value in values]
    return sum(items) / float(len(items)) if items else 0.0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _round_ms(value: object) -> float:
    return round(_safe_float(value), 3)


def _round_score(value: object) -> float:
    return round(max(0.0, min(1.0, _safe_float(value))), 6)


def _json_safe(value: object) -> object:
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "ACOUSTIC_EVIDENCE_DECODER_VERSION",
    "ACOUSTIC_EVIDENCE_EXTRACTOR_VERSION",
    "ACOUSTIC_EVIDENCE_SCHEMA_VERSION",
    "build_acoustic_evidence_pack",
    "candidate_priors_from_evidence_pack",
    "expected_phones_from_evidence_pack",
    "filename_slots_from_evidence_pack",
    "frame_posterior_from_evidence_pack",
    "read_evidence_jsonl",
    "runtime_events_from_evidence_pack",
    "summarize_event_candidate_reliability",
    "validate_acoustic_evidence_pack",
    "write_evidence_jsonl",
]
