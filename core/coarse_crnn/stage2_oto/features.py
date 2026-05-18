from __future__ import annotations

import math
from statistics import mean

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import (
    BoundaryCandidate,
    BoundaryDecodeResult,
    DecodedOtoRow,
    role_label_preferences,
)
from core.coarse_crnn.stage2_oto.types import (
    ALIAS_TYPE_VOCAB,
    ANCHOR_FIELDS,
    FORMAT_VOCAB,
    LANGUAGE_VOCAB,
    ROLE_VOCAB,
    Stage2FeatureBatch,
    Stage2FeatureRow,
    vocab_index,
)


FIELD_LABEL_PREFERENCES: dict[str, tuple[str, ...]] = {
    "offset_abs": ("syllable_onset", "consonant_onset", "silence_boundary"),
    "overlap_abs": ("syllable_onset", "consonant_onset", "vowel_start"),
    "pre_abs": ("vowel_start", "consonant_onset", "vowel_stable"),
    "consonant_abs": ("vowel_stable", "vowel_start", "transition_peak"),
    "cutoff_abs": ("vowel_end", "silence_boundary", "transition_peak"),
}


def build_stage2_feature_batch(
    *,
    decoded: BoundaryDecodeResult,
    candidates: list[BoundaryCandidate],
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    model_quality: float | None = None,
    audio_reliability: float | None = None,
) -> Stage2FeatureBatch:
    rows = tuple(
        build_stage2_feature_row(
            row=row,
            row_index=idx,
            row_count=len(decoded.rows),
            candidates=candidates,
            active_start_ms=active_start_ms,
            active_end_ms=active_end_ms if active_end_ms is not None else decoded.duration_ms,
            model_quality=model_quality,
            audio_reliability=audio_reliability,
        )
        for idx, row in enumerate(decoded.rows)
    )
    numeric_dim = len(rows[0].numeric) if rows else default_numeric_dim()
    return Stage2FeatureBatch(rows=rows, numeric_dim=int(numeric_dim))


def build_stage2_feature_row(
    *,
    row: DecodedOtoRow,
    row_index: int,
    row_count: int,
    candidates: list[BoundaryCandidate],
    active_start_ms: float,
    active_end_ms: float,
    model_quality: float | None,
    audio_reliability: float | None,
) -> Stage2FeatureRow:
    spec = row.spec
    duration = max(1.0, float(spec.duration_ms or 0.0))
    active_start = _clamp(float(active_start_ms), 0.0, duration)
    active_end = _clamp(float(active_end_ms), active_start + 1.0, duration)
    active_span = max(1.0, active_end - active_start)
    role = normalize_role(spec.role)
    prev_role = _role_from_neighbor(spec.prev_alias, fallback="other")
    next_role = _role_from_neighbor(spec.next_alias, fallback="other")
    alias_type = str((spec.meta or {}).get("alias_type", "") or role or "other").strip().lower()
    language = _normalize_vocab_text(spec.language, LANGUAGE_VOCAB)
    format_type = _normalize_vocab_text(spec.format_type, FORMAT_VOCAB)
    anchors = _row_anchor_tuple(row)
    diffs = (
        max(0.0, anchors[1] - anchors[0]),
        max(0.0, anchors[2] - anchors[1]),
        max(0.0, anchors[3] - anchors[2]),
        max(0.0, anchors[4] - anchors[3]),
    )
    slot_count = max(1, int(spec.slot_count or 1))
    slot_index = min(max(0, int(spec.slot_index or 0)), slot_count - 1)
    row_count_eff = max(1, int(row_count or 1))

    numeric: list[float] = [
        _norm(duration, 10000.0),
        _norm(active_start, duration),
        _norm(active_end, duration),
        _norm(active_span, duration),
        float(slot_index) / float(max(1, slot_count - 1)),
        _norm(float(slot_count), 32.0),
        float(max(0, int(row_index))) / float(max(1, row_count_eff - 1)),
        _norm(float(row_count_eff), 64.0),
        _clamp01(float(model_quality) if model_quality is not None else float(row.quality_score or 0.0)),
        _clamp01(float(audio_reliability) if audio_reliability is not None else 0.5),
        _clamp01(float(row.anchors.confidence or 0.0)),
        1.0 if bool(row.fallback_used) else 0.0,
    ]
    numeric.extend(_norm(anchor, duration) for anchor in anchors)
    numeric.extend(_norm(value, duration) for value in diffs)
    for field, anchor in zip(ANCHOR_FIELDS, anchors):
        cand = nearest_candidate_for_field(
            field=field,
            role=role,
            anchor_ms=anchor,
            candidates=candidates,
            duration_ms=duration,
        )
        if cand is None:
            numeric.extend([_norm(anchor, duration), 0.0, 0.0, 1.0])
        else:
            dt_ms = float(cand.time_ms) - float(anchor)
            numeric.extend(
                [
                    _norm(float(cand.time_ms), duration),
                    _clamp01(float(cand.score)),
                    _signed_norm(dt_ms, duration),
                    0.0,
                ]
            )
    return Stage2FeatureRow(
        numeric=tuple(float(x) for x in numeric),
        role_id=vocab_index(role, ROLE_VOCAB),
        prev_role_id=vocab_index(prev_role, ROLE_VOCAB),
        next_role_id=vocab_index(next_role, ROLE_VOCAB),
        alias_type_id=vocab_index(alias_type, ALIAS_TYPE_VOCAB),
        language_id=vocab_index(language, LANGUAGE_VOCAB),
        format_id=vocab_index(format_type, FORMAT_VOCAB),
        base_anchors_ms=tuple(float(v) for v in anchors),
    )


def nearest_candidate_for_field(
    *,
    field: str,
    role: str,
    anchor_ms: float,
    candidates: list[BoundaryCandidate],
    duration_ms: float,
) -> BoundaryCandidate | None:
    labels = FIELD_LABEL_PREFERENCES.get(str(field), ())
    role_labels = role_label_preferences(role)
    label_set = set(labels) | set(role_labels)
    rows = [item for item in candidates if str(item.kind) in label_set]
    if not rows:
        rows = list(candidates)
    if not rows:
        return None
    duration = max(1.0, float(duration_ms))
    anchor = float(anchor_ms)
    window = max(160.0, min(650.0, duration * 0.18))
    best: BoundaryCandidate | None = None
    best_score = -1.0e9
    for item in rows:
        dist = abs(float(item.time_ms) - anchor)
        proximity = max(0.0, 1.0 - (dist / window))
        label_bonus = 0.12 if str(item.kind) in labels else 0.0
        source_bonus = _candidate_source_bonus(field=field, item=item)
        score = float(item.score) * 0.76 + proximity * 0.22 + label_bonus + source_bonus
        if score > best_score:
            best = item
            best_score = score
    return best


def _candidate_source_bonus(*, field: str, item: BoundaryCandidate) -> float:
    if "model:phoneme_boundary" not in str(item.source or ""):
        return 0.0
    kind = str(item.kind or "")
    if kind == "silence_boundary":
        return -0.06
    bonus = 0.08
    if str(field) == "pre_abs" and kind in {"vowel_start", "consonant_onset", "vowel_stable"}:
        bonus += 0.10
    elif str(field) in {"offset_abs", "overlap_abs"} and kind in {"syllable_onset", "consonant_onset"}:
        bonus += 0.05
    elif str(field) == "consonant_abs" and kind in {"vowel_stable", "vowel_start"}:
        bonus += 0.07
    elif str(field) == "cutoff_abs" and kind == "vowel_end":
        bonus += 0.05
    return bonus


def candidate_distance_for_anchors(
    *,
    anchors_ms: tuple[float, float, float, float, float],
    role: str,
    candidates: list[BoundaryCandidate],
    duration_ms: float,
) -> float:
    distances: list[float] = []
    for field, anchor in zip(ANCHOR_FIELDS, anchors_ms):
        cand = nearest_candidate_for_field(
            field=field,
            role=role,
            anchor_ms=float(anchor),
            candidates=candidates,
            duration_ms=duration_ms,
        )
        if cand is not None:
            distances.append(abs(float(cand.time_ms) - float(anchor)))
    return float(max(distances)) if distances else 0.0


def default_numeric_dim() -> int:
    return 12 + 5 + 4 + (len(ANCHOR_FIELDS) * 4)


def _row_anchor_tuple(row: DecodedOtoRow) -> tuple[float, float, float, float, float]:
    anchors = row.anchors
    return (
        float(anchors.offset_abs),
        float(anchors.overlap_abs),
        float(anchors.pre_abs),
        float(anchors.consonant_abs),
        float(anchors.cutoff_abs),
    )


def _role_from_neighbor(alias: str, *, fallback: str) -> str:
    text = str(alias or "").strip()
    if not text:
        return fallback
    if text in {"br", "BR"}:
        return "br"
    if text.lower() in {"endbr", "end"}:
        return "endbr"
    if " " in text:
        return "vc"
    if text.startswith("-"):
        return "-cv"
    if text.endswith("-"):
        return "v-"
    return "cv"


def _normalize_vocab_text(value: object, vocab: tuple[str, ...]) -> str:
    text = str(value or "").strip().lower()
    if text in vocab:
        return text
    return "other"


def _norm(value: float, denom: float) -> float:
    d = max(1.0e-6, float(denom))
    return _clamp(float(value) / d, 0.0, 1.5)


def _signed_norm(value: float, denom: float) -> float:
    d = max(1.0e-6, float(denom))
    return _clamp(float(value) / d, -1.0, 1.0)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    if math.isnan(float(value)):
        return float(lo)
    return max(float(lo), min(float(hi), float(value)))


def mean_feature_value(rows: list[Stage2FeatureRow], index: int) -> float:
    values = [float(row.numeric[index]) for row in rows if len(row.numeric) > index]
    return float(mean(values)) if values else 0.0


__all__ = [
    "FIELD_LABEL_PREFERENCES",
    "build_stage2_feature_batch",
    "build_stage2_feature_row",
    "candidate_distance_for_anchors",
    "default_numeric_dim",
    "nearest_candidate_for_field",
]
