from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.timing.timing_anchor_profiles import get_anchor_profile
from core.timing.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock

from .kana import parse_kana_text
from .slot_viterbi import ExpectedSlot
from .types import DecodedEvent, FramePosterior, is_vowel_phone


@dataclass(frozen=True)
class OtoTiming:
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.offset, self.consonant, self.cutoff, self.preutterance, self.overlap)


@dataclass(frozen=True)
class OtoTemplateRow:
    wav: str
    alias: str
    timing: OtoTiming
    raw_line: str = ""


@dataclass(frozen=True)
class OtoAnchor:
    anchor_abs_ms: float
    score: float = 1.0
    role: str = "cv_boundary"
    frame_index: int | None = None
    vowel_nucleus_abs_ms: float | None = None
    vowel_start_abs_ms: float | None = None
    vowel_end_abs_ms: float | None = None
    previous_vowel_end_abs_ms: float | None = None
    next_onset_abs_ms: float | None = None
    next_vowel_abs_ms: float | None = None
    boundary_confidence: float = 0.0
    nucleus_confidence: float = 0.0
    expected_phone: str | None = None
    expected_phone_index: int | None = None
    slot_index: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasTargetCandidate:
    phone_index: int | None
    score: float
    reason: str


@dataclass(frozen=True)
class OtoAdapterConfig:
    mode: str = "template-preserve"
    language: str = "japanese"
    format_type: str = "CV"
    alias_type: str = "auto"
    min_anchor_score: float = 0.05
    max_anchor_shift_ms: float = 160.0
    pre_target_ms: float = 120.0
    ovl_gap_ms: float = 35.0
    cons_gap_ms: float = 40.0
    tail_margin_ms: float = 50.0
    previous_tail_keep_ms: float = 40.0
    vowel_nucleus_left_ratio: float = 0.46
    vowel_nucleus_right_ratio: float = 0.54
    vc_pre_max_ms: float = 80.0
    preserve_source_timing: bool = False


@dataclass(frozen=True)
class AdaptedOtoRow:
    wav: str
    alias: str
    timing: OtoTiming
    source_timing: OtoTiming | None
    anchor: OtoAnchor | None
    mode: str
    warnings: tuple[str, ...] = ()
    applied_rules: tuple[str, ...] = ()

    def format_line(self) -> str:
        return format_oto_line(self.wav, self.alias, self.timing)

    def to_json_dict(self) -> dict:
        payload = {
            "wav": self.wav,
            "alias": self.alias,
            "mode": self.mode,
            "timing": timing_to_dict(self.timing),
            "absolute": absolute_oto_positions(self.timing),
            "warnings": list(self.warnings),
            "applied_rules": list(self.applied_rules),
        }
        if self.source_timing is not None:
            payload["source_timing"] = timing_to_dict(self.source_timing)
            payload["source_absolute"] = absolute_oto_positions(self.source_timing)
        if self.anchor is not None:
            payload["anchor"] = {
                "anchor_abs_ms": self.anchor.anchor_abs_ms,
                "score": self.anchor.score,
                "role": self.anchor.role,
                "frame_index": self.anchor.frame_index,
                "vowel_nucleus_abs_ms": self.anchor.vowel_nucleus_abs_ms,
                "vowel_start_abs_ms": self.anchor.vowel_start_abs_ms,
                "vowel_end_abs_ms": self.anchor.vowel_end_abs_ms,
                "previous_vowel_end_abs_ms": self.anchor.previous_vowel_end_abs_ms,
                "next_onset_abs_ms": self.anchor.next_onset_abs_ms,
                "next_vowel_abs_ms": self.anchor.next_vowel_abs_ms,
                "boundary_confidence": self.anchor.boundary_confidence,
                "nucleus_confidence": self.anchor.nucleus_confidence,
                "expected_phone": self.anchor.expected_phone,
                "expected_phone_index": self.anchor.expected_phone_index,
                "slot_index": self.anchor.slot_index,
                "warnings": list(self.anchor.warnings),
            }
        return payload


def load_oto_template_rows(path: str | Path) -> list[OtoTemplateRow]:
    rows: list[OtoTemplateRow] = []
    for line in read_text_with_fallback(str(path)).splitlines():
        row = parse_template_oto_line(line)
        if row is not None:
            rows.append(row)
    return rows


def load_oto_template_rows_alias_only(path: str | Path) -> list[OtoTemplateRow]:
    rows = load_oto_template_rows(path)
    return [
        replace(
            row,
            timing=OtoTiming(
                offset=0.0,
                consonant=0.0,
                cutoff=0.0,
                preutterance=0.0,
                overlap=0.0,
            ),
        )
        for row in rows
    ]


def parse_template_oto_line(line: str) -> OtoTemplateRow | None:
    parsed = parse_oto_line(line)
    if not parsed:
        return None
    return OtoTemplateRow(
        wav=str(parsed["wav"]),
        alias=str(parsed["alias"]),
        timing=OtoTiming(
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        ),
        raw_line=line.rstrip("\n"),
    )


def adapt_template_row(
    row: OtoTemplateRow,
    anchor: OtoAnchor | None,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig | None = None,
) -> AdaptedOtoRow:
    cfg = config or OtoAdapterConfig()
    if not cfg.preserve_source_timing:
        return _bootstrap_template_alias_only(
            row,
            anchor,
            file_duration_ms=file_duration_ms,
            config=cfg,
        )
    if anchor is None:
        if _is_degenerate_timing(row.timing):
            fallback_anchor = _fallback_anchor_for_row(
                row,
                file_duration_ms=file_duration_ms,
                config=cfg,
                warning="missing_anchor_zero_template",
            )
            boot = bootstrap_row(
                row.wav,
                row.alias,
                fallback_anchor,
                file_duration_ms=file_duration_ms,
                config=cfg,
            )
            return replace(
                boot,
                source_timing=row.timing,
                mode="template-bootstrap",
                warnings=tuple(dict.fromkeys((*boot.warnings, "missing_anchor_zero_template"))),
            )
        return _kept_row(row, cfg, "missing_anchor")
    if _is_degenerate_timing(row.timing):
        boot = bootstrap_row(
            row.wav,
            row.alias,
            anchor,
            file_duration_ms=file_duration_ms,
            config=cfg,
        )
        bootstrap_warnings = [*boot.warnings, "zero_template_bootstrap"]
        if anchor.score < cfg.min_anchor_score:
            bootstrap_warnings.append(f"low_anchor_score:{anchor.score:.3f}")
        return replace(
            boot,
            source_timing=row.timing,
            mode="template-bootstrap",
            warnings=tuple(dict.fromkeys(bootstrap_warnings)),
        )

    if anchor.score < cfg.min_anchor_score:
        return _kept_row(row, cfg, f"low_anchor_score:{anchor.score:.3f}")

    current_anchor = row.timing.offset + row.timing.preutterance
    shift = anchor.anchor_abs_ms - current_anchor
    warnings = list(anchor.warnings)
    allow_large_slot_shift = "slot_decoded_event" in anchor.warnings
    if abs(shift) > cfg.max_anchor_shift_ms and not allow_large_slot_shift:
        limited = _clamp(shift, -cfg.max_anchor_shift_ms, cfg.max_anchor_shift_ms)
        anchor = replace(anchor, anchor_abs_ms=current_anchor + limited)
        warnings.append(f"anchor_shift_limited:{shift:.1f}->{limited:.1f}")
    elif abs(shift) > cfg.max_anchor_shift_ms and allow_large_slot_shift:
        warnings.append(f"slot_anchor_shift_allowed:{shift:.1f}")

    alias_type = _alias_type_for_row(row.alias, cfg.alias_type)
    profile = get_anchor_profile(cfg.language, cfg.format_type, alias_type)
    ctx = AnchorTimingContext(
        file_duration_ms=file_duration_ms,
        timeline_start_ms=0.0,
        timeline_end_ms=file_duration_ms,
        anchor_abs_ms=anchor.anchor_abs_ms,
        next_onset_abs_ms=anchor.next_onset_abs_ms,
        next_vowel_abs_ms=anchor.next_vowel_abs_ms,
        alias_type=alias_type,
        language=cfg.language,
        format_type=cfg.format_type,
        mapping_confidence=max(0.0, min(1.0, anchor.score)),
    )
    result = apply_anchor_lock(row.timing.as_tuple(), ctx, profile)
    timing = OtoTiming(
        offset=result.offset,
        consonant=result.consonant,
        cutoff=result.cutoff,
        preutterance=result.pre,
        overlap=result.ovl,
    )
    if profile is None:
        timing = _direct_anchor_shift(row.timing, anchor.anchor_abs_ms)
        warnings.append("profile_missing_direct_anchor")
    return AdaptedOtoRow(
        wav=row.wav,
        alias=row.alias,
        timing=_validate_timing(timing, file_duration_ms=file_duration_ms),
        source_timing=row.timing,
        anchor=anchor,
        mode="template-preserve",
        warnings=tuple(warnings),
        applied_rules=tuple(result.applied_rules),
    )


def bootstrap_row(
    wav: str,
    alias: str,
    anchor: OtoAnchor | None,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig | None = None,
) -> AdaptedOtoRow:
    cfg = config or OtoAdapterConfig(mode="bootstrap")
    if anchor is None:
        timing = _validate_timing(
            OtoTiming(0.0, cfg.pre_target_ms + cfg.cons_gap_ms, -max(180.0, file_duration_ms), cfg.pre_target_ms, max(0.0, cfg.pre_target_ms - cfg.ovl_gap_ms)),
            file_duration_ms=file_duration_ms,
        )
        return AdaptedOtoRow(wav=wav, alias=alias, timing=timing, source_timing=None, anchor=None, mode="bootstrap", warnings=("missing_anchor",))

    anchor_abs = _clamp(anchor.anchor_abs_ms, 0.0, file_duration_ms)
    alias_type = _alias_type_for_row(alias, cfg.alias_type)
    if alias_type == "v":
        nucleus_abs = _clamp(anchor.vowel_nucleus_abs_ms if anchor.vowel_nucleus_abs_ms is not None else anchor_abs, 0.0, file_duration_ms)
        vowel_start = _clamp(
            anchor.vowel_start_abs_ms if anchor.vowel_start_abs_ms is not None else max(0.0, nucleus_abs - 120.0),
            0.0,
            file_duration_ms,
        )
        vowel_end = _clamp(
            anchor.vowel_end_abs_ms if anchor.vowel_end_abs_ms is not None else min(file_duration_ms, nucleus_abs + 180.0),
            vowel_start,
            file_duration_ms,
        )
        left_span = max(28.0, nucleus_abs - vowel_start)
        right_span = max(36.0, vowel_end - nucleus_abs)
        pre = max(56.0, left_span * (1.0 + max(0.0, min(1.0, cfg.vowel_nucleus_left_ratio))))
        offset = max(nucleus_abs - pre, 0.0)
        overlap = max(0.0, pre - max(18.0, left_span * 0.72))
        consonant = pre + max(24.0, right_span * max(0.35, min(1.0, cfg.vowel_nucleus_right_ratio)))
        cutoff = -(max(consonant + 8.0, (vowel_end - offset) + cfg.tail_margin_ms))
        timing = _validate_timing(
            OtoTiming(offset=offset, consonant=consonant, cutoff=cutoff, preutterance=pre, overlap=overlap),
            file_duration_ms=file_duration_ms,
        )
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=timing,
            source_timing=None,
            anchor=anchor,
            mode="bootstrap",
            warnings=anchor.warnings,
        )
    elif alias_type == "vcv":
        prev_end = anchor.previous_vowel_end_abs_ms
        if prev_end is None:
            prev_end = max(0.0, anchor_abs - cfg.pre_target_ms)
        offset = max(float(prev_end) - cfg.previous_tail_keep_ms, 0.0)
    elif alias_type == "vc":
        offset = max(anchor_abs - (cfg.pre_target_ms * 0.75), 0.0)
    else:
        offset = max(anchor_abs - cfg.pre_target_ms, 0.0)
    pre = max(anchor_abs - offset, 0.0)
    if alias_type == "vc":
        pre = min(pre, max(12.0, float(cfg.vc_pre_max_ms)))
        offset = max(anchor_abs - pre, 0.0)
    overlap = max(0.0, pre - cfg.ovl_gap_ms)
    consonant = pre + cfg.cons_gap_ms
    if alias_type == "vc":
        cut_gap = _role_cut_gap_ms(cfg, alias_type)
        cutoff = -(max(consonant + 8.0, consonant + cut_gap))
    else:
        vowel_end = anchor.vowel_end_abs_ms if anchor.vowel_end_abs_ms is not None else min(file_duration_ms, anchor_abs + 180.0)
        cutoff = -(max(consonant + 8.0, float(vowel_end) - offset + cfg.tail_margin_ms))
    timing = _validate_timing(
        OtoTiming(offset=offset, consonant=consonant, cutoff=cutoff, preutterance=pre, overlap=overlap),
        file_duration_ms=file_duration_ms,
    )
    return AdaptedOtoRow(
        wav=wav,
        alias=alias,
        timing=timing,
        source_timing=None,
        anchor=anchor,
        mode="bootstrap",
        warnings=anchor.warnings,
    )


def anchors_from_prediction(
    posterior: FramePosterior,
    decoded_events: Sequence[DecodedEvent | Mapping[str, object]],
) -> list[OtoAnchor]:
    anchors: list[OtoAnchor] = []
    previous_vowel_end: float | None = None
    decoded = [_event_to_dict(event) for event in decoded_events]
    for idx, event in enumerate(decoded):
        if event["label"] not in {"cv_boundary", "phone_change", "vowel_nucleus"}:
            continue
        anchor_ms = float(event["time_ms"])
        span = estimate_vowel_span(posterior, anchor_ms)
        nucleus_time, nucleus_conf, nucleus_warnings = estimate_vowel_nucleus(
            posterior,
            anchor_ms,
            vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
            vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
        )
        next_vowel = next((float(item["time_ms"]) for item in decoded[idx + 1 :] if item["label"] == "vowel_nucleus"), None)
        warning = "slot_decoded_event" if event.get("expected_phone") else "decoded_event"
        role = str(event["label"])
        boundary_conf = _anchor_boundary_confidence(posterior, anchor_ms, role=role)
        if role == "vowel_nucleus":
            selected_nucleus = anchor_ms
            selected_nucleus_conf = max(float(event.get("score") or 0.0), nucleus_conf)
        else:
            selected_nucleus = nucleus_time if nucleus_time is not None else next_vowel
            selected_nucleus_conf = nucleus_conf
        anchor_warnings = [warning, f"decoded_order:{idx}"]
        anchor_warnings.extend(nucleus_warnings)
        if selected_nucleus_conf < 0.35:
            anchor_warnings.append(f"low_nucleus_confidence:{selected_nucleus_conf:.3f}")
        if boundary_conf < 0.30:
            anchor_warnings.append(f"low_boundary_confidence:{boundary_conf:.3f}")
        anchor = OtoAnchor(
            anchor_abs_ms=anchor_ms,
            score=float(event.get("score") or 0.0),
            role=role,
            frame_index=event.get("frame_index"),
            vowel_nucleus_abs_ms=selected_nucleus,
            vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
            vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
            previous_vowel_end_abs_ms=previous_vowel_end,
            next_vowel_abs_ms=next_vowel,
            boundary_confidence=boundary_conf,
            nucleus_confidence=selected_nucleus_conf,
            expected_phone=str(event.get("expected_phone") or "") or None,
            expected_phone_index=_int_or_none(event.get("expected_phone_index")),
            slot_index=_int_or_none(event.get("slot_index")),
            warnings=tuple(anchor_warnings),
        )
        anchors.append(anchor)
        previous_vowel_end = anchor.vowel_end_abs_ms or previous_vowel_end
    return anchors


def assign_template_row_anchors(
    posterior: FramePosterior,
    decoded_events: Sequence[DecodedEvent | Mapping[str, object]],
    template_rows: Sequence[OtoTemplateRow],
    *,
    min_score: float = 0.03,
    use_source_timing_prior: bool = True,
    expected_phones: Sequence[str] | None = None,
) -> list[OtoAnchor | None]:
    if not template_rows:
        return []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return [None for _ in template_rows]
    duration_ms = float(max(times[-1], 1.0))
    candidates = _posterior_anchor_candidates(posterior, min_score=min_score)
    for event in anchors_from_prediction(posterior, decoded_events):
        candidates.append(event)
    candidates.sort(key=lambda item: (item.anchor_abs_ms, -item.score))

    used: set[int] = set()
    out: list[OtoAnchor | None] = []
    last_time = -1.0
    last_target_phone_index: int | None = None
    alias_targets = _assign_alias_target_indices(template_rows, expected_phones or ())
    decoded_count = sum(1 for item in candidates if "slot_decoded_event" in item.warnings)
    for row_idx, row in enumerate(template_rows):
        role = _alias_type_for_row(row.alias, "auto")
        preferred_labels = _event_labels_for_alias_role(role)
        target_phone_index = alias_targets[row_idx] if row_idx < len(alias_targets) else None
        expected_time = _template_expected_time(
            row,
            row_idx,
            len(template_rows),
            duration_ms,
            use_source_timing_prior=use_source_timing_prior,
        )
        best_idx: int | None = None
        best_score = -1e9
        for cand_idx, candidate in enumerate(candidates):
            if cand_idx in used:
                continue
            same_target_reuse = (
                target_phone_index is not None
                and candidate.expected_phone_index == target_phone_index
                and last_target_phone_index == target_phone_index
            )
            target_backtrack = (
                target_phone_index is not None
                and last_target_phone_index is not None
                and target_phone_index < last_target_phone_index
            )
            if candidate.anchor_abs_ms + 1e-5 < last_time + 4.0 and not same_target_reuse and not target_backtrack:
                continue
            if target_phone_index is not None and candidate.expected_phone_index != target_phone_index:
                continue
            label_bonus = 0.25 if candidate.role in preferred_labels else -0.05
            if "slot_decoded_event" in candidate.warnings:
                decoded_bonus = 0.85 if candidate.expected_phone_index == target_phone_index else 0.62
            elif "decoded_event" in candidate.warnings:
                decoded_bonus = 0.35
            else:
                decoded_bonus = 0.0
            distance_penalty = abs(candidate.anchor_abs_ms - expected_time) / duration_ms
            order_penalty = _decoded_order_penalty(
                candidate,
                row_idx=row_idx,
                row_count=len(template_rows),
                decoded_count=decoded_count,
            )
            score = float(candidate.score) + label_bonus + decoded_bonus - (0.90 * distance_penalty) - order_penalty
            if score > best_score:
                best_score = score
                best_idx = cand_idx
        if best_idx is None:
            if target_phone_index is not None:
                target_phone_index = None
                for cand_idx, candidate in enumerate(candidates):
                    if cand_idx in used:
                        continue
                    if candidate.anchor_abs_ms + 1e-5 < last_time + 4.0:
                        continue
                    label_bonus = 0.25 if candidate.role in preferred_labels else -0.05
                    decoded_bonus = 0.35 if "decoded_event" in candidate.warnings else 0.0
                    distance_penalty = abs(candidate.anchor_abs_ms - expected_time) / duration_ms
                    score = float(candidate.score) + label_bonus + decoded_bonus - (0.90 * distance_penalty)
                    if score > best_score:
                        best_score = score
                        best_idx = cand_idx
            if best_idx is not None:
                pass
        if best_idx is None:
            synthetic = _synthetic_anchor_from_expected(row, expected_time, posterior, role)
            out.append(synthetic)
            last_time = synthetic.anchor_abs_ms
            continue
        if target_phone_index is None:
            used.add(best_idx)
        chosen = candidates[best_idx]
        span = estimate_vowel_span(posterior, chosen.anchor_abs_ms)
        warnings = _anchor_warnings_with_target(chosen, role, target_phone_index)
        if (
            target_phone_index is not None
            and last_target_phone_index is not None
            and target_phone_index < last_target_phone_index
        ):
            warnings.append("alias_target_backtrack")
        chosen = replace(
            chosen,
            role=role,
            vowel_start_abs_ms=span.get("vowel_start_abs_ms", chosen.vowel_start_abs_ms),
            vowel_end_abs_ms=span.get("vowel_end_abs_ms", chosen.vowel_end_abs_ms),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        if out:
            prev = next((item for item in reversed(out) if item is not None), None)
            if prev is not None:
                chosen = replace(chosen, previous_vowel_end_abs_ms=prev.vowel_end_abs_ms)
        out.append(chosen)
        last_time = chosen.anchor_abs_ms
        last_target_phone_index = chosen.expected_phone_index
    return out


def expected_slots_for_template_rows(
    template_rows: Sequence[OtoTemplateRow],
    expected_phones: Sequence[str],
) -> list[ExpectedSlot]:
    expected = [str(phone or "").strip().lower() for phone in expected_phones]
    if not template_rows or not expected:
        return []
    targets = _assign_alias_target_indices(template_rows, expected)
    out: list[ExpectedSlot] = []
    seen: set[tuple[int, str]] = set()
    for row, target_index in zip(template_rows, targets):
        if target_index is None or target_index < 0 or target_index >= len(expected):
            continue
        role = _alias_type_for_row(row.alias, "auto")
        event_label = _slot_event_label_for_alias_role(role)
        _append_expected_slot(
            out,
            seen,
            phone_index=int(target_index),
            phone=expected[target_index],
            role=role,
            event_label=event_label,
        )
        if role == "vc":
            next_vowel_index = _next_vowel_index(expected, int(target_index))
            if next_vowel_index is not None:
                _append_expected_slot(
                    out,
                    seen,
                    phone_index=int(next_vowel_index),
                    phone=expected[next_vowel_index],
                    role="implicit_cv",
                    event_label="cv_boundary",
                )
    return out


def _append_expected_slot(
    out: list[ExpectedSlot],
    seen: set[tuple[int, str]],
    *,
    phone_index: int,
    phone: str,
    role: str,
    event_label: str,
) -> None:
    key = (phone_index, event_label)
    if key in seen:
        return
    seen.add(key)
    out.append(
        ExpectedSlot(
            slot_index=len(out),
            phone_index=phone_index,
            phone=phone,
            role=role,
            event_label=event_label,
        )
    )


def estimate_vowel_span(posterior: FramePosterior, anchor_abs_ms: float, *, threshold: float = 0.35) -> dict:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    vowel = np.asarray(posterior.class_probs.get("vowel", []), dtype=np.float32)
    if times.size == 0 or vowel.shape[0] != times.shape[0]:
        return {}
    start_idx = int(np.searchsorted(times, anchor_abs_ms, side="left"))
    start_idx = max(0, min(start_idx, times.shape[0] - 1))
    while start_idx > 0 and float(vowel[start_idx - 1]) >= threshold:
        start_idx -= 1
    end_idx = int(np.searchsorted(times, anchor_abs_ms, side="left"))
    end_idx = max(0, min(end_idx, times.shape[0] - 1))
    while end_idx + 1 < times.shape[0] and float(vowel[end_idx + 1]) >= threshold:
        end_idx += 1
    return {
        "vowel_start_abs_ms": float(times[start_idx]),
        "vowel_end_abs_ms": float(times[end_idx]),
    }


def estimate_vowel_nucleus(
    posterior: FramePosterior,
    anchor_abs_ms: float,
    *,
    vowel_start_abs_ms: float | None = None,
    vowel_end_abs_ms: float | None = None,
) -> tuple[float | None, float, tuple[str, ...]]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return None, 0.0, ("empty_posterior",)
    vowel = np.asarray(posterior.class_probs.get("vowel", []), dtype=np.float32)
    if vowel.shape[0] != times.shape[0]:
        vowel = np.zeros_like(times)
    rms_track = _normalized_track(posterior, "rms", times)
    voicing_track = _normalized_track(posterior, "voicing", times)
    centroid = _normalized_track(posterior, "spectral_centroid", times)
    formant_stability = 1.0 - np.clip(np.abs(np.gradient(centroid.astype(np.float32))), 0.0, 1.0)

    if vowel_start_abs_ms is None or vowel_end_abs_ms is None:
        span = estimate_vowel_span(posterior, anchor_abs_ms)
        vowel_start_abs_ms = span.get("vowel_start_abs_ms")
        vowel_end_abs_ms = span.get("vowel_end_abs_ms")
    if vowel_start_abs_ms is None or vowel_end_abs_ms is None:
        center_idx = int(np.searchsorted(times, anchor_abs_ms, side="left"))
        center_idx = max(0, min(center_idx, times.shape[0] - 1))
        start_idx = max(0, center_idx - 3)
        end_idx = min(times.shape[0] - 1, center_idx + 3)
    else:
        start_idx = int(np.searchsorted(times, float(vowel_start_abs_ms), side="left"))
        end_idx = int(np.searchsorted(times, float(vowel_end_abs_ms), side="right")) - 1
        start_idx = max(0, min(start_idx, times.shape[0] - 1))
        end_idx = max(start_idx, min(end_idx, times.shape[0] - 1))

    if end_idx <= start_idx:
        return float(times[start_idx]), 0.0, ("nucleus_span_too_short",)

    idx_slice = slice(start_idx, end_idx + 1)
    local_vowel = np.clip(vowel[idx_slice], 0.0, 1.0)
    local_energy = np.clip(rms_track[idx_slice], 0.0, 1.0)
    local_voicing = np.clip(voicing_track[idx_slice], 0.0, 1.0)
    local_formant = np.clip(formant_stability[idx_slice], 0.0, 1.0)
    continuity = np.minimum(local_voicing, _left_context_max(local_voicing, frames=2))
    score = np.clip(
        0.34 * local_vowel + 0.30 * local_energy + 0.22 * local_formant + 0.14 * continuity,
        0.0,
        1.0,
    )
    local_idx = int(np.argmax(score))
    global_idx = int(start_idx + local_idx)
    warnings: list[str] = []
    conf = float(score[local_idx])
    if conf < 0.35:
        warnings.append(f"low_vowel_nucleus_confidence:{conf:.3f}")
    if float(local_voicing[local_idx]) < 0.25:
        warnings.append("weak_voicing_nucleus")
    if float(local_formant[local_idx]) < 0.20:
        warnings.append("unstable_formant_nucleus")
    return float(times[global_idx]), conf, tuple(warnings)


def _anchor_boundary_confidence(
    posterior: FramePosterior,
    anchor_abs_ms: float,
    *,
    role: str,
) -> float:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return 0.0
    idx = _window_idx(times, anchor_abs_ms)
    vowel = _normalized_track(posterior, "vowel", times, source="class")
    consonant = _normalized_track(posterior, "consonant", times, source="class")
    transition = _normalized_track(posterior, "transition_likelihood", times)
    voicing = _normalized_track(posterior, "voicing", times)
    if role == "vowel_nucleus":
        return float(np.clip((0.55 * voicing[idx]) + (0.45 * vowel[idx]), 0.0, 1.0))
    left_cons = float(np.max(consonant[max(0, idx - 3) : idx + 1])) if idx >= 0 else 0.0
    return float(np.clip((0.42 * transition[idx]) + (0.33 * vowel[idx]) + (0.25 * left_cons), 0.0, 1.0))


def format_oto_line(wav: str, alias: str, timing: OtoTiming) -> str:
    return (
        f"{wav}={alias},"
        f"{timing.offset:.3f},{timing.consonant:.3f},{timing.cutoff:.3f},"
        f"{timing.preutterance:.3f},{timing.overlap:.3f}"
    )


def timing_to_dict(timing: OtoTiming) -> dict:
    return {
        "offset": timing.offset,
        "consonant": timing.consonant,
        "cutoff": timing.cutoff,
        "preutterance": timing.preutterance,
        "overlap": timing.overlap,
    }


def absolute_oto_positions(timing: OtoTiming) -> dict:
    return {
        "offset_abs": timing.offset,
        "overlap_abs": timing.offset + timing.overlap,
        "preutterance_abs": timing.offset + timing.preutterance,
        "consonant_abs": timing.offset + timing.consonant,
        "cutoff_abs": timing.offset + abs(timing.cutoff),
    }


def _kept_row(row: OtoTemplateRow, cfg: OtoAdapterConfig, warning: str) -> AdaptedOtoRow:
    return AdaptedOtoRow(
        wav=row.wav,
        alias=row.alias,
        timing=row.timing,
        source_timing=row.timing,
        anchor=None,
        mode=cfg.mode,
        warnings=(warning,),
    )


def _bootstrap_template_alias_only(
    row: OtoTemplateRow,
    anchor: OtoAnchor | None,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    active_anchor = anchor
    warnings: list[str] = ["source_timing_discarded"]
    if active_anchor is None:
        active_anchor = _fallback_anchor_for_row(
            row,
            file_duration_ms=file_duration_ms,
            config=config,
            warning="missing_anchor_alias_only",
        )
    elif active_anchor.score < config.min_anchor_score:
        warnings.append(f"low_anchor_score:{active_anchor.score:.3f}")
    if _is_degenerate_timing(row.timing):
        warnings.append("zero_template_bootstrap")
    boot = bootstrap_row(
        row.wav,
        row.alias,
        active_anchor,
        file_duration_ms=file_duration_ms,
        config=config,
    )
    return replace(
        boot,
        source_timing=None,
        mode="template-bootstrap",
        warnings=tuple(dict.fromkeys((*boot.warnings, *warnings))),
    )


def _fallback_anchor_for_row(
    row: OtoTemplateRow,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig,
    warning: str,
) -> OtoAnchor:
    return OtoAnchor(
        anchor_abs_ms=min(max(config.pre_target_ms, 1.0), max(file_duration_ms * 0.35, 1.0)),
        score=0.0,
        role=_alias_type_for_row(row.alias, config.alias_type),
        warnings=(warning,),
    )


def _role_cut_gap_ms(config: OtoAdapterConfig, alias_type: str) -> float:
    profile = get_anchor_profile(config.language, config.format_type, alias_type)
    if profile is not None:
        return max(8.0, float(profile.cut_gap_target_ms))
    if alias_type == "vc":
        return 28.0
    return max(8.0, float(config.tail_margin_ms))


def _direct_anchor_shift(timing: OtoTiming, anchor_abs_ms: float) -> OtoTiming:
    cons_gap = max(timing.consonant - timing.preutterance, 30.0)
    cut_gap = max(abs(timing.cutoff) - timing.consonant, 50.0)
    ovl_gap = max(timing.preutterance - timing.overlap, 0.0)
    offset = max(anchor_abs_ms - timing.preutterance, 0.0)
    pre = max(anchor_abs_ms - offset, 0.0)
    return OtoTiming(
        offset=offset,
        consonant=pre + cons_gap,
        cutoff=-(pre + cons_gap + cut_gap),
        preutterance=pre,
        overlap=max(0.0, pre - ovl_gap),
    )


def _validate_timing(timing: OtoTiming, *, file_duration_ms: float) -> OtoTiming:
    offset = max(float(timing.offset), 0.0)
    pre = max(float(timing.preutterance), 0.0)
    ovl = max(0.0, min(float(timing.overlap), pre))
    consonant = max(float(timing.consonant), pre + 8.0)
    cutoff_abs = max(abs(float(timing.cutoff)), consonant + 8.0)
    if file_duration_ms > 0.0:
        cutoff_abs = min(max(cutoff_abs, consonant + 8.0), max(consonant + 8.0, file_duration_ms - offset + 80.0))
    return OtoTiming(offset=offset, consonant=consonant, cutoff=-cutoff_abs, preutterance=pre, overlap=ovl)


def _is_degenerate_timing(timing: OtoTiming) -> bool:
    return (
        abs(float(timing.offset)) <= 1.0
        and abs(float(timing.consonant)) <= 1.0
        and abs(float(timing.preutterance)) <= 1.0
        and abs(float(timing.overlap)) <= 1.0
    )


def _alias_type_for_row(alias: str, default_alias_type: str) -> str:
    alias_norm = str(alias or "").strip().lower()
    default_norm = str(default_alias_type or "cv").strip().lower()
    if default_norm in {"cv", "cv_head", "vcv", "vc", "vv", "v"}:
        return default_norm
    if alias_norm.startswith("-"):
        return "cv_head"
    phones = _alias_phones(alias_norm)
    if " " in alias_norm:
        tokens = [part for part in alias_norm.split() if part]
        first = _alias_phones(tokens[0]) if tokens else []
        second = _alias_phones(tokens[1]) if len(tokens) >= 2 else []
        first_vowel = bool(first and is_vowel_phone(first[-1]))
        if first_vowel and second:
            if len(second) == 1 and is_vowel_phone(second[0]):
                return "vv"
            return "vcv" if is_vowel_phone(second[-1]) else "vc"
        return "vcv"
    if len(phones) == 1 and is_vowel_phone(phones[0]):
        return "v"
    if len(phones) >= 2 and not is_vowel_phone(phones[0]) and is_vowel_phone(phones[-1]):
        return "cv"
    return "cv"


def _alias_phones(text: str) -> list[str]:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return []
    if normalized in {"を", "ヲ"}:
        return ["o"]
    if any("\u3040" <= char <= "\u30ff" for char in normalized):
        parsed_kana = parse_kana_text(normalized)
        if parsed_kana:
            return parsed_kana
    kana = _KANA_ALIAS_PHONES.get(normalized)
    if kana is not None:
        return list(kana)
    vowels = {"a", "i", "u", "e", "o", "n"}
    if normalized in vowels:
        return [normalized]
    phones: list[str] = []
    pos = 0
    y_clusters = {
        "ky": ("k", "y"),
        "gy": ("g", "y"),
        "ny": ("n", "y"),
        "hy": ("h", "y"),
        "by": ("b", "y"),
        "py": ("p", "y"),
        "my": ("m", "y"),
        "ry": ("r", "y"),
        "jy": ("j", "y"),
        "dy": ("d", "y"),
        "ty": ("t", "y"),
    }
    clusters = ("sh", "ch", "ts", "dh")
    while pos < len(normalized):
        if normalized[pos] in vowels:
            phones.append(normalized[pos])
            pos += 1
            continue
        y_cluster = next((item for item in y_clusters if normalized.startswith(item, pos)), None)
        if y_cluster:
            phones.extend(y_clusters[y_cluster])
            pos += len(y_cluster)
            continue
        cluster = next((item for item in clusters if normalized.startswith(item, pos)), None)
        if cluster:
            phones.append(cluster)
            pos += len(cluster)
            continue
        char = normalized[pos]
        if char.isalpha():
            phones.append(char)
            pos += 1
            continue
        return []
    return phones


_KANA_ALIAS_PHONES = {
    "あ": ["a"], "い": ["i"], "う": ["u"], "え": ["e"], "お": ["o"], "ん": ["n"],
    "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "u"], "け": ["k", "e"], "こ": ["k", "o"],
    "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "u"], "げ": ["g", "e"], "ご": ["g", "o"],
    "さ": ["s", "a"], "し": ["sh", "i"], "す": ["s", "u"], "せ": ["s", "e"], "そ": ["s", "o"],
    "ざ": ["z", "a"], "じ": ["j", "i"], "ず": ["z", "u"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
    "た": ["t", "a"], "ち": ["ch", "i"], "つ": ["ts", "u"], "て": ["t", "e"], "と": ["t", "o"],
    "だ": ["d", "a"], "で": ["d", "e"], "ど": ["d", "o"],
    "な": ["n", "a"], "に": ["n", "i"], "ぬ": ["n", "u"], "ね": ["n", "e"], "の": ["n", "o"],
    "は": ["h", "a"], "ひ": ["h", "i"], "ふ": ["f", "u"], "へ": ["h", "e"], "ほ": ["h", "o"],
    "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "u"], "べ": ["b", "e"], "ぼ": ["b", "o"],
    "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "u"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],
    "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "u"], "め": ["m", "e"], "も": ["m", "o"],
    "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
    "ら": ["r", "a"], "り": ["r", "i"], "る": ["r", "u"], "れ": ["r", "e"], "ろ": ["r", "o"],
    "わ": ["w", "a"], "を": ["w", "o"],
}


def _event_labels_for_alias_role(role: str) -> tuple[str, ...]:
    if role in {"vcv", "vv", "v"}:
        return ("phone_change", "vowel_nucleus", "cv_boundary")
    if role == "vc":
        return ("phone_change", "cv_boundary", "vowel_nucleus")
    return ("cv_boundary", "phone_change", "vowel_nucleus")


def _slot_event_label_for_alias_role(role: str) -> str:
    if role == "vc":
        return "phone_change"
    if role in {"v", "vv"}:
        return "vowel_nucleus"
    return "cv_boundary"


def _anchor_warnings_with_target(anchor: OtoAnchor, role: str, target_phone_index: int | None) -> list[str]:
    warnings = [*anchor.warnings, f"template_role:{role}"]
    if target_phone_index is not None:
        warnings.append(f"alias_target_phone_index:{target_phone_index}")
    return warnings


def _assign_alias_target_indices(
    rows: Sequence[OtoTemplateRow],
    expected_phones: Sequence[str],
) -> list[int | None]:
    expected = [str(phone or "").strip().lower() for phone in expected_phones]
    if not rows:
        return []
    if not expected:
        return [None for _ in rows]
    roles = [_alias_type_for_row(row.alias, "auto") for row in rows]
    candidate_steps = [
        _alias_target_candidates(row.alias, role, expected)
        or [AliasTargetCandidate(None, -1.0, "unmapped")]
        for row, role in zip(rows, roles)
    ]
    dp: list[list[tuple[float, int | None]]] = []
    row_count = len(rows)
    for row_idx, candidates in enumerate(candidate_steps):
        row_dp: list[tuple[float, int | None]] = []
        for cand_idx, candidate in enumerate(candidates):
            base_score = candidate.score + _alias_target_order_score(
                candidate.phone_index,
                row_idx=row_idx,
                row_count=row_count,
                expected_count=len(expected),
            )
            if row_idx == 0:
                row_dp.append((base_score, None))
                continue
            best_score = -1e9
            best_prev: int | None = None
            for prev_idx, prev_candidate in enumerate(candidate_steps[row_idx - 1]):
                prev_score, _ = dp[row_idx - 1][prev_idx]
                transition = _alias_target_transition_score(
                    prev_candidate.phone_index,
                    candidate.phone_index,
                    prev_role=roles[row_idx - 1],
                    role=roles[row_idx],
                )
                if transition <= -1e8:
                    continue
                score = prev_score + base_score + transition
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            row_dp.append((best_score, best_prev))
        dp.append(row_dp)
    final_scores = [score for score, _ in dp[-1]]
    if not final_scores or max(final_scores) <= -1e8:
        return _assign_alias_target_indices_greedy(rows, expected)
    selected: list[int] = []
    cand_idx = int(np.argmax(np.asarray(final_scores, dtype=np.float32)))
    for row_idx in range(len(candidate_steps) - 1, -1, -1):
        selected.append(cand_idx)
        _score, prev_idx = dp[row_idx][cand_idx]
        if prev_idx is None:
            break
        cand_idx = prev_idx
    selected.reverse()
    if len(selected) != len(rows):
        return _assign_alias_target_indices_greedy(rows, expected)
    return [candidate_steps[row_idx][cand_idx].phone_index for row_idx, cand_idx in enumerate(selected)]


def _assign_alias_target_indices_greedy(
    rows: Sequence[OtoTemplateRow],
    expected: Sequence[str],
) -> list[int | None]:
    out: list[int | None] = []
    last_target: int | None = None
    last_role = ""
    row_count = len(rows)
    for row_idx, row in enumerate(rows):
        role = _alias_type_for_row(row.alias, "auto")
        candidates = _alias_target_candidates(row.alias, role, expected)
        candidates = sorted(
            candidates,
            key=lambda candidate: candidate.score
            + _alias_target_order_score(
                candidate.phone_index,
                row_idx=row_idx,
                row_count=row_count,
                expected_count=len(expected),
            ),
            reverse=True,
        )
        chosen: int | None = None
        for candidate in candidates:
            transition = _alias_target_transition_score(
                last_target,
                candidate.phone_index,
                prev_role=last_role,
                role=role,
            )
            if last_target is None or transition > -1e8:
                chosen = candidate.phone_index
                break
        if chosen is None and candidates:
            chosen = candidates[0].phone_index
        out.append(chosen)
        if chosen is not None:
            last_target = chosen
            last_role = role
    return out


def _alias_target_candidates(alias: str, role: str, expected: Sequence[str]) -> list[AliasTargetCandidate]:
    phones = _alias_phone_sequence(alias)
    if not expected or not phones:
        return []
    out: list[AliasTargetCandidate] = []
    if role == "vc" and len(phones) >= 2:
        left = phones[0]
        right = phones[1:]
        for idx in range(0, max(0, len(expected) - 1)):
            if not _phone_matches(expected[idx], left):
                continue
            if not _vc_right_matches(expected, idx + 1, right):
                continue
            out.append(AliasTargetCandidate(idx + 1, 4.0, "vc_right_consonant"))
        return out
    if role in {"v", "vv"}:
        if role == "vv" and len(phones) >= 2:
            for start in _find_all_phone_sequences(expected, phones):
                out.append(AliasTargetCandidate(start + len(phones) - 1, 3.2, "vv_exact_sequence"))
            if out:
                return out
        target_phone = phones[-1]
        for idx, phone in enumerate(expected):
            if _phone_matches(phone, target_phone) and is_vowel_phone(phone):
                out.append(AliasTargetCandidate(idx, 2.3 if role == "vv" else 1.8, role))
        return out
    for start in _find_all_phone_sequences(expected, phones):
        out.append(AliasTargetCandidate(start + len(phones) - 1, 4.2, "exact_alias_sequence"))
    if out:
        return out
    target_phone = next((phone for phone in reversed(phones) if is_vowel_phone(phone)), phones[-1])
    for idx, phone in enumerate(expected):
        if _phone_matches(phone, target_phone) and is_vowel_phone(phone):
            out.append(AliasTargetCandidate(idx, 1.1, "target_phone_fallback"))
    return out


def _alias_target_order_score(
    phone_index: int | None,
    *,
    row_idx: int,
    row_count: int,
    expected_count: int,
) -> float:
    if phone_index is None or row_count <= 1 or expected_count <= 1:
        return -0.8
    row_pos = float(row_idx) / float(row_count - 1)
    phone_pos = float(phone_index) / float(expected_count - 1)
    return -0.75 * abs(row_pos - phone_pos)


def _alias_target_transition_score(
    prev: int | None,
    cur: int | None,
    *,
    prev_role: str,
    role: str,
) -> float:
    if prev is None or cur is None:
        return -0.25
    if cur < prev:
        return -1e9
    if cur == prev:
        if _can_share_alias_target(prev_role, role):
            return 0.55
        return -1e9
    return -0.015 * float(cur - prev)


def _can_share_alias_target(prev_role: str, role: str) -> bool:
    if prev_role in {"v", "vv"} and role in {"vv", "v"}:
        return True
    return False


def _alias_target_phone_index(
    alias: str,
    role: str,
    expected_phones: Sequence[str],
    *,
    min_target_index: int,
) -> int | None:
    expected = [str(phone or "").strip().lower() for phone in expected_phones]
    phones = _alias_phone_sequence(alias)
    if not expected or not phones:
        return None
    if role == "vc" and len(phones) >= 2:
        left = phones[0]
        right = phones[1:]
        for idx in range(0, max(0, len(expected) - 1)):
            if not _phone_matches(expected[idx], left):
                continue
            if not _vc_right_matches(expected, idx + 1, right):
                continue
            target = idx + 1
            if target >= min_target_index:
                return target
        return None
    if role in {"v", "vv"}:
        if role == "vv" and len(phones) >= 2:
            match = _find_phone_sequence(expected, phones, min_target_index=min_target_index)
            if match is not None:
                return match + len(phones) - 1
        target_phone = phones[-1]
        for idx, phone in enumerate(expected):
            if idx >= min_target_index and _phone_matches(phone, target_phone) and is_vowel_phone(phone):
                return idx
        return None
    match = _find_phone_sequence(expected, phones, min_target_index=min_target_index)
    if match is not None:
        return match + len(phones) - 1
    target_phone = next((phone for phone in reversed(phones) if is_vowel_phone(phone)), phones[-1])
    for idx, phone in enumerate(expected):
        if idx >= min_target_index and _phone_matches(phone, target_phone) and is_vowel_phone(phone):
            return idx
    return None


def _alias_phone_sequence(alias: str) -> list[str]:
    normalized = str(alias or "").strip().lower()
    if not normalized:
        return []
    if " " in normalized:
        phones: list[str] = []
        for token in normalized.split():
            phones.extend(_alias_phones(token))
        return phones
    return _alias_phones(normalized)


def _find_phone_sequence(expected: Sequence[str], phones: Sequence[str], *, min_target_index: int) -> int | None:
    if not expected or not phones or len(phones) > len(expected):
        return None
    for start in range(0, len(expected) - len(phones) + 1):
        target = start + len(phones) - 1
        if target < min_target_index:
            continue
        if _phone_sequence_matches(expected[start : start + len(phones)], phones):
            return start
    return None


def _vc_right_matches(expected: Sequence[str], start_idx: int, right: Sequence[str]) -> bool:
    if not right or start_idx >= len(expected):
        return False
    if _phone_sequence_matches(expected[start_idx : start_idx + len(right)], right):
        return True
    if len(right) == 2 and right[1] == "y" and _phone_matches(expected[start_idx], right[0]):
        return True
    return False


def _phone_sequence_matches(expected: Sequence[str], phones: Sequence[str]) -> bool:
    if len(expected) != len(phones):
        return False
    return all(_phone_matches(expected_phone, phone) for expected_phone, phone in zip(expected, phones))


def _phone_matches(expected_phone: str, alias_phone: str) -> bool:
    expected_norm = str(expected_phone or "").strip().lower()
    alias_norm = str(alias_phone or "").strip().lower()
    if expected_norm == alias_norm:
        return True
    equivalent = {
        "j": {"z", "j"},
        "z": {"z", "j"},
        "h": {"h", "f"},
        "f": {"h", "f"},
    }
    return expected_norm in equivalent.get(alias_norm, set())


def _find_all_phone_sequences(expected: Sequence[str], phones: Sequence[str]) -> list[int]:
    if not expected or not phones or len(phones) > len(expected):
        return []
    out: list[int] = []
    for start in range(0, len(expected) - len(phones) + 1):
        if _phone_sequence_matches(expected[start : start + len(phones)], phones):
            out.append(start)
    return out


def _next_vowel_index(expected: Sequence[str], start_idx: int) -> int | None:
    for idx in range(start_idx + 1, len(expected)):
        if is_vowel_phone(expected[idx]):
            return idx
    return None


def _decoded_order_penalty(
    candidate: OtoAnchor,
    *,
    row_idx: int,
    row_count: int,
    decoded_count: int,
) -> float:
    if "slot_decoded_event" not in candidate.warnings or decoded_count <= 1 or row_count <= 1:
        return 0.0
    order = next((item.split(":", 1)[1] for item in candidate.warnings if item.startswith("decoded_order:")), "")
    if not order.isdigit():
        return 0.0
    decoded_pos = float(int(order)) / float(max(decoded_count - 1, 1))
    row_pos = float(row_idx) / float(max(row_count - 1, 1))
    return 0.65 * abs(decoded_pos - row_pos)


def _template_expected_time(
    row: OtoTemplateRow,
    row_idx: int,
    row_count: int,
    duration_ms: float,
    *,
    use_source_timing_prior: bool,
) -> float:
    if use_source_timing_prior and not _is_degenerate_timing(row.timing):
        pre_abs = row.timing.offset + row.timing.preutterance
        if 0.0 < pre_abs < duration_ms:
            return float(pre_abs)
    return duration_ms * float(row_idx + 1) / float(row_count + 1)


def _synthetic_anchor_from_expected(
    row: OtoTemplateRow,
    expected_time: float,
    posterior: FramePosterior,
    role: str,
) -> OtoAnchor:
    span = estimate_vowel_span(posterior, expected_time)
    return OtoAnchor(
        anchor_abs_ms=float(expected_time),
        score=0.0,
        role=role,
        vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
        vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
        warnings=("synthetic_anchor:no_candidate", f"template_role:{role}"),
    )


def _posterior_anchor_candidates(posterior: FramePosterior, *, min_score: float) -> list[OtoAnchor]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return []
    out: list[OtoAnchor] = []
    for label in ("cv_boundary", "phone_change", "vowel_nucleus"):
        values = np.asarray(posterior.event_scores.get(label, []), dtype=np.float32)
        if values.shape[0] != times.shape[0]:
            continue
        peak_indices = _local_peak_indices(values, min_score=min_score)
        if not peak_indices and values.size:
            peak_indices = [int(np.argmax(values))]
        for idx in peak_indices:
            span = estimate_vowel_span(posterior, float(times[idx]))
            nucleus_time, nucleus_conf, nucleus_warnings = estimate_vowel_nucleus(
                posterior,
                float(times[idx]),
                vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
                vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
            )
            boundary_conf = _anchor_boundary_confidence(posterior, float(times[idx]), role=label)
            anchor_warnings = list(nucleus_warnings)
            if boundary_conf < 0.30:
                anchor_warnings.append(f"low_boundary_confidence:{boundary_conf:.3f}")
            out.append(
                OtoAnchor(
                    anchor_abs_ms=float(times[idx]),
                    score=float(values[idx]),
                    role=label,
                    frame_index=int(idx),
                    vowel_nucleus_abs_ms=(float(times[idx]) if label == "vowel_nucleus" else nucleus_time),
                    vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
                    vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
                    boundary_confidence=boundary_conf,
                    nucleus_confidence=(max(float(values[idx]), nucleus_conf) if label == "vowel_nucleus" else nucleus_conf),
                    warnings=tuple(anchor_warnings),
                )
            )
    return out


def _local_peak_indices(values: np.ndarray, *, min_score: float) -> list[int]:
    peaks: list[int] = []
    for idx, value in enumerate(values):
        if float(value) < min_score:
            continue
        left = values[idx - 1] if idx > 0 else -1.0
        right = values[idx + 1] if idx + 1 < values.shape[0] else -1.0
        if value >= left and value >= right:
            peaks.append(idx)
    return peaks


def _window_idx(times: np.ndarray, time_ms: float) -> int:
    idx = int(np.searchsorted(times, float(time_ms), side="left"))
    return max(0, min(idx, times.shape[0] - 1))


def _normalized_track(
    posterior: FramePosterior,
    key: str,
    times: np.ndarray,
    *,
    source: str = "acoustic",
) -> np.ndarray:
    if source == "class":
        values = np.asarray(posterior.class_probs.get(key, []), dtype=np.float32)
    else:
        values = np.asarray(posterior.acoustic_scores.get(key, []), dtype=np.float32)
    if values.shape[0] != times.shape[0]:
        return np.zeros_like(times)
    arr = values.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _left_context_max(values: np.ndarray, *, frames: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for idx in range(values.shape[0]):
        start = max(0, idx - frames)
        out[idx] = float(np.max(values[start : idx + 1])) if idx + 1 > start else float(values[idx])
    return out


def _event_to_dict(event: DecodedEvent | Mapping[str, object]) -> dict:
    if isinstance(event, DecodedEvent):
        return {
            "label": event.label,
            "time_ms": event.selected_time_ms,
            "score": event.score,
            "frame_index": event.frame_index,
            "expected_phone": event.expected_phone,
            "expected_phone_index": None,
            "slot_index": None,
        }
    return {
        "label": str(event.get("label")),
        "time_ms": float(event.get("selected_time_ms", event.get("time_ms", 0.0))),
        "score": float(event.get("score", 0.0)),
        "frame_index": event.get("frame_index"),
        "expected_phone": event.get("expected_phone"),
        "expected_phone_index": event.get("expected_phone_index", event.get("phone_index")),
        "slot_index": event.get("slot_index"),
    }


def _int_or_none(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))
