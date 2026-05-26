from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.timing.timing_anchor_profiles import get_anchor_profile
from core.timing.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock

from .kana import parse_kana_text
from .slot_viterbi import ExpectedSlot
from .types import DecodedEvent, FramePosterior, is_vowel_phone


TIMING_CLAMP_LARGE_DELTA_MS = 12.0
ALIAS_ATTACHED_PITCH_SUFFIX_RE = re.compile(r"[A-Ga-g](?:#|b)?[0-8]$")
JAPANESE_VCV_CUTOFF_TRIM_CAP_MS = 650.0
KOREAN_VCV_CUTOFF_TRIM_CAP_MS = 1100.0
JAPANESE_VCV_BOOTSTRAP_PROFILE = {
    "cv_head": (82.0, 53.0, 78.0),
    "vcv": (150.0, 100.0, 100.0),
    "vv": (150.0, 100.0, 100.0),
}
KOREAN_VCV_BOOTSTRAP_PROFILE = {
    "cv_head": (76.0, 55.0, 104.0),
    "vcv": (300.0, 220.0, 150.0),
    "vv": (360.0, 260.0, 145.0),
}
CVVC_CV_PRE_TARGET_MS = 150.0
CVVC_CV_PRE_MAX_MS = 180.0
CVVC_VC_CONTEXT_PRE_MAX_MS = 170.0
JAPANESE_CVVC_CV_HEAD_CONSONANT_HSMM_ANCHOR_LEAD_MS = 90.0
JAPANESE_CVVC_CV_HEAD_CONSONANT_CUTOFF_MAX_END_MS = 220.0
JAPANESE_CVVC_CV_HEAD_CONSONANT_VOWEL_START_TAIL_MS = 55.0
JAPANESE_CVVC_CV_HEAD_VOWEL_CUTOFF_TRIM_CAP_MS = 560.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_MIN_FORWARD_MS = 90.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_MAX_SHIFT_MS = 600.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_PRE_MS = 35.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_OVERLAP_MS = 10.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_CONSONANT_MS = 65.0
JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_CUTOFF_MS = 95.0
JAPANESE_CVVC_VC_SEQUENCE_RIGHT_TOKENS = frozenset({"w", "ky", "gy"})
JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS = frozenset({"ry"})
JAPANESE_CVVC_VC_SEQUENCE_PRE_MS = 110.0
JAPANESE_CVVC_VC_SEQUENCE_OVERLAP_MS = 100.0
JAPANESE_CVVC_VC_SEQUENCE_CONSONANT_MS = 120.0
JAPANESE_CVVC_VC_SEQUENCE_CUTOFF_MS = 146.0
JAPANESE_CVVC_VC_SEQUENCE_REPAIR_MIN_LATE_MS = 80.0
JAPANESE_CVVC_VC_SEQUENCE_LONG_TAIL_MS = 220.0
JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_MIN_GAP_MS = 420.0
JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_TARGET_BIAS_MS = 20.0
JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_MIN_FORWARD_MS = 20.0
JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_MIN_GAP_MS = 600.0
JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_NUCLEUS_NEAR_MS = 60.0
JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_TARGET_RATIO = 0.35
JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_TARGET_BIAS_MS = -20.0
JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_MIN_FORWARD_MS = 40.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_START_MS = 480.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_LATE_MS = 300.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_EARLY_MS = 220.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_PRE_MS = 80.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_OVERLAP_MS = 40.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_CONSONANT_MS = 220.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_TAIL_AFTER_START_MS = 240.0
JAPANESE_CVVC_FOLLOWING_GLIDE_PHONES = ("w", "i")
JAPANESE_CVVC_FOLLOWING_GLIDE_OFFSET_AFTER_PRE_MS = 65.0
JAPANESE_CVVC_FOLLOWING_GLIDE_PRE_MS = 110.0
JAPANESE_CVVC_FOLLOWING_GLIDE_OVERLAP_MS = 100.0
JAPANESE_CVVC_FOLLOWING_GLIDE_CONSONANT_MS = 210.0
JAPANESE_CVVC_FOLLOWING_YOON_RIGHT_TOKENS = frozenset({"ky", "gy"})
JAPANESE_CVVC_FOLLOWING_YOON_PRE_MS = 70.0
JAPANESE_CVVC_FOLLOWING_YOON_OVERLAP_MS = 60.0
JAPANESE_CVVC_FOLLOWING_YOON_CONSONANT_MS = 170.0
JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_PRE_MS = 50.0
JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_OVERLAP_MS = 40.0
JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_CONSONANT_MS = 150.0
JAPANESE_CVVC_POST_YOON_VC_OFFSET_FROM_CV_MS = 325.0
JAPANESE_CVVC_POST_E_YOON_VC_OFFSET_FROM_CV_MS = 275.0
JAPANESE_CVVC_POST_YOON_VC_REPAIR_MIN_LATE_MS = 40.0
JAPANESE_CVVC_PURE_VOWEL_VV_LEFT_KEEP_MS = 85.0
JAPANESE_CVVC_PURE_VOWEL_VV_PRE_MS = 110.0
JAPANESE_CVVC_PURE_VOWEL_VV_OVERLAP_MS = 100.0
JAPANESE_CVVC_PURE_VOWEL_VV_CONSONANT_MS = 210.0
JAPANESE_CVVC_PURE_VOWEL_V_OFFSET_AFTER_VV_MS = 110.0
JAPANESE_CVVC_PURE_VOWEL_V_PRE_MS = 90.0
JAPANESE_CVVC_PURE_VOWEL_V_OVERLAP_MS = 45.0
JAPANESE_CVVC_PURE_VOWEL_V_CONSONANT_MS = 100.0
JAPANESE_CVVC_PURE_VOWEL_REPAIR_MIN_LATE_MS = 80.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_STEP_MS = 500.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_TERMINAL_TAIL_MS = 430.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_HEAD_LEAD_MS = 120.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_HEAD_CUTOFF_MIN_MS = 360.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_PRE_MS = 300.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_OVERLAP_MS = 100.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_CONSONANT_MS = 420.0
JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_CUTOFF_MS = 600.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS = 480.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_SPAN_MS = 500.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_FIRST_OFFSET_MS = 250.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_HALF_STEP_MS = 250.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_TRANSITION_STEP_MS = 500.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_TRAILING_R_TAIL_MS = 1200.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_HEAD_CUTOFF_MIN_MS = 430.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_HEAD_VOWEL_RELEASE_CUTOFF_MIN_MS = 840.0
JAPANESE_CVVC_PURE_VOWEL_HEAD_CUTOFF_REPAIR_MAX_CURRENT_MS = 240.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_PRE_MS = 250.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_OVERLAP_MS = 83.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CONSONANT_MS = 420.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CUTOFF_MS = 600.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_V_PRE_MS = 0.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_V_OVERLAP_MS = 0.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_V_CONSONANT_MS = 125.0
JAPANESE_CVVC_PURE_VOWEL_ONSET_V_CUTOFF_MS = 360.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_FIRST_OFFSET_MS = 250.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_STEP_MS = 1000.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_MIN_EARLY_MS = 180.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_NEXT_VOWEL_LEAD_MS = 150.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_PRE_MS = 250.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_OVERLAP_MS = 83.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_CONSONANT_MS = 300.0
JAPANESE_CVVC_HEADLESS_VC_ONSET_CUTOFF_MS = 330.0
JAPANESE_CVVC_REGULAR_PAIR_ONSET_CENTER_AFTER_FIRST_NEXT_VOWEL_MS = 200.0
JAPANESE_CVVC_REGULAR_PAIR_ONSET_STEP_MS = 500.0
JAPANESE_CVVC_REGULAR_PAIR_ONSET_VC_CENTER_LEAD_MS = 170.0
JAPANESE_CVVC_REGULAR_PAIR_ONSET_CV_CENTER_LAG_MS = 170.0
JAPANESE_CVVC_REGULAR_PAIR_ONSET_MIN_FORWARD_MS = 140.0
JAPANESE_CVVC_GROUPED_VC_FIRST_OFFSET_FROM_CV_MS = 210.0
JAPANESE_CVVC_GROUPED_VC_FIRST_MIN_GAP_TO_CV_MS = 420.0
JAPANESE_CVVC_GROUPED_VC_FIRST_MIN_FORWARD_MS = 100.0
JAPANESE_CVVC_VC_CUTOFF_ORDER_TRIGGER_TAIL_MS = 5.0
JAPANESE_CVVC_VC_CUTOFF_ORDER_MIN_TAIL_MS = 45.0
JAPANESE_CVVC_INITIAL_V_REGULAR_PAIR_V_LEAD_MS = 120.0
JAPANESE_CVVC_INITIAL_V_REGULAR_PAIR_MIN_V_FORWARD_MS = 500.0
JAPANESE_CVVC_TERMINAL_RELEASE_R_TAIL_MS = 1120.0
JAPANESE_CVVC_TERMINAL_RELEASE_R_MIN_FORWARD_MS = 250.0
JAPANESE_CVVC_STANDALONE_RELEASE_R_OFFSET_RATIO = 0.5
JAPANESE_CVVC_STANDALONE_RELEASE_R_MIN_FORWARD_MS = 250.0
JAPANESE_CVVC_TERMINAL_STANDALONE_V_OFFSET_AFTER_TERMINAL_MS = 100.0
JAPANESE_CVVC_TERMINAL_STANDALONE_V_PRE_MS = 30.0
JAPANESE_CVVC_TERMINAL_STANDALONE_V_OVERLAP_MS = 20.0
JAPANESE_CVVC_TERMINAL_STANDALONE_V_CONSONANT_MS = 125.0
JAPANESE_CVVC_TERMINAL_STANDALONE_V_CUTOFF_MS = 260.0
JAPANESE_CVVC_TERMINAL_LEFT_KEEP_MS = 430.0
JAPANESE_CVVC_TERMINAL_END_TAIL_MS = 50.0
JAPANESE_CVVC_TERMINAL_PRE_MS = 200.0
JAPANESE_CVVC_TERMINAL_OVERLAP_MS = 100.0
JAPANESE_CVVC_TERMINAL_CONSONANT_MS = 320.0
JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_CUTOFF_REPAIR_MIN_LONG_MS = 40.0
JAPANESE_CVVC_INITIAL_SPACED_CV_OFFSET_REPAIR_MIN_EARLY_MS = 8.0
JAPANESE_CVVC_INITIAL_SPACED_CV_OFFSET_REPAIR_EDGE_MS = 25.0
JAPANESE_CVVC_FOLLOWING_CV_BLOCK_CUTOFF_CAP_MS = 400.0
JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_PAD_MS = 80.0
JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS = 120.0
JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_PAD_MS = 120.0
JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS = 200.0
JAPANESE_CVVC_CV_SEQUENCE_NEXT_CV_CUTOFF_PAD_MS = 80.0
JAPANESE_CVVC_CV_SEQUENCE_NEXT_CV_CUTOFF_MIN_EXCESS_MS = 120.0
JAPANESE_CVVC_FOLLOWING_REPAIR_RULES = frozenset(
    {"cvvc_following_glide_repair", "cvvc_following_yoon_repair"}
)
ALIAS_TARGET_SHARE_BONUS = 8.0
JAPANESE_VCV_HSMM_ANCHOR_LEAD_MS = {
    "cv_head": 35.0,
    "vcv": 205.0,
    "vv": 260.0,
}
KOREAN_VCV_HSMM_ANCHOR_LEAD_MS = {
    "cv_head": 180.0,
    "vcv": 240.0,
    "vv": 200.0,
}
JAPANESE_VCV_HSMM_ANCHOR_LEAD_BY_EVENT_MS = {
    ("cv_head", "vv_boundary"): 8.0,
    ("vcv", "cv_boundary"): 125.0,
    ("vcv", "vowel_nucleus"): 300.0,
    ("vv", "vv_boundary"): 315.0,
    ("vv", "vowel_nucleus"): 230.0,
}
KOREAN_VCV_HSMM_ANCHOR_LEAD_BY_EVENT_MS = {
    ("cv_head", "vowel_nucleus"): 220.0,
    ("vcv", "cv_boundary"): 195.0,
    ("vcv", "vowel_nucleus"): 264.0,
    ("vv", "vv_boundary"): 277.0,
}
JAPANESE_CVVC_HSMM_ANCHOR_LEAD_MS = {
    "cv": 50.0,
    "vc": 45.0,
    "vcv": 100.0,
    "vv": 40.0,
}


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
    source_row_index: int = -1


@dataclass(frozen=True)
class OtoAnchor:
    anchor_abs_ms: float
    score: float = 1.0
    role: str = "cv_boundary"
    frame_index: int | None = None
    refined_from_abs_ms: float | None = None
    local_refine_delta_ms: float | None = None
    local_refine_margin: float | None = None
    local_refine_window_ms: float | None = None
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
    source_event_label: str | None = None
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
                "refined_from_abs_ms": self.anchor.refined_from_abs_ms,
                "local_refine_delta_ms": self.anchor.local_refine_delta_ms,
                "local_refine_margin": self.anchor.local_refine_margin,
                "local_refine_window_ms": self.anchor.local_refine_window_ms,
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
                "source_event_label": self.anchor.source_event_label,
                "warnings": list(self.anchor.warnings),
            }
        return payload


def load_oto_template_rows(path: str | Path) -> list[OtoTemplateRow]:
    rows: list[OtoTemplateRow] = []
    for line_index, line in enumerate(read_text_with_fallback(str(path)).splitlines()):
        row = parse_template_oto_line(line, source_row_index=line_index)
        if row is not None:
            rows.append(row)
    return rows


def load_oto_template_rows_alias_only(path: str | Path) -> list[OtoTemplateRow]:
    rows = load_oto_template_rows(path)
    return [
        row
        if _is_nonphonetic_special_alias(row.alias)
        else replace(
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


def parse_template_oto_line(line: str, *, source_row_index: int = -1) -> OtoTemplateRow | None:
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
        source_row_index=int(source_row_index),
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

    alias_type = _adaptation_alias_type(row.alias, cfg.alias_type, anchor)
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
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=file_duration_ms)
    warnings.extend(timing_warnings)
    return AdaptedOtoRow(
        wav=row.wav,
        alias=row.alias,
        timing=timing,
        source_timing=row.timing,
        anchor=anchor,
        mode="template-preserve",
        warnings=tuple(dict.fromkeys(warnings)),
        applied_rules=tuple(result.applied_rules),
    )


def _adaptation_alias_type(alias: str, default_alias_type: str, anchor: OtoAnchor | None) -> str:
    role = str(getattr(anchor, "role", "") or "").strip().lower()
    if role in {"cv", "cv_head", "vcv", "vc", "vv", "v"}:
        return role
    return _alias_type_for_row(alias, default_alias_type)


def bootstrap_row(
    wav: str,
    alias: str,
    anchor: OtoAnchor | None,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig | None = None,
) -> AdaptedOtoRow:
    cfg = config or OtoAdapterConfig(mode="bootstrap")
    alias_type = _adaptation_alias_type(alias, cfg.alias_type, anchor)
    pre_target_ms, ovl_gap_ms, cons_gap_ms = _bootstrap_role_profile(cfg, alias_type)
    if anchor is None:
        timing, timing_warnings = _validate_timing_with_warnings(
            OtoTiming(0.0, pre_target_ms + cons_gap_ms, -max(180.0, file_duration_ms), pre_target_ms, max(0.0, pre_target_ms - ovl_gap_ms)),
            file_duration_ms=file_duration_ms,
        )
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=timing,
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            warnings=tuple(dict.fromkeys(("missing_anchor", *timing_warnings))),
        )

    if _is_terminal_silence_alias(alias):
        return _bootstrap_terminal_silence_row(
            wav,
            alias,
            anchor,
            file_duration_ms=file_duration_ms,
            config=cfg,
        )

    raw_anchor_abs = _clamp(anchor.anchor_abs_ms, 0.0, file_duration_ms)
    anchor_abs, anchor_adjust_warnings = _bootstrap_anchor_abs_ms(
        cfg,
        alias_type,
        anchor,
        alias=alias,
        file_duration_ms=file_duration_ms,
    )
    anchor_lead_ms = max(0.0, raw_anchor_abs - anchor_abs)
    anchor_warnings = list(anchor.warnings)
    anchor_warnings.extend(anchor_adjust_warnings)
    vc_pre_cap_ms = max(12.0, float(cfg.vc_pre_max_ms))
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
        cutoff_trim, cutoff_warnings = _bootstrap_cutoff_trim_ms(
            cfg,
            "v",
            max(consonant + 8.0, (vowel_end - offset) + cfg.tail_margin_ms),
            min_trim_ms=consonant + 8.0,
            file_duration_ms=file_duration_ms,
        )
        cutoff = -cutoff_trim
        timing, timing_warnings = _validate_timing_with_warnings(
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
            warnings=tuple(dict.fromkeys((*anchor.warnings, *cutoff_warnings, *timing_warnings))),
        )
    elif alias_type == "vcv":
        prev_end = anchor.previous_vowel_end_abs_ms
        if prev_end is None:
            prev_end = max(0.0, anchor_abs - cfg.pre_target_ms)
        max_prev_end = max(0.0, anchor_abs - max(24.0, cfg.pre_target_ms))
        if float(prev_end) > max_prev_end:
            anchor_warnings.append(f"vcv_previous_vowel_end_clamped:{float(prev_end):.1f}->{max_prev_end:.1f}")
            prev_end = max_prev_end
        offset = max(float(prev_end) - cfg.previous_tail_keep_ms, 0.0)
    elif alias_type == "vc":
        vc_pre_target_ms, vc_context_warnings = _cvvc_vc_context_pre_ms(cfg, anchor, anchor_abs)
        anchor_warnings.extend(vc_context_warnings)
        vc_pre_cap_ms = max(vc_pre_cap_ms, vc_pre_target_ms)
        offset = max(anchor_abs - vc_pre_target_ms, 0.0)
    elif alias_type == "cv":
        cv_pre_target_ms, cv_context_warnings = _cvvc_cv_context_pre_ms(cfg, anchor, anchor_abs)
        anchor_warnings.extend(cv_context_warnings)
        offset = max(anchor_abs - cv_pre_target_ms, 0.0)
    else:
        offset = max(anchor_abs - cfg.pre_target_ms, 0.0)
    pre = max(anchor_abs - offset, 0.0)
    if alias_type == "vc":
        pre = min(pre, vc_pre_cap_ms)
        offset = max(anchor_abs - pre, 0.0)
    elif alias_type in {"cv_head", "vcv", "vv"}:
        target_pre = min(float(pre_target_ms), anchor_abs)
        offset = max(anchor_abs - target_pre, 0.0)
        pre = max(anchor_abs - offset, 0.0)
    overlap = max(0.0, pre - ovl_gap_ms)
    consonant = pre + cons_gap_ms
    cutoff_warnings: tuple[str, ...] = ()
    if alias_type == "vc":
        cut_gap = _role_cut_gap_ms(cfg, alias_type)
        cutoff = -(max(consonant + 8.0, consonant + cut_gap))
    else:
        vowel_end = anchor.vowel_end_abs_ms if anchor.vowel_end_abs_ms is not None else min(file_duration_ms, anchor_abs + 180.0)
        raw_cutoff_trim = max(consonant + 8.0, float(vowel_end) - offset + cfg.tail_margin_ms - anchor_lead_ms)
        raw_cutoff_trim, cvvc_head_cutoff_warnings = _cvvc_cv_head_cutoff_trim_ms(
            cfg,
            alias_type,
            alias,
            anchor,
            raw_cutoff_trim,
            min_trim_ms=consonant + 8.0,
            offset_ms=offset,
            file_duration_ms=file_duration_ms,
        )
        cutoff_trim, cutoff_warnings = _bootstrap_cutoff_trim_ms(
            cfg,
            alias_type,
            raw_cutoff_trim,
            min_trim_ms=consonant + 8.0,
            file_duration_ms=file_duration_ms,
        )
        cutoff_warnings = (*cvvc_head_cutoff_warnings, *cutoff_warnings)
        cutoff = -cutoff_trim
    timing, timing_warnings = _validate_timing_with_warnings(
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
        warnings=tuple(dict.fromkeys((*anchor_warnings, *cutoff_warnings, *timing_warnings))),
    )


def anchors_from_prediction(
    posterior: FramePosterior,
    decoded_events: Sequence[DecodedEvent | Mapping[str, object]],
) -> list[OtoAnchor]:
    anchors: list[OtoAnchor] = []
    previous_vowel_end: float | None = None
    decoded = [_event_to_dict(event) for event in decoded_events]
    for idx, event in enumerate(decoded):
        if event["label"] not in {"cv_boundary", "phone_change", "vowel_nucleus", "vv_boundary"}:
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
        event_source = str(event.get("source") or "").strip()
        if event_source:
            anchor_warnings.append(f"event_source:{event_source}")
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
            source_event_label=role,
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
    last_time = -float("inf")
    last_target_phone_index: int | None = None
    alias_targets = _assign_alias_target_indices(template_rows, expected_phones or ())
    decoded_count = sum(1 for item in candidates if "slot_decoded_event" in item.warnings)
    for row_idx, row in enumerate(template_rows):
        base_role = _alias_type_for_row(row.alias, "auto")
        target_phone_index = alias_targets[row_idx] if row_idx < len(alias_targets) else None
        role = _contextual_alias_role(row.alias, base_role, expected_phones or (), target_phone_index)
        preferred_labels = _preferred_event_labels_for_alias(row.alias, role)
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
    return _refine_anchor_sequence_locally(posterior, out)


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
        if _is_terminal_silence_alias(row.alias):
            continue
        base_role = _alias_type_for_row(row.alias, "auto")
        role = _contextual_alias_role(row.alias, base_role, expected, int(target_index))
        event_label = _slot_event_label_for_alias_role(role)
        if role == "vv" and target_index > 0 and not is_vowel_phone(expected[int(target_index) - 1]):
            event_label = "cv_boundary"
        if role == "cv_head" and (
            not is_vowel_phone(expected[target_index])
            or _cv_head_phones_are_consonant_like(_alias_phone_sequence(row.alias))
        ):
            event_label = "phone_change"
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


def _is_terminal_silence_alias(alias: str) -> bool:
    parts = [part.strip().lower() for part in str(alias or "").split() if part.strip()]
    return len(parts) >= 2 and parts[-1] == "-"


def _is_nonphonetic_special_alias(alias: str) -> bool:
    normalized = str(alias or "").strip().lower()
    if not normalized:
        return True
    compact = "".join(part for part in normalized.split())
    if compact.startswith("br") and compact[2:].isdigit():
        return True
    if any(char in normalized for char in {"・", "･"}):
        return True
    return any(char in normalized for char in {"息", "吸"})


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


def repair_cvvc_vc_row_sequence(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig | None = None,
) -> list[AdaptedOtoRow]:
    cfg = config or OtoAdapterConfig()
    if not _is_japanese_cvvc_config(cfg):
        return list(rows)
    out = list(rows)
    for index, row in enumerate(out):
        previous = out[index - 1] if index > 0 else None
        out[index] = _repair_cvvc_vc_row_from_previous(previous, row, cfg)
    return out


def repair_cvvc_row_sequence(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig | None = None,
    *,
    file_duration_ms: float | None = None,
) -> list[AdaptedOtoRow]:
    cfg = config or OtoAdapterConfig()
    out = repair_cvvc_vc_row_sequence(rows, cfg)
    if not _is_japanese_cvvc_config(cfg):
        return out
    out = _repair_cvvc_initial_spaced_cv_rows(out, cfg)
    out = _repair_cvvc_initial_vowel_cv_head_rows(out, cfg)
    out = _repair_cvvc_initial_consonant_cv_head_rows(out, cfg)
    out = _repair_cvvc_pure_vowel_sequence_rows(out, cfg)
    out = _repair_cvvc_pure_vowel_onset_sequence_rows(out, cfg, file_duration_ms=file_duration_ms)
    out = _repair_cvvc_headless_vc_onset_sequence_rows(out, cfg)
    out = _repair_cvvc_terminal_standalone_vowel_rows(out, cfg)
    out = _repair_cvvc_pure_vowel_transition_rows(out, cfg)
    out = _repair_cvvc_sonorant_yoon_vc_rows(out, cfg)
    out = _repair_cvvc_following_glide_rows(out, cfg)
    out = _repair_cvvc_post_yoon_vc_rows(out, cfg)
    out = _repair_cvvc_following_glide_rows(out, cfg)
    out = _repair_cvvc_post_yoon_vc_rows(out, cfg)
    out = _repair_cvvc_following_glide_rows(out, cfg)
    out = _repair_cvvc_initial_v_regular_pair_onset_rows(out, cfg)
    out = _repair_cvvc_regular_pair_onset_sequence_rows(out, cfg)
    out = _repair_cvvc_grouped_vc_first_offset_rows(out, cfg)
    out = _repair_cvvc_cv_rejected_next_vowel_rows(out, cfg)
    out = _repair_cvvc_vc_next_vowel_local_rows(out, cfg)
    out = _repair_cvvc_terminal_release_r_rows(out, cfg, file_duration_ms=file_duration_ms)
    out = _repair_cvvc_following_cv_block_cutoff_rows(out, cfg)
    out = _repair_cvvc_cv_next_transition_cutoff_rows(out, cfg)
    out = _repair_cvvc_cv_sequence_next_cv_cutoff_rows(out, cfg)
    return _repair_cvvc_vc_cutoff_order_rows(out, cfg)


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
    if _is_nonphonetic_special_alias(row.alias):
        if not _is_degenerate_timing(row.timing):
            return AdaptedOtoRow(
                wav=row.wav,
                alias=row.alias,
                timing=row.timing,
                source_timing=row.timing,
                anchor=active_anchor,
                mode="template-special",
                warnings=("special_alias_source_timing_preserved",),
                applied_rules=("special_alias_source_timing_preserve",),
            )
        return AdaptedOtoRow(
            wav=row.wav,
            alias=row.alias,
            timing=row.timing,
            source_timing=None,
            anchor=active_anchor,
            mode="template-special",
            warnings=tuple(dict.fromkeys((*warnings, "special_alias_zero_template_preserved"))),
        )
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


def _bootstrap_role_profile(config: OtoAdapterConfig, alias_type: str) -> tuple[float, float, float]:
    role = str(alias_type or "").strip().lower()
    fmt = str(config.format_type or "").strip().lower()
    language = str(config.language or "").strip().lower()
    if fmt == "vcv":
        if language == "japanese" and role in JAPANESE_VCV_BOOTSTRAP_PROFILE:
            return JAPANESE_VCV_BOOTSTRAP_PROFILE[role]
        if language == "korean" and role in KOREAN_VCV_BOOTSTRAP_PROFILE:
            return KOREAN_VCV_BOOTSTRAP_PROFILE[role]
    return (
        float(config.pre_target_ms),
        float(config.ovl_gap_ms),
        float(config.cons_gap_ms),
    )


def _bootstrap_terminal_silence_row(
    wav: str,
    alias: str,
    anchor: OtoAnchor,
    *,
    file_duration_ms: float,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if _is_japanese_cvvc_config(config):
        cvvc_terminal = _bootstrap_japanese_cvvc_terminal_silence_row(
            wav,
            alias,
            anchor,
            file_duration_ms=file_duration_ms,
        )
        if cvvc_terminal is not None:
            return cvvc_terminal
    terminal_anchor = max(0.0, float(file_duration_ms or anchor.anchor_abs_ms or 0.0))
    pre = min(max(30.0, min(float(config.pre_target_ms), 110.0)), terminal_anchor)
    overlap = max(0.0, pre - 10.0)
    tail = 8.0
    consonant = pre
    active_ms = consonant + tail
    timing = OtoTiming(
        offset=max(0.0, terminal_anchor - active_ms),
        consonant=consonant,
        cutoff=-active_ms,
        preutterance=pre,
        overlap=overlap,
    )
    warnings = tuple(dict.fromkeys((*anchor.warnings, "terminal_silence_end_anchored")))
    return AdaptedOtoRow(
        wav=wav,
        alias=alias,
        timing=timing,
        source_timing=None,
        anchor=anchor,
        mode="bootstrap",
        warnings=warnings,
    )


def _bootstrap_japanese_cvvc_terminal_silence_row(
    wav: str,
    alias: str,
    anchor: OtoAnchor,
    *,
    file_duration_ms: float,
) -> AdaptedOtoRow | None:
    vowel_end = anchor.vowel_end_abs_ms
    if vowel_end is None:
        return None
    terminal_end = float(vowel_end)
    if not np.isfinite(terminal_end) or terminal_end <= 0.0:
        return None
    if float(file_duration_ms) > 0.0:
        terminal_end = _clamp(terminal_end, 0.0, float(file_duration_ms))
    target_end = terminal_end + float(JAPANESE_CVVC_TERMINAL_END_TAIL_MS)
    if float(file_duration_ms) > 0.0:
        target_end = min(target_end, float(file_duration_ms))
    offset = max(0.0, terminal_end - float(JAPANESE_CVVC_TERMINAL_LEFT_KEEP_MS))
    cutoff_abs = max(float(JAPANESE_CVVC_TERMINAL_CONSONANT_MS) + 8.0, target_end - offset)
    timing = OtoTiming(
        offset=offset,
        consonant=float(JAPANESE_CVVC_TERMINAL_CONSONANT_MS),
        cutoff=-cutoff_abs,
        preutterance=float(JAPANESE_CVVC_TERMINAL_PRE_MS),
        overlap=min(float(JAPANESE_CVVC_TERMINAL_OVERLAP_MS), float(JAPANESE_CVVC_TERMINAL_PRE_MS)),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=file_duration_ms)
    warnings = tuple(
        dict.fromkeys(
            (
                *anchor.warnings,
                f"terminal_silence_vowel_end_anchored:{float(vowel_end):.1f}",
                *timing_warnings,
            )
        )
    )
    return AdaptedOtoRow(
        wav=wav,
        alias=alias,
        timing=timing,
        source_timing=None,
        anchor=anchor,
        mode="bootstrap",
        warnings=warnings,
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )


def _bootstrap_anchor_abs_ms(
    config: OtoAdapterConfig,
    alias_type: str,
    anchor: OtoAnchor,
    *,
    alias: str = "",
    file_duration_ms: float,
) -> tuple[float, tuple[str, ...]]:
    raw = _clamp(anchor.anchor_abs_ms, 0.0, file_duration_ms)
    lead = _bootstrap_hsmm_anchor_lead_ms(config, alias_type, anchor, alias=alias)
    if lead <= 0.0:
        return raw, ()
    shifted = _clamp(raw - lead, 0.0, file_duration_ms)
    if abs(shifted - raw) <= 1e-6:
        return raw, ()
    return shifted, (f"hsmm_anchor_lead:{raw:.1f}->{shifted:.1f}",)


def _bootstrap_hsmm_anchor_lead_ms(
    config: OtoAdapterConfig,
    alias_type: str,
    anchor: OtoAnchor,
    *,
    alias: str = "",
) -> float:
    if _is_terminal_silence_alias(alias):
        return 0.0
    fmt = str(config.format_type or "").strip().lower()
    if "event_source:filename_hsmm" not in set(anchor.warnings):
        return 0.0
    role = str(alias_type or "").strip().lower()
    source_event_label = str(anchor.source_event_label or "").strip().lower()
    language = str(config.language or "").strip().lower()
    if fmt == "cvvc":
        if language == "japanese":
            if role == "cv_head" and _cv_head_phones_are_consonant_like(_alias_phone_sequence(alias)):
                return float(JAPANESE_CVVC_CV_HEAD_CONSONANT_HSMM_ANCHOR_LEAD_MS)
            return float(JAPANESE_CVVC_HSMM_ANCHOR_LEAD_MS.get(role, 0.0))
        return 0.0
    if fmt != "vcv":
        return 0.0
    if language == "japanese":
        if source_event_label:
            keyed = JAPANESE_VCV_HSMM_ANCHOR_LEAD_BY_EVENT_MS.get((role, source_event_label))
            if keyed is not None:
                return float(keyed)
        return float(JAPANESE_VCV_HSMM_ANCHOR_LEAD_MS.get(role, 0.0))
    if language == "korean":
        if source_event_label:
            keyed = KOREAN_VCV_HSMM_ANCHOR_LEAD_BY_EVENT_MS.get((role, source_event_label))
            if keyed is not None:
                return float(keyed)
        return float(KOREAN_VCV_HSMM_ANCHOR_LEAD_MS.get(role, 0.0))
    return 0.0


def _cvvc_cv_context_pre_ms(
    config: OtoAdapterConfig,
    anchor: OtoAnchor,
    anchor_abs_ms: float,
) -> tuple[float, tuple[str, ...]]:
    base = max(12.0, min(float(config.pre_target_ms), float(anchor_abs_ms)))
    if not _is_cvvc_format(config):
        return base, ()
    target = max(base, min(float(CVVC_CV_PRE_TARGET_MS), float(CVVC_CV_PRE_MAX_MS)))
    target = min(target, max(0.0, float(anchor_abs_ms)))
    if target <= base + 1e-6:
        return base, ()
    return target, (f"cvvc_cv_left_context_pre:{base:.1f}->{target:.1f}",)


def _cvvc_vc_context_pre_ms(
    config: OtoAdapterConfig,
    anchor: OtoAnchor,
    anchor_abs_ms: float,
) -> tuple[float, tuple[str, ...]]:
    base_cap = max(12.0, float(config.vc_pre_max_ms))
    base = min(max(12.0, float(config.pre_target_ms) * 0.75), base_cap, max(0.0, float(anchor_abs_ms)))
    if not _is_cvvc_format(config):
        return base, ()
    previous_end = anchor.previous_vowel_end_abs_ms
    if previous_end is None:
        return base, ()
    previous_end = float(previous_end)
    if previous_end <= 0.0 or previous_end >= float(anchor_abs_ms) - 8.0:
        return base, (f"cvvc_vc_previous_vowel_end_ignored:{previous_end:.1f}",)
    desired_offset = max(0.0, previous_end - max(24.0, float(config.previous_tail_keep_ms)))
    desired_pre = max(0.0, float(anchor_abs_ms) - desired_offset)
    capped = min(desired_pre, float(CVVC_VC_CONTEXT_PRE_MAX_MS), max(0.0, float(anchor_abs_ms)))
    if capped <= base + 1e-6:
        return base, ()
    if desired_pre > capped + 1e-6:
        return capped, (f"cvvc_vc_left_context_pre_capped:{desired_pre:.1f}->{capped:.1f}",)
    return capped, (f"cvvc_vc_left_context_pre:{base:.1f}->{capped:.1f}",)


def _cvvc_cv_head_cutoff_trim_ms(
    config: OtoAdapterConfig,
    alias_type: str,
    alias: str,
    anchor: OtoAnchor,
    raw_trim_ms: float,
    *,
    min_trim_ms: float,
    offset_ms: float,
    file_duration_ms: float,
) -> tuple[float, tuple[str, ...]]:
    if not _is_cvvc_format(config):
        return float(raw_trim_ms), ()
    if str(config.language or "").strip().lower() != "japanese":
        return float(raw_trim_ms), ()
    if str(alias_type or "").strip().lower() != "cv_head":
        return float(raw_trim_ms), ()
    phones = _alias_phone_sequence(alias)
    if _cv_head_phones_are_consonant_like(phones):
        cap_end = _cvvc_cv_head_consonant_cutoff_end_cap_ms(anchor, file_duration_ms=file_duration_ms)
        cap_trim = max(0.0, cap_end - max(0.0, float(offset_ms)))
        clamped = max(float(min_trim_ms), min(float(raw_trim_ms), cap_trim))
        if clamped < float(raw_trim_ms) - 1e-6:
            return clamped, (f"cvvc_initial_cv_cutoff_clamped:{float(raw_trim_ms):.1f}->{clamped:.1f}",)
        return float(raw_trim_ms), ()
    cap_trim = max(float(min_trim_ms), float(JAPANESE_CVVC_CV_HEAD_VOWEL_CUTOFF_TRIM_CAP_MS))
    if float(file_duration_ms) > 0.0:
        cap_trim = min(cap_trim, max(float(min_trim_ms), float(file_duration_ms) * 0.35))
    if float(raw_trim_ms) <= cap_trim + 1e-6:
        return float(raw_trim_ms), ()
    return cap_trim, (f"cvvc_initial_vowel_cutoff_clamped:{float(raw_trim_ms):.1f}->{cap_trim:.1f}",)


def _cvvc_cv_head_consonant_cutoff_end_cap_ms(anchor: OtoAnchor, *, file_duration_ms: float) -> float:
    candidates = [float(JAPANESE_CVVC_CV_HEAD_CONSONANT_CUTOFF_MAX_END_MS)]
    vowel_start = anchor.vowel_start_abs_ms
    if vowel_start is not None:
        candidates.append(max(0.0, float(vowel_start)) + float(JAPANESE_CVVC_CV_HEAD_CONSONANT_VOWEL_START_TAIL_MS))
    cap = min(candidates)
    if file_duration_ms > 0.0:
        cap = min(cap, max(0.0, float(file_duration_ms)))
    return max(0.0, cap)


def _repair_cvvc_vc_row_from_previous(
    previous: AdaptedOtoRow | None,
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if previous is None:
        return row
    if str(previous.wav or "").strip().lower() != str(row.wav or "").strip().lower():
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "vc":
        return row
    right_token = _cvvc_vc_alias_right_token(row.alias)
    if right_token not in JAPANESE_CVVC_VC_SEQUENCE_RIGHT_TOKENS:
        return row
    if not any(warning.startswith("cvvc_vc_left_context_pre_capped:") for warning in row.warnings):
        return row

    previous_absolute = absolute_oto_positions(previous.timing)
    previous_pre = float(previous_absolute["preutterance_abs"])
    previous_cutoff = float(previous_absolute["cutoff_abs"])
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (previous_pre, previous_cutoff, current_offset)):
        return row

    if right_token == "w":
        candidate_offset = previous_pre - float(config.previous_tail_keep_ms)
    else:
        previous_tail = max(0.0, previous_cutoff - previous_pre)
        reference = (
            previous_pre
            if previous_tail > float(JAPANESE_CVVC_VC_SEQUENCE_LONG_TAIL_MS)
            else previous_cutoff
        )
        candidate_offset = reference - float(config.previous_tail_keep_ms)
    candidate_offset = max(0.0, float(candidate_offset))
    if current_offset - candidate_offset <= float(JAPANESE_CVVC_VC_SEQUENCE_REPAIR_MIN_LATE_MS):
        return row

    timing = OtoTiming(
        offset=candidate_offset,
        consonant=float(JAPANESE_CVVC_VC_SEQUENCE_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_VC_SEQUENCE_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_VC_SEQUENCE_PRE_MS),
        overlap=float(JAPANESE_CVVC_VC_SEQUENCE_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_vc_sequence_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_vc_sequence_repair"))),
    )


def _repair_cvvc_sonorant_yoon_vc_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        previous = out[index - 1] if index > 0 else None
        out[index] = _repair_cvvc_sonorant_yoon_vc_row(previous, row, config)
    return out


def _repair_cvvc_sonorant_yoon_vc_row(
    previous: AdaptedOtoRow | None,
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if previous is None:
        return row
    if str(previous.wav or "").strip().lower() != str(row.wav or "").strip().lower():
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "vc":
        return row
    right_token = _cvvc_vc_alias_right_token(row.alias)
    if right_token not in JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS:
        return row
    if not any(warning.startswith("cvvc_vc_left_context_pre_capped:") for warning in row.warnings):
        return row

    reference_offset = float(previous.timing.offset)
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (reference_offset, current_offset)):
        return row
    if current_offset - reference_offset <= float(JAPANESE_CVVC_VC_SEQUENCE_REPAIR_MIN_LATE_MS):
        return row

    timing = OtoTiming(
        offset=max(0.0, reference_offset),
        consonant=float(JAPANESE_CVVC_VC_SEQUENCE_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_VC_SEQUENCE_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_VC_SEQUENCE_PRE_MS),
        overlap=float(JAPANESE_CVVC_VC_SEQUENCE_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_sonorant_yoon_vc_repaired:{current_offset:.1f}->{reference_offset:.1f}",
                    f"cvvc_sonorant_yoon_vc_reference:{previous.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_sonorant_yoon_vc_repair"))),
    )


def _repair_cvvc_cv_rejected_next_vowel_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    return [_repair_cvvc_cv_rejected_next_vowel_row(row, config) for row in rows]


def _repair_cvvc_cv_rejected_next_vowel_row(
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if _alias_type_for_row(row.alias, config.alias_type) != "cv":
        return row
    if row.anchor is None:
        return row
    if set(row.applied_rules) - {"cvvc_following_cv_block_cutoff_cap"}:
        return row

    warnings = set(row.warnings)
    warnings.update(row.anchor.warnings)
    if "local_refine_rejected_slot_boundary" not in warnings:
        return row

    anchor_abs = float(row.anchor.anchor_abs_ms)
    next_vowel = row.anchor.next_vowel_abs_ms
    nucleus_abs = row.anchor.vowel_nucleus_abs_ms
    if next_vowel is None or nucleus_abs is None:
        return row
    next_vowel = float(next_vowel)
    nucleus_abs = float(nucleus_abs)
    if not all(np.isfinite(value) for value in (anchor_abs, next_vowel, nucleus_abs)):
        return row
    if abs(nucleus_abs - anchor_abs) > float(JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_NUCLEUS_NEAR_MS):
        return row

    gap = next_vowel - anchor_abs
    if gap < float(JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_MIN_GAP_MS):
        return row

    current_pre_abs = float(row.timing.offset) + float(row.timing.preutterance)
    target_pre_abs = (
        anchor_abs
        + gap * float(JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_TARGET_RATIO)
        + float(JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_TARGET_BIAS_MS)
    )
    target_pre_abs = min(target_pre_abs, next_vowel - 5.0)
    if target_pre_abs - current_pre_abs < float(JAPANESE_CVVC_CV_REJECTED_NEXT_VOWEL_MIN_FORWARD_MS):
        return row
    candidate_offset = max(0.0, target_pre_abs - float(row.timing.preutterance))
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (current_pre_abs, target_pre_abs, candidate_offset, current_offset)):
        return row

    timing = OtoTiming(
        offset=candidate_offset,
        consonant=float(row.timing.consonant),
        cutoff=float(row.timing.cutoff),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_cv_rejected_next_vowel_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_cv_rejected_next_vowel_gap:{gap:.1f}",
                    f"cvvc_cv_rejected_next_vowel_pre_abs:{current_pre_abs:.1f}->{target_pre_abs:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_cv_rejected_next_vowel_repair"))),
    )


def _repair_cvvc_vc_next_vowel_local_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    return [_repair_cvvc_vc_next_vowel_local_row(row, config) for row in rows]


def _repair_cvvc_vc_next_vowel_local_row(
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if _alias_type_for_row(row.alias, config.alias_type) != "vc":
        return row
    if row.anchor is None:
        return row
    if row.applied_rules:
        return row
    warnings = set(row.warnings)
    warnings.update(row.anchor.warnings)
    if "local_refine_low_margin" not in warnings and "local_refine_rejected_slot_boundary" not in warnings:
        return row
    anchor_abs = float(row.anchor.anchor_abs_ms)
    next_vowel = row.anchor.next_vowel_abs_ms
    if next_vowel is None:
        return row
    next_vowel = float(next_vowel)
    if not all(np.isfinite(value) for value in (anchor_abs, next_vowel)):
        return row
    gap = next_vowel - anchor_abs
    if gap < float(JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_MIN_GAP_MS):
        return row

    current_pre_abs = float(row.timing.offset) + float(row.timing.preutterance)
    target_pre_abs = (
        (anchor_abs + next_vowel) * 0.5
        + float(JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_TARGET_BIAS_MS)
    )
    if target_pre_abs - current_pre_abs < float(JAPANESE_CVVC_VC_NEXT_VOWEL_LOCAL_MIN_FORWARD_MS):
        return row
    candidate_offset = max(0.0, target_pre_abs - float(row.timing.preutterance))
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (current_pre_abs, target_pre_abs, candidate_offset, current_offset)):
        return row

    timing = OtoTiming(
        offset=candidate_offset,
        consonant=float(row.timing.consonant),
        cutoff=float(row.timing.cutoff),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_vc_next_vowel_local_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_vc_next_vowel_local_gap:{gap:.1f}",
                    f"cvvc_vc_next_vowel_local_pre_abs:{current_pre_abs:.1f}->{target_pre_abs:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_vc_next_vowel_local_repair"))),
    )


def _repair_cvvc_pure_vowel_transition_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        phones = _cvvc_pure_vowel_vv_phones(row, config)
        if phones is None:
            continue
        if "cvvc_pure_vowel_onset_transition_repair" in row.applied_rules:
            continue
        left_phone, right_phone = phones
        following_v_indices = _following_cvvc_v_row_indices(out, index, right_phone, config)
        if not following_v_indices:
            continue
        if any("cvvc_pure_vowel_onset_v_repair" in out[v_index].applied_rules for v_index in following_v_indices):
            continue
        if not any("nucleus_span_too_short" in out[v_index].warnings for v_index in following_v_indices):
            continue
        previous_vowel_end = _cvvc_previous_vowel_end_abs(row)
        if previous_vowel_end is None:
            continue
        candidate_offset = max(0.0, previous_vowel_end - float(JAPANESE_CVVC_PURE_VOWEL_VV_LEFT_KEEP_MS))
        current_offset = float(row.timing.offset)
        if not all(np.isfinite(value) for value in (candidate_offset, current_offset)):
            continue
        if current_offset - candidate_offset <= float(JAPANESE_CVVC_PURE_VOWEL_REPAIR_MIN_LATE_MS):
            continue
        candidate_v_offset = max(
            0.0,
            candidate_offset + float(JAPANESE_CVVC_PURE_VOWEL_V_OFFSET_AFTER_VV_MS),
        )
        candidate_end = _cvvc_pure_vowel_transition_candidate_end(
            out,
            index,
            following_v_indices,
            candidate_offset=candidate_offset,
            candidate_v_offset=candidate_v_offset,
        )
        out[index] = _replace_cvvc_pure_vowel_transition_vv_timing(
            row,
            candidate_offset,
            candidate_end,
            left_phone=left_phone,
            right_phone=right_phone,
        )
        for v_index in following_v_indices:
            out[v_index] = _replace_cvvc_pure_vowel_v_timing(
                out[v_index],
                candidate_v_offset,
                candidate_end,
                reference_alias=out[index].alias,
            )
    return out


def _cvvc_pure_vowel_vv_phones(
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> tuple[str, str] | None:
    if _alias_type_for_row(row.alias, config.alias_type) != "vv":
        return None
    phones = tuple(_alias_phone_sequence(row.alias))
    if len(phones) != 2:
        return None
    left_phone, right_phone = (str(phone).strip().lower() for phone in phones)
    if not left_phone or not right_phone:
        return None
    if not is_vowel_phone(left_phone) or not is_vowel_phone(right_phone):
        return None
    return left_phone, right_phone


def _following_cvvc_v_row_indices(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    right_phone: str,
    config: OtoAdapterConfig,
) -> list[int]:
    out: list[int] = []
    current_wav = str(rows[start_index].wav or "").strip().lower()
    target = str(right_phone or "").strip().lower()
    for index in range(start_index + 1, len(rows)):
        row = rows[index]
        if str(row.wav or "").strip().lower() != current_wav:
            break
        if _alias_type_for_row(row.alias, config.alias_type) != "v":
            break
        phones = tuple(_alias_phone_sequence(row.alias))
        if len(phones) != 1 or str(phones[0]).strip().lower() != target:
            break
        out.append(index)
    return out


def _cvvc_previous_vowel_end_abs(row: AdaptedOtoRow) -> float | None:
    if row.anchor is None:
        return None
    previous_end = row.anchor.previous_vowel_end_abs_ms
    if previous_end is None:
        return None
    previous_end = float(previous_end)
    if not np.isfinite(previous_end) or previous_end <= 0.0:
        return None
    return previous_end


def _cvvc_pure_vowel_transition_candidate_end(
    rows: Sequence[AdaptedOtoRow],
    vv_index: int,
    following_v_indices: Sequence[int],
    *,
    candidate_offset: float,
    candidate_v_offset: float,
) -> float:
    ends = [
        float(absolute_oto_positions(rows[vv_index].timing)["cutoff_abs"]),
        candidate_offset + float(JAPANESE_CVVC_PURE_VOWEL_VV_CONSONANT_MS) + 8.0,
        candidate_v_offset + float(JAPANESE_CVVC_PURE_VOWEL_V_CONSONANT_MS) + 8.0,
    ]
    for index in following_v_indices:
        ends.append(float(absolute_oto_positions(rows[index].timing)["cutoff_abs"]))
    finite_ends = [float(value) for value in ends if np.isfinite(value)]
    if not finite_ends:
        return max(
            candidate_offset + float(JAPANESE_CVVC_PURE_VOWEL_VV_CONSONANT_MS) + 8.0,
            candidate_v_offset + float(JAPANESE_CVVC_PURE_VOWEL_V_CONSONANT_MS) + 8.0,
        )
    return max(finite_ends)


def _replace_cvvc_pure_vowel_transition_vv_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    candidate_end: float,
    *,
    left_phone: str,
    right_phone: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_PURE_VOWEL_VV_CONSONANT_MS),
        cutoff=-max(0.0, float(candidate_end) - float(candidate_offset)),
        preutterance=float(JAPANESE_CVVC_PURE_VOWEL_VV_PRE_MS),
        overlap=float(JAPANESE_CVVC_PURE_VOWEL_VV_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_pure_vowel_transition:{left_phone}->{right_phone}",
                    f"cvvc_pure_vowel_vv_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_pure_vowel_vv_end:{current_end:.1f}->{candidate_end:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_pure_vowel_transition_repair"))),
    )


def _replace_cvvc_pure_vowel_v_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    candidate_end: float,
    *,
    reference_alias: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    if current_offset - float(candidate_offset) <= float(JAPANESE_CVVC_PURE_VOWEL_REPAIR_MIN_LATE_MS):
        return row
    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_PURE_VOWEL_V_CONSONANT_MS),
        cutoff=-max(0.0, float(candidate_end) - float(candidate_offset)),
        preutterance=float(JAPANESE_CVVC_PURE_VOWEL_V_PRE_MS),
        overlap=float(JAPANESE_CVVC_PURE_VOWEL_V_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_pure_vowel_v_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_pure_vowel_v_end:{current_end:.1f}->{candidate_end:.1f}",
                    f"cvvc_pure_vowel_v_reference:{reference_alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_pure_vowel_v_repair"))),
    )


def _repair_cvvc_following_glide_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        previous = out[index - 1] if index > 0 else None
        next_vc = _next_cvvc_vc_row(out, index, config)
        out[index] = _repair_cvvc_following_glide_row(previous, row, next_vc, config)
    return out


def _repair_cvvc_following_cv_block_cutoff_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    direct_indices = set(_cvvc_following_cv_block_row_indices(out, config))
    companion_indices: set[int] = set()
    for index in sorted(direct_indices):
        if index < 0 or index >= len(out):
            continue
        before_timing = out[index].timing
        out[index] = _cap_cvvc_following_cv_block_cutoff(out[index])
        if out[index].timing == before_timing:
            continue
        for companion_index in (index - 1, index + 1):
            if companion_index < 0 or companion_index >= len(out):
                continue
            if companion_index in direct_indices or companion_index in companion_indices:
                continue
            companion = out[companion_index]
            if str(companion.wav or "").strip().lower() != str(out[index].wav or "").strip().lower():
                continue
            if _alias_type_for_row(companion.alias, config.alias_type) != "vcv":
                continue
            if not _oto_timing_close(companion.timing, before_timing):
                continue
            out[companion_index] = _cap_cvvc_following_cv_block_cutoff(companion)
            companion_indices.add(companion_index)
    return out


def _cvvc_following_cv_block_row_indices(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[int]:
    out: list[int] = []
    segment_start = 0
    while segment_start < len(rows):
        wav_key = str(rows[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(rows) and str(rows[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        out.extend(_cvvc_following_cv_block_row_indices_for_segment(rows, config, segment_start, segment_end))
        segment_start = segment_end
    return out


def _cvvc_following_cv_block_row_indices_for_segment(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> list[int]:
    out: list[int] = []
    seen_transition = False
    in_block = False
    for row_idx in range(start, end):
        row = rows[row_idx]
        alias = str(row.alias or "").strip()
        if not alias or _is_terminal_silence_alias(alias) or _is_nonphonetic_special_alias(alias):
            continue
        role = _alias_type_for_row(alias, config.alias_type)
        if not in_block:
            if role == "vc":
                seen_transition = True
                continue
            if seen_transition and _is_cvvc_following_cv_block_role(role):
                in_block = True
                out.append(row_idx)
                continue
            if role in {"cv_head", "vv"}:
                seen_transition = False
            continue
        if _is_cvvc_following_cv_block_role(role):
            out.append(row_idx)
            continue
        if role in {"vc", "vv", "cv_head"}:
            break
    return out


def _cap_cvvc_following_cv_block_cutoff(row: AdaptedOtoRow) -> AdaptedOtoRow:
    cutoff_abs = abs(float(row.timing.cutoff))
    min_cutoff_abs = max(0.0, float(row.timing.consonant) + 8.0)
    cap_abs = max(float(JAPANESE_CVVC_FOLLOWING_CV_BLOCK_CUTOFF_CAP_MS), min_cutoff_abs)
    if not all(np.isfinite(value) for value in (cutoff_abs, min_cutoff_abs, cap_abs)):
        return row
    if cutoff_abs <= cap_abs + 1e-6:
        return row
    timing = OtoTiming(
        offset=float(row.timing.offset),
        consonant=float(row.timing.consonant),
        cutoff=-cap_abs,
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    capped_abs = abs(float(timing.cutoff))
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_following_cv_block_cutoff_capped:{cutoff_abs:.1f}->{capped_abs:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_following_cv_block_cutoff_cap"))),
    )


def _repair_cvvc_cv_next_transition_cutoff_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    for index, row in enumerate(out):
        if _alias_type_for_row(row.alias, config.alias_type) != "cv":
            continue
        next_transition = _next_cvvc_cv_cutoff_transition_row(out, index, config)
        if next_transition is None:
            continue
        out[index] = _cap_cvvc_cv_cutoff_before_next_transition(row, next_transition)
    return out


def _next_cvvc_cv_cutoff_transition_row(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow | None:
    wav_key = str(rows[start_index].wav or "").strip().lower()
    source_row = rows[start_index]
    for index in range(start_index + 1, len(rows)):
        row = rows[index]
        if str(row.wav or "").strip().lower() != wav_key:
            return None
        alias = str(row.alias or "").strip()
        if not alias or _is_terminal_silence_alias(alias) or _is_nonphonetic_special_alias(alias):
            continue
        role = _alias_type_for_row(alias, config.alias_type)
        if role == "vcv" and _is_cvvc_same_phone_vcv_companion(source_row, row, config):
            continue
        if role in {"vc", "vv", "vcv"}:
            return row
        if role == "cv_head":
            return None
    return None


def _is_cvvc_same_phone_vcv_companion(
    source_row: AdaptedOtoRow,
    candidate: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> bool:
    if _alias_type_for_row(candidate.alias, config.alias_type) != "vcv":
        return False
    source_phones = tuple(_alias_phone_sequence(source_row.alias))
    candidate_phones = tuple(_alias_phone_sequence(candidate.alias))
    if not source_phones or source_phones != candidate_phones:
        return False
    return _oto_timing_close(source_row.timing, candidate.timing)


def _cap_cvvc_cv_cutoff_before_next_transition(
    row: AdaptedOtoRow,
    next_transition: AdaptedOtoRow,
) -> AdaptedOtoRow:
    offset = float(row.timing.offset)
    cutoff_abs = abs(float(row.timing.cutoff))
    next_offset = float(next_transition.timing.offset)
    if _alias_has_attached_pitch_suffix(row.alias):
        pad_ms = float(JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_PAD_MS)
        min_excess_ms = float(JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS)
    else:
        pad_ms = float(JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_PAD_MS)
        min_excess_ms = float(JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS)
    target_end = next_offset + pad_ms
    current_end = offset + cutoff_abs
    values = (offset, cutoff_abs, next_offset, target_end, current_end)
    if not all(np.isfinite(value) for value in values):
        return row
    if current_end <= target_end + min_excess_ms:
        return row
    min_cutoff_abs = max(0.0, float(row.timing.consonant) + 8.0)
    capped_abs = max(min_cutoff_abs, target_end - offset)
    if capped_abs >= cutoff_abs - 1e-6:
        return row
    timing = OtoTiming(
        offset=offset,
        consonant=float(row.timing.consonant),
        cutoff=-float(capped_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    applied_abs = abs(float(timing.cutoff))
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_cv_next_transition_cutoff_capped:{cutoff_abs:.1f}->{applied_abs:.1f}",
                    f"cvvc_cv_next_transition_cutoff_reference:{next_transition.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_cv_next_transition_cutoff_cap"))),
    )


def _repair_cvvc_cv_sequence_next_cv_cutoff_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    for index in range(len(out) - 1):
        current = out[index]
        following = out[index + 1]
        if str(current.wav or "").strip().lower() != str(following.wav or "").strip().lower():
            continue
        if _alias_type_for_row(current.alias, config.alias_type) != "cv":
            continue
        if _alias_type_for_row(following.alias, config.alias_type) != "cv":
            continue
        out[index] = _cap_cvvc_cv_sequence_cutoff_before_next_cv(current, following)
    return out


def _cap_cvvc_cv_sequence_cutoff_before_next_cv(
    row: AdaptedOtoRow,
    next_cv: AdaptedOtoRow,
) -> AdaptedOtoRow:
    offset = float(row.timing.offset)
    cutoff_abs = abs(float(row.timing.cutoff))
    next_offset = float(next_cv.timing.offset)
    target_end = next_offset + float(JAPANESE_CVVC_CV_SEQUENCE_NEXT_CV_CUTOFF_PAD_MS)
    current_end = offset + cutoff_abs
    values = (offset, cutoff_abs, next_offset, target_end, current_end)
    if not all(np.isfinite(value) for value in values):
        return row
    if current_end <= target_end + float(JAPANESE_CVVC_CV_SEQUENCE_NEXT_CV_CUTOFF_MIN_EXCESS_MS):
        return row
    min_cutoff_abs = max(0.0, float(row.timing.consonant) + 8.0)
    capped_abs = max(min_cutoff_abs, target_end - offset)
    if capped_abs >= cutoff_abs - 1e-6:
        return row
    timing = OtoTiming(
        offset=offset,
        consonant=float(row.timing.consonant),
        cutoff=-float(capped_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    applied_abs = abs(float(timing.cutoff))
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_cv_sequence_next_cv_cutoff_capped:{cutoff_abs:.1f}->{applied_abs:.1f}",
                    f"cvvc_cv_sequence_next_cv_cutoff_reference:{next_cv.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_cv_sequence_next_cv_cutoff_cap"))),
    )


def _oto_timing_close(left: OtoTiming, right: OtoTiming, *, tolerance_ms: float = 1e-6) -> bool:
    return all(
        abs(float(left_value) - float(right_value)) <= float(tolerance_ms)
        for left_value, right_value in zip(left.as_tuple(), right.as_tuple())
    )


def _repair_cvvc_initial_vowel_cv_head_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        next_vc = _next_cvvc_vc_row(out, index, config)
        out[index] = _repair_cvvc_initial_vowel_cv_head_row(row, next_vc, config)
    return out


def _repair_cvvc_initial_vowel_cv_head_row(
    row: AdaptedOtoRow,
    next_vc: AdaptedOtoRow | None,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if next_vc is None:
        return row
    if str(row.wav or "").strip().lower() != str(next_vc.wav or "").strip().lower():
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "cv_head":
        return row
    vowel_token = _cvvc_initial_cv_head_vowel_token(row.alias)
    if not vowel_token:
        return row
    if _cvvc_vc_alias_left_token(next_vc.alias) != vowel_token:
        return row

    row = _repair_cvvc_initial_vowel_cv_head_onset_row(row, next_vc, config)

    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    candidate_end = float(next_vc.timing.offset)
    min_end = float(row.timing.offset) + float(row.timing.consonant) + 8.0
    if not all(np.isfinite(value) for value in (current_end, candidate_end, min_end)):
        return row
    if candidate_end <= min_end:
        return row
    if current_end - candidate_end <= float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_CUTOFF_REPAIR_MIN_LONG_MS):
        return row

    timing = OtoTiming(
        offset=float(row.timing.offset),
        consonant=float(row.timing.consonant),
        cutoff=-max(0.0, candidate_end - float(row.timing.offset)),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_initial_vowel_cv_head_cutoff_repaired:{current_end:.1f}->{candidate_end:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_initial_vowel_cv_head_cutoff_repair"))),
    )


def _repair_cvvc_initial_vowel_cv_head_onset_row(
    row: AdaptedOtoRow,
    next_vc: AdaptedOtoRow | None,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if not _is_japanese_cvvc_config(config):
        return row
    if "cvvc_initial_vowel_cv_head_onset_repair" in row.applied_rules:
        return row
    if "cvvc_pure_vowel_sequence_head_repair" in row.applied_rules:
        return row
    if "cvvc_pure_vowel_onset_head_repair" in row.applied_rules:
        return row
    if row.anchor is None:
        return row
    if _anchor_expected_phone_index(row.anchor) != 0:
        return row
    active_start_f = _initial_vowel_cv_head_active_start_ms(row, next_vc)
    if active_start_f is None:
        return row
    current_offset = float(row.timing.offset)
    current_pre_abs = current_offset + float(row.timing.preutterance)
    if not all(np.isfinite(value) for value in (active_start_f, current_offset, current_pre_abs)):
        return row
    if active_start_f < float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_START_MS):
        return row
    late_delta = current_pre_abs - active_start_f
    early_delta = active_start_f - current_pre_abs
    if (
        late_delta < float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_LATE_MS)
        and early_delta < float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_EARLY_MS)
    ):
        return row

    pre = float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_PRE_MS)
    overlap = min(float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_OVERLAP_MS), pre)
    consonant = max(float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_CONSONANT_MS), pre + 8.0)
    candidate_offset = max(0.0, active_start_f - pre)
    candidate_end = max(
        candidate_offset + consonant + 8.0,
        active_start_f + float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_TAIL_AFTER_START_MS),
    )
    cutoff = -max(consonant + 8.0, candidate_end - candidate_offset)
    timing = OtoTiming(
        offset=candidate_offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=pre,
        overlap=overlap,
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_initial_vowel_cv_head_onset_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_initial_vowel_cv_head_onset_pre_abs:{current_pre_abs:.1f}->{active_start_f:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_initial_vowel_cv_head_onset_repair"))),
    )


def _initial_vowel_cv_head_active_start_ms(
    row: AdaptedOtoRow,
    next_vc: AdaptedOtoRow | None,
) -> float | None:
    candidates: list[float] = []
    for anchor in (row.anchor, next_vc.anchor if next_vc is not None else None):
        if anchor is None or anchor.vowel_start_abs_ms is None:
            continue
        value = float(anchor.vowel_start_abs_ms)
        if not np.isfinite(value):
            continue
        if value < float(JAPANESE_CVVC_INITIAL_VOWEL_CV_HEAD_ONSET_MIN_START_MS):
            continue
        candidates.append(value)
    if not candidates:
        return None
    if next_vc is not None:
        next_offset = float(next_vc.timing.offset)
        if np.isfinite(next_offset):
            return min(candidates, key=lambda value: abs(value - next_offset))
    return candidates[0]


def _repair_cvvc_initial_consonant_cv_head_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        reference_cv = _cvvc_initial_consonant_cv_head_reference(out, index, config)
        out[index] = _repair_cvvc_initial_consonant_cv_head_row(row, reference_cv, config)
    return out


def _repair_cvvc_initial_consonant_cv_head_row(
    row: AdaptedOtoRow,
    following_cv: AdaptedOtoRow | None,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if following_cv is None:
        return row
    if str(row.wav or "").strip().lower() != str(following_cv.wav or "").strip().lower():
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "cv_head":
        return row
    phones = _alias_phone_sequence(row.alias)
    if not _cv_head_phones_are_consonant_like(phones):
        return row

    current_offset = float(row.timing.offset)
    reference_offset = float(following_cv.timing.offset)
    if not all(np.isfinite(value) for value in (current_offset, reference_offset)):
        return row
    forward_ms = reference_offset - current_offset
    if forward_ms <= float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_MIN_FORWARD_MS):
        return row
    capped_offset = min(
        reference_offset,
        current_offset + float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_MAX_SHIFT_MS),
    )
    if capped_offset <= current_offset + 1e-6:
        return row

    cutoff_ms = max(
        float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_CUTOFF_MS),
        float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_CONSONANT_MS) + 8.0,
    )
    timing = OtoTiming(
        offset=capped_offset,
        consonant=float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_CONSONANT_MS),
        cutoff=-cutoff_ms,
        preutterance=float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_PRE_MS),
        overlap=float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    warnings: tuple[str, ...] = (
        f"cvvc_initial_consonant_cv_head_following_cv_repaired:{current_offset:.1f}->{capped_offset:.1f}",
        f"cvvc_initial_consonant_cv_head_following_cv_reference:{following_cv.alias}",
    )
    if capped_offset < reference_offset - 1e-6:
        warnings = (
            *warnings,
            f"cvvc_initial_consonant_cv_head_shift_capped:{forward_ms:.1f}->{float(JAPANESE_CVVC_INITIAL_CONSONANT_CV_HEAD_MAX_SHIFT_MS):.1f}",
        )
    return replace(
        row,
        timing=timing,
        warnings=tuple(dict.fromkeys((*row.warnings, *warnings, *timing_warnings))),
        applied_rules=tuple(
            dict.fromkeys((*row.applied_rules, "cvvc_initial_consonant_cv_head_following_cv_repair"))
        ),
    )


def _cvvc_initial_consonant_cv_head_reference(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow | None:
    immediate = _immediate_cvvc_vowel_head_after_consonant_head(rows, start_index, config)
    if immediate is not None:
        return immediate
    return _first_cvvc_following_cv_after_vc_block(rows, start_index, config)


def _immediate_cvvc_vowel_head_after_consonant_head(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow | None:
    if start_index + 1 >= len(rows):
        return None
    current_wav = str(rows[start_index].wav or "").strip().lower()
    next_row = rows[start_index + 1]
    if str(next_row.wav or "").strip().lower() != current_wav:
        return None
    role = _alias_type_for_row(next_row.alias, config.alias_type)
    if role != "cv_head":
        return None
    if _cv_head_phones_are_consonant_like(_alias_phone_sequence(next_row.alias)):
        return None
    return next_row


def _first_cvvc_following_cv_after_vc_block(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow | None:
    current_wav = str(rows[start_index].wav or "").strip().lower()
    seen_vc = False
    for row in rows[start_index + 1 :]:
        if str(row.wav or "").strip().lower() != current_wav:
            return None
        alias = str(row.alias or "").strip()
        if not alias or _is_nonphonetic_special_alias(alias):
            continue
        if _is_terminal_silence_alias(alias):
            return None
        role = _alias_type_for_row(alias, config.alias_type)
        if role == "vc":
            seen_vc = True
            continue
        if seen_vc and role in {"cv", "v"}:
            return row
        if role in {"cv_head", "vv"}:
            return None
    return None


def _repair_cvvc_pure_vowel_sequence_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_pure_vowel_sequence_segment(out, config, segment_start, segment_end)
        segment_start = segment_end
    return out


def _repair_cvvc_pure_vowel_sequence_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> None:
    if end - start < 3:
        return
    wav = str(rows[start].wav or "").strip()
    if not Path(wav).name.startswith("_"):
        return
    if _alias_type_for_row(rows[start].alias, config.alias_type) != "cv_head":
        return
    terminal = rows[end - 1]
    if not _is_terminal_silence_alias(terminal.alias):
        return
    if "japanese_cvvc_terminal_silence_vowel_end_anchor" not in terminal.applied_rules:
        return
    middle_indices = list(range(start + 1, end - 1))
    if len(middle_indices) < 2:
        return
    if any(not _is_cvvc_pure_vowel_sequence_middle_row(rows[index], config) for index in middle_indices):
        return

    terminal_offset = float(terminal.timing.offset)
    if not np.isfinite(terminal_offset):
        return
    candidate_offsets: list[float] = []
    middle_count = len(middle_indices)
    for order, _index in enumerate(middle_indices):
        remaining = middle_count - 1 - order
        offset = (
            terminal_offset
            - float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_TERMINAL_TAIL_MS)
            - (float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_STEP_MS) * float(remaining))
        )
        if not np.isfinite(offset) or offset < 0.0:
            return
        candidate_offsets.append(offset)

    for index, candidate_offset in zip(middle_indices, candidate_offsets):
        rows[index] = _replace_cvvc_pure_vowel_sequence_row_timing(
            rows[index],
            candidate_offset,
            terminal_alias=terminal.alias,
        )
    first_vv_offset = candidate_offsets[0]
    head_offset = max(0.0, first_vv_offset - float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_HEAD_LEAD_MS))
    rows[start] = _replace_cvvc_pure_vowel_sequence_head_offset(
        rows[start],
        head_offset,
        reference_alias=rows[middle_indices[0]].alias,
    )


def _is_cvvc_pure_vowel_sequence_middle_row(row: AdaptedOtoRow, config: OtoAdapterConfig) -> bool:
    alias = str(row.alias or "").strip()
    if " " not in alias:
        return False
    role = _alias_type_for_row(alias, config.alias_type)
    if role not in {"vc", "vcv", "vv"}:
        return False
    phones = _alias_phone_sequence(alias)
    return bool(phones) and all(is_vowel_phone(phone) for phone in phones)


def _replace_cvvc_pure_vowel_sequence_row_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    terminal_alias: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_PRE_MS),
        overlap=float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_VV_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_pure_vowel_sequence_row_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_pure_vowel_sequence_terminal_reference:{terminal_alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_pure_vowel_sequence_row_repair"))),
    )


def _replace_cvvc_pure_vowel_sequence_head_offset(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    reference_alias: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    cutoff_abs, cutoff_warnings, cutoff_repaired = _cvvc_pure_vowel_head_cutoff_abs(
        row,
        float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_HEAD_CUTOFF_MIN_MS),
        warning_prefix="cvvc_pure_vowel_sequence_head",
    )
    offset_repaired = np.isfinite(current_offset) and abs(current_offset - float(candidate_offset)) > 1e-6
    if not offset_repaired and not cutoff_repaired:
        return row
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(row.timing.consonant),
        cutoff=-float(cutoff_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    warnings: list[str] = []
    if offset_repaired:
        warnings.extend(
            (
                f"cvvc_pure_vowel_sequence_head_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                f"cvvc_pure_vowel_sequence_head_reference:{reference_alias}",
            )
        )
    warnings.extend(cutoff_warnings)
    applied_rules = [*row.applied_rules]
    if offset_repaired:
        applied_rules.append("cvvc_pure_vowel_sequence_head_repair")
    if cutoff_repaired:
        applied_rules.append("cvvc_pure_vowel_sequence_head_cutoff_repair")
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    *warnings,
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys(applied_rules)),
    )


def _cvvc_pure_vowel_head_cutoff_abs(
    row: AdaptedOtoRow,
    floor_ms: float,
    *,
    warning_prefix: str,
) -> tuple[float, tuple[str, ...], bool]:
    current_abs = abs(float(row.timing.cutoff))
    floor = max(float(floor_ms), float(row.timing.consonant) + 8.0)
    if not all(np.isfinite(value) for value in (current_abs, floor)):
        return current_abs, (), False
    if current_abs > float(JAPANESE_CVVC_PURE_VOWEL_HEAD_CUTOFF_REPAIR_MAX_CURRENT_MS):
        return current_abs, (), False
    if floor <= current_abs + 1e-6:
        return current_abs, (), False
    return (
        floor,
        (f"{warning_prefix}_cutoff_repaired:{current_abs:.1f}->{floor:.1f}",),
        True,
    )


def _repair_cvvc_pure_vowel_onset_sequence_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
    *,
    file_duration_ms: float | None,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_pure_vowel_onset_sequence_segment(
            out,
            config,
            segment_start,
            segment_end,
            file_duration_ms=file_duration_ms,
        )
        segment_start = segment_end
    return out


def _repair_cvvc_pure_vowel_onset_sequence_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
    *,
    file_duration_ms: float | None,
) -> None:
    if end - start < 3:
        return
    wav = str(rows[start].wav or "").strip()
    if not Path(wav).name.startswith("_"):
        return
    if any(
        _is_terminal_silence_alias(rows[index].alias)
        or _is_nonphonetic_special_alias(rows[index].alias)
        for index in range(start, end)
    ):
        return
    if _alias_type_for_row(rows[start].alias, config.alias_type) != "cv_head":
        _repair_cvvc_pure_vowel_onset_indexed_sequence_segment(
            rows,
            config,
            start,
            end,
            file_duration_ms=file_duration_ms,
        )
        return
    if end - start < 4:
        return
    for index in range(start + 1, end):
        row = rows[index]
        is_last = index == end - 1
        if is_last and _is_cvvc_trailing_r_row(row):
            continue
        if not _is_cvvc_pure_vowel_onset_sequence_row(row, config):
            return

    active_start = _cvvc_reliable_vowel_start_abs(rows[start:end])
    if active_start is None or active_start < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS):
        return

    head_offset = max(0.0, active_start - float(JAPANESE_CVVC_PURE_VOWEL_SEQUENCE_HEAD_LEAD_MS))
    has_trailing_release = _is_cvvc_trailing_r_row(rows[end - 1])
    rows[start] = _replace_cvvc_pure_vowel_onset_head_offset(
        rows[start],
        head_offset,
        active_start=active_start,
        cutoff_floor_ms=_cvvc_pure_vowel_onset_head_cutoff_floor_ms(
            rows[start],
            has_trailing_release=has_trailing_release,
        ),
    )

    cursor = active_start + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_FIRST_OFFSET_MS)
    previous_grid_offset: float | None = None
    duration = float(file_duration_ms or 0.0)
    if not np.isfinite(duration) or duration <= 0.0:
        duration = 0.0
    for index in range(start + 1, end):
        row = rows[index]
        is_last = index == end - 1
        if is_last and _is_cvvc_trailing_r_row(row):
            floor_offset = (
                previous_grid_offset + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRANSITION_STEP_MS)
                if previous_grid_offset is not None
                else cursor
            )
            duration_offset = duration - float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRAILING_R_TAIL_MS) if duration > 0.0 else floor_offset
            candidate_offset = max(float(floor_offset), float(duration_offset), 0.0)
            rows[index] = _replace_cvvc_pure_vowel_onset_transition_timing(
                row,
                candidate_offset,
                active_start=active_start,
                rule_name="cvvc_pure_vowel_onset_trailing_r_repair",
            )
            continue

        role = _alias_type_for_row(row.alias, config.alias_type)
        if role == "v":
            candidate_offset = (
                previous_grid_offset + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_HALF_STEP_MS)
                if previous_grid_offset is not None
                else cursor
            )
            rows[index] = _replace_cvvc_pure_vowel_onset_v_timing(
                row,
                candidate_offset,
                active_start=active_start,
            )
            cursor = max(
                cursor,
                candidate_offset + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_HALF_STEP_MS),
            )
            previous_grid_offset = candidate_offset
            continue

        candidate_offset = cursor
        rows[index] = _replace_cvvc_pure_vowel_onset_transition_timing(
            row,
            candidate_offset,
            active_start=active_start,
            rule_name="cvvc_pure_vowel_onset_transition_repair",
        )
        previous_grid_offset = candidate_offset
        cursor = candidate_offset + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRANSITION_STEP_MS)


def _repair_cvvc_pure_vowel_onset_indexed_sequence_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
    *,
    file_duration_ms: float | None,
) -> None:
    if end - start < 3:
        return
    for index in range(start, end):
        row = rows[index]
        is_last = index == end - 1
        if is_last and _is_cvvc_trailing_r_row(row):
            continue
        if not _is_cvvc_pure_vowel_onset_sequence_row(row, config):
            return
    active_start = _cvvc_headless_vc_onset_reference_abs(rows[start:end])
    if active_start is None or active_start < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS):
        return

    candidates: list[tuple[int, float, str]] = []
    duration = float(file_duration_ms or 0.0)
    if not np.isfinite(duration) or duration <= 0.0:
        duration = 0.0
    previous_candidate: float | None = None
    for index in range(start, end):
        row = rows[index]
        target_index = _anchor_expected_phone_index(row.anchor)
        if target_index is None or target_index <= 0:
            return
        is_last = index == end - 1
        if is_last and _is_cvvc_trailing_r_row(row):
            floor_offset = (
                previous_candidate + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRANSITION_STEP_MS)
                if previous_candidate is not None
                else active_start + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_FIRST_OFFSET_MS)
            )
            duration_offset = duration - float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRAILING_R_TAIL_MS) if duration > 0.0 else floor_offset
            candidate_offset = max(float(floor_offset), float(duration_offset), 0.0)
            rule_name = "cvvc_pure_vowel_onset_trailing_r_repair"
        else:
            base_offset = (
                active_start
                + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_FIRST_OFFSET_MS)
                + (float(target_index - 1) * float(JAPANESE_CVVC_PURE_VOWEL_ONSET_TRANSITION_STEP_MS))
            )
            if _alias_type_for_row(row.alias, config.alias_type) == "v":
                candidate_offset = base_offset + float(JAPANESE_CVVC_PURE_VOWEL_ONSET_HALF_STEP_MS)
                rule_name = "cvvc_pure_vowel_onset_v_repair"
            else:
                candidate_offset = base_offset
                rule_name = "cvvc_pure_vowel_onset_transition_repair"
        if previous_candidate is not None and candidate_offset < previous_candidate - 1e-6:
            return
        candidates.append((index, float(candidate_offset), rule_name))
        previous_candidate = float(candidate_offset)

    if not any(candidate_offset - float(rows[index].timing.offset) >= 180.0 for index, candidate_offset, _rule in candidates):
        return
    for index, candidate_offset, rule_name in candidates:
        row = rows[index]
        if rule_name == "cvvc_pure_vowel_onset_v_repair":
            rows[index] = _replace_cvvc_pure_vowel_onset_v_timing(
                row,
                candidate_offset,
                active_start=active_start,
            )
        else:
            rows[index] = _replace_cvvc_pure_vowel_onset_transition_timing(
                row,
                candidate_offset,
                active_start=active_start,
                rule_name=rule_name,
            )


def _anchor_expected_phone_index(anchor: OtoAnchor | None) -> int | None:
    if anchor is None or anchor.expected_phone_index is None:
        return None
    try:
        return int(anchor.expected_phone_index)
    except (TypeError, ValueError):
        return None


def _is_cvvc_pure_vowel_onset_sequence_row(row: AdaptedOtoRow, config: OtoAdapterConfig) -> bool:
    role = _alias_type_for_row(row.alias, config.alias_type)
    if role not in {"v", "vc", "vcv", "vv"}:
        return False
    phones = _alias_phone_sequence(row.alias)
    return bool(phones) and all(is_vowel_phone(phone) for phone in phones)


def _is_cvvc_trailing_r_row(row: AdaptedOtoRow) -> bool:
    return _is_cvvc_terminal_release_r_alias(row.alias)


def _is_cvvc_terminal_release_r_alias(alias: str) -> bool:
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return False
    left_phones = _alias_phone_sequence(parts[0])
    if len(left_phones) != 1 or not is_vowel_phone(left_phones[0]):
        return False
    right = _strip_alias_attached_pitch_suffix_token_preserve_case(parts[-1])
    return right == "R"


def _strip_alias_attached_pitch_suffix_token_preserve_case(token: str) -> str:
    normalized = str(token or "").strip()
    if not normalized:
        return ""
    match = ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(normalized)
    if match is None:
        return normalized
    if match.start() <= 0:
        return ""
    stripped = normalized[: match.start()].rstrip("_- ").strip()
    return stripped or normalized


def _cvvc_reliable_vowel_start_abs(rows: Sequence[AdaptedOtoRow]) -> float | None:
    starts: list[float] = []
    for row in rows:
        anchor = row.anchor
        if anchor is None:
            continue
        start = anchor.vowel_start_abs_ms
        end = anchor.vowel_end_abs_ms
        if start is None or end is None:
            continue
        start_f = float(start)
        end_f = float(end)
        if not np.isfinite(start_f) or not np.isfinite(end_f):
            continue
        if end_f - start_f < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_SPAN_MS):
            continue
        starts.append(start_f)
    return min(starts) if starts else None


def _replace_cvvc_pure_vowel_onset_head_offset(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    active_start: float,
    cutoff_floor_ms: float,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    cutoff_abs, cutoff_warnings, cutoff_repaired = _cvvc_pure_vowel_head_cutoff_abs(
        row,
        float(cutoff_floor_ms),
        warning_prefix="cvvc_pure_vowel_onset_head",
    )
    offset_repaired = np.isfinite(current_offset) and abs(current_offset - float(candidate_offset)) > 1e-6
    if not offset_repaired and not cutoff_repaired:
        return row
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(row.timing.consonant),
        cutoff=-float(cutoff_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    warnings: list[str] = []
    if offset_repaired:
        warnings.extend(
            (
                f"cvvc_pure_vowel_onset_head_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                f"cvvc_pure_vowel_onset_reference:{active_start:.1f}",
            )
        )
    warnings.extend(cutoff_warnings)
    applied_rules = [*row.applied_rules]
    if offset_repaired:
        applied_rules.append("cvvc_pure_vowel_onset_head_repair")
    if cutoff_repaired:
        applied_rules.append("cvvc_pure_vowel_onset_head_cutoff_repair")
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    *warnings,
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys(applied_rules)),
    )


def _cvvc_pure_vowel_onset_head_cutoff_floor_ms(
    row: AdaptedOtoRow,
    *,
    has_trailing_release: bool,
) -> float:
    phones = tuple(_alias_phone_sequence(row.alias))
    if has_trailing_release and _alias_token_is_non_n_vowel(phones):
        return float(JAPANESE_CVVC_PURE_VOWEL_ONSET_HEAD_VOWEL_RELEASE_CUTOFF_MIN_MS)
    return float(JAPANESE_CVVC_PURE_VOWEL_ONSET_HEAD_CUTOFF_MIN_MS)


def _replace_cvvc_pure_vowel_onset_transition_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    active_start: float,
    rule_name: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_PRE_MS),
        overlap=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"{rule_name}:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_pure_vowel_onset_reference:{active_start:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, rule_name))),
    )


def _replace_cvvc_pure_vowel_onset_v_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    active_start: float,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_V_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_PURE_VOWEL_ONSET_V_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_V_PRE_MS),
        overlap=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_V_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_pure_vowel_onset_v_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_pure_vowel_onset_reference:{active_start:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_pure_vowel_onset_v_repair"))),
    )


def _repair_cvvc_headless_vc_onset_sequence_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_headless_vc_onset_sequence_segment(out, config, segment_start, segment_end)
        segment_start = segment_end
    return out


def _repair_cvvc_headless_vc_onset_sequence_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> None:
    if end - start < 2:
        return
    wav = str(rows[start].wav or "").strip()
    if not Path(wav).name.startswith("_"):
        return
    if any(_is_terminal_silence_alias(rows[index].alias) for index in range(start, end)):
        return
    if not all(_is_cvvc_headless_vc_onset_sequence_row(rows[index], config) for index in range(start, end)):
        return
    if not any(_cvvc_alias_right_token_has_true_consonant(rows[index].alias) for index in range(start, end)):
        return

    active_start = _cvvc_headless_vc_onset_reference_abs(rows[start:end])
    if active_start is None or active_start < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS):
        return

    candidates: list[tuple[int, float]] = []
    for order, index in enumerate(range(start, end)):
        candidate_offset = (
            float(active_start)
            + float(JAPANESE_CVVC_HEADLESS_VC_ONSET_FIRST_OFFSET_MS)
            + (float(order) * float(JAPANESE_CVVC_HEADLESS_VC_ONSET_STEP_MS))
        )
        if candidates and candidate_offset < candidates[-1][1] - 1e-6:
            return
        candidates.append((index, float(candidate_offset)))

    if not any(
        candidate_offset - float(rows[index].timing.offset)
        >= float(JAPANESE_CVVC_HEADLESS_VC_ONSET_MIN_EARLY_MS)
        for index, candidate_offset in candidates
    ):
        return

    for index, candidate_offset in candidates:
        rows[index] = _replace_cvvc_headless_vc_onset_timing(
            rows[index],
            candidate_offset,
            active_start=float(active_start),
        )


def _is_cvvc_headless_vc_onset_sequence_row(row: AdaptedOtoRow, config: OtoAdapterConfig) -> bool:
    if _is_nonphonetic_special_alias(row.alias):
        return False
    role = _alias_type_for_row(row.alias, config.alias_type)
    if role not in {"vc", "vcv", "vv"}:
        return False
    tokens = [part.strip() for part in str(row.alias or "").split() if part.strip()]
    if len(tokens) < 2:
        return False
    left_phones = _alias_phones(tokens[0])
    if not left_phones or not is_vowel_phone(left_phones[-1]):
        return False
    phones = _alias_phone_sequence(row.alias)
    return len(phones) >= 2 and is_vowel_phone(phones[0])


def _cvvc_headless_vc_onset_reference_abs(rows: Sequence[AdaptedOtoRow]) -> float | None:
    next_vowel = _cvvc_first_next_vowel_abs(rows)
    if next_vowel is not None:
        return max(0.0, next_vowel - float(JAPANESE_CVVC_HEADLESS_VC_ONSET_NEXT_VOWEL_LEAD_MS))
    return _cvvc_reliable_vowel_start_abs(rows)


def _cvvc_first_next_vowel_abs(rows: Sequence[AdaptedOtoRow]) -> float | None:
    for row in rows:
        anchor = row.anchor
        if anchor is None or anchor.next_vowel_abs_ms is None:
            continue
        value = float(anchor.next_vowel_abs_ms)
        if np.isfinite(value):
            return value
    return None


def _cvvc_alias_right_token_has_true_consonant(alias: str) -> bool:
    tokens = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(tokens) < 2:
        return False
    right_phones = _alias_phones(tokens[-1])
    return any(_is_cvvc_true_consonant_phone(phone) for phone in right_phones)


def _is_cvvc_true_consonant_phone(phone: str) -> bool:
    normalized = str(phone or "").strip().lower()
    if not normalized or normalized == "n":
        return False
    return not is_vowel_phone(normalized)


def _replace_cvvc_headless_vc_onset_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    active_start: float,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_HEADLESS_VC_ONSET_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_HEADLESS_VC_ONSET_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_HEADLESS_VC_ONSET_PRE_MS),
        overlap=float(JAPANESE_CVVC_HEADLESS_VC_ONSET_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_headless_vc_onset_repair:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_headless_vc_onset_reference:{active_start:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_headless_vc_onset_repair"))),
    )


def _repair_cvvc_initial_v_regular_pair_onset_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_initial_v_regular_pair_onset_segment(out, config, segment_start, segment_end)
        segment_start = segment_end
    return out


def _repair_cvvc_initial_v_regular_pair_onset_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> None:
    length = end - start
    if length < 5 or length % 2 == 0:
        return
    wav = str(rows[start].wav or "").strip()
    if not Path(wav).name.startswith("_"):
        return
    if any(_is_terminal_silence_alias(rows[index].alias) for index in range(start, end)):
        return
    if _alias_type_for_row(rows[start].alias, config.alias_type) != "v":
        return
    if _anchor_expected_phone_index(rows[start].anchor) not in (0, None):
        return
    for index in range(start + 1, end, 2):
        if _alias_type_for_row(rows[index].alias, config.alias_type) != "vc":
            return
        if _alias_type_for_row(rows[index + 1].alias, config.alias_type) != "cv":
            return

    first_pair_next_vowel = _cvvc_first_next_vowel_abs(rows[start + 1 : end])
    if first_pair_next_vowel is None or first_pair_next_vowel < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS):
        return

    first_v_offset = max(
        0.0,
        float(first_pair_next_vowel) - float(JAPANESE_CVVC_INITIAL_V_REGULAR_PAIR_V_LEAD_MS),
    )
    candidates: list[tuple[int, float, str]] = [(start, first_v_offset, "v")]
    first_center = (
        float(first_pair_next_vowel)
        + float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_CENTER_AFTER_FIRST_NEXT_VOWEL_MS)
    )
    for pair_index, index in enumerate(range(start + 1, end, 2)):
        center = first_center + (float(pair_index) * float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_STEP_MS))
        candidates.append((index, center - float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_VC_CENTER_LEAD_MS), "vc"))
        candidates.append((index + 1, center + float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_CV_CENTER_LAG_MS), "cv"))

    shifts = [candidate_offset - float(rows[index].timing.offset) for index, candidate_offset, _role in candidates]
    if min(shifts) < -1e-6:
        return
    if shifts[0] < float(JAPANESE_CVVC_INITIAL_V_REGULAR_PAIR_MIN_V_FORWARD_MS):
        return
    if max(shifts[1:] or [0.0]) < float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_MIN_FORWARD_MS):
        return

    rows[start] = _replace_cvvc_pure_vowel_onset_v_timing(
        rows[start],
        first_v_offset,
        active_start=float(first_pair_next_vowel),
    )
    for index, candidate_offset, role in candidates[1:]:
        rows[index] = _replace_cvvc_regular_pair_onset_timing(
            rows[index],
            candidate_offset,
            reference=float(first_pair_next_vowel),
        )


def _repair_cvvc_regular_pair_onset_sequence_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_regular_pair_onset_sequence_segment(out, config, segment_start, segment_end)
        segment_start = segment_end
    return out


def _repair_cvvc_regular_pair_onset_sequence_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> None:
    length = end - start
    if length < 4:
        return
    wav = str(rows[start].wav or "").strip()
    if not Path(wav).name.startswith("_"):
        return
    if any(_is_terminal_silence_alias(rows[index].alias) for index in range(start, end)):
        return
    pairs = _cvvc_regular_pair_onset_pairs(rows, config, start, end)
    if len(pairs) < 2:
        return

    first_next_vowel = _cvvc_first_next_vowel_abs(
        [rows[index] for pair in pairs for index in pair]
    )
    if first_next_vowel is None or first_next_vowel < float(JAPANESE_CVVC_PURE_VOWEL_ONSET_MIN_START_MS):
        return

    first_center = (
        float(first_next_vowel)
        + float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_CENTER_AFTER_FIRST_NEXT_VOWEL_MS)
    )
    for pair_index, (vc_index, cv_index) in enumerate(pairs):
        center = first_center + (float(pair_index) * float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_STEP_MS))
        candidates = (
            center - float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_VC_CENTER_LEAD_MS),
            center + float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_CV_CENTER_LAG_MS),
        )
        shifts = (
            candidates[0] - float(rows[vc_index].timing.offset),
            candidates[1] - float(rows[cv_index].timing.offset),
        )
        if min(shifts) < -1e-6 or max(shifts) < float(JAPANESE_CVVC_REGULAR_PAIR_ONSET_MIN_FORWARD_MS):
            continue
        rows[vc_index] = _replace_cvvc_regular_pair_onset_timing(
            rows[vc_index],
            candidates[0],
            reference=float(first_next_vowel),
        )
        rows[cv_index] = _replace_cvvc_regular_pair_onset_timing(
            rows[cv_index],
            candidates[1],
            reference=float(first_next_vowel),
        )


def _cvvc_regular_pair_onset_pairs(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    strict_pairs = _cvvc_strict_regular_pair_onset_pairs(rows, config, start, end)
    if strict_pairs:
        return strict_pairs
    if not any(_alias_has_attached_pitch_suffix(rows[index].alias) for index in range(start, end)):
        return []
    if _alias_type_for_row(rows[start].alias, config.alias_type) != "vc":
        return []
    pairs: list[tuple[int, int]] = []
    index = start
    while index < end - 1:
        if _alias_type_for_row(rows[index].alias, config.alias_type) != "vc":
            break
        cv_index = index + 1
        if (
            cv_index + 1 < end
            and _is_cvvc_terminal_release_r_alias(rows[cv_index].alias)
            and _alias_type_for_row(rows[cv_index + 1].alias, config.alias_type) == "cv"
        ):
            pairs.append((index, cv_index + 1))
            index = cv_index + 2
            continue
        if _alias_type_for_row(rows[cv_index].alias, config.alias_type) != "cv":
            break
        pairs.append((index, cv_index))
        index = cv_index + 1
    return pairs


def _cvvc_strict_regular_pair_onset_pairs(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    length = end - start
    if length < 4 or length % 2 != 0:
        return []
    pairs: list[tuple[int, int]] = []
    for index in range(start, end, 2):
        if _alias_type_for_row(rows[index].alias, config.alias_type) != "vc":
            return []
        cv_index = index + 1
        if _alias_type_for_row(rows[cv_index].alias, config.alias_type) != "cv":
            return []
        pairs.append((index, cv_index))
    return pairs


def _replace_cvvc_regular_pair_onset_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    reference: float,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    shift = max(0.0, float(candidate_offset) - current_offset)
    cutoff_abs = max(float(row.timing.consonant) + 8.0, abs(float(row.timing.cutoff)) - shift)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(row.timing.consonant),
        cutoff=-float(cutoff_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_regular_pair_onset_repair:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_regular_pair_onset_reference:{reference:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_regular_pair_onset_repair"))),
    )


def _repair_cvvc_grouped_vc_first_offset_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    segment_start = 0
    while segment_start < len(out):
        wav_key = str(out[segment_start].wav or "").strip().lower()
        segment_end = segment_start + 1
        while segment_end < len(out) and str(out[segment_end].wav or "").strip().lower() == wav_key:
            segment_end += 1
        _repair_cvvc_grouped_vc_first_offset_segment(out, config, segment_start, segment_end)
        segment_start = segment_end
    return out


def _repair_cvvc_grouped_vc_first_offset_segment(
    rows: list[AdaptedOtoRow],
    config: OtoAdapterConfig,
    start: int,
    end: int,
) -> None:
    length = end - start
    if length < 4:
        return
    first_cv_index = start
    while first_cv_index < end and _alias_type_for_row(rows[first_cv_index].alias, config.alias_type) == "vc":
        first_cv_index += 1
    vc_count = first_cv_index - start
    if vc_count < 2 or first_cv_index >= end:
        return
    cv_count = 0
    scan = first_cv_index
    while scan < end and _alias_type_for_row(rows[scan].alias, config.alias_type) == "cv":
        cv_count += 1
        scan += 1
    if cv_count < 2:
        return
    first_vc = rows[start]
    first_cv = rows[first_cv_index]
    right_token = _cvvc_vc_alias_right_token(first_vc.alias)
    if not right_token:
        return
    for index in range(start + 1, first_cv_index):
        if _cvvc_vc_alias_right_token(rows[index].alias) != right_token:
            return
    if not _cvvc_vc_right_matches_cv_onset(first_vc.alias, first_cv.alias):
        cv_phones = tuple(_alias_phone_sequence(first_cv.alias))
        if cv_phones or _alias_type_for_row(first_cv.alias, config.alias_type) != "cv":
            return

    vc_offset = float(first_vc.timing.offset)
    cv_offset = float(first_cv.timing.offset)
    gap = cv_offset - vc_offset
    candidate_offset = cv_offset - float(JAPANESE_CVVC_GROUPED_VC_FIRST_OFFSET_FROM_CV_MS)
    shift = candidate_offset - vc_offset
    if not all(np.isfinite(value) for value in (vc_offset, cv_offset, gap, candidate_offset, shift)):
        return
    if gap <= float(JAPANESE_CVVC_GROUPED_VC_FIRST_MIN_GAP_TO_CV_MS):
        return
    if shift <= float(JAPANESE_CVVC_GROUPED_VC_FIRST_MIN_FORWARD_MS):
        return
    if candidate_offset < 0.0 or candidate_offset >= cv_offset - 1e-6:
        return

    rows[start] = _replace_cvvc_grouped_vc_first_offset(
        first_vc,
        candidate_offset,
        reference_alias=first_cv.alias,
        gap=gap,
    )


def _cvvc_vc_right_matches_cv_onset(vc_alias: str, cv_alias: str) -> bool:
    right_token = _cvvc_vc_alias_right_token(vc_alias)
    right_phones = tuple(_alias_phone_sequence(right_token))
    cv_phones = tuple(_alias_phone_sequence(cv_alias))
    if not right_phones or not cv_phones or len(cv_phones) <= len(right_phones):
        return False
    if all(is_vowel_phone(phone) for phone in right_phones):
        return False
    return cv_phones[: len(right_phones)] == right_phones


def _replace_cvvc_grouped_vc_first_offset(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    reference_alias: str,
    gap: float,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(row.timing.consonant),
        cutoff=float(row.timing.cutoff),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_grouped_vc_first_offset_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_grouped_vc_first_offset_reference:{reference_alias}",
                    f"cvvc_grouped_vc_first_offset_gap:{gap:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_grouped_vc_first_offset_repair"))),
    )


def _repair_cvvc_vc_cutoff_order_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out: list[AdaptedOtoRow] = []
    for row in rows:
        if _alias_type_for_row(row.alias, config.alias_type) != "vc":
            out.append(row)
            continue
        cutoff = float(row.timing.cutoff)
        consonant = float(row.timing.consonant)
        if not np.isfinite(cutoff) or not np.isfinite(consonant) or cutoff >= 0.0:
            out.append(row)
            continue
        current_tail = abs(cutoff) - consonant
        if current_tail > float(JAPANESE_CVVC_VC_CUTOFF_ORDER_TRIGGER_TAIL_MS):
            out.append(row)
            continue
        candidate_cutoff_abs = max(
            abs(cutoff),
            consonant + float(JAPANESE_CVVC_VC_CUTOFF_ORDER_MIN_TAIL_MS),
        )
        out.append(
            _replace_cvvc_vc_cutoff_order(
                row,
                candidate_cutoff_abs,
                current_tail=current_tail,
            )
        )
    return out


def _replace_cvvc_vc_cutoff_order(
    row: AdaptedOtoRow,
    candidate_cutoff_abs: float,
    *,
    current_tail: float,
) -> AdaptedOtoRow:
    current_cutoff = float(row.timing.cutoff)
    timing = OtoTiming(
        offset=float(row.timing.offset),
        consonant=float(row.timing.consonant),
        cutoff=-float(candidate_cutoff_abs),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_vc_cutoff_order_repaired:{current_cutoff:.1f}->{timing.cutoff:.1f}",
                    f"cvvc_vc_cutoff_order_tail:{current_tail:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_vc_cutoff_order_repair"))),
    )


def _repair_cvvc_terminal_release_r_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
    *,
    file_duration_ms: float | None,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    duration = float(file_duration_ms or 0.0)
    if not np.isfinite(duration) or duration <= 0.0:
        return list(rows)
    out = list(rows)
    terminal_by_wav_and_left: dict[tuple[str, str], AdaptedOtoRow] = {}
    for terminal in out:
        if not _is_terminal_silence_alias(terminal.alias):
            continue
        terminal_parts = [part.strip() for part in str(terminal.alias or "").split() if part.strip()]
        if len(terminal_parts) < 2 or terminal_parts[-1] != "-":
            continue
        terminal_phones = _alias_phone_sequence(terminal_parts[0])
        if len(terminal_phones) != 1:
            continue
        wav_key = str(terminal.wav or "").strip().lower()
        terminal_by_wav_and_left[(wav_key, str(terminal_phones[0]).strip().lower())] = terminal

    for index, row in enumerate(out):
        if not _is_cvvc_terminal_release_r_alias(row.alias):
            continue
        wav_name = Path(str(row.wav or "")).name
        wav_key = wav_name.lower()
        current_offset = float(row.timing.offset)
        if not np.isfinite(current_offset):
            continue
        left_phones = _cvvc_terminal_release_r_left_phones(row.alias)
        terminal = (
            terminal_by_wav_and_left.get((str(row.wav or "").strip().lower(), str(left_phones[0]).strip().lower()))
            if len(left_phones) == 1
            else None
        )
        if terminal is not None:
            terminal_offset = float(terminal.timing.offset)
            if (
                np.isfinite(terminal_offset)
                and terminal_offset - current_offset >= float(JAPANESE_CVVC_TERMINAL_RELEASE_R_MIN_FORWARD_MS)
            ):
                out[index] = _copy_cvvc_terminal_release_r_terminal_timing(row, terminal)
                continue
        if wav_name.startswith("_") and duration >= 5000.0:
            candidate_offset = max(
                0.0,
                duration - float(JAPANESE_CVVC_TERMINAL_RELEASE_R_TAIL_MS),
            )
            if candidate_offset - current_offset < float(JAPANESE_CVVC_TERMINAL_RELEASE_R_MIN_FORWARD_MS):
                continue
            out[index] = _replace_cvvc_terminal_release_r_timing(
                row,
                candidate_offset,
                profile="long_cvvc_tail",
                rule_name="cvvc_terminal_release_r_tail_repair",
            )
            continue
        if _is_cvvc_standalone_release_r_wav(wav_key):
            candidate_offset = max(
                0.0,
                duration * float(JAPANESE_CVVC_STANDALONE_RELEASE_R_OFFSET_RATIO),
            )
            if candidate_offset - current_offset < float(JAPANESE_CVVC_STANDALONE_RELEASE_R_MIN_FORWARD_MS):
                continue
            out[index] = _replace_cvvc_terminal_release_r_timing(
                row,
                candidate_offset,
                profile="standalone_midpoint",
                rule_name="cvvc_standalone_release_r_midpoint_repair",
            )
    return out


def _is_cvvc_standalone_release_r_wav(wav_name_lower: str) -> bool:
    return bool(re.fullmatch(r"[aeioun]r\.wav", str(wav_name_lower or "").strip().lower()))


def _cvvc_terminal_release_r_left_phones(alias: str) -> tuple[str, ...]:
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2 or parts[-1] != "R":
        return ()
    return tuple(str(phone).strip().lower() for phone in _alias_phone_sequence(parts[0]))


def _copy_cvvc_terminal_release_r_terminal_timing(
    row: AdaptedOtoRow,
    terminal: AdaptedOtoRow,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    timing = OtoTiming(
        offset=float(terminal.timing.offset),
        consonant=float(terminal.timing.consonant),
        cutoff=float(terminal.timing.cutoff),
        preutterance=float(terminal.timing.preutterance),
        overlap=float(terminal.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_terminal_release_r_terminal_copy_repaired:{current_offset:.1f}->{float(timing.offset):.1f}",
                    f"cvvc_terminal_release_r_terminal_reference:{terminal.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(
            dict.fromkeys((*row.applied_rules, "cvvc_terminal_release_r_terminal_copy_repair"))
        ),
    )


def _replace_cvvc_terminal_release_r_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    *,
    profile: str,
    rule_name: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    if not np.isfinite(current_offset) or abs(current_offset - float(candidate_offset)) <= 1e-6:
        return row
    if profile == "long_cvvc_tail":
        timing = OtoTiming(
            offset=float(candidate_offset),
            consonant=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CONSONANT_MS),
            cutoff=-float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_CUTOFF_MS),
            preutterance=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_PRE_MS),
            overlap=float(JAPANESE_CVVC_PURE_VOWEL_ONSET_VV_OVERLAP_MS),
        )
    else:
        timing = OtoTiming(
            offset=float(candidate_offset),
            consonant=float(row.timing.consonant),
            cutoff=float(row.timing.cutoff),
            preutterance=float(row.timing.preutterance),
            overlap=float(row.timing.overlap),
        )
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"{rule_name}:{current_offset:.1f}->{float(candidate_offset):.1f}",
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, rule_name))),
    )


def _repair_cvvc_terminal_standalone_vowel_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    if not _is_japanese_cvvc_config(config):
        return list(rows)
    out = list(rows)
    terminals: dict[str, AdaptedOtoRow] = {}
    for row in out:
        wav_key = str(row.wav or "").strip().lower()
        if not wav_key or not Path(str(row.wav or "")).name.startswith("_"):
            continue
        if _is_terminal_silence_alias(row.alias):
            terminals[wav_key] = row
    if not terminals:
        return out

    for index, row in enumerate(out):
        wav_key = str(row.wav or "").strip().lower()
        terminal = terminals.get(wav_key)
        if terminal is None:
            continue
        if _alias_type_for_row(row.alias, config.alias_type) != "v":
            continue
        if not _cvvc_standalone_vowel_matches_terminal(row, terminal):
            continue
        out[index] = _replace_cvvc_terminal_standalone_vowel_timing(row, terminal)
    return out


def _cvvc_standalone_vowel_matches_terminal(row: AdaptedOtoRow, terminal: AdaptedOtoRow) -> bool:
    row_phones = _alias_phone_sequence(row.alias)
    terminal_parts = [part.strip() for part in str(terminal.alias or "").split() if part.strip()]
    if len(row_phones) != 1 or len(terminal_parts) < 2 or terminal_parts[-1] != "-":
        return False
    terminal_phones = _alias_phone_sequence(terminal_parts[0])
    return len(terminal_phones) == 1 and row_phones[0] == terminal_phones[0]


def _replace_cvvc_terminal_standalone_vowel_timing(
    row: AdaptedOtoRow,
    terminal: AdaptedOtoRow,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    terminal_offset = float(terminal.timing.offset)
    if not np.isfinite(terminal_offset):
        return row
    candidate_offset = max(
        0.0,
        terminal_offset + float(JAPANESE_CVVC_TERMINAL_STANDALONE_V_OFFSET_AFTER_TERMINAL_MS),
    )
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(JAPANESE_CVVC_TERMINAL_STANDALONE_V_CONSONANT_MS),
        cutoff=-float(JAPANESE_CVVC_TERMINAL_STANDALONE_V_CUTOFF_MS),
        preutterance=float(JAPANESE_CVVC_TERMINAL_STANDALONE_V_PRE_MS),
        overlap=float(JAPANESE_CVVC_TERMINAL_STANDALONE_V_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_terminal_standalone_v_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_terminal_standalone_v_reference:{terminal.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_terminal_standalone_v_repair"))),
    )


def _repair_cvvc_post_yoon_vc_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        previous = out[index - 1] if index > 0 else None
        out[index] = _repair_cvvc_post_yoon_vc_row(previous, row, config)
    return out


def _repair_cvvc_post_yoon_vc_row(
    previous: AdaptedOtoRow | None,
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if previous is None:
        return row
    if str(previous.wav or "").strip().lower() != str(row.wav or "").strip().lower():
        return row
    if "cvvc_following_yoon_repair" not in previous.applied_rules:
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "vc":
        return row
    right_token = _cvvc_vc_alias_right_token(row.alias)
    if right_token not in JAPANESE_CVVC_FOLLOWING_YOON_RIGHT_TOKENS and right_token not in JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS:
        return row
    if not _cvvc_following_row_starts_with_token(previous.alias, right_token):
        return row

    left_token = _cvvc_vc_alias_left_token(row.alias)
    offset_from_cv = (
        float(JAPANESE_CVVC_POST_E_YOON_VC_OFFSET_FROM_CV_MS)
        if left_token == "e" and right_token in JAPANESE_CVVC_FOLLOWING_YOON_RIGHT_TOKENS
        else float(JAPANESE_CVVC_POST_YOON_VC_OFFSET_FROM_CV_MS)
    )
    candidate_offset = max(0.0, float(previous.timing.offset) + offset_from_cv)
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (candidate_offset, current_offset)):
        return row
    if current_offset - candidate_offset <= float(JAPANESE_CVVC_POST_YOON_VC_REPAIR_MIN_LATE_MS):
        return row

    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    if right_token in JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS:
        cutoff_ms = max(float(JAPANESE_CVVC_VC_SEQUENCE_CUTOFF_MS), current_end - candidate_offset)
    else:
        cutoff_ms = float(JAPANESE_CVVC_VC_SEQUENCE_CUTOFF_MS)
    timing = OtoTiming(
        offset=candidate_offset,
        consonant=float(JAPANESE_CVVC_VC_SEQUENCE_CONSONANT_MS),
        cutoff=-float(cutoff_ms),
        preutterance=float(JAPANESE_CVVC_VC_SEQUENCE_PRE_MS),
        overlap=float(JAPANESE_CVVC_VC_SEQUENCE_OVERLAP_MS),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_post_yoon_vc_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"cvvc_post_yoon_vc_offset_from_cv:{offset_from_cv:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_post_yoon_vc_repair"))),
    )


def _repair_cvvc_following_glide_row(
    previous: AdaptedOtoRow | None,
    row: AdaptedOtoRow,
    next_vc: AdaptedOtoRow | None,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if previous is None:
        return row
    if str(previous.wav or "").strip().lower() != str(row.wav or "").strip().lower():
        return row
    role = _alias_type_for_row(row.alias, config.alias_type)
    if role not in {"cv", "vcv"}:
        return row
    phones = tuple(_alias_phone_sequence(row.alias))

    previous_following_rule = next(
        (rule for rule in JAPANESE_CVVC_FOLLOWING_REPAIR_RULES if rule in previous.applied_rules),
        "",
    )
    if previous_following_rule and tuple(_alias_phone_sequence(previous.alias)) == phones:
        return _copy_cvvc_following_glide_timing(previous, row, rule_name=previous_following_rule)
    if (
        "cvvc_vc_sequence_repair" not in previous.applied_rules
        and "cvvc_sonorant_yoon_vc_repair" not in previous.applied_rules
        and "cvvc_post_yoon_vc_repair" not in previous.applied_rules
    ):
        return row

    repair_spec = _cvvc_following_row_repair_spec(previous, row, next_vc, phones)
    if repair_spec is None:
        return row
    (
        candidate_offset,
        candidate_end,
        preutterance_ms,
        overlap_ms,
        consonant_ms,
        rule_name,
        warning_prefix,
    ) = repair_spec
    previous_absolute = absolute_oto_positions(previous.timing)
    previous_pre = float(previous_absolute["preutterance_abs"])
    current_offset = float(row.timing.offset)
    if not all(np.isfinite(value) for value in (previous_pre, current_offset, candidate_offset, candidate_end)):
        return row
    min_end = candidate_offset + float(consonant_ms) + 8.0
    if candidate_end <= min_end:
        return row
    if current_offset - candidate_offset <= float(JAPANESE_CVVC_VC_SEQUENCE_REPAIR_MIN_LATE_MS):
        return row

    return _replace_cvvc_following_glide_timing(
        row,
        candidate_offset,
        candidate_end,
        preutterance_ms=preutterance_ms,
        overlap_ms=overlap_ms,
        consonant_ms=consonant_ms,
        rule_name=rule_name,
        warning_prefix=warning_prefix,
    )


def _cvvc_following_row_repair_spec(
    previous: AdaptedOtoRow,
    row: AdaptedOtoRow,
    next_vc: AdaptedOtoRow | None,
    phones: tuple[str, ...],
) -> tuple[float, float, float, float, float, str, str] | None:
    previous_pre = float(absolute_oto_positions(previous.timing)["preutterance_abs"])
    right_token = _cvvc_vc_alias_right_token(previous.alias)
    if right_token == JAPANESE_CVVC_FOLLOWING_GLIDE_PHONES[0] and phones == JAPANESE_CVVC_FOLLOWING_GLIDE_PHONES:
        if next_vc is None:
            return None
        return (
            max(0.0, previous_pre + float(JAPANESE_CVVC_FOLLOWING_GLIDE_OFFSET_AFTER_PRE_MS)),
            _cvvc_row_anchor_or_pre_abs(next_vc),
            float(JAPANESE_CVVC_FOLLOWING_GLIDE_PRE_MS),
            float(JAPANESE_CVVC_FOLLOWING_GLIDE_OVERLAP_MS),
            float(JAPANESE_CVVC_FOLLOWING_GLIDE_CONSONANT_MS),
            "cvvc_following_glide_repair",
            "cvvc_following_glide",
        )
    if right_token in JAPANESE_CVVC_FOLLOWING_YOON_RIGHT_TOKENS or right_token in JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS:
        right_phones = tuple(_alias_phone_sequence(right_token))
        if not right_phones or len(phones) <= len(right_phones) or phones[: len(right_phones)] != right_phones:
            return None
        if right_token in JAPANESE_CVVC_SONORANT_YOON_RIGHT_TOKENS:
            return (
                max(0.0, previous_pre),
                float(absolute_oto_positions(row.timing)["cutoff_abs"]),
                float(JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_PRE_MS),
                float(JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_OVERLAP_MS),
                float(JAPANESE_CVVC_FOLLOWING_SONORANT_YOON_CONSONANT_MS),
                "cvvc_following_yoon_repair",
                "cvvc_following_yoon",
            )
        return (
            max(0.0, previous_pre),
            float(absolute_oto_positions(row.timing)["cutoff_abs"]),
            float(JAPANESE_CVVC_FOLLOWING_YOON_PRE_MS),
            float(JAPANESE_CVVC_FOLLOWING_YOON_OVERLAP_MS),
            float(JAPANESE_CVVC_FOLLOWING_YOON_CONSONANT_MS),
            "cvvc_following_yoon_repair",
            "cvvc_following_yoon",
        )
    return None


def _cvvc_following_row_starts_with_token(alias: str, right_token: str) -> bool:
    right_phones = tuple(_alias_phone_sequence(right_token))
    phones = tuple(_alias_phone_sequence(alias))
    return bool(right_phones and len(phones) > len(right_phones) and phones[: len(right_phones)] == right_phones)


def _copy_cvvc_following_glide_timing(
    previous: AdaptedOtoRow,
    row: AdaptedOtoRow,
    *,
    rule_name: str,
) -> AdaptedOtoRow:
    if abs(float(row.timing.offset) - float(previous.timing.offset)) <= 1e-6:
        return row
    previous_end = absolute_oto_positions(previous.timing)["cutoff_abs"]
    warning_prefix = "cvvc_following_yoon" if rule_name == "cvvc_following_yoon_repair" else "cvvc_following_glide"
    return replace(
        row,
        timing=previous.timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"{warning_prefix}_copied:{float(row.timing.offset):.1f}->{float(previous.timing.offset):.1f}",
                    f"{warning_prefix}_end:{absolute_oto_positions(row.timing)['cutoff_abs']:.1f}->{float(previous_end):.1f}",
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, rule_name))),
    )


def _replace_cvvc_following_glide_timing(
    row: AdaptedOtoRow,
    candidate_offset: float,
    candidate_end: float,
    *,
    preutterance_ms: float,
    overlap_ms: float,
    consonant_ms: float,
    rule_name: str,
    warning_prefix: str,
) -> AdaptedOtoRow:
    current_offset = float(row.timing.offset)
    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    timing = OtoTiming(
        offset=float(candidate_offset),
        consonant=float(consonant_ms),
        cutoff=-max(0.0, float(candidate_end) - float(candidate_offset)),
        preutterance=float(preutterance_ms),
        overlap=float(overlap_ms),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"{warning_prefix}_repaired:{current_offset:.1f}->{candidate_offset:.1f}",
                    f"{warning_prefix}_end:{current_end:.1f}->{candidate_end:.1f}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, rule_name))),
    )


def _next_cvvc_vc_row(
    rows: Sequence[AdaptedOtoRow],
    start_index: int,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow | None:
    current_wav = str(rows[start_index].wav or "").strip().lower()
    for row in rows[start_index + 1 :]:
        if str(row.wav or "").strip().lower() != current_wav:
            return None
        if _alias_type_for_row(row.alias, config.alias_type) == "vc":
            return row
    return None


def _cvvc_row_anchor_or_pre_abs(row: AdaptedOtoRow) -> float:
    if row.anchor is not None:
        anchor_abs = float(row.anchor.anchor_abs_ms)
        if np.isfinite(anchor_abs) and anchor_abs > 0.0:
            return anchor_abs
    return float(absolute_oto_positions(row.timing)["preutterance_abs"])


def _cvvc_vc_alias_right_token(alias: str) -> str:
    parts = [part.strip().lower() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return ""
    return parts[1]


def _cvvc_vc_alias_left_token(alias: str) -> str:
    parts = [part.strip().lower() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return ""
    return parts[0]


def _cvvc_initial_cv_head_vowel_token(alias: str) -> str:
    normalized = str(alias or "").strip().lower()
    if not normalized.startswith("-"):
        return ""
    parts = [part.strip().lower() for part in normalized.split() if part.strip() and part.strip() != "-"]
    if not parts:
        compact = normalized[1:].strip()
        if compact:
            parts = [compact]
    if len(parts) != 1:
        return ""
    phones = _alias_phones(parts[0])
    if len(phones) == 1 and is_vowel_phone(phones[0]):
        return str(phones[0]).strip().lower()
    return ""


def _repair_cvvc_initial_spaced_cv_rows(
    rows: Sequence[AdaptedOtoRow],
    config: OtoAdapterConfig,
) -> list[AdaptedOtoRow]:
    out = list(rows)
    for index, row in enumerate(out):
        previous = out[index - 1] if index > 0 else None
        out[index] = _repair_cvvc_initial_spaced_cv_row(previous, row, config)
    return out


def _repair_cvvc_initial_spaced_cv_row(
    previous: AdaptedOtoRow | None,
    row: AdaptedOtoRow,
    config: OtoAdapterConfig,
) -> AdaptedOtoRow:
    if previous is None:
        return row
    if str(previous.wav or "").strip().lower() != str(row.wav or "").strip().lower():
        return row
    if " " not in str(row.alias or ""):
        return row
    if _alias_type_for_row(previous.alias, config.alias_type) != "cv":
        return row
    if _alias_type_for_row(row.alias, config.alias_type) != "vcv":
        return row
    if tuple(_alias_phone_sequence(previous.alias)) != tuple(_alias_phone_sequence(row.alias)):
        return row
    if not any(warning.startswith("vcv_previous_vowel_end_clamped:") for warning in row.warnings):
        return row

    current_offset = float(row.timing.offset)
    reference_offset = float(previous.timing.offset)
    if not all(np.isfinite(value) for value in (current_offset, reference_offset)):
        return row
    if current_offset > float(JAPANESE_CVVC_INITIAL_SPACED_CV_OFFSET_REPAIR_EDGE_MS):
        return row
    if reference_offset - current_offset <= float(JAPANESE_CVVC_INITIAL_SPACED_CV_OFFSET_REPAIR_MIN_EARLY_MS):
        return row

    current_end = float(absolute_oto_positions(row.timing)["cutoff_abs"])
    candidate_end = max(current_end, reference_offset + float(row.timing.consonant) + 8.0)
    timing = OtoTiming(
        offset=reference_offset,
        consonant=float(row.timing.consonant),
        cutoff=-max(0.0, candidate_end - reference_offset),
        preutterance=float(row.timing.preutterance),
        overlap=float(row.timing.overlap),
    )
    timing, timing_warnings = _validate_timing_with_warnings(timing, file_duration_ms=0.0)
    return replace(
        row,
        timing=timing,
        warnings=tuple(
            dict.fromkeys(
                (
                    *row.warnings,
                    f"cvvc_initial_spaced_cv_offset_repaired:{current_offset:.1f}->{reference_offset:.1f}",
                    f"cvvc_initial_spaced_cv_reference:{previous.alias}",
                    *timing_warnings,
                )
            )
        ),
        applied_rules=tuple(dict.fromkeys((*row.applied_rules, "cvvc_initial_spaced_cv_offset_repair"))),
    )


def _is_japanese_cvvc_config(config: OtoAdapterConfig) -> bool:
    return _is_cvvc_format(config) and str(config.language or "").strip().lower() == "japanese"


def _is_cvvc_format(config: OtoAdapterConfig) -> bool:
    return str(config.format_type or "").strip().lower() == "cvvc"


def _bootstrap_cutoff_trim_ms(
    config: OtoAdapterConfig,
    alias_type: str,
    raw_trim_ms: float,
    *,
    min_trim_ms: float,
    file_duration_ms: float,
) -> tuple[float, tuple[str, ...]]:
    trim = max(float(min_trim_ms), float(raw_trim_ms))
    cap = _bootstrap_cutoff_trim_cap_ms(config, alias_type, file_duration_ms=file_duration_ms)
    if cap is None:
        return trim, ()
    clamped = max(float(min_trim_ms), float(cap))
    if trim <= clamped + 1e-6:
        return trim, ()
    return clamped, (f"bootstrap_cutoff_tail_clamped:{trim:.1f}->{clamped:.1f}",)


def _bootstrap_cutoff_trim_cap_ms(
    config: OtoAdapterConfig,
    alias_type: str,
    *,
    file_duration_ms: float,
) -> float | None:
    if str(config.format_type or "").strip().lower() != "vcv":
        return None
    if str(alias_type or "").strip().lower() not in {"cv_head", "vcv", "vv", "v"}:
        return None
    duration_cap = max(0.0, float(file_duration_ms)) * 0.35 if file_duration_ms > 0.0 else 0.0
    language = str(config.language or "").strip().lower()
    if language == "japanese":
        base = JAPANESE_VCV_CUTOFF_TRIM_CAP_MS
    elif language == "korean":
        base = KOREAN_VCV_CUTOFF_TRIM_CAP_MS
    else:
        base = 900.0
    if duration_cap > 0.0:
        return max(240.0, min(float(base), duration_cap))
    return float(base)


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
    validated, _warnings = _validate_timing_with_warnings(timing, file_duration_ms=file_duration_ms)
    return validated


def _validate_timing_with_warnings(timing: OtoTiming, *, file_duration_ms: float) -> tuple[OtoTiming, tuple[str, ...]]:
    raw_offset = float(timing.offset)
    raw_pre = float(timing.preutterance)
    raw_overlap = float(timing.overlap)
    raw_consonant = float(timing.consonant)
    raw_cutoff = float(timing.cutoff)
    raw_cutoff_abs = abs(raw_cutoff)

    offset = max(raw_offset, 0.0)
    pre = max(raw_pre, 0.0)
    ovl = max(0.0, min(raw_overlap, pre))
    consonant = max(raw_consonant, pre + 8.0)
    cutoff_abs = max(raw_cutoff_abs, consonant + 8.0)
    if file_duration_ms > 0.0:
        cutoff_abs = min(max(cutoff_abs, consonant + 8.0), max(consonant + 8.0, file_duration_ms - offset + 80.0))
    validated = OtoTiming(offset=offset, consonant=consonant, cutoff=-cutoff_abs, preutterance=pre, overlap=ovl)

    warnings: list[str] = []
    deltas: list[float] = []
    for field, raw_value, clamped_value in (
        ("offset", raw_offset, offset),
        ("preutterance", raw_pre, pre),
        ("overlap", raw_overlap, ovl),
        ("consonant", raw_consonant, consonant),
        ("cutoff", raw_cutoff_abs, cutoff_abs),
    ):
        delta = abs(float(clamped_value) - float(raw_value))
        if delta > 1e-6:
            warnings.append(f"timing_clamped.{field}")
            deltas.append(delta)
    if raw_cutoff > 0.0:
        warnings.append("timing_clamped.cutoff_sign")
    if deltas:
        max_delta = max(deltas)
        warnings.append(f"timing_clamp_delta_ms:{max_delta:.1f}")
        if max_delta >= TIMING_CLAMP_LARGE_DELTA_MS:
            warnings.append("timing_clamp_large_delta")
    return validated, tuple(dict.fromkeys(warnings))


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
        if _alias_token_is_bare_n(first) and _alias_token_is_non_n_vowel(second):
            return "vcv"
        first_vowel = bool(first and is_vowel_phone(first[-1]))
        if first_vowel and second:
            if _alias_token_is_bare_n(second):
                return "vc"
            if len(second) == 1 and is_vowel_phone(second[0]):
                return "vv"
            return "vcv" if is_vowel_phone(second[-1]) else "vc"
        return "vcv"
    if len(phones) == 1 and is_vowel_phone(phones[0]):
        return "v"
    if len(phones) >= 2 and not is_vowel_phone(phones[0]) and is_vowel_phone(phones[-1]):
        return "cv"
    return "cv"


def _contextual_alias_role(
    alias: str,
    role: str,
    expected_phones: Sequence[str],
    target_phone_index: int | None,
) -> str:
    normalized_role = str(role or "").strip().lower()
    if target_phone_index is None:
        return normalized_role
    expected = [str(phone or "").strip().lower() for phone in expected_phones]
    target = int(target_phone_index)
    if normalized_role == "v" and _alias_is_wo_kana(alias):
        if 0 < target < len(expected) and expected[target - 1] == "w":
            return "cv"
        return normalized_role
    if normalized_role != "vc":
        return normalized_role
    if not _space_alias_has_bare_n_right(alias):
        return normalized_role
    if target < 0 or target >= len(expected) or expected[target] != "n":
        return normalized_role
    if target + 1 >= len(expected):
        return "vv"
    next_phone = expected[target + 1]
    if next_phone == "y" or is_vowel_phone(next_phone):
        return normalized_role
    return "vv"


def _alias_is_wo_kana(alias: str) -> bool:
    return str(alias or "").strip() in {"\u3092", "\u30f2"}


def _space_alias_has_bare_n_right(alias: str) -> bool:
    tokens = [part for part in str(alias or "").strip().lower().split() if part]
    if len(tokens) < 2:
        return False
    return _alias_token_is_bare_n(_alias_phones(tokens[1]))


def _alias_token_is_bare_n(phones: Sequence[str]) -> bool:
    normalized = [str(phone or "").strip().lower() for phone in phones if str(phone or "").strip()]
    return normalized == ["n"]


def _alias_token_is_non_n_vowel(phones: Sequence[str]) -> bool:
    normalized = [str(phone or "").strip().lower() for phone in phones if str(phone or "").strip()]
    return len(normalized) == 1 and normalized[0] != "n" and is_vowel_phone(normalized[0])


def _cv_head_phones_are_consonant_like(phones: Sequence[str]) -> bool:
    normalized = tuple(str(phone or "").strip().lower() for phone in phones if str(phone or "").strip())
    if not normalized:
        return False
    if normalized == ("n",):
        return True
    if normalized[0] == "n" and len(normalized) >= 2 and normalized[1] == "y":
        return True
    return not any(is_vowel_phone(phone) for phone in normalized)


def _alias_phones(text: str) -> list[str]:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return []
    normalized = _strip_alias_attached_pitch_suffix_token(normalized)
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
    vowels = {
        "yeo",
        "yae",
        "wae",
        "weo",
        "eo",
        "eu",
        "ae",
        "ya",
        "ye",
        "yo",
        "yu",
        "wa",
        "wo",
        "we",
        "wi",
        "ui",
        "a",
        "i",
        "u",
        "e",
        "o",
        "n",
    }
    vowel_units = tuple(sorted(vowels, key=len, reverse=True))
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
        vowel = next((item for item in vowel_units if normalized.startswith(item, pos)), None)
        if vowel:
            phones.append(vowel)
            pos += len(vowel)
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


def _strip_alias_attached_pitch_suffix_token(token: str) -> str:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return ""
    match = ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(normalized)
    if match is None:
        return normalized
    if match.start() <= 0:
        return ""
    stripped = normalized[: match.start()].rstrip("_- ").strip()
    return stripped or normalized


def _alias_has_attached_pitch_suffix(alias: str) -> bool:
    value = str(alias or "").strip()
    return bool(value and ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value))


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
    if role == "vv":
        return ("vv_boundary", "cv_boundary", "phone_change", "vowel_nucleus")
    if role in {"vcv", "v"}:
        return ("phone_change", "vowel_nucleus", "cv_boundary")
    if role == "vc":
        return ("phone_change", "cv_boundary", "vowel_nucleus")
    return ("cv_boundary", "vv_boundary", "phone_change", "vowel_nucleus")


def _preferred_event_labels_for_alias(alias: str, role: str) -> tuple[str, ...]:
    if (
        str(role or "").strip().lower() == "vcv"
        and _space_alias_starts_with_consonant_and_ends_with_vowel(alias)
    ):
        return ("cv_boundary", "phone_change", "vv_boundary")
    return _event_labels_for_alias_role(role)


def _space_alias_starts_with_consonant_and_ends_with_vowel(alias: str) -> bool:
    tokens = [part for part in str(alias or "").strip().lower().split() if part]
    if len(tokens) < 2:
        return False
    first = _alias_phones(tokens[0])
    second = _alias_phones(tokens[1])
    if not first or not second:
        return False
    return not is_vowel_phone(first[-1]) and is_vowel_phone(second[-1])


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
    targets = [candidate_steps[row_idx][cand_idx].phone_index for row_idx, cand_idx in enumerate(selected)]
    return _repair_cvvc_alias_target_sequence(rows, roles, expected, targets)


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
    roles = [_alias_type_for_row(row.alias, "auto") for row in rows]
    return _repair_cvvc_alias_target_sequence(rows, roles, expected, out)


def _repair_cvvc_alias_target_sequence(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
    expected: Sequence[str],
    targets: Sequence[int | None],
) -> list[int | None]:
    out = _repair_cvvc_initial_cv_head_targets(rows, roles, expected, targets)
    return _repair_cvvc_transition_then_cv_block_targets(rows, roles, expected, out)


def _repair_cvvc_initial_cv_head_targets(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
    expected: Sequence[str],
    targets: Sequence[int | None],
) -> list[int | None]:
    out = list(targets)
    for row_idx, row in enumerate(rows):
        if row_idx >= len(out) or row_idx >= len(roles):
            continue
        if roles[row_idx] != "cv_head":
            continue
        if not str(row.alias or "").strip().startswith("-"):
            continue
        current_target = out[row_idx]
        phones = _alias_phone_sequence(row.alias)
        first_following = _first_following_phonetic_template_row(rows, roles, row_idx)
        if first_following is None:
            continue
        follow_idx, follow_role = first_following
        follow_target = out[follow_idx] if follow_idx < len(out) else None
        if follow_target is None:
            continue
        if current_target is not None and follow_target <= current_target:
            continue
        if _cv_head_phones_are_consonant_like(phones):
            if follow_role != "vc":
                continue
            follow_phones = _alias_phone_sequence(rows[follow_idx].alias)
            if len(follow_phones) < 2:
                continue
            if not _phone_sequence_variants_match(follow_phones[1:], phones):
                continue
            out[row_idx] = follow_target
            continue
        if follow_role == "vv":
            follow_phones = _alias_phone_sequence(rows[follow_idx].alias)
            if len(follow_phones) < len(phones) or not _phone_sequence_variants_match(follow_phones[-len(phones) :], phones):
                continue
            out[row_idx] = follow_target
    return out


def _first_following_phonetic_template_row(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
    row_idx: int,
) -> tuple[int, str] | None:
    for follow_idx in range(row_idx + 1, min(len(rows), len(roles))):
        role = str(roles[follow_idx] or "").strip().lower()
        alias = str(rows[follow_idx].alias or "").strip()
        if not alias or _is_terminal_silence_alias(alias) or _is_nonphonetic_special_alias(alias):
            continue
        if role in {"cv_head", "vc", "vv", "v", "cv", "vcv"}:
            return follow_idx, role
    return None


def _repair_cvvc_transition_then_cv_block_targets(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
    expected: Sequence[str],
    targets: Sequence[int | None],
) -> list[int | None]:
    out = list(targets)
    block_start = _cvvc_following_cv_block_start(rows, roles)
    if block_start is None:
        return out
    tail_n_index = _first_moraic_n_tail_index(expected)
    block_end = _cvvc_following_cv_block_end(rows, roles, block_start)
    single_cv_tail_block = block_end == block_start + 1
    cursor = -1
    if single_cv_tail_block:
        previous_targets = [
            int(target)
            for target in out[:block_start]
            if target is not None and int(target) >= 0
        ]
        if previous_targets:
            cursor = max(previous_targets)
    for row_idx in range(block_start, min(len(rows), len(roles), len(out))):
        role = str(roles[row_idx] or "").strip().lower()
        if not _is_cvvc_following_cv_block_role(role):
            if role in {"vc", "vv", "cv_head"}:
                break
            continue
        candidates = _alias_target_candidates(rows[row_idx].alias, role, expected)
        candidate_indices = sorted(
            {
                int(candidate.phone_index)
                for candidate in candidates
                if candidate.phone_index is not None and int(candidate.phone_index) > cursor
            }
        )
        if not candidate_indices:
            continue
        if tail_n_index is not None:
            before_tail = [idx for idx in candidate_indices if idx < tail_n_index]
            if before_tail:
                candidate_indices = before_tail
        chosen = candidate_indices[0]
        out[row_idx] = chosen
        cursor = chosen
    return out


def _cvvc_following_cv_block_end(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
    block_start: int,
) -> int:
    block_end = int(block_start)
    limit = min(len(rows), len(roles))
    while block_end < limit:
        role = str(roles[block_end] or "").strip().lower()
        alias = str(rows[block_end].alias or "").strip()
        if not alias or _is_terminal_silence_alias(alias) or _is_nonphonetic_special_alias(alias):
            break
        if not _is_cvvc_following_cv_block_role(role):
            break
        block_end += 1
    return block_end


def _cvvc_following_cv_block_start(
    rows: Sequence[OtoTemplateRow],
    roles: Sequence[str],
) -> int | None:
    seen_transition = False
    for row_idx in range(min(len(rows), len(roles))):
        role = str(roles[row_idx] or "").strip().lower()
        alias = str(rows[row_idx].alias or "").strip()
        if not alias or _is_terminal_silence_alias(alias) or _is_nonphonetic_special_alias(alias):
            continue
        if role == "vc":
            seen_transition = True
            continue
        if seen_transition and _is_cvvc_following_cv_block_role(role):
            return row_idx
    return None


def _is_cvvc_following_cv_block_role(role: str) -> bool:
    return str(role or "").strip().lower() in {"cv", "v"}


def _first_moraic_n_tail_index(expected: Sequence[str]) -> int | None:
    normalized = [str(phone or "").strip().lower() for phone in expected]
    for idx, phone in enumerate(normalized):
        if phone != "n" or idx <= 0 or idx + 1 >= len(normalized):
            continue
        prev_phone = normalized[idx - 1]
        next_phone = normalized[idx + 1]
        if is_vowel_phone(prev_phone) and not is_vowel_phone(next_phone):
            return idx
    return None


def _phone_sequence_variants_match(left: Sequence[str], right: Sequence[str]) -> bool:
    if not left or not right:
        return False
    left_variants = _alias_match_phone_variants([str(phone or "").strip().lower() for phone in left])
    right_variants = _alias_match_phone_variants([str(phone or "").strip().lower() for phone in right])
    return any(tuple(lv) == tuple(rv) for lv in left_variants for rv in right_variants)


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
    if role == "cv_head" and _cv_head_phones_are_consonant_like(phones):
        for start, _matched_phones in _find_all_phone_sequence_variants(expected, _alias_match_phone_variants(phones)):
            out.append(AliasTargetCandidate(start, 4.1, "cv_head_onset_sequence"))
        if out:
            return out
    match_phones = _alias_match_phone_variants(phones)
    for start, matched_phones in _find_all_phone_sequence_variants(expected, match_phones):
        out.append(AliasTargetCandidate(start + len(matched_phones) - 1, 4.2, "exact_alias_sequence"))
    if out:
        return out
    target_phone = next((phone for phone in reversed(match_phones[0]) if is_vowel_phone(phone)), match_phones[0][-1])
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
        if prev_role == "vc" and role in {"cv", "vcv"}:
            return -0.20 - (0.006 * float(prev - cur))
        return -1e9
    if cur == prev:
        if _can_share_alias_target(prev_role, role):
            return ALIAS_TARGET_SHARE_BONUS
        return -1e9
    return -0.015 * float(cur - prev)


def _can_share_alias_target(prev_role: str, role: str) -> bool:
    if prev_role == "cv_head" and role == "v":
        return True
    if prev_role == "cv" and role == "vcv":
        return True
    if prev_role == "vcv" and role == "vcv":
        return True
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
    if role == "cv_head" and _cv_head_phones_are_consonant_like(phones):
        match = _find_phone_sequence_variant(expected, _alias_match_phone_variants(phones), min_target_index=min_target_index)
        if match is not None:
            start, _matched_phones = match
            return start
    match = _find_phone_sequence_variant(expected, _alias_match_phone_variants(phones), min_target_index=min_target_index)
    if match is not None:
        start, matched_phones = match
        return start + len(matched_phones) - 1
    match_phones = _alias_match_phone_variants(phones)[0]
    target_phone = next((phone for phone in reversed(match_phones) if is_vowel_phone(phone)), match_phones[-1])
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


def _find_phone_sequence_variant(
    expected: Sequence[str],
    phone_variants: Sequence[Sequence[str]],
    *,
    min_target_index: int,
) -> tuple[int, Sequence[str]] | None:
    for phones in phone_variants:
        start = _find_phone_sequence(expected, phones, min_target_index=min_target_index)
        if start is not None:
            return start, phones
    return None


def _find_all_phone_sequence_variants(
    expected: Sequence[str],
    phone_variants: Sequence[Sequence[str]],
) -> list[tuple[int, Sequence[str]]]:
    out: list[tuple[int, Sequence[str]]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for phones in phone_variants:
        for start in _find_all_phone_sequences(expected, phones):
            key = (start, tuple(phones))
            if key in seen:
                continue
            seen.add(key)
            out.append((start, phones))
    return out


def _vc_right_matches(expected: Sequence[str], start_idx: int, right: Sequence[str]) -> bool:
    if not right or start_idx >= len(expected):
        return False
    for phones in _alias_match_phone_variants(right):
        if _phone_sequence_matches(expected[start_idx : start_idx + len(phones)], phones):
            return True
    if len(right) == 2 and right[1] == "y" and _phone_matches(expected[start_idx], right[0]):
        return True
    if (
        len(right) == 2
        and right[1] == "w"
        and _phone_matches(expected[start_idx], right[0])
        and start_idx + 1 < len(expected)
        and _phone_matches(expected[start_idx + 1], "u")
    ):
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


def _alias_match_phone_variants(phones: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    base = tuple(str(phone or "").strip().lower() for phone in phones if str(phone or "").strip())
    out: list[tuple[str, ...]] = []
    seeds: list[tuple[str, ...]] = []

    def add_seed(seq: Sequence[str]) -> None:
        variant = tuple(str(phone or "").strip().lower() for phone in seq if str(phone or "").strip())
        if variant and variant not in seeds:
            seeds.append(variant)

    collapsed = _collapse_y_vowel_phones(base)
    add_seed(collapsed)
    add_seed(_drop_implicit_y_after_palatal_onset(base))
    add_seed(_drop_implicit_y_after_palatal_onset(collapsed))
    add_seed(_expand_optional_w_vowel_phones(base))
    add_seed(base)
    for variant in list(seeds):
        add_seed(_expand_compound_consonant_phones(variant))
    for variant in seeds:
        if variant and variant not in out:
            out.append(variant)
    return tuple(out) if out else (base,)


def _drop_implicit_y_after_palatal_onset(phones: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(phones):
        phone = str(phones[idx] or "").strip().lower()
        next_phone = str(phones[idx + 1] or "").strip().lower() if idx + 1 < len(phones) else ""
        after_next = str(phones[idx + 2] or "").strip().lower() if idx + 2 < len(phones) else ""
        if phone in {"j", "sh", "ch"} and next_phone == "y" and after_next in {"a", "u", "e", "o"}:
            out.extend((phone, after_next))
            idx += 3
            continue
        out.append(phone)
        idx += 1
    return tuple(out)


def _expand_compound_consonant_phones(phones: Sequence[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    changed = False
    for phone in phones:
        normalized = str(phone or "").strip().lower()
        replacement = {
            "sh": ("s", "h"),
            "ch": ("c", "h"),
            "ts": ("t", "s"),
            "dh": ("d", "h"),
        }.get(normalized)
        if replacement:
            expanded.extend(replacement)
            changed = True
        elif normalized:
            expanded.append(normalized)
    return tuple(expanded) if changed else tuple(str(phone or "").strip().lower() for phone in phones if str(phone or "").strip())


def _collapse_y_vowel_phones(phones: Sequence[str]) -> tuple[str, ...]:
    y_vowels = {
        "a": "ya",
        "ae": "yae",
        "eo": "yeo",
        "e": "ye",
        "o": "yo",
        "u": "yu",
    }
    out: list[str] = []
    idx = 0
    while idx < len(phones):
        phone = str(phones[idx] or "").strip().lower()
        next_phone = str(phones[idx + 1] or "").strip().lower() if idx + 1 < len(phones) else ""
        if phone == "y" and next_phone in y_vowels:
            out.append(y_vowels[next_phone])
            idx += 2
            continue
        out.append(phone)
        idx += 1
    return tuple(out)


def _expand_optional_w_vowel_phones(phones: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(phones):
        phone = str(phones[idx] or "").strip().lower()
        next_phone = str(phones[idx + 1] or "").strip().lower() if idx + 1 < len(phones) else ""
        if phone in {"k", "g"} and next_phone in {"a", "i", "e", "o"}:
            out.extend((phone, "w", next_phone))
            idx += 2
            continue
        out.append(phone)
        idx += 1
    return tuple(out)


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


def _refine_anchor_sequence_locally(
    posterior: FramePosterior,
    anchors: Sequence[OtoAnchor | None],
    *,
    window_ms: float = 25.0,
    attention_delta_ms: float = 12.0,
    low_margin_threshold: float = 0.04,
    min_order_gap_ms: float = 4.0,
) -> list[OtoAnchor | None]:
    if not anchors:
        return []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return list(anchors)
    refined: list[OtoAnchor | None] = list(anchors)
    max_window = max(1.0, float(window_ms))
    for idx, anchor in enumerate(refined):
        if anchor is None:
            continue
        scores = _local_refine_scores(posterior, anchor.role, times)
        if scores.shape[0] != times.shape[0] or not np.any(scores):
            continue
        source_time = float(anchor.anchor_abs_ms)
        prev_anchor = next((item for item in reversed(refined[:idx]) if item is not None), None)
        next_anchor = next((item for item in refined[idx + 1 :] if item is not None), None)
        lower = float(prev_anchor.anchor_abs_ms) + float(min_order_gap_ms) if prev_anchor is not None else -float("inf")
        upper = float(next_anchor.anchor_abs_ms) - float(min_order_gap_ms) if next_anchor is not None else float("inf")
        unrestricted = np.where((times >= source_time - max_window) & (times <= source_time + max_window))[0]
        if unrestricted.size == 0:
            continue
        unrestricted_best = int(unrestricted[int(np.argmax(scores[unrestricted]))])
        unrestricted_time = float(times[unrestricted_best])
        warnings = list(anchor.warnings)
        if unrestricted_time < lower - 1e-6 or unrestricted_time > upper + 1e-6:
            blocked_by_shared_target = (
                unrestricted_time < lower - 1e-6
                and _anchors_share_refine_target(prev_anchor, anchor)
            ) or (
                unrestricted_time > upper + 1e-6
                and _anchors_share_refine_target(anchor, next_anchor)
            )
            if blocked_by_shared_target:
                continue
            warnings.append("local_refine_rejected_slot_boundary")
            refined[idx] = replace(anchor, warnings=tuple(dict.fromkeys(warnings)))
            continue
        allowed = np.where(
            (times >= source_time - max_window)
            & (times <= source_time + max_window)
            & (times >= lower)
            & (times <= upper)
        )[0]
        if allowed.size == 0:
            continue
        best_idx = int(allowed[int(np.argmax(scores[allowed]))])
        ordered_scores = sorted((float(scores[item]) for item in allowed), reverse=True)
        best_score = float(scores[best_idx])
        second_score = ordered_scores[1] if len(ordered_scores) > 1 else best_score
        margin = max(0.0, best_score - float(second_score))
        refined_time = float(times[best_idx])
        delta = refined_time - source_time
        if abs(delta) <= 1e-6:
            continue
        span = estimate_vowel_span(posterior, refined_time)
        nucleus_time, nucleus_conf, nucleus_warnings = estimate_vowel_nucleus(
            posterior,
            refined_time,
            vowel_start_abs_ms=span.get("vowel_start_abs_ms"),
            vowel_end_abs_ms=span.get("vowel_end_abs_ms"),
        )
        warnings.append(f"local_refined_anchor:{source_time:.1f}->{refined_time:.1f}")
        warnings.append(f"local_refine_delta_ms:{delta:.1f}")
        if abs(delta) >= float(attention_delta_ms):
            warnings.append("local_refine_changed_anchor")
        if margin < float(low_margin_threshold):
            warnings.append("local_refine_low_margin")
        warnings.extend(nucleus_warnings)
        refined[idx] = replace(
            anchor,
            anchor_abs_ms=refined_time,
            frame_index=int(best_idx),
            refined_from_abs_ms=source_time,
            local_refine_delta_ms=float(delta),
            local_refine_margin=float(margin),
            local_refine_window_ms=max_window,
            vowel_start_abs_ms=span.get("vowel_start_abs_ms", anchor.vowel_start_abs_ms),
            vowel_end_abs_ms=span.get("vowel_end_abs_ms", anchor.vowel_end_abs_ms),
            vowel_nucleus_abs_ms=nucleus_time if nucleus_time is not None else anchor.vowel_nucleus_abs_ms,
            boundary_confidence=_anchor_boundary_confidence(posterior, refined_time, role=anchor.role),
            nucleus_confidence=max(float(anchor.nucleus_confidence or 0.0), float(nucleus_conf)),
            warnings=tuple(dict.fromkeys(warnings)),
        )
    return refined


def _anchors_share_refine_target(left: OtoAnchor | None, right: OtoAnchor | None) -> bool:
    if left is None or right is None:
        return False
    left_target = left.expected_phone_index
    right_target = right.expected_phone_index
    if left_target is None or right_target is None or int(left_target) != int(right_target):
        return False
    return _can_share_alias_target(str(left.role or ""), str(right.role or ""))


def _local_refine_scores(posterior: FramePosterior, role: str, times: np.ndarray) -> np.ndarray:
    role_norm = str(role or "").strip().lower()
    frame_count = int(times.shape[0])
    zeros = np.zeros((frame_count,), dtype=np.float32)
    vowel = _normalized_track(posterior, "vowel", times, source="class")
    consonant = _normalized_track(posterior, "consonant", times, source="class")
    transition = _normalized_track(posterior, "transition_likelihood", times)
    nucleus = _normalized_track(posterior, "nucleus_likelihood", times)
    voicing = _normalized_track(posterior, "voicing", times)
    if role_norm in {"v", "vowel_nucleus"}:
        event = _event_track(posterior, "vowel_nucleus", frame_count)
        return np.clip((0.62 * event) + (0.20 * vowel) + (0.12 * nucleus) + (0.06 * voicing), 0.0, 1.0)
    if role_norm == "vv":
        phone_change = _event_track(posterior, "phone_change", frame_count)
        cv_boundary = _event_track(posterior, "cv_boundary", frame_count)
        return np.clip((0.40 * phone_change) + (0.24 * cv_boundary) + (0.22 * transition) + (0.14 * vowel), 0.0, 1.0)
    if role_norm == "vc":
        event = _event_track(posterior, "phone_change", frame_count)
        return np.clip((0.66 * event) + (0.20 * transition) + (0.14 * consonant), 0.0, 1.0)
    event = _event_track(posterior, "cv_boundary", frame_count)
    return np.clip((0.66 * event) + (0.16 * transition) + (0.10 * vowel) + (0.08 * consonant), 0.0, 1.0) if frame_count else zeros


def _event_track(posterior: FramePosterior, key: str, frame_count: int) -> np.ndarray:
    values = np.asarray(posterior.event_scores.get(key, []), dtype=np.float32)
    if values.shape[0] != frame_count:
        return np.zeros((frame_count,), dtype=np.float32)
    return np.clip(values.astype(np.float32), 0.0, 1.0)


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
                    source_event_label=label,
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
            "source": "",
        }
    return {
        "label": str(event.get("label")),
        "time_ms": float(event.get("selected_time_ms", event.get("time_ms", 0.0))),
        "score": float(event.get("score", 0.0)),
        "frame_index": event.get("frame_index"),
        "expected_phone": event.get("expected_phone"),
        "expected_phone_index": event.get("expected_phone_index", event.get("phone_index")),
        "slot_index": event.get("slot_index"),
        "source": event.get("source"),
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
