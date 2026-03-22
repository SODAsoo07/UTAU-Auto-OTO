from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

from core.timing_anchor_profiles import AnchorTimingProfile


@dataclass
class AnchorTimingContext:
    file_duration_ms: float
    timeline_start_ms: float
    timeline_end_ms: float
    anchor_abs_ms: Optional[float]
    next_onset_abs_ms: Optional[float] = None
    next_vowel_abs_ms: Optional[float] = None
    voiced_onset_ms: Optional[float] = None
    alias_type: str = "cv"
    language: str = ""
    format_type: str = ""
    mapping_confidence: float = 1.0


@dataclass
class AnchorTimingResult:
    offset: float
    consonant: float
    cutoff: float
    pre: float
    ovl: float
    anchor_shift_ms: float = 0.0
    applied_rules: Sequence[str] = field(default_factory=list)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _blend(current: float, target: float, w: float) -> float:
    ww = _clamp(w, 0.0, 1.0)
    return (float(current) * (1.0 - ww)) + (float(target) * ww)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def adapt_profile_to_file(
    profile: Optional[AnchorTimingProfile],
    *,
    file_mean_energy: float = 0.0,
    file_voiced_ratio: float = 0.0,
    file_mean_syllable_dur_ms: float = 0.0,
    reference_energy: Optional[float] = None,
    reference_syllable_dur_ms: Optional[float] = None,
) -> Optional[AnchorTimingProfile]:
    if profile is None:
        return None
    if str(os.environ.get("UTOA_DISABLE_ANCHOR_PROFILE_ADAPT", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return profile

    energy = float(file_mean_energy or 0.0)
    syllable_ms = float(file_mean_syllable_dur_ms or 0.0)
    if energy <= 0.0 and syllable_ms <= 0.0:
        return profile

    ref_energy = float(reference_energy) if reference_energy is not None else _env_float("UTOA_ANCHOR_REF_ENERGY", 0.45)
    ref_syllable = float(reference_syllable_dur_ms) if reference_syllable_dur_ms is not None else _env_float(
        "UTOA_ANCHOR_REF_SYLLABLE_DUR_MS", 180.0
    )
    energy_ratio = (energy / ref_energy) if ref_energy > 1e-6 else 1.0
    speed_ratio = (ref_syllable / syllable_ms) if syllable_ms > 1e-6 else 1.0

    pre_window_scale = _clamp(speed_ratio, 0.80, 1.20)
    cons_gap_scale = _clamp(speed_ratio, 0.82, 1.18)
    cut_gap_scale = _clamp((energy_ratio * 0.3) + (speed_ratio * 0.7), 0.80, 1.20)

    return replace(
        profile,
        pre_window_before_ms=float(profile.pre_window_before_ms) * pre_window_scale,
        pre_window_after_ms=float(profile.pre_window_after_ms) * pre_window_scale,
        cons_gap_min_ms=float(profile.cons_gap_min_ms) * cons_gap_scale,
        cons_gap_max_ms=float(profile.cons_gap_max_ms) * cons_gap_scale,
        cons_gap_target_ms=float(profile.cons_gap_target_ms) * cons_gap_scale,
        cut_gap_min_ms=float(profile.cut_gap_min_ms) * cut_gap_scale,
        cut_gap_max_ms=float(profile.cut_gap_max_ms) * cut_gap_scale,
        cut_gap_target_ms=float(profile.cut_gap_target_ms) * cut_gap_scale,
    )


def _default_validate(offset: float, consonant: float, cutoff: float, pre: float, ovl: float) -> Tuple[float, float, float, float, float]:
    offset = max(float(offset), 0.0)
    pre = max(float(pre), 0.0)
    ovl = max(float(ovl), 0.0)
    consonant = max(float(consonant), 0.0)
    if ovl > pre:
        ovl = pre * 0.75
    if consonant < pre:
        consonant = pre + 30.0
    cutoff_abs = abs(float(cutoff))
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + 50.0
    return offset, consonant, -cutoff_abs, pre, ovl


def _max_anchor_shift_ms(ctx: AnchorTimingContext, profile: AnchorTimingProfile, lite: bool) -> float:
    wav_ms = max(float(ctx.file_duration_ms or 0.0), 1.0)
    limit = wav_ms * float(profile.max_anchor_shift_ratio)
    limit = _clamp(limit, profile.max_anchor_shift_min_ms, profile.max_anchor_shift_max_ms)
    if str(ctx.language or "").strip().lower() == "korean":
        if float(ctx.mapping_confidence or 1.0) < 0.6:
            limit *= 0.5
    if lite:
        limit *= float(profile.lite_shift_scale)
    return max(limit, 0.0)


def apply_anchor_lock(
    params: Tuple[float, float, float, float, float],
    ctx: AnchorTimingContext,
    profile: Optional[AnchorTimingProfile],
    *,
    validate_fn: Optional[Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]]] = None,
    lite: bool = False,
) -> AnchorTimingResult:
    offset, consonant, cutoff, pre, ovl = [float(v) for v in params]
    validator = validate_fn or _default_validate
    offset, consonant, cutoff, pre, ovl = validator(offset, consonant, cutoff, pre, ovl)
    if profile is None or ctx.anchor_abs_ms is None:
        return AnchorTimingResult(offset, consonant, cutoff, pre, ovl, 0.0, [])

    applied = []
    anchor_abs = float(ctx.anchor_abs_ms)
    pre_abs = float(offset) + float(pre)
    pre_lo = anchor_abs - float(profile.pre_window_before_ms)
    pre_hi = anchor_abs + float(profile.pre_window_after_ms)
    target_pre_abs = _clamp(pre_abs, pre_lo, pre_hi)

    shift = target_pre_abs - pre_abs
    shift_limit = _max_anchor_shift_ms(ctx, profile, lite=lite)
    if abs(shift) > shift_limit:
        shift = _clamp(shift, -shift_limit, shift_limit)
        applied.append("anchor_shift_limited")
    if abs(shift) > 0.01:
        offset += shift
        pre_abs = target_pre_abs if abs(target_pre_abs - (offset + pre)) <= 1.5 else (offset + pre)
        applied.append("pre_anchor_lock")

    if offset < 0.0:
        offset = 0.0
        pre = max(pre_abs, 0.0)
        applied.append("offset_floor")
    else:
        pre = max(pre_abs - offset, 0.0)

    if ctx.voiced_onset_ms is not None and profile is not None:
        voiced_abs = float(ctx.voiced_onset_ms)
        if pre_abs < voiced_abs - 0.5:
            target_pre_abs = _blend(pre_abs, voiced_abs, float(profile.voiced_onset_weight))
            voiced_shift = float(target_pre_abs - pre_abs)
            if abs(voiced_shift) > 0.01:
                if abs(voiced_shift) > shift_limit:
                    voiced_shift = _clamp(voiced_shift, -shift_limit, shift_limit)
                offset = float(offset) + voiced_shift
                if offset < 0.0:
                    offset = 0.0
                    pre = max(target_pre_abs, 0.0)
                else:
                    pre = max(target_pre_abs - offset, 0.0)
                pre_abs = float(offset) + float(pre)
                applied.append("voiced_onset_floor")

    if pre < float(profile.pre_floor_ms):
        desired_offset = pre_abs - float(profile.pre_floor_ms)
        if desired_offset >= 0.0:
            offset = desired_offset
            pre = float(profile.pre_floor_ms)
        else:
            offset = 0.0
            pre = max(pre_abs, float(profile.pre_floor_ms))
        applied.append("pre_floor")

    # Keep overlap tightly coupled to pre for rhythmic stability.
    ovl_gap_now = max(float(pre) - float(ovl), 0.0)
    ovl_gap = _clamp(ovl_gap_now, profile.ovl_gap_min_ms, profile.ovl_gap_max_ms)
    ovl_gap = _blend(
        ovl_gap,
        profile.ovl_gap_target_ms,
        profile.lite_blend_weight if lite else profile.blend_weight,
    )
    ovl = max(0.0, float(pre) - float(ovl_gap))
    applied.append("ovl_gap_shape")

    cons_gap_now = max(float(consonant) - float(pre), 8.0)
    cons_gap = _clamp(cons_gap_now, profile.cons_gap_min_ms, profile.cons_gap_max_ms)
    cons_gap = _blend(
        cons_gap,
        profile.cons_gap_target_ms,
        profile.lite_blend_weight if lite else profile.blend_weight,
    )
    consonant = float(pre) + float(cons_gap)
    applied.append("cons_gap_shape")

    cut_gap_now = max(abs(float(cutoff)) - float(consonant), 10.0)
    cut_gap = _clamp(cut_gap_now, profile.cut_gap_min_ms, profile.cut_gap_max_ms)
    cut_gap = _blend(
        cut_gap,
        profile.cut_gap_target_ms,
        profile.lite_blend_weight if lite else profile.blend_weight,
    )
    cutoff_abs = float(consonant) + float(cut_gap)

    if ctx.next_onset_abs_ms is not None and profile.cut_to_next_onset_allow_ms is not None:
        next_onset_rel = float(ctx.next_onset_abs_ms) - float(offset)
        max_cut = max(float(consonant) + 8.0, next_onset_rel + float(profile.cut_to_next_onset_allow_ms))
        if cutoff_abs > max_cut:
            cutoff_abs = max_cut
            applied.append("cutoff_next_onset_clamp")

    if ctx.next_vowel_abs_ms is not None and profile.cut_to_next_vowel_allow_ms is not None:
        next_vowel_rel = float(ctx.next_vowel_abs_ms) - float(offset)
        max_cut = max(float(consonant) + 8.0, next_vowel_rel + float(profile.cut_to_next_vowel_allow_ms))
        if cutoff_abs > max_cut:
            cutoff_abs = max_cut
            applied.append("cutoff_next_vowel_clamp")

    cutoff = -float(max(cutoff_abs, float(consonant) + 8.0))

    offset, consonant, cutoff, pre, ovl = validator(offset, consonant, cutoff, pre, ovl)
    return AnchorTimingResult(
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        anchor_shift_ms=float((offset + pre) - pre_abs),
        applied_rules=tuple(applied),
    )


def append_timing_anchor_log(log_path: str, record: Dict[str, object]) -> None:
    if not log_path:
        return
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        **record,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = [
    "AnchorTimingContext",
    "AnchorTimingResult",
    "adapt_profile_to_file",
    "apply_anchor_lock",
    "append_timing_anchor_log",
]

