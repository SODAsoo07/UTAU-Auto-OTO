from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from core.kr_oto_rules import _canonicalize_kr_coda, _extract_vc_right_token, is_plosive_ipa

ValidateFn = Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]]


@dataclass
class KrPostprocessResult:
    offset: float
    consonant: float
    cutoff: float
    pre: float
    ovl: float
    soft_off_shift: float = 0.0
    soft_cut_shift: float = 0.0
    cutoff_reduced: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float, float, float, float, float]:
        return (
            self.offset,
            self.consonant,
            self.cutoff,
            self.pre,
            self.ovl,
            self.soft_off_shift,
            self.soft_cut_shift,
            self.cutoff_reduced,
        )


def guard_kr_vc_cutoff_to_next_segment(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: ValidateFn,
    *,
    alias_text: str = "",
) -> Tuple[float, float, float, float, float]:
    """Clamp VC cutoff so it does not eat into the next syllable onset."""
    if syll_idx is None or syll_idx < 0:
        return validate_fn(offset, consonant, cutoff, pre, ovl)
    if not syllables_info or (syll_idx + 1) >= len(syllables_info):
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    next_syl = syllables_info[syll_idx + 1]
    next_phones = next_syl.get("phones") or []
    if not next_phones:
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    next_onset_rel = max((float(next_phones[0].minTime) * 1000.0) - float(offset), float(pre) + 10.0)
    next_seg_end_rel = max((float(next_phones[0].maxTime) * 1000.0) - float(offset), next_onset_rel + 6.0)

    coda = _canonicalize_kr_coda(_extract_vc_right_token(alias_text))
    if coda in {"k", "t", "p", "h"}:
        onset_margin = 4.0
        tail_keep = 8.0
        min_cons_gap = 12.0
    elif coda in {"n", "m", "ng"}:
        onset_margin = 10.0
        tail_keep = 10.0
        min_cons_gap = 16.0
    elif coda in {"l", "r"}:
        onset_margin = 14.0
        tail_keep = 12.0
        min_cons_gap = 20.0
    else:
        onset_margin = 10.0
        tail_keep = 10.0
        min_cons_gap = 14.0

    cutoff_cap = min(next_onset_rel + onset_margin, next_seg_end_rel - tail_keep)
    consonant = min(float(consonant), cutoff_cap - tail_keep)
    consonant = max(float(consonant), float(pre) + min_cons_gap)

    if cutoff_cap <= consonant + tail_keep:
        consonant = max(float(pre) + min_cons_gap, cutoff_cap - tail_keep)
    cutoff_abs = min(abs(float(cutoff)), max(consonant + tail_keep, cutoff_cap))
    cutoff_abs = min(cutoff_abs, cutoff_cap)
    cutoff = -cutoff_abs
    return validate_fn(offset, consonant, cutoff, pre, ovl)


def guard_kr_vv_cutoff_to_current_vowel(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: ValidateFn,
) -> Tuple[float, float, float, float, float]:
    if syll_idx is None or syll_idx < 0:
        return validate_fn(offset, consonant, cutoff, pre, ovl)
    if not syllables_info or syll_idx >= len(syllables_info):
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    curr_syl = syllables_info[syll_idx] or {}
    curr_phones = curr_syl.get("phones") or []
    if not curr_phones:
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    from core.kr_oto_rules import find_vowel_phone

    _, v_phone = find_vowel_phone(curr_phones)
    v_end_abs = float(getattr(v_phone, "maxTime", 0.0) or 0.0) * 1000.0

    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    cutoff_abs = abs(float(cutoff))

    tail_keep = 12.0
    min_cut_gap = 12.0
    v_end_rel = max(v_end_abs - float(offset), float(consonant) + min_cut_gap)
    cutoff_cap = v_end_rel + tail_keep

    if (syll_idx + 1) < len(syllables_info):
        next_syl = syllables_info[syll_idx + 1] or {}
        next_phones = next_syl.get("phones") or []
        if next_phones:
            next_onset_abs = float(next_phones[0].minTime) * 1000.0
            next_onset_rel = max(next_onset_abs - float(offset), float(pre) + 8.0)
            cutoff_cap = min(cutoff_cap, next_onset_rel - 4.0)

    min_cutoff = float(consonant) + min_cut_gap
    if cutoff_cap < min_cutoff:
        cutoff_cap = min_cutoff

    cutoff_abs = min(cutoff_abs, cutoff_cap)
    cutoff = -cutoff_abs
    return validate_fn(offset, consonant, cutoff, pre, ovl)


def guard_kr_cv_head_offset_to_current_onset(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: ValidateFn,
) -> Tuple[float, float, float, float, float]:
    """Keep CV(-CV) head offset near current onset to avoid excessive front silence."""
    if syll_idx is None or syll_idx < 0:
        return validate_fn(offset, consonant, cutoff, pre, ovl)
    if not syllables_info or syll_idx >= len(syllables_info):
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    # Apply only for true head rows.
    if syll_idx > 0:
        prev_syl = syllables_info[syll_idx - 1] or {}
        if prev_syl.get("phones"):
            return validate_fn(offset, consonant, cutoff, pre, ovl)

    curr_syl = syllables_info[syll_idx] or {}
    curr_phones = curr_syl.get("phones") or []
    if not curr_phones:
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    c_start = float(curr_phones[0].minTime) * 1000.0
    if c_start <= 0.5:
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    c_hint = (curr_phones[0].mark or "").strip().lower()
    try:
        from core.kr_oto_rules import find_vowel_phone

        _v_idx, v_phone = find_vowel_phone(curr_phones)
        c_end = float(getattr(v_phone, "minTime", 0.0) or 0.0) * 1000.0
    except Exception:
        c_end = float(curr_phones[0].maxTime) * 1000.0

    c_len = max(0.0, c_end - c_start)
    if is_plosive_ipa(c_hint) or c_hint in {"s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"}:
        lead_min, lead_max = 10.0, 18.0
    elif c_hint in {"m", "n", "ng", "l", "r", "y", "w", "ny"}:
        lead_min, lead_max = 8.0, 14.0
    else:
        lead_min, lead_max = 9.0, 16.0
    lead_keep = max(lead_min, min(lead_max, c_len * 0.45 + 6.0))

    desired_offset = max(0.0, c_start - lead_keep)
    if float(offset) >= desired_offset - 0.5:
        return validate_fn(offset, consonant, cutoff, pre, ovl)

    delta = desired_offset - float(offset)
    offset = desired_offset
    pre = max(0.0, float(pre) - delta)
    ovl = max(0.0, float(ovl) - delta)
    consonant = max(0.0, float(consonant) - delta)
    cutoff_abs = max(0.0, abs(float(cutoff)) - delta)
    cutoff = -cutoff_abs
    return validate_fn(offset, consonant, cutoff, pre, ovl)


def log_post_timing_events(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced):
    """Log notable postprocess timing changes."""
    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
        log_fn(
            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
        )
    if cutoff_reduced > 0.5:
        log_fn(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")


@dataclass
class KrPostprocessContext:
    file_format: str
    mel_ctx_for_file: object
    ph_intervals: Sequence
    syllables_info: Sequence[dict]
    validate_fn: ValidateFn
    soft_mel_guard_fn: Callable
    base_shape_blend_fn: Callable
    stabilize_fn: Callable
    recenter_fn: Callable
    cv_cutoff_guard_fn: Callable

    def _resolve_onset_hint(self, current_w_idx: Optional[int]) -> str:
        try:
            if current_w_idx is None:
                return ""
            idx = int(current_w_idx)
            if idx < 0 or idx >= len(self.syllables_info):
                return ""
            curr_syl = self.syllables_info[idx] or {}
            curr_phones = curr_syl.get("phones") or []
            if not curr_phones:
                return ""
            return str(getattr(curr_phones[0], "mark", "") or "").strip()
        except Exception:
            return ""

    def _apply_soft_mel_stage(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str,
        alias_text: str,
        onset_hint: str,
    ) -> Tuple[float, float, float, float, float, float, float]:
        if alias_type not in {"cv", "cv_head", "vcv"}:
            return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
        return self.soft_mel_guard_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type,
            self.mel_ctx_for_file,
            onset_hint=onset_hint,
            alias_text=alias_text,
            file_format=self.file_format,
        )

    def _apply_alias_specific_guards(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str,
        alias_text: str,
        current_w_idx: int,
    ) -> Tuple[float, float, float, float, float]:
        if alias_type == "cv":
            return guard_kr_cv_head_offset_to_current_onset(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                current_w_idx,
                self.syllables_info,
                self.validate_fn,
            )
        if alias_type == "vc":
            return guard_kr_vc_cutoff_to_next_segment(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                current_w_idx,
                self.syllables_info,
                self.validate_fn,
                alias_text=alias_text,
            )
        if alias_type == "vv":
            return guard_kr_vv_cutoff_to_current_vowel(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                current_w_idx,
                self.syllables_info,
                self.validate_fn,
            )
        return offset, consonant, cutoff, pre, ovl

    @staticmethod
    def _enforce_spacing_floor(
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str,
    ) -> Tuple[float, float, float, float, float]:
        # Keep conservative spacing floors for stability.
        if alias_type in {"cv", "cv_head"}:
            consonant = max(float(consonant), float(pre) + 58.0)
            cutoff = -max(abs(float(cutoff)), float(consonant) + 44.0)
        elif alias_type == "vc":
            consonant = max(float(consonant), float(pre) + 24.0)
            cutoff = -max(abs(float(cutoff)), float(consonant) + 16.0)
        return offset, consonant, cutoff, pre, ovl

    def apply_result(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str,
        alias_text: str,
        base_shape: dict,
        current_w_idx: int,
        is_vc_plosive_coda: bool = False,
        enable_stabilize: bool = True,
        enable_cutoff_guard: bool = True,
    ) -> KrPostprocessResult:
        soft_off_shift = 0.0
        soft_cut_shift = 0.0
        cutoff_reduced = 0.0
        offset, consonant, cutoff, pre, ovl = self.validate_fn(offset, consonant, cutoff, pre, ovl)

        onset_hint = self._resolve_onset_hint(current_w_idx)
        (
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            soft_off_shift,
            soft_cut_shift,
        ) = self._apply_soft_mel_stage(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            alias_text=alias_text,
            onset_hint=onset_hint,
        )

        if not is_vc_plosive_coda:
            offset, consonant, cutoff, pre, ovl = self.base_shape_blend_fn(
                offset, consonant, cutoff, pre, ovl, base_shape, alias_type=alias_type
            )

        if enable_stabilize:
            offset, consonant, cutoff, pre, ovl = self.stabilize_fn(
                offset, consonant, cutoff, pre, ovl, self.ph_intervals, alias_type=alias_type
            )

        offset, consonant, cutoff, pre, ovl = self.recenter_fn(
            offset, consonant, cutoff, pre, ovl, alias_type=alias_type, alias_text=alias_text
        )

        offset, consonant, cutoff, pre, ovl = self._apply_alias_specific_guards(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            alias_text=alias_text,
            current_w_idx=current_w_idx,
        )

        if enable_cutoff_guard and alias_type in {"cv", "cv_head"}:
            offset, consonant, cutoff, pre, cutoff_reduced = self.cv_cutoff_guard_fn(
                offset, consonant, cutoff, pre, current_w_idx, self.syllables_info
            )

        offset, consonant, cutoff, pre, ovl = self._enforce_spacing_floor(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
        )
        offset, consonant, cutoff, pre, ovl = self.validate_fn(offset, consonant, cutoff, pre, ovl)

        return KrPostprocessResult(
            offset=offset,
            consonant=consonant,
            cutoff=cutoff,
            pre=pre,
            ovl=ovl,
            soft_off_shift=soft_off_shift,
            soft_cut_shift=soft_cut_shift,
            cutoff_reduced=cutoff_reduced,
        )

    def apply(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str,
        alias_text: str,
        base_shape: dict,
        current_w_idx: int,
        is_vc_plosive_coda: bool = False,
        enable_stabilize: bool = True,
        enable_cutoff_guard: bool = True,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        result = self.apply_result(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            alias_text=alias_text,
            base_shape=base_shape,
            current_w_idx=current_w_idx,
            is_vc_plosive_coda=is_vc_plosive_coda,
            enable_stabilize=enable_stabilize,
            enable_cutoff_guard=enable_cutoff_guard,
        )
        return result.as_tuple()


__all__ = [
    "KrPostprocessContext",
    "KrPostprocessResult",
    "guard_kr_cv_head_offset_to_current_onset",
    "guard_kr_vc_cutoff_to_next_segment",
    "guard_kr_vv_cutoff_to_current_vowel",
    "log_post_timing_events",
]
