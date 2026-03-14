"""
Japanese OTO postprocess helpers.

Isolates final timing stabilization and CV_HEAD safety guards from the
main generator loop so they can be adjusted independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

from core.ja_oto_mapping import _clean_phone_mark

ValidateFn = Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]]

JA_PLOSIVE_CONSONANTS = {
    "k",
    "g",
    "t",
    "d",
    "b",
    "p",
    "kk",
    "tt",
    "pp",
    "dd",
    "gg",
    "bb",
    "ch",
    "ts",
    "q",
    "c",
    "j",
    "ky",
    "gy",
    "ty",
    "dy",
    "by",
    "py",
}
JA_SIBILANT_ONSETS = {"s", "z", "sh", "j", "ts", "dz", "ch"}


@dataclass
class JaPostAdjustResult:
    offset: float
    consonant: float
    cutoff: float
    pre: float
    ovl: float

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return self.offset, self.consonant, self.cutoff, self.pre, self.ovl


@dataclass
class JaPostprocessContext:
    phone_spans_ms: Sequence[Tuple[float, float]]
    timeline_start_ms: float
    effective_end_ms: float
    validate_fn: ValidateFn
    recenter_fn: Callable[..., Tuple[float, float, float, float, float]]
    extract_cv_bounds_fn: Callable[..., Tuple[float, float, float, float]]
    cv_onset_class_fn: Callable[..., Tuple[str, str]]
    file_format: str = ""
    syllables_info: Sequence[dict] = field(default_factory=list)
    ja_style_enabled: bool = False
    ja_style_profile: object = None
    autotune_profile: object = None
    style_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None
    autotune_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None

    def post_adjust_result(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str = "cv",
        alias_text: str = "",
        local_end_ms: Optional[float] = None,
        local_cut_allow_ms: Optional[float] = None,
    ) -> JaPostAdjustResult:
        return post_adjust_result(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            alias_text=alias_text,
            local_end_ms=local_end_ms,
            local_cut_allow_ms=local_cut_allow_ms,
            phone_spans_ms=self.phone_spans_ms,
            timeline_start_ms=self.timeline_start_ms,
            effective_end_ms=self.effective_end_ms,
            validate_fn=self.validate_fn,
            recenter_fn=self.recenter_fn,
            ja_style_enabled=self.ja_style_enabled,
            ja_style_profile=self.ja_style_profile,
            autotune_profile=self.autotune_profile,
            style_apply_fn=self.style_apply_fn,
            autotune_apply_fn=self.autotune_apply_fn,
        )

    def post_adjust(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        ovl: float,
        *,
        alias_type: str = "cv",
        alias_text: str = "",
        local_end_ms: Optional[float] = None,
        local_cut_allow_ms: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float]:
        return self.post_adjust_result(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            alias_text=alias_text,
            local_end_ms=local_end_ms,
            local_cut_allow_ms=local_cut_allow_ms,
        ).as_tuple()

    def guard_cv_cutoff_to_next_onset(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        syll_idx: Optional[int],
        *,
        alias_type: str = "cv",
        format_type: str = "",
        vowel_start_ms: Optional[float] = None,
        vowel_end_ms: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float]:
        return guard_ja_cv_cutoff_to_next_onset(
            offset,
            consonant,
            cutoff,
            pre,
            syll_idx,
            self.syllables_info,
            self.validate_fn,
            alias_type=alias_type,
            format_type=format_type,
            vowel_start_ms=vowel_start_ms,
            vowel_end_ms=vowel_end_ms,
        )

    def guard_cv_head_offset_to_onset(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        syll_idx: Optional[int],
        *,
        alias_text: str = "",
    ) -> Tuple[float, float, float, float, float]:
        return guard_ja_cv_head_offset_to_onset(
            offset,
            consonant,
            cutoff,
            pre,
            syll_idx,
            self.syllables_info,
            self.extract_cv_bounds_fn,
            self.cv_onset_class_fn,
            self.validate_fn,
            alias_text=alias_text,
        )

    def ensure_cv_head_min_vowel_coverage(
        self,
        offset: float,
        consonant: float,
        cutoff: float,
        pre: float,
        vowel_start_ms: float,
        vowel_end_ms: float,
    ) -> Tuple[float, float, float, float, float]:
        return ensure_ja_cv_head_min_vowel_coverage(
            offset,
            consonant,
            cutoff,
            pre,
            vowel_start_ms,
            vowel_end_ms,
            self.validate_fn,
        )

    def cv_head_min_cutoff(
        self,
        offset: float,
        consonant: float,
        pre: float,
        vowel_start_ms: float,
        vowel_end_ms: float,
    ) -> Optional[float]:
        return ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)


def _nearest_phone_edge_ms(phone_spans_ms: Sequence[Tuple[float, float]], anchor_ms: float) -> Tuple[float, float]:
    nearest = None
    nearest_dist = float("inf")
    for s_ms, e_ms in phone_spans_ms:
        if s_ms <= anchor_ms <= e_ms:
            return anchor_ms, 0.0
        ds = abs(anchor_ms - s_ms)
        de = abs(anchor_ms - e_ms)
        if ds < nearest_dist:
            nearest_dist = ds
            nearest = s_ms
        if de < nearest_dist:
            nearest_dist = de
            nearest = e_ms
    if nearest is None:
        return anchor_ms, 0.0
    return nearest, nearest_dist


def _surrounding_gap_ms(phone_spans_ms: Sequence[Tuple[float, float]], anchor_ms: float) -> Tuple[Optional[float], Optional[float], float]:
    prev_end = None
    next_start = None
    for s_ms, e_ms in phone_spans_ms:
        if s_ms <= anchor_ms <= e_ms:
            return None, None, 0.0
        if e_ms < anchor_ms:
            prev_end = e_ms
        elif s_ms > anchor_ms and next_start is None:
            next_start = s_ms
            break
    if prev_end is None or next_start is None or next_start <= prev_end:
        return None, None, 0.0
    return prev_end, next_start, (next_start - prev_end)


def _stabilize_pre_abs(
    pre_abs: float,
    *,
    alias_type: str,
    phone_spans_ms: Sequence[Tuple[float, float]],
) -> float:
    nearest_edge, nearest_dist = _nearest_phone_edge_ms(phone_spans_ms, pre_abs)
    prev_end_ms, next_start_ms, gap_len_ms = _surrounding_gap_ms(phone_spans_ms, pre_abs)
    if gap_len_ms < 55.0 and nearest_dist <= 34.0:
        return pre_abs

    target = nearest_edge
    if prev_end_ms is not None and next_start_ms is not None:
        if alias_type in ("vc", "vv"):
            target = max(prev_end_ms + 4.0, next_start_ms - 6.0)
        elif alias_type in ("cv", "cv_head"):
            target = max(prev_end_ms + 3.0, next_start_ms - 4.0)
        else:
            target = prev_end_ms
    if abs(target - pre_abs) >= 2.0:
        return target
    return pre_abs


def _clamp_pre_abs(
    pre_abs: float,
    *,
    alias_type: str,
    timeline_start_ms: float,
    effective_end_ms: float,
) -> float:
    min_pre_abs = max(timeline_start_ms - (20.0 if alias_type in ("vc", "vv", "vcv") else 10.0), 0.0)
    max_pre_abs = effective_end_ms + (30.0 if alias_type in ("vc", "vv", "vcv") else 80.0)
    return max(min_pre_abs, min(pre_abs, max_pre_abs))


def _offset_floor(alias_type: str, timeline_start_ms: float) -> float:
    return max(timeline_start_ms - (70.0 if alias_type in ("vc", "vv", "vcv") else 40.0), 0.0)


def _sync_offset_and_pre_abs(offset: float, pre_abs: float, *, offset_floor: float) -> Tuple[float, float]:
    if offset < offset_floor:
        offset = offset_floor
    if pre_abs - offset > 340.0:
        offset = max(pre_abs - 340.0, 0.0)
    if pre_abs < offset:
        pre_abs = offset + 10.0
    return offset, pre_abs


def _apply_vcv_pre_floor(offset: float, pre_abs: float, pre: float, *, offset_floor: float) -> Tuple[float, float, float]:
    vcv_pre_floor = 46.0
    if pre < vcv_pre_floor:
        expand_offset = max(pre_abs - vcv_pre_floor, offset_floor)
        if expand_offset < offset:
            offset = expand_offset
            pre = max(pre_abs - offset, 0.0)
    if pre < vcv_pre_floor:
        pre_abs = offset + vcv_pre_floor
        pre = vcv_pre_floor
    return offset, pre_abs, pre


def _min_cons_gap_by_alias(alias_type: str) -> float:
    if alias_type in {"cv", "cv_head"}:
        return 28.0
    if alias_type == "vc":
        return 16.0
    if alias_type == "vv":
        return 20.0
    if alias_type == "vcv":
        return 52.0
    return 25.0


def _resolve_cut_allow_and_cons_allow(alias_type: str, local_cut_allow_ms: Optional[float]) -> Tuple[float, float]:
    cut_allow_ms = 120.0 if local_cut_allow_ms is None else float(local_cut_allow_ms)
    if alias_type == "vcv":
        cut_allow_ms = max(cut_allow_ms, 88.0)
        cons_allow_ms = max(72.0, cut_allow_ms - 12.0)
    elif alias_type in {"cv", "cv_head"}:
        cut_allow_ms = max(cut_allow_ms, 72.0)
        cons_allow_ms = max(74.0, cut_allow_ms + 10.0)
    elif alias_type == "vc":
        cut_allow_ms = max(cut_allow_ms, 44.0)
        cons_allow_ms = max(52.0, cut_allow_ms + 4.0)
    else:
        cons_allow_ms = max(40.0, cut_allow_ms - 40.0)
    return cut_allow_ms, cons_allow_ms


def _enforce_cutoff_floor(alias_type: str, consonant: float, cutoff_abs: float, max_cut_abs: float) -> float:
    if alias_type == "vcv":
        min_cut_abs = consonant + 32.0
        if cutoff_abs < min_cut_abs:
            cutoff_abs = min(min_cut_abs, max_cut_abs)
    elif alias_type in {"cv", "cv_head"}:
        min_cut_abs = consonant + 16.0
        if cutoff_abs < min_cut_abs:
            cutoff_abs = min(min_cut_abs, max_cut_abs)
    elif alias_type == "vc":
        min_cut_abs = consonant + 10.0
        if cutoff_abs < min_cut_abs:
            cutoff_abs = min(min_cut_abs, max_cut_abs)
    return cutoff_abs


def _cap_fixed_region(
    alias_type: str,
    pre: float,
    consonant: float,
    cutoff_abs: float,
    *,
    original_consonant: float,
) -> Tuple[float, float]:
    if alias_type in {"cv", "cv_head"}:
        max_cons_gap = 46.0
        min_cut_gap = 12.0
        center_lead = 8.0
        center_span = 20.0
    elif alias_type == "vc":
        max_cons_gap = 26.0
        min_cut_gap = 8.0
        center_lead = 6.0
        center_span = 14.0
    elif alias_type == "vv":
        max_cons_gap = 34.0
        min_cut_gap = 10.0
        center_lead = 6.0
        center_span = 14.0
    elif alias_type == "vcv":
        max_cons_gap = 78.0
        min_cut_gap = 14.0
        center_lead = 8.0
        center_span = 18.0
    else:
        return consonant, cutoff_abs

    cons_gap = max(0.0, float(consonant) - float(pre))
    cons_gap = min(cons_gap, max_cons_gap)
    consonant = float(pre) + cons_gap

    target_center = max(float(consonant) + min_cut_gap, float(original_consonant) + center_lead)
    cut_min = max(float(consonant) + min_cut_gap, target_center - center_span)
    cut_max = max(cut_min + 4.0, target_center + center_span)
    cutoff_abs = min(max(float(cutoff_abs), cut_min), cut_max)
    return consonant, cutoff_abs


def _apply_style_and_autotune(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    *,
    alias_type: str,
    alias_text: str,
    ja_style_enabled: bool,
    ja_style_profile,
    autotune_profile,
    style_apply_fn,
    autotune_apply_fn,
) -> Tuple[float, float, float, float, float]:
    if ja_style_enabled and ja_style_profile and not autotune_profile and style_apply_fn:
        offset, consonant, cutoff, pre, ovl = style_apply_fn(
            alias_type,
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            ja_style_profile,
            alias_text=alias_text,
        )
    if autotune_profile and autotune_apply_fn:
        offset, consonant, cutoff, pre, ovl = autotune_apply_fn(
            alias_type,
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            autotune_profile,
        )
    return offset, consonant, cutoff, pre, ovl


def post_adjust_result(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    *,
    alias_type: str = "cv",
    alias_text: str = "",
    local_end_ms: Optional[float] = None,
    local_cut_allow_ms: Optional[float] = None,
    phone_spans_ms: Sequence[Tuple[float, float]],
    timeline_start_ms: float,
    effective_end_ms: float,
    validate_fn: ValidateFn,
    recenter_fn: Callable[..., Tuple[float, float, float, float, float]],
    ja_style_enabled: bool = False,
    ja_style_profile=None,
    autotune_profile=None,
    style_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
    autotune_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
) -> JaPostAdjustResult:
    alias_type = str(alias_type or "cv").strip().lower()
    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)

    pre_abs = float(offset) + float(pre)
    pre_abs = _stabilize_pre_abs(pre_abs, alias_type=alias_type, phone_spans_ms=phone_spans_ms)
    pre_abs = _clamp_pre_abs(
        pre_abs,
        alias_type=alias_type,
        timeline_start_ms=float(timeline_start_ms),
        effective_end_ms=float(effective_end_ms),
    )

    floor = _offset_floor(alias_type, float(timeline_start_ms))
    offset, pre_abs = _sync_offset_and_pre_abs(float(offset), float(pre_abs), offset_floor=floor)

    pre = max(pre_abs - float(offset), 0.0)
    if alias_type == "vcv":
        offset, pre_abs, pre = _apply_vcv_pre_floor(float(offset), float(pre_abs), float(pre), offset_floor=floor)

    if ovl > pre:
        ovl = pre * 0.72

    original_consonant = float(consonant)
    min_cons_gap = _min_cons_gap_by_alias(alias_type)
    if consonant < pre + min_cons_gap:
        consonant = pre + min_cons_gap

    cut_anchor_ms = float(effective_end_ms if local_end_ms is None else local_end_ms)
    cut_allow_ms, cons_allow_ms = _resolve_cut_allow_and_cons_allow(alias_type, local_cut_allow_ms)

    max_cons_abs = max((cut_anchor_ms + cons_allow_ms) - float(offset), float(pre) + 40.0)
    consonant = min(float(consonant), max_cons_abs)

    max_cut_abs = max((cut_anchor_ms + cut_allow_ms) - float(offset), float(consonant) + 35.0)
    cutoff_abs = min(abs(float(cutoff)), max_cut_abs)
    consonant, cutoff_abs = _cap_fixed_region(
        alias_type,
        float(pre),
        float(consonant),
        float(cutoff_abs),
        original_consonant=original_consonant,
    )
    consonant = min(float(consonant), max_cons_abs)
    cutoff_abs = min(float(cutoff_abs), max_cut_abs)
    cutoff_abs = _enforce_cutoff_floor(alias_type, float(consonant), float(cutoff_abs), float(max_cut_abs))
    cutoff = -float(cutoff_abs)

    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    offset, consonant, cutoff, pre, ovl = _apply_style_and_autotune(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias_text,
        ja_style_enabled=ja_style_enabled,
        ja_style_profile=ja_style_profile,
        autotune_profile=autotune_profile,
        style_apply_fn=style_apply_fn,
        autotune_apply_fn=autotune_apply_fn,
    )
    offset, consonant, cutoff, pre, ovl = recenter_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias_text,
    )
    return JaPostAdjustResult(
        offset=float(offset),
        consonant=float(consonant),
        cutoff=float(cutoff),
        pre=float(pre),
        ovl=float(ovl),
    )


def post_adjust_params(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    *,
    alias_type: str = "cv",
    alias_text: str = "",
    local_end_ms: Optional[float] = None,
    local_cut_allow_ms: Optional[float] = None,
    phone_spans_ms: Sequence[Tuple[float, float]],
    timeline_start_ms: float,
    effective_end_ms: float,
    validate_fn: ValidateFn,
    recenter_fn: Callable[..., Tuple[float, float, float, float, float]],
    ja_style_enabled: bool = False,
    ja_style_profile=None,
    autotune_profile=None,
    style_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
    autotune_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
) -> Tuple[float, float, float, float, float]:
    return post_adjust_result(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias_text,
        local_end_ms=local_end_ms,
        local_cut_allow_ms=local_cut_allow_ms,
        phone_spans_ms=phone_spans_ms,
        timeline_start_ms=timeline_start_ms,
        effective_end_ms=effective_end_ms,
        validate_fn=validate_fn,
        recenter_fn=recenter_fn,
        ja_style_enabled=ja_style_enabled,
        ja_style_profile=ja_style_profile,
        autotune_profile=autotune_profile,
        style_apply_fn=style_apply_fn,
        autotune_apply_fn=autotune_apply_fn,
    ).as_tuple()


def ja_cv_head_min_cutoff_abs(offset: float, consonant: float, pre: float, vowel_start_ms: float, vowel_end_ms: float) -> Optional[float]:
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)
    if v_len < 40.0:
        return None
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    keep_v_ms = min(210.0, max(85.0, v_len * 0.36))
    min_from_pre = min(185.0, max(95.0, v_len * 0.26))
    return max(float(consonant) + 12.0, vowel_start_rel + keep_v_ms, float(pre) + min_from_pre)


def _ja_cv_head_min_cutoff_abs(
    offset: float,
    consonant: float,
    pre: float,
    vowel_start_ms: float,
    vowel_end_ms: float,
) -> Optional[float]:
    """
    Backward-compatible alias kept for in-flight refactor safety.
    """
    return ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)


def guard_ja_cv_cutoff_to_next_onset(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: ValidateFn,
    *,
    alias_type: str = "cv",
    format_type: str = "",
    vowel_start_ms: Optional[float] = None,
    vowel_end_ms: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    if syll_idx is None or syll_idx < 0 or (syll_idx + 1) >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0
    next_syl = syllables_info[syll_idx + 1]
    next_phones = next_syl.get("phones") or []
    if not next_phones:
        return offset, consonant, cutoff, pre, 0.0

    next_mark = _clean_phone_mark(getattr(next_phones[0], "mark", ""))
    hard_next = next_mark in JA_PLOSIVE_CONSONANTS or next_mark in JA_SIBILANT_ONSETS or next_mark in {"ts", "ch", "j", "sh", "s", "z", "h"}
    is_cv_head = str(alias_type or "").strip().lower() == "cv_head"
    fmt = str(format_type or "").strip().lower()
    safety = 10.0 if is_cv_head else (18.0 if hard_next else 12.0)
    if not is_cv_head and fmt == "cvvc":
        safety += 4.0
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if is_cv_head:
        max_cutoff_abs = next_onset_rel + (10.0 if hard_next else 18.0)
    if max_cutoff_abs <= (pre + 18.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 14.0)
    consonant = max(consonant, pre + 10.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    if is_cv_head and vowel_start_ms is not None and vowel_end_ms is not None:
        min_cutoff_abs = ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)
        if min_cutoff_abs is not None:
            cutoff_abs = max(cutoff_abs, min(min_cutoff_abs, max_cutoff_abs))
    if cutoff_abs <= (consonant + 8.0):
        cutoff_abs = min(max_cutoff_abs, consonant + 10.0)
        if cutoff_abs <= (consonant + 6.0):
            consonant = max(pre + 8.0, cutoff_abs - 10.0)
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, _ovl = validate_fn(offset, consonant, cutoff, pre, 0.0)
    reduction = max(0.0, original_cutoff_abs - abs(cutoff))
    return offset, consonant, cutoff, pre, reduction


def guard_ja_cv_head_offset_to_onset(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    extract_cv_bounds_fn: Callable[..., Tuple[float, float, float, float]],
    cv_onset_class_fn: Callable[..., Tuple[str, str]],
    validate_fn: ValidateFn,
    *,
    alias_text: str = "",
) -> Tuple[float, float, float, float, float]:
    if syll_idx is None or syll_idx < 0 or syll_idx >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0
    curr_syl = syllables_info[syll_idx]
    curr_phones = curr_syl.get("phones") or []
    if not curr_phones:
        return offset, consonant, cutoff, pre, 0.0

    c_hint = curr_phones[0].mark if curr_phones else ""
    c_start, c_end, _n_start, _n_end = extract_cv_bounds_fn(curr_phones, alias_text=alias_text, alias_type="cv_head")
    cls, _onset = cv_onset_class_fn(alias_text, c_hint=c_hint, alias_type="cv_head")
    if cls == "voiceless":
        base_lead = 44.0
    elif cls == "voiced":
        base_lead = 36.0
    elif cls == "nasal":
        base_lead = 30.0
    else:
        base_lead = 34.0
    c_len = max(0.0, float(c_end) - float(c_start))
    lead_cap = min(base_lead, max(20.0, c_len + 16.0))
    offset_floor = max(0.0, float(c_start) - lead_cap)
    if offset >= offset_floor:
        return offset, consonant, cutoff, pre, 0.0

    new_offset = offset_floor
    new_pre = max(float(pre), 8.0)
    new_consonant = max(float(consonant), new_pre + 8.0)
    new_cut_abs = max(abs(float(cutoff)), new_consonant + 12.0)
    new_cutoff = -new_cut_abs
    new_offset, new_consonant, new_cutoff, new_pre, _ovl = validate_fn(new_offset, new_consonant, new_cutoff, new_pre, 0.0)
    reduced_ms = max(0.0, new_offset - offset)
    return new_offset, new_consonant, new_cutoff, new_pre, reduced_ms


def ensure_ja_cv_head_min_vowel_coverage(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    vowel_start_ms: float,
    vowel_end_ms: float,
    validate_fn: ValidateFn,
) -> Tuple[float, float, float, float, float]:
    min_cut_abs = ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)
    if min_cut_abs is None:
        return offset, consonant, cutoff, pre, 0.0

    cut_abs = abs(float(cutoff))
    if cut_abs >= min_cut_abs:
        return offset, consonant, cutoff, pre, 0.0

    new_cutoff = -min_cut_abs
    offset, consonant, new_cutoff, pre, _ovl = validate_fn(offset, consonant, new_cutoff, pre, 0.0)
    extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
    return offset, consonant, new_cutoff, pre, extended_ms


__all__ = [
    "JA_PLOSIVE_CONSONANTS",
    "JA_SIBILANT_ONSETS",
    "JaPostAdjustResult",
    "JaPostprocessContext",
    "ensure_ja_cv_head_min_vowel_coverage",
    "guard_ja_cv_cutoff_to_next_onset",
    "guard_ja_cv_head_offset_to_onset",
    "ja_cv_head_min_cutoff_abs",
    "post_adjust_params",
    "post_adjust_result",
]
