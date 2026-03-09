from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from core.kr_oto_rules import _canonicalize_kr_coda, _extract_vc_right_token


def guard_kr_vc_cutoff_to_next_segment(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
    *,
    alias_text: str = "",
) -> Tuple[float, float, float, float, float]:
    """VC cutoff가 다음 음절로 과도하게 넘어가지 않도록 제한합니다."""
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


def log_post_timing_events(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced):
    """후처리 가드의 의미있는 이동량만 간단히 기록합니다."""
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
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]]
    soft_mel_guard_fn: Callable
    base_shape_blend_fn: Callable
    stabilize_fn: Callable
    recenter_fn: Callable
    cv_cutoff_guard_fn: Callable

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
        soft_off_shift = 0.0
        soft_cut_shift = 0.0
        cutoff_reduced = 0.0
        # Enforce timeline order from the first postprocess step.
        offset, consonant, cutoff, pre, ovl = self.validate_fn(offset, consonant, cutoff, pre, ovl)

        if alias_type in {"cv", "cv_head", "vcv"}:
            offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = self.soft_mel_guard_fn(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                alias_type,
                self.mel_ctx_for_file,
                file_format=self.file_format,
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

        if alias_type == "vc":
            offset, consonant, cutoff, pre, ovl = guard_kr_vc_cutoff_to_next_segment(
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

        if enable_cutoff_guard and alias_type in {"cv", "cv_head"}:
            offset, consonant, cutoff, pre, cutoff_reduced = self.cv_cutoff_guard_fn(
                offset, consonant, cutoff, pre, current_w_idx, self.syllables_info
            )

        # 회귀 방지: CV/VC의 pre-con/cut 길이가 지나치게 짧아지지 않도록 최소 폭을 보장.
        if alias_type in {"cv", "cv_head"}:
            consonant = max(float(consonant), float(pre) + 58.0)
            cutoff_abs = max(abs(float(cutoff)), float(consonant) + 44.0)
            cutoff = -cutoff_abs
        elif alias_type == "vc":
            consonant = max(float(consonant), float(pre) + 24.0)
            cutoff_abs = max(abs(float(cutoff)), float(consonant) + 16.0)
            cutoff = -cutoff_abs

        offset, consonant, cutoff, pre, ovl = self.validate_fn(offset, consonant, cutoff, pre, ovl)
        return offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift, cutoff_reduced
