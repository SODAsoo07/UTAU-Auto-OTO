from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict

from core.kr_oto_cv import _compute_kr_cv_timing, _select_kr_cv_onset_slice
from core.kr_oto_rules import (
    _extract_alias_onset,
    _looks_like_vv_alias,
    find_vowel_phone,
    is_plosive_ipa,
    is_plosive_roman,
)
from core.kr_oto_vc import _compute_kr_vc_timing
from core.kr_oto_vv import (
    _compute_kr_cvvc_vv_timing_direct,
    _compute_kr_noninitial_vowel_timing,
    _compute_kr_vv_timing_from_vowel_bounds,
)


def compute_kr_general_base_timing(
    *,
    is_vc: bool,
    alias: str,
    alias_type: str,
    file_format: str,
    curr_phones: Sequence[Any],
    current_w_idx: int,
    selected_w_idx: int | None,
    bridge_next_idx: int | None,
    bridge_pair: Mapping[str, Any],
    syllables_info: Sequence[Mapping[str, Any]],
    c_start: float,
    c_end: float,
    n_start: float,
    n_end: float,
    cv_anchor_by_idx: Mapping[int, Any],
    is_diph: bool,
    kr_cv_v_ref_baseline: float,
    kr_cv_timing_mode: str,
    mel_ctx_for_file: Any,
    fname: str,
    estimate_mel_vowel_activity_span_fn: Callable[..., Any] | None = None,
    log_vv_refine_fn: Callable[[str, str, float, float, float, float], None] | None = None,
    compute_vc_timing_fn: Callable[..., Any] = _compute_kr_vc_timing,
    compute_cv_timing_fn: Callable[..., Any] = _compute_kr_cv_timing,
    select_cv_onset_slice_fn: Callable[..., Any] = _select_kr_cv_onset_slice,
    extract_alias_onset_fn: Callable[[str], str] = _extract_alias_onset,
    looks_like_vv_alias_fn: Callable[[str], bool] = _looks_like_vv_alias,
    find_vowel_phone_fn: Callable[[Sequence[Any]], Any] = find_vowel_phone,
    compute_cvvc_vv_timing_direct_fn: Callable[..., Any] = _compute_kr_cvvc_vv_timing_direct,
    compute_vv_timing_from_vowel_bounds_fn: Callable[..., Any] = _compute_kr_vv_timing_from_vowel_bounds,
    compute_noninitial_vowel_timing_fn: Callable[..., Any] = _compute_kr_noninitial_vowel_timing,
    is_plosive_ipa_fn: Callable[[str], bool] = is_plosive_ipa,
    is_plosive_roman_fn: Callable[[str], bool] = is_plosive_roman,
) -> Dict[str, Any]:
    cv_vowel_len = float(n_end) - float(n_start)
    is_vc_plosive_coda = False

    if is_vc:
        (
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            is_vc_plosive_coda,
        ) = compute_vc_timing_fn(
            alias,
            alias_type,
            file_format,
            curr_phones,
            current_w_idx,
            syllables_info,
            c_start,
            c_end,
            n_start,
            n_end,
            cv_anchor_by_idx,
            next_w_idx=bridge_next_idx,
            prev_cv_anchor=bridge_pair.get("prev_anchor"),
            next_cv_anchor=bridge_pair.get("next_anchor"),
        )
    else:
        onset_slice_hint = select_cv_onset_slice_fn(curr_phones) if curr_phones else None
        onset_idx_hint = int(onset_slice_hint[0]) if onset_slice_hint is not None else 0
        glide_dur_hint_ms = (
            float(onset_slice_hint[5])
            if onset_slice_hint is not None and len(onset_slice_hint) >= 6
            else 0.0
        )
        if onset_idx_hint < 0 or onset_idx_hint >= len(curr_phones or []):
            onset_idx_hint = 0
        c_hint = curr_phones[onset_idx_hint].mark if curr_phones else ""

        if is_diph:
            alias_onset = extract_alias_onset_fn(alias)
            offset, consonant, cutoff, pre, ovl = compute_cv_timing_fn(
                c_start,
                c_end,
                cv_vowel_len,
                c_hint,
                alias_onset,
                True,
                False,
                glide_dur_ms=glide_dur_hint_ms,
                v_ref_baseline=kr_cv_v_ref_baseline,
                cv_mode=kr_cv_timing_mode,
            )
        elif looks_like_vv_alias_fn(alias):
            if mel_ctx_for_file and estimate_mel_vowel_activity_span_fn is not None:
                try:
                    refined = estimate_mel_vowel_activity_span_fn(
                        mel_ctx_for_file,
                        n_start,
                        n_end,
                        search_pad_ms=120.0,
                        min_span_ms=20.0,
                    )
                    if refined is not None:
                        rv_start, rv_end = refined
                        if rv_end > rv_start and (
                            abs(float(rv_start) - float(n_start)) >= 8.0
                            or abs(float(rv_end) - float(n_end)) >= 16.0
                        ):
                            if log_vv_refine_fn is not None:
                                log_vv_refine_fn(
                                    fname,
                                    alias,
                                    float(n_start),
                                    float(n_end),
                                    float(rv_start),
                                    float(rv_end),
                                )
                            n_start, n_end = float(rv_start), float(rv_end)
                except Exception:
                    pass

            vv_direct = None
            if file_format == "cvvc":
                vv_direct = compute_cvvc_vv_timing_direct_fn(
                    selected_w_idx,
                    syllables_info,
                    n_start,
                    n_end,
                )
            elif selected_w_idx is not None and selected_w_idx >= 1 and selected_w_idx < len(syllables_info):
                prev_syl = syllables_info[selected_w_idx - 1]
                prev_phones = prev_syl.get("phones") or []
                if prev_phones:
                    _pv_idx, prev_v_phone = find_vowel_phone_fn(prev_phones)
                    prev_v_start = float(prev_v_phone.minTime) * 1000.0
                    prev_v_end = float(prev_v_phone.maxTime) * 1000.0
                    vv_direct = compute_vv_timing_from_vowel_bounds_fn(
                        prev_v_start,
                        prev_v_end,
                        n_start,
                        n_end,
                    )
            if vv_direct is not None:
                offset, consonant, cutoff, pre, ovl = vv_direct
            else:
                offset, consonant, cutoff, pre, ovl = compute_noninitial_vowel_timing_fn(
                    n_start,
                    n_end,
                )
        else:
            first_phone_plosive = bool(curr_phones) and is_plosive_ipa_fn(c_hint)
            alias_consonant = re.match(r"^([^aeiouyw]+)", str(alias).lower())
            roman_plosive = (
                bool(alias_consonant)
                and is_plosive_roman_fn(alias_consonant.group(1))
            )
            alias_onset = alias_consonant.group(1) if alias_consonant else ""
            is_plosive = first_phone_plosive or roman_plosive
            offset, consonant, cutoff, pre, ovl = compute_cv_timing_fn(
                c_start,
                c_end,
                cv_vowel_len,
                c_hint,
                alias_onset,
                False,
                is_plosive,
                v_ref_baseline=kr_cv_v_ref_baseline,
                cv_mode=kr_cv_timing_mode,
            )

    return {
        "offset": float(offset),
        "consonant": float(consonant),
        "cutoff": float(cutoff),
        "pre": float(pre),
        "ovl": float(ovl),
        "n_start": float(n_start),
        "n_end": float(n_end),
        "cv_vowel_len": float(n_end) - float(n_start),
        "is_vc_plosive_coda": bool(is_vc_plosive_coda),
    }


__all__ = ["compute_kr_general_base_timing"]
