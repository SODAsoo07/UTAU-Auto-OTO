from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, Optional


def run_kr_general_post_timing_adjustments(
    *,
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    alias: str,
    alias_type: str,
    file_format: str,
    is_vc: bool,
    selected_w_idx: Optional[int],
    current_w_idx: int,
    curr_phones: Sequence[Any],
    c_start: float,
    c_end: float,
    n_start: float,
    n_end: float,
    row_mapping_confidence: float,
    row_apply_mode: str,
    row_apply_reason_code: str,
    delta_clamp_scale: float,
    pitch_zone: str,
    base_shape: Optional[Dict[str, float]],
    mel_ctx_for_file: Any,
    syllables_info: Sequence[Mapping[str, Any]],
    ph_intervals: Sequence[Any],
    is_vc_plosive_coda: bool,
    kr_post_ctx: Any,
    alignment_weight: float,
    textgrid_trust_tier: str,
    textgrid_trust_score: float,
    mel_reliability_score: float,
    mel_reliability_floor: float,
    fname: str,
    debug_logging: bool,
    log_fn: Callable[[str], None],
    env_bool_fn: Callable[[str, bool], bool],
    resolve_mel_onset_weight_fn: Callable[..., float],
    estimate_mel_voiced_onset_fn: Callable[[Any, float], Optional[float]],
    apply_mel_voiced_onset_pre_shift_fn: Callable[..., tuple[float, float, float, float, float, float]],
    debug_offset_trace_fn: Callable[..., None],
    apply_post_timing_pipeline_fn: Callable[..., tuple[float, float, float, float, float, float, float, float]],
    extract_alias_onset_fn: Callable[[str], str],
    ensure_cv_min_vowel_coverage_fn: Callable[..., tuple[float, float, float, float, Any]],
    guard_cv_focus_window_fn: Callable[..., tuple[float, float, float, float, float, float, float]],
    apply_mel_naturalness_adjustments_fn: Callable[..., tuple[float, float, float, float, float, Mapping[str, Any]]],
    apply_conservative_delta_clamp_fn: Callable[..., tuple[float, float, float, float, float]],
) -> Dict[str, Any]:
    row_mel_voiced_onset_ms = None

    if mel_ctx_for_file and alias_type == "cv":
        pre_abs = float(offset) + float(pre)
        mel_weight = resolve_mel_onset_weight_fn(
            alignment_weight,
            textgrid_trust_tier,
            trust_score=textgrid_trust_score,
            mapping_confidence=row_mapping_confidence,
            mel_reliability=mel_reliability_score,
            mel_reliability_floor=mel_reliability_floor,
        )
        if mel_weight > 0.0 and not env_bool_fn("UTOA_DISABLE_MEL_ONSET_SHIFT", False):
            mel_onset = estimate_mel_voiced_onset_fn(mel_ctx_for_file, pre_abs)
            if mel_onset is not None and abs(float(mel_onset) - pre_abs) <= 120.0:
                before = (offset, pre, consonant, cutoff)
                (
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    _mel_shift,
                ) = apply_mel_voiced_onset_pre_shift_fn(
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    mel_onset,
                    weight=mel_weight,
                )
                row_mel_voiced_onset_ms = float(mel_onset)
                debug_offset_trace_fn(
                    log_fn,
                    "mel_onset_shift",
                    fname,
                    alias,
                    before,
                    (offset, pre, consonant, cutoff),
                    extra=f"mel={row_mel_voiced_onset_ms:.1f}",
                )

    if alias_type == "cv" and selected_w_idx is not None and 0 <= int(selected_w_idx) < len(syllables_info):
        row_syl = syllables_info[int(selected_w_idx)] or {}
        onset_dist = float(row_syl.get("mel_onset_distance_ms", 999.0) or 999.0)
        onset_energy = float(row_syl.get("mel_onset_energy", 0.0) or 0.0)
        if onset_dist < 40.0 and onset_energy > 0.20:
            ovl_cap = max(0.0, float(pre) - onset_dist - 8.0)
            ovl = max(0.0, min(float(ovl), ovl_cap))

    before_post = (offset, pre, consonant, cutoff)
    (
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        soft_off_shift,
        soft_cut_shift,
        cutoff_reduced,
    ) = apply_post_timing_pipeline_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias,
        file_format=file_format,
        mel_ctx_for_file=mel_ctx_for_file,
        base_shape=base_shape,
        ph_intervals=ph_intervals,
        current_w_idx=selected_w_idx if not is_vc else current_w_idx,
        syllables_info=syllables_info,
        is_vc_plosive_coda=is_vc_plosive_coda,
        enable_stabilize=True,
        enable_cutoff_guard=True,
        post_ctx=kr_post_ctx,
    )
    debug_offset_trace_fn(
        log_fn,
        "post_timing",
        fname,
        alias,
        before_post,
        (offset, pre, consonant, cutoff),
    )

    if (
        alias_type == "cv"
        and str(file_format or "").strip().lower() in {"cvvc", "cvc"}
        and env_bool_fn("UTOA_KR_CV_MIN_VOWEL_GUARD", True)
    ):
        alias_onset = extract_alias_onset_fn(alias)
        ipa_onset = curr_phones[0].mark if curr_phones else ""
        offset, consonant, cutoff, pre, _cv_vowel_extended = ensure_cv_min_vowel_coverage_fn(
            offset,
            consonant,
            cutoff,
            pre,
            n_start,
            n_end,
            alias_onset=alias_onset,
            ipa_onset=ipa_onset,
        )

    if alias_type == "cv" and not env_bool_fn("UTOA_DISABLE_CV_FOCUS_GUARD", False):
        alias_onset = extract_alias_onset_fn(alias)
        ipa_onset = curr_phones[0].mark if curr_phones else ""
        before_focus = (offset, pre, consonant, cutoff)
        (
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            cv_offset_pulled,
            cv_cutoff_trimmed,
        ) = guard_cv_focus_window_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            c_start,
            c_end,
            n_start,
            n_end,
            alias_onset=alias_onset,
            ipa_onset=ipa_onset,
        )
        debug_offset_trace_fn(
            log_fn,
            "cv_focus_guard",
            fname,
            alias,
            before_focus,
            (offset, pre, consonant, cutoff),
        )
        if (cv_offset_pulled >= 0.8) or (cv_cutoff_trimmed >= 0.8):
            log_fn(
                f"[MAP] {fname}: CV 포커스 가드 적용 "
                f"(offset -{cv_offset_pulled:.1f}ms, cutoff -{cv_cutoff_trimmed:.1f}ms) [{alias}]"
            )

    if row_apply_mode != "template_preserve":
        before_nat = (offset, pre, consonant, cutoff)
        (
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            nat_info,
        ) = apply_mel_naturalness_adjustments_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type=alias_type,
            c_start_ms=c_start,
            c_end_ms=c_end,
            vowel_start_ms=n_start,
            vowel_end_ms=n_end,
            mel_ctx=mel_ctx_for_file,
        )
        if nat_info:
            detail = ", ".join(
                f"{k}={float(v):.1f}"
                for k, v in nat_info.items()
                if isinstance(v, (int, float))
            )
            debug_offset_trace_fn(
                log_fn,
                "mel_naturalness",
                fname,
                alias,
                before_nat,
                (offset, pre, consonant, cutoff),
                extra=detail,
            )

    if row_apply_mode in {"conservative_apply", "template_preserve"}:
        offset, consonant, cutoff, pre, ovl = apply_conservative_delta_clamp_fn(
            mode=row_apply_mode,
            delta_clamp_scale=delta_clamp_scale,
            pitch_zone=pitch_zone,
            base_shape=base_shape,
            offset=offset,
            consonant=consonant,
            cutoff=cutoff,
            pre=pre,
            ovl=ovl,
        )
        if debug_logging:
            log_fn(
                f"[ROW] {fname}: mode={row_apply_mode} reason={row_apply_reason_code or '-'} "
                f"delta_scale={delta_clamp_scale:.2f} alias={alias}"
            )

    return {
        "offset": float(offset),
        "consonant": float(consonant),
        "cutoff": float(cutoff),
        "pre": float(pre),
        "ovl": float(ovl),
        "soft_off_shift": float(soft_off_shift),
        "soft_cut_shift": float(soft_cut_shift),
        "cutoff_reduced": float(cutoff_reduced),
        "row_mel_voiced_onset_ms": row_mel_voiced_onset_ms,
    }


__all__ = ["run_kr_general_post_timing_adjustments"]
