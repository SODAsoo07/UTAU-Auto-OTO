from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Dict


def analyze_kr_file_alias_families(
    lines: Sequence[str],
    *,
    classify_alias_fn: Callable[[str], str],
) -> Dict[str, bool]:
    has_explicit_vc_alias = False
    has_cv_family_alias = False

    for line in lines or []:
        parts = str(line).split("=", 1)
        if len(parts) < 2:
            continue
        alias = parts[1].split(",", 1)[0].strip()
        if not alias:
            continue
        alias_type = str(classify_alias_fn(alias) or "").strip().lower()
        if alias_type == "vc":
            has_explicit_vc_alias = True
        elif alias_type in {"cv", "cv_head", "vcv", "mono"}:
            has_cv_family_alias = True

    return {
        "file_has_explicit_vc_alias": bool(has_explicit_vc_alias),
        "file_has_cv_family_alias": bool(has_cv_family_alias),
    }


def resolve_kr_cv_timing_mode(
    *,
    file_format: str,
    file_has_explicit_vc_alias: bool,
    file_has_cv_family_alias: bool,
) -> Dict[str, Any]:
    format_norm = str(file_format or "").strip().lower()
    has_mixed_cv_vc = bool(file_has_explicit_vc_alias) and (
        bool(file_has_cv_family_alias) or format_norm in {"cvvc", "cvc", "vcv"}
    )
    if has_mixed_cv_vc:
        timing_mode = "vcv_context" if format_norm == "vcv" else "vc_context"
    else:
        timing_mode = "standalone"
    return {
        "format_norm": format_norm,
        "file_has_mixed_cv_vc": bool(has_mixed_cv_vc),
        "kr_cv_timing_mode": timing_mode,
    }


def prepare_kr_row_iteration_context(
    *,
    lines: Sequence[str],
    file_format: str,
    syllables_info: Sequence[MutableMapping[str, Any]],
    ph_intervals: Sequence[Any],
    filename_cv_targets: Sequence[Any],
    disable_cvvc_order_lock: bool,
    debug_logging: bool,
    fname: str,
    log_fn: Callable[[str], None],
    is_order_locked_format_fn: Callable[[str], bool],
    build_occurrence_map_fn: Callable[[Sequence[Any]], Any],
    build_vv_occurrence_map_fn: Callable[[Sequence[Any]], Any],
    classify_alias_fn: Callable[[str], str],
    compute_vowel_baseline_fn: Callable[[Sequence[Mapping[str, Any]]], float],
    estimate_cv_anchor_fn: Callable[..., Any],
) -> Dict[str, Any]:
    romaji_syllables = [
        syllable.get("roman_cv") or syllable.get("roman", "")
        for syllable in syllables_info
    ]
    kr_order_locked_format = bool(is_order_locked_format_fn(file_format))
    if kr_order_locked_format and disable_cvvc_order_lock:
        kr_order_locked_format = False
        if debug_logging:
            log_fn(f"[MAP] {fname}: KR CVVC/CVC filename order lock 비활성(UTOA_KR_DISABLE_CVVC_ORDER_LOCK=1)")

    occurrence_source = (
        filename_cv_targets
        if (kr_order_locked_format and filename_cv_targets)
        else syllables_info
    )
    cvvc_occurrence_map = (
        build_occurrence_map_fn(occurrence_source)
        if kr_order_locked_format
        else None
    )
    vv_occurrence_map = (
        build_vv_occurrence_map_fn(occurrence_source)
        if str(file_format or "").strip().lower() in {"cvvc", "vcv"}
        else None
    )

    alias_mix = analyze_kr_file_alias_families(
        lines,
        classify_alias_fn=classify_alias_fn,
    )
    timing_context = resolve_kr_cv_timing_mode(
        file_format=file_format,
        file_has_explicit_vc_alias=alias_mix["file_has_explicit_vc_alias"],
        file_has_cv_family_alias=alias_mix["file_has_cv_family_alias"],
    )

    cv_v_ref_baseline = float(compute_vowel_baseline_fn(syllables_info))
    for syllable in syllables_info or []:
        syllable["v_ref_baseline_ms"] = cv_v_ref_baseline

    if debug_logging:
        log_fn(
            f"[MAP] {fname}: KR CV timing mode={timing_context['kr_cv_timing_mode']} "
            f"(format={timing_context['format_norm'] or 'unknown'}, "
            f"vc_alias={'yes' if alias_mix['file_has_explicit_vc_alias'] else 'no'}, "
            f"cv_family={'yes' if alias_mix['file_has_cv_family_alias'] else 'no'}, "
            f"v_ref={cv_v_ref_baseline:.1f}ms)"
        )

    cv_anchor_by_idx = {
        i: estimate_cv_anchor_fn(
            syllables_info[i],
            ph_intervals,
            cv_mode=timing_context["kr_cv_timing_mode"],
        )
        for i in range(len(syllables_info))
    }

    return {
        "romaji_syllables": romaji_syllables,
        "kr_order_locked_format": bool(kr_order_locked_format),
        "kr_cvvc_occurrence_map": cvvc_occurrence_map,
        "kr_cvvc_occurrence_state": {},
        "kr_cvvc_vv_occurrence_map": vv_occurrence_map,
        "kr_cvvc_vv_occurrence_state": {},
        "file_has_explicit_vc_alias": alias_mix["file_has_explicit_vc_alias"],
        "file_has_cv_family_alias": alias_mix["file_has_cv_family_alias"],
        "file_has_mixed_cv_vc": timing_context["file_has_mixed_cv_vc"],
        "kr_cv_timing_mode": timing_context["kr_cv_timing_mode"],
        "kr_cv_v_ref_baseline": cv_v_ref_baseline,
        "cv_anchor_by_idx": cv_anchor_by_idx,
        "realized_cv_anchor_by_idx": {},
    }


__all__ = [
    "analyze_kr_file_alias_families",
    "prepare_kr_row_iteration_context",
    "resolve_kr_cv_timing_mode",
]
