from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Dict


def prepare_kr_general_cv_selection_context(
    *,
    alias: str,
    alias_type: str,
    fname: str,
    file_format: str,
    target_clean: str,
    current_w_idx: int,
    cv_seq_idx: int,
    bridge_pair: Mapping[str, Any],
    row_mapping_confidence: float,
    row_jump_default: int,
    row_jump_high_conf: int,
    file_mapping_conf_th: float,
    file_mapping_low_conf: bool,
    romaji_syllables: Sequence[str],
    kr_planned_cv_indices: Mapping[Any, Any],
    kr_cvvc_occurrence_map: Mapping[Any, Any],
    kr_cvvc_occurrence_state: MutableMapping[Any, Any],
    kr_cvvc_vv_occurrence_map: Mapping[Any, Any],
    kr_cvvc_vv_occurrence_state: MutableMapping[Any, Any],
    syllable_blank_confidences: Sequence[float],
    syllables_info: Sequence[Mapping[str, Any]],
    mapping_margin: float,
    row_margin_floor: float,
    row_conf_floor: float,
    row_blank_floor: Any,
    debug_logging: bool,
    log_fn: Callable[[str], None],
    resolve_vv_index_fn: Callable[..., Any],
    resolve_cvvc_occurrence_index_fn: Callable[..., Any],
    resolve_planned_cv_index_fn: Callable[..., Any],
    select_general_cv_index_fn: Callable[..., Mapping[str, Any]],
    remap_forced_cv_index_fn: Callable[..., Any],
    cv_match_score_fn: Callable[..., Any],
    split_syllable_parts_fn: Callable[..., Any],
    apply_row_confidence_penalty_fn: Callable[..., Any],
    resolve_cv_syllable_index_fn: Callable[..., Any],
    should_allow_exact_vowel_fix_fn: Callable[..., Any],
    find_cv_vowel_match_index_fn: Callable[..., Any],
    clamp_cv_index_to_order_fn: Callable[..., Any],
    blank_conf_at_fn: Callable[[Sequence[float], Any], Any],
    is_cv_syllable_active_fn: Callable[..., bool],
    decide_cv_row_abstain_fn: Callable[..., Mapping[str, Any]],
) -> Dict[str, Any]:
    forced_vv_idx = None
    planned_vv_idx = None
    if str(file_format or "").strip().lower() in {"cvvc", "vcv"} and alias_type == "vv":
        forced_vv_idx = resolve_vv_index_fn(
            alias,
            kr_cvvc_vv_occurrence_map or {},
            kr_cvvc_vv_occurrence_state,
        )
    if alias_type == "vv" and bridge_pair.get("next_idx") is not None:
        planned_vv_idx = int(bridge_pair["next_idx"])

    forced_cvvc_idx = resolve_cvvc_occurrence_index_fn(
        alias,
        alias_type,
        kr_cvvc_occurrence_map or {},
        kr_cvvc_occurrence_state,
        expected_idx=cv_seq_idx,
    )

    expected_cv_idx = int(cv_seq_idx)
    planned_cv_idx = None
    if alias_type == "cv":
        planned_cv_idx = resolve_planned_cv_index_fn(
            kr_planned_cv_indices,
            expected_cv_idx,
            target_clean,
            syllables_info,
            alias_type="cv",
        )
        if planned_cv_idx is not None and debug_logging and int(planned_cv_idx) != expected_cv_idx:
            log_fn(
                f"[MAP] {fname}: KR CV 앵커 계획 보정 "
                f"({expected_cv_idx + 1}->{int(planned_cv_idx) + 1}, {alias})"
            )

    general_cv_selection = select_general_cv_index_fn(
        alias=alias,
        alias_type=alias_type,
        fname=fname,
        file_format=file_format,
        target_clean=target_clean,
        current_w_idx=current_w_idx,
        cv_seq_idx=cv_seq_idx,
        row_mapping_confidence=row_mapping_confidence,
        row_jump_default=row_jump_default,
        row_jump_high_conf=row_jump_high_conf,
        file_mapping_conf_th=file_mapping_conf_th,
        file_mapping_low_conf=file_mapping_low_conf,
        romaji_syllables=romaji_syllables,
        forced_vv_idx=forced_vv_idx,
        planned_vv_idx=planned_vv_idx,
        planned_cv_idx=planned_cv_idx,
        forced_cvvc_idx=forced_cvvc_idx,
        remap_forced_cv_index_fn=remap_forced_cv_index_fn,
        cv_match_score_fn=cv_match_score_fn,
        split_syllable_parts_fn=split_syllable_parts_fn,
        apply_row_confidence_penalty_fn=apply_row_confidence_penalty_fn,
        resolve_cv_syllable_index_fn=resolve_cv_syllable_index_fn,
        should_allow_exact_vowel_fix_fn=should_allow_exact_vowel_fix_fn,
        find_cv_vowel_match_index_fn=find_cv_vowel_match_index_fn,
        clamp_cv_index_to_order_fn=clamp_cv_index_to_order_fn,
        syllable_blank_confidences=syllable_blank_confidences,
        syllables_info=syllables_info,
        log_fn=log_fn,
        debug_logging=debug_logging,
    )

    selected_w_idx = int(general_cv_selection["selected_w_idx"])
    cv_seq_idx = int(general_cv_selection["cv_seq_idx"])
    row_mapping_confidence = float(general_cv_selection["row_mapping_confidence"])
    row_jump_blocked = int(general_cv_selection["row_jump_blocked"])
    forced_selected_idx = general_cv_selection["forced_selected_idx"]
    row_blank_floor_safe = (
        float(row_blank_floor)
        if row_blank_floor is not None
        else 0.62
    )
    row_blank_conf = blank_conf_at_fn(syllable_blank_confidences, selected_w_idx)
    row_abstain = decide_cv_row_abstain_fn(
        alias_type=alias_type,
        alias_text=alias,
        format_type=file_format,
        candidate_idx=selected_w_idx,
        candidate_count=len(syllables_info),
        candidate_active=(
            is_cv_syllable_active_fn(syllables_info[selected_w_idx], require_vowel=True)
            if selected_w_idx is not None and 0 <= selected_w_idx < len(syllables_info)
            else False
        ),
        confidence_margin=mapping_margin,
        min_confidence_margin=row_margin_floor,
        row_confidence=row_mapping_confidence,
        min_row_confidence=row_conf_floor,
        blank_confidence=row_blank_conf,
        max_blank_confidence=row_blank_floor_safe,
        active_only_formats={"cvvc", "cv"},
        margin_formats={"cvvc", "cv"},
        blank_formats={"cvvc", "cvc", "cv"},
        min_confidence_margin_by_alias_type={"cv_head": row_margin_floor + 1.5, "vcv": row_margin_floor + 1.0},
        min_row_confidence_by_alias_type={"cv_head": row_conf_floor + 0.03, "vcv": row_conf_floor + 0.02},
        max_blank_confidence_by_alias_type={
            "cv_head": max(0.0, row_blank_floor_safe - 0.03),
            "vcv": max(0.0, row_blank_floor_safe - 0.02),
        },
    )

    return {
        "expected_cv_idx": int(general_cv_selection["expected_cv_idx"]),
        "selected_w_idx": int(selected_w_idx),
        "cv_seq_idx": int(cv_seq_idx),
        "row_mapping_confidence": float(row_mapping_confidence),
        "resolve_meta": dict(general_cv_selection.get("resolve_meta") or {}),
        "row_jump_blocked": int(row_jump_blocked),
        "forced_selected_idx": forced_selected_idx,
        "row_blank_floor_safe": float(row_blank_floor_safe),
        "row_blank_conf": row_blank_conf,
        "row_abstain": dict(row_abstain),
        "forced_vv_idx": forced_vv_idx,
        "planned_vv_idx": planned_vv_idx,
        "planned_cv_idx": planned_cv_idx,
        "forced_cvvc_idx": forced_cvvc_idx,
    }


__all__ = ["prepare_kr_general_cv_selection_context"]
