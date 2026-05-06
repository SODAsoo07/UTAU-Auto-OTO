from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Dict


def build_kr_plan_candidate_source(
    *,
    filename_cv_targets: Sequence[str],
    syllables_info: Sequence[Mapping[str, Any]],
) -> list[str]:
    return list(
        filename_cv_targets
        or [s.get("roman_cv") or s.get("roman", "") for s in (syllables_info or [])]
        or []
    )


def should_enable_kr_mel_plan(
    *,
    file_format: str,
    mel_ctx_for_file: Mapping[str, Any] | None,
    syllables_info: Sequence[Mapping[str, Any]],
    textgrid_trust_tier: str,
    low_phone_quality: bool,
    blank_conf_mean: float,
    alignment_weight: float,
    env_bool_fn: Callable[[str, bool], bool],
) -> bool:
    fmt = str(file_format or "").strip().lower()
    mel_plan_env = {
        "cvvc": "UTOA_KR_CVVC_MEL_PLAN",
        "cvc": "UTOA_KR_CVC_MEL_PLAN",
        "vcv": "UTOA_KR_VCV_MEL_PLAN",
        "cv": "UTOA_KR_CV_MEL_PLAN",
    }
    if fmt in mel_plan_env and mel_ctx_for_file and syllables_info:
        if (
            str(textgrid_trust_tier or "") != "high"
            or bool(low_phone_quality)
            or float(blank_conf_mean) >= 0.45
            or float(alignment_weight) < 0.65
        ):
            return bool(env_bool_fn(mel_plan_env[fmt], True))
    return False


def prepare_kr_mapping_runtime_context(
    *,
    syllables_info: Sequence[Mapping[str, Any]],
    sinsy_label_entries: Sequence[Mapping[str, Any]],
    filename_cv_targets: Sequence[str],
    file_format: str,
    alignment_ingest: Any,
    prefer_filename_sequence: bool,
    alignment_weight: float,
    phone_quality: Mapping[str, Any],
    used_words_based: bool,
    used_alias_based: bool,
    base_score: float,
    alt_score: float,
    file_mapping_conf_th: float,
    textgrid_trust_score: float,
    textgrid_trust_tier: str,
    low_phone_quality: bool,
    blank_conf_mean: float,
    syllable_blank_confidences: Sequence[float],
    mel_ctx_for_file: Mapping[str, Any] | None,
    mel_reliability_score: float,
    mel_reliability_floor: float,
    mel_weight_mode: str,
    mapping_reason_code: str,
    runtime_report: MutableMapping[str, Any] | None,
    fname: str,
    debug_logging: bool,
    estimate_mapping_confidence_fn: Callable[..., tuple[float, float]],
    recompute_common_plan_runtime_state_fn: Callable[..., Mapping[str, Any]],
    build_common_plan_context_fn: Callable[..., Mapping[str, Any]],
    resolve_runtime_mapping_policy_fn: Callable[..., Mapping[str, Any]],
    log_sinsy_plan_guard_fn: Callable[..., None],
    compute_runtime_low_conf_state_fn: Callable[..., Mapping[str, Any]],
    resolve_file_routing_profile_fn: Callable[..., Mapping[str, Any]],
    update_kr_mapping_runtime_report_fn: Callable[..., None],
    log_fn: Callable[[str], None],
    normalize_expected_fn: Callable[[str], str],
    normalize_label_fn: Callable[[str], str],
    cv_match_score_fn: Callable[[str, str], float],
    build_cv_anchor_plan_fn: Callable[..., Mapping[str, Any]],
    build_sinsy_guided_anchor_plan_fn: Callable[..., Mapping[str, Any]],
    build_adjacent_anchor_graph_fn: Callable[..., Any],
    resolve_plan_policy_fn: Callable[..., Mapping[str, Any]],
    env_bool_fn: Callable[[str, bool], bool],
) -> Dict[str, Any]:
    plan_candidate_source = build_kr_plan_candidate_source(
        filename_cv_targets=filename_cv_targets,
        syllables_info=syllables_info,
    )

    kr_plan_context_kwargs = {
        "planned_cv_source": plan_candidate_source,
        "syllables_info": syllables_info,
        "sinsy_label_entries": sinsy_label_entries,
        "format_type": file_format,
        "alignment_weight": alignment_weight,
        "prefer_sequence": prefer_filename_sequence,
        "normalize_expected_fn": normalize_expected_fn,
        "normalize_label_fn": normalize_label_fn,
        "label_match_score_fn": cv_match_score_fn,
        "should_enable_mel_plan_fn": lambda: should_enable_kr_mel_plan(
            file_format=file_format,
            mel_ctx_for_file=mel_ctx_for_file,
            syllables_info=syllables_info,
            textgrid_trust_tier=textgrid_trust_tier,
            low_phone_quality=low_phone_quality,
            blank_conf_mean=blank_conf_mean,
            alignment_weight=alignment_weight,
            env_bool_fn=env_bool_fn,
        ),
        "build_cv_anchor_plan_fn": build_cv_anchor_plan_fn,
        "build_sinsy_guided_anchor_plan_fn": build_sinsy_guided_anchor_plan_fn,
        "build_adjacent_anchor_graph_fn": build_adjacent_anchor_graph_fn,
        "resolve_plan_policy_fn": resolve_plan_policy_fn,
    }

    mapping_confidence, mapping_margin = estimate_mapping_confidence_fn(
        phone_quality,
        words_score=base_score,
        alias_score=alt_score,
        used_words_based=used_words_based,
        used_alias_based=used_alias_based,
    )
    kr_runtime_state = recompute_common_plan_runtime_state_fn(
        build_plan_context_fn=build_common_plan_context_fn,
        plan_context_kwargs=kr_plan_context_kwargs,
        ingest_snapshot=alignment_ingest,
        mapping_confidence=mapping_confidence,
        mapping_margin=mapping_margin,
        conf_threshold=file_mapping_conf_th,
        format_type=file_format,
        score_a=base_score,
        score_b=alt_score,
        sequence_lock_formats={"cvvc", "cvc", "vcv"},
        abstain_formats={"cvvc", "vcv", "cvc", "cv"},
        strict_formats={"cvvc", "vcv"},
        prefer_sequence=prefer_filename_sequence,
        alignment_trust=alignment_weight,
        resolve_runtime_mapping_policy_fn=resolve_runtime_mapping_policy_fn,
    )
    kr_cv_plan = dict(kr_runtime_state.get("cv_plan") or {})
    kr_planned_cv_indices = kr_runtime_state.get("planned_indices")
    kr_anchor_graph = kr_runtime_state.get("anchor_graph")
    kr_plan_policy = dict(kr_runtime_state.get("plan_policy") or {})
    runtime_policy = dict(kr_runtime_state.get("runtime_policy") or {})
    mapping_confidence = float(kr_runtime_state.get("mapping_confidence", mapping_confidence) or 0.0)
    mapping_margin = float(kr_runtime_state.get("mapping_margin", mapping_margin) or 0.0)

    log_sinsy_plan_guard_fn(
        sinsy_label_entries=sinsy_label_entries,
        cv_plan=kr_cv_plan,
        runtime_policy=runtime_policy,
        fname=fname,
        log_fn=log_fn,
        log_prefix="[MAP]",
    )

    mapping_confidence = float(runtime_policy.get("mapping_confidence", mapping_confidence))
    if syllable_blank_confidences and float(blank_conf_mean) >= 0.55:
        mapping_confidence = max(
            0.0,
            mapping_confidence - min(0.22, (float(blank_conf_mean) - 0.55) * 0.45 + 0.04),
        )
    if debug_logging and mapping_confidence < float(file_mapping_conf_th):
        log_fn(
            f"[MAP] {fname}: KR 매핑 신뢰도 낮음(conf={mapping_confidence:.2f}, "
            f"margin={mapping_margin:+.1f}, reason={mapping_reason_code})"
        )

    file_conf_floor = float(runtime_policy.get("file_conf_floor", file_mapping_conf_th))
    low_conf_state = dict(
        compute_runtime_low_conf_state_fn(
            runtime_policy=runtime_policy,
            mapping_confidence=mapping_confidence,
            conf_floor=file_conf_floor,
            blank_confidence_mean=blank_conf_mean,
            conf_below_reason="conf_below_floor",
            row_conf_floor_default=float(file_mapping_conf_th),
        )
    )
    file_mapping_low_conf = bool(low_conf_state.get("file_low_conf"))
    row_conf_floor = float(low_conf_state.get("row_conf_floor", file_mapping_conf_th) or file_mapping_conf_th)
    row_margin_floor = float(low_conf_state.get("row_margin_floor", 6.0) or 6.0)
    low_conf_reasons = list(low_conf_state.get("low_conf_reasons") or [])

    pitch_zone_for_file = str(
        mel_ctx_for_file.get("f0_pitch_zone", "") if mel_ctx_for_file else ""
    ).strip().lower()
    note_hint_hz_for_file = (
        float(mel_ctx_for_file.get("f0_note_hint_hz", 0.0) or 0.0)
        if mel_ctx_for_file
        else 0.0
    )
    f0_max_hz_for_file = (
        float(mel_ctx_for_file.get("f0_max_hz", 0.0) or 0.0)
        if mel_ctx_for_file
        else 0.0
    )

    routing_profile = dict(
        resolve_file_routing_profile_fn(
            file_format=str(file_format or ""),
            pitch_zone=pitch_zone_for_file,
            note_hint_hz=note_hint_hz_for_file,
            f0_max_hz=f0_max_hz_for_file,
            textgrid_trust_score=float(textgrid_trust_score),
            textgrid_trust_tier=str(textgrid_trust_tier or ""),
            mapping_confidence_base=float(mapping_confidence),
            blank_conf_mean=float(blank_conf_mean),
            mel_reliability_score=float(mel_reliability_score),
            mel_reliability_floor=float(mel_reliability_floor),
            file_mapping_low_conf=bool(file_mapping_low_conf),
            alignment_weight_base=float(alignment_weight),
            mel_weight_mode=str(mel_weight_mode or ""),
            file_mapping_conf_th=float(file_mapping_conf_th),
        )
    )
    force_mel_branch = bool(routing_profile.get("force_mel_branch"))
    resolved_mel_weight_mode = str(routing_profile.get("mel_weight_mode") or mel_weight_mode)
    alignment_weight_final = float(routing_profile.get("alignment_weight_final", alignment_weight))
    anchor_lock_lite = bool(routing_profile.get("anchor_lock_lite_default", False))
    row_blank_floor = routing_profile.get("row_blank_floor")
    delta_clamp_scale = float(routing_profile.get("delta_clamp_scale", 1.0))
    routing_profile_code = str(routing_profile.get("profile_code") or "hybrid_soft")

    log_fn(
        f"[ROUTING] file={fname} profile={routing_profile_code} "
        f"trust={float(textgrid_trust_score):.2f} blank={float(blank_conf_mean):.2f} "
        f"mel_rel={float(mel_reliability_score):.2f} align={alignment_weight_final:.2f}"
    )
    log_fn(
        f"[MAP] {fname}: align_base={float(alignment_weight):.2f}, align_final={alignment_weight_final:.2f}, "
        f"blank_mean={float(blank_conf_mean):.2f}, map_conf={mapping_confidence:.2f}, "
        f"mel_rel={float(mel_reliability_score):.2f}/{float(mel_reliability_floor):.2f}, "
        f"anchor_lock_lite={anchor_lock_lite}, "
        f"mel_weight_mode={resolved_mel_weight_mode}, force_mel_branch={force_mel_branch}, "
        f"delta_clamp_scale={delta_clamp_scale:.2f}"
    )

    if isinstance(runtime_report, MutableMapping):
        update_kr_mapping_runtime_report_fn(
            runtime_report,
            file_format=str(file_format or ""),
            mapping_confidence=float(mapping_confidence),
            mapping_margin=float(mapping_margin),
            mapping_tier=str(runtime_policy.get("mapping_tier") or ""),
            trust_score=float(textgrid_trust_score),
            trust_tier=str(textgrid_trust_tier or ""),
            file_conf_floor=float(file_conf_floor),
            row_conf_floor=float(row_conf_floor),
            row_margin_floor=float(row_margin_floor),
            file_low_conf=bool(file_mapping_low_conf),
            low_conf_reasons=list(low_conf_reasons),
            blank_confidence_mean=float(blank_conf_mean),
            plan_source=str(kr_cv_plan.get("source") or ""),
            plan_margin=float((kr_cv_plan.get("meta") or {}).get("margin", 0.0) or 0.0),
            plan_coverage=float(kr_plan_policy.get("coverage", 0.0) or 0.0),
            mapping_reason_code=str(mapping_reason_code or ""),
            row_blank_floor=(
                float(row_blank_floor)
                if row_blank_floor is not None
                else None
            ),
            extra_fields={
                "routing_profile": str(routing_profile_code),
                "alignment_weight_base": float(alignment_weight),
                "alignment_weight": float(alignment_weight_final),
                "mel_reliability_score": float(mel_reliability_score),
                "mel_reliability_floor": float(mel_reliability_floor),
                "mel_weight_mode": str(resolved_mel_weight_mode),
                "force_mel_branch": bool(force_mel_branch),
                "anchor_lock_lite": bool(anchor_lock_lite),
                "delta_clamp_scale": float(delta_clamp_scale),
                "high_pitch_mode": bool(routing_profile.get("high_pitch_mode", False)),
            },
        )

    return {
        "plan_candidate_source": plan_candidate_source,
        "kr_cv_plan": kr_cv_plan,
        "kr_planned_cv_indices": kr_planned_cv_indices,
        "kr_anchor_graph": kr_anchor_graph,
        "kr_plan_policy": kr_plan_policy,
        "runtime_policy": runtime_policy,
        "mapping_confidence_base": float(mapping_confidence),
        "mapping_margin": float(mapping_margin),
        "file_conf_floor": float(file_conf_floor),
        "low_conf_state": low_conf_state,
        "file_mapping_low_conf": bool(file_mapping_low_conf),
        "row_conf_floor": float(row_conf_floor),
        "row_margin_floor": float(row_margin_floor),
        "low_conf_reasons": low_conf_reasons,
        "pitch_zone_for_file": pitch_zone_for_file,
        "note_hint_hz_for_file": float(note_hint_hz_for_file),
        "f0_max_hz_for_file": float(f0_max_hz_for_file),
        "routing_profile": routing_profile,
        "force_mel_branch": bool(force_mel_branch),
        "mel_weight_mode": str(resolved_mel_weight_mode),
        "alignment_weight": float(alignment_weight_final),
        "anchor_lock_lite": bool(anchor_lock_lite),
        "row_blank_floor": row_blank_floor,
        "delta_clamp_scale": float(delta_clamp_scale),
        "routing_profile_code": routing_profile_code,
    }


__all__ = [
    "build_kr_plan_candidate_source",
    "prepare_kr_mapping_runtime_context",
    "should_enable_kr_mel_plan",
]
