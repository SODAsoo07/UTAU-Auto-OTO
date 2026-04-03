from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


def _clip01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def update_single_vowel_span_by_first_phone(
    *,
    phone_intervals: Sequence[Any],
    tg_path: str,
    single_vowel_span_by_tg_path: Dict[str, tuple[float, float]],
    norm_tg_path_key_fn: Callable[[str], str],
) -> bool:
    if len(phone_intervals or []) != 1:
        return False
    try:
        one = list(phone_intervals)[0]
        key = norm_tg_path_key_fn(str(tg_path or ""))
        single_vowel_span_by_tg_path[key] = (
            float(one.minTime) * 1000.0,
            float(one.maxTime) * 1000.0,
        )
        return True
    except Exception:
        return False


def extract_ja_alignment_ingest_state(
    *,
    alignment_ingest,
    fallback_format: str,
    normalize_format_type_fn: Callable[[str, str], str],
    low_conf_lock_threshold: float = 0.55,
) -> Dict[str, Any]:
    filename_syllables = list(alignment_ingest.extra.get("filename_syllables") or [])
    cv_targets = list(alignment_ingest.extra.get("cv_targets") or [])
    detected_format_raw = str(alignment_ingest.extra.get("detected_format") or "")
    format_type_raw = str(alignment_ingest.extra.get("format_type") or "")
    detected_format = normalize_format_type_fn("japanese", detected_format_raw)
    format_type = normalize_format_type_fn("japanese", format_type_raw)
    if format_type not in {"cv", "cvvc", "vcv", "vc_only", "br"}:
        fallback_fmt = str(fallback_format or "").strip().lower()
        if detected_format in {"cv", "cvvc", "vcv", "vc_only", "br"}:
            format_type = detected_format
        elif fallback_fmt in {"cv", "cvvc", "vcv"}:
            format_type = fallback_fmt
        else:
            format_type = "cvvc"

    phone_quality = alignment_ingest.phone_quality
    textgrid_trust_score = float(alignment_ingest.textgrid_trust_score or 0.0)
    alignment_weight = _clip01(textgrid_trust_score * 0.85)
    prefer_filename_sequence = bool(alignment_ingest.prefer_filename_sequence)
    if filename_syllables and alignment_weight < float(low_conf_lock_threshold):
        prefer_filename_sequence = True

    return {
        "wd_intervals": alignment_ingest.words,
        "ph_intervals": alignment_ingest.phones,
        "filename_syllables": filename_syllables,
        "cv_targets": cv_targets,
        "sinsy_label_entries": list(alignment_ingest.extra.get("sinsy_label_entries") or []),
        "detected_format": detected_format,
        "format_type": format_type,
        "ja_style_profile": alignment_ingest.extra.get("ja_style_profile"),
        "phone_quality": phone_quality,
        "low_quality_reasons": alignment_ingest.low_quality_reasons,
        "low_phone_quality": alignment_ingest.low_phone_quality,
        "forced_words_mapping": bool(alignment_ingest.extra.get("forced_words_mapping")),
        "timeline_start_ms": float(alignment_ingest.timeline_meta.get("timeline_start_ms", 0.0) or 0.0),
        "effective_end_ms": float(alignment_ingest.timeline_meta.get("effective_end_ms", 0.0) or 0.0),
        "phone_spans_ms": list(alignment_ingest.timeline_meta.get("phone_spans_ms") or []),
        "conf_th": float(alignment_ingest.extra.get("conf_th", 0.0) or 0.0),
        "textgrid_trust_score": textgrid_trust_score,
        "textgrid_trust_tier": str(alignment_ingest.textgrid_trust_tier or "low"),
        "prefer_filename_sequence": prefer_filename_sequence,
        "spn_ratio": float(phone_quality.get("spn_ratio_in_phone_tier", 0.0) or 0.0),
        "alignment_weight": alignment_weight,
    }


def extract_kr_alignment_ingest_state(
    *,
    alignment_ingest,
    low_conf_lock_threshold: float = 0.55,
) -> Dict[str, Any]:
    phone_quality = alignment_ingest.phone_quality
    filename_cv_targets = list(alignment_ingest.extra.get("filename_cv_targets") or [])
    targets_for_build = list(alignment_ingest.extra.get("targets_for_build") or [])
    textgrid_trust_score = float(alignment_ingest.textgrid_trust_score or 0.0)
    alignment_weight = _clip01(textgrid_trust_score * 0.85)
    prefer_filename_sequence = bool(alignment_ingest.prefer_filename_sequence)
    if filename_cv_targets and alignment_weight < float(low_conf_lock_threshold):
        prefer_filename_sequence = True
        targets_for_build = list(filename_cv_targets)

    return {
        "ph_intervals_all": alignment_ingest.phones_all,
        "ph_intervals": alignment_ingest.phones,
        "wd_intervals": alignment_ingest.words,
        "wav_duration_ms": float(alignment_ingest.wav_duration_ms or 0.0),
        "timeline_start_ms": float(alignment_ingest.timeline_meta.get("timeline_start_ms", 0.0) or 0.0),
        "timeline_end_ms": float(alignment_ingest.timeline_meta.get("timeline_end_ms", 0.0) or 0.0),
        "file_format": str(alignment_ingest.extra.get("file_format") or ""),
        "file_mapping_conf_th": float(alignment_ingest.extra.get("file_mapping_conf_th", 0.0) or 0.0),
        "filename_cv_targets": filename_cv_targets,
        "targets_for_build": targets_for_build,
        "sinsy_label_entries": list(alignment_ingest.extra.get("sinsy_label_entries") or []),
        "phone_quality": phone_quality,
        "low_quality_reasons": alignment_ingest.low_quality_reasons,
        "low_phone_quality": alignment_ingest.low_phone_quality,
        "force_words_phone_fill": bool(alignment_ingest.extra.get("force_words_phone_fill")),
        "textgrid_trust_score": textgrid_trust_score,
        "textgrid_trust_tier": str(alignment_ingest.textgrid_trust_tier or "low"),
        "prefer_filename_sequence": prefer_filename_sequence,
        "spn_ratio": float(phone_quality.get("spn_ratio_in_phone_tier", 0.0) or 0.0),
        "alignment_weight": alignment_weight,
    }


def build_common_plan_context(
    *,
    planned_cv_source: Sequence[str],
    syllables_info,
    sinsy_label_entries: Sequence[Mapping[str, Any]],
    format_type: str,
    alignment_weight: float,
    prefer_sequence: bool,
    normalize_expected_fn: Callable[[str], str],
    normalize_label_fn: Callable[[str], str],
    label_match_score_fn: Callable[[str, str], float],
    should_enable_mel_plan_fn: Callable[[], bool],
    build_cv_anchor_plan_fn: Callable[..., Dict[str, Any]],
    build_sinsy_guided_anchor_plan_fn: Callable[..., Dict[str, Any]],
    build_adjacent_anchor_graph_fn: Callable[[Any], Any],
    resolve_plan_policy_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    source_tokens = list(planned_cv_source or [])
    cv_plan: Dict[str, Any] = {"indices": None, "meta": {}, "source": ""}
    if source_tokens and sinsy_label_entries:
        cv_plan = build_sinsy_guided_anchor_plan_fn(
            expected_tokens=source_tokens,
            syllables_info=syllables_info,
            label_entries=sinsy_label_entries,
            normalize_expected_fn=normalize_expected_fn,
            normalize_label_fn=normalize_label_fn,
            label_match_score_fn=label_match_score_fn,
        )
    if not cv_plan.get("indices"):
        use_mel_plan = bool(should_enable_mel_plan_fn())
        if source_tokens:
            try:
                cv_plan = build_cv_anchor_plan_fn(
                    source_tokens,
                    syllables_info,
                    use_mel=bool(use_mel_plan),
                    format_type=str(format_type or ""),
                )
            except TypeError:
                cv_plan = build_cv_anchor_plan_fn(
                    source_tokens,
                    syllables_info,
                    use_mel=bool(use_mel_plan),
                )
        else:
            cv_plan = {"indices": None, "meta": {}}
    planned_indices = cv_plan.get("indices")
    return {
        "cv_plan": cv_plan,
        "planned_indices": planned_indices,
        "anchor_graph": build_adjacent_anchor_graph_fn(planned_indices),
        "plan_policy": resolve_plan_policy_fn(
            alignment_trust=float(alignment_weight or 0.0),
            plan_meta=cv_plan.get("meta"),
            expected_count=len(source_tokens),
            planned_count=len(planned_indices or []),
            format_type=str(format_type or ""),
            prefer_sequence=bool(prefer_sequence),
        ),
    }


def resolve_common_runtime_policy(
    *,
    ingest_snapshot,
    plan_policy,
    mapping_confidence: float,
    mapping_margin: float,
    conf_threshold: float,
    format_type: str,
    score_a: float,
    score_b: float,
    sequence_lock_formats: Iterable[str],
    abstain_formats: Iterable[str],
    prefer_sequence: bool,
    alignment_trust: float,
    resolve_runtime_mapping_policy_fn: Callable[..., Dict[str, Any]],
    strict_formats: Iterable[str] | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "ingest_snapshot": ingest_snapshot,
        "plan_policy": plan_policy,
        "mapping_confidence": float(mapping_confidence or 0.0),
        "mapping_margin": float(mapping_margin or 0.0),
        "conf_threshold": float(conf_threshold or 0.0),
        "format_type": str(format_type or ""),
        "score_a": float(score_a or 0.0),
        "score_b": float(score_b or 0.0),
        "sequence_lock_formats": set(sequence_lock_formats or []),
        "abstain_formats": set(abstain_formats or []),
        "prefer_sequence": bool(prefer_sequence),
        "alignment_trust": float(alignment_trust or 0.0),
    }
    if strict_formats is not None:
        kwargs["strict_formats"] = set(strict_formats)
    return dict(resolve_runtime_mapping_policy_fn(**kwargs) or {})


def recompute_common_plan_runtime_state(
    *,
    build_plan_context_fn: Callable[..., Dict[str, Any]],
    plan_context_kwargs: Mapping[str, Any],
    ingest_snapshot,
    mapping_confidence: float,
    mapping_margin: float,
    conf_threshold: float,
    format_type: str,
    score_a: float,
    score_b: float,
    sequence_lock_formats: Iterable[str],
    abstain_formats: Iterable[str],
    prefer_sequence: bool,
    alignment_trust: float,
    resolve_runtime_mapping_policy_fn: Callable[..., Dict[str, Any]],
    strict_formats: Iterable[str] | None = None,
) -> Dict[str, Any]:
    plan_state = dict(build_plan_context_fn(**dict(plan_context_kwargs or {})) or {})
    cv_plan = dict(plan_state.get("cv_plan") or {})
    planned_indices = plan_state.get("planned_indices")
    anchor_graph = plan_state.get("anchor_graph")
    plan_policy = dict(plan_state.get("plan_policy") or {})

    runtime_policy = resolve_common_runtime_policy(
        ingest_snapshot=ingest_snapshot,
        plan_policy=plan_policy,
        mapping_confidence=float(mapping_confidence or 0.0),
        mapping_margin=float(mapping_margin or 0.0),
        conf_threshold=float(conf_threshold or 0.0),
        format_type=str(format_type or ""),
        score_a=float(score_a or 0.0),
        score_b=float(score_b or 0.0),
        sequence_lock_formats=set(sequence_lock_formats or []),
        abstain_formats=set(abstain_formats or []),
        strict_formats=(set(strict_formats) if strict_formats is not None else None),
        prefer_sequence=bool(prefer_sequence),
        alignment_trust=float(alignment_trust or 0.0),
        resolve_runtime_mapping_policy_fn=resolve_runtime_mapping_policy_fn,
    )
    runtime_conf = float(runtime_policy.get("mapping_confidence", mapping_confidence) or 0.0)
    runtime_tier = str(runtime_policy.get("mapping_tier") or "")
    return {
        "cv_plan": cv_plan,
        "planned_indices": planned_indices,
        "anchor_graph": anchor_graph,
        "plan_policy": plan_policy,
        "runtime_policy": dict(runtime_policy or {}),
        "mapping_confidence": float(runtime_conf),
        "mapping_margin": float(mapping_margin or 0.0),
        "mapping_tier": runtime_tier,
    }


def normalize_runtime_postprocess_state(
    postprocess: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    data = dict(postprocess or {})
    row_blank_floor = data.get("row_blank_floor")
    return {
        "file_mapping_low_conf": bool(data.get("file_mapping_low_conf", False)),
        "file_conf_floor": float(data.get("file_conf_floor", 0.0) or 0.0),
        "row_conf_floor": float(data.get("row_conf_floor", 0.0) or 0.0),
        "row_margin_floor": float(data.get("row_margin_floor", 0.0) or 0.0),
        "row_blank_floor": (float(row_blank_floor) if row_blank_floor is not None else None),
        "low_conf_reasons": list(data.get("low_conf_reasons") or []),
        "sequence_lock_applied": bool(data.get("sequence_lock_applied", False)),
        "sequence_lock_reason": str(data.get("sequence_lock_reason") or ""),
    }


def build_language_runtime_state(
    *,
    language_prefix: str,
    mapping_confidence_base: float,
    mapping_margin: float,
    mapping_tier: str,
    cv_plan,
    planned_indices,
    anchor_graph,
    plan_policy,
    runtime_policy,
    postprocess: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    prefix = str(language_prefix or "").strip().lower()
    cv_plan_payload = dict(cv_plan or {})
    plan_policy_payload = dict(plan_policy or {})
    runtime_policy_payload = dict(runtime_policy or {})
    out: Dict[str, Any] = {
        "mapping_confidence_base": float(mapping_confidence_base or 0.0),
        "mapping_margin": float(mapping_margin or 0.0),
        "mapping_tier": str(mapping_tier or ""),
        "cv_plan": cv_plan_payload,
        "planned_indices": planned_indices,
        "anchor_graph": anchor_graph,
        "plan_policy": plan_policy_payload,
        "runtime_policy": runtime_policy_payload,
    }
    if prefix:
        out[f"{prefix}_cv_plan"] = cv_plan_payload
        out[f"{prefix}_planned_cv_indices"] = planned_indices
        out[f"{prefix}_anchor_graph"] = anchor_graph
        out[f"{prefix}_plan_policy"] = plan_policy_payload
    out.update(normalize_runtime_postprocess_state(postprocess))
    return out


def extract_language_runtime_state(
    state: Mapping[str, Any] | None,
    *,
    language_prefix: str,
    conf_floor_default: float = 0.0,
    row_margin_floor_default: float = 6.0,
) -> Dict[str, Any]:
    data = dict(state or {})
    prefix = str(language_prefix or "").strip().lower()
    prefixed_cv_plan = data.get(f"{prefix}_cv_plan") if prefix else None
    prefixed_planned = data.get(f"{prefix}_planned_cv_indices") if prefix else None
    prefixed_graph = data.get(f"{prefix}_anchor_graph") if prefix else None
    prefixed_policy = data.get(f"{prefix}_plan_policy") if prefix else None

    cv_plan = dict(data.get("cv_plan") or prefixed_cv_plan or {})
    planned_indices = data.get("planned_indices", prefixed_planned)
    anchor_graph = data.get("anchor_graph", prefixed_graph)
    plan_policy = dict(data.get("plan_policy") or prefixed_policy or {})
    runtime_policy = dict(data.get("runtime_policy") or {})

    mapping_confidence_base = float(
        data.get("mapping_confidence_base", data.get("mapping_confidence", 0.0)) or 0.0
    )
    mapping_margin = float(data.get("mapping_margin", 0.0) or 0.0)
    mapping_tier = str(data.get("mapping_tier") or runtime_policy.get("mapping_tier") or "")

    post = normalize_runtime_postprocess_state(
        {
            "file_mapping_low_conf": data.get("file_mapping_low_conf", False),
            "file_conf_floor": data.get("file_conf_floor", conf_floor_default),
            "row_conf_floor": data.get("row_conf_floor", conf_floor_default),
            "row_margin_floor": data.get("row_margin_floor", row_margin_floor_default),
            "row_blank_floor": data.get("row_blank_floor"),
            "low_conf_reasons": data.get("low_conf_reasons"),
            "sequence_lock_applied": data.get("sequence_lock_applied", False),
            "sequence_lock_reason": data.get("sequence_lock_reason", ""),
        }
    )
    runtime_policy.setdefault("sequence_lock_applied", post["sequence_lock_applied"])
    runtime_policy.setdefault("sequence_lock_reason", post["sequence_lock_reason"])

    out: Dict[str, Any] = {
        "mapping_confidence_base": float(mapping_confidence_base),
        "mapping_margin": float(mapping_margin),
        "mapping_tier": str(mapping_tier),
        "cv_plan": cv_plan,
        "planned_indices": planned_indices,
        "anchor_graph": anchor_graph,
        "plan_policy": plan_policy,
        "runtime_policy": runtime_policy,
    }
    if prefix:
        out[f"{prefix}_cv_plan"] = cv_plan
        out[f"{prefix}_planned_cv_indices"] = planned_indices
        out[f"{prefix}_anchor_graph"] = anchor_graph
        out[f"{prefix}_plan_policy"] = plan_policy
    out.update(post)
    return out


def log_sinsy_plan_guard(
    *,
    sinsy_label_entries,
    cv_plan,
    runtime_policy,
    fname: str,
    log_fn: Callable[[str], None],
    log_prefix: str = "[MAP]",
) -> None:
    if not sinsy_label_entries:
        return
    plan_source = str((cv_plan or {}).get("source") or "")
    if plan_source != "sinsy_labels":
        log_fn(
            f"{log_prefix} {fname}: Sinsy 라벨은 있으나 planner source가 fallback으로 선택됨 "
            f"(source={plan_source or 'fallback'})"
        )
        return
    plan_margin = float(((cv_plan or {}).get("meta") or {}).get("margin", 0.0) or 0.0)
    row_margin_floor = float((runtime_policy or {}).get("row_margin_floor", 6.0))
    if plan_margin < row_margin_floor:
        log_fn(
            f"{log_prefix} {fname}: Sinsy planner margin 낮음 "
            f"(margin={plan_margin:.1f} < {row_margin_floor:.1f})"
        )
