from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


def build_top_mapping_candidate_rows(
    mapping_candidates: Iterable[Mapping[str, Any]] | None,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        list(mapping_candidates or []),
        key=lambda item: (item or {}).get("objective", -10**9),
        reverse=True,
    )[: max(0, int(limit))]
    rows: List[Dict[str, Any]] = []
    for cand in ranked:
        c = dict(cand or {})
        rows.append(
            {
                "name": str(c.get("name", "")),
                "score": float(c.get("score", 0.0) or 0.0),
                "objective": float(c.get("objective", 0.0) or 0.0),
                "order_preserving": bool(c.get("order_preserving")),
                "lock_order": bool(c.get("lock_order")),
                "mean_syll_conf": float(c.get("mean_syll_conf", 0.0) or 0.0),
            }
        )
    return rows


def build_ja_mapping_candidate_trace(
    *,
    fname: str,
    format_type: str,
    mapping_reason_code: str,
    mapping_tier: str,
    mapping_confidence: float,
    mapping_margin: float,
    filename_order_locked: bool,
    forced_words_mapping: bool,
    alignment_weight: float,
    blank_conf_mean: float,
    anchor_lock_lite: bool,
    selected_candidate_name: str,
    filename_syllables: Sequence[str] | None,
    cv_targets: Sequence[str] | None,
    candidate_rows: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, Any]:
    return {
        "event": "ja_mapping_candidate",
        "file": str(fname or ""),
        "format_type": str(format_type or ""),
        "mapping_reason_code": str(mapping_reason_code or ""),
        "mapping_tier": str(mapping_tier or ""),
        "mapping_confidence": float(mapping_confidence or 0.0),
        "mapping_margin": float(mapping_margin or 0.0),
        "filename_order_locked": bool(filename_order_locked),
        "forced_words_mapping": bool(forced_words_mapping),
        "alignment_weight": float(alignment_weight or 0.0),
        "blank_conf_mean": float(blank_conf_mean or 0.0),
        "anchor_lock_lite": bool(anchor_lock_lite),
        "selected_candidate": str(selected_candidate_name or ""),
        "filename_syllables": list(filename_syllables or []),
        "cv_targets": list(cv_targets or []),
        "candidate_rows": [dict(row or {}) for row in (candidate_rows or [])],
    }


def maybe_append_cv_mapping_trace(
    *,
    fname: str,
    alias: str,
    alias_type: str,
    format_type: str,
    target_tok: str,
    expected_idx: int,
    mapped_idx: int,
    syllables_info: Sequence[Mapping[str, Any]] | None,
    mapping_tier: str,
    mapping_reason_code: str,
    mapping_confidence: float,
    filename_order_locked: bool,
    syllable_confidence_by_idx: Sequence[float] | None,
    should_trace_mapping_decision_fn: Callable[..., bool],
    append_mapping_trace_fn: Callable[[Dict[str, Any]], None],
    build_trace_record_fn: Callable[..., Dict[str, Any]],
    normalize_syllable_token_fn: Callable[[str], str],
    syllable_info_token_fn: Callable[[Mapping[str, Any]], str],
) -> bool:
    infos = list(syllables_info or [])
    if not infos:
        return False
    if expected_idx < 0 or mapped_idx < 0:
        return False
    if expected_idx >= len(infos) or mapped_idx >= len(infos):
        return False

    expected_tok = str(syllable_info_token_fn(infos[expected_idx]) or "")
    mapped_tok = str(syllable_info_token_fn(infos[mapped_idx]) or "")

    local_trace_conf = None
    conf_values = list(syllable_confidence_by_idx or [])
    if conf_values:
        conf_idx = max(0, min(int(expected_idx), len(conf_values) - 1))
        local_trace_conf = float(conf_values[conf_idx])

    should_trace = bool(
        should_trace_mapping_decision_fn(
            mapping_tier=mapping_tier,
            expected_idx=expected_idx,
            mapped_idx=mapped_idx,
            target_token=normalize_syllable_token_fn(target_tok),
            mapped_token=normalize_syllable_token_fn(mapped_tok),
        )
    )
    if not should_trace:
        return False

    append_mapping_trace_fn(
        build_trace_record_fn(
            fname=fname,
            alias=alias,
            alias_type=alias_type,
            format_type=format_type,
            target_tok=target_tok,
            expected_idx=expected_idx,
            mapped_idx=mapped_idx,
            expected_tok=expected_tok,
            mapped_tok=mapped_tok,
            mapping_tier=mapping_tier,
            mapping_reason_code=mapping_reason_code,
            mapping_confidence=mapping_confidence,
            filename_order_locked=filename_order_locked,
            local_conf=local_trace_conf,
        )
    )
    return True
