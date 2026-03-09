from __future__ import annotations


def build_candidate_index(candidates):
    return {
        str(c.get("name", "")): c
        for c in (candidates or [])
        if isinstance(c, dict) and c.get("name")
    }


def select_primary_mapping_candidate(
    mapping_candidates,
    *,
    format_type="",
    forced_words_mapping=False,
    alias_candidate_name="alias_phone_fallback",
):
    candidates = list(mapping_candidates or [])
    if not candidates:
        return None, "", {}

    candidate_by_name = build_candidate_index(candidates)
    alias_candidate = candidate_by_name.get(alias_candidate_name)
    fmt = str(format_type or "").strip().lower()

    selected_candidate = None
    mapping_reason_code = ""
    if forced_words_mapping:
        if candidate_by_name.get("filename_token"):
            selected_candidate = candidate_by_name["filename_token"]
            mapping_reason_code = "filename_words_lock"
        elif candidate_by_name.get("filename_linear_fallback"):
            selected_candidate = candidate_by_name["filename_linear_fallback"]
            mapping_reason_code = "filename_words_linear_lock"
        elif alias_candidate:
            selected_candidate = alias_candidate
            mapping_reason_code = "alias_words_fallback"
        else:
            selected_candidate = max(candidates, key=lambda c: c.get("objective", -10**9))
            mapping_reason_code = str(selected_candidate.get("name", ""))
    else:
        stage1_pool = list(candidates)
        if fmt in {"cvvc", "cv"}:
            order_pool = [c for c in candidates if c.get("order_preserving")]
            if order_pool:
                stage1_pool = order_pool
        selected_candidate = max(
            stage1_pool,
            key=lambda c: (
                c.get("objective", -10**9)
                + (3.0 if (fmt in {"cvvc", "cv"} and c.get("order_preserving")) else 0.0)
            ),
        )
        mapping_reason_code = str(selected_candidate.get("name", ""))

    return selected_candidate, mapping_reason_code, candidate_by_name


def maybe_promote_alias_candidate(
    *,
    selected_candidate,
    alias_candidate,
    provisional_conf,
    conf_threshold,
    format_type="",
):
    if (
        not alias_candidate
        or alias_candidate is selected_candidate
        or (str(format_type or "").strip().lower() in {"cvvc", "cv"} and selected_candidate and selected_candidate.get("lock_order"))
        or float(provisional_conf or 0.0) < float(conf_threshold or 0.0)
    ):
        return selected_candidate, False

    objective_gain = alias_candidate.get("objective", -10**9) - selected_candidate.get("objective", -10**9)
    score_gain = alias_candidate.get("score", -10**9) - selected_candidate.get("score", -10**9)
    conf_gain = alias_candidate.get("mean_syll_conf", 0.0) - selected_candidate.get("mean_syll_conf", 0.0)
    gain_th = 7.0 if str(format_type or "").strip().lower() in {"cvvc", "cv"} else 5.0
    if (
        alias_candidate.get("score", 0.0) >= 70.0
        and objective_gain >= gain_th
        and (score_gain >= 5.0 or conf_gain >= 0.06)
    ):
        return alias_candidate, True
    return selected_candidate, False


__all__ = [
    "build_candidate_index",
    "maybe_promote_alias_candidate",
    "select_primary_mapping_candidate",
]
