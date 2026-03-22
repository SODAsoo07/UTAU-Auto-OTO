from __future__ import annotations

import os

from core.generation.mapping_reason_codes import (
    JA_REASON_ALIAS_WORDS_FALLBACK,
    JA_REASON_FILENAME_WORDS_LINEAR_LOCK,
    JA_REASON_FILENAME_WORDS_LOCK,
)
from core.mapping_format_policy import ORDER_PRESERVING_CANDIDATE_FORMATS, normalize_format


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
            mapping_reason_code = JA_REASON_FILENAME_WORDS_LOCK
        elif candidate_by_name.get("filename_linear_fallback"):
            selected_candidate = candidate_by_name["filename_linear_fallback"]
            mapping_reason_code = JA_REASON_FILENAME_WORDS_LINEAR_LOCK
        elif alias_candidate:
            selected_candidate = alias_candidate
            mapping_reason_code = JA_REASON_ALIAS_WORDS_FALLBACK
        else:
            selected_candidate = max(candidates, key=lambda c: c.get("objective", -10**9))
            mapping_reason_code = str(selected_candidate.get("name", ""))
    else:
        stage1_pool = list(candidates)
        order_sensitive = fmt in ORDER_PRESERVING_CANDIDATE_FORMATS
        if order_sensitive:
            order_pool = [c for c in candidates if c.get("order_preserving")]
            if order_pool:
                # Relax order-preserving lock if the best ordered candidate is low quality
                # and a non-ordered candidate is significantly better.
                def _env_float(name, default):
                    raw = os.getenv(name)
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return float(default)

                relax_conf_max = _env_float("UTOA_JA_ORDER_LOCK_RELAX_CONF_MAX", 0.35)
                relax_conf_gain = _env_float("UTOA_JA_ORDER_LOCK_RELAX_CONF_GAIN", 0.12)
                relax_obj_gain = _env_float("UTOA_JA_ORDER_LOCK_RELAX_OBJ_GAIN", 6.0)

                best_order = max(order_pool, key=lambda c: c.get("objective", -10**9))
                best_any = max(candidates, key=lambda c: c.get("objective", -10**9))
                if best_any is not best_order:
                    order_conf = float(best_order.get("mean_syll_conf", 0.0) or 0.0)
                    any_conf = float(best_any.get("mean_syll_conf", 0.0) or 0.0)
                    obj_gain = float(best_any.get("objective", -10**9)) - float(best_order.get("objective", -10**9))
                    conf_gain = any_conf - order_conf
                    if order_conf < relax_conf_max and (conf_gain >= relax_conf_gain or obj_gain >= relax_obj_gain):
                        stage1_pool = list(candidates)
                    else:
                        stage1_pool = order_pool
                else:
                    stage1_pool = order_pool
        selected_candidate = max(
            stage1_pool,
            key=lambda c: (
                c.get("objective", -10**9)
                + (3.0 if (order_sensitive and c.get("order_preserving")) else 0.0)
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
    fmt = normalize_format(format_type)
    order_sensitive = fmt in ORDER_PRESERVING_CANDIDATE_FORMATS
    if (
        not alias_candidate
        or alias_candidate is selected_candidate
        or (order_sensitive and selected_candidate and selected_candidate.get("lock_order"))
        or float(provisional_conf or 0.0) < float(conf_threshold or 0.0)
    ):
        return selected_candidate, False

    objective_gain = alias_candidate.get("objective", -10**9) - selected_candidate.get("objective", -10**9)
    score_gain = alias_candidate.get("score", -10**9) - selected_candidate.get("score", -10**9)
    conf_gain = alias_candidate.get("mean_syll_conf", 0.0) - selected_candidate.get("mean_syll_conf", 0.0)
    gain_th = 7.0 if order_sensitive else 5.0
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
