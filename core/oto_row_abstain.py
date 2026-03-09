from __future__ import annotations


def decide_cv_row_abstain(
    *,
    alias_type,
    format_type,
    candidate_idx,
    candidate_count,
    candidate_active,
    active_only_formats=None,
    active_alias_types=None,
):
    fmt = str(format_type or "").strip().lower()
    a_type = str(alias_type or "").strip().lower()
    active_only_formats = {str(x).strip().lower() for x in (active_only_formats or set())}
    active_alias_types = {str(x).strip().lower() for x in (active_alias_types or {"cv", "cv_head"})}

    try:
        idx = int(candidate_idx)
    except Exception:
        idx = -1
    try:
        count = int(candidate_count)
    except Exception:
        count = 0

    if idx < 0 or idx >= count:
        return {
            "should_skip": True,
            "reason": "row_invalid_candidate_index",
            "diag_hint": f"idx={idx}; count={count}",
        }

    if fmt in active_only_formats and a_type in active_alias_types and not bool(candidate_active):
        return {
            "should_skip": True,
            "reason": "row_inactive_candidate",
            "diag_hint": f"idx={idx}; format={fmt}; alias_type={a_type}",
        }

    return {
        "should_skip": False,
        "reason": "",
        "diag_hint": "",
    }


__all__ = ["decide_cv_row_abstain"]
