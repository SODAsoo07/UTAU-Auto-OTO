from __future__ import annotations


def decide_cv_row_abstain(
    *,
    alias_type,
    format_type,
    candidate_idx,
    candidate_count,
    candidate_active,
    confidence_margin=None,
    min_confidence_margin=None,
    row_confidence=None,
    min_row_confidence=None,
    blank_confidence=None,
    max_blank_confidence=None,
    active_only_formats=None,
    active_alias_types=None,
    margin_alias_types=None,
    margin_formats=None,
    blank_formats=None,
):
    fmt = str(format_type or "").strip().lower()
    a_type = str(alias_type or "").strip().lower()
    active_only_formats = {str(x).strip().lower() for x in (active_only_formats or set())}
    active_alias_types = {str(x).strip().lower() for x in (active_alias_types or {"cv", "cv_head"})}
    margin_alias_types = {str(x).strip().lower() for x in (margin_alias_types or {"cv", "cv_head"})}
    margin_formats = {str(x).strip().lower() for x in (margin_formats or set())}
    blank_formats = {str(x).strip().lower() for x in (blank_formats or set())}

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

    if fmt in margin_formats and a_type in margin_alias_types:
        try:
            margin = float(confidence_margin)
        except Exception:
            margin = None
        try:
            min_margin = float(min_confidence_margin)
        except Exception:
            min_margin = None
        if margin is not None and min_margin is not None and margin < min_margin:
            return {
                "should_skip": True,
                "reason": "row_low_margin_candidate",
                "diag_hint": f"idx={idx}; margin={margin:.2f}; min_margin={min_margin:.2f}",
            }

    try:
        row_conf = None if row_confidence is None else float(row_confidence)
    except Exception:
        row_conf = None
    try:
        min_row_conf = None if min_row_confidence is None else float(min_row_confidence)
    except Exception:
        min_row_conf = None
    if a_type in margin_alias_types and row_conf is not None and min_row_conf is not None and row_conf < min_row_conf:
        return {
            "should_skip": True,
            "reason": "row_low_confidence",
            "diag_hint": f"idx={idx}; conf={row_conf:.2f}; min_conf={min_row_conf:.2f}",
        }

    if fmt in blank_formats and a_type in margin_alias_types:
        try:
            blank = None if blank_confidence is None else float(blank_confidence)
        except Exception:
            blank = None
        try:
            max_blank = None if max_blank_confidence is None else float(max_blank_confidence)
        except Exception:
            max_blank = None
        if blank is not None and max_blank is not None and blank >= max_blank:
            return {
                "should_skip": True,
                "reason": "row_high_blank_confidence",
                "diag_hint": f"idx={idx}; blank={blank:.2f}; max_blank={max_blank:.2f}",
            }

    return {
        "should_skip": False,
        "reason": "",
        "diag_hint": "",
    }


__all__ = ["decide_cv_row_abstain"]
