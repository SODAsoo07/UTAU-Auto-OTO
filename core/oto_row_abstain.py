from __future__ import annotations


def _normalize_key(value):
    return str(value or "").strip().lower()


def _extract_onset_token(alias_text):
    text = str(alias_text or "").strip().lower()
    if not text:
        return ""
    vowels = set("aeiou")
    out = []
    for ch in text:
        if ch in vowels:
            break
        if "a" <= ch <= "z":
            out.append(ch)
        elif out:
            break
    return "".join(out)


def _resolve_threshold(default_value, *, fmt, alias_type, per_format=None, per_alias_type=None):
    resolved = default_value
    if isinstance(per_format, dict):
        fmt_value = per_format.get(fmt, per_format.get("*"))
        if fmt_value is not None:
            resolved = fmt_value
    if isinstance(per_alias_type, dict):
        alias_value = per_alias_type.get(alias_type, per_alias_type.get("*"))
        if alias_value is not None:
            resolved = alias_value
    return resolved


def decide_cv_row_abstain(
    *,
    alias_type,
    alias_text=None,
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
    min_confidence_margin_by_format=None,
    min_confidence_margin_by_alias_type=None,
    min_row_confidence_by_format=None,
    min_row_confidence_by_alias_type=None,
    max_blank_confidence_by_format=None,
    max_blank_confidence_by_alias_type=None,
):
    fmt = _normalize_key(format_type)
    a_type = _normalize_key(alias_type)
    active_only_formats = {_normalize_key(x) for x in (active_only_formats or set())}
    active_alias_types = {_normalize_key(x) for x in (active_alias_types or {"cv", "cv_head"})}
    margin_alias_types = {_normalize_key(x) for x in (margin_alias_types or {"cv", "cv_head"})}
    margin_formats = {_normalize_key(x) for x in (margin_formats or set())}
    blank_formats = {_normalize_key(x) for x in (blank_formats or set())}

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

    margin = None
    min_margin = None
    if fmt in margin_formats and a_type in margin_alias_types:
        try:
            margin = float(confidence_margin)
        except Exception:
            margin = None
        try:
            min_margin = float(
                _resolve_threshold(
                    min_confidence_margin,
                    fmt=fmt,
                    alias_type=a_type,
                    per_format=min_confidence_margin_by_format,
                    per_alias_type=min_confidence_margin_by_alias_type,
                )
            )
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
        min_row_conf = float(
            _resolve_threshold(
                min_row_confidence,
                fmt=fmt,
                alias_type=a_type,
                per_format=min_row_confidence_by_format,
                per_alias_type=min_row_confidence_by_alias_type,
            )
        )
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
            max_blank = float(
                _resolve_threshold(
                    max_blank_confidence,
                    fmt=fmt,
                    alias_type=a_type,
                    per_format=max_blank_confidence_by_format,
                    per_alias_type=max_blank_confidence_by_alias_type,
                )
            )
        except Exception:
            max_blank = None
        if blank is not None and max_blank is not None and blank >= max_blank:
            head_cv_relax = bool(
                a_type in {"cv", "cv_head"}
                and idx == 0
                and count >= 2
                and fmt in {"cv", "cvc", "cvvc", "vcv"}
            )
            if head_cv_relax:
                # Avoid dropping head CV rows too aggressively; skipping here often
                # propagates template offset=0 into later ML refinement.
                conf_ok = True
                if row_conf is not None and min_row_conf is not None:
                    conf_ok = bool(row_conf >= (min_row_conf + 0.03))
                hard_blank_cap = max(0.88, float(max_blank) + 0.20)
                if conf_ok and blank < hard_blank_cap:
                    return {
                        "should_skip": False,
                        "reason": "",
                        "diag_hint": (
                            f"idx={idx}; blank_relaxed={blank:.2f}; hard_cap={hard_blank_cap:.2f}; "
                            f"conf={row_conf if row_conf is not None else -1:.2f}"
                        ),
                    }
            mid_unvoiced_cv_relax = bool(
                a_type in {"cv", "cv_head"}
                and idx > 0
                and idx < max(count - 1, 1)
                and fmt in {"cvc", "cvvc", "vcv"}
            )
            if mid_unvoiced_cv_relax:
                onset = _extract_onset_token(alias_text)
                unvoiced_onsets = {
                    "t", "tt", "d", "k", "kk", "g", "p", "pp", "b",
                    "s", "ss", "h", "ch", "j", "c", "q", "f",
                }
                conf_ok = True
                if row_conf is not None and min_row_conf is not None:
                    conf_ok = bool(row_conf >= (min_row_conf + 0.02))
                margin_ok = True
                if margin is not None and min_margin is not None:
                    margin_ok = bool(margin >= min_margin)
                # Unvoiced CV (e.g., ti/te/to) can look blank-heavy around burst/closure.
                if onset in unvoiced_onsets and conf_ok and margin_ok:
                    hard_blank_cap = max(0.84, float(max_blank) + 0.18)
                    if blank <= hard_blank_cap:
                        return {
                            "should_skip": False,
                            "reason": "",
                            "diag_hint": (
                                f"idx={idx}; blank_relaxed_mid={blank:.2f}; hard_cap={hard_blank_cap:.2f}; "
                                f"onset={onset}; conf={row_conf if row_conf is not None else -1:.2f}; "
                                f"margin={margin if margin is not None else -1:.2f}"
                            ),
                        }
            return {
                "should_skip": True,
                "reason": "row_high_blank_confidence",
                "diag_hint": (
                    f"idx={idx}; blank={blank:.2f}; max_blank={max_blank:.2f}; "
                    f"conf={row_conf if row_conf is not None else -1:.2f}; "
                    f"margin={margin if margin is not None else -1:.2f}"
                ),
            }

    return {
        "should_skip": False,
        "reason": "",
        "diag_hint": "",
    }


__all__ = ["decide_cv_row_abstain"]
