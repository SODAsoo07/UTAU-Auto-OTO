from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from core.format_type_utils import normalize_format_type
from core.oto_ml_policy import infer_alias_family, selector_error_weights
from core.oto_ml_features import canonicalize_feature_row

SELECTOR_FEATURE_VERSION = "v2"

SELECTOR_FEATURE_NAMES = [
    "language",
    "format_type",
    "alias_type",
    "alias_group",
    "candidate_mode",
    "candidate_index",
    "mapping_confidence",
    "jump_blocked_flag",
    "used_alias_based_syllables",
    "used_nuclei_fallback",
    "words_vs_alias_score_margin",
    "wav_duration_ms",
    "expected_anchor_ms",
    "base_offset_to_expected_ms",
    "base_pre_to_expected_ms",
    "base_cutoff_to_next_anchor_ms",
    "prev_phone_gap_ms",
    "next_phone_gap_ms",
    "energy_mean",
    "db_silence_ratio",
    "local_peak_db",
    "local_valley_db",
    "mel_window_energy_mean",
    "mel_window_silence_ratio",
    "coda_type",
    "vowel_class",
    "bridge_type",
    "mora_position",
    "onset_class",
    "voicing_class",
    "mapping_reason_code",
    "prev_alias_type",
    "next_alias_type",
    "cand_offset",
    "cand_cons",
    "cand_cutoff_abs",
    "cand_pre",
    "cand_ovl",
    "cand_pre_abs",
    "cand_cons_gap",
    "cand_cut_gap",
    "cand_ovl_ratio",
    "cand_offset_to_expected_ms",
    "cand_pre_to_expected_ms",
    "cand_cutoff_to_next_anchor_ms",
    "cand_delta_from_base_total",
    "cand_delta_from_base_anchor",
    "cand_overlap_margin",
    "cand_pre_cons_margin",
    "cand_leak_risk",
    "cand_silence_risk",
    "cand_jump_risk",
]

SELECTOR_CATEGORICAL_FEATURES = [
    "language",
    "format_type",
    "alias_type",
    "alias_group",
    "candidate_mode",
    "coda_type",
    "vowel_class",
    "bridge_type",
    "mora_position",
    "onset_class",
    "voicing_class",
    "mapping_reason_code",
    "prev_alias_type",
    "next_alias_type",
]

SELECTOR_FEATURE_DEFAULTS = {name: 0.0 for name in SELECTOR_FEATURE_NAMES}
for _name in SELECTOR_CATEGORICAL_FEATURES:
    SELECTOR_FEATURE_DEFAULTS[_name] = ""


def get_selector_feature_schema() -> Dict[str, object]:
    return {
        "feature_version": SELECTOR_FEATURE_VERSION,
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "categorical_features": list(SELECTOR_CATEGORICAL_FEATURES),
    }


def write_selector_feature_schema(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(get_selector_feature_schema(), f, ensure_ascii=False, indent=2)
    return path


def canonicalize_selector_feature_row(row: Dict[str, object], feature_names: Optional[List[str]] = None) -> Dict[str, object]:
    names = feature_names or SELECTOR_FEATURE_NAMES
    out = {}
    for name in names:
        val = row.get(name, SELECTOR_FEATURE_DEFAULTS.get(name, 0.0))
        if name in SELECTOR_CATEGORICAL_FEATURES:
            out[name] = "" if val is None else str(val)
        else:
            try:
                out[name] = float(val)
            except Exception:
                out[name] = 0.0
    return out


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_validate_func(language: str):
    if str(language or "").strip().lower() == "japanese":
        from core.ja_oto_generator import validate_oto_params
    else:
        from core.oto_generator import validate_oto_params
    return validate_oto_params


def _next_anchor_abs_ms(row: Dict[str, object]) -> float:
    cut_abs = _to_float(row.get("base_cutoff_abs"), 0.0)
    gap = _to_float(row.get("base_cutoff_to_next_anchor_ms"), 0.0)
    if gap == 0.0:
        return 0.0
    return cut_abs + gap


def _build_base_params(row: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    offset = _to_float(row.get("base_offset"), 0.0)
    cons = _to_float(row.get("base_cons"), 0.0)
    cut_abs = _to_float(row.get("base_cutoff_abs"), 0.0)
    pre = _to_float(row.get("base_pre"), 0.0)
    ovl = _to_float(row.get("base_ovl"), 0.0)
    return offset, cons, cut_abs, pre, ovl


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _finalize_candidate(
    language: str,
    offset: float,
    cons: float,
    cut_abs: float,
    pre: float,
    ovl: float,
    *,
    next_anchor_abs: float = 0.0,
    next_allow: float = 8.0,
    pre_ovl_gap_min: float = 8.0,
    cons_pre_gap_min: float = 10.0,
) -> Tuple[float, float, float, float, float]:
    pre = max(4.0, float(pre))
    ovl = min(max(0.0, float(ovl)), max(pre - float(pre_ovl_gap_min), 0.0))
    cons = max(float(cons), pre + float(cons_pre_gap_min))
    cut_abs = max(float(cut_abs), cons + 10.0)
    if next_anchor_abs > 0.0:
        cut_abs = min(cut_abs, max(cons + 10.0, next_anchor_abs + float(next_allow)))
    validate_oto_params = _get_validate_func(language)
    return validate_oto_params(float(offset), float(cons), -(float(cut_abs) - float(offset)), float(pre), float(ovl))


def _add_common_candidates(add_candidate, lang: str, offset: float, cons: float, cut_abs: float, pre: float, ovl: float, next_anchor_abs: float, expected: float, pre_abs: float) -> None:
    shift_soft = _clamp(expected - pre_abs, -14.0, 14.0) * 0.25
    add_candidate(
        "anchor_soft",
        _finalize_candidate(
            lang,
            offset,
            cons,
            cut_abs,
            pre + shift_soft,
            ovl,
            next_anchor_abs=next_anchor_abs,
            next_allow=7.0,
            pre_ovl_gap_min=8.0,
        ),
    )

    shift_tight = _clamp(expected - pre_abs, -22.0, 22.0) * 0.45
    add_candidate(
        "anchor_tight",
        _finalize_candidate(
            lang,
            offset,
            cons,
            cut_abs,
            pre + shift_tight,
            min(ovl, max((pre + shift_tight) - 10.0, 0.0)),
            next_anchor_abs=next_anchor_abs,
            next_allow=5.0,
            pre_ovl_gap_min=10.0,
        ),
    )

    add_candidate(
        "next_safe",
        _finalize_candidate(
            lang,
            offset,
            cons,
            cut_abs,
            pre * 0.98,
            min(ovl, max((pre * 0.98) - 9.0, 0.0)),
            next_anchor_abs=next_anchor_abs,
            next_allow=3.0,
            pre_ovl_gap_min=9.0,
            cons_pre_gap_min=11.0,
        ),
    )


def _add_japanese_candidates(add_candidate, format_type: str, alias_type: str, onset_class: str, mapping_conf: float, jump_blocked: bool, offset: float, cons: float, cut_abs: float, pre: float, ovl: float, next_anchor_abs: float, expected: float, pre_abs: float) -> None:
    if format_type == "cv":
        add_candidate(
            "ja_cv_hold",
            _finalize_candidate(
                "japanese",
                offset,
                cons,
                cut_abs,
                pre * 0.90,
                min(ovl, max((pre * 0.90) - 11.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=3.0,
                pre_ovl_gap_min=11.0,
                cons_pre_gap_min=12.0,
            ),
        )
        if (not jump_blocked) and mapping_conf >= 0.72:
            shift = max(0.0, _clamp(expected - pre_abs, -6.0, 16.0)) * 0.60
            add_candidate(
                "ja_cv_forward_soft",
                _finalize_candidate(
                    "japanese",
                    offset,
                    cons,
                    cut_abs,
                    pre + shift,
                    min(ovl, max((pre + shift) - 9.0, 0.0)),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=4.0,
                    pre_ovl_gap_min=9.0,
                    cons_pre_gap_min=10.0,
                ),
            )
        return

    if format_type == "cvvc":
        if alias_type in {"cv", "cv_head", "mono"}:
            add_candidate(
                "ja_cvvc_cv_hold",
                _finalize_candidate(
                "japanese",
                offset,
                cons,
                cut_abs,
                pre * 0.92,
                min(ovl, max((pre * 0.92) - 11.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=3.0,
                pre_ovl_gap_min=11.0,
                cons_pre_gap_min=12.0,
            ),
        )
            if (not jump_blocked) and mapping_conf >= 0.78 and onset_class not in {"fricative", "sibilant"}:
                shift = max(0.0, _clamp(expected - pre_abs, -4.0, 14.0)) * 0.55
                add_candidate(
                    "ja_cvvc_cv_forward",
                    _finalize_candidate(
                        "japanese",
                        offset,
                        cons,
                        cut_abs,
                        pre + shift,
                        min(ovl, max((pre + shift) - 9.0, 0.0)),
                        next_anchor_abs=next_anchor_abs,
                        next_allow=4.0,
                        pre_ovl_gap_min=9.0,
                        cons_pre_gap_min=10.0,
                    ),
                )
        elif alias_type in {"vc", "vv"}:
            base_pre = max(pre, 6.0)
            base_cons = max(cons, base_pre + 36.0)
            base_cut = max(cut_abs, base_cons + 22.0)
            add_candidate(
                "ja_cvvc_bridge_hold",
                _finalize_candidate(
                    "japanese",
                    offset,
                    base_cons,
                    base_cut,
                    base_pre,
                    max(0.0, base_pre - 10.0),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=8.0,
                    pre_ovl_gap_min=10.0,
                    cons_pre_gap_min=36.0,
                ),
            )
        return

    if format_type == "vcv":
        if alias_type in {"vc", "vv", "vcv"}:
            bridge_pre = max(pre, 6.0)
            bridge_cons = max(cons, bridge_pre + (56.0 if alias_type == "vv" else 34.0))
            bridge_cut = max(cut_abs, bridge_cons + (42.0 if alias_type == "vv" else 18.0))
            add_candidate(
                "ja_vcv_bridge_hold",
                _finalize_candidate(
                    "japanese",
                    offset,
                    bridge_cons,
                    bridge_cut,
                    bridge_pre,
                    max(0.0, bridge_pre - (8.0 if alias_type == "vv" else 10.0)),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=10.0,
                    pre_ovl_gap_min=(8.0 if alias_type == "vv" else 10.0),
                    cons_pre_gap_min=(56.0 if alias_type == "vv" else 34.0),
                ),
            )
            if (not jump_blocked) and mapping_conf >= 0.80:
                shift = _clamp(expected - pre_abs, -8.0, 10.0) * 0.22
                add_candidate(
                    "ja_vcv_bridge_forward",
                    _finalize_candidate(
                        "japanese",
                        offset,
                        bridge_cons,
                        bridge_cut,
                        bridge_pre + shift,
                        max(0.0, (bridge_pre + shift) - (8.0 if alias_type == "vv" else 10.0)),
                        next_anchor_abs=next_anchor_abs,
                        next_allow=10.0,
                        pre_ovl_gap_min=(8.0 if alias_type == "vv" else 10.0),
                        cons_pre_gap_min=(56.0 if alias_type == "vv" else 34.0),
                    ),
                )
        elif alias_type in {"cv", "cv_head"}:
            add_candidate(
                "ja_vcv_cv_hold",
                _finalize_candidate(
                "japanese",
                offset,
                cons,
                cut_abs,
                pre * 0.93,
                min(ovl, max((pre * 0.93) - 11.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=4.0,
                pre_ovl_gap_min=11.0,
                cons_pre_gap_min=12.0,
            ),
        )


def _add_korean_candidates(add_candidate, format_type: str, alias_type: str, coda_type: str, mapping_conf: float, offset: float, cons: float, cut_abs: float, pre: float, ovl: float, next_anchor_abs: float, expected: float, pre_abs: float) -> None:
    if format_type == "cv":
        add_candidate(
            "kr_cv_hold",
            _finalize_candidate(
                "korean",
                offset,
                cons,
                cut_abs,
                pre * 0.95,
                min(ovl, max((pre * 0.95) - 10.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=4.0,
                pre_ovl_gap_min=10.0,
                cons_pre_gap_min=11.0,
            ),
        )
        shift = _clamp(expected - pre_abs, -10.0, 12.0) * 0.35
        add_candidate(
            "kr_cv_anchor",
            _finalize_candidate(
                "korean",
                offset,
                cons,
                cut_abs,
                pre + shift,
                min(ovl, max((pre + shift) - 9.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=4.0,
                pre_ovl_gap_min=9.0,
                cons_pre_gap_min=10.0,
            ),
        )
        return

    if format_type == "cvc":
        if alias_type in {"cv", "cv_head", "mono"}:
            add_candidate(
                "kr_cvc_body_hold",
                _finalize_candidate(
                    "korean",
                    offset,
                    cons,
                    cut_abs,
                    pre * 0.96,
                    min(ovl, max((pre * 0.96) - 10.0, 0.0)),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=5.0,
                    pre_ovl_gap_min=10.0,
                    cons_pre_gap_min=11.0,
                ),
            )
        if alias_type == "vc":
            gap = 36.0 if coda_type == "stop" else 30.0
            cut_gap = 22.0 if coda_type == "stop" else 18.0
            add_candidate(
                "kr_cvc_tail_hold",
                _finalize_candidate(
                    "korean",
                    offset,
                    max(cons, pre + gap),
                    max(cut_abs, max(cons, pre + gap) + cut_gap),
                    max(pre, 6.0),
                    max(0.0, max(pre, 6.0) - 10.0),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=10.0,
                    pre_ovl_gap_min=10.0,
                    cons_pre_gap_min=gap,
                ),
            )
        return

    if format_type == "cvvc":
        if alias_type in {"vc", "vv"}:
            if alias_type == "vc":
                pre_ovl_gap = 11.0
                cons_pre_gap = 34.0 if coda_type == "stop" else 26.0
                cut_gap = 18.0 if coda_type == "stop" else 22.0
                next_allow = 10.0 if coda_type == "stop" else 16.0
            else:
                pre_ovl_gap = 7.0
                cons_pre_gap = 84.0
                cut_gap = 48.0
                next_allow = 18.0
            bridge_pre = max(pre, 6.0)
            bridge_cons = max(cons, bridge_pre + cons_pre_gap)
            bridge_cut_abs = max(cut_abs, bridge_cons + cut_gap)
            add_candidate(
                "kr_cvvc_bridge_tight",
                _finalize_candidate(
                    "korean",
                    offset,
                    bridge_cons + (2.0 if alias_type == "vv" else 1.0),
                    bridge_cut_abs + (4.0 if alias_type == "vv" else 2.0),
                    bridge_pre,
                    max(0.0, bridge_pre - pre_ovl_gap),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=next_allow,
                    pre_ovl_gap_min=pre_ovl_gap,
                    cons_pre_gap_min=cons_pre_gap,
                ),
            )
            if mapping_conf >= 0.78:
                add_candidate(
                    "kr_cvvc_bridge_wide",
                    _finalize_candidate(
                        "korean",
                        offset,
                        bridge_cons + 8.0,
                        bridge_cut_abs + 12.0,
                        bridge_pre,
                        max(0.0, bridge_pre - max(pre_ovl_gap - 2.0, 5.0)),
                        next_anchor_abs=next_anchor_abs,
                        next_allow=next_allow + 4.0,
                        pre_ovl_gap_min=max(pre_ovl_gap - 2.0, 5.0),
                        cons_pre_gap_min=cons_pre_gap + 8.0,
                    ),
                )
        elif alias_type in {"cv", "cv_head"}:
            add_candidate(
                "kr_cvvc_cv_hold",
                _finalize_candidate(
                    "korean",
                    offset,
                    cons,
                    cut_abs,
                    pre * 0.95,
                    min(ovl, max((pre * 0.95) - 10.0, 0.0)),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=4.0,
                    pre_ovl_gap_min=10.0,
                    cons_pre_gap_min=11.0,
                ),
            )
        return

    if format_type == "vcv":
        if alias_type in {"vc", "vv"}:
            pre_ovl_gap = 11.0 if alias_type == "vc" else 7.0
            cons_pre_gap = 30.0 if alias_type == "vc" else 84.0
            cut_gap = 20.0 if alias_type == "vc" else 48.0
            bridge_pre = max(pre, 6.0)
            bridge_cons = max(cons, bridge_pre + cons_pre_gap)
            bridge_cut_abs = max(cut_abs, bridge_cons + cut_gap)
            add_candidate(
                "kr_vcv_bridge_hold",
                _finalize_candidate(
                    "korean",
                    offset,
                    bridge_cons,
                    bridge_cut_abs,
                    bridge_pre,
                    max(0.0, bridge_pre - pre_ovl_gap),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=(12.0 if alias_type == "vv" else 16.0),
                    pre_ovl_gap_min=pre_ovl_gap,
                    cons_pre_gap_min=cons_pre_gap,
                ),
            )
        else:
            shift = _clamp(expected - pre_abs, -12.0, 12.0) * 0.30
            add_candidate(
                "kr_vcv_body_relaxed",
                _finalize_candidate(
                    "korean",
                    offset,
                    cons,
                    cut_abs + 4.0,
                    pre + shift + 2.0,
                    min(ovl, max((pre + shift + 2.0) - 8.0, 0.0)),
                    next_anchor_abs=next_anchor_abs,
                    next_allow=7.0,
                    pre_ovl_gap_min=8.0,
                    cons_pre_gap_min=10.0,
                ),
            )


def build_selector_candidates(language: str, feature_row: Dict[str, object]) -> List[Dict[str, object]]:
    lang = str(language or "").strip().lower()
    format_type = normalize_format_type(lang, feature_row.get("format_type", "")) or "general"
    alias_type = str(feature_row.get("alias_type", "") or "").strip().lower()
    onset_class = str(feature_row.get("onset_class", "") or "").strip().lower()
    coda_type = str(feature_row.get("coda_type", "") or "").strip().lower()
    mapping_conf = _to_float(feature_row.get("mapping_confidence"), 0.0)
    jump_blocked = _to_float(feature_row.get("jump_blocked_flag"), 0.0) > 0.0
    offset, cons, cut_abs, pre, ovl = _build_base_params(feature_row)
    pre_abs = offset + pre
    expected = _to_float(feature_row.get("expected_anchor_ms"), pre_abs)
    next_anchor_abs = _next_anchor_abs_ms(feature_row)
    out: List[Dict[str, object]] = []

    def add_candidate(mode: str, params: Tuple[float, float, float, float, float]) -> None:
        cand_offset, cand_cons, cand_cutoff, cand_pre, cand_ovl = params
        cand_cut_abs = float(cand_offset) + abs(float(cand_cutoff))
        signature = (
            round(float(cand_offset), 3),
            round(float(cand_cons), 3),
            round(float(cand_cutoff), 3),
            round(float(cand_pre), 3),
            round(float(cand_ovl), 3),
        )
        if any(existing.get("_signature") == signature for existing in out):
            return
        out.append(
            {
                "candidate_mode": mode,
                "offset": float(cand_offset),
                "cons": float(cand_cons),
                "cutoff": float(cand_cutoff),
                "cutoff_abs": cand_cut_abs,
                "pre": float(cand_pre),
                "ovl": float(cand_ovl),
                "_signature": signature,
            }
        )

    add_candidate(
        "base",
        _finalize_candidate(
            lang,
            offset,
            cons,
            cut_abs,
            pre,
            ovl,
            next_anchor_abs=next_anchor_abs,
            next_allow=8.0,
        ),
    )

    _add_common_candidates(add_candidate, lang, offset, cons, cut_abs, pre, ovl, next_anchor_abs, expected, pre_abs)

    if alias_type in {"cv", "cv_head", "mono"}:
        add_candidate(
            "cv_guard",
            _finalize_candidate(
                lang,
                offset,
                cons,
                cut_abs,
                pre * 0.94,
                min(ovl, max((pre * 0.94) - 10.0, 0.0)),
                next_anchor_abs=next_anchor_abs,
                next_allow=4.0,
                pre_ovl_gap_min=10.0,
                cons_pre_gap_min=11.0,
            ),
        )
    elif alias_type in {"vc", "vv"}:
        if alias_type == "vc":
            pre_ovl_gap = 11.0
            cons_pre_gap = 34.0
            cut_gap = 18.0
            next_allow = 10.0
        else:
            pre_ovl_gap = 7.0
            cons_pre_gap = 84.0
            cut_gap = 48.0
            next_allow = 18.0
        bridge_pre = max(pre, 6.0)
        bridge_cons = max(cons, bridge_pre + cons_pre_gap)
        bridge_cut_abs = max(cut_abs, bridge_cons + cut_gap)
        add_candidate(
            "bridge_tight",
            _finalize_candidate(
                lang,
                offset,
                bridge_cons,
                bridge_cut_abs,
                bridge_pre,
                max(0.0, bridge_pre - pre_ovl_gap),
                next_anchor_abs=next_anchor_abs,
                next_allow=next_allow,
                pre_ovl_gap_min=pre_ovl_gap,
                cons_pre_gap_min=cons_pre_gap,
            ),
        )

    if lang == "japanese":
        _add_japanese_candidates(
            add_candidate,
            format_type,
            alias_type,
            onset_class,
            mapping_conf,
            jump_blocked,
            offset,
            cons,
            cut_abs,
            pre,
            ovl,
            next_anchor_abs,
            expected,
            pre_abs,
        )
    elif lang == "korean":
        _add_korean_candidates(
            add_candidate,
            format_type,
            alias_type,
            coda_type,
            mapping_conf,
            offset,
            cons,
            cut_abs,
            pre,
            ovl,
            next_anchor_abs,
            expected,
            pre_abs,
        )

    for index, candidate in enumerate(out):
        candidate["candidate_index"] = float(index)
        candidate.pop("_signature", None)
    return out


def build_selector_feature_row(language: str, base_row: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    base = canonicalize_feature_row(base_row)
    out: Dict[str, object] = {}
    for name in SELECTOR_FEATURE_NAMES:
        if name in SELECTOR_CATEGORICAL_FEATURES:
            out[name] = str(base_row.get(name, base.get(name, SELECTOR_FEATURE_DEFAULTS.get(name, ""))) or "")
        else:
            out[name] = _to_float(base_row.get(name, base.get(name, SELECTOR_FEATURE_DEFAULTS.get(name, 0.0))), 0.0)

    cand_offset = _to_float(candidate.get("offset"), 0.0)
    cand_cons = _to_float(candidate.get("cons"), 0.0)
    cand_cut_abs = _to_float(candidate.get("cutoff_abs"), 0.0)
    cand_pre = _to_float(candidate.get("pre"), 0.0)
    cand_ovl = _to_float(candidate.get("ovl"), 0.0)
    expected = _to_float(base_row.get("expected_anchor_ms"), 0.0)
    next_anchor = _next_anchor_abs_ms(base_row)
    base_offset = _to_float(base_row.get("base_offset"), 0.0)
    base_pre_abs = base_offset + _to_float(base_row.get("base_pre"), 0.0)

    out["language"] = str(base_row.get("language", "") or "")
    out["format_type"] = str(base_row.get("format_type", "") or "")
    out["alias_type"] = str(base_row.get("alias_type", "") or "")
    out["alias_group"] = str(base_row.get("alias_group", "") or "")
    out["candidate_mode"] = str(candidate.get("candidate_mode", "base") or "base")
    out["candidate_index"] = _to_float(candidate.get("candidate_index"), 0.0)
    out["cand_offset"] = cand_offset
    out["cand_cons"] = cand_cons
    out["cand_cutoff_abs"] = cand_cut_abs
    out["cand_pre"] = cand_pre
    out["cand_ovl"] = cand_ovl
    out["cand_pre_abs"] = cand_offset + cand_pre
    out["cand_cons_gap"] = max(cand_cons - cand_pre, 0.0)
    out["cand_cut_gap"] = max(cand_cut_abs - cand_cons, 0.0)
    out["cand_ovl_ratio"] = 0.0 if cand_pre <= 1e-6 else (cand_ovl / max(cand_pre, 1e-6))
    out["cand_offset_to_expected_ms"] = cand_offset - expected
    out["cand_pre_to_expected_ms"] = (cand_offset + cand_pre) - expected
    out["cand_cutoff_to_next_anchor_ms"] = (next_anchor - cand_cut_abs) if next_anchor > 0.0 else 0.0
    out["cand_delta_from_base_total"] = (
        abs(cand_offset - base_offset)
        + abs(cand_cons - _to_float(base_row.get("base_cons"), 0.0))
        + abs(cand_cut_abs - _to_float(base_row.get("base_cutoff_abs"), 0.0))
        + abs(cand_pre - _to_float(base_row.get("base_pre"), 0.0))
        + abs(cand_ovl - _to_float(base_row.get("base_ovl"), 0.0))
    )
    out["cand_delta_from_base_anchor"] = abs((cand_offset + cand_pre) - base_pre_abs)
    out["cand_overlap_margin"] = cand_pre - cand_ovl
    out["cand_pre_cons_margin"] = cand_cons - cand_pre
    out["cand_leak_risk"] = max(0.0, -out["cand_cutoff_to_next_anchor_ms"])
    silence_ratio = max(
        _to_float(out.get("db_silence_ratio"), 0.0),
        _to_float(out.get("mel_window_silence_ratio"), 0.0),
    )
    out["cand_silence_risk"] = 1.0 if silence_ratio >= 0.65 else 0.0
    jump_blocked = _to_float(base_row.get("jump_blocked_flag"), 0.0) > 0.0
    anchor_jump = abs(_to_float(out.get("cand_pre_to_expected_ms"), 0.0))
    out["cand_jump_risk"] = 1.0 if jump_blocked or anchor_jump >= 28.0 else 0.0
    return canonicalize_selector_feature_row(out)


def _quality_against_manual(base_row: Dict[str, object], candidate: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    lang = str(base_row.get("language", "") or "").strip().lower()
    fmt = normalize_format_type(lang, str(base_row.get("format_type", "") or ""))
    family = infer_alias_family(lang, base_row)
    weights = selector_error_weights(lang, fmt, alias_family=family)
    manual_offset = _to_float(base_row.get("manual_offset"), 0.0)
    manual_cons = _to_float(base_row.get("manual_cons"), 0.0)
    manual_cut_abs = _to_float(base_row.get("base_cutoff_abs"), 0.0) + _to_float(base_row.get("delta_cutoff"), 0.0)
    manual_pre = _to_float(base_row.get("manual_pre"), 0.0)
    manual_ovl = _to_float(base_row.get("manual_ovl"), 0.0)

    err_offset = abs(_to_float(candidate.get("offset"), 0.0) - manual_offset)
    err_cons = abs(_to_float(candidate.get("cons"), 0.0) - manual_cons)
    err_cutoff = abs(_to_float(candidate.get("cutoff_abs"), 0.0) - manual_cut_abs)
    err_pre = abs(_to_float(candidate.get("pre"), 0.0) - manual_pre)
    err_ovl = abs(_to_float(candidate.get("ovl"), 0.0) - manual_ovl)
    weighted = (
        float(weights.get("offset", 1.25)) * err_offset
        + float(weights.get("cons", 0.85)) * err_cons
        + float(weights.get("cutoff", 0.95)) * err_cutoff
        + float(weights.get("pre", 1.10)) * err_pre
        + float(weights.get("ovl", 0.45)) * err_ovl
    )
    return -float(weighted), {
        "selector_error_weighted": float(weighted),
        "selector_error_offset": float(err_offset),
        "selector_error_cons": float(err_cons),
        "selector_error_cutoff": float(err_cutoff),
        "selector_error_pre": float(err_pre),
        "selector_error_ovl": float(err_ovl),
    }


def selector_dataset_fieldnames() -> List[str]:
    return [
        "voicebank_id",
        "wav",
        "alias",
        "wav_norm",
        "alias_norm",
        "occurrence_index",
        "line_index",
        "source_oto_id",
        "source_row_id",
        "selector_group_id",
        "selector_rank_label",
        "selector_is_best",
        "selector_quality_score",
        "selector_error_weighted",
        "selector_error_offset",
        "selector_error_cons",
        "selector_error_cutoff",
        "selector_error_pre",
        "selector_error_ovl",
        *SELECTOR_FEATURE_NAMES,
    ]


def _build_selector_group_id(base_row: Dict[str, object]) -> str:
    source_row_id = str(base_row.get("source_row_id", "") or "").strip()
    if source_row_id:
        return f"{base_row.get('voicebank_id', '')}|{source_row_id}"
    source_oto_id = str(base_row.get("source_oto_id", "") or "").strip()
    if source_oto_id:
        return (
            f"{base_row.get('voicebank_id', '')}|{source_oto_id}|"
            f"{int(_to_float(base_row.get('line_index'), 0.0))}|"
            f"{base_row.get('alias_norm', '')}|"
            f"{int(_to_float(base_row.get('occurrence_index'), 0.0))}"
        )
    return (
        f"{base_row.get('voicebank_id', '')}|"
        f"{base_row.get('wav_norm', base_row.get('wav', ''))}|"
        f"{base_row.get('alias_norm', '')}|"
        f"{int(_to_float(base_row.get('occurrence_index'), 0.0))}|"
        f"{int(_to_float(base_row.get('line_index'), 0.0))}"
    )


def build_selector_training_rows_from_delta_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for raw_row in rows:
        base_row = dict(raw_row)
        candidates = build_selector_candidates(str(base_row.get("language", "") or ""), base_row)
        selector_rows: List[Dict[str, object]] = []
        group_id = _build_selector_group_id(base_row)
        for candidate in candidates:
            feature_row = build_selector_feature_row(str(base_row.get("language", "") or ""), base_row, candidate)
            quality_score, err = _quality_against_manual(base_row, candidate)
            merged = {
                "voicebank_id": base_row.get("voicebank_id", ""),
                "wav": base_row.get("wav", ""),
                "alias": base_row.get("alias", ""),
                "wav_norm": base_row.get("wav_norm", ""),
                "alias_norm": base_row.get("alias_norm", ""),
                "occurrence_index": int(_to_float(base_row.get("occurrence_index"), 0.0)),
                "line_index": int(_to_float(base_row.get("line_index"), 0.0)),
                "source_oto_id": str(base_row.get("source_oto_id", "") or ""),
                "source_row_id": str(base_row.get("source_row_id", "") or ""),
                "selector_group_id": group_id,
                "selector_quality_score": float(quality_score),
                **err,
                **feature_row,
            }
            selector_rows.append(merged)
        selector_rows.sort(key=lambda row: float(row.get("selector_quality_score", 0.0)), reverse=True)
        total = len(selector_rows)
        for rank, row in enumerate(selector_rows):
            row["selector_rank_label"] = int(total - rank)
            row["selector_is_best"] = 1 if rank == 0 else 0
            out.append(row)
    return out


def read_delta_dataset_rows(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_selector_dataset_csv(path: str, rows: Sequence[Dict[str, object]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selector_dataset_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            merged = dict(row)
            merged.update(canonicalize_selector_feature_row(row))
            writer.writerow(merged)
    return path


def build_selector_dataset_csv_from_delta_dataset(
    delta_dataset_csv: str,
    out_csv: str,
    *,
    language: str = "",
    format_type: str = "",
    alias_types: Sequence[str] | None = None,
    alias_groups: Sequence[str] | None = None,
    require_train_keep: bool = False,
    min_mapping_confidence: float = 0.0,
    exclude_nuclei_fallback: bool = False,
) -> Dict[str, object]:
    rows = read_delta_dataset_rows(delta_dataset_csv)
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    alias_type_set = {str(v).strip().lower() for v in (alias_types or []) if str(v).strip()}
    alias_group_set = {str(v).strip().lower() for v in (alias_groups or []) if str(v).strip()}
    filtered = []
    for row in rows:
        if lang and str(row.get("language", "") or "").strip().lower() != lang:
            continue
        if fmt and str(row.get("format_type", "") or "").strip().lower() != fmt:
            continue
        if alias_type_set and str(row.get("alias_type", "") or "").strip().lower() not in alias_type_set:
            continue
        if alias_group_set and str(row.get("alias_group", "") or "").strip().lower() not in alias_group_set:
            continue
        if require_train_keep:
            try:
                if int(float(row.get("train_keep_default", 0) or 0)) <= 0:
                    continue
            except Exception:
                continue
        if float(min_mapping_confidence) > 0.0:
            try:
                if float(row.get("mapping_confidence", 0.0) or 0.0) < float(min_mapping_confidence):
                    continue
            except Exception:
                continue
        if exclude_nuclei_fallback:
            try:
                if int(float(row.get("used_nuclei_fallback", 0) or 0)) > 0:
                    continue
            except Exception:
                continue
        filtered.append(row)
    selector_rows = build_selector_training_rows_from_delta_rows(filtered)
    write_selector_dataset_csv(out_csv, selector_rows)
    return {
        "rows": int(len(selector_rows)),
        "groups": int(len({str(r.get("selector_group_id", "") or "") for r in selector_rows})),
        "out_csv": os.path.abspath(out_csv),
    }


def select_best_candidate(
    language: str,
    base_row: Dict[str, object],
    score_callback,
) -> Optional[Dict[str, object]]:
    candidates = build_selector_candidates(language, base_row)
    if not candidates:
        return None
    scored: List[Tuple[float, Dict[str, object], Dict[str, object]]] = []
    for candidate in candidates:
        feature_row = build_selector_feature_row(language, base_row, candidate)
        score = float(score_callback(feature_row))
        scored.append((score, candidate, feature_row))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate, best_feature_row = scored[0]
    return {
        "score": float(best_score),
        "candidate": dict(best_candidate),
        "feature_row": dict(best_feature_row),
        "scores": [
            {
                "candidate_mode": str(candidate.get("candidate_mode", "") or ""),
                "score": float(score),
            }
            for score, candidate, _feature in scored
        ],
    }


__all__ = [
    "SELECTOR_CATEGORICAL_FEATURES",
    "SELECTOR_FEATURE_NAMES",
    "build_selector_candidates",
    "build_selector_dataset_csv_from_delta_dataset",
    "build_selector_feature_row",
    "build_selector_training_rows_from_delta_rows",
    "canonicalize_selector_feature_row",
    "get_selector_feature_schema",
    "select_best_candidate",
    "write_selector_dataset_csv",
    "write_selector_feature_schema",
]
