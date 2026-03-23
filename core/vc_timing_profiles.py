from __future__ import annotations

import copy


JA_SIBILANT_ONSETS = {
    "s",
    "z",
    "sh",
    "j",
    "ts",
    "dz",
    "ch",
}

JA_PLOSIVE_ONSETS = {
    "k",
    "g",
    "t",
    "d",
    "b",
    "p",
    "q",
    "c",
    "kk",
    "tt",
    "pp",
    "dd",
    "gg",
    "bb",
    "ky",
    "gy",
    "ty",
    "dy",
    "by",
    "py",
}

JA_NASAL_ONSETS = {"m", "n", "ny", "ng", "ngy"}
JA_LIQUID_ONSETS = {"r", "ry", "l"}
JA_GLIDE_ONSETS = {"y", "w"}
JA_FRICATIVE_ONSETS = {"h", "f", "v", "hy", "s", "z", "sh"}


JA_CVVC_BRIDGE_TIMING = {
    "default": {
        "offset_pad": 86.0,
        "offset_pad_min": 42.0,
        "offset_pad_floor": 36.0,
        "offset_len_mul": 0.08,
        "pre_lead_mul": 0.35,
        "pre_lead_min": 8.0,
        "pre_lead_max": 30.0,
        "ovl_ratio": 0.50,
        "ovl_min_ratio": 0.40,
        "ovl_pre_margin": 6.0,
        "tail_margin_base": 10.0,
        "tail_margin_mul": 0.05,
        "cons_add_base": 36.0,
        "cons_add_mul": 0.12,
        "cons_add_min": 20.0,
        "cons_add_max": 68.0,
        "cons_floor": 18.0,
        "cons_to_next_margin": 8.0,
        "cut_add_base": 58.0,
        "cut_add_mul": 0.20,
        "cut_add_min": 34.0,
        "cut_add_max": 120.0,
        "cut_min_gap": 16.0,
        "cut_to_next_allow": 22.0,
    },
    "plosive": {
        "offset_pad": 92.0,
        "offset_len_mul": 0.07,
        "pre_lead_mul": 0.42,
        "ovl_ratio": 0.44,
        "ovl_min_ratio": 0.34,
        "tail_margin_base": 8.0,
        "tail_margin_mul": 0.03,
        "cons_add_base": 22.0,
        "cons_add_mul": 0.08,
        "cons_add_max": 46.0,
        "cons_floor": 14.0,
        "cut_add_base": 40.0,
        "cut_add_mul": 0.16,
        "cut_add_max": 82.0,
        "cut_to_next_allow": 16.0,
    },
    "sibilant": {
        "offset_pad": 114.0,
        "offset_len_mul": 0.10,
        "pre_lead_mul": 0.30,
        "ovl_ratio": 0.58,
        "ovl_min_ratio": 0.48,
        "tail_margin_base": 13.0,
        "tail_margin_mul": 0.06,
        "cons_add_base": 44.0,
        "cons_add_mul": 0.15,
        "cons_add_max": 82.0,
        "cut_add_base": 78.0,
        "cut_add_mul": 0.24,
        "cut_add_max": 148.0,
        "cut_to_next_allow": 26.0,
    },
    "nasal": {
        "offset_pad": 70.0,
        "offset_len_mul": 0.06,
        "pre_lead_mul": 0.22,
        "ovl_ratio": 0.62,
        "ovl_min_ratio": 0.50,
        "tail_margin_base": 11.0,
        "tail_margin_mul": 0.05,
        "cons_add_base": 56.0,
        "cons_add_mul": 0.16,
        "cons_add_max": 94.0,
        "cut_add_base": 88.0,
        "cut_add_mul": 0.22,
        "cut_add_max": 168.0,
        "cut_to_next_allow": 32.0,
    },
    "liquid": {
        "offset_pad": 64.0,
        "offset_len_mul": 0.06,
        "pre_lead_mul": 0.24,
        "ovl_ratio": 0.60,
        "ovl_min_ratio": 0.48,
        "tail_margin_base": 10.0,
        "tail_margin_mul": 0.05,
        "cons_add_base": 52.0,
        "cons_add_mul": 0.14,
        "cons_add_max": 88.0,
        "cut_add_base": 84.0,
        "cut_add_mul": 0.20,
        "cut_add_max": 152.0,
        "cut_to_next_allow": 28.0,
    },
    "glide": {
        "offset_pad": 66.0,
        "offset_len_mul": 0.06,
        "pre_lead_mul": 0.26,
        "ovl_ratio": 0.56,
        "ovl_min_ratio": 0.46,
        "tail_margin_base": 10.0,
        "tail_margin_mul": 0.05,
        "cons_add_base": 46.0,
        "cons_add_mul": 0.13,
        "cons_add_max": 84.0,
        "cut_add_base": 78.0,
        "cut_add_mul": 0.20,
        "cut_add_max": 148.0,
        "cut_to_next_allow": 26.0,
    },
    "fricative": {
        "offset_pad": 102.0,
        "offset_len_mul": 0.09,
        "pre_lead_mul": 0.30,
        "ovl_ratio": 0.55,
        "ovl_min_ratio": 0.46,
        "tail_margin_base": 12.0,
        "tail_margin_mul": 0.05,
        "cons_add_base": 42.0,
        "cons_add_mul": 0.14,
        "cons_add_max": 84.0,
        "cut_add_base": 72.0,
        "cut_add_mul": 0.22,
        "cut_add_max": 148.0,
        "cut_to_next_allow": 26.0,
    },
}


def classify_ja_cvvc_bridge_group(onset: str) -> str:
    o = str(onset or "").strip().lower()
    if o in JA_SIBILANT_ONSETS:
        return "sibilant"
    if o in JA_PLOSIVE_ONSETS:
        return "plosive"
    if o in JA_NASAL_ONSETS:
        return "nasal"
    if o in JA_LIQUID_ONSETS:
        return "liquid"
    if o in JA_GLIDE_ONSETS:
        return "glide"
    if o in JA_FRICATIVE_ONSETS:
        return "fricative"
    return "default"


def get_ja_cvvc_bridge_profile(onset: str) -> dict:
    key = classify_ja_cvvc_bridge_group(onset)
    merged = dict(JA_CVVC_BRIDGE_TIMING["default"])
    merged.update(JA_CVVC_BRIDGE_TIMING.get(key, {}))
    return merged


KR_STOPLIKE_BRIDGE_TOKENS = {
    "g",
    "k",
    "kk",
    "gg",
    "d",
    "t",
    "tt",
    "dd",
    "b",
    "p",
    "bb",
    "pp",
}
KR_FRICATIVE_BRIDGE_TOKENS = {"s", "ss", "h", "j", "jj", "ch", "c", "ts"}
KR_SONORANT_CODA_TOKENS = {"n", "m", "ng", "l", "r"}


def classify_kr_vc_coda_group(c_token: str = "", coda_canon: str = "") -> str:
    right = str(c_token or "").strip().lower()
    canon = str(coda_canon or "").strip().lower()
    if right in KR_STOPLIKE_BRIDGE_TOKENS or (not right and canon in {"k", "t", "p"}):
        return "stop"
    if right in KR_FRICATIVE_BRIDGE_TOKENS:
        return "fricative"
    if canon in KR_SONORANT_CODA_TOKENS:
        return "sonorant"
    return "other"


_KR_VC_PLOSIVE_CODA_PROFILES = {
    "hard_stop": {
        "boundary_back_ms": 14.0,
        "boundary_floor_from_vowel_start": 12.0,
        "coda_margin_ms": 12.0,
        "pre_min": 42.0,
        "pre_max": 132.0,
        "tail_floor": 10.0,
        "cons_mul": 0.54,
        "cons_min": 10.0,
        "cons_max": 30.0,
        "cut_mul": 0.56,
        "cut_min": 20.0,
        "cut_max": 52.0,
        "next_onset_pre_floor_gap": 12.0,
        "cutoff_soft_cap_to_next": 1.2,
        "cutoff_min_gap": 8.0,
        "cutoff_min_gap_retry": 6.0,
    },
    "soft_stop": {
        "boundary_back_ms": 11.0,
        "boundary_floor_from_vowel_start": 14.0,
        "coda_margin_ms": 9.0,
        "pre_min": 45.0,
        "pre_max": 145.0,
        "tail_floor": 12.0,
        "cons_mul": 0.62,
        "cons_min": 12.0,
        "cons_max": 38.0,
        "cut_mul": 0.70,
        "cut_min": 24.0,
        "cut_max": 64.0,
        "next_onset_pre_floor_gap": 12.0,
        "cutoff_soft_cap_to_next": 0.8,
        "cutoff_min_gap": 10.0,
        "cutoff_min_gap_retry": 8.0,
    },
}


def get_kr_vc_plosive_coda_profile(coda_canon: str) -> dict:
    canon = str(coda_canon or "").strip().lower()
    key = "hard_stop" if canon in {"t", "p"} else "soft_stop"
    return dict(_KR_VC_PLOSIVE_CODA_PROFILES[key])


_KR_VC_FALLBACK_PROFILES = {
    "default": {
        "offset_padding_base": 180.0,
        "offset_padding_ratio_short": 0.80,
        "offset_padding_min_short": 50.0,
        "next_onset_pre_floor_gap": 10.0,
    },
    "stop": {
        "n_ref_min": 90.0,
        "cons_mul": 0.30,
        "cons_min": 22.0,
        "cons_max": 58.0,
        "cut_mul": 0.34,
        "cut_min": 18.0,
        "cut_max": 42.0,
        "cons_cap_before_next": 7.0,
        "cons_floor_gap": 6.0,
        "cut_seed_before_next": 1.2,
        "cut_cap_before_next": 1.0,
        "cut_floor_gap": 4.0,
    },
    "fricative": {
        "n_ref_min": 80.0,
        "cons_mul": 0.42,
        "cons_min": 28.0,
        "cons_max": 84.0,
        "cut_mul": 0.48,
        "cut_min": 22.0,
        "cut_max": 58.0,
        "cons_cap_after_next": 8.0,
        "cons_floor_gap": 10.0,
        "cut_cap_after_next": 12.0,
        "cut_floor_gap": 8.0,
    },
    "sonorant": {
        "n_ref_min": 80.0,
        "cons_mul": 0.58,
        "cons_min": 34.0,
        "cons_max": 132.0,
        "cut_mul": 0.54,
        "cut_min": 26.0,
        "cut_max": 72.0,
        "cons_cap_after_next": 26.0,
        "cons_floor_gap": 14.0,
        "cut_cap_after_next": 30.0,
        "cut_floor_gap": 10.0,
    },
    "other": {
        "n_ref_min": 72.0,
        "cons_mul": 0.50,
        "cons_min": 28.0,
        "cons_max": 108.0,
        "cut_mul": 0.50,
        "cut_min": 22.0,
        "cut_max": 64.0,
        "cons_cap_after_next": 18.0,
        "cons_floor_gap": 12.0,
        "cut_cap_after_next": 22.0,
        "cut_floor_gap": 8.0,
    },
}


def get_kr_vc_fallback_profile(coda_group: str) -> dict:
    key = str(coda_group or "").strip().lower()
    merged = dict(_KR_VC_FALLBACK_PROFILES["default"])
    merged.update(_KR_VC_FALLBACK_PROFILES.get(key, _KR_VC_FALLBACK_PROFILES["other"]))
    return merged


_KR_CVVC_DIRECT_PROFILES = {
    "stop": {
        "tail_keep_mul": 0.42,
        "tail_keep_min": 56.0,
        "tail_keep_max": 118.0,
        "ovl_gap_mul": 0.52,
        "ovl_gap_min": 78.0,
        "ovl_gap_max": 122.0,
        "cons_gap_mul": 0.44,
        "cons_gap_min": 34.0,
        "cons_gap_max": 74.0,
        "cut_gap_mul": 0.58,
        "cut_gap_min": 30.0,
        "cut_gap_max": 64.0,
        "cutoff_margin": 16.0,
        "pre_floor": 112.0,
        "pre_ceil": 214.0,
        "cons_cap_after_onset": 8.0,
        "min_cons_gap": 16.0,
        "next_seg_end_margin": 8.0,
        "min_cut_tail": 8.0,
    },
    "fricative": {
        "tail_keep_mul": 0.48,
        "tail_keep_min": 66.0,
        "tail_keep_max": 132.0,
        "ovl_gap_mul": 0.56,
        "ovl_gap_min": 86.0,
        "ovl_gap_max": 132.0,
        "cons_gap_mul": 0.50,
        "cons_gap_min": 40.0,
        "cons_gap_max": 88.0,
        "cut_gap_mul": 0.66,
        "cut_gap_min": 34.0,
        "cut_gap_max": 80.0,
        "cutoff_margin": 22.0,
        "pre_floor": 124.0,
        "pre_ceil": 228.0,
        "cons_cap_after_onset": 16.0,
        "min_cons_gap": 16.0,
        "next_seg_end_margin": 10.0,
        "min_cut_tail": 8.0,
    },
    "nasal": {
        "tail_keep_mul": 0.54,
        "tail_keep_min": 82.0,
        "tail_keep_max": 142.0,
        "ovl_gap_mul": 0.58,
        "ovl_gap_min": 92.0,
        "ovl_gap_max": 138.0,
        "cons_gap_mul": 0.52,
        "cons_gap_min": 42.0,
        "cons_gap_max": 88.0,
        "cut_gap_mul": 0.72,
        "cut_gap_min": 38.0,
        "cut_gap_max": 86.0,
        "cutoff_margin": 26.0,
        "pre_floor": 138.0,
        "pre_ceil": 248.0,
        "cons_cap_after_onset": 22.0,
        "min_cons_gap": 18.0,
        "next_seg_end_margin": 10.0,
        "min_cut_tail": 10.0,
    },
    "liquid": {
        "tail_keep_mul": 0.62,
        "tail_keep_min": 96.0,
        "tail_keep_max": 166.0,
        "ovl_gap_mul": 0.64,
        "ovl_gap_min": 104.0,
        "ovl_gap_max": 152.0,
        "cons_gap_mul": 0.96,
        "cons_gap_min": 82.0,
        "cons_gap_max": 158.0,
        "cut_gap_mul": 0.88,
        "cut_gap_min": 52.0,
        "cut_gap_max": 118.0,
        "cutoff_margin": 42.0,
        "pre_floor": 168.0,
        "pre_ceil": 286.0,
        "cons_cap_after_onset": 42.0,
        "min_cons_gap": 28.0,
        "next_seg_end_margin": 14.0,
        "min_cut_tail": 14.0,
    },
    "other": {
        "tail_keep_mul": 0.50,
        "tail_keep_min": 72.0,
        "tail_keep_max": 136.0,
        "ovl_gap_mul": 0.56,
        "ovl_gap_min": 88.0,
        "ovl_gap_max": 132.0,
        "cons_gap_mul": 0.58,
        "cons_gap_min": 40.0,
        "cons_gap_max": 96.0,
        "cut_gap_mul": 0.74,
        "cut_gap_min": 36.0,
        "cut_gap_max": 92.0,
        "cutoff_margin": 28.0,
        "pre_floor": 128.0,
        "pre_ceil": 232.0,
        "cons_cap_after_onset": 28.0,
        "min_cons_gap": 16.0,
        "next_seg_end_margin": 12.0,
        "min_cut_tail": 12.0,
    },
}


def get_kr_cvvc_direct_profile(coda_group: str) -> dict:
    key = str(coda_group or "").strip().lower()
    return copy.deepcopy(_KR_CVVC_DIRECT_PROFILES.get(key, _KR_CVVC_DIRECT_PROFILES["other"]))


def classify_kr_cvvc_direct_group(right_token: str, coda_canon: str) -> str:
    right = str(right_token or "").strip().lower()
    canon = str(coda_canon or "").strip().lower()
    if right in KR_STOPLIKE_BRIDGE_TOKENS or (not right and canon in {"k", "t", "p"}):
        return "stop"
    if right in KR_FRICATIVE_BRIDGE_TOKENS:
        return "fricative"
    if canon in {"m", "n", "ng"}:
        return "nasal"
    if canon in {"l", "r"}:
        return "liquid"
    return "other"


__all__ = [
    "JA_CVVC_BRIDGE_TIMING",
    "JA_FRICATIVE_ONSETS",
    "JA_GLIDE_ONSETS",
    "JA_LIQUID_ONSETS",
    "JA_NASAL_ONSETS",
    "JA_PLOSIVE_ONSETS",
    "JA_SIBILANT_ONSETS",
    "KR_FRICATIVE_BRIDGE_TOKENS",
    "KR_SONORANT_CODA_TOKENS",
    "KR_STOPLIKE_BRIDGE_TOKENS",
    "classify_ja_cvvc_bridge_group",
    "classify_kr_cvvc_direct_group",
    "classify_kr_vc_coda_group",
    "get_ja_cvvc_bridge_profile",
    "get_kr_cvvc_direct_profile",
    "get_kr_vc_fallback_profile",
    "get_kr_vc_plosive_coda_profile",
]
