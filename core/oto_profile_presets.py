"""
Abstract OTO profile presets.

This module intentionally stores only aggregated timing statistics.
No raw oto lines, no wav data, and no personal filesystem paths.
"""

import copy


# Voice style labels for future preset selection UI.
# Keep wording user-friendly in Korean.
VOICE_STYLE_LABELS_KR = {
    "bright": "밝은",
    "dark": "어두운",
    "attack_strong": "강한",
    "breathy_soft": "부드러운(숨 많은)",
}


# Derived from Korean reference voices and reduced to anonymized bucket stats.
# Keys map to alias type from `classify_alias`.
KR_PROFILE_PRESET_GENERAL_V1 = {
    "version": 2,
    "language": "korean",
    "preset_name": "kr_reference_v2",
    "format_type": "general",
    "buckets": {
        "cv_head": {
            "n": 5856,
            "pre": 103.4440,
            "cons_gap": 125.0000,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3207667,
            "head_off_ratio": 0.1888888,
        },
        "cv": {
            "n": 9361,
            "pre": 93.9950,
            "cons_gap": 125.0000,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3750000,
            "head_off_ratio": 0.1328905,
        },
        "vcv": {
            "n": 3685,
            "pre": 195.0180,
            "cons_gap": 52.0070,
            "cut_gap": 101.7310,
            "ovl_ratio": 0.4186049,
            "head_off_ratio": 0.3500000,
        },
        "vc": {
            "n": 3517,
            "pre": 178.7660,
            "cons_gap": 47.4110,
            "cut_gap": 73.7000,
            "ovl_ratio": 0.4295286,
            "head_off_ratio": None,
        },
        "vv": {
            "n": 1504,
            "pre": 250.0000,
            "cons_gap": 125.0000,
            "cut_gap": 180.0,
            "ovl_ratio": 0.4186054,
            "head_off_ratio": 0.2557047,
        },
    },
}


KR_PROFILE_PRESET_CVC_V1 = {
    "version": 3,
    "language": "korean",
    "preset_name": "kr_cvc_reference_v3",
    "format_type": "cvc",
    "buckets": {
        "cv_head": {
            "n": 1620,
            "pre": 89.5700,
            "cons_gap": 88.4300,
            "cut_gap": 185.0900,
            "ovl_ratio": 0.3417048,
            "head_off_ratio": 0.1828282,
        },
        "cv": {
            "n": 9700,
            "pre": 80.1695,
            "cons_gap": 85.2600,
            "cut_gap": 140.5900,
            "ovl_ratio": 0.4062845,
            "head_off_ratio": 0.1655118,
        },
        "vcv": {
            "n": 1583,
            "pre": 126.9800,
            "cons_gap": 83.4500,
            "cut_gap": 60.7700,
            "ovl_ratio": 0.4933263,
            "head_off_ratio": 0.2315166,
        },
        "vc": {
            "n": 2011,
            "pre": 114.5100,
            "cons_gap": 54.4200,
            "cut_gap": 36.2800,
            "ovl_ratio": 0.4919000,
            "head_off_ratio": None,
        },
        "vv": {
            "n": 375,
            "pre": 164.1700,
            "cons_gap": 75.2800,
            "cut_gap": 169.6200,
            "ovl_ratio": 0.6448491,
            "head_off_ratio": 0.2529097,
        },
    },
}


def get_kr_profile_preset(format_type="general"):
    """Return a mutable copy of Korean preset profile by format."""
    ft = (format_type or "").strip().lower()
    if ft.startswith("cvc"):
        return copy.deepcopy(KR_PROFILE_PRESET_CVC_V1)
    return copy.deepcopy(KR_PROFILE_PRESET_GENERAL_V1)


# Derived from multiple Japanese reference banks (VCV-oriented set),
# reduced to anonymized aggregate timing stats.
JA_PROFILE_PRESET_VCV_V1 = {
    "version": 3,
    "language": "japanese",
    "preset_name": "ja_vcv_reference_v3",
    "buckets": {
        "cv_head": {
            "n": 7477,
            "pre": 100.712,
            "cons_gap": 109.754,
            "cut_gap": 208.333,
            "ovl_ratio": 0.333332,
            "head_offset": 809.0,
        },
        "vc": {
            "n": 46223,
            "pre": 250.0,
            "cons_gap": 110.415,
            "cut_gap": 208.333,
            "ovl_ratio": 0.333332,
            "head_offset": None,
        },
        "cv": {
            "n": 1571,
            "pre": 52.927,
            "cons_gap": 133.302,
            "cut_gap": 260.0,
            "ovl_ratio": 0.3145017,
            "head_offset": None,
        },
        "vcv": {
            "n": 1704,
            "pre": 200.0,
            "cons_gap": 100.0,
            "cut_gap": 166.666,
            "ovl_ratio": 0.33333,
            "head_offset": None,
        },
        "vv": {
            "n": 96,
            "pre": 200.0,
            "cons_gap": 100.0,
            "cut_gap": 166.666,
            "ovl_ratio": 0.33333,
            "head_offset": None,
        },
    },
}


# Derived from multiple Japanese reference banks (CVVC-oriented set),
# reduced to anonymized aggregate timing stats.
JA_PROFILE_PRESET_CVVC_V1 = {
    "version": 3,
    "language": "japanese",
    "preset_name": "ja_cvvc_reference_v3",
    "buckets": {
        "cv": {
            "n": 2881,
            "pre": 54.035,
            "cons_gap": 114.027,
            "cut_gap": 157.725,
            "ovl_ratio": 0.3333333,
            "head_offset": None,
        },
        "cv_head": {
            "n": 1005,
            "pre": 58.0,
            "cons_gap": 101.0,
            "cut_gap": 256.0,
            "ovl_ratio": 0.375,
            "head_offset": 927.38,
        },
        "vc": {
            "n": 4902,
            "pre": 214.286,
            "cons_gap": 36.771,
            "cut_gap": 46.43,
            "ovl_ratio": 0.3333318,
            "head_offset": None,
        },
        "vv": {
            "n": 48,
            "pre": 223.17,
            "cons_gap": 94.8,
            "cut_gap": 80.395,
            "ovl_ratio": 0.3878868,
            "head_offset": None,
        },
    },
}


def get_ja_profile_preset(format_type="cvvc"):
    """
    Return a mutable copy of Japanese preset profile by alias format.
    Accepted examples: cv/cvvc/vcv.
    """
    ft = (format_type or "").strip().lower()
    if ft.startswith("vcv"):
        return copy.deepcopy(JA_PROFILE_PRESET_VCV_V1)
    return copy.deepcopy(JA_PROFILE_PRESET_CVVC_V1)


def get_voice_style_labels_kr():
    """Return display labels for Korean voice-style presets."""
    return copy.deepcopy(VOICE_STYLE_LABELS_KR)
