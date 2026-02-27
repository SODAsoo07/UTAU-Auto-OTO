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
KR_PROFILE_PRESET_V1 = {
    "version": 1,
    "language": "korean",
    "preset_name": "kr_reference_v1",
    "buckets": {
        "cv_head": {
            "n": 1010,
            "pre": 99.1135,
            "cons_gap": 143.6530,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3999147,
            "head_off_ratio": 0.1441711,
        },
        "cv": {
            "n": 715,
            "pre": 99.0120,
            "cons_gap": 143.6520,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3300585,
            "head_off_ratio": 0.0845636,
        },
        "vcv": {
            "n": 235,
            "pre": 188.3840,
            "cons_gap": 34.5120,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3921524,
            "head_off_ratio": 0.2039032,
        },
        "vc": {
            "n": 145,
            "pre": 195.7850,
            "cons_gap": 45.1810,
            "cut_gap": 180.0,
            "ovl_ratio": 0.4651142,
            "head_off_ratio": None,
        },
        "vv": {
            "n": 116,
            "pre": 261.3330,
            "cons_gap": 109.7265,
            "cut_gap": 180.0,
            "ovl_ratio": 0.5818178,
            "head_off_ratio": 0.2040873,
        },
    },
}


def get_kr_profile_preset():
    """Return a mutable copy of Korean preset profile."""
    return copy.deepcopy(KR_PROFILE_PRESET_V1)


# Derived from multiple Japanese reference banks (VCV-oriented set),
# reduced to anonymized aggregate timing stats.
JA_PROFILE_PRESET_VCV_V1 = {
    "version": 1,
    "language": "japanese",
    "preset_name": "ja_vcv_reference_v2",
    "buckets": {
        "cv_head": {
            "n": 514,
            "pre": 190.0,
            "cons_gap": 125.0,
            "cut_gap": 260.0,
            "ovl_ratio": 0.385,
            "head_offset": 970.0,
        },
        "vc": {
            "n": 3254,
            "pre": 250.0,
            "cons_gap": 118.409,
            "cut_gap": 260.0,
            "ovl_ratio": 0.333,
            "head_offset": None,
        },
        "cv": {
            "n": 159,
            "pre": 51.96,
            "cons_gap": 122.493,
            "cut_gap": 260.0,
            "ovl_ratio": 0.332,
            "head_offset": None,
        },
        "vv": {
            "n": 184,
            "pre": 160.0,
            "cons_gap": 100.0,
            "cut_gap": 16.0,
            "ovl_ratio": 0.5,
            "head_offset": None,
        },
    },
}


# Derived from multiple Japanese reference banks (CVVC-oriented set),
# reduced to anonymized aggregate timing stats.
JA_PROFILE_PRESET_CVVC_V1 = {
    "version": 1,
    "language": "japanese",
    "preset_name": "ja_cvvc_reference_v2",
    "buckets": {
        "cv": {
            "n": 229,
            "pre": 62.59,
            "cons_gap": 83.9,
            "cut_gap": 246.18,
            "ovl_ratio": 0.353,
            "head_offset": None,
        },
        "cv_head": {
            "n": 23,
            "pre": 68.93,
            "cons_gap": 76.19,
            "cut_gap": 260.0,
            "ovl_ratio": 0.218,
            "head_offset": 946.03,
        },
        "vc": {
            "n": 197,
            "pre": 200.0,
            "cons_gap": 36.29,
            "cut_gap": 41.72,
            "ovl_ratio": 0.25,
            "head_offset": None,
        },
        "vv": {
            "n": 48,
            "pre": 300.0,
            "cons_gap": 100.0,
            "cut_gap": 150.8,
            "ovl_ratio": 0.333,
            "head_offset": None,
        },
    },
}


def get_ja_profile_preset(format_type="cvvc"):
    """
    Return a mutable copy of Japanese preset profile by alias format.
    Accepted examples: cvvc/cvc/vcv.
    """
    ft = (format_type or "").strip().lower()
    if ft.startswith("vcv"):
        return copy.deepcopy(JA_PROFILE_PRESET_VCV_V1)
    return copy.deepcopy(JA_PROFILE_PRESET_CVVC_V1)


def get_voice_style_labels_kr():
    """Return display labels for Korean voice-style presets."""
    return copy.deepcopy(VOICE_STYLE_LABELS_KR)
