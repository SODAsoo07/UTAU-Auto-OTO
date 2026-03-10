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
            "n": 8025,
            "pre": 130.6420,
            "cons_gap": 109.7500,
            "cut_gap": 168.9300,
            "ovl_ratio": 0.3333333,
            "head_off_ratio": 0.0941946,
        },
        "cv": {
            "n": 7055,
            "pre": 89.4080,
            "cons_gap": 118.8200,
            "cut_gap": 180.0,
            "ovl_ratio": 0.3409029,
            "head_off_ratio": 0.2690744,
        },
        "vcv": {
            "n": 54903,
            "pre": 230.1600,
            "cons_gap": 100.0000,
            "cut_gap": 180.0000,
            "ovl_ratio": 0.4285406,
            "head_off_ratio": 0.1317619,
        },
        "vc": {
            "n": 2906,
            "pre": 173.2100,
            "cons_gap": 51.5375,
            "cut_gap": 68.8880,
            "ovl_ratio": 0.3834407,
            "head_off_ratio": None,
        },
        "vv": {
            "n": 2723,
            "pre": 223.9580,
            "cons_gap": 125.0000,
            "cut_gap": 180.0,
            "ovl_ratio": 0.4555644,
            "head_off_ratio": 0.1807262,
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
            "n": 3594,
            "pre": 77.3060,
            "cons_gap": 72.4440,
            "cut_gap": 180.0000,
            "ovl_ratio": 0.3333333,
            "head_off_ratio": 0.1827004,
        },
        "cv": {
            "n": 14415,
            "pre": 79.8100,
            "cons_gap": 83.0000,
            "cut_gap": 150.8000,
            "ovl_ratio": 0.3999496,
            "head_off_ratio": 0.1642942,
        },
        "vcv": {
            "n": 3059,
            "pre": 126.9800,
            "cons_gap": 73.7000,
            "cut_gap": 48.9800,
            "ovl_ratio": 0.4907309,
            "head_off_ratio": 0.2466641,
        },
        "vc": {
            "n": 3051,
            "pre": 118.8200,
            "cons_gap": 46.2600,
            "cut_gap": 74.3100,
            "ovl_ratio": 0.5000000,
            "head_off_ratio": None,
        },
        "vv": {
            "n": 880,
            "pre": 166.8900,
            "cons_gap": 102.0400,
            "cut_gap": 180.0000,
            "ovl_ratio": 0.6432054,
            "head_off_ratio": 0.2268322,
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
            "n": 1134,
            "pre": 190.000,
            "cons_gap": 57.000,
            "cut_gap": 260.000,
            "ovl_ratio": 0.5833333,
            "head_offset": 476.0,
        },
        "vc": {
            "n": 221,
            "pre": 160.000,
            "cons_gap": 51.000,
            "cut_gap": 142.410,
            "ovl_ratio": 0.5000000,
            "head_offset": None,
        },
        "cv": {
            "n": 8,
            "pre": 92.500,
            "cons_gap": 60.000,
            "cut_gap": 17.500,
            "ovl_ratio": 0.4209272,
            "head_offset": None,
        },
        "vcv": {
            "n": 6567,
            "pre": 230.000,
            "cons_gap": 64.000,
            "cut_gap": 260.000,
            "ovl_ratio": 0.5283019,
            "head_offset": None,
        },
        "vv": {
            "n": 355,
            "pre": 160.000,
            "cons_gap": 82.000,
            "cut_gap": 198.780,
            "ovl_ratio": 0.5000000,
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
            "n": 5551,
            "pre": 56.000,
            "cons_gap": 108.565,
            "cut_gap": 142.021,
            "ovl_ratio": 0.2818167,
            "head_offset": None,
        },
        "cv_head": {
            "n": 781,
            "pre": 91.369,
            "cons_gap": 103.103,
            "cut_gap": 142.020,
            "ovl_ratio": 0.3891475,
            "head_offset": 934.660,
        },
        "vc": {
            "n": 4619,
            "pre": 214.284,
            "cons_gap": 31.399,
            "cut_gap": 31.399,
            "ovl_ratio": 0.3333320,
            "head_offset": None,
        },
        "vv": {
            "n": 1435,
            "pre": 248.269,
            "cons_gap": 131.520,
            "cut_gap": 176.843,
            "ovl_ratio": 0.3333364,
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
