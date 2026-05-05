from __future__ import annotations

from core.ja_mapping_select_v2 import _mel_guided_ja_cvvc_adjustment
from core.kr_mapping_select_v2 import _mel_guided_cvvc_adjustment, _token_invariant_guard_idx
import core.kr_oto_bridge as kr_bridge
from core.kr_oto_postprocess import guard_kr_vcv_pre_to_cv_boundary
from core.kr_timing_v2 import prepare_vcv_syllable_timing


_VOWELS = ("a", "e", "i", "o", "u", "eo", "eu", "oe", "wi", "wa", "wo", "we", "ui", "yu", "ya", "ye", "yo")


def _split_syllable_parts(token: str):
    text = str(token or "").strip().lower()
    if not text:
        return "", "", ""
    longest = ""
    pos = -1
    for v in _VOWELS:
        idx = text.find(v)
        if idx < 0:
            continue
        if len(v) > len(longest) or (len(v) == len(longest) and (pos < 0 or idx < pos)):
            longest = v
            pos = idx
    if not longest:
        return text, "", ""
    onset = text[:pos]
    coda = text[pos + len(longest) :]
    return onset, longest, coda


def test_kr_mel_remap_does_not_switch_to_vowel_mismatch():
    romaji = ["ga", "ge", "gi"]
    syllables_info = [
        {
            "mel_voiced_formant_conf": 0.10,
            "mel_unvoiced_diffuse_conf": 0.05,
            "mel_silence_sparse_conf": 0.05,
            "mel_breath_like_conf": 0.00,
        },
        {
            "mel_voiced_formant_conf": 1.00,
            "mel_unvoiced_diffuse_conf": 0.80,
            "mel_silence_sparse_conf": 0.00,
            "mel_breath_like_conf": 0.00,
        },
        {
            "mel_voiced_formant_conf": 0.15,
            "mel_unvoiced_diffuse_conf": 0.10,
            "mel_silence_sparse_conf": 0.00,
            "mel_breath_like_conf": 0.00,
        },
    ]

    def _score(target: str, cand: str) -> float:
        if target == cand:
            return 100.0
        return 96.0 if target[:1] == cand[:1] else 60.0

    out_idx, changed = _mel_guided_cvvc_adjustment(
        file_format="cvvc",
        alias_type="cv",
        target_clean="ga",
        expected_idx=0,
        selected_idx=0,
        row_mapping_confidence=0.44,
        file_mapping_conf_th=0.62,
        max_search_fwd=2,
        romaji_syllables=romaji,
        syllables_info=syllables_info,
        syllable_blank_confidences=[0.05, 0.05, 0.05],
        split_syllable_parts_fn=_split_syllable_parts,
        cv_match_score_fn=_score,
    )

    assert out_idx == 0
    assert changed is False


def test_ja_mel_remap_rejects_soft_zero_candidate():
    syllables_info = [
        {
            "roman": "ka",
            "blank_confidence": 0.10,
            "mel_voiced_formant_conf": 0.10,
            "mel_unvoiced_diffuse_conf": 0.05,
            "mel_silence_sparse_conf": 0.05,
            "mel_breath_like_conf": 0.00,
        },
        {
            "roman": "ke",
            "blank_confidence": 0.08,
            "mel_voiced_formant_conf": 1.00,
            "mel_unvoiced_diffuse_conf": 0.80,
            "mel_silence_sparse_conf": 0.00,
            "mel_breath_like_conf": 0.00,
        },
    ]

    out_idx, changed = _mel_guided_ja_cvvc_adjustment(
        format_type="cvvc",
        alias_type="cv",
        target_tok="ka",
        expected_idx=0,
        selected_idx=0,
        mapping_confidence=0.40,
        mapping_conf_threshold=0.62,
        max_search_fwd=2,
        syllables_info=syllables_info,
        normalize_syllable_token_fn=lambda s: str(s or "").strip().lower(),
        syllable_info_token_fn=lambda row: row.get("roman", ""),
        syllable_confidence_by_idx=None,
        order_locked=False,
        mel_remap_enabled=True,
    )

    assert out_idx == 0
    assert changed is False


def test_kr_cvvc_direct_keeps_tail_in_target_band():
    original_validate = kr_bridge._validate_oto_params
    kr_bridge._validate_oto_params = lambda offset, consonant, cutoff, pre, ovl, alias_type="": (
        float(offset),
        float(consonant),
        float(cutoff),
        float(pre),
        float(ovl),
    )
    try:
        params = kr_bridge._compute_kr_cvvc_vc_timing_direct("a g", 0.0, 180.0, 212.0, 340.0)
    finally:
        kr_bridge._validate_oto_params = original_validate
    assert params is not None
    offset, _cons, _cut, _pre, _ovl = params
    carried_tail = 180.0 - float(offset)
    assert carried_tail >= 58.0
    assert carried_tail <= 78.0


def test_kr_vcv_timing_keeps_consonant_near_onset_not_tail():
    class _Phone:
        def __init__(self, mark: str):
            self.mark = mark

    def _prepare_bounds(_syllables, idx):
        if idx == 0:
            return 0, [_Phone("m")], 0.0, 60.0, 60.0, 160.0
        return 1, [_Phone("g")], 180.0, 240.0, 240.0, 420.0

    def _adaptive_overlap(pre, _hint, mode="vcv"):
        assert mode == "vcv"
        return max(float(pre) * 0.5, 12.0)

    def _validate(offset, consonant, cutoff, pre, ovl):
        return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

    _idx, _seq, offset, consonant, _cutoff, pre, _ovl = prepare_vcv_syllable_timing(
        [{"phones": [1]}, {"phones": [1]}],
        current_w_idx=1,
        cv_seq_idx=1,
        diphthong_cv_consonant_ratio=0.5,
        forced_w_idx=1,
        prepare_cv_bounds_fn=_prepare_bounds,
        adaptive_overlap_fn=_adaptive_overlap,
        validate_fn=_validate,
    )

    onset_rel = 240.0 - float(offset)
    cv_end_rel = 420.0 - float(offset)
    assert consonant >= max(pre + 16.0, onset_rel + 14.0)
    assert consonant <= cv_end_rel - 8.0


def test_kr_vcv_timing_respects_prev_vowel_tail_band():
    class _Phone:
        def __init__(self, mark: str):
            self.mark = mark

    def _prepare_bounds(_syllables, idx):
        if idx == 0:
            return 0, [_Phone("a")], 0.0, 80.0, 80.0, 220.0
        return 1, [_Phone("s")], 240.0, 300.0, 300.0, 420.0

    _idx, _seq, offset, _consonant, _cutoff, _pre, _ovl = prepare_vcv_syllable_timing(
        [{"phones": [1]}, {"phones": [1]}],
        current_w_idx=1,
        cv_seq_idx=1,
        diphthong_cv_consonant_ratio=0.5,
        forced_w_idx=1,
        prepare_cv_bounds_fn=_prepare_bounds,
        adaptive_overlap_fn=lambda pre, _hint, mode="vcv": max(float(pre) * 0.5, 12.0),
        validate_fn=lambda o, c, k, p, v: (float(o), float(c), float(k), float(p), float(v)),
    )
    prev_v_start = 80.0
    prev_v_end = 220.0
    prev_len = prev_v_end - prev_v_start
    carried_tail = prev_v_end - float(offset)
    assert carried_tail >= (prev_len / 3.0) - 1.0
    assert carried_tail <= (prev_len * 0.40) + 1.0


def test_kr_token_invariant_guard_reverts_vowel_mismatch():
    romaji = ["ga", "ge", "gi"]
    out_idx, guarded = _token_invariant_guard_idx(
        alias_type="cv",
        target_clean="ga",
        expected_idx=0,
        selected_idx=1,
        row_mapping_confidence=0.64,
        file_mapping_conf_th=0.60,
        file_mapping_low_conf=False,
        romaji_syllables=romaji,
        syllable_blank_confidences=[0.05, 0.05, 0.05],
        split_syllable_parts_fn=_split_syllable_parts,
        cv_match_score_fn=lambda target, cand: 100.0 if target == cand else 70.0,
    )
    assert guarded is True
    assert out_idx == 0


def test_kr_vcv_pre_guard_aligns_to_current_onset():
    class _Ph:
        def __init__(self, min_t, max_t, mark):
            self.minTime = min_t
            self.maxTime = max_t
            self.mark = mark

    syllables = [
        {"phones": [_Ph(0.00, 0.08, "a"), _Ph(0.08, 0.22, "a")]},
        {"phones": [_Ph(0.24, 0.30, "g"), _Ph(0.30, 0.42, "a")]},
    ]
    offset, consonant, cutoff, pre, ovl = guard_kr_vcv_pre_to_cv_boundary(
        130.0,
        160.0,
        -260.0,
        110.0,
        70.0,
        1,
        syllables,
        lambda o, c, k, p, v: (float(o), float(c), float(k), float(p), float(v)),
    )
    pre_abs = float(offset) + float(pre)
    onset_abs = 300.0
    assert abs(pre_abs - onset_abs) <= 6.0
    assert float(consonant) >= float(pre) + 12.0
