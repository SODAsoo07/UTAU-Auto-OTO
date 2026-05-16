"""Unit tests for core.coarse_crnn.filename_anchor_timeline (Phase H option C)."""
from __future__ import annotations

import pytest

from core.coarse_crnn.filename_anchor_timeline import (
    BASE_TOKEN_DURATION_MS,
    KOREAN_CODA_BONUS_MS,
    KOREAN_GLIDE_BONUS_MS,
    TENSE_CONSONANT_BONUS_MS,
    _ko_token_coda,
    _ko_token_has_glide,
    _ko_token_has_tense_consonant,
    build_filename_anchor_timeline,
    estimate_token_duration_ms,
)


# ---------------------------------------------------------------------------
# Korean coda detection
# ---------------------------------------------------------------------------

def test_coda_detects_ng_suffix():
    assert _ko_token_coda("geung") == "ng"
    assert _ko_token_coda("jjeung") == "ng"


def test_coda_detects_n_m_l_suffixes():
    assert _ko_token_coda("geun") == "n"
    assert _ko_token_coda("geum") == "m"
    assert _ko_token_coda("geul") == "l"


def test_coda_returns_empty_when_no_coda():
    assert _ko_token_coda("ga") == ""
    assert _ko_token_coda("geu") == ""
    assert _ko_token_coda("gyo") == ""


def test_coda_returns_empty_for_blank_or_no_vowel():
    assert _ko_token_coda("") == ""
    assert _ko_token_coda("gg") == ""


# ---------------------------------------------------------------------------
# Glide / tense consonant detection
# ---------------------------------------------------------------------------

def test_glide_detects_yX_wX_clusters():
    assert _ko_token_has_glide("gya") is True
    assert _ko_token_has_glide("gwa") is True
    assert _ko_token_has_glide("kwi") is True


def test_glide_skips_pure_vowel_or_simple_cv():
    assert _ko_token_has_glide("ga") is False
    assert _ko_token_has_glide("a") is False
    assert _ko_token_has_glide("eu") is False
    # 'ya' is glide+vowel but no preceding consonant — not a CV glide cluster
    assert _ko_token_has_glide("ya") is False


def test_tense_consonant_detection():
    assert _ko_token_has_tense_consonant("kkae") is True
    assert _ko_token_has_tense_consonant("jjeu") is True
    assert _ko_token_has_tense_consonant("ppi") is True
    assert _ko_token_has_tense_consonant("ga") is False
    assert _ko_token_has_tense_consonant("geu") is False


# ---------------------------------------------------------------------------
# estimate_token_duration_ms
# ---------------------------------------------------------------------------

def test_estimate_duration_blank_returns_base():
    assert estimate_token_duration_ms("") == pytest.approx(BASE_TOKEN_DURATION_MS)


def test_estimate_duration_simple_cv():
    # 'ga' has no coda, no glide, no tense consonant → base
    assert estimate_token_duration_ms("ga") == pytest.approx(BASE_TOKEN_DURATION_MS)


def test_estimate_duration_with_coda_adds_bonus():
    base = estimate_token_duration_ms("ga")
    with_coda = estimate_token_duration_ms("gan")
    assert with_coda > base
    assert with_coda == pytest.approx(base + KOREAN_CODA_BONUS_MS["n"])


def test_estimate_duration_ng_coda_is_longest():
    assert estimate_token_duration_ms("gang") > estimate_token_duration_ms("gan")
    assert estimate_token_duration_ms("gang") > estimate_token_duration_ms("gam")


def test_estimate_duration_with_glide_adds_bonus():
    base = estimate_token_duration_ms("ga")
    with_glide = estimate_token_duration_ms("gya")
    assert with_glide == pytest.approx(base + KOREAN_GLIDE_BONUS_MS)


def test_estimate_duration_with_tense_adds_bonus():
    base = estimate_token_duration_ms("ga")
    tense = estimate_token_duration_ms("kka")
    assert tense == pytest.approx(base + TENSE_CONSONANT_BONUS_MS)


def test_estimate_duration_stacks_all_bonuses():
    # 'kkyang' = tense consonant + glide + ng coda
    expected = BASE_TOKEN_DURATION_MS + KOREAN_CODA_BONUS_MS["ng"] + KOREAN_GLIDE_BONUS_MS + TENSE_CONSONANT_BONUS_MS
    assert estimate_token_duration_ms("kkyang") == pytest.approx(expected)


def test_estimate_duration_japanese_uses_japanese_base():
    # Japanese kana romaji uses a different base
    ja = estimate_token_duration_ms("ka", language="japanese")
    ko = estimate_token_duration_ms("ka", language="korean")
    # ko: base 400; ja: base 400 (same) but japanese branch handles 'n' differently
    assert ja == pytest.approx(400.0)
    assert ko == pytest.approx(400.0)


def test_estimate_duration_japanese_n_is_shorter():
    n_dur = estimate_token_duration_ms("n", language="japanese")
    ka_dur = estimate_token_duration_ms("ka", language="japanese")
    assert n_dur < ka_dur


# ---------------------------------------------------------------------------
# build_filename_anchor_timeline
# ---------------------------------------------------------------------------

def test_build_returns_none_for_slot_count_one():
    out = build_filename_anchor_timeline(
        "_a.wav", slot_count=1, active_start_ms=0.0, active_end_ms=1000.0
    )
    assert out is None


def test_build_returns_none_when_token_count_mismatches():
    # 8-token filename but slot_count=4 → mismatch
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=4, active_start_ms=0.0, active_end_ms=4000.0,
    )
    assert out is None


def test_build_returns_none_when_active_span_invalid():
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=8, active_start_ms=2000.0, active_end_ms=2000.0,
    )
    assert out is None


def test_build_produces_correct_slot_count_for_matching_filename():
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=8, active_start_ms=1000.0, active_end_ms=5000.0,
    )
    assert out is not None
    assert len(out) == 8


def test_build_timeline_starts_at_active_start():
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=8, active_start_ms=1000.0, active_end_ms=5000.0,
    )
    assert out[0] == pytest.approx(1000.0)


def test_build_timeline_is_monotonic():
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=8, active_start_ms=1000.0, active_end_ms=5000.0,
    )
    for i in range(1, len(out)):
        assert out[i] > out[i - 1], f"timeline not monotonic at index {i}: {out}"


def test_build_timeline_respects_per_token_duration_weighting():
    """8-mora KO wav: first 4 tokens have coda (longer), last 4 are bare CV (shorter).
    The first-4 syllables should span MORE of the active region than the last 4."""
    out = build_filename_anchor_timeline(
        "_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav",
        slot_count=8, active_start_ms=0.0, active_end_ms=4000.0,
    )
    first_4_span = out[4] - out[0]  # geuNG..geu
    # Last 4 span: from start of geu (out[4]) to start of ga (out[7]) → 3 syllables
    # Compare per-syllable: first_4_span / 4 vs (active_end - out[4]) / 4
    first_4_per_syl = first_4_span / 4.0
    last_4_per_syl = (4000.0 - out[4]) / 4.0
    assert first_4_per_syl > last_4_per_syl, (
        f"coda-rich first 4 syllables should span more than coda-less last 4: "
        f"{first_4_per_syl:.1f} vs {last_4_per_syl:.1f}"
    )


def test_build_timeline_handles_japanese_wav():
    out = build_filename_anchor_timeline(
        "ba-bi-bu-be-bo-ba-N-ba.wav",
        slot_count=8, active_start_ms=0.0, active_end_ms=4000.0,
        language="japanese",
    )
    assert out is not None
    assert len(out) == 8
    # Position 6 (lowercase n in tokens) should have a shorter slot than its neighbors
    # because Japanese N has shorter duration than CV moras.
    slot_6_width = out[7] - out[6]
    slot_5_width = out[6] - out[5]
    assert slot_6_width < slot_5_width, (
        f"N-slot width {slot_6_width:.1f} should be smaller than CV-slot width {slot_5_width:.1f}"
    )


def test_build_timeline_handles_pure_vowel_wav():
    out = build_filename_anchor_timeline(
        "_a'a'i'a'u'a'e'a.wav",
        slot_count=8, active_start_ms=0.0, active_end_ms=4000.0,
    )
    assert out is not None
    assert len(out) == 8
    # All tokens are bare vowels → roughly equal spacing
    spacings = [out[i] - out[i - 1] for i in range(1, 8)]
    avg = sum(spacings) / len(spacings)
    for s in spacings:
        assert abs(s - avg) < 5.0, f"vowel-only wav should have equal spacing, got {spacings}"
