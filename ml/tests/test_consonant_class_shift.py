"""Unit tests for per-consonant-class runtime offset shift.

Covers the 2026-05-17 worst-case patch for ``b/d/g`` early-bias and
``n/m/l/r`` late-bias the user reported on top of F3+compound output.
"""
from __future__ import annotations

import pytest

from core.coarse_crnn.consonant_class_shift import (
    _CLASS_ENV_VAR,
    classify_onset,
    consonant_class_shift_ms,
    extract_alias_onset,
    shift_oto_params,
)


# ---------------------------------------------------------------------------
# extract_alias_onset
# ---------------------------------------------------------------------------

def test_extract_single_token_cv():
    assert extract_alias_onset("ba") == "b"
    assert extract_alias_onset("geu") == "g"
    assert extract_alias_onset("kkyo") == "kky"
    assert extract_alias_onset("ssw") == "ssw"


def test_extract_two_token_vc_uses_right_onset():
    assert extract_alias_onset("eu b") == "b"
    assert extract_alias_onset("a m") == "m"
    assert extract_alias_onset("eo g") == "g"
    assert extract_alias_onset("u kk") == "kk"


def test_extract_head_dash_uses_consonant():
    assert extract_alias_onset("- b") == "b"
    assert extract_alias_onset("- ka") == "k"
    assert extract_alias_onset("- ny") == "ny"


def test_extract_bare_vowel_returns_empty():
    assert extract_alias_onset("a") == ""
    assert extract_alias_onset("eu") == ""


def test_extract_glide_only_treated_as_onset():
    # "wa" is glide w + vowel a; onset = 'w' → classified as 'glide'
    assert extract_alias_onset("wa") == "w"
    assert extract_alias_onset("ya") == "y"


def test_extract_silence_breath_returns_empty():
    for a in ("", "br", "-", "r", "pau", "sil", "rest"):
        assert extract_alias_onset(a) == "", f"alias={a!r} should yield empty onset"


def test_extract_strips_ja_color_suffix():
    assert extract_alias_onset("a b_W") == "b"
    assert extract_alias_onset("hi_W") == "h"
    assert extract_alias_onset("a m1_W") == "m"


# ---------------------------------------------------------------------------
# classify_onset
# ---------------------------------------------------------------------------

def test_classify_voiced_stops():
    for o in ("b", "d", "g", "bb", "dd", "gg", "by", "dy", "gy", "bw", "dw", "gw"):
        assert classify_onset(o) == "voiced_stop", f"{o} should be voiced_stop"


def test_classify_voiceless_stops():
    for o in ("p", "t", "k", "pp", "tt", "kk", "ph", "th", "kh", "py", "ty", "ky", "pw", "tw", "kw"):
        assert classify_onset(o) == "voiceless_stop", f"{o} should be voiceless_stop"


def test_classify_sonorants():
    # ``r`` is split out as ``liquid_r``; ``ry``/``rw`` glide variants stay
    # in ``sonorant`` because they pattern with ny/my/ly in the diagnostic.
    for o in ("n", "m", "l", "ny", "my", "ly", "ry", "nw", "mw", "lw", "rw"):
        assert classify_onset(o) == "sonorant", f"{o} should be sonorant"


def test_classify_bare_r_is_liquid_r_not_sonorant():
    """Bare ``r`` is its own class so the sonorant shift doesn't over-correct."""
    assert classify_onset("r") == "liquid_r"
    # ``ry`` / ``rw`` glide variants are sonorant (pattern with n/m/l glides).
    assert classify_onset("ry") == "sonorant"
    assert classify_onset("rw") == "sonorant"


def test_classify_fricatives_and_affricates():
    for o in ("s", "ss", "h", "sy", "ssy", "hy"):
        assert classify_onset(o) == "fricative", f"{o} should be fricative"
    for o in ("j", "jj", "ch", "jy", "jjy"):
        assert classify_onset(o) == "affricate", f"{o} should be affricate"


def test_classify_nasal_coda_separate():
    assert classify_onset("ng") == "nasal_coda"


def test_classify_glide_only():
    assert classify_onset("y") == "glide"
    assert classify_onset("w") == "glide"


def test_classify_unknown_returns_empty():
    assert classify_onset("") == ""
    assert classify_onset("xyz") == ""


# ---------------------------------------------------------------------------
# consonant_class_shift_ms (env-driven)
# ---------------------------------------------------------------------------

def test_shift_ms_default_zero(monkeypatch):
    for env_var in _CLASS_ENV_VAR.values():
        monkeypatch.delenv(env_var, raising=False)
    assert consonant_class_shift_ms("b a") == 0.0
    assert consonant_class_shift_ms("a m") == 0.0


def test_shift_ms_voiced_stop_picks_up_env(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    assert consonant_class_shift_ms("b a") == 200.0
    assert consonant_class_shift_ms("eu b") == 200.0
    assert consonant_class_shift_ms("- ga") == 200.0


def test_shift_ms_sonorant_negative_env(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_SONORANT_MS", "-150")
    assert consonant_class_shift_ms("a m") == -150.0
    assert consonant_class_shift_ms("eu n") == -150.0
    assert consonant_class_shift_ms("- ly") == -150.0


def test_shift_ms_invalid_env_returns_zero(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "not_a_number")
    assert consonant_class_shift_ms("b a") == 0.0


def test_shift_ms_unknown_alias_returns_zero(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    assert consonant_class_shift_ms("a") == 0.0  # bare vowel
    assert consonant_class_shift_ms("br") == 0.0
    assert consonant_class_shift_ms("") == 0.0


def test_shift_ms_liquid_r_uses_dedicated_env(monkeypatch):
    """Bare ``r`` should NOT pick up the sonorant shift."""
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_SONORANT_MS", "-150")
    monkeypatch.delenv("UTOA_BOUNDARY_SHIFT_LIQUID_R_MS", raising=False)
    # ``- r`` should resolve to liquid_r class → no shift when env unset
    assert consonant_class_shift_ms("- r") == 0.0
    assert consonant_class_shift_ms("a r") == 0.0
    # ``ry`` and ``rw`` glide variants still get the sonorant shift
    assert consonant_class_shift_ms("a ry") == -150.0
    assert consonant_class_shift_ms("- rw") == -150.0


def test_shift_ms_liquid_r_env_applies(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_LIQUID_R_MS", "50")
    monkeypatch.delenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", raising=False)
    assert consonant_class_shift_ms("- r") == 50.0
    assert consonant_class_shift_ms("a r") == 50.0


def test_glide_multiplier_default_off_keeps_full_shift(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.delenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", raising=False)
    assert consonant_class_shift_ms("a by") == 200.0
    assert consonant_class_shift_ms("a bw") == 200.0


def test_glide_multiplier_halves_voiced_stop_glide(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", "0.5")
    # Glide-bearing variants halved
    assert consonant_class_shift_ms("a by") == 100.0
    assert consonant_class_shift_ms("a bw") == 100.0
    assert consonant_class_shift_ms("a dy") == 100.0
    assert consonant_class_shift_ms("a gw") == 100.0
    # Bare voiced stops unchanged
    assert consonant_class_shift_ms("a b") == 200.0
    assert consonant_class_shift_ms("a g") == 200.0


def test_glide_multiplier_also_applies_to_sonorant_glides(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_SONORANT_MS", "-150")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", "0.5")
    assert consonant_class_shift_ms("a ny") == -75.0
    assert consonant_class_shift_ms("a mw") == -75.0
    assert consonant_class_shift_ms("a ly") == -75.0
    # Bare sonorants unchanged
    assert consonant_class_shift_ms("a n") == -150.0


def test_glide_multiplier_invalid_value_ignored(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", "not_a_number")
    # Falls back to full shift
    assert consonant_class_shift_ms("a by") == 200.0


# ---------------------------------------------------------------------------
# role_shift_ms + combined_shift_ms
# ---------------------------------------------------------------------------

def test_role_shift_default_zero(monkeypatch):
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.delenv("UTOA_BOUNDARY_SHIFT_VC_MS", raising=False)
    assert role_shift_ms("vc") == 0.0
    assert role_shift_ms("v-cv") == 0.0
    assert role_shift_ms("cv") == 0.0
    assert role_shift_ms("") == 0.0


def test_role_shift_vc_env_applies(monkeypatch):
    """``vc`` role (CVVC ``a bw``/``eu g``/``u k``) gets the env shift.
    The 3-token ``v-cv`` role (VCV ``a ki a``) is intentionally NOT wired —
    CVVC vs VCV voicebanks have different bias profiles."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    assert role_shift_ms("vc") == 400.0
    # Other roles still 0 — including 'v-cv' (VCV format, not CVVC VC)
    assert role_shift_ms("v-cv") == 0.0
    assert role_shift_ms("cv") == 0.0


def test_role_shift_invalid_env_returns_zero(monkeypatch):
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "not_a_number")
    assert role_shift_ms("vc") == 0.0


def test_combined_shift_sums_class_and_role(monkeypatch):
    from core.coarse_crnn.consonant_class_shift import combined_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    # VC with bare voiced stop: class +200, role +400 → 600
    assert combined_shift_ms(alias="eu g", role="vc") == 600.0
    # Non-VC role with bare voiced stop: only class shift
    assert combined_shift_ms(alias="eu g", role="cv") == 200.0
    # 'v-cv' (3-token VCV) role: also no role shift wired
    assert combined_shift_ms(alias="eu g", role="v-cv") == 200.0
    # VC with sonorant: -150 + 400 = 250
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_SONORANT_MS", "-150")
    assert combined_shift_ms(alias="eu n", role="vc") == 250.0


def test_combined_shift_glide_multiplier_applies_to_role_shift(monkeypatch):
    """VC glide variants (by/dy/gy/ny/my/ly) need the glide multiplier on
    role_shift too — otherwise +400 VC push over-corrects them by 300~900ms
    (2026-05-17 VC diagnostic)."""
    from core.coarse_crnn.consonant_class_shift import combined_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", "0.5")
    # Bare 'g' VC: class 200 + role 400 = 600
    assert combined_shift_ms(alias="eu g", role="vc") == 600.0
    # Glide 'gy' VC: class 200*0.5=100 + role 400*0.5=200 = 300
    assert combined_shift_ms(alias="eu gy", role="vc") == 300.0
    # Glide 'by' VC: same math
    assert combined_shift_ms(alias="eu by", role="vc") == 300.0


def test_role_shift_skipped_for_ko_coda_vc(monkeypatch):
    """KO coda VC aliases (``u ng``, ``eo N``, ``a L``) must NOT get the VC
    role shift — they're V→coda inside the syllable, not V→next-CV. The
    2026-05-17 TABI diagnostic showed +400 push over-corrects coda VC by
    800–1200ms (``u ng``=+1217ms, ``eo N``=+1208ms)."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms, combined_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_SONORANT_MS", "-150")
    # Coda VC: role shift suppressed
    assert role_shift_ms("vc", alias="u ng") == 0.0
    assert role_shift_ms("vc", alias="eo N") == 0.0
    assert role_shift_ms("vc", alias="a L") == 0.0
    assert role_shift_ms("vc", alias="e NG") == 0.0
    assert role_shift_ms("vc", alias="i H") == 0.0
    # Onset VC: role shift still applies
    assert role_shift_ms("vc", alias="a bw") == 400.0
    assert role_shift_ms("vc", alias="eu g") == 400.0
    assert role_shift_ms("vc", alias="u k") == 400.0
    # combined_shift still applies class shift (sonorant -150) to coda 'n'
    # but skips the +400 role shift
    assert combined_shift_ms(alias="eo N", role="vc") == -150.0
    # Onset VC gets both
    assert combined_shift_ms(alias="eu n", role="vc") == 250.0


def test_role_shift_coda_vc_legacy_env_applies_to_all_classes(monkeypatch):
    """Backward-compat: ``UTOA_BOUNDARY_SHIFT_VC_CODA_MS`` still applies to
    every coda class when no class-specific env is set (matches v16 default)."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms, combined_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_CODA_MS", "-40")
    for env_key in (
        "UTOA_BOUNDARY_SHIFT_VC_CODA_SONORANT_MS",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_PLOSIVE_MS",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_ASPIRATE_MS",
    ):
        monkeypatch.delenv(env_key, raising=False)
    # All coda classes pick up the legacy −40 value
    assert role_shift_ms("vc", alias="u ng") == -40.0  # sonorant
    assert role_shift_ms("vc", alias="eo N") == -40.0  # sonorant
    assert role_shift_ms("vc", alias="i K") == -40.0   # plosive
    assert role_shift_ms("vc", alias="a H") == -40.0   # aspirate
    # Onset VC unaffected
    assert role_shift_ms("vc", alias="a bw") == 400.0


def test_role_shift_coda_vc_class_specific_overrides_legacy(monkeypatch):
    """Class-specific env takes precedence over the legacy single knob.
    The 2026-05-17 v17 split: sonorant=0, plosive=-90, aspirate=0."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "0")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_CODA_MS", "-40")  # legacy fallback
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_CODA_SONORANT_MS", "0")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_CODA_PLOSIVE_MS", "-90")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_CODA_ASPIRATE_MS", "0")
    # Each class uses its own value, legacy ignored
    assert role_shift_ms("vc", alias="u ng") == 0.0   # sonorant
    assert role_shift_ms("vc", alias="eo N") == 0.0   # sonorant
    assert role_shift_ms("vc", alias="i K") == -90.0  # plosive
    assert role_shift_ms("vc", alias="a T") == -90.0  # plosive
    assert role_shift_ms("vc", alias="a H") == 0.0    # aspirate


def test_role_shift_coda_vc_unset_falls_back_to_zero(monkeypatch):
    """When neither class-specific nor legacy env is set, coda VC stays at 0
    — the onset VC value should NOT leak through to coda."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    for env_key in (
        "UTOA_BOUNDARY_SHIFT_VC_CODA_MS",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_SONORANT_MS",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_PLOSIVE_MS",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_ASPIRATE_MS",
    ):
        monkeypatch.delenv(env_key, raising=False)
    assert role_shift_ms("vc", alias="u ng") == 0.0
    assert role_shift_ms("vc", alias="eo N") == 0.0
    assert role_shift_ms("vc", alias="i K") == 0.0
    # Onset VC unaffected
    assert role_shift_ms("vc", alias="a bw") == 400.0


def test_role_shift_no_alias_falls_back_to_role_only(monkeypatch):
    """Backward-compat: role_shift_ms without alias still works (returns
    raw env-driven value without coda-VC suppression)."""
    from core.coarse_crnn.consonant_class_shift import role_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VC_MS", "400")
    assert role_shift_ms("vc") == 400.0
    assert role_shift_ms("vc", alias="") == 400.0


def test_combined_shift_glide_multiplier_not_applied_when_no_role(monkeypatch):
    """When role shift is zero, glide multiplier on role_shift is a no-op."""
    from core.coarse_crnn.consonant_class_shift import combined_shift_ms
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS", "200")
    monkeypatch.setenv("UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER", "0.5")
    monkeypatch.delenv("UTOA_BOUNDARY_SHIFT_VC_MS", raising=False)
    # No role shift → just halved class shift (200 * 0.5)
    assert combined_shift_ms(alias="eu gy", role="vc") == 100.0


def test_combined_shift_returns_zero_when_no_env(monkeypatch):
    from core.coarse_crnn.consonant_class_shift import combined_shift_ms, _CLASS_ENV_VAR, _ROLE_SHIFT_ENV_VAR
    for v in _CLASS_ENV_VAR.values():
        monkeypatch.delenv(v, raising=False)
    for v in _ROLE_SHIFT_ENV_VAR.values():
        monkeypatch.delenv(v, raising=False)
    assert combined_shift_ms(alias="eu g", role="vc") == 0.0
    assert combined_shift_ms(alias="a", role="cv") == 0.0


# ---------------------------------------------------------------------------
# shift_oto_params
# ---------------------------------------------------------------------------

def test_shift_zero_returns_unchanged_dict():
    p = {"offset": 100.0, "preutterance": 50.0, "overlap": 20.0, "consonant": 80.0, "cutoff": -200.0}
    out = shift_oto_params(p, shift_ms=0.0, duration_ms=2000.0)
    assert out == p
    assert out is not p  # returns copy


def test_shift_positive_moves_offset_later():
    p = {"offset": 1000.0, "preutterance": 50.0, "overlap": 20.0, "consonant": 80.0, "cutoff": -200.0}
    out = shift_oto_params(p, shift_ms=200.0, duration_ms=5000.0)
    assert out["offset"] == 1200.0
    # Other fields unchanged (they're relative to offset)
    assert out["preutterance"] == 50.0
    assert out["consonant"] == 80.0
    assert out["cutoff"] == -200.0


def test_shift_negative_moves_offset_earlier():
    p = {"offset": 1000.0, "preutterance": 50.0, "overlap": 20.0, "consonant": 80.0, "cutoff": -200.0}
    out = shift_oto_params(p, shift_ms=-150.0, duration_ms=5000.0)
    assert out["offset"] == 850.0


def test_shift_clamps_to_zero_when_too_early():
    p = {"offset": 50.0, "preutterance": 30.0, "overlap": 15.0, "consonant": 60.0, "cutoff": -150.0}
    out = shift_oto_params(p, shift_ms=-500.0, duration_ms=2000.0)
    assert out["offset"] == 0.0


def test_shift_clamps_to_duration_when_too_late():
    p = {"offset": 1800.0, "preutterance": 30.0, "overlap": 15.0, "consonant": 60.0, "cutoff": -150.0}
    out = shift_oto_params(p, shift_ms=500.0, duration_ms=2000.0)
    # max offset = duration - 5 = 1995
    assert out["offset"] == 1995.0


def test_shift_sub_millisecond_ignored():
    p = {"offset": 1000.0, "preutterance": 50.0, "overlap": 20.0, "consonant": 80.0, "cutoff": -200.0}
    out = shift_oto_params(p, shift_ms=0.3, duration_ms=2000.0)
    assert out["offset"] == 1000.0  # unchanged
