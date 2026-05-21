from __future__ import annotations

from core.coarse_crnn.alias_role import (
    ALIAS_ROLES,
    classify_alias_role,
    is_diphthong,
    normalize_role,
)
from core.coarse_crnn.oto_param_priors import oto_role_prior


def test_special_overrides_all_other_signals():
    role = classify_alias_role("japanese", "ka", alias_type="cv", is_special=True)

    assert role == "special"


def test_head_cv_dash_prefix_maps_to_minus_cv():
    role = classify_alias_role("japanese", "- ka", alias_type="cv")

    assert role == "-cv"


def test_explicit_cv_head_alias_type_maps_to_minus_cv():
    role = classify_alias_role("japanese", "ka", alias_type="cv_head")

    assert role == "-cv"


def test_release_suffix_demotes_cv_to_cv_dash():
    role = classify_alias_role("japanese", "ka -", alias_type="cv")

    assert role == "cv-"


def test_release_suffix_demotes_vowel_to_v_dash():
    role = classify_alias_role("japanese", "a -", alias_type="vowel")

    assert role == "v-"


def test_endbreath_marker_keeps_role_distinct_from_pure_breath():
    end = classify_alias_role("korean", "a R", alias_type="br")
    plain = classify_alias_role("korean", "息", alias_type="br")

    assert end == "endbr"
    assert plain == "br"


def test_vc_vv_vcv_alias_types_map_to_role_unchanged():
    assert classify_alias_role("japanese", "a k", alias_type="vc") == "vc"
    assert classify_alias_role("japanese", "a i", alias_type="vv") == "vv"
    assert classify_alias_role("japanese", "a ka", alias_type="vcv") == "v-cv"


def test_korean_diphthong_jungseong_is_detected():
    assert is_diphthong("korean", "야")
    assert is_diphthong("korean", "와")
    assert is_diphthong("korean", "의")
    assert not is_diphthong("korean", "아")
    assert not is_diphthong("korean", "카")


def test_japanese_yoon_and_vowel_pair_count_as_diphthong():
    assert is_diphthong("japanese", "きゃ")
    assert is_diphthong("japanese", "ai")
    assert not is_diphthong("japanese", "か")
    assert not is_diphthong("japanese", "a")


def test_normalize_role_accepts_legacy_aliases():
    assert normalize_role("vcv") == "v-cv"
    assert normalize_role("cv_head") == "-cv"
    assert normalize_role("vowel") == "v"
    assert normalize_role("EndBR") == "endbr"
    assert normalize_role("nonsense") == "other"


def test_role_prior_diphthong_lengthens_onset_for_cv_family():
    base = oto_role_prior("cv")
    diph = oto_role_prior("cv", is_diphthong=True)

    assert diph["cons_gap"] > base["cons_gap"]
    assert diph["max_cons_gap"] > base["max_cons_gap"]
    assert diph["tail"] > base["tail"]


def test_role_prior_diphthong_does_not_inflate_pure_vowel():
    base = oto_role_prior("v")
    diph = oto_role_prior("v", is_diphthong=True)

    assert diph["cons_gap"] == base["cons_gap"]


def test_role_prior_special_v_borrows_consonant_floor_from_cv():
    pure_v = oto_role_prior("v")
    special_v = oto_role_prior("v", is_special=True)
    cv = oto_role_prior("cv")

    assert special_v["cons_gap"] >= cv["cons_gap"]
    assert special_v["max_cons_gap"] >= cv["max_cons_gap"]
    assert special_v["cons_gap"] > pure_v["cons_gap"]


def test_role_prior_keeps_dict_shape_consistent():
    expected = {"cons_gap", "tail", "min_cons_gap", "min_tail", "max_cons_gap", "max_tail"}
    for role in ALIAS_ROLES:
        prior = oto_role_prior(role)
        assert set(prior.keys()) == expected
        assert prior["min_cons_gap"] <= prior["cons_gap"] <= prior["max_cons_gap"]
        assert prior["min_tail"] <= prior["tail"] <= prior["max_tail"]
