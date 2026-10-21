import numpy as np

from universal_auto_oto.row_risk import load_profile, row_features, voiceless_right_onset


def test_voiceless_onset_rows_only():
    assert voiceless_right_onset("a ka") and voiceless_right_onset("- cha") and voiceless_right_onset("kkwa")
    assert not voiceless_right_onset("a ga") and not voiceless_right_onset("a k") and not voiceless_right_onset("a -")


def test_features_follow_the_fitted_definition():
    rows = [{"alias": "a ka", "role": "vcv", "confidence": 0.5, "reasons": ("hsmm_low_local_margin",), "pre_abs": 1000.0},
            {"alias": "a ta", "role": "vcv", "confidence": 0.6, "reasons": (), "pre_abs": 2000.0},
            {"alias": "a ga", "role": "vcv", "confidence": 0.7, "reasons": (), "pre_abs": 3000.0}]
    features = row_features(rows, np.array([1010.0, 2400.0]))
    assert features[0]["onset_dist"] == 10.0 and features[0]["onset_far"] == 0.0
    assert features[1]["onset_dist"] == 250.0 and features[1]["onset_far"] == 1.0  # capped, counted far
    assert features[2]["voiceless"] == 0.0 and features[2]["file_far_rate"] == 0.5
    assert set(load_profile()["features"]) == set(features[0])


def test_onset_rule_v2_reads_kana_hangul_and_romaji():
    from universal_auto_oto.row_risk import right_onset_phone, row_features

    assert [right_onset_phone(a) for a in ("か", "a し", "- ちゃ", "あ", "a -")] == ["k", "sh", "ch", None, None]
    assert [right_onset_phone(a) for a in ("카", "a 싸", "가", "아")] == ["k", "ss", "g", None]
    assert [right_onset_phone(a) for a in ("a tsu", "fu", "a ga")] == ["ts", "f", "g"]
    rows = [{"alias": "か", "role": "cv", "confidence": 0.5, "reasons": (), "pre_abs": 1000.0}]
    assert row_features(rows, np.array([1010.0]), onset_rule="v2", roles=("cv",))[0]["voiceless"] == 1.0
    assert row_features(rows, np.array([1010.0]))[0]["voiceless"] == 0.0  # the Korean v1 profile reads romaji only


def test_each_fitted_language_format_has_its_profile():
    from universal_auto_oto.row_risk import PROFILE_RESOURCES

    assert load_profile("ko", "vcv") is load_profile("korean", "cvc") is load_profile()
    for language, fmt in PROFILE_RESOURCES:
        profile = load_profile(language, fmt)
        if profile is None:  # not fitted yet: that format stays all-review
            continue
        assert 0.0 < float(profile["threshold"]) < 1.0
        assert len(profile["weights"]) == len(profile["features"]) + 1
    assert load_profile("korean", "unknown") is None
