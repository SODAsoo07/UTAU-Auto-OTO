from pathlib import Path

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
        assert profile is not None, (language, fmt)
        if profile.get("kind") == "gbm":  # folder-level profile: trees, then a per-oto.ini recalibration
            assert profile["trees"] and len(profile["edges"]) == len(profile["features"])
            assert profile["folder_level"]["features"] == ["risk", "tempo"]
            assert len(profile["folder_level"]["weights"]) == 3 and 0.0 < float(profile["threshold"]) < 1.0
            continue
        assert 0.0 < float(profile["threshold"]) < 1.0
        assert len(profile["weights"]) == len(profile["features"]) + 1
    assert load_profile("korean", "unknown") is None
    # CV is never split, so it has no profile
    assert load_profile("japanese", "cv") is None and load_profile("korean", "cv") is None


def test_japanese_profiles_split_each_oto_by_its_estimated_error_level():
    from universal_auto_oto.row_risk import (calibrated_risks, folder_clean, folder_level, risk_threshold, row_risks,
                                             tempo_irregularity)

    rows = [{"alias": alias, "role": role, "confidence": conf, "reasons": reasons, "pre_abs": pre,
             "offset_abs": pre - 120.0, "fixed_abs": pre + 60.0, "cutoff_abs": pre + 300.0}
            for alias, role, conf, reasons, pre in (
                ("- か", "cv_head", 0.9, (), 500.0), ("a k", "vc", 0.8, (), 900.0),
                ("か", "cv", 0.3, ("hsmm_low_local_margin", "hsmm_wrong_occurrence_risk"), 1300.0))]
    for fmt in ("cvvc", "vcv"):
        profile = load_profile("japanese", fmt)
        assert folder_level(profile) is not None and risk_threshold(profile, fmt) == float("-inf")
        risks = row_risks(rows, Path("unused.wav"), onsets=np.array([510.0, 1310.0]), profile=profile)
        assert len(risks) == 3 and all(0.0 < r < 1.0 for r in risks)
    assert folder_level(load_profile("korean", "vcv")) is None

    # Evenly spaced syllables: no irregularity; one swallowed syllable doubles a gap.
    take = lambda wav, pres: [{"role": "cv", "wav": wav, "pre_abs": p} for p in pres]  # noqa: E731
    assert tempo_irregularity(take("a.wav", [0, 500, 1000, 1500, 2000])) == 0.0
    assert tempo_irregularity(take("a.wav", [0, 500, 1000, 2000, 2500, 3000])) == 0.0  # median spread
    assert tempo_irregularity(take("a.wav", [0, 400, 1100, 1400, 2200])) > 0.3

    # The same row risks are shifted up in an irregular folder, so fewer rows stay clean.
    profile = load_profile("japanese", "vcv")
    risks = list(np.linspace(0.01, 0.3, 40))
    steady = take("a.wav", np.arange(40) * 500.0)
    uneven = take("a.wav", np.arange(40) * 500.0 + np.random.default_rng(0).uniform(-200.0, 200.0, 40))
    assert np.mean(calibrated_risks(risks, uneven, profile)) > np.mean(calibrated_risks(risks, steady, profile))
    assert sum(folder_clean(risks, uneven, profile)) < sum(folder_clean(risks, steady, profile))
