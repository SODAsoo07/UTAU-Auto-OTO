from __future__ import annotations

import wave
from types import SimpleNamespace


def _write_wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def test_crnn_oto_generator_preserves_aliases_and_replaces_params(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "a.wav=a,0,0,0,0,0\n"
        "a.wav=a k,0,0,0,0,0\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    out_path = tmp_path / "out.ini"

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    calls = []

    def fake_predict(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            params={
                "offset": 10.0,
                "consonant": 90.0,
                "cutoff": -220.0,
                "preutterance": 70.0,
                "overlap": 30.0,
            }
        )

    monkeypatch.setattr(gen, "load_oto_checkpoint", lambda *_a, **_k: (FakeModel(), SimpleNamespace(), {}))
    monkeypatch.setattr(gen, "predict_oto_with_model", fake_predict)

    processed, total, errors = gen.generate_oto_with_crnn_predictor(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        source_oto_path=str(source_oto),
        language="japanese",
        format_type="vcv",
        model_path=str(model_path),
        device="cpu",
        alias_suffix="C4",
    )

    assert errors == []
    assert processed == 2
    assert total == 2
    # `a` resolves to alias_role=v which has a tighter cutoff cap (max 92ms past
    # consonant) than the old alias_type=mono fallback. `a k` stays on the vc
    # role with the same 82ms cutoff cap as before.
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "a.wav=a_C4,10.000,90.000,-182.000,70.000,30.000",
        "a.wav=a k_C4,10.000,90.000,-172.000,70.000,30.000",
    ]
    assert calls[0]["alias"] == "a"
    assert calls[0]["next_alias"] == "a k"
    assert calls[1]["prev_alias"] == "a"
    assert calls[1]["row_index_in_wav"] == 1
    assert calls[1]["file_row_count"] == 2


def test_crnn_right_boundary_guard_shortens_overlong_cutoff():
    from core.coarse_crnn.oto_predictor_generator import _apply_conservative_right_boundary_guard

    params, changed = _apply_conservative_right_boundary_guard(
        {
            "offset": 20.0,
            "consonant": 460.0,
            "cutoff": -1200.0,
            "preutterance": 80.0,
            "overlap": 42.0,
        },
        language="japanese",
        alias="a k",
        duration_ms=1500.0,
    )

    assert changed
    assert params["offset"] == 20.0
    assert params["preutterance"] == 80.0
    assert params["overlap"] == 42.0
    assert params["consonant"] == 128.0
    assert params["cutoff"] == -210.0


def test_role_guard_limits_diphthong_tightens_cv_cap():
    from core.coarse_crnn.oto_predictor_generator import _right_guard_limits_for_role

    base = _right_guard_limits_for_role("cv", alias_type="cv")
    diph = _right_guard_limits_for_role("cv", alias_type="cv", is_diphthong=True)

    # max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap
    assert diph[0] < base[0]
    assert diph[2] < base[2]


def test_role_guard_limits_special_v_borrows_cv_floor():
    from core.coarse_crnn.oto_predictor_generator import _right_guard_limits_for_role

    pure_v = _right_guard_limits_for_role("v", alias_type="vowel")
    special_v = _right_guard_limits_for_role("v", alias_type="vowel", is_special=True)

    # Special-V should not be looser than CV on the consonant cap.
    cv = _right_guard_limits_for_role("cv", alias_type="cv")
    assert special_v[0] >= cv[0]
    assert special_v[1] >= cv[1]
    assert special_v[0] > pure_v[0]


def test_role_guard_limits_japanese_n_are_tighter_than_regular_vc():
    from core.coarse_crnn.oto_predictor_generator import _right_guard_limits_for_role

    regular = _right_guard_limits_for_role(
        "vc",
        alias_type="vc",
        language="japanese",
        alias="a k",
    )
    nasal_n = _right_guard_limits_for_role(
        "vc",
        alias_type="vc",
        language="japanese",
        alias="a N",
    )

    assert nasal_n[0] <= regular[0]
    assert nasal_n[2] <= regular[2]


def test_normalize_special_aliases_handles_iterables():
    from core.coarse_crnn.oto_predictor_generator import _alias_is_special, _normalize_special_aliases

    special = _normalize_special_aliases({"tt", "Pf"})

    assert "tt" in special
    assert "pf" in special  # case-insensitive
    assert _alias_is_special("a tt", special)
    assert _alias_is_special("Tt", special)
    assert not _alias_is_special("a k", special)
    assert not _alias_is_special("", special)


def test_korean_cvvc_role_fallback_blends_vc_toward_base(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    monkeypatch.setenv("UTOA_OTO_CRNN_KR_CVVC_ROLE_BASE_WEIGHT_VC", "0.80")
    monkeypatch.setattr(
        gen,
        "_analyze_activity_profile",
        lambda *_args, **_kwargs: {
            "active_start_ms": 80.0,
            "active_end_ms": 360.0,
            "active_core_start_ms": 120.0,
            "active_core_end_ms": 300.0,
            "duration_ms": 1000.0,
        },
    )
    predicted = {
        "offset": 100.0,
        "consonant": 830.0,
        "cutoff": -880.0,
        "preutterance": 820.0,
        "overlap": 790.0,
    }
    base = {
        "offset": 720.0,
        "consonant": 130.0,
        "cutoff": -260.0,
        "preutterance": 115.0,
        "overlap": 80.0,
    }

    out, reason = gen._apply_korean_cvvc_role_fallback(
        predicted_anchors={
            "offset": 100.0,
            "preutterance": 920.0,
            "consonant": 930.0,
            "cutoff": 980.0,
        },
        predicted_params=predicted,
        base_params=base,
        wav_path="dummy.wav",
        sample_rate=16000,
        cache={},
        voicebank_profile={},
        predicted_confidence=0.9,
        predicted_error_ms=None,
        predicted_low_confidence=False,
        language="korean",
        format_type="cvvc",
        alias="a k",
        duration_ms=1000.0,
        is_special=False,
    )

    assert "korean_cvvc_vc_role" in reason
    assert "syllable_shift_guard" in reason
    assert out["offset"] == predicted["offset"]
    assert out["preutterance"] < predicted["preutterance"]
    assert out["consonant"] < predicted["consonant"]
    assert out["offset"] != base["offset"]


def test_korean_cvvc_role_fallback_rejects_unusable_source_shape(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    monkeypatch.setattr(
        gen,
        "_analyze_activity_profile",
        lambda *_args, **_kwargs: {
            "active_start_ms": 250.0,
            "active_end_ms": 520.0,
            "active_core_start_ms": 300.0,
            "active_core_end_ms": 470.0,
            "duration_ms": 1000.0,
        },
    )
    predicted = {
        "offset": 880.0,
        "consonant": 910.0,
        "cutoff": -90.0,
        "preutterance": 900.0,
        "overlap": 850.0,
    }
    source_default = {
        "offset": 0.0,
        "consonant": 8.0,
        "cutoff": -15.0,
        "preutterance": 4.0,
        "overlap": 1.0,
    }

    out, reason = gen._apply_korean_cvvc_role_fallback(
        predicted_anchors={
            "offset": 880.0,
            "preutterance": 900.0,
            "consonant": 910.0,
            "cutoff": 970.0,
        },
        predicted_params=predicted,
        base_params=source_default,
        wav_path="dummy.wav",
        sample_rate=16000,
        cache={},
        voicebank_profile={},
        predicted_confidence=0.1,
        predicted_error_ms=300.0,
        predicted_low_confidence=True,
        language="korean",
        format_type="cvvc",
        alias="a k",
        duration_ms=1000.0,
        is_special=False,
    )

    assert reason == ""
    assert out == predicted


def test_source_shape_prior_can_keep_model_cutoff(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _blend_with_source_oto_shape

    predicted = {
        "offset": 100.0,
        "consonant": 820.0,
        "cutoff": -900.0,
        "preutterance": 780.0,
        "overlap": 760.0,
    }
    base = {
        "offset": 500.0,
        "consonant": 130.0,
        "cutoff": -260.0,
        "preutterance": 115.0,
        "overlap": 80.0,
    }

    out = _blend_with_source_oto_shape(
        predicted_params=predicted,
        base_params=base,
        shape_weight=0.80,
        cutoff_weight=0.0,
        duration_ms=1000.0,
    )

    assert out["offset"] == predicted["offset"]
    assert out["preutterance"] < predicted["preutterance"]
    assert out["consonant"] < predicted["consonant"]
    assert abs(abs(out["cutoff"]) - abs(predicted["cutoff"])) <= 1.0


def test_korean_cvvc_source_shape_cutoff_factor_is_role_specific(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _source_shape_cutoff_weight

    monkeypatch.setenv("UTOA_OTO_CRNN_KR_CVVC_SOURCE_SHAPE_CUTOFF_FACTOR_V_CV", "0.10")

    weight = _source_shape_cutoff_weight(
        shape_weight=0.80,
        language="korean",
        format_type="cvvc",
        alias="a k",
        is_special=False,
        role="v-cv",
    )

    assert abs(weight - 0.08) < 1e-9


def test_korean_cvvc_role_fallback_ignores_japanese():
    from core.coarse_crnn.oto_predictor_generator import _apply_korean_cvvc_role_fallback

    predicted = {
        "offset": 900.0,
        "consonant": 930.0,
        "cutoff": -120.0,
        "preutterance": 920.0,
        "overlap": 880.0,
    }

    out, reason = _apply_korean_cvvc_role_fallback(
        predicted_anchors={
            "offset": 900.0,
            "preutterance": 920.0,
            "consonant": 930.0,
            "cutoff": 980.0,
        },
        predicted_params=predicted,
        base_params={
            "offset": 100.0,
            "consonant": 130.0,
            "cutoff": -260.0,
            "preutterance": 115.0,
            "overlap": 80.0,
        },
        wav_path="dummy.wav",
        sample_rate=16000,
        cache={},
        voicebank_profile={},
        predicted_confidence=0.0,
        predicted_error_ms=500.0,
        predicted_low_confidence=True,
        language="japanese",
        format_type="cvvc",
        alias="a k",
        duration_ms=1000.0,
        is_special=False,
    )

    assert reason == ""
    assert out == predicted


def test_is_multi_signal_required_default_includes_korean_cvc_and_japanese_cvvc(monkeypatch):
    """Plan B (0-order) default slice list activates AND mode for the two
    slices where the baseline analysis showed user-facing headroom."""
    from core.coarse_crnn.oto_predictor_generator import _is_multi_signal_required

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", raising=False)
    assert _is_multi_signal_required("korean", "cvc") is True
    assert _is_multi_signal_required("japanese", "cvvc") is True
    # OR mode for slices not in the default list.
    assert _is_multi_signal_required("korean", "cvvc") is False
    assert _is_multi_signal_required("japanese", "vcv") is False


def test_is_multi_signal_required_respects_env_off(monkeypatch):
    """Setting the env var to 'off' (or 'none', '0', 'false') reverts to
    the legacy OR-trigger behaviour for every slice."""
    from core.coarse_crnn.oto_predictor_generator import _is_multi_signal_required

    for value in ("off", "none", "0", "false"):
        monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", value)
        assert _is_multi_signal_required("korean", "cvc") is False
        assert _is_multi_signal_required("japanese", "cvvc") is False


def test_is_multi_signal_required_supports_wildcards(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _is_multi_signal_required

    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", "korean/*")
    assert _is_multi_signal_required("korean", "cvc") is True
    assert _is_multi_signal_required("korean", "cvvc") is True
    assert _is_multi_signal_required("japanese", "cvvc") is False

    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", "*/cvvc")
    assert _is_multi_signal_required("korean", "cvvc") is True
    assert _is_multi_signal_required("japanese", "cvvc") is True
    assert _is_multi_signal_required("korean", "vcv") is False


def test_low_confidence_fallback_and_mode_suppresses_conf_only(monkeypatch):
    """In multi_v2 AND-mode slices, conf-only must NOT fire when activity is aligned.

    Regression guard for Plan B multi_v2: confidence < min_conf alone is not
    enough; the audio activity profile must also show misalignment. When the
    preutterance anchor falls within the active window (no shift risk, high
    continuity), trigger_activity=False and fallback is suppressed.
    """
    import os
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", raising=False)

    base_params = {
        "offset": 100.0,
        "consonant": 200.0,
        "cutoff": -300.0,
        "preutterance": 60.0,
        "overlap": 30.0,
    }
    predicted = {
        "offset": 110.0,
        "consonant": 210.0,
        "cutoff": -310.0,
        "preutterance": 65.0,
        "overlap": 32.0,
    }

    # Pre-populate the activity cache with an aligned profile.
    # pre_abs = offset(110) + preutterance(65) = 175 ms, which is well
    # inside [active_start=50, active_end=900] with margin=60.
    wav_key = os.path.abspath("dummy.wav").lower()
    aligned_profile = {
        "active_start_ms": 50.0,
        "active_end_ms": 900.0,
        "active_core_start_ms": 80.0,
        "active_core_end_ms": 870.0,
        "onset_ms": 60.0,
        "continuity_score": 0.9,
        "duration_ms": 1000.0,
    }
    cache = {wav_key: aligned_profile}

    # Confidence is low, predicted_error is None → conf-only trigger, but
    # activity is aligned → trigger_activity=False → multi_v2 suppresses.
    out, reason = _apply_low_confidence_fallback(
        predicted_params=predicted,
        predicted_confidence=0.0,
        predicted_error_ms=None,
        predicted_low_confidence=False,
        base_params=base_params,
        wav_path="dummy.wav",
        sample_rate=16000,
        cache=cache,
        voicebank_profile={},
        language="korean",
        format_type="cvc",
        alias="a",
        duration_ms=1000.0,
        is_special=False,
    )
    # multi_v2 AND mode: conf-only + aligned activity must not fire.
    assert reason == ""
    assert out == predicted


def test_low_confidence_fallback_or_mode_unchanged_for_korean_cvvc(monkeypatch):
    """OR mode (legacy) should still fire on confidence-only as before.

    korean/cvvc is intentionally NOT in the AND mode default list, so
    the legacy behaviour is preserved.
    """
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", raising=False)

    base_params = {
        "offset": 100.0,
        "consonant": 200.0,
        "cutoff": -300.0,
        "preutterance": 60.0,
        "overlap": 30.0,
    }
    predicted = {
        "offset": 110.0,
        "consonant": 210.0,
        "cutoff": -310.0,
        "preutterance": 65.0,
        "overlap": 32.0,
    }

    out, reason = _apply_low_confidence_fallback(
        predicted_params=predicted,
        predicted_confidence=0.0,
        predicted_error_ms=None,
        predicted_low_confidence=False,
        base_params=base_params,
        wav_path="dummy.wav",
        sample_rate=16000,
        cache={},
        voicebank_profile={},
        language="korean",
        format_type="cvvc",
        alias="a k",
        duration_ms=1000.0,
        is_special=False,
    )
    # OR mode: conf-only fires; reason includes "low_confidence" but NOT multi_v2.
    assert reason != ""
    assert "low_confidence" in reason
    assert "slice_and_mode" not in reason
    assert "multi_v2" not in reason


def test_low_confidence_fallback_multi_v2_fires_on_activity_risk(monkeypatch):
    """multi_v2: conf-only fires when the activity profile shows misalignment.

    When preutterance falls far outside the active audio window the
    activity corroboration signal is True, so (conf OR err) AND activity
    evaluates to True and fallback fires. Reason must include 'multi_v2'.
    """
    import os
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", raising=False)

    base_params = {
        "offset": 10.0,
        "consonant": 80.0,
        "cutoff": -200.0,
        "preutterance": 40.0,
        "overlap": 20.0,
    }
    predicted = {
        "offset": 10.0,
        "consonant": 80.0,
        "cutoff": -200.0,
        "preutterance": 5.0,
        "overlap": 20.0,
    }

    # pre_abs = offset(10) + preutterance(5) = 15 ms, but activity starts at
    # 500 ms. 15 < 500 - 60 (margin) → shift risk → trigger_activity=True.
    wav_key = os.path.abspath("shifted.wav").lower()
    shifted_profile = {
        "active_start_ms": 500.0,
        "active_end_ms": 900.0,
        "active_core_start_ms": 540.0,
        "active_core_end_ms": 870.0,
        "onset_ms": 510.0,
        "continuity_score": 0.85,
        "duration_ms": 1000.0,
    }
    cache = {wav_key: shifted_profile}

    out, reason = _apply_low_confidence_fallback(
        predicted_params=predicted,
        predicted_confidence=0.0,
        predicted_error_ms=None,
        predicted_low_confidence=False,
        base_params=base_params,
        wav_path="shifted.wav",
        sample_rate=16000,
        cache=cache,
        voicebank_profile={},
        language="korean",
        format_type="cvc",
        alias="a",
        duration_ms=1000.0,
        is_special=False,
    )
    assert reason != "", "multi_v2 must fire when activity shows shift risk"
    assert "multi_v2" in reason


def test_low_confidence_fallback_multi_v2_reason_replaces_slice_and_mode(monkeypatch):
    """Reason tag must be 'multi_v2', not the legacy 'slice_and_mode'."""
    import os
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", raising=False)

    base_params = {
        "offset": 10.0,
        "consonant": 80.0,
        "cutoff": -200.0,
        "preutterance": 40.0,
        "overlap": 20.0,
    }
    predicted = {
        "offset": 10.0,
        "consonant": 80.0,
        "cutoff": -200.0,
        "preutterance": 5.0,
        "overlap": 20.0,
    }

    wav_key = os.path.abspath("shifted2.wav").lower()
    cache = {wav_key: {
        "active_start_ms": 500.0,
        "active_end_ms": 900.0,
        "continuity_score": 0.85,
        "duration_ms": 1000.0,
    }}

    _, reason = _apply_low_confidence_fallback(
        predicted_params=predicted,
        predicted_confidence=0.0,
        predicted_error_ms=None,
        predicted_low_confidence=False,
        base_params=base_params,
        wav_path="shifted2.wav",
        sample_rate=16000,
        cache=cache,
        voicebank_profile={},
        language="korean",
        format_type="cvc",
        alias="a",
        duration_ms=1000.0,
        is_special=False,
    )
    assert "multi_v2" in reason
    assert "slice_and_mode" not in reason
