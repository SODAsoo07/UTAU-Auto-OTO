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
            },
            anchors=SimpleNamespace(
                offset=10.0,
                overlap=30.0,
                preutterance=70.0,
                consonant=90.0,
                cutoff=-220.0,
            ),
            confidence=1.0,
            low_confidence=False,
            predicted_error_ms=None,
            duration_ms=500.0,
        )

    monkeypatch.setattr(gen, "load_oto_checkpoint", lambda *_a, **_k: (FakeModel(), SimpleNamespace(), {}))
    monkeypatch.setattr(gen, "predict_oto_with_model", fake_predict)
    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_FALLBACK_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", "0")

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


def test_role_guard_limits_diphthong_lengthens_cv_cap():
    from core.coarse_crnn.oto_predictor_generator import _right_guard_limits_for_role

    base = _right_guard_limits_for_role("cv", alias_type="cv")
    diph = _right_guard_limits_for_role("cv", alias_type="cv", is_diphthong=True)

    # max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap
    assert diph[0] > base[0]
    assert diph[2] > base[2]


def test_role_guard_limits_special_v_borrows_cv_floor():
    from core.coarse_crnn.oto_predictor_generator import _right_guard_limits_for_role

    pure_v = _right_guard_limits_for_role("v", alias_type="vowel")
    special_v = _right_guard_limits_for_role("v", alias_type="vowel", is_special=True)

    # Special-V should not be looser than CV on the consonant cap.
    cv = _right_guard_limits_for_role("cv", alias_type="cv")
    assert special_v[0] >= cv[0]
    assert special_v[1] >= cv[1]
    assert special_v[0] > pure_v[0]


def test_normalize_special_aliases_handles_iterables():
    from core.coarse_crnn.oto_predictor_generator import _alias_is_special, _normalize_special_aliases

    special = _normalize_special_aliases({"tt", "Pf"})

    assert "tt" in special
    assert "pf" in special  # case-insensitive
    assert _alias_is_special("a tt", special)
    assert _alias_is_special("Tt", special)
    assert not _alias_is_special("a k", special)
    assert not _alias_is_special("", special)


def test_low_confidence_fallback_ignores_model_flag_by_default(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_USE_MODEL_FLAG", raising=False)
    out, reason = _apply_low_confidence_fallback(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -160.0,
            "preutterance": 55.0,
            "overlap": 20.0,
        },
        predicted_confidence=0.42,
        predicted_error_ms=390.0,
        predicted_low_confidence=True,
        confidence_components={"heatmap": 0.50},
        base_params={
            "offset": 0.0,
            "consonant": 1.0,
            "cutoff": -1.0,
            "preutterance": 0.0,
            "overlap": 0.0,
        },
        language="japanese",
        alias="a",
        duration_ms=1200.0,
        is_special=False,
    )

    assert reason == ""
    assert out["offset"] == 100.0


def test_low_confidence_fallback_requires_low_heatmap_for_error(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_REQUIRE_LOW_HEATMAP_FOR_ERROR", raising=False)
    out, reason = _apply_low_confidence_fallback(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -160.0,
            "preutterance": 55.0,
            "overlap": 20.0,
        },
        predicted_confidence=0.45,
        predicted_error_ms=700.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.55},
        base_params={
            "offset": 0.0,
            "consonant": 1.0,
            "cutoff": -1.0,
            "preutterance": 0.0,
            "overlap": 0.0,
        },
        language="japanese",
        alias="a",
        duration_ms=1200.0,
        is_special=False,
    )

    assert reason == ""
    assert out["offset"] == 100.0


def test_shift_params_into_activity_window_moves_early_anchor():
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_activity_window

    out = _shift_params_into_activity_window(
        predicted_anchors={
            "offset": 10.0,
            "overlap": 25.0,
            "preutterance": 60.0,
            "consonant": 95.0,
            "cutoff": 180.0,
        },
        duration_ms=1600.0,
        active_start_ms=700.0,
        active_end_ms=1200.0,
    )

    assert out
    # The model shape is preserved, but the absolute offset is moved near the
    # detected activity window instead of falling back to the source OTO row.
    assert 660.0 <= out["offset"] <= 680.0
    assert out["preutterance"] == 50.0
    assert out["consonant"] == 85.0


def test_row_activity_shift_is_disabled_by_default(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_row_activity_window

    monkeypatch.delenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", raising=False)

    out = _shift_params_into_row_activity_window(
        predicted_anchors={
            "offset": 520.0,
            "overlap": 540.0,
            "preutterance": 590.0,
            "consonant": 625.0,
            "cutoff": 720.0,
        },
        duration_ms=1600.0,
        active_start_ms=100.0,
        active_end_ms=1300.0,
        row_index=0,
        row_count=4,
    )

    assert out == {}


def test_row_activity_shift_moves_first_row_back_from_next_syllable_when_enabled(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_row_activity_window

    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", "1")

    out = _shift_params_into_row_activity_window(
        predicted_anchors={
            "offset": 520.0,
            "overlap": 540.0,
            "preutterance": 590.0,
            "consonant": 625.0,
            "cutoff": 720.0,
        },
        duration_ms=1600.0,
        active_start_ms=100.0,
        active_end_ms=1300.0,
        row_index=0,
        row_count=4,
    )

    assert out
    assert 110.0 <= out["offset"] <= 130.0
    assert out["preutterance"] == 70.0
    assert out["consonant"] == 105.0


def test_audio_candidate_snap_moves_only_nearby_onset(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=0.0,
            active_end_ms=500.0,
            onset_peaks=[SimpleNamespace(time_ms=150.0, strength=0.80)],
        )
    }

    monkeypatch.delenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ENABLE", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", "0")
    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="japanese",
        alias="ka",
        duration_ms=500.0,
        cache=cache,
    )

    assert reason.startswith("candidate_onset:")
    assert 105.0 <= out["offset"] <= 108.0
    assert out["preutterance"] == 40.0
    assert out["consonant"] == 80.0


def test_audio_candidate_snap_uses_monotonic_sequence_state(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=0.0,
            active_end_ms=500.0,
            onset_peaks=[
                SimpleNamespace(time_ms=150.0, strength=0.90),
                SimpleNamespace(time_ms=190.0, strength=0.80),
            ],
        )
    }
    state = {key: 0}
    monkeypatch.delenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", raising=False)

    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="japanese",
        alias="ka",
        duration_ms=500.0,
        cache=cache,
        sequence_state=state,
    )

    assert reason.startswith("candidate_onset:")
    assert state[key] == 1
    assert 118.0 <= out["offset"] <= 122.0


def test_audio_candidate_snap_ignores_far_onset(tmp_path):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=0.0,
            active_end_ms=500.0,
            onset_peaks=[SimpleNamespace(time_ms=260.0, strength=0.95)],
        )
    }

    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="japanese",
        alias="ka",
        duration_ms=500.0,
        cache=cache,
    )

    assert reason == ""
    assert out["offset"] == 100.0


def test_audio_candidate_snap_blocks_regressive_format_role_pair(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=0.0,
            active_end_ms=500.0,
            onset_peaks=[SimpleNamespace(time_ms=150.0, strength=0.80)],
        )
    }
    monkeypatch.delenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_BLOCK_FORMAT_ROLES", raising=False)

    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="japanese",
        format_type="cvc",
        alias="a ka",
        duration_ms=500.0,
        cache=cache,
    )

    assert reason == ""
    assert out["offset"] == 100.0
