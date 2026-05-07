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


def test_crnn_generator_applies_fallback_blend_when_enabled(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a.wav=a,10,80,-160,60,20\n", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    out_path = tmp_path / "out.ini"

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_predict(**_kwargs):
        return SimpleNamespace(
            params={
                "offset": 30.0,
                "consonant": 120.0,
                "cutoff": -220.0,
                "preutterance": 90.0,
                "overlap": 40.0,
            },
            duration_ms=1000.0,
            confidence=0.20,
            heatmap_confidence={
                "offset": 0.15,
                "overlap": 0.16,
                "preutterance": 0.14,
                "consonant": 0.18,
                "cutoff": 0.17,
            },
        )

    monkeypatch.setattr(gen, "load_oto_checkpoint", lambda *_a, **_k: (FakeModel(), SimpleNamespace(), {}))
    monkeypatch.setattr(gen, "predict_oto_with_model", fake_predict)
    monkeypatch.setenv("UTOA_OTO_CRNN_FALLBACK_ENABLE", "1")
    monkeypatch.setenv("UTOA_OTO_CRNN_FALLBACK_SCORE_THRESHOLD", "0.2")
    monkeypatch.setenv("UTOA_OTO_CRNN_FALLBACK_SOURCE_BLEND", "0.5")

    processed, total, errors = gen.generate_oto_with_crnn_predictor(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        source_oto_path=str(source_oto),
        language="japanese",
        format_type="vcv",
        model_path=str(model_path),
        device="cpu",
    )

    assert errors == []
    assert processed == 1 and total == 1
    line = out_path.read_text(encoding="utf-8").strip()
    assert line.startswith("a.wav=a,")
    # fallback 50% blend then right-boundary guard:
    # offset (30->20), consonant (120->100), pre (90->75), overlap (40->30),
    # cutoff (-220->-190) then capped to -186 by v-role guard.
    assert ",20.000,100.000,-186.000,75.000,30.000" in line
