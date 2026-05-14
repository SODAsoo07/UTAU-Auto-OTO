from __future__ import annotations

import wave
from types import SimpleNamespace
import numpy as np


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


def test_crnn_oto_generator_uses_filename_slot_for_shuffled_alias_rows(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "ka_ki_ku.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "ka_ki_ku.wav=ku,0,0,0,0,0\n"
        "ka_ki_ku.wav=ka,0,0,0,0,0\n"
        "ka_ki_ku.wav=ki,0,0,0,0,0\n",
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
    monkeypatch.setenv("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", "0")

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
    assert processed == 3
    assert total == 3
    assert [(call["alias"], call["row_index_in_wav"], call["file_row_count"]) for call in calls] == [
        ("ku", 2, 3),
        ("ka", 0, 3),
        ("ki", 1, 3),
    ]
    assert calls[1]["prev_alias"] == ""
    assert calls[1]["next_alias"] == "ki"


def test_crnn_oto_generator_maps_korean_vc_aliases_by_vowel_slot(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "ga_gi_gu.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "ga_gi_gu.wav=u k,0,0,0,0,0\n"
        "ga_gi_gu.wav=a k,0,0,0,0,0\n"
        "ga_gi_gu.wav=i k,0,0,0,0,0\n",
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
    monkeypatch.setenv("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_ENABLE", "0")

    processed, total, errors = gen.generate_oto_with_crnn_predictor(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        source_oto_path=str(source_oto),
        language="korean",
        format_type="cvvc",
        model_path=str(model_path),
        device="cpu",
    )

    assert errors == []
    assert processed == 3
    assert total == 3
    assert [(call["alias"], call["row_index_in_wav"], call["file_row_count"]) for call in calls] == [
        ("u k", 2, 3),
        ("a k", 0, 3),
        ("i k", 1, 3),
    ]


def test_crnn_oto_generator_uses_kana_filename_slots(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "か_き_く.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "か_き_く.wav=く,0,0,0,0,0\n"
        "か_き_く.wav=か,0,0,0,0,0\n"
        "か_き_く.wav=き,0,0,0,0,0\n",
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
    monkeypatch.setenv("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", "0")

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
    assert processed == 3
    assert total == 3
    assert [(call["alias"], call["row_index_in_wav"], call["file_row_count"]) for call in calls] == [
        ("く", 2, 3),
        ("か", 0, 3),
        ("き", 1, 3),
    ]


def test_alias_slot_lock_translates_wrong_slot_params(tmp_path):
    from core.coarse_crnn import oto_predictor_generator as gen
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak

    wav_path = tmp_path / "ka_ki_ku.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    state = {
        "row": SimpleNamespace(
            wav_abs=wav_abs,
            row_index=2,
            row_count=3,
            uses_filename_slots=True,
        ),
        "params": {
            "offset": 10.0,
            "consonant": 90.0,
            "cutoff": -520.0,
            "preutterance": 50.0,
            "overlap": 20.0,
        },
        "duration_ms": 800.0,
        "alias_slot_locked": False,
        "alias_slot_lock_hold": False,
    }
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=800.0,
        active_start_ms=60.0,
        active_end_ms=540.0,
        onset_peaks=[
            OnsetPeak(time_ms=60.0, strength=0.95),
            OnsetPeak(time_ms=300.0, strength=0.95),
            OnsetPeak(time_ms=540.0, strength=0.95),
        ],
        stable_vowel_segments=[],
        mel_delta_peaks=[],
    )

    moved, held = gen._apply_alias_slot_lock_alignment(
        row_states=[state],
        cache={wav_abs.lower(): audio},
        config=SimpleNamespace(),
        max_shift_ms=3000.0,
        min_move_ms=30.0,
    )

    assert moved == 1
    assert held == 0
    assert state["alias_slot_locked"] is True
    assert state["alias_slot_lock_reason"].startswith("direct:0->2:")
    assert state["params"]["offset"] == 490.0
    assert state["params"]["offset"] + state["params"]["preutterance"] == 540.0


def test_alias_slot_lock_direct_mode_lands_preutterance_on_target_slot(tmp_path):
    from core.coarse_crnn import oto_predictor_generator as gen
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak

    wav_path = tmp_path / "ka_ki_ku.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    state = {
        "row": SimpleNamespace(
            wav_abs=wav_abs,
            row_index=2,
            row_count=3,
            uses_filename_slots=True,
        ),
        "params": {
            "offset": 10.0,
            "consonant": 120.0,
            "cutoff": -500.0,
            "preutterance": 80.0,
            "overlap": 25.0,
        },
        "duration_ms": 800.0,
        "alias_slot_locked": False,
        "alias_slot_lock_hold": False,
    }
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=800.0,
        active_start_ms=60.0,
        active_end_ms=540.0,
        onset_peaks=[
            OnsetPeak(time_ms=60.0, strength=0.95),
            OnsetPeak(time_ms=300.0, strength=0.95),
            OnsetPeak(time_ms=540.0, strength=0.95),
        ],
        stable_vowel_segments=[],
        mel_delta_peaks=[],
    )

    moved, held = gen._apply_alias_slot_lock_alignment(
        row_states=[state],
        cache={wav_abs.lower(): audio},
        config=SimpleNamespace(),
        max_shift_ms=3000.0,
        min_move_ms=30.0,
        mode="direct",
    )

    assert moved == 1
    assert held == 0
    assert state["alias_slot_lock_reason"].startswith("direct:0->2:")
    assert state["params"]["offset"] == 460.0
    assert state["params"]["offset"] + state["params"]["preutterance"] == 540.0


def test_alias_slot_lock_vc_uses_transition_target_instead_of_slot_onset(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak, VowelSegmentCandidate

    wav_path = tmp_path / "a_ka.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    state = {
        "row": SimpleNamespace(
            wav_abs=wav_abs,
            alias="a k",
            alias_role="vc",
            row_index=0,
            row_count=2,
            uses_filename_slots=True,
        ),
        "params": {
            "offset": 20.0,
            "consonant": 85.0,
            "cutoff": -460.0,
            "preutterance": 40.0,
            "overlap": 18.0,
        },
        "duration_ms": 620.0,
        "alias_slot_locked": False,
        "alias_slot_lock_hold": False,
    }
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=620.0,
        active_start_ms=60.0,
        active_end_ms=540.0,
        onset_peaks=[
            OnsetPeak(time_ms=60.0, strength=0.95),
            OnsetPeak(time_ms=300.0, strength=0.92),
        ],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=90.0,
                end_ms=220.0,
                center_ms=155.0,
                vowel_id="a",
                vowel_confidence=0.92,
                voiced_score=0.90,
                stability_score=0.88,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[],
    )
    monkeypatch.setenv("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_TARGET_RATIO", "0.50")

    moved, held = gen._apply_alias_slot_lock_alignment(
        row_states=[state],
        cache={wav_abs.lower(): audio},
        config=SimpleNamespace(),
        max_shift_ms=3000.0,
        min_move_ms=30.0,
        mode="direct",
    )

    assert moved == 1
    assert held == 0
    # VC should land near the transition between vowel tail and next onset:
    # 220 + (300 - 220) * 0.50 = 260, not the left slot onset at 60.
    assert state["params"]["offset"] + state["params"]["preutterance"] == 260.0


def test_alias_slot_lock_vc_anchor_pair_constrains_target_window(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak, VowelSegmentCandidate

    wav_path = tmp_path / "a_ka.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=700.0,
        active_start_ms=40.0,
        active_end_ms=620.0,
        onset_peaks=[
            OnsetPeak(time_ms=60.0, strength=0.95),
            OnsetPeak(time_ms=540.0, strength=0.92),
        ],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=130.0,
                end_ms=220.0,
                center_ms=175.0,
                vowel_id="a",
                vowel_confidence=0.92,
                voiced_score=0.90,
                stability_score=0.88,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[],
    )
    monkeypatch.setenv("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_TARGET_RATIO", "0.50")

    target = gen._alias_slot_lock_target_time(
        audio,
        role="vc",
        target_slot_idx=0,
        slot_times=[60.0, 540.0],
        anchor_times=[120.0, 300.0],
    )

    # The next onset at 540 ms is outside the CV-anchor pair window, so VC must
    # stay inside the [120, 300] anchor pair instead of drifting toward 540 ms.
    assert target == 220.0


def test_crnn_oto_generator_does_not_use_base_oto_fallback_by_default(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a.wav=a k,0,0,0,0,0\n", encoding="utf-8")
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
                "offset": 100.0,
                "consonant": 110.0,
                "cutoff": -170.0,
                "preutterance": 80.0,
                "overlap": 30.0,
            },
            anchors=SimpleNamespace(
                offset=100.0,
                overlap=30.0,
                preutterance=80.0,
                consonant=110.0,
                cutoff=-170.0,
            ),
            confidence=0.10,
            low_confidence=True,
            predicted_error_ms=800.0,
            duration_ms=500.0,
        )

    monkeypatch.setattr(gen, "load_oto_checkpoint", lambda *_a, **_k: (FakeModel(), SimpleNamespace(), {}))
    monkeypatch.setattr(gen, "predict_oto_with_model", fake_predict)
    monkeypatch.delenv("UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK", raising=False)
    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_FALLBACK_ENABLE", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_ENABLE", "0")

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
    assert processed == 1
    assert total == 1
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "a.wav=a k,100.000,110.000,-170.000,80.000,30.000",
    ]


def test_crnn_oto_generator_blocks_experimental_boundary_model_by_default(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
    model_path = tmp_path / "oto_anchor_crnn_boundary_slot_e2.pt"
    model_path.write_bytes(b"fake")
    out_path = tmp_path / "out.ini"

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        gen,
        "load_oto_checkpoint",
        lambda *_a, **_k: (FakeModel(), SimpleNamespace(enable_boundary_slot_head=True), {}),
    )
    monkeypatch.delenv("UTOA_OTO_CRNN_ALLOW_EXPERIMENTAL_MODEL", raising=False)

    processed, total, errors = gen.generate_oto_with_crnn_predictor(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        source_oto_path=str(source_oto),
        language="japanese",
        format_type="vcv",
        model_path=str(model_path),
        device="cpu",
    )

    assert processed == 0
    assert total == 1
    assert "experimental model is blocked" in errors[0]
    assert not out_path.exists()


def test_crnn_oto_generator_blocks_base_fallback_on_low_order_quality(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as gen

    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a_b_c_d.wav")
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a_b_c_d.wav=a k,300,210,-380,180,90\n", encoding="utf-8")
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
                "offset": 80.0,
                "consonant": 120.0,
                "cutoff": -210.0,
                "preutterance": 95.0,
                "overlap": 45.0,
            },
            anchors=SimpleNamespace(
                offset=80.0,
                overlap=45.0,
                preutterance=95.0,
                consonant=120.0,
                cutoff=-210.0,
            ),
            confidence=0.10,
            low_confidence=True,
            predicted_error_ms=900.0,
            duration_ms=700.0,
        )

    monkeypatch.setattr(gen, "load_oto_checkpoint", lambda *_a, **_k: (FakeModel(), SimpleNamespace(), {}))
    monkeypatch.setattr(gen, "predict_oto_with_model", fake_predict)
    monkeypatch.setenv("UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK", "1")
    monkeypatch.delenv("UTOA_OTO_CRNN_BASE_FALLBACK_QUALITY_GATE_ENABLE", raising=False)
    monkeypatch.delenv("UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_ENABLE", "0")

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
    assert processed == 1
    assert total == 1
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "a_b_c_d.wav=a k,80.000,120.000,-202.000,95.000,45.000",
    ]


def test_filename_alias_order_score_cvvc_uses_core_alias_tokens():
    from core.coarse_crnn.oto_predictor_generator import _estimate_filename_alias_order_ratio

    score, expected_count, is_cvvc = _estimate_filename_alias_order_ratio(
        "_beu'bwa'bwi'bo'bwe'bu'bweo'bwa.wav",
        [
            "eu bw",
            "bwa",
            "a bw",
            "bwi",
            "i bw",
            "bo",
            "o bw",
            "bwe",
            "e bw",
            "bu",
            "u bw",
            "bweo",
            "eo bw",
        ],
    )

    assert is_cvvc
    assert expected_count >= 6
    assert score >= 0.30


def test_filename_alias_order_score_non_cvvc_remains_strict():
    from core.coarse_crnn.oto_predictor_generator import _estimate_filename_alias_order_ratio

    score, expected_count, is_cvvc = _estimate_filename_alias_order_ratio(
        "a_b_c_d.wav",
        ["a k"],
    )

    assert expected_count == 4
    assert isinstance(is_cvvc, bool)
    assert score < 0.42


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


def test_low_confidence_fallback_requires_multi_signal_votes(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_MIN_TRIGGER_VOTES", raising=False)
    out, reason = _apply_low_confidence_fallback(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -160.0,
            "preutterance": 55.0,
            "overlap": 20.0,
        },
        predicted_confidence=0.34,
        predicted_error_ms=320.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.31, "uncertainty": 0.010},
        base_params={
            "offset": 10.0,
            "consonant": 40.0,
            "cutoff": -90.0,
            "preutterance": 25.0,
            "overlap": 8.0,
        },
        language="japanese",
        alias="a",
        duration_ms=1200.0,
        is_special=False,
    )

    assert "low_confidence" in reason
    assert "component_low_quality" in reason
    assert out["offset"] < 100.0


def test_low_confidence_fallback_does_not_trigger_on_single_signal(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_MIN_TRIGGER_VOTES", raising=False)
    out, reason = _apply_low_confidence_fallback(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -160.0,
            "preutterance": 55.0,
            "overlap": 20.0,
        },
        predicted_confidence=0.34,
        predicted_error_ms=200.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.55, "uncertainty": 0.040},
        base_params={
            "offset": 10.0,
            "consonant": 40.0,
            "cutoff": -90.0,
            "preutterance": 25.0,
            "overlap": 8.0,
        },
        language="japanese",
        alias="a",
        duration_ms=1200.0,
        is_special=False,
    )

    assert reason == ""
    assert out["offset"] == 100.0


def test_low_confidence_fallback_role_adapt_relaxes_threshold(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_low_confidence_fallback

    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_ROLES", "other,cv,v,vc,vv,v-cv,special")
    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_CONF_DELTA", "0.05")
    monkeypatch.setenv("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_EXTRA_VOTES", "1")
    monkeypatch.delenv("UTOA_OTO_CRNN_LOW_CONF_MIN_TRIGGER_VOTES", raising=False)
    out, reason = _apply_low_confidence_fallback(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -160.0,
            "preutterance": 55.0,
            "overlap": 20.0,
        },
        predicted_confidence=0.32,
        predicted_error_ms=300.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.31, "uncertainty": 0.010},
        base_params={
            "offset": 10.0,
            "consonant": 40.0,
            "cutoff": -90.0,
            "preutterance": 25.0,
            "overlap": 8.0,
        },
        language="korean",
        alias="@@unknown@@",
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


def test_activity_indices_can_pick_strongest_run():
    from core.coarse_crnn.oto_predictor_generator import _sustained_frame_indices

    mask = np.asarray([False, True, True, False, True, True, True, False], dtype=bool)
    rms = np.asarray([0.01, 0.08, 0.09, 0.02, 0.22, 0.24, 0.20, 0.01], dtype=np.float32)

    idx = _sustained_frame_indices(mask, 2, values=rms, prefer_strongest=True)

    assert idx.size >= 2
    assert int(idx[0]) == 4


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


def test_row_activity_shift_auto_enables_on_low_snr(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_row_activity_window

    monkeypatch.delenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_LOW_SNR_ENABLE", "1")

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
        low_snr=True,
        language="korean",
        alias="ga",
        is_special=False,
    )

    assert out
    assert 110.0 <= out["offset"] <= 130.0
    assert out["preutterance"] == 70.0
    assert out["consonant"] == 105.0


def test_row_activity_shift_auto_enables_on_front_early_prediction(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_row_activity_window

    monkeypatch.delenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_FRONT_EARLY_ENABLE", "1")
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_EARLY_MARGIN_MS", "70")

    out = _shift_params_into_row_activity_window(
        predicted_anchors={
            "offset": 5.0,
            "overlap": 8.0,
            "preutterance": 10.0,
            "consonant": 20.0,
            "cutoff": 90.0,
        },
        duration_ms=1600.0,
        active_start_ms=100.0,
        active_end_ms=1300.0,
        row_index=2,
        row_count=4,
        low_snr=False,
        language="korean",
        alias="ga",
        is_special=False,
    )

    assert out
    assert out["offset"] > 700.0


def test_row_activity_shift_auto_respects_role_allowlist(monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _shift_params_into_row_activity_window

    monkeypatch.delenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", raising=False)
    monkeypatch.delenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_LOW_SNR_ENABLE", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_ROLES", "cv,-cv,special")

    out = _shift_params_into_row_activity_window(
        predicted_anchors={
            "offset": 5.0,
            "overlap": 8.0,
            "preutterance": 10.0,
            "consonant": 20.0,
            "cutoff": 90.0,
        },
        duration_ms=1600.0,
        active_start_ms=100.0,
        active_end_ms=1300.0,
        row_index=2,
        row_count=4,
        low_snr=True,
        language="korean",
        alias="a k",
        is_special=False,
    )

    assert out == {}


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
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", "0")
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
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", "0")

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


def test_audio_candidate_snap_requires_risky_prediction_by_default(tmp_path, monkeypatch):
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
    monkeypatch.delenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", raising=False)
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
        predicted_confidence=0.92,
        predicted_error_ms=120.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.70, "uncertainty": 0.080},
    )

    assert reason == ""
    assert out["offset"] == 100.0


def test_audio_candidate_snap_applies_when_prediction_is_early_outside_activity(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=220.0,
            active_end_ms=500.0,
            onset_peaks=[SimpleNamespace(time_ms=260.0, strength=0.82)],
        )
    }
    monkeypatch.delenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", raising=False)
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MAX_DELTA_MS", "400")

    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 20.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="japanese",
        alias="ka",
        duration_ms=600.0,
        cache=cache,
        predicted_confidence=0.92,
        predicted_error_ms=120.0,
        predicted_low_confidence=False,
        confidence_components={"heatmap": 0.70, "uncertainty": 0.080},
    )

    assert reason.startswith("candidate_onset:")
    assert out["offset"] > 20.0


def test_audio_candidate_snap_cvvc_role_cap_blocks_large_forward_jump(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_audio_candidate_snap

    wav_path = tmp_path / "cvvc.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: SimpleNamespace(
            active_start_ms=0.0,
            active_end_ms=600.0,
            onset_peaks=[SimpleNamespace(time_ms=260.0, strength=0.90)],
        )
    }

    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CV_MAX_FORWARD_DELTA_MS", "25")

    out, reason = _apply_audio_candidate_snap(
        predicted_params={
            "offset": 100.0,
            "consonant": 80.0,
            "cutoff": -170.0,
            "preutterance": 40.0,
            "overlap": 20.0,
        },
        wav_path=str(wav_path),
        language="korean",
        alias="ga",
        format_type="cvvc",
        duration_ms=600.0,
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


def test_row_order_alignment_enforces_monotonic_onset_mapping(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak
    from core.coarse_crnn.oto_predictor_generator import _PredictRow, _apply_monotonic_row_order_alignment

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    key = wav_abs.lower()
    cache = {
        key: AudioCandidates(
            wav_path=wav_abs,
            duration_ms=700.0,
            active_start_ms=0.0,
            active_end_ms=700.0,
            onset_peaks=[
                OnsetPeak(time_ms=120.0, strength=0.85),
                OnsetPeak(time_ms=320.0, strength=0.82),
            ],
            stable_vowel_segments=[],
            mel_delta_peaks=[],
        )
    }
    states = [
        {
            "row": _PredictRow(wav_rel="a.wav", wav_abs=wav_abs, alias="a", prev_alias="", next_alias="a k", row_index=0, row_count=2),
            "params": {"offset": 300.0, "consonant": 340.0, "cutoff": -430.0, "preutterance": 300.0, "overlap": 280.0},
            "duration_ms": 700.0,
            "predicted_confidence": 0.7,
            "predicted_error_ms": 120.0,
            "predicted_low_confidence": False,
            "confidence_components": {"heatmap": 0.6, "uncertainty": 0.09},
        },
        {
            "row": _PredictRow(wav_rel="a.wav", wav_abs=wav_abs, alias="ka", prev_alias="a", next_alias="", row_index=1, row_count=2),
            "params": {"offset": 120.0, "consonant": 160.0, "cutoff": -250.0, "preutterance": 120.0, "overlap": 100.0},
            "duration_ms": 700.0,
            "predicted_confidence": 0.7,
            "predicted_error_ms": 120.0,
            "predicted_low_confidence": False,
            "confidence_components": {"heatmap": 0.6, "uncertainty": 0.09},
        },
    ]

    aligned, held = _apply_monotonic_row_order_alignment(
        row_states=states,
        cache=cache,
        config=None,
        language="korean",
    )

    assert aligned >= 1
    assert held >= 0
    assert any(bool(state.get("aligned_by_row_order")) or bool(state.get("row_order_hold")) for state in states)


def test_row_order_alignment_holds_large_jump(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak
    from core.coarse_crnn.oto_predictor_generator import _PredictRow, _apply_monotonic_row_order_alignment

    wav_path = tmp_path / "b.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    key = wav_abs.lower()
    cache = {
        key: AudioCandidates(
            wav_path=wav_abs,
            duration_ms=1200.0,
            active_start_ms=0.0,
            active_end_ms=1200.0,
            onset_peaks=[
                OnsetPeak(time_ms=110.0, strength=0.90),
                OnsetPeak(time_ms=950.0, strength=0.88),
            ],
            stable_vowel_segments=[],
            mel_delta_peaks=[],
        )
    }
    states = [
        {
            "row": _PredictRow(wav_rel="b.wav", wav_abs=wav_abs, alias="a", prev_alias="", next_alias="a k", row_index=0, row_count=2),
            "params": {"offset": 90.0, "consonant": 130.0, "cutoff": -220.0, "preutterance": 90.0, "overlap": 70.0},
            "duration_ms": 1200.0,
            "predicted_confidence": 0.8,
            "predicted_error_ms": 90.0,
            "predicted_low_confidence": False,
            "confidence_components": {"heatmap": 0.7, "uncertainty": 0.09},
        },
        {
            "row": _PredictRow(wav_rel="b.wav", wav_abs=wav_abs, alias="ka", prev_alias="a", next_alias="", row_index=1, row_count=2),
            "params": {"offset": 120.0, "consonant": 160.0, "cutoff": -250.0, "preutterance": 120.0, "overlap": 100.0},
            "duration_ms": 1200.0,
            "predicted_confidence": 0.8,
            "predicted_error_ms": 90.0,
            "predicted_low_confidence": False,
            "confidence_components": {"heatmap": 0.7, "uncertainty": 0.09},
        },
    ]

    aligned, held = _apply_monotonic_row_order_alignment(
        row_states=states,
        cache=cache,
        config=None,
        language="korean",
    )

    assert aligned >= 1
    assert held >= 1


def test_row_order_alignment_skips_vc_boundary_rows_by_default(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak
    from core.coarse_crnn.oto_predictor_generator import _PredictRow, _apply_monotonic_row_order_alignment

    wav_path = tmp_path / "vc-order.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    cache = {
        wav_abs.lower(): AudioCandidates(
            wav_path=wav_abs,
            duration_ms=700.0,
            active_start_ms=0.0,
            active_end_ms=700.0,
            onset_peaks=[
                OnsetPeak(time_ms=120.0, strength=0.85),
                OnsetPeak(time_ms=320.0, strength=0.82),
            ],
            stable_vowel_segments=[],
            mel_delta_peaks=[],
        )
    }
    states = [
        {
            "row": _PredictRow(wav_rel="vc-order.wav", wav_abs=wav_abs, alias="a", prev_alias="", next_alias="a k", row_index=0, row_count=2),
            "params": {"offset": 300.0, "consonant": 340.0, "cutoff": -430.0, "preutterance": 300.0, "overlap": 280.0},
            "duration_ms": 700.0,
        },
        {
            "row": _PredictRow(wav_rel="vc-order.wav", wav_abs=wav_abs, alias="a k", prev_alias="a", next_alias="", row_index=1, row_count=2),
            "params": {"offset": 120.0, "consonant": 160.0, "cutoff": -250.0, "preutterance": 120.0, "overlap": 100.0},
            "duration_ms": 700.0,
        },
    ]

    aligned, held = _apply_monotonic_row_order_alignment(
        row_states=states,
        cache=cache,
        config=None,
        language="korean",
    )

    assert aligned == 0
    assert held == 0
    assert not any(bool(state.get("aligned_by_row_order")) or bool(state.get("row_order_hold")) for state in states)


def test_vc_candidate_matcher_moves_toward_vc_transition(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.oto_predictor_generator import _apply_vc_candidate_matcher

    wav_path = tmp_path / "vc.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {
        key: AudioCandidates(
            wav_path=str(wav_path),
            duration_ms=700.0,
            active_start_ms=0.0,
            active_end_ms=700.0,
            onset_peaks=[OnsetPeak(time_ms=300.0, strength=0.92)],
            stable_vowel_segments=[
                VowelSegmentCandidate(
                    segment_id=0,
                    start_ms=140.0,
                    end_ms=220.0,
                    center_ms=180.0,
                    vowel_id="a",
                    vowel_confidence=0.90,
                    voiced_score=0.85,
                    stability_score=0.80,
                    energy_mean=0.60,
                    mel_delta_mean=0.08,
                )
            ],
            mel_delta_peaks=[],
        )
    }
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_ENABLE", "1")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_REQUIRE_RISKY", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_REQUIRE_BOUNDARY_RISK", "0")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_BLEND", "1.0")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_TARGET_RATIO", "0.50")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_MAX_DELTA_MS", "260")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_ALLOWED_FORMAT_ROLES", "cvvc|vc")

    out, reason = _apply_vc_candidate_matcher(
        predicted_params={
            "offset": 80.0,
            "consonant": 120.0,
            "cutoff": -220.0,
            "preutterance": 95.0,
            "overlap": 60.0,
        },
        wav_path=str(wav_path),
        language="korean",
        alias="a k",
        format_type="cvvc",
        duration_ms=700.0,
        cache=cache,
    )

    assert reason.startswith("vc_matcher:")
    assert out["offset"] == 80.0
    assert out["preutterance"] > 95.0


def test_vc_candidate_matcher_skips_non_vc_role(tmp_path, monkeypatch):
    from core.coarse_crnn.oto_predictor_generator import _apply_vc_candidate_matcher

    wav_path = tmp_path / "nonvc.wav"
    wav_path.write_bytes(b"fake")
    key = str(wav_path.resolve()).lower()
    cache = {key: None}
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_ENABLE", "1")
    monkeypatch.setenv("UTOA_OTO_CRNN_VC_MATCHER_ALLOWED_FORMAT_ROLES", "cvvc|vc")

    original = {
        "offset": 30.0,
        "consonant": 70.0,
        "cutoff": -150.0,
        "preutterance": 50.0,
        "overlap": 20.0,
    }
    out, reason = _apply_vc_candidate_matcher(
        predicted_params=original,
        wav_path=str(wav_path),
        language="japanese",
        alias="a",
        format_type="cvvc",
        duration_ms=400.0,
        cache=cache,
    )

    assert reason == ""
    assert out == original
