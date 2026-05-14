from __future__ import annotations

from types import SimpleNamespace


def test_collect_boundary_candidates_prefers_role_sources():
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, MelDeltaPeak, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.oto_boundary_decoding import collect_boundary_candidates

    candidates = AudioCandidates(
        wav_path="x.wav",
        duration_ms=600.0,
        active_start_ms=30.0,
        active_end_ms=560.0,
        onset_peaks=[OnsetPeak(time_ms=310.0, strength=0.90)],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=120.0,
                end_ms=220.0,
                center_ms=170.0,
                vowel_id="a",
                vowel_confidence=0.90,
                voiced_score=0.85,
                stability_score=0.80,
                energy_mean=0.60,
                mel_delta_mean=0.08,
            )
        ],
        mel_delta_peaks=[MelDeltaPeak(time_ms=245.0, strength=0.70)],
    )

    vc = collect_boundary_candidates(candidates, role="vc", duration_ms=600.0)
    vv = collect_boundary_candidates(candidates, role="vv", duration_ms=600.0)

    assert any(item.source == "vowel_end" and item.time_ms == 220.0 for item in vc)
    assert any(item.source == "mel_delta" and item.time_ms == 245.0 for item in vv)


def test_correct_boundary_params_moves_right_side_only_for_vc():
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.oto_boundary_decoding import correct_boundary_params_from_candidates

    candidates = AudioCandidates(
        wav_path="x.wav",
        duration_ms=700.0,
        active_start_ms=0.0,
        active_end_ms=700.0,
        onset_peaks=[OnsetPeak(time_ms=300.0, strength=0.90)],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=140.0,
                end_ms=240.0,
                center_ms=190.0,
                vowel_id="a",
                vowel_confidence=0.90,
                voiced_score=0.90,
                stability_score=0.85,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[],
    )

    out, reason = correct_boundary_params_from_candidates(
        predicted_params={
            "offset": 100.0,
            "consonant": 90.0,
            "cutoff": -210.0,
            "preutterance": 70.0,
            "overlap": 30.0,
        },
        audio_candidates=candidates,
        role="vc",
        duration_ms=700.0,
        prev_anchor_ms=120.0,
        next_anchor_ms=340.0,
        max_delta_ms=250.0,
        blend=1.0,
    )

    assert reason.startswith("boundary_candidate:")
    assert out["offset"] == 100.0
    assert out["preutterance"] > 70.0


def test_decode_boundary_slot_graph_updates_boundary_state(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, MelDeltaPeak, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.oto_boundary_decoding import decode_boundary_slot_graph_for_states

    wav_path = tmp_path / "graph.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=700.0,
        active_start_ms=0.0,
        active_end_ms=700.0,
        onset_peaks=[OnsetPeak(time_ms=300.0, strength=0.90)],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=140.0,
                end_ms=235.0,
                center_ms=188.0,
                vowel_id="a",
                vowel_confidence=0.90,
                voiced_score=0.90,
                stability_score=0.85,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[MelDeltaPeak(time_ms=250.0, strength=0.70)],
    )
    states = [
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=0, alias="a"),
            "params": {"offset": 100.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 70.0, "overlap": 30.0},
            "duration_ms": 700.0,
        },
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=1, alias="a k"),
            "params": {"offset": 100.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 70.0, "overlap": 30.0},
            "duration_ms": 700.0,
        },
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=2, alias="ka"),
            "params": {"offset": 280.0, "consonant": 80.0, "cutoff": -190.0, "preutterance": 60.0, "overlap": 25.0},
            "duration_ms": 700.0,
        },
    ]
    role_by_alias = {"a": "v", "a k": "vc", "ka": "cv"}

    moved, held = decode_boundary_slot_graph_for_states(
        row_states=states,
        language="korean",
        role_resolver=lambda row: role_by_alias[row.alias],
        candidate_getter=lambda _path: audio,
        max_delta_ms=250.0,
        blend=1.0,
    )

    assert moved == 1
    assert held == 0
    assert states[1]["boundary_slot_graph"] is True
    assert states[1]["params"]["preutterance"] > 70.0
