from __future__ import annotations

from types import SimpleNamespace


def test_sequence_alignment_maps_late_vc_row_between_anchor_slots(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, MelDeltaPeak, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import decode_sequence_alignment_for_states

    wav_path = tmp_path / "a_ka.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=600.0,
        active_start_ms=20.0,
        active_end_ms=520.0,
        onset_peaks=[
            OnsetPeak(time_ms=100.0, strength=0.95),
            OnsetPeak(time_ms=300.0, strength=0.92),
        ],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=110.0,
                end_ms=220.0,
                center_ms=165.0,
                vowel_id="a",
                vowel_confidence=0.90,
                voiced_score=0.90,
                stability_score=0.86,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[MelDeltaPeak(time_ms=230.0, strength=0.70)],
    )
    states = [
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=0, alias="a"),
            "params": {"offset": 50.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 40.0, "overlap": 20.0},
            "duration_ms": 600.0,
        },
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=1, alias="ka"),
            "params": {"offset": 250.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 30.0, "overlap": 15.0},
            "duration_ms": 600.0,
        },
        {
            # CVVC-style source order can put VC rows after anchor rows. The
            # sequence aligner must use boundary ordinal, not raw row time order.
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=2, alias="a k"),
            "params": {"offset": 80.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 20.0, "overlap": 10.0},
            "duration_ms": 600.0,
        },
    ]
    roles = {"a": "v", "ka": "cv", "a k": "vc"}

    moved, held = decode_sequence_alignment_for_states(
        row_states=states,
        language="korean",
        role_resolver=lambda row: roles[row.alias],
        candidate_getter=lambda _path: audio,
        max_shift_ms=260.0,
        anchor_blend=1.0,
        boundary_blend=1.0,
        min_move_ms=1.0,
    )

    assert moved == 3
    assert held == 0
    assert states[0]["sequence_aligned"] is True
    assert states[1]["sequence_aligned"] is True
    assert states[2]["sequence_aligned"] is True
    assert states[2]["params"]["offset"] == 80.0
    assert states[2]["params"]["preutterance"] > 100.0
    assert "vowel_tail" in states[2]["sequence_alignment_reason"]


def test_sequence_alignment_collects_candidate_types():
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, MelDeltaPeak, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import collect_sequence_boundary_candidates

    audio = AudioCandidates(
        wav_path="x.wav",
        duration_ms=500.0,
        active_start_ms=30.0,
        active_end_ms=450.0,
        onset_peaks=[OnsetPeak(time_ms=100.0, strength=0.90)],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=120.0,
                end_ms=220.0,
                center_ms=170.0,
                vowel_id="a",
                vowel_confidence=0.88,
                voiced_score=0.90,
                stability_score=0.84,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[MelDeltaPeak(time_ms=240.0, strength=0.75)],
    )

    candidates = collect_sequence_boundary_candidates(audio)
    kinds = {cand.kind for cand in candidates}

    assert {"vowel_onset", "vowel_tail", "vowel_transition", "silence_boundary"}.issubset(kinds)


def test_sequence_alignment_uses_model_candidate_score_when_available(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak
    from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import decode_sequence_alignment_for_states

    wav_path = tmp_path / "a_ka.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=500.0,
        active_start_ms=10.0,
        active_end_ms=430.0,
        onset_peaks=[
            OnsetPeak(time_ms=100.0, strength=0.75),
            OnsetPeak(time_ms=118.0, strength=0.76),
        ],
        stable_vowel_segments=[],
        mel_delta_peaks=[],
    )
    states = [
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=0, alias="a"),
            "params": {"offset": 70.0, "consonant": 80.0, "cutoff": -200.0, "preutterance": 30.0, "overlap": 15.0},
            "duration_ms": 500.0,
            "sequence_candidate_classes": ["vowel_onset", "vowel_transition", "vowel_tail", "silence_boundary"],
            "sequence_candidate_times_ms": [100.0, 118.0],
            "sequence_candidate_scores": [
                [0.01, 0.0, 0.0, 0.0],
                [0.98, 0.0, 0.0, 0.0],
            ],
        }
    ]

    moved, held = decode_sequence_alignment_for_states(
        row_states=states,
        language="korean",
        role_resolver=lambda _row: "v",
        candidate_getter=lambda _path: audio,
        max_shift_ms=260.0,
        anchor_blend=1.0,
        boundary_blend=1.0,
        min_move_ms=1.0,
        model_score_blend=1.0,
    )

    assert moved == 1
    assert held == 0
    assert states[0]["params"]["offset"] == 88.0
    assert states[0]["params"]["offset"] + states[0]["params"]["preutterance"] == 118.0


def test_sequence_alignment_role_allowlist_keeps_anchor_rows_fixed(tmp_path):
    from core.coarse_crnn.oto_audio_candidates import AudioCandidates, OnsetPeak, VowelSegmentCandidate
    from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import decode_sequence_alignment_for_states

    wav_path = tmp_path / "a_ka.wav"
    wav_path.write_bytes(b"fake")
    wav_abs = str(wav_path.resolve())
    audio = AudioCandidates(
        wav_path=wav_abs,
        duration_ms=600.0,
        active_start_ms=20.0,
        active_end_ms=520.0,
        onset_peaks=[
            OnsetPeak(time_ms=100.0, strength=0.95),
            OnsetPeak(time_ms=300.0, strength=0.92),
        ],
        stable_vowel_segments=[
            VowelSegmentCandidate(
                segment_id=0,
                start_ms=120.0,
                end_ms=230.0,
                center_ms=175.0,
                vowel_id="a",
                vowel_confidence=0.90,
                voiced_score=0.90,
                stability_score=0.86,
                energy_mean=0.60,
                mel_delta_mean=0.05,
            )
        ],
        mel_delta_peaks=[],
    )
    states = [
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=0, alias="a"),
            "params": {"offset": 50.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 40.0, "overlap": 20.0},
            "duration_ms": 600.0,
        },
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=1, alias="ka"),
            "params": {"offset": 250.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 30.0, "overlap": 15.0},
            "duration_ms": 600.0,
        },
        {
            "row": SimpleNamespace(wav_abs=wav_abs, row_index=2, alias="a k"),
            "params": {"offset": 80.0, "consonant": 90.0, "cutoff": -210.0, "preutterance": 20.0, "overlap": 10.0},
            "duration_ms": 600.0,
        },
    ]
    original_anchor_offsets = [states[0]["params"]["offset"], states[1]["params"]["offset"]]
    roles = {"a": "v", "ka": "cv", "a k": "vc"}

    moved, held = decode_sequence_alignment_for_states(
        row_states=states,
        language="korean",
        role_resolver=lambda row: roles[row.alias],
        candidate_getter=lambda _path: audio,
        max_shift_ms=260.0,
        anchor_blend=1.0,
        boundary_blend=1.0,
        min_move_ms=1.0,
        allowed_roles={"vc"},
    )

    assert moved == 1
    assert held == 0
    assert [states[0]["params"]["offset"], states[1]["params"]["offset"]] == original_anchor_offsets
    assert states[0].get("sequence_aligned") is not True
    assert states[1].get("sequence_aligned") is not True
    assert states[2]["sequence_aligned"] is True
    assert states[2]["params"]["offset"] == 80.0
    assert states[2]["params"]["preutterance"] > 100.0
