"""Label-free structural regressions; these are not acoustic accuracy Gold."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.alignment import _refine_landmark
from universal_auto_oto.acoustic_evidence import extract_acoustic_evidence
from universal_auto_oto.domain import Landmark, PhoneSegment, StateSegment
from universal_auto_oto.hsmm_decoder import (
    HSMMStateSpec, HSMMStateFrameWindow, decode_segmental_hsmm,
    decode_segmental_hsmm_coarse_to_refine,
)


@pytest.mark.parametrize("full_coverage", [False, True])
def test_terminal_beam_preserves_actual_objective(full_coverage):
    state = HSMMStateSpec("v0", "vowel", 10, 100)
    result = decode_segmental_hsmm(
        [state], list(range(0, 100, 10)), {"v0": [1.0] * 10},
        beam_width_per_state=1, allow_trailing_gap=not full_coverage,
        trailing_gap_penalty_per_ms=0.1,
    )
    assert result.ok
    assert result.states[-1].end_ms == 100


@pytest.mark.parametrize("times", [[0, 0, 5], [0, 10, 5], [0, float("nan"), 10]])
def test_invalid_timebase_is_rejected(times):
    result = decode_segmental_hsmm([HSMMStateSpec("v", "v", 5, 15)], times, {"v": [1] * 3})
    assert not result.ok and result.reason == "invalid_frame_times"


def test_duplicate_occurrence_and_inverted_window_are_rejected():
    state = HSMMStateSpec("v", "v", 5, 10)
    assert decode_segmental_hsmm([state, state], [0, 5, 10], {}).reason == "duplicate_state_id"
    result = decode_segmental_hsmm(
        [state], [0, 5, 10], {}, state_frame_windows={"v": HSMMStateFrameWindow(2, 1)},
    )
    assert not result.ok and result.reason == "invalid_state_window"


@pytest.mark.parametrize("decoder", [decode_segmental_hsmm, decode_segmental_hsmm_coarse_to_refine])
def test_repeated_occurrences_are_equivariant_to_time_translation(decoder):
    times = np.arange(80) * 5.0
    states = [HSMMStateSpec(f"v{i}", "vowel", 100, 100) for i in range(2)]
    scores = {"v0": np.where((times >= 50) & (times < 150), 2., -2.),
              "v1": np.where((times >= 220) & (times < 320), 2., -2.)}
    kwargs = dict(allow_leading_gap=True, allow_internal_gaps=True, max_internal_gap_ms=100)
    a = decoder(states, times, scores, **kwargs)
    b = decoder(states, times + 135, scores, **kwargs)
    assert a.ok and b.ok
    assert [s.state_id for s in a.states] == ["v0", "v1"]
    assert [(s.start_frame, s.end_frame) for s in a.states] == [(s.start_frame, s.end_frame) for s in b.states]
    assert [s.start_ms + 135 for s in a.states] == [s.start_ms for s in b.states]


def test_refiner_cannot_jump_to_another_state_peak():
    states = tuple(StateSegment(f"p0.{role}", 0, role, role, start, end, 0.9)
                   for role, start, end in [("closure", 0, 40), ("release", 40, 60),
                                            ("transition", 60, 100), ("vowel_stable", 100, 200)])
    phone = PhoneSegment(0, "a", 0, 200, .9, states)
    times = np.arange(0, 205, 5.)
    flux = np.zeros(len(times))
    flux[times == 45] = .5
    flux[times == 100] = 1.
    evidence = SimpleNamespace(frame_times_ms=times, tracks={"flux": flux})
    result = _refine_landmark(Landmark("release", 40, 0, .9, "hsmm_raw", 40), evidence, phone)
    assert result.time_ms == 45
    assert result.raw_time_ms == 40
    assert result.phone_index == 0


def test_refiner_tied_peak_uses_nearest_time():
    phone = PhoneSegment(0, "a", 0, 200, .9, ())
    times = np.arange(0, 205, 5.)
    track = np.zeros(len(times))
    track[times == 60] = track[times == 110] = 1.
    result = _refine_landmark(
        Landmark("release", 100, 0, .9, "hsmm_raw", 100),
        SimpleNamespace(frame_times_ms=times, tracks={"flux": track}), phone,
    )
    assert result.time_ms == 110


@pytest.mark.parametrize("rate", [16000, 22050, 44100, 48000])
def test_timebase_metadata_matches_sample_quantized_frames(tmp_path, rate):
    path = tmp_path / "tone.wav"
    sf.write(path, .2 * np.sin(2 * np.pi * 220 * np.arange(rate // 2) / rate), rate)
    evidence = extract_acoustic_evidence(path, internal_sample_rate=rate, convert_to_mono=True, resampler="soxr_hq")
    timebase = evidence.metadata["timebase"]
    assert np.diff(evidence.frame_times_ms) == pytest.approx(timebase["frame_shift_ms"])
    assert timebase["frame_shift_ms"] == pytest.approx(round(rate * .005) * 1000 / rate)
    assert timebase["window_size_ms"] == pytest.approx(round(rate * .030) * 1000 / rate)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_padding_backend_keeps_tracks_identical_under_digital_prefix(tmp_path, rate):
    from universal_auto_oto.acoustic_evidence import PADDING_STABLE_FEATURE_BACKEND_ID
    signal = .2 * np.sin(2 * np.pi * 220 * np.arange(rate // 2) / rate)
    signals = [signal, np.concatenate((np.zeros(round(rate * .2)), signal))]
    evidence = []
    for index, samples in enumerate(signals):
        path = tmp_path / f"{index}.wav"
        sf.write(path, samples, rate, subtype="FLOAT")
        evidence.append(extract_acoustic_evidence(path, internal_sample_rate=16000,
            convert_to_mono=True, resampler="soxr_hq", feature_backend=PADDING_STABLE_FEATURE_BACKEND_ID))
    a, b = evidence
    assert b.frame_times_ms == pytest.approx(a.frame_times_ms + 200)
    for name in a.tracks:
        np.testing.assert_array_equal(a.tracks[name], b.tracks[name])


def test_padding_backend_preserves_explicit_silence_and_all_zero_input(tmp_path):
    from universal_auto_oto.acoustic_evidence import PADDING_STABLE_FEATURE_BACKEND_ID
    for values in [np.zeros(8000), np.r_[np.zeros(3200), np.ones(4800)*.1]]:
        path = tmp_path / "silence.wav"
        sf.write(path, values, 16000)
        e = extract_acoustic_evidence(path, internal_sample_rate=16000, convert_to_mono=True,
            resampler="soxr_hq", feature_backend=PADDING_STABLE_FEATURE_BACKEND_ID, preserve_leading_silence=True)
        assert e.metadata["analysis_origin_ms"] == 0
        assert e.duration_ms == 500
        assert all(np.isfinite(v).all() for v in e.tracks.values())
