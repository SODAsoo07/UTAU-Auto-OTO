from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from universal_auto_oto.alignment import _cvvc_slot_occurrence_priors
from universal_auto_oto.domain import CompiledState, GuideAnchor, RecordingSpec


def fixture(codas=True):
    phones, slots, states = [], [], []
    for i in range(4):
        start = len(phones)
        phones.extend(("d", "i"))
        if codas and i < 2:
            phones.append("n")
        slots.append({"start": start, "vowel": start + 1, "end": len(phones) - 1})
    for i, phone in enumerate(phones):
        states.append(CompiledState(f"p{i}", i, phone, "stable", "vowel_stable" if phone == "i" else "consonant",
                                    10., 1000., 300. if phone == "i" else 60.))
    recording = RecordingSpec("di_di_di_di.wav", tuple(phones), (), transcript_source="original_filename_only",
        label_source="none", metadata={"format_hint": "cvvc", "language_hint": "korean", "phone_slots": slots})
    times = np.arange(0., 2400., 5.)
    signal = sum(np.exp(-.5 * ((times - peak) / 10.) ** 2) for peak in (140., 640., 1140., 1640.))
    evidence = SimpleNamespace(frame_times_ms=times, active_start_ms=100., active_end_ms=2100.,
        tracks={key: signal for key in ("vowel_boundary_likelihood", "transition", "periodicity_rise", "onset_strength")})
    return states, evidence, recording


def test_coda_states_do_not_steal_time_from_later_cv_slots():
    later_centers = []
    for codas in (True, False):
        states, evidence, recording = fixture(codas)
        priors, applied, policy = _cvvc_slot_occurrence_priors(states, evidence, recording)
        assert applied and policy["anchors_ms"] == [140., 640., 1140., 1640.]
        assert policy["review_only"] and not policy["promotion_eligible"] and not policy["anchors_are_gold"]
        last = recording.metadata["phone_slots"][-1]["vowel"]
        later_centers.append(evidence.frame_times_ms[np.argmax(priors[f"p{last}"])])
    assert later_centers[0] == later_centers[1]


def test_weak_cues_and_authored_or_unsupported_topology_keep_existing_path():
    states, evidence, recording = fixture()
    weak = SimpleNamespace(**{**vars(evidence), "tracks": {key: np.zeros_like(value) for key, value in evidence.tracks.items()}})
    assert _cvvc_slot_occurrence_priors(states, weak, recording) is None
    for other in (replace(recording, transcript_source="voicebank_recording_manifest"),
                  replace(recording, label_source="manual_gold"),
                  replace(recording, guide_anchors=(GuideAnchor("start", 100., 1.),)),
                  replace(recording, metadata={**recording.metadata, "format_hint": "vcv"}),
                  replace(recording, metadata={**recording.metadata, "phone_slots": [{"start": 0, "vowel": 1, "end": 999}]})):
        assert _cvvc_slot_occurrence_priors(states, evidence, other) is None


def test_nearby_weak_peak_does_not_hide_a_valid_rhythm_cue():
    states, evidence, recording = fixture()
    times = evidence.frame_times_ms
    signal = sum(np.exp(-.5 * ((times - peak) / 10.) ** 2) for peak in (140., 1140., 1640.))
    signal += .34 * np.exp(-.5 * ((times - 640.) / 10.) ** 2)
    signal += .50 * np.exp(-.5 * ((times - 740.) / 10.) ** 2)
    evidence.tracks = {key: signal for key in evidence.tracks}
    _, applied, policy = _cvvc_slot_occurrence_priors(states, evidence, recording)
    assert applied and policy["anchors_ms"][1] == 740.


def test_grid_refit_moves_only_an_anchor_that_left_the_sung_grid():
    states, evidence, recording = fixture()
    # A long active tail inflates the first-pass period to 575 ms.
    evidence.active_end_ms = 2400.
    times = evidence.frame_times_ms
    # Slot 3 has its true onset on the 500 ms grid (1640) and a stronger peak
    # 150 ms late (1790) that the first pass prefers.
    signal = sum(np.exp(-.5 * ((times - peak) / 10.) ** 2) for peak in (140., 640., 1140.))
    signal += .60 * np.exp(-.5 * ((times - 1640.) / 10.) ** 2) + np.exp(-.5 * ((times - 1790.) / 10.) ** 2)
    evidence.tracks = {key: signal for key in evidence.tracks}
    _, applied, first = _cvvc_slot_occurrence_priors(states, evidence, recording)
    assert applied and first["anchors_ms"][3] == 1790. and "grid_refit" not in first
    _, applied, refit = _cvvc_slot_occurrence_priors(states, evidence, recording, grid_refit=True)
    assert applied and refit["anchors_ms"] == [140., 640., 1140., 1640.]
    assert refit["grid_refit"]["applied"] and refit["grid_refit"]["moved_slots"] == [3]


def test_grid_refit_keeps_first_pass_without_a_better_peak_near_the_grid():
    states, evidence, recording = fixture()
    evidence.active_end_ms = 2400.
    times = evidence.frame_times_ms
    signal = sum(np.exp(-.5 * ((times - peak) / 10.) ** 2) for peak in (140., 640., 1140., 1790.))
    evidence.tracks = {key: signal for key in evidence.tracks}
    _, applied, policy = _cvvc_slot_occurrence_priors(states, evidence, recording, grid_refit=True)
    assert applied and policy["anchors_ms"] == [140., 640., 1140., 1790.]
    assert policy["grid_refit"]["moved_slots"] == []
