from types import SimpleNamespace

import numpy as np

from universal_auto_oto.alignment import _cvvc_nasal_coda_boundaries
from universal_auto_oto.domain import RecordingSpec, StateSegment


def fixture(coda="n", murmur_start=300.0):
    phones = ("d", "eu", coda, "d", "eu")
    slots = [{"start": 0, "vowel": 1, "end": 2}, {"start": 3, "vowel": 4, "end": 4}]
    recording = RecordingSpec("deuN_deu.wav", phones, (), transcript_source="original_filename_only",
        label_source="none", metadata={"phone_slots": slots})
    spans = [(0, "closure", "closure", 0, 40), (0, "release", "burst", 40, 60),
             (1, "transition", "vowel_transition", 60, 100), (1, "stable", "vowel_stable", 100, 520),
             (2, "consonant", "consonant", 520, 580), (2, "transition", "transition", 580, 610),
             (3, "closure", "closure", 610, 650), (4, "stable", "vowel_stable", 650, 1000)]
    segments = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    times = np.arange(0.0, 1000.0, 5.0)
    low = np.where((times >= murmur_start) & (times < 610.0), 0.9, 0.3)
    evidence = SimpleNamespace(frame_times_ms=times, tracks={"low_band_ratio": low, "mid_band_ratio": 1.0 - low})
    return recording, evidence, segments


def test_nasal_coda_starts_at_the_band_balance_step_and_nothing_else_moves():
    recording, evidence, segments = fixture()
    adjusted, diagnostics = _cvvc_nasal_coda_boundaries(recording, evidence, segments)
    stable, consonant = adjusted[3], adjusted[4]
    assert abs(consonant.start_ms - 300.0) <= 10.0 and stable.end_ms == consonant.start_ms
    assert consonant.end_ms == 580.0 and consonant.raw_start_ms == 520.0
    assert [s for i, s in enumerate(adjusted) if i not in (3, 4)] == [s for i, s in enumerate(segments) if i not in (3, 4)]
    assert diagnostics["applied_count"] == 1 and not diagnostics["anchors_are_gold"]


def test_weak_step_and_lateral_coda_keep_the_decoded_boundary():
    recording, evidence, segments = fixture()
    flat = SimpleNamespace(frame_times_ms=evidence.frame_times_ms,
                           tracks={key: np.full_like(value, 0.5) for key, value in evidence.tracks.items()})
    adjusted, diagnostics = _cvvc_nasal_coda_boundaries(recording, flat, segments)
    assert adjusted == segments and diagnostics["slots"][0]["reason"] == "weak_band_balance_step"
    recording, evidence, segments = fixture(coda="l")
    adjusted, diagnostics = _cvvc_nasal_coda_boundaries(recording, evidence, segments)
    assert adjusted == segments and diagnostics["slots"] == []


def test_step_cannot_eat_the_vowel_transition_or_the_whole_coda():
    recording, evidence, segments = fixture(murmur_start=80.0)
    adjusted, _ = _cvvc_nasal_coda_boundaries(recording, evidence, segments)
    assert adjusted[4].start_ms >= 100.0 + 40.0
    assert adjusted[4].end_ms - adjusted[4].start_ms >= 30.0


def test_lateral_coda_reuses_the_files_vowel_length_before_nasal_codas():
    from universal_auto_oto.alignment import _codas_from_rhythm
    phones = ("g", "eu", "n", "g", "eu", "m", "g", "eu", "l")
    slots = [{"start": 0, "vowel": 1, "end": 2}, {"start": 3, "vowel": 4, "end": 5}, {"start": 6, "vowel": 7, "end": 8}]
    recording = RecordingSpec("geuN_geuM_geuL.wav", phones, (), transcript_source="original_filename_only",
        label_source="none", metadata={"phone_slots": slots})
    spans = [(7, "transition", "vowel_transition", 1000, 1040), (7, "stable", "vowel_stable", 1040, 1400),
             (8, "consonant", "consonant", 1400, 1460), (8, "transition", "transition", 1460, 1490)]
    segments = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    applied = [{"vowel_phone_index": 1, "applied": True, "step_ms": 150.0},
               {"vowel_phone_index": 4, "applied": True, "step_ms": 690.0}]
    starts = [StateSegment("p1.t", 1, "transition", "vowel_transition", 0.0, 40.0, .5, 0.0, 40.0),
              StateSegment("p4.t", 4, "transition", "vowel_transition", 500.0, 540.0, .5, 500.0, 540.0)]
    adjusted = starts + list(segments)
    slots_out = list(applied)
    _codas_from_rhythm(recording, adjusted, slots_out, laterals=True, weak_nasals=False)
    # vowel lengths 150 and 190 ms -> median 170 ms after the ㄹ syllable's vowel start
    assert adjusted[-2].start_ms == 1170.0 and adjusted[-3].end_ms == 1170.0
    assert slots_out[-1]["applied"] and slots_out[-1]["coda"] == "l"
    lone = list(starts[:1]) + list(segments)
    out = [applied[0]]
    _codas_from_rhythm(recording, lone, out, laterals=True, weak_nasals=False)
    assert out[-1]["reason"] == "fewer_than_two_nasal_codas" and lone[-2].start_ms == 1400.0


def test_weak_nasal_coda_falls_back_to_the_files_rhythm_only_after_a_consonant_onset():
    from universal_auto_oto.alignment import _codas_from_rhythm
    phones = ("g", "eu", "n", "g", "eu", "m", "g", "eu", "ng", "eu", "ng")
    slots = [{"start": 0, "vowel": 1, "end": 2}, {"start": 3, "vowel": 4, "end": 5},
             {"start": 6, "vowel": 7, "end": 8}, {"start": 9, "vowel": 9, "end": 10}]
    recording = RecordingSpec("geuN_geuM_geuNG_euNG.wav", phones, (), transcript_source="original_filename_only",
        label_source="none", metadata={"phone_slots": slots})
    spans = [(1, "transition", "vowel_transition", 0, 40), (4, "transition", "vowel_transition", 500, 540),
             (7, "transition", "vowel_transition", 1000, 1040), (7, "stable", "vowel_stable", 1040, 1400),
             (8, "consonant", "consonant", 1400, 1460),
             (9, "transition", "vowel_transition", 2000, 2040), (9, "stable", "vowel_stable", 2040, 2400),
             (10, "consonant", "consonant", 2400, 2460)]
    adjusted = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    weak = lambda vowel: {"vowel_phone_index": vowel, "coda_phone_index": vowel + 1, "coda": "ng",
                          "decoded_start_ms": 0.0, "step_strength": 0.05, "step_ms": 1300.0,
                          "applied": False, "reason": "weak_band_balance_step"}
    slots_out = [{"vowel_phone_index": 1, "applied": True, "step_ms": 150.0},
                 {"vowel_phone_index": 4, "applied": True, "step_ms": 690.0}, weak(7), weak(9)]
    _codas_from_rhythm(recording, adjusted, slots_out, laterals=False, weak_nasals=True)
    # median vowel length 170 ms after the weak syllable's vowel start
    assert adjusted[4].start_ms == 1170.0 and adjusted[3].end_ms == 1170.0
    assert slots_out[2]["applied"] and slots_out[2]["rhythm_fallback"] == "applied" and slots_out[2]["weak_step_ms"] == 1300.0
    # vowel-initial syllable: no onset anchors the vowel start, so it stays skipped
    assert slots_out[3]["applied"] is False and slots_out[3]["rhythm_fallback"] == "no_consonant_onset"
    assert adjusted[7].start_ms == 2400.0


def test_stop_closure_starts_where_the_level_drops(tmp_path):
    import soundfile as sf
    from universal_auto_oto.alignment import _stop_closure_onsets
    rate = 16000
    t = np.arange(int(rate * 0.6)) / rate
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio[int(rate * 0.30):int(rate * 0.38)] = 0.0             # closure 300-380 ms, then the vowel again
    sf.write(tmp_path / "aga.wav", audio, rate)
    recording = RecordingSpec("aga.wav", ("a", "g", "a"), (), transcript_source="original_filename_only",
        label_source="none", metadata={})
    spans = [(0, "stable", "vowel_stable", 0, 340), (1, "closure", "closure", 340, 375),
             (1, "release", "burst", 375, 385), (2, "stable", "vowel_stable", 385, 600)]
    segments = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    adjusted, policy = _stop_closure_onsets(recording, tmp_path / "aga.wav", segments)
    onset = adjusted[1].start_ms
    assert 295.0 <= onset <= 310.0 and adjusted[0].end_ms == onset
    assert policy["applied_count"] == 1 and adjusted[2:] == segments[2:]


def test_phrase_final_vowel_ends_where_voicing_stops(tmp_path):
    import soundfile as sf
    from universal_auto_oto.alignment import _ending_voicing_offsets
    rate = 16000
    t = np.arange(int(rate * 0.8)) / rate
    rng = np.random.default_rng(1)
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    audio[int(rate * 0.45):] = 0.02 * rng.standard_normal(len(audio) - int(rate * 0.45))   # breath after 450 ms
    sf.write(tmp_path / "a_exh.wav", audio, rate)
    recording = RecordingSpec("a_exh.wav", ("a", "exh"), (), transcript_source="original_filename_only",
        label_source="none", metadata={})
    spans = [(0, "stable", "vowel_stable", 0, 650), (1, "breath", "breath", 650, 800)]
    segments = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    adjusted, policy = _ending_voicing_offsets(recording, tmp_path / "a_exh.wav", segments)
    assert 420.0 <= adjusted[0].end_ms <= 460.0 and adjusted[1].start_ms == adjusted[0].end_ms
    assert policy["applied_count"] == 1


def test_lateral_without_nasal_anchors_uses_the_period_share():
    from universal_auto_oto.alignment import _codas_from_rhythm
    phones = ("g", "eu", "l", "g", "a", "l", "g", "i", "l")
    slots = [{"start": 0, "vowel": 1, "end": 2}, {"start": 3, "vowel": 4, "end": 5}, {"start": 6, "vowel": 7, "end": 8}]
    recording = RecordingSpec("geuL_gaL_giL.wav", phones, (), transcript_source="original_filename_only",
        label_source="none", metadata={"phone_slots": slots})
    spans = [(0, "closure", "closure", 0, 40), (1, "transition", "vowel_transition", 40, 80), (1, "stable", "vowel_stable", 80, 400),
             (2, "consonant", "consonant", 400, 500), (3, "closure", "closure", 500, 540),
             (4, "transition", "vowel_transition", 540, 580), (4, "stable", "vowel_stable", 580, 900),
             (5, "consonant", "consonant", 900, 1000), (6, "closure", "closure", 1000, 1040)]
    adjusted = [StateSegment(f"p{i}.{state}", i, state, role, float(a), float(b), 0.5, float(a), float(b))
                for i, state, role, a, b in spans]
    out = []
    _codas_from_rhythm(recording, adjusted, out, laterals=True, weak_nasals=False)
    assert out[0]["reason"] == "fewer_than_two_nasal_codas"
    out = []
    _codas_from_rhythm(recording, adjusted, out, laterals=True, weak_nasals=False, period_ratio=0.25)
    # syllable period 500 ms -> vowel 125 ms after its start at 40 ms
    assert out[0]["applied"] and adjusted[3].start_ms == 165.0 and adjusted[2].end_ms == 165.0
