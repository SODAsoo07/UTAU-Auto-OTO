import json

import numpy as np
import soundfile as sf

from universal_auto_oto.domain import SourceOtoRow
from universal_auto_oto.project_runtime import generate_project
from universal_auto_oto.sonorant_identity import SonorantTemplates, cepstra_at, has_adjacent_sonorants, is_sonorant
from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project

RATE = 16000
FORMANTS = {"a": (750.0, 1200.0, 2600.0), "i": (300.0, 2300.0, 3000.0)}


def _vowel(name, duration_ms, f0=200.0):
    t = np.arange(int(RATE * duration_ms / 1000.0)) / RATE
    out = np.zeros_like(t)
    for harmonic in range(1, int(5000 / f0)):
        f = harmonic * f0
        amp = sum(1.0 / (1.0 + ((f - formant) / 90.0) ** 2) for formant in FORMANTS[name])
        out += amp * np.sin(2 * np.pi * f * t)
    return out / np.abs(out).max()


def _write_vowels(path, plan):
    """Write silence + vowels with 20 ms crossfades; returns the true vowel start times (ms)."""
    fade = int(RATE * 0.020)
    signal = np.zeros(int(RATE * 0.1))
    starts = []
    for name, duration in plan:
        piece = _vowel(name, duration + 20) * 0.2
        starts.append(len(signal) / RATE * 1000.0 - (20.0 if starts else 0.0))
        if starts[1:]:
            ramp = np.linspace(0, 1, fade)
            signal[-fade:] = signal[-fade:] * (1 - ramp) + piece[:fade] * ramp
            signal = np.concatenate([signal, piece[fade:]])
        else:
            signal = np.concatenate([signal, piece])
    signal = np.concatenate([signal, np.zeros(int(RATE * 0.1))])
    sf.write(path, signal, RATE)
    return starts


def _project(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    plans = {
        "a'i'a'i.wav": [("a", 200), ("i", 520), ("a", 240), ("i", 460)],
        "i'a'i'a.wav": [("i", 480), ("a", 220), ("i", 500), ("a", 260)],
        "a'a'i'i.wav": [("a", 300), ("a", 300), ("i", 300), ("i", 300)],
    }
    truth = {name: _write_vowels(bank / name, plan) for name, plan in plans.items()}
    rows = {}
    for name in plans:
        vowels = name[:-4].split("'")
        rows[name] = [SourceOtoRow(name, f"{a} {b}", index) for index, (a, b) in enumerate(zip(vowels, vowels[1:]))]
    build = build_routed_voicebank_project(
        bank=bank, wavs=[bank / name for name in plans], base_rows_by_wav=rows,
        format_hint="vcv", language_hint="korean", recursive=False, output_encoding="utf-8")
    return build.project, truth


def _vowel_starts(output):
    starts = {}
    for path in (output / "timelines").glob("*.json"):
        timeline = json.loads(path.read_text(encoding="utf-8"))
        starts[timeline["wav"]] = [segment["start_ms"] for segment in timeline["phone_segments"]]
    return starts


def test_factors_need_two_phones_and_stay_within_weight():
    templates = SonorantTemplates(
        means={"a": np.zeros(12), "i": np.ones(12)}, variances={"a": np.ones(12), "i": np.ones(12)},
        frame_counts={"a": 100, "i": 100})
    frames = np.stack([np.zeros(12), np.ones(12)])
    assert templates.factors(["a", "a"], frames) == {}
    factors = templates.factors(["a", "i", "k"], frames, weight=0.4)
    assert sorted(factors) == ["a", "i"]
    assert factors["a"][0] == 1.0 and factors["i"][1] == 1.0
    assert 0.6 <= factors["a"][1] < 1.0 and 0.6 <= factors["i"][0] < 1.0


def test_adjacent_sonorants_follow_inventory_features(tmp_path):
    project, _ = _project(tmp_path)
    assert is_sonorant(project.inventory["a"]) and is_sonorant(project.inventory["i"])
    assert has_adjacent_sonorants(("a", "i"), project.inventory)
    assert not has_adjacent_sonorants(("a",), project.inventory)


def test_templates_lower_a_vowel_score_on_the_other_vowel_frames(tmp_path):
    project, truth = _project(tmp_path)
    wav = project.wav_path(next(recording for recording in project.recordings if recording.wav == "a'i'a'i.wav"))
    starts = truth["a'i'a'i.wav"]
    ends = starts[1:] + [starts[-1] + 460.0]
    segments = list(zip(["a", "i", "a", "i"], starts, ends))
    templates = SonorantTemplates.build([(wav, segments)], project.inventory)
    times = np.arange(150.0, ends[-1] - 50.0, 10.0)
    factors = templates.factors(["a", "i"], cepstra_at(wav, times), weight=0.5)
    label = np.array([next(name for name, start, end in segments if start + 30 <= t < end - 30) if any(
        start + 30 <= t < end - 30 for _, start, end in segments) else "" for t in times])
    assert factors["a"][label == "a"].mean() > 0.95 and factors["a"][label == "i"].mean() < 0.6
    assert factors["i"][label == "i"].mean() > 0.95 and factors["i"][label == "a"].mean() < 0.6


def test_two_pass_identity_reports_and_keeps_boundaries_on_the_spectral_change(tmp_path):
    project, truth = _project(tmp_path)
    plain_out, identity_out = tmp_path / "plain", tmp_path / "identity"
    plain = generate_project(project, output_oto=plain_out / "oto.ini")
    report = generate_project(project, output_oto=identity_out / "oto.ini",
                              alignment_options={"sonorant_identity_weight": 0.5})
    assert "sonorant_identity" not in plain
    assert report["sonorant_identity"]["realigned_recordings"] == 3
    assert report["sonorant_identity"]["templates"]["phones"] == ["a", "i"]
    identity_starts = _vowel_starts(identity_out)
    for wav in ("a'i'a'i.wav", "i'a'i'a.wav"):
        error = np.abs(np.subtract(identity_starts[wav][1:], truth[wav][1:]))
        assert error.max() <= 40.0, (wav, identity_starts[wav], truth[wav])
    timeline = json.loads(next((identity_out / "timelines").glob("*.json")).read_text(encoding="utf-8"))
    assert timeline["provenance"]["sonorant_identity"]["phones"] == ["a", "i"]


def test_syllabic_nasal_of_its_own_may_be_held_like_a_vowel(tmp_path):
    from universal_auto_oto.alignment import SYLLABIC_SONORANT_MAX_MS, _syllabic_sonorant_holds
    from universal_auto_oto.sequence_compiler import compile_recording_states

    bank = tmp_path / "bank"
    bank.mkdir()
    name = "ba-ba-bba-ma-M.wav"
    sf.write(bank / name, np.zeros(RATE), RATE)
    build = build_routed_voicebank_project(
        bank=bank, wavs=[bank / name], base_rows_by_wav={}, format_hint="cvc", language_hint="korean",
        recursive=False, output_encoding="utf-8")
    project, recording = build.project, build.project.recordings[0]
    assert recording.phones == ("b", "a", "b", "a", "bb", "a", "m", "a", "m")
    compiled = compile_recording_states(recording, project.inventory)
    held, policy = _syllabic_sonorant_holds(project, recording, compiled)
    assert policy["phone_indices"] == [8]
    changed = [state for state in held if state.instance_id in policy["state_instance_ids"]]
    assert [(state.phone_index, state.acoustic_role, state.max_ms) for state in changed] == [
        (8, "periodic", SYLLABIC_SONORANT_MAX_MS)]
    # The onset m of "ma" and every other state keep their compiled durations.
    untouched = [(a, b) for a, b in zip(compiled, held) if a.instance_id not in policy["state_instance_ids"]]
    assert all(a == b for a, b in untouched)


def test_pause_token_state_sits_before_the_syllable_after_the_dash(tmp_path):
    from types import SimpleNamespace

    from universal_auto_oto.alignment import (
        PAUSE_TOKEN_OFF_SILENCE_START_PENALTY, _insert_pause_token_states, _state_spec)
    from universal_auto_oto.sequence_compiler import compile_recording_states

    bank = tmp_path / "bank"
    bank.mkdir()
    name = "_ang'hang'-'al'hal.wav"
    sf.write(bank / name, np.zeros(RATE), RATE)
    build = build_routed_voicebank_project(
        bank=bank, wavs=[bank / name], base_rows_by_wav={}, format_hint="cvvc", language_hint="korean",
        recursive=False, output_encoding="utf-8")
    project, recording = build.project, build.project.recordings[0]
    assert recording.phones == ("a", "ng", "h", "a", "ng", "a", "l", "h", "a", "l")
    compiled = compile_recording_states(recording, project.inventory)
    specs = tuple(_state_spec(state) for state in compiled)
    times = np.arange(0.0, 3000.0, 10.0)
    # speech 200-1400 ms, pause 1400-1800 ms, speech 1800-2800 ms
    active = ((times >= 200) & (times < 1400)) | ((times >= 1800) & (times < 2800))
    evidence = SimpleNamespace(frame_times_ms=times, tracks={"active": active.astype(float), "low_energy": (~active).astype(float)})
    scores = {state.instance_id: np.zeros(len(times)) for state in compiled}
    out, out_scores, positions, policy, start_scores = _insert_pause_token_states(recording, compiled, specs, scores, evidence)
    first_al = next(i for i, state in enumerate(compiled) if state.phone_index == 5)
    assert positions == [first_al] and len(out) == len(specs) + 1
    pause = out[first_al]
    assert pause.state_type == "silence" and out[first_al + 1] == specs[first_al]
    assert policy["before_phone_indices"] == [5]
    assert policy["interior_silences_ms"] == [[1400.0, 1790.0]]
    start = start_scores[pause.state_id]
    assert start[(times >= 1400) & (times < 1800)].max() == 0.0
    assert start[times < 1400].max() == -PAUSE_TOKEN_OFF_SILENCE_START_PENALTY
    np.testing.assert_allclose(out_scores[pause.state_id], 1.0 - active)


def test_speech_scope_also_covers_consonants_and_glides() -> None:
    from universal_auto_oto.domain import PhoneDefinition
    from universal_auto_oto.sonorant_identity import has_adjacent_sonorants, in_identity_scope

    inventory = {
        "a": PhoneDefinition(id="a", display_name="a", features={"syllabic": True, "vowel_like": True}, states=()),
        "jj": PhoneDefinition(id="jj", display_name="jj", features={"manner": "stop"}, states=()),
        "y": PhoneDefinition(id="y", display_name="y", features={"manner": "unknown"}, states=()),
        "sil": PhoneDefinition(id="sil", display_name="sil", features={"manner": "silence"}, states=()),
        "br": PhoneDefinition(id="br", display_name="br", features={"manner": "breath"}, states=()),
    }
    assert [p for p in inventory if in_identity_scope(inventory[p])] == ["a"]
    assert [p for p in inventory if in_identity_scope(inventory[p], "speech")] == ["a", "jj", "y"]
    assert not has_adjacent_sonorants(["jj", "a"], inventory)
    assert has_adjacent_sonorants(["jj", "a"], inventory, "speech")
    assert not has_adjacent_sonorants(["sil", "br"], inventory, "speech")
