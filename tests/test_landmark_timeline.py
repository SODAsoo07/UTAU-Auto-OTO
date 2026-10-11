from pathlib import Path

import pytest

from universal_auto_oto.domain import Slot, Timeline, landmark_timeline_from_legacy


def test_legacy_timeline_converts_to_schema_v2_without_gold_claim() -> None:
    legacy = Timeline(
        wav_path=Path("sample.wav"),
        wav_sha256="abc",
        sample_rate=44100,
        duration_ms=500.0,
        active_start_ms=20.0,
        active_end_ms=470.0,
        slots=(Slot(0, 20.0, 470.0, 80.0, 130.0, 0.7, 0.8),),
        confidence=0.75,
    )
    timeline = landmark_timeline_from_legacy(legacy, phone_ids=("a",))
    assert timeline.schema_version == 2
    assert timeline.phone_segments[0].phone_id == "a"
    assert timeline.landmark_time(0, "vowel_transition") == 80.0
    assert timeline.provenance["label_source"] == "acoustic_heuristic_v1_pseudo"
    assert timeline.to_dict()["landmarks"][0]["source"] == "legacy_slot_pseudo"


def test_legacy_timeline_rejects_phone_count_mismatch() -> None:
    legacy = Timeline(
        wav_path=Path("sample.wav"),
        wav_sha256="abc",
        sample_rate=44100,
        duration_ms=100.0,
        active_start_ms=0.0,
        active_end_ms=100.0,
        slots=(Slot(0, 0.0, 100.0, 30.0, 50.0, 0.5, 0.5),),
        confidence=0.5,
    )
    with pytest.raises(ValueError, match="slot 수"):
        landmark_timeline_from_legacy(legacy, phone_ids=("a", "i"))
