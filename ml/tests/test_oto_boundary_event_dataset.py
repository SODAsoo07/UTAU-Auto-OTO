from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path


def _write_silence_wav(path: Path, *, duration_ms: float = 1000.0, rate: int = 16000) -> None:
    frames = int(round(rate * float(duration_ms) / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_build_oto_boundary_event_rows_marks_vc_sonorant_transition(tmp_path):
    from scripts.evaluate.build_oto_boundary_event_dataset import build_boundary_event_rows, split_rows

    _write_silence_wav(tmp_path / "sample.wav", duration_ms=1000.0)
    _write_silence_wav(tmp_path / "other.wav", duration_ms=900.0)
    gold = tmp_path / "oto.ini"
    gold.write_text(
        "\n".join(
            [
                "sample.wav=- k,100,120,-400,80,30",
                "sample.wav=a r,350,180,-300,110,40",
                "sample.wav=a a,550,180,-200,100,50",
                "other.wav=m a,120,160,-300,95,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, summary = build_boundary_event_rows(
        [
            {
                "case_id": "toy_gold",
                "voicebank_id": "toy_vb",
                "language": "japanese",
                "format_type": "cvvc",
                "gold": str(gold),
                "wav_dir": str(tmp_path),
                "label_source": "manual_oto_gold",
            }
        ],
        cutoff_end_mode="utau",
        relative_to=str(tmp_path),
    )

    assert summary["rows_total"] == 2
    assert summary["event_counts"]["consonant_onset"] >= 2
    sample = next(row for row in rows if row["wav_name"] == "sample.wav")
    assert sample["wav_path"] == "sample.wav"
    assert sample["label_source"] == "manual_oto_gold"
    assert sample["trainable"] is True

    vc_events = [
        event
        for event in sample["events"]
        if event["alias"] == "a r" and event["source_anchor"] == "offset_plus_preutterance"
    ]
    assert {event["label"] for event in vc_events} >= {"vowel_end", "consonant_onset"}
    consonant_onset = next(event for event in vc_events if event["label"] == "consonant_onset")
    assert consonant_onset["phone"] == "r"
    assert consonant_onset["time_ms"] == 460.0

    train, heldout, split_summary = split_rows(rows, heldout_ratio=0.5, seed=1)
    assert len(train) + len(heldout) == 2
    assert split_summary["groups_total"] == 2
    assert split_summary["excluded_non_trainable_rows"] == 0
    assert {row["group_id"] for row in train}.isdisjoint({row["group_id"] for row in heldout})

    rows_with_missing = [*rows, {"row_id": "missing", "group_id": "toy_gold::missing.wav", "trainable": False}]
    train2, heldout2, split_summary2 = split_rows(rows_with_missing, heldout_ratio=0.5, seed=1)
    assert len(train2) + len(heldout2) == 2
    assert split_summary2["excluded_non_trainable_rows"] == 1


def test_build_oto_boundary_event_dataset_cli_writes_manifests(tmp_path):
    from core.phoneme_boundary.targets import (
        build_boundary_target_map,
        events_from_manifest_row,
        read_boundary_manifest,
        resolve_wav_path,
    )

    _write_silence_wav(tmp_path / "sample.wav", duration_ms=1000.0)
    gold = tmp_path / "oto.ini"
    out_dir = tmp_path / "out"
    gold.write_text(
        "\n".join(
            [
                "sample.wav=- a,100,120,-400,80,30",
                "sample.wav=a n,350,180,-300,110,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate/build_oto_boundary_event_dataset.py",
            "--case-id",
            "toy_cli",
            "--language",
            "japanese",
            "--format-type",
            "cvvc",
            "--gold",
            str(gold),
            "--wav-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--heldout-ratio",
            "0.5",
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out_dir / "boundary_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert payload["rows_total"] == 1
    assert summary["rows_total"] == 1
    assert summary["split"]["groups_total"] == 1
    assert rows[0]["label_policy"]["generated_oto_disallowed"] is True
    assert any(event["label"] == "consonant_onset" and event["phone"] == "n" for event in rows[0]["events"])

    manifest_rows = read_boundary_manifest(str(out_dir / "boundary_events.jsonl"))
    assert Path(resolve_wav_path(manifest_rows[0], manifest_dir=str(out_dir))).is_file()
    events = events_from_manifest_row(manifest_rows[0])
    assert any(event.label == "consonant_onset" and event.phone == "n" for event in events)
    _times, target = build_boundary_target_map(events, duration_ms=1000.0, hop_ms=10.0, frame_count=120)
    assert target.shape[0] == 120
    assert float(target.max()) > 0.0


def test_build_oto_boundary_event_dataset_rejects_generated_labels(tmp_path):
    from scripts.evaluate.build_oto_boundary_event_dataset import build_boundary_event_rows

    _write_silence_wav(tmp_path / "sample.wav", duration_ms=1000.0)
    gold = tmp_path / "oto.ini"
    gold.write_text("sample.wav=a n,350,180,-300,110,40\n", encoding="utf-8")

    case = {
        "case_id": "bad_case",
        "language": "japanese",
        "format_type": "cvvc",
        "gold": str(gold),
        "wav_dir": str(tmp_path),
        "label_source": "generated",
        "trusted": False,
    }

    try:
        build_boundary_event_rows([case])
    except ValueError as exc:
        assert "rejected" in str(exc)
    else:
        raise AssertionError("generated labels must be rejected unless --allow-untrusted is used")
