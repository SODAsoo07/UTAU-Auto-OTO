from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

from scripts.dev.visualize_hsmm_timeline import load_timeline_record, write_timeline_review_html


def _write_silence_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<" + ("h" * 3200), *([0] * 3200)))


def test_hsmm_timeline_visualizer_renders_state_spans_and_events(tmp_path):
    wav_path = tmp_path / "ka.wav"
    _write_silence_wav(wav_path)
    timeline_path = tmp_path / "sample.timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "wav": wav_path.name,
                "wav_path": str(wav_path),
                "expected_phones": ["k", "a"],
                "selected_events": [],
                "adapted_rows": [],
                "hsmm": {
                    "ok": True,
                    "states": [
                        {
                            "state_id": "slot0.onset",
                            "state_type": "consonant",
                            "start_ms": 20.0,
                            "end_ms": 80.0,
                        },
                        {
                            "state_id": "slot0.vowel",
                            "state_type": "vowel",
                            "start_ms": 80.0,
                            "end_ms": 180.0,
                        },
                    ],
                    "events": [
                        {
                            "label": "cv_boundary",
                            "selected_time_ms": 80.0,
                            "score": 0.9,
                            "expected_phone": "a",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_timeline_record(timeline_path, wav="ka")
    out_path = tmp_path / "hsmm.html"
    write_timeline_review_html(record, out_path)
    output = out_path.read_text(encoding="utf-8")

    assert "HSMM selected state spans" in output
    assert "slot0.onset [20-80ms]" in output
    assert "slot0.vowel [80-180ms]" in output
    assert "cv_boundary a 0.90" in output
    assert output.count('data-hsmm-boundary="80.000"') == 2
