from __future__ import annotations

import math
import json
import wave
import struct

import numpy as np

from core.phoneme_boundary.heuristics import compute_acoustic_boundary_scores
from core.phoneme_boundary.decoder import decode_expected_boundaries, expected_events_from_plan
from core.phoneme_boundary.manifest import (
    BoundaryManifestBuildConfig,
    build_boundary_manifest,
    load_alias_rows,
    load_alias_rows_from_oto,
    load_textgrid_phone_intervals,
    write_manifest_jsonl,
)
from core.phoneme_boundary.plan import build_phoneme_plan
from core.phoneme_boundary.targets import build_boundary_target_map, build_phone_aux_target_map, events_from_manifest_row
from core.phoneme_boundary.types import BOUNDARY_LABELS, BoundaryEvent, BoundaryFrameScores, PHONE_STATE_LABELS


def test_boundary_targets_from_explicit_events():
    events = [
        BoundaryEvent(label="vowel_onset", time_ms=120.0, confidence=1.0),
        BoundaryEvent(label="vowel_nucleus", time_ms=180.0, confidence=0.8),
    ]
    _times, target = build_boundary_target_map(events, duration_ms=300.0, hop_ms=10.0, frame_count=40)
    assert target.shape == (40, len(BOUNDARY_LABELS))
    labels = {label: idx for idx, label in enumerate(BOUNDARY_LABELS)}
    assert float(target[:, labels["vowel_onset"]].max()) > 0.9
    assert 0.75 < float(target[:, labels["vowel_nucleus"]].max()) <= 0.800001


def test_manifest_phone_intervals_generate_cv_and_nucleus_events():
    row = {
        "language": "korean",
        "phones": [
            {"phone": "g", "start_ms": 50.0, "end_ms": 100.0},
            {"phone": "a", "start_ms": 100.0, "end_ms": 260.0, "nucleus_ms": 180.0},
        ],
    }
    events = events_from_manifest_row(row)
    pairs = {(event.label, event.phone, round(event.time_ms, 1)) for event in events}
    assert ("consonant_onset", "g", 50.0) in pairs
    assert ("vowel_onset", "a", 100.0) in pairs
    assert ("vowel_nucleus", "a", 180.0) in pairs
    aux = build_phone_aux_target_map(row, frame_count=40, hop_ms=10.0)
    assert PHONE_STATE_LABELS[int(aux.state_target[6])] == "consonant"
    assert PHONE_STATE_LABELS[int(aux.state_target[12])] == "vowel"
    assert int(aux.consonant_mask.sum()) > 0
    assert int(aux.vowel_mask.sum()) > 0


def test_plan_decoder_uses_alias_order_without_oto_params():
    plan = build_phoneme_plan(wav_path="ga_gi.wav", aliases=["ga", "gi"], language="korean", duration_ms=400.0)
    expected = expected_events_from_plan(plan)
    assert any(event.label == "vowel_onset" and event.phone == "a" for event in expected)
    assert any(event.label == "vowel_nucleus" and event.phone == "i" for event in expected)

    times = [float(i * 10) for i in range(41)]
    scores = {label: [0.0 for _ in times] for label in BOUNDARY_LABELS}
    label_to_idx = {label: idx for idx, label in enumerate(BOUNDARY_LABELS)}
    del label_to_idx
    for label, center in {
        "consonant_onset": 40.0,
        "vowel_onset": 90.0,
        "vowel_nucleus": 150.0,
        "phone_end": 220.0,
    }.items():
        for idx, time_ms in enumerate(times):
            scores[label][idx] = max(scores[label][idx], math.exp(-0.5 * ((time_ms - center) / 12.0) ** 2))
    score_map = BoundaryFrameScores(wav_path="ga_gi.wav", times_ms=times, scores=scores)
    decoded = decode_expected_boundaries(score_map, plan, min_score=0.05)
    assert decoded
    assert all(decoded[idx].selected_time_ms <= decoded[idx + 1].selected_time_ms for idx in range(len(decoded) - 1))


def test_model_forward_shape_when_torch_available():
    torch = __import__("torch")
    from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig, build_phoneme_boundary_detector

    cfg = PhonemeBoundaryDetectorConfig(n_mels=8, hidden=4, conv_channels=4, labels=("vowel_onset", "vowel_nucleus"))
    model = build_phoneme_boundary_detector(cfg)
    x = torch.from_numpy(np.zeros((2, 16, 8), dtype=np.float32))
    out = model(x)
    assert tuple(out["boundary_logits"].shape) == (2, 16, 2)
    assert tuple(out["quality_logits"].shape) == (2, 16)
    assert tuple(out["phone_state_logits"].shape) == (2, 16, 3)
    assert tuple(out["consonant_logits"].shape[:2]) == (2, 16)
    assert tuple(out["vowel_logits"].shape[:2]) == (2, 16)


def test_acoustic_heuristics_emit_boundary_tracks():
    sr = 16000
    silence = np.zeros((sr // 10,), dtype=np.float32)
    tone_t = np.arange(sr // 5, dtype=np.float32) / float(sr)
    tone = 0.5 * np.sin(2.0 * np.pi * 220.0 * tone_t).astype(np.float32)
    samples = np.concatenate([silence, tone])
    scores = compute_acoustic_boundary_scores(samples, sr, frame_ms=25.0, hop_ms=10.0)
    assert set(BOUNDARY_LABELS).issubset(scores.keys())
    assert max(scores["vowel_nucleus"]) > 0.2
    assert max(scores["vowel_onset"]) > 0.2


def test_manifest_builder_merges_alias_order_and_sidecar(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "001_ga.wav")
    _write_tone_wav(wav_dir / "002_gi.wav")
    aliases = tmp_path / "aliases.txt"
    aliases.write_text("g a|a g\ng i|i g\n", encoding="utf-8")
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "wav_name": "001_ga.wav",
                        "phones": [
                            {"phone": "g", "start_ms": 40.0, "end_ms": 90.0},
                            {"phone": "a", "start_ms": 90.0, "end_ms": 220.0, "nucleus_ms": 150.0},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "wav_name": "002_gi.wav",
                        "events": [
                            {"label": "vowel_onset", "time_ms": 88.0, "phone": "i"},
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=str(wav_dir),
            aliases_path=str(aliases),
            sidecar_path=str(sidecar),
            out_path=str(out),
            language="korean",
        )
    )
    assert result.summary["rows"] == 2
    assert result.summary["trainable_rows"] == 2
    assert result.rows[0]["aliases"] == ["g a", "a g"]
    assert result.rows[0]["trainable"] is True
    assert result.rows[0]["expected_phones"][0]["phone"] == "g"
    assert "phones" in result.rows[0]
    assert "events" in result.rows[1]
    assert write_manifest_jsonl(str(out), result.rows) == 2


def test_manifest_alias_loader_preserves_space_aliases(tmp_path):
    aliases = tmp_path / "aliases.txt"
    aliases.write_text("a g|g a\n[\"e n\", \"n a\"]\n", encoding="utf-8")
    rows = load_alias_rows(str(aliases))
    assert rows == [["a g", "g a"], ["e n", "n a"]]


def test_manifest_builder_accepts_frame_phone_sidecar(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "001_ga.wav")
    aliases = tmp_path / "aliases.txt"
    aliases.write_text("g a\n", encoding="utf-8")
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "wav_name": "001_ga.wav",
                "hop_ms": 10.0,
                "phone": ["", "", "g", "g", "a", "a", "a", ""],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=str(wav_dir),
            aliases_path=str(aliases),
            sidecar_path=str(sidecar),
            out_path=str(out),
            language="korean",
            require_targets=True,
        )
    )
    assert result.summary["trainable_rows"] == 1
    phones = result.rows[0]["phones"]
    assert [item["phone"] for item in phones] == ["sil", "g", "a", "sil"]
    assert phones[1]["start_ms"] == 20.0
    assert phones[2]["end_ms"] == 70.0


def test_manifest_builder_uses_source_oto_aliases_and_timing_rules(tmp_path):
    wav_dir = tmp_path / "vb"
    wav_dir.mkdir()
    wav_path = wav_dir / "001_ga.wav"
    _write_tone_wav(wav_path)
    (wav_dir / "oto.ini").write_text(
        "001_ga.wav=g a,10,20,-60,15,5\n001_ga.wav=a g,40,50,-60,45,15\n",
        encoding="utf-8",
    )
    (wav_dir / "001_ga.TextGrid").write_text(
        """File type = \"ooTextFile\"
Object class = \"TextGrid\"

item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"phones\"
        intervals: size = 3
        intervals [1]:
            xmin = 0.00
            xmax = 0.04
            text = \"\"
        intervals [2]:
            xmin = 0.04
            xmax = 0.08
            text = \"g\"
        intervals [3]:
            xmin = 0.08
            xmax = 0.14
            text = \"a\"
""",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=str(wav_dir),
            source_oto_path=str(wav_dir / "oto.ini"),
            out_path=str(out),
            language="korean",
            require_targets=True,
        )
    )
    assert result.summary["alias_source"] == "source_oto_alias_only"
    assert result.summary["source_oto_timing_rows"] == 1
    assert result.summary["trainable_rows"] == 1
    assert result.rows[0]["aliases"] == ["g a", "a g"]
    events = result.rows[0]["events"]
    labels = {item["label"] for item in events}
    assert "vowel_onset" in labels
    assert "vowel_nucleus" in labels
    assert "consonant_onset" in labels
    assert "phones" not in result.rows[0]


def test_manifest_builder_does_not_use_textgrid_labels_without_sidecar(tmp_path):
    wav_dir = tmp_path / "vb"
    wav_dir.mkdir()
    wav_path = wav_dir / "001_ga.wav"
    _write_tone_wav(wav_path)
    aliases = tmp_path / "aliases.txt"
    aliases.write_text("g a\n", encoding="utf-8")
    (wav_dir / "001_ga.TextGrid").write_text(
        """File type = \"ooTextFile\"
Object class = \"TextGrid\"
item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"phones\"
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = \"a\"
""",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=str(wav_dir),
            aliases_path=str(aliases),
            out_path=str(out),
            language="korean",
            require_targets=True,
        )
    )
    assert result.summary["missing_target_rows"] == 1
    assert result.summary["trainable_rows"] == 0
    assert len(result.rows) == 0


def test_source_oto_alias_loader_ignores_timing_values(tmp_path):
    wav_dir = tmp_path / "vb"
    wav_dir.mkdir()
    wav_path = wav_dir / "001.wav"
    _write_tone_wav(wav_path)
    oto = wav_dir / "oto.ini"
    oto.write_text("001.wav=a,100,200,-300,150,50\n001.wav=i,999,999,-999,999,999\n", encoding="utf-8")
    rows = load_alias_rows_from_oto(str(oto), [str(wav_path)])
    assert rows == [["a", "i"]]


def test_textgrid_phone_interval_loader(tmp_path):
    tg = tmp_path / "sample.TextGrid"
    tg.write_text(
        """File type = \"ooTextFile\"
Object class = \"TextGrid\"
item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"phones\"
        intervals: size = 2
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = \"\"
        intervals [2]:
            xmin = 0.1
            xmax = 0.3
            text = \"a\"
""",
        encoding="utf-8",
    )
    intervals = load_textgrid_phone_intervals(str(tg))
    assert intervals[0]["phone"] == "sil"
    assert intervals[1]["phone"] == "a"
    assert intervals[1]["start_ms"] == 100.0


def _write_tone_wav(path, *, sample_rate: int = 16000) -> None:
    samples = []
    for i in range(int(sample_rate * 0.12)):
        value = 0.35 * math.sin(2.0 * math.pi * 220.0 * (i / sample_rate))
        samples.append(max(-1.0, min(1.0, value)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for value in samples:
            wf.writeframes(struct.pack("<h", int(value * 32767)))
