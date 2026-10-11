from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.neural_evidence_dataset import (
    EVENT_TRACKS,
    NeuralEvidenceDatasetError,
    SUPPORTED_PUBLICDATA_LANGUAGES,
    build_publicdata_manifest,
    frame_targets,
    parse_htk_lab,
)


def _write_pair(root: Path, language_folder: str, stem: str, *, phone: str) -> None:
    wav_dir = root / language_folder / "corpus" / "wavs"
    lab_dir = root / language_folder / "corpus" / "labs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    sf.write(wav_dir / f"{stem}.wav", np.zeros(44100, dtype=np.float32), 44100)
    (lab_dir / f"{stem}.lab").write_text(
        f"0 1000000 sil\n1000000 5000000 {phone}\n5000000 10000000 a\n",
        encoding="utf-8",
    )


def test_publicdata_manifest_pairs_recursively_writes_hashes_and_audit(tmp_path: Path) -> None:
    _write_pair(tmp_path, "_한국어", "sample", phone="k")
    _write_pair(tmp_path, "_일본어", "sample_ja", phone="sh")
    result = build_publicdata_manifest(
        tmp_path,
        tmp_path / "manifest.csv",
        tmp_path / "audit.json",
        languages=("ko", "ja"),
        hash_files=True,
    )
    assert result["recording_count"] == 2
    with (tmp_path / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["language"] for row in rows} == {"ko", "ja"}
    assert all(row["wav_sha256"] and row["lab_sha256"] for row in rows)
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["summary"]["ambiguous_pair_count"] == 0


def test_publicdata_manifest_uses_csd_edited_segment_pairs_not_alternate_htk(tmp_path: Path) -> None:
    language_root = tmp_path / SUPPORTED_PUBLICDATA_LANGUAGES["ko"]
    wav_path = language_root / "CSD" / "CSD" / "korean" / "wav" / "sample.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.zeros(44100, dtype=np.float32), 44100)
    phone_lab = language_root / "CSD" / "htklab_of_CSD" / "phones" / "sample.lab"
    word_lab = language_root / "CSD" / "htklab_of_CSD" / "words" / "sample.lab"
    for lab_path, label in ((phone_lab, "a"), (word_lab, "word")):
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.write_text(f"0 10000000 {label}\n", encoding="utf-8")
    edited_wav = language_root / "CSD" / "CSD_editedlabels" / "CSD.ko" / "wavs" / "sample_seg000.wav"
    edited_lab = edited_wav.with_suffix(".lab")
    edited_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(edited_wav, np.zeros(44100, dtype=np.float32), 44100)
    edited_lab.write_text("0 10000000 a\n", encoding="utf-8")

    result = build_publicdata_manifest(
        tmp_path,
        tmp_path / "manifest.csv",
        tmp_path / "audit.json",
        languages=("ko",),
    )

    assert result["recording_count"] == 1
    row = next(csv.DictReader((tmp_path / "manifest.csv").open(encoding="utf-8", newline="")))
    assert Path(row["lab_path"]) == edited_lab.resolve()
    assert result["ambiguous_pair_count"] == 0
    assert result["excluded_source_count"] == 3


def test_frame_targets_use_soft_boundaries_and_mask_transition(tmp_path: Path) -> None:
    lab = tmp_path / "sample.lab"
    lab.write_text("0 2000000 sil\n2000000 5000000 k\n5000000 10000000 a\n", encoding="utf-8")
    intervals = parse_htk_lab(lab)
    targets, masks = frame_targets(np.arange(0.0, 1000.0, 5.0), intervals)
    assert set(targets) == set(EVENT_TRACKS)
    assert targets["boundary_likelihood"][40] > 0.9
    assert targets["vowel_onset_likelihood"][100] > 0.9
    assert not np.any(masks["transition_likelihood"])


def test_htk_lab_rejects_non_monotonic_intervals(tmp_path: Path) -> None:
    lab = tmp_path / "bad.lab"
    lab.write_text("0 2000000 a\n1000000 3000000 b\n", encoding="utf-8")
    try:
        parse_htk_lab(lab)
    except NeuralEvidenceDatasetError as error:
        assert "non-monotonic" in str(error)
    else:
        raise AssertionError("non-monotonic HTK label was accepted")
