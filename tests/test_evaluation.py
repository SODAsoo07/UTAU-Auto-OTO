from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.evaluation import (
    canonicalize_voicebank_oto,
    evaluate_oto,
    read_voicebank_oto_timing_records,
)


def test_evaluate_oto_uses_absolute_coordinates(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    sf.write(wav_dir / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    reference = tmp_path / "reference.ini"
    generated = tmp_path / "generated.ini"
    reference.write_text("a.wav=a,100,200,-100,80,40\n", encoding="utf-8")
    generated.write_text("a.wav=a,110,210,-90,85,45\n", encoding="utf-8")
    result = evaluate_oto(generated, reference, wav_dir)
    assert result["matched_rows"] == 1
    assert result["metrics_ms"]["offset_abs"]["mae"] == 10.0
    assert result["metrics_ms"]["pre_abs"]["mae"] == 15.0
    assert result["metrics_ms"]["fixed_abs"]["mae"] == 20.0
    assert result["metrics_ms"]["cutoff_abs"]["mae"] == 0.0


def test_evaluate_oto_writes_sorted_per_row_csv(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    sf.write(wav_dir / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    sf.write(wav_dir / "b.wav", np.zeros(16000, dtype=np.float32), 16000)
    reference = tmp_path / "reference.ini"
    generated = tmp_path / "generated.ini"
    output_csv = tmp_path / "rows.csv"
    reference.write_text("a.wav=a,100,200,-100,80,40\nb.wav=a b,100,200,-100,80,40\n", encoding="utf-8")
    generated.write_text("a.wav=a,110,210,-90,85,45\nb.wav=a b,100,200,-100,380,40\n", encoding="utf-8")

    result = evaluate_oto(generated, reference, wav_dir, output_csv=output_csv)

    rows = output_csv.read_text(encoding="utf-8-sig").splitlines()
    assert result["output_csv"] == str(output_csv.resolve())
    assert rows[0].startswith("wav,alias,occurrence,alias_role,")
    assert rows[1].startswith("b.wav,a b,0,vc,")
    assert "true" in rows[1].lower()


def test_evaluate_oto_resolves_subdirectory_and_blank_values(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    pitch = wav_dir / "C4"
    pitch.mkdir(parents=True)
    sf.write(pitch / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    reference = tmp_path / "reference.ini"
    generated = tmp_path / "generated.ini"
    reference.write_text("C4/a.wav=a,100,200,-300,80,40\n", encoding="utf-8")
    generated.write_text("C4\\a.wav=a,100,200,-300,80,\n", encoding="utf-8")

    result = evaluate_oto(generated, reference, wav_dir)

    assert result["matched_rows"] == 1
    assert result["metrics_ms"]["overlap_abs"]["mae"] == 40.0


def test_evaluate_oto_resolves_cp932_to_cp949_mojibake_filename(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    mojibake_name = "あ.wav".encode("cp932").decode("cp949")
    sf.write(wav_dir / mojibake_name, np.zeros(16000, dtype=np.float32), 16000)
    reference = tmp_path / "reference.ini"
    generated = tmp_path / "generated.ini"
    reference.write_text("あ.wav=あ,100,200,-300,80,40\n", encoding="utf-8")
    generated.write_text(f"{mojibake_name}=あ,100,200,-300,80,40\n", encoding="utf-8")

    result = evaluate_oto(generated, reference, wav_dir)

    assert result["matched_rows"] == 1


def test_evaluate_oto_resolves_legacy_metadata_prefix_before_real_wav_name(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    pitch = wav_dir / "A4"
    other_pitch = wav_dir / "B3"
    pitch.mkdir(parents=True)
    other_pitch.mkdir(parents=True)
    sf.write(pitch / "ga_gi.wav", np.zeros(16000, dtype=np.float32), 16000)
    sf.write(other_pitch / "ga_gi.wav", np.zeros(8000, dtype=np.float32), 16000)
    reference = tmp_path / "reference.ini"
    generated = tmp_path / "generated.ini"
    reference.write_text(
        "A4,1150.604,229.999,-549.294,162.515,74.ga_gi.wav=gaA4,100,200,-300,80,40\n",
        encoding="utf-8",
    )
    generated.write_text("A4/ga_gi.wav=gaA4,110,200,-300,80,40\n", encoding="utf-8")

    result = evaluate_oto(generated, reference, wav_dir)

    assert result["matched_rows"] == 1
    assert result["metrics_ms"]["offset_abs"]["mae"] == 10.0


def test_read_voicebank_oto_uses_each_pitch_directory_for_duplicate_basenames(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    for pitch, offset in (("A4", 100), ("B3", 200)):
        directory = wav_dir / pitch
        directory.mkdir(parents=True)
        sf.write(directory / "ga_gi.wav", np.zeros(16000, dtype=np.float32), 16000)
        directory.joinpath("oto.ini").write_text(
            f"{pitch},1,2,3,4,5.ga_gi.wav=ga{pitch},{offset},200,-300,80,40\n",
            encoding="utf-8",
        )

    rows = read_voicebank_oto_timing_records(wav_dir)

    assert [(row.wav, row.alias, row.offset_abs) for row in rows] == [
        ("A4/ga_gi.wav", "gaA4", 100.0),
        ("B3/ga_gi.wav", "gaB3", 200.0),
    ]


def test_canonicalize_voicebank_oto_preserves_raw_positive_cutoff_parameters(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    pitch = wav_dir / "A4"
    pitch.mkdir(parents=True)
    sf.write(pitch / "ga_gi.wav", np.zeros(16000, dtype=np.float32), 16000)
    source_oto = pitch / "oto.ini"
    raw = "A4,1,2,3,4,5.ga_gi.wav=gaA4,100,200,300,80,40"
    source_oto.write_text(raw + "\n", encoding="utf-8")
    output = tmp_path / "canonical.ini"

    result = canonicalize_voicebank_oto(wav_dir, output)

    assert output.read_text(encoding="utf-8") == "A4/ga_gi.wav=gaA4,100,200,300,80,40\n"
    assert result["oto_row_count"] == 1
    assert result["source_files_modified"] is False
    assert source_oto.read_text(encoding="utf-8") == raw + "\n"
