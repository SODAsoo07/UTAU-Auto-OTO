from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.cli import validate_oto
from universal_auto_oto.generator import GenerationOptions, generate_voicebank, inspect_voicebank


def _write_chain(path: Path, *, slots: int = 4, sample_rate: int = 16000) -> None:
    parts: list[np.ndarray] = [np.zeros(int(sample_rate * 0.12), dtype=np.float32)]
    for index in range(slots):
        consonant = np.random.default_rng(index).normal(0.0, 0.018, int(sample_rate * 0.07)).astype(np.float32)
        length = int(sample_rate * 0.28)
        times = np.arange(length, dtype=np.float32) / sample_rate
        vowel = (0.18 * np.sin(2 * np.pi * (180 + index * 15) * times)).astype(np.float32)
        parts.extend((consonant, vowel, np.zeros(int(sample_rate * 0.05), dtype=np.float32)))
    parts.append(np.zeros(int(sample_rate * 0.12), dtype=np.float32))
    sf.write(path, np.concatenate(parts), sample_rate, subtype="PCM_16")


def test_end_to_end_generation_uses_aliases_but_not_base_timing(tmp_path: Path):
    bank = tmp_path / "bank"
    output = tmp_path / "output"
    bank.mkdir()
    wav = bank / "ga_gi_gu_ge.wav"
    _write_chain(wav)
    base_oto = bank / "boto.ini"
    base_oto.write_text(
        "\n".join(
            [
                "ga'gi'gu'ge.wav=ga,0,0,0,0,0",
                "ga'gi'gu'ge.wav=a g,0,0,0,0,0",
                "ga'gi'gu'ge.wav=gi,0,0,0,0,0",
                "ga'gi'gu'ge.wav=i g,0,0,0,0,0",
                "ga'gi'gu'ge.wav=gu,0,0,0,0,0",
                "ga'gi'gu'ge.wav=u g,0,0,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inspected = inspect_voicebank(bank, base_oto)
    assert inspected["mapped_wavs"] == 1
    report = generate_voicebank(GenerationOptions(bank=bank, output=output, base_oto=base_oto))
    assert report["generated_rows"] == 6
    assert report["provenance"]["base_oto_timing_reused"] is False
    generated = (output / "oto.ini").read_text(encoding="utf-8")
    assert "ga_gi_gu_ge.wav=ga,0,0,0,0,0" not in generated
    validation = validate_oto(output / "oto.ini", bank)
    assert validation["valid"] is True, validation
    persisted = json.loads((output / "generation_report.json").read_text(encoding="utf-8"))
    assert persisted["generated_rows"] == 6


def test_output_inside_bank_is_rejected(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_chain(bank / "a.wav", slots=1)
    with pytest.raises(ValueError, match="직접 저장 허용"):
        generate_voicebank(GenerationOptions(bank=bank, output=bank, force=True))

    try:
        generate_voicebank(GenerationOptions(bank=bank, output=bank / "generated"))
    except ValueError as error:
        assert "원본 보호" in str(error)
    else:
        raise AssertionError("bank 내부 output이 허용됨")

    with pytest.raises(ValueError, match="하위 폴더"):
        generate_voicebank(
            GenerationOptions(
                bank=bank,
                output=bank / "generated",
                allow_in_place_output=True,
                force=True,
            )
        )
