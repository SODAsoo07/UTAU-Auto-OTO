from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingCandidate:
    language: str
    format_type: str
    root: str
    folder: str
    manual_oto: str = ""
    auto_oto: str = ""
    tg_dir: str = ""
    wav_dir: str = ""
    custom_phonemes: str = ""
    voicebank_id: str = ""
    status: str = "pending"
    reason: str = ""
    saved_rows: int = 0
    matched_rows: int = 0
    skipped_rows: int = 0
    out_csv: str = ""

