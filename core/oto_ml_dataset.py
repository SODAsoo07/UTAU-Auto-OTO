"""
Dataset builders for OTO ML correction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.oto_ml_features import build_training_rows, write_dataset_csv


def build_oto_ml_dataset(
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    voicebank_id: str = "",
    format_type_override: str = "",
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    return build_training_rows(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
        voicebank_id=voicebank_id,
        format_type_override=format_type_override,
    )


def build_and_save_oto_ml_dataset(
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    out_csv: str,
    custom_phonemes_path: str = "",
    voicebank_id: str = "",
    append: bool = False,
    format_type_override: str = "",
) -> Dict[str, int]:
    rows, stats = build_oto_ml_dataset(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
        voicebank_id=voicebank_id,
        format_type_override=format_type_override,
    )
    if rows:
        write_dataset_csv(out_csv, rows, append=append)
    stats = dict(stats)
    stats["saved_rows"] = len(rows)
    stats["out_csv"] = os.path.abspath(out_csv)
    return stats
