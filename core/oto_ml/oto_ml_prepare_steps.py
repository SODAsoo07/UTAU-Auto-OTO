from __future__ import annotations

import os
from typing import List

from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.ja_oto_generator import generate_ja_oto
from core.lab_generator import generate_dictionary, generate_labs
from core.oto_generator import generate_oto
from core.oto_ml_prepare_types import PreparedAutoPair


def _prepare_lab_and_dict(item: PreparedAutoPair, logs: List[str]) -> None:
    if item.language == "japanese":
        generate_ja_labs(item.work_dir, callback=logs.append)
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        generate_ja_dictionary(item.work_dir, item.dict_path, callback=logs.append)
    else:
        # For MFA alignment stability on Windows, keep Korean lab tokens ASCII-safe.
        # Hangul lab tokens can trigger graph-compilation failures on some banks.
        generate_labs(item.work_dir, convert_to_hangul=False, callback=logs.append)
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        generate_dictionary(item.work_dir, item.dict_path, callback=logs.append)


def _generate_auto_oto(item: PreparedAutoPair, logs: List[str]) -> None:
    item.auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
    # Training data preparation should stay model-agnostic.
    # Disable all ML/coupled refinements while generating auto oto pairs.
    ml_disabled_kwargs = {
        "enable_ml_correction": False,
        "ml_policy": "off",
    }
    if item.language == "japanese":
        generate_ja_oto(
            tg_folder=item.tg_dir,
            tpl_path=item.manual_oto,
            out_path=item.auto_oto,
            fallback_format=item.format_type,
            auto_format=item.format_type,
            callback=logs.append,
            **ml_disabled_kwargs,
        )
    else:
        generate_oto(
            tg_folder=item.tg_dir,
            tpl_path=item.manual_oto,
            out_path=item.auto_oto,
            fallback_format=item.format_type,
            auto_format=item.format_type,
            callback=logs.append,
            **ml_disabled_kwargs,
        )


__all__ = ["_generate_auto_oto", "_prepare_lab_and_dict"]
