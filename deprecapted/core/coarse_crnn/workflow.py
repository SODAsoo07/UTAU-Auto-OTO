from __future__ import annotations

from typing import Dict, Tuple

from core.pipeline_status import ALIGN_NOT_READY, make_runtime_report


_DEPRECATED_MESSAGE = (
    "Legacy CoarseCRNN alignment has been deprecated. "
    "Use MFA/sequence alignment for TextGrid alignment, or the boundary decoder OTO path for OTO generation."
)


def resolve_coarse_crnn_model_path(model_path: str = "") -> str:
    return str(model_path or "")


def check_coarse_crnn_ready(
    *,
    language: str = "korean",
    wav_folder: str = "",
    model_path: str = "",
    callback=None,
) -> Dict[str, object]:
    _ = callback
    return make_runtime_report(
        "align",
        ALIGN_NOT_READY,
        _DEPRECATED_MESSAGE,
        engine="coarse_crnn",
        language=str(language or "").strip().lower(),
        ready=False,
        model_path=resolve_coarse_crnn_model_path(model_path),
        wav_folder=str(wav_folder or ""),
        deprecated=True,
    )


def run_coarse_crnn_align(
    model_path: str,
    wav_folder: str,
    _dictionary_path: str,
    output_folder: str,
    *,
    language: str = "korean",
    format_hint: str = "",
    callback=None,
) -> Tuple[bool, str]:
    _ = (model_path, wav_folder, output_folder, language, format_hint, callback)
    return False, _DEPRECATED_MESSAGE


__all__ = ["check_coarse_crnn_ready", "resolve_coarse_crnn_model_path", "run_coarse_crnn_align"]
