from __future__ import annotations

import os
from typing import Any

from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder
from core.coarse_crnn.deprecated.direct_param import oto_predictor_generator as _legacy

DEFAULT_OTO_CRNN_MODEL_NAME = _legacy.DEFAULT_OTO_CRNN_MODEL_NAME


def generate_oto_with_crnn_predictor(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    language: str,
    format_type: str = "",
    model_path: str = "",
    device: str = "auto",
    alias_suffix: str = "",
    callback=None,
    special_aliases=None,
    engine: str = "",
):
    engine_code = _normalize_engine(engine or os.environ.get("UTOA_OTO_CRNN_ENGINE", ""))
    if engine_code == "boundary_decoder":
        return generate_oto_with_boundary_decoder(
            wav_dir=wav_dir,
            out_path=out_path,
            source_oto_path=source_oto_path,
            language=language,
            format_type=format_type,
            model_path=model_path,
            device=device,
            alias_suffix=alias_suffix,
            callback=callback,
            special_aliases=special_aliases,
        )
    _forward_legacy_monkeypatches()
    return _legacy.generate_oto_with_crnn_predictor(
        wav_dir=wav_dir,
        out_path=out_path,
        source_oto_path=source_oto_path,
        language=language,
        format_type=format_type,
        model_path=model_path,
        device=device,
        alias_suffix=alias_suffix,
        callback=callback,
        special_aliases=special_aliases,
    )


def _normalize_engine(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"boundary", "boundary_decoder", "boundary-scorer", "scorer", "new"}:
        return "boundary_decoder"
    return "legacy_direct"


def _forward_legacy_monkeypatches() -> None:
    # Keep pytest monkeypatch compatibility for tests that patch symbols on
    # core.coarse_crnn.oto_predictor_generator directly.
    for name in (
        "load_oto_checkpoint",
        "predict_oto_with_model",
        "resolve_oto_crnn_model_path",
    ):
        if name in globals():
            try:
                setattr(_legacy, name, globals()[name])
            except Exception:
                pass


def __getattr__(name: str) -> Any:
    return getattr(_legacy, name)


__all__ = list(getattr(_legacy, "__all__", []))
if "generate_oto_with_crnn_predictor" not in __all__:
    __all__.append("generate_oto_with_crnn_predictor")
if "DEFAULT_OTO_CRNN_MODEL_NAME" not in __all__:
    __all__.append("DEFAULT_OTO_CRNN_MODEL_NAME")
