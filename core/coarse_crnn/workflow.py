from __future__ import annotations

import os
from typing import Dict, Tuple

from core.coarse_crnn.inference import align_wav
from core.coarse_crnn.lang import normalize_language
from core.pipeline_status import ALIGN_MODEL_MISSING, ALIGN_NOT_READY, ALIGN_OUTPUT_EMPTY, ALIGN_RUN_FAILED, OK, make_runtime_report


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def resolve_coarse_crnn_model_path(model_path: str = "") -> str:
    explicit = str(model_path or "").strip() or str(os.environ.get("UTOA_COARSE_CRNN_MODEL", "") or "").strip()
    candidates = []
    if explicit:
        candidates.append(explicit)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.extend(
        [
            os.path.join(root, "models", "coarse_crnn", "coarse_crnn.pt"),
            os.path.join(root, "ML_models", "coarse_crnn", "coarse_crnn.pt"),
            os.path.join(root, "ml_workspace", "models", "coarse_crnn", "coarse_crnn.pt"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0]) if candidates else ""


def _list_top_level_wavs(wav_folder: str) -> list[str]:
    root = os.path.abspath(str(wav_folder or "").strip())
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, name) for name in sorted(os.listdir(root)) if str(name).lower().endswith(".wav")]


def check_coarse_crnn_ready(
    *,
    language: str = "korean",
    wav_folder: str = "",
    model_path: str = "",
    callback=None,
) -> Dict[str, object]:
    lang = normalize_language(language)
    if lang not in {"korean", "japanese"}:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            "coarse_crnn currently supports Korean/Japanese workflows only.",
            engine="coarse_crnn",
            language=lang,
            ready=False,
        )
    if wav_folder and not os.path.isdir(wav_folder):
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"WAV folder not found: {wav_folder}",
            engine="coarse_crnn",
            language=lang,
            ready=False,
        )
    wavs = _list_top_level_wavs(wav_folder) if wav_folder else []
    if wav_folder and not wavs:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"No WAV files found: {wav_folder}",
            engine="coarse_crnn",
            language=lang,
            ready=False,
        )
    resolved_model = resolve_coarse_crnn_model_path(model_path)
    if not resolved_model or not os.path.isfile(resolved_model):
        return make_runtime_report(
            "align",
            ALIGN_MODEL_MISSING,
            f"coarse_crnn model not found: {resolved_model or model_path}",
            engine="coarse_crnn",
            language=lang,
            ready=False,
            model_path=resolved_model,
        )
    try:
        __import__("torch")
    except Exception as exc:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"PyTorch import failed: {exc}",
            engine="coarse_crnn",
            language=lang,
            ready=False,
            model_path=resolved_model,
        )
    _emit(callback, f"[CoarseCRNN] ready: wav_files={len(wavs)}, model={resolved_model}")
    return make_runtime_report(
        "align",
        OK,
        "coarse_crnn alignment ready",
        engine="coarse_crnn",
        language=lang,
        ready=True,
        wav_count=int(len(wavs)),
        model_path=resolved_model,
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
    lang = normalize_language(language)
    ready = check_coarse_crnn_ready(language=lang, wav_folder=wav_folder, model_path=model_path, callback=callback)
    if str(ready.get("code", "")).upper() != OK:
        return False, str(ready.get("message", "coarse_crnn not ready") or "coarse_crnn not ready")
    resolved_model = str(ready.get("model_path", "") or resolve_coarse_crnn_model_path(model_path))
    wavs = _list_top_level_wavs(wav_folder)
    os.makedirs(output_folder, exist_ok=True)

    ok_count = 0
    errors: list[str] = []
    confidence_values: list[float] = []
    for wav_path in wavs:
        result = align_wav(
            wav_path=wav_path,
            output_dir=output_folder,
            model_path=resolved_model,
            language=lang,
            transcript="",
            device=str(os.environ.get("UTOA_COARSE_CRNN_DEVICE", "auto") or "auto"),
            export_debug=str(os.environ.get("UTOA_COARSE_CRNN_DEBUG", "1") or "1").lower() not in {"0", "false", "no"},
        )
        if result.success and os.path.isfile(result.textgrid_path):
            ok_count += 1
            confidence_values.append(float(result.overall_confidence))
            _emit(callback, f"[CoarseCRNN] aligned file={os.path.basename(wav_path)} confidence={result.overall_confidence:.3f}")
        else:
            err = f"{os.path.basename(wav_path)}: {result.error or 'TextGrid output missing'}"
            errors.append(err)
            _emit(callback, f"[CoarseCRNN] failed: {err}")

    if ok_count <= 0:
        return False, f"CoarseCRNN alignment failed: {errors[0] if errors else 'no output'}"
    if not any(str(name).lower().endswith(".textgrid") for name in os.listdir(output_folder)):
        return False, "TextGrid output missing after CoarseCRNN alignment."
    mean_conf = sum(confidence_values) / float(len(confidence_values)) if confidence_values else 0.0
    msg = f"coarse_crnn alignment complete ({ok_count}/{len(wavs)}), confidence_mean={mean_conf:.3f}"
    if errors:
        msg += f"; {len(errors)} file(s) failed"
    return True, msg


__all__ = ["check_coarse_crnn_ready", "resolve_coarse_crnn_model_path", "run_coarse_crnn_align"]
