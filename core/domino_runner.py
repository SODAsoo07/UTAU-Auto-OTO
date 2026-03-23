from __future__ import annotations

import importlib
import os
import wave
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.ja_domino_transcript import build_domino_phone_plan, load_ja_syllables_for_wav
from core.pipeline_status import (
    ALIGN_EXEC_MISSING,
    ALIGN_MODEL_MISSING,
    ALIGN_NOT_READY,
    ALIGN_RUN_FAILED,
    OK,
    make_runtime_report,
)


_DOMINO_DEFAULT_MIN_FRAME = 3


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_domino_model_path(onnx_path: str = "") -> str:
    explicit = str(onnx_path or "").strip()
    env_path = str(os.environ.get("UTOA_DOMINO_ONNX_PATH", "") or "").strip()

    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    if env_path:
        candidates.append(env_path)

    root = _project_root()
    candidates.extend(
        [
            os.path.join(root, "assets", "models", "domino", "phoneme_transition_model.onnx"),
            os.path.join(root, "models_installed", "domino", "phoneme_transition_model.onnx"),
            os.path.join(root, "onnx_model", "phoneme_transition_model.onnx"),
            os.path.join(root, "ML_models", "domino", "phoneme_transition_model.onnx"),
        ]
    )

    # pydomino-prebuilt wheels bundle the ONNX file under package/onnx_model/.
    try:
        pydomino_mod = importlib.import_module("pydomino")
        pkg_file = str(getattr(pydomino_mod, "__file__", "") or "").strip()
        if pkg_file:
            pkg_dir = os.path.dirname(os.path.abspath(pkg_file))
            candidates.append(os.path.join(pkg_dir, "onnx_model", "phoneme_transition_model.onnx"))
    except Exception:
        pass

    for path in candidates:
        if not path:
            continue
        resolved = os.path.abspath(path)
        if os.path.isfile(resolved):
            return resolved
    return ""


def _import_pydomino():
    try:
        pydomino = importlib.import_module("pydomino")
        return pydomino, ""
    except Exception as exc:
        return None, str(exc)


def _import_textgrid_module():
    try:
        import textgrid  # type: ignore

        return textgrid, ""
    except Exception as exc:
        return None, str(exc)


def _list_top_level_wavs(wav_folder: str) -> List[str]:
    root = os.path.abspath(str(wav_folder or "").strip())
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".wav"):
            out.append(os.path.join(root, name))
    return out


def _read_pcm_as_float_mono(path: str) -> Tuple[np.ndarray, int, float]:
    with wave.open(path, "rb") as wav_file:
        channels = int(wav_file.getnchannels() or 1)
        sampwidth = int(wav_file.getsampwidth() or 2)
        framerate = int(wav_file.getframerate() or 16000)
        frame_count = int(wav_file.getnframes() or 0)
        raw = wav_file.readframes(frame_count)

    if frame_count <= 0:
        return np.zeros((0,), dtype=np.float32), framerate, 0.0

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        data_u8 = np.frombuffer(raw, dtype=np.uint8)
        data_u8 = data_u8.reshape(-1, 3)
        data_i32 = (
            data_u8[:, 0].astype(np.int32)
            | (data_u8[:, 1].astype(np.int32) << 8)
            | (data_u8[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        data_i32 = (data_i32 ^ sign_bit) - sign_bit
        data = data_i32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        sample_count = data.shape[0] // channels
        if sample_count <= 0:
            return np.zeros((0,), dtype=np.float32), framerate, 0.0
        data = data[: sample_count * channels].reshape(sample_count, channels).mean(axis=1)

    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    duration_sec = float(frame_count) / float(max(framerate, 1))
    return data, framerate, duration_sec


def _resample_linear(signal: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr <= 0 or dst_sr <= 0:
        return signal.astype(np.float32)
    if src_sr == dst_sr:
        return signal.astype(np.float32)
    if signal.size <= 1:
        return signal.astype(np.float32)

    src_len = int(signal.size)
    dst_len = max(1, int(round(src_len * float(dst_sr) / float(src_sr))))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float64)
    out = np.interp(dst_x, src_x, signal.astype(np.float64)).astype(np.float32)
    return out


def _sanitize_aligned_rows(rows: Sequence[Sequence[object]], duration_sec: float) -> List[Tuple[float, float, str]]:
    duration = max(float(duration_sec or 0.0), 0.0)
    cleaned: List[Tuple[float, float, str]] = []
    prev_end = 0.0

    for row in rows or []:
        if row is None or len(row) < 3:
            continue
        try:
            start = float(row[0])
            end = float(row[1])
        except Exception:
            continue
        mark = str(row[2] or "").strip() or "pau"

        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end < start:
            end = start

        if start < prev_end:
            start = prev_end
            if end < start:
                end = start

        if end - start < 1e-4:
            end = min(duration, start + 0.001)

        cleaned.append((start, end, mark))
        prev_end = end

    if not cleaned and duration > 0.0:
        cleaned = [(0.0, duration, "pau")]

    if cleaned:
        first_start, first_end, first_mark = cleaned[0]
        if first_start > 0.0:
            cleaned.insert(0, (0.0, first_start, "pau" if first_mark != "pau" else first_mark))

        last_start, last_end, last_mark = cleaned[-1]
        if duration > 0.0 and last_end < duration:
            cleaned.append((last_end, duration, "pau" if last_mark != "pau" else last_mark))

    return cleaned


def _build_word_rows(
    phone_rows: Sequence[Tuple[float, float, str]],
    word_spans: Sequence[Tuple[str, int, int]],
    duration_sec: float,
) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    duration = max(float(duration_sec or 0.0), 0.0)

    for label, s_idx, e_idx in word_spans or []:
        if s_idx < 0 or e_idx < s_idx:
            continue
        if s_idx >= len(phone_rows) or e_idx >= len(phone_rows):
            continue
        start = max(0.0, min(duration, float(phone_rows[s_idx][0])))
        end = max(0.0, min(duration, float(phone_rows[e_idx][1])))
        if end <= start:
            end = min(duration, start + 0.001)
        out.append((start, end, str(label or "").strip() or "unk"))

    return out


def _write_textgrid(
    path: str,
    *,
    phone_rows: Sequence[Tuple[float, float, str]],
    word_rows: Sequence[Tuple[float, float, str]],
    duration_sec: float,
):
    textgrid_mod, err = _import_textgrid_module()
    if textgrid_mod is None:
        raise RuntimeError(f"python-textgrid import failed: {err}")

    max_t = max(float(duration_sec or 0.0), 0.0)
    if phone_rows:
        max_t = max(max_t, float(phone_rows[-1][1]))
    if word_rows:
        max_t = max(max_t, float(word_rows[-1][1]))

    tg = textgrid_mod.TextGrid(minTime=0.0, maxTime=max_t)

    phone_tier = textgrid_mod.IntervalTier(name="phones", minTime=0.0, maxTime=max_t)
    if not phone_rows and max_t > 0.0:
        phone_rows = [(0.0, max_t, "pau")]
    for start, end, mark in phone_rows:
        phone_tier.add(max(0.0, float(start)), max(float(start), float(end)), str(mark or "").strip())
    tg.append(phone_tier)

    word_tier = textgrid_mod.IntervalTier(name="words", minTime=0.0, maxTime=max_t)
    if not word_rows and max_t > 0.0:
        word_rows = [(0.0, max_t, "spn")]
    for start, end, mark in word_rows:
        word_tier.add(max(0.0, float(start)), max(float(start), float(end)), str(mark or "").strip())
    tg.append(word_tier)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tg.write(path)


def check_domino_ready(
    *,
    language: str = "japanese",
    wav_folder: str = "",
    onnx_path: str = "",
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "").strip().lower() or "japanese"
    if lang != "japanese":
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            "Domino aligner supports Japanese only.",
            engine="domino",
            language=lang,
            ready=False,
            domino_onnx_path="",
        )

    if wav_folder and not os.path.isdir(wav_folder):
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"WAV folder not found: {wav_folder}",
            engine="domino",
            language=lang,
            ready=False,
            domino_onnx_path="",
        )

    resolved_onnx = _resolve_domino_model_path(onnx_path)
    if not resolved_onnx:
        return make_runtime_report(
            "align",
            ALIGN_MODEL_MISSING,
            "Domino ONNX model not found. Set UTOA_DOMINO_ONNX_PATH or install phoneme_transition_model.onnx.",
            engine="domino",
            language=lang,
            ready=False,
            domino_onnx_path="",
        )

    pydomino_mod, pydomino_err = _import_pydomino()
    if pydomino_mod is None:
        return make_runtime_report(
            "align",
            ALIGN_EXEC_MISSING,
            f"pydomino import failed: {pydomino_err}",
            engine="domino",
            language=lang,
            ready=False,
            domino_onnx_path=resolved_onnx,
        )

    textgrid_mod, textgrid_err = _import_textgrid_module()
    if textgrid_mod is None:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"python-textgrid import failed: {textgrid_err}",
            engine="domino",
            language=lang,
            ready=False,
            domino_onnx_path=resolved_onnx,
        )

    _emit(callback, f"[Domino] ready: onnx={resolved_onnx}")
    return make_runtime_report(
        "align",
        OK,
        "Domino alignment ready",
        engine="domino",
        language=lang,
        ready=True,
        domino_onnx_path=resolved_onnx,
    )


def run_domino_align(
    onnx_path: str,
    wav_folder: str,
    _dictionary_path: str,
    output_folder: str,
    *,
    language: str = "japanese",
    callback=None,
    min_frame: int = _DOMINO_DEFAULT_MIN_FRAME,
) -> Tuple[bool, str]:
    lang = str(language or "").strip().lower() or "japanese"
    ready = check_domino_ready(
        language=lang,
        wav_folder=wav_folder,
        onnx_path=onnx_path,
        callback=callback,
    )
    if str(ready.get("code", "")).upper() != OK:
        return False, str(ready.get("message", "Domino not ready") or "Domino not ready")

    resolved_onnx = str(ready.get("domino_onnx_path", "") or "")
    wav_files = _list_top_level_wavs(wav_folder)
    if not wav_files:
        return False, f"No WAV files found: {wav_folder}"

    pydomino_mod, pydomino_err = _import_pydomino()
    if pydomino_mod is None:
        return False, f"pydomino import failed: {pydomino_err}"

    try:
        min_frame_env = int(float(str(os.environ.get("UTOA_DOMINO_MIN_FRAME", "") or min_frame)))
    except Exception:
        min_frame_env = int(min_frame)
    min_frame_env = max(1, min_frame_env)

    os.makedirs(output_folder, exist_ok=True)

    aligner = None
    ok_count = 0
    errors: List[str] = []

    try:
        aligner = pydomino_mod.Aligner(resolved_onnx)

        for wav_path in wav_files:
            wav_name = os.path.basename(wav_path)
            try:
                syllables, syllable_source = load_ja_syllables_for_wav(wav_path)
                phones, word_spans = build_domino_phone_plan(syllables)
                if len(phones) < 3:
                    raise RuntimeError("insufficient phoneme tokens")

                signal, src_sr, duration_sec = _read_pcm_as_float_mono(wav_path)
                signal_16k = _resample_linear(signal, src_sr, 16000)
                if signal_16k.size <= 0:
                    raise RuntimeError("empty waveform")

                _emit(
                    callback,
                    f"[Domino] align file={wav_name} source={syllable_source} phones={len(phones)} min_frame={min_frame_env}",
                )
                aligned_rows = aligner.align(signal_16k, " ".join(phones), int(min_frame_env))
                phone_rows = _sanitize_aligned_rows(aligned_rows, duration_sec)
                if not phone_rows:
                    raise RuntimeError("empty alignment output")

                word_rows = _build_word_rows(phone_rows, word_spans, duration_sec)
                textgrid_name = os.path.splitext(wav_name)[0] + ".TextGrid"
                textgrid_path = os.path.join(output_folder, textgrid_name)
                _write_textgrid(
                    textgrid_path,
                    phone_rows=phone_rows,
                    word_rows=word_rows,
                    duration_sec=duration_sec,
                )
                ok_count += 1
            except Exception as exc:
                err = f"{wav_name}: {exc}"
                errors.append(err)
                _emit(callback, f"[Domino] failed: {err}")

    except Exception as exc:
        return False, f"Domino runtime error: {exc}"
    finally:
        try:
            if aligner is not None and hasattr(aligner, "release"):
                aligner.release()
        except Exception:
            pass

    if ok_count <= 0:
        message = "Domino alignment failed"
        if errors:
            message = f"{message}: {errors[0]}"
        return False, message

    summary = f"domino alignment complete ({ok_count}/{len(wav_files)})"
    if errors:
        summary += f"; {len(errors)} file(s) failed"
    return True, summary


__all__ = [
    "check_domino_ready",
    "run_domino_align",
]

