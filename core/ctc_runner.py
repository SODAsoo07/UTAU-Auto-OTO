from __future__ import annotations

import importlib
import logging
import os
import re
import wave
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.ja_lab_generator import parse_ja_filename, split_ja_romaji_syllable
from core.lab_generator import decompose_hangul_to_roman
from core.kr_oto_rules import _split_kr_syllable_parts
from core.pipeline_status import (
    ALIGN_EXEC_MISSING,
    ALIGN_MODEL_MISSING,
    ALIGN_NOT_READY,
    ALIGN_RUN_FAILED,
    OK,
    make_runtime_report,
)
from core.textio_utils import read_text_auto

_CTC_SAMPLE_RATE = 16000
_PAUSE_MARKERS = {"", "pau", "sil", "sp", "spn", "br", "bre", "breath", "r"}
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z']+")
_HANGUL_RE = re.compile(r"[가-힣]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")


def _suppress_torio_ffmpeg_debug_logs() -> None:
    if str(os.environ.get("UTOA_CTC_SHOW_TORIO_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ("torio", "torio._extension", "torio._extension.utils"):
        try:
            logging.getLogger(name).setLevel(logging.INFO)
        except Exception:
            continue


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_torch_modules():
    try:
        _suppress_torio_ffmpeg_debug_logs()
        torch_mod = importlib.import_module("torch")
        torchaudio_mod = importlib.import_module("torchaudio")
        return torch_mod, torchaudio_mod, ""
    except Exception as exc:
        return None, None, str(exc)


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
        framerate = int(wav_file.getframerate() or _CTC_SAMPLE_RATE)
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
        data_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
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


def _split_ascii_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _ASCII_TOKEN_RE.finditer(str(text or ""))]


def _normalize_korean_token(token: str) -> List[str]:
    raw = str(token or "").strip()
    if not raw:
        return []
    if not _HANGUL_RE.search(raw):
        return _split_ascii_tokens(raw)

    out: List[str] = []
    for ch in raw:
        if _HANGUL_RE.match(ch):
            parts = [p for p in decompose_hangul_to_roman(ch) if str(p or "").strip()]
            joined = "".join(parts).strip().lower()
            if joined:
                out.append(joined)
        else:
            out.extend(_split_ascii_tokens(ch))
    return out


def _normalize_japanese_token(token: str) -> List[str]:
    raw = str(token or "").strip()
    if not raw:
        return []
    parsed = parse_ja_filename(raw)
    if parsed:
        out = []
        for item in parsed:
            out.extend(_split_ascii_tokens(item))
        return out
    return _split_ascii_tokens(raw)


def _normalize_transcript_words(language: str, raw_text: str) -> List[str]:
    lang = str(language or "").strip().lower()
    content = str(raw_text or "").strip()
    if not content:
        return []

    tokens = [t for t in re.split(r"\s+", content) if t.strip()]
    if len(tokens) == 1:
        tokens = re.split(r"[\s,_\-/|]+", tokens[0])
        tokens = [t for t in tokens if t.strip()]

    out: List[str] = []
    for token in tokens:
        if lang == "japanese" or _JAPANESE_RE.search(token):
            out.extend(_normalize_japanese_token(token))
        elif lang == "korean" or _HANGUL_RE.search(token):
            out.extend(_normalize_korean_token(token))
        else:
            out.extend(_split_ascii_tokens(token))
    return [w for w in out if w and w not in _PAUSE_MARKERS]


def _load_transcript_words(wav_path: str, language: str) -> Tuple[List[str], str]:
    wav_abs = os.path.abspath(str(wav_path or "").strip())
    base = os.path.splitext(os.path.basename(wav_abs))[0]
    lab_path = os.path.splitext(wav_abs)[0] + ".lab"
    lang = str(language or "").strip().lower()

    if lang == "japanese":
        filename_words = _normalize_japanese_token(base)
    elif lang == "korean":
        filename_words = _normalize_korean_token(base)
    else:
        filename_words = _split_ascii_tokens(base)
    filename_words = [w for w in filename_words if w and w not in _PAUSE_MARKERS]

    if os.path.isfile(lab_path):
        content, _enc, err = read_text_auto(lab_path)
        if not err:
            words = _normalize_transcript_words(language, str(content or ""))
            if words:
                if len(words) <= 1 and len(filename_words) >= 2:
                    return filename_words, "filename"
                return words, "lab"

    return filename_words, "filename"


def _sanitize_word_for_mms(word: str, allowed_chars: set[str]) -> str:
    cleaned = "".join(ch for ch in str(word or "").strip().lower() if ch in allowed_chars)
    cleaned = cleaned.strip("'")
    return cleaned


def _sanitize_rows(
    rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    filler_mark: str,
) -> List[Tuple[float, float, str]]:
    duration = max(float(duration_sec or 0.0), 0.0)
    sorted_rows = sorted(rows or [], key=lambda item: (float(item[0]), float(item[1])))
    out: List[Tuple[float, float, str]] = []
    prev_end = 0.0

    for start_raw, end_raw, mark_raw in sorted_rows:
        mark = str(mark_raw or "").strip() or filler_mark
        start = max(0.0, min(duration, float(start_raw)))
        end = max(0.0, min(duration, float(end_raw)))
        if end < start:
            end = start
        if start > prev_end + 1e-4:
            out.append((prev_end, start, filler_mark))
        if start < prev_end:
            start = prev_end
        if end <= start:
            end = min(duration, start + 0.001)
        out.append((start, end, mark))
        prev_end = end

    if not out and duration > 0.0:
        out.append((0.0, duration, filler_mark))
        return out

    if prev_end < duration:
        out.append((prev_end, duration, filler_mark))

    compact: List[Tuple[float, float, str]] = []
    for start, end, mark in out:
        if not compact:
            compact.append((start, end, mark))
            continue
        p_start, p_end, p_mark = compact[-1]
        if abs(start - p_end) <= 1e-4 and mark == p_mark:
            compact[-1] = (p_start, max(p_end, end), p_mark)
        else:
            compact.append((start, end, mark))
    return compact


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
    word_tier = textgrid_mod.IntervalTier(name="words", minTime=0.0, maxTime=max_t)

    for start, end, mark in phone_rows:
        phone_tier.add(max(0.0, float(start)), max(float(start), float(end)), str(mark or "").strip())
    for start, end, mark in word_rows:
        word_tier.add(max(0.0, float(start)), max(float(start), float(end)), str(mark or "").strip())

    tg.append(phone_tier)
    tg.append(word_tier)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tg.write(path)


def _ctc_cvn_enabled() -> bool:
    if str(os.environ.get("UTOA_CVN_CORRECTION_ENABLE", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return False
    return str(os.environ.get("UTOA_CTC_CV_ADAPTER_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _ctc_cvn_low_conf_only_enabled() -> bool:
    return str(os.environ.get("UTOA_CVN_LOW_CONF_ONLY", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ctc_cvn_low_conf_threshold() -> float:
    raw = str(
        os.environ.get(
            "UTOA_CVN_LOW_CONF_THRESHOLD",
            os.environ.get("UTOA_CTC_CVN_LOW_CONF_THRESHOLD", "0.80"),
        )
        or ""
    ).strip()
    try:
        val = float(raw)
    except Exception:
        return 0.80
    return max(0.0, min(1.0, val))


def _ctc_weak_voice_assist_status() -> tuple[bool, float]:
    enabled = str(os.environ.get("UTOA_MFA_WEAK_VOICE_ASSIST_ENABLE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    raw = str(os.environ.get("UTOA_MFA_WEAK_VOICE_PREEMPH_MIX", "0.35") or "").strip()
    try:
        mix = float(raw)
    except Exception:
        mix = 0.35
    mix = max(0.0, min(1.0, mix))
    return enabled, mix


def _span_score_value(span) -> float | None:
    for key in ("score", "prob", "probability", "confidence"):
        try:
            val = getattr(span, key)
        except Exception:
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            return fval
    return None


def _resolve_cvn_model_path(language: str) -> str:
    lang = str(language or "").strip().lower()
    root = _project_root()
    env_keys = [
        f"UTOA_CTC_CVN_MODEL_PATH_{lang.upper()}",
        "UTOA_CTC_CVN_MODEL_PATH",
        f"UTOA_CVN_MODEL_PATH_{lang.upper()}",
        "UTOA_CVN_MODEL_PATH",
    ]
    candidates = [str(os.environ.get(k, "") or "").strip() for k in env_keys]
    candidates.extend(
        [
            os.path.join(root, "ml_workspace", "models", "cvn", lang, "cv_binary_full_tuned.npz"),
            os.path.join(root, "ml_workspace", "models", "cvn", lang, "cv_binary_full.npz"),
            os.path.join(root, "ML_models", "cvn", lang, "cv_binary_full.npz"),
            os.path.join(root, "models_installed", "cvn", lang, "cv_binary_full.npz"),
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = os.path.abspath(candidate)
        if os.path.isfile(resolved):
            return resolved
    return ""


def _split_word_cv(word: str, language: str) -> Tuple[str, str]:
    token = str(word or "").strip().lower()
    if not token:
        return "", ""
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        onset, vowel = split_ja_romaji_syllable(token)
        return str(onset or "").strip(), str(vowel or "").strip() or token
    if lang == "korean":
        onset, vowel, coda = _split_kr_syllable_parts(token)
        onset = str(onset or "").strip()
        vowel = str(vowel or "").strip()
        coda = str(coda or "").strip()
        if not vowel:
            return "", token
        return onset, (vowel + coda) if coda else vowel

    m = re.search(r"[aeiou]", token)
    if not m:
        return "", token
    idx = int(m.start())
    return token[:idx], token[idx:]


def _estimate_cv_boundary_sec(posterior, start_sec: float, end_sec: float) -> float:
    s = float(start_sec)
    e = float(end_sec)
    if e <= s:
        return s
    duration = e - s
    default = s + (duration * 0.35)

    try:
        times = np.asarray(getattr(posterior, "times_ms"), dtype=np.float32) / 1000.0
        labels = np.asarray(getattr(posterior, "labels"))
        probs = np.asarray(getattr(posterior, "probs"), dtype=np.float32)
        order = list(getattr(posterior, "label_order", ("C", "V")) or ("C", "V"))
    except Exception:
        return default

    if times.size <= 0 or labels.size <= 0:
        return default

    idx = np.where((times >= s) & (times <= e))[0]
    if idx.size <= 0:
        return default

    chosen = None
    for k, frame_idx in enumerate(idx):
        if str(labels[frame_idx]) == "V":
            if np.any(np.asarray([str(labels[j]) == "C" for j in idx[:k]], dtype=bool)):
                chosen = int(frame_idx)
                break

    if chosen is None and probs.ndim == 2 and probs.shape[0] == times.shape[0]:
        try:
            v_col = int(order.index("V"))
            c_col = int(order.index("C"))
            local = idx[np.argmax(probs[idx, v_col] - probs[idx, c_col])]
            chosen = int(local)
        except Exception:
            chosen = None

    if chosen is None:
        boundary = default
    else:
        boundary = float(times[chosen])

    min_gap = max(0.006, duration * 0.10)
    lo = s + min_gap
    hi = e - min_gap
    if hi <= lo:
        return s + (duration * 0.5)
    return max(lo, min(hi, boundary))


def _build_phone_rows_with_cvn_adapter(
    *,
    wav_path: str,
    language: str,
    word_rows: Sequence[Tuple[float, float, str]],
    word_char_rows: Sequence[Sequence[Tuple[float, float, str]]] | None = None,
    word_confidences: Sequence[float | None] | None = None,
    callback=None,
) -> Tuple[List[Tuple[float, float, str]], str]:
    if not _ctc_cvn_enabled():
        return [], ""

    try:
        from core.cvn_classifier import predict_cvn
        from core.cvn_feature_extractor import extract_frame_features
    except Exception as exc:
        _emit(callback, f"[CTC][Adapter] CVN import failed, skip adapter: {exc}")
        return [], ""

    cvn_model = _resolve_cvn_model_path(language)
    backend = "logreg" if cvn_model else "rule"

    try:
        features = extract_frame_features(wav_path)
        posterior = predict_cvn(
            features,
            model_path=cvn_model,
            backend=backend,
            smooth_window=5,
            min_run_frames=3,
        )
    except Exception as exc:
        _emit(callback, f"[CTC][Adapter] CVN inference failed, skip adapter: {exc}")
        return [], ""

    phone_rows: List[Tuple[float, float, str]] = []
    split_count = 0
    candidate_count = 0
    low_conf_only = _ctc_cvn_low_conf_only_enabled()
    low_conf_threshold = _ctc_cvn_low_conf_threshold()
    low_conf_eligible = 0
    for idx, (start, end, token) in enumerate(word_rows):
        per_word_char_rows: Sequence[Tuple[float, float, str]] = ()
        if word_char_rows is not None and idx < len(word_char_rows):
            per_word_char_rows = word_char_rows[idx] or ()
        word_score = (
            word_confidences[idx]
            if word_confidences is not None and idx < len(word_confidences)
            else None
        )
        onset, vowel = _split_word_cv(str(token or ""), language)
        if onset and vowel and float(end) - float(start) > 0.015:
            candidate_count += 1
            allow_split = True
            if low_conf_only:
                allow_split = word_score is not None and float(word_score) < low_conf_threshold
                if allow_split:
                    low_conf_eligible += 1
            if not allow_split:
                if per_word_char_rows:
                    phone_rows.extend(per_word_char_rows)
                else:
                    phone_rows.append((float(start), float(end), str(token or "")))
                continue
            boundary = _estimate_cv_boundary_sec(posterior, float(start), float(end))
            phone_rows.append((float(start), float(boundary), onset))
            phone_rows.append((float(boundary), float(end), vowel))
            split_count += 1
        else:
            if per_word_char_rows:
                phone_rows.extend(per_word_char_rows)
            else:
                mark = vowel or onset or str(token or "")
                phone_rows.append((float(start), float(end), mark))

    adapter_info = f"cvn_backend={backend}, cv_split={split_count}/{len(word_rows)}"
    weak_assist_enabled, weak_assist_mix = _ctc_weak_voice_assist_status()
    if weak_assist_enabled:
        adapter_info += f", weak_assist=1, weak_mix={weak_assist_mix:.2f}"
    if low_conf_only:
        adapter_info += (
            f", cv_low_conf_only=1, cv_low_conf_thr={low_conf_threshold:.2f},"
            f" cv_low_conf_eligible={low_conf_eligible}/{candidate_count}"
        )
    return phone_rows, adapter_info


def check_ctc_ready(
    *,
    language: str = "korean",
    wav_folder: str = "",
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "").strip().lower() or "korean"
    if lang not in {"korean", "japanese"}:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            "CTC aligner currently supports Korean/Japanese workflows only.",
            engine="ctc",
            language=lang,
            ready=False,
        )

    if wav_folder and not os.path.isdir(wav_folder):
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"WAV folder not found: {wav_folder}",
            engine="ctc",
            language=lang,
            ready=False,
        )

    wav_files = _list_top_level_wavs(wav_folder) if wav_folder else []
    if wav_folder and not wav_files:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"No WAV files found: {wav_folder}",
            engine="ctc",
            language=lang,
            ready=False,
        )

    torch_mod, torchaudio_mod, import_err = _import_torch_modules()
    if torch_mod is None or torchaudio_mod is None:
        return make_runtime_report(
            "align",
            ALIGN_EXEC_MISSING,
            f"CTC runtime import failed (torch/torchaudio): {import_err}",
            engine="ctc",
            language=lang,
            ready=False,
        )

    textgrid_mod, textgrid_err = _import_textgrid_module()
    if textgrid_mod is None:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"python-textgrid import failed: {textgrid_err}",
            engine="ctc",
            language=lang,
            ready=False,
        )

    try:
        _ = torchaudio_mod.pipelines.MMS_FA
    except Exception as exc:
        return make_runtime_report(
            "align",
            ALIGN_MODEL_MISSING,
            f"MMS CTC bundle unavailable: {exc}",
            engine="ctc",
            language=lang,
            ready=False,
        )

    _emit(callback, "[CTC] ready: backend=MMS_FA")
    return make_runtime_report(
        "align",
        OK,
        "CTC alignment ready",
        engine="ctc",
        language=lang,
        ready=True,
    )


def run_ctc_align(
    _ctc_path: str,
    wav_folder: str,
    _dictionary_path: str,
    output_folder: str,
    *,
    language: str = "korean",
    callback=None,
) -> Tuple[bool, str]:
    lang = str(language or "").strip().lower() or "korean"
    ready = check_ctc_ready(language=lang, wav_folder=wav_folder, callback=callback)
    if str(ready.get("code", "")).upper() != OK:
        return False, str(ready.get("message", "CTC not ready") or "CTC not ready")

    torch_mod, torchaudio_mod, import_err = _import_torch_modules()
    if torch_mod is None or torchaudio_mod is None:
        return False, f"CTC runtime import failed: {import_err}"

    try:
        bundle = torchaudio_mod.pipelines.MMS_FA
        model = bundle.get_model(with_star=False)
        model.eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        labels = tuple(bundle.get_labels(star=None))
    except Exception as exc:
        return False, f"CTC backend initialization failed: {exc}"

    allowed_chars = {str(ch) for ch in labels if str(ch) not in {"-", "*"}}
    id_to_label = {idx: str(label) for idx, label in enumerate(labels)}

    wav_files = _list_top_level_wavs(wav_folder)
    if not wav_files:
        return False, f"No WAV files found: {wav_folder}"

    max_files_env = str(os.environ.get("UTOA_CTC_MAX_FILES", "")).strip()
    if max_files_env:
        try:
            max_files = max(1, int(float(max_files_env)))
            wav_files = wav_files[:max_files]
        except Exception:
            pass

    os.makedirs(output_folder, exist_ok=True)
    ok_count = 0
    errors: List[str] = []

    for wav_path in wav_files:
        wav_name = os.path.basename(wav_path)
        try:
            words_raw, source = _load_transcript_words(wav_path, lang)
            words = []
            for item in words_raw:
                normalized = _sanitize_word_for_mms(item, allowed_chars)
                if normalized and normalized not in _PAUSE_MARKERS:
                    words.append(normalized)
            if not words:
                raise RuntimeError("empty transcript after normalization")

            audio_np, sr, _duration_src = _read_pcm_as_float_mono(wav_path)
            if audio_np.size <= 0:
                raise RuntimeError("empty waveform")

            wave_t = torch_mod.from_numpy(audio_np).to(dtype=torch_mod.float32).unsqueeze(0)
            if int(sr) != int(bundle.sample_rate):
                wave_t = torchaudio_mod.functional.resample(wave_t, int(sr), int(bundle.sample_rate))

            with torch_mod.inference_mode():
                emissions, _ = model(wave_t)

            token_ids = tokenizer(words)
            spans_by_word = aligner(emissions[0], token_ids)
            if len(spans_by_word) != len(words):
                raise RuntimeError("token span/word count mismatch")

            duration_sec = float(wave_t.shape[1]) / float(max(bundle.sample_rate, 1))
            frame_ratio = duration_sec / float(max(int(emissions.shape[1]), 1))

            word_rows_raw: List[Tuple[float, float, str]] = []
            phone_rows_char: List[Tuple[float, float, str]] = []
            word_char_rows: List[List[Tuple[float, float, str]]] = []
            word_confidences: List[float | None] = []
            for idx, spans in enumerate(spans_by_word):
                if not spans:
                    continue
                w_start = float(spans[0].start) * frame_ratio
                w_end = float(spans[-1].end) * frame_ratio
                word_rows_raw.append((w_start, w_end, words[idx]))
                score_vals: List[float] = []
                per_word_char: List[Tuple[float, float, str]] = []
                for span in spans:
                    span_score = _span_score_value(span)
                    if span_score is not None:
                        score_vals.append(span_score)
                    mark = id_to_label.get(int(span.token), "").strip()
                    if not mark or mark in {"-", "*"}:
                        continue
                    p_start = float(span.start) * frame_ratio
                    p_end = float(span.end) * frame_ratio
                    char_row = (p_start, p_end, mark)
                    phone_rows_char.append(char_row)
                    per_word_char.append(char_row)
                if not per_word_char:
                    per_word_char.append((w_start, w_end, words[idx]))
                word_char_rows.append(per_word_char)
                if score_vals:
                    word_confidences.append(float(np.mean(np.asarray(score_vals, dtype=np.float32))))
                else:
                    word_confidences.append(None)

            if not word_rows_raw:
                raise RuntimeError("empty word alignment output")

            adapter_rows, adapter_info = _build_phone_rows_with_cvn_adapter(
                wav_path=wav_path,
                language=lang,
                word_rows=word_rows_raw,
                word_char_rows=word_char_rows,
                word_confidences=word_confidences,
                callback=callback,
            )
            phone_rows_raw = adapter_rows if adapter_rows else phone_rows_char
            if not phone_rows_raw:
                phone_rows_raw = [(start, end, mark) for start, end, mark in word_rows_raw]

            word_rows = _sanitize_rows(word_rows_raw, duration_sec=duration_sec, filler_mark="spn")
            phone_rows = _sanitize_rows(phone_rows_raw, duration_sec=duration_sec, filler_mark="pau")

            tg_name = os.path.splitext(wav_name)[0] + ".TextGrid"
            tg_path = os.path.join(output_folder, tg_name)
            _write_textgrid(
                tg_path,
                phone_rows=phone_rows,
                word_rows=word_rows,
                duration_sec=duration_sec,
            )
            ok_count += 1
            extra = f", {adapter_info}" if adapter_info else ""
            _emit(
                callback,
                f"[CTC] aligned file={wav_name} words={len(words)} source={source}{extra}",
            )
        except Exception as exc:
            err = f"{wav_name}: {exc}"
            errors.append(err)
            _emit(callback, f"[CTC] failed: {err}")

    if ok_count <= 0:
        if errors:
            return False, f"CTC alignment failed: {errors[0]}"
        return False, "CTC alignment failed."

    summary = f"ctc alignment complete ({ok_count}/{len(wav_files)})"
    if errors:
        summary += f"; {len(errors)} file(s) failed"
    return True, summary


__all__ = [
    "check_ctc_ready",
    "run_ctc_align",
]
