from __future__ import annotations

import json
import os
import re
import wave
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.ja_lab_generator import parse_ja_filename, split_ja_romaji_syllable
from core.kr_oto_rules import _split_kr_syllable_parts
from core.lab_generator import decompose_hangul_to_roman
from core.pipeline_status import ALIGN_NOT_READY, ALIGN_RUN_FAILED, OK, make_runtime_report
from core.textio_utils import read_text_auto

_PAUSE_MARKERS = {"", "pau", "sil", "sp", "spn", "br", "bre", "breath", "r"}
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+")
_HANGUL_RE = re.compile(r"[가-힣]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_KR_ORDER_INITIAL_CANON = {
    "k": "g",
    "gg": "kk",
    "t": "d",
    "dd": "tt",
    "p": "b",
    "bb": "pp",
    "z": "j",
    "zz": "jj",
    "c": "ch",
    "q": "k",
    "kh": "k",
    "tx": "t",
    "th": "t",
    "ph": "p",
    "f": "p",
    "sh": "s",
}
_KR_ORDER_CODA_CANON = {
    "g": "k",
    "t": "d",
    "p": "b",
    "r": "l",
    "gg": "kk",
    "bb": "pp",
    "dd": "tt",
    "zz": "jj",
    "c": "ch",
    "q": "k",
    "kh": "k",
    "tx": "t",
    "th": "t",
    "ph": "p",
    "f": "p",
    "sh": "s",
}
_KR_ORDER_VOWEL_CANON = {
    "eui": "ui",
    "yi": "ui",
    "weo": "wo",
}
_VOICELESS_ONSETS = {
    "k",
    "ky",
    "s",
    "sh",
    "t",
    "ts",
    "ch",
    "h",
    "hy",
    "f",
    "p",
    "py",
    "q",
    "c",
}
_VOICED_ONSETS = {
    "g",
    "gy",
    "z",
    "j",
    "d",
    "dz",
    "b",
    "by",
    "m",
    "my",
    "n",
    "ny",
    "r",
    "ry",
    "w",
    "y",
    "v",
}
_SONORANT_ONSETS = {"m", "my", "n", "ny", "r", "ry", "w", "y", "l", "ly"}
_JP_NASAL_G_ONSETS = {"g", "gy", "gw"}
_PLOSIVE_ONSETS = {
    "k",
    "ky",
    "g",
    "gy",
    "gw",
    "t",
    "d",
    "ts",
    "dz",
    "ch",
    "j",
    "p",
    "py",
    "b",
    "by",
    "q",
    "c",
}


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        out = float(default)
    else:
        try:
            out = float(raw)
        except Exception:
            out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return float(out)


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        out = int(default)
    else:
        try:
            out = int(float(raw))
        except Exception:
            out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    if max_value is not None:
        out = min(int(max_value), out)
    return int(out)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _profile_float(profile: Dict[str, object], key: str, default: float) -> float:
    try:
        return float(profile.get(key, default))
    except Exception:
        return float(default)


def _profile_int(profile: Dict[str, object], key: str, default: int) -> int:
    try:
        return int(float(profile.get(key, default)))
    except Exception:
        return int(default)


def _profile_bool(profile: Dict[str, object], key: str, default: bool) -> bool:
    raw = profile.get(key, default)
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _seq_cvn_enabled() -> bool:
    if str(os.environ.get("UTOA_CVN_CORRECTION_ENABLE", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return False
    return _env_bool("UTOA_SEQUENCE_CVN_ASSIST_ENABLE", default=True)


def _resolve_cvn_model_path(language: str) -> str:
    lang = str(language or "").strip().lower()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_keys = [
        f"UTOA_SEQUENCE_CVN_MODEL_PATH_{lang.upper()}",
        "UTOA_SEQUENCE_CVN_MODEL_PATH",
        f"UTOA_CVN_MODEL_PATH_{lang.upper()}",
        "UTOA_CVN_MODEL_PATH",
    ]
    candidates = [str(os.environ.get(k, "") or "").strip() for k in env_keys]
    candidates.extend(
        [
            os.path.join(repo_root, "ml_workspace", "models", "cvn", lang, "cv_binary_full_tuned.npz"),
            os.path.join(repo_root, "ml_workspace", "models", "cvn", lang, "cv_binary_full.npz"),
            os.path.join(repo_root, "ML_models", "cvn", lang, "cv_binary_full.npz"),
            os.path.join(repo_root, "models_installed", "cvn", lang, "cv_binary_full.npz"),
        ]
    )
    for path in candidates:
        if not path:
            continue
        resolved = os.path.abspath(path)
        if os.path.isfile(resolved):
            return resolved
    return ""


def _load_sequence_profile(language: str, callback=None) -> Dict[str, object]:
    lang = str(language or "").strip().lower() or "korean"
    explicit_path = str(os.environ.get("UTOA_SEQUENCE_ALIGN_PROFILE_PATH", "") or "").strip()
    profile_path = ""
    if explicit_path and os.path.isfile(explicit_path):
        profile_path = os.path.abspath(explicit_path)
    if not profile_path:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(repo_root, "models", f"sequence_aligner_profile_{lang}.json")
        if os.path.isfile(candidate):
            profile_path = candidate
    if not profile_path:
        return {}

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        params_obj = payload.get("params", payload)
        profile = dict(params_obj) if isinstance(params_obj, dict) else {}
        if "frame_ms" not in profile and isinstance(payload.get("frame_ms"), (int, float, str)):
            profile["frame_ms"] = payload.get("frame_ms")
        if "hop_ms" not in profile and isinstance(payload.get("hop_ms"), (int, float, str)):
            profile["hop_ms"] = payload.get("hop_ms")
        _emit(callback, f"[SEQ] profile loaded: {profile_path}")
        return profile
    except Exception as exc:
        _emit(callback, f"[SEQ] profile load failed: {exc}")
        return {}


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
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if str(name).lower().endswith(".wav"):
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


def _highpass_one_pole(signal: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size <= 1 or sr <= 0 or cutoff_hz <= 0.0:
        return x.astype(np.float32, copy=True)
    fc = float(max(1.0, min(float(cutoff_hz), float(sr) * 0.45)))
    rc = 1.0 / (2.0 * np.pi * fc)
    dt = 1.0 / float(sr)
    alpha = float(rc / (rc + dt))
    y = np.zeros_like(x, dtype=np.float32)
    prev_y = 0.0
    prev_x = float(x[0])
    for i in range(1, int(x.size)):
        xi = float(x[i])
        yi = alpha * (prev_y + xi - prev_x)
        y[i] = float(yi)
        prev_y = yi
        prev_x = xi
    return y


def _prepare_analysis_signal(
    signal: np.ndarray,
    sr: int,
    *,
    dc_remove: bool = True,
    highpass_hz: float = 50.0,
    max_gain_db: float = 7.0,
    target_rms: float = 0.08,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size <= 0:
        return x.astype(np.float32, copy=True)
    out = x.astype(np.float32, copy=True)
    if dc_remove:
        out = out - float(np.mean(out, dtype=np.float64))
    if highpass_hz > 0.0:
        out = _highpass_one_pole(out, int(sr), float(highpass_hz))
    rms = float(np.sqrt(np.mean(np.square(out), dtype=np.float64) + 1e-12))
    tgt = float(max(1e-4, target_rms))
    desired_gain_db = 20.0 * float(np.log10(tgt / max(rms, 1e-8)))
    gain_cap = float(max(0.0, max_gain_db))
    gain_db = float(np.clip(desired_gain_db, -gain_cap, gain_cap))
    gain = float(np.power(10.0, gain_db / 20.0))
    out = np.clip(out * gain, -1.0, 1.0).astype(np.float32)
    return out


def _build_frame_voicing_mask(
    rms: np.ndarray,
    *,
    silence_threshold: float,
    min_active_frames: int,
    max_gap_frames: int,
    zcr: np.ndarray | None = None,
    hf_ratio: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    lookback_frames: int = 1,
) -> np.ndarray:
    mask = _smooth_activity_mask(
        rms,
        threshold=float(silence_threshold),
        min_active_frames=max(1, int(min_active_frames)),
        max_gap_frames=max(0, int(max_gap_frames)),
    )
    if zcr is None or hf_ratio is None or flux is None:
        return mask
    return _expand_activity_with_consonant_cues(
        mask,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf_ratio,
        flux=flux,
        silence_threshold=float(silence_threshold),
        lookback_frames=max(1, int(lookback_frames)),
    )


def _build_ap_sp_mask(
    rms: np.ndarray,
    *,
    silence_threshold: float,
    zcr: np.ndarray | None = None,
    hf_ratio: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    min_run_frames: int = 2,
    max_gap_frames: int = 1,
) -> np.ndarray:
    r = np.asarray(rms, dtype=np.float32).reshape(-1)
    n = int(r.size)
    if n <= 0:
        return np.zeros((0,), dtype=bool)
    if zcr is None or hf_ratio is None or flux is None:
        return np.zeros((n,), dtype=bool)

    z = np.asarray(zcr, dtype=np.float32).reshape(-1)
    h = np.asarray(hf_ratio, dtype=np.float32).reshape(-1)
    f = np.asarray(flux, dtype=np.float32).reshape(-1)
    if z.size != n or h.size != n or f.size != n:
        return np.zeros((n,), dtype=bool)

    sil = float(max(1e-7, silence_threshold))
    z_hi = float(np.quantile(z, 0.68))
    h_hi = float(np.quantile(h, 0.66))
    f_mid = float(np.quantile(f, 0.55))

    ap = np.zeros((n,), dtype=bool)
    for i in range(n):
        hard_sil = bool(r[i] <= sil * 0.78)
        low_energy = bool(r[i] <= sil * 1.10)
        breath_like = bool(
            low_energy
            and z[i] >= z_hi * 0.90
            and h[i] >= h_hi * 0.86
            and f[i] >= f_mid * 0.92
        )
        ap[i] = bool(hard_sil or breath_like)

    # Fill tiny gaps in AP/SP regions.
    gap = max(0, int(max_gap_frames))
    if gap > 0 and n >= 3:
        i = 0
        while i < n:
            if ap[i]:
                i += 1
                continue
            s = i
            while i < n and (not ap[i]):
                i += 1
            e = i
            if s > 0 and e < n and ap[s - 1] and ap[e] and (e - s) <= gap:
                ap[s:e] = True

    # Remove tiny blips.
    min_run = max(1, int(min_run_frames))
    if min_run > 1:
        i = 0
        while i < n:
            if not ap[i]:
                i += 1
                continue
            s = i
            while i < n and ap[i]:
                i += 1
            e = i
            if (e - s) < min_run:
                ap[s:e] = False

    return ap


def _inject_ap_labels(
    labels: Sequence[str],
    *,
    ap_mask: np.ndarray | None,
    rms: np.ndarray,
    silence_threshold: float,
) -> List[str]:
    out = [str(x or "").strip().lower() for x in (labels or [])]
    if ap_mask is None:
        return out
    ap = np.asarray(ap_mask, dtype=bool).reshape(-1)
    r = np.asarray(rms, dtype=np.float32).reshape(-1)
    n = len(out)
    if ap.size != n or r.size != n:
        return out

    sil = float(max(1e-7, silence_threshold))
    for i in range(n):
        if not bool(ap[i]):
            continue
        cur = out[i]
        if cur in {"unvoiced_onset", "voiced_onset"}:
            continue
        if float(r[i]) <= sil * 0.82:
            out[i] = "sil"
        else:
            out[i] = "breath"
    return out


def _split_ascii_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _ASCII_TOKEN_RE.finditer(str(text or ""))]


def _canonical_kr_order_token(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return ""
    onset, vowel, coda = _split_kr_syllable_parts(text)
    if vowel:
        onset = _KR_ORDER_INITIAL_CANON.get(str(onset or "").lower(), str(onset or "").lower())
        vowel = _KR_ORDER_VOWEL_CANON.get(str(vowel or "").lower(), str(vowel or "").lower())
        coda = _KR_ORDER_CODA_CANON.get(str(coda or "").lower(), str(coda or "").lower())
        return f"{onset}{vowel}{coda}"
    return _KR_ORDER_CODA_CANON.get(text, _KR_ORDER_INITIAL_CANON.get(text, _KR_ORDER_VOWEL_CANON.get(text, text)))


def _order_tokens_match(language: str, left: str, right: str) -> bool:
    lval = str(left or "").strip().lower()
    rval = str(right or "").strip().lower()
    if lval == rval:
        return True
    if str(language or "").strip().lower() == "korean":
        return _canonical_kr_order_token(lval) == _canonical_kr_order_token(rval)
    return False


def _order_match_ratio(language: str, words: Sequence[str], filename_words: Sequence[str]) -> float:
    common = min(len(words or []), len(filename_words or []))
    if common <= 0:
        return 0.0
    match_count = sum(
        1
        for idx in range(common)
        if _order_tokens_match(language, str((words or [])[idx] or ""), str((filename_words or [])[idx] or ""))
    )
    return float(match_count) / float(common)


def _normalize_korean_token(token: str) -> List[str]:
    raw = str(token or "").strip()
    if not raw:
        return []
    if not _HANGUL_RE.search(raw):
        return _split_ascii_tokens(raw)

    out: List[str] = []
    for ch in raw:
        if _HANGUL_RE.match(ch):
            try:
                parts = [p for p in decompose_hangul_to_roman(ch) if str(p or "").strip()]
            except Exception:
                parts = []
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
        out: List[str] = []
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


def _filename_words_for_wav(wav_path: str, language: str) -> List[str]:
    wav_abs = os.path.abspath(str(wav_path or "").strip())
    base = os.path.splitext(os.path.basename(wav_abs))[0]
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        filename_words = _normalize_japanese_token(base)
    elif lang == "korean":
        filename_words = _normalize_korean_token(base)
    else:
        filename_words = _split_ascii_tokens(base)
    return [w for w in filename_words if w and w not in _PAUSE_MARKERS]


def _load_transcript_words(wav_path: str, language: str) -> Tuple[List[str], str]:
    wav_abs = os.path.abspath(str(wav_path or "").strip())
    lab_path = os.path.splitext(wav_abs)[0] + ".lab"
    filename_words = _filename_words_for_wav(wav_abs, language)

    if os.path.isfile(lab_path):
        content, _enc, err = read_text_auto(lab_path)
        if not err:
            words = _normalize_transcript_words(language, str(content or ""))
            if words:
                if (
                    str(language or "").strip().lower() == "korean"
                    and filename_words
                    and len(words) == len(filename_words)
                    and _order_match_ratio(language, words, filename_words) >= 0.98
                ):
                    return filename_words, "filename"
                return words, "lab"

    if filename_words:
        return filename_words, "filename"
    return [], "empty"


def _framewise_rms(samples: np.ndarray, sr: int, *, frame_ms: int, hop_ms: int) -> Tuple[np.ndarray, float]:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.size <= 0:
        return np.zeros((0,), dtype=np.float32), 0.0
    sr_eff = max(int(sr), 1)
    frame_len = max(1, int(round(float(sr_eff) * float(frame_ms) / 1000.0)))
    hop_len = max(1, int(round(float(sr_eff) * float(hop_ms) / 1000.0)))

    if signal.size < frame_len:
        padded = np.zeros((frame_len,), dtype=np.float32)
        padded[: signal.size] = signal
        signal = padded

    starts = list(range(0, max(1, signal.size - frame_len + 1), hop_len))
    if starts and starts[-1] + frame_len < signal.size:
        starts.append(max(0, signal.size - frame_len))

    rms_values: List[float] = []
    for st in starts:
        en = min(signal.size, st + frame_len)
        win = signal[st:en]
        if win.size < frame_len:
            padded = np.zeros((frame_len,), dtype=np.float32)
            padded[: win.size] = win
            win = padded
        val = float(np.sqrt(float(np.mean(np.square(win), dtype=np.float64)) + 1e-12))
        rms_values.append(val)

    hop_sec = float(hop_len) / float(sr_eff)
    return np.asarray(rms_values, dtype=np.float32), float(hop_sec)


def _framewise_consonant_cues(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: int,
    hop_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.size <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    sr_eff = max(int(sr), 1)
    frame_len = max(1, int(round(float(sr_eff) * float(frame_ms) / 1000.0)))
    hop_len = max(1, int(round(float(sr_eff) * float(hop_ms) / 1000.0)))

    if signal.size < frame_len:
        padded = np.zeros((frame_len,), dtype=np.float32)
        padded[: signal.size] = signal
        signal = padded

    starts = list(range(0, max(1, signal.size - frame_len + 1), hop_len))
    if starts and starts[-1] + frame_len < signal.size:
        starts.append(max(0, signal.size - frame_len))

    window = np.hanning(frame_len).astype(np.float32)
    hf_start_hz = min(3500.0, max(1800.0, float(sr_eff) * 0.18))
    prev_spec: np.ndarray | None = None
    zcr_vals: List[float] = []
    hf_vals: List[float] = []
    flux_vals: List[float] = []

    for st in starts:
        en = min(signal.size, st + frame_len)
        win = signal[st:en]
        if win.size < frame_len:
            padded = np.zeros((frame_len,), dtype=np.float32)
            padded[: win.size] = win
            win = padded

        signs = np.signbit(win)
        zcr = float(np.mean(np.not_equal(signs[1:], signs[:-1]).astype(np.float32))) if win.size > 1 else 0.0
        zcr_vals.append(zcr)

        spec = np.abs(np.fft.rfft(win * window)).astype(np.float32)
        total = float(np.sum(spec) + 1e-9)
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sr_eff))
        hf_mask = freqs >= float(hf_start_hz)
        hf = float(np.sum(spec[hf_mask])) / total if np.any(hf_mask) else 0.0
        hf_vals.append(hf)

        if prev_spec is None:
            flux_vals.append(0.0)
        else:
            diff = np.maximum(spec - prev_spec, 0.0)
            flux_vals.append(float(np.sum(diff)) / float(np.sum(prev_spec) + 1e-9))
        prev_spec = spec

    return (
        np.asarray(zcr_vals, dtype=np.float32),
        np.asarray(hf_vals, dtype=np.float32),
        np.asarray(flux_vals, dtype=np.float32),
    )


def _expand_activity_with_consonant_cues(
    activity_mask: np.ndarray,
    *,
    rms: np.ndarray,
    zcr: np.ndarray,
    hf_ratio: np.ndarray,
    flux: np.ndarray,
    silence_threshold: float,
    lookback_frames: int,
) -> np.ndarray:
    mask = np.asarray(activity_mask, dtype=bool).copy()
    n = int(mask.size)
    if n <= 0:
        return mask
    if any(np.asarray(arr).size != n for arr in (rms, zcr, hf_ratio, flux)):
        return mask

    r = np.asarray(rms, dtype=np.float32)
    z = np.asarray(zcr, dtype=np.float32)
    h = np.asarray(hf_ratio, dtype=np.float32)
    f = np.asarray(flux, dtype=np.float32)
    lookback = max(0, int(lookback_frames))
    if lookback <= 0:
        return mask

    z_hi = float(np.quantile(z, 0.68))
    z_lo = float(np.quantile(z, 0.40))
    h_hi = float(np.quantile(h, 0.66))
    h_mid = float(np.quantile(h, 0.52))
    f_hi = float(np.quantile(f, 0.70))
    f_mid = float(np.quantile(f, 0.55))
    noise_floor = float(np.quantile(r, 0.18))
    rms_lo = float(max(1e-6, float(silence_threshold) * 0.58))
    rms_voiced = float(max(1e-6, float(silence_threshold) * 0.72))
    blank_guard = float(max(noise_floor * 1.90, float(silence_threshold) * 0.40, 1e-6))

    starts: List[int] = []
    for i in range(n):
        if mask[i] and (i == 0 or not mask[i - 1]):
            starts.append(i)

    for start in starts:
        start_is_unvoiced = bool(
            (z[start] >= z_hi * 0.94)
            and (h[start] >= h_hi * 0.82 or f[start] >= f_hi * 0.88)
        )
        local_lookback = min(lookback, 2 if start_is_unvoiced else 1)
        if local_lookback <= 0:
            continue

        misses = 0
        left = max(0, start - local_lookback)
        for i in range(start - 1, left - 1, -1):
            if r[i] < blank_guard:
                break

            next_idx = min(n - 1, i + 1)
            rising = bool(r[next_idx] >= r[i] * 1.06 or f[i] >= f_mid)
            unvoiced_cue = (
                (r[i] >= rms_lo)
                and (z[i] >= z_hi * 0.98)
                and (h[i] >= h_hi * 0.85 or f[i] >= f_hi)
                and rising
            )
            voiced_cue = (
                (r[i] >= rms_voiced)
                and (z[i] <= z_lo)
                and (h[i] >= h_mid or f[i] >= f_mid)
                and rising
            )
            if unvoiced_cue or voiced_cue:
                mask[i] = True
                misses = 0
                continue
            misses += 1
            if misses >= 1:
                break

    return mask


def _align_series_to_frames(
    values: np.ndarray,
    *,
    times_sec: np.ndarray | None,
    frame_count: int,
    hop_sec: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    n = max(0, int(frame_count))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if vals.size <= 0:
        return np.zeros((n,), dtype=np.float32)
    if vals.size == n:
        return np.clip(vals.astype(np.float32), 0.0, 1.0)

    if times_sec is not None and np.asarray(times_sec).size == vals.size:
        t = np.asarray(times_sec, dtype=np.float32).reshape(-1)
        if t.size >= 2:
            if np.any(np.diff(t) < 0.0):
                t = np.maximum.accumulate(t)
            target_t = (np.arange(n, dtype=np.float32) + 0.5) * float(max(hop_sec, 1e-4))
            interp = np.interp(
                target_t.astype(np.float64),
                t.astype(np.float64),
                vals.astype(np.float64),
                left=float(vals[0]),
                right=float(vals[-1]),
            )
            return np.clip(interp.astype(np.float32), 0.0, 1.0)

    src_x = np.linspace(0.0, 1.0, num=max(1, vals.size), endpoint=True, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=max(1, n), endpoint=True, dtype=np.float64)
    out = np.interp(dst_x, src_x, vals.astype(np.float64))
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _compute_sequence_cvn_consonant_prob(
    *,
    wav_path: str,
    language: str,
    frame_ms: int,
    hop_ms: int,
    frame_count: int,
    hop_sec: float,
    backend_override: str | None = None,
    smooth_window: int | None = None,
    min_run_frames: int | None = None,
    c_threshold: float | None = None,
    callback=None,
) -> Tuple[np.ndarray | None, str]:
    if not _seq_cvn_enabled():
        return None, "disabled"

    try:
        from core.cvn_classifier import predict_cvn
        from core.cvn_feature_extractor import extract_frame_features
    except Exception as exc:
        _emit(callback, f"[SEQ][CVN] import failed: {exc}")
        return None, "import_failed"

    model_path = _resolve_cvn_model_path(language)
    backend_seed = str(backend_override or "").strip().lower()
    backend = str(os.environ.get("UTOA_SEQUENCE_CVN_BACKEND", backend_seed) or "").strip().lower()
    if backend not in {"rule", "logreg", "gbdt"}:
        backend = "logreg" if model_path else "rule"

    smooth_window_val = _env_int(
        "UTOA_SEQUENCE_CVN_SMOOTH_WINDOW",
        int(smooth_window if smooth_window is not None else 5),
        min_value=1,
        max_value=25,
    )
    min_run_frames_val = _env_int(
        "UTOA_SEQUENCE_CVN_MIN_RUN_FRAMES",
        int(min_run_frames if min_run_frames is not None else 3),
        min_value=1,
        max_value=40,
    )
    c_threshold_raw = str(os.environ.get("UTOA_SEQUENCE_CVN_C_THRESHOLD", "") or "").strip()
    if c_threshold_raw:
        try:
            c_threshold_val = float(c_threshold_raw)
        except Exception:
            c_threshold_val = None
    else:
        c_threshold_val = c_threshold

    try:
        features = extract_frame_features(
            wav_path,
            frame_ms=float(frame_ms),
            hop_ms=float(hop_ms),
        )
        posterior = predict_cvn(
            features,
            model_path=model_path,
            backend=backend,  # type: ignore[arg-type]
            smooth_window=int(smooth_window_val),
            min_run_frames=int(min_run_frames_val),
            c_threshold=c_threshold_val,
        )
    except Exception as exc:
        _emit(callback, f"[SEQ][CVN] inference failed: {exc}")
        return None, "infer_failed"

    try:
        order = [str(x) for x in list(getattr(posterior, "label_order", ("C", "V")) or ("C", "V"))]
        c_idx = int(order.index("C"))
        probs = np.asarray(getattr(posterior, "probs"), dtype=np.float32)
        if probs.ndim != 2 or probs.shape[0] <= 0 or probs.shape[1] <= c_idx:
            return None, "invalid_posterior"
        c_prob = np.asarray(probs[:, c_idx], dtype=np.float32)
        t_sec = np.asarray(getattr(posterior, "times_ms"), dtype=np.float32).reshape(-1) / 1000.0
        aligned = _align_series_to_frames(
            c_prob,
            times_sec=t_sec,
            frame_count=int(frame_count),
            hop_sec=float(hop_sec),
        )
        model_name = "learned" if model_path else "rule"
        return aligned, f"{model_name}/{backend}"
    except Exception as exc:
        _emit(callback, f"[SEQ][CVN] posterior parse failed: {exc}")
        return None, "parse_failed"


def _resolve_silence_threshold(
    values: np.ndarray,
    *,
    base_quantile: float,
    scale: float,
    min_active_ratio: float,
    max_active_ratio: float,
) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return 1e-5
    q = float(np.clip(base_quantile, 0.02, 0.70))
    noise_q = float(np.clip(q * 0.55, 0.02, 0.40))
    speech_q = float(np.clip(max(0.55, q + 0.34), 0.55, 0.98))
    base = float(np.quantile(arr, q)) * float(scale)
    noise = float(np.quantile(arr, noise_q))
    speech = float(np.quantile(arr, speech_q))
    threshold = max(1e-6, base, noise + max(1e-8, (speech - noise)) * 0.18)

    lo = float(np.clip(min_active_ratio, 0.05, 0.70))
    hi = float(np.clip(max_active_ratio, lo + 0.05, 0.98))
    for _ in range(8):
        active_ratio = float(np.mean(arr >= threshold))
        if active_ratio > hi:
            threshold *= 1.13
            continue
        if active_ratio < lo:
            threshold *= 0.84
            continue
        break
    return float(max(1e-6, threshold))


def _smooth_activity_mask(
    values: np.ndarray,
    *,
    threshold: float,
    min_active_frames: int,
    max_gap_frames: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return np.zeros((0,), dtype=bool)
    mask = np.asarray(arr >= float(threshold), dtype=bool)
    n = int(mask.size)
    if n <= 0:
        return mask

    # Fill tiny silence gaps between voiced spans.
    if max_gap_frames > 0:
        i = 0
        while i < n:
            if mask[i]:
                i += 1
                continue
            start = i
            while i < n and (not mask[i]):
                i += 1
            end = i
            left_on = start > 0 and bool(mask[start - 1])
            right_on = end < n and bool(mask[end])
            if left_on and right_on and (end - start) <= max_gap_frames:
                mask[start:end] = True

    # Remove very short active bursts.
    if min_active_frames > 1:
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            while i < n and mask[i]:
                i += 1
            end = i
            if (end - start) < min_active_frames:
                mask[start:end] = False

    return mask


def _estimate_sequence_labels(
    rms: np.ndarray,
    *,
    silence_threshold: float,
    activity_mask: np.ndarray | None = None,
    zcr: np.ndarray | None = None,
    hf_ratio: np.ndarray | None = None,
    flux: np.ndarray | None = None,
) -> List[str]:
    values = np.asarray(rms, dtype=np.float32)
    if values.size <= 0:
        return []

    mask = (
        np.asarray(activity_mask, dtype=bool)
        if activity_mask is not None and np.asarray(activity_mask).size == values.size
        else np.asarray(values >= float(silence_threshold), dtype=bool)
    )
    labels: List[str] = ["sil"] * int(values.size)
    weak_threshold = float(silence_threshold) * 0.72
    has_cues = (
        zcr is not None
        and hf_ratio is not None
        and flux is not None
        and np.asarray(zcr).size == values.size
        and np.asarray(hf_ratio).size == values.size
        and np.asarray(flux).size == values.size
    )
    if has_cues:
        z = np.asarray(zcr, dtype=np.float32)
        h = np.asarray(hf_ratio, dtype=np.float32)
        f = np.asarray(flux, dtype=np.float32)
        z_hi = float(np.quantile(z, 0.68))
        z_lo = float(np.quantile(z, 0.40))
        h_hi = float(np.quantile(h, 0.66))
        h_mid = float(np.quantile(h, 0.52))
        f_hi = float(np.quantile(f, 0.70))
        f_mid = float(np.quantile(f, 0.55))
    else:
        z = np.zeros_like(values)
        h = np.zeros_like(values)
        f = np.zeros_like(values)
        z_hi = z_lo = h_hi = h_mid = f_hi = f_mid = 0.0

    idx = 0
    total = int(values.size)
    while idx < total:
        val = float(values[idx])
        if not bool(mask[idx]):
            if val < weak_threshold:
                labels[idx] = "sil"
            else:
                labels[idx] = "weak_noise"
            idx += 1
            continue
        if val < weak_threshold:
            labels[idx] = "sil"
            idx += 1
            continue

        start = idx
        while idx < total and bool(mask[idx]):
            idx += 1
        end = idx
        span = max(1, end - start)

        onset_len = max(1, int(round(span * 0.14)))
        trans_len = max(1, int(round(span * 0.22)))
        tail_len = max(1, int(round(span * 0.18)))

        for frame_idx in range(start, end):
            rel = frame_idx - start
            remain = end - frame_idx
            if rel < onset_len:
                if has_cues:
                    unvoiced_cue = (
                        (z[frame_idx] >= z_hi)
                        and (h[frame_idx] >= h_hi * 0.85 or f[frame_idx] >= f_hi)
                        and (values[frame_idx] >= weak_threshold * 0.94)
                    )
                    voiced_cue = (
                        (z[frame_idx] <= z_lo)
                        and (h[frame_idx] >= h_mid or f[frame_idx] >= f_mid)
                        and (values[frame_idx] >= weak_threshold)
                    )
                    if unvoiced_cue:
                        labels[frame_idx] = "unvoiced_onset"
                    elif voiced_cue:
                        labels[frame_idx] = "voiced_onset"
                    else:
                        labels[frame_idx] = "voiced_onset"
                else:
                    labels[frame_idx] = "voiced_onset"
            elif rel < onset_len + trans_len:
                labels[frame_idx] = "vowel_transition"
            elif remain <= tail_len:
                labels[frame_idx] = "tail"
            else:
                labels[frame_idx] = "stable_vowel"

        if has_cues:
            if (z[start] >= z_hi * 0.98 and (h[start] >= h_hi * 0.84 or f[start] >= f_hi)) and values[start] >= weak_threshold * 0.92:
                labels[start] = "unvoiced_onset"
        elif span >= 2 and float(values[start]) < float(values[min(start + 1, end - 1)]) * 0.78:
            labels[start] = "unvoiced_onset"
        if start > 0 and labels[start - 1] not in {"sil", "weak_noise", "breath"}:
            labels[start] = "bridge"

    return labels


def _sanitize_rows(
    rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    filler_mark: str,
    merge_same_mark: bool = True,
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
        if merge_same_mark and abs(start - p_end) <= 1e-4 and mark == p_mark:
            compact[-1] = (p_start, max(p_end, end), p_mark)
        else:
            compact.append((start, end, mark))
    return compact


def _normalize_sequence_format_hint(format_hint: str) -> str:
    raw = str(format_hint or "").strip().lower()
    if not raw:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    parts = {part for part in compact.split("_") if part}
    parts.add(compact)
    if "cvvc" in parts or compact.endswith("cvvc"):
        return "cvvc"
    if "vcv" in parts or compact.endswith("vcv"):
        return "vcv"
    if "cvc" in parts or compact.endswith("cvc"):
        return "cvc"
    if "cv" in parts or compact.endswith("cv"):
        return "cv"
    return compact


def _sequence_word_weight(language: str, token: str, format_hint: str = "") -> float:
    text = str(token or "").strip().lower()
    if not text:
        return 1.0
    lang = str(language or "").strip().lower()
    if lang == "korean":
        onset, vowel, coda = _split_kr_syllable_parts(text)
        if vowel:
            weight = 1.0
            if coda:
                weight += 0.08
            if vowel in {"ya", "yae", "yeo", "ye", "wa", "wae", "wo", "we", "wi", "yu", "ui", "eui", "yi", "weo"}:
                weight += 0.04
            if onset in {"ch", "jj", "kk", "tt", "pp", "ss", "gg", "dd", "bb"}:
                weight += 0.03
            return float(np.clip(weight, 0.82, 1.18))
        return 0.72 if text in {"m", "n", "ng", "l", "r"} else 0.85
    if lang == "japanese":
        onset, vowel = split_ja_romaji_syllable(text)
        if vowel:
            return 1.05 if onset in {"ky", "gy", "sh", "ch", "ny", "hy", "my", "ry", "by", "py"} else 1.0
        return 0.82
    compact = re.sub(r"[^A-Za-z0-9]+", "", text)
    return max(1.0, float(len(compact) if compact else len(text)))


def _resolve_word_rows(
    words: Sequence[str],
    *,
    duration_sec: float,
    labels: Sequence[str],
    hop_sec: float,
    min_span_ms: float,
    language: str = "",
    format_hint: str = "",
) -> List[Tuple[float, float, str]]:
    duration = max(float(duration_sec or 0.0), 0.0)
    if duration <= 0.0:
        return []
    clean_words = [str(w or "").strip() for w in (words or []) if str(w or "").strip()]
    if not clean_words:
        return _sanitize_rows([], duration_sec=duration, filler_mark="spn", merge_same_mark=False)

    min_span_sec = max(0.001, float(min_span_ms) / 1000.0)

    weights: List[float] = []
    for token in clean_words:
        weights.append(float(_sequence_word_weight(language, token, format_hint=format_hint)))
    total_weight = max(float(np.sum(np.asarray(weights, dtype=np.float32))), 1.0)

    active_segments: List[Tuple[float, float]] = []
    if labels and hop_sec > 0.0:
        for idx, label in enumerate(labels):
            lb = str(label or "").strip().lower()
            if lb in {"sil", "weak_noise", "breath"}:
                continue
            seg_start = max(0.0, min(duration, float(idx) * float(hop_sec)))
            seg_end = max(seg_start, min(duration, float(idx + 1) * float(hop_sec)))
            if seg_end - seg_start > 1e-6:
                active_segments.append((seg_start, seg_end))

    if not active_segments:
        active_segments = [(0.0, duration)]
    active_start = float(active_segments[0][0])
    active_end = float(active_segments[-1][1])
    span_total = float(np.sum([end - start for start, end in active_segments], dtype=np.float64))
    if span_total < min_span_sec:
        active_segments = [(0.0, duration)]
        active_start = 0.0
        active_end = duration
        span_total = duration
    span_total = max(min_span_sec, float(span_total))

    count = max(1, len(clean_words))
    lang = str(language or "").strip().lower()
    fmt = _normalize_sequence_format_hint(format_hint)
    if lang in {"korean", "japanese"} and fmt in {"vcv", "cvvc"} and count >= 4 and active_end > active_start:
        tail_trim_ratio = _env_float(
            "UTOA_SEQUENCE_ALIGN_MULTI_TAIL_TRIM_RATIO",
            0.90,
            min_value=0.72,
            max_value=1.0,
        )
        span_width = float(active_end) - float(active_start)
        min_keep_end = float(active_start) + max(float(min_span_sec) * float(count), span_width * 0.70)
        tempo_end = max(min_keep_end, float(active_start) + (span_width * float(tail_trim_ratio)))
        tempo_end = min(float(active_end), float(tempo_end))
        if tempo_end < float(active_end) - max(0.030, span_width * 0.025):
            clipped_segments: List[Tuple[float, float]] = []
            for seg_start, seg_end in active_segments:
                if float(seg_start) >= tempo_end:
                    break
                clipped_end = min(float(seg_end), tempo_end)
                if clipped_end > float(seg_start) + 1e-6:
                    clipped_segments.append((float(seg_start), float(clipped_end)))
            if clipped_segments:
                active_segments = clipped_segments
                active_end = float(clipped_segments[-1][1])
                span_total = float(np.sum([end - start for start, end in active_segments], dtype=np.float64))
                span_total = max(min_span_sec, float(span_total))

    min_step_sec = min(min_span_sec, max(0.004, span_total / (float(count) * 3.0)))
    raw = np.asarray(weights, dtype=np.float64) / float(total_weight) * float(span_total)
    spans = np.maximum(raw, float(min_step_sec))
    correction = float(span_total - np.sum(spans))
    if spans.size > 0:
        spans[-1] = max(1e-4, float(spans[-1] + correction))
        if spans[-1] < min_step_sec:
            spans[-1] = min_step_sec

    def _offset_to_time(offset_sec: float) -> float:
        rem = max(0.0, float(offset_sec))
        for seg_start, seg_end in active_segments:
            seg_len = max(0.0, float(seg_end) - float(seg_start))
            if rem <= seg_len + 1e-9:
                return float(seg_start) + rem
            rem -= seg_len
        return float(active_end)

    rows: List[Tuple[float, float, str]] = []
    cursor = active_start
    consumed = 0.0
    for idx, token in enumerate(clean_words):
        if idx == len(clean_words) - 1:
            end = active_end
        else:
            consumed += float(spans[idx])
            end = min(active_end, _offset_to_time(consumed))
        if end <= cursor:
            end = min(active_end, cursor + 0.001)
        rows.append((cursor, end, token))
        cursor = end

    return _sanitize_rows(rows, duration_sec=duration, filler_mark="spn", merge_same_mark=False)


def _insert_internal_pause_rows(
    word_rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    labels: Sequence[str] | None,
    rms: np.ndarray,
    zcr: np.ndarray | None = None,
    hf_ratio: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    ap_mask: np.ndarray | None = None,
    hop_sec: float,
    silence_threshold: float,
    min_pause_ms: float = 20.0,
    search_ms: float = 42.0,
    min_side_ms: float = 10.0,
    energy_pause_scale: float = 0.95,
    hard_silence_scale: float = 0.80,
    unvoiced_guard_scale: float = 0.62,
) -> List[Tuple[float, float, str]]:
    rows = [(float(s), float(e), str(m or "")) for s, e, m in (word_rows or [])]
    if len(rows) <= 1 or hop_sec <= 0.0 or labels is None:
        return rows

    label_seq = [str(lb or "").strip().lower() for lb in labels]
    n = int(len(label_seq))
    if n <= 2:
        return rows
    r = np.asarray(rms, dtype=np.float32)
    if r.size != n:
        return rows

    min_pause_sec = max(0.008, float(min_pause_ms) / 1000.0)
    min_pause_frames = max(1, int(round(min_pause_sec / max(float(hop_sec), 1e-4))))
    search_frames = max(1, int(round((float(search_ms) / 1000.0) / max(float(hop_sec), 1e-4))))
    min_side_sec = max(0.005, float(min_side_ms) / 1000.0)
    silence_like = {"sil", "weak_noise", "breath"}
    energy_pause_scale = float(np.clip(float(energy_pause_scale), 0.70, 1.10))
    hard_silence_scale = float(np.clip(float(hard_silence_scale), 0.50, energy_pause_scale))
    unvoiced_guard_scale = float(np.clip(float(unvoiced_guard_scale), 0.45, 0.95))

    z = np.asarray(zcr, dtype=np.float32).reshape(-1) if zcr is not None else np.zeros((0,), dtype=np.float32)
    h = np.asarray(hf_ratio, dtype=np.float32).reshape(-1) if hf_ratio is not None else np.zeros((0,), dtype=np.float32)
    f = np.asarray(flux, dtype=np.float32).reshape(-1) if flux is not None else np.zeros((0,), dtype=np.float32)
    has_spec = bool(z.size == n and h.size == n and f.size == n)
    if has_spec:
        z_hi = float(np.quantile(z, 0.68))
        h_hi = float(np.quantile(h, 0.66))
        f_hi = float(np.quantile(f, 0.70))
        z_mid = float(np.quantile(z, 0.52))
    else:
        z_hi = h_hi = f_hi = z_mid = 0.0
    ap = np.asarray(ap_mask, dtype=bool).reshape(-1) if ap_mask is not None else np.zeros((0,), dtype=bool)
    has_ap = bool(ap.size == n)

    def _looks_unvoiced_like(idx: int) -> bool:
        if (not has_spec) or idx < 0 or idx >= n:
            return False
        zc = float(z[idx])
        hf = float(h[idx])
        fl = float(f[idx])
        z_gate = max(0.10, z_hi * 0.96)
        hf_gate = max(0.12, h_hi * 0.86)
        hf_flux_gate = max(0.14, h_hi * 0.97)
        flux_gate = max(0.10, f_hi * 0.90)
        z_mid_gate = max(0.10, z_mid * 0.95)
        return bool(
            (zc >= z_gate and hf >= hf_gate)
            or (hf >= hf_flux_gate and fl >= flux_gate and zc >= z_mid_gate)
        )

    def _is_pause_frame(idx: int) -> bool:
        if idx < 0 or idx >= n:
            return False
        if has_ap and bool(ap[idx]):
            return True
        lb = label_seq[idx]
        if lb in silence_like:
            return True
        val = float(r[idx])
        if val <= float(silence_threshold) * hard_silence_scale:
            if _looks_unvoiced_like(idx) and val > float(silence_threshold) * unvoiced_guard_scale:
                return False
            return True
        if val <= float(silence_threshold) * energy_pause_scale:
            if _looks_unvoiced_like(idx):
                return False
            return True
        return False

    updated = list(rows)
    i = 0
    while i < len(updated) - 1:
        left_start, left_end, left_mark = updated[i]
        right_start, right_end, right_mark = updated[i + 1]
        if right_end <= left_start + (min_side_sec * 2.0):
            i += 1
            continue

        if str(left_mark).strip().lower() in _PAUSE_MARKERS or str(right_mark).strip().lower() in _PAUSE_MARKERS:
            i += 1
            continue

        boundary_idx = int(np.floor(float(left_end) / float(hop_sec)))
        boundary_idx = max(0, min(n - 1, boundary_idx))
        scan_start = max(0, boundary_idx - search_frames)
        scan_end = min(n - 1, boundary_idx + search_frames)
        if scan_end <= scan_start:
            i += 1
            continue

        best = None
        run_start = None
        for k in range(scan_start, scan_end + 1):
            if _is_pause_frame(k):
                if run_start is None:
                    run_start = k
                continue
            if run_start is not None:
                s = int(run_start)
                e = int(k)
                run_start = None
                run_len = e - s
                if run_len < min_pause_frames:
                    continue
                dist = 0
                if boundary_idx < s:
                    dist = s - boundary_idx
                elif boundary_idx >= e:
                    dist = boundary_idx - e + 1
                overlaps = bool(s <= boundary_idx < e)
                energy = float(np.median(r[s:e])) if e > s else float(r[min(n - 1, s)])
                if energy > float(silence_threshold) * energy_pause_scale:
                    continue
                if has_spec:
                    unvoiced_ratio = float(np.mean([1.0 if _looks_unvoiced_like(t) else 0.0 for t in range(s, e)]))
                    if unvoiced_ratio >= 0.72:
                        continue
                    if unvoiced_ratio >= 0.40 and energy > float(silence_threshold) * unvoiced_guard_scale:
                        continue
                score = (1000.0 if overlaps else 0.0) + (run_len * 3.0) - (dist * 8.0)
                if best is None or score > best[0]:
                    best = (score, s, e)
        if run_start is not None:
            s = int(run_start)
            e = int(scan_end + 1)
            run_len = e - s
            if run_len >= min_pause_frames:
                dist = 0
                if boundary_idx < s:
                    dist = s - boundary_idx
                elif boundary_idx >= e:
                    dist = boundary_idx - e + 1
                overlaps = bool(s <= boundary_idx < e)
                energy = float(np.median(r[s:e])) if e > s else float(r[min(n - 1, s)])
                if energy <= float(silence_threshold) * energy_pause_scale:
                    skip_candidate = False
                    if has_spec:
                        unvoiced_ratio = float(np.mean([1.0 if _looks_unvoiced_like(t) else 0.0 for t in range(s, e)]))
                        if unvoiced_ratio >= 0.72:
                            skip_candidate = True
                        if unvoiced_ratio >= 0.40 and energy > float(silence_threshold) * unvoiced_guard_scale:
                            skip_candidate = True
                    if not skip_candidate:
                        score = (1000.0 if overlaps else 0.0) + (run_len * 3.0) - (dist * 8.0)
                        if best is None or score > best[0]:
                            best = (score, s, e)

        if best is None:
            i += 1
            continue

        _score, s, e = best
        pause_start = float(max(0.0, min(float(duration_sec), float(s) * float(hop_sec))))
        pause_end = float(max(pause_start, min(float(duration_sec), float(e) * float(hop_sec))))
        pause_start = max(float(left_start) + min_side_sec, pause_start)
        pause_end = min(float(right_end) - min_side_sec, pause_end)
        if pause_end - pause_start < min_pause_sec:
            i += 1
            continue

        if pause_start <= float(left_start) + min_side_sec:
            i += 1
            continue
        if pause_end >= float(right_end) - min_side_sec:
            i += 1
            continue

        updated[i] = (float(left_start), float(pause_start), left_mark)
        updated[i + 1] = (max(float(right_start), float(pause_end)), float(right_end), right_mark)
        updated.insert(i + 1, (float(pause_start), float(pause_end), "spn"))
        i += 2

    return _sanitize_rows(updated, duration_sec=float(duration_sec), filler_mark="spn", merge_same_mark=False)


def _apply_uniform_boundary_advance(
    word_rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    advance_ms: float,
    min_side_ms: float = 8.0,
) -> List[Tuple[float, float, str]]:
    rows = [(float(s), float(e), str(m or "")) for s, e, m in (word_rows or [])]
    if len(rows) <= 1:
        return rows
    adv = float(max(0.0, advance_ms)) / 1000.0
    if adv <= 1e-7:
        return rows

    min_side_sec = max(0.004, float(min_side_ms) / 1000.0)
    updated = list(rows)
    for i in range(len(updated) - 1):
        left_start, left_end, left_mark = updated[i]
        right_start, right_end, right_mark = updated[i + 1]
        if right_end <= left_start + (min_side_sec * 2.0):
            continue
        if str(left_mark).strip().lower() in _PAUSE_MARKERS or str(right_mark).strip().lower() in _PAUSE_MARKERS:
            continue
        cur_boundary = float(left_end)
        low = float(left_start) + min_side_sec
        high = float(right_end) - min_side_sec
        if high <= low:
            continue
        new_boundary = min(high, max(low, cur_boundary - adv))
        if abs(new_boundary - cur_boundary) < 1e-7:
            continue
        updated[i] = (float(left_start), float(new_boundary), left_mark)
        updated[i + 1] = (max(float(right_start), float(new_boundary)), float(right_end), right_mark)
    return _sanitize_rows(updated, duration_sec=float(duration_sec), filler_mark="spn", merge_same_mark=False)


def _refine_word_boundaries_with_onset_cues(
    word_rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    rms: np.ndarray,
    zcr: np.ndarray,
    hf_ratio: np.ndarray,
    flux: np.ndarray,
    labels: Sequence[str] | None = None,
    language: str = "korean",
    hop_sec: float,
    silence_threshold: float,
    min_side_ms: float = 12.0,
    search_ms: float = 42.0,
    drift_guard_level: int = 1,
    drift_guard_max_shift_ms: float = 28.0,
    onset_snap_right_bias: float = 0.0008,
    plosive_onset_lookback_ms: float = 56.0,
    plosive_left_bias_ms: float = 4.0,
    filename_soft_lock: bool = False,
    soft_lock_conf_threshold: float = 0.46,
    soft_lock_max_shift_ms: float = 22.0,
    soft_lock_release_run: int = 3,
    cvn_c_prob: np.ndarray | None = None,
    cvn_assist_mix: float = 0.55,
) -> List[Tuple[float, float, str]]:
    rows = [(float(s), float(e), str(m or "")) for s, e, m in (word_rows or [])]
    if len(rows) <= 1 or hop_sec <= 0.0:
        return rows

    r = np.asarray(rms, dtype=np.float32)
    z = np.asarray(zcr, dtype=np.float32)
    h = np.asarray(hf_ratio, dtype=np.float32)
    f = np.asarray(flux, dtype=np.float32)
    n = int(r.size)
    if n <= 1 or any(arr.size != n for arr in (z, h, f)):
        return rows
    cvn = np.asarray(cvn_c_prob, dtype=np.float32).reshape(-1) if cvn_c_prob is not None else np.zeros((0,), dtype=np.float32)
    has_cvn = bool(cvn.size == n)
    if has_cvn:
        cvn = np.clip(cvn.astype(np.float32), 0.0, 1.0)
        cvn_hi = float(np.quantile(cvn, 0.66))
        cvn_v = float(np.quantile(cvn, 0.40))
    else:
        cvn_hi = 0.0
        cvn_v = 0.0
    cvn_mix = float(np.clip(float(cvn_assist_mix), 0.0, 1.0))

    label_seq: List[str] = []
    if labels is not None:
        label_seq = [str(lb or "").strip().lower() for lb in labels]
    has_label_seq = bool(len(label_seq) == n)

    z_hi = float(np.quantile(z, 0.68))
    z_lo = float(np.quantile(z, 0.40))
    h_hi = float(np.quantile(h, 0.66))
    h_mid = float(np.quantile(h, 0.52))
    f_mid = float(np.quantile(f, 0.55))

    min_side_sec = max(0.005, float(min_side_ms) / 1000.0)
    search_frames = max(1, int(round((float(search_ms) / 1000.0) / float(hop_sec))))
    tail_guard_frames = max(2, int(round(0.030 / max(float(hop_sec), 1e-4))))
    guard_level = max(0, min(3, int(drift_guard_level)))
    max_shift_frames = max(
        1,
        int(round((max(2.0, float(drift_guard_max_shift_ms)) / 1000.0) / max(float(hop_sec), 1e-4))),
    )
    plosive_lookback_frames = max(
        1,
        int(round((max(6.0, float(plosive_onset_lookback_ms)) / 1000.0) / max(float(hop_sec), 1e-4))),
    )
    plosive_left_bias_frames = max(
        0,
        int(round((max(0.0, float(plosive_left_bias_ms)) / 1000.0) / max(float(hop_sec), 1e-4))),
    )
    onset_snap_right_bias = float(np.clip(float(onset_snap_right_bias), -0.004, 0.004))
    soft_lock_enabled = bool(filename_soft_lock and has_label_seq)
    soft_lock_conf_threshold = float(np.clip(float(soft_lock_conf_threshold), 0.05, 0.95))
    soft_lock_max_shift_frames = max(
        1,
        int(round((max(4.0, float(soft_lock_max_shift_ms)) / 1000.0) / max(float(hop_sec), 1e-4))),
    )
    soft_lock_release_run = max(1, int(soft_lock_release_run))
    base_boundaries = [float(row[1]) for row in rows[:-1]]
    soft_lock_active = bool(soft_lock_enabled)
    soft_lock_mismatch_run = 0

    def _boundary_confidence_idx(
        idx: int,
        *,
        right_bucket: str,
        left_limit: int,
        right_limit: int,
    ) -> float:
        i0 = max(0, min(n - 1, int(idx)))
        w0 = max(left_limit, i0 - 2)
        w1 = min(right_limit, i0 + 2)
        if w1 < w0:
            w0, w1 = i0, i0
        onset_labels = {"unvoiced_onset", "voiced_onset"}
        onset_hit = 0.0
        if has_label_seq:
            onset_hit = 1.0 if any(label_seq[t] in onset_labels for t in range(w0, w1 + 1)) else 0.0

        spec_peak = max(
            float(np.max(z[w0 : w1 + 1]) / max(z_hi, 1e-6)) if w1 >= w0 else 0.0,
            float(np.max(h[w0 : w1 + 1]) / max(h_mid, 1e-6)) if w1 >= w0 else 0.0,
            float(np.max(f[w0 : w1 + 1]) / max(f_mid, 1e-6)) if w1 >= w0 else 0.0,
        )
        spec_score = float(np.clip(spec_peak / 1.35, 0.0, 1.0))
        prev_idx = max(0, i0 - 1)
        next_idx = min(n - 1, i0 + 1)
        local_ref = max(1e-6, float(np.median(r[w0 : w1 + 1])) if w1 >= w0 else float(r[i0]))
        rise = float(r[next_idx] - r[prev_idx])
        rise_score = float(np.clip(rise / max(0.001, local_ref * 0.12), 0.0, 1.0))
        if has_cvn:
            cvn_score = float(np.clip(float(np.max(cvn[w0 : w1 + 1])) if w1 >= w0 else float(cvn[i0]), 0.0, 1.0))
        else:
            cvn_score = onset_hit

        conf = (
            0.34 * onset_hit
            + 0.28 * spec_score
            + 0.18 * rise_score
            + 0.20 * cvn_score
        )
        if right_bucket == "voiceless":
            conf += 0.10 * onset_hit + 0.08 * cvn_score
        elif right_bucket == "voiced":
            conf += 0.05 * cvn_score
        if r[i0] <= float(silence_threshold) * 0.94 and spec_score < 0.35:
            conf *= 0.82
        return float(np.clip(conf, 0.0, 1.0))

    def _apply_boundary_candidate(
        i: int,
        *,
        left_start: float,
        right_end: float,
        left_mark: str,
        right_mark: str,
        left_limit: int,
        right_limit: int,
        right_bucket: str,
        new_boundary: float,
    ) -> Tuple[bool, float, int]:
        nonlocal soft_lock_active, soft_lock_mismatch_run

        candidate = float(new_boundary)
        candidate = min(float(right_end) - min_side_sec, max(float(left_start) + min_side_sec, candidate))
        if candidate <= float(left_start) + min_side_sec or candidate >= float(right_end) - min_side_sec:
            return False, float(left_start), max(0, min(n - 2, int(np.floor(float(left_start) / float(hop_sec)))))

        cand_idx = int(np.floor(candidate / float(hop_sec)))
        cand_idx = max(0, min(n - 2, cand_idx))

        if soft_lock_active:
            conf = _boundary_confidence_idx(
                cand_idx,
                right_bucket=right_bucket,
                left_limit=left_limit,
                right_limit=right_limit,
            )
            if conf < soft_lock_conf_threshold and i < len(base_boundaries):
                anchor_idx = int(np.floor(float(base_boundaries[i]) / float(hop_sec)))
                anchor_idx = max(0, min(n - 2, anchor_idx))
                lo_idx = max(left_limit, anchor_idx - soft_lock_max_shift_frames)
                hi_idx = min(right_limit, anchor_idx + soft_lock_max_shift_frames)
                if lo_idx <= hi_idx:
                    clamped_idx = min(hi_idx, max(lo_idx, cand_idx))
                    if clamped_idx != cand_idx:
                        soft_lock_mismatch_run += 1
                        if soft_lock_mismatch_run >= soft_lock_release_run:
                            soft_lock_active = False
                            soft_lock_mismatch_run = 0
                        else:
                            cand_idx = clamped_idx
                    else:
                        soft_lock_mismatch_run = 0
            else:
                soft_lock_mismatch_run = 0

        candidate = float(cand_idx) * float(hop_sec)
        candidate = min(float(right_end) - min_side_sec, max(float(left_start) + min_side_sec, candidate))
        if candidate <= float(left_start) + min_side_sec or candidate >= float(right_end) - min_side_sec:
            return False, float(left_start), max(0, min(n - 2, int(np.floor(float(left_start) / float(hop_sec)))))

        updated[i] = (float(left_start), float(candidate), left_mark)
        updated[i + 1] = (float(candidate), float(right_end), right_mark)
        return True, float(candidate), int(max(0, min(n - 2, np.floor(float(candidate) / float(hop_sec)))))

    updated = list(rows)
    for i in range(len(updated) - 1):
        left_start, left_end, left_mark = updated[i]
        right_start, right_end, right_mark = updated[i + 1]
        if right_end <= left_start + min_side_sec * 2.0:
            continue

        if str(right_mark).strip().lower() in _PAUSE_MARKERS or str(right_mark).strip().lower() in {"spn", "sil"}:
            continue
        if str(left_mark).strip().lower() in _PAUSE_MARKERS:
            continue

        right_onset = _token_onset(language, str(right_mark))
        right_bucket = _onset_bucket(right_onset)
        is_plosive_onset = _is_plosive_onset(right_onset)
        is_h_family = bool(right_onset in {"h", "hy", "f"})
        is_sonorant_onset = bool(right_bucket == "sonorant")
        is_nasalized_g_onset = bool(
            str(language or "").strip().lower() == "japanese" and right_onset in _JP_NASAL_G_ONSETS
        )
        left_vowel_only = _is_vowel_only_token(language, str(left_mark))
        right_vowel_only = _is_vowel_only_token(language, str(right_mark))

        left_limit = int(np.ceil((float(left_start) + min_side_sec) / float(hop_sec)))
        right_limit = int(np.floor((float(right_end) - min_side_sec) / float(hop_sec)))
        if right_limit <= left_limit:
            continue

        boundary_idx = int(np.floor(float(left_end) / float(hop_sec)))
        boundary_idx = max(0, min(n - 2, boundary_idx))

        # Drift guard:
        # If boundary lands inside a decaying vowel tail (low novelty, low rise),
        # move boundary right by the continuation run to avoid cumulative early shift.
        if has_label_seq and guard_level > 0 and (not is_plosive_onset):
            cont_labels = {"tail", "stable_vowel", "vowel_transition"}
            max_t = min(right_limit, boundary_idx + search_frames)
            cont_run = 0
            prev_energy = float(r[boundary_idx])
            if label_seq[boundary_idx] not in cont_labels:
                max_t = boundary_idx
            if guard_level <= 1:
                z_factor = 0.89
                h_factor = 0.98
                decay_factor = 1.003
                flux_factor = 0.95
                min_run = 5
            elif guard_level == 2:
                z_factor = 0.90
                h_factor = 1.00
                decay_factor = 1.005
                flux_factor = 1.00
                min_run = 4
            else:
                z_factor = 0.92
                h_factor = 1.02
                decay_factor = 1.010
                flux_factor = 1.03
                min_run = 3
            for t in range(boundary_idx + 1, max_t + 1):
                lb = label_seq[t]
                if lb not in cont_labels:
                    break
                cur_energy = float(r[t])
                cur_flux = float(f[t])
                vowel_like = bool(z[t] <= z_hi * z_factor and h[t] <= h_mid * h_factor)
                cvn_vowel_like = bool((not has_cvn) or (cvn[t] <= max(0.46, cvn_v * 1.08)))
                decaying = bool(cur_energy <= prev_energy * decay_factor)
                low_novel = bool(cur_flux <= f_mid * flux_factor)
                if vowel_like and cvn_vowel_like and decaying and low_novel:
                    cont_run += 1
                    prev_energy = max(cur_energy, 1e-7)
                    continue
                break
            if cont_run >= min_run:
                drift_idx = min(right_limit, boundary_idx + min(cont_run, max_shift_frames))
                drift_boundary = float(drift_idx) * float(hop_sec)
                drift_boundary = min(
                    float(right_end) - min_side_sec,
                    max(float(left_start) + min_side_sec, drift_boundary),
                )
                if float(left_start) + min_side_sec < drift_boundary < float(right_end) - min_side_sec:
                    moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                        i,
                        left_start=float(left_start),
                        right_end=float(right_end),
                        left_mark=left_mark,
                        right_mark=right_mark,
                        left_limit=left_limit,
                        right_limit=right_limit,
                        right_bucket=right_bucket,
                        new_boundary=float(drift_boundary),
                    )
                    if moved:
                        left_end = float(left_end_new)
                        boundary_idx = int(boundary_idx_new)

        if guard_level >= 2 and (not is_plosive_onset):
            # Tail guard:
            # If the right side still looks like decaying continuation (not a real onset),
            # delay boundary to the first reliable onset cue to avoid early drift.
            l0 = max(left_limit, boundary_idx - 2)
            l1 = max(l0 + 1, min(n, boundary_idx + 1))
            r0 = max(0, boundary_idx + 1)
            r1 = max(r0 + 1, min(n, boundary_idx + 1 + tail_guard_frames))
            left_tail = float(np.median(r[l0:l1])) if l1 > l0 else float(r[boundary_idx])
            right_head = float(np.median(r[r0:r1])) if r1 > r0 else float(r[min(n - 1, boundary_idx + 1)])
            right_flux = float(np.median(f[r0:r1])) if r1 > r0 else float(f[min(n - 1, boundary_idx + 1)])
            right_rise = float(r[min(n - 1, boundary_idx + 1 + tail_guard_frames)] - r[min(n - 1, boundary_idx + 1)])
            right_z = float(np.median(z[r0:r1])) if r1 > r0 else float(z[min(n - 1, boundary_idx + 1)])
            right_h = float(np.median(h[r0:r1])) if r1 > r0 else float(h[min(n - 1, boundary_idx + 1)])
            right_vowel_like = bool(right_z <= z_hi * 0.94 and right_h <= h_mid * 1.03)
            tail_like = bool(
                right_vowel_like
                and ((not has_cvn) or (float(np.median(cvn[r0:r1])) <= max(0.48, cvn_v * 1.10)))
                and right_head <= max(float(silence_threshold) * 1.10, left_tail * 1.08)
                and right_rise < max(0.0020, left_tail * 0.10)
                and right_flux <= f_mid * 1.05
            )
            if tail_like:
                search_tail_end = min(right_limit, boundary_idx + search_frames * 2)
                delayed_idx = -1
                for k in range(boundary_idx + 1, search_tail_end + 1):
                    prev_idx = max(0, k - 1)
                    next_idx = min(n - 1, k + 1)
                    rise = float(r[next_idx] - r[prev_idx])
                    onset_like = bool(
                        (r[k] >= max(float(silence_threshold) * 0.98, left_tail * 1.05))
                        and rise > max(0.0008, left_tail * 0.03)
                        and (f[k] >= f_mid * 1.03 or z[k] >= z_hi * 0.98)
                    )
                    if onset_like:
                        delayed_idx = k
                        break
                if delayed_idx > boundary_idx:
                    delayed_boundary = float(delayed_idx) * float(hop_sec)
                    delayed_boundary = min(
                        float(right_end) - min_side_sec,
                        max(float(left_start) + min_side_sec, delayed_boundary),
                    )
                    if float(left_start) + min_side_sec < delayed_boundary < float(right_end) - min_side_sec:
                        moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                            i,
                            left_start=float(left_start),
                            right_end=float(right_end),
                            left_mark=left_mark,
                            right_mark=right_mark,
                            left_limit=left_limit,
                            right_limit=right_limit,
                            right_bucket=right_bucket,
                            new_boundary=float(delayed_boundary),
                        )
                        if moved:
                            left_end = float(left_end_new)
                            boundary_idx = int(boundary_idx_new)

        # For voiceless CV onset, snap boundary to the start of unvoiced onset
        # so previous vowel does not swallow the consonant span.
        if has_label_seq:
            if right_bucket == "voiceless":
                pre_frames = max(1, int(round((float(search_ms) * 0.85 / 1000.0) / float(hop_sec))))
                post_frames = max(1, int(round((float(search_ms) * 1.20 / 1000.0) / float(hop_sec))))
                scan_start = max(left_limit, boundary_idx - pre_frames)
                scan_end = min(right_limit, boundary_idx + post_frames)
                if scan_end > scan_start:
                    lookahead = max(2, int(round(0.030 / max(float(hop_sec), 1e-4))))
                    candidates: List[int] = []
                    for k in range(scan_start, scan_end + 1):
                        cvn_onset_like = bool(has_cvn and cvn[k] >= max(0.56, cvn_hi * 0.96))
                        if label_seq[k] != "unvoiced_onset" and (not cvn_onset_like):
                            continue
                        if k > 0 and label_seq[k - 1] == "unvoiced_onset":
                            continue
                        tail = min(n - 1, k + lookahead)
                        has_vowel_like = any(
                            label_seq[t] in {"vowel_transition", "stable_vowel", "tail"}
                            for t in range(k, tail + 1)
                        )
                        if (not has_vowel_like) and has_cvn:
                            has_vowel_like = bool(np.min(cvn[k : tail + 1]) <= max(0.44, cvn_v * 1.06))
                        if not has_vowel_like:
                            continue
                        candidates.append(k)
                    if candidates:
                        best_k = min(
                            candidates,
                            key=lambda k: (0 if k >= boundary_idx else 1, abs(int(k) - int(boundary_idx))),
                        )
                        new_boundary = float(best_k) * float(hop_sec)
                        new_boundary = min(
                            float(right_end) - min_side_sec,
                            max(float(left_start) + min_side_sec, new_boundary),
                        )
                        if float(left_start) + min_side_sec < new_boundary < float(right_end) - min_side_sec:
                            moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                                i,
                                left_start=float(left_start),
                                right_end=float(right_end),
                                left_mark=left_mark,
                                right_mark=right_mark,
                                left_limit=left_limit,
                                right_limit=right_limit,
                                right_bucket=right_bucket,
                                new_boundary=float(new_boundary),
                            )
                            if moved:
                                left_end = float(left_end_new)
                                boundary_idx = int(boundary_idx_new)

        # Plosive CV onset:
        # pull boundary slightly earlier to closure/burst onset so the previous vowel
        # does not absorb the next syllable's consonant.
        if is_plosive_onset and has_label_seq:
            pre_frames = max(search_frames, plosive_lookback_frames)
            post_frames = max(1, int(round((float(search_ms) * 0.70 / 1000.0) / max(float(hop_sec), 1e-4))))
            scan_start = max(left_limit, boundary_idx - pre_frames)
            scan_end = min(right_limit, boundary_idx + post_frames)
            if scan_end > scan_start:
                lookahead = max(2, int(round(0.028 / max(float(hop_sec), 1e-4))))
                onset_labels = {"unvoiced_onset", "voiced_onset"}
                candidates: List[int] = []
                cvn_thr = max(0.52, cvn_hi * 0.92)
                for k in range(scan_start, scan_end + 1):
                    onset_like = bool(label_seq[k] in onset_labels)
                    cvn_onset_like = bool(has_cvn and cvn[k] >= cvn_thr)
                    burst_like = bool(
                        f[k] >= f_mid * 0.97
                        or z[k] >= z_hi * 0.93
                        or h[k] >= h_mid * 1.06
                    )
                    if not (onset_like or cvn_onset_like or burst_like):
                        continue
                    if k > scan_start:
                        prev_onset_like = bool(label_seq[k - 1] in onset_labels)
                        prev_cvn_onset_like = bool(has_cvn and cvn[k - 1] >= cvn_thr)
                        if prev_onset_like or prev_cvn_onset_like:
                            continue
                    tail = min(n - 1, k + lookahead)
                    has_vowel_like = any(
                        label_seq[t] in {"vowel_transition", "stable_vowel", "tail"}
                        for t in range(k, tail + 1)
                    )
                    if (not has_vowel_like) and has_cvn:
                        has_vowel_like = bool(np.min(cvn[k : tail + 1]) <= max(0.44, cvn_v * 1.08))
                    if not has_vowel_like:
                        continue
                    candidates.append(k)
                if candidates:
                    target_idx = max(scan_start, boundary_idx - plosive_left_bias_frames)
                    best_k = min(
                        candidates,
                        key=lambda k: (
                            0 if k <= boundary_idx else 1,
                            abs(int(k) - int(target_idx)),
                            abs(int(k) - int(boundary_idx)),
                        ),
                    )
                    plosive_boundary = float(best_k) * float(hop_sec)
                    plosive_boundary = min(
                        float(right_end) - min_side_sec,
                        max(float(left_start) + min_side_sec, plosive_boundary),
                    )
                    if float(left_start) + min_side_sec < plosive_boundary < float(right_end) - min_side_sec:
                        moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                            i,
                            left_start=float(left_start),
                            right_end=float(right_end),
                            left_mark=left_mark,
                            right_mark=right_mark,
                            left_limit=left_limit,
                            right_limit=right_limit,
                            right_bucket=right_bucket,
                            new_boundary=float(plosive_boundary),
                        )
                        if moved:
                            left_end = float(left_end_new)
                            boundary_idx = int(boundary_idx_new)

        # Legato vowel-vowel transition:
        # avoid onset-biased snapping and prefer a local spectral/energy valley near the prior boundary.
        if left_vowel_only and right_vowel_only:
            vv_scan = max(1, int(round((float(search_ms) * 0.75 / 1000.0) / max(float(hop_sec), 1e-4))))
            scan_start = max(left_limit, boundary_idx - vv_scan)
            scan_end = min(right_limit, boundary_idx + vv_scan)
            if scan_end > scan_start:
                ref_l = max(scan_start, boundary_idx - 3)
                ref_r = min(scan_end + 1, boundary_idx + 4)
                local_ref = float(np.median(r[ref_l:ref_r])) if ref_r > ref_l else float(r[boundary_idx])
                local_ref = max(1e-6, local_ref)
                best_idx = -1
                best_score = 1e9
                for k in range(scan_start, scan_end + 1):
                    dist = abs(int(k) - int(boundary_idx))
                    if dist > max_shift_frames:
                        continue
                    score = (
                        1.00 * (float(r[k]) / local_ref)
                        + 0.42 * float(f[k] / max(f_mid, 1e-6))
                        + 0.16 * float(h[k] / max(h_mid, 1e-6))
                        + 0.04 * float(dist)
                    )
                    if score < best_score:
                        best_score = score
                        best_idx = k
                if best_idx >= 0:
                    vv_boundary = float(best_idx) * float(hop_sec)
                    vv_boundary = min(
                        float(right_end) - min_side_sec,
                        max(float(left_start) + min_side_sec, vv_boundary),
                    )
                    if float(left_start) + min_side_sec < vv_boundary < float(right_end) - min_side_sec:
                        moved, _left_end_new, _boundary_idx_new = _apply_boundary_candidate(
                            i,
                            left_start=float(left_start),
                            right_end=float(right_end),
                            left_mark=left_mark,
                            right_mark=right_mark,
                            left_limit=left_limit,
                            right_limit=right_limit,
                            right_bucket=right_bucket,
                            new_boundary=float(vv_boundary),
                        )
                        if moved:
                            continue

        # h/f-like weak unvoiced onset:
        # detect a subtle fricative rise (zcr/hf) even when waveform is continuous.
        if is_h_family:
            hf_scan = max(1, int(round((float(search_ms) * 0.90 / 1000.0) / max(float(hop_sec), 1e-4))))
            scan_start = max(left_limit, boundary_idx - hf_scan)
            scan_end = min(right_limit, boundary_idx + hf_scan)
            if scan_end > scan_start:
                best_idx = -1
                best_score = -1e9
                for k in range(scan_start, scan_end + 1):
                    dist = abs(int(k) - int(boundary_idx))
                    if dist > max_shift_frames:
                        continue
                    prev_idx = max(scan_start, k - 3)
                    local_left = float(np.median(r[prev_idx:k + 1])) if k >= prev_idx else float(r[k])
                    local_left = max(1e-6, local_left)
                    fricative_like = bool(
                        (z[k] >= z_hi * 0.90 or h[k] >= h_hi * 0.86)
                        and f[k] >= f_mid * 0.82
                    )
                    if not fricative_like:
                        continue
                    weak_energy = bool(r[k] <= max(local_left * 0.96, float(silence_threshold) * 1.16))
                    if not weak_energy:
                        continue
                    cvn_bonus = (cvn_mix * 0.32 * float(cvn[k])) if has_cvn else 0.0
                    score = (
                        0.50 * float(z[k] / max(z_hi, 1e-6))
                        + 0.72 * float(h[k] / max(h_mid, 1e-6))
                        + 0.35 * float(f[k] / max(f_mid, 1e-6))
                        - 0.42 * float(r[k] / local_left)
                        + cvn_bonus
                        - 0.06 * float(dist)
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = k
                if best_idx >= 0:
                    h_boundary = float(best_idx) * float(hop_sec)
                    h_boundary = min(
                        float(right_end) - min_side_sec,
                        max(float(left_start) + min_side_sec, h_boundary),
                    )
                    if float(left_start) + min_side_sec < h_boundary < float(right_end) - min_side_sec:
                        moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                            i,
                            left_start=float(left_start),
                            right_end=float(right_end),
                            left_mark=left_mark,
                            right_mark=right_mark,
                            left_limit=left_limit,
                            right_limit=right_limit,
                            right_bucket=right_bucket,
                            new_boundary=float(h_boundary),
                        )
                        if moved:
                            left_end = float(left_end_new)
                            boundary_idx = int(boundary_idx_new)

        # m/n-like sonorant onset (and JP nasalized g/gy):
        # prefer a local low-novelty valley near the current boundary.
        if is_sonorant_onset or is_nasalized_g_onset:
            sn_scan = max(1, int(round((float(search_ms) * 0.75 / 1000.0) / max(float(hop_sec), 1e-4))))
            scan_start = max(left_limit, boundary_idx - sn_scan)
            scan_end = min(right_limit, boundary_idx + sn_scan)
            if scan_end > scan_start:
                ref_l = max(scan_start, boundary_idx - 4)
                ref_r = min(scan_end + 1, boundary_idx + 5)
                local_ref = float(np.median(r[ref_l:ref_r])) if ref_r > ref_l else float(r[boundary_idx])
                local_ref = max(1e-6, local_ref)
                z_cap = z_hi * (0.96 if is_nasalized_g_onset else 0.90)
                h_cap = h_mid * (1.14 if is_nasalized_g_onset else 1.08)
                f_cap = f_mid * (1.16 if is_nasalized_g_onset else 1.10)
                best_idx = -1
                best_score = 1e9
                for k in range(scan_start + 1, scan_end):
                    dist = abs(int(k) - int(boundary_idx))
                    if dist > max_shift_frames:
                        continue
                    prev_idx = max(scan_start, k - 1)
                    next_idx = min(scan_end, k + 1)
                    valley = bool(r[k] <= r[prev_idx] * 1.01 and r[k] <= r[next_idx] * 1.01)
                    if not valley and not (
                        is_nasalized_g_onset and abs(float(r[next_idx] - r[prev_idx])) <= max(0.0010, local_ref * 0.03)
                    ):
                        continue
                    if z[k] > z_cap:
                        continue
                    if h[k] > h_cap:
                        continue
                    if f[k] > f_cap:
                        continue
                    if has_cvn and cvn[k] < max(0.32, cvn_v * 0.86):
                        continue
                    cvn_penalty = (cvn_mix * 0.12 * float(max(0.0, 0.46 - cvn[k]))) if has_cvn else 0.0
                    score = (
                        0.92 * float(r[k] / local_ref)
                        + 0.20 * float(f[k] / max(f_mid, 1e-6))
                        + 0.10 * float(max(0.0, z[k] - z_lo) / max(z_hi, 1e-6))
                        + cvn_penalty
                        + 0.06 * float(dist)
                    )
                    if score < best_score:
                        best_score = score
                        best_idx = k
                if best_idx >= 0:
                    sn_boundary = float(best_idx) * float(hop_sec)
                    sn_boundary = min(
                        float(right_end) - min_side_sec,
                        max(float(left_start) + min_side_sec, sn_boundary),
                    )
                    if float(left_start) + min_side_sec < sn_boundary < float(right_end) - min_side_sec:
                        moved, left_end_new, boundary_idx_new = _apply_boundary_candidate(
                            i,
                            left_start=float(left_start),
                            right_end=float(right_end),
                            left_mark=left_mark,
                            right_mark=right_mark,
                            left_limit=left_limit,
                            right_limit=right_limit,
                            right_bucket=right_bucket,
                            new_boundary=float(sn_boundary),
                        )
                        if moved:
                            left_end = float(left_end_new)
                            boundary_idx = int(boundary_idx_new)

        # If boundary lies in a low-energy valley, pull it right to onset region.
        if not (
            r[boundary_idx] < float(silence_threshold) * 1.12
            and r[min(n - 1, boundary_idx + 1)] < float(silence_threshold) * 1.20
        ):
            continue

        search_start = max(boundary_idx, left_limit)
        search_end = min(right_limit, boundary_idx + search_frames)
        if search_end <= search_start:
            continue

        best_idx = -1
        best_score = -1e9
        for k in range(search_start, search_end + 1):
            if r[k] < float(silence_threshold) * 0.95:
                continue
            prev_idx = max(0, k - 1)
            next_idx = min(n - 1, k + 1)
            rise = float(r[next_idx] - r[prev_idx])
            if rise <= 0.0:
                continue
            onset_like = bool((f[k] >= f_mid) or (z[k] >= z_hi * 0.95))
            if has_cvn and cvn[k] >= max(0.58, cvn_hi * 0.95):
                onset_like = True
            if not onset_like:
                continue
            cvn_bonus = (cvn_mix * 0.30 * float(cvn[k])) if has_cvn else 0.0
            score = (
                rise
                + 0.35 * float(f[k])
                + 0.22 * float(z[k])
                + 0.18 * float(h[k] >= h_mid)
                + cvn_bonus
                + onset_snap_right_bias * float(k - boundary_idx)
            )
            if score > best_score:
                best_score = score
                best_idx = k

        if best_idx <= boundary_idx:
            continue
        new_boundary = float(best_idx) * float(hop_sec)
        new_boundary = min(float(right_end) - min_side_sec, max(float(left_start) + min_side_sec, new_boundary))
        if new_boundary <= float(left_start) + min_side_sec or new_boundary >= float(right_end) - min_side_sec:
            continue

        _apply_boundary_candidate(
            i,
            left_start=float(left_start),
            right_end=float(right_end),
            left_mark=left_mark,
            right_mark=right_mark,
            left_limit=left_limit,
            right_limit=right_limit,
            right_bucket=right_bucket,
            new_boundary=float(new_boundary),
        )

    return _sanitize_rows(updated, duration_sec=float(duration_sec), filler_mark="spn", merge_same_mark=False)


def _split_token_to_phone_parts(language: str, token: str) -> List[str]:
    raw = str(token or "").strip().lower()
    if not raw:
        return []
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        onset, vowel = split_ja_romaji_syllable(raw)
        out = [part for part in (onset, vowel) if str(part or "").strip()]
        if out:
            return out
    if lang == "korean":
        try:
            parts = _split_kr_syllable_parts(raw)
        except Exception:
            parts = ()
        out = [str(part).strip().lower() for part in parts if str(part or "").strip()]
        if out:
            return out

    match = re.match(r"^([^aeiouy]*)([aeiouy].*)$", raw)
    if match and match.group(1) and match.group(2):
        return [match.group(1), match.group(2)]
    return [raw]


def _onset_bucket(onset: str) -> str:
    key = str(onset or "").strip().lower()
    if not key:
        return "none"
    if key in _SONORANT_ONSETS:
        return "sonorant"
    if key in _VOICELESS_ONSETS:
        return "voiceless"
    if key in _VOICED_ONSETS:
        return "voiced"
    return "other"


def _token_onset(language: str, token: str) -> str:
    parts = _split_token_to_phone_parts(language, token)
    if len(parts) >= 2:
        return str(parts[0] or "").strip().lower()
    return ""


def _is_plosive_onset(onset: str) -> bool:
    return str(onset or "").strip().lower() in _PLOSIVE_ONSETS


def _is_vowel_only_token(language: str, token: str) -> bool:
    parts = _split_token_to_phone_parts(language, token)
    if not parts:
        return False
    if len(parts) == 1:
        return True
    onset = str(parts[0] or "").strip().lower()
    return not onset


def _base_cv_split_ratio(onset: str) -> float:
    kind = _onset_bucket(onset)
    if kind == "voiceless":
        return 0.34
    if kind == "voiced":
        return 0.38
    if kind == "sonorant":
        return 0.42
    if kind == "none":
        return 0.12
    return 0.36


def _estimate_cv_split_ratio_from_labels(
    *,
    start_sec: float,
    end_sec: float,
    labels: Sequence[str],
    cvn_c_prob: Sequence[float] | None,
    hop_sec: float,
    default_ratio: float,
) -> float:
    if not labels or hop_sec <= 0.0 or end_sec <= start_sec:
        return float(np.clip(default_ratio, 0.16, 0.58))

    total = int(len(labels))
    i0 = max(0, min(total - 1, int(np.floor(float(start_sec) / float(hop_sec)))))
    i1 = max(i0 + 1, min(total, int(np.ceil(float(end_sec) / float(hop_sec)))))
    seg = [str(x or "").strip().lower() for x in labels[i0:i1]]
    span = max(1, len(seg))
    if span <= 1:
        return float(np.clip(default_ratio, 0.16, 0.58))

    first_stable = None
    for idx, lb in enumerate(seg):
        if lb in {"stable_vowel", "vowel_transition"}:
            first_stable = idx
            break
    if first_stable is None:
        return float(np.clip(default_ratio, 0.16, 0.58))

    cue_ratio = float(first_stable) / float(span)
    cue_ratio = float(np.clip(cue_ratio, 0.16, 0.62))
    cvn_ratio = None
    if cvn_c_prob is not None:
        cvn_arr = np.asarray(cvn_c_prob, dtype=np.float32).reshape(-1)
        if cvn_arr.size == total:
            c_seg = cvn_arr[i0:i1]
            if c_seg.size > 0:
                cutoff = max(0.42, min(0.56, float(np.quantile(c_seg, 0.45))))
                crossing = None
                for idx, val in enumerate(c_seg):
                    if float(val) <= cutoff:
                        crossing = idx
                        break
                if crossing is not None:
                    cvn_ratio = float(np.clip(float(crossing) / float(span), 0.14, 0.64))
    if cvn_ratio is None:
        blended = float(default_ratio) * 0.62 + float(cue_ratio) * 0.38
    else:
        blended = float(default_ratio) * 0.48 + float(cue_ratio) * 0.30 + float(cvn_ratio) * 0.22
    return float(np.clip(blended, 0.16, 0.58))


def _phone_state_of_mark(mark: str) -> str:
    token = str(mark or "").strip().lower()
    if (not token) or token in _PAUSE_MARKERS or token in {"spn", "sil", "pau", "br", "breath"}:
        return "sil"
    if token in {"y", "w"}:
        return "c"
    if re.search(r"[aeiou]", token):
        return "v"
    return "c"


def _build_frame_state_score_vectors(
    labels: Sequence[str],
    cvn_c_prob: Sequence[float] | None,
) -> Dict[str, np.ndarray]:
    label_seq = [str(x or "").strip().lower() for x in (labels or [])]
    n = len(label_seq)
    c_scores = np.zeros((n,), dtype=np.float32)
    v_scores = np.zeros((n,), dtype=np.float32)
    s_scores = np.zeros((n,), dtype=np.float32)
    cvn = np.asarray(cvn_c_prob, dtype=np.float32).reshape(-1) if cvn_c_prob is not None else np.zeros((0,), dtype=np.float32)
    has_cvn = bool(cvn.size == n)

    for i, lb in enumerate(label_seq):
        if lb == "unvoiced_onset":
            c, v, s = 2.1, -0.9, -0.6
        elif lb == "voiced_onset":
            c, v, s = 1.7, -0.5, -0.5
        elif lb == "bridge":
            c, v, s = 0.9, 0.2, -0.4
        elif lb == "vowel_transition":
            c, v, s = -0.4, 1.2, -0.5
        elif lb == "stable_vowel":
            c, v, s = -1.0, 2.0, -0.7
        elif lb == "tail":
            c, v, s = -0.7, 0.9, -0.4
        elif lb == "breath":
            c, v, s = 0.2, -0.2, 1.2
        elif lb == "weak_noise":
            c, v, s = -0.2, -0.8, 1.0
        elif lb == "sil":
            c, v, s = -0.7, -1.2, 2.0
        else:
            c, v, s = -0.2, -0.2, -0.2
        if has_cvn:
            cp = float(np.clip(cvn[i], 0.0, 1.0))
            c += 0.85 * ((cp * 2.0) - 1.0)
            v += 0.85 * (((1.0 - cp) * 2.0) - 1.0)
        c_scores[i] = float(c)
        v_scores[i] = float(v)
        s_scores[i] = float(s)

    return {"c": c_scores, "v": v_scores, "sil": s_scores}


def _build_phone_duration_priors(
    phone_marks: Sequence[str],
    phone_states: Sequence[str],
    *,
    total_frames: int,
    format_hint: str,
    short_wav: bool,
) -> List[Tuple[int, int, float]]:
    count = max(1, len(phone_marks))
    total = max(count, int(total_frames))
    avg = float(total) / float(count)

    mins: List[int] = []
    maxs: List[int] = []
    targets: List[float] = []
    for idx in range(count):
        state = str((phone_states or [""])[idx] or "c").strip().lower()
        mark = str((phone_marks or [""])[idx] or "").strip().lower()
        if state == "v":
            min_f = max(1, int(np.floor(avg * 0.28)))
            max_f = max(min_f + 1, int(np.ceil(avg * (2.00 if short_wav else 2.35))))
            target = avg * 1.15
        elif state == "sil":
            min_f = 1
            max_f = max(min_f + 1, int(np.ceil(avg * (1.55 if short_wav else 1.85))))
            target = avg * 0.72
        else:
            onset_kind = _onset_bucket(mark)
            min_f = max(1, int(np.floor(avg * 0.10)))
            max_scale = 1.15 if short_wav else 1.45
            target_scale = 0.58
            if onset_kind == "sonorant":
                max_scale += 0.25
                target_scale += 0.12
            elif onset_kind == "voiced":
                max_scale += 0.10
                target_scale += 0.06
            if _is_plosive_onset(mark):
                max_scale += 0.08
                target_scale += 0.04
            max_f = max(min_f + 1, int(np.ceil(avg * max_scale)))
            target = avg * target_scale
        mins.append(int(min_f))
        maxs.append(int(max_f))
        targets.append(float(target))

    def _apply_ratio(idx: int, lo: float, hi: float, target_ratio: float | None = None) -> None:
        if idx < 0 or idx >= count:
            return
        lo_abs = max(1, int(round(float(total) * float(lo))))
        hi_abs = max(lo_abs + 1, int(round(float(total) * float(hi))))
        mins[idx] = max(mins[idx], lo_abs)
        maxs[idx] = min(maxs[idx], hi_abs)
        if mins[idx] > maxs[idx]:
            mins[idx] = max(1, min(mins[idx], hi_abs))
            maxs[idx] = max(mins[idx] + 1, hi_abs)
        if target_ratio is not None:
            targets[idx] = float(total) * float(target_ratio)

    fmt = _normalize_sequence_format_hint(format_hint)
    if fmt == "cv" and count == 2 and list(phone_states) == ["c", "v"]:
        c_hi = 0.46 if short_wav else 0.50
        _apply_ratio(0, 0.08, c_hi, 0.34)
        _apply_ratio(1, 1.0 - c_hi, 0.92, 0.66)
    elif fmt == "vc" and count == 2 and list(phone_states) == ["v", "c"]:
        c_hi = 0.44 if short_wav else 0.52
        _apply_ratio(0, 0.46, 0.90, 0.64)
        _apply_ratio(1, 0.10, c_hi, 0.36)
    elif fmt == "cvc" and count == 3 and list(phone_states) == ["c", "v", "c"]:
        c_hi = 0.34 if short_wav else 0.40
        _apply_ratio(0, 0.08, c_hi, 0.24)
        _apply_ratio(1, 0.28, 0.70, 0.52)
        _apply_ratio(2, 0.10, c_hi, 0.24)
    elif fmt == "vcv" and count == 3 and list(phone_states) == ["v", "c", "v"]:
        c_hi = 0.30 if short_wav else 0.36
        _apply_ratio(0, 0.18, 0.48, 0.33)
        _apply_ratio(1, 0.10, c_hi, 0.20)
        _apply_ratio(2, 0.20, 0.58, 0.47)
    elif fmt == "cvvc" and count >= 4:
        if phone_states[0] == "c":
            _apply_ratio(0, 0.06, 0.30 if short_wav else 0.34, 0.16)
        if phone_states[-1] == "c":
            _apply_ratio(count - 1, 0.06, 0.32 if short_wav else 0.38, 0.18)
        mid_idx = [i for i in range(1, count - 1) if phone_states[i] == "v"]
        for i in mid_idx:
            _apply_ratio(i, 0.12, 0.58, 0.22)

    min_sum = int(sum(mins))
    if min_sum > total:
        overflow = min_sum - total
        adjustable = sorted(
            [i for i, v in enumerate(mins) if v > 1],
            key=lambda i: mins[i],
            reverse=True,
        )
        for idx in adjustable:
            if overflow <= 0:
                break
            cut = min(overflow, mins[idx] - 1)
            mins[idx] -= int(cut)
            overflow -= int(cut)

    max_sum = int(sum(maxs))
    if max_sum < total:
        deficit = total - max_sum
        step = max(1, int(np.ceil(float(deficit) / float(count))))
        for idx in range(count):
            maxs[idx] += step
            deficit -= step
            if deficit <= 0:
                break

    out: List[Tuple[int, int, float]] = []
    for idx in range(count):
        min_f = max(1, int(mins[idx]))
        max_f = max(min_f + 1, int(maxs[idx]))
        target = float(np.clip(float(targets[idx]), float(min_f), float(max_f)))
        out.append((min_f, max_f, target))
    return out


def _boundary_search_window_frames(
    *,
    right_state: str,
    right_mark: str,
    hop_sec: float,
    short_wav: bool,
    format_hint: str,
) -> int:
    if hop_sec <= 0.0:
        return 4
    state = str(right_state or "").strip().lower()
    mark = str(right_mark or "").strip().lower()
    base_ms = 22.0 if state == "v" else 34.0
    if state == "sil":
        base_ms = 18.0
    if _is_plosive_onset(mark):
        base_ms += 8.0
    if short_wav:
        base_ms *= 0.72
    if _normalize_sequence_format_hint(format_hint) in {"cv", "vc", "cvc", "vcv", "cvvc"}:
        base_ms *= 0.90
    frames = int(round((float(base_ms) / 1000.0) / max(float(hop_sec), 1e-6)))
    return max(2, min(28, frames))


def _boundary_alignment_score(
    boundary_idx: int,
    *,
    left_state: str,
    right_state: str,
    right_mark: str,
    labels: Sequence[str],
    cvn_c_prob: Sequence[float] | None,
) -> float:
    n = len(labels or [])
    if n <= 0:
        return 0.0
    idx = max(0, min(n - 1, int(boundary_idx)))
    label_seq = [str(x or "").strip().lower() for x in labels]
    cvn = np.asarray(cvn_c_prob, dtype=np.float32).reshape(-1) if cvn_c_prob is not None else np.zeros((0,), dtype=np.float32)
    has_cvn = bool(cvn.size == n)

    left = str(left_state or "").strip().lower()
    right = str(right_state or "").strip().lower()
    mark = str(right_mark or "").strip().lower()
    right_window = label_seq[idx : min(n, idx + 4)]
    left_window = label_seq[max(0, idx - 3) : idx + 1]

    score = 0.0
    if right == "c":
        onset_hits = sum(1 for lb in right_window if lb in {"unvoiced_onset", "voiced_onset", "bridge"})
        score += 0.70 * float(onset_hits)
        if has_cvn:
            cvn_peak = float(np.max(cvn[max(0, idx - 1) : min(n, idx + 2)]))
            score += 1.10 * (cvn_peak - 0.50)
        if _is_plosive_onset(mark):
            score += 0.16
        if "stable_vowel" in right_window:
            score -= 0.28
    elif right == "v":
        vow_hits = sum(1 for lb in right_window if lb in {"vowel_transition", "stable_vowel", "tail"})
        score += 0.60 * float(vow_hits)
        if any(lb in {"unvoiced_onset", "voiced_onset", "bridge"} for lb in left_window):
            score += 0.45
        if has_cvn:
            cvn_local = float(np.mean(cvn[max(0, idx - 1) : min(n, idx + 2)]))
            score += 0.90 * (0.50 - cvn_local)
    else:
        sil_hits = sum(1 for lb in right_window if lb in {"sil", "weak_noise", "breath"})
        score += 0.75 * float(sil_hits)

    if left == "v" and right == "v":
        if label_seq[idx] in {"vowel_transition", "tail"}:
            score += 0.45
        elif label_seq[idx] in {"unvoiced_onset", "voiced_onset"}:
            score -= 0.25
    return float(score)


def _decode_phone_boundaries_viterbi(
    phone_rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    labels: Sequence[str],
    cvn_c_prob: Sequence[float] | None,
    hop_sec: float,
    format_hint: str,
    duration_penalty_scale: float = 0.95,
    short_wav_threshold_ms: float = 430.0,
) -> List[Tuple[float, float, str]]:
    rows = [(float(s), float(e), str(m or "")) for s, e, m in (phone_rows or [])]
    if len(rows) <= 1 or hop_sec <= 0.0:
        return rows
    n_frames = len(labels or [])
    if n_frames <= 2:
        return rows

    phone_count = len(rows)
    marks = [str(m or "").strip() for _, _, m in rows]
    states = [_phone_state_of_mark(m) for m in marks]
    base_boundaries = [int(np.floor(float(rows[0][0]) / float(hop_sec)))]
    for i in range(phone_count - 1):
        b = int(round(float(rows[i][1]) / float(hop_sec)))
        base_boundaries.append(b)
    base_boundaries.append(int(np.ceil(float(rows[-1][1]) / float(hop_sec))))
    start_idx = max(0, min(n_frames - 2, base_boundaries[0]))
    end_idx = max(start_idx + phone_count, min(n_frames, base_boundaries[-1]))
    if end_idx - start_idx < phone_count:
        return rows

    short_wav = bool(float(duration_sec) * 1000.0 <= float(short_wav_threshold_ms) or (end_idx - start_idx) <= 84)
    priors = _build_phone_duration_priors(
        marks,
        states,
        total_frames=int(end_idx - start_idx),
        format_hint=format_hint,
        short_wav=short_wav,
    )
    if len(priors) != phone_count:
        return rows

    state_scores = _build_frame_state_score_vectors(labels, cvn_c_prob)
    cumsum: Dict[str, np.ndarray] = {}
    for key, arr in state_scores.items():
        cumsum[key] = np.concatenate(([0.0], np.cumsum(np.asarray(arr, dtype=np.float64))))

    def _segment_score(phone_idx: int, seg_start: int, seg_end: int) -> float:
        state = str(states[phone_idx] or "c")
        pref = cumsum.get(state, cumsum["c"])
        s = max(0, min(n_frames, int(seg_start)))
        e = max(s, min(n_frames, int(seg_end)))
        base = float(pref[e] - pref[s])
        min_f, max_f, target = priors[phone_idx]
        dur = max(1, int(e - s))
        if dur < int(min_f) or dur > int(max_f):
            return float("-inf")
        penalty = -abs(float(dur) - float(target)) / max(float(target), 1.0)
        return base + float(duration_penalty_scale) * float(penalty)

    candidates: Dict[int, List[int]] = {}
    for j in range(1, phone_count):
        right_state = states[j]
        right_mark = marks[j]
        win = _boundary_search_window_frames(
            right_state=right_state,
            right_mark=right_mark,
            hop_sec=float(hop_sec),
            short_wav=short_wav,
            format_hint=format_hint,
        )
        base = max(start_idx + j, min(end_idx - (phone_count - j), int(base_boundaries[j])))
        lo = max(start_idx + j, base - win)
        hi = min(end_idx - (phone_count - j), base + win)
        if hi < lo:
            return rows
        cands = set(range(lo, hi + 1))
        # Cue-concentrated candidates for stability on short WAVs.
        for k in range(lo, hi + 1):
            lb = str(labels[k] or "").strip().lower()
            if right_state == "c":
                is_onset = lb in {"unvoiced_onset", "voiced_onset", "bridge"}
                if not is_onset and cvn_c_prob is not None:
                    cvn_arr = np.asarray(cvn_c_prob, dtype=np.float32).reshape(-1)
                    if cvn_arr.size == n_frames:
                        is_onset = bool(float(cvn_arr[k]) >= 0.58)
                if is_onset:
                    cands.add(k)
            elif right_state == "v":
                if lb in {"vowel_transition", "stable_vowel", "tail"}:
                    cands.add(k)
            else:
                if lb in {"sil", "weak_noise", "breath"}:
                    cands.add(k)
        candidates[j] = sorted(cands)
        if not candidates[j]:
            return rows

    stage_scores: Dict[int, Dict[int, float]] = {0: {start_idx: 0.0}}
    stage_back: Dict[int, Dict[int, int]] = {}
    for j in range(1, phone_count):
        prev_stage = stage_scores.get(j - 1, {})
        if not prev_stage:
            return rows
        cur_scores: Dict[int, float] = {}
        cur_back: Dict[int, int] = {}
        for cand in candidates[j]:
            best_score = float("-inf")
            best_prev = -1
            for prev_boundary, prev_score in prev_stage.items():
                seg = _segment_score(j - 1, int(prev_boundary), int(cand))
                if not np.isfinite(seg):
                    continue
                trans = _boundary_alignment_score(
                    int(cand),
                    left_state=states[j - 1],
                    right_state=states[j],
                    right_mark=marks[j],
                    labels=labels,
                    cvn_c_prob=cvn_c_prob,
                )
                cur = float(prev_score) + float(seg) + float(trans)
                if cur > best_score:
                    best_score = cur
                    best_prev = int(prev_boundary)
            if best_prev >= 0:
                cur_scores[int(cand)] = float(best_score)
                cur_back[int(cand)] = int(best_prev)
        if not cur_scores:
            return rows
        stage_scores[j] = cur_scores
        stage_back[j] = cur_back

    last_stage = stage_scores.get(phone_count - 1, {})
    if not last_stage:
        return rows
    final_best = float("-inf")
    final_boundary = -1
    for prev_boundary, prev_score in last_stage.items():
        seg = _segment_score(phone_count - 1, int(prev_boundary), int(end_idx))
        if not np.isfinite(seg):
            continue
        cur = float(prev_score) + float(seg)
        if cur > final_best:
            final_best = cur
            final_boundary = int(prev_boundary)
    if final_boundary < 0:
        return rows

    boundaries = [0] * (phone_count + 1)
    boundaries[0] = int(start_idx)
    boundaries[-1] = int(end_idx)
    boundaries[phone_count - 1] = int(final_boundary)
    for j in range(phone_count - 1, 0, -1):
        cur = boundaries[j]
        prev = stage_back.get(j, {}).get(int(cur))
        if prev is None:
            return rows
        boundaries[j - 1] = int(prev)
    boundaries[0] = int(start_idx)

    for i in range(phone_count):
        if boundaries[i + 1] <= boundaries[i]:
            return rows

    refined: List[Tuple[float, float, str]] = []
    for i, mark in enumerate(marks):
        st = max(0, min(n_frames, int(boundaries[i])))
        en = max(st + 1, min(n_frames, int(boundaries[i + 1])))
        start_sec = float(st) * float(hop_sec)
        end_sec = float(en) * float(hop_sec)
        refined.append((start_sec, end_sec, mark))
    return _sanitize_rows(refined, duration_sec=float(duration_sec), filler_mark="pau")


def _build_phone_rows(
    word_rows: Sequence[Tuple[float, float, str]],
    *,
    duration_sec: float,
    language: str,
    labels: Sequence[str] | None = None,
    cvn_c_prob: Sequence[float] | None = None,
    hop_sec: float = 0.0,
    format_hint: str = "",
    viterbi_enable: bool = True,
    viterbi_duration_penalty_scale: float = 0.95,
    viterbi_short_wav_threshold_ms: float = 430.0,
) -> List[Tuple[float, float, str]]:
    rows: List[Tuple[float, float, str]] = []
    for start, end, mark in word_rows:
        token = str(mark or "").strip()
        if token.lower() in _PAUSE_MARKERS or token in {"spn", "sil"}:
            rows.append((float(start), float(end), "pau"))
            continue
        parts = _split_token_to_phone_parts(language, token)
        if len(parts) <= 1:
            rows.append((float(start), float(end), token))
            continue

        if len(parts) == 2:
            onset = str(parts[0] or "").strip().lower()
            default_ratio = _base_cv_split_ratio(onset)
            ratio = _estimate_cv_split_ratio_from_labels(
                start_sec=float(start),
                end_sec=float(end),
                labels=list(labels or []),
                cvn_c_prob=cvn_c_prob,
                hop_sec=float(hop_sec),
                default_ratio=float(default_ratio),
            )
            split = float(start) + max(0.001, (float(end) - float(start)) * float(ratio))
            split = min(float(end) - 0.0005, max(float(start) + 0.0005, split))
            rows.append((float(start), float(split), str(parts[0])))
            rows.append((float(split), float(end), str(parts[1])))
            continue

        span = max(0.001, float(end) - float(start))
        cursor = float(start)
        step = span / float(len(parts))
        for idx, part in enumerate(parts):
            next_cursor = float(end) if idx == len(parts) - 1 else min(float(end), cursor + step)
            rows.append((cursor, next_cursor, str(part)))
            cursor = next_cursor
    rows = _sanitize_rows(rows, duration_sec=duration_sec, filler_mark="pau")
    if (not viterbi_enable) or len(rows) <= 1 or (not labels) or hop_sec <= 0.0:
        return rows
    refined = _decode_phone_boundaries_viterbi(
        rows,
        duration_sec=float(duration_sec),
        labels=labels,
        cvn_c_prob=cvn_c_prob,
        hop_sec=float(hop_sec),
        format_hint=str(format_hint or ""),
        duration_penalty_scale=float(viterbi_duration_penalty_scale),
        short_wav_threshold_ms=float(viterbi_short_wav_threshold_ms),
    )
    if len(refined) != len(rows):
        return rows
    return refined


def _label_distribution(labels: Sequence[str]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for label in labels or []:
        key = str(label or "").strip() or "unknown"
        dist[key] = int(dist.get(key, 0)) + 1
    return dist


def _write_textgrid(
    path: str,
    *,
    phone_rows: Sequence[Tuple[float, float, str]],
    word_rows: Sequence[Tuple[float, float, str]],
    duration_sec: float,
) -> None:
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

    # Metadata tier used by downstream OTO mapping to auto-detect sequence aligner output.
    meta_max_t = max(float(max_t), 0.001)
    meta_tier = textgrid_mod.IntervalTier(name="utoa_meta", minTime=0.0, maxTime=meta_max_t)
    meta_tier.add(0.0, meta_max_t, "aligner=sequence")
    tg.append(meta_tier)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tg.write(path)


def check_sequence_aligner_ready(
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
            "Sequence aligner currently supports Korean/Japanese workflows only.",
            engine="sequence",
            language=lang,
            ready=False,
        )

    if wav_folder and not os.path.isdir(wav_folder):
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"WAV folder not found: {wav_folder}",
            engine="sequence",
            language=lang,
            ready=False,
        )

    wav_files = _list_top_level_wavs(wav_folder)
    if wav_folder and not wav_files:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"No WAV files found: {wav_folder}",
            engine="sequence",
            language=lang,
            ready=False,
        )

    textgrid_mod, textgrid_err = _import_textgrid_module()
    if textgrid_mod is None:
        return make_runtime_report(
            "align",
            ALIGN_NOT_READY,
            f"python-textgrid import failed: {textgrid_err}",
            engine="sequence",
            language=lang,
            ready=False,
        )

    _emit(callback, f"[SEQ] ready: wav_files={len(wav_files)}")
    return make_runtime_report(
        "align",
        OK,
        "Sequence alignment ready",
        engine="sequence",
        language=lang,
        ready=True,
        wav_count=int(len(wav_files)),
    )


def run_sequence_align(
    _sequence_path: str,
    wav_folder: str,
    _dictionary_path: str,
    output_folder: str,
    *,
    language: str = "korean",
    format_hint: str = "",
    callback=None,
) -> Tuple[bool, str]:
    lang = str(language or "").strip().lower() or "korean"
    ready = check_sequence_aligner_ready(language=lang, wav_folder=wav_folder, callback=callback)
    if str(ready.get("code", "")).upper() != OK:
        return False, str(ready.get("message", "sequence aligner not ready") or "sequence aligner not ready")

    wav_files = _list_top_level_wavs(wav_folder)
    if not wav_files:
        return False, f"No WAV files found: {wav_folder}"

    format_norm = _normalize_sequence_format_hint(format_hint)
    _emit(callback, f"[SEQ] start: wav_files={len(wav_files)} format={format_norm or '-'}")

    profile = _load_sequence_profile(lang, callback=callback)

    frame_ms = _env_int(
        "UTOA_SEQUENCE_ALIGN_FRAME_MS",
        _profile_int(profile, "frame_ms", 22),
        min_value=8,
        max_value=80,
    )
    hop_ms = _env_int(
        "UTOA_SEQUENCE_ALIGN_HOP_MS",
        _profile_int(profile, "hop_ms", 5),
        min_value=2,
        max_value=30,
    )
    analysis_dc_remove = _env_bool(
        "UTOA_SEQUENCE_ANALYSIS_DC_REMOVE",
        default=bool(profile.get("analysis_dc_remove", True)),
    )
    analysis_highpass_hz = _env_float(
        "UTOA_SEQUENCE_ANALYSIS_HIGHPASS_HZ",
        _profile_float(profile, "analysis_highpass_hz", 50.0),
        min_value=0.0,
        max_value=120.0,
    )
    analysis_max_gain_db = _env_float(
        "UTOA_SEQUENCE_ANALYSIS_MAX_GAIN_DB",
        _profile_float(profile, "analysis_max_gain_db", 7.0),
        min_value=0.0,
        max_value=12.0,
    )
    analysis_target_rms = _env_float(
        "UTOA_SEQUENCE_ANALYSIS_TARGET_RMS",
        _profile_float(profile, "analysis_target_rms", 0.08),
        min_value=0.01,
        max_value=0.30,
    )
    silence_q = _env_float(
        "UTOA_SEQUENCE_ALIGN_SILENCE_Q",
        _profile_float(profile, "silence_q", 0.18),
        min_value=0.02,
        max_value=0.60,
    )
    silence_scale = _env_float(
        "UTOA_SEQUENCE_ALIGN_SILENCE_SCALE",
        _profile_float(profile, "silence_scale", 0.90),
        min_value=0.40,
        max_value=1.30,
    )
    min_span_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_MIN_SPAN_MS",
        _profile_float(profile, "min_span_ms", 18.0),
        min_value=5.0,
        max_value=250.0,
    )
    min_active_ratio = _env_float(
        "UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MIN",
        _profile_float(profile, "active_ratio_min", 0.20),
        min_value=0.05,
        max_value=0.70,
    )
    max_active_ratio = _env_float(
        "UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MAX",
        _profile_float(profile, "active_ratio_max", 0.90),
        min_value=0.20,
        max_value=0.98,
    )
    min_active_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_MIN_ACTIVE_MS",
        _profile_float(profile, "min_active_ms", 24.0),
        min_value=5.0,
        max_value=400.0,
    )
    max_gap_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_MAX_GAP_MS",
        _profile_float(profile, "max_gap_ms", 28.0),
        min_value=0.0,
        max_value=200.0,
    )
    onset_lookback_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_ONSET_LOOKBACK_MS",
        _profile_float(profile, "onset_lookback_ms", 42.0),
        min_value=5.0,
        max_value=180.0,
    )
    drift_guard_level = _env_int(
        "UTOA_SEQUENCE_ALIGN_DRIFT_GUARD_LEVEL",
        _profile_int(profile, "drift_guard_level", 1),
        min_value=0,
        max_value=3,
    )
    drift_guard_max_shift_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_DRIFT_GUARD_MAX_SHIFT_MS",
        _profile_float(profile, "drift_guard_max_shift_ms", 28.0),
        min_value=4.0,
        max_value=90.0,
    )
    onset_snap_right_bias = _env_float(
        "UTOA_SEQUENCE_ALIGN_ONSET_SNAP_RIGHT_BIAS",
        _profile_float(profile, "onset_snap_right_bias", 0.0008),
        min_value=-0.004,
        max_value=0.004,
    )
    plosive_onset_lookback_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_PLOSIVE_ONSET_LOOKBACK_MS",
        _profile_float(profile, "plosive_onset_lookback_ms", 56.0),
        min_value=6.0,
        max_value=220.0,
    )
    plosive_left_bias_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_PLOSIVE_LEFT_BIAS_MS",
        _profile_float(profile, "plosive_left_bias_ms", 4.0),
        min_value=0.0,
        max_value=24.0,
    )
    boundary_advance_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_BOUNDARY_ADVANCE_MS",
        _profile_float(profile, "boundary_advance_ms", 0.0),
        min_value=0.0,
        max_value=20.0,
    )
    pause_min_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_PAUSE_MIN_MS",
        _profile_float(profile, "pause_min_ms", 20.0),
        min_value=8.0,
        max_value=120.0,
    )
    pause_search_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_PAUSE_SEARCH_MS",
        _profile_float(profile, "pause_search_ms", 42.0),
        min_value=10.0,
        max_value=160.0,
    )
    pause_energy_scale = _env_float(
        "UTOA_SEQUENCE_ALIGN_PAUSE_ENERGY_SCALE",
        _profile_float(profile, "pause_energy_scale", 0.95),
        min_value=0.70,
        max_value=1.10,
    )
    pause_hard_silence_scale = _env_float(
        "UTOA_SEQUENCE_ALIGN_PAUSE_HARD_SILENCE_SCALE",
        _profile_float(profile, "pause_hard_silence_scale", 0.80),
        min_value=0.50,
        max_value=1.00,
    )
    pause_unvoiced_guard_scale = _env_float(
        "UTOA_SEQUENCE_ALIGN_PAUSE_UNVOICED_GUARD_SCALE",
        _profile_float(profile, "pause_unvoiced_guard_scale", 0.62),
        min_value=0.45,
        max_value=0.95,
    )
    ap_detect_enable = _env_bool(
        "UTOA_SEQUENCE_ALIGN_AP_DETECT_ENABLE",
        default=_profile_bool(profile, "ap_detect_enable", True),
    )
    ap_min_run_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_AP_MIN_RUN_MS",
        _profile_float(profile, "ap_min_run_ms", 12.0),
        min_value=2.0,
        max_value=120.0,
    )
    ap_max_gap_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_AP_MAX_GAP_MS",
        _profile_float(profile, "ap_max_gap_ms", 6.0),
        min_value=0.0,
        max_value=48.0,
    )
    filename_soft_lock_enable = _env_bool(
        "UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_ENABLE",
        default=_profile_bool(profile, "filename_soft_lock_enable", True),
    )
    filename_soft_lock_conf_threshold = _env_float(
        "UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_CONF_TH",
        _profile_float(profile, "filename_soft_lock_conf_threshold", 0.46),
        min_value=0.05,
        max_value=0.95,
    )
    filename_soft_lock_max_shift_ms = _env_float(
        "UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_MAX_SHIFT_MS",
        _profile_float(profile, "filename_soft_lock_max_shift_ms", 22.0),
        min_value=4.0,
        max_value=120.0,
    )
    filename_soft_lock_release_run = _env_int(
        "UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_RELEASE_RUN",
        _profile_int(profile, "filename_soft_lock_release_run", 3),
        min_value=1,
        max_value=8,
    )
    cvn_assist_mix = _env_float(
        "UTOA_SEQUENCE_CVN_ASSIST_MIX",
        _profile_float(profile, "cvn_assist_mix", 0.55),
        min_value=0.0,
        max_value=1.0,
    )
    phone_viterbi_enable = _env_bool(
        "UTOA_SEQUENCE_PHONE_VITERBI_ENABLE",
        default=_profile_bool(profile, "phone_viterbi_enable", True),
    )
    phone_viterbi_duration_penalty_scale = _env_float(
        "UTOA_SEQUENCE_PHONE_VITERBI_DURATION_PENALTY",
        _profile_float(profile, "phone_viterbi_duration_penalty_scale", 0.95),
        min_value=0.10,
        max_value=3.00,
    )
    phone_viterbi_short_wav_threshold_ms = _env_float(
        "UTOA_SEQUENCE_PHONE_VITERBI_SHORT_WAV_MS",
        _profile_float(profile, "phone_viterbi_short_wav_threshold_ms", 430.0),
        min_value=120.0,
        max_value=1200.0,
    )
    cvn_backend = str(os.environ.get("UTOA_SEQUENCE_CVN_BACKEND", str(profile.get("cvn_backend", "") or "")) or "").strip().lower()
    if cvn_backend not in {"", "rule", "logreg", "gbdt"}:
        cvn_backend = ""
    cvn_smooth_window = _env_int(
        "UTOA_SEQUENCE_CVN_SMOOTH_WINDOW",
        _profile_int(profile, "cvn_smooth_window", 5),
        min_value=1,
        max_value=25,
    )
    cvn_min_run_frames = _env_int(
        "UTOA_SEQUENCE_CVN_MIN_RUN_FRAMES",
        _profile_int(profile, "cvn_min_run_frames", 3),
        min_value=1,
        max_value=40,
    )
    cvn_c_threshold = None
    if "cvn_c_threshold" in profile:
        try:
            cvn_c_threshold = float(profile.get("cvn_c_threshold"))
        except Exception:
            cvn_c_threshold = None
    env_c_thr = str(os.environ.get("UTOA_SEQUENCE_CVN_C_THRESHOLD", "") or "").strip()
    if env_c_thr:
        try:
            cvn_c_threshold = float(env_c_thr)
        except Exception:
            cvn_c_threshold = None
    if cvn_c_threshold is not None:
        cvn_c_threshold = float(np.clip(float(cvn_c_threshold), 0.01, 0.99))

    os.makedirs(output_folder, exist_ok=True)

    ok_count = 0
    errors: List[str] = []

    for wav_path in wav_files:
        wav_name = os.path.basename(wav_path)
        try:
            words, source = _load_transcript_words(wav_path, lang)
            if not words:
                words = ["spn"]
                source = "fallback"
            filename_words = _filename_words_for_wav(wav_path, lang)
            token_count = max(1, len(words))
            local_min_active_ratio = float(min_active_ratio)
            local_max_active_ratio = float(max_active_ratio)
            if token_count <= 2:
                local_min_active_ratio = max(
                    local_min_active_ratio,
                    _env_float(
                        "UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MIN",
                        _profile_float(profile, "mono_active_ratio_min", 0.55),
                        min_value=0.10,
                        max_value=0.95,
                    ),
                )
                local_max_active_ratio = max(
                    local_max_active_ratio,
                    _env_float(
                        "UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MAX",
                        _profile_float(profile, "mono_active_ratio_max", 0.95),
                        min_value=0.20,
                        max_value=0.99,
                    ),
                )
            elif token_count >= 6:
                local_max_active_ratio = min(
                    local_max_active_ratio,
                    _env_float(
                        "UTOA_SEQUENCE_ALIGN_MULTI_ACTIVE_RATIO_MAX",
                        _profile_float(profile, "multi_active_ratio_max", 0.88),
                        min_value=0.20,
                        max_value=0.98,
                    ),
                )

            signal, sr, duration_sec = _read_pcm_as_float_mono(wav_path)
            if signal.size <= 0 or float(duration_sec) <= 0.0:
                raise RuntimeError("empty waveform")
            analysis_signal = _prepare_analysis_signal(
                signal,
                sr,
                dc_remove=bool(analysis_dc_remove),
                highpass_hz=float(analysis_highpass_hz),
                max_gain_db=float(analysis_max_gain_db),
                target_rms=float(analysis_target_rms),
            )

            rms, hop_sec = _framewise_rms(analysis_signal, sr, frame_ms=frame_ms, hop_ms=hop_ms)
            if rms.size <= 0 or hop_sec <= 0.0:
                raise RuntimeError("empty frame sequence")
            zcr, hf_ratio, flux = _framewise_consonant_cues(analysis_signal, sr, frame_ms=frame_ms, hop_ms=hop_ms)

            silence_threshold = _resolve_silence_threshold(
                rms,
                base_quantile=silence_q,
                scale=silence_scale,
                min_active_ratio=local_min_active_ratio,
                max_active_ratio=local_max_active_ratio,
            )
            min_active_frames = max(1, int(round((float(min_active_ms) / 1000.0) / max(hop_sec, 1e-4))))
            max_gap_frames = max(0, int(round((float(max_gap_ms) / 1000.0) / max(hop_sec, 1e-4))))
            voicing_mask = _build_frame_voicing_mask(
                rms,
                silence_threshold=float(silence_threshold),
                min_active_frames=int(min_active_frames),
                max_gap_frames=int(max_gap_frames),
                zcr=zcr,
                hf_ratio=hf_ratio,
                flux=flux,
                lookback_frames=max(1, int(round((float(onset_lookback_ms) / 1000.0) / max(hop_sec, 1e-4)))),
            )
            ap_mask = None
            if bool(ap_detect_enable):
                ap_mask = _build_ap_sp_mask(
                    rms,
                    silence_threshold=float(silence_threshold),
                    zcr=zcr,
                    hf_ratio=hf_ratio,
                    flux=flux,
                    min_run_frames=max(1, int(round((float(ap_min_run_ms) / 1000.0) / max(hop_sec, 1e-4)))),
                    max_gap_frames=max(0, int(round((float(ap_max_gap_ms) / 1000.0) / max(hop_sec, 1e-4)))),
                )
            labels = _estimate_sequence_labels(
                rms,
                silence_threshold=silence_threshold,
                activity_mask=voicing_mask,
                zcr=zcr,
                hf_ratio=hf_ratio,
                flux=flux,
            )
            labels = _inject_ap_labels(
                labels,
                ap_mask=ap_mask,
                rms=rms,
                silence_threshold=float(silence_threshold),
            )
            cvn_c_prob, cvn_backend = _compute_sequence_cvn_consonant_prob(
                wav_path=wav_path,
                language=lang,
                frame_ms=int(frame_ms),
                hop_ms=int(hop_ms),
                frame_count=int(rms.size),
                hop_sec=float(hop_sec),
                backend_override=(cvn_backend or None),
                smooth_window=int(cvn_smooth_window),
                min_run_frames=int(cvn_min_run_frames),
                c_threshold=cvn_c_threshold,
                callback=callback,
            )

            word_rows = _resolve_word_rows(
                words,
                duration_sec=duration_sec,
                labels=labels,
                hop_sec=hop_sec,
                min_span_ms=min_span_ms,
                language=lang,
                format_hint=format_hint,
            )
            soft_lock_ready = False
            if bool(filename_soft_lock_enable) and filename_words and len(words) >= 2:
                order_match_ratio = _order_match_ratio(lang, words, filename_words)
                soft_lock_ready = bool(source == "filename" or order_match_ratio >= 0.70)
            word_rows = _refine_word_boundaries_with_onset_cues(
                word_rows,
                duration_sec=duration_sec,
                rms=rms,
                zcr=zcr,
                hf_ratio=hf_ratio,
                flux=flux,
                labels=labels,
                language=lang,
                hop_sec=hop_sec,
                silence_threshold=float(silence_threshold),
                drift_guard_level=int(drift_guard_level),
                drift_guard_max_shift_ms=float(drift_guard_max_shift_ms),
                onset_snap_right_bias=float(onset_snap_right_bias),
                plosive_onset_lookback_ms=float(plosive_onset_lookback_ms),
                plosive_left_bias_ms=float(plosive_left_bias_ms),
                filename_soft_lock=bool(soft_lock_ready),
                soft_lock_conf_threshold=float(filename_soft_lock_conf_threshold),
                soft_lock_max_shift_ms=float(filename_soft_lock_max_shift_ms),
                soft_lock_release_run=int(filename_soft_lock_release_run),
                cvn_c_prob=cvn_c_prob,
                cvn_assist_mix=float(cvn_assist_mix),
            )
            word_rows = _insert_internal_pause_rows(
                word_rows,
                duration_sec=duration_sec,
                labels=labels,
                rms=rms,
                zcr=zcr,
                hf_ratio=hf_ratio,
                flux=flux,
                ap_mask=ap_mask,
                hop_sec=hop_sec,
                silence_threshold=float(silence_threshold),
                min_pause_ms=float(pause_min_ms),
                search_ms=float(pause_search_ms),
                energy_pause_scale=float(pause_energy_scale),
                hard_silence_scale=float(pause_hard_silence_scale),
                unvoiced_guard_scale=float(pause_unvoiced_guard_scale),
            )
            word_rows = _apply_uniform_boundary_advance(
                word_rows,
                duration_sec=duration_sec,
                advance_ms=float(boundary_advance_ms),
            )
            phone_rows = _build_phone_rows(
                word_rows,
                duration_sec=duration_sec,
                language=lang,
                labels=labels,
                cvn_c_prob=cvn_c_prob,
                hop_sec=hop_sec,
                format_hint=format_norm,
                viterbi_enable=bool(phone_viterbi_enable),
                viterbi_duration_penalty_scale=float(phone_viterbi_duration_penalty_scale),
                viterbi_short_wav_threshold_ms=float(phone_viterbi_short_wav_threshold_ms),
            )

            textgrid_name = os.path.splitext(wav_name)[0] + ".TextGrid"
            textgrid_path = os.path.join(output_folder, textgrid_name)
            _write_textgrid(
                textgrid_path,
                phone_rows=phone_rows,
                word_rows=word_rows,
                duration_sec=duration_sec,
            )

            dist = _label_distribution(labels)
            _emit(
                callback,
                f"[SEQ] aligned file={wav_name} words={len(words)} source={source} stable={dist.get('stable_vowel', 0)} cvn={cvn_backend}"
                f" lock={'on' if soft_lock_ready else 'off'}",
            )
            ok_count += 1
        except Exception as exc:
            err = f"{wav_name}: {exc}"
            errors.append(err)
            _emit(callback, f"[SEQ] failed: {err}")

    if ok_count <= 0:
        if errors:
            return False, f"Sequence alignment failed: {errors[0]}"
        return False, "Sequence alignment failed."

    summary = f"sequence alignment complete ({ok_count}/{len(wav_files)})"
    if errors:
        summary += f"; {len(errors)} file(s) failed"
    return True, summary


__all__ = [
    "check_sequence_aligner_ready",
    "run_sequence_align",
]
