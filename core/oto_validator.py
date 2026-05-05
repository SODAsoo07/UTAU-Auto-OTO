"""
OTO timing validator using combined signals:
- filename / recording-list order hints
- mel-spectrogram energy and onset cues
- TextGrid phone boundaries
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
import wave
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.textio_utils import read_text_auto
from core.ja_lab_generator import parse_ja_filename
from core.kr_oto_rules import should_ignore_korean_alias, classify_alias as _classify_kr_alias
from core.oto_normalization import normalize_wav_key


SIL_MARKS = {"", "sil", "sp", "spn", "pau"}

# TICKET-007: Module-level filterbank cache to avoid recomputing per file.
_FB_CACHE: Dict[Tuple[int, int, int], "np.ndarray"] = {}


def _cached_mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> "np.ndarray":
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, n_fft, n_mels)
    return _FB_CACHE[key]


def _build_textgrid_index(tg_folder: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not tg_folder or not os.path.exists(tg_folder):
        return index
    for dirpath, _dirnames, filenames in os.walk(tg_folder):
        for f_name in filenames:
            if not f_name.lower().endswith(".textgrid"):
                continue
            base = os.path.splitext(f_name)[0]
            path = os.path.join(dirpath, f_name)
            index.setdefault(base.lower(), path)
            index.setdefault(normalize_wav_key(f_name), path)
            index.setdefault(normalize_wav_key(base + ".wav"), path)
    return index


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0
    m_min = _hz_to_mel(f_min)
    m_max = _hz_to_mel(f_max)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in m_points], dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if right <= left:
            continue
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        for j in range(left, center):
            fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / float(right - center)
    return fb


def _read_wav_mono(wav_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        return None, None, f"WAV read failed: {e}"

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, None, f"Unsupported sample width: {sampwidth}"

    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)

    return data, sr, None


def _mel_activity(audio: np.ndarray, sr: int) -> Dict[str, float]:
    # TICKET-007: Apply duration cap to avoid O(N) overhead on long files.
    import os as _os
    _max_dur_s = float(str(_os.environ.get("UTOA_VALIDATOR_MAX_DURATION_S", "30")).strip() or "30")
    try:
        _max_dur_s = max(1.0, float(_max_dur_s))
    except Exception:
        _max_dur_s = 30.0
    _max_samples = int(_max_dur_s * sr)
    if len(audio) > _max_samples:
        audio = audio[:_max_samples]

    if len(audio) == 0:
        return {
            "active_start_ms": 0.0,
            "active_end_ms": 0.0,
            "duration_ms": 0.0,
            "onset_ms": 0.0,
            "frame_times_ms": np.array([], dtype=np.float64),
            "active_mask": np.array([], dtype=bool),
        }

    n_fft = 1024
    hop = max(1, int(sr * 0.005))
    win = min(n_fft, max(256, int(sr * 0.025)))
    window = np.hanning(win).astype(np.float64)
    # TICKET-007: Use cached filterbank — avoids recomputing per file.
    fb = _cached_mel_filterbank(sr, n_fft, n_mels=40)

    # TICKET-007: Vectorized framing via stride_tricks instead of Python loop.
    audio_f64 = audio.astype(np.float64)
    n_audio = len(audio_f64)
    # Pad audio so the last frame is complete.
    pad_len = win + (n_fft - win) if n_fft > win else win
    audio_padded = np.concatenate([audio_f64, np.zeros(pad_len, dtype=np.float64)])
    n_frames_possible = max((n_audio - win) // hop + 1, 1)
    # Build strided view: shape (n_frames, win)
    try:
        shape = (n_frames_possible, win)
        strides = (audio_padded.strides[0] * hop, audio_padded.strides[0])
        frames_2d = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides)
        # Apply window and zero-pad to n_fft
        frames_win = frames_2d * window[np.newaxis, :]
        if n_fft > win:
            frames_win = np.pad(frames_win, ((0, 0), (0, n_fft - win)))
        # Batch FFT
        specs = np.fft.rfft(frames_win, n=n_fft, axis=1)
        power_2d = specs.real ** 2 + specs.imag ** 2
        mel_2d = power_2d @ fb.T
        mel_e = np.log1p(np.maximum(mel_2d, 0.0)).mean(axis=1)
    except Exception:
        # Fallback to scalar loop if stride tricks fail (e.g., non-contiguous array).
        frames = []
        for st in range(0, max(n_audio - win + 1, 1), hop):
            fr = audio_f64[st:st + win]
            if len(fr) < win:
                fr = np.concatenate([fr, np.zeros(win - len(fr), dtype=np.float64)])
            fr = fr * window
            if n_fft > win:
                fr = np.pad(fr, (0, n_fft - win))
            spec = np.fft.rfft(fr)
            power = spec.real ** 2 + spec.imag ** 2
            mel = fb @ power
            frames.append(np.log1p(np.maximum(mel, 0.0)))
        if not frames:
            duration_ms = n_audio * 1000.0 / sr
            return {
                "active_start_ms": 0.0,
                "active_end_ms": duration_ms,
                "duration_ms": duration_ms,
                "onset_ms": 0.0,
                "frame_times_ms": np.array([], dtype=np.float64),
                "active_mask": np.array([], dtype=bool),
            }
        mel_e = np.array(frames, dtype=np.float64).mean(axis=1)

    if len(mel_e) == 0:
        duration_ms = n_audio * 1000.0 / sr
        return {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "onset_ms": 0.0,
            "frame_times_ms": np.array([], dtype=np.float64),
            "active_mask": np.array([], dtype=bool),
        }

    duration_ms = n_audio * 1000.0 / sr

    p20 = np.percentile(mel_e, 20)
    p95 = np.percentile(mel_e, 95)
    th = p20 + (p95 - p20) * 0.28
    act_idx = np.where(mel_e >= th)[0]
    active_mask = mel_e >= th

    if len(act_idx) == 0:
        active_start = 0.0
        active_end = duration_ms
    else:
        active_start = act_idx[0] * hop * 1000.0 / sr
        active_end = min(duration_ms, (act_idx[-1] + 1) * hop * 1000.0 / sr)

    flux = np.diff(mel_e, prepend=mel_e[0])
    onset_candidates = np.where(flux > np.percentile(flux, 85))[0]
    onset_ms = (onset_candidates[0] * hop * 1000.0 / sr) if len(onset_candidates) else active_start
    frame_times_ms = (np.arange(len(mel_e), dtype=np.float64) * hop * 1000.0 / sr)

    return {
        "active_start_ms": float(active_start),
        "active_end_ms": float(active_end),
        "duration_ms": float(duration_ms),
        "onset_ms": float(onset_ms),
        "frame_times_ms": frame_times_ms,
        "active_mask": active_mask,
    }


def _blank_span_activity_stats(mel_sig: Dict[str, object], start_ms: float, end_ms: float) -> Dict[str, float]:
    stats = {
        "frame_count": 0,
        "blank_ratio": 0.0,
        "active_ratio": 0.0,
        "onset_blank_ratio": 0.0,
    }
    if np is None:
        return stats

    times = mel_sig.get("frame_times_ms")
    active_mask = mel_sig.get("active_mask")
    if times is None or active_mask is None or len(times) == 0 or len(active_mask) != len(times):
        return stats

    start = float(min(start_ms, end_ms))
    end = float(max(start_ms, end_ms))
    if end <= start:
        return stats

    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        left = int(np.searchsorted(times, start))
        right = int(np.searchsorted(times, end))
        left = max(0, min(left, len(times) - 1))
        right = max(0, min(right, len(times) - 1))
        if right < left:
            left, right = right, left
        mask = np.zeros(len(times), dtype=bool)
        mask[left:right + 1] = True

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return stats

    active_sel = np.asarray(active_mask[idxs], dtype=np.float64)
    blank_sel = 1.0 - active_sel
    onset_end = min(end, start + 56.0)
    onset_mask = (times[idxs] >= start) & (times[idxs] <= onset_end)
    if not np.any(onset_mask):
        onset_mask = np.ones(len(idxs), dtype=bool)

    stats["frame_count"] = int(idxs.size)
    stats["active_ratio"] = float(np.mean(active_sel))
    stats["blank_ratio"] = float(np.mean(blank_sel))
    stats["onset_blank_ratio"] = float(np.mean(blank_sel[onset_mask])) if np.any(onset_mask) else 0.0
    return stats


def _should_warn_blank_only_span(alias_type: str, stats: Dict[str, float]) -> bool:
    blank_ratio = float(stats.get("blank_ratio", 0.0))
    active_ratio = float(stats.get("active_ratio", 0.0))
    onset_blank_ratio = float(stats.get("onset_blank_ratio", 0.0))
    if int(stats.get("frame_count", 0) or 0) <= 0:
        return False
    alias_key = str(alias_type or "").strip().lower()
    if alias_key in ("cv", "cv_head", "vv"):
        return (
            (blank_ratio >= 0.94 and active_ratio <= 0.06)
            or (onset_blank_ratio >= 0.92 and active_ratio <= 0.10)
        )
    if alias_key == "vc":
        return blank_ratio >= 0.98 and active_ratio <= 0.03
    return False


def _load_phone_bounds_ms(tg_path: str) -> Tuple[List[float], List[float]]:
    try:
        import textgrid
    except Exception:
        return [], []
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception:
        return [], []

    ph_tier = None
    for t in tg:
        if getattr(t, "name", "") == "phones":
            ph_tier = t
            break
    if not ph_tier:
        return [], []

    starts, ends = [], []
    for it in ph_tier:
        mk = (it.mark or "").strip().lower()
        if mk in SIL_MARKS:
            continue
        starts.append(float(it.minTime * 1000.0))
        ends.append(float(it.maxTime * 1000.0))
    return starts, ends


def _nearest_dist_ms(v: float, points: List[float]) -> float:
    if not points:
        return 99999.0
    return min(abs(v - p) for p in points)


def _parse_oto_line(line: str):
    if "=" not in line:
        return None
    wav, right = line.split("=", 1)
    parts = right.split(",")
    if len(parts) < 6:
        return None
    try:
        alias = parts[0].strip()
        offset = float(parts[1])
        cons = float(parts[2])
        cutoff = float(parts[3])
        pre = float(parts[4])
        ovl = float(parts[5])
    except Exception:
        return None
    return {
        "wav": wav.strip(),
        "alias": alias,
        "offset": offset,
        "cons": cons,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
    }


def _should_skip_validation_row(row: Dict[str, float | str]) -> bool:
    alias = str(row.get("alias", "") or "").strip()
    if not alias:
        return True

    # Generated tail placeholders use a zeroed timing stub such as
    # cons=6, cutoff=-4, pre=0, ovl=0. They are not meaningful timing targets
    # and otherwise dominate validation error counts.
    try:
        cons = float(row.get("cons", 0.0) or 0.0)
        cutoff_abs = abs(float(row.get("cutoff", 0.0) or 0.0))
        pre = float(row.get("pre", 0.0) or 0.0)
        ovl = float(row.get("ovl", 0.0) or 0.0)
    except Exception:
        return False

    return (
        pre <= 0.0
        and ovl <= 0.0
        and 0.0 < cons <= 8.0
        and 0.0 < cutoff_abs <= 8.0
        and cutoff_abs <= cons
    )


def _norm_name(name: str) -> str:
    normalized = normalize_wav_key(name)
    if normalized:
        return normalized
    s = (name or "").strip().lower()
    return unicodedata.normalize("NFKC", s)


def _build_wav_name_index(wav_dir: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    try:
        names = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    except Exception:
        return idx
    for n in names:
        p = os.path.join(wav_dir, n)
        keys = set()
        n0 = n.strip().lower()
        keys.add(n0)
        keys.add(_norm_name(n0))
        if n0.startswith("_"):
            k = n0[1:]
            keys.add(k)
            keys.add(_norm_name(k))
        else:
            k = "_" + n0
            keys.add(k)
            keys.add(_norm_name(k))
        for k in keys:
            if k and k not in idx:
                idx[k] = p
    return idx


def _resolve_wav_path(wav_name: str, wav_dir: str, wav_idx: Dict[str, str]) -> Optional[str]:
    direct = os.path.join(wav_dir, wav_name)
    if os.path.exists(direct):
        return direct

    q = (wav_name or "").strip().lower()
    candidates = [
        q,
        _norm_name(q),
        q[1:] if q.startswith("_") else "_" + q,
        _norm_name(q[1:] if q.startswith("_") else "_" + q),
    ]
    for k in candidates:
        if k in wav_idx and os.path.exists(wav_idx[k]):
            return wav_idx[k]
    return None


def _score_wav_dir_match(wav_names: List[str], cand_dir: str) -> Tuple[int, int]:
    try:
        cand_files = [f for f in os.listdir(cand_dir) if f.lower().endswith(".wav")]
    except Exception:
        return 0, 0
    if not cand_files:
        return 0, 0

    cand_set = set()
    for n in cand_files:
        n0 = n.strip().lower()
        cand_set.add(n0)
        cand_set.add(_norm_name(n0))
        if n0.startswith("_"):
            cand_set.add(n0[1:])
            cand_set.add(_norm_name(n0[1:]))
    m = 0
    for w in wav_names:
        w0 = (w or "").strip().lower()
        if (
            w0 in cand_set
            or _norm_name(w0) in cand_set
            or ("_" + w0) in cand_set
            or _norm_name("_" + w0) in cand_set
        ):
            m += 1
    return m, len(cand_files)


def _iter_dirs_limited(root: str, max_depth: int = 2, max_dirs: int = 500):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return
    stack = [(root, 0)]
    seen = set()
    yielded = 0
    while stack and yielded < max_dirs:
        cur, depth = stack.pop()
        cur_n = os.path.normcase(cur)
        if cur_n in seen:
            continue
        seen.add(cur_n)
        yield cur
        yielded += 1
        if depth >= max_depth:
            continue
        try:
            with os.scandir(cur) as it:
                for ent in it:
                    if ent.is_dir(follow_symlinks=False):
                        stack.append((ent.path, depth + 1))
        except Exception:
            continue


def _find_best_wav_dir_hint(target_wavs: List[str], roots: List[str]) -> Optional[Tuple[str, int, int]]:
    best = None
    best_score = 0
    checked = 0
    for r in roots:
        for d in _iter_dirs_limited(r, max_depth=2, max_dirs=450):
            checked += 1
            m, total = _score_wav_dir_match(target_wavs, d)
            if m <= 0:
                continue
            if m > best_score:
                best_score = m
                best = (d, m, total)
    return best


def _load_recording_list_map(wav_dir: str) -> Dict[str, List[str]]:
    result = {}
    try:
        files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".txt")]
    except Exception:
        return result

    candidates = []
    for f in files:
        lower = f.lower()
        if any(k in lower for k in ["list", "record", "rec", "리스트", "녹음", "録音"]):
            candidates.append(os.path.join(wav_dir, f))
    for path in candidates[:3]:
        text, _, err = read_text_auto(path)
        if err or not text:
            continue
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith(("#", ";", "//")):
                continue
            toks = re.split(r"\s+", s)
            if len(toks) < 2:
                continue
            first = toks[0]
            if first.lower().endswith(".wav"):
                key = os.path.basename(first).lower()
                result[key] = toks[1:]
    return result


def _sequence_mismatch_ratio(expected: List[str], observed: List[str]) -> Tuple[int, int]:
    """
    Returns (matches, total_observed). A low match count indicates order/shift mismatch.
    """
    if not expected or not observed:
        return 0, len(observed)

    exp = [x.strip().lower() for x in expected if x and x.strip()]
    obs = [x.strip().lower() for x in observed if x and x.strip()]
    if not exp or not obs:
        return 0, len(obs)

    cur = 0
    matched = 0
    for tok in obs:
        found = False
        for i in range(cur, len(exp)):
            if exp[i] == tok:
                matched += 1
                cur = i + 1
                found = True
                break
        if not found:
            # try one soft restart to tolerate a single local skip
            for i in range(max(0, cur - 2), len(exp)):
                if exp[i] == tok:
                    matched += 1
                    cur = i + 1
                    break
    return matched, len(obs)


def _alias_observed_units(rows: List[Dict[str, object]], classify_fn, language: str) -> List[str]:
    observed = []
    for row in rows:
        alias = str(row.get("alias", "") or "")
        alias_type = classify_fn(alias)
        # CV/VCV-like aliases best reflect filename/list syllable progression
        if alias_type not in ("cv", "mono", "vcv", "vv"):
            continue
        if language == "japanese":
            units = parse_ja_filename(alias.replace(" ", "_"))
        else:
            units = re.findall(r"[a-z]+", alias.lower())
        if units:
            observed.append(units[-1].lower())
    return observed


def validate_oto_timing(
    wav_dir: str,
    tg_folder: str,
    oto_path: str,
    language: str = "japanese",
    callback=None,
) -> Dict[str, object]:
    """
    Automatic OTO validation by combining:
    filename/list order + mel activity + TextGrid boundaries.
    """
    def log(msg: str):
        if callback:
            callback(msg)

    summary = {
        "total_lines": 0,
        "checked_files": 0,
        "warnings": 0,
        "errors": 0,
        "report_path": "",
        "message": "",
    }

    if np is None:
        summary["errors"] = 1
        summary["message"] = "numpy가 없어 자동 검증을 실행할 수 없습니다."
        log(f"⚠ 자동 검증 스킵: {summary['message']}")
        return summary

    if not os.path.exists(oto_path):
        summary["errors"] = 1
        summary["message"] = f"oto.ini not found: {oto_path}"
        log(f"⚠ 자동 검증 스킵: {summary['message']}")
        return summary

    parsed_lines = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            p = _parse_oto_line(s)
            if not p:
                continue
            if language == "korean" and should_ignore_korean_alias(p.get("alias", "")):
                continue
            if _should_skip_validation_row(p):
                continue
            if p:
                parsed_lines.append(p)
    summary["total_lines"] = len(parsed_lines)
    if not parsed_lines:
        summary["message"] = "검증할 OTO 라인이 없습니다."
        return summary

    by_wav = defaultdict(list)
    for row in parsed_lines:
        by_wav[row["wav"]].append(row)

    if language == "japanese":
        from core.ja_oto_generator import classify_ja_alias as classify_fn
    else:
        from core.oto_generator import classify_alias as classify_fn

    list_map = _load_recording_list_map(wav_dir)
    issues = []
    missing_wavs: List[str] = []
    wav_idx = _build_wav_name_index(wav_dir)
    tg_idx = _build_textgrid_index(tg_folder)
    is_japanese = (language == "japanese")
    # Japanese multi-syllable files naturally have later offsets on subsequent aliases.
    # Keep strict checks for parameter ordering, but relax global-time heuristics.
    offset_head_late_ms = 360.0 if is_japanese else 260.0
    pre_before_margin_ms = 60.0 if is_japanese else 40.0
    pre_after_margin_ms = 220.0 if is_japanese else 100.0
    cutoff_after_margin_ms = 320.0 if is_japanese else 260.0
    # TICKET-003: Tightened Korean threshold from 180ms to 120ms to better
    # reflect MFA resolution and catch boundary drift earlier.
    boundary_dist_warn_ms = 260.0 if is_japanese else 120.0

    for wav_name, rows in by_wav.items():
        wav_path = _resolve_wav_path(wav_name, wav_dir, wav_idx)
        if not wav_path:
            issues.append(("error", wav_name, "", "WAV file missing"))
            missing_wavs.append(wav_name)
            continue

        audio, sr, wav_err = _read_wav_mono(wav_path)
        if wav_err or audio is None:
            issues.append(("error", wav_name, "", f"Audio decode failed: {wav_err}"))
            continue

        mel_sig = _mel_activity(audio, sr)
        base = os.path.splitext(wav_name)[0]
        tg_path = tg_idx.get(base.lower()) or tg_idx.get(normalize_wav_key(wav_name)) or ""
        ph_starts, ph_ends = _load_phone_bounds_ms(tg_path) if os.path.exists(tg_path) else ([], [])
        boundary_check_enabled = bool(ph_starts)
        if is_japanese and ph_starts:
            boundary_check_enabled = len(ph_starts) >= max(4, len(rows) // 3)
        summary["checked_files"] += 1

        pre_abs_seq = []
        anchor_pre_abs_seq = []
        for row_idx, row in enumerate(rows):
            alias = row["alias"]
            alias_type = classify_fn(alias)
            off = row["offset"]
            ovl = off + row["ovl"]
            pre = off + row["pre"]
            cons = off + row["cons"]
            cut = off + abs(row["cutoff"])
            pre_abs_seq.append(pre)
            if alias_type not in ("vc", "vv"):
                anchor_pre_abs_seq.append(pre)

            if not (off <= ovl <= pre <= cons < cut):
                issues.append(("error", wav_name, alias, "Parameter order invalid"))

            # Offset-late check is meaningful mostly for the first alias.
            # Later aliases in multi-mora files are expected to be far from active_start.
            if row_idx == 0 and alias_type not in ("vc", "vv") and off > mel_sig["active_start_ms"] + offset_head_late_ms:
                issues.append(("warn", wav_name, alias, f"Offset seems late ({off:.1f}ms)"))
            early_margin = 120.0
            if is_japanese and alias_type in ("vc", "vv"):
                early_margin = 220.0
            elif is_japanese and alias_type in ("cv_head", "cv", "vcv"):
                early_margin = 160.0
            if off < mel_sig["active_start_ms"] - early_margin:
                issues.append(("warn", wav_name, alias, f"Offset before active region ({off:.1f}ms)"))

            pre_before_margin = pre_before_margin_ms
            if is_japanese and alias_type in ("cv_head", "cv", "vcv"):
                pre_before_margin += 40.0
            if pre < mel_sig["active_start_ms"] - pre_before_margin or pre > mel_sig["active_end_ms"] + pre_after_margin_ms:
                issues.append(("warn", wav_name, alias, f"Preutterance out of active region ({pre:.1f}ms)"))
            if cut > mel_sig["active_end_ms"] + cutoff_after_margin_ms:
                issues.append(("warn", wav_name, alias, f"Cutoff too long after active end ({cut:.1f}ms)"))
            blank_stats = _blank_span_activity_stats(mel_sig, off, cut)
            if _should_warn_blank_only_span(alias_type, blank_stats):
                issues.append(
                    (
                        "warn",
                        wav_name,
                        alias,
                        f"Offset-cutoff span is blank-heavy ({blank_stats['blank_ratio']:.2f})",
                    )
                )

            if boundary_check_enabled:
                if alias_type in ("vc", "vv"):
                    d = _nearest_dist_ms(pre, ph_ends)
                else:
                    d = min(_nearest_dist_ms(pre, ph_starts), _nearest_dist_ms(pre, ph_ends))
                if d > boundary_dist_warn_ms:
                    issues.append(("warn", wav_name, alias, f"Preutterance far from phone boundary ({d:.1f}ms)"))

        seq_for_check = pre_abs_seq
        if is_japanese and len(anchor_pre_abs_seq) >= 4:
            seq_for_check = anchor_pre_abs_seq
        if len(seq_for_check) >= 4:
            inv = 0
            for i in range(1, len(seq_for_check)):
                if seq_for_check[i] + 5 < seq_for_check[i - 1]:
                    inv += 1
            if inv >= max(2, len(seq_for_check) // 4):
                issues.append(("warn", wav_name, "", "Alias timing sequence appears shifted/backward"))

        rec_aliases = list_map.get(wav_name.lower())
        if rec_aliases:
            oto_aliases = [r["alias"] for r in rows]
            if len(rec_aliases) >= 3 and len(oto_aliases) >= 3:
                # 느슨한 순서 검증: 동일 alias 집합의 상대 순서 역전 비율 체크
                pos = {a: i for i, a in enumerate(oto_aliases)}
                pairs = 0
                bad = 0
                for i in range(len(rec_aliases) - 1):
                    a = rec_aliases[i]
                    b = rec_aliases[i + 1]
                    if a in pos and b in pos:
                        pairs += 1
                        if pos[b] < pos[a]:
                            bad += 1
                if pairs >= 3 and bad >= max(2, pairs // 3):
                    issues.append(("warn", wav_name, "", "Recording list order mismatch with OTO aliases"))

        # filename-based hint for Japanese vowel chains
        if language == "japanese":
            s = parse_ja_filename(base)
            if len(s) >= 4 and all(re.match(r"^[aiueo]$", x) for x in s):
                spread = max(pre_abs_seq) - min(pre_abs_seq) if pre_abs_seq else 0
                if spread < 80:
                    issues.append(("warn", wav_name, "", "Vowel-chain file has too little timing spread"))

    warn_count = sum(1 for x in issues if x[0] == "warn")
    err_count = sum(1 for x in issues if x[0] == "error")
    summary["warnings"] = warn_count
    summary["errors"] = err_count

    # If very few/no files could be checked, this is usually a dataset/path mismatch
    # rather than timing quality regression.
    total_wavs = len(by_wav)
    low_match = summary["checked_files"] <= max(3, int(total_wavs * 0.10))
    if low_match and by_wav:
        if summary["checked_files"] == 0:
            msg = (
                "checked_files=0: OTO의 WAV 파일명과 검증 WAV 폴더가 맞지 않습니다. "
                "타이밍 검증이 수행되지 않았습니다."
            )
        else:
            msg = (
                f"WAV 매칭률이 매우 낮습니다 ({summary['checked_files']}/{total_wavs}). "
                "검증 결과가 실제 품질을 반영하지 않을 수 있습니다."
            )
        summary["message"] = msg
        log(f"⚠ {msg}")
        if missing_wavs:
            log(f"   예시 누락 파일: {missing_wavs[0]}")

        roots = []
        for p in [
            wav_dir,
            os.path.dirname(wav_dir),
            tg_folder,
            os.path.dirname(tg_folder),
            os.path.dirname(oto_path),
            os.path.dirname(os.path.dirname(oto_path)),
        ]:
            if p and os.path.isdir(p):
                roots.append(os.path.abspath(p))
        # unique while preserving order
        uniq_roots = []
        seen_roots = set()
        for r in roots:
            rn = os.path.normcase(r)
            if rn in seen_roots:
                continue
            seen_roots.add(rn)
            uniq_roots.append(r)

        hint = _find_best_wav_dir_hint(list(by_wav.keys()), uniq_roots)
        if hint:
            h_dir, h_match, h_total = hint
            log(f"💡 후보 WAV 폴더: {h_dir} (matched {h_match}/{total_wavs}, wavs={h_total})")
            issues.append(("warn", "(validation)", "", f"Suggested WAV folder: {h_dir} ({h_match}/{total_wavs} match)"))

    report_path = oto_path + ".validation.txt"
    summary["report_path"] = report_path
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== OTO Auto Validation Report ===\n")
            f.write(f"oto: {oto_path}\n")
            f.write(f"wav_dir: {wav_dir}\n")
            f.write(f"tg_folder: {tg_folder}\n")
            f.write(f"language: {language}\n")
            f.write(f"total_lines: {summary['total_lines']}\n")
            f.write(f"checked_files: {summary['checked_files']}\n")
            f.write(f"errors: {err_count}\n")
            f.write(f"warnings: {warn_count}\n\n")
            if not issues:
                f.write("No issues detected.\n")
            else:
                for sev, wav_name, alias, msg in issues:
                    f.write(f"[{sev.upper()}] {wav_name}")
                    if alias:
                        f.write(f" :: {alias}")
                    f.write(f" :: {msg}\n")
        log(f"🧪 OTO 자동 검증 완료: error {err_count}, warning {warn_count}")
        log(f"📝 검증 리포트: {report_path}")
    except Exception as e:
        log(f"⚠ 검증 리포트 저장 실패: {e}")

    return summary


# ---------------------------------------------------------------------------
# TICKET-003: Pre-generation TextGrid coverage validation
# ---------------------------------------------------------------------------

def _count_phone_intervals(tg_path: str) -> int:
    """Count non-silence phone intervals in a TextGrid file."""
    try:
        text = read_text_auto(tg_path)
    except Exception:
        return 0
    count = 0
    in_phones = False
    for line in text.splitlines():
        stripped = line.strip()
        if "phones" in stripped.lower() or "phone" in stripped.lower():
            in_phones = True
        if not in_phones:
            continue
        if stripped.startswith("text ="):
            val = stripped[len("text ="):].strip().strip('"')
            if val and val.lower() not in SIL_MARKS:
                count += 1
    return count


def validate_textgrid_coverage(
    tg_folder: str,
    language: str = "korean",
    callback=None,
    coverage_warn_threshold: float = 0.60,
) -> Dict[str, object]:
    """TICKET-003: Validate that TextGrid files contain a reasonable phone count.

    Scans all TextGrid files and computes the ratio of files that have at
    least one non-silence phone interval.  A median coverage below
    ``coverage_warn_threshold`` (default 0.60) is reported as drift.

    Returns a dict with keys:
    - ``coverage_ok`` (bool)
    - ``drift_detected`` (bool)
    - ``median_phone_count`` (float)
    - ``files_with_phones`` (int)
    - ``total_tg_files`` (int)
    - ``coverage_ratio`` (float)
    - ``message`` (str)
    """

    def _log(msg: str) -> None:
        if callback:
            try:
                callback(msg)
            except Exception:
                pass

    result: Dict[str, object] = {
        "coverage_ok": True,
        "drift_detected": False,
        "median_phone_count": 0.0,
        "files_with_phones": 0,
        "total_tg_files": 0,
        "coverage_ratio": 1.0,
        "message": "",
    }

    if not tg_folder or not os.path.isdir(tg_folder):
        result["coverage_ok"] = False
        result["message"] = f"TextGrid folder not found: {tg_folder}"
        return result

    tg_paths = []
    for dirpath, _dirs, filenames in os.walk(tg_folder):
        for fn in filenames:
            if fn.lower().endswith(".textgrid"):
                tg_paths.append(os.path.join(dirpath, fn))

    if not tg_paths:
        result["coverage_ok"] = False
        result["message"] = "No TextGrid files found."
        return result

    phone_counts = []
    for tp in tg_paths:
        phone_counts.append(_count_phone_intervals(tp))

    if np is None:
        import statistics
        median_count = statistics.median(phone_counts) if phone_counts else 0.0
    else:
        median_count = float(np.median(phone_counts)) if phone_counts else 0.0

    files_with_phones = sum(1 for c in phone_counts if c > 0)
    total = len(phone_counts)
    coverage_ratio = files_with_phones / max(total, 1)
    drift = coverage_ratio < float(coverage_warn_threshold)

    result["median_phone_count"] = float(median_count)
    result["files_with_phones"] = files_with_phones
    result["total_tg_files"] = total
    result["coverage_ratio"] = round(coverage_ratio, 4)
    result["drift_detected"] = drift
    result["coverage_ok"] = not drift

    if drift:
        msg = (
            f"[TextGrid Coverage] DRIFT DETECTED: only {files_with_phones}/{total} files "
            f"have phone intervals (ratio={coverage_ratio:.2f} < threshold={coverage_warn_threshold:.2f}). "
            f"Median phone count per file: {median_count:.1f}. "
            "Consider re-running alignment with a higher-quality aligner."
        )
        result["message"] = msg
        _log(f"⚠ {msg}")
    else:
        msg = (
            f"[TextGrid Coverage] OK: {files_with_phones}/{total} files with phones "
            f"(ratio={coverage_ratio:.2f}, median_phones={median_count:.1f})"
        )
        result["message"] = msg
        _log(msg)

    return result


# ---------------------------------------------------------------------------
# TICKET-006: Post-generation alias coverage audit
# ---------------------------------------------------------------------------

# Minimum expected counts per alias family per format.
# Override individual minimums via environment variables.
_ALIAS_MIN_COUNTS_KR_CVVC = {
    "cv": ("UTOA_MIN_CV_COUNT", 10),
    "vc": ("UTOA_MIN_VC_COUNT", 10),
    "vv": ("UTOA_MIN_VV_COUNT", 5),
}
_ALIAS_MIN_COUNTS_JA_VCV = {
    "vcv": ("UTOA_MIN_VCV_COUNT", 20),
    "cv": ("UTOA_MIN_CV_COUNT", 10),
}
_ALIAS_MIN_COUNTS_KR_CV = {
    "cv": ("UTOA_MIN_CV_COUNT", 10),
}
_ALIAS_MIN_COUNTS_JA_CV = {
    "cv": ("UTOA_MIN_CV_COUNT", 10),
}


def _resolve_min_count(env_name: str, default: int) -> int:
    raw = str(os.environ.get(env_name, "")).strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def audit_alias_coverage(
    oto_path: str,
    language: str = "korean",
    format_type: str = "cvvc",
    callback=None,
) -> Dict[str, object]:
    """TICKET-006: Audit alias family coverage in a generated OTO file.

    Classifies each alias and compares per-family counts against minimum
    expected thresholds for the given format type.

    Returns a dict with keys:
    - ``coverage_ok`` (bool)
    - ``missing_families`` (List[str])
    - ``low_count_families`` (Dict[str, int])  # family -> actual count
    - ``family_counts`` (Dict[str, int])
    - ``message`` (str)
    """

    def _log(msg: str) -> None:
        if callback:
            try:
                callback(msg)
            except Exception:
                pass

    result: Dict[str, object] = {
        "coverage_ok": True,
        "missing_families": [],
        "low_count_families": {},
        "family_counts": {},
        "message": "",
    }

    if not oto_path or not os.path.isfile(oto_path):
        result["coverage_ok"] = False
        result["message"] = f"OTO file not found: {oto_path}"
        return result

    lang = str(language or "korean").strip().lower()
    fmt = str(format_type or "cvvc").strip().lower()

    # Determine minimum count table for this format.
    if lang in {"japanese", "ja", "jp"}:
        if "vcv" in fmt:
            min_table = _ALIAS_MIN_COUNTS_JA_VCV
        else:
            min_table = _ALIAS_MIN_COUNTS_JA_CV
        classify_fn = None  # lazy import below
    else:
        if "cvvc" in fmt or "cvc" in fmt:
            min_table = _ALIAS_MIN_COUNTS_KR_CVVC
        else:
            min_table = _ALIAS_MIN_COUNTS_KR_CV
        classify_fn = _classify_kr_alias

    # Lazy import for Japanese classifier to avoid circular imports.
    if classify_fn is None:
        try:
            from core.ja_oto_mapping import classify_ja_alias as _cj
            classify_fn = _cj
        except Exception:
            classify_fn = _classify_kr_alias

    family_counts: Dict[str, int] = {}
    try:
        lines = read_text_auto(oto_path).splitlines()
    except Exception as exc:
        result["coverage_ok"] = False
        result["message"] = f"Failed to read OTO file: {exc}"
        return result

    for line in lines:
        line = line.strip()
        if not line or "=" not in line or "," not in line:
            continue
        left, right = line.split("=", 1)
        parts = right.split(",")
        if not parts:
            continue
        alias = parts[0].strip()
        if not alias:
            continue
        try:
            family = classify_fn(alias)
        except Exception:
            family = "general"
        family_counts[family] = family_counts.get(family, 0) + 1

    result["family_counts"] = dict(family_counts)

    missing = []
    low_count = {}
    for family, (env_name, default_min) in min_table.items():
        required = _resolve_min_count(env_name, default_min)
        actual = family_counts.get(family, 0)
        if actual == 0:
            missing.append(family)
        elif actual < required:
            low_count[family] = actual

    result["missing_families"] = missing
    result["low_count_families"] = low_count
    coverage_ok = not missing and not low_count
    result["coverage_ok"] = coverage_ok

    if not coverage_ok:
        parts_msg = []
        if missing:
            parts_msg.append(f"missing families: {missing}")
        if low_count:
            parts_msg.append(f"low-count families: {low_count}")
        msg = (
            f"[Alias Coverage] ⚠ Coverage issues detected ({fmt} {lang}): "
            + "; ".join(parts_msg)
            + ". Consider using a higher-quality aligner or checking LAB files."
        )
        result["message"] = msg
        _log(f"⚠ {msg}")
    else:
        msg = (
            f"[Alias Coverage] OK ({fmt} {lang}): "
            + ", ".join(f"{k}={v}" for k, v in sorted(family_counts.items()))
        )
        result["message"] = msg
        _log(msg)

    return result
