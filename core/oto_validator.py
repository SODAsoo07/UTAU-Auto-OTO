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
from core.kr_oto_rules import should_ignore_korean_alias
from core.oto_normalization import normalize_wav_key


SIL_MARKS = {"", "sil", "sp", "spn", "pau"}
_VALIDATION_MEL_CACHE: Dict[Tuple[str, int, int, bool], Dict[str, object]] = {}
_VALIDATION_PHONE_CACHE: Dict[Tuple[str, int, int], Tuple[List[float], List[float]]] = {}
_VALIDATION_CACHE_MAX = 512


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(float(raw))
    except Exception:
        return int(default)
    return max(int(min_value), min(int(max_value), value))


def _trim_cache(cache: Dict[object, object], max_size: int = _VALIDATION_CACHE_MAX) -> None:
    if len(cache) <= max_size:
        return
    remove_count = max(1, len(cache) - max_size)
    for key in list(cache.keys())[:remove_count]:
        cache.pop(key, None)


def _sample_items(items: List[Tuple[str, List[Dict[str, object]]]], max_count: int) -> List[Tuple[str, List[Dict[str, object]]]]:
    total = len(items)
    if max_count <= 0 or total <= max_count:
        return items
    if max_count == 1:
        return [items[0]]

    selected = []
    seen = set()
    step = (total - 1) / float(max_count - 1)
    for i in range(max_count):
        idx = int(round(i * step))
        idx = max(0, min(total - 1, idx))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(items[idx])
    if selected[-1][0] != items[-1][0]:
        selected[-1] = items[-1]
    return selected


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    qv = max(0.0, min(100.0, float(q)))
    pos = (len(arr) - 1) * (qv / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    weight = pos - lo
    return (arr[lo] * (1.0 - weight)) + (arr[hi] * weight)


def _init_alias_metric_bucket() -> Dict[str, List[float]]:
    return {
        "offset_late_ms": [],
        "pre_boundary_dist_ms": [],
        "cutoff_overrun_ms": [],
        "blank_ratio": [],
    }


def _append_alias_metric(
    metrics: Dict[str, Dict[str, List[float]]],
    alias_type: str,
    key: str,
    value: float,
) -> None:
    if not key:
        return
    alias = str(alias_type or "unknown").strip().lower() or "unknown"
    bucket = metrics.setdefault(alias, _init_alias_metric_bucket())
    try:
        val = float(value)
    except Exception:
        return
    if not math.isfinite(val):
        return
    if key not in bucket:
        bucket[key] = []
    bucket[key].append(val)


def _summarize_alias_metrics(metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for alias_type, values_by_key in sorted((metrics or {}).items()):
        if not isinstance(values_by_key, dict):
            continue
        metric_out = {}
        for metric_key, values in values_by_key.items():
            if not isinstance(values, list) or not values:
                continue
            metric_out[metric_key] = {
                "count": len(values),
                "p50": round(_percentile(values, 50.0), 3),
                "p90": round(_percentile(values, 90.0), 3),
            }
        if metric_out:
            out[str(alias_type)] = metric_out
    return out


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


def _mel_activity(audio: np.ndarray, sr: int, *, fast: bool = False) -> Dict[str, float]:
    if len(audio) == 0:
        return {
            "active_start_ms": 0.0,
            "active_end_ms": 0.0,
            "duration_ms": 0.0,
            "onset_ms": 0.0,
            "frame_times_ms": np.array([], dtype=np.float64),
            "active_mask": np.array([], dtype=bool),
        }

    n_fft = 512 if fast else 1024
    hop = max(1, int(sr * (0.010 if fast else 0.005)))
    win = min(n_fft, max(256, int(sr * (0.020 if fast else 0.025))))
    window = np.hanning(win).astype(np.float64)
    fb = _mel_filterbank(sr, n_fft, n_mels=24 if fast else 40)

    frames = []
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr = audio[st:st + win]
        if len(fr) < win:
            pad = np.zeros(win - len(fr), dtype=np.float32)
            fr = np.concatenate([fr, pad], axis=0)
        fr = fr.astype(np.float64) * window
        if n_fft > win:
            fr = np.pad(fr, (0, n_fft - win))
        spec = np.fft.rfft(fr)
        power = (spec.real ** 2 + spec.imag ** 2)
        mel = fb @ power
        frames.append(np.log1p(np.maximum(mel, 0.0)))

    if not frames:
        duration_ms = len(audio) * 1000.0 / sr
        return {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "onset_ms": 0.0,
            "frame_times_ms": np.array([], dtype=np.float64),
            "active_mask": np.array([], dtype=bool),
        }

    mel_e = np.array(frames, dtype=np.float64).mean(axis=1)
    duration_ms = len(audio) * 1000.0 / sr

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
        "checked_wavs_total": 0,
        "sampled_wavs": 0,
        "warnings": 0,
        "errors": 0,
        "truncated_issues": 0,
        "report_path": "",
        "message": "",
        "alias_metrics": {},
    }

    fast_mode = _env_bool("UTOA_OTO_AUTO_VALIDATE_FAST", True)
    max_files = _env_int("UTOA_OTO_AUTO_VALIDATE_MAX_FILES", 180, min_value=10, max_value=50000)
    max_issues = _env_int("UTOA_OTO_AUTO_VALIDATE_MAX_ISSUES", 1200, min_value=100, max_value=20000)
    use_textgrid = _env_bool("UTOA_OTO_AUTO_VALIDATE_USE_TEXTGRID", not fast_mode)
    log(
        f"🧪 auto-validate options: fast={int(fast_mode)}, max_files={max_files}, "
        f"max_issues={max_issues}, textgrid={int(use_textgrid)}"
    )

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
    wav_items = sorted(by_wav.items(), key=lambda x: x[0].lower())
    summary["checked_wavs_total"] = len(wav_items)
    sampled = _sample_items(wav_items, max_files) if fast_mode else wav_items
    summary["sampled_wavs"] = len(sampled)
    if len(sampled) < len(wav_items):
        log(
            f"🧪 자동 검증 샘플링 적용: {len(sampled)}/{len(wav_items)} files "
            f"(fast={int(bool(fast_mode))}, max_files={max_files})"
        )

    if language == "japanese":
        from core.ja_oto_generator import classify_ja_alias as classify_fn
    else:
        from core.oto_generator import classify_alias as classify_fn

    list_map = _load_recording_list_map(wav_dir) if not fast_mode else {}
    issues = []
    alias_metric_values: Dict[str, Dict[str, List[float]]] = {}
    warn_count_total = 0
    err_count_total = 0
    truncated_issue_count = 0

    def add_issue(level: str, wav_name: str, alias: str, msg: str) -> None:
        nonlocal warn_count_total, err_count_total, truncated_issue_count
        if str(level).lower() == "warn":
            warn_count_total += 1
        else:
            err_count_total += 1
        if len(issues) < max_issues:
            issues.append((level, wav_name, alias, msg))
        else:
            truncated_issue_count += 1

    missing_wavs: List[str] = []
    wav_idx = _build_wav_name_index(wav_dir)
    tg_idx = _build_textgrid_index(tg_folder) if use_textgrid else {}
    is_japanese = (language == "japanese")
    # Japanese multi-syllable files naturally have later offsets on subsequent aliases.
    # Keep strict checks for parameter ordering, but relax global-time heuristics.
    offset_head_late_ms = 360.0 if is_japanese else 260.0
    pre_before_margin_ms = 60.0 if is_japanese else 40.0
    pre_after_margin_ms = 220.0 if is_japanese else 100.0
    cutoff_after_margin_ms = 320.0 if is_japanese else 260.0
    boundary_dist_warn_ms = 260.0 if is_japanese else 180.0

    for wav_name, rows in sampled:
        wav_path = _resolve_wav_path(wav_name, wav_dir, wav_idx)
        if not wav_path:
            add_issue("error", wav_name, "", "WAV file missing")
            missing_wavs.append(wav_name)
            continue

        audio, sr, wav_err = _read_wav_mono(wav_path)
        if wav_err or audio is None:
            add_issue("error", wav_name, "", f"Audio decode failed: {wav_err}")
            continue

        try:
            wav_stat = os.stat(wav_path)
            mel_cache_key = (
                os.path.normcase(os.path.abspath(wav_path)),
                int(wav_stat.st_mtime_ns),
                int(wav_stat.st_size),
                bool(fast_mode),
            )
        except Exception:
            mel_cache_key = (os.path.normcase(os.path.abspath(wav_path)), 0, 0, bool(fast_mode))

        mel_sig = _VALIDATION_MEL_CACHE.get(mel_cache_key)
        if mel_sig is None:
            mel_sig = _mel_activity(audio, sr, fast=fast_mode)
            _VALIDATION_MEL_CACHE[mel_cache_key] = mel_sig
            _trim_cache(_VALIDATION_MEL_CACHE)

        base = os.path.splitext(wav_name)[0]
        tg_path = tg_idx.get(base.lower()) or tg_idx.get(normalize_wav_key(wav_name)) or ""
        if use_textgrid and tg_path and os.path.exists(tg_path):
            try:
                tg_stat = os.stat(tg_path)
                tg_cache_key = (
                    os.path.normcase(os.path.abspath(tg_path)),
                    int(tg_stat.st_mtime_ns),
                    int(tg_stat.st_size),
                )
            except Exception:
                tg_cache_key = (os.path.normcase(os.path.abspath(tg_path)), 0, 0)
            cached_bounds = _VALIDATION_PHONE_CACHE.get(tg_cache_key)
            if cached_bounds is None:
                cached_bounds = _load_phone_bounds_ms(tg_path)
                _VALIDATION_PHONE_CACHE[tg_cache_key] = cached_bounds
                _trim_cache(_VALIDATION_PHONE_CACHE)
            ph_starts, ph_ends = cached_bounds
        else:
            ph_starts, ph_ends = ([], [])
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
            _append_alias_metric(
                alias_metric_values,
                alias_type,
                "offset_late_ms",
                max(0.0, off - float(mel_sig["active_start_ms"])),
            )
            _append_alias_metric(
                alias_metric_values,
                alias_type,
                "cutoff_overrun_ms",
                max(0.0, cut - float(mel_sig["active_end_ms"])),
            )
            pre_abs_seq.append(pre)
            if alias_type not in ("vc", "vv"):
                anchor_pre_abs_seq.append(pre)

            if not (off <= ovl <= pre <= cons < cut):
                add_issue("error", wav_name, alias, "Parameter order invalid")

            # Offset-late check is meaningful mostly for the first alias.
            # Later aliases in multi-mora files are expected to be far from active_start.
            if row_idx == 0 and alias_type not in ("vc", "vv") and off > mel_sig["active_start_ms"] + offset_head_late_ms:
                add_issue("warn", wav_name, alias, f"Offset seems late ({off:.1f}ms)")
            early_margin = 120.0
            if is_japanese and alias_type in ("vc", "vv"):
                early_margin = 220.0
            elif is_japanese and alias_type in ("cv_head", "cv", "vcv"):
                early_margin = 160.0
            if off < mel_sig["active_start_ms"] - early_margin:
                add_issue("warn", wav_name, alias, f"Offset before active region ({off:.1f}ms)")

            pre_before_margin = pre_before_margin_ms
            if is_japanese and alias_type in ("cv_head", "cv", "vcv"):
                pre_before_margin += 40.0
            if pre < mel_sig["active_start_ms"] - pre_before_margin or pre > mel_sig["active_end_ms"] + pre_after_margin_ms:
                add_issue("warn", wav_name, alias, f"Preutterance out of active region ({pre:.1f}ms)")
            if cut > mel_sig["active_end_ms"] + cutoff_after_margin_ms:
                add_issue("warn", wav_name, alias, f"Cutoff too long after active end ({cut:.1f}ms)")
            blank_stats = _blank_span_activity_stats(mel_sig, off, cut)
            _append_alias_metric(
                alias_metric_values,
                alias_type,
                "blank_ratio",
                float(blank_stats.get("blank_ratio", 0.0) or 0.0),
            )
            if _should_warn_blank_only_span(alias_type, blank_stats):
                add_issue(
                    "warn",
                    wav_name,
                    alias,
                    f"Offset-cutoff span is blank-heavy ({blank_stats['blank_ratio']:.2f})",
                )

            if boundary_check_enabled:
                if alias_type in ("vc", "vv"):
                    d = _nearest_dist_ms(pre, ph_ends)
                else:
                    d = min(_nearest_dist_ms(pre, ph_starts), _nearest_dist_ms(pre, ph_ends))
                _append_alias_metric(alias_metric_values, alias_type, "pre_boundary_dist_ms", d)
                if d > boundary_dist_warn_ms:
                    add_issue("warn", wav_name, alias, f"Preutterance far from phone boundary ({d:.1f}ms)")

        seq_for_check = pre_abs_seq
        if is_japanese and len(anchor_pre_abs_seq) >= 4:
            seq_for_check = anchor_pre_abs_seq
        if len(seq_for_check) >= 4:
            inv = 0
            for i in range(1, len(seq_for_check)):
                if seq_for_check[i] + 5 < seq_for_check[i - 1]:
                    inv += 1
            if inv >= max(2, len(seq_for_check) // 4):
                add_issue("warn", wav_name, "", "Alias timing sequence appears shifted/backward")

        rec_aliases = list_map.get(wav_name.lower()) if not fast_mode else None
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
                    add_issue("warn", wav_name, "", "Recording list order mismatch with OTO aliases")

        # filename-based hint for Japanese vowel chains
        if language == "japanese" and not fast_mode:
            s = parse_ja_filename(base)
            if len(s) >= 4 and all(re.match(r"^[aiueo]$", x) for x in s):
                spread = max(pre_abs_seq) - min(pre_abs_seq) if pre_abs_seq else 0
                if spread < 80:
                    add_issue("warn", wav_name, "", "Vowel-chain file has too little timing spread")

    # If very few/no files could be checked, this is usually a dataset/path mismatch
    # rather than timing quality regression.
    total_wavs = len(wav_items)
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
        add_issue("warn", "(validation)", "", msg)
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
            add_issue("warn", "(validation)", "", f"Suggested WAV folder: {h_dir} ({h_match}/{total_wavs} match)")

    warn_count = int(warn_count_total)
    err_count = int(err_count_total)
    summary["warnings"] = warn_count
    summary["errors"] = err_count
    summary["truncated_issues"] = int(truncated_issue_count)
    summary["alias_metrics"] = _summarize_alias_metrics(alias_metric_values)
    if truncated_issue_count > 0 and len(issues) < max_issues:
        issues.append(("warn", "(validation)", "", f"Issues truncated: +{truncated_issue_count} more (max={max_issues})"))

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
            alias_metrics = summary.get("alias_metrics") if isinstance(summary, dict) else {}
            if isinstance(alias_metrics, dict) and alias_metrics:
                f.write("alias_metrics:\n")
                for alias_type, metric_map in sorted(alias_metrics.items()):
                    f.write(f"  - {alias_type}\n")
                    if not isinstance(metric_map, dict):
                        continue
                    for metric_key, metric_stat in sorted(metric_map.items()):
                        if not isinstance(metric_stat, dict):
                            continue
                        f.write(
                            "    "
                            + f"{metric_key}: n={int(metric_stat.get('count', 0) or 0)} "
                            + f"p50={float(metric_stat.get('p50', 0.0) or 0.0):.3f} "
                            + f"p90={float(metric_stat.get('p90', 0.0) or 0.0):.3f}\n"
                        )
                f.write("\n")
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
