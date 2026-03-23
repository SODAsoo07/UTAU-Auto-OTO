from __future__ import annotations

import csv
import json
import os
import re
import wave
from array import array
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import textgrid
except Exception:  # pragma: no cover
    textgrid = None

from core.format_type_utils import normalize_format_type
from core.oto_ml.autofree.schema import CONFIDENCE_TARGET_NAME, FEATURE_NAMES
from core.oto_ml_features import classify_alias_type, detect_format_type, parse_oto_rows
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.pipeline_status import has_textgrid_files


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _infer_voicebank_id(voicebank_id: str = "", *paths: str) -> str:
    if str(voicebank_id or "").strip():
        return str(voicebank_id).strip()
    for candidate in paths:
        if not candidate:
            continue
        p = candidate if os.path.isdir(candidate) else os.path.dirname(os.path.abspath(candidate))
        base = os.path.basename(os.path.abspath(p))
        if base:
            return base
    return "unknown_voicebank"


def _read_wav_mono_float(path: str) -> Tuple[List[float], int]:
    if not path or not os.path.isfile(path):
        return [], 0
    try:
        with wave.open(path, "rb") as wf:
            channels = int(wf.getnchannels())
            sample_width = int(wf.getsampwidth())
            sr = int(wf.getframerate())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)
    except Exception:
        return [], 0
    if not raw or channels <= 0 or sr <= 0:
        return [], 0

    if sample_width == 1:
        vals = array("B")
        vals.frombytes(raw)
        if channels > 1:
            vals = array("B", vals[::channels])
        return [((int(v) - 128) / 128.0) for v in vals], sr
    if sample_width == 2:
        vals = array("h")
        vals.frombytes(raw)
        if channels > 1:
            vals = array("h", vals[::channels])
        return [float(v) / 32768.0 for v in vals], sr
    if sample_width == 4:
        vals = array("i")
        vals.frombytes(raw)
        if channels > 1:
            vals = array("i", vals[::channels])
        return [float(v) / 2147483648.0 for v in vals], sr
    return [], sr


def _segment_stats(samples: Sequence[float], sr: int, start_ms: float, end_ms: float) -> Dict[str, float]:
    out = {"mel_window_energy_mean": 0.0, "mel_window_silence_ratio": 1.0, "mel_window_voiced_ratio": 0.0}
    if not samples or sr <= 0:
        return out
    start_idx = int(max(0.0, float(start_ms)) * float(sr) / 1000.0)
    end_idx = int(max(float(start_ms), float(end_ms)) * float(sr) / 1000.0)
    end_idx = min(end_idx, len(samples))
    if end_idx <= start_idx:
        return out
    window = samples[start_idx:end_idx]
    n = float(len(window))
    abs_vals = [abs(float(v)) for v in window]
    out["mel_window_energy_mean"] = float(sum(abs_vals) / n)
    out["mel_window_silence_ratio"] = float(sum(1 for v in abs_vals if v < 0.01) / n)
    out["mel_window_voiced_ratio"] = float(sum(1 for v in abs_vals if v > 0.03) / n)
    return out


def _file_audio_stats(samples: Sequence[float]) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    n = float(len(samples))
    abs_vals = [abs(float(v)) for v in samples]
    return float(sum(abs_vals) / n), float(sum(1 for v in abs_vals if v > 0.03) / n)


def _wav_duration_ms(samples: Sequence[float], sr: int, wav_path: str = "") -> float:
    if samples and sr > 0:
        return float(len(samples)) * 1000.0 / float(sr)
    if not wav_path or not os.path.isfile(wav_path):
        return 0.0
    try:
        with wave.open(wav_path, "rb") as wf:
            nframes = int(wf.getnframes())
            sr2 = int(wf.getframerate())
        return float(nframes) * 1000.0 / float(sr2) if sr2 > 0 else 0.0
    except Exception:
        return 0.0


def _list_wav_files(wav_dir: str) -> List[str]:
    out: List[str] = []
    if not wav_dir or not os.path.isdir(wav_dir):
        return out
    for dp, _dns, fns in os.walk(wav_dir):
        for fn in fns:
            if fn.lower().endswith(".wav"):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


def _build_tg_index(tg_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not tg_dir or not os.path.isdir(tg_dir):
        return index
    for dp, _dns, fns in os.walk(tg_dir):
        for fn in fns:
            if not fn.lower().endswith(".textgrid"):
                continue
            full = os.path.join(dp, fn)
            stem = os.path.splitext(fn)[0]
            index[normalize_wav_key(stem)] = full
            index[normalize_wav_key(stem + ".wav")] = full
    return index


def _load_tg_intervals(tg_path: str) -> List[Dict[str, float]]:
    if not tg_path or not os.path.isfile(tg_path) or textgrid is None:
        return []
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception:
        return []
    chosen = None
    for tier in tg:
        name = str(getattr(tier, "name", "")).strip().lower()
        if name == "phones" or "phone" in name:
            chosen = tier
            break
    if chosen is None:
        for tier in tg:
            if getattr(tier, "intervals", None):
                chosen = tier
                break
    if chosen is None:
        return []
    out: List[Dict[str, float]] = []
    for iv in chosen:
        start = _to_float(getattr(iv, "minTime", 0.0), 0.0) * 1000.0
        end = _to_float(getattr(iv, "maxTime", 0.0), 0.0) * 1000.0
        if end > start:
            out.append(
                {
                    "start_ms": float(start),
                    "end_ms": float(end),
                    "mark": str(getattr(iv, "mark", "") or ""),
                }
            )
    return out


_BLANK_TEXTGRID_MARKS = {
    "",
    "sil",
    "silence",
    "sp",
    "spn",
    "pau",
    "pause",
    "rest",
    "_",
    "-",
}

_PAUSE_ALIASES = {
    "",
    "r",
    "rr",
    "sil",
    "sp",
    "spn",
    "pau",
    "pause",
    "rest",
    "_",
    "-",
}


def _normalize_short_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _is_blank_interval_mark(mark: object) -> bool:
    m = _normalize_short_text(mark)
    if not m:
        return True
    if m in _BLANK_TEXTGRID_MARKS:
        return True
    return bool(re.fullmatch(r"sil\d+|sp\d+|pau\d+|rest\d+", m))


def _is_pause_alias(alias: str) -> bool:
    a = _normalize_short_text(alias)
    if a in _PAUSE_ALIASES:
        return True
    return a.startswith("br")


def _interval_distance_ms(anchor_ms: float, interval: Dict[str, float]) -> float:
    start = _to_float(interval.get("start_ms"), 0.0)
    return abs(anchor_ms - start)


def _select_interval_pool(intervals: List[Dict[str, float]], alias: str) -> List[Dict[str, float]]:
    if not intervals:
        return []
    if _is_pause_alias(alias):
        return intervals
    non_blank = [iv for iv in intervals if not _is_blank_interval_mark(iv.get("mark", ""))]
    return non_blank or intervals


def _select_interval_for_row(
    *,
    row: Dict[str, object],
    row_index: int,
    row_count: int,
    intervals: List[Dict[str, float]],
) -> Dict[str, float]:
    alias = str(row.get("alias", "") or "")
    pool = _select_interval_pool(intervals, alias)
    if not pool:
        return {"start_ms": 0.0, "end_ms": 1.0, "mark": ""}

    mapped = pool[_map_index(row_index, row_count, len(pool))]
    anchor = _to_float(row.get("offset"), -1.0)
    if anchor >= 0.0:
        # If the current OTO row is already collapsed to file start (offset~0),
        # anchoring by offset repeatedly snaps to the first non-blank interval.
        # For non-leading rows, prefer order-based mapping to break this loop.
        mapped_start = _to_float(mapped.get("start_ms"), 0.0)
        if row_count > 1 and row_index > 0 and anchor <= 2.0 and mapped_start >= 10.0:
            return mapped
        nearest = pool[0]
        best_dist = _interval_distance_ms(anchor, nearest)
        for cand in pool[1:]:
            dist = _interval_distance_ms(anchor, cand)
            if dist < best_dist:
                nearest, best_dist = cand, dist
        return nearest
    return mapped


def _map_index(src_idx: int, src_count: int, dst_count: int) -> int:
    if src_count <= 1 or dst_count <= 1:
        return 0
    ratio = float(src_idx) / float(max(src_count - 1, 1))
    out = int(round(ratio * float(max(dst_count - 1, 1))))
    return max(0, min(dst_count - 1, out))


def _filename_tokens(wav_name: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(str(wav_name or "")))[0]
    parts = [p.strip() for p in re.split(r"[\s_\-]+", stem) if p.strip()]
    return parts or ([stem] if stem else [])


def _load_token_map(token_map_path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not token_map_path or not os.path.isfile(token_map_path):
        return out
    ext = os.path.splitext(token_map_path)[1].lower()
    if ext == ".json":
        try:
            with open(token_map_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    key = normalize_wav_key(str(k))
                    if not key:
                        continue
                    vals = [str(x).strip() for x in (v if isinstance(v, list) else re.split(r"[,\s]+", str(v))) if str(x).strip()]
                    if vals:
                        out[key] = vals
        except Exception:
            return out
        return out
    try:
        with open(token_map_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = str(raw or "").strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    left, right = line.split(":", 1)
                elif "," in line:
                    left, right = line.split(",", 1)
                else:
                    continue
                key = normalize_wav_key(left.strip())
                vals = [p.strip() for p in re.split(r"[,\s]+", right) if p.strip()]
                if key and vals:
                    out[key] = vals
    except Exception:
        return out
    return out


def _resolve_tokens_for_wav(
    wav_name: str,
    token_map: Dict[str, List[str]],
    *,
    allow_filename_tokens: bool = True,
) -> Tuple[List[str], str]:
    key = normalize_wav_key(wav_name)
    vals = list(token_map.get(key, []) or [])
    if vals:
        return vals, "external_map"
    if not allow_filename_tokens:
        return [], ""
    vals = _filename_tokens(wav_name)
    return (vals, "filename") if vals else ([], "")


def _resolve_format(language: str, aliases: Sequence[str], format_override: str = "") -> str:
    forced = normalize_format_type(language, format_override)
    if forced:
        return forced
    try:
        detected = detect_format_type(language, list(aliases))
    except Exception:
        detected = ""
    return normalize_format_type(language, detected) or "general"


def _safe_alias_type(language: str, alias: str) -> str:
    try:
        return str(classify_alias_type(language, alias)).strip().lower()
    except Exception:
        return ""


def _boundary_payload(
    *,
    onset_ms: float,
    tail_ms: float,
    segment_stats: Dict[str, float],
    source_confidence: float,
    source_detail: str,
    boundary_info: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    boundary = dict(boundary_info or {})
    onset = max(0.0, float(onset_ms))
    tail = max(onset + 1.0, float(tail_ms))
    span = max(1.0, tail - onset)
    voiced_onset = _to_float(boundary.get("voiced_onset_ms"), onset)
    nucleus_start = _to_float(boundary.get("nucleus_start_ms"), onset + (span * 0.25))
    nucleus_end = _to_float(boundary.get("nucleus_end_ms"), onset + (span * 0.75))
    nucleus_start = _clamp(nucleus_start, onset, tail - 1.0)
    nucleus_end = _clamp(max(nucleus_start + 1.0, nucleus_end), nucleus_start + 1.0, tail)
    return {
        "source_detail": str(boundary.get("source_detail", source_detail) or source_detail),
        "voiced_onset_ms": float(_clamp(voiced_onset, onset, tail)),
        "nucleus_start_ms": float(nucleus_start),
        "nucleus_end_ms": float(nucleus_end),
        "nucleus_span_ms": float(max(0.0, nucleus_end - nucleus_start)),
        "tail_margin_ms": float(max(0.0, tail - nucleus_end)),
        "blank_confidence": float(
            _clamp(
                _to_float(boundary.get("blank_confidence"), segment_stats.get("mel_window_silence_ratio", 1.0)),
                0.0,
                1.0,
            )
        ),
        "voiced_confidence": float(
            _clamp(
                _to_float(boundary.get("voiced_confidence"), segment_stats.get("mel_window_voiced_ratio", 0.0)),
                0.0,
                1.0,
            )
        ),
        "unvoiced_confidence": float(_clamp(_to_float(boundary.get("unvoiced_confidence"), 0.0), 0.0, 1.0)),
        "boundary_confidence": float(
            _clamp(_to_float(boundary.get("boundary_confidence"), source_confidence), 0.0, 1.0)
        ),
    }


def _resolve_audio_source_confidence(token_source: str, boundary_confidence: float, source_detail: str) -> float:
    token = str(token_source or "").strip().lower()
    detail = str(source_detail or "").strip().lower()
    base = 0.52
    if token == "external_map":
        base = 0.64
    elif token == "oto_rows":
        base = 0.72
    conf = (base * 0.58) + (_clamp(boundary_confidence, 0.0, 1.0) * 0.42)
    if detail == "uniform_split":
        conf -= 0.10
    elif detail == "boundary_guided_pause":
        conf -= 0.06
    return float(_clamp(conf, 0.05, 0.96))


def _make_row(
    *,
    language: str,
    format_type: str,
    voicebank_id: str,
    wav_name: str,
    alias: str,
    row_index_in_wav: int,
    source_mode: str,
    token_source: str,
    source_confidence: float,
    onset_ms: float,
    tail_ms: float,
    segment_stats: Dict[str, float],
    file_mean_energy: float,
    file_voiced_ratio: float,
    source_detail: str = "",
    boundary_info: Optional[Dict[str, object]] = None,
    line_index: int = -1,
) -> Dict[str, object]:
    onset_ms = max(0.0, float(onset_ms))
    tail_ms = max(onset_ms + 1.0, float(tail_ms))
    dur_ms = max(1.0, tail_ms - onset_ms)
    alias_norm = canonicalize_alias_for_matching(language, alias)
    boundary_payload = _boundary_payload(
        onset_ms=onset_ms,
        tail_ms=tail_ms,
        segment_stats=segment_stats,
        source_confidence=source_confidence,
        source_detail=source_detail,
        boundary_info=boundary_info,
    )
    return {
        "row_uid": "",
        "language": str(language).strip().lower(),
        "format_type": str(format_type).strip().lower(),
        "voicebank_id": str(voicebank_id),
        "wav": str(wav_name),
        "alias": str(alias),
        "wav_norm": normalize_wav_key(wav_name),
        "alias_norm": alias_norm,
        "occurrence_index": 0,
        "row_index_in_wav": int(row_index_in_wav),
        "line_index": int(line_index),
        "source_mode": str(source_mode),
        "source_detail": str(boundary_payload.get("source_detail", source_detail) or source_detail),
        "token_source": str(token_source),
        "source_confidence": float(_clamp(source_confidence, 0.0, 1.0)),
        "onset_ms": float(onset_ms),
        "voiced_onset_ms": float(_to_float(boundary_payload.get("voiced_onset_ms"), onset_ms)),
        "nucleus_start_ms": float(_to_float(boundary_payload.get("nucleus_start_ms"), onset_ms + (dur_ms * 0.25))),
        "nucleus_end_ms": float(_to_float(boundary_payload.get("nucleus_end_ms"), onset_ms + (dur_ms * 0.75))),
        "nucleus_span_ms": float(_to_float(boundary_payload.get("nucleus_span_ms"), dur_ms * 0.5)),
        "tail_ms": float(tail_ms),
        "tail_margin_ms": float(_to_float(boundary_payload.get("tail_margin_ms"), max(0.0, tail_ms - (onset_ms + (dur_ms * 0.75))))),
        "blank_confidence": float(_to_float(boundary_payload.get("blank_confidence"), _clamp(_to_float(segment_stats.get("mel_window_silence_ratio"), 1.0), 0.0, 1.0))),
        "voiced_confidence": float(_to_float(boundary_payload.get("voiced_confidence"), 0.0)),
        "unvoiced_confidence": float(_to_float(boundary_payload.get("unvoiced_confidence"), 0.0)),
        "boundary_confidence": float(_to_float(boundary_payload.get("boundary_confidence"), source_confidence)),
        "mel_window_energy_mean": float(_to_float(segment_stats.get("mel_window_energy_mean"), 0.0)),
        "mel_window_silence_ratio": float(_clamp(_to_float(segment_stats.get("mel_window_silence_ratio"), 1.0), 0.0, 1.0)),
        "mel_window_voiced_ratio": float(_clamp(_to_float(segment_stats.get("mel_window_voiced_ratio"), 0.0), 0.0, 1.0)),
        "prev_gap_ms": 0.0,
        "next_gap_ms": 0.0,
        "neighbor_boundary_gap_ms": 0.0,
        "file_row_count": 0,
        "file_mean_energy": float(file_mean_energy),
        "file_voiced_ratio": float(file_voiced_ratio),
    }


def _apply_occurrence_and_context(rows: List[Dict[str, object]]) -> None:
    by_wav: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_wav[str(row.get("wav_norm", ""))].append(row)
    for wav_norm, wav_rows in by_wav.items():
        wav_rows.sort(key=lambda item: int(item.get("row_index_in_wav", 0)))
        counts: Dict[str, int] = {}
        row_count = len(wav_rows)
        file_mean_energy = _to_float(wav_rows[0].get("file_mean_energy"), 0.0) if wav_rows else 0.0
        file_voiced_ratio = _to_float(wav_rows[0].get("file_voiced_ratio"), 0.0) if wav_rows else 0.0
        for idx, row in enumerate(wav_rows):
            alias_norm = str(row.get("alias_norm", "") or "")
            occ = counts.get(alias_norm, 0)
            counts[alias_norm] = occ + 1
            row["occurrence_index"] = int(occ)
            row["file_row_count"] = int(row_count)
            row["file_mean_energy"] = float(file_mean_energy)
            row["file_voiced_ratio"] = float(file_voiced_ratio)
            onset = _to_float(row.get("onset_ms"), 0.0)
            tail = _to_float(row.get("tail_ms"), 0.0)
            row["prev_gap_ms"] = onset - _to_float(wav_rows[idx - 1].get("tail_ms"), 0.0) if idx > 0 else 0.0
            row["next_gap_ms"] = _to_float(wav_rows[idx + 1].get("onset_ms"), 0.0) - tail if idx + 1 < row_count else 0.0
            row["nucleus_span_ms"] = max(0.0, _to_float(row.get("nucleus_end_ms"), 0.0) - _to_float(row.get("nucleus_start_ms"), 0.0))
            row["tail_margin_ms"] = max(0.0, tail - _to_float(row.get("nucleus_end_ms"), tail))
            gap_candidates = [abs(float(v)) for v in (row["prev_gap_ms"], row["next_gap_ms"]) if abs(float(v)) > 1e-6]
            row["neighbor_boundary_gap_ms"] = min(gap_candidates) if gap_candidates else 0.0
            row["row_uid"] = (
                f"{row.get('voicebank_id', 'unknown')}:{wav_norm}:"
                f"{alias_norm}:{int(occ)}:{int(row.get('row_index_in_wav', idx))}"
            )


def build_textgrid_rows(
    *,
    language: str,
    wav_dir: str,
    tg_dir: str,
    format_type_override: str = "",
    voicebank_id: str = "",
    token_map_path: str = "",
    allow_filename_tokens: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    lang = str(language).strip().lower()
    vb_id = _infer_voicebank_id(voicebank_id, wav_dir, tg_dir)
    token_map = _load_token_map(token_map_path)
    tg_index = _build_tg_index(tg_dir)
    rows: List[Dict[str, object]] = []
    stats = {"wav_files": 0, "rows_built": 0, "skip_textgrid_missing": 0, "skip_textgrid_empty": 0, "skip_token_missing": 0}
    for wav_path in _list_wav_files(wav_dir):
        stats["wav_files"] += 1
        wav_name = os.path.basename(wav_path)
        wav_norm = normalize_wav_key(wav_name)
        tg_path = tg_index.get(wav_norm, "")
        if not tg_path:
            stats["skip_textgrid_missing"] += 1
            continue
        intervals = _load_tg_intervals(tg_path)
        if not intervals:
            stats["skip_textgrid_empty"] += 1
            continue
        tokens, token_source = _resolve_tokens_for_wav(wav_name, token_map, allow_filename_tokens=allow_filename_tokens)
        if not tokens:
            stats["skip_token_missing"] += 1
            continue
        samples, sr = _read_wav_mono_float(wav_path)
        file_energy, file_voiced = _file_audio_stats(samples)
        fmt = _resolve_format(lang, tokens, format_override=format_type_override)
        for idx, token in enumerate(tokens):
            pool = _select_interval_pool(intervals, str(token))
            iv = pool[_map_index(idx, len(tokens), len(pool))]
            onset = _to_float(iv.get("start_ms"), 0.0)
            tail = _to_float(iv.get("end_ms"), onset + 1.0)
            seg = _segment_stats(samples, sr, onset, tail)
            rows.append(
                _make_row(
                    language=lang,
                    format_type=fmt,
                    voicebank_id=vb_id,
                    wav_name=wav_name,
                    alias=str(token),
                    row_index_in_wav=idx,
                    source_mode="textgrid",
                    token_source=token_source or "filename",
                    source_confidence=0.90 if token_source == "external_map" else 0.80,
                    onset_ms=onset,
                    tail_ms=tail,
                    segment_stats=seg,
                    file_mean_energy=file_energy,
                    file_voiced_ratio=file_voiced,
                    line_index=-1,
                )
            )
            stats["rows_built"] += 1
    _apply_occurrence_and_context(rows)
    return rows, stats


def build_audio_only_rows(
    *,
    language: str,
    wav_dir: str,
    format_type_override: str = "",
    voicebank_id: str = "",
    token_map_path: str = "",
    allow_filename_tokens: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    lang = str(language).strip().lower()
    vb_id = _infer_voicebank_id(voicebank_id, wav_dir)
    token_map = _load_token_map(token_map_path)
    rows: List[Dict[str, object]] = []
    stats = {"wav_files": 0, "rows_built": 0, "skip_token_missing": 0, "skip_wav_invalid": 0}
    for wav_path in _list_wav_files(wav_dir):
        stats["wav_files"] += 1
        wav_name = os.path.basename(wav_path)
        tokens, token_source = _resolve_tokens_for_wav(wav_name, token_map, allow_filename_tokens=allow_filename_tokens)
        if not tokens:
            stats["skip_token_missing"] += 1
            continue
        samples, sr = _read_wav_mono_float(wav_path)
        duration_ms = _wav_duration_ms(samples, sr, wav_path=wav_path)
        if duration_ms <= 1.0:
            stats["skip_wav_invalid"] += 1
            continue
        file_energy, file_voiced = _file_audio_stats(samples)
        fmt = _resolve_format(lang, tokens, format_override=format_type_override)
        count = len(tokens)
        for idx, token in enumerate(tokens):
            start = float(idx) * duration_ms / float(max(count, 1))
            end = float(idx + 1) * duration_ms / float(max(count, 1))
            seg = _segment_stats(samples, sr, start, end)
            rows.append(
                _make_row(
                    language=lang,
                    format_type=fmt,
                    voicebank_id=vb_id,
                    wav_name=wav_name,
                    alias=str(token),
                    row_index_in_wav=idx,
                    source_mode="audio_only",
                    token_source=token_source or "filename",
                    source_confidence=0.65 if token_source == "external_map" else 0.55,
                    onset_ms=start,
                    tail_ms=end,
                    segment_stats=seg,
                    file_mean_energy=file_energy,
                    file_voiced_ratio=file_voiced,
                    line_index=-1,
                )
            )
            stats["rows_built"] += 1
    _apply_occurrence_and_context(rows)
    return rows, stats


def resolve_manual_cutoff_abs(
    *,
    manual_offset: float,
    manual_cutoff: float,
    hint_cutoff_abs: float = 0.0,
    wav_duration_ms: float = 0.0,
) -> Tuple[Optional[float], str]:
    if float(manual_cutoff) < 0.0:
        return float(manual_offset + abs(float(manual_cutoff))), "negative_relative"

    cutoff_abs_raw = abs(float(manual_cutoff))
    candidates = [
        ("positive_relative", float(manual_offset + cutoff_abs_raw)),
        ("positive_absolute", float(cutoff_abs_raw)),
    ]
    wav_ms = max(0.0, float(wav_duration_ms))
    if wav_ms > 0.0:
        candidates.append(("positive_from_wav_end", float(manual_offset + max(wav_ms - cutoff_abs_raw, 0.0))))

    target = float(hint_cutoff_abs or 0.0)
    scored: List[Tuple[float, str, float]] = []
    for mode, cand in candidates:
        score = abs(cand - target)
        if wav_ms > 0.0:
            if cand > (wav_ms + 80.0):
                score += 2000.0 + (cand - wav_ms)
            if cand < (manual_offset + 4.0):
                score += 800.0 + ((manual_offset + 4.0) - cand)
        scored.append((score, mode, cand))
    scored.sort(key=lambda item: item[0])
    chosen_mode, chosen_abs = scored[0][1], float(scored[0][2])
    if wav_ms > 0.0:
        chosen_abs = min(chosen_abs, max(0.0, wav_ms - 2.0))
    if chosen_abs <= (float(manual_offset) + 2.0):
        return None, "invalid_cutoff"
    return chosen_abs, chosen_mode


def _manual_match_key(row: Dict[str, object]) -> Tuple[str, str, int]:
    return (
        str(row.get("wav_norm", "")),
        str(row.get("alias_norm", "")),
        int(row.get("occurrence_index", 0)),
    )


def _max_time_shift_ms(row: Dict[str, object]) -> float:
    span = _to_float(row.get("tail_ms"), 0.0) - _to_float(row.get("onset_ms"), 0.0)
    if span <= 0.0:
        return 1200.0
    return max(400.0, min(2400.0, span * 6.0))


def _score_quality(source_confidence: float, match_strategy: str) -> float:
    base = _clamp(source_confidence, 0.0, 1.0)
    if match_strategy == "exact_key":
        return _clamp(base + 0.10, 0.0, 1.0)
    if match_strategy == "time_nearest":
        return _clamp(base * 0.78, 0.0, 1.0)
    return _clamp(base * 0.4, 0.0, 1.0)


def build_rows_for_training(
    *,
    language: str,
    manual_oto_path: str,
    wav_dir: str,
    tg_dir: str = "",
    source_mode: str = "auto",
    token_map_path: str = "",
    format_type_override: str = "",
    voicebank_id: str = "",
    allow_filename_tokens: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    lang = str(language).strip().lower()
    mode = str(source_mode or "auto").strip().lower()
    use_textgrid = mode == "textgrid" or (mode == "auto" and has_textgrid_files(tg_dir))
    if use_textgrid:
        feature_rows, build_stats = build_textgrid_rows(
            language=lang,
            wav_dir=wav_dir,
            tg_dir=tg_dir,
            format_type_override=format_type_override,
            voicebank_id=voicebank_id,
            token_map_path=token_map_path,
            allow_filename_tokens=allow_filename_tokens,
        )
    else:
        feature_rows, build_stats = build_audio_only_rows(
            language=lang,
            wav_dir=wav_dir,
            format_type_override=format_type_override,
            voicebank_id=voicebank_id,
            token_map_path=token_map_path,
            allow_filename_tokens=allow_filename_tokens,
        )

    manual_rows = parse_oto_rows(manual_oto_path, language=lang)
    manual_by_key: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    manual_by_wav_alias: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    manual_by_wav: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in manual_rows:
        manual_by_key[_manual_match_key(row)].append(row)
        manual_by_wav_alias[(str(row.get("wav_norm", "")), str(row.get("alias_norm", "")))].append(row)
        manual_by_wav[str(row.get("wav_norm", ""))].append(row)

    used_manual = set()
    matched: List[Dict[str, object]] = []
    stats = {
        "feature_rows": len(feature_rows),
        "manual_rows": len(manual_rows),
        "matched_rows": 0,
        "skipped_rows": 0,
        "matched_exact_key": 0,
        "matched_time_nearest": 0,
        "skipped_cutoff_invalid": 0,
        "skip_token_missing": int(build_stats.get("skip_token_missing", 0)),
    }
    stats.update({f"build_{k}": int(v) for k, v in build_stats.items() if isinstance(v, int)})

    for feat in feature_rows:
        key = _manual_match_key(feat)
        manual_row = None
        strategy = ""

        for cand in manual_by_key.get(key, []):
            cand_id = str(cand.get("source_row_id", "") or cand.get("line_index", ""))
            if cand_id in used_manual:
                continue
            manual_row = cand
            strategy = "exact_key"
            break

        if manual_row is None:
            wav_norm = str(feat.get("wav_norm", ""))
            alias_norm = str(feat.get("alias_norm", ""))
            pool = []
            for cand in manual_by_wav_alias.get((wav_norm, alias_norm), []):
                cand_id = str(cand.get("source_row_id", "") or cand.get("line_index", ""))
                if cand_id not in used_manual:
                    pool.append(cand)
            if not pool:
                for cand in manual_by_wav.get(wav_norm, []):
                    cand_id = str(cand.get("source_row_id", "") or cand.get("line_index", ""))
                    if cand_id not in used_manual:
                        pool.append(cand)
            if pool:
                onset = _to_float(feat.get("onset_ms"), 0.0)
                pool.sort(key=lambda r: abs(_to_float(r.get("offset"), 0.0) - onset))
                cand = pool[0]
                diff = abs(_to_float(cand.get("offset"), 0.0) - onset)
                if diff <= _max_time_shift_ms(feat):
                    manual_row = cand
                    strategy = "time_nearest"

        if manual_row is None:
            stats["skipped_rows"] += 1
            continue

        manual_id = str(manual_row.get("source_row_id", "") or manual_row.get("line_index", ""))
        used_manual.add(manual_id)
        manual_offset = _to_float(manual_row.get("offset"), 0.0)
        cutoff_abs, cutoff_mode = resolve_manual_cutoff_abs(
            manual_offset=manual_offset,
            manual_cutoff=_to_float(manual_row.get("cutoff"), 0.0),
            hint_cutoff_abs=_to_float(feat.get("tail_ms"), 0.0),
            wav_duration_ms=_to_float(feat.get("tail_ms"), 0.0),
        )
        if cutoff_abs is None:
            stats["skipped_rows"] += 1
            stats["skipped_cutoff_invalid"] += 1
            continue

        row = dict(feat)
        row["target_offset_ms"] = float(manual_offset)
        row["target_cons_ms"] = float(max(0.0, _to_float(manual_row.get("cons"), 0.0)))
        row["target_cutoff_abs_ms"] = float(cutoff_abs)
        row["target_pre_ms"] = float(max(0.0, _to_float(manual_row.get("pre"), 0.0)))
        row["target_ovl_ms"] = float(_to_float(manual_row.get("ovl"), 0.0))
        row["label_source"] = "manual_oto"
        row["match_strategy"] = str(strategy or "unmatched")
        quality = _score_quality(_to_float(row.get("source_confidence"), 0.0), row["match_strategy"])
        row["quality_score"] = float(quality)
        row[CONFIDENCE_TARGET_NAME] = float(quality)
        row["sample_weight"] = float(_clamp(0.35 + (0.95 * quality), 0.2, 1.5))
        row["skip_with_reason"] = ""
        row["manual_cutoff_mode"] = str(cutoff_mode)
        for name in FEATURE_NAMES:
            row.setdefault(name, 0.0)
        matched.append(row)
        stats["matched_rows"] += 1
        if strategy == "exact_key":
            stats["matched_exact_key"] += 1
        elif strategy == "time_nearest":
            stats["matched_time_nearest"] += 1
    return matched, stats


def build_rows_for_inference(
    *,
    language: str,
    oto_path: str,
    wav_dir: str,
    tg_dir: str = "",
    token_map_path: str = "",
    format_type_override: str = "",
    voicebank_id: str = "",
    allow_filename_tokens: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    lang = str(language).strip().lower()
    vb_id = _infer_voicebank_id(voicebank_id, oto_path, wav_dir, tg_dir)
    token_map = _load_token_map(token_map_path)
    tg_index = _build_tg_index(tg_dir)
    oto_rows = parse_oto_rows(oto_path, language=lang)
    by_wav: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in oto_rows:
        by_wav[str(row.get("wav", ""))].append(row)
    for rows in by_wav.values():
        rows.sort(key=lambda item: int(item.get("line_index", 0)))

    wav_index = {normalize_wav_key(os.path.basename(path)): path for path in _list_wav_files(wav_dir)}
    out_rows: List[Dict[str, object]] = []
    stats = {
        "oto_rows": len(oto_rows),
        "rows_built": 0,
        "skip_token_missing": 0,
        "skip_wav_invalid": 0,
        "source_mode_textgrid_rows": 0,
        "source_mode_audio_only_rows": 0,
    }

    for wav_name, wav_rows in by_wav.items():
        wav_norm = normalize_wav_key(wav_name)
        wav_path = wav_index.get(wav_norm, "")
        samples, sr = _read_wav_mono_float(wav_path)
        duration_ms = _wav_duration_ms(samples, sr, wav_path=wav_path)
        if duration_ms <= 1.0:
            stats["skip_wav_invalid"] += len(wav_rows)
            continue
        file_energy, file_voiced = _file_audio_stats(samples)
        fmt = _resolve_format(lang, [str(row.get("alias", "")) for row in wav_rows], format_override=format_type_override)
        intervals = _load_tg_intervals(tg_index.get(wav_norm, ""))
        if intervals:
            for idx, row in enumerate(wav_rows):
                alias = str(row.get("alias", "") or "")
                iv = _select_interval_for_row(
                    row=row,
                    row_index=idx,
                    row_count=len(wav_rows),
                    intervals=intervals,
                )
                onset = _to_float(iv.get("start_ms"), 0.0)
                tail = _to_float(iv.get("end_ms"), onset + 1.0)
                seg = _segment_stats(samples, sr, onset, tail)
                item = _make_row(
                    language=lang,
                    format_type=fmt,
                    voicebank_id=vb_id,
                    wav_name=wav_name,
                    alias=alias,
                    row_index_in_wav=idx,
                    source_mode="textgrid",
                    token_source="filename",
                    source_confidence=0.80,
                    onset_ms=onset,
                    tail_ms=tail,
                    segment_stats=seg,
                    file_mean_energy=file_energy,
                    file_voiced_ratio=file_voiced,
                    line_index=int(row.get("line_index", -1)),
                )
                item["occurrence_index"] = int(row.get("occurrence_index", 0))
                item["current_offset"] = _to_float(row.get("offset"), 0.0)
                item["current_cons"] = _to_float(row.get("cons"), 0.0)
                item["current_cutoff"] = _to_float(row.get("cutoff"), 0.0)
                item["current_pre"] = _to_float(row.get("pre"), 0.0)
                item["current_ovl"] = _to_float(row.get("ovl"), 0.0)
                item["alias_type"] = _safe_alias_type(lang, alias)
                out_rows.append(item)
                stats["rows_built"] += 1
                stats["source_mode_textgrid_rows"] += 1
            continue

        tokens, token_source = _resolve_tokens_for_wav(wav_name, token_map, allow_filename_tokens=allow_filename_tokens)
        if not tokens:
            stats["skip_token_missing"] += len(wav_rows)
            continue
        count = len(wav_rows)
        for idx, row in enumerate(wav_rows):
            alias = str(row.get("alias", "") or "")
            start = float(idx) * duration_ms / float(max(count, 1))
            end = float(idx + 1) * duration_ms / float(max(count, 1))
            seg = _segment_stats(samples, sr, start, end)
            item = _make_row(
                language=lang,
                format_type=fmt,
                voicebank_id=vb_id,
                wav_name=wav_name,
                alias=alias,
                row_index_in_wav=idx,
                source_mode="audio_only",
                token_source=token_source or "filename",
                source_confidence=0.60 if token_source == "external_map" else 0.52,
                onset_ms=start,
                tail_ms=end,
                segment_stats=seg,
                file_mean_energy=file_energy,
                file_voiced_ratio=file_voiced,
                line_index=int(row.get("line_index", -1)),
            )
            item["occurrence_index"] = int(row.get("occurrence_index", 0))
            item["current_offset"] = _to_float(row.get("offset"), 0.0)
            item["current_cons"] = _to_float(row.get("cons"), 0.0)
            item["current_cutoff"] = _to_float(row.get("cutoff"), 0.0)
            item["current_pre"] = _to_float(row.get("pre"), 0.0)
            item["current_ovl"] = _to_float(row.get("ovl"), 0.0)
            item["alias_type"] = _safe_alias_type(lang, alias)
            out_rows.append(item)
            stats["rows_built"] += 1
            stats["source_mode_audio_only_rows"] += 1

    _apply_occurrence_and_context(out_rows)
    return out_rows, stats


__all__ = [
    "build_audio_only_rows",
    "build_rows_for_inference",
    "build_rows_for_training",
    "build_textgrid_rows",
    "resolve_manual_cutoff_abs",
]
