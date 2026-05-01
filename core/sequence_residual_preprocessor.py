from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_features import classify_alias_type
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.sequence_aligner import (
    _apply_uniform_boundary_advance,
    _build_ap_sp_mask,
    _build_frame_voicing_mask,
    _compute_sequence_cvn_consonant_prob,
    _env_bool,
    _env_float,
    _env_int,
    _estimate_sequence_labels,
    _framewise_consonant_cues,
    _framewise_rms,
    _filename_words_for_wav,
    _inject_ap_labels,
    _insert_internal_pause_rows,
    _load_sequence_profile,
    _load_transcript_words,
    _order_match_ratio,
    _prepare_analysis_signal,
    _profile_bool,
    _profile_float,
    _profile_int,
    _read_pcm_as_float_mono,
    _refine_word_boundaries_with_onset_cues,
    _resolve_silence_threshold,
    _resolve_word_rows,
)


PAUSE_ALIAS_TYPES = {"br", "sil", "pau", "sp", "breath"}
ENDBREATH_ALIAS_TYPE = "endbreath"


@dataclass(frozen=True)
class ManualOtoRow:
    wav: str
    alias: str
    offset_ms: float
    cons_ms: float
    cutoff: float
    pre_ms: float
    ovl_ms: float
    line_index: int
    wav_row_index: int
    alias_occurrence: int
    alias_type: str
    alias_norm: str
    wav_norm: str


@dataclass(frozen=True)
class SequenceAnalysis:
    wav_path: str
    wav_name: str
    duration_ms: float
    words: List[str]
    source: str
    filename_words: List[str]
    word_rows: List[Tuple[float, float, str]]
    phone_rows: List[Tuple[float, float, str]]
    labels: List[str]
    rms: np.ndarray
    zcr: np.ndarray
    hf_ratio: np.ndarray
    flux: np.ndarray
    cvn_c_prob: np.ndarray | None
    hop_sec: float
    silence_threshold: float
    soft_lock_ready: bool
    cvn_backend: str


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _safe_cutoff_abs_ms(offset_ms: float, cutoff: float, wav_duration_ms: float) -> float:
    off = max(0.0, _safe_float(offset_ms))
    cut = _safe_float(cutoff)
    dur = max(0.0, _safe_float(wav_duration_ms))
    if cut < 0.0:
        resolved = off + abs(cut)
    elif dur > 1.0 and cut > max(300.0, dur * 0.45):
        resolved = dur - cut
    else:
        resolved = off + cut
    if dur > 1.0:
        resolved = min(resolved, max(off + 1.0, dur - 1.0))
    return max(off + 1.0, resolved)


def _manual_timing_looks_configured(
    *,
    offset_ms: float,
    cons_ms: float,
    cutoff: float,
    pre_ms: float,
    ovl_ms: float,
    cutoff_abs_ms: float,
) -> bool:
    off = _safe_float(offset_ms)
    cons = _safe_float(cons_ms)
    pre = _safe_float(pre_ms)
    ovl = _safe_float(ovl_ms)
    cut = _safe_float(cutoff)
    cut_abs = _safe_float(cutoff_abs_ms)
    if abs(off) < 1.0 and max(abs(cons), abs(pre), abs(ovl)) < 1.0:
        return False
    if max(abs(off), abs(cons), abs(pre), abs(ovl)) < 1.0 and abs(cut) <= 1.5:
        return False
    if cons < 1.0 and pre < 1.0 and ovl < 1.0 and cut_abs <= off + 2.0:
        return False
    if cut_abs <= off + 0.5:
        return False
    return True


def _looks_like_endbreath(wav: str, alias: str) -> bool:
    alias_text = str(alias or "").strip()
    wav_stem = os.path.splitext(os.path.basename(str(wav or "").strip()))[0]
    if not alias_text and not wav_stem:
        return False
    alias_norm = re.sub(r"\s+", " ", alias_text).strip()
    alias_low = alias_norm.lower()
    wav_low = wav_stem.lower()

    if alias_low in {"r", "br", "bre", "breath"}:
        return True
    if re.search(r"(?:^|\s)[a-z\uac00-\ud7a3]+(?:\s+|[_'-]+)r$", alias_low):
        return True
    if re.search(r"(?:^|\s)r$", alias_low):
        return True
    if re.search(r"(?:^|[_'\-\s])r$", wav_low):
        return True
    return False


def _classify_alias(language: str, alias: str, wav: str = "") -> str:
    if _looks_like_endbreath(wav, alias):
        return ENDBREATH_ALIAS_TYPE
    try:
        value = str(classify_alias_type(language, alias) or "").strip().lower()
        if value:
            return value
    except Exception:
        pass
    text = str(alias or "").strip().lower()
    if not text:
        return "unknown"
    if text in {"br", "bre", "breath", "r", "sil", "sp"}:
        return "br"
    if text.startswith("- "):
        return "cv_head"
    if " " in text:
        return "vc"
    return "cv"


def _read_manual_oto_rows(path: str, language: str) -> List[ManualOtoRow]:
    text = read_text_with_fallback(path)
    raw_rows: List[Dict[str, object]] = []
    for line_index, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav = str(parsed.get("wav", "") or "").strip()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav or not alias:
            continue
        raw_rows.append(
            {
                "wav": wav,
                "alias": alias,
                "offset": _safe_float(parsed.get("offset", 0.0)),
                "cons": _safe_float(parsed.get("cons", 0.0)),
                "cutoff": _safe_float(parsed.get("cutoff", 0.0)),
                "pre": _safe_float(parsed.get("pre", 0.0)),
                "ovl": _safe_float(parsed.get("ovl", 0.0)),
                "line_index": int(line_index),
            }
        )

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[normalize_wav_key(str(row["wav"]))].append(row)

    rows: List[ManualOtoRow] = []
    for wav_norm, items in grouped.items():
        occurrence_counter: Dict[str, int] = defaultdict(int)
        for wav_row_index, row in enumerate(items):
            alias = str(row["alias"])
            alias_norm = canonicalize_alias_for_matching(language, alias)
            occ = int(occurrence_counter[alias_norm])
            occurrence_counter[alias_norm] += 1
            rows.append(
                ManualOtoRow(
                    wav=str(row["wav"]),
                    alias=alias,
                    offset_ms=_safe_float(row["offset"]),
                    cons_ms=_safe_float(row["cons"]),
                    cutoff=_safe_float(row["cutoff"]),
                    pre_ms=_safe_float(row["pre"]),
                    ovl_ms=_safe_float(row["ovl"]),
                    line_index=int(row["line_index"]),
                    wav_row_index=int(wav_row_index),
                    alias_occurrence=occ,
                    alias_type=_classify_alias(language, alias, str(row["wav"])),
                    alias_norm=alias_norm,
                    wav_norm=wav_norm,
                )
            )
    rows.sort(key=lambda r: r.line_index)
    return rows


def _build_wav_index(voicebank_dir: str) -> Dict[str, str]:
    root = os.path.abspath(str(voicebank_dir or "").strip())
    index: Dict[str, str] = {}
    if not os.path.isdir(root):
        return index
    for dp, _dns, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".wav"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, root)
            for key in {
                normalize_wav_key(fn),
                normalize_wav_key(rel),
                normalize_wav_key(os.path.splitext(rel)[0]),
            }:
                if key and key not in index:
                    index[key] = path
    return index


def _load_sequence_runtime_config(language: str, profile: Dict[str, object]) -> Dict[str, object]:
    return {
        "frame_ms": _env_int("UTOA_SEQUENCE_ALIGN_FRAME_MS", _profile_int(profile, "frame_ms", 22), min_value=8, max_value=80),
        "hop_ms": _env_int("UTOA_SEQUENCE_ALIGN_HOP_MS", _profile_int(profile, "hop_ms", 5), min_value=2, max_value=30),
        "analysis_dc_remove": _env_bool("UTOA_SEQUENCE_ANALYSIS_DC_REMOVE", default=bool(profile.get("analysis_dc_remove", True))),
        "analysis_highpass_hz": _env_float("UTOA_SEQUENCE_ANALYSIS_HIGHPASS_HZ", _profile_float(profile, "analysis_highpass_hz", 50.0), min_value=0.0, max_value=120.0),
        "analysis_max_gain_db": _env_float("UTOA_SEQUENCE_ANALYSIS_MAX_GAIN_DB", _profile_float(profile, "analysis_max_gain_db", 7.0), min_value=0.0, max_value=12.0),
        "analysis_target_rms": _env_float("UTOA_SEQUENCE_ANALYSIS_TARGET_RMS", _profile_float(profile, "analysis_target_rms", 0.08), min_value=0.01, max_value=0.30),
        "silence_q": _env_float("UTOA_SEQUENCE_ALIGN_SILENCE_Q", _profile_float(profile, "silence_q", 0.18), min_value=0.02, max_value=0.60),
        "silence_scale": _env_float("UTOA_SEQUENCE_ALIGN_SILENCE_SCALE", _profile_float(profile, "silence_scale", 0.90), min_value=0.40, max_value=1.30),
        "min_span_ms": _env_float("UTOA_SEQUENCE_ALIGN_MIN_SPAN_MS", _profile_float(profile, "min_span_ms", 18.0), min_value=5.0, max_value=250.0),
        "min_active_ratio": _env_float("UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MIN", _profile_float(profile, "active_ratio_min", 0.20), min_value=0.05, max_value=0.70),
        "max_active_ratio": _env_float("UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MAX", _profile_float(profile, "active_ratio_max", 0.90), min_value=0.20, max_value=0.98),
        "min_active_ms": _env_float("UTOA_SEQUENCE_ALIGN_MIN_ACTIVE_MS", _profile_float(profile, "min_active_ms", 24.0), min_value=5.0, max_value=400.0),
        "max_gap_ms": _env_float("UTOA_SEQUENCE_ALIGN_MAX_GAP_MS", _profile_float(profile, "max_gap_ms", 28.0), min_value=0.0, max_value=200.0),
        "onset_lookback_ms": _env_float("UTOA_SEQUENCE_ALIGN_ONSET_LOOKBACK_MS", _profile_float(profile, "onset_lookback_ms", 42.0), min_value=5.0, max_value=180.0),
        "drift_guard_level": _env_int("UTOA_SEQUENCE_ALIGN_DRIFT_GUARD_LEVEL", _profile_int(profile, "drift_guard_level", 1), min_value=0, max_value=3),
        "drift_guard_max_shift_ms": _env_float("UTOA_SEQUENCE_ALIGN_DRIFT_GUARD_MAX_SHIFT_MS", _profile_float(profile, "drift_guard_max_shift_ms", 28.0), min_value=4.0, max_value=90.0),
        "onset_snap_right_bias": _env_float("UTOA_SEQUENCE_ALIGN_ONSET_SNAP_RIGHT_BIAS", _profile_float(profile, "onset_snap_right_bias", 0.0008), min_value=-0.004, max_value=0.004),
        "plosive_onset_lookback_ms": _env_float("UTOA_SEQUENCE_ALIGN_PLOSIVE_ONSET_LOOKBACK_MS", _profile_float(profile, "plosive_onset_lookback_ms", 56.0), min_value=6.0, max_value=220.0),
        "plosive_left_bias_ms": _env_float("UTOA_SEQUENCE_ALIGN_PLOSIVE_LEFT_BIAS_MS", _profile_float(profile, "plosive_left_bias_ms", 4.0), min_value=0.0, max_value=24.0),
        "boundary_advance_ms": _env_float("UTOA_SEQUENCE_ALIGN_BOUNDARY_ADVANCE_MS", _profile_float(profile, "boundary_advance_ms", 0.0), min_value=0.0, max_value=20.0),
        "pause_min_ms": _env_float("UTOA_SEQUENCE_ALIGN_PAUSE_MIN_MS", _profile_float(profile, "pause_min_ms", 20.0), min_value=8.0, max_value=120.0),
        "pause_search_ms": _env_float("UTOA_SEQUENCE_ALIGN_PAUSE_SEARCH_MS", _profile_float(profile, "pause_search_ms", 42.0), min_value=10.0, max_value=160.0),
        "pause_energy_scale": _env_float("UTOA_SEQUENCE_ALIGN_PAUSE_ENERGY_SCALE", _profile_float(profile, "pause_energy_scale", 0.95), min_value=0.70, max_value=1.10),
        "pause_hard_silence_scale": _env_float("UTOA_SEQUENCE_ALIGN_PAUSE_HARD_SILENCE_SCALE", _profile_float(profile, "pause_hard_silence_scale", 0.80), min_value=0.50, max_value=1.00),
        "pause_unvoiced_guard_scale": _env_float("UTOA_SEQUENCE_ALIGN_PAUSE_UNVOICED_GUARD_SCALE", _profile_float(profile, "pause_unvoiced_guard_scale", 0.62), min_value=0.45, max_value=0.95),
        "ap_detect_enable": _env_bool("UTOA_SEQUENCE_ALIGN_AP_DETECT_ENABLE", default=_profile_bool(profile, "ap_detect_enable", True)),
        "ap_min_run_ms": _env_float("UTOA_SEQUENCE_ALIGN_AP_MIN_RUN_MS", _profile_float(profile, "ap_min_run_ms", 12.0), min_value=2.0, max_value=120.0),
        "ap_max_gap_ms": _env_float("UTOA_SEQUENCE_ALIGN_AP_MAX_GAP_MS", _profile_float(profile, "ap_max_gap_ms", 6.0), min_value=0.0, max_value=48.0),
        "filename_soft_lock_enable": _env_bool("UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_ENABLE", default=_profile_bool(profile, "filename_soft_lock_enable", True)),
        "filename_soft_lock_conf_threshold": _env_float("UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_CONF_TH", _profile_float(profile, "filename_soft_lock_conf_threshold", 0.46), min_value=0.05, max_value=0.95),
        "filename_soft_lock_max_shift_ms": _env_float("UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_MAX_SHIFT_MS", _profile_float(profile, "filename_soft_lock_max_shift_ms", 22.0), min_value=4.0, max_value=120.0),
        "filename_soft_lock_release_run": _env_int("UTOA_SEQUENCE_ALIGN_FILENAME_SOFT_LOCK_RELEASE_RUN", _profile_int(profile, "filename_soft_lock_release_run", 3), min_value=1, max_value=8),
        "cvn_assist_mix": _env_float("UTOA_SEQUENCE_CVN_ASSIST_MIX", _profile_float(profile, "cvn_assist_mix", 0.55), min_value=0.0, max_value=1.0),
        "cvn_backend": str(os.environ.get("UTOA_SEQUENCE_CVN_BACKEND", str(profile.get("cvn_backend", "") or "")) or "").strip().lower(),
        "cvn_smooth_window": _env_int("UTOA_SEQUENCE_CVN_SMOOTH_WINDOW", _profile_int(profile, "cvn_smooth_window", 5), min_value=1, max_value=25),
        "cvn_min_run_frames": _env_int("UTOA_SEQUENCE_CVN_MIN_RUN_FRAMES", _profile_int(profile, "cvn_min_run_frames", 3), min_value=1, max_value=40),
        "cvn_c_threshold": _safe_float(profile.get("cvn_c_threshold"), 0.0) if "cvn_c_threshold" in profile else None,
    }


def _analyze_wav_sequence(
    wav_path: str,
    *,
    language: str,
    format_hint: str,
    profile: Dict[str, object],
    callback=None,
) -> SequenceAnalysis:
    cfg = _load_sequence_runtime_config(language, profile)
    frame_ms = int(cfg["frame_ms"])
    hop_ms = int(cfg["hop_ms"])
    words, source = _load_transcript_words(wav_path, language)
    if not words:
        words = ["spn"]
        source = "fallback"
    filename_words = _filename_words_for_wav(wav_path, language)

    token_count = max(1, len(words))
    min_active_ratio = float(cfg["min_active_ratio"])
    max_active_ratio = float(cfg["max_active_ratio"])
    if token_count <= 2:
        min_active_ratio = max(
            min_active_ratio,
            _env_float("UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MIN", _profile_float(profile, "mono_active_ratio_min", 0.55), min_value=0.10, max_value=0.95),
        )
        max_active_ratio = max(
            max_active_ratio,
            _env_float("UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MAX", _profile_float(profile, "mono_active_ratio_max", 0.95), min_value=0.20, max_value=0.99),
        )
    elif token_count >= 6:
        max_active_ratio = min(
            max_active_ratio,
            _env_float("UTOA_SEQUENCE_ALIGN_MULTI_ACTIVE_RATIO_MAX", _profile_float(profile, "multi_active_ratio_max", 0.88), min_value=0.20, max_value=0.98),
        )

    signal, sr, duration_sec = _read_pcm_as_float_mono(wav_path)
    if signal.size <= 0 or duration_sec <= 0.0:
        raise RuntimeError("empty waveform")
    analysis_signal = _prepare_analysis_signal(
        signal,
        sr,
        dc_remove=bool(cfg["analysis_dc_remove"]),
        highpass_hz=float(cfg["analysis_highpass_hz"]),
        max_gain_db=float(cfg["analysis_max_gain_db"]),
        target_rms=float(cfg["analysis_target_rms"]),
    )
    rms, hop_sec = _framewise_rms(analysis_signal, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if rms.size <= 0 or hop_sec <= 0.0:
        raise RuntimeError("empty frame sequence")
    zcr, hf_ratio, flux = _framewise_consonant_cues(analysis_signal, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    silence_threshold = _resolve_silence_threshold(
        rms,
        base_quantile=float(cfg["silence_q"]),
        scale=float(cfg["silence_scale"]),
        min_active_ratio=min_active_ratio,
        max_active_ratio=max_active_ratio,
    )
    min_active_frames = max(1, int(round((float(cfg["min_active_ms"]) / 1000.0) / max(hop_sec, 1e-4))))
    max_gap_frames = max(0, int(round((float(cfg["max_gap_ms"]) / 1000.0) / max(hop_sec, 1e-4))))
    voicing_mask = _build_frame_voicing_mask(
        rms,
        silence_threshold=float(silence_threshold),
        min_active_frames=min_active_frames,
        max_gap_frames=max_gap_frames,
        zcr=zcr,
        hf_ratio=hf_ratio,
        flux=flux,
        lookback_frames=max(1, int(round((float(cfg["onset_lookback_ms"]) / 1000.0) / max(hop_sec, 1e-4)))),
    )
    ap_mask = None
    if bool(cfg["ap_detect_enable"]):
        ap_mask = _build_ap_sp_mask(
            rms,
            silence_threshold=float(silence_threshold),
            zcr=zcr,
            hf_ratio=hf_ratio,
            flux=flux,
            min_run_frames=max(1, int(round((float(cfg["ap_min_run_ms"]) / 1000.0) / max(hop_sec, 1e-4)))),
            max_gap_frames=max(0, int(round((float(cfg["ap_max_gap_ms"]) / 1000.0) / max(hop_sec, 1e-4)))),
        )
    labels = _estimate_sequence_labels(
        rms,
        silence_threshold=float(silence_threshold),
        activity_mask=voicing_mask,
        zcr=zcr,
        hf_ratio=hf_ratio,
        flux=flux,
    )
    labels = _inject_ap_labels(labels, ap_mask=ap_mask, rms=rms, silence_threshold=float(silence_threshold))
    cvn_c_prob, cvn_backend = _compute_sequence_cvn_consonant_prob(
        wav_path=wav_path,
        language=language,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        frame_count=int(rms.size),
        hop_sec=float(hop_sec),
        backend_override=str(cfg["cvn_backend"] or "") or None,
        smooth_window=int(cfg["cvn_smooth_window"]),
        min_run_frames=int(cfg["cvn_min_run_frames"]),
        c_threshold=cfg["cvn_c_threshold"],
        callback=callback,
    )

    word_rows = _resolve_word_rows(
        words,
        duration_sec=float(duration_sec),
        labels=labels,
        hop_sec=float(hop_sec),
        min_span_ms=float(cfg["min_span_ms"]),
        language=language,
        format_hint=format_hint,
    )
    soft_lock_ready = False
    if bool(cfg["filename_soft_lock_enable"]) and filename_words and len(words) >= 2:
        order_match_ratio = _order_match_ratio(language, words, filename_words)
        soft_lock_ready = bool(source == "filename" or order_match_ratio >= 0.70)
    word_rows = _refine_word_boundaries_with_onset_cues(
        word_rows,
        duration_sec=float(duration_sec),
        rms=rms,
        zcr=zcr,
        hf_ratio=hf_ratio,
        flux=flux,
        labels=labels,
        language=language,
        hop_sec=float(hop_sec),
        silence_threshold=float(silence_threshold),
        drift_guard_level=int(cfg["drift_guard_level"]),
        drift_guard_max_shift_ms=float(cfg["drift_guard_max_shift_ms"]),
        onset_snap_right_bias=float(cfg["onset_snap_right_bias"]),
        plosive_onset_lookback_ms=float(cfg["plosive_onset_lookback_ms"]),
        plosive_left_bias_ms=float(cfg["plosive_left_bias_ms"]),
        filename_soft_lock=soft_lock_ready,
        soft_lock_conf_threshold=float(cfg["filename_soft_lock_conf_threshold"]),
        soft_lock_max_shift_ms=float(cfg["filename_soft_lock_max_shift_ms"]),
        soft_lock_release_run=int(cfg["filename_soft_lock_release_run"]),
        cvn_c_prob=cvn_c_prob,
        cvn_assist_mix=float(cfg["cvn_assist_mix"]),
    )
    word_rows = _insert_internal_pause_rows(
        word_rows,
        duration_sec=float(duration_sec),
        labels=labels,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf_ratio,
        flux=flux,
        ap_mask=ap_mask,
        hop_sec=float(hop_sec),
        silence_threshold=float(silence_threshold),
        min_pause_ms=float(cfg["pause_min_ms"]),
        search_ms=float(cfg["pause_search_ms"]),
        energy_pause_scale=float(cfg["pause_energy_scale"]),
        hard_silence_scale=float(cfg["pause_hard_silence_scale"]),
        unvoiced_guard_scale=float(cfg["pause_unvoiced_guard_scale"]),
    )
    word_rows = _apply_uniform_boundary_advance(
        word_rows,
        duration_sec=float(duration_sec),
        advance_ms=float(cfg["boundary_advance_ms"]),
    )

    # Phone rows are currently diagnostic features for downstream residual
    # models. Keeping them here avoids requiring TextGrid generation.
    try:
        from core.sequence_aligner import _build_phone_rows

        phone_rows = _build_phone_rows(
            word_rows,
            duration_sec=float(duration_sec),
            language=language,
            labels=labels,
            cvn_c_prob=cvn_c_prob,
            hop_sec=float(hop_sec),
        )
    except Exception:
        phone_rows = []

    return SequenceAnalysis(
        wav_path=wav_path,
        wav_name=os.path.basename(wav_path),
        duration_ms=float(duration_sec) * 1000.0,
        words=[str(x) for x in words],
        source=str(source),
        filename_words=[str(x) for x in filename_words],
        word_rows=list(word_rows),
        phone_rows=list(phone_rows),
        labels=list(labels),
        rms=np.asarray(rms, dtype=np.float32),
        zcr=np.asarray(zcr, dtype=np.float32),
        hf_ratio=np.asarray(hf_ratio, dtype=np.float32),
        flux=np.asarray(flux, dtype=np.float32),
        cvn_c_prob=None if cvn_c_prob is None else np.asarray(cvn_c_prob, dtype=np.float32),
        hop_sec=float(hop_sec),
        silence_threshold=float(silence_threshold),
        soft_lock_ready=soft_lock_ready,
        cvn_backend=str(cvn_backend or ""),
    )


def _non_pause_word_rows(analysis: SequenceAnalysis) -> List[Tuple[float, float, str]]:
    out = []
    for start, end, mark in analysis.word_rows:
        if str(mark or "").strip().lower() in {"", "spn", "sil", "pau", "br", "breath"}:
            continue
        out.append((float(start) * 1000.0, float(end) * 1000.0, str(mark)))
    return out


def _pick_anchor(
    row: ManualOtoRow,
    wav_rows: Sequence[ManualOtoRow],
    analysis: SequenceAnalysis,
) -> Tuple[float, float, str, int, str]:
    anchors = _non_pause_word_rows(analysis)
    if not anchors:
        return 0.0, analysis.duration_ms, "fallback_full_file", 0, "spn"
    manual_cutoff_abs = _safe_cutoff_abs_ms(row.offset_ms, row.cutoff, analysis.duration_ms)
    if _manual_timing_looks_configured(
        offset_ms=row.offset_ms,
        cons_ms=row.cons_ms,
        cutoff=row.cutoff,
        pre_ms=row.pre_ms,
        ovl_ms=row.ovl_ms,
        cutoff_abs_ms=manual_cutoff_abs,
    ):
        teacher_point = float(np.clip(row.offset_ms + max(0.0, row.pre_ms), 0.0, analysis.duration_ms))
        best_idx = 0
        best_score = float("inf")
        for idx, (start, end, _token) in enumerate(anchors):
            if float(start) <= teacher_point <= float(end):
                center = (float(start) + float(end)) * 0.5
                score = abs(teacher_point - center) * 0.15
            else:
                score = min(abs(teacher_point - float(start)), abs(teacher_point - float(end)))
            if score < best_score:
                best_score = score
                best_idx = idx
        start, end, token = anchors[best_idx]
        return float(start), float(end), "manual_teacher_point", best_idx, str(token)
    if len(wav_rows) <= 1:
        idx = 0
    else:
        ratio = float(row.wav_row_index) / float(max(1, len(wav_rows) - 1))
        idx = int(round(ratio * float(max(0, len(anchors) - 1))))
    idx = max(0, min(len(anchors) - 1, idx))
    start, end, token = anchors[idx]
    return float(start), float(end), "file_order_ratio", idx, str(token)


def _baseline_params_from_anchor(start_ms: float, end_ms: float, alias_type: str) -> Dict[str, float]:
    start = max(0.0, float(start_ms))
    end = max(start + 1.0, float(end_ms))
    dur = max(1.0, end - start)
    a_type = str(alias_type or "").strip().lower()
    if a_type == ENDBREATH_ALIAS_TYPE:
        pre = float(np.clip(dur * 0.30, 6.0, 110.0))
        cons = float(np.clip(pre + dur * 0.24, pre + 6.0, max(pre + 6.0, dur - 2.0)))
        ovl = float(np.clip(pre * 0.34, 0.0, max(0.0, pre - 4.0)))
    elif a_type in {"vc", "vv", "vcv"}:
        pre = float(np.clip(dur * 0.35, 8.0, 140.0))
        cons = float(np.clip(pre + dur * 0.34, pre + 8.0, max(pre + 8.0, dur - 2.0)))
        ovl = float(np.clip(pre * 0.42, 0.0, max(0.0, pre - 4.0)))
    elif a_type == "cv_head":
        pre = float(np.clip(dur * 0.40, 20.0, 170.0))
        cons = float(np.clip(pre + dur * 0.24, pre + 8.0, max(pre + 8.0, dur - 2.0)))
        ovl = float(np.clip(pre * 0.28, 0.0, max(0.0, pre - 4.0)))
    else:
        pre = float(np.clip(dur * 0.43, 16.0, 180.0))
        cons = float(np.clip(pre + dur * 0.26, pre + 8.0, max(pre + 8.0, dur - 2.0)))
        ovl = float(np.clip(pre * 0.45, 0.0, max(0.0, pre - 4.0)))
    return {
        "base_offset": start,
        "base_pre": pre,
        "base_cons": cons,
        "base_cutoff_abs": end,
        "base_cutoff": -(end - start),
        "base_ovl": ovl,
    }


def _slice_stats(values: np.ndarray, start_ms: float, end_ms: float, hop_sec: float) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0 or hop_sec <= 0.0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    i0 = max(0, int(math.floor((float(start_ms) / 1000.0) / float(hop_sec))))
    i1 = min(arr.size, int(math.ceil((float(end_ms) / 1000.0) / float(hop_sec))) + 1)
    if i1 <= i0:
        i1 = min(arr.size, i0 + 1)
    sub = arr[i0:i1]
    if sub.size <= 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(sub)),
        "max": float(np.max(sub)),
        "min": float(np.min(sub)),
        "std": float(np.std(sub)),
    }


def _label_stats(labels: Sequence[str], start_ms: float, end_ms: float, hop_sec: float) -> Dict[str, float]:
    if not labels or hop_sec <= 0.0:
        return {"silence_ratio": 0.0, "vowel_ratio": 0.0, "onset_ratio": 0.0, "breath_ratio": 0.0}
    i0 = max(0, int(math.floor((float(start_ms) / 1000.0) / float(hop_sec))))
    i1 = min(len(labels), int(math.ceil((float(end_ms) / 1000.0) / float(hop_sec))) + 1)
    if i1 <= i0:
        i1 = min(len(labels), i0 + 1)
    subset = [str(x or "").strip().lower() for x in labels[i0:i1]]
    n = max(1, len(subset))
    return {
        "silence_ratio": float(sum(x in {"sil", "weak_noise"} for x in subset)) / n,
        "vowel_ratio": float(sum(x in {"stable_vowel", "vowel_transition"} for x in subset)) / n,
        "onset_ratio": float(sum(x in {"voiced_onset", "unvoiced_onset"} for x in subset)) / n,
        "breath_ratio": float(sum(x == "breath" for x in subset)) / n,
    }


def _row_to_record(
    row: ManualOtoRow,
    wav_rows: Sequence[ManualOtoRow],
    analysis: SequenceAnalysis,
    *,
    voicebank_id: str,
    language: str,
    format_type: str,
) -> Dict[str, object]:
    anchor_start, anchor_end, anchor_strategy, anchor_index, anchor_token = _pick_anchor(row, wav_rows, analysis)
    base = _baseline_params_from_anchor(anchor_start, anchor_end, row.alias_type)
    manual_cutoff_abs = _safe_cutoff_abs_ms(row.offset_ms, row.cutoff, analysis.duration_ms)
    manual_timing_valid = _manual_timing_looks_configured(
        offset_ms=row.offset_ms,
        cons_ms=row.cons_ms,
        cutoff=row.cutoff,
        pre_ms=row.pre_ms,
        ovl_ms=row.ovl_ms,
        cutoff_abs_ms=manual_cutoff_abs,
    )
    anchor_mid = (anchor_start + anchor_end) * 0.5
    local_start = max(0.0, anchor_start - 45.0)
    local_end = min(analysis.duration_ms, anchor_end + 45.0)
    onset_start = max(0.0, anchor_start - 35.0)
    onset_end = min(analysis.duration_ms, anchor_start + 55.0)
    tail_start = max(0.0, anchor_end - 55.0)
    tail_end = min(analysis.duration_ms, anchor_end + 35.0)

    rms_all = _slice_stats(analysis.rms, local_start, local_end, analysis.hop_sec)
    rms_onset = _slice_stats(analysis.rms, onset_start, onset_end, analysis.hop_sec)
    rms_tail = _slice_stats(analysis.rms, tail_start, tail_end, analysis.hop_sec)
    zcr_onset = _slice_stats(analysis.zcr, onset_start, onset_end, analysis.hop_sec)
    hf_onset = _slice_stats(analysis.hf_ratio, onset_start, onset_end, analysis.hop_sec)
    flux_onset = _slice_stats(analysis.flux, onset_start, onset_end, analysis.hop_sec)
    labels_local = _label_stats(analysis.labels, local_start, local_end, analysis.hop_sec)

    row_count = max(1, len(wav_rows))
    word_count = max(1, len(_non_pause_word_rows(analysis)))
    return {
        "schema": "sequence_residual_preprocessor_v1",
        "voicebank_id": voicebank_id,
        "language": language,
        "format_type": format_type,
        "wav": row.wav,
        "wav_norm": row.wav_norm,
        "wav_path": analysis.wav_path,
        "alias": row.alias,
        "alias_norm": row.alias_norm,
        "alias_type": row.alias_type,
        "alias_group": ENDBREATH_ALIAS_TYPE if row.alias_type == ENDBREATH_ALIAS_TYPE else "bridge" if row.alias_type in {"vc", "vv", "vcv"} else "cv" if row.alias_type in {"cv", "cv_head", "mono"} else row.alias_type,
        "alias_occurrence": row.alias_occurrence,
        "line_index": row.line_index,
        "row_index_in_wav": row.wav_row_index,
        "row_ratio_in_wav": float(row.wav_row_index) / float(max(1, row_count - 1)),
        "file_row_count": row_count,
        "sequence_word_count": word_count,
        "sequence_source": analysis.source,
        "sequence_soft_lock": bool(analysis.soft_lock_ready),
        "sequence_cvn_backend": analysis.cvn_backend,
        "filename_words": analysis.filename_words,
        "sequence_words": analysis.words,
        "anchor_match_strategy": anchor_strategy,
        "anchor_index": anchor_index,
        "anchor_token": anchor_token,
        "anchor_start_ms": anchor_start,
        "anchor_end_ms": anchor_end,
        "anchor_mid_ms": anchor_mid,
        "anchor_len_ms": max(0.0, anchor_end - anchor_start),
        "wav_duration_ms": analysis.duration_ms,
        "base_offset": base["base_offset"],
        "base_pre": base["base_pre"],
        "base_cons": base["base_cons"],
        "base_cutoff": base["base_cutoff"],
        "base_cutoff_abs": base["base_cutoff_abs"],
        "base_ovl": base["base_ovl"],
        "manual_offset": row.offset_ms,
        "manual_pre": row.pre_ms,
        "manual_cons": row.cons_ms,
        "manual_cutoff": row.cutoff,
        "manual_cutoff_abs": manual_cutoff_abs,
        "manual_ovl": row.ovl_ms,
        "delta_offset": row.offset_ms - float(base["base_offset"]),
        "delta_pre": row.pre_ms - float(base["base_pre"]),
        "delta_cons": row.cons_ms - float(base["base_cons"]),
        "delta_cutoff": manual_cutoff_abs - float(base["base_cutoff_abs"]),
        "delta_ovl": row.ovl_ms - float(base["base_ovl"]),
        "rms_local_mean": rms_all["mean"],
        "rms_local_max": rms_all["max"],
        "rms_local_std": rms_all["std"],
        "rms_onset_mean": rms_onset["mean"],
        "rms_onset_max": rms_onset["max"],
        "rms_tail_mean": rms_tail["mean"],
        "rms_tail_min": rms_tail["min"],
        "zcr_onset_mean": zcr_onset["mean"],
        "zcr_onset_max": zcr_onset["max"],
        "hf_onset_mean": hf_onset["mean"],
        "hf_onset_max": hf_onset["max"],
        "flux_onset_mean": flux_onset["mean"],
        "flux_onset_max": flux_onset["max"],
        "label_silence_ratio": labels_local["silence_ratio"],
        "label_vowel_ratio": labels_local["vowel_ratio"],
        "label_onset_ratio": labels_local["onset_ratio"],
        "label_breath_ratio": labels_local["breath_ratio"],
        "manual_timing_valid": bool(manual_timing_valid),
        "skip_reason": "pause_alias" if row.alias_type in PAUSE_ALIAS_TYPES else "unconfigured_manual_timing" if not manual_timing_valid else "",
        "skip_training": row.alias_type in PAUSE_ALIAS_TYPES or not manual_timing_valid,
    }


def _summarize_records(records: Sequence[Dict[str, object]], errors: Sequence[str]) -> Dict[str, object]:
    usable = [r for r in records if not bool(r.get("skip_training", False))]
    alias_counter = Counter(str(r.get("alias_type", "unknown") or "unknown") for r in records)
    source_counter = Counter(str(r.get("sequence_source", "unknown") or "unknown") for r in records)
    lock_counter = Counter("on" if bool(r.get("sequence_soft_lock")) else "off" for r in records)

    def _abs_stats(name: str) -> Dict[str, float]:
        vals = [abs(_safe_float(r.get(name, 0.0))) for r in usable]
        if not vals:
            return {"mae": 0.0, "median_abs": 0.0, "p90_abs": 0.0, "max_abs": 0.0}
        sorted_vals = sorted(vals)
        p90_idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * 0.90)))
        return {
            "mae": float(sum(vals) / len(vals)),
            "median_abs": float(median(vals)),
            "p90_abs": float(sorted_vals[p90_idx]),
            "max_abs": float(max(vals)),
        }

    return {
        "schema": "sequence_residual_preprocessor_summary_v1",
        "rows": len(records),
        "usable_rows": len(usable),
        "files": len({str(r.get("wav_norm", "")) for r in records}),
        "errors": list(errors),
        "error_count": len(errors),
        "alias_type_counts": dict(alias_counter),
        "skip_reason_counts": dict(Counter(str(r.get("skip_reason", "") or "usable") for r in records)),
        "sequence_source_counts": dict(source_counter),
        "sequence_soft_lock_counts": dict(lock_counter),
        "residual_abs": {
            "offset": _abs_stats("delta_offset"),
            "pre": _abs_stats("delta_pre"),
            "cons": _abs_stats("delta_cons"),
            "cutoff_abs": _abs_stats("delta_cutoff"),
            "ovl": _abs_stats("delta_ovl"),
        },
    }


def build_sequence_residual_dataset(
    *,
    voicebank_dir: str,
    manual_oto_path: str,
    language: str = "korean",
    format_type: str = "",
    max_files: int = 0,
    callback=None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    lang = str(language or "").strip().lower() or "korean"
    fmt = str(format_type or "").strip().lower()
    voicebank_abs = os.path.abspath(str(voicebank_dir or "").strip())
    oto_abs = os.path.abspath(str(manual_oto_path or "").strip())
    voicebank_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", os.path.basename(voicebank_abs).strip()) or "voicebank"
    rows = _read_manual_oto_rows(oto_abs, lang)
    wav_index = _build_wav_index(voicebank_abs)
    grouped: Dict[str, List[ManualOtoRow]] = defaultdict(list)
    for row in rows:
        grouped[row.wav_norm].append(row)

    profile = _load_sequence_profile(lang, callback=callback)
    records: List[Dict[str, object]] = []
    errors: List[str] = []
    processed_files = 0
    for wav_norm, wav_rows in grouped.items():
        if max_files and processed_files >= int(max_files):
            break
        wav_path = wav_index.get(wav_norm)
        if not wav_path:
            errors.append(f"missing_wav:{wav_rows[0].wav}")
            continue
        try:
            analysis = _analyze_wav_sequence(
                wav_path,
                language=lang,
                format_hint=fmt,
                profile=profile,
                callback=callback,
            )
            for row in wav_rows:
                records.append(
                    _row_to_record(
                        row,
                        wav_rows,
                        analysis,
                        voicebank_id=voicebank_id,
                        language=lang,
                        format_type=fmt,
                    )
                )
            processed_files += 1
            _emit(callback, f"[SEQ-RESID] file={os.path.basename(wav_path)} rows={len(wav_rows)} words={len(analysis.words)} source={analysis.source}")
        except Exception as exc:
            errors.append(f"{wav_rows[0].wav}:{exc}")
            _emit(callback, f"[SEQ-RESID] failed file={wav_rows[0].wav}: {exc}")

    summary = _summarize_records(records, errors)
    summary.update(
        {
            "voicebank_dir": voicebank_abs,
            "manual_oto_path": oto_abs,
            "language": lang,
            "format_type": fmt,
            "processed_files": processed_files,
            "manual_wav_groups": len(grouped),
        }
    )
    return records, summary


def write_sequence_residual_dataset(
    *,
    voicebank_dir: str,
    manual_oto_path: str,
    output_dir: str,
    language: str = "korean",
    format_type: str = "",
    max_files: int = 0,
    callback=None,
) -> Dict[str, object]:
    records, summary = build_sequence_residual_dataset(
        voicebank_dir=voicebank_dir,
        manual_oto_path=manual_oto_path,
        language=language,
        format_type=format_type,
        max_files=max_files,
        callback=callback,
    )
    out_dir = os.path.abspath(str(output_dir or "").strip())
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "sequence_residual_rows.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    with open(rows_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return {
        "rows_path": rows_path,
        "summary_path": summary_path,
        "summary": summary,
    }


__all__ = [
    "build_sequence_residual_dataset",
    "write_sequence_residual_dataset",
]
