"""
Pre-ML mel-based safety clamp for OTO offset/cutoff.

Runs before the ML refinement stage in both KR and JA finalize pipelines.
Detects rows whose offset or cutoff land deep inside mel-detected silence
and soft-clamps them toward the nearest mel boundary candidate.

This is NOT a replacement for the per-row soft guard
(_apply_soft_mel_offset_cutoff_guard) applied during generation.  That guard
addresses large-scale misplacement at row-creation time.  This file-level
clamp catches residual cases that slipped through — especially for VC/VV
alias types that the per-row guard skips — and pre-corrects them so the
ML delta refiner has a better starting point.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_normalization import normalize_wav_key

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _clamp01(v: float) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _frame_blank_confidence(
    *,
    e_v: float,
    db_v: float,
    sil_v: float,
    f0_v: float,
    voiced_v: float,
    unvoiced_v: float,
    db_sil_th: float,
) -> float:
    db_low = _clamp01((float(db_sil_th) + 3.0 - float(db_v)) / 8.0)
    en_low = _clamp01((0.12 - float(e_v)) / 0.12)
    f0_low = _clamp01((0.20 - float(f0_v)) / 0.20)
    voiced_high = _clamp01(float(voiced_v))
    unvoiced_high = _clamp01(float(unvoiced_v))
    sil_high = _clamp01(float(sil_v))

    # Blank if silence/dB/energy are all weak, with low F0 as auxiliary cue.
    # Unvoiced confidence is a negative term to avoid treating fricative/plosive onset as blank.
    score = (
        (0.46 * sil_high)
        + (0.24 * db_low)
        + (0.20 * en_low)
        + (0.10 * f0_low)
        - (0.24 * voiced_high)
        - (0.14 * unvoiced_high)
    )
    return _clamp01(score)


def _frame_sound_confidence(
    *,
    e_v: float,
    db_v: float,
    sil_v: float,
    f0_v: float,
    voiced_v: float,
    unvoiced_v: float,
    db_sil_th: float,
) -> float:
    db_sound = _clamp01((float(db_v) - (float(db_sil_th) + 0.8)) / 10.0)
    en_sound = _clamp01((float(e_v) - 0.06) / 0.22)
    f0_sound = _clamp01((float(f0_v) - 0.18) / 0.42)
    voiced_high = _clamp01(float(voiced_v))
    unvoiced_high = _clamp01(float(unvoiced_v))
    sil_high = _clamp01(float(sil_v))
    score = (
        (0.34 * voiced_high)
        + (0.28 * unvoiced_high)
        + (0.20 * db_sound)
        + (0.14 * en_sound)
        + (0.04 * f0_sound)
        - (0.30 * sil_high)
    )
    return _clamp01(score)


# ---------------------------------------------------------------------------
# Core clamp logic
# ---------------------------------------------------------------------------

def _find_first_sound_frame(
    t_ms,
    en,
    db_arr,
    f0_arr,
    cls_voiced,
    cls_unvoiced,
    cls_silence,
    db_sil_th: float,
    search_start_ms: float,
    search_end_ms: float,
) -> Optional[float]:
    """Find the first mel frame that looks like audible sound in [start, end]."""
    if t_ms is None or len(t_ms) == 0:
        return None
    mask = np.where((t_ms >= search_start_ms) & (t_ms <= search_end_ms))[0]
    if len(mask) == 0:
        return None
    sound_frame_th = _env_float("UTOA_SAFETY_CLAMP_SOUND_FRAME_TH", 0.36)
    blank_frame_th = _env_float("UTOA_SAFETY_CLAMP_BLANK_FRAME_TH", 0.62)
    for idx in mask:
        voiced_v = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(en) else 0.0
        unvoiced_v = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(en) else 0.0
        silence_v = float(cls_silence[idx]) if cls_silence is not None and len(cls_silence) == len(en) else 0.0
        db_v = float(db_arr[idx]) if db_arr is not None and len(db_arr) == len(en) else -60.0
        e_v = float(en[idx])
        f0_v = float(f0_arr[idx]) if f0_arr is not None and len(f0_arr) == len(en) else 0.0
        sound_conf = _frame_sound_confidence(
            e_v=e_v,
            db_v=db_v,
            sil_v=silence_v,
            f0_v=f0_v,
            voiced_v=voiced_v,
            unvoiced_v=unvoiced_v,
            db_sil_th=db_sil_th,
        )
        blank_conf = _frame_blank_confidence(
            e_v=e_v,
            db_v=db_v,
            sil_v=silence_v,
            f0_v=f0_v,
            voiced_v=voiced_v,
            unvoiced_v=unvoiced_v,
            db_sil_th=db_sil_th,
        )
        if sound_conf >= sound_frame_th and blank_conf <= blank_frame_th:
            return float(t_ms[idx])
        if db_v > (db_sil_th + 1.2) and e_v > 0.10 and silence_v < 0.62:
            return float(t_ms[idx])
    return None


def _find_last_sound_frame(
    t_ms,
    en,
    db_arr,
    f0_arr,
    cls_voiced,
    cls_unvoiced,
    cls_silence,
    db_sil_th: float,
    search_start_ms: float,
    search_end_ms: float,
) -> Optional[float]:
    """Find the last mel frame that looks like audible sound in [start, end]."""
    if t_ms is None or len(t_ms) == 0:
        return None
    mask = np.where((t_ms >= search_start_ms) & (t_ms <= search_end_ms))[0]
    if len(mask) == 0:
        return None
    sound_frame_th = _env_float("UTOA_SAFETY_CLAMP_SOUND_FRAME_TH", 0.36)
    blank_frame_th = _env_float("UTOA_SAFETY_CLAMP_BLANK_FRAME_TH", 0.62)
    for idx in reversed(mask):
        voiced_v = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(en) else 0.0
        unvoiced_v = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(en) else 0.0
        silence_v = float(cls_silence[idx]) if cls_silence is not None and len(cls_silence) == len(en) else 0.0
        db_v = float(db_arr[idx]) if db_arr is not None and len(db_arr) == len(en) else -60.0
        e_v = float(en[idx])
        f0_v = float(f0_arr[idx]) if f0_arr is not None and len(f0_arr) == len(en) else 0.0
        sound_conf = _frame_sound_confidence(
            e_v=e_v,
            db_v=db_v,
            sil_v=silence_v,
            f0_v=f0_v,
            voiced_v=voiced_v,
            unvoiced_v=unvoiced_v,
            db_sil_th=db_sil_th,
        )
        blank_conf = _frame_blank_confidence(
            e_v=e_v,
            db_v=db_v,
            sil_v=silence_v,
            f0_v=f0_v,
            voiced_v=voiced_v,
            unvoiced_v=unvoiced_v,
            db_sil_th=db_sil_th,
        )
        if sound_conf >= sound_frame_th and blank_conf <= blank_frame_th:
            return float(t_ms[idx])
        if db_v > (db_sil_th + 1.2) and e_v > 0.10 and silence_v < 0.62:
            return float(t_ms[idx])
    return None


def _compute_silence_ratio(
    t_ms,
    en,
    db_arr,
    f0_arr,
    cls_voiced,
    cls_unvoiced,
    cls_silence,
    db_sil_th: float,
    start_ms: float,
    end_ms: float,
) -> float:
    """Fraction of frames in [start, end] that are classified as silence."""
    if t_ms is None or len(t_ms) == 0:
        return 0.0
    mask = np.where((t_ms >= start_ms) & (t_ms <= end_ms))[0]
    if len(mask) == 0:
        return 0.0
    blank_frame_th = _env_float("UTOA_SAFETY_CLAMP_BLANK_FRAME_TH", 0.62)
    flags = []
    for idx in mask:
        voiced_v = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(en) else 0.0
        unvoiced_v = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(en) else 0.0
        silence_v = float(cls_silence[idx]) if cls_silence is not None and len(cls_silence) == len(en) else 0.0
        db_v = float(db_arr[idx]) if db_arr is not None and len(db_arr) == len(en) else -60.0
        e_v = float(en[idx])
        f0_v = float(f0_arr[idx]) if f0_arr is not None and len(f0_arr) == len(en) else 0.0
        blank_conf = _frame_blank_confidence(
            e_v=e_v,
            db_v=db_v,
            sil_v=silence_v,
            f0_v=f0_v,
            voiced_v=voiced_v,
            unvoiced_v=unvoiced_v,
            db_sil_th=db_sil_th,
        )
        flags.append(blank_conf >= blank_frame_th)
    if flags:
        return float(np.mean(np.asarray(flags, dtype=np.float64)))
    return 0.0


# ---------------------------------------------------------------------------
# Per-row clamp
# ---------------------------------------------------------------------------

def _clamp_row(
    row: Dict[str, object],
    alias_type: str,
    mel_ctx: Dict[str, object],
    *,
    language: str = "",
    validate_fn: Callable,
) -> bool:
    """
    Attempt to clamp one row's offset/cutoff away from silence.
    Returns True if any parameter was modified.
    """
    t_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    db_arr = mel_ctx.get("db_db")
    f0_arr = mel_ctx.get("f0_voicing")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    cls_unvoiced = mel_ctx.get("cls_unvoiced_diffuse")
    cls_silence = mel_ctx.get("cls_silence_sparse")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))

    if t_ms is None or en is None or len(t_ms) < 8:
        return False

    off = float(row["offset"])
    pre_rel = float(row["pre"])
    cons_rel = float(row["cons"])
    cut_rel = abs(float(row["cutoff"]))
    ovl_rel = float(row["ovl"])

    pre_abs = off + pre_rel
    cut_abs = off + cut_rel

    if cut_abs <= pre_abs + 16.0:
        return False

    changed = False

    # --- Offset clamp: push offset forward if it sits in deep silence ---
    # Only for CV/CV_HEAD/VCV that have onset-side silence
    lang = str(language or "").strip().lower()
    offset_margin_ms = _env_float("UTOA_SAFETY_CLAMP_OFFSET_MARGIN_MS", 12.0)
    offset_sil_threshold = _env_float(
        "UTOA_SAFETY_CLAMP_OFFSET_SIL_THRESHOLD",
        0.68 if lang == "korean" else 0.72,
    )

    if alias_type in {"cv", "cv_head", "vcv"}:
        # Check silence ratio in the offset → pre region
        check_end = min(pre_abs, off + 80.0)
        sil_ratio = _compute_silence_ratio(
            t_ms,
            en,
            db_arr,
            f0_arr,
            cls_voiced,
            cls_unvoiced,
            cls_silence,
            db_sil_th,
            off,
            check_end,
        )
        if sil_ratio >= offset_sil_threshold:
            first_sound = _find_first_sound_frame(
                t_ms, en, db_arr, f0_arr, cls_voiced, cls_unvoiced, cls_silence, db_sil_th,
                max(0.0, off), pre_abs + 20.0,
            )
            if first_sound is not None and first_sound > off + offset_margin_ms:
                new_off = max(0.0, first_sound - offset_margin_ms)
                if new_off > off + 4.0 and new_off < pre_abs - 8.0:
                    # Shift offset forward; adjust relative parameters
                    shift = new_off - off
                    row["offset"] = new_off
                    row["pre"] = max(0.0, pre_rel - shift)
                    row["cons"] = max(row["pre"] + 6.0, cons_rel - shift)
                    row["ovl"] = max(0.0, min(ovl_rel - shift, row["pre"] * 0.82))
                    # cutoff stays at same absolute position
                    row["cutoff"] = -(cut_abs - new_off)
                    changed = True

    # --- Cutoff clamp: pull cutoff back if tail is deep silence ---
    cutoff_sil_threshold = _env_float(
        "UTOA_SAFETY_CLAMP_CUTOFF_SIL_THRESHOLD",
        0.66 if lang == "korean" else 0.70,
    )
    cutoff_margin_ms = _env_float("UTOA_SAFETY_CLAMP_CUTOFF_MARGIN_MS", 8.0)
    min_active_ms = _env_float("UTOA_SAFETY_CLAMP_MIN_ACTIVE_MS", 20.0)

    # Re-read in case offset shifted
    off = float(row["offset"])
    pre_abs = off + float(row["pre"])
    cons_abs = off + float(row["cons"])
    cut_abs = off + abs(float(row["cutoff"]))

    # Check silence in the cons → cutoff tail region (all alias types)
    tail_start = max(cons_abs, pre_abs + 10.0)
    if cut_abs > tail_start + 20.0:
        sil_ratio = _compute_silence_ratio(
            t_ms,
            en,
            db_arr,
            f0_arr,
            cls_voiced,
            cls_unvoiced,
            cls_silence,
            db_sil_th,
            tail_start,
            cut_abs,
        )
        if sil_ratio >= cutoff_sil_threshold:
            last_sound = _find_last_sound_frame(
                t_ms, en, db_arr, f0_arr, cls_voiced, cls_unvoiced, cls_silence, db_sil_th,
                pre_abs, cut_abs,
            )
            if last_sound is not None:
                new_cut_abs = last_sound + cutoff_margin_ms
                # Ensure minimum active region
                if new_cut_abs >= cons_abs + min_active_ms and new_cut_abs < cut_abs - 4.0:
                    row["cutoff"] = -(new_cut_abs - off)
                    # Adjust cons if it exceeds new cutoff
                    if cons_abs > new_cut_abs - 12.0:
                        row["cons"] = max(float(row["pre"]) + 6.0, new_cut_abs - off - 12.0)
                    changed = True

    if changed:
        # Re-validate
        o, c, ct, p, ov = validate_fn(
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        row["offset"] = o
        row["cons"] = c
        row["cutoff"] = ct
        row["pre"] = p
        row["ovl"] = ov

    return changed


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

def apply_mel_safety_clamp_to_oto_file(
    oto_path: str,
    wav_dir: str,
    *,
    classify_alias_fn: Callable[[str], str],
    validate_fn: Callable,
    normalize_key_fn: Optional[Callable] = None,
    language: str = "",
) -> int:
    """
    Apply mel-based safety clamping to all rows in an oto file.

    Returns the number of changed rows.
    """
    if _env_flag("UTOA_DISABLE_MEL_SAFETY_CLAMP", False):
        return 0
    if np is None:
        return 0
    if not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    from core.oto_generator import _find_wav_path_for_name, _mel_envelope, _read_wav_mono_np

    norm_key = normalize_key_fn or normalize_wav_key

    raw_lines: List[str] = []
    parsed: List[tuple] = []
    for idx, raw in enumerate(read_text_with_fallback(oto_path).splitlines()):
        line = raw.rstrip("\n")
        raw_lines.append(line)
        row = parse_oto_line(line)
        if row:
            parsed.append((idx, row))
    if not parsed:
        return 0

    # Build wav index
    wav_index: Dict[str, str] = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                key = norm_key(fn)
                if key:
                    wav_index[key] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav: Dict[str, List[tuple]] = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))

    alias_type_cache: Dict[str, str] = {}

    def _classify_cached(alias_text: str) -> str:
        key = str(alias_text or "")
        out = alias_type_cache.get(key)
        if out is None:
            out = classify_alias_fn(key)
            alias_type_cache[key] = out
        return out

    mel_cache: Dict[str, object] = {}
    changed = 0

    for wav_name, rows in by_wav.items():
        wav_path = _find_wav_path_for_name(wav_name, wav_dir, wav_index)
        if not wav_path:
            continue
        mel_ctx = mel_cache.get(wav_path)
        if mel_ctx is None:
            audio, sr = _read_wav_mono_np(wav_path)
            mel_ctx = _mel_envelope(audio, sr)
            mel_cache[wav_path] = mel_ctx
        if not mel_ctx:
            continue

        for _line_idx, row in rows:
            alias_type = _classify_cached(str(row.get("alias", "")))
            if alias_type not in {"cv", "cv_head", "vcv", "vc", "vv"}:
                continue
            if _clamp_row(
                row,
                alias_type,
                mel_ctx,
                language=language,
                validate_fn=validate_fn,
            ):
                changed += 1

    if changed <= 0:
        return 0

    # Write back
    replace_map = {line_idx: row for line_idx, row in parsed}
    out_lines: List[str] = []
    for i, line in enumerate(raw_lines):
        row = replace_map.get(i)
        if not row:
            out_lines.append(line)
            continue
        o = float(row["offset"])
        c = float(row["cons"])
        ct = float(row["cutoff"])
        p = float(row["pre"])
        ov = float(row["ovl"])
        out_lines.append(f"{row['wav']}={row['alias']},{o:.2f},{c:.2f},{ct:.2f},{p:.2f},{ov:.2f}")

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


__all__ = ["apply_mel_safety_clamp_to_oto_file"]
