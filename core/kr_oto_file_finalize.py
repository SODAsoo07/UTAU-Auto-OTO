from __future__ import annotations

import os
from dataclasses import dataclass

from core.kr_oto_file_ops import (
    _apply_kr_bridge_coherence_to_oto_file,
    _apply_kr_profile_to_oto_file,
)
from core.kr_oto_rules import classify_alias
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.kr_oto_file_consistency import apply_file_consistency_to_oto_file
from core.post_file_pipeline import (
    log_changed_lines,
    resolve_wav_dir_from_tg_folder,
    run_ml_post_stage,
)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass(frozen=True)
class KrPostFilePipelineContext:
    out_path: str
    tg_folder: str
    kr_profile: object
    custom_map: object
    custom_phonemes_path: str
    enable_ml_correction: bool
    auto_gen_format: str
    ml_policy: object
    runtime_report: object
    log_fn: object
    validate_fn: object
    normalize_key_fn: object


def apply_kr_mel_refine_to_oto_file(
    oto_path,
    wav_dir,
    *,
    custom_map=None,
    validate_fn,
    normalize_key_fn,
):
    """Refine KR CV-like cutoff timing with a mel-energy valley search."""
    if os.environ.get("UTOA_DISABLE_MEL_REFINER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if np is None or not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    from core.oto_generator import _find_wav_path_for_name, _mel_envelope, _read_wav_mono_np

    raw_lines = []
    parsed = []
    for idx, raw in enumerate(read_text_with_fallback(oto_path).splitlines()):
        line = raw.rstrip("\n")
        raw_lines.append(line)
        row = parse_oto_line(line)
        if row:
            parsed.append((idx, row))
    if not parsed:
        return 0

    wav_index = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                wav_index[normalize_key_fn(fn)] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))
    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_type = alias_type_cache.get(alias_text)
        if alias_type is None:
            alias_type = classify_alias(alias_text, custom_map)
            alias_type_cache[alias_text] = alias_type
        return alias_type

    mel_cache = {}
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

        t_ms = mel_ctx["times_ms"]
        en = mel_ctx["energy"]
        db_arr = mel_ctx.get("db_db")
        f0v_arr = mel_ctx.get("f0_voicing")
        db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
        if db_arr is None or len(db_arr) != len(en):
            db_arr = np.zeros_like(en, dtype=np.float64)
        if f0v_arr is None or len(f0v_arr) != len(en):
            f0v_arr = np.zeros_like(en, dtype=np.float64)
        if len(t_ms) < 8:
            continue

        for i, (_line_idx, row) in enumerate(rows):
            alias_type = _classify_cached(row["alias"])
            if alias_type not in {"cv", "cv_head"}:
                continue

            off = float(row["offset"])
            pre_abs = off + float(row["pre"])
            cons_abs = off + float(row["cons"])
            cut_abs = off + abs(float(row["cutoff"]))
            if cut_abs <= pre_abs + 24.0:
                continue

            next_anchor = None
            for j in range(i + 1, len(rows)):
                next_alias_type = _classify_cached(rows[j][1]["alias"])
                if next_alias_type in {"cv", "cv_head", "vcv", "mono"}:
                    next_anchor = float(rows[j][1]["offset"]) + float(rows[j][1]["pre"])
                    break

            search_start = pre_abs + 14.0
            search_end = cut_abs - 8.0
            if next_anchor is not None:
                search_end = min(search_end, next_anchor - 8.0)
            if search_end <= search_start + 25.0:
                continue

            mask = np.where((t_ms >= search_start) & (t_ms <= search_end))[0]
            if len(mask) < 5:
                continue

            local_db = db_arr[mask]
            silence_flags = local_db <= db_sil_th
            candidate_mask = mask[silence_flags] if np.any(silence_flags) else mask

            best_idx = int(candidate_mask[0])
            best_score = -1e9
            for ci in candidate_mask:
                e_v = float(en[ci])
                db_v = float(db_arr[ci])
                f0_v = float(f0v_arr[ci])
                silence_bonus = 0.28 if db_v <= db_sil_th else 0.0
                score = (1.0 - e_v) + silence_bonus - (0.08 * f0_v)
                if score > best_score:
                    best_score = score
                    best_idx = int(ci)

            valley_t = float(t_ms[best_idx])
            valley_e = float(en[best_idx])
            valley_db = float(db_arr[best_idx])
            valley_f0v = float(f0v_arr[best_idx])
            cut_idx = int(np.argmin(np.abs(t_ms - cut_abs)))
            cut_e = float(en[cut_idx])
            cut_db = float(db_arr[cut_idx])
            contrast = cut_e - valley_e
            db_drop = cut_db - valley_db

            if contrast < 0.12 and db_drop < 2.5:
                continue
            if valley_e > 0.38 and valley_db > (db_sil_th + 6.0):
                continue
            if valley_f0v > 0.70 and contrast < 0.18:
                continue
            if valley_t >= cut_abs - 12.0:
                continue

            target_cut_abs = valley_t + 2.0
            min_cut_abs = pre_abs + 20.0
            if target_cut_abs <= min_cut_abs:
                continue

            new_cons_abs = min(cons_abs, target_cut_abs - 12.0)
            new_cons_abs = max(new_cons_abs, pre_abs + 8.0)
            row["cons"] = max(new_cons_abs - off, 0.0)
            row["cutoff"] = -(target_cut_abs - off)
            changed += 1

    if changed <= 0:
        return 0

    replace_map = {line_idx: row for line_idx, row in parsed}
    out_lines = []
    for i, line in enumerate(raw_lines):
        row = replace_map.get(i)
        if not row:
            out_lines.append(line)
            continue
        o, c, ct, p, ov = validate_fn(
            row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"]
        )
        out_lines.append(f"{row['wav']}={row['alias']},{o:.2f},{c:.2f},{ct:.2f},{p:.2f},{ov:.2f}")

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def run_kr_post_file_pipeline(context: KrPostFilePipelineContext):
    wav_dir = resolve_wav_dir_from_tg_folder(context.tg_folder)

    if context.kr_profile:
        changed = _apply_kr_profile_to_oto_file(
            context.out_path,
            wav_dir,
            context.kr_profile,
            custom_map=context.custom_map,
        )
        log_changed_lines(context.log_fn, "[KR-Profile]", changed, "reference profile changed")

    run_ml_post_stage(
        language="korean",
        out_path=context.out_path,
        tg_folder=context.tg_folder,
        custom_phonemes_path=context.custom_phonemes_path,
        enable_ml_correction=context.enable_ml_correction,
        format_override=context.auto_gen_format,
        ml_policy=context.ml_policy,
        runtime_report=context.runtime_report,
        log_fn=context.log_fn,
    )

    mel_changed = apply_kr_mel_refine_to_oto_file(
        context.out_path,
        wav_dir,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        normalize_key_fn=context.normalize_key_fn,
    )
    log_changed_lines(context.log_fn, "[KR-Mel]", mel_changed, "mel cutoff refine changed")

    bridge_changed = _apply_kr_bridge_coherence_to_oto_file(
        context.out_path,
        custom_map=context.custom_map,
    )
    log_changed_lines(context.log_fn, "[KR-Bridge]", bridge_changed, "VC/CV coherence changed")

    # 파일 단위 일관성 후처리 (인접 연속성, 급변 스무딩, 순서 강제)
    consistency_stats = apply_file_consistency_to_oto_file(
        context.out_path,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        log_fn=context.log_fn,
    )
    log_changed_lines(
        context.log_fn, "[KR-Consistency]",
        consistency_stats.get("total_changed", 0),
        "file consistency changed",
    )


__all__ = [
    "KrPostFilePipelineContext",
    "apply_kr_mel_refine_to_oto_file",
    "run_kr_post_file_pipeline",
]
