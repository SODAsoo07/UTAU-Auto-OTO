from __future__ import annotations

import os
import wave
from dataclasses import dataclass

from core.kr_oto_file_ops import (
    _apply_kr_bridge_coherence_to_oto_file,
    _apply_kr_profile_to_oto_file,
)
from core.kr_oto_rules import (
    KR_PLOSIVE_ONSETS,
    KR_SIBILANT_ONSETS,
    _extract_alias_onset,
    classify_alias,
)
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.mel_refine_profile import resolve_mel_refine_float
from core.kr_oto_file_consistency import apply_file_consistency_to_oto_file
from core.mel_safety_clamp import apply_mel_safety_clamp_to_oto_file
from core.oto_post_regression_guard import (
    apply_oto_post_regression_guard,
    capture_oto_baseline_snapshot,
)
from core.generator_finish import write_oto_lines
from core.post_file_pipeline import (
    log_changed_lines,
    resolve_wav_dir_from_tg_folder,
    run_ml_post_stage,
)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _env_float(name, default):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _kr_onset_group(onset: str) -> str:
    token = str(onset or "").strip().lower()
    if token in KR_SIBILANT_ONSETS:
        return "sibilant"
    if token in KR_PLOSIVE_ONSETS:
        return "plosive"
    return "other"


def _mel_refine_float(
    key: str,
    default: float,
    *,
    alias_type: str = "",
    onset_group: str = "",
    format_type: str = "",
) -> float:
    return resolve_mel_refine_float(
        "korean",
        key,
        default,
        alias_type=alias_type,
        onset_group=onset_group,
        format_type=format_type,
    )


def _normalize_ml_route(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto", "automatic", "policy", "v2", "autofree_v1", "autofree", "no-mfa", "no_mfa", "nomfa", "b", "route_b"}:
        return "autofree_v1"
    return "legacy"


def _current_ml_route() -> str:
    return _normalize_ml_route(os.environ.get("UTOA_ML_ROUTE", "v2"))


def _call_validate(validate_fn, offset, consonant, cutoff, pre, ovl, *, alias_type=""):
    try:
        return validate_fn(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    except TypeError:
        return validate_fn(offset, consonant, cutoff, pre, ovl)


def sanitize_kr_oto_for_wav_duration(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    wav_duration_ms,
    *,
    alias_type="",
    validate_fn,
):
    """Clamp one KR OTO row to the real wav timeline to avoid render-time geometry errors."""
    offset, consonant, cutoff, pre, ovl = _call_validate(
        validate_fn,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
    )
    try:
        dur_ms = float(wav_duration_ms or 0.0)
    except Exception:
        dur_ms = 0.0
    if dur_ms <= 0.0:
        return offset, consonant, cutoff, pre, ovl

    a_type = str(alias_type or "").strip().lower()
    min_room_after_offset = 16.0 if a_type in {"vc", "vv", "vcv"} else 22.0
    offset = max(0.0, min(float(offset), max(dur_ms - min_room_after_offset, 0.0)))

    cutoff_abs = abs(float(cutoff))
    max_cutoff_abs = max(dur_ms - offset - 6.0, 0.0)
    cutoff_abs = min(cutoff_abs, max_cutoff_abs)

    active_len = max(dur_ms - offset - cutoff_abs, 0.0)
    if active_len < 10.0:
        target_active_len = min(max(12.0, active_len), max(dur_ms - offset - 2.0, 0.0))
        cutoff_abs = max(0.0, dur_ms - offset - target_active_len)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)

    max_cons = max(active_len - 6.0, 0.0)
    consonant = min(float(consonant), max_cons)
    if consonant < 8.0 and active_len >= 10.0:
        consonant = min(max(active_len * 0.72, 8.0), max_cons)

    pre = min(max(float(pre), 0.0), max(consonant - 6.0, 0.0))
    ovl_cap = pre * 0.82 if pre > 0.0 else 0.0
    ovl = min(max(float(ovl), 0.0), ovl_cap)

    if consonant <= pre + 4.0:
        if max_cons >= pre + 6.0:
            consonant = pre + 6.0
        else:
            consonant = max_cons
            pre = max(0.0, consonant - 6.0)
            ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    if dur_ms - offset - cutoff_abs <= 2.0:
        cutoff_abs = max(0.0, dur_ms - offset - 4.0)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)
        max_cons = max(active_len - 4.0, 0.0)
        consonant = min(consonant, max_cons)
        pre = min(pre, max(consonant - 4.0, 0.0))
        ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    cutoff = -max(cutoff_abs, 0.0)
    return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)


def _wav_duration_ms(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "rb") as wf:
            return (wf.getnframes() * 1000.0) / float(wf.getframerate())
    except Exception:
        return 0.0


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
    ml_route: str = ""


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

    def _row_anchor(row):
        try:
            return float(row.get("offset", 0.0)) + float(row.get("pre", 0.0))
        except Exception:
            try:
                return float(row.get("offset", 0.0))
            except Exception:
                return 0.0

    mel_cache = {}
    changed = 0
    for wav_name, rows in by_wav.items():
        wav_path = _find_wav_path_for_name(wav_name, wav_dir, wav_index)
        if not wav_path:
            continue
        ordered_rows = sorted(rows, key=lambda item: (_row_anchor(item[1]), item[0]))
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
        high_arr = mel_ctx.get("high_ratio")
        f2_arr = mel_ctx.get("f2_ratio")
        f3_arr = mel_ctx.get("f3_ratio")
        db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
        if db_arr is None or len(db_arr) != len(en):
            db_arr = np.zeros_like(en, dtype=np.float64)
        if f0v_arr is None or len(f0v_arr) != len(en):
            f0v_arr = np.zeros_like(en, dtype=np.float64)
        if high_arr is None or len(high_arr) != len(en):
            high_arr = np.zeros_like(en, dtype=np.float64)
        if f2_arr is None or len(f2_arr) != len(en):
            f2_arr = np.zeros_like(en, dtype=np.float64)
        if f3_arr is None or len(f3_arr) != len(en):
            f3_arr = np.zeros_like(en, dtype=np.float64)
        if len(t_ms) < 8:
            continue

        for i, (_line_idx, row) in enumerate(ordered_rows):
            alias_type = _classify_cached(row["alias"])
            if alias_type not in {"cv", "cv_head"}:
                continue
            onset = str(_extract_alias_onset(row["alias"]) or "").strip().lower()
            is_sibilant = onset in KR_SIBILANT_ONSETS
            is_plosive = onset in KR_PLOSIVE_ONSETS
            onset_group = _kr_onset_group(onset)

            off = float(row["offset"])
            pre_abs = off + float(row["pre"])
            cons_abs = off + float(row["cons"])
            cut_abs = off + abs(float(row["cutoff"]))
            if cut_abs <= pre_abs + 24.0:
                continue

            next_anchor = None
            for j in range(i + 1, len(ordered_rows)):
                next_alias_type = _classify_cached(ordered_rows[j][1]["alias"])
                if next_alias_type in {"cv", "cv_head", "vcv", "mono"}:
                    next_anchor = float(ordered_rows[j][1]["offset"]) + float(ordered_rows[j][1]["pre"])
                    break

            search_start = pre_abs + _mel_refine_float(
                "UTOA_KR_MEL_REFINE_SEARCH_START_FROM_PRE_MS",
                14.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            search_end = cut_abs - _mel_refine_float(
                "UTOA_KR_MEL_REFINE_SEARCH_END_FROM_CUT_MS",
                8.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            if next_anchor is not None:
                search_end = min(
                    search_end,
                    next_anchor
                    - _mel_refine_float(
                        "UTOA_KR_MEL_REFINE_NEXT_ANCHOR_MARGIN_MS",
                        8.0,
                        alias_type=alias_type,
                        onset_group=onset_group,
                    ),
                )
            if search_end <= search_start + _mel_refine_float(
                "UTOA_KR_MEL_REFINE_MIN_SEARCH_SPAN_MS",
                25.0,
                alias_type=alias_type,
                onset_group=onset_group,
            ):
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
                h_v = float(high_arr[ci])
                fm_v = float(f2_arr[ci] + f3_arr[ci])
                silence_bonus = 0.28 if db_v <= db_sil_th else 0.0
                score = (1.0 - e_v) + silence_bonus - (0.08 * f0_v) - (0.10 * h_v) - (0.06 * fm_v)
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

            contrast_min = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_CONTRAST_MIN",
                0.11,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            db_drop_min = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_DB_DROP_MIN",
                2.3,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            if is_sibilant:
                contrast_min += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_SIBILANT_CONTRAST_BONUS",
                    0.03,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )
                db_drop_min += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_SIBILANT_DB_DROP_BONUS",
                    0.4,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )
            elif is_plosive:
                contrast_min += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_PLOSIVE_CONTRAST_BONUS",
                    0.01,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )
                db_drop_min += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_PLOSIVE_DB_DROP_BONUS",
                    0.2,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )

            if contrast < contrast_min and db_drop < db_drop_min:
                continue
            valley_energy_cap = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_VALLEY_ENERGY_MAX",
                0.36,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            valley_db_cap = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_VALLEY_DB_MARGIN",
                6.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            if is_sibilant:
                valley_energy_cap += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_SIBILANT_VALLEY_E_DELTA",
                    0.02,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )
            if valley_e > valley_energy_cap and valley_db > (db_sil_th + valley_db_cap):
                continue
            valley_f0_cap = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_VALLEY_F0_MAX",
                0.70,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            f0_contrast_guard = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_F0_CONTRAST_MIN",
                0.17,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            if valley_f0v > valley_f0_cap and contrast < f0_contrast_guard:
                continue
            tail_guard_ms = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_TAIL_GUARD_MS",
                12.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            if is_sibilant:
                tail_guard_ms += _mel_refine_float(
                    "UTOA_KR_MEL_REFINE_SIBILANT_TAIL_GUARD_ADD_MS",
                    2.0,
                    alias_type=alias_type,
                    onset_group=onset_group,
                )
            if valley_t >= cut_abs - tail_guard_ms:
                continue

            cut_shift_ms = _mel_refine_float(
                "UTOA_KR_MEL_REFINE_TARGET_SHIFT_MS",
                2.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
            target_cut_abs = valley_t + cut_shift_ms
            min_cut_abs = pre_abs + _mel_refine_float(
                "UTOA_KR_MEL_REFINE_MIN_CUT_FROM_PRE_MS",
                20.0,
                alias_type=alias_type,
                onset_group=onset_group,
            )
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

    write_oto_lines(oto_path, out_lines)
    return changed


def apply_kr_wav_duration_safety_to_oto_file(
    oto_path: str,
    wav_dir: str,
    *,
    custom_map=None,
    validate_fn,
    normalize_key_fn,
) -> int:
    if not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    lines = read_text_with_fallback(oto_path).splitlines()
    if not lines:
        return 0

    wav_index = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                key = normalize_key_fn(fn) if callable(normalize_key_fn) else fn.lower()
                wav_index[str(key)] = os.path.join(wav_dir, fn)
    except Exception:
        return 0

    dur_cache = {}
    out_lines = []
    changed = 0
    alias_type_cache = {}

    def _classify_cached(alias_text: str) -> str:
        key = str(alias_text or "")
        out = alias_type_cache.get(key)
        if out is None:
            out = classify_alias(key, custom_map)
            alias_type_cache[key] = out
        return out

    def _resolve_wav(row_wav: str) -> str:
        candidates = [str(row_wav or "")]
        base_name = os.path.basename(str(row_wav or ""))
        if base_name and base_name not in candidates:
            candidates.append(base_name)
        for cand in candidates:
            if not cand:
                continue
            key = normalize_key_fn(cand) if callable(normalize_key_fn) else cand.lower()
            path = wav_index.get(str(key))
            if path:
                return path
        return ""

    for line in lines:
        row = parse_oto_line(line)
        if not row:
            out_lines.append(line)
            continue

        wav_path = _resolve_wav(str(row.get("wav", "")))
        if not wav_path:
            out_lines.append(line)
            continue

        dur_ms = dur_cache.get(wav_path)
        if dur_ms is None:
            dur_ms = _wav_duration_ms(wav_path)
            dur_cache[wav_path] = dur_ms
        if dur_ms <= 0.0:
            out_lines.append(line)
            continue

        alias_type = _classify_cached(str(row.get("alias", "")))
        o2, c2, ct2, p2, ov2 = sanitize_kr_oto_for_wav_duration(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            dur_ms,
            alias_type=alias_type,
            validate_fn=validate_fn,
        )
        if (
            abs(o2 - float(row["offset"])) > 1e-6
            or abs(c2 - float(row["cons"])) > 1e-6
            or abs(ct2 - float(row["cutoff"])) > 1e-6
            or abs(p2 - float(row["pre"])) > 1e-6
            or abs(ov2 - float(row["ovl"])) > 1e-6
        ):
            changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{p2:.2f},{ov2:.2f}"
            )
        else:
            out_lines.append(line)

    if changed <= 0:
        return 0

    write_oto_lines(oto_path, [line.rstrip("\n") for line in out_lines])
    return changed


def run_kr_post_file_pipeline(context: KrPostFilePipelineContext):
    wav_dir = resolve_wav_dir_from_tg_folder(context.tg_folder)
    ml_route = _normalize_ml_route(getattr(context, "ml_route", "") or _current_ml_route())
    format_type = str(getattr(context, "auto_gen_format", "") or "").strip().lower()
    guard_snapshot = None
    if format_type in {"cvc", "cvvc"}:
        guard_snapshot = capture_oto_baseline_snapshot(context.out_path)
    if callable(context.log_fn):
        context.log_fn(f"[OTO-ML] KR finalize route={ml_route}")

    if context.kr_profile:
        changed = _apply_kr_profile_to_oto_file(
            context.out_path,
            wav_dir,
            context.kr_profile,
            custom_map=context.custom_map,
        )
        log_changed_lines(context.log_fn, "[KR-Profile]", changed, "reference profile changed")

    safety_changed = apply_mel_safety_clamp_to_oto_file(
        context.out_path,
        wav_dir,
        classify_alias_fn=lambda alias_text: classify_alias(alias_text, context.custom_map),
        validate_fn=context.validate_fn,
        normalize_key_fn=context.normalize_key_fn,
        language="korean",
    )
    log_changed_lines(context.log_fn, "[KR-Mel-Safety]", safety_changed, "mel safety clamp changed")

    def _run_kr_legacy_post_filters() -> None:
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
            mapping_context=context.runtime_report,
        )
        log_changed_lines(
            context.log_fn, "[KR-Consistency]",
            consistency_stats.get("total_changed", 0),
            "file consistency changed",
        )

    def _run_ml_stage() -> None:
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
            ml_route=ml_route,
        )

    # v2/No-MFA route keeps the autofree residual shaping at the final stage.
    # v1 route retains the previous order for compatibility.
    if ml_route == "autofree_v1":
        if callable(context.log_fn):
            context.log_fn("[OTO-ML] KR finalize order: v2/No-MFA post-filters -> autofree ML")
        _run_kr_legacy_post_filters()
        _run_ml_stage()
    else:
        if callable(context.log_fn):
            context.log_fn("[OTO-ML] KR finalize order: v1 ML -> v1 post-filters")
        _run_ml_stage()
        _run_kr_legacy_post_filters()

    safety_changed = apply_kr_wav_duration_safety_to_oto_file(
        context.out_path,
        wav_dir,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        normalize_key_fn=context.normalize_key_fn,
    )
    log_changed_lines(
        context.log_fn,
        "[KR-Safety]",
        safety_changed,
        "wav-duration safety changed",
    )

    if guard_snapshot is not None:
        guard_stats = apply_oto_post_regression_guard(
            context.out_path,
            baseline_snapshot=guard_snapshot,
            alias_type_resolver=lambda alias_text: classify_alias(alias_text, context.custom_map),
            log_fn=context.log_fn,
            log_tag="[KR-RegressionGuard]",
        )
        guard_changed = int(guard_stats.get("reverted", 0) or 0)
        if guard_changed > 0 and callable(context.log_fn):
            reasons = guard_stats.get("reasons") or {}
            reason_text = ", ".join(
                f"{name}:{int(count)}"
                for name, count in sorted(reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            )
            checked = int(guard_stats.get("checked", 0) or 0)
            if reason_text:
                context.log_fn(
                    f"[KR-RegressionGuard] post-pass rollback: {guard_changed}/{checked} rows ({reason_text})"
                )
            else:
                context.log_fn(
                    f"[KR-RegressionGuard] post-pass rollback: {guard_changed}/{checked} rows"
                )


__all__ = [
    "KrPostFilePipelineContext",
    "apply_kr_mel_refine_to_oto_file",
    "apply_kr_wav_duration_safety_to_oto_file",
    "run_kr_post_file_pipeline",
    "sanitize_kr_oto_for_wav_duration",
]
