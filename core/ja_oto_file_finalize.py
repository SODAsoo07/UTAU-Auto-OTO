from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from core.generation.file_index import build_wav_index as build_generation_wav_index
from core.ja_oto_finalize import _convert_ja_internal_cutoff_to_oto_field
from core.ja_oto_file_consistency import apply_ja_vc_neighbor_to_oto_file
from core.oto_continuity_clamp import apply_continuity_clamp_to_oto_file
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_normalization import normalize_wav_key
from core.mel_safety_clamp import apply_mel_safety_clamp_to_oto_file
from core.post_file_pipeline import (
    log_changed_lines,
    resolve_wav_dir_from_tg_folder,
    run_ml_post_stage,
)


def _env_float(name, default):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _normalize_ml_route(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"e2e_hybrid", "e2e", "hybrid"}:
        return "e2e_hybrid"
    return "legacy"


def _current_ml_route() -> str:
    return _normalize_ml_route(os.environ.get("UTOA_ML_ROUTE", "legacy"))


def _format_oto_row_line(row: Dict[str, object]) -> str:
    return (
        f"{row['wav']}={row['alias']},"
        f"{float(row['offset']):.2f},{float(row['cons']):.2f},{float(row['cutoff']):.2f},"
        f"{float(row['pre']):.2f},{float(row['ovl']):.2f}"
    )


@dataclass(frozen=True)
class _OtoSnapshot:
    lines: List[str]
    rows_by_index: Dict[int, Dict[str, object]]
    alias_type_by_index: Dict[int, str]
    total_rows: int


def _take_oto_snapshot(oto_path: str, *, custom_map=None, classify_alias_fn=None) -> _OtoSnapshot:
    text = read_text_with_fallback(oto_path) if oto_path and os.path.exists(oto_path) else ""
    lines = text.splitlines()
    rows_by_index: Dict[int, Dict[str, object]] = {}
    alias_type_by_index: Dict[int, str] = {}
    for idx, line in enumerate(lines):
        row = parse_oto_line(line)
        if not row:
            continue
        rows_by_index[idx] = dict(row)
        if callable(classify_alias_fn):
            alias_type_by_index[idx] = str(
                classify_alias_fn(str(row.get("alias", "")), custom_map) or ""
            ).strip().lower()
    return _OtoSnapshot(
        lines=list(lines),
        rows_by_index=rows_by_index,
        alias_type_by_index=alias_type_by_index,
        total_rows=len(rows_by_index),
    )


def _rows_different(a: Optional[Dict[str, object]], b: Optional[Dict[str, object]]) -> bool:
    if not a or not b:
        return False
    keys = ("offset", "cons", "cutoff", "pre", "ovl")
    for key in keys:
        if abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) > 1e-6:
            return True
    return False


def _resolve_gate_scope(stage_name: str) -> str:
    stage = str(stage_name or "").strip().lower()
    stage_key = "UTOA_JA_QG_SCOPE_ML" if stage == "ml" else "UTOA_JA_QG_SCOPE_POST"
    raw = str(os.environ.get(stage_key, os.environ.get("UTOA_JA_QG_SCOPE", "cv_family"))).strip().lower()
    if raw in {"all", "cv_family", "vc_family", "none"}:
        return raw
    return "cv_family"


def _gate_scope_aliases(scope: str) -> Optional[set]:
    name = str(scope or "").strip().lower()
    if name == "none":
        return set()
    if name == "all":
        return None
    if name == "vc_family":
        return {"vc", "vv"}
    return {"cv", "cv_head", "vcv", "mono"}


def _extract_quality_gate_risk(runtime_report: object) -> float:
    if not isinstance(runtime_report, dict):
        return 0.0
    risk = 0.0
    ml = runtime_report.get("ml")
    if isinstance(ml, dict):
        rel = ml.get("reliability_metrics")
        if isinstance(rel, dict):
            rows = int(rel.get("rows", 0) or 0)
            if rows > 0:
                blank_ratio = float(rel.get("blank_flag_rows", 0) or 0.0) / float(rows)
                mel_unreliable_ratio = float(rel.get("mel_unreliable_rows", 0) or 0.0) / float(rows)
                abstain_ratio = float(rel.get("abstain_rows", 0) or 0.0) / float(rows)
                risk = max(risk, blank_ratio, mel_unreliable_ratio, abstain_ratio)
    mapping = runtime_report.get("mapping")
    if isinstance(mapping, dict):
        if bool(mapping.get("file_low_conf", False)):
            risk = max(risk, 0.50)
        try:
            blank_mean = float(mapping.get("blank_confidence_mean", 0.0) or 0.0)
            if blank_mean >= 0.45:
                risk = max(risk, min(1.0, blank_mean))
        except Exception:
            pass
    return float(_clamp(risk, 0.0, 1.0))


def _resolve_stage_gate_profile(stage_name: str, high_risk_ratio: float) -> Dict[str, float]:
    stage = str(stage_name or "").strip().lower()
    if stage == "ml":
        soft = _env_float("UTOA_JA_QG_ML_SOFT_RATIO", 0.40)
        hard = _env_float("UTOA_JA_QG_ML_HARD_RATIO", 0.60)
        soft_scale = _env_float("UTOA_JA_QG_ML_SOFT_SCALE", 0.72)
    else:
        soft = _env_float("UTOA_JA_QG_POST_SOFT_RATIO", 0.32)
        hard = _env_float("UTOA_JA_QG_POST_HARD_RATIO", 0.52)
        soft_scale = _env_float("UTOA_JA_QG_POST_SOFT_SCALE", 0.70)

    risk_soft = _env_float("UTOA_JA_QG_HIGH_RISK_SOFT", 0.30)
    risk_hard = _env_float("UTOA_JA_QG_HIGH_RISK_HARD", 0.50)
    if float(high_risk_ratio) >= float(risk_soft):
        soft = max(0.05, soft - 0.10)
        hard = max(soft + 0.05, hard - 0.08)
    if float(high_risk_ratio) >= float(risk_hard):
        soft = max(0.05, soft - 0.08)
        hard = max(soft + 0.05, hard - 0.08)
    return {
        "soft_ratio": float(_clamp(soft, 0.01, 1.0)),
        "hard_ratio": float(_clamp(hard, 0.05, 1.0)),
        "soft_scale": float(_clamp(soft_scale, 0.05, 0.98)),
    }


def _apply_stage_quality_gate(
    oto_path: str,
    *,
    stage_name: str,
    before_snapshot: Optional[_OtoSnapshot],
    custom_map,
    validate_fn,
    classify_alias_fn,
    log_fn=None,
    runtime_report: object = None,
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "stage": str(stage_name or ""),
        "enabled": False,
        "scope": "none",
        "stage_changed_lines": 0,
        "stage_changed_ratio": 0.0,
        "risk_ratio": 0.0,
        "soft_ratio": 0.0,
        "hard_ratio": 0.0,
        "action": "off",
        "gate_adjusted_lines": 0,
    }
    if not _env_bool("UTOA_JA_QG_ENABLE", True):
        return stats
    if not before_snapshot or not oto_path or not os.path.exists(oto_path):
        return stats

    after_snapshot = _take_oto_snapshot(
        oto_path,
        custom_map=custom_map,
        classify_alias_fn=classify_alias_fn,
    )
    total_rows = int(max(before_snapshot.total_rows, after_snapshot.total_rows, 1))
    changed_indices = [
        idx
        for idx, after_row in after_snapshot.rows_by_index.items()
        if _rows_different(before_snapshot.rows_by_index.get(idx), after_row)
    ]
    changed_count = int(len(changed_indices))
    changed_ratio = float(changed_count) / float(total_rows)
    risk_ratio = _extract_quality_gate_risk(runtime_report)
    profile = _resolve_stage_gate_profile(stage_name, risk_ratio)
    scope = _resolve_gate_scope(stage_name)
    scope_aliases = _gate_scope_aliases(scope)
    target_indices: List[int] = []
    if scope_aliases == set():
        target_indices = []
    elif scope_aliases is None:
        target_indices = list(changed_indices)
    else:
        for idx in changed_indices:
            alias_type = str(
                after_snapshot.alias_type_by_index.get(idx)
                or before_snapshot.alias_type_by_index.get(idx)
                or ""
            ).strip().lower()
            if alias_type in scope_aliases:
                target_indices.append(idx)
    target_count = int(len(target_indices))

    stats.update(
        {
            "enabled": True,
            "scope": str(scope),
            "stage_changed_lines": changed_count,
            "stage_changed_ratio": float(changed_ratio),
            "risk_ratio": float(risk_ratio),
            "soft_ratio": float(profile["soft_ratio"]),
            "hard_ratio": float(profile["hard_ratio"]),
        }
    )
    if target_count <= 0:
        stats["action"] = "off"
        return stats

    if changed_ratio < float(profile["soft_ratio"]):
        stats["action"] = "off"
        return stats
    action = "hard_rollback" if changed_ratio >= float(profile["hard_ratio"]) else "soft_scale"
    stats["action"] = action

    out_lines = list(after_snapshot.lines)
    gate_adjusted = 0
    if action == "hard_rollback":
        for idx in target_indices:
            base_row = before_snapshot.rows_by_index.get(idx)
            if not base_row:
                continue
            out_lines[idx] = _format_oto_row_line(base_row)
            gate_adjusted += 1
    else:
        scale = float(profile["soft_scale"])
        for idx in target_indices:
            base_row = before_snapshot.rows_by_index.get(idx)
            curr_row = after_snapshot.rows_by_index.get(idx)
            if not base_row or not curr_row:
                continue
            alias_type = str(
                after_snapshot.alias_type_by_index.get(idx)
                or before_snapshot.alias_type_by_index.get(idx)
                or ""
            ).strip().lower()
            try:
                o = float(base_row["offset"]) + ((float(curr_row["offset"]) - float(base_row["offset"])) * scale)
                c = float(base_row["cons"]) + ((float(curr_row["cons"]) - float(base_row["cons"])) * scale)
                ct = float(base_row["cutoff"]) + ((float(curr_row["cutoff"]) - float(base_row["cutoff"])) * scale)
                p = float(base_row["pre"]) + ((float(curr_row["pre"]) - float(base_row["pre"])) * scale)
                ov = float(base_row["ovl"]) + ((float(curr_row["ovl"]) - float(base_row["ovl"])) * scale)
                try:
                    o, c, ct, p, ov = validate_fn(o, c, ct, p, ov, alias_type=alias_type)
                except TypeError:
                    o, c, ct, p, ov = validate_fn(o, c, ct, p, ov)
                new_row = dict(curr_row)
                new_row["offset"] = float(o)
                new_row["cons"] = float(c)
                new_row["cutoff"] = float(ct)
                new_row["pre"] = float(p)
                new_row["ovl"] = float(ov)
                out_lines[idx] = _format_oto_row_line(new_row)
                gate_adjusted += 1
            except Exception:
                continue

    if gate_adjusted > 0:
        with open(oto_path, "w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(str(line).rstrip("\n") + "\n")
    stats["gate_adjusted_lines"] = int(gate_adjusted)

    if callable(log_fn):
        log_fn(
            "[JA-QualityGate] "
            f"stage={stage_name}, action={action}, scope={scope}, "
            f"changed={changed_count}/{total_rows} ({(100.0 * changed_ratio):.1f}%), "
            f"risk={risk_ratio:.2f}, soft={profile['soft_ratio']:.2f}, hard={profile['hard_ratio']:.2f}, "
            f"adjusted={gate_adjusted}"
        )
    if isinstance(runtime_report, dict):
        qg = runtime_report.setdefault("quality_gate", {})
        if isinstance(qg, dict):
            qg[str(stage_name)] = dict(stats)
    return stats


@dataclass(frozen=True)
class JaPostFilePipelineContext:
    out_path: str
    tg_folder: str
    custom_map: object
    custom_phonemes_path: str
    enable_ml_correction: bool
    forced_format: str
    fallback_format: str
    ml_policy: object
    runtime_report: object
    log_fn: object
    validate_fn: object
    classify_alias_fn: object
    ml_route: str = ""


def apply_ja_mel_refine_to_oto_file(
    oto_path,
    wav_dir,
    *,
    custom_map=None,
    validate_fn: Callable[[float, float, float, float, float], tuple[float, float, float, float, float]],
    classify_alias_fn: Callable[[str, object], str],
):
    """Refine JA CV-like cutoff timing with a mel-energy valley search."""
    if os.environ.get("UTOA_DISABLE_MEL_REFINER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0
    try:
        import numpy as np
    except Exception:
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

    wav_index = build_generation_wav_index(
        wav_dir,
        normalize_key_fn=normalize_wav_key,
        recursive=False,
    )

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))
    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_type = alias_type_cache.get(alias_text)
        if alias_type is None:
            alias_type = classify_alias_fn(alias_text, custom_map)
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
            if alias_type not in {"cv", "cv_head", "vcv"}:
                continue

            off = float(row["offset"])
            pre_abs = off + float(row["pre"])
            cons_abs = off + float(row["cons"])
            cut_abs = off + abs(float(row["cutoff"]))
            if cut_abs <= pre_abs + 24.0:
                continue

            next_anchor = None
            for j in range(i + 1, len(rows)):
                alias_type_2 = _classify_cached(rows[j][1]["alias"])
                if alias_type_2 in {"cv", "cv_head", "vcv", "mono"}:
                    next_anchor = float(rows[j][1]["offset"]) + float(rows[j][1]["pre"])
                    break

            search_start = pre_abs + _env_float("UTOA_JA_MEL_REFINE_SEARCH_START_FROM_PRE_MS", 14.0)
            search_end = cut_abs - _env_float("UTOA_JA_MEL_REFINE_SEARCH_END_FROM_CUT_MS", 8.0)
            if next_anchor is not None:
                search_end = min(search_end, next_anchor - _env_float("UTOA_JA_MEL_REFINE_NEXT_ANCHOR_MARGIN_MS", 8.0))
            if search_end <= search_start + _env_float("UTOA_JA_MEL_REFINE_MIN_SEARCH_SPAN_MS", 25.0):
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

            contrast_min = _env_float("UTOA_JA_MEL_REFINE_CONTRAST_MIN", 0.11)
            db_drop_min = _env_float("UTOA_JA_MEL_REFINE_DB_DROP_MIN", 2.2)
            if contrast < contrast_min and db_drop < db_drop_min:
                continue
            valley_energy_cap = _env_float("UTOA_JA_MEL_REFINE_VALLEY_ENERGY_MAX", 0.40)
            valley_db_cap = _env_float("UTOA_JA_MEL_REFINE_VALLEY_DB_MARGIN", 6.0)
            if valley_e > valley_energy_cap and valley_db > (db_sil_th + valley_db_cap):
                continue
            valley_f0_cap = _env_float("UTOA_JA_MEL_REFINE_VALLEY_F0_MAX", 0.72)
            f0_contrast_guard = _env_float("UTOA_JA_MEL_REFINE_F0_CONTRAST_MIN", 0.16)
            if valley_f0v > valley_f0_cap and contrast < f0_contrast_guard:
                continue
            tail_guard_ms = _env_float("UTOA_JA_MEL_REFINE_TAIL_GUARD_MS", 12.0)
            if valley_t >= cut_abs - tail_guard_ms:
                continue

            target_cut_abs = valley_t + _env_float("UTOA_JA_MEL_REFINE_TARGET_SHIFT_MS", 2.0)
            min_cut_abs = pre_abs + _env_float("UTOA_JA_MEL_REFINE_MIN_CUT_FROM_PRE_MS", 20.0)
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


def run_ja_post_file_pipeline(context: JaPostFilePipelineContext):
    ml_route = _normalize_ml_route(getattr(context, "ml_route", "") or _current_ml_route())
    if callable(context.log_fn):
        context.log_fn(f"[OTO-ML] JA finalize route={ml_route}")
    wav_dir_for_mel = resolve_wav_dir_from_tg_folder(context.tg_folder)
    classify_alias = lambda alias_text, custom_map=None: context.classify_alias_fn(alias_text, custom_map)

    safety_changed = apply_mel_safety_clamp_to_oto_file(
        context.out_path,
        wav_dir_for_mel,
        classify_alias_fn=lambda alias_text: classify_alias(alias_text, context.custom_map),
        validate_fn=context.validate_fn,
        normalize_key_fn=normalize_wav_key,
        language="japanese",
    )
    log_changed_lines(context.log_fn, "[JA-Mel-Safety]", safety_changed, "mel safety clamp changed")

    ml_before_snapshot = _take_oto_snapshot(
        context.out_path,
        custom_map=context.custom_map,
        classify_alias_fn=classify_alias,
    )
    run_ml_post_stage(
        language="japanese",
        out_path=context.out_path,
        tg_folder=context.tg_folder,
        custom_phonemes_path=context.custom_phonemes_path,
        enable_ml_correction=context.enable_ml_correction,
        format_override=(context.forced_format or context.fallback_format or "cvvc"),
        ml_policy=context.ml_policy,
        runtime_report=context.runtime_report,
        log_fn=context.log_fn,
        ml_route=ml_route,
    )
    _apply_stage_quality_gate(
        context.out_path,
        stage_name="ml",
        before_snapshot=ml_before_snapshot,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        classify_alias_fn=classify_alias,
        log_fn=context.log_fn,
        runtime_report=context.runtime_report,
    )

    post_before_snapshot = _take_oto_snapshot(
        context.out_path,
        custom_map=context.custom_map,
        classify_alias_fn=classify_alias,
    )
    try:
        mel_changed = apply_ja_mel_refine_to_oto_file(
            context.out_path,
            wav_dir_for_mel,
            custom_map=context.custom_map,
            validate_fn=context.validate_fn,
            classify_alias_fn=context.classify_alias_fn,
        )
        log_changed_lines(context.log_fn, "[JA-Mel]", mel_changed, "mel cutoff refine changed")
    except Exception as exc:
        context.log_fn(f"[JA-Mel] mel cutoff refine failed: {exc}")

    try:
        vc_neighbor_stats = apply_ja_vc_neighbor_to_oto_file(
            context.out_path,
            custom_map=context.custom_map,
            validate_fn=context.validate_fn,
            log_fn=context.log_fn,
        )
        log_changed_lines(
            context.log_fn,
            "[JA-Consistency]",
            vc_neighbor_stats.get("total_changed", 0),
            "file consistency changed",
        )
    except Exception as exc:
        context.log_fn(f"[JA-Consistency] vc neighbor adjust failed: {exc}")

    _apply_stage_quality_gate(
        context.out_path,
        stage_name="post",
        before_snapshot=post_before_snapshot,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        classify_alias_fn=classify_alias,
        log_fn=context.log_fn,
        runtime_report=context.runtime_report,
    )

    try:
        final_changed = _convert_ja_internal_cutoff_to_oto_field(context.out_path, wav_dir_for_mel)
        log_changed_lines(context.log_fn, "[JA-Finalize]", final_changed, "cutoff finalize changed")
    except Exception as exc:
        context.log_fn(f"[JA-Finalize] cutoff finalize failed: {exc}")

    try:
        cont_stats = apply_continuity_clamp_to_oto_file(
            context.out_path,
            classify_alias_fn=classify_alias,
            validate_fn=context.validate_fn,
            custom_map=context.custom_map,
            env_prefix="UTOA_JA",
            language="japanese",
            format_type=(context.forced_format or context.fallback_format or "cvvc"),
            log_fn=context.log_fn,
            log_tag="[JA-Continuity]",
        )
        log_changed_lines(
            context.log_fn,
            "[JA-Continuity]",
            int(cont_stats.get("rows_adjusted", 0)),
            "continuity clamp changed",
        )
    except Exception as exc:
        context.log_fn(f"[JA-Continuity] clamp failed: {exc}")


__all__ = [
    "JaPostFilePipelineContext",
    "apply_ja_mel_refine_to_oto_file",
    "run_ja_post_file_pipeline",
]
