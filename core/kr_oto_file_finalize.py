from __future__ import annotations

import math
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from core.kr_oto_file_consistency import apply_file_consistency_to_oto_file
from core.oto_continuity_clamp import apply_continuity_clamp_to_oto_file
from core.mel_safety_clamp import apply_mel_safety_clamp_to_oto_file
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


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return float(arr[0])
    qq = _clamp(float(q), 0.0, 1.0)
    pos = (len(arr) - 1) * qq
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    w = float(pos - lo)
    return float((1.0 - w) * float(arr[lo]) + w * float(arr[hi]))


def _soft_clip(value: float, lo: float, hi: float, blend: float) -> float:
    v = float(value)
    l = float(lo)
    h = float(hi)
    if l > h:
        l, h = h, l
    b = _clamp(float(blend), 0.0, 1.0)
    if v < l:
        return float(v + ((l - v) * b))
    if v > h:
        return float(v - ((v - h) * b))
    return float(v)


def _format_oto_row_line(row: Dict[str, object]) -> str:
    return (
        f"{row['wav']}={row['alias']},"
        f"{float(row['offset']):.2f},{float(row['cons']):.2f},{float(row['cutoff']):.2f},"
        f"{float(row['pre']):.2f},{float(row['ovl']):.2f}"
    )


@dataclass
class _OtoSnapshot:
    lines: List[str]
    rows_by_index: Dict[int, Dict[str, object]]
    alias_type_by_index: Dict[int, str]
    total_rows: int


def _take_oto_snapshot(oto_path: str, *, custom_map=None) -> _OtoSnapshot:
    text = read_text_with_fallback(oto_path) if oto_path and os.path.exists(oto_path) else ""
    lines = text.splitlines()
    rows_by_index: Dict[int, Dict[str, object]] = {}
    alias_type_by_index: Dict[int, str] = {}
    for idx, line in enumerate(lines):
        row = parse_oto_line(line)
        if not row:
            continue
        rows_by_index[idx] = dict(row)
        alias_type_by_index[idx] = str(classify_alias(str(row.get("alias", "")), custom_map) or "").strip().lower()
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
    for k in keys:
        if abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) > 1e-6:
            return True
    return False


def _resolve_gate_scope(stage_name: str) -> str:
    stage = str(stage_name or "").strip().lower()
    stage_key = "UTOA_KR_QG_SCOPE_ML" if stage == "ml" else "UTOA_KR_QG_SCOPE_POST"
    raw = str(os.environ.get(stage_key, os.environ.get("UTOA_KR_QG_SCOPE", "cv_family"))).strip().lower()
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
        soft = _env_float("UTOA_KR_QG_ML_SOFT_RATIO", 0.45)
        hard = _env_float("UTOA_KR_QG_ML_HARD_RATIO", 0.65)
        soft_scale = _env_float("UTOA_KR_QG_ML_SOFT_SCALE", 0.70)
    else:
        soft = _env_float("UTOA_KR_QG_POST_SOFT_RATIO", 0.35)
        hard = _env_float("UTOA_KR_QG_POST_HARD_RATIO", 0.55)
        soft_scale = _env_float("UTOA_KR_QG_POST_SOFT_SCALE", 0.68)

    risk_soft = _env_float("UTOA_KR_QG_HIGH_RISK_SOFT", 0.30)
    risk_hard = _env_float("UTOA_KR_QG_HIGH_RISK_HARD", 0.50)
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
    if not _env_bool("UTOA_KR_QG_ENABLE", True):
        return stats
    if not before_snapshot or not oto_path or not os.path.exists(oto_path):
        return stats

    after_snapshot = _take_oto_snapshot(oto_path, custom_map=custom_map)
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
    target_indices = []
    if scope_aliases == set():
        target_indices = []
    elif scope_aliases is None:
        target_indices = list(changed_indices)
    else:
        for idx in changed_indices:
            alias_type = str(after_snapshot.alias_type_by_index.get(idx) or before_snapshot.alias_type_by_index.get(idx) or "").strip().lower()
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
            alias_type = str(after_snapshot.alias_type_by_index.get(idx) or before_snapshot.alias_type_by_index.get(idx) or "").strip().lower()
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
            "[KR-QualityGate] "
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


def _apply_kr_bank_autocalibration_to_oto_file(
    oto_path: str,
    *,
    custom_map=None,
    validate_fn=None,
    log_fn=None,
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "enabled": False,
        "changed_lines": 0,
        "groups": {},
    }
    if not _env_bool("UTOA_KR_BANK_AUTOCAL_ENABLE", True):
        return stats
    if not oto_path or not os.path.exists(oto_path):
        return stats
    if validate_fn is None:
        return stats

    text = read_text_with_fallback(oto_path)
    lines = text.splitlines()
    parsed_rows: Dict[int, Dict[str, object]] = {}
    alias_types: Dict[int, str] = {}
    for idx, line in enumerate(lines):
        row = parse_oto_line(line)
        if not row:
            continue
        a_type = str(classify_alias(str(row.get("alias", "")), custom_map) or "").strip().lower()
        parsed_rows[idx] = dict(row)
        alias_types[idx] = a_type

    if not parsed_rows:
        return stats

    cv_min_rows = int(max(4, _env_float("UTOA_KR_BANK_AUTOCAL_MIN_ROWS_CV", 18)))
    vc_min_rows = int(max(4, _env_float("UTOA_KR_BANK_AUTOCAL_MIN_ROWS_VC", 12)))
    iqr_mul = float(max(0.4, _env_float("UTOA_KR_BANK_AUTOCAL_IQR_MUL", 1.8)))
    blend = float(_clamp(_env_float("UTOA_KR_BANK_AUTOCAL_BLEND", 0.72), 0.05, 1.0))

    group_keys: Dict[int, str] = {}
    group_values: Dict[str, Dict[str, List[float]]] = {
        "cv_family": {"pre": [], "cons_gap": [], "cut_gap": []},
        "vc_family": {"pre": [], "cons_gap": [], "cut_gap": []},
    }
    for idx, row in parsed_rows.items():
        a_type = alias_types.get(idx, "")
        if a_type in {"cv", "cv_head", "vcv", "mono"}:
            gk = "cv_family"
        elif a_type in {"vc", "vv"}:
            gk = "vc_family"
        else:
            continue
        pre = float(row["pre"])
        cons = float(row["cons"])
        cut_abs = abs(float(row["cutoff"]))
        group_keys[idx] = gk
        group_values[gk]["pre"].append(pre)
        group_values[gk]["cons_gap"].append(max(cons - pre, 6.0))
        group_values[gk]["cut_gap"].append(max(cut_abs - cons, 8.0))

    bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for gk, values_map in group_values.items():
        row_count = len(values_map["pre"])
        min_rows = cv_min_rows if gk == "cv_family" else vc_min_rows
        if row_count < min_rows:
            continue
        bounds[gk] = {}
        for name, values in values_map.items():
            q1 = _quantile(values, 0.25)
            q3 = _quantile(values, 0.75)
            iqr = max(0.0, q3 - q1)
            lo = q1 - (iqr_mul * iqr)
            hi = q3 + (iqr_mul * iqr)
            if name == "pre":
                lo = max(lo, 12.0)
                hi = min(max(hi, lo + 8.0), 340.0)
            elif name == "cons_gap":
                lo = max(lo, 8.0)
                hi = min(max(hi, lo + 8.0), 320.0)
            else:
                lo = max(lo, 10.0)
                hi = min(max(hi, lo + 8.0), 360.0)
            bounds[gk][name] = (float(lo), float(hi))

    if not bounds:
        stats["enabled"] = True
        return stats

    changed = 0
    for idx, row in parsed_rows.items():
        gk = group_keys.get(idx)
        if not gk or gk not in bounds:
            continue
        a_type = alias_types.get(idx, "")
        pre_old = float(row["pre"])
        cons_old = float(row["cons"])
        cut_abs_old = abs(float(row["cutoff"]))
        cons_gap_old = max(cons_old - pre_old, 6.0)
        cut_gap_old = max(cut_abs_old - cons_old, 8.0)
        pre_new = _soft_clip(pre_old, *bounds[gk]["pre"], blend=blend)
        cons_gap_new = _soft_clip(cons_gap_old, *bounds[gk]["cons_gap"], blend=blend)
        cut_gap_new = _soft_clip(cut_gap_old, *bounds[gk]["cut_gap"], blend=blend)
        cons_new = pre_new + cons_gap_new
        cut_new = -(cons_new + cut_gap_new)
        try:
            o2, c2, ct2, p2, ov2 = validate_fn(
                float(row["offset"]),
                float(cons_new),
                float(cut_new),
                float(pre_new),
                float(row["ovl"]),
                alias_type=a_type,
            )
        except TypeError:
            o2, c2, ct2, p2, ov2 = validate_fn(
                float(row["offset"]),
                float(cons_new),
                float(cut_new),
                float(pre_new),
                float(row["ovl"]),
            )
        if (
            abs(float(o2) - float(row["offset"])) > 1e-6
            or abs(float(c2) - float(row["cons"])) > 1e-6
            or abs(float(ct2) - float(row["cutoff"])) > 1e-6
            or abs(float(p2) - float(row["pre"])) > 1e-6
            or abs(float(ov2) - float(row["ovl"])) > 1e-6
        ):
            row["offset"] = float(o2)
            row["cons"] = float(c2)
            row["cutoff"] = float(ct2)
            row["pre"] = float(p2)
            row["ovl"] = float(ov2)
            lines[idx] = _format_oto_row_line(row)
            changed += 1

    if changed > 0:
        with open(oto_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line).rstrip("\n") + "\n")

    stats["enabled"] = True
    stats["changed_lines"] = int(changed)
    stats["groups"] = {
        gk: {
            "rows": int(len(group_values[gk]["pre"])),
            "pre_bounds": list(bounds[gk]["pre"]),
            "cons_gap_bounds": list(bounds[gk]["cons_gap"]),
            "cut_gap_bounds": list(bounds[gk]["cut_gap"]),
        }
        for gk in bounds.keys()
    }
    if callable(log_fn):
        cv_rows = int(len(group_values["cv_family"]["pre"]))
        vc_rows = int(len(group_values["vc_family"]["pre"]))
        log_fn(
            f"[KR-AutoCal] bank autocal changed: {changed} lines "
            f"(cv_rows={cv_rows}, vc_rows={vc_rows}, blend={blend:.2f}, iqr={iqr_mul:.2f})"
        )
    return stats


def _normalize_ml_route(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"e2e_hybrid", "e2e", "hybrid"}:
        return "e2e_hybrid"
    return "legacy"


def _current_ml_route() -> str:
    return _normalize_ml_route(os.environ.get("UTOA_ML_ROUTE", "legacy"))


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

            search_start = pre_abs + _env_float("UTOA_KR_MEL_REFINE_SEARCH_START_FROM_PRE_MS", 14.0)
            search_end = cut_abs - _env_float("UTOA_KR_MEL_REFINE_SEARCH_END_FROM_CUT_MS", 8.0)
            if next_anchor is not None:
                search_end = min(search_end, next_anchor - _env_float("UTOA_KR_MEL_REFINE_NEXT_ANCHOR_MARGIN_MS", 8.0))
            if search_end <= search_start + _env_float("UTOA_KR_MEL_REFINE_MIN_SEARCH_SPAN_MS", 25.0):
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

            contrast_min = _env_float("UTOA_KR_MEL_REFINE_CONTRAST_MIN", 0.11)
            db_drop_min = _env_float("UTOA_KR_MEL_REFINE_DB_DROP_MIN", 2.3)
            if is_sibilant:
                contrast_min += _env_float("UTOA_KR_MEL_REFINE_SIBILANT_CONTRAST_BONUS", 0.03)
                db_drop_min += _env_float("UTOA_KR_MEL_REFINE_SIBILANT_DB_DROP_BONUS", 0.4)
            elif is_plosive:
                contrast_min += _env_float("UTOA_KR_MEL_REFINE_PLOSIVE_CONTRAST_BONUS", 0.01)
                db_drop_min += _env_float("UTOA_KR_MEL_REFINE_PLOSIVE_DB_DROP_BONUS", 0.2)

            if contrast < contrast_min and db_drop < db_drop_min:
                continue
            valley_energy_cap = _env_float("UTOA_KR_MEL_REFINE_VALLEY_ENERGY_MAX", 0.36)
            valley_db_cap = _env_float("UTOA_KR_MEL_REFINE_VALLEY_DB_MARGIN", 6.0)
            if is_sibilant:
                valley_energy_cap += _env_float("UTOA_KR_MEL_REFINE_SIBILANT_VALLEY_E_DELTA", 0.02)
            if valley_e > valley_energy_cap and valley_db > (db_sil_th + valley_db_cap):
                continue
            valley_f0_cap = _env_float("UTOA_KR_MEL_REFINE_VALLEY_F0_MAX", 0.70)
            f0_contrast_guard = _env_float("UTOA_KR_MEL_REFINE_F0_CONTRAST_MIN", 0.17)
            if valley_f0v > valley_f0_cap and contrast < f0_contrast_guard:
                continue
            tail_guard_ms = _env_float("UTOA_KR_MEL_REFINE_TAIL_GUARD_MS", 12.0)
            if is_sibilant:
                tail_guard_ms += _env_float("UTOA_KR_MEL_REFINE_SIBILANT_TAIL_GUARD_ADD_MS", 2.0)
            if valley_t >= cut_abs - tail_guard_ms:
                continue

            cut_shift_ms = _env_float("UTOA_KR_MEL_REFINE_TARGET_SHIFT_MS", 2.0)
            target_cut_abs = valley_t + cut_shift_ms
            min_cut_abs = pre_abs + _env_float("UTOA_KR_MEL_REFINE_MIN_CUT_FROM_PRE_MS", 20.0)
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

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line.rstrip("\n") + "\n")
    return changed


def run_kr_post_file_pipeline(context: KrPostFilePipelineContext):
    wav_dir = resolve_wav_dir_from_tg_folder(context.tg_folder)
    ml_route = _normalize_ml_route(getattr(context, "ml_route", "") or _current_ml_route())
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

    bank_autocal_stats = _apply_kr_bank_autocalibration_to_oto_file(
        context.out_path,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        log_fn=context.log_fn,
    )
    if isinstance(context.runtime_report, dict):
        context.runtime_report["bank_autocal"] = dict(bank_autocal_stats)

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

    if callable(context.log_fn):
        context.log_fn("[OTO-ML] KR finalize order: legacy ML -> legacy post-filters")
    ml_before_snapshot = _take_oto_snapshot(context.out_path, custom_map=context.custom_map)
    _run_ml_stage()
    ml_qg_stats = _apply_stage_quality_gate(
        context.out_path,
        stage_name="ml",
        before_snapshot=ml_before_snapshot,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        log_fn=context.log_fn,
        runtime_report=context.runtime_report,
    )
    if isinstance(context.runtime_report, dict):
        ml_report = context.runtime_report.get("ml")
        if isinstance(ml_report, dict):
            ml_report["quality_gate"] = dict(ml_qg_stats)

    post_before_snapshot = _take_oto_snapshot(context.out_path, custom_map=context.custom_map)
    _run_kr_legacy_post_filters()
    _apply_stage_quality_gate(
        context.out_path,
        stage_name="post",
        before_snapshot=post_before_snapshot,
        custom_map=context.custom_map,
        validate_fn=context.validate_fn,
        log_fn=context.log_fn,
        runtime_report=context.runtime_report,
    )

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

    try:
        cont_stats = apply_continuity_clamp_to_oto_file(
            context.out_path,
            classify_alias_fn=lambda alias_text, custom_map=None: classify_alias(alias_text, custom_map),
            validate_fn=context.validate_fn,
            custom_map=context.custom_map,
            env_prefix="UTOA_KR",
            language="korean",
            format_type=(context.auto_gen_format or "cvvc"),
            log_fn=context.log_fn,
            log_tag="[KR-Continuity]",
        )
        log_changed_lines(
            context.log_fn,
            "[KR-Continuity]",
            int(cont_stats.get("rows_adjusted", 0)),
            "continuity clamp changed",
        )
    except Exception as exc:
        if callable(context.log_fn):
            context.log_fn(f"[KR-Continuity] clamp failed: {exc}")


__all__ = [
    "KrPostFilePipelineContext",
    "apply_kr_mel_refine_to_oto_file",
    "apply_kr_wav_duration_safety_to_oto_file",
    "run_kr_post_file_pipeline",
    "sanitize_kr_oto_for_wav_duration",
]
