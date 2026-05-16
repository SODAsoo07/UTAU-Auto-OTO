from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from core.alias_family_tuning import resolve_continuity_tuning
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _call_validate(validate_fn, offset, consonant, cutoff, pre, ovl, *, alias_type=""):
    try:
        return validate_fn(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    except TypeError:
        return validate_fn(offset, consonant, cutoff, pre, ovl)


def _row_anchor(row: Dict[str, object]) -> float:
    try:
        return float(row.get("offset", 0.0)) + float(row.get("pre", 0.0))
    except Exception:
        try:
            return float(row.get("offset", 0.0))
        except Exception:
            return 0.0


def _alias_family(alias_type: str) -> str:
    a = str(alias_type or "").strip().lower()
    if a in {"cv", "cv_head", "mono", "vcv"}:
        return "cv"
    if a in {"vc", "vv"}:
        return "vc"
    return ""


def _scope_allows(scope: str, family: str) -> bool:
    s = str(scope or "").strip().lower()
    if s == "all":
        return family in {"cv", "vc", "vv"}
    if s == "cv_only":
        return family == "cv"
    if s == "vc_only":
        return family in {"vc", "vv"}
    return family in {"cv", "vc", "vv"}


def _format_row_line(row: Dict[str, object]) -> str:
    return (
        f"{row['wav']}={row['alias']},"
        f"{float(row['offset']):.2f},{float(row['cons']):.2f},{float(row['cutoff']):.2f},"
        f"{float(row['pre']):.2f},{float(row['ovl']):.2f}"
    )


def apply_continuity_clamp_to_oto_file(
    oto_path: str,
    *,
    classify_alias_fn: Callable[[str, object], str],
    validate_fn: Callable,
    custom_map=None,
    env_prefix: str = "UTOA_JA",
    language: str = "",
    format_type: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
    log_tag: str = "[Continuity]",
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "enabled": False,
        "scope": "cv_vc",
        "rows_total": 0,
        "rows_adjusted": 0,
    }
    if not oto_path or not os.path.exists(oto_path):
        return stats

    enable_key = f"{env_prefix}_CONT_CLAMP_ENABLE"
    if not _env_bool(enable_key, True):
        return stats

    scope_key = f"{env_prefix}_CONT_CLAMP_SCOPE"
    scope = str(os.environ.get(scope_key, "cv_vc")).strip().lower()
    if scope not in {"cv_vc", "cv_only", "vc_only", "all"}:
        scope = "cv_vc"

    global_backtrack_key = f"{env_prefix}_CONT_CLAMP_OFFSET_BACKTRACK_MAX"
    global_min_gap_key = f"{env_prefix}_CONT_CLAMP_OFFSET_MIN_GAP"
    global_step_max_key = f"{env_prefix}_CONT_CLAMP_OFFSET_STEP_MAX"
    global_cut_delta_key = f"{env_prefix}_CONT_CLAMP_CUTOFF_DELTA_MAX"
    global_min_cut_gap_key = f"{env_prefix}_CONT_CLAMP_MIN_CUTOFF_GAP"

    global_backtrack = max(0.0, _env_float(global_backtrack_key, -1.0))
    global_min_gap = max(0.0, _env_float(global_min_gap_key, -1.0))
    global_step_max = _env_float(global_step_max_key, -1.0)
    global_cut_delta = _env_float(global_cut_delta_key, -1.0)
    global_min_cut_gap = _env_float(global_min_cut_gap_key, -1.0)

    lines = read_text_with_fallback(oto_path).splitlines()
    if not lines:
        return stats

    rows_by_wav: Dict[str, List[Tuple[int, Dict[str, object], str, str]]] = {}
    parsed_rows: Dict[int, Dict[str, object]] = {}
    alias_types: Dict[int, str] = {}
    for idx, line in enumerate(lines):
        row = parse_oto_line(line)
        if not row:
            continue
        alias_type = str(classify_alias_fn(str(row.get("alias", "")), custom_map) or "").strip().lower()
        family = _alias_family(alias_type)
        parsed_rows[idx] = dict(row)
        alias_types[idx] = alias_type
        rows_by_wav.setdefault(str(row.get("wav", "")), []).append((idx, parsed_rows[idx], alias_type, family))

    if not parsed_rows:
        stats["enabled"] = True
        stats["scope"] = scope
        return stats

    adjusted = 0
    target_rows = 0
    for _wav, items in rows_by_wav.items():
        ordered = sorted(items, key=lambda item: (_row_anchor(item[1]), item[0]))
        prev_by_family: Dict[str, Dict[str, float]] = {}
        for line_idx, row, alias_type, family in ordered:
            if not _scope_allows(scope, family):
                continue
            if family not in {"cv", "vc", "vv"}:
                continue
            target_rows += 1

            before = (
                float(row.get("offset", 0.0)),
                float(row.get("cons", 0.0)),
                float(row.get("cutoff", 0.0)),
                float(row.get("pre", 0.0)),
                float(row.get("ovl", 0.0)),
            )

            off = float(row["offset"])
            cons = float(row["cons"])
            cut = float(row["cutoff"])
            pre = float(row["pre"])
            ovl = float(row["ovl"])
            cut_abs = abs(cut)

            _bucket, _edge_role, _cons_type, tuning = resolve_continuity_tuning(
                language=str(language or ""),
                format_type=str(format_type or ""),
                alias_type=str(alias_type or ""),
                alias_text=str(row.get("alias", "") or ""),
            )
            backtrack_max = float(tuning.backtrack_max)
            min_gap = float(tuning.min_gap)
            step_max = float(tuning.step_max)
            cutoff_delta_max = float(tuning.cutoff_delta_max)
            min_cut_gap = float(tuning.min_cut_gap)

            if global_backtrack >= 0.0:
                backtrack_max = float(global_backtrack)
            if global_min_gap >= 0.0:
                min_gap = float(global_min_gap)
            if global_step_max >= 0.0:
                step_max = max(min_gap + 20.0, float(global_step_max))
            if global_cut_delta >= 0.0:
                cutoff_delta_max = max(20.0, float(global_cut_delta))
            if global_min_cut_gap >= 0.0:
                min_cut_gap = max(2.0, float(global_min_cut_gap))

            prev = prev_by_family.get(family)
            if prev is not None:
                lower = float(prev["offset"]) - backtrack_max
                upper = float(prev["offset"]) + step_max
                off = _clamp(off, lower, upper)
                off = max(off, float(prev["offset"]) + min_gap)

                cut_lower = max(cons + min_cut_gap, float(prev["cut_abs"]) - cutoff_delta_max)
                cut_upper = max(cut_lower + min_cut_gap, float(prev["cut_abs"]) + cutoff_delta_max)
                cut_abs = _clamp(cut_abs, cut_lower, cut_upper)
                cut = -max(cut_abs, 0.0)

            try:
                off, cons, cut, pre, ovl = _call_validate(
                    validate_fn,
                    off,
                    cons,
                    cut,
                    pre,
                    ovl,
                    alias_type=alias_type,
                )
            except Exception:
                continue

            row["offset"] = float(off)
            row["cons"] = float(cons)
            row["cutoff"] = float(cut)
            row["pre"] = float(pre)
            row["ovl"] = float(ovl)
            parsed_rows[line_idx] = dict(row)

            after = (
                float(row["offset"]),
                float(row["cons"]),
                float(row["cutoff"]),
                float(row["pre"]),
                float(row["ovl"]),
            )
            if any(abs(a - b) > 1e-6 for a, b in zip(before, after)):
                adjusted += 1

            prev_by_family[family] = {
                "offset": float(row["offset"]),
                "cut_abs": abs(float(row["cutoff"])),
            }

    if adjusted > 0:
        out_lines: List[str] = []
        for idx, line in enumerate(lines):
            row = parsed_rows.get(idx)
            if not row:
                out_lines.append(line)
                continue
            out_lines.append(_format_row_line(row))
        with open(oto_path, "w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(str(line).rstrip("\n") + "\n")

    stats.update(
        {
            "enabled": True,
            "scope": scope,
            "rows_total": int(target_rows),
            "rows_adjusted": int(adjusted),
        }
    )
    if callable(log_fn):
        log_fn(
            f"{log_tag} clamp: adjusted={int(adjusted)}/{int(target_rows)} "
            f"(scope={scope}, lang={str(language or '')}, fmt={str(format_type or '')})"
        )
    return stats


__all__ = [
    "apply_continuity_clamp_to_oto_file",
]
