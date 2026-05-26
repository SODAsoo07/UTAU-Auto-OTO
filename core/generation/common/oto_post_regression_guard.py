from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from core.generator_finish import write_oto_lines
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class OtoBaselineSnapshot:
    by_key: dict[tuple[str, str, int], dict[str, object]]


def _iter_rows_with_occurrence(lines: list[str]):
    counters: dict[tuple[str, str], int] = {}
    for idx, line in enumerate(lines):
        row = parse_oto_line(line)
        if not row:
            continue
        wav = str(row.get("wav", "") or "")
        alias = str(row.get("alias", "") or "")
        base_key = (wav, alias)
        occ = int(counters.get(base_key, 0))
        counters[base_key] = occ + 1
        yield idx, (wav, alias, occ), row, str(line).rstrip("\n")


def capture_oto_baseline_snapshot(oto_path: str) -> OtoBaselineSnapshot:
    by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    if not oto_path or not os.path.exists(oto_path):
        return OtoBaselineSnapshot(by_key={})
    lines = read_text_with_fallback(oto_path).splitlines()
    for idx, key, row, line in _iter_rows_with_occurrence(lines):
        by_key[key] = {
            "line_idx": int(idx),
            "line": line,
            "row": dict(row),
        }
    return OtoBaselineSnapshot(by_key=by_key)


def _alias_family(alias_type: str) -> str:
    a_type = str(alias_type or "").strip().lower()
    if a_type in {"vc", "vv", "vcv", "vc_only"}:
        return "bridge"
    if a_type in {"cv", "cv_head", "mono", "cvvc", "cvc", "vv_only"}:
        return "cv"
    return "other"


def _row_geom(row: dict[str, object]) -> dict[str, float]:
    off = float(row.get("offset", 0.0) or 0.0)
    pre = max(float(row.get("pre", 0.0) or 0.0), 0.0)
    cons = max(float(row.get("cons", 0.0) or 0.0), 0.0)
    cut_abs = abs(float(row.get("cutoff", 0.0) or 0.0))
    pre_abs = off + pre
    return {
        "offset": off,
        "pre": pre,
        "cons": cons,
        "cut_abs": cut_abs,
        "pre_abs": pre_abs,
        "cons_gap": max(cons - pre, 0.0),
        "tail": max(cut_abs - cons, 0.0),
    }


def _needs_revert(before: dict[str, float], after: dict[str, float], *, family: str) -> tuple[bool, list[str]]:
    if family == "bridge":
        pre_shift_limit = 34.0
        cut_shift_limit = 28.0
        cons_gap_ratio = 0.34
        tail_ratio = 0.30
        min_cons_gap = 10.0
        min_tail = 18.0
    else:
        pre_shift_limit = 46.0
        cut_shift_limit = 34.0
        cons_gap_ratio = 0.28
        tail_ratio = 0.34
        min_cons_gap = 12.0
        min_tail = 22.0

    reasons: list[str] = []
    pre_shift = abs(float(after["pre_abs"]) - float(before["pre_abs"]))
    cut_shift = abs(float(after["cut_abs"]) - float(before["cut_abs"]))
    cons_shift = abs(float(after["cons"]) - float(before["cons"]))

    if after["cons"] <= (after["pre"] + 4.0) and before["cons"] >= (before["pre"] + 10.0):
        reasons.append("cons_collapse")
    if after["cut_abs"] <= (after["cons"] + 6.0) and before["cut_abs"] >= (before["cons"] + 18.0):
        reasons.append("cutoff_collapse")
    if pre_shift >= pre_shift_limit and cut_shift >= cut_shift_limit:
        reasons.append("large_anchor_shift")
    if before["cons_gap"] >= min_cons_gap and after["cons_gap"] <= max(5.0, before["cons_gap"] * cons_gap_ratio):
        reasons.append("cons_gap_shrink")
    if before["tail"] >= min_tail and after["tail"] <= max(7.0, before["tail"] * tail_ratio):
        reasons.append("tail_shrink")
    if before["pre"] >= 18.0 and after["pre"] <= (before["pre"] * 0.42) and cons_shift >= 16.0:
        reasons.append("pre_shrink")

    if not reasons:
        return False, []

    hard_reasons = {"cons_collapse", "cutoff_collapse", "large_anchor_shift"}
    hard_hit = any(r in hard_reasons for r in reasons)
    minor_count = len([r for r in reasons if r not in hard_reasons])
    return bool(hard_hit or minor_count >= 2), reasons


def apply_oto_post_regression_guard(
    oto_path: str,
    *,
    baseline_snapshot: OtoBaselineSnapshot,
    alias_type_resolver: Callable[[str], str],
    log_fn: Callable[[str], None] | None = None,
    log_tag: str = "[OTO-RegressionGuard]",
) -> dict[str, object]:
    stats: dict[str, object] = {
        "checked": 0,
        "reverted": 0,
        "reasons": {},
    }
    if not oto_path or not os.path.exists(oto_path):
        return stats
    if not isinstance(baseline_snapshot, OtoBaselineSnapshot):
        return stats
    if not baseline_snapshot.by_key:
        return stats

    current_lines = read_text_with_fallback(oto_path).splitlines()
    if not current_lines:
        return stats

    debug_detail = _env_bool("UTOA_POST_GUARD_DEBUG", False)
    detail_logs: list[str] = []
    reason_counts: dict[str, int] = {}
    changed = 0
    checked = 0

    for line_idx, key, curr_row, _line in _iter_rows_with_occurrence(current_lines):
        base_entry = baseline_snapshot.by_key.get(key)
        if not base_entry:
            continue

        alias = str(curr_row.get("alias", "") or "")
        try:
            alias_type = str(alias_type_resolver(alias) or "")
        except Exception:
            alias_type = ""
        family = _alias_family(alias_type)
        if family == "other":
            continue

        checked += 1
        before = _row_geom(base_entry["row"])
        after = _row_geom(curr_row)
        should_revert, reasons = _needs_revert(before, after, family=family)
        if not should_revert:
            continue

        base_line = str(base_entry.get("line", "") or "")
        if current_lines[line_idx].rstrip("\n") == base_line:
            continue
        current_lines[line_idx] = base_line
        changed += 1
        for reason in reasons:
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        if debug_detail and len(detail_logs) < 40:
            detail_logs.append(
                f"{log_tag} rollback alias={alias} type={alias_type or 'unknown'} "
                f"reason={'+'.join(reasons)} "
                f"(pre_abs {before['pre_abs']:.1f}->{after['pre_abs']:.1f}, "
                f"cons_gap {before['cons_gap']:.1f}->{after['cons_gap']:.1f}, "
                f"tail {before['tail']:.1f}->{after['tail']:.1f})"
            )

    if changed > 0:
        write_oto_lines(oto_path, [str(line).rstrip("\n") for line in current_lines])

    if callable(log_fn) and debug_detail:
        for msg in detail_logs:
            log_fn(msg)

    stats["checked"] = int(checked)
    stats["reverted"] = int(changed)
    stats["reasons"] = reason_counts
    return stats


__all__ = [
    "OtoBaselineSnapshot",
    "apply_oto_post_regression_guard",
    "capture_oto_baseline_snapshot",
]
