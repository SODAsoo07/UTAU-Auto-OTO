from __future__ import annotations

import os
from typing import Iterable


FATAL_GENERATION_FAILURE_REASONS = frozenset(
    {
        "textgrid_missing",
        "textgrid_load_failed",
        "tier_missing",
        "empty_intervals",
        "mapping_failed_empty_intervals",
        "no_valid_alias",
        "mapping_failed",
        "mapping_failed_spn_heavy",
        "mapping_failed_insufficient_phones",
        "mapping_failed_no_words_support",
        "file_exception",
    }
)

_SKIP_EMPTY_INTERVALS_SPECIAL_STEMS = frozenset(
    {
        "br",
        "br2",
        "br3",
        "bre",
        "breath",
    }
)


def _stem_for_special_skip(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(str(fname or "").strip()))[0].strip().lower()
    while stem.startswith("_"):
        stem = stem[1:]
    return stem


def is_zero_placeholder_line(line: str) -> bool:
    if not line or "=" not in line:
        return False
    try:
        parts = line.split("=", 1)[1].split(",")
        if len(parts) < 6:
            return False
        values = [float(str(parts[i]).strip()) for i in range(1, 6)]
        return all(abs(value) < 1e-6 for value in values)
    except Exception:
        return False


def lines_are_zero_placeholders(lines: Iterable[str]) -> bool:
    parsed = [line for line in (lines or []) if line and "=" in line]
    return bool(parsed) and all(is_zero_placeholder_line(line) for line in parsed)


def build_auto_placeholder_block_error(
    *,
    fname: str,
    lines: Iterable[str],
    reason: str,
    use_template: bool,
) -> str:
    reason_key = str(reason or "").strip().lower()
    stem = _stem_for_special_skip(fname)
    if (
        reason_key in {"empty_intervals", "mapping_failed_empty_intervals"}
        and stem in _SKIP_EMPTY_INTERVALS_SPECIAL_STEMS
    ):
        # Breath-like dedicated files can legitimately have sparse/empty intervals.
        # Treat these as skip targets instead of blocking the entire generation save.
        return ""
    is_zero_placeholder = lines_are_zero_placeholders(lines)
    is_fatal_generation_failure = reason_key in FATAL_GENERATION_FAILURE_REASONS
    if not is_zero_placeholder and not is_fatal_generation_failure:
        return ""

    if is_fatal_generation_failure:
        return (
            f"[ERROR] {fname}: 자동 설정에 필요한 TextGrid/tier/interval/mapping 정보를 확보하지 못해 저장을 중단했습니다. "
            f"reason={reason}. TextGrid의 phones/words tier 이름, 정렬 결과, WAV-TextGrid 파일명 매칭을 확인하세요."
        )
    if use_template:
        return (
            f"[ERROR] {fname}: 0값 템플릿 OTO를 보정하지 못해 저장을 중단했습니다. "
            f"reason={reason}. TextGrid의 phones/words tier 이름, 정렬 결과, WAV-TextGrid 파일명 매칭을 확인하세요."
        )
    return (
        f"[ERROR] {fname}: 자동 생성 CVC/CV 계열 placeholder OTO가 보정되지 않아 저장을 중단했습니다. "
        f"reason={reason}. TextGrid의 phones/words tier 이름, 정렬 결과, WAV-TextGrid 파일명 매칭을 확인하세요."
    )


def build_offset_zero_collapse_error(
    lines: Iterable[str],
    *,
    auto_gen_format: str,
) -> str:
    total_rows = 0
    zero_offset_rows = 0
    for line in lines or []:
        if not line or "=" not in line:
            continue
        try:
            parts = line.split("=", 1)[1].split(",")
            if len(parts) < 6:
                continue
            offset_value = float(str(parts[1]).strip())
        except Exception:
            continue
        total_rows += 1
        if abs(offset_value) < 1e-6:
            zero_offset_rows += 1
    if total_rows < 5:
        return ""
    ratio = zero_offset_rows / float(total_rows)
    if ratio < 0.90:
        return ""
    return (
        f"[ERROR] OTO 생성 결과의 offset이 비정상적으로 0에 몰려 저장을 중단했습니다 "
        f"(zero_offset_rows={zero_offset_rows}/{total_rows}, ratio={ratio:.1%}, format={auto_gen_format}). "
        "자동 alias placeholder가 보정되지 않았을 가능성이 큽니다. TextGrid phones tier, 정렬 품질, "
        "WAV-TextGrid 파일명 매칭을 확인하세요."
    )


__all__ = [
    "FATAL_GENERATION_FAILURE_REASONS",
    "build_auto_placeholder_block_error",
    "build_offset_zero_collapse_error",
    "is_zero_placeholder_line",
    "lines_are_zero_placeholders",
]
