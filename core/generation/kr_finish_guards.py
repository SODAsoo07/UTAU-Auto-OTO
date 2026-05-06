from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Dict, List

from core.generation.placeholder_guards import build_offset_zero_collapse_error
from core.generator_finish import finalize_generator_finish


def maybe_block_kr_generation_output(
    *,
    placeholder_block_errors: Sequence[Any],
    final_lines: Sequence[str],
    auto_gen_format: str,
    runtime_report: Any,
    errors: List[str],
    log_fn: Callable[[str], None],
    finish_context: Any,
    log_unset_summary_fn: Callable[[], None],
    build_offset_zero_collapse_error_fn: Callable[..., str] = build_offset_zero_collapse_error,
    finalize_generator_finish_fn: Callable[[Any], None] = finalize_generator_finish,
) -> bool:
    if placeholder_block_errors:
        summary_err = (
            f"[ERROR] 자동 설정을 완료할 수 없는 파일이 있어 OTO 저장을 중단했습니다 "
            f"(blocked_files={len(placeholder_block_errors)}, format={auto_gen_format})."
        )
        log_fn(summary_err)
        errors.append(summary_err)
        if isinstance(runtime_report, dict):
            report: Dict[str, Any] = runtime_report
            report["auto_placeholder_block_errors"] = list(placeholder_block_errors)
            report["auto_generation_block_errors"] = list(placeholder_block_errors)
        finalize_generator_finish_fn(finish_context)
        log_unset_summary_fn()
        return True

    zero_collapse_err = build_offset_zero_collapse_error_fn(
        final_lines,
        auto_gen_format=auto_gen_format,
    )
    if zero_collapse_err:
        log_fn(zero_collapse_err)
        errors.append(zero_collapse_err)
        if isinstance(runtime_report, dict):
            runtime_report["offset_zero_collapse_error"] = zero_collapse_err
        finalize_generator_finish_fn(finish_context)
        log_unset_summary_fn()
        return True

    return False


__all__ = ["maybe_block_kr_generation_output"]
