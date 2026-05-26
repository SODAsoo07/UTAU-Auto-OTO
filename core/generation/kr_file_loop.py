from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from .file_stages import handle_kr_file_context_status, handle_kr_loop_prep_status
from .placeholder_guards import build_auto_placeholder_block_error


def update_wav_name_map_from_file_context(
    *,
    wav_name_map: MutableMapping[str, str],
    fname: str,
    file_ctx: Any,
) -> str:
    output_wav_name = str(getattr(file_ctx, "output_wav_name", "") or "")
    if not output_wav_name:
        return ""
    wav_name_map.setdefault(str(fname), output_wav_name)
    real_wav_name = str(getattr(file_ctx, "real_wav_name", "") or "")
    if real_wav_name and real_wav_name != fname:
        wav_name_map.setdefault(real_wav_name, output_wav_name)
    return output_wav_name


def rollback_auto_placeholder_fallback(
    *,
    final_lines: list[str],
    errors: list[str],
    placeholder_block_errors: list[str],
    prev_final_len: int,
    fname: str,
    lines: list[str],
    reason: str,
    use_template: bool,
    log_fn: Callable[[str], None],
) -> bool:
    err = build_auto_placeholder_block_error(
        fname=fname,
        lines=lines,
        reason=reason,
        use_template=use_template,
    )
    if not err:
        return False
    del final_lines[int(prev_final_len) :]
    log_fn(err)
    errors.append(err)
    placeholder_block_errors.append(err)
    return True


def handle_kr_file_context_status_with_guard(
    *,
    file_ctx: Any,
    fname: str,
    lines: list[str],
    final_lines: list[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    prev_final_len: int,
    use_template: bool,
    errors: list[str],
    placeholder_block_errors: list[str],
) -> bool:
    handled = handle_kr_file_context_status(
        file_ctx=file_ctx,
        fname=fname,
        lines=lines,
        final_lines=final_lines,
        alias_suffix=alias_suffix,
        log_fn=log_fn,
        record_unset_lines_fn=record_unset_lines_fn,
        apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
    )
    if handled:
        rollback_auto_placeholder_fallback(
            final_lines=final_lines,
            errors=errors,
            placeholder_block_errors=placeholder_block_errors,
            prev_final_len=prev_final_len,
            fname=fname,
            lines=lines,
            reason=str(getattr(file_ctx, "status", "file_context") or "file_context"),
            use_template=use_template,
            log_fn=log_fn,
        )
    return handled


def handle_kr_loop_prep_status_with_guard(
    *,
    loop_prep: Any,
    fname: str,
    lines: list[str],
    final_lines: list[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    prev_final_len: int,
    use_template: bool,
    errors: list[str],
    placeholder_block_errors: list[str],
) -> bool:
    handled = handle_kr_loop_prep_status(
        loop_prep=loop_prep,
        fname=fname,
        lines=lines,
        final_lines=final_lines,
        alias_suffix=alias_suffix,
        log_fn=log_fn,
        record_unset_lines_fn=record_unset_lines_fn,
        apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
    )
    if handled:
        rollback_auto_placeholder_fallback(
            final_lines=final_lines,
            errors=errors,
            placeholder_block_errors=placeholder_block_errors,
            prev_final_len=prev_final_len,
            fname=fname,
            lines=lines,
            reason=str(getattr(loop_prep, "status", "loop_prep") or "loop_prep"),
            use_template=use_template,
            log_fn=log_fn,
        )
    return handled


__all__ = [
    "handle_kr_file_context_status_with_guard",
    "handle_kr_loop_prep_status_with_guard",
    "rollback_auto_placeholder_fallback",
    "update_wav_name_map_from_file_context",
]
