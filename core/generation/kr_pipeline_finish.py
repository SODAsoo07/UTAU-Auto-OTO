from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from typing import Any

from core.generation.kr_finish_guards import maybe_block_kr_generation_output
from core.generation.kr_missing_vowels import append_kr_missing_single_vowel_aliases
from core.generation.kr_output_finish import persist_kr_generation_output
from core.generation.review_queue import (
    export_review_queue_if_enabled,
    update_review_queue_runtime_report,
)
from core.generator_finish import GeneratorFinishContext, finalize_generator_finish


def finalize_kr_generation_pipeline(
    *,
    processed: int,
    total: int,
    errors: MutableSequence[str],
    gen_missing_vowels: bool,
    final_lines: MutableSequence[str],
    file_groups: Any,
    tg_entries: Any,
    single_vowel_span_by_tg_path: Any,
    norm_tg_path_key_fn: Callable[..., Any],
    load_textgrid_fn: Callable[..., Any],
    coerce_interval_tier_fn: Callable[..., Any],
    is_interval_like_tier_fn: Callable[..., Any],
    compute_noninitial_vowel_timing_fn: Callable[..., Any],
    validate_fn: Callable[..., Any],
    append_alias_rows_fn: Callable[..., Any],
    generate_openutau: bool,
    alias_suffix: str,
    log_fn: Callable[[str], None],
    placeholder_block_errors: Sequence[Any],
    auto_gen_format: str,
    runtime_report: Any,
    anchor_stats: Any,
    anchor_log_path: Any,
    anchor_log_dir: Any,
    cleanup_timing_jsonl: bool,
    log_unset_summary_fn: Callable[[], None],
    out_path: str,
    tg_folder: str,
    kr_profile: Any,
    custom_map: Any,
    custom_phonemes_path: str,
    enable_ml_correction: bool,
    ml_policy: Any,
    normalize_key_fn: Callable[..., Any],
    wav_name_map: dict[str, str],
    apply_output_wav_name_map_fn: Callable[..., Any],
    review_queue_export: bool,
    review_queue_path: str,
    row_apply_decisions: Sequence[dict[str, Any]],
    row_apply_mode_counts: dict[str, int],
    append_missing_vowels_fn: Callable[..., Any] = append_kr_missing_single_vowel_aliases,
    finish_context_factory: Callable[..., Any] = GeneratorFinishContext,
    maybe_block_output_fn: Callable[..., bool] = maybe_block_kr_generation_output,
    persist_output_fn: Callable[..., Any] = persist_kr_generation_output,
    export_review_queue_fn: Callable[..., Any] = export_review_queue_if_enabled,
    update_review_queue_report_fn: Callable[..., Any] = update_review_queue_runtime_report,
    finalize_finish_fn: Callable[[Any], None] = finalize_generator_finish,
) -> tuple[int, int, MutableSequence[str]]:
    if gen_missing_vowels:
        log_fn("누락된 단모음 alias 자동 생성을 시작합니다...")
        append_missing_vowels_fn(
            final_lines=final_lines,
            file_groups=file_groups,
            tg_entries=tg_entries,
            single_vowel_span_by_tg_path=single_vowel_span_by_tg_path,
            norm_tg_path_key_fn=norm_tg_path_key_fn,
            load_textgrid_fn=load_textgrid_fn,
            coerce_interval_tier_fn=coerce_interval_tier_fn,
            is_interval_like_tier_fn=is_interval_like_tier_fn,
            compute_noninitial_vowel_timing_fn=compute_noninitial_vowel_timing_fn,
            validate_fn=validate_fn,
            append_alias_rows_fn=append_alias_rows_fn,
            generate_openutau=generate_openutau,
            alias_suffix=alias_suffix,
            log_added_fn=lambda tg_info, alias: log_fn(
                f"추가: 단모음 alias 생성 -> {tg_info['real_name']} [{alias}]"
            ),
        )

    finish_context = finish_context_factory(
        log_fn=log_fn,
        anchor_stats=anchor_stats,
        anchor_log_path=anchor_log_path,
        anchor_log_dir=anchor_log_dir,
        cleanup_timing_jsonl=cleanup_timing_jsonl,
        timing_jsonl_prefix="timing_anchor_kr_",
    )

    if maybe_block_output_fn(
        placeholder_block_errors=placeholder_block_errors,
        final_lines=final_lines,
        auto_gen_format=auto_gen_format,
        runtime_report=runtime_report,
        errors=errors,
        log_fn=log_fn,
        finish_context=finish_context,
        log_unset_summary_fn=log_unset_summary_fn,
    ):
        return processed, total, errors

    persist_output_fn(
        out_path=out_path,
        final_lines=final_lines,
        tg_folder=tg_folder,
        kr_profile=kr_profile,
        custom_map=custom_map,
        custom_phonemes_path=custom_phonemes_path,
        enable_ml_correction=enable_ml_correction,
        auto_gen_format=auto_gen_format,
        ml_policy=ml_policy,
        runtime_report=runtime_report,
        log_fn=log_fn,
        validate_fn=validate_fn,
        normalize_key_fn=normalize_key_fn,
        wav_name_map=wav_name_map,
        apply_output_wav_name_map_fn=apply_output_wav_name_map_fn,
        errors=errors,
    )

    export_review_queue_fn(
        review_queue_export=review_queue_export,
        review_queue_path=review_queue_path,
        row_apply_decisions=row_apply_decisions,
        log_fn=log_fn,
    )

    if isinstance(runtime_report, dict):
        update_review_queue_report_fn(
            runtime_report,
            row_apply_mode_counts=row_apply_mode_counts,
            review_queue_export=review_queue_export,
            row_apply_decisions=row_apply_decisions,
            review_queue_path=review_queue_path,
        )

    finalize_finish_fn(finish_context)
    log_unset_summary_fn()
    return processed, total, errors


__all__ = ["finalize_kr_generation_pipeline"]
