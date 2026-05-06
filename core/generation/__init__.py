"""Shared generation contracts and normalization helpers."""

from .contracts import (
    GenerationRequest,
    attach_request_metadata,
    finalize_runtime_report,
    initialize_runtime_report,
)
from .normalizer import (
    KoreanNormalizedOptions,
    JapaneseNormalizedOptions,
    normalize_korean_generation_options,
    normalize_japanese_generation_options,
)
from .file_stages import (
    append_preserved_lines,
    handle_mapping_abstain_fallback,
    handle_mapping_failure_fallback,
    handle_ja_file_context_status,
    handle_ja_loop_prep_status,
    handle_kr_file_context_status,
    handle_kr_loop_prep_status,
    resolve_mapping_line_alias,
    resolve_mapping_failure_reason,
)
from .file_index import (
    build_wav_index,
    has_direct_wav_files,
    iter_files_with_suffix,
    iter_textgrid_files,
    iter_wav_files,
    list_wav_names,
)
from .placeholder_guards import (
    build_auto_placeholder_block_error,
    build_offset_zero_collapse_error,
    is_zero_placeholder_line,
    lines_are_zero_placeholders,
)
from .review_queue import (
    build_review_queue_rows,
    export_review_queue_if_enabled,
    update_review_queue_runtime_report,
)
from .kr_file_loop import (
    handle_kr_file_context_status_with_guard,
    handle_kr_loop_prep_status_with_guard,
    rollback_auto_placeholder_fallback,
    update_wav_name_map_from_file_context,
)
from .kr_file_runtime import prepare_kr_row_iteration_context
from .kr_base_timing import compute_kr_general_base_timing
from .kr_mapping_fallback import (
    force_kr_runtime_abstain_to_sequence_lock,
    handle_kr_mapping_failure_with_guard,
    has_kr_syllable_mapping_failure,
)
from .kr_missing_vowels import append_kr_missing_single_vowel_aliases
from .kr_finish_guards import maybe_block_kr_generation_output
from .kr_output_finish import persist_kr_generation_output
from .kr_pipeline_finish import finalize_kr_generation_pipeline
from .kr_post_timing import run_kr_general_post_timing_adjustments
from .kr_row_prep import (
    prepare_kr_alias_row_start,
    resolve_kr_row_jump_limits,
    try_append_kr_br_alias_row,
)
from .kr_row_context import prepare_kr_general_row_bounds
from .kr_row_selection import prepare_kr_general_cv_selection_context
from .row_apply import (
    apply_zero_template_compute_policy,
    build_review_required_unset,
    decide_row_application,
    evaluate_row_apply_policy,
    record_row_apply_decision,
)
from .kr_syllable_source import prepare_kr_syllable_source_context
from .kr_runtime_policy import prepare_kr_mapping_runtime_context
from .mapping_runtime import (
    build_common_mapping_runtime_payload,
    compute_runtime_low_conf_state,
    format_alignment_guard_summary,
    format_mapping_reason_schema_summary,
    format_mapping_summary,
    update_ja_mapping_runtime_report,
    update_mapping_vc_bridge_runtime_report,
    update_kr_mapping_runtime_report,
)
from .mapping_reason_logs import (
    build_ja_mapping_reason_log,
    build_kr_mapping_reason_log,
)
from .mapping_trace import (
    build_ja_mapping_candidate_trace,
    build_top_mapping_candidate_rows,
    maybe_append_cv_mapping_trace,
)
from .mapping_reason_codes import (
    COMMON_MAPPING_REASON_CODES,
    JA_MAPPING_REASON_CODES,
    KR_MAPPING_REASON_CODES,
    canonicalize_mapping_reason_code,
    get_known_mapping_reason_codes,
    is_known_mapping_reason_code,
    normalize_mapping_reason_code,
    resolve_mapping_reason_schema,
)
from .runtime_diagnostics import (
    build_mapping_abstain_meta,
    build_mapping_failure_meta,
)
from .plan_runtime import (
    build_language_runtime_state,
    build_common_plan_context,
    extract_language_runtime_state,
    log_sinsy_plan_guard,
    normalize_runtime_postprocess_state,
    recompute_common_plan_runtime_state,
    resolve_common_runtime_policy,
)

__all__ = [
    "GenerationRequest",
    "KoreanNormalizedOptions",
    "JapaneseNormalizedOptions",
    "attach_request_metadata",
    "append_preserved_lines",
    "append_kr_missing_single_vowel_aliases",
    "apply_zero_template_compute_policy",
    "build_auto_placeholder_block_error",
    "build_offset_zero_collapse_error",
    "build_review_queue_rows",
    "build_review_required_unset",
    "build_wav_index",
    "compute_kr_general_base_timing",
    "force_kr_runtime_abstain_to_sequence_lock",
    "decide_row_application",
    "export_review_queue_if_enabled",
    "handle_mapping_abstain_fallback",
    "handle_mapping_failure_fallback",
    "finalize_runtime_report",
    "handle_ja_file_context_status",
    "handle_ja_loop_prep_status",
    "handle_kr_file_context_status",
    "handle_kr_file_context_status_with_guard",
    "handle_kr_loop_prep_status",
    "handle_kr_loop_prep_status_with_guard",
    "handle_kr_mapping_failure_with_guard",
    "has_direct_wav_files",
    "has_kr_syllable_mapping_failure",
    "initialize_runtime_report",
    "iter_files_with_suffix",
    "iter_textgrid_files",
    "iter_wav_files",
    "is_zero_placeholder_line",
    "lines_are_zero_placeholders",
    "list_wav_names",
    "maybe_block_kr_generation_output",
    "persist_kr_generation_output",
    "finalize_kr_generation_pipeline",
    "prepare_kr_alias_row_start",
    "prepare_kr_general_cv_selection_context",
    "prepare_kr_general_row_bounds",
    "prepare_kr_row_iteration_context",
    "prepare_kr_syllable_source_context",
    "prepare_kr_mapping_runtime_context",
    "rollback_auto_placeholder_fallback",
    "resolve_kr_row_jump_limits",
    "evaluate_row_apply_policy",
    "record_row_apply_decision",
    "run_kr_general_post_timing_adjustments",
    "try_append_kr_br_alias_row",
    "build_common_mapping_runtime_payload",
    "compute_runtime_low_conf_state",
    "format_alignment_guard_summary",
    "format_mapping_reason_schema_summary",
    "format_mapping_summary",
    "build_ja_mapping_reason_log",
    "build_kr_mapping_reason_log",
    "build_ja_mapping_candidate_trace",
    "build_top_mapping_candidate_rows",
    "maybe_append_cv_mapping_trace",
    "COMMON_MAPPING_REASON_CODES",
    "KR_MAPPING_REASON_CODES",
    "JA_MAPPING_REASON_CODES",
    "canonicalize_mapping_reason_code",
    "get_known_mapping_reason_codes",
    "is_known_mapping_reason_code",
    "normalize_mapping_reason_code",
    "resolve_mapping_reason_schema",
    "build_mapping_failure_meta",
    "build_mapping_abstain_meta",
    "update_ja_mapping_runtime_report",
    "update_mapping_vc_bridge_runtime_report",
    "update_kr_mapping_runtime_report",
    "update_review_queue_runtime_report",
    "update_wav_name_map_from_file_context",
    "normalize_korean_generation_options",
    "normalize_japanese_generation_options",
    "build_common_plan_context",
    "build_language_runtime_state",
    "extract_language_runtime_state",
    "normalize_runtime_postprocess_state",
    "recompute_common_plan_runtime_state",
    "resolve_common_runtime_policy",
    "resolve_mapping_line_alias",
    "resolve_mapping_failure_reason",
    "log_sinsy_plan_guard",
]
