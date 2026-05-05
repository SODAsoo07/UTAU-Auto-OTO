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
from .output_finalize import (
    persist_japanese_generation_output,
    persist_korean_generation_output,
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
    "build_wav_index",
    "handle_mapping_abstain_fallback",
    "handle_mapping_failure_fallback",
    "finalize_runtime_report",
    "handle_ja_file_context_status",
    "handle_ja_loop_prep_status",
    "handle_kr_file_context_status",
    "handle_kr_loop_prep_status",
    "has_direct_wav_files",
    "initialize_runtime_report",
    "iter_files_with_suffix",
    "iter_textgrid_files",
    "iter_wav_files",
    "list_wav_names",
    "persist_japanese_generation_output",
    "persist_korean_generation_output",
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
