from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, Dict, List

from core.generator_finish import write_oto_lines

logger = logging.getLogger(__name__)


def persist_kr_generation_output(
    *,
    out_path: str,
    final_lines: Sequence[str],
    tg_folder: str,
    kr_profile: Any,
    custom_map: Any,
    custom_phonemes_path: str,
    enable_ml_correction: bool,
    auto_gen_format: str,
    ml_policy: Any,
    runtime_report: Any,
    log_fn: Callable[[str], None],
    validate_fn: Callable[..., Any],
    normalize_key_fn: Callable[..., Any],
    wav_name_map: Dict[str, str],
    apply_output_wav_name_map_fn: Callable[[str, Dict[str, str]], int],
    errors: List[str],
    write_oto_lines_fn: Callable[[str, Sequence[str]], None] = write_oto_lines,
    run_kr_post_file_pipeline_fn: Callable[[Any], None] | None = None,
    post_file_context_factory: Callable[..., Any] | None = None,
) -> int:
    try:
        if run_kr_post_file_pipeline_fn is None or post_file_context_factory is None:
            from core.kr_oto_file_finalize import KrPostFilePipelineContext, run_kr_post_file_pipeline

            if run_kr_post_file_pipeline_fn is None:
                run_kr_post_file_pipeline_fn = run_kr_post_file_pipeline
            if post_file_context_factory is None:
                post_file_context_factory = KrPostFilePipelineContext

        write_oto_lines_fn(out_path, final_lines)
        log_fn(f"1차 생성 완료: OTO 저장 -> {out_path}")
        run_kr_post_file_pipeline_fn(
            post_file_context_factory(
                out_path=out_path,
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
                ml_route=os.environ.get("UTOA_ML_ROUTE", "legacy"),
            )
        )
        renamed = apply_output_wav_name_map_fn(out_path, wav_name_map)
        if renamed:
            log_fn(f"WAV 이름 매핑 적용: {renamed}개")
        return int(renamed or 0)
    except Exception as exc:
        err = f"OTO 저장 실패: {exc}"
        logger.error(err)
        errors.append(err)
        return 0


__all__ = ["persist_kr_generation_output"]
