from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List

from core.distribution_guard import is_training_paths_enabled
from core.generator_finish import write_oto_lines

logger = logging.getLogger(__name__)


def persist_korean_generation_output(
    *,
    out_path: str,
    final_lines: List[str],
    tg_folder: str,
    kr_profile,
    custom_map,
    custom_phonemes_path: str,
    enable_ml_correction: bool,
    auto_gen_format: str,
    ml_policy,
    runtime_report,
    log_fn: Callable[[str], None],
    validate_fn,
    normalize_key_fn,
    wav_name_map: Dict[str, str],
    apply_output_wav_name_map_fn: Callable[[str, Dict[str, str]], int],
    oto_encoding: str = "",
    errors: List[str],
) -> None:
    from core.kr_oto_file_finalize import KrPostFilePipelineContext, run_kr_post_file_pipeline

    try:
        written_enc = write_oto_lines(out_path, final_lines, encoding=oto_encoding or None)
        log_fn(f"OTO intermediate save complete: {out_path} (encoding={written_enc})")
        run_kr_post_file_pipeline(
            KrPostFilePipelineContext(
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
                ml_route=os.environ.get("UTOA_ML_ROUTE", "v2"),
            )
        )
        renamed = apply_output_wav_name_map_fn(out_path, wav_name_map)
        if renamed:
            log_fn(f"WAV name mapping applied: {renamed} files")
    except Exception as exc:
        err = f"OTO save/finalize failed: {exc}"
        logger.error(err)
        errors.append(err)


def persist_japanese_generation_output(
    *,
    out_path: str,
    final_lines: List[str],
    profile_path: str,
    had_profile_on_start: bool,
    custom_map,
    log_fn: Callable[[str], None],
    validate_fn,
    errors: List[str],
    tg_folder: str,
    custom_phonemes_path: str,
    enable_ml_correction: bool,
    forced_format: str,
    fallback_format: str,
    ml_policy,
    runtime_report,
    classify_alias_fn,
    oto_encoding: str = "",
) -> None:
    from core.ja_oto_file_finalize import JaPostFilePipelineContext, run_ja_post_file_pipeline

    try:
        written_enc = write_oto_lines(out_path, final_lines, encoding=oto_encoding or None)
        log_fn(f"Japanese OTO intermediate save complete: {out_path} (encoding={written_enc})")
    except Exception as exc:
        err = f"OTO save failed: {exc}"
        logger.error(err)
        errors.append(err)

    if is_training_paths_enabled():
        try:
            from core.ja_oto_autotune import JaAutotuneRefreshContext, refresh_ja_autotune_profile_after_generation

            refresh_ja_autotune_profile_after_generation(
                JaAutotuneRefreshContext(
                    out_path=out_path,
                    profile_path=profile_path,
                    had_profile_on_start=had_profile_on_start,
                    custom_map=custom_map,
                    log_fn=log_fn,
                    validate_fn=validate_fn,
                )
            )
        except Exception as exc:
            log_fn(f"[AutoTune] profile refresh failed: {exc}")
    else:
        log_fn("[AutoTune] distribution build: profile retraining step skipped.")

    run_ja_post_file_pipeline(
        JaPostFilePipelineContext(
            out_path=out_path,
            tg_folder=tg_folder,
            custom_map=custom_map,
            custom_phonemes_path=custom_phonemes_path,
            enable_ml_correction=enable_ml_correction,
            forced_format=forced_format,
            fallback_format=fallback_format,
            ml_policy=ml_policy,
            runtime_report=runtime_report,
            log_fn=log_fn,
            validate_fn=validate_fn,
            classify_alias_fn=classify_alias_fn,
            ml_route=os.environ.get("UTOA_ML_ROUTE", "v2"),
        )
    )
