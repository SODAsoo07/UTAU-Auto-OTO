"""WAV-level multiprocessing for the HSMM OTO workflow.

Provides `process_wavs_parallel` which distributes per-WAV compute
(feature extraction, HSMM decoding, anchor assignment) across multiple
processes to utilize all CPU cores.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .evidence_pack import (
    build_acoustic_evidence_pack,
    candidate_priors_from_evidence_pack,
)
from .hsmm_adapter import decode_filename_slots_with_hsmm, template_grid_event_priors
from .oto_adapter import (
    OtoAdapterConfig,
    OtoTemplateRow,
    adapt_template_row,
    anchors_from_prediction,
    assign_template_row_anchors,
    bootstrap_row,
    repair_cvvc_row_sequence,
    timeline_expected_slots_for_template_rows,
)
from .row_plan import (
    RowPlanRecord,
    build_filename_slots,
    build_filename_template_rows,
)
from .runtime_inference import predict_wav
from .types import DecodedEvent


def _default_max_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, 8))


def _process_single_wav(
    *,
    wav_path: str,
    wav_key: str,
    template_group: list | None,
    template_guide_pre_ms: list | tuple = (),
    row_plan_slots: list,
    row_plan_records: list,
    expected_phones: list[str],
    expected_slots: list | None,
    checkpoint_path: str,
    encoder: str | None,
    device: str | None,
    use_slot_viterbi: bool,
    use_hsmm_decoder: bool,
    language: str,
    format_type: str,
    alias_type: str,
    vc_pre_max_ms: float,
    one_step_shift_repair_enabled: bool,
    alias_suffix: str,
    policy_cv_anchor_weight: float,
    policy_cv_posterior_weight: float,
    policy_nucleus_anchor_weight: float,
    policy_nucleus_posterior_weight: float,
    policy_row_anchor_weight: float,
    policy_row_boundary_weight: float,
    policy_boundary_evidence_low_confidence: float,
) -> dict:
    """Process a single WAV file — runs in a worker process.

    Returns a serializable dict with all results needed by the caller.
    """
    from .workflow import (
        WorldV1RuntimePolicy,
        _adapter_config_for_row_plan_record,
        _build_boundary_evidence,
        _build_timeline_debug_record,
        _adapted_row_debug,
        _collect_prediction_metrics,
        _event_source_for_oto,
        _expected_phones,
        _hsmm_event_sequence_matches_expected,
        _hsmm_runtime_replacement_rejection_reason,
        _repair_anchor_monotonicity,
        _row_confidence,
        _runtime_events_for_hsmm_guard,
        _safe_conf,
        _template_group_is_nonphonetic_special_only,
        _template_row_plan_context,
        _wav_duration_ms,
    )
    from core.generation.common.oto_file_utils import apply_alias_suffix

    policy = WorldV1RuntimePolicy(
        cv_anchor_weight=policy_cv_anchor_weight,
        cv_posterior_weight=policy_cv_posterior_weight,
        nucleus_anchor_weight=policy_nucleus_anchor_weight,
        nucleus_posterior_weight=policy_nucleus_posterior_weight,
        row_anchor_weight=policy_row_anchor_weight,
        row_boundary_weight=policy_row_boundary_weight,
        boundary_evidence_low_confidence=policy_boundary_evidence_low_confidence,
    )

    prediction = predict_wav(
        wav_path,
        checkpoint_path=checkpoint_path,
        expected_phones=expected_phones,
        expected_slots=expected_slots,
        encoder=encoder,
        device=device,
        use_slot_viterbi=use_slot_viterbi,
        language=language,
    )
    prediction_events = list(_event_source_for_oto(prediction))
    evidence_pack = build_acoustic_evidence_pack(
        prediction.posterior,
        wav_name=wav_key,
        expected_phones=expected_phones,
        filename_slots=row_plan_slots,
        runtime_events=prediction_events,
    )

    local_warnings: list[str] = []
    slot_scores: list[float] = []
    rule_based_per_wav: list[str] = []
    _collect_prediction_metrics(prediction, slot_scores, local_warnings, rule_based_per_wav)
    slot_warning_count = len(prediction.slot_result.warnings if prediction.slot_result is not None else ())

    decoded_source = list(prediction_events)
    selected_event_source = "runtime_prediction"
    hsmm = None
    hsmm_skip_no_expected = (
        bool(use_hsmm_decoder)
        and bool(row_plan_slots)
        and not bool(expected_slots)
        and (template_group is not None and _template_group_is_nonphonetic_special_only(template_group))
    )

    if use_hsmm_decoder and row_plan_slots and (expected_slots or not hsmm_skip_no_expected):
        candidate_priors = candidate_priors_from_evidence_pack(evidence_pack, row_plan_slots)
        # Guide-grid priors are calibrated for Korean VCV guide-BGM recordings;
        # keep other language/format combos untouched until measured.
        grid_priors: tuple = ()
        if str(language or "").strip().lower().startswith("ko") and str(format_type or "").strip().lower() == "vcv":
            grid_priors = template_grid_event_priors(
                prediction.posterior,
                row_plan_slots,
                template_guide_pre_ms or (),
            )
        hsmm = decode_filename_slots_with_hsmm(
            prediction.posterior,
            row_plan_slots,
            event_priors=(*prediction_events, *candidate_priors, *grid_priors),
            language=language,
        )
        if hsmm.ok and hsmm.events:
            hsmm_reject_reason = ""
            hsmm_sequence_ok = _hsmm_event_sequence_matches_expected(
                expected_slots or (),
                hsmm.events,
                runtime_events=_runtime_events_for_hsmm_guard(prediction, prediction_events),
            )
            if hsmm_sequence_ok:
                hsmm_reject_reason = _hsmm_runtime_replacement_rejection_reason(
                    expected_slots or (),
                    prediction_events,
                    language=language,
                    format_type=format_type,
                    template_rows=template_group or [],
                )
            if hsmm_sequence_ok and not hsmm_reject_reason:
                decoded_source = list(hsmm.events)
                selected_event_source = "filename_hsmm"
                local_warnings.append("filename_hsmm_decoder_used")
            else:
                local_warnings.append(
                    f"filename_hsmm_decoder_rejected:{hsmm_reject_reason or 'event_sequence_mismatch'}"
                )
        else:
            local_warnings.append(f"filename_hsmm_decoder_failed:{hsmm.result.reason}")
    elif hsmm_skip_no_expected:
        local_warnings.append(f"filename_hsmm_decoder_skipped:no_expected_slots:{wav_key}")
    elif use_hsmm_decoder:
        local_warnings.append(f"filename_hsmm_decoder_skipped:no_filename_slots:{wav_key}")

    file_duration_ms = _wav_duration_ms(wav_path, local_warnings)
    adapter_config = OtoAdapterConfig(
        mode="template-preserve" if template_group else "bootstrap",
        language=language,
        format_type=format_type,
        alias_type=alias_type,
        vc_pre_max_ms=vc_pre_max_ms,
    )

    adapted_rows = []
    row_anchors_list = []
    if template_group:
        row_anchors_list = assign_template_row_anchors(
            prediction.posterior,
            decoded_source,
            template_group,
            min_score=0.02,
            use_source_timing_prior=False,
            expected_phones=expected_phones,
            language=language,
        )
        if one_step_shift_repair_enabled:
            row_anchors_list = _repair_anchor_monotonicity(row_anchors_list)
        for row_idx, (template_row, anchor) in enumerate(zip(template_group, row_anchors_list)):
            adapted_rows.append(
                adapt_template_row(
                    template_row,
                    anchor,
                    file_duration_ms=file_duration_ms,
                    config=_adapter_config_for_row_plan_record(
                        adapter_config,
                        row_plan_records[row_idx] if row_idx < len(row_plan_records) else None,
                    ),
                )
            )
    else:
        anchors = anchors_from_prediction(prediction.posterior, decoded_source)
        adapted_rows.append(
            bootstrap_row(
                wav_key,
                Path(wav_key).stem,
                anchors[0] if anchors else None,
                file_duration_ms=file_duration_ms,
                config=OtoAdapterConfig(
                    mode="bootstrap",
                    language=language,
                    format_type=format_type,
                    alias_type=alias_type,
                    vc_pre_max_ms=vc_pre_max_ms,
                ),
            )
        )

    adapted_rows = repair_cvvc_row_sequence(
        adapted_rows,
        adapter_config,
        file_duration_ms=file_duration_ms,
        posterior=prediction.posterior,
    )

    row_results = []
    oto_lines = []
    anchor_scores = []
    cv_conf_values = []
    nucleus_conf_values = []
    row_anchor_hits = 0

    for adapted in adapted_rows:
        if adapted.anchor is not None:
            row_anchor_hits += 1
            anchor_scores.append(float(adapted.anchor.score))
        evidence = _build_boundary_evidence(prediction.posterior, adapted, policy=policy)
        row_conf = _row_confidence(adapted, evidence, policy=policy)
        cv_conf_values.append(_safe_conf(evidence.cv_confidence if evidence.cv_boundary is not None else 0.0))
        nucleus_conf_values.append(_safe_conf(evidence.nucleus_confidence if evidence.nucleus is not None else 0.0))
        row_results.append({
            "wav_key": wav_key.lower() if not template_group else wav_key,
            "alias": str(adapted.alias or ""),
            "oto_params": {
                "offset": float(adapted.timing.offset),
                "consonant": float(adapted.timing.consonant),
                "cutoff": float(adapted.timing.cutoff),
                "preutterance": float(adapted.timing.preutterance),
                "overlap": float(adapted.timing.overlap),
            },
            "confidence": row_conf,
            "cv_confidence": evidence.cv_confidence if evidence.cv_boundary is not None else 0.0,
            "nucleus_confidence": evidence.nucleus_confidence if evidence.nucleus is not None else 0.0,
            "warnings": list(dict.fromkeys((*adapted.warnings, *evidence.warnings))),
        })
        oto_lines.append(apply_alias_suffix(adapted.format_line(), alias_suffix))

    timeline_record = _build_timeline_debug_record(
        wav_name=wav_key,
        wav_path=wav_path,
        use_hsmm_decoder_requested=bool(use_hsmm_decoder),
        selected_event_source=selected_event_source,
        expected_phones=expected_phones,
        filename_slots=row_plan_slots if use_hsmm_decoder else (),
        expected_slots=expected_slots or (),
        prediction=prediction,
        prediction_events=prediction_events,
        selected_events=decoded_source,
        hsmm=hsmm,
        adapted_rows=[
            _adapted_row_debug(
                adapted,
                row_index=row_idx,
                row_plan_record=(
                    _template_row_plan_context(
                        (template_group[row_idx] if template_group else None),
                        row_idx,
                        anchor=adapted.anchor,
                        filename_slots=row_plan_slots,
                    )
                    if template_group
                    else (row_plan_records[row_idx] if row_idx < len(row_plan_records) else None)
                ),
            )
            for row_idx, adapted in enumerate(adapted_rows)
        ],
        evidence=evidence_pack,
    )

    return {
        "wav_key": wav_key,
        "row_results": row_results,
        "oto_lines": oto_lines,
        "warnings": local_warnings,
        "slot_scores": slot_scores,
        "slot_warning_count": slot_warning_count,
        "anchor_scores": anchor_scores,
        "cv_conf_values": cv_conf_values,
        "nucleus_conf_values": nucleus_conf_values,
        "row_anchor_hits": row_anchor_hits,
        "rows_total": len(adapted_rows),
        "rule_based_per_wav": rule_based_per_wav,
        "timeline_record": timeline_record,
        "evidence_pack": evidence_pack,
    }


def process_wavs_parallel(
    wav_tasks: list[dict],
    *,
    max_workers: int | None = None,
    callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Process multiple WAV tasks in parallel using ProcessPoolExecutor.

    Args:
        wav_tasks: List of kwarg dicts for _process_single_wav.
        max_workers: Number of worker processes. Defaults to cpu_count - 1.
        callback: Progress callback (called from main process).

    Returns:
        List of result dicts in the same order as wav_tasks.
    """
    workers = max_workers if max_workers is not None else _default_max_workers()
    total = len(wav_tasks)

    if workers <= 1 or total <= 2:
        results = []
        for i, task in enumerate(wav_tasks):
            if callback:
                callback(f"[No-MFA/MFA-Free] wav {i+1}/{total}: {task['wav_key']}")
            results.append(_process_single_wav(**task))
        return results

    if callback:
        callback(f"[No-MFA/MFA-Free] parallel processing: {total} WAVs × {workers} workers")

    results: list[dict | None] = [None] * total
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for i, task in enumerate(wav_tasks):
            future = executor.submit(_process_single_wav, **task)
            future_to_index[future] = i

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            completed += 1
            try:
                results[idx] = future.result()
            except Exception as exc:
                if callback:
                    callback(f"[No-MFA/MFA-Free] worker error on {wav_tasks[idx]['wav_key']}: {exc}")
                results[idx] = {
                    "wav_key": wav_tasks[idx]["wav_key"],
                    "row_results": [],
                    "oto_lines": [],
                    "warnings": [f"parallel_worker_error:{exc}"],
                    "slot_scores": [],
                    "slot_warning_count": 0,
                    "anchor_scores": [],
                    "cv_conf_values": [],
                    "nucleus_conf_values": [],
                    "row_anchor_hits": 0,
                    "rows_total": 0,
                    "rule_based_per_wav": [],
                    "timeline_record": {},
                    "evidence_pack": {},
                }
            if callback and (completed % 5 == 0 or completed == total):
                callback(f"[No-MFA/MFA-Free] progress {completed}/{total}")

    return results  # type: ignore[return-value]
