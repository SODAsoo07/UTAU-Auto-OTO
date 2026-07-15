from __future__ import annotations

import os
import math
import re
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence

from core.generation.common.oto_alias_family import alias_family
from core.generation.common.oto_file_utils import apply_alias_suffix
from .evidence_pack import (
    build_acoustic_evidence_pack,
    candidate_priors_from_evidence_pack,
    summarize_event_candidate_reliability,
)
from .hsmm_adapter import decode_filename_slots_with_hsmm
from .manifest_audit import infer_filename_phone_sequence
from .oto_adapter import (
    OtoAdapterConfig,
    OtoTemplateRow,
    _assign_alias_target_indices,
    adapt_template_row,
    anchors_from_prediction,
    assign_template_row_anchors,
    bootstrap_row,
    _alias_phone_sequence,
    _alias_type_for_row,
    _is_nonphonetic_special_alias,
    _is_japanese_language_name,
    _is_korean_language_name,
    _phone_matches,
    _phone_sequence_variants_match,
    load_oto_template_rows_alias_only,
    repair_cvvc_row_sequence,
    timeline_expected_slots_for_template_rows,
)
from .row_plan import (
    RowPlanRecord,
    build_filename_row_plan,
    build_filename_slots,
    build_filename_template_rows,
    filename_phone_sequence_from_slots,
)
from .runtime_inference import RuntimePrediction, predict_wav

# Weights for the aggregate NoMfaWorkflowReport.confidence score. Must sum to 1.0.
_CONF_WEIGHT_ANCHOR = 0.30
_CONF_WEIGHT_SLOT = 0.28
_CONF_WEIGHT_CV_BOUNDARY = 0.22
_CONF_WEIGHT_NUCLEUS = 0.20
# Slot warnings beyond _CONF_SLOT_WARNING_FREE each subtract _CONF_SLOT_WARNING_PENALTY.
_CONF_SLOT_WARNING_FREE = 2
_CONF_SLOT_WARNING_PENALTY = 0.02

# Boundary-evidence repair offsets (ms).
_C_ONSET_LEAD_MS = 18.0  # consonant onset placed this far before the vowel start
_C_ONSET_REPAIR_MS = 6.0  # fallback gap when c_onset lands after the cv boundary
_V_OFFSET_REPAIR_MS = 8.0  # fallback gap when v_offset lands before the cv boundary
TIMELINE_DEBUG_SCHEMA_VERSION = 1
JA_CVVC_HSMM_RUNTIME_ALIGNED_FIRST_GAP_MS = 240.0


@dataclass(frozen=True)
class BoundaryEvidence:
    c_onset: float | None
    cv_boundary: float | None
    v_offset: float | None
    nucleus: float | None
    cv_confidence: float
    nucleus_confidence: float
    confidence: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoMfaRowResult:
    wav_key: str
    alias: str
    oto_params: Mapping[str, float]
    confidence: float
    boundary_evidence: BoundaryEvidence
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoMfaWorkflowGuard:
    min_confidence: float = 0.52
    min_anchor_ratio: float = 0.65
    max_slot_warnings: int = 6
    cv_boundary_min_conf: float = 0.32
    vowel_nucleus_min_conf: float = 0.34
    vc_pre_max_ms: float = 80.0
    one_step_shift_repair_enabled: bool = True


@dataclass(frozen=True)
class WorldV1RuntimePolicy:
    cv_anchor_weight: float = 0.62
    cv_posterior_weight: float = 0.38
    nucleus_anchor_weight: float = 0.58
    nucleus_posterior_weight: float = 0.42
    row_anchor_weight: float = 0.55
    row_boundary_weight: float = 0.45
    boundary_evidence_low_confidence: float = 0.35
    cv_boundary_min_conf: float | None = None
    vowel_nucleus_min_conf: float | None = None
    vc_pre_max_ms: float | None = None
    one_step_shift_repair_enabled: bool | None = None


@dataclass(frozen=True)
class NoMfaWorkflowReport:
    processed: int
    total: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    confidence: float
    fallback_hint: str
    mode: str
    guard_failed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    rows: tuple[NoMfaRowResult, ...] = ()
    timeline_debug: tuple[Mapping[str, object], ...] = ()
    evidence_debug: tuple[Mapping[str, object], ...] = ()


def generate_no_mfa_oto_with_model_context(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str = "",
    alias_suffix: str = "",
    checkpoint_path: str = "",
    language: str = "japanese",
    format_type: str = "CV",
    alias_type: str = "auto",
    encoder: str | None = None,
    device: str | None = None,
    use_slot_viterbi: bool = True,
    use_hsmm_decoder: bool = False,
    guard: NoMfaWorkflowGuard | None = None,
    runtime_policy: WorldV1RuntimePolicy | None = None,
    callback: Callable[[str], None] | None = None,
    max_workers: int | None = None,
) -> NoMfaWorkflowReport:
    cfg = guard or NoMfaWorkflowGuard()
    policy = runtime_policy or WorldV1RuntimePolicy()
    cv_boundary_min_conf = (
        float(policy.cv_boundary_min_conf)
        if policy.cv_boundary_min_conf is not None
        else float(cfg.cv_boundary_min_conf)
    )
    nucleus_min_conf = (
        float(policy.vowel_nucleus_min_conf)
        if policy.vowel_nucleus_min_conf is not None
        else float(cfg.vowel_nucleus_min_conf)
    )
    vc_pre_max_ms = float(policy.vc_pre_max_ms) if policy.vc_pre_max_ms is not None else float(cfg.vc_pre_max_ms)
    one_step_shift_repair_enabled = (
        bool(policy.one_step_shift_repair_enabled)
        if policy.one_step_shift_repair_enabled is not None
        else bool(cfg.one_step_shift_repair_enabled)
    )
    wav_root = os.path.abspath(str(wav_dir or "").strip())
    if not os.path.isdir(wav_root):
        return NoMfaWorkflowReport(
            processed=0,
            total=0,
            errors=(f"wav_dir_not_found:{wav_root}",),
            warnings=(),
            confidence=0.0,
            fallback_hint="manual_review_required",
            mode="mfa_free_ssl_slot",
            guard_failed=True,
        )

    source_oto = str(source_oto_path or "").strip()
    if source_oto and not os.path.isfile(source_oto):
        source_oto = ""

    rows_total = 0
    row_anchor_hits = 0
    all_lines: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    anchor_scores: list[float] = []
    slot_scores: list[float] = []
    slot_warning_count = 0
    row_results: list[NoMfaRowResult] = []
    cv_conf_values: list[float] = []
    nucleus_conf_values: list[float] = []
    rule_based_per_wav: list[str] = []
    timeline_debug: list[dict[str, object]] = []
    evidence_debug: list[dict[str, object]] = []

    if source_oto:
        warnings.append("base_oto_alias_list_used")
        warnings.append("base_oto_alias_surface_used")
        warnings.append("base_oto_alias_order_ignored")
        template_rows = load_oto_template_rows_alias_only(source_oto)
        templates_by_wav: dict[str, list] = {}
        for row in template_rows:
            templates_by_wav.setdefault(str(row.wav).lower(), []).append(row)

        # Guide-grid prior input: the alias-only rows above have their timings
        # zeroed by design, so load the raw rows once more to extract each
        # wav's guide-time preutterance grid (used only as a weak HSMM prior).
        from .hsmm_adapter import template_mora_pre_grid
        from .oto_adapter import load_oto_template_rows as _load_full_template_rows

        full_rows_by_wav: dict[str, list] = {}
        for row in _load_full_template_rows(source_oto):
            full_rows_by_wav.setdefault(str(row.wav).lower(), []).append(row)
        guide_pre_by_wav: dict[str, list[float]] = {
            wav_name: template_mora_pre_grid(rows)
            for wav_name, rows in full_rows_by_wav.items()
        }

        # Index actual wavs by a separator-normalised name so a base OTO that
        # references reclist apostrophes (ga'gi'gu.wav) still matches recordings
        # saved with underscores (ga_gi_gu.wav). Without this the mismatched wavs
        # are dropped from the output entirely.
        actual_wavs_by_norm: dict[str, str] = {}
        try:
            for _name in os.listdir(wav_root):
                if _name.lower().endswith(".wav"):
                    actual_wavs_by_norm.setdefault(_normalize_wav_name(_name), _name)
        except Exception:
            actual_wavs_by_norm = {}

        from .parallel import process_wavs_parallel

        total_wav_groups = len(templates_by_wav)
        wav_tasks = []
        wav_task_keys: list[str] = []
        for raw_wav_key in sorted(templates_by_wav):
            wav_key = raw_wav_key
            template_group = _template_group_in_filename_order(
                raw_wav_key,
                templates_by_wav[raw_wav_key],
                language=language,
                format_type=format_type,
            )
            wav_path = os.path.join(wav_root, wav_key)
            if not os.path.isfile(wav_path):
                resolved = actual_wavs_by_norm.get(_normalize_wav_name(raw_wav_key))
                if resolved and os.path.isfile(os.path.join(wav_root, resolved)):
                    wav_key = resolved
                    wav_path = os.path.join(wav_root, resolved)
                    template_group = [replace(r, wav=resolved) for r in template_group]
                    warnings.append(f"wav_name_normalized:{raw_wav_key}->{resolved}")
                else:
                    warnings.append(f"missing_wav:{raw_wav_key}")
                    continue

            row_plan_slots = build_filename_slots(wav_key, language=language, format_type=format_type)
            expected_phones = _expected_phones(wav_key, [], filename_slots=row_plan_slots)
            expected_slots = timeline_expected_slots_for_template_rows(
                template_group, expected_phones, language=language,
            )
            wav_task_keys.append(wav_key)
            wav_tasks.append({
                "wav_path": wav_path,
                "wav_key": wav_key,
                "template_group": template_group,
                "template_guide_pre_ms": list(guide_pre_by_wav.get(raw_wav_key, [])),
                "row_plan_slots": row_plan_slots,
                "row_plan_records": [],
                "expected_phones": expected_phones,
                "expected_slots": expected_slots,
                "checkpoint_path": checkpoint_path,
                "encoder": encoder,
                "device": device,
                "use_slot_viterbi": use_slot_viterbi,
                "use_hsmm_decoder": use_hsmm_decoder,
                "language": language,
                "format_type": format_type,
                "alias_type": alias_type,
                "vc_pre_max_ms": vc_pre_max_ms,
                "one_step_shift_repair_enabled": one_step_shift_repair_enabled,
                "alias_suffix": alias_suffix,
                "policy_cv_anchor_weight": policy.cv_anchor_weight,
                "policy_cv_posterior_weight": policy.cv_posterior_weight,
                "policy_nucleus_anchor_weight": policy.nucleus_anchor_weight,
                "policy_nucleus_posterior_weight": policy.nucleus_posterior_weight,
                "policy_row_anchor_weight": policy.row_anchor_weight,
                "policy_row_boundary_weight": policy.row_boundary_weight,
                "policy_boundary_evidence_low_confidence": policy.boundary_evidence_low_confidence,
            })

        parallel_results = process_wavs_parallel(
            wav_tasks,
            max_workers=max_workers,
            callback=callback,
        )

        source_ordered_lines: list[tuple[int, int, str]] = []
        source_ordered_results: list[tuple[int, int, NoMfaRowResult]] = []
        for result in parallel_results:
            if not result:
                continue
            warnings.extend(result["warnings"])
            slot_scores.extend(result["slot_scores"])
            slot_warning_count += result["slot_warning_count"]
            anchor_scores.extend(result["anchor_scores"])
            cv_conf_values.extend(result["cv_conf_values"])
            nucleus_conf_values.extend(result["nucleus_conf_values"])
            row_anchor_hits += result["row_anchor_hits"]
            rule_based_per_wav.extend(result["rule_based_per_wav"])
            if result.get("timeline_record"):
                timeline_debug.append(result["timeline_record"])
            if result.get("evidence_pack"):
                evidence_debug.append(result["evidence_pack"])
            for row_data in result["row_results"]:
                rows_total += 1
                source_order = rows_total
                source_sequence = len(source_ordered_lines)
                source_ordered_results.append(
                    (
                        source_order,
                        source_sequence,
                        NoMfaRowResult(
                            wav_key=row_data["wav_key"],
                            alias=row_data["alias"],
                            oto_params=row_data["oto_params"],
                            confidence=row_data["confidence"],
                            boundary_evidence=BoundaryEvidence(
                                c_onset=None,
                                cv_boundary=None,
                                v_offset=None,
                                nucleus=None,
                                cv_confidence=row_data.get("cv_confidence", 0.0),
                                nucleus_confidence=row_data.get("nucleus_confidence", 0.0),
                                confidence=row_data["confidence"],
                                warnings=tuple(row_data.get("warnings", ())),
                            ),
                            warnings=tuple(row_data.get("warnings", ())),
                        ),
                    )
                )
            for line in result["oto_lines"]:
                source_ordered_lines.append((rows_total, len(source_ordered_lines), line))

        row_results.extend(row for _source_order, _source_sequence, row in sorted(source_ordered_results))
        all_lines.extend(line for _source_order, _source_sequence, line in sorted(source_ordered_lines))
    else:
        from .parallel import process_wavs_parallel

        wav_files = sorted(Path(wav_root).glob("*.wav"))
        total_wav_files = len(wav_files)

        wav_tasks = []
        all_row_plan_records: list[list] = []
        for wav_path in wav_files:
            template_group, row_plan_phones, row_plan_records_local = build_filename_template_rows(
                wav_path.name,
                language=language,
                format_type=format_type,
            )
            row_plan_slots = build_filename_slots(wav_path.name, language=language, format_type=format_type)
            expected_phones = list(row_plan_phones or _expected_phones(wav_path.name, [], filename_slots=row_plan_slots))
            expected_slots = (
                timeline_expected_slots_for_template_rows(template_group, expected_phones, language=language)
                if template_group and expected_phones
                else None
            )
            all_row_plan_records.append(list(row_plan_records_local))
            wav_tasks.append({
                "wav_path": str(wav_path),
                "wav_key": wav_path.name,
                "template_group": template_group if template_group else None,
                "row_plan_slots": row_plan_slots,
                "row_plan_records": list(row_plan_records_local),
                "expected_phones": expected_phones,
                "expected_slots": expected_slots,
                "checkpoint_path": checkpoint_path,
                "encoder": encoder,
                "device": device,
                "use_slot_viterbi": use_slot_viterbi,
                "use_hsmm_decoder": use_hsmm_decoder,
                "language": language,
                "format_type": format_type,
                "alias_type": alias_type,
                "vc_pre_max_ms": vc_pre_max_ms,
                "one_step_shift_repair_enabled": one_step_shift_repair_enabled,
                "alias_suffix": alias_suffix,
                "policy_cv_anchor_weight": policy.cv_anchor_weight,
                "policy_cv_posterior_weight": policy.cv_posterior_weight,
                "policy_nucleus_anchor_weight": policy.nucleus_anchor_weight,
                "policy_nucleus_posterior_weight": policy.nucleus_posterior_weight,
                "policy_row_anchor_weight": policy.row_anchor_weight,
                "policy_row_boundary_weight": policy.row_boundary_weight,
                "policy_boundary_evidence_low_confidence": policy.boundary_evidence_low_confidence,
            })

        parallel_results = process_wavs_parallel(
            wav_tasks,
            max_workers=max_workers,
            callback=callback,
        )

        for result in parallel_results:
            if not result:
                continue
            warnings.extend(result["warnings"])
            slot_scores.extend(result["slot_scores"])
            slot_warning_count += result["slot_warning_count"]
            anchor_scores.extend(result["anchor_scores"])
            cv_conf_values.extend(result["cv_conf_values"])
            nucleus_conf_values.extend(result["nucleus_conf_values"])
            row_anchor_hits += result["row_anchor_hits"]
            rows_total += result["rows_total"]
            rule_based_per_wav.extend(result["rule_based_per_wav"])
            all_lines.extend(result["oto_lines"])
            if result.get("timeline_record"):
                timeline_debug.append(result["timeline_record"])
            if result.get("evidence_pack"):
                evidence_debug.append(result["evidence_pack"])
            for row_data in result["row_results"]:
                row_results.append(
                    NoMfaRowResult(
                        wav_key=row_data["wav_key"],
                        alias=row_data["alias"],
                        oto_params=row_data["oto_params"],
                        confidence=row_data["confidence"],
                        boundary_evidence=BoundaryEvidence(
                            c_onset=None,
                            cv_boundary=None,
                            v_offset=None,
                            nucleus=None,
                            cv_confidence=row_data.get("cv_confidence", 0.0),
                            nucleus_confidence=row_data.get("nucleus_confidence", 0.0),
                            confidence=row_data["confidence"],
                            warnings=tuple(row_data.get("warnings", ())),
                        ),
                        warnings=tuple(row_data.get("warnings", ())),
                    )
                )

    if rows_total <= 0 or not all_lines:
        errors.append("no_rows_generated")
        return NoMfaWorkflowReport(
            processed=0,
            total=max(rows_total, 0),
            errors=tuple(errors),
            warnings=tuple(warnings),
            confidence=0.0,
            fallback_hint="manual_review_required",
            mode="mfa_free_ssl_slot",
            guard_failed=True,
        )

    out_file = _normalize_out_path(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(all_lines).rstrip() + "\n")

    anchor_ratio = float(row_anchor_hits) / float(max(1, rows_total))
    avg_anchor = float(mean(anchor_scores)) if anchor_scores else 0.0
    avg_slot = float(mean(slot_scores)) if slot_scores else 0.0
    avg_cv_conf = float(mean(cv_conf_values)) if cv_conf_values else 0.0
    avg_nucleus_conf = float(mean(nucleus_conf_values)) if nucleus_conf_values else 0.0
    confidence = max(
        0.0,
        min(
            1.0,
            (_CONF_WEIGHT_ANCHOR * avg_anchor)
            + (_CONF_WEIGHT_SLOT * avg_slot)
            + (_CONF_WEIGHT_CV_BOUNDARY * avg_cv_conf)
            + (_CONF_WEIGHT_NUCLEUS * avg_nucleus_conf)
            - (_CONF_SLOT_WARNING_PENALTY * float(max(0, slot_warning_count - _CONF_SLOT_WARNING_FREE))),
        ),
    )
    guard_reasons: list[str] = []
    if confidence < float(cfg.min_confidence):
        guard_reasons.append(f"low_confidence:{confidence:.3f}")
    if anchor_ratio < float(cfg.min_anchor_ratio):
        guard_reasons.append(f"low_anchor_ratio:{anchor_ratio:.3f}")
    if slot_warning_count > int(cfg.max_slot_warnings):
        guard_reasons.append(f"slot_warning_overflow:{slot_warning_count}")
    if avg_cv_conf < cv_boundary_min_conf:
        guard_reasons.append(f"cv_boundary_confidence_low:{avg_cv_conf:.3f}")
    if avg_nucleus_conf < nucleus_min_conf:
        guard_reasons.append(f"vowel_nucleus_confidence_low:{avg_nucleus_conf:.3f}")

    predicted_wavs = len(rule_based_per_wav)
    rule_based_count = sum(1 for reason in rule_based_per_wav if reason)
    rule_based_ratio = float(rule_based_count) / float(max(1, predicted_wavs))
    inference_failures = sorted(
        {reason for reason in rule_based_per_wav if reason.startswith("checkpoint_inference_failed")}
    )
    if rule_based_count:
        warnings.append(f"rule_based_inference:{rule_based_count}/{predicted_wavs}")
    if inference_failures:
        # A checkpoint was configured but failed to load/run: the run silently
        # degraded to the lower-quality rule-based path, so fail the guard.
        guard_reasons.append(f"checkpoint_inference_failed:{rule_based_count}/{predicted_wavs}")
        warnings.extend(inference_failures)

    guard_failed = bool(guard_reasons)
    if guard_failed:
        warnings.extend(guard_reasons)
    if callback:
        callback(
            "[No-MFA/MFA-Free] "
            f"rows={rows_total} anchor_ratio={anchor_ratio:.2f} conf={confidence:.2f} "
            f"cv_conf={avg_cv_conf:.2f} nucleus_conf={avg_nucleus_conf:.2f} "
            f"slot_warn={slot_warning_count} guard={'fail' if guard_failed else 'pass'}"
        )
    return NoMfaWorkflowReport(
        processed=rows_total,
        total=rows_total,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=confidence,
        fallback_hint="manual_review_required",
        mode="mfa_free_ssl_slot",
        guard_failed=guard_failed,
        metrics={
            "anchor_ratio": float(anchor_ratio),
            "slot_warning_count": float(slot_warning_count),
            "avg_anchor_score": float(avg_anchor),
            "avg_slot_score": float(avg_slot),
            "cv_boundary_confidence": float(avg_cv_conf),
            "vowel_nucleus_confidence": float(avg_nucleus_conf),
            "rule_based_ratio": float(rule_based_ratio),
            "acoustic_feature_set_world_v1": 1.0,
            "runtime_policy_cv_anchor_weight": float(policy.cv_anchor_weight),
            "runtime_policy_cv_posterior_weight": float(policy.cv_posterior_weight),
            "runtime_policy_nucleus_anchor_weight": float(policy.nucleus_anchor_weight),
            "runtime_policy_nucleus_posterior_weight": float(policy.nucleus_posterior_weight),
            "runtime_policy_row_anchor_weight": float(policy.row_anchor_weight),
            "runtime_policy_row_boundary_weight": float(policy.row_boundary_weight),
        },
        rows=tuple(row_results),
        timeline_debug=tuple(timeline_debug),
        evidence_debug=tuple(evidence_debug),
    )


def _template_group_in_filename_order(
    wav_key: str,
    template_group: Sequence[OtoTemplateRow],
    *,
    language: str = "",
    format_type: str = "",
) -> list[OtoTemplateRow]:
    rows = list(template_group)
    if not rows:
        return []

    korean_vcv = (
        _is_korean_language_name(language)
        and str(format_type or "").strip().lower() == "vcv"
    )
    records = build_filename_row_plan(wav_key, language=language, format_type=format_type)
    if korean_vcv:
        head_positions = [
            index
            for index, row in enumerate(rows)
            if _alias_type_for_row(row.alias, "auto") == "cv_head"
        ]
        multi_phrase_head_targets = _korean_vcv_multi_phrase_head_targets(wav_key, rows, records)
        if multi_phrase_head_targets:
            rows = [
                replace(row, expected_phone_indices=multi_phrase_head_targets[id(row)])
                if id(row) in multi_phrase_head_targets
                else row
                for row in rows
            ]
        # A few banks register two complete alias dialects for the same WAV
        # (for example kk* and gg*). Each dialect starts with its own head row.
        # Match each block against a fresh copy of the row plan so the second
        # dialect does not consume only the leftovers from the first.
        if len(head_positions) > 1 and head_positions[0] == 0 and not multi_phrase_head_targets:
            boundaries = [*head_positions, len(rows)]
            split_ordered: list[OtoTemplateRow] = []
            for start, end in zip(boundaries, boundaries[1:]):
                split_ordered.extend(
                    _template_group_in_filename_order(
                        wav_key,
                        rows[start:end],
                        language=language,
                        format_type=format_type,
                    )
                )
            # Interleave dialect rows by target slot. Concatenating complete
            # dialect blocks would reset expected indices from the file end to
            # the head, which a monotonic slot decoder can only satisfy by
            # pushing the second dialect several seconds late.
            return sorted(
                split_ordered,
                key=lambda row: (
                    int(row.expected_phone_indices[-1])
                    if row.expected_phone_indices
                    else 10**9,
                    int(getattr(row, "source_row_index", -1)),
                ),
            )

    if not records:
        return sorted(rows, key=_template_row_semantic_sort_key)

    target_hints = _template_source_target_hints(
        wav_key,
        rows,
        language=language,
        format_type=format_type,
    )
    if _is_korean_language_name(language) and str(format_type or "").strip().lower() == "vcv":
        target_hints.update(_korean_vcv_expanded_filename_target_hints(wav_key, rows, records))
    remaining = list(rows)
    ordered: list[OtoTemplateRow] = []
    for record in records:
        match_index = _matching_template_row_index(
            remaining,
            record,
            records,
            language=language,
            target_hints=target_hints,
        )
        if match_index is None:
            continue
        matched = remaining.pop(match_index)
        if record.expected_phone_indices and matched.expected_phone_indices is None:
            matched = replace(matched, expected_phone_indices=record.expected_phone_indices)
        ordered.append(matched)
    # A common Korean VCV convention duplicates the initial CV as both
    # ``- nyeo`` and ``nyeo``. The filename row plan has only one head record,
    # so a duplicate stored later in the OTO block would otherwise attach to a
    # repeated final syllable. Bind an adjacent candidate, or a unique
    # phone-identical candidate, to the same first-slot anchor.
    if records and remaining and _is_korean_language_name(language):
        head_record = records[0]
        head_row = next(
            (
                row
                for row in [*ordered, *remaining]
                if _alias_type_for_row(row.alias, "auto") == "cv_head"
            ),
            None,
        )
        if head_row is not None:
            head_phones = tuple(_template_alias_phone_sequence(head_row.alias, language=language))
            if head_row.expected_phone_indices is None:
                replacement = replace(head_row, expected_phone_indices=head_record.expected_phone_indices)
                if head_row in ordered:
                    ordered[ordered.index(head_row)] = replacement
                else:
                    remaining[remaining.index(head_row)] = replacement
                head_row = replacement
            matching_duplicate_indices = [
                idx
                for idx, row in enumerate(remaining)
                if _alias_type_for_row(row.alias, "auto") in {"cv", "v"}
                and tuple(_template_alias_phone_sequence(row.alias, language=language)) == head_phones
            ]
            adjacent_duplicate_index = next(
                (
                    idx
                    for idx in matching_duplicate_indices
                    if int(getattr(remaining[idx], "source_row_index", -1))
                    == int(getattr(head_row, "source_row_index", -1)) + 1
                ),
                None,
            )
            # Winnie-style banks place the one plain-CV duplicate at the end of
            # the WAV's OTO block even though it aliases the initial ``- CV``.
            # A unique phone-identical candidate is structurally unambiguous;
            # multiple candidates remain adjacency-only to avoid stealing a
            # genuine later CV registration.
            duplicate_index = (
                adjacent_duplicate_index
                if adjacent_duplicate_index is not None
                else matching_duplicate_indices[0]
                if len(matching_duplicate_indices) == 1
                else None
            )
            if duplicate_index is not None:
                duplicate = remaining.pop(duplicate_index)
                duplicate = replace(duplicate, expected_phone_indices=head_record.expected_phone_indices)
                if head_row in remaining:
                    remaining.remove(head_row)
                    ordered.insert(0, head_row)
                head_index = ordered.index(head_row)
                ordered.insert(head_index + 1, duplicate)
    # CV_head aliases like "- w", "- ka" (consonant-led head) represent the file's
    # first phonetic onset. If one ended up in remaining (no row_plan match) it
    # would be appended at the end, causing anchor assignment to place it near the
    # file end — a catastrophic 3-4 second error. Promote unmatched consonant-led
    # cv_head to position 0. Vowel-only heads ("- あ") are NOT promoted because in
    # VV-chain reclists they don't necessarily correspond to the file start.
    unmatched_heads: list[OtoTemplateRow] = []
    targeted_unmatched_heads: list[OtoTemplateRow] = []
    unmatched_rest: list[OtoTemplateRow] = []
    for row in remaining:
        if (
            _alias_type_for_row(row.alias, "auto") == "cv_head"
            and _cv_head_has_consonant_onset(row.alias)
        ):
            if row.expected_phone_indices:
                targeted_unmatched_heads.append(row)
            else:
                unmatched_heads.append(row)
        else:
            unmatched_rest.append(row)
    if unmatched_rest:
        ordered.extend(sorted(unmatched_rest, key=_template_row_semantic_sort_key))
    if targeted_unmatched_heads:
        ordered.extend(targeted_unmatched_heads)
        ordered = sorted(
            ordered,
            key=lambda row: (
                int(row.expected_phone_indices[-1])
                if row.expected_phone_indices
                else 10**9,
                int(getattr(row, "source_row_index", -1)),
            ),
        )
    if unmatched_heads:
        ordered = unmatched_heads + ordered
    return ordered


def _korean_vcv_multi_phrase_head_targets(
    wav_key: str,
    rows: Sequence[OtoTemplateRow],
    records: Sequence[RowPlanRecord],
) -> dict[int, tuple[int, ...]]:
    """Distinguish multiple recorded phrases from multiple alias dialects."""
    stem = Path(str(wav_key or "")).stem.strip("_")
    if len([part for part in stem.split("_") if part]) <= 1:
        return {}
    heads = [row for row in rows if _alias_type_for_row(row.alias, "auto") == "cv_head"]
    if len(heads) <= 1 or not records:
        return {}

    hints: dict[int, tuple[int, ...]] = {}
    cursor = -1
    for head in heads:
        head_phones = tuple(_template_alias_phone_sequence(head.alias, language="korean"))
        if not head_phones:
            return {}
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for record in records:
            indices = tuple(int(value) for value in record.expected_phone_indices)
            if not indices or indices[-1] <= cursor:
                continue
            record_phones = tuple(_template_alias_phone_sequence(record.alias, language="korean"))
            if len(record_phones) < len(head_phones):
                continue
            suffix = record_phones[-len(head_phones) :]
            if suffix == head_phones or _phone_sequence_variants_match(suffix, head_phones):
                candidates.append((indices[-1], indices))
        if not candidates:
            return {}
        target, indices = min(candidates, key=lambda item: item[0])
        hints[id(head)] = indices
        cursor = target

    if len({indices[-1] for indices in hints.values()}) != len(heads):
        return {}
    return hints


def _korean_vcv_expanded_filename_target_hints(
    wav_key: str,
    rows: Sequence[OtoTemplateRow],
    records: Sequence[RowPlanRecord],
) -> dict[int, int]:
    """Map Korean source transitions monotonically onto filename slots.

    This is also needed for apostrophe-separated reclists.  Their source OTO
    often abbreviates the left glide vowel (``yu -> u``, ``yeo -> eo``) and
    omits every other transition, so phone-only DP can attach an early
    ``u yeo`` row to the repeated final ``i yeo`` slot.
    """

    record_candidates: list[tuple[str, tuple[str, ...], int]] = []
    for record in records:
        role = str(record.role_family or "").strip().lower()
        if role not in {"vcv", "vv"} or not record.expected_phone_indices:
            continue
        alias_tokens = [part for part in str(record.alias or "").split() if part]
        if not alias_tokens:
            continue
        right_phones = tuple(_alias_phone_sequence(alias_tokens[-1]))
        if right_phones:
            record_candidates.append((role, right_phones, int(record.expected_phone_indices[-1])))

    hints: dict[int, int] = {}
    cursor = -1
    for row in rows:
        role = _alias_type_for_row(row.alias, "auto")
        if role not in {"vcv", "vv"}:
            continue
        alias_tokens = [part for part in str(row.alias or "").split() if part]
        if not alias_tokens:
            continue
        right_phones = tuple(_alias_phone_sequence(alias_tokens[-1]))
        candidates = [
            target
            for record_role, record_right, target in record_candidates
            if record_role == role
            and target > cursor
            and (
                right_phones == record_right
                or _phone_sequence_variants_match(right_phones, record_right)
            )
        ]
        if not candidates:
            continue
        target = min(candidates)
        hints[id(row)] = target
        cursor = target

    # Some reclists insert a dialect alias that deliberately does not resemble
    # the literal filename transition (e.g. ``o fyeu`` for the fyo->eu slot).
    # If a contiguous unmatched source run fits exactly into the unused row-plan
    # targets between two reliable neighbors, fill that structural gap in order.
    record_targets = [
        int(record.expected_phone_indices[-1])
        for record in records
        if record.expected_phone_indices
    ]
    used_targets = set(hints.values())
    index = 0
    while index < len(rows):
        if id(rows[index]) in hints:
            index += 1
            continue
        start = index
        while index < len(rows) and id(rows[index]) not in hints:
            index += 1
        end = index
        if start == 0 or end >= len(rows):
            continue
        left_target = hints.get(id(rows[start - 1]))
        right_target = hints.get(id(rows[end]))
        if left_target is None or right_target is None or right_target <= left_target:
            continue
        available = [
            target
            for target in record_targets
            if left_target < target < right_target and target not in used_targets
        ]
        run = rows[start:end]
        if len(available) != len(run):
            continue
        for row, target in zip(run, available):
            hints[id(row)] = target
            used_targets.add(target)
    return hints


def _template_source_target_hints(
    wav_key: str,
    rows: Sequence[OtoTemplateRow],
    *,
    language: str,
    format_type: str,
) -> dict[int, int]:
    slots = build_filename_slots(wav_key, language=language, format_type=format_type)
    expected = filename_phone_sequence_from_slots(slots)
    if not expected:
        return {}
    targets = _assign_alias_target_indices(rows, expected, language=language)
    hints: dict[int, int] = {}
    for row, target in zip(rows, targets):
        if target is None:
            continue
        phones = tuple(_template_alias_phone_sequence(row.alias, language=language))
        target_index = int(target)
        glide_surface = any(str(phone).strip().lower() in {"w", "y", "wa", "wi", "we", "wo", "ya", "yu", "ye", "yo"} for phone in phones)
        standalone_wo_surface = (
            _alias_type_for_row(row.alias, "auto") == "v"
            and phones == ("o",)
            and target_index > 0
            and str(expected[target_index - 1]).strip().lower() == "w"
        )
        if glide_surface or standalone_wo_surface:
            hints[id(row)] = target_index
    return hints


def _matching_template_row_index(
    rows: Sequence[OtoTemplateRow],
    record: RowPlanRecord,
    records: Sequence[RowPlanRecord] = (),
    *,
    language: str = "",
    target_hints: Mapping[int, int] | None = None,
) -> int | None:
    candidates: list[tuple[int, tuple[object, ...], int]] = []
    for idx, row in enumerate(rows):
        score = _template_row_record_match_score(
            row,
            record,
            records,
            language=language,
            preferred_target=(target_hints or {}).get(id(row)),
        )
        if score <= 0:
            continue
        candidates.append((score, _template_row_semantic_sort_key(row), idx))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (-item[0], item[1]))[2]


def _template_row_record_match_score(
    row: OtoTemplateRow,
    record: RowPlanRecord,
    records: Sequence[RowPlanRecord] = (),
    *,
    language: str = "",
    preferred_target: int | None = None,
) -> int:
    alias = str(getattr(row, "alias", "") or "").strip()
    record_alias = str(getattr(record, "alias", "") or "").strip()
    if not alias or not record_alias:
        return 0
    if _is_nonphonetic_special_alias(alias):
        return 0

    source_phones = tuple(_template_alias_phone_sequence(alias, language=language))
    record_phones = tuple(_alias_phone_sequence(record_alias))
    record_role = str(getattr(record, "role_family", "") or "").strip().lower()
    record_targets = tuple(int(value) for value in getattr(record, "expected_phone_indices", ()) or ())
    first_slot = int(getattr(record, "left_slot_index", -1)) == 0 and int(getattr(record, "right_slot_index", -1)) == 0
    source_role = _alias_type_for_row(alias, "auto")
    if (
        preferred_target is not None
        and source_role != "cv_head"
        and int(preferred_target) not in record_targets
    ):
        return 0
    phones_compatible = (
        source_phones == record_phones
        or _phone_sequence_variants_match(source_phones, record_phones)
        or (
            _is_japanese_language_name(language)
            and _japanese_liquid_phone_sequences_match(source_phones, record_phones)
        )
    )
    yoon_vc_compatible = _yoon_vc_template_row_matches_record(source_phones, record_phones, record_role)
    cv_head_initial_compatible = (
        first_slot
        and source_role == "cv_head"
        and _cv_head_template_row_matches_initial_record(source_phones, record_phones, record_role)
    )
    cv_head_exact_compatible = (
        source_role == "cv_head"
        and record_role in {"cv", "v"}
        and phones_compatible
    )
    standalone_glide_v_compatible = (
        _is_japanese_language_name(language)
        and source_role == "v"
        and record_role == "cv"
        and (
            (
                phones_compatible
                and len(source_phones) >= 2
                and str(source_phones[0]).strip().lower() in {"w", "y"}
            )
            or (
                source_phones == ("o",)
                and record_phones == ("wo",)
            )
        )
    )
    korean_vcv_target_compatible = (
        _is_korean_language_name(language)
        and source_role in {"vcv", "vv"}
        and record_role in {"vcv", "vv"}
        and preferred_target is not None
        and int(preferred_target) in record_targets
    )
    if not source_phones or not (
        phones_compatible
        or yoon_vc_compatible
        or cv_head_initial_compatible
        or cv_head_exact_compatible
        or standalone_glide_v_compatible
        or korean_vcv_target_compatible
    ):
        return 0

    if (
        first_slot
        and source_role == "cv"
        and record_role == "cv"
        and _initial_repeated_plain_cv_record(record, records)
    ):
        return 0
    if first_slot and source_role == "cv_head" and record_role in {"cv", "v"}:
        score = 1100
    elif cv_head_exact_compatible:
        score = 975 if alias == record_alias else 925
    elif standalone_glide_v_compatible:
        score = 850
    elif source_role == record_role:
        if yoon_vc_compatible:
            score = 880
        else:
            score = 1000 if alias == record_alias else 900
    elif korean_vcv_target_compatible:
        score = 875
    else:
        return 0

    if preferred_target is not None:
        if int(preferred_target) in record_targets:
            score += 250
    return score


def _template_alias_phone_sequence(alias: str, *, language: str) -> tuple[str, ...]:
    phones = tuple(_alias_phone_sequence(alias))
    if phones or not _is_japanese_language_name(language):
        return phones
    liquid_normalized = re.sub(
        r"l(?=[\u3040-\u30ff])",
        "",
        str(alias or ""),
        flags=re.IGNORECASE,
    )
    if liquid_normalized == str(alias or ""):
        return phones
    return tuple(_alias_phone_sequence(liquid_normalized))


def _japanese_liquid_phone_sequences_match(
    source_phones: Sequence[str],
    record_phones: Sequence[str],
) -> bool:
    if len(source_phones) != len(record_phones) or not source_phones:
        return False
    saw_liquid_variant = False
    for source_phone, record_phone in zip(source_phones, record_phones):
        source = str(source_phone or "").strip().lower()
        record = str(record_phone or "").strip().lower()
        if source == record or _phone_matches(source, record):
            continue
        if {source, record} == {"l", "r"}:
            saw_liquid_variant = True
            continue
        return False
    return saw_liquid_variant


def _initial_repeated_plain_cv_record(record: RowPlanRecord, records: Sequence[RowPlanRecord]) -> bool:
    if str(getattr(record, "role_family", "") or "").strip().lower() != "cv":
        return False
    if int(getattr(record, "left_slot_index", -1)) != 0 or int(getattr(record, "right_slot_index", -1)) != 0:
        return False
    record_phones = tuple(_alias_phone_sequence(str(getattr(record, "alias", "") or "")))
    if len(record_phones) < 2:
        return False
    for other in records:
        if other is record:
            continue
        if str(getattr(other, "role_family", "") or "").strip().lower() != "cv":
            continue
        if int(getattr(other, "left_slot_index", -1)) <= 0 or int(getattr(other, "right_slot_index", -1)) <= 0:
            continue
        if tuple(_alias_phone_sequence(str(getattr(other, "alias", "") or ""))) == record_phones:
            return True
    return False


def _cv_head_template_row_matches_initial_record(
    source_phones: Sequence[str],
    record_phones: Sequence[str],
    record_role: str,
) -> bool:
    if str(record_role or "").strip().lower() not in {"cv", "v"}:
        return False
    source = tuple(str(phone or "").strip().lower() for phone in source_phones if str(phone or "").strip())
    record = tuple(str(phone or "").strip().lower() for phone in record_phones if str(phone or "").strip())
    if not source or not record:
        return False
    if source == record:
        return True
    if len(source) >= len(record):
        return False
    return tuple(record[: len(source)]) == source


def _yoon_vc_template_row_matches_record(
    source_phones: Sequence[str],
    record_phones: Sequence[str],
    record_role: str,
) -> bool:
    if str(record_role or "").strip().lower() != "vc":
        return False
    if len(source_phones) != 3 or len(record_phones) != 2:
        return False
    if tuple(source_phones[:2]) != tuple(record_phones):
        return False
    return str(source_phones[2]).strip().lower() == "y"


def _template_row_semantic_sort_key(row: OtoTemplateRow) -> tuple[object, ...]:
    alias = str(getattr(row, "alias", "") or "").strip()
    role = _alias_type_for_row(alias, "auto") if alias else ""
    role_priority = {
        "cv_head": 0,
        "cv": 1,
        "v": 1,
        "vc": 2,
        "vcv": 3,
        "vv": 3,
    }.get(role, 9)
    return (
        role_priority,
        tuple(_alias_phone_sequence(alias)),
        alias.lower(),
        alias,
    )


def _build_timeline_debug_record(
    *,
    wav_name: str,
    wav_path: str,
    use_hsmm_decoder_requested: bool,
    selected_event_source: str,
    expected_phones: Iterable[str],
    filename_slots: Iterable[object],
    expected_slots: Iterable[object],
    prediction: RuntimePrediction,
    prediction_events: Iterable[Mapping[str, object]],
    selected_events: Iterable[Mapping[str, object]],
    hsmm: object | None,
    adapted_rows: Iterable[Mapping[str, object]],
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": TIMELINE_DEBUG_SCHEMA_VERSION,
        "wav": str(wav_name or ""),
        "wav_path": os.path.abspath(str(wav_path or "")),
        "use_hsmm_decoder_requested": bool(use_hsmm_decoder_requested),
        "selected_event_source": str(selected_event_source or ""),
        "expected_phones": [str(phone) for phone in expected_phones],
        "filename_slots": [_json_safe(item) for item in filename_slots],
        "expected_slots": [_expected_slot_debug(item) for item in expected_slots],
        "posterior_summary": _posterior_summary(prediction.posterior),
        "prediction_events": [_event_debug(event) for event in prediction_events],
        "selected_events": [_event_debug(event) for event in selected_events],
        "slot_assignments": [
            {
                "slot_index": int(assignment.slot_index),
                "phone_index": int(assignment.phone_index),
                "phone": str(assignment.phone),
                "role": str(assignment.role),
                "event_label": str(assignment.event_label),
                "selected_time_ms": _round_ms(assignment.selected_time_ms),
                "score": _round_score(assignment.score),
                "frame_index": int(assignment.frame_index),
                "expected_time_ms": (
                    _round_ms(assignment.expected_time_ms)
                    if assignment.expected_time_ms is not None
                    else None
                ),
            }
            for assignment in (prediction.slot_result.assignments if prediction.slot_result is not None else ())
        ],
        "slot_warnings": list(prediction.slot_result.warnings if prediction.slot_result is not None else ()),
        "slot_average_score": (
            _round_score(prediction.slot_result.average_score)
            if prediction.slot_result is not None
            else None
        ),
        "hsmm": _hsmm_debug(hsmm) if hsmm is not None else None,
        "evidence": _evidence_debug(evidence),
        "adapted_rows": [dict(row) for row in adapted_rows],
    }


def _evidence_debug(evidence: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(evidence, Mapping):
        return {}
    return {
        "schema_version": evidence.get("schema_version", ""),
        "wav_sha256": evidence.get("wav_sha256", ""),
        "extractor_version": evidence.get("extractor_version", ""),
        "decoder_version": evidence.get("decoder_version", ""),
        "timebase": dict(evidence.get("timebase", {}) or {}) if isinstance(evidence.get("timebase"), Mapping) else {},
        "reliability": dict(evidence.get("reliability", {}) or {}) if isinstance(evidence.get("reliability"), Mapping) else {},
        "event_candidate_count": len(list(evidence.get("event_candidates", []) or [])),
        "event_candidate_summary": summarize_event_candidate_reliability(evidence),
    }


def _posterior_summary(posterior) -> dict[str, object]:
    frame_count = int(posterior.frame_count())
    times = list(posterior.times_ms)
    duration_ms = float(times[-1]) if times else 0.0
    if len(times) >= 2:
        duration_ms += max(0.0, float(times[-1]) - float(times[-2]))
    return {
        "frame_count": frame_count,
        "duration_ms": _round_ms(duration_ms),
        "metadata": dict(posterior.metadata or {}),
        "event_peaks": {
            label: _top_track_peaks(posterior.times_ms, posterior.event_scores.get(label, ()))
            for label in ("phone_change", "cv_boundary", "vowel_nucleus")
        },
        "class_peaks": {
            label: _top_track_peaks(posterior.times_ms, posterior.class_probs.get(label, ()), limit=3)
            for label in ("consonant", "vowel")
        },
    }


def _hsmm_debug(hsmm) -> dict[str, object]:
    return {
        "ok": bool(hsmm.ok),
        "reason": str(hsmm.result.reason),
        "score": _round_score(hsmm.result.score),
        "timeout": bool(hsmm.result.timeout),
        "pruned_endpoint_count": int(hsmm.result.pruned_endpoint_count),
        "meta": dict(hsmm.result.meta or {}),
        "states": [
            {
                "state_id": str(state.state_id),
                "state_type": str(state.state_type),
                "start_ms": _round_ms(state.start_ms),
                "end_ms": _round_ms(state.end_ms),
                "duration_ms": _round_ms(state.duration_ms),
                "score": _round_score(state.score),
                "start_frame": int(state.start_frame),
                "end_frame": int(state.end_frame),
            }
            for state in hsmm.result.states
        ],
        "events": [_event_debug(event) for event in hsmm.events],
        "diagnostics": _json_safe(getattr(hsmm, "diagnostics", {})),
        "state_specs": [
            {
                "state_id": str(state.state_id),
                "state_type": str(state.state_type),
                "min_duration_ms": _round_ms(state.min_duration_ms),
                "max_duration_ms": _round_ms(state.max_duration_ms),
                "mode_duration_ms": _round_ms(state.mode_duration_ms),
                "duration_sigma_ms": _round_ms(state.duration_sigma_ms),
            }
            for state in hsmm.states
        ],
    }


def _adapted_row_debug(adapted, *, row_index: int, row_plan_record: object | None = None) -> dict[str, object]:
    payload = adapted.to_json_dict()
    payload["row_index"] = int(row_index)
    payload["raw_line"] = adapted.format_line()
    if row_plan_record is not None:
        payload["row_plan"] = _json_safe(row_plan_record)
    return payload


def _template_row_plan_context(
    template_row: object,
    row_index: int,
    *,
    anchor: object | None = None,
    filename_slots: Iterable[object] = (),
) -> dict[str, object]:
    alias = str(getattr(template_row, "alias", "") or "")
    role = str(getattr(anchor, "role", "") or "").strip().lower()
    alias_role_family = alias_family(alias)
    source_timing_trusted = bool(
        alias_role_family == "policy_breath" and not _template_row_timing_is_degenerate(template_row)
    )
    phone_index = _int_or_none(getattr(anchor, "expected_phone_index", None)) if anchor is not None else None
    slot_index = _filename_slot_index_for_phone(filename_slots, phone_index)
    left_slot_index = slot_index
    right_slot_index = slot_index
    if role == "vc" and slot_index is not None:
        left_slot_index = max(0, int(slot_index) - 1)
        right_slot_index = int(slot_index)
    role_family = alias_role_family if alias_role_family == "policy_breath" else role
    return {
        "wav": str(getattr(template_row, "wav", "") or ""),
        "alias": alias,
        "row_index": int(row_index),
        "slot_index": int(slot_index) if slot_index is not None else -1,
        "left_slot_index": int(left_slot_index) if left_slot_index is not None else -1,
        "right_slot_index": int(right_slot_index) if right_slot_index is not None else -1,
        "source_row_index": int(getattr(template_row, "source_row_index", -1)),
        "source_timing_trusted": source_timing_trusted,
        "role_family": role_family,
        "expected_phone_index": int(phone_index) if phone_index is not None else -1,
        "warnings": ["template_special_source_timing_preserved"]
        if source_timing_trusted
        else ["template_source_timing_untrusted"],
    }


def _template_row_timing_is_degenerate(template_row: object) -> bool:
    timing = getattr(template_row, "timing", None)
    return all(
        abs(float(getattr(timing, field, 0.0) or 0.0)) <= 1e-6
        for field in ("offset", "consonant", "cutoff", "preutterance", "overlap")
    )


def _filename_slot_index_for_phone(filename_slots: Iterable[object], phone_index: int | None) -> int | None:
    if phone_index is None:
        return None
    target = int(phone_index)
    for slot in filename_slots:
        start = _int_or_none(getattr(slot, "phone_start_index", None))
        vowel = _int_or_none(getattr(slot, "vowel_phone_index", None))
        slot_index = _int_or_none(getattr(slot, "slot_index", None))
        if start is None or vowel is None or slot_index is None:
            continue
        if int(start) <= target <= int(vowel):
            return int(slot_index)
    return None


def _expected_slot_debug(slot: object) -> dict[str, object]:
    return {
        "slot_index": int(getattr(slot, "slot_index", -1)),
        "phone_index": int(getattr(slot, "phone_index", -1)),
        "phone": str(getattr(slot, "phone", "")),
        "role": str(getattr(slot, "role", "")),
        "event_label": str(getattr(slot, "event_label", "")),
    }


def _event_debug(event: Mapping[str, object]) -> dict[str, object]:
    return {
        "label": str(event.get("label", "")),
        "selected_time_ms": _round_ms(event.get("selected_time_ms", event.get("time_ms", 0.0))),
        "score": _round_score(event.get("score", 0.0)),
        "expected_phone": str(event.get("expected_phone", "") or ""),
        "expected_phone_index": _int_or_none(event.get("expected_phone_index")),
        "slot_index": _int_or_none(event.get("slot_index")),
        "frame_index": _int_or_none(event.get("frame_index")),
        "source": str(event.get("source", "") or ""),
    }


def _top_track_peaks(
    times: Iterable[float],
    values: Iterable[float],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    pairs = [
        (idx, float(time_ms), float(score))
        for idx, (time_ms, score) in enumerate(zip(times, values))
    ]
    pairs.sort(key=lambda item: item[2], reverse=True)
    return [
        {
            "frame_index": int(idx),
            "time_ms": _round_ms(time_ms),
            "score": _round_score(score),
        }
        for idx, time_ms, score in pairs[: max(0, int(limit))]
    ]


def _json_safe(value: object) -> object:
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _round_ms(value: object) -> float:
    try:
        return round(float(value), 3)
    except Exception:
        return 0.0


def _round_score(value: object) -> float:
    try:
        return round(float(value), 6)
    except Exception:
        return 0.0


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _posterior_rule_based_reason(prediction: RuntimePrediction) -> str:
    meta = prediction.posterior.metadata or {}
    if not bool(meta.get("rule_based")):
        return ""
    return str(meta.get("rule_fallback_reason") or "rule_based")


def _collect_prediction_metrics(
    prediction: RuntimePrediction,
    slot_scores: list[float],
    warnings: list[str],
    rule_based_per_wav: list[str],
) -> None:
    rule_based_per_wav.append(_posterior_rule_based_reason(prediction))
    slot_result = prediction.slot_result
    if slot_result is None:
        warnings.append("slot_result_missing")
        return
    if slot_result.average_score is not None:
        slot_scores.append(float(slot_result.average_score))
    for warning in slot_result.warnings:
        warnings.append(str(warning))


def _event_source_for_oto(prediction: RuntimePrediction) -> Iterable[dict[str, object]]:
    if prediction.slot_result is None or not prediction.slot_result.assignments:
        return [
            {
                "label": event.label,
                "selected_time_ms": event.selected_time_ms,
                "score": event.score,
                "frame_index": event.frame_index,
            }
            for event in prediction.decoded_events
        ]
    return [
        {
            "label": assignment.event_label,
            "selected_time_ms": assignment.selected_time_ms,
            "score": assignment.score,
            "frame_index": assignment.frame_index,
            "expected_phone": assignment.phone,
            "expected_phone_index": assignment.phone_index,
            "slot_index": assignment.slot_index,
        }
        for assignment in prediction.slot_result.assignments
    ]


def _template_group_is_nonphonetic_special_only(template_group: Iterable[object]) -> bool:
    aliases = [str(getattr(row, "alias", "") or "").strip() for row in template_group]
    aliases = [alias for alias in aliases if alias]
    return bool(aliases) and all(_is_nonphonetic_special_alias(alias) for alias in aliases)


def _runtime_events_for_hsmm_guard(
    prediction: RuntimePrediction,
    prediction_events: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    meta = prediction.posterior.metadata or {}
    events = tuple(prediction_events)
    if bool(meta.get("rule_based")):
        return ()
    if _runtime_event_sequence_collapsed_for_hsmm_guard(prediction, events):
        return ()
    return events


def _runtime_event_sequence_collapsed_for_hsmm_guard(
    prediction: RuntimePrediction,
    events: Iterable[Mapping[str, object]],
    *,
    min_event_count: int = 4,
    min_first_ratio: float = 0.30,
    max_span_ratio: float = 0.30,
) -> bool:
    event_list = tuple(events)
    if len(event_list) < int(min_event_count):
        return False
    times = [
        float(event.get("selected_time_ms", event.get("time_ms", 0.0)) or 0.0)
        for event in event_list
        if str(event.get("label", "")) in {"phone_change", "cv_boundary", "vowel_nucleus", "vv_boundary"}
    ]
    if len(times) < int(min_event_count):
        return False
    phone_indices = sorted(
        {
            _int_field(event.get("expected_phone_index", -1), default=-1)
            for event in event_list
            if _int_field(event.get("expected_phone_index", -1), default=-1) >= 0
        }
    )
    if len(phone_indices) < int(min_event_count) or int(phone_indices[-1]) - int(phone_indices[0]) < int(min_event_count) - 1:
        return False
    posterior_times = list(getattr(prediction.posterior, "times_ms", ()) or ())
    if not posterior_times:
        return False
    duration_ms = max(float(posterior_times[-1]), 1.0)
    if len(posterior_times) >= 2:
        duration_ms += max(0.0, float(posterior_times[-1]) - float(posterior_times[-2]))
    first_ratio = min(times) / duration_ms
    span_ratio = (max(times) - min(times)) / duration_ms
    return first_ratio >= float(min_first_ratio) and span_ratio <= float(max_span_ratio)


def _adapter_config_for_row_plan_record(
    config: OtoAdapterConfig,
    row_plan_record: object | None,
) -> OtoAdapterConfig:
    role = str(getattr(row_plan_record, "role_family", "") or "").strip().lower()
    if role in {"cv", "cv_head", "vcv", "vc", "vv", "v"} and role != str(config.alias_type or "").strip().lower():
        return replace(config, alias_type=role)
    return config


def _hsmm_runtime_replacement_rejection_reason(
    expected_slots: Iterable[object],
    runtime_events: Iterable[Mapping[str, object]],
    *,
    language: str,
    format_type: str,
    template_rows: Iterable[object] = (),
) -> str:
    if not _is_japanese_cvvc_workflow(language=language, format_type=format_type):
        return ""
    if not _ja_cvvc_hsmm_runtime_gap_guard_enabled():
        return ""
    first_vc: object | None = None
    first_cv: object | None = None
    for slot in expected_slots:
        label = str(getattr(slot, "event_label", "") or "").strip()
        role = str(getattr(slot, "role", "") or "").strip().lower()
        if first_vc is None and label == "phone_change" and role == "vc":
            first_vc = slot
            continue
        if first_vc is not None and label == "cv_boundary" and role in {"implicit_cv", "cv", "cv_head"}:
            first_cv = slot
            break
    if first_vc is None or first_cv is None:
        return ""
    first_vc_time = _event_time_for_slot(runtime_events, label="phone_change", phone_index=getattr(first_vc, "phone_index", -1))
    first_cv_time = _event_time_for_slot(runtime_events, label="cv_boundary", phone_index=getattr(first_cv, "phone_index", -1))
    if first_vc_time is None or first_cv_time is None:
        return ""
    gap = float(first_cv_time) - float(first_vc_time)
    if gap > float(JA_CVVC_HSMM_RUNTIME_ALIGNED_FIRST_GAP_MS):
        return f"runtime_first_vc_cv_gap_aligned:{gap:.1f}ms"
    return ""


def _ja_cvvc_hsmm_runtime_gap_guard_enabled() -> bool:
    raw = str(os.environ.get("UTOA_NO_MFA_JA_CVVC_HSMM_RUNTIME_GAP_GUARD", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _is_japanese_cvvc_workflow(*, language: str, format_type: str) -> bool:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    return lang in {"japanese", "ja", "jp"} and fmt == "cvvc"


def _event_time_for_slot(
    events: Iterable[Mapping[str, object]],
    *,
    label: str,
    phone_index: object,
) -> float | None:
    target_index = _int_field(phone_index, default=-1)
    if target_index < 0:
        return None
    for event in events:
        if str(event.get("label", "") or "").strip() != str(label):
            continue
        if _int_field(event.get("expected_phone_index", -1), default=-1) != target_index:
            continue
        try:
            value = float(event.get("selected_time_ms", event.get("time_ms", 0.0)) or 0.0)
        except Exception:
            return None
        if math.isfinite(value):
            return value
    return None


def _hsmm_event_sequence_matches_expected(
    expected_slots: Iterable[object],
    hsmm_events: Iterable[Mapping[str, object]],
    *,
    runtime_events: Iterable[Mapping[str, object]] = (),
    max_runtime_delta_ms: float = 250.0,
) -> bool:
    expected = [
        (
            str(getattr(slot, "event_label", "")),
            _int_field(getattr(slot, "phone_index", -1), default=-1),
            str(getattr(slot, "role", "") or ""),
        )
        for slot in expected_slots
    ]
    expected = [item for item in expected if item[0]]
    if not expected:
        return True
    hsmm_items: list[tuple[tuple[str, int], Mapping[str, object], str]] = []
    hsmm_cursor = 0
    hsmm_list = list(hsmm_events)
    previous_vc_used_vowel_boundary = False
    for expected_key in expected:
        matched: tuple[tuple[str, int], Mapping[str, object], str] | None = None
        for idx in range(hsmm_cursor, len(hsmm_list)):
            event = hsmm_list[idx]
            label = str(event.get("label", ""))
            if label not in {"phone_change", "cv_boundary", "vowel_nucleus", "vv_boundary"}:
                continue
            key = (label, _int_field(event.get("expected_phone_index", -1), default=-1))
            if _hsmm_event_matches_expected_slot(expected_key, key) or (
                previous_vc_used_vowel_boundary and _hsmm_event_matches_following_vowel_boundary(expected_key, key)
            ):
                matched = ((expected_key[0], expected_key[1]), event, expected_key[2])
                hsmm_cursor = idx + 1
                break
        if matched is None:
            # implicit_cv boundaries are interpolated soft anchors that the expected
            # sequence can over-produce (e.g. a VC targeting a moraic-n emits a
            # cv_boundary at the *next* phone, which for JA moraic-n lands on the
            # following onset consonant the HSMM never anchors). Once at least one
            # primary anchor has matched, skipping such an unmatched interior
            # implicit_cv keeps a structurally correct decode from falling back to the
            # (often sparser) runtime events. Primary anchors (cv/vc/cv_head
            # phone_change + real cv_boundary) are still required in order, and an
            # implicit_cv that is the only/leading anchor still rejects, so a genuine
            # one-syllable shift or a structureless decode is still caught.
            if str(expected_key[2]).strip().lower() == "implicit_cv" and hsmm_items:
                continue
            return False
        previous_vc_used_vowel_boundary = (
            str(expected_key[2]).strip().lower() in {"vc", "cv_head"}
            and str(matched[1].get("label", "")).strip().lower() in {"vv_boundary", "vowel_nucleus"}
        )
        hsmm_items.append(matched)

    runtime_by_key: dict[tuple[str, int], list[float]] = {}
    for event in runtime_events:
        label = str(event.get("label", ""))
        if label not in {"phone_change", "cv_boundary", "vowel_nucleus"}:
            continue
        key = (label, _int_field(event.get("expected_phone_index", -1), default=-1))
        runtime_by_key.setdefault(key, []).append(float(event.get("selected_time_ms", 0.0) or 0.0))
    runtime_skip_phone_indices = {
        int(key[1])
        for key, event, expected_role in hsmm_items
        if _hsmm_runtime_delta_guard_can_skip(expected_role, str(event.get("label", "")))
    }
    for key, event, expected_role in hsmm_items:
        if _hsmm_runtime_delta_guard_can_skip(expected_role, str(event.get("label", ""))):
            continue
        if int(key[1]) in runtime_skip_phone_indices and str(event.get("label", "")) in {"vv_boundary", "vowel_nucleus"}:
            continue
        runtime_times = runtime_by_key.get(key) or []
        if not runtime_times:
            continue
        hsmm_time = float(event.get("selected_time_ms", 0.0) or 0.0)
        if min(abs(hsmm_time - runtime_time) for runtime_time in runtime_times) > float(max_runtime_delta_ms):
            return False
    return True


def _hsmm_runtime_delta_guard_can_skip(expected_role: str, actual_label: str) -> bool:
    return (
        str(expected_role or "").strip().lower() == "cv_head"
        and str(actual_label or "").strip().lower() in {"vv_boundary", "vowel_nucleus"}
    )


def _hsmm_event_matches_expected_slot(
    expected: tuple[str, int, str],
    actual: tuple[str, int],
) -> bool:
    expected_label, expected_phone_index, expected_role = expected
    actual_label, actual_phone_index = actual
    if int(actual_phone_index) != int(expected_phone_index):
        return False
    if actual_label == expected_label:
        return True
    if (
        str(expected_label) == "phone_change"
        and str(expected_role).strip().lower() == "vc"
        and str(actual_label) in {"vv_boundary", "vowel_nucleus"}
    ):
        return True
    if (
        str(expected_label) in {"phone_change", "cv_boundary"}
        and str(expected_role).strip().lower() == "cv_head"
        and str(actual_label) in {"vv_boundary", "vowel_nucleus"}
    ):
        return True
    if (
        str(expected_label) == "cv_boundary"
        and str(expected_role).strip().lower() == "vcv"
        and str(actual_label) in {"vv_boundary", "vowel_nucleus"}
    ):
        return True
    # A vowel-only first slot has no consonant state, so the filename HSMM
    # emits its only reliable anchor as a nucleus. For a cv_head row that is
    # still the right ordered slot; later local refinement can pull it to the
    # nearest boundary evidence. Do not apply this to implicit/interior CV
    # boundaries, where losing the consonant boundary would be unsafe.
    return (
        str(expected_label) == "cv_boundary"
        and str(expected_role).strip().lower() == "cv_head"
        and str(actual_label) in {"vv_boundary", "vowel_nucleus"}
    )


def _hsmm_event_matches_following_vowel_boundary(
    expected: tuple[str, int, str],
    actual: tuple[str, int],
) -> bool:
    expected_label, expected_phone_index, expected_role = expected
    actual_label, actual_phone_index = actual
    return (
        str(expected_label) == "cv_boundary"
        and str(expected_role).strip().lower() == "implicit_cv"
        and str(actual_label) in {"vv_boundary", "vowel_nucleus"}
        and int(actual_phone_index) == int(expected_phone_index)
    )


def _int_field(value: object, *, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _expected_phones(wav_name: str, contexts: list, *, filename_slots: Iterable[object] = ()) -> list[str]:
    slots = tuple(filename_slots)
    phones = infer_filename_phone_sequence(str(wav_name or ""))
    slot_phones = [
        str(phone or "").strip().lower()
        for phone in filename_phone_sequence_from_slots(slots)
        if str(phone or "").strip()
    ]
    if slot_phones and _prefer_filename_slot_phone_sequence(slots, slot_phones, phones):
        return slot_phones
    if phones:
        return phones
    if not contexts:
        return []
    tokens = []
    for token in contexts[0].filename.canonical_tokens:
        t = str(token or "").strip().lower()
        if not t:
            continue
        if t in {"sil", "pau", "sp", "ap"}:
            continue
        tokens.append(t)
    return tokens


def _prefer_filename_slot_phone_sequence(
    filename_slots: Iterable[object],
    slot_phones: Iterable[str],
    inferred_phones: Iterable[str],
) -> bool:
    slot_phone_list = [str(phone or "").strip().lower() for phone in slot_phones if str(phone or "").strip()]
    inferred_phone_list = [str(phone or "").strip().lower() for phone in inferred_phones if str(phone or "").strip()]
    if not slot_phone_list:
        return False
    if not inferred_phone_list:
        return True
    if tuple(slot_phone_list) == tuple(inferred_phone_list):
        return True
    if _expanded_multichar_phones(slot_phone_list) == tuple(inferred_phone_list):
        return any(len(phone) > 1 for phone in slot_phone_list)
    yoon_vowels = {"ya", "yu", "ye", "yo"}
    for slot in filename_slots:
        onset = str(getattr(slot, "onset", "") or "").strip().lower()
        vowel = str(getattr(slot, "vowel", "") or "").strip().lower()
        if onset and vowel in yoon_vowels and len(tuple(getattr(slot, "phones", ()) or ())) == 2:
            return len(slot_phone_list) < len(inferred_phone_list)
    return False


def _expanded_multichar_phones(phones: Iterable[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for phone in phones:
        text = str(phone or "").strip().lower()
        if not text:
            continue
        if len(text) > 1 and all(char.isalpha() for char in text):
            expanded.extend(text)
        else:
            expanded.append(text)
    return tuple(expanded)


def _cv_head_has_consonant_onset(alias: str) -> bool:
    """Return True if a cv_head alias starts with a consonant, e.g. '- w', '- ka'."""
    stripped = str(alias or "").strip()
    if stripped.startswith("-"):
        stripped = stripped[1:].strip()
    # Remove common suffix tokens
    for suffix in ("_D4", "_C4", "_A3", "_F4", "_S", "_P"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
    if not stripped:
        return False
    first = stripped[0].lower()
    vowels = set("aiueoあいうえおアイウエオ")
    return first not in vowels


def _normalize_wav_name(name: str) -> str:
    """Canonicalise a wav filename for matching across separator variants.

    UTAU/Windows commonly rewrite reclist apostrophes into underscores, so a base
    OTO that references ``ga'gi'gu.wav`` must still match a recorded
    ``ga_gi_gu.wav``. Unify apostrophe / hyphen / space separators to '_'.
    """
    text = os.path.basename(str(name or "")).strip().lower()
    for ch in ("'", "’", "ʼ", "`", "\"", "-", " "):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _normalize_out_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if raw.endswith(("\\", "/")):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    if os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    return raw


def _repair_anchor_monotonicity(anchors: list) -> list:
    if not anchors:
        return anchors
    out = list(anchors)
    last_time = -1.0
    for idx, anchor in enumerate(out):
        if anchor is None:
            continue
        if float(anchor.anchor_abs_ms) + 1e-5 < last_time:
            if _anchor_allows_row_order_backtrack(anchor):
                last_time = float(anchor.anchor_abs_ms)
                continue
            repaired_time = last_time + 4.0
            out[idx] = replace(
                anchor,
                anchor_abs_ms=repaired_time,
                warnings=tuple(dict.fromkeys((*anchor.warnings, f"monotonic_repaired:{anchor.anchor_abs_ms:.1f}->{repaired_time:.1f}"))),
            )
            last_time = repaired_time
        else:
            last_time = float(anchor.anchor_abs_ms)
    return out


def _anchor_allows_row_order_backtrack(anchor) -> bool:
    return any(str(warning).startswith("alias_target_backtrack") for warning in getattr(anchor, "warnings", ()) or ())


def _safe_conf(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_boundary_evidence(posterior, adapted, *, policy: WorldV1RuntimePolicy) -> BoundaryEvidence:
    anchor = adapted.anchor
    if anchor is None:
        return BoundaryEvidence(
            c_onset=None,
            cv_boundary=None,
            v_offset=None,
            nucleus=None,
            cv_confidence=0.0,
            nucleus_confidence=0.0,
            confidence=0.0,
            warnings=("missing_anchor",),
        )
    c_onset = None
    cv_boundary = float(anchor.anchor_abs_ms)
    if anchor.vowel_start_abs_ms is not None:
        c_onset = max(0.0, float(anchor.vowel_start_abs_ms) - _C_ONSET_LEAD_MS)
    v_offset = float(anchor.vowel_end_abs_ms) if anchor.vowel_end_abs_ms is not None else None
    nucleus = float(anchor.vowel_nucleus_abs_ms) if anchor.vowel_nucleus_abs_ms is not None else cv_boundary
    warnings: list[str] = []
    if c_onset is not None and cv_boundary is not None and c_onset > cv_boundary:
        c_onset = max(0.0, cv_boundary - _C_ONSET_REPAIR_MS)
        warnings.append("c_onset_repaired")
    if v_offset is not None and cv_boundary is not None and v_offset < cv_boundary:
        v_offset = cv_boundary + _V_OFFSET_REPAIR_MS
        warnings.append("v_offset_repaired")
    if nucleus is not None and cv_boundary is not None and nucleus < cv_boundary:
        nucleus = cv_boundary
        warnings.append("nucleus_repaired")
    cv_anchor_conf = _safe_conf(float(anchor.boundary_confidence))
    nucleus_anchor_conf = _safe_conf(float(anchor.nucleus_confidence))
    cv_event_score = _event_score_at_ms(posterior, "cv_boundary", cv_boundary)
    nucleus_event_score = _event_score_at_ms(posterior, "vowel_nucleus", nucleus if nucleus is not None else cv_boundary)
    cv_wa, cv_wp = _normalized_pair(policy.cv_anchor_weight, policy.cv_posterior_weight)
    vn_wa, vn_wp = _normalized_pair(policy.nucleus_anchor_weight, policy.nucleus_posterior_weight)
    cv_conf = _safe_conf((cv_wa * cv_anchor_conf) + (cv_wp * cv_event_score))
    nucleus_conf = _safe_conf((vn_wa * nucleus_anchor_conf) + (vn_wp * nucleus_event_score))
    conf = _safe_conf((0.5 * cv_conf) + (0.5 * nucleus_conf))
    if conf < float(policy.boundary_evidence_low_confidence):
        warnings.append(f"low_boundary_evidence_confidence:{conf:.3f}")
    return BoundaryEvidence(
        c_onset=c_onset,
        cv_boundary=cv_boundary,
        v_offset=v_offset,
        nucleus=nucleus,
        cv_confidence=cv_conf,
        nucleus_confidence=nucleus_conf,
        confidence=conf,
        warnings=tuple(dict.fromkeys((*anchor.warnings, *warnings))),
    )


def _row_confidence(adapted, evidence: BoundaryEvidence, *, policy: WorldV1RuntimePolicy) -> float:
    base_anchor = float(adapted.anchor.score) if adapted.anchor is not None else 0.0
    row_wa, row_wb = _normalized_pair(policy.row_anchor_weight, policy.row_boundary_weight)
    warning_penalty = 0.04 * float(len(adapted.warnings))
    return _safe_conf((row_wa * base_anchor) + (row_wb * evidence.confidence) - warning_penalty)


def _event_score_at_ms(posterior, event_label: str, time_ms: float | None) -> float:
    if time_ms is None:
        return 0.0
    times = posterior.times_ms or []
    scores = posterior.event_scores.get(event_label) or []
    if not times or not scores:
        return 0.0
    if len(times) != len(scores):
        return 0.0
    nearest = min(range(len(times)), key=lambda idx: abs(float(times[idx]) - float(time_ms)))
    return _safe_conf(float(scores[nearest]))


def _normalized_pair(a: float, b: float) -> tuple[float, float]:
    wa = max(0.0, float(a))
    wb = max(0.0, float(b))
    total = wa + wb
    if total <= 1e-9:
        return 0.5, 0.5
    return wa / total, wb / total


def _wav_duration_ms(path: str, warnings: list[str] | None = None) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = max(1, int(handle.getframerate()))
            frames = max(0, int(handle.getnframes()))
            return 1000.0 * float(frames) / float(rate)
    except (wave.Error, EOFError, OSError) as exc:
        if warnings is not None:
            warnings.append(f"wav_duration_unavailable:{os.path.basename(str(path))}:{exc}")
        return 0.0
