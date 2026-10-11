from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import soundfile as sf

from .annotations import load_annotation
from .project import load_phone_inventory


FORMATS = ("korean_vcv", "korean_cvvc")
SPLITS = ("train", "validation", "locked_test")
OTO_ANCHOR_SLOTS = ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
_SHA256_LENGTH = 64

EvidenceInput = Path | Mapping[str, object]


class PromotionGateError(ValueError):
    pass


@dataclass(frozen=True)
class GoldAuditPolicy:
    """Minimum independently-recorded clips required per Korean format.

    The locked profile totals 30 recordings per format (60 overall), matching
    the minimum universe used by the locked quality comparison. Values are
    configurable so a small development corpus can exercise the same contract
    without weakening the release profile.
    """

    development_minimum_train_per_format: int = 2
    development_minimum_validation_per_format: int = 2
    development_minimum_locked_test_per_format: int = 1
    locked_minimum_train_per_format: int = 18
    locked_minimum_validation_per_format: int = 6
    locked_minimum_locked_test_per_format: int = 6
    development_double_annotation_fraction: float = 0.10
    locked_double_annotation_fraction: float = 0.10
    development_minimum_double_annotated_per_format: int = 1
    locked_minimum_double_annotated_per_format: int = 3


@dataclass(frozen=True)
class ReleaseGatePolicy:
    require_locked_quality_gate: bool = True
    require_locked_gold_audit: bool = True
    minimum_blind_trials: int = 20
    minimum_blind_listeners: int = 3
    minimum_blind_trials_per_format: int = 5
    minimum_candidate_preference_rate: float = 0.55


def audit_manual_gold_manifest(
    manifest_path: Path,
    *,
    stage: str = "development",
    output_json: Path | None = None,
    policy: GoldAuditPolicy | None = None,
) -> dict[str, object]:
    """Audit a Korean VCV/CVVC manual-Gold corpus without inventing evidence.

    Every recording must point to existing WAV and phone-annotation files whose
    SHA-256 digests match the manifest. A configurable subset must additionally
    include a real adjudication file and at least two named annotators.
    """

    resolved_manifest = manifest_path.resolve()
    payload = _load_json_mapping(resolved_manifest)
    normalized_stage = _normalize_stage(stage)
    resolved_policy = policy or GoldAuditPolicy()
    checks: list[dict[str, object]] = []

    _check_equal(checks, "schema_version", 1, payload.get("schema_version"))
    _check_equal(checks, "purpose", "manual_phone_gold_dataset", payload.get("purpose"))
    _check_nonempty(checks, "dataset_id", payload.get("dataset_id"))
    provenance = _as_mapping(payload.get("provenance"))
    _check_equal(
        checks,
        "provenance.reference_kind",
        "manual_phone_gold",
        provenance.get("reference_kind"),
    )
    _check_equal(
        checks,
        "provenance.creation_method",
        "human_manual",
        provenance.get("creation_method"),
    )
    _check_nonempty(checks, "provenance.annotation_protocol_id", provenance.get("annotation_protocol_id"))
    _check_sha(checks, "provenance.annotation_protocol_sha256", provenance.get("annotation_protocol_sha256"))
    _check_sha(checks, "provenance.source_inventory_sha256", provenance.get("source_inventory_sha256"))
    metadata_files = _as_mapping(provenance.get("metadata_files"))
    if metadata_files:
        _audit_gold_provenance_files(checks, provenance, metadata_files, resolved_manifest.parent)
    elif normalized_stage == "locked":
        _check_condition(
            checks,
            "provenance.metadata_files.available",
            expected="self_contained_verified_metadata",
            actual="missing",
            passed=False,
            reason="locked_gold_requires_self_contained_metadata",
        )
    if normalized_stage == "locked":
        _check_equal(
            checks,
            "provenance.release_locked_eligible",
            True,
            provenance.get("release_locked_eligible"),
        )
        _check_equal(
            checks,
            "provenance.prior_model_exposure_status",
            "model_unseen",
            provenance.get("prior_model_exposure_status"),
        )

    raw_recordings = payload.get("recordings")
    recordings = _as_sequence(raw_recordings)
    _check_condition(
        checks,
        "recordings.available",
        expected="non_empty_sequence",
        actual=len(recordings),
        passed=bool(recordings),
        reason="recordings_missing",
    )
    if metadata_files:
        dataset_content = {
            "preparation_manifest_sha256": provenance.get("preparation_manifest_sha256"),
            "metadata_files": metadata_files,
            "permission_evidence": _as_sequence(provenance.get("permission_evidence")),
            "phone_inventories": _as_sequence(provenance.get("phone_inventories")),
            "recordings": recordings,
        }
        calculated_content_sha256 = canonical_json_sha256(dataset_content)
        _check_equal(
            checks,
            "provenance.dataset_content_sha256",
            calculated_content_sha256,
            provenance.get("dataset_content_sha256"),
        )
        _check_equal(
            checks,
            "dataset_id.content_binding",
            f"korean-vcv-cvvc-manual-gold-{calculated_content_sha256[:16]}",
            payload.get("dataset_id"),
        )
    expected_permission_bank_ids = {
        _text(_as_mapping(recording).get("bank_id"))
        for recording in recordings
        if _text(_as_mapping(recording).get("bank_id"))
    }
    if metadata_files:
        _audit_gold_permission_coverage(
            checks,
            provenance,
            expected_permission_bank_ids,
        )

    manifest_root = resolved_manifest.parent
    seen_recording_ids: set[str] = set()
    speaker_splits: dict[str, set[str]] = {}
    counts = {format_name: {split: 0 for split in SPLITS} for format_name in FORMATS}
    double_counts = {format_name: 0 for format_name in FORMATS}
    recording_rows: list[dict[str, object]] = []
    anchor_inventory_rows: list[dict[str, object]] = []
    seen_anchor_identity: set[tuple[str, str, str, int]] = set()

    for index, raw_recording in enumerate(recordings):
        path = f"recordings[{index}]"
        recording = _as_mapping(raw_recording)
        row_failures_before = _failure_count(checks)
        recording_id = _text(recording.get("recording_id"))
        _check_nonempty(checks, f"{path}.recording_id", recording_id)
        duplicate = bool(recording_id and recording_id in seen_recording_ids)
        _check_condition(
            checks,
            f"{path}.recording_id.unique",
            expected=True,
            actual=not duplicate,
            passed=not duplicate,
            reason="duplicate_recording_id",
        )
        if recording_id:
            seen_recording_ids.add(recording_id)

        format_name = _canonical_format(recording)
        _check_condition(
            checks,
            f"{path}.format",
            expected=list(FORMATS),
            actual=format_name or recording.get("format"),
            passed=format_name in FORMATS,
            reason="unsupported_or_non_korean_format",
        )
        split = _text(recording.get("split")).casefold()
        _check_condition(
            checks,
            f"{path}.split",
            expected=list(SPLITS),
            actual=split,
            passed=split in SPLITS,
            reason="invalid_split",
        )
        root_speaker_id = _text(recording.get("root_speaker_id"))
        _check_nonempty(checks, f"{path}.root_speaker_id", root_speaker_id)
        bank_id = _text(recording.get("bank_id"))
        wav_key = _text(recording.get("wav_key"))
        _check_nonempty(checks, f"{path}.bank_id", bank_id)
        _check_nonempty(checks, f"{path}.wav_key", wav_key)
        if normalized_stage == "locked":
            _check_equal(
                checks,
                f"{path}.release_locked_eligible",
                True,
                recording.get("release_locked_eligible"),
            )
            _check_equal(
                checks,
                f"{path}.prior_model_exposure_status",
                "model_unseen",
                recording.get("prior_model_exposure_status"),
            )
        if format_name in FORMATS and split in SPLITS:
            counts[format_name][split] += 1
        if root_speaker_id and split in SPLITS:
            speaker_splits.setdefault(root_speaker_id, set()).add(split)

        wav = _as_mapping(recording.get("wav"))
        require_self_contained = bool(metadata_files) or normalized_stage == "locked"
        wav_ok = _check_file_evidence(
            checks,
            f"{path}.wav",
            wav,
            manifest_root,
            require_within_root=require_self_contained,
        )
        wav_path = _resolve_evidence_path(manifest_root, wav)
        wav_sha256 = _text(wav.get("sha256")).casefold()
        wav_duration_ms: float | None = None
        if wav_ok and wav_path is not None:
            try:
                wav_info = sf.info(wav_path)
                wav_duration_ms = 1000.0 * wav_info.frames / float(wav_info.samplerate)
                _check_condition(
                    checks,
                    f"{path}.wav.decodable",
                    expected="decodable_audio",
                    actual=wav_info.format,
                    passed=wav_info.frames > 0 and wav_info.samplerate > 0,
                    reason="wav_decode_failed_or_empty",
                )
            except Exception as error:
                _check_condition(
                    checks,
                    f"{path}.wav.decodable",
                    expected="decodable_audio",
                    actual=type(error).__name__,
                    passed=False,
                    reason="wav_decode_failed_or_empty",
                )
        annotation = _as_mapping(recording.get("annotation"))
        _check_equal(
            checks,
            f"{path}.annotation.reference_kind",
            "manual_phone_gold",
            annotation.get("reference_kind"),
        )
        _check_equal(
            checks,
            f"{path}.annotation.creation_method",
            "human_manual",
            annotation.get("creation_method"),
        )
        _check_nonempty(checks, f"{path}.annotation.annotator_id", annotation.get("annotator_id"))
        annotation_ok = _check_file_evidence(
            checks,
            f"{path}.annotation",
            annotation,
            manifest_root,
            require_within_root=require_self_contained,
        )
        annotation_path = _resolve_evidence_path(manifest_root, annotation)
        annotation_semantics_ok = False
        annotation_inventory_ok = False
        if annotation_ok and annotation_path is not None:
            try:
                document = load_annotation(annotation_path)
                boundaries = [marker for marker in document.markers if marker.label == "phone_boundary"]
                start = [marker for marker in document.markers if marker.label == "usable_start"]
                end = [marker for marker in document.markers if marker.label == "usable_end"]
                gold_markers = [*start, *boundaries, *end]
                expected_pairs = {(index, index + 1) for index in range(len(document.phones) - 1)}
                actual_pairs = {(marker.left_phone_index, marker.right_phone_index) for marker in boundaries}
                ordered_boundaries = sorted(
                    boundaries,
                    key=lambda marker: int(marker.right_phone_index or 0),
                )
                interval_edges = (
                    [start[0].time_ms, *(marker.time_ms for marker in ordered_boundaries), end[0].time_ms]
                    if len(start) == 1 and len(end) == 1
                    else []
                )
                annotation_semantics_ok = bool(
                    document.label_source == "manual_gold"
                    and document.wav_sha256 == wav_sha256
                    and document.phones
                    and len(start) == 1
                    and len(end) == 1
                    and len(boundaries) == len(document.phones) - 1
                    and actual_pairs == expected_pairs
                    and bool(gold_markers)
                    and all(
                        marker.annotator == _text(annotation.get("annotator_id"))
                        and marker.source
                        in {"human_manual", "manual_vlabeler_phone_interval"}
                        and marker.confidence == 1.0
                        for marker in gold_markers
                    )
                    and (wav_duration_ms is None or 0.0 <= start[0].time_ms < end[0].time_ms <= wav_duration_ms + 0.001)
                    and all(start[0].time_ms < marker.time_ms < end[0].time_ms for marker in boundaries)
                    and all(
                        right > left + 0.0005
                        for left, right in zip(interval_edges, interval_edges[1:])
                    )
                )
                inventory_evidence = _as_mapping(annotation.get("inventory"))
                if inventory_evidence:
                    inventory_file_ok = _check_file_evidence(
                        checks,
                        f"{path}.annotation.inventory",
                        inventory_evidence,
                        manifest_root,
                        require_within_root=require_self_contained,
                    )
                    inventory_path = _resolve_evidence_path(
                        manifest_root,
                        inventory_evidence,
                    )
                    if inventory_file_ok and inventory_path is not None:
                        inventory, inventory_semantic_sha256 = load_phone_inventory(
                            inventory_path
                        )
                        supplied_phone_ids = sorted(
                            _text(value)
                            for value in _as_sequence(
                                inventory_evidence.get("phone_ids")
                            )
                        )
                        annotation_inventory_ok = bool(
                            inventory_semantic_sha256 == document.inventory_sha256
                            and _text(
                                inventory_evidence.get("inventory_sha256")
                            ).casefold()
                            == inventory_semantic_sha256
                            and supplied_phone_ids == sorted(inventory)
                            and all(phone in inventory for phone in document.phones)
                        )
                elif not require_self_contained:
                    annotation_inventory_ok = True
            except Exception:
                annotation_semantics_ok = False
                annotation_inventory_ok = False
        _check_condition(
            checks,
            f"{path}.annotation.manual_gold_semantics",
            expected="valid_manual_gold_annotation_bound_to_wav",
            actual=annotation_semantics_ok,
            passed=annotation_semantics_ok,
            reason="annotation_semantics_or_wav_binding_invalid",
        )
        _check_condition(
            checks,
            f"{path}.annotation.inventory_binding",
            expected="verified_self_contained_inventory_matching_annotation",
            actual=annotation_inventory_ok,
            passed=annotation_inventory_ok,
            reason="annotation_inventory_evidence_missing_or_mismatched",
        )

        raw_expected_slot_ids = _as_sequence(recording.get("expected_slot_ids"))
        expected_slot_ids = [_text(value) for value in raw_expected_slot_ids]
        expected_slots_valid = bool(expected_slot_ids) and all(expected_slot_ids) and len(set(expected_slot_ids)) == len(expected_slot_ids)
        _check_condition(
            checks,
            f"{path}.expected_slot_ids",
            expected="non_empty_unique_slot_ids",
            actual=expected_slot_ids,
            passed=expected_slots_valid,
            reason="expected_slot_inventory_missing_or_invalid",
        )
        anchor_failures_before = _failure_count(checks)
        anchor_rows = _as_sequence(recording.get("oto_anchor_rows"))
        anchors_ok = bool(anchor_rows)
        _check_condition(
            checks,
            f"{path}.oto_anchor_rows.available",
            expected="one_or_more_rows",
            actual=len(anchor_rows),
            passed=anchors_ok,
            reason="oto_anchor_rows_missing",
        )
        seen_slot_ids: set[str] = set()
        for anchor_index, raw_anchor_row in enumerate(anchor_rows):
            anchor_path = f"{path}.oto_anchor_rows[{anchor_index}]"
            anchor_row = _as_mapping(raw_anchor_row)
            slot_id = _text(anchor_row.get("slot_id"))
            _check_nonempty(checks, f"{anchor_path}.slot_id", slot_id)
            unique_slot = bool(slot_id and slot_id not in seen_slot_ids)
            _check_condition(
                checks,
                f"{anchor_path}.slot_id.unique",
                expected=True,
                actual=unique_slot,
                passed=unique_slot,
                reason="duplicate_or_empty_slot_id",
            )
            if slot_id:
                seen_slot_ids.add(slot_id)
            _check_equal(
                checks,
                f"{anchor_path}.reference_kind",
                "manual_oto_anchor_gold",
                anchor_row.get("reference_kind"),
            )
            alias = _text(anchor_row.get("alias"))
            occurrence = anchor_row.get("occurrence")
            valid_occurrence = (
                isinstance(occurrence, int)
                and not isinstance(occurrence, bool)
                and int(occurrence) >= 0
            )
            _check_nonempty(checks, f"{anchor_path}.alias", alias)
            _check_condition(
                checks,
                f"{anchor_path}.occurrence",
                expected="non_negative_integer",
                actual=occurrence,
                passed=valid_occurrence,
                reason="invalid_alias_occurrence",
            )
            anchor_identity = (bank_id, wav_key, alias, int(occurrence) if valid_occurrence else -1)
            identity_unique = bool(
                bank_id
                and wav_key
                and alias
                and valid_occurrence
                and anchor_identity not in seen_anchor_identity
            )
            _check_condition(
                checks,
                f"{anchor_path}.exact_identity.unique",
                expected=True,
                actual=identity_unique,
                passed=identity_unique,
                reason="duplicate_or_invalid_exact_oto_identity",
            )
            if identity_unique:
                seen_anchor_identity.add(anchor_identity)
            anchors = _as_mapping(anchor_row.get("anchors"))
            canonical_anchors: dict[str, float] = {}
            for anchor_name in OTO_ANCHOR_SLOTS:
                value = anchors.get(anchor_name)
                valid_number = _is_finite_number(value)
                anchors_ok = anchors_ok and valid_number
                _check_condition(
                    checks,
                    f"{anchor_path}.anchors.{anchor_name}",
                    expected="finite_number_ms",
                    actual=value,
                    passed=valid_number,
                    reason="oto_anchor_missing_or_non_numeric",
                )
                if valid_number:
                    canonical_anchors[anchor_name] = float(value)
            if identity_unique and len(canonical_anchors) == len(OTO_ANCHOR_SLOTS):
                ordered = [canonical_anchors[name] for name in OTO_ANCHOR_SLOTS]
                anchors_in_range = bool(
                    wav_duration_ms is not None
                    and ordered[0] >= -1e-6
                    and ordered[-1] <= wav_duration_ms + 0.001
                    and all(left <= right + 1e-6 for left, right in zip(ordered, ordered[1:]))
                )
                _check_condition(
                    checks,
                    f"{anchor_path}.anchors.order_and_wav_range",
                    expected="0<=offset<=overlap<=pre<=fixed<=cutoff<=wav_duration",
                    actual=ordered,
                    passed=anchors_in_range,
                    reason="oto_anchor_order_or_wav_range_invalid",
                )
                anchor_inventory_rows.append(
                    {
                        "bank_id": bank_id,
                        "wav_key": wav_key,
                        "alias": alias,
                        "occurrence": int(occurrence),
                        "anchors": canonical_anchors,
                    }
                )
        _check_condition(
            checks,
            f"{path}.oto_anchor_rows.slot_coverage",
            expected=sorted(expected_slot_ids),
            actual=sorted(seen_slot_ids),
            passed=expected_slots_valid and set(seen_slot_ids) == set(expected_slot_ids),
            reason="oto_anchor_slot_coverage_mismatch",
        )
        anchors_ok = anchors_ok and _failure_count(checks) == anchor_failures_before

        review_evidence = _as_mapping(recording.get("oto_review_evidence"))
        if review_evidence:
            _audit_gold_oto_review_binding(
                checks,
                path,
                recording,
                review_evidence,
                anchor_rows,
                manifest_root,
            )
        elif normalized_stage == "locked":
            _check_condition(
                checks,
                f"{path}.oto_review_evidence.available",
                expected="hash_bound_primary_and_adjudication_evidence",
                actual="missing",
                passed=False,
                reason="locked_gold_requires_oto_review_evidence",
            )

        double_annotation = _as_mapping(recording.get("double_annotation"))
        double_ok = _audit_double_annotation(
            checks,
            path,
            double_annotation,
            manifest_root,
            required=False,
            require_within_root=require_self_contained,
        )
        if format_name in FORMATS and double_ok:
            double_counts[format_name] += 1

        recording_rows.append(
            {
                "recording_id": recording_id,
                "format": format_name,
                "split": split,
                "root_speaker_id": root_speaker_id,
                "wav_verified": wav_ok,
                "annotation_verified": annotation_ok,
                "oto_anchor_rows_verified": anchors_ok,
                "double_annotation_verified": double_ok,
                "passed": _failure_count(checks) == row_failures_before,
            }
        )

    leaking_speakers = sorted(speaker for speaker, assigned in speaker_splits.items() if len(assigned) > 1)
    _check_condition(
        checks,
        "speaker_disjoint_splits",
        expected=[],
        actual=leaking_speakers,
        passed=not leaking_speakers,
        reason="root_speaker_split_leakage",
    )

    minimums = _minimum_counts(normalized_stage, resolved_policy)
    fraction = (
        resolved_policy.locked_double_annotation_fraction
        if normalized_stage == "locked"
        else resolved_policy.development_double_annotation_fraction
    )
    minimum_double = (
        resolved_policy.locked_minimum_double_annotated_per_format
        if normalized_stage == "locked"
        else resolved_policy.development_minimum_double_annotated_per_format
    )
    if not 0.0 <= fraction <= 1.0:
        raise PromotionGateError("double annotation fraction must be between 0 and 1")
    for format_name in FORMATS:
        for split in SPLITS:
            required = minimums[split]
            actual = counts[format_name][split]
            _check_condition(
                checks,
                f"coverage.{format_name}.{split}",
                expected=required,
                actual=actual,
                passed=actual >= required,
                reason="minimum_recording_count_not_met",
            )
        total = sum(counts[format_name].values())
        required_double = max(int(minimum_double), int(math.ceil(total * fraction)))
        _check_condition(
            checks,
            f"double_annotation.{format_name}",
            expected=required_double,
            actual=double_counts[format_name],
            passed=double_counts[format_name] >= required_double,
            reason="double_annotation_adjudication_minimum_not_met",
        )

    failures = [check for check in checks if not bool(check["passed"])]
    anchor_inventory_rows.sort(
        key=lambda item: (
            str(item["bank_id"]),
            str(item["wav_key"]),
            str(item["alias"]),
            int(item["occurrence"]),
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "gate_kind": "manual_gold_dataset_audit",
        "stage": normalized_stage,
        "passed": not failures,
        "manifest": str(resolved_manifest),
        "manifest_sha256": _file_sha256(resolved_manifest),
        "oto_anchor_inventory_sha256": canonical_json_sha256(anchor_inventory_rows),
        "oto_anchor_row_count": len(anchor_inventory_rows),
        "policy": asdict(resolved_policy),
        "required_formats": list(FORMATS),
        "required_splits": list(SPLITS),
        "counts_by_format_and_split": counts,
        "double_annotated_counts_by_format": double_counts,
        "speaker_count": len(speaker_splits),
        "speaker_split_leakage": leaking_speakers,
        "recording_count": len(recordings),
        "recordings": recording_rows,
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "checks": checks,
        "failures": failures,
    }
    if output_json is not None:
        write_deterministic_json(output_json, report)
    return report


def evaluate_release_gate(
    quality_gate: EvidenceInput | None,
    runtime_smoke: EvidenceInput | None,
    blind_ab: EvidenceInput | None,
    *,
    gold_audit: EvidenceInput | None = None,
    output_json: Path | None = None,
    policy: ReleaseGatePolicy | None = None,
) -> dict[str, object]:
    """Fail-closed release decision over machine and human evidence.

    This function only validates blind-listening evidence supplied by a human.
    It deliberately has no API that fabricates, completes, or infers responses.
    """

    resolved_policy = policy or ReleaseGatePolicy()
    checks: list[dict[str, object]] = []
    quality, quality_source = _safe_load_evidence("quality_gate", quality_gate)
    runtime, runtime_source = _safe_load_evidence("runtime_smoke", runtime_smoke)
    listening, listening_source = _safe_load_evidence("blind_ab", blind_ab)
    gold, gold_source = _safe_load_evidence("gold_audit", gold_audit)

    _check_evidence_loaded(checks, "quality_gate", quality_source)
    if quality is not None:
        _check_equal(checks, "quality_gate.schema_version", 1, quality.get("schema_version"))
        _check_equal(
            checks,
            "quality_gate.gate_kind",
            "frozen_exe_manual_oto_anchor_quality",
            quality.get("gate_kind"),
        )
        _check_equal(checks, "quality_gate.passed", True, quality.get("passed"))
        _check_equal(checks, "quality_gate.failed_check_count", 0, quality.get("failed_check_count"))
        if resolved_policy.require_locked_quality_gate:
            _check_equal(checks, "quality_gate.stage", "locked", quality.get("stage"))
        _check_sha(checks, "quality_gate.baseline_sha256", quality.get("baseline_sha256"))
        _check_sha(checks, "quality_gate.candidate_sha256", quality.get("candidate_sha256"))

    if resolved_policy.require_locked_gold_audit or gold_audit is not None:
        _check_evidence_loaded(checks, "gold_audit", gold_source)
        if gold is not None:
            _check_equal(checks, "gold_audit.schema_version", 1, gold.get("schema_version"))
            _check_equal(checks, "gold_audit.gate_kind", "manual_gold_dataset_audit", gold.get("gate_kind"))
            _check_equal(checks, "gold_audit.passed", True, gold.get("passed"))
            _check_equal(checks, "gold_audit.failed_check_count", 0, gold.get("failed_check_count"))
            if resolved_policy.require_locked_gold_audit:
                _check_equal(checks, "gold_audit.stage", "locked", gold.get("stage"))
            _check_sha(checks, "gold_audit.manifest_sha256", gold.get("manifest_sha256"))

    _check_evidence_loaded(checks, "runtime_smoke", runtime_source)
    if runtime is not None:
        _audit_runtime_smoke(checks, runtime)

    _check_evidence_loaded(checks, "blind_ab", listening_source)
    listening_summary: dict[str, object] = {}
    if listening is not None:
        listening_summary = _audit_blind_ab(checks, listening, resolved_policy)

    if quality is not None and runtime is not None:
        _check_equal(
            checks,
            "evidence_binding.candidate_runtime_artifact_sha256",
            quality.get("candidate_sha256"),
            runtime.get("artifact_sha256"),
        )
    if quality is not None and gold is not None:
        _check_equal(
            checks,
            "evidence_binding.gold_audit_sha256",
            quality.get("gold_audit_sha256"),
            gold_source.get("sha256"),
        )
        _check_equal(
            checks,
            "evidence_binding.gold_manifest_sha256",
            quality.get("gold_manifest_sha256"),
            gold.get("manifest_sha256"),
        )
        _check_equal(
            checks,
            "evidence_binding.gold_anchor_inventory_sha256",
            quality.get("gold_anchor_inventory_sha256"),
            gold.get("oto_anchor_inventory_sha256"),
        )
    if quality is not None and listening is not None:
        _check_equal(
            checks,
            "evidence_binding.blind_ab_baseline_sha256",
            quality.get("baseline_sha256"),
            listening.get("baseline_artifact_sha256"),
        )
        _check_equal(
            checks,
            "evidence_binding.blind_ab_candidate_sha256",
            quality.get("candidate_sha256"),
            listening.get("candidate_artifact_sha256"),
        )

    failures = [check for check in checks if not bool(check["passed"])]
    report: dict[str, object] = {
        "schema_version": 1,
        "gate_kind": "packaged_release_promotion",
        "passed": not failures,
        "policy": asdict(resolved_policy),
        "evidence": {
            "quality_gate": quality_source,
            "gold_audit": gold_source,
            "runtime_smoke": runtime_source,
            "blind_ab": listening_source,
        },
        "blind_ab_summary": listening_summary,
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "checks": checks,
        "failures": failures,
    }
    if output_json is not None:
        write_deterministic_json(output_json, report)
    return report


def canonical_json_sha256(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_deterministic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write stable UTF-8 JSON (sorted keys, LF, final newline)."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _audit_double_annotation(
    checks: list[dict[str, object]],
    recording_path: str,
    evidence: Mapping[str, object],
    manifest_root: Path,
    *,
    required: bool,
    require_within_root: bool = False,
) -> bool:
    if not evidence:
        if required:
            _check_condition(
                checks,
                f"{recording_path}.double_annotation",
                expected="adjudicated_human_evidence",
                actual="missing",
                passed=False,
                reason="double_annotation_missing",
            )
        return False
    before = _failure_count(checks)
    prefix = f"{recording_path}.double_annotation"
    _check_equal(checks, f"{prefix}.status", "adjudicated", evidence.get("status"))
    _check_equal(checks, f"{prefix}.creation_method", "human_manual", evidence.get("creation_method"))
    annotators = sorted({_text(value) for value in _as_sequence(evidence.get("annotator_ids")) if _text(value)})
    _check_condition(
        checks,
        f"{prefix}.annotator_ids",
        expected="at_least_two_distinct_humans",
        actual=annotators,
        passed=len(annotators) >= 2,
        reason="two_annotators_required",
    )
    _check_nonempty(checks, f"{prefix}.adjudicator_id", evidence.get("adjudicator_id"))
    _check_file_evidence(
        checks,
        f"{prefix}.adjudication",
        _as_mapping(evidence.get("adjudication")),
        manifest_root,
        require_within_root=require_within_root,
    )
    return _failure_count(checks) == before


def _audit_runtime_smoke(checks: list[dict[str, object]], runtime: Mapping[str, object]) -> None:
    _check_equal(checks, "runtime_smoke.schema_version", 1, runtime.get("schema_version"))
    _check_equal(checks, "runtime_smoke.purpose", "packaged_runtime_smoke", runtime.get("purpose"))
    _check_equal(checks, "runtime_smoke.passed", True, runtime.get("passed"))
    _check_nonempty(checks, "runtime_smoke.artifact_path", runtime.get("artifact_path"))
    _check_sha(checks, "runtime_smoke.artifact_sha256", runtime.get("artifact_sha256"))
    provenance = _as_mapping(runtime.get("provenance"))
    _check_equal(
        checks,
        "runtime_smoke.provenance.capture_method",
        "observed_process",
        provenance.get("capture_method"),
    )
    _check_nonempty(checks, "runtime_smoke.provenance.observed_by", provenance.get("observed_by"))
    _check_nonempty(checks, "runtime_smoke.provenance.observed_at", provenance.get("observed_at"))
    required_outputs = _as_sequence(runtime.get("required_outputs"))
    _check_condition(
        checks,
        "runtime_smoke.required_outputs.available",
        expected="one_or_more_nonempty_outputs",
        actual=len(required_outputs),
        passed=bool(required_outputs),
        reason="generated_output_evidence_missing",
    )
    invalid_outputs: list[int] = []
    for index, raw_output in enumerate(required_outputs):
        output = _as_mapping(raw_output)
        if (
            not _text(output.get("path"))
            or output.get("safe_within_working_dir") is not True
            or output.get("exists") is not True
            or output.get("nonempty") is not True
            or not _valid_sha256(output.get("sha256"))
            or not isinstance(output.get("size_bytes"), int)
            or isinstance(output.get("size_bytes"), bool)
            or int(output.get("size_bytes", 0)) <= 0
        ):
            invalid_outputs.append(index)
    _check_condition(
        checks,
        "runtime_smoke.required_outputs.valid",
        expected=[],
        actual=invalid_outputs,
        passed=bool(required_outputs) and not invalid_outputs,
        reason="generated_output_evidence_invalid",
    )
    smoke_checks = _as_sequence(runtime.get("checks"))
    _check_condition(
        checks,
        "runtime_smoke.checks.available",
        expected="one_or_more_observed_checks",
        actual=len(smoke_checks),
        passed=bool(smoke_checks),
        reason="runtime_checks_missing",
    )
    failed_names: list[str] = []
    for index, raw_check in enumerate(smoke_checks):
        item = _as_mapping(raw_check)
        name = _text(item.get("name")) or f"check_{index}"
        if item.get("passed") is not True:
            failed_names.append(name)
    _check_condition(
        checks,
        "runtime_smoke.checks.all_passed",
        expected=[],
        actual=failed_names,
        passed=bool(smoke_checks) and not failed_names,
        reason="runtime_check_failed",
    )


def _audit_blind_ab(
    checks: list[dict[str, object]],
    evidence: Mapping[str, object],
    policy: ReleaseGatePolicy,
) -> dict[str, object]:
    _check_equal(checks, "blind_ab.schema_version", 1, evidence.get("schema_version"))
    _check_equal(checks, "blind_ab.purpose", "human_blind_ab_evaluation", evidence.get("purpose"))
    _check_equal(checks, "blind_ab.passed", True, evidence.get("passed"))
    _check_nonempty(checks, "blind_ab.protocol_id", evidence.get("protocol_id"))
    _check_sha(checks, "blind_ab.protocol_sha256", evidence.get("protocol_sha256"))
    _check_sha(checks, "blind_ab.baseline_artifact_sha256", evidence.get("baseline_artifact_sha256"))
    _check_sha(checks, "blind_ab.candidate_artifact_sha256", evidence.get("candidate_artifact_sha256"))
    provenance = _as_mapping(evidence.get("provenance"))
    _check_equal(checks, "blind_ab.provenance.reference_kind", "human_listening", provenance.get("reference_kind"))
    _check_equal(
        checks,
        "blind_ab.provenance.collection_method",
        "human_blind_ab",
        provenance.get("collection_method"),
    )
    _check_equal(
        checks,
        "blind_ab.provenance.evidence_origin",
        "human_entered",
        provenance.get("evidence_origin"),
    )
    attestation = _as_mapping(evidence.get("human_attestation"))
    _check_equal(
        checks,
        "blind_ab.human_attestation.is_human_collected",
        True,
        attestation.get("is_human_collected"),
    )
    _check_nonempty(checks, "blind_ab.human_attestation.attested_by", attestation.get("attested_by"))
    _check_nonempty(checks, "blind_ab.human_attestation.attested_at", attestation.get("attested_at"))

    trials = _as_sequence(evidence.get("trials"))
    supplied_trials_sha = evidence.get("trials_sha256")
    _check_sha(checks, "blind_ab.trials_sha256", supplied_trials_sha)
    calculated_trials_sha = canonical_json_sha256(trials)
    _check_equal(checks, "blind_ab.trials_sha256.matches", calculated_trials_sha, supplied_trials_sha)

    listener_ids: set[str] = set()
    trial_ids: set[str] = set()
    duplicate_trials: list[str] = []
    invalid_trials: list[int] = []
    format_counts = {format_name: 0 for format_name in FORMATS}
    winner_counts = {"baseline": 0, "candidate": 0, "tie": 0}
    for index, raw_trial in enumerate(trials):
        trial = _as_mapping(raw_trial)
        trial_id = _text(trial.get("trial_id"))
        listener_id = _text(trial.get("listener_id"))
        format_name = _canonical_format(trial)
        winner = _text(trial.get("winner")).casefold()
        if trial_id in trial_ids and trial_id:
            duplicate_trials.append(trial_id)
        if trial_id:
            trial_ids.add(trial_id)
        if listener_id:
            listener_ids.add(listener_id)
        if (
            not trial_id
            or not listener_id
            or format_name not in FORMATS
            or winner not in winner_counts
        ):
            invalid_trials.append(index)
            continue
        format_counts[format_name] += 1
        winner_counts[winner] += 1
    _check_condition(
        checks,
        "blind_ab.trials.valid",
        expected={"invalid": [], "duplicates": []},
        actual={"invalid": invalid_trials, "duplicates": sorted(set(duplicate_trials))},
        passed=not invalid_trials and not duplicate_trials,
        reason="blind_trial_invalid_or_duplicate",
    )
    _check_condition(
        checks,
        "blind_ab.minimum_trials",
        expected=policy.minimum_blind_trials,
        actual=len(trials),
        passed=len(trials) >= policy.minimum_blind_trials,
        reason="blind_trial_minimum_not_met",
    )
    _check_condition(
        checks,
        "blind_ab.minimum_listeners",
        expected=policy.minimum_blind_listeners,
        actual=len(listener_ids),
        passed=len(listener_ids) >= policy.minimum_blind_listeners,
        reason="blind_listener_minimum_not_met",
    )
    for format_name in FORMATS:
        _check_condition(
            checks,
            f"blind_ab.coverage.{format_name}",
            expected=policy.minimum_blind_trials_per_format,
            actual=format_counts[format_name],
            passed=format_counts[format_name] >= policy.minimum_blind_trials_per_format,
            reason="blind_format_coverage_not_met",
        )
    decisive = winner_counts["baseline"] + winner_counts["candidate"]
    preference = winner_counts["candidate"] / decisive if decisive else 0.0
    _check_condition(
        checks,
        "blind_ab.candidate_preference_rate",
        expected=policy.minimum_candidate_preference_rate,
        actual=preference,
        passed=decisive > 0 and preference >= policy.minimum_candidate_preference_rate,
        reason="candidate_preference_threshold_not_met",
    )
    return {
        "trial_count": len(trials),
        "listener_count": len(listener_ids),
        "counts_by_format": format_counts,
        "winner_counts": winner_counts,
        "candidate_preference_rate": preference,
        "trials_sha256": calculated_trials_sha,
    }


def _safe_load_evidence(
    name: str,
    value: EvidenceInput | None,
) -> tuple[Mapping[str, object] | None, dict[str, object]]:
    if value is None:
        return None, {"status": "missing", "source": None, "sha256": None, "error": "evidence_missing"}
    try:
        if isinstance(value, Path):
            resolved = value.resolve()
            payload = _load_json_mapping(resolved)
            return payload, {
                "status": "loaded",
                "source": str(resolved),
                "sha256": _file_sha256(resolved),
                "error": None,
            }
        if isinstance(value, Mapping):
            payload = dict(value)
            return payload, {
                "status": "loaded",
                "source": "in_memory",
                "sha256": canonical_json_sha256(payload),
                "error": None,
            }
        raise PromotionGateError(f"{name} evidence must be a Path or mapping")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, {
            "status": "invalid",
            "source": str(value),
            "sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_evidence_loaded(
    checks: list[dict[str, object]],
    name: str,
    source: Mapping[str, object],
) -> None:
    _check_condition(
        checks,
        f"{name}.evidence_loaded",
        expected="loaded",
        actual=source.get("status"),
        passed=source.get("status") == "loaded",
        reason=_text(source.get("error")) or "evidence_missing_or_invalid",
    )


def _minimum_counts(stage: str, policy: GoldAuditPolicy) -> dict[str, int]:
    if stage == "locked":
        return {
            "train": policy.locked_minimum_train_per_format,
            "validation": policy.locked_minimum_validation_per_format,
            "locked_test": policy.locked_minimum_locked_test_per_format,
        }
    return {
        "train": policy.development_minimum_train_per_format,
        "validation": policy.development_minimum_validation_per_format,
        "locked_test": policy.development_minimum_locked_test_per_format,
    }


def _canonical_format(value: Mapping[str, object]) -> str:
    raw_format = _text(value.get("format")).casefold().replace("-", "_")
    language = _text(value.get("language", "ko")).casefold().replace("-", "_")
    if raw_format in FORMATS:
        return raw_format
    if language in {"ko", "kor", "korean", "ko_kr"} and raw_format in {"vcv", "cvvc"}:
        return f"korean_{raw_format}"
    return ""


def _check_file_evidence(
    checks: list[dict[str, object]],
    prefix: str,
    evidence: Mapping[str, object],
    root: Path,
    *,
    require_within_root: bool = False,
) -> bool:
    before = _failure_count(checks)
    raw_path = _text(evidence.get("path"))
    _check_nonempty(checks, f"{prefix}.path", raw_path)
    supplied_sha = evidence.get("sha256")
    _check_sha(checks, f"{prefix}.sha256", supplied_sha)
    resolved = (root / raw_path).resolve() if raw_path and not Path(raw_path).is_absolute() else Path(raw_path).resolve() if raw_path else None
    within_root = False
    if resolved is not None:
        try:
            resolved.relative_to(root.resolve())
            within_root = True
        except ValueError:
            within_root = False
    if require_within_root:
        _check_condition(
            checks,
            f"{prefix}.within_manifest_root",
            expected=True,
            actual=within_root,
            passed=within_root,
            reason="self_contained_evidence_path_escape",
        )
    exists = bool(resolved and resolved.is_file())
    _check_condition(
        checks,
        f"{prefix}.file_exists",
        expected=True,
        actual=exists,
        passed=exists,
        reason="evidence_file_missing",
    )
    if exists and _valid_sha256(supplied_sha):
        actual_sha = _file_sha256(resolved)  # type: ignore[arg-type]
        _check_equal(checks, f"{prefix}.sha256.matches", str(supplied_sha).casefold(), actual_sha)
    else:
        _check_condition(
            checks,
            f"{prefix}.sha256.matches",
            expected="verified_file_digest",
            actual=None,
            passed=False,
            reason="file_or_sha_unavailable",
        )
    return _failure_count(checks) == before


def _resolve_evidence_path(root: Path, evidence: Mapping[str, object]) -> Path | None:
    raw_path = _text(evidence.get("path"))
    if not raw_path:
        return None
    candidate = Path(raw_path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _audit_gold_provenance_files(
    checks: list[dict[str, object]],
    provenance: Mapping[str, object],
    metadata_files: Mapping[str, object],
    manifest_root: Path,
) -> None:
    required = {
        "REVIEW_PROTOCOL.md": "annotation_protocol_sha256",
        "source_inventory.csv": "source_inventory_sha256",
        "speaker_map.csv": "speaker_map_sha256",
        "license_review.csv": "license_review_sha256",
        "source_attestation.json": "source_attestation_sha256",
    }
    for name, digest_field in required.items():
        evidence = _as_mapping(metadata_files.get(name))
        file_ok = _check_file_evidence(
            checks,
            f"provenance.metadata_files.{name}",
            evidence,
            manifest_root,
            require_within_root=True,
        )
        _check_equal(
            checks,
            f"provenance.{digest_field}.matches_metadata_file",
            _text(evidence.get("sha256")).casefold(),
            _text(provenance.get(digest_field)).casefold(),
        )
        if name == "source_attestation.json" and file_ok:
            path = _resolve_evidence_path(manifest_root, evidence)
            try:
                attestation = _load_json_mapping(path) if path is not None else {}
            except Exception:
                attestation = {}
            _check_equal(checks, "provenance.source_attestation.status", "signed", attestation.get("status"))
            _check_equal(
                checks,
                "provenance.source_attestation.creation_method_claim",
                "human_manual",
                attestation.get("creation_method_claim"),
            )
            _check_nonempty(checks, "provenance.source_attestation.attestor_id", attestation.get("attestor_id"))
            _check_nonempty(checks, "provenance.source_attestation.signed_at", attestation.get("signed_at"))
            _check_equal(
                checks,
                "provenance.source_attestation.source_inventory_sha256",
                _text(provenance.get("source_inventory_sha256")).casefold(),
                _text(attestation.get("source_inventory_sha256")).casefold(),
            )

    raw_permissions = _as_sequence(provenance.get("permission_evidence"))
    _check_condition(
        checks,
        "provenance.permission_evidence.available",
        expected="one_or_more_verified_permission_files",
        actual=len(raw_permissions),
        passed=bool(raw_permissions),
        reason="permission_evidence_missing",
    )
    for index, raw in enumerate(raw_permissions):
        evidence = _as_mapping(raw)
        _check_nonempty(checks, f"provenance.permission_evidence[{index}].reviewer_id", evidence.get("reviewer_id"))
        _check_nonempty(checks, f"provenance.permission_evidence[{index}].bank_id", evidence.get("bank_id"))
        _check_file_evidence(
            checks,
            f"provenance.permission_evidence[{index}]",
            evidence,
            manifest_root,
            require_within_root=True,
        )


def _audit_gold_permission_coverage(
    checks: list[dict[str, object]],
    provenance: Mapping[str, object],
    expected_bank_ids: set[str],
) -> None:
    permission_rows = [
        _as_mapping(raw)
        for raw in _as_sequence(provenance.get("permission_evidence"))
    ]
    actual = [_text(row.get("bank_id")) for row in permission_rows if _text(row.get("bank_id"))]
    _check_condition(
        checks,
        "provenance.permission_evidence.bank_coverage",
        expected=sorted(expected_bank_ids),
        actual=sorted(actual),
        passed=set(actual) == expected_bank_ids and len(actual) == len(expected_bank_ids),
        reason="permission_bank_coverage_missing_extra_or_duplicate",
    )


def _audit_gold_oto_review_binding(
    checks: list[dict[str, object]],
    recording_path: str,
    recording: Mapping[str, object],
    review_evidence: Mapping[str, object],
    raw_anchor_rows: Sequence[object],
    manifest_root: Path,
) -> None:
    primary = _as_mapping(review_evidence.get("primary"))
    primary_ok = _check_file_evidence(
        checks,
        f"{recording_path}.oto_review_evidence.primary",
        primary,
        manifest_root,
        require_within_root=True,
    )
    primary_path = _resolve_evidence_path(manifest_root, primary)
    primary_payload: Mapping[str, object] = {}
    if primary_ok and primary_path is not None:
        try:
            primary_payload = _load_json_mapping(primary_path)
        except Exception:
            primary_payload = {}
    primary_rows = _audit_gold_review_payload_schema(
        checks,
        f"{recording_path}.oto_review_evidence.primary.payload",
        primary_payload,
        recording,
        primary,
        arm="primary",
    )
    _check_equal(
        checks,
        f"{recording_path}.oto_review_evidence.primary.anchor_sha256",
        primary.get("reviewed_anchor_sha256"),
        primary_payload.get("anchor_sha256"),
    )
    _check_equal(
        checks,
        f"{recording_path}.oto_review_evidence.primary.annotator_id",
        primary.get("annotator_id"),
        primary_payload.get("annotator_id"),
    )
    _check_nonempty(
        checks,
        f"{recording_path}.oto_review_evidence.primary.annotator_id.nonempty",
        primary.get("annotator_id"),
    )
    double_annotation = _as_mapping(recording.get("double_annotation"))
    if double_annotation:
        adjudication = _as_mapping(double_annotation.get("adjudication"))
        adjudication_path = _resolve_evidence_path(manifest_root, adjudication)
        try:
            adjudication_payload = (
                _load_json_mapping(adjudication_path) if adjudication_path is not None else {}
            )
        except Exception:
            adjudication_payload = {}
        authoritative_rows = _as_sequence(adjudication_payload.get("final_anchor_rows"))
        secondary = _as_mapping(review_evidence.get("secondary"))
        secondary_ok = _check_file_evidence(
            checks,
            f"{recording_path}.oto_review_evidence.secondary",
            secondary,
            manifest_root,
            require_within_root=True,
        )
        secondary_path = _resolve_evidence_path(manifest_root, secondary)
        try:
            secondary_payload = _load_json_mapping(secondary_path) if secondary_ok and secondary_path else {}
        except Exception:
            secondary_payload = {}
        secondary_rows = _audit_gold_review_payload_schema(
            checks,
            f"{recording_path}.oto_review_evidence.secondary.payload",
            secondary_payload,
            recording,
            secondary,
            arm="secondary",
        )
        _check_equal(
            checks,
            f"{recording_path}.oto_review_evidence.secondary.anchor_sha256",
            secondary.get("reviewed_anchor_sha256"),
            secondary_payload.get("anchor_sha256"),
        )
        _check_nonempty(
            checks,
            f"{recording_path}.oto_review_evidence.secondary.annotator_id.nonempty",
            secondary.get("annotator_id"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.primary_oto_review_sha256",
            primary.get("sha256"),
            adjudication_payload.get("primary_oto_review_sha256"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.primary_reviewed_anchor_sha256",
            primary.get("reviewed_anchor_sha256"),
            adjudication_payload.get("primary_reviewed_anchor_sha256"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.secondary_oto_review_sha256",
            secondary.get("sha256"),
            adjudication_payload.get("secondary_oto_review_sha256"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.secondary_reviewed_anchor_sha256",
            secondary.get("reviewed_anchor_sha256"),
            adjudication_payload.get("secondary_reviewed_anchor_sha256"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.schema_version",
            1,
            adjudication_payload.get("schema_version"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.purpose",
            "manual_oto_double_annotation_adjudication",
            adjudication_payload.get("purpose"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.status",
            "adjudicated",
            adjudication_payload.get("status"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.recording_id",
            recording.get("recording_id"),
            adjudication_payload.get("recording_id"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.wav_sha256",
            _as_mapping(recording.get("wav")).get("sha256"),
            adjudication_payload.get("wav_sha256"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.primary_annotator_id",
            primary.get("annotator_id"),
            adjudication_payload.get("primary_annotator_id"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.secondary_annotator_id",
            secondary.get("annotator_id"),
            adjudication_payload.get("secondary_annotator_id"),
        )
        _check_equal(
            checks,
            f"{recording_path}.adjudication.adjudicator_id",
            double_annotation.get("adjudicator_id"),
            adjudication_payload.get("adjudicator_id"),
        )
        wrapper_annotators = sorted(_text(value) for value in _as_sequence(double_annotation.get("annotator_ids")))
        _check_equal(
            checks,
            f"{recording_path}.adjudication.annotator_binding",
            sorted([_text(primary.get("annotator_id")), _text(secondary.get("annotator_id"))]),
            wrapper_annotators,
        )
        final_rows = _as_sequence(adjudication_payload.get("final_anchor_rows"))
        _check_condition(
            checks,
            f"{recording_path}.adjudication.final_anchor_reference_kind",
            expected="all_manual_oto_anchor_gold",
            actual=[_as_mapping(row).get("reference_kind") for row in final_rows],
            passed=bool(final_rows) and all(
                _as_mapping(row).get("reference_kind") == "manual_oto_anchor_gold"
                for row in final_rows
            ),
            reason="adjudicated_anchor_reference_kind_invalid",
        )
        _check_condition(
            checks,
            f"{recording_path}.oto_review_evidence.secondary.rows_available",
            expected="one_or_more_rows",
            actual=len(secondary_rows),
            passed=bool(secondary_rows),
            reason="secondary_review_rows_missing",
        )
    else:
        authoritative_rows = primary_rows
    expected_rows = _canonical_gold_anchor_rows(authoritative_rows)
    actual_rows = _canonical_gold_anchor_rows(raw_anchor_rows)
    _check_equal(
        checks,
        f"{recording_path}.oto_review_evidence.anchor_rows_match",
        expected_rows,
        actual_rows,
    )


def _audit_gold_review_payload_schema(
    checks: list[dict[str, object]],
    prefix: str,
    payload: Mapping[str, object],
    recording: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    arm: str,
) -> Sequence[object]:
    rows = _as_sequence(payload.get("rows"))
    _check_equal(checks, f"{prefix}.schema_version", 1, payload.get("schema_version"))
    _check_equal(
        checks,
        f"{prefix}.purpose",
        "manual_oto_reviewed_anchor_digest",
        payload.get("purpose"),
    )
    _check_equal(checks, f"{prefix}.status", "reviewed", payload.get("status"))
    _check_equal(checks, f"{prefix}.arm", arm, payload.get("arm"))
    _check_equal(
        checks,
        f"{prefix}.recording_id",
        recording.get("recording_id"),
        payload.get("recording_id"),
    )
    _check_equal(checks, f"{prefix}.bank_id", recording.get("bank_id"), payload.get("bank_id"))
    _check_equal(checks, f"{prefix}.wav_key", recording.get("wav_key"), payload.get("wav_key"))
    _check_equal(
        checks,
        f"{prefix}.source_wav_sha256",
        _as_mapping(recording.get("wav")).get("sha256"),
        payload.get("source_wav_sha256"),
    )
    canonical_payload = {
        "schema_version": 1,
        "purpose": "manual_oto_reviewed_anchor_digest",
        "arm": arm,
        "recording_id": payload.get("recording_id"),
        "bank_id": payload.get("bank_id"),
        "wav_key": payload.get("wav_key"),
        "source_wav_sha256": payload.get("source_wav_sha256"),
        "rows": rows,
    }
    calculated_digest = canonical_json_sha256(canonical_payload)
    _check_equal(checks, f"{prefix}.anchor_sha256.recomputed", calculated_digest, payload.get("anchor_sha256"))
    _check_equal(
        checks,
        f"{prefix}.reviewed_anchor_sha256.recomputed",
        calculated_digest,
        evidence.get("reviewed_anchor_sha256"),
    )
    return rows


def _canonical_gold_anchor_rows(rows: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in rows:
        row = _as_mapping(raw)
        anchors = _as_mapping(row.get("anchors"))
        try:
            output.append(
                {
                    "slot_id": _text(row.get("slot_id")),
                    "alias": _text(row.get("alias")),
                    "occurrence": int(row.get("occurrence", -1)),
                    "anchors": {
                        name: float(anchors[name])
                        for name in OTO_ANCHOR_SLOTS
                    },
                }
            )
        except (KeyError, TypeError, ValueError):
            return []
    output.sort(key=lambda row: str(row["slot_id"]))
    return output


def _normalize_stage(stage: str) -> str:
    normalized = _text(stage).casefold().replace("-", "_")
    if normalized not in {"development", "locked"}:
        raise PromotionGateError(f"unsupported promotion stage: {stage}")
    return normalized


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PromotionGateError(f"JSON root must be a mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_equal(
    checks: list[dict[str, object]],
    name: str,
    expected: object,
    actual: object,
) -> None:
    equal = actual is expected if isinstance(expected, bool) else actual == expected
    _check_condition(
        checks,
        name,
        expected=expected,
        actual=actual,
        passed=equal,
        reason="value_mismatch",
    )


def _check_nonempty(checks: list[dict[str, object]], name: str, value: object) -> None:
    _check_condition(
        checks,
        name,
        expected="non_empty_text",
        actual=value,
        passed=bool(_text(value)),
        reason="required_text_missing",
    )


def _check_sha(checks: list[dict[str, object]], name: str, value: object) -> None:
    _check_condition(
        checks,
        name,
        expected="lowercase_or_uppercase_sha256_hex",
        actual=value,
        passed=_valid_sha256(value),
        reason="sha256_missing_or_invalid",
    )


def _check_condition(
    checks: list[dict[str, object]],
    name: str,
    *,
    expected: object,
    actual: object,
    passed: bool,
    reason: str,
) -> None:
    checks.append(
        {
            "check": name,
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
            "reason": "" if passed else reason,
        }
    )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return value


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _failure_count(checks: Sequence[Mapping[str, object]]) -> int:
    return sum(not bool(check.get("passed")) for check in checks)


__all__ = [
    "FORMATS",
    "OTO_ANCHOR_SLOTS",
    "SPLITS",
    "GoldAuditPolicy",
    "PromotionGateError",
    "ReleaseGatePolicy",
    "audit_manual_gold_manifest",
    "canonical_json_sha256",
    "evaluate_release_gate",
    "write_deterministic_json",
]
