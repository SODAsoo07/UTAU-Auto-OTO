from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from universal_auto_oto.promotion_gate import (
    GoldAuditPolicy,
    ReleaseGatePolicy,
    audit_manual_gold_manifest,
    canonical_json_sha256,
    evaluate_release_gate,
    write_deterministic_json,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_file(path: Path, text: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path)}


def _write_fixture_wav(path: Path) -> dict[str, str]:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    return {"path": str(path), "sha256": _sha(path)}


def _write_fixture_annotation(path: Path, wav_sha256: str) -> dict[str, str]:
    payload = {
        "annotation_id": path.stem,
        "revision": 1,
        "wav": path.with_suffix(".wav").name,
        "wav_sha256": wav_sha256,
        "schema_version": 1,
        "inventory_sha256": "f" * 64,
        "transcript_source": "human_transcript",
        "label_source": "manual_gold",
        "phones": ["a"],
        "parent_sha256": None,
        "markers": [
            {
                "label": "usable_start",
                "time_ms": 0.0,
                "confidence": 1.0,
                "annotator": "annotator_a",
                "source": "human_manual",
                "phone_index": None,
                "left_phone_index": None,
                "right_phone_index": None,
                "note": "",
            },
            {
                "label": "usable_end",
                "time_ms": 500.0,
                "confidence": 1.0,
                "annotator": "annotator_a",
                "source": "human_manual",
                "phone_index": None,
                "left_phone_index": None,
                "right_phone_index": None,
                "note": "",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path)}


def _gold_manifest(tmp_path: Path, *, per_split: int = 1) -> tuple[Path, dict[str, object]]:
    recordings: list[dict[str, object]] = []
    for format_name in ("korean_vcv", "korean_cvvc"):
        for split in ("train", "validation", "locked_test"):
            for index in range(per_split):
                recording_id = f"{format_name}_{split}_{index}"
                wav = _write_fixture_wav(tmp_path / "files" / f"{recording_id}.wav")
                annotation = _write_fixture_annotation(
                    tmp_path / "files" / f"{recording_id}.phones.json",
                    wav["sha256"],
                )
                annotation.update(
                    {
                        "reference_kind": "manual_phone_gold",
                        "creation_method": "human_manual",
                        "annotator_id": "annotator_a",
                    }
                )
                wav["path"] = Path(wav["path"]).relative_to(tmp_path).as_posix()
                annotation["path"] = Path(annotation["path"]).relative_to(tmp_path).as_posix()
                row: dict[str, object] = {
                    "recording_id": recording_id,
                    "language": "ko",
                    "format": format_name.removeprefix("korean_"),
                    "root_speaker_id": f"speaker_{recording_id}",
                    "bank_id": f"bank_{recording_id}",
                    "wav_key": f"{recording_id}.wav",
                    "split": split,
                    "expected_slot_ids": ["slot_001"],
                    "wav": wav,
                    "annotation": annotation,
                    "oto_anchor_rows": [
                        {
                            "slot_id": "slot_001",
                            "reference_kind": "manual_oto_anchor_gold",
                            "alias": "- a",
                            "occurrence": 0,
                            "anchors": {
                                "offset_abs": 10.0,
                                "overlap_abs": 20.0,
                                "pre_abs": 30.0,
                                "fixed_abs": 40.0,
                                "cutoff_abs": 100.0,
                            },
                        }
                    ],
                }
                if split == "train" and index == 0:
                    adjudication = _write_fixture_file(
                        tmp_path / "files" / f"{recording_id}.adjudication.json",
                        "{}",
                    )
                    row["double_annotation"] = {
                        "status": "adjudicated",
                        "creation_method": "human_manual",
                        "annotator_ids": ["annotator_a", "annotator_b"],
                        "adjudicator_id": "adjudicator_c",
                        "adjudication": adjudication,
                    }
                    adjudication["path"] = Path(adjudication["path"]).relative_to(tmp_path).as_posix()
                recordings.append(row)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "manual_phone_gold_dataset",
        "dataset_id": "fixture-gold-v1",
        "provenance": {
            "reference_kind": "manual_phone_gold",
            "creation_method": "human_manual",
            "annotation_protocol_id": "ko-phone-gold-v1",
            "annotation_protocol_sha256": "a" * 64,
            "source_inventory_sha256": "b" * 64,
        },
        "recordings": recordings,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path, manifest


def _small_gold_policy() -> GoldAuditPolicy:
    return GoldAuditPolicy(
        development_minimum_train_per_format=1,
        development_minimum_validation_per_format=1,
        development_minimum_locked_test_per_format=1,
        locked_minimum_train_per_format=2,
        locked_minimum_validation_per_format=2,
        locked_minimum_locked_test_per_format=2,
        development_double_annotation_fraction=0.1,
        locked_double_annotation_fraction=0.1,
        development_minimum_double_annotated_per_format=1,
        locked_minimum_double_annotated_per_format=1,
    )


def test_gold_audit_accepts_verified_manual_corpus(tmp_path: Path) -> None:
    manifest_path, _ = _gold_manifest(tmp_path)
    output = tmp_path / "audit.json"

    result = audit_manual_gold_manifest(
        manifest_path,
        stage="development",
        output_json=output,
        policy=_small_gold_policy(),
    )

    assert result["passed"] is True
    assert result["speaker_split_leakage"] == []
    assert result["double_annotated_counts_by_format"] == {"korean_vcv": 1, "korean_cvvc": 1}
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_sha256"] == _sha(manifest_path)


def test_gold_audit_locked_profile_enforces_larger_counts(tmp_path: Path) -> None:
    manifest_path, _ = _gold_manifest(tmp_path)

    result = audit_manual_gold_manifest(manifest_path, stage="locked", policy=_small_gold_policy())

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "coverage.korean_vcv.train" in failed
    assert "coverage.korean_cvvc.locked_test" in failed


def test_gold_audit_locked_requires_explicit_model_unseen_release_eligibility(tmp_path: Path) -> None:
    manifest_path, manifest = _gold_manifest(tmp_path, per_split=2)

    exposed = audit_manual_gold_manifest(manifest_path, stage="locked", policy=_small_gold_policy())
    exposed_failures = {row["check"] for row in exposed["failures"]}
    assert "provenance.release_locked_eligible" in exposed_failures
    assert "provenance.prior_model_exposure_status" in exposed_failures

    provenance = manifest["provenance"]
    assert isinstance(provenance, dict)
    provenance["release_locked_eligible"] = True
    provenance["prior_model_exposure_status"] = "model_unseen"
    for row in manifest["recordings"]:
        row["release_locked_eligible"] = True
        row["prior_model_exposure_status"] = "model_unseen"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    sealed = audit_manual_gold_manifest(manifest_path, stage="locked", policy=_small_gold_policy())
    sealed_failures = {row["check"] for row in sealed["failures"]}
    assert "provenance.release_locked_eligible" not in sealed_failures
    assert "provenance.prior_model_exposure_status" not in sealed_failures
    assert "provenance.metadata_files.available" in sealed_failures


def test_gold_audit_rejects_split_leakage_non_manual_labels_and_missing_anchors(tmp_path: Path) -> None:
    manifest_path, manifest = _gold_manifest(tmp_path)
    rows = manifest["recordings"]
    assert isinstance(rows, list)
    rows[0]["root_speaker_id"] = "leaked_speaker"
    rows[1]["root_speaker_id"] = "leaked_speaker"
    rows[0]["annotation"]["reference_kind"] = "machine_silver"
    rows[0]["oto_anchor_rows"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_manual_gold_manifest(manifest_path, policy=_small_gold_policy())

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "speaker_disjoint_splits" in failed
    assert "recordings[0].annotation.reference_kind" in failed
    assert "recordings[0].oto_anchor_rows.available" in failed


def test_gold_audit_rejects_missing_exact_oto_identity_and_duplicate_occurrence(tmp_path: Path) -> None:
    manifest_path, manifest = _gold_manifest(tmp_path)
    rows = manifest["recordings"]
    assert isinstance(rows, list)
    rows[0].pop("wav_key")
    duplicate = deepcopy(rows[1]["oto_anchor_rows"][0])
    duplicate["slot_id"] = "slot_002"
    rows[1]["expected_slot_ids"] = ["slot_001", "slot_002"]
    rows[1]["oto_anchor_rows"].append(duplicate)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_manual_gold_manifest(manifest_path, policy=_small_gold_policy())

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "recordings[0].wav_key" in failed
    assert "recordings[1].oto_anchor_rows[1].exact_identity.unique" in failed


def _quality_evidence() -> dict[str, object]:
    gold = _gold_audit_evidence()
    return {
        "schema_version": 1,
        "gate_kind": "frozen_exe_manual_oto_anchor_quality",
        "stage": "locked",
        "passed": True,
        "failed_check_count": 0,
        "baseline_sha256": "1" * 64,
        "candidate_sha256": "4" * 64,
        "gold_audit_sha256": canonical_json_sha256(gold),
        "gold_manifest_sha256": gold["manifest_sha256"],
        "gold_anchor_inventory_sha256": gold["oto_anchor_inventory_sha256"],
    }


def _gold_audit_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate_kind": "manual_gold_dataset_audit",
        "stage": "locked",
        "passed": True,
        "failed_check_count": 0,
        "manifest_sha256": "3" * 64,
        "oto_anchor_inventory_sha256": "6" * 64,
    }


def _runtime_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "packaged_runtime_smoke",
        "passed": True,
        "artifact_path": "dist/UniversalAutoOTO.exe",
        "artifact_sha256": "4" * 64,
        "provenance": {
            "capture_method": "observed_process",
            "observed_by": "release_operator",
            "observed_at": "2026-08-12T12:00:00Z",
        },
        "required_outputs": [
            {
                "path": "oto.ini",
                "safe_within_working_dir": True,
                "exists": True,
                "nonempty": True,
                "size_bytes": 128,
                "sha256": "7" * 64,
            }
        ],
        "checks": [
            {"name": "process_started", "passed": True},
            {"name": "output_generated", "passed": True},
        ],
    }


def _blind_evidence() -> dict[str, object]:
    trials = [
        {"trial_id": "t1", "listener_id": "listener_1", "format": "korean_vcv", "winner": "candidate"},
        {"trial_id": "t2", "listener_id": "listener_2", "format": "korean_vcv", "winner": "candidate"},
        {"trial_id": "t3", "listener_id": "listener_1", "format": "korean_cvvc", "winner": "baseline"},
        {"trial_id": "t4", "listener_id": "listener_2", "format": "korean_cvvc", "winner": "tie"},
    ]
    return {
        "schema_version": 1,
        "purpose": "human_blind_ab_evaluation",
        "passed": True,
        "protocol_id": "ko-vcv-cvvc-blind-ab-v1",
        "protocol_sha256": "5" * 64,
        "baseline_artifact_sha256": "1" * 64,
        "candidate_artifact_sha256": "4" * 64,
        "trials_sha256": canonical_json_sha256(trials),
        "provenance": {
            "reference_kind": "human_listening",
            "collection_method": "human_blind_ab",
            "evidence_origin": "human_entered",
        },
        "human_attestation": {
            "is_human_collected": True,
            "attested_by": "listening_coordinator",
            "attested_at": "2026-08-12T13:00:00Z",
        },
        "trials": trials,
    }


def _small_release_policy() -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        minimum_blind_trials=4,
        minimum_blind_listeners=2,
        minimum_blind_trials_per_format=2,
        minimum_candidate_preference_rate=0.6,
    )


def test_release_gate_passes_only_with_all_locked_machine_and_human_evidence(tmp_path: Path) -> None:
    output = tmp_path / "release-gate.json"

    result = evaluate_release_gate(
        _quality_evidence(),
        _runtime_evidence(),
        _blind_evidence(),
        gold_audit=_gold_audit_evidence(),
        output_json=output,
        policy=_small_release_policy(),
    )

    assert result["passed"] is True
    assert result["blind_ab_summary"]["candidate_preference_rate"] == 2 / 3
    first = output.read_bytes()
    write_deterministic_json(output, result)
    assert output.read_bytes() == first


def test_release_gate_does_not_infer_missing_human_attestation() -> None:
    listening = deepcopy(_blind_evidence())
    listening.pop("human_attestation")

    result = evaluate_release_gate(
        _quality_evidence(),
        _runtime_evidence(),
        listening,
        gold_audit=_gold_audit_evidence(),
        policy=_small_release_policy(),
    )

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "blind_ab.human_attestation.is_human_collected" in failed
    assert "blind_ab.human_attestation.attested_by" in failed


def test_release_gate_rejects_unbound_candidate_and_gold_evidence() -> None:
    quality = _quality_evidence()
    quality["candidate_sha256"] = "9" * 64
    quality["gold_manifest_sha256"] = "8" * 64

    result = evaluate_release_gate(
        quality,
        _runtime_evidence(),
        _blind_evidence(),
        gold_audit=_gold_audit_evidence(),
        policy=_small_release_policy(),
    )

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "evidence_binding.candidate_runtime_artifact_sha256" in failed
    assert "evidence_binding.blind_ab_candidate_sha256" in failed
    assert "evidence_binding.gold_manifest_sha256" in failed


def test_release_gate_rejects_startup_only_runtime_without_generated_output() -> None:
    runtime = _runtime_evidence()
    runtime["required_outputs"] = []

    result = evaluate_release_gate(
        _quality_evidence(),
        runtime,
        _blind_evidence(),
        gold_audit=_gold_audit_evidence(),
        policy=_small_release_policy(),
    )

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "runtime_smoke.required_outputs.available" in failed
    assert "runtime_smoke.required_outputs.valid" in failed


def test_release_gate_fails_closed_for_missing_or_nonpassing_evidence() -> None:
    quality = _quality_evidence()
    quality["passed"] = False

    result = evaluate_release_gate(
        quality,
        None,
        None,
        gold_audit=None,
        policy=_small_release_policy(),
    )

    assert result["passed"] is False
    failed = {row["check"] for row in result["failures"]}
    assert "quality_gate.passed" in failed
    assert "gold_audit.evidence_loaded" in failed
    assert "runtime_smoke.evidence_loaded" in failed
    assert "blind_ab.evidence_loaded" in failed


def test_canonical_hash_is_key_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})
