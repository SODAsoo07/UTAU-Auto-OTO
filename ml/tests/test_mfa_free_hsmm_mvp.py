from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.alignment.hsmm_decoder import (
    HSMMStateFrameWindow,
    HSMMStateSpec,
    decode_segmental_hsmm,
    decode_segmental_hsmm_coarse_to_refine,
)
from core.generation.common.final_merge import merge_split_oto
from core.generation.common.validation_splitter import (
    SPLIT_ATTENTION_ONLY,
    SPLIT_CLEAN,
    SPLIT_FIX_REQUIRED,
    build_row_id,
    review_split_guard_reasons,
    split_oto_file,
    validate_oto_lines,
)
from core.mfa_free_oto.evidence_pack import (
    build_acoustic_evidence_pack,
    candidate_priors_from_evidence_pack,
    expected_phones_from_evidence_pack,
    filename_slots_from_evidence_pack,
    frame_posterior_from_evidence_pack,
    runtime_events_from_evidence_pack,
    summarize_event_candidate_reliability,
    validate_acoustic_evidence_pack,
)
from core.mfa_free_oto.hsmm_adapter import (
    _max_internal_gap_ms,
    _max_leading_gap_ms,
    build_hsmm_states_for_slots,
    decode_filename_slots_with_hsmm,
)
from core.mfa_free_oto.oto_adapter import (
    OtoAdapterConfig,
    OtoAnchor,
    expected_slots_for_template_rows,
    timeline_expected_slots_for_template_rows,
)
import core.mfa_free_oto.row_plan as row_plan_module
from core.mfa_free_oto.row_plan import (
    build_filename_row_plan,
    build_filename_slots,
    build_filename_template_rows,
    filename_phone_sequence_from_slots,
)
from core.mfa_free_oto.slot_viterbi import ExpectedSlot
from core.mfa_free_oto.types import EVENT_LABELS, FRAME_LABELS, FramePosterior
from core.mfa_free_oto.workflow import (
    _adapter_config_for_row_plan_record,
    _expected_phones,
    _hsmm_event_sequence_matches_expected,
    _runtime_events_for_hsmm_guard,
    _template_row_plan_context,
)
from core.model_context.timebase import TimebaseContract, cache_key_matches_timebase, timebase_from_mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_backed_comparison_json(
    path: Path,
    *,
    baseline_oto: Path,
    hsmm_oto: Path,
    hsmm_validation_jsonl: Path,
) -> None:
    from scripts.dev.mfa_free_oto_review import _compare_generated_oto_params

    row_param_delta = _compare_generated_oto_params(
        str(baseline_oto),
        str(hsmm_oto),
        hsmm_validation_path=str(hsmm_validation_jsonl),
        row_delta_limit=0,
    )
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "comparison": {
                    "row_count_delta": int(row_param_delta["hsmm_rows"]) - int(row_param_delta["baseline_rows"]),
                    "row_param_delta": row_param_delta,
                    "timeline_event_delta": {
                        "available": False,
                        "changed_events": 0,
                        "matched_events": 0,
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_validation_splitter_writes_disjoint_subsets_and_summary(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=b,0,10,-50,30,40",
                "a.wav=c,0,100,250,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="case",
        wav_durations_ms={"a.wav": 400.0},
    )

    assert result.counts["clean"] == 1
    assert result.counts["fix_required"] == 1
    assert result.counts["attention_only"] == 1
    assert (tmp_path / "split" / "case.fix_required.ini").read_text(encoding="utf-8").strip().startswith("a.wav=b")
    assert (tmp_path / "split" / "case.attention_only.ini").read_text(encoding="utf-8").strip().startswith("a.wav=c")
    assert (tmp_path / "split" / "case.clean.ini").read_text(encoding="utf-8").strip().startswith("a.wav=a")
    summary = json.loads((tmp_path / "split" / "case.summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["total"] == 3
    assert summary["generated_oto_sha256"] == _sha256_file(generated)
    assert summary["validation_jsonl_sha256"] == _sha256_file(tmp_path / "split" / "case.validation.jsonl")
    assert summary["output_sha256"]["fix_required"] == _sha256_file(tmp_path / "split" / "case.fix_required.ini")
    assert summary["output_sha256"]["attention_only"] == _sha256_file(tmp_path / "split" / "case.attention_only.ini")
    assert summary["output_sha256"]["clean"] == _sha256_file(tmp_path / "split" / "case.clean.ini")
    assert summary["output_sha256"]["review_all"] == _sha256_file(tmp_path / "split" / "case.review_all.ini")
    assert summary["output_sha256"]["validation_csv"] == _sha256_file(tmp_path / "split" / "case.validation.csv")
    assert "summary" not in summary["output_sha256"]
    assert "review_session" not in summary["output_sha256"]
    assert summary["output_paths"]["review_session"].endswith("case.review_session.json")
    assert summary["role_counts"] == {"cv": 2, "v": 1}
    assert summary["reason_counts"]["fix_required.too_short_fixed"] == 1
    assert summary["reason_counts_by_split"][SPLIT_FIX_REQUIRED]["fix_required.too_short_fixed"] == 1
    assert summary["reason_counts_by_split"][SPLIT_ATTENTION_ONLY]["attention.positive_cutoff"] == 1
    assert summary["metric_positive_row_counts"]["fixed_ms"] == 3
    assert summary["metric_min_values"]["fixed_ms"] == 10.0
    assert summary["metric_max_values"]["fixed_ms"] == 100.0
    review_session = json.loads((tmp_path / "split" / "case.review_session.json").read_text(encoding="utf-8"))
    assert review_session["state"] == "review_required"
    assert review_session["counts"]["total"] == 3
    assert review_session["lineage"]["validation_jsonl_sha256"] == _sha256_file(
        tmp_path / "split" / "case.validation.jsonl"
    )
    assert review_session["lineage"]["summary_sha256"] == _sha256_file(tmp_path / "split" / "case.summary.json")
    assert review_session["apply_policy"]["generated_ini_must_not_replace_voicebank_oto"] is True
    assert review_session["recommended_next_steps"][0]["id"] == "verify_review_session"
    assert "session-status" in review_session["recommended_next_steps"][0]["command_args"]
    assert review_session["recommended_next_steps"][1]["id"] == "prepare_review_edits"
    assert "prepare-edits" in review_session["recommended_next_steps"][1]["command_args"]
    assert review_session["recommended_next_steps"][2]["edited_paths"]["fix_required"].endswith(
        "case.edited_fix_required.ini"
    )
    assert review_session["recommended_next_steps"][3]["id"] == "merge_final"
    merge_args = review_session["recommended_next_steps"][3]["command_args"]
    assert "merge-session" in merge_args
    assert "--review-session" in merge_args
    assert str(tmp_path / "split" / "case.fix_required.ini") not in merge_args
    gate_args = review_session["recommended_next_steps"][4]["command_args"]
    assert "evaluate-session" in gate_args
    assert "--review-session" in gate_args
    assert "--validation-jsonl" not in gate_args
    assert review_session["recommended_next_steps"][5]["id"] == "install_final_dry_run"
    install_template_args = review_session["recommended_next_steps"][5]["command_template_args"]
    assert "install-session" in install_template_args
    assert "--review-session" in install_template_args
    assert "--target-oto" in install_template_args
    assert review_session["recommended_next_steps"][5]["dry_run_by_default"] is True


def test_review_split_guard_flags_large_fix_required_ratio_only_for_real_sessions():
    assert review_split_guard_reasons({"total": 381, "fix_required": 169, "attention_only": 198, "clean": 14}) == (
        "review_split_fix_required_ratio_high:169/381",
    )
    assert review_split_guard_reasons({"total": 509, "fix_required": 0, "attention_only": 509, "clean": 0}) == ()
    assert review_split_guard_reasons({"total": 3, "fix_required": 1, "attention_only": 1, "clean": 1}) == ()


def test_validation_splitter_keeps_zero_breath_template_rows_clean():
    records = validate_oto_lines(
        [
            "\u606f 2.wav=\u606f 2,0,0,0,0,0",
            "\u606f 2.wav=br2,0,0,0,0,0",
            "\u606f 2.wav=\u5438 2,0,0,0,0,0",
        ],
        wav_durations_ms={"\u606f 2.wav": 950.0},
        row_diagnostics={
            0: {"rule_based_inference_count": 1.0},
            1: {"rule_based_inference_count": 1.0},
            2: {"rule_based_inference_count": 1.0, "local_refine_low_margin_count": 1.0},
        },
    )

    assert [record.split for record in records] == [SPLIT_CLEAN, SPLIT_CLEAN, SPLIT_CLEAN]
    assert [record.reasons for record in records] == [("clean",), ("clean",), ("clean",)]


def test_validation_splitter_preserves_row_plan_context_in_records(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "dup.wav=a,0,100,-700,50,20",
                "dup.wav=a,120,100,-580,50,20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row_diagnostics = {
        0: {
            "row_plan": {
                "slot_index": 0,
                "left_slot_index": 0,
                "right_slot_index": 0,
                "role_family": "v",
                "source_row_index": 10,
                "source_timing_trusted": False,
            }
        },
        1: {
            "row_plan": {
                "slot_index": 2,
                "left_slot_index": 2,
                "right_slot_index": 2,
                "role_family": "v",
                "source_row_index": 11,
                "source_timing_trusted": False,
            }
        },
    }

    result = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="dup",
        wav_durations_ms={"dup.wav": 1000.0},
        row_diagnostics=row_diagnostics,
    )

    first, second = result.records
    assert first.slot_index == 0
    assert first.left_slot_index == 0
    assert first.right_slot_index == 0
    assert first.source_row_index == 10
    assert first.source_timing_trusted is False
    assert second.slot_index == 2
    assert second.source_row_index == 11
    assert first.row_id == build_row_id(
        wav="dup.wav",
        alias="a",
        line_index=0,
        occurrence_index=0,
        slot_index=0,
        source_row_index=10,
        role_family="v",
    )
    assert second.row_id == build_row_id(
        wav="dup.wav",
        alias="a",
        line_index=1,
        occurrence_index=1,
        slot_index=2,
        source_row_index=11,
        role_family="v",
    )

    validation_rows = [
        json.loads(line)
        for line in Path(result.output_paths["validation_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert validation_rows[0]["slot_index"] == 0
    assert validation_rows[1]["slot_index"] == 2
    assert validation_rows[1]["source_row_index"] == 11
    csv_text = Path(result.output_paths["validation_csv"]).read_text(encoding="utf-8-sig")
    assert "slot_index" in csv_text
    assert "source_row_index" in csv_text


def test_session_commands_reject_nonobject_or_empty_review_session(tmp_path):
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    cases = {
        "nonobject.review_session.json": ("[]\n", "json_root_not_object"),
        "empty.review_session.json": ("{}\n", "empty_review_session"),
    }
    command_args = [
        ["session-status"],
        ["prepare-edits"],
        ["merge-session"],
        ["evaluate-session"],
        ["install-session", "--target-oto", str(tmp_path / "voicebank" / "oto.ini")],
    ]

    for filename, (text, expected_error) in cases.items():
        session_path = tmp_path / filename
        session_path.write_text(text, encoding="utf-8")
        for args in command_args:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    *args,
                    "--review-session",
                    str(session_path),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
            )
            assert proc.returncode == 2, (args, proc.stdout, proc.stderr)
            payload = json.loads(proc.stdout)
            assert payload["command"] == args[0]
            assert payload["reason"] == "invalid_review_session"
            assert payload["error"] == expected_error


def test_final_merge_blocks_unresolved_fix_required_and_writes_final(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=b,0,10,-50,30,40",
                "a.wav=c,0,100,250,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(str(generated), str(tmp_path / "split"), name="case", wav_durations_ms={"a.wav": 400.0})
    final_path = tmp_path / "split" / "case.final.ini"

    blocked = merge_split_oto(
        validation_jsonl_path=split.output_paths["validation_jsonl"],
        final_path=str(final_path),
        edited_fix_required_path=split.output_paths["fix_required"],
    )
    assert not blocked.ok
    assert blocked.unresolved_row_ids
    assert not final_path.exists()

    edited_fix = tmp_path / "split" / "case.edited_fix_required.ini"
    wrong_fix = tmp_path / "split" / "case.wrong_fix_required.ini"
    wrong_fix.write_text("a.wav=wrong,0,90,-260,45,20\n", encoding="utf-8")
    wrong_merge = merge_split_oto(
        validation_jsonl_path=split.output_paths["validation_jsonl"],
        final_path=str(tmp_path / "split" / "case.wrong.final.ini"),
        edited_fix_required_path=str(wrong_fix),
    )
    assert not wrong_merge.ok
    assert wrong_merge.merge_input_errors
    wrong_report = json.loads(Path(wrong_merge.report_path).read_text(encoding="utf-8"))
    assert wrong_report["merge_input_error_count"] == 1
    assert wrong_report["merge_input_errors"][0].startswith("fix_required.row_mismatch:")

    edited_fix.write_text("a.wav=b,0,90,-260,45,20\n", encoding="utf-8")
    merged = merge_split_oto(
        validation_jsonl_path=split.output_paths["validation_jsonl"],
        final_path=str(final_path),
        edited_fix_required_path=str(edited_fix),
    )
    assert merged.ok
    merge_report = json.loads(Path(merged.report_path).read_text(encoding="utf-8"))
    assert merge_report["schema_version"] == 2
    assert Path(merge_report["validation_jsonl_path"]).resolve() == Path(split.output_paths["validation_jsonl"]).resolve()
    assert len(merge_report["validation_jsonl_sha256"]) == 64
    assert merge_report["merged_row_count"] == 3
    merged_rows = merge_report["merged_rows"]
    assert [row["row_id"] for row in merged_rows] == [record.row_id for record in split.records]
    assert [row["output_source"] for row in merged_rows] == [
        "clean",
        "edited_fix_required",
        "attention_only_original",
    ]
    assert merged_rows[0]["slot_index"] == -1
    assert merged_rows[0]["source_row_index"] == -1
    assert merged_rows[0]["source_timing_trusted"] is False
    assert merged_rows[1]["line_index"] == 1
    assert merged_rows[1]["changed_from_validation"] is True
    assert merged_rows[1]["line_sha256"] == hashlib.sha256(
        "a.wav=b,0,90,-260,45,20".encode("utf-8")
    ).hexdigest()
    assert merged_rows[1]["validation_raw_line_sha256"] == hashlib.sha256(
        "a.wav=b,0,10,-50,30,40".encode("utf-8")
    ).hexdigest()
    assert merged_rows[2]["changed_from_validation"] is False
    final_lines = final_path.read_text(encoding="utf-8").splitlines()
    assert final_lines == [
        "a.wav=a,0,100,-350,50,25",
        "a.wav=b,0,90,-260,45,20",
        "a.wav=c,0,100,250,50,25",
    ]
    assert merged.attention_only_acknowledged is False
    assert merged.attention_only_unacknowledged_count == 1

    acknowledged_final = tmp_path / "split" / "case.acknowledged.final.ini"
    acknowledged = merge_split_oto(
        validation_jsonl_path=split.output_paths["validation_jsonl"],
        final_path=str(acknowledged_final),
        edited_fix_required_path=str(edited_fix),
        acknowledge_attention_only=True,
    )
    assert acknowledged.ok
    assert acknowledged.attention_only_acknowledged is True
    assert acknowledged.attention_only_unacknowledged_count == 0


def test_final_merge_rejects_invalid_validation_jsonl_without_writing_final(tmp_path):
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    missing_validation = tmp_path / "missing.validation.jsonl"
    missing_final = tmp_path / "missing.final.ini"
    missing_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(missing_validation),
            "--final",
            str(missing_final),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_proc.returncode == 2
    missing_payload = json.loads(missing_proc.stdout)
    assert missing_payload["validation_errors"] == ["validation.file_missing"]
    assert not missing_final.exists()

    empty_validation = tmp_path / "empty.validation.jsonl"
    empty_validation.write_text("", encoding="utf-8")
    empty_final = tmp_path / "empty.final.ini"
    empty_result = merge_split_oto(
        validation_jsonl_path=str(empty_validation),
        final_path=str(empty_final),
    )
    assert not empty_result.ok
    assert empty_result.validation_errors == ("validation.empty",)
    assert not empty_final.exists()
    empty_report = json.loads(Path(empty_result.report_path).read_text(encoding="utf-8"))
    assert empty_report["validation_error_count"] == 1

    future_schema_validation = tmp_path / "future.validation.jsonl"
    future_schema_validation.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "row_id": "row_000000_future",
                "line_index": 0,
                "raw_line": "a.wav=a,0,100,-350,50,25",
                "wav": "a.wav",
                "alias": "a",
                "occurrence_index": 0,
                "split": "clean",
                "reasons": ["clean"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    stale_final = tmp_path / "future.final.ini"
    stale_result = merge_split_oto(
        validation_jsonl_path=str(future_schema_validation),
        final_path=str(stale_final),
    )
    assert not stale_result.ok
    assert stale_result.validation_errors == ("validation.schema_version_unsupported",)
    assert not stale_final.exists()

    malformed_validation = tmp_path / "malformed.validation.jsonl"
    malformed_validation.write_text("{bad json}\n", encoding="utf-8")
    gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(malformed_validation),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_proc.returncode == 2
    gate_payload = json.loads(gate_proc.stdout)
    assert gate_payload["metrics"]["validation"]["validation_errors"] == ["validation.json_invalid"]
    assert "validation_jsonl_valid" in gate_payload["gate_report"]["failed_checks"]
    assert "validation_jsonl_invalid" in gate_payload["warnings"]


def test_evaluate_gate_requires_attention_only_acknowledgement_for_apply(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=c,0,100,250,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(str(generated), str(tmp_path / "split"), name="case", wav_durations_ms={"a.wav": 400.0})
    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    final_path = tmp_path / "case.final.ini"
    merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--final",
            str(final_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_proc.returncode == 0, merge_proc.stderr
    merge_payload = json.loads(merge_proc.stdout)
    assert merge_payload["attention_only_unacknowledged_count"] == 1

    gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_proc.returncode == 2
    gate_payload = json.loads(gate_proc.stdout)
    assert gate_payload["gate_report"]["apply_candidate"] is False
    assert "attention_only_acknowledged_or_absent" in gate_payload["gate_report"]["failed_checks"]
    assert "attention_only_unacknowledged" in gate_payload["warnings"]

    acknowledged_final = tmp_path / "case.acknowledged.final.ini"
    acknowledged_merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--final",
            str(acknowledged_final),
            "--acknowledge-attention-only",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert acknowledged_merge_proc.returncode == 0, acknowledged_merge_proc.stderr
    acknowledged_merge = json.loads(acknowledged_merge_proc.stdout)
    assert acknowledged_merge["attention_only_unacknowledged_count"] == 0

    acknowledged_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(acknowledged_merge["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert acknowledged_gate_proc.returncode == 0, acknowledged_gate_proc.stdout
    assert json.loads(acknowledged_gate_proc.stdout)["gate_report"]["apply_candidate"] is True


def test_install_final_is_dry_run_by_default_and_requires_matching_gate_hash(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=i,120,100,-300,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(str(generated), str(tmp_path / "split"), name="case", wav_durations_ms={"a.wav": 500.0})
    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    final_path = tmp_path / "case.final.ini"
    target_oto = tmp_path / "무생물" / "oto.ini"
    target_oto.parent.mkdir()
    target_oto.write_text("old.wav=old,0,100,-200,50,25\n", encoding="utf-8")

    merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--final",
            str(final_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_proc.returncode == 0, merge_proc.stderr
    merge_payload = json.loads(merge_proc.stdout)
    gate_report = tmp_path / "gate.json"
    gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(gate_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_proc.returncode == 0, gate_proc.stderr

    invalid_comparison = tmp_path / "invalid_comparison.json"
    invalid_comparison.write_text("{not-json\n", encoding="utf-8")
    invalid_comparison_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(invalid_comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_comparison_gate_proc.returncode == 2
    invalid_comparison_gate = json.loads(invalid_comparison_gate_proc.stdout)
    assert invalid_comparison_gate["metrics"]["comparison"]["comparison_json_invalid"] is True
    assert "invalid_comparison_json" in invalid_comparison_gate["warnings"]
    assert "comparison_available" in invalid_comparison_gate["gate_report"]["failed_checks"]

    invalid_merge_report = tmp_path / "invalid.merge_report.json"
    invalid_merge_report.write_text("{not-json\n", encoding="utf-8")
    invalid_merge_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(invalid_merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_merge_gate_proc.returncode == 2
    invalid_merge_gate = json.loads(invalid_merge_gate_proc.stdout)
    assert invalid_merge_gate["metrics"]["merge"]["merge_report_invalid"] is True
    assert "invalid_merge_report" in invalid_merge_gate["warnings"]
    assert "merge_report_available" in invalid_merge_gate["gate_report"]["failed_checks"]

    invalid_quality_report = tmp_path / "invalid_quality_evidence.json"
    invalid_quality_report.write_text("{not-json\n", encoding="utf-8")
    invalid_quality_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(invalid_quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_quality_gate_proc.returncode == 2
    invalid_quality_gate = json.loads(invalid_quality_gate_proc.stdout)
    assert invalid_quality_gate["metrics"]["quality_evidence"]["reason"] == "invalid_quality_evidence_report"
    assert "invalid_quality_evidence_report" in invalid_quality_gate["warnings"]
    assert "quality_evidence_report_valid" in invalid_quality_gate["gate_report"]["failed_checks"]

    nonobject_comparison = tmp_path / "nonobject_comparison.json"
    nonobject_comparison.write_text("[]\n", encoding="utf-8")
    nonobject_comparison_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(nonobject_comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_comparison_gate_proc.returncode == 2
    nonobject_comparison_gate = json.loads(nonobject_comparison_gate_proc.stdout)
    assert nonobject_comparison_gate["metrics"]["comparison"]["comparison_json_invalid"] is True
    assert nonobject_comparison_gate["metrics"]["comparison"]["comparison_json_error"] == "json_root_not_object"

    nonobject_merge_report = tmp_path / "nonobject.merge_report.json"
    nonobject_merge_report.write_text("[]\n", encoding="utf-8")
    nonobject_merge_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(nonobject_merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_merge_gate_proc.returncode == 2
    nonobject_merge_gate = json.loads(nonobject_merge_gate_proc.stdout)
    assert nonobject_merge_gate["metrics"]["merge"]["merge_report_invalid"] is True
    assert nonobject_merge_gate["metrics"]["merge"]["merge_report_error"] == "json_root_not_object"

    nonobject_quality_report = tmp_path / "nonobject_quality_evidence.json"
    nonobject_quality_report.write_text("[]\n", encoding="utf-8")
    nonobject_quality_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "listening",
            "--quality-evidence-report",
            str(nonobject_quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_quality_gate_proc.returncode == 2
    nonobject_quality_gate = json.loads(nonobject_quality_gate_proc.stdout)
    assert nonobject_quality_gate["metrics"]["quality_evidence"]["reason"] == "invalid_quality_evidence_report"
    assert nonobject_quality_gate["metrics"]["quality_evidence"]["error"] == "json_root_not_object"

    empty_comparison = tmp_path / "empty_comparison.json"
    empty_comparison.write_text("{}\n", encoding="utf-8")
    empty_comparison_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(empty_comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_comparison_gate_proc.returncode == 2
    empty_comparison_gate = json.loads(empty_comparison_gate_proc.stdout)
    assert empty_comparison_gate["metrics"]["comparison"]["comparison_json_invalid"] is True
    assert empty_comparison_gate["metrics"]["comparison"]["comparison_json_error"] == "empty_comparison_json"

    empty_merge_report = tmp_path / "empty.merge_report.json"
    empty_merge_report.write_text("{}\n", encoding="utf-8")
    empty_merge_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(empty_merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_merge_gate_proc.returncode == 2
    empty_merge_gate = json.loads(empty_merge_gate_proc.stdout)
    assert empty_merge_gate["metrics"]["merge"]["merge_report_invalid"] is True
    assert empty_merge_gate["metrics"]["merge"]["merge_report_error"] == "empty_merge_report"

    empty_quality_report = tmp_path / "empty_quality_evidence.json"
    empty_quality_report.write_text("{}\n", encoding="utf-8")
    empty_quality_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "listening",
            "--quality-evidence-report",
            str(empty_quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_quality_gate_proc.returncode == 2
    empty_quality_gate = json.loads(empty_quality_gate_proc.stdout)
    assert empty_quality_gate["metrics"]["quality_evidence"]["reason"] == "invalid_quality_evidence_report"
    assert empty_quality_gate["metrics"]["quality_evidence"]["error"] == "empty_quality_evidence_report"

    invalid_merge_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(invalid_merge_report),
            "--out",
            str(tmp_path / "invalid_merge_quality.json"),
            "--evidence-type",
            "listening",
            "--passed",
            "--reviewed-rows",
            "2",
            "--listening-summary",
            "pytest listening smoke passed",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_merge_quality_proc.returncode == 2
    invalid_merge_quality = json.loads(invalid_merge_quality_proc.stdout)
    assert invalid_merge_quality["reason"] == "invalid_merge_report"
    assert "error" in invalid_merge_quality

    nonobject_merge_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(nonobject_merge_report),
            "--out",
            str(tmp_path / "nonobject_merge_quality.json"),
            "--evidence-type",
            "listening",
            "--passed",
            "--reviewed-rows",
            "2",
            "--listening-summary",
            "pytest listening smoke passed",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_merge_quality_proc.returncode == 2
    nonobject_merge_quality = json.loads(nonobject_merge_quality_proc.stdout)
    assert nonobject_merge_quality["reason"] == "invalid_merge_report"
    assert nonobject_merge_quality["error"] == "json_root_not_object"

    empty_merge_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(empty_merge_report),
            "--out",
            str(tmp_path / "empty_merge_quality.json"),
            "--evidence-type",
            "listening",
            "--passed",
            "--reviewed-rows",
            "2",
            "--listening-summary",
            "pytest listening smoke passed",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_merge_quality_proc.returncode == 2
    empty_merge_quality = json.loads(empty_merge_quality_proc.stdout)
    assert empty_merge_quality["reason"] == "invalid_merge_report"
    assert empty_merge_quality["error"] == "empty_merge_report"

    invalid_gate_report = tmp_path / "invalid_gate.json"
    invalid_gate_report.write_text("{not-json\n", encoding="utf-8")
    invalid_gate_install_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(invalid_gate_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_gate_install_proc.returncode == 2
    invalid_gate_install = json.loads(invalid_gate_install_proc.stdout)
    assert invalid_gate_install["reason"] == "invalid_gate_report"
    assert "error" in invalid_gate_install
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    nonobject_gate_report = tmp_path / "nonobject_gate.json"
    nonobject_gate_report.write_text("[]\n", encoding="utf-8")
    nonobject_gate_install_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(nonobject_gate_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_gate_install_proc.returncode == 2
    nonobject_gate_install = json.loads(nonobject_gate_install_proc.stdout)
    assert nonobject_gate_install["reason"] == "invalid_gate_report"
    assert nonobject_gate_install["error"] == "json_root_not_object"
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    default_without_report_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_without_report_proc.returncode == 2
    default_without_report = json.loads(default_without_report_proc.stdout)
    assert default_without_report["gate_report"]["default_on_candidate"] is False
    assert "quality_evidence_report_valid" in default_without_report["gate_report"]["failed_checks"]
    assert "quality_evidence_report_missing" in default_without_report["warnings"]

    missing_manual_reference_report = tmp_path / "missing_manual_reference_quality.json"
    missing_manual_reference_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(missing_manual_reference_report),
            "--evidence-type",
            "manual-reference",
            "--passed",
            "--reviewed-rows",
            "2",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_manual_reference_proc.returncode == 2
    missing_manual_reference = json.loads(missing_manual_reference_proc.stdout)
    assert "manual_reference_oto_required" in missing_manual_reference["failure_reasons"]
    assert missing_manual_reference_report.is_file()

    manual_reference_oto = tmp_path / "manual_reference.ini"
    manual_reference_oto.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")
    quality_report = tmp_path / "quality_evidence.json"
    record_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(quality_report),
            "--evidence-type",
            "manual-reference",
            "--passed",
            "--reviewed-rows",
            "2",
            "--reviewer",
            "pytest",
            "--manual-reference-oto",
            str(manual_reference_oto),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert record_quality_proc.returncode == 0, record_quality_proc.stderr
    recorded_quality = json.loads(record_quality_proc.stdout)
    assert recorded_quality["ok"] is True
    assert recorded_quality["final_sha256"]
    assert recorded_quality["manual_reference_oto_sha256"]
    assert recorded_quality["manual_reference_comparison"]["available"] is True
    assert recorded_quality["manual_reference_comparison"]["matched_rows"] == 2
    assert quality_report.is_file()

    tampered_quality_merge_report = tmp_path / "tampered_quality.merge_report.json"
    tampered_quality_merge = json.loads(Path(merge_payload["report_path"]).read_text(encoding="utf-8"))
    tampered_quality_merge["merged_rows"][0]["line_sha256"] = "0" * 64
    tampered_quality_merge_report.write_text(
        json.dumps(tampered_quality_merge, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_quality_report = tmp_path / "tampered_quality_evidence.json"
    tampered_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(tampered_quality_merge_report),
            "--out",
            str(tampered_quality_report),
            "--evidence-type",
            "manual-reference",
            "--passed",
            "--reviewed-rows",
            "2",
            "--manual-reference-oto",
            str(manual_reference_oto),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert tampered_quality_proc.returncode == 2
    tampered_quality = json.loads(tampered_quality_proc.stdout)
    assert "merge_rows_lineage_current_mismatch" in tampered_quality["failure_reasons"]
    assert tampered_quality["merge_metrics"]["merged_rows_current_match"] is False
    assert tampered_quality_report.is_file()

    stale_manual_quality_report = tmp_path / "stale_manual_quality_evidence.json"
    stale_manual_quality = dict(recorded_quality)
    stale_manual_quality.pop("manual_reference_oto", None)
    stale_manual_quality.pop("manual_reference_comparison", None)
    stale_manual_quality_report.write_text(json.dumps(stale_manual_quality), encoding="utf-8")
    stale_manual_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(stale_manual_quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_manual_gate_proc.returncode == 2
    stale_manual_gate = json.loads(stale_manual_gate_proc.stdout)
    stale_manual_reasons = stale_manual_gate["metrics"]["quality_evidence"]["failure_reasons"]
    assert "manual_reference_oto_required" in stale_manual_reasons
    assert "manual_reference_comparison_unavailable" in stale_manual_reasons

    original_manual_reference_text = manual_reference_oto.read_text(encoding="utf-8")
    manual_reference_oto.write_text(original_manual_reference_text + "\n", encoding="utf-8")
    hash_mismatch_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert hash_mismatch_gate_proc.returncode == 2
    hash_mismatch_gate = json.loads(hash_mismatch_gate_proc.stdout)
    hash_mismatch_reasons = hash_mismatch_gate["metrics"]["quality_evidence"]["failure_reasons"]
    assert "manual_reference_oto_sha256_mismatch" in hash_mismatch_reasons
    assert hash_mismatch_gate["metrics"]["quality_evidence"]["manual_reference_oto_sha256_match"] is False
    manual_reference_oto.write_text(original_manual_reference_text, encoding="utf-8")

    manual_reference_oto.write_text(
        original_manual_reference_text.replace("a.wav=a,0,", "a.wav=a,999,"),
        encoding="utf-8",
    )
    mutated_reference_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert mutated_reference_gate_proc.returncode == 2
    mutated_reference_gate = json.loads(mutated_reference_gate_proc.stdout)
    mutated_reasons = mutated_reference_gate["metrics"]["quality_evidence"]["failure_reasons"]
    assert "manual_reference_recompare_max_delta_exceeded" in mutated_reasons
    assert (
        mutated_reference_gate["metrics"]["quality_evidence"]["manual_reference_recompare_max_delta_actual_ms"]
        > 250
    )
    manual_reference_oto.write_text(original_manual_reference_text, encoding="utf-8")

    bad_manual_reference_oto = tmp_path / "bad_manual_reference.ini"
    bad_manual_reference_oto.write_text(
        final_path.read_text(encoding="utf-8").replace("a.wav=a,0,", "a.wav=a,999,"),
        encoding="utf-8",
    )
    bad_manual_quality_report = tmp_path / "bad_manual_quality_evidence.json"
    bad_manual_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(bad_manual_quality_report),
            "--evidence-type",
            "manual-reference",
            "--passed",
            "--reviewed-rows",
            "2",
            "--manual-reference-oto",
            str(bad_manual_reference_oto),
            "--manual-reference-max-delta-ms",
            "250",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert bad_manual_quality_proc.returncode == 2
    bad_manual_quality = json.loads(bad_manual_quality_proc.stdout)
    assert "manual_reference_max_delta_exceeded" in bad_manual_quality["failure_reasons"]
    assert bad_manual_quality["manual_reference_comparison"]["changed_rows"] >= 1
    assert bad_manual_quality_report.is_file()

    listening_quality_report = tmp_path / "listening_quality_evidence.json"
    listening_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(listening_quality_report),
            "--evidence-type",
            "listening",
            "--passed",
            "--reviewed-rows",
            "2",
            "--listening-summary",
            "pytest listening smoke passed",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert listening_quality_proc.returncode == 0, listening_quality_proc.stderr
    assert json.loads(listening_quality_proc.stdout)["listening_summary"] == "pytest listening smoke passed"

    stale_listening_quality_report = tmp_path / "stale_listening_quality_evidence.json"
    stale_listening_quality_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_type": "listening",
                "passed": True,
                "critical_failures": 0,
                "reviewed_rows": 2,
                "final_sha256": recorded_quality["final_sha256"],
            }
        ),
        encoding="utf-8",
    )
    stale_listening_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "listening",
            "--quality-evidence-report",
            str(stale_listening_quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_listening_gate_proc.returncode == 2
    stale_listening_gate = json.loads(stale_listening_gate_proc.stdout)
    assert "listening_summary_required" in stale_listening_gate["metrics"]["quality_evidence"]["failure_reasons"]

    failed_quality_report = tmp_path / "failed_quality_evidence.json"
    failed_quality_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "record-quality-evidence",
            "--final",
            str(final_path),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--out",
            str(failed_quality_report),
            "--evidence-type",
            "listening",
            "--failed",
            "--critical-failures",
            "1",
            "--reviewed-rows",
            "2",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert failed_quality_proc.returncode == 2
    failed_quality = json.loads(failed_quality_proc.stdout)
    assert failed_quality["ok"] is False
    assert "quality_evidence_not_passed" in failed_quality["failure_reasons"]
    assert "listening_summary_required" in failed_quality["failure_reasons"]
    assert failed_quality_report.is_file()

    default_gate_report = tmp_path / "default_gate.json"
    default_with_report_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(quality_report),
            "--out",
            str(default_gate_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_with_report_proc.returncode == 0, default_with_report_proc.stderr
    default_with_report = json.loads(default_with_report_proc.stdout)
    assert default_with_report["gate_report"]["default_on_candidate"] is True
    assert default_with_report["metrics"]["quality_evidence"]["final_sha256_match"] is True

    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    original_validation_text = validation_jsonl.read_text(encoding="utf-8")
    alternate_validation_jsonl = tmp_path / "alternate.validation.jsonl"
    alternate_validation_jsonl.write_text(original_validation_text, encoding="utf-8")
    alternate_validation_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(alternate_validation_jsonl),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert alternate_validation_gate_proc.returncode == 2
    alternate_validation_gate = json.loads(alternate_validation_gate_proc.stdout)
    assert "merge_validation_jsonl_current_match" in alternate_validation_gate["gate_report"]["failed_checks"]
    assert "merge_validation_jsonl_current_mismatch" in alternate_validation_gate["warnings"]
    assert alternate_validation_gate["metrics"]["merge"]["validation_jsonl_path_current_match"] is False
    assert alternate_validation_gate["metrics"]["merge"]["validation_jsonl_sha256_current_match"] is True

    validation_jsonl.write_text(original_validation_text + "\n", encoding="utf-8")
    stale_validation_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(validation_jsonl),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_validation_gate_proc.returncode == 2
    stale_validation_gate = json.loads(stale_validation_gate_proc.stdout)
    assert "merge_validation_jsonl_current_match" in stale_validation_gate["gate_report"]["failed_checks"]
    assert "merge_validation_jsonl_current_mismatch" in stale_validation_gate["warnings"]
    assert stale_validation_gate["metrics"]["merge"]["validation_jsonl_path_current_match"] is True
    assert stale_validation_gate["metrics"]["merge"]["validation_jsonl_sha256_current_match"] is False
    validation_jsonl.write_text(original_validation_text, encoding="utf-8")

    stale_comparison = tmp_path / "stale_comparison.json"
    stale_comparison_payload = json.loads(comparison.read_text(encoding="utf-8"))
    stale_comparison_payload["comparison"]["row_param_delta"]["hsmm_validation_sha256"] = "0" * 64
    stale_comparison.write_text(json.dumps(stale_comparison_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    stale_comparison_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(validation_jsonl),
            "--comparison-json",
            str(stale_comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_comparison_gate_proc.returncode == 2
    stale_comparison_gate = json.loads(stale_comparison_gate_proc.stdout)
    assert "comparison_validation_jsonl_current_match" in stale_comparison_gate["gate_report"]["failed_checks"]
    assert "comparison_validation_jsonl_current_mismatch" in stale_comparison_gate["warnings"]
    assert stale_comparison_gate["metrics"]["comparison"]["validation_jsonl_path_current_match"] is True
    assert stale_comparison_gate["metrics"]["comparison"]["validation_jsonl_sha256_current_match"] is False
    assert stale_comparison_gate["metrics"]["merge"]["validation_jsonl_current_match"] is True

    original_final_text = final_path.read_text(encoding="utf-8")
    final_path.write_text("a.wav=a,999,100,-350,50,25\n", encoding="utf-8")
    stale_final_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--quality-evidence",
            "manual-reference",
            "--quality-evidence-report",
            str(quality_report),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_final_gate_proc.returncode == 2
    stale_final_gate = json.loads(stale_final_gate_proc.stdout)
    assert stale_final_gate["gate_report"]["apply_candidate"] is False
    assert "merge_final_sha256_current_match" in stale_final_gate["gate_report"]["failed_checks"]
    assert "merge_final_sha256_current_mismatch" in stale_final_gate["warnings"]
    assert stale_final_gate["metrics"]["merge"]["final_sha256_current_match"] is False
    final_path.write_text(original_final_text, encoding="utf-8")

    tampered_merge_report = tmp_path / "case.tampered.merge_report.json"
    tampered_merge_payload = json.loads(Path(merge_payload["report_path"]).read_text(encoding="utf-8"))
    tampered_merge_payload["merged_rows"][1]["line_sha256"] = "0" * 64
    tampered_merge_report.write_text(
        json.dumps(tampered_merge_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_merge_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(split.output_paths["validation_jsonl"]),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(tampered_merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert tampered_merge_gate_proc.returncode == 2
    tampered_merge_gate = json.loads(tampered_merge_gate_proc.stdout)
    assert "merge_rows_lineage_current_match" in tampered_merge_gate["gate_report"]["failed_checks"]
    assert "merge_rows_lineage_current_mismatch" in tampered_merge_gate["warnings"]
    assert tampered_merge_gate["metrics"]["merge"]["merged_rows_current_match"] is False
    assert "merged_rows.1.line_sha256_mismatch" in tampered_merge_gate["metrics"]["merge"]["merged_rows_errors"]

    dry_run_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(gate_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert dry_run_proc.returncode == 0, dry_run_proc.stderr
    dry_run_payload = json.loads(dry_run_proc.stdout)
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["installed"] is False
    assert dry_run_payload["would_replace"] is True
    assert dry_run_payload["gate_target"] == "apply_candidate"
    assert dry_run_payload["gate_checks"]["comparison_prior_provenance_available"] is True
    assert dry_run_payload["gate_metrics"]["comparison"]["prior_provenance_required"] is False
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    default_dry_run_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(default_gate_report),
            "--require-default-on",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_dry_run_proc.returncode == 0, default_dry_run_proc.stderr
    default_dry_run_payload = json.loads(default_dry_run_proc.stdout)
    assert default_dry_run_payload["dry_run"] is True
    assert default_dry_run_payload["gate_target"] == "default_on_candidate"
    assert default_dry_run_payload["gate_checks"]["comparison_prior_provenance_available"] is True

    replace_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(gate_report),
            "--replace",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert replace_proc.returncode == 0, replace_proc.stderr
    replace_payload = json.loads(replace_proc.stdout)
    assert replace_payload["installed"] is True
    assert Path(replace_payload["backup_path"]).is_file()
    installed_text = target_oto.read_text(encoding="utf-8")
    assert installed_text == final_path.read_text(encoding="utf-8")

    final_path.write_text("a.wav=a,999,100,-350,50,25\n", encoding="utf-8")
    stale_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(final_path),
            "--target-oto",
            str(target_oto),
            "--gate-report",
            str(gate_report),
            "--replace",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_proc.returncode == 2
    stale_payload = json.loads(stale_proc.stdout)
    assert stale_payload["reason"] == "gate_failed:apply_candidate"
    assert stale_payload["gate_checks"]["merge_final_sha256_current_match"] is False
    assert stale_payload["gate_checks"]["comparison_prior_provenance_available"] is True
    assert target_oto.read_text(encoding="utf-8") == installed_text


def test_install_final_rejects_merge_report_missing_final_sha_after_gate_recompute(tmp_path, monkeypatch):
    from scripts.dev import mfa_free_oto_review as review

    final_path = tmp_path / "case.final.ini"
    final_path.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    target_oto = tmp_path / "voicebank" / "oto.ini"
    target_oto.parent.mkdir()
    target_oto.write_text("old.wav=old,0,100,-200,50,25\n", encoding="utf-8")
    gate_report = tmp_path / "gate.json"
    gate_report.write_text('{"ok": true}\n', encoding="utf-8")
    merge_report = tmp_path / "case.final.merge_report.json"
    merge_report.write_text(
        json.dumps({"ok": True, "final_path": str(final_path)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def fake_recompute_gate_from_report(payload):
        return {
            "merge_report": str(merge_report),
            "gate_report": {"apply_candidate": True, "default_on_candidate": False, "failed_checks": []},
            "checks": {},
            "metrics": {},
            "warnings": [],
        }

    monkeypatch.setattr(review, "_recompute_gate_from_report", fake_recompute_gate_from_report)

    report = review._build_install_final_report(
        final_path=str(final_path),
        target_oto=str(target_oto),
        gate_report_path=str(gate_report),
        replace=True,
    )

    assert report["ok"] is False
    assert report["reason"] == "merge_report_missing_final_sha256"
    assert report["final_sha256"] == _sha256_file(final_path)
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")


def test_install_session_infers_final_and_gate_from_review_session(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=i,120,100,-300,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(str(generated), str(tmp_path / "split"), name="case", wav_durations_ms={"a.wav": 500.0})
    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    review_session = Path(split.output_paths["review_session"])
    review_payload = json.loads(review_session.read_text(encoding="utf-8"))
    gate_report = Path(review_payload["recommended_next_steps"][4]["expected_report"])
    target_oto = tmp_path / "voicebank" / "oto.ini"
    target_oto.parent.mkdir()
    target_oto.write_text("old.wav=old,0,100,-200,50,25\n", encoding="utf-8")
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    merge_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge-session",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_session_proc.returncode == 0, merge_session_proc.stderr
    merge_session_payload = json.loads(merge_session_proc.stdout)

    evaluate_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-session",
            "--review-session",
            str(review_session),
            "--comparison-json",
            str(comparison),
            "--out",
            str(gate_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert evaluate_session_proc.returncode == 0, evaluate_session_proc.stderr
    assert gate_report.is_file()

    ready_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert ready_status_proc.returncode == 0, ready_status_proc.stderr
    ready_status = json.loads(ready_status_proc.stdout)
    assert ready_status["ok"] is True
    assert ready_status["ready_to_install"] is True
    assert ready_status["ready_to_install_reason"] == "ready"
    assert ready_status["install_readiness"]["requires_default_on"] is False

    default_required_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
            "--require-default-on",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_required_status_proc.returncode == 0, default_required_status_proc.stderr
    default_required_status = json.loads(default_required_status_proc.stdout)
    assert default_required_status["ok"] is True
    assert default_required_status["ready_to_install"] is False
    assert default_required_status["ready_to_install_reason"] == "default_on_gate_failed"
    assert default_required_status["install_readiness"]["requires_default_on"] is True
    assert default_required_status["next_action"] == "record_quality_evidence_or_inspect_default_on_gate_failures"

    install_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-session",
            "--review-session",
            str(review_session),
            "--target-oto",
            str(target_oto),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "cp932"},
        text=True,
        capture_output=True,
    )
    assert install_session_proc.returncode == 0, install_session_proc.stderr
    install_session_payload = json.loads(install_session_proc.stdout)
    assert install_session_payload["command"] == "install-session"
    assert install_session_payload["dry_run"] is True
    assert install_session_payload["installed"] is False
    assert install_session_payload["would_replace"] is True
    assert install_session_payload["used_paths"]["final"] == merge_session_payload["final_path"]
    assert install_session_payload["used_paths"]["gate_report"] == str(gate_report)
    assert install_session_payload["gate_target"] == "apply_candidate"
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    default_required_install_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-session",
            "--review-session",
            str(review_session),
            "--target-oto",
            str(target_oto),
            "--require-default-on",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_required_install_session_proc.returncode == 2
    default_required_install_session = json.loads(default_required_install_session_proc.stdout)
    assert default_required_install_session["reason"] == "gate_failed:default_on_candidate"
    assert default_required_install_session["session_status"]["ready_to_install"] is False
    assert default_required_install_session["session_status"]["ready_to_install_reason"] == "default_on_gate_failed"
    assert default_required_install_session["session_status"]["install_readiness"]["requires_default_on"] is True
    assert default_required_install_session["session_status"]["next_action"] == (
        "record_quality_evidence_or_inspect_default_on_gate_failures"
    )
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    gate_report.write_text("{not-json\n", encoding="utf-8")
    invalid_gate_install_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-session",
            "--review-session",
            str(review_session),
            "--target-oto",
            str(target_oto),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_gate_install_session_proc.returncode == 2
    invalid_gate_install_session = json.loads(invalid_gate_install_session_proc.stdout)
    assert invalid_gate_install_session["command"] == "install-session"
    assert invalid_gate_install_session["reason"] == "invalid_gate_report"
    assert invalid_gate_install_session["used_paths"]["gate_report"] == str(gate_report)
    assert invalid_gate_install_session["session_status"]["ok"] is True
    assert invalid_gate_install_session["session_status"]["ready_to_install"] is False
    assert invalid_gate_install_session["session_status"]["ready_to_install_reason"] == "invalid_gate_report"
    assert "gate_report_invalid" in invalid_gate_install_session["session_status"]["warnings"]
    assert invalid_gate_install_session["session_status"]["next_action"] == "rerun_evaluate_apply_gate"
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    gate_report.write_text("[]\n", encoding="utf-8")
    nonobject_gate_install_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-session",
            "--review-session",
            str(review_session),
            "--target-oto",
            str(target_oto),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_gate_install_session_proc.returncode == 2
    nonobject_gate_install_session = json.loads(nonobject_gate_install_session_proc.stdout)
    assert nonobject_gate_install_session["command"] == "install-session"
    assert nonobject_gate_install_session["reason"] == "invalid_gate_report"
    assert nonobject_gate_install_session["error"] == "json_root_not_object"
    assert "gate_report_invalid" in nonobject_gate_install_session["session_status"]["warnings"]
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")

    gate_report.write_text("{}\n", encoding="utf-8")
    empty_gate_install_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-session",
            "--review-session",
            str(review_session),
            "--target-oto",
            str(target_oto),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_gate_install_session_proc.returncode == 2
    empty_gate_install_session = json.loads(empty_gate_install_session_proc.stdout)
    assert empty_gate_install_session["command"] == "install-session"
    assert empty_gate_install_session["reason"] == "invalid_gate_report"
    assert empty_gate_install_session["error"] == "empty_gate_report"
    assert "gate_report_invalid" in empty_gate_install_session["session_status"]["warnings"]
    assert empty_gate_install_session["session_status"]["next_action"] == "rerun_evaluate_apply_gate"
    assert target_oto.read_text(encoding="utf-8").startswith("old.wav=old")


def test_timebase_fingerprint_rejects_mismatched_cache():
    contract = TimebaseContract(
        sample_rate=44100,
        analysis_sample_rate=16000,
        frame_shift_ms=5.0,
        window_size_ms=25.0,
    )
    assert contract.frame_time_anchor == "left"
    assert contract.frame_to_ms(2) == 10.0
    center_contract = TimebaseContract(
        sample_rate=44100,
        analysis_sample_rate=16000,
        frame_shift_ms=5.0,
        window_size_ms=25.0,
        frame_time_anchor="center",
    )
    assert center_contract.frame_to_ms(2) == 22.5
    assert cache_key_matches_timebase({"timebase_fingerprint": contract.cache_fingerprint()}, contract)
    changed = TimebaseContract(
        sample_rate=44100,
        analysis_sample_rate=16000,
        frame_shift_ms=10.0,
        window_size_ms=25.0,
    )
    assert not cache_key_matches_timebase({"timebase_fingerprint": contract.cache_fingerprint()}, changed)


def test_segmental_hsmm_decoder_recovers_known_sequence_intervals():
    times = [float(i * 10) for i in range(40)]
    states = [
        HSMMStateSpec("slot0.c", "consonant", 50.0, 90.0, mode_duration_ms=70.0, duration_sigma_ms=10.0),
        HSMMStateSpec("slot0.v", "vowel", 100.0, 160.0, mode_duration_ms=130.0, duration_sigma_ms=15.0),
        HSMMStateSpec("slot1.c", "consonant", 50.0, 90.0, mode_duration_ms=70.0, duration_sigma_ms=10.0),
        HSMMStateSpec("slot1.v", "vowel", 100.0, 160.0, mode_duration_ms=130.0, duration_sigma_ms=15.0),
    ]

    def score_range(start: float, end: float) -> list[float]:
        return [2.0 if start <= t < end else -2.0 for t in times]

    frame_scores = {
        "slot0.c": score_range(0.0, 70.0),
        "slot0.v": score_range(70.0, 200.0),
        "slot1.c": score_range(200.0, 270.0),
        "slot1.v": score_range(270.0, 400.0),
    }
    result = decode_segmental_hsmm(states, times, frame_scores, beam_width_per_state=32, timeout_ms=1000)

    assert result.ok
    assert [(s.start_ms, s.end_ms) for s in result.states] == [
        (0.0, 70.0),
        (70.0, 200.0),
        (200.0, 270.0),
        (270.0, 400.0),
    ]


def test_segmental_hsmm_decoder_honors_state_frame_windows():
    times = [float(i * 10) for i in range(50)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 80.0, 130.0, mode_duration_ms=100.0, duration_sigma_ms=12.0),
    ]
    frame_scores = {
        "slot0.v": [2.0 if 300.0 <= t < 400.0 else 1.7 if 80.0 <= t < 180.0 else -2.0 for t in times],
    }

    unrestricted = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
    )
    restricted = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        state_frame_windows={
            "slot0.v": HSMMStateFrameWindow(
                start_min_frame=6,
                start_max_frame=12,
                end_min_frame=16,
                end_max_frame=22,
            )
        },
    )

    assert unrestricted.ok
    assert unrestricted.states[0].start_ms == 300.0
    assert restricted.ok
    assert 60.0 <= restricted.states[0].start_ms <= 120.0
    assert 160.0 <= restricted.states[0].end_ms <= 220.0


def test_segmental_hsmm_coarse_to_refine_recovers_known_sequence_intervals():
    times = [float(i * 10) for i in range(80)]
    states = [
        HSMMStateSpec("slot0.c", "consonant", 40.0, 90.0, mode_duration_ms=60.0, duration_sigma_ms=12.0),
        HSMMStateSpec("slot0.v", "vowel", 90.0, 170.0, mode_duration_ms=130.0, duration_sigma_ms=18.0),
        HSMMStateSpec("slot1.c", "consonant", 40.0, 90.0, mode_duration_ms=60.0, duration_sigma_ms=12.0),
        HSMMStateSpec("slot1.v", "vowel", 90.0, 170.0, mode_duration_ms=130.0, duration_sigma_ms=18.0),
    ]

    def score_range(start: float, end: float) -> list[float]:
        return [2.0 if start <= t < end else -2.0 for t in times]

    frame_scores = {
        "slot0.c": score_range(50.0, 110.0),
        "slot0.v": score_range(110.0, 240.0),
        "slot1.c": score_range(390.0, 450.0),
        "slot1.v": score_range(450.0, 580.0),
    }
    result = decode_segmental_hsmm_coarse_to_refine(
        states,
        times,
        frame_scores,
        coarse_stride=4,
        refine_window_ms=70.0,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        allow_internal_gaps=True,
        max_internal_gap_ms=220.0,
    )

    assert result.ok
    assert result.meta["coarse_to_refine"] is True
    assert result.meta["coarse_stride"] == 4
    assert [(s.start_ms, s.end_ms) for s in result.states] == [
        (50.0, 110.0),
        (110.0, 240.0),
        (390.0, 450.0),
        (450.0, 580.0),
    ]


def test_segmental_hsmm_decoder_can_skip_leading_gap_for_late_segment():
    times = [float(i * 10) for i in range(60)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
    ]
    frame_scores = {
        "slot0.v": [2.0 if 220.0 <= t < 340.0 else -2.0 for t in times],
    }
    no_gap = decode_segmental_hsmm(states, times, frame_scores, beam_width_per_state=32, timeout_ms=1000)
    with_gap = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
    )

    assert no_gap.ok
    assert no_gap.states[0].start_ms == 0.0
    assert with_gap.ok
    assert (with_gap.states[0].start_ms, with_gap.states[0].end_ms) == (220.0, 340.0)
    assert with_gap.meta["leading_gap_ms"] == 220.0


def test_segmental_hsmm_decoder_can_cap_leading_gap_for_dense_sequences():
    times = [float(i * 10) for i in range(60)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
    ]
    frame_scores = {
        "slot0.v": [2.0 if 220.0 <= t < 340.0 else -2.0 for t in times],
    }
    capped = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        max_leading_gap_ms=100.0,
    )

    assert capped.ok
    assert capped.states[0].start_ms <= 100.0
    assert capped.meta["max_leading_gap_ms"] == 100.0


def test_segmental_hsmm_decoder_can_skip_internal_gap_between_slots():
    times = [float(i * 10) for i in range(80)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
        HSMMStateSpec("slot1.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
    ]
    frame_scores = {
        "slot0.v": [2.0 if 80.0 <= t < 200.0 else -2.0 for t in times],
        "slot1.v": [2.0 if 520.0 <= t < 640.0 else -2.0 for t in times],
    }
    no_gap = decode_segmental_hsmm(states, times, frame_scores, beam_width_per_state=32, timeout_ms=1000)
    with_gap = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        allow_internal_gaps=True,
        max_internal_gap_ms=400.0,
        internal_gap_penalty_per_ms=0.00005,
    )

    assert no_gap.ok
    assert no_gap.states[1].start_ms < 520.0
    assert with_gap.ok
    assert [(s.start_ms, s.end_ms) for s in with_gap.states] == [(80.0, 200.0), (520.0, 640.0)]
    assert with_gap.meta["internal_gap_ms"] == 320.0


def test_filename_hsmm_internal_gap_cap_scales_with_state_density():
    assert _max_internal_gap_ms(3000.0, 12) == pytest.approx(237.5)
    assert _max_internal_gap_ms(10000.0, 12) == pytest.approx(260.0)
    assert _max_internal_gap_ms(400.0, 12) == pytest.approx(90.0)
    assert _max_internal_gap_ms(0.0, 0) == pytest.approx(180.0)


def test_filename_hsmm_leading_gap_cap_preserves_single_vowel_banks():
    assert _max_leading_gap_ms(3000.0, 12) == pytest.approx(237.5)
    assert _max_leading_gap_ms(10000.0, 12) == pytest.approx(260.0)
    assert _max_leading_gap_ms(400.0, 12) == pytest.approx(90.0)
    assert _max_leading_gap_ms(3000.0, 1) == pytest.approx(900.0)


def test_hsmm_event_sequence_guard_rejects_distant_runtime_disagreement():
    expected_slots = [ExpectedSlot(0, 0, "i", "v", "vowel_nucleus")]
    hsmm_events = [
        {
            "label": "vowel_nucleus",
            "expected_phone_index": 0,
            "selected_time_ms": 1200.0,
        }
    ]
    close_runtime = [
        {
            "label": "vowel_nucleus",
            "expected_phone_index": 0,
            "selected_time_ms": 1400.0,
        }
    ]
    far_runtime = [
        {
            "label": "vowel_nucleus",
            "expected_phone_index": 0,
            "selected_time_ms": 1510.0,
        }
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events, runtime_events=close_runtime)
    assert not _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events, runtime_events=far_runtime)


def test_hsmm_event_sequence_guard_accepts_cv_head_vowel_only_nucleus():
    expected_slots = [
        ExpectedSlot(0, 0, "a", "cv_head", "cv_boundary"),
        ExpectedSlot(1, 1, "i", "vv", "vowel_nucleus"),
    ]
    hsmm_events = [
        {"label": "vowel_nucleus", "expected_phone_index": 0, "selected_time_ms": 520.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 760.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_accepts_cv_head_vv_boundary_then_vowel_row():
    expected_slots = [
        ExpectedSlot(0, 0, "a", "cv_head", "cv_boundary"),
        ExpectedSlot(1, 0, "a", "v", "vowel_nucleus"),
        ExpectedSlot(2, 1, "i", "vv", "vowel_nucleus"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 0, "selected_time_ms": 380.0},
        {"label": "vowel_nucleus", "expected_phone_index": 0, "selected_time_ms": 520.0},
        {"label": "vv_boundary", "expected_phone_index": 1, "selected_time_ms": 680.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 760.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_ignores_distant_runtime_for_vowel_only_cv_head():
    expected_slots = [
        ExpectedSlot(0, 1, "a", "cv_head", "cv_boundary"),
        ExpectedSlot(1, 1, "a", "vv", "vowel_nucleus"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 1, "selected_time_ms": 790.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 930.0},
    ]
    runtime_events = [
        {"label": "cv_boundary", "expected_phone_index": 1, "selected_time_ms": 2610.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 2680.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events, runtime_events=runtime_events)


def test_hsmm_event_sequence_guard_accepts_vc_target_with_vowel_boundary_anchor():
    expected_slots = [
        ExpectedSlot(0, 1, "o", "cv_head", "cv_boundary"),
        ExpectedSlot(1, 1, "o", "vv", "vowel_nucleus"),
        ExpectedSlot(2, 2, "n", "vc", "phone_change"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 1, "selected_time_ms": 790.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 930.0},
        {"label": "vv_boundary", "expected_phone_index": 2, "selected_time_ms": 1310.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_accepts_implicit_cv_with_vowel_boundary_anchor():
    expected_slots = [
        ExpectedSlot(0, 2, "n", "vc", "phone_change"),
        ExpectedSlot(1, 3, "o", "implicit_cv", "cv_boundary"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 2, "selected_time_ms": 1310.0},
        {"label": "vowel_nucleus", "expected_phone_index": 2, "selected_time_ms": 1442.0},
        {"label": "vv_boundary", "expected_phone_index": 3, "selected_time_ms": 1750.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_accepts_implicit_cv_after_vowel_only_cv_head():
    expected_slots = [
        ExpectedSlot(0, 1, "n", "cv_head", "phone_change"),
        ExpectedSlot(1, 2, "a", "implicit_cv", "cv_boundary"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 1, "selected_time_ms": 820.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 968.5},
        {"label": "vv_boundary", "expected_phone_index": 2, "selected_time_ms": 1350.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_accepts_moraic_n_vcv_vowel_boundary():
    expected_slots = [
        ExpectedSlot(0, 0, "n", "v", "vowel_nucleus"),
        ExpectedSlot(1, 1, "i", "vcv", "cv_boundary"),
        ExpectedSlot(2, 2, "n", "vv", "vowel_nucleus"),
    ]
    hsmm_events = [
        {"label": "vv_boundary", "expected_phone_index": 0, "selected_time_ms": 260.0},
        {"label": "vowel_nucleus", "expected_phone_index": 0, "selected_time_ms": 419.5},
        {"label": "vv_boundary", "expected_phone_index": 1, "selected_time_ms": 810.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 964.0},
        {"label": "vv_boundary", "expected_phone_index": 2, "selected_time_ms": 1350.0},
        {"label": "vowel_nucleus", "expected_phone_index": 2, "selected_time_ms": 1493.0},
    ]

    assert _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)


def test_hsmm_event_sequence_guard_keeps_interior_cv_boundary_strict():
    expected_slots = [
        ExpectedSlot(0, 0, "a", "implicit_cv", "cv_boundary"),
    ]
    hsmm_events = [
        {"label": "vowel_nucleus", "expected_phone_index": 0, "selected_time_ms": 520.0},
    ]
    vv_boundary_events = [
        {"label": "vv_boundary", "expected_phone_index": 0, "selected_time_ms": 480.0},
    ]

    assert not _hsmm_event_sequence_matches_expected(expected_slots, hsmm_events)
    assert not _hsmm_event_sequence_matches_expected(expected_slots, vv_boundary_events)


def test_hsmm_runtime_guard_ignores_rule_based_runtime_events():
    prediction = SimpleNamespace(posterior=SimpleNamespace(metadata={"rule_based": True}))
    events = ({"label": "vowel_nucleus", "selected_time_ms": 1200.0},)

    assert _runtime_events_for_hsmm_guard(prediction, events) == ()


def test_hsmm_runtime_guard_keeps_model_runtime_events():
    prediction = SimpleNamespace(posterior=SimpleNamespace(metadata={"rule_based": False}))
    events = ({"label": "vowel_nucleus", "selected_time_ms": 1200.0},)

    assert _runtime_events_for_hsmm_guard(prediction, events) == events


def test_hsmm_runtime_guard_ignores_collapsed_late_model_events():
    prediction = SimpleNamespace(
        posterior=SimpleNamespace(
            metadata={"rule_based": False},
            times_ms=[float(idx * 10) for idx in range(501)],
        )
    )
    events = (
        {"label": "cv_boundary", "expected_phone_index": 0, "selected_time_ms": 2100.0},
        {"label": "vowel_nucleus", "expected_phone_index": 1, "selected_time_ms": 2140.0},
        {"label": "vowel_nucleus", "expected_phone_index": 2, "selected_time_ms": 2180.0},
        {"label": "vowel_nucleus", "expected_phone_index": 3, "selected_time_ms": 2220.0},
    )

    assert _runtime_events_for_hsmm_guard(prediction, events) == ()


def test_workflow_expected_phones_expands_romaji_yoon_cvvc_onsets():
    slots = build_filename_slots("kya-kyu-kye-kyo-kya.wav", language="japanese", format_type="cvvc")

    assert _expected_phones("kya-kyu-kye-kyo-kya.wav", [], filename_slots=slots) == [
        "k",
        "y",
        "a",
        "k",
        "y",
        "u",
        "k",
        "y",
        "e",
        "k",
        "y",
        "o",
        "k",
        "y",
        "a",
    ]


def test_workflow_expected_phones_keeps_standalone_y_w_expanded():
    ya_slots = build_filename_slots("ya-yu-ye-yo.wav", language="japanese", format_type="cvvc")
    wa_slots = build_filename_slots("wa-wi-we-wo.wav", language="japanese", format_type="cvvc")

    assert _expected_phones("ya-yu-ye-yo.wav", [], filename_slots=ya_slots) == ["y", "a", "y", "u", "y", "e", "y", "o"]
    assert _expected_phones("wa-wi-we-wo.wav", [], filename_slots=wa_slots) == ["w", "a", "w", "i", "w", "e", "w", "o"]


def test_workflow_expected_phones_preserves_japanese_multigraph_onsets():
    slots = build_filename_slots("sha-shi-chu-tsu.wav", language="japanese", format_type="cvvc")

    assert _expected_phones("sha-shi-chu-tsu.wav", [], filename_slots=slots) == [
        "sh",
        "a",
        "sh",
        "i",
        "ch",
        "u",
        "ts",
        "u",
    ]


def test_adapter_config_uses_row_plan_role_family_for_vcv_rows():
    records = build_filename_row_plan("_a'bba'e.wav", language="korean", format_type="vcv")
    config = OtoAdapterConfig(language="korean", format_type="vcv", alias_type="auto")

    row_config = _adapter_config_for_row_plan_record(config, records[1])
    vv_config = _adapter_config_for_row_plan_record(config, records[2])

    assert row_config.alias_type == "vcv"
    assert vv_config.alias_type == "vv"
    assert config.alias_type == "auto"


def test_template_row_plan_context_maps_anchor_phone_to_filename_slot():
    slots = build_filename_slots("あかき.wav", language="japanese", format_type="CVVC")
    template_row = SimpleNamespace(wav="あかき.wav", alias="a k", source_row_index=7)

    context = _template_row_plan_context(
        template_row,
        3,
        anchor=OtoAnchor(anchor_abs_ms=500.0, score=0.8, role="vc", expected_phone_index=1),
        filename_slots=slots,
    )

    assert context["slot_index"] == 1
    assert context["left_slot_index"] == 0
    assert context["right_slot_index"] == 1
    assert context["role_family"] == "vc"
    assert context["expected_phone_index"] == 1
    assert context["source_row_index"] == 7


def test_validate_oto_lines_exposes_expected_split_labels():
    records = validate_oto_lines(
        [
            "a.wav=a,0,100,-350,50,25",
            "a.wav=b,0,10,-50,30,40",
            "a.wav=c,0,100,250,50,25",
        ],
        wav_durations_ms={"a.wav": 400.0},
    )
    assert [record.split for record in records] == [SPLIT_CLEAN, SPLIT_FIX_REQUIRED, SPLIT_ATTENTION_ONLY]


def test_validate_oto_lines_uses_hsmm_diagnostics_for_attention_and_fix():
    records = validate_oto_lines(
        [
            "a.wav=a,0,100,-350,50,25",
            "a.wav=i,0,100,-350,50,25",
            "a.wav=u,0,100,-350,50,25",
        ],
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            1: {"hsmm_min_selected_vs_best_local_margin": -0.05},
            2: {"hsmm_max_duration_z_abs": 4.0},
        },
    )

    assert records[0].split == SPLIT_CLEAN
    assert records[1].split == SPLIT_ATTENTION_ONLY
    assert "attention.hsmm_local_margin_low" in records[1].reasons
    assert records[1].metrics["hsmm_min_selected_vs_best_local_margin"] == -0.05
    assert records[2].split == SPLIT_FIX_REQUIRED
    assert "fix_required.hsmm_duration_prior_violation" in records[2].reasons
    assert records[2].metrics["hsmm_max_duration_z_abs"] == 4.0


def test_validate_oto_lines_exposes_rule_based_fallback_as_attention():
    from scripts.dev.mfa_free_oto_review import _validation_gate_metrics

    records = validate_oto_lines(
        [
            "a.wav=a,0,100,-350,50,25",
            "a.wav=i,0,100,-350,50,25",
        ],
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
            1: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_failed_count": 1.0,
            },
        },
    )

    assert records[0].split == SPLIT_ATTENTION_ONLY
    assert "attention.rule_based_inference" in records[0].reasons
    assert records[0].metrics["rule_based_checkpoint_missing_count"] == 1.0
    assert records[1].split == SPLIT_ATTENTION_ONLY
    assert "attention.rule_based_checkpoint_failed" in records[1].reasons
    assert records[1].metrics["rule_based_checkpoint_failed_count"] == 1.0
    metrics = _validation_gate_metrics(records)
    assert metrics["rule_based_inference_rows"] == 2
    assert metrics["rule_based_checkpoint_missing_rows"] == 1
    assert metrics["rule_based_checkpoint_failed_rows"] == 1


def test_validate_oto_lines_calibrates_korean_cvc_rule_based_clean_split():
    records = validate_oto_lines(
        [
            "ba.wav=bi,1440,160,-550,60,25",
            "ba.wav=bu,2030,160,-400,120,85",
            "ba.wav=i b,1840,120,-154,80,45",
            "ba.wav=be,2450,160,-250,60,25",
        ],
        language="korean",
        format_type="cvc",
        wav_durations_ms={"ba.wav": 5000.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
            1: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
            2: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
            3: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_failed_count": 1.0,
            },
        },
    )

    assert records[0].split == SPLIT_CLEAN
    assert records[0].reasons == ("clean",)
    assert records[1].split == SPLIT_ATTENTION_ONLY
    assert records[1].reasons == ("attention.korean_cvc_default_cv_profile",)
    assert records[2].split == SPLIT_CLEAN
    assert records[2].reasons == ("clean",)
    assert records[3].split == SPLIT_ATTENTION_ONLY
    assert "attention.rule_based_checkpoint_failed" in records[3].reasons


def test_validate_oto_lines_marks_korean_cvc_pitch_suffix_rule_based_attention():
    records = validate_oto_lines(
        [
            "ba.wav=biC4S,1440,160,-550,60,25",
            "ba.wav=i bC4S,1840,120,-154,80,45",
            "ba.wav=bu,2030,160,-400,60,25",
        ],
        language="korean",
        format_type="cvc",
        wav_durations_ms={"ba.wav": 5000.0},
        row_diagnostics={
            0: {"rule_based_inference_count": 1.0, "rule_based_checkpoint_missing_count": 1.0},
            1: {"rule_based_inference_count": 1.0, "rule_based_checkpoint_missing_count": 1.0},
            2: {"rule_based_inference_count": 1.0, "rule_based_checkpoint_missing_count": 1.0},
        },
    )

    assert records[0].split == SPLIT_ATTENTION_ONLY
    assert records[0].reasons == ("attention.korean_cvc_pitch_suffix_rule_based",)
    assert records[1].split == SPLIT_ATTENTION_ONLY
    assert records[1].reasons == ("attention.korean_cvc_pitch_suffix_rule_based",)
    assert records[2].split == SPLIT_CLEAN
    assert records[2].reasons == ("clean",)


def test_validate_oto_lines_trusts_pitch_suffixed_breath_source_timing_metric():
    records = validate_oto_lines(
        ["br.wav=brC4S,755.33,510.43,-510.43,195.01,94.1"],
        language="korean",
        format_type="cvc",
        wav_durations_ms={"br.wav": 2742.8571428571427},
        row_diagnostics={
            0: {
                "special_alias_source_timing_preserve_count": 1.0,
                "row_plan": {
                    "role_family": "cv",
                    "source_timing_trusted": False,
                    "source_row_index": 12,
                },
            },
        },
    )

    assert len(records) == 1
    assert records[0].split == SPLIT_CLEAN
    assert records[0].reasons == ("clean",)
    assert records[0].role_family == "br"
    assert records[0].source_timing_trusted is True


def test_validate_oto_lines_calibrates_korean_cvc_low_margin_clean_split():
    records = validate_oto_lines(
        [
            "ba.wav=bi,1440,160,-550,60,25",
            "ba.wav=bu,2030,160,-400,120,85",
            "ba.wav=be,2450,160,-430,60,25",
            "ja.wav=ji,1440,160,-550,60,25",
        ],
        language="korean",
        format_type="cvc",
        wav_durations_ms={"ba.wav": 5000.0, "ja.wav": 5000.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
                "local_refine_low_margin_count": 1.0,
                "local_refine_max_delta_ms": 4.5,
                "local_refine_min_margin": 0.01,
            },
            1: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
                "local_refine_low_margin_count": 1.0,
                "local_refine_max_delta_ms": 4.5,
                "local_refine_min_margin": 0.01,
            },
            2: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
                "local_refine_low_margin_count": 1.0,
                "local_refine_max_delta_ms": 25.0,
                "local_refine_min_margin": 0.01,
            },
            3: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
                "local_refine_low_margin_count": 1.0,
                "local_refine_max_delta_ms": 4.5,
                "local_refine_min_margin": 0.0005,
            },
        },
    )

    assert records[0].split == SPLIT_CLEAN
    assert records[0].reasons == ("clean",)
    assert records[1].split == SPLIT_ATTENTION_ONLY
    assert records[1].reasons == ("attention.korean_cvc_default_cv_profile",)
    assert records[2].split == SPLIT_ATTENTION_ONLY
    assert records[2].reasons == ("attention.local_refine_low_margin",)
    assert records[3].split == SPLIT_ATTENTION_ONLY
    assert records[3].reasons == ("attention.local_refine_low_margin",)


def test_validate_oto_lines_keeps_non_korean_low_margin_attention():
    records = validate_oto_lines(
        ["ka.wav=ka,1440,160,-550,60,25"],
        language="japanese",
        format_type="cvvc",
        wav_durations_ms={"ka.wav": 5000.0},
        row_diagnostics={
            0: {
                "local_refine_low_margin_count": 1.0,
                "local_refine_max_delta_ms": 4.5,
                "local_refine_min_margin": 0.01,
            },
        },
    )

    assert records[0].split == SPLIT_ATTENTION_ONLY
    assert records[0].reasons == ("attention.local_refine_low_margin",)


def test_split_oto_file_passes_format_type_to_korean_cvc_validation(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "ba.wav=bi,1440,160,-550,60,25",
                "ba.wav=bu,2030,160,-400,120,85",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    split = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="case",
        language="korean",
        wav_durations_ms={"ba.wav": 5000.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
            1: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
            },
        },
        session_metadata={"format_type": "cvc"},
    )

    assert split.counts["clean"] == 1
    assert split.counts["attention_only"] == 1
    assert Path(split.output_paths["clean"]).read_text(encoding="utf-8").strip().endswith("=bi,1440,160,-550,60,25")
    assert Path(split.output_paths["attention_only"]).read_text(encoding="utf-8").strip().endswith(
        "=bu,2030,160,-400,120,85"
    )


def test_validate_oto_lines_does_not_force_attention_for_hsmm_primary_rule_based_rows():
    records = validate_oto_lines(
        ["a.wav=a,0,100,-350,50,25"],
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_missing_count": 1.0,
                "hsmm_decoder_used": 1.0,
                "hsmm_min_selected_vs_best_local_margin": 0.25,
            },
        },
    )

    assert records[0].split == SPLIT_CLEAN
    assert records[0].reasons == ("clean",)
    assert records[0].metrics["rule_based_checkpoint_missing_count"] == 1.0


def test_evaluate_gate_blocks_apply_when_checkpoint_fallback_failed(tmp_path):
    from scripts.dev.mfa_free_oto_review import _build_artifact_gate_report

    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=i,120,100,-300,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="case",
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            0: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_failed_count": 1.0,
            },
            1: {
                "rule_based_inference_count": 1.0,
                "rule_based_checkpoint_failed_count": 1.0,
            },
        },
    )
    final_path = tmp_path / "split" / "case.final.ini"
    merge = merge_split_oto(
        validation_jsonl_path=split.output_paths["validation_jsonl"],
        final_path=str(final_path),
        edited_attention_only_path=split.output_paths["attention_only"],
        acknowledge_attention_only=True,
    )
    assert merge.ok is True
    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )

    report = _build_artifact_gate_report(
        validation_jsonl_path=str(validation_jsonl),
        comparison_json_path=str(comparison),
        merge_report_path=merge.report_path,
    )

    assert report["metrics"]["validation"]["rule_based_checkpoint_failed_rows"] == 2
    assert report["checks"]["no_rule_based_checkpoint_failed_rows"] is False
    assert report["gate_report"]["apply_candidate"] is False
    assert "no_rule_based_checkpoint_failed_rows" in report["gate_report"]["failed_checks"]
    assert "rule_based_checkpoint_failed_rows_present" in report["warnings"]


def test_validate_oto_lines_uses_evidence_candidate_flags_for_vv_and_sonorants():
    records = validate_oto_lines(
        [
            "a.wav=a i,0,100,-350,50,25",
            "a.wav=a u,0,100,-350,50,25",
            "a.wav=a m,0,100,-350,50,25",
            "a.wav=a n,0,100,-350,50,25",
        ],
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            0: {"evidence_vv_boundary_ambiguous_count": 1},
            1: {"evidence_vv_boundary_low_feature_agreement_count": 1},
            2: {"evidence_sonorant_constriction_ambiguous_count": 1},
            3: {"evidence_sonorant_constriction_low_feature_agreement_count": 1},
        },
    )

    assert records[0].split == SPLIT_ATTENTION_ONLY
    assert "attention.evidence_vv_boundary_ambiguous" in records[0].reasons
    assert records[1].split == SPLIT_FIX_REQUIRED
    assert "fix_required.evidence_vv_boundary_low_feature_agreement" in records[1].reasons
    assert records[2].split == SPLIT_ATTENTION_ONLY
    assert "attention.evidence_sonorant_ambiguous" in records[2].reasons
    assert records[3].split == SPLIT_FIX_REQUIRED
    assert "fix_required.evidence_sonorant_low_feature_agreement" in records[3].reasons
    assert records[3].metrics["evidence_sonorant_constriction_low_feature_agreement_count"] == 1.0


def test_mfa_free_oto_review_cli_split_and_guarded_merge(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=b,0,10,-50,30,40",
                "a.wav=c,0,100,250,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 400.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    split_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert split_proc.returncode == 0, split_proc.stderr
    split_payload = json.loads(split_proc.stdout)
    assert split_payload["counts"] == {"attention_only": 1, "clean": 1, "fix_required": 1, "total": 3}

    final_path = review_dir / "case.final.ini"
    blocked_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(review_dir / "case.validation.jsonl"),
            "--final",
            str(final_path),
            "--edited-fix-required",
            str(review_dir / "case.fix_required.ini"),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert blocked_proc.returncode == 2
    assert not final_path.exists()

    edited_fix = review_dir / "case.edited_fix_required.ini"
    edited_fix.write_text("a.wav=b,0,90,-260,45,20\n", encoding="utf-8")
    merged_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(review_dir / "case.validation.jsonl"),
            "--final",
            str(final_path),
            "--edited-fix-required",
            str(edited_fix),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merged_proc.returncode == 0, merged_proc.stderr
    assert json.loads(merged_proc.stdout)["ok"] is True
    assert final_path.read_text(encoding="utf-8").splitlines()[1] == "a.wav=b,0,90,-260,45,20"


def test_mfa_free_oto_review_cli_session_status_detects_stale_split_artifact(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=b,0,10,-50,30,40",
                "a.wav=c,0,100,250,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(str(generated), str(tmp_path / "review"), name="case", wav_durations_ms={"a.wav": 400.0})
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    review_session = tmp_path / "review" / "case.review_session.json"

    current_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert current_proc.returncode == 0, current_proc.stderr
    current_payload = json.loads(current_proc.stdout)
    assert current_payload["ok"] is True
    assert current_payload["checks"]["validation_jsonl_current_match"] is True
    assert current_payload["checks"]["split_outputs_current_match"] is True
    assert current_payload["validation_counts"] == {"attention_only": 1, "clean": 1, "fix_required": 1, "total": 3}
    assert current_payload["edit_files"]["fix_required"]["target_exists"] is False
    assert current_payload["next_action"] == "prepare_review_edits"
    assert current_payload["ready_to_install"] is False
    assert current_payload["ready_to_install_reason"] == "missing_merge_report"

    prepare_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "prepare-edits",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert prepare_proc.returncode == 0, prepare_proc.stderr
    prepare_payload = json.loads(prepare_proc.stdout)
    edited_fix = review_session.parent / "case.edited_fix_required.ini"
    edited_attention = review_session.parent / "case.edited_attention_only.ini"
    assert prepare_payload["outputs"]["fix_required"]["action"] == "created"
    assert edited_fix.read_text(encoding="utf-8") == Path(split.output_paths["fix_required"]).read_text(encoding="utf-8")
    assert edited_attention.read_text(encoding="utf-8") == Path(split.output_paths["attention_only"]).read_text(encoding="utf-8")

    edited_fix.write_text("a.wav=b,0,90,-260,45,20\n", encoding="utf-8")
    no_overwrite_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "prepare-edits",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert no_overwrite_proc.returncode == 0, no_overwrite_proc.stderr
    assert json.loads(no_overwrite_proc.stdout)["outputs"]["fix_required"]["action"] == "kept_existing"
    assert edited_fix.read_text(encoding="utf-8") == "a.wav=b,0,90,-260,45,20\n"

    prepared_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert prepared_status_proc.returncode == 0, prepared_status_proc.stderr
    prepared_status = json.loads(prepared_status_proc.stdout)
    assert prepared_status["edit_files"]["fix_required"]["target_exists"] is True
    assert prepared_status["next_action"] == "edit_fix_required_then_merge"
    assert prepared_status["ready_to_install"] is False
    assert prepared_status["ready_to_install_reason"] == "missing_merge_report"

    merge_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge-session",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_session_proc.returncode == 0, merge_session_proc.stderr
    merge_session_payload = json.loads(merge_session_proc.stdout)
    assert merge_session_payload["ok"] is True
    assert merge_session_payload["command"] == "merge-session"
    assert merge_session_payload["used_paths"]["edited_fix_required"] == str(edited_fix)
    assert Path(merge_session_payload["final_path"]).is_file()
    assert Path(merge_session_payload["report_path"]).is_file()

    evaluate_session_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-session",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert evaluate_session_proc.returncode == 0, evaluate_session_proc.stderr
    evaluate_session_payload = json.loads(evaluate_session_proc.stdout)
    assert evaluate_session_payload["command"] == "evaluate-session"
    assert evaluate_session_payload["used_paths"]["merge_report"] == merge_session_payload["report_path"]
    assert evaluate_session_payload["metrics"]["merge"]["merge_ok"] is True
    assert "comparison_json_missing_or_unavailable" in evaluate_session_payload["warnings"]

    gate_report_path = review_session.parent / "case.gate.json"
    gate_report_path.write_text("{not-json\n", encoding="utf-8")
    invalid_gate_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_gate_status_proc.returncode == 0, invalid_gate_status_proc.stderr
    invalid_gate_status = json.loads(invalid_gate_status_proc.stdout)
    assert invalid_gate_status["ok"] is True
    assert invalid_gate_status["gate"]["path_exists"] is True
    assert invalid_gate_status["gate"]["invalid"] is True
    assert invalid_gate_status["next_action"] == "rerun_evaluate_apply_gate"
    assert "gate_report_invalid" in invalid_gate_status["warnings"]
    assert invalid_gate_status["ready_to_install"] is False
    assert invalid_gate_status["ready_to_install_reason"] == "invalid_gate_report"

    gate_report_path.write_text("[]\n", encoding="utf-8")
    nonobject_gate_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_gate_status_proc.returncode == 0, nonobject_gate_status_proc.stderr
    nonobject_gate_status = json.loads(nonobject_gate_status_proc.stdout)
    assert nonobject_gate_status["ok"] is True
    assert nonobject_gate_status["gate"]["invalid"] is True
    assert nonobject_gate_status["gate"]["error"] == "json_root_not_object"
    assert nonobject_gate_status["next_action"] == "rerun_evaluate_apply_gate"
    assert "gate_report_invalid" in nonobject_gate_status["warnings"]
    assert nonobject_gate_status["ready_to_install"] is False
    assert nonobject_gate_status["ready_to_install_reason"] == "invalid_gate_report"

    gate_report_path.write_text("{}\n", encoding="utf-8")
    empty_gate_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_gate_status_proc.returncode == 0, empty_gate_status_proc.stderr
    empty_gate_status = json.loads(empty_gate_status_proc.stdout)
    assert empty_gate_status["ok"] is True
    assert empty_gate_status["gate"]["invalid"] is True
    assert empty_gate_status["gate"]["error"] == "empty_gate_report"
    assert empty_gate_status["next_action"] == "rerun_evaluate_apply_gate"
    assert "gate_report_invalid" in empty_gate_status["warnings"]
    assert empty_gate_status["ready_to_install"] is False
    assert empty_gate_status["ready_to_install_reason"] == "invalid_gate_report"
    gate_report_path.unlink()

    merge_report_path = Path(merge_session_payload["report_path"])
    original_merge_report = merge_report_path.read_text(encoding="utf-8")
    tampered_merge_report = json.loads(original_merge_report)
    tampered_merge_report["merged_rows"][1]["line_sha256"] = "0" * 64
    merge_report_path.write_text(
        json.dumps(tampered_merge_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert tampered_status_proc.returncode == 2
    tampered_status = json.loads(tampered_status_proc.stdout)
    assert tampered_status["ok"] is False
    assert tampered_status["next_action"] == "rerun_merge_final"
    assert tampered_status["checks"]["merge_report_current_match"] is False
    assert tampered_status["checks"]["merge_report_final_sha256_current_match"] is True
    assert tampered_status["checks"]["merge_report_rows_current_match"] is False
    assert tampered_status["merge"]["merged_rows_current_match"] is False
    assert "merged_rows.1.line_sha256_mismatch" in tampered_status["merge"]["merged_rows_errors"]
    assert "merge_rows_lineage_current_mismatch" in tampered_status["warnings"]
    merge_report_path.write_text(original_merge_report, encoding="utf-8")

    final_path = Path(merge_session_payload["final_path"])
    original_final = final_path.read_text(encoding="utf-8")
    final_path.write_text(original_final.replace("0,90,-260", "0,91,-260"), encoding="utf-8")
    stale_final_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_final_status_proc.returncode == 2
    stale_final_status = json.loads(stale_final_status_proc.stdout)
    assert stale_final_status["ok"] is False
    assert stale_final_status["next_action"] == "rerun_merge_final"
    assert stale_final_status["checks"]["merge_report_current_match"] is False
    assert stale_final_status["checks"]["merge_report_final_sha256_current_match"] is False
    assert stale_final_status["merge"]["final_sha256_current_match"] is False
    assert "merge_final_sha256_current_mismatch" in stale_final_status["warnings"]
    final_path.write_text(original_final, encoding="utf-8")

    merge_report_path.write_text("{not-json\n", encoding="utf-8")
    invalid_merge_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_merge_status_proc.returncode == 2
    invalid_merge_status = json.loads(invalid_merge_status_proc.stdout)
    assert invalid_merge_status["ok"] is False
    assert invalid_merge_status["next_action"] == "rerun_merge_final"
    assert invalid_merge_status["checks"]["merge_report_valid"] is False
    assert invalid_merge_status["checks"]["merge_report_current_match"] is False
    assert invalid_merge_status["merge"]["invalid"] is True
    assert "merge_report_invalid" in invalid_merge_status["warnings"]

    merge_report_path.write_text("[]\n", encoding="utf-8")
    nonobject_merge_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert nonobject_merge_status_proc.returncode == 2
    nonobject_merge_status = json.loads(nonobject_merge_status_proc.stdout)
    assert nonobject_merge_status["ok"] is False
    assert nonobject_merge_status["next_action"] == "rerun_merge_final"
    assert nonobject_merge_status["checks"]["merge_report_valid"] is False
    assert nonobject_merge_status["checks"]["merge_report_current_match"] is False
    assert nonobject_merge_status["merge"]["invalid"] is True
    assert nonobject_merge_status["merge"]["error"] == "json_root_not_object"
    assert "merge_report_invalid" in nonobject_merge_status["warnings"]

    merge_report_path.write_text("{}\n", encoding="utf-8")
    empty_merge_status_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert empty_merge_status_proc.returncode == 2
    empty_merge_status = json.loads(empty_merge_status_proc.stdout)
    assert empty_merge_status["ok"] is False
    assert empty_merge_status["next_action"] == "rerun_merge_final"
    assert empty_merge_status["checks"]["merge_report_valid"] is False
    assert empty_merge_status["checks"]["merge_report_current_match"] is False
    assert empty_merge_status["merge"]["invalid"] is True
    assert empty_merge_status["merge"]["error"] == "empty_merge_report"
    assert "merge_report_invalid" in empty_merge_status["warnings"]
    merge_report_path.write_text(original_merge_report, encoding="utf-8")

    clean_path = Path(split.output_paths["clean"])
    clean_path.write_text(clean_path.read_text(encoding="utf-8") + "a.wav=extra,0,100,-350,50,25\n", encoding="utf-8")

    stale_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_proc.returncode == 2
    stale_payload = json.loads(stale_proc.stdout)
    assert stale_payload["ok"] is False
    assert stale_payload["checks"]["split_outputs_current_match"] is False
    assert stale_payload["artifacts"]["clean"]["sha256_match"] is False
    assert "artifact_sha256_mismatch:clean" in stale_payload["warnings"]
    assert "review_queue_counts_mismatch" in stale_payload["warnings"]


def test_mfa_free_oto_review_cli_split_accepts_timeline_diagnostics(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [{"raw_line": "a.wav=a,0,100,-350,50,25"}],
                "hsmm": {
                    "diagnostics": {
                        "state_count": 1,
                        "event_prior_state_count": 1,
                        "min_selected_vs_best_local_margin": 0.0,
                        "min_selected_vs_second_local_margin": 0.0,
                        "max_duration_z_abs": 4.0,
                    }
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 0, "clean": 0, "fix_required": 1, "total": 1}
    validation = json.loads((review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert validation["split"] == SPLIT_FIX_REQUIRED
    assert "fix_required.hsmm_duration_prior_violation" in validation["reasons"]
    assert validation["metrics"]["hsmm_max_duration_z_abs"] == 4.0


def test_mfa_free_oto_review_cli_split_flags_local_refine_attention(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [
                    {
                        "raw_line": "a.wav=a,0,100,-350,50,25",
                        "anchor": {
                            "anchor_abs_ms": 62.0,
                            "refined_from_abs_ms": 40.0,
                            "local_refine_delta_ms": 42.0,
                            "local_refine_margin": 0.20,
                            "warnings": ["local_refine_changed_anchor"],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 1, "clean": 0, "fix_required": 0, "total": 1}
    validation = json.loads((review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert validation["split"] == SPLIT_ATTENTION_ONLY
    assert "attention.local_refine_changed_anchor" in validation["reasons"]
    assert validation["metrics"]["local_refine_max_delta_ms"] == 42.0


def test_mfa_free_oto_review_cli_keeps_small_local_refine_delta_clean(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [
                    {
                        "raw_line": "a.wav=a,0,100,-350,50,25",
                        "anchor": {
                            "anchor_abs_ms": 62.0,
                            "refined_from_abs_ms": 40.0,
                            "local_refine_delta_ms": 22.0,
                            "local_refine_margin": 0.20,
                            "warnings": ["local_refine_changed_anchor"],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 0, "clean": 1, "fix_required": 0, "total": 1}
    validation = json.loads((review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert validation["split"] == SPLIT_CLEAN
    assert validation["reasons"] == ["clean"]
    assert validation["metrics"]["local_refine_max_delta_ms"] == 22.0


def test_mfa_free_oto_review_cli_split_flags_timing_clamp_attention(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [
                    {
                        "raw_line": "a.wav=a,0,100,-350,50,25",
                        "warnings": [
                            "timing_clamped.consonant",
                            "timing_clamp_delta_ms:42.0",
                            "timing_clamp_large_delta",
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 1, "clean": 0, "fix_required": 0, "total": 1}
    validation = json.loads((review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert validation["split"] == SPLIT_ATTENTION_ONLY
    assert "attention.timing_clamp_large_delta" in validation["reasons"]
    assert validation["metrics"]["timing_clamp_count"] == 1.0
    assert validation["metrics"]["timing_clamp_consonant_count"] == 1.0
    assert validation["metrics"]["timing_clamp_max_delta_ms"] == 42.0


def test_evaluate_gate_blocks_default_on_for_large_timing_clamp_but_allows_acknowledged_apply(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    split = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="case",
        wav_durations_ms={"a.wav": 500.0},
        row_diagnostics={
            0: {
                "timing_clamp_count": 1.0,
                "timing_clamp_consonant_count": 1.0,
                "timing_clamp_max_delta_ms": 42.0,
                "timing_clamp_large_delta_count": 1.0,
            }
        },
    )
    validation_jsonl = Path(split.output_paths["validation_jsonl"])
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    final_path = tmp_path / "case.final.ini"
    merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(validation_jsonl),
            "--final",
            str(final_path),
            "--acknowledge-attention-only",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_proc.returncode == 0, merge_proc.stderr
    merge_payload = json.loads(merge_proc.stdout)

    apply_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(validation_jsonl),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert apply_proc.returncode == 0, apply_proc.stdout
    apply_payload = json.loads(apply_proc.stdout)
    assert apply_payload["gate_report"]["apply_candidate"] is True
    assert apply_payload["metrics"]["validation"]["timing_clamp_large_delta_rows"] == 1

    default_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(validation_jsonl),
            "--comparison-json",
            str(comparison),
            "--merge-report",
            str(merge_payload["report_path"]),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_proc.returncode == 2
    default_payload = json.loads(default_proc.stdout)
    assert default_payload["gate_report"]["default_on_candidate"] is False
    assert "no_timing_clamp_large_delta_rows" in default_payload["gate_report"]["failed_checks"]
    assert "timing_clamp_large_delta_rows_present" in default_payload["warnings"]


def test_mfa_free_oto_review_cli_split_promotes_structural_row_warnings(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=i,0,100,-350,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [
                    {
                        "raw_line": "a.wav=a,0,100,-350,50,25",
                        "warnings": ["island_count_mismatch", "preutterance_slot_shift:+1"],
                    },
                    {
                        "raw_line": "a.wav=i,0,100,-350,50,25",
                        "warnings": ["preutterance:local_gate_low:0.123"],
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 1, "clean": 0, "fix_required": 1, "total": 2}
    validations = [
        json.loads(line)
        for line in (review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert validations[0]["split"] == SPLIT_FIX_REQUIRED
    assert "fix_required.structural_island_count_mismatch" in validations[0]["reasons"]
    assert "fix_required.preutterance_slot_shift" in validations[0]["reasons"]
    assert validations[0]["metrics"]["row_warning_island_count_mismatch_count"] == 1.0
    assert validations[1]["split"] == SPLIT_ATTENTION_ONLY
    assert "attention.local_gate_low" in validations[1]["reasons"]
    assert validations[1]["metrics"]["row_warning_local_gate_low_count"] == 1.0
    summary = json.loads((review_dir / "case.summary.json").read_text(encoding="utf-8"))
    assert summary["reason_counts_by_split"][SPLIT_FIX_REQUIRED]["fix_required.preutterance_slot_shift"] == 1
    assert summary["reason_counts_by_split"][SPLIT_ATTENTION_ONLY]["attention.local_gate_low"] == 1
    assert summary["metric_positive_row_counts"]["row_warning_island_count_mismatch_count"] == 1
    assert summary["metric_positive_row_counts"]["row_warning_local_gate_low_count"] == 1

    validation_jsonl = review_dir / "case.validation.jsonl"
    comparison = tmp_path / "comparison.json"
    _write_source_backed_comparison_json(
        comparison,
        baseline_oto=generated,
        hsmm_oto=generated,
        hsmm_validation_jsonl=validation_jsonl,
    )
    gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(validation_jsonl),
            "--comparison-json",
            str(comparison),
            "--gate-target",
            "default-on",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_proc.returncode == 2
    gate_payload = json.loads(gate_proc.stdout)
    validation_metrics = gate_payload["metrics"]["validation"]
    assert validation_metrics["structural_warning_rows"] == 1
    assert validation_metrics["structural_island_count_mismatch_rows"] == 1
    assert validation_metrics["preutterance_slot_shift_rows"] == 1
    assert validation_metrics["local_gate_attention_rows"] == 1
    assert validation_metrics["local_gate_low_rows"] == 1
    assert "no_structural_warning_rows" in gate_payload["gate_report"]["failed_checks"]
    assert "structural_warning_rows_present" in gate_payload["warnings"]
    assert "local_gate_attention_rows_present" in gate_payload["warnings"]


def test_mfa_free_oto_review_cli_maps_timeline_diagnostics_by_row_plan_slot(tmp_path):
    generated = tmp_path / "generated.ini"
    generated.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-350,50,25",
                "a.wav=i,0,100,-350,50,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timeline = tmp_path / "case.timeline.jsonl"
    timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "adapted_rows": [
                    {
                        "raw_line": "a.wav=a,0,100,-350,50,25",
                        "row_plan": {
                            "slot_index": 0,
                            "left_slot_index": 0,
                            "right_slot_index": 0,
                            "source_row_index": 20,
                            "source_timing_trusted": False,
                        },
                    },
                    {
                        "raw_line": "a.wav=i,0,100,-350,50,25",
                        "row_plan": {
                            "slot_index": 1,
                            "left_slot_index": 1,
                            "right_slot_index": 1,
                            "source_row_index": 21,
                            "source_timing_trusted": False,
                        },
                    },
                ],
                "hsmm": {
                    "diagnostics": {
                        "state_count": 2,
                        "event_prior_state_count": 2,
                        "min_selected_vs_best_local_margin": -0.20,
                        "min_selected_vs_second_local_margin": 0.0,
                        "max_duration_z_abs": 4.0,
                        "states": [
                            {
                                "state_id": "slot0.vowel",
                                "selected_vs_best_local_margin": 0.0,
                                "selected_vs_second_local_margin": 0.2,
                                "duration_z_abs": 0.2,
                                "has_event_prior": True,
                            },
                            {
                                "state_id": "slot1.vowel",
                                "selected_vs_best_local_margin": -0.20,
                                "selected_vs_second_local_margin": -0.10,
                                "duration_z_abs": 4.0,
                                "has_event_prior": True,
                            },
                        ],
                    }
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"a.wav": 500.0}), encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "split",
            "--generated",
            str(generated),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--durations-json",
            str(durations),
            "--timeline-jsonl",
            str(timeline),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"attention_only": 0, "clean": 1, "fix_required": 1, "total": 2}
    validations = [
        json.loads(line)
        for line in (review_dir / "case.validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert validations[0]["split"] == SPLIT_CLEAN
    assert validations[0]["slot_index"] == 0
    assert validations[0]["source_row_index"] == 20
    assert validations[0]["metrics"]["hsmm_max_duration_z_abs"] == 0.2
    assert validations[1]["split"] == SPLIT_FIX_REQUIRED
    assert validations[1]["slot_index"] == 1
    assert validations[1]["source_row_index"] == 21
    assert "fix_required.hsmm_local_margin_bad" in validations[1]["reasons"]
    assert "fix_required.hsmm_duration_prior_violation" in validations[1]["reasons"]


def test_mfa_free_oto_review_cli_generate_split_smoke(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    _write_tone_wav(wav_path, duration_s=0.35)
    template = tmp_path / "template.ini"
    template.write_text("a.wav=ka,999,888,-777,666,555\n", encoding="utf-8")
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--wav-dir",
            str(wav_dir),
            "--template-oto",
            str(template),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--encoder",
            "acoustic",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["processed"] == 1
    assert payload["split_counts"]["total"] == 1
    generated = review_dir / "case.generated.ini"
    assert generated.is_file()
    generated_text = generated.read_text(encoding="utf-8")
    assert "a.wav=ka," in generated_text
    assert "999,888,-777,666,555" not in generated_text
    validation_jsonl = Path(payload["split_output_paths"]["validation_jsonl"])
    assert validation_jsonl.is_file()
    validation_row = json.loads(validation_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert validation_row["source_row_index"] == 0
    assert validation_row["source_timing_trusted"] is False
    timeline_record = json.loads(Path(payload["timeline_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert timeline_record["adapted_rows"][0]["row_plan"]["source_row_index"] == 0
    assert timeline_record["adapted_rows"][0]["row_plan"]["source_timing_trusted"] is False


def test_filename_row_plan_generates_cvvc_alias_order_without_template():
    records = build_filename_row_plan("va_vi_vu.wav", language="japanese", format_type="cvvc")

    assert [record.alias for record in records] == ["va", "a v", "vi", "i v", "vu"]
    assert [record.role_family for record in records] == ["cv", "vc", "cv", "vc", "cv"]
    assert all(record.source_timing_trusted is False for record in records)
    assert [(record.left_slot_index, record.right_slot_index) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
    ]

    slots = build_filename_slots("ga_geu_geo.wav", language="korean", format_type="cvvc")
    assert [slot.token for slot in slots] == ["ga", "geu", "geo"]
    assert filename_phone_sequence_from_slots(slots) == ("g", "a", "g", "eu", "g", "eo")


def test_korean_cvc_filename_slots_keep_coda_phones_for_template_alignment():
    slots = build_filename_slots(
        "mam'mim'mum'mem'mom'meum'meom'mam.wav",
        language="korean",
        format_type="cvc",
    )

    assert [(slot.token, slot.onset, slot.vowel, slot.coda_phones, slot.phones) for slot in slots] == [
        ("ma", "m", "a", ("m",), ("m", "a", "m")),
        ("mi", "m", "i", ("m",), ("m", "i", "m")),
        ("mu", "m", "u", ("m",), ("m", "u", "m")),
        ("me", "m", "e", ("m",), ("m", "e", "m")),
        ("mo", "m", "o", ("m",), ("m", "o", "m")),
        ("meu", "m", "eu", ("m",), ("m", "eu", "m")),
        ("meo", "m", "eo", ("m",), ("m", "eo", "m")),
        ("ma", "m", "a", ("m",), ("m", "a", "m")),
    ]
    assert filename_phone_sequence_from_slots(slots) == (
        "m",
        "a",
        "m",
        "m",
        "i",
        "m",
        "m",
        "u",
        "m",
        "m",
        "e",
        "m",
        "m",
        "o",
        "m",
        "m",
        "eu",
        "m",
        "m",
        "eo",
        "m",
        "m",
        "a",
        "m",
    )


def test_korean_cvc_template_aliases_target_coda_rows_without_one_step_shift():
    from core.mfa_free_oto.oto_adapter import (
        OtoTemplateRow,
        OtoTiming,
        _assign_alias_target_indices,
        _alias_phone_sequence,
        _alias_type_for_row,
    )

    rows = [
        OtoTemplateRow("x.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0), "")
        for alias in ("ma", "am", "a m", "mi", "im", "i m", "mu")
    ]
    phones = ("m", "a", "m", "m", "i", "m", "m", "u", "m")

    assert [_alias_type_for_row(row.alias, "auto") for row in rows] == [
        "cv",
        "vc",
        "vc",
        "cv",
        "vc",
        "vc",
        "cv",
    ]
    assert _assign_alias_target_indices(rows, phones) == [1, 2, 2, 4, 5, 5, 7]

    timeline_slots = timeline_expected_slots_for_template_rows(rows, phones, language="korean")
    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in timeline_slots] == [
        (1, "a", "cv", "cv_boundary"),
        (2, "m", "vc", "phone_change"),
        (4, "i", "implicit_cv", "cv_boundary"),
        (5, "m", "vc", "phone_change"),
        (7, "u", "implicit_cv", "cv_boundary"),
    ]
    assert all(slot.phone_index not in {3, 6} for slot in timeline_slots)

    y_rows = [
        OtoTemplateRow("y.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0), "")
        for alias in ("jjya", "jjye", "jjyeo", "jjyo", "jjyu", "jjeui")
    ]
    y_phones = ("ya", "j", "j", "ya", "j", "j", "ye", "j", "j", "yeo", "j", "j", "yo", "j", "j", "yu", "j", "j", "eui")
    assert _alias_phone_sequence("jjeui") == ["j", "j", "eui"]
    assert _assign_alias_target_indices(y_rows, y_phones) == [3, 6, 9, 12, 15, 18]


def test_japanese_cvvc_kana_filename_slots_preserve_kw_and_yoon_onsets():
    kw_slots = build_filename_slots("\u304f\u3041\u304f\u3043.wav", language="japanese", format_type="cvvc")

    assert [(slot.token, slot.onset, slot.vowel, slot.phones) for slot in kw_slots] == [
        ("kwa", "kw", "a", ("k", "w", "a")),
        ("kwi", "kw", "i", ("k", "w", "i")),
    ]
    assert filename_phone_sequence_from_slots(kw_slots) == ("k", "w", "a", "k", "w", "i")

    yoon_slots = build_filename_slots("\u307f\u3083\u307f\u307f\u3085.wav", language="japanese", format_type="cvvc")
    assert [(slot.token, slot.onset, slot.vowel, slot.phones) for slot in yoon_slots] == [
        ("mya", "my", "a", ("m", "y", "a")),
        ("mi", "m", "i", ("m", "i")),
        ("myu", "my", "u", ("m", "y", "u")),
    ]

    ny_slots = build_filename_slots("\u306b\u3083\u306b\u306b\u3085.wav", language="japanese", format_type="cvvc")
    assert [(slot.token, slot.onset, slot.vowel, slot.phones) for slot in ny_slots] == [
        ("nya", "ny", "a", ("n", "y", "a")),
        ("ni", "n", "i", ("n", "i")),
        ("nyu", "ny", "u", ("n", "y", "u")),
    ]
    assert filename_phone_sequence_from_slots(ny_slots) == ("n", "y", "a", "n", "i", "n", "y", "u")


def test_filename_row_plan_generates_vcv_alias_order_without_template():
    records = build_filename_row_plan("_a'bbeo'bba.wav", language="korean", format_type="vcv")

    assert [record.alias for record in records] == ["- a", "a bbeo", "eo bba"]
    assert [record.role_family for record in records] == ["cv_head", "vcv", "vcv"]
    assert all(record.source_timing_trusted is False for record in records)
    assert [(record.left_slot_index, record.right_slot_index) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 2),
    ]

    japanese = build_filename_row_plan("va_vi_vu.wav", language="japanese", format_type="vcv")
    assert [record.alias for record in japanese] == ["- va", "a vi", "i vu"]
    assert [record.role_family for record in japanese] == ["cv_head", "vcv", "vcv"]


def test_filename_vcv_expected_slots_match_palatal_vowel_tokens():
    rows, phones, _records = build_filename_template_rows("_i'myo'i'myu'mya'i'myeo.wav", language="korean", format_type="vcv")

    slots = expected_slots_for_template_rows(rows, phones)

    assert [row.alias for row in rows] == ["- i", "i myo", "yo i", "i myu", "yu mya", "ya i", "i myeo"]
    assert phones == ("i", "m", "yo", "i", "m", "yu", "m", "ya", "i", "m", "yeo")
    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in slots] == [
        (0, "i", "cv_head", "cv_boundary"),
        (2, "yo", "vcv", "cv_boundary"),
        (3, "i", "vv", "vowel_nucleus"),
        (5, "yu", "vcv", "cv_boundary"),
        (7, "ya", "vcv", "cv_boundary"),
        (8, "i", "vv", "vowel_nucleus"),
        (10, "yeo", "vcv", "cv_boundary"),
    ]


def test_korean_vcv_row_plan_treats_standalone_y_vowels_as_vowels():
    rows, phones, records = build_filename_template_rows("_yu'yeo'yae'wae'weo.wav", language="korean", format_type="vcv")

    slots = build_filename_slots("_yu'yeo'yae'wae'weo.wav", language="korean", format_type="vcv")
    expected_slots = expected_slots_for_template_rows(rows, phones)

    assert [(slot.token, slot.onset, slot.vowel, slot.phones) for slot in slots] == [
        ("yu", "", "yu", ("yu",)),
        ("yeo", "", "yeo", ("yeo",)),
        ("yae", "", "yae", ("yae",)),
        ("wae", "", "wae", ("wae",)),
        ("weo", "", "weo", ("weo",)),
    ]
    assert [record.alias for record in records] == ["- yu", "yu yeo", "yeo yae", "yae wae", "wae weo"]
    assert [record.role_family for record in records] == ["cv_head", "vv", "vv", "vv", "vv"]
    assert phones == ("yu", "yeo", "yae", "wae", "weo")
    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in expected_slots] == [
        (0, "yu", "cv_head", "cv_boundary"),
        (1, "yeo", "vv", "vowel_nucleus"),
        (2, "yae", "vv", "vowel_nucleus"),
        (3, "wae", "vv", "vowel_nucleus"),
        (4, "weo", "vv", "vowel_nucleus"),
    ]


def test_filename_row_plan_keeps_fallback_slot_indexes_contiguous(monkeypatch):
    monkeypatch.setattr(row_plan_module, "filename_syllable_order_tokens", lambda *_args, **_kwargs: ("zzzq",))
    monkeypatch.setattr(row_plan_module, "infer_filename_phone_sequence", lambda *_args, **_kwargs: ("b", "a", "d", "a"))

    slots = row_plan_module.build_filename_slots("fallback.wav", language="korean", format_type="vcv")
    assert [slot.slot_index for slot in slots] == [0, 1]

    records = row_plan_module.build_filename_row_plan("fallback.wav", language="korean", format_type="vcv")
    assert [record.alias for record in records] == ["- ba", "a da"]


def test_filename_hsmm_adapter_decodes_slot_events_from_acoustic_posterior():
    times = [float(i * 10) for i in range(40)]
    slots = build_filename_slots("ka_ki.wav", language="japanese", format_type="cvvc")

    def active(start: float, end: float) -> list[float]:
        return [0.90 if start <= t < end else 0.05 for t in times]

    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    class_probs["consonant"] = [
        max(a, b)
        for a, b in zip(active(0.0, 70.0), active(200.0, 270.0))
    ]
    class_probs["vowel"] = [
        max(a, b)
        for a, b in zip(active(70.0, 200.0), active(270.0, 400.0))
    ]
    class_probs["silence"] = [0.02 for _ in times]
    event_scores = {label: [0.02 for _ in times] for label in EVENT_LABELS}
    event_scores["cv_boundary"] = [0.95 if t in {70.0, 270.0} else 0.02 for t in times]
    event_scores["phone_change"] = [0.95 if t in {0.0, 200.0} else 0.02 for t in times]
    event_scores["vowel_nucleus"] = [0.95 if t in {140.0, 330.0} else 0.02 for t in times]
    posterior = FramePosterior(
        wav_path="ka_ki.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "transition_likelihood": event_scores["cv_boundary"],
            "voicing": class_probs["vowel"],
            "nucleus_likelihood": event_scores["vowel_nucleus"],
        },
    )

    decoded = decode_filename_slots_with_hsmm(posterior, slots, beam_width_per_state=48, timeout_ms=1000)

    assert decoded.ok
    labels = [event["label"] for event in decoded.events]
    assert labels == [
        "phone_change",
        "cv_boundary",
        "vowel_nucleus",
        "phone_change",
        "cv_boundary",
        "vowel_nucleus",
    ]
    cv_times = [event["selected_time_ms"] for event in decoded.events if event["label"] == "cv_boundary"]
    phone_change_times = [event["selected_time_ms"] for event in decoded.events if event["label"] == "phone_change"]
    assert cv_times == pytest.approx([70.0, 270.0], abs=25.0)
    assert phone_change_times == pytest.approx([0.0, 200.0], abs=25.0)
    diagnostics = decoded.diagnostics
    assert diagnostics["state_count"] == 4
    assert diagnostics["decoded_state_count"] == 4
    assert len(diagnostics["states"]) == 4
    assert all("selected_vs_best_local_margin" in item for item in diagnostics["states"])


def test_filename_hsmm_adapter_adds_sonorant_hold_states_without_changing_events():
    times = [float(i * 10) for i in range(50)]
    slots = build_filename_slots("ma_mi.wav", language="japanese", format_type="cvvc")
    states = build_hsmm_states_for_slots(slots, duration_ms=500.0, enable_sonorant_hold=True)

    assert [state.state_id for state in states] == [
        "slot0.onset",
        "slot0.onset_hold",
        "slot0.vowel",
        "slot1.onset",
        "slot1.onset_hold",
        "slot1.vowel",
    ]

    def active(start: float, end: float) -> list[float]:
        return [0.90 if start <= t < end else 0.05 for t in times]

    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    class_probs["consonant"] = [
        max(a, b)
        for a, b in zip(active(0.0, 80.0), active(210.0, 290.0))
    ]
    class_probs["vowel"] = [
        max(a, b)
        for a, b in zip(active(80.0, 210.0), active(290.0, 430.0))
    ]
    class_probs["silence"] = [0.02 for _ in times]
    event_scores = {label: [0.02 for _ in times] for label in EVENT_LABELS}
    event_scores["phone_change"] = [0.95 if t in {0.0, 210.0} else 0.02 for t in times]
    event_scores["cv_boundary"] = [0.95 if t in {80.0, 290.0} else 0.02 for t in times]
    event_scores["vowel_nucleus"] = [0.95 if t in {150.0, 360.0} else 0.02 for t in times]
    posterior = FramePosterior(
        wav_path="ma_mi.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "transition_likelihood": event_scores["cv_boundary"],
            "onset_strength": event_scores["phone_change"],
            "sonorant_onset_likelihood": event_scores["phone_change"],
            "voicing": [0.88 if 0.0 <= t < 430.0 else 0.05 for t in times],
            "nucleus_likelihood": event_scores["vowel_nucleus"],
        },
    )

    decoded = decode_filename_slots_with_hsmm(
        posterior,
        slots,
        beam_width_per_state=48,
        timeout_ms=1000,
        use_coarse_to_refine=True,
        enable_sonorant_hold=True,
        use_vowel_start_score=True,
    )

    assert decoded.ok
    assert decoded.diagnostics["state_count"] == 6
    labels = [event["label"] for event in decoded.events]
    assert labels == [
        "phone_change",
        "cv_boundary",
        "vowel_nucleus",
        "phone_change",
        "cv_boundary",
        "vowel_nucleus",
    ]
    phone_change_times = [event["selected_time_ms"] for event in decoded.events if event["label"] == "phone_change"]
    cv_times = [event["selected_time_ms"] for event in decoded.events if event["label"] == "cv_boundary"]
    assert phone_change_times == pytest.approx([0.0, 210.0], abs=30.0)
    assert cv_times == pytest.approx([80.0, 290.0], abs=35.0)


def test_acoustic_evidence_pack_exports_soft_candidates_and_reliability(tmp_path):
    wav_path = tmp_path / "ka.wav"
    wav_path.write_bytes(b"fake-wav-for-evidence-hash")
    times = [float(i * 10) for i in range(8)]
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    class_probs["vowel"] = [0.10, 0.20, 0.88, 0.82, 0.20, 0.10, 0.05, 0.05]
    class_probs["consonant"] = [0.72, 0.64, 0.10, 0.05, 0.40, 0.62, 0.20, 0.10]
    class_probs["silence"] = [0.02 for _ in times]
    event_scores = {label: [0.02 for _ in times] for label in EVENT_LABELS}
    event_scores["cv_boundary"] = [0.05, 0.12, 0.92, 0.20, 0.08, 0.04, 0.03, 0.02]
    event_scores["vowel_nucleus"] = [0.02, 0.05, 0.20, 0.91, 0.18, 0.04, 0.02, 0.02]
    event_scores["phone_change"] = [0.88, 0.18, 0.05, 0.04, 0.70, 0.10, 0.02, 0.02]
    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "transition_likelihood": event_scores["cv_boundary"],
            "voicing": class_probs["vowel"],
            "nucleus_likelihood": event_scores["vowel_nucleus"],
            "sonorant_onset_likelihood": [0.02, 0.10, 0.55, 0.15, 0.04, 0.02, 0.02, 0.02],
        },
        metadata={
            "encoder": "acoustic",
            "rule_based": True,
            "acoustic_feature_set": "world_v1",
            "sample_rate": 44100,
            "analysis_sample_rate": 16000,
            "window_size_ms": 25.0,
            "frame_time_anchor": "left",
            "feature_latency_ms": 0.0,
            "padding_policy": "reflect",
            "resample_method": "linear",
        },
    )

    payload = build_acoustic_evidence_pack(
        posterior,
        wav_name="ka.wav",
        expected_phones=["k", "a"],
        filename_slots=build_filename_slots("ka.wav", language="japanese", format_type="cvvc"),
        runtime_events=[
            {
                "label": "cv_boundary",
                "expected_phone_index": 1,
                "selected_time_ms": 20.0,
                "score": 0.92,
            }
        ],
    )

    assert payload["schema_version"] == 1
    assert payload["wav_sha256"] == _sha256_file(wav_path)
    assert payload["extractor_version"] == "acoustic_evidence_pack_v1"
    assert payload["decoder_version"] == "runtime_event_prior_v1"
    assert validate_acoustic_evidence_pack(payload) == ()
    assert payload["timebase"]["frame_shift_ms"] == 10.0
    assert payload["timebase"]["sample_rate"] == 44100
    assert payload["timebase"]["analysis_sample_rate"] == 16000
    assert payload["timebase"]["frame_time_anchor"] == "left"
    assert payload["timebase"]["contract"]["window_size_ms"] == 25.0
    assert payload["timebase"]["contract"]["frame_time_anchor"] == "left"
    assert payload["timebase"]["contract"]["feature_latency_ms"] == 0.0
    assert payload["timebase"]["fingerprint"] == payload["timebase_fingerprint"]
    assert len(payload["timebase_fingerprint"]) == 40
    assert payload["reliability"]["rule_based"] is True
    assert payload["frame_posterior"]["event_scores"]["cv_boundary"][2] == pytest.approx(0.92)
    candidates = payload["event_candidates"]
    cv = next(item for item in candidates if item["event_type"] == "cv_boundary")
    assert cv["time_ms"] == 20.0
    assert cv["margin"] > 0.70
    assert "feature_agreement" in cv
    assert any(item["event_type"] == "sonorant_constriction" for item in candidates)
    summary = payload["event_candidate_summary"]
    assert summary["event_candidate_count"] == len(candidates)
    assert summary["event_candidate_type_counts"]["cv_boundary"] >= 1
    assert summarize_event_candidate_reliability(payload)["event_candidate_count"] == len(candidates)
    replayed = frame_posterior_from_evidence_pack(payload)
    replayed_slots = filename_slots_from_evidence_pack(payload, language="japanese", format_type="cvvc")
    assert replayed.times_ms == pytest.approx(times)
    assert expected_phones_from_evidence_pack(payload) == ("k", "a")
    assert [slot.token for slot in replayed_slots] == ["ka"]
    assert runtime_events_from_evidence_pack(payload)[0]["expected_phone_index"] == 1

    stale_payload = dict(payload)
    stale_payload["schema_version"] = 999
    assert "evidence.schema_version_unsupported" in validate_acoustic_evidence_pack(stale_payload)

    missing_version_payload = dict(payload)
    missing_version_payload.pop("extractor_version", None)
    assert "evidence.extractor_version_missing" in validate_acoustic_evidence_pack(missing_version_payload)

    unsupported_version_payload = dict(payload)
    unsupported_version_payload["decoder_version"] = "future_decoder_v999"
    assert "evidence.decoder_version_unsupported" in validate_acoustic_evidence_pack(unsupported_version_payload)

    missing_timebase_contract_payload = json.loads(json.dumps(payload))
    missing_timebase_contract_payload["timebase"].pop("contract")
    assert "evidence.timebase.contract_missing" in validate_acoustic_evidence_pack(missing_timebase_contract_payload)

    missing_top_level_fingerprint_payload = json.loads(json.dumps(payload))
    missing_top_level_fingerprint_payload.pop("timebase_fingerprint")
    assert "evidence.timebase_fingerprint_missing" in validate_acoustic_evidence_pack(
        missing_top_level_fingerprint_payload
    )

    invalid_top_level_fingerprint_payload = json.loads(json.dumps(payload))
    invalid_top_level_fingerprint_payload["timebase_fingerprint"] = "not-a-sha1"
    assert "evidence.timebase_fingerprint_invalid" in validate_acoustic_evidence_pack(
        invalid_top_level_fingerprint_payload
    )

    mismatched_top_level_fingerprint_payload = json.loads(json.dumps(payload))
    mismatched_top_level_fingerprint_payload["timebase_fingerprint"] = "0" * 40
    assert "evidence.timebase_fingerprint_mismatch" in validate_acoustic_evidence_pack(
        mismatched_top_level_fingerprint_payload
    )

    future_timebase_contract_payload = json.loads(json.dumps(payload))
    future_timebase_contract_payload["timebase"]["contract"]["schema_version"] = 999
    future_timebase_contract = timebase_from_mapping(future_timebase_contract_payload["timebase"]["contract"])
    future_timebase_fingerprint = future_timebase_contract.cache_fingerprint()
    future_timebase_contract_payload["timebase"]["fingerprint"] = future_timebase_fingerprint
    future_timebase_contract_payload["timebase_fingerprint"] = future_timebase_fingerprint
    assert "evidence.timebase.contract_schema_version_unsupported" in validate_acoustic_evidence_pack(
        future_timebase_contract_payload
    )

    missing_timebase_schema_payload = json.loads(json.dumps(payload))
    missing_timebase_schema_payload["timebase"]["contract"].pop("schema_version")
    missing_timebase_schema_contract = timebase_from_mapping(missing_timebase_schema_payload["timebase"]["contract"])
    missing_timebase_schema_fingerprint = missing_timebase_schema_contract.cache_fingerprint()
    missing_timebase_schema_payload["timebase"]["fingerprint"] = missing_timebase_schema_fingerprint
    missing_timebase_schema_payload["timebase_fingerprint"] = missing_timebase_schema_fingerprint
    assert "evidence.timebase.contract_schema_version_missing" in validate_acoustic_evidence_pack(
        missing_timebase_schema_payload
    )

    invalid_timebase_schema_payload = json.loads(json.dumps(payload))
    invalid_timebase_schema_payload["timebase"]["contract"]["schema_version"] = "v1"
    invalid_timebase_schema_contract = timebase_from_mapping(invalid_timebase_schema_payload["timebase"]["contract"])
    invalid_timebase_schema_fingerprint = invalid_timebase_schema_contract.cache_fingerprint()
    invalid_timebase_schema_payload["timebase"]["fingerprint"] = invalid_timebase_schema_fingerprint
    invalid_timebase_schema_payload["timebase_fingerprint"] = invalid_timebase_schema_fingerprint
    assert "evidence.timebase.contract_schema_version_invalid" in validate_acoustic_evidence_pack(
        invalid_timebase_schema_payload
    )

    missing_summary_sample_rate_payload = json.loads(json.dumps(payload))
    missing_summary_sample_rate_payload["timebase"].pop("sample_rate")
    assert "evidence.timebase.sample_rate_missing" in validate_acoustic_evidence_pack(
        missing_summary_sample_rate_payload
    )

    missing_summary_latency_payload = json.loads(json.dumps(payload))
    missing_summary_latency_payload["timebase"].pop("feature_latency_ms")
    assert "evidence.timebase.feature_latency_ms_missing" in validate_acoustic_evidence_pack(
        missing_summary_latency_payload
    )

    invalid_summary_sample_rate_payload = json.loads(json.dumps(payload))
    invalid_summary_sample_rate_payload["timebase"]["sample_rate"] = 0
    invalid_summary_sample_rate_payload["timebase"]["contract"]["sample_rate"] = 0
    invalid_summary_sample_rate_contract = timebase_from_mapping(
        invalid_summary_sample_rate_payload["timebase"]["contract"]
    )
    invalid_summary_sample_rate_fingerprint = invalid_summary_sample_rate_contract.cache_fingerprint()
    invalid_summary_sample_rate_payload["timebase"]["fingerprint"] = invalid_summary_sample_rate_fingerprint
    invalid_summary_sample_rate_payload["timebase_fingerprint"] = invalid_summary_sample_rate_fingerprint
    assert "evidence.timebase.sample_rate_invalid" in validate_acoustic_evidence_pack(
        invalid_summary_sample_rate_payload
    )

    invalid_summary_analysis_rate_payload = json.loads(json.dumps(payload))
    invalid_summary_analysis_rate_payload["timebase"]["analysis_sample_rate"] = 0
    invalid_summary_analysis_rate_payload["timebase"]["contract"]["analysis_sample_rate"] = 0
    invalid_summary_analysis_rate_contract = timebase_from_mapping(
        invalid_summary_analysis_rate_payload["timebase"]["contract"]
    )
    invalid_summary_analysis_rate_fingerprint = invalid_summary_analysis_rate_contract.cache_fingerprint()
    invalid_summary_analysis_rate_payload["timebase"]["fingerprint"] = invalid_summary_analysis_rate_fingerprint
    invalid_summary_analysis_rate_payload["timebase_fingerprint"] = invalid_summary_analysis_rate_fingerprint
    assert "evidence.timebase.analysis_sample_rate_invalid" in validate_acoustic_evidence_pack(
        invalid_summary_analysis_rate_payload
    )

    invalid_summary_anchor_payload = json.loads(json.dumps(payload))
    invalid_summary_anchor_payload["timebase"]["frame_time_anchor"] = "middle"
    invalid_summary_anchor_payload["timebase"]["contract"]["frame_time_anchor"] = "middle"
    invalid_summary_anchor_contract = timebase_from_mapping(invalid_summary_anchor_payload["timebase"]["contract"])
    invalid_summary_anchor_fingerprint = invalid_summary_anchor_contract.cache_fingerprint()
    invalid_summary_anchor_payload["timebase"]["fingerprint"] = invalid_summary_anchor_fingerprint
    invalid_summary_anchor_payload["timebase_fingerprint"] = invalid_summary_anchor_fingerprint
    assert "evidence.timebase.frame_time_anchor_invalid" in validate_acoustic_evidence_pack(
        invalid_summary_anchor_payload
    )

    invalid_summary_latency_payload = json.loads(json.dumps(payload))
    invalid_summary_latency_payload["timebase"]["feature_latency_ms"] = "late"
    invalid_summary_latency_payload["timebase"]["contract"]["feature_latency_ms"] = "late"
    invalid_summary_latency_contract = timebase_from_mapping(invalid_summary_latency_payload["timebase"]["contract"])
    invalid_summary_latency_fingerprint = invalid_summary_latency_contract.cache_fingerprint()
    invalid_summary_latency_payload["timebase"]["fingerprint"] = invalid_summary_latency_fingerprint
    invalid_summary_latency_payload["timebase_fingerprint"] = invalid_summary_latency_fingerprint
    assert "evidence.timebase.feature_latency_ms_invalid" in validate_acoustic_evidence_pack(
        invalid_summary_latency_payload
    )

    mismatched_timebase_payload = json.loads(json.dumps(payload))
    mismatched_timebase_payload["timebase"]["contract"]["frame_shift_ms"] = 5.0
    assert "evidence.timebase.fingerprint_mismatch" in validate_acoustic_evidence_pack(mismatched_timebase_payload)
    assert "evidence.timebase.contract_frame_shift_mismatch" in validate_acoustic_evidence_pack(
        mismatched_timebase_payload
    )

    mismatched_anchor_payload = json.loads(json.dumps(payload))
    mismatched_anchor_payload["timebase"]["frame_time_anchor"] = "center"
    assert "evidence.timebase.contract_frame_time_anchor_mismatch" in validate_acoustic_evidence_pack(
        mismatched_anchor_payload
    )

    mismatched_latency_payload = json.loads(json.dumps(payload))
    mismatched_latency_payload["timebase"]["feature_latency_ms"] = 12.5
    assert "evidence.timebase.contract_feature_latency_mismatch" in validate_acoustic_evidence_pack(
        mismatched_latency_payload
    )

    mismatched_resample_payload = json.loads(json.dumps(payload))
    mismatched_resample_payload["timebase"]["resample_method"] = "soxr"
    assert "evidence.timebase.contract_resample_method_mismatch" in validate_acoustic_evidence_pack(
        mismatched_resample_payload
    )

    missing_track_payload = json.loads(json.dumps(payload))
    missing_track_payload["frame_posterior"]["class_probs"].pop("vowel")
    assert "frame_posterior.class_probs.missing:vowel" in validate_acoustic_evidence_pack(missing_track_payload)

    wav_path.write_bytes(b"tampered-wav-for-evidence-hash")
    assert "evidence.wav_sha256_mismatch" in validate_acoustic_evidence_pack(payload)


def test_filename_hsmm_adapter_uses_runtime_event_prior_for_single_vowel():
    times = [float(i * 10) for i in range(80)]
    slots = build_filename_slots("i.wav", language="japanese", format_type="cvvc")
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    class_probs["vowel"] = [0.80 if 40.0 <= t < 220.0 else 0.20 for t in times]
    class_probs["silence"] = [0.02 for _ in times]
    event_scores = {label: [0.02 for _ in times] for label in EVENT_LABELS}
    posterior = FramePosterior(
        wav_path="i.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "voicing": class_probs["vowel"],
            "nucleus_likelihood": [0.02 for _ in times],
        },
    )

    baseline = decode_filename_slots_with_hsmm(posterior, slots, beam_width_per_state=80, timeout_ms=1000)
    with_prior = decode_filename_slots_with_hsmm(
        posterior,
        slots,
        beam_width_per_state=80,
        timeout_ms=1000,
        event_priors=[
            {
                "label": "vowel_nucleus",
                "expected_phone_index": 0,
                "selected_time_ms": 520.0,
                "score": 1.0,
            }
        ],
    )

    assert baseline.ok
    assert with_prior.ok
    baseline_nucleus = next(event["selected_time_ms"] for event in baseline.events if event["label"] == "vowel_nucleus")
    prior_nucleus = next(event["selected_time_ms"] for event in with_prior.events if event["label"] == "vowel_nucleus")
    assert baseline_nucleus == pytest.approx(155.5, abs=20.0)
    assert prior_nucleus == pytest.approx(520.0, abs=25.0)
    assert prior_nucleus - baseline_nucleus > 300.0
    assert with_prior.result.meta["leading_gap_ms"] > baseline.result.meta["leading_gap_ms"]
    assert with_prior.diagnostics["event_prior_state_count"] == 1
    assert with_prior.diagnostics["event_prior_input_count"] == 1
    assert with_prior.diagnostics["event_prior_mapped_count"] == 1
    assert with_prior.diagnostics["event_prior_source_counts"]["runtime_event"] == 1
    assert with_prior.diagnostics["runtime_prior_count"] == 1
    assert with_prior.diagnostics["states"][0]["event_prior_selected_mean"] > 0.0
    assert with_prior.diagnostics["states"][0]["event_prior_source_counts"]["runtime_event"] == 1


def test_evidence_candidate_priors_can_steer_filename_hsmm():
    times = [float(i * 10) for i in range(80)]
    slots = build_filename_slots("i.wav", language="japanese", format_type="cvvc")
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    class_probs["vowel"] = [0.80 if 40.0 <= t < 220.0 else 0.20 for t in times]
    class_probs["silence"] = [0.02 for _ in times]
    event_scores = {label: [0.02 for _ in times] for label in EVENT_LABELS}
    posterior = FramePosterior(
        wav_path="i.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "voicing": class_probs["vowel"],
            "nucleus_likelihood": [0.02 for _ in times],
        },
    )
    evidence = {
        "timebase": {"duration_ms": 800.0},
        "event_candidates": [
            {
                "event_type": "vowel_nucleus",
                "time_ms": 520.0,
                "score": 1.0,
                "margin": 0.5,
                "feature_agreement": 0.9,
                "flags": [],
            },
            {
                "event_type": "vowel_nucleus",
                "time_ms": 120.0,
                "score": 0.9,
                "margin": 0.01,
                "feature_agreement": 0.9,
                "flags": ["low_margin"],
            },
        ],
    }

    candidate_priors = candidate_priors_from_evidence_pack(evidence, slots)
    assert len(candidate_priors) == 1
    assert candidate_priors[0]["source"] == "evidence_candidate"
    assert candidate_priors[0]["label"] == "vowel_nucleus"
    assert candidate_priors[0]["expected_phone_index"] == 0

    baseline = decode_filename_slots_with_hsmm(posterior, slots, beam_width_per_state=80, timeout_ms=1000)
    with_candidate = decode_filename_slots_with_hsmm(
        posterior,
        slots,
        beam_width_per_state=80,
        timeout_ms=1000,
        event_priors=candidate_priors,
    )

    assert baseline.ok
    assert with_candidate.ok
    baseline_nucleus = next(event["selected_time_ms"] for event in baseline.events if event["label"] == "vowel_nucleus")
    candidate_nucleus = next(event["selected_time_ms"] for event in with_candidate.events if event["label"] == "vowel_nucleus")
    assert baseline_nucleus == pytest.approx(155.5, abs=20.0)
    assert candidate_nucleus == pytest.approx(520.0, abs=25.0)
    assert with_candidate.diagnostics["event_prior_state_count"] == 1
    assert with_candidate.diagnostics["event_prior_input_count"] == 1
    assert with_candidate.diagnostics["event_prior_mapped_count"] == 1
    assert with_candidate.diagnostics["candidate_prior_count"] == 1
    assert with_candidate.diagnostics["event_prior_source_counts"]["evidence_candidate"] == 1
    assert with_candidate.diagnostics["event_prior_label_counts"]["vowel_nucleus"] == 1
    assert with_candidate.diagnostics["event_prior_summary"]["states"]["slot0.vowel"]["source_counts"]["evidence_candidate"] == 1
    assert with_candidate.diagnostics["states"][0]["evidence_candidate_prior_count"] == 1
    assert with_candidate.diagnostics["states"][0]["event_prior_label_counts"]["vowel_nucleus"] == 1


def test_mfa_free_oto_review_cli_generate_uses_filename_row_plan_without_template(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "va_vi_vu.wav", duration_s=0.55, sample_rate=44100)
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--wav-dir",
            str(wav_dir),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--format-type",
            "cvvc",
            "--encoder",
            "acoustic",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["processed"] == 5
    assert payload["split_counts"]["total"] == 5
    assert payload["row_plan_rows"] == 5
    generated_oto_path = Path(payload["generated_oto_path"])
    assert generated_oto_path.is_file()
    assert payload["generated_oto_sha256"] == _sha256_file(generated_oto_path)
    review_session_path = Path(payload["split_output_paths"]["review_session"])
    assert review_session_path.is_file()
    review_session = json.loads(review_session_path.read_text(encoding="utf-8"))
    assert review_session["metadata"]["created_by"] == "mfa_free_oto_review.generate"
    assert review_session["metadata"]["source_timing_trusted"] is False
    assert review_session["apply_policy"]["requires_evaluate_gate"] is True
    metadata_artifacts = review_session["lineage"]["metadata_artifacts"]
    row_plan_path = Path(payload["row_plan_path"])
    assert row_plan_path.is_file()
    assert payload["row_plan_sha256"] == _sha256_file(row_plan_path)
    assert metadata_artifacts["row_plan"]["path"] == str(row_plan_path)
    assert metadata_artifacts["row_plan"]["sha256"] == _sha256_file(row_plan_path)
    assert metadata_artifacts["row_plan"]["row_count"] == 5
    row_plan_rows = [json.loads(line) for line in row_plan_path.read_text(encoding="utf-8").splitlines()]
    assert len(row_plan_rows) == 5
    assert [row["alias"] for row in row_plan_rows] == ["va", "a v", "vi", "i v", "vu"]
    assert all(row["source_timing_trusted"] is False for row in row_plan_rows)
    timeline_path = Path(payload["timeline_path"])
    assert timeline_path.is_file()
    assert payload["timeline_sha256"] == _sha256_file(timeline_path)
    assert metadata_artifacts["timeline"]["path"] == str(timeline_path)
    assert metadata_artifacts["timeline"]["sha256"] == _sha256_file(timeline_path)
    timeline_record = json.loads(timeline_path.read_text(encoding="utf-8").splitlines()[0])
    assert timeline_record["adapted_rows"][0]["row_plan"]["source_timing_trusted"] is False
    assert "source_timing" not in timeline_record["adapted_rows"][0]
    validation_jsonl_path = Path(payload["split_output_paths"]["validation_jsonl"])
    validation_rows = [json.loads(line) for line in validation_jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [row["slot_index"] for row in validation_rows] == [0, 0, 1, 1, 2]
    assert [(row["left_slot_index"], row["right_slot_index"]) for row in validation_rows] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    assert all(row["source_timing_trusted"] is False for row in validation_rows)
    assert timeline_record["evidence"]["wav_sha256"]
    assert timeline_record["evidence"]["extractor_version"] == "acoustic_evidence_pack_v1"
    assert timeline_record["evidence"]["event_candidate_summary"]["event_candidate_count"] > 0
    evidence_path = Path(payload["evidence_path"])
    assert evidence_path.is_file()
    assert payload["evidence_sha256"] == _sha256_file(evidence_path)
    assert metadata_artifacts["evidence"]["path"] == str(evidence_path)
    assert metadata_artifacts["evidence"]["sha256"] == _sha256_file(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
    assert evidence["wav"] == "va_vi_vu.wav"
    assert evidence["wav_sha256"] == _sha256_file(wav_dir / "va_vi_vu.wav")
    assert evidence["extractor_version"] == "acoustic_evidence_pack_v1"
    assert evidence["decoder_version"] == "runtime_event_prior_v1"
    assert evidence["timebase"]["frame_count"] > 0
    assert evidence["timebase"]["contract"]["sample_rate"] == 44100
    assert evidence["timebase"]["contract"]["analysis_sample_rate"] == 16000
    assert evidence["timebase"]["contract"]["resample_method"] == "linear"
    assert evidence["timebase"]["contract"]["frame_time_anchor"] == "left"
    assert evidence["timebase"]["contract"]["feature_latency_ms"] == 0.0
    assert evidence["timebase"]["fingerprint"] == evidence["timebase_fingerprint"]
    assert evidence["event_candidates"]
    assert evidence["event_candidate_summary"]["event_candidate_count"] == len(evidence["event_candidates"])
    assert evidence["filename_slots"]
    assert evidence["runtime_events"]
    assert validate_acoustic_evidence_pack(evidence) == ()

    invalid_evidence_path = review_dir / "invalid.evidence.jsonl"
    invalid_evidence = dict(evidence)
    invalid_evidence["schema_version"] = 999
    invalid_evidence_path.write_text(json.dumps(invalid_evidence, ensure_ascii=False) + "\n", encoding="utf-8")
    invalid_replay_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(invalid_evidence_path),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert invalid_replay_proc.returncode == 2
    invalid_replay = json.loads(invalid_replay_proc.stdout)
    assert invalid_replay["reason"] == "invalid_evidence_schema"
    assert invalid_replay["invalid_evidence_count"] == 1
    assert "evidence.schema_version_unsupported" in invalid_replay["invalid_evidence_rows"][0]["errors"]

    stale_wav_evidence_path = review_dir / "stale_wav.evidence.jsonl"
    stale_wav_evidence = dict(evidence)
    stale_wav_evidence["wav_sha256"] = "0" * 64
    stale_wav_evidence_path.write_text(json.dumps(stale_wav_evidence, ensure_ascii=False) + "\n", encoding="utf-8")
    stale_wav_replay_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(stale_wav_evidence_path),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_wav_replay_proc.returncode == 2
    stale_wav_replay = json.loads(stale_wav_replay_proc.stdout)
    assert stale_wav_replay["reason"] == "invalid_evidence_schema"
    assert "evidence.wav_sha256_mismatch" in stale_wav_replay["invalid_evidence_rows"][0]["errors"]

    stale_version_evidence_path = review_dir / "stale_version.evidence.jsonl"
    stale_version_evidence = dict(evidence)
    stale_version_evidence["extractor_version"] = "old_extractor_v0"
    stale_version_evidence_path.write_text(json.dumps(stale_version_evidence, ensure_ascii=False) + "\n", encoding="utf-8")
    stale_version_replay_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(stale_version_evidence_path),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_version_replay_proc.returncode == 2
    stale_version_replay = json.loads(stale_version_replay_proc.stdout)
    assert stale_version_replay["reason"] == "invalid_evidence_schema"
    assert "evidence.extractor_version_unsupported" in stale_version_replay["invalid_evidence_rows"][0]["errors"]

    future_timebase_evidence_path = review_dir / "future_timebase.evidence.jsonl"
    future_timebase_evidence = json.loads(json.dumps(evidence))
    future_timebase_evidence["timebase"]["contract"]["schema_version"] = 999
    future_timebase_contract = timebase_from_mapping(future_timebase_evidence["timebase"]["contract"])
    future_timebase_fingerprint = future_timebase_contract.cache_fingerprint()
    future_timebase_evidence["timebase"]["fingerprint"] = future_timebase_fingerprint
    future_timebase_evidence["timebase_fingerprint"] = future_timebase_fingerprint
    future_timebase_evidence_path.write_text(
        json.dumps(future_timebase_evidence, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    future_timebase_replay_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(future_timebase_evidence_path),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert future_timebase_replay_proc.returncode == 2
    future_timebase_replay = json.loads(future_timebase_replay_proc.stdout)
    assert future_timebase_replay["reason"] == "invalid_evidence_schema"
    assert "evidence.timebase.contract_schema_version_unsupported" in (
        future_timebase_replay["invalid_evidence_rows"][0]["errors"]
    )

    replay_report = review_dir / "case.evidence_hsmm_replay.json"
    replay_timeline = review_dir / "case.evidence_hsmm_replay.timeline.jsonl"
    replay_oto = review_dir / "case.evidence_hsmm_replay.generated.ini"
    replay_split = review_dir / "case.evidence_hsmm_replay.split"
    replay_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--out",
            str(replay_report),
            "--timeline-out",
            str(replay_timeline),
            "--assembled-oto-out",
            str(replay_oto),
            "--split-out-dir",
            str(replay_split),
            "--split-name",
            "case_replay",
            "--reference-generated",
            str(payload["generated_oto_path"]),
            "--reference-validation-jsonl",
            str(payload["split_output_paths"]["validation_jsonl"]),
            "--reference-timeline",
            str(payload["timeline_path"]),
            "--row-delta-limit",
            "0",
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert replay_proc.returncode == 0, replay_proc.stderr
    replay_payload = json.loads(replay_proc.stdout)
    assert replay_payload["ok"] is True
    assert replay_payload["decoded_ok"] == 1
    assert replay_payload["evidence_jsonl_sha256"] == _sha256_file(evidence_path)
    assert replay_payload["timeline_sha256"] == _sha256_file(replay_timeline)
    assert replay_payload["assembled_oto_sha256"] == _sha256_file(replay_oto)
    assert replay_payload["assembled_rows"] == 5
    assert replay_payload["split_counts"]["total"] == 5
    assert replay_payload["split_output_paths"]["validation_jsonl"].endswith("case_replay.validation.jsonl")
    replay_session = json.loads(
        Path(replay_payload["split_output_paths"]["review_session"]).read_text(encoding="utf-8")
    )
    assert replay_session["metadata"]["created_by"] == "mfa_free_oto_review.replay-evidence-hsmm"
    assert replay_session["metadata"]["evidence_jsonl_sha256"] == _sha256_file(evidence_path)
    row_param_delta = replay_payload["reference_comparison"]["row_param_delta"]
    assert row_param_delta["available"] is True
    assert row_param_delta["matched_rows"] == 5
    assert row_param_delta["baseline_oto_sha256"] == _sha256_file(generated_oto_path)
    assert row_param_delta["hsmm_oto_sha256"] == _sha256_file(replay_oto)
    assert replay_payload["reference_comparison"]["timeline_event_delta"]["available"] is True
    assert (
        replay_payload["reference_comparison"]["timeline_event_delta"]["hsmm_timeline_sha256"]
        == _sha256_file(replay_timeline)
    )
    assert replay_report.is_file()
    assert replay_timeline.is_file()
    assert replay_oto.is_file()
    assert (replay_split / "case_replay.validation.jsonl").is_file()
    replay_record = json.loads(replay_timeline.read_text(encoding="utf-8").splitlines()[0])
    assert replay_record["source"] == "evidence_replay"
    assert replay_record["selected_event_source"] == "filename_hsmm"
    assert replay_record["evidence"]["wav_sha256"] == evidence["wav_sha256"]
    assert replay_record["evidence"]["extractor_version"] == "acoustic_evidence_pack_v1"
    assert replay_record["evidence"]["event_candidate_summary"]["event_candidate_count"] == len(
        evidence["event_candidates"]
    )
    assert replay_record["selected_events"]
    assert len(replay_record["adapted_rows"]) == 5
    assert replay_record["hsmm"]["diagnostics"]["event_prior_state_count"] >= 1
    replay_oto_text = replay_oto.read_text(encoding="utf-8")
    assert "va_vi_vu.wav=va," in replay_oto_text
    assert "va_vi_vu.wav=a v," in replay_oto_text
    assert "va_vi_vu.wav=i v," in replay_oto_text
    assert "va_vi_vu.wav=vu," in replay_oto_text

    gate_report = review_dir / "case.evidence_hsmm_replay.gate.json"
    gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--out",
            str(gate_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_proc.returncode == 0, gate_proc.stderr
    gate_payload = json.loads(gate_proc.stdout)
    assert gate_payload["ok"] is True
    assert gate_payload["metrics"]["comparison"]["matched_rows"] == 5
    assert gate_payload["metrics"]["comparison"]["evidence_jsonl_current_match"] is True
    assert gate_payload["metrics"]["comparison"]["evidence_jsonl_rows_valid"] is True
    assert gate_payload["metrics"]["comparison"]["evidence_wav_sha256_current_match"] is True
    assert gate_payload["metrics"]["comparison"]["timeline_jsonl_current_match"] is True
    assert gate_payload["metrics"]["comparison"]["oto_current_match"] is True
    assert gate_payload["gate_report"]["apply_candidate"] is False
    assert gate_payload["gate_report"]["default_on_candidate"] is False
    assert "manual_or_listening_evidence_missing" in gate_payload["warnings"]
    assert "merge_report_missing" in gate_payload["warnings"]
    assert gate_report.is_file()

    replay_final = review_dir / "case.evidence_hsmm_replay.final.ini"
    replay_fix_required = replay_split / "case_replay.fix_required.ini"
    replay_edited_fix_required = replay_split / "case_replay.edited_fix_required.ini"
    edited_fix_lines: list[str] = []
    for line in replay_fix_required.read_text(encoding="utf-8").splitlines():
        filename, rest = line.split("=", 1)
        fields = rest.split(",")
        fields[1] = f"{float(fields[1]) + 0.001:.3f}"
        edited_fix_lines.append(f"{filename}={','.join(fields)}")
    replay_edited_fix_required.write_text("\n".join(edited_fix_lines) + "\n", encoding="utf-8")
    merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "merge",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--final",
            str(replay_final),
            "--edited-fix-required",
            str(replay_edited_fix_required),
            "--acknowledge-attention-only",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert merge_proc.returncode == 0, merge_proc.stderr
    merge_payload = json.loads(merge_proc.stdout)
    assert merge_payload["ok"] is True
    merge_report = Path(merge_payload["report_path"])
    assert replay_final.is_file()
    assert merge_report.is_file()

    gate_with_merge_report = review_dir / "case.evidence_hsmm_replay.apply.gate.json"
    gate_with_merge_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--out",
            str(gate_with_merge_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert gate_with_merge_proc.returncode == 0, gate_with_merge_proc.stderr
    gate_with_merge_payload = json.loads(gate_with_merge_proc.stdout)
    assert gate_with_merge_payload["gate_report"]["apply_candidate"] is True
    assert gate_with_merge_payload["gate_report"]["default_on_candidate"] is False
    assert gate_with_merge_payload["metrics"]["comparison"]["row_param_delta_current_match"] is True
    assert gate_with_merge_payload["metrics"]["comparison"]["timeline_event_delta_current_match"] is True
    assert gate_with_merge_payload["metrics"]["comparison"]["prior_provenance_required"] is True
    assert gate_with_merge_payload["metrics"]["comparison"]["prior_provenance_available"] is True
    assert gate_with_merge_report.is_file()

    future_timebase_report = review_dir / "case.evidence_hsmm_replay.future_timebase.json"
    future_timebase_report_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    future_timebase_report_payload["evidence_jsonl"] = str(future_timebase_evidence_path)
    future_timebase_report_payload["evidence_jsonl_sha256"] = _sha256_file(future_timebase_evidence_path)
    future_timebase_report.write_text(
        json.dumps(future_timebase_report_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    future_timebase_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(future_timebase_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert future_timebase_gate_proc.returncode == 2
    future_timebase_gate = json.loads(future_timebase_gate_proc.stdout)
    assert "comparison_evidence_jsonl_rows_valid" in future_timebase_gate["gate_report"]["failed_checks"]
    assert "comparison_evidence_jsonl_rows_invalid" in future_timebase_gate["warnings"]
    assert future_timebase_gate["metrics"]["comparison"]["evidence_jsonl_current_match"] is True
    assert future_timebase_gate["metrics"]["comparison"]["evidence_jsonl_rows_valid"] is False
    assert future_timebase_gate["metrics"]["comparison"]["evidence_jsonl_invalid_count"] == 1
    assert "evidence.timebase.contract_schema_version_unsupported" in (
        future_timebase_gate["metrics"]["comparison"]["evidence_jsonl_invalid_rows"][0]["errors"]
    )

    install_target = review_dir / "voicebank" / "oto.ini"
    install_target.parent.mkdir()
    install_target.write_text("old.wav=old,0,100,-300,50,25\n", encoding="utf-8")

    wav_for_stale_evidence = wav_dir / "va_vi_vu.wav"
    original_wav_bytes = wav_for_stale_evidence.read_bytes()
    wav_for_stale_evidence.write_bytes(original_wav_bytes + b"stale acoustic source")
    stale_wav_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    stale_wav_install_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "install-final",
            "--final",
            str(replay_final),
            "--target-oto",
            str(install_target),
            "--gate-report",
            str(gate_with_merge_report),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    wav_for_stale_evidence.write_bytes(original_wav_bytes)
    assert stale_wav_gate_proc.returncode == 2
    stale_wav_gate = json.loads(stale_wav_gate_proc.stdout)
    assert "comparison_evidence_jsonl_rows_valid" in stale_wav_gate["gate_report"]["failed_checks"]
    assert "comparison_evidence_wav_sha256_current_match" in stale_wav_gate["gate_report"]["failed_checks"]
    assert "comparison_evidence_jsonl_rows_invalid" in stale_wav_gate["warnings"]
    assert "comparison_evidence_wav_sha256_current_mismatch" in stale_wav_gate["warnings"]
    assert stale_wav_gate["metrics"]["comparison"]["evidence_jsonl_current_match"] is True
    assert stale_wav_gate["metrics"]["comparison"]["evidence_jsonl_rows_valid"] is False
    assert stale_wav_gate["metrics"]["comparison"]["evidence_wav_sha256_current_match"] is False
    assert stale_wav_gate["metrics"]["comparison"]["evidence_wav_sha256_mismatch_count"] == 1
    assert stale_wav_install_proc.returncode == 2
    stale_wav_install = json.loads(stale_wav_install_proc.stdout)
    assert stale_wav_install["reason"] == "gate_failed:apply_candidate"
    assert stale_wav_install["gate_checks"]["comparison_evidence_jsonl_rows_valid"] is False
    assert stale_wav_install["gate_checks"]["comparison_evidence_wav_sha256_current_match"] is False
    assert "comparison_evidence_wav_sha256_current_mismatch" in stale_wav_install["warnings"]
    assert stale_wav_install["gate_metrics"]["comparison"]["evidence_wav_sha256_mismatch_count"] == 1
    assert install_target.read_text(encoding="utf-8").startswith("old.wav=old")

    missing_prior_provenance_report = review_dir / "case.evidence_hsmm_replay.missing_prior_provenance.json"
    missing_prior_provenance_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    for record in list(missing_prior_provenance_payload.get("records", []) or []):
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        for key in (
            "event_prior_input_count",
            "event_prior_mapped_count",
            "event_prior_ignored_count",
            "event_prior_source_counts",
            "event_prior_label_counts",
            "candidate_prior_count",
            "runtime_prior_count",
        ):
            diagnostics.pop(key, None)
        for state in list(diagnostics.get("states", []) or []):
            if not isinstance(state, dict):
                continue
            for key in (
                "event_prior_mapped_count",
                "event_prior_source_counts",
                "event_prior_label_counts",
                "evidence_candidate_prior_count",
                "runtime_prior_count",
            ):
                state.pop(key, None)
    missing_prior_provenance_report.write_text(
        json.dumps(missing_prior_provenance_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing_prior_provenance_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(missing_prior_provenance_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_prior_provenance_gate_proc.returncode == 2
    missing_prior_provenance_gate = json.loads(missing_prior_provenance_gate_proc.stdout)
    assert "comparison_prior_provenance_available" in missing_prior_provenance_gate["gate_report"]["failed_checks"]
    assert "comparison_prior_provenance_missing" in missing_prior_provenance_gate["warnings"]
    assert missing_prior_provenance_gate["metrics"]["comparison"]["prior_provenance_required"] is True
    assert missing_prior_provenance_gate["metrics"]["comparison"]["prior_provenance_available"] is False
    missing_fields = missing_prior_provenance_gate["metrics"]["comparison"]["prior_provenance_missing_fields"]
    assert any(item["field"] == "event_prior_input_count" for item in missing_fields)

    tampered_row_report = review_dir / "case.evidence_hsmm_replay.tampered_row.json"
    tampered_row_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    tampered_row_payload["reference_comparison"]["row_param_delta"]["matched_rows"] -= 1
    tampered_row_report.write_text(json.dumps(tampered_row_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tampered_row_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(tampered_row_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert tampered_row_gate_proc.returncode == 2
    tampered_row_gate = json.loads(tampered_row_gate_proc.stdout)
    assert "comparison_row_param_delta_current_match" in tampered_row_gate["gate_report"]["failed_checks"]
    assert "comparison_row_param_delta_current_mismatch" in tampered_row_gate["warnings"]
    assert tampered_row_gate["metrics"]["comparison"]["row_param_delta_current_match"] is False
    assert tampered_row_gate["metrics"]["comparison"]["row_param_delta_mismatched_fields"][0]["field"] == "matched_rows"

    tampered_timeline_report = review_dir / "case.evidence_hsmm_replay.tampered_timeline.json"
    tampered_timeline_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    tampered_timeline_payload["reference_comparison"]["timeline_event_delta"]["matched_events"] += 1
    tampered_timeline_report.write_text(
        json.dumps(tampered_timeline_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tampered_timeline_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(tampered_timeline_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert tampered_timeline_gate_proc.returncode == 2
    tampered_timeline_gate = json.loads(tampered_timeline_gate_proc.stdout)
    assert "comparison_timeline_event_delta_current_match" in tampered_timeline_gate["gate_report"]["failed_checks"]
    assert "comparison_timeline_event_delta_current_mismatch" in tampered_timeline_gate["warnings"]
    assert tampered_timeline_gate["metrics"]["comparison"]["timeline_event_delta_current_match"] is False
    assert (
        tampered_timeline_gate["metrics"]["comparison"]["timeline_event_delta_mismatched_fields"][0]["field"]
        == "matched_events"
    )

    missing_row_source_report = review_dir / "case.evidence_hsmm_replay.missing_row_source.json"
    missing_row_source_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    row_delta_for_missing_source = missing_row_source_payload["reference_comparison"]["row_param_delta"]
    for key in ("baseline_oto_path", "baseline_oto_sha256", "hsmm_oto_path", "hsmm_oto_sha256"):
        row_delta_for_missing_source.pop(key, None)
    missing_row_source_payload.pop("assembled_oto_path", None)
    missing_row_source_payload.pop("assembled_oto_sha256", None)
    missing_row_source_report.write_text(
        json.dumps(missing_row_source_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing_row_source_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(missing_row_source_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_row_source_gate_proc.returncode == 2
    missing_row_source_gate = json.loads(missing_row_source_gate_proc.stdout)
    assert "comparison_row_param_delta_current_match" in missing_row_source_gate["gate_report"]["failed_checks"]
    assert "comparison_row_param_delta_recompute_source_missing" in missing_row_source_gate["warnings"]
    assert missing_row_source_gate["metrics"]["comparison"]["row_param_delta_recompute_source_required"] is True
    assert (
        missing_row_source_gate["metrics"]["comparison"]["row_param_delta_recompute_error"]
        == "missing_recompute_source_path"
    )

    missing_timeline_source_report = review_dir / "case.evidence_hsmm_replay.missing_timeline_source.json"
    missing_timeline_source_payload = json.loads(replay_report.read_text(encoding="utf-8"))
    timeline_delta_for_missing_source = missing_timeline_source_payload["reference_comparison"]["timeline_event_delta"]
    for key in (
        "baseline_timeline_path",
        "baseline_timeline_sha256",
        "hsmm_timeline_path",
        "hsmm_timeline_sha256",
    ):
        timeline_delta_for_missing_source.pop(key, None)
    missing_timeline_source_payload.pop("timeline_path", None)
    missing_timeline_source_payload.pop("timeline_sha256", None)
    missing_timeline_source_report.write_text(
        json.dumps(missing_timeline_source_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing_timeline_source_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(missing_timeline_source_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_timeline_source_gate_proc.returncode == 2
    missing_timeline_source_gate = json.loads(missing_timeline_source_gate_proc.stdout)
    assert (
        "comparison_timeline_event_delta_current_match"
        in missing_timeline_source_gate["gate_report"]["failed_checks"]
    )
    assert (
        "comparison_timeline_event_delta_recompute_source_missing"
        in missing_timeline_source_gate["warnings"]
    )
    assert (
        missing_timeline_source_gate["metrics"]["comparison"]["timeline_event_delta_recompute_source_required"]
        is True
    )
    assert (
        missing_timeline_source_gate["metrics"]["comparison"]["timeline_event_delta_recompute_error"]
        == "missing_recompute_source_path"
    )

    original_evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence_path.write_text(original_evidence_text + "\n", encoding="utf-8")
    stale_evidence_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_evidence_gate_proc.returncode == 2
    stale_evidence_gate = json.loads(stale_evidence_gate_proc.stdout)
    assert "comparison_evidence_jsonl_current_match" in stale_evidence_gate["gate_report"]["failed_checks"]
    assert "comparison_evidence_jsonl_current_mismatch" in stale_evidence_gate["warnings"]
    assert stale_evidence_gate["metrics"]["comparison"]["evidence_jsonl_sha256_current_match"] is False
    assert stale_evidence_gate["metrics"]["comparison"]["validation_jsonl_current_match"] is True
    evidence_path.write_text(original_evidence_text, encoding="utf-8")

    original_replay_timeline_text = replay_timeline.read_text(encoding="utf-8")
    replay_timeline.write_text(original_replay_timeline_text + "\n", encoding="utf-8")
    stale_timeline_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_timeline_gate_proc.returncode == 2
    stale_timeline_gate = json.loads(stale_timeline_gate_proc.stdout)
    assert "comparison_timeline_jsonl_current_match" in stale_timeline_gate["gate_report"]["failed_checks"]
    assert "comparison_timeline_jsonl_current_mismatch" in stale_timeline_gate["warnings"]
    assert stale_timeline_gate["metrics"]["comparison"]["timeline_jsonl_sha256_current_match"] is False
    assert stale_timeline_gate["metrics"]["comparison"]["validation_jsonl_current_match"] is True
    assert stale_timeline_gate["metrics"]["comparison"]["evidence_jsonl_current_match"] is True
    replay_timeline.write_text(original_replay_timeline_text, encoding="utf-8")

    original_replay_oto_text = replay_oto.read_text(encoding="utf-8")
    replay_oto.write_text(original_replay_oto_text + "\n", encoding="utf-8")
    stale_oto_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_oto_gate_proc.returncode == 2
    stale_oto_gate = json.loads(stale_oto_gate_proc.stdout)
    assert "comparison_oto_current_match" in stale_oto_gate["gate_report"]["failed_checks"]
    assert "comparison_oto_current_mismatch" in stale_oto_gate["warnings"]
    assert stale_oto_gate["metrics"]["comparison"]["oto_sha256_current_match"] is False
    assert stale_oto_gate["metrics"]["comparison"]["validation_jsonl_current_match"] is True
    assert stale_oto_gate["metrics"]["comparison"]["timeline_jsonl_current_match"] is True
    assert stale_oto_gate["metrics"]["comparison"]["evidence_jsonl_current_match"] is True
    replay_oto.write_text(original_replay_oto_text, encoding="utf-8")

    default_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert default_gate_proc.returncode == 2
    assert json.loads(default_gate_proc.stdout)["gate_report"]["default_on_candidate"] is False

    apply_gate_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-gate",
            "--validation-jsonl",
            str(replay_payload["split_output_paths"]["validation_jsonl"]),
            "--comparison-json",
            str(replay_report),
            "--merge-report",
            str(merge_report),
            "--gate-target",
            "apply",
            "--fail-on-gate-failure",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert apply_gate_proc.returncode == 0
    assert json.loads(apply_gate_proc.stdout)["gate_report"]["apply_candidate"] is True

    missing_assembled_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--split-out-dir",
            str(review_dir / "invalid_split"),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_assembled_proc.returncode == 2
    assert json.loads(missing_assembled_proc.stdout)["reason"] == "split_out_dir_requires_assembled_oto_out"
    missing_compare_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--reference-generated",
            str(payload["generated_oto_path"]),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_compare_proc.returncode == 2
    assert json.loads(missing_compare_proc.stdout)["reason"] == "reference_generated_requires_assembled_oto_out"

    missing_validation_reference_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--reference-validation-jsonl",
            str(payload["split_output_paths"]["validation_jsonl"]),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_validation_reference_proc.returncode == 2
    assert (
        json.loads(missing_validation_reference_proc.stdout)["reason"]
        == "reference_validation_jsonl_requires_reference_generated"
    )

    missing_replay_split_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--assembled-oto-out",
            str(review_dir / "invalid_reference.generated.ini"),
            "--reference-generated",
            str(payload["generated_oto_path"]),
            "--reference-validation-jsonl",
            str(payload["split_output_paths"]["validation_jsonl"]),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_replay_split_proc.returncode == 2
    assert json.loads(missing_replay_split_proc.stdout)["reason"] == "reference_validation_jsonl_requires_split_out_dir"

    missing_timeline_out_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--reference-timeline",
            str(payload["timeline_path"]),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert missing_timeline_out_proc.returncode == 2
    assert json.loads(missing_timeline_out_proc.stdout)["reason"] == "reference_timeline_requires_timeline_out"

    unavailable_reference_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay-evidence-hsmm",
            "--evidence-jsonl",
            str(evidence_path),
            "--assembled-oto-out",
            str(review_dir / "missing_reference.generated.ini"),
            "--reference-generated",
            str(review_dir / "does_not_exist.generated.ini"),
            "--format-type",
            "cvvc",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert unavailable_reference_proc.returncode == 2
    unavailable_reference_payload = json.loads(unavailable_reference_proc.stdout)
    assert unavailable_reference_payload["ok"] is False
    assert unavailable_reference_payload["reference_comparison_failures"] == [
        {"comparison": "row_param_delta", "reason": "missing_generated_oto_path"}
    ]
    generated = (review_dir / "case.generated.ini").read_text(encoding="utf-8")
    assert "va_vi_vu.wav=va," in generated
    assert "va_vi_vu.wav=a v," in generated
    assert "va_vi_vu.wav=i v," in generated
    assert "va_vi_vu.wav=vu," in generated


def test_mfa_free_oto_review_cli_session_status_checks_metadata_artifacts(tmp_path):
    generated = tmp_path / "case.generated.ini"
    generated.write_text("a.wav=a,0,100,-350,50,25\n", encoding="utf-8")
    timeline = tmp_path / "case.timeline.jsonl"
    evidence = tmp_path / "case.evidence.jsonl"
    row_plan = tmp_path / "case.row_plan.jsonl"
    timeline.write_text("{\"wav\":\"a.wav\"}\n", encoding="utf-8")
    evidence.write_text("{\"wav\":\"a.wav\"}\n", encoding="utf-8")
    row_plan.write_text("{\"wav\":\"a.wav\",\"alias\":\"a\"}\n", encoding="utf-8")

    split = split_oto_file(
        str(generated),
        str(tmp_path / "split"),
        name="case",
        wav_durations_ms={"a.wav": 400.0},
        session_metadata={
            "created_by": "test",
            "timeline_path": str(timeline),
            "evidence_path": str(evidence),
            "row_plan_path": str(row_plan),
            "row_plan_rows": 1,
            "source_timing_trusted": False,
        },
    )
    review_session_path = Path(split.output_paths["review_session"])
    review_session = json.loads(review_session_path.read_text(encoding="utf-8"))
    metadata_artifacts = review_session["lineage"]["metadata_artifacts"]
    assert metadata_artifacts["timeline"]["sha256"] == _sha256_file(timeline)
    assert metadata_artifacts["evidence"]["sha256"] == _sha256_file(evidence)
    assert metadata_artifacts["row_plan"]["sha256"] == _sha256_file(row_plan)

    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"
    current_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert current_proc.returncode == 0, current_proc.stderr
    current_status = json.loads(current_proc.stdout)
    assert current_status["checks"]["metadata_artifacts_current_match"] is True
    assert current_status["metadata_artifacts"]["timeline"]["sha256_match"] is True

    legacy_session = json.loads(review_session_path.read_text(encoding="utf-8"))
    legacy_session["lineage"].pop("metadata_artifacts", None)
    legacy_session_path = review_session_path.with_name("case.legacy.review_session.json")
    legacy_session_path.write_text(json.dumps(legacy_session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    legacy_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(legacy_session_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert legacy_proc.returncode == 2
    legacy_status = json.loads(legacy_proc.stdout)
    assert legacy_status["checks"]["metadata_artifacts_current_match"] is False
    assert legacy_status["metadata_artifacts"]["timeline"]["lineage_missing"] is True
    assert "metadata_artifact_lineage_missing:timeline" in legacy_status["warnings"]

    timeline.write_text(timeline.read_text(encoding="utf-8") + "{\"wav\":\"changed.wav\"}\n", encoding="utf-8")
    stale_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "session-status",
            "--review-session",
            str(review_session_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_proc.returncode == 2
    stale_status = json.loads(stale_proc.stdout)
    assert stale_status["checks"]["metadata_artifacts_current_match"] is False
    assert stale_status["metadata_artifacts"]["timeline"]["sha256_match"] is False
    assert "metadata_artifact_sha256_mismatch:timeline" in stale_status["warnings"]
    assert stale_status["next_action"] == "regenerate_or_repair_review_split"

    stale_prepare_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "prepare-edits",
            "--review-session",
            str(review_session_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert stale_prepare_proc.returncode == 2
    stale_prepare = json.loads(stale_prepare_proc.stdout)
    assert stale_prepare["reason"] == "review_session_not_current"
    assert stale_prepare["session_status_checks"]["metadata_artifacts_current_match"] is False
    assert "metadata_artifact_sha256_mismatch:timeline" in stale_prepare["session_status_warnings"]


def test_mfa_free_oto_review_cli_generate_can_use_hsmm_decoder(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "ka_ki.wav", duration_s=0.45)
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--wav-dir",
            str(wav_dir),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--format-type",
            "cvvc",
            "--encoder",
            "acoustic",
            "--use-hsmm-decoder",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["row_plan_rows"] == 3
    assert any("filename_hsmm_decoder" in warning for warning in payload["warnings"])


def test_mfa_free_oto_review_cli_can_use_single_vowel_hsmm_after_leading_gap(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "i.wav", duration_s=0.45)
    review_dir = tmp_path / "review"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--wav-dir",
            str(wav_dir),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--format-type",
            "cvvc",
            "--encoder",
            "acoustic",
            "--use-hsmm-decoder",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "filename_hsmm_decoder_used" in payload["warnings"]
    assert "filename_hsmm_decoder_rejected:event_sequence_mismatch" not in payload["warnings"]
    timeline_path = Path(payload["timeline_path"])
    record = json.loads(timeline_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["selected_event_source"] == "filename_hsmm"
    assert record["hsmm"]["ok"] is True
    assert [event["label"] for event in record["selected_events"]] == ["vv_boundary", "vowel_nucleus"]
    assert record["hsmm"]["meta"]["leading_gap_ms"] > 0.0
    assert record["hsmm"]["diagnostics"]["state_count"] == 1
    assert record["hsmm"]["diagnostics"]["event_prior_state_count"] == 1
    assert "selected_vs_best_local_margin" in record["hsmm"]["diagnostics"]["states"][0]


def test_mfa_free_oto_review_cli_compare_hsmm_writes_delta_report(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_tone_wav(wav_dir / "ka_ki.wav", duration_s=0.45)
    review_dir = tmp_path / "compare"
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "compare-hsmm",
            "--wav-dir",
            str(wav_dir),
            "--out-dir",
            str(review_dir),
            "--name",
            "case",
            "--format-type",
            "cvvc",
            "--encoder",
            "acoustic",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["baseline"]["use_hsmm_decoder"] is False
    assert payload["hsmm"]["use_hsmm_decoder"] is True
    assert payload["baseline"]["generated_oto_sha256"] == _sha256_file(Path(payload["baseline"]["generated_oto_path"]))
    assert payload["hsmm"]["generated_oto_sha256"] == _sha256_file(Path(payload["hsmm"]["generated_oto_path"]))
    assert Path(payload["baseline"]["timeline_path"]).is_file()
    assert Path(payload["hsmm"]["timeline_path"]).is_file()
    assert payload["baseline"]["timeline_sha256"] == _sha256_file(Path(payload["baseline"]["timeline_path"]))
    assert payload["hsmm"]["timeline_sha256"] == _sha256_file(Path(payload["hsmm"]["timeline_path"]))
    assert payload["comparison"]["baseline_rows"] == payload["comparison"]["hsmm_rows"]
    assert payload["comparison"]["hsmm_decoder_used"] is True
    assert payload["comparison"]["row_param_delta"]["baseline_oto_sha256"] == payload["baseline"]["generated_oto_sha256"]
    assert payload["comparison"]["row_param_delta"]["hsmm_oto_sha256"] == payload["hsmm"]["generated_oto_sha256"]
    assert payload["comparison"]["timeline_event_delta"]["available"] is True
    assert payload["comparison"]["timeline_event_delta"]["baseline_timeline_sha256"] == payload["baseline"]["timeline_sha256"]
    assert payload["comparison"]["timeline_event_delta"]["hsmm_timeline_sha256"] == payload["hsmm"]["timeline_sha256"]
    assert payload["comparison"]["hsmm_diagnostics"]["available"] is True
    assert payload["comparison"]["hsmm_diagnostics"]["total_state_count"] > 0
    assert payload["comparison"]["hsmm_diagnostics"]["state_rows_total"] > 0
    if payload["comparison"]["hsmm_diagnostics"]["event_prior_state_count"] > 0:
        assert payload["comparison"]["hsmm_diagnostics"]["event_prior_input_count"] > 0
        assert payload["comparison"]["hsmm_diagnostics"]["event_prior_mapped_count"] > 0
        assert payload["comparison"]["hsmm_diagnostics"]["event_prior_source_counts"]
        assert payload["comparison"]["hsmm_diagnostics"]["event_prior_label_counts"]
    report_path = Path(payload["comparison_report_path"])
    assert report_path.is_file()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["command"] == "compare-hsmm"
    assert (review_dir / "case.baseline" / "case.generated.ini").is_file()
    assert (review_dir / "case.hsmm" / "case.generated.ini").is_file()


def test_compare_hsmm_reports_row_param_deltas_from_generated_oto(tmp_path):
    baseline_oto = tmp_path / "baseline.ini"
    hsmm_oto = tmp_path / "hsmm.ini"
    baseline_oto.write_text(
        "\n".join(
            [
                "a.wav=a,0,100,-300,50,25",
                "a.wav=a k,150,120,-180,80,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hsmm_oto.write_text(
        "\n".join(
            [
                "a.wav=a,10,90,-310,55,20",
                "a.wav=a k,150,120,-180,80,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    module = runpy.run_path(str(REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"))
    comparison = module["_compare_generate_payloads"](
        {
            "ok": True,
            "processed": 2,
            "split_counts": {"total": 2, "clean": 2},
            "generated_oto_path": str(baseline_oto),
        },
        {
            "ok": True,
            "processed": 2,
            "split_counts": {"total": 2, "clean": 2},
            "generated_oto_path": str(hsmm_oto),
        },
        row_delta_limit=10,
    )

    delta = comparison["row_param_delta"]
    assert delta["available"] is True
    assert delta["baseline_oto_sha256"] == _sha256_file(baseline_oto)
    assert delta["hsmm_oto_sha256"] == _sha256_file(hsmm_oto)
    assert delta["matched_rows"] == 2
    assert delta["changed_rows"] == 1
    assert delta["mean_abs_delta_ms"]["offset"] == 5.0
    assert delta["max_abs_delta_ms"]["cutoff"]["value_ms"] == 10.0
    assert delta["largest_rows"][0]["alias"] == "a"
    assert delta["largest_rows"][0]["deltas_ms"] == {
        "offset": 10.0,
        "consonant": -10.0,
        "cutoff": -10.0,
        "preutterance": 5.0,
        "overlap": -5.0,
    }


def test_compare_hsmm_reports_role_delta_summary_from_validation_jsonl(tmp_path):
    baseline_oto = tmp_path / "baseline.ini"
    hsmm_oto = tmp_path / "hsmm.ini"
    baseline_oto.write_text(
        "\n".join(
            [
                "_a-a-i-a.wav=a a,100,160,-600,120,85",
                "_a-a-i-a.wav=ka,900,190,-420,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hsmm_oto.write_text(
        "\n".join(
            [
                "_a-a-i-a.wav=a a,90,420,-600,300,100",
                "_a-a-i-a.wav=ka,900,190,-420,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = split_oto_file(
        str(hsmm_oto),
        str(tmp_path / "split"),
        name="hsmm",
        row_diagnostics={
            0: {"row_plan": {"role_family": "vv"}},
            1: {"row_plan": {"role_family": "cv"}},
        },
    )
    module = runpy.run_path(str(REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"))

    delta = module["_compare_generated_oto_params"](
        str(baseline_oto),
        str(hsmm_oto),
        hsmm_validation_path=split.output_paths["validation_jsonl"],
        row_delta_limit=10,
    )

    assert delta["role_delta_summary"]["vv"]["matched_rows"] == 1
    assert delta["role_delta_summary"]["vv"]["mean_signed_delta_ms"]["preutterance"] == 180.0
    assert delta["role_delta_summary"]["vv"]["mean_signed_delta_ms"]["overlap"] == 15.0
    assert delta["role_delta_summary"]["vv"]["absolute_mean_signed_delta_ms"]["pre_abs"] == 170.0
    assert delta["role_delta_summary"]["cv"]["matched_rows"] == 1
    assert delta["role_delta_summary"]["cv"]["changed_rows"] == 0
    assert delta["absolute_mean_abs_delta_ms"]["pre_abs"] == 85.0
    assert delta["largest_rows"][0]["absolute_deltas_ms"]["pre_abs"] == 170.0


def test_evaluate_lightgbm_postprocess_reports_reference_improvement(tmp_path):
    reference = tmp_path / "reference.ini"
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    out = tmp_path / "lgbm_eval.json"
    reference.write_text(
        "\n".join(
            [
                "a.wav=a,10,100,-300,50,25",
                "a.wav=a k,150,120,-180,80,35",
                "a.wav=i,260,90,-160,70,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pre.write_text(
        "\n".join(
            [
                "a.wav=a,30,100,-300,50,25",
                "a.wav=a k,170,120,-180,80,35",
                "a.wav=i,260,90,-160,70,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "a.wav=a,12,100,-300,50,25",
                "a.wav=a k,154,120,-180,80,35",
                "a.wav=i,260,90,-160,70,35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script = REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-lightgbm-postprocess",
            "--reference-oto",
            str(reference),
            "--pre-lightgbm-oto",
            str(pre),
            "--post-lightgbm-oto",
            str(post),
            "--cutoff-end-mode",
            "raw",
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["improvement"]["by_param"]["offset"]["pre_mae_ms"] == 13.3
    assert payload["improvement"]["by_param"]["offset"]["post_mae_ms"] == 2.0
    assert payload["pre_vs_reference"]["by_alias_family"]["vc"]["by_param"]["offset"]["MAE_ms"] == 20.0
    assert payload["improvement"]["by_alias_family"]["vc"]["by_param"]["offset"] == {
        "pre_mae_ms": 20.0,
        "post_mae_ms": 4.0,
        "delta_mae_ms": -16.0,
    }
    assert payload["recommended_use"]["safe_candidate"] is True
    assert payload["recommended_use"]["matched_rows"] == 3
    assert payload["recommended_use"]["changed_ratio"] == 0.6667
    assert payload["pre_vs_post"]["changed_rows"] == 2
    assert out.is_file()


def test_compare_hsmm_reports_timeline_event_deltas(tmp_path):
    baseline_timeline = tmp_path / "baseline.timeline.jsonl"
    hsmm_timeline = tmp_path / "hsmm.timeline.jsonl"
    baseline_timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "runtime_prediction",
                "selected_events": [
                    {"label": "cv_boundary", "selected_time_ms": 100.0, "score": 0.9, "expected_phone": "a"},
                    {"label": "vowel_nucleus", "selected_time_ms": 150.0, "score": 0.8, "expected_phone": "a"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    hsmm_timeline.write_text(
        json.dumps(
            {
                "wav": "a.wav",
                "selected_event_source": "filename_hsmm",
                "selected_events": [
                    {"label": "cv_boundary", "selected_time_ms": 120.0, "score": 0.7, "expected_phone": "a"},
                    {"label": "vowel_nucleus", "selected_time_ms": 180.0, "score": 0.6, "expected_phone": "a"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    module = runpy.run_path(str(REPO_ROOT / "scripts/dev/mfa_free_oto_review.py"))
    delta = module["_compare_timeline_events"](
        str(baseline_timeline),
        str(hsmm_timeline),
        row_delta_limit=10,
    )

    assert delta["available"] is True
    assert delta["matched_events"] == 2
    assert delta["changed_events"] == 2
    assert delta["largest_events"][0]["delta_ms"] == 30.0
    assert delta["largest_events"][0]["hsmm_event_source"] == "filename_hsmm"


def _write_tone_wav(path: Path, *, duration_s: float = 0.3, sample_rate: int = 16000) -> None:
    frame_count = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(frame_count):
            value = int(12000 * math.sin(2.0 * math.pi * 220.0 * idx / sample_rate))
            frames.extend(struct.pack("<h", value))
        handle.writeframes(bytes(frames))
