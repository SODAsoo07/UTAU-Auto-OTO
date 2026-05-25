from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.generation.common.final_merge import merge_split_oto
from core.generation.common.oto_file_utils import apply_alias_suffix, parse_oto_line, read_text_with_fallback
from core.generation.common.validation_diagnostics import (
    adapter_warning_validation_metrics,
    runtime_metadata_validation_metrics,
)
from core.generation.common.validation_splitter import (
    load_validation_records,
    load_validation_records_with_errors,
    split_oto_file,
)
from core.mfa_free_oto.evidence_pack import (
    candidate_priors_from_evidence_pack,
    expected_phones_from_evidence_pack,
    filename_slots_from_evidence_pack,
    frame_posterior_from_evidence_pack,
    read_evidence_jsonl,
    runtime_events_from_evidence_pack,
    summarize_event_candidate_reliability,
    validate_acoustic_evidence_pack,
)
from core.mfa_free_oto.hsmm_adapter import decode_filename_slots_with_hsmm
from core.mfa_free_oto.oto_adapter import OtoAdapterConfig, adapt_template_row, assign_template_row_anchors
from core.mfa_free_oto.row_plan import build_filename_row_plan, build_filename_template_rows, write_row_plan_jsonl
from core.mfa_free_oto.workflow import generate_no_mfa_oto_with_model_context

_OTO_PARAM_FIELDS = (
    ("offset", "offset"),
    ("consonant", "cons"),
    ("cutoff", "cutoff"),
    ("preutterance", "pre"),
    ("overlap", "ovl"),
)

_QUALITY_EVIDENCE_TYPES = {"none", "manual-reference", "listening", "manual-reference-and-listening"}


def _load_durations_json(path: str) -> Mapping[str, float]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit("--durations-json must be a JSON object mapping wav name to duration_ms.")
    durations: dict[str, float] = {}
    for key, value in raw.items():
        try:
            durations[str(key)] = float(value)
        except Exception as exc:
            raise SystemExit(f"Invalid duration for {key!r}: {value!r}") from exc
    return durations


def _print_json(payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text_lines(path: str | Path, lines: list[str] | tuple[str, ...]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(line).rstrip("\n") for line in lines if str(line).strip())
    out.write_text((text + "\n") if text else "", encoding="utf-8")


def _row_diagnostics_from_timeline_path(path: str) -> dict[int, dict[str, object]]:
    if not path:
        return {}
    timeline_path = Path(path)
    if not timeline_path.is_file():
        return {}
    rows: list[Mapping[str, object]] = []
    with timeline_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            data = json.loads(raw)
            if isinstance(data, Mapping):
                rows.append(data)
    return _row_diagnostics_from_timeline_records(rows)


def _row_diagnostics_from_timeline_records(records: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    line_index = 0
    for record in records:
        for _row in list(record.get("adapted_rows", []) or []):
            metrics = _hsmm_validation_metrics_for_row(record, _row)
            if metrics:
                out[line_index] = dict(metrics)
            line_index += 1
    return out


def _hsmm_validation_metrics_for_row(record: Mapping[str, object], row: object) -> dict[str, object]:
    evidence_metrics = _evidence_validation_metrics_from_timeline_record(record)
    runtime_metrics = _runtime_validation_metrics_from_timeline_record(record)
    row_data = dict(row or {}) if isinstance(row, Mapping) else {}
    adapter_metrics = adapter_warning_validation_metrics(row_data)
    row_plan = row_data.get("row_plan")
    row_plan_payload = dict(row_plan) if isinstance(row_plan, Mapping) else None
    aggregate = {
        **runtime_metrics,
        **evidence_metrics,
        **_hsmm_validation_metrics_from_timeline_record(record),
        **adapter_metrics,
    }
    if row_plan_payload is not None:
        aggregate["row_plan"] = row_plan_payload
    if not aggregate:
        return {}
    if not isinstance(row_plan, Mapping):
        return aggregate
    left_slot = _int_or_none(row_plan.get("left_slot_index"))
    right_slot = _int_or_none(row_plan.get("right_slot_index"))
    if left_slot is None or right_slot is None:
        return aggregate
    lo = min(left_slot, right_slot)
    hi = max(left_slot, right_slot)
    hsmm = record.get("hsmm")
    diagnostics = dict(hsmm.get("diagnostics", {}) or {}) if isinstance(hsmm, Mapping) else {}
    states = [
        dict(state or {})
        for state in list(diagnostics.get("states", []) or [])
        if _state_slot_index(str(dict(state or {}).get("state_id", ""))) in range(lo, hi + 1)
    ]
    if not states:
        return aggregate
    best_margins = [_float_or_none(state.get("selected_vs_best_local_margin")) for state in states]
    second_margins = [_float_or_none(state.get("selected_vs_second_local_margin")) for state in states]
    duration_z_values = [_float_or_none(state.get("duration_z_abs")) for state in states]
    event_prior_count = sum(1 for state in states if bool(state.get("has_event_prior")))
    candidate_prior_count = sum(int(state.get("evidence_candidate_prior_count", 0) or 0) for state in states)
    runtime_prior_count = sum(int(state.get("runtime_prior_count", 0) or 0) for state in states)
    mapped_prior_count = sum(int(state.get("event_prior_mapped_count", 0) or 0) for state in states)
    return {
        **runtime_metrics,
        **evidence_metrics,
        **adapter_metrics,
        "row_plan": row_plan_payload,
        "hsmm_decoder_used": aggregate.get("hsmm_decoder_used", 0.0),
        "hsmm_state_count": float(len(states)),
        "hsmm_event_prior_state_count": float(event_prior_count),
        "hsmm_event_prior_mapped_count": float(mapped_prior_count),
        "hsmm_candidate_prior_count": float(candidate_prior_count),
        "hsmm_runtime_prior_count": float(runtime_prior_count),
        "hsmm_min_selected_vs_best_local_margin": min(value for value in best_margins if value is not None) if any(value is not None for value in best_margins) else 0.0,
        "hsmm_min_selected_vs_second_local_margin": min(value for value in second_margins if value is not None) if any(value is not None for value in second_margins) else 0.0,
        "hsmm_max_duration_z_abs": max(value for value in duration_z_values if value is not None) if any(value is not None for value in duration_z_values) else 0.0,
    }


def _runtime_validation_metrics_from_timeline_record(record: Mapping[str, object]) -> dict[str, object]:
    summary = record.get("posterior_summary")
    metadata = summary.get("metadata") if isinstance(summary, Mapping) else {}
    return runtime_metadata_validation_metrics(metadata if isinstance(metadata, Mapping) else {})


def _evidence_validation_metrics_from_timeline_record(record: Mapping[str, object]) -> dict[str, object]:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        return {}
    summary = evidence.get("event_candidate_summary")
    if not isinstance(summary, Mapping):
        summary = summarize_event_candidate_reliability(evidence)
    flag_counts = summary.get("event_candidate_flag_counts", {})
    type_flag_counts = summary.get("event_candidate_type_flag_counts", {})
    if not isinstance(flag_counts, Mapping):
        flag_counts = {}
    if not isinstance(type_flag_counts, Mapping):
        type_flag_counts = {}
    return {
        "evidence_event_candidate_count": _float_or_none(summary.get("event_candidate_count")) or 0.0,
        "evidence_event_candidate_flagged_count": _float_or_none(summary.get("event_candidate_flagged_count")) or 0.0,
        "evidence_event_candidate_min_margin": _float_or_none(summary.get("event_candidate_min_margin")) or 0.0,
        "evidence_event_candidate_min_feature_agreement": _float_or_none(
            summary.get("event_candidate_min_feature_agreement")
        )
        or 0.0,
        "evidence_event_candidate_max_ood": _float_or_none(summary.get("event_candidate_max_ood")) or 0.0,
        "evidence_low_margin_count": _float_or_none(flag_counts.get("low_margin")) or 0.0,
        "evidence_low_feature_agreement_count": _float_or_none(flag_counts.get("low_feature_agreement")) or 0.0,
        "evidence_ambiguous_vv_boundary_count": _float_or_none(flag_counts.get("ambiguous_vv_boundary")) or 0.0,
        "evidence_ambiguous_sonorant_constriction_count": _float_or_none(
            flag_counts.get("ambiguous_sonorant_constriction")
        )
        or 0.0,
        "evidence_vv_boundary_low_margin_count": _float_or_none(type_flag_counts.get("vv_boundary_low_margin"))
        or 0.0,
        "evidence_vv_boundary_low_feature_agreement_count": _float_or_none(
            type_flag_counts.get("vv_boundary_low_feature_agreement")
        )
        or 0.0,
        "evidence_vv_boundary_ambiguous_count": _float_or_none(
            type_flag_counts.get("vv_boundary_ambiguous_vv_boundary")
        )
        or 0.0,
        "evidence_sonorant_constriction_low_margin_count": _float_or_none(
            type_flag_counts.get("sonorant_constriction_low_margin")
        )
        or 0.0,
        "evidence_sonorant_constriction_low_feature_agreement_count": _float_or_none(
            type_flag_counts.get("sonorant_constriction_low_feature_agreement")
        )
        or 0.0,
        "evidence_sonorant_constriction_ambiguous_count": _float_or_none(
            type_flag_counts.get("sonorant_constriction_ambiguous_sonorant_constriction")
        )
        or 0.0,
    }


def _hsmm_validation_metrics_from_timeline_record(record: Mapping[str, object]) -> dict[str, object]:
    hsmm = record.get("hsmm")
    if not isinstance(hsmm, Mapping):
        return {}
    diagnostics = hsmm.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    summary = diagnostics.get("event_prior_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "hsmm_decoder_used": 1.0 if str(record.get("selected_event_source", "")) == "filename_hsmm" else 0.0,
        "hsmm_state_count": _float_or_none(diagnostics.get("state_count")) or 0.0,
        "hsmm_event_prior_state_count": _float_or_none(diagnostics.get("event_prior_state_count")) or 0.0,
        "hsmm_event_prior_mapped_count": _float_or_none(diagnostics.get("event_prior_mapped_count")) or 0.0,
        "hsmm_event_prior_ignored_count": _float_or_none(diagnostics.get("event_prior_ignored_count")) or 0.0,
        "hsmm_candidate_prior_count": _float_or_none(diagnostics.get("candidate_prior_count")) or 0.0,
        "hsmm_runtime_prior_count": _float_or_none(diagnostics.get("runtime_prior_count")) or 0.0,
        "hsmm_event_prior_summary_mapped_count": _float_or_none(summary.get("mapped_count")) or 0.0,
        "hsmm_event_prior_summary_ignored_count": _float_or_none(summary.get("ignored_count")) or 0.0,
        "hsmm_min_selected_vs_best_local_margin": _float_or_none(diagnostics.get("min_selected_vs_best_local_margin")) or 0.0,
        "hsmm_min_selected_vs_second_local_margin": _float_or_none(diagnostics.get("min_selected_vs_second_local_margin")) or 0.0,
        "hsmm_max_duration_z_abs": _float_or_none(diagnostics.get("max_duration_z_abs")) or 0.0,
    }


def _state_slot_index(state_id: str) -> int | None:
    text = str(state_id or "")
    if not text.startswith("slot"):
        return None
    tail = text[4:]
    digits = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _cmd_split(args: argparse.Namespace) -> int:
    timeline_jsonl_sha256 = ""
    if args.timeline_jsonl and Path(str(args.timeline_jsonl)).is_file():
        timeline_jsonl_sha256 = _sha256_file(Path(str(args.timeline_jsonl)))
    result = split_oto_file(
        args.generated,
        args.out_dir,
        name=args.name,
        language=args.language,
        wav_root=args.wav_root,
        wav_durations_ms=_load_durations_json(args.durations_json),
        row_diagnostics=_row_diagnostics_from_timeline_path(args.timeline_jsonl),
        session_metadata={
            "created_by": "mfa_free_oto_review.split",
            "timeline_jsonl": str(args.timeline_jsonl or ""),
            "timeline_jsonl_sha256": timeline_jsonl_sha256,
            "wav_root": str(args.wav_root or ""),
            "source_timing_trusted": False,
        },
    )
    _print_json(
        {
            "ok": True,
            "command": "split",
            "counts": dict(result.counts),
            "output_paths": dict(result.output_paths),
            "summary_path": result.output_paths.get("summary", ""),
        }
    )
    return 0


def _run_generate_review(
    args: argparse.Namespace,
    *,
    out_dir: str | Path,
    stem: str,
    use_hsmm_decoder: bool,
) -> tuple[int, dict[str, object]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(stem or "mfa_free").strip() or "mfa_free"
    generated_path = out_dir / f"{stem}.generated.ini"
    messages: list[str] = []
    report = generate_no_mfa_oto_with_model_context(
        wav_dir=args.wav_dir,
        out_path=str(generated_path),
        source_oto_path=args.template_oto,
        alias_suffix=args.alias_suffix,
        checkpoint_path=args.checkpoint,
        language=args.language,
        format_type=args.format_type,
        alias_type=args.alias_type,
        encoder=args.encoder,
        device=args.device or None,
        use_slot_viterbi=not bool(args.disable_slot_viterbi),
        use_hsmm_decoder=bool(use_hsmm_decoder),
        callback=messages.append,
    )
    if report.errors or not generated_path.is_file():
        return 2, {
            "ok": False,
            "generated_oto_path": str(generated_path),
            "processed": int(report.processed),
            "total": int(report.total),
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "messages": messages,
            "use_hsmm_decoder": bool(use_hsmm_decoder),
        }

    timeline_rows = list(report.timeline_debug or ())
    evidence_rows = list(report.evidence_debug or ())
    row_plan_records = []
    row_plan_path = ""
    row_plan_sha256 = ""
    if not str(args.template_oto or "").strip():
        for wav_path in sorted(Path(args.wav_dir).glob("*.wav")):
            row_plan_records.extend(
                build_filename_row_plan(
                    wav_path.name,
                    language=args.language,
                    format_type=args.format_type,
                )
            )
        if row_plan_records:
            row_plan_path = str(out_dir / f"{stem}.row_plan.jsonl")
            write_row_plan_jsonl(row_plan_path, row_plan_records)
            row_plan_sha256 = _sha256_file(Path(row_plan_path))
    timeline_path = ""
    timeline_sha256 = ""
    timeline_wavs = 0
    if timeline_rows:
        timeline_path = str(out_dir / f"{stem}.timeline.jsonl")
        _write_jsonl(timeline_path, timeline_rows)
        timeline_sha256 = _sha256_file(Path(timeline_path))
        timeline_wavs = len(timeline_rows)
    evidence_path = ""
    evidence_sha256 = ""
    evidence_wavs = 0
    if evidence_rows:
        evidence_path = str(out_dir / f"{stem}.evidence.jsonl")
        _write_jsonl(evidence_path, evidence_rows)
        evidence_sha256 = _sha256_file(Path(evidence_path))
        evidence_wavs = len(evidence_rows)
    split = split_oto_file(
        str(generated_path),
        str(out_dir),
        name=stem,
        language=args.language,
        wav_root=args.wav_dir,
        row_diagnostics=_row_diagnostics_from_timeline_records(timeline_rows),
        session_metadata={
            "created_by": "mfa_free_oto_review.generate",
            "wav_dir": str(args.wav_dir or ""),
            "template_oto": str(args.template_oto or ""),
            "source_timing_trusted": False,
            "checkpoint": str(args.checkpoint or ""),
            "language": args.language,
            "format_type": args.format_type,
            "alias_type": args.alias_type,
            "encoder": args.encoder,
            "use_hsmm_decoder": bool(use_hsmm_decoder),
            "row_plan_path": row_plan_path,
            "row_plan_sha256": row_plan_sha256,
            "row_plan_rows": len(row_plan_records),
            "timeline_path": timeline_path,
            "timeline_sha256": timeline_sha256,
            "timeline_wavs": timeline_wavs,
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha256,
            "evidence_wavs": evidence_wavs,
            "preferred_encoding_source": str(generated_path),
        },
    )
    return 0, {
        "ok": True,
        "generated_oto_path": str(generated_path),
        "generated_oto_sha256": _sha256_file(generated_path),
        "processed": int(report.processed),
        "total": int(report.total),
        "confidence": float(report.confidence),
        "guard_failed": bool(report.guard_failed),
        "warnings": list(report.warnings),
        "messages": messages,
        "metrics": dict(report.metrics or {}),
        "row_plan_path": row_plan_path,
        "row_plan_sha256": row_plan_sha256,
        "row_plan_rows": len(row_plan_records),
        "timeline_path": timeline_path,
        "timeline_sha256": timeline_sha256,
        "timeline_wavs": timeline_wavs,
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "evidence_wavs": evidence_wavs,
        "split_counts": dict(split.counts),
        "split_output_paths": dict(split.output_paths),
        "use_hsmm_decoder": bool(use_hsmm_decoder),
    }


def _cmd_generate(args: argparse.Namespace) -> int:
    stem = str(args.name or "mfa_free").strip() or "mfa_free"
    code, payload = _run_generate_review(
        args,
        out_dir=args.out_dir,
        stem=stem,
        use_hsmm_decoder=bool(args.use_hsmm_decoder),
    )
    payload = {"command": "generate", **payload}
    _print_json(payload)
    return code


def _cmd_compare_hsmm(args: argparse.Namespace) -> int:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = str(args.name or "mfa_free").strip() or "mfa_free"
    baseline_code, baseline = _run_generate_review(
        args,
        out_dir=root / f"{stem}.baseline",
        stem=stem,
        use_hsmm_decoder=False,
    )
    hsmm_code, hsmm = _run_generate_review(
        args,
        out_dir=root / f"{stem}.hsmm",
        stem=stem,
        use_hsmm_decoder=True,
    )
    comparison = _compare_generate_payloads(
        baseline,
        hsmm,
        row_delta_limit=max(0, int(args.row_delta_limit)),
    )
    payload = {
        "ok": bool(baseline.get("ok")) and bool(hsmm.get("ok")),
        "command": "compare-hsmm",
        "baseline": baseline,
        "hsmm": hsmm,
        "comparison": comparison,
    }
    report_path = root / f"{stem}.hsmm_comparison.json"
    _write_json(report_path, payload)
    payload["comparison_report_path"] = str(report_path)
    _print_json(payload)
    return 0 if payload["ok"] and baseline_code == 0 and hsmm_code == 0 else 2


def _cmd_replay_evidence_hsmm(args: argparse.Namespace) -> int:
    evidence_file = Path(str(args.evidence_jsonl or ""))
    evidence_sha256 = _sha256_file(evidence_file).strip().lower() if evidence_file.is_file() else ""
    if args.split_out_dir and not args.assembled_oto_out:
        _print_json(
            {
                "ok": False,
                "command": "replay-evidence-hsmm",
                "reason": "split_out_dir_requires_assembled_oto_out",
            }
        )
        return 2
    if args.reference_generated and not args.assembled_oto_out:
        _print_json(
            {
                "ok": False,
                "command": "replay-evidence-hsmm",
                "reason": "reference_generated_requires_assembled_oto_out",
            }
        )
        return 2
    if args.reference_validation_jsonl and not args.reference_generated:
        _print_json(
            {
                "ok": False,
                "command": "replay-evidence-hsmm",
                "reason": "reference_validation_jsonl_requires_reference_generated",
            }
        )
        return 2
    if args.reference_validation_jsonl and not args.split_out_dir:
        _print_json(
            {
                "ok": False,
                "command": "replay-evidence-hsmm",
                "reason": "reference_validation_jsonl_requires_split_out_dir",
            }
        )
        return 2
    if args.reference_timeline and not args.timeline_out:
        _print_json(
            {
                "ok": False,
                "command": "replay-evidence-hsmm",
                "reason": "reference_timeline_requires_timeline_out",
            }
        )
        return 2
    try:
        rows = read_evidence_jsonl(args.evidence_jsonl)
    except Exception as exc:
        report = {
            "ok": False,
            "command": "replay-evidence-hsmm",
            "reason": "invalid_evidence_jsonl",
            "evidence_jsonl": str(args.evidence_jsonl),
            "evidence_jsonl_sha256": evidence_sha256,
            "error": str(exc),
        }
        if args.out:
            _write_json(args.out, report)
            report["output_path"] = str(args.out)
        _print_json(report)
        return 2
    invalid_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        validation_errors = validate_acoustic_evidence_pack(row)
        if validation_errors:
            invalid_rows.append(
                {
                    "row_index": int(idx),
                    "wav": str(row.get("wav", "") or row.get("wav_path", "") or f"row_{idx}"),
                    "errors": list(validation_errors),
                }
            )
    if invalid_rows:
        report = {
            "ok": False,
            "command": "replay-evidence-hsmm",
            "reason": "invalid_evidence_schema",
            "evidence_jsonl": str(args.evidence_jsonl),
            "evidence_jsonl_sha256": evidence_sha256,
            "total_records": len(rows),
            "invalid_evidence_count": len(invalid_rows),
            "invalid_evidence_rows": invalid_rows,
        }
        if args.out:
            _write_json(args.out, report)
            report["output_path"] = str(args.out)
        _print_json(report)
        return 2
    decoded_ok = 0
    skipped = 0
    hard_errors = 0
    records: list[dict[str, object]] = []
    timeline_records: list[dict[str, object]] = []
    assembled_lines: list[str] = []
    for idx, row in enumerate(rows):
        wav = str(row.get("wav", "") or row.get("wav_path", "") or f"row_{idx}")
        try:
            posterior = frame_posterior_from_evidence_pack(row)
            slots = filename_slots_from_evidence_pack(
                row,
                language=args.language,
                format_type=args.format_type,
            )
            expected_phones = expected_phones_from_evidence_pack(row)
            runtime_events = runtime_events_from_evidence_pack(row)
            candidate_priors = candidate_priors_from_evidence_pack(row, slots)
            if not slots:
                skipped += 1
                timeline_records.append(
                    _replay_timeline_record(
                        wav=wav,
                        evidence=row,
                        expected_phones=expected_phones,
                        slots=(),
                        runtime_events=runtime_events,
                        candidate_priors=candidate_priors,
                        selected_events=(),
                        diagnostics={},
                        meta={},
                        ok=False,
                        skipped=True,
                        reason="no_filename_slots",
                        adapted_rows=(),
                    )
                )
                records.append(
                    {
                        "wav": wav,
                        "ok": False,
                        "skipped": True,
                        "reason": "no_filename_slots",
                        "expected_phone_count": len(expected_phones),
                    }
                )
                continue
            decoded = decode_filename_slots_with_hsmm(
                posterior,
                slots,
                beam_width_per_state=int(args.beam_width),
                timeout_ms=int(args.timeout_ms),
                event_priors=(*runtime_events, *candidate_priors),
            )
            if decoded.ok:
                decoded_ok += 1
            assembled_rows = (
                _assemble_replay_oto_rows(
                    wav=wav,
                    evidence=row,
                    posterior=posterior,
                    selected_events=decoded.events,
                    expected_phones=expected_phones,
                    language=args.language,
                    format_type=args.format_type,
                    alias_type=args.alias_type,
                )
                if decoded.ok and args.assembled_oto_out
                else []
            )
            for item in assembled_rows:
                assembled_lines.append(apply_alias_suffix(str(item.get("raw_line", "")), args.alias_suffix))
            timeline_records.append(
                _replay_timeline_record(
                    wav=wav,
                    evidence=row,
                    expected_phones=expected_phones,
                    slots=slots,
                    runtime_events=runtime_events,
                    candidate_priors=candidate_priors,
                    selected_events=decoded.events,
                    diagnostics=decoded.diagnostics,
                    meta=dict(decoded.result.meta or {}),
                    ok=bool(decoded.ok),
                    skipped=False,
                    reason=str(decoded.result.reason),
                    adapted_rows=assembled_rows,
                )
            )
            records.append(
                {
                    "wav": wav,
                    "ok": bool(decoded.ok),
                    "skipped": False,
                    "reason": str(decoded.result.reason),
                    "slot_count": len(slots),
                    "expected_phone_count": len(expected_phones),
                    "runtime_event_count": len(runtime_events),
                    "candidate_prior_count": len(candidate_priors),
                    "state_count": len(decoded.states),
                    "decoded_state_count": len(decoded.result.states),
                    "event_count": len(decoded.events),
                    "assembled_row_count": len(assembled_rows),
                    "events": [_event_summary(event) for event in decoded.events],
                    "diagnostics": decoded.diagnostics,
                    "meta": dict(decoded.result.meta or {}),
                }
            )
        except Exception as exc:
            hard_errors += 1
            timeline_records.append(
                _replay_timeline_record(
                    wav=wav,
                    evidence=row,
                    expected_phones=(),
                    slots=(),
                    runtime_events=(),
                    selected_events=(),
                    diagnostics={},
                    meta={},
                    ok=False,
                    skipped=False,
                    reason=f"replay_failed:{exc}",
                    adapted_rows=(),
                )
            )
            records.append(
                {
                    "wav": wav,
                    "ok": False,
                    "skipped": False,
                    "reason": f"replay_failed:{exc}",
                }
            )
    report = {
        "ok": bool(rows) and hard_errors == 0 and decoded_ok > 0,
        "command": "replay-evidence-hsmm",
        "evidence_jsonl": str(args.evidence_jsonl),
        "evidence_jsonl_sha256": evidence_sha256,
        "total_records": len(rows),
        "decoded_ok": decoded_ok,
        "skipped": skipped,
        "hard_errors": hard_errors,
        "records": records,
    }
    timeline_out_sha256 = ""
    if args.timeline_out:
        _write_jsonl(args.timeline_out, timeline_records)
        timeline_out_sha256 = _sha256_file(Path(args.timeline_out))
        report["timeline_path"] = str(args.timeline_out)
        report["timeline_sha256"] = timeline_out_sha256
        report["timeline_wavs"] = len(timeline_records)
    if args.assembled_oto_out:
        _write_text_lines(args.assembled_oto_out, assembled_lines)
        report["assembled_oto_path"] = str(args.assembled_oto_out)
        report["assembled_oto_sha256"] = _sha256_file(Path(args.assembled_oto_out))
        report["assembled_rows"] = len(assembled_lines)
        split_output_paths: dict[str, str] = {}
        if args.split_out_dir:
            split_name = str(args.split_name or Path(str(args.assembled_oto_out)).stem).strip()
            split = split_oto_file(
                args.assembled_oto_out,
                args.split_out_dir,
                name=split_name,
                language=args.language,
                wav_root=args.wav_root,
                wav_durations_ms=_load_durations_json(args.durations_json),
                row_diagnostics=_row_diagnostics_from_timeline_records(timeline_records),
                session_metadata={
                    "created_by": "mfa_free_oto_review.replay-evidence-hsmm",
                    "evidence_jsonl": str(args.evidence_jsonl or ""),
                    "evidence_jsonl_sha256": evidence_sha256,
                    "assembled_oto_out": str(args.assembled_oto_out or ""),
                    "assembled_oto_sha256": str(report.get("assembled_oto_sha256", "") or ""),
                    "timeline_out": str(args.timeline_out or ""),
                    "timeline_out_sha256": timeline_out_sha256,
                    "reference_generated": str(args.reference_generated or ""),
                    "reference_validation_jsonl": str(args.reference_validation_jsonl or ""),
                    "reference_timeline": str(args.reference_timeline or ""),
                    "source_timing_trusted": False,
                    "language": args.language,
                    "format_type": args.format_type,
                    "alias_type": args.alias_type,
                },
            )
            report["split_counts"] = dict(split.counts)
            report["split_output_paths"] = dict(split.output_paths)
            split_output_paths = dict(split.output_paths)
    else:
        split_output_paths = {}
    reference_comparison: dict[str, object] = {}
    row_delta_limit = max(0, int(args.row_delta_limit))
    if args.reference_generated:
        reference_comparison["row_param_delta"] = _compare_generated_oto_params(
            args.reference_generated,
            args.assembled_oto_out,
            baseline_validation_path=args.reference_validation_jsonl,
            hsmm_validation_path=str(split_output_paths.get("validation_jsonl", "")),
            row_delta_limit=row_delta_limit,
        )
    if args.reference_timeline:
        reference_comparison["timeline_event_delta"] = _compare_timeline_events(
            args.reference_timeline,
            args.timeline_out,
            row_delta_limit=row_delta_limit,
        )
    if reference_comparison:
        report["reference_comparison"] = reference_comparison
        reference_failures = _reference_comparison_failures(reference_comparison)
        if reference_failures:
            report["ok"] = False
            report["reference_comparison_failures"] = reference_failures
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if report["ok"] else 2


def _cmd_merge(args: argparse.Namespace) -> int:
    result = merge_split_oto(
        validation_jsonl_path=args.validation_jsonl,
        final_path=args.final,
        edited_fix_required_path=args.edited_fix_required,
        edited_attention_only_path=args.edited_attention_only,
        allow_unresolved=bool(args.allow_unresolved),
        acknowledge_attention_only=bool(args.acknowledge_attention_only),
        language=args.language,
        preferred_encoding_source=args.preferred_encoding_source,
    )
    _print_json(
        {
            "ok": result.ok,
            "command": "merge",
            "final_path": result.final_path,
            "report_path": result.report_path,
            "row_count": result.row_count,
            "unresolved_count": len(result.unresolved_row_ids),
            "unresolved_row_ids": list(result.unresolved_row_ids),
            "validation_error_count": len(result.validation_errors),
            "validation_errors": list(result.validation_errors),
            "merge_input_error_count": len(result.merge_input_errors),
            "merge_input_errors": list(result.merge_input_errors),
            "attention_only_acknowledged": bool(result.attention_only_acknowledged),
            "attention_only_unacknowledged_count": int(result.attention_only_unacknowledged_count),
        }
    )
    return 0 if result.ok else 2


def _cmd_merge_session(args: argparse.Namespace) -> int:
    report = _build_merge_session_report(
        review_session_path=args.review_session,
        final_path=args.final,
        allow_unresolved=bool(args.allow_unresolved),
        acknowledge_attention_only=bool(args.acknowledge_attention_only),
        language=args.language,
        preferred_encoding_source=args.preferred_encoding_source,
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _cmd_evaluate_gate(args: argparse.Namespace) -> int:
    report = _build_artifact_gate_report(
        validation_jsonl_path=args.validation_jsonl,
        comparison_json_path=args.comparison_json,
        merge_report_path=args.merge_report,
        attention_ratio_max=float(args.attention_ratio_max),
        quality_evidence=str(args.quality_evidence),
        quality_evidence_report_path=str(args.quality_evidence_report),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    if bool(args.fail_on_gate_failure):
        target = str(args.gate_target or "default-on").strip().lower()
        gates = dict(report.get("gate_report", {}) or {})
        passed = bool(gates.get("apply_candidate" if target == "apply" else "default_on_candidate"))
        return 0 if passed else 2
    return 0 if bool(report.get("ok")) else 2


def _cmd_evaluate_session(args: argparse.Namespace) -> int:
    report = _build_evaluate_session_report(
        review_session_path=args.review_session,
        comparison_json_path=args.comparison_json,
        merge_report_path=args.merge_report,
        attention_ratio_max=float(args.attention_ratio_max),
        quality_evidence=str(args.quality_evidence),
        quality_evidence_report_path=str(args.quality_evidence_report),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    if bool(args.fail_on_gate_failure):
        target = str(args.gate_target or "apply").strip().lower()
        gates = dict(report.get("gate_report", {}) or {})
        passed = bool(gates.get("default_on_candidate" if target == "default-on" else "apply_candidate"))
        return 0 if passed else 2
    return 0 if bool(report.get("ok")) else 2


def _cmd_session_status(args: argparse.Namespace) -> int:
    report = _build_review_session_status(
        review_session_path=args.review_session,
        require_default_on=bool(args.require_default_on),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _cmd_prepare_edits(args: argparse.Namespace) -> int:
    report = _build_prepare_edits_report(
        review_session_path=args.review_session,
        overwrite=bool(args.overwrite),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _cmd_record_quality_evidence(args: argparse.Namespace) -> int:
    report = _build_quality_evidence_record(
        final_path=args.final,
        merge_report_path=args.merge_report,
        evidence_type=args.evidence_type,
        passed=bool(args.passed),
        critical_failures=int(args.critical_failures),
        reviewed_rows=int(args.reviewed_rows),
        reviewer=str(args.reviewer),
        notes=str(args.notes),
        manual_reference=str(args.manual_reference),
        manual_reference_oto=str(args.manual_reference_oto),
        manual_reference_max_delta_ms=float(args.manual_reference_max_delta_ms),
        manual_reference_row_delta_limit=int(args.manual_reference_row_delta_limit),
        listening_summary=str(args.listening_summary),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _cmd_install_final(args: argparse.Namespace) -> int:
    report = _build_install_final_report(
        final_path=args.final,
        target_oto=args.target_oto,
        gate_report_path=args.gate_report,
        backup_dir=args.backup_dir,
        replace=bool(args.replace),
        require_default_on=bool(args.require_default_on),
        allow_create=bool(args.allow_create),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _cmd_install_session(args: argparse.Namespace) -> int:
    report = _build_install_session_report(
        review_session_path=args.review_session,
        target_oto=args.target_oto,
        gate_report_path=args.gate_report,
        final_path=args.final,
        backup_dir=args.backup_dir,
        replace=bool(args.replace),
        require_default_on=bool(args.require_default_on),
        allow_create=bool(args.allow_create),
    )
    if args.out:
        _write_json(args.out, report)
        report["output_path"] = str(args.out)
    _print_json(report)
    return 0 if bool(report.get("ok")) else 2


def _build_review_session_status(*, review_session_path: str, require_default_on: bool = False) -> dict[str, object]:
    session_file = Path(str(review_session_path or ""))
    base = {
        "ok": False,
        "command": "session-status",
        "review_session": str(session_file),
    }
    if not session_file.is_file():
        return {**base, "reason": "missing_review_session"}
    session_payload, session_load_error = _load_review_session_payload(session_file)
    if session_load_error:
        return {**base, "reason": "invalid_review_session", "error": session_load_error}

    base_dir = session_file.parent
    lineage = dict(session_payload.get("lineage", {}) or {})
    output_paths = dict(lineage.get("output_paths", {}) or {})
    output_sha256 = dict(lineage.get("output_sha256", {}) or {})
    counts = {str(key): int(value or 0) for key, value in dict(session_payload.get("counts", {}) or {}).items()}
    artifacts: dict[str, object] = {}
    warnings: list[str] = []

    generated_status = _session_file_status(
        "generated_oto",
        session_payload.get("generated_oto_path", ""),
        expected_sha256=str(session_payload.get("generated_oto_sha256", "") or ""),
        base_dir=base_dir,
    )
    artifacts["generated_oto"] = generated_status

    validation_path = str(lineage.get("validation_jsonl_path", "") or output_paths.get("validation_jsonl", "") or "")
    validation_status = _session_file_status(
        "validation_jsonl",
        validation_path,
        expected_sha256=str(lineage.get("validation_jsonl_sha256", "") or output_sha256.get("validation_jsonl", "") or ""),
        base_dir=base_dir,
    )
    artifacts["validation_jsonl"] = validation_status

    summary_status = _session_file_status(
        "summary",
        lineage.get("summary_path", "") or output_paths.get("summary", ""),
        expected_sha256=str(lineage.get("summary_sha256", "") or ""),
        base_dir=base_dir,
    )
    artifacts["summary"] = summary_status

    metadata_artifacts_payload = dict(lineage.get("metadata_artifacts", {}) or {})
    metadata_payload = dict(session_payload.get("metadata", {}) or {})
    metadata_artifacts: dict[str, object] = {}
    metadata_artifact_checks: dict[str, bool] = {}
    for name, item in sorted(metadata_artifacts_payload.items()):
        if not isinstance(item, Mapping):
            metadata_artifacts[str(name)] = {
                "name": str(name),
                "exists": False,
                "sha256_match": False,
                "error": "invalid_metadata_artifact",
            }
            metadata_artifact_checks[str(name)] = False
            warnings.append(f"metadata_artifact_invalid:{name}")
            continue
        artifact = dict(item)
        status = _session_file_status(
            f"metadata_artifact:{name}",
            artifact.get("path", ""),
            expected_sha256=str(artifact.get("sha256", "") or ""),
            base_dir=base_dir,
        )
        expected_sha = str(status.get("expected_sha256", "") or "")
        current = bool(status.get("exists")) and bool(status.get("sha256_match")) and bool(expected_sha)
        status["role"] = str(artifact.get("role", "") or "")
        status["row_count"] = artifact.get("row_count", None)
        status["path_key"] = str(artifact.get("path_key", "") or "")
        metadata_artifacts[str(name)] = status
        metadata_artifact_checks[str(name)] = current
        if not bool(status.get("exists")):
            warnings.append(f"metadata_artifact_missing:{name}")
        elif not expected_sha:
            warnings.append(f"metadata_artifact_sha256_missing:{name}")
        elif status.get("sha256_match") is False:
            warnings.append(f"metadata_artifact_sha256_mismatch:{name}")
    for name, hint in _session_metadata_artifact_hints(metadata_payload).items():
        if name in metadata_artifacts_payload:
            continue
        status = _session_file_status(
            f"metadata_artifact:{name}",
            hint.get("path", ""),
            expected_sha256="",
            base_dir=base_dir,
        )
        status["role"] = str(hint.get("role", "") or "")
        status["path_key"] = str(hint.get("path_key", "") or "")
        status["lineage_missing"] = True
        metadata_artifacts[str(name)] = status
        metadata_artifact_checks[str(name)] = False
        warnings.append(f"metadata_artifact_lineage_missing:{name}")

    split_artifact_checks: dict[str, bool] = {}
    for name, expected_sha in sorted(output_sha256.items()):
        if name in {"summary", "review_session"}:
            continue
        status = _session_file_status(
            str(name),
            output_paths.get(name, ""),
            expected_sha256=str(expected_sha or ""),
            base_dir=base_dir,
        )
        artifacts[str(name)] = status
        split_artifact_checks[str(name)] = bool(status.get("exists")) and bool(status.get("sha256_match", True))
        if status.get("exists") and status.get("sha256_match") is False:
            warnings.append(f"artifact_sha256_mismatch:{name}")
        elif not status.get("exists"):
            warnings.append(f"artifact_missing:{name}")

    validation_records = []
    validation_errors: list[str] = []
    validation_metrics: dict[str, object] = {}
    validation_resolved = str(validation_status.get("resolved_path", "") or "")
    if bool(validation_status.get("exists")):
        validation_records, loaded_errors = load_validation_records_with_errors(validation_resolved, require_rows=True)
        validation_errors = list(loaded_errors)
        validation_metrics = _validation_gate_metrics(validation_records, validation_errors=validation_errors)
    else:
        validation_metrics = {
            "total_rows": 0,
            "clean_rows": 0,
            "attention_only_rows": 0,
            "fix_required_rows": 0,
            "validation_error_count": 1,
            "validation_errors": ["missing_validation_jsonl"],
        }

    validation_counts = {
        "total": int(validation_metrics.get("total_rows", 0) or 0),
        "clean": int(validation_metrics.get("clean_rows", 0) or 0),
        "attention_only": int(validation_metrics.get("attention_only_rows", 0) or 0),
        "fix_required": int(validation_metrics.get("fix_required_rows", 0) or 0),
    }
    validation_counts_match = _counts_match_session(counts, validation_counts)

    queue_counts = _session_queue_counts(session_payload, base_dir=base_dir)
    queue_counts_match = all(
        bool(dict(queue_counts.get(name, {}) or {}).get("count_match", False))
        for name in ("fix_required", "attention_only", "clean", "review_all")
    )

    summary_status_payload = _session_summary_status(
        summary_status,
        counts=counts,
        generated_sha=str(generated_status.get("sha256", "") or ""),
        validation_sha=str(validation_status.get("sha256", "") or ""),
    )

    merge_step = _recommended_step(session_payload, "merge_final")
    gate_step = _recommended_step(session_payload, "evaluate_apply_gate")
    merge_report_path = str(merge_step.get("expected_report", "") or "")
    gate_report_path = str(gate_step.get("expected_report", "") or "")
    edit_files = _session_edit_file_status(session_payload, base_dir=base_dir)
    merge_status = _session_merge_status(
        merge_report_path,
        validation_resolved,
        base_dir=base_dir,
        validation_records=validation_records,
    )
    gate_status = _session_gate_status(gate_report_path, base_dir=base_dir)

    if not bool(generated_status.get("exists")):
        warnings.append("generated_oto_missing")
    elif generated_status.get("sha256_match") is False:
        warnings.append("generated_oto_sha256_mismatch")
    if not bool(validation_status.get("exists")):
        warnings.append("validation_jsonl_missing")
    elif validation_status.get("sha256_match") is False:
        warnings.append("validation_jsonl_sha256_mismatch")
    if not bool(summary_status.get("exists")):
        warnings.append("summary_missing")
    elif summary_status.get("sha256_match") is False:
        warnings.append("summary_sha256_mismatch")
    if not validation_counts_match:
        warnings.append("validation_counts_mismatch")
    if not queue_counts_match:
        warnings.append("review_queue_counts_mismatch")
    if not bool(summary_status_payload.get("counts_match_session", False)):
        warnings.append("summary_counts_mismatch")
    if not bool(summary_status_payload.get("generated_oto_sha256_current_match", False)):
        warnings.append("summary_generated_oto_sha256_mismatch")
    if not bool(summary_status_payload.get("validation_jsonl_sha256_current_match", False)):
        warnings.append("summary_validation_jsonl_sha256_mismatch")
    if validation_errors:
        warnings.append("validation_jsonl_invalid")
    merge_report_exists = bool(merge_status.get("path_exists", merge_status.get("available", False)))
    merge_report_valid = not merge_report_exists or bool(merge_status.get("available", False))
    merge_validation_current = not merge_report_exists or bool(merge_status.get("validation_jsonl_current_match", False))
    merge_final_current = not merge_report_exists or bool(merge_status.get("final_sha256_current_match", False))
    merge_rows_current = not merge_report_exists or bool(merge_status.get("merged_rows_current_match", False))
    merge_report_current = bool(merge_report_valid and merge_validation_current and merge_final_current and merge_rows_current)
    if merge_report_exists and not bool(merge_status.get("available", False)):
        warnings.append("merge_report_invalid")
    if merge_report_exists and bool(merge_status.get("available", False)) and not bool(
        merge_status.get("validation_jsonl_current_match", False)
    ):
        warnings.append("merge_validation_jsonl_current_mismatch")
    if merge_report_exists and bool(merge_status.get("available", False)) and not bool(
        merge_status.get("final_sha256_current_match", False)
    ):
        warnings.append("merge_final_sha256_current_mismatch")
    if merge_report_exists and bool(merge_status.get("available", False)) and not bool(
        merge_status.get("merged_rows_current_match", False)
    ):
        warnings.append("merge_rows_lineage_current_mismatch")
    if bool(gate_status.get("path_exists", gate_status.get("available", False))):
        if not bool(gate_status.get("available", False)) or bool(gate_status.get("invalid", False)):
            warnings.append("gate_report_invalid")
        elif not bool(gate_status.get("recomputed", False)):
            warnings.append("gate_report_recompute_failed")

    checks = {
        "review_session_loaded": True,
        "generated_oto_current_match": bool(generated_status.get("exists"))
        and bool(generated_status.get("sha256_match", True)),
        "validation_jsonl_current_match": bool(validation_status.get("exists"))
        and bool(validation_status.get("sha256_match", True)),
        "summary_current_match": bool(summary_status.get("exists")) and bool(summary_status.get("sha256_match", True)),
        "split_outputs_current_match": all(split_artifact_checks.values()) if split_artifact_checks else False,
        "validation_jsonl_valid": int(validation_metrics.get("validation_error_count", 0) or 0) == 0,
        "validation_counts_match_session": validation_counts_match,
        "review_queue_counts_match_session": queue_counts_match,
        "summary_counts_match_session": bool(summary_status_payload.get("counts_match_session", False)),
        "summary_generated_oto_sha256_current_match": bool(
            summary_status_payload.get("generated_oto_sha256_current_match", False)
        ),
        "summary_validation_jsonl_sha256_current_match": bool(
            summary_status_payload.get("validation_jsonl_sha256_current_match", False)
        ),
        "metadata_artifacts_current_match": all(metadata_artifact_checks.values())
        if metadata_artifact_checks
        else True,
        "merge_report_valid": merge_report_valid,
        "merge_report_validation_jsonl_current_match": merge_validation_current,
        "merge_report_final_sha256_current_match": merge_final_current,
        "merge_report_rows_current_match": merge_rows_current,
        "merge_report_current_match": merge_report_current,
    }
    session_current = all(bool(value) for value in checks.values())
    next_action = _review_session_next_action(
        session_current=session_current,
        counts=counts,
        edit_files=edit_files,
        merge_status=merge_status,
        gate_status=gate_status,
        require_default_on=bool(require_default_on),
    )
    install_readiness = _review_session_install_readiness(
        session_current=session_current,
        merge_status=merge_status,
        gate_status=gate_status,
        require_default_on=bool(require_default_on),
    )
    return {
        **base,
        "ok": bool(session_current),
        "reason": "ok" if session_current else "stale_or_incomplete_review_session",
        "ready_to_install": bool(install_readiness.get("ready", False)),
        "ready_to_install_reason": str(install_readiness.get("reason", "")),
        "install_readiness": install_readiness,
        "state": str(session_payload.get("state", "") or ""),
        "checks": checks,
        "warnings": warnings,
        "counts": counts,
        "validation_counts": validation_counts,
        "queue_counts": queue_counts,
        "edit_files": edit_files,
        "artifacts": artifacts,
        "metadata_artifacts": metadata_artifacts,
        "summary": summary_status_payload,
        "merge": merge_status,
        "gate": gate_status,
        "next_action": next_action,
        "recommended_next_steps": list(session_payload.get("recommended_next_steps", []) or []),
        "apply_policy": dict(session_payload.get("apply_policy", {}) or {}),
    }


def _build_prepare_edits_report(*, review_session_path: str, overwrite: bool = False) -> dict[str, object]:
    session_file = Path(str(review_session_path or ""))
    base = {
        "ok": False,
        "command": "prepare-edits",
        "review_session": str(session_file),
        "overwrite": bool(overwrite),
    }
    if not session_file.is_file():
        return {**base, "reason": "missing_review_session"}
    session_payload, session_load_error = _load_review_session_payload(session_file)
    if session_load_error:
        return {**base, "reason": "invalid_review_session", "error": session_load_error}

    session_status = _build_review_session_status(review_session_path=str(session_file))
    if not bool(session_status.get("ok", False)):
        return {
            **base,
            "reason": "review_session_not_current",
            "session_status_warnings": list(session_status.get("warnings", []) or []),
            "session_status_checks": dict(session_status.get("checks", {}) or {}),
        }

    base_dir = session_file.parent
    outputs: dict[str, object] = {}
    errors: list[str] = []
    for name, paths in _review_edit_file_pairs(session_payload, base_dir=base_dir).items():
        source = paths.get("source")
        target = paths.get("target")
        if source is None or target is None:
            errors.append(f"{name}.missing_path")
            continue
        if not source.is_file():
            errors.append(f"{name}.missing_source")
            outputs[name] = _prepare_edit_output_status(source, target, action="missing_source")
            continue
        if source.resolve() == target.resolve():
            errors.append(f"{name}.target_matches_source")
            outputs[name] = _prepare_edit_output_status(source, target, action="target_matches_source")
            continue
        if target.exists() and not overwrite:
            outputs[name] = _prepare_edit_output_status(source, target, action="kept_existing")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        outputs[name] = _prepare_edit_output_status(source, target, action="overwritten" if overwrite else "created")

    ok = not errors
    return {
        **base,
        "ok": bool(ok),
        "reason": "ok" if ok else errors[0],
        "outputs": outputs,
        "errors": errors,
        "session_status": {
            "ok": True,
            "next_action": str(session_status.get("next_action", "") or ""),
            "ready_to_install": bool(session_status.get("ready_to_install", False)),
            "ready_to_install_reason": str(session_status.get("ready_to_install_reason", "") or ""),
            "warnings": list(session_status.get("warnings", []) or []),
        },
    }


def _build_merge_session_report(
    *,
    review_session_path: str,
    final_path: str = "",
    allow_unresolved: bool = False,
    acknowledge_attention_only: bool = False,
    language: str = "",
    preferred_encoding_source: str = "",
) -> dict[str, object]:
    session_file = Path(str(review_session_path or ""))
    base = {
        "ok": False,
        "command": "merge-session",
        "review_session": str(session_file),
    }
    if not session_file.is_file():
        return {**base, "reason": "missing_review_session"}
    session_payload, session_load_error = _load_review_session_payload(session_file)
    if session_load_error:
        return {**base, "reason": "invalid_review_session", "error": session_load_error}

    session_status = _build_review_session_status(review_session_path=str(session_file))
    if not bool(session_status.get("ok", False)):
        return {
            **base,
            "reason": "review_session_not_current",
            "session_status_warnings": list(session_status.get("warnings", []) or []),
            "session_status_checks": dict(session_status.get("checks", {}) or {}),
        }

    counts = dict(session_payload.get("counts", {}) or {})
    base_dir = session_file.parent
    edit_pairs = _review_edit_file_pairs(session_payload, base_dir=base_dir)
    missing_inputs: list[str] = []
    edited_fix = edit_pairs.get("fix_required", {}).get("target")
    edited_attention = edit_pairs.get("attention_only", {}).get("target")
    if int(counts.get("fix_required", 0) or 0) > 0 and not (edited_fix and edited_fix.is_file()):
        missing_inputs.append("missing_edited_fix_required")
    if int(counts.get("attention_only", 0) or 0) > 0 and not (edited_attention and edited_attention.is_file()):
        missing_inputs.append("missing_edited_attention_only")
    if missing_inputs:
        return {
            **base,
            "reason": missing_inputs[0],
            "missing_inputs": missing_inputs,
            "edit_files": dict(session_status.get("edit_files", {}) or {}),
        }

    lineage = dict(session_payload.get("lineage", {}) or {})
    validation_jsonl = str(lineage.get("validation_jsonl_path", "") or "")
    merge_step = _recommended_step(session_payload, "merge_final")
    merge_args = [str(item) for item in list(merge_step.get("command_args", []) or [])]
    final_candidate = str(final_path or "").strip() or _command_arg_value(merge_args, "--final")
    if not final_candidate:
        return {**base, "reason": "missing_final_path"}
    preferred = (
        str(preferred_encoding_source or "").strip()
        or _command_arg_value(merge_args, "--preferred-encoding-source")
        or str(session_payload.get("generated_oto_path", "") or "")
    )
    lang = str(language or "").strip() or _command_arg_value(merge_args, "--language")
    result = merge_split_oto(
        validation_jsonl_path=validation_jsonl,
        final_path=final_candidate,
        edited_fix_required_path=str(edited_fix or ""),
        edited_attention_only_path=str(edited_attention or ""),
        allow_unresolved=bool(allow_unresolved),
        acknowledge_attention_only=bool(acknowledge_attention_only),
        language=lang,
        preferred_encoding_source=preferred,
    )
    return {
        **base,
        "ok": bool(result.ok),
        "reason": "ok" if result.ok else "merge_failed",
        "final_path": result.final_path,
        "report_path": result.report_path,
        "row_count": result.row_count,
        "unresolved_count": len(result.unresolved_row_ids),
        "unresolved_row_ids": list(result.unresolved_row_ids),
        "validation_error_count": len(result.validation_errors),
        "validation_errors": list(result.validation_errors),
        "merge_input_error_count": len(result.merge_input_errors),
        "merge_input_errors": list(result.merge_input_errors),
        "attention_only_acknowledged": bool(result.attention_only_acknowledged),
        "attention_only_unacknowledged_count": int(result.attention_only_unacknowledged_count),
        "used_paths": {
            "validation_jsonl": validation_jsonl,
            "edited_fix_required": str(edited_fix or ""),
            "edited_attention_only": str(edited_attention or ""),
            "preferred_encoding_source": preferred,
        },
        "session_status": {
            "ok": True,
            "next_action": str(session_status.get("next_action", "") or ""),
            "ready_to_install": bool(session_status.get("ready_to_install", False)),
            "ready_to_install_reason": str(session_status.get("ready_to_install_reason", "") or ""),
            "warnings": list(session_status.get("warnings", []) or []),
        },
    }


def _build_evaluate_session_report(
    *,
    review_session_path: str,
    comparison_json_path: str = "",
    merge_report_path: str = "",
    attention_ratio_max: float = 0.50,
    quality_evidence: str = "none",
    quality_evidence_report_path: str = "",
) -> dict[str, object]:
    session_file = Path(str(review_session_path or ""))
    base = {
        "ok": False,
        "command": "evaluate-session",
        "review_session": str(session_file),
    }
    if not session_file.is_file():
        return {**base, "reason": "missing_review_session"}
    session_payload, session_load_error = _load_review_session_payload(session_file)
    if session_load_error:
        return {**base, "reason": "invalid_review_session", "error": session_load_error}

    session_status = _build_review_session_status(review_session_path=str(session_file))
    if not bool(session_status.get("ok", False)):
        return {
            **base,
            "reason": "review_session_not_current",
            "session_status_warnings": list(session_status.get("warnings", []) or []),
            "session_status_checks": dict(session_status.get("checks", {}) or {}),
        }

    lineage = dict(session_payload.get("lineage", {}) or {})
    validation_jsonl = str(lineage.get("validation_jsonl_path", "") or "")
    gate_step = _recommended_step(session_payload, "evaluate_apply_gate")
    gate_args = [str(item) for item in list(gate_step.get("command_args", []) or [])]
    comparison_path = str(comparison_json_path or "").strip()
    if not comparison_path:
        comparison_path = _command_arg_value(gate_args, "--comparison-json")
    if not comparison_path:
        comparison_path = str(dict(session_payload.get("metadata", {}) or {}).get("comparison_report_path", "") or "")
    merge_report = str(merge_report_path or "").strip() or _command_arg_value(gate_args, "--merge-report")
    if not merge_report:
        merge_step = _recommended_step(session_payload, "merge_final")
        final_path = _command_arg_value([str(item) for item in list(merge_step.get("command_args", []) or [])], "--final")
        merge_report = str(Path(final_path).with_suffix(".merge_report.json")) if final_path else ""

    report = _build_artifact_gate_report(
        validation_jsonl_path=validation_jsonl,
        comparison_json_path=comparison_path,
        merge_report_path=merge_report,
        attention_ratio_max=float(attention_ratio_max),
        quality_evidence=str(quality_evidence),
        quality_evidence_report_path=str(quality_evidence_report_path),
    )
    return {
        **report,
        "command": "evaluate-session",
        "review_session": str(session_file),
        "used_paths": {
            "validation_jsonl": validation_jsonl,
            "comparison_json": comparison_path,
            "merge_report": merge_report,
            "quality_evidence_report": str(quality_evidence_report_path or ""),
        },
        "session_status": {
            "ok": True,
            "next_action": str(session_status.get("next_action", "") or ""),
            "ready_to_install": bool(session_status.get("ready_to_install", False)),
            "ready_to_install_reason": str(session_status.get("ready_to_install_reason", "") or ""),
            "warnings": list(session_status.get("warnings", []) or []),
        },
    }


def _session_file_status(
    name: str,
    path_value: object,
    *,
    expected_sha256: str = "",
    base_dir: Path,
) -> dict[str, object]:
    path_text = str(path_value or "")
    target = _resolve_session_path(path_text, base_dir=base_dir)
    exists = bool(target and target.is_file())
    current_sha = _sha256_file(target) if exists and target is not None else ""
    expected_sha = str(expected_sha256 or "").strip().lower()
    sha_match = bool(exists and current_sha == expected_sha) if expected_sha else bool(exists)
    return {
        "name": str(name),
        "path": path_text,
        "resolved_path": str(target) if target is not None else "",
        "exists": exists,
        "expected_sha256": expected_sha,
        "sha256": current_sha,
        "sha256_match": bool(sha_match),
    }


def _session_metadata_artifact_hints(metadata: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    hints: dict[str, Mapping[str, object]] = {}
    for name, role, path_key in (
        ("row_plan", "row_plan_jsonl", "row_plan_path"),
        ("timeline", "timeline_jsonl", "timeline_path"),
        ("timeline_input", "timeline_jsonl", "timeline_jsonl"),
        ("timeline_replay", "timeline_jsonl", "timeline_out"),
        ("evidence", "evidence_jsonl", "evidence_path"),
        ("evidence_input", "evidence_jsonl", "evidence_jsonl"),
    ):
        path = str(metadata.get(path_key, "") or "").strip()
        if not path:
            continue
        hints[name] = {
            "path": path,
            "path_key": path_key,
            "role": role,
        }
    return hints


def _resolve_session_path(path_value: str, *, base_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return base_dir / path


def _session_queue_counts(session_payload: Mapping[str, object], *, base_dir: Path) -> dict[str, object]:
    lineage = dict(session_payload.get("lineage", {}) or {})
    output_paths = dict(lineage.get("output_paths", {}) or {})
    review_queues = dict(session_payload.get("review_queues", {}) or {})
    out: dict[str, object] = {}
    for name in ("fix_required", "attention_only", "clean"):
        queue = dict(review_queues.get(name, {}) or {})
        target = _resolve_session_path(str(queue.get("path", "") or output_paths.get(name, "") or ""), base_dir=base_dir)
        line_count = _nonempty_line_count(target) if target and target.is_file() else 0
        out[name] = {
            "path": str(queue.get("path", "") or output_paths.get(name, "") or ""),
            "resolved_path": str(target) if target is not None else "",
            "exists": bool(target and target.is_file()),
            "expected_count": int(queue.get("count", 0) or 0),
            "line_count": int(line_count),
            "count_match": int(queue.get("count", 0) or 0) == int(line_count),
        }
    review_all_path = str(output_paths.get("review_all", "") or "")
    edit_step = _recommended_step(session_payload, "edit_review_queues")
    edit_paths = dict(edit_step.get("paths", {}) or {}) if edit_step else {}
    if not review_all_path:
        review_all_path = str(edit_paths.get("review_all", "") or "")
    review_all = _resolve_session_path(review_all_path, base_dir=base_dir)
    review_all_count = _nonempty_line_count(review_all) if review_all and review_all.is_file() else 0
    expected_review_all_count = int(dict(session_payload.get("counts", {}) or {}).get("fix_required", 0) or 0) + int(
        dict(session_payload.get("counts", {}) or {}).get("attention_only", 0) or 0
    )
    out["review_all"] = {
        "path": review_all_path,
        "resolved_path": str(review_all) if review_all is not None else "",
        "exists": bool(review_all and review_all.is_file()),
        "expected_count": expected_review_all_count,
        "line_count": int(review_all_count),
        "count_match": expected_review_all_count == int(review_all_count),
    }
    return out


def _session_edit_file_status(session_payload: Mapping[str, object], *, base_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, paths in _review_edit_file_pairs(session_payload, base_dir=base_dir).items():
        source = paths.get("source")
        target = paths.get("target")
        source_exists = bool(source and source.is_file())
        target_exists = bool(target and target.is_file())
        source_count = _nonempty_line_count(source) if source_exists else 0
        target_count = _nonempty_line_count(target) if target_exists else 0
        out[name] = {
            "source_path": str(source) if source is not None else "",
            "target_path": str(target) if target is not None else "",
            "source_exists": source_exists,
            "target_exists": target_exists,
            "source_line_count": int(source_count),
            "target_line_count": int(target_count),
            "line_count_match": bool(target_exists and source_count == target_count),
        }
    return out


def _review_edit_file_pairs(session_payload: Mapping[str, object], *, base_dir: Path) -> dict[str, dict[str, Path | None]]:
    edit_step = _recommended_step(session_payload, "edit_review_queues")
    paths = dict(edit_step.get("paths", {}) or {}) if edit_step else {}
    edited_paths = dict(edit_step.get("edited_paths", {}) or {}) if edit_step else {}
    lineage = dict(session_payload.get("lineage", {}) or {})
    output_paths = dict(lineage.get("output_paths", {}) or {})
    pairs: dict[str, dict[str, Path | None]] = {}
    for name in ("fix_required", "attention_only"):
        source_text = str(paths.get(name, "") or output_paths.get(name, "") or "")
        source = _resolve_session_path(source_text, base_dir=base_dir)
        target_text = str(edited_paths.get(name, "") or "")
        if not target_text:
            target_text = _default_edited_path_text(source_text, split_name=name)
        target = _resolve_session_path(target_text, base_dir=base_dir)
        pairs[name] = {"source": source, "target": target}
    return pairs


def _default_edited_path_text(source_path: str, *, split_name: str) -> str:
    text = str(source_path or "")
    if not text:
        return ""
    suffix = f".{split_name}.ini"
    edited_suffix = f".edited_{split_name}.ini"
    if text.endswith(suffix):
        return text[: -len(suffix)] + edited_suffix
    path = Path(text)
    return str(path.with_name(f"{path.stem}.edited_{split_name}{path.suffix or '.ini'}"))


def _prepare_edit_output_status(source: Path, target: Path, *, action: str) -> dict[str, object]:
    return {
        "action": str(action),
        "source_path": str(source),
        "target_path": str(target),
        "source_exists": source.is_file(),
        "target_exists": target.is_file(),
        "source_sha256": _sha256_file(source) if source.is_file() else "",
        "target_sha256": _sha256_file(target) if target.is_file() else "",
        "source_line_count": _nonempty_line_count(source),
        "target_line_count": _nonempty_line_count(target),
    }


def _nonempty_line_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _counts_match_session(expected: Mapping[str, int], current: Mapping[str, int]) -> bool:
    for key in ("total", "fix_required", "attention_only", "clean"):
        if key not in expected or key not in current:
            return False
        if int(expected.get(key, 0) or 0) != int(current.get(key, 0) or 0):
            return False
    return True


def _session_summary_status(
    summary_status: Mapping[str, object],
    *,
    counts: Mapping[str, int],
    generated_sha: str,
    validation_sha: str,
) -> dict[str, object]:
    if not bool(summary_status.get("exists")):
        return {
            "available": False,
            "counts_match_session": False,
            "generated_oto_sha256_current_match": False,
            "validation_jsonl_sha256_current_match": False,
        }
    try:
        summary_payload = _load_json_file(str(summary_status.get("resolved_path", "") or ""))
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "counts_match_session": False,
            "generated_oto_sha256_current_match": False,
            "validation_jsonl_sha256_current_match": False,
        }
    summary_counts = {
        str(key): int(value or 0)
        for key, value in dict(summary_payload.get("counts", {}) or {}).items()
    }
    summary_generated_sha = str(summary_payload.get("generated_oto_sha256", "") or "").strip().lower()
    summary_validation_sha = str(summary_payload.get("validation_jsonl_sha256", "") or "").strip().lower()
    return {
        "available": bool(summary_payload),
        "counts": summary_counts,
        "counts_match_session": _counts_match_session(counts, summary_counts),
        "generated_oto_sha256": summary_generated_sha,
        "generated_oto_sha256_current_match": bool(generated_sha and summary_generated_sha == generated_sha),
        "validation_jsonl_sha256": summary_validation_sha,
        "validation_jsonl_sha256_current_match": bool(validation_sha and summary_validation_sha == validation_sha),
    }


def _recommended_step(session_payload: Mapping[str, object], step_id: str) -> dict[str, object]:
    for step in list(session_payload.get("recommended_next_steps", []) or []):
        if isinstance(step, Mapping) and str(step.get("id", "") or "") == step_id:
            return dict(step)
    return {}


def _command_arg_value(args: Sequence[str], flag: str) -> str:
    items = [str(item) for item in args]
    try:
        index = items.index(str(flag))
    except ValueError:
        return ""
    if index + 1 >= len(items):
        return ""
    return str(items[index + 1])


def _session_merge_status(
    merge_report_path: str,
    validation_jsonl_path: str,
    *,
    base_dir: Path,
    validation_records: Sequence[object] = (),
) -> dict[str, object]:
    target = _resolve_session_path(merge_report_path, base_dir=base_dir)
    base = {
        "path": str(merge_report_path or ""),
        "resolved_path": str(target) if target is not None else "",
        "path_exists": bool(target and target.is_file()),
        "available": bool(target and target.is_file()),
    }
    if not target or not target.is_file():
        return base
    try:
        payload, load_error = _load_json_object_file(str(target))
    except Exception as exc:
        return {**base, "available": False, "invalid": True, "error": str(exc)}
    if load_error:
        metrics = _merge_gate_metrics(
            {},
            validation_jsonl_path=validation_jsonl_path,
            validation_records=validation_records,
        )
        return {**base, **metrics, "available": False, "invalid": True, "error": load_error}
    metrics = _merge_gate_metrics(
        payload,
        validation_jsonl_path=validation_jsonl_path,
        validation_records=validation_records,
    )
    if not bool(metrics.get("merge_report_available", False)):
        return {**base, **metrics, "available": False, "invalid": True, "error": "empty_merge_report"}
    return {**base, **metrics, "available": True}


def _session_gate_status(gate_report_path: str, *, base_dir: Path) -> dict[str, object]:
    target = _resolve_session_path(gate_report_path, base_dir=base_dir)
    base = {
        "path": str(gate_report_path or ""),
        "resolved_path": str(target) if target is not None else "",
        "path_exists": bool(target and target.is_file()),
        "available": bool(target and target.is_file()),
    }
    if not target or not target.is_file():
        return base
    try:
        payload, load_error = _load_json_object_file(str(target))
    except Exception as exc:
        return {**base, "available": False, "invalid": True, "error": str(exc)}
    if load_error:
        return {**base, "available": False, "invalid": True, "error": load_error}
    if not payload:
        return {**base, "available": False, "invalid": True, "error": "empty_gate_report"}
    try:
        recomputed = _recompute_gate_from_report(payload)
    except Exception as exc:
        return {**base, "available": True, "recomputed": False, "error": str(exc)}
    gates = dict(recomputed.get("gate_report", {}) or {}) if recomputed else {}
    if not recomputed:
        return {
            **base,
            "available": True,
            "error": "gate_report_recompute_failed",
            "recomputed": False,
        }
    return {
        **base,
        "available": True,
        "recomputed": bool(recomputed),
        "apply_candidate": bool(gates.get("apply_candidate", False)),
        "default_on_candidate": bool(gates.get("default_on_candidate", False)),
        "failed_checks": list(gates.get("failed_checks", []) or []),
        "warnings": list(recomputed.get("warnings", []) or []) if recomputed else [],
    }


def _review_session_next_action(
    *,
    session_current: bool,
    counts: Mapping[str, int],
    edit_files: Mapping[str, object],
    merge_status: Mapping[str, object],
    gate_status: Mapping[str, object],
    require_default_on: bool = False,
) -> str:
    if not session_current:
        if bool(merge_status.get("path_exists", merge_status.get("available", False))) and not (
            bool(merge_status.get("available", False))
            and bool(merge_status.get("validation_jsonl_current_match", False))
            and bool(merge_status.get("final_sha256_current_match", False))
            and bool(merge_status.get("merged_rows_current_match", False))
        ):
            return "rerun_merge_final"
        return "regenerate_or_repair_review_split"
    if not bool(merge_status.get("available", False)):
        if any(not bool(dict(edit_files.get(name, {}) or {}).get("target_exists", False)) for name in ("fix_required", "attention_only")):
            return "prepare_review_edits"
        if int(counts.get("fix_required", 0) or 0) > 0:
            return "edit_fix_required_then_merge"
        if int(counts.get("attention_only", 0) or 0) > 0:
            return "review_attention_only_then_merge"
        return "run_merge_final"
    if not bool(merge_status.get("merge_ok", False)):
        return "fix_merge_errors_or_unresolved_rows"
    if int(merge_status.get("merge_input_error_count", 0) or 0) > 0:
        return "fix_merge_input_errors"
    if int(merge_status.get("unresolved_count", 0) or 0) > 0:
        return "edit_unresolved_rows_then_merge"
    if not bool(merge_status.get("merged_rows_current_match", False)):
        return "rerun_merge_final"
    if bool(gate_status.get("path_exists", gate_status.get("available", False))) and (
        not bool(gate_status.get("available", False))
        or bool(gate_status.get("invalid", False))
        or not bool(gate_status.get("recomputed", False))
    ):
        return "rerun_evaluate_apply_gate"
    if int(merge_status.get("attention_only_unacknowledged_count", 0) or 0) > 0:
        return "review_attention_only_then_rerun_merge_with_acknowledgement"
    if not bool(gate_status.get("available", False)):
        return "run_evaluate_apply_gate"
    if not bool(gate_status.get("recomputed", False)):
        return "rerun_evaluate_apply_gate"
    if not bool(gate_status.get("apply_candidate", False)):
        return "inspect_gate_failures"
    if bool(require_default_on):
        if bool(gate_status.get("default_on_candidate", False)):
            return "install_final_default_on_dry_run"
        return "record_quality_evidence_or_inspect_default_on_gate_failures"
    return "install_final_dry_run_or_record_quality_evidence"


def _review_session_install_readiness(
    *,
    session_current: bool,
    merge_status: Mapping[str, object],
    gate_status: Mapping[str, object],
    require_default_on: bool = False,
) -> dict[str, object]:
    base = {
        "ready": False,
        "reason": "",
        "requires_default_on": bool(require_default_on),
    }
    if not session_current:
        return {**base, "reason": "review_session_not_current"}
    if not bool(merge_status.get("available", False)):
        return {**base, "reason": "missing_merge_report"}
    if not bool(merge_status.get("merge_ok", False)):
        return {**base, "reason": "merge_report_not_ok"}
    if int(merge_status.get("merge_input_error_count", 0) or 0) > 0:
        return {**base, "reason": "merge_input_errors"}
    if int(merge_status.get("unresolved_count", 0) or 0) > 0:
        return {**base, "reason": "merge_unresolved_rows"}
    if not bool(merge_status.get("validation_jsonl_current_match", False)):
        return {**base, "reason": "merge_validation_jsonl_not_current"}
    if not bool(merge_status.get("final_sha256_current_match", False)):
        return {**base, "reason": "merge_final_sha256_not_current"}
    if not bool(merge_status.get("merged_rows_current_match", False)):
        return {**base, "reason": "merge_rows_lineage_not_current"}
    if bool(gate_status.get("path_exists", gate_status.get("available", False))) and (
        not bool(gate_status.get("available", False)) or bool(gate_status.get("invalid", False))
    ):
        return {**base, "reason": "invalid_gate_report"}
    if not bool(gate_status.get("available", False)):
        return {**base, "reason": "missing_gate_report"}
    if not bool(gate_status.get("recomputed", False)):
        return {**base, "reason": "gate_report_recompute_failed"}
    if not bool(gate_status.get("apply_candidate", False)):
        return {**base, "reason": "apply_gate_failed"}
    if bool(require_default_on) and not bool(gate_status.get("default_on_candidate", False)):
        return {**base, "reason": "default_on_gate_failed"}
    return {**base, "ready": True, "reason": "ready"}


def _build_quality_evidence_record(
    *,
    final_path: str,
    merge_report_path: str,
    evidence_type: str,
    passed: bool,
    critical_failures: int,
    reviewed_rows: int,
    reviewer: str = "",
    notes: str = "",
    manual_reference: str = "",
    manual_reference_oto: str = "",
    manual_reference_max_delta_ms: float = 250.0,
    manual_reference_row_delta_limit: int = 100,
    listening_summary: str = "",
) -> dict[str, object]:
    final = Path(str(final_path or ""))
    merge_report = Path(str(merge_report_path or ""))
    evidence = str(evidence_type or "none").strip().lower() or "none"
    base = {
        "ok": False,
        "command": "record-quality-evidence",
        "schema_version": 1,
        "evidence_type": evidence,
        "final_path": str(final),
        "merge_report": str(merge_report),
    }
    if evidence not in (_QUALITY_EVIDENCE_TYPES - {"none"}):
        return {**base, "reason": "invalid_evidence_type"}
    if not final.is_file():
        return {**base, "reason": "missing_final_ini"}
    if not merge_report.is_file():
        return {**base, "reason": "missing_merge_report"}

    try:
        merge_payload, merge_load_error = _load_json_object_file(str(merge_report))
    except Exception as exc:
        return {**base, "reason": "invalid_merge_report", "error": str(exc)}
    if merge_load_error:
        return {**base, "reason": "invalid_merge_report", "error": merge_load_error}
    if not merge_payload:
        if _json_report_path_exists(str(merge_report)):
            return {**base, "reason": "invalid_merge_report", "error": "empty_merge_report"}
        return {**base, "reason": "invalid_merge_report"}
    merge_final = Path(str(merge_payload.get("final_path", "") or ""))
    if not merge_final.is_file() or merge_final.resolve() != final.resolve():
        return {**base, "reason": "merge_report_final_mismatch", "merge_final_path": str(merge_final)}
    final_sha = _sha256_file(final)
    merge_sha = str(merge_payload.get("final_sha256", "") or "").strip().lower()
    if not merge_sha:
        return {**base, "reason": "merge_report_missing_final_sha256", "final_sha256": final_sha}
    if final_sha != merge_sha:
        return {
            **base,
            "reason": "final_sha256_mismatch",
            "final_sha256": final_sha,
            "merge_report_final_sha256": merge_sha,
        }

    validation_jsonl_path = str(merge_payload.get("validation_jsonl_path", "") or "")
    validation_records, validation_errors = load_validation_records_with_errors(validation_jsonl_path, require_rows=True)
    merge_metrics = _merge_gate_metrics(
        merge_payload,
        validation_jsonl_path=validation_jsonl_path,
        validation_records=validation_records,
    )

    critical_count = max(0, int(critical_failures))
    reviewed_count = max(0, int(reviewed_rows))
    failure_reasons: list[str] = []
    manual_reference_raw_path = str(manual_reference_oto or "").strip()
    manual_reference_path = str(Path(manual_reference_raw_path).resolve()) if manual_reference_raw_path else ""
    manual_reference_sha = ""
    listening_text = str(listening_summary or "").strip()
    if not passed:
        failure_reasons.append("quality_evidence_not_passed")
    if critical_count > 0:
        failure_reasons.append("quality_evidence_critical_failures")
    if reviewed_count <= 0:
        failure_reasons.append("quality_evidence_no_reviewed_rows")
    if not bool(merge_metrics.get("merge_ok", False)):
        failure_reasons.append("merge_report_not_ok")
    if validation_errors:
        failure_reasons.append("merge_validation_jsonl_invalid")
    if not bool(merge_metrics.get("validation_jsonl_current_match", False)):
        failure_reasons.append("merge_validation_jsonl_current_mismatch")
    if not bool(merge_metrics.get("final_sha256_current_match", False)):
        failure_reasons.append("merge_final_sha256_current_mismatch")
    if not bool(merge_metrics.get("merged_rows_current_match", False)):
        failure_reasons.append("merge_rows_lineage_current_mismatch")
    if int(merge_metrics.get("unresolved_count", 0) or 0) > 0:
        failure_reasons.append("merge_unresolved_rows")
    if int(merge_metrics.get("merge_input_error_count", 0) or 0) > 0:
        failure_reasons.append("merge_input_errors")
    if evidence in {"manual-reference", "manual-reference-and-listening"} and not manual_reference_path:
        failure_reasons.append("manual_reference_oto_required")
    if evidence in {"listening", "manual-reference-and-listening"} and not listening_text:
        failure_reasons.append("listening_summary_required")
    manual_reference_comparison: dict[str, object] = {}
    if manual_reference_path:
        manual_reference_file = Path(manual_reference_path)
        if manual_reference_file.is_file():
            manual_reference_sha = _sha256_file(manual_reference_file)
        manual_reference_comparison = _compare_generated_oto_params(
            manual_reference_path,
            str(final),
            row_delta_limit=max(0, int(manual_reference_row_delta_limit)),
        )
        if not bool(manual_reference_comparison.get("available", False)):
            failure_reasons.append("manual_reference_comparison_unavailable")
        if int(manual_reference_comparison.get("missing_in_hsmm_count", 0) or 0) > 0:
            failure_reasons.append("manual_reference_candidate_missing_rows")
        if int(manual_reference_comparison.get("hsmm_invalid_rows", 0) or 0) > 0:
            failure_reasons.append("manual_reference_candidate_invalid_rows")
        max_delta = _max_row_param_delta_ms(manual_reference_comparison)
        if max_delta > float(manual_reference_max_delta_ms):
            failure_reasons.append("manual_reference_max_delta_exceeded")
    report_valid = not failure_reasons
    return {
        **base,
        "ok": bool(report_valid),
        "reason": "ok" if report_valid else failure_reasons[0],
        "report_valid": bool(report_valid),
        "failure_reasons": failure_reasons,
        "passed": bool(passed),
        "critical_failures": critical_count,
        "reviewed_rows": reviewed_count,
        "final_sha256": final_sha,
        "merge_report_final_sha256": merge_sha,
        "merge_metrics": merge_metrics,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reviewer": str(reviewer or ""),
        "notes": str(notes or ""),
        "manual_reference": str(manual_reference or ""),
        "manual_reference_oto": manual_reference_path,
        "manual_reference_oto_sha256": manual_reference_sha,
        "manual_reference_max_delta_ms": float(manual_reference_max_delta_ms),
        "manual_reference_row_delta_limit": max(0, int(manual_reference_row_delta_limit)),
        "manual_reference_comparison": manual_reference_comparison,
        "listening_summary": listening_text,
    }


def _build_install_final_report(
    *,
    final_path: str,
    target_oto: str,
    gate_report_path: str,
    backup_dir: str = "",
    replace: bool = False,
    require_default_on: bool = False,
    allow_create: bool = False,
) -> dict[str, object]:
    final = Path(str(final_path or ""))
    target = _resolve_target_oto_path(target_oto)
    gate_report_file = Path(str(gate_report_path or ""))
    base = {
        "ok": False,
        "command": "install-final",
        "final_path": str(final),
        "target_oto": str(target),
        "gate_report_path": str(gate_report_file),
        "replace": bool(replace),
        "dry_run": not bool(replace),
        "require_default_on": bool(require_default_on),
    }
    if not final.is_file():
        return {**base, "reason": "missing_final_ini"}
    if not gate_report_file.is_file():
        return {**base, "reason": "missing_gate_report"}
    if not target.parent.is_dir():
        return {**base, "reason": "missing_target_parent"}
    target_exists = target.is_file()
    if not target_exists and not allow_create:
        return {**base, "reason": "missing_target_oto"}
    if target.resolve() == final.resolve():
        return {**base, "reason": "target_matches_final_path"}

    try:
        gate_payload, gate_load_error = _load_json_object_file(str(gate_report_file))
    except Exception as exc:
        return {**base, "reason": "invalid_gate_report", "error": str(exc)}
    if gate_load_error:
        return {**base, "reason": "invalid_gate_report", "error": gate_load_error}
    if not gate_payload:
        return {**base, "reason": "invalid_gate_report", "error": "empty_gate_report"}
    try:
        recomputed = _recompute_gate_from_report(gate_payload)
    except Exception as exc:
        return {**base, "reason": "invalid_or_stale_gate_report", "error": str(exc)}
    if not recomputed:
        return {**base, "reason": "invalid_gate_report"}
    gate = dict(recomputed.get("gate_report", {}) or {})
    gate_checks = dict(recomputed.get("checks", {}) or {})
    gate_metrics = dict(recomputed.get("metrics", {}) or {})
    gate_warnings = list(recomputed.get("warnings", []) or [])
    gate_key = "default_on_candidate" if require_default_on else "apply_candidate"
    if not bool(gate.get(gate_key, False)):
        return {
            **base,
            "reason": f"gate_failed:{gate_key}",
            "gate_target": gate_key,
            "gate_report": gate,
            "gate_checks": gate_checks,
            "gate_metrics": gate_metrics,
            "warnings": gate_warnings,
        }

    merge_report_path = str(recomputed.get("merge_report", "") or "")
    if not merge_report_path:
        return {**base, "reason": "missing_merge_report"}
    try:
        merge_payload, merge_load_error = _load_json_object_file(merge_report_path)
    except Exception as exc:
        return {**base, "reason": "invalid_merge_report", "error": str(exc)}
    if merge_load_error:
        return {**base, "reason": "invalid_merge_report", "error": merge_load_error}
    if not merge_payload:
        return {**base, "reason": "invalid_merge_report", "error": "empty_merge_report"}
    merge_final = Path(str(merge_payload.get("final_path", "") or ""))
    if not merge_final.is_file() or merge_final.resolve() != final.resolve():
        return {**base, "reason": "merge_report_final_mismatch", "merge_final_path": str(merge_final)}
    final_sha = _sha256_file(final)
    merge_sha = str(merge_payload.get("final_sha256", "") or "").strip().lower()
    if not merge_sha:
        return {**base, "reason": "merge_report_missing_final_sha256", "final_sha256": final_sha}
    if merge_sha and final_sha != merge_sha:
        return {
            **base,
            "reason": "final_sha256_mismatch",
            "final_sha256": final_sha,
            "merge_report_final_sha256": merge_sha,
        }

    backup_path = ""
    if target_exists:
        backup_root = Path(str(backup_dir or "")) if str(backup_dir or "").strip() else target.parent
        backup_path = str(backup_root / _backup_name(target))
    planned = {
        **base,
        "ok": True,
        "reason": "dry_run" if not replace else "installed",
        "gate_target": gate_key,
        "gate_report": gate,
        "gate_checks": gate_checks,
        "gate_metrics": gate_metrics,
        "warnings": gate_warnings,
        "target_exists": bool(target_exists),
        "backup_path": backup_path,
        "final_sha256": final_sha,
        "would_replace": True,
        "installed": False,
    }
    if not replace:
        return planned

    if target_exists:
        backup_target = Path(backup_path)
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_target)
    shutil.copy2(final, target)
    target_sha = _sha256_file(target)
    if target_sha != final_sha:
        return {
            **planned,
            "ok": False,
            "reason": "installed_sha256_mismatch",
            "target_sha256": target_sha,
        }
    return {
        **planned,
        "installed": True,
        "target_sha256": target_sha,
    }


def _build_install_session_report(
    *,
    review_session_path: str,
    target_oto: str,
    gate_report_path: str = "",
    final_path: str = "",
    backup_dir: str = "",
    replace: bool = False,
    require_default_on: bool = False,
    allow_create: bool = False,
) -> dict[str, object]:
    session_file = Path(str(review_session_path or ""))
    base = {
        "ok": False,
        "command": "install-session",
        "review_session": str(session_file),
        "target_oto": str(target_oto or ""),
    }
    if not session_file.is_file():
        return {**base, "reason": "missing_review_session"}
    session_payload, session_load_error = _load_review_session_payload(session_file)
    if session_load_error:
        return {**base, "reason": "invalid_review_session", "error": session_load_error}

    session_status = _build_review_session_status(
        review_session_path=str(session_file),
        require_default_on=bool(require_default_on),
    )
    if not bool(session_status.get("ok", False)):
        return {
            **base,
            "reason": "review_session_not_current",
            "session_status_warnings": list(session_status.get("warnings", []) or []),
            "session_status_checks": dict(session_status.get("checks", {}) or {}),
        }

    base_dir = session_file.parent
    gate_step = _recommended_step(session_payload, "evaluate_apply_gate")
    gate_candidate = str(gate_report_path or "").strip() or str(gate_step.get("expected_report", "") or "")
    gate_file = _resolve_session_path(gate_candidate, base_dir=base_dir)
    gate_path = str(gate_file) if gate_file is not None else gate_candidate

    final_candidate = str(final_path or "").strip()
    merge_report_path = ""
    if gate_file and gate_file.is_file():
        try:
            gate_payload = _load_json_file(str(gate_file))
            merge_report_path = str(gate_payload.get("merge_report", "") or "")
        except Exception:
            merge_report_path = ""
    if merge_report_path:
        merge_report_file = _resolve_session_path(merge_report_path, base_dir=base_dir)
        if merge_report_file and merge_report_file.is_file():
            try:
                merge_payload = _load_json_file(str(merge_report_file))
                final_candidate = final_candidate or str(merge_payload.get("final_path", "") or "")
            except Exception:
                pass
    if not final_candidate:
        merge_step = _recommended_step(session_payload, "merge_final")
        merge_args = [str(item) for item in list(merge_step.get("command_args", []) or [])]
        final_candidate = _command_arg_value(merge_args, "--final")

    final_file = _resolve_session_path(final_candidate, base_dir=base_dir)
    final_path_resolved = str(final_file) if final_file is not None else final_candidate
    install_report = _build_install_final_report(
        final_path=final_path_resolved,
        target_oto=target_oto,
        gate_report_path=gate_path,
        backup_dir=backup_dir,
        replace=bool(replace),
        require_default_on=bool(require_default_on),
        allow_create=bool(allow_create),
    )
    return {
        **install_report,
        "command": "install-session",
        "review_session": str(session_file),
        "used_paths": {
            "final": final_path_resolved,
            "gate_report": gate_path,
            "merge_report": merge_report_path,
        },
        "session_status": {
            "ok": True,
            "next_action": str(session_status.get("next_action", "") or ""),
            "ready_to_install": bool(session_status.get("ready_to_install", False)),
            "ready_to_install_reason": str(session_status.get("ready_to_install_reason", "") or ""),
            "install_readiness": dict(session_status.get("install_readiness", {}) or {}),
            "warnings": list(session_status.get("warnings", []) or []),
        },
    }


def _build_artifact_gate_report(
    *,
    validation_jsonl_path: str,
    comparison_json_path: str = "",
    merge_report_path: str = "",
    attention_ratio_max: float = 0.50,
    quality_evidence: str = "none",
    quality_evidence_report_path: str = "",
) -> dict[str, object]:
    validation_records, validation_errors = load_validation_records_with_errors(validation_jsonl_path, require_rows=True)
    validation_metrics = _validation_gate_metrics(validation_records, validation_errors=validation_errors)
    comparison_parse_error = ""
    merge_parse_error = ""
    try:
        comparison_payload, comparison_load_error = _load_json_object_file(comparison_json_path)
    except Exception as exc:
        comparison_payload = {}
        comparison_parse_error = str(exc)
    else:
        if comparison_load_error:
            comparison_parse_error = comparison_load_error
        elif not comparison_payload and _json_report_path_exists(comparison_json_path):
            comparison_parse_error = "empty_comparison_json"
    comparison_metrics = _comparison_gate_metrics(comparison_payload, validation_jsonl_path=validation_jsonl_path)
    if comparison_parse_error:
        comparison_metrics = {
            **comparison_metrics,
            "reason": "invalid_comparison_json",
            "comparison_json_invalid": True,
            "comparison_json_error": comparison_parse_error,
        }
    try:
        merge_payload, merge_load_error = _load_json_object_file(merge_report_path)
    except Exception as exc:
        merge_payload = {}
        merge_parse_error = str(exc)
    else:
        if merge_load_error:
            merge_parse_error = merge_load_error
        elif not merge_payload and _json_report_path_exists(merge_report_path):
            merge_parse_error = "empty_merge_report"
    merge_metrics = _merge_gate_metrics(
        merge_payload,
        validation_jsonl_path=validation_jsonl_path,
        validation_records=validation_records,
    )
    if merge_parse_error:
        merge_metrics = {
            **merge_metrics,
            "reason": "invalid_merge_report",
            "merge_report_invalid": True,
            "merge_report_error": merge_parse_error,
        }
    evidence = str(quality_evidence or "none").strip().lower() or "none"
    if evidence not in _QUALITY_EVIDENCE_TYPES:
        evidence = "none"
    evidence_metrics = _quality_evidence_gate_metrics(
        quality_evidence_report_path,
        requested_evidence=evidence,
        merge_metrics=merge_metrics,
    )
    if evidence == "none" and str(evidence_metrics.get("evidence_type", "none")) in _QUALITY_EVIDENCE_TYPES:
        evidence = str(evidence_metrics.get("evidence_type", "none") or "none")

    checks: dict[str, bool] = {
        "validation_records_present": int(validation_metrics["total_rows"]) > 0,
        "validation_jsonl_valid": int(validation_metrics["validation_error_count"]) == 0,
        "no_fix_required_rows": int(validation_metrics["fix_required_rows"]) == 0,
        "clean_invalid_oto_order": int(validation_metrics["clean_invalid_oto_order"]) == 0,
        "clean_decoder_timeout": int(validation_metrics["clean_decoder_timeout"]) == 0,
        "decoder_timeout_count": int(validation_metrics["decoder_timeout_count"]) == 0,
        "no_timing_clamp_large_delta_rows": int(validation_metrics["timing_clamp_large_delta_rows"]) == 0,
        "no_structural_warning_rows": int(validation_metrics["structural_warning_rows"]) == 0,
        "no_rule_based_inference_rows": int(validation_metrics["rule_based_inference_rows"]) == 0,
        "no_rule_based_checkpoint_failed_rows": int(validation_metrics["rule_based_checkpoint_failed_rows"]) == 0,
        "attention_only_ratio_default_on": float(validation_metrics["attention_only_ratio"]) <= float(attention_ratio_max),
        "comparison_available": bool(comparison_metrics.get("comparison_available")),
        "comparison_validation_jsonl_current_match": bool(
            comparison_metrics.get("validation_jsonl_current_match", False)
        ),
        "comparison_evidence_jsonl_current_match": bool(
            comparison_metrics.get("evidence_jsonl_current_match", True)
        ),
        "comparison_evidence_jsonl_rows_valid": bool(
            comparison_metrics.get("evidence_jsonl_rows_valid", True)
        ),
        "comparison_evidence_wav_sha256_current_match": bool(
            comparison_metrics.get("evidence_wav_sha256_current_match", True)
        ),
        "comparison_timeline_jsonl_current_match": bool(
            comparison_metrics.get("timeline_jsonl_current_match", True)
        ),
        "comparison_oto_current_match": bool(
            comparison_metrics.get("oto_current_match", True)
        ),
        "comparison_prior_provenance_available": bool(
            comparison_metrics.get("prior_provenance_available", True)
        ),
        "comparison_row_param_delta_current_match": bool(
            comparison_metrics.get("row_param_delta_current_match", True)
        ),
        "comparison_timeline_event_delta_current_match": bool(
            comparison_metrics.get("timeline_event_delta_current_match", True)
        ),
        "alias_coverage_delta_non_negative": int(comparison_metrics.get("alias_coverage_delta", -1)) >= 0,
        "no_missing_rows_in_candidate": int(comparison_metrics.get("missing_in_candidate_count", 1)) == 0,
        "reference_comparison_available": int(comparison_metrics.get("reference_comparison_failure_count", 0)) == 0,
        "merge_report_available": bool(merge_metrics.get("merge_report_available")),
        "merge_ok": bool(merge_metrics.get("merge_ok", False)),
        "merge_unresolved_zero": int(merge_metrics.get("unresolved_count", 0)) == 0,
        "merge_validation_jsonl_current_match": bool(merge_metrics.get("validation_jsonl_current_match", False)),
        "merge_final_sha256_current_match": bool(merge_metrics.get("final_sha256_current_match", False)),
        "merge_rows_lineage_current_match": bool(merge_metrics.get("merged_rows_current_match", False)),
        "attention_only_acknowledged_or_absent": int(merge_metrics.get("attention_only_unacknowledged_count", 0)) == 0,
        "quality_evidence_present": evidence != "none",
        "quality_evidence_report_valid": bool(evidence_metrics.get("report_valid", False)),
        "quality_evidence_final_sha_match": bool(evidence_metrics.get("final_sha256_match", False)),
    }
    apply_check_names = (
        "validation_records_present",
        "validation_jsonl_valid",
        "clean_invalid_oto_order",
        "clean_decoder_timeout",
        "decoder_timeout_count",
        "comparison_available",
        "comparison_validation_jsonl_current_match",
        "comparison_evidence_jsonl_current_match",
        "comparison_evidence_jsonl_rows_valid",
        "comparison_evidence_wav_sha256_current_match",
        "comparison_timeline_jsonl_current_match",
        "comparison_oto_current_match",
        "comparison_prior_provenance_available",
        "comparison_row_param_delta_current_match",
        "comparison_timeline_event_delta_current_match",
        "alias_coverage_delta_non_negative",
        "no_missing_rows_in_candidate",
        "reference_comparison_available",
        "no_rule_based_checkpoint_failed_rows",
        "merge_report_available",
        "merge_ok",
        "merge_unresolved_zero",
        "merge_validation_jsonl_current_match",
        "merge_final_sha256_current_match",
        "merge_rows_lineage_current_match",
        "attention_only_acknowledged_or_absent",
    )
    default_on_check_names = tuple(checks)
    apply_candidate = all(checks[name] for name in apply_check_names)
    default_on_candidate = all(checks[name] for name in default_on_check_names)
    failed_checks = [name for name, passed in checks.items() if not passed]
    report = {
        "ok": True,
        "command": "evaluate-gate",
        "validation_jsonl": str(validation_jsonl_path or ""),
        "comparison_json": str(comparison_json_path or ""),
        "merge_report": str(merge_report_path or ""),
        "quality_evidence": evidence,
        "quality_evidence_report": str(quality_evidence_report_path or ""),
        "metrics": {
            "validation": validation_metrics,
            "comparison": comparison_metrics,
            "merge": merge_metrics,
            "quality_evidence": evidence_metrics,
        },
        "checks": checks,
        "gate_report": {
            "apply_candidate": bool(apply_candidate),
            "default_on_candidate": bool(default_on_candidate),
            "failed_checks": failed_checks,
            "attention_ratio_max": float(attention_ratio_max),
            "quality_evidence_report_required_for_default_on": True,
        },
        "warnings": _gate_warnings(validation_metrics, comparison_metrics, merge_metrics, evidence, evidence_metrics),
    }
    return report


def _validation_gate_metrics(
    records: Sequence[object],
    *,
    validation_errors: Sequence[str] = (),
) -> dict[str, object]:
    total = len(records)
    counts = {
        "total_rows": total,
        "clean_rows": sum(1 for record in records if getattr(record, "split", "") == "clean"),
        "attention_only_rows": sum(1 for record in records if getattr(record, "split", "") == "attention_only"),
        "fix_required_rows": sum(1 for record in records if getattr(record, "split", "") == "fix_required"),
        "validation_error_count": len(tuple(validation_errors)),
        "validation_errors": list(validation_errors),
    }
    reason_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    vc_vv_worst = 0
    clean_invalid = 0
    clean_timeout = 0
    decoder_timeout = 0
    clean_hard = 0
    structural_warning_rows = 0
    structural_island_count_mismatch_rows = 0
    preutterance_slot_shift_rows = 0
    local_gate_attention_rows = 0
    local_gate_low_rows = 0
    missing_local_gate_rows = 0
    rule_based_inference_rows = 0
    rule_based_checkpoint_failed_rows = 0
    rule_based_checkpoint_missing_rows = 0
    for record in records:
        split = str(getattr(record, "split", "") or "")
        role = str(getattr(record, "role_family", "") or "other")
        role_counts[role] = role_counts.get(role, 0) + 1
        reasons = [str(item) for item in getattr(record, "reasons", ()) or ()]
        hard_reasons = [reason for reason in reasons if reason.startswith("fix_required.")]
        has_structural_island = "fix_required.structural_island_count_mismatch" in reasons
        has_preutterance_shift = "fix_required.preutterance_slot_shift" in reasons
        has_local_gate_low = "attention.local_gate_low" in reasons
        has_missing_local_gate = "attention.missing_local_gate" in reasons
        has_rule_based = (
            "attention.rule_based_inference" in reasons
            or "attention.rule_based_checkpoint_failed" in reasons
        )
        has_rule_based_failed = "attention.rule_based_checkpoint_failed" in reasons
        if has_structural_island or has_preutterance_shift:
            structural_warning_rows += 1
        if has_structural_island:
            structural_island_count_mismatch_rows += 1
        if has_preutterance_shift:
            preutterance_slot_shift_rows += 1
        if has_local_gate_low or has_missing_local_gate:
            local_gate_attention_rows += 1
        if has_local_gate_low:
            local_gate_low_rows += 1
        if has_missing_local_gate:
            missing_local_gate_rows += 1
        if has_rule_based:
            rule_based_inference_rows += 1
        if has_rule_based_failed:
            rule_based_checkpoint_failed_rows += 1
        if "attention.rule_based_inference" in reasons:
            metrics = getattr(record, "metrics", {}) or {}
            try:
                if float(dict(metrics).get("rule_based_checkpoint_missing_count", 0.0) or 0.0) > 0.0:
                    rule_based_checkpoint_missing_rows += 1
            except Exception:
                pass
        if role in {"vc", "vv"} and split == "fix_required":
            vc_vv_worst += 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if "decoder_timeout" in reason or "timeline_decode_failed" in reason:
                decoder_timeout += 1
                if split == "clean":
                    clean_timeout += 1
            if "invalid_oto_order" in reason and split == "clean":
                clean_invalid += 1
        if split == "clean" and hard_reasons:
            clean_hard += 1
    return {
        **counts,
        "attention_only_ratio": round(counts["attention_only_rows"] / max(1, total), 6),
        "fix_required_ratio": round(counts["fix_required_rows"] / max(1, total), 6),
        "clean_hard_failure_count": clean_hard,
        "clean_hard_failure_rate": round(clean_hard / max(1, counts["clean_rows"]), 6),
        "clean_invalid_oto_order": clean_invalid,
        "clean_decoder_timeout": clean_timeout,
        "decoder_timeout_count": decoder_timeout,
        "timing_clamped_rows": int(
            reason_counts.get("attention.timing_clamped", 0)
            + reason_counts.get("attention.timing_clamp_large_delta", 0)
        ),
        "timing_clamp_large_delta_rows": int(reason_counts.get("attention.timing_clamp_large_delta", 0)),
        "structural_warning_rows": structural_warning_rows,
        "structural_island_count_mismatch_rows": structural_island_count_mismatch_rows,
        "preutterance_slot_shift_rows": preutterance_slot_shift_rows,
        "local_gate_attention_rows": local_gate_attention_rows,
        "local_gate_low_rows": local_gate_low_rows,
        "missing_local_gate_rows": missing_local_gate_rows,
        "rule_based_inference_rows": rule_based_inference_rows,
        "rule_based_checkpoint_missing_rows": rule_based_checkpoint_missing_rows,
        "rule_based_checkpoint_failed_rows": rule_based_checkpoint_failed_rows,
        "vc_vv_worst_row_count": vc_vv_worst,
        "role_counts": role_counts,
        "reason_counts": reason_counts,
    }


def _quality_evidence_gate_metrics(
    path: str,
    *,
    requested_evidence: str,
    merge_metrics: Mapping[str, object],
) -> dict[str, object]:
    requested = str(requested_evidence or "none").strip().lower() or "none"
    if requested not in _QUALITY_EVIDENCE_TYPES:
        requested = "none"
    base = {
        "quality_evidence_report_available": False,
        "quality_evidence_report_path": str(path or ""),
        "report_valid": False,
        "reason": "missing_quality_evidence_report",
        "evidence_type": "none",
        "requested_evidence": requested,
        "requested_evidence_satisfied": False,
        "passed": False,
        "critical_failures": 0,
        "reviewed_rows": 0,
        "schema_version": 0,
        "final_sha256": "",
        "merge_final_sha256": str(merge_metrics.get("final_sha256", "") or ""),
        "final_sha256_match": False,
    }
    if not str(path or "").strip():
        return base
    try:
        payload, load_error = _load_json_object_file(path)
    except Exception as exc:
        return {
            **base,
            "quality_evidence_report_available": True,
            "reason": "invalid_quality_evidence_report",
            "failure_reasons": ["invalid_quality_evidence_report"],
            "error": str(exc),
        }
    if load_error:
        return {
            **base,
            "quality_evidence_report_available": True,
            "reason": "invalid_quality_evidence_report",
            "failure_reasons": ["invalid_quality_evidence_report"],
            "error": load_error,
        }
    if not payload:
        if _json_report_path_exists(path):
            return {
                **base,
                "quality_evidence_report_available": True,
                "reason": "invalid_quality_evidence_report",
                "failure_reasons": ["invalid_quality_evidence_report"],
                "error": "empty_quality_evidence_report",
            }
        return {**base, "reason": "missing_or_invalid_quality_evidence_report"}

    schema_version = _safe_int(payload.get("schema_version", 0))
    evidence_type = str(
        payload.get("evidence_type", payload.get("quality_evidence", "none")) or "none"
    ).strip().lower()
    if evidence_type not in _QUALITY_EVIDENCE_TYPES:
        evidence_type = "none"
    passed = bool(payload.get("passed", payload.get("ok", False)))
    critical_failures = max(0, _safe_int(payload.get("critical_failures", 0)))
    reviewed_rows = max(
        0,
        _safe_int(
            payload.get(
                "reviewed_rows",
                payload.get("reviewed_count", payload.get("total_reviewed_rows", 0)),
            )
        ),
    )
    final_sha = str(payload.get("final_sha256", "") or "").strip().lower()
    merge_sha = str(merge_metrics.get("final_sha256", "") or "").strip().lower()
    final_sha_match = bool(final_sha and merge_sha and final_sha == merge_sha)
    evidence_satisfied = _quality_evidence_satisfies(requested, evidence_type)
    manual_reference_oto = str(payload.get("manual_reference_oto", "") or "").strip()
    manual_reference_report_sha = str(payload.get("manual_reference_oto_sha256", "") or "").strip().lower()
    manual_reference_current_sha = ""
    if manual_reference_oto:
        manual_reference_file = Path(manual_reference_oto)
        if manual_reference_file.is_file():
            manual_reference_current_sha = _sha256_file(manual_reference_file).strip().lower()
    manual_reference_comparison_payload = payload.get("manual_reference_comparison", {})
    manual_reference_comparison = (
        manual_reference_comparison_payload
        if isinstance(manual_reference_comparison_payload, Mapping)
        else {}
    )
    manual_reference_comparison_available = bool(manual_reference_comparison.get("available", False))
    manual_reference_max_delta_ms = _float_or_default(
        payload.get("manual_reference_max_delta_ms", 250.0),
        250.0,
    )
    manual_reference_max_delta_actual = _max_row_param_delta_ms(manual_reference_comparison)
    manual_reference_row_delta_limit = max(
        0,
        _safe_int(
            payload.get(
                "manual_reference_row_delta_limit",
                manual_reference_comparison.get("row_delta_limit", 100),
            )
        ),
    )
    manual_reference_recomparison: dict[str, object] = {}
    manual_reference_recomparison_available = False
    manual_reference_recompare_max_delta_actual = 0.0
    merge_final_path = str(merge_metrics.get("final_path", "") or "").strip()
    if manual_reference_oto and merge_final_path:
        manual_reference_recomparison = _compare_generated_oto_params(
            manual_reference_oto,
            merge_final_path,
            row_delta_limit=manual_reference_row_delta_limit,
        )
        manual_reference_recomparison_available = bool(manual_reference_recomparison.get("available", False))
        manual_reference_recompare_max_delta_actual = _max_row_param_delta_ms(manual_reference_recomparison)
    listening_summary = str(payload.get("listening_summary", "") or "").strip()
    failure_reasons: list[str] = []
    if schema_version < 1:
        failure_reasons.append("quality_evidence_schema_version_invalid")
    if evidence_type == "none":
        failure_reasons.append("quality_evidence_type_missing")
    if not evidence_satisfied:
        failure_reasons.append("quality_evidence_type_mismatch")
    if not passed:
        failure_reasons.append("quality_evidence_not_passed")
    if critical_failures > 0:
        failure_reasons.append("quality_evidence_critical_failures")
    if reviewed_rows <= 0:
        failure_reasons.append("quality_evidence_no_reviewed_rows")
    if not final_sha_match:
        failure_reasons.append("quality_evidence_final_sha_mismatch")
    if evidence_type in {"manual-reference", "manual-reference-and-listening"}:
        if not manual_reference_oto:
            failure_reasons.append("manual_reference_oto_required")
        if not manual_reference_report_sha:
            failure_reasons.append("manual_reference_oto_sha256_missing")
        elif not manual_reference_current_sha:
            failure_reasons.append("manual_reference_oto_sha256_unavailable")
        elif manual_reference_report_sha != manual_reference_current_sha:
            failure_reasons.append("manual_reference_oto_sha256_mismatch")
        if not manual_reference_comparison_available:
            failure_reasons.append("manual_reference_comparison_unavailable")
        else:
            if _safe_int(manual_reference_comparison.get("matched_rows", 0)) <= 0:
                failure_reasons.append("manual_reference_no_matched_rows")
            if _safe_int(manual_reference_comparison.get("missing_in_hsmm_count", 0)) > 0:
                failure_reasons.append("manual_reference_candidate_missing_rows")
            if _safe_int(manual_reference_comparison.get("hsmm_invalid_rows", 0)) > 0:
                failure_reasons.append("manual_reference_candidate_invalid_rows")
            if manual_reference_max_delta_actual > manual_reference_max_delta_ms:
                failure_reasons.append("manual_reference_max_delta_exceeded")
        if not manual_reference_recomparison_available:
            failure_reasons.append("manual_reference_recompare_unavailable")
        else:
            if _safe_int(manual_reference_recomparison.get("matched_rows", 0)) <= 0:
                failure_reasons.append("manual_reference_recompare_no_matched_rows")
            if _safe_int(manual_reference_recomparison.get("missing_in_hsmm_count", 0)) > 0:
                failure_reasons.append("manual_reference_recompare_candidate_missing_rows")
            if _safe_int(manual_reference_recomparison.get("hsmm_invalid_rows", 0)) > 0:
                failure_reasons.append("manual_reference_recompare_candidate_invalid_rows")
            if manual_reference_recompare_max_delta_actual > manual_reference_max_delta_ms:
                failure_reasons.append("manual_reference_recompare_max_delta_exceeded")
    if evidence_type in {"listening", "manual-reference-and-listening"} and not listening_summary:
        failure_reasons.append("listening_summary_required")

    return {
        **base,
        "quality_evidence_report_available": True,
        "report_valid": not failure_reasons,
        "reason": "ok" if not failure_reasons else failure_reasons[0],
        "failure_reasons": failure_reasons,
        "evidence_type": evidence_type,
        "requested_evidence_satisfied": evidence_satisfied,
        "passed": passed,
        "critical_failures": critical_failures,
        "reviewed_rows": reviewed_rows,
        "schema_version": schema_version,
        "final_sha256": final_sha,
        "final_sha256_match": final_sha_match,
        "manual_reference_oto": manual_reference_oto,
        "manual_reference_oto_sha256": manual_reference_report_sha,
        "manual_reference_oto_current_sha256": manual_reference_current_sha,
        "manual_reference_oto_sha256_match": bool(
            manual_reference_report_sha
            and manual_reference_current_sha
            and manual_reference_report_sha == manual_reference_current_sha
        ),
        "manual_reference_comparison_available": manual_reference_comparison_available,
        "manual_reference_matched_rows": _safe_int(manual_reference_comparison.get("matched_rows", 0)),
        "manual_reference_max_delta_ms": manual_reference_max_delta_ms,
        "manual_reference_max_delta_actual_ms": manual_reference_max_delta_actual,
        "manual_reference_row_delta_limit": manual_reference_row_delta_limit,
        "manual_reference_recompare_available": manual_reference_recomparison_available,
        "manual_reference_recompare_matched_rows": _safe_int(
            manual_reference_recomparison.get("matched_rows", 0)
        ),
        "manual_reference_recompare_max_delta_actual_ms": manual_reference_recompare_max_delta_actual,
        "manual_reference_recompare_reason": str(manual_reference_recomparison.get("reason", "ok") or "ok"),
        "listening_summary_present": bool(listening_summary),
    }


def _quality_evidence_satisfies(requested: str, actual: str) -> bool:
    requested_norm = str(requested or "none").strip().lower() or "none"
    actual_norm = str(actual or "none").strip().lower() or "none"
    if actual_norm not in _QUALITY_EVIDENCE_TYPES or actual_norm == "none":
        return False
    if requested_norm in {"", "none"}:
        return True
    if requested_norm == actual_norm:
        return True
    return requested_norm in {"manual-reference", "listening"} and actual_norm == "manual-reference-and-listening"


def _max_row_param_delta_ms(comparison: Mapping[str, object]) -> float:
    max_by_param = comparison.get("max_abs_delta_ms", {})
    if not isinstance(max_by_param, Mapping):
        return 0.0
    values: list[float] = []
    for item in max_by_param.values():
        if not isinstance(item, Mapping):
            continue
        try:
            values.append(float(item.get("value_ms", 0.0) or 0.0))
        except Exception:
            continue
    return max(values) if values else 0.0


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _comparison_gate_metrics(
    payload: Mapping[str, object],
    *,
    validation_jsonl_path: str = "",
) -> dict[str, object]:
    current_validation_path = str(validation_jsonl_path or "")
    current_validation_sha = ""
    current_validation_exists = False
    if current_validation_path:
        current_validation_file = Path(current_validation_path)
        current_validation_exists = current_validation_file.is_file()
        if current_validation_exists:
            current_validation_sha = _sha256_file(current_validation_file).strip().lower()
    evidence_lineage = _comparison_evidence_lineage(payload)
    timeline_lineage = _comparison_timeline_lineage(payload)
    oto_lineage = _comparison_oto_lineage(payload)
    prior_provenance = _comparison_prior_provenance_metrics(payload)
    if not payload:
        return {
            "comparison_available": False,
            "reason": "missing_comparison_json",
            "alias_coverage_delta": -1,
            "missing_in_candidate_count": 1,
            "reference_comparison_failure_count": 0,
            "validation_jsonl_path": "",
            "validation_jsonl_sha256": "",
            "current_validation_jsonl_path": current_validation_path,
            "current_validation_jsonl_sha256": current_validation_sha,
            "validation_jsonl_exists": current_validation_exists,
            "validation_jsonl_path_current_match": False,
            "validation_jsonl_sha256_current_match": False,
            "validation_jsonl_current_match": False,
            **evidence_lineage,
            **timeline_lineage,
            **oto_lineage,
            **prior_provenance,
        }
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        comparison = payload.get("reference_comparison")
    if not isinstance(comparison, Mapping):
        return {
            "comparison_available": False,
            "reason": "missing_comparison_block",
            "alias_coverage_delta": -1,
            "missing_in_candidate_count": 1,
            "reference_comparison_failure_count": len(list(payload.get("reference_comparison_failures", []) or [])),
            "validation_jsonl_path": "",
            "validation_jsonl_sha256": "",
            "current_validation_jsonl_path": current_validation_path,
            "current_validation_jsonl_sha256": current_validation_sha,
            "validation_jsonl_exists": current_validation_exists,
            "validation_jsonl_path_current_match": False,
            "validation_jsonl_sha256_current_match": False,
            "validation_jsonl_current_match": False,
            **evidence_lineage,
            **timeline_lineage,
            **oto_lineage,
            **prior_provenance,
        }
    row_delta = comparison.get("row_param_delta")
    timeline_delta = comparison.get("timeline_event_delta")
    row_delta_map = row_delta if isinstance(row_delta, Mapping) else {}
    timeline_delta_map = timeline_delta if isinstance(timeline_delta, Mapping) else {}
    row_available = bool(row_delta_map.get("available", False))
    timeline_available = bool(timeline_delta_map.get("available", False))
    recompute_source_required = bool(row_available or timeline_available)
    row_integrity = _row_param_delta_integrity(
        row_delta_map,
        source_required=bool(row_available),
        baseline_oto_path=str(oto_lineage.get("baseline_oto_path", "") or ""),
        hsmm_oto_path=str(oto_lineage.get("hsmm_oto_path", "") or ""),
    )
    timeline_integrity = _timeline_event_delta_integrity(
        timeline_delta_map,
        source_required=bool(timeline_available),
        baseline_timeline_path=str(timeline_lineage.get("baseline_timeline_path", "") or ""),
        hsmm_timeline_path=str(timeline_lineage.get("hsmm_timeline_path", "") or ""),
    )
    baseline_rows = int(row_delta_map.get("baseline_rows", 0) or 0)
    candidate_rows = int(row_delta_map.get("hsmm_rows", 0) or 0)
    alias_coverage_delta = int(comparison.get("row_count_delta", candidate_rows - baseline_rows) or 0)
    missing_in_candidate = int(row_delta_map.get("missing_in_hsmm_count", 0) or 0)
    failures = list(payload.get("reference_comparison_failures", []) or [])
    reported_validation_path = str(row_delta_map.get("hsmm_validation_path", "") or "")
    reported_validation_sha = str(row_delta_map.get("hsmm_validation_sha256", "") or "").strip().lower()
    validation_path_match = _same_path(reported_validation_path, current_validation_path)
    validation_sha_match = bool(reported_validation_sha and current_validation_sha and reported_validation_sha == current_validation_sha)
    return {
        "comparison_available": row_available or timeline_available,
        "row_param_delta_available": row_available,
        "timeline_event_delta_available": timeline_available,
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "matched_rows": int(row_delta_map.get("matched_rows", 0) or 0),
        "alias_coverage_delta": alias_coverage_delta,
        "missing_in_candidate_count": missing_in_candidate,
        "missing_in_baseline_count": int(row_delta_map.get("missing_in_baseline_count", 0) or 0),
        "changed_rows": int(row_delta_map.get("changed_rows", 0) or 0),
        "split_changed_rows": int(row_delta_map.get("split_changed_rows", 0) or 0),
        "timeline_changed_events": int(timeline_delta_map.get("changed_events", 0) or 0),
        "timeline_matched_events": int(timeline_delta_map.get("matched_events", 0) or 0),
        "reference_comparison_failure_count": len(failures),
        "validation_jsonl_path": reported_validation_path,
        "validation_jsonl_sha256": reported_validation_sha,
        "current_validation_jsonl_path": current_validation_path,
        "current_validation_jsonl_sha256": current_validation_sha,
        "validation_jsonl_exists": current_validation_exists,
        "validation_jsonl_path_current_match": bool(validation_path_match),
        "validation_jsonl_sha256_current_match": bool(validation_sha_match),
        "validation_jsonl_current_match": bool(validation_path_match and validation_sha_match),
        "comparison_recompute_source_required": bool(recompute_source_required),
        **evidence_lineage,
        **timeline_lineage,
        **oto_lineage,
        **row_integrity,
        **timeline_integrity,
        **prior_provenance,
    }


def _comparison_prior_provenance_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    required = False
    missing: list[dict[str, object]] = []
    checked = 0
    for source, diagnostics in _comparison_hsmm_diagnostic_blocks(payload):
        checked += 1
        status = _diagnostics_prior_provenance_status(diagnostics, source=source)
        if bool(status["required"]):
            required = True
            missing.extend(list(status["missing_fields"]))
    return {
        "prior_provenance_required": bool(required),
        "prior_provenance_checked_blocks": int(checked),
        "prior_provenance_missing_count": int(len(missing)),
        "prior_provenance_missing_fields": missing[:50],
        "prior_provenance_available": bool((not required) or not missing),
    }


def _comparison_hsmm_diagnostic_blocks(payload: Mapping[str, object]) -> list[tuple[str, Mapping[str, object]]]:
    if not isinstance(payload, Mapping):
        return []
    blocks: list[tuple[str, Mapping[str, object]]] = []
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        comparison = payload.get("reference_comparison")
    if isinstance(comparison, Mapping):
        diagnostics = comparison.get("hsmm_diagnostics")
        if isinstance(diagnostics, Mapping):
            blocks.append(("comparison.hsmm_diagnostics", diagnostics))
    diagnostics = payload.get("hsmm_diagnostics")
    if isinstance(diagnostics, Mapping):
        blocks.append(("hsmm_diagnostics", diagnostics))
    for idx, record in enumerate(list(payload.get("records", []) or [])):
        if not isinstance(record, Mapping):
            continue
        diagnostics = record.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            blocks.append((f"records[{idx}].diagnostics", diagnostics))
    return blocks


def _diagnostics_prior_provenance_status(
    diagnostics: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    prior_state_count = _safe_int(diagnostics.get("event_prior_state_count", 0))
    mapped_count = _safe_int(diagnostics.get("event_prior_mapped_count", 0))
    candidate_count = _safe_int(diagnostics.get("candidate_prior_count", 0))
    runtime_count = _safe_int(diagnostics.get("runtime_prior_count", 0))
    required = prior_state_count > 0 or mapped_count > 0 or candidate_count > 0 or runtime_count > 0
    missing: list[dict[str, object]] = []
    if not required:
        return {"required": False, "missing_fields": missing}
    for key in (
        "event_prior_input_count",
        "event_prior_mapped_count",
        "event_prior_ignored_count",
        "event_prior_source_counts",
        "event_prior_label_counts",
        "candidate_prior_count",
        "runtime_prior_count",
    ):
        if key not in diagnostics:
            missing.append({"source": source, "field": key})
    for state_key in ("states", "lowest_margin_states"):
        states = diagnostics.get(state_key)
        if not isinstance(states, (list, tuple)):
            continue
        for idx, state in enumerate(states):
            if not isinstance(state, Mapping):
                continue
            state_has_prior = bool(state.get("has_event_prior")) or _safe_int(
                state.get("event_prior_mapped_count", 0)
            ) > 0
            if not state_has_prior:
                continue
            for key in (
                "event_prior_mapped_count",
                "event_prior_source_counts",
                "event_prior_label_counts",
                "evidence_candidate_prior_count",
                "runtime_prior_count",
            ):
                if key not in state:
                    missing.append({"source": f"{source}.{state_key}[{idx}]", "field": key})
    return {"required": True, "missing_fields": missing}


def _comparison_recompute_source_required(payload: Mapping[str, object]) -> bool:
    command = str(payload.get("command", "") or "").strip().lower()
    if command in {"compare-hsmm", "replay-evidence-hsmm"}:
        return True
    return isinstance(payload.get("baseline"), Mapping) or isinstance(payload.get("hsmm"), Mapping)


def _comparison_evidence_lineage(payload: Mapping[str, object]) -> dict[str, object]:
    evidence_path = ""
    evidence_sha = ""
    if isinstance(payload, Mapping):
        evidence_path = str(payload.get("evidence_jsonl", "") or "")
        evidence_sha = str(payload.get("evidence_jsonl_sha256", "") or "").strip().lower()
        hsmm_payload = payload.get("hsmm")
        if not evidence_path and isinstance(hsmm_payload, Mapping):
            evidence_path = str(hsmm_payload.get("evidence_path", "") or "")
            evidence_sha = str(hsmm_payload.get("evidence_sha256", "") or "").strip().lower()
    required = bool(evidence_path or evidence_sha)
    current_sha = ""
    exists = False
    if evidence_path:
        evidence_file = Path(evidence_path)
        exists = evidence_file.is_file()
        if exists:
            current_sha = _sha256_file(evidence_file).strip().lower()
    validation = _evidence_jsonl_current_validation(evidence_path, exists=exists)
    sha_match = bool(evidence_sha and current_sha and evidence_sha == current_sha)
    return {
        "evidence_jsonl_lineage_required": bool(required),
        "evidence_jsonl_path": evidence_path,
        "evidence_jsonl_sha256": evidence_sha,
        "current_evidence_jsonl_sha256": current_sha,
        "evidence_jsonl_exists": bool(exists),
        "evidence_jsonl_sha256_current_match": bool(sha_match),
        "evidence_jsonl_current_match": bool((not required) or sha_match),
        **validation,
    }


def _evidence_jsonl_current_validation(evidence_path: str, *, exists: bool) -> dict[str, object]:
    if not evidence_path or not exists:
        return {
            "evidence_jsonl_rows_checked": 0,
            "evidence_jsonl_invalid_count": 0,
            "evidence_jsonl_invalid_rows": [],
            "evidence_jsonl_rows_valid": True,
            "evidence_wav_sha256_error_count": 0,
            "evidence_wav_sha256_mismatch_count": 0,
            "evidence_wav_sha256_missing_count": 0,
            "evidence_wav_sha256_current_match": True,
        }
    try:
        rows = read_evidence_jsonl(evidence_path)
    except Exception as exc:
        return {
            "evidence_jsonl_rows_checked": 0,
            "evidence_jsonl_invalid_count": 1,
            "evidence_jsonl_invalid_rows": [{"row_index": 0, "wav": "", "errors": [f"invalid_evidence_jsonl:{exc}"]}],
            "evidence_jsonl_rows_valid": False,
            "evidence_wav_sha256_error_count": 0,
            "evidence_wav_sha256_mismatch_count": 0,
            "evidence_wav_sha256_missing_count": 0,
            "evidence_wav_sha256_current_match": True,
        }
    invalid_rows: list[dict[str, object]] = []
    wav_error_count = 0
    wav_mismatch_count = 0
    wav_missing_count = 0
    for idx, row in enumerate(rows):
        errors = list(validate_acoustic_evidence_pack(row))
        if not errors:
            continue
        wav_errors = [item for item in errors if str(item).startswith("evidence.wav_sha256")]
        wav_error_count += len(wav_errors)
        wav_mismatch_count += sum(1 for item in wav_errors if str(item) == "evidence.wav_sha256_mismatch")
        wav_missing_count += sum(1 for item in wav_errors if str(item) == "evidence.wav_sha256_missing")
        invalid_rows.append(
            {
                "row_index": idx,
                "wav": str(row.get("wav", "") or row.get("wav_path", "") or f"row_{idx}"),
                "errors": errors[:20],
            }
        )
    return {
        "evidence_jsonl_rows_checked": len(rows),
        "evidence_jsonl_invalid_count": len(invalid_rows),
        "evidence_jsonl_invalid_rows": invalid_rows[:20],
        "evidence_jsonl_rows_valid": len(invalid_rows) == 0,
        "evidence_wav_sha256_error_count": wav_error_count,
        "evidence_wav_sha256_mismatch_count": wav_mismatch_count,
        "evidence_wav_sha256_missing_count": wav_missing_count,
        "evidence_wav_sha256_current_match": wav_error_count == 0,
    }


def _comparison_timeline_lineage(payload: Mapping[str, object]) -> dict[str, object]:
    baseline_path = ""
    baseline_sha = ""
    hsmm_path = ""
    hsmm_sha = ""
    if isinstance(payload, Mapping):
        comparison = payload.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = payload.get("reference_comparison")
        if isinstance(comparison, Mapping):
            timeline_delta = comparison.get("timeline_event_delta")
            if isinstance(timeline_delta, Mapping):
                baseline_path = str(timeline_delta.get("baseline_timeline_path", "") or "")
                baseline_sha = str(timeline_delta.get("baseline_timeline_sha256", "") or "").strip().lower()
                hsmm_path = str(timeline_delta.get("hsmm_timeline_path", "") or "")
                hsmm_sha = str(timeline_delta.get("hsmm_timeline_sha256", "") or "").strip().lower()
        baseline_payload = payload.get("baseline")
        if not baseline_path and isinstance(baseline_payload, Mapping):
            baseline_path = str(baseline_payload.get("timeline_path", "") or "")
            baseline_sha = str(baseline_payload.get("timeline_sha256", "") or "").strip().lower()
        hsmm_payload = payload.get("hsmm")
        if not hsmm_path and isinstance(hsmm_payload, Mapping):
            hsmm_path = str(hsmm_payload.get("timeline_path", "") or "")
            hsmm_sha = str(hsmm_payload.get("timeline_sha256", "") or "").strip().lower()
        if not hsmm_path:
            hsmm_path = str(payload.get("timeline_path", "") or "")
            hsmm_sha = str(payload.get("timeline_sha256", "") or "").strip().lower()

    baseline = _artifact_lineage_side(baseline_path, baseline_sha)
    hsmm = _artifact_lineage_side(hsmm_path, hsmm_sha)
    required = bool(baseline["required"] or hsmm["required"])
    current_match = bool((not required) or (baseline["current_match"] and hsmm["current_match"]))
    return {
        "timeline_jsonl_lineage_required": bool(required),
        "baseline_timeline_path": baseline_path,
        "baseline_timeline_sha256": baseline_sha,
        "current_baseline_timeline_sha256": str(baseline["current_sha256"]),
        "baseline_timeline_exists": bool(baseline["exists"]),
        "baseline_timeline_sha256_current_match": bool(baseline["current_match"]),
        "hsmm_timeline_path": hsmm_path,
        "hsmm_timeline_sha256": hsmm_sha,
        "current_hsmm_timeline_sha256": str(hsmm["current_sha256"]),
        "hsmm_timeline_exists": bool(hsmm["exists"]),
        "hsmm_timeline_sha256_current_match": bool(hsmm["current_match"]),
        "timeline_jsonl_sha256_current_match": current_match,
        "timeline_jsonl_current_match": current_match,
    }


def _comparison_oto_lineage(payload: Mapping[str, object]) -> dict[str, object]:
    baseline_path = ""
    baseline_sha = ""
    hsmm_path = ""
    hsmm_sha = ""
    if isinstance(payload, Mapping):
        comparison = payload.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = payload.get("reference_comparison")
        if isinstance(comparison, Mapping):
            row_delta = comparison.get("row_param_delta")
            if isinstance(row_delta, Mapping):
                baseline_path = str(row_delta.get("baseline_oto_path", "") or "")
                baseline_sha = str(row_delta.get("baseline_oto_sha256", "") or "").strip().lower()
                hsmm_path = str(row_delta.get("hsmm_oto_path", "") or "")
                hsmm_sha = str(row_delta.get("hsmm_oto_sha256", "") or "").strip().lower()
        baseline_payload = payload.get("baseline")
        if not baseline_path and isinstance(baseline_payload, Mapping):
            baseline_path = str(baseline_payload.get("generated_oto_path", "") or "")
            baseline_sha = str(baseline_payload.get("generated_oto_sha256", "") or "").strip().lower()
        hsmm_payload = payload.get("hsmm")
        if not hsmm_path and isinstance(hsmm_payload, Mapping):
            hsmm_path = str(hsmm_payload.get("generated_oto_path", "") or "")
            hsmm_sha = str(hsmm_payload.get("generated_oto_sha256", "") or "").strip().lower()
        if not hsmm_path:
            hsmm_path = str(payload.get("assembled_oto_path", payload.get("generated_oto_path", "")) or "")
            hsmm_sha = str(payload.get("assembled_oto_sha256", payload.get("generated_oto_sha256", "")) or "").strip().lower()

    baseline = _artifact_lineage_side(baseline_path, baseline_sha)
    hsmm = _artifact_lineage_side(hsmm_path, hsmm_sha)
    required = bool(baseline["required"] or hsmm["required"])
    current_match = bool((not required) or (baseline["current_match"] and hsmm["current_match"]))
    return {
        "oto_lineage_required": bool(required),
        "baseline_oto_path": baseline_path,
        "baseline_oto_sha256": baseline_sha,
        "current_baseline_oto_sha256": str(baseline["current_sha256"]),
        "baseline_oto_exists": bool(baseline["exists"]),
        "baseline_oto_sha256_current_match": bool(baseline["current_match"]),
        "hsmm_oto_path": hsmm_path,
        "hsmm_oto_sha256": hsmm_sha,
        "current_hsmm_oto_sha256": str(hsmm["current_sha256"]),
        "hsmm_oto_exists": bool(hsmm["exists"]),
        "hsmm_oto_sha256_current_match": bool(hsmm["current_match"]),
        "oto_sha256_current_match": current_match,
        "oto_current_match": current_match,
    }


def _row_param_delta_integrity(
    row_delta: Mapping[str, object],
    *,
    source_required: bool = False,
    baseline_oto_path: str = "",
    hsmm_oto_path: str = "",
) -> dict[str, object]:
    baseline_path = str(row_delta.get("baseline_oto_path", "") or "") or str(baseline_oto_path or "")
    hsmm_path = str(row_delta.get("hsmm_oto_path", "") or "") or str(hsmm_oto_path or "")
    required = bool(source_required or baseline_path or hsmm_path)
    source_missing = bool(source_required and (not baseline_path or not hsmm_path))
    base = {
        "row_param_delta_recompute_required": bool(required),
        "row_param_delta_recompute_source_required": bool(source_required),
        "row_param_delta_recomputed_available": False,
        "row_param_delta_current_match": not required,
        "row_param_delta_mismatched_fields": [],
        "row_param_delta_recomputed_summary": {},
    }
    if not required:
        return base
    if source_missing:
        return {
            **base,
            "row_param_delta_recompute_error": "missing_recompute_source_path",
        }
    try:
        recomputed = _compare_generated_oto_params(
            baseline_path,
            hsmm_path,
            baseline_validation_path=str(row_delta.get("baseline_validation_path", "") or ""),
            hsmm_validation_path=str(row_delta.get("hsmm_validation_path", "") or ""),
            row_delta_limit=0,
        )
    except Exception as exc:
        return {
            **base,
            "row_param_delta_recompute_error": str(exc),
        }
    fields = (
        "available",
        "reason",
        "baseline_rows",
        "hsmm_rows",
        "matched_rows",
        "missing_in_baseline_count",
        "missing_in_hsmm_count",
        "changed_rows",
        "split_changed_rows",
        "baseline_invalid_rows",
        "hsmm_invalid_rows",
        "mean_abs_delta_ms",
        "max_abs_delta_ms",
    )
    mismatches = _comparison_mismatched_fields(row_delta, recomputed, fields)
    return {
        **base,
        "row_param_delta_recomputed_available": bool(recomputed.get("available", False)),
        "row_param_delta_current_match": not mismatches,
        "row_param_delta_mismatched_fields": mismatches,
        "row_param_delta_recomputed_summary": {
            "available": bool(recomputed.get("available", False)),
            "reason": str(recomputed.get("reason", "") or ""),
            "baseline_rows": _safe_int(recomputed.get("baseline_rows", 0)),
            "hsmm_rows": _safe_int(recomputed.get("hsmm_rows", 0)),
            "matched_rows": _safe_int(recomputed.get("matched_rows", 0)),
            "missing_in_baseline_count": _safe_int(recomputed.get("missing_in_baseline_count", 0)),
            "missing_in_hsmm_count": _safe_int(recomputed.get("missing_in_hsmm_count", 0)),
            "changed_rows": _safe_int(recomputed.get("changed_rows", 0)),
            "split_changed_rows": _safe_int(recomputed.get("split_changed_rows", 0)),
        },
    }


def _timeline_event_delta_integrity(
    timeline_delta: Mapping[str, object],
    *,
    source_required: bool = False,
    baseline_timeline_path: str = "",
    hsmm_timeline_path: str = "",
) -> dict[str, object]:
    baseline_path = str(timeline_delta.get("baseline_timeline_path", "") or "") or str(baseline_timeline_path or "")
    hsmm_path = str(timeline_delta.get("hsmm_timeline_path", "") or "") or str(hsmm_timeline_path or "")
    required = bool(source_required or baseline_path or hsmm_path)
    source_missing = bool(source_required and (not baseline_path or not hsmm_path))
    base = {
        "timeline_event_delta_recompute_required": bool(required),
        "timeline_event_delta_recompute_source_required": bool(source_required),
        "timeline_event_delta_recomputed_available": False,
        "timeline_event_delta_current_match": not required,
        "timeline_event_delta_mismatched_fields": [],
        "timeline_event_delta_recomputed_summary": {},
    }
    if not required:
        return base
    if source_missing:
        return {
            **base,
            "timeline_event_delta_recompute_error": "missing_recompute_source_path",
        }
    try:
        recomputed = _compare_timeline_events(
            baseline_path,
            hsmm_path,
            row_delta_limit=0,
        )
    except Exception as exc:
        return {
            **base,
            "timeline_event_delta_recompute_error": str(exc),
        }
    fields = (
        "available",
        "reason",
        "baseline_wavs",
        "hsmm_wavs",
        "wavs_compared",
        "missing_in_baseline_count",
        "missing_in_hsmm_count",
        "matched_events",
        "changed_events",
        "event_delta_rows_total",
    )
    mismatches = _comparison_mismatched_fields(timeline_delta, recomputed, fields)
    return {
        **base,
        "timeline_event_delta_recomputed_available": bool(recomputed.get("available", False)),
        "timeline_event_delta_current_match": not mismatches,
        "timeline_event_delta_mismatched_fields": mismatches,
        "timeline_event_delta_recomputed_summary": {
            "available": bool(recomputed.get("available", False)),
            "reason": str(recomputed.get("reason", "") or ""),
            "baseline_wavs": _safe_int(recomputed.get("baseline_wavs", 0)),
            "hsmm_wavs": _safe_int(recomputed.get("hsmm_wavs", 0)),
            "wavs_compared": _safe_int(recomputed.get("wavs_compared", 0)),
            "missing_in_baseline_count": _safe_int(recomputed.get("missing_in_baseline_count", 0)),
            "missing_in_hsmm_count": _safe_int(recomputed.get("missing_in_hsmm_count", 0)),
            "matched_events": _safe_int(recomputed.get("matched_events", 0)),
            "changed_events": _safe_int(recomputed.get("changed_events", 0)),
        },
    }


def _comparison_mismatched_fields(
    reported: Mapping[str, object],
    recomputed: Mapping[str, object],
    fields: Sequence[str],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for field in fields:
        reported_value = reported.get(field, "")
        recomputed_value = recomputed.get(field, "")
        if _canonical_json_value(reported_value) != _canonical_json_value(recomputed_value):
            mismatches.append(
                {
                    "field": field,
                    "reported": reported_value,
                    "recomputed": recomputed_value,
                }
            )
    return mismatches


def _canonical_json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _artifact_lineage_side(path: str, sha256: str) -> dict[str, object]:
    required = bool(path or sha256)
    current_sha = ""
    exists = False
    if path:
        timeline_file = Path(path)
        exists = timeline_file.is_file()
        if exists:
            current_sha = _sha256_file(timeline_file).strip().lower()
    current_match = bool((not required) or (path and sha256 and exists and current_sha == sha256))
    return {
        "required": bool(required),
        "exists": bool(exists),
        "current_sha256": current_sha,
        "current_match": bool(current_match),
    }


_MERGE_OUTPUT_SOURCES = frozenset(
    {
        "clean",
        "edited_fix_required",
        "edited_attention_only",
        "attention_only_original",
        "fix_required_unresolved_original",
    }
)


def _merge_gate_metrics(
    payload: Mapping[str, object],
    *,
    validation_jsonl_path: str = "",
    validation_records: Sequence[object] = (),
) -> dict[str, object]:
    current_validation_path = str(validation_jsonl_path or "")
    current_validation_sha = ""
    current_validation_exists = False
    if current_validation_path:
        current_validation_file = Path(current_validation_path)
        current_validation_exists = current_validation_file.is_file()
        if current_validation_exists:
            current_validation_sha = _sha256_file(current_validation_file).strip().lower()
    if not payload:
        return {
            "merge_report_available": False,
            "unresolved_count": 0,
            "validation_error_count": 0,
            "validation_errors": [],
            "merge_input_error_count": 0,
            "merge_input_errors": [],
            "validation_jsonl_path": "",
            "validation_jsonl_sha256": "",
            "current_validation_jsonl_path": current_validation_path,
            "current_validation_jsonl_sha256": current_validation_sha,
            "validation_jsonl_exists": current_validation_exists,
            "validation_jsonl_path_current_match": False,
            "validation_jsonl_sha256_current_match": False,
            "validation_jsonl_current_match": False,
            "final_path": "",
            "final_sha256": "",
            "current_final_sha256": "",
            "final_sha256_current_match": False,
            "merged_rows_available": False,
            "merged_row_count": 0,
            "merged_rows_current_match": False,
            "merged_rows_error_count": 0,
            "merged_rows_errors": [],
        }
    reported_validation_path = str(payload.get("validation_jsonl_path", "") or "")
    reported_validation_sha = str(payload.get("validation_jsonl_sha256", "") or "").strip().lower()
    validation_path_match = _same_path(reported_validation_path, current_validation_path)
    validation_sha_match = bool(reported_validation_sha and current_validation_sha and reported_validation_sha == current_validation_sha)
    final_path = str(payload.get("final_path", "") or "")
    final_sha = str(payload.get("final_sha256", "") or "").strip().lower()
    current_final_sha = ""
    final_exists = False
    if final_path:
        final_file = Path(final_path)
        final_exists = final_file.is_file()
        if final_exists:
            current_final_sha = _sha256_file(final_file).strip().lower()
    merged_row_metrics = _merge_rows_lineage_metrics(
        payload,
        final_path=final_path,
        final_exists=final_exists,
        validation_records=validation_records,
    )
    return {
        "merge_report_available": True,
        "merge_ok": bool(payload.get("ok", False)),
        "unresolved_count": int(payload.get("unresolved_count", 0) or 0),
        "validation_error_count": int(payload.get("validation_error_count", 0) or 0),
        "validation_errors": list(payload.get("validation_errors", []) or []),
        "merge_input_error_count": int(payload.get("merge_input_error_count", 0) or 0),
        "merge_input_errors": list(payload.get("merge_input_errors", []) or []),
        "attention_only_count": int(payload.get("attention_only_count", 0) or 0),
        "attention_only_acknowledged": bool(payload.get("attention_only_acknowledged", False)),
        "attention_only_unacknowledged_count": int(payload.get("attention_only_unacknowledged_count", 0) or 0),
        "validation_jsonl_path": reported_validation_path,
        "validation_jsonl_sha256": reported_validation_sha,
        "current_validation_jsonl_path": current_validation_path,
        "current_validation_jsonl_sha256": current_validation_sha,
        "validation_jsonl_exists": current_validation_exists,
        "validation_jsonl_path_current_match": bool(validation_path_match),
        "validation_jsonl_sha256_current_match": bool(validation_sha_match),
        "validation_jsonl_current_match": bool(validation_path_match and validation_sha_match),
        "final_path": final_path,
        "final_exists": final_exists,
        "final_sha256": final_sha,
        "current_final_sha256": current_final_sha,
        "final_sha256_current_match": bool(final_sha and current_final_sha and final_sha == current_final_sha),
        **merged_row_metrics,
    }


def _merge_rows_lineage_metrics(
    payload: Mapping[str, object],
    *,
    final_path: str,
    final_exists: bool,
    validation_records: Sequence[object] = (),
) -> dict[str, object]:
    errors: list[str] = []
    merged_rows_raw = payload.get("merged_rows")
    merged_rows = list(merged_rows_raw) if isinstance(merged_rows_raw, list) else []
    if not isinstance(merged_rows_raw, list):
        errors.append("merged_rows_missing")

    reported_merged_count = _safe_int(payload.get("merged_row_count", len(merged_rows)))
    reported_row_count = _safe_int(payload.get("row_count", len(merged_rows)))
    if reported_merged_count != len(merged_rows):
        errors.append("merged_row_count_mismatch")
    if reported_row_count != len(merged_rows):
        errors.append("merged_row_count_row_count_mismatch")

    final_lines: list[str] = []
    if final_exists and final_path:
        try:
            final_lines = read_text_with_fallback(final_path).splitlines()
        except Exception:
            errors.append("final_lines_read_failed")
    elif merged_rows:
        errors.append("final_file_missing_for_merged_rows")

    if final_lines and len(final_lines) != len(merged_rows):
        errors.append("merged_row_count_final_line_count_mismatch")

    expected_records = sorted(validation_records, key=lambda item: int(getattr(item, "line_index", 0) or 0))
    if expected_records and len(expected_records) != len(merged_rows):
        errors.append("merged_row_count_validation_record_count_mismatch")

    for idx, item in enumerate(merged_rows):
        if not isinstance(item, Mapping):
            errors.append(f"merged_rows.{idx}.invalid")
            continue
        output_index = _safe_int(item.get("output_index", -1))
        if output_index != idx:
            errors.append(f"merged_rows.{idx}.output_index_mismatch")
        output_source = str(item.get("output_source", "") or "")
        if output_source not in _MERGE_OUTPUT_SOURCES:
            errors.append(f"merged_rows.{idx}.output_source_invalid")
        if idx < len(final_lines):
            expected_hash = _line_sha256(final_lines[idx])
            reported_hash = str(item.get("line_sha256", "") or "").strip().lower()
            if not reported_hash or reported_hash != expected_hash:
                errors.append(f"merged_rows.{idx}.line_sha256_mismatch")
        if idx < len(expected_records):
            record = expected_records[idx]
            _compare_merge_lineage_field(errors, item, "row_id", str(getattr(record, "row_id", "") or ""), idx)
            _compare_merge_lineage_field(errors, item, "line_index", _record_int(record, "line_index"), idx)
            _compare_merge_lineage_field(errors, item, "split", str(getattr(record, "split", "") or ""), idx)
            _compare_merge_lineage_field(errors, item, "wav", str(getattr(record, "wav", "") or ""), idx)
            _compare_merge_lineage_field(errors, item, "alias", str(getattr(record, "alias", "") or ""), idx)
            _compare_merge_lineage_field(
                errors,
                item,
                "occurrence_index",
                _record_int(record, "occurrence_index"),
                idx,
            )
            _compare_merge_lineage_field(errors, item, "slot_index", _record_int(record, "slot_index"), idx)
            _compare_merge_lineage_field(
                errors,
                item,
                "left_slot_index",
                _record_int(record, "left_slot_index"),
                idx,
            )
            _compare_merge_lineage_field(
                errors,
                item,
                "right_slot_index",
                _record_int(record, "right_slot_index"),
                idx,
            )
            _compare_merge_lineage_field(
                errors,
                item,
                "source_row_index",
                _record_int(record, "source_row_index"),
                idx,
            )
            _compare_merge_lineage_field(
                errors,
                item,
                "source_timing_trusted",
                bool(getattr(record, "source_timing_trusted", False)),
                idx,
            )
            split = str(getattr(record, "split", "") or "")
            if split == "clean" and output_source != "clean":
                errors.append(f"merged_rows.{idx}.output_source_split_mismatch")
            if split == "fix_required" and output_source not in {"edited_fix_required", "fix_required_unresolved_original"}:
                errors.append(f"merged_rows.{idx}.output_source_split_mismatch")
            if split == "attention_only" and output_source not in {"edited_attention_only", "attention_only_original"}:
                errors.append(f"merged_rows.{idx}.output_source_split_mismatch")

    return {
        "merged_rows_available": isinstance(merged_rows_raw, list),
        "merged_row_count": len(merged_rows),
        "reported_merged_row_count": reported_merged_count,
        "merged_rows_final_line_count": len(final_lines),
        "merged_rows_validation_record_count": len(expected_records),
        "merged_rows_error_count": len(errors),
        "merged_rows_errors": errors[:20],
        "merged_rows_current_match": len(errors) == 0,
    }


def _record_int(record: object, field: str, default: int = -1) -> int:
    try:
        return int(getattr(record, field))
    except Exception:
        return int(default)


def _compare_merge_lineage_field(
    errors: list[str],
    row: Mapping[str, object],
    field: str,
    expected: object,
    idx: int,
) -> None:
    raw_value = row.get(field)
    if isinstance(expected, bool):
        current = bool(raw_value)
    elif isinstance(expected, int):
        current = _safe_int(raw_value)
    else:
        current = str(raw_value or "")
    if current != expected:
        errors.append(f"merged_rows.{idx}.{field}_mismatch")


def _line_sha256(line: str) -> str:
    return hashlib.sha256(str(line or "").encode("utf-8")).hexdigest()


def _gate_warnings(
    validation_metrics: Mapping[str, object],
    comparison_metrics: Mapping[str, object],
    merge_metrics: Mapping[str, object],
    evidence: str,
    evidence_metrics: Mapping[str, object],
) -> list[str]:
    warnings: list[str] = []
    if evidence == "none":
        warnings.append("manual_or_listening_evidence_missing")
    if int(validation_metrics.get("validation_error_count", 0) or 0) > 0:
        warnings.append("validation_jsonl_invalid")
    if evidence != "none" and not bool(evidence_metrics.get("quality_evidence_report_available", False)):
        warnings.append("quality_evidence_report_missing")
    elif evidence != "none" and not bool(evidence_metrics.get("report_valid", False)):
        warnings.extend(str(item) for item in (evidence_metrics.get("failure_reasons", []) or []))
    if bool(comparison_metrics.get("comparison_json_invalid", False)):
        warnings.append("invalid_comparison_json")
    if not bool(comparison_metrics.get("comparison_available")):
        warnings.append("comparison_json_missing_or_unavailable")
    else:
        if not bool(comparison_metrics.get("validation_jsonl_current_match", False)):
            warnings.append("comparison_validation_jsonl_current_mismatch")
        if not bool(comparison_metrics.get("evidence_jsonl_current_match", True)):
            warnings.append("comparison_evidence_jsonl_current_mismatch")
        if not bool(comparison_metrics.get("evidence_jsonl_rows_valid", True)):
            warnings.append("comparison_evidence_jsonl_rows_invalid")
        if not bool(comparison_metrics.get("evidence_wav_sha256_current_match", True)):
            warnings.append("comparison_evidence_wav_sha256_current_mismatch")
        if not bool(comparison_metrics.get("timeline_jsonl_current_match", True)):
            warnings.append("comparison_timeline_jsonl_current_mismatch")
        if not bool(comparison_metrics.get("oto_current_match", True)):
            warnings.append("comparison_oto_current_mismatch")
        if not bool(comparison_metrics.get("prior_provenance_available", True)):
            warnings.append("comparison_prior_provenance_missing")
        if not bool(comparison_metrics.get("row_param_delta_current_match", True)):
            warnings.append("comparison_row_param_delta_current_mismatch")
            if str(comparison_metrics.get("row_param_delta_recompute_error", "")) == "missing_recompute_source_path":
                warnings.append("comparison_row_param_delta_recompute_source_missing")
        if not bool(comparison_metrics.get("timeline_event_delta_current_match", True)):
            warnings.append("comparison_timeline_event_delta_current_mismatch")
            if str(comparison_metrics.get("timeline_event_delta_recompute_error", "")) == "missing_recompute_source_path":
                warnings.append("comparison_timeline_event_delta_recompute_source_missing")
    if int(comparison_metrics.get("missing_in_candidate_count", 0) or 0) > 0:
        warnings.append("candidate_missing_rows")
    if bool(merge_metrics.get("merge_report_invalid", False)):
        warnings.append("invalid_merge_report")
    if not bool(merge_metrics.get("merge_report_available")):
        warnings.append("merge_report_missing")
    elif not bool(merge_metrics.get("merge_ok")):
        warnings.append("merge_report_not_ok")
    if int(merge_metrics.get("unresolved_count", 0) or 0) > 0:
        warnings.append("merge_unresolved_rows")
    if int(merge_metrics.get("validation_error_count", 0) or 0) > 0:
        warnings.append("merge_validation_errors")
    if int(merge_metrics.get("merge_input_error_count", 0) or 0) > 0:
        warnings.append("merge_input_errors")
    if bool(merge_metrics.get("merge_report_available")) and not bool(
        merge_metrics.get("validation_jsonl_current_match", False)
    ):
        warnings.append("merge_validation_jsonl_current_mismatch")
    if bool(merge_metrics.get("merge_report_available")) and not bool(
        merge_metrics.get("final_sha256_current_match", False)
    ):
        warnings.append("merge_final_sha256_current_mismatch")
    if bool(merge_metrics.get("merge_report_available")) and not bool(
        merge_metrics.get("merged_rows_current_match", False)
    ):
        warnings.append("merge_rows_lineage_current_mismatch")
    if int(merge_metrics.get("attention_only_unacknowledged_count", 0) or 0) > 0:
        warnings.append("attention_only_unacknowledged")
    if int(validation_metrics.get("fix_required_rows", 0) or 0) > 0:
        warnings.append("fix_required_rows_present")
    if int(validation_metrics.get("timing_clamp_large_delta_rows", 0) or 0) > 0:
        warnings.append("timing_clamp_large_delta_rows_present")
    if int(validation_metrics.get("structural_warning_rows", 0) or 0) > 0:
        warnings.append("structural_warning_rows_present")
    if int(validation_metrics.get("local_gate_attention_rows", 0) or 0) > 0:
        warnings.append("local_gate_attention_rows_present")
    if int(validation_metrics.get("rule_based_inference_rows", 0) or 0) > 0:
        warnings.append("rule_based_inference_rows_present")
    if int(validation_metrics.get("rule_based_checkpoint_failed_rows", 0) or 0) > 0:
        warnings.append("rule_based_checkpoint_failed_rows_present")
    if float(validation_metrics.get("attention_only_ratio", 0.0) or 0.0) > 0.5:
        warnings.append("attention_only_ratio_high")
    return warnings


def _load_json_file(path: str) -> dict[str, object]:
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data or {}) if isinstance(data, Mapping) else {}


def _load_review_session_payload(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload, load_error = _load_json_object_file(str(path))
    except Exception as exc:
        return {}, str(exc)
    if load_error:
        return {}, load_error
    if not payload:
        return {}, "empty_review_session"
    return payload, ""


def _load_json_object_file(path: str) -> tuple[dict[str, object], str]:
    if not str(path or "").strip():
        return {}, ""
    target = Path(path)
    if not target.is_file():
        return {}, ""
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        return {}, "json_root_not_object"
    return dict(data or {}), ""


def _json_report_path_exists(path: str) -> bool:
    text = str(path or "").strip()
    return bool(text and Path(text).is_file())


def _recompute_gate_from_report(gate_payload: Mapping[str, object]) -> dict[str, object]:
    validation = str(gate_payload.get("validation_jsonl", "") or "")
    if not validation:
        return {}
    gate_block = dict(gate_payload.get("gate_report", {}) or {})
    return _build_artifact_gate_report(
        validation_jsonl_path=validation,
        comparison_json_path=str(gate_payload.get("comparison_json", "") or ""),
        merge_report_path=str(gate_payload.get("merge_report", "") or ""),
        attention_ratio_max=float(gate_block.get("attention_ratio_max", 0.50) or 0.50),
        quality_evidence=str(gate_payload.get("quality_evidence", "none") or "none"),
        quality_evidence_report_path=str(gate_payload.get("quality_evidence_report", "") or ""),
    )


def _resolve_target_oto_path(path: str) -> Path:
    target = Path(str(path or ""))
    text = str(path or "")
    if target.is_dir() or text.endswith(("/", "\\")):
        return target / "oto.ini"
    return target


def _backup_name(target: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{target.name}.{timestamp}.bak"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _same_path(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except OSError:
        return str(left) == str(right)


def _compare_generate_payloads(
    baseline: Mapping[str, object],
    hsmm: Mapping[str, object],
    *,
    row_delta_limit: int = 100,
) -> dict[str, object]:
    base_counts = dict(baseline.get("split_counts", {}) or {})
    hsmm_counts = dict(hsmm.get("split_counts", {}) or {})
    keys = sorted(set(base_counts) | set(hsmm_counts) | {"total", "fix_required", "attention_only", "clean"})
    split_delta = {
        key: int(hsmm_counts.get(key, 0) or 0) - int(base_counts.get(key, 0) or 0)
        for key in keys
    }
    return {
        "baseline_ok": bool(baseline.get("ok")),
        "hsmm_ok": bool(hsmm.get("ok")),
        "baseline_rows": int(baseline.get("processed", 0) or 0),
        "hsmm_rows": int(hsmm.get("processed", 0) or 0),
        "row_count_delta": int(hsmm.get("processed", 0) or 0) - int(baseline.get("processed", 0) or 0),
        "split_count_delta": split_delta,
        "fix_required_delta": split_delta.get("fix_required", 0),
        "attention_only_delta": split_delta.get("attention_only", 0),
        "clean_delta": split_delta.get("clean", 0),
        "hsmm_warning_count": len(list(hsmm.get("warnings", []) or [])),
        "baseline_warning_count": len(list(baseline.get("warnings", []) or [])),
        "hsmm_decoder_used": any("filename_hsmm_decoder_used" in str(item) for item in (hsmm.get("warnings", []) or [])),
        "row_param_delta": _compare_generated_oto_params(
            baseline.get("generated_oto_path", ""),
            hsmm.get("generated_oto_path", ""),
            baseline_validation_path=_validation_jsonl_path(baseline),
            hsmm_validation_path=_validation_jsonl_path(hsmm),
            row_delta_limit=row_delta_limit,
        ),
        "timeline_event_delta": _compare_timeline_events(
            baseline.get("timeline_path", ""),
            hsmm.get("timeline_path", ""),
            row_delta_limit=row_delta_limit,
        ),
        "hsmm_diagnostics": _summarize_hsmm_timeline_diagnostics(
            hsmm.get("timeline_path", ""),
            row_delta_limit=row_delta_limit,
        ),
    }


def _validation_jsonl_path(payload: Mapping[str, object]) -> str:
    paths = payload.get("split_output_paths", {}) or {}
    if isinstance(paths, Mapping):
        return str(paths.get("validation_jsonl", "") or "")
    return ""


def _compare_timeline_events(
    baseline_timeline_path: object,
    hsmm_timeline_path: object,
    *,
    row_delta_limit: int = 100,
) -> dict[str, object]:
    base_path = Path(str(baseline_timeline_path or ""))
    hsmm_path = Path(str(hsmm_timeline_path or ""))
    base_sha = _sha256_file(base_path).strip().lower() if base_path.is_file() else ""
    hsmm_sha = _sha256_file(hsmm_path).strip().lower() if hsmm_path.is_file() else ""
    if not base_path.is_file() or not hsmm_path.is_file():
        return {
            "available": False,
            "reason": "missing_timeline_path",
            "baseline_timeline_path": str(base_path) if str(base_path) != "." else "",
            "baseline_timeline_sha256": base_sha,
            "hsmm_timeline_path": str(hsmm_path) if str(hsmm_path) != "." else "",
            "hsmm_timeline_sha256": hsmm_sha,
        }

    base_records = _load_timeline_records(base_path)
    hsmm_records = _load_timeline_records(hsmm_path)
    all_wavs = sorted(set(base_records) | set(hsmm_records))
    missing_in_baseline: list[str] = []
    missing_in_hsmm: list[str] = []
    event_deltas: list[dict[str, object]] = []
    matched_events = 0

    for wav in all_wavs:
        base = base_records.get(wav)
        hsmm_record = hsmm_records.get(wav)
        if base is None:
            missing_in_baseline.append(wav)
            continue
        if hsmm_record is None:
            missing_in_hsmm.append(wav)
            continue
        base_events = list(base.get("selected_events", []) or [])
        hsmm_events = list(hsmm_record.get("selected_events", []) or [])
        max_count = max(len(base_events), len(hsmm_events))
        for idx in range(max_count):
            base_event = base_events[idx] if idx < len(base_events) else None
            hsmm_event = hsmm_events[idx] if idx < len(hsmm_events) else None
            if base_event is None:
                event_deltas.append(
                    {
                        "wav": wav,
                        "event_index": idx,
                        "missing_side": "baseline",
                        "hsmm_event": _event_summary(hsmm_event),
                    }
                )
                continue
            if hsmm_event is None:
                event_deltas.append(
                    {
                        "wav": wav,
                        "event_index": idx,
                        "missing_side": "hsmm",
                        "baseline_event": _event_summary(base_event),
                    }
                )
                continue
            matched_events += 1
            base_time = float(dict(base_event).get("selected_time_ms", 0.0) or 0.0)
            hsmm_time = float(dict(hsmm_event).get("selected_time_ms", 0.0) or 0.0)
            delta = hsmm_time - base_time
            label_changed = str(dict(base_event).get("label", "")) != str(dict(hsmm_event).get("label", ""))
            phone_changed = str(dict(base_event).get("expected_phone", "")) != str(dict(hsmm_event).get("expected_phone", ""))
            if abs(delta) > 1e-6 or label_changed or phone_changed:
                event_deltas.append(
                    {
                        "wav": wav,
                        "event_index": idx,
                        "delta_ms": _round_ms(delta),
                        "abs_delta_ms": _round_ms(abs(delta)),
                        "label_changed": label_changed,
                        "expected_phone_changed": phone_changed,
                        "baseline_event": _event_summary(base_event),
                        "hsmm_event": _event_summary(hsmm_event),
                        "baseline_event_source": str(base.get("selected_event_source", "")),
                        "hsmm_event_source": str(hsmm_record.get("selected_event_source", "")),
                    }
                )

    event_deltas.sort(
        key=lambda item: (
            -float(item.get("abs_delta_ms", 0.0) or 0.0),
            str(item.get("wav", "")),
            int(item.get("event_index", 10**9) or 10**9),
        )
    )
    included = event_deltas[:row_delta_limit] if row_delta_limit > 0 else event_deltas
    return {
        "available": True,
        "baseline_timeline_path": str(base_path),
        "baseline_timeline_sha256": base_sha,
        "hsmm_timeline_path": str(hsmm_path),
        "hsmm_timeline_sha256": hsmm_sha,
        "baseline_wavs": len(base_records),
        "hsmm_wavs": len(hsmm_records),
        "wavs_compared": len(all_wavs) - len(missing_in_baseline) - len(missing_in_hsmm),
        "missing_in_baseline": missing_in_baseline[:row_delta_limit] if row_delta_limit > 0 else missing_in_baseline,
        "missing_in_hsmm": missing_in_hsmm[:row_delta_limit] if row_delta_limit > 0 else missing_in_hsmm,
        "missing_in_baseline_count": len(missing_in_baseline),
        "missing_in_hsmm_count": len(missing_in_hsmm),
        "matched_events": matched_events,
        "changed_events": len(event_deltas),
        "row_delta_limit": row_delta_limit,
        "event_delta_rows_total": len(event_deltas),
        "event_delta_rows_included": len(included),
        "largest_events": included,
    }


def _load_timeline_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            data = json.loads(raw)
            wav = str(data.get("wav", "") or "")
            if wav:
                records[wav] = data
    return records


def _summarize_hsmm_timeline_diagnostics(
    hsmm_timeline_path: object,
    *,
    row_delta_limit: int = 100,
) -> dict[str, object]:
    path = Path(str(hsmm_timeline_path or ""))
    if not path.is_file():
        return {
            "available": False,
            "reason": "missing_timeline_path",
            "hsmm_timeline_path": str(path) if str(path) != "." else "",
        }
    records = _load_timeline_records(path)
    state_rows: list[dict[str, object]] = []
    hsmm_record_count = 0
    hsmm_used_wavs = 0
    total_state_count = 0
    total_event_prior_state_count = 0
    total_event_prior_input_count = 0
    total_event_prior_mapped_count = 0
    total_event_prior_ignored_count = 0
    total_candidate_prior_count = 0
    total_runtime_prior_count = 0
    total_event_prior_source_counts: dict[str, int] = {}
    total_event_prior_label_counts: dict[str, int] = {}
    min_best_margin: float | None = None
    min_second_margin: float | None = None
    max_duration_z: float = 0.0
    for wav, record in records.items():
        hsmm = record.get("hsmm")
        if not isinstance(hsmm, Mapping):
            continue
        diagnostics = hsmm.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        hsmm_record_count += 1
        if str(record.get("selected_event_source", "")) == "filename_hsmm":
            hsmm_used_wavs += 1
        total_state_count += int(diagnostics.get("state_count", 0) or 0)
        total_event_prior_state_count += int(diagnostics.get("event_prior_state_count", 0) or 0)
        total_event_prior_input_count += int(diagnostics.get("event_prior_input_count", 0) or 0)
        total_event_prior_mapped_count += int(diagnostics.get("event_prior_mapped_count", 0) or 0)
        total_event_prior_ignored_count += int(diagnostics.get("event_prior_ignored_count", 0) or 0)
        total_candidate_prior_count += int(diagnostics.get("candidate_prior_count", 0) or 0)
        total_runtime_prior_count += int(diagnostics.get("runtime_prior_count", 0) or 0)
        _merge_count_map(total_event_prior_source_counts, diagnostics.get("event_prior_source_counts", {}))
        _merge_count_map(total_event_prior_label_counts, diagnostics.get("event_prior_label_counts", {}))
        best_margin = _float_or_none(diagnostics.get("min_selected_vs_best_local_margin"))
        second_margin = _float_or_none(diagnostics.get("min_selected_vs_second_local_margin"))
        duration_z = _float_or_none(diagnostics.get("max_duration_z_abs"))
        if best_margin is not None:
            min_best_margin = best_margin if min_best_margin is None else min(min_best_margin, best_margin)
        if second_margin is not None:
            min_second_margin = second_margin if min_second_margin is None else min(min_second_margin, second_margin)
        if duration_z is not None:
            max_duration_z = max(max_duration_z, duration_z)
        for state in list(diagnostics.get("states", []) or []):
            state_data = dict(state or {})
            state_rows.append(
                {
                    "wav": wav,
                    "state_id": str(state_data.get("state_id", "")),
                    "state_type": str(state_data.get("state_type", "")),
                    "selected_vs_best_local_margin": round(float(state_data.get("selected_vs_best_local_margin", 0.0) or 0.0), 6),
                    "selected_vs_second_local_margin": round(float(state_data.get("selected_vs_second_local_margin", 0.0) or 0.0), 6),
                    "duration_z_abs": round(float(state_data.get("duration_z_abs", 0.0) or 0.0), 6),
                    "event_prior_selected_mean": round(float(state_data.get("event_prior_selected_mean", 0.0) or 0.0), 6),
                    "has_event_prior": bool(state_data.get("has_event_prior")),
                    "event_prior_mapped_count": int(state_data.get("event_prior_mapped_count", 0) or 0),
                    "evidence_candidate_prior_count": int(state_data.get("evidence_candidate_prior_count", 0) or 0),
                    "runtime_prior_count": int(state_data.get("runtime_prior_count", 0) or 0),
                    "event_prior_source_counts": dict(state_data.get("event_prior_source_counts", {}) or {}),
                    "event_prior_label_counts": dict(state_data.get("event_prior_label_counts", {}) or {}),
                }
            )
    state_rows.sort(
        key=lambda item: (
            float(item.get("selected_vs_best_local_margin", 0.0) or 0.0),
            float(item.get("selected_vs_second_local_margin", 0.0) or 0.0),
            -float(item.get("duration_z_abs", 0.0) or 0.0),
            str(item.get("wav", "")),
            str(item.get("state_id", "")),
        )
    )
    included = state_rows[:row_delta_limit] if row_delta_limit > 0 else state_rows
    return {
        "available": True,
        "hsmm_timeline_path": str(path),
        "timeline_wavs": len(records),
        "hsmm_record_count": hsmm_record_count,
        "hsmm_used_wavs": hsmm_used_wavs,
        "total_state_count": total_state_count,
        "event_prior_state_count": total_event_prior_state_count,
        "event_prior_input_count": total_event_prior_input_count,
        "event_prior_mapped_count": total_event_prior_mapped_count,
        "event_prior_ignored_count": total_event_prior_ignored_count,
        "event_prior_source_counts": dict(sorted(total_event_prior_source_counts.items())),
        "event_prior_label_counts": dict(sorted(total_event_prior_label_counts.items())),
        "candidate_prior_count": total_candidate_prior_count,
        "runtime_prior_count": total_runtime_prior_count,
        "min_selected_vs_best_local_margin": round(float(min_best_margin or 0.0), 6),
        "min_selected_vs_second_local_margin": round(float(min_second_margin or 0.0), 6),
        "max_duration_z_abs": round(float(max_duration_z), 6),
        "state_rows_total": len(state_rows),
        "state_rows_included": len(included),
        "lowest_margin_states": included,
    }


def _merge_count_map(target: dict[str, int], source: object) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        try:
            count = int(value or 0)
        except Exception:
            continue
        if count:
            target[str(key)] = int(target.get(str(key), 0) or 0) + count


def _event_summary(event: object) -> dict[str, object]:
    data = dict(event or {})
    return {
        "label": str(data.get("label", "")),
        "selected_time_ms": _round_ms(float(data.get("selected_time_ms", 0.0) or 0.0)),
        "score": round(float(data.get("score", 0.0) or 0.0), 6),
        "expected_phone": str(data.get("expected_phone", "") or ""),
        "expected_phone_index": data.get("expected_phone_index"),
        "slot_index": data.get("slot_index"),
        "frame_index": data.get("frame_index"),
        "source": str(data.get("source", "") or ""),
    }


def _replay_timeline_record(
    *,
    wav: str,
    evidence: Mapping[str, object],
    expected_phones: tuple[str, ...] | list[str],
    slots: object,
    runtime_events: object,
    selected_events: object,
    diagnostics: Mapping[str, object],
    meta: Mapping[str, object],
    ok: bool,
    skipped: bool,
    reason: str,
    candidate_priors: object = (),
    adapted_rows: object = (),
) -> dict[str, object]:
    hsmm_block: dict[str, object] = {
        "ok": bool(ok),
        "skipped": bool(skipped),
        "reason": str(reason),
        "meta": dict(meta or {}),
    }
    if diagnostics:
        hsmm_block["diagnostics"] = dict(diagnostics or {})
    return {
        "schema_version": 1,
        "wav": wav,
        "source": "evidence_replay",
        "selected_event_source": "filename_hsmm" if ok else "evidence_replay_skipped" if skipped else "evidence_replay_failed",
        "expected_phones": [str(phone) for phone in list(expected_phones or ())],
        "filename_slots": [_filename_slot_summary(slot) for slot in list(slots or ())],
        "runtime_events": [_event_summary(event) for event in list(runtime_events or ())],
        "candidate_priors": [_event_summary(event) for event in list(candidate_priors or ())],
        "selected_events": [_event_summary(event) for event in list(selected_events or ())],
        "adapted_rows": list(adapted_rows or ()),
        "evidence": {
            "schema_version": evidence.get("schema_version", ""),
            "wav_sha256": evidence.get("wav_sha256", ""),
            "extractor_version": evidence.get("extractor_version", ""),
            "decoder_version": evidence.get("decoder_version", ""),
            "timebase": dict(evidence.get("timebase", {}) or {}) if isinstance(evidence.get("timebase"), Mapping) else {},
            "reliability": dict(evidence.get("reliability", {}) or {}) if isinstance(evidence.get("reliability"), Mapping) else {},
            "event_candidate_count": len(list(evidence.get("event_candidates", []) or [])),
            "event_candidate_summary": summarize_event_candidate_reliability(evidence),
        },
        "hsmm": hsmm_block,
    }


def _assemble_replay_oto_rows(
    *,
    wav: str,
    evidence: Mapping[str, object],
    posterior: object,
    selected_events: object,
    expected_phones: tuple[str, ...],
    language: str,
    format_type: str,
    alias_type: str,
) -> list[dict[str, object]]:
    template_rows, row_plan_phones, row_plan_records = build_filename_template_rows(
        wav,
        language=language,
        format_type=format_type,
    )
    if not template_rows:
        return []
    phones = tuple(expected_phones or row_plan_phones)
    duration_ms = _evidence_duration_ms(evidence, posterior)
    anchors = assign_template_row_anchors(
        posterior,
        list(selected_events or ()),
        template_rows,
        min_score=0.02,
        use_source_timing_prior=False,
        expected_phones=phones,
    )
    config = OtoAdapterConfig(
        mode="template-preserve",
        language=language,
        format_type=format_type,
        alias_type=alias_type,
    )
    rows: list[dict[str, object]] = []
    for row_index, (template_row, anchor) in enumerate(zip(template_rows, anchors)):
        adapted = adapt_template_row(
            template_row,
            anchor,
            file_duration_ms=duration_ms,
            config=_adapter_config_for_row_plan_record(
                config,
                row_plan_records[row_index] if row_index < len(row_plan_records) else None,
            ),
        )
        payload = adapted.to_json_dict()
        payload["row_index"] = row_index
        payload["raw_line"] = adapted.format_line()
        if row_index < len(row_plan_records):
            payload["row_plan"] = row_plan_records[row_index].to_json_dict()
        rows.append(payload)
    return rows


def _adapter_config_for_row_plan_record(
    config: OtoAdapterConfig,
    row_plan_record: object | None,
) -> OtoAdapterConfig:
    role = str(getattr(row_plan_record, "role_family", "") or "").strip().lower()
    if role in {"cv", "cv_head", "vcv", "vc", "vv", "v"} and role != str(config.alias_type or "").strip().lower():
        return replace(config, alias_type=role)
    return config


def _evidence_duration_ms(evidence: Mapping[str, object], posterior: object) -> float:
    timebase = evidence.get("timebase")
    if isinstance(timebase, Mapping):
        duration = _float_or_none(timebase.get("duration_ms"))
        if duration is not None and duration > 0.0:
            return float(duration)
    times = list(getattr(posterior, "times_ms", ()) or ())
    if times:
        return max(1.0, float(max(times)))
    return 1.0


def _filename_slot_summary(slot: object) -> dict[str, object]:
    return {
        "slot_index": getattr(slot, "slot_index", None),
        "token": str(getattr(slot, "token", "") or ""),
        "vowel": str(getattr(slot, "vowel", "") or ""),
        "phone_start_index": getattr(slot, "phone_start_index", None),
        "phone_end_index": getattr(slot, "phone_end_index", None),
        "event_label": str(getattr(slot, "event_label", "") or ""),
        "onset_phones": [str(item) for item in list(getattr(slot, "onset_phones", ()) or ())],
    }


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _compare_generated_oto_params(
    baseline_oto_path: object,
    hsmm_oto_path: object,
    *,
    baseline_validation_path: str = "",
    hsmm_validation_path: str = "",
    row_delta_limit: int = 100,
) -> dict[str, object]:
    base_path = Path(str(baseline_oto_path or ""))
    hsmm_path = Path(str(hsmm_oto_path or ""))
    baseline_oto_sha = _sha256_file(base_path).strip().lower() if base_path.is_file() else ""
    hsmm_oto_sha = _sha256_file(hsmm_path).strip().lower() if hsmm_path.is_file() else ""
    baseline_validation_file = Path(str(baseline_validation_path or ""))
    hsmm_validation_file = Path(str(hsmm_validation_path or ""))
    baseline_validation_sha = (
        _sha256_file(baseline_validation_file).strip().lower() if baseline_validation_file.is_file() else ""
    )
    hsmm_validation_sha = _sha256_file(hsmm_validation_file).strip().lower() if hsmm_validation_file.is_file() else ""
    if not base_path.is_file() or not hsmm_path.is_file():
        return {
            "available": False,
            "reason": "missing_generated_oto_path",
            "baseline_oto_path": str(base_path) if str(base_path) != "." else "",
            "baseline_oto_sha256": baseline_oto_sha,
            "hsmm_oto_path": str(hsmm_path) if str(hsmm_path) != "." else "",
            "hsmm_oto_sha256": hsmm_oto_sha,
            "baseline_validation_path": str(baseline_validation_file) if str(baseline_validation_file) != "." else "",
            "baseline_validation_sha256": baseline_validation_sha,
            "hsmm_validation_path": str(hsmm_validation_file) if str(hsmm_validation_file) != "." else "",
            "hsmm_validation_sha256": hsmm_validation_sha,
        }

    base_rows, base_invalid = _load_oto_param_rows(base_path)
    hsmm_rows, hsmm_invalid = _load_oto_param_rows(hsmm_path)
    base_validation = _load_validation_map(baseline_validation_path)
    hsmm_validation = _load_validation_map(hsmm_validation_path)

    all_keys = sorted(
        set(base_rows) | set(hsmm_rows),
        key=lambda key: (
            min(
                int(base_rows.get(key, {}).get("line_index", 10**9)),
                int(hsmm_rows.get(key, {}).get("line_index", 10**9)),
            ),
            key[0],
            key[1],
            key[2],
        ),
    )
    param_abs_sums = {name: 0.0 for name, _ in _OTO_PARAM_FIELDS}
    param_max = {
        name: {"value_ms": 0.0, "wav": "", "alias": "", "occurrence_index": 0}
        for name, _ in _OTO_PARAM_FIELDS
    }
    matched_rows = 0
    changed_rows = 0
    split_changed_rows = 0
    missing_in_baseline: list[dict[str, object]] = []
    missing_in_hsmm: list[dict[str, object]] = []
    row_deltas: list[dict[str, object]] = []

    for key in all_keys:
        base = base_rows.get(key)
        hsmm_row = hsmm_rows.get(key)
        if base is None:
            missing_in_baseline.append(_row_key_dict(key, hsmm_row))
            continue
        if hsmm_row is None:
            missing_in_hsmm.append(_row_key_dict(key, base))
            continue

        matched_rows += 1
        params: dict[str, float] = {}
        max_abs_delta = 0.0
        for param_name, _ in _OTO_PARAM_FIELDS:
            base_value = float(dict(base["params"])[param_name])
            hsmm_value = float(dict(hsmm_row["params"])[param_name])
            delta = hsmm_value - base_value
            abs_delta = abs(delta)
            params[param_name] = _round_ms(delta)
            param_abs_sums[param_name] += abs_delta
            if abs_delta > float(param_max[param_name]["value_ms"]):
                param_max[param_name] = {
                    "value_ms": _round_ms(abs_delta),
                    "wav": str(key[0]),
                    "alias": str(key[1]),
                    "occurrence_index": int(key[2]),
                }
            max_abs_delta = max(max_abs_delta, abs_delta)

        base_status = base_validation.get(key, {})
        hsmm_status = hsmm_validation.get(key, {})
        base_split = str(base_status.get("split", ""))
        hsmm_split = str(hsmm_status.get("split", ""))
        split_changed = bool(base_split or hsmm_split) and base_split != hsmm_split
        if max_abs_delta > 1e-6:
            changed_rows += 1
        if split_changed:
            split_changed_rows += 1
        if max_abs_delta > 1e-6 or split_changed:
            row_deltas.append(
                {
                    "wav": str(key[0]),
                    "alias": str(key[1]),
                    "occurrence_index": int(key[2]),
                    "baseline_line_index": int(base["line_index"]),
                    "hsmm_line_index": int(hsmm_row["line_index"]),
                    "max_abs_delta_ms": _round_ms(max_abs_delta),
                    "deltas_ms": params,
                    "baseline_params_ms": dict(base["params"]),
                    "hsmm_params_ms": dict(hsmm_row["params"]),
                    "baseline_split": base_split,
                    "hsmm_split": hsmm_split,
                    "split_changed": split_changed,
                    "baseline_reasons": list(base_status.get("reasons", []) or []),
                    "hsmm_reasons": list(hsmm_status.get("reasons", []) or []),
                    "role_family": str(hsmm_status.get("role_family") or base_status.get("role_family") or ""),
                }
            )

    row_deltas.sort(
        key=lambda item: (
            -float(item.get("max_abs_delta_ms", 0.0) or 0.0),
            str(item.get("wav", "")),
            int(item.get("baseline_line_index", 10**9) or 10**9),
        )
    )
    if row_delta_limit > 0:
        included_rows = row_deltas[:row_delta_limit]
    else:
        included_rows = row_deltas

    mean_abs_delta = {
        name: _round_ms(param_abs_sums[name] / matched_rows) if matched_rows else 0.0
        for name, _ in _OTO_PARAM_FIELDS
    }
    return {
        "available": True,
        "baseline_oto_path": str(base_path),
        "baseline_oto_sha256": baseline_oto_sha,
        "hsmm_oto_path": str(hsmm_path),
        "hsmm_oto_sha256": hsmm_oto_sha,
        "baseline_validation_path": str(baseline_validation_file) if str(baseline_validation_file) != "." else "",
        "baseline_validation_sha256": baseline_validation_sha,
        "hsmm_validation_path": str(hsmm_validation_file) if str(hsmm_validation_file) != "." else "",
        "hsmm_validation_sha256": hsmm_validation_sha,
        "baseline_invalid_rows": base_invalid,
        "hsmm_invalid_rows": hsmm_invalid,
        "baseline_rows": len(base_rows),
        "hsmm_rows": len(hsmm_rows),
        "matched_rows": matched_rows,
        "missing_in_baseline_count": len(missing_in_baseline),
        "missing_in_hsmm_count": len(missing_in_hsmm),
        "missing_in_baseline": missing_in_baseline[:row_delta_limit] if row_delta_limit > 0 else missing_in_baseline,
        "missing_in_hsmm": missing_in_hsmm[:row_delta_limit] if row_delta_limit > 0 else missing_in_hsmm,
        "changed_rows": changed_rows,
        "split_changed_rows": split_changed_rows,
        "mean_abs_delta_ms": mean_abs_delta,
        "max_abs_delta_ms": param_max,
        "row_delta_limit": row_delta_limit,
        "row_delta_rows_total": len(row_deltas),
        "row_delta_rows_included": len(included_rows),
        "largest_rows": included_rows,
    }


def _reference_comparison_failures(reference_comparison: Mapping[str, object]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for name, result in reference_comparison.items():
        if not isinstance(result, Mapping):
            continue
        if bool(result.get("available", False)):
            continue
        failures.append(
            {
                "comparison": str(name),
                "reason": str(result.get("reason", "unavailable")),
            }
        )
    return failures


def _load_oto_param_rows(path: Path) -> tuple[dict[tuple[str, str, int], dict[str, object]], int]:
    rows: dict[tuple[str, str, int], dict[str, object]] = {}
    occurrence: dict[tuple[str, str], int] = {}
    invalid = 0
    for line_index, raw in enumerate(read_text_with_fallback(str(path)).splitlines()):
        text = str(raw or "").rstrip("\r\n")
        if not text.strip():
            continue
        parsed = parse_oto_line(text)
        if parsed is None:
            invalid += 1
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence_index = occurrence.get(base_key, 0)
        occurrence[base_key] = occurrence_index + 1
        key = (wav, alias, occurrence_index)
        rows[key] = {
            "line_index": line_index,
            "raw_line": text,
            "params": {
                param_name: _round_ms(float(parsed[field_name]))
                for param_name, field_name in _OTO_PARAM_FIELDS
            },
        }
    return rows, invalid


def _load_validation_map(path: str) -> dict[tuple[str, str, int], dict[str, object]]:
    records: dict[tuple[str, str, int], dict[str, object]] = {}
    for record in load_validation_records(str(path or "")):
        key = (record.wav, record.alias, int(record.occurrence_index))
        records[key] = {
            "row_id": record.row_id,
            "split": record.split,
            "reasons": list(record.reasons),
            "role_family": record.role_family,
        }
    return records


def _row_key_dict(key: tuple[str, str, int], row: Mapping[str, object] | None) -> dict[str, object]:
    return {
        "wav": str(key[0]),
        "alias": str(key[1]),
        "occurrence_index": int(key[2]),
        "line_index": int(row.get("line_index", -1)) if row is not None else -1,
    }


def _round_ms(value: float) -> float:
    return round(float(value), 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MFA-free OTO review helper. Splits a generated OTO draft into review files "
            "and merges edited review files into a guarded final.ini candidate."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Split generated.ini into fix/attention/clean review files.")
    split_parser.add_argument("--generated", required=True, help="Generated OTO draft path. This is not replaced.")
    split_parser.add_argument("--out-dir", required=True, help="Output directory for review files.")
    split_parser.add_argument("--name", default="", help="Output file stem. Defaults to generated file stem.")
    split_parser.add_argument("--language", default="", help="Language hint for OTO encoding, e.g. japanese.")
    split_parser.add_argument("--wav-root", default="", help="Voicebank wav root used to read missing durations.")
    split_parser.add_argument(
        "--durations-json",
        default="",
        help="Optional JSON object mapping wav file name to duration_ms for deterministic review.",
    )
    split_parser.add_argument(
        "--timeline-jsonl",
        default="",
        help="Optional timeline.jsonl sidecar used to add HSMM diagnostics to validation reasons.",
    )
    split_parser.set_defaults(func=_cmd_split)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a source-timing-free OTO draft from WAVs, then emit review split files.",
    )
    generate_parser.add_argument("--wav-dir", required=True, help="Directory containing WAV files.")
    generate_parser.add_argument("--out-dir", required=True, help="Output directory for generated/review files.")
    generate_parser.add_argument("--name", default="mfa_free", help="Output file stem.")
    generate_parser.add_argument(
        "--template-oto",
        default="",
        help="Optional OTO template used for wav+alias row order only. Timing values are ignored.",
    )
    generate_parser.add_argument("--checkpoint", default="", help="Optional MFA-free checkpoint path.")
    generate_parser.add_argument("--language", default="japanese")
    generate_parser.add_argument("--format-type", default="CV")
    generate_parser.add_argument("--alias-type", default="auto")
    generate_parser.add_argument("--alias-suffix", default="")
    generate_parser.add_argument("--encoder", default="acoustic", help="Feature encoder. Default keeps MVP dependency-light.")
    generate_parser.add_argument("--device", default="", help="Torch device when a checkpoint is used.")
    generate_parser.add_argument("--disable-slot-viterbi", action="store_true")
    generate_parser.add_argument("--use-hsmm-decoder", action="store_true", help="Use filename row-plan HSMM events for no-template generation.")
    generate_parser.set_defaults(func=_cmd_generate)

    compare_parser = subparsers.add_parser(
        "compare-hsmm",
        help="Run baseline and HSMM no-template generation, then compare split, row, and timeline diagnostics.",
    )
    compare_parser.add_argument("--wav-dir", required=True, help="Directory containing WAV files.")
    compare_parser.add_argument("--out-dir", required=True, help="Output directory for comparison artifacts.")
    compare_parser.add_argument("--name", default="mfa_free", help="Output file stem.")
    compare_parser.add_argument(
        "--template-oto",
        default="",
        help="Optional OTO template used for wav+alias row order only. Timing values are ignored.",
    )
    compare_parser.add_argument("--checkpoint", default="", help="Optional MFA-free checkpoint path.")
    compare_parser.add_argument("--language", default="japanese")
    compare_parser.add_argument("--format-type", default="CV")
    compare_parser.add_argument("--alias-type", default="auto")
    compare_parser.add_argument("--alias-suffix", default="")
    compare_parser.add_argument("--encoder", default="acoustic", help="Feature encoder. Default keeps MVP dependency-light.")
    compare_parser.add_argument("--device", default="", help="Torch device when a checkpoint is used.")
    compare_parser.add_argument("--disable-slot-viterbi", action="store_true")
    compare_parser.add_argument(
        "--row-delta-limit",
        type=int,
        default=100,
        help="Maximum changed/missing row records embedded in the comparison JSON. Use 0 for all rows.",
    )
    compare_parser.set_defaults(func=_cmd_compare_hsmm)

    replay_parser = subparsers.add_parser(
        "replay-evidence-hsmm",
        help="Replay filename-slot HSMM decoding from a generated evidence.jsonl sidecar.",
    )
    replay_parser.add_argument("--evidence-jsonl", required=True, help="Evidence sidecar emitted by generate/compare-hsmm.")
    replay_parser.add_argument("--out", default="", help="Optional JSON report path.")
    replay_parser.add_argument("--timeline-out", default="", help="Optional timeline.jsonl replay sidecar path.")
    replay_parser.add_argument("--assembled-oto-out", default="", help="Optional diagnostic OTO draft assembled from replayed HSMM events.")
    replay_parser.add_argument("--split-out-dir", default="", help="Optional review split directory for --assembled-oto-out.")
    replay_parser.add_argument("--split-name", default="", help="Optional review split stem for --split-out-dir.")
    replay_parser.add_argument("--wav-root", default="", help="Optional wav root used to read durations during split validation.")
    replay_parser.add_argument(
        "--durations-json",
        default="",
        help="Optional JSON object mapping wav file name to duration_ms for deterministic split validation.",
    )
    replay_parser.add_argument("--reference-generated", default="", help="Optional generated.ini to compare against assembled replay OTO.")
    replay_parser.add_argument(
        "--reference-validation-jsonl",
        default="",
        help="Optional validation.jsonl for --reference-generated. Requires --split-out-dir for replay split comparison.",
    )
    replay_parser.add_argument("--reference-timeline", default="", help="Optional timeline.jsonl to compare against replay timeline.")
    replay_parser.add_argument(
        "--row-delta-limit",
        type=int,
        default=100,
        help="Maximum changed/missing row records embedded in reference comparisons. Use 0 for all rows.",
    )
    replay_parser.add_argument("--language", default="japanese")
    replay_parser.add_argument("--format-type", default="CV")
    replay_parser.add_argument("--alias-type", default="auto")
    replay_parser.add_argument("--alias-suffix", default="")
    replay_parser.add_argument("--beam-width", type=int, default=64)
    replay_parser.add_argument("--timeout-ms", type=int, default=5000)
    replay_parser.set_defaults(func=_cmd_replay_evidence_hsmm)

    merge_parser = subparsers.add_parser("merge", help="Merge clean rows and edited review rows into final.ini.")
    merge_parser.add_argument("--validation-jsonl", required=True, help="validation.jsonl emitted by split.")
    merge_parser.add_argument("--final", required=True, help="Output final.ini candidate path.")
    merge_parser.add_argument("--edited-fix-required", default="", help="Edited fix_required.ini path.")
    merge_parser.add_argument("--edited-attention-only", default="", help="Edited attention_only.ini path.")
    merge_parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write final.ini even if fix_required rows were not changed. Off by default.",
    )
    merge_parser.add_argument(
        "--acknowledge-attention-only",
        action="store_true",
        help="Mark unchanged attention_only rows as manually reviewed/accepted in the merge report.",
    )
    merge_parser.add_argument("--language", default="", help="Language hint for OTO encoding, e.g. japanese.")
    merge_parser.add_argument("--preferred-encoding-source", default="", help="Source file for output encoding detection.")
    merge_parser.set_defaults(func=_cmd_merge)

    merge_session_parser = subparsers.add_parser(
        "merge-session",
        help="Merge a review session using paths recorded in review_session.json.",
    )
    merge_session_parser.add_argument("--review-session", required=True, help="review_session.json emitted by split/generate/replay.")
    merge_session_parser.add_argument("--final", default="", help="Optional final.ini candidate path override.")
    merge_session_parser.add_argument("--out", default="", help="Optional JSON merge-session report path.")
    merge_session_parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write final.ini even if fix_required rows were not changed. Off by default.",
    )
    merge_session_parser.add_argument(
        "--acknowledge-attention-only",
        action="store_true",
        help="Mark unchanged attention_only rows as manually reviewed/accepted in the merge report.",
    )
    merge_session_parser.add_argument("--language", default="", help="Language hint for OTO encoding, e.g. japanese.")
    merge_session_parser.add_argument("--preferred-encoding-source", default="", help="Source file for output encoding detection.")
    merge_session_parser.set_defaults(func=_cmd_merge_session)

    gate_parser = subparsers.add_parser(
        "evaluate-gate",
        help="Evaluate MFA-free OTO review artifacts for apply/default-on gate eligibility.",
    )
    gate_parser.add_argument("--validation-jsonl", required=True, help="validation.jsonl emitted by split/generate/replay split.")
    gate_parser.add_argument("--comparison-json", default="", help="Optional compare-hsmm or replay-evidence-hsmm JSON report.")
    gate_parser.add_argument("--merge-report", default="", help="Optional final.ini merge report JSON.")
    gate_parser.add_argument("--out", default="", help="Optional JSON gate report path.")
    gate_parser.add_argument(
        "--quality-evidence",
        default="none",
        choices=("none", "manual-reference", "listening", "manual-reference-and-listening"),
        help="Expected external quality evidence type for default-on consideration. Default stays fail-closed.",
    )
    gate_parser.add_argument(
        "--quality-evidence-report",
        default="",
        help=(
            "Optional JSON report proving manual/listening evidence for this final.ini. "
            "Default-on requires a valid report whose final_sha256 matches the merge report."
        ),
    )
    gate_parser.add_argument(
        "--attention-ratio-max",
        type=float,
        default=0.50,
        help="Maximum attention_only ratio allowed for default-on candidate status.",
    )
    gate_parser.add_argument(
        "--gate-target",
        default="default-on",
        choices=("apply", "default-on"),
        help="Gate used with --fail-on-gate-failure.",
    )
    gate_parser.add_argument(
        "--fail-on-gate-failure",
        action="store_true",
        help="Return exit code 2 when the selected gate target does not pass.",
    )
    gate_parser.set_defaults(func=_cmd_evaluate_gate)

    session_gate_parser = subparsers.add_parser(
        "evaluate-session",
        help="Evaluate a review session gate using paths recorded in review_session.json.",
    )
    session_gate_parser.add_argument("--review-session", required=True, help="review_session.json emitted by split/generate/replay.")
    session_gate_parser.add_argument("--comparison-json", default="", help="Optional compare-hsmm or replay-evidence-hsmm JSON override.")
    session_gate_parser.add_argument("--merge-report", default="", help="Optional final.ini merge report JSON override.")
    session_gate_parser.add_argument("--out", default="", help="Optional JSON gate report path.")
    session_gate_parser.add_argument(
        "--quality-evidence",
        default="none",
        choices=("none", "manual-reference", "listening", "manual-reference-and-listening"),
        help="Expected external quality evidence type for default-on consideration. Default stays fail-closed.",
    )
    session_gate_parser.add_argument(
        "--quality-evidence-report",
        default="",
        help="Optional JSON report proving manual/listening evidence for this final.ini.",
    )
    session_gate_parser.add_argument(
        "--attention-ratio-max",
        type=float,
        default=0.50,
        help="Maximum attention_only ratio allowed for default-on candidate status.",
    )
    session_gate_parser.add_argument(
        "--gate-target",
        default="apply",
        choices=("apply", "default-on"),
        help="Gate used with --fail-on-gate-failure.",
    )
    session_gate_parser.add_argument(
        "--fail-on-gate-failure",
        action="store_true",
        help="Return exit code 2 when the selected gate target does not pass.",
    )
    session_gate_parser.set_defaults(func=_cmd_evaluate_session)

    session_parser = subparsers.add_parser(
        "session-status",
        help="Inspect a review_session.json manifest and verify current split artifact lineage.",
    )
    session_parser.add_argument("--review-session", required=True, help="review_session.json emitted by split/generate/replay.")
    session_parser.add_argument("--out", default="", help="Optional JSON status report path.")
    session_parser.add_argument(
        "--require-default-on",
        action="store_true",
        help="Report ready_to_install using default-on gate status instead of apply gate status.",
    )
    session_parser.set_defaults(func=_cmd_session_status)

    prepare_parser = subparsers.add_parser(
        "prepare-edits",
        help="Copy review queue files to edited files without changing the original split artifacts.",
    )
    prepare_parser.add_argument("--review-session", required=True, help="review_session.json emitted by split/generate/replay.")
    prepare_parser.add_argument("--out", default="", help="Optional JSON prepare report path.")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing edited files. Off by default to avoid losing manual edits.",
    )
    prepare_parser.set_defaults(func=_cmd_prepare_edits)

    quality_parser = subparsers.add_parser(
        "record-quality-evidence",
        help="Write a manual/listening quality evidence report tied to a final.ini SHA.",
    )
    quality_parser.add_argument("--final", required=True, help="final.ini candidate that was manually/listening reviewed.")
    quality_parser.add_argument("--merge-report", required=True, help="Merge report for the reviewed final.ini.")
    quality_parser.add_argument("--out", required=True, help="Output quality evidence JSON report path.")
    quality_parser.add_argument(
        "--evidence-type",
        required=True,
        choices=("manual-reference", "listening", "manual-reference-and-listening"),
        help="Type of external quality evidence captured in this report.",
    )
    pass_group = quality_parser.add_mutually_exclusive_group(required=True)
    pass_group.add_argument("--passed", action="store_true", help="Mark the reviewed final.ini as passing evidence review.")
    pass_group.add_argument("--failed", action="store_true", help="Mark the reviewed final.ini as failing evidence review.")
    quality_parser.add_argument("--critical-failures", type=int, default=0, help="Critical failure count found in review.")
    quality_parser.add_argument("--reviewed-rows", type=int, required=True, help="Number of OTO rows reviewed.")
    quality_parser.add_argument("--reviewer", default="", help="Optional reviewer/operator label.")
    quality_parser.add_argument("--notes", default="", help="Optional short evidence notes.")
    quality_parser.add_argument("--manual-reference", default="", help="Optional manual reference artifact path or label.")
    quality_parser.add_argument(
        "--manual-reference-oto",
        default="",
        help="Manual/reference oto.ini to compare against final.ini; required for manual-reference evidence types.",
    )
    quality_parser.add_argument(
        "--manual-reference-max-delta-ms",
        type=float,
        default=250.0,
        help="Maximum allowed absolute OTO parameter delta against --manual-reference-oto for a passing report.",
    )
    quality_parser.add_argument(
        "--manual-reference-row-delta-limit",
        type=int,
        default=100,
        help="Maximum manual-reference row delta records embedded in the evidence report. Use 0 for all rows.",
    )
    quality_parser.add_argument(
        "--listening-summary",
        default="",
        help="Listening test artifact path or summary; required for listening evidence types.",
    )
    quality_parser.set_defaults(func=_cmd_record_quality_evidence)

    install_parser = subparsers.add_parser(
        "install-final",
        help="Dry-run or install a gated final.ini candidate into a target voicebank oto.ini.",
    )
    install_parser.add_argument("--final", required=True, help="final.ini candidate emitted by merge.")
    install_parser.add_argument("--target-oto", required=True, help="Target oto.ini path or voicebank directory.")
    install_parser.add_argument("--gate-report", required=True, help="evaluate-gate JSON report produced for this candidate.")
    install_parser.add_argument("--out", default="", help="Optional JSON install report path.")
    install_parser.add_argument("--backup-dir", default="", help="Optional directory for oto.ini backups. Defaults to target directory.")
    install_parser.add_argument(
        "--replace",
        action="store_true",
        help="Actually replace target oto.ini after gate, merge report, and SHA checks pass. Default is dry-run.",
    )
    install_parser.add_argument(
        "--require-default-on",
        action="store_true",
        help="Require default-on gate status instead of apply gate status before replacement.",
    )
    install_parser.add_argument(
        "--allow-create",
        action="store_true",
        help="Allow creating a missing target oto.ini. Existing voicebanks should leave this off.",
    )
    install_parser.set_defaults(func=_cmd_install_final)

    install_session_parser = subparsers.add_parser(
        "install-session",
        help="Dry-run or install a gated review session final.ini into a target voicebank oto.ini.",
    )
    install_session_parser.add_argument("--review-session", required=True, help="review_session.json emitted by split/generate/replay.")
    install_session_parser.add_argument("--target-oto", required=True, help="Target oto.ini path or voicebank directory.")
    install_session_parser.add_argument("--gate-report", default="", help="Optional gate report override. Defaults to the session expected report.")
    install_session_parser.add_argument("--final", default="", help="Optional final.ini override. Defaults to the session merge final.")
    install_session_parser.add_argument("--out", default="", help="Optional JSON install report path.")
    install_session_parser.add_argument("--backup-dir", default="", help="Optional directory for oto.ini backups. Defaults to target directory.")
    install_session_parser.add_argument(
        "--replace",
        action="store_true",
        help="Actually replace target oto.ini after gate, merge report, and SHA checks pass. Default is dry-run.",
    )
    install_session_parser.add_argument(
        "--require-default-on",
        action="store_true",
        help="Require default-on gate status instead of apply gate status before replacement.",
    )
    install_session_parser.add_argument(
        "--allow-create",
        action="store_true",
        help="Allow creating a missing target oto.ini. Existing voicebanks should leave this off.",
    )
    install_session_parser.set_defaults(func=_cmd_install_session)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
