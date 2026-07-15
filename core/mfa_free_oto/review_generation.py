from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Mapping

from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter
from core.generation.common.validation_diagnostics import (
    adapter_warning_validation_metrics,
    runtime_metadata_validation_metrics,
)
from core.generation.common.validation_splitter import review_split_guard_reasons, split_oto_file
from core.mfa_free_oto.evidence_pack import summarize_event_candidate_reliability
from core.mfa_free_oto.row_plan import build_filename_row_plan, write_row_plan_jsonl
from core.mfa_free_oto.workflow import generate_no_mfa_oto_with_model_context


def generate_hsmm_oto_review(
    *,
    wav_dir: str,
    out_dir: str,
    name: str = "mfa_free",
    template_oto: str = "",
    language: str = "japanese",
    format_type: str = "CV",
    alias_type: str = "auto",
    encoder: str = "acoustic_world_v1",
    checkpoint_path: str = "",
    device: str = "",
    apply_lightgbm: bool = False,
    lightgbm_policy: str = "auto",
    lightgbm_model_dir: str = "",
    callback: Callable[[str], None] | None = None,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Generate a UI-safe HSMM OTO preview without importing scripts/dev at runtime."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = str(name or "mfa_free").strip() or "mfa_free"
    generated_path = root / f"{stem}.generated.ini"
    messages: list[str] = []

    def emit(message: str) -> None:
        text = str(message or "")
        if not text:
            return
        messages.append(text)
        if callback is not None:
            try:
                callback(text)
            except Exception:
                pass

    report = generate_no_mfa_oto_with_model_context(
        wav_dir=wav_dir,
        out_path=str(generated_path),
        source_oto_path=template_oto,
        alias_suffix="",
        checkpoint_path=checkpoint_path,
        language=language,
        format_type=format_type,
        alias_type=alias_type,
        encoder=encoder,
        device=device or None,
        use_slot_viterbi=True,
        use_hsmm_decoder=True,
        callback=emit,
        max_workers=max_workers,
    )
    if report.errors or not generated_path.is_file():
        return {
            "ok": False,
            "generated_oto_path": str(generated_path),
            "processed": int(report.processed),
            "total": int(report.total),
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "messages": messages,
            "use_hsmm_decoder": True,
        }

    file_consistency = _apply_hsmm_file_consistency(
        generated_path=generated_path,
        language=language,
        format_type=format_type,
        emit=emit,
    )

    lightgbm_postprocess = _apply_lightgbm_postprocess(
        generated_path=generated_path,
        out_dir=root,
        stem=stem,
        wav_dir=wav_dir,
        language=language,
        format_type=format_type,
        apply_lightgbm=bool(apply_lightgbm),
        lightgbm_policy=lightgbm_policy,
        lightgbm_model_dir=lightgbm_model_dir,
        emit=emit,
    )

    timeline_rows = list(report.timeline_debug or ())
    evidence_rows = list(report.evidence_debug or ())
    row_plan_records = []
    row_plan_path = ""
    row_plan_sha256 = ""
    if not str(template_oto or "").strip():
        for wav_path in sorted(Path(wav_dir).glob("*.wav")):
            row_plan_records.extend(
                build_filename_row_plan(
                    wav_path.name,
                    language=language,
                    format_type=format_type,
                )
            )
        if row_plan_records:
            row_plan_path = str(root / f"{stem}.row_plan.jsonl")
            write_row_plan_jsonl(row_plan_path, row_plan_records)
            row_plan_sha256 = _sha256_file(Path(row_plan_path))

    timeline_path = ""
    timeline_sha256 = ""
    if timeline_rows:
        timeline_path = str(root / f"{stem}.timeline.jsonl")
        _write_jsonl(Path(timeline_path), timeline_rows)
        timeline_sha256 = _sha256_file(Path(timeline_path))

    evidence_path = ""
    evidence_sha256 = ""
    if evidence_rows:
        evidence_path = str(root / f"{stem}.evidence.jsonl")
        _write_jsonl(Path(evidence_path), evidence_rows)
        evidence_sha256 = _sha256_file(Path(evidence_path))

    row_provenance_records = _row_provenance_records_from_timeline_records(timeline_rows)
    row_provenance_path = ""
    row_provenance_sha256 = ""
    if row_provenance_records:
        row_provenance_path = str(root / f"{stem}.row_provenance.jsonl")
        _write_jsonl(Path(row_provenance_path), row_provenance_records)
        row_provenance_sha256 = _sha256_file(Path(row_provenance_path))

    split = split_oto_file(
        str(generated_path),
        str(root),
        name=stem,
        language=language,
        wav_root=wav_dir,
        row_diagnostics=_row_diagnostics_from_timeline_records(timeline_rows),
        session_metadata={
            "created_by": "core.mfa_free_oto.review_generation.generate_hsmm_oto_review",
            "wav_dir": str(wav_dir or ""),
            "template_oto": str(template_oto or ""),
            "source_timing_trusted": False,
            "checkpoint": str(checkpoint_path or ""),
            "language": language,
            "format_type": format_type,
            "alias_type": alias_type,
            "encoder": encoder,
            "use_hsmm_decoder": True,
            "row_plan_path": row_plan_path,
            "row_plan_sha256": row_plan_sha256,
            "row_plan_rows": len(row_plan_records),
            "timeline_path": timeline_path,
            "timeline_sha256": timeline_sha256,
            "timeline_wavs": len(timeline_rows),
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha256,
            "evidence_wavs": len(evidence_rows),
            "row_provenance_path": row_provenance_path,
            "row_provenance_sha256": row_provenance_sha256,
            "row_provenance_rows": len(row_provenance_records),
            "file_consistency": dict(file_consistency),
            "lightgbm_postprocess": dict(lightgbm_postprocess),
            "preferred_encoding_source": str(generated_path),
        },
    )
    split_guard_reasons = review_split_guard_reasons(split.counts)
    if split_guard_reasons:
        emit(
            "[No-MFA/MFA-Free] review split guard=fail "
            + ",".join(str(reason) for reason in split_guard_reasons)
        )
    warnings = list(report.warnings)
    warnings.extend(str(reason) for reason in split_guard_reasons)
    guard_failed = bool(report.guard_failed or split_guard_reasons)
    return {
        "ok": True,
        "generated_oto_path": str(generated_path),
        "generated_oto_sha256": _sha256_file(generated_path),
        "processed": int(report.processed),
        "total": int(report.total),
        "confidence": float(report.confidence),
        "guard_failed": guard_failed,
        "warnings": list(dict.fromkeys(warnings)),
        "messages": messages,
        "metrics": dict(report.metrics or {}),
        "review_split_guard_reasons": list(split_guard_reasons),
        "row_plan_path": row_plan_path,
        "row_plan_sha256": row_plan_sha256,
        "row_plan_rows": len(row_plan_records),
        "timeline_path": timeline_path,
        "timeline_sha256": timeline_sha256,
        "timeline_wavs": len(timeline_rows),
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "evidence_wavs": len(evidence_rows),
        "row_provenance_path": row_provenance_path,
        "row_provenance_sha256": row_provenance_sha256,
        "row_provenance_rows": len(row_provenance_records),
        "file_consistency": dict(file_consistency),
        "lightgbm_postprocess": dict(lightgbm_postprocess),
        "split_counts": dict(split.counts),
        "split_output_paths": dict(split.output_paths),
        "use_hsmm_decoder": True,
    }


def _apply_hsmm_file_consistency(
    *,
    generated_path: Path,
    language: str,
    format_type: str,
    emit: Callable[[str], None],
) -> dict[str, object]:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if lang != "japanese" or "cvvc" not in fmt:
        return {"enabled": False, "reason": "scope_not_japanese_cvvc"}
    if not generated_path.is_file():
        return {"enabled": True, "status": "skipped", "reason": "generated_oto_missing"}
    try:
        from core.generation.common.oto_generator import validate_oto_params
        from core.generation.ja.ja_oto_file_consistency import apply_ja_vc_neighbor_to_oto_file

        stats = apply_ja_vc_neighbor_to_oto_file(
            str(generated_path),
            validate_fn=validate_oto_params,
            log_fn=emit,
        )
        changed = int(stats.get("total_changed", 0) or 0)
        if changed > 0:
            emit(f"[No-MFA] file consistency: changed={changed}.")
        return {
            "enabled": True,
            "status": "applied" if changed > 0 else "no_change",
            **dict(stats),
        }
    except Exception as exc:
        emit(f"[No-MFA] file consistency failed: {exc}")
        return {"enabled": True, "status": "failed", "reason": str(exc)}


def _apply_lightgbm_postprocess(
    *,
    generated_path: Path,
    out_dir: Path,
    stem: str,
    wav_dir: str,
    language: str,
    format_type: str,
    apply_lightgbm: bool,
    lightgbm_policy: str,
    lightgbm_model_dir: str,
    emit: Callable[[str], None],
) -> dict[str, object]:
    if not apply_lightgbm:
        return {"enabled": False}
    lang = str(language or "").strip().lower() or "japanese"
    fmt = str(format_type or "").strip().lower()
    if not generated_path.is_file():
        return {"enabled": True, "status": "skipped", "reason": "generated_oto_missing"}

    pre_path = out_dir / f"{stem}.pre_lightgbm.ini"
    shutil.copyfile(generated_path, pre_path)
    before_sha = _sha256_file(pre_path)
    lightgbm_report: dict[str, object] = {}
    env_keys = (
        "UTOA_ML_ENSEMBLE_ENABLE",
        "UTOA_ML_ROUTE",
        "UTOA_JA_OTO_ML_DIR",
        "UTOA_KR_OTO_ML_DIR",
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}

    try:
        model_dir_arg = str(lightgbm_model_dir or "").strip()
        if model_dir_arg:
            model_dir_arg = str(Path(model_dir_arg).resolve())
            if lang == "japanese":
                os.environ["UTOA_JA_OTO_ML_DIR"] = model_dir_arg
            else:
                os.environ["UTOA_KR_OTO_ML_DIR"] = model_dir_arg
        os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "0"
        os.environ["UTOA_ML_ROUTE"] = "legacy"

        from core.runtime.runtime_config import reset_ml_config
        from core.oto_ml_refiner import _resolve_lightgbm_model_dir, apply_oto_ml_to_oto_file

        reset_ml_config()
        resolved_lightgbm_dir = _resolve_lightgbm_model_dir(lang, fmt) or ""
        if not resolved_lightgbm_dir:
            emit("[No-MFA] LightGBM postprocess skipped: model not found.")
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "lightgbm_model_not_found",
                "pre_lightgbm_path": str(pre_path),
                "pre_lightgbm_sha256": before_sha,
                "model_dir": "",
                "changed": 0,
            }

        policy = str(lightgbm_policy or "auto").strip().lower() or "auto"
        changed = int(
            apply_oto_ml_to_oto_file(
                lang,
                str(generated_path),
                tg_dir="",
                wav_dir=str(wav_dir or ""),
                custom_phonemes_path="",
                callback=emit,
                enabled=True,
                format_override=fmt,
                policy=policy,
                report=lightgbm_report,
                ml_route="legacy",
            )
            or 0
        )
        safety_report = apply_no_mfa_lightgbm_safety_filter(
            pre_oto_path=str(pre_path),
            post_oto_path=str(generated_path),
            language=lang,
            format_type=fmt,
        )
        after_sha = _sha256_file(generated_path)
        status = str(lightgbm_report.get("status", "") or ("applied" if changed else "no_change"))
        emit(f"[No-MFA] LightGBM postprocess: status={status}, changed={changed}.")
        if bool(safety_report.get("enabled")) and int(safety_report.get("changed_rows", 0) or 0) > 0:
            emit(
                "[No-MFA] LightGBM safety filter: "
                f"changed_rows={int(safety_report.get('changed_rows', 0) or 0)}, "
                f"restored_terminal={int(safety_report.get('restored_terminal_rows', 0) or 0)}, "
                f"restored_v_offset_cutoff={int(safety_report.get('restored_standalone_vowel_offset_cutoff_rows', 0) or 0)}, "
                f"restored_cv_head_cutoff={int(safety_report.get('restored_cv_head_cutoff_rows', 0) or 0)}, "
                f"restored_spaced_consonant={int(safety_report.get('restored_spaced_consonant_rows', 0) or 0)}, "
                f"restored_vv_overlap={int(safety_report.get('restored_vv_overlap_rows', 0) or 0)}, "
                f"restored_terminal_vc_overlap={int(safety_report.get('restored_terminal_vc_overlap_rows', 0) or 0)}, "
                f"capped_overlap={int(safety_report.get('capped_overlap_rows', 0) or 0)}, "
                f"repaired_parameter_order={int(safety_report.get('repaired_parameter_order_rows', 0) or 0)}."
            )
        return {
            "enabled": True,
            "status": status,
            "policy": policy,
            "pre_lightgbm_path": str(pre_path),
            "pre_lightgbm_sha256": before_sha,
            "generated_oto_sha256_before": before_sha,
            "generated_oto_sha256_after": after_sha,
            "model_dir": str(resolved_lightgbm_dir),
            "changed": int(changed),
            "report": dict(lightgbm_report),
            "safety_filter": dict(safety_report),
        }
    except Exception as exc:
        if pre_path.is_file():
            shutil.copyfile(pre_path, generated_path)
        emit(f"[No-MFA] LightGBM postprocess failed; restored pre-LightGBM OTO: {exc}")
        return {
            "enabled": True,
            "status": "failed",
            "reason": str(exc),
            "pre_lightgbm_path": str(pre_path),
            "pre_lightgbm_sha256": before_sha,
            "restored_pre_lightgbm": True,
            "changed": 0,
            "report": dict(lightgbm_report),
        }
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            from core.runtime.runtime_config import reset_ml_config

            reset_ml_config()
        except Exception:
            pass


def _row_diagnostics_from_timeline_records(records: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    line_index = 0
    for record in records:
        for row in list(record.get("adapted_rows", []) or []):
            metrics = _hsmm_validation_metrics_for_row(record, row)
            if metrics:
                out[line_index] = dict(metrics)
            line_index += 1
    return out


def _row_provenance_records_from_timeline_records(
    records: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    global_index = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        wav = str(record.get("wav", "") or "")
        selected_event_source = str(record.get("selected_event_source", "") or "")
        hsmm = record.get("hsmm")
        hsmm_summary = _row_provenance_hsmm_summary(hsmm if isinstance(hsmm, Mapping) else None)
        for local_index, row in enumerate(list(record.get("adapted_rows", []) or [])):
            if not isinstance(row, Mapping):
                continue
            anchor = row.get("anchor")
            row_plan = row.get("row_plan")
            anchor_map = dict(anchor or {}) if isinstance(anchor, Mapping) else {}
            row_plan_map = dict(row_plan or {}) if isinstance(row_plan, Mapping) else {}
            slot_index = _provenance_int(
                row_plan_map.get("slot_index", anchor_map.get("slot_index", -1)),
                default=-1,
            )
            left_slot_index = _provenance_int(row_plan_map.get("left_slot_index", slot_index), default=slot_index)
            right_slot_index = _provenance_int(row_plan_map.get("right_slot_index", slot_index), default=slot_index)
            out.append(
                {
                    "schema_version": 1,
                    "global_row_index": int(global_index),
                    "row_index": _provenance_int(row.get("row_index", local_index), default=int(local_index)),
                    "wav": wav,
                    "alias": str(row.get("alias", "") or ""),
                    "mode": str(row.get("mode", "") or ""),
                    "role": str(row_plan_map.get("role_family", anchor_map.get("role", "")) or ""),
                    "selected_event_source": selected_event_source,
                    "slot": {
                        "slot_index": int(slot_index),
                        "left_slot_index": int(left_slot_index),
                        "right_slot_index": int(right_slot_index),
                        "expected_phone_index": _provenance_int(
                            row_plan_map.get(
                                "expected_phone_index",
                                anchor_map.get("expected_phone_index", -1),
                            ),
                            default=-1,
                        ),
                    },
                    "timing": _json_safe(row.get("timing", {})),
                    "absolute": _json_safe(row.get("absolute", {})),
                    "anchor": {
                        "anchor_abs_ms": anchor_map.get("anchor_abs_ms"),
                        "score": anchor_map.get("score"),
                        "role": anchor_map.get("role"),
                        "source_event_label": anchor_map.get("source_event_label"),
                        "expected_phone": anchor_map.get("expected_phone"),
                        "expected_phone_index": anchor_map.get("expected_phone_index"),
                        "slot_index": anchor_map.get("slot_index"),
                        "boundary_confidence": anchor_map.get("boundary_confidence"),
                        "nucleus_confidence": anchor_map.get("nucleus_confidence"),
                        "warnings": list(anchor_map.get("warnings", []) or []),
                    },
                    "row_plan": _json_safe(row_plan_map),
                    "applied_rules": list(row.get("applied_rules", []) or []),
                    "warnings": list(row.get("warnings", []) or []),
                    "hsmm": dict(hsmm_summary),
                }
            )
            global_index += 1
    return out


def _row_provenance_hsmm_summary(hsmm: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(hsmm, Mapping):
        return {"available": False}
    return {
        "available": True,
        "ok": bool(hsmm.get("ok")),
        "reason": str(hsmm.get("reason", "") or ""),
        "score": hsmm.get("score"),
    }


def _provenance_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    if not aggregate or not isinstance(row_plan, Mapping):
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
    global_margins = [_float_or_none(state.get("selected_vs_global_best_margin")) for state in states]
    wrong_occurrence_count = sum(1 for state in states if bool(state.get("wrong_occurrence_risk")))
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
        "hsmm_min_selected_vs_best_local_margin": min(value for value in best_margins if value is not None)
        if any(value is not None for value in best_margins)
        else 0.0,
        "hsmm_min_selected_vs_second_local_margin": min(value for value in second_margins if value is not None)
        if any(value is not None for value in second_margins)
        else 0.0,
        "hsmm_min_selected_vs_global_best_margin": min(value for value in global_margins if value is not None)
        if any(value is not None for value in global_margins)
        else 0.0,
        "hsmm_wrong_occurrence_risk_count": float(wrong_occurrence_count),
        "hsmm_max_duration_z_abs": max(value for value in duration_z_values if value is not None)
        if any(value is not None for value in duration_z_values)
        else 0.0,
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
        "hsmm_min_selected_vs_best_local_margin": _float_or_none(diagnostics.get("min_selected_vs_best_local_margin"))
        or 0.0,
        "hsmm_min_selected_vs_second_local_margin": _float_or_none(
            diagnostics.get("min_selected_vs_second_local_margin")
        )
        or 0.0,
        "hsmm_min_selected_vs_global_best_margin": _float_or_none(
            diagnostics.get("min_selected_vs_global_best_margin")
        )
        or 0.0,
        "hsmm_wrong_occurrence_risk_count": _float_or_none(diagnostics.get("wrong_occurrence_risk_state_count"))
        or 0.0,
        "hsmm_max_duration_z_abs": _float_or_none(diagnostics.get("max_duration_z_abs")) or 0.0,
    }


def _state_slot_index(state_id: str) -> int | None:
    text = str(state_id or "")
    if not text.startswith("slot"):
        return None
    digits = []
    for char in text[4:]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number == number else None


def _write_jsonl(path: Path, rows: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["generate_hsmm_oto_review"]
