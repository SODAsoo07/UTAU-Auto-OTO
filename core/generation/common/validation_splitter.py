from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

from core.generation.common.oto_encoding import detect_text_file_encoding, write_oto_lines_with_encoding
from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.model_context.oto_params import resolve_cutoff_abs_ms, wav_duration_ms


VALIDATION_SPLIT_SCHEMA_VERSION = 1

SPLIT_FIX_REQUIRED = "fix_required"
SPLIT_ATTENTION_ONLY = "attention_only"
SPLIT_CLEAN = "clean"
VALIDATION_SPLITS = frozenset({SPLIT_FIX_REQUIRED, SPLIT_ATTENTION_ONLY, SPLIT_CLEAN})


@dataclass(frozen=True)
class ValidationRecord:
    row_id: str
    line_index: int
    raw_line: str
    wav: str
    alias: str
    occurrence_index: int
    split: str
    reasons: tuple[str, ...]
    slot_index: int = -1
    left_slot_index: int = -1
    right_slot_index: int = -1
    source_row_index: int = -1
    source_timing_trusted: bool = False
    offset_abs: float = 0.0
    overlap_abs: float = 0.0
    pre_abs: float = 0.0
    fixed_end_abs: float = 0.0
    cutoff_abs: float = 0.0
    duration_ms: float = 0.0
    role_family: str = "other"
    schema_version: int = VALIDATION_SPLIT_SCHEMA_VERSION
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["metrics"] = dict(self.metrics)
        return data


@dataclass(frozen=True)
class SplitResult:
    records: tuple[ValidationRecord, ...]
    counts: Mapping[str, int]
    output_paths: Mapping[str, str]
    summary: Mapping[str, object]


def build_row_id(
    *,
    wav: str,
    alias: str,
    line_index: int,
    occurrence_index: int,
    slot_index: int = -1,
    source_row_index: int = -1,
    role_family: str = "other",
) -> str:
    payload = {
        "wav": str(wav or ""),
        "alias": str(alias or ""),
        "line_index": int(line_index),
        "occurrence_index": int(occurrence_index),
        "slot_index": int(slot_index),
        "source_row_index": int(source_row_index),
        "role_family": str(role_family or "other"),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"row_{int(line_index):06d}_{digest}"


def classify_alias_role(alias: str) -> str:
    text = str(alias or "").strip().lower()
    if not text:
        return "other"
    parts = text.split()
    if len(parts) >= 2:
        left, right = parts[0], parts[-1]
        if _looks_vowel(left) and _looks_vowel(right):
            return "vv"
        if _looks_vowel(left):
            return "vc"
        return "cv"
    compact = "".join(part for part in text.split())
    if text in {"br", "breath", "sil", "pau", "sp"}:
        return "br"
    if compact.startswith("br") and compact[2:].isdigit():
        return "br"
    if _looks_vowel(text):
        return "v"
    return "cv"


def validate_oto_lines(
    lines: Sequence[str],
    *,
    language: str = "",
    format_type: str = "",
    wav_root: str = "",
    wav_durations_ms: Mapping[str, float] | None = None,
    row_diagnostics: Mapping[int, Mapping[str, object]] | None = None,
    min_fixed_ms: float = 20.0,
    min_tail_ms: float = 5.0,
    hsmm_local_margin_attention: float = -0.03,
    hsmm_local_margin_fix: float = -0.12,
    hsmm_duration_z_attention: float = 2.0,
    hsmm_duration_z_fix: float = 3.5,
    local_refine_changed_attention_delta_ms: float = 40.0,
) -> tuple[ValidationRecord, ...]:
    durations = dict(wav_durations_ms or {})
    diagnostics_by_line = dict(row_diagnostics or {})
    occurrence: dict[tuple[str, str], int] = {}
    records: list[ValidationRecord] = []

    for idx, raw in enumerate(lines):
        text = str(raw or "").rstrip("\r\n")
        if not text.strip():
            continue
        row = parse_oto_line(text)
        if row is None:
            row_id = build_row_id(
                wav="",
                alias="",
                line_index=idx,
                occurrence_index=0,
                role_family="invalid",
            )
            records.append(
                ValidationRecord(
                    row_id=row_id,
                    line_index=idx,
                    raw_line=text,
                    wav="",
                    alias="",
                    occurrence_index=0,
                    split=SPLIT_FIX_REQUIRED,
                    reasons=("fix_required.invalid_oto_line",),
                    role_family="invalid",
                )
            )
            continue

        wav = str(row["wav"])
        alias = str(row["alias"])
        diagnostics = diagnostics_by_line.get(idx, {})
        row_plan_context = _row_plan_context(diagnostics)
        role = str(row_plan_context.get("role_family") or classify_alias_role(alias))
        key = (wav, alias)
        occurrence_index = occurrence.get(key, 0)
        occurrence[key] = occurrence_index + 1
        duration_ms = _resolve_duration_ms(wav, wav_root=wav_root, durations=durations)
        offset = float(row["offset"])
        overlap = float(row["ovl"])
        pre = float(row["pre"])
        fixed = float(row["cons"])
        cutoff = float(row["cutoff"])
        cutoff_abs = resolve_cutoff_abs_ms(offset_ms=offset, cutoff=cutoff, duration_ms=duration_ms)
        overlap_abs = offset + max(0.0, overlap)
        pre_abs = offset + max(0.0, pre)
        fixed_abs = offset + max(0.0, fixed)
        special_zero_template = _is_special_zero_template_row(
            wav=wav,
            alias=alias,
            offset=offset,
            overlap=overlap,
            pre=pre,
            fixed=fixed,
            cutoff=cutoff,
        )
        trusted_nonphonetic_template = bool(row_plan_context.get("source_timing_trusted", False)) and role in {
            "br",
            "policy_breath",
        }

        fix_reasons: list[str] = []
        attention_reasons: list[str] = []
        if not special_zero_template and not trusted_nonphonetic_template:
            if offset < 0.0:
                fix_reasons.append("fix_required.invalid_oto_order")
            if overlap_abs > pre_abs + 1e-6 or pre_abs > fixed_abs + 1e-6 or fixed_abs >= cutoff_abs + 1e-6:
                fix_reasons.append("fix_required.invalid_oto_order")
            if fixed < float(min_fixed_ms):
                fix_reasons.append("fix_required.too_short_fixed")
            if cutoff_abs <= fixed_abs + float(min_tail_ms):
                fix_reasons.append("fix_required.cutoff_invalid")
            if cutoff >= 0.0:
                attention_reasons.append("attention.positive_cutoff")
            if duration_ms <= 0.0:
                attention_reasons.append("attention.duration_unknown")
            if role in {"vc", "vv"} and overlap <= 0.0:
                attention_reasons.append("attention.short_tail")
        diagnostic_metrics = _numeric_diagnostics(diagnostics)
        if not special_zero_template and not trusted_nonphonetic_template:
            if _korean_cvc_default_cv_profile_needs_attention(
                language=language,
                format_type=format_type,
                role=role,
                alias=alias,
                pre=pre,
                overlap=overlap,
            ):
                attention_reasons.append("attention.korean_cvc_default_cv_profile")
            local_margin = diagnostic_metrics.get("hsmm_min_selected_vs_best_local_margin")
            if local_margin is not None:
                if local_margin < float(hsmm_local_margin_fix):
                    fix_reasons.append("fix_required.hsmm_local_margin_bad")
                elif local_margin < float(hsmm_local_margin_attention):
                    attention_reasons.append("attention.hsmm_local_margin_low")
            duration_z = diagnostic_metrics.get("hsmm_max_duration_z_abs")
            if duration_z is not None:
                if duration_z >= float(hsmm_duration_z_fix):
                    fix_reasons.append("fix_required.hsmm_duration_prior_violation")
                elif duration_z >= float(hsmm_duration_z_attention):
                    attention_reasons.append("attention.hsmm_duration_prior_deviation")
            if role == "vv":
                if _metric_positive(diagnostic_metrics, "evidence_vv_boundary_low_feature_agreement_count"):
                    fix_reasons.append("fix_required.evidence_vv_boundary_low_feature_agreement")
                elif _metric_positive(diagnostic_metrics, "evidence_vv_boundary_ambiguous_count") or _metric_positive(
                    diagnostic_metrics,
                    "evidence_vv_boundary_low_margin_count",
                ):
                    attention_reasons.append("attention.evidence_vv_boundary_ambiguous")
            if role in {"cv", "vc"} and _alias_has_sonorant(alias):
                if _metric_positive(diagnostic_metrics, "evidence_sonorant_constriction_low_feature_agreement_count"):
                    fix_reasons.append("fix_required.evidence_sonorant_low_feature_agreement")
                elif _metric_positive(
                    diagnostic_metrics,
                    "evidence_sonorant_constriction_ambiguous_count",
                ) or _metric_positive(
                    diagnostic_metrics,
                    "evidence_sonorant_constriction_low_margin_count",
                ):
                    attention_reasons.append("attention.evidence_sonorant_ambiguous")
            if _metric_positive(diagnostic_metrics, "local_refine_changed_anchor_count"):
                local_refine_delta = diagnostic_metrics.get("local_refine_max_delta_ms")
                if local_refine_delta is None or abs(float(local_refine_delta)) >= float(
                    local_refine_changed_attention_delta_ms
                ):
                    attention_reasons.append("attention.local_refine_changed_anchor")
            if _metric_positive(diagnostic_metrics, "local_refine_low_margin_count") and _local_refine_low_margin_needs_attention(
                language=language,
                format_type=format_type,
                role=role,
                diagnostic_metrics=diagnostic_metrics,
            ):
                attention_reasons.append("attention.local_refine_low_margin")
            if _metric_positive(diagnostic_metrics, "local_refine_rejected_slot_boundary_count"):
                attention_reasons.append("attention.local_refine_rejected_slot_boundary")
            if _metric_positive(diagnostic_metrics, "timing_clamp_large_delta_count"):
                attention_reasons.append("attention.timing_clamp_large_delta")
            elif _metric_positive(diagnostic_metrics, "timing_clamp_count"):
                attention_reasons.append("attention.timing_clamped")
            if _metric_positive(diagnostic_metrics, "row_warning_island_count_mismatch_count"):
                fix_reasons.append("fix_required.structural_island_count_mismatch")
            if _metric_positive(diagnostic_metrics, "row_warning_preutterance_slot_shift_count"):
                fix_reasons.append("fix_required.preutterance_slot_shift")
            if _metric_positive(diagnostic_metrics, "row_warning_local_gate_low_count"):
                attention_reasons.append("attention.local_gate_low")
            if _metric_positive(diagnostic_metrics, "row_warning_missing_local_gate_count"):
                attention_reasons.append("attention.missing_local_gate")
            if _metric_positive(diagnostic_metrics, "rule_based_checkpoint_failed_count"):
                attention_reasons.append("attention.rule_based_checkpoint_failed")
            elif _metric_positive(diagnostic_metrics, "rule_based_inference_count") and not _metric_positive(
                diagnostic_metrics,
                "hsmm_decoder_used",
            ) and _rule_based_inference_needs_attention(
                language=language,
                format_type=format_type,
                role=role,
                diagnostic_metrics=diagnostic_metrics,
            ):
                attention_reasons.append("attention.rule_based_inference")

        if fix_reasons:
            split = SPLIT_FIX_REQUIRED
            reasons = tuple(_dedupe(fix_reasons + attention_reasons))
        elif attention_reasons:
            split = SPLIT_ATTENTION_ONLY
            reasons = tuple(_dedupe(attention_reasons))
        else:
            split = SPLIT_CLEAN
            reasons = ("clean",)

        row_id = build_row_id(
            wav=wav,
            alias=alias,
            line_index=idx,
            occurrence_index=occurrence_index,
            slot_index=int(row_plan_context["slot_index"]),
            source_row_index=int(row_plan_context["source_row_index"]),
            role_family=role,
        )
        records.append(
            ValidationRecord(
                row_id=row_id,
                line_index=idx,
                raw_line=text,
                wav=wav,
                alias=alias,
                occurrence_index=occurrence_index,
                split=split,
                reasons=reasons,
                slot_index=int(row_plan_context["slot_index"]),
                left_slot_index=int(row_plan_context["left_slot_index"]),
                right_slot_index=int(row_plan_context["right_slot_index"]),
                source_row_index=int(row_plan_context["source_row_index"]),
                source_timing_trusted=bool(row_plan_context["source_timing_trusted"]),
                offset_abs=offset,
                overlap_abs=overlap_abs,
                pre_abs=pre_abs,
                fixed_end_abs=fixed_abs,
                cutoff_abs=cutoff_abs,
                duration_ms=duration_ms,
                role_family=role,
                metrics={
                    "fixed_ms": fixed,
                    "tail_ms": max(0.0, cutoff_abs - fixed_abs),
                    "pre_ms": pre,
                    "overlap_ms": overlap,
                    **diagnostic_metrics,
                },
            )
        )
    return tuple(records)


def split_oto_file(
    generated_oto_path: str,
    output_dir: str,
    *,
    name: str | None = None,
    language: str = "",
    wav_root: str = "",
    wav_durations_ms: Mapping[str, float] | None = None,
    row_diagnostics: Mapping[int, Mapping[str, object]] | None = None,
    session_metadata: Mapping[str, object] | None = None,
) -> SplitResult:
    text = read_text_with_fallback(generated_oto_path)
    lines = text.splitlines()
    metadata = dict(session_metadata or {})
    records = validate_oto_lines(
        lines,
        language=language,
        format_type=str(metadata.get("format_type", "") or ""),
        wav_root=wav_root,
        wav_durations_ms=wav_durations_ms,
        row_diagnostics=row_diagnostics,
    )
    base = str(name or os.path.splitext(os.path.basename(generated_oto_path))[0] or "oto")
    os.makedirs(output_dir, exist_ok=True)
    preferred = detect_text_file_encoding(generated_oto_path)

    split_lines = {
        SPLIT_FIX_REQUIRED: [r.raw_line for r in records if r.split == SPLIT_FIX_REQUIRED],
        SPLIT_ATTENTION_ONLY: [r.raw_line for r in records if r.split == SPLIT_ATTENTION_ONLY],
        SPLIT_CLEAN: [r.raw_line for r in records if r.split == SPLIT_CLEAN],
    }
    review_lines = split_lines[SPLIT_FIX_REQUIRED] + split_lines[SPLIT_ATTENTION_ONLY]
    output_paths = {
        "fix_required": os.path.join(output_dir, f"{base}.fix_required.ini"),
        "attention_only": os.path.join(output_dir, f"{base}.attention_only.ini"),
        "clean": os.path.join(output_dir, f"{base}.clean.ini"),
        "review_all": os.path.join(output_dir, f"{base}.review_all.ini"),
        "validation_jsonl": os.path.join(output_dir, f"{base}.validation.jsonl"),
        "validation_csv": os.path.join(output_dir, f"{base}.validation.csv"),
        "summary": os.path.join(output_dir, f"{base}.summary.json"),
        "review_session": os.path.join(output_dir, f"{base}.review_session.json"),
    }

    write_oto_lines_with_encoding(
        output_paths["fix_required"],
        split_lines[SPLIT_FIX_REQUIRED],
        language=language,
        preferred_encoding=preferred,
    )
    write_oto_lines_with_encoding(
        output_paths["attention_only"],
        split_lines[SPLIT_ATTENTION_ONLY],
        language=language,
        preferred_encoding=preferred,
    )
    write_oto_lines_with_encoding(
        output_paths["clean"],
        split_lines[SPLIT_CLEAN],
        language=language,
        preferred_encoding=preferred,
    )
    write_oto_lines_with_encoding(
        output_paths["review_all"],
        review_lines,
        language=language,
        preferred_encoding=preferred,
    )
    _write_jsonl(output_paths["validation_jsonl"], [r.to_json_dict() for r in records])
    _write_csv(output_paths["validation_csv"], records)
    output_sha256 = {
        key: _sha256_file(path)
        for key, path in output_paths.items()
        if key not in {"summary", "review_session"} and os.path.isfile(path)
    }

    counts = {
        "total": len(records),
        "fix_required": len(split_lines[SPLIT_FIX_REQUIRED]),
        "attention_only": len(split_lines[SPLIT_ATTENTION_ONLY]),
        "clean": len(split_lines[SPLIT_CLEAN]),
    }
    summary = {
        "schema_version": VALIDATION_SPLIT_SCHEMA_VERSION,
        "generated_oto_path": os.path.abspath(generated_oto_path),
        "generated_oto_sha256": _sha256_file(generated_oto_path) if os.path.isfile(generated_oto_path) else "",
        "counts": counts,
        "role_counts": _role_counts(records),
        "reason_counts": _reason_counts(records),
        "reason_counts_by_split": _reason_counts_by_split(records),
        "metric_positive_row_counts": _metric_positive_row_counts(records),
        "metric_min_values": _metric_min_values(records),
        "metric_max_values": _metric_max_values(records),
        "output_paths": output_paths,
        "output_sha256": output_sha256,
        "validation_jsonl_sha256": output_sha256.get("validation_jsonl", ""),
    }
    _write_json(output_paths["summary"], summary)
    session_output_sha256 = dict(output_sha256)
    if os.path.isfile(output_paths["summary"]):
        session_output_sha256["summary"] = _sha256_file(output_paths["summary"])
    review_session = _build_review_session_payload(
        generated_oto_path=generated_oto_path,
        output_dir=output_dir,
        base=base,
        counts=counts,
        output_paths=output_paths,
        output_sha256=session_output_sha256,
        metadata=metadata,
    )
    _write_json(output_paths["review_session"], review_session)
    return SplitResult(records=records, counts=counts, output_paths=output_paths, summary=summary)


def load_validation_records(path: str) -> tuple[ValidationRecord, ...]:
    records: list[ValidationRecord] = []
    if not path or not os.path.isfile(path):
        return ()
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            data = json.loads(raw)
            records.append(
                ValidationRecord(
                    row_id=str(data.get("row_id", "")),
                    line_index=int(data.get("line_index", 0)),
                    raw_line=str(data.get("raw_line", "")),
                    wav=str(data.get("wav", "")),
                    alias=str(data.get("alias", "")),
                    occurrence_index=int(data.get("occurrence_index", 0)),
                    split=str(data.get("split", "")),
                    reasons=tuple(str(x) for x in data.get("reasons", ()) or ()),
                    slot_index=int(data.get("slot_index", -1) or -1),
                    left_slot_index=int(data.get("left_slot_index", -1) or -1),
                    right_slot_index=int(data.get("right_slot_index", -1) or -1),
                    source_row_index=int(data.get("source_row_index", -1) or -1),
                    source_timing_trusted=bool(data.get("source_timing_trusted", False)),
                    offset_abs=float(data.get("offset_abs", 0.0) or 0.0),
                    overlap_abs=float(data.get("overlap_abs", 0.0) or 0.0),
                    pre_abs=float(data.get("pre_abs", 0.0) or 0.0),
                    fixed_end_abs=float(data.get("fixed_end_abs", 0.0) or 0.0),
                    cutoff_abs=float(data.get("cutoff_abs", 0.0) or 0.0),
                    duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
                    role_family=str(data.get("role_family", "other") or "other"),
                    schema_version=int(data.get("schema_version", VALIDATION_SPLIT_SCHEMA_VERSION) or 1),
                    metrics=dict(data.get("metrics", {}) or {}),
                )
            )
    return tuple(records)


def load_validation_records_with_errors(
    path: str,
    *,
    require_rows: bool = False,
) -> tuple[tuple[ValidationRecord, ...], tuple[str, ...]]:
    if not path or not os.path.isfile(path):
        return (), ("validation.file_missing",)
    try:
        records = load_validation_records(path)
    except json.JSONDecodeError:
        return (), ("validation.json_invalid",)
    except Exception:
        return (), ("validation.read_failed",)
    return records, validate_validation_records(records, require_rows=require_rows)


def validate_validation_records(
    records: Sequence[ValidationRecord],
    *,
    require_rows: bool = False,
) -> tuple[str, ...]:
    errors: list[str] = []
    if require_rows and not records:
        errors.append("validation.empty")
    seen_row_ids: set[str] = set()
    seen_line_indices: set[int] = set()
    for record in records:
        row_id = str(record.row_id or "").strip()
        if not row_id:
            errors.append("validation.row_id_missing")
        elif row_id in seen_row_ids:
            errors.append("validation.row_id_duplicate")
        seen_row_ids.add(row_id)

        line_index = int(record.line_index)
        if line_index < 0:
            errors.append("validation.line_index_invalid")
        elif line_index in seen_line_indices:
            errors.append("validation.line_index_duplicate")
        seen_line_indices.add(line_index)

        if int(record.schema_version) != VALIDATION_SPLIT_SCHEMA_VERSION:
            errors.append("validation.schema_version_unsupported")
        split = str(record.split or "")
        if split not in VALIDATION_SPLITS:
            errors.append("validation.split_invalid")

        reasons = tuple(str(item) for item in record.reasons)
        parsed = parse_oto_line(record.raw_line)
        if parsed is None:
            if "fix_required.invalid_oto_line" not in reasons:
                errors.append("validation.raw_line_invalid")
            continue
        if str(parsed.get("wav", "")) != str(record.wav) or str(parsed.get("alias", "")) != str(record.alias):
            errors.append("validation.raw_line_mismatch")
    return tuple(dict.fromkeys(errors))


def _resolve_duration_ms(wav: str, *, wav_root: str, durations: Mapping[str, float]) -> float:
    if wav in durations:
        return float(durations[wav])
    lower = wav.lower()
    for key, value in durations.items():
        if str(key).lower() == lower:
            return float(value)
    if wav_root:
        return wav_duration_ms(os.path.join(wav_root, wav), missing_value=0.0)
    return 0.0


def _is_special_zero_template_row(
    *,
    wav: str,
    alias: str,
    offset: float,
    overlap: float,
    pre: float,
    fixed: float,
    cutoff: float,
) -> bool:
    if any(abs(float(value)) > 1e-6 for value in (offset, overlap, pre, fixed, cutoff)):
        return False
    text = f"{wav} {alias}".strip().lower()
    compact_alias = "".join(part for part in str(alias or "").strip().lower().split())
    if compact_alias in {"br", "breath", "sil", "pau", "sp"}:
        return True
    if compact_alias.startswith("br") and compact_alias[2:].isdigit():
        return True
    return any(token in text for token in ("\u606f", "\u5438", "\u5410"))


def _looks_vowel(text: str) -> bool:
    token = str(text or "").strip().lower()
    return token in {"a", "i", "u", "e", "o", "eo", "eu", "ae"} or (
        len(token) == 1 and token in "aiueo"
    )


def _alias_has_sonorant(alias: str) -> bool:
    parts = [part for part in str(alias or "").strip().lower().split() if part]
    tokens = parts if parts else [str(alias or "").strip().lower()]
    sonorants = {"m", "n", "ng", "ny", "r", "l", "w", "y", "j"}
    for token in tokens:
        if token in sonorants:
            return True
        for sonorant in sonorants:
            if token.startswith(sonorant) and token != "a":
                return True
    return False


def _is_korean_cvc_validation_scope(*, language: str, format_type: str) -> bool:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    return lang in {"korean", "ko", "kr"} and "cvc" in fmt


def _korean_cvc_default_cv_profile_needs_attention(
    *,
    language: str,
    format_type: str,
    role: str,
    alias: str,
    pre: float,
    overlap: float,
) -> bool:
    if not _is_korean_cvc_validation_scope(language=language, format_type=format_type):
        return False
    if str(role or "").strip().lower() != "cv":
        return False
    if re.search(r"\s", str(alias or "")):
        return False
    return float(pre) >= 100.0 and float(overlap) >= 70.0


def _rule_based_inference_needs_attention(
    *,
    language: str,
    format_type: str,
    role: str,
    diagnostic_metrics: Mapping[str, float],
) -> bool:
    if not _is_korean_cvc_validation_scope(language=language, format_type=format_type):
        return True
    if _metric_positive(diagnostic_metrics, "rule_based_checkpoint_failed_count"):
        return True
    if str(role or "").strip().lower() in {"br", "policy_breath"}:
        return False
    return False


def _local_refine_low_margin_needs_attention(
    *,
    language: str,
    format_type: str,
    role: str,
    diagnostic_metrics: Mapping[str, float],
) -> bool:
    if not _is_korean_cvc_validation_scope(language=language, format_type=format_type):
        return True
    if str(role or "").strip().lower() in {"br", "policy_breath"}:
        return False
    local_refine_delta = diagnostic_metrics.get("local_refine_max_delta_ms")
    local_refine_margin = diagnostic_metrics.get("local_refine_min_margin")
    if local_refine_delta is None or local_refine_margin is None:
        return True
    if abs(float(local_refine_delta)) > 20.0:
        return True
    if float(local_refine_margin) < 0.001:
        return True
    return False


def _metric_positive(metrics: Mapping[str, float], key: str) -> bool:
    try:
        return float(metrics.get(key, 0.0) or 0.0) > 0.0
    except Exception:
        return False


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _numeric_diagnostics(diagnostics: Mapping[str, object] | object) -> dict[str, float]:
    if not isinstance(diagnostics, Mapping):
        return {}
    out: dict[str, float] = {}
    allowed = {
        "hsmm_decoder_used",
        "hsmm_candidate_prior_count",
        "hsmm_event_prior_state_count",
        "hsmm_event_prior_mapped_count",
        "hsmm_event_prior_ignored_count",
        "hsmm_event_prior_summary_mapped_count",
        "hsmm_event_prior_summary_ignored_count",
        "hsmm_max_duration_z_abs",
        "hsmm_min_selected_vs_best_local_margin",
        "hsmm_min_selected_vs_second_local_margin",
        "hsmm_runtime_prior_count",
        "hsmm_state_count",
        "evidence_event_candidate_count",
        "evidence_event_candidate_flagged_count",
        "evidence_event_candidate_min_margin",
        "evidence_event_candidate_min_feature_agreement",
        "evidence_event_candidate_max_ood",
        "evidence_low_margin_count",
        "evidence_low_feature_agreement_count",
        "evidence_ambiguous_vv_boundary_count",
        "evidence_ambiguous_sonorant_constriction_count",
        "evidence_vv_boundary_low_margin_count",
        "evidence_vv_boundary_low_feature_agreement_count",
        "evidence_vv_boundary_ambiguous_count",
        "evidence_sonorant_constriction_low_margin_count",
        "evidence_sonorant_constriction_low_feature_agreement_count",
        "evidence_sonorant_constriction_ambiguous_count",
        "local_refine_max_delta_ms",
        "local_refine_min_margin",
        "local_refine_changed_anchor_count",
        "local_refine_low_margin_count",
        "local_refine_rejected_slot_boundary_count",
        "timing_clamp_count",
        "timing_clamp_offset_count",
        "timing_clamp_preutterance_count",
        "timing_clamp_overlap_count",
        "timing_clamp_consonant_count",
        "timing_clamp_cutoff_count",
        "timing_clamp_cutoff_sign_count",
        "timing_clamp_max_delta_ms",
        "timing_clamp_large_delta_count",
        "row_warning_island_count_mismatch_count",
        "row_warning_preutterance_slot_shift_count",
        "row_warning_local_gate_low_count",
        "row_warning_missing_local_gate_count",
        "rule_based_inference_count",
        "rule_based_checkpoint_missing_count",
        "rule_based_checkpoint_failed_count",
    }
    for key in allowed:
        if key not in diagnostics:
            continue
        try:
            out[key] = float(diagnostics[key])
        except Exception:
            continue
    return out


def _row_plan_context(diagnostics: Mapping[str, object] | object) -> dict[str, object]:
    context = {
        "slot_index": -1,
        "left_slot_index": -1,
        "right_slot_index": -1,
        "source_row_index": -1,
        "source_timing_trusted": False,
        "role_family": "",
    }
    if not isinstance(diagnostics, Mapping):
        return context
    row_plan = diagnostics.get("row_plan")
    if not isinstance(row_plan, Mapping):
        return context
    context["slot_index"] = _safe_int(row_plan.get("slot_index"), -1)
    context["left_slot_index"] = _safe_int(row_plan.get("left_slot_index"), -1)
    context["right_slot_index"] = _safe_int(row_plan.get("right_slot_index"), -1)
    context["source_row_index"] = _safe_int(row_plan.get("source_row_index"), -1)
    context["source_timing_trusted"] = bool(row_plan.get("source_timing_trusted", False))
    role = str(row_plan.get("role_family", "") or "").strip().lower()
    if role:
        context["role_family"] = role
    return context


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _reason_counts(records: Sequence[ValidationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _reason_counts_by_split(records: Sequence[ValidationRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {split: {} for split in sorted(VALIDATION_SPLITS)}
    for record in records:
        split = str(record.split or "")
        bucket = counts.setdefault(split, {})
        for reason in record.reasons:
            bucket[str(reason)] = bucket.get(str(reason), 0) + 1
    return counts


def _role_counts(records: Sequence[ValidationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        role = str(record.role_family or "other")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _metric_positive_row_counts(records: Sequence[ValidationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for key, value in dict(record.metrics or {}).items():
            parsed = _finite_metric_value(value)
            if parsed is not None and parsed > 0.0:
                counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def _metric_min_values(records: Sequence[ValidationRecord]) -> dict[str, float]:
    out: dict[str, float] = {}
    for record in records:
        for key, value in dict(record.metrics or {}).items():
            parsed = _finite_metric_value(value)
            if parsed is None:
                continue
            name = str(key)
            out[name] = parsed if name not in out else min(out[name], parsed)
    return out


def _metric_max_values(records: Sequence[ValidationRecord]) -> dict[str, float]:
    out: dict[str, float] = {}
    for record in records:
        for key, value in dict(record.metrics or {}).items():
            parsed = _finite_metric_value(value)
            if parsed is None:
                continue
            name = str(key)
            out[name] = parsed if name not in out else max(out[name], parsed)
    return out


def _build_review_session_payload(
    *,
    generated_oto_path: str,
    output_dir: str,
    base: str,
    counts: Mapping[str, int],
    output_paths: Mapping[str, str],
    output_sha256: Mapping[str, str],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    final_path = os.path.abspath(os.path.join(output_dir, f"{base}.final.ini"))
    merge_report_path = os.path.splitext(final_path)[0] + ".merge_report.json"
    gate_report_path = os.path.abspath(os.path.join(output_dir, f"{base}.gate.json"))
    session_status_report_path = os.path.abspath(os.path.join(output_dir, f"{base}.session_status.json"))
    edited_fix_required_path = os.path.abspath(os.path.join(output_dir, f"{base}.edited_fix_required.ini"))
    edited_attention_only_path = os.path.abspath(os.path.join(output_dir, f"{base}.edited_attention_only.ini"))
    validation_path = str(output_paths.get("validation_jsonl", "") or "")
    comparison_path = str(metadata.get("comparison_report_path", "") or "")
    preferred_encoding_source = str(metadata.get("preferred_encoding_source", generated_oto_path) or generated_oto_path)
    metadata_artifacts = _review_session_metadata_artifacts(metadata)
    session_status_args = [
        "python",
        "scripts/dev/mfa_free_oto_review.py",
        "session-status",
        "--review-session",
        str(output_paths.get("review_session", "") or ""),
        "--out",
        session_status_report_path,
    ]
    prepare_edits_report_path = os.path.abspath(os.path.join(output_dir, f"{base}.prepare_edits.json"))
    prepare_edits_args = [
        "python",
        "scripts/dev/mfa_free_oto_review.py",
        "prepare-edits",
        "--review-session",
        str(output_paths.get("review_session", "") or ""),
        "--out",
        prepare_edits_report_path,
    ]
    merge_args = [
        "python",
        "scripts/dev/mfa_free_oto_review.py",
        "merge-session",
        "--review-session",
        str(output_paths.get("review_session", "") or ""),
        "--final",
        final_path,
    ]
    if preferred_encoding_source:
        merge_args.extend(["--preferred-encoding-source", preferred_encoding_source])
    gate_args = [
        "python",
        "scripts/dev/mfa_free_oto_review.py",
        "evaluate-session",
        "--review-session",
        str(output_paths.get("review_session", "") or ""),
        "--out",
        gate_report_path,
        "--gate-target",
        "apply",
        "--fail-on-gate-failure",
    ]
    if comparison_path:
        gate_args.extend(["--comparison-json", comparison_path])
    install_template_args = [
        "python",
        "scripts/dev/mfa_free_oto_review.py",
        "install-session",
        "--review-session",
        str(output_paths.get("review_session", "") or ""),
        "--target-oto",
        "<target-oto-or-voicebank-dir>",
    ]
    return {
        "schema_version": 1,
        "validation_split_schema_version": VALIDATION_SPLIT_SCHEMA_VERSION,
        "state": "review_required",
        "generated_oto_path": os.path.abspath(generated_oto_path),
        "generated_oto_sha256": _sha256_file(generated_oto_path) if os.path.isfile(generated_oto_path) else "",
        "counts": dict(counts),
        "review_queues": {
            SPLIT_FIX_REQUIRED: {
                "path": str(output_paths.get("fix_required", "") or ""),
                "count": int(counts.get("fix_required", 0) or 0),
                "requires_edit_before_apply": True,
            },
            SPLIT_ATTENTION_ONLY: {
                "path": str(output_paths.get("attention_only", "") or ""),
                "count": int(counts.get("attention_only", 0) or 0),
                "requires_review_or_acknowledgement": True,
            },
            SPLIT_CLEAN: {
                "path": str(output_paths.get("clean", "") or ""),
                "count": int(counts.get("clean", 0) or 0),
                "requires_edit_before_apply": False,
            },
        },
        "lineage": {
            "validation_jsonl_path": validation_path,
            "validation_jsonl_sha256": str(output_sha256.get("validation_jsonl", "") or ""),
            "summary_path": str(output_paths.get("summary", "") or ""),
            "summary_sha256": str(output_sha256.get("summary", "") or ""),
            "output_paths": dict(output_paths),
            "output_sha256": dict(output_sha256),
            "metadata_artifacts": metadata_artifacts,
        },
        "apply_policy": {
            "apply_candidate": False,
            "default_on_candidate": False,
            "requires_manual_merge": True,
            "requires_evaluate_gate": True,
            "requires_quality_evidence_for_default_on": True,
            "generated_ini_must_not_replace_voicebank_oto": True,
        },
        "recommended_next_steps": [
            {
                "id": "verify_review_session",
                "description": "Verify generated review artifacts still match this manifest before editing or merging.",
                "command_args": session_status_args,
                "expected_report": session_status_report_path,
            },
            {
                "id": "prepare_review_edits",
                "description": "Copy original review queues into edited files without modifying split artifacts.",
                "command_args": prepare_edits_args,
                "expected_report": prepare_edits_report_path,
            },
            {
                "id": "edit_review_queues",
                "description": "Edit prepared fix_required rows and review prepared attention_only rows before merging.",
                "paths": {
                    "fix_required": str(output_paths.get("fix_required", "") or ""),
                    "attention_only": str(output_paths.get("attention_only", "") or ""),
                    "clean": str(output_paths.get("clean", "") or ""),
                    "review_all": str(output_paths.get("review_all", "") or ""),
                },
                "edited_paths": {
                    "fix_required": edited_fix_required_path,
                    "attention_only": edited_attention_only_path,
                },
            },
            {
                "id": "merge_final",
                "description": "Merge clean rows and edited review rows into a final.ini candidate.",
                "command_args": merge_args,
                "manual_review_required_args": ["--acknowledge-attention-only"]
                if int(counts.get("attention_only", 0) or 0) > 0
                else [],
                "expected_report": merge_report_path,
            },
            {
                "id": "evaluate_apply_gate",
                "description": "Recheck lineage and merge status before any install step.",
                "command_args": gate_args,
                "expected_report": gate_report_path,
            },
            {
                "id": "install_final_dry_run",
                "description": "Dry-run the final.ini install from this review session after the apply gate passes.",
                "command_template_args": install_template_args,
                "requires_operator_target_oto": True,
                "requires_apply_gate": True,
                "dry_run_by_default": True,
            },
        ],
        "metadata": _json_safe(metadata),
    }


def _review_session_metadata_artifacts(metadata: Mapping[str, object]) -> dict[str, object]:
    artifacts: dict[str, object] = {}

    def add(
        name: str,
        *,
        role: str,
        path_key: str,
        sha_key: str = "",
        row_count_key: str = "",
    ) -> None:
        path = str(metadata.get(path_key, "") or "").strip()
        if not path:
            return
        sha = str(metadata.get(sha_key, "") or "").strip().lower() if sha_key else ""
        if not sha and os.path.isfile(path):
            sha = _sha256_file(path)
        item: dict[str, object] = {
            "role": role,
            "path": path,
            "sha256": sha,
            "path_key": path_key,
        }
        if sha_key:
            item["sha256_key"] = sha_key
        if row_count_key:
            try:
                item["row_count"] = int(metadata.get(row_count_key, 0) or 0)
            except Exception:
                item["row_count"] = 0
        artifacts[name] = item

    add("row_plan", role="row_plan_jsonl", path_key="row_plan_path", sha_key="row_plan_sha256", row_count_key="row_plan_rows")
    add("timeline", role="timeline_jsonl", path_key="timeline_path", sha_key="timeline_sha256", row_count_key="timeline_wavs")
    add("timeline_input", role="timeline_jsonl", path_key="timeline_jsonl", sha_key="timeline_jsonl_sha256")
    add("timeline_replay", role="timeline_jsonl", path_key="timeline_out", sha_key="timeline_out_sha256")
    add("evidence", role="evidence_jsonl", path_key="evidence_path", sha_key="evidence_sha256", row_count_key="evidence_wavs")
    add("evidence_input", role="evidence_jsonl", path_key="evidence_jsonl", sha_key="evidence_jsonl_sha256")
    return artifacts


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _finite_metric_value(value: object) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or parsed in {float("inf"), -float("inf")}:
        return None
    return parsed


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_jsonl(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: str, records: Sequence[ValidationRecord]) -> None:
    fields = [
        "row_id",
        "line_index",
        "wav",
        "alias",
        "occurrence_index",
        "split",
        "reasons",
        "role_family",
        "slot_index",
        "left_slot_index",
        "right_slot_index",
        "source_row_index",
        "source_timing_trusted",
        "offset_abs",
        "overlap_abs",
        "pre_abs",
        "fixed_end_abs",
        "cutoff_abs",
        "duration_ms",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = record.to_json_dict()
            row["reasons"] = "|".join(record.reasons)
            writer.writerow({field: row.get(field, "") for field in fields})


__all__ = [
    "SPLIT_ATTENTION_ONLY",
    "SPLIT_CLEAN",
    "SPLIT_FIX_REQUIRED",
    "SplitResult",
    "VALIDATION_SPLIT_SCHEMA_VERSION",
    "VALIDATION_SPLITS",
    "ValidationRecord",
    "build_row_id",
    "classify_alias_role",
    "load_validation_records",
    "load_validation_records_with_errors",
    "split_oto_file",
    "validate_validation_records",
    "validate_oto_lines",
]
