from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Mapping

from core.generation.common.oto_file_utils import apply_alias_suffix
from core.model_context.builder import row_specs_to_model_contexts
from core.model_context.oto_rows import load_row_specs_from_source_oto

from .manifest_audit import infer_filename_phone_sequence
from .oto_adapter import (
    OtoAdapterConfig,
    adapt_template_row,
    anchors_from_prediction,
    assign_template_row_anchors,
    bootstrap_row,
    expected_slots_for_template_rows,
    load_oto_template_rows_alias_only,
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


@dataclass(frozen=True)
class BoundaryEvidence:
    c_onset: float | None
    cv_boundary: float | None
    v_offset: float | None
    nucleus: float | None
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
    guard: NoMfaWorkflowGuard | None = None,
    callback: Callable[[str], None] | None = None,
) -> NoMfaWorkflowReport:
    cfg = guard or NoMfaWorkflowGuard()
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

    if source_oto:
        template_rows = load_oto_template_rows_alias_only(source_oto)
        templates_by_wav: dict[str, list] = {}
        for row in template_rows:
            templates_by_wav.setdefault(str(row.wav).lower(), []).append(row)
        row_specs_by_wav = load_row_specs_from_source_oto(
            source_oto_path=source_oto,
            wav_dir=wav_root,
            language=language,
            format_type=format_type,
            alias_suffix=alias_suffix,
            ignore_source_timing=True,
        )
        contexts = row_specs_to_model_contexts(row_specs_by_wav, audio_mode="duration")
        contexts_by_wav_name: dict[str, list] = {}
        for context in contexts:
            contexts_by_wav_name.setdefault(str(context.wav.wav_name).lower(), []).append(context)

        for wav_key, template_group in templates_by_wav.items():
            wav_path = os.path.join(wav_root, wav_key)
            if not os.path.isfile(wav_path):
                warnings.append(f"missing_wav:{wav_key}")
                continue
            expected_phones = _expected_phones(wav_key, contexts_by_wav_name.get(wav_key, []))
            expected_slots = expected_slots_for_template_rows(template_group, expected_phones)
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
            _collect_prediction_metrics(prediction, slot_scores, warnings, rule_based_per_wav)
            slot_warning_count += len(prediction.slot_result.warnings if prediction.slot_result is not None else ())
            decoded_source = _event_source_for_oto(prediction)
            file_duration_ms = _wav_duration_ms(wav_path, warnings)
            row_anchors = assign_template_row_anchors(
                prediction.posterior,
                decoded_source,
                template_group,
                min_score=0.02,
                use_source_timing_prior=False,
                expected_phones=expected_phones,
            )
            if cfg.one_step_shift_repair_enabled:
                row_anchors = _repair_anchor_monotonicity(row_anchors)
            adapter_config = OtoAdapterConfig(
                mode="template-preserve",
                language=language,
                format_type=format_type,
                alias_type=alias_type,
                vc_pre_max_ms=float(cfg.vc_pre_max_ms),
            )
            for template_row, anchor in zip(template_group, row_anchors):
                adapted = adapt_template_row(
                    template_row,
                    anchor,
                    file_duration_ms=file_duration_ms,
                    config=adapter_config,
                )
                rows_total += 1
                if adapted.anchor is not None:
                    row_anchor_hits += 1
                    anchor_scores.append(float(adapted.anchor.score))
                evidence = _build_boundary_evidence(prediction.posterior, adapted)
                row_conf = _row_confidence(adapted, evidence)
                cv_conf_values.append(_safe_conf(evidence.confidence if evidence.cv_boundary is not None else 0.0))
                nucleus_conf_values.append(_safe_conf(_nucleus_confidence_of(adapted)))
                row_results.append(
                    NoMfaRowResult(
                        wav_key=wav_key,
                        alias=str(adapted.alias or ""),
                        oto_params={
                            "offset": float(adapted.timing.offset),
                            "consonant": float(adapted.timing.consonant),
                            "cutoff": float(adapted.timing.cutoff),
                            "preutterance": float(adapted.timing.preutterance),
                            "overlap": float(adapted.timing.overlap),
                        },
                        confidence=row_conf,
                        boundary_evidence=evidence,
                        warnings=tuple(dict.fromkeys((*adapted.warnings, *evidence.warnings))),
                    )
                )
                all_lines.append(apply_alias_suffix(adapted.format_line(), alias_suffix))
    else:
        wav_files = sorted(Path(wav_root).glob("*.wav"))
        for wav_path in wav_files:
            prediction = predict_wav(
                wav_path,
                checkpoint_path=checkpoint_path,
                expected_phones=infer_filename_phone_sequence(wav_path.name),
                encoder=encoder,
                device=device,
                use_slot_viterbi=use_slot_viterbi,
                language=language,
            )
            _collect_prediction_metrics(prediction, slot_scores, warnings, rule_based_per_wav)
            slot_warning_count += len(prediction.slot_result.warnings if prediction.slot_result is not None else ())
            anchors = anchors_from_prediction(prediction.posterior, _event_source_for_oto(prediction))
            adapted = bootstrap_row(
                wav_path.name,
                wav_path.stem,
                anchors[0] if anchors else None,
                file_duration_ms=_wav_duration_ms(str(wav_path), warnings),
                config=OtoAdapterConfig(
                    mode="bootstrap",
                    language=language,
                    format_type=format_type,
                    alias_type=alias_type,
                    vc_pre_max_ms=float(cfg.vc_pre_max_ms),
                ),
            )
            rows_total += 1
            if adapted.anchor is not None:
                row_anchor_hits += 1
                anchor_scores.append(float(adapted.anchor.score))
            evidence = _build_boundary_evidence(prediction.posterior, adapted)
            row_conf = _row_confidence(adapted, evidence)
            cv_conf_values.append(_safe_conf(evidence.confidence if evidence.cv_boundary is not None else 0.0))
            nucleus_conf_values.append(_safe_conf(_nucleus_confidence_of(adapted)))
            row_results.append(
                NoMfaRowResult(
                    wav_key=str(wav_path.name).lower(),
                    alias=str(adapted.alias or ""),
                    oto_params={
                        "offset": float(adapted.timing.offset),
                        "consonant": float(adapted.timing.consonant),
                        "cutoff": float(adapted.timing.cutoff),
                        "preutterance": float(adapted.timing.preutterance),
                        "overlap": float(adapted.timing.overlap),
                    },
                    confidence=row_conf,
                    boundary_evidence=evidence,
                    warnings=tuple(dict.fromkeys((*adapted.warnings, *evidence.warnings))),
                )
            )
            all_lines.append(apply_alias_suffix(adapted.format_line(), alias_suffix))

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
    if avg_cv_conf < float(cfg.cv_boundary_min_conf):
        guard_reasons.append(f"cv_boundary_confidence_low:{avg_cv_conf:.3f}")
    if avg_nucleus_conf < float(cfg.vowel_nucleus_min_conf):
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
        },
        rows=tuple(row_results),
    )


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


def _expected_phones(wav_name: str, contexts: list) -> list[str]:
    phones = infer_filename_phone_sequence(str(wav_name or ""))
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


def _safe_conf(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _nucleus_confidence_of(adapted) -> float:
    if adapted.anchor is None:
        return 0.0
    return _safe_conf(float(adapted.anchor.nucleus_confidence))


def _build_boundary_evidence(posterior, adapted) -> BoundaryEvidence:
    anchor = adapted.anchor
    if anchor is None:
        return BoundaryEvidence(
            c_onset=None,
            cv_boundary=None,
            v_offset=None,
            nucleus=None,
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
    conf = _safe_conf(0.55 * float(anchor.boundary_confidence) + 0.45 * float(anchor.nucleus_confidence))
    if conf < 0.35:
        warnings.append(f"low_boundary_evidence_confidence:{conf:.3f}")
    return BoundaryEvidence(
        c_onset=c_onset,
        cv_boundary=cv_boundary,
        v_offset=v_offset,
        nucleus=nucleus,
        confidence=conf,
        warnings=tuple(dict.fromkeys((*anchor.warnings, *warnings))),
    )


def _row_confidence(adapted, evidence: BoundaryEvidence) -> float:
    base_anchor = float(adapted.anchor.score) if adapted.anchor is not None else 0.0
    warning_penalty = 0.04 * float(len(adapted.warnings))
    return _safe_conf((0.55 * base_anchor) + (0.45 * evidence.confidence) - warning_penalty)


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
