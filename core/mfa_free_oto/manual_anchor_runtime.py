from __future__ import annotations

import csv
import json
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.model_context.filename import canonicalize_order_token, parse_filename_context
from core.model_context.manual_oto_anchor import (
    classify_manual_oto_alias_role,
    manual_oto_alias_order_terms,
)
from core.model_context.oto_params import wav_duration_ms

from .manual_oto_candidates import (
    ALIAS_FAMILY_VALUES,
    FORMAT_VALUES,
    LANGUAGE_VALUES,
    ROLE_VALUES,
    SCORABLE_ANCHORS,
    ManualOtoCandidateTracks,
    extract_manual_oto_candidate_tracks,
    manual_oto_alias_family,
    manual_oto_candidate_feature_names,
    manual_oto_candidate_features,
)
from .manual_oto_decoder import (
    CandidateOption,
    build_joint_anchor_options,
    decode_joint_anchor_lattice,
)
from .oto_adapter import (
    OtoTemplateRow,
    OtoTiming,
    format_oto_line,
    load_oto_template_rows,
    timing_to_dict,
)
from .reclist_convention import (
    STYLE_FOLLOWING_ONSET_VC,
    analyze_reclist_convention,
    classify_korean_cvvc_alias_role,
    is_korean_cvvc_vowel_term,
)
from .review_overlay import write_review_html
from .types import FramePosterior
from .vowel_island import (
    SlotIslandAssignment,
    VowelIsland,
    VowelIslandDecode,
    assign_slots_to_islands,
    assignment_is_safe,
    extract_vowel_islands,
    fit_islands_to_slot_count,
    island_overlay_events,
)


DEFAULT_MANUAL_ANCHOR_SCORER_ENV = "UTOA_MANUAL_OTO_ANCHOR_SCORER"


@dataclass(frozen=True)
class ManualAnchorScorer:
    path: str
    anchors: tuple[str, ...]
    encoder: str
    models: Mapping[str, object]
    failure_gate_models: Mapping[str, object]
    local_failure_models: Mapping[str, object]
    relative_anchor_priors: Mapping[str, object]
    island_anchor_priors: Mapping[str, object]
    min_score: float = 0.28
    top_k_fallback: int = 24
    scorer_kind: str = ""
    joint_top_per_anchor: int = 6
    joint_max_options_per_row: int = 80
    family_prior_weight: float = 1.0
    joint_family_prior_weight: float = 0.0
    require_primary_transition_token: bool = False
    supervised_gate_threshold: float = 0.80
    local_gate_threshold: float = 0.80
    supervised_gate_thresholds_by_anchor: Mapping[str, float] | None = None
    payload_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManualAnchorPreviewConfig:
    wav_dir: str
    source_oto_path: str
    out_oto_path: str
    out_json_path: str
    overlay_dir: str = ""
    language: str = "japanese"
    format_type: str = "CV"
    limit: int | None = None


def resolve_latest_manual_oto_anchor_scorer(*, base_dir: str = "") -> str:
    candidates: list[str] = []
    env_path = str(os.environ.get(DEFAULT_MANUAL_ANCHOR_SCORER_ENV, "") or "").strip()
    if env_path:
        candidates.append(env_path)
    roots: list[str] = []
    for base in (base_dir, os.getcwd()):
        root = os.path.abspath(os.path.join(str(base or "").strip(), "ml_workspace", "manual_oto_anchor"))
        if root and root not in roots and os.path.isdir(root):
            roots.append(root)
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.lower() == "manual_oto_anchor_scorer.pkl":
                    candidates.append(os.path.join(dirpath, name))
    existing = [os.path.abspath(path) for path in candidates if path and os.path.isfile(path)]
    if not existing:
        return ""
    existing.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return existing[0]


def load_manual_oto_anchor_scorer(path: str | os.PathLike[str]) -> ManualAnchorScorer:
    scorer_path = os.path.abspath(os.fspath(path))
    payload = _load_pickle_with_numpy_random_compat(scorer_path)
    if not isinstance(payload, dict):
        raise ValueError(f"manual OTO anchor scorer payload is not a dict: {scorer_path}")
    schema = str(payload.get("schema_version", "") or "")
    if schema and schema != "manual_oto_anchor_scorer_v1":
        raise ValueError(f"unsupported manual OTO anchor scorer schema: {schema}")
    anchors = tuple(str(anchor) for anchor in (payload.get("anchors") or SCORABLE_ANCHORS) if str(anchor) in SCORABLE_ANCHORS)
    if not anchors:
        anchors = SCORABLE_ANCHORS
    models = dict(payload.get("models") or {})
    warnings: list[str] = []
    missing = [anchor for anchor in anchors if anchor not in models]
    if missing:
        warnings.append("missing_models:" + ",".join(missing))
    expected_features = manual_oto_candidate_feature_names()
    payload_features = list(payload.get("feature_names") or [])
    if payload_features and payload_features != expected_features:
        warnings.append("feature_schema_mismatch")
    return ManualAnchorScorer(
        path=scorer_path,
        anchors=anchors,
        encoder=str(payload.get("encoder") or "acoustic"),
        models=models,
        failure_gate_models=dict(payload.get("failure_gate_models") or {}),
        local_failure_models=dict(payload.get("local_failure_models") or {}),
        relative_anchor_priors=dict(payload.get("relative_anchor_gap_priors") or payload.get("relative_anchor_priors") or {}),
        island_anchor_priors=dict(payload.get("island_anchor_position_priors") or payload.get("island_anchor_priors") or {}),
        min_score=float(payload.get("min_score", 0.28) or 0.28),
        top_k_fallback=int(payload.get("top_k_fallback", 24) or 24),
        scorer_kind=str(payload.get("scorer_kind") or ""),
        joint_top_per_anchor=int(payload.get("joint_top_per_anchor", 6) or 6),
        joint_max_options_per_row=int(payload.get("joint_max_options_per_row", 80) or 80),
        family_prior_weight=float(payload.get("family_prior_weight", 1.0) or 1.0),
        joint_family_prior_weight=float(payload.get("joint_family_prior_weight", 0.0) or 0.0),
        require_primary_transition_token=bool(payload.get("require_primary_transition_token", False)),
        supervised_gate_threshold=float(payload.get("supervised_gate_threshold", 0.80) or 0.80),
        local_gate_threshold=float(payload.get("local_gate_threshold", 0.80) or 0.80),
        supervised_gate_thresholds_by_anchor={
            str(key): float(value)
            for key, value in dict(payload.get("supervised_gate_thresholds_by_anchor") or {}).items()
        },
        payload_warnings=tuple(warnings),
    )


def _load_pickle_with_numpy_random_compat(path: str) -> object:
    _install_numpy_random_pickle_compat()
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (TypeError, ValueError) as exc:
        if not _looks_like_numpy_random_pickle_error(exc):
            raise
    return _load_pickle_with_numpy_random_proxy(path)


def _looks_like_numpy_random_pickle_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "BitGenerator" in message
        or "state must be a dict" in message
        or "state must be for a" in message
        or "numpy.random" in message
    )


def _install_numpy_random_pickle_compat() -> None:
    """Accept NumPy RNG class objects in scorer pickles made on newer NumPy.

    Some runtime environments still ship a `numpy.random._pickle` helper that
    only accepts BitGenerator names as strings. Newer pickles may pass the
    class object itself, which otherwise fails before the sklearn models load.
    """
    try:
        import numpy.random._pickle as random_pickle
    except Exception:
        return
    original = getattr(random_pickle, "__bit_generator_ctor", None)
    if not callable(original) or bool(getattr(original, "_utoa_accepts_bitgen_class", False)):
        return
    bit_generators = getattr(random_pickle, "BitGenerators", {})

    def _compat_bit_generator_ctor(bit_generator="MT19937"):
        if isinstance(bit_generator, type):
            try:
                return bit_generator()
            except Exception:
                name = str(getattr(bit_generator, "__name__", "") or "")
                if name and name in bit_generators:
                    return bit_generators[name]()
        try:
            return original(bit_generator)
        except ValueError:
            name = str(getattr(bit_generator, "__name__", "") or "")
            if name and name in bit_generators:
                return bit_generators[name]()
            raise

    _compat_bit_generator_ctor._utoa_accepts_bitgen_class = True  # type: ignore[attr-defined]
    random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor


def _load_pickle_with_numpy_random_proxy(path: str) -> object:
    try:
        import numpy.random._pickle as random_pickle
        from numpy.random import BitGenerator, Generator, PCG64
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)

    bit_generators = getattr(random_pickle, "BitGenerators", {})
    original_bit_generator_ctor = getattr(random_pickle, "__bit_generator_ctor", None)

    class _CompatPCG64:
        _utoa_numpy_random_proxy = True

        def __init__(self, *args, **kwargs):
            self._state = None

        def __setstate__(self, state):
            self._state = state

        def __getstate__(self):
            return self._state

    def _as_real_bit_generator(bit_generator="MT19937"):
        if isinstance(bit_generator, BitGenerator):
            return bit_generator
        if isinstance(bit_generator, type):
            try:
                return bit_generator()
            except Exception:
                name = str(getattr(bit_generator, "__name__", "") or "")
                if name and name in bit_generators:
                    return bit_generators[name]()
        if bool(getattr(bit_generator, "_utoa_numpy_random_proxy", False)):
            real = PCG64()
            state = getattr(bit_generator, "_state", None)
            if isinstance(state, dict):
                try:
                    real.state = state
                except Exception:
                    try:
                        real.__setstate__(state)
                    except Exception:
                        pass
            return real
        if callable(original_bit_generator_ctor):
            try:
                return original_bit_generator_ctor(bit_generator)
            except ValueError:
                name = str(getattr(bit_generator, "__name__", "") or "")
                if name and name in bit_generators:
                    return bit_generators[name]()
                raise
        return PCG64()

    def _compat_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
        if isinstance(bit_generator_name, Generator):
            return bit_generator_name
        if isinstance(bit_generator_name, BitGenerator) or bool(
            getattr(bit_generator_name, "_utoa_numpy_random_proxy", False)
        ):
            return Generator(_as_real_bit_generator(bit_generator_name))
        ctor = bit_generator_ctor or _as_real_bit_generator
        return Generator(ctor(bit_generator_name))

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy.random._pcg64" and name == "PCG64":
                return _CompatPCG64
            if module == "numpy.random._pickle" and name == "__bit_generator_ctor":
                return _as_real_bit_generator
            if module == "numpy.random._pickle" and name == "__generator_ctor":
                return _compat_generator_ctor
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return _CompatUnpickler(handle).load()


def generate_manual_oto_anchor_preview(
    scorer_path: str,
    config: ManualAnchorPreviewConfig,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    scorer = load_manual_oto_anchor_scorer(scorer_path)
    source_oto = os.path.abspath(str(config.source_oto_path or "").strip())
    if not os.path.isfile(source_oto):
        raise FileNotFoundError(f"source oto.ini is required for manual anchor preview: {source_oto}")
    wav_dir = os.path.abspath(str(config.wav_dir or "").strip())
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    template_rows = load_oto_template_rows(source_oto)
    if not template_rows:
        raise ValueError(f"source oto.ini has no usable rows: {source_oto}")
    rows_by_wav_name: dict[str, list[OtoTemplateRow]] = defaultdict(list)
    for row in template_rows:
        rows_by_wav_name[row.wav.lower()].append(row)

    wav_paths = _collect_wav_paths(wav_dir, rows_by_wav_name, limit=config.limit)
    if not wav_paths:
        raise ValueError("no matching WAV files found for source oto.ini rows")
    wav_total = len(wav_paths)
    _emit_progress(progress_callback, f"progress 0/{wav_total} wav 준비")

    out_lines: list[str] = []
    records: list[dict[str, object]] = []
    overlay_root = Path(config.overlay_dir) if str(config.overlay_dir or "").strip() else None
    if overlay_root:
        overlay_root.mkdir(parents=True, exist_ok=True)

    for wav_index, wav_path in enumerate(wav_paths, start=1):
        wav_name = os.path.basename(wav_path)
        _emit_progress(progress_callback, f"progress {wav_index - 1}/{wav_total} wav 시작: {wav_name}")
        source_rows = rows_by_wav_name.get(wav_name.lower()) or []
        if not source_rows:
            _emit_progress(progress_callback, f"progress {wav_index}/{wav_total} wav 건너뜀: {wav_name}")
            continue
        duration_ms = wav_duration_ms(wav_path, missing_value=0.0)
        if duration_ms <= 0.0:
            records.append({"wav": wav_name, "wav_path": wav_path, "error": "wav_duration_unavailable"})
            _emit_progress(progress_callback, f"progress {wav_index}/{wav_total} wav 오류: {wav_name}")
            continue
        runtime_rows = _runtime_rows_for_wav(
            wav_path,
            source_rows,
            duration_ms=duration_ms,
            language=config.language,
            format_type=config.format_type,
            voicebank_id=os.path.basename(os.path.normpath(wav_dir)),
        )
        try:
            tracks = extract_manual_oto_candidate_tracks(
                wav_path,
                encoder=scorer.encoder,
                min_score=scorer.min_score,
                top_k_fallback=scorer.top_k_fallback,
            )
        except Exception as exc:
            records.append({"wav": wav_name, "wav_path": wav_path, "error": f"candidate_extract_failed:{exc}"})
            _emit_progress(progress_callback, f"progress {wav_index}/{wav_total} wav 오류: {wav_name}")
            continue

        wav_result = _predict_wav_rows(runtime_rows, tracks, scorer)
        generated_rows = wav_result["generated_oto_rows"]
        out_lines.extend(str(item["line"]) for item in generated_rows)
        record = {
            "wav": wav_name,
            "wav_path": wav_path,
            "duration_ms": duration_ms,
            "rows": wav_result["row_records"],
            "generated_oto_rows": generated_rows,
            "island_decode": wav_result["island_decode"],
            "reclist_convention": runtime_rows[0].get("reclist_convention", {}) if runtime_rows else {},
            "scorer_payload_warnings": list(scorer.payload_warnings),
        }
        records.append(record)
        if overlay_root:
            _write_manual_anchor_overlay(overlay_root, wav_path, runtime_rows, tracks, wav_result)
        _emit_progress(progress_callback, f"progress {wav_index}/{wav_total} wav 완료: {wav_name}")

    out_oto = Path(config.out_oto_path)
    out_oto.parent.mkdir(parents=True, exist_ok=True)
    out_oto.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    quality_sidecars = _write_preview_quality_sidecars(out_oto, records)
    out_json = Path(config.out_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "manual_oto_anchor_preview_v1",
        "scorer": scorer.path,
        "scorer_kind": scorer.scorer_kind,
        "encoder": scorer.encoder,
        "anchors": list(scorer.anchors),
        "language": config.language,
        "format_type": config.format_type,
        "source_oto": source_oto,
        "wav_dir": wav_dir,
        "reclist_convention_summary": _reclist_convention_summary(records),
        "quality_summary": quality_sidecars["summary"],
        "quality_sidecars": {
            "safe_oto": quality_sidecars["safe_oto"],
            "needs_review_oto": quality_sidecars["needs_review_oto"],
            "needs_review_csv": quality_sidecars["needs_review_csv"],
        },
        "rows": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "wav_count": len(wav_paths),
        "oto_rows": len(out_lines),
        "out_oto": str(out_oto),
        "out_json": str(out_json),
        "quality_summary": quality_sidecars["summary"],
        "quality_sidecars": {
            "safe_oto": quality_sidecars["safe_oto"],
            "needs_review_oto": quality_sidecars["needs_review_oto"],
            "needs_review_csv": quality_sidecars["needs_review_csv"],
        },
        "overlay_dir": str(overlay_root) if overlay_root else "",
        "scorer": scorer.path,
        "scorer_payload_warnings": list(scorer.payload_warnings),
    }


def _reclist_convention_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    style_counts: dict[str, int] = {}
    low_confidence = 0
    warnings: dict[str, int] = {}
    for record in records:
        convention = record.get("reclist_convention") if isinstance(record, Mapping) else None
        if not isinstance(convention, Mapping):
            continue
        style = str(convention.get("vc_pre_style") or "unknown")
        style_counts[style] = style_counts.get(style, 0) + 1
        try:
            confidence = float(convention.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < 0.65:
            low_confidence += 1
        for warning in convention.get("warnings", []) or []:
            key = str(warning)
            warnings[key] = warnings.get(key, 0) + 1
    return {
        "style_counts": style_counts,
        "low_confidence_wavs": int(low_confidence),
        "warnings": warnings,
    }


_PREVIEW_REVIEW_WARNINGS = frozenset(
    {
        "island_count_mismatch",
        "missing_island",
        "missing_islands",
        "low_dp_margin",
        "low_assignment_confidence",
        "preutterance_island_mismatch",
        "fallback_preutterance",
        "fallback_offset_gap",
        "fallback_fixed_end_gap",
        "transition_decoder_large_refine",
        "role_landmark_large_refine",
        "low_cv_landmark_confidence",
        "low_tail_landmark_confidence",
        "low_vv_landmark_confidence",
        "vowel_onset_far_from_island_start",
        "vowel_offset_far_from_island_end",
        "vv_transition_outside_expected_span",
        "reclist_convention_low_confidence",
        "nucleus_before_vowel_onset",
    }
)
_PREVIEW_REVIEW_WARNING_PREFIXES = ("preutterance_slot_shift:",)
_PREVIEW_ADVISORY_WARNINGS = frozenset({"island_count_repaired"})
_PREVIEW_ADVISORY_SUBSTRINGS = ("local_gate_low", "missing_local_gate")


def _classify_preview_row_quality(
    row: Mapping[str, object],
    *,
    absolute: Mapping[str, object],
    warnings: Sequence[object],
    local_gate_warnings: Sequence[object],
    sources: Mapping[str, object],
    acoustic_landmarks: Mapping[str, object] | None,
    duration_ms: float,
) -> dict[str, object]:
    reasons: list[str] = []
    advisory: list[str] = []

    for warning in warnings:
        text = str(warning or "").strip()
        if not text:
            continue
        if _preview_warning_requires_review(text):
            reasons.append(_preview_quality_reason(text))
        elif _preview_warning_is_advisory(text):
            advisory.append(text)
    for warning in local_gate_warnings:
        text = str(warning or "").strip()
        if text:
            advisory.append(text)

    role = str(row.get("alias_role", "") or "").strip().lower()
    alias = str(row.get("alias", "") or "")
    format_type = str(row.get("format_type", "") or "").strip().lower()
    family = manual_oto_alias_family(alias)
    if not alias.strip() or role in {"invalid", "other"}:
        reasons.append("invalid_alias_requires_review")
    if family == "breath_silence" or role in {"br", "sil"}:
        reasons.append("breath_or_silence_requires_review")
    if _is_korean_cvvc_row(row) and role == "cv":
        reasons.append("korean_cvvc_cv_requires_review")
    if _is_japanese_row(row) and format_type == "vcv":
        reasons.append("japanese_vcv_requires_review")
    if role == "v":
        reasons.append("role_v_requires_review")
    if role == "vv":
        reasons.append("role_vv_requires_review")
    if _alias_has_high_risk_tail_marker(alias, role=role):
        reasons.append("tail_marker_alias_requires_review")

    landmark_source = ""
    if isinstance(acoustic_landmarks, Mapping):
        landmark_source = str(acoustic_landmarks.get("source", "") or "")
    elif role in {"cv", "vc", "v", "vv"} and family != "breath_silence":
        reasons.append("missing_acoustic_landmark")
    if role == "vc" and landmark_source.startswith("acoustic_tail_"):
        reasons.append("tail_landmark_requires_review")

    _append_timing_sanity_reasons(reasons, absolute=absolute, duration_ms=duration_ms)

    status = "needs_review" if reasons else "safe"
    out = {
        "status": status,
        "reasons": _dedupe_texts(reasons),
        "advisory": _dedupe_texts(advisory),
    }
    if any("local_gate" in item for item in out["advisory"]):
        out["local_gate_policy"] = "advisory"
    return out


def _preview_warning_requires_review(warning: str) -> bool:
    if warning in _PREVIEW_REVIEW_WARNINGS:
        return True
    return any(warning.startswith(prefix) for prefix in _PREVIEW_REVIEW_WARNING_PREFIXES)


def _preview_warning_is_advisory(warning: str) -> bool:
    if warning in _PREVIEW_ADVISORY_WARNINGS:
        return True
    return any(part in warning for part in _PREVIEW_ADVISORY_SUBSTRINGS)


def _preview_quality_reason(warning: str) -> str:
    if warning.startswith("preutterance_slot_shift:"):
        return "preutterance_slot_shift"
    return warning


def _alias_has_high_risk_tail_marker(alias: str, *, role: str) -> bool:
    if role not in {"vc", "v", "vv"}:
        return False
    text = str(alias or "").strip()
    if not text:
        return False
    if text.endswith(("-", "*")):
        return True
    return any(marker in text for marker in ("R", "H", "*"))


def _append_timing_sanity_reasons(
    reasons: list[str],
    *,
    absolute: Mapping[str, object],
    duration_ms: float,
) -> None:
    offset_abs = _float_or_none(absolute.get("offset_abs"))
    overlap_abs = _float_or_none(absolute.get("overlap_abs"))
    preutterance_abs = _float_or_none(absolute.get("preutterance_abs"))
    fixed_end_abs = _float_or_none(absolute.get("consonant_abs"))
    cutoff_abs = _float_or_none(absolute.get("cutoff_abs"))
    duration = max(1.0, float(duration_ms))
    if overlap_abs is not None and preutterance_abs is not None and overlap_abs > preutterance_abs + 1e-3:
        reasons.append("overlap_after_preutterance")
    if preutterance_abs is not None and fixed_end_abs is not None and preutterance_abs > fixed_end_abs + 1e-3:
        reasons.append("preutterance_after_fixed_end")
    if fixed_end_abs is not None and cutoff_abs is not None and cutoff_abs <= fixed_end_abs + 1e-3:
        reasons.append("cutoff_before_fixed_end")
    if offset_abs is not None and preutterance_abs is not None and offset_abs > preutterance_abs + 1e-3:
        reasons.append("offset_after_preutterance")
    if cutoff_abs is not None and (cutoff_abs < -1e-3 or cutoff_abs > duration + 1e-3):
        reasons.append("cutoff_out_of_duration")


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _dedupe_texts(values: Sequence[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _preview_sidecar_paths(out_oto: Path) -> dict[str, Path]:
    suffix = out_oto.suffix or ".ini"
    stem = out_oto.stem if out_oto.suffix else out_oto.name
    return {
        "safe_oto": out_oto.with_name(f"{stem}.safe{suffix}"),
        "needs_review_oto": out_oto.with_name(f"{stem}.needs_review{suffix}"),
        "needs_review_csv": out_oto.with_name(f"{stem}.needs_review.csv"),
    }


def _write_preview_quality_sidecars(out_oto: Path, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paths = _preview_sidecar_paths(out_oto)
    safe_lines: list[str] = []
    needs_review_lines: list[str] = []
    needs_review_rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for generated in record.get("generated_oto_rows", []) or []:
            if not isinstance(generated, Mapping):
                continue
            quality = generated.get("quality") if isinstance(generated.get("quality"), Mapping) else {}
            status = str(quality.get("status", "") or "").strip().lower()
            line = str(generated.get("line", "") or "")
            if not line:
                continue
            if status == "safe":
                safe_lines.append(line)
            else:
                needs_review_lines.append(line)
                needs_review_rows.append(
                    {
                        "wav": generated.get("wav", record.get("wav", "")),
                        "alias": generated.get("alias", ""),
                        "alias_role": generated.get("alias_role", ""),
                        "status": status or "needs_review",
                        "reasons": "|".join(str(item) for item in quality.get("reasons", []) or []),
                        "advisory": "|".join(str(item) for item in quality.get("advisory", []) or []),
                        "warnings": "|".join(str(item) for item in generated.get("warnings", []) or []),
                        "sources": json.dumps(generated.get("sources", {}) or {}, ensure_ascii=False, sort_keys=True),
                        "line": line,
                    }
                )
    paths["safe_oto"].write_text("\n".join(safe_lines) + ("\n" if safe_lines else ""), encoding="utf-8")
    paths["needs_review_oto"].write_text(
        "\n".join(needs_review_lines) + ("\n" if needs_review_lines else ""),
        encoding="utf-8",
    )
    _write_needs_review_csv(paths["needs_review_csv"], needs_review_rows)
    summary = _preview_quality_summary(records)
    return {
        "safe_oto": str(paths["safe_oto"]),
        "needs_review_oto": str(paths["needs_review_oto"]),
        "needs_review_csv": str(paths["needs_review_csv"]),
        "summary": summary,
    }


def _write_needs_review_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["wav", "alias", "alias_role", "status", "reasons", "advisory", "warnings", "sources", "line"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _preview_quality_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    advisory_counts: dict[str, int] = {}
    total = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for generated in record.get("generated_oto_rows", []) or []:
            if not isinstance(generated, Mapping):
                continue
            total += 1
            quality = generated.get("quality") if isinstance(generated.get("quality"), Mapping) else {}
            status = str(quality.get("status", "") or "needs_review")
            status_counts[status] = status_counts.get(status, 0) + 1
            for reason in quality.get("reasons", []) or []:
                key = str(reason)
                reason_counts[key] = reason_counts.get(key, 0) + 1
            for advisory in quality.get("advisory", []) or []:
                key = str(advisory).split(":", 1)[0]
                advisory_counts[key] = advisory_counts.get(key, 0) + 1
    return {
        "total_rows": int(total),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "advisory_counts": advisory_counts,
    }


def _emit_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def _collect_wav_paths(
    wav_dir: str,
    rows_by_wav_name: Mapping[str, Sequence[OtoTemplateRow]],
    *,
    limit: int | None = None,
) -> list[str]:
    out: list[str] = []
    for key in sorted(rows_by_wav_name):
        rows = rows_by_wav_name.get(key) or ()
        name = str(rows[0].wav if rows else key)
        path = os.path.join(wav_dir, name)
        if os.path.isfile(path):
            out.append(os.path.abspath(path))
    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def _runtime_rows_for_wav(
    wav_path: str,
    source_rows: Sequence[OtoTemplateRow],
    *,
    duration_ms: float,
    language: str,
    format_type: str,
    voicebank_id: str = "",
) -> list[dict[str, object]]:
    wav_name = os.path.basename(wav_path)
    filename = parse_filename_context(wav_name, language=language, format_type=format_type)
    slot_count = max(1, len(source_rows))
    source_roles = [
        _runtime_alias_role(
            source.alias,
            language=language,
            format_type=format_type,
        )
        for source in source_rows
    ]
    reclist_convention = analyze_reclist_convention(
        [source.alias for source in source_rows],
        roles=source_roles,
        language=language,
        format_type=format_type,
    )
    reclist_convention_json = reclist_convention.to_json()
    rows: list[dict[str, object]] = []
    for idx, source in enumerate(source_rows):
        role = source_roles[idx]
        rows.append(
            {
                "wav_path": os.path.abspath(wav_path),
                "wav_name": wav_name,
                "alias": source.alias,
                "alias_norm": str(source.alias or "").strip().lower(),
                "alias_role": role,
                "format_type": str(format_type or "").strip().lower(),
                "language": str(language or "").strip().lower(),
                "slot_index": int(idx),
                "slot_count": int(slot_count),
                "slot_pos_norm": _centered_slot_pos({"slot_index": idx, "slot_count": slot_count}),
                "duration_ms": float(duration_ms),
                "filename_tokens": list(filename.tokens),
                "filename_canonical_tokens": list(filename.canonical_tokens),
                "voicebank_id": str(voicebank_id or ""),
                "source_order": int(idx),
                "vc_pre_style": reclist_convention.vc_pre_style,
                "reclist_convention": reclist_convention_json,
            }
        )
    return rows


def _runtime_alias_role(
    alias: object,
    *,
    language: str,
    format_type: str,
) -> str:
    role = classify_manual_oto_alias_role(
        alias,
        language=language,
        format_type=format_type,
    )
    language_text = str(language or "").strip().lower()
    format_text = str(format_type or "").strip().lower()
    if language_text not in {"ko", "kr", "korean"} or format_text != "cvvc":
        return role
    return classify_korean_cvvc_alias_role(
        alias,
        language=language,
        format_type=format_type,
        fallback=role,
    )


def _korean_cvvc_plain_vowel_term(term: object) -> bool:
    return is_korean_cvvc_vowel_term(term)


def _predict_wav_rows(
    rows: list[dict[str, object]],
    tracks: ManualOtoCandidateTracks,
    scorer: ManualAnchorScorer,
) -> dict[str, object]:
    anchors = tuple(anchor for anchor in scorer.anchors if anchor in scorer.models)
    constrained = _decode_lattice_predictions(
        rows,
        tracks,
        scorer,
        rows_with_pos=_row_slot_ordered_rows(rows),
        family_prior_weight=scorer.family_prior_weight,
        slot_order_penalty=1.8,
        slot_transition_penalty=2.4,
        slot_backward_penalty=4.5,
    )
    filename_slot = _decode_lattice_predictions(
        rows,
        tracks,
        scorer,
        rows_with_pos=_filename_ordered_rows(
            rows,
            require_primary_transition_token=scorer.require_primary_transition_token,
        ),
        family_prior_weight=0.0,
        slot_order_penalty=1.4,
        slot_transition_penalty=2.4,
        slot_backward_penalty=4.5,
    )
    joint = _decode_lattice_predictions(
        rows,
        tracks,
        scorer,
        rows_with_pos=_filename_ordered_rows(
            rows,
            require_primary_transition_token=scorer.require_primary_transition_token,
        ),
        family_prior_weight=scorer.joint_family_prior_weight,
        slot_order_penalty=1.0,
        slot_transition_penalty=1.6,
        slot_backward_penalty=3.0,
    )

    slot_count = _vowel_island_slot_count(rows)
    raw_islands = extract_vowel_islands(tracks)
    islands = fit_islands_to_slot_count(tracks, raw_islands, slot_count=slot_count)
    island_decode = assign_slots_to_islands(islands, slot_count=slot_count, duration_ms=float(tracks.duration_ms))
    if len(islands) != int(slot_count):
        island_decode = _mark_island_count_mismatch(island_decode)
    elif len(raw_islands) != int(slot_count):
        island_decode = _mark_island_count_repaired(island_decode)

    generated_rows: list[dict[str, object]] = []
    row_records: list[dict[str, object]] = []
    for row in rows:
        row_result = _predict_row(
            row,
            tracks,
            scorer,
            anchors=anchors,
            constrained=constrained,
            filename_slot=filename_slot,
            joint=joint,
            island_decode=island_decode,
            slot_count=slot_count,
        )
        generated_rows.append(row_result["generated_row"])
        row_records.append(row_result["record"])
    return {
        "generated_oto_rows": generated_rows,
        "row_records": row_records,
        "island_decode": _island_decode_json(island_decode),
    }


def _decode_lattice_predictions(
    rows: list[dict[str, object]],
    tracks: ManualOtoCandidateTracks,
    scorer: ManualAnchorScorer,
    *,
    rows_with_pos: Sequence[tuple[dict[str, object], float]],
    family_prior_weight: float,
    slot_order_penalty: float,
    slot_transition_penalty: float,
    slot_backward_penalty: float,
) -> dict[tuple[int, str], dict[str, object]]:
    anchors = tuple(anchor for anchor in scorer.anchors if anchor in scorer.models)
    if not anchors:
        return {}
    options_by_row: list[list[object]] = []
    row_refs: list[dict[str, object]] = []
    for row, slot_pos in rows_with_pos:
        feature_row = _row_with_slot_pos(row, slot_pos)
        options_by_anchor: dict[str, list[CandidateOption]] = {}
        for anchor in anchors:
            model = scorer.models.get(anchor)
            indices = [int(idx) for idx in tracks.candidate_indices.get(anchor, ()) if 0 <= int(idx) < tracks.times_ms.size]
            if model is None or not indices:
                continue
            features = np.stack(
                [
                    manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                    for idx in indices
                ],
                axis=0,
            ).astype(np.float32)
            scores = _score_candidates(model, features)
            time_lookup = _time_norms(tracks, indices)
            adjusted_scores: list[float] = []
            for pos, idx in enumerate(indices):
                adjusted_scores.append(
                    float(scores[pos])
                    + _family_anchor_prior_score(
                        row,
                        anchor,
                        candidate_time_norm=float(time_lookup.get(idx, 0.0)),
                        slot_pos_norm=float(slot_pos),
                        weight=float(family_prior_weight),
                    )
                )
            options_by_anchor[anchor] = [
                CandidateOption(
                    candidate_index=int(idx),
                    time_ms=float(tracks.times_ms[idx]),
                    score=float(adjusted_scores[pos]),
                    order_norm=float(time_lookup.get(idx, 0.0)),
                    slot_pos_norm=float(slot_pos),
                )
                for pos, idx in enumerate(indices)
            ]
        options = build_joint_anchor_options(
            options_by_anchor,
            anchors=anchors,
            top_per_anchor=scorer.joint_top_per_anchor,
            max_options=scorer.joint_max_options_per_row,
            slot_order_penalty=float(slot_order_penalty),
        )
        options_by_row.append(options)
        row_refs.append(row)
    decoded = decode_joint_anchor_lattice(
        options_by_row,
        slot_order_penalty=float(slot_order_penalty),
        slot_transition_penalty=float(slot_transition_penalty),
        slot_backward_penalty=float(slot_backward_penalty),
    )
    out: dict[tuple[int, str], dict[str, object]] = {}
    for row, selected in zip(row_refs, decoded, strict=False):
        if selected is None:
            continue
        for anchor in anchors:
            selected_idx = selected.anchor_indices.get(anchor)
            if selected_idx is None:
                continue
            out[(id(row), anchor)] = {
                "pred_ms": float(tracks.times_ms[int(selected_idx)]),
                "joint_score": float(selected.score),
                "slot_pos_norm": float(selected.slot_pos_norm),
                "time_order_norm": float(selected.time_order_norm),
                "anchor": str(anchor),
                "candidate_index": int(selected_idx),
            }
    return out


def _predict_row(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    scorer: ManualAnchorScorer,
    *,
    anchors: tuple[str, ...],
    constrained: Mapping[tuple[int, str], dict[str, object]],
    filename_slot: Mapping[tuple[int, str], dict[str, object]],
    joint: Mapping[tuple[int, str], dict[str, object]],
    island_decode: VowelIslandDecode,
    slot_count: int,
) -> dict[str, object]:
    duration = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    slot_index = _row_vowel_island_slot_index(row, slot_count)
    assignment = _island_assignment_for_row(row, island_decode.assignments, slot_count)
    island = _island_for_assignment(assignment, island_decode.islands)
    island_context = _manual_anchor_island_context(assignment, island_decode.islands)
    predictions: dict[str, float] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    acoustic_landmarks = _role_acoustic_landmarks(row, tracks, island, assignment, island_decode.islands)
    reclist_convention = row.get("reclist_convention") if isinstance(row, Mapping) else {}

    if island is None:
        warnings.append("missing_island")
    elif not assignment_is_safe(assignment, island):
        warnings.extend(str(item) for item in getattr(assignment, "warnings", ()) or ())
    if isinstance(acoustic_landmarks, Mapping):
        warnings.extend(str(item) for item in acoustic_landmarks.get("warnings", []) or [])
    if isinstance(reclist_convention, Mapping):
        try:
            convention_confidence = float(reclist_convention.get("confidence", 0.0) or 0.0)
        except Exception:
            convention_confidence = 0.0
        if str(row.get("alias_role", "") or "").strip().lower() == "vc" and convention_confidence < 0.65:
            warnings.append("reclist_convention_low_confidence")
        warnings.extend(f"reclist_convention:{item}" for item in reclist_convention.get("warnings", []) or [])

    pre_prediction = constrained.get((id(row), "preutterance"))
    pre_safe_prob = _supervised_safe_probability(
        row,
        "preutterance",
        pre_prediction,
        scorer=scorer,
        filename_pred=filename_slot.get((id(row), "preutterance")),
        joint_pred=joint.get((id(row), "preutterance")),
    )
    pre_ms = None
    role = str(row.get("alias_role", "") or "").strip().lower()
    if _is_japanese_row(row) and role == "vv" and island is not None:
        fallback_pre = _default_island_preutterance_ms(row, island)
        landmark_pre = _landmark_preutterance_ms(row, acoustic_landmarks, island, fallback_ms=fallback_pre)
        if landmark_pre is not None:
            pre_ms = landmark_pre
            sources["preutterance"] = str(acoustic_landmarks.get("preutterance_source", "role_landmark"))
            if abs(float(landmark_pre) - float(fallback_pre)) > 12.0:
                warnings.append("role_landmark_refined_preutterance")
            if abs(float(landmark_pre) - float(fallback_pre)) > 90.0:
                warnings.append("role_landmark_large_refine")
    deterministic_pre = _korean_cvvc_island_preutterance_ms(row, island)
    if pre_ms is None and deterministic_pre is not None:
        transition_pre = _korean_cvvc_transition_preutterance_ms(
            row,
            island,
            acoustic_landmarks,
            fallback_ms=deterministic_pre,
        )
        if transition_pre is not None:
            pre_ms, pre_source = transition_pre
            sources["preutterance"] = pre_source
            if abs(float(pre_ms) - float(deterministic_pre)) > 12.0:
                warnings.append("transition_decoder_refined_preutterance")
            if abs(float(pre_ms) - float(deterministic_pre)) > 90.0:
                warnings.append("transition_decoder_large_refine")
        else:
            landmark_pre = _landmark_preutterance_ms(row, acoustic_landmarks, island, fallback_ms=deterministic_pre)
            if landmark_pre is not None:
                pre_ms = landmark_pre
                sources["preutterance"] = str(acoustic_landmarks.get("preutterance_source", "role_landmark"))
                if abs(float(landmark_pre) - float(deterministic_pre)) > 12.0:
                    warnings.append("role_landmark_refined_preutterance")
                if abs(float(landmark_pre) - float(deterministic_pre)) > 90.0:
                    warnings.append("role_landmark_large_refine")
        if pre_ms is None:
            pre_ms = deterministic_pre
            sources["preutterance"] = "korean_cvvc_vowel_island"
    if pre_ms is None and pre_prediction is not None:
        pred = float(pre_prediction.get("pred_ms", 0.0) or 0.0)
        delta = _slot_shift_delta(row, pred_ms=pred, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        prediction_matches_island = _prediction_matches_island(
            row,
            "preutterance",
            pred_ms=pred,
            island=island,
        )
        if (delta == 0 or _passes_gate("preutterance", pre_safe_prob, scorer)) and prediction_matches_island:
            pre_ms = pred
            sources["preutterance"] = (
                "constrained_gate"
                if _passes_gate("preutterance", pre_safe_prob, scorer)
                else "constrained_slot_exact"
            )
        else:
            if not prediction_matches_island:
                warnings.append("preutterance_island_mismatch")
            else:
                warnings.append(f"preutterance_slot_shift:{_slot_shift_bucket(delta)}")
    if pre_ms is None and island is not None:
        island_pred = _predict_island_anchor_ms(
            row,
            tracks,
            anchor="preutterance",
            island=island,
            relative_anchor_priors=dict(scorer.relative_anchor_priors),
            island_anchor_priors=dict(scorer.island_anchor_priors),
        )
        if island_pred is not None:
            pre_ms = _slot_guarded_anchor_ms(
                row,
                tracks,
                "preutterance",
                float(island_pred),
                slot_count=slot_count,
                slot_index=slot_index,
            )
            sources["preutterance"] = "vowel_island"
    if pre_ms is None:
        expected = _slot_expected_anchor_ms(row, "preutterance", duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        pre_ms = _candidate_near_expected_ms(tracks, "preutterance", expected, window_ms=max(80.0, duration / max(2.0, slot_count)))
        sources["preutterance"] = "slot_expected_fallback"
        warnings.append("fallback_preutterance")
    predictions["preutterance"] = min(duration, max(0.0, float(pre_ms)))

    offset_ms = _dependent_offset_ms(
        row,
        tracks,
        preutterance_ms=predictions["preutterance"],
        relative_anchor_priors=dict(scorer.relative_anchor_priors),
    )
    if offset_ms is None:
        offset_ms = max(0.0, predictions["preutterance"] - _default_offset_gap_ms(row))
        warnings.append("fallback_offset_gap")
    predictions["offset"] = min(predictions["preutterance"], max(0.0, float(offset_ms)))
    sources["offset"] = "local_from_preutterance"

    fixed_ms = _dependent_fixed_end_ms(
        row,
        tracks,
        preutterance_ms=predictions["preutterance"],
        relative_anchor_priors=dict(scorer.relative_anchor_priors),
    )
    fixed_prediction = constrained.get((id(row), "fixed_end"))
    fixed_safe_prob = _supervised_safe_probability(
        row,
        "fixed_end",
        fixed_prediction,
        scorer=scorer,
        filename_pred=filename_slot.get((id(row), "fixed_end")),
        joint_pred=joint.get((id(row), "fixed_end")),
    )
    if fixed_prediction is not None and not _is_korean_cvvc_row(row) and _passes_gate("fixed_end", fixed_safe_prob, scorer):
        candidate_fixed = float(fixed_prediction.get("pred_ms", 0.0) or 0.0)
        delta = _slot_shift_delta(row, pred_ms=candidate_fixed, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        if (
            delta == 0
            and candidate_fixed >= predictions["preutterance"]
            and _prediction_matches_island(row, "fixed_end", pred_ms=candidate_fixed, island=island)
        ):
            fixed_ms = candidate_fixed
            sources["fixed_end"] = "constrained_gate"
    if fixed_ms is None:
        fixed_ms = predictions["preutterance"] + _default_fixed_gap_ms(row)
        warnings.append("fallback_fixed_end_gap")
    predictions["fixed_end"] = min(duration, max(predictions["preutterance"] + 8.0, float(fixed_ms)))
    sources.setdefault("fixed_end", "relative_from_preutterance")

    cutoff_ms = _dependent_cutoff_abs_ms(
        row,
        duration_ms=duration,
        fixed_end_ms=predictions["fixed_end"],
    )
    if cutoff_ms is not None:
        predictions["cutoff"] = float(cutoff_ms)
        sources["cutoff"] = "dependent_from_fixed_end"

    overlap_pref = _dependent_overlap_ms(
        row,
        offset_ms=predictions["offset"],
        preutterance_ms=predictions["preutterance"],
        relative_anchor_priors=dict(scorer.relative_anchor_priors),
    )
    overlap_ms, overlap_context = _safe_overlap_ms(
        row,
        offset_ms=predictions["offset"],
        preutterance_ms=predictions["preutterance"],
        preferred_overlap_ms=overlap_pref,
    )
    predictions["overlap"] = float(overlap_ms)
    sources["overlap"] = "safe_relative_from_preutterance"

    guard_warnings, overlap_context = _apply_manual_anchor_timing_guards(
        row,
        predictions,
        island_context=island_context,
        acoustic_landmarks=acoustic_landmarks,
        duration_ms=duration,
        overlap_context=overlap_context,
    )
    if guard_warnings:
        warnings.extend(guard_warnings)
        sources["timing_guard"] = "manual_anchor_mfa_style_guards"

    local_gate_warnings: list[str] = []
    for anchor in anchors:
        pred = predictions.get(anchor)
        if pred is None:
            continue
        local_prob = _local_failure_safe_probability(
            scorer.local_failure_models.get(anchor),
            _local_failure_features(
                row,
                tracks,
                anchor,
                pred_ms=float(pred),
                predictions=predictions,
                safe_overlap_context=overlap_context,
                pre_safe_prob=pre_safe_prob,
                slot_count=slot_count,
                slot_index=slot_index,
                all_slot_exact=_all_slot_exact(row, predictions, duration, slot_count, slot_index),
                boundary_slot_exact=_boundary_slot_exact(row, predictions, duration, slot_count, slot_index),
            ),
        )
        if local_prob is None:
            local_gate_warnings.append(f"{anchor}:missing_local_gate")
        elif local_prob < float(scorer.local_gate_threshold):
            local_gate_warnings.append(f"{anchor}:local_gate_low:{local_prob:.3f}")
    warnings.extend(local_gate_warnings)

    timing, cutoff_abs = _timing_from_predictions(row, predictions, island=island, duration_ms=duration)
    absolute = _utau_absolute_oto_positions(timing, duration_ms=duration)
    quality = _classify_preview_row_quality(
        row,
        absolute=absolute,
        warnings=warnings,
        local_gate_warnings=local_gate_warnings,
        sources=sources,
        acoustic_landmarks=acoustic_landmarks,
        duration_ms=duration,
    )
    generated = {
        "wav": row.get("wav_name"),
        "alias": row.get("alias"),
        "alias_role": row.get("alias_role"),
        "line": format_oto_line(str(row.get("wav_name") or ""), str(row.get("alias") or ""), timing),
        "timing": timing_to_dict(timing),
        "absolute": absolute,
        "sources": dict(sources),
        "warnings": list(dict.fromkeys(warnings)),
        "quality": quality,
        "overlap_policy": overlap_context,
        "reclist_convention": dict(reclist_convention or {}) if isinstance(reclist_convention, Mapping) else {},
        "acoustic_landmarks": dict(acoustic_landmarks or {}),
        "island_context": _serialize_island_context(island_context),
    }
    record = {
        "alias": row.get("alias"),
        "alias_role": row.get("alias_role"),
        "slot_index": int(row.get("slot_index", 0) or 0),
        "slot_count": int(row.get("slot_count", 1) or 1),
        "island_slot_index": int(slot_index),
        "predictions_abs_ms": {
            **{key: float(value) for key, value in predictions.items()},
            "cutoff": float(cutoff_abs),
        },
        "timing": timing_to_dict(timing),
        "absolute": absolute,
        "sources": dict(sources),
        "preutterance_safe_prob": pre_safe_prob,
        "fixed_end_safe_prob": fixed_safe_prob,
        "local_gate_warnings": local_gate_warnings,
        "warnings": list(dict.fromkeys(warnings)),
        "quality": quality,
        "island": _island_json(island),
        "island_context": _serialize_island_context(island_context),
        "assignment": _assignment_json(assignment),
        "reclist_convention": dict(reclist_convention or {}) if isinstance(reclist_convention, Mapping) else {},
        "acoustic_landmarks": dict(acoustic_landmarks or {}),
    }
    return {"generated_row": generated, "record": record}


def _timing_from_predictions(
    row: Mapping[str, object],
    predictions: Mapping[str, float],
    *,
    island: VowelIsland | None,
    duration_ms: float,
) -> tuple[OtoTiming, float]:
    duration = max(1.0, float(duration_ms))
    offset = min(duration - 1.0, max(0.0, float(predictions.get("offset", 0.0) or 0.0)))
    pre_abs = min(duration, max(offset, float(predictions.get("preutterance", offset) or offset)))
    overlap_abs = min(pre_abs, max(offset, float(predictions.get("overlap", offset) or offset)))
    fixed_abs = min(duration, max(pre_abs + 1.0, float(predictions.get("fixed_end", pre_abs + 80.0) or pre_abs + 80.0)))
    predicted_cutoff = predictions.get("cutoff")
    if predicted_cutoff is not None and np.isfinite(float(predicted_cutoff)):
        cutoff_abs = min(duration, max(fixed_abs + 1.0, float(predicted_cutoff)))
    else:
        tail_gap = _default_utau_cutoff_tail_gap_ms(row)
        cutoff_abs = duration - min(max(1.0, float(tail_gap)), max(1.0, duration - fixed_abs - 1.0))
        cutoff_abs = min(duration, max(fixed_abs + 1.0, cutoff_abs))
    timing = OtoTiming(
        offset=float(offset),
        consonant=float(max(1.0, fixed_abs - offset)),
        cutoff=float(-max(1.0, duration - cutoff_abs)),
        preutterance=float(max(0.0, pre_abs - offset)),
        overlap=float(max(0.0, overlap_abs - offset)),
    )
    return timing, cutoff_abs


def _apply_manual_anchor_timing_guards(
    row: Mapping[str, object],
    predictions: dict[str, float],
    *,
    island_context: Mapping[str, object],
    acoustic_landmarks: Mapping[str, object] | None,
    duration_ms: float,
    overlap_context: Mapping[str, object],
) -> tuple[list[str], dict[str, object]]:
    warnings: list[str] = []
    breath_warning = _manual_anchor_breath_silence_guard(
        row,
        predictions,
        duration_ms=duration_ms,
    )
    if breath_warning:
        warnings.append(breath_warning)

    cv_warning = _manual_anchor_cv_boundary_guard(
        row,
        predictions,
        acoustic_landmarks=acoustic_landmarks,
        duration_ms=duration_ms,
    )
    if cv_warning:
        warnings.append(cv_warning)

    cutoff_warning = _manual_anchor_vc_vv_cutoff_guard(
        row,
        predictions,
        island_context=island_context,
        duration_ms=duration_ms,
    )
    if cutoff_warning:
        warnings.append(cutoff_warning)

    adaptive_overlap = _manual_anchor_adaptive_overlap_ms(
        row,
        offset_ms=predictions.get("offset", 0.0),
        preutterance_ms=predictions.get("preutterance", predictions.get("offset", 0.0)),
    )
    if adaptive_overlap is not None:
        previous_overlap = float(predictions.get("overlap", adaptive_overlap) or adaptive_overlap)
        overlap, next_context = _safe_overlap_ms(
            row,
            offset_ms=float(predictions.get("offset", 0.0) or 0.0),
            preutterance_ms=float(predictions.get("preutterance", 0.0) or 0.0),
            preferred_overlap_ms=adaptive_overlap,
        )
        predictions["overlap"] = float(overlap)
        overlap_context = {**dict(next_context), "adaptive_source": "mfa_style_overlap_ratio"}
        if abs(float(overlap) - previous_overlap) > 1.0:
            warnings.append("mfa_style_adaptive_overlap")

    validation_warnings = _manual_anchor_validate_prediction_order(
        row,
        predictions,
        duration_ms=duration_ms,
    )
    warnings.extend(validation_warnings)
    return _dedupe_texts(warnings), dict(overlap_context)


def _manual_anchor_breath_preutterance_ms(duration_ms: float) -> float:
    duration = max(1.0, float(duration_ms))
    target = min(760.0, max(360.0, duration * 0.30))
    return min(duration, max(0.0, min(duration - 8.0, target)))


def _manual_anchor_breath_fixed_end_ms(
    row: Mapping[str, object],
    *,
    duration_ms: float,
    preutterance_ms: float,
) -> float:
    duration = max(1.0, float(duration_ms))
    pre = min(duration, max(0.0, float(preutterance_ms)))
    ratio = 0.54 if duration >= 3600.0 else 0.48
    target = max(pre + 300.0, duration * ratio)
    return min(duration, max(pre + 4.0, min(duration - 4.0, target)))


def _manual_anchor_breath_silence_guard(
    row: Mapping[str, object],
    predictions: dict[str, float],
    *,
    duration_ms: float,
) -> str | None:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if family != "breath_silence" and role not in {"br", "sil"}:
        return None

    before = {key: float(value) for key, value in predictions.items() if _is_finite_number(value)}
    duration = max(1.0, float(duration_ms))
    pre = _manual_anchor_breath_preutterance_ms(duration)
    fixed = _manual_anchor_breath_fixed_end_ms(row, duration_ms=duration, preutterance_ms=pre)
    predictions["preutterance"] = float(pre)
    predictions["offset"] = max(0.0, float(pre) - 50.0)
    predictions["overlap"] = max(float(predictions["offset"]), float(pre) - 25.0)
    predictions["fixed_end"] = float(fixed)

    cutoff = predictions.get("cutoff")
    if cutoff is None or not _is_finite_number(cutoff) or float(cutoff) < float(fixed) + 4.0:
        predictions["cutoff"] = min(duration, float(fixed) + 4.0)

    for key in ("offset", "overlap", "preutterance", "fixed_end", "cutoff"):
        if abs(float(predictions.get(key, 0.0) or 0.0) - float(before.get(key, predictions.get(key, 0.0)) or 0.0)) > 1.0:
            return "mfa_style_breath_silence_guard"
    return None


def _manual_anchor_cv_boundary_guard(
    row: Mapping[str, object],
    predictions: dict[str, float],
    *,
    acoustic_landmarks: Mapping[str, object] | None,
    duration_ms: float,
) -> str | None:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if family == "breath_silence" or role in {"br", "sil"}:
        return None
    korean_cv = _is_korean_cvvc_row(row) and role == "cv"
    japanese_cv_like = _is_japanese_row(row) and role in {"cv", "cv_head", "vcv"}
    if not (korean_cv or japanese_cv_like):
        return None
    if not isinstance(acoustic_landmarks, Mapping):
        return None
    try:
        confidence = float(acoustic_landmarks.get("confidence", 0.0) or 0.0)
        vowel_onset = float(acoustic_landmarks.get("vowel_onset_ms", 0.0) or 0.0)
    except Exception:
        return None
    if not (np.isfinite(confidence) and np.isfinite(vowel_onset)) or confidence < 0.18:
        return None

    timing_class = ""
    if japanese_cv_like:
        lead_ms, lag_ms, _max_refine = _japanese_cv_landmark_window_ms(row)
    else:
        timing_class = _korean_cvvc_cv_timing_class(row)
        if timing_class == "initial_consonant":
            lead_ms, lag_ms = 82.0, 26.0
        elif timing_class == "vf":
            lead_ms, lag_ms = 58.0, 42.0
        elif timing_class == "apostrophe":
            lead_ms, lag_ms = 24.0, 58.0
        else:
            lead_ms, lag_ms = 16.0, 58.0

    duration = max(1.0, float(duration_ms))
    lower = max(0.0, vowel_onset - lead_ms)
    upper = min(duration, vowel_onset + lag_ms)
    previous = float(predictions.get("preutterance", vowel_onset) or vowel_onset)
    guarded = min(upper, max(lower, previous))
    if abs(guarded - previous) <= 1.0:
        return None
    predictions["preutterance"] = float(guarded)
    if "offset" in predictions:
        if japanese_cv_like:
            min_offset_gap = 18.0 if role == "vcv" else 12.0
        else:
            min_offset_gap = 8.0 if timing_class == "initial_consonant" else 12.0
        predictions["offset"] = min(float(predictions["offset"]), max(0.0, guarded - min_offset_gap))
    return "mfa_style_ja_cv_boundary_guard" if japanese_cv_like else "mfa_style_cv_boundary_guard"


def _manual_anchor_vc_vv_cutoff_guard(
    row: Mapping[str, object],
    predictions: dict[str, float],
    *,
    island_context: Mapping[str, object],
    duration_ms: float,
) -> str | None:
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role not in {"vc", "vv"} or not (_is_korean_cvvc_row(row) or _is_japanese_row(row)):
        return None
    try:
        cutoff = float(predictions.get("cutoff", 0.0) or 0.0)
        fixed = float(predictions.get("fixed_end", 0.0) or 0.0)
        pre = float(predictions.get("preutterance", fixed) or fixed)
    except Exception:
        return None
    if not (np.isfinite(cutoff) and np.isfinite(fixed) and np.isfinite(pre)):
        return None

    current = island_context.get("current") if isinstance(island_context, Mapping) else None
    next_island = island_context.get("next") if isinstance(island_context, Mapping) else None
    _, cutoff_gap = _manual_anchor_validation_gaps(row)
    duration = max(1.0, float(duration_ms))
    floor = min(duration, max(fixed + cutoff_gap, pre + 4.0))

    if role == "vc":
        if not isinstance(next_island, VowelIsland):
            return None
        if _is_japanese_row(row):
            floor = max(floor, float(next_island.start_ms) + _manual_anchor_japanese_vc_cutoff_floor_margin_ms(row))
        else:
            onset_margin = _manual_anchor_vc_cutoff_margin_ms(_manual_anchor_consonant_hint(row))
            floor = max(floor, float(next_island.start_ms) + onset_margin)
    elif role == "vv":
        if not isinstance(current, VowelIsland):
            return None
        floor = max(floor, float(current.end_ms) + (10.0 if _is_japanese_row(row) else 14.0))

    floor = min(duration, max(0.0, float(floor)))
    if cutoff >= floor - 1.0:
        return None
    predictions["cutoff"] = float(floor)
    return f"mfa_style_{role}_cutoff_guard"


def _manual_anchor_validate_prediction_order(
    row: Mapping[str, object],
    predictions: dict[str, float],
    *,
    duration_ms: float,
) -> list[str]:
    before = {key: float(value) for key, value in predictions.items() if _is_finite_number(value)}
    duration = max(1.0, float(duration_ms))
    fixed_gap, cutoff_gap = _manual_anchor_validation_gaps(row)

    offset = min(duration - 1.0, max(0.0, float(predictions.get("offset", 0.0) or 0.0)))
    pre = min(duration, max(offset, float(predictions.get("preutterance", offset) or offset)))
    overlap = min(pre, max(offset, float(predictions.get("overlap", offset) or offset)))

    max_fixed = max(pre + 1.0, duration - min(cutoff_gap, max(1.0, duration - pre - 1.0)))
    fixed = max(pre + fixed_gap, float(predictions.get("fixed_end", pre + fixed_gap) or pre + fixed_gap))
    fixed = min(duration, max(pre + 1.0, min(fixed, max_fixed)))

    cutoff = float(predictions.get("cutoff", fixed + cutoff_gap) or fixed + cutoff_gap)
    cutoff = min(duration, max(fixed + cutoff_gap, cutoff))
    if cutoff > duration:
        cutoff = duration
    if cutoff <= fixed:
        fixed = min(max(pre + 1.0, duration - cutoff_gap), fixed)
        cutoff = min(duration, max(fixed + 1.0, cutoff))

    predictions["offset"] = float(offset)
    predictions["preutterance"] = float(pre)
    predictions["overlap"] = float(overlap)
    predictions["fixed_end"] = float(fixed)
    predictions["cutoff"] = float(cutoff)

    adjusted = []
    for key in ("offset", "overlap", "preutterance", "fixed_end", "cutoff"):
        if abs(float(predictions.get(key, 0.0) or 0.0) - float(before.get(key, predictions.get(key, 0.0)) or 0.0)) > 1.0:
            adjusted.append(key)
    if not adjusted:
        return []
    return [f"mfa_style_order_validation:{','.join(adjusted)}"]


def _manual_anchor_validation_gaps(row: Mapping[str, object]) -> tuple[float, float]:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if family == "breath_silence" or role in {"br", "sil"}:
        return 4.0, 4.0
    if _is_japanese_row(row):
        if role == "vcv":
            return 18.0, 12.0
        if role in {"cv", "cv_head"}:
            return 18.0, 14.0
        if role == "vc":
            return 10.0, 8.0
        if role == "vv":
            return 16.0, 12.0
    if _is_korean_cvvc_row(row) and role == "cv":
        timing_class = _korean_cvvc_cv_timing_class(row)
        if timing_class == "initial_consonant":
            return 8.0, 6.0
        if timing_class == "apostrophe":
            return 12.0, 10.0
        return 14.0, 10.0
    if role == "vc":
        return 8.0, 8.0
    if role in {"v", "vv"}:
        return 12.0, 10.0
    return 16.0, 12.0


def _manual_anchor_adaptive_overlap_ms(
    row: Mapping[str, object],
    *,
    offset_ms: object,
    preutterance_ms: object,
) -> float | None:
    if manual_oto_alias_family(row.get("alias", "")) == "breath_silence":
        return None
    try:
        offset = float(offset_ms)
        pre = float(preutterance_ms)
    except Exception:
        return None
    if not (np.isfinite(offset) and np.isfinite(pre)):
        return None
    span = max(0.0, pre - offset)
    if span <= 0.0:
        return None
    adaptive_gap = _manual_anchor_adaptive_overlap_gap(row, span)
    if adaptive_gap is None:
        return None
    gap = float(adaptive_gap)
    korean_gaps = _korean_cvvc_relative_gaps(row)
    if korean_gaps is not None:
        base_gap = float(korean_gaps["overlap_gap"])
        role = str(row.get("alias_role", "") or "").strip().lower()
        adaptive_weight = 0.20 if role == "cv" else 0.30
        gap = (1.0 - adaptive_weight) * base_gap + adaptive_weight * gap
    gap = min(span, max(0.0, gap))
    return float(pre - gap)


def _manual_anchor_adaptive_overlap_gap(row: Mapping[str, object], span_ms: float) -> float | None:
    span = max(0.0, float(span_ms))
    if span <= 0.0:
        return None
    if _is_japanese_row(row):
        return _manual_anchor_japanese_adaptive_overlap_gap(row, span)
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if family == "breath_silence" or role in {"br", "sil"}:
        return 0.0
    if role in {"vc", "vcv"} or family in {"leading_n", "vowel_transition"}:
        ratio = 0.44
        min_gap = 14.0
    elif role in {"v", "vv"}:
        ratio = 0.54
        min_gap = 14.0
    else:
        ratio = 0.46
        min_gap = 8.0

    hint = _manual_anchor_consonant_hint(row)
    if _manual_anchor_hint_is_tense(hint):
        ratio -= 0.20
    elif _manual_anchor_hint_is_hard(hint):
        ratio -= 0.14
    elif _manual_anchor_hint_is_aspirate(hint):
        ratio += 0.02
    elif _manual_anchor_hint_is_fricative(hint):
        ratio += 0.05
    elif _manual_anchor_hint_is_sonorant(hint):
        ratio += 0.08
    ratio = min(0.72, max(0.20, ratio))
    gap = span - (span * ratio)
    return float(min(span, max(min_gap, gap)))


def _manual_anchor_consonant_hint(row: Mapping[str, object]) -> str:
    role = str(row.get("alias_role", "") or "").strip().lower()
    alias = row.get("alias", "")
    terms = _korean_cvvc_alias_order_terms(alias, role=role) if _is_korean_cvvc_row(row) else []
    if not terms:
        terms = manual_oto_alias_order_terms(
            alias,
            language=str(row.get("language", "") or ""),
            format_type=str(row.get("format_type", "") or ""),
            alias_role=role,
        )
    token = ""
    if role in {"cv", "cv_head", "vcv"} and terms:
        token = terms[0]
    elif role in {"vc", "vv"} and len(terms) >= 2:
        token = terms[1]
    elif terms:
        token = terms[-1]
    else:
        text = str(alias or "").strip().lower().split("_", 1)[0].strip("*-_ '")
        token = text.split()[0] if text.split() else text
    token = "".join(ch for ch in _plain_order_token_for_matching(token) if ch.isalnum())
    return _manual_anchor_base_consonant(token)


def _manual_anchor_base_consonant(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return ""
    for prefix in (
        "ng",
        "kk",
        "tt",
        "pp",
        "ss",
        "jj",
        "ch",
        "kh",
        "th",
        "ph",
        "rr",
        "sh",
        "ts",
        "g",
        "k",
        "d",
        "t",
        "b",
        "p",
        "s",
        "j",
        "c",
        "h",
        "m",
        "n",
        "l",
        "r",
    ):
        if text.startswith(prefix):
            return prefix
    return ""


def _manual_anchor_hint_is_tense(hint: str) -> bool:
    return hint in {"kk", "tt", "pp", "ss", "jj"}


def _manual_anchor_hint_is_hard(hint: str) -> bool:
    return hint in {"g", "k", "d", "t", "b", "p", "c", "j"}


def _manual_anchor_hint_is_aspirate(hint: str) -> bool:
    return hint in {"ch", "kh", "th", "ph", "h"}


def _manual_anchor_hint_is_fricative(hint: str) -> bool:
    return hint in {"s", "ss", "sh"}


def _manual_anchor_hint_is_sonorant(hint: str) -> bool:
    return hint in {"m", "n", "ng", "l", "r", "rr"}


def _manual_anchor_vc_cutoff_margin_ms(hint: str) -> float:
    if _manual_anchor_hint_is_tense(hint) or _manual_anchor_hint_is_hard(hint):
        return 4.0
    if hint in {"l", "r", "rr"}:
        return 14.0
    if hint in {"m", "n", "ng"}:
        return 10.0
    if _manual_anchor_hint_is_aspirate(hint) or _manual_anchor_hint_is_fricative(hint):
        return 8.0
    return 10.0


def _manual_anchor_japanese_adaptive_overlap_gap(row: Mapping[str, object], span_ms: float) -> float:
    span = max(0.0, float(span_ms))
    role = str(row.get("alias_role", "") or "").strip().lower()
    hint = _manual_anchor_consonant_hint(row)
    onset_class = _manual_anchor_japanese_onset_class(row)
    if role == "vv":
        min_gap, max_gap, target_gap = 4.0, 10.0, 7.0
    elif role in {"vc", "vcv"}:
        if onset_class in {"voiceless", "plosive", "sibilant", "fricative"}:
            min_gap, max_gap, target_gap = (8.0, 16.0, 11.0) if role == "vcv" else (10.0, 20.0, 14.0)
        elif onset_class in {"nasal", "liquid", "glide"} or _manual_anchor_hint_is_sonorant(hint):
            min_gap, max_gap, target_gap = (5.0, 12.0, 8.0) if role == "vcv" else (6.0, 14.0, 9.0)
        else:
            min_gap, max_gap, target_gap = (6.0, 13.0, 9.0) if role == "vcv" else (7.0, 15.0, 10.0)
        adaptive_floor = max(0.0, span * 0.12)
        target_gap = max(float(target_gap), adaptive_floor)
        max_gap = min(float(max_gap), max(span - 1.0, float(min_gap)))
        min_gap = min(max(float(min_gap), min(adaptive_floor, max_gap)), max_gap)
        return float(min(span, max(min_gap, min(max_gap, target_gap))))

    ratio = 0.40 if role == "cv_head" else 0.46
    if onset_class in {"voiceless", "plosive"} or _manual_anchor_hint_is_hard(hint):
        ratio -= 0.16
    elif onset_class in {"sibilant", "fricative"} or _manual_anchor_hint_is_fricative(hint):
        ratio += 0.05
    elif onset_class in {"nasal", "liquid", "glide"} or _manual_anchor_hint_is_sonorant(hint):
        ratio += 0.08
    ratio = min(0.72, max(0.20, ratio))
    min_gap = 10.0 if role == "cv_head" else 8.0
    return float(min(span, max(min_gap, span - (span * ratio))))


def _manual_anchor_japanese_onset_class(row: Mapping[str, object]) -> str:
    hint = _manual_anchor_consonant_hint(row)
    if not hint:
        return "other"
    if hint in {"m", "n", "ny", "ng"}:
        return "nasal"
    if hint in {"r", "ry", "l", "rr"}:
        return "liquid"
    if hint in {"y", "w"}:
        return "glide"
    if hint in {"s", "z", "sh", "j", "ch", "ts", "dz", "ss", "zz", "jj"}:
        return "sibilant"
    if hint in {"h", "f", "v", "hy"}:
        return "fricative"
    if hint in {"k", "g", "t", "d", "b", "p", "q", "c", "kk", "tt", "pp", "dd", "gg", "bb", "ky", "gy", "ty", "dy", "by", "py"}:
        return "plosive"
    return "other"


def _manual_anchor_japanese_vc_cutoff_floor_margin_ms(row: Mapping[str, object]) -> float:
    onset_class = _manual_anchor_japanese_onset_class(row)
    if onset_class in {"plosive", "sibilant", "fricative"}:
        return -2.0
    if onset_class in {"nasal", "liquid", "glide"}:
        return 2.0
    return 0.0


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _utau_absolute_oto_positions(timing: OtoTiming, *, duration_ms: float) -> dict[str, float]:
    duration = max(1.0, float(duration_ms))
    if float(timing.cutoff) < 0.0:
        cutoff_abs = duration + float(timing.cutoff)
    else:
        cutoff_abs = float(timing.offset) + float(timing.cutoff)
    return {
        "offset_abs": float(timing.offset),
        "overlap_abs": float(timing.offset) + float(timing.overlap),
        "preutterance_abs": float(timing.offset) + float(timing.preutterance),
        "consonant_abs": float(timing.offset) + float(timing.consonant),
        "cutoff_abs": float(cutoff_abs),
    }


def _posterior_from_tracks(wav_path: str, tracks: ManualOtoCandidateTracks) -> FramePosterior:
    times = [float(value) for value in np.asarray(tracks.times_ms, dtype=np.float32).tolist()]
    size = len(times)
    zeros = [0.0] * size

    def track(name: str) -> list[float]:
        arr = np.asarray(tracks.tracks.get(name, zeros), dtype=np.float32)
        if arr.size != size:
            arr = np.resize(arr, (size,)).astype(np.float32)
        return [float(np.clip(value, 0.0, 1.0)) for value in arr.tolist()]

    def score(name: str) -> list[float]:
        arr = np.asarray(tracks.anchor_scores.get(name, zeros), dtype=np.float32)
        if arr.size != size:
            arr = np.resize(arr, (size,)).astype(np.float32)
        return [float(np.clip(value, 0.0, 1.0)) for value in arr.tolist()]

    nucleus = track("nucleus")
    transition = track("transition")
    silence = track("silence")
    return FramePosterior(
        wav_path=wav_path,
        times_ms=times,
        class_probs={
            "silence": silence,
            "consonant": transition,
            "vowel": nucleus,
            "other": [max(0.0, 1.0 - max(s, n, t)) for s, n, t in zip(silence, nucleus, transition)],
        },
        event_scores={
            "cv_boundary": score("preutterance"),
            "vowel_nucleus": nucleus,
            "phone_change": transition,
        },
        acoustic_scores={key: track(key) for key in tracks.tracks},
        metadata={
            "source": "manual_oto_anchor_runtime",
            "encoder": tracks.encoder,
            **dict(tracks.timebase_metadata or {}),
        },
    )


def _write_manual_anchor_overlay(
    overlay_dir: Path,
    wav_path: str,
    rows: list[dict[str, object]],
    tracks: ManualOtoCandidateTracks,
    wav_result: Mapping[str, object],
) -> None:
    row = {
        "row_id": Path(wav_path).stem,
        "wav_name": os.path.basename(wav_path),
        "wav_path": wav_path,
        "duration_ms": float(tracks.duration_ms),
        "expected_phones": _filename_tokens_for_row(rows[0]) if rows else [],
        "label_source": "manual_oto_anchor_preview",
        "auxiliary_events": [
            *island_overlay_events(_decode_from_json(wav_result.get("island_decode") or {})),
            *_cv_landmark_overlay_events(wav_result),
        ],
    }
    decoded_events: list[dict[str, object]] = []
    for generated in wav_result.get("generated_oto_rows", []):
        if not isinstance(generated, dict):
            continue
        absolute = generated.get("absolute") if isinstance(generated.get("absolute"), dict) else {}
        for label, key in (
            ("model_pred_anchor", "preutterance_abs"),
            ("manual_anchor", "offset_abs"),
            ("joint_pred_anchor", "consonant_abs"),
        ):
            if key not in absolute:
                continue
            decoded_events.append(
                {
                    "label": label,
                    "time_ms": float(absolute.get(key) or 0.0),
                    "score": 1.0,
                    "alias": str(generated.get("alias") or ""),
                }
            )
    write_review_html(
        overlay_dir / f"{Path(wav_path).stem}.html",
        row,
        posterior=_posterior_from_tracks(wav_path, tracks),
        decoded_events=decoded_events,
        generated_oto_rows=list(wav_result.get("generated_oto_rows", []) or []),
    )


def _cv_landmark_overlay_events(wav_result: Mapping[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    rows = wav_result.get("row_records", [])
    if not isinstance(rows, list):
        return out
    for record in rows:
        if not isinstance(record, Mapping):
            continue
        landmarks = record.get("acoustic_landmarks")
        if not isinstance(landmarks, Mapping):
            continue
        source = str(landmarks.get("source") or "")
        if source.startswith("acoustic_cv_"):
            label_keys = (
                ("cv_transient_onset", "transient_onset_ms"),
                ("cv_vowel_onset", "vowel_onset_ms"),
                ("cv_stable_vowel_start", "stable_vowel_start_ms"),
                ("cv_vowel_nucleus", "vowel_nucleus_ms"),
                ("cv_vowel_offset", "vowel_offset_ms"),
            )
        elif source.startswith("acoustic_tail_"):
            label_keys = (
                ("tail_vowel_offset", "vowel_offset_ms"),
                ("tail_transition", "transition_tail_ms"),
            )
        elif source.startswith("acoustic_vv_"):
            label_keys = (
                ("vv_transition_valley", "transition_valley_ms"),
                ("vv_left_nucleus", "left_vowel_nucleus_ms"),
                ("vv_right_onset", "right_vowel_onset_ms"),
            )
        else:
            label_keys = ()
        alias = str(record.get("alias") or "")
        confidence = float(landmarks.get("confidence", 0.0) or 0.0)
        for label, key in label_keys:
            if key not in landmarks:
                continue
            out.append(
                {
                    "label": label,
                    "time_ms": float(landmarks.get(key) or 0.0),
                    "score": confidence,
                    "alias": alias,
                }
            )
    return out


def _score_candidates(model: object, features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(features), dtype=np.float32).reshape(-1)
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(features)
            if getattr(probs, "shape", (0, 0))[1] >= 2:
                return np.asarray(probs[:, 1], dtype=np.float32)
        except Exception:
            pass
    return np.asarray(model.predict(features), dtype=np.float32).reshape(-1)


def _centered_slot_pos(row: Mapping[str, object]) -> float:
    try:
        slot_index = int(row.get("slot_index", 0) or 0)
        slot_count = int(row.get("slot_count", 1) or 1)
    except Exception:
        return float(row.get("slot_pos_norm", 0.0) or 0.0)
    if slot_count <= 1:
        return 0.5
    return min(0.98, max(0.02, (slot_index + 0.5) / max(1.0, float(slot_count))))


def _row_with_slot_pos(row: dict[str, object], slot_pos_norm: float) -> dict[str, object]:
    out = dict(row)
    out["slot_pos_norm"] = min(0.98, max(0.02, float(slot_pos_norm)))
    return out


def _filename_tokens_for_row(row: Mapping[str, object]) -> list[str]:
    language = str(row.get("language", "") or "").strip().lower()
    if language in {"ko", "kr", "korean"}:
        direct = _direct_korean_filename_tokens(row)
        if direct and str(row.get("format_type", "") or "").strip().lower() == "cvvc":
            return direct
        raw_tokens = row.get("filename_tokens") or []
        if isinstance(raw_tokens, list) and raw_tokens:
            return [_plain_order_token_for_matching(item) for item in raw_tokens if _plain_order_token_for_matching(item)]
    if language in {"ja", "jp", "japanese"}:
        raw_tokens = row.get("filename_tokens") or []
        if isinstance(raw_tokens, list) and raw_tokens:
            return [_plain_order_token_for_matching(item) for item in raw_tokens if _plain_order_token_for_matching(item)]
    raw = row.get("filename_canonical_tokens") or []
    if isinstance(raw, list) and raw:
        return [canonicalize_order_token(item) for item in raw if canonicalize_order_token(item)]
    wav_name = str(row.get("wav_name", "") or Path(str(row.get("wav_path", "") or "")).name)
    context = parse_filename_context(wav_name, language=language, format_type=str(row.get("format_type", "") or ""))
    if language in {"ko", "kr", "korean"}:
        return [_plain_order_token_for_matching(item) for item in context.tokens if _plain_order_token_for_matching(item)]
    return [canonicalize_order_token(item) for item in context.canonical_tokens if canonicalize_order_token(item)]


def _direct_korean_filename_tokens(row: Mapping[str, object]) -> list[str]:
    wav_name = str(row.get("wav_name", "") or Path(str(row.get("wav_path", "") or "")).name)
    if not wav_name:
        return []
    stem = Path(wav_name).stem.strip()
    if not stem:
        return []
    stem = stem.lstrip("_")
    parts = stem.replace("_", "'").split("'")
    out: list[str] = []
    for part in parts:
        token = str(part or "").strip().strip("_-")
        if not token:
            continue
        if "(" in token:
            token = token.split("(", 1)[0].strip()
        token = "".join(ch for ch in token if ch.isalnum())
        token = _plain_order_token_for_matching(token)
        if token:
            out.append(token)
    return out


def _filename_slot_tokens_for_row(row: Mapping[str, object]) -> list[str]:
    tokens = _filename_tokens_for_row(row)
    if _is_korean_cvvc_row(row):
        filtered = [token for token in tokens if _korean_cvvc_token_is_vowel_slot(token)]
        if filtered:
            return filtered
    return tokens


def _infer_filename_slot_pos(
    row: Mapping[str, object],
    tokens: list[str],
    *,
    require_primary_transition_token: bool = False,
) -> float | None:
    if not tokens:
        return None
    terms = manual_oto_alias_order_terms(
        row.get("alias", ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias_role=str(row.get("alias_role", "") or ""),
    )
    if not terms:
        return None
    fallback_pos = _centered_slot_pos(row)
    role = str(row.get("alias_role", "") or "").strip().lower()
    language = str(row.get("language", "") or "").strip().lower()
    if language in {"ja", "jp", "japanese"} and role in {"vcv", "vv"}:
        pair_pos = _japanese_transition_slot_pos(row, tokens, terms, fallback_pos=fallback_pos)
        if pair_pos is not None:
            return pair_pos
    primary_term = terms[0] if terms else ""
    primary_best_score = 0
    best_idx: int | None = None
    best_score = -1
    for idx, token in enumerate(tokens):
        score = 0
        for term_idx, term in enumerate(terms):
            if not term:
                continue
            if term == token:
                score += 8 if term_idx == 0 else 4
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 4)
            elif token.startswith(term) or term.startswith(token):
                score += 4 if term_idx == 0 else 2
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 2)
            elif term in token or token in term:
                score += 2 if term_idx == 0 else 1
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 1)
        if score > best_score:
            best_score = score
            best_idx = idx
        elif score == best_score and best_idx is not None:
            current_pos = (idx + 0.5) / max(1.0, float(len(tokens)))
            best_pos = (best_idx + 0.5) / max(1.0, float(len(tokens)))
            if abs(current_pos - fallback_pos) < abs(best_pos - fallback_pos):
                best_idx = idx
    if best_idx is None or best_score <= 0:
        return None
    if bool(require_primary_transition_token) and role in {"vc", "vcv", "vv"} and primary_term and primary_best_score <= 0:
        return None
    return min(0.98, max(0.02, (best_idx + 0.5) / max(1.0, float(len(tokens)))))


def _japanese_transition_slot_pos(
    row: Mapping[str, object],
    tokens: Sequence[object],
    terms: Sequence[str],
    *,
    fallback_pos: float,
) -> float | None:
    if len(tokens) < 2 or len(terms) < 2:
        return None
    target = _plain_order_token_for_matching(terms[0])
    previous = _plain_order_token_for_matching(terms[-1])
    if not target:
        return None
    best: tuple[float, int] | None = None
    for idx, token in enumerate(tokens):
        token_text = _plain_order_token_for_matching(token)
        target_score = _japanese_syllable_match_score(token_text, target)
        if target_score <= 0.0:
            continue
        prev_score = 0.0
        if previous and idx > 0:
            prev_score = _japanese_syllable_match_score(
                _plain_order_token_for_matching(tokens[idx - 1]),
                previous,
                allow_vowel_only=True,
            )
        candidate_pos = (float(idx) + 0.5) / max(1.0, float(len(tokens)))
        score = (10.0 * target_score) + (4.0 * prev_score) - (0.45 * abs(candidate_pos - float(fallback_pos)))
        if best is None or score > best[0]:
            best = (score, idx)
    if best is None or best[0] <= 0.0:
        return None
    return min(0.98, max(0.02, (float(best[1]) + 0.5) / max(1.0, float(len(tokens)))))


def _japanese_syllable_match_score(token: str, term: str, *, allow_vowel_only: bool = False) -> float:
    token_norm = _plain_order_token_for_matching(token)
    term_norm = _plain_order_token_for_matching(term)
    if not token_norm or not term_norm:
        return 0.0
    if token_norm == term_norm:
        return 4.0
    if canonicalize_order_token(token_norm) == canonicalize_order_token(term_norm):
        return 3.5
    token_onset, token_vowel = _japanese_syllable_parts(token_norm)
    term_onset, term_vowel = _japanese_syllable_parts(term_norm)
    if token_vowel and term_vowel and token_vowel == term_vowel:
        if _japanese_onset_matches(token_onset, term_onset):
            return 3.0
        if allow_vowel_only:
            return 1.4
    return 0.0


def _japanese_syllable_parts(value: str) -> tuple[str, str]:
    text = _plain_order_token_for_matching(value)
    if not text:
        return "", ""
    if text in {"n", "nn", "xn", "m"}:
        return "", "n"
    for vowel in ("a", "i", "u", "e", "o"):
        if text.endswith(vowel):
            return text[: -len(vowel)], vowel
    return text, ""


def _japanese_onset_matches(left: str, right: str) -> bool:
    left_norm = _japanese_onset_norm(left)
    right_norm = _japanese_onset_norm(right)
    return bool(left_norm or right_norm) and left_norm == right_norm


def _japanese_onset_norm(value: str) -> str:
    text = _plain_order_token_for_matching(value)
    if text in {"z", "j"}:
        return "j"
    if text in {"ts", "t"}:
        return "t"
    if text in {"ch", "ty", "ti"}:
        return "ch"
    return text


def _filename_ordered_rows(
    wav_rows: list[dict[str, object]],
    *,
    require_primary_transition_token: bool = False,
) -> list[tuple[dict[str, object], float]]:
    if not wav_rows:
        return []
    tokens = _filename_slot_tokens_for_row(wav_rows[0])
    ordered: list[tuple[dict[str, object], float, int, bool]] = []
    inferred = 0
    for input_idx, row in enumerate(wav_rows):
        pos = _infer_filename_slot_pos(
            row,
            tokens,
            require_primary_transition_token=bool(require_primary_transition_token),
        )
        inferred += 1 if pos is not None else 0
        ordered.append((row, float(pos if pos is not None else _centered_slot_pos(row)), input_idx, pos is not None))
    if inferred >= 2:
        ordered.sort(key=lambda item: (item[1], item[2]))
    return [(row, pos) for row, pos, _input_idx, _ok in ordered]


def _row_slot_ordered_rows(wav_rows: list[dict[str, object]]) -> list[tuple[dict[str, object], float]]:
    ordered: list[tuple[dict[str, object], float, int]] = []
    for input_idx, row in enumerate(wav_rows):
        ordered.append((row, _centered_slot_pos(row), input_idx))
    ordered.sort(key=lambda item: (int(item[0].get("slot_index", item[2]) or item[2]), item[2]))
    return [(row, pos) for row, pos, _input_idx in ordered]


def _vowel_island_slot_count(wav_rows: list[dict[str, object]]) -> int:
    if not wav_rows:
        return 1
    max_count = 1
    max_index = 0
    for row in wav_rows:
        try:
            max_count = max(max_count, int(row.get("slot_count", 1) or 1))
            max_index = max(max_index, int(row.get("slot_index", 0) or 0))
        except Exception:
            continue
    manifest_count = max(1, max(max_count, max_index + 1))
    tokens = _filename_slot_tokens_for_row(wav_rows[0])
    language = str(wav_rows[0].get("language", "") or "").strip().lower()
    format_type = str(wav_rows[0].get("format_type", "") or "").strip().lower()
    if language in {"ko", "kr", "korean"} and format_type == "cvvc" and len(tokens) >= 2:
        return len(tokens)
    if language in {"ja", "jp", "japanese"} and format_type == "vcv" and len(tokens) >= 3:
        inferred_indices: list[int] = []
        for row in wav_rows:
            pos = _infer_filename_slot_pos(row, tokens)
            if pos is None:
                continue
            idx = int(round(float(pos) * float(len(tokens)) - 0.5))
            if 0 <= idx < len(tokens):
                inferred_indices.append(idx)
        if inferred_indices:
            return len(tokens)
    if format_type in {"cv", "cvvc", "cvc"} and len(tokens) == manifest_count + 1:
        inferred_indices: list[int] = []
        for row in wav_rows:
            pos = _infer_filename_slot_pos(row, tokens)
            if pos is None:
                continue
            idx = int(round(float(pos) * float(len(tokens)) - 0.5))
            if 0 <= idx < len(tokens):
                inferred_indices.append(idx)
        if inferred_indices and min(inferred_indices) >= 1:
            return len(tokens)
    if manifest_count >= 2:
        return manifest_count
    if len(tokens) >= 2:
        return len(tokens)
    return manifest_count


def _row_vowel_island_slot_index(row: Mapping[str, object], slot_count: int) -> int:
    count = max(1, int(slot_count))
    tokens = _filename_slot_tokens_for_row(row)
    korean_cvvc_idx = _korean_cvvc_row_syllable_index(row, tokens, count)
    if korean_cvvc_idx is not None:
        return korean_cvvc_idx
    try:
        row_count = max(1, int(row.get("slot_count", count) or count))
        row_index = int(row.get("slot_index", 0) or 0)
        if count != row_count and len(tokens) == count:
            pos = _infer_filename_slot_pos(row, tokens)
            if pos is not None:
                idx = int(round(float(pos) * float(count) - 0.5))
                return max(0, min(count - 1, idx))
        if 0 <= row_index < row_count:
            pos = (float(row_index) + 0.5) / max(1.0, float(row_count))
            idx = int(round(pos * float(count) - 0.5))
            return max(0, min(count - 1, idx))
    except Exception:
        pass
    pos = _infer_filename_slot_pos(row, tokens)
    if pos is None:
        pos = _centered_slot_pos(row)
    idx = int(round(float(pos) * float(count) - 0.5))
    return max(0, min(count - 1, idx))


def _korean_cvvc_row_syllable_index(
    row: Mapping[str, object],
    tokens: list[str],
    slot_count: int,
) -> int | None:
    if not _is_korean_cvvc_row(row):
        return None
    if not tokens:
        return None
    count = max(1, min(int(slot_count), len(tokens)))
    try:
        source_order = int(row.get("source_order", row.get("slot_index", 0)) or 0)
    except Exception:
        source_order = 0
    role = str(row.get("alias_role", "") or "").strip().lower()
    terms = _korean_cvvc_alias_order_terms(row.get("alias", ""), role=role) or manual_oto_alias_order_terms(
        row.get("alias", ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias_role=role,
    )
    normalized_tokens = [_plain_order_token_for_matching(token) for token in tokens]

    scaled_target = _scaled_source_order_target(row, count=count, source_order=source_order)
    pair_target = _korean_cvvc_source_pair_target(row, count=count, source_order=source_order)

    if role in {"cv", "v"} and terms:
        is_start_alias = str(row.get("alias", "") or "").strip().startswith("-")
        target = 0.0 if is_start_alias else (pair_target if pair_target is not None else scaled_target)
        idx = _nearest_token_match_index(normalized_tokens, terms[0], target=target)
        if idx is not None:
            if role == "cv" and not is_start_alias and not str(row.get("alias", "") or "").strip().startswith("'"):
                idx = _avoid_terminal_duplicate_cv_slot(normalized_tokens, terms[0], idx)
            return max(0, min(count - 1, idx))
        if is_start_alias:
            return 0

    if role == "vc" and terms:
        left = canonicalize_order_token(terms[0])
        right = canonicalize_order_token(terms[1]) if len(terms) >= 2 else ""
        target = pair_target if pair_target is not None else scaled_target
        if len(terms) == 1 and _looks_vowel_term(left):
            candidates = [
                idx
                for idx, current in enumerate(normalized_tokens)
                if _term_matches_token(left, current)
            ]
            if candidates:
                return max(0, min(count - 1, min(candidates, key=lambda idx: abs(idx - float(target)))))
        if right in {"-", "*"} or str(row.get("alias", "")).strip().endswith(("*", "-")):
            idx = _nearest_token_match_index(normalized_tokens, left, target=len(normalized_tokens) - 1)
            if idx is not None:
                return max(0, min(count - 1, idx))
        if _looks_vowel_term(left):
            current_candidates = [
                idx
                for idx, current in enumerate(normalized_tokens)
                if _term_matches_token(left, current) and _vc_right_matches_current_token(right, current)
            ]
            if current_candidates:
                open_syllable_candidates = [
                    idx for idx in current_candidates if not _token_has_korean_coda(normalized_tokens[idx])
                ]
                if open_syllable_candidates:
                    current_candidates = open_syllable_candidates
                return max(0, min(count - 1, min(current_candidates, key=lambda idx: abs(idx - max(0.0, float(target))))))
        else:
            coda_candidates = [
                idx
                for idx, current in enumerate(normalized_tokens)
                if _token_coda_part(current) == left
            ]
            if coda_candidates:
                return max(0, min(count - 1, min(coda_candidates, key=lambda idx: abs(idx - max(0.0, float(target))))))
        candidates: list[int] = []
        for idx in range(max(0, len(normalized_tokens) - 1)):
            current = normalized_tokens[idx]
            following = normalized_tokens[idx + 1]
            if not _term_matches_token(left, current):
                continue
            if right and not _term_matches_token(right, following) and not following.startswith(right):
                continue
            candidates.append(idx)
        if candidates:
            return max(0, min(count - 1, min(candidates, key=lambda idx: abs(idx - max(0.0, float(target))))))
        if not _looks_vowel_term(left):
            idx = _nearest_token_match_index(normalized_tokens, left, target=scaled_target)
            if idx is not None:
                return max(0, min(count - 1, idx))

    if role == "vv" and len(terms) >= 2:
        left = canonicalize_order_token(terms[0])
        right = canonicalize_order_token(terms[1])
        candidates: list[int] = []
        for idx in range(max(0, len(normalized_tokens) - 1)):
            if _term_matches_token(left, normalized_tokens[idx]) and _term_matches_token(right, normalized_tokens[idx + 1]):
                candidates.append(idx + 1)
        if candidates:
            if _looks_vowel_term(left):
                open_current = [
                    idx + 1
                    for idx in range(max(0, len(normalized_tokens) - 1))
                    if idx + 1 in candidates and not _token_has_korean_coda(normalized_tokens[idx])
                ]
                if open_current:
                    candidates = open_current
            target = max(0.0, pair_target if pair_target is not None else scaled_target)
            return max(0, min(count - 1, min(candidates, key=lambda idx: abs(idx - target))))

    return max(0, min(count - 1, int(round(scaled_target))))


def _scaled_source_order_target(
    row: Mapping[str, object],
    *,
    count: int,
    source_order: int,
) -> float:
    slot_count = max(1, int(count))
    try:
        source_count = max(1, int(row.get("slot_count", slot_count) or slot_count))
    except Exception:
        source_count = slot_count
    if source_count <= 1:
        return 0.0
    if source_count == slot_count:
        return float(source_order)
    pos = min(1.0, max(0.0, (float(source_order) + 1.0) / float(source_count)))
    return pos * float(max(0, slot_count - 1))


def _korean_cvvc_source_pair_target(
    row: Mapping[str, object],
    *,
    count: int,
    source_order: int,
) -> float | None:
    role = str(row.get("alias_role", "") or "").strip().lower()
    try:
        source_count = max(1, int(row.get("slot_count", count) or count))
    except Exception:
        return None
    slot_count = max(1, int(count))
    if source_count < max(3, slot_count + 2):
        return None
    order = max(0, int(source_order))
    if role == "vc":
        return min(float(slot_count - 1), max(0.0, float(order) / 2.0))
    if role == "v" and _korean_cvvc_v_closing_apostrophe(row):
        return min(float(slot_count - 1), max(0.0, float(order) / 2.0))
    if role in {"cv", "v"} and str(row.get("alias", "") or "").strip().startswith(("'", "-")):
        return min(float(slot_count - 1), max(0.0, (float(order) + 1.0) / 2.0))
    return None


def _korean_cvvc_alias_order_terms(alias: object, *, role: str) -> list[str]:
    text = str(alias or "").strip().lower().split("_", 1)[0]
    if not text:
        return []
    parts = [part for part in text.replace("-", " - ").split() if part and part != "-"]
    if not parts:
        return []
    role_text = str(role or "").strip().lower()
    if role_text in {"vc", "vv"} and len(parts) >= 2:
        ordered = [parts[0], parts[-1], *parts[1:-1]]
    elif role_text == "v":
        ordered = [parts[0]] if len(parts) >= 2 and parts[-1].strip("'") == "" else [parts[-1]]
    else:
        ordered = parts
    out: list[str] = []
    for part in ordered:
        token = "".join(ch for ch in _plain_order_token_for_matching(part) if ch.isalnum())
        if token:
            out.append(token)
    return out


def _nearest_token_match_index(tokens: Sequence[str], term: str, *, target: float) -> int | None:
    normalized = _plain_order_token_for_matching(term)
    if not normalized:
        return None
    candidates = [
        (idx, _term_match_score(normalized, token))
        for idx, token in enumerate(tokens)
        if _term_matches_token(normalized, token)
    ]
    if not candidates:
        return None
    return int(max(candidates, key=lambda item: (item[1], -abs(float(item[0]) - float(target))))[0])


def _avoid_terminal_duplicate_cv_slot(tokens: Sequence[str], term: str, selected_idx: int) -> int:
    normalized = _plain_order_token_for_matching(term)
    if not normalized or selected_idx != len(tokens) - 1:
        return int(selected_idx)
    exact = [
        idx
        for idx, token in enumerate(tokens)
        if _plain_order_token_for_matching(token) == normalized
    ]
    if len(exact) >= 2 and exact[-1] == selected_idx:
        return int(exact[-2])
    return int(selected_idx)


def _term_matches_token(term: str, token: str) -> bool:
    return _term_match_score(term, token) > 0


def _term_match_score(term: str, token: str) -> int:
    left = _plain_order_token_for_matching(term)
    right = _plain_order_token_for_matching(token)
    if not left or not right:
        return 0
    if _looks_vowel_term(left):
        vowel_surface, vowel_core = _token_vowel_parts(right)
        if not vowel_surface:
            return 0
        if left == right:
            return 6
        if left == vowel_surface:
            return 5
        if left == vowel_core:
            return 4
        return 0
    if left == right:
        return 7
    cv_surface_score = _korean_cv_surface_match_score(left, right)
    if cv_surface_score > 0:
        return cv_surface_score
    if right.startswith(left):
        return 4
    if right.endswith(left):
        return 3
    if _glide_onset_term_matches_token(left, right):
        return 3
    if _liquid_glide_term_matches_token(left, right):
        return 3
    if left.startswith(right):
        return 2
    if left in right or right in left:
        return 1
    return 0


def _korean_cv_surface_match_score(term: str, token: str) -> int:
    text = _plain_order_token_for_matching(term)
    if not text or _looks_vowel_term(text):
        return 0
    term_surface = ""
    term_surface_pos: int | None = None
    for surface in _KOREAN_VOWEL_SURFACES:
        pos = text.find(surface)
        if pos < 0:
            continue
        if term_surface_pos is None or pos < term_surface_pos or (pos == term_surface_pos and len(surface) > len(term_surface)):
            term_surface_pos = pos
            term_surface = surface
    if term_surface_pos is None or term_surface_pos <= 0:
        return 0
    consonant = text[:term_surface_pos]
    suffix = text[term_surface_pos + len(term_surface) :]
    if suffix:
        return 0
    token_surface, token_core = _token_vowel_parts(token)
    if not token_surface:
        return 0
    term_core = _KOREAN_VOWEL_CORE_BY_SURFACE.get(term_surface, term_surface)
    if term_surface != token_surface and term_core != token_core:
        return 0
    onset = _token_onset_part(token)
    coda = _token_coda_part(token)
    base = _right_onset_base(consonant)
    if coda in {consonant, base} or onset in {consonant, base}:
        return 5
    if not coda and not onset:
        return 5
    if coda.startswith(base) or base.startswith(coda) or onset.startswith(base) or base.startswith(onset):
        return 4
    return 0


_KOREAN_VOWEL_CORE_BY_SURFACE = {
    "weo": "eo",
    "yeo": "eo",
    "eu": "eu",
    "eo": "eo",
    "wa": "a",
    "wi": "i",
    "we": "e",
    "wo": "o",
    "ya": "a",
    "ye": "e",
    "yo": "o",
    "yu": "u",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}
_KOREAN_VOWEL_SURFACES = tuple(
    sorted(_KOREAN_VOWEL_CORE_BY_SURFACE, key=lambda item: (-len(item), item))
)
_KOREAN_VOWEL_TERMS = frozenset(_KOREAN_VOWEL_CORE_BY_SURFACE) | frozenset(
    _KOREAN_VOWEL_CORE_BY_SURFACE.values()
)


def _looks_vowel_term(term: str) -> bool:
    text = _plain_order_token_for_matching(term)
    return bool(text) and text in _KOREAN_VOWEL_TERMS


def _token_vowel_parts(token: str) -> tuple[str, str]:
    text = _plain_order_token_for_matching(token)
    if not text:
        return "", ""
    for surface in _KOREAN_VOWEL_SURFACES:
        if surface in text:
            return surface, _KOREAN_VOWEL_CORE_BY_SURFACE.get(surface, surface)
    return "", ""


def _liquid_glide_term_matches_token(term: str, token: str) -> bool:
    left = _plain_order_token_for_matching(term)
    right = _plain_order_token_for_matching(token)
    if left not in {"ry", "ly", "rw", "lw"} or not right.startswith(("r", "l")):
        return False
    surface, core = _token_vowel_parts(right)
    if left.endswith("y"):
        return surface in {"i", "ya", "ye", "yo", "yu", "yeo"} or core in {"i", "e", "eo"}
    return surface in {"u", "wa", "wi", "we", "wo", "weo"} or core == "u"


def _glide_onset_term_matches_token(term: str, token: str) -> bool:
    left = _plain_order_token_for_matching(term)
    right = _plain_order_token_for_matching(token)
    if len(left) < 2 or not left.endswith(("y", "w")):
        return False
    onset = _token_onset_part(right)
    base = left[:-1]
    if not onset or not base or not onset.startswith(base):
        return False
    surface, core = _token_vowel_parts(right)
    if left.endswith("y"):
        return surface in {"i", "ya", "ye", "yo", "yu", "yeo"} or core in {"i", "e", "eo"}
    return surface in {"u", "wa", "wi", "we", "wo", "weo"} or core == "u"


def _vc_right_matches_current_token(right_term: str, token: str) -> bool:
    right = _plain_order_token_for_matching(right_term)
    current = _plain_order_token_for_matching(token)
    if not right or not current:
        return True
    if _token_coda_part(current) == right:
        return True
    surface, core = _token_vowel_parts(current)
    if right == "y" and (surface in {"ya", "ye", "yo", "yu", "yeo"} or core in {"i", "e", "eo"}):
        return True
    if right == "w" and (surface in {"wa", "wi", "we", "wo", "weo"} or core == "u"):
        return True
    onset = _token_onset_part(current)
    onset_base = _right_onset_base(right)
    if onset and onset_base and (onset.startswith(onset_base) or onset_base.startswith(onset)):
        return True
    return _glide_onset_term_matches_token(right, current) or _liquid_glide_term_matches_token(right, current)


def _token_onset_part(token: str) -> str:
    text = _plain_order_token_for_matching(token)
    if not text:
        return ""
    best_pos: int | None = None
    best_surface = ""
    for surface in _KOREAN_VOWEL_SURFACES:
        pos = text.find(surface)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos or (pos == best_pos and len(surface) > len(best_surface)):
            best_pos = pos
            best_surface = surface
    if best_pos is None:
        return text
    return text[:best_pos]


def _token_coda_part(token: str) -> str:
    text = _plain_order_token_for_matching(token)
    if not text:
        return ""
    best_pos: int | None = None
    best_surface = ""
    for surface in _KOREAN_VOWEL_SURFACES:
        pos = text.find(surface)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos or (pos == best_pos and len(surface) > len(best_surface)):
            best_pos = pos
            best_surface = surface
    if best_pos is None:
        return ""
    return text[best_pos + len(best_surface) :]


def _token_has_korean_coda(token: str) -> bool:
    return bool(_token_coda_part(token))


def _korean_cvvc_token_is_vowel_slot(token: object) -> bool:
    text = _plain_order_token_for_matching(token)
    if not text or text in {"cl", "br", "pau", "rest", "r"}:
        return False
    surface, _core = _token_vowel_parts(text)
    return bool(surface)


def _right_onset_base(term: str) -> str:
    text = _plain_order_token_for_matching(term)
    if text in {"ng", "n", "m", "l"}:
        return text
    if text.endswith(("w", "y")) and len(text) > 1:
        return text[:-1]
    return text


def _plain_order_token_for_matching(token: object) -> str:
    return str(token or "").strip().lower().strip("*-_")


def _time_norms(tracks: ManualOtoCandidateTracks, indices: list[int]) -> dict[int, float]:
    duration = max(1.0, float(tracks.duration_ms or 1.0))
    return {
        idx: min(1.0, max(0.0, float(tracks.times_ms[idx]) / duration))
        for idx in indices
        if 0 <= idx < tracks.times_ms.size
    }


def _candidate_near_expected_ms(
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    expected_ms: float,
    *,
    window_ms: float,
) -> float:
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size == 0:
        return max(0.0, float(expected_ms))
    duration_ms = max(1.0, float(tracks.duration_ms or 1.0))
    expected = min(duration_ms, max(0.0, float(expected_ms)))
    indices = [int(idx) for idx in tracks.candidate_indices.get(anchor, ()) if 0 <= int(idx) < int(times.size)]
    if not indices:
        return float(times[int(np.argmin(np.abs(times - expected)))])
    idxs = np.asarray(indices, dtype=np.int32)
    distances = np.abs(times[idxs] - expected)
    local_mask = distances <= float(window_ms)
    if np.any(local_mask):
        idxs = idxs[local_mask]
        distances = distances[local_mask]
    else:
        nearest = int(idxs[int(np.argmin(distances))])
        return float(times[nearest])
    score_track = np.asarray(tracks.anchor_scores.get(anchor, np.zeros((times.size,), dtype=np.float32)), dtype=np.float32)
    if score_track.size != times.size:
        score_track = np.resize(score_track, (times.size,)).astype(np.float32)
    local_scores = score_track[idxs]
    score = 0.25 * _unit_vector(local_scores) - 0.75 * np.clip(distances / max(1.0, float(window_ms)), 0.0, 2.0)
    return float(times[int(idxs[int(np.argmax(score))])])


def _predict_island_anchor_ms(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    anchor: str,
    island: VowelIsland,
    relative_anchor_priors: Mapping[str, object] | None,
    island_anchor_priors: Mapping[str, object] | None = None,
) -> float | None:
    pre_ms = max(0.0, float(island.start_ms))
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    ratio = _island_anchor_ratio(island_anchor_priors, row, anchor)
    if ratio is not None:
        span_ms = max(20.0, float(island.end_ms) - float(island.start_ms))
        expected = float(island.start_ms) + float(ratio) * span_ms
        window = 90.0 if anchor in {"preutterance", "fixed_end"} else 130.0
        return min(duration_ms, max(0.0, _candidate_near_expected_ms(tracks, anchor, expected, window_ms=window)))
    if anchor == "preutterance":
        return min(duration_ms, _default_island_preutterance_ms(row, island))
    if anchor == "fixed_end":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "fixed_end_from_preutterance", row)
        if gap is None:
            gap = _default_fixed_gap_ms(row)
        return min(duration_ms, max(pre_ms + 20.0, min(float(island.end_ms), pre_ms + float(gap))))
    if anchor == "overlap":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "overlap_from_preutterance", row)
        if gap is None:
            return None
        return min(duration_ms, max(0.0, pre_ms - float(gap)))
    if anchor == "offset":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "offset_from_preutterance", row)
        if gap is None:
            return None
        return _dependent_offset_ms(row, tracks, preutterance_ms=pre_ms, relative_anchor_priors=relative_anchor_priors)
    return None


def _default_island_preutterance_ms(row: Mapping[str, object], island: VowelIsland) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    language = str(row.get("language", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    start = float(island.start_ms)
    end = float(island.end_ms)
    span = max(1.0, end - start)
    if language in {"ko", "kr", "korean"} and format_type == "cvvc":
        if role == "vc":
            return end
        if role in {"v"}:
            profile = _korean_cvvc_v_timing_profile(row)
            if profile is not None and "pre_duration_ratio" in profile:
                duration = max(1.0, float(row.get("duration_ms", end) or end or 1.0))
                return min(duration, max(0.0, duration * float(profile["pre_duration_ratio"])))
            if _korean_cvvc_v_alias_is_tail(row):
                return end
            return start + 0.12 * span
        if role == "vv":
            return start
        return start + 0.12 * span
    if _is_japanese_row(row) and role in {"cv_head", "vcv"}:
        return start
    if role in {"vc", "vcv"}:
        return end
    if role == "vv":
        return start + 0.35 * span
    return start


def _korean_cvvc_island_preutterance_ms(
    row: Mapping[str, object],
    island: VowelIsland | None,
) -> float | None:
    if island is None:
        return None
    if not _is_korean_cvvc_row(row):
        return None
    return _default_island_preutterance_ms(row, island)


def _cv_landmark_row(row: Mapping[str, object]) -> bool:
    role = str(row.get("alias_role", "") or "").strip().lower()
    if (
        role != "cv"
        and not (_is_korean_cvvc_row(row) and role == "v" and not _korean_cvvc_v_alias_is_tail(row))
        and not (_is_japanese_row(row) and role in {"cv_head", "vcv"})
    ):
        return False
    family = manual_oto_alias_family(row.get("alias", ""))
    return family != "breath_silence"


def _tail_landmark_row(row: Mapping[str, object]) -> bool:
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role not in {"vc"} and not (_is_korean_cvvc_row(row) and role == "v" and _korean_cvvc_v_alias_is_tail(row)):
        return False
    family = manual_oto_alias_family(row.get("alias", ""))
    return family != "breath_silence"


def _vv_landmark_row(row: Mapping[str, object]) -> bool:
    return str(row.get("alias_role", "") or "").strip().lower() == "vv"


def _track_array(tracks: ManualOtoCandidateTracks, key: str, size: int) -> np.ndarray:
    values = np.asarray(tracks.tracks.get(key, np.zeros((size,), dtype=np.float32)), dtype=np.float32)
    if values.size != size:
        values = np.resize(values, (size,)).astype(np.float32)
    return values


def _nearest_time_index(times: np.ndarray, time_ms: float) -> int:
    if times.size == 0:
        return 0
    idx = int(np.searchsorted(times, float(time_ms), side="left"))
    if idx <= 0:
        return 0
    if idx >= times.size:
        return int(times.size - 1)
    left = idx - 1
    return int(left if abs(float(times[left]) - float(time_ms)) <= abs(float(times[idx]) - float(time_ms)) else idx)


def _window_indices(times: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    if times.size == 0:
        return np.zeros((0,), dtype=np.int32)
    start = min(float(start_ms), float(end_ms))
    end = max(float(start_ms), float(end_ms))
    return np.flatnonzero((times >= start) & (times <= end)).astype(np.int32)


def _role_acoustic_landmarks(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    island: VowelIsland | None,
    assignment: SlotIslandAssignment | None,
    islands: tuple[VowelIsland, ...],
) -> dict[str, object] | None:
    if _cv_landmark_row(row):
        return _cv_acoustic_landmarks(row, tracks, island)
    if _tail_landmark_row(row):
        return _tail_acoustic_landmarks(row, tracks, island)
    if _vv_landmark_row(row):
        previous = _previous_island_for_assignment(assignment, islands)
        return _vv_acoustic_landmarks(row, tracks, previous, island)
    return None


def _previous_island_for_assignment(
    assignment: SlotIslandAssignment | None,
    islands: tuple[VowelIsland, ...],
) -> VowelIsland | None:
    if assignment is None or assignment.island_index is None:
        return None
    idx = int(assignment.island_index) - 1
    if idx < 0 or idx >= len(islands):
        return None
    return islands[idx]


def _cv_acoustic_landmarks(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    island: VowelIsland | None,
) -> dict[str, object] | None:
    if island is None or not _cv_landmark_row(row):
        return None
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    size = int(times.size)
    if size <= 0:
        return None

    transition = _unit_vector(_track_array(tracks, "transition", size))
    transient = _unit_vector(_track_array(tracks, "transient", size))
    vowel_onset = _unit_vector(_track_array(tracks, "vowel_onset", size))
    stable_vowel = _unit_vector(_track_array(tracks, "stable_vowel", size))
    vowel_offset = _unit_vector(_track_array(tracks, "vowel_offset", size))
    vowel_energy = _unit_vector(_track_array(tracks, "vowel_energy", size))
    nucleus = _unit_vector(_track_array(tracks, "nucleus", size))
    rms = _unit_vector(_track_array(tracks, "rms", size))
    activity_edge = _unit_vector(_track_array(tracks, "activity_edge", size))
    voicing = _unit_vector(_track_array(tracks, "voicing", size))
    stability = _unit_vector(_track_array(tracks, "world_stability", size))
    silence = np.clip(_track_array(tracks, "silence", size), 0.0, 1.0)
    vowel_rise = _unit_vector(np.maximum(0.0, np.gradient(vowel_energy)) if vowel_energy.size > 1 else vowel_energy)
    vowel_fall = _unit_vector(np.maximum(0.0, -np.gradient(vowel_energy)) if vowel_energy.size > 1 else vowel_energy)

    island_start = float(island.start_ms)
    island_end = float(island.end_ms)
    nucleus_hint = float(island.nucleus_ms)
    onset_window = _window_indices(
        times,
        max(0.0, min(float(island.left_valley_ms), island_start) - 35.0),
        min(island_end, max(island_start + 42.0, min(nucleus_hint - 8.0, island_start + 150.0))),
    )
    if onset_window.size == 0:
        onset_window = _window_indices(times, max(0.0, island_start - 55.0), min(island_end, island_start + 120.0))
    if onset_window.size == 0:
        onset_idx = _nearest_time_index(times, island_start)
    else:
        distance = np.abs(times[onset_window] - island_start) / 105.0
        score = (
            0.30 * vowel_onset[onset_window]
            + 0.21 * transition[onset_window]
            + 0.19 * transient[onset_window]
            + 0.12 * vowel_rise[onset_window]
            + 0.08 * activity_edge[onset_window]
            + 0.06 * rms[onset_window]
            + 0.04 * (1.0 - silence[onset_window])
            - 0.22 * np.clip(distance, 0.0, 2.0)
        )
        onset_idx = int(onset_window[int(np.argmax(score))])

    onset_ms = float(times[onset_idx])
    transient_window = _window_indices(times, max(0.0, onset_ms - 85.0), min(island_end, onset_ms + 28.0))
    if transient_window.size == 0:
        transient_idx = onset_idx
    else:
        score = 0.48 * transient[transient_window] + 0.30 * transition[transient_window] + 0.14 * activity_edge[transient_window] + 0.08 * vowel_onset[transient_window]
        transient_idx = int(transient_window[int(np.argmax(score))])

    nucleus_window = _window_indices(times, max(island_start, onset_ms + 12.0), island_end)
    if nucleus_window.size == 0:
        nucleus_idx = _nearest_time_index(times, nucleus_hint)
    else:
        edge = np.minimum(
            np.abs(times[nucleus_window] - island_start),
            np.abs(times[nucleus_window] - island_end),
        ) / max(1.0, island_end - island_start)
        score = (
            0.32 * nucleus[nucleus_window]
            + 0.22 * stable_vowel[nucleus_window]
            + 0.18 * voicing[nucleus_window]
            + 0.14 * stability[nucleus_window]
            + 0.09 * rms[nucleus_window]
            + 0.05 * vowel_energy[nucleus_window]
            - 0.16 * np.clip(edge, 0.0, 1.0)
        )
        nucleus_idx = int(nucleus_window[int(np.argmax(score))])

    stable_window = _window_indices(times, min(island_end, onset_ms + 8.0), min(island_end, float(times[nucleus_idx]) + 32.0))
    if stable_window.size == 0:
        stable_idx = nucleus_idx
    else:
        local_stable = stable_vowel[stable_window]
        threshold = max(0.34, float(np.percentile(local_stable, 55.0)) if local_stable.size else 0.34)
        threshold_hits = stable_window[local_stable >= threshold]
        if threshold_hits.size:
            stable_idx = int(threshold_hits[0])
        else:
            distance = np.abs((times[stable_window] - onset_ms) - 48.0) / 120.0
            score = 0.42 * stable_vowel[stable_window] + 0.24 * nucleus[stable_window] + 0.18 * voicing[stable_window] + 0.16 * rms[stable_window] - 0.12 * distance
            stable_idx = int(stable_window[int(np.argmax(score))])

    offset_window = _window_indices(times, max(float(times[nucleus_idx]) + 8.0, island_start), min(float(tracks.duration_ms), island_end + 90.0))
    if offset_window.size == 0:
        offset_idx = _nearest_time_index(times, island_end)
    else:
        distance = np.abs(times[offset_window] - island_end) / 120.0
        score = 0.36 * vowel_offset[offset_window] + 0.20 * transition[offset_window] + 0.18 * vowel_fall[offset_window] + 0.14 * activity_edge[offset_window] + 0.12 * silence[offset_window] - 0.16 * np.clip(distance, 0.0, 2.0)
        offset_idx = int(offset_window[int(np.argmax(score))])

    selected_scores = [
        float(vowel_onset[onset_idx]),
        float(transient[transient_idx]),
        float(stable_vowel[stable_idx]),
        float(nucleus[nucleus_idx]),
        float(vowel_offset[offset_idx]),
    ]
    confidence = float(np.mean(selected_scores))
    warnings: list[str] = []
    if confidence < 0.26:
        warnings.append("low_cv_landmark_confidence")
    if abs(onset_ms - island_start) > 85.0:
        warnings.append("vowel_onset_far_from_island_start")
    if float(times[nucleus_idx]) < onset_ms:
        warnings.append("nucleus_before_vowel_onset")

    return {
        "source": "acoustic_cv_landmark_v1",
        "preutterance_source": "cv_landmark_vowel_onset",
        "transient_onset_ms": float(times[transient_idx]),
        "consonant_release_ms": float(times[transient_idx]),
        "vowel_onset_ms": onset_ms,
        "stable_vowel_start_ms": float(times[stable_idx]),
        "vowel_nucleus_ms": float(times[nucleus_idx]),
        "vowel_offset_ms": float(times[offset_idx]),
        "confidence": confidence,
        "warnings": warnings,
        "scores": {
            "transient": float(transient[transient_idx]),
            "vowel_onset": float(vowel_onset[onset_idx]),
            "stable_vowel": float(stable_vowel[stable_idx]),
            "vowel_nucleus": float(nucleus[nucleus_idx]),
            "vowel_offset": float(vowel_offset[offset_idx]),
        },
    }


def _tail_acoustic_landmarks(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    island: VowelIsland | None,
) -> dict[str, object] | None:
    if island is None or not _tail_landmark_row(row):
        return None
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    size = int(times.size)
    if size <= 0:
        return None
    transition = _unit_vector(_track_array(tracks, "transition", size))
    vowel_offset = _unit_vector(_track_array(tracks, "vowel_offset", size))
    vowel_energy = _unit_vector(_track_array(tracks, "vowel_energy", size))
    activity_edge = _unit_vector(_track_array(tracks, "activity_edge", size))
    rms = _unit_vector(_track_array(tracks, "rms", size))
    silence = np.clip(_track_array(tracks, "silence", size), 0.0, 1.0)
    vowel_fall = _unit_vector(np.maximum(0.0, -np.gradient(vowel_energy)) if vowel_energy.size > 1 else vowel_energy)

    start = float(island.start_ms)
    end = float(island.end_ms)
    idxs = _window_indices(times, max(start, float(island.nucleus_ms) + 12.0), min(float(tracks.duration_ms), end + 110.0))
    if idxs.size == 0:
        offset_idx = _nearest_time_index(times, end)
    else:
        distance = np.abs(times[idxs] - end) / 130.0
        score = (
            0.34 * vowel_offset[idxs]
            + 0.20 * transition[idxs]
            + 0.17 * vowel_fall[idxs]
            + 0.13 * activity_edge[idxs]
            + 0.09 * silence[idxs]
            + 0.07 * (1.0 - rms[idxs])
            - 0.20 * np.clip(distance, 0.0, 2.0)
        )
        offset_idx = int(idxs[int(np.argmax(score))])
    confidence = float(
        np.mean(
            [
                float(vowel_offset[offset_idx]),
                float(transition[offset_idx]),
                float(activity_edge[offset_idx]),
                float(1.0 - min(1.0, max(0.0, abs(float(times[offset_idx]) - end) / 130.0))),
            ]
        )
    )
    warnings: list[str] = []
    if confidence < 0.22:
        warnings.append("low_tail_landmark_confidence")
    if abs(float(times[offset_idx]) - end) > 120.0:
        warnings.append("vowel_offset_far_from_island_end")
    return {
        "source": "acoustic_tail_landmark_v1",
        "preutterance_source": "tail_landmark_vowel_offset",
        "vowel_offset_ms": float(times[offset_idx]),
        "vowel_nucleus_ms": float(island.nucleus_ms),
        "transition_tail_ms": float(times[offset_idx]),
        "confidence": confidence,
        "warnings": warnings,
        "scores": {
            "vowel_offset": float(vowel_offset[offset_idx]),
            "transition": float(transition[offset_idx]),
            "activity_edge": float(activity_edge[offset_idx]),
        },
    }


def _vv_acoustic_landmarks(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    previous_island: VowelIsland | None,
    island: VowelIsland | None,
) -> dict[str, object] | None:
    if island is None or previous_island is None or not _vv_landmark_row(row):
        return None
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    size = int(times.size)
    if size <= 0:
        return None
    transition = _unit_vector(_track_array(tracks, "transition", size))
    activity_edge = _unit_vector(_track_array(tracks, "activity_edge", size))
    vowel_energy = _unit_vector(_track_array(tracks, "vowel_energy", size))
    stable_vowel = _unit_vector(_track_array(tracks, "stable_vowel", size))
    rms = _unit_vector(_track_array(tracks, "rms", size))

    left_nucleus = float(previous_island.nucleus_ms)
    right_start = float(island.start_ms)
    right_nucleus = float(island.nucleus_ms)
    expected = left_nucleus + 0.56 * max(1.0, right_start - left_nucleus)
    idxs = _window_indices(times, max(float(previous_island.start_ms), left_nucleus - 40.0), min(right_nucleus, right_start + 80.0))
    if idxs.size == 0:
        valley_idx = _nearest_time_index(times, right_start)
    else:
        distance = np.abs(times[idxs] - expected) / max(90.0, right_start - left_nucleus)
        score = (
            0.24 * (1.0 - vowel_energy[idxs])
            + 0.22 * transition[idxs]
            + 0.18 * activity_edge[idxs]
            + 0.16 * (1.0 - stable_vowel[idxs])
            + 0.08 * (1.0 - rms[idxs])
            - 0.20 * np.clip(distance, 0.0, 2.0)
        )
        valley_idx = int(idxs[int(np.argmax(score))])
    confidence = float(
        np.mean(
            [
                float(transition[valley_idx]),
                float(activity_edge[valley_idx]),
                float(1.0 - vowel_energy[valley_idx]),
                float(1.0 - min(1.0, max(0.0, abs(float(times[valley_idx]) - expected) / max(90.0, right_start - left_nucleus)))),
            ]
        )
    )
    warnings: list[str] = []
    if confidence < 0.24:
        warnings.append("low_vv_landmark_confidence")
    if not (float(previous_island.start_ms) <= float(times[valley_idx]) <= right_nucleus):
        warnings.append("vv_transition_outside_expected_span")
    return {
        "source": "acoustic_vv_landmark_v1",
        "preutterance_source": "japanese_vv_landmark_blend" if _is_japanese_row(row) else "vv_landmark_transition_valley",
        "transition_valley_ms": float(times[valley_idx]),
        "left_vowel_nucleus_ms": left_nucleus,
        "right_vowel_onset_ms": right_start,
        "right_vowel_nucleus_ms": right_nucleus,
        "confidence": confidence,
        "warnings": warnings,
        "scores": {
            "transition": float(transition[valley_idx]),
            "activity_edge": float(activity_edge[valley_idx]),
            "inverse_vowel_energy": float(1.0 - vowel_energy[valley_idx]),
        },
    }


def _landmark_preutterance_ms(
    row: Mapping[str, object],
    landmarks: Mapping[str, object] | None,
    island: VowelIsland | None,
    *,
    fallback_ms: float,
) -> float | None:
    if not isinstance(landmarks, Mapping):
        return None
    role = str(row.get("alias_role", "") or "").strip().lower()
    try:
        confidence = float(landmarks.get("confidence", 0.0) or 0.0)
    except Exception:
        return None
    if confidence < 0.18:
        return None
    if _is_japanese_row(row) and role in {"cv", "cv_head", "vcv"}:
        selected = _japanese_cv_landmark_preutterance_ms(row, landmarks, fallback_ms=float(fallback_ms))
        if selected is None:
            return None
    elif _is_japanese_row(row) and role == "vv":
        selected = _japanese_vv_landmark_preutterance_ms(row, landmarks, fallback_ms=float(fallback_ms))
        if selected is None:
            return None
    elif role == "cv" and _is_korean_cvvc_row(row):
        selected = _korean_cvvc_cv_landmark_preutterance_ms(row, landmarks, fallback_ms=float(fallback_ms))
        if selected is None:
            return None
    elif role in {"cv", "v"} and not (_is_korean_cvvc_row(row) and role == "v" and _korean_cvvc_v_alias_is_tail(row)):
        selected = landmarks.get("vowel_onset_ms", fallback_ms)
    elif role in {"vc", "v"}:
        return None
    elif role == "vv":
        return None
    else:
        return None
    try:
        pred = float(selected or fallback_ms)
    except Exception:
        return None
    if not np.isfinite(pred):
        return None
    if island is None:
        return max(0.0, pred)
    if _is_japanese_row(row) and role in {"cv", "cv_head", "vcv"}:
        lead_ms, lag_ms, max_refine = _japanese_cv_landmark_window_ms(row)
        lo = max(0.0, pred - lead_ms)
        hi = min(float(island.end_ms), pred + lag_ms)
    elif _is_japanese_row(row) and role == "vv":
        try:
            valley = float(landmarks.get("transition_valley_ms", pred) or pred)
            right_onset = float(landmarks.get("right_vowel_onset_ms", pred) or pred)
        except Exception:
            valley = pred
            right_onset = pred
        lo = max(0.0, min(valley, right_onset) - 90.0)
        hi = min(
            float(island.end_ms),
            max(
                float(island.start_ms) + 170.0,
                right_onset + 130.0,
                valley + 170.0,
            ),
        )
        max_refine = 620.0
    elif role in {"cv", "v"}:
        pre_class = _korean_cvvc_cv_timing_class(row)
        lead_ms = 135.0 if pre_class == "initial_consonant" else 90.0 if pre_class == "vf" else 45.0
        lo = max(0.0, min(float(island.left_valley_ms), float(island.start_ms)) - lead_ms)
        hi = min(float(island.end_ms), max(float(island.start_ms) + 155.0, float(island.nucleus_ms)))
        max_refine = 220.0 if pre_class == "initial_consonant" else 150.0 if pre_class == "vf" else 130.0
    elif role in {"vc", "v"}:
        lo = max(0.0, float(island.nucleus_ms) - 30.0)
        hi = min(float(island.end_ms) + 120.0, max(float(island.end_ms), float(fallback_ms)) + 120.0)
        max_refine = 150.0
    else:
        lo = max(0.0, float(island.start_ms) - 360.0)
        hi = min(float(island.nucleus_ms), float(island.start_ms) + 100.0)
        max_refine = 240.0
    pred = min(hi, max(lo, pred))
    if abs(pred - float(fallback_ms)) > max_refine and confidence < 0.46:
        return float(fallback_ms)
    return max(0.0, float(pred))


def _japanese_vv_landmark_preutterance_ms(
    row: Mapping[str, object],
    landmarks: Mapping[str, object],
    *,
    fallback_ms: float,
) -> float | None:
    try:
        valley = float(landmarks.get("transition_valley_ms", fallback_ms) or fallback_ms)
        right_onset = float(landmarks.get("right_vowel_onset_ms", fallback_ms) or fallback_ms)
    except Exception:
        return None
    if not (np.isfinite(valley) and np.isfinite(right_onset)):
        return None
    bridge_ms = right_onset - valley
    pred = valley + (0.75 * bridge_ms) + 80.0
    pred = min(right_onset + 110.0, pred)
    pred = max(valley + 20.0, pred)
    return max(0.0, float(pred))


def _japanese_cv_landmark_preutterance_ms(
    row: Mapping[str, object],
    landmarks: Mapping[str, object],
    *,
    fallback_ms: float,
) -> float | None:
    try:
        onset = float(landmarks.get("vowel_onset_ms", fallback_ms) or fallback_ms)
    except Exception:
        return None
    if not np.isfinite(onset):
        return None
    return max(0.0, onset)


def _japanese_cv_landmark_window_ms(row: Mapping[str, object]) -> tuple[float, float, float]:
    role = str(row.get("alias_role", "") or "").strip().lower()
    onset_class = _manual_anchor_japanese_onset_class(row)
    if role == "vcv":
        lead_ms, lag_ms, max_refine = 90.0, 58.0, 520.0
    elif role == "cv_head":
        lead_ms, lag_ms, max_refine = 72.0, 78.0, 220.0
    else:
        lead_ms, lag_ms, max_refine = 52.0, 72.0, 170.0
    if onset_class in {"voiceless", "plosive", "sibilant", "fricative"}:
        lead_ms += 10.0
        lag_ms = max(34.0, lag_ms - 8.0)
    elif onset_class in {"nasal", "liquid", "glide"}:
        lead_ms = max(24.0, lead_ms - 10.0)
        lag_ms += 8.0
    return float(lead_ms), float(lag_ms), float(max_refine)


def _korean_cvvc_transition_preutterance_ms(
    row: Mapping[str, object],
    island: VowelIsland | None,
    landmarks: Mapping[str, object] | None,
    *,
    fallback_ms: float,
) -> tuple[float, str] | None:
    if island is None or not _is_korean_cvvc_row(row):
        return None
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role == "vc":
        if _korean_cvvc_following_onset_style(row):
            start = float(island.start_ms)
            gap = _korean_cvvc_following_onset_pre_gap_ms(row)
            return max(0.0, start - gap), "korean_cvvc_vc_following_onset"
        profile = _korean_cvvc_transition_profile(row)
        if profile is None:
            return None
        ratio = profile.get("pre_ratio")
        if ratio is None:
            return None
        start = float(island.start_ms)
        end = float(island.end_ms)
        span = max(1.0, end - start)
        if span < 80.0:
            return None
        pred = start + min(0.95, max(0.05, float(ratio))) * span
        return max(0.0, min(end, max(start, float(pred)))), f"korean_cvvc_vc_transition_{profile['right']}"
    if role == "vv":
        if not isinstance(landmarks, Mapping):
            return None
        try:
            confidence = float(landmarks.get("confidence", 0.0) or 0.0)
            valley = float(landmarks.get("transition_valley_ms", fallback_ms) or fallback_ms)
            right_onset = float(landmarks.get("right_vowel_onset_ms", island.start_ms) or island.start_ms)
        except Exception:
            return None
        if not (np.isfinite(confidence) and np.isfinite(valley) and np.isfinite(right_onset)):
            return None
        if confidence < 0.18:
            return None
        profile_pre = _korean_cvvc_vv_profile_preutterance_ms(
            row,
            island,
            valley_ms=valley,
            right_onset_ms=right_onset,
        )
        if profile_pre is not None:
            return profile_pre
        if right_onset - 240.0 <= valley <= right_onset + 70.0:
            return max(0.0, float(valley)), "korean_cvvc_vv_transition_valley"
    return None


def _korean_cvvc_vv_profile_preutterance_ms(
    row: Mapping[str, object],
    island: VowelIsland,
    *,
    valley_ms: float,
    right_onset_ms: float,
) -> tuple[float, str] | None:
    terms = _korean_cvvc_alias_order_terms(str(row.get("alias", "") or ""), role="vv")
    if len(terms) < 2:
        return None
    left = _plain_order_token_for_matching(terms[0])
    right = _plain_order_token_for_matching(terms[1])
    if not left or not right:
        return None
    start = float(island.start_ms)
    nucleus = float(island.nucleus_ms)
    end = float(island.end_ms)
    valley = float(valley_ms)
    right_onset = float(right_onset_ms)
    if not all(np.isfinite(value) for value in (start, nucleus, end, valley, right_onset)):
        return None

    nucleus_span = max(0.0, nucleus - start)
    island_span = max(0.0, end - start)
    valley_before_current = valley <= start + 1e-3
    valley_inside_current = start < valley <= start + 130.0

    if left == right == "a" and valley <= start + 30.0 and island_span >= 300.0:
        return max(0.0, start + 0.48 * island_span), "korean_cvvc_vv_profile_open_repeat"
    if right == "eo":
        if valley_inside_current:
            return max(0.0, valley), "korean_cvvc_vv_profile_right_eo_valley"
        if valley_before_current and nucleus_span >= 80.0:
            return max(0.0, start + 0.60 * nucleus_span), "korean_cvvc_vv_profile_right_eo"
    if left == "u" and right in {"e", "o", "u", "eu"} and nucleus_span >= 80.0:
        ratio = 0.84 if right == "e" else 0.60
        return max(0.0, start + ratio * nucleus_span), f"korean_cvvc_vv_profile_u_{right}"
    if right == "o" and left in {"o", "u", "eu"} and valley_before_current and nucleus_span >= 80.0:
        return max(0.0, start + 0.60 * nucleus_span), "korean_cvvc_vv_profile_right_o"
    return None


def _prediction_matches_island(
    row: Mapping[str, object],
    anchor: str,
    *,
    pred_ms: float,
    island: VowelIsland | None,
) -> bool:
    if island is None:
        return True
    pred = float(pred_ms)
    role = str(row.get("alias_role", "") or "").strip().lower()
    start = float(island.start_ms)
    end = float(island.end_ms)
    span = max(1.0, end - start)
    if anchor == "preutterance":
        left_margin = 120.0 if role in {"vc", "vcv", "vv"} else 80.0
        right_margin = max(80.0, 0.30 * span)
        return (start - left_margin) <= pred <= (end + right_margin)
    if anchor == "fixed_end":
        return (start - 60.0) <= pred <= (end + max(120.0, 0.40 * span))
    if anchor == "overlap":
        return (start - max(180.0, 0.70 * span)) <= pred <= (end + 80.0)
    if anchor == "offset":
        return (start - max(260.0, 1.20 * span)) <= pred <= (end + 40.0)
    return True


def _default_offset_gap_ms(row: Mapping[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if role in {"v", "vv"}:
        return 70.0
    if role in {"vc", "vcv"} or family in {"vowel_transition", "leading_n"}:
        return 150.0
    if family == "breath_silence":
        return 40.0
    return 120.0


def _default_fixed_gap_ms(row: Mapping[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    fmt = str(row.get("format_type", "") or "").strip().lower()
    if role in {"vc", "vv"}:
        return 95.0
    if role == "vcv":
        return 115.0
    if fmt in {"cv", "cvc"}:
        return 170.0
    return 135.0


def _default_cutoff_gap_ms(row: Mapping[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role in {"vc", "vv", "v"}:
        return 110.0
    return 220.0


def _korean_cvvc_transition_profile(row: Mapping[str, object]) -> dict[str, float | str] | None:
    if not _is_korean_cvvc_row(row):
        return None
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role != "vc":
        return None
    raw_right = _korean_cvvc_raw_vc_right_token(row)
    raw_profiles: dict[str, dict[str, float]] = {
        "M": {"pre_ratio": 0.75, "fixed_gap": 63.0, "cutoff_tail_gap": 221.0},
        "N": {"pre_ratio": 0.65, "fixed_gap": 63.0, "cutoff_tail_gap": 221.0},
        "P": {"pre_ratio": 0.85, "fixed_gap": 31.0, "cutoff_tail_gap": 224.0},
        "T": {"pre_ratio": 0.90, "fixed_gap": 26.0, "cutoff_tail_gap": 194.0},
        "K": {"pre_ratio": 0.95, "fixed_gap": 15.0, "cutoff_tail_gap": 180.0},
    }
    raw_data = raw_profiles.get(raw_right)
    if raw_data is not None:
        return {"right": raw_right.lower(), "raw_right": raw_right, **raw_data}
    terms = _korean_cvvc_alias_order_terms(row.get("alias", ""), role=role)
    if len(terms) < 2:
        return None
    right = str(terms[1])
    profiles: dict[str, dict[str, float]] = {
        "l": {"pre_ratio": 0.50, "fixed_gap": 61.0, "cutoff_tail_gap": 220.0},
        "ng": {"pre_ratio": 0.60, "fixed_gap": 63.0, "cutoff_tail_gap": 221.0},
        "ly": {"pre_ratio": 0.60, "fixed_gap": 49.0, "cutoff_tail_gap": 177.0},
        "lw": {"pre_ratio": 0.75, "fixed_gap": 63.0, "cutoff_tail_gap": 221.0},
    }
    data = profiles.get(right)
    if data is None:
        return None
    return {"right": right, **data}


def _korean_cvvc_raw_vc_right_token(row: Mapping[str, object]) -> str:
    if not _is_korean_cvvc_row(row):
        return ""
    role = str(row.get("alias_role", "") or "").strip().lower()
    if role != "vc":
        return ""
    parts = str(row.get("alias", "") or "").strip().split()
    if len(parts) < 2:
        return ""
    return parts[-1].strip()


def _korean_cvvc_following_onset_style(row: Mapping[str, object]) -> bool:
    style = str(row.get("vc_pre_style", "") or "").strip().lower()
    return _is_korean_cvvc_row(row) and style in {STYLE_FOLLOWING_ONSET_VC, "following_onset"}


def _korean_cvvc_following_onset_pre_gap_ms(row: Mapping[str, object]) -> float:
    terms = _korean_cvvc_alias_order_terms(row.get("alias", ""), role="vc")
    right = terms[1] if len(terms) >= 2 else ""
    if right in {"r", "rr"}:
        return 115.0
    if right in {"m", "n", "ng", "l"}:
        return 70.0
    return 80.0


def _korean_cvvc_cv_timing_class(row: Mapping[str, object]) -> str:
    if not _is_korean_cvvc_row(row):
        return ""
    if str(row.get("alias_role", "") or "").strip().lower() != "cv":
        return ""
    if _is_korean_cvvc_vf_row(row):
        return "vf"
    alias = str(row.get("alias", "") or "").strip()
    if alias.startswith("-"):
        term = _plain_order_token_for_matching(alias[1:].strip())
        if _korean_cvvc_plain_vowel_term(term):
            return "initial_vowel"
        return "initial_consonant"
    if alias.startswith("'"):
        return "apostrophe"
    first = alias[:1].lower()
    if first in {"n", "l", "r", "m"}:
        return "sonorant_onset"
    return "plain"


def _korean_cvvc_cv_landmark_preutterance_ms(
    row: Mapping[str, object],
    landmarks: Mapping[str, object],
    *,
    fallback_ms: float,
) -> float | None:
    try:
        onset = float(landmarks.get("vowel_onset_ms", fallback_ms) or fallback_ms)
    except Exception:
        return None
    if not np.isfinite(onset):
        return None
    timing_class = _korean_cvvc_cv_timing_class(row)
    if timing_class == "initial_consonant":
        return max(0.0, onset - 62.0)
    if timing_class == "vf":
        return max(0.0, onset - 30.0)
    return max(0.0, onset)


def _korean_cvvc_cv_relative_gaps(row: Mapping[str, object]) -> dict[str, float] | None:
    timing_class = _korean_cvvc_cv_timing_class(row)
    if not timing_class:
        return None
    if timing_class == "initial_consonant":
        return {"offset_gap": 36.0, "overlap_gap": 18.0, "fixed_gap": 18.0}
    if timing_class == "initial_vowel":
        return {"offset_gap": 42.0, "overlap_gap": 21.0, "fixed_gap": 184.0}
    if timing_class == "apostrophe":
        return {"offset_gap": 77.0, "overlap_gap": 38.0, "fixed_gap": 59.0}
    if timing_class == "vf":
        return {"offset_gap": 153.0, "overlap_gap": 76.0, "fixed_gap": 370.0}
    if timing_class == "sonorant_onset":
        return {"offset_gap": 46.0, "overlap_gap": 23.0, "fixed_gap": 142.0}
    return {"offset_gap": 55.0, "overlap_gap": 45.0, "fixed_gap": 142.0}


def _korean_cvvc_cv_cutoff_tail_gap_ms(row: Mapping[str, object]) -> float | None:
    timing_class = _korean_cvvc_cv_timing_class(row)
    if not timing_class:
        return None
    if timing_class == "initial_consonant":
        return 95.0
    if timing_class == "initial_vowel":
        return 390.0
    if timing_class == "apostrophe":
        return 190.0
    if timing_class == "vf":
        return 1530.0
    if timing_class == "sonorant_onset":
        return 315.0
    return 310.0


def _default_utau_cutoff_tail_gap_ms(row: Mapping[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    language = str(row.get("language", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    v_profile = _korean_cvvc_v_timing_profile(row)
    if v_profile is not None:
        return float(v_profile["cutoff_tail_gap"])
    if _is_korean_cvvc_vf_row(row):
        if role == "vc":
            return 610.0
        if role == "cv":
            return float(_korean_cvvc_cv_cutoff_tail_gap_ms(row) or 1530.0)
        return 1045.0
    if family == "breath_silence" or role in {"br", "sil"}:
        return 1200.0
    if language in {"ko", "kr", "korean"} and format_type == "cvvc":
        if role == "vc" and _korean_cvvc_following_onset_style(row):
            return 370.0
        transition_profile = _korean_cvvc_transition_profile(row)
        if transition_profile is not None:
            return float(transition_profile["cutoff_tail_gap"])
        if role == "vc":
            return 115.0
        if role == "cv":
            return float(_korean_cvvc_cv_cutoff_tail_gap_ms(row) or 310.0)
        if role == "vv":
            return 430.0
        if role == "v":
            return 330.0
    return _default_cutoff_gap_ms(row)


def _dependent_cutoff_abs_ms(
    row: Mapping[str, object],
    *,
    duration_ms: float,
    fixed_end_ms: float,
) -> float | None:
    duration = max(1.0, float(duration_ms))
    fixed_abs = min(duration - 1.0, max(0.0, float(fixed_end_ms)))
    tail_gap = _default_utau_cutoff_tail_gap_ms(row)
    cutoff_abs = duration - min(max(1.0, float(tail_gap)), max(1.0, duration - fixed_abs - 1.0))
    return min(duration, max(fixed_abs + 1.0, float(cutoff_abs)))


def _dependent_offset_ms(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    relative_anchor_priors: Mapping[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    korean_gaps = _korean_cvvc_relative_gaps(row)
    if korean_gaps is not None:
        return min(duration_ms, max(0.0, float(preutterance_ms) - float(korean_gaps["offset_gap"])))
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "offset_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is None:
        if bool(require_stable_prior):
            return None
        gap = _default_offset_gap_ms(row)
    expected = float(preutterance_ms) - max(0.0, float(gap))
    window = max(55.0, min(190.0, abs(float(gap)) * 0.75 + 35.0))
    min_gap = 8.0 if str(row.get("alias_role", "") or "").strip().lower() in {"v", "vv"} else 16.0
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size:
        start = max(0.0, expected - window)
        end = min(float(preutterance_ms) - min_gap, expected + window)
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            idxs = np.flatnonzero(mask)
            size = int(times.size)
            offset_score = _unit_vector(
                np.asarray(tracks.anchor_scores.get("offset", np.zeros((size,), dtype=np.float32)), dtype=np.float32)
            )
            transition = _unit_vector(np.asarray(tracks.tracks.get("transition", np.zeros((size,), dtype=np.float32)), dtype=np.float32))
            activity_edge = _unit_vector(
                np.asarray(tracks.tracks.get("activity_edge", np.zeros((size,), dtype=np.float32)), dtype=np.float32)
            )
            distance = np.abs(times[idxs] - expected) / max(1.0, window)
            score = 0.08 * offset_score[idxs] + 0.06 * transition[idxs] + 0.04 * activity_edge[idxs] - np.clip(distance, 0.0, 2.0)
            pred = float(times[int(idxs[int(np.argmax(score))])])
        else:
            pred = expected
    else:
        pred = expected
    return min(duration_ms, max(0.0, min(float(pred), float(preutterance_ms) - min_gap)))


def _default_overlap_ratio(row: Mapping[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if role in {"v", "vv"}:
        return 0.50
    if role in {"vc", "vcv"} or family in {"vowel_transition", "leading_n"}:
        return 1.0 / 3.0
    if family == "breath_silence":
        return 0.0
    return 1.0 / 3.0


def _korean_cvvc_relative_gaps(row: Mapping[str, object]) -> dict[str, float] | None:
    if not _is_korean_cvvc_row(row):
        return None
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    v_profile = _korean_cvvc_v_timing_profile(row)
    if v_profile is not None:
        return {
            "offset_gap": float(v_profile["offset_gap"]),
            "overlap_gap": float(v_profile["overlap_gap"]),
            "fixed_gap": float(v_profile["fixed_gap"]),
        }
    if _is_korean_cvvc_vf_row(row):
        if role == "vc":
            return {"offset_gap": 118.0, "overlap_gap": 59.0, "fixed_gap": 326.0}
        if role == "cv":
            return _korean_cvvc_cv_relative_gaps(row)
        return {"offset_gap": 167.0, "overlap_gap": 84.0, "fixed_gap": 409.0}
    if role in {"br", "sil"} or family == "breath_silence":
        return {"offset_gap": 50.0, "overlap_gap": 25.0, "fixed_gap": 860.0}
    if role == "cv":
        return _korean_cvvc_cv_relative_gaps(row)
    if role == "vc":
        if _korean_cvvc_following_onset_style(row):
            return {"offset_gap": 244.0, "overlap_gap": 167.0, "fixed_gap": 77.0}
        transition_profile = _korean_cvvc_transition_profile(row)
        if transition_profile is not None:
            return {
                "offset_gap": 70.0,
                "overlap_gap": 35.0,
                "fixed_gap": float(transition_profile["fixed_gap"]),
            }
        return {"offset_gap": 70.0, "overlap_gap": 35.0, "fixed_gap": 16.0}
    if role == "vv":
        return {"offset_gap": 87.0, "overlap_gap": 43.0, "fixed_gap": 184.0}
    if role == "v":
        return {"offset_gap": 50.0, "overlap_gap": 25.0, "fixed_gap": 152.0}
    return {"offset_gap": 53.0, "overlap_gap": 32.0, "fixed_gap": 142.0}


def _korean_cvvc_v_timing_profile(row: Mapping[str, object]) -> dict[str, float] | None:
    if not _is_korean_cvvc_row(row):
        return None
    if str(row.get("alias_role", "") or "").strip().lower() != "v":
        return None
    if _korean_cvvc_v_closing_apostrophe(row):
        return {
            "offset_gap": 77.0,
            "overlap_gap": 38.0,
            "fixed_gap": 110.0,
            "cutoff_tail_gap": 220.0,
        }
    parts = str(row.get("alias", "") or "").strip().split()
    if len(parts) < 2:
        return None
    left = parts[0].strip()
    tail = parts[-1].strip()
    if _is_korean_cvvc_vf_row(row) and tail == "*":
        pre_ratio = 0.60 if left == "u" else 0.74
        return {
            "pre_duration_ratio": pre_ratio,
            "offset_gap": 118.0,
            "overlap_gap": 59.0,
            "fixed_gap": 326.0,
            "cutoff_tail_gap": 610.0,
        }
    if tail == "-" and left in {"M", "N", "NG"}:
        return {
            "offset_gap": 50.0,
            "overlap_gap": 25.0,
            "fixed_gap": 12.0,
            "cutoff_tail_gap": 70.0,
        }
    if tail == "-" and left == "L":
        return {
            "offset_gap": 80.0,
            "overlap_gap": 40.0,
            "fixed_gap": 360.0,
            "cutoff_tail_gap": 500.0,
        }
    return None


def _is_korean_cvvc_row(row: Mapping[str, object]) -> bool:
    language = str(row.get("language", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    return language in {"ko", "kr", "korean"} and format_type == "cvvc"


def _is_japanese_row(row: Mapping[str, object]) -> bool:
    language = str(row.get("language", "") or "").strip().lower()
    return language in {"ja", "jp", "japanese"}


def _korean_cvvc_v_alias_is_tail(row: Mapping[str, object]) -> bool:
    if not _is_korean_cvvc_row(row):
        return False
    if str(row.get("alias_role", "") or "").strip().lower() != "v":
        return False
    if _korean_cvvc_v_closing_apostrophe(row):
        return True
    alias = str(row.get("alias", "") or "").strip()
    return bool(alias.endswith(("-", "*")))


def _korean_cvvc_v_closing_apostrophe(row: Mapping[str, object]) -> bool:
    if str(row.get("alias_role", "") or "").strip().lower() != "v":
        return False
    parts = str(row.get("alias", "") or "").strip().split()
    return len(parts) >= 2 and parts[-1].strip("'") == "" and bool(parts[0].strip())


def _is_korean_cvvc_vf_row(row: Mapping[str, object]) -> bool:
    if not _is_korean_cvvc_row(row):
        return False
    wav = str(row.get("wav_name") or row.get("wav_path") or "").strip().lower()
    return "(vf)" in wav


def _dependent_overlap_ms(
    row: Mapping[str, object],
    *,
    offset_ms: float,
    preutterance_ms: float,
    relative_anchor_priors: Mapping[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    offset = float(offset_ms)
    pre = max(offset, float(preutterance_ms))
    korean_gaps = _korean_cvvc_relative_gaps(row)
    if korean_gaps is not None:
        adaptive = _manual_anchor_adaptive_overlap_ms(row, offset_ms=offset, preutterance_ms=pre)
        if adaptive is not None:
            return min(pre, max(offset, float(adaptive)))
        return min(pre, max(offset, pre - float(korean_gaps["overlap_gap"])))
    span = max(0.0, pre - offset)
    ratio = _relative_anchor_ratio(
        relative_anchor_priors,
        "overlap_ratio_from_offset_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if ratio is None:
        ratio = _default_overlap_ratio(row)
    ratio_pred = offset + min(1.0, max(0.0, float(ratio))) * span
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "overlap_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is not None:
        pred = (0.10 * ratio_pred) + (0.90 * (pre - float(gap)))
    elif bool(require_stable_prior):
        return None
    else:
        pred = ratio_pred
    return min(pre, max(offset, float(pred)))


def _dependent_fixed_end_ms(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    relative_anchor_priors: Mapping[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    korean_gaps = _korean_cvvc_relative_gaps(row)
    if korean_gaps is not None:
        return min(duration_ms, max(float(preutterance_ms) + 8.0, float(preutterance_ms) + float(korean_gaps["fixed_gap"])))
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "fixed_end_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is None:
        if bool(require_stable_prior):
            return None
        gap = _default_fixed_gap_ms(row)
    expected = min(duration_ms, max(0.0, float(preutterance_ms) + float(gap)))
    window = max(55.0, min(240.0, abs(float(gap)) * 0.70 + 45.0))
    pred = _candidate_near_expected_ms(tracks, "fixed_end", expected, window_ms=window)
    return min(duration_ms, max(float(preutterance_ms) + 8.0, float(pred)))


def _safe_overlap_gap_config(row: Mapping[str, object], span_ms: float) -> tuple[float, float, float, str]:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    span = max(0.0, float(span_ms))
    if family == "breath_silence":
        min_gap, role_gap, ratio, label = 0.0, 0.0, 0.0, "breath_silence"
    elif role in {"vc", "vcv"} or family in {"leading_n", "vowel_transition"}:
        min_gap, role_gap, ratio, label = 30.0, 85.0, 0.50, "transition"
    elif role in {"v", "vv"}:
        min_gap, role_gap, ratio, label = 20.0, 55.0, 0.50, "vowel"
    else:
        min_gap, role_gap, ratio, label = 18.0, 45.0, 0.38, "cv"
    if span <= 0.0:
        return 0.0, 0.0, 0.0, label
    target_gap = min(float(role_gap), span * float(ratio))
    max_gap = min(span, max(float(min_gap), max(float(role_gap), span * 0.72)))
    min_gap = min(float(min_gap), max_gap)
    gap = min(max_gap, max(min_gap, target_gap))
    return float(gap), float(min_gap), float(max_gap), label


def _safe_overlap_ms(
    row: Mapping[str, object],
    *,
    offset_ms: float,
    preutterance_ms: float,
    preferred_overlap_ms: float | None = None,
) -> tuple[float, dict[str, object]]:
    offset = float(offset_ms)
    pre = max(offset, float(preutterance_ms))
    span = max(0.0, pre - offset)
    gap, min_gap, max_gap, label = _safe_overlap_gap_config(row, span)
    if preferred_overlap_ms is None or not np.isfinite(float(preferred_overlap_ms)):
        preferred_overlap = pre - gap
    else:
        preferred_overlap = min(pre, max(offset, float(preferred_overlap_ms)))
    preferred_gap = max(0.0, pre - preferred_overlap)
    clamped_gap = min(max_gap, max(min_gap, preferred_gap))
    overlap = min(pre, max(offset, pre - clamped_gap))
    return overlap, {
        "policy": label,
        "span_ms": float(span),
        "gap_ms": float(pre - overlap),
        "preferred_gap_ms": float(preferred_gap),
        "min_gap_ms": float(min_gap),
        "max_gap_ms": float(max_gap),
        "clamped": bool(abs(float(pre - overlap) - preferred_gap) > 1e-6),
        "ratio": float((overlap - offset) / span) if span > 0.0 else 0.0,
    }


def _relative_gap_bucket_keys(row: Mapping[str, object]) -> list[str]:
    language = str(row.get("language", "") or "").strip().lower() or "*"
    format_type = str(row.get("format_type", "") or "").strip().lower() or "*"
    role = str(row.get("alias_role", "") or "").strip().lower() or "*"
    family = manual_oto_alias_family(row.get("alias", ""))
    bank = str(row.get("voicebank_id", "") or "").strip().lower() or "*"
    return [
        f"{language}|{format_type}|{role}|{family}|{bank}",
        f"{language}|{format_type}|{role}|*|{bank}",
        f"{language}|{format_type}|{role}|{family}",
        f"{language}|{format_type}|{role}|*",
        f"{format_type}|{role}|{family}|{bank}",
        f"{format_type}|{role}|*|{bank}",
        f"{format_type}|{role}|*",
        f"{role}|*",
        "*",
    ]


def _relative_anchor_gap_ms(
    priors: Mapping[str, object] | None,
    kind: str,
    row: Mapping[str, object],
    *,
    require_stable: bool = False,
) -> float | None:
    if not isinstance(priors, Mapping):
        return None
    buckets = priors.get("buckets")
    if not isinstance(buckets, Mapping):
        return None
    by_kind = buckets.get(kind)
    if not isinstance(by_kind, Mapping):
        return None
    stats_by_kind = priors.get("stats")
    kind_stats = stats_by_kind.get(kind) if isinstance(stats_by_kind, Mapping) else {}
    if not isinstance(kind_stats, Mapping):
        kind_stats = {}
    for key in _relative_gap_bucket_keys(row):
        value = by_kind.get(key)
        if value is None:
            continue
        stat = kind_stats.get(key)
        if bool(require_stable) and isinstance(stat, Mapping) and not bool(stat.get("stable", False)):
            continue
        try:
            gap = float(value)
        except Exception:
            continue
        if np.isfinite(gap):
            return gap
    return None


def _relative_anchor_ratio(
    priors: Mapping[str, object] | None,
    kind: str,
    row: Mapping[str, object],
    *,
    require_stable: bool = False,
) -> float | None:
    return _relative_anchor_gap_ms(priors, kind, row, require_stable=bool(require_stable))


def _island_anchor_bucket_keys(row: Mapping[str, object], anchor: str) -> list[str]:
    language = str(row.get("language", "") or "").strip().lower() or "*"
    format_type = str(row.get("format_type", "") or "").strip().lower() or "*"
    role = str(row.get("alias_role", "") or "").strip().lower() or "*"
    family = manual_oto_alias_family(row.get("alias", ""))
    return [
        f"{language}|{format_type}|{role}|{family}|{anchor}",
        f"{language}|{format_type}|{role}|*|{anchor}",
        f"{format_type}|{role}|*|{anchor}",
        f"{role}|*|{anchor}",
        f"*|{anchor}",
    ]


def _island_anchor_ratio(
    priors: Mapping[str, object] | None,
    row: Mapping[str, object],
    anchor: str,
) -> float | None:
    if not isinstance(priors, Mapping):
        return None
    buckets = priors.get("buckets")
    counts = priors.get("counts")
    if not isinstance(buckets, Mapping):
        return None
    if not isinstance(counts, Mapping):
        counts = {}
    for key in _island_anchor_bucket_keys(row, anchor):
        if int(counts.get(key, 0) or 0) < 6:
            continue
        value = buckets.get(key)
        if value is None:
            continue
        try:
            ratio = float(value)
        except Exception:
            continue
        if np.isfinite(ratio):
            return ratio
    return None


def _island_assignment_for_row(
    row: Mapping[str, object],
    assignments: tuple[SlotIslandAssignment, ...],
    slot_count: int,
) -> SlotIslandAssignment | None:
    if not assignments:
        return None
    idx = _row_vowel_island_slot_index(row, slot_count)
    if idx >= len(assignments):
        return None
    return assignments[idx]


def _island_for_assignment(
    assignment: SlotIslandAssignment | None,
    islands: tuple[VowelIsland, ...],
) -> VowelIsland | None:
    if assignment is None or assignment.island_index is None:
        return None
    idx = int(assignment.island_index)
    if idx < 0 or idx >= len(islands):
        return None
    return islands[idx]


def _manual_anchor_island_context(
    assignment: SlotIslandAssignment | None,
    islands: tuple[VowelIsland, ...],
) -> dict[str, object]:
    if assignment is None or assignment.island_index is None:
        return {"index": None, "previous": None, "current": None, "next": None}
    idx = int(assignment.island_index)
    if idx < 0 or idx >= len(islands):
        return {"index": None, "previous": None, "current": None, "next": None}
    previous = islands[idx - 1] if idx > 0 else None
    current = islands[idx]
    next_island = islands[idx + 1] if idx + 1 < len(islands) else None
    return {"index": idx, "previous": previous, "current": current, "next": next_island}


def _slot_shift_delta(
    row: Mapping[str, object],
    *,
    pred_ms: float,
    duration_ms: float,
    slot_count: int | None = None,
    slot_index: int | None = None,
) -> int | None:
    try:
        count = int(slot_count if slot_count is not None else row.get("slot_count", 1) or 1)
        target_idx = int(slot_index if slot_index is not None else row.get("slot_index", 0) or 0)
    except Exception:
        return None
    if count <= 1:
        return 0
    duration = max(1.0, float(duration_ms))
    pred_pos = min(0.999, max(0.0, float(pred_ms) / duration))
    pred_idx = int(round(pred_pos * float(count) - 0.5))
    pred_idx = max(0, min(count - 1, pred_idx))
    target_idx = max(0, min(count - 1, target_idx))
    return int(pred_idx - target_idx)


def _slot_shift_bucket(delta: int | None) -> str:
    if delta is None:
        return "unknown"
    if int(delta) == 0:
        return "exact"
    if abs(int(delta)) == 1:
        return "one_step"
    return "multi_step"


def _slot_expected_anchor_ms(
    row: Mapping[str, object],
    anchor: str,
    *,
    duration_ms: float,
    slot_count: int,
    slot_index: int,
) -> float:
    count = max(1, int(slot_count))
    idx = max(0, min(count - 1, int(slot_index)))
    duration = max(1.0, float(duration_ms))
    if count <= 1:
        return duration * 0.5
    span = duration / float(count)
    center = (float(idx) + 0.5) * span
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    shifts = {"offset": -0.22, "overlap": -0.15, "preutterance": -0.05, "fixed_end": 0.08}
    if role in {"vc", "vcv", "vv"} or family == "vowel_transition":
        shifts = {"offset": -0.28, "overlap": -0.20, "preutterance": -0.08, "fixed_end": 0.02}
    if family == "leading_n":
        shifts = {"offset": -0.32, "overlap": -0.24, "preutterance": -0.10, "fixed_end": 0.04}
    return min(duration, max(0.0, center + float(shifts.get(anchor, 0.0)) * span))


def _slot_guarded_anchor_ms(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    pred_ms: float,
    *,
    slot_count: int,
    slot_index: int,
) -> float:
    duration = max(1.0, float(tracks.duration_ms or row.get("duration_ms", 1.0) or 1.0))
    delta = _slot_shift_delta(row, pred_ms=pred_ms, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
    count = max(1, int(slot_count))
    if count <= 1 or delta == 0:
        return float(pred_ms)
    expected = _slot_expected_anchor_ms(row, anchor, duration_ms=duration, slot_count=count, slot_index=slot_index)
    window = max(70.0, min(240.0, 0.42 * duration / float(count)))
    return _candidate_near_expected_ms(tracks, anchor, expected, window_ms=window)


def _family_anchor_prior_score(
    row: Mapping[str, object],
    anchor: str,
    *,
    candidate_time_norm: float,
    slot_pos_norm: float,
    weight: float,
) -> float:
    if float(weight) <= 0.0:
        return 0.0
    alias_family = manual_oto_alias_family(row.get("alias", ""))
    role = str(row.get("alias_role", "") or "").strip().lower()
    try:
        slot_count = max(1, int(row.get("slot_count", 1) or 1))
    except Exception:
        slot_count = 1
    slot_span = 1.0 / float(slot_count)
    slot_weight = 0.45 if slot_count <= 1 else (0.75 if slot_count <= 3 else 1.0)
    shift_by_anchor = {"offset": -0.20, "overlap": -0.14, "preutterance": -0.04, "fixed_end": 0.06}
    window_by_anchor = {"offset": 0.16, "overlap": 0.15, "preutterance": 0.15, "fixed_end": 0.18}
    family_weight = 1.0
    if role in {"vc", "vcv", "vv"} or alias_family == "vowel_transition":
        shift_by_anchor = {"offset": -0.27, "overlap": -0.21, "preutterance": -0.09, "fixed_end": 0.00}
        window_by_anchor = {"offset": 0.18, "overlap": 0.17, "preutterance": 0.17, "fixed_end": 0.20}
        family_weight = 1.15
    if alias_family == "leading_n":
        shift_by_anchor = {"offset": -0.32, "overlap": -0.24, "preutterance": -0.12, "fixed_end": 0.02}
        window_by_anchor = {"offset": 0.20, "overlap": 0.19, "preutterance": 0.18, "fixed_end": 0.22}
        family_weight = 1.25
    elif alias_family in {"weak_suffix", "power_suffix"}:
        window_by_anchor = {"offset": 0.20, "overlap": 0.19, "preutterance": 0.19, "fixed_end": 0.23}
        family_weight = 0.80
    shift = float(shift_by_anchor.get(anchor, 0.0)) * slot_span
    window = max(0.04, float(window_by_anchor.get(anchor, 0.18)) * slot_span)
    expected = min(0.98, max(0.02, float(slot_pos_norm) + shift))
    overflow = max(0.0, abs(float(candidate_time_norm) - expected) - window)
    if overflow <= 0.0:
        return 0.0
    scaled = overflow / max(0.04, slot_span)
    return -min(3.0, float(weight) * slot_weight * family_weight * scaled)


_MOJIBAKE_MARKERS = frozenset("繧縺繝蠑譁荳蜿逕譛髫鬟譖邱")


def _looks_mojibake(value: object) -> bool:
    text = str(value or "")
    if "\ufffd" in text:
        return True
    return sum(1 for ch in text if ch in _MOJIBAKE_MARKERS) >= 2


def _alias_primary_token_count(row: Mapping[str, object], filename_tokens: list[str]) -> int:
    terms = manual_oto_alias_order_terms(
        row.get("alias", ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias_role=str(row.get("alias_role", "") or ""),
    )
    if not terms or not filename_tokens:
        return 0
    primary = canonicalize_order_token(terms[0])
    if not primary:
        return 0
    count = 0
    for token in filename_tokens:
        current = canonicalize_order_token(token)
        if current == primary or current.startswith(primary) or primary.startswith(current) or primary in current or current in primary:
            count += 1
    return count


def _gate_one_hot(value: object, choices: tuple[str, ...]) -> list[float]:
    text = str(value or "").strip().lower()
    return [1.0 if text == choice else 0.0 for choice in choices]


def _gate_features(
    row: Mapping[str, object],
    anchor: str,
    primary: Mapping[str, object] | None,
    *,
    filename_pred: Mapping[str, object] | None,
    joint_pred: Mapping[str, object] | None,
    filename_slot_pos: float | None,
    filename_tokens: list[str],
) -> np.ndarray:
    try:
        slot_index = float(row.get("slot_index", 0.0) or 0.0)
        slot_count = max(1.0, float(row.get("slot_count", 1.0) or 1.0))
    except Exception:
        slot_index = 0.0
        slot_count = 1.0
    alias_family = manual_oto_alias_family(row.get("alias", ""))
    duration = max(1.0, float(row.get("duration_ms", 1.0) or 1.0))
    if primary is None:
        primary_pred = 0.0
        primary_slot = _centered_slot_pos(row)
        time_order = 0.0
        joint_score = -999.0
    else:
        primary_pred = float(primary.get("pred_ms", 0.0) or 0.0)
        primary_slot = float(primary.get("slot_pos_norm", _centered_slot_pos(row)) or 0.0)
        time_order = float(primary.get("time_order_norm", 0.0) or 0.0)
        joint_score = float(primary.get("joint_score", 0.0) or 0.0)
    if filename_slot_pos is None:
        slot_gap = 1.0
        filename_slot_available = 0.0
    else:
        slot_gap = abs(primary_slot - float(filename_slot_pos))
        filename_slot_available = 1.0
    if filename_pred is None:
        filename_pred_gap = 1.0
        filename_pred_available = 0.0
    else:
        filename_pred_gap = min(1.0, abs(primary_pred - float(filename_pred.get("pred_ms", 0.0) or 0.0)) / duration)
        filename_pred_available = 1.0
    if joint_pred is None:
        joint_pred_gap = 1.0
        joint_pred_available = 0.0
    else:
        joint_pred_gap = min(1.0, abs(primary_pred - float(joint_pred.get("pred_ms", 0.0) or 0.0)) / duration)
        joint_pred_available = 1.0
    values: list[float] = []
    values.extend(_gate_one_hot(anchor, SCORABLE_ANCHORS))
    values.extend(_gate_one_hot(row.get("alias_role", ""), ROLE_VALUES))
    values.extend(_gate_one_hot(row.get("format_type", ""), FORMAT_VALUES))
    values.extend(_gate_one_hot(row.get("language", ""), LANGUAGE_VALUES))
    values.extend(_gate_one_hot(alias_family, ALIAS_FAMILY_VALUES))
    values.extend(
        [
            slot_index / max(1.0, slot_count - 1.0),
            min(slot_count, 64.0) / 64.0,
            1.0 if filename_tokens else 0.0,
            filename_slot_available,
            float(slot_gap),
            filename_pred_available,
            float(filename_pred_gap),
            joint_pred_available,
            float(joint_pred_gap),
            abs(time_order - primary_slot),
            max(-10.0, min(10.0, joint_score)) / 10.0,
            min(4.0, float(_alias_primary_token_count(row, filename_tokens))) / 4.0,
            1.0
            if (
                _looks_mojibake(row.get("alias", ""))
                or _looks_mojibake(row.get("wav_name", ""))
                or _looks_mojibake(row.get("voicebank_id", ""))
            )
            else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _supervised_safe_probability(
    row: Mapping[str, object],
    anchor: str,
    prediction: Mapping[str, object] | None,
    *,
    scorer: ManualAnchorScorer,
    filename_pred: Mapping[str, object] | None,
    joint_pred: Mapping[str, object] | None,
) -> float | None:
    gate_model = scorer.failure_gate_models.get(anchor)
    if gate_model is None or prediction is None:
        return None
    tokens = _filename_tokens_for_row(row)
    filename_slot_pos = _infer_filename_slot_pos(
        row,
        tokens,
        require_primary_transition_token=scorer.require_primary_transition_token,
    )
    features = _gate_features(
        row,
        anchor,
        prediction,
        filename_pred=filename_pred,
        joint_pred=joint_pred,
        filename_slot_pos=filename_slot_pos,
        filename_tokens=tokens,
    )
    return _predict_safe_probability(gate_model, features)


def _predict_safe_probability(model: object | None, features: np.ndarray | None) -> float | None:
    if model is None or features is None:
        return None
    try:
        return float(model.predict_proba(features.reshape(1, -1))[0, 1])
    except Exception:
        try:
            return float(model.predict(features.reshape(1, -1))[0])
        except Exception:
            return None


def _passes_gate(anchor: str, probability: float | None, scorer: ManualAnchorScorer) -> bool:
    if probability is None:
        return False
    thresholds = scorer.supervised_gate_thresholds_by_anchor or {}
    threshold = float(thresholds.get(anchor, scorer.supervised_gate_threshold))
    return float(probability) >= threshold


def _nearest_track_value(tracks: ManualOtoCandidateTracks, values: object, pred_ms: float) -> float:
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if times.size == 0 or arr.size == 0:
        return 0.0
    if arr.size != times.size:
        arr = np.resize(arr, (times.size,)).astype(np.float32)
    idx = int(np.searchsorted(times, float(pred_ms), side="left"))
    if idx <= 0:
        best_idx = 0
    elif idx >= times.size:
        best_idx = int(times.size - 1)
    else:
        left = idx - 1
        best_idx = left if abs(float(times[left]) - float(pred_ms)) <= abs(float(times[idx]) - float(pred_ms)) else idx
    return float(arr[best_idx])


def _local_failure_features(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    *,
    pred_ms: float,
    predictions: Mapping[str, float],
    safe_overlap_context: Mapping[str, object],
    pre_safe_prob: float | None,
    slot_count: int,
    slot_index: int,
    all_slot_exact: bool,
    boundary_slot_exact: bool,
) -> np.ndarray:
    count = max(1, int(slot_count))
    index = max(0, min(count - 1, int(slot_index)))
    duration = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    pred = min(duration, max(0.0, float(pred_ms)))
    expected_slot_pos = 0.5 if count <= 1 else (float(index) + 0.5) / float(count)
    pred_pos = pred / duration
    delta = _slot_shift_delta(row, pred_ms=pred, duration_ms=duration, slot_count=count, slot_index=index)
    offset = predictions.get("offset")
    overlap = predictions.get("overlap")
    pre = predictions.get("preutterance")
    fixed = predictions.get("fixed_end")
    offset_f = float(offset) if offset is not None else pred
    pre_f = float(pre) if pre is not None else pred
    span = max(1.0, pre_f - offset_f)
    overlap_f = float(overlap) if overlap is not None else offset_f
    fixed_f = float(fixed) if fixed is not None else pre_f
    alias_family = manual_oto_alias_family(row.get("alias", ""))
    anchor_scores = tracks.anchor_scores.get(anchor, np.zeros((np.asarray(tracks.times_ms).size,), dtype=np.float32))
    values: list[float] = []
    values.extend(_gate_one_hot(anchor, SCORABLE_ANCHORS))
    values.extend(_gate_one_hot(row.get("alias_role", ""), ROLE_VALUES))
    values.extend(_gate_one_hot(row.get("format_type", ""), FORMAT_VALUES))
    values.extend(_gate_one_hot(row.get("language", ""), LANGUAGE_VALUES))
    values.extend(_gate_one_hot(alias_family, ALIAS_FAMILY_VALUES))
    values.extend(
        [
            index / max(1.0, count - 1.0),
            min(count, 64.0) / 64.0,
            min(duration, 10000.0) / 10000.0,
            pred_pos,
            expected_slot_pos,
            abs(pred_pos - expected_slot_pos),
            min(4.0, abs(float(delta if delta is not None else 0))) / 4.0,
            max(-1.0, min(1.0, float(delta if delta is not None else 0))),
            1.0 if pre_safe_prob is not None else 0.0,
            float(pre_safe_prob) if pre_safe_prob is not None and np.isfinite(float(pre_safe_prob)) else 0.0,
            1.0 if offset is not None else 0.0,
            1.0 if overlap is not None else 0.0,
            1.0 if pre is not None else 0.0,
            1.0 if fixed is not None else 0.0,
            max(-1.0, min(1.0, (pre_f - offset_f) / duration)),
            max(-1.0, min(1.0, (pre_f - overlap_f) / duration)),
            max(-1.0, min(1.0, (fixed_f - pre_f) / duration)),
            max(-0.5, min(1.5, (overlap_f - offset_f) / span)),
            max(-1.0, min(1.0, (pred - offset_f) / duration)),
            max(-1.0, min(1.0, (pred - pre_f) / duration)),
            max(0.0, min(1.0, float(safe_overlap_context.get("gap_ms", 0.0) or 0.0) / duration)),
            max(0.0, min(1.0, float(safe_overlap_context.get("ratio", 0.0) or 0.0))),
            1.0 if bool(safe_overlap_context.get("clamped", False)) else 0.0,
            1.0 if bool(all_slot_exact) else 0.0,
            1.0 if bool(boundary_slot_exact) else 0.0,
            _nearest_track_value(tracks, anchor_scores, pred),
            _nearest_track_value(tracks, tracks.tracks.get("transition", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("nucleus", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("rms", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("activity_edge", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("silence", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("voicing", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("flux", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("onset", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("world_stability", ()), pred),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _local_failure_safe_probability(model: object | None, features: np.ndarray | None) -> float | None:
    return _predict_safe_probability(model, features)


def _all_slot_exact(
    row: Mapping[str, object],
    predictions: Mapping[str, float],
    duration: float,
    slot_count: int,
    slot_index: int,
) -> bool:
    for pred in predictions.values():
        delta = _slot_shift_delta(row, pred_ms=float(pred), duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        if delta is None or int(delta) != 0:
            return False
    return True


def _boundary_slot_exact(
    row: Mapping[str, object],
    predictions: Mapping[str, float],
    duration: float,
    slot_count: int,
    slot_index: int,
) -> bool:
    for anchor in ("overlap", "preutterance", "fixed_end"):
        if anchor not in predictions:
            continue
        delta = _slot_shift_delta(row, pred_ms=float(predictions[anchor]), duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        if delta is None or int(delta) != 0:
            return False
    return True


def _unit_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _mark_island_count_mismatch(decode: VowelIslandDecode) -> VowelIslandDecode:
    warning = "island_count_mismatch"
    return VowelIslandDecode(
        islands=decode.islands,
        assignments=tuple(
            SlotIslandAssignment(
                slot_index=assignment.slot_index,
                island_index=assignment.island_index,
                score=assignment.score,
                margin=assignment.margin,
                warnings=tuple(dict.fromkeys((*assignment.warnings, warning))),
            )
            for assignment in decode.assignments
        ),
        score=decode.score,
        margin=decode.margin,
        warnings=tuple(dict.fromkeys((*decode.warnings, warning))),
    )


def _mark_island_count_repaired(decode: VowelIslandDecode) -> VowelIslandDecode:
    warning = "island_count_repaired"
    return VowelIslandDecode(
        islands=decode.islands,
        assignments=tuple(
            SlotIslandAssignment(
                slot_index=assignment.slot_index,
                island_index=assignment.island_index,
                score=assignment.score,
                margin=assignment.margin,
                warnings=tuple(dict.fromkeys((*assignment.warnings, warning))),
            )
            for assignment in decode.assignments
        ),
        score=decode.score,
        margin=decode.margin,
        warnings=tuple(dict.fromkeys((*decode.warnings, warning))),
    )


def _island_json(island: VowelIsland | None) -> dict[str, object] | None:
    if island is None:
        return None
    return {
        "start_ms": float(island.start_ms),
        "nucleus_ms": float(island.nucleus_ms),
        "end_ms": float(island.end_ms),
        "confidence": float(island.confidence),
        "left_valley_ms": float(island.left_valley_ms),
        "right_valley_ms": float(island.right_valley_ms),
        "start_index": int(island.start_index),
        "nucleus_index": int(island.nucleus_index),
        "end_index": int(island.end_index),
    }


def _assignment_json(assignment: SlotIslandAssignment | None) -> dict[str, object] | None:
    if assignment is None:
        return None
    return {
        "slot_index": int(assignment.slot_index),
        "island_index": None if assignment.island_index is None else int(assignment.island_index),
        "score": float(assignment.score),
        "margin": float(assignment.margin),
        "warnings": list(assignment.warnings),
    }


def _serialize_island_context(context: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(context, Mapping):
        return {"index": None, "previous": None, "current": None, "next": None}
    return {
        "index": context.get("index"),
        "previous": _island_json(context.get("previous") if isinstance(context.get("previous"), VowelIsland) else None),
        "current": _island_json(context.get("current") if isinstance(context.get("current"), VowelIsland) else None),
        "next": _island_json(context.get("next") if isinstance(context.get("next"), VowelIsland) else None),
    }


def _island_decode_json(decode: VowelIslandDecode) -> dict[str, object]:
    return {
        "islands": [_island_json(item) for item in decode.islands],
        "assignments": [_assignment_json(item) for item in decode.assignments],
        "score": float(decode.score),
        "margin": float(decode.margin),
        "warnings": list(decode.warnings),
    }


def _decode_from_json(data: Mapping[str, object]) -> VowelIslandDecode:
    islands: list[VowelIsland] = []
    for item in (data.get("islands", []) if isinstance(data, Mapping) else []):
        if not isinstance(item, Mapping):
            continue
        islands.append(
            VowelIsland(
                start_ms=float(item.get("start_ms", 0.0) or 0.0),
                nucleus_ms=float(item.get("nucleus_ms", 0.0) or 0.0),
                end_ms=float(item.get("end_ms", 0.0) or 0.0),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                left_valley_ms=float(item.get("left_valley_ms", 0.0) or 0.0),
                right_valley_ms=float(item.get("right_valley_ms", 0.0) or 0.0),
                start_index=int(item.get("start_index", 0) or 0),
                nucleus_index=int(item.get("nucleus_index", 0) or 0),
                end_index=int(item.get("end_index", 0) or 0),
            )
        )
    assignments: list[SlotIslandAssignment] = []
    for item in (data.get("assignments", []) if isinstance(data, Mapping) else []):
        if not isinstance(item, Mapping):
            continue
        island_index = item.get("island_index")
        assignments.append(
            SlotIslandAssignment(
                slot_index=int(item.get("slot_index", 0) or 0),
                island_index=None if island_index is None else int(island_index),
                score=float(item.get("score", 0.0) or 0.0),
                margin=float(item.get("margin", 0.0) or 0.0),
                warnings=tuple(str(value) for value in item.get("warnings", []) or []),
            )
        )
    return VowelIslandDecode(
        islands=tuple(islands),
        assignments=tuple(assignments),
        score=float(data.get("score", 0.0) or 0.0) if isinstance(data, Mapping) else 0.0,
        margin=float(data.get("margin", 0.0) or 0.0) if isinstance(data, Mapping) else 0.0,
        warnings=tuple(str(value) for value in (data.get("warnings", []) if isinstance(data, Mapping) else []) or []),
    )


__all__ = [
    "DEFAULT_MANUAL_ANCHOR_SCORER_ENV",
    "ManualAnchorPreviewConfig",
    "ManualAnchorScorer",
    "generate_manual_oto_anchor_preview",
    "load_manual_oto_anchor_scorer",
    "resolve_latest_manual_oto_anchor_scorer",
]
