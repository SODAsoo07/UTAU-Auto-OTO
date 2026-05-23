from __future__ import annotations

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
    absolute_oto_positions,
    format_oto_line,
    load_oto_template_rows,
    timing_to_dict,
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
            "scorer_payload_warnings": list(scorer.payload_warnings),
        }
        records.append(record)
        if overlay_root:
            _write_manual_anchor_overlay(overlay_root, wav_path, runtime_rows, tracks, wav_result)
        _emit_progress(progress_callback, f"progress {wav_index}/{wav_total} wav 완료: {wav_name}")

    out_oto = Path(config.out_oto_path)
    out_oto.parent.mkdir(parents=True, exist_ok=True)
    out_oto.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
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
        "rows": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "wav_count": len(wav_paths),
        "oto_rows": len(out_lines),
        "out_oto": str(out_oto),
        "out_json": str(out_json),
        "overlay_dir": str(overlay_root) if overlay_root else "",
        "scorer": scorer.path,
        "scorer_payload_warnings": list(scorer.payload_warnings),
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
    rows: list[dict[str, object]] = []
    for idx, source in enumerate(source_rows):
        role = classify_manual_oto_alias_role(
            source.alias,
            language=language,
            format_type=format_type,
        )
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
            }
        )
    return rows


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
    if len(raw_islands) != int(slot_count):
        island_decode = _mark_island_count_mismatch(island_decode)

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
    predictions: dict[str, float] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []

    if island is None:
        warnings.append("missing_island")
    elif not assignment_is_safe(assignment, island):
        warnings.extend(str(item) for item in getattr(assignment, "warnings", ()) or ())

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
    if pre_prediction is not None:
        pred = float(pre_prediction.get("pred_ms", 0.0) or 0.0)
        delta = _slot_shift_delta(row, pred_ms=pred, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        if delta == 0 or _passes_gate("preutterance", pre_safe_prob, scorer):
            pre_ms = pred
            sources["preutterance"] = (
                "constrained_gate"
                if _passes_gate("preutterance", pre_safe_prob, scorer)
                else "constrained_slot_exact"
            )
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
    if fixed_prediction is not None and _passes_gate("fixed_end", fixed_safe_prob, scorer):
        candidate_fixed = float(fixed_prediction.get("pred_ms", 0.0) or 0.0)
        delta = _slot_shift_delta(row, pred_ms=candidate_fixed, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
        if delta == 0 and candidate_fixed >= predictions["preutterance"]:
            fixed_ms = candidate_fixed
            sources["fixed_end"] = "constrained_gate"
    if fixed_ms is None:
        fixed_ms = predictions["preutterance"] + _default_fixed_gap_ms(row)
        warnings.append("fallback_fixed_end_gap")
    predictions["fixed_end"] = min(duration, max(predictions["preutterance"] + 8.0, float(fixed_ms)))
    sources.setdefault("fixed_end", "relative_from_preutterance")

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
    absolute = absolute_oto_positions(timing)
    generated = {
        "wav": row.get("wav_name"),
        "alias": row.get("alias"),
        "line": format_oto_line(str(row.get("wav_name") or ""), str(row.get("alias") or ""), timing),
        "timing": timing_to_dict(timing),
        "absolute": absolute,
        "sources": dict(sources),
        "warnings": list(dict.fromkeys(warnings)),
        "overlap_policy": overlap_context,
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
        "island": _island_json(island),
        "assignment": _assignment_json(assignment),
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
    if island is not None:
        cutoff_abs = max(fixed_abs + 1.0, min(duration, max(float(island.end_ms) + 70.0, fixed_abs + 35.0)))
    else:
        cutoff_abs = max(fixed_abs + 1.0, min(duration, fixed_abs + _default_cutoff_gap_ms(row)))
    cutoff_abs = min(duration, max(fixed_abs + 1.0, cutoff_abs))
    timing = OtoTiming(
        offset=float(offset),
        consonant=float(max(1.0, fixed_abs - offset)),
        cutoff=float(-max(1.0, cutoff_abs - offset)),
        preutterance=float(max(0.0, pre_abs - offset)),
        overlap=float(max(0.0, overlap_abs - offset)),
    )
    return timing, cutoff_abs


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
        metadata={"source": "manual_oto_anchor_runtime", "encoder": tracks.encoder},
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
        "auxiliary_events": island_overlay_events(_decode_from_json(wav_result.get("island_decode") or {})),
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
    raw = row.get("filename_canonical_tokens") or []
    if isinstance(raw, list) and raw:
        return [canonicalize_order_token(item) for item in raw if canonicalize_order_token(item)]
    wav_name = str(row.get("wav_name", "") or Path(str(row.get("wav_path", "") or "")).name)
    language = str(row.get("language", "") or "")
    context = parse_filename_context(wav_name, language=language, format_type=str(row.get("format_type", "") or ""))
    return [canonicalize_order_token(item) for item in context.canonical_tokens if canonicalize_order_token(item)]


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
                score += 4
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 4)
            elif token.startswith(term) or term.startswith(token):
                score += 2
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 2)
            elif term in token or token in term:
                score += 1
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


def _filename_ordered_rows(
    wav_rows: list[dict[str, object]],
    *,
    require_primary_transition_token: bool = False,
) -> list[tuple[dict[str, object], float]]:
    if not wav_rows:
        return []
    tokens = _filename_tokens_for_row(wav_rows[0])
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
    tokens = _filename_tokens_for_row(wav_rows[0])
    format_type = str(wav_rows[0].get("format_type", "") or "").strip().lower()
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
    tokens = _filename_tokens_for_row(row)
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
        return min(duration_ms, pre_ms)
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


def _dependent_offset_ms(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    relative_anchor_priors: Mapping[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
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
