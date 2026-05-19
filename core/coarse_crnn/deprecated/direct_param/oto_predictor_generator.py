from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from core.coarse_crnn.alias_role import classify_alias_role, is_diphthong as _is_diphthong, normalize_role
from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.oto_audio_candidates import (
    AudioCandidates,
    compute_audio_candidates,
    find_best_vowel_segment,
    find_nearest_onset_after,
)
from core.coarse_crnn.deprecated.direct_param.oto_boundary_decoding import decode_boundary_slot_graph_for_states
from core.coarse_crnn.deprecated.direct_param.oto_inference import predict_oto_with_model
from core.coarse_crnn.deprecated.direct_param.oto_model import load_oto_checkpoint
from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import decode_sequence_alignment_for_states, parse_sequence_role_allowlist
from core.coarse_crnn.oto_targets import OtoAnchors, anchors_to_oto_params, oto_params_to_anchors, parse_alias_components
from core.coarse_crnn.slot_graph import slot_role_order_norm, slot_type_for_role
from core.coarse_crnn.torch_utils import resolve_torch_device
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_features import classify_alias_type, detect_format_type
from core.oto_normalization import normalize_wav_key


DEFAULT_OTO_CRNN_MODEL_NAME = "oto_anchor_crnn_role_v2.pt"
_EXPERIMENTAL_OTO_CRNN_MODEL_MARKERS = ("boundary_slot",)
_ORDER_TOKEN_RE = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3\u3041-\u3096\u30A1-\u30FA\u30FC]+")
_HANGUL_CHAR_RE = re.compile(r"[\uAC00-\uD7A3]")
_KANA_RE = re.compile(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]")
_TOKEN_SEP_RE = re.compile(r"[\s,_\-/|']+")
_CODA_SUFFIXES = ("ng", "n", "m", "l", "h")
_VOWEL_ID_NORM = {
    "a": 1.0 / 6.0,
    "i": 2.0 / 6.0,
    "u": 3.0 / 6.0,
    "eu": 3.0 / 6.0,
    "e": 4.0 / 6.0,
    "eo": 4.0 / 6.0,
    "o": 5.0 / 6.0,
}


@dataclass(frozen=True)
class _PredictRow:
    wav_rel: str
    wav_abs: str
    alias: str
    prev_alias: str
    next_alias: str
    row_index: int
    row_count: int
    is_special: bool = False
    prev_is_special: bool = False
    next_is_special: bool = False
    base_params: dict[str, float] | None = None
    base_fallback_allowed: bool = True
    base_fallback_order_ratio: float = 1.0
    uses_filename_slots: bool = False
    alias_role: str = ""


def generate_oto_with_crnn_predictor(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    language: str,
    format_type: str = "",
    model_path: str = "",
    device: str = "auto",
    alias_suffix: str = "",
    callback: Callable[[str], None] | None = None,
    special_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_no_mfa_source_oto(wav_dir=wav_dir, source_hint=source_oto_path)
    if not source:
        return 0, 0, ["CRNN OTO ・溢ｸ｡・ｩ ・・ｴ・､ OTO・ｼ ・ｾ・ ・ｻ嵂溢慣・壱共."]
    model_file = resolve_oto_crnn_model_path(model_path)
    if not model_file:
        return 0, 0, ["CRNN OTO ・溢ｸ｡ ・ｨ・ｸ・・・ｾ・ ・ｻ嵂溢慣・壱共. ・ｨ・ｸ ・ｽ・罹･ｼ ・・倣紛 ・ｼ・ｸ・・"]
    output_file = _normalize_output_oto_path(out_path)
    if not output_file:
        return 0, 0, ["・罹･ OTO ・ｽ・懋ｰ ・・牟 ・溢慣・壱共."]

    lang = str(language or "").strip().lower() or "korean"
    rows, total, missing = _prepare_prediction_rows(
        wav_dir=wav_dir,
        source_oto_path=source,
        language=lang,
        special_aliases=_normalize_special_aliases(special_aliases),
    )
    if not rows:
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            return 0, total, [f"CRNN OTO ・溢ｸ｡ ・・・WAV ・､・ｭ ・､甯ｨ: {sample}"]
        return 0, total, ["CRNN OTO ・溢ｸ｡ ・・・嵂餓擽 ・・牟 ・溢慣・壱共."]

    _log(callback, f"[OTO-CRNN] source={source}")
    _log(callback, f"[OTO-CRNN] model={model_file}")
    _log(callback, f"[OTO-CRNN] device={device} rows={len(rows)}/{total}")
    fmt = _resolve_effective_format(
        language=language,
        requested_format=format_type,
        aliases=[row.alias for row in rows],
        callback=callback,
    )

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    try:
        model, config, _meta = load_oto_checkpoint(model_file, map_location=str(torch_device))
        if _is_experimental_oto_crnn_model(model_file, config) and not _env_bool(
            "UTOA_OTO_CRNN_ALLOW_EXPERIMENTAL_MODEL",
            False,
        ):
            name = os.path.basename(str(model_file))
            return (
                0,
                total,
                [
                    "CRNN OTO experimental model is blocked by default: "
                    f"{name}. Set UTOA_OTO_CRNN_ALLOW_EXPERIMENTAL_MODEL=1 only for controlled tests."
                ],
            )
        model = model.to(torch_device).eval()
    except Exception as exc:
        return 0, total, [f"CRNN OTO ・ｨ・ｸ ・罹糖 ・､甯ｨ: {exc}"]

    out_lines: list[str] = []
    row_states: list[dict[str, Any]] = []
    errors: list[str] = []
    guard_changed = 0
    low_conf_fallback_count = 0
    activity_fallback_count = 0
    candidate_snap_count = 0
    vc_matcher_count = 0
    final_right_guard_count = 0
    boundary_slot_graph_count = 0
    boundary_slot_graph_hold_count = 0
    alias_slot_lock_count = 0
    alias_slot_lock_hold_count = 0
    sequence_aligned_count = 0
    sequence_align_hold_count = 0
    row_order_aligned_count = 0
    row_order_hold_count = 0
    base_fallback_blocked_rows = 0
    base_fallback_blocked_wavs: set[str] = set()
    activity_profile_cache: dict[str, dict[str, float]] = {}
    audio_candidate_cache: dict[str, AudioCandidates | None] = {}
    audio_candidate_sequence_state: dict[str, int] = {}
    use_base_oto_fallback = _env_bool("UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK", False)
    row_order_align_enable = _env_bool("UTOA_OTO_CRNN_ROW_ORDER_ALIGN_ENABLE", True)
    suffix = _normalize_alias_suffix(alias_suffix)
    for idx, row in enumerate(rows):
        try:
            pred = predict_oto_with_model(
                model=model,
                config=config,
                wav_path=row.wav_abs,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                prev_alias=row.prev_alias,
                next_alias=row.next_alias,
                row_index_in_wav=row.row_index,
                file_row_count=row.row_count,
                device=str(torch_device),
                is_special=row.is_special,
                prev_is_special=row.prev_is_special,
                next_is_special=row.next_is_special,
            )
            alias = _apply_alias_suffix(row.alias, suffix)
            params, changed = _apply_conservative_right_boundary_guard(
                pred.params,
                language=lang,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            base_params = None
            if use_base_oto_fallback and row.base_params:
                if row.base_fallback_allowed:
                    base_params = dict(row.base_params)
                else:
                    base_fallback_blocked_rows += 1
                    base_fallback_blocked_wavs.add(row.wav_rel.lower())
            params, fallback_reason = _apply_low_confidence_fallback(
                predicted_params=params,
                predicted_confidence=float(getattr(pred, "confidence", 0.0) or 0.0),
                predicted_error_ms=getattr(pred, "predicted_error_ms", None),
                predicted_low_confidence=bool(getattr(pred, "low_confidence", False)),
                confidence_components=getattr(pred, "confidence_components", None),
                base_params=base_params,
                language=lang,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            if fallback_reason:
                low_conf_fallback_count += 1
            params, activity_reason = _apply_activity_window_fallback(
                predicted_anchors={
                    "offset": float(getattr(pred.anchors, "offset", 0.0)),
                    "overlap": float(getattr(pred.anchors, "overlap", 0.0)),
                    "preutterance": float(getattr(pred.anchors, "preutterance", 0.0)),
                    "consonant": float(getattr(pred.anchors, "consonant", 0.0)),
                    "cutoff": float(getattr(pred.anchors, "cutoff", 0.0)),
                },
                predicted_params=params,
                base_params=base_params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                row_index=row.row_index,
                row_count=row.row_count,
                language=lang,
                alias=row.alias,
                is_special=row.is_special,
            )
            if activity_reason:
                activity_fallback_count += 1
            guard_changed += 1 if changed else 0
            row_states.append(
                {
                    "row": row,
                    "alias": alias,
                    "params": dict(params),
                    "duration_ms": float(getattr(pred, "duration_ms", 0.0) or 0.0),
                    "predicted_confidence": float(getattr(pred, "confidence", 0.0) or 0.0),
                    "predicted_error_ms": getattr(pred, "predicted_error_ms", None),
                    "predicted_low_confidence": bool(getattr(pred, "low_confidence", False)),
                    "confidence_components": dict(getattr(pred, "confidence_components", None) or {}),
                    "sequence_candidate_times_ms": list(getattr(pred, "sequence_candidate_times_ms", None) or []),
                    "sequence_candidate_scores": list(getattr(pred, "sequence_candidate_scores", None) or []),
                    "sequence_candidate_classes": list(getattr(pred, "sequence_candidate_classes", None) or []),
                    "aligned_by_row_order": False,
                    "row_order_hold": False,
                    "boundary_slot_graph": False,
                    "boundary_slot_graph_reason": "",
                    "alias_slot_locked": False,
                    "alias_slot_lock_hold": False,
                    "alias_slot_lock_reason": "",
                    "sequence_aligned": False,
                    "sequence_alignment_hold": False,
                    "sequence_alignment_reason": "",
                }
            )
        except Exception as exc:
            errors.append(f"{row.wav_rel}={row.alias}: {exc}")
        if callback and (idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(rows)):
            _log(callback, f"[OTO-CRNN] rows={idx + 1}/{len(rows)}")

    if errors:
        return len(row_states), total, errors[:20]
    if not row_states:
        return 0, total, ["CRNN OTO ・溢ｸ｡ ・ｰ・ｼ・ ・・来・ｵ・壱共."]

    sequence_align_enable = _env_bool("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", True)
    if row_order_align_enable and not sequence_align_enable:
        aligned, held = _apply_monotonic_row_order_alignment(
            row_states=row_states,
            cache=audio_candidate_cache,
            config=config,
            language=lang,
            callback=callback,
        )
        row_order_aligned_count += int(aligned)
        row_order_hold_count += int(held)

    if _env_bool("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_ENABLE", True):
        moved, held = _apply_alias_slot_lock_alignment(
            row_states=row_states,
            cache=audio_candidate_cache,
            config=config,
            max_shift_ms=max(0.0, _env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MAX_SHIFT_MS", 3000.0)),
            min_move_ms=max(0.0, _env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MIN_MOVE_MS", 30.0)),
            mode=str(os.environ.get("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MODE", "direct") or "direct"),
        )
        alias_slot_lock_count += int(moved)
        alias_slot_lock_hold_count += int(held)

    if sequence_align_enable:
        aligned, held = decode_sequence_alignment_for_states(
            row_states=row_states,
            language=lang,
            role_resolver=lambda row: _safe_alias_role(
                lang,
                getattr(row, "alias", ""),
                alias_type=_safe_alias_type(lang, getattr(row, "alias", "")),
                is_special=bool(getattr(row, "is_special", False)),
            ),
            candidate_getter=lambda wav_abs: _get_audio_candidates(str(wav_abs), cache=audio_candidate_cache, config=config),
            max_shift_ms=max(0.0, _env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MAX_SHIFT_MS", 220.0)),
            anchor_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ANCHOR_BLEND", 0.60), 0.0, 1.0),
            boundary_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_BOUNDARY_BLEND", 0.45), 0.0, 1.0),
            min_move_ms=max(0.0, _env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MIN_MOVE_MS", 8.0)),
            model_score_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MODEL_SCORE_BLEND", 0.35), 0.0, 1.0),
            allowed_roles=parse_sequence_role_allowlist(os.environ.get("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ROLES", "vc,vv")),
        )
        sequence_aligned_count += int(aligned)
        sequence_align_hold_count += int(held)

    if _env_bool("UTOA_OTO_CRNN_BOUNDARY_SLOT_GRAPH_ENABLE", False):
        moved, held = decode_boundary_slot_graph_for_states(
            row_states=row_states,
            language=lang,
            role_resolver=lambda row: _safe_alias_role(
                lang,
                getattr(row, "alias", ""),
                alias_type=_safe_alias_type(lang, getattr(row, "alias", "")),
                is_special=bool(getattr(row, "is_special", False)),
            ),
            candidate_getter=lambda wav_abs: _get_audio_candidates(str(wav_abs), cache=audio_candidate_cache, config=config),
            max_delta_ms=max(0.0, _env_float("UTOA_OTO_CRNN_BOUNDARY_SLOT_GRAPH_MAX_DELTA_MS", 260.0)),
            blend=_clamp(_env_float("UTOA_OTO_CRNN_BOUNDARY_SLOT_GRAPH_BLEND", 0.72), 0.0, 1.0),
        )
        boundary_slot_graph_count += int(moved)
        boundary_slot_graph_hold_count += int(held)

    allow_legacy_candidate_postprocess = (not sequence_align_enable) or _env_bool(
        "UTOA_OTO_CRNN_SEQUENCE_ALIGN_ALLOW_LEGACY_CANDIDATE_POSTPROCESS",
        False,
    )
    for state in row_states:
        row = state["row"]
        params = dict(state["params"])
        if allow_legacy_candidate_postprocess:
            params, vc_reason = _apply_vc_candidate_matcher(
                predicted_params=params,
                wav_path=row.wav_abs,
                language=lang,
                alias=row.alias,
                format_type=fmt,
                duration_ms=float(state.get("duration_ms", 0.0) or 0.0),
                cache=audio_candidate_cache,
                config=config,
                is_special=row.is_special,
                predicted_confidence=float(state.get("predicted_confidence", 0.0) or 0.0),
                predicted_error_ms=state.get("predicted_error_ms", None),
                predicted_low_confidence=bool(state.get("predicted_low_confidence", False)),
                confidence_components=state.get("confidence_components", None),
            )
            if vc_reason:
                vc_matcher_count += 1
                params, _ = _apply_conservative_right_boundary_guard(
                    params,
                    language=lang,
                    alias=row.alias,
                    duration_ms=float(state.get("duration_ms", 0.0) or 0.0),
                    is_special=row.is_special,
                )
            params, snap_reason = _apply_audio_candidate_snap(
                predicted_params=params,
                wav_path=row.wav_abs,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                duration_ms=float(state.get("duration_ms", 0.0) or 0.0),
                cache=audio_candidate_cache,
                sequence_state=audio_candidate_sequence_state,
                config=config,
                is_special=row.is_special,
                predicted_confidence=float(state.get("predicted_confidence", 0.0) or 0.0),
                predicted_error_ms=state.get("predicted_error_ms", None),
                predicted_low_confidence=bool(state.get("predicted_low_confidence", False)),
                confidence_components=state.get("confidence_components", None),
            )
            if snap_reason:
                candidate_snap_count += 1
                params, _snap_guard_changed = _apply_conservative_right_boundary_guard(
                    params,
                    language=lang,
                    alias=row.alias,
                    duration_ms=float(state.get("duration_ms", 0.0) or 0.0),
                    is_special=row.is_special,
                )
        state["params"] = params
        if _env_bool("UTOA_OTO_CRNN_FINAL_RIGHT_GUARD_ENABLE", True):
            guarded, guard_changed = _apply_conservative_right_boundary_guard(
                dict(state["params"]),
                language=lang,
                alias=row.alias,
                duration_ms=float(state.get("duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            if guard_changed:
                final_right_guard_count += 1
                state["params"] = guarded
                params = guarded
        out_lines.append(
            f"{row.wav_rel}={state['alias']},"
            f"{float(params['offset']):.3f},"
            f"{float(params['consonant']):.3f},"
            f"{float(params['cutoff']):.3f},"
            f"{float(params['preutterance']):.3f},"
            f"{float(params['overlap']):.3f}"
        )

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines).rstrip() + "\n")
    except OSError as exc:
        return 0, total, [f"CRNN OTO ・罹･ ・・･ ・､甯ｨ: {output_file} ({exc})"]

    if missing:
        sample = ", ".join(sorted(missing)[:5])
        suffix_text = "..." if len(missing) > 5 else ""
        _log(callback, f"[OTO-CRNN] ・､・ｭ ・､甯ｨ wav: {sample}{suffix_text}")
    if guard_changed:
        _log(callback, f"[OTO-CRNN] right-boundary guard adjusted rows={guard_changed}/{len(out_lines)}")
    if low_conf_fallback_count:
        _log(callback, f"[OTO-CRNN] low-confidence fallback rows={low_conf_fallback_count}/{len(out_lines)}")
    if activity_fallback_count:
        _log(callback, f"[OTO-CRNN] activity-window fallback rows={activity_fallback_count}/{len(out_lines)}")
    if candidate_snap_count:
        _log(callback, f"[OTO-CRNN] audio-candidate snap rows={candidate_snap_count}/{len(out_lines)}")
    if vc_matcher_count:
        _log(callback, f"[OTO-CRNN] vc-candidate matcher rows={vc_matcher_count}/{len(out_lines)}")
    if final_right_guard_count:
        _log(callback, f"[OTO-CRNN] final right-boundary guarded rows={final_right_guard_count}/{len(out_lines)}")
    if boundary_slot_graph_count:
        _log(callback, f"[OTO-CRNN] boundary slot graph rows={boundary_slot_graph_count}/{len(out_lines)}")
    if boundary_slot_graph_hold_count:
        _log(callback, f"[OTO-CRNN] boundary slot graph held rows={boundary_slot_graph_hold_count}/{len(out_lines)}")
    if alias_slot_lock_count:
        _log(callback, f"[OTO-CRNN] alias slot locked rows={alias_slot_lock_count}/{len(out_lines)}")
    if alias_slot_lock_hold_count:
        _log(callback, f"[OTO-CRNN] alias slot lock held rows={alias_slot_lock_hold_count}/{len(out_lines)}")
    if sequence_aligned_count:
        _log(callback, f"[OTO-CRNN] sequence aligned rows={sequence_aligned_count}/{len(out_lines)}")
    if sequence_align_hold_count:
        _log(callback, f"[OTO-CRNN] sequence alignment held rows={sequence_align_hold_count}/{len(out_lines)}")
    if row_order_aligned_count:
        _log(callback, f"[OTO-CRNN] row-order aligned rows={row_order_aligned_count}/{len(out_lines)}")
    if row_order_hold_count:
        _log(callback, f"[OTO-CRNN] row-order hold rows={row_order_hold_count}/{len(out_lines)}")
    if use_base_oto_fallback and base_fallback_blocked_rows > 0:
        _log(
            callback,
            "[OTO-CRNN] base-fallback quality gate blocked "
            f"rows={base_fallback_blocked_rows}/{len(out_lines)} wavs={len(base_fallback_blocked_wavs)}",
        )
    _log(callback, f"[OTO-CRNN] written={len(out_lines)} total={total}")
    return len(out_lines), total, []


def resolve_oto_crnn_model_path(path_hint: str = "") -> str:
    candidates = []
    for raw in (path_hint, os.environ.get("UTOA_OTO_CRNN_MODEL_PATH", "")):
        raw_text = str(raw or "").strip()
        if raw_text:
            candidates.append(raw_text)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.extend(
        [
            os.path.join(root, "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "assets", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "ml_workspace", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
        ]
    )
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
    return ""


def _is_experimental_oto_crnn_model(model_path: str, config: object) -> bool:
    # NOTE: enable_boundary_slot_head used to gate here, but the boundary slot
    # head became the train_oto default 窶・every standard model now ships with
    # it, so config-based gating blocked all of them. Experimental models are
    # now flagged explicitly via a filename marker instead.
    name = os.path.basename(str(model_path or "")).lower()
    return any(marker in name for marker in _EXPERIMENTAL_OTO_CRNN_MODEL_MARKERS)


def _prepare_prediction_rows(
    *,
    wav_dir: str,
    source_oto_path: str,
    language: str = "",
    special_aliases: frozenset[str] | None = None,
) -> tuple[list[_PredictRow], int, set[str]]:
    wav_root = os.path.abspath(str(wav_dir or "").strip())
    exact, normalized = _build_wav_lookup(wav_root)
    raw_rows = []
    total = 0
    missing: set[str] = set()
    for raw in read_text_with_fallback(source_oto_path).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        total += 1
        wav_name = os.path.basename(str(parsed.get("wav", "") or "").replace("\\", "/").strip())
        mapped = exact.get(wav_name.lower(), "")
        if not mapped:
            mapped = normalized.get(normalize_wav_key(wav_name), "")
        if not mapped:
            missing.add(wav_name)
            continue
        raw_rows.append(
            {
                "wav_rel": mapped,
                "wav_abs": os.path.join(wav_root, mapped.replace("/", os.sep)),
                "alias": str(parsed.get("alias", "") or "").strip(),
                "base_params": _normalize_base_params(parsed),
            }
        )

    special_set: frozenset[str] = special_aliases or frozenset()

    per_wav: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        per_wav.setdefault(row["wav_rel"], []).append(row)

    quality_gate_enable = _env_bool("UTOA_OTO_CRNN_BASE_FALLBACK_QUALITY_GATE_ENABLE", True)
    quality_gate_min_ratio = _env_float("UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO", 0.42)
    quality_gate_min_ratio_cvvc = _env_float("UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO_CVVC", 0.34)
    quality_gate_min_expected_tokens = max(2, int(_env_float("UTOA_OTO_CRNN_BASE_FALLBACK_MIN_EXPECTED_TOKENS", 4.0)))
    quality_gate_min_expected_tokens_cvvc = max(
        2,
        int(_env_float("UTOA_OTO_CRNN_BASE_FALLBACK_MIN_EXPECTED_TOKENS_CVVC", float(quality_gate_min_expected_tokens))),
    )
    per_wav_fallback_allowed: dict[str, bool] = {}
    per_wav_fallback_ratio: dict[str, float] = {}
    for wav_rel, items in per_wav.items():
        aliases = [str(item.get("alias", "") or "") for item in items]
        ratio, expected_count, is_cvvc_like = _estimate_filename_alias_order_ratio(wav_rel, aliases, language=language)
        allow = True
        min_expected = quality_gate_min_expected_tokens_cvvc if is_cvvc_like else quality_gate_min_expected_tokens
        min_ratio = quality_gate_min_ratio_cvvc if is_cvvc_like else quality_gate_min_ratio
        if quality_gate_enable and expected_count >= min_expected:
            allow = ratio >= float(min_ratio)
        per_wav_fallback_allowed[wav_rel] = bool(allow)
        per_wav_fallback_ratio[wav_rel] = float(ratio)

    out: list[_PredictRow] = []
    for wav_rel, items in per_wav.items():
        count = len(items)
        sequenced_items = _assign_filename_slot_indices(
            wav_rel=wav_rel,
            items=items,
            language=language,
        )
        sequence_neighbors = _sequence_neighbors(sequenced_items)
        fallback_allowed = bool(per_wav_fallback_allowed.get(wav_rel, True))
        fallback_ratio = float(per_wav_fallback_ratio.get(wav_rel, 1.0))
        for idx, row in enumerate(sequenced_items):
            prev_alias, next_alias = sequence_neighbors.get(id(row), ("", ""))
            alias_text = str(row["alias"])
            out.append(
                _PredictRow(
                    wav_rel=wav_rel,
                    wav_abs=str(row["wav_abs"]),
                    alias=alias_text,
                    prev_alias=prev_alias,
                    next_alias=next_alias,
                    row_index=int(row.get("model_row_index", idx) or 0),
                    row_count=max(1, int(row.get("model_row_count", count) or count)),
                    is_special=_alias_is_special(alias_text, special_set),
                    prev_is_special=_alias_is_special(prev_alias, special_set),
                    next_is_special=_alias_is_special(next_alias, special_set),
                    base_params=dict(row.get("base_params") or {}) or None,
                    base_fallback_allowed=fallback_allowed,
                    base_fallback_order_ratio=fallback_ratio,
                    uses_filename_slots=bool(row.get("uses_filename_slots", False)),
                    alias_role=str(row.get("alias_role", "") or ""),
                )
            )
    return out, total, missing


def _normalize_special_aliases(
    special_aliases: set[str] | list[str] | tuple[str, ...] | None,
) -> frozenset[str]:
    if not special_aliases:
        return frozenset()
    cleaned: set[str] = set()
    for item in special_aliases:
        text = str(item or "").strip().lower()
        if text:
            cleaned.add(text)
    return frozenset(cleaned)


def _apply_alias_slot_lock_alignment(
    *,
    row_states: list[dict[str, Any]],
    cache: dict[str, AudioCandidates | None],
    config: object,
    max_shift_ms: float,
    min_move_ms: float,
    mode: str = "direct",
) -> tuple[int, int]:
    moved = 0
    held = 0
    groups: dict[str, list[dict[str, Any]]] = {}
    for state in row_states:
        row = state.get("row")
        if row is None or not bool(getattr(row, "uses_filename_slots", False)):
            continue
        groups.setdefault(str(getattr(row, "wav_abs", "")), []).append(state)

    for wav_abs, states in groups.items():
        if not states:
            continue
        slot_count = max(1, max(int(getattr(state.get("row"), "row_count", 1) or 1) for state in states))
        if slot_count < 2:
            continue
        audio = _get_audio_candidates(str(wav_abs), cache=cache, config=config)
        if audio is None:
            held += len(states)
            continue
        slot_times = _pick_alias_slot_times(audio, slot_count=slot_count)
        if len(slot_times) < slot_count:
            held += len(states)
            continue
        resolved: list[tuple[dict[str, Any], int, str]] = []
        for state in states:
            row = state.get("row")
            target_idx = max(0, min(slot_count - 1, int(getattr(row, "row_index", 0) or 0)))
            role = _role_for_predict_row(row)
            resolved.append((state, target_idx, role))

        # First lock stable syllable anchors, then use that resulting timeline
        # to constrain VC placement between neighboring anchors.
        for state, target_idx, role in resolved:
            if role == "vc":
                continue
            target_time_ms = _alias_slot_lock_target_time(
                audio,
                role=role,
                target_slot_idx=target_idx,
                slot_times=slot_times,
            )
            changed = _translate_state_to_alias_slot(
                state,
                target_slot_idx=target_idx,
                slot_times=slot_times,
                target_time_ms=target_time_ms,
                max_shift_ms=max_shift_ms,
                min_move_ms=min_move_ms,
                mode=mode,
            )
            if changed:
                moved += 1
            elif bool(state.get("alias_slot_lock_hold", False)):
                held += 1

        anchor_times = _build_alias_slot_anchor_timeline(resolved, slot_times=slot_times)

        for state, target_idx, role in resolved:
            if role != "vc":
                continue
            target_time_ms = _alias_slot_lock_target_time(
                audio,
                role=role,
                target_slot_idx=target_idx,
                slot_times=slot_times,
                anchor_times=anchor_times,
            )
            changed = _translate_state_to_alias_slot(
                state,
                target_slot_idx=target_idx,
                slot_times=slot_times,
                target_time_ms=target_time_ms,
                max_shift_ms=max_shift_ms,
                min_move_ms=min_move_ms,
                mode=mode,
            )
            if changed:
                moved += 1
            elif bool(state.get("alias_slot_lock_hold", False)):
                held += 1
    return moved, held


def _role_for_predict_row(row: object) -> str:
    role = normalize_role(str(getattr(row, "alias_role", "") or ""))
    if role:
        return role
    language = str(getattr(row, "language", "") or "")
    alias = str(getattr(row, "alias", "") or "")
    return normalize_role(
        _safe_alias_role(
            language,
            alias,
            alias_type=_safe_alias_type(language, alias),
            is_special=bool(getattr(row, "is_special", False)),
        )
    )


def _build_alias_slot_anchor_timeline(
    resolved_states: list[tuple[dict[str, Any], int, str]],
    *,
    slot_times: list[float],
) -> list[float | None]:
    slot_count = len(slot_times)
    by_slot: list[list[float]] = [[] for _ in range(slot_count)]
    anchor_roles = _env_csv_set("UTOA_OTO_CRNN_VC_ANCHOR_ROLES", {"-cv", "cv", "v", "special"})
    min_conf = _clamp(_env_float("UTOA_OTO_CRNN_VC_ANCHOR_MIN_CONFIDENCE", 0.20), 0.0, 1.0)
    for state, target_idx, role in resolved_states:
        if normalize_role(role) not in anchor_roles:
            continue
        conf = float(state.get("predicted_confidence", 1.0) if state.get("predicted_confidence", None) is not None else 1.0)
        if conf < min_conf:
            continue
        duration = max(1.0, float(state.get("duration_ms", 0.0) or 1.0))
        params = dict(state.get("params") or {})
        try:
            anchors = oto_params_to_anchors(
                offset=float(params.get("offset", 0.0) or 0.0),
                consonant=float(params.get("consonant", 0.0) or 0.0),
                cutoff=float(params.get("cutoff", 0.0) or 0.0),
                preutterance=float(params.get("preutterance", 0.0) or 0.0),
                overlap=float(params.get("overlap", 0.0) or 0.0),
                duration_ms=duration,
            )
        except Exception:
            continue
        idx = max(0, min(slot_count - 1, int(target_idx)))
        by_slot[idx].append(float(anchors.preutterance))
    out: list[float | None] = []
    for values in by_slot:
        if not values:
            out.append(None)
        else:
            out.append(float(np.median(np.asarray(values, dtype=np.float64))))
    return out


def _alias_slot_lock_target_time(
    audio: AudioCandidates,
    *,
    role: str,
    target_slot_idx: int,
    slot_times: list[float],
    anchor_times: list[float | None] | None = None,
) -> float:
    idx = max(0, min(len(slot_times) - 1, int(target_slot_idx)))
    slot_time = float(slot_times[idx])
    if normalize_role(role) != "vc":
        return slot_time

    duration = max(1.0, float(getattr(audio, "duration_ms", 0.0) or 1.0))
    anchor_pair = _vc_anchor_pair_bounds(anchor_times, idx=idx, duration_ms=duration)
    if anchor_pair is not None:
        left_anchor, right_anchor = anchor_pair
        target = _vc_target_between_anchors(
            audio,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            duration_ms=duration,
        )
        if target is not None:
            return target

    active_end = min(duration, float(getattr(audio, "active_end_ms", duration) or duration))
    right_slot = float(slot_times[idx + 1]) if idx + 1 < len(slot_times) else active_end
    if right_slot <= slot_time:
        right_slot = max(slot_time + 1.0, active_end)

    best_seg = None
    best_score = -1.0e9
    for seg in list(getattr(audio, "stable_vowel_segments", []) or []):
        start = float(getattr(seg, "start_ms", getattr(seg, "center_ms", 0.0)) or 0.0)
        end = float(getattr(seg, "end_ms", getattr(seg, "center_ms", 0.0)) or 0.0)
        center = float(getattr(seg, "center_ms", (start + end) * 0.5) or 0.0)
        if end < slot_time - 35.0 or start > right_slot + 120.0:
            continue
        conf = float(getattr(seg, "vowel_confidence", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        stable = float(getattr(seg, "stability_score", 0.0) or 0.0)
        quality = (0.45 * conf) + (0.35 * voiced) + (0.20 * stable)
        distance_penalty = abs(center - slot_time) / max(1.0, right_slot - slot_time)
        score = quality - (0.42 * distance_penalty)
        if score > best_score:
            best_score = score
            best_seg = seg

    fallback_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_FALLBACK_RATIO", 0.62), 0.0, 1.0)
    if best_seg is None:
        return slot_time + ((right_slot - slot_time) * fallback_ratio)

    vowel_end = float(getattr(best_seg, "end_ms", getattr(best_seg, "center_ms", slot_time)) or slot_time)
    onset_window = max(60.0, min(520.0, (right_slot - vowel_end) + 120.0))
    onset = find_nearest_onset_after(audio, vowel_end, search_window_ms=onset_window)
    transition_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_TARGET_RATIO", 0.62), 0.0, 1.0)
    if onset is None:
        return max(0.0, min(duration, vowel_end))
    onset_ms = float(getattr(onset, "time_ms", 0.0) or 0.0)
    if onset_ms <= vowel_end:
        return max(0.0, min(duration, vowel_end))
    target = vowel_end + ((onset_ms - vowel_end) * transition_ratio)
    return max(0.0, min(duration, target))


def _vc_anchor_pair_bounds(
    anchor_times: list[float | None] | None,
    *,
    idx: int,
    duration_ms: float,
) -> tuple[float, float] | None:
    if not anchor_times:
        return None
    if idx < 0 or idx >= len(anchor_times):
        return None
    left = anchor_times[idx]
    if left is None:
        return None
    right = None
    for right_idx in range(idx + 1, len(anchor_times)):
        value = anchor_times[right_idx]
        if value is not None:
            right = value
            break
    if right is None:
        right = max(float(left) + 1.0, float(duration_ms))
    if float(right) <= float(left) + _env_float("UTOA_OTO_CRNN_VC_ANCHOR_MIN_GAP_MS", 24.0):
        return None
    return float(left), float(right)


def _vc_target_between_anchors(
    audio: AudioCandidates,
    *,
    left_anchor: float,
    right_anchor: float,
    duration_ms: float,
) -> float | None:
    duration = max(1.0, float(duration_ms))
    left = max(0.0, min(duration, float(left_anchor)))
    right = max(left + 1.0, min(duration, float(right_anchor)))
    left_pad = max(0.0, _env_float("UTOA_OTO_CRNN_VC_ANCHOR_LEFT_PAD_MS", 45.0))
    right_pad = max(0.0, _env_float("UTOA_OTO_CRNN_VC_ANCHOR_RIGHT_PAD_MS", 60.0))
    window_lo = max(0.0, left - left_pad)
    window_hi = min(duration, right + right_pad)
    fallback_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_FALLBACK_RATIO", 0.62), 0.0, 1.0)
    fallback = left + ((right - left) * fallback_ratio)

    best_seg = None
    best_score = -1.0e9
    for seg in list(getattr(audio, "stable_vowel_segments", []) or []):
        start = float(getattr(seg, "start_ms", getattr(seg, "center_ms", 0.0)) or 0.0)
        end = float(getattr(seg, "end_ms", getattr(seg, "center_ms", 0.0)) or 0.0)
        center = float(getattr(seg, "center_ms", (start + end) * 0.5) or 0.0)
        if end < window_lo or start > window_hi:
            continue
        if center > right + right_pad:
            continue
        conf = float(getattr(seg, "vowel_confidence", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        stable = float(getattr(seg, "stability_score", 0.0) or 0.0)
        quality = (0.45 * conf) + (0.35 * voiced) + (0.20 * stable)
        distance_penalty = abs(center - left) / max(1.0, right - left)
        score = quality - (0.35 * distance_penalty)
        if score > best_score:
            best_score = score
            best_seg = seg

    if best_seg is None:
        return max(window_lo, min(window_hi, fallback))

    vowel_end = float(getattr(best_seg, "end_ms", getattr(best_seg, "center_ms", fallback)) or fallback)
    onset_window = max(40.0, window_hi - vowel_end)
    onset = find_nearest_onset_after(audio, vowel_end, search_window_ms=onset_window)
    transition_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_TARGET_RATIO", 0.62), 0.0, 1.0)
    if onset is None:
        return max(window_lo, min(window_hi, vowel_end))
    onset_ms = float(getattr(onset, "time_ms", 0.0) or 0.0)
    if onset_ms <= vowel_end or onset_ms > window_hi:
        return max(window_lo, min(window_hi, vowel_end))
    target = vowel_end + ((onset_ms - vowel_end) * transition_ratio)
    return max(window_lo, min(window_hi, target))


def _pick_alias_slot_times(audio: AudioCandidates, *, slot_count: int) -> list[float]:
    count = max(1, int(slot_count))
    duration = max(1.0, float(getattr(audio, "duration_ms", 0.0) or 1.0))
    active_start = max(0.0, float(getattr(audio, "active_start_ms", 0.0) or 0.0))
    active_end = max(active_start + 1.0, min(duration, float(getattr(audio, "active_end_ms", duration) or duration)))
    raw: list[tuple[float, float]] = []
    for peak in list(getattr(audio, "onset_peaks", []) or []):
        t = float(getattr(peak, "time_ms", 0.0) or 0.0)
        if active_start - 80.0 <= t <= active_end + 80.0:
            raw.append((t, max(0.05, float(getattr(peak, "strength", 0.0) or 0.0))))
    for seg in list(getattr(audio, "stable_vowel_segments", []) or []):
        conf = float(getattr(seg, "vowel_confidence", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        stable = float(getattr(seg, "stability_score", 0.0) or 0.0)
        score = max(0.05, min(1.0, (0.45 * conf) + (0.35 * voiced) + (0.20 * stable)))
        t = float(getattr(seg, "start_ms", getattr(seg, "center_ms", 0.0)) or 0.0)
        if active_start - 80.0 <= t <= active_end + 80.0:
            raw.append((t, score * 0.85))
    candidates = _merge_slot_time_candidates(raw, merge_ms=35.0)
    if len(candidates) < count:
        return _even_slot_times(active_start, active_end, count)
    return _select_monotonic_slot_times(candidates, active_start=active_start, active_end=active_end, slot_count=count)


def _merge_slot_time_candidates(candidates: list[tuple[float, float]], *, merge_ms: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for time_ms, score in sorted(candidates, key=lambda item: (float(item[0]), -float(item[1]))):
        if out and abs(float(time_ms) - float(out[-1][0])) <= float(merge_ms):
            prev_t, prev_s = out[-1]
            total = max(1e-6, float(prev_s) + float(score))
            merged_t = ((float(prev_t) * float(prev_s)) + (float(time_ms) * float(score))) / total
            out[-1] = (merged_t, max(float(prev_s), float(score)))
        else:
            out.append((float(time_ms), float(score)))
    return out


def _even_slot_times(active_start: float, active_end: float, slot_count: int) -> list[float]:
    count = max(1, int(slot_count))
    if count == 1:
        return [(float(active_start) + float(active_end)) * 0.5]
    span = max(1.0, float(active_end) - float(active_start))
    return [float(active_start) + (span * float(idx) / float(count - 1)) for idx in range(count)]


def _select_monotonic_slot_times(
    candidates: list[tuple[float, float]],
    *,
    active_start: float,
    active_end: float,
    slot_count: int,
) -> list[float]:
    count = max(1, int(slot_count))
    n = len(candidates)
    if n < count:
        return _even_slot_times(active_start, active_end, count)
    ideal = _even_slot_times(active_start, active_end, count)
    inf = float("inf")
    dp = np.full((count, n), inf, dtype=np.float64)
    prev = np.full((count, n), -1, dtype=np.int32)
    for j, (time_ms, score) in enumerate(candidates):
        dp[0, j] = abs(float(time_ms) - ideal[0]) - (float(score) * 45.0)
    for i in range(1, count):
        for j, (time_ms, score) in enumerate(candidates):
            if j < i:
                continue
            base = abs(float(time_ms) - ideal[i]) - (float(score) * 45.0)
            best = inf
            best_prev = -1
            for k in range(i - 1, j):
                gap = float(time_ms) - float(candidates[k][0])
                if gap < 35.0:
                    continue
                skip_penalty = float(j - k - 1) * 8.0
                value = float(dp[i - 1, k]) + base + skip_penalty
                if value < best:
                    best = value
                    best_prev = k
            if best_prev >= 0:
                dp[i, j] = best
                prev[i, j] = best_prev
    last = int(np.argmin(dp[count - 1]))
    if not np.isfinite(dp[count - 1, last]):
        return _even_slot_times(active_start, active_end, count)
    path = [0] * count
    j = last
    for i in range(count - 1, -1, -1):
        path[i] = int(j)
        j = int(prev[i, j]) if i > 0 else -1
    return [float(candidates[idx][0]) for idx in path]


def _translate_state_to_alias_slot(
    state: dict[str, Any],
    *,
    target_slot_idx: int,
    slot_times: list[float],
    target_time_ms: float | None = None,
    max_shift_ms: float,
    min_move_ms: float,
    mode: str = "direct",
) -> bool:
    params = dict(state.get("params") or {})
    duration = max(1.0, float(state.get("duration_ms", 0.0) or 1.0))
    try:
        anchors = oto_params_to_anchors(
            offset=float(params.get("offset", 0.0) or 0.0),
            consonant=float(params.get("consonant", 0.0) or 0.0),
            cutoff=float(params.get("cutoff", 0.0) or 0.0),
            preutterance=float(params.get("preutterance", 0.0) or 0.0),
            overlap=float(params.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
    except Exception:
        state["alias_slot_lock_hold"] = True
        state["alias_slot_lock_reason"] = "hold:invalid_params"
        return False
    current = float(anchors.preutterance)
    nearest_idx = min(range(len(slot_times)), key=lambda idx: abs(float(slot_times[idx]) - current))
    target_idx = max(0, min(len(slot_times) - 1, int(target_slot_idx)))
    lock_mode = str(mode or "direct").strip().lower()
    target_time = float(slot_times[target_idx] if target_time_ms is None else target_time_ms)
    if lock_mode in {"direct", "target", "target_preutterance", "absolute"}:
        delta = target_time - current
    else:
        if nearest_idx == target_idx:
            return False
        delta = target_time - float(slot_times[nearest_idx])
    if abs(delta) > float(max_shift_ms):
        state["alias_slot_lock_hold"] = True
        state["alias_slot_lock_reason"] = f"hold:max_shift:{lock_mode}:{nearest_idx}->{target_idx}:{delta:.1f}"
        return False
    if abs(delta) < float(min_move_ms):
        state["alias_slot_lock_hold"] = True
        state["alias_slot_lock_reason"] = f"hold:min_move:{lock_mode}:{nearest_idx}->{target_idx}:{delta:.1f}"
        return False
    shifted = OtoAnchors(
        offset=float(anchors.offset) + delta,
        overlap=float(anchors.overlap) + delta,
        preutterance=float(anchors.preutterance) + delta,
        consonant=float(anchors.consonant) + delta,
        cutoff=float(anchors.cutoff) + delta,
    )
    state["params"] = anchors_to_oto_params(shifted, duration_ms=duration)
    state["alias_slot_locked"] = True
    state["alias_slot_lock_reason"] = f"{lock_mode}:{nearest_idx}->{target_idx}:{delta:.1f}"
    return True


def _alias_is_special(alias: str, special_set: frozenset[str]) -> bool:
    if not special_set:
        return False
    text = str(alias or "").strip().lower()
    if not text:
        return False
    if text in special_set:
        return True
    # Match by individual token so a single special phoneme like "tt" tags
    # any alias that contains it (e.g. "a tt", "tt a", "a tt a").
    for token in text.split():
        if token in special_set:
            return True
    return False


def _assign_filename_slot_indices(
    *,
    wav_rel: str,
    items: list[dict[str, str]],
    language: str,
) -> list[dict[str, str]]:
    expected = _extract_filename_tokens(wav_rel, language=language)
    if len(expected) < 2 or not _env_bool("UTOA_OTO_CRNN_FILENAME_SLOT_ENABLE", True):
        for idx, row in enumerate(items):
            row["source_row_index"] = idx
            row["model_row_index"] = idx
            row["model_row_count"] = len(items)
            row["uses_filename_slots"] = False
            row["alias_role"] = _safe_alias_role(
                language,
                str(row.get("alias", "") or ""),
                alias_type=_safe_alias_type(language, str(row.get("alias", "") or "")),
                is_special=False,
            )
        return items

    cursors: dict[str, int] = {}
    slot_count = len(expected)
    source_count = max(1, len(items))
    for idx, row in enumerate(items):
        alias = str(row.get("alias", "") or "")
        alias_type = _safe_alias_type(language, alias)
        role = _safe_alias_role(language, alias, alias_type=alias_type, is_special=False)
        token = _slot_token_for_alias(alias, role=role, language=language)
        slot_idx = _find_slot_index_for_alias(expected, token, cursors, role=role, language=language)
        if slot_idx is None:
            if source_count <= 1:
                slot_idx = 0
            else:
                slot_idx = int(round((float(idx) / float(source_count - 1)) * float(max(0, slot_count - 1))))
        row["source_row_index"] = idx
        row["model_row_index"] = max(0, min(slot_count - 1, int(slot_idx)))
        row["model_row_count"] = slot_count
        row["uses_filename_slots"] = True
        row["alias_role"] = role
    return items


def _sequence_neighbors(items: list[dict[str, str]]) -> dict[int, tuple[str, str]]:
    ordered = sorted(
        items,
        key=lambda row: (
            int(row.get("model_row_index", row.get("source_row_index", 0)) or 0),
            slot_role_order_norm(row.get("alias_role", "")),
            int(row.get("source_row_index", 0) or 0),
        ),
    )
    out: dict[int, tuple[str, str]] = {}
    for idx, row in enumerate(ordered):
        prev_alias = str(ordered[idx - 1].get("alias", "") or "") if idx > 0 else ""
        next_alias = str(ordered[idx + 1].get("alias", "") or "") if idx + 1 < len(ordered) else ""
        out[id(row)] = (prev_alias, next_alias)
    return out


def _slot_token_for_alias(alias: str, *, role: str, language: str = "") -> str:
    tokens = _extract_alias_vowelish_tokens(alias, language=language)
    if not tokens:
        return ""
    role_text = normalize_role(role)
    if role_text in {"vv", "v-cv"} and len(tokens) >= 2:
        return tokens[-1]
    return tokens[0]


def _extract_alias_vowelish_tokens(alias: str, *, language: str = "") -> list[str]:
    out: list[str] = []
    for raw in _extract_text_syllable_tokens(alias, language=language):
        token = _normalize_order_token(raw)
        if token and any(ch in token for ch in "aeiou"):
            out.append(token)
    return out


def _find_slot_index(expected: list[str], token: str, cursors: dict[str, int]) -> int | None:
    token_text = _normalize_order_token(token)
    if not token_text:
        return None
    start = int(cursors.get(token_text, 0) or 0)
    for idx in range(max(0, start), len(expected)):
        if expected[idx] == token_text:
            cursors[token_text] = idx + 1
            return idx
    for idx, candidate in enumerate(expected):
        if candidate == token_text:
            return idx
    return None


def _find_slot_index_for_alias(
    expected: list[str],
    token: str,
    cursors: dict[str, int],
    *,
    role: str,
    language: str = "",
) -> int | None:
    exact = _find_slot_index(expected, token, cursors)
    if exact is not None:
        return exact
    lang = str(language or "").strip().lower()
    role_text = normalize_role(role)
    if lang != "korean" and not any(_roman_vowel_key(item) for item in expected):
        return None
    if role_text not in {"v", "v-", "vc", "vv", "v-cv"}:
        return None
    vowel = _roman_vowel_key(token)
    if not vowel:
        return None
    cursor_key = f"vowel:{vowel}"
    start = int(cursors.get(cursor_key, 0) or 0)
    for idx in range(max(0, start), len(expected)):
        if _roman_vowel_key(expected[idx]) == vowel:
            cursors[cursor_key] = idx + 1
            return idx
    for idx, candidate in enumerate(expected):
        if _roman_vowel_key(candidate) == vowel:
            return idx
    return None


_ROMAN_VOWEL_KEYS = (
    "yae",
    "yeo",
    "wae",
    "weo",
    "ae",
    "eo",
    "eu",
    "ui",
    "ya",
    "ye",
    "yo",
    "yu",
    "wa",
    "we",
    "wi",
    "oe",
    "a",
    "e",
    "i",
    "o",
    "u",
)


def _roman_vowel_key(token: str) -> str:
    text = _normalize_order_token(token)
    if not text:
        return ""
    for vowel in _ROMAN_VOWEL_KEYS:
        if vowel in text:
            return vowel
    return ""


def _build_wav_lookup(wav_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    if not os.path.isdir(wav_dir):
        return exact, normalized
    for current_root, _dirs, files in os.walk(wav_dir):
        for file_name in files:
            if not str(file_name).lower().endswith(".wav"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, wav_dir).replace("\\", "/")
            exact.setdefault(str(file_name).strip().lower(), rel_path)
            norm = normalize_wav_key(file_name)
            if norm:
                normalized.setdefault(norm, rel_path)
    return exact, normalized


def _apply_conservative_right_boundary_guard(
    params: dict[str, float],
    *,
    language: str,
    alias: str,
    duration_ms: float,
    is_special: bool = False,
) -> tuple[dict[str, float], bool]:
    if str(os.environ.get("UTOA_OTO_CRNN_RIGHT_GUARD_ENABLE", "1")).strip().lower() in {"0", "false", "off", "no"}:
        return dict(params), False

    out = {key: float(value) for key, value in dict(params).items()}
    pre = max(0.0, float(out.get("preutterance", 0.0)))
    ovl = max(0.0, min(float(out.get("overlap", 0.0)), pre))
    out["overlap"] = ovl

    alias_type = _safe_alias_type(language, alias)
    role = _safe_alias_role(language, alias, alias_type=alias_type, is_special=bool(is_special))
    diphthong = _safe_is_diphthong(language, alias)
    max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap = _right_guard_limits_for_role(
        role,
        alias_type=alias_type,
        is_diphthong=diphthong,
        is_special=bool(is_special),
    )
    max_cons_gap = _env_float("UTOA_OTO_CRNN_MAX_CONS_GAP_MS", max_cons_gap)
    max_cut_gap = _env_float("UTOA_OTO_CRNN_MAX_CUTOFF_GAP_MS", max_cut_gap)

    cons_raw = max(0.0, float(out.get("consonant", 0.0)))
    cons_min = pre + max(1.0, min_cons_gap)
    cons_max = pre + max(cons_min - pre, max_cons_gap)
    cons = _clamp(cons_raw, cons_min, cons_max)

    cutoff_mag_raw = abs(float(out.get("cutoff", 0.0)))
    cutoff_min = cons + max(1.0, min_cut_gap)
    cutoff_max = cons + max(cutoff_min - cons, max_cut_gap)
    offset = max(0.0, float(out.get("offset", 0.0)))
    if duration_ms > 1.0:
        cutoff_max = min(cutoff_max, max(cutoff_min, float(duration_ms) - offset - 1.0))
    cutoff_mag = _clamp(cutoff_mag_raw, cutoff_min, cutoff_max)

    out["consonant"] = cons
    out["cutoff"] = -cutoff_mag
    changed = (
        abs(cons - cons_raw) > 1e-6
        or abs(cutoff_mag - cutoff_mag_raw) > 1e-6
        or abs(ovl - float(params.get("overlap", 0.0))) > 1e-6
    )
    return out, changed


def _right_guard_limits(alias_type: str) -> tuple[float, float, float, float]:
    kind = str(alias_type or "").strip().lower()
    # return max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap
    if kind == "cv_head":
        return 92.0, 18.0, 150.0, 36.0
    if kind == "cv":
        return 78.0, 16.0, 130.0, 34.0
    if kind == "vcv":
        return 82.0, 16.0, 118.0, 30.0
    if kind == "vc":
        return 48.0, 10.0, 82.0, 22.0
    if kind == "vv":
        return 92.0, 16.0, 142.0, 32.0
    if kind == "br":
        return 42.0, 8.0, 72.0, 18.0
    return 82.0, 14.0, 128.0, 30.0


# Role-axis right-boundary limits. Mirrors `_right_guard_limits` shape but
# branches on alias_role so the same row gets the same guard regardless of
# voicebank format wrapper.
_ROLE_GUARD_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "-cv":  (92.0, 18.0, 150.0, 36.0),
    "cv":   (78.0, 16.0, 130.0, 34.0),
    "cv-":  (60.0, 10.0, 110.0, 24.0),
    "v":    (36.0,  4.0,  92.0, 14.0),
    "v-":   (36.0,  4.0, 104.0, 18.0),
    "vc":   (48.0, 10.0,  82.0, 22.0),
    "vv":   (72.0, 14.0, 116.0, 26.0),
    "v-cv": (82.0, 16.0, 118.0, 30.0),
    "endbr":(40.0,  6.0,  78.0, 16.0),
    "br":   (34.0,  4.0,  64.0, 12.0),
    "other":(82.0, 14.0, 128.0, 30.0),
    "special":(88.0, 18.0, 134.0, 32.0),
}


def _right_guard_limits_for_role(
    role: str,
    *,
    alias_type: str = "",
    is_diphthong: bool = False,
    is_special: bool = False,
) -> tuple[float, float, float, float]:
    role_text = "special" if bool(is_special) else normalize_role(role)
    if role_text in _ROLE_GUARD_LIMITS:
        max_cons, min_cons, max_cut, min_cut = _ROLE_GUARD_LIMITS[role_text]
    else:
        # Fall back to the legacy alias_type table for any role this code
        # path hasn't enumerated yet 窶・keeps unknown rows under the old guard.
        max_cons, min_cons, max_cut, min_cut = _right_guard_limits(alias_type)
    if bool(is_diphthong) and role_text in {"cv", "-cv", "cv-", "v-cv", "vv"}:
        max_cons *= 1.30
        max_cut *= 1.20
    if bool(is_special) and role_text in {"v", "v-"}:
        cv_max, cv_min, _cut_max, _cut_min = _ROLE_GUARD_LIMITS["cv"]
        max_cons = max(max_cons, cv_max)
        min_cons = max(min_cons, cv_min)
    return max_cons, min_cons, max_cut, min_cut


def _safe_alias_type(language: str, alias: str) -> str:
    try:
        return str(classify_alias_type(language, alias) or "other").strip().lower() or "other"
    except Exception:
        text = str(alias or "").strip()
        if text.startswith("- "):
            return "cv_head"
        if " " in text:
            return "vcv"
        return "cv" if text else "other"


def _safe_alias_role(
    language: str,
    alias: str,
    *,
    alias_type: str = "",
    is_special: bool = False,
) -> str:
    try:
        return classify_alias_role(
            language,
            alias,
            alias_type=alias_type,
            is_special=bool(is_special),
        )
    except Exception:
        return "special" if bool(is_special) else "other"


def _safe_is_diphthong(language: str, alias: str) -> bool:
    try:
        return bool(_is_diphthong(language, alias))
    except Exception:
        return False


def _normalize_order_token(token: str) -> str:
    text = str(token or "").strip().lower().strip("*-")
    if not text:
        return ""
    for suffix in _CODA_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _extract_filename_tokens(wav_rel: str, *, language: str = "") -> list[str]:
    stem = os.path.splitext(os.path.basename(str(wav_rel or "")))[0]
    tokens = _extract_text_syllable_tokens(stem, language=language)
    if tokens:
        return tokens
    out: list[str] = []
    for raw in _ORDER_TOKEN_RE.findall(stem):
        token = _normalize_order_token(raw)
        if token:
            out.append(token)
    return out


def _extract_alias_head_tokens(aliases: list[str], *, language: str = "") -> list[str]:
    out: list[str] = []
    for alias in aliases:
        parts = _extract_text_syllable_tokens(alias, language=language) or _ORDER_TOKEN_RE.findall(str(alias or ""))
        if not parts:
            continue
        token = _normalize_order_token(parts[0])
        if not token:
            continue
        if not any(ch in token for ch in "aeiou"):
            continue
        out.append(token)
    return out


def _extract_alias_core_tokens(aliases: list[str], *, language: str = "") -> list[str]:
    out: list[str] = []
    for alias in aliases:
        text = str(alias or "").strip()
        if not text or " " in text:
            continue
        parts = _extract_text_syllable_tokens(text, language=language) or _ORDER_TOKEN_RE.findall(text)
        if not parts:
            continue
        token = _normalize_order_token(parts[0])
        if not token:
            continue
        if not any(ch in token for ch in "aeiou"):
            continue
        out.append(token)
    return out


def _extract_text_syllable_tokens(text: object, *, language: str = "") -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    lang = str(language or "").strip().lower()
    if lang == "japanese" or _KANA_RE.search(raw):
        try:
            from core.ja_lab_generator import parse_ja_filename

            return [_normalize_order_token(item) for item in (parse_ja_filename(raw) or []) if _normalize_order_token(item)]
        except Exception:
            pass
    if lang == "korean" or _HANGUL_CHAR_RE.search(raw):
        tokens: list[str] = []
        for part in [item for item in _TOKEN_SEP_RE.split(raw) if item]:
            if _HANGUL_CHAR_RE.search(part):
                tokens.extend(_hangul_text_to_romaji_syllables(part))
            else:
                tokens.extend(_normalize_order_token(item) for item in _ORDER_TOKEN_RE.findall(part))
        return [token for token in tokens if token]
    return [_normalize_order_token(item) for item in _ORDER_TOKEN_RE.findall(raw) if _normalize_order_token(item)]


def _hangul_text_to_romaji_syllables(text: str) -> list[str]:
    try:
        from core.lab_generator import decompose_hangul_to_roman
    except Exception:
        decompose_hangul_to_roman = None
    out: list[str] = []
    for ch in str(text or ""):
        if _HANGUL_CHAR_RE.fullmatch(ch) and decompose_hangul_to_roman is not None:
            try:
                parts = [str(item or "").strip().lower() for item in decompose_hangul_to_roman(ch)]
            except Exception:
                parts = []
            token = _normalize_order_token("".join(part for part in parts if part))
            if token:
                out.append(token)
        else:
            token = _normalize_order_token(ch)
            if token:
                out.append(token)
    return out


def _ordered_ratio(expected: list[str], observed: list[str]) -> float:
    if not expected:
        return 1.0
    cursor = 0
    matched = 0
    for token in observed:
        if cursor >= len(expected):
            break
        if token == expected[cursor]:
            matched += 1
            cursor += 1
    return float(matched) / float(len(expected))


def _looks_like_cvvc_wav(wav_rel: str, aliases: list[str], *, language: str = "") -> bool:
    stem = os.path.splitext(os.path.basename(str(wav_rel or "")))[0].lower()
    if "'" in stem:
        return True
    token_count = len(_extract_filename_tokens(wav_rel, language=language))
    if token_count >= 6:
        return True
    if aliases:
        space_count = sum(1 for alias in aliases if " " in str(alias or ""))
        if (float(space_count) / float(len(aliases))) >= 0.35:
            return True
    return False


def _best_order_ratio(expected: list[str], observed: list[str]) -> float:
    if not observed:
        return 0.0
    expected_count = len(expected)
    candidates: list[list[str]] = [expected]
    if expected_count >= 3:
        candidates.append(expected[1:])
        candidates.append(expected[:-1])
    if expected_count >= 4:
        candidates.append(expected[1:-1])
    return max(_ordered_ratio(candidate, observed) for candidate in candidates if candidate)


def _estimate_filename_alias_order_ratio(wav_rel: str, aliases: list[str], *, language: str = "") -> tuple[float, int, bool]:
    expected = _extract_filename_tokens(wav_rel, language=language)
    expected_count = len(expected)
    if expected_count <= 0:
        return 1.0, 0, False
    is_cvvc_like = _looks_like_cvvc_wav(wav_rel, aliases, language=language)
    observed_all = _extract_alias_head_tokens(aliases, language=language)
    observed_core = _extract_alias_core_tokens(aliases, language=language)
    if is_cvvc_like:
        observed = observed_core or observed_all
        if not observed:
            return 0.0, expected_count, True
        ratio = _best_order_ratio(expected, observed)
        support = min(1.0, float(len(observed)) / float(max(1, expected_count)))
        score = (float(ratio) * 0.80) + (float(support) * 0.20)
        return float(score), expected_count, True
    if not observed_all:
        return 0.0, expected_count, False
    ratio = _best_order_ratio(expected, observed_all)
    return float(ratio), expected_count, False


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "off", "no"}


def _normalize_base_params(parsed: dict[str, Any] | None) -> dict[str, float]:
    row = dict(parsed or {})
    return {
        "offset": float(row.get("offset", 0.0) or 0.0),
        "consonant": float(row.get("cons", 0.0) or 0.0),
        "cutoff": float(row.get("cutoff", 0.0) or 0.0),
        "preutterance": float(row.get("pre", 0.0) or 0.0),
        "overlap": float(row.get("ovl", 0.0) or 0.0),
    }


def _apply_low_confidence_fallback(
    *,
    predicted_params: dict[str, float],
    predicted_confidence: float,
    predicted_error_ms: float | None,
    predicted_low_confidence: bool,
    confidence_components: dict[str, float] | None,
    base_params: dict[str, float] | None,
    language: str,
    alias: str,
    duration_ms: float,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_LOW_CONF_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    if not base_params:
        return dict(predicted_params), ""
    min_conf = _env_float("UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE", 0.35)
    max_error_ms = _env_float("UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS", 450.0)
    min_votes = max(1, int(round(_env_float("UTOA_OTO_CRNN_LOW_CONF_MIN_TRIGGER_VOTES", 2.0))))
    min_conf, max_error_ms, min_votes = _role_adjusted_low_conf_thresholds(
        min_conf=min_conf,
        max_error_ms=max_error_ms,
        min_votes=min_votes,
        language=language,
        alias=alias,
        is_special=bool(is_special),
    )
    trigger_conf = float(predicted_confidence) < float(min_conf)
    calibrated_error_ms = None
    if predicted_error_ms is not None:
        calibrated_error_ms = float(predicted_error_ms)
    trigger_err = calibrated_error_ms is not None and float(calibrated_error_ms) > float(max_error_ms)
    heatmap_conf = None
    uncertainty_conf = None
    try:
        heatmap_conf = float(dict(confidence_components or {}).get("heatmap"))
    except Exception:
        heatmap_conf = None
    try:
        raw_unc = dict(confidence_components or {}).get("uncertainty")
        uncertainty_conf = float(raw_unc) if raw_unc is not None else None
    except Exception:
        uncertainty_conf = None
    if trigger_err and _env_bool("UTOA_OTO_CRNN_LOW_CONF_REQUIRE_LOW_HEATMAP_FOR_ERROR", True):
        if heatmap_conf is not None and heatmap_conf >= _env_float("UTOA_OTO_CRNN_LOW_CONF_HEATMAP_MAX_FOR_ERROR", 0.35):
            trigger_err = False
    trigger_model = bool(predicted_low_confidence) and _env_bool("UTOA_OTO_CRNN_LOW_CONF_USE_MODEL_FLAG", False)
    component_signal = False
    if _env_bool("UTOA_OTO_CRNN_LOW_CONF_COMPONENT_SIGNAL_ENABLE", True):
        if heatmap_conf is not None and uncertainty_conf is not None:
            component_signal = bool(
                float(heatmap_conf) <= _env_float("UTOA_OTO_CRNN_LOW_CONF_MAX_HEATMAP", 0.32)
                and float(uncertainty_conf) <= _env_float("UTOA_OTO_CRNN_LOW_CONF_MAX_UNCERTAINTY", 0.014)
            )
    votes = int(bool(trigger_conf)) + int(bool(trigger_err)) + int(bool(component_signal))
    trigger = bool(trigger_model or votes >= min_votes)
    if not trigger:
        return dict(predicted_params), ""

    base = {key: float(value) for key, value in dict(base_params).items()}
    model = {key: float(value) for key, value in dict(predicted_params).items()}
    conf_gap = max(0.0, float(min_conf) - float(predicted_confidence))
    conf_denom = max(float(min_conf), 1e-6)
    strength = min(1.0, conf_gap / conf_denom)
    base_weight = _clamp(0.55 + (0.35 * strength), 0.45, 0.90)
    if trigger_err:
        base_weight = max(base_weight, 0.70)
    if component_signal and trigger_conf:
        base_weight = max(base_weight, 0.68)
    fused = {
        "offset": (model["offset"] * (1.0 - base_weight)) + (base["offset"] * base_weight),
        "consonant": (model["consonant"] * (1.0 - base_weight)) + (base["consonant"] * base_weight),
        "cutoff": (model["cutoff"] * (1.0 - base_weight)) + (base["cutoff"] * base_weight),
        "preutterance": (model["preutterance"] * (1.0 - base_weight)) + (base["preutterance"] * base_weight),
        "overlap": (model["overlap"] * (1.0 - base_weight)) + (base["overlap"] * base_weight),
    }
    guarded, _ = _apply_conservative_right_boundary_guard(
        fused,
        language=language,
        alias=alias,
        duration_ms=duration_ms,
        is_special=is_special,
    )
    tags: list[str] = []
    if trigger_conf:
        tags.append("low_confidence")
    if trigger_err:
        tags.append("high_predicted_error")
    if component_signal:
        tags.append("component_low_quality")
    if not tags:
        tags.append("low_confidence")
    reason = "+".join(tags)
    return guarded, reason


def _apply_monotonic_row_order_alignment(
    *,
    row_states: list[dict[str, Any]],
    cache: dict[str, AudioCandidates | None],
    config: Any | None,
    language: str,
    callback: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for state in row_states:
        row = state.get("row")
        if row is None:
            continue
        groups.setdefault(str(row.wav_rel), []).append(state)

    aligned = 0
    held = 0
    for wav_rel, states in groups.items():
        states.sort(key=lambda state: int(getattr(state.get("row"), "row_index", 0)))
        allowed_roles = _env_csv_set(
            "UTOA_OTO_CRNN_ROW_ORDER_ALIGN_ROLES",
            {"-cv", "cv", "v", "special"},
        )
        scoped_states = []
        for state in states:
            row = state["row"]
            role = normalize_role(
                _safe_alias_role(
                    language,
                    row.alias,
                    alias_type=_safe_alias_type(language, row.alias),
                    is_special=bool(row.is_special),
                )
            )
            if role in allowed_roles and slot_type_for_role(role) == "anchor":
                scoped_states.append(state)
        states = scoped_states
        if len(states) <= 1:
            continue
        wav_path = str(states[0]["row"].wav_abs)
        candidates = _get_audio_candidates(wav_path, cache=cache, config=config)
        if candidates is None:
            continue
        onset_points = _select_alignment_onset_points(
            candidates,
            row_count=len(states),
        )
        if len(onset_points) < len(states):
            continue
        path = _monotonic_onset_path(states, onset_points, language=language)
        if not path:
            continue
        prev_target = None
        prev_peak_index = None
        for row_idx, peak_idx in enumerate(path):
            state = states[row_idx]
            row = state["row"]
            params = dict(state["params"])
            duration_ms = float(state.get("duration_ms", 0.0) or 0.0)
            try:
                anchors = oto_params_to_anchors(
                    offset=float(params.get("offset", 0.0) or 0.0),
                    consonant=float(params.get("consonant", 0.0) or 0.0),
                    cutoff=float(params.get("cutoff", 0.0) or 0.0),
                    preutterance=float(params.get("preutterance", 0.0) or 0.0),
                    overlap=float(params.get("overlap", 0.0) or 0.0),
                    duration_ms=max(1.0, duration_ms),
                )
            except Exception:
                continue
            target_ms = float(onset_points[peak_idx][0])
            pred_mid = float(anchors.preutterance + anchors.consonant) * 0.5
            delta_ms = target_ms - pred_mid
            role = normalize_role(
                _safe_alias_role(
                    language,
                    row.alias,
                    alias_type=_safe_alias_type(language, row.alias),
                    is_special=bool(row.is_special),
                )
            )
            min_gap_ms = _row_order_role_min_gap_ms(role)
            max_shift_ms = max(0.0, _env_float("UTOA_OTO_CRNN_ROW_ORDER_MAX_SHIFT_MS", 260.0))
            max_step_gap_ms = max(0.0, _env_float("UTOA_OTO_CRNN_ROW_ORDER_MAX_STEP_GAP_MS", 420.0))
            max_skip_points = max(0, int(round(_env_float("UTOA_OTO_CRNN_ROW_ORDER_MAX_SKIP_POINTS", 2.0))))
            if prev_target is not None and target_ms < float(prev_target) + float(min_gap_ms):
                state["row_order_hold"] = True
                held += 1
                continue
            if prev_target is not None and (target_ms - float(prev_target)) > max_step_gap_ms:
                state["row_order_hold"] = True
                held += 1
                continue
            if prev_peak_index is not None and (int(peak_idx) - int(prev_peak_index) - 1) > max_skip_points:
                state["row_order_hold"] = True
                held += 1
                continue
            if abs(delta_ms) > max_shift_ms:
                state["row_order_hold"] = True
                held += 1
                continue
            shifted = _shift_params_to_target_mid(
                predicted_anchors={
                    "offset": float(anchors.offset),
                    "overlap": float(anchors.overlap),
                    "preutterance": float(anchors.preutterance),
                    "consonant": float(anchors.consonant),
                    "cutoff": float(anchors.cutoff),
                },
                duration_ms=max(1.0, duration_ms),
                target_mid_ms=target_ms,
                max_shift_ms=max_shift_ms,
            )
            min_move = _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_MOVE_MS", 3.0)
            if shifted and abs(delta_ms) >= min_move:
                guarded, _ = _apply_conservative_right_boundary_guard(
                    dict(shifted),
                    language=language,
                    alias=row.alias,
                    duration_ms=max(1.0, duration_ms),
                    is_special=bool(row.is_special),
                )
                state["params"] = dict(guarded)
                state["aligned_by_row_order"] = True
                aligned += 1
                prev_target = target_ms
                prev_peak_index = int(peak_idx)
            else:
                prev_target = target_ms
                prev_peak_index = int(peak_idx)
        _log(callback, f"[OTO-CRNN] row-order pass wav={os.path.basename(wav_rel)} rows={len(states)}")
    return aligned, held


def _select_alignment_onset_points(
    candidates: AudioCandidates,
    *,
    row_count: int,
) -> list[tuple[float, float]]:
    min_strength = _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_PEAK_STRENGTH", 0.18)
    active_pad = _env_float("UTOA_OTO_CRNN_ROW_ORDER_ACTIVE_PAD_MS", 120.0)
    active_start = float(getattr(candidates, "active_start_ms", 0.0) or 0.0) - active_pad
    active_end = float(getattr(candidates, "active_end_ms", 0.0) or 0.0) + active_pad
    points: list[tuple[float, float]] = []
    for peak in list(getattr(candidates, "onset_peaks", []) or []):
        time_ms = float(getattr(peak, "time_ms", 0.0) or 0.0)
        strength = float(getattr(peak, "strength", 0.0) or 0.0)
        if strength < min_strength:
            continue
        if time_ms < active_start or time_ms > active_end:
            continue
        points.append((time_ms, strength))
    points.sort(key=lambda item: item[0])
    if len(points) >= row_count:
        return points

    seg_points = []
    for seg in list(getattr(candidates, "stable_vowel_segments", []) or []):
        center = float(getattr(seg, "center_ms", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        if center < active_start or center > active_end:
            continue
        seg_points.append((center, max(0.05, voiced)))
    seg_points.sort(key=lambda item: item[0])
    merged = points + [item for item in seg_points if item not in points]
    merged.sort(key=lambda item: item[0])
    return merged


def _monotonic_onset_path(
    states: list[dict[str, Any]],
    onset_points: list[tuple[float, float]],
    *,
    language: str,
) -> list[int]:
    n_rows = len(states)
    n_peaks = len(onset_points)
    if n_rows <= 0 or n_peaks < n_rows:
        return []
    mids = []
    for state in states:
        params = dict(state.get("params") or {})
        try:
            anchors = oto_params_to_anchors(
                offset=float(params.get("offset", 0.0) or 0.0),
                consonant=float(params.get("consonant", 0.0) or 0.0),
                cutoff=float(params.get("cutoff", 0.0) or 0.0),
                preutterance=float(params.get("preutterance", 0.0) or 0.0),
                overlap=float(params.get("overlap", 0.0) or 0.0),
                duration_ms=max(1.0, float(state.get("duration_ms", 0.0) or 1.0)),
            )
            mids.append((float(anchors.preutterance) + float(anchors.consonant)) * 0.5)
        except Exception:
            mids.append(0.0)

    inf = float("inf")
    dp = np.full((n_rows, n_peaks), inf, dtype=np.float64)
    prev = np.full((n_rows, n_peaks), -1, dtype=np.int32)
    idx_penalty = _env_float("UTOA_OTO_CRNN_ROW_ORDER_INDEX_PENALTY", 40.0)
    skip_penalty = _env_float("UTOA_OTO_CRNN_ROW_ORDER_SKIP_PENALTY", 35.0)
    for j in range(n_peaks):
        rel_i = 0.0
        rel_j = float(j) / float(max(1, n_peaks - 1))
        dp[0, j] = abs(float(onset_points[j][0]) - mids[0]) + (abs(rel_j - rel_i) * idx_penalty)
    for i in range(1, n_rows):
        row = states[i]["row"]
        role = normalize_role(
            _safe_alias_role(
                language,
                row.alias,
                alias_type=_safe_alias_type(language, row.alias),
                is_special=bool(row.is_special),
            )
        )
        min_gap = _row_order_role_min_gap_ms(role)
        rel_i = float(i) / float(max(1, n_rows - 1))
        for j in range(i, n_peaks):
            target = float(onset_points[j][0])
            rel_j = float(j) / float(max(1, n_peaks - 1))
            base = abs(target - mids[i]) + (abs(rel_j - rel_i) * idx_penalty)
            best_score = inf
            best_prev = -1
            for k in range(i - 1, j):
                prev_t = float(onset_points[k][0])
                if target < prev_t + min_gap:
                    continue
                trans = float(dp[i - 1, k]) + base + (float(j - k - 1) * skip_penalty)
                if trans < best_score:
                    best_score = trans
                    best_prev = k
            if best_prev >= 0:
                dp[i, j] = best_score
                prev[i, j] = int(best_prev)
    last_j = int(np.argmin(dp[n_rows - 1]))
    if not np.isfinite(dp[n_rows - 1, last_j]):
        return []
    path = [0] * n_rows
    j = last_j
    for i in range(n_rows - 1, -1, -1):
        path[i] = int(j)
        j = int(prev[i, j]) if i > 0 else -1
    if any(path[idx] <= path[idx - 1] for idx in range(1, len(path))):
        return []
    return path


def _row_order_role_min_gap_ms(role_text: str) -> float:
    role = normalize_role(role_text)
    if role == "-cv":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_HEAD_MS", 26.0)
    if role == "cv":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_CV_MS", 16.0)
    if role == "cv-":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_CVTAIL_MS", 12.0)
    if role == "vc":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_VC_MS", 14.0)
    if role == "vv":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_VV_MS", 20.0)
    if role == "v-cv":
        return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_VCV_MS", 18.0)
    return _env_float("UTOA_OTO_CRNN_ROW_ORDER_MIN_GAP_DEFAULT_MS", 14.0)


def _resolve_effective_format(
    *,
    language: str,
    requested_format: str,
    aliases: list[str],
    callback: Callable[[str], None] | None = None,
) -> str:
    req = str(requested_format or "").strip().lower()
    if req and req not in {"other", "general", "auto"}:
        _log(callback, f"[OTO-CRNN] format={req} (requested)")
        return req
    detected = ""
    try:
        detected = str(detect_format_type(str(language or "").strip().lower(), list(aliases or [])) or "").strip().lower()
    except Exception:
        detected = ""
    if detected:
        _log(callback, f"[OTO-CRNN] format={detected} (detected)")
        return detected
    fallback = "general" if req in {"", "auto"} else (req or "general")
    _log(callback, f"[OTO-CRNN] format={fallback} (fallback)")
    return fallback


def _apply_activity_window_fallback(
    *,
    predicted_anchors: dict[str, float],
    predicted_params: dict[str, float],
    base_params: dict[str, float] | None,
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    row_index: int = 0,
    row_count: int = 1,
    language: str = "",
    alias: str = "",
    is_special: bool = False,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    profile = _analyze_activity_profile(wav_path, sample_rate=sample_rate, cache=cache)
    if not profile:
        return dict(predicted_params), ""
    row_shifted = _shift_params_into_row_activity_window(
        predicted_anchors=predicted_anchors,
        duration_ms=float(profile.get("duration_ms", 0.0) or 0.0),
        active_start_ms=float(profile.get("active_start_ms", 0.0) or 0.0),
        active_end_ms=float(profile.get("active_end_ms", 0.0) or 0.0),
        row_index=row_index,
        row_count=row_count,
        low_snr=bool(float(profile.get("low_snr", 0.0) or 0.0) >= 0.5),
        language=language,
        alias=alias,
        is_special=is_special,
    )
    if row_shifted:
        return row_shifted, "activity_row_shift"
    if not _is_anchor_outside_activity(predicted_anchors, profile):
        return dict(predicted_params), ""
    shifted = _shift_params_into_activity_window(
        predicted_anchors=predicted_anchors,
        duration_ms=float(profile.get("duration_ms", 0.0) or 0.0),
        active_start_ms=float(profile.get("active_start_ms", 0.0) or 0.0),
        active_end_ms=float(profile.get("active_end_ms", 0.0) or 0.0),
    )
    if shifted:
        return shifted, "activity_shift"
    if base_params and _env_bool("UTOA_OTO_CRNN_ACTIVITY_ALLOW_BASE_FALLBACK", False):
        return {key: float(value) for key, value in dict(base_params).items()}, "activity_outlier"
    return dict(predicted_params), ""


def _apply_audio_candidate_snap(
    *,
    predicted_params: dict[str, float],
    wav_path: str,
    language: str,
    alias: str,
    format_type: str = "",
    duration_ms: float,
    cache: dict[str, AudioCandidates | None],
    sequence_state: dict[str, int] | None = None,
    config: Any | None = None,
    is_special: bool = False,
    predicted_confidence: float | None = None,
    predicted_error_ms: float | None = None,
    predicted_low_confidence: bool = False,
    confidence_components: dict[str, float] | None = None,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ENABLE", True):
        return dict(predicted_params), ""
    role = _safe_alias_role(language, alias, alias_type=_safe_alias_type(language, alias), is_special=bool(is_special))
    allowed_roles = _env_csv_set(
        "UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ROLES",
        {"-cv", "cv", "cv-", "v-cv", "special"},
    )
    role_text = "special" if bool(is_special) else normalize_role(role)
    if role_text not in allowed_roles:
        return dict(predicted_params), ""
    format_text = _normalize_format_key(format_type) or _normalize_format_key(
        _infer_alias_format_for_runtime(language=language, alias=alias)
    )
    blocked_pairs = _env_pair_set(
        "UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_BLOCK_FORMAT_ROLES",
        {("cvc", "v-cv"), ("cvvc", "v-cv"), ("vcv", "-cv")},
    )
    if (format_text, role_text) in blocked_pairs:
        return dict(predicted_params), ""

    candidates = _get_audio_candidates(wav_path, cache=cache, config=config)
    if candidates is None:
        return dict(predicted_params), ""

    try:
        anchors = oto_params_to_anchors(
            offset=float(predicted_params.get("offset", 0.0) or 0.0),
            consonant=float(predicted_params.get("consonant", 0.0) or 0.0),
            cutoff=float(predicted_params.get("cutoff", 0.0) or 0.0),
            preutterance=float(predicted_params.get("preutterance", 0.0) or 0.0),
            overlap=float(predicted_params.get("overlap", 0.0) or 0.0),
            duration_ms=float(duration_ms),
        )
    except Exception:
        return dict(predicted_params), ""

    target_ms = float(anchors.preutterance)
    max_delta = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MAX_DELTA_MS", 55.0))
    min_strength = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MIN_STRENGTH", 0.20)
    forward_cap = float("inf")
    backward_cap = float("inf")
    if format_text == "cvvc":
        if role_text == "-cv":
            forward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_HEAD_MAX_FORWARD_DELTA_MS", 23.0))
            backward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_HEAD_MAX_BACKWARD_DELTA_MS", 48.0))
        elif role_text == "cv":
            forward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CV_MAX_FORWARD_DELTA_MS", 32.0))
            backward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CV_MAX_BACKWARD_DELTA_MS", 52.0))
        elif role_text == "cv-":
            forward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CVTAIL_MAX_FORWARD_DELTA_MS", 19.0))
            backward_cap = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CVTAIL_MAX_BACKWARD_DELTA_MS", 36.0))
    active_start = float(getattr(candidates, "active_start_ms", 0.0) or 0.0)
    active_end = float(getattr(candidates, "active_end_ms", duration_ms) or duration_ms)
    active_pad = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ACTIVE_PAD_MS", 80.0)
    if _env_bool("UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY", True):
        if not _is_snap_risky_prediction(
            target_ms=target_ms,
            active_start_ms=active_start,
            active_end_ms=active_end,
            predicted_confidence=predicted_confidence,
            predicted_error_ms=predicted_error_ms,
            predicted_low_confidence=predicted_low_confidence,
            confidence_components=confidence_components,
        ):
            return dict(predicted_params), ""
    sequence_enabled = _env_bool("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", True)
    sequence_key = os.path.abspath(str(wav_path or "")).lower()
    min_peak_index = -1
    if sequence_enabled and sequence_state is not None:
        min_peak_index = int(sequence_state.get(sequence_key, -1))
        max_delta = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_MAX_DELTA_MS", 100.0))

    peaks = []
    for peak_index, peak in enumerate(getattr(candidates, "onset_peaks", []) or []):
        if sequence_enabled and peak_index <= min_peak_index:
            continue
        time_ms = float(getattr(peak, "time_ms", 0.0) or 0.0)
        strength = float(getattr(peak, "strength", 0.0) or 0.0)
        if strength < min_strength:
            continue
        if time_ms < active_start - active_pad or time_ms > active_end + active_pad:
            continue
        delta = time_ms - target_ms
        if delta > forward_cap:
            continue
        if -delta > backward_cap:
            continue
        if abs(delta) <= max_delta:
            if sequence_enabled:
                skipped = max(0, int(peak_index) - int(min_peak_index) - 1)
                jump_penalty = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_JUMP_PENALTY", 0.20)
                score = abs(delta) - (30.0 * strength) + (jump_penalty * float(skipped))
            else:
                score = abs(delta)
            peaks.append((score, -strength, peak_index, time_ms, strength))
    if not peaks:
        return dict(predicted_params), ""

    _score, _neg_strength, peak_index, candidate_ms, strength = min(peaks)
    if sequence_enabled:
        blend = _clamp(_env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_BLEND", 0.40), 0.0, 1.0)
    else:
        blend = _clamp(_env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_BLEND", 0.65), 0.0, 1.0)
    delta = (float(candidate_ms) - target_ms) * blend
    if abs(delta) < _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MIN_MOVE_MS", 4.0):
        return dict(predicted_params), ""

    shifted = OtoAnchors(
        offset=anchors.offset + delta,
        overlap=anchors.overlap + delta,
        preutterance=anchors.preutterance + delta,
        consonant=anchors.consonant + delta,
        cutoff=anchors.cutoff + delta,
    )
    if sequence_enabled and sequence_state is not None:
        sequence_state[sequence_key] = int(peak_index)
    return anchors_to_oto_params(shifted, duration_ms=float(duration_ms)), f"candidate_onset:{strength:.3f}"


def _apply_vc_candidate_matcher(
    *,
    predicted_params: dict[str, float],
    wav_path: str,
    language: str,
    alias: str,
    format_type: str = "",
    duration_ms: float,
    cache: dict[str, AudioCandidates | None],
    config: Any | None = None,
    is_special: bool = False,
    predicted_confidence: float | None = None,
    predicted_error_ms: float | None = None,
    predicted_low_confidence: bool = False,
    confidence_components: dict[str, float] | None = None,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_VC_MATCHER_ENABLE", True):
        return dict(predicted_params), ""
    role = _safe_alias_role(language, alias, alias_type=_safe_alias_type(language, alias), is_special=bool(is_special))
    role_text = "special" if bool(is_special) else normalize_role(role)
    if role_text != "vc":
        return dict(predicted_params), ""
    format_text = _normalize_format_key(format_type) or _normalize_format_key(
        _infer_alias_format_for_runtime(language=language, alias=alias)
    )
    allowed_pairs = _env_pair_set(
        "UTOA_OTO_CRNN_VC_MATCHER_ALLOWED_FORMAT_ROLES",
        {("cvvc", "vc")},
    )
    if (format_text, role_text) not in allowed_pairs:
        return dict(predicted_params), ""

    components = parse_alias_components(alias, language=language)
    left_vowel_id = str(components.get("left_vowel_id", "") or "").strip().lower()
    right_consonant = str(components.get("right_consonant", "") or "").strip().lower()
    vowel_norm = float(_VOWEL_ID_NORM.get(left_vowel_id, 0.0))
    if vowel_norm <= 0.0 or not right_consonant:
        return dict(predicted_params), ""

    candidates = _get_audio_candidates(wav_path, cache=cache, config=config)
    if candidates is None:
        return dict(predicted_params), ""

    try:
        anchors = oto_params_to_anchors(
            offset=float(predicted_params.get("offset", 0.0) or 0.0),
            consonant=float(predicted_params.get("consonant", 0.0) or 0.0),
            cutoff=float(predicted_params.get("cutoff", 0.0) or 0.0),
            preutterance=float(predicted_params.get("preutterance", 0.0) or 0.0),
            overlap=float(predicted_params.get("overlap", 0.0) or 0.0),
            duration_ms=float(duration_ms),
        )
    except Exception:
        return dict(predicted_params), ""

    target_ms = float(anchors.preutterance)
    if float(predicted_params.get("preutterance", 0.0) or 0.0) <= _env_float("UTOA_OTO_CRNN_VC_MATCHER_MIN_PREUTTERANCE_MS", 2.0):
        return dict(predicted_params), ""
    active_start = float(getattr(candidates, "active_start_ms", 0.0) or 0.0)
    active_end = float(getattr(candidates, "active_end_ms", duration_ms) or duration_ms)
    active_tol = max(0.0, _env_float("UTOA_OTO_CRNN_VC_MATCHER_ACTIVE_TOLERANCE_MS", 70.0))
    early_boundary_risk = target_ms < (active_start - active_tol)
    late_boundary_risk = target_ms > (active_end + active_tol)
    if _env_bool("UTOA_OTO_CRNN_VC_MATCHER_REQUIRE_BOUNDARY_RISK", True):
        if not (early_boundary_risk or late_boundary_risk):
            return dict(predicted_params), ""
    if _env_bool("UTOA_OTO_CRNN_VC_MATCHER_REQUIRE_RISKY", True):
        if not _is_snap_risky_prediction(
            target_ms=target_ms,
            active_start_ms=active_start,
            active_end_ms=active_end,
            predicted_confidence=predicted_confidence,
            predicted_error_ms=predicted_error_ms,
            predicted_low_confidence=predicted_low_confidence,
            confidence_components=confidence_components,
        ):
            return dict(predicted_params), ""

    hint_weight = _clamp(_env_float("UTOA_OTO_CRNN_VC_MATCHER_HINT_WEIGHT", 0.35), 0.0, 1.0)
    seg = find_best_vowel_segment(
        candidates,
        vowel_norm,
        hint_center_ms=target_ms,
        hint_weight=hint_weight,
    )
    if seg is None:
        return dict(predicted_params), ""

    onset_window_ms = max(60.0, _env_float("UTOA_OTO_CRNN_VC_MATCHER_ONSET_WINDOW_MS", 420.0))
    onset = find_nearest_onset_after(
        candidates,
        float(seg.end_ms),
        search_window_ms=onset_window_ms,
    )
    if onset is None:
        return dict(predicted_params), ""

    target_pos = _clamp(_env_float("UTOA_OTO_CRNN_VC_MATCHER_TARGET_RATIO", 0.62), 0.0, 1.0)
    onset_ms = float(getattr(onset, "time_ms", 0.0) or 0.0)
    if onset_ms <= float(seg.end_ms) + 1e-3:
        return dict(predicted_params), ""
    candidate_pre = float(seg.end_ms) + ((onset_ms - float(seg.end_ms)) * target_pos)
    raw_delta = candidate_pre - target_ms
    if early_boundary_risk and raw_delta <= 0.0:
        return dict(predicted_params), ""
    if late_boundary_risk and raw_delta >= 0.0:
        return dict(predicted_params), ""

    max_delta = max(0.0, _env_float("UTOA_OTO_CRNN_VC_MATCHER_MAX_DELTA_MS", 220.0))
    if abs(raw_delta) > max_delta:
        return dict(predicted_params), ""

    blend = _clamp(_env_float("UTOA_OTO_CRNN_VC_MATCHER_BLEND", 0.55), 0.0, 1.0)
    delta = raw_delta * blend
    max_move = max(0.0, _env_float("UTOA_OTO_CRNN_VC_MATCHER_MAX_MOVE_MS", 65.0))
    if max_move > 0.0:
        delta = _clamp(delta, -max_move, max_move)
    if abs(delta) < _env_float("UTOA_OTO_CRNN_VC_MATCHER_MIN_MOVE_MS", 5.0):
        return dict(predicted_params), ""

    min_conf = _clamp(_env_float("UTOA_OTO_CRNN_VC_MATCHER_MIN_CANDIDATE_CONFIDENCE", 0.25), 0.0, 1.0)
    candidate_conf = float(
        (0.55 * float(getattr(seg, "vowel_confidence", 0.0) or 0.0))
        + (0.45 * float(getattr(onset, "strength", 0.0) or 0.0))
    )
    if candidate_conf < min_conf:
        return dict(predicted_params), ""

    # VC matcher should affect transition timing itself. Keep offset fixed and
    # move the right-side anchors so preutterance/consonant/cutoff can change
    # in param space (plain anchor translation keeps preutterance param intact).
    shifted = OtoAnchors(
        offset=anchors.offset,
        overlap=anchors.overlap + delta,
        preutterance=anchors.preutterance + delta,
        consonant=anchors.consonant + delta,
        cutoff=anchors.cutoff + delta,
    )
    return anchors_to_oto_params(shifted, duration_ms=float(duration_ms)), f"vc_matcher:{candidate_conf:.3f}"


def _is_snap_risky_prediction(
    *,
    target_ms: float,
    active_start_ms: float,
    active_end_ms: float,
    predicted_confidence: float | None,
    predicted_error_ms: float | None,
    predicted_low_confidence: bool,
    confidence_components: dict[str, float] | None,
) -> bool:
    early_tol = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_RISK_ACTIVE_TOLERANCE_MS", 70.0))
    if float(target_ms) < float(active_start_ms) - early_tol:
        return True
    if float(target_ms) > float(active_end_ms) + early_tol:
        return True
    if bool(predicted_low_confidence):
        return True
    conf_value = None
    if predicted_confidence is not None:
        try:
            conf_value = float(predicted_confidence)
        except Exception:
            conf_value = None
    if conf_value is not None and conf_value < _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_RISK_MAX_CONFIDENCE", 0.48):
        return True
    if predicted_error_ms is not None:
        try:
            err = float(predicted_error_ms)
        except Exception:
            err = None
        if err is not None and err > _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_RISK_MIN_PREDICTED_ERROR_MS", 380.0):
            return True
    heatmap = None
    uncertainty = None
    try:
        comp = dict(confidence_components or {})
    except Exception:
        comp = {}
    try:
        raw_heat = comp.get("heatmap")
        heatmap = float(raw_heat) if raw_heat is not None else None
    except Exception:
        heatmap = None
    try:
        raw_unc = comp.get("uncertainty")
        uncertainty = float(raw_unc) if raw_unc is not None else None
    except Exception:
        uncertainty = None
    if heatmap is not None and uncertainty is not None:
        if (
            heatmap <= _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_RISK_MAX_HEATMAP", 0.34)
            and uncertainty <= _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_RISK_MAX_UNCERTAINTY", 0.018)
        ):
            return True
    return False


def _get_audio_candidates(
    wav_path: str,
    *,
    cache: dict[str, AudioCandidates | None],
    config: Any | None = None,
) -> AudioCandidates | None:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return cache[key]
    try:
        candidates = compute_audio_candidates(
            str(wav_path),
            target_sr=int(getattr(config, "sample_rate", 16000) or 16000),
            n_mels=int(getattr(config, "n_mels", 64) or 64),
            frame_ms=float(getattr(config, "frame_ms", 25.0) or 25.0),
            hop_ms=float(getattr(config, "hop_ms", 10.0) or 10.0),
        )
    except Exception:
        candidates = None
    cache[key] = candidates
    return candidates


def _env_csv_set(name: str, default: set[str]) -> set[str]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return set(default)
    out = {normalize_role(item.strip()) for item in raw.split(",") if item.strip()}
    return out or set(default)


def _env_pair_set(name: str, default: set[tuple[str, str]]) -> set[tuple[str, str]]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return set(default)
    out: set[tuple[str, str]] = set()
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if "|" not in text:
            continue
        left, right = text.split("|", 1)
        fmt = _normalize_format_key(left)
        role = normalize_role(right)
        if fmt and role:
            out.add((fmt, role))
    return out


def _normalize_format_key(value: object) -> str:
    return str(value or "").strip().lower() or "unknown"


def _role_adjusted_low_conf_thresholds(
    *,
    min_conf: float,
    max_error_ms: float,
    min_votes: int,
    language: str,
    alias: str,
    is_special: bool,
) -> tuple[float, float, int]:
    if not _env_bool("UTOA_OTO_CRNN_LOW_CONF_ROLE_ADAPT_ENABLE", True):
        return float(min_conf), float(max_error_ms), int(min_votes)
    alias_type = _safe_alias_type(language, alias)
    role_text = normalize_role(
        _safe_alias_role(
            language,
            alias,
            alias_type=alias_type,
            is_special=bool(is_special),
        )
    )
    relax_roles = _env_csv_set(
        "UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_ROLES",
        {"vc", "v-cv", "vv"},
    )
    if role_text not in relax_roles:
        return float(min_conf), float(max_error_ms), int(min_votes)
    conf_delta = max(0.0, _env_float("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_CONF_DELTA", 0.04))
    err_delta_ms = max(0.0, _env_float("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_ERROR_DELTA_MS", 90.0))
    votes_delta = max(0, int(round(_env_float("UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_EXTRA_VOTES", 1.0))))
    return (
        max(0.05, float(min_conf) - conf_delta),
        float(max_error_ms) + err_delta_ms,
        max(1, int(min_votes) + votes_delta),
    )


def _infer_alias_format_for_runtime(*, language: str, alias: str) -> str:
    # Runtime generation already resolved one format for the whole batch, but
    # snap risk is role-and-bank-format dependent. Alias text gives a useful
    # local fallback when the batch format is broad or absent.
    try:
        return str(detect_format_type(str(language or "").strip().lower(), [str(alias or "")]) or "").strip().lower()
    except Exception:
        return ""


def _shift_params_into_row_activity_window(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    active_start_ms: float,
    active_end_ms: float,
    row_index: int,
    row_count: int,
    low_snr: bool = False,
    language: str = "",
    alias: str = "",
    is_special: bool = False,
) -> dict[str, float]:
    count = max(1, int(row_count))
    mid = _anchor_mid_ms(predicted_anchors)
    auto_early_margin_ms = max(0.0, _env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_EARLY_MARGIN_MS", 70.0))
    front_early = mid < (max(0.0, float(active_start_ms)) - auto_early_margin_ms)
    row_shift_enable = _env_bool("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", False)
    low_snr_auto_enable = _env_bool("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_LOW_SNR_ENABLE", False)
    front_auto_enable = _env_bool("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_FRONT_EARLY_ENABLE", False)
    role = _safe_alias_role(language, alias, alias_type=_safe_alias_type(language, alias), is_special=bool(is_special))
    role_text = "special" if bool(is_special) else normalize_role(role)
    auto_roles = _env_csv_set("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_ROLES", {"-cv", "cv", "special"})
    auto_role_allowed = role_text in auto_roles
    auto_trigger = auto_role_allowed and ((bool(low_snr) and low_snr_auto_enable) or (front_early and front_auto_enable))
    if count <= 1 or not (row_shift_enable or auto_trigger):
        return {}
    duration = max(1.0, float(duration_ms))
    active_start = max(0.0, float(active_start_ms))
    active_end = min(duration, max(active_start, float(active_end_ms)))
    active_span = active_end - active_start
    if active_span <= 40.0:
        return {}

    idx = max(0, min(count - 1, int(row_index)))
    row_width = active_span / float(count)
    if row_width < _env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_MIN_WIDTH_MS", 35.0):
        return {}
    row_start = active_start + (float(idx) * row_width)
    row_end = active_start + (float(idx + 1) * row_width)
    tolerance = _env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_TOLERANCE_MS", 90.0)
    if row_start - tolerance <= mid <= row_end + tolerance:
        return {}

    inset_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_TARGET_RATIO", 0.35), 0.05, 0.95)
    target_mid = row_start + (row_width * inset_ratio)
    return _shift_params_to_target_mid(
        predicted_anchors=predicted_anchors,
        duration_ms=duration,
        target_mid_ms=target_mid,
    )


def _shift_params_into_activity_window(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    active_start_ms: float,
    active_end_ms: float,
) -> dict[str, float]:
    duration = max(1.0, float(duration_ms))
    start_ms = max(0.0, float(active_start_ms))
    end_ms = min(duration, max(start_ms, float(active_end_ms)))
    if end_ms <= start_ms + 20.0:
        return {}
    mid = _anchor_mid_ms(predicted_anchors)
    tolerance_ms = _env_float("UTOA_OTO_CRNN_ACTIVITY_EARLY_TOLERANCE_MS", 80.0)
    if mid >= start_ms - max(0.0, float(tolerance_ms)):
        return {}
    target_mid = start_ms + _env_float("UTOA_OTO_CRNN_ACTIVITY_TARGET_INSET_MS", 35.0)
    return _shift_params_to_target_mid(
        predicted_anchors=predicted_anchors,
        duration_ms=duration,
        target_mid_ms=target_mid,
        max_shift_ms=_env_float("UTOA_OTO_CRNN_ACTIVITY_MAX_SHIFT_MS", 3000.0),
    )


def _shift_params_to_target_mid(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    target_mid_ms: float,
    max_shift_ms: float | None = None,
) -> dict[str, float]:
    duration = max(1.0, float(duration_ms))
    anchors = {
        "offset": float(predicted_anchors.get("offset", 0.0) or 0.0),
        "overlap": float(predicted_anchors.get("overlap", predicted_anchors.get("offset", 0.0)) or 0.0),
        "preutterance": float(predicted_anchors.get("preutterance", 0.0) or 0.0),
        "consonant": float(predicted_anchors.get("consonant", 0.0) or 0.0),
        "cutoff": float(predicted_anchors.get("cutoff", 0.0) or 0.0),
    }
    mid = (anchors["preutterance"] + anchors["consonant"]) * 0.5
    delta = float(target_mid_ms) - float(mid)
    if abs(delta) <= 1e-6:
        return {}
    if max_shift_ms is not None:
        cap = max(0.0, float(max_shift_ms))
        delta = _clamp(delta, -cap, cap)
    if delta > 0.0:
        delta = min(delta, max(0.0, duration - anchors["cutoff"] - 1.0))
    else:
        delta = max(delta, -min(anchors.values()))
    if abs(delta) <= 1e-6:
        return {}
    shifted = OtoAnchors(
        offset=anchors["offset"] + delta,
        overlap=anchors["overlap"] + delta,
        preutterance=anchors["preutterance"] + delta,
        consonant=anchors["consonant"] + delta,
        cutoff=anchors["cutoff"] + delta,
    )
    return anchors_to_oto_params(shifted, duration_ms=duration)


def _anchor_mid_ms(predicted_anchors: dict[str, float]) -> float:
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    return (pre + cons) * 0.5


def _analyze_activity_profile(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return dict(cache[key])
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        cache[key] = {}
        return {}
    duration_ms = max(1.0, float(duration_sec) * 1000.0)
    if samples is None or int(getattr(samples, "shape", [0])[0]) <= 0 or sr <= 0:
        profile = {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "low_snr": 0.0,
            "snr_proxy": 1.0,
        }
        cache[key] = profile
        return dict(profile)
    data = np.asarray(samples, dtype=np.float32)
    if data.size <= 0:
        profile = {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "low_snr": 0.0,
            "snr_proxy": 1.0,
        }
        cache[key] = profile
        return dict(profile)

    window_ms = _env_float("UTOA_OTO_CRNN_ACTIVITY_WINDOW_MS", 15.0)
    window_samples = max(1, int(round(float(sr) * max(1.0, float(window_ms)) / 1000.0)))
    if data.size <= window_samples:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)

    stride_samples = max(1, window_samples // 2)
    frame_count = 1 + int((data.size - window_samples) // stride_samples)
    rms = np.empty((frame_count,), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * stride_samples
        frame = data[start : start + window_samples]
        rms[idx] = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-8:
        profile = {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "low_snr": 0.0,
            "snr_proxy": 1.0,
        }
        cache[key] = profile
        return dict(profile)
    low = float(np.percentile(rms, _env_float("UTOA_OTO_CRNN_ACTIVITY_NOISE_PERCENTILE", 20.0)))
    high = float(np.percentile(rms, _env_float("UTOA_OTO_CRNN_ACTIVITY_SPEECH_PERCENTILE", 88.0)))
    dynamic = low + ((max(high, peak) - low) * _env_float("UTOA_OTO_CRNN_ACTIVITY_DYNAMIC_RATIO", 0.35))
    threshold = max(dynamic, peak * _env_float("UTOA_OTO_CRNN_ACTIVITY_PEAK_RATIO", 0.04), _env_float("UTOA_OTO_CRNN_ACTIVITY_ABS_FLOOR", 1e-5))
    stride_ms = (float(stride_samples) / float(sr)) * 1000.0
    min_run = max(1, int(round(_env_float("UTOA_OTO_CRNN_ACTIVITY_MIN_ACTIVE_MS", 30.0) / max(stride_ms, 1e-6))))
    snr_proxy = (high - low) / max(peak, 1e-6)
    low_snr = bool(snr_proxy < _env_float("UTOA_OTO_CRNN_ACTIVITY_LOW_SNR_PROXY_MAX", 0.18))
    if low_snr and _env_bool("UTOA_OTO_CRNN_ACTIVITY_NOISE_ROBUST_ENABLE", True):
        threshold = max(
            threshold,
            low + ((max(high, peak) - low) * _env_float("UTOA_OTO_CRNN_ACTIVITY_NOISE_ROBUST_DYNAMIC_RATIO", 0.52)),
        )
        robust_min_active_ms = _env_float("UTOA_OTO_CRNN_ACTIVITY_NOISE_ROBUST_MIN_ACTIVE_MS", 55.0)
        min_run = max(min_run, max(1, int(round(robust_min_active_ms / max(stride_ms, 1e-6)))))
    active_frames = _sustained_frame_indices(
        rms >= threshold,
        min_run,
        values=rms,
        prefer_strongest=_env_bool("UTOA_OTO_CRNN_ACTIVITY_PICK_STRONGEST_RUN", True),
    )
    if active_frames.size <= 0:
        threshold = max(low + ((peak - low) * 0.20), _env_float("UTOA_OTO_CRNN_ACTIVITY_ABS_FLOOR", 1e-5))
        active_frames = _sustained_frame_indices(
            rms >= threshold,
            min_run,
            values=rms,
            prefer_strongest=_env_bool("UTOA_OTO_CRNN_ACTIVITY_PICK_STRONGEST_RUN", True),
        )
    if active_frames.size <= 0:
        profile = {
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "low_snr": 1.0 if low_snr else 0.0,
            "snr_proxy": float(snr_proxy),
        }
        cache[key] = profile
        return dict(profile)

    start_ms = float(active_frames[0]) * stride_ms
    end_ms = (float(active_frames[-1]) * stride_ms) + float(window_ms)
    pad_ms = 25.0
    active_start = max(0.0, start_ms - pad_ms)
    active_end = min(duration_ms, end_ms + pad_ms)
    if active_end - active_start < 40.0:
        active_start = 0.0
        active_end = duration_ms
    profile = {
        "active_start_ms": float(active_start),
        "active_end_ms": float(active_end),
        "duration_ms": float(duration_ms),
        "low_snr": 1.0 if low_snr else 0.0,
        "snr_proxy": float(snr_proxy),
    }
    cache[key] = profile
    return dict(profile)


def _sustained_frame_indices(
    mask: np.ndarray,
    min_run: int,
    *,
    values: np.ndarray | None = None,
    prefer_strongest: bool = False,
) -> np.ndarray:
    return _sustained_frame_indices_impl(mask, min_run, values=values, prefer_strongest=prefer_strongest)


def _sustained_frame_indices_impl(
    mask: np.ndarray,
    min_run: int,
    *,
    values: np.ndarray | None,
    prefer_strongest: bool,
) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    run = max(1, int(min_run))
    if arr.size <= 0:
        return np.asarray([], dtype=np.int64)
    if run <= 1:
        idx = np.flatnonzero(arr)
        if idx.size <= 0:
            return np.asarray([], dtype=np.int64)
        if not bool(prefer_strongest) or values is None:
            return idx
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        if vals.size != arr.size:
            return idx
        best = int(idx[np.argmax(vals[idx])])
        return np.arange(best, idx[-1] + 1, dtype=np.int64)[arr[best:]]
    count = 0
    runs: list[tuple[int, int]] = []
    start = -1
    for idx, value in enumerate(arr):
        if bool(value):
            count += 1
            if count == run:
                start = idx - run + 1
        else:
            if start >= 0:
                runs.append((start, idx - 1))
                start = -1
            count = 0
    if start >= 0:
        runs.append((start, arr.size - 1))
    if not runs:
        return np.asarray([], dtype=np.int64)
    if not bool(prefer_strongest) or values is None:
        first, last = runs[0]
        return np.arange(first, last + 1, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size != arr.size:
        first, last = runs[0]
        return np.arange(first, last + 1, dtype=np.int64)
    len_weight = _env_float("UTOA_OTO_CRNN_ACTIVITY_STRONGEST_RUN_LENGTH_WEIGHT", 0.02)
    best_score = None
    best_run = runs[0]
    for first, last in runs:
        seg = vals[first : last + 1]
        if seg.size <= 0:
            continue
        mean_energy = float(np.mean(seg))
        length_score = float(seg.size) * float(len_weight)
        score = mean_energy + length_score
        if best_score is None or score > best_score:
            best_score = score
            best_run = (first, last)
    first, last = best_run
    return np.arange(first, last + 1, dtype=np.int64)


def _is_anchor_outside_activity(predicted_anchors: dict[str, float], profile: dict[str, float]) -> bool:
    start_ms = float(profile.get("active_start_ms", 0.0) or 0.0)
    end_ms = float(profile.get("active_end_ms", 0.0) or 0.0)
    if end_ms <= start_ms + 20.0:
        return False
    offset = float(predicted_anchors.get("offset", 0.0) or 0.0)
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    cutoff = float(predicted_anchors.get("cutoff", 0.0) or 0.0)
    mid = (pre + cons) * 0.5
    if mid < (start_ms - 80.0) or mid > (end_ms + 80.0):
        return True
    if cutoff < (start_ms - 40.0):
        return True
    if offset > (end_ms + 40.0):
        return True
    return False


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if raw.endswith(("\\", "/")) or os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    return raw


def _normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _apply_alias_suffix(alias: str, suffix: str) -> str:
    text = str(alias or "").strip()
    if not text or not suffix:
        return text
    return f"{text}_{suffix}"


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


__all__ = [
    "DEFAULT_OTO_CRNN_MODEL_NAME",
    "generate_oto_with_crnn_predictor",
    "resolve_oto_crnn_model_path",
]

