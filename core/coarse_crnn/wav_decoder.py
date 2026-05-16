from __future__ import annotations

from collections import defaultdict
import os
from statistics import mean

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.anchor_timeline import (
    DEFAULT_VALLEY_MIN_DEPTH_RATIO,
    DEFAULT_VALLEY_MIN_GAP_MS,
    anchor_timeline_mode_from_env,
    build_hybrid_anchor_timeline,
    env_float as _anchor_env_float,
)
from core.coarse_crnn.boundary_types import (
    TRANSITION_ROLES,
    BoundaryCandidate,
    BoundaryDecodeResult,
    DecodedOtoRow,
    OtoRowSpec,
    role_label_preferences,
)
from core.coarse_crnn.filename_anchor_timeline import build_filename_anchor_timeline
from core.coarse_crnn.lang import infer_language_from_path
from core.coarse_crnn.oto_param_builder import build_absolute_anchors_for_role


def decode_wav_rows(
    *,
    wav_path: str,
    duration_ms: float,
    row_specs: list[OtoRowSpec],
    candidates: list[BoundaryCandidate],
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    min_anchor_gap_ms: float = 24.0,
    model_quality: float | None = None,
    audio_reliability: float | None = None,
) -> BoundaryDecodeResult:
    if not row_specs:
        return BoundaryDecodeResult(
            wav_path=wav_path,
            duration_ms=float(duration_ms),
            rows=[],
            anchor_timeline_ms=[],
            fallback_count=0,
        )
    slot_count = max(1, max(int(spec.slot_count) for spec in row_specs))
    anchor_timeline: list[float] | None = None
    timeline_mode = anchor_timeline_mode_from_env(default="linear")
    active_end_base = float(active_end_ms) if active_end_ms is not None else float(duration_ms)
    active_end_pad_ms = max(0.0, _env_float("UTOA_BOUNDARY_ACTIVE_END_PAD_MS", 0.0))
    active_end_eff = min(float(duration_ms), active_end_base + active_end_pad_ms)
    filename_end_pad_ms = max(0.0, _env_float("UTOA_BOUNDARY_FILENAME_END_PAD_MS", 0.0))
    if timeline_mode == "filename" and wav_path:
        try:
            language = (
                str(row_specs[0].language)
                or infer_language_from_path(str(wav_path))
                or ""
            )
            filename_only = os.path.basename(str(wav_path))
            filename_active_end = min(float(duration_ms), active_end_eff + filename_end_pad_ms)
            anchor_timeline = build_filename_anchor_timeline(
                filename_only,
                slot_count=slot_count,
                active_start_ms=float(active_start_ms),
                active_end_ms=filename_active_end,
                language=language,
            )
        except Exception:
            anchor_timeline = None
    elif timeline_mode == "hybrid" and wav_path:
        try:
            hybrid = build_hybrid_anchor_timeline(
                wav_path=str(wav_path),
                slot_count=slot_count,
                active_start_ms=float(active_start_ms),
                active_end_ms=active_end_eff,
                candidates=candidates,
                valley_min_depth_ratio=_anchor_env_float(
                    "UTOA_BOUNDARY_ANCHOR_VALLEY_MIN_DEPTH",
                    DEFAULT_VALLEY_MIN_DEPTH_RATIO,
                ),
                valley_min_gap_ms=_anchor_env_float(
                    "UTOA_BOUNDARY_ANCHOR_VALLEY_MIN_GAP_MS",
                    DEFAULT_VALLEY_MIN_GAP_MS,
                ),
            )
        except Exception:
            hybrid = None
        if hybrid is not None and hybrid.timeline:
            anchor_timeline = list(hybrid.timeline)
    if anchor_timeline is None:
        anchor_timeline = _build_anchor_timeline(
            slot_count=slot_count,
            duration_ms=duration_ms,
            candidates=candidates,
            active_start_ms=active_start_ms,
            active_end_ms=active_end_eff,
            min_anchor_gap_ms=min_anchor_gap_ms,
        )
    anchor_shift_ms = _env_float("UTOA_BOUNDARY_ANCHOR_SHIFT_MS", 0.0)
    if abs(float(anchor_shift_ms)) > 1e-6:
        anchor_timeline = _shift_anchor_timeline(
            timeline=anchor_timeline,
            shift_ms=float(anchor_shift_ms),
            active_start_ms=float(active_start_ms),
            active_end_ms=float(active_end_eff),
        )
    by_kind: dict[str, list[BoundaryCandidate]] = defaultdict(list)
    for cand in candidates:
        by_kind[cand.kind].append(cand)
    for bucket in by_kind.values():
        bucket.sort(key=lambda item: item.time_ms)

    sorted_specs = sorted(row_specs, key=lambda item: int(item.line_index))
    resolved_model_quality = _resolve_model_quality(candidates, fallback=model_quality)
    resolved_audio_reliability = _resolve_audio_reliability(candidates, fallback=audio_reliability)

    row_contexts: list[dict[str, float | str | int]] = []
    row_options: list[list[dict[str, float | str | bool | int]]] = []
    for spec in sorted_specs:
        role = normalize_role(spec.role)
        slot = min(max(0, int(spec.slot_index)), len(anchor_timeline) - 1)
        left_anchor = anchor_timeline[min(max(0, spec.slot_index), len(anchor_timeline) - 1)]
        has_right_anchor = spec.slot_index + 1 < len(anchor_timeline)
        right_anchor = (
            anchor_timeline[min(max(0, spec.slot_index + 1), len(anchor_timeline) - 1)]
            if has_right_anchor
            else float(active_end_eff)
        )
        options = _build_row_options(
            role=role,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            by_kind=by_kind,
            active_start_ms=float(active_start_ms),
            active_end_ms=float(active_end_eff),
            model_quality=resolved_model_quality,
            audio_reliability=resolved_audio_reliability,
        )
        row_options.append(options)
        row_contexts.append(
            {
                "role": role,
                "slot": int(slot),
                "left_anchor": float(left_anchor),
                "right_anchor": float(right_anchor),
            }
        )

    chosen_indices = _decode_global_path(row_options=row_options, row_contexts=row_contexts)
    rows: list[DecodedOtoRow] = []
    fallback_count = 0
    for idx, spec in enumerate(sorted_specs):
        ctx = row_contexts[idx]
        opt = row_options[idx][chosen_indices[idx]]
        role = str(ctx["role"])
        left_anchor = float(ctx["left_anchor"])
        right_anchor = float(ctx["right_anchor"])
        selected_time = _clamp_selected_time(
            role=role,
            selected_time=float(opt["time_ms"]),
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            active_start_ms=float(active_start_ms),
            active_end_ms=float(active_end_eff),
        )
        fallback_used = bool(opt.get("fallback", False))
        quality_score = _clamp01(float(opt.get("quality", 0.0)))
        anchors = build_absolute_anchors_for_role(
            role=role,
            center_ms=selected_time,
            duration_ms=float(duration_ms),
            left_anchor_ms=left_anchor,
            # Always pass right anchor. For single-slot rows this becomes
            # active_end and still provides a hard right-edge clamp.
            right_anchor_ms=right_anchor,
            active_start_ms=float(active_start_ms),
            active_end_ms=float(active_end_eff),
            confidence=float(_clamp01(float(opt.get("score", 0.0)))),
            reason=str(opt.get("source", "fallback:anchor_pair")),
        )
        if fallback_used:
            fallback_count += 1
        rows.append(
            DecodedOtoRow(
                spec=spec,
                anchors=anchors,
                selected_time_ms=float(selected_time),
                fallback_used=bool(fallback_used),
                reason=str(opt.get("source", "fallback:anchor_pair")),
                quality_score=quality_score,
            )
        )
    return BoundaryDecodeResult(
        wav_path=wav_path,
        duration_ms=float(duration_ms),
        rows=rows,
        anchor_timeline_ms=anchor_timeline,
        fallback_count=int(fallback_count),
    )


def _build_anchor_timeline(
    *,
    slot_count: int,
    duration_ms: float,
    candidates: list[BoundaryCandidate],
    active_start_ms: float,
    active_end_ms: float | None,
    min_anchor_gap_ms: float,
) -> list[float]:
    active_end = float(active_end_ms) if active_end_ms is not None else float(duration_ms)
    active_start = max(0.0, min(active_end, float(active_start_ms)))
    duration = max(active_start + 1.0, active_end)
    onset_like = [row for row in candidates if row.kind in {"syllable_onset", "vowel_start", "consonant_onset"}]
    onset_like.sort(key=lambda item: item.time_ms)
    timeline: list[float] = []
    for idx in range(max(1, int(slot_count))):
        prior = active_start if slot_count <= 1 else active_start + ((duration - active_start) * float(idx) / float(max(1, slot_count - 1)))
        picked = _pick_closest(onset_like, prior, window_ms=180.0)
        timeline.append(float(prior if picked is None else picked.time_ms))
    for idx in range(1, len(timeline)):
        min_allowed = timeline[idx - 1] + float(min_anchor_gap_ms)
        if timeline[idx] < min_allowed:
            timeline[idx] = min_allowed
    if timeline:
        timeline[-1] = min(timeline[-1], duration)
    return timeline


def _shift_anchor_timeline(
    *,
    timeline: list[float],
    shift_ms: float,
    active_start_ms: float,
    active_end_ms: float,
) -> list[float]:
    if not timeline:
        return timeline
    lo = max(0.0, min(float(active_start_ms), float(active_end_ms)))
    hi = max(lo + 1.0, float(active_end_ms))
    shifted = [_clamp(float(t) + float(shift_ms), lo, hi) for t in timeline]
    for idx in range(1, len(shifted)):
        if shifted[idx] < shifted[idx - 1]:
            shifted[idx] = shifted[idx - 1]
    return shifted


def _pick_closest(candidates: list[BoundaryCandidate], target_ms: float, *, window_ms: float) -> BoundaryCandidate | None:
    best = None
    best_cost = None
    for cand in candidates:
        dist = abs(float(cand.time_ms) - float(target_ms))
        if dist > float(window_ms):
            continue
        cost = dist - (float(cand.score) * 32.0)
        if best is None or cost < float(best_cost):
            best = cand
            best_cost = cost
    return best


def _build_row_options(
    *,
    role: str,
    left_anchor: float,
    right_anchor: float,
    by_kind: dict[str, list[BoundaryCandidate]],
    active_start_ms: float,
    active_end_ms: float,
    model_quality: float,
    audio_reliability: float,
) -> list[dict[str, float | str | bool | int]]:
    labels = role_label_preferences(role)
    role_key = normalize_role(role)
    if role_key in TRANSITION_ROLES:
        lo = float(left_anchor) - 28.0
        hi = float(right_anchor) + 42.0
    else:
        lo = float(left_anchor) - 90.0
        hi = float(right_anchor) + 90.0
    options: list[dict[str, float | str | bool | int]] = []
    span = max(1.0, float(right_anchor) - float(left_anchor))
    if role_key == "vc":
        vc_target_ratio = _clamp(_env_float("UTOA_BOUNDARY_VC_TARGET_RATIO", 0.63), 0.45, 0.90)
        target = float(left_anchor) + (vc_target_ratio * span)
    else:
        target = 0.5 * (left_anchor + right_anchor) if role_key in TRANSITION_ROLES else left_anchor
    for label in labels:
        for idx, cand in enumerate(by_kind.get(label, [])):
            t = float(cand.time_ms)
            if t < lo or t > hi:
                continue
            clamped_time = _clamp_selected_time(
                role=role,
                selected_time=t,
                left_anchor=left_anchor,
                right_anchor=right_anchor,
                active_start_ms=active_start_ms,
                active_end_ms=active_end_ms,
            )
            quality_hint = _infer_candidate_quality(
                candidate=cand,
                model_quality=model_quality,
                audio_reliability=audio_reliability,
            )
            emission = _option_emission_score(
                role=role,
                time_ms=clamped_time,
                target_ms=target,
                base_score=float(cand.score),
                fallback=False,
                quality_hint=quality_hint,
                source=str(cand.source),
                kind=str(cand.kind),
                left_anchor=left_anchor,
                right_anchor=right_anchor,
            )
            options.append(
                {
                    "time_ms": float(clamped_time),
                    "score": _clamp01(float(cand.score)),
                    "quality": _clamp01(float(quality_hint)),
                    "source": str(cand.source),
                    "kind": str(cand.kind),
                    "fallback": False,
                    "row_emission": float(emission),
                    "candidate_index": int(idx),
                }
            )

    if role_key == "vc":
        vc_target_ratio = _clamp(_env_float("UTOA_BOUNDARY_VC_TARGET_RATIO", 0.63), 0.45, 0.90)
        fallback_time = float(left_anchor) + (vc_target_ratio * span)
    else:
        fallback_time = 0.5 * (left_anchor + right_anchor) if role_key in TRANSITION_ROLES else left_anchor
    fallback_time = _clamp_selected_time(
        role=role,
        selected_time=float(fallback_time),
        left_anchor=float(left_anchor),
        right_anchor=float(right_anchor),
        active_start_ms=float(active_start_ms),
        active_end_ms=float(active_end_ms),
    )
    fallback_quality = 0.5 * (_clamp01(model_quality) + _clamp01(audio_reliability))
    fallback_emission = _option_emission_score(
        role=role,
        time_ms=fallback_time,
        target_ms=target,
        base_score=0.0,
        fallback=True,
        quality_hint=fallback_quality,
        source="fallback:anchor_pair",
        kind="fallback",
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )
    options.append(
        {
            "time_ms": float(fallback_time),
            "score": 0.0,
            "quality": _clamp01(float(fallback_quality)),
            "source": "fallback:anchor_pair",
            "kind": "fallback",
            "fallback": True,
            "row_emission": float(fallback_emission),
            "candidate_index": -1,
        }
    )
    return _dedupe_options(options)


def _decode_global_path(
    *,
    row_options: list[list[dict[str, float | str | bool | int]]],
    row_contexts: list[dict[str, float | str | int]],
) -> list[int]:
    if not row_options:
        return []
    neg_inf = -1.0e18
    best: list[list[float]] = [[neg_inf for _ in opts] for opts in row_options]
    back: list[list[int]] = [[-1 for _ in opts] for opts in row_options]
    for j, opt in enumerate(row_options[0]):
        best[0][j] = float(opt.get("row_emission", 0.0))
    for i in range(1, len(row_options)):
        cur_opts = row_options[i]
        prev_opts = row_options[i - 1]
        cur_role = str(row_contexts[i].get("role", "other"))
        for j, cur in enumerate(cur_opts):
            cur_t = float(cur.get("time_ms", 0.0))
            cur_emission = float(cur.get("row_emission", 0.0))
            best_score = neg_inf
            best_prev = -1
            for k, prv in enumerate(prev_opts):
                prev_score = float(best[i - 1][k])
                if prev_score <= neg_inf / 2.0:
                    continue
                prev_t = float(prv.get("time_ms", 0.0))
                trans = _transition_score(
                    prev_time_ms=prev_t,
                    cur_time_ms=cur_t,
                    cur_role=cur_role,
                    prev_option=prv,
                    cur_option=cur,
                )
                if trans <= -1.0e12:
                    continue
                total = prev_score + trans + cur_emission
                if total > best_score:
                    best_score = total
                    best_prev = k
            if best_prev < 0:
                best_score = cur_emission - 5.0
            best[i][j] = best_score
            back[i][j] = best_prev

    last_row = len(row_options) - 1
    end_idx = max(range(len(row_options[last_row])), key=lambda idx: best[last_row][idx])
    path = [0 for _ in row_options]
    path[last_row] = int(end_idx)
    for i in range(last_row, 0, -1):
        prev_idx = int(back[i][path[i]])
        if prev_idx < 0:
            prev_idx = max(range(len(row_options[i - 1])), key=lambda idx: best[i - 1][idx])
        path[i - 1] = prev_idx
    return path


def _transition_score(
    *,
    prev_time_ms: float,
    cur_time_ms: float,
    cur_role: str,
    prev_option: dict[str, float | str | bool | int],
    cur_option: dict[str, float | str | bool | int],
) -> float:
    if float(cur_time_ms) + 1e-6 < float(prev_time_ms):
        return -1.0e15
    dt = float(cur_time_ms) - float(prev_time_ms)
    role_key = normalize_role(cur_role)
    min_step = _min_monotonic_step_ms(cur_role)
    penalty = 0.0
    if dt < min_step:
        penalty -= (min_step - dt) * 0.42
    if role_key == "vc":
        vc_min_dt = _clamp(_env_float("UTOA_BOUNDARY_VC_MIN_DT_FROM_PREV_MS", 0.0), 0.0, 220.0)
        if vc_min_dt > 1e-6 and dt < vc_min_dt:
            penalty -= (vc_min_dt - dt) * 0.55
    if dt > 460.0:
        penalty -= (dt - 460.0) * 0.01
    prev_fallback = bool(prev_option.get("fallback", False))
    cur_fallback = bool(cur_option.get("fallback", False))
    if prev_fallback and cur_fallback:
        penalty -= 0.18
    prev_kind = str(prev_option.get("kind", ""))
    cur_kind = str(cur_option.get("kind", ""))
    if prev_kind == cur_kind and dt < 3.0:
        penalty -= 0.28
    return penalty


def _option_emission_score(
    *,
    role: str,
    time_ms: float,
    target_ms: float,
    base_score: float,
    fallback: bool,
    quality_hint: float,
    source: str,
    kind: str,
    left_anchor: float,
    right_anchor: float,
) -> float:
    role_key = normalize_role(role)
    span = max(1.0, float(right_anchor) - float(left_anchor))
    dist = abs(float(time_ms) - float(target_ms))
    dist_scale = max(18.0, 0.50 * span) if role_key in TRANSITION_ROLES else max(22.0, 0.70 * span)
    conf_term = float(_clamp01(base_score)) * 2.65
    quality_term = (float(_clamp01(quality_hint)) - 0.5) * 0.95
    dist_term = -float(dist) / float(dist_scale)
    source_term = 0.0
    text = str(source or "")
    if text.startswith("model"):
        source_term += 0.12
    elif text.startswith("audio"):
        source_term += 0.06
    elif "merged" in text:
        source_term += 0.04
    vc_shape_term = 0.0
    if role_key == "vc":
        ratio = (float(time_ms) - float(left_anchor)) / float(span)
        early_start_ratio = _clamp(_env_float("UTOA_BOUNDARY_VC_EARLY_PENALTY_START_RATIO", 0.56), 0.35, 0.90)
        early_weight = _clamp(_env_float("UTOA_BOUNDARY_VC_EARLY_PENALTY_WEIGHT", 0.95), 0.0, 5.0)
        if ratio < early_start_ratio:
            vc_shape_term -= (early_start_ratio - ratio) * early_weight
        vc_vowel_end_penalty = _clamp(_env_float("UTOA_BOUNDARY_VC_VOWEL_END_PENALTY", 0.22), 0.0, 2.0)
        vc_transition_bonus = _clamp(_env_float("UTOA_BOUNDARY_VC_TRANSITION_KIND_BONUS", 0.10), 0.0, 2.0)
        kind_key = str(kind or "")
        if kind_key == "vowel_end":
            vc_shape_term -= vc_vowel_end_penalty
        elif kind_key in {"transition_peak", "next_onset"}:
            vc_shape_term += vc_transition_bonus
    fallback_penalty = -0.95 if fallback else 0.0
    return conf_term + quality_term + dist_term + source_term + vc_shape_term + fallback_penalty


def _infer_candidate_quality(*, candidate: BoundaryCandidate, model_quality: float, audio_reliability: float) -> float:
    src = str(candidate.source or "")
    if src.startswith("model"):
        return _clamp01((0.70 * _clamp01(float(candidate.score))) + (0.30 * _clamp01(model_quality)))
    if src.startswith("audio"):
        return _clamp01((0.65 * _clamp01(float(candidate.score))) + (0.35 * _clamp01(audio_reliability)))
    blend = 0.5 * (_clamp01(model_quality) + _clamp01(audio_reliability))
    return _clamp01((0.70 * _clamp01(float(candidate.score))) + (0.30 * blend))


def _dedupe_options(options: list[dict[str, float | str | bool | int]]) -> list[dict[str, float | str | bool | int]]:
    best_by_bucket: dict[int, dict[str, float | str | bool | int]] = {}
    for opt in options:
        bucket = int(round(float(opt.get("time_ms", 0.0)) * 2.0))  # 0.5ms bucket
        prev = best_by_bucket.get(bucket)
        cur_score = float(opt.get("row_emission", 0.0))
        if prev is None or cur_score > float(prev.get("row_emission", 0.0)):
            best_by_bucket[bucket] = opt
    out = list(best_by_bucket.values())
    out.sort(key=lambda item: float(item.get("time_ms", 0.0)))
    return out


def _resolve_model_quality(candidates: list[BoundaryCandidate], *, fallback: float | None) -> float:
    if fallback is not None:
        return _clamp01(float(fallback))
    vals = [float(c.score) for c in candidates if str(c.source).startswith("model")]
    if not vals:
        return 0.50
    return _clamp01(float(mean(vals)))


def _resolve_audio_reliability(candidates: list[BoundaryCandidate], *, fallback: float | None) -> float:
    if fallback is not None:
        return _clamp01(float(fallback))
    vals = [float(c.score) for c in candidates if str(c.source).startswith("audio")]
    if not vals:
        return 0.50
    return _clamp01(float(mean(vals)))


def _min_monotonic_step_ms(role: str) -> float:
    base = _env_float("UTOA_BOUNDARY_ROW_MONOTONIC_STEP_MS", 1.2)
    role_key = normalize_role(role)
    if role_key in TRANSITION_ROLES:
        return _clamp(base, 0.0, 1.0)
    return _clamp(base, 0.0, 6.0)


def _clamp_selected_time(
    *,
    role: str,
    selected_time: float,
    left_anchor: float,
    right_anchor: float,
    active_start_ms: float,
    active_end_ms: float,
) -> float:
    lo_active = max(0.0, min(float(active_start_ms), float(active_end_ms)))
    hi_active = max(lo_active + 1.0, float(active_end_ms))
    lo = lo_active
    hi = hi_active
    role_key = normalize_role(role)
    span = max(1.0, float(right_anchor) - float(left_anchor))
    if role_key == "vc":
        vc_lo_ratio = _clamp(_env_float("UTOA_BOUNDARY_VC_CLAMP_LEFT_RATIO", 0.38), 0.20, 0.75)
        vc_hi_margin = _clamp(_env_float("UTOA_BOUNDARY_VC_CLAMP_RIGHT_MARGIN_MS", 18.0), 6.0, 64.0)
        lo = max(lo_active, float(left_anchor) + (vc_lo_ratio * span))
        hi = min(hi_active, float(right_anchor) - vc_hi_margin)
    elif role_key in TRANSITION_ROLES:
        lo = max(lo_active, float(left_anchor) + 0.28 * span)
        hi = min(hi_active, float(right_anchor) - 12.0)
    elif role_key in {"cv", "v", "-cv", "other"}:
        lo = max(lo_active, float(left_anchor) - 12.0)
        hi = min(hi_active, float(left_anchor) + 46.0)
    if hi <= lo:
        lo = lo_active
        hi = hi_active
    return max(lo, min(hi, float(selected_time)))


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


__all__ = ["decode_wav_rows"]
