from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.alignment.hsmm_decoder import (
    HSMMDecodeResult,
    HSMMStateSpec,
    decode_segmental_hsmm,
    decode_segmental_hsmm_coarse_to_refine,
)

from .row_plan import FilenameSlot
from .types import FramePosterior


HSMM_ADAPTER_SCHEMA_VERSION = 1
_SONORANT_PHONES = frozenset({"m", "n", "ng", "ny", "r", "l", "w", "y", "j"})


@dataclass(frozen=True)
class FilenameHSMMDecode:
    result: HSMMDecodeResult
    events: tuple[dict[str, object], ...]
    states: tuple[HSMMStateSpec, ...]
    frame_scores: Mapping[str, tuple[float, ...]]
    state_interval_priors: Mapping[str, tuple[float, ...]]
    diagnostics: Mapping[str, object]
    schema_version: int = HSMM_ADAPTER_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return bool(self.result.ok)


def build_hsmm_states_for_slots(
    slots: Sequence[FilenameSlot],
    *,
    duration_ms: float = 0.0,
    enable_sonorant_hold: bool = False,
) -> tuple[HSMMStateSpec, ...]:
    if not slots:
        return ()
    state_count = sum(
        1 + _onset_state_count(slot.onset_phones, enable_sonorant_hold=enable_sonorant_hold)
        for slot in slots
    )
    base_duration = float(duration_ms or 0.0) / float(max(1, state_count)) if duration_ms > 0.0 else 90.0
    states: list[HSMMStateSpec] = []
    for slot in slots:
        sonorant_onset = bool(enable_sonorant_hold) and _phones_include_sonorant(slot.onset_phones)
        if slot.onset_phones:
            cons_mode = _clamp(
                base_duration * (0.48 if sonorant_onset else 0.65),
                22.0 if sonorant_onset else 28.0,
                95.0,
            )
            states.append(
                HSMMStateSpec(
                    state_id=_state_id(slot, "onset"),
                    state_type="consonant",
                    min_duration_ms=max(8.0, min(24.0, cons_mode * 0.45)),
                    max_duration_ms=max(48.0, min(240.0, cons_mode * 2.8)),
                    mode_duration_ms=cons_mode,
                    duration_sigma_ms=max(12.0, cons_mode * 0.55),
                )
            )
            if sonorant_onset:
                hold_mode = _clamp(base_duration * 0.45, 24.0, 105.0)
                states.append(
                    HSMMStateSpec(
                        state_id=_state_id(slot, "onset_hold"),
                        state_type="consonant",
                        min_duration_ms=max(12.0, min(36.0, hold_mode * 0.42)),
                        max_duration_ms=max(54.0, min(220.0, hold_mode * 2.6)),
                        mode_duration_ms=hold_mode,
                        duration_sigma_ms=max(16.0, hold_mode * 0.62),
                    )
                )
        vowel_scale = 1.12 if sonorant_onset else (1.20 if slot.onset_phones else 1.55)
        vowel_mode = _clamp(base_duration * vowel_scale, 48.0, 240.0)
        states.append(
            HSMMStateSpec(
                state_id=_state_id(slot, "vowel"),
                state_type="vowel",
                min_duration_ms=max(18.0, min(52.0, vowel_mode * 0.40)),
                max_duration_ms=max(80.0, min(900.0, vowel_mode * 3.4)),
                mode_duration_ms=vowel_mode,
                duration_sigma_ms=max(20.0, vowel_mode * 0.62),
                # Held vowels routinely run far longer than the nominal mode.
                # Penalising the over-length side symmetrically caused the vowel
                # segment to be trimmed from the front, pushing the C->V boundary
                # (the OTO offset anchor) late. Keep the over-length penalty weak
                # so the vowel locks to its acoustic onset and only the tail moves.
                over_duration_penalty_scale=0.12,
            )
        )
    return tuple(states)


def build_hsmm_frame_scores(
    posterior: FramePosterior,
    states: Sequence[HSMMStateSpec],
    *,
    slots: Sequence[FilenameSlot] = (),
    state_interval_priors: Mapping[str, Sequence[float]] | None = None,
    use_vowel_start_score: bool = False,
) -> dict[str, tuple[float, ...]]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    frame_count = int(times.size)
    if frame_count <= 0:
        return {state.state_id: () for state in states}

    silence = _track(posterior.class_probs, "silence", frame_count)
    consonant = _track(posterior.class_probs, "consonant", frame_count)
    vowel = _track(posterior.class_probs, "vowel", frame_count)
    transition = _first_track(
        frame_count,
        posterior.acoustic_scores,
        posterior.event_scores,
        keys=("transition_likelihood", "cv_boundary", "phone_change", "flux_likelihood"),
    )
    vowel_start = _first_track(
        frame_count,
        posterior.event_scores,
        posterior.acoustic_scores,
        keys=("cv_boundary", "vowel_boundary_likelihood", "phone_change", "transition_likelihood"),
    )
    onset = _first_track(
        frame_count,
        posterior.acoustic_scores,
        posterior.event_scores,
        keys=("onset_strength", "sonorant_onset_likelihood", "phone_change", "cv_boundary"),
    )
    voicing = _first_track(frame_count, posterior.acoustic_scores, keys=("voicing", "world_voicing"))
    nucleus = _first_track(
        frame_count,
        posterior.acoustic_scores,
        posterior.event_scores,
        keys=("nucleus_likelihood", "world_nucleus", "vowel_nucleus"),
    )
    active = np.clip(1.0 - silence, 0.0, 1.0)
    consonant_score = np.clip(
        (0.44 * consonant) + (0.24 * transition) + (0.18 * onset) + (0.14 * active),
        0.0,
        1.0,
    )
    sonorant_consonant_score = np.clip(
        (0.34 * onset) + (0.24 * transition) + (0.22 * voicing) + (0.20 * active),
        0.0,
        1.0,
    )
    sonorant_hold_score = np.clip(
        (0.30 * voicing) + (0.26 * active) + (0.22 * consonant) + (0.22 * (1.0 - transition)),
        0.0,
        1.0,
    )
    vowel_score = np.clip(
        (0.50 * vowel) + (0.22 * voicing) + (0.18 * nucleus) + (0.10 * active),
        0.0,
        1.0,
    )

    out: dict[str, tuple[float, ...]] = {}
    duration = _posterior_duration_ms(times)
    order_count = max(1, len(states))
    sonorant_onset_state_ids = {
        _state_id(slot, "onset")
        for slot in slots
        if slot.onset_phones and _phones_include_sonorant(slot.onset_phones)
    }
    sonorant_hold_state_ids = {
        _state_id(slot, "onset_hold")
        for slot in slots
        if slot.onset_phones and _phones_include_sonorant(slot.onset_phones)
    }
    for order, state in enumerate(states):
        if state.state_type == "consonant" and state.state_id in sonorant_onset_state_ids:
            base = sonorant_consonant_score
        elif state.state_type == "consonant" and state.state_id in sonorant_hold_state_ids:
            base = sonorant_hold_score
        else:
            base = consonant_score if state.state_type == "consonant" else vowel_score
        prior = _position_prior(times, expected_ms=(order + 0.5) * duration / float(order_count), sigma_ms=max(40.0, duration / float(order_count + 1)))
        is_vowel = state.state_type == "vowel"
        event_prior = _track(state_interval_priors or {}, state.state_id, frame_count)
        start_prior = (
            _state_start_interval_track(times, vowel_start, state.mode_duration_ms)
            if bool(use_vowel_start_score) and state.state_type == "vowel"
            else np.zeros((frame_count,), dtype=np.float32)
        )

        # The uniform-position prior biases each state toward its expected slot.
        # For vowels that pull is harmful: it drags the segment toward the file
        # centre and trims the acoustic onset, shifting the C->V boundary late.
        # Suppress the positional pull on vowels (handing the freed weight back to
        # the acoustic base score) so the vowel locks to its real onset -- but only
        # when no stronger locating signal is present. When an explicit runtime
        # event prior is supplied it should keep steering the segment, so leave the
        # blend untouched there.
        # TUNED: reduced position-prior pull (0.18→0.11 for vowels, 1.0→0.60 general)
        # to curb timing dispersion caused by the uniform-position prior dragging
        # segments away from their acoustic boundaries.
        pos_w = 0.11 if (is_vowel and not np.any(event_prior)) else 0.60

        def _blend(base_coef: float, prior_coef: float, *extra: tuple[float, np.ndarray]) -> np.ndarray:
            scaled_prior = prior_coef * pos_w
            effective_base = base_coef + (prior_coef - scaled_prior)
            total = (effective_base * base) + (scaled_prior * prior)
            for coef, track in extra:
                total = total + (coef * track)
            return np.clip(total, 0.0, 1.0)

        if np.any(event_prior):
            if np.any(start_prior):
                scored = _blend(0.40, 0.06, (0.34, event_prior), (0.20, start_prior))
            else:
                scored = _blend(0.52, 0.10, (0.38, event_prior))
        elif np.any(start_prior):
            # Lean hard on the (trained) cv_boundary track so vowel onsets follow
            # real acoustic boundaries instead of the duration/position prior; the
            # HSMM ordering interpolates unsegmentable identical-vowel runs.
            scored = _blend(0.56, 0.08, (0.36, start_prior))
        else:
            scored = _blend(0.86, 0.14)
        out[state.state_id] = tuple(float(value) for value in scored)
    return out


def _find_ordered_nucleus_peaks(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    min_separation_ms: float,
) -> list[tuple[float, float]]:
    """Return up to ``count`` strong, well-separated nucleus peaks in time order.

    Greedy by height with a minimum time separation so two sub-peaks inside one
    vowel are not both selected. Returns an empty list unless at least ``count``
    confident peaks are found, so callers can fall back to prior-free behaviour.
    """
    if times.size == 0 or scores.size != times.size or count <= 0:
        return []
    peak_value = float(np.max(scores))
    if peak_value <= 0.0:
        return []
    threshold = max(0.22, peak_value * 0.45)
    candidates: list[int] = []
    for idx in range(scores.size):
        value = float(scores[idx])
        if value < threshold:
            continue
        if idx > 0 and scores[idx - 1] > value:
            continue
        if idx < scores.size - 1 and scores[idx + 1] > value:
            continue
        candidates.append(idx)
    if len(candidates) < count:
        return []
    separation = max(1.0, float(min_separation_ms))
    chosen: list[int] = []
    # Greedy by height; ``chosen`` therefore ends up in descending-height order.
    for idx in sorted(candidates, key=lambda i: -float(scores[i])):
        if all(abs(float(times[idx]) - float(times[j])) >= separation for j in chosen):
            chosen.append(idx)
    if len(chosen) < count:
        return []
    strongest = chosen[:count]  # keep the ``count`` strongest separated peaks
    strongest.sort(key=lambda i: float(times[i]))
    return [(float(times[i]), float(scores[i])) for i in strongest]


def _auto_vowel_nucleus_event_priors(
    posterior: FramePosterior,
    slots: Sequence[FilenameSlot],
    *,
    duration_ms: float,
    existing_priors: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    if not slots:
        return []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    frame_count = int(times.size)
    if frame_count <= 0:
        return []
    nucleus = _first_track(
        frame_count,
        posterior.acoustic_scores,
        posterior.event_scores,
        keys=("nucleus_likelihood", "world_nucleus", "vowel_nucleus"),
    )
    if not np.any(nucleus):
        return []
    # Vowel slots already covered by a caller-supplied nucleus prior are left to
    # that stronger evidence; only fill the rest.
    covered: set[int] = set()
    for event in existing_priors:
        if str(event.get("label", "")) == "vowel_nucleus":
            idx = _int_or_none(event.get("expected_phone_index"))
            if idx is not None:
                covered.add(int(idx))
    vowel_slots = list(slots)
    vowel_count = len(vowel_slots)
    if vowel_count <= 0:
        return []
    nominal_ms = float(duration_ms) / float(vowel_count) if duration_ms > 0.0 else 0.0
    # Separation only needs to merge sub-peaks inside a single vowel; keep it small
    # enough that genuinely distinct (possibly short/uneven) vowels are not merged.
    min_separation_ms = max(60.0, min(130.0, nominal_ms * 0.45))
    peaks = _find_ordered_nucleus_peaks(
        times, nucleus, count=vowel_count, min_separation_ms=min_separation_ms
    )
    if len(peaks) != vowel_count:
        return []
    priors: list[dict[str, object]] = []
    for slot, (peak_ms, peak_height) in zip(vowel_slots, peaks):
        if int(slot.vowel_phone_index) in covered:
            continue
        priors.append(
            {
                "label": "vowel_nucleus",
                "expected_phone_index": int(slot.vowel_phone_index),
                "selected_time_ms": float(peak_ms),
                "score": float(max(0.0, min(1.0, peak_height))),
                "source": "acoustic_nucleus_auto",
            }
        )
    return priors


_EVENT_SNAP_WINDOW_MS = 45.0
_EVENT_SNAP_MIN_GAP_MS = 15.0
_EVENT_SNAP_TRACK_KEYS = {
    "cv_boundary": ("cv_boundary", "vowel_boundary_likelihood", "transition_likelihood"),
    "vv_boundary": ("cv_boundary", "vowel_boundary_likelihood", "transition_likelihood"),
    "vowel_nucleus": ("nucleus_likelihood", "world_nucleus", "vowel_nucleus"),
    "phone_change": ("phone_change", "onset_strength", "sonorant_onset_likelihood"),
}


def _snap_events_to_posterior_tracks(
    events: Sequence[Mapping[str, object]],
    posterior: FramePosterior,
    times: np.ndarray,
    *,
    window_ms: float = _EVENT_SNAP_WINDOW_MS,
    min_confidence: float = 0.28,
) -> tuple[dict[str, object], ...]:
    frame_count = int(times.size)
    if frame_count == 0 or not events:
        return tuple(dict(event) for event in events)
    track_cache: dict[str, np.ndarray] = {}

    def _resolve_track(label: str) -> np.ndarray | None:
        if label in track_cache:
            return track_cache[label]
        keys = _EVENT_SNAP_TRACK_KEYS.get(label)
        track = (
            _first_track(frame_count, posterior.event_scores, posterior.acoustic_scores, keys=keys)
            if keys
            else None
        )
        track_cache[label] = track if (track is not None and np.any(track)) else None
        return track_cache[label]

    ordered = sorted(
        (dict(event) for event in events),
        key=lambda event: float(event.get("selected_time_ms", 0.0) or 0.0),
    )
    prev_time = -1e9
    for idx, event in enumerate(ordered):
        label = str(event.get("label", ""))
        track = _resolve_track(label)
        cur_time = float(event.get("selected_time_ms", 0.0) or 0.0)
        next_time = (
            float(ordered[idx + 1].get("selected_time_ms", 0.0) or 0.0)
            if idx + 1 < len(ordered)
            else 1e9
        )
        lower = prev_time + _EVENT_SNAP_MIN_GAP_MS
        upper = next_time - _EVENT_SNAP_MIN_GAP_MS
        if track is not None and upper > lower:
            lo = max(cur_time - window_ms, lower)
            hi = min(cur_time + window_ms, upper)
            mask = (times >= lo) & (times <= hi)
            if np.any(mask):
                window_idx = np.flatnonzero(mask)
                peak_local = window_idx[int(np.argmax(track[window_idx]))]
                if float(track[peak_local]) >= float(min_confidence):
                    snapped = float(times[peak_local])
                    if lower <= snapped <= upper:
                        event["selected_time_ms"] = snapped
                        event["frame_index"] = int(peak_local)
                        cur_time = snapped
        prev_time = cur_time
    return tuple(ordered)


def _voiced_end_ms(posterior: FramePosterior, times: np.ndarray, *, default: float) -> float:
    frame_count = int(times.size)
    if frame_count == 0:
        return float(default)
    silence = _track(posterior.class_probs, "silence", frame_count)
    active = np.clip(1.0 - silence, 0.0, 1.0)
    if not np.any(active > 0.5):
        return float(default)
    idx = np.flatnonzero(active > 0.5)
    return float(times[int(idx[-1])])


def _structured_vowel_onset_event_priors(
    posterior: FramePosterior,
    slots: Sequence[FilenameSlot],
    *,
    duration_ms: float,
    existing_priors: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Pin each vowel onset using the filename structure + the cv_boundary track.

    The filename tells us which adjacent vowels are *identical and unseparated*
    (no acoustic boundary exists, e.g. a continuous "a a") versus *distinct*
    (a real C->V or V->V edge the trained cv_boundary head fires on). We expect
    exactly ``1 + (#distinct boundaries)`` clean peaks; when that holds we assign
    peaks to the distinct boundaries and evenly interpolate the identical runs.
    This fixes VV-chain mismapping that duration priors alone smear.
    """
    if not slots:
        return []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    frame_count = int(times.size)
    if frame_count == 0:
        return []
    track = _first_track(
        frame_count, posterior.event_scores, posterior.acoustic_scores,
        keys=("cv_boundary", "vowel_boundary_likelihood"),
    )
    if not np.any(track):
        return []
    # (#1) reinforce distinct-vowel sensitivity with a formant/spectral transition
    # signal so a real boundary is not missed when the cv_boundary head is weak.
    formant = _first_track(
        frame_count, posterior.acoustic_scores,
        keys=("world_transition", "spectral_shape_delta_likelihood", "transition_likelihood"),
    )
    if np.any(formant):
        track = np.clip(np.maximum(track, 0.55 * formant), 0.0, 1.0)

    interp = [False] * len(slots)
    for k in range(1, len(slots)):
        cur, prev = slots[k], slots[k - 1]
        if (not cur.onset_phones) and (not prev.coda_phones) and str(cur.vowel_phone) == str(prev.vowel_phone):
            interp[k] = True
    distinct = sum(1 for k in range(1, len(slots)) if not interp[k])
    expected_peaks = 1 + distinct
    if expected_peaks < 1 or expected_peaks > len(slots):
        return []
    nominal = float(duration_ms) / float(max(1, len(slots))) if duration_ms > 0.0 else 0.0
    separation = max(70.0, min(170.0, nominal * 0.5))
    peaks = _find_ordered_nucleus_peaks(times, track, count=expected_peaks, min_separation_ms=separation)
    if len(peaks) != expected_peaks:
        return []
    peak_times = [p[0] for p in peaks]

    onset: list[float | None] = [None] * len(slots)
    onset[0] = peak_times[0]
    pi = 1
    for k in range(1, len(slots)):
        if interp[k]:
            onset[k] = None
        else:
            onset[k] = peak_times[pi]
            pi += 1
    voiced_end = _voiced_end_ms(posterior, times, default=float(duration_ms))
    k = 1
    while k < len(slots):
        if onset[k] is None:
            j = k
            while j < len(slots) and onset[j] is None:
                j += 1
            left = float(onset[k - 1])
            right = float(onset[j]) if j < len(slots) and onset[j] is not None else float(voiced_end)
            segments = j - (k - 1)
            for idx in range(k, j):
                frac = float(idx - (k - 1)) / float(max(1, segments))
                onset[idx] = left + frac * (right - left)
            k = j
        else:
            k += 1

    covered: set[tuple[str, int]] = set()
    for event in existing_priors:
        label = str(event.get("label", ""))
        if label in ("cv_boundary", "vv_boundary"):
            idx = _int_or_none(event.get("expected_phone_index"))
            if idx is not None:
                covered.add((label, int(idx)))
    out: list[dict[str, object]] = []
    for k, slot in enumerate(slots):
        if onset[k] is None:
            continue
        label = "cv_boundary" if slot.onset_phones else "vv_boundary"
        if (label, int(slot.vowel_phone_index)) in covered:
            continue
        out.append({
            "label": label,
            "expected_phone_index": int(slot.vowel_phone_index),
            "selected_time_ms": float(onset[k]),
            "score": 0.9,
            "source": "structured_onset",
        })
    return out


def decode_filename_slots_with_hsmm(
    posterior: FramePosterior,
    slots: Sequence[FilenameSlot],
    *,
    beam_width_per_state: int = 64,
    timeout_ms: int = 5000,
    event_priors: Iterable[Mapping[str, object]] = (),
    use_coarse_to_refine: bool | None = None,
    coarse_stride: int = 4,
    refine_window_ms: float = 110.0,
    language: str = "",
    enable_sonorant_hold: bool | None = None,
    use_vowel_start_score: bool | None = None,
) -> FilenameHSMMDecode:
    event_prior_items = tuple(dict(event or {}) for event in event_priors)
    times = [float(value) for value in posterior.times_ms]
    duration = _posterior_duration_ms(np.asarray(times, dtype=np.float32))
    # Auto-pin each vowel to its acoustic nucleus peak. Phone-agnostic vowel
    # scores + the duration prior otherwise collapse uneven/long vowels onto the
    # nominal mode duration, placing the offset on the wrong syllable. Injected as
    # vowel_nucleus event priors so the strong (0.30-weight) interval-prior path
    # attracts each vowel segment to its real nucleus. Only fires when the nucleus
    # track yields one cleanly separated peak per vowel; otherwise it is a no-op.
    auto_onset_priors = _structured_vowel_onset_event_priors(
        posterior, slots, duration_ms=duration, existing_priors=event_prior_items
    )
    if auto_onset_priors:
        event_prior_items = (*event_prior_items, *auto_onset_priors)
    auto_nucleus_priors = _auto_vowel_nucleus_event_priors(
        posterior, slots, duration_ms=duration, existing_priors=event_prior_items
    )
    if auto_nucleus_priors:
        event_prior_items = (*event_prior_items, *auto_nucleus_priors)
    korean_mode = _is_korean_language(language)
    coarse_enabled = korean_mode if use_coarse_to_refine is None else bool(use_coarse_to_refine)
    sonorant_hold_enabled = korean_mode if enable_sonorant_hold is None else bool(enable_sonorant_hold)
    # Locking the vowel segment to its acoustic onset (rising CV/vowel-boundary
    # edge) matters for every language, not just Korean — otherwise long held
    # vowels drift the C->V boundary late. Default it on across the board.
    vowel_start_enabled = True if use_vowel_start_score is None else bool(use_vowel_start_score)
    states = build_hsmm_states_for_slots(
        slots,
        duration_ms=duration,
        enable_sonorant_hold=sonorant_hold_enabled,
    )
    state_interval_priors = build_state_interval_priors_for_slots(
        posterior,
        slots,
        states,
        event_priors=event_prior_items,
    )
    frame_scores = build_hsmm_frame_scores(
        posterior,
        states,
        slots=slots,
        state_interval_priors=state_interval_priors,
        use_vowel_start_score=vowel_start_enabled,
    )
    decode_fn = decode_segmental_hsmm_coarse_to_refine if bool(coarse_enabled) else decode_segmental_hsmm
    # The first state may need to skip a long lead-in (breath/silence before the
    # first phone). A fixed small cap forces the first vowel into that silence and
    # shifts the whole sequence early -> wrong pronunciation. Allow the leading gap
    # up to the acoustic onset (where the recording stops being silent), bounded so
    # it never skips real voiced content.
    max_leading_gap = _resolve_max_leading_gap_ms(posterior, np.asarray(times, dtype=np.float32), duration, len(states))
    decode_kwargs = {
        "beam_width_per_state": beam_width_per_state,
        "timeout_ms": timeout_ms,
        "allow_leading_gap": True,
        "allow_trailing_gap": True,
        "leading_gap_penalty_per_ms": 0.00005,
        "max_leading_gap_ms": max_leading_gap,
        "trailing_gap_penalty_per_ms": 0.00002,
        "allow_internal_gaps": True,
        "internal_gap_penalty_per_ms": 0.0003,
        "max_internal_gap_ms": _max_internal_gap_ms(duration, len(states)),
    }
    if bool(coarse_enabled):
        decode_kwargs.update(
            {
                "coarse_stride": int(coarse_stride),
                "refine_window_ms": float(refine_window_ms),
                "fallback_to_full": True,
            }
        )
    result = decode_fn(states, times, frame_scores, **decode_kwargs)
    events = hsmm_result_to_slot_events(result, slots) if result.ok else ()
    if events:
        # Decode-level refinement: snap each decoded event to the nearby peak of
        # the posterior's own event track. The HSMM segmentation fixes ordering
        # and structure, but its duration/position priors smear the exact boundary
        # times; the acoustic event tracks localise them. Bounded window keeps the
        # global order intact, and a confidence floor means weak/flat tracks (rule
        # encoder) barely move while sharp tracks (trained model) snap hard.
        events = _snap_events_to_posterior_tracks(
            events, posterior, np.asarray(times, dtype=np.float32)
        )
    diagnostics = build_hsmm_diagnostics(
        result,
        states,
        frame_scores,
        state_interval_priors=state_interval_priors,
        event_prior_summary=summarize_hsmm_event_priors_for_slots(slots, event_prior_items),
    )
    return FilenameHSMMDecode(
        result=result,
        events=tuple(events),
        states=states,
        frame_scores=frame_scores,
        state_interval_priors=state_interval_priors,
        diagnostics=diagnostics,
    )


def _max_internal_gap_ms(duration_ms: float, state_count: int) -> float:
    if duration_ms <= 0.0 or state_count <= 0:
        return 180.0
    nominal_state_ms = float(duration_ms) / float(max(1, state_count))
    return _clamp(nominal_state_ms * 0.95, 90.0, 260.0)


def _max_leading_gap_ms(duration_ms: float, state_count: int) -> float:
    if state_count <= 2:
        return 900.0
    if duration_ms <= 0.0 or state_count <= 0:
        return 220.0
    nominal_state_ms = float(duration_ms) / float(max(1, state_count))
    return _clamp(nominal_state_ms * 0.95, 90.0, 260.0)


def _acoustic_active_onset_ms(posterior: FramePosterior, times: np.ndarray) -> float:
    """First time (ms) the recording stops being silent.

    Uses the silence class (falling edge) with the active/voiced energy as a
    fallback. Returns 0.0 when no clear silent lead-in is present.
    """
    frame_count = int(times.size)
    if frame_count == 0:
        return 0.0
    silence = _track(posterior.class_probs, "silence", frame_count)
    active = np.clip(1.0 - silence, 0.0, 1.0)
    if not np.any(active > 0.5):
        voicing = _first_track(
            frame_count, posterior.acoustic_scores, keys=("voicing", "world_voicing")
        )
        vowel = _track(posterior.class_probs, "vowel", frame_count)
        active = np.clip(np.maximum(voicing, vowel), 0.0, 1.0)
        if not np.any(active > 0.5):
            return 0.0
    idx = int(np.argmax(active > 0.5))
    return float(times[idx]) if idx < frame_count else 0.0


def _resolve_max_leading_gap_ms(
    posterior: FramePosterior,
    times: np.ndarray,
    duration_ms: float,
    state_count: int,
) -> float:
    base = _max_leading_gap_ms(duration_ms, state_count)
    if state_count <= 2 or duration_ms <= 0.0:
        return base
    onset_ms = _acoustic_active_onset_ms(posterior, times)
    if onset_ms <= 0.0:
        return base
    # Permit skipping the silent lead-in (plus a small margin) but never more than
    # a safety ceiling, so the decoder cannot dump every state into a high-scoring
    # tail and abandon the early voiced content.
    # TUNED: ceiling 0.6→0.35, margin 120→60ms — tighter bounds reduce
    # one-step syllable shift caused by excessive leading-gap allowance.
    ceiling = max(base, 0.35 * float(duration_ms))
    return float(min(ceiling, max(base, onset_ms + 60.0)))


def build_hsmm_diagnostics(
    result: HSMMDecodeResult,
    states: Sequence[HSMMStateSpec],
    frame_scores: Mapping[str, Sequence[float]],
    *,
    state_interval_priors: Mapping[str, Sequence[float]] | None = None,
    event_prior_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    frame_shift = _float_from_mapping(result.meta, "frame_shift_ms", default=5.0)
    state_specs = {state.state_id: state for state in states}
    prior_summary = _normalize_event_prior_summary(event_prior_summary)
    prior_states = dict(prior_summary.get("states", {}) or {})
    state_diagnostics: list[dict[str, object]] = []
    for decoded in result.states:
        spec = state_specs.get(decoded.state_id)
        scores = np.asarray(frame_scores.get(decoded.state_id, ()), dtype=np.float32)
        if spec is None or scores.size == 0:
            continue
        state_prior_summary = dict(prior_states.get(str(decoded.state_id), {}) or {})
        state_source_counts = _string_int_counts(state_prior_summary.get("source_counts", {}))
        state_label_counts = _string_int_counts(state_prior_summary.get("label_counts", {}))
        state_candidate_count = int(state_source_counts.get("evidence_candidate", 0))
        state_mapped_count = int(state_prior_summary.get("mapped_count", 0) or 0)
        start_frame = max(0, min(int(decoded.start_frame), int(scores.size)))
        end_frame = max(start_frame + 1, min(int(decoded.end_frame), int(scores.size)))
        selected_duration_ms = float(end_frame - start_frame) * frame_shift
        selected_mean = _segment_mean_array(scores, start_frame, end_frame)
        duration_score = _state_duration_score(spec, selected_duration_ms)
        selected_total = selected_mean + duration_score
        local = _best_local_state_segments(
            scores,
            spec,
            frame_shift_ms=frame_shift,
            selected_start_frame=start_frame,
            selected_end_frame=end_frame,
        )
        prior = np.asarray((state_interval_priors or {}).get(decoded.state_id, ()), dtype=np.float32)
        prior_selected_mean = _segment_mean_array(prior, start_frame, min(end_frame, int(prior.size))) if prior.size else 0.0
        duration_z = _state_duration_z(spec, selected_duration_ms)
        state_diagnostics.append(
            {
                "state_id": str(decoded.state_id),
                "state_type": str(decoded.state_type),
                "selected_mean_score": float(selected_mean),
                "selected_duration_score": float(duration_score),
                "selected_total_score": float(selected_total),
                "best_local_total_score": float(local["best_total_score"]),
                "second_local_total_score": float(local["second_total_score"]),
                "selected_vs_best_local_margin": float(selected_total - float(local["best_total_score"])),
                "selected_vs_second_local_margin": float(selected_total - float(local["second_total_score"])),
                "best_local_start_frame": int(local["best_start_frame"]),
                "best_local_end_frame": int(local["best_end_frame"]),
                "duration_mode_delta_ms": float(selected_duration_ms - float(spec.mode_duration_ms or 0.0)),
                "duration_z_abs": float(abs(duration_z)),
                "has_event_prior": bool(prior.size and np.any(prior > 0.0)),
                "event_prior_peak_score": float(np.max(prior)) if prior.size else 0.0,
                "event_prior_selected_mean": float(prior_selected_mean),
                "event_prior_mapped_count": state_mapped_count,
                "event_prior_source_counts": dict(state_source_counts),
                "event_prior_label_counts": dict(state_label_counts),
                "evidence_candidate_prior_count": state_candidate_count,
                "runtime_prior_count": max(0, state_mapped_count - state_candidate_count),
            }
        )
    selected_vs_best = [float(item["selected_vs_best_local_margin"]) for item in state_diagnostics]
    selected_vs_second = [float(item["selected_vs_second_local_margin"]) for item in state_diagnostics]
    duration_z_values = [float(item["duration_z_abs"]) for item in state_diagnostics]
    source_counts = _string_int_counts(prior_summary.get("source_counts", {}))
    candidate_count = int(source_counts.get("evidence_candidate", 0))
    mapped_count = int(prior_summary.get("mapped_count", 0) or 0)
    return {
        "ok": bool(result.ok),
        "state_count": int(len(states)),
        "decoded_state_count": int(len(result.states)),
        "event_prior_state_count": int(sum(1 for item in state_diagnostics if bool(item["has_event_prior"]))),
        "event_prior_input_count": int(prior_summary.get("input_count", 0) or 0),
        "event_prior_mapped_count": mapped_count,
        "event_prior_ignored_count": int(prior_summary.get("ignored_count", 0) or 0),
        "event_prior_source_counts": dict(source_counts),
        "event_prior_label_counts": _string_int_counts(prior_summary.get("label_counts", {})),
        "candidate_prior_count": candidate_count,
        "runtime_prior_count": max(0, mapped_count - candidate_count),
        "event_prior_summary": prior_summary,
        "min_selected_vs_best_local_margin": float(min(selected_vs_best)) if selected_vs_best else 0.0,
        "min_selected_vs_second_local_margin": float(min(selected_vs_second)) if selected_vs_second else 0.0,
        "max_duration_z_abs": float(max(duration_z_values)) if duration_z_values else 0.0,
        "states": tuple(state_diagnostics),
    }


def summarize_hsmm_event_priors_for_slots(
    slots: Sequence[FilenameSlot],
    event_priors: Iterable[Mapping[str, object]] = (),
) -> dict[str, object]:
    input_count = 0
    ignored_count = 0
    source_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    states: dict[str, dict[str, object]] = {}

    for event in event_priors:
        input_count += 1
        label = str(event.get("label", "") or "")
        phone_index = _int_or_none(event.get("expected_phone_index"))
        state_id = _state_id_for_event_prior(slots, label, phone_index)
        if not state_id:
            ignored_count += 1
            continue
        source = _event_source(event)
        _increment(source_counts, source)
        _increment(label_counts, label)
        state_summary = states.setdefault(
            state_id,
            {
                "state_id": state_id,
                "mapped_count": 0,
                "source_counts": {},
                "label_counts": {},
            },
        )
        state_summary["mapped_count"] = int(state_summary.get("mapped_count", 0) or 0) + 1
        _increment(state_summary["source_counts"], source)
        _increment(state_summary["label_counts"], label)

    mapped_count = sum(int(item.get("mapped_count", 0) or 0) for item in states.values())
    return {
        "input_count": int(input_count),
        "mapped_count": int(mapped_count),
        "ignored_count": int(ignored_count),
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "states": {key: _normalize_event_prior_state_summary(value) for key, value in sorted(states.items())},
    }


def build_state_interval_priors_for_slots(
    posterior: FramePosterior,
    slots: Sequence[FilenameSlot],
    states: Sequence[HSMMStateSpec],
    *,
    event_priors: Iterable[Mapping[str, object]] = (),
) -> dict[str, tuple[float, ...]]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    frame_count = int(times.size)
    if frame_count <= 0:
        return {}
    priors_by_state: dict[str, np.ndarray] = {}
    state_by_id = {state.state_id: state for state in states}
    events_by_key: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for event in event_priors:
        label = str(event.get("label", ""))
        phone_index = _int_or_none(event.get("expected_phone_index"))
        if not label or phone_index is None:
            continue
        events_by_key.setdefault((label, int(phone_index)), []).append(event)

    for slot in slots:
        onset_state_id = _state_id(slot, "onset")
        vowel_state_id = _state_id(slot, "vowel")
        onset_state = state_by_id.get(onset_state_id)
        vowel_state = state_by_id.get(vowel_state_id)
        if onset_state is not None and slot.onset_phones:
            onset_events = [
                *events_by_key.get(("phone_change", int(slot.phone_start_index)), []),
                *events_by_key.get(("sonorant_constriction", int(slot.phone_start_index)), []),
            ]
            for event in onset_events:
                event_time = _event_time_ms(event)
                priors_by_state[onset_state_id] = np.maximum(
                    priors_by_state.get(onset_state_id, np.zeros((frame_count,), dtype=np.float32)),
                    _event_weight(event) * _interval_prior_from_start(times, event_time, onset_state.mode_duration_ms),
                )
        if vowel_state is not None:
            start_events: list[Mapping[str, object]] = []
            if slot.onset_phones:
                start_events.extend(events_by_key.get(("cv_boundary", int(slot.vowel_phone_index)), []))
            else:
                start_events.extend(events_by_key.get(("vv_boundary", int(slot.vowel_phone_index)), []))
            for event in start_events:
                event_time = _event_time_ms(event)
                priors_by_state[vowel_state_id] = np.maximum(
                    priors_by_state.get(vowel_state_id, np.zeros((frame_count,), dtype=np.float32)),
                    _event_weight(event) * _interval_prior_from_start(times, event_time, vowel_state.mode_duration_ms),
                )
        if vowel_state is not None:
            for event in events_by_key.get(("vowel_nucleus", int(slot.vowel_phone_index)), []):
                event_time = _event_time_ms(event)
                priors_by_state[vowel_state_id] = np.maximum(
                    priors_by_state.get(vowel_state_id, np.zeros((frame_count,), dtype=np.float32)),
                    _event_weight(event) * _interval_prior_from_nucleus(times, event_time, vowel_state.mode_duration_ms),
                )
    return {key: tuple(float(value) for value in np.clip(value, 0.0, 1.0)) for key, value in priors_by_state.items()}


def hsmm_result_to_slot_events(
    result: HSMMDecodeResult,
    slots: Sequence[FilenameSlot],
) -> tuple[dict[str, object], ...]:
    if not result.ok:
        return ()
    by_id = {state.state_id: state for state in result.states}
    events: list[dict[str, object]] = []
    event_index = 0
    for slot in slots:
        onset_state = by_id.get(_state_id(slot, "onset"))
        onset_hold_state = by_id.get(_state_id(slot, "onset_hold"))
        vowel_state = by_id.get(_state_id(slot, "vowel"))
        if onset_state is not None and slot.onset_phones:
            events.append(
                _event(
                    label="phone_change",
                    time_ms=onset_state.start_ms,
                    score=onset_state.score,
                    expected_phone=slot.onset_phones[0],
                    expected_phone_index=slot.phone_start_index,
                    slot_index=event_index,
                    frame_index=onset_state.start_frame,
                )
            )
            event_index += 1
        if vowel_state is not None and slot.onset_phones:
            # The C->V boundary is where the consonant adjacent to the vowel
            # ends. The decoded consonant end is locked by the consonant track,
            # whereas the vowel segment start can be pulled inward by the
            # duration prior on long vowels. Prefer that consonant end (the hold
            # sub-state when present, else the onset) so the offset anchor stays
            # on the true onset rather than drifting late into the vowel.
            cv_boundary_ms = vowel_state.start_ms
            cv_boundary_frame = vowel_state.start_frame
            preceding_consonant = onset_hold_state or onset_state
            if (
                preceding_consonant is not None
                and preceding_consonant.end_ms <= vowel_state.start_ms
            ):
                cv_boundary_ms = preceding_consonant.end_ms
                cv_boundary_frame = preceding_consonant.end_frame
            events.append(
                _event(
                    label="cv_boundary",
                    time_ms=cv_boundary_ms,
                    score=vowel_state.score,
                    expected_phone=slot.vowel_phone,
                    expected_phone_index=slot.vowel_phone_index,
                    slot_index=event_index,
                    frame_index=cv_boundary_frame,
                )
            )
            event_index += 1
        if vowel_state is not None and not slot.onset_phones:
            events.append(
                _event(
                    label="vv_boundary",
                    time_ms=vowel_state.start_ms,
                    score=vowel_state.score,
                    expected_phone=slot.vowel_phone,
                    expected_phone_index=slot.vowel_phone_index,
                    slot_index=event_index,
                    frame_index=vowel_state.start_frame,
                )
            )
            event_index += 1
        if vowel_state is not None:
            nucleus_time = vowel_state.start_ms + (0.55 * max(0.0, vowel_state.duration_ms))
            events.append(
                _event(
                    label="vowel_nucleus",
                    time_ms=nucleus_time,
                    score=vowel_state.score,
                    expected_phone=slot.vowel_phone,
                    expected_phone_index=slot.vowel_phone_index,
                    slot_index=event_index,
                    frame_index=max(vowel_state.start_frame, min(vowel_state.end_frame - 1, int(round((vowel_state.start_frame + vowel_state.end_frame - 1) * 0.55)))),
                )
            )
            event_index += 1
    return tuple(events)


def _event(
    *,
    label: str,
    time_ms: float,
    score: float,
    expected_phone: str,
    expected_phone_index: int,
    slot_index: int,
    frame_index: int,
) -> dict[str, object]:
    return {
        "label": label,
        "selected_time_ms": float(time_ms),
        "score": float(max(0.0, min(1.0, score))),
        "expected_phone": str(expected_phone),
        "expected_phone_index": int(expected_phone_index),
        "slot_index": int(slot_index),
        "frame_index": int(frame_index),
        "source": "filename_hsmm",
    }


def _state_id(slot: FilenameSlot, kind: str) -> str:
    return f"slot{int(slot.slot_index)}.{kind}"


def _phones_include_sonorant(phones: Sequence[str]) -> bool:
    for phone in phones:
        normalized = str(phone or "").strip().lower()
        if not normalized:
            continue
        if normalized in _SONORANT_PHONES or normalized.startswith(("m", "n", "r", "l")):
            return True
    return False


def _onset_state_count(phones: Sequence[str], *, enable_sonorant_hold: bool = False) -> int:
    if not phones:
        return 0
    return 2 if bool(enable_sonorant_hold) and _phones_include_sonorant(phones) else 1


def _is_korean_language(language: str) -> bool:
    value = str(language or "").strip().lower()
    return value.startswith(("ko", "kr")) or value == "korean"


def _track(mapping: Mapping[str, Sequence[float]], key: str, frame_count: int) -> np.ndarray:
    values = mapping.get(key)
    if values is None:
        return np.zeros((frame_count,), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < frame_count:
        arr = np.pad(arr, (0, frame_count - arr.size), mode="constant")
    return np.clip(arr[:frame_count], 0.0, 1.0).astype(np.float32)


def _first_track(
    frame_count: int,
    *mappings: Mapping[str, Sequence[float]],
    keys: Sequence[str],
) -> np.ndarray:
    for key in keys:
        for mapping in mappings:
            arr = _track(mapping, key, frame_count)
            if np.any(arr):
                return arr
    return np.zeros((frame_count,), dtype=np.float32)


def _position_prior(times: np.ndarray, *, expected_ms: float, sigma_ms: float) -> np.ndarray:
    if times.size == 0:
        return times.astype(np.float32)
    sigma = max(1.0, float(sigma_ms))
    z = (times.astype(np.float32) - float(expected_ms)) / sigma
    return np.exp(-0.5 * z * z).astype(np.float32)


def _state_start_interval_track(
    times: np.ndarray,
    boundary_scores: np.ndarray,
    mode_duration_ms: float,
    *,
    top_k: int = 12,
) -> np.ndarray:
    if times.size == 0 or boundary_scores.size != times.size or not np.any(boundary_scores):
        return np.zeros_like(times, dtype=np.float32)
    peak = float(np.max(boundary_scores))
    if peak < 0.18:
        return np.zeros_like(times, dtype=np.float32)
    threshold = max(0.18, peak * 0.55)
    indices = np.flatnonzero(boundary_scores >= threshold)
    if indices.size == 0:
        return np.zeros_like(times, dtype=np.float32)
    if indices.size > top_k:
        order = np.argsort(boundary_scores[indices])[-top_k:]
        indices = indices[order]
    duration = max(20.0, float(mode_duration_ms or 0.0))
    sigma = max(14.0, duration * 0.24)
    out = np.zeros_like(times, dtype=np.float32)
    for idx in indices:
        center = float(times[int(idx)]) + (0.50 * duration)
        weight = float(boundary_scores[int(idx)])
        out = np.maximum(out, weight * _position_prior(times, expected_ms=center, sigma_ms=sigma))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _interval_prior_from_start(times: np.ndarray, start_ms: float, mode_duration_ms: float) -> np.ndarray:
    duration = max(20.0, float(mode_duration_ms or 0.0))
    center = float(start_ms) + (0.40 * duration)
    sigma = max(16.0, duration * 0.28)
    return _position_prior(times, expected_ms=center, sigma_ms=sigma)


def _interval_prior_from_nucleus(times: np.ndarray, nucleus_ms: float, mode_duration_ms: float) -> np.ndarray:
    duration = max(20.0, float(mode_duration_ms or 0.0))
    start = float(nucleus_ms) - (0.55 * duration)
    center = start + (0.50 * duration)
    sigma = max(18.0, duration * 0.36)
    return _position_prior(times, expected_ms=center, sigma_ms=sigma)


def _state_id_for_event_prior(
    slots: Sequence[FilenameSlot],
    label: str,
    phone_index: int | None,
) -> str:
    if phone_index is None:
        return ""
    label = str(label or "")
    for slot in slots:
        if label in {"phone_change", "sonorant_constriction"}:
            if slot.onset_phones and int(slot.phone_start_index) == int(phone_index):
                return _state_id(slot, "onset")
            continue
        if label == "cv_boundary":
            if slot.onset_phones and int(slot.vowel_phone_index) == int(phone_index):
                return _state_id(slot, "vowel")
            continue
        if label == "vv_boundary":
            if not slot.onset_phones and int(slot.vowel_phone_index) == int(phone_index):
                return _state_id(slot, "vowel")
            continue
        if label == "vowel_nucleus" and int(slot.vowel_phone_index) == int(phone_index):
            return _state_id(slot, "vowel")
    return ""


def _event_source(event: Mapping[str, object]) -> str:
    source = str(event.get("source", "") or "").strip()
    return source or "runtime_event"


def _increment(counts: dict[str, int], key: str) -> None:
    if not key:
        return
    counts[key] = int(counts.get(key, 0) or 0) + 1


def _normalize_event_prior_summary(summary: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        return {
            "input_count": 0,
            "mapped_count": 0,
            "ignored_count": 0,
            "source_counts": {},
            "label_counts": {},
            "states": {},
        }
    states = summary.get("states", {})
    state_items = states.items() if isinstance(states, Mapping) else ()
    return {
        "input_count": int(summary.get("input_count", 0) or 0),
        "mapped_count": int(summary.get("mapped_count", 0) or 0),
        "ignored_count": int(summary.get("ignored_count", 0) or 0),
        "source_counts": _string_int_counts(summary.get("source_counts", {})),
        "label_counts": _string_int_counts(summary.get("label_counts", {})),
        "states": {
            str(state_id): _normalize_event_prior_state_summary(state_summary)
            for state_id, state_summary in state_items
        },
    }


def _normalize_event_prior_state_summary(summary: object) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        return {
            "state_id": "",
            "mapped_count": 0,
            "source_counts": {},
            "label_counts": {},
        }
    return {
        "state_id": str(summary.get("state_id", "") or ""),
        "mapped_count": int(summary.get("mapped_count", 0) or 0),
        "source_counts": _string_int_counts(summary.get("source_counts", {})),
        "label_counts": _string_int_counts(summary.get("label_counts", {})),
    }


def _string_int_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, count in value.items():
        try:
            int_count = int(count or 0)
        except Exception:
            continue
        if int_count:
            out[str(key)] = int_count
    return dict(sorted(out.items()))


def _best_local_state_segments(
    scores: np.ndarray,
    spec: HSMMStateSpec,
    *,
    frame_shift_ms: float,
    selected_start_frame: int,
    selected_end_frame: int,
) -> dict[str, float | int]:
    frame_count = int(scores.size)
    min_frames = max(1, int(np.ceil(float(spec.min_duration_ms) / max(0.001, frame_shift_ms))))
    max_frames = max(min_frames, int(np.ceil(float(spec.max_duration_ms) / max(0.001, frame_shift_ms))))
    max_frames = min(max_frames, frame_count)
    best = {"score": float("-inf"), "start": 0, "end": 0}
    second = {"score": float("-inf"), "start": 0, "end": 0}
    prefix = np.concatenate(([0.0], np.cumsum(scores.astype(np.float64))))
    for start in range(0, max(0, frame_count - min_frames + 1)):
        for end in range(start + min_frames, min(frame_count, start + max_frames) + 1):
            duration_ms = float(end - start) * frame_shift_ms
            mean_score = float((prefix[end] - prefix[start]) / float(end - start))
            total = mean_score + _state_duration_score(spec, duration_ms)
            same_segment = start == int(selected_start_frame) and end == int(selected_end_frame)
            if total > best["score"]:
                if best["score"] != float("-inf"):
                    second = best
                best = {"score": float(total), "start": int(start), "end": int(end)}
                continue
            if not same_segment and total > second["score"]:
                second = {"score": float(total), "start": int(start), "end": int(end)}
    if best["score"] == float("-inf"):
        return {
            "best_total_score": 0.0,
            "best_start_frame": int(selected_start_frame),
            "best_end_frame": int(selected_end_frame),
            "second_total_score": 0.0,
            "second_start_frame": int(selected_start_frame),
            "second_end_frame": int(selected_end_frame),
        }
    if second["score"] == float("-inf"):
        second = best
    return {
        "best_total_score": float(best["score"]),
        "best_start_frame": int(best["start"]),
        "best_end_frame": int(best["end"]),
        "second_total_score": float(second["score"]),
        "second_start_frame": int(second["start"]),
        "second_end_frame": int(second["end"]),
    }


def _segment_mean_array(values: np.ndarray, start_frame: int, end_frame: int) -> float:
    if values.size == 0:
        return 0.0
    start = max(0, min(int(start_frame), int(values.size)))
    if start >= int(values.size):
        return 0.0
    end = max(start + 1, min(int(end_frame), int(values.size)))
    if end <= start:
        return 0.0
    return float(np.mean(values[start:end]))


def _state_duration_score(spec: HSMMStateSpec, duration_ms: float) -> float:
    z = _state_duration_z(spec, duration_ms)
    return -0.5 * z * z


def _state_duration_z(spec: HSMMStateSpec, duration_ms: float) -> float:
    mode = float(spec.mode_duration_ms or 0.0)
    if mode <= 0.0:
        return 0.0
    sigma = max(1.0, float(spec.duration_sigma_ms or 40.0))
    return (float(duration_ms) - mode) / sigma


def _event_time_ms(event: Mapping[str, object]) -> float:
    try:
        return float(event.get("selected_time_ms", event.get("time_ms", 0.0)) or 0.0)
    except Exception:
        return 0.0


def _event_weight(event: Mapping[str, object]) -> float:
    try:
        return max(0.0, min(1.0, float(event.get("score", 1.0) or 1.0)))
    except Exception:
        return 1.0


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_from_mapping(mapping: Mapping[str, object], key: str, *, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return float(default)


def _posterior_duration_ms(times: np.ndarray) -> float:
    if times.size == 0:
        return 0.0
    if times.size == 1:
        return max(1.0, float(times[0]))
    diffs = np.diff(times.astype(np.float32))
    positive = diffs[diffs > 0.0]
    shift = float(np.median(positive)) if positive.size else 5.0
    return float(times[-1]) + max(0.001, shift)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "FilenameHSMMDecode",
    "HSMM_ADAPTER_SCHEMA_VERSION",
    "build_hsmm_diagnostics",
    "build_hsmm_frame_scores",
    "build_hsmm_states_for_slots",
    "build_state_interval_priors_for_slots",
    "decode_filename_slots_with_hsmm",
    "hsmm_result_to_slot_events",
    "summarize_hsmm_event_priors_for_slots",
]
