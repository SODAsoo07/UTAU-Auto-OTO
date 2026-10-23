"""Row-level review risk for language-pack banks (English VCCV first).

A gradient-boosted tree model (profiles/<pack>_row_risk_v1.json) predicts P(|pre error| > 60 ms
after the labeler's own style offset) from reference-free signals: decoder warnings, template
fallbacks, alias shape, phone durations against the bank's own medians, how far the row strays from
rows of the same shape in the same folder (VCCV lists are usually sung to a metronome) and acoustic
cues around the preutterance.

The features need the whole folder (duration medians, tempo grid), so the model runs once after
every recording is rendered. The split is by rank: the riskiest REVIEW_SHARE of the rows go to
review, because the predicted probabilities do not transfer between banks as well as the ranking
does. Leave-one-bank-out on five banks (2026-10-13): the riskiest 30% hold 62-83% of the rows that
are more than 60 ms off and 74-90% of those more than 100 ms off.
"""

from __future__ import annotations

import collections
import json
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .generic_alias_binding import parse_pack_alias, style_keys
from .language_pack import LanguagePack
from .pack_landmarks import PACK_RUMBLE_HIGHPASS_HZ
from .voicing import frame_voicing

REVIEW_SHARE = 0.30
INSPECT_FIRST_SHARE = 0.10
FLAGS = ("hsmm_low_local_margin", "hsmm_wrong_occurrence_risk", "hsmm_unmodeled_internal_gap", "hsmm_coverage_collapse",
         "template_constraint_repaired")
USAGE = ("local_boundary_ambiguity", "tail_extends_into_following_phone", "global_acoustic_competitor",
         "repeated_phone_ambiguity", "gap_between_referenced_phones")
CATEGORIES = ("inspect_first", "spot_check", "connection_check")
SHAPES = ("-CV", "CV", "V C", "VC", "VC-", "C C", "VC C", "CCV", "-CCV", "CC", "VV", "V CC", "_CV", "V", "-V", "V-",
          "CC-", "-CC", "CCCV")
CLASSES = ("V", "son", "stop", "affricate", "tap", "fric_v", "fric_u")
HOP_MS = 5.0


@lru_cache(maxsize=4)
def load_profile(pack_id: str) -> Mapping[str, object] | None:
    resource = files("universal_auto_oto").joinpath(f"profiles/{pack_id}_row_risk_v1.json")
    if not resource.is_file():
        return None
    return json.loads(resource.read_text(encoding="utf-8"))


def wav_tracks(wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times ms, voiced, high-passed energy dB) every 5 ms."""
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = sosfiltfilt(butter(4, PACK_RUMBLE_HIGHPASS_HZ, "highpass", fs=rate, output="sos"), samples.mean(axis=1))
    hop, window = int(rate * HOP_MS / 1000.0), int(rate * 0.02)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(mono, (window // 2, window)), window)[::hop]
    energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    voicing = frame_voicing(wav_path)
    times = np.arange(len(energy)) * HOP_MS
    voiced = np.interp(times, voicing.times, voicing.voiced().astype(float)) > 0.5
    return times, voiced, energy


def _log_ratio(segment: Mapping[str, object], medians: Mapping[str, float]) -> float:
    duration = float(segment["end_ms"]) - float(segment["start_ms"])
    return math.log(max(5.0, duration) / max(5.0, medians.get(str(segment["phone_id"]), 100.0)))


def _nearest(times: np.ndarray, value: float) -> float:
    return math.log1p(float(np.min(np.abs(times - value)))) if len(times) else 6.0


def _acoustic(tracks, segments, indices, pre: float, nuclei) -> dict[str, float]:
    times, voiced, energy = tracks
    top = float(energy.max())
    span = (times >= segments[indices[0]]["start_ms"]) & (times < segments[indices[-1]]["end_ms"])
    whole = (times >= segments[0]["start_ms"]) & (times < segments[-1]["end_ms"])
    voiced_share = []
    for index in nuclei:
        inside = (times >= segments[index]["start_ms"]) & (times < segments[index]["end_ms"])
        voiced_share.append(float(voiced[inside].mean()) if inside.any() else 0.0)
    onsets = times[1:][voiced[1:] & ~voiced[:-1]]
    offsets = times[1:][~voiced[1:] & voiced[:-1]]
    drops = times[1:][np.diff(energy) < -1.5]
    return {
        "ac_vowel_voiced_min": min(voiced_share, default=1.0),
        "ac_span_quiet": float((energy[span] < top - 35).mean()) if span.any() else 0.0,
        "ac_outside_active": math.log1p(HOP_MS * float(((energy > top - 30) & ~whole).sum())),
        "ac_pre_onset_dist": _nearest(onsets, pre),
        "ac_pre_offset_dist": _nearest(offsets, pre),
        "ac_pre_drop_dist": _nearest(drops, pre),
    }


def _landmark_features(timeline: Mapping[str, object], segments, indices, nuclei) -> dict[str, float]:
    """How far the pack landmarks sit from the decoded boundaries, and how far local refinement moved
    the bound phones' landmarks: both mean the boundary is uncertain."""
    bound = set(indices)
    offset_gap = onset_gap = shift = 0.0
    has_offset = has_onset = 0.0
    for landmark in timeline.get("landmarks", ()) or ():
        index = landmark.get("phone_index")
        if index not in bound:
            continue
        time = float(landmark["time_ms"])
        if landmark.get("label") == "vowel_offset" and index in nuclei:
            offset_gap = max(offset_gap, abs(time - float(segments[index]["end_ms"])))
            has_offset = 1.0
        elif landmark.get("label") == "vowel_onset" and index in nuclei:
            onset_gap = max(onset_gap, abs(time - float(segments[index]["start_ms"])))
            has_onset = 1.0
        raw = landmark.get("raw_time_ms")
        if raw is not None:
            shift = max(shift, abs(time - float(raw)))
    return {"lm_offset_gap": math.log1p(offset_gap), "lm_onset_gap": math.log1p(onset_gap),
            "lm_has_offset": has_offset, "lm_has_onset": has_onset, "lm_refine_shift": math.log1p(shift)}


def row_features(
    pack: LanguagePack,
    reports: Sequence[Mapping[str, object]],
    timelines: Mapping[int, Mapping[str, object]],
    wav_paths: Mapping[int, Path],
) -> list[dict[str, float]]:
    """Features for row reports of one generation; consistency is measured per output oto.ini."""
    by_output: dict[str, list[int]] = collections.defaultdict(list)
    for position, report in enumerate(reports):
        by_output[str(report.get("output_oto", ""))].append(position)
    features: list[dict[str, float]] = [{} for _ in reports]
    tracks_cache: dict[int, tuple] = {}
    for positions in by_output.values():
        recordings = sorted({int(reports[p]["recording_index"]) for p in positions})
        durations: dict[str, list[float]] = collections.defaultdict(list)
        for recording in recordings:
            for segment in timelines[recording]["phone_segments"]:
                durations[segment["phone_id"]].append(segment["end_ms"] - segment["start_ms"])
        medians = {phone: float(np.median(values)) for phone, values in durations.items()}
        context = {}
        for position in positions:
            report = reports[position]
            recording = int(report["recording_index"])
            timeline = timelines[recording]
            segments = timeline["phone_segments"]
            indices = [i for i in report["phone_indices"] if i < len(segments)]
            parsed = parse_pack_alias(pack, str(report["alias"]))
            shape = parsed.shape if parsed else "?"
            klass = style_keys(pack, parsed)[0].split("|")[1] if parsed else "?"
            pre = float(report["absolute"]["pre_abs"])
            ratios = [_log_ratio(segments[i], medians) for i in indices]
            nuclei = [i for i in indices if pack.is_nucleus(segments[i]["phone_id"])]
            anchor = segments[nuclei[0]]["start_ms"] if nuclei else segments[indices[0]]["start_ms"]
            bounds = sorted({s["start_ms"] for s in segments} | {s["end_ms"] for s in segments})
            phones = [segments[i]["phone_id"] for i in indices]
            nk = [k for k, phone in enumerate(phones) if pack.is_nucleus(phone)]
            pivot = (phones[nk[0] - 1] if nk and nk[0] > 0 else phones[nk[0] + 1] if nk and nk[0] + 1 < len(phones)
                     else phones[0])
            reasons = set(report.get("reasons", []))
            usage = report.get("usage_review", {}) or {}
            item = {
                "conf": float(report.get("confidence", 0.5)), "tl_conf": float(timeline.get("confidence", 0.5)),
                "fallback": float(report.get("fallback_count", 0)), "missing": float(len(report.get("missing_landmarks", []))),
                **{f"flag_{x}": float(x in reasons) for x in FLAGS},
                **{f"use_{x}": float(x in usage.get("reasons", [])) for x in USAGE},
                **{f"cat_{x}": float(usage.get("category") == x) for x in CATEGORIES},
                **{f"shape_{x}": float(shape == x) for x in SHAPES},
                **{f"cls_{x}": float(klass == x) for x in CLASSES},
                "dur_max": max(ratios, default=0.0), "dur_min": min(ratios, default=0.0),
                "dur_absmax": max((abs(x) for x in ratios), default=0.0),
                "file_dur_absmax": max(abs(_log_ratio(s, medians)) for s in segments),
                "bound_dist": math.log1p(min(abs(pre - b) for b in bounds)) if bounds else 0.0,
                "n_phones": float(len(segments)),
            }
            if recording not in tracks_cache:
                tracks_cache = {recording: wav_tracks(wav_paths[recording])}  # rows arrive grouped by recording
            item.update(_acoustic(tracks_cache[recording], segments, indices, pre, nuclei))
            item.update(_landmark_features(timeline, segments, indices, nuclei))
            times, _voiced, energy = tracks_cache[recording]
            speaking = times[energy > energy.max() - 30]
            speech_end = float(speaking[-1]) if len(speaking) else float(segments[-1]["end_ms"])
            features[position] = item
            context[position] = {"shape": shape, "slot": report.get("slot_index", -1), "pivot": pivot, "pre": pre,
                                 "rel": pre - anchor,
                                 "rele": pre - (segments[nuclei[-1]]["end_ms"] if nuclei else segments[indices[-1]]["end_ms"]),
                                 "endrel": pre - speech_end,
                                 "brel": min((pre - b for b in bounds), key=abs) if bounds else 0.0}
        # Consistency with rows of the same shape in this folder: tempo grid (shape, slot), pre
        # relative to the nucleus start and end (by shape, by shape and pivot consonant), to the
        # nearest boundary and to the end of the speech.
        for name, key, value, need in (("tempo", lambda c: (c["shape"], c["slot"]), "pre", 5),
                                       ("rel", lambda c: c["shape"], "rel", 5),
                                       ("relp", lambda c: (c["shape"], c["pivot"]), "rel", 3),
                                       ("brel", lambda c: c["shape"], "brel", 5),
                                       ("rele", lambda c: c["shape"], "rele", 5),
                                       ("relep", lambda c: (c["shape"], c["pivot"]), "rele", 3),
                                       ("endrel", lambda c: c["shape"], "endrel", 5)):
            groups: dict[object, list[int]] = collections.defaultdict(list)
            for position in positions:
                groups[key(context[position])].append(position)
            for members in groups.values():
                values = np.array([context[p][value] for p in members])
                middle = float(np.median(values))
                spread = float(np.median(np.abs(values - middle))) if len(members) >= need else 0.0
                for position, v in zip(members, values):
                    deviation = abs(float(v) - middle) if len(members) >= need else 0.0
                    features[position][f"{name}_dev"] = math.log1p(deviation)
                    features[position][f"{name}_spread"] = math.log1p(spread)
                    features[position][f"{name}_z"] = min(10.0, deviation / max(15.0, spread))
    return features


def predict(profile: Mapping[str, object], features: Sequence[Mapping[str, float]]) -> np.ndarray:
    names = list(profile["features"])
    matrix = np.asarray([[item.get(name, 0.0) for name in names] for item in features], dtype=np.float64)
    matrix = matrix.reshape(-1, len(names))
    edges = [np.asarray(e, dtype=np.float64) for e in profile["edges"]]
    binned = np.stack([np.searchsorted(edges[j], matrix[:, j], side="right") for j in range(len(names))], 1) \
        if len(matrix) else np.zeros((0, len(names)), dtype=int)
    score = np.full(len(matrix), float(profile["base"]))
    rate = float(profile["learning_rate"])
    for tree in profile["trees"]:
        score += rate * _apply_tree(tree, binned)
    return 1.0 / (1.0 + np.exp(-score))


def _apply_tree(tree, binned: np.ndarray) -> np.ndarray:
    out = np.zeros(len(binned))
    stack = [(tree, np.arange(len(binned)))]
    while stack:
        node, rows = stack.pop()
        if node[0] == "leaf":
            out[rows] = node[1]
        else:
            left = binned[rows, node[1]] <= node[2]
            stack += [(node[3], rows[left]), (node[4], rows[~left])]
    return out


def review_ranks(risks: Sequence[float]) -> list[str]:
    """'inspect_first' (riskiest 10%), 'review' (next 20%) or 'clean' for each row."""
    count = len(risks)
    order = np.argsort(-np.asarray(risks, dtype=np.float64), kind="stable")
    review, inspect = math.ceil(REVIEW_SHARE * count), math.ceil(INSPECT_FIRST_SHARE * count)
    labels = ["clean"] * count
    for rank, position in enumerate(order):
        if rank < inspect:
            labels[position] = "inspect_first"
        elif rank < review:
            labels[position] = "review"
    return labels


def risk_hints(item: Mapping[str, float]) -> list[str]:
    """Readable cues behind a high risk, for the review list."""
    hints = []
    if item.get("tempo_z", 0.0) >= 3.0:
        hints.append("pack_risk_off_tempo")
    if item.get("dur_absmax", 0.0) >= math.log(2.5):
        hints.append("pack_risk_phone_length")
    if item.get("flag_hsmm_unmodeled_internal_gap") or item.get("ac_span_quiet", 0.0) >= 0.5:
        hints.append("pack_risk_silence_or_noise")
    if item.get("relp_z", 0.0) >= 3.0 or item.get("rel_z", 0.0) >= 3.0:
        hints.append("pack_risk_unusual_position")
    if item.get("fallback", 0.0) or item.get("missing", 0.0):
        hints.append("pack_risk_landmark_fallback")
    return hints


__all__ = ["REVIEW_SHARE", "load_profile", "predict", "review_ranks", "risk_hints", "row_features", "wav_tracks"]
