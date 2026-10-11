"""Row-level risk of a large preutterance error, for the clean/review split of Korean VCV/CVC.

A logistic model (profiles/korean_row_risk_v1.json) fitted on 34,523 generated rows of 12 Korean
banks with hand-made OTOs (7 VCV, 5 CVC; 2026-10-11) predicts P(|pre error| > 80 ms) from the
row's confidence, decoder warnings, role and its agreement with the vowel's voicing onset after a
voiceless consonant (hand OTOs put pre there; median +2 ms). The threshold keeps the clean rows'
error rate at 2% on the training banks; leave-one-bank-out: 79% clean, 2.1% errors among them,
67% of the errors routed to review.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from .voicing import frame_voicing

PROFILE_RESOURCE = "profiles/korean_row_risk_v1.json"
FLAGS = ("hsmm_low_local_margin", "hsmm_wrong_occurrence_risk", "hsmm_unmodeled_internal_gap", "hsmm_coverage_collapse",
         "fallback_landmark_multiple", "template_constraint_repaired", "soft_constraint_violation",
         "alias_type_requires_review")
ROLES = ("cv", "cv_head", "vcv", "vv", "vc", "v", "release")
VOICELESS_ONSETS = ("kk", "tt", "pp", "ss", "jj", "ch", "k", "t", "p", "s", "h")
LANDMARK_DISTANCE_CAP_MS = 250.0
FAR_FROM_LANDMARK_MS = 60.0


@lru_cache(maxsize=1)
def load_profile() -> Mapping[str, object]:
    return json.loads(files("universal_auto_oto").joinpath(PROFILE_RESOURCE).read_text(encoding="utf-8"))


def voicing_onsets_ms(wav_path: Path) -> np.ndarray:
    """Voicing onsets: >=30 ms unvoiced then >=40 ms voiced (autocorrelation >= 0.5, F0 70-600 Hz)."""
    samples = sf.info(str(wav_path))
    rate = samples.samplerate
    if samples.frames < int(round(rate * 0.030)) + 14 * int(round(rate * 0.005)):
        return np.zeros(0)
    voicing = frame_voicing(wav_path)
    voiced, times = voicing.voiced(), voicing.times
    onsets = [times[k] for k in range(6, len(voiced) - 8)
              if voiced[k] and not voiced[k - 6:k].any() and voiced[k:k + 8].all()]
    return np.asarray(onsets, dtype=np.float64)


def voiceless_right_onset(alias: str) -> bool:
    parts = alias.split()
    if not parts or (len(parts) == 2 and parts[1] in ("-", "R", "h")):
        return False
    right = parts[-1]
    return any(right.startswith(c) and len(right) > len(c) and right[len(c)] in "aeiouwy" for c in VOICELESS_ONSETS)


def row_features(rows: Sequence[Mapping[str, object]], onsets: np.ndarray) -> list[dict[str, float]]:
    """rows: alias, role, confidence, reasons, pre_abs (template preutterance) for one recording."""
    features = []
    for row in rows:
        reasons = set(row["reasons"])
        voiceless = voiceless_right_onset(str(row["alias"])) and len(onsets) > 0
        distance = float(np.min(np.abs(onsets - float(row["pre_abs"])))) if voiceless else 0.0
        # As fitted: the distance is capped, and a far landmark (sonorant-led file) counts as far.
        features.append({
            "conf": float(row["confidence"]),
            **{flag: float(flag in reasons) for flag in FLAGS},
            **{f"role_{role}": float(row["role"] == role) for role in ROLES},
            "voiceless": float(voiceless), "onset_dist": min(distance, LANDMARK_DISTANCE_CAP_MS) if voiceless else 0.0,
            "onset_far": float(voiceless and distance > FAR_FROM_LANDMARK_MS), "n_reasons": float(len(reasons)),
        })
    landmarks = [item["onset_far"] for item in features if item["voiceless"]]
    for item in features:
        item["file_far_rate"] = float(np.mean(landmarks)) if landmarks else 0.0
        item["file_has_landmarks"] = float(bool(landmarks))
    return features


def row_risks(rows: Sequence[Mapping[str, object]], wav_path: Path, onsets: np.ndarray | None = None) -> list[float]:
    profile = load_profile()
    names = list(profile["features"])
    weights = np.asarray(profile["weights"], dtype=np.float64)
    mean, scale = np.asarray(profile["mean"], dtype=np.float64), np.asarray(profile["scale"], dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)  # a feature constant in training carries no weight
    features = row_features(rows, voicing_onsets_ms(wav_path) if onsets is None else onsets)
    matrix = np.asarray([[item[name] for name in names] for item in features], dtype=np.float64).reshape(-1, len(names))
    logits = weights[0] + ((matrix - mean) / scale) @ weights[1:]
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def risk_threshold() -> float:
    return float(load_profile()["threshold"])


__all__ = ["row_features", "row_risks", "risk_threshold", "voicing_onsets_ms", "voiceless_right_onset"]
