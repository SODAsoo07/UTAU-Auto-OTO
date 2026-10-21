"""Row-level risk of a large preutterance error, for the clean/review split.

Each language/format with a fitted model has a logistic profile (profiles/*row_risk*.json) that
predicts P(|pre error| > 80 ms) from the row's confidence, decoder warnings, role and its agreement
with the vowel's voicing onset after a voiceless consonant (hand OTOs put pre there).

Korean VCV/CVC (profiles/korean_row_risk_v1.json): 34,523 generated rows of 12 Korean banks with
hand-made OTOs (7 VCV, 5 CVC; 2026-10-11). Its threshold keeps the clean rows' error rate at 2% on
the training banks; leave-one-bank-out: 79% clean, 2.1% errors among them, 67% of the errors routed
to review. It reads the onset from romaji aliases only (onset rule "v1"); the later profiles also read
kana and Hangul aliases (onset rule "v2") and list their own roles. A format without a profile stays
all-review.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from .kana import parse_kana_slots
from .voicing import frame_voicing

PROFILE_RESOURCE = "profiles/korean_row_risk_v1.json"
# (language, format) -> profile; languages are normalised by _language_key.
PROFILE_RESOURCES: Mapping[tuple[str, str], str] = {
    ("korean", "vcv"): PROFILE_RESOURCE,
    ("korean", "cvc"): PROFILE_RESOURCE,
    ("korean", "cv"): "profiles/row_risk_korean_cv_v1.json",
    ("korean", "cvvc"): "profiles/row_risk_korean_cvvc_v1.json",
    ("japanese", "cv"): "profiles/row_risk_japanese_cv_v1.json",
    ("japanese", "cvvc"): "profiles/row_risk_japanese_cvvc_v1.json",
    ("japanese", "vcv"): "profiles/row_risk_japanese_vcv_v1.json",
}
FLAGS = ("hsmm_low_local_margin", "hsmm_wrong_occurrence_risk", "hsmm_unmodeled_internal_gap", "hsmm_coverage_collapse",
         "fallback_landmark_multiple", "template_constraint_repaired", "soft_constraint_violation",
         "alias_type_requires_review")
ROLES = ("cv", "cv_head", "vcv", "vv", "vc", "v", "release")
VOICELESS_ONSETS = ("kk", "tt", "pp", "ss", "jj", "ch", "k", "t", "p", "s", "h")
# Onset rule v2: also Japanese sh/ts/f, and onsets read from kana or Hangul.
VOICELESS_ONSETS_V2 = frozenset({*VOICELESS_ONSETS, "sh", "ts", "f"})
_HANGUL_ONSETS = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h")
_ROMAN_ONSETS = sorted(VOICELESS_ONSETS_V2 | {"g", "n", "d", "r", "m", "b", "j", "z", "y", "w", "v", "ny", "gy"},
                       key=len, reverse=True)
LANDMARK_DISTANCE_CAP_MS = 250.0
FAR_FROM_LANDMARK_MS = 60.0


def _language_key(language: str | None) -> str:
    value = (language or "").casefold()
    return {"ko": "korean", "kr": "korean", "ja": "japanese", "jp": "japanese"}.get(value, value)


@lru_cache(maxsize=None)
def _load_resource(resource: str) -> Mapping[str, object] | None:
    path = files("universal_auto_oto").joinpath(resource)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_profile(language: str | None = "korean", format_hint: str | None = "vcv") -> Mapping[str, object] | None:
    """The fitted profile of a language/format, or None when there is none (the rows stay review)."""

    resource = PROFILE_RESOURCES.get((_language_key(language), (format_hint or "").casefold()))
    return _load_resource(resource) if resource else None


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
    """Onset rule v1 (Korean VCV/CVC profile): romaji right part starting with a voiceless consonant."""
    parts = alias.split()
    if not parts or (len(parts) == 2 and parts[1] in ("-", "R", "h")):
        return False
    right = parts[-1]
    return any(right.startswith(c) and len(right) > len(c) and right[len(c)] in "aeiouwy" for c in VOICELESS_ONSETS)


def right_onset_phone(alias: str) -> str | None:
    """Onset consonant of the alias's right part, read from romaji, kana or Hangul (None: no onset)."""
    parts = alias.split()
    if not parts or (len(parts) == 2 and parts[1] in ("-", "R", "h")):
        return None
    right = parts[-1]
    kana = parse_kana_slots(right)
    if kana:
        return kana[0][0] if len(kana[0]) > 1 else None
    first = right[0]
    if "가" <= first <= "힣":
        return _HANGUL_ONSETS[(ord(first) - 0xAC00) // 588] or None
    lowered = right.casefold()
    for onset in _ROMAN_ONSETS:
        if lowered.startswith(onset) and len(lowered) > len(onset) and lowered[len(onset)] in "aeiouwy":
            return onset
    return None


def _voiceless(alias: str, onset_rule: str) -> bool:
    if onset_rule == "v2":
        return right_onset_phone(alias) in VOICELESS_ONSETS_V2
    return voiceless_right_onset(alias)


def _log_ms(value: float) -> float:
    return float(np.log1p(max(0.0, min(float(value), 5000.0))))


def row_features(rows: Sequence[Mapping[str, object]], onsets: np.ndarray, *, onset_rule: str = "v1",
                 roles: Sequence[str] = ROLES, extended: bool = False) -> list[dict[str, float]]:
    """rows: alias, role, confidence, reasons, pre_abs (template preutterance) for one recording.

    extended (feature set v2) also reads the row's shape (offset_abs, fixed_abs, cutoff_abs,
    fallback_count, missing_landmarks) and adds file-level state: a misaligned take drags all its rows.
    """
    features = []
    for row in rows:
        reasons = set(row["reasons"])
        voiceless = _voiceless(str(row["alias"]), onset_rule) and len(onsets) > 0
        distance = float(np.min(np.abs(onsets - float(row["pre_abs"])))) if voiceless else 0.0
        # As fitted: the distance is capped, and a far landmark (sonorant-led file) counts as far.
        features.append({
            "conf": float(row["confidence"]),
            **{flag: float(flag in reasons) for flag in FLAGS},
            **{f"role_{role}": float(row["role"] == role) for role in roles},
            "voiceless": float(voiceless), "onset_dist": min(distance, LANDMARK_DISTANCE_CAP_MS) if voiceless else 0.0,
            "onset_far": float(voiceless and distance > FAR_FROM_LANDMARK_MS), "n_reasons": float(len(reasons)),
        })
    landmarks = [item["onset_far"] for item in features if item["voiceless"]]
    for item in features:
        item["file_far_rate"] = float(np.mean(landmarks)) if landmarks else 0.0
        item["file_has_landmarks"] = float(bool(landmarks))
    if extended:
        confidences = np.asarray([float(row["confidence"]) for row in rows]) if rows else np.zeros(1)
        reasons = [set(row["reasons"]) for row in rows]
        file_state = {
            "file_conf_mean": float(confidences.mean()), "file_conf_min": float(confidences.min()),
            "file_rows": float(np.log1p(len(rows))),
            "file_flag_rate": float(np.mean([len(r) for r in reasons])) if reasons else 0.0,
            "file_wrongocc_rate": float(np.mean(["hsmm_wrong_occurrence_risk" in r for r in reasons])) if reasons else 0.0,
        }
        for row, item in zip(rows, features):
            offset, pre = float(row["offset_abs"]), float(row["pre_abs"])
            item.update({
                "lead": _log_ms(pre - offset), "consonant": _log_ms(float(row["fixed_abs"]) - offset),
                "span": _log_ms(float(row["cutoff_abs"]) - offset),
                # Drift grows along a take: later rows of a long list file go wrong more often.
                "pre_position": _log_ms(pre),
                "fixed_after_pre": float(np.clip(float(row["fixed_abs"]) - pre, -500.0, 500.0)),
                "fallback": float(int(row.get("fallback_count", 0)) > 0),
                "missing_landmarks": float(int(row.get("missing_landmarks", 0)) > 0),
                **file_state,
            })
    return features


def _logistic(model: Mapping[str, object], features: Sequence[Mapping[str, float]]) -> np.ndarray:
    names = list(model["features"])
    weights = np.asarray(model["weights"], dtype=np.float64)
    mean, scale = np.asarray(model["mean"], dtype=np.float64), np.asarray(model["scale"], dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)  # a feature constant in training carries no weight
    matrix = np.asarray([[item[name] for name in names] for item in features], dtype=np.float64).reshape(-1, len(names))
    return 1.0 / (1.0 + np.exp(-(weights[0] + ((matrix - mean) / scale) @ weights[1:])))


def row_risks(rows: Sequence[Mapping[str, object]], wav_path: Path, onsets: np.ndarray | None = None,
              profile: Mapping[str, object] | None = None) -> list[float]:
    profile = profile if profile is not None else load_profile()
    features = row_features(rows, voicing_onsets_ms(wav_path) if onsets is None else onsets,
                            onset_rule=str(profile.get("onset_rule", "v1")), roles=tuple(profile.get("roles", ROLES)),
                            extended=bool(profile.get("extended_features")))
    role_models = profile.get("role_models")
    if not role_models:
        return _logistic(profile, features).tolist()
    # One logistic model per role; roles without a model use "other".
    risks = np.zeros(len(features))
    for role in {str(row["role"]) for row in rows}:
        indices = [i for i, row in enumerate(rows) if str(row["role"]) == role]
        model = role_models.get(role) or role_models["other"]
        risks[indices] = _logistic(model, [features[i] for i in indices])
    return risks.tolist()


def risk_threshold(profile: Mapping[str, object] | None = None, format_hint: str | None = None) -> float:
    """The profile's clean threshold; a profile shared by formats may hold one per format."""
    profile = profile if profile is not None else load_profile()
    per_format = profile.get("thresholds") or {}
    return float(per_format.get((format_hint or "").casefold(), profile["threshold"]))


__all__ = ["load_profile", "right_onset_phone", "row_features", "row_risks", "risk_threshold", "voicing_onsets_ms",
           "voiceless_right_onset"]
