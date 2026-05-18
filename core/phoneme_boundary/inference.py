from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.phoneme_boundary.features import FeatureConfig, load_boundary_features
from core.phoneme_boundary.heuristics import (
    blend_boundary_scores,
    blend_boundary_scores_adaptive,
    compute_acoustic_boundary_scores,
)
from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig, load_phoneme_boundary_checkpoint
from core.phoneme_boundary.types import BOUNDARY_LABELS, BoundaryEvent, BoundaryFrameScores, normalize_boundary_label


def infer_boundary_scores_with_model(
    *,
    model,
    config: PhonemeBoundaryDetectorConfig,
    wav_path: str,
    device: str = "cpu",
    use_acoustic_heuristics: bool = True,
    heuristic_weight: float = 0.28,
) -> BoundaryFrameScores:
    torch = __import__("torch")
    torch_device = _resolve_torch_device(torch, device)
    feats = load_boundary_features(
        wav_path,
        FeatureConfig(
            sample_rate=int(config.sample_rate),
            n_mels=int(config.n_mels),
            frame_ms=float(config.frame_ms),
            hop_ms=float(config.hop_ms),
        ),
    )
    labels = tuple(config.labels or BOUNDARY_LABELS)
    if feats.features.shape[0] <= 0:
        return BoundaryFrameScores(
            wav_path=str(wav_path),
            times_ms=[],
            scores={label: [] for label in labels},
            quality_scores=[],
            phone_state_scores={label: [] for label in config.phone_state_labels}
            if bool(config.enable_phone_state_head)
            else None,
            consonant_scores={label: [] for label in config.consonant_labels}
            if bool(config.enable_phone_identity_heads)
            else None,
            vowel_scores={label: [] for label in config.vowel_labels} if bool(config.enable_phone_identity_heads) else None,
            heuristic_scores={},
        )
    with torch.no_grad():
        x = torch.from_numpy(feats.features.astype(np.float32)).unsqueeze(0).to(torch_device)
        out = model.to(torch_device)(x)
        logits = out["boundary_logits"].squeeze(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        q_logits = out.get("quality_logits")
        q_probs = None
        if q_logits is not None:
            q_probs = torch.sigmoid(q_logits.squeeze(0)).detach().cpu().numpy().astype(np.float32)
        state_probs = _softmax_probs(torch, out.get("phone_state_logits"))
        consonant_probs = _softmax_probs(torch, out.get("consonant_logits"))
        vowel_probs = _softmax_probs(torch, out.get("vowel_logits"))
    times = (np.arange(probs.shape[0], dtype=np.float32) * float(feats.hop_ms)).tolist()
    scores = {label: probs[:, idx].tolist() for idx, label in enumerate(labels)}
    heuristic_scores = compute_acoustic_boundary_scores(
        feats.samples,
        feats.sample_rate,
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
        frame_count=int(probs.shape[0]),
    )
    if bool(use_acoustic_heuristics):
        if q_probs is not None and len(q_probs) == int(probs.shape[0]):
            scores = blend_boundary_scores_adaptive(
                scores,
                heuristic_scores,
                quality_scores=q_probs.tolist(),
                base_weight=max(0.0, float(heuristic_weight) * 0.35),
                low_conf_threshold=0.55,
                max_weight=max(0.0, float(heuristic_weight)),
            )
        else:
            scores = blend_boundary_scores(scores, heuristic_scores, heuristic_weight=float(heuristic_weight))
    return BoundaryFrameScores(
        wav_path=str(wav_path),
        times_ms=times,
        scores=scores,
        quality_scores=q_probs.tolist() if q_probs is not None else [],
        phone_state_scores=_prob_dict(config.phone_state_labels, state_probs),
        consonant_scores=_prob_dict(config.consonant_labels, consonant_probs),
        vowel_scores=_prob_dict(config.vowel_labels, vowel_probs),
        heuristic_scores=heuristic_scores,
    )


def predict_boundary_scores(
    *,
    model_path: str,
    wav_path: str,
    device: str = "auto",
    use_acoustic_heuristics: bool = True,
    heuristic_weight: float = 0.28,
) -> BoundaryFrameScores:
    torch = __import__("torch")
    torch_device = _resolve_torch_device(torch, device)
    model, config, _meta = load_phoneme_boundary_checkpoint(model_path, map_location=str(torch_device))
    return infer_boundary_scores_with_model(
        model=model,
        config=config,
        wav_path=wav_path,
        device=str(torch_device),
        use_acoustic_heuristics=bool(use_acoustic_heuristics),
        heuristic_weight=float(heuristic_weight),
    )


def peak_events_from_scores(
    score_map: BoundaryFrameScores,
    *,
    min_score: float = 0.35,
    min_gap_ms: float = 35.0,
    labels: tuple[str, ...] | None = None,
) -> list[BoundaryEvent]:
    out: list[BoundaryEvent] = []
    requested = labels or tuple(score_map.scores.keys())
    for label in requested:
        label_key = normalize_boundary_label(label)
        values = list(score_map.scores.get(label_key, []) or [])
        times = list(score_map.times_ms)
        if not values or not times:
            continue
        last_time = -1e9
        for idx, value in enumerate(values):
            v = float(value)
            if v < float(min_score):
                continue
            prev_v = float(values[idx - 1]) if idx > 0 else -1.0
            next_v = float(values[idx + 1]) if idx + 1 < len(values) else -1.0
            if v < prev_v or v < next_v:
                continue
            time_ms = float(times[min(idx, len(times) - 1)])
            if time_ms - last_time < float(min_gap_ms):
                if out and out[-1].label == label_key and v > out[-1].confidence:
                    out[-1] = BoundaryEvent(label_key, time_ms, confidence=v, source="peak")
                    last_time = time_ms
                continue
            out.append(BoundaryEvent(label_key, time_ms, confidence=v, source="peak"))
            last_time = time_ms
    return sorted(out, key=lambda item: (float(item.time_ms), item.label))


def write_boundary_scores_json(path: str, score_map: BoundaryFrameScores, *, meta: dict[str, Any] | None = None) -> None:
    payload = {
        "wav_path": score_map.wav_path,
        "times_ms": list(score_map.times_ms),
        "scores": {k: list(v) for k, v in score_map.scores.items()},
        "quality_scores": list(score_map.quality_scores or []),
        "phone_state_scores": {k: list(v) for k, v in (score_map.phone_state_scores or {}).items()},
        "consonant_scores": {k: list(v) for k, v in (score_map.consonant_scores or {}).items()},
        "vowel_scores": {k: list(v) for k, v in (score_map.vowel_scores or {}).items()},
        "heuristic_scores": {k: list(v) for k, v in (score_map.heuristic_scores or {}).items()},
        "meta": dict(meta or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolve_torch_device(torch, device: str):
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def _softmax_probs(torch, logits):
    if logits is None:
        return None
    values = torch.softmax(logits.squeeze(0), dim=-1)
    return values.detach().cpu().numpy().astype(np.float32)


def _prob_dict(labels, probs) -> dict[str, list[float]] | None:
    if probs is None:
        return None
    return {label: probs[:, idx].tolist() for idx, label in enumerate(labels)}


__all__ = [
    "infer_boundary_scores_with_model",
    "peak_events_from_scores",
    "predict_boundary_scores",
    "write_boundary_scores_json",
]
