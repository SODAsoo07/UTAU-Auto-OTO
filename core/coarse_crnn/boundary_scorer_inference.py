from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig, load_boundary_checkpoint
from core.coarse_crnn.boundary_types import BoundaryFrameScores
from core.coarse_crnn.training import resolve_torch_device


def infer_boundary_scores_with_model(
    *,
    model,
    config: BoundaryScorerConfig,
    wav_path: str,
    device: str = "cpu",
) -> BoundaryFrameScores:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    samples, sr, _duration = load_wav_mono(wav_path, target_sr=int(config.sample_rate))
    features, hop_sec = log_mel_spectrogram(
        samples,
        sr,
        n_mels=int(config.n_mels),
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
    )
    if features.shape[0] <= 0:
        return BoundaryFrameScores(wav_path=wav_path, times_ms=[], scores={label: [] for label in config.labels})
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(torch_device)
        out = model.to(torch_device)(x)
        logits = out["boundary_logits"].squeeze(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    times = (np.arange(probs.shape[0], dtype=np.float32) * (float(hop_sec) * 1000.0)).tolist()
    scores = {label: probs[:, idx].tolist() for idx, label in enumerate(config.labels)}
    return BoundaryFrameScores(wav_path=wav_path, times_ms=times, scores=scores)


def predict_boundary_scores(*, model_path: str, wav_path: str, device: str = "auto") -> BoundaryFrameScores:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model, config, _meta = load_boundary_checkpoint(model_path, map_location=str(torch_device))
    return infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(torch_device))


def write_boundary_scores_json(path: str, score_map: BoundaryFrameScores, *, meta: dict[str, Any] | None = None) -> None:
    payload = {
        "wav_path": score_map.wav_path,
        "times_ms": list(score_map.times_ms),
        "scores": {k: list(v) for k, v in score_map.scores.items()},
        "meta": dict(meta or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = [
    "infer_boundary_scores_with_model",
    "predict_boundary_scores",
    "write_boundary_scores_json",
]

