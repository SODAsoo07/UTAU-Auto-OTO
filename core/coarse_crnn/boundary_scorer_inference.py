from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig, load_boundary_checkpoint
from core.coarse_crnn.boundary_types import BoundaryFrameScores
from core.coarse_crnn.torch_utils import resolve_torch_device


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
        return BoundaryFrameScores(
            wav_path=wav_path,
            times_ms=[],
            scores={label: [] for label in config.labels},
            quality_scores=[],
            cvs_scores={label: [] for label in config.cvs_labels} if bool(config.enable_phone_aux_heads) else None,
            consonant_scores={label: [] for label in config.consonant_labels} if bool(config.enable_phone_aux_heads) else None,
            vowel_scores={label: [] for label in config.vowel_labels} if bool(config.enable_phone_aux_heads) else None,
            consonant_family_scores={label: [] for label in config.consonant_family_labels}
            if bool(config.enable_phone_family_heads)
            else None,
            vowel_nucleus_scores={label: [] for label in config.vowel_nucleus_labels}
            if bool(config.enable_phone_family_heads)
            else None,
            vowel_glide_scores={label: [] for label in config.vowel_glide_labels}
            if bool(config.enable_phone_family_heads)
            else None,
        )
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(torch_device)
        out = model.to(torch_device)(x)
        logits = out["boundary_logits"].squeeze(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        q_logits = out.get("quality_logits")
        q_probs = None
        if q_logits is not None:
            q_probs = torch.sigmoid(q_logits.squeeze(0)).detach().cpu().numpy().astype(np.float32)
        cvs_probs = _softmax_probs(torch, out.get("cvs_logits"))
        consonant_probs = _softmax_probs(torch, out.get("consonant_logits"))
        vowel_probs = _softmax_probs(torch, out.get("vowel_logits"))
        consonant_family_probs = _softmax_probs(torch, out.get("consonant_family_logits"))
        vowel_nucleus_probs = _softmax_probs(torch, out.get("vowel_nucleus_logits"))
        vowel_glide_probs = _softmax_probs(torch, out.get("vowel_glide_logits"))
    times = (np.arange(probs.shape[0], dtype=np.float32) * (float(hop_sec) * 1000.0)).tolist()
    scores = {label: probs[:, idx].tolist() for idx, label in enumerate(config.labels)}
    quality_scores = q_probs.tolist() if q_probs is not None else []
    cvs_scores = _prob_dict(config.cvs_labels, cvs_probs)
    consonant_scores = _prob_dict(config.consonant_labels, consonant_probs)
    vowel_scores = _prob_dict(config.vowel_labels, vowel_probs)
    consonant_family_scores = _prob_dict(config.consonant_family_labels, consonant_family_probs)
    vowel_nucleus_scores = _prob_dict(config.vowel_nucleus_labels, vowel_nucleus_probs)
    vowel_glide_scores = _prob_dict(config.vowel_glide_labels, vowel_glide_probs)
    return BoundaryFrameScores(
        wav_path=wav_path,
        times_ms=times,
        scores=scores,
        quality_scores=quality_scores,
        cvs_scores=cvs_scores,
        consonant_scores=consonant_scores,
        vowel_scores=vowel_scores,
        consonant_family_scores=consonant_family_scores,
        vowel_nucleus_scores=vowel_nucleus_scores,
        vowel_glide_scores=vowel_glide_scores,
    )


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
        "quality_scores": list(score_map.quality_scores or []),
        "cvs_scores": {k: list(v) for k, v in (score_map.cvs_scores or {}).items()},
        "consonant_scores": {k: list(v) for k, v in (score_map.consonant_scores or {}).items()},
        "vowel_scores": {k: list(v) for k, v in (score_map.vowel_scores or {}).items()},
        "consonant_family_scores": {k: list(v) for k, v in (score_map.consonant_family_scores or {}).items()},
        "vowel_nucleus_scores": {k: list(v) for k, v in (score_map.vowel_nucleus_scores or {}).items()},
        "vowel_glide_scores": {k: list(v) for k, v in (score_map.vowel_glide_scores or {}).items()},
        "meta": dict(meta or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = [
    "infer_boundary_scores_with_model",
    "predict_boundary_scores",
    "write_boundary_scores_json",
]


def _softmax_probs(torch, logits):
    if logits is None:
        return None
    values = torch.softmax(logits.squeeze(0), dim=-1)
    return values.detach().cpu().numpy().astype(np.float32)


def _prob_dict(labels, probs) -> dict[str, list[float]] | None:
    if probs is None:
        return None
    return {label: probs[:, idx].tolist() for idx, label in enumerate(labels)}

