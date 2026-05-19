from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import AcousticFeatureConfig, FeatureBatch, extract_features
from .manifest import load_manifest_jsonl
from .model import MfaFreeFrameModelConfig, build_frame_model
from .targets import IGNORE_INDEX, rasterize_targets
from .types import EVENT_LABELS, FRAME_LABELS


@dataclass(frozen=True)
class TrainPocConfig:
    manifest_path: str
    out_path: str
    encoder: str = "acoustic"
    epochs: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.10
    max_rows: int | None = None
    seed: int = 1337
    device: str | None = None
    event_loss_weight: float = 1.0
    specaugment: bool = False
    time_mask_frames: int = 8
    freq_mask_bins: int = 8
    mask_count: int = 1
    frame_class_weighting: str = "none"


def train_poc(config: TrainPocConfig) -> dict:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PoC training requires torch.") from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rows = load_manifest_jsonl(config.manifest_path, require_manual=True, require_labels=True)
    if config.max_rows is not None:
        rows = rows[: config.max_rows]
    if not rows:
        raise ValueError("No trainable manual-gold rows found.")
    batches = [_load_feature_batch(row, config.encoder, config.device) for row in rows]
    input_dim = int(batches[0].features.shape[1])
    if any(int(batch.features.shape[1]) != input_dim for batch in batches):
        raise ValueError("Feature dimensions differ between rows; use one encoder per run.")
    model_cfg = MfaFreeFrameModelConfig(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
        frame_labels=FRAME_LABELS,
        event_labels=EVENT_LABELS,
    )
    model = build_frame_model(model_cfg)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    class_weight = _frame_class_weight_tensor(rows, batches, mode=config.frame_class_weighting)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    history: list[dict] = []
    order = list(range(len(rows)))
    for epoch in range(1, max(1, config.epochs) + 1):
        random.shuffle(order)
        losses: list[float] = []
        class_losses: list[float] = []
        event_losses: list[float] = []
        model.train()
        for row_idx in order:
            row = rows[row_idx]
            batch = batches[row_idx]
            targets = rasterize_targets(row, batch.times_ms)
            features = batch.features
            if config.specaugment:
                features = _apply_feature_masking(
                    features,
                    time_mask_frames=config.time_mask_frames,
                    freq_mask_bins=config.freq_mask_bins,
                    mask_count=config.mask_count,
                )
            x = torch.from_numpy(features[None, :, :]).to(device)
            frame_class = torch.from_numpy(targets.frame_class[None, :]).long().to(device)
            frame_mask = torch.from_numpy(targets.frame_mask[None, :]).float().to(device)
            event_targets = torch.from_numpy(targets.event_targets[None, :, :]).float().to(device)
            output = model(x)
            frame_logits = output["frame_logits"]
            event_logits = output["event_logits"]
            flat_loss = F.cross_entropy(
                frame_logits.reshape(-1, frame_logits.shape[-1]),
                frame_class.reshape(-1),
                weight=class_weight,
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).reshape_as(frame_class).float()
            valid = (frame_class != IGNORE_INDEX).float() * torch.clamp(frame_mask, min=0.0, max=1.0)
            class_loss = (flat_loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)
            event_loss = F.binary_cross_entropy_with_logits(event_logits, event_targets)
            loss = class_loss + float(config.event_loss_weight) * event_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            class_losses.append(float(class_loss.detach().cpu()))
            event_losses.append(float(event_loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "frame_loss": float(np.mean(class_losses)),
                "event_loss": float(np.mean(event_losses)),
            }
        )
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mfa_free_oto_frame_model_v1",
            "model_config": model_cfg.to_dict(),
            "state_dict": model.state_dict(),
            "encoder": config.encoder,
            "acoustic_config": asdict(AcousticFeatureConfig()),
            "frame_labels": list(FRAME_LABELS),
            "event_labels": list(EVENT_LABELS),
            "train_config": asdict(config),
            "history": history,
        },
        str(out_path),
    )
    return {
        "rows": len(rows),
        "checkpoint": str(out_path),
        "encoder": config.encoder,
        "input_dim": input_dim,
        "history": history,
    }


def _load_feature_batch(row: dict, encoder: str, device: str | None) -> FeatureBatch:
    return extract_features(row["wav_path"], encoder=encoder, device=device)


def _frame_class_weight_tensor(rows: Sequence[dict], batches: Sequence[FeatureBatch], *, mode: str):
    normalized = str(mode or "none").strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    if normalized != "balanced":
        raise ValueError(f"Unsupported frame_class_weighting: {mode}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Class weighting requires torch.") from exc
    counts = np.zeros((len(FRAME_LABELS),), dtype=np.float64)
    for row, batch in zip(rows, batches):
        targets = rasterize_targets(row, batch.times_ms)
        valid = targets.frame_class != IGNORE_INDEX
        if np.any(valid):
            counts += np.bincount(
                targets.frame_class[valid].astype(np.int64),
                minlength=len(FRAME_LABELS),
            )[: len(FRAME_LABELS)]
    total = float(np.sum(counts))
    weights = np.ones_like(counts, dtype=np.float32)
    active = counts > 0.0
    if total > 0.0 and np.any(active):
        weights[active] = (total / (float(np.sum(active)) * counts[active])).astype(np.float32)
        weights = weights / max(1e-6, float(np.mean(weights[active])))
    return torch.from_numpy(weights.astype(np.float32))


def _apply_feature_masking(
    features: np.ndarray,
    *,
    time_mask_frames: int,
    freq_mask_bins: int,
    mask_count: int,
) -> np.ndarray:
    out = np.array(features, copy=True)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        return out
    for _ in range(max(0, int(mask_count))):
        if time_mask_frames > 0 and out.shape[0] > 1:
            width = random.randint(1, min(int(time_mask_frames), out.shape[0]))
            start = random.randint(0, max(0, out.shape[0] - width))
            out[start : start + width, :] = 0.0
        if freq_mask_bins > 0 and out.shape[1] > 1:
            width = random.randint(1, min(int(freq_mask_bins), out.shape[1]))
            start = random.randint(0, max(0, out.shape[1] - width))
            out[:, start : start + width] = 0.0
    return out.astype(np.float32)


def save_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
