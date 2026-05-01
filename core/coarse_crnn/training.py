from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.lab_io import frame_targets_from_segments, load_aligned_segments
from core.coarse_crnn.model import CoarseCrnnConfig, build_model, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 1e-3
    batch_size: int = 4
    max_frames: int = 1800
    seed: int = 1337
    device: str = "cpu"
    val_ratio: float = 0.08


class CoarseFrameDataset:
    def __init__(self, rows: list[dict[str, Any]], model_config: CoarseCrnnConfig, *, max_frames: int = 1800, train: bool = True):
        self.rows = list(rows or [])
        self.config = model_config
        self.max_frames = int(max_frames)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        wav_path = str(row.get("audio", "") or "")
        language = str(row.get("language", "") or "")
        label_path = str(row.get("label_path", "") or "")
        label_format = str(row.get("label_format", "") or "")
        samples, sr, duration = load_wav_mono(wav_path, target_sr=int(self.config.sample_rate))
        features, hop_sec = log_mel_spectrogram(
            samples,
            sr,
            n_mels=int(self.config.n_mels),
            frame_ms=float(self.config.frame_ms),
            hop_ms=float(self.config.hop_ms),
        )
        segments = load_aligned_segments(label_path, label_format)
        targets = frame_targets_from_segments(
            segments,
            frame_count=int(features.shape[0]),
            hop_sec=float(hop_sec),
            duration_sec=float(duration),
            language=language,
        )
        if self.max_frames > 0 and features.shape[0] > self.max_frames:
            if self.train:
                start = random.randint(0, int(features.shape[0] - self.max_frames))
            else:
                start = 0
            end = start + self.max_frames
            features = features[start:end]
            targets = targets[start:end]
        return features.astype(np.float32), targets.astype(np.int64), float(row.get("weight", 1.0) or 1.0)


def train_from_manifest(rows: list[dict[str, Any]], output_path: str, *, train_config: TrainConfig | None = None, model_config: CoarseCrnnConfig | None = None) -> dict[str, Any]:
    torch = __import__("torch")
    nn = __import__("torch.nn").nn
    data = list(rows or [])
    if not data:
        raise ValueError("manifest has no training rows")
    cfg = train_config or TrainConfig()
    model_cfg = model_config or CoarseCrnnConfig()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    random.shuffle(data)
    val_count = max(1, int(round(len(data) * float(cfg.val_ratio)))) if len(data) >= 10 else 0
    val_rows = data[:val_count]
    train_rows = data[val_count:] if val_count else data
    if not train_rows:
        train_rows = data
        val_rows = []

    train_ds = CoarseFrameDataset(train_rows, model_cfg, max_frames=int(cfg.max_frames), train=True)
    val_ds = CoarseFrameDataset(val_rows, model_cfg, max_frames=int(cfg.max_frames), train=False) if val_rows else None
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate) if val_ds else None

    device = torch.device(str(cfg.device))
    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    history: list[dict[str, float]] = []
    best_val = None
    best_state = None
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_frames = 0
        for x, y, w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_raw = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
            mask = (y != -100).float()
            weighted = loss_raw * mask * w[:, None]
            denom = torch.clamp((mask * w[:, None]).sum(), min=1.0)
            loss = weighted.sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            frames = int(mask.sum().detach().cpu().item())
            train_loss += float(loss.detach().cpu().item()) * max(1, frames)
            train_frames += max(1, frames)
        row = {"epoch": float(epoch), "train_loss": float(train_loss / max(1, train_frames))}
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            row["val_loss"] = float(val_loss)
            row["val_acc"] = float(val_acc)
            if best_val is None or val_loss < best_val:
                best_val = float(val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_checkpoint(
        output_path,
        model.cpu(),
        model_cfg,
        meta={
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "history": history,
        },
    )
    return {"output_path": os.path.abspath(output_path), "history": history, "train_rows": len(train_rows), "val_rows": len(val_rows)}


def _collate(batch):
    torch = __import__("torch")
    max_len = max(int(item[0].shape[0]) for item in batch)
    n_mels = int(batch[0][0].shape[1])
    xs = np.zeros((len(batch), max_len, n_mels), dtype=np.float32)
    ys = np.full((len(batch), max_len), -100, dtype=np.int64)
    ws = np.ones((len(batch),), dtype=np.float32)
    for idx, (features, targets, weight) in enumerate(batch):
        n = int(features.shape[0])
        xs[idx, :n, :] = features
        ys[idx, :n] = targets
        ws[idx] = float(weight)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ws)


def _evaluate(model, loader, criterion, device) -> tuple[float, float]:
    torch = __import__("torch")
    model.eval()
    loss_sum = 0.0
    frame_sum = 0
    correct = 0
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_raw = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
            mask = y != -100
            loss_sum += float(loss_raw[mask].sum().detach().cpu().item()) if bool(mask.any()) else 0.0
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred[mask] == y[mask]).sum().detach().cpu().item()) if bool(mask.any()) else 0
            frame_sum += int(mask.sum().detach().cpu().item())
    return float(loss_sum / max(1, frame_sum)), float(correct / max(1, frame_sum))


__all__ = ["CoarseFrameDataset", "TrainConfig", "train_from_manifest"]
