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
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    log_every: int = 100
    val_ratio: float = 0.08
    augment: bool = True
    gain_db: float = 3.0
    noise_std: float = 0.004
    class_balance: bool = True
    boundary_loss_weight: float = 0.12


class CoarseFrameDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_config: CoarseCrnnConfig,
        *,
        max_frames: int = 1800,
        train: bool = True,
        augment: bool = False,
        gain_db: float = 3.0,
        noise_std: float = 0.004,
    ):
        self.rows = list(rows or [])
        self.config = model_config
        self.max_frames = int(max_frames)
        self.train = bool(train)
        self.augment = bool(augment)
        self.gain_db = float(gain_db)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        wav_path = str(row.get("audio", "") or "")
        language = str(row.get("language", "") or "")
        label_path = str(row.get("label_path", "") or "")
        label_format = str(row.get("label_format", "") or "")
        samples, sr, duration = load_wav_mono(wav_path, target_sr=int(self.config.sample_rate))
        if self.train and self.augment:
            samples = _augment_samples(samples, gain_db=self.gain_db, noise_std=self.noise_std)
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
        boundary_targets = _boundary_targets_from_frame_targets(targets)
        return features.astype(np.float32), targets.astype(np.int64), boundary_targets.astype(np.float32), float(row.get("weight", 1.0) or 1.0)


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

    train_ds = CoarseFrameDataset(
        train_rows,
        model_cfg,
        max_frames=int(cfg.max_frames),
        train=True,
        augment=bool(cfg.augment),
        gain_db=float(cfg.gain_db),
        noise_std=float(cfg.noise_std),
    )
    val_ds = CoarseFrameDataset(val_rows, model_cfg, max_frames=int(cfg.max_frames), train=False) if val_rows else None
    num_workers = max(0, int(cfg.num_workers))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(torch, cfg.device),
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=num_workers,
            pin_memory=_pin_memory_enabled(torch, cfg.device),
        )
        if val_ds
        else None
    )

    device = resolve_torch_device(torch, str(cfg.device))
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = _make_grad_scaler(torch, enabled=use_amp)
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
        for batch_idx, (x, y, b, w) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, enabled=use_amp):
                outputs = model.forward_all(x) if hasattr(model, "forward_all") else {"logits": model(x)}
                logits = outputs["logits"]
                loss_raw = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
                mask = (y != -100).float()
                if bool(cfg.class_balance):
                    class_weights = _batch_class_weights(y, logits.shape[-1], torch)
                    safe_y = torch.clamp(y, min=0)
                    loss_raw = loss_raw * class_weights[safe_y]
                weighted = loss_raw * mask * w[:, None]
                denom = torch.clamp((mask * w[:, None]).sum(), min=1.0)
                loss = weighted.sum() / denom
                if "boundary_logits" in outputs and float(cfg.boundary_loss_weight) > 0.0:
                    boundary_logits = outputs["boundary_logits"]
                    boundary_raw = nn.functional.binary_cross_entropy_with_logits(boundary_logits, b, reduction="none")
                    boundary_loss = (boundary_raw * mask * w[:, None]).sum() / denom
                    loss = loss + boundary_loss * float(cfg.boundary_loss_weight)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            frames = int(mask.sum().detach().cpu().item())
            train_loss += float(loss.detach().cpu().item()) * max(1, frames)
            train_frames += max(1, frames)
            if int(cfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(cfg.log_every) == 0):
                print(
                    f"[coarse_crnn][train] epoch={epoch}/{int(cfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f} "
                    f"device={device} amp={int(use_amp)}",
                    flush=True,
                )
        row = {"epoch": float(epoch), "train_loss": float(train_loss / max(1, train_frames))}
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device, class_balance=bool(cfg.class_balance))
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
            "device": str(device),
            "amp": bool(use_amp),
            "augment": bool(cfg.augment),
            "class_balance": bool(cfg.class_balance),
            "boundary_loss_weight": float(cfg.boundary_loss_weight),
        },
    )
    return {
        "output_path": os.path.abspath(output_path),
        "history": history,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "device": str(device),
        "amp": bool(use_amp),
    }


def _collate(batch):
    torch = __import__("torch")
    max_len = max(int(item[0].shape[0]) for item in batch)
    n_mels = int(batch[0][0].shape[1])
    xs = np.zeros((len(batch), max_len, n_mels), dtype=np.float32)
    ys = np.full((len(batch), max_len), -100, dtype=np.int64)
    bs = np.zeros((len(batch), max_len), dtype=np.float32)
    ws = np.ones((len(batch),), dtype=np.float32)
    for idx, (features, targets, boundaries, weight) in enumerate(batch):
        n = int(features.shape[0])
        xs[idx, :n, :] = features
        ys[idx, :n] = targets
        bs[idx, :n] = boundaries
        ws[idx] = float(weight)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(bs), torch.from_numpy(ws)


def _evaluate(model, loader, criterion, device, *, class_balance: bool = False) -> tuple[float, float]:
    torch = __import__("torch")
    model.eval()
    loss_sum = 0.0
    frame_sum = 0
    correct = 0
    with torch.no_grad():
        for x, y, _b, _w in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model.forward_all(x)["logits"] if hasattr(model, "forward_all") else model(x)
            loss_raw = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
            if bool(class_balance):
                class_weights = _batch_class_weights(y, logits.shape[-1], torch)
                safe_y = torch.clamp(y, min=0)
                loss_raw = loss_raw * class_weights[safe_y]
            mask = y != -100
            loss_sum += float(loss_raw[mask].sum().detach().cpu().item()) if bool(mask.any()) else 0.0
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred[mask] == y[mask]).sum().detach().cpu().item()) if bool(mask.any()) else 0
            frame_sum += int(mask.sum().detach().cpu().item())
    return float(loss_sum / max(1, frame_sum)), float(correct / max(1, frame_sum))


def _augment_samples(samples: np.ndarray, *, gain_db: float, noise_std: float) -> np.ndarray:
    out = np.asarray(samples, dtype=np.float32).copy()
    if out.size <= 0:
        return out
    if gain_db > 0:
        db = random.uniform(-float(gain_db), float(gain_db))
        out *= float(10.0 ** (db / 20.0))
    if noise_std > 0:
        rms = float(np.sqrt(np.mean(np.square(out))) + 1e-8)
        out += np.random.normal(0.0, rms * float(noise_std), size=out.shape).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _boundary_targets_from_frame_targets(targets: np.ndarray) -> np.ndarray:
    y = np.asarray(targets, dtype=np.int64)
    out = np.zeros((int(y.shape[0]),), dtype=np.float32)
    if y.shape[0] <= 1:
        return out
    for idx in range(1, int(y.shape[0])):
        if int(y[idx]) == -100 or int(y[idx - 1]) == -100:
            continue
        if int(y[idx]) != int(y[idx - 1]):
            out[idx] = 1.0
            out[idx - 1] = 1.0
            if idx + 1 < int(y.shape[0]) and int(y[idx + 1]) != -100:
                out[idx + 1] = 0.5
    return out


def _batch_class_weights(y, class_count: int, torch):
    valid = y[y != -100]
    if int(valid.numel()) <= 0:
        return torch.ones((int(class_count),), device=y.device, dtype=torch.float32)
    counts = torch.bincount(valid.to(torch.long), minlength=int(class_count)).float().to(y.device)
    weights = torch.rsqrt(torch.clamp(counts, min=1.0))
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return weights


def resolve_torch_device(torch, requested: str):
    text = str(requested or "auto").strip().lower()
    if text in {"auto", "cuda_auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was requested ({requested}) but this PyTorch build cannot use CUDA. "
                f"torch={getattr(torch, '__version__', 'unknown')}"
            )
        return torch.device(text)
    return torch.device(text or "cpu")


def _pin_memory_enabled(torch, requested: str) -> bool:
    text = str(requested or "auto").strip().lower()
    return bool((text in {"auto", "cuda_auto", "gpu"} or text.startswith("cuda")) and torch.cuda.is_available())


def _make_grad_scaler(torch, *, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            return None


class _NullAutocast:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _autocast(torch, *, enabled: bool):
    if not enabled:
        return _NullAutocast()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.autocast(enabled=True)
        except Exception:
            return _NullAutocast()


__all__ = ["CoarseFrameDataset", "TrainConfig", "resolve_torch_device", "train_from_manifest"]
