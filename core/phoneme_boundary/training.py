from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.phoneme_boundary.features import FeatureConfig, load_boundary_features
from core.phoneme_boundary.model import (
    PhonemeBoundaryDetectorConfig,
    build_phoneme_boundary_detector,
    save_phoneme_boundary_checkpoint,
)
from core.phoneme_boundary.targets import (
    build_boundary_target_map,
    build_phone_aux_target_map,
    events_from_manifest_row,
    resolve_wav_path,
)
from core.phoneme_boundary.types import IGNORE_INDEX


@dataclass
class PhonemeBoundaryTrainConfig:
    epochs: int = 4
    lr: float = 1e-3
    batch_size: int = 4
    max_frames: int = 1600
    seed: int = 1337
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    log_every: int = 80
    val_ratio: float = 0.08
    pos_weight: float = 2.5
    quality_loss_weight: float = 0.03
    phone_state_loss_weight: float = 0.35
    consonant_loss_weight: float = 0.20
    vowel_loss_weight: float = 0.20


class PhonemeBoundaryDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_config: PhonemeBoundaryDetectorConfig,
        *,
        manifest_dir: str = "",
        max_frames: int = 1600,
        train: bool = True,
    ):
        self.rows = list(rows or [])
        self.config = model_config
        self.manifest_dir = str(manifest_dir or "")
        self.max_frames = int(max_frames)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        wav_path = resolve_wav_path(row, manifest_dir=self.manifest_dir)
        events = events_from_manifest_row(row)
        if not wav_path:
            raise ValueError("manifest row missing wav_path/wav/audio")
        if not events:
            raise ValueError(f"manifest row has no boundary events: {wav_path}")
        feats = load_boundary_features(
            wav_path,
            FeatureConfig(
                sample_rate=int(self.config.sample_rate),
                n_mels=int(self.config.n_mels),
                frame_ms=float(self.config.frame_ms),
                hop_ms=float(self.config.hop_ms),
            ),
        )
        _times, target = build_boundary_target_map(
            events,
            duration_ms=float(feats.duration_ms),
            hop_ms=float(feats.hop_ms),
            frame_count=int(feats.features.shape[0]),
            labels=tuple(self.config.labels),
        )
        features = feats.features
        phone_aux = build_phone_aux_target_map(
            row,
            frame_count=int(features.shape[0]),
            hop_ms=float(feats.hop_ms),
            phone_state_labels=tuple(self.config.phone_state_labels),
            consonant_labels=tuple(self.config.consonant_labels),
            vowel_labels=tuple(self.config.vowel_labels),
        )
        if self.max_frames > 0 and features.shape[0] > self.max_frames:
            if self.train:
                start = random.randint(0, int(features.shape[0] - self.max_frames))
            else:
                start = 0
            end = start + self.max_frames
            features = features[start:end]
            target = target[start:end]
            phone_aux = phone_aux.__class__(
                state_target=phone_aux.state_target[start:end],
                state_mask=phone_aux.state_mask[start:end],
                consonant_target=phone_aux.consonant_target[start:end],
                consonant_mask=phone_aux.consonant_mask[start:end],
                vowel_target=phone_aux.vowel_target[start:end],
                vowel_mask=phone_aux.vowel_mask[start:end],
            )
        quality = target.max(axis=1).astype(np.float32)
        return features.astype(np.float32), target.astype(np.float32), quality, phone_aux


def train_phoneme_boundary_from_manifest(
    rows: list[dict[str, Any]],
    output_path: str,
    *,
    manifest_dir: str = "",
    train_config: PhonemeBoundaryTrainConfig | None = None,
    model_config: PhonemeBoundaryDetectorConfig | None = None,
) -> dict[str, Any]:
    torch = __import__("torch")
    nn = __import__("torch.nn").nn
    data = [row for row in list(rows or []) if isinstance(row, dict)]
    if not data:
        raise ValueError("manifest has no rows")
    cfg = train_config or PhonemeBoundaryTrainConfig()
    model_cfg = model_config or PhonemeBoundaryDetectorConfig()
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

    train_ds = PhonemeBoundaryDataset(
        train_rows,
        model_cfg,
        manifest_dir=manifest_dir,
        max_frames=int(cfg.max_frames),
        train=True,
    )
    val_ds = (
        PhonemeBoundaryDataset(
            val_rows,
            model_cfg,
            manifest_dir=manifest_dir,
            max_frames=int(cfg.max_frames),
            train=False,
        )
        if val_rows
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=_collate,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=_pin_memory_enabled(torch, cfg.device),
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=max(0, int(cfg.num_workers)),
            pin_memory=_pin_memory_enabled(torch, cfg.device),
        )
        if val_ds
        else None
    )
    device = _resolve_torch_device(torch, str(cfg.device))
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = _make_grad_scaler(torch, enabled=use_amp)
    model = build_phoneme_boundary_detector(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    pos_weight = torch.full((len(model_cfg.labels),), float(cfg.pos_weight), dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_val: float | None = None
    best_state = None

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_frames = 0
        for batch_idx, (x, y, q, aux, mask) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            aux = {key: value.to(device, non_blocking=True) for key, value in aux.items()}
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, enabled=use_amp):
                out = model(x)
                raw = nn.functional.binary_cross_entropy_with_logits(
                    out["boundary_logits"],
                    y,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                denom = torch.clamp(mask.sum() * y.shape[-1], min=1.0)
                loss = (raw * mask[:, :, None]).sum() / denom
                if out.get("quality_logits") is not None and float(cfg.quality_loss_weight) > 0.0:
                    q_raw = nn.functional.binary_cross_entropy_with_logits(out["quality_logits"], q, reduction="none")
                    q_loss = (q_raw * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                    loss = loss + q_loss * float(cfg.quality_loss_weight)
                loss = loss + _phone_aux_loss(
                    out,
                    aux,
                    nn,
                    state_weight=float(cfg.phone_state_loss_weight),
                    consonant_weight=float(cfg.consonant_loss_weight),
                    vowel_weight=float(cfg.vowel_loss_weight),
                )
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
            total_loss += float(loss.detach().cpu().item()) * max(1, frames)
            total_frames += max(1, frames)
            if int(cfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(cfg.log_every) == 0):
                print(
                    f"[phoneme_boundary][train] epoch={epoch}/{int(cfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} loss={float(loss.detach().cpu().item()):.4f} "
                    f"device={device} amp={int(use_amp)}",
                    flush=True,
                )
        row = {"epoch": float(epoch), "train_loss": float(total_loss / max(1, total_frames))}
        if val_loader is not None:
            val_loss = _evaluate(
                model,
                val_loader,
                pos_weight,
                device,
                torch,
                nn,
                quality_weight=float(cfg.quality_loss_weight),
                state_weight=float(cfg.phone_state_loss_weight),
                consonant_weight=float(cfg.consonant_loss_weight),
                vowel_weight=float(cfg.vowel_loss_weight),
            )
            row["val_loss"] = float(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = float(val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_phoneme_boundary_checkpoint(
        output_path,
        model.cpu(),
        model_cfg,
        meta={
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "history": history,
            "device": str(device),
            "amp": bool(use_amp),
            "objective": "phoneme_boundary_frame_events",
            "uses_phone_aux_heads": bool(model_cfg.enable_phone_state_head or model_cfg.enable_phone_identity_heads),
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
    n_labels = int(batch[0][1].shape[1])
    xs = np.zeros((len(batch), max_len, n_mels), dtype=np.float32)
    ys = np.zeros((len(batch), max_len, n_labels), dtype=np.float32)
    qs = np.zeros((len(batch), max_len), dtype=np.float32)
    state_target = np.full((len(batch), max_len), IGNORE_INDEX, dtype=np.int64)
    state_mask = np.zeros((len(batch), max_len), dtype=np.float32)
    consonant_target = np.full((len(batch), max_len), IGNORE_INDEX, dtype=np.int64)
    consonant_mask = np.zeros((len(batch), max_len), dtype=np.float32)
    vowel_target = np.full((len(batch), max_len), IGNORE_INDEX, dtype=np.int64)
    vowel_mask = np.zeros((len(batch), max_len), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    for idx, (features, target, quality, phone_aux) in enumerate(batch):
        n = int(features.shape[0])
        xs[idx, :n, :] = features
        ys[idx, :n, :] = target
        qs[idx, :n] = quality
        state_target[idx, :n] = phone_aux.state_target[:n]
        state_mask[idx, :n] = phone_aux.state_mask[:n]
        consonant_target[idx, :n] = phone_aux.consonant_target[:n]
        consonant_mask[idx, :n] = phone_aux.consonant_mask[:n]
        vowel_target[idx, :n] = phone_aux.vowel_target[:n]
        vowel_mask[idx, :n] = phone_aux.vowel_mask[:n]
        mask[idx, :n] = 1.0
    aux = {
        "state_target": torch.from_numpy(state_target),
        "state_mask": torch.from_numpy(state_mask),
        "consonant_target": torch.from_numpy(consonant_target),
        "consonant_mask": torch.from_numpy(consonant_mask),
        "vowel_target": torch.from_numpy(vowel_target),
        "vowel_mask": torch.from_numpy(vowel_mask),
    }
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(qs), aux, torch.from_numpy(mask)


def _evaluate(
    model,
    loader,
    pos_weight,
    device,
    torch,
    nn,
    *,
    quality_weight: float,
    state_weight: float,
    consonant_weight: float,
    vowel_weight: float,
) -> float:
    model.eval()
    total = 0.0
    frames = 0
    with torch.no_grad():
        for x, y, q, aux, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            aux = {key: value.to(device, non_blocking=True) for key, value in aux.items()}
            mask = mask.to(device, non_blocking=True)
            out = model(x)
            raw = nn.functional.binary_cross_entropy_with_logits(
                out["boundary_logits"],
                y,
                reduction="none",
                pos_weight=pos_weight,
            )
            loss = (raw * mask[:, :, None]).sum() / torch.clamp(mask.sum() * y.shape[-1], min=1.0)
            if out.get("quality_logits") is not None and float(quality_weight) > 0.0:
                q_raw = nn.functional.binary_cross_entropy_with_logits(out["quality_logits"], q, reduction="none")
                loss = loss + ((q_raw * mask).sum() / torch.clamp(mask.sum(), min=1.0)) * float(quality_weight)
            loss = loss + _phone_aux_loss(
                out,
                aux,
                nn,
                state_weight=float(state_weight),
                consonant_weight=float(consonant_weight),
                vowel_weight=float(vowel_weight),
            )
            n = int(mask.sum().detach().cpu().item())
            total += float(loss.detach().cpu().item()) * max(1, n)
            frames += max(1, n)
    return float(total / max(1, frames))


def _phone_aux_loss(out, aux, nn, *, state_weight: float, consonant_weight: float, vowel_weight: float):
    loss = 0.0
    if out.get("phone_state_logits") is not None and float(state_weight) > 0.0:
        loss = loss + _masked_ce(
            nn,
            out["phone_state_logits"],
            aux["state_target"],
            aux["state_mask"],
        ) * float(state_weight)
    if out.get("consonant_logits") is not None and float(consonant_weight) > 0.0:
        loss = loss + _masked_ce(
            nn,
            out["consonant_logits"],
            aux["consonant_target"],
            aux["consonant_mask"],
        ) * float(consonant_weight)
    if out.get("vowel_logits") is not None and float(vowel_weight) > 0.0:
        loss = loss + _masked_ce(
            nn,
            out["vowel_logits"],
            aux["vowel_target"],
            aux["vowel_mask"],
        ) * float(vowel_weight)
    return loss


def _masked_ce(nn, logits, target, mask):
    if float(mask.sum().detach().cpu().item()) <= 0.0:
        return logits.sum() * 0.0
    raw = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(target.shape)
    return (raw * mask).sum() / mask.sum().clamp(min=1.0)


def _resolve_torch_device(torch, device: str):
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def _pin_memory_enabled(torch, device: str) -> bool:
    text = str(device or "auto").strip().lower()
    return bool((text in {"auto", "cuda"} or text.startswith("cuda")) and torch.cuda.is_available())


def _make_grad_scaler(torch, *, enabled: bool):
    if not enabled:
        return None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp.GradScaler(enabled=True)
    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
        return cuda_amp.GradScaler(enabled=True)
    return None


def _autocast(torch, *, enabled: bool):
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        try:
            return amp.autocast("cuda", enabled=True)
        except TypeError:
            return amp.autocast(enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


__all__ = [
    "PhonemeBoundaryDataset",
    "PhonemeBoundaryTrainConfig",
    "train_phoneme_boundary_from_manifest",
]
