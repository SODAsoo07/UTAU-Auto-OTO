from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_scorer_model import (
    BoundaryScorerConfig,
    build_boundary_scorer,
    save_boundary_checkpoint,
)
from core.coarse_crnn.boundary_targets import build_boundary_target_map, training_rows_to_wav_groups
from core.coarse_crnn.boundary_types import TRANSITION_ROLES
from core.coarse_crnn.training import resolve_torch_device


@dataclass
class BoundaryTrainConfig:
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
    quality_loss_weight: float = 0.06
    pos_weight: float = 2.5
    boundary_time_loss_weight: float = 0.08
    hard_case_oversample: bool = True
    hard_case_weight: float = 1.8
    hard_case_min_ratio: float = 0.10
    hard_case_alias_regex: str = ""


class _BoundaryDataset:
    def __init__(
        self,
        grouped_rows: dict[str, list[tuple[Any, Any]]],
        model_cfg: BoundaryScorerConfig,
        *,
        max_frames: int,
        train: bool,
        hard_case_weight: float = 1.8,
        hard_case_min_ratio: float = 0.10,
        hard_case_alias_regex: str = "",
    ):
        self.items = list(grouped_rows.items())
        self.cfg = model_cfg
        self.max_frames = int(max_frames)
        self.train = bool(train)
        self.sample_weights = _build_sample_weights(
            self.items,
            hard_case_weight=float(hard_case_weight),
            hard_case_min_ratio=float(hard_case_min_ratio),
            hard_case_alias_regex=str(hard_case_alias_regex or ""),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, rows = self.items[int(idx)]
        samples, sr, duration = load_wav_mono(wav_path, target_sr=int(self.cfg.sample_rate))
        features, hop_sec = log_mel_spectrogram(
            samples,
            sr,
            n_mels=int(self.cfg.n_mels),
            frame_ms=float(self.cfg.frame_ms),
            hop_ms=float(self.cfg.hop_ms),
        )
        frame_count = int(features.shape[0])
        duration_ms = max(1.0, float(duration) * 1000.0)
        _times, target = build_boundary_target_map(
            rows,
            duration_ms=duration_ms,
            hop_ms=float(hop_sec) * 1000.0,
            frame_count=frame_count,
        )
        if self.max_frames > 0 and frame_count > self.max_frames:
            if self.train:
                start = random.randint(0, frame_count - self.max_frames)
            else:
                start = 0
            end = start + self.max_frames
            features = features[start:end]
            target = target[start:end]
        return features.astype(np.float32), target.astype(np.float32)


def _collate(batch):
    torch = __import__("torch")
    n = len(batch)
    tmax = max(int(item[0].shape[0]) for item in batch)
    mels = int(batch[0][0].shape[1])
    labels = int(batch[0][1].shape[1])
    xs = np.zeros((n, tmax, mels), dtype=np.float32)
    ys = np.zeros((n, tmax, labels), dtype=np.float32)
    mask = np.zeros((n, tmax), dtype=np.float32)
    for idx, (x, y) in enumerate(batch):
        t = int(x.shape[0])
        xs[idx, :t, :] = x
        ys[idx, :t, :] = y
        mask[idx, :t] = 1.0
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(mask)


def train_boundary_from_manifest(
    rows: list[dict[str, Any]],
    output_path: str,
    *,
    train_config: BoundaryTrainConfig | None = None,
    model_config: BoundaryScorerConfig | None = None,
) -> dict[str, Any]:
    torch = __import__("torch")
    data = list(rows or [])
    if not data:
        raise ValueError("manifest has no rows")
    cfg = train_config or BoundaryTrainConfig()
    model_cfg = model_config or BoundaryScorerConfig()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    grouped = training_rows_to_wav_groups(data)
    if not grouped:
        raise ValueError("manifest has no valid wav groups")
    keys = list(grouped.keys())
    random.shuffle(keys)
    val_count = max(1, int(round(len(keys) * float(cfg.val_ratio)))) if len(keys) >= 10 else 0
    val_keys = set(keys[:val_count])
    train_groups = {k: grouped[k] for k in keys if k not in val_keys}
    val_groups = {k: grouped[k] for k in keys if k in val_keys}
    if not train_groups:
        train_groups = grouped
        val_groups = {}

    train_ds = _BoundaryDataset(
        train_groups,
        model_cfg,
        max_frames=int(cfg.max_frames),
        train=True,
        hard_case_weight=float(cfg.hard_case_weight),
        hard_case_min_ratio=float(cfg.hard_case_min_ratio),
        hard_case_alias_regex=str(cfg.hard_case_alias_regex or ""),
    )
    val_ds = _BoundaryDataset(val_groups, model_cfg, max_frames=int(cfg.max_frames), train=False) if val_groups else None
    sampler = None
    shuffle = True
    if bool(cfg.hard_case_oversample) and len(train_ds) > 1:
        weights = train_ds.sample_weights
        if any(float(w) > 1.0001 for w in weights):
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        collate_fn=_collate,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=True,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=max(0, int(cfg.num_workers)),
            pin_memory=True,
        )
        if val_ds
        else None
    )

    device = resolve_torch_device(torch, str(cfg.device))
    use_amp = bool(cfg.amp and device.type == "cuda")
    model = build_boundary_scorer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, float]] = []
    best_val = None
    best_state = None
    pos_weight = torch.tensor(float(cfg.pos_weight), device=device)

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_idx, (x, y, mask) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                logits = out["boundary_logits"]
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                denom = torch.clamp(mask.sum() * y.shape[-1], min=1.0)
                loss = (raw * mask[:, :, None]).sum() / denom
                q_logits = out.get("quality_logits")
                if q_logits is not None and float(cfg.quality_loss_weight) > 0.0:
                    q_target = torch.clamp(y.max(dim=2).values, 0.0, 1.0)
                    q_raw = torch.nn.functional.binary_cross_entropy_with_logits(q_logits, q_target, reduction="none")
                    q_loss = (q_raw * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                    loss = loss + (q_loss * float(cfg.quality_loss_weight))
                if float(cfg.boundary_time_loss_weight) > 0.0:
                    t_loss = _boundary_time_regression_loss(logits, y, mask, hop_ms=float(model_cfg.hop_ms))
                    loss = loss + (t_loss * float(cfg.boundary_time_loss_weight))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            frames = float(mask.sum().detach().cpu().item())
            loss_sum += float(loss.detach().cpu().item()) * max(1.0, frames)
            weight_sum += max(1.0, frames)
            if int(cfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(cfg.log_every) == 0):
                print(
                    f"[boundary][train] epoch={epoch}/{int(cfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f}",
                    flush=True,
                )
        row = {"epoch": float(epoch), "train_loss": float(loss_sum / max(1.0, weight_sum))}
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device=device, pos_weight=pos_weight)
            row["val_loss"] = float(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = float(val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_boundary_checkpoint(
        output_path,
        model.cpu(),
        model_cfg,
        meta={
            "history": history,
            "train_wavs": len(train_groups),
            "val_wavs": len(val_groups),
            "device": str(device),
            "hard_case_oversample": bool(cfg.hard_case_oversample),
            "hard_case_weight": float(cfg.hard_case_weight),
            "hard_case_min_ratio": float(cfg.hard_case_min_ratio),
            "boundary_time_loss_weight": float(cfg.boundary_time_loss_weight),
        },
    )
    return {
        "output_path": os.path.abspath(output_path),
        "history": history,
        "train_wavs": len(train_groups),
        "val_wavs": len(val_groups),
    }


def _evaluate(model, loader, *, device, pos_weight) -> float:
    torch = __import__("torch")
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(x)["boundary_logits"]
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
                pos_weight=pos_weight,
            )
            denom = torch.clamp(mask.sum() * y.shape[-1], min=1.0)
            loss = (raw * mask[:, :, None]).sum() / denom
            frames = float(mask.sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * max(1.0, frames)
            total_weight += max(1.0, frames)
    return float(total_loss / max(1.0, total_weight))


def _build_sample_weights(
    items: list[tuple[str, list[tuple[Any, Any]]]],
    *,
    hard_case_weight: float,
    hard_case_min_ratio: float,
    hard_case_alias_regex: str,
) -> list[float]:
    regex = None
    if str(hard_case_alias_regex or "").strip():
        try:
            regex = re.compile(str(hard_case_alias_regex), flags=re.IGNORECASE)
        except Exception:
            regex = None
    out: list[float] = []
    add_weight = max(0.0, float(hard_case_weight))
    min_ratio = max(0.0, min(1.0, float(hard_case_min_ratio)))
    for _wav_path, rows in items:
        if not rows:
            out.append(1.0)
            continue
        score = 0.0
        for spec, _anchors in rows:
            role = normalize_role(getattr(spec, "role", "other"))
            alias = str(getattr(spec, "alias", "") or "")
            meta = getattr(spec, "meta", {}) or {}
            alias_type = str(meta.get("alias_type", "") or "").strip().lower()
            transition_type = str(meta.get("transition_type", "") or "").strip().lower()
            row_hard = 0.0
            if role in TRANSITION_ROLES:
                row_hard += 1.0
            if alias_type in {"vc", "vv", "vcv"}:
                row_hard += 0.8
            if transition_type in {"vc", "vv"}:
                row_hard += 0.5
            if regex is not None and regex.search(alias):
                row_hard += 1.2
            score += row_hard
        ratio = score / max(1.0, float(len(rows)))
        if ratio < min_ratio:
            out.append(1.0)
        else:
            out.append(1.0 + (add_weight * ratio))
    return out


def _boundary_time_regression_loss(logits, target, mask, *, hop_ms: float):
    torch = __import__("torch")
    bsz, tmax, labels = logits.shape
    if tmax <= 1 or labels <= 0:
        return logits.new_tensor(0.0)
    time_idx = torch.arange(tmax, device=logits.device, dtype=logits.dtype).view(1, tmax, 1)
    valid_mask = mask[:, :, None]
    pred_prob = torch.sigmoid(logits) * valid_mask
    tgt_prob = target * valid_mask
    pred_mass = pred_prob.sum(dim=1)
    tgt_mass = tgt_prob.sum(dim=1)
    pred_center = (pred_prob * time_idx).sum(dim=1) / torch.clamp(pred_mass, min=1e-6)
    tgt_center = (tgt_prob * time_idx).sum(dim=1) / torch.clamp(tgt_mass, min=1e-6)
    valid = tgt_mass > 0.20
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    pred_ms = pred_center * float(hop_ms)
    tgt_ms = tgt_center * float(hop_ms)
    raw = torch.nn.functional.smooth_l1_loss(pred_ms, tgt_ms, reduction="none")
    return raw[valid].mean()


__all__ = ["BoundaryTrainConfig", "train_boundary_from_manifest"]
