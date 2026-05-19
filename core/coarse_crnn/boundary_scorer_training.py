from __future__ import annotations

import math
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
from core.coarse_crnn.boundary_targets import (
    build_boundary_target_map,
    build_phone_aware_target_map,
    compute_per_row_audio_extents,
    inject_audio_anchors_into_rows,
    training_rows_to_wav_groups,
)
from core.coarse_crnn.boundary_types import PHONE_AWARE_IGNORE_INDEX, TRANSITION_ROLES
from core.coarse_crnn.torch_utils import resolve_torch_device


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
    boundary_label_pos_weights: tuple[str, ...] = (
        "syllable_onset=1.55",
        "vowel_start=1.50",
        "consonant_onset=1.35",
        "next_onset=1.25",
    )
    boundary_time_loss_weight: float = 0.08
    hard_case_oversample: bool = True
    hard_case_weight: float = 1.8
    hard_case_min_ratio: float = 0.10
    hard_case_alias_regex: str = ""
    n_like_weight: float = 0.80
    weak_wav_weight: float = 0.65
    weak_wav_rms_db_threshold: float = -32.0
    weak_wav_auto_percentile: float = 0.15
    cvs_loss_weight: float = 0.0
    consonant_id_loss_weight: float = 0.0
    vowel_id_loss_weight: float = 0.0
    consonant_family_loss_weight: float = 0.0
    vowel_nucleus_loss_weight: float = 0.0
    vowel_glide_loss_weight: float = 0.0
    phone_aux_class_balanced: bool = False
    phone_aux_class_balance_power: float = 0.5
    phone_aux_class_balance_max: float = 8.0
    init_from: str = ""
    freeze_non_phone_aux: bool = False


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
        n_like_weight: float = 0.80,
        weak_wav_weight: float = 0.65,
        weak_wav_rms_db_threshold: float = -32.0,
        weak_wav_auto_percentile: float = 0.15,
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
            sample_rate=int(self.cfg.sample_rate),
            n_like_weight=float(n_like_weight),
            weak_wav_weight=float(weak_wav_weight),
            weak_wav_rms_db_threshold=float(weak_wav_rms_db_threshold),
            weak_wav_auto_percentile=float(weak_wav_auto_percentile),
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
        # Audio-driven syllable_onset / silence_boundary anchors. Computed
        # once per wav from the mel features and injected into a per-row copy
        # of the anchors so target builders see the natural acoustic edges
        # instead of the (often tight) source-OTO offsets/cutoffs.
        try:
            extents = compute_per_row_audio_extents(
                features=features,
                hop_ms=float(hop_sec) * 1000.0,
                rows=rows,
            )
            rows_for_targets = inject_audio_anchors_into_rows(rows, extents)
        except Exception:
            rows_for_targets = rows
        _times, target = build_boundary_target_map(
            rows_for_targets,
            duration_ms=duration_ms,
            hop_ms=float(hop_sec) * 1000.0,
            frame_count=frame_count,
        )
        phone_target = None
        if bool(getattr(self.cfg, "enable_phone_aux_heads", False)) or bool(getattr(self.cfg, "enable_phone_family_heads", False)):
            _phone_times, phone_target = build_phone_aware_target_map(
                rows_for_targets,
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
            if phone_target is not None:
                phone_target = phone_target.__class__(
                    cvs_target=phone_target.cvs_target[start:end],
                    cvs_mask=phone_target.cvs_mask[start:end],
                    consonant_target=phone_target.consonant_target[start:end],
                    consonant_mask=phone_target.consonant_mask[start:end],
                    vowel_target=phone_target.vowel_target[start:end],
                    vowel_mask=phone_target.vowel_mask[start:end],
                    consonant_family_target=phone_target.consonant_family_target[start:end],
                    consonant_family_mask=phone_target.consonant_family_mask[start:end],
                    vowel_nucleus_target=phone_target.vowel_nucleus_target[start:end],
                    vowel_nucleus_mask=phone_target.vowel_nucleus_mask[start:end],
                    vowel_glide_target=phone_target.vowel_glide_target[start:end],
                    vowel_glide_mask=phone_target.vowel_glide_mask[start:end],
                )
        if phone_target is None:
            return features.astype(np.float32), target.astype(np.float32)
        return features.astype(np.float32), target.astype(np.float32), phone_target


def _collate(batch):
    torch = __import__("torch")
    n = len(batch)
    tmax = max(int(item[0].shape[0]) for item in batch)
    mels = int(batch[0][0].shape[1])
    labels = int(batch[0][1].shape[1])
    has_phone = len(batch[0]) >= 3 and batch[0][2] is not None
    xs = np.zeros((n, tmax, mels), dtype=np.float32)
    ys = np.zeros((n, tmax, labels), dtype=np.float32)
    mask = np.zeros((n, tmax), dtype=np.float32)
    if has_phone:
        cvs_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        cvs_mask = np.zeros((n, tmax), dtype=np.float32)
        consonant_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        consonant_mask = np.zeros((n, tmax), dtype=np.float32)
        vowel_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        vowel_mask = np.zeros((n, tmax), dtype=np.float32)
        consonant_family_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        consonant_family_mask = np.zeros((n, tmax), dtype=np.float32)
        vowel_nucleus_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        vowel_nucleus_mask = np.zeros((n, tmax), dtype=np.float32)
        vowel_glide_target = np.full((n, tmax), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
        vowel_glide_mask = np.zeros((n, tmax), dtype=np.float32)
    for idx, item in enumerate(batch):
        x, y = item[0], item[1]
        t = int(x.shape[0])
        xs[idx, :t, :] = x
        ys[idx, :t, :] = y
        mask[idx, :t] = 1.0
        if has_phone:
            phone = item[2]
            cvs_target[idx, :t] = phone.cvs_target[:t]
            cvs_mask[idx, :t] = phone.cvs_mask[:t]
            consonant_target[idx, :t] = phone.consonant_target[:t]
            consonant_mask[idx, :t] = phone.consonant_mask[:t]
            vowel_target[idx, :t] = phone.vowel_target[:t]
            vowel_mask[idx, :t] = phone.vowel_mask[:t]
            consonant_family_target[idx, :t] = phone.consonant_family_target[:t]
            consonant_family_mask[idx, :t] = phone.consonant_family_mask[:t]
            vowel_nucleus_target[idx, :t] = phone.vowel_nucleus_target[:t]
            vowel_nucleus_mask[idx, :t] = phone.vowel_nucleus_mask[:t]
            vowel_glide_target[idx, :t] = phone.vowel_glide_target[:t]
            vowel_glide_mask[idx, :t] = phone.vowel_glide_mask[:t]
    if not has_phone:
        return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(mask)
    return (
        torch.from_numpy(xs),
        torch.from_numpy(ys),
        torch.from_numpy(mask),
        {
            "cvs_target": torch.from_numpy(cvs_target),
            "cvs_mask": torch.from_numpy(cvs_mask),
            "consonant_target": torch.from_numpy(consonant_target),
            "consonant_mask": torch.from_numpy(consonant_mask),
            "vowel_target": torch.from_numpy(vowel_target),
            "vowel_mask": torch.from_numpy(vowel_mask),
            "consonant_family_target": torch.from_numpy(consonant_family_target),
            "consonant_family_mask": torch.from_numpy(consonant_family_mask),
            "vowel_nucleus_target": torch.from_numpy(vowel_nucleus_target),
            "vowel_nucleus_mask": torch.from_numpy(vowel_nucleus_mask),
            "vowel_glide_target": torch.from_numpy(vowel_glide_target),
            "vowel_glide_mask": torch.from_numpy(vowel_glide_mask),
        },
    )


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
        n_like_weight=float(cfg.n_like_weight),
        weak_wav_weight=float(cfg.weak_wav_weight),
        weak_wav_rms_db_threshold=float(cfg.weak_wav_rms_db_threshold),
        weak_wav_auto_percentile=float(cfg.weak_wav_auto_percentile),
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
    init_report = _load_matching_state_from_checkpoint(model, str(cfg.init_from or "").strip()) if str(cfg.init_from or "").strip() else {}
    freeze_report = _freeze_non_phone_aux_params(model) if bool(cfg.freeze_non_phone_aux) else {}
    if bool(cfg.freeze_non_phone_aux) and max(
        float(cfg.cvs_loss_weight),
        float(cfg.consonant_id_loss_weight),
        float(cfg.vowel_id_loss_weight),
        float(cfg.consonant_family_loss_weight),
        float(cfg.vowel_nucleus_loss_weight),
        float(cfg.vowel_glide_loss_weight),
    ) <= 0.0:
        raise ValueError("--freeze-non-phone-aux requires a positive phone auxiliary loss weight")
    trainable_params = [param for param in model.parameters() if bool(getattr(param, "requires_grad", True))]
    if not trainable_params:
        raise ValueError("boundary scorer has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, float]] = []
    best_val = None
    best_state = None
    pos_weight = _boundary_pos_weight_tensor(
        torch,
        labels=tuple(model_cfg.labels),
        base_pos_weight=float(cfg.pos_weight),
        label_weights=tuple(cfg.boundary_label_pos_weights or ()),
        device=device,
    )

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y, mask, phone_aux = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            phone_aux = _move_phone_aux(phone_aux, device=device)
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
                aux_loss = _phone_aux_loss(out, phone_aux, mask=mask, cfg=cfg)
                if aux_loss is not None:
                    loss = loss + aux_loss
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
            val_loss = _evaluate(model, val_loader, device=device, pos_weight=pos_weight, cfg=cfg)
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
            "n_like_weight": float(cfg.n_like_weight),
            "weak_wav_weight": float(cfg.weak_wav_weight),
            "weak_wav_rms_db_threshold": float(cfg.weak_wav_rms_db_threshold),
            "weak_wav_auto_percentile": float(cfg.weak_wav_auto_percentile),
            "pos_weight": float(cfg.pos_weight),
            "boundary_label_pos_weights": list(cfg.boundary_label_pos_weights or ()),
            "boundary_time_loss_weight": float(cfg.boundary_time_loss_weight),
            "phone_aux_enabled": bool(model_cfg.enable_phone_aux_heads),
            "phone_family_enabled": bool(model_cfg.enable_phone_family_heads),
            "cvs_loss_weight": float(cfg.cvs_loss_weight),
            "consonant_id_loss_weight": float(cfg.consonant_id_loss_weight),
            "vowel_id_loss_weight": float(cfg.vowel_id_loss_weight),
            "consonant_family_loss_weight": float(cfg.consonant_family_loss_weight),
            "vowel_nucleus_loss_weight": float(cfg.vowel_nucleus_loss_weight),
            "vowel_glide_loss_weight": float(cfg.vowel_glide_loss_weight),
            "phone_aux_class_balanced": bool(cfg.phone_aux_class_balanced),
            "phone_aux_class_balance_power": float(cfg.phone_aux_class_balance_power),
            "phone_aux_class_balance_max": float(cfg.phone_aux_class_balance_max),
            "init_from": str(cfg.init_from or ""),
            "init_report": dict(init_report or {}),
            "freeze_non_phone_aux": bool(cfg.freeze_non_phone_aux),
            "freeze_report": dict(freeze_report or {}),
        },
    )
    return {
        "output_path": os.path.abspath(output_path),
        "history": history,
        "train_wavs": len(train_groups),
        "val_wavs": len(val_groups),
        "init_report": dict(init_report or {}),
        "freeze_report": dict(freeze_report or {}),
    }


def _evaluate(model, loader, *, device, pos_weight, cfg: BoundaryTrainConfig | None = None) -> float:
    torch = __import__("torch")
    train_cfg = cfg or BoundaryTrainConfig()
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y, mask, phone_aux = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            phone_aux = _move_phone_aux(phone_aux, device=device)
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
            aux_loss = _phone_aux_loss(out, phone_aux, mask=mask, cfg=train_cfg)
            if aux_loss is not None:
                loss = loss + aux_loss
            frames = float(mask.sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * max(1.0, frames)
            total_weight += max(1.0, frames)
    return float(total_loss / max(1.0, total_weight))


def _boundary_pos_weight_tensor(
    torch,
    *,
    labels: tuple[str, ...],
    base_pos_weight: float,
    label_weights: tuple[str, ...],
    device,
):
    base = max(0.01, float(base_pos_weight))
    weights = [base for _label in labels]
    multipliers = _parse_label_weight_overrides(label_weights)
    for idx, label in enumerate(labels):
        key = str(label or "").strip().lower()
        if key in multipliers:
            weights[idx] = base * float(multipliers[key])
    return torch.tensor(weights, dtype=torch.float32, device=device).view(1, 1, len(weights))


def _parse_label_weight_overrides(items: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
        elif ":" in text:
            key, value = text.split(":", 1)
        else:
            continue
        key = key.strip().lower()
        if not key:
            continue
        try:
            out[key] = max(0.05, min(8.0, float(value)))
        except Exception:
            continue
    return out


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        return batch[0], batch[1], batch[2], batch[3]
    return batch[0], batch[1], batch[2], None


def _move_phone_aux(phone_aux, *, device):
    if not isinstance(phone_aux, dict):
        return None
    return {key: value.to(device, non_blocking=True) for key, value in phone_aux.items()}


def _load_matching_state_from_checkpoint(model, path: str) -> dict[str, Any]:
    if not path:
        return {}
    torch = __import__("torch")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"init checkpoint must be a dict: {path}")
    source = payload.get("state_dict")
    if not isinstance(source, dict):
        raise ValueError(f"init checkpoint missing state_dict: {path}")
    current = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        if key not in current:
            skipped.append(str(key))
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append(str(key))
            continue
        current[key] = value.detach().cpu().clone()
        loaded.append(str(key))
    loaded_set = set(loaded)
    missing_target = [str(key) for key in current.keys() if str(key) not in loaded_set]
    model.load_state_dict(current, strict=True)
    return {
        "path": os.path.abspath(str(path)),
        "loaded_keys": int(len(loaded)),
        "skipped_keys": int(len(skipped)),
        "missing_target_keys": int(len(missing_target)),
        "loaded_key_names": loaded[:24],
        "skipped_key_names": skipped[:24],
        "missing_target_key_names": missing_target[:24],
    }


def _freeze_non_phone_aux_params(model) -> dict[str, Any]:
    phone_prefixes = (
        "cvs_head.",
        "consonant_head.",
        "vowel_head.",
        "consonant_family_head.",
        "vowel_nucleus_head.",
        "vowel_glide_head.",
    )
    trainable: list[str] = []
    frozen: list[str] = []
    for name, param in model.named_parameters():
        keep_trainable = str(name).startswith(phone_prefixes)
        param.requires_grad = bool(keep_trainable)
        if keep_trainable:
            trainable.append(str(name))
        else:
            frozen.append(str(name))
    if not trainable:
        raise ValueError("--freeze-non-phone-aux requires --enable-phone-aux-heads")
    return {
        "trainable_keys": int(len(trainable)),
        "frozen_keys": int(len(frozen)),
        "trainable_key_names": trainable,
        "frozen_key_names": frozen[:24],
    }


def _phone_aux_loss(out: dict[str, Any], phone_aux, *, mask, cfg: BoundaryTrainConfig):
    torch = __import__("torch")
    if not isinstance(phone_aux, dict):
        return None
    total = None
    for logits_key, target_key, mask_key, weight in (
        ("cvs_logits", "cvs_target", "cvs_mask", float(cfg.cvs_loss_weight)),
        ("consonant_logits", "consonant_target", "consonant_mask", float(cfg.consonant_id_loss_weight)),
        ("vowel_logits", "vowel_target", "vowel_mask", float(cfg.vowel_id_loss_weight)),
        (
            "consonant_family_logits",
            "consonant_family_target",
            "consonant_family_mask",
            float(cfg.consonant_family_loss_weight),
        ),
        (
            "vowel_nucleus_logits",
            "vowel_nucleus_target",
            "vowel_nucleus_mask",
            float(cfg.vowel_nucleus_loss_weight),
        ),
        (
            "vowel_glide_logits",
            "vowel_glide_target",
            "vowel_glide_mask",
            float(cfg.vowel_glide_loss_weight),
        ),
    ):
        if weight <= 0.0:
            continue
        logits = out.get(logits_key)
        if logits is None:
            continue
        target = phone_aux.get(target_key)
        aux_mask = phone_aux.get(mask_key)
        if target is None or aux_mask is None:
            continue
        valid = (aux_mask > 0.5) & (mask > 0.5) & (target >= 0) & (target < int(logits.shape[-1]))
        if not bool(valid.any()):
            continue
        loss = _masked_phone_cross_entropy(
            logits[valid],
            target[valid].long(),
            class_balanced=bool(cfg.phone_aux_class_balanced),
            balance_power=float(cfg.phone_aux_class_balance_power),
            max_weight=float(cfg.phone_aux_class_balance_max),
        ) * float(weight)
        total = loss if total is None else total + loss
    return total


def _masked_phone_cross_entropy(
    logits,
    target,
    *,
    class_balanced: bool,
    balance_power: float,
    max_weight: float,
):
    torch = __import__("torch")
    if not bool(class_balanced) or int(target.numel()) <= 0:
        return torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    classes = int(logits.shape[-1])
    # Compute class-balance weights in fp32 regardless of autocast: bincount /
    # pow / clamp mixes Python scalars (Float) with tensors and emits Float, so
    # holding `weights` as Half (under AMP) triggers an index_put dtype error.
    counts = torch.bincount(target.detach().long(), minlength=classes).to(
        device=logits.device, dtype=torch.float32
    )
    nonzero = counts > 0
    if not bool(nonzero.any()):
        return torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    mean_count = counts[nonzero].mean()
    weights = torch.ones((classes,), device=logits.device, dtype=torch.float32)
    power = max(0.0, min(2.0, float(balance_power)))
    max_w = max(1.0, float(max_weight))
    weights[nonzero] = torch.pow(mean_count / torch.clamp(counts[nonzero], min=1.0), power)
    weights = torch.clamp(weights, min=0.25, max=max_w).to(dtype=logits.dtype)
    return torch.nn.functional.cross_entropy(logits, target, reduction="mean", weight=weights)


def _build_sample_weights(
    items: list[tuple[str, list[tuple[Any, Any]]]],
    *,
    hard_case_weight: float,
    hard_case_min_ratio: float,
    hard_case_alias_regex: str,
    sample_rate: int,
    n_like_weight: float,
    weak_wav_weight: float,
    weak_wav_rms_db_threshold: float,
    weak_wav_auto_percentile: float,
) -> list[float]:
    regex = None
    if str(hard_case_alias_regex or "").strip():
        try:
            regex = re.compile(str(hard_case_alias_regex), flags=re.IGNORECASE)
        except Exception:
            regex = None
    wav_rms_db: dict[str, float] = {}
    for wav_path, _rows in items:
        wav_rms_db[str(wav_path)] = _wav_rms_db(str(wav_path), sample_rate=int(sample_rate))
    weak_thr = float(weak_wav_rms_db_threshold)
    if float(weak_wav_auto_percentile) > 0.0:
        db_values = [float(v) for v in wav_rms_db.values() if np.isfinite(v)]
        if db_values:
            auto_thr = float(np.quantile(np.asarray(db_values, dtype=np.float32), float(max(0.0, min(1.0, weak_wav_auto_percentile)))))
            weak_thr = min(float(weak_thr), float(auto_thr))
    out: list[float] = []
    add_weight = max(0.0, float(hard_case_weight))
    n_like_add = max(0.0, float(n_like_weight))
    weak_wav_add = max(0.0, float(weak_wav_weight))
    min_ratio = max(0.0, min(1.0, float(hard_case_min_ratio)))
    for wav_path, rows in items:
        if not rows:
            out.append(1.0)
            continue
        wav_db = float(wav_rms_db.get(str(wav_path), float("nan")))
        weak_wav = bool(np.isfinite(wav_db) and wav_db <= float(weak_thr))
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
            if _is_n_like_alias(alias, meta=meta):
                row_hard += n_like_add
            # P2-a: bumped 1.2 竊・2.0. With the default coda-bridge regex
            # (set in train_boundary_oto.py), this lifts the 14.7k KO coda-
            # bridge rows to ~3ﾃ・the weight of generic transitions 窶・the
            # 2026-05-17 wall analysis showed they need disproportionate
            # supervision because the model decoder treats their plosive
            # closure as the boundary instead of the burst.
            if regex is not None and regex.search(alias):
                row_hard += 2.0
            # P2-c: right-onset class differentiation. Plosive/aspirate
            # right-onsets in coda-bridge aliases (``L p``, ``NG H``,
            # ``M b``, ``L k``) cause 竏・00~竏・700 ms early decoding bias
            # vs sonorant/fricative which stay within ﾂｱ300 ms. Add extra
            # supervision specifically for those patterns.
            row_hard += _coda_bridge_onset_bonus(alias)
            if weak_wav:
                row_hard += weak_wav_add
            score += row_hard
        ratio = score / max(1.0, float(len(rows)))
        if ratio < min_ratio:
            out.append(1.0)
        else:
            out.append(1.0 + (add_weight * ratio))
    return out


_KO_CODA_LEFT_TOKENS: frozenset[str] = frozenset({
    "n", "m", "l", "ng",
    "N", "M", "L", "NG", "H", "T", "P", "K", "R",
})
# Plosive/aspirate right-onsets cause the worst decoding bias on coda-bridge
# (2026-05-17 wall analysis: 竏・00~竏・700 ms early). Includes basic + tense +
# aspirated stops + ``h``/``H`` + ``R``. Sonorants/fricatives intentionally
# excluded 窶・they decode within ﾂｱ300 ms without extra supervision.
_KO_HARD_RIGHT_ONSETS: frozenset[str] = frozenset({
    "g", "gg", "k", "kk", "kh",
    "d", "dd", "t", "tt", "th",
    "b", "bb", "p", "pp", "ph",
    "h", "H", "R",
})


def _coda_bridge_onset_bonus(alias: str) -> float:
    """Extra hard-case score for KO coda-bridge with plosive/aspirate onset.

    Returns 1.5 when the alias matches `<KO_coda> <plosive_or_aspirate>`,
    otherwise 0.0. Designed to overweight specifically the patterns that
    underperformed in the 2026-05-17 raw-model diagnostic 窶・keeping the
    bonus narrow avoids inflating already-easy categories.
    """
    text = str(alias or "").strip()
    parts = text.split()
    if len(parts) != 2:
        return 0.0
    left = parts[0].split("_", 1)[0]
    if left not in _KO_CODA_LEFT_TOKENS:
        return 0.0
    right = parts[1].split("_", 1)[0]
    if not right:
        return 0.0
    # Match longest first so ``gg``/``kk``/``tt`` etc. are detected before
    # falling through to single-char ``g``/``k``/``t``.
    for token in sorted(_KO_HARD_RIGHT_ONSETS, key=len, reverse=True):
        if right.startswith(token):
            return 1.5
    return 0.0


def _is_n_like_alias(alias: str, *, meta: dict[str, Any]) -> bool:
    right_c = str(meta.get("right_consonant", "") or "").strip().lower()
    if _is_n_like_token(right_c):
        return True
    tokens = [token for token in str(alias or "").strip().lower().split() if token]
    return any(_is_n_like_token(token) for token in tokens)


def _is_n_like_token(token: str) -> bool:
    text = str(token or "").strip().lower()
    if not text:
        return False
    if text in {"n", "ng", "m", "l", "r"}:
        return True
    if text.startswith(("n", "ng", "m", "l", "r")) and len(text) <= 3:
        return True
    return False


def _wav_rms_db(wav_path: str, *, sample_rate: int) -> float:
    try:
        samples, _sr, _dur = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        return float("nan")
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(arr))))
    return float(20.0 * math.log10(max(rms, 1e-8)))


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

