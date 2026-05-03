from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.oto_model import (
    OtoCrnnConfig,
    alias_type_id,
    build_oto_model,
    format_id,
    language_id,
    save_oto_checkpoint,
    transition_type_id,
    uses_relative_param_head,
    right_boundary_prior_blend_for,
)
from core.coarse_crnn.oto_param_priors import decode_relative_oto_params, normalize_relative_oto_target, relative_params_to_anchors
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES
from core.coarse_crnn.oto_windowing import crop_oto_target_window, row_window_args, should_use_vcv_target_window, target_window_frames_for
from core.coarse_crnn.training import _autocast, _make_grad_scaler, _pin_memory_enabled, resolve_torch_device


@dataclass
class OtoTrainConfig:
    epochs: int = 4
    lr: float = 1e-3
    batch_size: int = 8
    max_frames: int = 1200
    seed: int = 1337
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    log_every: int = 100
    val_ratio: float = 0.08
    heatmap_sigma_frames: float = 2.0
    heatmap_loss_weight: float = 1.0
    scalar_loss_weight: float = 0.55
    order_loss_weight: float = 0.08
    vcv_loss_weight: float = 1.35
    cvvc_loss_weight: float = 1.15
    cvc_loss_weight: float = 1.05


class OtoAnchorDataset:
    def __init__(self, rows: list[dict[str, Any]], model_config: OtoCrnnConfig, *, train_config: OtoTrainConfig, train: bool):
        self.rows = list(rows or [])
        self.model_config = model_config
        self.train_config = train_config
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        wav_path = str(row.get("audio", "") or "")
        samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=int(self.model_config.sample_rate))
        features, hop_sec = log_mel_spectrogram(
            samples,
            sr,
            n_mels=int(self.model_config.n_mels),
            frame_ms=float(self.model_config.frame_ms),
            hop_ms=float(self.model_config.hop_ms),
        )
        if features.shape[0] <= 0:
            features = np.zeros((1, int(self.model_config.n_mels)), dtype=np.float32)
        anchors_ms = _anchor_array_from_row(row)
        full_duration_ms = float(duration_sec) * 1000.0
        if should_use_vcv_target_window(
            row.get("format_type", ""),
            enabled=bool(self.model_config.enable_vcv_target_window),
            formats=tuple(getattr(self.model_config, "target_window_formats", ("vcv",))),
            alias_type=row.get("alias_type", ""),
            cvvc_alias_types=tuple(getattr(self.model_config, "cvvc_target_window_alias_types", ("vc", "vv"))),
        ):
            row_index, row_count = row_window_args(row)
            features, anchors_ms_or_none, duration_ms, _start_frame = crop_oto_target_window(
                features,
                anchors_ms,
                hop_sec=float(hop_sec),
                duration_ms=full_duration_ms,
                row_index_in_wav=row_index,
                file_row_count=row_count,
                window_frames=target_window_frames_for(
                    self.model_config,
                    row.get("format_type", ""),
                    int(self.model_config.vcv_target_window_frames),
                ),
            )
            anchors_ms = np.asarray(anchors_ms_or_none, dtype=np.float32)
        else:
            features, anchors_ms, duration_ms = _crop_around_anchors(
                features,
                anchors_ms,
                hop_sec=float(hop_sec),
                duration_ms=full_duration_ms,
                max_frames=int(self.train_config.max_frames),
                train=self.train,
            )
        heatmap = _make_anchor_heatmap(
            anchors_ms,
            frame_count=int(features.shape[0]),
            hop_sec=float(hop_sec),
            sigma_frames=float(self.train_config.heatmap_sigma_frames),
        )
        if uses_relative_param_head(self.model_config):
            scalar = np.asarray(
                normalize_relative_oto_target(
                    anchors_ms,
                    duration_ms=float(duration_ms),
                    format_type=row.get("format_type", ""),
                    alias_type=row.get("alias_type", ""),
                    transition_type=row.get("transition_type", ""),
                ),
                dtype=np.float32,
            )
        else:
            scalar = np.clip(anchors_ms / max(float(duration_ms), 1.0), 0.0, 1.0).astype(np.float32)
        language = language_id(self.model_config, row.get("language", ""))
        fmt = format_id(self.model_config, row.get("format_type", ""))
        alias_id = alias_type_id(self.model_config, row.get("alias_type", ""))
        transition_id = transition_type_id(self.model_config, row.get("transition_type", ""))
        prev_alias_id = alias_type_id(self.model_config, row.get("prev_alias_type", ""))
        next_alias_id = alias_type_id(self.model_config, row.get("next_alias_type", ""))
        prev_transition_id = transition_type_id(self.model_config, row.get("prev_transition_type", ""))
        next_transition_id = transition_type_id(self.model_config, row.get("next_transition_type", ""))
        context = _context_array_from_row(row)
        weight = float(row.get("sample_weight", row.get("weight", 1.0)) or 1.0)
        weight *= _format_loss_multiplier(row, self.train_config)
        return (
            features.astype(np.float32),
            heatmap.astype(np.float32),
            scalar,
            anchors_ms.astype(np.float32),
            float(duration_ms),
            int(language),
            int(fmt),
            context.astype(np.float32),
            int(alias_id),
            int(transition_id),
            int(prev_alias_id),
            int(next_alias_id),
            int(prev_transition_id),
            int(next_transition_id),
            max(0.05, min(2.0, weight)),
        )


def train_oto_from_manifest(
    rows: list[dict[str, Any]],
    output_path: str,
    *,
    val_rows: list[dict[str, Any]] | None = None,
    train_config: OtoTrainConfig | None = None,
    model_config: OtoCrnnConfig | None = None,
) -> dict[str, Any]:
    torch = __import__("torch")
    nn = __import__("torch.nn").nn
    data = [row for row in rows if str(row.get("audio", "") or "")]
    if not data:
        raise ValueError("OTO manifest has no training rows")
    fixed_val_rows = [row for row in (val_rows or []) if str(row.get("audio", "") or "")]
    cfg = train_config or OtoTrainConfig()
    model_cfg = model_config or OtoCrnnConfig()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    random.shuffle(data)
    if fixed_val_rows:
        val_rows_final = fixed_val_rows
        train_rows = data
    else:
        val_count = max(1, int(round(len(data) * float(cfg.val_ratio)))) if len(data) >= 10 else 0
        val_rows_final = data[:val_count]
        train_rows = data[val_count:] if val_count else data
    if not train_rows:
        train_rows = data
        val_rows_final = []

    train_ds = OtoAnchorDataset(train_rows, model_cfg, train_config=cfg, train=True)
    val_ds = OtoAnchorDataset(val_rows_final, model_cfg, train_config=cfg, train=False) if val_rows_final else None
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
            batch_size=max(1, min(8, int(cfg.batch_size))),
            shuffle=False,
            collate_fn=_collate,
            num_workers=max(0, int(cfg.num_workers)),
            pin_memory=_pin_memory_enabled(torch, cfg.device),
        )
        if val_ds
        else None
    )

    device = resolve_torch_device(torch, str(cfg.device))
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = _make_grad_scaler(torch, enabled=use_amp)
    model = build_oto_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_val = None
    best_state = None
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        row_sum = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, heat, scalar, _anchors_ms, _duration_ms, lang, fmt, context, alias_id, transition_id, prev_alias_id, next_alias_id, prev_transition_id, next_transition_id, weight, mask = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, enabled=use_amp):
                outputs = model(
                    x,
                    lang,
                    fmt,
                    context,
                    alias_id,
                    transition_id,
                    prev_alias_id,
                    next_alias_id,
                    prev_transition_id,
                    next_transition_id,
                )
                loss = _oto_loss(outputs, heat, scalar, weight, mask, nn, cfg, relative_scalar=uses_relative_param_head(model_cfg))
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
            rows_in_batch = int(x.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * rows_in_batch
            row_sum += rows_in_batch
            if int(cfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(cfg.log_every) == 0):
                print(
                    f"[oto_anchor][train] epoch={epoch}/{int(cfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f} "
                    f"device={device} amp={int(use_amp)}",
                    flush=True,
                )
        row = {"epoch": float(epoch), "train_loss": float(loss_sum / max(1, row_sum))}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device, nn, cfg, model_cfg)
            row.update(val_metrics)
            val_loss = float(val_metrics["val_loss"])
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_oto_checkpoint(
        output_path,
        model.cpu(),
        model_cfg,
        meta={
            "train_rows": len(train_rows),
            "val_rows": len(val_rows_final),
            "fixed_val_manifest": bool(fixed_val_rows),
            "history": history,
            "device": str(device),
            "amp": bool(use_amp),
        },
    )
    return {
        "output_path": os.path.abspath(output_path),
        "history": history,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows_final),
        "device": str(device),
        "amp": bool(use_amp),
    }


def _evaluate(model, loader, device, nn, cfg: OtoTrainConfig, model_cfg: OtoCrnnConfig) -> dict[str, float]:
    torch = __import__("torch")
    model.eval()
    loss_sum = 0.0
    row_sum = 0
    mae_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x, heat, scalar, anchors_ms, duration_ms, lang, fmt, context, alias_id, transition_id, prev_alias_id, next_alias_id, prev_transition_id, next_transition_id, weight, mask = _move_batch(batch, device)
            outputs = model(
                x,
                lang,
                fmt,
                context,
                alias_id,
                transition_id,
                prev_alias_id,
                next_alias_id,
                prev_transition_id,
                next_transition_id,
            )
            relative_scalar = uses_relative_param_head(model_cfg)
            loss = _oto_loss(outputs, heat, scalar, weight, mask, nn, cfg, relative_scalar=relative_scalar)
            pred_scalar = torch.sigmoid(outputs["scalar_logits"])
            if relative_scalar:
                mae_values.extend(
                    _relative_anchor_mae_values(
                        pred_scalar.detach().cpu().numpy(),
                        duration_ms.detach().cpu().numpy(),
                        fmt.detach().cpu().numpy(),
                        alias_id.detach().cpu().numpy(),
                        transition_id.detach().cpu().numpy(),
                        anchors_ms.detach().cpu().numpy(),
                        model_cfg,
                    )
                )
            else:
                pred_ms = pred_scalar * duration_ms[:, None]
                mae = torch.abs(pred_ms - anchors_ms).mean(dim=1)
                mae_values.extend(float(v) for v in mae.detach().cpu().tolist())
            rows_in_batch = int(x.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * rows_in_batch
            row_sum += rows_in_batch
    return {
        "val_loss": float(loss_sum / max(1, row_sum)),
        "val_anchor_mae_ms": float(sum(mae_values) / max(1, len(mae_values))),
    }


def _oto_loss(outputs, heat, scalar, weight, mask, nn, cfg: OtoTrainConfig, *, relative_scalar: bool = False):
    torch = __import__("torch")
    pred_heat = torch.sigmoid(outputs["heatmap_logits"])
    heat_loss_raw = (pred_heat - heat) ** 2
    heat_denom = torch.clamp(mask[:, :, None].sum() * heat.shape[-1], min=1.0)
    heat_loss = (heat_loss_raw * mask[:, :, None] * weight[:, None, None]).sum() / heat_denom
    scalar_pred = torch.sigmoid(outputs["scalar_logits"])
    scalar_loss_raw = nn.functional.smooth_l1_loss(scalar_pred, scalar, reduction="none")
    scalar_loss = (scalar_loss_raw.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    order_loss = _relative_order_penalty(scalar_pred) if relative_scalar else _order_penalty(scalar_pred)
    return (
        heat_loss * float(cfg.heatmap_loss_weight)
        + scalar_loss * float(cfg.scalar_loss_weight)
        + order_loss * float(cfg.order_loss_weight)
    )


def _order_penalty(pred):
    torch = __import__("torch")
    # order: offset <= overlap <= preutterance <= consonant <= cutoff
    diffs = pred[:, :-1] - pred[:, 1:]
    return torch.relu(diffs).mean()


def _relative_order_penalty(pred):
    torch = __import__("torch")
    # Relative target order only requires overlap_delta <= pre_delta.
    return torch.relu(pred[:, 1] - pred[:, 2]).mean()


def _relative_anchor_mae_values(pred_scalar, duration_ms, format_ids, alias_ids, transition_ids, anchors_ms, model_cfg: OtoCrnnConfig) -> list[float]:
    values: list[float] = []
    formats = list(model_cfg.format_types)
    aliases = list(model_cfg.alias_types)
    transitions = list(model_cfg.transition_types)
    for scalar_row, duration, fmt_idx, alias_idx, transition_idx, anchor_row in zip(pred_scalar, duration_ms, format_ids, alias_ids, transition_ids, anchors_ms):
        fmt_i = int(fmt_idx)
        alias_i = int(alias_idx)
        transition_i = int(transition_idx)
        format_type = formats[fmt_i] if 0 <= fmt_i < len(formats) else "other"
        alias_type = aliases[alias_i] if 0 <= alias_i < len(aliases) else "other"
        transition_type = transitions[transition_i] if 0 <= transition_i < len(transitions) else "other"
        params = decode_relative_oto_params(
            scalar_row,
            duration_ms=float(duration),
            format_type=format_type,
            alias_type=alias_type,
            transition_type=transition_type,
            prior_blend=right_boundary_prior_blend_for(model_cfg, format_type),
        )
        pred_anchors = np.asarray(relative_params_to_anchors(params, duration_ms=float(duration)), dtype=np.float32)
        values.append(float(np.abs(pred_anchors - np.asarray(anchor_row, dtype=np.float32)).mean()))
    return values


def _collate(batch):
    torch = __import__("torch")
    max_len = max(int(item[0].shape[0]) for item in batch)
    n_mels = int(batch[0][0].shape[1])
    anchor_count = len(OTO_ANCHOR_NAMES)
    xs = np.zeros((len(batch), max_len, n_mels), dtype=np.float32)
    heats = np.zeros((len(batch), max_len, anchor_count), dtype=np.float32)
    masks = np.zeros((len(batch), max_len), dtype=np.float32)
    scalars = np.zeros((len(batch), anchor_count), dtype=np.float32)
    anchors_ms = np.zeros((len(batch), anchor_count), dtype=np.float32)
    durations = np.ones((len(batch),), dtype=np.float32)
    langs = np.zeros((len(batch),), dtype=np.int64)
    fmts = np.zeros((len(batch),), dtype=np.int64)
    contexts = np.zeros((len(batch), 12), dtype=np.float32)
    alias_ids = np.zeros((len(batch),), dtype=np.int64)
    transition_ids = np.zeros((len(batch),), dtype=np.int64)
    prev_alias_ids = np.zeros((len(batch),), dtype=np.int64)
    next_alias_ids = np.zeros((len(batch),), dtype=np.int64)
    prev_transition_ids = np.zeros((len(batch),), dtype=np.int64)
    next_transition_ids = np.zeros((len(batch),), dtype=np.int64)
    weights = np.ones((len(batch),), dtype=np.float32)
    for idx, (
        features,
        heatmap,
        scalar,
        anchor_ms,
        duration_ms,
        language,
        fmt,
        context,
        alias_id,
        transition_id,
        prev_alias_id,
        next_alias_id,
        prev_transition_id,
        next_transition_id,
        weight,
    ) in enumerate(batch):
        n = int(features.shape[0])
        xs[idx, :n] = features
        heats[idx, :n] = heatmap
        masks[idx, :n] = 1.0
        scalars[idx] = scalar
        anchors_ms[idx] = anchor_ms
        durations[idx] = max(1.0, float(duration_ms))
        langs[idx] = int(language)
        fmts[idx] = int(fmt)
        contexts[idx] = context
        alias_ids[idx] = int(alias_id)
        transition_ids[idx] = int(transition_id)
        prev_alias_ids[idx] = int(prev_alias_id)
        next_alias_ids[idx] = int(next_alias_id)
        prev_transition_ids[idx] = int(prev_transition_id)
        next_transition_ids[idx] = int(next_transition_id)
        weights[idx] = float(weight)
    return (
        torch.from_numpy(xs),
        torch.from_numpy(heats),
        torch.from_numpy(scalars),
        torch.from_numpy(anchors_ms),
        torch.from_numpy(durations),
        torch.from_numpy(langs),
        torch.from_numpy(fmts),
        torch.from_numpy(contexts),
        torch.from_numpy(alias_ids),
        torch.from_numpy(transition_ids),
        torch.from_numpy(prev_alias_ids),
        torch.from_numpy(next_alias_ids),
        torch.from_numpy(prev_transition_ids),
        torch.from_numpy(next_transition_ids),
        torch.from_numpy(weights),
        torch.from_numpy(masks),
    )


def _move_batch(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def _anchor_array_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row.get("anchor_offset_ms", row.get("target_offset_ms", 0.0)) or 0.0),
            float(row.get("anchor_overlap_ms", 0.0) or 0.0),
            float(row.get("anchor_preutterance_ms", 0.0) or 0.0),
            float(row.get("anchor_consonant_ms", 0.0) or 0.0),
            float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0),
        ],
        dtype=np.float32,
    )


def _context_array_from_row(row: dict[str, Any]) -> np.ndarray:
    count = max(1.0, float(row.get("file_row_count", 1.0) or 1.0))
    ratio = float(row.get("row_ratio_in_wav", 0.0) or 0.0)
    count_norm = min(1.0, count / 64.0)
    phone_count = min(1.0, max(0.0, float(row.get("alias_phone_count", 0.0) or 0.0)) / 6.0)
    return np.asarray(
        [
            max(0.0, min(1.0, ratio)),
            count_norm,
            phone_count,
            max(0.0, min(1.0, float(row.get("alias_starts_vowel", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_ends_vowel", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_has_space", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_vc", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_cv", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_vv", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("is_head_row", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("is_tail_row", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("prev_alias_ends_vowel", 0.0) or 0.0))),
        ],
        dtype=np.float32,
    )


def _format_loss_multiplier(row: dict[str, Any], cfg: OtoTrainConfig) -> float:
    fmt = str(row.get("format_type", "") or "").strip().lower()
    if fmt == "vcv":
        return max(0.05, float(cfg.vcv_loss_weight))
    if fmt == "cvvc":
        return max(0.05, float(cfg.cvvc_loss_weight))
    if fmt == "cvc":
        return max(0.05, float(cfg.cvc_loss_weight))
    return 1.0


def _crop_around_anchors(
    features: np.ndarray,
    anchors_ms: np.ndarray,
    *,
    hop_sec: float,
    duration_ms: float,
    max_frames: int,
    train: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = int(features.shape[0])
    if max_frames <= 0 or n <= int(max_frames):
        return features, np.clip(anchors_ms, 0.0, max(float(duration_ms), 1.0)), max(float(duration_ms), 1.0)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    min_f = int(np.floor(float(np.min(anchors_ms)) / hop_ms))
    max_f = int(np.ceil(float(np.max(anchors_ms)) / hop_ms))
    center = int(round((min_f + max_f) * 0.5))
    lo = max(0, max_f - int(max_frames) + 12)
    hi = min(n - int(max_frames), min_f - 12)
    if lo <= hi:
        start = random.randint(lo, hi) if train else int(round((lo + hi) * 0.5))
    else:
        start = max(0, min(n - int(max_frames), center - int(max_frames) // 2))
    end = start + int(max_frames)
    shifted = anchors_ms - float(start) * hop_ms
    crop_duration_ms = float(max_frames) * hop_ms
    shifted = np.clip(shifted, 0.0, crop_duration_ms)
    return features[start:end], shifted.astype(np.float32), crop_duration_ms


def _make_anchor_heatmap(anchors_ms: np.ndarray, *, frame_count: int, hop_sec: float, sigma_frames: float) -> np.ndarray:
    n = int(frame_count)
    anchor_count = len(OTO_ANCHOR_NAMES)
    out = np.zeros((n, anchor_count), dtype=np.float32)
    if n <= 0:
        return out
    frames = np.arange(n, dtype=np.float32)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    sigma = max(float(sigma_frames), 0.5)
    for idx, anchor_ms in enumerate(anchors_ms):
        center = float(anchor_ms) / hop_ms
        out[:, idx] = np.exp(-0.5 * ((frames - center) / sigma) ** 2)
    return out


__all__ = ["OtoAnchorDataset", "OtoTrainConfig", "train_oto_from_manifest"]
