from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:
    GroupShuffleSplit = None

from core.oto_ml_features import TARGET_NAMES, get_delta_clip_limits, write_feature_schema
from core.oto_torch_dataset import load_torch_dataset_index
from core.oto_torch_model import OtoTorchRegressor


TARGET_WEIGHTS = torch.tensor([1.2, 0.8, 1.0, 1.2, 1.0], dtype=torch.float32)

TASK_LOSS_PROFILES: Dict[str, Dict[str, object]] = {
    "default": {
        "target_weights": [1.2, 0.8, 1.0, 1.2, 1.0],
        "constraint_weight": 0.10,
        "rhythm_weight": 0.05,
    },
    "korean_vc_sonorant": {
        "target_weights": [1.4, 0.9, 1.1, 1.25, 1.1],
        "constraint_weight": 0.12,
        "rhythm_weight": 0.08,
    },
    "korean_vc_stop": {
        "target_weights": [1.8, 0.6, 1.6, 0.45, 0.25],
        "constraint_weight": 0.22,
        "rhythm_weight": 0.02,
    },
}


def _task_loss_profile(task_name: str) -> Dict[str, object]:
    name = str(task_name or "").strip().lower()
    if name in {"korean_bridge_cvvc_vc_sonorant", "korean_bridge_cvc_vc_sonorant"}:
        return TASK_LOSS_PROFILES["korean_vc_sonorant"]
    if name in {"korean_bridge_cvvc_vc_stop", "korean_bridge_cvc_vc_stop"}:
        return TASK_LOSS_PROFILES["korean_vc_stop"]
    return TASK_LOSS_PROFILES["default"]


class TorchShardDataset(Dataset):
    def __init__(self, shard_entries, numeric_names, categorical_names, target_names, categorical_vocabs, normalizer):
        self.shard_entries = list(shard_entries)
        self.numeric_names = list(numeric_names)
        self.categorical_names = list(categorical_names)
        self.target_names = list(target_names)
        self.categorical_vocabs = dict(categorical_vocabs)
        self.normalizer = dict(normalizer)
        self._offsets = []
        total = 0
        for entry in self.shard_entries:
            self._offsets.append(total)
            total += int(entry["rows"])
        self.total_rows = total
        self._active_shard_idx = -1
        self._active_data = None

    def __len__(self):
        return self.total_rows

    def _locate(self, idx: int) -> Tuple[int, int]:
        for shard_idx in range(len(self.shard_entries) - 1, -1, -1):
            if idx >= self._offsets[shard_idx]:
                return shard_idx, idx - self._offsets[shard_idx]
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int):
        if self._active_shard_idx == shard_idx and self._active_data is not None:
            return self._active_data
        path = self.shard_entries[shard_idx]["path"]
        data = np.load(path, allow_pickle=False)
        self._active_shard_idx = shard_idx
        self._active_data = data
        return data

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(int(idx))
        data = self._load_shard(shard_idx)
        mel = data["mel_windows"][local_idx].astype(np.float32)
        numeric = data["numeric_features"][local_idx].astype(np.float32)
        raw_numeric = numeric.copy()
        mean = np.asarray(self.normalizer["mean"], dtype=np.float32)
        std = np.asarray(self.normalizer["std"], dtype=np.float32)
        numeric = (numeric - mean) / std
        categorical_raw = data["categorical_features"][local_idx]
        cats = {}
        for i, name in enumerate(self.categorical_names):
            vocab = self.categorical_vocabs[name]
            token = str(categorical_raw[i])
            cats[name] = np.int64(vocab.get(token, 0))
        cats["voicebank_id"] = np.int64(self.categorical_vocabs["voicebank_id"].get(str(data["voicebank_id"][local_idx]), 0))
        target = data["targets"][local_idx].astype(np.float32)
        metadata = {
            "alias_group": str(data["alias_group"][local_idx]),
            "alias_type": str(data["alias_type"][local_idx]),
            "voicebank_id": str(data["voicebank_id"][local_idx]),
        }
        return {
            "mel_window": mel,
            "numeric_features": numeric,
            "raw_numeric_features": raw_numeric,
            "categorical_features": cats,
            "targets": target,
            "metadata": metadata,
        }


def _collate(batch):
    mel = torch.tensor(np.stack([item["mel_window"] for item in batch], axis=0), dtype=torch.float32)
    num = torch.tensor(np.stack([item["numeric_features"] for item in batch], axis=0), dtype=torch.float32)
    raw_num = torch.tensor(np.stack([item["raw_numeric_features"] for item in batch], axis=0), dtype=torch.float32)
    target = torch.tensor(np.stack([item["targets"] for item in batch], axis=0), dtype=torch.float32)
    cat_names = list(batch[0]["categorical_features"].keys())
    cats = {}
    for name in cat_names:
        cats[name] = torch.tensor([item["categorical_features"][name] for item in batch], dtype=torch.long)
    metadata = [item["metadata"] for item in batch]
    return {
        "mel_window": mel,
        "numeric_features": num,
        "raw_numeric_features": raw_num,
        "categorical_features": cats,
        "targets": target,
        "metadata": metadata,
    }


def _group_split(shards: List[Dict[str, object]], test_size: float = 0.2):
    if len(shards) <= 1:
        return shards, shards
    if GroupShuffleSplit is not None:
        groups = [entry["voicebank_id"] for entry in shards]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        idx = np.arange(len(shards))
        train_idx, valid_idx = next(splitter.split(idx, groups=groups))
        train_shards = [shards[i] for i in train_idx]
        valid_shards = [shards[i] for i in valid_idx]
        return train_shards, valid_shards
    split_at = max(1, int(len(shards) * (1.0 - test_size)))
    return shards[:split_at], shards[split_at:]


def _scan_train_stats(shards: List[Dict[str, object]], categorical_names: List[str]):
    numeric_sum = None
    numeric_sq = None
    count = 0
    cat_tokens: Dict[str, set] = {name: set() for name in categorical_names}
    voicebanks = {"__UNK__"}
    for entry in shards:
        data = np.load(entry["path"], allow_pickle=False)
        num = data["numeric_features"].astype(np.float64)
        if numeric_sum is None:
            numeric_sum = num.sum(axis=0)
            numeric_sq = (num ** 2).sum(axis=0)
        else:
            numeric_sum += num.sum(axis=0)
            numeric_sq += (num ** 2).sum(axis=0)
        count += num.shape[0]
        cat = data["categorical_features"]
        for idx, name in enumerate(categorical_names):
            cat_tokens[name].update(str(v) for v in cat[:, idx])
        voicebanks.update(str(v) for v in data["voicebank_id"])
    mean = numeric_sum / max(count, 1)
    var = np.maximum((numeric_sq / max(count, 1)) - (mean ** 2), 1e-6)
    std = np.sqrt(var)
    normalizer = {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}
    vocab = {}
    for name in categorical_names:
        items = sorted(v for v in cat_tokens[name] if v)
        vocab[name] = {"__UNK__": 0}
        for i, token in enumerate(items, start=1):
            vocab[name][token] = i
    vb_items = sorted(v for v in voicebanks if v and v != "__UNK__")
    vocab["voicebank_id"] = {"__UNK__": 0}
    for i, token in enumerate(vb_items, start=1):
        vocab["voicebank_id"][token] = i
    return normalizer, vocab


def _move_batch(batch, device):
    batch["mel_window"] = batch["mel_window"].to(device, non_blocking=True)
    batch["numeric_features"] = batch["numeric_features"].to(device, non_blocking=True)
    batch["raw_numeric_features"] = batch["raw_numeric_features"].to(device, non_blocking=True)
    for name in list(batch["categorical_features"].keys()):
        batch["categorical_features"][name] = batch["categorical_features"][name].to(device, non_blocking=True)
    batch["targets"] = batch["targets"].to(device, non_blocking=True)
    return batch


def _weighted_huber(pred, target, target_weights):
    loss = nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    return (loss * target_weights.to(loss.device)).mean()


def _constraint_loss(pred, batch, numeric_names):
    idx = {name: i for i, name in enumerate(numeric_names)}
    numeric = batch["raw_numeric_features"]
    def denorm(name):
        return numeric[:, idx[name]]
    base_offset = denorm("base_offset")
    base_cons = denorm("base_cons")
    base_cut_abs = denorm("base_cutoff_abs")
    base_pre = denorm("base_pre")
    base_ovl = denorm("base_ovl")
    pred_offset = base_offset + pred[:, 0]
    pred_cons = base_cons + pred[:, 1]
    pred_cut_abs = base_cut_abs + pred[:, 2]
    pred_pre = base_pre + pred[:, 3]
    pred_ovl = base_ovl + pred[:, 4]
    active_len = pred_cut_abs - pred_offset
    penalties = []
    penalties.append(torch.relu(pred_pre - pred_cons))
    penalties.append(torch.relu(pred_ovl - pred_pre))
    penalties.append(torch.relu(pred_cons - active_len))
    penalties.append(torch.relu(pred_offset - pred_cut_abs + 8.0))
    return sum(p.mean() for p in penalties) / len(penalties)


def _rhythm_loss(pred, batch, numeric_names):
    idx = {name: i for i, name in enumerate(numeric_names)}
    numeric = batch["raw_numeric_features"]
    base_pre = numeric[:, idx["base_pre"]]
    base_ovl = numeric[:, idx["base_ovl"]]
    pred_pre = base_pre + pred[:, 3]
    pred_ovl = base_ovl + pred[:, 4]
    base_gap = base_pre - base_ovl
    pred_gap = pred_pre - pred_ovl
    diff = torch.abs(pred_gap - base_gap)
    return torch.relu(diff - 40.0).mean() / 40.0


def _evaluate(model, loader, device, numeric_names):
    model.eval()
    total = 0
    preds = []
    truths = []
    alias_groups = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model(batch["mel_window"], batch["numeric_features"], batch["categorical_features"])
            preds.append(pred.detach().cpu().numpy())
            truths.append(batch["targets"].detach().cpu().numpy())
            alias_groups.extend(meta.get("alias_group", "") for meta in batch["metadata"])
            total += pred.shape[0]
    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
    truth_arr = np.concatenate(truths, axis=0) if truths else np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
    metrics = {}
    for i, target in enumerate(TARGET_NAMES):
        baseline_mae = float(np.mean(np.abs(truth_arr[:, i]))) if len(truth_arr) else 0.0
        model_mae = float(np.mean(np.abs(truth_arr[:, i] - pred_arr[:, i]))) if len(truth_arr) else 0.0
        metrics[target] = {"baseline_mae": baseline_mae, "model_mae": model_mae}
    alias_group_metrics = {}
    if len(truth_arr) and alias_groups:
        grouped = {}
        for idx, group in enumerate(alias_groups):
            grouped.setdefault(group, []).append(idx)
        for group, indices in grouped.items():
            alias_group_metrics[group] = {}
            for i, target in enumerate(TARGET_NAMES):
                t = truth_arr[indices, i]
                p = pred_arr[indices, i]
                alias_group_metrics[group][target] = {
                    "baseline_mae": float(np.mean(np.abs(t))),
                    "model_mae": float(np.mean(np.abs(t - p))),
                }
    return {"metrics": metrics, "alias_group_metrics": alias_group_metrics}


def train_torch_bundle(
    dataset_dir: str,
    out_dir: str,
    language: str,
    task_name: str,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    device_override: str = "",
    log_every_steps: int = 100,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.5,
    lr_reduce_factor: float = 0.5,
    lr_reduce_patience: int = 1,
    lr_reduce_min_lr: float = 1e-5,
    lr_reduce_threshold: float = 0.5,
) -> Dict[str, object]:
    index = load_torch_dataset_index(os.path.join(dataset_dir, "index.json"))
    shards = list(index.get("shards") or [])
    if not shards:
        raise RuntimeError("No shards found for torch dataset")
    categorical_names = list(index.get("categorical_feature_names") or [])
    numeric_names = list(index.get("numeric_feature_names") or [])
    target_names = list(index.get("target_names") or TARGET_NAMES)
    train_shards, valid_shards = _group_split(shards)
    if not valid_shards:
        valid_shards = train_shards[-1:]
    normalizer, vocab = _scan_train_stats(train_shards, categorical_names)

    train_ds = TorchShardDataset(train_shards, numeric_names, categorical_names, target_names, vocab, normalizer)
    valid_ds = TorchShardDataset(valid_shards, numeric_names, categorical_names, target_names, vocab, normalizer)

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(use_amp and device.type == "cuda")
    loader_kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda", "collate_fn": _collate}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = OtoTorchRegressor(
        numeric_dim=len(numeric_names),
        categorical_vocab_sizes={name: len(vocab[name]) for name in categorical_names + ["voicebank_id"]},
        categorical_feature_names=categorical_names + ["voicebank_id"],
        voicebank_vocab_size=len(vocab["voicebank_id"]),
        mel_bins=int(index.get("mel_bins", 80)),
        window_frames=int(index.get("window_frames", 160)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_reduce_factor),
        patience=int(lr_reduce_patience),
        threshold=float(lr_reduce_threshold),
        threshold_mode="abs",
        min_lr=float(lr_reduce_min_lr),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    loss_profile = _task_loss_profile(task_name)
    target_weights = torch.tensor(loss_profile["target_weights"], dtype=torch.float32)
    constraint_weight = float(loss_profile["constraint_weight"])
    rhythm_weight = float(loss_profile["rhythm_weight"])

    best_state = None
    best_valid = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    history = []
    global_step = 0
    os.makedirs(out_dir, exist_ok=True)

    log_every = max(0, int(log_every_steps))
    print(
        f"[TorchTrain] start task={task_name} lang={language} device={device.type} "
        f"epochs={int(epochs)} batch={int(batch_size)} amp={'ON' if amp_enabled else 'OFF'} "
        f"loss_weights={loss_profile['target_weights']} constraint_w={constraint_weight} rhythm_w={rhythm_weight} "
        f"early_stop_patience={int(early_stopping_patience)} early_stop_min_delta={float(early_stopping_min_delta)} "
        f"lr_reduce_factor={float(lr_reduce_factor)} lr_reduce_patience={int(lr_reduce_patience)} "
        f"lr_reduce_min_lr={float(lr_reduce_min_lr)} lr_reduce_threshold={float(lr_reduce_threshold)}"
    )

    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(batch["mel_window"], batch["numeric_features"], batch["categorical_features"])
                loss_main = _weighted_huber(pred, batch["targets"], target_weights)
                loss_constraint = _constraint_loss(pred, batch, numeric_names)
                loss_rhythm = _rhythm_loss(pred, batch, numeric_names)
                loss = loss_main + (constraint_weight * loss_constraint) + (rhythm_weight * loss_rhythm)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            step_count += 1
            global_step += 1
            if log_every and (global_step % log_every == 0):
                print(
                    f"[TorchTrain] epoch={epoch + 1}/{int(epochs)} "
                    f"step={step_count}/{len(train_loader)} global_step={global_step} "
                    f"loss={loss_value:.6f}"
                )

        valid_summary = _evaluate(model, valid_loader, device, numeric_names)
        valid_loss = sum(metric["model_mae"] for metric in valid_summary["metrics"].values())
        prev_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(valid_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        epoch_train_loss = epoch_loss / max(step_count, 1)
        print(
            f"[TorchTrain] epoch_end epoch={epoch + 1}/{int(epochs)} "
            f"train_loss={epoch_train_loss:.6f} valid_score={valid_loss:.6f} lr={new_lr:.8f}"
        )
        if new_lr < prev_lr:
            print(
                f"[TorchTrain] lr_reduced epoch={epoch + 1} "
                f"prev_lr={prev_lr:.8f} new_lr={new_lr:.8f}"
            )
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "valid_score": valid_loss,
            "lr": new_lr,
        })
        improved = (best_valid - valid_loss) > float(early_stopping_min_delta)
        if improved:
            best_valid = valid_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            print(
                f"[TorchTrain] checkpoint_saved epoch={best_epoch} "
                f"global_step={global_step} valid_score={best_valid:.6f}"
            )
        else:
            no_improve_epochs += 1
            if int(early_stopping_patience) > 0 and no_improve_epochs >= int(early_stopping_patience):
                print(
                    f"[TorchTrain] early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch} best_valid={best_valid:.6f} "
                    f"no_improve_epochs={no_improve_epochs}"
                )
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    final_eval = _evaluate(model, valid_loader, device, numeric_names)

    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    torch_config = {
        "model_type": "small_cnn_tabular_regressor",
        "numeric_feature_names": numeric_names,
        "categorical_feature_names": categorical_names + ["voicebank_id"],
        "target_names": target_names,
        "mel_bins": int(index.get("mel_bins", 80)),
        "window_frames": int(index.get("window_frames", 160)),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "amp_enabled": bool(amp_enabled),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": float(lr_reduce_factor),
            "patience": int(lr_reduce_patience),
            "min_lr": float(lr_reduce_min_lr),
            "threshold": float(lr_reduce_threshold),
        },
        "loss_profile": {
            "target_weights": list(loss_profile["target_weights"]),
            "constraint_weight": constraint_weight,
            "rhythm_weight": rhythm_weight,
        },
    }
    with open(os.path.join(out_dir, "torch_config.json"), "w", encoding="utf-8") as f:
        json.dump(torch_config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "normalizer.json"), "w", encoding="utf-8") as f:
        json.dump(normalizer, f, ensure_ascii=False, indent=2)
    voicebank_vocab = vocab.pop("voicebank_id")
    with open(os.path.join(out_dir, "voicebank_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(voicebank_vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "categorical_vocabs.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))

    meta = {
        "backend": "pytorch",
        "language": str(language).strip().lower(),
        "format_type": str(index.get("formats", [""])[0] if index.get("formats") else "general"),
        "task_name": task_name,
        "model_version": "v1",
        "feature_version": "v4",
        "audio_window_frames": int(index.get("window_frames", 160)),
        "mel_bins": int(index.get("mel_bins", 80)),
        "feature_names": numeric_names + categorical_names,
        "categorical_features": categorical_names + ["voicebank_id"],
        "targets": list(target_names),
        "delta_clip_limits": get_delta_clip_limits(language),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(sum(entry["rows"] for entry in train_shards)),
        "voicebank_count": int(len({entry['voicebank_id'] for entry in shards})),
        "holdout_metrics": final_eval["metrics"],
        "holdout_alias_group_metrics": final_eval["alias_group_metrics"],
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_valid),
        "history": history,
        "early_stopping": {
            "patience": int(early_stopping_patience),
            "min_delta": float(early_stopping_min_delta),
            "stopped_early": bool(int(early_stopping_patience) > 0 and no_improve_epochs >= int(early_stopping_patience)),
        },
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": float(lr_reduce_factor),
            "patience": int(lr_reduce_patience),
            "min_lr": float(lr_reduce_min_lr),
            "threshold": float(lr_reduce_threshold),
        },
        "loss_profile": {
            "target_weights": list(loss_profile["target_weights"]),
            "constraint_weight": constraint_weight,
            "rhythm_weight": rhythm_weight,
        },
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": final_eval["metrics"], "alias_group_metrics": final_eval["alias_group_metrics"]}, f, ensure_ascii=False, indent=2)
    return meta
