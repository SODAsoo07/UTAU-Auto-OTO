"""PyTorch backend for OTO ML correction."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from core.oto_torch_features import DEFAULT_MEL_BINS, DEFAULT_WINDOW_FRAMES, build_tabular_parts, extract_centered_mel_window
from core.oto_torch_model import OtoTorchRegressor


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pytorch_bundle(model_dir, meta=None, schema=None):
    model_dir = os.path.abspath(model_dir)
    config = _load_json(os.path.join(model_dir, "torch_config.json"))
    normalizer = _load_json(os.path.join(model_dir, "normalizer.json"))
    voicebank_vocab = _load_json(os.path.join(model_dir, "voicebank_vocab.json"))
    categorical_vocabs = _load_json(os.path.join(model_dir, "categorical_vocabs.json"))
    categorical_vocabs["voicebank_id"] = voicebank_vocab
    categorical_names = list(config.get("categorical_feature_names") or [])
    numeric_names = list(config.get("numeric_feature_names") or [])
    model = OtoTorchRegressor(
        numeric_dim=len(numeric_names),
        categorical_vocab_sizes={name: len(categorical_vocabs.get(name, {})) for name in categorical_names},
        categorical_feature_names=categorical_names,
        voicebank_vocab_size=len(voicebank_vocab),
        mel_bins=int(config.get("mel_bins", DEFAULT_MEL_BINS)),
        window_frames=int(config.get("window_frames", DEFAULT_WINDOW_FRAMES)),
    )
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "config": config,
        "normalizer": normalizer,
        "categorical_vocabs": categorical_vocabs,
    }


def _feature_row_to_inputs(payload, feature_row: Dict[str, object]):
    config = payload["config"]
    categorical_names = [name for name in list(config.get("categorical_feature_names") or []) if name != "voicebank_id"]
    numeric, categorical = build_tabular_parts(feature_row, categorical_features=categorical_names)
    normalizer = payload["normalizer"]
    mean = np.asarray(normalizer.get("mean") or [0.0] * len(numeric), dtype=np.float32)
    std = np.asarray(normalizer.get("std") or [1.0] * len(numeric), dtype=np.float32)
    numeric = (numeric - mean) / np.maximum(std, 1e-6)

    mel = feature_row.get("mel_window")
    if mel is None:
        wav_path = str(feature_row.get("wav_path", "") or "").strip()
        if wav_path and os.path.isfile(wav_path):
            center_ms = float(feature_row.get("torch_window_center_ms", 0.0) or (float(feature_row.get("base_offset", 0.0) or 0.0) + float(feature_row.get("base_pre", 0.0) or 0.0)))
            mel, _stats = extract_centered_mel_window(
                wav_path,
                center_ms,
                window_frames=int(config.get("window_frames", DEFAULT_WINDOW_FRAMES)),
                mel_bins=int(config.get("mel_bins", DEFAULT_MEL_BINS)),
            )
        else:
            mel = np.zeros((int(config.get("mel_bins", DEFAULT_MEL_BINS)), int(config.get("window_frames", DEFAULT_WINDOW_FRAMES))), dtype=np.float32)
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        mel = np.zeros((int(config.get("mel_bins", DEFAULT_MEL_BINS)), int(config.get("window_frames", DEFAULT_WINDOW_FRAMES))), dtype=np.float32)

    cat_ids = {}
    for name in categorical_names:
        vocab = payload["categorical_vocabs"].get(name, {"__UNK__": 0})
        cat_ids[name] = int(vocab.get(categorical.get(name, ""), 0))
    vb_vocab = payload["categorical_vocabs"].get("voicebank_id", {"__UNK__": 0})
    cat_ids["voicebank_id"] = int(vb_vocab.get(str(feature_row.get("voicebank_id", "") or ""), 0))
    return mel, numeric.astype(np.float32), cat_ids


def predict_pytorch_deltas(payload, feature_row, meta=None, schema=None):
    model = payload["model"]
    mel, numeric, cat_ids = _feature_row_to_inputs(payload, feature_row)
    categorical_names = list(payload["config"].get("categorical_feature_names") or [])
    mel_t = torch.tensor(mel[None, :, :], dtype=torch.float32)
    num_t = torch.tensor(numeric[None, :], dtype=torch.float32)
    cat_t = {name: torch.tensor([cat_ids[name]], dtype=torch.long) for name in categorical_names}
    with torch.no_grad():
        pred = model(mel_t, num_t, cat_t).cpu().numpy()[0]
    target_names = list(payload["config"].get("target_names") or ["delta_offset", "delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"])
    return {name: float(pred[idx]) for idx, name in enumerate(target_names)}
