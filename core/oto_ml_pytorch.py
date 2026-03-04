"""PyTorch backend for OTO ML correction."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch

from core.oto_ml_features import canonicalize_feature_row
from core.oto_torch_features import DEFAULT_MEL_BINS, DEFAULT_WINDOW_FRAMES, build_tabular_parts, extract_centered_mel_window
from core.oto_torch_model import OtoTorchRegressor

logger = logging.getLogger(__name__)


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
    meta_target_mode = ""
    if isinstance(meta, dict):
        meta_target_mode = str(meta.get("target_mode", "") or "").strip().lower()
    target_mode = str(config.get("target_mode", "") or meta_target_mode).strip().lower()
    residual_base_model_dir = str(
        config.get("residual_base_model_dir", "") or (meta.get("residual_base_model_dir", "") if isinstance(meta, dict) else "")
    ).strip()
    residual_payload = None
    residual_meta = None
    residual_schema = None
    if target_mode == "residual_over_lightgbm":
        if residual_base_model_dir:
            if not os.path.isabs(residual_base_model_dir):
                residual_base_model_dir = os.path.abspath(os.path.join(model_dir, residual_base_model_dir))
            meta_path = os.path.join(residual_base_model_dir, "model_meta.json")
            schema_path = os.path.join(residual_base_model_dir, "feature_schema.json")
            if os.path.isfile(meta_path):
                residual_meta = _load_json(meta_path)
            if os.path.isfile(schema_path):
                residual_schema = _load_json(schema_path)
            if residual_meta:
                try:
                    from core.oto_ml_lightgbm import load_lightgbm_bundle

                    residual_payload = load_lightgbm_bundle(
                        residual_base_model_dir,
                        meta=residual_meta,
                        schema=residual_schema,
                    )
                except Exception as e:
                    logger.warning("Failed to load residual base LightGBM bundle (%s): %s", residual_base_model_dir, e)
        else:
            logger.warning("PyTorch bundle target_mode=residual_over_lightgbm but residual_base_model_dir is empty: %s", model_dir)
    return {
        "model": model,
        "config": config,
        "normalizer": normalizer,
        "categorical_vocabs": categorical_vocabs,
        "target_mode": target_mode or "absolute_delta",
        "residual_base_model_dir": residual_base_model_dir,
        "residual_base_payload": residual_payload,
        "residual_base_meta": residual_meta or {},
        "residual_base_schema": residual_schema or {},
    }


def _feature_row_to_inputs(payload, feature_row: Dict[str, object]):
    config = payload["config"]
    categorical_names = [name for name in list(config.get("categorical_feature_names") or []) if name != "voicebank_id"]
    numeric_names = list(config.get("numeric_feature_names") or [])
    if numeric_names:
        canonical = canonicalize_feature_row(feature_row, feature_names=numeric_names)
        numeric = np.asarray([float(canonical.get(name, 0.0) or 0.0) for name in numeric_names], dtype=np.float32)
        categorical = {name: str(feature_row.get(name, "") or "") for name in categorical_names}
    else:
        numeric, categorical = build_tabular_parts(feature_row, categorical_features=categorical_names)
    normalizer = payload["normalizer"]
    mean = np.asarray(normalizer.get("mean") or [0.0] * len(numeric), dtype=np.float32)
    std = np.asarray(normalizer.get("std") or [1.0] * len(numeric), dtype=np.float32)
    if len(mean) != len(numeric):
        dim = min(len(mean), len(numeric))
        if dim <= 0:
            dim = len(numeric)
        mean = mean[:dim]
        std = std[:dim]
        numeric = numeric[:dim]
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
    out = {name: float(pred[idx]) for idx, name in enumerate(target_names)}

    if str(payload.get("target_mode", "")).strip().lower() == "residual_over_lightgbm":
        base_payload = payload.get("residual_base_payload")
        if base_payload is not None:
            try:
                from core.oto_ml_lightgbm import predict_lightgbm_deltas

                base = predict_lightgbm_deltas(
                    base_payload,
                    feature_row,
                    meta=payload.get("residual_base_meta") or {},
                    schema=payload.get("residual_base_schema") or {},
                )
                for name in target_names:
                    out[name] = float(base.get(name, 0.0)) + float(out.get(name, 0.0))
            except Exception as e:
                logger.warning("Residual merge failed in PyTorch backend: %s", e)
    return out
