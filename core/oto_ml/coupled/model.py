"""
Coupled mel+OTO model architecture and constants.

모델 구조 정의, 디바이스 유틸리티, 런타임 의존성 체크를 포함합니다.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.oto_ml.features.schema import (
    ANCHOR_TARGET_NAMES,
    CATEGORICAL_FEATURES,
    DELTA_TARGET_NAMES,
    FEATURE_NAMES,
    TARGET_NAMES,
)

COUPLED_BACKEND = "coupled_nn_v1"
COUPLED_BACKEND_RAWMEL = "coupled_nn_v2_rawmel"
COUPLED_MODEL_FILE = "coupled_model.pt"
COUPLED_MODEL_ONNX_FILE = "coupled_model.onnx"
COUPLED_MODEL_ONNX_META_FILE = "coupled_model_onnx_meta.json"
PATCH_FEATURES = [
    "onset_patch_energy_mean",
    "onset_patch_voiced_ratio",
    "onset_patch_unvoiced_ratio",
    "tail_patch_energy_mean",
    "tail_patch_silence_ratio",
    "blank_span_confidence",
    "syllable_blank_confidence",
    "syllable_mel_voiced_conf",
    "syllable_mel_silence_conf",
    "syllable_mel_unvoiced_conf",
    "syllable_mel_breath_conf",
]


# ── Runtime dependency checks ────────────────────────────────────────────────

def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for coupled backend")


def _require_training_stack():
    _require_numpy()
    try:
        import pandas  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"pandas is required: {exc}") from exc
    try:
        from sklearn.metrics import mean_absolute_error  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"scikit-learn is required: {exc}") from exc


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch is required for coupled backend: {exc}") from exc


def _import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"onnxruntime is required for coupled ONNX backend: {exc}") from exc


# ── Utility functions ────────────────────────────────────────────────────────

def _stable_hash_to_unit(text: str) -> float:
    raw = str(text or "").strip().encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return float(value % 1_000_000) / 1_000_000.0


def _stable_hash_to_index(text: str, buckets: int) -> int:
    buckets_n = max(2, int(buckets or 0))
    raw_text = str(text or "").strip()
    if not raw_text:
        return 0
    raw = raw_text.encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return 1 + (value % (buckets_n - 1))


def _default_categorical_bucket_sizes(categorical_features: List[str]) -> List[int]:
    size_map = {
        "language": 8,
        "format_type": 16,
        "alias_type": 16,
        "alias_group": 24,
        "onset_class": 32,
        "voicing_class": 16,
        "coda_type": 24,
        "vowel_class": 32,
        "mora_position": 16,
        "bridge_type": 24,
        "prev_alias_type": 16,
        "next_alias_type": 16,
        "mapping_reason_code": 32,
    }
    return [int(size_map.get(str(name or "").strip(), 16)) for name in categorical_features]


def _row_feature_vector(feature_row: Dict[str, Any], feature_names: List[str], categorical_features: List[str]) -> "np.ndarray":
    from core.oto_ml.features.schema import canonicalize_feature_row
    canon = canonicalize_feature_row(feature_row, feature_names=feature_names)
    out: List[float] = []
    cat_set = set(categorical_features)
    for name in feature_names:
        if name in cat_set:
            out.append(_stable_hash_to_unit(canon.get(name, "")))
        else:
            try:
                out.append(float(canon.get(name, 0.0)))
            except Exception:
                out.append(0.0)
    return np.asarray(out, dtype=np.float32)


def _row_categorical_index_vector(
    feature_row: Dict[str, Any],
    categorical_features: List[str],
    categorical_bucket_sizes: Optional[List[int]] = None,
) -> "np.ndarray":
    bucket_sizes = list(categorical_bucket_sizes or _default_categorical_bucket_sizes(categorical_features))
    if len(bucket_sizes) < len(categorical_features):
        bucket_sizes.extend([16] * (len(categorical_features) - len(bucket_sizes)))
    vals: List[int] = []
    for idx, key in enumerate(categorical_features):
        vals.append(_stable_hash_to_index(feature_row.get(key, ""), bucket_sizes[idx]))
    return np.asarray(vals, dtype=np.int64)


def _row_patch_vector(feature_row: Dict[str, Any]) -> "np.ndarray":
    vals: List[float] = []
    for key in PATCH_FEATURES:
        try:
            vals.append(float(feature_row.get(key, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def _resolve_device(torch, requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return req


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


# ── Model architecture ───────────────────────────────────────────────────────

def _embedding_dim_for_bucket_size(bucket_size: int) -> int:
    size = max(2, int(bucket_size or 0))
    return max(4, min(16, size // 4))


def _build_model(
    torch,
    nn,
    in_dim: int,
    patch_dim: int,
    hidden_dim: int = 160,
    aux_dim: int = 0,
    categorical_bucket_sizes: Optional[List[int]] = None,
    head_mode: str = "single",
):
    class _CoupledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.patch_net = nn.Sequential(
                nn.Linear(patch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.categorical_bucket_sizes = [max(2, int(v or 0)) for v in (categorical_bucket_sizes or [])]
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(size, _embedding_dim_for_bucket_size(size)) for size in self.categorical_bucket_sizes]
            )
            self.cat_cond_dim = int(sum(_embedding_dim_for_bucket_size(size) for size in self.categorical_bucket_sizes))
            self.joint = nn.Sequential(
                nn.Linear(hidden_dim + 64 + self.cat_cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            mode = str(head_mode or "single").strip().lower()
            self.head_mode = mode
            if mode == "split":
                self.anchor_head = nn.Linear(hidden_dim // 2, len(ANCHOR_TARGET_NAMES))
                self.delta_head = nn.Linear(hidden_dim // 2, len(DELTA_TARGET_NAMES))
            else:
                self.anchor_head = None
                self.delta_head = nn.Linear(hidden_dim // 2, len(TARGET_NAMES))
            self.conf_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
            self.aux_dim = int(aux_dim)
            if self.aux_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, self.aux_dim),
                )
            else:
                self.aux_head = None

        def _cat_repr(self, x, cat_idx):
            if not self.cat_embeddings:
                return None
            if cat_idx is None:
                return torch.zeros((x.shape[0], self.cat_cond_dim), dtype=x.dtype, device=x.device)
            parts = []
            for i, embedding in enumerate(self.cat_embeddings):
                idx = cat_idx[:, i].long().clamp(min=0, max=embedding.num_embeddings - 1)
                parts.append(embedding(idx))
            return torch.cat(parts, dim=1)

        def forward(self, x, patch, cat_idx=None):
            xf = self.feature_net(x)
            xp = self.patch_net(patch)
            pieces = [xf, xp]
            cat_repr = self._cat_repr(x, cat_idx)
            if cat_repr is not None:
                pieces.append(cat_repr)
            z = self.joint(torch.cat(pieces, dim=1))
            conf = self.conf_head(z)
            if self.head_mode == "split":
                anchor = self.anchor_head(z)
                delta = self.delta_head(z)
                if self.aux_head is not None:
                    aux = self.aux_head(z)
                    return anchor, delta, conf, aux
                return anchor, delta, conf
            deltas = self.delta_head(z)
            if self.aux_head is not None:
                aux = self.aux_head(z)
                return deltas, conf, aux
            return deltas, conf

    return _CoupledModel()


def _build_model_rawmel(
    torch,
    nn,
    *,
    in_dim: int,
    patch_dim: int,
    mel_bins: int,
    onset_frames: int,
    tail_frames: int,
    hidden_dim: int = 160,
    aux_dim: int = 0,
    categorical_bucket_sizes: Optional[List[int]] = None,
    head_mode: str = "single",
):
    class _RawMelEncoder(nn.Module):
        def __init__(self, in_bins: int, in_frames: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                # Keep a small temporal axis so onset/tail position cues are not fully averaged out.
                nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 1)),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4, 96),
                nn.ReLU(),
                nn.Linear(96, 64),
                nn.ReLU(),
            )

        def forward(self, x):
            x = self.net(x)
            return self.proj(x)

    class _CoupledRawMel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.patch_net = nn.Sequential(
                nn.Linear(patch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.onset_encoder = _RawMelEncoder(mel_bins, onset_frames)
            self.tail_encoder = _RawMelEncoder(mel_bins, tail_frames)
            self.categorical_bucket_sizes = [max(2, int(v or 0)) for v in (categorical_bucket_sizes or [])]
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(size, _embedding_dim_for_bucket_size(size)) for size in self.categorical_bucket_sizes]
            )
            self.cat_cond_dim = int(sum(_embedding_dim_for_bucket_size(size) for size in self.categorical_bucket_sizes))
            joint_in = hidden_dim + 64 + 64 + 64 + self.cat_cond_dim
            self.joint = nn.Sequential(
                nn.Linear(joint_in, hidden_dim + 32),
                nn.ReLU(),
                nn.Linear(hidden_dim + 32, hidden_dim // 2),
                nn.ReLU(),
            )
            mode = str(head_mode or "single").strip().lower()
            self.head_mode = mode
            if mode == "split":
                self.anchor_head = nn.Linear(hidden_dim // 2, len(ANCHOR_TARGET_NAMES))
                self.delta_head = nn.Linear(hidden_dim // 2, len(DELTA_TARGET_NAMES))
            else:
                self.anchor_head = None
                self.delta_head = nn.Linear(hidden_dim // 2, len(TARGET_NAMES))
            self.conf_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
            self.aux_dim = int(aux_dim)
            if self.aux_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, self.aux_dim),
                )
            else:
                self.aux_head = None

        def _cat_repr(self, x, cat_idx):
            if not self.cat_embeddings:
                return None
            if cat_idx is None:
                return torch.zeros((x.shape[0], self.cat_cond_dim), dtype=x.dtype, device=x.device)
            parts = []
            for i, embedding in enumerate(self.cat_embeddings):
                idx = cat_idx[:, i].long().clamp(min=0, max=embedding.num_embeddings - 1)
                parts.append(embedding(idx))
            return torch.cat(parts, dim=1)

        def forward(self, x, patch, onset, tail, cat_idx=None):
            xf = self.feature_net(x)
            xp = self.patch_net(patch)
            xo = self.onset_encoder(onset)
            xt = self.tail_encoder(tail)
            pieces = [xf, xp, xo, xt]
            cat_repr = self._cat_repr(x, cat_idx)
            if cat_repr is not None:
                pieces.append(cat_repr)
            z = self.joint(torch.cat(pieces, dim=1))
            conf = self.conf_head(z)
            if self.head_mode == "split":
                anchor = self.anchor_head(z)
                delta = self.delta_head(z)
                if self.aux_head is not None:
                    aux = self.aux_head(z)
                    return anchor, delta, conf, aux
                return anchor, delta, conf
            deltas = self.delta_head(z)
            if self.aux_head is not None:
                aux = self.aux_head(z)
                return deltas, conf, aux
            return deltas, conf

    return _CoupledRawMel()
