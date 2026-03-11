"""
Coupled mel+OTO model architecture and constants.

모델 구조 정의, 디바이스 유틸리티, 런타임 의존성 체크를 포함합니다.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.oto_ml.features.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET_NAMES

COUPLED_BACKEND = "coupled_nn_v1"
COUPLED_MODEL_FILE = "coupled_model.pt"
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


# ── Utility functions ────────────────────────────────────────────────────────

def _stable_hash_to_unit(text: str) -> float:
    raw = str(text or "").strip().encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return float(value % 1_000_000) / 1_000_000.0


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

def _build_model(torch, nn, in_dim: int, patch_dim: int, hidden_dim: int = 160, aux_dim: int = 0):
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
            self.joint = nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
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

        def forward(self, x, patch):
            xf = self.feature_net(x)
            xp = self.patch_net(patch)
            z = self.joint(torch.cat([xf, xp], dim=1))
            deltas = self.delta_head(z)
            conf = self.conf_head(z)
            if self.aux_head is not None:
                aux = self.aux_head(z)
                return deltas, conf, aux
            return deltas, conf

    return _CoupledModel()
