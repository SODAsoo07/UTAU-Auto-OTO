from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.coarse_crnn.boundary_types import BOUNDARY_LABELS


@dataclass
class BoundaryScorerConfig:
    n_mels: int = 64
    hidden: int = 128
    conv_channels: int = 96
    rnn_layers: int = 1
    dropout: float = 0.10
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    labels: tuple[str, ...] = tuple(BOUNDARY_LABELS)
    enable_quality_head: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BoundaryScorerConfig":
        data = dict(payload or {})
        if "labels" in data and not isinstance(data["labels"], tuple):
            data["labels"] = tuple(data["labels"] or list(BOUNDARY_LABELS))
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["labels"] = list(self.labels)
        return out


def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_boundary_scorer(config: BoundaryScorerConfig):
    _torch, nn = _import_torch()

    class BoundaryScorer(nn.Module):
        def __init__(self, cfg: BoundaryScorerConfig):
            super().__init__()
            channels = int(cfg.conv_channels)
            self.encoder = nn.Sequential(
                nn.Conv1d(int(cfg.n_mels), channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            )
            self.rnn = nn.GRU(
                input_size=channels,
                hidden_size=int(cfg.hidden),
                num_layers=int(cfg.rnn_layers),
                batch_first=True,
                bidirectional=True,
                dropout=float(cfg.dropout) if int(cfg.rnn_layers) > 1 else 0.0,
            )
            encoded = int(cfg.hidden) * 2
            self.boundary_head = nn.Linear(encoded, len(cfg.labels))
            self.quality_head = nn.Linear(encoded, 1) if bool(cfg.enable_quality_head) else None

        def encode(self, x):
            y = self.encoder(x.transpose(1, 2)).transpose(1, 2)
            y, _ = self.rnn(y)
            return y

        def forward(self, x):
            encoded = self.encode(x)
            out = {
                "boundary_logits": self.boundary_head(encoded),
                "quality_logits": None,
            }
            if self.quality_head is not None:
                out["quality_logits"] = self.quality_head(encoded).squeeze(-1)
            return out

    return BoundaryScorer(config)


def save_boundary_checkpoint(path: str, model, config: BoundaryScorerConfig, *, meta: dict[str, Any] | None = None) -> None:
    torch, _nn = _import_torch()
    payload = {
        "backend": "oto_boundary_scorer_v1",
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, str(path))


def load_boundary_checkpoint(path: str, *, map_location: str = "cpu"):
    torch, _nn = _import_torch()
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("boundary scorer checkpoint payload must be a dict")
    config = BoundaryScorerConfig.from_dict(payload.get("config"))
    model = build_boundary_scorer(config)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("boundary scorer checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, dict(payload.get("meta") or {})


__all__ = [
    "BoundaryScorerConfig",
    "build_boundary_scorer",
    "load_boundary_checkpoint",
    "save_boundary_checkpoint",
]

