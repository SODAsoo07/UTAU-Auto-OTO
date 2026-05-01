from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.coarse_crnn.labels import COARSE_LABELS


@dataclass
class CoarseCrnnConfig:
    n_mels: int = 64
    hidden: int = 96
    conv_channels: int = 96
    rnn_layers: int = 1
    dropout: float = 0.10
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    labels: tuple[str, ...] = tuple(COARSE_LABELS)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CoarseCrnnConfig":
        data = dict(payload or {})
        if "labels" in data and not isinstance(data["labels"], tuple):
            data["labels"] = tuple(data["labels"] or COARSE_LABELS)
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        return data


def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_model(config: CoarseCrnnConfig):
    _torch, nn = _import_torch()

    class CoarseCRNN(nn.Module):
        def __init__(self, cfg: CoarseCrnnConfig):
            super().__init__()
            c = int(cfg.conv_channels)
            self.net = nn.Sequential(
                nn.Conv1d(int(cfg.n_mels), c, kernel_size=5, padding=2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Conv1d(c, c, kernel_size=5, padding=2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
            )
            self.rnn = nn.GRU(
                input_size=c,
                hidden_size=int(cfg.hidden),
                num_layers=int(cfg.rnn_layers),
                batch_first=True,
                bidirectional=True,
                dropout=float(cfg.dropout) if int(cfg.rnn_layers) > 1 else 0.0,
            )
            self.head = nn.Linear(int(cfg.hidden) * 2, len(cfg.labels))

        def forward(self, x):
            # x: [batch, frames, mel]
            y = x.transpose(1, 2)
            y = self.net(y).transpose(1, 2)
            y, _ = self.rnn(y)
            return self.head(y)

    return CoarseCRNN(config)


def save_checkpoint(path: str, model, config: CoarseCrnnConfig, *, meta: dict[str, Any] | None = None) -> None:
    torch, _nn = _import_torch()
    payload = {
        "backend": "coarse_crnn_v1",
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, str(path))


def load_checkpoint(path: str, *, map_location: str = "cpu"):
    torch, _nn = _import_torch()
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("coarse_crnn checkpoint payload must be a dict")
    config = CoarseCrnnConfig.from_dict(payload.get("config"))
    model = build_model(config)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("coarse_crnn checkpoint is missing state_dict")
    model.load_state_dict(state)
    model.eval()
    return model, config, dict(payload.get("meta") or {})


__all__ = ["CoarseCrnnConfig", "build_model", "load_checkpoint", "save_checkpoint"]
