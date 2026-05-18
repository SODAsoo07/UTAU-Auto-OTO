from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.phoneme_boundary.types import BOUNDARY_LABELS, CONSONANT_LABELS, PHONE_STATE_LABELS, VOWEL_LABELS


CHECKPOINT_BACKEND = "phoneme_boundary_detector_v1"


@dataclass
class PhonemeBoundaryDetectorConfig:
    n_mels: int = 64
    hidden: int = 128
    conv_channels: int = 96
    rnn_layers: int = 1
    dropout: float = 0.10
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    arch_type: str = "crnn"
    labels: tuple[str, ...] = tuple(BOUNDARY_LABELS)
    enable_quality_head: bool = True
    enable_phone_state_head: bool = True
    enable_phone_identity_heads: bool = True
    phone_state_labels: tuple[str, ...] = tuple(PHONE_STATE_LABELS)
    consonant_labels: tuple[str, ...] = tuple(CONSONANT_LABELS)
    vowel_labels: tuple[str, ...] = tuple(VOWEL_LABELS)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PhonemeBoundaryDetectorConfig":
        data = dict(payload or {})
        tuple_defaults = {
            "labels": tuple(BOUNDARY_LABELS),
            "phone_state_labels": tuple(PHONE_STATE_LABELS),
            "consonant_labels": tuple(CONSONANT_LABELS),
            "vowel_labels": tuple(VOWEL_LABELS),
        }
        for key, default in tuple_defaults.items():
            if key in data and not isinstance(data[key], tuple):
                data[key] = tuple(data[key] or default)
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["labels"] = list(self.labels)
        out["phone_state_labels"] = list(self.phone_state_labels)
        out["consonant_labels"] = list(self.consonant_labels)
        out["vowel_labels"] = list(self.vowel_labels)
        return out


def build_phoneme_boundary_detector(config: PhonemeBoundaryDetectorConfig):
    torch, nn = _import_torch()
    arch_type = str(config.arch_type or "crnn").strip().lower()
    if arch_type not in {"crnn", "conv_only"}:
        raise ValueError(f"unknown phoneme boundary detector arch_type: {arch_type!r}")

    class PhonemeBoundaryDetector(nn.Module):
        def __init__(self, cfg: PhonemeBoundaryDetectorConfig):
            super().__init__()
            self.arch_type = arch_type
            encoded = int(cfg.hidden) * 2
            if arch_type == "crnn":
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
            else:
                self.encoder = _build_conv_only_encoder(cfg)
                self.rnn = None
            self.boundary_head = nn.Linear(encoded, len(cfg.labels))
            self.quality_head = nn.Linear(encoded, 1) if bool(cfg.enable_quality_head) else None
            self.phone_state_head = (
                nn.Linear(encoded, len(cfg.phone_state_labels)) if bool(cfg.enable_phone_state_head) else None
            )
            self.consonant_head = (
                nn.Linear(encoded, len(cfg.consonant_labels)) if bool(cfg.enable_phone_identity_heads) else None
            )
            self.vowel_head = nn.Linear(encoded, len(cfg.vowel_labels)) if bool(cfg.enable_phone_identity_heads) else None

        def encode(self, x):
            y = self.encoder(x.transpose(1, 2)).transpose(1, 2)
            if self.rnn is not None:
                y, _ = self.rnn(y)
            return y

        def forward(self, x):
            encoded = self.encode(x)
            out = {
                "boundary_logits": self.boundary_head(encoded),
                "quality_logits": None,
                "phone_state_logits": None,
                "consonant_logits": None,
                "vowel_logits": None,
            }
            if self.quality_head is not None:
                out["quality_logits"] = self.quality_head(encoded).squeeze(-1)
            if self.phone_state_head is not None:
                out["phone_state_logits"] = self.phone_state_head(encoded)
            if self.consonant_head is not None:
                out["consonant_logits"] = self.consonant_head(encoded)
            if self.vowel_head is not None:
                out["vowel_logits"] = self.vowel_head(encoded)
            return out

    return PhonemeBoundaryDetector(config)


def save_phoneme_boundary_checkpoint(
    path: str,
    model,
    config: PhonemeBoundaryDetectorConfig,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    torch, _nn = _import_torch()
    payload = {
        "backend": CHECKPOINT_BACKEND,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, str(path))


def load_phoneme_boundary_checkpoint(path: str, *, map_location: str = "cpu"):
    torch, _nn = _import_torch()
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("phoneme boundary checkpoint payload must be a dict")
    backend = str(payload.get("backend", "") or "")
    if backend and backend != CHECKPOINT_BACKEND:
        raise ValueError(f"unsupported phoneme boundary checkpoint backend: {backend}")
    config = PhonemeBoundaryDetectorConfig.from_dict(payload.get("config"))
    model = build_phoneme_boundary_detector(config)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("phoneme boundary checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, dict(payload.get("meta") or {})


def _build_conv_only_encoder(cfg: PhonemeBoundaryDetectorConfig):
    _torch, nn = _import_torch()
    base = int(cfg.conv_channels)
    out_ch = int(cfg.hidden) * 2
    ch1 = base
    ch2 = max(base, base * 4 // 3)
    ch3 = max(ch2, base * 2)
    ch4 = out_ch

    def _block(in_ch: int, out_ch_: int, dilation: int, use_dropout: bool):
        layers = [
            nn.Conv1d(in_ch, out_ch_, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch_),
            nn.GELU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(float(cfg.dropout)))
        return nn.Sequential(*layers)

    return nn.Sequential(
        _block(int(cfg.n_mels), ch1, 1, False),
        _block(ch1, ch2, 2, False),
        _block(ch2, ch3, 4, True),
        _block(ch3, ch4, 8, True),
    )


def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


__all__ = [
    "CHECKPOINT_BACKEND",
    "PhonemeBoundaryDetectorConfig",
    "build_phoneme_boundary_detector",
    "load_phoneme_boundary_checkpoint",
    "save_phoneme_boundary_checkpoint",
]
