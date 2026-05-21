from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.coarse_crnn.boundary_types import (
    BOUNDARY_LABELS,
    PHONE_AWARE_CONSONANT_FAMILY_LABELS,
    PHONE_AWARE_CONSONANT_LABELS,
    PHONE_AWARE_CVS_LABELS,
    PHONE_AWARE_VOWEL_GLIDE_LABELS,
    PHONE_AWARE_VOWEL_LABELS,
    PHONE_AWARE_VOWEL_NUCLEUS_LABELS,
)


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
    # "crnn" = legacy 2-conv + 1L BiLSTM encoder (default).
    # "conv_only" = 4-block dilated 1D conv encoder used as the distillation
    # student for an SSL-teacher path. Same head shapes/output channels
    # (hidden*2), so checkpoints share the head dimensionality but not the
    # encoder weights.
    arch_type: str = "crnn"
    labels: tuple[str, ...] = tuple(BOUNDARY_LABELS)
    enable_quality_head: bool = True
    enable_phone_aux_heads: bool = False
    enable_phone_family_heads: bool = False
    cvs_labels: tuple[str, ...] = tuple(PHONE_AWARE_CVS_LABELS)
    consonant_labels: tuple[str, ...] = tuple(PHONE_AWARE_CONSONANT_LABELS)
    vowel_labels: tuple[str, ...] = tuple(PHONE_AWARE_VOWEL_LABELS)
    consonant_family_labels: tuple[str, ...] = tuple(PHONE_AWARE_CONSONANT_FAMILY_LABELS)
    vowel_nucleus_labels: tuple[str, ...] = tuple(PHONE_AWARE_VOWEL_NUCLEUS_LABELS)
    vowel_glide_labels: tuple[str, ...] = tuple(PHONE_AWARE_VOWEL_GLIDE_LABELS)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BoundaryScorerConfig":
        data = dict(payload or {})
        tuple_defaults = {
            "labels": tuple(BOUNDARY_LABELS),
            "cvs_labels": tuple(PHONE_AWARE_CVS_LABELS),
            "consonant_labels": tuple(PHONE_AWARE_CONSONANT_LABELS),
            "vowel_labels": tuple(PHONE_AWARE_VOWEL_LABELS),
            "consonant_family_labels": tuple(PHONE_AWARE_CONSONANT_FAMILY_LABELS),
            "vowel_nucleus_labels": tuple(PHONE_AWARE_VOWEL_NUCLEUS_LABELS),
            "vowel_glide_labels": tuple(PHONE_AWARE_VOWEL_GLIDE_LABELS),
        }
        for key, default in tuple_defaults.items():
            if key in data and not isinstance(data[key], tuple):
                data[key] = tuple(data[key] or list(default))
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["labels"] = list(self.labels)
        out["cvs_labels"] = list(self.cvs_labels)
        out["consonant_labels"] = list(self.consonant_labels)
        out["vowel_labels"] = list(self.vowel_labels)
        out["consonant_family_labels"] = list(self.consonant_family_labels)
        out["vowel_nucleus_labels"] = list(self.vowel_nucleus_labels)
        out["vowel_glide_labels"] = list(self.vowel_glide_labels)
        return out


def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def _build_conv_only_encoder(cfg: BoundaryScorerConfig):
    """Dilated 4-block 1D-conv encoder. Output channels = hidden*2 so the same
    head Linear layers can be reused. Receptive field with k=5, d=[1,2,4,8] is
    61 frames (~610 ms at 10 ms hop), covering boundary context and the
    spectral neighborhood needed for phone identity distillation.
    """
    _torch, nn = _import_torch()
    base = int(cfg.conv_channels)
    out_ch = int(cfg.hidden) * 2
    ch1 = base
    ch2 = max(base, base * 4 // 3)
    ch3 = max(ch2, base * 2)
    ch4 = out_ch
    dilations = (1, 2, 4, 8)

    def _block(in_ch: int, out_ch_: int, dilation: int, use_dropout: bool):
        padding = 2 * dilation
        layers = [
            nn.Conv1d(in_ch, out_ch_, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch_),
            nn.GELU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(float(cfg.dropout)))
        return nn.Sequential(*layers)

    return nn.Sequential(
        _block(int(cfg.n_mels), ch1, dilations[0], use_dropout=False),
        _block(ch1, ch2, dilations[1], use_dropout=False),
        _block(ch2, ch3, dilations[2], use_dropout=True),
        _block(ch3, ch4, dilations[3], use_dropout=True),
    )


def build_boundary_scorer(config: BoundaryScorerConfig):
    _torch, nn = _import_torch()

    arch_type = str(getattr(config, "arch_type", "crnn") or "crnn").lower()
    if arch_type not in {"crnn", "conv_only"}:
        raise ValueError(f"unknown boundary scorer arch_type: {arch_type!r}")

    class BoundaryScorer(nn.Module):
        def __init__(self, cfg: BoundaryScorerConfig):
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
                # conv_only: encoder already emits hidden*2 channels, no RNN.
                self.encoder = _build_conv_only_encoder(cfg)
                self.rnn = None
            self.boundary_head = nn.Linear(encoded, len(cfg.labels))
            self.quality_head = nn.Linear(encoded, 1) if bool(cfg.enable_quality_head) else None
            self.cvs_head = nn.Linear(encoded, len(cfg.cvs_labels)) if bool(cfg.enable_phone_aux_heads) else None
            self.consonant_head = (
                nn.Linear(encoded, len(cfg.consonant_labels)) if bool(cfg.enable_phone_aux_heads) else None
            )
            self.vowel_head = nn.Linear(encoded, len(cfg.vowel_labels)) if bool(cfg.enable_phone_aux_heads) else None
            self.consonant_family_head = (
                nn.Linear(encoded, len(cfg.consonant_family_labels))
                if bool(cfg.enable_phone_family_heads)
                else None
            )
            self.vowel_nucleus_head = (
                nn.Linear(encoded, len(cfg.vowel_nucleus_labels))
                if bool(cfg.enable_phone_family_heads)
                else None
            )
            self.vowel_glide_head = (
                nn.Linear(encoded, len(cfg.vowel_glide_labels))
                if bool(cfg.enable_phone_family_heads)
                else None
            )

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
                "cvs_logits": None,
                "consonant_logits": None,
                "vowel_logits": None,
                "consonant_family_logits": None,
                "vowel_nucleus_logits": None,
                "vowel_glide_logits": None,
            }
            if self.quality_head is not None:
                out["quality_logits"] = self.quality_head(encoded).squeeze(-1)
            if self.cvs_head is not None:
                out["cvs_logits"] = self.cvs_head(encoded)
            if self.consonant_head is not None:
                out["consonant_logits"] = self.consonant_head(encoded)
            if self.vowel_head is not None:
                out["vowel_logits"] = self.vowel_head(encoded)
            if self.consonant_family_head is not None:
                out["consonant_family_logits"] = self.consonant_family_head(encoded)
            if self.vowel_nucleus_head is not None:
                out["vowel_nucleus_logits"] = self.vowel_nucleus_head(encoded)
            if self.vowel_glide_head is not None:
                out["vowel_glide_logits"] = self.vowel_glide_head(encoded)
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
