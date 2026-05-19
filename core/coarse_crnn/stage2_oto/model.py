from __future__ import annotations

from typing import Any

from core.coarse_crnn.stage2_oto.types import (
    Stage2CheckpointMeta,
    Stage2ModelConfig,
)


def _torch():
    import torch

    return torch


class Stage2OtoAssigner(_torch().nn.Module):
    """Sequence-aware OTO anchor assigner.

    The model consumes row identity, role/alias context, BoundaryScorer/PB/audio
    candidate summaries and the deterministic boundary-decoder anchor as an
    acoustic fallback feature. It does not consume source OTO parameter values.
    """

    def __init__(self, config: Stage2ModelConfig):
        torch = _torch()
        super().__init__()
        self.config = config
        self.role_emb = torch.nn.Embedding(config.role_count, config.role_emb_dim)
        self.prev_role_emb = torch.nn.Embedding(config.role_count, config.role_emb_dim)
        self.next_role_emb = torch.nn.Embedding(config.role_count, config.role_emb_dim)
        self.alias_type_emb = torch.nn.Embedding(config.alias_type_count, config.alias_type_emb_dim)
        self.language_emb = torch.nn.Embedding(config.language_count, config.language_emb_dim)
        self.format_emb = torch.nn.Embedding(config.format_count, config.format_emb_dim)
        in_dim = (
            int(config.numeric_dim)
            + int(config.role_emb_dim) * 3
            + int(config.alias_type_emb_dim)
            + int(config.language_emb_dim)
            + int(config.format_emb_dim)
        )
        hidden = int(config.hidden_dim)
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.08),
        )
        self.encoder = torch.nn.GRU(
            hidden,
            hidden,
            num_layers=max(1, int(config.gru_layers)),
            batch_first=True,
            bidirectional=True,
            dropout=0.08 if int(config.gru_layers) > 1 else 0.0,
        )
        head_in = hidden * 2
        self.anchor_head = torch.nn.Sequential(
            torch.nn.Linear(head_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 5),
        )
        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(head_in, max(16, hidden // 2)),
            torch.nn.GELU(),
            torch.nn.Linear(max(16, hidden // 2), 1),
        )

    def forward(
        self,
        numeric,
        role_id,
        prev_role_id,
        next_role_id,
        alias_type_id,
        language_id,
        format_id,
    ):
        torch = _torch()
        emb = torch.cat(
            [
                self.role_emb(role_id),
                self.prev_role_emb(prev_role_id),
                self.next_role_emb(next_role_id),
                self.alias_type_emb(alias_type_id),
                self.language_emb(language_id),
                self.format_emb(format_id),
            ],
            dim=-1,
        )
        x = torch.cat([numeric, emb], dim=-1)
        x = self.input_proj(x)
        y, _ = self.encoder(x)
        raw = self.anchor_head(y)
        anchors_norm = _structured_anchors_norm(torch, raw)
        confidence = torch.sigmoid(self.confidence_head(y)).squeeze(-1)
        return anchors_norm, confidence


def _structured_anchors_norm(torch, raw):
    """Map unconstrained logits to monotonic normalized anchors.

    This keeps output ordering stable in-model:
    offset <= overlap <= pre <= consonant <= cutoff <= 1.
    """
    offset = torch.sigmoid(raw[..., 0]).clamp(0.0, 0.995)

    remain = (1.0 - offset).clamp(min=1.0e-5)
    step = torch.sigmoid(raw[..., 1]) * remain
    overlap = offset + step

    remain = (1.0 - overlap).clamp(min=1.0e-5)
    step = torch.sigmoid(raw[..., 2]) * remain
    pre = overlap + step

    remain = (1.0 - pre).clamp(min=1.0e-5)
    step = torch.sigmoid(raw[..., 3]) * remain
    consonant = pre + step

    remain = (1.0 - consonant).clamp(min=1.0e-5)
    step = torch.sigmoid(raw[..., 4]) * remain
    cutoff = consonant + step

    return torch.stack([offset, overlap, pre, consonant, cutoff], dim=-1)


def build_stage2_model(config: Stage2ModelConfig) -> Stage2OtoAssigner:
    if config.numeric_dim <= 0:
        raise ValueError("Stage2 numeric_dim must be positive")
    return Stage2OtoAssigner(config)


def checkpoint_payload(
    *,
    model: Stage2OtoAssigner,
    config: Stage2ModelConfig,
    meta: Stage2CheckpointMeta | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "meta": (meta or Stage2CheckpointMeta()).to_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def load_stage2_checkpoint(path: str, *, map_location: str | None = "cpu"):
    torch = _torch()
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Invalid Stage2 checkpoint payload")
    config_raw = payload.get("model_config") or payload.get("config") or {}
    if not isinstance(config_raw, dict):
        raise ValueError("Invalid Stage2 checkpoint config")
    config = Stage2ModelConfig.from_mapping(config_raw)
    model = build_stage2_model(config)
    state = payload.get("state_dict") or payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Stage2 checkpoint missing state_dict")
    model.load_state_dict(state, strict=True)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return model, config, meta


__all__ = [
    "Stage2OtoAssigner",
    "build_stage2_model",
    "checkpoint_payload",
    "load_stage2_checkpoint",
]
