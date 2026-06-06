from __future__ import annotations

from dataclasses import asdict, dataclass

from .types import EVENT_LABELS, FRAME_LABELS


@dataclass(frozen=True)
class MfaFreeFrameModelConfig:
    input_dim: int
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.10
    frame_labels: tuple[str, ...] = FRAME_LABELS
    event_labels: tuple[str, ...] = EVENT_LABELS
    slot_event_bins: int = 0
    slot_event_position_features: bool = False
    slot_event_query_conditioned: bool = False
    slot_event_detach_backbone: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["frame_labels"] = list(self.frame_labels)
        data["event_labels"] = list(self.event_labels)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MfaFreeFrameModelConfig":
        return cls(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data.get("hidden_dim", 128)),
            layers=int(data.get("layers", 3)),
            dropout=float(data.get("dropout", 0.10)),
            frame_labels=tuple(data.get("frame_labels", FRAME_LABELS)),
            event_labels=tuple(data.get("event_labels", EVENT_LABELS)),
            slot_event_bins=max(0, int(data.get("slot_event_bins", 0))),
            slot_event_position_features=bool(data.get("slot_event_position_features", False)),
            slot_event_query_conditioned=bool(data.get("slot_event_query_conditioned", False)),
            slot_event_detach_backbone=bool(data.get("slot_event_detach_backbone", False)),
        )


def build_frame_model(config: MfaFreeFrameModelConfig):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Training/inference requires torch. Install ML dependencies first.") from exc

    class MfaFreeFrameModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.input_dim)
            self.proj = nn.Linear(config.input_dim, config.hidden_dim)
            blocks = []
            for _ in range(max(1, config.layers)):
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=2),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
                        nn.GELU(),
                    )
                )
            self.blocks = nn.ModuleList(blocks)
            self.frame_head = nn.Linear(config.hidden_dim, len(config.frame_labels))
            self.event_head = nn.Linear(config.hidden_dim, len(config.event_labels))
            self.slot_query_proj = (
                nn.Linear(4, config.hidden_dim)
                if int(config.slot_event_bins) > 0 and config.slot_event_query_conditioned
                else None
            )
            self.slot_event_key = (
                nn.Linear(config.hidden_dim, config.hidden_dim * len(config.event_labels))
                if self.slot_query_proj is not None
                else None
            )
            self.slot_event_head = (
                nn.Linear(
                    config.hidden_dim + (4 if config.slot_event_position_features else 0),
                    int(config.slot_event_bins) * len(config.event_labels),
                )
                if int(config.slot_event_bins) > 0 and self.slot_query_proj is None
                else None
            )

        def forward(self, x, *, slot_count: int | None = None):
            h = self.input_norm(x)
            h = self.proj(h)
            h_time = h.transpose(1, 2)
            for block in self.blocks:
                h_time = h_time + block(h_time)
            h = h_time.transpose(1, 2)
            output = {
                "frame_logits": self.frame_head(h),
                "event_logits": self.event_head(h),
            }
            if self.slot_query_proj is not None and self.slot_event_key is not None:
                active_slots = max(1, min(int(slot_count or config.slot_event_bins), int(config.slot_event_bins)))
                active_positions = torch.arange(
                    1,
                    active_slots + 1,
                    dtype=h.dtype,
                    device=h.device,
                ) / float(active_slots + 1)
                if active_slots < int(config.slot_event_bins):
                    inactive = torch.ones(
                        int(config.slot_event_bins) - active_slots,
                        dtype=h.dtype,
                        device=h.device,
                    )
                    positions = torch.cat((active_positions, inactive), dim=0)
                else:
                    positions = active_positions
                query_features = torch.stack(
                    (
                        positions,
                        positions.square(),
                        torch.sin(torch.pi * positions),
                        torch.cos(torch.pi * positions),
                    ),
                    dim=-1,
                )
                queries = self.slot_query_proj(query_features)
                slot_source = h.detach() if config.slot_event_detach_backbone else h
                keys = self.slot_event_key(slot_source).reshape(
                    h.shape[0],
                    h.shape[1],
                    len(config.event_labels),
                    config.hidden_dim,
                )
                slot_logits = torch.einsum("bteh,sh->btse", keys, queries)
                slot_logits = slot_logits / float(config.hidden_dim) ** 0.5
                frame_positions = torch.linspace(
                    0.0,
                    1.0,
                    int(h.shape[1]),
                    dtype=h.dtype,
                    device=h.device,
                )
                position_bias = -2.0 * (
                    (frame_positions[:, None] - positions[None, :]) / 0.20
                ).square()
                output["slot_event_logits"] = slot_logits + position_bias[None, :, :, None]
            elif self.slot_event_head is not None:
                slot_h = h
                if config.slot_event_position_features:
                    frame_count = max(1, int(h.shape[1]))
                    position = torch.linspace(0.0, 1.0, frame_count, dtype=h.dtype, device=h.device)
                    position_features = torch.stack(
                        (
                            position,
                            position.square(),
                            torch.sin(torch.pi * position),
                            torch.cos(torch.pi * position),
                        ),
                        dim=-1,
                    )
                    slot_h = torch.cat(
                        (h, position_features.unsqueeze(0).expand(h.shape[0], -1, -1)),
                        dim=-1,
                    )
                output["slot_event_logits"] = self.slot_event_head(slot_h).reshape(
                    h.shape[0],
                    h.shape[1],
                    int(config.slot_event_bins),
                    len(config.event_labels),
                )
            return output

    return MfaFreeFrameModel()
