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

        def forward(self, x):
            h = self.input_norm(x)
            h = self.proj(h)
            h_time = h.transpose(1, 2)
            for block in self.blocks:
                h_time = h_time + block(h_time)
            h = h_time.transpose(1, 2)
            return {
                "frame_logits": self.frame_head(h),
                "event_logits": self.event_head(h),
            }

    return MfaFreeFrameModel()
