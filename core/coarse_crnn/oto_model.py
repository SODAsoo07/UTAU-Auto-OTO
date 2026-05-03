from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES


@dataclass
class OtoCrnnConfig:
    n_mels: int = 64
    hidden: int = 96
    conv_channels: int = 96
    rnn_layers: int = 1
    dropout: float = 0.10
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    cond_dim: int = 24
    numeric_context_dim: int = 12
    enable_format_residual_heads: bool = False
    enable_vcv_target_window: bool = True
    vcv_target_window_frames: int = 240
    target_window_formats: tuple[str, ...] = ("vcv", "cvvc")
    target_window_frame_overrides: tuple[str, ...] = ("vcv=240", "cvvc=360")
    cvvc_target_window_alias_types: tuple[str, ...] = ("vc", "vv")
    anchor_heatmap_blend: float = 0.70
    vcv_window_heatmap_blend: float = 0.30
    scalar_target_mode: str = "relative_params"
    right_boundary_prior_blend: float = 0.45
    right_boundary_prior_blends: tuple[str, ...] = ("vcv=0.45", "cvvc=0.25", "cv=0.10", "cvc=0.10", "other=0.10")
    anchor_names: tuple[str, ...] = tuple(OTO_ANCHOR_NAMES)
    languages: tuple[str, ...] = ("korean", "japanese")
    format_types: tuple[str, ...] = ("cv", "cvc", "cvvc", "vcv", "c_plus_v", "general", "other")
    alias_types: tuple[str, ...] = (
        "cv",
        "cv_head",
        "vc",
        "vv",
        "vcv",
        "br",
        "mono",
        "vowel",
        "consonant",
        "other",
    )
    transition_types: tuple[str, ...] = ("cv", "vc", "vv", "cc", "vowel", "consonant", "br", "multi", "other")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OtoCrnnConfig":
        data = dict(payload or {})
        if payload is not None and "numeric_context_dim" not in data:
            data["numeric_context_dim"] = 0
        if payload is not None and "alias_types" not in data:
            data["alias_types"] = ("other",)
        if payload is not None and "transition_types" not in data:
            data["transition_types"] = ("other",)
        if payload is not None and "enable_format_residual_heads" not in data:
            data["enable_format_residual_heads"] = False
        if payload is not None and "enable_vcv_target_window" not in data:
            data["enable_vcv_target_window"] = False
        if payload is not None and "vcv_target_window_frames" not in data:
            data["vcv_target_window_frames"] = 0
        if payload is not None and "target_window_formats" not in data:
            data["target_window_formats"] = ("vcv",)
        if payload is not None and "target_window_frame_overrides" not in data:
            data["target_window_frame_overrides"] = ()
        if payload is not None and "cvvc_target_window_alias_types" not in data:
            data["cvvc_target_window_alias_types"] = ("cv", "cv_head", "vc", "vv", "vcv", "mono", "br", "other")
        if payload is not None and "scalar_target_mode" not in data:
            data["scalar_target_mode"] = "absolute_anchors"
        if payload is not None and "right_boundary_prior_blend" not in data:
            data["right_boundary_prior_blend"] = 0.0
        if payload is not None and "right_boundary_prior_blends" not in data:
            data["right_boundary_prior_blends"] = ()
        for key in (
            "anchor_names",
            "languages",
            "format_types",
            "alias_types",
            "transition_types",
            "target_window_formats",
            "target_window_frame_overrides",
            "cvvc_target_window_alias_types",
            "right_boundary_prior_blends",
        ):
            if key in data and not isinstance(data[key], tuple):
                data[key] = tuple(data[key] or getattr(cls, key))
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["anchor_names"] = list(self.anchor_names)
        data["languages"] = list(self.languages)
        data["format_types"] = list(self.format_types)
        data["alias_types"] = list(self.alias_types)
        data["transition_types"] = list(self.transition_types)
        data["target_window_formats"] = list(self.target_window_formats)
        data["target_window_frame_overrides"] = list(self.target_window_frame_overrides)
        data["cvvc_target_window_alias_types"] = list(self.cvvc_target_window_alias_types)
        data["right_boundary_prior_blends"] = list(self.right_boundary_prior_blends)
        return data


def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_oto_model(config: OtoCrnnConfig):
    _torch, nn = _import_torch()

    class OtoAnchorCRNN(nn.Module):
        def __init__(self, cfg: OtoCrnnConfig):
            super().__init__()
            c = int(cfg.conv_channels)
            hidden2 = int(cfg.hidden) * 2
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
            lang_count = max(1, len(cfg.languages))
            fmt_count = max(1, len(cfg.format_types))
            alias_count = max(1, len(cfg.alias_types))
            transition_count = max(1, len(cfg.transition_types))
            self.language_emb = nn.Embedding(lang_count, int(cfg.cond_dim))
            self.format_emb = nn.Embedding(fmt_count, int(cfg.cond_dim))
            self.alias_emb = nn.Embedding(alias_count, int(cfg.cond_dim))
            self.transition_emb = nn.Embedding(transition_count, int(cfg.cond_dim))
            self.prev_alias_emb = nn.Embedding(alias_count, int(cfg.cond_dim))
            self.next_alias_emb = nn.Embedding(alias_count, int(cfg.cond_dim))
            self.prev_transition_emb = nn.Embedding(transition_count, int(cfg.cond_dim))
            self.next_transition_emb = nn.Embedding(transition_count, int(cfg.cond_dim))
            self.context_proj = (
                nn.Sequential(nn.Linear(int(cfg.numeric_context_dim), int(cfg.cond_dim)), nn.Tanh())
                if int(cfg.numeric_context_dim) > 0
                else None
            )
            cond_in = int(cfg.cond_dim) * (9 if self.context_proj is not None else 8)
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_in, hidden2),
                nn.Tanh(),
            )
            self.heatmap_head = nn.Sequential(
                nn.Linear(hidden2, hidden2),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(hidden2, len(cfg.anchor_names)),
            )
            self.scalar_head = nn.Sequential(
                nn.Linear(hidden2 * 2, hidden2),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(hidden2, len(cfg.anchor_names)),
            )
            if bool(cfg.enable_format_residual_heads):
                self.format_heatmap_heads = nn.ModuleList(
                    [nn.Linear(hidden2, len(cfg.anchor_names)) for _ in range(fmt_count)]
                )
                self.format_scalar_heads = nn.ModuleList(
                    [nn.Linear(hidden2 * 2, len(cfg.anchor_names)) for _ in range(fmt_count)]
                )
            else:
                self.format_heatmap_heads = None
                self.format_scalar_heads = None
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden2 * 2, hidden2 // 2),
                nn.ReLU(),
                nn.Linear(hidden2 // 2, 1),
            )

        def encode(self, x):
            y = x.transpose(1, 2)
            y = self.net(y).transpose(1, 2)
            y, _ = self.rnn(y)
            return y

        def forward(
            self,
            x,
            language_ids=None,
            format_ids=None,
            context=None,
            alias_type_ids=None,
            transition_type_ids=None,
            prev_alias_type_ids=None,
            next_alias_type_ids=None,
            prev_transition_type_ids=None,
            next_transition_type_ids=None,
        ):
            torch = __import__("torch")
            encoded = self.encode(x)
            batch = int(encoded.shape[0])
            device = encoded.device
            if language_ids is None:
                language_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if format_ids is None:
                format_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if alias_type_ids is None:
                alias_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if transition_type_ids is None:
                transition_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if prev_alias_type_ids is None:
                prev_alias_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if next_alias_type_ids is None:
                next_alias_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if prev_transition_type_ids is None:
                prev_transition_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            if next_transition_type_ids is None:
                next_transition_type_ids = torch.zeros((batch,), dtype=torch.long, device=device)
            language_ids = language_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.languages) - 1))
            format_ids = format_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.format_types) - 1))
            alias_type_ids = alias_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.alias_types) - 1))
            transition_type_ids = transition_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.transition_types) - 1))
            prev_alias_type_ids = prev_alias_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.alias_types) - 1))
            next_alias_type_ids = next_alias_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.alias_types) - 1))
            prev_transition_type_ids = prev_transition_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.transition_types) - 1))
            next_transition_type_ids = next_transition_type_ids.to(device=device, dtype=torch.long).clamp(0, max(0, len(config.transition_types) - 1))
            cond_parts = [
                self.language_emb(language_ids),
                self.format_emb(format_ids),
                self.alias_emb(alias_type_ids),
                self.transition_emb(transition_type_ids),
                self.prev_alias_emb(prev_alias_type_ids),
                self.next_alias_emb(next_alias_type_ids),
                self.prev_transition_emb(prev_transition_type_ids),
                self.next_transition_emb(next_transition_type_ids),
            ]
            if self.context_proj is not None:
                if context is None:
                    context = torch.zeros((batch, int(config.numeric_context_dim)), dtype=torch.float32, device=device)
                context = context.to(device=device, dtype=torch.float32)
                if int(context.shape[-1]) != int(config.numeric_context_dim):
                    context = torch.zeros((batch, int(config.numeric_context_dim)), dtype=torch.float32, device=device)
                cond_parts.append(self.context_proj(context))
            cond = self.cond_proj(torch.cat(cond_parts, dim=-1))
            encoded = encoded + cond[:, None, :]
            heatmap_logits = self.heatmap_head(encoded)
            pooled_mean = encoded.mean(dim=1)
            pooled_max = encoded.max(dim=1).values
            pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
            scalar_logits = self.scalar_head(pooled)
            if self.format_heatmap_heads is not None and self.format_scalar_heads is not None:
                heatmap_residual = torch.zeros_like(heatmap_logits)
                scalar_residual = torch.zeros_like(scalar_logits)
                for fmt_idx in torch.unique(format_ids).detach().cpu().tolist():
                    fmt_i = int(fmt_idx)
                    mask = format_ids == fmt_i
                    heatmap_residual[mask] = self.format_heatmap_heads[fmt_i](encoded[mask])
                    scalar_residual[mask] = self.format_scalar_heads[fmt_i](pooled[mask])
                heatmap_logits = heatmap_logits + 0.35 * heatmap_residual
                scalar_logits = scalar_logits + 0.35 * scalar_residual
            confidence_logits = self.confidence_head(pooled).squeeze(-1)
            return {
                "heatmap_logits": heatmap_logits,
                "scalar_logits": scalar_logits,
                "confidence_logits": confidence_logits,
            }

    return OtoAnchorCRNN(config)


def language_id(config: OtoCrnnConfig, language: object) -> int:
    text = str(language or "").strip().lower()
    try:
        return list(config.languages).index(text)
    except ValueError:
        return 0


def format_id(config: OtoCrnnConfig, format_type: object) -> int:
    text = str(format_type or "").strip().lower() or "other"
    try:
        return list(config.format_types).index(text)
    except ValueError:
        return list(config.format_types).index("other") if "other" in config.format_types else 0


def alias_type_id(config: OtoCrnnConfig, alias_type: object) -> int:
    text = str(alias_type or "").strip().lower() or "other"
    try:
        return list(config.alias_types).index(text)
    except ValueError:
        return list(config.alias_types).index("other") if "other" in config.alias_types else 0


def transition_type_id(config: OtoCrnnConfig, transition_type: object) -> int:
    text = str(transition_type or "").strip().lower() or "other"
    try:
        return list(config.transition_types).index(text)
    except ValueError:
        return list(config.transition_types).index("other") if "other" in config.transition_types else 0


def uses_relative_param_head(config: OtoCrnnConfig) -> bool:
    mode = str(getattr(config, "scalar_target_mode", "") or "").strip().lower()
    return mode in {"relative", "relative_params", "oto_params", "params"}


def right_boundary_prior_blend_for(config: OtoCrnnConfig, format_type: object) -> float:
    fmt = str(format_type or "").strip().lower() or "other"
    table = _parse_key_value_tuple(getattr(config, "right_boundary_prior_blends", ()))
    value = table.get(fmt, table.get("other"))
    if value is None:
        value = float(getattr(config, "right_boundary_prior_blend", 0.45))
    return max(0.0, min(1.0, float(value)))


def _parse_key_value_tuple(items: object) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or ():
        text = str(item or "").strip()
        if not text or "=" not in text:
            continue
        key, raw = text.split("=", 1)
        key = key.strip().lower()
        if not key:
            continue
        try:
            out[key] = float(raw)
        except Exception:
            continue
    return out


def save_oto_checkpoint(path: str, model, config: OtoCrnnConfig, *, meta: dict[str, Any] | None = None) -> None:
    torch, _nn = _import_torch()
    payload = {
        "backend": "oto_anchor_crnn_v1",
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, str(path))


def load_oto_checkpoint(path: str, *, map_location: str = "cpu"):
    torch, _nn = _import_torch()
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("oto checkpoint payload must be a dict")
    config = OtoCrnnConfig.from_dict(payload.get("config"))
    model = build_oto_model(config)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("oto checkpoint is missing state_dict")
    model.load_state_dict(state)
    model.eval()
    return model, config, dict(payload.get("meta") or {})


__all__ = [
    "OtoCrnnConfig",
    "alias_type_id",
    "build_oto_model",
    "format_id",
    "language_id",
    "load_oto_checkpoint",
    "right_boundary_prior_blend_for",
    "save_oto_checkpoint",
    "transition_type_id",
    "uses_relative_param_head",
]
