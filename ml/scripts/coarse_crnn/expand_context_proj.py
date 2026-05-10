"""Expand context_proj weights from 12-dim → 15-dim for vowel-ID context features.

New dims (12, 13, 14) are zero-initialized so the model's output is identical
to the old model on startup; the new features are learned from scratch.

Usage:
    python ml/scripts/coarse_crnn/expand_context_proj.py \
        --src ml_workspace/models/coarse_crnn/oto_anchor_crnn_e3_finetune.pt \
        --dst ml_workspace/models/coarse_crnn/oto_anchor_crnn_ctx15_warmstart.pt \
        --old-dim 12 --new-dim 15
"""
from __future__ import annotations

import argparse
import copy

import torch


def expand_linear_weight(w: torch.Tensor, old_dim: int, new_dim: int) -> torch.Tensor:
    """Expand [out, old_dim] → [out, new_dim] by zero-padding new columns."""
    assert w.shape[1] == old_dim, f"expected in_features={old_dim}, got {w.shape[1]}"
    extra = torch.zeros(w.shape[0], new_dim - old_dim, dtype=w.dtype, device=w.device)
    return torch.cat([w, extra], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--old-dim", type=int, default=12)
    parser.add_argument("--new-dim", type=int, default=15)
    args = parser.parse_args()

    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    # Support both checkpoint layouts: {state_dict, config, ...} and {model_state, config, ...}
    state: dict = payload.get("state_dict", payload.get("model_state", payload))
    cfg: dict = payload.get("config", {})

    old_dim = int(args.old_dim)
    new_dim = int(args.new_dim)

    # The context projection is named "context_proj.0.weight" in OtoCrnnModel
    proj_key = "context_proj.0.weight"
    if proj_key not in state:
        raise KeyError(f"Key '{proj_key}' not found in checkpoint. Available keys with 'context': "
                       + str([k for k in state if "context" in k]))

    w = state[proj_key]
    print(f"[expand_context_proj] {proj_key}: {list(w.shape)} → expanding to [{w.shape[0]}, {new_dim}]")
    state[proj_key] = expand_linear_weight(w, old_dim, new_dim)

    # Update config so the saved checkpoint reflects new dim
    if "numeric_context_dim" in cfg:
        cfg["numeric_context_dim"] = new_dim
        print(f"[expand_context_proj] config.numeric_context_dim: {old_dim} → {new_dim}")
    else:
        print(f"[expand_context_proj] WARNING: 'numeric_context_dim' not in config dict; the saved checkpoint may not auto-load correctly")

    new_payload = copy.copy(payload)
    # Write back under whichever key was used in the source
    if "state_dict" in payload:
        new_payload["state_dict"] = state
    else:
        new_payload["model_state"] = state
    new_payload["config"] = cfg

    torch.save(new_payload, args.dst)
    print(f"[expand_context_proj] saved → {args.dst}")


if __name__ == "__main__":
    main()
