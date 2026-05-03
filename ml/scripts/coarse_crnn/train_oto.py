from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.oto_model import OtoCrnnConfig
from core.coarse_crnn.oto_targets import read_jsonl
from core.coarse_crnn.oto_training import OtoTrainConfig, train_oto_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CRNN-based OTO parameter predictor.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--val-manifest", default="", help="Optional fixed validation manifest, preferably voicebank-held-out.")
    parser.add_argument("--out", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional deterministic cap for smoke tests.")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Optional deterministic cap for fixed validation rows.")
    parser.add_argument("--vcv-loss-weight", type=float, default=1.35)
    parser.add_argument("--cvvc-loss-weight", type=float, default=1.15)
    parser.add_argument("--cvc-loss-weight", type=float, default=1.05)
    parser.add_argument("--format-residual-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vcv-target-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcv-window-frames", type=int, default=240)
    # cvvc excluded: vc/vv aliases are not uniformly distributed across audio
    # (CV rows come first, VC rows later), so row_ratio-based windowing yields
    # 1000-2500 ms offset errors. The heatmap finds the correct position instead.
    parser.add_argument("--target-window-formats", default="vcv")
    parser.add_argument("--target-window-frame-overrides", default="vcv=240")
    parser.add_argument("--cvvc-target-window-alias-types", default="")
    parser.add_argument("--anchor-heatmap-blend", type=float, default=0.70)
    parser.add_argument("--vcv-window-heatmap-blend", type=float, default=0.30)
    parser.add_argument(
        "--scalar-target-mode",
        choices=("relative_params", "absolute_anchors"),
        default="relative_params",
        help="relative_params predicts offset/pre deltas plus constrained consonant/cutoff tail gaps.",
    )
    parser.add_argument("--right-boundary-prior-blend", type=float, default=0.45)
    parser.add_argument("--right-boundary-prior-blends", default="vcv=0.45,cvvc=0.25,cv=0.10,cvc=0.10,other=0.10")
    parser.add_argument(
        "--alias-role-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add alias_role embeddings (-CV/CV/CV-/V/V-/VC/VV/V-CV/EndBR/BR/OTHER/special) to the model conditioning. New checkpoint architecture; not loadable by older inference paths without role plumbing.",
    )
    parser.add_argument(
        "--extra-alias-flags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a small projection that ingests is_diphthong and is_special as scalar conditioning features.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    val_rows = read_jsonl(args.val_manifest) if str(args.val_manifest or "").strip() else None
    if val_rows is not None and int(args.max_val_rows) > 0:
        val_rows = val_rows[: int(args.max_val_rows)]
    model_cfg = OtoCrnnConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        enable_format_residual_heads=bool(args.format_residual_heads),
        enable_vcv_target_window=bool(args.vcv_target_window),
        vcv_target_window_frames=int(args.vcv_window_frames),
        target_window_formats=tuple(item.strip().lower() for item in str(args.target_window_formats).split(",") if item.strip()),
        target_window_frame_overrides=tuple(item.strip().lower() for item in str(args.target_window_frame_overrides).split(",") if item.strip()),
        cvvc_target_window_alias_types=tuple(item.strip().lower() for item in str(args.cvvc_target_window_alias_types).split(",") if item.strip()),
        anchor_heatmap_blend=float(args.anchor_heatmap_blend),
        vcv_window_heatmap_blend=float(args.vcv_window_heatmap_blend),
        scalar_target_mode=str(args.scalar_target_mode),
        right_boundary_prior_blend=float(args.right_boundary_prior_blend),
        right_boundary_prior_blends=tuple(item.strip().lower() for item in str(args.right_boundary_prior_blends).split(",") if item.strip()),
        enable_alias_role_embedding=bool(args.alias_role_embedding),
        enable_extra_alias_flags=bool(args.extra_alias_flags),
    )
    train_cfg = OtoTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
        val_ratio=float(args.val_ratio),
        vcv_loss_weight=float(args.vcv_loss_weight),
        cvvc_loss_weight=float(args.cvvc_loss_weight),
        cvc_loss_weight=float(args.cvc_loss_weight),
    )
    result = train_oto_from_manifest(rows, args.out, val_rows=val_rows, train_config=train_cfg, model_config=model_cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
