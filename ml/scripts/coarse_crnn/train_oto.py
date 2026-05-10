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
    parser.add_argument("--num-workers", type=int, default=0, help="0 keeps single-process loading, -1 means auto worker count.")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-cache-dir", default=os.path.join("ml_workspace", "coarse_crnn", "feature_cache"))
    parser.add_argument("--feature-cache-readonly", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--vc-role-loss-weight", type=float, default=2.5,
                        help="Loss multiplier for vc-role aliases (hard to predict in cvvc voicebanks).")
    parser.add_argument("--vv-role-loss-weight", type=float, default=2.0,
                        help="Loss multiplier for vv-role aliases.")
    parser.add_argument("--format-residual-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vcv-target-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcv-window-frames", type=int, default=240)
    # cvvc excluded: vc/vv aliases are not uniformly distributed across audio
    # (CV rows come first, VC rows later), so row_ratio-based windowing yields
    # 1000-2500 ms offset errors. The heatmap finds the correct position instead.
    parser.add_argument("--target-window-formats", default="vcv")
    parser.add_argument("--target-window-frame-overrides", default="vcv=240")
    parser.add_argument("--target-window-role-frame-overrides", default="")
    parser.add_argument("--cvvc-target-window-alias-types", default="")
    parser.add_argument("--cvvc-target-window-alias-roles", default="vc,vv")
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
        "--right-boundary-prior-role-blends",
        default="-cv=0.35,cv=0.30,cv-=0.25,v=0.18,v-=0.22,vc=0.20,vv=0.24,v-cv=0.32,endbr=0.18,br=0.15,special=0.28,other=0.30",
    )
    parser.add_argument("--uncertainty-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.30)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.10)
    parser.add_argument("--confidence-target-error-scale", type=float, default=0.08)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voicebank-balance-power", type=float, default=0.55)
    parser.add_argument("--language-balance-power", type=float, default=0.0)
    parser.add_argument("--role-balance-power", type=float, default=0.35)
    parser.add_argument("--format-balance-power", type=float, default=0.20)
    parser.add_argument("--cvvc-vc-sampling-boost", type=float, default=1.0)
    parser.add_argument("--cvvc-vv-sampling-boost", type=float, default=1.0)
    parser.add_argument("--cvvc-vc-multi-sampling-boost", type=float, default=1.0)
    parser.add_argument("--language-format-role-sampling-boosts", default="", help="Comma-separated 'lang/fmt/role=N' boost specs")
    parser.add_argument("--row-order-violation-alpha", type=float, default=0.0, help=">0이면 순서 이상 row의 loss를 down-weight")
    parser.add_argument("--hard-case-mining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-case-top-ratio", type=float, default=0.25)
    parser.add_argument("--hard-case-boost", type=float, default=2.5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--selection-val-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-hard-failure-weight", type=float, default=2.5)
    parser.add_argument("--selection-worst-voicebank-weight", type=float, default=1.5)
    parser.add_argument("--selection-worst-voicebank-target-acc50", type=float, default=0.50)
    parser.add_argument("--checkpoint-save-every-epochs", type=int, default=0, help=">0이면 N epoch마다 중간 체크포인트 저장")
    parser.add_argument("--init-from", default="", help="기존 체크포인트에서 가중치를 불러와 fine-tune 시작 (compatible 전략)")
    parser.add_argument("--two-stage-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-stage-refine-window-frames", type=int, default=320)
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
        target_window_role_frame_overrides=tuple(item.strip().lower() for item in str(args.target_window_role_frame_overrides).split(",") if item.strip()),
        cvvc_target_window_alias_types=tuple(item.strip().lower() for item in str(args.cvvc_target_window_alias_types).split(",") if item.strip()),
        cvvc_target_window_alias_roles=tuple(item.strip().lower() for item in str(args.cvvc_target_window_alias_roles).split(",") if item.strip()),
        anchor_heatmap_blend=float(args.anchor_heatmap_blend),
        vcv_window_heatmap_blend=float(args.vcv_window_heatmap_blend),
        scalar_target_mode=str(args.scalar_target_mode),
        right_boundary_prior_blend=float(args.right_boundary_prior_blend),
        right_boundary_prior_blends=tuple(item.strip().lower() for item in str(args.right_boundary_prior_blends).split(",") if item.strip()),
        right_boundary_prior_role_blends=tuple(item.strip().lower() for item in str(args.right_boundary_prior_role_blends).split(",") if item.strip()),
        enable_uncertainty_head=bool(args.uncertainty_head),
        enable_two_stage_refine=bool(args.two_stage_refine),
        two_stage_refine_window_frames=int(args.two_stage_refine_window_frames),
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
        dataloader_prefetch_factor=max(1, int(args.dataloader_prefetch_factor)),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
        enable_tf32=bool(args.tf32),
        enable_cudnn_benchmark=bool(args.cudnn_benchmark),
        enable_feature_cache=bool(args.feature_cache),
        feature_cache_dir=str(args.feature_cache_dir),
        feature_cache_readonly=bool(args.feature_cache_readonly),
        log_every=int(args.log_every),
        val_ratio=float(args.val_ratio),
        vcv_loss_weight=float(args.vcv_loss_weight),
        cvvc_loss_weight=float(args.cvvc_loss_weight),
        cvc_loss_weight=float(args.cvc_loss_weight),
        vc_role_loss_weight=float(args.vc_role_loss_weight),
        vv_role_loss_weight=float(args.vv_role_loss_weight),
        uncertainty_loss_weight=float(args.uncertainty_loss_weight),
        confidence_loss_weight=float(args.confidence_loss_weight),
        confidence_target_error_scale=float(args.confidence_target_error_scale),
        enable_balanced_sampling=bool(args.balanced_sampling),
        voicebank_balance_power=float(args.voicebank_balance_power),
        language_balance_power=float(args.language_balance_power),
        role_balance_power=float(args.role_balance_power),
        format_balance_power=float(args.format_balance_power),
        cvvc_vc_sampling_boost=float(args.cvvc_vc_sampling_boost),
        cvvc_vv_sampling_boost=float(args.cvvc_vv_sampling_boost),
        cvvc_vc_multi_sampling_boost=float(args.cvvc_vc_multi_sampling_boost),
        language_format_role_sampling_boosts=tuple(
            item.strip() for item in str(args.language_format_role_sampling_boosts or "").split(",") if item.strip()
        ),
        row_order_violation_alpha=float(args.row_order_violation_alpha),
        enable_hard_case_mining=bool(args.hard_case_mining),
        hard_case_top_ratio=float(args.hard_case_top_ratio),
        hard_case_boost=float(args.hard_case_boost),
        early_stop_patience=int(args.early_stop_patience),
        selection_val_loss_weight=float(args.selection_val_loss_weight),
        selection_hard_failure_weight=float(args.selection_hard_failure_weight),
        selection_worst_voicebank_weight=float(args.selection_worst_voicebank_weight),
        selection_worst_voicebank_target_acc50=float(args.selection_worst_voicebank_target_acc50),
        checkpoint_save_every_epochs=max(0, int(args.checkpoint_save_every_epochs)),
    )
    result = train_oto_from_manifest(rows, args.out, val_rows=val_rows, train_config=train_cfg, model_config=model_cfg, init_checkpoint=str(args.init_from or ""))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
