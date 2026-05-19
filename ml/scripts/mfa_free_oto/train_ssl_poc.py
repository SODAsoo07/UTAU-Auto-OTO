from __future__ import annotations

import argparse
import json

from core.mfa_free_oto.training import TrainPocConfig, train_poc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the first MFA-free OTO SSL/acoustic multi-head PoC on manual gold labels."
    )
    parser.add_argument("--manifest", required=True, help="Manual-gold JSONL manifest.")
    parser.add_argument("--out", required=True, help="Output checkpoint path.")
    parser.add_argument(
        "--encoder",
        default="acoustic",
        help="Feature encoder: acoustic, wavlm-base-plus, xlsr-300m, or a Hugging Face model id.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda.")
    parser.add_argument("--event-loss-weight", type=float, default=1.0)
    parser.add_argument("--specaugment", action="store_true", help="Apply light time/frequency feature masking during training.")
    parser.add_argument("--time-mask-frames", type=int, default=8)
    parser.add_argument("--freq-mask-bins", type=int, default=8)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument(
        "--frame-class-weighting",
        default="none",
        choices=("none", "balanced"),
        help="Frame CE class weighting. Use balanced for very small imbalanced goldsets.",
    )
    args = parser.parse_args()

    result = train_poc(
        TrainPocConfig(
            manifest_path=args.manifest,
            out_path=args.out,
            encoder=args.encoder,
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            max_rows=args.max_rows,
            seed=args.seed,
            device=args.device,
            event_loss_weight=args.event_loss_weight,
            specaugment=args.specaugment,
            time_mask_frames=args.time_mask_frames,
            freq_mask_bins=args.freq_mask_bins,
            mask_count=args.mask_count,
            frame_class_weighting=args.frame_class_weighting,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
