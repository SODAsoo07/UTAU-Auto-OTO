from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig
from core.phoneme_boundary.targets import read_boundary_manifest
from core.phoneme_boundary.training import PhonemeBoundaryTrainConfig, train_phoneme_boundary_from_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the phoneme-boundary detector. This does not train OTO parameters.")
    ap.add_argument("--manifest", required=True, help="JSON/JSONL rows with wav_path/audio plus boundary events or phone intervals.")
    ap.add_argument("--out", default=os.path.join("ml_workspace", "models", "phoneme_boundary", "phoneme_boundary_v1.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--log-every", type=int, default=80)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--conv-channels", type=int, default=96)
    ap.add_argument("--arch", choices=["crnn", "conv_only"], default="crnn")
    ap.add_argument("--phone-state-head", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--phone-identity-heads", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--quality-loss-weight", type=float, default=0.03)
    ap.add_argument("--phone-state-loss-weight", type=float, default=0.35)
    ap.add_argument("--consonant-loss-weight", type=float, default=0.20)
    ap.add_argument("--vowel-loss-weight", type=float, default=0.20)
    args = ap.parse_args()

    rows = read_boundary_manifest(args.manifest)
    model_cfg = PhonemeBoundaryDetectorConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        arch_type=str(args.arch),
        enable_phone_state_head=bool(args.phone_state_head),
        enable_phone_identity_heads=bool(args.phone_identity_heads),
    )
    train_cfg = PhonemeBoundaryTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        val_ratio=float(args.val_ratio),
        log_every=int(args.log_every),
        pos_weight=float(args.pos_weight),
        quality_loss_weight=float(args.quality_loss_weight),
        phone_state_loss_weight=float(args.phone_state_loss_weight),
        consonant_loss_weight=float(args.consonant_loss_weight),
        vowel_loss_weight=float(args.vowel_loss_weight),
    )
    result = train_phoneme_boundary_from_manifest(
        rows,
        args.out,
        manifest_dir=os.path.dirname(os.path.abspath(args.manifest)),
        train_config=train_cfg,
        model_config=model_cfg,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
