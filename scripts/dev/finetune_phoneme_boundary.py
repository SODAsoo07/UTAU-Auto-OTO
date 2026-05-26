"""Fine-tune an existing phoneme-boundary checkpoint on a (small) gold manifest.
Reads the checkpoint's config so the architecture matches, then continues
training with a lower LR. By default disables the JP-unfriendly phone-identity
aux losses (the JP gold phones don't match the KO-centric label set)."""
from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig, load_phoneme_boundary_checkpoint
from core.phoneme_boundary.targets import read_boundary_manifest
from core.phoneme_boundary.training import PhonemeBoundaryTrainConfig, train_phoneme_boundary_from_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="Pretrained checkpoint to start from.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--phone-aux", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable phone-identity aux losses. Off by default for JP gold on KO-trained model.")
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    _model, cfg, _meta = load_phoneme_boundary_checkpoint(args.init, map_location="cpu")
    print(f"[finetune] loaded init config: arch={cfg.arch_type} n_mels={cfg.n_mels} hidden={cfg.hidden}")

    rows = read_boundary_manifest(args.manifest)
    print(f"[finetune] manifest rows: {len(rows)}")

    train_cfg = PhonemeBoundaryTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        val_ratio=float(args.val_ratio),
        device=str(args.device),
        num_workers=int(args.num_workers),
        pos_weight=float(args.pos_weight),
        phone_state_loss_weight=0.0 if not args.phone_aux else 0.35,
        consonant_loss_weight=0.0 if not args.phone_aux else 0.20,
        vowel_loss_weight=0.0 if not args.phone_aux else 0.20,
        quality_loss_weight=0.03,
        log_every=20,
    )
    result = train_phoneme_boundary_from_manifest(
        rows,
        args.out,
        manifest_dir=os.path.dirname(os.path.abspath(args.manifest)),
        train_config=train_cfg,
        model_config=cfg,
        init_checkpoint=args.init,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
