from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig
from core.phoneme_boundary.targets import read_boundary_manifest
from core.phoneme_boundary.training import PhonemeBoundaryTrainConfig, train_phoneme_boundary_from_manifest
from core.phoneme_boundary.types import BOUNDARY_LABELS, normalize_boundary_label


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
    ap.add_argument("--boundary-vowel-onset-weight", type=float, default=1.0)
    ap.add_argument("--boundary-vowel-nucleus-weight", type=float, default=1.0)
    ap.add_argument(
        "--boundary-label-weights",
        default="",
        help=(
            "Comma-separated label:weight overrides for frame-event BCE, "
            "e.g. consonant_onset:1.6,vowel_end:1.3,phone_start:0.7."
        ),
    )
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
        boundary_vowel_onset_weight=float(args.boundary_vowel_onset_weight),
        boundary_vowel_nucleus_weight=float(args.boundary_vowel_nucleus_weight),
        boundary_label_weights=_parse_boundary_label_weights(args.boundary_label_weights),
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


def _parse_boundary_label_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    allowed = set(BOUNDARY_LABELS)
    for item in str(text or "").split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"boundary label weight must be label:weight: {part!r}")
        label_raw, value_raw = part.split(":", 1)
        label = normalize_boundary_label(label_raw)
        if label not in allowed:
            raise ValueError(f"unknown boundary label weight: {label_raw!r}")
        value = float(value_raw)
        if value <= 0.0:
            raise ValueError(f"boundary label weight must be > 0: {part!r}")
        out[label] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
