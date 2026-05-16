from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig
from core.coarse_crnn.boundary_scorer_training import BoundaryTrainConfig, train_boundary_from_manifest


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            text = str(raw).strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Train frame-level boundary scorer for OTO decoding.")
    ap.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits_full", "oto_train.jsonl"))
    ap.add_argument("--out", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_boundary_scorer_v1.pt"))
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
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--quality-loss-weight", type=float, default=0.06)
    ap.add_argument("--boundary-time-loss-weight", type=float, default=0.08)
    ap.add_argument("--hard-case-oversample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hard-case-weight", type=float, default=1.8)
    ap.add_argument("--hard-case-min-ratio", type=float, default=0.10)
    ap.add_argument("--hard-case-alias-regex", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rows = _read_jsonl(args.manifest)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    cfg = BoundaryScorerConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
    )
    tcfg = BoundaryTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
        val_ratio=float(args.val_ratio),
        quality_loss_weight=float(args.quality_loss_weight),
        pos_weight=float(args.pos_weight),
        boundary_time_loss_weight=float(args.boundary_time_loss_weight),
        hard_case_oversample=bool(args.hard_case_oversample),
        hard_case_weight=float(args.hard_case_weight),
        hard_case_min_ratio=float(args.hard_case_min_ratio),
        hard_case_alias_regex=str(args.hard_case_alias_regex or ""),
    )
    result = train_boundary_from_manifest(rows, args.out, train_config=tcfg, model_config=cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
