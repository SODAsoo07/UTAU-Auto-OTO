from __future__ import annotations

import argparse
import json

from core.mfa_free_oto.evaluation import evaluate_manifest, save_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an MFA-free OTO PoC checkpoint against manual gold labels."
    )
    parser.add_argument("--manifest", required=True, help="Manual-gold JSONL manifest.")
    parser.add_argument("--checkpoint", required=True, help="PoC checkpoint from train_ssl_poc.py.")
    parser.add_argument("--out", help="Optional JSON metrics output path.")
    parser.add_argument("--encoder", help="Override encoder stored in checkpoint.")
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda.")
    parser.add_argument("--boundary-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--disable-slot-viterbi", action="store_true", help="Evaluate generic event decoding without filename-slot Viterbi.")
    args = parser.parse_args()

    metrics = evaluate_manifest(
        args.manifest,
        args.checkpoint,
        encoder=args.encoder,
        device=args.device,
        boundary_tolerance_ms=args.boundary_tolerance_ms,
        use_slot_viterbi=not args.disable_slot_viterbi,
    )
    if args.out:
        save_json(args.out, metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
