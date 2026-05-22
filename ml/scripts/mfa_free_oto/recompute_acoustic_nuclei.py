from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.acoustic_nucleus import (
    AcousticNucleusConfig,
    relabel_vowel_nuclei_manifest,
    write_summary_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute vowel_nucleus events from acoustic evidence while preserving lab midpoint diagnostics."
    )
    parser.add_argument("--manifest", required=True, help="Input manual-gold JSONL manifest.")
    parser.add_argument("--out", required=True, help="Output JSONL manifest with acoustic vowel_nucleus labels.")
    parser.add_argument("--summary-out", help="Optional JSON summary path.")
    parser.add_argument("--encoder", default="acoustic_world_v1")
    parser.add_argument("--device", help="Optional torch device for SSL encoders.")
    parser.add_argument("--edge-margin-ms", type=float, default=28.0)
    parser.add_argument("--min-edge-margin-ratio", type=float, default=0.18)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    args = parser.parse_args()

    config = AcousticNucleusConfig(
        edge_margin_ms=args.edge_margin_ms,
        min_edge_margin_ratio=args.min_edge_margin_ratio,
        min_confidence=args.min_confidence,
    )
    _rows, summary = relabel_vowel_nuclei_manifest(
        args.manifest,
        args.out,
        encoder=args.encoder,
        device=args.device,
        config=config,
    )
    payload = {
        **asdict(summary),
        "manifest": args.manifest,
        "out": args.out,
        "encoder": args.encoder,
        "config": asdict(config),
    }
    if args.summary_out:
        write_summary_json(args.summary_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
