from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.evaluation import evaluate_phoneme_boundary_manifest
from core.phoneme_boundary.targets import read_boundary_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate phoneme-boundary detector against explicit boundary events.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--search-radius-ms", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = read_boundary_manifest(args.manifest)
    result = evaluate_phoneme_boundary_manifest(
        rows,
        model_path=args.model,
        manifest_dir=os.path.dirname(os.path.abspath(args.manifest)),
        device=str(args.device),
        search_radius_ms=float(args.search_radius_ms),
    )
    if str(args.out or "").strip():
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
