from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.data_discovery import read_manifest
from core.coarse_crnn.evaluate import AlignmentEvalConfig, evaluate_manifest, write_eval_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate coarse_crnn alignment quality against gold TextGrid/timed lab.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "manifest_full.jsonl"))
    parser.add_argument("--model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "coarse_crnn.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-per-language", type=int, default=100)
    parser.add_argument("--source", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "alignment_eval.json"))
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    cfg = AlignmentEvalConfig(
        model_path=args.model,
        device=args.device,
        max_items=int(args.max_items),
        max_per_language=int(args.max_per_language),
        source=str(args.source or ""),
        seed=int(args.seed),
    )
    result = evaluate_manifest(rows, cfg)
    write_eval_json(args.out, result)
    compact = {key: value for key, value in result.items() if key not in {"files", "failures", "model_meta"}}
    compact["out"] = os.path.abspath(args.out)
    compact["failures"] = result.get("failures", [])[:5]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if int(result.get("evaluated_items", 0) or 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
