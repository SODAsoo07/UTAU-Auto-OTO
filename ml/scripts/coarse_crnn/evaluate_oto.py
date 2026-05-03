from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.oto_evaluate import OtoEvalConfig, evaluate_oto_manifest, write_oto_eval_json
from core.coarse_crnn.oto_targets import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the OTO anchor CRNN against oto.ini parameters.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="")
    parser.add_argument("--format-type", default="")
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "oto_eval.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    cfg = OtoEvalConfig(
        model_path=args.model,
        device=str(args.device),
        max_items=int(args.max_items),
        seed=int(args.seed),
        language=str(args.language or ""),
        format_type=str(args.format_type or ""),
    )
    result = evaluate_oto_manifest(rows, cfg)
    write_oto_eval_json(args.out, result)
    compact = {key: value for key, value in result.items() if key not in {"files", "failures", "model_meta"}}
    compact["out"] = os.path.abspath(args.out)
    compact["failures"] = result.get("failures", [])[:5]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if int(result.get("evaluated_items", 0) or 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
