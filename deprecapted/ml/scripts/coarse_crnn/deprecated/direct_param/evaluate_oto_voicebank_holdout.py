from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from core.coarse_crnn.deprecated.direct_param.oto_evaluate import OtoEvalConfig, evaluate_oto_manifest
from core.coarse_crnn.oto_targets import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OTO model per voicebank (holdout-style).")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="")
    parser.add_argument("--format-type", default="")
    parser.add_argument("--max-items-per-bank", type=int, default=800)
    parser.add_argument("--min-items-per-bank", type=int, default=30)
    parser.add_argument("--max-banks", type=int, default=0)
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "oto_voicebank_holdout_eval.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    filtered = [
        row
        for row in rows
        if (not args.language or str(row.get("language", "") or "") == str(args.language))
        and (not args.format_type or str(row.get("format_type", "") or "") == str(args.format_type))
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        groups[str(row.get("voicebank_id", "") or "unknown_voicebank")].append(row)
    selected_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    if int(args.max_banks) > 0:
        selected_groups = selected_groups[: int(args.max_banks)]

    per_voicebank: list[dict[str, Any]] = []
    for idx, (voicebank_id, bank_rows) in enumerate(selected_groups):
        if len(bank_rows) < int(args.min_items_per_bank):
            continue
        cfg = OtoEvalConfig(
            model_path=str(args.model),
            device=str(args.device),
            seed=int(args.seed) + idx,
            max_items=int(args.max_items_per_bank),
            gate_enabled=False,
        )
        result = evaluate_oto_manifest(bank_rows, cfg)
        per_voicebank.append(
            {
                "voicebank_id": voicebank_id,
                "rows": len(bank_rows),
                "evaluated_items": int(result.get("evaluated_items", 0) or 0),
                "preutterance_acc_50ms": (result.get("preutterance_acc_50ms")),
                "hard_failure_rate": (result.get("hard_failure_rate")),
                "param_mae_ms": dict(result.get("param_mae_ms") or {}),
            }
        )

    worst_acc = min(
        per_voicebank,
        key=lambda item: float(item["preutterance_acc_50ms"]) if item.get("preutterance_acc_50ms") is not None else 1.0,
    ) if per_voicebank else None
    worst_fail = max(
        per_voicebank,
        key=lambda item: float(item["hard_failure_rate"]) if item.get("hard_failure_rate") is not None else 0.0,
    ) if per_voicebank else None
    payload = {
        "manifest": os.path.abspath(str(args.manifest)),
        "model": os.path.abspath(str(args.model)),
        "banks_total": len(selected_groups),
        "banks_evaluated": len(per_voicebank),
        "min_items_per_bank": int(args.min_items_per_bank),
        "max_items_per_bank": int(args.max_items_per_bank),
        "worst_preutterance_acc_50ms": worst_acc,
        "worst_hard_failure_rate": worst_fail,
        "per_voicebank": per_voicebank,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if per_voicebank else 2


if __name__ == "__main__":
    raise SystemExit(main())


