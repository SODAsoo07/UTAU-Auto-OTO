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
    parser.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-min-evaluated-items", type=int, default=200)
    parser.add_argument("--gate-min-preutterance-acc-50ms", type=float, default=0.72)
    parser.add_argument(
        "--gate-max-param-mae-ms",
        default="offset=90,consonant=90,cutoff_abs=140,preutterance=55,overlap=70",
    )
    parser.add_argument("--gate-alias-context-min-files", type=int, default=25)
    parser.add_argument("--gate-alias-context-min-preutterance-acc-50ms", type=float, default=0.55)
    parser.add_argument("--gate-max-hard-failure-rate", type=float, default=0.08)
    parser.add_argument("--gate-min-worst-voicebank-preutterance-acc-50ms", type=float, default=0.45)
    parser.add_argument("--gate-max-worst-voicebank-hard-failure-rate", type=float, default=0.22)
    parser.add_argument("--gate-max-fp-breath-include-rate", type=float, default=0.20)
    parser.add_argument("--gate-max-fn-voiced-miss-rate", type=float, default=0.55)
    parser.add_argument("--gate-max-s-family-leak-rate", type=float, default=0.25)
    parser.add_argument("--gate-max-onset-lag-p90-ms", type=float, default=380.0)
    parser.add_argument("--gate-max-syllable-shift-rate", type=float, default=0.45)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    cfg = OtoEvalConfig(
        model_path=args.model,
        device=str(args.device),
        max_items=int(args.max_items),
        seed=int(args.seed),
        language=str(args.language or ""),
        format_type=str(args.format_type or ""),
        gate_enabled=bool(args.gate),
        gate_min_evaluated_items=int(args.gate_min_evaluated_items),
        gate_min_preutterance_acc_50ms=float(args.gate_min_preutterance_acc_50ms),
        gate_max_param_mae_ms=tuple(
            item.strip() for item in str(args.gate_max_param_mae_ms or "").split(",") if item.strip()
        ),
        gate_alias_context_min_files=int(args.gate_alias_context_min_files),
        gate_alias_context_min_preutterance_acc_50ms=float(args.gate_alias_context_min_preutterance_acc_50ms),
        gate_max_hard_failure_rate=float(args.gate_max_hard_failure_rate),
        gate_min_worst_voicebank_preutterance_acc_50ms=float(args.gate_min_worst_voicebank_preutterance_acc_50ms),
        gate_max_worst_voicebank_hard_failure_rate=float(args.gate_max_worst_voicebank_hard_failure_rate),
        gate_max_fp_breath_include_rate=float(args.gate_max_fp_breath_include_rate),
        gate_max_fn_voiced_miss_rate=float(args.gate_max_fn_voiced_miss_rate),
        gate_max_s_family_leak_rate=float(args.gate_max_s_family_leak_rate),
        gate_max_onset_lag_p90_ms=float(args.gate_max_onset_lag_p90_ms),
        gate_max_syllable_shift_rate=float(args.gate_max_syllable_shift_rate),
    )
    result = evaluate_oto_manifest(rows, cfg)
    write_oto_eval_json(args.out, result)
    compact = {key: value for key, value in result.items() if key not in {"files", "failures", "model_meta"}}
    compact["out"] = os.path.abspath(args.out)
    compact["failures"] = result.get("failures", [])[:5]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if int(result.get("evaluated_items", 0) or 0) <= 0:
        return 2
    gate = result.get("gate") or {}
    if bool(args.gate) and not bool(gate.get("passed", False)):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
