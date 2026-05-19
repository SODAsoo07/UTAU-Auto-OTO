from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.mfa_free_oto.evaluation import evaluate_manifest
from core.mfa_free_oto.training import TrainPocConfig, train_poc


DEFAULT_ENCODERS = ("acoustic", "acoustic-aux", "wavlm-base-plus", "wavlm-base-plus+aux")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run small MFA-free OTO feature ablations over acoustic, SSL, aux, and slot-Viterbi settings."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", action="append", help="Encoder to test. Repeatable. Defaults to acoustic/acoustic-aux/WavLM variants.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument(
        "--frame-class-weighting",
        default="none",
        choices=("none", "balanced"),
        help="Frame CE class weighting. Use balanced for very small imbalanced goldsets.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoders = tuple(args.encoder or DEFAULT_ENCODERS)
    summaries = []
    for encoder in encoders:
        ckpt = out_dir / f"{_safe_name(encoder)}.pt"
        try:
            if not (args.skip_existing and ckpt.exists()):
                train_poc(
                    TrainPocConfig(
                        manifest_path=args.train_manifest,
                        out_path=str(ckpt),
                        encoder=encoder,
                        epochs=args.epochs,
                        hidden_dim=args.hidden_dim,
                        layers=args.layers,
                        max_rows=args.max_rows,
                        device=args.device,
                        specaugment=args.specaugment,
                        frame_class_weighting=args.frame_class_weighting,
                    )
                )
            for use_slot in (False, True):
                metrics = evaluate_manifest(
                    args.eval_manifest,
                    ckpt,
                    encoder=encoder,
                    device=args.device,
                    use_slot_viterbi=use_slot,
                )
                metrics_path = out_dir / f"{_safe_name(encoder)}__slot_{int(use_slot)}.json"
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                summaries.append(_summary_row(encoder, use_slot, metrics, metrics_path))
        except Exception as exc:
            summaries.append({"encoder": encoder, "status": "error", "error": str(exc)})
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "runs": summaries}, ensure_ascii=False, indent=2))
    return 0


def _summary_row(encoder: str, use_slot: bool, metrics: dict, metrics_path: Path) -> dict:
    return {
        "encoder": encoder,
        "use_slot_viterbi": use_slot,
        "status": "ok",
        "metrics_path": str(metrics_path),
        "frame_macro_f1": metrics.get("frame_macro_f1"),
        "cv_boundary_median_error_ms": metrics.get("cv_boundary_median_error_ms"),
        "cv_boundary_p90_error_ms": metrics.get("cv_boundary_p90_error_ms"),
        "hard_failure_rate": metrics.get("hard_failure_rate"),
        "catastrophic_row_mismatch_rate": metrics.get("catastrophic_row_mismatch_rate"),
        "passed": (metrics.get("gate_report") or {}).get("passed"),
    }


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace("+", "_plus_")


if __name__ == "__main__":
    raise SystemExit(main())
