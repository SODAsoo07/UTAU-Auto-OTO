from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class SweepCase:
    name: str
    cvvc_vc_sampling_boost: float
    cvvc_vv_sampling_boost: float
    cvvc_vc_multi_sampling_boost: float
    language_format_role_sampling_boosts: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_cases() -> list[SweepCase]:
    # 좁은 범위의 실사용 튜닝 4개: cvvc vc/vv 밸런스 + vc/multi 집중을 분리 검증
    return [
        SweepCase(
            name="cvvc_balanced_1p2",
            cvvc_vc_sampling_boost=1.2,
            cvvc_vv_sampling_boost=1.2,
            cvvc_vc_multi_sampling_boost=1.2,
            language_format_role_sampling_boosts="korean/cvvc/*=1.6,korean/cvvc/vc=1.4,korean/cvvc/vv=1.4,korean/cvc/v-cv=1.2",
        ),
        SweepCase(
            name="cvvc_vc_focus_2p0",
            cvvc_vc_sampling_boost=1.2,
            cvvc_vv_sampling_boost=1.0,
            cvvc_vc_multi_sampling_boost=2.0,
            language_format_role_sampling_boosts="korean/cvvc/*=1.6,korean/cvvc/vc=1.6,korean/cvvc/vv=1.2,korean/cvc/v-cv=1.2",
        ),
        SweepCase(
            name="cvvc_vc_multi_hard_3p0",
            cvvc_vc_sampling_boost=1.0,
            cvvc_vv_sampling_boost=1.0,
            cvvc_vc_multi_sampling_boost=3.0,
            language_format_role_sampling_boosts="korean/cvvc/*=1.6,korean/cvvc/vc=1.6,korean/cvvc/vv=1.2,korean/cvc/v-cv=1.2",
        ),
        SweepCase(
            name="cvvc_vc_vv_dual_2p4",
            cvvc_vc_sampling_boost=1.4,
            cvvc_vv_sampling_boost=1.2,
            cvvc_vc_multi_sampling_boost=2.4,
            language_format_role_sampling_boosts="korean/cvvc/*=1.7,korean/cvvc/vc=1.6,korean/cvvc/vv=1.4,korean/cvc/v-cv=1.2",
        ),
    ]


def _run(cmd: list[str], *, cwd: Path, allow_exit_codes: set[int] | None = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if allow_exit_codes is None:
        allow_exit_codes = {0}
    if proc.returncode not in allow_exit_codes:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}")
    return int(proc.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _score_eval(payload: dict[str, Any]) -> float:
    by_format = payload.get("by_format") or {}
    by_context = payload.get("by_alias_context") or {}
    cvvc = by_format.get("cvvc") or {}
    cvvc_vc_multi = by_context.get("cvvc|vc|multi") or {}
    return (
        float(payload.get("preutterance_acc_50ms", 0.0) or 0.0) * 1000.0
        - float(payload.get("hard_failure_rate", 0.0) or 0.0) * 180.0
        - float(payload.get("fn_voiced_miss_rate", 0.0) or 0.0) * 140.0
        - float(payload.get("syllable_shift_rate", 0.0) or 0.0) * 100.0
        - float(payload.get("low_confidence_rate", 0.0) or 0.0) * 120.0
        - float(payload.get("onset_lag_p90_ms", 0.0) or 0.0) * 0.10
        + float(cvvc.get("preutterance_acc_50ms", 0.0) or 0.0) * 280.0
        - float(cvvc.get("hard_failure_rate", 0.0) or 0.0) * 160.0
        + float(cvvc_vc_multi.get("preutterance_acc_50ms", 0.0) or 0.0) * 150.0
    )


def _build_train_cmd(
    *,
    python_exe: str,
    manifest: Path,
    val_manifest: Path,
    out_model: Path,
    resume_model: Path,
    device: str,
    epochs: int,
    case: SweepCase,
    checkpoint_save_every_epochs: int,
) -> list[str]:
    return [
        python_exe,
        "ml/scripts/coarse_crnn/train_oto.py",
        "--manifest",
        str(manifest),
        "--val-manifest",
        str(val_manifest),
        "--out",
        str(out_model),
        "--resume-model",
        str(resume_model),
        "--resume-strategy",
        "compatible",
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        "8",
        "--max-frames",
        "1200",
        "--lr",
        "2e-4",
        "--device",
        device,
        "--amp",
        "--feature-cache",
        "--vcv-loss-weight",
        "1.35",
        "--cvvc-loss-weight",
        "1.35",
        "--cvc-loss-weight",
        "1.05",
        "--vc-role-loss-weight",
        "3.2",
        "--vv-role-loss-weight",
        "2.3",
        "--language-balance-power",
        "0.35",
        "--voicebank-balance-power",
        "0.60",
        "--format-balance-power",
        "0.40",
        "--role-balance-power",
        "0.32",
        "--language-format-role-sampling-boosts",
        case.language_format_role_sampling_boosts,
        "--cvvc-vc-sampling-boost",
        str(case.cvvc_vc_sampling_boost),
        "--cvvc-vv-sampling-boost",
        str(case.cvvc_vv_sampling_boost),
        "--cvvc-vc-multi-sampling-boost",
        str(case.cvvc_vc_multi_sampling_boost),
        "--hard-case-mining",
        "--hard-case-top-ratio",
        "0.28",
        "--hard-case-boost",
        "2.6",
        "--activity-quality-filter",
        "--confidence-calibration",
        "--early-stop-patience",
        "1",
        "--checkpoint-save-every-epochs",
        str(int(max(0, checkpoint_save_every_epochs))),
    ]


def _build_eval_cmd(
    *,
    python_exe: str,
    manifest: Path,
    model: Path,
    device: str,
    out_eval: Path,
    max_items: int,
    seed: int,
) -> list[str]:
    return [
        python_exe,
        "ml/scripts/coarse_crnn/evaluate_oto.py",
        "--manifest",
        str(manifest),
        "--model",
        str(model),
        "--device",
        device,
        "--max-items",
        str(int(max_items)),
        "--seed",
        str(int(seed)),
        "--out",
        str(out_eval),
        "--no-gate",
    ]


def _compact_metrics(payload: dict[str, Any]) -> dict[str, float]:
    by_format = payload.get("by_format") or {}
    by_context = payload.get("by_alias_context") or {}
    cvvc = by_format.get("cvvc") or {}
    cvvc_vc_multi = by_context.get("cvvc|vc|multi") or {}
    return {
        "preutterance_acc_50ms": float(payload.get("preutterance_acc_50ms", 0.0) or 0.0),
        "hard_failure_rate": float(payload.get("hard_failure_rate", 0.0) or 0.0),
        "fn_voiced_miss_rate": float(payload.get("fn_voiced_miss_rate", 0.0) or 0.0),
        "syllable_shift_rate": float(payload.get("syllable_shift_rate", 0.0) or 0.0),
        "low_confidence_rate": float(payload.get("low_confidence_rate", 0.0) or 0.0),
        "onset_lag_p90_ms": float(payload.get("onset_lag_p90_ms", 0.0) or 0.0),
        "cvvc_preutterance_acc_50ms": float(cvvc.get("preutterance_acc_50ms", 0.0) or 0.0),
        "cvvc_hard_failure_rate": float(cvvc.get("hard_failure_rate", 0.0) or 0.0),
        "cvvc_vc_multi_preutterance_acc_50ms": float(cvvc_vc_multi.get("preutterance_acc_50ms", 0.0) or 0.0),
    }


def _is_ranking_valid(ranked: list[dict[str, Any]]) -> tuple[bool, str]:
    if not ranked:
        return False, "no_results"
    low_conf = [float((row.get("metrics") or {}).get("low_confidence_rate", 0.0) or 0.0) for row in ranked]
    if low_conf and min(low_conf) >= 0.95:
        return False, "all_cases_low_confidence_saturated"
    scores = [float(row.get("score", 0.0) or 0.0) for row in ranked]
    if max(scores) - min(scores) < 1e-6:
        return False, "all_scores_identical"
    return True, ""


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OTO CVVC Sampler Sweep",
        "",
        f"- 생성 시각(KST): {summary.get('generated_at_kst', '')}",
        f"- run_name: `{summary.get('run_name', '')}`",
        f"- resume_model: `{summary.get('resume_model', '')}`",
        f"- train_manifest: `{summary.get('train_manifest', '')}`",
        f"- val_manifest: `{summary.get('val_manifest', '')}`",
        f"- eval_manifest: `{summary.get('eval_manifest', '')}`",
        f"- epochs(case): {summary.get('epochs_per_case', 0)}",
        f"- ranking_valid: `{summary.get('ranking_valid', True)}`",
        f"- ranking_note: `{summary.get('ranking_note', '')}`",
        "",
        "## Ranked",
        "",
        "| rank | case | score | pre50 | hard | fn_miss | shift | low_conf | onset_p90 | cvvc_pre50 | cvvc_hard | cvvc_vc_multi_pre50 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary.get("ranked_results", []), start=1):
        m = row.get("metrics") or {}
        lines.append(
            "| {rank} | `{name}` | {score:.3f} | {pre50:.4f} | {hard:.4f} | {fn:.4f} | {shift:.4f} | {low_conf:.4f} | {onset:.2f} | {cvvc_pre:.4f} | {cvvc_hard:.4f} | {ctx:.4f} |".format(
                rank=idx,
                name=row.get("name", ""),
                score=float(row.get("score", 0.0) or 0.0),
                pre50=float(m.get("preutterance_acc_50ms", 0.0) or 0.0),
                hard=float(m.get("hard_failure_rate", 0.0) or 0.0),
                fn=float(m.get("fn_voiced_miss_rate", 0.0) or 0.0),
                shift=float(m.get("syllable_shift_rate", 0.0) or 0.0),
                low_conf=float(m.get("low_confidence_rate", 0.0) or 0.0),
                onset=float(m.get("onset_lag_p90_ms", 0.0) or 0.0),
                cvvc_pre=float(m.get("cvvc_preutterance_acc_50ms", 0.0) or 0.0),
                cvvc_hard=float(m.get("cvvc_hard_failure_rate", 0.0) or 0.0),
                ctx=float(m.get("cvvc_vc_multi_preutterance_acc_50ms", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Next",
            "",
            "1. ranking_valid=false면 sampler 추가 탐색 대신 fallback/guard 계측부터 수행",
            "2. `s` 계열 치찰 자음 alias를 role/context로 분리해 fallback trigger를 재평가",
            "3. ranking_valid=true일 때만 1위/2위 모델로 미학습 화자(일본어/한국어) 청음 테스트",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a focused CVVC sampler sweep and rank eval outputs.")
    parser.add_argument("--run-name", default="", help="Output folder suffix. default: timestamp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits_prepared", "oto_train.jsonl"))
    parser.add_argument("--val-manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits_prepared", "oto_val.jsonl"))
    parser.add_argument("--eval-manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits_prepared", "oto_test.jsonl"))
    parser.add_argument("--resume-model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn_role_v3_patch_e10_gatev1.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-eval-items", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint-save-every-epochs", type=int, default=0)
    args = parser.parse_args()

    repo = _repo_root()
    run_name = str(args.run_name or f"cvvc_sampler_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir = repo / "ml_workspace" / "coarse_crnn" / "sweeps" / run_name
    models_dir = out_dir / "models"
    evals_dir = out_dir / "evals"
    models_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    eval_manifest = Path(args.eval_manifest)
    resume_model = Path(args.resume_model)
    if not train_manifest.is_absolute():
        train_manifest = repo / train_manifest
    if not val_manifest.is_absolute():
        val_manifest = repo / val_manifest
    if not eval_manifest.is_absolute():
        eval_manifest = repo / eval_manifest
    if not resume_model.is_absolute():
        resume_model = repo / resume_model

    results: list[dict[str, Any]] = []
    for case in _default_cases():
        model_path = models_dir / f"{case.name}.pt"
        eval_path = evals_dir / f"oto_eval_{case.name}.json"
        print(f"[sweep] train: {case.name}")
        train_cmd = _build_train_cmd(
            python_exe=str(args.python),
            manifest=train_manifest,
            val_manifest=val_manifest,
            out_model=model_path,
            resume_model=resume_model,
            device=str(args.device),
            epochs=int(args.epochs),
            case=case,
            checkpoint_save_every_epochs=int(args.checkpoint_save_every_epochs),
        )
        _run(train_cmd, cwd=repo)

        print(f"[sweep] eval : {case.name}")
        eval_cmd = _build_eval_cmd(
            python_exe=str(args.python),
            manifest=eval_manifest,
            model=model_path,
            device=str(args.device),
            out_eval=eval_path,
            max_items=int(args.max_eval_items),
            seed=int(args.seed),
        )
        _run(eval_cmd, cwd=repo)
        eval_payload = _read_json(eval_path)
        metrics = _compact_metrics(eval_payload)
        score = _score_eval(eval_payload)
        results.append(
            {
                "name": case.name,
                "model_path": str(model_path.resolve()),
                "eval_path": str(eval_path.resolve()),
                "sampling": {
                    "cvvc_vc_sampling_boost": case.cvvc_vc_sampling_boost,
                    "cvvc_vv_sampling_boost": case.cvvc_vv_sampling_boost,
                    "cvvc_vc_multi_sampling_boost": case.cvvc_vc_multi_sampling_boost,
                    "language_format_role_sampling_boosts": case.language_format_role_sampling_boosts,
                },
                "score": float(score),
                "metrics": metrics,
            }
        )

    ranked = sorted(results, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    ranking_valid, ranking_note = _is_ranking_valid(ranked)
    summary = {
        "generated_at_kst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "repo_root": str(repo.resolve()),
        "resume_model": str(resume_model.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "eval_manifest": str(eval_manifest.resolve()),
        "epochs_per_case": int(args.epochs),
        "ranking_valid": bool(ranking_valid),
        "ranking_note": str(ranking_note),
        "ranked_results": ranked,
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"out_dir": str(out_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
