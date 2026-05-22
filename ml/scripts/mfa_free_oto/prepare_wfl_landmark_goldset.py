from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.acoustic_nucleus import (
    AcousticNucleusConfig,
    relabel_vowel_nuclei_manifest,
    write_summary_json as write_nucleus_summary_json,
)
from core.mfa_free_oto.evaluation import evaluate_manifest, save_json
from core.mfa_free_oto.htk_lab import row_from_htk_lab, write_summary_json, HtkGoldBuildSummary
from core.mfa_free_oto.manifest import load_manifest_jsonl, write_manifest_jsonl
from core.mfa_free_oto.manifest_audit import audit_manifest_path, top_flagged_rows
from core.mfa_free_oto.review_overlay import posterior_from_json, write_review_html
from core.mfa_free_oto.training import TrainPocConfig, train_poc
from ml.scripts.mfa_free_oto.split_gold_manifest import split_rows


DEFAULT_LAB_ROOT = (
    r"C:\Users\oyh57\SODAsoo1\VocalSynth\Tools\WFL-UTAU"
    r"\runs\alignmented_japanese_20260522_024221"
)
DEFAULT_WAV_ROOT = (
    r"C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
    r"\dataset_staged\japanese\alignmented"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare WFL-constrained HTK labs as MFA-free OTO acoustic landmark training data."
    )
    parser.add_argument("--lab-root", default=DEFAULT_LAB_ROOT)
    parser.add_argument("--wav-root", default=DEFAULT_WAV_ROOT)
    parser.add_argument(
        "--out-dir",
        default=r"ml_workspace\mfa_free_oto\wfl_alignmented_landmark",
        help="Output directory for manifests, audits, checkpoint, metrics, and overlays.",
    )
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--dataset-id", default="wfl_alignmented_japanese")
    parser.add_argument("--encoder", default="acoustic_world_v1")
    parser.add_argument("--device", default="")
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--event-loss-weight", type=float, default=1.3)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "audit"
    overlay_dir = out_dir / "overlays"
    audit_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    lab_manifest = out_dir / "wfl_lab_midpoint_manifest.jsonl"
    lab_summary_path = out_dir / "wfl_lab_midpoint_summary.json"
    rows, lab_summary = build_manifest_from_parallel_roots(
        lab_root=Path(args.lab_root),
        wav_root=Path(args.wav_root),
        out_path=lab_manifest,
        language=args.language,
        dataset_id=args.dataset_id,
    )
    write_summary_json(lab_summary_path, lab_summary)
    if not rows:
        raise SystemExit(f"No paired WFL .lab/.wav rows found: lab={args.lab_root} wav={args.wav_root}")

    lab_audit_path = audit_dir / "wfl_lab_midpoint.audit.json"
    lab_flagged_path = audit_dir / "wfl_lab_midpoint.flagged.json"
    lab_audit = audit_manifest_path(lab_manifest)
    lab_audit_path.write_text(json.dumps(lab_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lab_flagged_path.write_text(
        json.dumps(
            {
                "manifest_path": str(lab_manifest),
                "rows": lab_audit["rows"],
                "flag_counts": lab_audit["flag_counts"],
                "flagged_rows": top_flagged_rows(lab_audit, limit=120),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    nucleus_manifest = out_dir / "wfl_landmark_manifest.jsonl"
    nucleus_summary_path = out_dir / "wfl_acoustic_nucleus_summary.json"
    nucleus_config = AcousticNucleusConfig()
    _nucleus_rows, nucleus_summary = relabel_vowel_nuclei_manifest(
        lab_manifest,
        nucleus_manifest,
        encoder=args.encoder,
        device=(args.device or None),
        config=nucleus_config,
    )
    nucleus_payload = {
        **asdict(nucleus_summary),
        "input_manifest": str(lab_manifest),
        "out_manifest": str(nucleus_manifest),
        "encoder": args.encoder,
        "config": asdict(nucleus_config),
    }
    write_nucleus_summary_json(nucleus_summary_path, nucleus_payload)

    nucleus_audit_path = audit_dir / "wfl_landmark.audit.json"
    nucleus_audit = audit_manifest_path(nucleus_manifest)
    nucleus_audit_path.write_text(json.dumps(nucleus_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rows = load_manifest_jsonl(nucleus_manifest, require_manual=True, require_labels=True)
    train_rows, eval_rows = split_rows(all_rows, eval_ratio=args.eval_ratio, seed=args.seed)
    train_manifest = out_dir / "wfl_landmark_train.jsonl"
    eval_manifest = out_dir / "wfl_landmark_eval.jsonl"
    write_manifest_jsonl(train_manifest, train_rows)
    write_manifest_jsonl(eval_manifest, eval_rows)

    checkpoint = out_dir / "wfl_landmark_tiny.pt"
    train_result = None
    eval_result = None
    if not args.skip_train:
        train_result = train_poc(
            TrainPocConfig(
                manifest_path=str(train_manifest),
                out_path=str(checkpoint),
                encoder=args.encoder,
                epochs=args.epochs,
                learning_rate=args.lr,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                max_rows=args.max_rows,
                seed=args.seed,
                device=(args.device or None),
                event_loss_weight=args.event_loss_weight,
                specaugment=True,
                frame_class_weighting="balanced",
                frame_focus_weighting="boundary_focus",
            )
        )
        if not args.skip_eval:
            eval_result = evaluate_manifest(
                eval_manifest,
                checkpoint,
                encoder=args.encoder,
                device=(args.device or None),
                boundary_tolerance_ms=40.0,
                nucleus_tolerance_ms=50.0,
                use_slot_viterbi=True,
            )
            save_json(out_dir / "wfl_landmark_eval_metrics.json", eval_result)
            _write_eval_overlays(eval_rows, eval_result, overlay_dir, max_overlays=args.max_overlays)

    summary = {
        "lab_root": str(Path(args.lab_root).resolve()),
        "wav_root": str(Path(args.wav_root).resolve()),
        "out_dir": str(out_dir),
        "lab_manifest": str(lab_manifest),
        "nucleus_manifest": str(nucleus_manifest),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "checkpoint": str(checkpoint) if checkpoint.exists() else "",
        "lab_summary": asdict(lab_summary),
        "nucleus_summary": nucleus_payload,
        "split": {
            "seed": args.seed,
            "eval_ratio": args.eval_ratio,
            "rows": len(all_rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
        "train_result": train_result,
        "eval_metrics": _compact_eval(eval_result) if eval_result else None,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_manifest_from_parallel_roots(
    *,
    lab_root: Path,
    wav_root: Path,
    out_path: Path,
    language: str,
    dataset_id: str,
) -> tuple[list[dict], HtkGoldBuildSummary]:
    lab_root = lab_root.resolve()
    wav_root = wav_root.resolve()
    rows: list[dict] = []
    skipped_without_wav = 0
    dropped_segments = 0
    for lab_path in sorted(lab_root.rglob("*.lab")):
        rel = lab_path.relative_to(lab_root)
        wav_path = (wav_root / rel).with_suffix(".wav")
        if not wav_path.is_file():
            skipped_without_wav += 1
            continue
        format_type = rel.parts[0] if rel.parts else "unknown"
        row = row_from_htk_lab(
            lab_path,
            wav_path=wav_path,
            language=language,
            format_type=format_type,
            dataset_id=dataset_id,
            relative_to=None,
        )
        row_id = str(rel.with_suffix("")).replace("\\", "/")
        row["row_id"] = row_id
        row["row_group"] = "/".join(rel.parts[: min(3, len(rel.parts))])
        row["label_origin"] = "wfl_constrained_pseudo_lab"
        row["label_policy"]["wfl_constrained"] = True
        row["label_policy"]["filename_is_label_constraint"] = True
        row["label_policy"]["manual_gold_compatibility_note"] = (
            "This row is stored as manual_gold only to reuse the current training validator; "
            "the actual source is WFL constrained pseudo-lab."
        )
        diag_path = lab_path.with_suffix(".wfl_utau.json")
        if diag_path.is_file():
            row["wfl_diagnostics_path"] = str(diag_path)
        dropped_segments += len(row.get("dropped_lab_segments") or [])
        rows.append(row)
    write_manifest_jsonl(out_path, rows)
    summary = HtkGoldBuildSummary(
        rows=len(rows),
        labelled_rows=len(rows),
        skipped_without_wav=skipped_without_wav,
        dropped_segments=dropped_segments,
        source_roots=[str(lab_root), str(wav_root)],
        out_path=str(out_path),
    )
    return rows, summary


def _write_eval_overlays(rows: list[dict], metrics: dict, overlay_dir: Path, *, max_overlays: int) -> None:
    by_row_id = {str(row.get("row_id") or row.get("wav_name")): row for row in rows}
    row_results = metrics.get("row_results") or []
    predictions = {str(item.get("row_id")): item for item in metrics.get("predictions") or []}
    ranked = sorted(row_results, key=_row_severity, reverse=True)
    for idx, row_result in enumerate(ranked[: max(0, int(max_overlays))], start=1):
        row_id = str(row_result.get("row_id") or "")
        row = by_row_id.get(row_id)
        pred = predictions.get(row_id)
        if row is None or pred is None:
            continue
        posterior = posterior_from_json(pred, str(row.get("wav_path") or ""))
        write_review_html(
            overlay_dir / f"{idx:03d}_{_safe_name(row_id)}.html",
            row,
            posterior=posterior,
            decoded_events=pred.get("decoded_events") or [],
            width=1360,
        )


def _row_severity(row_result: dict) -> float:
    score = 0.0
    if row_result.get("catastrophic"):
        score += 10000.0
    cv = row_result.get("cv_boundary") or {}
    nucleus = row_result.get("vowel_nucleus") or {}
    score += float(cv.get("p90_error_ms") or cv.get("median_error_ms") or 0.0)
    score += 0.5 * float(nucleus.get("p90_error_ms") or nucleus.get("median_error_ms") or 0.0)
    score += 100.0 * float(cv.get("missed") or 0)
    score += 50.0 * float(nucleus.get("missed") or 0)
    return score


def _compact_eval(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    keys = [
        "rows",
        "encoder",
        "frame_macro_f1",
        "cv_boundary_median_error_ms",
        "cv_boundary_p90_error_ms",
        "vowel_nucleus_median_error_ms",
        "vowel_nucleus_p90_error_ms",
        "hard_failure_rate",
        "catastrophic_rate",
        "gate_report",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _safe_name(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out)[:120] or "row"


if __name__ == "__main__":
    raise SystemExit(main())
