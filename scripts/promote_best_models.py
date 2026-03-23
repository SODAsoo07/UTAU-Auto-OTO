from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_DIR = str(Path(__file__).resolve().parents[1])
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


@dataclass
class Candidate:
    model_dir: str
    backend: str
    language: str
    format_type: str
    alias_family: str
    score: float
    baseline: Optional[float]
    train_rows: int
    voicebank_count: int
    metric_source: str


def _load_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f) or {})
    except Exception:
        return {}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_alias_family(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw or raw in {"all", "none", "general"}:
        return ""
    return raw


def _sanitize_backend_token(backend: str) -> str:
    token = str(backend or "").strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = token.strip("_")
    return token or "unknown"


def _extract_metric_from_target_dict(target_dict: Dict[str, object], mae_key: str) -> Tuple[Optional[float], Optional[float], int]:
    maes: List[float] = []
    baselines: List[float] = []
    count = 0
    for _target, payload in (target_dict or {}).items():
        if not isinstance(payload, dict):
            continue
        mae_val = payload.get(mae_key)
        if mae_val is None:
            continue
        maes.append(_to_float(mae_val))
        if payload.get("baseline_mae") is not None:
            baselines.append(_to_float(payload.get("baseline_mae")))
        count += 1
    if count <= 0:
        return None, None, 0
    mean_mae = float(sum(maes) / max(1, len(maes)))
    mean_base = float(sum(baselines) / max(1, len(baselines))) if baselines else None
    return mean_mae, mean_base, count


def _extract_score(model_dir: str, meta: Dict[str, object]) -> Tuple[Optional[float], Optional[float], str]:
    backend = str(meta.get("backend", "") or "").strip().lower()
    holdout = meta.get("holdout_metrics")
    eval_summary = _load_json(os.path.join(model_dir, "eval_summary.json"))

    # Backend-specific primary metric source priority.
    if backend == "ensemble_v1":
        # Preferred: meta model MAE (actual final ensemble output).
        if isinstance(holdout, dict):
            meta_metrics = holdout.get("meta")
            if isinstance(meta_metrics, dict):
                mae, base, n = _extract_metric_from_target_dict(meta_metrics, "meta_mae")
                if mae is not None and n > 0:
                    return mae, base, "holdout_metrics.meta.meta_mae"
                mae, base, n = _extract_metric_from_target_dict(meta_metrics, "model_mae")
                if mae is not None and n > 0:
                    return mae, base, "holdout_metrics.meta.model_mae"
            lgb_metrics = holdout.get("lightgbm")
            if isinstance(lgb_metrics, dict):
                mae, base, n = _extract_metric_from_target_dict(lgb_metrics, "model_mae")
                if mae is not None and n > 0:
                    return mae, base, "holdout_metrics.lightgbm.model_mae(fallback)"
        if isinstance(eval_summary, dict):
            meta_block = eval_summary.get("meta")
            if isinstance(meta_block, dict):
                metric_block = meta_block.get("metrics")
                if isinstance(metric_block, dict):
                    mae, base, n = _extract_metric_from_target_dict(metric_block, "meta_mae")
                    if mae is not None and n > 0:
                        return mae, base, "eval_summary.meta.metrics.meta_mae"
                    mae, base, n = _extract_metric_from_target_dict(metric_block, "model_mae")
                    if mae is not None and n > 0:
                        return mae, base, "eval_summary.meta.metrics.model_mae"
            lgb_block = eval_summary.get("lightgbm")
            if isinstance(lgb_block, dict):
                metric_block = lgb_block.get("metrics")
                if isinstance(metric_block, dict):
                    mae, base, n = _extract_metric_from_target_dict(metric_block, "model_mae")
                    if mae is not None and n > 0:
                        return mae, base, "eval_summary.lightgbm.metrics.model_mae(fallback)"
        return None, None, ""

    # LightGBM / coupled backend metrics.
    if isinstance(holdout, dict):
        mae, base, n = _extract_metric_from_target_dict(holdout, "model_mae")
        if mae is not None and n > 0:
            return mae, base, "holdout_metrics.model_mae"
    if isinstance(eval_summary, dict):
        metric_block = eval_summary.get("metrics")
        if isinstance(metric_block, dict):
            mae, base, n = _extract_metric_from_target_dict(metric_block, "model_mae")
            if mae is not None and n > 0:
                return mae, base, "eval_summary.metrics.model_mae"
    return None, None, ""


def _collect_candidates(
    scan_roots: List[str],
    include_backends: List[str],
    include_languages: List[str],
    include_formats: List[str],
) -> List[Candidate]:
    out: List[Candidate] = []
    backend_set = {str(v).strip().lower() for v in include_backends if str(v).strip()}
    language_set = {str(v).strip().lower() for v in include_languages if str(v).strip()}
    format_set = {str(v).strip().lower() for v in include_formats if str(v).strip()}
    for root in scan_roots:
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            continue
        for current_root, _dirs, files in os.walk(abs_root):
            if "model_meta.json" not in files:
                continue
            model_dir = os.path.abspath(current_root)
            meta = _load_json(os.path.join(model_dir, "model_meta.json"))
            backend = str(meta.get("backend", "") or "").strip().lower()
            if backend_set and backend not in backend_set:
                continue
            language = str(meta.get("language", "") or "").strip().lower()
            format_type = str(meta.get("format_type", "") or "").strip().lower()
            if language_set and language not in language_set:
                continue
            if format_set and format_type not in format_set:
                continue
            alias_family = _normalize_alias_family(
                meta.get("alias_family", "") or (meta.get("filters") or {}).get("alias_family", "")
            )
            if not language or not format_type:
                continue
            score, baseline, source = _extract_score(model_dir, meta)
            if score is None:
                continue
            out.append(
                Candidate(
                    model_dir=model_dir,
                    backend=backend,
                    language=language,
                    format_type=format_type,
                    alias_family=alias_family,
                    score=float(score),
                    baseline=(float(baseline) if baseline is not None else None),
                    train_rows=int(_to_float(meta.get("train_rows", 0), 0.0)),
                    voicebank_count=int(_to_float(meta.get("voicebank_count", 0), 0.0)),
                    metric_source=source,
                )
            )
    return out


def _group_key(candidate: Candidate, keep_alias_family: bool) -> Tuple[str, str, str, str]:
    return (
        candidate.language,
        candidate.format_type,
        candidate.backend,
        candidate.alias_family if keep_alias_family else "",
    )


def _select_best(candidates: List[Candidate], keep_alias_family: bool) -> Dict[Tuple[str, str, str, str], Candidate]:
    best: Dict[Tuple[str, str, str, str], Candidate] = {}
    for item in candidates:
        key = _group_key(item, keep_alias_family=keep_alias_family)
        prev = best.get(key)
        if prev is None:
            best[key] = item
            continue
        # Primary: lower MAE, tie-break: larger train_rows, then larger voicebank_count.
        if item.score < prev.score:
            best[key] = item
            continue
        if item.score == prev.score:
            if item.train_rows > prev.train_rows:
                best[key] = item
                continue
            if item.train_rows == prev.train_rows and item.voicebank_count > prev.voicebank_count:
                best[key] = item
                continue
    return best


def _install_target_path(install_root: str, item: Candidate, keep_alias_family: bool) -> str:
    parts = [install_root, item.language, item.format_type]
    alias_family = item.alias_family if keep_alias_family else ""
    if alias_family:
        parts.extend(["families", alias_family])
    parts.append(f"vbest_{_sanitize_backend_token(item.backend)}")
    return os.path.join(*parts)


def _copy_model_tree(src: str, dst: str, dry_run: bool) -> None:
    if dry_run:
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)


def _should_export_onnx(backend: str) -> bool:
    return str(backend).strip().lower() in {"ensemble_v1", "coupled_nn_v1", "coupled_nn_v2_rawmel"}


def _run_onnx_export_for_target(target_dir: str, dry_run: bool) -> Dict[str, object]:
    cmd = [sys.executable, os.path.join(APP_DIR, "scripts", "export_coupled_onnx.py"), "--model-dir", target_dir]
    if dry_run:
        return {"command": cmd, "returncode": 0, "dry_run": True}
    proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "dry_run": False,
    }


def _manifest_path(default_path: str) -> str:
    path = os.path.abspath(default_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select best-performing OTO ML models per language/format/backend and install them."
    )
    parser.add_argument(
        "--scan-roots",
        default="ML_models,ml_workspace/models",
        help="Comma-separated roots to scan for model_meta.json",
    )
    parser.add_argument(
        "--include-backends",
        default="lightgbm,ensemble_v1,coupled_nn_v2_rawmel,coupled_nn_v1",
        help="Comma-separated backend names to consider.",
    )
    parser.add_argument(
        "--include-languages",
        default="korean,japanese",
        help="Comma-separated languages to consider.",
    )
    parser.add_argument(
        "--include-formats",
        default="cv,cvc,cvvc,vcv,general",
        help="Comma-separated format types to consider.",
    )
    parser.add_argument(
        "--install-root",
        default="models_installed/oto_ml_best",
        help="Install root for selected best models.",
    )
    parser.add_argument(
        "--collapse-alias-family",
        action="store_true",
        help="Do not split selection by alias_family (select one per language/format/backend only).",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="After install, run coupled ONNX export for selected ensemble/coupled targets.",
    )
    parser.add_argument(
        "--manifest-out",
        default="ml_workspace/exports/best_model_selection_manifest.json",
        help="Output JSON summary path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show selections without copying or conversion.")
    args = parser.parse_args()

    scan_roots = [part.strip() for part in str(args.scan_roots or "").split(",") if part.strip()]
    include_backends = [part.strip() for part in str(args.include_backends or "").split(",") if part.strip()]
    include_languages = [part.strip() for part in str(args.include_languages or "").split(",") if part.strip()]
    include_formats = [part.strip() for part in str(args.include_formats or "").split(",") if part.strip()]
    keep_alias_family = not bool(args.collapse_alias_family)
    install_root = os.path.abspath(str(args.install_root or "models_installed/oto_ml"))
    manifest_out = _manifest_path(str(args.manifest_out or "ml_workspace/exports/best_model_selection_manifest.json"))

    candidates = _collect_candidates(scan_roots, include_backends, include_languages, include_formats)
    if not candidates:
        print("No scored model candidates found.")
        return 1

    selected = _select_best(candidates, keep_alias_family=keep_alias_family)
    rows = []
    onnx_logs = []
    for key in sorted(selected.keys()):
        item = selected[key]
        target_dir = _install_target_path(install_root, item, keep_alias_family=keep_alias_family)
        _copy_model_tree(item.model_dir, target_dir, dry_run=bool(args.dry_run))
        onnx_result = None
        if bool(args.export_onnx) and _should_export_onnx(item.backend):
            onnx_result = _run_onnx_export_for_target(target_dir, dry_run=bool(args.dry_run))
            onnx_logs.append(
                {
                    "backend": item.backend,
                    "source": item.model_dir,
                    "target": target_dir,
                    "onnx_returncode": int(onnx_result.get("returncode", 1)),
                }
            )
        improvement = None
        if item.baseline is not None and item.baseline > 0.0:
            improvement = float((item.baseline - item.score) / item.baseline)
        rows.append(
            {
                "language": item.language,
                "format_type": item.format_type,
                "backend": item.backend,
                "alias_family": item.alias_family,
                "score_mae": item.score,
                "baseline_mae": item.baseline,
                "relative_improvement": improvement,
                "metric_source": item.metric_source,
                "train_rows": item.train_rows,
                "voicebank_count": item.voicebank_count,
                "source_model_dir": item.model_dir,
                "installed_model_dir": target_dir,
                "onnx": onnx_result,
            }
        )
        print(
            f"[BEST] {item.language}/{item.format_type}"
            f"{('/' + item.alias_family) if item.alias_family else ''}"
            f" backend={item.backend} score={item.score:.4f} -> {target_dir}"
        )

    summary = {
        "scan_roots": [os.path.abspath(v) for v in scan_roots],
        "include_backends": include_backends,
        "include_languages": include_languages,
        "include_formats": include_formats,
        "install_root": install_root,
        "keep_alias_family": bool(keep_alias_family),
        "dry_run": bool(args.dry_run),
        "export_onnx": bool(args.export_onnx),
        "candidate_count": len(candidates),
        "selected_count": len(rows),
        "selections": rows,
        "onnx_runs": onnx_logs,
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Manifest written: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
