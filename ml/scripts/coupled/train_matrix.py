from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import normalize_format_type, normalize_language_name
from core.oto_ml.features.mel_patches import MEL_PATCH_CACHE_VERSION, default_patch_cache_root
from core.oto_ml_collection import build_datasets_from_candidates
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml_coupled import evaluate_coupled_bundle, train_coupled_bundle, train_coupled_bundle_rawmel
from core.runtime_encoding import bootstrap_utf8_runtime


def _split_csv_tokens(raw: str) -> List[str]:
    return [v.strip() for v in str(raw or "").split(",") if v.strip()]


def _parse_languages(raw: str) -> List[str]:
    tokens = _split_csv_tokens(raw)
    if not tokens:
        tokens = ["korean", "japanese"]
    out: List[str] = []
    for token in tokens:
        t = str(token).strip().lower()
        if t in {"all", "both", "*"}:
            out.extend(["korean", "japanese"])
            continue
        lang = normalize_language_name(t)
        if lang in {"korean", "japanese"}:
            out.append(lang)
    # preserve order, unique
    seen = set()
    dedup = []
    for lang in out:
        if lang in seen:
            continue
        seen.add(lang)
        dedup.append(lang)
    return dedup


def _parse_format_filter(raw: str) -> Dict[str, Optional[set]]:
    """
    grammar:
      - 'auto' / '' / '*' : no explicit filter per language
      - 'cv,cvc'          : shared filter for both languages
      - 'korean=cv,cvc;japanese=cvvc,vcv'
    """
    text = str(raw or "").strip()
    if not text or text.lower() in {"auto", "*", "all"}:
        return {}

    out: Dict[str, Optional[set]] = {}
    if ";" in text or "=" in text:
        chunks = [c.strip() for c in text.split(";") if c.strip()]
        for chunk in chunks:
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            lang = normalize_language_name(k.strip().lower())
            if lang not in {"korean", "japanese"}:
                continue
            fmt_set = set()
            for f in _split_csv_tokens(v):
                nf = normalize_format_type(lang, f)
                if nf:
                    fmt_set.add(nf)
            out[lang] = fmt_set
        return out

    shared = set()
    for f in _split_csv_tokens(text):
        for lang in ("korean", "japanese"):
            nf = normalize_format_type(lang, f)
            if nf:
                shared.add(nf)
    if shared:
        out["korean"] = set(shared)
        out["japanese"] = set(shared)
    return out


def _auto_rawmel_cache_dir(language: str, format_type: str) -> str:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    root = default_patch_cache_root()
    base = os.path.join(root, lang, fmt, str(MEL_PATCH_CACHE_VERSION))
    if not os.path.isdir(base):
        return ""
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        manifest = os.path.join(path, "manifest.json")
        if not os.path.isfile(manifest):
            continue
        try:
            mtime = os.path.getmtime(manifest)
        except Exception:
            mtime = 0.0
        candidates.append((mtime, path))
    if not candidates:
        return ""
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _default_min_mapping_conf(language: str, fmt: str) -> float:
    lang = normalize_language_name(language)
    f = normalize_format_type(lang, fmt) or "general"
    if lang == "korean":
        table = {
            "cv": 0.55,
            "cvc": 0.75,
            "cvvc": 0.78,
            "vcv": 0.72,
            "general": 0.50,
        }
        return float(table.get(f, 0.50))
    table = {
        "cv": 0.55,
        "cvc": 0.60,
        "cvvc": 0.70,
        "vcv": 0.68,
        "general": 0.50,
    }
    return float(table.get(f, 0.50))


def _collect_jobs_from_built(built_result: Dict[str, object]) -> List[Tuple[str, str, str, int]]:
    jobs: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for item in list(built_result.get("candidates") or []):
        try:
            status = str(getattr(item, "status", "") or "")
            language = str(getattr(item, "language", "") or "").strip().lower()
            fmt = str(getattr(item, "format_type", "") or "").strip().lower()
            out_csv = str(getattr(item, "out_csv", "") or "")
            saved_rows = int(getattr(item, "saved_rows", 0) or 0)
        except Exception:
            continue
        if status != "built" or not out_csv:
            continue
        key = (language, fmt)
        prev = jobs.get(key)
        if prev is None:
            jobs[key] = (out_csv, max(saved_rows, 0))
        else:
            jobs[key] = (out_csv, int(prev[1]) + max(saved_rows, 0))
    out: List[Tuple[str, str, str, int]] = []
    for (language, fmt), (out_csv, rows) in sorted(jobs.items()):
        if out_csv:
            out.append((language, fmt, out_csv, int(rows)))
    return out


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Batch coupled training by language/format matrix.")
    ap.add_argument("--dataset-root", required=True, help="Staged dataset root (contains language/format folders)")
    ap.add_argument("--workspace-root", default=os.path.join(ROOT, "ml_workspace"))
    ap.add_argument("--model-root", default=os.path.join(ROOT, "ML_models"))
    ap.add_argument("--languages", default="korean,japanese", help="korean,japanese|both|all")
    ap.add_argument(
        "--formats",
        default="auto",
        help="auto | cv,cvc,... | korean=cv,cvc;japanese=cvvc,vcv",
    )
    ap.add_argument(
        "--auto-oto-policy",
        default="require",
        choices=["require", "generate-temp", "generate-persist"],
        help="Policy when auto oto is missing during dataset build",
    )
    ap.add_argument("--backend", default="coupled_nn_v1", help="coupled_nn_v1 or coupled_nn_v2_rawmel")
    ap.add_argument("--rawmel-cache", default="", help="Raw mel cache dir (v2 only). Empty=auto per job")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-confidence", type=float, default=0.50)
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--alias-types", default="")
    ap.add_argument("--alias-family", default="")
    ap.add_argument(
        "--min-mapping-confidence",
        type=float,
        default=-1.0,
        help="If <0, use per-language/format defaults",
    )
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset build; train from existing CSVs only")
    ap.add_argument("--skip-train", action="store_true", help="Build datasets only")
    ap.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    ap.add_argument("--require-all", action="store_true", help="Fail if any selected language/format job fails or missing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    workspace_root = os.path.abspath(args.workspace_root)
    model_root = os.path.abspath(args.model_root)
    os.makedirs(workspace_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    languages = _parse_languages(args.languages)
    if not languages:
        raise SystemExit("No valid languages selected.")
    format_filter = _parse_format_filter(args.formats)

    candidates = discover_training_candidates_from_dataset_root(
        dataset_root,
        auto_oto_policy=args.auto_oto_policy,
    )
    selected = []
    for c in candidates:
        lang = normalize_language_name(getattr(c, "language", ""))
        fmt = normalize_format_type(lang, getattr(c, "format_type", "")) or str(getattr(c, "format_type", "")).strip().lower()
        if lang not in languages:
            continue
        wanted = format_filter.get(lang)
        if isinstance(wanted, set) and wanted and fmt not in wanted:
            continue
        selected.append(c)

    if not selected:
        raise SystemExit("No candidates matched the selected language/format filters.")

    summary: Dict[str, object] = {
        "dataset_root": dataset_root,
        "workspace_root": workspace_root,
        "model_root": model_root,
        "languages": languages,
        "format_filter": {k: sorted(v) for k, v in format_filter.items()},
        "selected_candidates": len(selected),
        "jobs": [],
        "failures": [],
    }

    jobs: List[Tuple[str, str, str, int]] = []
    if args.skip_build:
        # fallback: infer expected CSV path by selected language/format combos
        combos = sorted({(normalize_language_name(getattr(c, "language", "")), normalize_format_type(getattr(c, "language", ""), getattr(c, "format_type", "")) or str(getattr(c, "format_type", "")).strip().lower()) for c in selected})
        for lang, fmt in combos:
            csv_path = os.path.join(workspace_root, "datasets", lang, f"dataset_{lang}_{fmt}.csv")
            if os.path.isfile(csv_path):
                jobs.append((lang, fmt, csv_path, -1))
            elif args.require_all:
                raise SystemExit(f"Missing dataset CSV for {lang}/{fmt}: {csv_path}")
    else:
        built = build_datasets_from_candidates(
            selected,
            workspace_root,
            auto_oto_policy=args.auto_oto_policy,
            progress_callback=lambda message: print(message, flush=True),
        )
        jobs = _collect_jobs_from_built(built)
        summary["build_summary"] = built.get("summary", {})
        if not jobs:
            raise SystemExit("No datasets were built from selected candidates.")

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    backend = str(args.backend or "coupled_nn_v1").strip().lower()

    for lang, fmt, dataset_csv, built_rows in jobs:
        job = {
            "language": lang,
            "format_type": fmt,
            "dataset_csv": dataset_csv,
            "built_rows": int(built_rows),
            "status": "pending",
        }
        summary["jobs"].append(job)

        out_dir = os.path.join(model_root, lang, fmt, ("v2_coupled_rawmel" if backend == "coupled_nn_v2_rawmel" else "v1_coupled"))
        eval_report = os.path.join(out_dir, "eval_summary.json")
        job["model_dir"] = out_dir
        job["eval_report"] = eval_report

        mmc = float(args.min_mapping_confidence)
        if mmc < 0.0:
            mmc = _default_min_mapping_conf(lang, fmt)
        job["min_mapping_confidence"] = float(mmc)

        if args.dry_run:
            job["status"] = "dry_run"
            continue

        try:
            os.makedirs(out_dir, exist_ok=True)
            if not args.skip_train:
                if backend == "coupled_nn_v2_rawmel":
                    rawmel_cache = str(args.rawmel_cache or "").strip()
                    if rawmel_cache and not os.path.isdir(rawmel_cache):
                        rawmel_cache = ""
                    if not rawmel_cache:
                        rawmel_cache = _auto_rawmel_cache_dir(lang, fmt)
                    if not rawmel_cache:
                        raise RuntimeError(f"rawmel cache not found for {lang}/{fmt}")
                    job["rawmel_cache"] = rawmel_cache
                    train_meta = train_coupled_bundle_rawmel(
                        language=lang,
                        format_type=fmt,
                        dataset_csv=dataset_csv,
                        out_dir=out_dir,
                        rawmel_cache_dir=rawmel_cache,
                        group_column=args.group_column,
                        alias_types=alias_types,
                        alias_family=str(args.alias_family or ""),
                        min_mapping_confidence=float(mmc),
                        device=args.device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        learning_rate=float(args.learning_rate),
                        min_confidence=float(args.min_confidence),
                        progress_every=100,
                    )
                else:
                    train_meta = train_coupled_bundle(
                        language=lang,
                        format_type=fmt,
                        dataset_csv=dataset_csv,
                        out_dir=out_dir,
                        group_column=args.group_column,
                        alias_types=alias_types,
                        alias_family=str(args.alias_family or ""),
                        min_mapping_confidence=float(mmc),
                        device=args.device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        learning_rate=float(args.learning_rate),
                        min_confidence=float(args.min_confidence),
                        progress_every=100,
                    )
                job["train_meta"] = train_meta

            if not args.skip_eval:
                eval_summary = evaluate_coupled_bundle(
                    model_dir=out_dir,
                    dataset_csv=dataset_csv,
                    language=lang,
                    format_type=fmt,
                    device=args.device,
                    rawmel_cache_dir=str(job.get("rawmel_cache", "") or ""),
                )
                os.makedirs(os.path.dirname(eval_report), exist_ok=True)
                with open(eval_report, "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2)
                job["eval_summary"] = eval_summary

            job["status"] = "ok"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            summary["failures"].append(
                {
                    "language": lang,
                    "format_type": fmt,
                    "dataset_csv": dataset_csv,
                    "error": str(exc),
                }
            )
            if args.require_all:
                break

    summary["job_count"] = len(summary["jobs"])
    summary["ok_count"] = sum(1 for j in summary["jobs"] if j.get("status") in {"ok", "dry_run"})
    summary["failed_count"] = len(summary["failures"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_all and summary["failed_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
