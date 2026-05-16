import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = str(Path(__file__).resolve().parents[4])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _load_collection_deps():
    from core.oto_ml_collection import (  # noqa: E402
        build_datasets_from_candidates,
        discover_training_candidates_from_dataset_root,
        write_candidate_csv,
        write_candidate_manifest,
    )

    return (
        build_datasets_from_candidates,
        discover_training_candidates_from_dataset_root,
        write_candidate_csv,
        write_candidate_manifest,
    )


def _build_summary(candidates) -> Dict[str, object]:
    by_lang_fmt: Dict[str, Dict[str, Dict[str, int]]] = {}
    status_counts: Dict[str, int] = {}
    for item in candidates:
        lang = str(getattr(item, "language", "") or "").strip().lower() or "unknown"
        fmt = str(getattr(item, "format_type", "") or "").strip().lower() or "unknown"
        status = str(getattr(item, "status", "") or "pending").strip().lower() or "pending"
        lang_bucket = by_lang_fmt.setdefault(lang, {})
        fmt_bucket = lang_bucket.setdefault(fmt, {"total": 0, "ready": 0, "built": 0, "skip": 0})
        fmt_bucket["total"] += 1
        if status in fmt_bucket:
            fmt_bucket[status] += 1
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "total": int(len(candidates)),
        "status_counts": status_counts,
        "by_language_format": by_lang_fmt,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-classify OTO ML training candidates from a staged dataset root."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root organized as <root>/<language>/<format>/<voicebank>/...",
    )
    parser.add_argument(
        "--workspace-root",
        default=os.path.join(ROOT, "ml_workspace"),
        help="Workspace root for manifests and optional dataset CSV outputs.",
    )
    parser.add_argument(
        "--auto-oto-policy",
        default="require",
        choices=["require", "generate-temp", "generate-persist"],
        help="Policy for missing auto oto in candidate discovery.",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Build dataset CSV files under <workspace-root>/datasets after classification.",
    )
    parser.add_argument("--manifest-json", default="", help="Output path for candidate manifest JSON.")
    parser.add_argument("--manifest-csv", default="", help="Output path for candidate manifest CSV.")
    parser.add_argument("--summary-json", default="", help="Output path for summary JSON.")
    return parser.parse_args()


def main():
    args = _parse_args()
    (
        build_datasets_from_candidates,
        discover_training_candidates_from_dataset_root,
        write_candidate_csv,
        write_candidate_manifest,
    ) = _load_collection_deps()
    dataset_root = os.path.abspath(str(args.dataset_root).strip())
    workspace_root = os.path.abspath(str(args.workspace_root).strip())
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    os.makedirs(workspace_root, exist_ok=True)

    manifest_dir = os.path.join(workspace_root, "_manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_json = (
        os.path.abspath(str(args.manifest_json).strip())
        if str(args.manifest_json).strip()
        else os.path.join(manifest_dir, "training_candidates.json")
    )
    manifest_csv = (
        os.path.abspath(str(args.manifest_csv).strip())
        if str(args.manifest_csv).strip()
        else os.path.join(manifest_dir, "training_candidates.csv")
    )
    summary_json = (
        os.path.abspath(str(args.summary_json).strip())
        if str(args.summary_json).strip()
        else os.path.join(manifest_dir, "training_candidates_summary.json")
    )

    candidates = discover_training_candidates_from_dataset_root(
        dataset_root=dataset_root,
        auto_oto_policy=str(args.auto_oto_policy or "require").strip().lower(),
    )
    write_candidate_manifest(manifest_json, candidates)
    write_candidate_csv(manifest_csv, candidates)

    summary: Dict[str, object] = _build_summary(candidates)
    if bool(args.build_datasets):
        build_result = build_datasets_from_candidates(
            candidates=candidates,
            workspace_root=workspace_root,
            auto_oto_policy=str(args.auto_oto_policy or "require").strip().lower(),
            progress_callback=print,
        )
        summary["build"] = {
            "summary": dict(build_result.get("summary") or {}),
            "built_candidates": int(len(build_result.get("candidates") or [])),
        }
        # rewrite manifests with post-build statuses/saved rows
        write_candidate_manifest(manifest_json, candidates)
        write_candidate_csv(manifest_csv, candidates)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"dataset_root={dataset_root}")
    print(f"workspace_root={workspace_root}")
    print(f"manifest_json={manifest_json}")
    print(f"manifest_csv={manifest_csv}")
    print(f"summary_json={summary_json}")
    print(f"candidates_total={summary.get('total', 0)}")
    print(f"status_counts={json.dumps(summary.get('status_counts', {}), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
