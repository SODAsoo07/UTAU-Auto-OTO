from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_collection import (
    build_datasets_from_candidates,
    discover_training_candidates_from_dataset_root,
    discover_training_candidates,
    write_candidate_csv,
    write_candidate_manifest,
)


def main():
    ap = argparse.ArgumentParser(description="Discover recursive OTO ML training candidates and build datasets.")
    ap.add_argument(
        "--config",
        default=os.path.join(ROOT, "ml", "configs", "training_data_roots.yaml"),
        help="Training roots YAML config",
    )
    ap.add_argument(
        "--workspace-root",
        default=os.path.abspath(os.path.join(ROOT, "..", "ml_workspace")),
        help="Workspace root for datasets/reports",
    )
    ap.add_argument(
        "--manifest-json",
        default="",
        help="Optional override for manifest json path",
    )
    ap.add_argument(
        "--manifest-csv",
        default="",
        help="Optional override for manifest csv path",
    )
    ap.add_argument(
        "--dataset-root",
        default=os.path.join(ROOT, "dataset"),
        help="Staged dataset root to scan directly (default: Auto_OTO/dataset)",
    )
    ap.add_argument(
        "--use-config-roots",
        action="store_true",
        help="Use training_data_roots.yaml direct roots instead of staged dataset",
    )
    args = ap.parse_args()

    workspace_root = os.path.abspath(args.workspace_root)
    manifest_json = args.manifest_json or os.path.join(workspace_root, "reports", "training_candidates.json")
    manifest_csv = args.manifest_csv or os.path.join(workspace_root, "reports", "training_candidates.csv")
    total_steps = 4

    dataset_root = os.path.abspath(args.dataset_root)
    print(f"[1/{total_steps}] Discovering training candidates...", flush=True)
    if (not args.use_config_roots) and os.path.isdir(dataset_root):
        candidates = discover_training_candidates_from_dataset_root(dataset_root)
        # Fallback for empty staged trees.
        if not candidates:
            candidates = discover_training_candidates(args.config)
    else:
        candidates = discover_training_candidates(args.config)

    print(f"[2/{total_steps}] Writing candidate manifests... (count={len(candidates)})", flush=True)
    write_candidate_manifest(manifest_json, candidates)
    write_candidate_csv(manifest_csv, candidates)

    print(f"[3/{total_steps}] Building datasets from candidates...", flush=True)
    result = build_datasets_from_candidates(candidates, workspace_root=workspace_root)

    final_json = os.path.join(workspace_root, "reports", "training_build_results.json")
    print(f"[4/{total_steps}] Writing build results...", flush=True)
    write_candidate_manifest(final_json, result["candidates"])

    print(json.dumps({
        "summary": result["summary"],
        "manifest_json": os.path.abspath(manifest_json),
        "manifest_csv": os.path.abspath(manifest_csv),
        "result_json": os.path.abspath(final_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
