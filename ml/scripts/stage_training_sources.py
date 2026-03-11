from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_staging import (
    stage_training_sources,
    write_stage_manifest_csv,
    write_stage_manifest_json,
)
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Copy training-source files into Auto_OTO/dataset.")
    ap.add_argument(
        "--config",
        default=os.path.join(ROOT, "ml", "configs", "training_data_roots.yaml"),
        help="Training roots YAML config",
    )
    ap.add_argument(
        "--dataset-root",
        default=os.path.join(ROOT, "dataset"),
        help="Destination dataset root under current project",
    )
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    results = stage_training_sources(args.config, dataset_root=dataset_root)
    json_path = os.path.join(dataset_root, "_manifest", "staged_sources.json")
    csv_path = os.path.join(dataset_root, "_manifest", "staged_sources.csv")
    write_stage_manifest_json(json_path, results)
    write_stage_manifest_csv(csv_path, results)

    summary = {
        "dataset_root": dataset_root,
        "total": len(results),
        "staged": sum(1 for r in results if r.status == "staged"),
        "skipped": sum(1 for r in results if r.status != "staged"),
        "copied_files": sum(r.copied_files for r in results),
        "copied_wavs": sum(r.copied_wavs for r in results),
        "copied_otos": sum(r.copied_otos for r in results),
        "manifest_json": os.path.abspath(json_path),
        "manifest_csv": os.path.abspath(csv_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
