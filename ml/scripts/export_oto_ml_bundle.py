from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_export import export_multiple_model_bundles, resolve_bundle_dir


def main():
    ap = argparse.ArgumentParser(description="Export distributable OTO ML model bundles.")
    ap.add_argument(
        "--asset-root",
        default=os.path.join(ROOT, "assets", "models", "oto_ml"),
        help="Asset root containing trained model bundles",
    )
    ap.add_argument(
        "--export-root",
        default=os.path.abspath(os.path.join(ROOT, "..", "ml_workspace", "exports", "model_bundles")),
        help="Output directory for exported bundles",
    )
    ap.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Bundle spec relative to asset root, e.g. japanese/cvvc/v1",
    )
    ap.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Explicit model directory to export",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Also create zip archives per bundle",
    )
    args = ap.parse_args()

    model_dirs = [os.path.abspath(p) for p in args.model_dir]
    for spec in args.bundle:
        model_dirs.append(resolve_bundle_dir(args.asset_root, spec))
    if not model_dirs:
        ap.error("At least one --bundle or --model-dir is required.")

    result = export_multiple_model_bundles(model_dirs, args.export_root, create_zip=args.zip)
    print(json.dumps({
        "export_root": result["export_root"],
        "bundle_count": result["bundle_count"],
        "manifest": os.path.join(result["export_root"], "export_manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
