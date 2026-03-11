from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_export import export_multiple_model_bundles, resolve_bundle_dir
from core.runtime_encoding import bootstrap_utf8_runtime


def _resolve_model_dir_arg(asset_root: str, bundle_arg: str) -> str:
    raw = str(bundle_arg or "").strip()
    if not raw:
        return ""

    normalized = raw.replace("/", os.sep).replace("\\", os.sep)
    asset_root_abs = os.path.abspath(asset_root)
    asset_prefix = os.path.normcase(os.path.join("assets", "models", "oto_ml"))

    # 1) Absolute path as-is.
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)

    # 2) Relative path from current working dir.
    cwd_candidate = os.path.abspath(normalized)
    if os.path.isdir(cwd_candidate):
        return cwd_candidate

    # 3) Relative path from repository root.
    root_candidate = os.path.abspath(os.path.join(ROOT, normalized))
    if os.path.isdir(root_candidate):
        return root_candidate

    # 4) If user passed "assets/models/oto_ml/...", trim this prefix.
    norm_rel = os.path.normcase(os.path.normpath(normalized))
    if norm_rel.startswith(asset_prefix + os.sep):
        tail = normalized.split(os.sep, 3)[3] if len(normalized.split(os.sep)) >= 4 else ""
        if tail:
            return resolve_bundle_dir(asset_root_abs, tail)

    # 5) Default: treat as asset-root relative bundle spec.
    return resolve_bundle_dir(asset_root_abs, normalized)


def main():
    bootstrap_utf8_runtime()
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
        help="Bundle path or spec (absolute path, cwd-relative path, or asset-root-relative spec like japanese/cvvc/v1)",
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
        resolved = _resolve_model_dir_arg(args.asset_root, spec)
        if resolved:
            model_dirs.append(resolved)
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
