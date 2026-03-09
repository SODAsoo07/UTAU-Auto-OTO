"""
Export OTO ML bundles and deploy them into a built UI runtime folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_bundle_install import install_exported_bundle
from core.oto_ml_export import export_model_bundle, resolve_bundle_dir


DEFAULT_APP_NAME = "UTAU_Auto_OTO"


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _default_asset_root() -> str:
    return os.path.join(ROOT, "assets", "models", "oto_ml")


def _default_export_root() -> str:
    return os.path.join(ROOT, "exports", "oto_ml")


def _candidate_ui_roots() -> List[str]:
    return [
        os.path.join(ROOT, "UTAU_Auto_OTO_Release", DEFAULT_APP_NAME),
        os.path.join(ROOT, "dist", DEFAULT_APP_NAME),
        ROOT,
    ]


def _resolve_ui_root(raw_ui_root: str) -> str:
    if raw_ui_root:
        path = os.path.abspath(raw_ui_root)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"UI root not found: {path}")
        return path
    for candidate in _candidate_ui_roots():
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a UI runtime folder automatically. "
        "Use --ui-root or --install-root explicitly."
    )


def _resolve_model_dir(asset_root: str, bundle_spec: str) -> str:
    raw = str(bundle_spec or "").strip()
    if not raw:
        raise ValueError("Empty bundle spec.")
    if os.path.isdir(raw):
        return os.path.abspath(raw)
    raw_path = os.path.abspath(raw)
    if os.path.isdir(raw_path):
        return raw_path
    model_dir = resolve_bundle_dir(asset_root, raw)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model bundle directory not found: {bundle_spec}")
    return model_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OTO ML model bundles and copy them into a UI runtime folder."
    )
    parser.add_argument(
        "bundle_specs",
        nargs="+",
        help=(
            "Model directory paths or paths relative to assets/models/oto_ml. "
            "Example: japanese/cvvc/v1 or assets/models/oto_ml/japanese/cvvc/families/cv/v1"
        ),
    )
    parser.add_argument(
        "--asset-root",
        default=_default_asset_root(),
        help="Base directory for relative bundle specs.",
    )
    parser.add_argument(
        "--export-root",
        default=_default_export_root(),
        help="Directory where exported bundles are written.",
    )
    parser.add_argument(
        "--ui-root",
        default="",
        help=(
            "Built UI application root. "
            "Default search order: UTAU_Auto_OTO_Release/UTAU_Auto_OTO, dist/UTAU_Auto_OTO, repo root."
        ),
    )
    parser.add_argument(
        "--install-root",
        default="",
        help="Exact models_installed/oto_ml root. Overrides --ui-root when set.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create zip bundles in the export directory.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_console_encoding()
    args = _parse_args()
    asset_root = os.path.abspath(args.asset_root)
    export_root = os.path.abspath(args.export_root)
    os.makedirs(export_root, exist_ok=True)

    if args.install_root:
        install_root = os.path.abspath(args.install_root)
        os.makedirs(install_root, exist_ok=True)
        ui_root = os.path.dirname(os.path.dirname(install_root))
    else:
        ui_root = _resolve_ui_root(args.ui_root)
        install_root = os.path.join(ui_root, "models_installed", "oto_ml")
        os.makedirs(install_root, exist_ok=True)

    results = []
    for bundle_spec in args.bundle_specs:
        model_dir = _resolve_model_dir(asset_root, bundle_spec)
        manifest = export_model_bundle(model_dir, export_root, create_zip=bool(args.zip))
        install_result = install_exported_bundle(manifest["export_dir"], install_root=install_root)
        results.append({
            "bundle_spec": bundle_spec,
            "model_dir": model_dir,
            "bundle_slug": manifest["bundle_slug"],
            "export_dir": manifest["export_dir"],
            "zip_path": manifest.get("zip_path", ""),
            "installed_dir": install_result["installed_dir"],
            "language": install_result["language"],
            "format_type": install_result["format_type"],
            "alias_family": install_result.get("alias_family", ""),
            "model_version": install_result.get("model_version", ""),
        })

    summary = {
        "asset_root": asset_root,
        "export_root": export_root,
        "ui_root": ui_root,
        "install_root": install_root,
        "bundle_count": len(results),
        "bundles": results,
    }
    summary_path = os.path.join(export_root, "deploy_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print()
    print(f"Deploy complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


