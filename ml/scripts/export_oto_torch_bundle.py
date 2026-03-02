from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_export import export_torch_bundle


def main():
    ap = argparse.ArgumentParser(description="Export PyTorch OTO bridge bundle.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bundle-slug", default="")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    manifest = export_torch_bundle(
        model_dir=args.model_dir,
        export_root=args.out_dir,
        bundle_slug=args.bundle_slug,
        create_zip=args.zip,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
