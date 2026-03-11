from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_bundle_install import install_exported_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Install exported OTO ML bundle for runtime use.")
    ap.add_argument("--bundle", required=True, help="Bundle directory or zip path")
    ap.add_argument("--install-root", default="", help="Optional install root")
    args = ap.parse_args()
    result = install_exported_bundle(args.bundle, install_root=args.install_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
