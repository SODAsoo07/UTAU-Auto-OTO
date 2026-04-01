from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.oto_profile_presets import get_ja_profile_preset, get_kr_profile_preset


def _build_shared_payload(language: str, format_type: str) -> dict:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if lang == "korean":
        preset = get_kr_profile_preset(fmt)
        mode = "absolute"
    elif lang == "japanese":
        preset = get_ja_profile_preset(fmt)
        mode = "style"
    else:
        raise ValueError(f"unsupported language: {language}")

    return {
        "schema": "utoa_shared_oto_profile_v1",
        "language": lang,
        "format_type": fmt,
        "mode": mode,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "core.oto_profile_presets",
        "profile": preset,
    }


def _target_formats(language: str) -> list[str]:
    lang = str(language or "").strip().lower()
    if lang == "korean":
        return ["general", "cv", "cvc", "cvvc", "vcv"]
    if lang == "japanese":
        return ["general", "cv", "cvc", "cvvc", "vcv"]
    raise ValueError(f"unsupported language: {language}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate shared OTO profile JSONs.")
    parser.add_argument(
        "--out-dir",
        default=str(Path("assets") / "profiles" / "shared_oto_profiles"),
        help="output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for language in ("korean", "japanese"):
        for fmt in _target_formats(language):
            payload = _build_shared_payload(language, fmt)
            out_path = out_dir / f"{language}_{fmt}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[OK] {out_path}")
            written += 1

    print(f"[DONE] wrote {written} shared profile files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
