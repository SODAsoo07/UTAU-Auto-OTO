from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_ml_features import build_training_rows, write_dataset_csv


def _to_bool(text: str, default: bool = True) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _filter_rows(rows: List[Dict[str, object]], *, cv_only: bool) -> List[Dict[str, object]]:
    if not cv_only:
        return list(rows)
    out: List[Dict[str, object]] = []
    for row in rows:
        alias_type = str(row.get("alias_type", "") or "").strip().lower()
        if alias_type in {"cv", "cv_head"}:
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build E2E OTO training dataset (Phase A).")
    parser.add_argument("--language", required=True, choices=["korean", "japanese"])
    parser.add_argument("--auto-oto", default="", help="Auto OTO path. Optional when auto generation policy is enabled.")
    parser.add_argument("--manual-oto", required=True, help="Manual OTO path (label source).")
    parser.add_argument("--tg-dir", required=True, help="TextGrid directory.")
    parser.add_argument("--wav-dir", required=True, help="WAV directory.")
    parser.add_argument("--out-csv", required=True, help="Output dataset CSV path.")
    parser.add_argument("--out-summary", default="", help="Optional summary JSON path.")
    parser.add_argument("--custom-phonemes", default="")
    parser.add_argument("--voicebank-id", default="")
    parser.add_argument("--format-type", default="", help="Format override (Phase A recommends cv).")
    parser.add_argument("--auto-oto-policy", default="generate-temp", choices=["require", "generate-temp", "generate-persist"])
    parser.add_argument("--cv-only", default="1", help="1/0. Keep only cv/cv_head rows (default: 1).")
    args = parser.parse_args()

    rows, stats = build_training_rows(
        language=str(args.language),
        auto_oto_path=str(args.auto_oto or ""),
        manual_oto_path=str(args.manual_oto),
        tg_dir=str(args.tg_dir),
        wav_dir=str(args.wav_dir),
        custom_phonemes_path=str(args.custom_phonemes or ""),
        voicebank_id=str(args.voicebank_id or ""),
        format_type_override=str(args.format_type or ""),
        auto_oto_policy=str(args.auto_oto_policy or "generate-temp"),
    )

    cv_only = _to_bool(args.cv_only, True)
    filtered = _filter_rows(rows, cv_only=cv_only)

    out_csv = os.path.abspath(str(args.out_csv))
    write_dataset_csv(out_csv, filtered, append=False)

    summary = {
        "language": str(args.language),
        "manual_oto": os.path.abspath(str(args.manual_oto)),
        "auto_oto": os.path.abspath(str(args.auto_oto)) if str(args.auto_oto or "").strip() else "",
        "tg_dir": os.path.abspath(str(args.tg_dir)),
        "wav_dir": os.path.abspath(str(args.wav_dir)),
        "out_csv": out_csv,
        "format_type": str(args.format_type or ""),
        "cv_only": bool(cv_only),
        "rows_total": int(len(rows)),
        "rows_written": int(len(filtered)),
        "stats": dict(stats or {}),
    }

    out_summary = str(args.out_summary or "").strip()
    if not out_summary:
        out_summary = os.path.splitext(out_csv)[0] + ".summary.json"
    out_summary = os.path.abspath(out_summary)
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[E2E-Dataset] rows={len(filtered)} csv={out_csv}")
    print(f"[E2E-Dataset] summary={out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
