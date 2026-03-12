from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import normalize_format_type, normalize_language_name
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml_coupled import build_and_save_coupled_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build coupled dataset CSV from dataset root (bulk).")
    ap.add_argument("--dataset-root", default=os.path.join(ROOT, "dataset"), help="Dataset root")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Optional max candidates to process")
    ap.add_argument("--report", default="", help="Optional JSON report path")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(dataset_root)

    lang = normalize_language_name(args.lang)
    fmt = normalize_format_type(lang, args.format) or str(args.format or "").strip().lower()
    if not fmt:
        raise SystemExit("Invalid --format")

    candidates = discover_training_candidates_from_dataset_root(dataset_root)
    filtered = [c for c in candidates if str(c.language).strip().lower() == lang and str(c.format_type).strip().lower() == fmt]

    results = []
    append = False
    processed = 0
    for idx, c in enumerate(filtered, start=1):
        if args.limit and processed >= int(args.limit):
            break
        entry = {
            "language": c.language,
            "format_type": c.format_type,
            "root": c.root,
            "folder": c.folder,
            "manual_oto": c.manual_oto,
            "auto_oto": c.auto_oto,
            "tg_dir": c.tg_dir,
            "wav_dir": c.wav_dir,
            "custom_phonemes": c.custom_phonemes,
            "voicebank_id": c.voicebank_id,
            "status": c.status,
            "reason": c.reason,
        }
        if c.status != "ready":
            results.append(entry)
            continue
        stats = build_and_save_coupled_dataset(
            language=lang,
            auto_oto_path=c.auto_oto,
            manual_oto_path=c.manual_oto,
            tg_dir=c.tg_dir,
            wav_dir=c.wav_dir,
            out_csv=args.out,
            custom_phonemes_path=c.custom_phonemes,
            voicebank_id=c.voicebank_id,
            append=append,
            format_type_override=fmt,
        )
        processed += 1
        append = True
        entry.update(stats)
        entry["status"] = "built"
        results.append(entry)
        if processed % 10 == 0:
            print(f"[BUILD] processed={processed} out={os.path.abspath(args.out)}")

    summary = {
        "language": lang,
        "format_type": fmt,
        "dataset_root": dataset_root,
        "out_csv": os.path.abspath(args.out),
        "candidates_total": int(len(filtered)),
        "candidates_built": int(sum(1 for r in results if r.get("status") == "built")),
        "rows_saved": int(sum(int(r.get("saved_rows", 0) or 0) for r in results)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "items": results}, f, ensure_ascii=False, indent=2)
        print(report_path)


if __name__ == "__main__":
    main()
