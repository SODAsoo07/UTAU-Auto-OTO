from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.mfa_free_oto.manifest_audit import audit_manifest_path, top_flagged_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit MFA-free OTO manual-gold manifests before training/evaluation."
    )
    parser.add_argument("--manifest", action="append", required=True, help="Input JSONL manifest. Repeatable.")
    parser.add_argument("--out-dir", required=True, help="Directory for audit JSON outputs.")
    parser.add_argument(
        "--write-clean-manifest",
        action="store_true",
        help="Write a companion *.clean.jsonl excluding hard-invalid rows.",
    )
    parser.add_argument("--flag-limit", type=int, default=80, help="Rows to include in *.flagged.json.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for manifest in args.manifest:
        manifest_path = Path(manifest)
        stem = manifest_path.stem
        clean_out = out_dir / f"{stem}.clean.jsonl" if args.write_clean_manifest else None
        report = audit_manifest_path(manifest_path, clean_out=clean_out)
        full_path = out_dir / f"{stem}.audit.json"
        flagged_path = out_dir / f"{stem}.flagged.json"
        full_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        flagged_path.write_text(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "rows": report["rows"],
                    "flag_counts": report["flag_counts"],
                    "flagged_rows": top_flagged_rows(report, limit=args.flag_limit),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        summaries.append(
            {
                "manifest_path": str(manifest_path),
                "audit_path": str(full_path),
                "flagged_path": str(flagged_path),
                "clean_manifest": report.get("clean_manifest"),
                "rows": report["rows"],
                "clean_rows": report["clean_rows"],
                "flagged_rows": report["flagged_rows"],
                "hard_flag_rows": report["hard_flag_rows"],
                "slot_metric_eligible_rows": report["slot_metric_eligible_rows"],
                "flag_counts": report["flag_counts"],
                "label_duration_ratio": report["label_duration_ratio"],
            }
        )
    summary = {"manifests": summaries}
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

