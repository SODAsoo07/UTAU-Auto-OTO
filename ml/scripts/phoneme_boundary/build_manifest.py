from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.manifest import (
    BoundaryManifestBuildConfig,
    build_boundary_manifest,
    write_manifest_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a phoneme-boundary manifest from WAV order, user aliases, and optional boundary sidecar."
    )
    ap.add_argument("--wav-dir", required=True, help="Directory containing wav files in recording order.")
    ap.add_argument("--aliases", default="", help="User-provided alias rows. One row must match one wav.")
    ap.add_argument("--source-oto", default="", help="Optional oto.ini used only for wav->alias strings; timing values are ignored.")
    ap.add_argument("--sidecar", default="", help="Optional JSON/JSONL with phone intervals or boundary events.")
    ap.add_argument("--textgrid-dir", default="", help="Optional directory containing matching .TextGrid files.")
    ap.add_argument("--use-textgrids", action="store_true", help="Use matching TextGrid phone tier as phone intervals when no sidecar row exists.")
    ap.add_argument("--out", default=os.path.join("ml_workspace", "phoneme_boundary", "manifest.jsonl"))
    ap.add_argument("--language", default="")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--alias-separator", default="|", help="Separator for aliases inside one wav row. Default: |")
    ap.add_argument("--split-whitespace", action="store_true", help="Split one alias row on whitespace when no separator exists.")
    ap.add_argument("--absolute-paths", action="store_true", help="Write absolute wav paths instead of paths relative to manifest.")
    ap.add_argument("--strict", action="store_true", help="Fail on wav/alias count mismatch or missing required targets.")
    ap.add_argument("--require-targets", action="store_true", help="Only keep rows with sidecar phones/events.")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()
    if not str(args.aliases or "").strip() and not str(args.source_oto or "").strip():
        ap.error("one of --aliases or --source-oto is required")

    result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=str(args.wav_dir),
            aliases_path=str(args.aliases),
            source_oto_path=str(args.source_oto),
            sidecar_path=str(args.sidecar),
            textgrid_dir=str(args.textgrid_dir),
            use_textgrids=bool(args.use_textgrids),
            out_path=str(args.out),
            language=str(args.language),
            recursive=bool(args.recursive),
            alias_separator=str(args.alias_separator),
            split_whitespace=bool(args.split_whitespace),
            absolute_paths=bool(args.absolute_paths),
            strict=bool(args.strict),
            require_targets=bool(args.require_targets),
            max_rows=int(args.max_rows),
            sample_rate=int(args.sample_rate),
        )
    )
    count = write_manifest_jsonl(str(args.out), result.rows)
    summary = dict(result.summary)
    summary["out"] = os.path.abspath(str(args.out))
    summary["written_rows"] = int(count)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if count <= 0:
        return 2
    if bool(args.require_targets) and int(summary.get("trainable_rows", 0)) <= 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
