from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mfa_free_oto.manual_anchor_runtime import (  # noqa: E402
    ManualAnchorPreviewConfig,
    generate_manual_oto_anchor_preview,
    resolve_latest_manual_oto_anchor_scorer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run manual_oto_anchor_scorer.pkl against a source oto.ini and write preview OTO/JSON/overlays."
    )
    parser.add_argument("--scorer", help="manual_oto_anchor_scorer.pkl. Defaults to env/latest ml_workspace/manual_oto_anchor.")
    parser.add_argument("--wav-dir", required=True, help="Directory containing WAV files referenced by source oto.ini.")
    parser.add_argument("--source-oto", required=True, help="Source oto.ini. Alias rows are used; source timing is ignored.")
    parser.add_argument("--out-oto", required=True, help="Generated preview oto.ini output.")
    parser.add_argument("--out-json", required=True, help="Prediction JSON output.")
    parser.add_argument("--overlay-dir", help="Optional review overlay directory.")
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--format-type", default="CV")
    parser.add_argument("--limit", type=int, help="Limit WAV count for smoke runs.")
    args = parser.parse_args()

    scorer = str(args.scorer or "").strip()
    if not scorer:
        scorer = resolve_latest_manual_oto_anchor_scorer(base_dir=str(REPO_ROOT))
    if not scorer:
        raise SystemExit("manual_oto_anchor_scorer.pkl not found. Set --scorer or UTOA_MANUAL_OTO_ANCHOR_SCORER.")

    def _print_progress(message: str) -> None:
        print(f"[progress] {message}", flush=True)

    summary = generate_manual_oto_anchor_preview(
        scorer,
        ManualAnchorPreviewConfig(
            wav_dir=args.wav_dir,
            source_oto_path=args.source_oto,
            out_oto_path=args.out_oto,
            out_json_path=args.out_json,
            overlay_dir=args.overlay_dir or "",
            language=args.language,
            format_type=args.format_type,
            limit=args.limit,
        ),
        progress_callback=_print_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
