from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.evaluation import evaluate_phoneme_boundary_manifest
from core.phoneme_boundary.manifest import (
    BoundaryManifestBuildConfig,
    build_boundary_manifest,
    write_manifest_jsonl,
)
from core.phoneme_boundary.model import PhonemeBoundaryDetectorConfig
from core.phoneme_boundary.training import PhonemeBoundaryTrainConfig, train_phoneme_boundary_from_manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a real voicebank smoke: build phoneme-boundary manifest, train one small model, then evaluate it."
    )
    ap.add_argument("--voicebank-dir", required=True, help="Voicebank/staged folder containing wavs and usually oto.ini/TextGrid files.")
    ap.add_argument("--source-oto", default="", help="oto.ini used only for aliases. Defaults to <voicebank-dir>/oto.ini.")
    ap.add_argument("--sidecar", default="", help="Optional JSON/JSONL phone/boundary sidecar. TextGrid is used when this is absent.")
    ap.add_argument("--out-dir", default=os.path.join("ml_workspace", "phoneme_boundary", "real_voicebank_smoke"))
    ap.add_argument("--language", default="")
    ap.add_argument("--max-rows", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--n-mels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--conv-channels", type=int, default=8)
    ap.add_argument("--arch", choices=["crnn", "conv_only"], default="crnn")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--use-textgrids", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    voicebank_dir = os.path.abspath(str(args.voicebank_dir))
    source_oto = str(args.source_oto or "").strip() or os.path.join(voicebank_dir, "oto.ini")
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    model_path = os.path.join(out_dir, "phoneme_boundary_smoke.pt")
    eval_path = os.path.join(out_dir, "eval.json")
    summary_path = os.path.join(out_dir, "summary.json")

    manifest_result = build_boundary_manifest(
        BoundaryManifestBuildConfig(
            wav_dir=voicebank_dir,
            aliases_path="",
            source_oto_path=source_oto,
            sidecar_path=str(args.sidecar or ""),
            textgrid_dir=voicebank_dir,
            use_textgrids=bool(args.use_textgrids),
            out_path=manifest_path,
            language=str(args.language or ""),
            require_targets=True,
            max_rows=int(args.max_rows),
        )
    )
    write_manifest_jsonl(manifest_path, manifest_result.rows)
    trainable_rows = int(manifest_result.summary.get("trainable_rows", 0))
    if trainable_rows <= 0:
        payload = {
            "status": "no_trainable_rows",
            "manifest_summary": manifest_result.summary,
            "manifest": manifest_path,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    model_cfg = PhonemeBoundaryDetectorConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        arch_type=str(args.arch),
    )
    train_cfg = PhonemeBoundaryTrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=False,
        num_workers=int(args.num_workers),
        val_ratio=float(args.val_ratio),
        log_every=int(args.log_every),
    )
    train_summary = train_phoneme_boundary_from_manifest(
        manifest_result.rows,
        model_path,
        manifest_dir=os.path.dirname(manifest_path),
        train_config=train_cfg,
        model_config=model_cfg,
    )
    eval_summary = evaluate_phoneme_boundary_manifest(
        manifest_result.rows,
        model_path=model_path,
        manifest_dir=os.path.dirname(manifest_path),
        device=str(args.device),
    )
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(eval_summary, handle, ensure_ascii=False, indent=2)
    payload = {
        "status": "ok",
        "voicebank_dir": voicebank_dir,
        "manifest": manifest_path,
        "model": model_path,
        "eval": eval_path,
        "manifest_summary": manifest_result.summary,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
