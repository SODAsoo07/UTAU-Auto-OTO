from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mfa_free_oto.manifest_audit import infer_filename_phone_sequence
from core.mfa_free_oto.oto_adapter import (
    OtoAdapterConfig,
    OtoTemplateRow,
    adapt_template_row,
    assign_template_row_anchors,
    anchors_from_prediction,
    bootstrap_row,
    load_oto_template_rows,
)
from core.mfa_free_oto.review_overlay import write_review_html
from core.mfa_free_oto.runtime_inference import RuntimePrediction, predict_wav


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MFA-free OTO checkpoint inference and write preview oto.ini/anchor JSON/review overlays."
    )
    parser.add_argument("--checkpoint", required=True, help="MFA-free OTO PoC checkpoint.")
    parser.add_argument("--wav-dir", help="Directory containing WAV files.")
    parser.add_argument("--wav", action="append", help="Single WAV path. May be passed multiple times.")
    parser.add_argument("--source-oto", help="Existing/template oto.ini for template-preserve mode.")
    parser.add_argument("--out-oto", required=True, help="Preview oto.ini output.")
    parser.add_argument("--out-json", required=True, help="Anchor/adaptation JSON output.")
    parser.add_argument("--overlay-dir", help="Optional directory for one HTML review overlay per WAV.")
    parser.add_argument("--mode", choices=("template-preserve", "bootstrap"), help="Defaults to template-preserve when --source-oto is set, otherwise bootstrap.")
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--format-type", default="CV")
    parser.add_argument("--alias-type", default="auto", help="auto, cv, cv_head, vcv, vc, vv, or v.")
    parser.add_argument("--encoder", help="Override encoder stored in checkpoint.")
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--limit", type=int, help="Limit WAV count for smoke runs.")
    parser.add_argument("--disable-slot-viterbi", action="store_true")
    args = parser.parse_args()

    mode = args.mode or ("template-preserve" if args.source_oto else "bootstrap")
    wav_paths = _collect_wavs(args.wav_dir, args.wav, limit=args.limit)
    if not wav_paths:
        raise SystemExit("No WAV files found. Pass --wav-dir or --wav.")

    config = OtoAdapterConfig(
        mode=mode,
        language=args.language,
        format_type=args.format_type,
        alias_type=args.alias_type,
    )
    template_by_wav: dict[str, list[OtoTemplateRow]] = defaultdict(list)
    if args.source_oto:
        for row in load_oto_template_rows(args.source_oto):
            template_by_wav[row.wav.lower()].append(row)
        targets = [path for path in wav_paths if path.name.lower() in template_by_wav]
    else:
        targets = wav_paths
    if args.limit:
        targets = targets[: args.limit]

    out_lines: list[str] = []
    records: list[dict] = []
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in targets:
        expected_phones = infer_filename_phone_sequence(wav_path.name)
        prediction = predict_wav(
            wav_path,
            checkpoint_path=args.checkpoint,
            expected_phones=expected_phones,
            encoder=args.encoder,
            device=args.device,
            use_slot_viterbi=not args.disable_slot_viterbi,
        )
        event_source = _slot_events_for_oto(prediction)
        anchors = anchors_from_prediction(prediction.posterior, event_source)
        file_duration_ms = _duration_ms(wav_path)
        template_rows = template_by_wav.get(wav_path.name.lower()) or []
        adapted = []
        if mode == "template-preserve" and template_rows:
            row_anchors = assign_template_row_anchors(
                prediction.posterior,
                event_source,
                template_rows,
                min_score=min(config.min_anchor_score, 0.03),
                use_source_timing_prior=config.preserve_source_timing,
                expected_phones=expected_phones,
            )
            for template_row, anchor in zip(template_rows, row_anchors):
                adapted.append(
                    adapt_template_row(
                        template_row,
                        anchor,
                        file_duration_ms=file_duration_ms,
                        config=config,
                    )
                )
        else:
            alias = wav_path.stem
            anchor = anchors[0] if anchors else None
            adapted.append(
                bootstrap_row(
                    wav_path.name,
                    alias,
                    anchor,
                    file_duration_ms=file_duration_ms,
                    config=OtoAdapterConfig(
                        mode="bootstrap",
                        language=args.language,
                        format_type=args.format_type,
                        alias_type=args.alias_type,
                    ),
                )
            )
        out_lines.extend(row.format_line() for row in adapted)
        record = {
            "wav": wav_path.name,
            "wav_path": str(wav_path),
            "expected_phones": expected_phones,
            "prediction": prediction.to_json_dict(),
            "anchors": [
                {
                    "anchor_abs_ms": anchor.anchor_abs_ms,
                    "score": anchor.score,
                    "role": anchor.role,
                    "frame_index": anchor.frame_index,
                    "vowel_start_abs_ms": anchor.vowel_start_abs_ms,
                    "vowel_end_abs_ms": anchor.vowel_end_abs_ms,
                    "warnings": list(anchor.warnings),
                }
                for anchor in anchors
            ],
            "generated_oto_rows": [row.to_json_dict() for row in adapted],
        }
        records.append(record)
        if overlay_dir:
            _write_overlay(overlay_dir, wav_path, expected_phones, prediction, record)

    Path(args.out_oto).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_oto).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "mode": mode,
                "language": args.language,
                "format_type": args.format_type,
                "alias_type": args.alias_type,
                "rows": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"wav_count": len(targets), "oto_rows": len(out_lines), "out_oto": args.out_oto, "out_json": args.out_json}, ensure_ascii=False, indent=2))
    return 0


def _collect_wavs(wav_dir: str | None, wav_args: list[str] | None, *, limit: int | None) -> list[Path]:
    paths: list[Path] = []
    if wav_dir:
        paths.extend(sorted(Path(wav_dir).glob("*.wav")))
    for item in wav_args or []:
        path = Path(item)
        if path.is_file():
            paths.append(path)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve()).lower()] = path
    out = list(unique.values())
    return out[:limit] if limit else out


def _duration_ms(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        return 1000.0 * float(handle.getnframes()) / float(handle.getframerate())


def _write_overlay(
    overlay_dir: Path,
    wav_path: Path,
    expected_phones: list[str],
    prediction: RuntimePrediction,
    record: dict,
) -> None:
    row = {
        "row_id": wav_path.stem,
        "wav_name": wav_path.name,
        "wav_path": str(wav_path),
        "duration_ms": _duration_ms(wav_path),
        "expected_phones": expected_phones,
        "label_source": "mfa_free_preview",
    }
    write_review_html(
        overlay_dir / f"{wav_path.stem}.html",
        row,
        posterior=prediction.posterior,
        decoded_events=prediction.decoded_events,
        generated_oto_rows=record["generated_oto_rows"],
    )


def _slot_events_for_oto(prediction: RuntimePrediction):
    if prediction.slot_result is None or not prediction.slot_result.assignments:
        return prediction.decoded_events
    return [
        {
            "label": assignment.event_label,
            "selected_time_ms": assignment.selected_time_ms,
            "score": assignment.score,
            "frame_index": assignment.frame_index,
            "expected_phone": assignment.phone,
            "expected_phone_index": assignment.phone_index,
            "slot_index": assignment.slot_index,
        }
        for assignment in prediction.slot_result.assignments
    ]


if __name__ == "__main__":
    raise SystemExit(main())
