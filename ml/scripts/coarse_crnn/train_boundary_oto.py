from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig
from core.coarse_crnn.boundary_scorer_training import BoundaryTrainConfig, train_boundary_from_manifest
from core.coarse_crnn.boundary_targets import normalize_ko_coda_in_alias
from core.coarse_crnn.oto_targets import resolve_cutoff_abs_ms


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            text = str(raw).strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Train frame-level boundary scorer for OTO decoding.")
    ap.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits_full", "oto_train.jsonl"))
    ap.add_argument("--out", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_boundary_scorer_v1.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--log-every", type=int, default=80)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--conv-channels", type=int, default=96)
    ap.add_argument(
        "--arch",
        choices=["crnn", "conv_only"],
        default="crnn",
        help="Encoder architecture. 'crnn' = legacy 2-conv + BiLSTM. "
        "'conv_only' = 4-block dilated 1D-conv (distillation student for SSL teacher).",
    )
    ap.add_argument("--enable-phone-aux-heads", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--enable-phone-family-heads", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cvs-loss-weight", type=float, default=0.0)
    ap.add_argument("--consonant-id-loss-weight", type=float, default=0.0)
    ap.add_argument("--vowel-id-loss-weight", type=float, default=0.0)
    ap.add_argument("--consonant-family-loss-weight", type=float, default=0.0)
    ap.add_argument("--vowel-nucleus-loss-weight", type=float, default=0.0)
    ap.add_argument("--vowel-glide-loss-weight", type=float, default=0.0)
    ap.add_argument("--phone-aux-class-balanced", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--phone-aux-class-balance-power", type=float, default=0.5)
    ap.add_argument("--phone-aux-class-balance-max", type=float, default=8.0)
    ap.add_argument("--init-from", default="", help="Warm-start matching weights from an existing boundary scorer checkpoint.")
    ap.add_argument(
        "--freeze-non-phone-aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only phone auxiliary heads after warm-starting the boundary scorer.",
    )
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--quality-loss-weight", type=float, default=0.06)
    # P2-b: was 0.08 — frame-level BCE dominated and boundary timing was
    # under-supervised, contributing to the coda-bridge ±200 ms tail.
    ap.add_argument("--boundary-time-loss-weight", type=float, default=0.20)
    ap.add_argument("--hard-case-oversample", action=argparse.BooleanOptionalAction, default=True)
    # P2-a: bumped 1.8 → 2.5 so the per-row hard-score (now also boosted by
    # the coda-bridge regex and plosive right-onset detection below) lifts
    # KO coda-bridge wavs farther above generic ones.
    ap.add_argument("--hard-case-weight", type=float, default=2.5)
    ap.add_argument("--hard-case-min-ratio", type=float, default=0.10)
    # P2-a: regex catches KO coda-bridge aliases (left token = nasal/liquid
    # coda or uppercase plosive coda, followed by next-syllable onset). The
    # 2026-05-17 wall analysis identified these as the single biggest source
    # of error (31% of total |Δ| in raw-model output) and they are 38× under-
    # represented in their uppercase form vs lowercase in training data.
    ap.add_argument(
        "--hard-case-alias-regex",
        default=r"^(N|M|L|NG|H|T|P|K|R|n|m|l|ng)\s+\S",
    )
    ap.add_argument("--n-like-weight", type=float, default=0.80)
    ap.add_argument("--weak-wav-weight", type=float, default=0.65)
    ap.add_argument("--weak-wav-rms-db-threshold", type=float, default=-32.0)
    ap.add_argument("--weak-wav-auto-percentile", type=float, default=0.15)
    ap.add_argument("--normalize-target-fields", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-invalid-target-rows", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-wav-backward-rows", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-backward-step-ms", type=float, default=35.0)
    ap.add_argument("--max-wav-backward-rate", type=float, default=0.20)
    ap.add_argument("--clean-summary", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rows = _read_jsonl(args.manifest)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    clean_stats: dict[str, Any] = {"rows_in": int(len(rows))}
    if bool(args.normalize_target_fields):
        rows, clean_stats = _normalize_and_filter_rows(
            rows,
            drop_invalid=bool(args.drop_invalid_target_rows),
            drop_wav_backward_rows=bool(args.drop_wav_backward_rows),
            max_backward_step_ms=float(args.max_backward_step_ms),
            max_wav_backward_rate=float(args.max_wav_backward_rate),
        )
        if str(args.clean_summary or "").strip():
            summary_path = os.path.abspath(str(args.clean_summary))
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(clean_stats, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"manifest_clean": clean_stats}, ensure_ascii=False))
    cfg = BoundaryScorerConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        enable_phone_aux_heads=bool(args.enable_phone_aux_heads),
        enable_phone_family_heads=bool(args.enable_phone_family_heads),
        arch_type=str(args.arch),
    )
    tcfg = BoundaryTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
        val_ratio=float(args.val_ratio),
        quality_loss_weight=float(args.quality_loss_weight),
        pos_weight=float(args.pos_weight),
        boundary_time_loss_weight=float(args.boundary_time_loss_weight),
        hard_case_oversample=bool(args.hard_case_oversample),
        hard_case_weight=float(args.hard_case_weight),
        hard_case_min_ratio=float(args.hard_case_min_ratio),
        hard_case_alias_regex=str(args.hard_case_alias_regex or ""),
        n_like_weight=float(args.n_like_weight),
        weak_wav_weight=float(args.weak_wav_weight),
        weak_wav_rms_db_threshold=float(args.weak_wav_rms_db_threshold),
        weak_wav_auto_percentile=float(args.weak_wav_auto_percentile),
        cvs_loss_weight=float(args.cvs_loss_weight),
        consonant_id_loss_weight=float(args.consonant_id_loss_weight),
        vowel_id_loss_weight=float(args.vowel_id_loss_weight),
        consonant_family_loss_weight=float(args.consonant_family_loss_weight),
        vowel_nucleus_loss_weight=float(args.vowel_nucleus_loss_weight),
        vowel_glide_loss_weight=float(args.vowel_glide_loss_weight),
        phone_aux_class_balanced=bool(args.phone_aux_class_balanced),
        phone_aux_class_balance_power=float(args.phone_aux_class_balance_power),
        phone_aux_class_balance_max=float(args.phone_aux_class_balance_max),
        init_from=str(args.init_from or ""),
        freeze_non_phone_aux=bool(args.freeze_non_phone_aux),
    )
    result = train_boundary_from_manifest(rows, args.out, train_config=tcfg, model_config=cfg)
    result["manifest_clean"] = clean_stats
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _normalize_and_filter_rows(
    rows: list[dict[str, Any]],
    *,
    drop_invalid: bool,
    drop_wav_backward_rows: bool,
    max_backward_step_ms: float,
    max_wav_backward_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    dropped_invalid = 0
    for row in rows:
        item = _normalize_target_row(row)
        if item is None:
            dropped_invalid += 1
            if not drop_invalid:
                cleaned.append(dict(row))
            continue
        cleaned.append(item)

    dropped_wav_keys: set[str] = set()
    if drop_wav_backward_rows:
        by_wav: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cleaned:
            wav_key = os.path.abspath(str(row.get("audio", "") or str(row.get("wav", "") or "")).strip()).lower()
            by_wav[wav_key].append(row)
        for wav_key, wav_rows in by_wav.items():
            ordered = sorted(
                wav_rows,
                key=lambda r: (
                    int(r.get("row_index_in_wav", r.get("line_index", 0)) or 0),
                    int(r.get("line_index", 0) or 0),
                ),
            )
            violations = 0
            comp = 0
            prev_pre = None
            for row in ordered:
                off = float(row.get("target_offset_ms", 0.0) or 0.0)
                pre = float(row.get("target_preutterance_ms", 0.0) or 0.0)
                cur_pre_abs = off + max(0.0, pre)
                if prev_pre is not None:
                    comp += 1
                    if cur_pre_abs + float(max_backward_step_ms) < float(prev_pre):
                        violations += 1
                prev_pre = cur_pre_abs
            if comp > 0 and (float(violations) / float(comp)) > float(max_wav_backward_rate):
                dropped_wav_keys.add(wav_key)
        if dropped_wav_keys:
            cleaned = [
                row
                for row in cleaned
                if os.path.abspath(str(row.get("audio", "") or str(row.get("wav", "") or "")).strip()).lower()
                not in dropped_wav_keys
            ]

    stats = {
        "rows_in": int(len(rows)),
        "rows_out": int(len(cleaned)),
        "dropped_invalid_rows": int(dropped_invalid),
        "dropped_wavs_non_monotonic": int(len(dropped_wav_keys)),
        "drop_rate": float(max(0, len(rows) - len(cleaned))) / float(max(1, len(rows))),
    }
    return cleaned, stats


def _normalize_target_row(row: dict[str, Any]) -> dict[str, Any] | None:
    out = dict(row)
    # P1-a: case-normalize KO uppercase coda tokens so 377 uppercase rows
    # merge into the 13.7k lowercase coda-bridge supervision (2026-05-17
    # coda-bridge wall analysis). Only KO; other languages pass through.
    if str(out.get("language", "") or "").strip().lower() == "korean":
        alias_norm = normalize_ko_coda_in_alias(str(out.get("alias", "") or ""))
        if alias_norm != out.get("alias"):
            out["alias"] = alias_norm
    duration = max(1.0, _to_float(out.get("duration_ms", 0.0), 1.0))
    off = _to_float(out.get("anchor_offset_ms", out.get("target_offset_ms", out.get("offset", 0.0))), 0.0)
    off = max(0.0, off)

    if "anchor_preutterance_ms" in out:
        pre_abs = _to_float(out.get("anchor_preutterance_ms"), off)
    else:
        pre_abs = off + max(0.0, _to_float(out.get("target_preutterance_ms", out.get("pre", 0.0)), 0.0))
    if "anchor_overlap_ms" in out:
        ovl_abs = _to_float(out.get("anchor_overlap_ms"), off)
    else:
        ovl_abs = off + max(0.0, _to_float(out.get("target_overlap_ms", out.get("ovl", 0.0)), 0.0))
    if "anchor_consonant_ms" in out:
        cons_abs = _to_float(out.get("anchor_consonant_ms"), pre_abs + 1.0)
    else:
        cons_abs = off + max(1.0, _to_float(out.get("target_consonant_ms", out.get("cons", 1.0)), 1.0))

    cut_abs_field = _to_float(out.get("anchor_cutoff_ms", out.get("target_cutoff_abs_ms", float("nan"))), float("nan"))
    if cut_abs_field == cut_abs_field and cut_abs_field > 0.0:
        cutoff_abs = float(cut_abs_field)
    else:
        cut_raw = _to_float(out.get("target_cutoff", out.get("cutoff", 0.0)), 0.0)
        cutoff_abs = float(resolve_cutoff_abs_ms(offset_ms=float(off), cutoff=float(cut_raw), duration_ms=float(duration)))

    pre_abs = max(off, pre_abs)
    ovl_abs = min(pre_abs, max(off, ovl_abs))
    cons_abs = max(pre_abs + 1.0, cons_abs)
    cutoff_abs = max(cons_abs + 1.0, min(duration, cutoff_abs))
    if cutoff_abs <= off + 2.0:
        return None

    out["anchor_offset_ms"] = float(off)
    out["anchor_overlap_ms"] = float(ovl_abs)
    out["anchor_preutterance_ms"] = float(pre_abs)
    out["anchor_consonant_ms"] = float(cons_abs)
    out["anchor_cutoff_ms"] = float(cutoff_abs)
    out["target_offset_ms"] = float(off)
    out["target_overlap_ms"] = float(max(0.0, ovl_abs - off))
    out["target_preutterance_ms"] = float(max(0.0, pre_abs - off))
    out["target_consonant_ms"] = float(max(1.0, cons_abs - off))
    out["target_cutoff_abs_ms"] = float(cutoff_abs)
    out["target_cutoff"] = float(-max(1.0, cutoff_abs - off))
    return out


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


if __name__ == "__main__":
    raise SystemExit(main())
