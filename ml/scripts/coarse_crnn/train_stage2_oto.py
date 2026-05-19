from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from core.coarse_crnn.boundary_candidates import audio_candidates_to_boundary_candidates, merge_candidates
from core.coarse_crnn.boundary_targets import (
    absolute_anchors_to_oto_params,
    load_row_specs_from_source_oto,
    oto_row_to_absolute_anchors,
)
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.stage2_oto.features import build_stage2_feature_batch
from core.coarse_crnn.stage2_oto.model import build_stage2_model, checkpoint_payload
from core.coarse_crnn.stage2_oto.types import (
    ANCHOR_FIELDS,
    FORMAT_VOCAB,
    ROLE_VOCAB,
    Stage2CheckpointMeta,
    Stage2ModelConfig,
)
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.oto_file_utils import parse_oto_line, read_text_with_fallback

_BASE_FEATURE_COUNT = 12 + 5 + 4
_CAND_BLOCK = 4


def main() -> int:
    ap = argparse.ArgumentParser(description="Train source-independent Stage2 OTO Assigner.")
    ap.add_argument("--dataset-root", default="dataset_staged")
    ap.add_argument("--bank", action="append", default=[], help="Specific voicebank directory. Can be repeated.")
    ap.add_argument("--out", default="ml_workspace/models/coarse_crnn/oto_stage2_assigner_v1.pt")
    ap.add_argument("--language", default="")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--max-banks", type=int, default=16)
    ap.add_argument("--max-wavs", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--min-val-sequences", type=int, default=128)
    ap.add_argument("--confidence-loss-weight", type=float, default=0.15)
    ap.add_argument("--candidate-loss-weight", type=float, default=0.20)
    ap.add_argument("--confidence-mae-scale-ms", type=float, default=240.0)
    ap.add_argument(
        "--confidence-target-mode",
        choices=("improvement", "error"),
        default="improvement",
        help="Train confidence as Stage2-vs-base improvement probability, or legacy gold-error closeness.",
    )
    ap.add_argument("--confidence-improvement-margin-ms", type=float, default=10.0)
    ap.add_argument("--confidence-improvement-scale-ms", type=float, default=35.0)
    ap.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sampling-balance-power", type=float, default=0.75)
    ap.add_argument("--sampling-balance-max-weight", type=float, default=3.0)
    ap.add_argument("--role-loss-balance-power", type=float, default=0.55)
    ap.add_argument("--format-loss-balance-power", type=float, default=0.40)
    ap.add_argument("--loss-balance-min-weight", type=float, default=0.35)
    ap.add_argument("--loss-balance-max-weight", type=float, default=3.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260518)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch = __import__("torch")
    device = _resolve_device(torch, args.device)
    sequences = _build_sequences(
        dataset_root=args.dataset_root,
        banks=args.bank,
        language=args.language,
        format_type=args.format_type,
        max_banks=max(1, int(args.max_banks)),
        max_wavs=max(1, int(args.max_wavs)),
        max_rows=max(1, int(args.max_rows)),
    )
    if not sequences:
        raise SystemExit("no Stage2 training sequences found")
    train_sequences, val_sequences = _split_sequences(
        sequences,
        val_ratio=float(args.val_ratio),
        min_val_sequences=max(0, int(args.min_val_sequences)),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                "phase": "dataset",
                "all_sequences": len(sequences),
                "train_sequences": len(train_sequences),
                "val_sequences": len(val_sequences),
                "rows_total": sum(len(item["features"].rows) for item in sequences),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    numeric_dim = len(sequences[0]["features"].rows[0].numeric)
    config = Stage2ModelConfig(numeric_dim=numeric_dim, hidden_dim=int(args.hidden_dim))
    model = build_stage2_model(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    loss_history: list[dict[str, float]] = []
    best_state = _snapshot_state_dict(model)
    best_epoch = 0
    best_score = float("inf")
    conf_w = max(0.0, float(args.confidence_loss_weight))
    cand_w = max(0.0, float(args.candidate_loss_weight))
    conf_mae_scale_ms = max(1.0, float(args.confidence_mae_scale_ms))
    conf_target_mode = str(args.confidence_target_mode or "improvement").strip().lower()
    conf_improvement_margin_ms = max(0.0, float(args.confidence_improvement_margin_ms))
    conf_improvement_scale_ms = max(1.0, float(args.confidence_improvement_scale_ms))
    balance = _build_balance_plan(
        train_sequences,
        sampling_power=float(args.sampling_balance_power),
        sampling_max_weight=float(args.sampling_balance_max_weight),
        role_loss_power=float(args.role_loss_balance_power),
        format_loss_power=float(args.format_loss_balance_power),
        row_min_weight=float(args.loss_balance_min_weight),
        row_max_weight=float(args.loss_balance_max_weight),
    )
    print(
        json.dumps(
            {
                "phase": "balance",
                "balanced_sampling": bool(args.balanced_sampling),
                "sampling_groups": balance["sampling_groups"],
                "role_counts": balance["role_counts_named"],
                "format_counts": balance["format_counts_named"],
                "role_loss_weights": balance["role_weights_named"],
                "format_loss_weights": balance["format_weights_named"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(max(1, int(args.epochs))):
        epoch_train_sequences = _sample_sequences_for_epoch(
            train_sequences,
            sequence_weights=balance["sequence_weights"],
            balanced_sampling=bool(args.balanced_sampling),
        )
        train_stats = _run_epoch(
            torch=torch,
            model=model,
            sequences=epoch_train_sequences,
            device=device,
            optim=optim,
            training=True,
            conf_weight=conf_w,
            cand_weight=cand_w,
            conf_mae_scale_ms=conf_mae_scale_ms,
            conf_target_mode=conf_target_mode,
            conf_improvement_margin_ms=conf_improvement_margin_ms,
            conf_improvement_scale_ms=conf_improvement_scale_ms,
            role_loss_weights=balance["role_weights"],
            format_loss_weights=balance["format_weights"],
            apply_loss_balancing=True,
        )
        val_stats = _run_epoch(
            torch=torch,
            model=model,
            sequences=val_sequences,
            device=device,
            optim=None,
            training=False,
            conf_weight=conf_w,
            cand_weight=cand_w,
            conf_mae_scale_ms=conf_mae_scale_ms,
            conf_target_mode=conf_target_mode,
            conf_improvement_margin_ms=conf_improvement_margin_ms,
            conf_improvement_scale_ms=conf_improvement_scale_ms,
            role_loss_weights=balance["role_weights"],
            format_loss_weights=balance["format_weights"],
            apply_loss_balancing=False,
        )
        metric = float(val_stats["loss"]) if val_sequences else float(train_stats["loss"])
        if metric < best_score:
            best_score = metric
            best_epoch = epoch + 1
            best_state = _snapshot_state_dict(model)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_stats["loss"]),
            "train_anchor_loss": float(train_stats["anchor_loss"]),
            "train_conf_loss": float(train_stats["conf_loss"]),
            "train_candidate_loss": float(train_stats["candidate_loss"]),
            "val_loss": float(val_stats["loss"]),
            "val_anchor_loss": float(val_stats["anchor_loss"]),
            "val_conf_loss": float(val_stats["conf_loss"]),
            "val_candidate_loss": float(val_stats["candidate_loss"]),
            "best_epoch": float(best_epoch),
            "best_metric": float(best_score),
        }
        loss_history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    model.load_state_dict(best_state, strict=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model=model.cpu().eval(),
        config=config,
        meta=Stage2CheckpointMeta(),
        extra={
            "train_sequences": len(train_sequences),
            "val_sequences": len(val_sequences),
            "train_rows": sum(len(item["features"].rows) for item in train_sequences),
            "val_rows": sum(len(item["features"].rows) for item in val_sequences),
            "best_epoch": int(best_epoch),
            "best_metric": float(best_score),
            "loss_history": loss_history,
            "confidence_loss_weight": conf_w,
            "candidate_loss_weight": cand_w,
            "confidence_mae_scale_ms": conf_mae_scale_ms,
            "confidence_target_mode": conf_target_mode,
            "confidence_improvement_margin_ms": conf_improvement_margin_ms,
            "confidence_improvement_scale_ms": conf_improvement_scale_ms,
            "balanced_sampling": bool(args.balanced_sampling),
            "sampling_balance_power": float(args.sampling_balance_power),
            "role_loss_balance_power": float(args.role_loss_balance_power),
            "format_loss_balance_power": float(args.format_loss_balance_power),
            "loss_balance_min_weight": float(args.loss_balance_min_weight),
            "loss_balance_max_weight": float(args.loss_balance_max_weight),
            "role_loss_weights": balance["role_weights_named"],
            "format_loss_weights": balance["format_weights_named"],
            "source_policy": "source_oto_params_not_used",
        },
    )
    torch.save(payload, str(out))
    print(json.dumps({"saved": str(out.resolve()), "numeric_dim": numeric_dim}, ensure_ascii=False, indent=2))
    return 0


def _build_sequences(
    *,
    dataset_root: str,
    banks: list[str],
    language: str,
    format_type: str,
    max_banks: int,
    max_wavs: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    bank_dirs = [Path(item) for item in banks if str(item or "").strip()]
    if not bank_dirs:
        bank_dirs = _discover_banks(Path(dataset_root), max_banks=max_banks)
    sequences: list[dict[str, Any]] = []
    wavs_seen = 0
    rows_seen = 0
    for bank in bank_dirs[:max_banks]:
        oto_path = bank / "oto.ini"
        if not oto_path.is_file():
            continue
        lang = _infer_language(bank, language)
        fmt = format_type or _infer_format(bank)
        targets = _load_gold_targets(str(oto_path), str(bank))
        row_specs_by_wav = load_row_specs_from_source_oto(
            source_oto_path=str(oto_path),
            wav_dir=str(bank),
            language=lang,
            format_type=fmt,
        )
        for wav_path, specs in sorted(row_specs_by_wav.items()):
            if wavs_seen >= max_wavs or rows_seen >= max_rows:
                return sequences
            if not specs:
                continue
            try:
                audio = compute_audio_candidates(wav_path)
                audio_cands = audio_candidates_to_boundary_candidates(audio)
                merged = merge_candidates(model_candidates=[], audio_candidates=audio_cands)
                decoded = decode_wav_rows(
                    wav_path=wav_path,
                    duration_ms=float(audio.duration_ms),
                    row_specs=specs,
                    candidates=merged,
                    active_start_ms=float(audio.active_start_ms),
                    active_end_ms=float(audio.active_end_ms),
                    model_quality=0.5,
                    audio_reliability=0.5,
                )
            except Exception as exc:
                print(json.dumps({"skip_wav": wav_path, "reason": str(exc)}, ensure_ascii=False))
                continue
            target_rows: list[tuple[float, float, float, float, float]] = []
            keep_indices: list[int] = []
            for idx, row in enumerate(decoded.rows):
                key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
                target = targets.get(key)
                if target is None:
                    continue
                duration = max(1.0, float(row.spec.duration_ms or audio.duration_ms or 0.0))
                target_rows.append(
                    (
                        float(target.offset_abs) / duration,
                        float(target.overlap_abs) / duration,
                        float(target.pre_abs) / duration,
                        float(target.consonant_abs) / duration,
                        float(target.cutoff_abs) / duration,
                    )
                )
                keep_indices.append(idx)
            if not keep_indices:
                continue
            # Keep the full row sequence for context; supervise all matched rows.
            feature_batch = build_stage2_feature_batch(
                decoded=decoded,
                candidates=merged,
                active_start_ms=float(audio.active_start_ms),
                active_end_ms=float(audio.active_end_ms),
                model_quality=0.5,
                audio_reliability=0.5,
            )
            supervised_target = []
            target_by_idx = {idx: target_rows[pos] for pos, idx in enumerate(keep_indices)}
            for idx, row in enumerate(decoded.rows):
                if idx in target_by_idx:
                    supervised_target.append(target_by_idx[idx])
                else:
                    duration = max(1.0, float(row.spec.duration_ms or audio.duration_ms or 0.0))
                    anchors = row.anchors
                    supervised_target.append(
                        (
                            float(anchors.offset_abs) / duration,
                            float(anchors.overlap_abs) / duration,
                            float(anchors.pre_abs) / duration,
                            float(anchors.consonant_abs) / duration,
                            float(anchors.cutoff_abs) / duration,
                        )
                    )
            seq_role_id, seq_format_id = _dominant_role_format_ids(feature_batch)
            sequences.append(
                {
                    "features": feature_batch,
                    "target": supervised_target,
                    "seq_role_id": int(seq_role_id),
                    "seq_format_id": int(seq_format_id),
                }
            )
            wavs_seen += 1
            rows_seen += len(decoded.rows)
    return sequences


def _feature_tensors(torch, batch, *, device: str):
    return (
        torch.tensor([[row.numeric for row in batch.rows]], dtype=torch.float32, device=device),
        torch.tensor([[row.role_id for row in batch.rows]], dtype=torch.long, device=device),
        torch.tensor([[row.prev_role_id for row in batch.rows]], dtype=torch.long, device=device),
        torch.tensor([[row.next_role_id for row in batch.rows]], dtype=torch.long, device=device),
        torch.tensor([[row.alias_type_id for row in batch.rows]], dtype=torch.long, device=device),
        torch.tensor([[row.language_id for row in batch.rows]], dtype=torch.long, device=device),
        torch.tensor([[row.format_id for row in batch.rows]], dtype=torch.long, device=device),
    )


def _split_sequences(
    sequences: list[dict[str, Any]],
    *,
    val_ratio: float,
    min_val_sequences: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(sequences)
    if len(items) <= 1 or val_ratio <= 0.0:
        return items, []
    idx = list(range(len(items)))
    rng = random.Random(int(seed) ^ 0xA5A5)
    rng.shuffle(idx)
    val_count = int(round(len(items) * max(0.0, min(0.5, float(val_ratio)))))
    if len(items) >= max(2, int(min_val_sequences)):
        val_count = max(1, val_count)
    val_count = max(0, min(val_count, len(items) - 1))
    if val_count <= 0:
        return items, []
    val_ids = set(idx[:val_count])
    train = [item for i, item in enumerate(items) if i not in val_ids]
    val = [item for i, item in enumerate(items) if i in val_ids]
    if not train:
        return items, []
    return train, val


def _run_epoch(
    *,
    torch,
    model,
    sequences: list[dict[str, Any]],
    device: str,
    optim,
    training: bool,
    conf_weight: float,
    cand_weight: float,
    conf_mae_scale_ms: float,
    conf_target_mode: str,
    conf_improvement_margin_ms: float,
    conf_improvement_scale_ms: float,
    role_loss_weights: dict[int, float],
    format_loss_weights: dict[int, float],
    apply_loss_balancing: bool,
) -> dict[str, float]:
    if not sequences:
        return {"loss": 0.0, "anchor_loss": 0.0, "conf_loss": 0.0, "candidate_loss": 0.0}
    model.train(mode=bool(training))
    total = 0.0
    total_anchor = 0.0
    total_conf = 0.0
    total_cand = 0.0
    with torch.set_grad_enabled(bool(training)):
        for item in sequences:
            batch = item["features"]
            target = item["target"]
            numeric, role_id, prev_role_id, next_role_id, alias_type_id, language_id, format_id = _feature_tensors(
                torch,
                batch,
                device=device,
            )
            target_t = torch.tensor([target], dtype=torch.float32, device=device)
            pred, conf = model(
                numeric,
                role_id,
                prev_role_id,
                next_role_id,
                alias_type_id,
                language_id,
                format_id,
            )
            row_weights = _row_loss_weights(
                torch,
                batch=batch,
                device=device,
                role_loss_weights=role_loss_weights if apply_loss_balancing else {},
                format_loss_weights=format_loss_weights if apply_loss_balancing else {},
            )
            anchor_per_field = torch.nn.functional.smooth_l1_loss(pred, target_t, beta=0.02, reduction="none")
            anchor_per_row = anchor_per_field.mean(dim=-1)
            anchor_loss = _weighted_row_mean(anchor_per_row, row_weights)
            conf_target = _confidence_target(
                torch,
                pred=pred,
                target_t=target_t,
                numeric=numeric,
                mode=conf_target_mode,
                error_scale_ms=conf_mae_scale_ms,
                improvement_margin_ms=conf_improvement_margin_ms,
                improvement_scale_ms=conf_improvement_scale_ms,
            )
            conf_per_row = torch.nn.functional.binary_cross_entropy(conf, conf_target, reduction="none")
            conf_loss = _weighted_row_mean(conf_per_row, row_weights)
            cand_loss = _candidate_anchor_loss(torch, pred=pred, numeric=numeric, row_weights=row_weights)
            loss = anchor_loss + (float(conf_weight) * conf_loss) + (float(cand_weight) * cand_loss)
            if training:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optim.step()
            total += float(loss.detach().cpu())
            total_anchor += float(anchor_loss.detach().cpu())
            total_conf += float(conf_loss.detach().cpu())
            total_cand += float(cand_loss.detach().cpu())
    denom = float(max(1, len(sequences)))
    return {
        "loss": total / denom,
        "anchor_loss": total_anchor / denom,
        "conf_loss": total_conf / denom,
        "candidate_loss": total_cand / denom,
    }


def _confidence_target(
    torch,
    *,
    pred,
    target_t,
    numeric,
    mode: str,
    error_scale_ms: float,
    improvement_margin_ms: float,
    improvement_scale_ms: float,
):
    if str(mode or "").strip().lower() == "error":
        return _confidence_target_from_error(
            torch,
            pred=pred,
            target_t=target_t,
            numeric=numeric,
            scale_ms=error_scale_ms,
        )
    return _confidence_target_from_improvement(
        torch,
        pred=pred,
        target_t=target_t,
        numeric=numeric,
        margin_ms=improvement_margin_ms,
        scale_ms=improvement_scale_ms,
    )


def _confidence_target_from_error(torch, *, pred, target_t, numeric, scale_ms: float):
    duration_ms = torch.clamp(numeric[..., 0], min=1.0e-4) * 10000.0
    err_ms = torch.abs(pred - target_t) * duration_ms.unsqueeze(-1)
    row_mae = err_ms.mean(dim=-1)
    target = torch.exp(-row_mae / max(1.0, float(scale_ms)))
    return target.detach().clamp(0.0, 1.0)


def _confidence_target_from_improvement(torch, *, pred, target_t, numeric, margin_ms: float, scale_ms: float):
    duration_ms = torch.clamp(numeric[..., 0], min=1.0e-4) * 10000.0
    base = numeric[..., 12:17]
    if int(base.shape[-1]) != len(ANCHOR_FIELDS):
        return _confidence_target_from_error(
            torch,
            pred=pred,
            target_t=target_t,
            numeric=numeric,
            scale_ms=max(1.0, float(scale_ms)),
        )
    base_err_ms = torch.abs(base - target_t) * duration_ms.unsqueeze(-1)
    pred_err_ms = torch.abs(pred - target_t) * duration_ms.unsqueeze(-1)
    improvement_ms = base_err_ms.mean(dim=-1) - pred_err_ms.mean(dim=-1)
    logits = (improvement_ms - float(margin_ms)) / max(1.0, float(scale_ms))
    return torch.sigmoid(logits).detach().clamp(0.0, 1.0)


def _candidate_anchor_loss(torch, *, pred, numeric, row_weights=None):
    field_count = len(ANCHOR_FIELDS)
    needed = _BASE_FEATURE_COUNT + (field_count * _CAND_BLOCK)
    if int(numeric.shape[-1]) < needed:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    cand_time_idx = [_BASE_FEATURE_COUNT + (i * _CAND_BLOCK) for i in range(field_count)]
    cand_missing_idx = [_BASE_FEATURE_COUNT + (i * _CAND_BLOCK) + 3 for i in range(field_count)]
    cand_time = torch.stack([numeric[..., idx] for idx in cand_time_idx], dim=-1).clamp(0.0, 1.0)
    cand_missing = torch.stack([numeric[..., idx] for idx in cand_missing_idx], dim=-1)
    mask = (cand_missing < 0.5).to(dtype=pred.dtype)
    loss = torch.nn.functional.smooth_l1_loss(pred, cand_time, beta=0.02, reduction="none")
    row_numer = (loss * mask).sum(dim=-1)
    row_denom = mask.sum(dim=-1).clamp(min=1.0)
    row_loss = row_numer / row_denom
    if row_weights is None:
        return row_loss.mean()
    return _weighted_row_mean(row_loss, row_weights)


def _row_loss_weights(
    torch,
    *,
    batch,
    device: str,
    role_loss_weights: dict[int, float],
    format_loss_weights: dict[int, float],
):
    weights = []
    for row in batch.rows:
        role_w = float(role_loss_weights.get(int(row.role_id), 1.0))
        format_w = float(format_loss_weights.get(int(row.format_id), 1.0))
        weights.append(max(0.05, role_w * format_w))
    if not weights:
        return torch.ones((1, 0), dtype=torch.float32, device=device)
    return torch.tensor([weights], dtype=torch.float32, device=device)


def _weighted_row_mean(values, weights):
    numer = (values * weights).sum()
    denom = weights.sum().clamp(min=1.0e-6)
    return numer / denom


def _sample_sequences_for_epoch(
    sequences: list[dict[str, Any]],
    *,
    sequence_weights: list[float],
    balanced_sampling: bool,
) -> list[dict[str, Any]]:
    if not sequences:
        return []
    if bool(balanced_sampling) and len(sequence_weights) == len(sequences):
        return random.choices(sequences, weights=sequence_weights, k=len(sequences))
    copied = list(sequences)
    random.shuffle(copied)
    return copied


def _build_balance_plan(
    train_sequences: list[dict[str, Any]],
    *,
    sampling_power: float,
    sampling_max_weight: float,
    role_loss_power: float,
    format_loss_power: float,
    row_min_weight: float,
    row_max_weight: float,
) -> dict[str, Any]:
    role_counts: Counter[int] = Counter()
    format_counts: Counter[int] = Counter()
    seq_groups: list[tuple[int, int]] = []
    seq_group_counts: Counter[tuple[int, int]] = Counter()
    for item in train_sequences:
        batch = item["features"]
        seq_role = int(item.get("seq_role_id", 0) or 0)
        seq_format = int(item.get("seq_format_id", 0) or 0)
        seq_groups.append((seq_format, seq_role))
        seq_group_counts[(seq_format, seq_role)] += 1
        for row in batch.rows:
            role_counts[int(row.role_id)] += 1
            format_counts[int(row.format_id)] += 1
    group_weights = _inverse_frequency_weights(
        seq_group_counts,
        power=max(0.0, float(sampling_power)),
        min_weight=max(0.20, 1.0 / max(1.0, float(sampling_max_weight))),
        max_weight=max(1.0, float(sampling_max_weight)),
    )
    role_weights = _inverse_frequency_weights(
        role_counts,
        power=max(0.0, float(role_loss_power)),
        min_weight=max(0.05, float(row_min_weight)),
        max_weight=max(1.0, float(row_max_weight)),
    )
    format_weights = _inverse_frequency_weights(
        format_counts,
        power=max(0.0, float(format_loss_power)),
        min_weight=max(0.05, float(row_min_weight)),
        max_weight=max(1.0, float(row_max_weight)),
    )
    sequence_weights = [float(group_weights.get(key, 1.0)) for key in seq_groups]
    return {
        "sequence_weights": sequence_weights,
        "sampling_groups": {
            _format_role_key_name(key[0], key[1]): int(seq_group_counts.get(key, 0))
            for key in sorted(seq_group_counts.keys())
        },
        "role_counts_named": _id_counter_to_named(role_counts, ROLE_VOCAB),
        "format_counts_named": _id_counter_to_named(format_counts, FORMAT_VOCAB),
        "role_weights": role_weights,
        "format_weights": format_weights,
        "role_weights_named": _id_float_map_to_named(role_weights, ROLE_VOCAB),
        "format_weights_named": _id_float_map_to_named(format_weights, FORMAT_VOCAB),
    }


def _inverse_frequency_weights(
    counts: Counter,
    *,
    power: float,
    min_weight: float,
    max_weight: float,
) -> dict[Any, float]:
    if not counts:
        return {}
    if float(power) <= 0.0:
        return {key: 1.0 for key in counts.keys()}
    lo = max(0.01, float(min_weight))
    hi = max(lo, float(max_weight))
    totals = {key: max(1, int(value)) for key, value in counts.items()}
    total_count = float(sum(totals.values()))
    raw = {key: (total_count / float(val)) ** float(power) for key, val in totals.items()}
    mean = sum(raw[key] * float(totals[key]) for key in totals.keys()) / max(1.0, total_count)
    out: dict[Any, float] = {}
    for key in totals.keys():
        norm = raw[key] / max(1.0e-9, float(mean))
        out[key] = _clamp_float(norm, lo, hi)
    return out


def _dominant_role_format_ids(feature_batch) -> tuple[int, int]:
    role_counter: Counter[int] = Counter()
    format_counter: Counter[int] = Counter()
    for row in feature_batch.rows:
        role_counter[int(row.role_id)] += 1
        format_counter[int(row.format_id)] += 1
    role_id = _counter_mode(role_counter)
    format_id = _counter_mode(format_counter)
    return int(role_id), int(format_id)


def _counter_mode(counter: Counter[int]) -> int:
    if not counter:
        return 0
    return int(sorted(counter.items(), key=lambda item: (-int(item[1]), int(item[0])))[0][0])


def _id_counter_to_named(counter: Counter[int], vocab: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, count in sorted(counter.items()):
        if int(count) <= 0:
            continue
        name = vocab[int(idx)] if 0 <= int(idx) < len(vocab) else str(idx)
        out[str(name)] = int(count)
    return out


def _id_float_map_to_named(values: dict[int, float], vocab: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, val in sorted(values.items()):
        name = vocab[int(idx)] if 0 <= int(idx) < len(vocab) else str(idx)
        out[str(name)] = round(float(val), 6)
    return out


def _format_role_key_name(format_id: int, role_id: int) -> str:
    fmt = FORMAT_VOCAB[int(format_id)] if 0 <= int(format_id) < len(FORMAT_VOCAB) else str(format_id)
    role = ROLE_VOCAB[int(role_id)] if 0 <= int(role_id) < len(ROLE_VOCAB) else str(role_id)
    return f"{fmt}/{role}"


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _snapshot_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _discover_banks(root: Path, *, max_banks: int) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for oto in root.rglob("oto.ini"):
        if len(out) >= max_banks:
            break
        bank = oto.parent
        if any(bank.glob("*.wav")):
            out.append(bank)
    return out


def _load_gold_targets(oto_path: str, wav_dir: str):
    targets = {}
    text = read_text_with_fallback(oto_path)
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        wav_path = os.path.join(wav_dir, wav_name)
        duration = _wav_duration_ms(wav_path)
        anchors = oto_row_to_absolute_anchors(
            offset=float(parsed.get("offset", 0.0) or 0.0),
            consonant=float(parsed.get("consonant", 0.0) or 0.0),
            cutoff=float(parsed.get("cutoff", 0.0) or 0.0),
            preutterance=float(parsed.get("preutterance", 0.0) or 0.0),
            overlap=float(parsed.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
        targets[(wav_name, str(parsed.get("alias", "") or ""), int(line_idx))] = anchors
    return targets


def _wav_duration_ms(path: str) -> float:
    import wave

    if not path or not os.path.isfile(path):
        return 1.0
    with wave.open(path, "rb") as wf:
        frames = int(wf.getnframes() or 0)
        sr = int(wf.getframerate() or 0)
    return float(frames) * 1000.0 / float(sr) if frames > 0 and sr > 0 else 1.0


def _infer_language(path: Path, fallback: str) -> str:
    if fallback:
        return fallback
    parts = {part.lower() for part in path.parts}
    if "japanese" in parts:
        return "japanese"
    if "korean" in parts:
        return "korean"
    return "korean"


def _infer_format(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for item in reversed(parts):
        if item in {"cv", "cvc", "cvvc", "vcv", "cmpx"}:
            return item
    return "other"


def _resolve_device(torch, value: str) -> str:
    text = str(value or "auto").strip().lower()
    if text == "cuda" and torch.cuda.is_available():
        return "cuda"
    if text == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
