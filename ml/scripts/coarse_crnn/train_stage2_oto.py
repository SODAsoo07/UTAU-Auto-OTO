from __future__ import annotations

import argparse
import json
import os
import random
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
from core.coarse_crnn.stage2_oto.types import Stage2CheckpointMeta, Stage2ModelConfig
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


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
    numeric_dim = len(sequences[0]["features"].rows[0].numeric)
    config = Stage2ModelConfig(numeric_dim=numeric_dim, hidden_dim=int(args.hidden_dim))
    model = build_stage2_model(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    loss_history: list[float] = []
    for epoch in range(max(1, int(args.epochs))):
        random.shuffle(sequences)
        losses: list[float] = []
        model.train()
        for item in sequences:
            batch = item["features"]
            target = item["target"]
            numeric, role_id, prev_role_id, next_role_id, alias_type_id, language_id, format_id = _feature_tensors(
                torch,
                batch,
                device=device,
            )
            target_t = torch.tensor([target], dtype=torch.float32, device=device)
            pred, conf = model(numeric, role_id, prev_role_id, next_role_id, alias_type_id, language_id, format_id)
            mse = torch.nn.functional.smooth_l1_loss(pred, target_t, beta=0.02)
            conf_target = torch.ones_like(conf)
            conf_loss = torch.nn.functional.binary_cross_entropy(conf, conf_target)
            loss = mse + 0.03 * conf_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        loss_history.append(mean_loss)
        print(json.dumps({"epoch": epoch + 1, "loss": mean_loss, "sequences": len(sequences)}, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model=model.cpu().eval(),
        config=config,
        meta=Stage2CheckpointMeta(),
        extra={
            "train_sequences": len(sequences),
            "train_rows": sum(len(item["features"].rows) for item in sequences),
            "loss_history": loss_history,
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
            sequences.append({"features": feature_batch, "target": supervised_target})
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
