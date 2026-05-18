"""Distillation trainer for the boundary scorer student (conv_only arch).

Status: Phase 2 (CRNN -> conv_only self-distillation) working.

Two teacher backends:

1. `boundary_ckpt` (working): loads any existing boundary scorer checkpoint
   (e.g. v5_codabridge or phone_aux_warm) and uses its soft logits as the
   distillation target. Validates the conv_only student architecture and the
   distillation loop itself.

2. `ssl` (stub, raises NotImplementedError): wav2vec2 / HuBERT teacher. Full
   design in `14-8-SSL-백본-이식-계획서.md`.

Loss composition (per logits group):

    L_total = (1 - alpha) * L_hard + alpha * L_kd

Where L_hard is the standard boundary BCE + boundary_time regression +
phone_aux CE (reused verbatim from `boundary_scorer_training`), and L_kd is
the per-head KL divergence between student and teacher logits, temperature-
scaled and masked to valid frames only.

The script is a thin orchestrator: it reuses `_BoundaryDataset`, `_collate`,
`_unpack_batch`, `_move_phone_aux`, `_phone_aux_loss`,
`_boundary_time_regression_loss`, `_load_matching_state_from_checkpoint`,
`_freeze_non_phone_aux_params` from `boundary_scorer_training` so the
hard-loss path stays identical to the regular training script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import numpy as np

from core.coarse_crnn.boundary_scorer_model import (
    BoundaryScorerConfig,
    build_boundary_scorer,
    load_boundary_checkpoint,
    save_boundary_checkpoint,
)
from core.coarse_crnn.boundary_scorer_training import (
    BoundaryTrainConfig,
    _BoundaryDataset,
    _boundary_time_regression_loss,
    _collate,
    _load_matching_state_from_checkpoint,
    _move_phone_aux,
    _phone_aux_loss,
    _unpack_batch,
)
from core.coarse_crnn.boundary_targets import training_rows_to_wav_groups
from core.coarse_crnn.training import resolve_torch_device


# Heads that participate in distillation. Boundary is BCE (per-class
# independent sigmoid), the rest are softmax CE.
_SIGMOID_LOGIT_KEYS = ("boundary_logits",)
_SOFTMAX_LOGIT_KEYS = (
    "cvs_logits",
    "consonant_logits",
    "vowel_logits",
    "consonant_family_logits",
    "vowel_nucleus_logits",
    "vowel_glide_logits",
)


class _Teacher:
    def to(self, device):
        return self

    def eval(self):
        return self

    def forward_logits(self, mel_input):
        raise NotImplementedError


class BoundaryCheckpointTeacher(_Teacher):
    """Teacher = any existing boundary scorer checkpoint (crnn or conv_only)."""

    def __init__(self, checkpoint_path: str, device):
        import torch

        self._torch = torch
        self.model, self.config, self.meta = load_boundary_checkpoint(
            checkpoint_path, map_location=str(device)
        )
        self.model = self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def forward_logits(self, mel_input):
        with self._torch.no_grad():
            return self.model(mel_input)


class SSLBackboneTeacher(_Teacher):
    """SSL backbone teacher (wav2vec2 / HuBERT family). Not wired yet."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "SSL teacher is not wired yet. See 14-8-SSL-백본-이식-계획서.md."
        )


def _resolve_teacher(args, device) -> _Teacher:
    kind = str(args.teacher).strip().lower()
    if kind == "boundary_ckpt":
        if not str(args.teacher_path or "").strip():
            raise SystemExit("--teacher boundary_ckpt requires --teacher-path")
        return BoundaryCheckpointTeacher(str(args.teacher_path), device=device)
    if kind == "ssl":
        return SSLBackboneTeacher(
            ssl_model=str(args.ssl_model or ""),
            head_checkpoint=str(args.ssl_head_path or ""),
            device=device,
        )
    raise SystemExit(f"unknown --teacher: {kind!r}")


def _distill_loss(student_out, teacher_out, *, mask, alpha: float, temperature: float):
    """Returns alpha * sum_of_per_head_KL, or None if no head pair is active.

    Per-head KL is masked by `mask` (frame validity) and scaled by `tau^2` per
    the standard Hinton distillation formulation. Computed in fp32 regardless
    of autocast context: under AMP+T=1.0 the teacher sigmoid/softmax can
    saturate in fp16 (probabilities exactly 0 or 1), producing 0*(-inf) NaNs
    in the entropy terms.
    """
    import torch
    import torch.nn.functional as F

    total = None
    tau = max(1e-3, float(temperature))
    frame_mask_2d = mask.float()
    denom_2d = torch.clamp(frame_mask_2d.sum(), min=1.0)

    for key in _SIGMOID_LOGIT_KEYS:
        s_logits = student_out.get(key)
        t_logits = teacher_out.get(key)
        if s_logits is None or t_logits is None:
            continue
        s_f = s_logits.float()
        t_f = t_logits.float()
        s_log_p = F.logsigmoid(s_f / tau)
        s_log_1mp = F.logsigmoid(-s_f / tau)
        t_p = torch.sigmoid(t_f / tau).detach().clamp(min=1e-6, max=1.0 - 1e-6)
        t_log = torch.log(t_p)
        t_log_1m = torch.log1p(-t_p)
        kl = t_p * (t_log - s_log_p) + (1.0 - t_p) * (t_log_1m - s_log_1mp)
        per_frame = kl.mean(dim=-1)
        loss = (per_frame * frame_mask_2d).sum() / denom_2d * (tau * tau)
        total = loss if total is None else total + loss

    for key in _SOFTMAX_LOGIT_KEYS:
        s_logits = student_out.get(key)
        t_logits = teacher_out.get(key)
        if s_logits is None or t_logits is None:
            continue
        s_f = s_logits.float()
        t_f = t_logits.float()
        s_log_p = F.log_softmax(s_f / tau, dim=-1)
        t_p = F.softmax(t_f / tau, dim=-1).detach().clamp(min=1e-6)
        # Renormalize after clamp so the row sums stay ~1 (loose; the bias
        # is bounded by class_count * 1e-6).
        t_p = t_p / t_p.sum(dim=-1, keepdim=True)
        t_log = torch.log(t_p)
        kl_per_frame = (t_p * (t_log - s_log_p)).sum(dim=-1)
        loss = (kl_per_frame * frame_mask_2d).sum() / denom_2d * (tau * tau)
        total = loss if total is None else total + loss

    if total is None:
        return None
    return total * float(alpha)


def _compute_hard_loss(out, *, y, mask, phone_aux, cfg: BoundaryTrainConfig, pos_weight, hop_ms: float):
    """Reuses the exact loss composition from boundary_scorer_training."""
    import torch
    import torch.nn.functional as F

    logits = out["boundary_logits"]
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    denom = torch.clamp(mask.sum() * y.shape[-1], min=1.0)
    loss = (raw * mask[:, :, None]).sum() / denom
    q_logits = out.get("quality_logits")
    if q_logits is not None and float(cfg.quality_loss_weight) > 0.0:
        q_target = torch.clamp(y.max(dim=2).values, 0.0, 1.0)
        q_raw = F.binary_cross_entropy_with_logits(q_logits, q_target, reduction="none")
        q_loss = (q_raw * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        loss = loss + (q_loss * float(cfg.quality_loss_weight))
    if float(cfg.boundary_time_loss_weight) > 0.0:
        t_loss = _boundary_time_regression_loss(logits, y, mask, hop_ms=float(hop_ms))
        loss = loss + (t_loss * float(cfg.boundary_time_loss_weight))
    aux = _phone_aux_loss(out, phone_aux, mask=mask, cfg=cfg)
    if aux is not None:
        loss = loss + aux
    return loss


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    ap = argparse.ArgumentParser(
        description=(
            "Knowledge-distillation trainer for the conv_only boundary scorer "
            "student. Working teacher: boundary_ckpt. SSL teacher is a stub."
        )
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--teacher",
        choices=["boundary_ckpt", "ssl"],
        default="boundary_ckpt",
    )
    ap.add_argument("--teacher-path", default="")
    ap.add_argument("--ssl-model", default="")
    ap.add_argument("--ssl-head-path", default="")
    ap.add_argument("--alpha", type=float, default=0.7, help="Weight on the KL distillation term. (1-alpha) on hard loss.")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=8)
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
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-wavs", type=int, default=0, help="Limit training wav count (smoke).")
    # Phone aux head enable + loss weights (mirror train_boundary_oto.py defaults).
    ap.add_argument("--enable-phone-aux-heads", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--enable-phone-family-heads", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cvs-loss-weight", type=float, default=0.08)
    ap.add_argument("--consonant-id-loss-weight", type=float, default=0.05)
    ap.add_argument("--vowel-id-loss-weight", type=float, default=0.05)
    ap.add_argument("--consonant-family-loss-weight", type=float, default=0.0)
    ap.add_argument("--vowel-nucleus-loss-weight", type=float, default=0.0)
    ap.add_argument("--vowel-glide-loss-weight", type=float, default=0.0)
    ap.add_argument("--phone-aux-class-balanced", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--phone-aux-class-balance-power", type=float, default=0.5)
    ap.add_argument("--phone-aux-class-balance-max", type=float, default=8.0)
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--quality-loss-weight", type=float, default=0.06)
    ap.add_argument("--boundary-time-loss-weight", type=float, default=0.20)
    ap.add_argument("--hard-case-oversample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hard-case-weight", type=float, default=2.5)
    ap.add_argument("--hard-case-min-ratio", type=float, default=0.10)
    ap.add_argument(
        "--hard-case-alias-regex",
        default=r"^(N|M|L|NG|H|T|P|K|R|n|m|l|ng)\s+\S",
    )
    ap.add_argument("--n-like-weight", type=float, default=0.80)
    ap.add_argument("--weak-wav-weight", type=float, default=0.65)
    ap.add_argument("--weak-wav-rms-db-threshold", type=float, default=-32.0)
    ap.add_argument("--weak-wav-auto-percentile", type=float, default=0.15)
    ap.add_argument("--init-from", default="", help="Warm-start matching weights from a student checkpoint.")
    args = ap.parse_args()

    import torch

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_torch_device(torch, str(args.device))

    # Student config: conv_only encoder, head layout matching the teacher when
    # possible. Phone aux heads are enabled by default so that KD can supervise
    # them even if the teacher only has them via the head dict.
    student_cfg = BoundaryScorerConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        enable_phone_aux_heads=bool(args.enable_phone_aux_heads),
        enable_phone_family_heads=bool(args.enable_phone_family_heads),
        arch_type="conv_only",
    )

    # BoundaryTrainConfig drives the hard-loss reuse (mirrors train_boundary_oto).
    tcfg = BoundaryTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        seed=int(args.seed),
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
        freeze_non_phone_aux=False,  # student trained end-to-end during distillation
    )

    # Read manifest, group, split.
    rows = _read_jsonl(str(args.manifest))
    grouped = training_rows_to_wav_groups(rows)
    if not grouped:
        raise SystemExit(f"no usable rows in manifest: {args.manifest}")
    keys = list(grouped.keys())
    random.shuffle(keys)
    if int(args.max_wavs) > 0:
        keys = keys[: int(args.max_wavs)]
    val_count = max(1, int(round(len(keys) * float(tcfg.val_ratio)))) if len(keys) >= 10 else 0
    val_keys = set(keys[:val_count])
    train_groups = {k: grouped[k] for k in keys if k not in val_keys}
    val_groups = {k: grouped[k] for k in keys if k in val_keys}
    if not train_groups:
        train_groups = grouped
        val_groups = {}

    train_ds = _BoundaryDataset(
        train_groups,
        student_cfg,
        max_frames=int(tcfg.max_frames),
        train=True,
        hard_case_weight=float(tcfg.hard_case_weight),
        hard_case_min_ratio=float(tcfg.hard_case_min_ratio),
        hard_case_alias_regex=str(tcfg.hard_case_alias_regex or ""),
        n_like_weight=float(tcfg.n_like_weight),
        weak_wav_weight=float(tcfg.weak_wav_weight),
        weak_wav_rms_db_threshold=float(tcfg.weak_wav_rms_db_threshold),
        weak_wav_auto_percentile=float(tcfg.weak_wav_auto_percentile),
    )
    val_ds = (
        _BoundaryDataset(val_groups, student_cfg, max_frames=int(tcfg.max_frames), train=False)
        if val_groups
        else None
    )

    sampler = None
    shuffle = True
    if bool(tcfg.hard_case_oversample) and len(train_ds) > 1:
        weights = train_ds.sample_weights
        if any(float(w) > 1.0001 for w in weights):
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(tcfg.batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        collate_fn=_collate,
        num_workers=max(0, int(tcfg.num_workers)),
        pin_memory=True,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=max(0, int(tcfg.num_workers)),
            pin_memory=True,
        )
        if val_ds
        else None
    )

    # Student model + optional warm-start.
    student = build_boundary_scorer(student_cfg).to(device)
    init_report = (
        _load_matching_state_from_checkpoint(student, str(tcfg.init_from or "").strip())
        if str(tcfg.init_from or "").strip()
        else {}
    )

    teacher = _resolve_teacher(args, device=device)

    use_amp = bool(tcfg.amp and device.type == "cuda")
    pos_weight = torch.tensor(float(tcfg.pos_weight), device=device)
    trainable_params = [p for p in student.parameters() if bool(getattr(p, "requires_grad", True))]
    if not trainable_params:
        raise SystemExit("student has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(tcfg.lr), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, float]] = []
    best_val = None
    best_state = None
    hop_ms = float(student_cfg.hop_ms)

    print(
        json.dumps(
            {
                "phase": "start",
                "student_params": int(sum(p.numel() for p in student.parameters())),
                "train_wavs": int(len(train_groups)),
                "val_wavs": int(len(val_groups)),
                "teacher": str(args.teacher),
                "teacher_path": str(args.teacher_path or ""),
                "alpha": float(args.alpha),
                "temperature": float(args.temperature),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, int(tcfg.epochs) + 1):
        student.train()
        loss_sum = 0.0
        hard_sum = 0.0
        kd_sum = 0.0
        weight_sum = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y, mask, phone_aux = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            phone_aux = _move_phone_aux(phone_aux, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                s_out = student(x)
                # Teacher forward outside autocast region is fine because the
                # teacher already runs in eval/no_grad and we only need its
                # logits as detached targets.
                t_out = teacher.forward_logits(x)
                hard_loss = _compute_hard_loss(
                    s_out,
                    y=y,
                    mask=mask,
                    phone_aux=phone_aux,
                    cfg=tcfg,
                    pos_weight=pos_weight,
                    hop_ms=hop_ms,
                )
                kd_loss = _distill_loss(
                    s_out,
                    t_out,
                    mask=mask,
                    alpha=float(args.alpha),
                    temperature=float(args.temperature),
                )
                total_loss = (1.0 - float(args.alpha)) * hard_loss
                if kd_loss is not None:
                    total_loss = total_loss + kd_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            frames = float(mask.sum().detach().cpu().item())
            w = max(1.0, frames)
            loss_sum += float(total_loss.detach().cpu().item()) * w
            hard_sum += float(hard_loss.detach().cpu().item()) * w
            if kd_loss is not None:
                kd_sum += float(kd_loss.detach().cpu().item()) * w
            weight_sum += w
            if int(tcfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(tcfg.log_every) == 0):
                print(
                    f"[distill][train] epoch={epoch}/{int(tcfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(total_loss.detach().cpu().item()):.4f} "
                    f"hard={float(hard_loss.detach().cpu().item()):.4f} "
                    f"kd={float(kd_loss.detach().cpu().item()) if kd_loss is not None else 0.0:.4f}",
                    flush=True,
                )

        row = {
            "epoch": float(epoch),
            "train_loss": float(loss_sum / max(1.0, weight_sum)),
            "train_hard": float(hard_sum / max(1.0, weight_sum)),
            "train_kd": float(kd_sum / max(1.0, weight_sum)),
        }
        if val_loader is not None:
            student.eval()
            v_sum = 0.0
            v_w = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, mask, phone_aux = _unpack_batch(batch)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    phone_aux = _move_phone_aux(phone_aux, device=device)
                    s_out = student(x)
                    hard = _compute_hard_loss(
                        s_out,
                        y=y,
                        mask=mask,
                        phone_aux=phone_aux,
                        cfg=tcfg,
                        pos_weight=pos_weight,
                        hop_ms=hop_ms,
                    )
                    frames = float(mask.sum().detach().cpu().item())
                    w = max(1.0, frames)
                    v_sum += float(hard.detach().cpu().item()) * w
                    v_w += w
            val_loss = float(v_sum / max(1.0, v_w))
            row["val_loss"] = val_loss
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        history.append(row)
        print(f"[distill][epoch] {json.dumps(row, ensure_ascii=False)}", flush=True)

    if best_state is not None:
        student.load_state_dict(best_state)

    out_path = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = {
        "history": history,
        "train_wavs": int(len(train_groups)),
        "val_wavs": int(len(val_groups)),
        "device": str(device),
        "distillation": {
            "teacher_kind": str(args.teacher),
            "teacher_path": str(args.teacher_path or ""),
            "alpha": float(args.alpha),
            "temperature": float(args.temperature),
            "logit_kd_keys": list(_SIGMOID_LOGIT_KEYS + _SOFTMAX_LOGIT_KEYS),
        },
        "phone_aux_enabled": bool(student_cfg.enable_phone_aux_heads),
        "phone_family_enabled": bool(student_cfg.enable_phone_family_heads),
        "cvs_loss_weight": float(tcfg.cvs_loss_weight),
        "consonant_id_loss_weight": float(tcfg.consonant_id_loss_weight),
        "vowel_id_loss_weight": float(tcfg.vowel_id_loss_weight),
        "consonant_family_loss_weight": float(tcfg.consonant_family_loss_weight),
        "vowel_nucleus_loss_weight": float(tcfg.vowel_nucleus_loss_weight),
        "vowel_glide_loss_weight": float(tcfg.vowel_glide_loss_weight),
        "phone_aux_class_balanced": bool(tcfg.phone_aux_class_balanced),
        "boundary_time_loss_weight": float(tcfg.boundary_time_loss_weight),
        "hard_case_oversample": bool(tcfg.hard_case_oversample),
        "hard_case_weight": float(tcfg.hard_case_weight),
        "init_from": str(tcfg.init_from or ""),
        "init_report": dict(init_report or {}),
    }
    save_boundary_checkpoint(out_path, student.cpu(), student_cfg, meta=meta)
    print(json.dumps({"phase": "done", "output_path": out_path, "best_val": best_val, "history": history}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
