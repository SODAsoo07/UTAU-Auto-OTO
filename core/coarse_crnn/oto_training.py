from __future__ import annotations

import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any
import hashlib
import json

import numpy as np

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.oto_model import (
    OtoCrnnConfig,
    alias_role_id,
    alias_type_id,
    build_oto_model,
    format_id,
    language_id,
    save_oto_checkpoint,
    transition_type_id,
    uses_relative_param_head,
    right_boundary_prior_blend_for_context,
)
from core.coarse_crnn.oto_ctc import phone_ids_from_alias
from core.coarse_crnn.oto_param_priors import decode_relative_oto_params, normalize_relative_oto_target, relative_params_to_anchors
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES, extract_alias_features
from core.coarse_crnn.oto_windowing import crop_oto_target_window, row_window_args, should_use_vcv_target_window, target_window_frames_for
from core.coarse_crnn.training import _autocast, _make_grad_scaler, _pin_memory_enabled, resolve_torch_device


@dataclass
class OtoTrainConfig:
    epochs: int = 4
    lr: float = 1e-3
    batch_size: int = 8
    max_frames: int = 1200
    seed: int = 1337
    device: str = "auto"
    amp: bool = True
    # num_workers <= 0 means auto-resolve from CPU cores.
    num_workers: int = 0
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    # Number of batches to prefetch to CUDA with a dedicated stream (H2D overlap).
    # 0 disables CUDA-side prefetch.
    cuda_prefetch_batches: int = 0
    # Apply the same CUDA prefetching in validation/evaluation.
    cuda_prefetch_eval: bool = True
    enable_tf32: bool = False
    enable_cudnn_benchmark: bool = False
    enable_feature_cache: bool = True
    feature_cache_dir: str = os.path.join("ml_workspace", "coarse_crnn", "feature_cache")
    feature_cache_readonly: bool = False
    log_every: int = 100
    val_ratio: float = 0.08
    heatmap_sigma_frames: float = 2.0
    heatmap_loss_weight: float = 1.0
    scalar_loss_weight: float = 0.55
    order_loss_weight: float = 0.08
    vcv_loss_weight: float = 1.35
    cvvc_loss_weight: float = 1.15
    cvc_loss_weight: float = 1.05
    # Role-based loss weights. vc and vv are the hardest to predict
    # (vc in cvvc voicebanks has the highest MAE) so they get upweighted.
    vc_role_loss_weight: float = 2.5
    vv_role_loss_weight: float = 2.0
    # v2 1-B: row-order violation row의 학습 weight 하향. SONG_CMYK_CVVC에서
    # row_index 순서와 실제 target_offset 순서가 어긋난 row(전체의 32.8%)는
    # row_ratio feature를 noise로 만든다. manifest의 ``row_order_violation_score``
    # 와 결합해 ``weight *= 1.0 / (1.0 + alpha * clip(score, 0, 3))`` 로
    # 다운스케일. alpha=0.0이면 비활성화 (legacy 호환). v2 권장 시작값 0.5.
    row_order_violation_alpha: float = 0.0
    # CTC alignment auxiliary loss weight (3-A). 0.0 disables the loss even
    # if the model has the CTC head enabled, which is useful for ablation.
    ctc_loss_weight: float = 0.0
    # CTC head warmup. During the first N global training steps the model
    # backpropagates ONLY the CTC alignment loss; anchor regression / heatmap
    # / confidence losses are skipped. This lets a freshly initialized
    # ctc_head stabilize before its random gradients can knock the
    # pretrained anchor heads off-distribution. 0 disables warmup.
    ctc_warmup_steps: int = 0
    uncertainty_loss_weight: float = 0.30
    confidence_loss_weight: float = 0.10
    confidence_target_error_scale: float = 0.08
    min_confidence_target: float = 0.02
    max_confidence_target: float = 0.98
    enable_balanced_sampling: bool = True
    balanced_sampling_replacement: bool = True
    balanced_sampling_size_factor: float = 1.0
    voicebank_balance_power: float = 0.55
    # Explicit language balancing is separate from voicebank balancing. This
    # keeps Korean/Japanese CVVC from being hidden behind voicebank counts.
    language_balance_power: float = 0.25
    # Keep role upweighting, but avoid drowning out format identity on OOD banks.
    role_balance_power: float = 0.30
    format_balance_power: float = 0.35
    # Optional slash-pattern boosts: "korean/cvvc/*=1.8,korean/cvvc/vc=2.2".
    language_format_role_sampling_boosts: tuple[str, ...] = ()
    # Direct sampling boost for sparse/high-error CVVC contexts.
    # `cvvc|vc|*` = cvvc format with vc alias-role or vc alias-type.
    cvvc_vc_sampling_boost: float = 1.0
    cvvc_vv_sampling_boost: float = 1.0
    # Extra narrow boost: only cvvc|vc|multi rows.
    cvvc_vc_multi_sampling_boost: float = 1.0
    enable_hard_case_mining: bool = True
    hard_case_top_ratio: float = 0.25
    hard_case_boost: float = 2.5
    early_stop_patience: int = 0
    # OOD-first model selection: prioritize worst voicebank and hard-failure tail
    # over raw val_loss.
    selection_val_loss_weight: float = 0.15
    selection_anchor_mae_weight: float = 0.90
    selection_preutterance_gap_weight: float = 2.0
    selection_hard_failure_weight: float = 2.5
    selection_worst_voicebank_weight: float = 1.5
    selection_worst_voicebank_hard_failure_weight: float = 2.0
    selection_preutterance_target_acc50: float = 0.55
    selection_worst_voicebank_target_acc50: float = 0.50
    # Data-quality filter for strong leading-silence / weak-activity OOD cases.
    enable_activity_quality_filter: bool = True
    activity_leading_silence_max_ratio: float = 0.42
    activity_min_active_span_ratio: float = 0.16
    activity_quality_outside_penalty: float = 0.35
    activity_quality_min_multiplier: float = 0.10
    activity_quality_drop_low_weight: bool = False
    activity_quality_drop_threshold: float = 0.08
    activity_quality_max_drop_ratio: float = 0.35
    activity_quality_sample_rate: int = 16000
    # Confidence calibration derived from validation predictions.
    enable_confidence_calibration: bool = True
    confidence_calibration_good_error_ms: float = 50.0
    confidence_calibration_low_conf_target: float = 0.58
    confidence_calibration_error_gate_ms: float = 80.0
    checkpoint_save_every_epochs: int = 0


class OtoAnchorDataset:
    def __init__(self, rows: list[dict[str, Any]], model_config: OtoCrnnConfig, *, train_config: OtoTrainConfig, train: bool):
        self.rows = list(rows or [])
        self.model_config = model_config
        self.train_config = train_config
        self.train = bool(train)
        self.feature_cache_enabled = bool(getattr(train_config, "enable_feature_cache", False))
        self.feature_cache_readonly = bool(getattr(train_config, "feature_cache_readonly", False))
        self.feature_cache_dir = os.path.abspath(str(getattr(train_config, "feature_cache_dir", "")))
        if self.feature_cache_enabled and self.feature_cache_dir and (not self.feature_cache_readonly):
            os.makedirs(self.feature_cache_dir, exist_ok=True)
        self.voicebank_vocab = sorted(
            {str(row.get("voicebank_id", "") or "unknown_voicebank") for row in self.rows}
        )
        self.voicebank_to_id = {name: idx for idx, name in enumerate(self.voicebank_vocab)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        wav_path = str(row.get("audio", "") or "")
        alias_text = str(row.get("alias", "") or "")
        language_text = str(row.get("language", "") or "")
        alias_features = extract_alias_features(alias_text, language=language_text)
        alias_role_text = str(row.get("alias_role", "") or "")
        manifest_has_role = row.get("alias_role") is not None
        if not alias_role_text:
            alias_role_text = str(alias_features.get("alias_role", "") or "")
        if manifest_has_role:
            is_diphthong = bool(float(row.get("is_diphthong", 0.0) or 0.0) >= 0.5)
            is_special = bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5)
        else:
            is_diphthong = bool(float(alias_features.get("is_diphthong", 0.0) or 0.0) >= 0.5)
            is_special = bool(float(alias_features.get("is_special", 0.0) or 0.0) >= 0.5)
        if normalize_role(alias_role_text) == "special":
            is_special = True
        features, hop_sec, duration_sec = self._load_or_compute_features(wav_path)
        if features.shape[0] <= 0:
            features = np.zeros((1, int(self.model_config.n_mels)), dtype=np.float32)
        anchors_ms = _anchor_array_from_row(row)
        full_duration_ms = float(duration_sec) * 1000.0
        if should_use_vcv_target_window(
            row.get("format_type", ""),
            enabled=bool(self.model_config.enable_vcv_target_window),
            formats=tuple(getattr(self.model_config, "target_window_formats", ("vcv",))),
            alias_type=row.get("alias_type", ""),
            alias_role=alias_role_text,
            cvvc_alias_types=tuple(getattr(self.model_config, "cvvc_target_window_alias_types", ("vc", "vv"))),
            cvvc_alias_roles=tuple(getattr(self.model_config, "cvvc_target_window_alias_roles", ("vc", "vv"))),
        ):
            row_index, row_count = row_window_args(row)
            features, anchors_ms_or_none, duration_ms, _start_frame = crop_oto_target_window(
                features,
                anchors_ms,
                hop_sec=float(hop_sec),
                duration_ms=full_duration_ms,
                row_index_in_wav=row_index,
                file_row_count=row_count,
                window_frames=target_window_frames_for(
                    self.model_config,
                    row.get("format_type", ""),
                    int(self.model_config.vcv_target_window_frames),
                    alias_role=alias_role_text,
                ),
            )
            anchors_ms = np.asarray(anchors_ms_or_none, dtype=np.float32)
        else:
            features, anchors_ms, duration_ms = _crop_around_anchors(
                features,
                anchors_ms,
                hop_sec=float(hop_sec),
                duration_ms=full_duration_ms,
                max_frames=int(self.train_config.max_frames),
                train=self.train,
            )
        heatmap = _make_anchor_heatmap(
            anchors_ms,
            frame_count=int(features.shape[0]),
            hop_sec=float(hop_sec),
            sigma_frames=float(self.train_config.heatmap_sigma_frames),
        )
        if uses_relative_param_head(self.model_config):
            scalar = np.asarray(
                normalize_relative_oto_target(
                    anchors_ms,
                    duration_ms=float(duration_ms),
                    format_type=row.get("format_type", ""),
                    alias_type=row.get("alias_type", ""),
                    transition_type=row.get("transition_type", ""),
                    alias_role=alias_role_text,
                    is_diphthong=is_diphthong,
                    is_special=is_special,
                ),
                dtype=np.float32,
            )
        else:
            scalar = np.clip(anchors_ms / max(float(duration_ms), 1.0), 0.0, 1.0).astype(np.float32)
        language = language_id(self.model_config, row.get("language", ""))
        fmt = format_id(self.model_config, row.get("format_type", ""))
        alias_id = alias_type_id(self.model_config, row.get("alias_type", ""))
        transition_id = transition_type_id(self.model_config, row.get("transition_type", ""))
        prev_alias_id = alias_type_id(self.model_config, row.get("prev_alias_type", ""))
        next_alias_id = alias_type_id(self.model_config, row.get("next_alias_type", ""))
        prev_transition_id = transition_type_id(self.model_config, row.get("prev_transition_type", ""))
        next_transition_id = transition_type_id(self.model_config, row.get("next_transition_type", ""))
        role_id = alias_role_id(self.model_config, alias_role_text)
        # Compute neighbor roles on-the-fly if manifest lacks them (old manifests).
        prev_role_text = str(row.get("prev_alias_role", "") or "")
        if not prev_role_text or row.get("prev_alias_role") is None:
            _prev_feats = extract_alias_features(
                str(row.get("prev_alias", "") or ""),
                language=str(row.get("language", "") or ""),
            )
            prev_role_text = str(_prev_feats.get("alias_role", "") or "")
        next_role_text = str(row.get("next_alias_role", "") or "")
        if not next_role_text or row.get("next_alias_role") is None:
            _next_feats = extract_alias_features(
                str(row.get("next_alias", "") or ""),
                language=str(row.get("language", "") or ""),
            )
            next_role_text = str(_next_feats.get("alias_role", "") or "")
        prev_role_id = alias_role_id(self.model_config, prev_role_text)
        next_role_id = alias_role_id(self.model_config, next_role_text)
        extra_flags = np.asarray(
            [1.0 if is_diphthong else 0.0, 1.0 if is_special else 0.0],
            dtype=np.float32,
        )
        context = _context_array_from_row(row)
        weight = float(row.get("sample_weight", row.get("weight", 1.0)) or 1.0)
        weight *= _format_loss_multiplier(row, self.train_config)
        weight *= _role_loss_multiplier(alias_role_text, self.train_config)
        weight *= _row_order_violation_multiplier(row, self.train_config)
        voicebank_id = self.voicebank_to_id.get(
            str(row.get("voicebank_id", "") or "unknown_voicebank"),
            0,
        )
        # CTC alignment target: phone-level token ids (3-A). Computed for every
        # row regardless of model.enable_ctc_head; the loss is gated by
        # OtoTrainConfig.ctc_loss_weight, and rows that produce no phones get
        # length=0 and are masked out in _oto_loss.
        phone_ids = np.asarray(
            phone_ids_from_alias(alias_text, language=language_text),
            dtype=np.int64,
        )
        return (
            features.astype(np.float32),
            heatmap.astype(np.float32),
            scalar,
            anchors_ms.astype(np.float32),
            float(duration_ms),
            int(language),
            int(fmt),
            context.astype(np.float32),
            int(alias_id),
            int(transition_id),
            int(prev_alias_id),
            int(next_alias_id),
            int(prev_transition_id),
            int(next_transition_id),
            int(role_id),
            int(prev_role_id),
            int(next_role_id),
            extra_flags,
            max(0.05, min(2.0, weight)),
            int(voicebank_id),
            int(idx),
            phone_ids,
        )

    def _load_or_compute_features(self, wav_path: str) -> tuple[np.ndarray, float, float]:
        if self.feature_cache_enabled and self.feature_cache_dir:
            cache_path = _feature_cache_path(
                self.feature_cache_dir,
                wav_path,
                sample_rate=int(self.model_config.sample_rate),
                n_mels=int(self.model_config.n_mels),
                frame_ms=float(self.model_config.frame_ms),
                hop_ms=float(self.model_config.hop_ms),
            )
            cached = _load_feature_cache(cache_path)
            if cached is not None:
                return cached
        samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=int(self.model_config.sample_rate))
        features, hop_sec = log_mel_spectrogram(
            samples,
            sr,
            n_mels=int(self.model_config.n_mels),
            frame_ms=float(self.model_config.frame_ms),
            hop_ms=float(self.model_config.hop_ms),
        )
        out = (features.astype(np.float32), float(hop_sec), float(duration_sec))
        if self.feature_cache_enabled and self.feature_cache_dir and (not self.feature_cache_readonly):
            _save_feature_cache(cache_path, out[0], out[1], out[2])
        return out


def train_oto_from_manifest(
    rows: list[dict[str, Any]],
    output_path: str,
    *,
    val_rows: list[dict[str, Any]] | None = None,
    train_config: OtoTrainConfig | None = None,
    model_config: OtoCrnnConfig | None = None,
    init_state_dict: dict[str, Any] | None = None,
    init_state_strict: bool = True,
    init_tag: str = "",
) -> dict[str, Any]:
    torch = __import__("torch")
    nn = __import__("torch.nn").nn
    data = [row for row in rows if str(row.get("audio", "") or "")]
    if not data:
        raise ValueError("OTO manifest has no training rows")
    fixed_val_rows = [row for row in (val_rows or []) if str(row.get("audio", "") or "")]
    cfg = train_config or OtoTrainConfig()
    model_cfg = model_config or OtoCrnnConfig()
    if bool(getattr(cfg, "enable_activity_quality_filter", True)):
        data = _apply_activity_quality_filter(data, cfg, stage_label="train")
        if fixed_val_rows:
            fixed_val_rows = _apply_activity_quality_filter(fixed_val_rows, cfg, stage_label="val")
        if not data:
            raise ValueError("OTO manifest has no rows after activity-quality filter")
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    random.shuffle(data)
    if fixed_val_rows:
        val_rows_final = fixed_val_rows
        train_rows = data
    else:
        val_count = max(1, int(round(len(data) * float(cfg.val_ratio)))) if len(data) >= 10 else 0
        val_rows_final = data[:val_count]
        train_rows = data[val_count:] if val_count else data
    if not train_rows:
        train_rows = data
        val_rows_final = []

    train_ds = OtoAnchorDataset(train_rows, model_cfg, train_config=cfg, train=True)
    val_ds = OtoAnchorDataset(val_rows_final, model_cfg, train_config=cfg, train=False) if val_rows_final else None
    hard_case_boosts = np.ones((len(train_ds),), dtype=np.float32)
    loader_workers = _resolve_num_workers(int(cfg.num_workers))
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=max(1, min(8, int(cfg.batch_size))),
            shuffle=False,
            collate_fn=_collate,
            num_workers=loader_workers,
            persistent_workers=bool(cfg.dataloader_persistent_workers) and loader_workers > 0,
            prefetch_factor=int(cfg.dataloader_prefetch_factor) if loader_workers > 0 else None,
            pin_memory=_pin_memory_enabled(torch, cfg.device),
        )
        if val_ds
        else None
    )

    device = resolve_torch_device(torch, str(cfg.device))
    if device.type == "cuda":
        if bool(getattr(cfg, "enable_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        if bool(getattr(cfg, "enable_cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = _make_grad_scaler(torch, enabled=use_amp)
    model = build_oto_model(model_cfg).to(device)
    init_load_summary: dict[str, Any] = {}
    if init_state_dict:
        if bool(init_state_strict):
            model.load_state_dict(init_state_dict, strict=True)
            init_load_summary = {
                "strategy": "strict",
                "loaded": int(len(init_state_dict)),
                "skipped": 0,
            }
        else:
            compatible_state, init_load_summary = _filter_compatible_init_state(init_state_dict, model.state_dict())
            missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            init_load_summary["missing_after_load"] = int(len(missing))
            init_load_summary["unexpected_after_load"] = int(len(unexpected))
        if str(init_tag or "").strip():
            print(
                f"[oto_anchor][train] loaded init_state from {init_tag} "
                f"strategy={init_load_summary.get('strategy', 'strict')} "
                f"loaded={init_load_summary.get('loaded', 0)} "
                f"skipped={init_load_summary.get('skipped', 0)}",
                flush=True,
            )
        else:
            print(
                f"[oto_anchor][train] loaded init_state "
                f"strategy={init_load_summary.get('strategy', 'strict')} "
                f"loaded={init_load_summary.get('loaded', 0)} "
                f"skipped={init_load_summary.get('skipped', 0)}",
                flush=True,
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_val = None
    best_selection_score = None
    best_state = None
    latest_state = None
    best_conf_calibration = {
        "scale": float(getattr(model_cfg, "confidence_calibration_scale", 1.0) or 1.0),
        "bias": float(getattr(model_cfg, "confidence_calibration_bias", 0.0) or 0.0),
        "low_threshold": float(getattr(model_cfg, "confidence_low_threshold", 0.58) or 0.58),
        "error_threshold_ms": float(getattr(model_cfg, "predicted_error_low_threshold_ms", 80.0) or 80.0),
    }
    stagnant_epochs = 0
    # Tracks total optimizer steps across epochs so the CTC head warmup
    # (cfg.ctc_warmup_steps) measures wall steps, not steps within a single
    # epoch. A multi-epoch resume picks up where the previous run left off
    # only when the warmup completes inside the current run, which is the
    # desired behavior for short smoke runs.
    global_train_step = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        train_loader = _build_train_loader(
            torch=torch,
            dataset=train_ds,
            cfg=cfg,
            hard_case_boosts=hard_case_boosts,
            loader_workers=loader_workers,
        )
        model.train()
        loss_sum = 0.0
        row_sum = 0
        epoch_hard_scores: dict[int, float] = {}
        for batch_idx, batch in enumerate(
            _iter_device_batches(
                train_loader,
                device,
                torch=torch,
                prefetch_batches=max(0, int(getattr(cfg, "cuda_prefetch_batches", 0) or 0)),
            ),
            start=1,
        ):
            (
                x,
                heat,
                scalar,
                _anchors_ms,
                _duration_ms,
                lang,
                fmt,
                context,
                alias_id,
                transition_id,
                prev_alias_id,
                next_alias_id,
                prev_transition_id,
                next_transition_id,
                role_id,
                prev_role_id,
                next_role_id,
                extra_flags,
                weight,
                _voicebank_ids,
                sample_indices,
                mask,
                phone_targets,
                phone_lengths,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, enabled=use_amp):
                outputs = model(
                    x,
                    lang,
                    fmt,
                    context,
                    alias_id,
                    transition_id,
                    prev_alias_id,
                    next_alias_id,
                    prev_transition_id,
                    next_transition_id,
                    alias_role_ids=role_id,
                    prev_alias_role_ids=prev_role_id,
                    next_alias_role_ids=next_role_id,
                    extra_alias_flags=extra_flags,
                )
                loss = _train_step_loss(
                    outputs,
                    heat,
                    scalar,
                    weight,
                    mask,
                    nn,
                    cfg,
                    relative_scalar=uses_relative_param_head(model_cfg),
                    phone_targets=phone_targets,
                    phone_lengths=phone_lengths,
                    global_step=global_train_step,
                )
                if bool(getattr(cfg, "enable_hard_case_mining", False)):
                    _collect_epoch_hard_scores(
                        outputs=outputs,
                        scalar_target=scalar,
                        sample_indices=sample_indices,
                        store=epoch_hard_scores,
                    )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            rows_in_batch = int(x.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * rows_in_batch
            row_sum += rows_in_batch
            # Advance the wall-step counter AFTER the optimizer step so warmup
            # boundary checks (`global_step < warmup_steps`) are zero-indexed.
            global_train_step += 1
            if int(cfg.log_every) > 0 and (batch_idx == 1 or batch_idx % int(cfg.log_every) == 0):
                warmup_tag = (
                    " [ctc-warmup]"
                    if _ctc_warmup_active(global_train_step - 1, getattr(cfg, "ctc_warmup_steps", 0))
                    else ""
                )
                print(
                    f"[oto_anchor][train] epoch={epoch}/{int(cfg.epochs)} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f} "
                    f"device={device} amp={int(use_amp)}{warmup_tag}",
                    flush=True,
                )
        row = {"epoch": float(epoch), "train_loss": float(loss_sum / max(1, row_sum))}
        if bool(getattr(cfg, "enable_hard_case_mining", False)):
            hard_case_boosts = _build_hard_case_boosts(
                size=len(train_ds),
                score_by_index=epoch_hard_scores,
                top_ratio=float(getattr(cfg, "hard_case_top_ratio", 0.25)),
                boost=float(getattr(cfg, "hard_case_boost", 2.5)),
            )
            row["hard_case_count"] = float(int(np.sum(hard_case_boosts > 1.0)))
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device, nn, cfg, model_cfg)
            row.update(val_metrics)
            val_loss = float(val_metrics["val_loss"])
            selection_score = _selection_score(val_metrics, cfg)
            row["val_selection_score"] = float(selection_score)
            if best_selection_score is None or selection_score < best_selection_score:
                best_selection_score = selection_score
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_conf_calibration = {
                    "scale": float(val_metrics.get("val_conf_calibration_scale", best_conf_calibration["scale"])),
                    "bias": float(val_metrics.get("val_conf_calibration_bias", best_conf_calibration["bias"])),
                    "low_threshold": float(
                        val_metrics.get("val_confidence_low_threshold", best_conf_calibration["low_threshold"])
                    ),
                    "error_threshold_ms": float(
                        val_metrics.get("val_predicted_error_low_threshold_ms", best_conf_calibration["error_threshold_ms"])
                    ),
                }
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
        else:
            latest_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        save_every = int(getattr(cfg, "checkpoint_save_every_epochs", 0) or 0)
        if save_every > 0 and (epoch % save_every == 0):
            epoch_output_path = _epoch_checkpoint_path(output_path, epoch)
            save_oto_checkpoint(
                epoch_output_path,
                model.cpu(),
                model_cfg,
                meta={
                    "train_rows": len(train_rows),
                    "val_rows": len(val_rows_final),
                    "fixed_val_manifest": bool(fixed_val_rows),
                    "history": history,
                    "device": str(device),
                    "amp": bool(use_amp),
                    "best_val_loss": best_val,
                    "best_selection_score": best_selection_score,
                    "best_confidence_calibration": dict(best_conf_calibration),
                    "checkpoint_type": "periodic_epoch",
                    "epoch": int(epoch),
                },
            )
            model = model.to(device)
        if int(getattr(cfg, "early_stop_patience", 0) or 0) > 0 and stagnant_epochs >= int(cfg.early_stop_patience):
            print(
                f"[oto_anchor][train] early-stop epoch={epoch} stagnant={stagnant_epochs} patience={int(cfg.early_stop_patience)}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif latest_state is not None:
        model.load_state_dict(latest_state)
    model_cfg.confidence_calibration_scale = float(best_conf_calibration["scale"])
    model_cfg.confidence_calibration_bias = float(best_conf_calibration["bias"])
    model_cfg.confidence_low_threshold = float(best_conf_calibration["low_threshold"])
    model_cfg.predicted_error_low_threshold_ms = float(best_conf_calibration["error_threshold_ms"])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_oto_checkpoint(
        output_path,
        model.cpu(),
        model_cfg,
        meta={
            "train_rows": len(train_rows),
            "val_rows": len(val_rows_final),
            "fixed_val_manifest": bool(fixed_val_rows),
            "init_tag": str(init_tag or ""),
            "init_load_summary": dict(init_load_summary),
            "history": history,
            "device": str(device),
            "amp": bool(use_amp),
            "best_val_loss": best_val,
            "best_selection_score": best_selection_score,
            "best_confidence_calibration": dict(best_conf_calibration),
        },
    )
    return {
        "output_path": os.path.abspath(output_path),
        "history": history,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows_final),
        "init_tag": str(init_tag or ""),
        "init_load_summary": init_load_summary,
        "device": str(device),
        "amp": bool(use_amp),
    }


def _filter_compatible_init_state(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    compatible: dict[str, Any] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in dict(source_state or {}).items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped_missing.append(str(key))
            continue
        source_shape = tuple(getattr(value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if source_shape != target_shape:
            skipped_shape.append(str(key))
            continue
        compatible[key] = value
    summary = {
        "strategy": "compatible",
        "loaded": int(len(compatible)),
        "skipped": int(len(skipped_missing) + len(skipped_shape)),
        "skipped_missing": skipped_missing[:25],
        "skipped_shape": skipped_shape[:25],
        "skipped_missing_count": int(len(skipped_missing)),
        "skipped_shape_count": int(len(skipped_shape)),
    }
    return compatible, summary


def _evaluate(model, loader, device, nn, cfg: OtoTrainConfig, model_cfg: OtoCrnnConfig) -> dict[str, float]:
    torch = __import__("torch")
    model.eval()
    loss_sum = 0.0
    row_sum = 0
    mae_values: list[float] = []
    pre_errors_all: list[float] = []
    hard_failure_count = 0
    voicebank_pre_errors: dict[int, list[float]] = defaultdict(list)
    voicebank_hard_counts: dict[int, int] = defaultdict(int)
    voicebank_counts: dict[int, int] = defaultdict(int)
    confidence_raw_all: list[float] = []
    with torch.no_grad():
        eval_prefetch = (
            max(0, int(getattr(cfg, "cuda_prefetch_batches", 0) or 0))
            if bool(getattr(cfg, "cuda_prefetch_eval", True))
            else 0
        )
        for batch in _iter_device_batches(loader, device, torch=torch, prefetch_batches=eval_prefetch):
            (
                x,
                heat,
                scalar,
                anchors_ms,
                duration_ms,
                lang,
                fmt,
                context,
                alias_id,
                transition_id,
                prev_alias_id,
                next_alias_id,
                prev_transition_id,
                next_transition_id,
                role_id,
                prev_role_id,
                next_role_id,
                extra_flags,
                weight,
                voicebank_ids,
                _sample_indices,
                mask,
                phone_targets,
                phone_lengths,
            ) = batch
            outputs = model(
                x,
                lang,
                fmt,
                context,
                alias_id,
                transition_id,
                prev_alias_id,
                next_alias_id,
                prev_transition_id,
                next_transition_id,
                alias_role_ids=role_id,
                prev_alias_role_ids=prev_role_id,
                next_alias_role_ids=next_role_id,
                extra_alias_flags=extra_flags,
            )
            relative_scalar = uses_relative_param_head(model_cfg)
            loss = _oto_loss(
                outputs,
                heat,
                scalar,
                weight,
                mask,
                nn,
                cfg,
                relative_scalar=relative_scalar,
                phone_targets=phone_targets,
                phone_lengths=phone_lengths,
            )
            pred_scalar = torch.sigmoid(outputs["scalar_logits"])
            if relative_scalar:
                pred_anchors = _relative_anchor_predictions(
                    pred_scalar.detach().cpu().numpy(),
                    duration_ms.detach().cpu().numpy(),
                    fmt.detach().cpu().numpy(),
                    alias_id.detach().cpu().numpy(),
                    transition_id.detach().cpu().numpy(),
                    role_id.detach().cpu().numpy(),
                    extra_flags.detach().cpu().numpy(),
                    model_cfg,
                )
                target_anchors = anchors_ms.detach().cpu().numpy().astype(np.float32)
                batch_mae = np.abs(pred_anchors - target_anchors).mean(axis=1).tolist()
                mae_values.extend(float(v) for v in batch_mae)
                pre_errors = np.abs(pred_anchors[:, 2] - target_anchors[:, 2]).tolist()
                pre_errors_all.extend(float(v) for v in pre_errors)
                hard_flags = _hard_failure_flags_from_anchors(pred_anchors, duration_ms.detach().cpu().numpy())
            else:
                pred_ms = pred_scalar * duration_ms[:, None]
                mae = torch.abs(pred_ms - anchors_ms).mean(dim=1)
                mae_values.extend(float(v) for v in mae.detach().cpu().tolist())
                pred_anchors = pred_ms.detach().cpu().numpy().astype(np.float32)
                target_anchors = anchors_ms.detach().cpu().numpy().astype(np.float32)
                pre_errors = np.abs(pred_anchors[:, 2] - target_anchors[:, 2]).tolist()
                pre_errors_all.extend(float(v) for v in pre_errors)
                hard_flags = _hard_failure_flags_from_anchors(pred_anchors, duration_ms.detach().cpu().numpy())
            confidence_raw_all.extend(
                _confidence_raw_scores(
                    outputs=outputs,
                    duration_ms_np=duration_ms.detach().cpu().numpy(),
                    error_scale_ms=float(getattr(model_cfg, "confidence_error_scale_ms", 75.0) or 75.0),
                )
            )
            vb_arr = voicebank_ids.detach().cpu().numpy()
            for i, vb in enumerate(vb_arr.tolist()):
                vb_i = int(vb)
                voicebank_counts[vb_i] += 1
                p_err = float(pre_errors[i]) if i < len(pre_errors) else 0.0
                voicebank_pre_errors[vb_i].append(p_err)
                if bool(hard_flags[i]):
                    voicebank_hard_counts[vb_i] += 1
                    hard_failure_count += 1
            rows_in_batch = int(x.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * rows_in_batch
            row_sum += rows_in_batch
    pre_acc50 = _hit_rate_np(pre_errors_all, 50.0)
    worst_vb_acc50 = _worst_voicebank_acc50(voicebank_pre_errors)
    worst_vb_hard_failure = _worst_voicebank_hard_failure_rate(voicebank_hard_counts, voicebank_counts)
    conf_cal = _derive_confidence_calibration(
        confidence_raw=confidence_raw_all,
        pre_errors_ms=pre_errors_all,
        cfg=cfg,
    )
    return {
        "val_loss": float(loss_sum / max(1, row_sum)),
        "val_anchor_mae_ms": float(sum(mae_values) / max(1, len(mae_values))),
        "val_preutterance_acc_50ms": float(pre_acc50),
        "val_hard_failure_rate": float(hard_failure_count) / float(max(1, row_sum)),
        "val_worst_voicebank_preutterance_acc_50ms": float(worst_vb_acc50),
        "val_worst_voicebank_hard_failure_rate": float(worst_vb_hard_failure),
        "val_conf_calibration_scale": float(conf_cal["scale"]),
        "val_conf_calibration_bias": float(conf_cal["bias"]),
        "val_confidence_low_threshold": float(conf_cal["low_threshold"]),
        "val_predicted_error_low_threshold_ms": float(conf_cal["error_threshold_ms"]),
        "val_conf_good_mean": float(conf_cal["good_mean"]),
        "val_conf_bad_mean": float(conf_cal["bad_mean"]),
    }


def _epoch_checkpoint_path(output_path: str, epoch: int) -> str:
    root, ext = os.path.splitext(str(output_path))
    suffix = f".e{int(epoch):03d}"
    if not ext:
        return f"{root}{suffix}"
    return f"{root}{suffix}{ext}"


def _oto_loss(
    outputs,
    heat,
    scalar,
    weight,
    mask,
    nn,
    cfg: OtoTrainConfig,
    *,
    relative_scalar: bool = False,
    phone_targets=None,
    phone_lengths=None,
):
    torch = __import__("torch")
    pred_heat = torch.sigmoid(outputs["heatmap_logits"])
    heat_loss_raw = (pred_heat - heat) ** 2
    heat_denom = torch.clamp(mask[:, :, None].sum() * heat.shape[-1], min=1.0)
    heat_loss = (heat_loss_raw * mask[:, :, None] * weight[:, None, None]).sum() / heat_denom
    scalar_pred = torch.sigmoid(outputs["scalar_logits"])
    scalar_loss_raw = nn.functional.smooth_l1_loss(scalar_pred, scalar, reduction="none")
    scalar_loss = (scalar_loss_raw.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    scalar_logvar = outputs.get("scalar_logvar")
    if scalar_logvar is not None:
        residual_sq = (scalar_pred - scalar) ** 2
        inv_var = torch.exp(-scalar_logvar)
        uncertainty_nll = 0.5 * ((residual_sq * inv_var) + scalar_logvar)
        uncertainty_loss = (uncertainty_nll.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    else:
        uncertainty_loss = scalar_pred.new_tensor(0.0)
    conf_logits = outputs.get("confidence_logits")
    if conf_logits is not None:
        error_scale = max(1e-4, float(getattr(cfg, "confidence_target_error_scale", 0.08)))
        conf_target = torch.exp(-torch.abs(scalar_pred.detach() - scalar).mean(dim=1) / error_scale)
        conf_target = conf_target.clamp(
            min=float(getattr(cfg, "min_confidence_target", 0.02)),
            max=float(getattr(cfg, "max_confidence_target", 0.98)),
        )
        conf_loss_raw = nn.functional.binary_cross_entropy_with_logits(conf_logits, conf_target, reduction="none")
        conf_loss = (conf_loss_raw * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    else:
        conf_loss = scalar_pred.new_tensor(0.0)
    order_loss = _relative_order_penalty(scalar_pred) if relative_scalar else _order_penalty(scalar_pred)
    ctc_loss = _ctc_alignment_loss(
        outputs.get("ctc_logits"),
        phone_targets,
        phone_lengths,
        mask,
        nn,
        cfg,
    )
    return (
        heat_loss * float(cfg.heatmap_loss_weight)
        + scalar_loss * float(cfg.scalar_loss_weight)
        + uncertainty_loss * float(getattr(cfg, "uncertainty_loss_weight", 0.0))
        + conf_loss * float(getattr(cfg, "confidence_loss_weight", 0.0))
        + order_loss * float(cfg.order_loss_weight)
        + ctc_loss * float(getattr(cfg, "ctc_loss_weight", 0.0))
    )


def _ctc_warmup_active(global_step: int, warmup_steps: int) -> bool:
    """Return True while CTC head warmup should suppress anchor losses.

    A 0 (or negative) `warmup_steps` disables warmup outright.
    """
    return int(warmup_steps) > 0 and int(global_step) < int(warmup_steps)


def _train_step_loss(
    outputs,
    heat,
    scalar,
    weight,
    mask,
    nn,
    cfg: OtoTrainConfig,
    *,
    relative_scalar: bool,
    phone_targets,
    phone_lengths,
    global_step: int,
):
    """Dispatch between the warmup-only-CTC loss and the full multi-task loss.

    During warmup we deliberately discard the anchor regression / heatmap /
    confidence losses so that gradients flowing back into the encoder come
    only from CTC. After `cfg.ctc_warmup_steps` global steps the function
    falls back to the standard `_oto_loss` path.
    """
    if _ctc_warmup_active(global_step, getattr(cfg, "ctc_warmup_steps", 0)):
        ctc_loss = _ctc_alignment_loss(
            outputs.get("ctc_logits"),
            phone_targets,
            phone_lengths,
            mask,
            nn,
            cfg,
        )
        return ctc_loss * float(getattr(cfg, "ctc_loss_weight", 0.0))
    return _oto_loss(
        outputs,
        heat,
        scalar,
        weight,
        mask,
        nn,
        cfg,
        relative_scalar=relative_scalar,
        phone_targets=phone_targets,
        phone_lengths=phone_lengths,
    )


def _ctc_alignment_loss(ctc_logits, phone_targets, phone_lengths, mask, nn, cfg: OtoTrainConfig):
    """CTC alignment auxiliary loss (3-A in CRNN-정확도-개선-방안-v2).

    Returns 0.0 when:
    - the model lacks a CTC head (ctc_logits is None)
    - the CTC loss weight is 0 (caller still calls this; we short-circuit cheaply)
    - no row in the batch has a phone target (lengths are all 0)
    - any safety check fails
    """
    torch = __import__("torch")
    if ctc_logits is None:
        # Model has no CTC head; return a zero scalar on the right device.
        return mask.new_tensor(0.0)
    if float(getattr(cfg, "ctc_loss_weight", 0.0)) <= 0.0:
        return ctc_logits.new_tensor(0.0)
    if phone_targets is None or phone_lengths is None:
        return ctc_logits.new_tensor(0.0)
    target_lengths = phone_lengths.long().clamp(min=0)
    valid = target_lengths > 0
    if not bool(valid.any()):
        return ctc_logits.new_tensor(0.0)
    # CTC inputs must satisfy: input_length >= target_length. Guard by clipping
    # target rows whose phone length exceeds the audio frame count, which is
    # rare but possible for very short clips.
    input_lengths = mask.sum(dim=1).long().clamp(min=1)
    fits = input_lengths >= target_lengths
    valid = valid & fits
    if not bool(valid.any()):
        return ctc_logits.new_tensor(0.0)
    # log_softmax on the last (vocab) dim, then transpose to (T, B, V) for CTCLoss.
    log_probs = nn.functional.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    # Filter out rows that are length-0 or do not fit, to avoid CTCLoss blowing up.
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
    log_probs = log_probs.index_select(1, valid_idx)
    # phone_targets is (B, max_phone). Filter rows, then trim to max valid length.
    targets = phone_targets.index_select(0, valid_idx).long()
    in_lens = input_lengths.index_select(0, valid_idx)
    tgt_lens = target_lengths.index_select(0, valid_idx)
    # Also trim the target tensor to the longest valid target length to keep it tight.
    max_tgt = int(tgt_lens.max().item())
    if max_tgt > 0:
        targets = targets[:, :max_tgt]
    ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    loss = ctc_fn(log_probs, targets, in_lens, tgt_lens)
    if not torch.isfinite(loss):
        return ctc_logits.new_tensor(0.0)
    return loss


def _order_penalty(pred):
    torch = __import__("torch")
    # order: offset <= overlap <= preutterance <= consonant <= cutoff
    diffs = pred[:, :-1] - pred[:, 1:]
    return torch.relu(diffs).mean()


def _relative_order_penalty(pred):
    torch = __import__("torch")
    # Relative target order only requires overlap_delta <= pre_delta.
    return torch.relu(pred[:, 1] - pred[:, 2]).mean()


def _relative_anchor_mae_values(
    pred_scalar,
    duration_ms,
    format_ids,
    alias_ids,
    transition_ids,
    role_ids,
    extra_flags,
    anchors_ms,
    model_cfg: OtoCrnnConfig,
) -> list[float]:
    values: list[float] = []
    formats = list(model_cfg.format_types)
    aliases = list(model_cfg.alias_types)
    transitions = list(model_cfg.transition_types)
    roles = list(model_cfg.alias_roles)
    use_role = bool(getattr(model_cfg, "enable_alias_role_embedding", False)) and len(roles) > 1
    for scalar_row, duration, fmt_idx, alias_idx, transition_idx, role_idx, flags_row, anchor_row in zip(
        pred_scalar, duration_ms, format_ids, alias_ids, transition_ids, role_ids, extra_flags, anchors_ms
    ):
        fmt_i = int(fmt_idx)
        alias_i = int(alias_idx)
        transition_i = int(transition_idx)
        role_i = int(role_idx)
        format_type = formats[fmt_i] if 0 <= fmt_i < len(formats) else "other"
        alias_type = aliases[alias_i] if 0 <= alias_i < len(aliases) else "other"
        transition_type = transitions[transition_i] if 0 <= transition_i < len(transitions) else "other"
        role_text = roles[role_i] if (use_role and 0 <= role_i < len(roles)) else ""
        flags_arr = np.asarray(flags_row, dtype=np.float32)
        is_diphthong = bool(flags_arr.shape[0] > 0 and flags_arr[0] >= 0.5)
        is_special = bool(flags_arr.shape[0] > 1 and flags_arr[1] >= 0.5)
        params = decode_relative_oto_params(
            scalar_row,
            duration_ms=float(duration),
            format_type=format_type,
            alias_type=alias_type,
            transition_type=transition_type,
            prior_blend=right_boundary_prior_blend_for_context(
                model_cfg,
                format_type,
                alias_role=role_text,
                is_special=bool(is_special),
            ),
            alias_role=role_text,
            is_diphthong=is_diphthong,
            is_special=is_special,
        )
        pred_anchors = np.asarray(relative_params_to_anchors(params, duration_ms=float(duration)), dtype=np.float32)
        values.append(float(np.abs(pred_anchors - np.asarray(anchor_row, dtype=np.float32)).mean()))
    return values


def _collate(batch):
    torch = __import__("torch")
    max_len = max(int(item[0].shape[0]) for item in batch)
    n_mels = int(batch[0][0].shape[1])
    anchor_count = len(OTO_ANCHOR_NAMES)
    xs = np.zeros((len(batch), max_len, n_mels), dtype=np.float32)
    heats = np.zeros((len(batch), max_len, anchor_count), dtype=np.float32)
    masks = np.zeros((len(batch), max_len), dtype=np.float32)
    scalars = np.zeros((len(batch), anchor_count), dtype=np.float32)
    anchors_ms = np.zeros((len(batch), anchor_count), dtype=np.float32)
    durations = np.ones((len(batch),), dtype=np.float32)
    langs = np.zeros((len(batch),), dtype=np.int64)
    fmts = np.zeros((len(batch),), dtype=np.int64)
    contexts = np.zeros((len(batch), 12), dtype=np.float32)
    alias_ids = np.zeros((len(batch),), dtype=np.int64)
    transition_ids = np.zeros((len(batch),), dtype=np.int64)
    prev_alias_ids = np.zeros((len(batch),), dtype=np.int64)
    next_alias_ids = np.zeros((len(batch),), dtype=np.int64)
    prev_transition_ids = np.zeros((len(batch),), dtype=np.int64)
    next_transition_ids = np.zeros((len(batch),), dtype=np.int64)
    role_ids = np.zeros((len(batch),), dtype=np.int64)
    prev_role_ids = np.zeros((len(batch),), dtype=np.int64)
    next_role_ids = np.zeros((len(batch),), dtype=np.int64)
    extra_flags = np.zeros((len(batch), 2), dtype=np.float32)
    weights = np.ones((len(batch),), dtype=np.float32)
    voicebank_ids = np.zeros((len(batch),), dtype=np.int64)
    sample_indices = np.zeros((len(batch),), dtype=np.int64)
    # CTC phone targets: collect per-row 1D arrays first, then pad to max_phone.
    phone_id_arrays: list[np.ndarray] = []
    for item in batch:
        # Backward compat: older items may not include phone_ids; treat as empty.
        if len(item) >= 22:
            phone_id_arrays.append(np.asarray(item[21], dtype=np.int64).reshape(-1))
        else:
            phone_id_arrays.append(np.zeros((0,), dtype=np.int64))
    max_phone = max(1, max((int(arr.shape[0]) for arr in phone_id_arrays), default=1))
    phone_targets = np.zeros((len(batch), max_phone), dtype=np.int64)
    phone_lengths = np.zeros((len(batch),), dtype=np.int64)
    for idx, arr in enumerate(phone_id_arrays):
        n_p = int(arr.shape[0])
        if n_p > 0:
            phone_targets[idx, :n_p] = arr
        phone_lengths[idx] = n_p
    for idx, item in enumerate(batch):
        # Unpack tolerating older 21-tuple items.
        (
            features,
            heatmap,
            scalar,
            anchor_ms,
            duration_ms,
            language,
            fmt,
            context,
            alias_id,
            transition_id,
            prev_alias_id,
            next_alias_id,
            prev_transition_id,
            next_transition_id,
            role_id,
            prev_role_id,
            next_role_id,
            flags,
            weight,
            voicebank_id,
            sample_index,
        ) = item[:21]
        n = int(features.shape[0])
        xs[idx, :n] = features
        heats[idx, :n] = heatmap
        masks[idx, :n] = 1.0
        scalars[idx] = scalar
        anchors_ms[idx] = anchor_ms
        durations[idx] = max(1.0, float(duration_ms))
        langs[idx] = int(language)
        fmts[idx] = int(fmt)
        contexts[idx] = context
        alias_ids[idx] = int(alias_id)
        transition_ids[idx] = int(transition_id)
        prev_alias_ids[idx] = int(prev_alias_id)
        next_alias_ids[idx] = int(next_alias_id)
        prev_transition_ids[idx] = int(prev_transition_id)
        next_transition_ids[idx] = int(next_transition_id)
        role_ids[idx] = int(role_id)
        prev_role_ids[idx] = int(prev_role_id)
        next_role_ids[idx] = int(next_role_id)
        flags_arr = np.asarray(flags, dtype=np.float32)
        if flags_arr.shape[0] >= 2:
            extra_flags[idx] = flags_arr[:2]
        weights[idx] = float(weight)
        voicebank_ids[idx] = int(voicebank_id)
        sample_indices[idx] = int(sample_index)
    return (
        torch.from_numpy(xs),
        torch.from_numpy(heats),
        torch.from_numpy(scalars),
        torch.from_numpy(anchors_ms),
        torch.from_numpy(durations),
        torch.from_numpy(langs),
        torch.from_numpy(fmts),
        torch.from_numpy(contexts),
        torch.from_numpy(alias_ids),
        torch.from_numpy(transition_ids),
        torch.from_numpy(prev_alias_ids),
        torch.from_numpy(next_alias_ids),
        torch.from_numpy(prev_transition_ids),
        torch.from_numpy(next_transition_ids),
        torch.from_numpy(role_ids),
        torch.from_numpy(prev_role_ids),
        torch.from_numpy(next_role_ids),
        torch.from_numpy(extra_flags),
        torch.from_numpy(weights),
        torch.from_numpy(voicebank_ids),
        torch.from_numpy(sample_indices),
        torch.from_numpy(masks),
        torch.from_numpy(phone_targets),
        torch.from_numpy(phone_lengths),
    )


def _move_batch(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def _iter_device_batches(loader, device, *, torch, prefetch_batches: int = 0):
    prefetch_n = max(0, int(prefetch_batches))
    if device.type != "cuda" or prefetch_n <= 0:
        for raw in loader:
            yield _move_batch(raw, device)
        return
    iterator = iter(loader)
    stream = torch.cuda.Stream(device=device)
    queued: deque[tuple[Any, ...]] = deque()

    def _fill() -> None:
        while len(queued) < prefetch_n:
            try:
                raw = next(iterator)
            except StopIteration:
                break
            with torch.cuda.stream(stream):
                queued.append(_move_batch(raw, device))

    _fill()
    while queued:
        current_stream = torch.cuda.current_stream(device=device)
        current_stream.wait_stream(stream)
        batch = queued.popleft()
        for item in batch:
            if hasattr(item, "record_stream"):
                item.record_stream(current_stream)
        _fill()
        yield batch


def _anchor_array_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row.get("anchor_offset_ms", row.get("target_offset_ms", 0.0)) or 0.0),
            float(row.get("anchor_overlap_ms", 0.0) or 0.0),
            float(row.get("anchor_preutterance_ms", 0.0) or 0.0),
            float(row.get("anchor_consonant_ms", 0.0) or 0.0),
            float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0),
        ],
        dtype=np.float32,
    )


def _context_array_from_row(row: dict[str, Any]) -> np.ndarray:
    count = max(1.0, float(row.get("file_row_count", 1.0) or 1.0))
    ratio = float(row.get("row_ratio_in_wav", 0.0) or 0.0)
    count_norm = min(1.0, count / 64.0)
    phone_count = min(1.0, max(0.0, float(row.get("alias_phone_count", 0.0) or 0.0)) / 6.0)
    return np.asarray(
        [
            max(0.0, min(1.0, ratio)),
            count_norm,
            phone_count,
            max(0.0, min(1.0, float(row.get("alias_starts_vowel", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_ends_vowel", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_has_space", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_vc", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_cv", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("alias_is_vv", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("is_head_row", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("is_tail_row", 0.0) or 0.0))),
            max(0.0, min(1.0, float(row.get("prev_alias_ends_vowel", 0.0) or 0.0))),
        ],
        dtype=np.float32,
    )


def _format_loss_multiplier(row: dict[str, Any], cfg: OtoTrainConfig) -> float:
    fmt = str(row.get("format_type", "") or "").strip().lower()
    if fmt == "vcv":
        return max(0.05, float(cfg.vcv_loss_weight))
    if fmt == "cvvc":
        return max(0.05, float(cfg.cvvc_loss_weight))
    if fmt == "cvc":
        return max(0.05, float(cfg.cvc_loss_weight))
    return 1.0


def _role_loss_multiplier(alias_role: str, cfg: OtoTrainConfig) -> float:
    """Upweight hard-to-predict alias roles during training.

    vc aliases in cvvc voicebanks have the highest MAE in evaluation (often
    5-10x the preutterance MAE of cv aliases).  vv aliases are also hard.
    All other roles use the default multiplier of 1.0.
    """
    role = str(alias_role or "").strip().lower()
    if role == "vc":
        return max(0.05, float(cfg.vc_role_loss_weight))
    if role == "vv":
        return max(0.05, float(cfg.vv_role_loss_weight))
    return 1.0


def _row_order_violation_multiplier(row: dict[str, Any], cfg: OtoTrainConfig) -> float:
    """Downweight rows whose row_ratio_in_wav is far from where the actual
    target_offset sits ([[CRNN-정확도-개선-방안-v2]] 1-B). Returns 1.0 when
    alpha is 0 (disabled), otherwise ``1.0 / (1.0 + alpha * clip(score, 0, 3))``.

    The score column is populated by ``oto_targets._compute_row_order_violation_score``;
    if missing (older manifest), the multiplier silently falls back to 1.0.
    """
    alpha = float(getattr(cfg, "row_order_violation_alpha", 0.0) or 0.0)
    if alpha <= 0.0:
        return 1.0
    raw = row.get("row_order_violation_score")
    if raw is None:
        return 1.0
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not (score == score):  # NaN guard
        return 1.0
    score = max(0.0, min(3.0, score))
    denom = 1.0 + alpha * score
    if denom <= 1e-6:
        return 1.0
    return float(1.0 / denom)


def _crop_around_anchors(
    features: np.ndarray,
    anchors_ms: np.ndarray,
    *,
    hop_sec: float,
    duration_ms: float,
    max_frames: int,
    train: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = int(features.shape[0])
    if max_frames <= 0 or n <= int(max_frames):
        return features, np.clip(anchors_ms, 0.0, max(float(duration_ms), 1.0)), max(float(duration_ms), 1.0)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    min_f = int(np.floor(float(np.min(anchors_ms)) / hop_ms))
    max_f = int(np.ceil(float(np.max(anchors_ms)) / hop_ms))
    center = int(round((min_f + max_f) * 0.5))
    lo = max(0, max_f - int(max_frames) + 12)
    hi = min(n - int(max_frames), min_f - 12)
    if lo <= hi:
        start = random.randint(lo, hi) if train else int(round((lo + hi) * 0.5))
    else:
        start = max(0, min(n - int(max_frames), center - int(max_frames) // 2))
    end = start + int(max_frames)
    shifted = anchors_ms - float(start) * hop_ms
    crop_duration_ms = float(max_frames) * hop_ms
    shifted = np.clip(shifted, 0.0, crop_duration_ms)
    return features[start:end], shifted.astype(np.float32), crop_duration_ms


def _make_anchor_heatmap(anchors_ms: np.ndarray, *, frame_count: int, hop_sec: float, sigma_frames: float) -> np.ndarray:
    n = int(frame_count)
    anchor_count = len(OTO_ANCHOR_NAMES)
    out = np.zeros((n, anchor_count), dtype=np.float32)
    if n <= 0:
        return out
    frames = np.arange(n, dtype=np.float32)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    sigma = max(float(sigma_frames), 0.5)
    for idx, anchor_ms in enumerate(anchors_ms):
        center = float(anchor_ms) / hop_ms
        out[:, idx] = np.exp(-0.5 * ((frames - center) / sigma) ** 2)
    return out


def _build_train_loader(
    *,
    torch,
    dataset: OtoAnchorDataset,
    cfg: OtoTrainConfig,
    hard_case_boosts: np.ndarray,
    loader_workers: int,
):
    sampler = None
    shuffle = True
    if bool(getattr(cfg, "enable_balanced_sampling", False)):
        weights = _build_train_sampling_weights(dataset.rows, cfg, hard_case_boosts=hard_case_boosts)
        if weights.size > 0:
            sample_count = max(1, int(round(len(dataset) * max(0.25, float(getattr(cfg, "balanced_sampling_size_factor", 1.0))))))
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(weights.astype(np.float64)),
                num_samples=sample_count,
                replacement=bool(getattr(cfg, "balanced_sampling_replacement", True)),
            )
            shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=max(0, int(loader_workers)),
        persistent_workers=bool(cfg.dataloader_persistent_workers) and int(loader_workers) > 0,
        prefetch_factor=int(cfg.dataloader_prefetch_factor) if int(loader_workers) > 0 else None,
        pin_memory=_pin_memory_enabled(torch, cfg.device),
    )


def _resolve_num_workers(requested: int) -> int:
    if int(requested) >= 0:
        return int(requested)
    cpu_total = int(os.cpu_count() or 4)
    # Keep 1-2 cores free for UI/OS and cap worker fan-out on Windows.
    resolved = max(2, min(8, cpu_total - 2))
    return int(resolved)


def _build_train_sampling_weights(rows: list[dict[str, Any]], cfg: OtoTrainConfig, *, hard_case_boosts: np.ndarray) -> np.ndarray:
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    vb_counter = Counter(str(row.get("voicebank_id", "") or "unknown_voicebank") for row in rows)
    lang_counter = Counter(str(row.get("language", "") or "unknown").strip().lower() or "unknown" for row in rows)
    role_counter = Counter(str(row.get("alias_role", "") or "other").strip().lower() or "other" for row in rows)
    fmt_counter = Counter(str(row.get("format_type", "") or "other").strip().lower() or "other" for row in rows)
    vb_power = float(getattr(cfg, "voicebank_balance_power", 0.55))
    lang_power = float(getattr(cfg, "language_balance_power", 0.25))
    role_power = float(getattr(cfg, "role_balance_power", 0.35))
    fmt_power = float(getattr(cfg, "format_balance_power", 0.20))
    slice_boosts = _parse_slice_sampling_boosts(getattr(cfg, "language_format_role_sampling_boosts", ()))
    cvvc_vc_boost = max(1.0, float(getattr(cfg, "cvvc_vc_sampling_boost", 1.0) or 1.0))
    cvvc_vv_boost = max(1.0, float(getattr(cfg, "cvvc_vv_sampling_boost", 1.0) or 1.0))
    cvvc_vc_multi_boost = max(1.0, float(getattr(cfg, "cvvc_vc_multi_sampling_boost", 1.0) or 1.0))
    boosted_slices: Counter[str] = Counter()
    boosted_vc = 0
    boosted_vv = 0
    boosted_vc_multi = 0
    out = np.ones((len(rows),), dtype=np.float32)
    for idx, row in enumerate(rows):
        vb = str(row.get("voicebank_id", "") or "unknown_voicebank")
        language = str(row.get("language", "") or "unknown").strip().lower() or "unknown"
        role = str(row.get("alias_role", "") or "other").strip().lower() or "other"
        alias_type = str(row.get("alias_type", "") or "other").strip().lower() or "other"
        transition = str(row.get("transition_type", "") or "other").strip().lower() or "other"
        fmt = str(row.get("format_type", "") or "other").strip().lower() or "other"
        w = float(row.get("sample_weight", row.get("weight", 1.0)) or 1.0)
        w /= float(max(1, vb_counter.get(vb, 1))) ** vb_power
        w /= float(max(1, lang_counter.get(language, 1))) ** lang_power
        w /= float(max(1, role_counter.get(role, 1))) ** role_power
        w /= float(max(1, fmt_counter.get(fmt, 1))) ** fmt_power
        slice_key = _language_format_role_key(language, fmt, role)
        for pattern, boost in slice_boosts:
            if _matches_slice_pattern(slice_key, pattern):
                w *= boost
                boosted_slices[pattern] += 1
        if fmt == "cvvc":
            if role == "vc" or alias_type == "vc":
                w *= cvvc_vc_boost
                boosted_vc += 1
            elif role == "vv" or alias_type == "vv":
                w *= cvvc_vv_boost
                boosted_vv += 1
            if cvvc_vc_multi_boost > 1.0 and (role == "vc" or alias_type == "vc") and transition == "multi":
                w *= cvvc_vc_multi_boost
                boosted_vc_multi += 1
        if idx < int(hard_case_boosts.shape[0]):
            w *= float(max(1.0, hard_case_boosts[idx]))
        out[idx] = max(1e-5, float(w))
    if cvvc_vc_boost > 1.0 or cvvc_vv_boost > 1.0 or cvvc_vc_multi_boost > 1.0:
        print(
            f"[oto_anchor][sampling] cvvc_vc_boost={cvvc_vc_boost:.2f} rows={boosted_vc} "
            f"cvvc_vv_boost={cvvc_vv_boost:.2f} rows={boosted_vv} "
            f"cvvc_vc_multi_boost={cvvc_vc_multi_boost:.2f} rows={boosted_vc_multi}",
            flush=True,
        )
    if slice_boosts:
        summary = ", ".join(f"{pattern}={boost:.2f} rows={int(boosted_slices.get(pattern, 0))}" for pattern, boost in slice_boosts)
        print(f"[oto_anchor][sampling] slice_boosts {summary}", flush=True)
    return out


def _parse_slice_sampling_boosts(items: object) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    if isinstance(items, str):
        raw_items = [item.strip() for item in items.split(",")]
    else:
        raw_items = list(items or ())
    for item in raw_items:
        text = str(item or "").strip().lower()
        if not text or "=" not in text:
            continue
        pattern, raw_boost = text.split("=", 1)
        pattern = pattern.strip().replace("|", "/")
        if len(pattern.split("/")) != 3:
            continue
        try:
            boost = float(raw_boost)
        except Exception:
            continue
        if boost <= 0.0:
            continue
        out.append((pattern, float(boost)))
    return tuple(out)


def _language_format_role_key(language: object, format_type: object, alias_role: object) -> str:
    return "/".join(
        [
            str(language or "unknown").strip().lower() or "unknown",
            str(format_type or "other").strip().lower() or "other",
            normalize_role(alias_role),
        ]
    )


def _matches_slice_pattern(key: str, pattern: str) -> bool:
    key_parts = str(key or "").split("/")
    pattern_parts = str(pattern or "").split("/")
    if len(key_parts) != 3 or len(pattern_parts) != 3:
        return False
    return all(pattern_part == "*" or key_part == pattern_part for key_part, pattern_part in zip(key_parts, pattern_parts))


def _collect_epoch_hard_scores(*, outputs, scalar_target, sample_indices, store: dict[int, float]) -> None:
    scalar_pred = outputs.get("scalar_logits")
    if scalar_pred is None:
        return
    torch = __import__("torch")
    pred = torch.sigmoid(scalar_pred).detach()
    err = torch.abs(pred - scalar_target).mean(dim=1).detach().cpu().numpy()
    idx_arr = sample_indices.detach().cpu().numpy()
    for s_idx, score in zip(idx_arr.tolist(), err.tolist()):
        i = int(s_idx)
        value = float(score)
        prev = store.get(i)
        if prev is None or value > prev:
            store[i] = value


def _build_hard_case_boosts(*, size: int, score_by_index: dict[int, float], top_ratio: float, boost: float) -> np.ndarray:
    out = np.ones((max(0, int(size)),), dtype=np.float32)
    if not score_by_index or size <= 0:
        return out
    ratio = max(0.01, min(0.90, float(top_ratio)))
    top_k = max(1, int(round(float(size) * ratio)))
    ranked = sorted(score_by_index.items(), key=lambda item: float(item[1]), reverse=True)
    picked = ranked[:top_k]
    boost_value = max(1.0, float(boost))
    for idx, _score in picked:
        if 0 <= int(idx) < size:
            out[int(idx)] = boost_value
    return out


def _selection_score(val_metrics: dict[str, float], cfg: OtoTrainConfig) -> float:
    val_loss = max(0.0, float(val_metrics.get("val_loss", 0.0) or 0.0))
    val_anchor_mae = max(0.0, float(val_metrics.get("val_anchor_mae_ms", 0.0) or 0.0))
    pre_acc = float(val_metrics.get("val_preutterance_acc_50ms", 0.0) or 0.0)
    hard_fail = float(val_metrics.get("val_hard_failure_rate", 0.0) or 0.0)
    worst_acc = float(val_metrics.get("val_worst_voicebank_preutterance_acc_50ms", 0.0) or 0.0)
    worst_hard_fail = float(val_metrics.get("val_worst_voicebank_hard_failure_rate", 0.0) or 0.0)
    target_acc = float(getattr(cfg, "selection_worst_voicebank_target_acc50", 0.50))
    target_pre_acc = float(getattr(cfg, "selection_preutterance_target_acc50", 0.55))
    acc_gap = max(0.0, target_acc - worst_acc)
    pre_gap = max(0.0, target_pre_acc - pre_acc)
    return (
        val_loss * float(getattr(cfg, "selection_val_loss_weight", 1.0))
        + (val_anchor_mae / 1000.0) * float(getattr(cfg, "selection_anchor_mae_weight", 0.0))
        + hard_fail * float(getattr(cfg, "selection_hard_failure_weight", 2.5))
        + acc_gap * float(getattr(cfg, "selection_worst_voicebank_weight", 1.5))
        + worst_hard_fail * float(getattr(cfg, "selection_worst_voicebank_hard_failure_weight", 0.0))
        + pre_gap * float(getattr(cfg, "selection_preutterance_gap_weight", 0.0))
    )


def _confidence_raw_scores(*, outputs, duration_ms_np: np.ndarray, error_scale_ms: float) -> list[float]:
    torch = __import__("torch")
    heat_probs = torch.sigmoid(outputs["heatmap_logits"]).detach()
    # Per-anchor peak confidence across time, then averaged.
    heat_anchor_peak = torch.amax(heat_probs, dim=1)
    heat_conf = heat_anchor_peak.mean(dim=1).detach().cpu().numpy().astype(np.float32)
    conf_logits = outputs.get("confidence_logits")
    if conf_logits is not None:
        conf_head = torch.sigmoid(conf_logits).detach().cpu().numpy().astype(np.float32)
    else:
        conf_head = np.zeros((heat_conf.shape[0],), dtype=np.float32)
    scalar_logvar = outputs.get("scalar_logvar")
    if scalar_logvar is not None:
        std = np.sqrt(np.exp(scalar_logvar.detach().cpu().numpy().astype(np.float32)))
        pred_err_ms = std.mean(axis=1) * np.maximum(duration_ms_np.astype(np.float32), 1.0)
        uncertainty = np.exp(-pred_err_ms / max(1.0, float(error_scale_ms)))
        raw = (heat_conf * 0.30) + (conf_head * 0.35) + (uncertainty * 0.35)
    else:
        raw = (heat_conf * 0.55) + (conf_head * 0.45)
    return [float(np.clip(v, 0.0, 1.0)) for v in raw.tolist()]


def _derive_confidence_calibration(
    *,
    confidence_raw: list[float],
    pre_errors_ms: list[float],
    cfg: OtoTrainConfig,
) -> dict[str, float]:
    target_thr = float(getattr(cfg, "confidence_calibration_low_conf_target", 0.58))
    err_gate = float(getattr(cfg, "confidence_calibration_error_gate_ms", 80.0))
    good_err = float(getattr(cfg, "confidence_calibration_good_error_ms", 50.0))
    out = {
        "scale": 1.0,
        "bias": 0.0,
        "low_threshold": target_thr,
        "error_threshold_ms": err_gate,
        "good_mean": 0.0,
        "bad_mean": 0.0,
    }
    if (
        not bool(getattr(cfg, "enable_confidence_calibration", True))
        or len(confidence_raw) <= 24
        or len(confidence_raw) != len(pre_errors_ms)
    ):
        return out
    conf = np.asarray(confidence_raw, dtype=np.float32)
    errs = np.asarray(pre_errors_ms, dtype=np.float32)
    good_mask = errs <= float(good_err)
    bad_mask = np.logical_not(good_mask)
    if int(np.sum(good_mask)) < 20 or int(np.sum(bad_mask)) < 20:
        return out
    good_conf = conf[good_mask]
    bad_conf = conf[bad_mask]
    mu_good = float(np.mean(good_conf))
    mu_bad = float(np.mean(bad_conf))
    out["good_mean"] = mu_good
    out["bad_mean"] = mu_bad
    sep = max(0.02, mu_good - mu_bad)
    scale = _clamp(0.50 / sep, 0.50, 3.00)
    mid = (mu_good + mu_bad) * 0.5
    bias = target_thr - (scale * mid)
    cal_conf = np.clip((conf * scale) + bias, 0.0, 1.0)
    high_err_conf = cal_conf[errs >= float(err_gate)]
    low_thr = float(target_thr)
    if high_err_conf.size >= 20:
        low_thr = float(np.quantile(high_err_conf, 0.80))
        low_thr = _clamp(low_thr, max(0.40, target_thr), 0.88)
    out["scale"] = float(scale)
    out["bias"] = float(bias)
    out["low_threshold"] = float(low_thr)
    return out


def _relative_anchor_predictions(
    pred_scalar,
    duration_ms,
    format_ids,
    alias_ids,
    transition_ids,
    role_ids,
    extra_flags,
    model_cfg: OtoCrnnConfig,
) -> np.ndarray:
    formats = list(model_cfg.format_types)
    aliases = list(model_cfg.alias_types)
    transitions = list(model_cfg.transition_types)
    roles = list(model_cfg.alias_roles)
    use_role = bool(getattr(model_cfg, "enable_alias_role_embedding", False)) and len(roles) > 1
    out = np.zeros((len(pred_scalar), len(OTO_ANCHOR_NAMES)), dtype=np.float32)
    for idx, (scalar_row, duration, fmt_idx, alias_idx, transition_idx, role_idx, flags_row) in enumerate(
        zip(pred_scalar, duration_ms, format_ids, alias_ids, transition_ids, role_ids, extra_flags)
    ):
        fmt_i = int(fmt_idx)
        alias_i = int(alias_idx)
        transition_i = int(transition_idx)
        role_i = int(role_idx)
        format_type = formats[fmt_i] if 0 <= fmt_i < len(formats) else "other"
        alias_type = aliases[alias_i] if 0 <= alias_i < len(aliases) else "other"
        transition_type = transitions[transition_i] if 0 <= transition_i < len(transitions) else "other"
        role_text = roles[role_i] if (use_role and 0 <= role_i < len(roles)) else ""
        flags_arr = np.asarray(flags_row, dtype=np.float32)
        is_diphthong = bool(flags_arr.shape[0] > 0 and flags_arr[0] >= 0.5)
        is_special = bool(flags_arr.shape[0] > 1 and flags_arr[1] >= 0.5)
        params = decode_relative_oto_params(
            scalar_row,
            duration_ms=float(duration),
            format_type=format_type,
            alias_type=alias_type,
            transition_type=transition_type,
            prior_blend=right_boundary_prior_blend_for_context(
                model_cfg,
                format_type,
                alias_role=role_text,
                is_special=bool(is_special),
            ),
            alias_role=role_text,
            is_diphthong=is_diphthong,
            is_special=is_special,
        )
        out[idx] = np.asarray(relative_params_to_anchors(params, duration_ms=float(duration)), dtype=np.float32)
    return out


def _hard_failure_flags_from_anchors(pred_anchors: np.ndarray, durations_ms: np.ndarray) -> list[bool]:
    flags: list[bool] = []
    for anchor_row, duration in zip(pred_anchors, durations_ms):
        dur = max(1.0, float(duration))
        offset = float(anchor_row[0])
        cutoff = float(anchor_row[4])
        bad = False
        if cutoff <= offset + 5.0:
            bad = True
        if offset >= dur * 0.90:
            bad = True
        if cutoff <= dur * 0.08:
            bad = True
        flags.append(bool(bad))
    return flags


def _hit_rate_np(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    t = float(threshold)
    return float(sum(1 for value in values if float(value) <= t)) / float(len(values))


def _worst_voicebank_acc50(voicebank_pre_errors: dict[int, list[float]]) -> float:
    if not voicebank_pre_errors:
        return 0.0
    worst = 1.0
    for errors in voicebank_pre_errors.values():
        acc = _hit_rate_np([float(v) for v in errors], 50.0) if errors else 0.0
        worst = min(worst, float(acc))
    return float(worst)


def _worst_voicebank_hard_failure_rate(voicebank_hard_counts: dict[int, int], voicebank_counts: dict[int, int]) -> float:
    if not voicebank_counts:
        return 0.0
    worst = 0.0
    for vb, count in voicebank_counts.items():
        denom = max(1, int(count))
        rate = float(voicebank_hard_counts.get(vb, 0)) / float(denom)
        worst = max(worst, rate)
    return float(worst)


def _apply_activity_quality_filter(rows: list[dict[str, Any]], cfg: OtoTrainConfig, *, stage_label: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    cache: dict[str, dict[str, float]] = {}
    prepared: list[tuple[dict[str, Any], bool]] = []
    kept: list[dict[str, Any]] = []
    dropped = 0
    weakened = 0
    for row in rows:
        row2 = dict(row)
        weight_mult = _row_activity_quality_multiplier(row2, cfg, cache)
        base_weight = float(row2.get("sample_weight", row2.get("weight", 1.0)) or 1.0)
        new_weight = max(0.01, base_weight * float(weight_mult))
        row2["sample_weight"] = float(new_weight)
        if new_weight < base_weight - 1e-6:
            weakened += 1
        should_drop = bool(getattr(cfg, "activity_quality_drop_low_weight", False)) and new_weight < float(
            getattr(cfg, "activity_quality_drop_threshold", 0.08)
        )
        prepared.append((row2, should_drop))

    if bool(getattr(cfg, "activity_quality_drop_low_weight", False)):
        dropped = int(sum(1 for _row, should_drop in prepared if should_drop))
        drop_ratio = float(dropped) / float(max(1, len(rows)))
        max_drop_ratio = float(getattr(cfg, "activity_quality_max_drop_ratio", 0.35) or 0.35)
        if drop_ratio > max_drop_ratio:
            print(
                f"[oto_anchor][data] {stage_label}: requested_drop_ratio={drop_ratio:.3f} exceeds max_drop_ratio={max_drop_ratio:.3f}; keep-all fallback",
                flush=True,
            )
            kept = [row2 for row2, _ in prepared]
            dropped = 0
        else:
            kept = [row2 for row2, should_drop in prepared if not should_drop]
    else:
        kept = [row2 for row2, _ in prepared]

    print(
        f"[oto_anchor][data] {stage_label}: rows={len(rows)} kept={len(kept)} dropped={dropped} "
        f"reweighted={weakened}",
        flush=True,
    )
    return kept


def _row_activity_quality_multiplier(
    row: dict[str, Any],
    cfg: OtoTrainConfig,
    cache: dict[str, dict[str, float]],
) -> float:
    wav_path = str(row.get("audio", "") or "")
    if not wav_path:
        return 1.0
    profile = _activity_profile_for_wav(
        wav_path,
        sample_rate=int(getattr(cfg, "activity_quality_sample_rate", 16000)),
        cache=cache,
    )
    if not profile:
        return 1.0
    duration = max(
        1.0,
        float(row.get("duration_ms", 0.0) or 0.0),
        float(profile.get("duration_ms", 0.0) or 0.0),
    )
    active_start = _clamp(float(profile.get("active_start_ms", 0.0) or 0.0), 0.0, duration)
    active_end = _clamp(float(profile.get("active_end_ms", duration) or duration), active_start, duration)
    lead_ratio = active_start / duration
    span_ratio = max(0.0, active_end - active_start) / duration
    mult = 1.0
    max_lead = float(getattr(cfg, "activity_leading_silence_max_ratio", 0.42))
    if lead_ratio > max_lead:
        tail = max(1e-4, 1.0 - max_lead)
        excess = min(1.0, (lead_ratio - max_lead) / tail)
        mult *= (1.0 - (0.70 * excess))
    min_span = float(getattr(cfg, "activity_min_active_span_ratio", 0.16))
    if span_ratio < min_span:
        mult *= max(0.10, span_ratio / max(min_span, 1e-4))
    # Penalize rows whose anchor midpoint lands outside the sustained active zone.
    pre = float(row.get("anchor_preutterance_ms", row.get("target_preutterance_ms", 0.0)) or 0.0)
    cons = float(row.get("anchor_consonant_ms", row.get("target_consonant_ms", 0.0)) or 0.0)
    offset = float(row.get("anchor_offset_ms", row.get("target_offset_ms", 0.0)) or 0.0)
    cutoff = float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0)
    mid = (pre + cons) * 0.5
    outside = bool(
        mid < (active_start - 100.0)
        or mid > (active_end + 100.0)
        or cutoff < (active_start - 60.0)
        or offset > (active_end + 60.0)
    )
    if outside:
        mult *= float(getattr(cfg, "activity_quality_outside_penalty", 0.35))
    return _clamp(mult, float(getattr(cfg, "activity_quality_min_multiplier", 0.10)), 1.0)


def _activity_profile_for_wav(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = os.path.abspath(str(wav_path or "")).lower()
    hit = cache.get(key)
    if hit is not None:
        return dict(hit)
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        cache[key] = {}
        return {}
    duration_ms = max(1.0, float(duration_sec) * 1000.0)
    if samples is None or int(getattr(samples, "shape", [0])[0]) <= 0 or sr <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    abs_arr = np.abs(arr)
    peak = float(np.max(abs_arr))
    if peak <= 1e-7:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    # Breath-robust energy envelope: short frame RMS + minimum run detection.
    frame = max(80, int(round(float(sr) * 0.008)))
    hop = max(40, int(round(float(sr) * 0.004)))
    if arr.size < frame:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    frames = []
    for start in range(0, int(arr.size) - frame + 1, hop):
        seg = arr[start : start + frame]
        frames.append(float(np.sqrt(np.mean(np.square(seg), dtype=np.float32))))
    env = np.asarray(frames, dtype=np.float32)
    if env.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    noise = float(np.percentile(env, 35.0))
    strong = float(np.percentile(env, 92.0))
    thr = max(noise * 2.8, strong * 0.28, 1e-5)
    active = env >= thr
    min_run_frames = max(4, int(round(0.028 / max(float(hop) / float(sr), 1e-5))))
    active = _enforce_min_run(active, min_run_frames)
    idx = np.flatnonzero(active)
    if idx.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    start_ms = (float(idx[0]) * float(hop) / float(sr)) * 1000.0
    end_ms = ((float(idx[-1]) * float(hop) + float(frame)) / float(sr)) * 1000.0
    pad_ms = 20.0
    active_start = _clamp(start_ms - pad_ms, 0.0, duration_ms)
    active_end = _clamp(end_ms + pad_ms, active_start, duration_ms)
    if active_end - active_start < 45.0:
        active_start = 0.0
        active_end = duration_ms
    profile = {
        "active_start_ms": float(active_start),
        "active_end_ms": float(active_end),
        "duration_ms": float(duration_ms),
    }
    cache[key] = profile
    return dict(profile)


def _enforce_min_run(flags: np.ndarray, min_run_frames: int) -> np.ndarray:
    arr = np.asarray(flags, dtype=bool).copy()
    if arr.size <= 0 or int(min_run_frames) <= 1:
        return arr
    run_start = None
    for idx, on in enumerate(arr.tolist() + [False]):
        if on and run_start is None:
            run_start = idx
            continue
        if (not on) and run_start is not None:
            run_len = idx - run_start
            if run_len < int(min_run_frames):
                arr[run_start:idx] = False
            run_start = None
    return arr


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _feature_cache_path(
    cache_root: str,
    wav_path: str,
    *,
    sample_rate: int,
    n_mels: int,
    frame_ms: float,
    hop_ms: float,
) -> str:
    abs_wav = os.path.abspath(str(wav_path))
    try:
        st = os.stat(abs_wav)
        size = int(st.st_size)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    except Exception:
        size = -1
        mtime_ns = -1
    key_src = {
        "v": 1,
        "wav": abs_wav.lower(),
        "size": size,
        "mtime_ns": mtime_ns,
        "sample_rate": int(sample_rate),
        "n_mels": int(n_mels),
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
    }
    packed = json.dumps(key_src, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(packed).hexdigest()
    sub = digest[:2]
    return os.path.join(cache_root, sub, f"{digest}.npz")


def _load_feature_cache(path: str) -> tuple[np.ndarray, float, float] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as obj:
            features = np.asarray(obj["features"], dtype=np.float32)
            hop_sec = float(np.asarray(obj["hop_sec"]).reshape(-1)[0])
            duration_sec = float(np.asarray(obj["duration_sec"]).reshape(-1)[0])
        return features, hop_sec, duration_sec
    except Exception:
        return None


def _save_feature_cache(path: str, features: np.ndarray, hop_sec: float, duration_sec: float) -> None:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.npz"
        np.savez_compressed(
            tmp,
            features=np.asarray(features, dtype=np.float32),
            hop_sec=np.asarray([float(hop_sec)], dtype=np.float32),
            duration_sec=np.asarray([float(duration_sec)], dtype=np.float32),
        )
        if os.path.exists(tmp):
            os.replace(tmp, path)
    except Exception:
        return None


__all__ = ["OtoAnchorDataset", "OtoTrainConfig", "train_oto_from_manifest"]
