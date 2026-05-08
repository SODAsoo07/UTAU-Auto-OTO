from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
from core.coarse_crnn.oto_model import OtoCrnnConfig, load_oto_checkpoint
from core.coarse_crnn.oto_targets import read_jsonl
from core.coarse_crnn.oto_training import OtoTrainConfig, train_oto_from_manifest


def _row_cap_key(row: dict) -> tuple[str, str, str]:
    language = str(row.get("language") or row.get("lang") or "unknown").strip().lower() or "unknown"
    format_type = str(row.get("format_type") or row.get("format") or "unknown").strip().lower() or "unknown"
    role = str(row.get("alias_role") or row.get("alias_type") or "other").strip().lower() or "other"
    return language, format_type, role


def _cap_rows_stratified(rows: list[dict], limit: int) -> list[dict]:
    """Deterministically cap sorted manifests without losing later language/role slices."""
    if limit <= 0 or len(rows) <= limit:
        return list(rows)

    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_row_cap_key(row)].append(row)

    keys = sorted(buckets.keys())
    positions = {key: 0 for key in keys}
    selected: list[dict] = []
    active = list(keys)
    while active and len(selected) < limit:
        next_active: list[tuple[str, str, str]] = []
        for key in active:
            position = positions[key]
            bucket = buckets[key]
            if position < len(bucket):
                selected.append(bucket[position])
                positions[key] = position + 1
                if positions[key] < len(bucket):
                    next_active.append(key)
                if len(selected) >= limit:
                    break
        active = next_active
    return selected


def _merge_sampling_boost_rules(raw: str, key: str, value: float | None) -> str:
    text = str(raw or "").strip()
    if value is None:
        return text
    v = float(value)
    if v <= 0.0:
        return text
    rule = f"{str(key).strip().lower()}={v:.6g}"
    if not text:
        return rule
    entries = [item.strip() for item in text.split(",") if item.strip()]
    prefix = f"{str(key).strip().lower()}="
    if any(item.lower().startswith(prefix) for item in entries):
        return text
    entries.append(rule)
    return ",".join(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CRNN-based OTO parameter predictor.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--val-manifest", default="", help="Optional fixed validation manifest, preferably voicebank-held-out.")
    parser.add_argument("--out", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--resume-model", default="", help="Optional checkpoint path to continue fine-tuning from.")
    parser.add_argument(
        "--resume-strategy",
        choices=("strict", "compatible"),
        default="strict",
        help="strict requires identical checkpoint architecture; compatible loads only same-name/same-shape tensors.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0, help="0 keeps single-process loading, -1 means auto worker count.")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    # Compatibility aliases (older local command snippets).
    parser.add_argument("--loader-prefetch-factor", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loader-persistent-workers", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-feature-cache-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cuda-prefetch-batches", type=int, default=0,
                        help="Number of batches to prefetch to CUDA on a side stream (0 disables).")
    parser.add_argument("--cuda-prefetch-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply CUDA prefetch in validation/evaluation loops as well.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-cache-dir", default=os.path.join("ml_workspace", "coarse_crnn", "feature_cache"))
    parser.add_argument("--feature-cache-readonly", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional stratified deterministic cap for smoke tests.")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Optional stratified deterministic cap for fixed validation rows.")
    parser.add_argument("--vcv-loss-weight", type=float, default=1.35)
    parser.add_argument("--cvvc-loss-weight", type=float, default=1.15)
    parser.add_argument("--cvc-loss-weight", type=float, default=1.05)
    parser.add_argument("--vc-role-loss-weight", type=float, default=2.5,
                        help="Loss multiplier for vc-role aliases (hard to predict in cvvc voicebanks).")
    parser.add_argument("--vv-role-loss-weight", type=float, default=2.0,
                        help="Loss multiplier for vv-role aliases.")
    parser.add_argument(
        "--row-order-violation-alpha",
        type=float,
        default=0.0,
        help="Downweight rows whose row_ratio mismatches the actual target offset (v2 1-B). "
             "0 disables the multiplier (legacy behaviour). Recommended starting value: 0.5; "
             "compare 0.5 vs 1.0 after a 1-2 week run. Requires a manifest built with the "
             "row_order_violation_score column populated.",
    )
    parser.add_argument("--format-residual-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vcv-target-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcv-window-frames", type=int, default=240)
    # cvvc excluded: vc/vv aliases are not uniformly distributed across audio
    # (CV rows come first, VC rows later), so row_ratio-based windowing yields
    # 1000-2500 ms offset errors. The heatmap finds the correct position instead.
    parser.add_argument("--target-window-formats", default="vcv")
    parser.add_argument("--target-window-frame-overrides", default="vcv=240")
    parser.add_argument("--target-window-role-frame-overrides", default="")
    parser.add_argument("--cvvc-target-window-alias-types", default="")
    parser.add_argument("--cvvc-target-window-alias-roles", default="vc,vv")
    parser.add_argument("--anchor-heatmap-blend", type=float, default=0.70)
    parser.add_argument("--vcv-window-heatmap-blend", type=float, default=0.30)
    parser.add_argument(
        "--scalar-target-mode",
        choices=("relative_params", "absolute_anchors"),
        default="relative_params",
        help="relative_params predicts offset/pre deltas plus constrained consonant/cutoff tail gaps.",
    )
    parser.add_argument("--right-boundary-prior-blend", type=float, default=0.45)
    parser.add_argument("--right-boundary-prior-blends", default="vcv=0.45,cvvc=0.25,cv=0.10,cvc=0.10,other=0.10")
    parser.add_argument(
        "--right-boundary-prior-role-blends",
        default="-cv=0.35,cv=0.30,cv-=0.25,v=0.18,v-=0.22,vc=0.20,vv=0.24,v-cv=0.32,endbr=0.18,br=0.15,special=0.28,other=0.30",
    )
    parser.add_argument("--uncertainty-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.30)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.10)
    parser.add_argument("--confidence-target-error-scale", type=float, default=0.08)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voicebank-balance-power", type=float, default=0.55)
    parser.add_argument("--language-balance-power", type=float, default=0.25,
                        help="Weighted sampler language balancing power. Use >0 to keep Korean/Japanese mix explicit.")
    parser.add_argument("--role-balance-power", type=float, default=0.30)
    parser.add_argument("--format-balance-power", type=float, default=0.35)
    parser.add_argument("--language-format-role-sampling-boosts", default="",
                        help="Comma-separated slash-pattern sampler boosts, e.g. korean/cvvc/*=1.8,korean/cvvc/vc=2.2.")
    parser.add_argument("--cvvc-vc-sampling-boost", type=float, default=1.0,
                        help="Weighted sampler boost for cvvc|vc|* rows.")
    parser.add_argument("--cvvc-vv-sampling-boost", type=float, default=1.0,
                        help="Weighted sampler boost for cvvc|vv|* rows.")
    parser.add_argument("--cvvc-vc-multi-sampling-boost", type=float, default=1.0,
                        help="Extra weighted sampler boost for cvvc|vc|multi rows.")
    # Compatibility aliases for older local command snippets.
    parser.add_argument("--cvvc-vc-multi-loss-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cvvc-vv-loss-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--korean-vcv-head-loss-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--korean-vcv-vcv-loss-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cvvc-specialist-head", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cvvc-specialist-gain", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hard-case-mining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-case-top-ratio", type=float, default=0.25)
    parser.add_argument("--hard-case-boost", type=float, default=2.5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--selection-val-loss-weight", type=float, default=0.15)
    parser.add_argument("--selection-anchor-mae-weight", type=float, default=0.90)
    parser.add_argument("--selection-preutterance-gap-weight", type=float, default=2.0)
    parser.add_argument("--selection-hard-failure-weight", type=float, default=2.5)
    parser.add_argument("--selection-worst-voicebank-weight", type=float, default=1.5)
    parser.add_argument("--selection-worst-voicebank-hard-failure-weight", type=float, default=2.0)
    parser.add_argument("--selection-preutterance-target-acc50", type=float, default=0.55)
    parser.add_argument("--selection-worst-voicebank-target-acc50", type=float, default=0.50)
    parser.add_argument("--activity-quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activity-leading-silence-max-ratio", type=float, default=0.42)
    parser.add_argument("--activity-min-active-span-ratio", type=float, default=0.16)
    parser.add_argument("--activity-quality-outside-penalty", type=float, default=0.35)
    parser.add_argument("--activity-quality-min-multiplier", type=float, default=0.10)
    parser.add_argument("--activity-quality-drop-low-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activity-quality-drop-threshold", type=float, default=0.08)
    parser.add_argument("--activity-quality-max-drop-ratio", type=float, default=0.35)
    parser.add_argument("--activity-quality-sample-rate", type=int, default=16000)
    parser.add_argument("--confidence-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence-calibration-good-error-ms", type=float, default=50.0)
    parser.add_argument("--confidence-calibration-low-conf-target", type=float, default=0.58)
    parser.add_argument("--confidence-calibration-error-gate-ms", type=float, default=80.0)
    parser.add_argument("--checkpoint-save-every-epochs", type=int, default=0, help=">0이면 N epoch마다 중간 체크포인트 저장")
    parser.add_argument("--two-stage-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-stage-refine-window-frames", type=int, default=320)
    parser.add_argument(
        "--alias-role-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add alias_role embeddings (-CV/CV/CV-/V/V-/VC/VV/V-CV/EndBR/BR/OTHER/special) to the model conditioning. New checkpoint architecture; not loadable by older inference paths without role plumbing.",
    )
    parser.add_argument(
        "--extra-alias-flags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a small projection that ingests is_diphthong and is_special as scalar conditioning features.",
    )
    parser.add_argument(
        "--enable-ctc-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a CTC alignment head over the BiGRU encoder for phone-level supervision (3-A in CRNN-정확도-개선-방안-v2). Old checkpoints stay loadable since the head module is only created when this flag is on.",
    )
    parser.add_argument(
        "--ctc-loss-weight",
        type=float,
        default=0.0,
        help="Multiplier for the CTC alignment auxiliary loss. 0 disables the loss even if --enable-ctc-head is set, useful for ablations. Recommended starting value: 0.3 (grid: 0.1, 0.3, 1.0).",
    )
    parser.add_argument(
        "--ctc-warmup-steps",
        type=int,
        default=0,
        help="During the first N global training steps, train ONLY the CTC alignment loss (anchor regression / heatmap / confidence losses are skipped). Lets a freshly initialized ctc_head stabilize before its random gradients knock the pretrained anchor heads off-distribution. 0 disables warmup. Suggested for warm-start smoke runs: ~150-300.",
    )
    parser.add_argument(
        "--enable-audio-onset-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append onset_ratio, active_start_ratio, continuity_score (derived from mel energy) to the context vector, expanding it from 12→15 dims. Also sets numeric_context_dim=15 on the model config.",
    )
    parser.add_argument(
        "--onset-loss-weight",
        type=float,
        default=0.50,
        help="MSE loss weight for the onset regression auxiliary head. Only has an effect when --enable-onset-head is set.",
    )
    parser.add_argument(
        "--enable-onset-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add an onset regression head to the model that predicts onset_ratio from the spectrogram pooled representation. Old checkpoints load cleanly since the head is only created when this flag is on.",
    )
    args = parser.parse_args()
    # Backward-compatible remapping of deprecated argument names.
    if args.cvvc_vc_multi_loss_weight is not None:
        args.vc_role_loss_weight = float(args.cvvc_vc_multi_loss_weight)
    if args.cvvc_vv_loss_weight is not None:
        args.vv_role_loss_weight = float(args.cvvc_vv_loss_weight)
    args.language_format_role_sampling_boosts = _merge_sampling_boost_rules(
        str(args.language_format_role_sampling_boosts or ""),
        "korean/vcv/-cv",
        args.korean_vcv_head_loss_weight,
    )
    args.language_format_role_sampling_boosts = _merge_sampling_boost_rules(
        str(args.language_format_role_sampling_boosts or ""),
        "korean/vcv/v-cv",
        args.korean_vcv_vcv_loss_weight,
    )
    if bool(args.cvvc_specialist_head or False):
        # This branch does not expose a dedicated cvvc specialist head module.
        # Use the nearest available approximation: per-format residual heads.
        args.format_residual_heads = True
    # Loader alias mapping.
    if args.loader_prefetch_factor is not None:
        args.dataloader_prefetch_factor = int(args.loader_prefetch_factor)
    if args.loader_persistent_workers is not None:
        args.dataloader_persistent_workers = bool(args.loader_persistent_workers)
    # cpu-feature-cache-size was used for in-memory cache experiments in another
    # branch. On this branch we use on-disk feature cache; map intent to enable.
    if args.cpu_feature_cache_size is not None:
        args.feature_cache = bool(int(args.cpu_feature_cache_size) > 0)

    rows = read_jsonl(args.manifest)
    if int(args.max_rows) > 0:
        rows = _cap_rows_stratified(rows, int(args.max_rows))
    val_rows = read_jsonl(args.val_manifest) if str(args.val_manifest or "").strip() else None
    if val_rows is not None and int(args.max_val_rows) > 0:
        val_rows = _cap_rows_stratified(val_rows, int(args.max_val_rows))
    model_cfg = OtoCrnnConfig(
        n_mels=int(args.n_mels),
        hidden=int(args.hidden),
        conv_channels=int(args.conv_channels),
        enable_format_residual_heads=bool(args.format_residual_heads),
        enable_vcv_target_window=bool(args.vcv_target_window),
        vcv_target_window_frames=int(args.vcv_window_frames),
        target_window_formats=tuple(item.strip().lower() for item in str(args.target_window_formats).split(",") if item.strip()),
        target_window_frame_overrides=tuple(item.strip().lower() for item in str(args.target_window_frame_overrides).split(",") if item.strip()),
        target_window_role_frame_overrides=tuple(item.strip().lower() for item in str(args.target_window_role_frame_overrides).split(",") if item.strip()),
        cvvc_target_window_alias_types=tuple(item.strip().lower() for item in str(args.cvvc_target_window_alias_types).split(",") if item.strip()),
        cvvc_target_window_alias_roles=tuple(item.strip().lower() for item in str(args.cvvc_target_window_alias_roles).split(",") if item.strip()),
        anchor_heatmap_blend=float(args.anchor_heatmap_blend),
        vcv_window_heatmap_blend=float(args.vcv_window_heatmap_blend),
        scalar_target_mode=str(args.scalar_target_mode),
        right_boundary_prior_blend=float(args.right_boundary_prior_blend),
        right_boundary_prior_blends=tuple(item.strip().lower() for item in str(args.right_boundary_prior_blends).split(",") if item.strip()),
        right_boundary_prior_role_blends=tuple(item.strip().lower() for item in str(args.right_boundary_prior_role_blends).split(",") if item.strip()),
        enable_uncertainty_head=bool(args.uncertainty_head),
        confidence_calibration_scale=1.0,
        confidence_calibration_bias=0.0,
        confidence_low_threshold=float(args.confidence_calibration_low_conf_target),
        predicted_error_low_threshold_ms=float(args.confidence_calibration_error_gate_ms),
        enable_two_stage_refine=bool(args.two_stage_refine),
        two_stage_refine_window_frames=int(args.two_stage_refine_window_frames),
        enable_alias_role_embedding=bool(args.alias_role_embedding),
        enable_extra_alias_flags=bool(args.extra_alias_flags),
        enable_ctc_head=bool(args.enable_ctc_head),
        # Pin vocab size explicitly so the saved checkpoint records the
        # resolved value (matches OtoCrnnConfig default but guards against
        # future default drift).
        ctc_phone_vocab_size=int(CTC_VOCAB_SIZE),
        numeric_context_dim=15 if bool(args.enable_audio_onset_context) else 12,
        enable_onset_head=bool(args.enable_onset_head),
    )
    train_cfg = OtoTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        dataloader_prefetch_factor=max(1, int(args.dataloader_prefetch_factor)),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
        cuda_prefetch_batches=max(0, int(args.cuda_prefetch_batches)),
        cuda_prefetch_eval=bool(args.cuda_prefetch_eval),
        enable_tf32=bool(args.tf32),
        enable_cudnn_benchmark=bool(args.cudnn_benchmark),
        enable_feature_cache=bool(args.feature_cache),
        feature_cache_dir=str(args.feature_cache_dir),
        feature_cache_readonly=bool(args.feature_cache_readonly),
        log_every=int(args.log_every),
        val_ratio=float(args.val_ratio),
        vcv_loss_weight=float(args.vcv_loss_weight),
        cvvc_loss_weight=float(args.cvvc_loss_weight),
        cvc_loss_weight=float(args.cvc_loss_weight),
        vc_role_loss_weight=float(args.vc_role_loss_weight),
        vv_role_loss_weight=float(args.vv_role_loss_weight),
        row_order_violation_alpha=float(args.row_order_violation_alpha),
        ctc_loss_weight=float(args.ctc_loss_weight),
        ctc_warmup_steps=int(args.ctc_warmup_steps),
        uncertainty_loss_weight=float(args.uncertainty_loss_weight),
        confidence_loss_weight=float(args.confidence_loss_weight),
        confidence_target_error_scale=float(args.confidence_target_error_scale),
        enable_balanced_sampling=bool(args.balanced_sampling),
        voicebank_balance_power=float(args.voicebank_balance_power),
        language_balance_power=float(args.language_balance_power),
        role_balance_power=float(args.role_balance_power),
        format_balance_power=float(args.format_balance_power),
        language_format_role_sampling_boosts=tuple(
            item.strip().lower()
            for item in str(args.language_format_role_sampling_boosts or "").split(",")
            if item.strip()
        ),
        cvvc_vc_sampling_boost=float(args.cvvc_vc_sampling_boost),
        cvvc_vv_sampling_boost=float(args.cvvc_vv_sampling_boost),
        cvvc_vc_multi_sampling_boost=float(args.cvvc_vc_multi_sampling_boost),
        enable_hard_case_mining=bool(args.hard_case_mining),
        hard_case_top_ratio=float(args.hard_case_top_ratio),
        hard_case_boost=float(args.hard_case_boost),
        early_stop_patience=int(args.early_stop_patience),
        selection_val_loss_weight=float(args.selection_val_loss_weight),
        selection_anchor_mae_weight=float(args.selection_anchor_mae_weight),
        selection_preutterance_gap_weight=float(args.selection_preutterance_gap_weight),
        selection_hard_failure_weight=float(args.selection_hard_failure_weight),
        selection_worst_voicebank_weight=float(args.selection_worst_voicebank_weight),
        selection_worst_voicebank_hard_failure_weight=float(args.selection_worst_voicebank_hard_failure_weight),
        selection_preutterance_target_acc50=float(args.selection_preutterance_target_acc50),
        selection_worst_voicebank_target_acc50=float(args.selection_worst_voicebank_target_acc50),
        enable_activity_quality_filter=bool(args.activity_quality_filter),
        activity_leading_silence_max_ratio=float(args.activity_leading_silence_max_ratio),
        activity_min_active_span_ratio=float(args.activity_min_active_span_ratio),
        activity_quality_outside_penalty=float(args.activity_quality_outside_penalty),
        activity_quality_min_multiplier=float(args.activity_quality_min_multiplier),
        activity_quality_drop_low_weight=bool(args.activity_quality_drop_low_weight),
        activity_quality_drop_threshold=float(args.activity_quality_drop_threshold),
        activity_quality_max_drop_ratio=float(args.activity_quality_max_drop_ratio),
        activity_quality_sample_rate=int(args.activity_quality_sample_rate),
        enable_confidence_calibration=bool(args.confidence_calibration),
        confidence_calibration_good_error_ms=float(args.confidence_calibration_good_error_ms),
        confidence_calibration_low_conf_target=float(args.confidence_calibration_low_conf_target),
        confidence_calibration_error_gate_ms=float(args.confidence_calibration_error_gate_ms),
        checkpoint_save_every_epochs=max(0, int(args.checkpoint_save_every_epochs)),
        enable_audio_onset_context=bool(args.enable_audio_onset_context),
        onset_loss_weight=float(args.onset_loss_weight),
    )
    init_state = None
    init_tag = ""
    if str(args.resume_model or "").strip():
        init_model, _init_cfg, _meta = load_oto_checkpoint(str(args.resume_model), map_location="cpu")
        init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        init_tag = os.path.abspath(str(args.resume_model))
    result = train_oto_from_manifest(
        rows,
        args.out,
        val_rows=val_rows,
        train_config=train_cfg,
        model_config=model_cfg,
        init_state_dict=init_state,
        init_state_strict=str(args.resume_strategy) == "strict",
        init_tag=init_tag,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
