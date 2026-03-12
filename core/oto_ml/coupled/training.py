"""
Coupled mel+OTO training.

학습 루프, 데이터 전처리, 데이터셋 빌드, 평가 로직을 포함합니다.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
    try:  # pragma: no cover
        from pandas.errors import ParserError
    except Exception:  # pragma: no cover
        ParserError = Exception
except Exception:  # pragma: no cover
    pd = None
    ParserError = Exception

try:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    mean_absolute_error = None
    GroupShuffleSplit = None

from core.format_type_utils import normalize_format_type
from core.oto_ml.coupled.model import (
    CATEGORICAL_FEATURES,
    COUPLED_BACKEND,
    COUPLED_BACKEND_RAWMEL,
    COUPLED_MODEL_FILE,
    FEATURE_NAMES,
    PATCH_FEATURES,
    TARGET_NAMES,
    _build_model,
    _build_model_rawmel,
    _env_int,
    _import_torch,
    _require_training_stack,
    _resolve_device,
    _row_feature_vector,
    _row_patch_vector,
)
from core.oto_ml.features.schema import (
    AUX_TARGET_NAMES,
    get_feature_schema,
    write_dataset_csv,
    write_feature_schema,
)
from core.oto_ml.features.mel_patches import (
    MelPatchCacheIndex,
    make_mel_patch_key,
    patch_spec_hash,
)
from core.oto_ml.pairing.vc_cv_pairing import _batch_pair_positions, _build_vc_cv_pair_map

logger = logging.getLogger(__name__)


def _read_dataset_csv(path: str):
    df = pd.read_csv(path, low_memory=False)
    return df


def _read_dataset_csv_resilient(path: str):
    try:
        return _read_dataset_csv(path)
    except Exception as exc:
        if isinstance(exc, ParserError):
            msg = f"[TRAIN] CSV parse failed, retrying with python engine (bad lines may be skipped): {exc}"
            print(msg)
            logger.warning(msg)
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="warn")
            except Exception:
                msg2 = "[TRAIN] CSV parse still failing; retrying with relaxed quoting (skipping bad lines)."
                print(msg2)
                logger.warning(msg2)
                return pd.read_csv(
                    path,
                    engine="python",
                    on_bad_lines="skip",
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                )
        raise


def _prepare_training_frame(
    df,
    language: str,
    format_type: str,
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
):
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == lang]
    if "format_type" in df.columns and fmt and fmt != "general":
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(lang, v)) == fmt
        ]
    if alias_types and "alias_type" in df.columns:
        normalized = [str(v).strip().lower() for v in alias_types if str(v).strip()]
        if normalized:
            df = df[df["alias_type"].astype(str).str.lower().isin(normalized)]
    if float(min_mapping_confidence) > 0.0 and "mapping_confidence" in df.columns:
        df = df[pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0.0) >= float(min_mapping_confidence)]
    return df


def build_and_save_coupled_dataset(
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    out_csv: str,
    custom_phonemes_path: str = "",
    voicebank_id: str = "",
    append: bool = False,
    format_type_override: str = "",
) -> Dict[str, int]:
    from core.oto_ml_features import build_training_rows

    rows, stats = build_training_rows(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
        voicebank_id=voicebank_id,
        format_type_override=format_type_override,
    )
    if rows:
        write_dataset_csv(out_csv, rows, append=append)
    out = dict(stats)
    out["saved_rows"] = int(len(rows))
    out["out_csv"] = os.path.abspath(out_csv)
    return out


def train_coupled_bundle(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
    device: str = "auto",
    epochs: int = 70,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    min_confidence: float = 0.55,
    progress_every: int = 0,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    torch, nn, F = _import_torch()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    df = _read_dataset_csv_resilient(dataset_csv)
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        min_mapping_confidence=min_mapping_confidence,
    )
    df = df.reset_index(drop=True)
    if len(df) < 16:
        raise RuntimeError("Coupled dataset is too small (need >= 16 rows).")

    schema = get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]

    x_rows = []
    p_rows = []
    for _, row in df.iterrows():
        as_dict = row.to_dict()
        x_rows.append(_row_feature_vector(as_dict, feature_names, categorical_features))
        p_rows.append(_row_patch_vector(as_dict))
    X = np.asarray(x_rows, dtype=np.float32)
    P = np.asarray(p_rows, dtype=np.float32)
    Y = np.stack(
        [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in TARGET_NAMES],
        axis=1,
    )
    use_aux = all(name in df.columns for name in AUX_TARGET_NAMES)
    if use_aux:
        A = np.stack(
            [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in AUX_TARGET_NAMES],
            axis=1,
        )
        vowel_start_valid = pd.to_numeric(df["curr_vowel_start_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        vowel_end_valid = pd.to_numeric(df["curr_vowel_end_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        next_onset_valid = (
            pd.to_numeric(df["base_cutoff_to_next_anchor_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        )
        aux_mask = np.stack(
            [vowel_start_valid.astype(np.float32), vowel_end_valid.astype(np.float32), next_onset_valid.astype(np.float32)],
            axis=1,
        )
    else:
        A = None
        aux_mask = None
    base = np.stack(
        [
            pd.to_numeric(df["base_offset"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cons"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cutoff_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_pre"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_ovl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    boundary = np.stack(
        [
            pd.to_numeric(df["mel_offset_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["mel_cutoff_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    if "sample_weight" in df.columns:
        W = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    else:
        W = np.ones((len(df),), dtype=np.float32)

    if group_column in df.columns and df[group_column].nunique() >= 2 and GroupShuffleSplit is not None:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(split.split(df, groups=df[group_column]))
    else:
        split_at = max(1, int(len(df) * 0.8))
        train_idx = np.arange(split_at)
        valid_idx = np.arange(split_at, len(df))
        if len(valid_idx) <= 0:
            valid_idx = np.arange(max(0, len(df) - 1), len(df))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    pair_map = _build_vc_cv_pair_map(df)
    pair_weight = float(os.environ.get("UTOA_ML_VC_CV_PAIR_WEIGHT", 0.06) or 0.06)
    if pair_weight < 0.0:
        pair_weight = 0.0
    # Bidirectional pairing doubles constraints; halve the weight to keep parity.
    pair_weight = pair_weight * 0.5
    if pair_map:
        train_mask = np.zeros((len(df),), dtype=bool)
        valid_mask = np.zeros((len(df),), dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        pair_train_count = int(sum(1 for a, b in pair_map.items() if train_mask[a] and train_mask[b]))
        pair_valid_count = int(sum(1 for a, b in pair_map.items() if valid_mask[a] and valid_mask[b]))
        pair_total_count = int(len(pair_map))
        global_to_train = {int(g): i for i, g in enumerate(train_idx.tolist())}
        global_to_valid = {int(g): i for i, g in enumerate(valid_idx.tolist())}
        pair_map_train = {
            int(global_to_train[a]): int(global_to_train[b])
            for a, b in pair_map.items()
            if a in global_to_train and b in global_to_train
        }
        pair_map_valid = {
            int(global_to_valid[a]): int(global_to_valid[b])
            for a, b in pair_map.items()
            if a in global_to_valid and b in global_to_valid
        }
        val_src_pos = list(pair_map_valid.keys())
        val_dst_pos = [pair_map_valid[k] for k in val_src_pos]
    else:
        pair_train_count = 0
        pair_valid_count = 0
        pair_total_count = 0
        pair_map_train = {}
        pair_map_valid = {}
        val_src_pos = []
        val_dst_pos = []

    aux_dim = len(AUX_TARGET_NAMES) if use_aux else 0
    model = _build_model(torch, nn, in_dim=int(X.shape[1]), patch_dim=int(P.shape[1]), aux_dim=aux_dim)
    run_device = _resolve_device(torch, requested=device)
    if isinstance(run_device, str):
        run_device = torch.device(run_device)
    model = model.to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=run_device)
    P_train = torch.tensor(P[train_idx], dtype=torch.float32, device=run_device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32, device=run_device)
    B_train = torch.tensor(base[train_idx], dtype=torch.float32, device=run_device)
    M_train = torch.tensor(boundary[train_idx], dtype=torch.float32, device=run_device)
    W_train = torch.tensor(W[train_idx], dtype=torch.float32, device=run_device)

    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32, device=run_device)
    P_valid = torch.tensor(P[valid_idx], dtype=torch.float32, device=run_device)
    Y_valid = torch.tensor(Y[valid_idx], dtype=torch.float32, device=run_device)
    B_valid = torch.tensor(base[valid_idx], dtype=torch.float32, device=run_device)
    M_valid = torch.tensor(boundary[valid_idx], dtype=torch.float32, device=run_device)
    W_valid = torch.tensor(W[valid_idx], dtype=torch.float32, device=run_device)
    if use_aux:
        A_train = torch.tensor(A[train_idx], dtype=torch.float32, device=run_device)
        AM_train = torch.tensor(aux_mask[train_idx], dtype=torch.float32, device=run_device)
        A_valid = torch.tensor(A[valid_idx], dtype=torch.float32, device=run_device)
        AM_valid = torch.tensor(aux_mask[valid_idx], dtype=torch.float32, device=run_device)
    else:
        A_train = None
        AM_train = None
        A_valid = None
        AM_valid = None

    target_weights = torch.tensor([1.00, 0.90, 0.95, 1.00, 0.65], dtype=torch.float32, device=run_device).view(1, -1)
    cons_margin = 10.0
    cut_margin = 10.0
    aux_weight = 0.08

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 10

    train_n = int(X_train.shape[0])
    batch_n = max(1, int(batch_size))
    epochs_n = max(1, int(epochs))
    progress_every = int(progress_every)
    total_batches = max(1, int((train_n + batch_n - 1) / batch_n))
    if progress_every > 0:
        print(f"[TRAIN] rows={train_n} batches={total_batches} epochs={epochs_n} device={run_device}")

    for epoch in range(epochs_n):
        model.train()
        perm = torch.randperm(train_n, device=run_device)
        for batch_i, start in enumerate(range(0, train_n, batch_n), start=1):
            batch_idx = perm[start:start + batch_n]
            xb = X_train[batch_idx]
            pb = P_train[batch_idx]
            yb = Y_train[batch_idx]
            bb = B_train[batch_idx]
            mb = M_train[batch_idx]
            wb = W_train[batch_idx]

            out = model(xb, pb)
            if use_aux:
                pred, conf, aux_pred = out
            else:
                pred, conf = out
                aux_pred = None
            base_loss = F.smooth_l1_loss(pred, yb, reduction="none")
            base_loss = torch.mean(base_loss * target_weights, dim=1)
            base_loss = torch.mean(base_loss * wb)

            offset = bb[:, 0] + pred[:, 0]
            consonant = bb[:, 1] + pred[:, 1]
            cutoff_abs = bb[:, 2] + pred[:, 2]
            pre = bb[:, 3] + pred[:, 3]
            ovl = bb[:, 4] + pred[:, 4]
            penalty = (
                torch.relu(ovl - pre)
                + torch.relu((pre + cons_margin) - consonant)
                + torch.relu((consonant + cut_margin) - cutoff_abs)
                + torch.relu(-offset)
            )
            penalty_loss = torch.mean(penalty * wb)

            align_loss = F.smooth_l1_loss(offset, mb[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs, mb[:, 1], reduction="none"
            )
            align_loss = torch.mean(align_loss * wb)

            err = torch.mean(torch.abs(pred - yb), dim=1)
            conf_target = torch.exp(-torch.clamp(err / 80.0, min=0.0, max=8.0))
            conf_loss = F.binary_cross_entropy(conf.squeeze(1), conf_target.detach(), reduction="none")
            conf_loss = torch.mean(conf_loss * wb)

            aux_loss = 0.0
            if use_aux and aux_pred is not None and A_train is not None and AM_train is not None:
                ab = A_train[batch_idx]
                am = AM_train[batch_idx]
                aux_err = F.smooth_l1_loss(aux_pred, ab, reduction="none")
                mask_sum = torch.clamp(torch.sum(am, dim=1), min=1.0)
                aux_err = torch.sum(aux_err * am, dim=1) / mask_sum
                aux_loss = torch.mean(aux_err * wb)

            pair_loss = 0.0
            if pair_weight > 0.0 and pair_map_train:
                batch_indices = [int(i) for i in batch_idx.detach().cpu().tolist()]
                src_pos, dst_pos = _batch_pair_positions(batch_indices, pair_map_train)
                if src_pos:
                    src_t = torch.tensor(src_pos, device=run_device, dtype=torch.long)
                    dst_t = torch.tensor(dst_pos, device=run_device, dtype=torch.long)
                    pred_offset = bb[:, 0] + pred[:, 0]
                    true_offset = bb[:, 0] + yb[:, 0]
                    pred_gap = pred_offset[dst_t] - pred_offset[src_t]
                    true_gap = true_offset[dst_t] - true_offset[src_t]
                    pair_err = F.smooth_l1_loss(pred_gap, true_gap, reduction="none")
                    pair_w = 0.5 * (wb[src_t] + wb[dst_t])
                    pair_loss = torch.mean(pair_err * pair_w)

            total_loss = (
                base_loss
                + (0.25 * penalty_loss)
                + (0.12 * align_loss)
                + (0.05 * conf_loss)
                + (aux_weight * aux_loss)
                + (pair_weight * pair_loss)
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if progress_every > 0 and (batch_i % progress_every == 0 or batch_i == total_batches):
                loss_val = float(total_loss.detach().cpu().item())
                print(f"[TRAIN] epoch={epoch + 1}/{epochs_n} batch={batch_i}/{total_batches} loss={loss_val:.4f}")

        model.eval()
        with torch.no_grad():
            out_val = model(X_valid, P_valid)
            if use_aux:
                pred_val, conf_val, aux_val = out_val
            else:
                pred_val, conf_val = out_val
                aux_val = None
            val_base = F.smooth_l1_loss(pred_val, Y_valid, reduction="none")
            val_base = torch.mean(val_base * target_weights, dim=1)
            val_base = torch.mean(val_base * W_valid)
            offset_v = B_valid[:, 0] + pred_val[:, 0]
            consonant_v = B_valid[:, 1] + pred_val[:, 1]
            cutoff_abs_v = B_valid[:, 2] + pred_val[:, 2]
            pre_v = B_valid[:, 3] + pred_val[:, 3]
            ovl_v = B_valid[:, 4] + pred_val[:, 4]
            val_penalty = (
                torch.relu(ovl_v - pre_v)
                + torch.relu((pre_v + cons_margin) - consonant_v)
                + torch.relu((consonant_v + cut_margin) - cutoff_abs_v)
                + torch.relu(-offset_v)
            )
            val_penalty = torch.mean(val_penalty * W_valid)
            val_align = F.smooth_l1_loss(offset_v, M_valid[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs_v, M_valid[:, 1], reduction="none"
            )
            val_align = torch.mean(val_align * W_valid)
            err_v = torch.mean(torch.abs(pred_val - Y_valid), dim=1)
            conf_target_v = torch.exp(-torch.clamp(err_v / 80.0, min=0.0, max=8.0))
            val_conf = F.binary_cross_entropy(conf_val.squeeze(1), conf_target_v.detach(), reduction="none")
            val_conf = torch.mean(val_conf * W_valid)
            val_aux = 0.0
            if use_aux and aux_val is not None and A_valid is not None and AM_valid is not None:
                aux_err_v = F.smooth_l1_loss(aux_val, A_valid, reduction="none")
                mask_sum_v = torch.clamp(torch.sum(AM_valid, dim=1), min=1.0)
                aux_err_v = torch.sum(aux_err_v * AM_valid, dim=1) / mask_sum_v
                val_aux = torch.mean(aux_err_v * W_valid)
            val_pair = 0.0
            if pair_weight > 0.0 and val_src_pos:
                src_t = torch.tensor(val_src_pos, device=run_device, dtype=torch.long)
                dst_t = torch.tensor(val_dst_pos, device=run_device, dtype=torch.long)
                pred_offset_v = B_valid[:, 0] + pred_val[:, 0]
                true_offset_v = B_valid[:, 0] + Y_valid[:, 0]
                pred_gap_v = pred_offset_v[dst_t] - pred_offset_v[src_t]
                true_gap_v = true_offset_v[dst_t] - true_offset_v[src_t]
                pair_err_v = F.smooth_l1_loss(pred_gap_v, true_gap_v, reduction="none")
                pair_w_v = 0.5 * (W_valid[src_t] + W_valid[dst_t])
                val_pair = torch.mean(pair_err_v * pair_w_v)
            val_total = float(
                (
                    val_base
                    + (0.25 * val_penalty)
                    + (0.12 * val_align)
                    + (0.05 * val_conf)
                    + (aux_weight * val_aux)
                    + (pair_weight * val_pair)
                ).item()
            )

        new_best = False
        if val_total < best_val:
            best_val = val_total
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            new_best = True
        else:
            wait += 1
            if wait >= patience:
                break
        if progress_every > 0:
            status = "best" if new_best else "wait"
            print(
                f"[TRAIN] epoch={epoch + 1}/{epochs_n} val_loss={val_total:.4f} best={best_val:.4f} "
                f"patience={wait}/{patience} status={status}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_valid = model(X_valid, P_valid)
        if use_aux:
            pred_valid, conf_valid, aux_valid = out_valid
        else:
            pred_valid, conf_valid = out_valid
            aux_valid = None
    pred_valid_np = pred_valid.detach().cpu().numpy()
    conf_valid_np = conf_valid.detach().cpu().numpy().reshape(-1)
    truth_valid_np = Y[valid_idx]

    metrics = {}
    for col_i, target in enumerate(TARGET_NAMES):
        truth = truth_valid_np[:, col_i]
        pred = pred_valid_np[:, col_i]
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    aux_metrics = {}
    if use_aux and aux_valid is not None and A is not None and aux_mask is not None:
        aux_pred_np = aux_valid.detach().cpu().numpy()
        aux_truth_np = A[valid_idx]
        aux_mask_np = aux_mask[valid_idx]
        for col_i, target in enumerate(AUX_TARGET_NAMES):
            mask = aux_mask_np[:, col_i] > 0.5
            if np.any(mask):
                aux_metrics[target] = {
                    "rows": int(np.sum(mask)),
                    "model_mae": float(mean_absolute_error(aux_truth_np[mask, col_i], aux_pred_np[mask, col_i])),
                }
            else:
                aux_metrics[target] = {"rows": 0, "model_mae": 0.0}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "target_names": list(TARGET_NAMES),
            "patch_features": list(PATCH_FEATURES),
            "in_dim": int(X.shape[1]),
            "patch_dim": int(P.shape[1]),
            "hidden_dim": 160,
            "aux_dim": int(aux_dim),
            "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))

    meta = {
        "backend": COUPLED_BACKEND,
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v1",
        "feature_version": schema.get("feature_version", ""),
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "targets": list(TARGET_NAMES),
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": list(PATCH_FEATURES),
        "min_confidence": float(min_confidence),
        "vc_cv_pair_weight": float(pair_weight),
        "vc_cv_pair_max_gap": int(os.environ.get("UTOA_ML_VC_CV_MAX_GAP", 5) or 5),
        "vc_cv_pairs_total": int(pair_total_count),
        "vc_cv_pairs_train": int(pair_train_count),
        "vc_cv_pairs_valid": int(pair_valid_count),
        "fallback_order": [COUPLED_BACKEND, "lightgbm", "base"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
        "aux_holdout_metrics": aux_metrics,
        "holdout_confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
        "device_used": str(run_device),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "aux_metrics": aux_metrics,
                "confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_min": float(np.min(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_max": float(np.max(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "vc_cv_pair_weight": float(pair_weight),
                "vc_cv_pairs_total": int(pair_total_count),
                "vc_cv_pairs_valid": int(pair_valid_count),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta


def _frames_from_spec(window_ms: List[float], hop_ms: float) -> int:
    if not window_ms or len(window_ms) != 2:
        return 1
    span = float(window_ms[1]) - float(window_ms[0])
    return max(1, int(round(span / float(hop_ms))) + 1)


def train_coupled_bundle_rawmel(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    rawmel_cache_dir: str,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
    device: str = "auto",
    epochs: int = 70,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    min_confidence: float = 0.55,
    progress_every: int = 0,
    rawmel_prefetch: str = "none",
    rawmel_max_shard_cache: int = 2,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)
    if not rawmel_cache_dir or not os.path.isdir(rawmel_cache_dir):
        raise FileNotFoundError(rawmel_cache_dir)

    torch, nn, F = _import_torch()
    df = _read_dataset_csv_resilient(dataset_csv)
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        min_mapping_confidence=min_mapping_confidence,
    )
    df = df.reset_index(drop=True)
    if len(df) < 16:
        raise RuntimeError("Coupled dataset is too small (need >= 16 rows).")

    required_key_cols = [
        "language",
        "format_type",
        "voicebank_id",
        "wav_norm",
        "alias_norm",
        "occurrence_index",
        "row_index_in_wav",
    ]
    if "mel_patch_key" not in df.columns:
        missing = [c for c in required_key_cols if c not in df.columns]
        if missing:
            raise RuntimeError(
                "Dataset is missing mel_patch_key and required columns: "
                + ", ".join(missing)
                + " (rebuild dataset CSV)."
            )
        print("[TRAIN] mel_patch_key missing; computing from columns.")
        def _build_patch_key(row):
            row_lang = str(row.get("language", "") or "").strip().lower()
            row_fmt = normalize_format_type(row_lang, row.get("format_type", "")) or str(row.get("format_type", "") or "").strip().lower()
            return make_mel_patch_key(
                language=row_lang,
                format_type=row_fmt,
                voicebank_id=str(row.get("voicebank_id", "") or ""),
                wav_norm=str(row.get("wav_norm", "") or ""),
                alias_norm=str(row.get("alias_norm", "") or ""),
                occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
                row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
            )
        df["mel_patch_key"] = df.apply(_build_patch_key, axis=1)
    else:
        # Fill empty keys if present
        key_series = df["mel_patch_key"].astype(str)
        empty_mask = key_series.str.strip() == ""
        if empty_mask.any():
            missing = [c for c in required_key_cols if c not in df.columns]
            if missing:
                raise RuntimeError(
                    "Dataset has empty mel_patch_key but missing required columns: "
                    + ", ".join(missing)
                    + " (rebuild dataset CSV)."
                )
            print(f"[TRAIN] mel_patch_key empty rows={int(empty_mask.sum())}; computing from columns.")
            def _build_patch_key(row):
                row_lang = str(row.get("language", "") or "").strip().lower()
                row_fmt = normalize_format_type(row_lang, row.get("format_type", "")) or str(row.get("format_type", "") or "").strip().lower()
                return make_mel_patch_key(
                    language=row_lang,
                    format_type=row_fmt,
                    voicebank_id=str(row.get("voicebank_id", "") or ""),
                    wav_norm=str(row.get("wav_norm", "") or ""),
                    alias_norm=str(row.get("alias_norm", "") or ""),
                    occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
                    row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
                )
            df.loc[empty_mask, "mel_patch_key"] = df[empty_mask].apply(_build_patch_key, axis=1)

    cache_index = MelPatchCacheIndex.load(rawmel_cache_dir, max_shard_cache=int(rawmel_max_shard_cache))
    patch_spec = cache_index.patch_spec or {}
    hop_ms = float(patch_spec.get("frame_hop_ms", 5.0))
    mel_bins = int(patch_spec.get("mel_bins", 80))
    onset_frames = _frames_from_spec(patch_spec.get("onset_window_ms", [0.0, 0.0]), hop_ms)
    tail_frames = _frames_from_spec(patch_spec.get("tail_window_ms", [0.0, 0.0]), hop_ms)
    patch_hash = patch_spec_hash(patch_spec)

    keys = df["mel_patch_key"].astype(str).tolist()
    missing_keys = [k for k in keys if not cache_index.has_key(k)]
    if missing_keys:
        raise RuntimeError(f"Raw mel cache missing keys (count={len(missing_keys)}).")

    schema = get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]

    x_rows = []
    p_rows = []
    for _, row in df.iterrows():
        as_dict = row.to_dict()
        x_rows.append(_row_feature_vector(as_dict, feature_names, categorical_features))
        p_rows.append(_row_patch_vector(as_dict))
    X = np.asarray(x_rows, dtype=np.float32)
    P = np.asarray(p_rows, dtype=np.float32)
    Y = np.stack(
        [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in TARGET_NAMES],
        axis=1,
    )
    use_aux = all(name in df.columns for name in AUX_TARGET_NAMES)
    if use_aux:
        A = np.stack(
            [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in AUX_TARGET_NAMES],
            axis=1,
        )
        vowel_start_valid = pd.to_numeric(df["curr_vowel_start_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        vowel_end_valid = pd.to_numeric(df["curr_vowel_end_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        next_onset_valid = (
            pd.to_numeric(df["base_cutoff_to_next_anchor_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        )
        aux_mask = np.stack(
            [vowel_start_valid.astype(np.float32), vowel_end_valid.astype(np.float32), next_onset_valid.astype(np.float32)],
            axis=1,
        )
    else:
        A = None
        aux_mask = None
    base = np.stack(
        [
            pd.to_numeric(df["base_offset"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cons"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cutoff_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_pre"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_ovl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    boundary = np.stack(
        [
            pd.to_numeric(df["mel_offset_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["mel_cutoff_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    if "sample_weight" in df.columns:
        W = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    else:
        W = np.ones((len(df),), dtype=np.float32)

    if group_column in df.columns and df[group_column].nunique() >= 2 and GroupShuffleSplit is not None:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(split.split(df, groups=df[group_column]))
    else:
        split_at = max(1, int(len(df) * 0.8))
        train_idx = np.arange(split_at)
        valid_idx = np.arange(split_at, len(df))
        if len(valid_idx) <= 0:
            valid_idx = np.arange(max(0, len(df) - 1), len(df))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    pair_map = _build_vc_cv_pair_map(df)
    pair_weight = float(os.environ.get("UTOA_ML_VC_CV_PAIR_WEIGHT", 0.06) or 0.06)
    if pair_weight < 0.0:
        pair_weight = 0.0
    pair_weight = pair_weight * 0.5
    if pair_map:
        train_mask = np.zeros((len(df),), dtype=bool)
        valid_mask = np.zeros((len(df),), dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        pair_train_count = int(sum(1 for a, b in pair_map.items() if train_mask[a] and train_mask[b]))
        pair_valid_count = int(sum(1 for a, b in pair_map.items() if valid_mask[a] and valid_mask[b]))
        pair_total_count = int(len(pair_map))
        global_to_train = {int(g): i for i, g in enumerate(train_idx.tolist())}
        global_to_valid = {int(g): i for i, g in enumerate(valid_idx.tolist())}
        pair_map_train = {
            int(global_to_train[a]): int(global_to_train[b])
            for a, b in pair_map.items()
            if a in global_to_train and b in global_to_train
        }
        pair_map_valid = {
            int(global_to_valid[a]): int(global_to_valid[b])
            for a, b in pair_map.items()
            if a in global_to_valid and b in global_to_valid
        }
        val_src_pos = list(pair_map_valid.keys())
        val_dst_pos = [pair_map_valid[k] for k in val_src_pos]
    else:
        pair_train_count = 0
        pair_valid_count = 0
        pair_total_count = 0
        pair_map_train = {}
        pair_map_valid = {}
        val_src_pos = []
        val_dst_pos = []

    aux_dim = len(AUX_TARGET_NAMES) if use_aux else 0
    model = _build_model_rawmel(
        torch,
        nn,
        in_dim=int(X.shape[1]),
        patch_dim=int(P.shape[1]),
        mel_bins=int(mel_bins),
        onset_frames=int(onset_frames),
        tail_frames=int(tail_frames),
        aux_dim=aux_dim,
    )
    run_device = _resolve_device(torch, requested=device)
    if isinstance(run_device, str):
        run_device = torch.device(run_device)
    model = model.to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=run_device)
    P_train = torch.tensor(P[train_idx], dtype=torch.float32, device=run_device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32, device=run_device)
    B_train = torch.tensor(base[train_idx], dtype=torch.float32, device=run_device)
    M_train = torch.tensor(boundary[train_idx], dtype=torch.float32, device=run_device)
    W_train = torch.tensor(W[train_idx], dtype=torch.float32, device=run_device)

    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32, device=run_device)
    P_valid = torch.tensor(P[valid_idx], dtype=torch.float32, device=run_device)
    Y_valid = torch.tensor(Y[valid_idx], dtype=torch.float32, device=run_device)
    B_valid = torch.tensor(base[valid_idx], dtype=torch.float32, device=run_device)
    M_valid = torch.tensor(boundary[valid_idx], dtype=torch.float32, device=run_device)
    W_valid = torch.tensor(W[valid_idx], dtype=torch.float32, device=run_device)
    if use_aux:
        A_train = torch.tensor(A[train_idx], dtype=torch.float32, device=run_device)
        AM_train = torch.tensor(aux_mask[train_idx], dtype=torch.float32, device=run_device)
        A_valid = torch.tensor(A[valid_idx], dtype=torch.float32, device=run_device)
        AM_valid = torch.tensor(aux_mask[valid_idx], dtype=torch.float32, device=run_device)
    else:
        A_train = None
        AM_train = None
        A_valid = None
        AM_valid = None

    keys_train = [keys[i] for i in train_idx.tolist()]
    keys_valid = [keys[i] for i in valid_idx.tolist()]

    prefetch_mode = str(rawmel_prefetch or "none").strip().lower()
    if prefetch_mode not in ("none", "train"):
        prefetch_mode = "none"
    onset_train_cache = None
    tail_train_cache = None
    onset_valid_cache = None
    tail_valid_cache = None
    if prefetch_mode == "train":
        print("[TRAIN] rawmel prefetch: train")
        onset_train_cache, tail_train_cache = cache_index.get_batch(keys_train)
        onset_valid_cache, tail_valid_cache = cache_index.get_batch(keys_valid)

    def _np_to_device(np_arr):
        arr = np.ascontiguousarray(np_arr, dtype=np.float32)
        t = torch.from_numpy(arr)
        if run_device.type == "cuda":
            t = t.pin_memory().to(run_device, non_blocking=True)
        else:
            t = t.to(run_device)
        return t

    target_weights = torch.tensor([1.00, 0.90, 0.95, 1.00, 0.65], dtype=torch.float32, device=run_device).view(1, -1)
    cons_margin = 10.0
    cut_margin = 10.0
    aux_weight = 0.08

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 10

    train_n = int(X_train.shape[0])
    batch_n = max(1, int(batch_size))
    epochs_n = max(1, int(epochs))
    progress_every = int(progress_every)
    total_batches = max(1, int((train_n + batch_n - 1) / batch_n))
    if progress_every > 0:
        print(
            f"[TRAIN] rows={train_n} batches={total_batches} epochs={epochs_n} device={run_device} "
            f"rawmel={mel_bins}x(onset={onset_frames},tail={tail_frames})"
        )

    for epoch in range(epochs_n):
        model.train()
        perm = torch.randperm(train_n, device=run_device)
        for batch_i, start in enumerate(range(0, train_n, batch_n), start=1):
            batch_idx = perm[start:start + batch_n]
            idx_list = [int(i) for i in batch_idx.detach().cpu().tolist()]
            xb = X_train[batch_idx]
            pb = P_train[batch_idx]
            yb = Y_train[batch_idx]
            bb = B_train[batch_idx]
            mb = M_train[batch_idx]
            wb = W_train[batch_idx]
            if prefetch_mode == "train" and onset_train_cache is not None and tail_train_cache is not None:
                onset_np = onset_train_cache[idx_list]
                tail_np = tail_train_cache[idx_list]
            else:
                batch_keys = [keys_train[i] for i in idx_list]
                onset_np, tail_np = cache_index.get_batch(batch_keys)
            onset_t = _np_to_device(onset_np).unsqueeze(1)
            tail_t = _np_to_device(tail_np).unsqueeze(1)

            out = model(xb, pb, onset_t, tail_t)
            if use_aux:
                pred, conf, aux_pred = out
            else:
                pred, conf = out
                aux_pred = None
            base_loss = F.smooth_l1_loss(pred, yb, reduction="none")
            base_loss = torch.mean(base_loss * target_weights, dim=1)
            base_loss = torch.mean(base_loss * wb)

            offset = bb[:, 0] + pred[:, 0]
            consonant = bb[:, 1] + pred[:, 1]
            cutoff_abs = bb[:, 2] + pred[:, 2]
            pre = bb[:, 3] + pred[:, 3]
            ovl = bb[:, 4] + pred[:, 4]
            penalty = (
                torch.relu(ovl - pre)
                + torch.relu((pre + cons_margin) - consonant)
                + torch.relu((consonant + cut_margin) - cutoff_abs)
                + torch.relu(-offset)
            )
            penalty_loss = torch.mean(penalty * wb)

            align_loss = F.smooth_l1_loss(offset, mb[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs, mb[:, 1], reduction="none"
            )
            align_loss = torch.mean(align_loss * wb)

            err = torch.mean(torch.abs(pred - yb), dim=1)
            conf_target = torch.exp(-torch.clamp(err / 80.0, min=0.0, max=8.0))
            conf_loss = F.binary_cross_entropy(conf.squeeze(1), conf_target.detach(), reduction="none")
            conf_loss = torch.mean(conf_loss * wb)

            aux_loss = 0.0
            if use_aux and aux_pred is not None and A_train is not None and AM_train is not None:
                ab = A_train[batch_idx]
                am = AM_train[batch_idx]
                aux_err = F.smooth_l1_loss(aux_pred, ab, reduction="none")
                mask_sum = torch.clamp(torch.sum(am, dim=1), min=1.0)
                aux_err = torch.sum(aux_err * am, dim=1) / mask_sum
                aux_loss = torch.mean(aux_err * wb)

            pair_loss = 0.0
            if pair_weight > 0.0 and pair_map_train:
                src_pos, dst_pos = _batch_pair_positions(idx_list, pair_map_train)
                if src_pos:
                    src_t = torch.tensor(src_pos, device=run_device, dtype=torch.long)
                    dst_t = torch.tensor(dst_pos, device=run_device, dtype=torch.long)
                    pred_offset = bb[:, 0] + pred[:, 0]
                    true_offset = bb[:, 0] + yb[:, 0]
                    pred_gap = pred_offset[dst_t] - pred_offset[src_t]
                    true_gap = true_offset[dst_t] - true_offset[src_t]
                    pair_err = F.smooth_l1_loss(pred_gap, true_gap, reduction="none")
                    pair_w = 0.5 * (wb[src_t] + wb[dst_t])
                    pair_loss = torch.mean(pair_err * pair_w)

            total_loss = (
                base_loss
                + (0.25 * penalty_loss)
                + (0.12 * align_loss)
                + (0.05 * conf_loss)
                + (aux_weight * aux_loss)
                + (pair_weight * pair_loss)
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if progress_every > 0 and (batch_i % progress_every == 0 or batch_i == total_batches):
                loss_val = float(total_loss.detach().cpu().item())
                print(f"[TRAIN] epoch={epoch + 1}/{epochs_n} batch={batch_i}/{total_batches} loss={loss_val:.4f}")

        model.eval()
        with torch.no_grad():
            if prefetch_mode == "train" and onset_valid_cache is not None and tail_valid_cache is not None:
                onset_valid_np = onset_valid_cache
                tail_valid_np = tail_valid_cache
            else:
                onset_valid_np, tail_valid_np = cache_index.get_batch(keys_valid)
            onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
            tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
            out_val = model(X_valid, P_valid, onset_valid_t, tail_valid_t)
            if use_aux:
                pred_val, conf_val, aux_val = out_val
            else:
                pred_val, conf_val = out_val
                aux_val = None
            val_base = F.smooth_l1_loss(pred_val, Y_valid, reduction="none")
            val_base = torch.mean(val_base * target_weights, dim=1)
            val_base = torch.mean(val_base * W_valid)
            offset_v = B_valid[:, 0] + pred_val[:, 0]
            consonant_v = B_valid[:, 1] + pred_val[:, 1]
            cutoff_abs_v = B_valid[:, 2] + pred_val[:, 2]
            pre_v = B_valid[:, 3] + pred_val[:, 3]
            ovl_v = B_valid[:, 4] + pred_val[:, 4]
            val_penalty = (
                torch.relu(ovl_v - pre_v)
                + torch.relu((pre_v + cons_margin) - consonant_v)
                + torch.relu((consonant_v + cut_margin) - cutoff_abs_v)
                + torch.relu(-offset_v)
            )
            val_penalty = torch.mean(val_penalty * W_valid)
            val_align = F.smooth_l1_loss(offset_v, M_valid[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs_v, M_valid[:, 1], reduction="none"
            )
            val_align = torch.mean(val_align * W_valid)
            err_v = torch.mean(torch.abs(pred_val - Y_valid), dim=1)
            conf_target_v = torch.exp(-torch.clamp(err_v / 80.0, min=0.0, max=8.0))
            val_conf = F.binary_cross_entropy(conf_val.squeeze(1), conf_target_v.detach(), reduction="none")
            val_conf = torch.mean(val_conf * W_valid)
            val_aux = 0.0
            if use_aux and aux_val is not None and A is not None and AM_valid is not None:
                aux_err_v = F.smooth_l1_loss(aux_val, A_valid, reduction="none")
                mask_sum_v = torch.clamp(torch.sum(AM_valid, dim=1), min=1.0)
                aux_err_v = torch.sum(aux_err_v * AM_valid, dim=1) / mask_sum_v
                val_aux = torch.mean(aux_err_v * W_valid)
            val_pair = 0.0
            if pair_weight > 0.0 and val_src_pos:
                src_t = torch.tensor(val_src_pos, device=run_device, dtype=torch.long)
                dst_t = torch.tensor(val_dst_pos, device=run_device, dtype=torch.long)
                pred_offset_v = B_valid[:, 0] + pred_val[:, 0]
                true_offset_v = B_valid[:, 0] + Y_valid[:, 0]
                pred_gap_v = pred_offset_v[dst_t] - pred_offset_v[src_t]
                true_gap_v = true_offset_v[dst_t] - true_offset_v[src_t]
                pair_err_v = F.smooth_l1_loss(pred_gap_v, true_gap_v, reduction="none")
                pair_w_v = 0.5 * (W_valid[src_t] + W_valid[dst_t])
                val_pair = torch.mean(pair_err_v * pair_w_v)
            val_total = float(
                (
                    val_base
                    + (0.25 * val_penalty)
                    + (0.12 * val_align)
                    + (0.05 * val_conf)
                    + (aux_weight * val_aux)
                    + (pair_weight * val_pair)
                ).item()
            )

        new_best = False
        if val_total < best_val:
            best_val = val_total
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            new_best = True
        else:
            wait += 1
            if wait >= patience:
                break
        if progress_every > 0:
            status = "best" if new_best else "wait"
            print(
                f"[TRAIN] epoch={epoch + 1}/{epochs_n} val_loss={val_total:.4f} best={best_val:.4f} "
                f"patience={wait}/{patience} status={status}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if prefetch_mode == "train" and onset_valid_cache is not None and tail_valid_cache is not None:
            onset_valid_np = onset_valid_cache
            tail_valid_np = tail_valid_cache
        else:
            onset_valid_np, tail_valid_np = cache_index.get_batch(keys_valid)
        onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
        tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
        out_valid = model(X_valid, P_valid, onset_valid_t, tail_valid_t)
        if use_aux:
            pred_valid, conf_valid, aux_valid = out_valid
        else:
            pred_valid, conf_valid = out_valid
            aux_valid = None
    pred_valid_np = pred_valid.detach().cpu().numpy()
    conf_valid_np = conf_valid.detach().cpu().numpy().reshape(-1)
    truth_valid_np = Y[valid_idx]

    metrics = {}
    for col_i, target in enumerate(TARGET_NAMES):
        truth = truth_valid_np[:, col_i]
        pred = pred_valid_np[:, col_i]
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    aux_metrics = {}
    if use_aux and aux_valid is not None and A is not None and aux_mask is not None:
        aux_pred_np = aux_valid.detach().cpu().numpy()
        aux_truth_np = A[valid_idx]
        aux_mask_np = aux_mask[valid_idx]
        for col_i, target in enumerate(AUX_TARGET_NAMES):
            mask = aux_mask_np[:, col_i] > 0.5
            if np.any(mask):
                aux_metrics[target] = {
                    "rows": int(np.sum(mask)),
                    "model_mae": float(mean_absolute_error(aux_truth_np[mask, col_i], aux_pred_np[mask, col_i])),
                }
            else:
                aux_metrics[target] = {"rows": 0, "model_mae": 0.0}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "target_names": list(TARGET_NAMES),
            "patch_features": list(PATCH_FEATURES),
            "in_dim": int(X.shape[1]),
            "patch_dim": int(P.shape[1]),
            "hidden_dim": 160,
            "aux_dim": int(aux_dim),
            "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
            "rawmel_enabled": True,
            "mel_bins": int(mel_bins),
            "onset_frames": int(onset_frames),
            "tail_frames": int(tail_frames),
            "mel_patch_spec": patch_spec,
            "mel_patch_spec_hash": patch_hash,
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))

    meta = {
        "backend": COUPLED_BACKEND_RAWMEL,
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v2",
        "feature_version": schema.get("feature_version", ""),
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "targets": list(TARGET_NAMES),
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": dict(patch_spec),
        "mel_patch_spec_hash": patch_hash,
        "mel_bins": int(mel_bins),
        "onset_frames": int(onset_frames),
        "tail_frames": int(tail_frames),
        "min_confidence": float(min_confidence),
        "vc_cv_pair_weight": float(pair_weight),
        "vc_cv_pair_max_gap": int(os.environ.get("UTOA_ML_VC_CV_MAX_GAP", 5) or 5),
        "vc_cv_pairs_total": int(pair_total_count),
        "vc_cv_pairs_train": int(pair_train_count),
        "vc_cv_pairs_valid": int(pair_valid_count),
        "fallback_order": [COUPLED_BACKEND_RAWMEL, COUPLED_BACKEND, "lightgbm", "base"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
        "aux_holdout_metrics": aux_metrics,
        "holdout_confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
        "device_used": str(run_device),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "aux_metrics": aux_metrics,
                "confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_min": float(np.min(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_max": float(np.max(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "vc_cv_pair_weight": float(pair_weight),
                "vc_cv_pairs_total": int(pair_total_count),
                "vc_cv_pairs_valid": int(pair_valid_count),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta


def evaluate_coupled_bundle(
    model_dir: str,
    dataset_csv: str,
    *,
    language: str = "",
    format_type: str = "",
    device: str = "auto",
    rawmel_cache_dir: str = "",
) -> Dict[str, Any]:
    _require_training_stack()
    from core.oto_ml.coupled.inference import load_coupled_bundle, predict_coupled_deltas

    payload_meta = {}
    meta_path = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            payload_meta = json.load(f)

    bundle = load_coupled_bundle(model_dir, meta=payload_meta, schema=get_feature_schema(), device=device)
    df = _read_dataset_csv_resilient(dataset_csv)
    lang = str(language or payload_meta.get("language", "")).strip().lower()
    fmt = normalize_format_type(lang, format_type or payload_meta.get("format_type", "")) or "general"
    df = _prepare_training_frame(df, language=lang, format_type=fmt)
    if len(df) == 0:
        return {"rows": 0, "targets": {}, "confidence_mean": 0.0}

    preds = []
    confs = []
    rawmel_enabled = bool(bundle.get("rawmel_enabled", False))
    cache_index = None
    if rawmel_enabled:
        if not rawmel_cache_dir:
            raise RuntimeError("rawmel_cache_dir is required for coupled_nn_v2_rawmel evaluation")
        cache_index = MelPatchCacheIndex.load(rawmel_cache_dir)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if rawmel_enabled and cache_index is not None:
            key = str(row_dict.get("mel_patch_key", "") or "").strip()
            if not key:
                raise RuntimeError("missing mel_patch_key in dataset")
            onset_patch, tail_patch = cache_index.get(key)
            row_dict["mel_onset_patch"] = onset_patch
            row_dict["mel_tail_patch"] = tail_patch
        d, c = predict_coupled_deltas(bundle, row_dict)
        preds.append(d)
        confs.append(float(c))

    summary = {"rows": int(len(df)), "targets": {}, "confidence_mean": float(np.mean(confs)) if confs else 0.0}
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred = np.asarray([float(p.get(target, 0.0)) for p in preds], dtype=np.float64)
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    return summary
