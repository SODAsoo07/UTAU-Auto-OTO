from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from core.cvn_classifier import CVN_LABELS, predict_cvn
from core.cvn_feature_extractor import extract_frame_features
from core.model_context.audio import build_wav_index
from core.model_context.oto_params import safe_cutoff_abs_ms, wav_duration_ms
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml_features import classify_alias_type


LABEL_TO_INDEX = {"C": 0, "V": 1}
IGNORE_LABEL = -1
ProgressFn = Callable[[str], None]

_PAUSE_ALIASES = {
    "",
    "r",
    "rr",
    "rest",
    "sil",
    "silence",
    "sp",
    "spn",
    "pau",
    "pause",
    "_",
    "-",
}


@dataclass(frozen=True)
class OtoRow:
    wav: str
    alias: str
    offset_ms: float
    cons_ms: float
    cutoff: float
    pre_ms: float
    ovl_ms: float


@dataclass(frozen=True)
class CvnDatasetBundle:
    x: np.ndarray
    y: np.ndarray
    file_ids: np.ndarray
    rule_pred_labels: np.ndarray
    stats: dict[str, object]
    rule_baseline_confusion: np.ndarray


def _normalize_alias(alias: str) -> str:
    text = str(alias or "").strip().lower()
    text = "".join(ch for ch in text if not ch.isspace())
    return text


def _is_pause_alias(alias: str) -> bool:
    norm = _normalize_alias(alias)
    if norm in _PAUSE_ALIASES:
        return True
    if norm.startswith("br"):
        return True
    return False


def _safe_alias_type(language: str, alias: str) -> str:
    try:
        return str(classify_alias_type(language, alias)).strip().lower()
    except Exception:
        return ""


def _feature_matrix_for_linear(sample_rate: int, rms: np.ndarray, zcr: np.ndarray, centroid: np.ndarray, flatness: np.ndarray, f0_hz: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    nyquist = max(float(sample_rate) * 0.5, 1.0)
    centroid_norm = np.clip(np.asarray(centroid, dtype=np.float32) / float(nyquist), 0.0, 1.0)
    flatness_norm = np.clip(np.asarray(flatness, dtype=np.float32), 0.0, 1.0)
    f0_norm = np.clip(np.asarray(f0_hz, dtype=np.float32) / 500.0, 0.0, 1.0)
    return np.column_stack(
        [
            np.asarray(rms, dtype=np.float32),
            np.asarray(zcr, dtype=np.float32),
            centroid_norm,
            flatness_norm,
            f0_norm,
            np.asarray(voiced, dtype=np.float32),
        ]
    ).astype(np.float32)


def _emit_progress(progress_fn: ProgressFn | None, message: str) -> None:
    if callable(progress_fn):
        try:
            progress_fn(str(message))
        except Exception:
            pass


def _parse_float_csv(text: str | None) -> list[float]:
    raw = str(text or "").strip()
    if not raw:
        return []
    out: list[float] = []
    for tok in raw.split(","):
        part = str(tok).strip()
        if not part:
            continue
        try:
            val = float(part)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(float(val))
    return out


def _parse_oto_rows(oto_path: str) -> list[OtoRow]:
    if not oto_path or not os.path.isfile(oto_path):
        return []
    text = read_text_with_fallback(oto_path)
    out: list[OtoRow] = []
    for raw in text.splitlines():
        row = parse_oto_line(raw)
        if not row:
            continue
        out.append(
            OtoRow(
                wav=str(row.get("wav", "") or ""),
                alias=str(row.get("alias", "") or ""),
                offset_ms=float(row.get("offset", 0.0) or 0.0),
                cons_ms=float(row.get("cons", 0.0) or 0.0),
                cutoff=float(row.get("cutoff", 0.0) or 0.0),
                pre_ms=float(row.get("pre", 0.0) or 0.0),
                ovl_ms=float(row.get("ovl", 0.0) or 0.0),
            )
        )
    return out


def _group_rows_by_wav(rows: Sequence[OtoRow]) -> dict[str, list[OtoRow]]:
    grouped: dict[str, list[OtoRow]] = {}
    for row in rows:
        wav_key = str(row.wav or "").strip()
        if not wav_key:
            continue
        grouped.setdefault(wav_key, []).append(row)
    for key, wav_rows in grouped.items():
        wav_rows.sort(key=lambda r: float(r.offset_ms))
        grouped[key] = wav_rows
    return grouped


def label_frames_from_oto_rows(
    *,
    times_ms: np.ndarray,
    wav_duration_ms: float,
    oto_rows: Sequence[OtoRow],
    language: str,
    label_stats: dict[str, int] | None = None,
) -> np.ndarray:
    times = np.asarray(times_ms, dtype=np.float32).reshape(-1)
    if times.size <= 0:
        return np.zeros((0,), dtype=np.int64)

    votes = np.zeros((times.size, len(CVN_LABELS)), dtype=np.float32)
    for row in oto_rows:
        onset = max(0.0, float(row.offset_ms))
        cutoff_abs = safe_cutoff_abs_ms(onset, float(row.cutoff), float(wav_duration_ms), convention="legacy_candidate")
        if cutoff_abs <= (onset + 1.0):
            continue

        in_seg = (times >= onset) & (times <= cutoff_abs)
        if not np.any(in_seg):
            continue

        alias_type = _safe_alias_type(language, row.alias)
        if _is_pause_alias(row.alias) or alias_type in {"br", "vc", "vcv", "vv", "mono"}:
            if label_stats is not None:
                label_stats["skip_non_cv_family"] = int(label_stats.get("skip_non_cv_family", 0)) + 1
            continue

        if alias_type in {"cv", "cv_head"}:
            # CV alias: preutterance end is the consonant end / vowel start.
            split = onset + max(0.0, float(row.pre_ms))
        else:
            # Unknown alias type fallback.
            split = onset + max(0.0, float(row.cons_ms))

        cons_end = split
        if cons_end >= cutoff_abs:
            cons_end = max(onset + 2.0, cutoff_abs - 2.0)
        if cons_end <= onset:
            cons_end = onset + min(12.0, max(1.0, cutoff_abs - onset))

        cons_mask = in_seg & (times <= cons_end)
        vowel_mask = in_seg & (times > cons_end)
        if np.any(cons_mask):
            votes[cons_mask, LABEL_TO_INDEX["C"]] += 1.0
        if np.any(vowel_mask):
            votes[vowel_mask, LABEL_TO_INDEX["V"]] += 1.0

    has_vote = np.sum(votes, axis=1) > 1e-6
    labels = np.full((times.size,), int(IGNORE_LABEL), dtype=np.int64)
    if np.any(has_vote):
        labels[has_vote] = np.argmax(votes[has_vote], axis=1).astype(np.int64)
    if label_stats is not None:
        label_stats["frames_ignored_no_vote"] = int(label_stats.get("frames_ignored_no_vote", 0)) + int(np.sum(~has_vote))
    return labels


def _recover_ignore_labels_from_acoustics(
    labels: np.ndarray,
    *,
    rms: np.ndarray,
    zcr: np.ndarray,
    flatness: np.ndarray,
    f0_hz: np.ndarray,
    voiced_mask: np.ndarray,
    label_stats: dict[str, int] | None = None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).copy()
    if y.size <= 0:
        return y
    ignore = y == int(IGNORE_LABEL)
    if not np.any(ignore):
        return y

    rms_v = np.asarray(rms, dtype=np.float32).reshape(-1)
    zcr_v = np.asarray(zcr, dtype=np.float32).reshape(-1)
    flat_v = np.asarray(flatness, dtype=np.float32).reshape(-1)
    f0_v = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    voiced_v = np.asarray(voiced_mask, dtype=bool).reshape(-1)
    n = min(y.size, rms_v.size, zcr_v.size, flat_v.size, f0_v.size, voiced_v.size)
    if n <= 0:
        return y
    if n < y.size:
        y = y[:n]
        ignore = ignore[:n]
    rms_v = rms_v[:n]
    zcr_v = zcr_v[:n]
    flat_v = flat_v[:n]
    f0_v = f0_v[:n]
    voiced_v = voiced_v[:n]

    rms_p20 = float(np.percentile(rms_v, 20.0)) if rms_v.size else 0.0
    rms_p50 = float(np.percentile(rms_v, 50.0)) if rms_v.size else 0.0
    active_th = max(1e-4, min(rms_p50 * 0.45, rms_p20 * 1.6 + 5e-4))

    # Breath/lip/noise frames remain ignored in C/V mode.
    noise_ignore_mask = ignore & (~voiced_v) & (rms_v >= active_th) & ((flat_v >= 0.45) | (zcr_v >= 0.10))
    if np.any(noise_ignore_mask) and label_stats is not None:
        label_stats["frames_noise_ignored"] = int(label_stats.get("frames_noise_ignored", 0)) + int(np.sum(noise_ignore_mask))

    # Vocal fry (creaky voiced): treat as consonant for this project policy -> C
    fry_c_mask = ignore & voiced_v & (f0_v >= 35.0) & (f0_v <= 110.0) & (zcr_v <= 0.20) & (rms_v >= (active_th * 0.8))
    if np.any(fry_c_mask):
        y[fry_c_mask] = int(LABEL_TO_INDEX["C"])
        if label_stats is not None:
            label_stats["frames_recovered_fry_c"] = int(label_stats.get("frames_recovered_fry_c", 0)) + int(np.sum(fry_c_mask))

    return y


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = len(CVN_LABELS)) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y_true.size <= 0 or y_pred.size <= 0:
        return cm
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    n = min(yt.size, yp.size)
    np.add.at(cm, (yt[:n], yp[:n]), 1)
    return cm


def _classification_report(confusion: np.ndarray) -> dict[str, object]:
    cm = np.asarray(confusion, dtype=np.int64)
    total = int(np.sum(cm))
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for idx, label in enumerate(CVN_LABELS):
        tp = float(cm[idx, idx])
        fp = float(np.sum(cm[:, idx]) - tp)
        fn = float(np.sum(cm[idx, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(cm[idx, :])),
        }
        f1_values.append(float(f1))
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "total_frames": total,
    }


def _class_recall(metrics: dict[str, object], label: str) -> float:
    try:
        per_class = dict(metrics.get("per_class", {}) or {})
        info = dict(per_class.get(str(label), {}) or {})
        return float(info.get("recall", 0.0) or 0.0)
    except Exception:
        return 0.0


def _validate_collapse_guard(
    *,
    val_metrics: dict[str, object],
    enable_guard: bool,
    min_recall_c: float,
    min_recall_v: float,
) -> None:
    if not bool(enable_guard):
        return
    c_recall = _class_recall(val_metrics, "C")
    v_recall = _class_recall(val_metrics, "V")
    c_min = float(max(0.0, min(1.0, float(min_recall_c))))
    v_min = float(max(0.0, min(1.0, float(min_recall_v))))
    if c_recall < c_min or v_recall < v_min:
        raise RuntimeError(
            (
                "Collapsed CV model detected: "
                f"val_recall(C)={c_recall:.6f} < {c_min:.6f} or "
                f"val_recall(V)={v_recall:.6f} < {v_min:.6f}. "
                "Model artifact was blocked."
            )
        )


def build_cvn_dataset_from_staged_root(
    *,
    dataset_root: str,
    language: str,
    formats: Iterable[str] | None = None,
    max_voicebanks: int = 0,
    max_wavs_per_voicebank: int = 0,
    target_sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 80,
    seed: int = 42,
    progress_fn: ProgressFn | None = None,
) -> CvnDatasetBundle:
    lang = str(language or "").strip().lower()
    rng = random.Random(int(seed))
    allow_formats = {str(f).strip().lower() for f in (formats or []) if str(f).strip()}
    if not allow_formats:
        allow_formats = {"cv", "cvc", "cvvc"}

    candidates = discover_training_candidates_from_dataset_root(dataset_root, auto_oto_policy="generate-temp")
    candidates = [c for c in candidates if str(c.language).strip().lower() == lang]
    if allow_formats:
        candidates = [c for c in candidates if str(c.format_type).strip().lower() in allow_formats]
    candidates = [c for c in candidates if str(c.status).strip().lower() == "ready"]
    rng.shuffle(candidates)
    if int(max_voicebanks) > 0:
        candidates = candidates[: int(max_voicebanks)]
    _emit_progress(
        progress_fn,
        f"[CVN] dataset candidates: language={lang}, ready_voicebanks={len(candidates)}",
    )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []

    rule_cm = np.zeros((len(CVN_LABELS), len(CVN_LABELS)), dtype=np.int64)
    stats = {
        "language": lang,
        "voicebanks_total": len(candidates),
        "voicebanks_used": 0,
        "wav_total_seen": 0,
        "wav_used": 0,
        "wav_missing_for_oto": 0,
        "wav_feature_fail": 0,
        "frames_total": 0,
        "frames_ignored_no_vote": 0,
    }

    next_file_id = 0
    for vb_idx, cand in enumerate(candidates, start=1):
        vb_name = os.path.basename(str(getattr(cand, "voicebank_dir", "") or "").rstrip("\\/")) or os.path.basename(
            str(getattr(cand, "wav_dir", "") or "").rstrip("\\/")
        )
        if not vb_name:
            vb_name = os.path.basename(str(getattr(cand, "manual_oto", "") or "").rstrip("\\/"))
        _emit_progress(
            progress_fn,
            f"[CVN] voicebank {vb_idx}/{max(1, len(candidates))}: {vb_name}",
        )

        rows = _parse_oto_rows(str(cand.manual_oto))
        by_wav = _group_rows_by_wav(rows)
        if not by_wav:
            continue
        wav_index = build_wav_index(
            str(cand.wav_dir),
            normalize_key_fn=lambda name: str(name or "").strip().lower(),
        )
        if not wav_index:
            continue

        wav_items = list(by_wav.items())
        rng.shuffle(wav_items)
        if int(max_wavs_per_voicebank) > 0:
            wav_items = wav_items[: int(max_wavs_per_voicebank)]

        used_in_vb = 0
        wav_count = int(len(wav_items))
        for wav_idx, (wav_name, wav_rows) in enumerate(wav_items, start=1):
            stats["wav_total_seen"] = int(stats["wav_total_seen"]) + 1
            wav_path = wav_index.get(str(wav_name).strip().lower(), "")
            if not wav_path:
                stats["wav_missing_for_oto"] = int(stats["wav_missing_for_oto"]) + 1
                continue

            try:
                features = extract_frame_features(
                    wav_path,
                    target_sr=int(target_sr),
                    frame_ms=float(frame_ms),
                    hop_ms=float(hop_ms),
                    n_mels=int(n_mels),
                )
            except Exception:
                stats["wav_feature_fail"] = int(stats["wav_feature_fail"]) + 1
                continue

            wav_dur = wav_duration_ms(wav_path)
            if wav_dur <= 0.0:
                approx_end = float(features.times_ms[-1]) + (float(features.frame_ms) * 0.5)
                wav_dur = max(approx_end, 1.0)

            y = label_frames_from_oto_rows(
                times_ms=np.asarray(features.times_ms, dtype=np.float32),
                wav_duration_ms=wav_dur,
                oto_rows=wav_rows,
                language=lang,
                label_stats=stats,
            )
            if y.size <= 0:
                continue
            y = _recover_ignore_labels_from_acoustics(
                y,
                rms=np.asarray(features.rms, dtype=np.float32),
                zcr=np.asarray(features.zcr, dtype=np.float32),
                flatness=np.asarray(features.spectral_flatness, dtype=np.float32),
                f0_hz=np.asarray(features.f0_hz, dtype=np.float32),
                voiced_mask=np.asarray(features.voiced_mask, dtype=bool),
                label_stats=stats,
            )

            x = _feature_matrix_for_linear(
                sample_rate=int(features.sample_rate),
                rms=np.asarray(features.rms, dtype=np.float32),
                zcr=np.asarray(features.zcr, dtype=np.float32),
                centroid=np.asarray(features.spectral_centroid, dtype=np.float32),
                flatness=np.asarray(features.spectral_flatness, dtype=np.float32),
                f0_hz=np.asarray(features.f0_hz, dtype=np.float32),
                voiced=np.asarray(features.voiced_mask, dtype=np.float32),
            )
            if x.shape[0] != y.shape[0]:
                n = min(int(x.shape[0]), int(y.shape[0]))
                if n <= 0:
                    continue
                x = x[:n]
                y = y[:n]

            rule_out = predict_cvn(features, backend="rule", smooth_window=5, min_run_frames=3)
            rule_idx = np.asarray([LABEL_TO_INDEX.get(str(label), -1) for label in rule_out.labels], dtype=np.int64)
            n = min(int(x.shape[0]), int(y.shape[0]), int(rule_idx.shape[0]))
            if n <= 0:
                continue
            x = x[:n]
            y = y[:n]
            rule_idx = rule_idx[:n]
            valid_mask = (y >= 0) & (rule_idx >= 0)
            if not np.any(valid_mask):
                continue
            x = x[valid_mask]
            y = y[valid_mask]
            rule_idx = rule_idx[valid_mask]
            rule_cm = rule_cm + _confusion_matrix(y, rule_idx, num_classes=len(CVN_LABELS))

            file_vec = np.full((x.shape[0],), int(next_file_id), dtype=np.int64)
            next_file_id += 1

            x_parts.append(x.astype(np.float32))
            y_parts.append(y.astype(np.int64))
            file_parts.append(file_vec)
            rule_parts.append(rule_idx.astype(np.int64))
            used_in_vb += 1
            stats["wav_used"] = int(stats["wav_used"]) + 1
            stats["frames_total"] = int(stats["frames_total"]) + int(y.shape[0])
            if wav_idx == wav_count or (wav_idx % 10) == 0:
                _emit_progress(
                    progress_fn,
                    (
                        f"[CVN] voicebank {vb_idx}/{max(1, len(candidates))} "
                        f"wav {wav_idx}/{max(1, wav_count)} | used={used_in_vb} | "
                        f"frames_total={int(stats['frames_total'])}"
                    ),
                )

        if used_in_vb > 0:
            stats["voicebanks_used"] = int(stats["voicebanks_used"]) + 1

    if not x_parts:
        return CvnDatasetBundle(
            x=np.zeros((0, 6), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            file_ids=np.zeros((0,), dtype=np.int64),
            rule_pred_labels=np.zeros((0,), dtype=np.int64),
            stats=stats,
            rule_baseline_confusion=rule_cm,
        )

    x_all = np.concatenate(x_parts, axis=0).astype(np.float32)
    y_all = np.concatenate(y_parts, axis=0).astype(np.int64)
    file_all = np.concatenate(file_parts, axis=0).astype(np.int64)
    rule_all = np.concatenate(rule_parts, axis=0).astype(np.int64)
    return CvnDatasetBundle(
        x=x_all,
        y=y_all,
        file_ids=file_all,
        rule_pred_labels=rule_all,
        stats=stats,
        rule_baseline_confusion=rule_cm,
    )


def _weighted_softmax_train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int = len(CVN_LABELS),
    epochs: int = 180,
    lr: float = 0.08,
    l2: float = 1e-4,
    class_weight_power: float = 0.35,
    class_weight_min: float = 0.50,
    class_weight_max: float = 2.50,
    progress_fn: ProgressFn | None = None,
    progress_interval: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n, d = x.shape
    if n <= 0:
        raise ValueError("empty training set")
    y_int = np.asarray(y, dtype=np.int64).reshape(-1)
    if y_int.shape[0] != n:
        raise ValueError("x/y shape mismatch")

    w = np.zeros((d, num_classes), dtype=np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    counts = np.bincount(y_int, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_base = float(n) / (float(num_classes) * counts)
    class_weights = np.power(inv_base, float(max(0.0, class_weight_power)))
    class_weights = class_weights / max(float(np.mean(class_weights)), 1e-8)
    cw_min = float(max(0.05, class_weight_min))
    cw_max = float(max(cw_min, class_weight_max))
    class_weights = np.clip(class_weights, cw_min, cw_max)
    sample_w = class_weights[y_int]
    sample_w = sample_w * (float(n) / float(np.sum(sample_w)))
    sample_w = sample_w.astype(np.float32)

    y_one = np.zeros((n, num_classes), dtype=np.float32)
    y_one[np.arange(n), y_int] = 1.0

    num_epochs = int(max(1, epochs))
    log_interval = int(max(1, progress_interval))
    history_loss: list[float] = []
    for epoch_idx in range(num_epochs):
        logits = (x @ w) + b
        logits = logits - np.max(logits, axis=1, keepdims=True)
        expv = np.exp(logits).astype(np.float32)
        probs = expv / np.maximum(np.sum(expv, axis=1, keepdims=True), 1e-8)

        diff = (probs - y_one) * sample_w[:, None]
        grad_w = (x.T @ diff) / float(n)
        grad_b = np.mean(diff, axis=0)
        grad_w = grad_w + (float(l2) * w)

        w = (w - (float(lr) * grad_w)).astype(np.float32)
        b = (b - (float(lr) * grad_b)).astype(np.float32)

        p_true = np.sum(probs * y_one, axis=1)
        ce = -np.log(np.maximum(p_true, 1e-8)) * sample_w
        loss = float(np.mean(ce) + (0.5 * float(l2) * float(np.sum(w * w))))
        history_loss.append(loss)
        epoch_no = int(epoch_idx + 1)
        if epoch_no == 1 or epoch_no == num_epochs or (epoch_no % log_interval) == 0:
            _emit_progress(progress_fn, f"[CVN] train epoch {epoch_no}/{num_epochs} loss={loss:.6f}")

    return w, b, {"loss_start": float(history_loss[0]), "loss_end": float(history_loss[-1])}


def _predict_linear_probs(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = (x @ weights) + bias
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits)
    return (expv / np.maximum(np.sum(expv, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _predict_linear(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, *, c_threshold: float = 0.5) -> np.ndarray:
    probs = _predict_linear_probs(x, weights, bias)
    th = float(np.clip(float(c_threshold), 0.01, 0.99))
    return np.where(probs[:, LABEL_TO_INDEX["C"]] >= th, LABEL_TO_INDEX["C"], LABEL_TO_INDEX["V"]).astype(np.int64)


def _tune_c_threshold(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    candidate_thresholds: Sequence[float],
) -> tuple[float, dict[str, object], list[dict[str, float]]]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float32)
    n = min(int(y.size), int(p.shape[0]))
    if n <= 0:
        zero = _classification_report(np.zeros((len(CVN_LABELS), len(CVN_LABELS)), dtype=np.int64))
        return 0.5, zero, []

    cands = []
    for raw in candidate_thresholds:
        try:
            th = float(raw)
        except Exception:
            continue
        if not np.isfinite(th):
            continue
        cands.append(float(np.clip(th, 0.01, 0.99)))
    if not cands:
        cands = [0.5]
    cands = sorted(set(cands))

    table: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    best_th = 0.5
    best_metrics: dict[str, object] = _classification_report(np.zeros((len(CVN_LABELS), len(CVN_LABELS)), dtype=np.int64))
    for th in cands:
        y_pred = np.where(p[:n, LABEL_TO_INDEX["C"]] >= th, LABEL_TO_INDEX["C"], LABEL_TO_INDEX["V"]).astype(np.int64)
        metrics = _classification_report(_confusion_matrix(y[:n], y_pred, num_classes=len(CVN_LABELS)))
        macro_f1 = float(metrics.get("macro_f1", 0.0) or 0.0)
        c_recall = _class_recall(metrics, "C")
        v_recall = _class_recall(metrics, "V")
        table.append(
            {
                "c_threshold": float(th),
                "macro_f1": float(macro_f1),
                "c_recall": float(c_recall),
                "v_recall": float(v_recall),
            }
        )
        key = (float(macro_f1), float(c_recall), float(v_recall))
        if best_key is None or key > best_key:
            best_key = key
            best_th = float(th)
            best_metrics = metrics
    return float(best_th), best_metrics, table


def _split_train_val_by_file(
    file_ids: np.ndarray,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(file_ids, dtype=np.int64).reshape(-1)
    if ids.size <= 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    uniq = np.unique(ids)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)
    n_val = int(max(1, round(float(uniq.size) * float(max(0.05, min(0.5, val_ratio))))))
    val_files = set(int(v) for v in uniq[:n_val].tolist())
    val_mask = np.asarray([int(fid) in val_files for fid in ids], dtype=bool)
    train_mask = ~val_mask
    if not np.any(train_mask):
        train_mask = np.ones_like(val_mask, dtype=bool)
        val_mask = np.zeros_like(val_mask, dtype=bool)
    return train_mask, val_mask


def _rebalance_training_indices(
    y: np.ndarray,
    *,
    seed: int = 42,
    max_n_to_cv_ratio: float = 3.0,
) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    idx_c = np.where(labels == LABEL_TO_INDEX["C"])[0]
    idx_v = np.where(labels == LABEL_TO_INDEX["V"])[0]
    if idx_c.size <= 0 or idx_v.size <= 0:
        return np.arange(labels.size, dtype=np.int64)
    ratio = float(max(1.0, max_n_to_cv_ratio))
    rng = np.random.default_rng(int(seed))
    c_count = int(idx_c.size)
    v_count = int(idx_v.size)
    if v_count > int(round(c_count * ratio)):
        idx_v = rng.choice(idx_v, size=int(round(c_count * ratio)), replace=False)
    elif c_count > int(round(v_count * ratio)):
        idx_c = rng.choice(idx_c, size=int(round(v_count * ratio)), replace=False)

    selected = np.concatenate([idx_c.astype(np.int64), idx_v.astype(np.int64)], axis=0)
    selected.sort()
    return selected.astype(np.int64)


def train_cvn_linear_from_dataset(
    *,
    dataset_root: str,
    language: str,
    formats: Iterable[str] | None,
    out_model: str,
    out_report: str = "",
    max_voicebanks: int = 0,
    max_wavs_per_voicebank: int = 0,
    target_sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 80,
    seed: int = 42,
    val_ratio: float = 0.2,
    epochs: int = 180,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    max_n_to_cv_ratio: float = 3.0,
    class_weight_power: float = 0.35,
    class_weight_min: float = 0.50,
    class_weight_max: float = 2.50,
    max_n_to_cv_ratio_grid: Sequence[float] | None = None,
    class_weight_power_grid: Sequence[float] | None = None,
    c_threshold_grid: Sequence[float] | None = None,
    collapse_guard_enable: bool = True,
    collapse_guard_min_c_recall: float = 1e-6,
    collapse_guard_min_v_recall: float = 1e-6,
    progress_fn: ProgressFn | None = None,
    epoch_log_interval: int = 10,
) -> dict[str, object]:
    _emit_progress(
        progress_fn,
        f"[CVN] build dataset: root={os.path.abspath(str(dataset_root))}, language={str(language)}",
    )
    bundle = build_cvn_dataset_from_staged_root(
        dataset_root=dataset_root,
        language=language,
        formats=formats,
        max_voicebanks=max_voicebanks,
        max_wavs_per_voicebank=max_wavs_per_voicebank,
        target_sr=target_sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
        seed=seed,
        progress_fn=progress_fn,
    )
    x = np.asarray(bundle.x, dtype=np.float32)
    y = np.asarray(bundle.y, dtype=np.int64)
    file_ids = np.asarray(bundle.file_ids, dtype=np.int64)
    rule_pred_labels = np.asarray(bundle.rule_pred_labels, dtype=np.int64)
    if x.shape[0] <= 0 or y.shape[0] <= 0:
        raise RuntimeError("No training frames collected from dataset root.")
    n_total = min(int(x.shape[0]), int(y.shape[0]), int(file_ids.shape[0]), int(rule_pred_labels.shape[0]))
    if n_total <= 0:
        raise RuntimeError("No aligned training frames collected from dataset root.")
    if n_total < int(y.shape[0]):
        x = x[:n_total]
        y = y[:n_total]
        file_ids = file_ids[:n_total]
        rule_pred_labels = rule_pred_labels[:n_total]
    _emit_progress(
        progress_fn,
        (
            f"[CVN] dataset ready: frames={int(y.shape[0])}, files={int(np.unique(file_ids).size)}, "
            f"voicebanks_used={int(bundle.stats.get('voicebanks_used', 0))}"
        ),
    )

    train_mask, val_mask = _split_train_val_by_file(file_ids, val_ratio=val_ratio, seed=seed)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_val = x[val_mask] if np.any(val_mask) else np.zeros((0, x.shape[1]), dtype=np.float32)
    y_val = y[val_mask] if np.any(val_mask) else np.zeros((0,), dtype=np.int64)
    rule_train = rule_pred_labels[train_mask]
    rule_val = rule_pred_labels[val_mask] if np.any(val_mask) else np.zeros((0,), dtype=np.int64)

    ratio_grid = [float(max_n_to_cv_ratio)]
    if max_n_to_cv_ratio_grid:
        ratio_grid = [float(r) for r in max_n_to_cv_ratio_grid if np.isfinite(float(r))]
        if not ratio_grid:
            ratio_grid = [float(max_n_to_cv_ratio)]

    power_grid = [float(class_weight_power)]
    if class_weight_power_grid:
        power_grid = [float(p) for p in class_weight_power_grid if np.isfinite(float(p))]
        if not power_grid:
            power_grid = [float(class_weight_power)]

    threshold_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    if c_threshold_grid:
        threshold_grid = [float(t) for t in c_threshold_grid if np.isfinite(float(t))]
        if not threshold_grid:
            threshold_grid = [0.5]

    _emit_progress(
        progress_fn,
        (
            f"[CVN] hyper sweep: ratios={len(ratio_grid)}, powers={len(power_grid)}, "
            f"thresholds={len(threshold_grid)}"
        ),
    )

    candidate_rows: list[dict[str, object]] = []
    best_score = float("-inf")
    best_c_recall = float("-inf")
    best_candidate: dict[str, object] | None = None
    total_candidates = int(max(1, len(ratio_grid) * len(power_grid)))
    cand_idx = 0

    for ratio in ratio_grid:
        rebalance_idx = _rebalance_training_indices(y_train, seed=seed, max_n_to_cv_ratio=float(ratio))
        x_train_fit = x_train[rebalance_idx]
        y_train_fit = y_train[rebalance_idx]
        for power in power_grid:
            cand_idx += 1
            _emit_progress(
                progress_fn,
                (
                    f"[CVN] candidate {cand_idx}/{total_candidates}: "
                    f"ratio={float(ratio):.4f}, class_weight_power={float(power):.4f}, "
                    f"train_fit={int(y_train_fit.size)}"
                ),
            )
            weights, bias, train_info = _weighted_softmax_train(
                x_train_fit,
                y_train_fit,
                num_classes=len(CVN_LABELS),
                epochs=epochs,
                lr=learning_rate,
                l2=l2,
                class_weight_power=float(power),
                class_weight_min=class_weight_min,
                class_weight_max=class_weight_max,
                progress_fn=progress_fn,
                progress_interval=epoch_log_interval,
            )

            train_probs = _predict_linear_probs(x_train, weights, bias)
            train_argmax = _classification_report(
                _confusion_matrix(y_train, np.argmax(train_probs, axis=1).astype(np.int64), num_classes=len(CVN_LABELS))
            )
            if y_val.size > 0:
                val_probs = _predict_linear_probs(x_val, weights, bias)
                val_argmax = _classification_report(
                    _confusion_matrix(y_val, np.argmax(val_probs, axis=1).astype(np.int64), num_classes=len(CVN_LABELS))
                )
                best_th, val_metrics_candidate, th_table = _tune_c_threshold(
                    y_true=y_val,
                    probs=val_probs,
                    candidate_thresholds=threshold_grid,
                )
            else:
                val_probs = np.zeros((0, len(CVN_LABELS)), dtype=np.float32)
                val_argmax = _classification_report(np.zeros((len(CVN_LABELS), len(CVN_LABELS)), dtype=np.int64))
                best_th, val_metrics_candidate, th_table = _tune_c_threshold(
                    y_true=y_train,
                    probs=train_probs,
                    candidate_thresholds=threshold_grid,
                )

            y_pred_train = _predict_linear(x_train, weights, bias, c_threshold=float(best_th))
            train_metrics_candidate = _classification_report(_confusion_matrix(y_train, y_pred_train, num_classes=len(CVN_LABELS)))

            score = float(val_metrics_candidate.get("macro_f1", 0.0) or 0.0) if y_val.size > 0 else float(
                train_metrics_candidate.get("macro_f1", 0.0) or 0.0
            )
            c_recall = _class_recall(val_metrics_candidate if y_val.size > 0 else train_metrics_candidate, "C")
            row = {
                "ratio": float(ratio),
                "class_weight_power": float(power),
                "c_threshold": float(best_th),
                "score_macro_f1": float(score),
                "score_c_recall": float(c_recall),
                "val_macro_f1_tuned": float(val_metrics_candidate.get("macro_f1", 0.0) or 0.0),
                "val_macro_f1_argmax": float(val_argmax.get("macro_f1", 0.0) or 0.0),
                "train_macro_f1_tuned": float(train_metrics_candidate.get("macro_f1", 0.0) or 0.0),
                "train_macro_f1_argmax": float(train_argmax.get("macro_f1", 0.0) or 0.0),
            }
            candidate_rows.append(row)
            _emit_progress(
                progress_fn,
                (
                    f"[CVN] candidate result: val_macro_f1_tuned={float(row['val_macro_f1_tuned']):.4f}, "
                    f"val_macro_f1_argmax={float(row['val_macro_f1_argmax']):.4f}, "
                    f"best_c_threshold={float(best_th):.3f}"
                ),
            )

            if (score > best_score) or (score == best_score and c_recall > best_c_recall) or (best_candidate is None):
                best_score = float(score)
                best_c_recall = float(c_recall)
                best_candidate = {
                    "ratio": float(ratio),
                    "class_weight_power": float(power),
                    "c_threshold": float(best_th),
                    "weights": weights,
                    "bias": bias,
                    "train_info": train_info,
                    "train_metrics": train_metrics_candidate,
                    "val_metrics": val_metrics_candidate,
                    "train_argmax": train_argmax,
                    "val_argmax": val_argmax,
                    "threshold_table": th_table,
                    "train_fit_frames": int(y_train_fit.size),
                }

    if best_candidate is None:
        raise RuntimeError("CVN training sweep produced no valid candidate.")

    weights = np.asarray(best_candidate["weights"], dtype=np.float32)
    bias = np.asarray(best_candidate["bias"], dtype=np.float32)
    train_info = dict(best_candidate["train_info"])
    best_c_threshold = float(best_candidate["c_threshold"])
    train_metrics = dict(best_candidate["train_metrics"])
    val_metrics = dict(best_candidate["val_metrics"])
    train_metrics_argmax = dict(best_candidate["train_argmax"])
    val_metrics_argmax = dict(best_candidate["val_argmax"])
    best_train_fit_frames = int(best_candidate["train_fit_frames"])

    rule_overall_metrics = _classification_report(_confusion_matrix(y, rule_pred_labels, num_classes=len(CVN_LABELS)))
    rule_train_metrics = _classification_report(_confusion_matrix(y_train, rule_train, num_classes=len(CVN_LABELS)))
    if y_val.size > 0:
        rule_val_metrics = _classification_report(_confusion_matrix(y_val, rule_val, num_classes=len(CVN_LABELS)))
    else:
        rule_val_metrics = _classification_report(np.zeros((len(CVN_LABELS), len(CVN_LABELS)), dtype=np.int64))

    _validate_collapse_guard(
        val_metrics=val_metrics,
        enable_guard=bool(collapse_guard_enable),
        min_recall_c=float(collapse_guard_min_c_recall),
        min_recall_v=float(collapse_guard_min_v_recall),
    )

    out_model_abs = os.path.abspath(str(out_model))
    os.makedirs(os.path.dirname(out_model_abs), exist_ok=True)
    np.savez(
        out_model_abs,
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        c_threshold=np.asarray(best_c_threshold, dtype=np.float32),
        label_order=np.asarray(CVN_LABELS),
        feature_order=np.asarray(["rms", "zcr", "centroid_norm", "flatness", "f0_norm", "voiced"]),
    )
    _emit_progress(progress_fn, f"[CVN] model saved: {out_model_abs}")

    report = {
        "config": {
            "dataset_root": os.path.abspath(str(dataset_root)),
            "language": str(language),
            "formats": list(formats or []),
            "max_voicebanks": int(max_voicebanks),
            "max_wavs_per_voicebank": int(max_wavs_per_voicebank),
            "target_sr": int(target_sr),
            "frame_ms": float(frame_ms),
            "hop_ms": float(hop_ms),
            "n_mels": int(n_mels),
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "max_n_to_cv_ratio": float(best_candidate["ratio"]),
            "class_weight_power": float(best_candidate["class_weight_power"]),
            "class_weight_min": float(class_weight_min),
            "class_weight_max": float(class_weight_max),
            "c_threshold": float(best_c_threshold),
            "sweep": {
                "max_n_to_cv_ratio_grid": [float(v) for v in ratio_grid],
                "class_weight_power_grid": [float(v) for v in power_grid],
                "c_threshold_grid": [float(v) for v in threshold_grid],
                "num_candidates": int(len(candidate_rows)),
            },
            "collapse_guard_enable": bool(collapse_guard_enable),
            "collapse_guard_min_c_recall": float(collapse_guard_min_c_recall),
            "collapse_guard_min_v_recall": float(collapse_guard_min_v_recall),
        },
        "dataset_stats": dict(bundle.stats),
        "class_distribution": {
            "C": int(np.sum(y == LABEL_TO_INDEX["C"])),
            "V": int(np.sum(y == LABEL_TO_INDEX["V"])),
        },
        "split": {
            "train_frames": int(y_train.size),
            "train_fit_frames": int(best_train_fit_frames),
            "val_frames": int(y_val.size),
            "train_files": int(np.unique(file_ids[train_mask]).size),
            "val_files": int(np.unique(file_ids[val_mask]).size) if np.any(val_mask) else 0,
        },
        "train_info": train_info,
        "rule_baseline": rule_val_metrics,
        "rule_baseline_split": {
            "train": rule_train_metrics,
            "val": rule_val_metrics,
            "overall": rule_overall_metrics,
        },
        "model_metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
        "model_metrics_argmax": {
            "train": train_metrics_argmax,
            "val": val_metrics_argmax,
        },
        "tuning": {
            "best_candidate": {
                "max_n_to_cv_ratio": float(best_candidate["ratio"]),
                "class_weight_power": float(best_candidate["class_weight_power"]),
                "c_threshold": float(best_c_threshold),
                "score_macro_f1": float(best_score),
                "score_c_recall": float(best_c_recall),
            },
            "threshold_eval_table": list(best_candidate.get("threshold_table", [])),
            "candidate_results": candidate_rows,
        },
        "artifacts": {
            "model_npz": out_model_abs,
        },
    }

    if str(out_report or "").strip():
        out_report_abs = os.path.abspath(str(out_report))
        os.makedirs(os.path.dirname(out_report_abs), exist_ok=True)
        with open(out_report_abs, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["artifacts"]["report_json"] = out_report_abs
        _emit_progress(progress_fn, f"[CVN] report saved: {out_report_abs}")

    _emit_progress(
        progress_fn,
        (
            f"[CVN] done: val_macro_f1={float(((report.get('model_metrics', {}) or {}).get('val', {}) or {}).get('macro_f1', 0.0)):.4f}, "
            f"rule_val_macro_f1={float((report.get('rule_baseline', {}) or {}).get('macro_f1', 0.0)):.4f}"
        ),
    )

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight C/V linear backend from staged UTAU dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset_staged root.")
    parser.add_argument("--language", required=True, choices=["korean", "japanese"], help="Target language.")
    parser.add_argument("--formats", default="cv,cvc,cvvc", help="Comma-separated format filters (default: cv,cvc,cvvc).")
    parser.add_argument("--out-model", required=True, help="Output .npz model path (weights/bias).")
    parser.add_argument("--out-report", default="", help="Optional report json path.")
    parser.add_argument("--max-voicebanks", type=int, default=0, help="Limit number of voicebanks (0=all).")
    parser.add_argument("--max-wavs-per-voicebank", type=int, default=0, help="Limit wavs per voicebank (0=all).")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-n-to-cv-ratio", type=float, default=3.0, help="Max majority/minority ratio inside C/V training split.")
    parser.add_argument("--class-weight-power", type=float, default=0.35, help="Class-weight inverse frequency exponent (0=off).")
    parser.add_argument("--class-weight-min", type=float, default=0.50, help="Lower clip bound for normalized class weights.")
    parser.add_argument("--class-weight-max", type=float, default=2.50, help="Upper clip bound for normalized class weights.")
    parser.add_argument(
        "--sweep-max-ratio-list",
        default="",
        help="Optional CSV sweep for max_n_to_cv_ratio (e.g. 1.5,2.0,3.0). Empty means single --max-n-to-cv-ratio.",
    )
    parser.add_argument(
        "--sweep-class-weight-power-list",
        default="",
        help="Optional CSV sweep for class_weight_power (e.g. 0.35,0.55,0.75). Empty means single --class-weight-power.",
    )
    parser.add_argument(
        "--sweep-c-threshold-list",
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65",
        help="CSV sweep for C decision threshold on posterior P(C).",
    )
    parser.add_argument("--disable-collapse-guard", action="store_true", help="Allow saving even when C/V val recall collapse is detected.")
    parser.add_argument("--collapse-guard-min-c-recall", type=float, default=1e-6, help="Minimum required val recall for class C.")
    parser.add_argument("--collapse-guard-min-v-recall", type=float, default=1e-6, help="Minimum required val recall for class V.")
    parser.add_argument("--epoch-log-interval", type=int, default=10, help="How often to print epoch progress.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress log output.")
    return parser.parse_args()


def _safe_json_print(payload: dict[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    print(text)


def main() -> int:
    args = _parse_args()
    dataset_root = os.path.abspath(str(args.dataset_root))
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    formats = [p.strip().lower() for p in str(args.formats or "").split(",") if p.strip()]
    sweep_ratio = _parse_float_csv(str(args.sweep_max_ratio_list))
    sweep_power = _parse_float_csv(str(args.sweep_class_weight_power_list))
    sweep_threshold = _parse_float_csv(str(args.sweep_c_threshold_list))
    progress_fn: ProgressFn | None = None if bool(args.quiet) else (lambda msg: print(str(msg), flush=True))
    report = train_cvn_linear_from_dataset(
        dataset_root=dataset_root,
        language=str(args.language).strip().lower(),
        formats=formats,
        out_model=str(args.out_model),
        out_report=str(args.out_report),
        max_voicebanks=int(args.max_voicebanks),
        max_wavs_per_voicebank=int(args.max_wavs_per_voicebank),
        target_sr=int(args.target_sr),
        frame_ms=float(args.frame_ms),
        hop_ms=float(args.hop_ms),
        n_mels=int(args.n_mels),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        max_n_to_cv_ratio=float(args.max_n_to_cv_ratio),
        class_weight_power=float(args.class_weight_power),
        class_weight_min=float(args.class_weight_min),
        class_weight_max=float(args.class_weight_max),
        max_n_to_cv_ratio_grid=sweep_ratio,
        class_weight_power_grid=sweep_power,
        c_threshold_grid=sweep_threshold,
        collapse_guard_enable=(not bool(args.disable_collapse_guard)),
        collapse_guard_min_c_recall=float(args.collapse_guard_min_c_recall),
        collapse_guard_min_v_recall=float(args.collapse_guard_min_v_recall),
        progress_fn=progress_fn,
        epoch_log_interval=int(args.epoch_log_interval),
    )

    summary = {
        "dataset_stats": report.get("dataset_stats", {}),
        "class_distribution": report.get("class_distribution", {}),
        "rule_val_macro_f1": ((report.get("rule_baseline", {}) or {}).get("macro_f1", 0.0)),
        "model_val_macro_f1": ((report.get("model_metrics", {}) or {}).get("val", {}) or {}).get("macro_f1", 0.0),
        "model_val_macro_f1_argmax": ((report.get("model_metrics_argmax", {}) or {}).get("val", {}) or {}).get("macro_f1", 0.0),
        "best_c_threshold": (((report.get("tuning", {}) or {}).get("best_candidate", {}) or {}).get("c_threshold", 0.5)),
        "artifacts": report.get("artifacts", {}),
    }
    _safe_json_print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
