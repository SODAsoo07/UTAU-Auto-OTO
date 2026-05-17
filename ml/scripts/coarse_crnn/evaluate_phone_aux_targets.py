from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import build_phone_aware_target_map, training_rows_to_wav_groups
from core.coarse_crnn.boundary_types import PHONE_AWARE_IGNORE_INDEX
from core.coarse_crnn.training import resolve_torch_device


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    ap = argparse.ArgumentParser(description="Evaluate phone-aware auxiliary heads against pseudo-target masks.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-wavs", type=int, default=0)
    ap.add_argument(
        "--identity-languages",
        default="",
        help="Override UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS while building pseudo-targets.",
    )
    args = ap.parse_args()

    if str(args.identity_languages or "").strip():
        os.environ["UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS"] = str(args.identity_languages).strip()

    rows = _read_jsonl(args.manifest)
    grouped = training_rows_to_wav_groups(rows)
    wav_items = sorted(grouped.items())
    if int(args.max_wavs) > 0:
        wav_items = wav_items[: int(args.max_wavs)]

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, args.device)
    model, cfg, meta = load_boundary_checkpoint(args.model, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    if not bool(getattr(cfg, "enable_phone_aux_heads", False)) and not bool(getattr(cfg, "enable_phone_family_heads", False)):
        raise SystemExit("model does not contain phone-aware auxiliary heads")

    accum = {}
    if bool(getattr(cfg, "enable_phone_aux_heads", False)):
        accum.update(
            {
                "cvs": _HeadStats(list(cfg.cvs_labels)),
                "consonant": _HeadStats(list(cfg.consonant_labels)),
                "vowel": _HeadStats(list(cfg.vowel_labels)),
            }
        )
    if bool(getattr(cfg, "enable_phone_family_heads", False)):
        accum.update(
            {
                "consonant_family": _HeadStats(list(cfg.consonant_family_labels)),
                "vowel_nucleus": _HeadStats(list(cfg.vowel_nucleus_labels)),
                "vowel_glide": _HeadStats(list(cfg.vowel_glide_labels)),
            }
        )
    role_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    skipped_wavs = 0

    for wav_path, spec_anchor_rows in wav_items:
        if not spec_anchor_rows:
            continue
        specs = [spec for spec, _anchors in spec_anchor_rows]
        lang = _dominant([str(spec.language or "unknown") for spec in specs])
        language_counts[lang] += len(specs)
        for spec in specs:
            role_counts[str(spec.role or "other")] += 1
        score_map = infer_boundary_scores_with_model(model=model, config=cfg, wav_path=wav_path, device=str(torch_device))
        frame_count = len(score_map.times_ms)
        if frame_count <= 0:
            skipped_wavs += 1
            continue
        hop_ms = _infer_hop_ms(score_map.times_ms, default=float(cfg.hop_ms))
        duration_ms = max(float(getattr(specs[0], "duration_ms", 0.0) or 0.0), float(score_map.times_ms[-1]) + hop_ms)
        _times, target = build_phone_aware_target_map(
            spec_anchor_rows,
            duration_ms=duration_ms,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        if "cvs" in accum:
            accum["cvs"].update(
                _score_matrix(score_map.cvs_scores, cfg.cvs_labels, frame_count),
                target.cvs_target,
                target.cvs_mask,
                language=lang,
            )
        if "consonant" in accum:
            accum["consonant"].update(
                _score_matrix(score_map.consonant_scores, cfg.consonant_labels, frame_count),
                target.consonant_target,
                target.consonant_mask,
                language=lang,
            )
        if "vowel" in accum:
            accum["vowel"].update(
                _score_matrix(score_map.vowel_scores, cfg.vowel_labels, frame_count),
                target.vowel_target,
                target.vowel_mask,
                language=lang,
            )
        if "consonant_family" in accum:
            accum["consonant_family"].update(
                _score_matrix(score_map.consonant_family_scores, cfg.consonant_family_labels, frame_count),
                target.consonant_family_target,
                target.consonant_family_mask,
                language=lang,
            )
        if "vowel_nucleus" in accum:
            accum["vowel_nucleus"].update(
                _score_matrix(score_map.vowel_nucleus_scores, cfg.vowel_nucleus_labels, frame_count),
                target.vowel_nucleus_target,
                target.vowel_nucleus_mask,
                language=lang,
            )
        if "vowel_glide" in accum:
            accum["vowel_glide"].update(
                _score_matrix(score_map.vowel_glide_scores, cfg.vowel_glide_labels, frame_count),
                target.vowel_glide_target,
                target.vowel_glide_mask,
                language=lang,
            )

    report = {
        "manifest": str(args.manifest),
        "model": str(args.model),
        "wavs": len(wav_items),
        "skipped_wavs": int(skipped_wavs),
        "rows": int(sum(language_counts.values())),
        "identity_languages": str(os.environ.get("UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS", "") or ""),
        "model_meta": {
            "history": list((meta or {}).get("history", [])) if isinstance(meta, dict) else [],
            "phone_aux_enabled": bool((meta or {}).get("phone_aux_enabled", False)) if isinstance(meta, dict) else False,
            "phone_family_enabled": bool((meta or {}).get("phone_family_enabled", False)) if isinstance(meta, dict) else False,
            "cvs_loss_weight": (meta or {}).get("cvs_loss_weight") if isinstance(meta, dict) else None,
            "consonant_id_loss_weight": (meta or {}).get("consonant_id_loss_weight") if isinstance(meta, dict) else None,
            "vowel_id_loss_weight": (meta or {}).get("vowel_id_loss_weight") if isinstance(meta, dict) else None,
            "consonant_family_loss_weight": (meta or {}).get("consonant_family_loss_weight") if isinstance(meta, dict) else None,
            "vowel_nucleus_loss_weight": (meta or {}).get("vowel_nucleus_loss_weight") if isinstance(meta, dict) else None,
            "vowel_glide_loss_weight": (meta or {}).get("vowel_glide_loss_weight") if isinstance(meta, dict) else None,
            "phone_aux_class_balanced": (meta or {}).get("phone_aux_class_balanced") if isinstance(meta, dict) else None,
        },
        "role_counts": dict(sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "language_counts": dict(sorted(language_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "heads": {name: stats.report() for name, stats in accum.items()},
    }
    report["decoder_useful_metrics"] = _decoder_useful_metrics(report["heads"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


class _HeadStats:
    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.total = 0
        self.correct = 0
        self.target_prob_sum = 0.0
        self.pred_conf_sum = 0.0
        self.by_language: dict[str, dict[str, float]] = defaultdict(_blank_counts)
        self.by_label: dict[str, dict[str, float]] = defaultdict(_blank_counts)
        self.pred_counter: Counter[str] = Counter()

    def update(self, probs: np.ndarray, target: np.ndarray, mask: np.ndarray, *, language: str) -> None:
        if probs.size <= 0:
            return
        valid = (np.asarray(mask, dtype=np.float32) > 0.0) & (np.asarray(target, dtype=np.int64) != PHONE_AWARE_IGNORE_INDEX)
        if not np.any(valid):
            return
        probs_valid = probs[valid]
        target_valid = np.asarray(target, dtype=np.int64)[valid]
        pred_idx = np.argmax(probs_valid, axis=1)
        pred_conf = np.max(probs_valid, axis=1)
        target_prob = probs_valid[np.arange(len(target_valid)), target_valid]
        correct = pred_idx == target_valid
        self.total += int(len(target_valid))
        self.correct += int(np.sum(correct))
        self.target_prob_sum += float(np.sum(target_prob))
        self.pred_conf_sum += float(np.sum(pred_conf))
        lang_bucket = self.by_language[str(language or "unknown")]
        _add_counts(lang_bucket, len(target_valid), int(np.sum(correct)), float(np.sum(target_prob)), float(np.sum(pred_conf)))
        for pred in pred_idx.tolist():
            pred_label = self.labels[int(pred)] if 0 <= int(pred) < len(self.labels) else "<bad>"
            self.pred_counter[pred_label] += 1
        for idx, ok, prob, conf in zip(target_valid.tolist(), correct.tolist(), target_prob.tolist(), pred_conf.tolist()):
            label = self.labels[int(idx)] if 0 <= int(idx) < len(self.labels) else "<bad>"
            bucket = self.by_label[label]
            bucket["total"] += 1
            if bool(ok):
                bucket["correct"] += 1
            bucket["target_prob_sum"] += float(prob)
            bucket["pred_conf_sum"] += float(conf)

    def report(self) -> dict[str, Any]:
        return {
            "masked_frames": int(self.total),
            "top1_accuracy": _rate(self.correct, self.total),
            "avg_target_prob": _mean(self.target_prob_sum, self.total),
            "avg_pred_conf": _mean(self.pred_conf_sum, self.total),
            "by_language": {key: _counts_report(value) for key, value in sorted(self.by_language.items())},
            "by_label": {key: _counts_report(value) for key, value in sorted(self.by_label.items())},
            "worst_labels_by_accuracy": _worst_labels(self.by_label, self.labels, key="accuracy"),
            "lowest_labels_by_target_prob": _worst_labels(self.by_label, self.labels, key="target_prob"),
            "top_predicted_labels": self.pred_counter.most_common(12),
        }


def _decoder_useful_metrics(heads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metric_focus": [
            "cvs.consonant/vowel separation",
            "consonant_family broad timing class",
            "vowel_nucleus and vowel_glide broad vowel identity",
        ],
        "primary": {},
        "fine_identity_reference_only": {},
    }
    cvs = heads.get("cvs") or {}
    cvs_labels = cvs.get("by_label") if isinstance(cvs, dict) else {}
    if isinstance(cvs_labels, dict):
        for label in ("consonant", "vowel", "silence"):
            item = cvs_labels.get(label)
            if isinstance(item, dict):
                out["primary"][f"cvs_{label}_top1_accuracy"] = item.get("top1_accuracy", 0.0)
                out["primary"][f"cvs_{label}_avg_target_prob"] = item.get("avg_target_prob", 0.0)
    for key in ("consonant_family", "vowel_nucleus", "vowel_glide"):
        item = heads.get(key)
        if isinstance(item, dict):
            out["primary"][f"{key}_top1_accuracy"] = item.get("top1_accuracy", 0.0)
            out["primary"][f"{key}_avg_target_prob"] = item.get("avg_target_prob", 0.0)
            out["primary"][f"{key}_masked_frames"] = item.get("masked_frames", 0)
    for key in ("consonant", "vowel"):
        item = heads.get(key)
        if isinstance(item, dict):
            out["fine_identity_reference_only"][f"{key}_top1_accuracy"] = item.get("top1_accuracy", 0.0)
            out["fine_identity_reference_only"][f"{key}_avg_target_prob"] = item.get("avg_target_prob", 0.0)
    return out


def _score_matrix(scores: dict[str, list[float]] | None, labels: list[str], frame_count: int) -> np.ndarray:
    if not scores:
        return np.zeros((0, 0), dtype=np.float32)
    cols = []
    for label in labels:
        values = list(scores.get(label, []))
        if len(values) < frame_count:
            values = values + [0.0] * (frame_count - len(values))
        cols.append(np.asarray(values[:frame_count], dtype=np.float32))
    return np.stack(cols, axis=1) if cols else np.zeros((0, 0), dtype=np.float32)


def _infer_hop_ms(times_ms: list[float], *, default: float) -> float:
    if len(times_ms) >= 2:
        delta = float(times_ms[1]) - float(times_ms[0])
        if delta > 0:
            return delta
    return float(default)


def _dominant(values: list[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def _blank_counts() -> dict[str, float]:
    return {"total": 0.0, "correct": 0.0, "target_prob_sum": 0.0, "pred_conf_sum": 0.0}


def _add_counts(bucket: dict[str, float], total: int, correct: int, target_prob_sum: float, pred_conf_sum: float) -> None:
    bucket["total"] += float(total)
    bucket["correct"] += float(correct)
    bucket["target_prob_sum"] += float(target_prob_sum)
    bucket["pred_conf_sum"] += float(pred_conf_sum)


def _counts_report(bucket: dict[str, float]) -> dict[str, float]:
    total = int(bucket.get("total", 0.0) or 0.0)
    correct = int(bucket.get("correct", 0.0) or 0.0)
    return {
        "masked_frames": total,
        "top1_accuracy": _rate(correct, total),
        "avg_target_prob": _mean(float(bucket.get("target_prob_sum", 0.0) or 0.0), total),
        "avg_pred_conf": _mean(float(bucket.get("pred_conf_sum", 0.0) or 0.0), total),
    }


def _worst_labels(by_label: dict[str, dict[str, float]], labels: list[str], *, key: str) -> list[dict[str, Any]]:
    rows = []
    for label, bucket in by_label.items():
        total = int(bucket.get("total", 0.0) or 0.0)
        if total < 20:
            continue
        correct = int(bucket.get("correct", 0.0) or 0.0)
        row = {
            "label": label,
            "masked_frames": total,
            "top1_accuracy": _rate(correct, total),
            "avg_target_prob": _mean(float(bucket.get("target_prob_sum", 0.0) or 0.0), total),
            "avg_pred_conf": _mean(float(bucket.get("pred_conf_sum", 0.0) or 0.0), total),
        }
        rows.append(row)
    if key == "target_prob":
        rows.sort(key=lambda item: (float(item["avg_target_prob"]), -int(item["masked_frames"]), str(item["label"])))
    else:
        rows.sort(key=lambda item: (float(item["top1_accuracy"]), -int(item["masked_frames"]), str(item["label"])))
    return rows[:12]


def _rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if int(denom) > 0 else 0.0


def _mean(total: float, denom: int) -> float:
    return float(total) / float(denom) if int(denom) > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
