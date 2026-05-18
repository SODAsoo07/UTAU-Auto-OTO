from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from core.coarse_crnn.boundary_types import BoundaryCandidate, BoundaryDecodeResult, DecodedOtoRow
from core.coarse_crnn.stage2_oto.constraints import (
    anchors_to_dataclass,
    apply_stage2_gate,
    max_candidate_distance_from_env,
    min_confidence_from_env,
)
from core.coarse_crnn.stage2_oto.features import build_stage2_feature_batch
from core.coarse_crnn.stage2_oto.model import Stage2OtoAssigner, load_stage2_checkpoint
from core.coarse_crnn.stage2_oto.types import Stage2ModelConfig


DEFAULT_STAGE2_MODEL_NAME = "oto_stage2_assigner_v1.pt"
STAGE2_MODEL_GLOB = "oto_stage2_assigner*.pt"


@dataclass
class Stage2Bundle:
    model: Stage2OtoAssigner
    config: Stage2ModelConfig
    path: str
    device: str
    meta: dict[str, object]


@dataclass(frozen=True)
class Stage2ApplyResult:
    decoded: BoundaryDecodeResult
    accepted_rows: int
    fallback_rows: int
    errors: tuple[str, ...] = ()


def apply_stage2_to_decode(
    *,
    decoded: BoundaryDecodeResult,
    candidates: list[BoundaryCandidate],
    bundle: Stage2Bundle,
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    model_quality: float | None = None,
    audio_reliability: float | None = None,
    min_confidence: float | None = None,
    max_candidate_distance_ms: float | None = None,
) -> Stage2ApplyResult:
    if not decoded.rows:
        return Stage2ApplyResult(decoded=decoded, accepted_rows=0, fallback_rows=0)
    torch = __import__("torch")
    feature_batch = build_stage2_feature_batch(
        decoded=decoded,
        candidates=candidates,
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        model_quality=model_quality,
        audio_reliability=audio_reliability,
    )
    if feature_batch.numeric_dim != int(bundle.config.numeric_dim):
        return Stage2ApplyResult(
            decoded=decoded,
            accepted_rows=0,
            fallback_rows=len(decoded.rows),
            errors=(f"stage2 feature dim mismatch: got {feature_batch.numeric_dim}, expected {bundle.config.numeric_dim}",),
        )
    device = str(bundle.device or "cpu")
    numeric = torch.tensor([row.numeric for row in feature_batch.rows], dtype=torch.float32, device=device).unsqueeze(0)
    role_id = torch.tensor([[row.role_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    prev_role_id = torch.tensor([[row.prev_role_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    next_role_id = torch.tensor([[row.next_role_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    alias_type_id = torch.tensor([[row.alias_type_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    language_id = torch.tensor([[row.language_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    format_id = torch.tensor([[row.format_id for row in feature_batch.rows]], dtype=torch.long, device=device)
    with torch.no_grad():
        anchors_norm, confidence = bundle.model(
            numeric,
            role_id,
            prev_role_id,
            next_role_id,
            alias_type_id,
            language_id,
            format_id,
        )
    anchors_norm = anchors_norm.squeeze(0).detach().cpu().tolist()
    confidence_values = confidence.squeeze(0).detach().cpu().tolist()
    rows: list[DecodedOtoRow] = []
    accepted = 0
    fallback = 0
    min_conf = min_confidence_from_env() if min_confidence is None else float(min_confidence)
    max_dist = (
        max_candidate_distance_from_env()
        if max_candidate_distance_ms is None
        else float(max_candidate_distance_ms)
    )
    for idx, row in enumerate(decoded.rows):
        duration = max(1.0, float(row.spec.duration_ms or decoded.duration_ms or 0.0))
        pred = tuple(float(v) * duration for v in anchors_norm[idx])
        conf = float(confidence_values[idx])
        decision = apply_stage2_gate(
            row=row,
            predicted_anchors_ms=pred,  # type: ignore[arg-type]
            confidence=conf,
            candidates=candidates,
            min_confidence=min_conf,
            max_candidate_distance_ms=max_dist,
        )
        if decision.accepted:
            accepted += 1
        else:
            fallback += 1
        anchors = anchors_to_dataclass(
            decision.anchors_ms,
            confidence=decision.confidence,
            reason=decision.reason,
        )
        rows.append(
            DecodedOtoRow(
                spec=row.spec,
                anchors=anchors,
                selected_time_ms=float(anchors.pre_abs),
                fallback_used=bool(row.fallback_used or not decision.accepted),
                reason=decision.reason,
                quality_score=float(row.quality_score),
            )
        )
    return Stage2ApplyResult(
        decoded=BoundaryDecodeResult(
            wav_path=decoded.wav_path,
            duration_ms=decoded.duration_ms,
            rows=rows,
            anchor_timeline_ms=list(decoded.anchor_timeline_ms),
            fallback_count=int(decoded.fallback_count) + int(fallback),
        ),
        accepted_rows=int(accepted),
        fallback_rows=int(fallback),
    )


def load_stage2_bundle(path: str, *, map_location: str = "cpu", device: str = "cpu") -> Stage2Bundle:
    model, config, meta = load_stage2_checkpoint(path, map_location=map_location)
    model = model.to(device).eval()
    return Stage2Bundle(
        model=model,
        config=config,
        path=os.path.abspath(path),
        device=str(device),
        meta=dict(meta or {}),
    )


def resolve_stage2_model_path(path_hint: str = "") -> str:
    candidates: list[str] = []
    for raw in (
        path_hint,
        os.environ.get("UTOA_STAGE2_OTO_MODEL_PATH", ""),
        os.environ.get("UTOA_STAGE2_OTO_MODEL", ""),
    ):
        text = str(raw or "").strip()
        if text:
            candidates.append(text)
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
        if not any(sep in candidate for sep in ("/", "\\", os.sep)):
            for root in _stage2_model_roots():
                attempt = os.path.abspath(os.path.join(root, candidate))
                if os.path.isfile(attempt):
                    return attempt
    for root in _stage2_model_roots():
        fixed = os.path.join(root, DEFAULT_STAGE2_MODEL_NAME)
        if os.path.isfile(fixed):
            return os.path.abspath(fixed)
    discovered: list[str] = []
    for root in _stage2_model_roots():
        discovered.extend(glob.glob(os.path.join(root, STAGE2_MODEL_GLOB)))
    discovered = [os.path.abspath(path) for path in discovered if os.path.isfile(path)]
    if discovered:
        discovered.sort(key=lambda path: (os.path.getmtime(path), path.lower()), reverse=True)
        return discovered[0]
    return ""


def list_available_stage2_models() -> list[dict[str, object]]:
    seen: set[str] = set()
    items: list[dict[str, object]] = []
    for root in _stage2_model_roots():
        for path in glob.glob(os.path.join(root, STAGE2_MODEL_GLOB)):
            full = os.path.abspath(path)
            key = full.lower()
            if key in seen or not os.path.isfile(full):
                continue
            seen.add(key)
            name = os.path.basename(full)
            items.append(
                {
                    "name": name,
                    "path": full,
                    "label": _friendly_stage2_label(name),
                    "mtime": float(os.path.getmtime(full)),
                }
            )
    items.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
    return items


def stage2_enabled_from_env(default: bool = False) -> bool:
    raw = str(os.environ.get("UTOA_STAGE2_OTO_ENABLE", "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _stage2_model_roots() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out = [
        os.path.join(root, "models", "coarse_crnn"),
        os.path.join(root, "assets", "models", "coarse_crnn"),
        os.path.join(root, "ml_workspace", "models", "coarse_crnn"),
    ]
    dedup: list[str] = []
    seen: set[str] = set()
    for item in out:
        full = os.path.abspath(item)
        if full.lower() in seen:
            continue
        seen.add(full.lower())
        dedup.append(full)
    return dedup


def _friendly_stage2_label(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    lower = text.lower()
    tags: list[str] = []
    if "smoke" in lower:
        tags.append("smoke")
    if "v1" in lower:
        tags.append("v1")
    if "audio" in lower:
        tags.append("audio")
    return f"{text}  ·  {', '.join(tags)}" if tags else text


__all__ = [
    "Stage2ApplyResult",
    "Stage2Bundle",
    "apply_stage2_to_decode",
    "list_available_stage2_models",
    "load_stage2_bundle",
    "resolve_stage2_model_path",
    "stage2_enabled_from_env",
]
