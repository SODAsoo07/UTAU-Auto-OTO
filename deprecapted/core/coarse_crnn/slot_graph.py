from __future__ import annotations

from typing import Any, Sequence

from core.coarse_crnn.alias_role import normalize_role
from core.model_context.filename import filename_order_tokens as _model_context_filename_order_tokens


SLOT_TYPES: tuple[str, ...] = ("other", "anchor", "boundary", "final", "silence")

_ANCHOR_ROLES = {"-cv", "cv", "v", "special"}
_BOUNDARY_ROLES = {"vc", "vv", "v-cv"}
_FINAL_ROLES = {"cv-", "v-", "endbr"}
_SILENCE_ROLES = {"br"}

_ROLE_ORDER = {
    "-cv": 0.10,
    "cv": 0.20,
    "v": 0.30,
    "cv-": 0.45,
    "v-": 0.50,
    "vc": 0.58,
    "vv": 0.66,
    "v-cv": 0.74,
    "endbr": 0.88,
    "br": 0.94,
    "special": 0.25,
    "other": 0.0,
}


def filename_order_tokens(wav_name: object) -> list[str]:
    return _model_context_filename_order_tokens(wav_name)


def slot_type_for_role(alias_role: object) -> str:
    role = normalize_role(str(alias_role or ""))
    if role in _ANCHOR_ROLES:
        return "anchor"
    if role in _BOUNDARY_ROLES:
        return "boundary"
    if role in _FINAL_ROLES:
        return "final"
    if role in _SILENCE_ROLES:
        return "silence"
    return "other"


def slot_type_id_norm(slot_type: object) -> float:
    text = str(slot_type or "").strip().lower()
    try:
        idx = SLOT_TYPES.index(text)
    except ValueError:
        idx = 0
    return float(idx) / float(max(1, len(SLOT_TYPES) - 1))


def slot_role_order_norm(alias_role: object) -> float:
    role = normalize_role(str(alias_role or ""))
    return float(_ROLE_ORDER.get(role, 0.0))


def apply_slot_context(rows: Sequence[dict[str, Any]]) -> None:
    items = [row for row in rows if isinstance(row, dict)]
    if not items:
        return
    expected_count = max(len(filename_order_tokens(items[0].get("wav", ""))), 0)
    expected_anchors = max(1, expected_count)
    expected_boundaries = max(1, expected_anchors - 1)

    by_type: dict[str, list[dict[str, Any]]] = {slot_type: [] for slot_type in SLOT_TYPES}
    for row in items:
        role = normalize_role(str(row.get("alias_role", "") or ""))
        slot_type = slot_type_for_role(role)
        row["slot_type"] = slot_type
        row["slot_role_order_norm"] = slot_role_order_norm(role)
        row["slot_is_anchor"] = 1.0 if slot_type == "anchor" else 0.0
        row["slot_is_boundary"] = 1.0 if slot_type == "boundary" else 0.0
        row["slot_is_final"] = 1.0 if slot_type == "final" else 0.0
        row["slot_is_silence"] = 1.0 if slot_type == "silence" else 0.0
        row["slot_type_id_norm"] = slot_type_id_norm(slot_type)
        by_type.setdefault(slot_type, []).append(row)

    _assign_slot_ordinals(by_type.get("anchor", []), expected_anchors)
    _assign_slot_ordinals(by_type.get("boundary", []), expected_boundaries)
    _assign_slot_ordinals(by_type.get("final", []), 1, force_ratio=1.0)
    _assign_slot_ordinals(by_type.get("silence", []), 1, force_ratio=1.0)
    _assign_slot_ordinals(by_type.get("other", []), max(1, len(by_type.get("other", []))))


def _assign_slot_ordinals(rows: Sequence[dict[str, Any]], expected_slots: int, *, force_ratio: float | None = None) -> None:
    count = len(rows)
    denom = float(max(1, int(expected_slots) - 1))
    for idx, row in enumerate(rows):
        slot_index = min(max(0, int(idx)), max(0, int(expected_slots) - 1))
        ratio = float(force_ratio) if force_ratio is not None else (float(slot_index) / denom if denom > 0 else 0.0)
        row["slot_index"] = int(slot_index)
        row["slot_count"] = int(max(1, expected_slots))
        row["slot_ratio_in_wav"] = max(0.0, min(1.0, float(ratio)))
        row["slot_ordinal_in_type"] = int(idx)
        row["slot_type_count"] = int(max(1, count))
        row["slot_left_index"] = int(max(0, slot_index - 1)) if str(row.get("slot_type")) == "boundary" else int(slot_index)
        row["slot_right_index"] = int(slot_index + 1) if str(row.get("slot_type")) == "boundary" else int(slot_index)
        row["slot_mapping_confidence"] = 0.75 if str(row.get("slot_type")) == "boundary" else 1.0


def slot_context_from_row(row: dict[str, Any]) -> list[float]:
    role = str(row.get("alias_role", "") or "")
    slot_type = str(row.get("slot_type", "") or slot_type_for_role(role))
    return [
        slot_type_id_norm(slot_type),
        _clamp01(row.get("slot_is_anchor", 1.0 if slot_type == "anchor" else 0.0)),
        _clamp01(row.get("slot_is_boundary", 1.0 if slot_type == "boundary" else 0.0)),
        _clamp01(row.get("slot_is_final", 1.0 if slot_type == "final" else 0.0)),
        _clamp01(row.get("slot_ratio_in_wav", row.get("row_ratio_in_wav", 0.0))),
        _clamp01(row.get("slot_role_order_norm", slot_role_order_norm(role))),
    ]


def slot_context_from_role(
    alias_role: object,
    *,
    row_index_in_wav: int = 0,
    file_row_count: int = 1,
) -> list[float]:
    role = normalize_role(str(alias_role or ""))
    slot_type = slot_type_for_role(role)
    count = max(1, int(file_row_count))
    if slot_type == "boundary":
        ratio = 0.5
    elif slot_type in {"final", "silence"}:
        ratio = 1.0
    else:
        ratio = 0.0 if count <= 1 else max(0.0, min(1.0, float(int(row_index_in_wav)) / float(count - 1)))
    return [
        slot_type_id_norm(slot_type),
        1.0 if slot_type == "anchor" else 0.0,
        1.0 if slot_type == "boundary" else 0.0,
        1.0 if slot_type == "final" else 0.0,
        ratio,
        slot_role_order_norm(role),
    ]


def _clamp01(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


__all__ = [
    "SLOT_TYPES",
    "apply_slot_context",
    "filename_order_tokens",
    "slot_context_from_role",
    "slot_context_from_row",
    "slot_role_order_norm",
    "slot_type_for_role",
    "slot_type_id_norm",
]
