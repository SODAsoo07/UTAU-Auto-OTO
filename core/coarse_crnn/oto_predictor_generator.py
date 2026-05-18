from __future__ import annotations

import glob
import os

from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder

DEFAULT_OTO_CRNN_MODEL_NAME = "deprecated_direct_param_oto_crnn.pt"
DEFAULT_BOUNDARY_SCORER_MODEL_NAME = "oto_boundary_scorer_v3_slotfix.pt"
_BOUNDARY_SCORER_MODEL_GLOB = "oto_boundary_scorer*.pt"
_AUTO_EXCLUDED_BOUNDARY_MODEL_MARKERS = ("phone_aux", "_smoke")
_DIRECT_PARAM_DEPRECATED_ERROR = (
    "Direct-parameter OTO CRNN은 폐기되었습니다. "
    "CRNN OTO는 boundary_decoder만 지원합니다. "
    "필요하면 ml/scripts/coarse_crnn/deprecated/direct_param 아래의 보존 스크립트를 명시적으로 실행하세요."
)


def generate_oto_with_crnn_predictor(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    language: str,
    format_type: str = "",
    model_path: str = "",
    device: str = "auto",
    alias_suffix: str = "",
    callback=None,
    special_aliases=None,
    engine: str = "",
):
    engine_code = _normalize_engine(engine or os.environ.get("UTOA_OTO_CRNN_ENGINE", ""))
    if engine_code == "boundary_decoder":
        resolved_model = resolve_boundary_scorer_model_path(model_path)
        if not resolved_model:
            return 0, 0, [
                "Boundary scorer 모델을 찾지 못했습니다. "
                "UTOA_OTO_CRNN_MODEL/UTOA_BOUNDARY_SCORER_MODEL_PATH를 지정하거나 "
                "ml_workspace/models/coarse_crnn 아래에 oto_boundary_scorer*.pt를 배치하세요."
            ]
        return generate_oto_with_boundary_decoder(
            wav_dir=wav_dir,
            out_path=out_path,
            source_oto_path=source_oto_path,
            language=language,
            format_type=format_type,
            model_path=resolved_model,
            device=device,
            alias_suffix=alias_suffix,
            callback=callback,
            special_aliases=special_aliases,
        )
    return 0, 0, [_DIRECT_PARAM_DEPRECATED_ERROR]


def resolve_boundary_scorer_model_path(path_hint: str = "") -> str:
    candidates: list[str] = []
    for raw in (
        path_hint,
        os.environ.get("UTOA_BOUNDARY_SCORER_MODEL_PATH", ""),
        os.environ.get("UTOA_BOUNDARY_SCORER_MODEL", ""),
        os.environ.get("UTOA_OTO_CRNN_MODEL", ""),
        os.environ.get("UTOA_OTO_CRNN_MODEL_PATH", ""),
    ):
        text = str(raw or "").strip()
        if text:
            candidates.append(text)
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
        # Plain basename (no separators) → search the standard roots.
        if not any(sep in candidate for sep in ("/", "\\", os.sep)):
            for root in _boundary_model_roots():
                attempt = os.path.abspath(os.path.join(root, candidate))
                if os.path.isfile(attempt):
                    return attempt

    roots = _boundary_model_roots()
    for root in roots:
        fixed = os.path.join(root, DEFAULT_BOUNDARY_SCORER_MODEL_NAME)
        if os.path.isfile(fixed):
            return os.path.abspath(fixed)

    discovered: list[str] = []
    for root in roots:
        pattern = os.path.join(root, _BOUNDARY_SCORER_MODEL_GLOB)
        discovered.extend(glob.glob(pattern))
    discovered = [os.path.abspath(path) for path in discovered if os.path.isfile(path)]
    if not _allow_experimental_boundary_model_auto_pick():
        discovered = [path for path in discovered if not _is_experimental_boundary_model(path)]
    if discovered:
        discovered.sort(key=lambda path: (os.path.getmtime(path), path.lower()), reverse=True)
        return discovered[0]
    return ""


def list_available_boundary_scorer_models() -> list[dict[str, object]]:
    """Enumerate boundary scorer checkpoints discoverable in the standard roots.

    Returns dicts with keys:
        - name (str)            : basename, used as the UI choice token
        - path (str)            : absolute path
        - label (str)           : human-readable label for the dropdown
        - is_experimental (bool): matches the auto-pick exclusion list
        - mtime (float)         : last-modified time

    Sorted newest-mtime first. Used by the UI to populate a model picker.
    """
    seen: set[str] = set()
    items: list[dict[str, object]] = []
    for root in _boundary_model_roots():
        pattern = os.path.join(root, _BOUNDARY_SCORER_MODEL_GLOB)
        for path in glob.glob(pattern):
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
                    "label": _friendly_boundary_model_label(name),
                    "is_experimental": _is_experimental_boundary_model(full),
                    "mtime": float(os.path.getmtime(full)),
                }
            )
    items.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
    return items


def _friendly_boundary_model_label(name: str) -> str:
    """Short human label for the dropdown. Falls back to the raw filename."""
    text = str(name or "").strip()
    if not text:
        return ""
    lower = text.lower()
    tags: list[str] = []
    if "convonly" in lower or "conv_only" in lower:
        tags.append("conv-only")
    if "selfdistill" in lower or "self_distill" in lower or "distill" in lower:
        tags.append("증류")
    if "phone_aux_warm" in lower:
        tags.append("phone-aware warm")
    elif "phone_aux" in lower:
        tags.append("phone-aware")
    if "v5_codabridge" in lower or "codabridge" in lower:
        tags.append("v5 codabridge")
    if "v4" in lower and "boundary_scorer_v4" in lower:
        tags.append("v4")
    if "v3_slotfix" in lower:
        tags.append("v3 slotfix (기본)")
    if "_smoke" in lower:
        tags.append("smoke")
    if not tags:
        return text
    return f"{text}  ·  {', '.join(tags)}"


def _boundary_model_roots() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def _normalize_engine(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"legacy", "legacy_direct", "legacy-direct", "direct", "old"}:
        return "legacy_direct"
    if text in {"", "boundary", "boundary_decoder", "boundary-scorer", "scorer", "new"}:
        return "boundary_decoder"
    return "boundary_decoder"


def _allow_experimental_boundary_model_auto_pick() -> bool:
    raw = str(os.environ.get("UTOA_BOUNDARY_SCORER_ALLOW_EXPERIMENTAL", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_experimental_boundary_model(path: str) -> bool:
    name = os.path.basename(str(path or "")).lower()
    return any(marker in name for marker in _AUTO_EXCLUDED_BOUNDARY_MODEL_MARKERS)


__all__ = [
    "DEFAULT_BOUNDARY_SCORER_MODEL_NAME",
    "DEFAULT_OTO_CRNN_MODEL_NAME",
    "generate_oto_with_crnn_predictor",
    "list_available_boundary_scorer_models",
    "resolve_boundary_scorer_model_path",
]
