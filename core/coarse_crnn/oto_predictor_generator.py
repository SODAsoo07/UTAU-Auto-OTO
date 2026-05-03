from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Any

from core.coarse_crnn.oto_inference import predict_oto_with_model
from core.coarse_crnn.oto_model import load_oto_checkpoint
from core.coarse_crnn.training import resolve_torch_device
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_features import classify_alias_type
from core.oto_normalization import normalize_wav_key


DEFAULT_OTO_CRNN_MODEL_NAME = "oto_anchor_crnn_vcv_window_1epoch.pt"


@dataclass(frozen=True)
class _PredictRow:
    wav_rel: str
    wav_abs: str
    alias: str
    prev_alias: str
    next_alias: str
    row_index: int
    row_count: int


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
    callback: Callable[[str], None] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_no_mfa_source_oto(wav_dir=wav_dir, source_hint=source_oto_path)
    if not source:
        return 0, 0, ["CRNN OTO 예측용 베이스 OTO를 찾지 못했습니다."]
    model_file = resolve_oto_crnn_model_path(model_path)
    if not model_file:
        return 0, 0, ["CRNN OTO 예측 모델을 찾지 못했습니다. 모델 경로를 지정해 주세요."]
    output_file = _normalize_output_oto_path(out_path)
    if not output_file:
        return 0, 0, ["출력 OTO 경로가 비어 있습니다."]

    rows, total, missing = _prepare_prediction_rows(wav_dir=wav_dir, source_oto_path=source)
    if not rows:
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            return 0, total, [f"CRNN OTO 예측 대상 WAV 매칭 실패: {sample}"]
        return 0, total, ["CRNN OTO 예측 대상 행이 비어 있습니다."]

    _log(callback, f"[OTO-CRNN] source={source}")
    _log(callback, f"[OTO-CRNN] model={model_file}")
    _log(callback, f"[OTO-CRNN] device={device} rows={len(rows)}/{total}")

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    try:
        model, config, _meta = load_oto_checkpoint(model_file, map_location=str(torch_device))
        model = model.to(torch_device).eval()
    except Exception as exc:
        return 0, total, [f"CRNN OTO 모델 로드 실패: {exc}"]

    out_lines: list[str] = []
    errors: list[str] = []
    guard_changed = 0
    lang = str(language or "").strip().lower() or "korean"
    fmt = str(format_type or "").strip().lower() or "other"
    suffix = _normalize_alias_suffix(alias_suffix)
    for idx, row in enumerate(rows):
        try:
            pred = predict_oto_with_model(
                model=model,
                config=config,
                wav_path=row.wav_abs,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                prev_alias=row.prev_alias,
                next_alias=row.next_alias,
                row_index_in_wav=row.row_index,
                file_row_count=row.row_count,
                device=str(torch_device),
            )
            alias = _apply_alias_suffix(row.alias, suffix)
            params, changed = _apply_conservative_right_boundary_guard(
                pred.params,
                language=lang,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
            )
            guard_changed += 1 if changed else 0
            out_lines.append(
                f"{row.wav_rel}={alias},"
                f"{float(params['offset']):.3f},"
                f"{float(params['consonant']):.3f},"
                f"{float(params['cutoff']):.3f},"
                f"{float(params['preutterance']):.3f},"
                f"{float(params['overlap']):.3f}"
            )
        except Exception as exc:
            errors.append(f"{row.wav_rel}={row.alias}: {exc}")
        if callback and (idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(rows)):
            _log(callback, f"[OTO-CRNN] rows={idx + 1}/{len(rows)}")

    if errors:
        return len(out_lines), total, errors[:20]
    if not out_lines:
        return 0, total, ["CRNN OTO 예측 결과가 비었습니다."]

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines).rstrip() + "\n")
    except OSError as exc:
        return 0, total, [f"CRNN OTO 출력 저장 실패: {output_file} ({exc})"]

    if missing:
        sample = ", ".join(sorted(missing)[:5])
        suffix_text = "..." if len(missing) > 5 else ""
        _log(callback, f"[OTO-CRNN] 매칭 실패 wav: {sample}{suffix_text}")
    if guard_changed:
        _log(callback, f"[OTO-CRNN] right-boundary guard adjusted rows={guard_changed}/{len(out_lines)}")
    _log(callback, f"[OTO-CRNN] written={len(out_lines)} total={total}")
    return len(out_lines), total, []


def resolve_oto_crnn_model_path(path_hint: str = "") -> str:
    candidates = []
    for raw in (path_hint, os.environ.get("UTOA_OTO_CRNN_MODEL_PATH", "")):
        raw_text = str(raw or "").strip()
        if raw_text:
            candidates.append(raw_text)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.extend(
        [
            os.path.join(root, "ml_workspace", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "assets", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
        ]
    )
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
    return ""


def _prepare_prediction_rows(*, wav_dir: str, source_oto_path: str) -> tuple[list[_PredictRow], int, set[str]]:
    wav_root = os.path.abspath(str(wav_dir or "").strip())
    exact, normalized = _build_wav_lookup(wav_root)
    raw_rows = []
    total = 0
    missing: set[str] = set()
    for raw in read_text_with_fallback(source_oto_path).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        total += 1
        wav_name = os.path.basename(str(parsed.get("wav", "") or "").replace("\\", "/").strip())
        mapped = exact.get(wav_name.lower(), "")
        if not mapped:
            mapped = normalized.get(normalize_wav_key(wav_name), "")
        if not mapped:
            missing.add(wav_name)
            continue
        raw_rows.append(
            {
                "wav_rel": mapped,
                "wav_abs": os.path.join(wav_root, mapped.replace("/", os.sep)),
                "alias": str(parsed.get("alias", "") or "").strip(),
            }
        )

    per_wav: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        per_wav.setdefault(row["wav_rel"], []).append(row)

    out: list[_PredictRow] = []
    for wav_rel, items in per_wav.items():
        count = len(items)
        for idx, row in enumerate(items):
            prev_alias = str(items[idx - 1].get("alias", "") or "") if idx > 0 else ""
            next_alias = str(items[idx + 1].get("alias", "") or "") if idx + 1 < count else ""
            out.append(
                _PredictRow(
                    wav_rel=wav_rel,
                    wav_abs=str(row["wav_abs"]),
                    alias=str(row["alias"]),
                    prev_alias=prev_alias,
                    next_alias=next_alias,
                    row_index=idx,
                    row_count=count,
                )
            )
    return out, total, missing


def _build_wav_lookup(wav_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    if not os.path.isdir(wav_dir):
        return exact, normalized
    for current_root, _dirs, files in os.walk(wav_dir):
        for file_name in files:
            if not str(file_name).lower().endswith(".wav"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, wav_dir).replace("\\", "/")
            exact.setdefault(str(file_name).strip().lower(), rel_path)
            norm = normalize_wav_key(file_name)
            if norm:
                normalized.setdefault(norm, rel_path)
    return exact, normalized


def _apply_conservative_right_boundary_guard(
    params: dict[str, float],
    *,
    language: str,
    alias: str,
    duration_ms: float,
) -> tuple[dict[str, float], bool]:
    if str(os.environ.get("UTOA_OTO_CRNN_RIGHT_GUARD_ENABLE", "1")).strip().lower() in {"0", "false", "off", "no"}:
        return dict(params), False

    out = {key: float(value) for key, value in dict(params).items()}
    pre = max(0.0, float(out.get("preutterance", 0.0)))
    ovl = max(0.0, min(float(out.get("overlap", 0.0)), pre))
    out["overlap"] = ovl

    alias_type = _safe_alias_type(language, alias)
    max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap = _right_guard_limits(alias_type)
    max_cons_gap = _env_float("UTOA_OTO_CRNN_MAX_CONS_GAP_MS", max_cons_gap)
    max_cut_gap = _env_float("UTOA_OTO_CRNN_MAX_CUTOFF_GAP_MS", max_cut_gap)

    cons_raw = max(0.0, float(out.get("consonant", 0.0)))
    cons_min = pre + max(1.0, min_cons_gap)
    cons_max = pre + max(cons_min - pre, max_cons_gap)
    cons = _clamp(cons_raw, cons_min, cons_max)

    cutoff_mag_raw = abs(float(out.get("cutoff", 0.0)))
    cutoff_min = cons + max(1.0, min_cut_gap)
    cutoff_max = cons + max(cutoff_min - cons, max_cut_gap)
    offset = max(0.0, float(out.get("offset", 0.0)))
    if duration_ms > 1.0:
        cutoff_max = min(cutoff_max, max(cutoff_min, float(duration_ms) - offset - 1.0))
    cutoff_mag = _clamp(cutoff_mag_raw, cutoff_min, cutoff_max)

    out["consonant"] = cons
    out["cutoff"] = -cutoff_mag
    changed = (
        abs(cons - cons_raw) > 1e-6
        or abs(cutoff_mag - cutoff_mag_raw) > 1e-6
        or abs(ovl - float(params.get("overlap", 0.0))) > 1e-6
    )
    return out, changed


def _right_guard_limits(alias_type: str) -> tuple[float, float, float, float]:
    kind = str(alias_type or "").strip().lower()
    # return max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap
    if kind == "cv_head":
        return 92.0, 18.0, 150.0, 36.0
    if kind == "cv":
        return 78.0, 16.0, 130.0, 34.0
    if kind == "vcv":
        return 82.0, 16.0, 118.0, 30.0
    if kind == "vc":
        return 48.0, 10.0, 82.0, 22.0
    if kind == "vv":
        return 92.0, 16.0, 142.0, 32.0
    if kind == "br":
        return 42.0, 8.0, 72.0, 18.0
    return 82.0, 14.0, 128.0, 30.0


def _safe_alias_type(language: str, alias: str) -> str:
    try:
        return str(classify_alias_type(language, alias) or "other").strip().lower() or "other"
    except Exception:
        text = str(alias or "").strip()
        if text.startswith("- "):
            return "cv_head"
        if " " in text:
            return "vcv"
        return "cv" if text else "other"


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if raw.endswith(("\\", "/")) or os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    return raw


def _normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _apply_alias_suffix(alias: str, suffix: str) -> str:
    text = str(alias or "").strip()
    if not text or not suffix:
        return text
    return f"{text}_{suffix}"


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


__all__ = [
    "DEFAULT_OTO_CRNN_MODEL_NAME",
    "generate_oto_with_crnn_predictor",
    "resolve_oto_crnn_model_path",
]
