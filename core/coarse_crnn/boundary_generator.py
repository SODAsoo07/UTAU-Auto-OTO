from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params, load_row_specs_from_source_oto
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates


def generate_oto_with_boundary_decoder(
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
    special_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_no_mfa_source_oto(wav_dir=wav_dir, source_hint=source_oto_path)
    if not source:
        return 0, 0, ["Boundary scorer OTO 생성용 source oto.ini를 찾지 못했습니다."]
    model_file = str(model_path or "").strip()
    if not model_file:
        return 0, 0, ["Boundary scorer 모델 경로가 비어 있습니다."]
    if not os.path.isfile(model_file):
        return 0, 0, [f"Boundary scorer 모델을 찾지 못했습니다: {model_file}"]

    output_file = _normalize_output_oto_path(out_path)
    if not output_file:
        return 0, 0, ["출력 oto.ini 경로가 유효하지 않습니다."]

    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=source,
        wav_dir=wav_dir,
        language=language,
        format_type=format_type,
        alias_suffix=alias_suffix,
        special_aliases=special_aliases,
    )
    total = sum(len(items) for items in rows_by_wav.values())
    if total <= 0:
        return 0, 0, ["source oto.ini에서 처리 가능한 행을 찾지 못했습니다."]

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model, config, _meta = load_boundary_checkpoint(model_file, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    _log(callback, f"[OTO-Boundary] source={source}")
    _log(callback, f"[OTO-Boundary] model={model_file}")
    _log(callback, f"[OTO-Boundary] device={torch_device} wavs={len(rows_by_wav)} rows={total}")

    decoded_params: dict[tuple[str, str, int], dict[str, float]] = {}
    errors: list[str] = []
    for wav_idx, (wav_path, specs) in enumerate(sorted(rows_by_wav.items()), start=1):
        try:
            score_map = infer_boundary_scores_with_model(
                model=model,
                config=config,
                wav_path=wav_path,
                device=str(torch_device),
            )
            audio = compute_audio_candidates(wav_path)
            model_cands = peak_candidates_from_scores(score_map)
            audio_cands = audio_candidates_to_boundary_candidates(audio)
            merged = merge_candidates(model_candidates=model_cands, audio_candidates=audio_cands)
            decoded = decode_wav_rows(
                wav_path=wav_path,
                duration_ms=float(audio.duration_ms),
                row_specs=specs,
                candidates=merged,
                active_start_ms=float(audio.active_start_ms),
                active_end_ms=float(audio.active_end_ms),
            )
            _log(
                callback,
                f"[OTO-Boundary] wav={wav_idx}/{len(rows_by_wav)} peaks={len(model_cands)} merged={len(merged)} "
                f"fallback={decoded.fallback_count}/{len(decoded.rows)}",
            )
            for row in decoded.rows:
                params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
                key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
                decoded_params[key] = params
        except Exception as exc:
            errors.append(f"{os.path.basename(wav_path)}: {exc}")

    processed = 0
    output_lines: list[str] = []
    occurrence: dict[tuple[str, str], int] = defaultdict(int)
    for line_idx, raw in enumerate(read_text_with_fallback(source).splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        alias = str(parsed.get("alias", "") or "")
        pair = (wav_name, alias)
        occ = occurrence[pair]
        occurrence[pair] += 1
        key = (wav_name, alias, line_idx)
        params = decoded_params.get(key)
        if params is None:
            params = {
                "offset": float(parsed.get("offset", 0.0) or 0.0),
                "consonant": float(parsed.get("cons", 0.0) or 0.0),
                "cutoff": float(parsed.get("cutoff", 0.0) or 0.0),
                "preutterance": float(parsed.get("pre", 0.0) or 0.0),
                "overlap": float(parsed.get("ovl", 0.0) or 0.0),
            }
            if occ == 0:
                errors.append(f"{wav_name}:{alias} decode miss -> source fallback")
        else:
            processed += 1
        alias_out = _apply_suffix(alias, alias_suffix)
        output_lines.append(
            f"{wav_name}={alias_out},{params['offset']:.3f},{params['consonant']:.3f},"
            f"{params['cutoff']:.3f},{params['preutterance']:.3f},{params['overlap']:.3f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + ("\n" if output_lines else ""))
    _log(callback, f"[OTO-Boundary] wrote={output_file} processed={processed}/{total}")
    return int(processed), int(total), errors


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    if raw.lower().endswith(".ini"):
        return os.path.abspath(raw)
    return os.path.join(os.path.abspath(raw), "oto.ini")


def _apply_suffix(alias: str, suffix: str) -> str:
    base = str(alias or "").strip()
    add = str(suffix or "").strip()
    if not base or not add:
        return base
    if base.endswith(add):
        return base
    return f"{base}{add}"


def _log(callback: Callable[[str], None] | None, text: str) -> None:
    if callback is None:
        return
    try:
        callback(str(text))
    except Exception:
        pass


__all__ = ["generate_oto_with_boundary_decoder"]

