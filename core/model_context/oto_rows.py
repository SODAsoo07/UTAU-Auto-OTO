from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping

from core.model_context.audio import build_audio_context
from core.model_context.filename import parse_filename_context
from core.model_context.oto_params import source_params_to_context
from core.model_context.types import AliasRowContext, ModelSampleContext, OtoRowSpec, WavIdentity
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def model_context_from_row_spec(
    spec: OtoRowSpec,
    *,
    audio_mode: str = "duration",
) -> ModelSampleContext:
    wav_name = spec.wav_name or os.path.basename(spec.wav_path)
    wav_stem = os.path.splitext(os.path.basename(wav_name))[0]
    filename = parse_filename_context(wav_name, language=spec.language, format_type=spec.format_type)
    source_params: Mapping[str, object] = spec.source_params or {
        "offset": 0.0,
        "consonant": 0.0,
        "cutoff": 0.0,
        "preutterance": 0.0,
        "overlap": 0.0,
    }
    oto = source_params_to_context(source_params, duration_ms=spec.duration_ms)
    audio = build_audio_context(spec.wav_path, mode=audio_mode)
    return ModelSampleContext(
        wav=WavIdentity(
            wav_path=spec.wav_path,
            wav_name=wav_name,
            wav_stem=wav_stem,
            language=spec.language,
            format_type=spec.format_type,
        ),
        filename=filename,
        alias=AliasRowContext(
            alias=spec.alias,
            alias_norm=str(spec.alias or "").strip(),
            alias_role=spec.role,
            alias_type=str(spec.meta.get("alias_type", "") or "other") if isinstance(spec.meta, dict) else "other",
            slot_index=int(spec.slot_index),
            slot_count=int(max(1, spec.slot_count)),
            prev_alias=spec.prev_alias,
            next_alias=spec.next_alias,
            line_index=int(spec.line_index),
        ),
        oto=oto,
        audio=audio,
        warnings=tuple(oto.repair_warnings),
        meta=dict(spec.meta or {}),
    )


def iter_model_contexts_from_row_specs(
    row_specs_by_wav: Mapping[str, Iterable[OtoRowSpec]],
    *,
    audio_mode: str = "duration",
) -> Iterator[ModelSampleContext]:
    for rows in row_specs_by_wav.values():
        for spec in rows:
            yield model_context_from_row_spec(spec, audio_mode=audio_mode)


def load_row_specs_from_source_oto(
    *,
    source_oto_path: str,
    wav_dir: str,
    language: str = "",
    format_type: str = "",
    alias_suffix: str = "",
    ignore_source_timing: bool = True,
) -> dict[str, list[OtoRowSpec]]:
    text = read_text_with_fallback(str(source_oto_path or ""))
    grouped: dict[str, list[OtoRowSpec]] = {}
    pending: list[dict[str, object]] = []
    for line_index, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "").strip()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav_name or not alias:
            continue
        pending.append({"line_index": line_index, "parsed": parsed, "wav_name": wav_name, "alias": alias})

    by_wav: dict[str, list[dict[str, object]]] = {}
    for row in pending:
        by_wav.setdefault(str(row["wav_name"]), []).append(row)

    for wav_name, rows in by_wav.items():
        wav_path = os.path.join(str(wav_dir or ""), wav_name)
        duration = build_audio_context(wav_path, mode="duration").duration_ms
        slot_count = max(1, len(rows))
        specs: list[OtoRowSpec] = []
        for idx, row in enumerate(rows):
            parsed = dict(row.get("parsed") or {})
            alias = str(row.get("alias") or "")
            prev_alias = str(rows[idx - 1].get("alias") or "") if idx > 0 else ""
            next_alias = str(rows[idx + 1].get("alias") or "") if idx + 1 < len(rows) else ""
            role = _infer_alias_role(alias)
            specs.append(
                OtoRowSpec(
                    wav_name=wav_name,
                    wav_path=wav_path,
                    alias=alias,
                    role=role,
                    slot_index=idx,
                    slot_count=slot_count,
                    prev_alias=prev_alias,
                    next_alias=next_alias,
                    language=str(language or ""),
                    format_type=str(format_type or ""),
                    line_index=int(row.get("line_index", 0) or 0),
                    duration_ms=duration,
                    source_params=(
                        {
                            "offset": 0.0,
                            "consonant": 0.0,
                            "cutoff": 0.0,
                            "preutterance": 0.0,
                            "overlap": 0.0,
                        }
                        if ignore_source_timing
                        else {
                            "offset": float(parsed.get("offset", 0.0) or 0.0),
                            "consonant": float(parsed.get("consonant", parsed.get("cons", 0.0)) or 0.0),
                            "cutoff": float(parsed.get("cutoff", 0.0) or 0.0),
                            "preutterance": float(parsed.get("preutterance", parsed.get("pre", 0.0)) or 0.0),
                            "overlap": float(parsed.get("overlap", parsed.get("ovl", 0.0)) or 0.0),
                        }
                    ),
                    alias_suffix=str(alias_suffix or ""),
                    meta={
                        "alias_type": role,
                        "source_timing_ignored": bool(ignore_source_timing),
                    },
                )
            )
        grouped[wav_path] = specs
    return grouped


def iter_model_contexts_from_source_oto(*args, audio_mode: str = "duration", **kwargs) -> Iterator[ModelSampleContext]:
    specs = load_row_specs_from_source_oto(*args, **kwargs)
    yield from iter_model_contexts_from_row_specs(specs, audio_mode=audio_mode)


def _infer_alias_role(alias: object) -> str:
    text = str(alias or "").strip().lower()
    if not text:
        return "other"
    if text in {"br", "bre", "breath", "sil", "sp", "pau"}:
        return "br"
    if text in {"endbr", "end"}:
        return "endbr"
    if text.startswith("-"):
        return "-cv"
    if text.endswith("-"):
        return "v-"
    if " " in text:
        left, right = [part for part in text.split(" ") if part][:2] if len([part for part in text.split(" ") if part]) >= 2 else ("", "")
        if left and right:
            return "vv" if left[-1:] in "aeiou" and right[:1] in "aeiou" else "vc"
        return "vc"
    return "cv"
