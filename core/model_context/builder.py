from __future__ import annotations

from collections.abc import Iterable

from core.model_context.oto_rows import iter_model_contexts_from_source_oto, model_context_from_row_spec
from core.model_context.types import BoundaryDecodeResult, DecodedOtoRow, ModelSampleContext, OtoRowSpec


def build_model_sample_contexts(*args, audio_mode: str = "duration", **kwargs) -> list[ModelSampleContext]:
    return list(iter_model_contexts_from_source_oto(*args, audio_mode=audio_mode, **kwargs))


def context_to_decoded_row(context: ModelSampleContext) -> DecodedOtoRow:
    spec = OtoRowSpec(
        wav_name=context.wav.wav_name,
        wav_path=context.wav.wav_path,
        alias=context.alias.alias,
        role=context.alias.alias_role,
        slot_index=context.alias.slot_index,
        slot_count=context.alias.slot_count,
        prev_alias=context.alias.prev_alias,
        next_alias=context.alias.next_alias,
        language=context.wav.language,
        format_type=context.wav.format_type,
        line_index=context.alias.line_index,
        duration_ms=context.oto.duration_ms,
        source_params=dict(context.oto.raw_params),
        meta=dict(context.meta or {}),
    )
    return DecodedOtoRow(
        spec=spec,
        anchors=context.oto.to_absolute_anchors(),
        selected_time_ms=context.oto.anchor_mapping()["pre_abs"],
        fallback_used=not context.oto.usable_as_target,
        reason=context.oto.reason or "model_context",
        quality_score=1.0 if context.oto.usable_as_target else 0.0,
    )


def contexts_to_boundary_decode_result(
    contexts: Iterable[ModelSampleContext],
    *,
    wav_path: str | None = None,
    duration_ms: float | None = None,
) -> BoundaryDecodeResult:
    rows = [context_to_decoded_row(context) for context in contexts]
    first = rows[0].spec if rows else None
    timeline = sorted({float(row.anchors.pre_abs) for row in rows})
    return BoundaryDecodeResult(
        wav_path=str(wav_path or (first.wav_path if first else "")),
        duration_ms=float(duration_ms if duration_ms is not None else (first.duration_ms if first else 0.0)),
        rows=rows,
        anchor_timeline_ms=timeline,
        fallback_count=sum(1 for row in rows if row.fallback_used),
    )


def row_specs_to_model_contexts(row_specs_by_wav, *, audio_mode: str = "duration") -> list[ModelSampleContext]:
    out: list[ModelSampleContext] = []
    for rows in row_specs_by_wav.values():
        for spec in rows:
            out.append(model_context_from_row_spec(spec, audio_mode=audio_mode))
    return out
