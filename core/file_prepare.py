from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, MutableMapping, Optional, Sequence


@dataclass
class PreparedFileContext:
    fname: str
    lines: Sequence[str]
    status: str = "ok"
    tg_info: Optional[Mapping[str, object]] = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    error_message: str = ""
    tg_path: str = ""
    real_wav_name: str = ""
    output_wav_name: str = ""
    wav_path_for_signal: str = ""
    mel_ctx_for_file: object = None
    wav_duration_ms: float = 0.0
    tg: object = None
    phone_tier: object = None
    word_tier: object = None


def prepare_file_context(
    *,
    fname: str,
    lines: Sequence[str],
    resolve_tg_info_fn: Callable[[str], object],
    missing_diagnostics_fn: Optional[Callable[[str], dict[str, object]]] = None,
    wav_root_for_signal: str,
    wav_index_for_signal: Mapping[str, str],
    mel_cache_for_signal: MutableMapping[str, object],
    find_wav_path_fn: Callable[[str, str, Mapping[str, str]], str],
    read_wav_fn: Callable[[str], tuple[object, int]],
    mel_envelope_fn: Callable[[object, int], object],
    wav_duration_fn: Optional[Callable[[str], float]] = None,
) -> PreparedFileContext:
    context = PreparedFileContext(fname=fname, lines=lines)
    tg_info = resolve_tg_info_fn(fname)
    if not tg_info:
        context.status = "textgrid_missing"
        if missing_diagnostics_fn:
            try:
                context.diagnostics = dict(missing_diagnostics_fn(fname) or {})
            except Exception:
                context.diagnostics = {}
        return context

    context.tg_info = tg_info
    context.tg_path = str(tg_info.get("path", "") or "")
    context.real_wav_name = str(tg_info.get("real_name", "") or "")
    context.output_wav_name = str(tg_info.get("output_name", context.real_wav_name) or context.real_wav_name)
    context.wav_path_for_signal = find_wav_path_fn(
        context.output_wav_name or context.real_wav_name,
        wav_root_for_signal,
        wav_index_for_signal,
    )
    if not context.wav_path_for_signal:
        return context

    mel_ctx = mel_cache_for_signal.get(context.wav_path_for_signal)
    if mel_ctx is None:
        audio_sig, sr_sig = read_wav_fn(context.wav_path_for_signal)
        mel_ctx = mel_envelope_fn(audio_sig, sr_sig)
        mel_cache_for_signal[context.wav_path_for_signal] = mel_ctx
        if sr_sig:
            context.wav_duration_ms = (len(audio_sig) / float(sr_sig)) * 1000.0
    else:
        if wav_duration_fn:
            try:
                context.wav_duration_ms = float(wav_duration_fn(context.wav_path_for_signal) or 0.0)
            except Exception:
                context.wav_duration_ms = 0.0
    context.mel_ctx_for_file = mel_ctx
    return context


def load_named_tiers(
    context: PreparedFileContext,
    *,
    load_textgrid_fn: Callable[[str], object],
    phone_tier_name: str = "phones",
    word_tier_name: str = "words",
    tier_predicate: Optional[Callable[[object], bool]] = None,
) -> PreparedFileContext:
    if context.status != "ok":
        return context
    try:
        tg = load_textgrid_fn(context.tg_path)
    except Exception as exc:
        context.status = "textgrid_load_failed"
        context.error_message = str(exc)
        return context

    context.tg = tg
    for tier in tg:
        if tier_predicate and not tier_predicate(tier):
            continue
        tier_name = getattr(tier, "name", None)
        if tier_name == phone_tier_name:
            context.phone_tier = tier
        elif tier_name == word_tier_name:
            context.word_tier = tier
    if context.phone_tier is None:
        context.status = "tier_missing"
    return context


__all__ = [
    "PreparedFileContext",
    "load_named_tiers",
    "prepare_file_context",
]
