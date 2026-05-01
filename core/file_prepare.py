from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, MutableMapping, Optional, Sequence


def _norm_path_key(path: str) -> str:
    try:
        return str(path or "").strip().replace("\\", "/").lower()
    except Exception:
        return str(path or "")


def _guess_alignment_source_from_path(tg_path: str) -> tuple[str, str]:
    norm = _norm_path_key(tg_path)
    if not norm:
        return "", ""
    parts = [p for p in re.split(r"[\\/]+", norm) if p]
    if any(p in {"sequence", "sequence_words", "sequence_aligner", "seq_align", "seq_aligner"} for p in parts):
        return "sequence", "path_hint"
    if any(p in {"mfa", "montreal", "montreal_forced_aligner", "montreal-forced-aligner"} for p in parts):
        return "mfa", "path_hint"
    return "", ""


def _probe_alignment_source_sidecar(tg_path: str) -> tuple[str, str, dict[str, object]]:
    base, _ = os.path.splitext(os.path.abspath(str(tg_path or "").strip()))
    if not base:
        return "", "", {}
    seq_conf = base + ".seqconf.json"
    if os.path.isfile(seq_conf):
        return "sequence", "sidecar", {"path": seq_conf}
    return "", "", {}


def _detect_alignment_source_from_tg(tg) -> tuple[str, str, dict[str, object]]:
    if tg is None:
        return "", "", {}
    for tier in tg:
        tier_name = str(getattr(tier, "name", "") or "").strip().lower()
        if tier_name not in {"utoa_meta", "aligner_meta", "meta", "metadata"}:
            continue
        try:
            intervals = list(tier)
        except Exception:
            intervals = []
        for interval in intervals[:4]:
            mark = str(getattr(interval, "mark", "") or "").strip().lower()
            if not mark:
                continue
            if ("aligner=sequence" in mark) or ("source=sequence" in mark) or (mark == "sequence"):
                return "sequence", "meta_tier", {"tier": tier_name, "mark": mark}
            if ("aligner=mfa" in mark) or ("source=mfa" in mark) or (mark == "mfa"):
                return "mfa", "meta_tier", {"tier": tier_name, "mark": mark}
    return "", "", {}


@dataclass(frozen=True)
class NormalizedInterval:
    minTime: float
    maxTime: float
    mark: str


class NormalizedIntervalTier:
    def __init__(self, source_tier: object, intervals: Sequence[NormalizedInterval]):
        self.source_tier = source_tier
        self.name = str(getattr(source_tier, "name", "") or "")
        self.intervals = list(intervals or [])

    def __iter__(self):
        return iter(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, index):
        return self.intervals[index]


def _iter_tier_entries(tier: object):
    for attr in ("intervals", "_intervals", "entries", "items"):
        entries = getattr(tier, attr, None)
        if entries is not None and not callable(entries):
            try:
                return list(entries)
            except Exception:
                pass
    try:
        return list(tier)
    except Exception:
        return []


def _get_first_existing_attr(obj: object, names: Sequence[str]):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if not callable(value):
                return value
    return None


def _normalize_interval_entry(entry: object) -> Optional[NormalizedInterval]:
    start = _get_first_existing_attr(entry, ("minTime", "xmin", "start", "start_time", "min_time"))
    end = _get_first_existing_attr(entry, ("maxTime", "xmax", "end", "end_time", "max_time"))
    mark = _get_first_existing_attr(entry, ("mark", "text", "label", "name"))
    if start is None or end is None:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            start, end, mark = entry[0], entry[1], entry[2]
        else:
            return None
    try:
        start_f = float(start)
        end_f = float(end)
    except Exception:
        return None
    if end_f < start_f:
        return None
    return NormalizedInterval(start_f, end_f, str(mark or ""))


def coerce_interval_tier(tier: object):
    entries = _iter_tier_entries(tier)
    if not entries:
        return None
    intervals = []
    for entry in entries:
        normalized = _normalize_interval_entry(entry)
        if normalized is None:
            return None
        intervals.append(normalized)
    if not intervals:
        return None
    first = entries[0]
    if (
        hasattr(first, "minTime")
        and hasattr(first, "maxTime")
        and hasattr(first, "mark")
        and hasattr(tier, "__iter__")
    ):
        return tier
    return NormalizedIntervalTier(tier, intervals)


def is_interval_like_tier(tier: object) -> bool:
    return coerce_interval_tier(tier) is not None


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
    alignment_source: str = ""
    alignment_source_reason: str = ""
    alignment_source_meta: dict[str, object] = field(default_factory=dict)


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
    mel_envelope_fn: Callable[..., object],
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
    source_from_info = str(tg_info.get("alignment_source", tg_info.get("source", "")) or "").strip().lower()
    if source_from_info in {"sequence", "mfa"}:
        context.alignment_source = source_from_info
        context.alignment_source_reason = "tg_info"
    else:
        source_hint, reason_hint = _guess_alignment_source_from_path(context.tg_path)
        if source_hint:
            context.alignment_source = source_hint
            context.alignment_source_reason = reason_hint
        sidecar_source, sidecar_reason, sidecar_meta = _probe_alignment_source_sidecar(context.tg_path)
        if sidecar_source:
            context.alignment_source = sidecar_source
            context.alignment_source_reason = sidecar_reason
            context.alignment_source_meta = dict(sidecar_meta or {})

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
        try:
            mel_ctx = mel_envelope_fn(
                audio_sig,
                sr_sig,
                wav_name_hint=(context.output_wav_name or context.real_wav_name),
                wav_path_hint=context.wav_path_for_signal,
            )
        except TypeError:
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
    preloaded_tg_by_path: Optional[Mapping[str, object]] = None,
    phone_tier_name: str = "phones",
    word_tier_name: str = "words",
    tier_predicate: Optional[Callable[[object], bool]] = None,
) -> PreparedFileContext:
    if context.status != "ok":
        return context
    tg = None
    if preloaded_tg_by_path:
        tg = preloaded_tg_by_path.get(_norm_path_key(context.tg_path))
    try:
        if tg is None:
            tg = load_textgrid_fn(context.tg_path)
    except Exception as exc:
        context.status = "textgrid_load_failed"
        context.error_message = str(exc)
        return context

    context.tg = tg
    src, reason, meta = _detect_alignment_source_from_tg(tg)
    if src:
        context.alignment_source = src
        context.alignment_source_reason = reason
        context.alignment_source_meta = dict(meta or {})
    phone_tier_key = str(phone_tier_name or "").strip().lower()
    word_tier_key = str(word_tier_name or "").strip().lower()
    for tier in tg:
        if tier_predicate and not tier_predicate(tier):
            continue
        tier_name = str(getattr(tier, "name", "") or "").strip().lower()
        if tier_name == phone_tier_key:
            context.phone_tier = coerce_interval_tier(tier)
        elif tier_name == word_tier_key:
            context.word_tier = coerce_interval_tier(tier)
    if context.phone_tier is None:
        context.status = "tier_missing"
    return context


__all__ = [
    "NormalizedInterval",
    "NormalizedIntervalTier",
    "PreparedFileContext",
    "coerce_interval_tier",
    "is_interval_like_tier",
    "load_named_tiers",
    "prepare_file_context",
]
