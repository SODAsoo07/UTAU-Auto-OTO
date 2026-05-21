from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.phoneme_boundary.audio import load_wav_mono
from core.phoneme_boundary.plan import build_phoneme_plan


@dataclass(frozen=True)
class BoundaryManifestBuildConfig:
    wav_dir: str
    aliases_path: str = ""
    out_path: str = ""
    language: str = ""
    source_oto_path: str = ""
    sidecar_path: str = ""
    recursive: bool = False
    alias_separator: str = "|"
    split_whitespace: bool = False
    absolute_paths: bool = False
    strict: bool = False
    require_targets: bool = False
    max_rows: int = 0
    sample_rate: int = 16000


@dataclass(frozen=True)
class BoundaryManifestBuildResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def build_boundary_manifest(config: BoundaryManifestBuildConfig) -> BoundaryManifestBuildResult:
    wav_paths = discover_wavs(config.wav_dir, recursive=bool(config.recursive))
    if int(config.max_rows) > 0:
        wav_paths = wav_paths[: int(config.max_rows)]
    if str(config.aliases_path or "").strip():
        alias_rows = load_alias_rows(
            config.aliases_path,
            separator=str(config.alias_separator or "|"),
            split_whitespace=bool(config.split_whitespace),
        )
        alias_source = "aliases"
    elif str(config.source_oto_path or "").strip():
        alias_rows = load_alias_rows_from_oto(config.source_oto_path, wav_paths)
        alias_source = "source_oto_alias_only"
    else:
        alias_rows = []
        alias_source = ""
    sidecar = load_sidecar_rows(config.sidecar_path) if str(config.sidecar_path or "").strip() else {}
    source_oto_timing = load_source_oto_rows(config.source_oto_path) if str(config.source_oto_path or "").strip() else {}
    manifest_dir = os.path.dirname(os.path.abspath(config.out_path)) if config.out_path else os.getcwd()
    rows: list[dict[str, Any]] = []
    missing_alias = 0
    missing_targets = 0
    source_oto_event_rows = 0
    sidecar_row_ids = {id(row) for row in sidecar.values()}
    used_sidecar: set[int] = set()

    for idx, wav_path in enumerate(wav_paths):
        aliases = alias_rows[idx] if idx < len(alias_rows) else []
        if not aliases:
            missing_alias += 1
            if bool(config.strict):
                raise ValueError(f"missing alias row for wav index {idx}: {wav_path}")
            continue
        duration_ms = _duration_ms(wav_path, sample_rate=int(config.sample_rate))
        row = _row_from_wav_aliases(
            wav_path,
            aliases,
            language=str(config.language or ""),
            duration_ms=duration_ms,
            manifest_dir=manifest_dir,
            absolute_paths=bool(config.absolute_paths),
        )
        matches = _sidecar_matches(sidecar, wav_path=wav_path, index=idx)
        if matches:
            sidecar_row = dict(matches[0])
            used_sidecar.add(id(matches[0]))
            _merge_sidecar_targets(row, sidecar_row)
        elif source_oto_timing:
            oto_rows = _source_oto_matches(source_oto_timing, wav_path=wav_path, index=idx)
            if oto_rows:
                events = _events_from_source_oto_rows(
                    row,
                    oto_rows,
                    duration_ms=float(duration_ms),
                )
                if events:
                    row["events"] = events
                    source_oto_event_rows += 1
        trainable = _row_has_training_targets(row)
        row["trainable"] = bool(trainable)
        if not trainable:
            missing_targets += 1
            if bool(config.require_targets):
                if bool(config.strict):
                    raise ValueError(f"sidecar targets missing for wav index {idx}: {wav_path}")
                continue
        rows.append(row)

    if bool(config.strict) and len(alias_rows) != len(wav_paths):
        raise ValueError(f"alias row count ({len(alias_rows)}) does not match wav count ({len(wav_paths)})")

    sidecar_unused = max(0, len(sidecar_row_ids) - len(used_sidecar))
    summary = {
        "wav_dir": os.path.abspath(config.wav_dir),
        "aliases_path": os.path.abspath(config.aliases_path) if str(config.aliases_path or "").strip() else "",
        "source_oto_path": os.path.abspath(config.source_oto_path) if str(config.source_oto_path or "").strip() else "",
        "alias_source": alias_source,
        "sidecar_path": os.path.abspath(config.sidecar_path) if str(config.sidecar_path or "").strip() else "",
        "source_oto_timing_rows": int(source_oto_event_rows),
        "rows": int(len(rows)),
        "wav_count": int(len(wav_paths)),
        "alias_rows": int(len(alias_rows)),
        "missing_alias_rows": int(missing_alias),
        "trainable_rows": int(sum(1 for row in rows if bool(row.get("trainable")))),
        "missing_target_rows": int(missing_targets),
        "unused_alias_rows": max(0, int(len(alias_rows) - len(wav_paths))),
        "unused_sidecar_rows": int(sidecar_unused),
        "language": str(config.language or ""),
        "require_targets": bool(config.require_targets),
    }
    return BoundaryManifestBuildResult(rows=rows, summary=summary)


def discover_wavs(wav_dir: str, *, recursive: bool = False) -> list[str]:
    root = os.path.abspath(str(wav_dir or ""))
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    if recursive:
        for cur, _dirs, files in os.walk(root):
            for name in sorted(files):
                if name.lower().endswith(".wav"):
                    out.append(os.path.join(cur, name))
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(".wav"):
                out.append(path)
    return sorted(out, key=lambda p: _natural_key(os.path.relpath(p, root)))


def load_alias_rows(path: str, *, separator: str = "|", split_whitespace: bool = False) -> list[list[str]]:
    if not path or not os.path.isfile(path):
        return []
    if str(path).lower().endswith(".jsonl"):
        rows: list[list[str]] = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                rows.append(_aliases_from_payload(json.loads(text), separator=separator, split_whitespace=split_whitespace))
        return rows
    if str(path).lower().endswith(".json"):
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [_aliases_from_payload(item, separator=separator, split_whitespace=split_whitespace) for item in payload]
        items = payload.get("rows", payload.get("items", [])) if isinstance(payload, dict) else []
        if isinstance(items, list):
            return [_aliases_from_payload(item, separator=separator, split_whitespace=split_whitespace) for item in items]
        return []
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            text = raw.strip()
            if not text or text.lstrip().startswith("#"):
                continue
            rows.append(_split_alias_line(text, separator=separator, split_whitespace=split_whitespace))
    return rows


def load_alias_rows_from_oto(source_oto_path: str, wav_paths: list[str]) -> list[list[str]]:
    if not source_oto_path or not os.path.isfile(source_oto_path):
        return [[] for _path in wav_paths]
    grouped: dict[str, list[str]] = {}
    text = read_text_with_fallback(source_oto_path)
    for raw in text.splitlines():
        parsed = _parse_oto_alias_only(raw)
        if parsed is None:
            continue
        wav_name, alias = parsed
        if not alias:
            continue
        for key in _path_lookup_keys(wav_name, index=-1):
            if key.startswith("index:") or key == "-1":
                continue
            grouped.setdefault(key, []).append(alias)
    rows: list[list[str]] = []
    for wav_path in wav_paths:
        aliases: list[str] = []
        for key in _path_lookup_keys(wav_path, index=-1):
            if key.startswith("index:") or key == "-1":
                continue
            values = grouped.get(key)
            if values:
                aliases = list(values)
                break
        rows.append(aliases)
    return rows


def load_sidecar_rows(path: str) -> dict[str, dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return {}
    items: list[dict[str, Any]] = []
    if str(path).lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    items.append(payload)
    else:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            items.extend([row for row in payload if isinstance(row, dict)])
        elif isinstance(payload, dict):
            rows = payload.get("rows", payload.get("items", None))
            if isinstance(rows, list):
                items.extend([row for row in rows if isinstance(row, dict)])
            else:
                items.append(payload)
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(items):
        for key in _sidecar_keys(row, index=idx):
            out.setdefault(key, row)
    return out


def load_source_oto_rows(path: str) -> dict[str, list[dict[str, Any]]]:
    if not path or not os.path.isfile(path):
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    text = read_text_with_fallback(path)
    for raw in text.splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "").strip()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav_name or not alias:
            continue
        item = {
            "wav": wav_name,
            "alias": alias,
            "offset": _safe_float(parsed.get("offset", 0.0), 0.0),
            "cons": _safe_float(parsed.get("cons", 0.0), 0.0),
            "cutoff": _safe_float(parsed.get("cutoff", 0.0), 0.0),
            "pre": _safe_float(parsed.get("pre", 0.0), 0.0),
            "ovl": _safe_float(parsed.get("ovl", 0.0), 0.0),
        }
        for key in _path_lookup_keys(wav_name, index=-1):
            if key.startswith("index:") or key == "-1":
                continue
            grouped.setdefault(key, []).append(item)
    return grouped


def _source_oto_matches(source_oto_rows: dict[str, list[dict[str, Any]]], *, wav_path: str, index: int) -> list[dict[str, Any]]:
    keys = _path_lookup_keys(wav_path, index=index)
    for key in keys:
        if key.startswith("index:") or key == str(index):
            continue
        rows = source_oto_rows.get(key)
        if rows:
            return [dict(item) for item in rows]
    return []


def _events_from_source_oto_rows(row: dict[str, Any], source_rows: list[dict[str, Any]], *, duration_ms: float) -> list[dict[str, Any]]:
    aliases = [str(item or "").strip() for item in list(row.get("aliases", []) or [])]
    expected = _expected_phones_by_alias(list(row.get("expected_phones", []) or []))
    events: list[dict[str, Any]] = []
    for alias_index, source_row in enumerate(source_rows):
        if aliases and alias_index >= len(aliases):
            break
        alias = aliases[alias_index] if alias_index < len(aliases) else str(source_row.get("alias", "") or "")
        alias_phones = expected.get(alias_index, [])
        events.extend(
            _events_from_source_oto_rule_row(
                source_row=source_row,
                alias=alias,
                alias_index=alias_index,
                alias_phones=alias_phones,
                duration_ms=float(duration_ms),
            )
        )
    return events


def _events_from_source_oto_rule_row(
    *,
    source_row: dict[str, Any],
    alias: str,
    alias_index: int,
    alias_phones: list[dict[str, Any]],
    duration_ms: float,
) -> list[dict[str, Any]]:
    offset_ms = max(0.0, _safe_float(source_row.get("offset", 0.0), 0.0))
    cutoff_abs_ms = _safe_cutoff_abs_ms(
        offset_ms,
        _safe_float(source_row.get("cutoff", 0.0), 0.0),
        float(duration_ms),
    )
    if cutoff_abs_ms <= (offset_ms + 2.0):
        return []

    pre_abs_ms = _clamp(
        offset_ms + max(0.0, _safe_float(source_row.get("pre", 0.0), 0.0)),
        offset_ms + 1.0,
        cutoff_abs_ms - 1.0,
    )
    cons_abs_ms = _clamp(
        offset_ms + max(0.0, _safe_float(source_row.get("cons", 0.0), 0.0)),
        offset_ms + 1.0,
        cutoff_abs_ms - 1.0,
    )
    if pre_abs_ms <= (offset_ms + 1.5):
        pre_abs_ms = _clamp(cons_abs_ms, offset_ms + 1.0, cutoff_abs_ms - 1.0)

    role = _infer_alias_role(alias, alias_phones)
    vowel_phone = _select_vowel_phone(alias_phones, role=role)
    consonant_phone = _select_consonant_phone(alias_phones)
    cons_span_ms = pre_abs_ms - offset_ms
    vowel_span_ms = cutoff_abs_ms - pre_abs_ms
    cv_boundary_reliable = cons_span_ms >= 8.0 and vowel_span_ms >= 18.0
    can_split_c_v = cv_boundary_reliable and abs(cons_abs_ms - pre_abs_ms) <= 26.0
    events: list[dict[str, Any]] = []

    events.append(_make_event("phone_start", offset_ms, alias_index, "", 0.70, rule=0))
    events.append(_make_event("phone_end", cutoff_abs_ms, alias_index, "", 0.70, rule=0))

    if role in {"cv", "cv_head", "vcv"}:
        # Rule 1: prioritize CV boundary (vowel onset) reliability first.
        events.append(_make_event("vowel_onset", pre_abs_ms, alias_index, vowel_phone, 1.00, rule=1))
        events.append(_make_event("vowel_end", cutoff_abs_ms, alias_index, vowel_phone, 0.80, rule=1))
        nucleus_ms = _estimate_vowel_nucleus_ms(pre_abs_ms, cutoff_abs_ms, role=role)
        events.append(_make_event("vowel_nucleus", nucleus_ms, alias_index, vowel_phone, 0.84, rule=1))
        # Rule 3: split C/V only when CV boundary reliability is sufficient.
        if can_split_c_v:
            events.append(_make_event("consonant_onset", offset_ms, alias_index, consonant_phone, 0.92, rule=3))
    elif role == "vv":
        # Rule 2: for VV, prioritize vowel nucleus localization.
        onset = _clamp(pre_abs_ms, offset_ms + 1.0, cutoff_abs_ms - 1.0)
        nucleus_ms = _estimate_vowel_nucleus_ms(onset, cutoff_abs_ms, role=role)
        events.append(_make_event("vowel_onset", onset, alias_index, vowel_phone, 0.78, rule=2))
        events.append(_make_event("vowel_nucleus", nucleus_ms, alias_index, vowel_phone, 1.00, rule=2))
        events.append(_make_event("vowel_end", cutoff_abs_ms, alias_index, vowel_phone, 0.84, rule=2))
    elif role in {"mono", "v"}:
        onset = _clamp(pre_abs_ms, offset_ms + 1.0, cutoff_abs_ms - 1.0)
        nucleus_ms = _estimate_vowel_nucleus_ms(onset, cutoff_abs_ms, role=role)
        events.append(_make_event("vowel_onset", onset, alias_index, vowel_phone, 0.86, rule=1))
        events.append(_make_event("vowel_nucleus", nucleus_ms, alias_index, vowel_phone, 0.88, rule=1))
        events.append(_make_event("vowel_end", cutoff_abs_ms, alias_index, vowel_phone, 0.80, rule=1))
    elif role == "vc":
        events.append(_make_event("vowel_end", pre_abs_ms, alias_index, vowel_phone, 0.74, rule=1))
    return events


def _expected_phones_by_alias(expected_phones: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for item in expected_phones:
        if not isinstance(item, dict):
            continue
        idx = _safe_int(item.get("alias_index", -1), -1)
        if idx < 0:
            continue
        out.setdefault(idx, []).append(item)
    for idx, items in out.items():
        items.sort(key=lambda row: _safe_int(row.get("phone_index", 0), 0))
        out[idx] = items
    return out


def _infer_alias_role(alias: str, alias_phones: list[dict[str, Any]]) -> str:
    states: list[str] = []
    for item in alias_phones:
        state = _coarse_state(item.get("coarse", ""))
        if state != "silence":
            states.append(state)
    if not states:
        text = str(alias or "").strip().lower()
        if not text:
            return "unknown"
        if text.startswith("- "):
            return "cv_head"
        return "unknown"
    if len(states) >= 2:
        if states[0] == "consonant" and "vowel" in states[1:]:
            text = str(alias or "").strip()
            return "cv_head" if text.startswith("-") else "cv"
        if states[0] == "vowel":
            if "vowel" in states[1:]:
                return "vv"
            if "consonant" in states[1:]:
                return "vc"
    if states[0] == "vowel":
        return "mono"
    if states[0] == "consonant":
        return "cons"
    return "unknown"


def _coarse_state(coarse: object) -> str:
    text = str(coarse or "").strip().upper()
    if text.startswith("C_"):
        return "consonant"
    if text == "V":
        return "vowel"
    return "silence"


def _select_vowel_phone(alias_phones: list[dict[str, Any]], *, role: str) -> str:
    vowels = [str(item.get("phone", "") or "") for item in alias_phones if _coarse_state(item.get("coarse", "")) == "vowel"]
    if not vowels:
        return ""
    if role == "vv":
        return vowels[-1]
    return vowels[0]


def _select_consonant_phone(alias_phones: list[dict[str, Any]]) -> str:
    for item in alias_phones:
        if _coarse_state(item.get("coarse", "")) == "consonant":
            return str(item.get("phone", "") or "")
    return ""


def _estimate_vowel_nucleus_ms(vowel_onset_ms: float, vowel_end_ms: float, *, role: str) -> float:
    start = float(vowel_onset_ms)
    end = float(vowel_end_ms)
    span = max(1.0, end - start)
    if role == "vv":
        ratio = 0.58
    elif role in {"cv", "cv_head", "vcv"}:
        ratio = 0.54
    else:
        ratio = 0.50
    return _clamp(start + (span * ratio), start + 4.0, end - 4.0)


def _make_event(label: str, time_ms: float, alias_index: int, phone: str, confidence: float, *, rule: int) -> dict[str, Any]:
    return {
        "label": str(label),
        "time_ms": max(0.0, float(time_ms)),
        "alias_index": int(alias_index),
        "phone": str(phone or ""),
        "confidence": _clamp(float(confidence), 0.0, 1.0),
        "source": "source_oto_timing_rules",
        "rule": int(rule),
    }


def _safe_cutoff_abs_ms(offset_ms: float, cutoff: float, wav_duration_ms: float) -> float:
    off = max(0.0, float(offset_ms))
    cut = float(cutoff)
    wav_dur = max(0.0, float(wav_duration_ms))
    if cut < 0.0:
        resolved = off + abs(cut)
    else:
        rel = off + cut
        abs_candidate = cut
        candidates = [rel]
        if abs_candidate > off:
            candidates.append(abs_candidate)
        if wav_dur > 1.0:
            candidates = [cand for cand in candidates if cand <= (wav_dur + 2.0)] or candidates
        resolved = min(candidates, key=lambda cand: abs(float(cand) - rel))
    if wav_dur > 1.0:
        resolved = min(resolved, wav_dur - 1.0)
    return max(off + 1.0, float(resolved))


def _clamp(value: float, lo: float, hi: float) -> float:
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return max(lower, min(upper, float(value)))


def load_matching_textgrid_phone_intervals(wav_path: str, *, textgrid_dir: str = "") -> list[dict[str, Any]]:
    path = matching_textgrid_path(wav_path, textgrid_dir=textgrid_dir)
    if not path:
        return []
    return load_textgrid_phone_intervals(path)


def matching_textgrid_path(wav_path: str, *, textgrid_dir: str = "") -> str:
    wav_base = os.path.basename(str(wav_path or ""))
    stem, _ext = os.path.splitext(wav_base)
    candidates = []
    if textgrid_dir:
        candidates.append(os.path.join(textgrid_dir, stem + ".TextGrid"))
        candidates.append(os.path.join(textgrid_dir, stem + ".textgrid"))
    candidates.append(os.path.splitext(str(wav_path))[0] + ".TextGrid")
    candidates.append(os.path.splitext(str(wav_path))[0] + ".textgrid")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def load_textgrid_phone_intervals(path: str) -> list[dict[str, Any]]:
    text = read_text_with_fallback(path)
    tiers: list[dict[str, Any]] = []
    tier: dict[str, Any] | None = None
    interval: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("item ["):
            if interval is not None and tier is not None:
                tier.setdefault("intervals", []).append(interval)
                interval = None
            if tier is not None:
                tiers.append(tier)
            tier = {"name": "", "intervals": []}
            continue
        if tier is None:
            continue
        if line.startswith("name") and "=" in line:
            tier["name"] = _unquote_textgrid_value(line.split("=", 1)[1])
            continue
        if line.startswith("intervals ["):
            if interval is not None:
                tier.setdefault("intervals", []).append(interval)
            interval = {}
            continue
        if interval is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key in {"xmin", "xmax"}:
            try:
                interval[key] = float(value)
            except ValueError:
                pass
        elif key == "text":
            interval["text"] = _unquote_textgrid_value(value)
    if interval is not None and tier is not None:
        tier.setdefault("intervals", []).append(interval)
    if tier is not None:
        tiers.append(tier)
    phone_tier = _select_phone_tier(tiers)
    out: list[dict[str, Any]] = []
    for item in phone_tier:
        try:
            start_ms = float(item.get("xmin", 0.0)) * 1000.0
            end_ms = float(item.get("xmax", 0.0)) * 1000.0
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        phone = _normalize_frame_phone(item.get("text", ""))
        row = {
            "phone": phone,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "source": "textgrid",
        }
        if phone not in {"sil", "br"}:
            row["confidence"] = 0.92
        out.append(row)
    return out


def write_manifest_jsonl(path: str, rows: list[dict[str, Any]]) -> int:
    out_path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return int(len(rows))


def _row_from_wav_aliases(
    wav_path: str,
    aliases: list[str],
    *,
    language: str,
    duration_ms: float,
    manifest_dir: str,
    absolute_paths: bool,
) -> dict[str, Any]:
    stored_wav = os.path.abspath(wav_path) if absolute_paths else os.path.relpath(os.path.abspath(wav_path), manifest_dir)
    plan = build_phoneme_plan(wav_path=wav_path, aliases=aliases, language=language, duration_ms=duration_ms)
    phones = []
    for alias_plan in plan.aliases:
        for phone in alias_plan.phones:
            phones.append(
                {
                    "phone": phone.text,
                    "phone_index": int(phone.index),
                    "alias": alias_plan.alias,
                    "alias_index": int(alias_plan.alias_index),
                    "coarse": phone.coarse,
                }
            )
    return {
        "wav_path": stored_wav,
        "wav_name": os.path.basename(str(wav_path)),
        "language": str(language or ""),
        "duration_ms": float(duration_ms),
        "aliases": list(aliases),
        "expected_phones": phones,
        "source": "phoneme_boundary_manifest_builder",
    }


def _aliases_from_payload(payload: Any, *, separator: str, split_whitespace: bool) -> list[str]:
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        raw = payload.get("aliases", payload.get("alias_list", payload.get("pronunciation", payload.get("text", ""))))
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return _split_alias_line(str(raw or ""), separator=separator, split_whitespace=split_whitespace)
    return _split_alias_line(str(payload or ""), separator=separator, split_whitespace=split_whitespace)


def _parse_oto_alias_only(line: str) -> tuple[str, str] | None:
    text = str(line or "").strip()
    if not text or "=" not in text:
        return None
    wav_name, rest = text.split("=", 1)
    if "," not in rest:
        return None
    alias = rest.split(",", 1)[0].strip()
    wav = wav_name.strip().replace("\\", "/")
    if not wav or not alias:
        return None
    return wav, alias


def _split_alias_line(text: str, *, separator: str, split_whitespace: bool) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    sep = str(separator or "|")
    if sep and sep in value:
        return [part.strip() for part in value.split(sep) if part.strip()]
    if "\t" in value:
        return [part.strip() for part in value.split("\t") if part.strip()]
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    if bool(split_whitespace):
        return [part.strip() for part in value.split() if part.strip()]
    return [value]


def _sidecar_matches(sidecar: dict[str, dict[str, Any]], *, wav_path: str, index: int) -> tuple[dict[str, Any], str] | None:
    keys = _path_lookup_keys(wav_path, index=index)
    for key in keys:
        row = sidecar.get(key)
        if row is not None:
            return row, key
    return None


def _merge_sidecar_targets(row: dict[str, Any], sidecar_row: dict[str, Any]) -> None:
    for key in ("phones", "phone_intervals", "events", "boundaries", "boundary_events"):
        value = sidecar_row.get(key)
        if isinstance(value, list):
            row[key] = value
    if "phones" not in row and "phone_intervals" not in row:
        intervals = _frame_phone_intervals_from_sidecar(sidecar_row)
        if intervals:
            row["phones"] = intervals
    for key in ("speaker", "voicebank_id", "recording_id", "sample_weight", "confidence"):
        if key in sidecar_row:
            row[key] = sidecar_row[key]
    meta = {k: v for k, v in sidecar_row.items() if k not in set(row) and k not in {"wav", "audio", "wav_path"}}
    if meta:
        row["sidecar_meta"] = meta


def _row_has_training_targets(row: dict[str, Any]) -> bool:
    for key in ("phones", "phone_intervals", "events", "boundaries", "boundary_events"):
        value = row.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def _frame_phone_intervals_from_sidecar(sidecar_row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_frames = sidecar_row.get("phone", sidecar_row.get("phones_by_frame", sidecar_row.get("frame_phones", None)))
    if not isinstance(raw_frames, list) or not raw_frames:
        return []
    try:
        hop_ms = float(sidecar_row.get("hop_ms", sidecar_row.get("frame_ms", 10.0)))
    except Exception:
        hop_ms = 10.0
    hop_ms = max(1e-6, hop_ms)
    intervals: list[dict[str, Any]] = []
    current = _normalize_frame_phone(raw_frames[0])
    start = 0
    for idx, raw_phone in enumerate(raw_frames[1:], start=1):
        phone = _normalize_frame_phone(raw_phone)
        if phone == current:
            continue
        _append_frame_interval(intervals, current, start, idx, hop_ms)
        current = phone
        start = idx
    _append_frame_interval(intervals, current, start, len(raw_frames), hop_ms)
    return intervals


def _append_frame_interval(
    intervals: list[dict[str, Any]],
    phone: str,
    start_frame: int,
    end_frame: int,
    hop_ms: float,
) -> None:
    if end_frame <= start_frame:
        return
    start_ms = float(start_frame) * float(hop_ms)
    end_ms = float(end_frame) * float(hop_ms)
    item = {
        "phone": phone,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source": "frame_phone_sidecar",
    }
    if phone not in {"sil", "br"}:
        item["confidence"] = 0.85
    intervals.append(item)


def _normalize_frame_phone(phone: object) -> str:
    text = str(phone or "").strip()
    if not text or text in {"_", "-", "silence", "sp"}:
        return "sil"
    return text


def _unquote_textgrid_value(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    elif text.startswith('"'):
        text = text[1:]
    return text.strip()


def _select_phone_tier(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for tier in tiers:
        name = str(tier.get("name", "") or "").strip().lower()
        if name in {"phones", "phone", "phoneme", "phonemes"}:
            return list(tier.get("intervals", []) or [])
    for tier in tiers:
        intervals = list(tier.get("intervals", []) or [])
        if intervals:
            return intervals
    return []


def _sidecar_keys(row: dict[str, Any], *, index: int) -> list[str]:
    out = {f"index:{index}", str(index)}
    for key in ("wav_path", "wav", "audio", "wav_name", "file", "filename"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        out.update(_path_lookup_keys(value, index=index))
    return sorted(out)


def _path_lookup_keys(path: str, *, index: int) -> list[str]:
    text = str(path or "").replace("\\", "/").strip()
    base = os.path.basename(text)
    stem, _ext = os.path.splitext(base)
    keys = {
        f"index:{int(index)}",
        str(int(index)),
        text.lower(),
        base.lower(),
        stem.lower(),
    }
    try:
        keys.add(os.path.abspath(path).replace("\\", "/").lower())
    except Exception:
        pass
    return sorted(k for k in keys if k)


def _duration_ms(path: str, *, sample_rate: int) -> float:
    try:
        _samples, _sr, duration_sec = load_wav_mono(path, target_sr=int(sample_rate))
        return float(duration_sec) * 1000.0
    except Exception:
        return 0.0


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _natural_key(text: str) -> list[object]:
    parts: list[object] = []
    buf = ""
    in_digit = False
    for ch in str(text):
        if ch.isdigit() != in_digit and buf:
            parts.append(int(buf) if in_digit else buf.lower())
            buf = ""
        in_digit = ch.isdigit()
        buf += ch
    if buf:
        parts.append(int(buf) if in_digit else buf.lower())
    return parts


__all__ = [
    "BoundaryManifestBuildConfig",
    "BoundaryManifestBuildResult",
    "build_boundary_manifest",
    "discover_wavs",
    "load_alias_rows",
    "load_alias_rows_from_oto",
    "load_source_oto_rows",
    "load_matching_textgrid_phone_intervals",
    "load_sidecar_rows",
    "load_textgrid_phone_intervals",
    "matching_textgrid_path",
    "write_manifest_jsonl",
]
