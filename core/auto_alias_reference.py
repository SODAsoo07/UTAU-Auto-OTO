from __future__ import annotations

import os
from typing import Callable

from core.format_type_utils import (
    normalize_auto_format_value,
    normalize_format_type,
    normalize_language_name,
)
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_normalization import canonicalize_alias_for_matching

_ALIAS_REFERENCE_CACHE: dict[tuple[str, str, tuple[str, ...]], dict[str, object]] = {}


def _split_env_paths(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ";").split(";")]
    return [p for p in parts if p]


def _normalize_existing_dirs(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        try:
            ab = os.path.abspath(str(item or "").strip())
        except Exception:
            continue
        if not ab or not os.path.isdir(ab):
            continue
        key = os.path.normcase(ab)
        if key in seen:
            continue
        seen.add(key)
        out.append(ab)
    return out


def _guess_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _discover_reference_roots(*, wav_root: str = "", app_dir: str = "") -> list[str]:
    roots: list[str] = []
    wr = str(wav_root or "").strip()
    if wr:
        wav_abs = os.path.abspath(wr)
        roots.extend(
            [
                os.path.join(wav_abs, "data"),
                os.path.join(wav_abs, "Data"),
                os.path.join(wav_abs, "reference"),
                os.path.join(wav_abs, "references"),
                wav_abs,
                os.path.join(os.path.dirname(wav_abs), "data"),
                os.path.join(os.path.dirname(wav_abs), "Data"),
            ]
        )

    base_dir = str(app_dir or "").strip() or _guess_repo_root()
    roots.extend(
        [
            os.path.join(base_dir, "assets", "data", "oto_aliases"),
            os.path.join(base_dir, "assets", "profiles", "oto_aliases"),
            os.path.join(base_dir, "data", "oto_aliases"),
            os.path.join(base_dir, "data"),
        ]
    )

    roots.extend(_split_env_paths(os.environ.get("UTOA_AUTO_ALIAS_DATA_DIRS", "")))
    return _normalize_existing_dirs(roots)


def _build_roots_signature(roots: list[str]) -> tuple[tuple[str, int], ...]:
    items: list[tuple[str, int]] = []
    for root in roots:
        try:
            mtime_ns = int(os.stat(root).st_mtime_ns)
        except Exception:
            mtime_ns = -1
        items.append((os.path.normcase(os.path.abspath(root)), mtime_ns))
    return tuple(items)


def _clone_reference_payload(payload: dict[str, object]) -> dict[str, object]:
    aliases = payload.get("aliases")
    by_type = payload.get("by_type")
    format_counts = payload.get("format_counts")
    return {
        "language": str(payload.get("language", "") or ""),
        "format_type": str(payload.get("format_type", "") or ""),
        "aliases": set(aliases or []) if isinstance(aliases, set) else set(),
        "by_type": {
            str(k): set(v or [])
            for k, v in (by_type.items() if isinstance(by_type, dict) else [])
        },
        "format_counts": {
            str(k): int(v)
            for k, v in (format_counts.items() if isinstance(format_counts, dict) else [])
        },
        "roots": list(payload.get("roots") or []),
        "scanned_files": int(payload.get("scanned_files", 0) or 0),
        "matched_files": int(payload.get("matched_files", 0) or 0),
    }


def _iter_oto_ini_candidates(root_dir: str, *, max_depth: int = 3, max_files: int = 120):
    root = os.path.abspath(str(root_dir or "").strip())
    if not root or not os.path.isdir(root):
        return
    root_depth = root.rstrip("\\/").count(os.sep)
    emitted = 0
    for current_root, dir_names, file_names in os.walk(root):
        current_depth = current_root.rstrip("\\/").count(os.sep) - root_depth
        if current_depth > max_depth:
            dir_names[:] = []
            continue
        for file_name in file_names:
            lower_name = str(file_name or "").strip().lower()
            if not lower_name.endswith(".ini"):
                continue
            if ("oto" not in lower_name) and ("base" not in lower_name):
                continue
            path = os.path.join(current_root, file_name)
            if not os.path.isfile(path):
                continue
            yield path
            emitted += 1
            if emitted >= max_files:
                return


def _load_oto_aliases(path: str, *, language: str, max_aliases: int = 480) -> list[str]:
    aliases: list[str] = []
    for raw in read_text_with_fallback(path).splitlines():
        row = parse_oto_line(raw)
        if not row:
            continue
        alias_raw = str(row.get("alias", "") or "").strip()
        if not alias_raw:
            continue
        alias_norm = canonicalize_alias_for_matching(language, alias_raw)
        if not alias_norm:
            continue
        aliases.append(alias_norm)
        if len(aliases) >= max_aliases:
            break
    return aliases


def _resolve_format_classifier(language: str):
    lang = normalize_language_name(language)
    if lang == "japanese":
        from core.ja_oto_mapping import classify_ja_alias, detect_ja_alias_format

        return detect_ja_alias_format, classify_ja_alias
    from core.kr_oto_rules import classify_alias, detect_alias_format

    return detect_alias_format, classify_alias


def _is_compatible_format(target_format: str, file_format: str) -> bool:
    target = str(target_format or "").strip().lower()
    current = str(file_format or "").strip().lower()
    if not target:
        return True
    if not current:
        return False
    if current == target:
        return True
    if target == "cvvc" and current in {"cvc", "vcv"}:
        return True
    if target == "cvc" and current in {"cvvc"}:
        return True
    if target == "cv" and current in {"cvc"}:
        return True
    if target == "vcv" and current in {"cvvc"}:
        return True
    return False


def collect_auto_alias_reference(
    *,
    language: str,
    format_type: str,
    wav_root: str = "",
    app_dir: str = "",
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, object]:
    lang = normalize_language_name(language)
    target_format = normalize_auto_format_value(lang, format_type) or normalize_format_type(lang, format_type)
    detect_format_fn, classify_alias_fn = _resolve_format_classifier(lang)

    roots = _discover_reference_roots(wav_root=wav_root, app_dir=app_dir)
    cache_key = (
        lang,
        str(target_format or ""),
        tuple(os.path.normcase(os.path.abspath(path)) for path in roots),
    )
    roots_signature = _build_roots_signature(roots)
    cached = _ALIAS_REFERENCE_CACHE.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == roots_signature:
        payload = _clone_reference_payload(dict(cached.get("payload") or {}))
        if callable(log_fn):
            log_fn(
                "[AutoAlias] reference cache hit: "
                f"language={lang}, target_format={target_format or 'auto'}, "
                f"files={payload.get('matched_files', 0)}/{payload.get('scanned_files', 0)}, "
                f"aliases={len(payload.get('aliases', set()))}"
            )
        return payload

    alias_all: set[str] = set()
    alias_by_type: dict[str, set[str]] = {}
    scanned_files = 0
    matched_files = 0
    format_counts: dict[str, int] = {}

    for root in roots:
        for ini_path in _iter_oto_ini_candidates(root):
            scanned_files += 1
            aliases = _load_oto_aliases(ini_path, language=lang)
            if len(aliases) < 6:
                continue
            try:
                detected_raw = detect_format_fn(aliases, custom_map=None)
            except TypeError:
                detected_raw = detect_format_fn(aliases)
            except Exception:
                continue
            detected_fmt = normalize_format_type(lang, str(detected_raw or "").strip())
            detected_fmt = detected_fmt or "unknown"
            format_counts[detected_fmt] = int(format_counts.get(detected_fmt, 0)) + 1
            if not _is_compatible_format(target_format, detected_fmt):
                continue
            matched_files += 1
            for alias in aliases:
                alias_all.add(alias)
                try:
                    alias_type = str(classify_alias_fn(alias, None) or "").strip().lower()
                except TypeError:
                    alias_type = str(classify_alias_fn(alias) or "").strip().lower()
                except Exception:
                    alias_type = ""
                alias_type = alias_type or "other"
                alias_by_type.setdefault(alias_type, set()).add(alias)

    if callable(log_fn):
        log_fn(
            "[AutoAlias] reference scan: "
            f"language={lang}, target_format={target_format or 'auto'}, "
            f"roots={len(roots)}, files={matched_files}/{scanned_files}, aliases={len(alias_all)}"
        )
        if format_counts:
            parts = ", ".join(
                f"{fmt}={cnt}" for fmt, cnt in sorted(format_counts.items(), key=lambda item: item[0])
            )
            log_fn(f"[AutoAlias] detected format distribution: {parts}")

    payload = {
        "language": lang,
        "format_type": target_format,
        "aliases": alias_all,
        "by_type": alias_by_type,
        "format_counts": format_counts,
        "roots": roots,
        "scanned_files": scanned_files,
        "matched_files": matched_files,
    }
    _ALIAS_REFERENCE_CACHE[cache_key] = {
        "signature": roots_signature,
        "payload": _clone_reference_payload(payload),
    }
    return _clone_reference_payload(payload)


def choose_alias_candidate(
    *,
    language: str,
    candidates: list[str],
    alias_reference: dict[str, object] | None = None,
    alias_type_hint: str = "",
) -> str:
    cand_list = [str(c or "").strip() for c in (candidates or []) if str(c or "").strip()]
    if not cand_list:
        return ""
    if not isinstance(alias_reference, dict):
        return cand_list[0]

    lang = normalize_language_name(language)
    all_aliases = alias_reference.get("aliases")
    if not isinstance(all_aliases, set) or not all_aliases:
        return cand_list[0]

    type_set: set[str] = set()
    hinted_type = str(alias_type_hint or "").strip().lower()
    by_type = alias_reference.get("by_type")
    if hinted_type and isinstance(by_type, dict):
        hinted_values = by_type.get(hinted_type)
        if isinstance(hinted_values, set):
            type_set = hinted_values

    for cand in cand_list:
        norm = canonicalize_alias_for_matching(lang, cand)
        if norm and type_set and norm in type_set:
            return cand
    for cand in cand_list:
        norm = canonicalize_alias_for_matching(lang, cand)
        if norm and norm in all_aliases:
            return cand
    return cand_list[0]


__all__ = [
    "collect_auto_alias_reference",
    "choose_alias_candidate",
]
