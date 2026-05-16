from __future__ import annotations

import os
from typing import Callable

from core.oto_normalization import normalize_wav_key
from core.oto_generator import classify_alias as classify_kr_alias
from core.oto_generator import validate_oto_params


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


def _normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _apply_suffix_to_oto_line(line: str, suffix: str) -> str:
    normalized_suffix = _normalize_alias_suffix(suffix)
    if not normalized_suffix or "=" not in str(line or ""):
        return line
    left, right = str(line).split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        alias = alias.strip()
        if alias:
            alias = f"{alias}_{normalized_suffix}"
        return f"{left}={alias},{rest}"
    alias = right.strip()
    if alias:
        alias = f"{alias}_{normalized_suffix}"
    return f"{left}={alias}"


def _has_invalid_param_order(offset: float, cons: float, cutoff: float, pre: float, ovl: float) -> bool:
    # Same order constraint used by validator:
    # off <= off+ovl <= off+pre <= off+cons < off+abs(cutoff)
    # which reduces to:
    # 0 <= ovl <= pre <= cons < abs(cutoff)
    try:
        return not (0.0 <= float(ovl) <= float(pre) <= float(cons) < abs(float(cutoff)))
    except Exception:
        return True


def _sanitize_oto_timing_if_needed(line: str) -> tuple[str, bool]:
    text = str(line or "").strip()
    if not text or "=" not in text:
        return text, False
    left, right = text.split("=", 1)
    parts = [p.strip() for p in right.split(",")]
    if len(parts) < 6:
        return text, False

    alias = str(parts[0] or "").strip()
    try:
        offset = float(parts[1])
        cons = float(parts[2])
        cutoff = float(parts[3])
        pre = float(parts[4])
        ovl = float(parts[5])
    except Exception:
        return text, False

    if not _has_invalid_param_order(offset, cons, cutoff, pre, ovl):
        return text, False

    alias_type = ""
    try:
        alias_type = str(classify_kr_alias(alias) or "")
    except Exception:
        alias_type = ""

    offset, cons, cutoff, pre, ovl = validate_oto_params(
        offset,
        cons,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
    )
    rebuilt = [
        alias,
        f"{float(offset):.2f}",
        f"{float(cons):.2f}",
        f"{float(cutoff):.2f}",
        f"{float(pre):.2f}",
        f"{float(ovl):.2f}",
    ]
    if len(parts) > 6:
        rebuilt.extend(parts[6:])
    return f"{left}={','.join(rebuilt)}", True


def _pick_existing_oto(path: str) -> str:
    base = str(path or "").strip()
    if not base:
        return ""
    if os.path.isfile(base):
        return os.path.abspath(base)
    if os.path.isdir(base):
        for name in ("baseoto.ini", "oto.ini"):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return ""


def resolve_kr_cmpx_preview_source_oto(*, wav_dir: str, source_hint: str = "") -> str:
    candidates: list[str] = []
    hint = _pick_existing_oto(source_hint)
    if hint:
        candidates.append(hint)

    env_oto = _pick_existing_oto(os.environ.get("UTOA_KR_CMPX_PREVIEW_SOURCE_OTO", ""))
    if env_oto:
        candidates.append(env_oto)

    env_bank = _pick_existing_oto(os.environ.get("UTOA_KR_CMPX_PREVIEW_SOURCE_BANK", ""))
    if env_bank:
        candidates.append(env_bank)

    wav_dir_abs = os.path.abspath(str(wav_dir or "").strip())
    if wav_dir_abs and os.path.isdir(wav_dir_abs):
        wav_base = os.path.basename(wav_dir_abs).strip().lower()
        parent = os.path.dirname(wav_dir_abs)
        if os.path.isdir(parent):
            sibling_candidates: list[tuple[int, str]] = []
            for child in os.listdir(parent):
                child_path = os.path.join(parent, child)
                if not os.path.isdir(child_path):
                    continue
                child_low = str(child).strip().lower()
                if child_low == wav_base:
                    continue
                if "cmpx" not in child_low:
                    continue
                picked = _pick_existing_oto(child_path)
                if picked:
                    score = 10
                    if "autooto" in child_low or "auto_oto" in child_low:
                        score = 0
                    elif "auto" in child_low:
                        score = 3
                    sibling_candidates.append((score, picked))
            sibling_candidates.sort(key=lambda item: (item[0], item[1].lower()))
            for _score, picked in sibling_candidates:
                candidates.append(picked)
        for local_name in ("baseoto.ini", "oto.ini"):
            local_oto = os.path.join(wav_dir_abs, local_name)
            if os.path.isfile(local_oto):
                candidates.append(os.path.abspath(local_oto))

    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        return candidate
    return ""


def _load_oto_lines(source_oto_path: str) -> list[str]:
    path = str(source_oto_path or "").strip()
    if not path:
        return []
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                rows = []
                for raw in handle:
                    line = str(raw or "").strip()
                    if not line or "=" not in line:
                        continue
                    rows.append(line)
                return rows
        except Exception:
            continue
    return []


def _build_wav_lookup(wav_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    root = os.path.abspath(str(wav_dir or "").strip())
    if not os.path.isdir(root):
        return exact, normalized
    for current_root, _dirs, files in os.walk(root):
        for file_name in files:
            if not str(file_name).lower().endswith(".wav"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            key = str(file_name).strip().lower()
            exact.setdefault(key, rel_path)
            norm = normalize_wav_key(file_name)
            if norm:
                normalized.setdefault(norm, rel_path)
    return exact, normalized


def generate_kr_cmpx_preview_oto(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    alias_suffix: str = "",
    callback: Callable[[str], None] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_kr_cmpx_preview_source_oto(
        wav_dir=wav_dir,
        source_hint=source_oto_path,
    )
    if not source:
        return 0, 0, ["CMPX preview source oto.ini를 찾지 못했습니다."]

    source_rows = _load_oto_lines(source)
    if not source_rows:
        return 0, 0, [f"CMPX preview source OTO 읽기 실패: {source}"]

    exact_lookup, norm_lookup = _build_wav_lookup(wav_dir)
    if not exact_lookup and not norm_lookup:
        return 0, len(source_rows), [f"WAV 파일을 찾지 못했습니다: {wav_dir}"]

    out_lines: list[str] = []
    missing_wavs: set[str] = set()
    seen: set[tuple[str, str]] = set()
    exact_hits = 0
    norm_hits = 0
    adjusted_param_order = 0
    total = 0
    for raw_line in source_rows:
        if "=" not in raw_line:
            continue
        total += 1
        left, right = raw_line.split("=", 1)
        source_wav_name = os.path.basename(str(left or "").replace("\\", "/").strip())
        source_wav_name = source_wav_name or str(left or "").strip()
        mapped_wav = exact_lookup.get(source_wav_name.lower(), "")
        if mapped_wav:
            exact_hits += 1
        else:
            mapped_wav = norm_lookup.get(normalize_wav_key(source_wav_name), "")
            if mapped_wav:
                norm_hits += 1
        if not mapped_wav:
            missing_wavs.add(source_wav_name)
            continue

        candidate = f"{mapped_wav}={right.strip()}"
        candidate, adjusted = _sanitize_oto_timing_if_needed(candidate)
        if adjusted:
            adjusted_param_order += 1
        candidate = _apply_suffix_to_oto_line(candidate, alias_suffix)
        alias_key = ""
        if "=" in candidate:
            _l, _r = candidate.split("=", 1)
            alias_key = _r.split(",", 1)[0].strip().lower()
        dedupe_key = (mapped_wav.lower(), alias_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out_lines.append(candidate)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in out_lines:
            handle.write(row + "\n")

    errors: list[str] = []
    if missing_wavs:
        sample = ", ".join(sorted(missing_wavs)[:6])
        if len(missing_wavs) > 6:
            sample += ", ..."
        errors.append(f"missing wav file: {len(missing_wavs)} entries (e.g. {sample})")

    _log(
        callback,
        (
            f"[KR-CMPX] source={source} written={len(out_lines)} total={total} "
            f"(exact={exact_hits}, norm={norm_hits}, missing={len(missing_wavs)}, "
            f"order_fixed={adjusted_param_order})"
        ),
    )
    return len(out_lines), total, errors


__all__ = [
    "generate_kr_cmpx_preview_oto",
    "resolve_kr_cmpx_preview_source_oto",
]
