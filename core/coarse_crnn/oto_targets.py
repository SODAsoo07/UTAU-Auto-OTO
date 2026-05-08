from __future__ import annotations

import json
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.format_type_utils import normalize_format_type
from core.coarse_crnn.alias_role import classify_alias_role, is_diphthong as _is_diphthong
from core.coarse_crnn.labels import coarse_for_phone
from core.coarse_crnn.lang import phones_from_text
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_features import classify_alias_type
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key

OTO_ANCHOR_NAMES = ("offset", "overlap", "preutterance", "consonant", "cutoff")
OTO_PARAM_NAMES = ("offset", "consonant", "cutoff", "preutterance", "overlap")


@dataclass(frozen=True)
class OtoAnchors:
    offset: float
    overlap: float
    preutterance: float
    consonant: float
    cutoff: float

    def to_dict(self) -> dict[str, float]:
        return {
            "offset": float(self.offset),
            "overlap": float(self.overlap),
            "preutterance": float(self.preutterance),
            "consonant": float(self.consonant),
            "cutoff": float(self.cutoff),
        }


def wav_duration_ms(path: str) -> float:
    if not path or not os.path.isfile(path):
        return 0.0
    try:
        with wave.open(str(path), "rb") as wf:
            frames = int(wf.getnframes() or 0)
            sr = int(wf.getframerate() or 0)
        return float(frames) * 1000.0 / float(sr) if frames > 0 and sr > 0 else 0.0
    except Exception:
        try:
            from scipy.io import wavfile  # type: ignore

            sr, data = wavfile.read(str(path))
            return float(data.shape[0]) * 1000.0 / float(sr) if sr > 0 else 0.0
        except Exception:
            return 0.0


def oto_params_to_anchors(
    *,
    offset: float,
    consonant: float,
    cutoff: float,
    preutterance: float,
    overlap: float,
    duration_ms: float,
) -> OtoAnchors:
    offset_ms = max(0.0, float(offset))
    overlap_ms = offset_ms + max(0.0, float(overlap))
    pre_ms = offset_ms + max(0.0, float(preutterance))
    consonant_ms = offset_ms + max(0.0, float(consonant))
    cutoff_ms = resolve_cutoff_abs_ms(offset_ms=offset_ms, cutoff=float(cutoff), duration_ms=float(duration_ms))
    return repair_anchors(
        OtoAnchors(
            offset=offset_ms,
            overlap=overlap_ms,
            preutterance=pre_ms,
            consonant=consonant_ms,
            cutoff=cutoff_ms,
        ),
        duration_ms=duration_ms,
    )


def resolve_cutoff_abs_ms(*, offset_ms: float, cutoff: float, duration_ms: float) -> float:
    offset = max(0.0, float(offset_ms))
    duration = max(0.0, float(duration_ms))
    raw = float(cutoff)
    if raw < 0.0:
        value = offset + abs(raw)
    else:
        # UTAU positive cutoff is ambiguous in real-world oto.ini files. Treat
        # it as offset-relative first, then clamp to the waveform.
        value = offset + raw
    if duration > 0.0:
        value = min(value, max(offset + 1.0, duration - 1.0))
    return max(offset + 1.0, value)


def anchors_to_oto_params(anchors: OtoAnchors | dict[str, float], *, duration_ms: float) -> dict[str, float]:
    fixed = repair_anchors(_coerce_anchors(anchors), duration_ms=duration_ms)
    offset = float(fixed.offset)
    return {
        "offset": offset,
        "consonant": max(1.0, float(fixed.consonant) - offset),
        "cutoff": -max(1.0, float(fixed.cutoff) - offset),
        "preutterance": max(0.0, float(fixed.preutterance) - offset),
        "overlap": max(0.0, float(fixed.overlap) - offset),
    }


def repair_anchors(anchors: OtoAnchors, *, duration_ms: float) -> OtoAnchors:
    duration = max(1.0, float(duration_ms))
    offset = _clamp(float(anchors.offset), 0.0, duration - 1.0)
    overlap = _clamp(float(anchors.overlap), offset, duration - 1.0)
    pre = _clamp(float(anchors.preutterance), overlap, duration - 1.0)
    consonant = _clamp(float(anchors.consonant), pre + 1.0, duration - 1.0)
    cutoff = _clamp(float(anchors.cutoff), consonant + 1.0, duration)
    return OtoAnchors(offset=offset, overlap=overlap, preutterance=pre, consonant=consonant, cutoff=cutoff)


def oto_row_quality(row: dict[str, Any], *, duration_ms: float) -> tuple[float, str]:
    try:
        anchors = oto_params_to_anchors(
            offset=float(row["offset"]),
            consonant=float(row["cons"]),
            cutoff=float(row["cutoff"]),
            preutterance=float(row["pre"]),
            overlap=float(row["ovl"]),
            duration_ms=duration_ms,
        )
    except Exception:
        return 0.0, "bad_numeric"
    if duration_ms <= 60.0:
        return 0.0, "bad_duration"
    if anchors.cutoff <= anchors.offset + 4.0:
        return 0.0, "bad_cutoff"
    score = 1.0
    if anchors.offset > duration_ms * 0.82:
        score *= 0.45
    if anchors.cutoff < duration_ms * 0.12:
        score *= 0.45
    if float(row.get("pre", 0.0) or 0.0) <= 1.0:
        score *= 0.65
    if float(row.get("cons", 0.0) or 0.0) <= 1.0:
        score *= 0.65
    if _looks_like_placeholder(row):
        score *= 0.50
    if score < 0.25:
        return score, "low_quality"
    return score, "ok"


# v2 1-A: silence/breath alias 패턴 ([[CRNN-정확도-개선-방안-v2]] 1-A 참고).
# SONG_CMYK_CVVC 분석에서 23.4%의 row가 alias=""인 blank로 확인됐고,
# 이 row들은 학습 target으로 그대로 쓰이면 cvvc|cv|other slice를 통계적으로
# 망친다 (preutterance MAE +60 ms, hard_failure_rate +1%p).
# 분류 결과는 manifest의 ``blank_alias_kind`` 컬럼에 기록되며, 학습 단계에서
# 학습 제외 (Plan B 기본) 또는 special/breath role 재분류 (ablation)로
# 사용한다. 분류 자체는 row를 drop하지 않는다 — 평가 시점에서도 동일한
# manifest로 by_blank_alias slice 집계가 가능해야 하기 때문.
_BLANK_ALIAS_SILENCE_TOKENS: frozenset[str] = frozenset(
    [
        "r",         # UTAU 표준 표기
        "br",        # breath
        "breath",
        "pau",
        "sil",
        "silence",
        "rest",
        "ng",        # 일부 보이스뱅크에서 nasal silence 표기
        "息",         # 일본어 breath
    ]
)


def _classify_blank_alias(alias: object) -> str:
    """Return one of ``""``, ``"blank"``, ``"silence"``.

    - ``"blank"``: alias 문자열이 비었거나 공백만 있음 (SONG_CMYK_CVVC 케이스).
    - ``"silence"``: alias가 silence/breath 토큰만 포함 (R, pau, sil, br 등).
      hyphen-prefixed (``- R``, ``-R``)와 single-token 모두 매칭.
    - ``""``: 정상 alias.

    Single-source-of-truth: 학습/평가 양쪽이 같은 함수를 호출해야 한다.
    """
    text = str(alias or "").strip()
    if not text:
        return "blank"
    # Strip hyphen prefix used by some recipe naming conventions.
    cleaned = text
    while cleaned.startswith("-"):
        cleaned = cleaned[1:].lstrip()
    if not cleaned:
        return "blank"
    tokens = [tok for tok in cleaned.lower().replace("-", " ").split() if tok]
    if not tokens:
        return "blank"
    silence_set = _BLANK_ALIAS_SILENCE_TOKENS
    if all(tok in silence_set for tok in tokens):
        return "silence"
    return ""


def iter_oto_manifest_rows(dataset_root: str) -> list[dict[str, Any]]:
    base = Path(dataset_root)
    if not base.exists():
        return []
    rows: list[dict[str, Any]] = []
    for oto_path in sorted(base.rglob("oto.ini")):
        language = infer_language_from_path(str(oto_path))
        if language not in {"korean", "japanese"}:
            continue
        format_type = infer_format_type_from_path(language, str(oto_path))
        voicebank_id = infer_voicebank_id(str(oto_path), str(base))
        text = read_text_with_fallback(str(oto_path))
        wav_cache: dict[str, str] = {}
        for line_index, raw in enumerate(text.splitlines()):
            parsed = parse_oto_line(raw)
            if not parsed:
                continue
            wav_name = str(parsed.get("wav", "") or "")
            wav_path = wav_cache.get(wav_name)
            if wav_path is None:
                wav_path = resolve_oto_wav_path(str(oto_path), wav_name)
                wav_cache[wav_name] = wav_path
            if not wav_path:
                continue
            duration_ms = wav_duration_ms(wav_path)
            quality, reason = oto_row_quality(parsed, duration_ms=duration_ms)
            if quality <= 0.0:
                continue
            anchors = oto_params_to_anchors(
                offset=float(parsed["offset"]),
                consonant=float(parsed["cons"]),
                cutoff=float(parsed["cutoff"]),
                preutterance=float(parsed["pre"]),
                overlap=float(parsed["ovl"]),
                duration_ms=duration_ms,
            )
            alias = str(parsed.get("alias", "") or "").strip()
            alias_features = extract_alias_features(alias, language=language)
            blank_alias_kind = _classify_blank_alias(alias)
            rows.append(
                {
                    "audio": os.path.abspath(wav_path),
                    "wav": wav_name,
                    "alias": alias,
                    "alias_norm": canonicalize_alias_for_matching(language, alias),
                    **alias_features,
                    "wav_norm": normalize_wav_key(wav_name),
                    "language": language,
                    "format_type": format_type or "other",
                    "voicebank_id": voicebank_id,
                    "oto_path": os.path.abspath(str(oto_path)),
                    "line_index": int(line_index),
                    "duration_ms": round(float(duration_ms), 4),
                    "target_offset_ms": float(parsed["offset"]),
                    "target_consonant_ms": float(parsed["cons"]),
                    "target_cutoff": float(parsed["cutoff"]),
                    "target_preutterance_ms": float(parsed["pre"]),
                    "target_overlap_ms": float(parsed["ovl"]),
                    "target_cutoff_abs_ms": float(anchors.cutoff),
                    "anchor_offset_ms": float(anchors.offset),
                    "anchor_overlap_ms": float(anchors.overlap),
                    "anchor_preutterance_ms": float(anchors.preutterance),
                    "anchor_consonant_ms": float(anchors.consonant),
                    "anchor_cutoff_ms": float(anchors.cutoff),
                    "sample_weight": round(float(quality), 4),
                    "quality_reason": reason,
                    "source": "oto_ini",
                    # v2 1-A blank alias 분리: 학습 시 제외 또는 별도 처리.
                    # ""=정상 alias, "blank"=빈 문자열, "silence"=R/pau/sil 패턴.
                    "blank_alias_kind": blank_alias_kind,
                }
            )
    _apply_row_context(rows)
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


# v2 1-B: row-order violation score ([[CRNN-정확도-개선-방안-v2]] 1-B 참고).
# SONG_CMYK_CVVC에서 row의 32.8%가 row_index 순서와 실제 target_offset
# 순서가 어긋나, ``abs(offset - row_ratio*duration)``의 평균이 2202 ms에
# 달한다 (Sato CVVC는 958 ms). 모델은 ``row_ratio_in_wav`` feature와 row
# position embedding에 의존하기 때문에 이 미스매치 row는 학습 signal로
# 그대로 들어가면 anchor regression을 흔든다. 분류 자체는 manifest에 컬럼
# (``row_order_violation_score``)으로 기록만 하고, 학습 단계에서
# ``OtoTrainConfig.row_order_violation_alpha``로 sample_weight 다운스케일.
# 평가에서는 같은 컬럼을 slice 키로 써서 violation 큰 row 별도 집계 가능.
_ROW_ORDER_VIOLATION_EPS_MS: float = 200.0


def _compute_row_order_violation_score(
    *,
    target_offset_ms: float,
    row_ratio: float,
    duration_ms: float,
    eps_ms: float = _ROW_ORDER_VIOLATION_EPS_MS,
) -> float:
    """Row-order violation score: ``|offset - row_ratio * duration| / max(duration, eps_ms)``.

    Returns 0.0 for non-positive duration (defensive, prevents division by
    zero). The ``eps_ms`` floor keeps very short / corrupted rows from
    blowing the score up. Score is clipped to ``[0, 10]`` upstream — the
    raw return is unbounded but typical values are 0.0–3.0.

    Single source of truth: callers in :func:`_apply_row_context` and the
    training-time weight reweighter (``oto_training._row_order_violation_multiplier``)
    use this exact function so the manifest column and the runtime weight
    cannot drift apart.
    """
    duration = max(0.0, float(duration_ms or 0.0))
    if duration <= 0.0:
        return 0.0
    expected = float(row_ratio or 0.0) * duration
    delta = abs(float(target_offset_ms or 0.0) - expected)
    return float(delta) / max(duration, float(eps_ms))


def _apply_row_context(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("oto_path", "") or ""), str(row.get("wav", "") or ""))
        groups.setdefault(key, []).append(row)
    for items in groups.values():
        items.sort(key=lambda item: int(item.get("line_index", 0) or 0))
        count = len(items)
        for idx, row in enumerate(items):
            row["row_index_in_wav"] = idx
            row["file_row_count"] = count
            row_ratio = float(idx) / float(max(1, count - 1)) if count > 1 else 0.0
            row["row_ratio_in_wav"] = row_ratio
            row["is_head_row"] = 1.0 if idx == 0 else 0.0
            row["is_tail_row"] = 1.0 if idx == count - 1 else 0.0
            row["row_order_violation_score"] = round(
                _compute_row_order_violation_score(
                    target_offset_ms=float(row.get("target_offset_ms", 0.0) or 0.0),
                    row_ratio=row_ratio,
                    duration_ms=float(row.get("duration_ms", 0.0) or 0.0),
                ),
                4,
            )
            _apply_neighbor_alias(row, "prev", items[idx - 1] if idx > 0 else None)
            _apply_neighbor_alias(row, "next", items[idx + 1] if idx + 1 < count else None)


def _apply_neighbor_alias(row: dict[str, Any], prefix: str, neighbor: dict[str, Any] | None) -> None:
    if neighbor is None:
        row[f"{prefix}_alias"] = ""
        row[f"{prefix}_alias_type"] = "other"
        row[f"{prefix}_transition_type"] = "other"
        row[f"{prefix}_alias_role"] = "other"
        row[f"{prefix}_alias_starts_vowel"] = 0.0
        row[f"{prefix}_alias_ends_vowel"] = 0.0
        return
    row[f"{prefix}_alias"] = str(neighbor.get("alias", "") or "")
    row[f"{prefix}_alias_type"] = str(neighbor.get("alias_type", "") or "other")
    row[f"{prefix}_transition_type"] = str(neighbor.get("transition_type", "") or "other")
    row[f"{prefix}_alias_role"] = str(neighbor.get("alias_role", "") or "other")
    row[f"{prefix}_alias_starts_vowel"] = float(neighbor.get("alias_starts_vowel", 0.0) or 0.0)
    row[f"{prefix}_alias_ends_vowel"] = float(neighbor.get("alias_ends_vowel", 0.0) or 0.0)


def extract_alias_features(
    alias: str,
    *,
    language: str,
    is_special: bool = False,
) -> dict[str, object]:
    text = str(alias or "").strip()
    try:
        alias_type = str(classify_alias_type(language, text) or "other").strip().lower()
    except Exception:
        alias_type = "other"
    phones = _alias_phones(text, language=language)
    coarse = [coarse_for_phone(phone, language=language) for phone in phones]
    starts_vowel = bool(coarse and coarse[0] == "V")
    ends_vowel = bool(coarse and coarse[-1] == "V")
    transition_type = _transition_type(coarse, alias_type=alias_type)
    diphthong = bool(_is_diphthong(language, text))
    role = classify_alias_role(
        language,
        text,
        alias_type=alias_type,
        transition_type=transition_type,
        is_special=bool(is_special),
    )
    return {
        "alias_type": alias_type or "other",
        "transition_type": transition_type,
        "alias_role": role,
        "is_diphthong": 1.0 if diphthong else 0.0,
        "is_special": 1.0 if bool(is_special) else 0.0,
        "alias_phone_count": len(phones),
        "alias_starts_vowel": 1.0 if starts_vowel else 0.0,
        "alias_ends_vowel": 1.0 if ends_vowel else 0.0,
        "alias_has_space": 1.0 if bool(re.search(r"\s+", text)) else 0.0,
        "alias_is_vc": 1.0 if transition_type == "vc" else 0.0,
        "alias_is_cv": 1.0 if transition_type == "cv" else 0.0,
        "alias_is_vv": 1.0 if transition_type == "vv" else 0.0,
    }


def _alias_phones(alias: str, *, language: str) -> list[str]:
    text = str(alias or "").strip()
    if not text:
        return []
    if re.search(r"\s+", text):
        return [token for token in re.split(r"\s+", text) if token]
    phones = phones_from_text(text, language)
    if phones:
        return phones
    return [text]


def _transition_type(coarse: list[str], *, alias_type: str) -> str:
    alias_type_text = str(alias_type or "").strip().lower()
    if alias_type_text in {"br", "sil", "pause"}:
        return "br"
    if len(coarse) <= 0:
        return "other"
    if len(coarse) == 1:
        return "vowel" if coarse[0] == "V" else "consonant" if coarse[0].startswith("C_") else "other"
    first = "v" if coarse[0] == "V" else "c" if coarse[0].startswith("C_") else "x"
    last = "v" if coarse[-1] == "V" else "c" if coarse[-1].startswith("C_") else "x"
    value = first + last
    if value in {"cv", "vc", "vv", "cc"}:
        return value
    return "multi"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = str(raw or "").strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def resolve_oto_wav_path(oto_path: str, wav_name: str) -> str:
    if not wav_name:
        return ""
    oto_dir = os.path.dirname(os.path.abspath(oto_path))
    direct = os.path.join(oto_dir, wav_name)
    if os.path.isfile(direct):
        return os.path.abspath(direct)
    target = normalize_wav_key(wav_name)
    if not target:
        return ""
    for path in Path(oto_dir).rglob("*.wav"):
        if normalize_wav_key(path.name) == target:
            return str(path.resolve())
    return ""


def infer_language_from_path(path: str) -> str:
    text = str(path or "").replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    if "korean" in parts or "ko" in parts or "한국" in text:
        return "korean"
    if "japanese" in parts or "ja" in parts or "jp" in parts or "utagoe" in text:
        return "japanese"
    return ""


def infer_format_type_from_path(language: str, path: str) -> str:
    parts = [part for part in str(path or "").replace("\\", "/").split("/") if part]
    if parts and "." in parts[-1]:
        parts = parts[:-1]
    normalized_parts = [(part, _format_token(part)) for part in parts]
    for part, token in normalized_parts:
        if token in {"cvvc", "vcv", "cvc", "cv", "c_plus_v"}:
            return normalize_format_type(language, token)
    for part, token in reversed(normalized_parts):
        if token in {"korean", "japanese", "dataset_staged"}:
            continue
        if "cvvc" in token:
            return normalize_format_type(language, "cvvc")
        if "vcv" in token:
            return normalize_format_type(language, "vcv")
        if "cvc" in token or "coc" in token:
            return normalize_format_type(language, "cvc")
        if token.endswith("jpcv") or token.endswith("krcv") or token.endswith("_cv"):
            return normalize_format_type(language, "cv")
    return "other"


def _format_token(value: object) -> str:
    text = str(value or "").strip().lower().replace("\\", "/")
    text = text.replace("-", "_").replace(" ", "")
    return text


def infer_voicebank_id(oto_path: str, root: str) -> str:
    oto_dir = os.path.dirname(os.path.abspath(oto_path))
    root_abs = os.path.abspath(root)
    try:
        rel = os.path.relpath(oto_dir, root_abs)
    except Exception:
        rel = os.path.basename(oto_dir)
    parts = [part for part in rel.split(os.sep) if part]
    candidates: list[str] = []
    for part in parts:
        token = _format_token(part)
        if token in {"korean", "japanese", "cv", "cvc", "cvvc", "vcv", "c_plus_v", "dataset_staged"}:
            continue
        if _looks_like_pitch_folder(token):
            continue
        candidates.append(part)
    return candidates[0] if candidates else os.path.basename(oto_dir) or "unknown_voicebank"


def _looks_like_pitch_folder(token: str) -> bool:
    if re.fullmatch(r"[a-g](?:#|b)?[0-8]", token):
        return True
    if re.fullmatch(r"(soft|power|weak|solid|standard|noise)?_?[a-g](?:#|b)?[0-8]", token):
        return True
    return token in {"main", "default", "normal", "soft", "power", "weak", "solid", "standard", "noise"}


def _coerce_anchors(anchors: OtoAnchors | dict[str, float]) -> OtoAnchors:
    if isinstance(anchors, OtoAnchors):
        return anchors
    return OtoAnchors(
        offset=float(anchors.get("offset", 0.0)),
        overlap=float(anchors.get("overlap", 0.0)),
        preutterance=float(anchors.get("preutterance", 0.0)),
        consonant=float(anchors.get("consonant", 0.0)),
        cutoff=float(anchors.get("cutoff", 0.0)),
    )


def _looks_like_placeholder(row: dict[str, Any]) -> bool:
    vals = (
        round(float(row.get("offset", 0.0) or 0.0), 2),
        round(float(row.get("cons", 0.0) or 0.0), 2),
        round(float(row.get("cutoff", 0.0) or 0.0), 2),
        round(float(row.get("pre", 0.0) or 0.0), 2),
        round(float(row.get("ovl", 0.0) or 0.0), 2),
    )
    common = {
        (1000.0, 450.0, -900.0, 100.0, 25.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    }
    return vals in common


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "OTO_ANCHOR_NAMES",
    "OTO_PARAM_NAMES",
    "OtoAnchors",
    "anchors_to_oto_params",
    "infer_format_type_from_path",
    "infer_language_from_path",
    "iter_oto_manifest_rows",
    "extract_alias_features",
    "oto_params_to_anchors",
    "read_jsonl",
    "repair_anchors",
    "resolve_cutoff_abs_ms",
    "resolve_oto_wav_path",
    "wav_duration_ms",
    "write_jsonl",
]
