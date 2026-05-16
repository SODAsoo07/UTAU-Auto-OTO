from __future__ import annotations

import datetime
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

from core.generation.file_index import iter_textgrid_files
from core.lab_generator import load_custom_phonemes
from core.oto_normalization import normalize_wav_key

_KANJI_RE = re.compile(r"[\u3400-\u9fff]")
_SUFFIX_ROMAN_RE = re.compile(r"[A-Za-z]")
# 숨소리 표기로 간주해 접미사로 벗기지 않을 한자 토큰
_BREATH_KANJI_TOKENS = {"息", "吸", "吐", "呼", "気", "氣"}


def _looks_like_suffix_token(token: str) -> bool:
    text = unicodedata.normalize("NFKC", str(token or "")).strip()
    if not text:
        return False
    if _KANJI_RE.search(text):
        kanji_chars = [ch for ch in text if _KANJI_RE.match(ch)]
        if kanji_chars and all(ch in _BREATH_KANJI_TOKENS for ch in kanji_chars):
            return False
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if _SUFFIX_ROMAN_RE.search(text):
        return True
    return False


def _split_suffix_token(base_name: str) -> tuple[str, str]:
    raw = unicodedata.normalize("NFKC", str(base_name or "")).strip()
    if "_" not in raw:
        return raw, ""
    core, suffix = raw.rsplit("_", 1)
    core = core.strip()
    suffix = suffix.strip()
    if not core or not suffix:
        return raw, ""
    if not _looks_like_suffix_token(suffix):
        return raw, ""
    return core, suffix


def _extract_suffix_token(name: str) -> str:
    base = os.path.splitext(os.path.basename(str(name or "")))[0]
    _, suffix = _split_suffix_token(base)
    return suffix.lower()


def _normalize_core_key(name: str) -> str:
    base = os.path.splitext(os.path.basename(str(name or "")))[0]
    core, _ = _split_suffix_token(base)
    return normalize_wav_key(core)


@dataclass
class JaTextGridPreparation:
    tg_folder: str
    normalize_key_fn: Callable[[str], str]
    tg_entries: list[dict]
    tg_exact_map: dict[str, dict]
    tg_norm_map: dict[str, list[dict]]
    tg_core_norm_map: dict[str, list[dict]]

    def _pick_with_suffix(self, candidates: list[dict], *, wav_name: str) -> Optional[dict]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        target_suffix = _extract_suffix_token(wav_name)
        if target_suffix:
            matched = [
                c
                for c in candidates
                if str(c.get("suffix_token", "") or "").strip().lower() == target_suffix
            ]
            if len(matched) == 1:
                return matched[0]
            if matched:
                return matched[0]
        return None

    def resolve_tg_info(self, fname: str, log_fn: Optional[Callable[[str], None]] = None):
        wav_name = os.path.basename((fname or "").strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in self.tg_exact_map:
            return self.tg_exact_map[base_lower]

        norm_name = self.normalize_key_fn(wav_name)
        if norm_name:
            candidates = self.tg_norm_map.get(norm_name, [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                exact_name = wav_name.lower()
                same_name = [c for c in candidates if c["real_name"].lower() == exact_name]
                if len(same_name) == 1:
                    return same_name[0]
                suffix_picked = self._pick_with_suffix(candidates, wav_name=wav_name)
                if suffix_picked is not None:
                    return suffix_picked
                if callable(log_fn):
                    log_fn(
                        f"⚠ TextGrid ambiguous by norm key: {wav_name} "
                        f"(norm={norm_name}, candidates={len(candidates)})"
                    )
                return None

        core_norm = _normalize_core_key(wav_name)
        if not core_norm:
            return None
        core_candidates = self.tg_core_norm_map.get(core_norm, [])
        if len(core_candidates) == 1:
            return core_candidates[0]
        if len(core_candidates) > 1:
            suffix_picked = self._pick_with_suffix(core_candidates, wav_name=wav_name)
            if suffix_picked is not None:
                return suffix_picked
            if callable(log_fn):
                log_fn(
                    f"⚠ TextGrid ambiguous by core norm key: {wav_name} "
                    f"(core_norm={core_norm}, candidates={len(core_candidates)})"
                )
            return None
        return None

    def textgrid_missing_diagnostics(self, fname: str) -> dict[str, object]:
        wav_name = os.path.basename((fname or "").strip())
        base = os.path.splitext(wav_name)[0]
        norm = self.normalize_key_fn(wav_name)
        core_norm = _normalize_core_key(wav_name)
        base_candidates = {base}
        if base.startswith("_"):
            base_candidates.add(base[1:])
        else:
            base_candidates.add("_" + base)
        candidate_paths = [os.path.join(self.tg_folder, c + ".TextGrid") for c in sorted(base_candidates)]
        norm_candidates = [c.get("path", "") for c in (self.tg_norm_map.get(norm, []) or [])][:5]
        core_norm_candidates = [c.get("path", "") for c in (self.tg_core_norm_map.get(core_norm, []) or [])][:5]
        return {
            "wav_basename": wav_name,
            "lookup_base": base,
            "lookup_norm": norm,
            "lookup_core_norm": core_norm,
            "candidate_paths": candidate_paths,
            "norm_candidates": norm_candidates,
            "core_norm_candidates": core_norm_candidates,
            "diag_hint": (
                f"norm={norm}, candidates={len(norm_candidates)}, "
                f"core_norm={core_norm}, core_candidates={len(core_norm_candidates)}"
            ),
        }

    def template_match_stats(self, lines, log_fn: Optional[Callable[[str], None]] = None) -> tuple[int, int, float]:
        file_names = set()
        for line in (lines or []):
            if "=" not in line:
                continue
            file_names.add(line.split("=", 1)[0].strip())
        total = len(file_names)
        if total == 0:
            return 0, 0, 0.0
        matched = 0
        for fname in file_names:
            if self.resolve_tg_info(fname, log_fn=log_fn):
                matched += 1
        return matched, total, (matched / float(total))


@dataclass(frozen=True)
class JaTracePreparation:
    anchor_log_dir: str
    anchor_log_path: str
    mapping_trace_path: str


@dataclass
class JaGenerationSetupResult:
    custom_map: object
    template_lines: list[str]
    use_template: bool
    file_groups: dict[str, list[str]]
    tg_index: JaTextGridPreparation


def build_ja_trace_preparation(project_dir: str, now: Optional[datetime.datetime] = None) -> JaTracePreparation:
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    anchor_log_dir = os.path.join(project_dir, "logs")
    return JaTracePreparation(
        anchor_log_dir=anchor_log_dir,
        anchor_log_path=os.path.join(anchor_log_dir, f"timing_anchor_ja_{stamp}.jsonl"),
        mapping_trace_path=os.path.join(anchor_log_dir, f"ja_mapping_trace_{stamp}.jsonl"),
    )


def build_ja_textgrid_preparation(
    tg_folder: str,
    *,
    normalize_key_fn: Callable[[str], str],
) -> JaTextGridPreparation:
    tg_entries = []
    tg_exact_map: dict[str, dict] = {}
    tg_norm_map: dict[str, list[dict]] = {}
    tg_core_norm_map: dict[str, list[dict]] = {}
    if os.path.exists(tg_folder):
        for dirpath, f_name in iter_textgrid_files(tg_folder, recursive=False):
            base = os.path.splitext(f_name)[0]
            info = {
                "path": os.path.join(dirpath, f_name),
                "real_name": base + ".wav",
                "base_lower": base.lower(),
                "norm_key": normalize_key_fn(f_name),
                "core_norm_key": _normalize_core_key(f_name),
                "suffix_token": _extract_suffix_token(f_name),
            }
            tg_entries.append(info)
            if info["base_lower"] not in tg_exact_map:
                tg_exact_map[info["base_lower"]] = info
            if info["norm_key"]:
                tg_norm_map.setdefault(info["norm_key"], []).append(info)
            if info["core_norm_key"]:
                tg_core_norm_map.setdefault(info["core_norm_key"], []).append(info)
    return JaTextGridPreparation(
        tg_folder=tg_folder,
        normalize_key_fn=normalize_key_fn,
        tg_entries=tg_entries,
        tg_exact_map=tg_exact_map,
        tg_norm_map=tg_norm_map,
        tg_core_norm_map=tg_core_norm_map,
    )


def prepare_ja_generation_setup(
    *,
    tg_folder: str,
    tpl_path: str,
    auto_gen_format: str,
    custom_phonemes_path: str,
    log_fn: Callable[[str], None],
    normalize_key_fn: Callable[[str], str],
    load_template_lines_fn: Callable[[str], tuple[list[str], str, str, str]],
    existing_wav_names: Optional[set[str]] = None,
    existing_wav_only: bool = True,
) -> JaGenerationSetupResult:
    if tpl_path and not os.path.exists(tpl_path):
        log_fn(f"⚠ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log_fn(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""

    template_lines: list[str] = []
    if tpl_path:
        lines, _detected_enc, warning, err = load_template_lines_fn(tpl_path)
        if err:
            log_fn(f"{err}")
            log_fn(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
        if warning:
            log_fn(warning)
        template_lines = list(lines or [])
        if existing_wav_only and existing_wav_names:
            existing_lower = {
                os.path.basename(str(name or "")).strip().lower()
                for name in existing_wav_names
                if str(name or "").strip()
            }
            filtered_lines: list[str] = []
            dropped_rows = 0
            dropped_files: set[str] = set()
            for row in template_lines:
                if "=" not in row:
                    continue
                wav_name = row.split("=", 1)[0].strip()
                wav_base = os.path.basename(wav_name).strip().lower()
                if wav_base in existing_lower:
                    filtered_lines.append(row)
                else:
                    dropped_rows += 1
                    if wav_base:
                        dropped_files.add(wav_base)
            if dropped_rows > 0:
                log_fn(
                    f"ℹ 템플릿에서 존재하지 않는 WAV 행 제외: "
                    f"rows={dropped_rows}, files={len(dropped_files)}"
                )
            template_lines = filtered_lines

    tg_index = build_ja_textgrid_preparation(
        tg_folder,
        normalize_key_fn=normalize_key_fn,
    )
    custom_map = load_custom_phonemes(custom_phonemes_path)

    use_template = bool(template_lines)
    file_groups: dict[str, list[str]] = {}
    if use_template:
        t_match, t_total, t_ratio = tg_index.template_match_stats(template_lines, log_fn=log_fn)
        if t_total == 0 or t_match == 0 or t_ratio < 0.25:
            log_fn(
                f"⚠ 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"→ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환"
            )
            use_template = False
        else:
            log_fn(f"✅ 템플릿 기반 OTO 사용 ({t_match}/{t_total}, {t_ratio:.1%})")

    if use_template:
        for line in template_lines:
            fname = line.split("=", 1)[0]
            file_groups.setdefault(fname, []).append(line)

    return JaGenerationSetupResult(
        custom_map=custom_map,
        template_lines=template_lines,
        use_template=use_template,
        file_groups=file_groups,
        tg_index=tg_index,
    )


__all__ = [
    "JaGenerationSetupResult",
    "JaTextGridPreparation",
    "JaTracePreparation",
    "build_ja_textgrid_preparation",
    "build_ja_trace_preparation",
    "prepare_ja_generation_setup",
]
