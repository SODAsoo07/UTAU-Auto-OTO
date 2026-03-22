from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Callable, Optional

from core.lab_generator import load_custom_phonemes


@dataclass
class JaTextGridPreparation:
    tg_folder: str
    normalize_key_fn: Callable[[str], str]
    tg_entries: list[dict]
    tg_exact_map: dict[str, dict]
    tg_norm_map: dict[str, list[dict]]

    def resolve_tg_info(self, fname: str, log_fn: Optional[Callable[[str], None]] = None):
        wav_name = os.path.basename((fname or "").strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in self.tg_exact_map:
            return self.tg_exact_map[base_lower]

        norm_name = self.normalize_key_fn(wav_name)
        if not norm_name:
            return None
        candidates = self.tg_norm_map.get(norm_name, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            exact_name = wav_name.lower()
            same_name = [c for c in candidates if c["real_name"].lower() == exact_name]
            if len(same_name) == 1:
                return same_name[0]
            if callable(log_fn):
                log_fn(
                    f"⚠️ 파일명 매핑 충돌: {wav_name} "
                    f"(정규화 키 {norm_name}, 후보 {len(candidates)}개) → 원본 파일명 유지"
                )
            return None
        return None

    def textgrid_missing_diagnostics(self, fname: str) -> dict[str, object]:
        wav_name = os.path.basename((fname or "").strip())
        base = os.path.splitext(wav_name)[0]
        norm = self.normalize_key_fn(wav_name)
        base_candidates = {base}
        if base.startswith("_"):
            base_candidates.add(base[1:])
        else:
            base_candidates.add("_" + base)
        candidate_paths = [os.path.join(self.tg_folder, c + ".TextGrid") for c in sorted(base_candidates)]
        norm_candidates = [c.get("path", "") for c in (self.tg_norm_map.get(norm, []) or [])][:5]
        return {
            "wav_basename": wav_name,
            "lookup_base": base,
            "lookup_norm": norm,
            "candidate_paths": candidate_paths,
            "norm_candidates": norm_candidates,
            "diag_hint": f"norm={norm}, candidates={len(norm_candidates)}",
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
    template_encoding: str
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
    if os.path.exists(tg_folder):
        for f_name in os.listdir(tg_folder):
            if not f_name.lower().endswith(".textgrid"):
                continue
            base = os.path.splitext(f_name)[0]
            info = {
                "path": os.path.join(tg_folder, f_name),
                "real_name": base + ".wav",
                "base_lower": base.lower(),
                "norm_key": normalize_key_fn(f_name),
            }
            tg_entries.append(info)
            if info["base_lower"] not in tg_exact_map:
                tg_exact_map[info["base_lower"]] = info
            if info["norm_key"]:
                tg_norm_map.setdefault(info["norm_key"], []).append(info)
    return JaTextGridPreparation(
        tg_folder=tg_folder,
        normalize_key_fn=normalize_key_fn,
        tg_entries=tg_entries,
        tg_exact_map=tg_exact_map,
        tg_norm_map=tg_norm_map,
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
) -> JaGenerationSetupResult:
    if tpl_path and not os.path.exists(tpl_path):
        log_fn(f"⚠️ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log_fn(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""

    template_lines: list[str] = []
    template_encoding = ""
    if tpl_path:
        lines, detected_enc, warning, err = load_template_lines_fn(tpl_path)
        if err:
            log_fn(f"{err}")
            log_fn(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
            detected_enc = ""
        if warning:
            log_fn(warning)
        template_lines = list(lines or [])
        template_encoding = str(detected_enc or "").strip()

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
                f"⚠️ 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"→ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환"
            )
            use_template = False
        else:
            log_fn(f"📌 템플릿 베이스 OTO 사용 ({t_match}/{t_total}, {t_ratio:.1%})")

    if use_template:
        for line in template_lines:
            fname = line.split("=", 1)[0]
            file_groups.setdefault(fname, []).append(line)

    return JaGenerationSetupResult(
        custom_map=custom_map,
        template_lines=template_lines,
        template_encoding=template_encoding,
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
