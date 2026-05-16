from __future__ import annotations

import os
from typing import Callable


def build_ja_auto_file_groups(
    *,
    tg_entries,
    auto_gen_format: str,
    log_fn: Callable[[str], None],
    load_textgrid_fn: Callable[[str], object],
    parse_filename_fn: Callable[[str], list[str]],
    split_syllable_fn: Callable[[str], tuple[str, str]],
    is_vowel_chain_fn: Callable[[list[str]], bool],
) -> dict[str, list[str]]:
    file_groups: dict[str, list[str]] = {}

    def _resolve_vowel_onset(syllable: str) -> tuple[str, str]:
        onset, vowel = split_syllable_fn(syllable)
        if vowel in {"a", "i", "u", "e", "o", "N"}:
            return vowel, onset
        if syllable in {"n", "nn", "xn", "m"}:
            return "n", ""
        return "a", onset

    for tg_info in tg_entries:
        tg_path = tg_info["path"]
        real_name = tg_info["real_name"]

        try:
            # 일본어 자동 alias는 파일명 기반이므로 TextGrid 전체 파싱은 생략해 I/O를 줄인다.
            if not os.path.isfile(tg_path):
                continue
            base_name = os.path.splitext(real_name)[0]
            syllables = parse_filename_fn(base_name)
            if not syllables:
                continue

            lines: list[str] = []
            is_long = base_name.lower().endswith("long") or len(syllables) == 1
            local_format = auto_gen_format
            if is_vowel_chain_fn(syllables):
                local_format = "vcv"
                log_fn(f"🧭 {real_name}: 모음 연속음 파일 감지 → 자동 생성 포맷 VCV 적용")

            if is_long:
                for syllable in syllables:
                    lines.append(f"{real_name}={syllable},0,0,0,0,0")
            else:
                for idx, syllable in enumerate(syllables):
                    vowel, onset = _resolve_vowel_onset(syllable)
                    if local_format == "vcv":
                        if idx == 0:
                            lines.append(f"{real_name}=- {syllable},0,0,0,0,0")
                        else:
                            prev_vowel, _ = _resolve_vowel_onset(syllables[idx - 1])
                            lines.append(f"{real_name}={prev_vowel} {syllable},0,0,0,0,0")
                    elif local_format in {"cv", "cvc", "cvvc"}:
                        lines.append(f"{real_name}={syllable},0,0,0,0,0")
                        if idx > 0 and local_format == "cvvc":
                            prev_vowel, _ = _resolve_vowel_onset(syllables[idx - 1])
                            if onset:
                                lines.append(f"{real_name}={prev_vowel} {onset},0,0,0,0,0")
                            else:
                                lines.append(f"{real_name}={prev_vowel} {vowel},0,0,0,0,0")

            if lines:
                file_groups[real_name] = lines
        except Exception as exc:
            log_fn(f"⚠️ {real_name}: 자동 템플릿 생성 실패 ({exc})")

    return file_groups


__all__ = ["build_ja_auto_file_groups"]
