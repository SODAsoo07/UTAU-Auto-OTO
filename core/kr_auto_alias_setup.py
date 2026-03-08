from __future__ import annotations

import os
from typing import Callable


def build_kr_auto_file_groups(
    *,
    tg_entries,
    auto_gen_format: str,
    log_fn: Callable[[str], None],
    load_textgrid_fn: Callable[[str], object],
    decompose_hangul_to_roman_fn: Callable[[str], list[str]],
    split_syllable_parts_fn: Callable[[str], tuple[str, str, str]],
    kr_vowels: set[str],
    kr_consonants: set[str],
) -> dict[str, list[str]]:
    file_groups: dict[str, list[str]] = {}
    coda_alias_map = {
        "ng": "NG",
        "n": "N",
        "l": "L",
        "r": "L",
        "m": "M",
        "k": "K",
        "g": "K",
        "t": "T",
        "d": "T",
        "s": "T",
        "ss": "T",
        "j": "T",
        "jj": "T",
        "ch": "T",
        "h": "H",
        "p": "P",
        "b": "P",
    }

    def _word_to_roman(word: str) -> str:
        parts = []
        for ch in word:
            parts.extend(decompose_hangul_to_roman_fn(ch))
        return "".join(parts).lower()

    for tg_info in tg_entries:
        tg_path = tg_info["path"]
        real_name = tg_info["real_name"]
        try:
            tg = load_textgrid_fn(tg_path)
            word_tier = None
            for tier in tg:
                if hasattr(tier, "name") and tier.name == "words":
                    word_tier = tier
                    break
            if not word_tier:
                continue

            wd_intervals = [i for i in word_tier if i.mark.strip() not in {"", "sil", "spn", "pau"}]
            if not wd_intervals:
                continue

            lines: list[str] = []
            base_name = os.path.splitext(real_name)[0].lower()
            is_long = base_name.endswith("long") or len(wd_intervals) == 1

            if is_long:
                for word in wd_intervals:
                    roman = _word_to_roman(word.mark)
                    alias = roman.capitalize() if roman not in kr_vowels else roman
                    lines.append(f"{real_name}={alias},0,0,0,0,0")
            else:
                for idx, word in enumerate(wd_intervals):
                    roman = _word_to_roman(word.mark)
                    if auto_gen_format == "vcv":
                        if idx == 0:
                            lines.append(f"{real_name}=- {roman},0,0,0,0,0")
                        else:
                            prev_roman = _word_to_roman(wd_intervals[idx - 1].mark)
                            _, prev_vowel, prev_coda = split_syllable_parts_fn(prev_roman)
                            prev_coda_alias = coda_alias_map.get((prev_coda or "").lower(), "")
                            if prev_coda_alias:
                                lines.append(f"{real_name}={prev_coda_alias} {roman},0,0,0,0,0")
                            else:
                                lines.append(f"{real_name}={(prev_vowel or 'a')} {roman},0,0,0,0,0")
                    elif auto_gen_format in {"cv", "cvc", "cvvc"}:
                        lines.append(f"{real_name}={roman},0,0,0,0,0")
                        if idx > 0 and auto_gen_format == "cvvc":
                            prev_roman = _word_to_roman(wd_intervals[idx - 1].mark)
                            _, prev_vowel, _ = split_syllable_parts_fn(prev_roman)
                            prev_vowel = prev_vowel or "a"
                            cur_start_cons = ""
                            for char in roman:
                                if char in kr_consonants:
                                    cur_start_cons += char
                                else:
                                    break
                            if cur_start_cons:
                                lines.append(f"{real_name}={prev_vowel} {cur_start_cons},0,0,0,0,0")
                            else:
                                _, cur_vowel, _ = split_syllable_parts_fn(roman)
                                lines.append(f"{real_name}={prev_vowel} {(cur_vowel or 'a')},0,0,0,0,0")

            if lines:
                file_groups[real_name] = lines
        except Exception as exc:
            log_fn(f"경고: {real_name} 자동 템플릿 생성 실패 ({exc})")

    return file_groups


__all__ = ["build_kr_auto_file_groups"]
