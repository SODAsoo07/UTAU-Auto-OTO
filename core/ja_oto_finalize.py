from __future__ import annotations

import os


def sanitize_ja_oto_for_wav_duration(offset, consonant, cutoff, pre, ovl, wav_duration_ms, alias_type="cv"):
    """
    wav 길이 기준으로 최종 OTO 값을 안전한 영역 안에 다시 넣습니다.
    OpenUtau의 cutoff는 파일 끝 기준이므로, 단순히 cutoff_abs > consonant만으로는
    `cutoff before offset` 오류를 막을 수 없습니다.
    """
    from core.ja_oto_generator import validate_oto_params

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    try:
        dur_ms = float(wav_duration_ms or 0.0)
    except Exception:
        dur_ms = 0.0
    if dur_ms <= 0.0:
        return offset, consonant, cutoff, pre, ovl

    min_room_after_offset = 16.0 if alias_type in {"vc", "vv", "vcv"} else 22.0
    offset = max(0.0, min(float(offset), max(dur_ms - min_room_after_offset, 0.0)))

    cutoff_abs = abs(float(cutoff))
    max_cutoff_abs = max(dur_ms - offset - 6.0, 0.0)
    cutoff_abs = min(cutoff_abs, max_cutoff_abs)

    active_len = max(dur_ms - offset - cutoff_abs, 0.0)
    if active_len < 10.0:
        target_active_len = min(max(12.0, active_len), max(dur_ms - offset - 2.0, 0.0))
        cutoff_abs = max(0.0, dur_ms - offset - target_active_len)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)

    max_cons = max(active_len - 6.0, 0.0)
    consonant = min(float(consonant), max_cons)
    if consonant < 8.0 and active_len >= 10.0:
        consonant = min(max(active_len * 0.72, 8.0), max_cons)

    pre = min(float(pre), max(consonant - 6.0, 0.0))
    if pre < 0.0:
        pre = 0.0
    ovl = min(float(ovl), pre * 0.82 if pre > 0.0 else 0.0)
    if ovl < 0.0:
        ovl = 0.0

    if consonant <= pre + 4.0:
        if max_cons >= pre + 6.0:
            consonant = pre + 6.0
        else:
            consonant = max_cons
            pre = max(0.0, consonant - 6.0)
            ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    if dur_ms - offset - cutoff_abs <= 2.0:
        cutoff_abs = max(0.0, dur_ms - offset - 4.0)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)
        max_cons = max(active_len - 4.0, 0.0)
        consonant = min(consonant, max_cons)
        pre = min(pre, max(consonant - 4.0, 0.0))
        ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    cutoff = -max(cutoff_abs, 0.0)
    return offset, consonant, cutoff, pre, ovl


def _sanitize_ja_internal_params_for_wav_duration(offset, consonant, cutoff, pre, ovl, wav_duration_ms, alias_type="cv"):
    """
    일본어 생성기 내부 좌표계 기준으로 파라미터를 안전한 범위로 다시 넣습니다.
    여기서 cutoff는 OTO 파일의 right blank가 아니라, offset 이후 active length로 취급합니다.
    """
    from core.ja_oto_generator import validate_oto_params

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    try:
        dur_ms = float(wav_duration_ms or 0.0)
    except Exception:
        dur_ms = 0.0
    if dur_ms <= 0.0:
        return offset, consonant, cutoff, pre, ovl

    min_room_after_offset = 16.0 if alias_type in {"vc", "vv", "vcv"} else 22.0
    offset = max(0.0, min(float(offset), max(dur_ms - min_room_after_offset, 0.0)))

    max_active_len = max(dur_ms - offset - 6.0, 0.0)
    active_len = max(min(abs(float(cutoff)), max_active_len), 0.0)
    if active_len < 10.0:
        active_len = min(max(12.0, active_len), max(dur_ms - offset - 2.0, 0.0))

    max_cons = max(active_len - 6.0, 0.0)
    consonant = min(float(consonant), max_cons)
    if consonant < 8.0 and active_len >= 10.0:
        consonant = min(max(active_len * 0.72, 8.0), max_cons)

    pre = min(float(pre), max(consonant - 6.0, 0.0))
    pre = max(pre, 0.0)
    ovl = min(float(ovl), pre * 0.82 if pre > 0.0 else 0.0)
    ovl = max(ovl, 0.0)

    if consonant <= pre + 4.0:
        if max_cons >= pre + 6.0:
            consonant = pre + 6.0
        else:
            consonant = max_cons
            pre = max(0.0, consonant - 6.0)
            ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    if active_len <= 6.0:
        active_len = min(max(dur_ms - offset - 2.0, 0.0), 8.0)
        consonant = min(consonant, max(active_len - 2.0, 0.0))
        pre = min(pre, max(consonant - 2.0, 0.0))
        ovl = min(ovl, pre * 0.80 if pre > 0.0 else 0.0)

    cutoff = -max(active_len, 0.0)
    return offset, consonant, cutoff, pre, ovl


def _convert_ja_internal_cutoff_to_oto_field(oto_path, wav_dir):
    """
    일본어 OTO 최종 저장 단계 안전화.

    과거에는 내부 cutoff를 별도 좌표계로 보고 OTO cutoff 필드로 재변환했지만,
    실제 생성 결과를 비교해 보니 1차 생성본 cutoff 값이 이미 OTO 파일 의미론과
    호환되는 상태였다. 추가 변환을 적용하면 같은 파일 안 alias들의 cutoff가
    파일 끝 근처 공통 위치로 붕괴한다.

    현재는 좌표계 변환을 하지 않고, 최종 OTO 파일 의미론 기준 sanitize만 수행한다.
    """
    from core.ja_oto_generator import _parse_oto_line_profile, normalize_key
    from core.ja_oto_mapping import classify_ja_alias
    from core.oto_encoding import normalize_shift_jis_encoding, write_oto_lines_with_encoding
    from core.oto_generator import _find_wav_path_for_name, _wav_duration_ms
    from core.textio_utils import read_text_auto

    if not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    raw_text, detected_encoding, _read_err = read_text_auto(oto_path)
    if raw_text is None:
        return 0
    lines = str(raw_text).splitlines()
    preferred_encoding = normalize_shift_jis_encoding(detected_encoding)

    wav_index = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                wav_index[normalize_key(fn)] = os.path.join(wav_dir, fn)
    except Exception:
        return 0

    def _needs_finalize_sanitize(row, dur_ms):
        try:
            offset = float(row["offset"])
            consonant = float(row["cons"])
            cutoff = float(row["cutoff"])
            pre = float(row["pre"])
            ovl = float(row["ovl"])
            dur = float(dur_ms or 0.0)
        except Exception:
            return True
        if dur <= 0.0:
            return False
        cutoff_abs = abs(cutoff)
        active_len = dur - offset - cutoff_abs
        if cutoff >= 0.0:
            return True
        if offset < 0.0 or pre < 0.0 or ovl < 0.0:
            return True
        if pre > consonant + 1.0:
            return True
        if ovl > pre + 1.0:
            return True
        if cutoff_abs <= consonant + 1.0:
            return True
        if active_len <= 3.0:
            return True
        if active_len <= consonant + 4.0:
            return True
        return False

    out_lines = []
    changed = 0
    for line in lines:
        row = _parse_oto_line_profile(line)
        if not row:
            out_lines.append(line)
            continue

        wav_path = _find_wav_path_for_name(row["wav"], wav_dir, wav_index)
        dur_ms = _wav_duration_ms(wav_path) if wav_path else 0.0
        alias_type = classify_ja_alias(row["alias"])
        if dur_ms <= 0.0:
            out_lines.append(line)
            continue
        if not _needs_finalize_sanitize(row, dur_ms):
            out_lines.append(line)
            continue

        offset, consonant, cutoff_field, pre, ovl = sanitize_ja_oto_for_wav_duration(
            row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"], dur_ms, alias_type=alias_type
        )

        new_line = (
            f"{row['wav']}={row['alias']},"
            f"{offset:.2f},{consonant:.2f},{cutoff_field:.2f},{pre:.2f},{ovl:.2f}"
        )
        if new_line != line:
            changed += 1
        out_lines.append(new_line)

    if changed:
        write_oto_lines_with_encoding(
            oto_path,
            out_lines,
            language="japanese",
            preferred_encoding=preferred_encoding,
        )
    return changed


__all__ = [
    "sanitize_ja_oto_for_wav_duration",
    "_sanitize_ja_internal_params_for_wav_duration",
    "_convert_ja_internal_cutoff_to_oto_field",
]
