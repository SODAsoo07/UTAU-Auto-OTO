from __future__ import annotations

import os

from core.coarse_crnn.labels import coarse_for_phone
from core.coarse_crnn.lang import phones_from_wav_name


def estimate_reclist_phone_bounds(
    wav_path: str,
    phones: list[str],
    *,
    language: str,
    active_range_sec: tuple[float, float] | None,
) -> list[tuple[float, float]] | None:
    phone_list = [str(phone).strip() for phone in phones if str(phone).strip()]
    if not phone_list or active_range_sec is None:
        return None
    name_phones = phones_from_wav_name(os.path.basename(str(wav_path or "")), language)
    if not _compatible_name_phones(name_phones, phone_list, language=language):
        return None
    start, end = float(active_range_sec[0]), float(active_range_sec[1])
    if end <= start:
        return None
    priors = [_duration_prior(coarse_for_phone(phone, language=language)) for phone in phone_list]
    total = sum(priors)
    if total <= 0.0:
        return None
    cursor = start
    out: list[tuple[float, float]] = []
    for prior in priors:
        next_cursor = cursor + (end - start) * prior / total
        out.append((cursor, next_cursor))
        cursor = next_cursor
    return out


def _compatible_name_phones(name_phones: list[str], phones: list[str], *, language: str) -> bool:
    if not name_phones:
        return False
    if len(name_phones) == len(phones):
        return True
    if abs(len(name_phones) - len(phones)) > 2:
        return False
    name_coarse = [coarse_for_phone(phone, language=language) for phone in name_phones]
    target_coarse = [coarse_for_phone(phone, language=language) for phone in phones]
    common = min(len(name_coarse), len(target_coarse))
    if common <= 0:
        return False
    matches = sum(1 for idx in range(common) if name_coarse[idx] == target_coarse[idx])
    return matches / float(common) >= 0.70


def _duration_prior(coarse: str) -> float:
    return {
        "V": 0.090,
        "C_OBS": 0.045,
        "C_FR": 0.060,
        "C_SON": 0.055,
        "SP": 0.050,
        "BR": 0.120,
        "SIL": 0.100,
    }.get(str(coarse), 0.060)


__all__ = ["estimate_reclist_phone_bounds"]
