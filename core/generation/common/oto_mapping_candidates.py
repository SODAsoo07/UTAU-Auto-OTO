from __future__ import annotations


def collect_phone_activity_metrics(
    phones,
    *,
    clean_mark_fn,
    is_vowel_fn,
    silence_like=None,
):
    silence_like = set(silence_like or {"", "sil", "pau", "sp", "spn", "br", "bre", "noise"})
    phone_list = list(phones or [])
    if not phone_list:
        return 0.0, 0.0, 0

    active_ms = 0.0
    vowel_ms = 0.0
    non_sil_count = 0
    for phone in phone_list:
        mark = clean_mark_fn(getattr(phone, "mark", ""))
        if mark in silence_like:
            continue
        try:
            dur_ms = max(
                (float(getattr(phone, "maxTime", 0.0)) - float(getattr(phone, "minTime", 0.0))) * 1000.0,
                0.0,
            )
        except Exception:
            dur_ms = 0.0
        if dur_ms <= 0.0:
            continue
        non_sil_count += 1
        active_ms += dur_ms
        if is_vowel_fn(mark):
            vowel_ms += dur_ms
    return active_ms, vowel_ms, non_sil_count


def is_active_candidate(
    phones,
    *,
    clean_mark_fn,
    is_vowel_fn,
    require_vowel=True,
    min_active_ms=16.0,
    min_vowel_ms=10.0,
    silence_like=None,
):
    active_ms, vowel_ms, non_sil_count = collect_phone_activity_metrics(
        phones,
        clean_mark_fn=clean_mark_fn,
        is_vowel_fn=is_vowel_fn,
        silence_like=silence_like,
    )
    if non_sil_count <= 0:
        return False
    if active_ms < float(min_active_ms):
        return False
    if require_vowel and vowel_ms < float(min_vowel_ms):
        return False
    return True


__all__ = [
    "collect_phone_activity_metrics",
    "is_active_candidate",
]
