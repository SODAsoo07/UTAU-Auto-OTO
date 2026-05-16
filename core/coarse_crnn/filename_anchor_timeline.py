"""Filename-token-based slot anchor timeline.

For voicebanks whose wav filenames encode syllable structure as
apostrophe/hyphen-separated tokens (e.g.
``_geuNG'geuN'geuM'geuL'geu'geo'ge'ga.wav`` for an 8-mora Korean CV-VC bank,
or ``ba-bi-bu-be-bo-ba-N-ba.wav`` for a Japanese 連続音 bank), the filename
itself is the strongest available prior for *where each syllable starts*. The
RMS-valley / onset-density hybrid (`anchor_timeline.build_hybrid_anchor_timeline`)
tried to recover the same information from raw audio and failed on real
multi-mora wavs because syllable-internal valleys (consonant transients,
breaths) over-trigger the detector.

This module instead:

1. Parses the filename via `filename_order_tokens` (already in the codebase)
   into a list of syllable tokens, one per slot.
2. Estimates each token's relative duration from its structure (coda adds
   ~60-80 ms, diphthong glides add ~30 ms, double consonants add ~20 ms).
3. Distributes the active region across slots proportional to those
   per-token durations.

The output is the same shape `_build_anchor_timeline` produces in
`wav_decoder`: a list of `slot_count` start-of-slot times in milliseconds.
Returns `None` when the filename can't be parsed into exactly `slot_count`
tokens, so the caller can fall back to linear/hybrid logic instead of
silently producing a worse anchor placement.
"""
from __future__ import annotations

from core.coarse_crnn.slot_graph import filename_order_tokens


BASE_TOKEN_DURATION_MS: float = 400.0
KOREAN_CODA_BONUS_MS: dict[str, float] = {
    "ng": 75.0,
    "n": 55.0,
    "m": 55.0,
    "l": 55.0,
}
KOREAN_GLIDE_BONUS_MS: float = 30.0
TENSE_CONSONANT_BONUS_MS: float = 20.0
JAPANESE_NASAL_MORA_DURATION_MS: float = 250.0  # ん typically shorter than CV
JAPANESE_BASE_DURATION_MS: float = 400.0


# Korean monophthong / diphthong cores (longest first for greedy match).
_KO_VOWEL_CORES: tuple[str, ...] = (
    "weo", "wae", "yeo",
    "eu", "eo", "ae", "oe", "ui",
    "ya", "yu", "yo", "ye",
    "wa", "wi", "we", "wo",
    "a", "i", "u", "e", "o",
)
_KO_CODA_CANDIDATES: tuple[str, ...] = ("ng", "n", "m", "l")


def _ko_token_coda(token: str) -> str:
    """Return the coda suffix of a Korean romanized token, or '' if none."""
    low = token.lower()
    if not low:
        return ""
    # Find vowel core, then anything after is potential coda.
    last_vowel_end = -1
    last_vowel_core = ""
    for vowel in _KO_VOWEL_CORES:
        idx = low.rfind(vowel)
        if idx < 0:
            continue
        end = idx + len(vowel)
        if end > last_vowel_end:
            last_vowel_end = end
            last_vowel_core = vowel
    if last_vowel_end < 0:
        return ""
    tail = low[last_vowel_end:]
    if not tail:
        return ""
    # Greedy match against known coda tokens.
    for coda in _KO_CODA_CANDIDATES:
        if tail == coda:
            return coda
    return ""


def _ko_token_has_glide(token: str) -> bool:
    low = token.lower()
    if len(low) < 2:
        return False
    # Detect onset glide cluster like "gy", "gw", "py", "kk-w-" (rare).
    for ch_idx in range(len(low) - 1):
        if low[ch_idx] in "ywh":
            continue
        next_ch = low[ch_idx + 1]
        if next_ch in {"y", "w"}:
            return True
    return False


def _ko_token_has_tense_consonant(token: str) -> bool:
    low = token.lower()
    return any(
        cluster in low
        for cluster in ("kk", "gg", "tt", "dd", "pp", "bb", "ss", "jj", "cc")
    )


def estimate_token_duration_ms(
    token: str,
    *,
    language: str = "korean",
    base_duration_ms: float = BASE_TOKEN_DURATION_MS,
) -> float:
    """Approximate syllable duration from its romanized form.

    The absolute values do not matter for slot placement — only relative
    ratios drive the timeline distribution after scaling to the active
    region. Per-language heuristics aim to capture the systematic
    "syllables with coda take longer" effect that breaks pure linear
    partitioning on multi-mora wavs.
    """
    low = str(token or "").strip().lower()
    if not low:
        return float(base_duration_ms)

    lang = str(language or "").strip().lower()
    if lang in {"japanese", "ja", "jp", "jpn"}:
        if low in {"n"}:
            return float(JAPANESE_NASAL_MORA_DURATION_MS)
        return float(JAPANESE_BASE_DURATION_MS)

    duration = float(base_duration_ms)
    coda = _ko_token_coda(low)
    if coda:
        duration += KOREAN_CODA_BONUS_MS.get(coda, 50.0)
    if _ko_token_has_glide(low):
        duration += KOREAN_GLIDE_BONUS_MS
    if _ko_token_has_tense_consonant(low):
        duration += TENSE_CONSONANT_BONUS_MS
    return duration


def build_filename_anchor_timeline(
    wav_filename: str,
    *,
    slot_count: int,
    active_start_ms: float,
    active_end_ms: float,
    language: str = "korean",
) -> list[float] | None:
    """Return per-slot anchor times (ms) derived from filename token durations.

    Returns ``None`` when:
    - ``slot_count <= 1`` (degenerate, caller should bypass this layer)
    - the filename parses into a token count that disagrees with ``slot_count``
    - the active span is non-positive
    - estimated total duration is zero
    """
    slot_count = max(1, int(slot_count))
    if slot_count == 1:
        return None
    tokens = filename_order_tokens(wav_filename)
    if len(tokens) != slot_count:
        return None
    active_start = float(active_start_ms)
    active_end = float(active_end_ms)
    if active_end <= active_start:
        return None

    durations = [estimate_token_duration_ms(t, language=language) for t in tokens]
    total = sum(durations)
    if total <= 0.0:
        return None

    span = active_end - active_start
    scale = span / total
    timeline: list[float] = []
    pos = active_start
    for d in durations:
        timeline.append(float(pos))
        pos += float(d) * scale
    return timeline


__all__ = [
    "BASE_TOKEN_DURATION_MS",
    "JAPANESE_BASE_DURATION_MS",
    "JAPANESE_NASAL_MORA_DURATION_MS",
    "KOREAN_CODA_BONUS_MS",
    "KOREAN_GLIDE_BONUS_MS",
    "TENSE_CONSONANT_BONUS_MS",
    "build_filename_anchor_timeline",
    "estimate_token_duration_ms",
]
