"""Per-consonant-class runtime offset shift for systematic model bias.

The 2026-05-17 KO bias diagnostic (per-onset breakdown of F3+compound
output) showed two consistent, opposite-direction biases the model could not
fix on its own:

- **Voiced stops** (``b``, ``d``, ``g``, ``bw``, ``gw``, ``dw``…) land **−150
  to −250 ms early** because the closure→release transient is weak and the
  scorer's ``consonant_onset`` peaks slightly before the actual burst.
- **Sonorants** (``n``, ``m``, ``l``, ``r`` and their ``ny``/``my``/``ly``/
  ``ry`` glide forms) land **+80 to +225 ms late** because the formant
  transitions resemble a vowel onset and the scorer waits for clearer
  evidence inside the next vowel.

These are model-level errors that proper retraining will eventually fix.
Until then, the runtime can apply a small per-class shift (positive = move
offset later, negative = earlier) to the OTO output. All env vars default
to zero, so the shift is a strict opt-in.
"""
from __future__ import annotations

import os
from typing import Any


_VOICED_STOPS: frozenset[str] = frozenset({"b", "d", "g", "bb", "dd", "gg"})
_VOICELESS_STOPS: frozenset[str] = frozenset(
    {"p", "t", "k", "pp", "tt", "kk", "ph", "th", "kh"}
)
_AFFRICATES: frozenset[str] = frozenset({"j", "jj", "ch", "c"})
_FRICATIVES: frozenset[str] = frozenset({"s", "ss", "h"})
# Sonorants exclude bare ``r`` — see _LIQUID_R below. ``ry``/``rw`` glide
# variants remain in this group because their acoustic profile patterns
# with other nasal/lateral glides rather than with the bare tap/trill.
_SONORANTS: frozenset[str] = frozenset({"n", "m", "l"})
# Bare ``r`` gets its own class: in the 2026-05-17 KO diagnostic it was the
# only sonorant biased -124 ms early instead of +80~+225 ms late, so the
# sonorant -150 ms shift over-corrected it to -274 ms.
_LIQUID_R: frozenset[str] = frozenset({"r"})
_NASAL_CODA: frozenset[str] = frozenset({"ng"})

_CLASS_ENV_VAR: dict[str, str] = {
    "voiced_stop": "UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS",
    "voiceless_stop": "UTOA_BOUNDARY_SHIFT_VOICELESS_STOP_MS",
    "affricate": "UTOA_BOUNDARY_SHIFT_AFFRICATE_MS",
    "fricative": "UTOA_BOUNDARY_SHIFT_FRICATIVE_MS",
    "sonorant": "UTOA_BOUNDARY_SHIFT_SONORANT_MS",
    "liquid_r": "UTOA_BOUNDARY_SHIFT_LIQUID_R_MS",
    "nasal_coda": "UTOA_BOUNDARY_SHIFT_NASAL_CODA_MS",
    "glide": "UTOA_BOUNDARY_SHIFT_GLIDE_MS",
}

# Multiplier applied to a class's base shift when the onset ends in a y/w
# glide (e.g. ``by``, ``dw``, ``gy``). Glide variants in the diagnostic
# showed roughly half the base bias because the glide itself adds a
# softer onset transient — applying the same shift over-corrects them.
# Default 1.0 (off); set to ~0.5 in F3 preset to halve voiced_stop+glide
# shift while keeping bare-consonant shift at full strength.
_GLIDE_MULTIPLIER_ENV: str = "UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER"

_KO_CODA_SONORANT_TOKENS: frozenset[str] = frozenset({"N", "M", "L", "R", "NG", "ng"})
_KO_CODA_PLOSIVE_TOKENS: frozenset[str] = frozenset({"K", "T", "P"})
_KO_CODA_ASPIRATE_TOKENS: frozenset[str] = frozenset({"H"})


def _ko_coda_class(alias: str) -> str:
    """Classify a KO coda VC alias into ``sonorant`` / ``plosive`` / ``aspirate``.

    Returns ``""`` for non-coda aliases. The 2026-05-17 KO TABI breakdown
    showed the three classes need very different shifts:
    - Sonorant (L/M/N/NG/R/ng): source-OTO == manual exactly, shift must be 0
    - Plosive (K/T/P): +40 to +148 ms late even after source rollback
    - Aspirate (H): close to 0, individual outliers only
    """
    text = str(alias or "").strip()
    parts = text.split()
    if len(parts) != 2:
        return ""
    right = parts[1].split("_")[0]
    if not right:
        return ""
    if right in _KO_CODA_SONORANT_TOKENS:
        return "sonorant"
    if right in _KO_CODA_PLOSIVE_TOKENS:
        return "plosive"
    if right in _KO_CODA_ASPIRATE_TOKENS:
        return "aspirate"
    return ""


def _is_ko_coda_vc_alias(alias: str) -> bool:
    """Return True for any KO coda VC alias (sonorant/plosive/aspirate).

    Kept for backward compatibility with the original suppression check —
    callers that need the sub-class should use ``_ko_coda_class``.
    """
    return _ko_coda_class(alias) != ""


# Per-role offset shift, applied on top of the consonant-class shift.
# The 2026-05-17 VC diagnostic (user-reported "V-CV에서 V만 포함되는 오류"
# on CVVC voicebanks — internally these are ``vc`` role, not ``v-cv`` which
# is reserved for 3-token VCV-format ``a ki a`` aliases) showed bare-
# consonant VC rows (``a bw``/``eu g``/``u k`` etc.) land −500 to −800 ms
# early because the next-syllable consonant transient is too weak for the
# scorer and the pre/cons anchors collapse into the prior vowel. A positive
# shift here pulls the entire VC row later so the consonant region actually
# contains the consonant instead of just the vowel tail.
_ROLE_SHIFT_ENV_VAR: dict[str, str] = {
    "vc": "UTOA_BOUNDARY_SHIFT_VC_MS",
}

# Coda VC shift knobs — per coda articulation class (2026-05-17 KO TABI
# diagnostic showed wildly different biases per class):
#   - Sonorant (L/M/N/NG/R/ng): source-OTO == manual, default 0
#   - Plosive (K/T/P): +40 to +148 ms late after source rollback, default −90
#   - Aspirate (H): near 0 with outlier wavs, default 0
# When the role is ``vc`` AND the alias matches a coda token, role_shift_ms
# returns the class-specific value *instead of* the onset VC shift.
_VC_CODA_SHIFT_ENV_BY_CLASS: dict[str, str] = {
    "sonorant": "UTOA_BOUNDARY_SHIFT_VC_CODA_SONORANT_MS",
    "plosive": "UTOA_BOUNDARY_SHIFT_VC_CODA_PLOSIVE_MS",
    "aspirate": "UTOA_BOUNDARY_SHIFT_VC_CODA_ASPIRATE_MS",
}
# Backward-compat env: when set AND the class-specific env is unset, all
# coda classes use this single value (matches the v16 single-knob behavior).
_VC_CODA_SHIFT_ENV_LEGACY: str = "UTOA_BOUNDARY_SHIFT_VC_CODA_MS"


def extract_alias_onset(alias: str) -> str:
    """Return the right-most syllable's onset consonant cluster (lowercased).

    Examples (Korean romanization):
    - ``"b a"``  → ``"b"``
    - ``"- b"``  → ``"b"``
    - ``"eu by"`` → ``"by"``
    - ``"ga"``    → ``"g"``
    - ``"kkyo"``  → ``"kky"``
    - ``"br"`` / ``"-"`` / ``"a"`` (bare vowel) → ``""``
    Japanese color tags after ``_`` are stripped.
    """
    text = str(alias or "").strip().lower()
    if not text:
        return ""
    if text in {"br", "-", "r", "pau", "sil", "rest"}:
        return ""
    parts = text.split()
    if parts and parts[0] == "-" and len(parts) >= 2:
        token = parts[1]
    elif len(parts) == 2:
        # Two-token alias: pick whichever side actually carries a consonant.
        # KO CVVC `eu b` (vc): right token starts with consonant -> use right.
        # KO CV-style `b a`:    right is bare vowel -> use left.
        right = parts[1].split("_")[0]
        if right and right[0] not in "aiueo":
            token = right
        else:
            token = parts[0]
    elif len(parts) == 1 and not text.startswith("-"):
        token = parts[0]
    else:
        return ""
    token = token.split("_")[0]
    onset = ""
    for ch in token:
        if ch in "aiueo":
            break
        if ch.isalpha():
            onset += ch
        else:
            break
    return onset


def classify_onset(onset: str) -> str:
    """Map an onset cluster to one of the broad articulatory classes used by the shift table.

    Note on ``r``: bare ``r`` goes to ``liquid_r`` (its own class), while
    ``ry``/``rw`` glide variants fall into ``sonorant`` because in the
    diagnostic they cluster with ``ny``/``my``/``ly`` rather than with
    bare ``r``.
    """
    if not onset:
        return ""
    onset_low = onset.lower()
    if onset_low in _NASAL_CODA:
        return "nasal_coda"
    if onset_low in _LIQUID_R:
        return "liquid_r"
    base = onset_low
    if len(base) > 1 and base.endswith(("y", "w")):
        base = base[:-1]
    if base in _VOICED_STOPS:
        return "voiced_stop"
    if base in _VOICELESS_STOPS:
        return "voiceless_stop"
    if base in _AFFRICATES:
        return "affricate"
    if base in _FRICATIVES:
        return "fricative"
    if base in _SONORANTS:
        return "sonorant"
    if base in _LIQUID_R:
        # Glide variants ``ry``/``rw`` fall into sonorant, not liquid_r.
        return "sonorant"
    if onset_low in {"y", "w"}:
        return "glide"
    return ""


def consonant_class_shift_ms(alias: str) -> float:
    """Look up the per-class shift in ms for ``alias`` (0 when no env override).

    Glide variants (onset ending in ``y``/``w``) get their class's shift
    multiplied by ``UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER`` (default 1.0).
    The diagnostic showed that ``bw``/``dw``/``gw``/``by``/``dy``/``gy``
    needed roughly half the voiced-stop shift to avoid over-correction.
    """
    onset = extract_alias_onset(alias)
    cls = classify_onset(onset)
    env_var = _CLASS_ENV_VAR.get(cls, "")
    if not env_var:
        return 0.0
    raw = str(os.environ.get(env_var, "") or "").strip()
    if not raw:
        return 0.0
    try:
        shift = float(raw)
    except ValueError:
        return 0.0
    onset_low = str(onset or "").lower()
    if len(onset_low) > 1 and onset_low.endswith(("y", "w")):
        mult_raw = str(os.environ.get(_GLIDE_MULTIPLIER_ENV, "") or "").strip()
        if mult_raw:
            try:
                shift *= float(mult_raw)
            except ValueError:
                pass
    return shift


def shift_oto_params(
    params: dict[str, Any],
    *,
    shift_ms: float,
    duration_ms: float,
) -> dict[str, Any]:
    """Apply ``shift_ms`` to the OTO row's offset only.

    Preutterance / overlap / consonant / cutoff in the OTO format are all
    stored relative to the offset, so shifting the offset moves every anchor
    in lock-step and preserves the syllable's internal shape. Cutoff (which
    encodes a negative offset-relative magnitude) thus also moves with it.

    The new offset is clamped to ``[0, duration_ms − 5]`` so the row stays
    inside the wav. Sub-millisecond shifts are ignored to avoid spurious
    rewrites.
    """
    out = dict(params)
    if abs(float(shift_ms)) < 0.5:
        return out
    duration = max(10.0, float(duration_ms))
    offset = float(params.get("offset", 0.0) or 0.0)
    new_offset = max(0.0, min(duration - 5.0, offset + float(shift_ms)))
    out["offset"] = float(new_offset)
    return out


def role_shift_ms(role: str, *, alias: str = "") -> float:
    """Return the per-role offset shift in ms (default 0 unless env override).

    Currently only the ``vc`` role is wired through
    ``UTOA_BOUNDARY_SHIFT_VC_MS``; other roles return 0. Composed with the
    consonant-class shift by the caller (``boundary_generator``).

    When ``alias`` is provided AND the ``vc`` role's right-token matches a
    KO coda convention (``ng``/uppercase coda letters), the role shift is
    suppressed — coda VC reference is inside the syllable, not at the next-
    syllable onset, so the +400 onset-VC push over-corrects them.
    """
    role_low = str(role or "").strip().lower()
    env_var = _ROLE_SHIFT_ENV_VAR.get(role_low, "")
    if not env_var:
        return 0.0
    if role_low == "vc" and alias:
        coda_class = _ko_coda_class(alias)
        if coda_class:
            # Coda VC: prefer class-specific env, fall back to legacy single
            # env, then to 0 (suppression).
            class_env = _VC_CODA_SHIFT_ENV_BY_CLASS.get(coda_class, "")
            for candidate_env in (class_env, _VC_CODA_SHIFT_ENV_LEGACY):
                if not candidate_env:
                    continue
                coda_raw = str(os.environ.get(candidate_env, "") or "").strip()
                if not coda_raw:
                    continue
                try:
                    return float(coda_raw)
                except ValueError:
                    continue
            return 0.0
    raw = str(os.environ.get(env_var, "") or "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def combined_shift_ms(*, alias: str, role: str) -> float:
    """Sum of consonant-class shift and role-specific shift, in ms.

    The glide multiplier (``UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER``) is
    applied to both the class shift (inside ``consonant_class_shift_ms``)
    and the role shift here, because glide-bearing onsets like ``ny``/
    ``my``/``ly``/``by``/``dy``/``gy`` already place anchors closer to the
    next syllable than their bare-consonant counterparts; without the
    multiplier on the role shift, the VC +400 ms push over-corrects them
    by 300–900 ms (2026-05-17 VC diagnostic).
    """
    class_shift = float(consonant_class_shift_ms(alias))
    role_shift = float(role_shift_ms(role, alias=alias))
    if role_shift != 0.0:
        onset_low = str(extract_alias_onset(alias) or "").lower()
        if len(onset_low) > 1 and onset_low.endswith(("y", "w")):
            mult_raw = str(os.environ.get(_GLIDE_MULTIPLIER_ENV, "") or "").strip()
            if mult_raw:
                try:
                    role_shift *= float(mult_raw)
                except ValueError:
                    pass
    return class_shift + role_shift


__all__ = [
    "classify_onset",
    "combined_shift_ms",
    "consonant_class_shift_ms",
    "extract_alias_onset",
    "role_shift_ms",
    "shift_oto_params",
]
