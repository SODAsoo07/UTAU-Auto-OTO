from __future__ import annotations

MODEL_CONTEXT_SCHEMA_VERSION = 1

OTO_PARAM_ORDER: tuple[str, ...] = (
    "offset",
    "consonant",
    "cutoff",
    "preutterance",
    "overlap",
)

ANCHOR_ORDER: tuple[str, ...] = (
    "offset_abs",
    "overlap_abs",
    "pre_abs",
    "consonant_abs",
    "cutoff_abs",
)

ROLE_VOCAB: tuple[str, ...] = (
    "-cv",
    "cv",
    "v",
    "vc",
    "vv",
    "v-cv",
    "cv-",
    "v-",
    "br",
    "endbr",
    "special",
    "other",
)

ALIAS_TYPE_VOCAB: tuple[str, ...] = (
    "anchor",
    "boundary",
    "final",
    "silence",
    "other",
)

LANGUAGE_VOCAB: tuple[str, ...] = ("unknown", "ja", "kr", "multi")
FORMAT_TYPE_VOCAB: tuple[str, ...] = ("unknown", "cv", "cvvc", "vcv", "vccv", "c-v", "other")
