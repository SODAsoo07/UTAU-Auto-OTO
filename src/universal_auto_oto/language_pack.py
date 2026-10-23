"""Data-driven language packs: symbol inventory, filename grammar and alias shapes.

A pack lets a voicebank whose language/format has no hand-written route in
voicebank_project_adapter be read from YAML alone: file names and aliases are
tokenized with the pack's symbol table, aliases are bound to the recording's
phone occurrences by generic_alias_binding, and the pack's own templates render
the OTO rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .domain import PhoneDefinition, PhoneStateDefinition
from .pack_style import PackStyle, parse_style

PACK_RESOURCE_DIR = "language_packs"
PHONE_CLASSES = frozenset({"vowel", "syllabic", "consonant"})


class LanguagePackError(ValueError):
    pass


@dataclass(frozen=True)
class ShapeRule:
    shape: str
    pattern: re.Pattern[str]
    template: str
    role: str


@dataclass(frozen=True)
class LanguagePack:
    pack_id: str
    display_name: str
    languages: frozenset[str]
    formats: frozenset[str]
    case_sensitive: bool
    phones: Mapping[str, Mapping[str, object]]
    symbols: Mapping[str, str]
    alias_equivalents: Mapping[str, tuple[tuple[str, ...], ...]]
    separators: str
    drop_unmarked_consonant_prefix: bool
    strip_trailing: str
    start_markers: str
    head_marker: str
    after_consonant_marker: str
    end_marker: str
    unmarked_consonant_start_avoids_file_start: bool
    shape_rules: tuple[ShapeRule, ...]
    templates: Mapping[str, Mapping[str, object]]
    source: str
    sha256: str
    # Default labeler style (pack_style.py); a fitted profile can replace it.
    style: PackStyle | None = None
    # Development alignment profile the GUI uses for this pack (None: the GUI's recommendation).
    alignment_profile: str | None = None

    def template_ref(self, template_id: str) -> str:
        return f"pack:{self.pack_id}:{template_id}"

    def is_nucleus(self, phone: str) -> bool:
        return self.phones[phone]["class"] in {"vowel", "syllabic"}

    def phone_class(self, phone: str) -> str:
        """Coarse class for style keys: V, son, stop, affricate, tap, fric_v, fric_u."""
        spec = self.phones[phone]
        if spec["class"] != "consonant":
            return "V"
        manner = str(spec.get("manner", "unknown"))
        if manner in {"nasal", "liquid", "glide"}:
            return "son"
        if manner == "fricative":
            return "fric_v" if spec.get("voiced") else "fric_u"
        return manner

    def tokenize(self, text: str) -> tuple[str, ...] | None:
        """Longest-match symbols -> phone ids; None when any character is left over."""

        source = text if self.case_sensitive else text.casefold()
        result: list[str] = []
        index = 0
        while index < len(source):
            for symbol in self._ordered_symbols:
                if source.startswith(symbol, index):
                    result.append(self.symbols[symbol])
                    index += len(symbol)
                    break
            else:
                return None
        return tuple(result)

    @property
    def _ordered_symbols(self) -> tuple[str, ...]:
        return _ordered_symbols(self.pack_id, tuple(self.symbols))

    def parse_filename(self, name: str) -> tuple[tuple[str, ...], ...] | None:
        """File name -> phone ids grouped by separator-delimited chunks.

        Returns None when a chunk does not tokenize; the caller decides whether
        that is a review stop or an excluded recording.
        """

        stem = Path(name).stem.strip()
        chunks = [chunk for chunk in re.split(f"[{re.escape(self.separators)}]", stem)]
        parsed: list[tuple[str, ...]] = []
        for position, raw in enumerate(chunks):
            chunk = raw.strip()
            if self.strip_trailing:
                chunk = chunk.rstrip(self.strip_trailing)
            chunk = chunk.lstrip(self.start_markers)
            if not chunk:
                continue
            phones = self.tokenize(chunk)
            if phones is None:
                return None
            unmarked = position == 0 and raw[:1] not in self.start_markers
            if (self.drop_unmarked_consonant_prefix and unmarked and len(chunks) > 1
                    and not any(self.is_nucleus(phone) for phone in phones)):
                continue
            parsed.append(phones)
        return tuple(parsed) if parsed else None

    def phone_definition(self, phone: str) -> PhoneDefinition:
        spec = self.phones[phone]
        klass = str(spec["class"])
        manner = str(spec.get("manner", "vowel" if klass == "vowel" else "unknown"))
        voiced = bool(spec.get("voiced", klass != "consonant"))
        if klass in {"vowel", "syllabic"}:
            features: dict[str, object] = {"syllabic": True, "voiced": True, "vowel_like": True}
            if klass == "syllabic":
                features.update(manner=manner, sonorant=True)
            states = (
                PhoneStateDefinition("transition", "vowel_transition", 15.0, 220.0, 55.0),
                PhoneStateDefinition("stable", "vowel_stable", 45.0, 1400.0, 360.0),
            )
        elif manner in {"stop", "affricate"}:
            features = {"manner": manner, "voiced": voiced, "continuant": False}
            states = (
                PhoneStateDefinition("closure", "closure", 8.0, 220.0, 45.0),
                PhoneStateDefinition("release", "burst", 5.0, 80.0, 14.0),
                PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
            )
        elif manner == "tap":
            features = {"manner": "stop", "voiced": True, "continuant": False}
            states = (
                PhoneStateDefinition("closure", "closure", 5.0, 60.0, 20.0),
                PhoneStateDefinition("transition", "transition", 5.0, 80.0, 20.0),
            )
        elif manner == "fricative":
            features = {"manner": "fricative", "voiced": voiced, "continuant": True}
            states = (
                PhoneStateDefinition("frication", "frication", 12.0, 300.0, 75.0),
                PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
            )
        elif manner in {"nasal", "liquid", "glide"}:
            features = {"manner": "liquid" if manner == "glide" else manner, "voiced": True,
                        "sonorant": True, "continuant": True}
            states = (
                PhoneStateDefinition("consonant", "consonant", 10.0, 260.0, 55.0),
                PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
            )
        else:
            features = {"manner": "unknown", "voiced": voiced, "continuant": True}
            states = (
                PhoneStateDefinition("consonant", "consonant", 10.0, 260.0, 55.0),
                PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
            )
        return PhoneDefinition(id=phone, display_name=phone, features=features, states=states)

    def shape_of(self, groups: Sequence[Sequence[str]], prefix: str, suffix: str) -> str:
        parts = []
        for group in groups:
            letters = "".join("V" if self.is_nucleus(phone) else "C" for phone in group)
            parts.append(letters)
        return f"{prefix}{' '.join(parts)}{suffix}"

    def rule_for(self, shape: str) -> ShapeRule | None:
        for rule in self.shape_rules:
            if rule.pattern.fullmatch(shape):
                return rule
        return None


@lru_cache(maxsize=None)
def _ordered_symbols(_pack_id: str, symbols: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(symbols, key=lambda value: (-len(value), value)))


def _shape_pattern(shape: str) -> re.Pattern[str]:
    # "C+V+C*" style: letters are literal classes, +/* are repetition, spaces and markers literal.
    pattern = []
    for character in shape:
        if character in "CV":
            pattern.append(character)
        elif character in "+*":
            pattern.append(character)
        else:
            pattern.append(re.escape(character))
    return re.compile("".join(pattern))


def load_language_pack_text(text: str, source: str) -> LanguagePack:
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise LanguagePackError(f"{source}: schema_version must be 1")
    pack_id = str(raw.get("pack_id") or "").strip()
    if not re.fullmatch(r"[a-z0-9_]+", pack_id):
        raise LanguagePackError(f"{source}: pack_id must be [a-z0-9_]+")
    phones = raw.get("phones")
    symbols = raw.get("symbols")
    if not isinstance(phones, Mapping) or not phones or not isinstance(symbols, Mapping) or not symbols:
        raise LanguagePackError(f"{source}: phones and symbols are required")
    parsed_phones: dict[str, Mapping[str, object]] = {}
    for phone, spec in phones.items():
        phone_id = str(phone)
        if phone_id != phone_id.casefold() or not isinstance(spec, Mapping) or spec.get("class") not in PHONE_CLASSES:
            raise LanguagePackError(f"{source}: invalid phone {phone_id!r} (lowercase id and class required)")
        parsed_phones[phone_id] = dict(spec)
    parsed_symbols = {str(symbol): str(phone) for symbol, phone in symbols.items()}
    unknown = sorted(set(parsed_symbols.values()) - set(parsed_phones))
    if unknown:
        raise LanguagePackError(f"{source}: symbols map to unknown phones {unknown}")
    equivalents: dict[str, tuple[tuple[str, ...], ...]] = {}
    for phone, options in (raw.get("alias_equivalents") or {}).items():
        sequences = tuple(tuple(str(value) for value in option) for option in options)
        if str(phone) not in parsed_phones or any(value not in parsed_phones for seq in sequences for value in seq):
            raise LanguagePackError(f"{source}: alias_equivalents uses unknown phones ({phone})")
        equivalents[str(phone)] = sequences
    filename = raw.get("filename") or {}
    alias = raw.get("alias") or {}
    templates = raw.get("templates") or {}
    if not isinstance(templates, Mapping):
        raise LanguagePackError(f"{source}: templates must be a mapping")
    rules: list[ShapeRule] = []
    for item in raw.get("shape_templates") or ():
        template = str(item["template"])
        if template not in templates:
            raise LanguagePackError(f"{source}: shape {item['shape']!r} names unknown template {template!r}")
        rules.append(ShapeRule(str(item["shape"]), _shape_pattern(str(item["shape"])), template,
                               str(item.get("role") or templates[template].get("role") or template)))
    return LanguagePack(
        pack_id=pack_id,
        display_name=str(raw.get("display_name") or pack_id),
        languages=frozenset(str(value).casefold() for value in raw.get("languages") or ()),
        formats=frozenset(str(value).casefold() for value in raw.get("formats") or ()),
        case_sensitive=bool(raw.get("case_sensitive", False)),
        phones=parsed_phones,
        symbols=parsed_symbols,
        alias_equivalents=equivalents,
        separators=str(filename.get("separators", "_")),
        drop_unmarked_consonant_prefix=bool(filename.get("drop_unmarked_consonant_prefix", False)),
        strip_trailing=str(filename.get("strip_trailing", "")),
        start_markers=str(filename.get("start_markers", "")),
        head_marker=str(alias.get("head_marker", "-")),
        after_consonant_marker=str(alias.get("after_consonant_marker", "")),
        end_marker=str(alias.get("end_marker", "-")),
        unmarked_consonant_start_avoids_file_start=bool(alias.get("unmarked_consonant_start_avoids_file_start", False)),
        shape_rules=tuple(rules),
        templates=dict(templates),
        source=source,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        style=parse_style(raw["style"], pack_id=pack_id, source=f"{source}#style") if raw.get("style") else None,
        alignment_profile=str(raw["alignment_profile"]) if raw.get("alignment_profile") else None,
    )


@lru_cache(maxsize=None)
def builtin_language_packs() -> tuple[LanguagePack, ...]:
    directory = files("universal_auto_oto").joinpath(PACK_RESOURCE_DIR)
    packs = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name.endswith(".yaml"):
            packs.append(load_language_pack_text(entry.read_text(encoding="utf-8"),
                                                 f"package:{PACK_RESOURCE_DIR}/{entry.name}"))
    return tuple(packs)


def find_language_pack(language_hint: str | None, format_hint: str | None) -> LanguagePack | None:
    language = (language_hint or "").strip().casefold()
    format_type = (format_hint or "").strip().casefold()
    for pack in builtin_language_packs():
        if language in pack.languages and format_type in pack.formats:
            return pack
    return None


def language_pack_by_id(pack_id: str) -> LanguagePack:
    for pack in builtin_language_packs():
        if pack.pack_id == pack_id:
            return pack
    raise LanguagePackError(f"unknown language pack: {pack_id}")


def pack_formats() -> frozenset[str]:
    return frozenset(fmt for pack in builtin_language_packs() for fmt in pack.formats)


__all__ = [
    "LanguagePack",
    "LanguagePackError",
    "ShapeRule",
    "builtin_language_packs",
    "find_language_pack",
    "language_pack_by_id",
    "load_language_pack_text",
    "pack_formats",
]
