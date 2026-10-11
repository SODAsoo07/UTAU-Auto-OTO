from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Mapping

import yaml

from .domain import (
    AcousticSubtype,
    AliasPlan,
    AliasType,
    EdgeRole,
    PhoneDefinition,
    RecordingSpec,
    ResolvedAliasPlan,
    StructuralRole,
    TransitionType,
)


RULE_RESOURCE = "profiles/alias_type_rules_v1.yaml"
DERIVED_FEATURES = (
    "stop_like",
    "fricative_like",
    "affricate_like",
    "approximant_like",
    "glide_like",
    "vowel_like",
)
BOOLEAN_FEATURES = (
    "voiced",
    "syllabic",
    "sonorant",
    "continuant",
    "nasal",
    "breathy",
    "irregular_periodic",
    "transition_dominant",
    *DERIVED_FEATURES,
)
SUBTYPE_TEMPLATE_IDS = frozenset(
    {
        "cv_hard",
        "cv_soft",
        "cv_softer_hard",
        "cv_glide",
        "vc_hard",
        "vc_soft_voiced",
        "vc_soft_unvoiced",
    }
)


class FeatureNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedPhoneFeatures:
    values: Mapping[str, object]
    missing_canonical: tuple[str, ...]


@dataclass(frozen=True)
class AliasTypeRule:
    id: str
    priority: int
    structural_roles: tuple[StructuralRole, ...]
    when: Mapping[str, bool]
    subtype: AcousticSubtype
    template_id: str
    transition_type: TransitionType


@dataclass(frozen=True)
class AliasTypeRuleSet:
    id: str
    sha256: str
    rules: tuple[AliasTypeRule, ...]


def normalize_phone_features(phone: PhoneDefinition) -> NormalizedPhoneFeatures:
    raw = dict(phone.features)
    manner_value = raw.get("manner")
    if manner_value is not None and not isinstance(manner_value, str):
        raise FeatureNormalizationError(f"{phone.id}.features.manner must be a string")
    manner = manner_value.strip().casefold() if isinstance(manner_value, str) else None
    if manner_value is not None and not manner:
        raise FeatureNormalizationError(f"{phone.id}.features.manner must not be empty")

    values: dict[str, object] = dict(raw)
    if manner is not None:
        values["manner"] = manner
    place_value = raw.get("place")
    if place_value is not None and not isinstance(place_value, str):
        raise FeatureNormalizationError(f"{phone.id}.features.place must be a string")
    place = place_value.strip().casefold() if isinstance(place_value, str) else None
    if place_value is not None and not place:
        raise FeatureNormalizationError(f"{phone.id}.features.place must not be empty")
    if place is not None:
        values["place"] = place
    for name in BOOLEAN_FEATURES:
        value = raw.get(name)
        if value is not None and not isinstance(value, bool):
            raise FeatureNormalizationError(f"{phone.id}.features.{name} must be boolean")

    expected: dict[str, bool | None] = {
        "stop_like": None if manner is None else manner in {"stop", "plosive"},
        "fricative_like": None if manner is None else manner in {"fricative", "sibilant"},
        "affricate_like": None if manner is None else manner == "affricate",
        "approximant_like": None if manner is None else manner in {"approximant", "liquid"},
        "glide_like": (
            None
            if manner is None or manner == "approximant"
            else manner in {"glide", "semivowel"}
        ),
        "vowel_like": (
            True
            if raw.get("syllabic") is True or manner in {"vowel", "diphthong"}
            else None if raw.get("syllabic") is None and manner is None else False
        ),
    }
    for name, derived in expected.items():
        explicit = raw.get(name)
        if explicit is not None and derived is not None and explicit != derived:
            raise FeatureNormalizationError(
                f"{phone.id}.features.{name}={explicit!r} conflicts with canonical features ({derived!r})"
            )
        if explicit is not None:
            values[name] = explicit
        elif derived is not None:
            values[name] = derived

    if "sonorant" not in values and manner is not None:
        values["sonorant"] = manner in {
            "sonorant",
            "nasal",
            "approximant",
            "liquid",
            "glide",
            "semivowel",
            "vowel",
        }
    if "continuant" not in values and manner is not None:
        values["continuant"] = manner not in {"stop", "plosive", "affricate"}

    missing = tuple(name for name in ("manner", "voiced") if name not in values)
    return NormalizedPhoneFeatures(values=values, missing_canonical=missing)


def classify_alias_type(
    inventory: Mapping[str, PhoneDefinition],
    recording: RecordingSpec,
    plan: AliasPlan,
    *,
    rule_set: AliasTypeRuleSet | None = None,
) -> AliasType:
    rules = rule_set or load_alias_type_rules()
    structural_role = _structural_role(plan.role)
    edge_role = _edge_role(plan, len(recording.phones))
    edge_source = "alias_head_marker" if _has_alias_head_marker(plan.alias) else "phone_topology"
    reasons = [
        f"structural_role:{structural_role.value}",
        f"edge_role:{edge_role.value}",
        f"edge_role_source:{edge_source}",
    ]

    if structural_role in {StructuralRole.CC, StructuralRole.CCV}:
        return AliasType(
            structural_role,
            AcousticSubtype.UNKNOWN,
            TransitionType.CLUSTER,
            edge_role,
            0.0,
            (*reasons, "scope_excluded:CC_CCV"),
        )
    if structural_role is StructuralRole.BREATH:
        return AliasType(structural_role, AcousticSubtype.BREATH, TransitionType.BREATH, edge_role, 1.0, tuple(reasons))
    if structural_role is StructuralRole.SPECIAL:
        return AliasType(structural_role, AcousticSubtype.SPECIAL, TransitionType.UNKNOWN, edge_role, 1.0, tuple(reasons))
    if structural_role is StructuralRole.SILENCE:
        return AliasType(structural_role, AcousticSubtype.PLAIN, TransitionType.PLAIN, edge_role, 1.0, tuple(reasons))
    if structural_role in {StructuralRole.V, StructuralRole.V_BLEND, StructuralRole.V_END, StructuralRole.VV, StructuralRole.VCV}:
        transition = TransitionType.RELEASE if structural_role is StructuralRole.V_END else TransitionType.PLAIN
        return AliasType(structural_role, AcousticSubtype.PLAIN, transition, edge_role, 1.0, tuple(reasons))
    if structural_role not in {StructuralRole.CV, StructuralRole.VC}:
        return AliasType(
            structural_role,
            AcousticSubtype.UNKNOWN,
            TransitionType.UNKNOWN,
            edge_role,
            0.0,
            (*reasons, "unsupported_structural_role"),
        )

    candidates: list[tuple[int, PhoneDefinition, NormalizedPhoneFeatures]] = []
    for phone_index in plan.phone_indices:
        if phone_index < 0 or phone_index >= len(recording.phones):
            continue
        phone = inventory[recording.phones[phone_index]]
        normalized = normalize_phone_features(phone)
        if normalized.values.get("vowel_like") is not True:
            candidates.append((phone_index, phone, normalized))
    if not candidates:
        return AliasType(
            structural_role,
            AcousticSubtype.UNKNOWN,
            TransitionType.UNKNOWN,
            edge_role,
            0.0,
            (*reasons, "consonant_target_missing"),
        )
    target_index, target_phone, normalized = candidates[0] if structural_role is StructuralRole.CV else candidates[-1]
    reasons.extend((f"target_phone_index:{target_index}", f"target_phone:{target_phone.id}"))
    for rule in rules.rules:
        if structural_role not in rule.structural_roles:
            continue
        if all(normalized.values.get(name) is expected for name, expected in rule.when.items()):
            return AliasType(
                structural_role,
                rule.subtype,
                rule.transition_type,
                edge_role,
                1.0,
                (*reasons, f"rule:{rule.id}"),
            )
    missing = ",".join(normalized.missing_canonical) or "none"
    return AliasType(
        structural_role,
        AcousticSubtype.UNKNOWN,
        TransitionType.UNKNOWN,
        edge_role,
        0.0,
        (*reasons, f"no_subtype_rule_match:missing={missing}"),
    )


def resolve_alias_plan(
    inventory: Mapping[str, PhoneDefinition],
    recording: RecordingSpec,
    plan: AliasPlan,
    *,
    rule_set: AliasTypeRuleSet | None = None,
) -> ResolvedAliasPlan:
    rules = rule_set or load_alias_type_rules()
    alias_type = classify_alias_type(inventory, recording, plan, rule_set=rules)
    recommended = _recommended_template(alias_type)
    warnings: list[str] = []
    if alias_type.structural_role in {StructuralRole.CC, StructuralRole.CCV}:
        warnings.append("alias_type_scope_excluded_cc_ccv")
    if alias_type.acoustic_subtype is AcousticSubtype.UNKNOWN:
        warnings.append("alias_type_unknown")
    if recommended is not None and _template_short_id(plan.template) != _template_short_id(recommended):
        warnings.append("explicit_template_differs_from_classifier_recommendation")
    return ResolvedAliasPlan(
        source=plan,
        alias_type=alias_type,
        template_id=plan.template,
        selection_source="explicit",
        recommended_template_id=recommended,
        warnings=tuple(warnings),
        rule_set_sha256=rules.sha256,
    )


@lru_cache(maxsize=1)
def load_alias_type_rules() -> AliasTypeRuleSet:
    resource = files("universal_auto_oto").joinpath(RULE_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("alias type rule schema_version must be 1")
    rule_set_id = raw.get("rule_set_id")
    rows = raw.get("rules")
    if not isinstance(rule_set_id, str) or not rule_set_id or not isinstance(rows, list):
        raise ValueError("alias type rule set is invalid")
    parsed: list[AliasTypeRule] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"alias type rule {index} must be a mapping")
        allowed = {"id", "priority", "structural_roles", "when", "result"}
        if set(row) != allowed:
            raise ValueError(f"alias type rule {index} keys are invalid")
        rule_id = row["id"]
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
            raise ValueError(f"alias type rule id is invalid or duplicated: {rule_id!r}")
        seen.add(rule_id)
        roles_raw = row["structural_roles"]
        when_raw = row["when"]
        result = row["result"]
        if not isinstance(roles_raw, list) or not isinstance(when_raw, Mapping) or not isinstance(result, Mapping):
            raise ValueError(f"alias type rule {rule_id} shape is invalid")
        if set(result) != {"subtype", "template", "transition_type"}:
            raise ValueError(f"alias type rule {rule_id} result keys are invalid")
        conditions: dict[str, bool] = {}
        for name, value in when_raw.items():
            if name not in BOOLEAN_FEATURES:
                raise ValueError(f"alias type rule {rule_id} uses unsupported feature: {name}")
            if not isinstance(value, bool):
                raise ValueError(f"alias type rule {rule_id} condition must be boolean: {name}")
            conditions[str(name)] = value
        parsed.append(
            AliasTypeRule(
                id=rule_id,
                priority=int(row["priority"]),
                structural_roles=tuple(StructuralRole(str(value)) for value in roles_raw),
                when=conditions,
                subtype=AcousticSubtype(str(result["subtype"])),
                template_id=str(result["template"]),
                transition_type=TransitionType(str(result["transition_type"])),
            )
        )
    parsed.sort(key=lambda item: (-item.priority, item.id))
    return AliasTypeRuleSet(rule_set_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), tuple(parsed))


def _structural_role(value: str) -> StructuralRole:
    aliases = {
        "v": StructuralRole.V,
        "v_plain": StructuralRole.V,
        "v_head": StructuralRole.V,
        "v_blend": StructuralRole.V_BLEND,
        "v_end": StructuralRole.V_END,
        "release": StructuralRole.V_END,
        "cv": StructuralRole.CV,
        "cv_head": StructuralRole.CV,
        "cv_hard": StructuralRole.CV,
        "cv_soft": StructuralRole.CV,
        "cv_softer_hard": StructuralRole.CV,
        "cv_glide": StructuralRole.CV,
        "vc": StructuralRole.VC,
        "vc_hard": StructuralRole.VC,
        "vc_soft_voiced": StructuralRole.VC,
        "vc_soft_unvoiced": StructuralRole.VC,
        "vv": StructuralRole.VV,
        "vv_plain": StructuralRole.VV,
        "vcv": StructuralRole.VCV,
        "vcv_plain": StructuralRole.VCV,
        "cc": StructuralRole.CC,
        "ccv": StructuralRole.CCV,
        "c": StructuralRole.C,
        "breath": StructuralRole.BREATH,
        "silence": StructuralRole.SILENCE,
        "special": StructuralRole.SPECIAL,
    }
    return aliases.get(value.strip().casefold(), StructuralRole.UNKNOWN)


def _edge_role(plan: AliasPlan, phone_count: int) -> EdgeRole:
    if _has_alias_head_marker(plan.alias):
        return EdgeRole.HEAD
    if not plan.phone_indices:
        return EdgeRole.ISOLATED
    touches_head = min(plan.phone_indices) == 0
    touches_tail = max(plan.phone_indices) == phone_count - 1
    if touches_head and touches_tail:
        return EdgeRole.ISOLATED
    if touches_head:
        return EdgeRole.HEAD
    if touches_tail:
        return EdgeRole.TAIL
    return EdgeRole.MEDIAL


def _has_alias_head_marker(alias: str) -> bool:
    return alias.lstrip().startswith("- ")


def _recommended_template(alias_type: AliasType) -> str | None:
    mapping = {
        (StructuralRole.V, AcousticSubtype.PLAIN): "builtin:v_plain",
        (StructuralRole.V_BLEND, AcousticSubtype.PLAIN): "builtin:v_blend",
        (StructuralRole.V_END, AcousticSubtype.PLAIN): "builtin:v_end",
        (StructuralRole.VV, AcousticSubtype.PLAIN): "builtin:vv_plain",
        (StructuralRole.VCV, AcousticSubtype.PLAIN): "builtin:vcv_plain",
        (StructuralRole.CV, AcousticSubtype.HARD): "builtin:cv_hard",
        (StructuralRole.CV, AcousticSubtype.SOFTER_HARD): "builtin:cv_softer_hard",
        (StructuralRole.CV, AcousticSubtype.SOFT_VOICED): "builtin:cv_soft",
        (StructuralRole.CV, AcousticSubtype.SOFT_UNVOICED): "builtin:cv_soft",
        (StructuralRole.CV, AcousticSubtype.GLIDE): "builtin:cv_glide",
        (StructuralRole.VC, AcousticSubtype.HARD): "builtin:vc_hard",
        (StructuralRole.VC, AcousticSubtype.SOFT_VOICED): "builtin:vc_soft_voiced",
        (StructuralRole.VC, AcousticSubtype.SOFT_UNVOICED): "builtin:vc_soft_unvoiced",
        (StructuralRole.BREATH, AcousticSubtype.BREATH): "builtin:breath",
        (StructuralRole.SILENCE, AcousticSubtype.PLAIN): "builtin:silence",
        (StructuralRole.SPECIAL, AcousticSubtype.SPECIAL): "builtin:special_generic",
    }
    return mapping.get((alias_type.structural_role, alias_type.acoustic_subtype))


def _template_short_id(value: str) -> str:
    return value.split("#")[-1].split(":")[-1].strip().casefold()
