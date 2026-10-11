from pathlib import Path

import pytest

from universal_auto_oto.alias_classifier import (
    FeatureNormalizationError,
    classify_alias_type,
    load_alias_type_rules,
    normalize_phone_features,
    resolve_alias_plan,
)
from universal_auto_oto.domain import (
    AcousticSubtype,
    AliasPlan,
    EdgeRole,
    PhoneDefinition,
    PhoneStateDefinition,
    RecordingSpec,
    StructuralRole,
)
from universal_auto_oto.project import ProjectValidationError, load_phone_inventory


def _phone(phone_id: str, **features: object) -> PhoneDefinition:
    return PhoneDefinition(
        id=phone_id,
        display_name=phone_id,
        features=features,
        states=(PhoneStateDefinition("interval", "phone_interval", 10.0, 200.0, 50.0),),
    )


def _recording(role: str, phones: tuple[str, ...] = ("c", "a")) -> tuple[RecordingSpec, AliasPlan]:
    plan = AliasPlan("alias", tuple(range(len(phones))), "builtin:cv", role)
    return (
        RecordingSpec("x.wav", phones, (plan,), "test", "none"),
        plan,
    )


@pytest.mark.parametrize(
    ("features", "role", "expected"),
    [
        ({"manner": "stop", "voiced": False}, "cv", AcousticSubtype.HARD),
        ({"manner": "stop", "voiced": True}, "cv", AcousticSubtype.SOFTER_HARD),
        ({"manner": "fricative", "voiced": False}, "cv", AcousticSubtype.SOFT_UNVOICED),
        ({"manner": "nasal", "voiced": True}, "cv", AcousticSubtype.SOFT_VOICED),
        ({"manner": "liquid", "voiced": True}, "cv", AcousticSubtype.SOFT_VOICED),
        ({"manner": "glide", "voiced": True}, "cv", AcousticSubtype.GLIDE),
        (
            {"manner": "approximant", "voiced": True, "transition_dominant": True},
            "cv",
            AcousticSubtype.GLIDE,
        ),
        ({"manner": "stop", "voiced": True}, "vc", AcousticSubtype.HARD),
    ],
)
def test_classifier_uses_feature_rules(features: dict[str, object], role: str, expected: AcousticSubtype) -> None:
    inventory = {
        "c": _phone("c", **features),
        "a": _phone("a", manner="vowel", voiced=True, syllabic=True),
    }
    recording, plan = _recording(role)
    classified = classify_alias_type(inventory, recording, plan)
    assert classified.acoustic_subtype is expected
    assert classified.confidence == 1.0
    assert any(reason.startswith("rule:") for reason in classified.reasons)


def test_cc_and_ccv_are_explicitly_out_of_scope() -> None:
    inventory = {"c": _phone("c", manner="stop", voiced=False)}
    recording, plan = _recording("cc", ("c",))
    classified = classify_alias_type(inventory, recording, plan)
    assert classified.structural_role is StructuralRole.CC
    assert classified.acoustic_subtype is AcousticSubtype.UNKNOWN
    assert "scope_excluded:CC_CCV" in classified.reasons


def test_alias_dash_space_marker_overrides_phone_topology_as_head_pronunciation() -> None:
    inventory = {
        "a": _phone("a", manner="vowel", voiced=True, syllabic=True),
        "k": _phone("k", manner="stop", voiced=False),
    }
    plan = AliasPlan("- ka", (1, 2), "builtin:cv", "cv")
    recording = RecordingSpec("x.wav", ("a", "k", "a", "a"), (plan,), "test", "none")

    classified = classify_alias_type(inventory, recording, plan)

    assert classified.edge_role is EdgeRole.HEAD
    assert "edge_role_source:alias_head_marker" in classified.reasons


def test_literal_sonorant_manner_is_soft_voiced() -> None:
    inventory = {
        "c": _phone("c", manner="sonorant", voiced=True),
        "a": _phone("a", manner="vowel", voiced=True, syllabic=True),
    }
    recording, plan = _recording("cv")

    classified = classify_alias_type(inventory, recording, plan)

    assert classified.acoustic_subtype is AcousticSubtype.SOFT_VOICED


def test_explicit_template_wins_and_recommendation_is_trace_only() -> None:
    inventory = {
        "c": _phone("c", manner="stop", voiced=False),
        "a": _phone("a", manner="vowel", voiced=True, syllabic=True),
    }
    recording, plan = _recording("cv")
    resolved = resolve_alias_plan(inventory, recording, plan)
    assert resolved.template_id == "builtin:cv"
    assert resolved.selection_source == "explicit"
    assert resolved.recommended_template_id == "builtin:cv_hard"
    assert "explicit_template_differs_from_classifier_recommendation" in resolved.warnings
    assert len(resolved.rule_set_sha256) == 64


@pytest.mark.parametrize(
    ("role", "expected_subtype", "expected_template"),
    [
        ("v", AcousticSubtype.PLAIN, "builtin:v_plain"),
        ("v_blend", AcousticSubtype.PLAIN, "builtin:v_blend"),
        ("v_end", AcousticSubtype.PLAIN, "builtin:v_end"),
        ("vv", AcousticSubtype.PLAIN, "builtin:vv_plain"),
        ("vcv", AcousticSubtype.PLAIN, "builtin:vcv_plain"),
        ("breath", AcousticSubtype.BREATH, "builtin:breath"),
        ("silence", AcousticSubtype.PLAIN, "builtin:silence"),
        ("special", AcousticSubtype.SPECIAL, "builtin:special_generic"),
    ],
)
def test_safe_non_subtype_roles_have_traceable_builtin_recommendations(
    role: str,
    expected_subtype: AcousticSubtype,
    expected_template: str,
) -> None:
    inventory = {"c": _phone("c", manner="special", voiced=False)}
    recording, plan = _recording(role, ("c",))
    resolved = resolve_alias_plan(inventory, recording, plan)
    assert resolved.alias_type.acoustic_subtype is expected_subtype
    assert resolved.recommended_template_id == expected_template


def test_normalizer_rejects_derived_feature_conflict() -> None:
    phone = _phone("k", manner="stop", voiced=False, stop_like=False)
    with pytest.raises(FeatureNormalizationError, match="conflicts"):
        normalize_phone_features(phone)


def test_normalizer_canonicalizes_place_and_rejects_invalid_optional_boolean() -> None:
    normalized = normalize_phone_features(_phone("k", manner="Stop", place="Velar", voiced=False))
    assert normalized.values["manner"] == "stop"
    assert normalized.values["place"] == "velar"
    with pytest.raises(FeatureNormalizationError, match="transition_dominant must be boolean"):
        normalize_phone_features(
            _phone("w", manner="approximant", place="labial", voiced=True, transition_dominant="yes")
        )


def test_project_inventory_rejects_derived_feature_conflict(tmp_path: Path) -> None:
    path = tmp_path / "phones.yaml"
    path.write_text(
        """format_version: 1
phones:
  k:
    display_name: K
    features: {manner: stop, voiced: false, stop_like: false}
    states: [{id: closure, acoustic_role: closure, min_ms: 10, max_ms: 100}]
""",
        encoding="utf-8",
    )
    with pytest.raises(ProjectValidationError, match="conflicts"):
        load_phone_inventory(path)


def test_rule_resource_is_versioned_and_hashed() -> None:
    rules = load_alias_type_rules()
    assert rules.id == "builtin_alias_type_rules_v1"
    assert len(rules.sha256) == 64
    assert rules.rules == tuple(sorted(rules.rules, key=lambda item: (-item.priority, item.id)))
