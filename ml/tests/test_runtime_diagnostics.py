from core.generation.runtime_diagnostics import (
    build_mapping_abstain_meta,
    build_mapping_failure_meta,
)


def test_build_mapping_failure_meta_without_reason_in_diag():
    out = build_mapping_failure_meta(
        mapping_confidence=0.63,
        mapping_reason_code="filename_token",
        spn_ratio=0.24,
        include_reason_in_diag=False,
        extra={"forced_words_mapping": False},
    )
    assert out["diag_hint"] == "spn_ratio=0.24; conf=0.63"
    assert out["mapping_confidence"] == 0.63
    assert out["mapping_reason_code"] == "filename_token"
    assert out["forced_words_mapping"] is False
    assert "mapping_tier" not in out


def test_build_mapping_failure_meta_with_reason_and_tier():
    out = build_mapping_failure_meta(
        mapping_confidence=0.52,
        mapping_reason_code="alias_recover",
        spn_ratio=0.30,
        include_reason_in_diag=True,
        mapping_tier="low",
    )
    assert out["diag_hint"] == "spn_ratio=0.30; conf=0.52; reason=alias_recover"
    assert out["mapping_tier"] == "low"


def test_build_mapping_abstain_meta():
    out = build_mapping_abstain_meta(
        mapping_confidence=0.41,
        mapping_reason_code="filename_linear_low_conf",
        mapping_tier="low",
        plan_policy={"coverage": 0.75},
    )
    assert out["mapping_confidence"] == 0.41
    assert out["mapping_reason_code"] == "filename_linear_low_conf"
    assert out["mapping_tier"] == "low"
    assert out["plan_policy"] == {"coverage": 0.75}
