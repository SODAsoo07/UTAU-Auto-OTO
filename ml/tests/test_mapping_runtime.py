from core.generation.mapping_runtime import (
    build_common_mapping_runtime_payload,
    compute_runtime_low_conf_state,
    format_alignment_guard_summary,
    format_mapping_reason_schema_summary,
    format_mapping_summary,
    update_ja_mapping_runtime_report,
    update_kr_mapping_runtime_report,
)


def test_build_common_mapping_runtime_payload_normalizes_reason_schema():
    payload = build_common_mapping_runtime_payload(
        language="ja",
        fallback_reason_code="filename_token",
        format_value="cvvc",
        mapping_confidence=0.72,
        mapping_margin=8.0,
        mapping_tier="mid",
        trust_score=0.61,
        trust_tier="mid",
        row_conf_floor=0.60,
        row_margin_floor=6.0,
        file_low_conf=False,
        low_conf_reasons=["x"],
        blank_confidence_mean=0.12,
        plan_source="fallback",
        plan_margin=7.0,
        plan_coverage=0.9,
        mapping_reason_code="custom_reason_x",
    )
    assert payload["format"] == "cvvc"
    assert payload["mapping_reason_code"] == "custom_reason_x"
    assert payload["mapping_reason_known"] is False
    assert payload["mapping_reason_stage"] == "unknown"


def test_update_kr_mapping_runtime_report_sets_payload():
    report = {}
    update_kr_mapping_runtime_report(
        report,
        file_format="cvvc",
        mapping_confidence=0.72,
        mapping_margin=8.2,
        mapping_tier="mid",
        trust_score=0.61,
        trust_tier="mid",
        file_conf_floor=0.64,
        row_conf_floor=0.60,
        row_margin_floor=6.0,
        file_low_conf=False,
        low_conf_reasons=["a"],
        blank_confidence_mean=0.12,
        plan_source="sinsy_labels",
        plan_margin=7.2,
        plan_coverage=0.95,
        mapping_reason_code="filename_token",
        row_blank_floor=0.63,
    )
    mapping = report.get("mapping") or {}
    assert mapping.get("format") == "cvvc"
    assert abs(float(mapping.get("mapping_confidence")) - 0.72) < 1e-9
    assert mapping.get("row_blank_floor") == 0.63
    assert mapping.get("mapping_reason_code") == "filename_token"
    assert mapping.get("mapping_reason_known") is True
    assert mapping.get("mapping_reason_stage") == "candidate_select"


def test_update_kr_mapping_runtime_report_accepts_extra_fields():
    report = {}
    update_kr_mapping_runtime_report(
        report,
        file_format="cvvc",
        mapping_confidence=0.72,
        mapping_margin=8.2,
        mapping_tier="mid",
        trust_score=0.61,
        trust_tier="mid",
        file_conf_floor=0.64,
        row_conf_floor=0.60,
        row_margin_floor=6.0,
        file_low_conf=False,
        low_conf_reasons=["a"],
        blank_confidence_mean=0.12,
        plan_source="sinsy_labels",
        plan_margin=7.2,
        plan_coverage=0.95,
        mapping_reason_code="filename_token",
        extra_fields={
            "routing_profile": "hybrid_soft",
            "alignment_weight": 0.77,
            "flags": {"anchor_lock_lite": True},
            "levels": [0.1, 0.2],
        },
    )
    mapping = report.get("mapping") or {}
    assert mapping.get("routing_profile") == "hybrid_soft"
    assert mapping.get("alignment_weight") == 0.77
    assert (mapping.get("flags") or {}).get("anchor_lock_lite") is True
    assert mapping.get("levels") == [0.1, 0.2]


def test_update_ja_mapping_runtime_report_sets_payload():
    report = {}
    update_ja_mapping_runtime_report(
        report,
        format_type="cvvc",
        mapping_confidence=0.67,
        mapping_margin=5.1,
        mapping_tier="low",
        trust_score=0.54,
        trust_tier="low",
        conf_threshold=0.62,
        row_conf_floor=0.58,
        row_margin_floor=5.5,
        file_low_conf=True,
        low_conf_reasons=["conf_below_threshold"],
        blank_confidence_mean=0.44,
        plan_source="fallback",
        plan_margin=3.0,
        plan_coverage=0.66,
        mapping_reason_code="alias_recover",
        filename_order_locked=False,
        forced_words_mapping=False,
    )
    mapping = report.get("mapping") or {}
    assert mapping.get("format") == "cvvc"
    assert mapping.get("file_low_conf") is True
    assert mapping.get("mapping_reason_code") == "alias_recover"
    assert mapping.get("mapping_reason_known") is True
    assert mapping.get("mapping_reason_stage") == "candidate_select"


def test_update_ja_mapping_runtime_report_accepts_extra_fields():
    report = {}
    update_ja_mapping_runtime_report(
        report,
        format_type="cvvc",
        mapping_confidence=0.67,
        mapping_margin=5.1,
        mapping_tier="low",
        trust_score=0.54,
        trust_tier="low",
        conf_threshold=0.62,
        row_conf_floor=0.58,
        row_margin_floor=5.5,
        file_low_conf=True,
        low_conf_reasons=["conf_below_threshold"],
        blank_confidence_mean=0.44,
        plan_source="fallback",
        plan_margin=3.0,
        plan_coverage=0.66,
        mapping_reason_code="alias_recover",
        filename_order_locked=False,
        forced_words_mapping=False,
        extra_fields={
            "alignment_weight": 0.55,
            "anchor_lock_lite": True,
        },
    )
    mapping = report.get("mapping") or {}
    assert mapping.get("alignment_weight") == 0.55
    assert mapping.get("anchor_lock_lite") is True


def test_format_mapping_summary():
    text = format_mapping_summary(
        {
            "format": "cvvc",
            "mapping_tier": "mid",
            "mapping_confidence": 0.734,
            "mapping_reason_code": "filename_token",
        }
    )
    assert text == "format=cvvc tier=mid conf=0.73 reason=filename_token"


def test_format_mapping_reason_schema_summary():
    text = format_mapping_reason_schema_summary(
        {
            "mapping_reason_known": True,
            "mapping_reason_stage": "candidate_select",
            "mapping_reason_cause": "filename_token_match",
            "mapping_reason_recovery": "none",
        }
    )
    assert text == "known=True stage=candidate_select cause=filename_token_match recovery=none"


def test_format_mapping_reason_schema_summary_defaults():
    text = format_mapping_reason_schema_summary({})
    assert text == "known=False stage=unknown cause=unknown recovery=inspect_runtime_log"


def test_mapping_runtime_common_keys_across_languages():
    kr_report = {}
    ja_report = {}
    update_kr_mapping_runtime_report(
        kr_report,
        file_format="cvvc",
        mapping_confidence=0.81,
        mapping_margin=12.0,
        mapping_tier="high",
        trust_score=0.77,
        trust_tier="high",
        file_conf_floor=0.64,
        row_conf_floor=0.61,
        row_margin_floor=6.5,
        file_low_conf=False,
        low_conf_reasons=[],
        blank_confidence_mean=0.09,
        plan_source="sinsy_labels",
        plan_margin=9.0,
        plan_coverage=0.96,
        mapping_reason_code="filename_token",
    )
    update_ja_mapping_runtime_report(
        ja_report,
        format_type="cvvc",
        mapping_confidence=0.79,
        mapping_margin=10.5,
        mapping_tier="high",
        trust_score=0.74,
        trust_tier="high",
        conf_threshold=0.62,
        row_conf_floor=0.60,
        row_margin_floor=6.0,
        file_low_conf=False,
        low_conf_reasons=[],
        blank_confidence_mean=0.10,
        plan_source="fallback",
        plan_margin=7.4,
        plan_coverage=0.88,
        mapping_reason_code="filename_token",
        filename_order_locked=True,
        forced_words_mapping=False,
    )
    common_keys = {
        "format",
        "mapping_confidence",
        "mapping_margin",
        "mapping_tier",
        "trust_score",
        "trust_tier",
        "row_conf_floor",
        "row_margin_floor",
        "file_low_conf",
        "low_conf_reasons",
        "blank_confidence_mean",
        "plan_source",
        "plan_margin",
        "plan_coverage",
        "mapping_reason_code",
        "mapping_reason_known",
        "mapping_reason_stage",
        "mapping_reason_cause",
        "mapping_reason_recovery",
    }
    kr_mapping = kr_report.get("mapping") or {}
    ja_mapping = ja_report.get("mapping") or {}
    assert common_keys.issubset(set(kr_mapping.keys()))
    assert common_keys.issubset(set(ja_mapping.keys()))


def test_compute_runtime_low_conf_state_adds_reasons():
    out = compute_runtime_low_conf_state(
        runtime_policy={"low_conf_reasons": ["base"], "is_low_conf": False},
        mapping_confidence=0.51,
        conf_floor=0.62,
        blank_confidence_mean=0.60,
        conf_below_reason="conf_below_threshold",
        row_conf_floor_default=0.62,
    )
    assert out["file_low_conf"] is True
    assert "blank_confidence_high" in out["low_conf_reasons"]
    assert "conf_below_threshold" in out["low_conf_reasons"]
    assert out["row_conf_floor"] == 0.62


def test_format_alignment_guard_summary():
    text = format_alignment_guard_summary(
        alignment_weight=0.644,
        blank_confidence_mean=0.151,
        anchor_lock_lite=False,
    )
    assert text == "alignment_weight=0.64, blank_mean=0.15, anchor_lock_lite=False"


def test_update_mapping_runtime_report_unknown_reason_kept():
    report = {}
    update_ja_mapping_runtime_report(
        report,
        format_type="cvvc",
        mapping_confidence=0.4,
        mapping_margin=2.0,
        mapping_tier="low",
        trust_score=0.3,
        trust_tier="low",
        conf_threshold=0.62,
        row_conf_floor=0.58,
        row_margin_floor=5.5,
        file_low_conf=True,
        low_conf_reasons=["conf_below_threshold"],
        blank_confidence_mean=0.52,
        plan_source="fallback",
        plan_margin=2.0,
        plan_coverage=0.4,
        mapping_reason_code="custom_reason_x",
        filename_order_locked=False,
        forced_words_mapping=False,
    )
    mapping = report.get("mapping") or {}
    assert mapping.get("mapping_reason_code") == "custom_reason_x"
    assert mapping.get("mapping_reason_known") is False
    assert mapping.get("mapping_reason_stage") == "unknown"
