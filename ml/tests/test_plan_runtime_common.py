from core.generation.plan_runtime import (
    build_common_plan_context,
    build_language_runtime_state,
    extract_language_runtime_state,
    log_sinsy_plan_guard,
    normalize_runtime_postprocess_state,
    recompute_common_plan_runtime_state,
    resolve_common_runtime_policy,
)


def test_build_common_plan_context_fallback_path():
    out = build_common_plan_context(
        planned_cv_source=["ka", "ra"],
        syllables_info=[{"word": "ka"}, {"word": "ra"}],
        sinsy_label_entries=[],
        format_type="cvvc",
        alignment_weight=0.8,
        prefer_sequence=True,
        normalize_expected_fn=lambda s: s,
        normalize_label_fn=lambda s: s,
        label_match_score_fn=lambda _a, _b: 0.0,
        should_enable_mel_plan_fn=lambda: True,
        build_cv_anchor_plan_fn=lambda expected_tokens, _infos, use_mel=False: {
            "indices": list(range(len(expected_tokens))),
            "meta": {"margin": 7.0, "use_mel": bool(use_mel)},
            "source": "fallback",
        },
        build_sinsy_guided_anchor_plan_fn=lambda **_kwargs: {"indices": None, "meta": {}, "source": ""},
        build_adjacent_anchor_graph_fn=lambda indices: {"count": len(indices or [])},
        resolve_plan_policy_fn=lambda **kwargs: {"coverage": 1.0, "prefer_sequence": bool(kwargs.get("prefer_sequence"))},
    )
    assert out["cv_plan"]["indices"] == [0, 1]
    assert out["cv_plan"]["meta"]["use_mel"] is True
    assert out["anchor_graph"]["count"] == 2
    assert out["plan_policy"]["prefer_sequence"] is True


def test_resolve_common_runtime_policy_forwards_strict_formats():
    captured = {}

    def _resolver(**kwargs):
        captured.update(kwargs)
        return {"mapping_confidence": 0.4}

    out = resolve_common_runtime_policy(
        ingest_snapshot={},
        plan_policy={"coverage": 0.9},
        mapping_confidence=0.6,
        mapping_margin=8.0,
        conf_threshold=0.62,
        format_type="cvvc",
        score_a=10.0,
        score_b=7.0,
        sequence_lock_formats={"cvvc"},
        abstain_formats={"cvvc", "cv"},
        strict_formats={"cvvc"},
        prefer_sequence=True,
        alignment_trust=0.7,
        resolve_runtime_mapping_policy_fn=_resolver,
    )
    assert out["mapping_confidence"] == 0.4
    assert captured["strict_formats"] == {"cvvc"}
    assert captured["sequence_lock_formats"] == {"cvvc"}


def test_log_sinsy_plan_guard_emits_margin_warning():
    logs = []
    log_sinsy_plan_guard(
        sinsy_label_entries=[{"token": "ka"}],
        cv_plan={"source": "sinsy_labels", "meta": {"margin": 4.0}},
        runtime_policy={"row_margin_floor": 6.0},
        fname="x.wav",
        log_fn=logs.append,
    )
    assert any("margin" in line for line in logs)


def test_recompute_common_plan_runtime_state_builds_and_resolves():
    captured = {}

    def _resolver(**kwargs):
        captured.update(kwargs)
        return {"mapping_confidence": 0.52, "mapping_tier": "mid", "force_sequence_lock": True}

    out = recompute_common_plan_runtime_state(
        build_plan_context_fn=lambda **_kwargs: {
            "cv_plan": {"indices": [0], "meta": {"margin": 7.0}, "source": "fallback"},
            "planned_indices": [0],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 1.0},
        },
        plan_context_kwargs={"dummy": 1},
        ingest_snapshot={},
        mapping_confidence=0.61,
        mapping_margin=8.0,
        conf_threshold=0.62,
        format_type="cvvc",
        score_a=12.0,
        score_b=9.0,
        sequence_lock_formats={"cvvc"},
        abstain_formats={"cvvc", "cv"},
        strict_formats={"cvvc"},
        prefer_sequence=True,
        alignment_trust=0.7,
        resolve_runtime_mapping_policy_fn=_resolver,
    )
    assert out["mapping_confidence"] == 0.52
    assert out["mapping_margin"] == 8.0
    assert out["mapping_tier"] == "mid"
    assert out["cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["runtime_policy"]["force_sequence_lock"] is True
    assert captured["strict_formats"] == {"cvvc"}


def test_normalize_runtime_postprocess_state_defaults():
    out = normalize_runtime_postprocess_state()
    assert out["file_mapping_low_conf"] is False
    assert out["file_conf_floor"] == 0.0
    assert out["row_conf_floor"] == 0.0
    assert out["row_margin_floor"] == 0.0
    assert out["row_blank_floor"] is None
    assert out["low_conf_reasons"] == []
    assert out["sequence_lock_applied"] is False
    assert out["sequence_lock_reason"] == ""


def test_build_language_runtime_state_includes_prefix_and_postprocess():
    out = build_language_runtime_state(
        language_prefix="ja",
        mapping_confidence_base=0.58,
        mapping_margin=7.0,
        mapping_tier="mid",
        cv_plan={"indices": [0]},
        planned_indices=[0],
        anchor_graph={"count": 1},
        plan_policy={"coverage": 1.0},
        runtime_policy={"force_sequence_lock": True},
        postprocess={
            "file_mapping_low_conf": True,
            "file_conf_floor": 0.62,
            "row_conf_floor": 0.60,
            "row_margin_floor": 6.0,
            "row_blank_floor": 0.64,
            "low_conf_reasons": ["conf_below_threshold"],
            "sequence_lock_applied": True,
            "sequence_lock_reason": "filename_low_conf_fallback",
        },
    )
    assert out["mapping_confidence_base"] == 0.58
    assert out["mapping_margin"] == 7.0
    assert out["mapping_tier"] == "mid"
    assert out["cv_plan"]["indices"] == [0]
    assert out["ja_cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["ja_planned_cv_indices"] == [0]
    assert out["file_mapping_low_conf"] is True
    assert out["sequence_lock_applied"] is True
    assert out["sequence_lock_reason"] == "filename_low_conf_fallback"


def test_extract_language_runtime_state_prefers_common_then_prefixed():
    out = extract_language_runtime_state(
        {
            "mapping_confidence_base": 0.63,
            "mapping_margin": 8.0,
            "mapping_tier": "mid",
            "cv_plan": {"indices": [1]},
            "planned_indices": [1],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 0.9},
            "runtime_policy": {"mapping_tier": "mid"},
            "file_mapping_low_conf": True,
            "file_conf_floor": 0.64,
            "row_conf_floor": 0.60,
            "row_margin_floor": 6.0,
            "row_blank_floor": 0.64,
            "low_conf_reasons": ["conf_below_floor"],
            "sequence_lock_applied": False,
            "sequence_lock_reason": "",
            "kr_cv_plan": {"indices": [9]},
            "kr_planned_cv_indices": [9],
        },
        language_prefix="kr",
        conf_floor_default=0.62,
        row_margin_floor_default=6.0,
    )
    assert out["mapping_confidence_base"] == 0.63
    assert out["mapping_margin"] == 8.0
    assert out["mapping_tier"] == "mid"
    assert out["cv_plan"]["indices"] == [1]
    assert out["kr_cv_plan"]["indices"] == [1]
    assert out["planned_indices"] == [1]
    assert out["kr_planned_cv_indices"] == [1]
    assert out["file_mapping_low_conf"] is True
    assert out["row_blank_floor"] == 0.64


def test_extract_language_runtime_state_uses_prefixed_fallback():
    out = extract_language_runtime_state(
        {
            "mapping_confidence_base": 0.58,
            "mapping_margin": 7.0,
            "mapping_tier": "mid",
            "ja_cv_plan": {"indices": [0]},
            "ja_planned_cv_indices": [0],
            "ja_anchor_graph": {"count": 1},
            "ja_plan_policy": {"coverage": 1.0},
            "runtime_policy": {},
            "sequence_lock_applied": True,
            "sequence_lock_reason": "filename_low_conf_fallback",
        },
        language_prefix="ja",
        conf_floor_default=0.62,
        row_margin_floor_default=6.0,
    )
    assert out["cv_plan"]["indices"] == [0]
    assert out["ja_cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["ja_planned_cv_indices"] == [0]
    assert out["sequence_lock_applied"] is True
    assert out["runtime_policy"]["sequence_lock_applied"] is True
