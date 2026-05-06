from core.generation.kr_runtime_policy import (
    build_kr_plan_candidate_source,
    prepare_kr_mapping_runtime_context,
    should_enable_kr_mel_plan,
)


def test_build_kr_plan_candidate_source_prefers_filename_targets():
    assert build_kr_plan_candidate_source(
        filename_cv_targets=["ka"],
        syllables_info=[{"roman_cv": "na"}],
    ) == ["ka"]

    assert build_kr_plan_candidate_source(
        filename_cv_targets=[],
        syllables_info=[{"roman_cv": "na"}, {"roman": "a"}],
    ) == ["na", "a"]


def test_should_enable_kr_mel_plan_only_when_trust_is_not_strong():
    assert should_enable_kr_mel_plan(
        file_format="cvvc",
        mel_ctx_for_file={"ok": True},
        syllables_info=[{"roman_cv": "ka"}],
        textgrid_trust_tier="high",
        low_phone_quality=False,
        blank_conf_mean=0.1,
        alignment_weight=0.8,
        env_bool_fn=lambda _name, _default: True,
    ) is False

    assert should_enable_kr_mel_plan(
        file_format="cvvc",
        mel_ctx_for_file={"ok": True},
        syllables_info=[{"roman_cv": "ka"}],
        textgrid_trust_tier="low",
        low_phone_quality=False,
        blank_conf_mean=0.1,
        alignment_weight=0.8,
        env_bool_fn=lambda name, default: name == "UTOA_KR_CVVC_MEL_PLAN" and default,
    ) is True


def test_prepare_kr_mapping_runtime_context_resolves_policy_routing_and_report():
    logs = []
    report = {}
    report_calls = []

    def recompute(**kwargs):
        assert kwargs["plan_context_kwargs"]["should_enable_mel_plan_fn"]() is True
        return {
            "cv_plan": {"source": "mel", "meta": {"margin": 3.5}},
            "planned_indices": [1, 2],
            "anchor_graph": {"edges": 1},
            "plan_policy": {"coverage": 0.75},
            "runtime_policy": {
                "mapping_confidence": 0.57,
                "file_conf_floor": 0.55,
                "mapping_tier": "mid",
            },
            "mapping_confidence": 0.59,
            "mapping_margin": 4.0,
        }

    def low_conf(**kwargs):
        assert round(kwargs["mapping_confidence"], 4) == 0.5075
        return {
            "file_low_conf": True,
            "row_conf_floor": 0.58,
            "row_margin_floor": 7.0,
            "low_conf_reasons": ["conf_below_floor"],
        }

    def routing(**kwargs):
        assert round(kwargs["mapping_confidence_base"], 4) == 0.5075
        assert kwargs["pitch_zone"] == "high"
        return {
            "force_mel_branch": True,
            "mel_weight_mode": "mel_boost",
            "alignment_weight_final": 0.42,
            "anchor_lock_lite_default": True,
            "row_blank_floor": 0.61,
            "delta_clamp_scale": 0.5,
            "profile_code": "high_pitch_safe",
            "high_pitch_mode": True,
        }

    def update_report(runtime_report, **kwargs):
        report_calls.append(kwargs)
        runtime_report["mapping"] = kwargs

    context = prepare_kr_mapping_runtime_context(
        syllables_info=[{"roman_cv": "ka"}],
        sinsy_label_entries=[],
        filename_cv_targets=[],
        file_format="cvvc",
        alignment_ingest=object(),
        prefer_filename_sequence=False,
        alignment_weight=0.64,
        phone_quality={},
        used_words_based=True,
        used_alias_based=False,
        base_score=2.0,
        alt_score=4.0,
        file_mapping_conf_th=0.56,
        textgrid_trust_score=0.5,
        textgrid_trust_tier="low",
        low_phone_quality=False,
        blank_conf_mean=0.6,
        syllable_blank_confidences=[0.6],
        mel_ctx_for_file={"f0_pitch_zone": "high", "f0_note_hint_hz": 440.0, "f0_max_hz": 880.0},
        mel_reliability_score=0.3,
        mel_reliability_floor=0.44,
        mel_weight_mode="auto",
        mapping_reason_code="alias_based_cvvc",
        runtime_report=report,
        fname="take.wav",
        debug_logging=True,
        estimate_mapping_confidence_fn=lambda *_args, **_kwargs: (0.62, 5.0),
        recompute_common_plan_runtime_state_fn=recompute,
        build_common_plan_context_fn=lambda **_kwargs: {},
        resolve_runtime_mapping_policy_fn=lambda **_kwargs: {},
        log_sinsy_plan_guard_fn=lambda **kwargs: logs.append(f"sinsy:{kwargs['fname']}"),
        compute_runtime_low_conf_state_fn=low_conf,
        resolve_file_routing_profile_fn=routing,
        update_kr_mapping_runtime_report_fn=update_report,
        log_fn=logs.append,
        normalize_expected_fn=lambda value: value,
        normalize_label_fn=lambda value: value,
        cv_match_score_fn=lambda _target, _label: 1.0,
        build_cv_anchor_plan_fn=lambda **_kwargs: {},
        build_sinsy_guided_anchor_plan_fn=lambda **_kwargs: {},
        build_adjacent_anchor_graph_fn=lambda _indices: {},
        resolve_plan_policy_fn=lambda **_kwargs: {},
        env_bool_fn=lambda _name, _default: True,
    )

    assert context["kr_cv_plan"]["source"] == "mel"
    assert context["kr_planned_cv_indices"] == [1, 2]
    assert context["file_mapping_low_conf"] is True
    assert context["routing_profile_code"] == "high_pitch_safe"
    assert context["force_mel_branch"] is True
    assert context["alignment_weight"] == 0.42
    assert report_calls and report["mapping"]["extra_fields"]["routing_profile"] == "high_pitch_safe"
    assert any("KR 매핑 신뢰도 낮음" in log for log in logs)
