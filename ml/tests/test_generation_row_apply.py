from core.generation.row_apply import (
    apply_zero_template_compute_policy,
    build_review_required_unset,
    decide_row_application,
    record_row_apply_decision,
)


def test_decide_row_application_forces_conservative_for_low_conf_abstain():
    out = decide_row_application(
        routing_profile="hybrid_soft",
        pitch_zone="",
        row_mapping_confidence=0.42,
        row_conf_floor=0.62,
        row_blank_confidence=0.1,
        row_blank_floor=0.64,
        row_jump_blocked=0,
        forced_selected=False,
        file_mapping_low_conf=False,
        row_abstain_skip=True,
        row_abstain_reason="row_low_confidence",
    )

    assert out["mode"] == "conservative_apply"
    assert out["reason_code"] == "row_low_confidence"
    assert out["reasons"] == ["row_low_confidence", "low_conf_force_auto"]


def test_decide_row_application_uses_review_for_hard_abstain_reason():
    out = decide_row_application(
        routing_profile="hybrid_soft",
        pitch_zone="",
        row_mapping_confidence=0.8,
        row_conf_floor=0.62,
        row_blank_confidence=0.1,
        row_blank_floor=0.64,
        row_jump_blocked=0,
        forced_selected=False,
        file_mapping_low_conf=False,
        row_abstain_skip=True,
        row_abstain_reason="inactive_candidate",
    )

    assert out["mode"] == "review_required"
    assert out["reason_code"] == "inactive_candidate"


def test_decide_row_application_combines_soft_risk_reasons():
    out = decide_row_application(
        routing_profile="high_pitch_safe",
        pitch_zone="high",
        row_mapping_confidence=0.55,
        row_conf_floor=0.62,
        row_blank_confidence=0.7,
        row_blank_floor=0.64,
        row_jump_blocked=1,
        forced_selected=True,
        file_mapping_low_conf=True,
        row_abstain_skip=False,
    )

    assert out["mode"] == "conservative_apply"
    assert out["reason_code"] == "low_model_conf"
    assert "blank_conf_high" in out["reasons"]
    assert "jump_blocked" in out["reasons"]
    assert "plan_mismatch" in out["reasons"]
    assert "high_pitch_unstable" in out["reasons"]


def test_apply_zero_template_compute_policy_converts_review_required():
    logs = []
    row_apply = {
        "mode": "review_required",
        "reason_code": "inactive_candidate",
        "reasons": ["inactive_candidate"],
    }

    out = apply_zero_template_compute_policy(
        row_apply,
        line="a.wav=ka,0,0,0,0,0",
        use_template=True,
        is_zero_placeholder_line_fn=lambda _line: True,
        fname="a.wav",
        alias="ka",
        debug_logging=True,
        log_fn=logs.append,
    )

    assert out is row_apply
    assert row_apply["mode"] == "conservative_apply"
    assert row_apply["reason_code"] == "inactive_candidate_zero_template_compute"
    assert row_apply["reasons"] == ["inactive_candidate", "zero_template_compute"]
    assert logs and "0값 템플릿 행" in logs[0]


def test_record_row_apply_decision_updates_counts_and_payload():
    counts = {"full_apply": 0, "conservative_apply": 0}
    decisions = []
    row_apply = {
        "mode": "conservative_apply",
        "reason_code": "low_model_conf",
        "reasons": ["low_model_conf"],
    }

    mode, reason = record_row_apply_decision(
        row_apply_mode_counts=counts,
        row_apply_decisions=decisions,
        row_apply=row_apply,
        fname="a.wav",
        alias="ka",
        alias_type="cv",
        file_format="cvvc",
        selected_w_idx=2,
        row_mapping_confidence=0.51,
        row_blank_confidence=0.7,
        row_blank_floor=0.64,
        routing_profile="acoustic_safe",
        pitch_zone="",
    )

    assert mode == "conservative_apply"
    assert reason == "low_model_conf"
    assert counts["conservative_apply"] == 1
    assert decisions[0]["alias"] == "ka"
    assert decisions[0]["selected_w_idx"] == 2
    assert decisions[0]["blank_confidence"] == 0.7


def test_build_review_required_unset_returns_reason_and_meta():
    reason, meta = build_review_required_unset(
        row_apply={"mode": "review_required", "reason_code": "inactive_candidate"},
        row_abstain={"reason": "fallback_reason", "diag_hint": "candidate inactive"},
        routing_profile="acoustic_safe",
    )

    assert reason == "inactive_candidate"
    assert meta == {
        "diag_hint": "candidate inactive",
        "apply_mode": "review_required",
        "routing_profile": "acoustic_safe",
    }
