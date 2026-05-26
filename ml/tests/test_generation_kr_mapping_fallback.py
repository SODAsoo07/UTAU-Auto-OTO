from core.generation.kr_mapping_fallback import (
    force_kr_runtime_abstain_to_sequence_lock,
    handle_kr_mapping_failure_with_guard,
    has_kr_syllable_mapping_failure,
)


def _suffix(line: str, suffix: str) -> str:
    if not suffix or "=" not in line:
        return line
    left, right = line.split("=", 1)
    alias, rest = right.split(",", 1)
    return f"{left}={alias}_{suffix},{rest}"


def test_has_kr_syllable_mapping_failure_detects_empty_phones():
    assert has_kr_syllable_mapping_failure([]) is True
    assert has_kr_syllable_mapping_failure([{"phones": []}]) is True
    assert has_kr_syllable_mapping_failure([{"phones": [object()]}]) is False


def test_handle_kr_mapping_failure_with_guard_records_preserves_then_rolls_back():
    final_lines = ["keep.wav=a,1,2,3,4,5"]
    errors = []
    blocked = []
    logs = []
    unset = []

    handled = handle_kr_mapping_failure_with_guard(
        syllables_info=[{"phones": []}],
        low_quality_reasons=["spn_heavy"],
        low_phone_quality=True,
        has_word_intervals=True,
        fname="bad.wav",
        lines=["bad.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="C4",
        log_fn=logs.append,
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
        use_template=False,
        errors=errors,
        placeholder_block_errors=blocked,
        spn_ratio=0.72,
        phone_quality={"spn_ratio_in_phone_tier": 0.72},
        force_words_phone_fill=False,
        mapping_confidence=0.31,
        mapping_reason_code="alias_based_empty_words",
    )

    assert handled is True
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]
    assert unset and unset[0][0][0] == "mapping_failed_spn_heavy"
    assert unset[0][1]["meta"]["diag_hint"] == "spn_ratio=0.72; conf=0.31"
    assert unset[0][1]["meta"]["mapping_reason_code"] == "alias_based_empty_words"
    assert errors == blocked
    assert any("음절 경계 해석 실패" in line for line in logs)
    assert any("TextGrid/tier/interval/mapping" in line for line in logs)


def test_handle_kr_mapping_failure_with_guard_ignores_valid_syllables():
    final_lines = ["keep.wav=a,1,2,3,4,5"]
    handled = handle_kr_mapping_failure_with_guard(
        syllables_info=[{"phones": [object()]}],
        low_quality_reasons=[],
        low_phone_quality=False,
        has_word_intervals=True,
        fname="ok.wav",
        lines=["ok.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="",
        log_fn=lambda _msg: None,
        record_unset_lines_fn=lambda *args, **kwargs: None,
        apply_suffix_to_oto_line_fn=_suffix,
        use_template=False,
        errors=[],
        placeholder_block_errors=[],
        spn_ratio=0.0,
        phone_quality={},
        force_words_phone_fill=False,
        mapping_confidence=0.8,
        mapping_reason_code="words_keep_high_conf",
    )

    assert handled is False
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]


def test_force_kr_runtime_abstain_to_sequence_lock_mutates_policy():
    policy = {"should_abstain": True, "low_conf_reasons": ["base"]}
    logs = []

    handled = force_kr_runtime_abstain_to_sequence_lock(
        runtime_policy=policy,
        fname="low.wav",
        textgrid_trust_score=0.42,
        alignment_weight=0.36,
        plan_policy={"coverage": 0.51, "margin": 1.25},
        log_fn=logs.append,
    )

    assert handled is True
    assert policy["should_abstain"] is False
    assert policy["force_sequence_lock"] is True
    assert policy["low_conf_force_auto"] is True
    assert policy["low_conf_reasons"] == ["base", "planner_abstain_forced_auto"]
    assert logs and "순서 잠금 보수 자동설정" in logs[0]
