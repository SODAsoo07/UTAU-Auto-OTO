from core.generation.kr_mapping_fallback import force_kr_runtime_abstain_to_sequence_lock
from core.generation.mapping_runtime import compute_runtime_low_conf_state
from core.generation.plan_runtime import build_common_plan_context, recompute_common_plan_runtime_state
from core.kr_candidate_selection_v2 import select_kr_syllable_source
from core.oto_generator import _estimate_kr_mapping_confidence, resolve_file_routing_profile


def test_select_kr_syllable_source_words_only():
    words = [{"word": "ga", "phones": [object()], "roman_cv": "ga"}]

    state = select_kr_syllable_source(
        file_format="cvvc",
        prefer_filename_sequence=False,
        low_phone_quality=False,
        used_words_based=True,
        words_syllables=words,
        alias_syllables=None,
        targets_for_build=[],
        score_mapping_fn=lambda *_args: 0.0,
        should_prefer_alias_fn=lambda *_args: False,
        compute_glide_mismatch_fn=lambda *_args: 0.0,
        is_order_locked_format_fn=lambda fmt: fmt in {"cvvc", "cvc", "vcv"},
    )

    assert state["syllables_info"] == words
    assert state["used_words_based"] is True
    assert state["used_alias_based"] is False
    assert state["mapping_reason_code"] == "words_keep"


def test_select_kr_syllable_source_falls_back_to_alias_for_empty_word_phones():
    words = [{"word": "ga", "phones": [], "roman_cv": "ga"}]
    alias = [{"word": "ga", "phones": [object()], "roman_cv": "ga"}]

    state = select_kr_syllable_source(
        file_format="cvvc",
        prefer_filename_sequence=False,
        low_phone_quality=True,
        used_words_based=True,
        words_syllables=words,
        alias_syllables=alias,
        targets_for_build=["ga"],
        score_mapping_fn=lambda infos, _targets: 10.0 * len(infos),
        should_prefer_alias_fn=lambda *_args: False,
        compute_glide_mismatch_fn=lambda *_args: 0.0,
        is_order_locked_format_fn=lambda fmt: fmt in {"cvvc", "cvc", "vcv"},
    )

    assert state["syllables_info"] == alias
    assert state["used_words_based"] is False
    assert state["used_alias_based"] is True
    assert state["mapping_reason_code"] == "alias_based_empty_words"


def test_build_common_plan_context_fallback():
    out = build_common_plan_context(
        planned_cv_source=["ga", "na"],
        syllables_info=[{"word": "ga"}, {"word": "na"}],
        sinsy_label_entries=[],
        format_type="cvvc",
        alignment_weight=0.8,
        prefer_sequence=True,
        normalize_expected_fn=lambda text: text,
        normalize_label_fn=lambda text: text,
        label_match_score_fn=lambda _target, _label: 1.0,
        should_enable_mel_plan_fn=lambda: False,
        build_cv_anchor_plan_fn=lambda tokens, _infos, **_kwargs: {
            "indices": list(range(len(tokens))),
            "meta": {"margin": 7.2},
            "source": "fallback",
        },
        build_sinsy_guided_anchor_plan_fn=lambda **_kwargs: {"indices": None, "meta": {}},
        build_adjacent_anchor_graph_fn=lambda indices: {"count": len(indices or [])},
        resolve_plan_policy_fn=lambda **kwargs: {
            "coverage": 1.0,
            "margin": float((kwargs.get("plan_meta") or {}).get("margin", 0.0)),
            "prefer_sequence": bool(kwargs.get("prefer_sequence")),
        },
    )

    assert out["cv_plan"]["indices"] == [0, 1]
    assert out["anchor_graph"]["count"] == 2
    assert out["plan_policy"]["prefer_sequence"] is True


def test_recompute_common_plan_runtime_state_uses_runtime_policy_result():
    out = recompute_common_plan_runtime_state(
        build_plan_context_fn=lambda **_kwargs: {
            "cv_plan": {"indices": [0], "meta": {"margin": 7.2}, "source": "fallback"},
            "planned_indices": [0],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 1.0},
        },
        plan_context_kwargs={},
        ingest_snapshot={},
        mapping_confidence=0.71,
        mapping_margin=8.0,
        conf_threshold=0.62,
        format_type="cvvc",
        score_a=30.0,
        score_b=12.0,
        sequence_lock_formats={"cvvc", "cvc", "vcv"},
        abstain_formats={"cvvc", "vcv", "cvc", "cv"},
        strict_formats={"cvvc", "vcv"},
        prefer_sequence=True,
        alignment_trust=0.8,
        resolve_runtime_mapping_policy_fn=lambda **_kwargs: {
            "mapping_confidence": 0.68,
            "mapping_tier": "mid",
            "force_sequence_lock": True,
        },
    )

    assert out["cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["runtime_policy"]["force_sequence_lock"] is True
    assert out["mapping_confidence"] == 0.68
    assert out["mapping_tier"] == "mid"


def test_compute_runtime_low_conf_state_and_routing_profile_blank_gate(monkeypatch):
    monkeypatch.delenv("UTOA_KR_CVVC_ROW_BLANK_FLOOR", raising=False)
    low_conf = compute_runtime_low_conf_state(
        runtime_policy={"low_conf_reasons": [], "is_low_conf": False},
        mapping_confidence=0.51,
        conf_floor=0.62,
        blank_confidence_mean=0.60,
        conf_below_reason="conf_below_floor",
        row_conf_floor_default=0.62,
    )
    routing = resolve_file_routing_profile(
        file_format="cvvc",
        pitch_zone="",
        note_hint_hz=0.0,
        f0_max_hz=0.0,
        textgrid_trust_score=0.42,
        textgrid_trust_tier="low",
        mapping_confidence_base=0.51,
        blank_conf_mean=0.60,
        mel_reliability_score=0.7,
        mel_reliability_floor=0.44,
        file_mapping_low_conf=bool(low_conf["file_low_conf"]),
        alignment_weight_base=0.36,
        mel_weight_mode="auto",
        file_mapping_conf_th=0.62,
    )

    assert low_conf["file_low_conf"] is True
    assert "blank_confidence_high" in low_conf["low_conf_reasons"]
    assert "conf_below_floor" in low_conf["low_conf_reasons"]
    assert routing["profile_code"] == "acoustic_safe"
    assert routing["row_blank_floor"] == 0.64


def test_estimate_kr_mapping_confidence_penalizes_spn_heavy():
    clean_conf, clean_margin = _estimate_kr_mapping_confidence(
        {
            "spn_ratio_in_phone_tier": 0.0,
            "phones_vs_expected_syllables_ratio": 1.0,
            "known_vowel_phone_count": 2,
            "expected_syllables": 2,
            "low_confidence_reasons": [],
        },
        words_score=60.0,
        alias_score=45.0,
        used_words_based=True,
        used_alias_based=False,
    )
    noisy_conf, noisy_margin = _estimate_kr_mapping_confidence(
        {
            "spn_ratio_in_phone_tier": 0.8,
            "phones_vs_expected_syllables_ratio": 0.3,
            "known_vowel_phone_count": 0,
            "expected_syllables": 2,
            "low_confidence_reasons": ["spn_heavy", "insufficient_phones"],
        },
        words_score=20.0,
        alias_score=35.0,
        used_words_based=False,
        used_alias_based=True,
    )

    assert clean_conf > noisy_conf
    assert clean_margin == 15.0
    assert noisy_margin == -15.0


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
