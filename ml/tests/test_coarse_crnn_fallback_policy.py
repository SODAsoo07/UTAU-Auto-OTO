from __future__ import annotations


def test_fallback_decision_uses_multiple_signals():
    from core.coarse_crnn.oto_fallback_policy import FallbackPolicy, decide_fallback, signals_from_prediction

    signals = signals_from_prediction(
        confidence=0.24,
        heatmap_confidence={"offset": 0.18, "overlap": 0.22, "preutterance": 0.19, "consonant": 0.26, "cutoff": 0.21},
        alias_type="vc",
        alias_role="vc",
        transition_type="multi",
        row_ratio=0.85,
        is_special=False,
    )
    policy = FallbackPolicy(enabled=True, score_threshold=0.60)
    decision = decide_fallback(signals, policy)

    assert decision.apply
    assert decision.score >= 0.60
    assert "low_conf" in decision.reasons
    assert "low_anchor_conf" in decision.reasons


def test_blend_with_source_keeps_cutoff_negative_and_overlap_bounded():
    from core.coarse_crnn.oto_fallback_policy import blend_with_source

    out = blend_with_source(
        {
            "offset": 20.0,
            "consonant": 140.0,
            "cutoff": -200.0,
            "preutterance": 90.0,
            "overlap": 110.0,
        },
        {
            "offset": 10.0,
            "consonant": 80.0,
            "cutoff": 150.0,
            "preutterance": 70.0,
            "overlap": 20.0,
        },
        source_blend=0.5,
    )
    assert out["cutoff"] <= 0.0
    assert out["overlap"] <= out["preutterance"]
    assert out["offset"] >= 0.0
    assert out["consonant"] >= 0.0
