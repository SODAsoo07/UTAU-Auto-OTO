from core import ja_oto_generator as ja_gen
from core.ja_oto_generator import (
    _apply_ja_force_sequence_lock_fallback,
    _build_ja_plan_context,
    _derive_ja_selected_candidate_state,
    _log_ja_mapping_selection,
    _recompute_ja_runtime_state,
    _select_ja_mapping_candidate_with_guards,
    _should_enable_ja_cvvc_mel_plan,
)


def test_select_ja_mapping_candidate_with_guards_promotes_alias():
    mapping_candidates = [
        {
            "name": "filename_token",
            "objective": 10.0,
            "score": 55.0,
            "mean_syll_conf": 0.75,
            "use_filename": True,
            "use_alias": False,
            "lock_order": False,
            "order_preserving": True,
            "infos": [{"word": "a", "phones": [object()]}],
            "syllable_confidences": [0.8],
            "words_align_score": 80.0,
        },
        {
            "name": "alias_phone_fallback",
            "objective": 22.0,
            "score": 82.0,
            "mean_syll_conf": 0.92,
            "use_filename": False,
            "use_alias": True,
            "lock_order": False,
            "order_preserving": False,
            "infos": [{"word": "a", "phones": [object()]}],
            "syllable_confidences": [0.95],
            "words_align_score": 84.0,
        },
    ]
    selected, reason, _by_name, alias = _select_ja_mapping_candidate_with_guards(
        mapping_candidates=mapping_candidates,
        format_type="other",
        forced_words_mapping=True,
        phone_quality={"expected_syllables": 1, "known_vowel_phone_count": 1},
        words_tier_confidence=1.0,
        conf_th=0.1,
        cv_targets=["a"],
        low_phone_quality=False,
        debug_reason_logging=False,
        log_fn=lambda _msg: None,
        fname="x.wav",
    )
    assert selected is alias
    assert reason == "alias_recover"


def test_derive_ja_selected_candidate_state_empty():
    state = _derive_ja_selected_candidate_state(
        selected_candidate=None,
        format_type="cvvc",
        mel_ctx_for_file=None,
        textgrid_trust_score=0.5,
    )
    assert state["syllables_info"] == []
    assert state["base_score"] == -1.0
    assert state["filename_order_locked"] is False
    assert 0.0 <= state["alignment_weight"] <= 1.0


def test_should_enable_ja_cvvc_mel_plan_low_trust(monkeypatch):
    monkeypatch.setattr(ja_gen, "_env_bool", lambda _name, default=False: bool(default))
    enabled = _should_enable_ja_cvvc_mel_plan(
        format_type="cvvc",
        mel_ctx_for_file={"energy": [0.1]},
        syllables_info=[{"word": "ka"}],
        textgrid_trust_tier="low",
        low_phone_quality=False,
        blank_conf_mean=0.2,
        alignment_weight=0.8,
        syllable_confidence_by_idx=[0.9],
    )
    assert enabled is True


def test_build_ja_plan_context_fallback(monkeypatch):
    monkeypatch.setattr(ja_gen, "_should_enable_ja_cvvc_mel_plan", lambda **_kwargs: False)
    monkeypatch.setattr(
        ja_gen,
        "_build_ja_cv_anchor_plan_v2",
        lambda expected_tokens, _infos, use_mel=False: {
            "indices": list(range(len(expected_tokens))),
            "meta": {"margin": 8.0, "use_mel": bool(use_mel)},
            "source": "fallback",
        },
    )
    monkeypatch.setattr(ja_gen, "build_adjacent_anchor_graph", lambda indices: {"count": len(indices or [])})
    monkeypatch.setattr(
        ja_gen,
        "resolve_plan_policy",
        lambda **kwargs: {
            "coverage": 1.0,
            "margin": float((kwargs.get("plan_meta") or {}).get("margin", 0.0)),
            "prefer_sequence": bool(kwargs.get("prefer_sequence")),
        },
    )

    out = _build_ja_plan_context(
        planned_cv_source=["ka", "ra"],
        sinsy_label_entries=[],
        syllables_info=[{"word": "ka"}, {"word": "ra"}],
        format_type="cvvc",
        textgrid_trust_tier="high",
        low_phone_quality=False,
        blank_conf_mean=0.1,
        alignment_weight=0.9,
        syllable_confidence_by_idx=[0.9, 0.95],
        mel_ctx_for_file={"energy": [0.1]},
        prefer_filename_sequence=True,
    )

    assert out["cv_plan"]["indices"] == [0, 1]
    assert out["anchor_graph"]["count"] == 2
    assert out["plan_policy"]["prefer_sequence"] is True


def test_log_ja_mapping_selection_emits_alignment_line():
    logs = []
    _log_ja_mapping_selection(
        log_fn=logs.append,
        fname="sample.wav",
        selected_candidate={"name": "alias_phone_fallback"},
        mapping_reason_code="alias_recover",
        filename_syllable_count=2,
        cv_target_count=2,
        phone_interval_count=5,
        word_interval_count=2,
        spn_ratio=0.12,
        base_score=11.0,
        alt_score=19.0,
        alignment_weight=0.64,
        blank_conf_mean=0.15,
        anchor_lock_lite=False,
    )
    assert any("alignment_weight" in line for line in logs)


def test_apply_ja_force_sequence_lock_fallback(monkeypatch):
    logs = []
    monkeypatch.setattr(
        ja_gen,
        "_derive_ja_selected_candidate_state",
        lambda **_kwargs: {
            "syllables_info": [{"word": "ka"}],
            "used_filename_based": True,
            "used_alias_based": False,
            "filename_order_locked": False,
            "base_score": 35.0,
            "syllable_confidence_by_idx": [0.7],
            "blank_conf_mean": 0.2,
            "alignment_weight": 0.7,
            "anchor_lock_lite": False,
        },
    )
    monkeypatch.setattr(ja_gen, "_estimate_ja_mapping_confidence", lambda *_args, **_kwargs: (0.5, 9.0))
    monkeypatch.setattr(
        ja_gen,
        "_build_ja_plan_context",
        lambda **_kwargs: {
            "cv_plan": {"indices": [0], "meta": {"margin": 8.0}, "source": "fallback"},
            "planned_indices": [0],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 1.0},
        },
    )
    monkeypatch.setattr(
        ja_gen,
        "resolve_runtime_mapping_policy",
        lambda **_kwargs: {
            "mapping_confidence": 0.42,
            "mapping_tier": "low",
            "force_sequence_lock": True,
        },
    )

    out = _apply_ja_force_sequence_lock_fallback(
        runtime_policy={"force_sequence_lock": True},
        filename_order_locked=False,
        candidate_by_name={
            "filename_linear_fallback": {
                "name": "filename_linear_fallback",
                "score": 35.0,
                "words_align_score": 55.0,
            }
        },
        selected_candidate={"name": "words_tier"},
        format_type="cvvc",
        mel_ctx_for_file={"energy": [0.1]},
        textgrid_trust_score=0.6,
        phone_quality={"expected_syllables": 1},
        base_score=20.0,
        alt_score=10.0,
        forced_words_mapping=False,
        words_tier_confidence=0.8,
        planned_cv_source=["ka"],
        sinsy_label_entries=[],
        textgrid_trust_tier="low",
        low_phone_quality=False,
        alignment_ingest={},
        conf_th=0.6,
        fname="ja.wav",
        log_fn=logs.append,
    )

    assert out is not None
    assert out["filename_order_locked"] is True
    assert out["mapping_reason_code"] == "filename_linear_low_conf"
    assert out["mapping_tier"] == "low"
    assert out["selected_candidate"]["name"] == "filename_linear_fallback"
    assert out["sequence_lock_applied"] is True
    assert out["sequence_lock_reason"] == "filename_low_conf_fallback"
    assert any("trust=" in line for line in logs)


def test_recompute_ja_runtime_state(monkeypatch):
    monkeypatch.setattr(ja_gen, "_estimate_ja_mapping_confidence", lambda *_args, **_kwargs: (0.61, 7.0))
    monkeypatch.setattr(
        ja_gen,
        "recompute_common_plan_runtime_state",
        lambda **_kwargs: {
            "cv_plan": {"indices": [0], "meta": {"margin": 8.0}, "source": "fallback"},
            "planned_indices": [0],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 1.0},
            "runtime_policy": {"mapping_confidence": 0.58, "mapping_tier": "mid"},
            "mapping_confidence": 0.58,
            "mapping_margin": 7.0,
            "mapping_tier": "mid",
        },
    )

    out = _recompute_ja_runtime_state(
        alignment_ingest={},
        conf_th=0.62,
        format_type="cvvc",
        base_score=30.0,
        alt_score=10.0,
        phone_quality={"expected_syllables": 1},
        used_filename_based=True,
        used_alias_based=False,
        forced_words_mapping=False,
        syllable_confidence_by_idx=[0.8],
        words_align_score=77.0,
        words_tier_confidence=0.9,
        planned_cv_source=["ka"],
        sinsy_label_entries=[],
        syllables_info=[{"word": "ka"}],
        textgrid_trust_tier="high",
        low_phone_quality=False,
        blank_conf_mean=0.1,
        alignment_weight=0.8,
        mel_ctx_for_file={"energy": [0.1]},
        prefer_filename_sequence=True,
    )
    assert out["mapping_confidence_base"] == 0.58
    assert out["mapping_margin"] == 7.0
    assert out["mapping_tier"] == "mid"
    assert out["cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["ja_cv_plan"]["indices"] == [0]
    assert out["sequence_lock_applied"] is False
    assert out["sequence_lock_reason"] == ""
