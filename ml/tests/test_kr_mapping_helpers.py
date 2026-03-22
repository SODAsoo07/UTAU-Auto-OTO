from types import SimpleNamespace

from core import oto_generator as kr_gen
from core.oto_generator import (
    _build_kr_plan_context,
    _build_kr_words_syllables,
    _compute_kr_runtime_mapping_state,
    _log_kr_mapping_selection,
    _recompute_kr_runtime_state,
    _select_kr_syllable_source_state,
)


def test_build_kr_words_syllables_basic():
    wd = [SimpleNamespace(mark="ga", minTime=0.0, maxTime=1.0)]
    ph = [
        SimpleNamespace(mark="g", minTime=0.0, maxTime=0.4),
        SimpleNamespace(mark="a", minTime=0.4, maxTime=0.9),
    ]
    out = _build_kr_words_syllables(
        wd,
        ph,
        force_words_phone_fill=False,
        decompose_hangul_to_roman_fn=lambda _ch: ["ga"],
    )
    assert len(out) == 1
    assert out[0]["word"] == "ga"
    assert out[0]["roman"] == "gaga"
    assert out[0]["roman_cv"] == "ga"


def test_select_kr_syllable_source_state_words_only():
    wd = [SimpleNamespace(mark="ga", minTime=0.0, maxTime=1.0)]
    ph = [
        SimpleNamespace(mark="g", minTime=0.0, maxTime=0.4),
        SimpleNamespace(mark="a", minTime=0.4, maxTime=0.9),
    ]
    state = _select_kr_syllable_source_state(
        file_format="cvvc",
        prefer_filename_sequence=False,
        low_phone_quality=False,
        wd_intervals=wd,
        ph_intervals=ph,
        targets_for_build=[],
        force_words_phone_fill=False,
        decompose_hangul_to_roman_fn=lambda _ch: ["ga"],
        mel_ctx_for_file=None,
        textgrid_trust_score=0.5,
    )
    assert state["used_words_based"] is True
    assert state["used_alias_based"] is False
    assert state["mapping_reason_code"] == "words_keep"
    assert len(state["syllables_info"]) == 1
    assert 0.0 <= state["alignment_weight"] <= 1.0


def test_log_kr_mapping_selection_emits_lines():
    logs = []
    _log_kr_mapping_selection(
        log_fn=logs.append,
        fname="a.wav",
        mapping_reason_code="alias_phone_minimal",
        base_score=10.0,
        alt_score=20.0,
        words_glide_mismatch_ratio=0.0,
        low_quality_reasons=[],
        alias_syllables_present=True,
        targets_for_build=["ga"],
        textgrid_trust_tier="mid",
        textgrid_trust_score=0.5,
        alignment_weight=0.4,
        blank_conf_mean=0.1,
        anchor_lock_lite=True,
    )
    assert any("phones" in line for line in logs)
    assert any("alignment_weight" in line for line in logs)


def test_build_kr_plan_context_fallback(monkeypatch):
    monkeypatch.setattr(
        kr_gen,
        "_build_kr_cv_anchor_plan_v2",
        lambda expected_tokens, _infos, use_mel=False: {
            "indices": list(range(len(expected_tokens))),
            "meta": {"margin": 7.2, "use_mel": bool(use_mel)},
            "source": "fallback",
        },
    )
    monkeypatch.setattr(kr_gen, "build_adjacent_anchor_graph", lambda indices: {"count": len(indices or [])})
    monkeypatch.setattr(
        kr_gen,
        "resolve_plan_policy",
        lambda **kwargs: {
            "coverage": 1.0,
            "margin": float((kwargs.get("plan_meta") or {}).get("margin", 0.0)),
            "prefer_sequence": bool(kwargs.get("prefer_sequence")),
        },
    )

    out = _build_kr_plan_context(
        plan_candidate_source=["ga", "na"],
        syllables_info=[{"word": "ga"}, {"word": "na"}],
        sinsy_label_entries=[],
        file_format="cvvc",
        textgrid_trust_tier="high",
        low_phone_quality=False,
        blank_conf_mean=0.12,
        alignment_weight=0.8,
        mel_ctx_for_file={"energy": [0.1]},
        prefer_filename_sequence=True,
    )
    assert out["cv_plan"]["indices"] == [0, 1]
    assert out["anchor_graph"]["count"] == 2
    assert out["plan_policy"]["prefer_sequence"] is True


def test_compute_kr_runtime_mapping_state_applies_blank_gate(monkeypatch):
    logs = []
    monkeypatch.setattr(kr_gen, "_env_float", lambda _name, default=0.0: float(default))
    out = _compute_kr_runtime_mapping_state(
        runtime_policy={
            "mapping_confidence": 0.66,
            "mapping_tier": "mid",
            "is_low_conf": False,
            "file_conf_floor": 0.65,
            "row_conf_floor": 0.60,
            "row_margin_floor": 6.0,
            "strict_mode": True,
            "low_conf_reasons": [],
        },
        mapping_confidence_base=0.7,
        mapping_margin=8.0,
        blank_conf_mean=0.56,
        syllable_blank_confidences=[0.7],
        file_mapping_conf_th=0.62,
        file_format="cvvc",
        kr_mapping_debug_reason_logging=True,
        fname="kr.wav",
        mapping_reason_code="words_keep",
        log_fn=logs.append,
    )
    assert out["mapping_confidence_base"] < 0.66
    assert out["file_mapping_low_conf"] is True
    assert out["row_blank_floor"] == 0.64
    assert "blank_confidence_high" in out["low_conf_reasons"]
    assert any("reason=words_keep" in line for line in logs)


def test_recompute_kr_runtime_state(monkeypatch):
    monkeypatch.setattr(kr_gen, "_estimate_kr_mapping_confidence", lambda *_args, **_kwargs: (0.71, 8.0))
    monkeypatch.setattr(
        kr_gen,
        "recompute_common_plan_runtime_state",
        lambda **_kwargs: {
            "cv_plan": {"indices": [0], "meta": {"margin": 7.2}, "source": "fallback"},
            "planned_indices": [0],
            "anchor_graph": {"count": 1},
            "plan_policy": {"coverage": 1.0},
            "runtime_policy": {"mapping_confidence": 0.68, "mapping_tier": "mid", "file_conf_floor": 0.64},
            "mapping_confidence": 0.68,
            "mapping_margin": 8.0,
            "mapping_tier": "mid",
        },
    )
    monkeypatch.setattr(kr_gen, "log_sinsy_plan_guard", lambda **_kwargs: None)
    monkeypatch.setattr(
        kr_gen,
        "_compute_kr_runtime_mapping_state",
        lambda **_kwargs: {
            "runtime_policy": {"mapping_tier": "mid"},
            "mapping_confidence_base": 0.63,
            "mapping_margin": 8.0,
            "file_mapping_low_conf": True,
            "file_conf_floor": 0.64,
            "row_conf_floor": 0.60,
            "row_margin_floor": 6.0,
            "row_blank_floor": 0.64,
            "low_conf_reasons": ["conf_below_floor"],
        },
    )

    out = _recompute_kr_runtime_state(
        alignment_ingest={},
        file_mapping_conf_th=0.62,
        file_format="cvvc",
        phone_quality={"expected_syllables": 1},
        base_score=30.0,
        alt_score=12.0,
        used_words_based=True,
        used_alias_based=False,
        mapping_reason_code="words_keep",
        kr_mapping_debug_reason_logging=True,
        fname="kr.wav",
        blank_conf_mean=0.20,
        syllable_blank_confidences=[0.2],
        plan_candidate_source=["ga"],
        sinsy_label_entries=[],
        syllables_info=[{"word": "ga"}],
        textgrid_trust_tier="high",
        low_phone_quality=False,
        alignment_weight=0.8,
        mel_ctx_for_file={"energy": [0.1]},
        prefer_filename_sequence=True,
        log_fn=lambda _msg: None,
    )
    assert out["mapping_confidence_base"] == 0.63
    assert out["mapping_margin"] == 8.0
    assert out["mapping_tier"] == "mid"
    assert out["cv_plan"]["indices"] == [0]
    assert out["planned_indices"] == [0]
    assert out["kr_cv_plan"]["indices"] == [0]
    assert out["kr_planned_cv_indices"] == [0]
    assert out["file_mapping_low_conf"] is True
    assert out["sequence_lock_applied"] is False
