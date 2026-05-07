from types import SimpleNamespace

from core.generation.kr_post_timing import run_kr_general_post_timing_adjustments


def _base_callbacks(events):
    def debug_trace(_log_fn, stage, _fname, _alias, _before, _after, extra=""):
        events.append(("trace", stage, extra))

    def post_pipeline(offset, consonant, cutoff, pre, ovl, **kwargs):
        events.append(("post", kwargs["current_w_idx"], kwargs["alias_type"]))
        return offset + 1, consonant + 1, cutoff - 1, pre + 1, ovl + 1, 2.0, 3.0, 4.0

    def naturalness(offset, consonant, cutoff, pre, ovl, **_kwargs):
        events.append(("naturalness",))
        return offset + 4, consonant + 4, cutoff - 4, pre + 4, ovl + 4, {"shift": 1.2}

    def clamp(*, offset, consonant, cutoff, pre, ovl, **_kwargs):
        events.append(("clamp",))
        return offset + 5, consonant + 5, cutoff - 5, pre + 5, ovl + 5

    return {
        "env_bool_fn": lambda _name, default=False: bool(default),
        "resolve_mel_onset_weight_fn": lambda *_args, **_kwargs: 0.5,
        "estimate_mel_voiced_onset_fn": lambda _ctx, _pre_abs: 35.0,
        "apply_mel_voiced_onset_pre_shift_fn": (
            lambda offset, consonant, cutoff, pre, ovl, _mel, **_kwargs: (
                offset + 10,
                consonant + 10,
                cutoff - 10,
                pre + 10,
                ovl + 10,
                10.0,
            )
        ),
        "debug_offset_trace_fn": debug_trace,
        "apply_post_timing_pipeline_fn": post_pipeline,
        "extract_alias_onset_fn": lambda alias: alias[0],
        "ensure_cv_min_vowel_coverage_fn": (
            lambda offset, consonant, cutoff, pre, *_args, **_kwargs: (
                offset + 2,
                consonant + 2,
                cutoff - 2,
                pre + 2,
                True,
            )
        ),
        "guard_cv_focus_window_fn": (
            lambda offset, consonant, cutoff, pre, ovl, *_args, **_kwargs: (
                offset + 3,
                consonant + 3,
                cutoff - 3,
                pre + 3,
                ovl + 3,
                1.0,
                1.0,
            )
        ),
        "apply_mel_naturalness_adjustments_fn": naturalness,
        "apply_conservative_delta_clamp_fn": clamp,
    }


def test_run_kr_general_post_timing_adjustments_applies_cv_sequence():
    events = []
    logs = []
    out = run_kr_general_post_timing_adjustments(
        offset=10,
        consonant=30,
        cutoff=-100,
        pre=20,
        ovl=12,
        alias="ka",
        alias_type="cv",
        file_format="cvvc",
        is_vc=False,
        selected_w_idx=0,
        current_w_idx=0,
        curr_phones=[SimpleNamespace(mark="k")],
        c_start=0,
        c_end=40,
        n_start=40,
        n_end=160,
        row_mapping_confidence=0.5,
        row_apply_mode="conservative_apply",
        row_apply_reason_code="low_model_conf",
        delta_clamp_scale=0.75,
        pitch_zone="",
        base_shape={"offset": 9.0},
        mel_ctx_for_file={"energy": [1.0]},
        syllables_info=[{"mel_onset_distance_ms": 20.0, "mel_onset_energy": 0.5}],
        ph_intervals=[],
        is_vc_plosive_coda=False,
        kr_post_ctx={},
        alignment_weight=0.5,
        textgrid_trust_tier="mid",
        textgrid_trust_score=0.6,
        mel_reliability_score=0.7,
        mel_reliability_floor=0.4,
        fname="a.wav",
        debug_logging=True,
        log_fn=logs.append,
        **_base_callbacks(events),
    )

    assert out["offset"] == 35.0
    assert out["consonant"] == 55.0
    assert out["cutoff"] == -125.0
    assert out["pre"] == 45.0
    assert out["ovl"] == 15.0
    assert out["soft_off_shift"] == 2.0
    assert out["soft_cut_shift"] == 3.0
    assert out["cutoff_reduced"] == 4.0
    assert out["row_mel_voiced_onset_ms"] == 35.0
    assert ("post", 0, "cv") in events
    assert ("naturalness",) in events
    assert ("clamp",) in events
    assert any("CV 포커스 가드 적용" in line for line in logs)
    assert any("mode=conservative_apply" in line for line in logs)


def test_run_kr_general_post_timing_adjustments_skips_naturalness_for_template_preserve():
    events = []
    out = run_kr_general_post_timing_adjustments(
        offset=10,
        consonant=30,
        cutoff=-100,
        pre=20,
        ovl=12,
        alias="ng a",
        alias_type="vc",
        file_format="cvvc",
        is_vc=True,
        selected_w_idx=1,
        current_w_idx=0,
        curr_phones=[SimpleNamespace(mark="ng")],
        c_start=0,
        c_end=40,
        n_start=40,
        n_end=160,
        row_mapping_confidence=0.8,
        row_apply_mode="template_preserve",
        row_apply_reason_code="",
        delta_clamp_scale=0.75,
        pitch_zone="",
        base_shape={},
        mel_ctx_for_file=None,
        syllables_info=[{}, {}],
        ph_intervals=[],
        is_vc_plosive_coda=False,
        kr_post_ctx={},
        alignment_weight=0.8,
        textgrid_trust_tier="high",
        textgrid_trust_score=0.9,
        mel_reliability_score=0.9,
        mel_reliability_floor=0.4,
        fname="a.wav",
        debug_logging=False,
        log_fn=lambda _msg: None,
        **_base_callbacks(events),
    )

    assert out["offset"] == 16.0
    assert out["consonant"] == 36.0
    assert out["cutoff"] == -106.0
    assert out["pre"] == 26.0
    assert out["ovl"] == 18.0
    assert out["row_mel_voiced_onset_ms"] is None
    assert ("post", 0, "vc") in events
    assert ("naturalness",) not in events
    assert ("clamp",) in events
