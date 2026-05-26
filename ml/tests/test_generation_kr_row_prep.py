from types import SimpleNamespace

from core.generation.kr_row_prep import (
    prepare_kr_alias_row_start,
    resolve_kr_row_jump_limits,
    try_append_kr_br_alias_row,
)


def test_prepare_kr_alias_row_start_preserves_malformed_and_empty_alias():
    final_lines = []
    unset = []

    malformed = prepare_kr_alias_row_start(
        line="bad row",
        fname="a.wav",
        real_wav_name="a.wav",
        alias_suffix="_s",
        final_lines=final_lines,
        mapping_confidence_base=0.8,
        record_unset_fn=lambda *args, **_kwargs: unset.append(args),
        apply_suffix_to_oto_line_fn=lambda line, suffix: f"{line}{suffix}",
        extract_base_timing_shape_fn=lambda _line: {"shape": 1},
        classify_alias_fn=lambda _alias: "cv",
    )
    empty = prepare_kr_alias_row_start(
        line="orig.wav=,1,2,3,4,5",
        fname="a.wav",
        real_wav_name="real.wav",
        alias_suffix="_s",
        final_lines=final_lines,
        mapping_confidence_base=0.8,
        record_unset_fn=lambda *args, **_kwargs: unset.append(args),
        apply_suffix_to_oto_line_fn=lambda line, suffix: f"{line}{suffix}",
        extract_base_timing_shape_fn=lambda _line: {"shape": 1},
        classify_alias_fn=lambda _alias: "cv",
    )

    assert malformed["should_continue"] is True
    assert empty["should_continue"] is True
    assert unset[0][:3] == ("malformed_line", "a.wav", "bad row")
    assert unset[1][:3] == ("empty_alias", "a.wav", "orig.wav=,1,2,3,4,5")
    assert final_lines == ["bad row_s", "real.wav=,1,2,3,4,5_s"]


def test_prepare_kr_alias_row_start_returns_alias_state():
    out = prepare_kr_alias_row_start(
        line="a.wav=ka,1,2,3,4,5",
        fname="a.wav",
        real_wav_name="a.wav",
        alias_suffix="",
        final_lines=[],
        mapping_confidence_base=0.73,
        record_unset_fn=lambda *_args, **_kwargs: None,
        apply_suffix_to_oto_line_fn=lambda line, _suffix: line,
        extract_base_timing_shape_fn=lambda line: {"line": line},
        classify_alias_fn=lambda alias: "cv" if alias == "ka" else "unknown",
    )

    assert out["should_continue"] is False
    assert out["alias"] == "ka"
    assert out["alias_type"] == "cv"
    assert out["row_mapping_confidence"] == 0.73
    assert out["row_apply_mode"] == "full_apply"


def test_resolve_kr_row_jump_limits_applies_low_conf_and_alias_rules():
    cv_head = resolve_kr_row_jump_limits(
        alias_type="cv_head",
        mapping_only_enabled=False,
        order_locked_format=True,
        textgrid_trust_tier="low",
        file_mapping_low_conf=True,
        max_index_jump_default=2,
        max_index_jump_high_conf=4,
    )
    vv = resolve_kr_row_jump_limits(
        alias_type="vv",
        mapping_only_enabled=False,
        order_locked_format=False,
        textgrid_trust_tier="high",
        file_mapping_low_conf=False,
        max_index_jump_default=1,
        max_index_jump_high_conf=2,
    )

    assert cv_head == {"row_jump_default": 0, "row_jump_high_conf": 0}
    assert vv == {"row_jump_default": 2, "row_jump_high_conf": 3}


def test_try_append_kr_br_alias_row_uses_phone_span():
    final_lines = []
    calls = []
    intervals = [
        SimpleNamespace(minTime=0.2, maxTime=0.3),
        SimpleNamespace(minTime=0.4, maxTime=0.7),
    ]

    handled = try_append_kr_br_alias_row(
        alias_type="br",
        ph_intervals=intervals,
        final_lines=final_lines,
        real_wav_name="a.wav",
        alias="br",
        generate_openutau=False,
        alias_suffix="_s",
        wav_duration_ms=1000.0,
        validate_fn=lambda offset, consonant, cutoff, pre, ovl: (offset, consonant, cutoff, pre, ovl),
        append_alias_rows_fn=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert handled is True
    args, kwargs = calls[0]
    assert args[0] is final_lines
    assert args[1] == "a.wav"
    assert args[2] == "br"
    assert args[3:8] == (170.0, 100.0, -425.0, 0.0, 0.0)
    assert kwargs["alias_suffix"] == "_s"
