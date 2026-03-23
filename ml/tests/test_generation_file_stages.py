from types import SimpleNamespace

from core.generation.file_stages import (
    append_preserved_lines,
    handle_mapping_abstain_fallback,
    handle_mapping_failure_fallback,
    handle_ja_file_context_status,
    handle_kr_loop_prep_status,
    resolve_mapping_line_alias,
    resolve_mapping_failure_reason,
)


def _suffix(line: str, suffix: str) -> str:
    if not suffix:
        return line
    if "=" not in line:
        return line
    left, right = line.split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        return f"{left}={alias}_{suffix},{rest}"
    return f"{left}={right}_{suffix}"


def test_append_preserved_lines_applies_suffix():
    out = []
    append_preserved_lines(
        out,
        ["a.wav=ka,0,0,0,0,0"],
        alias_suffix="C4",
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert out == ["a.wav=ka_C4,0,0,0,0,0"]


def test_handle_ja_file_context_status_textgrid_missing_collects_meta():
    out = []
    logs = []
    unset = []
    file_ctx = SimpleNamespace(
        status="textgrid_missing",
        diagnostics={"lookup_norm": "a", "norm_candidates": ["b"], "candidate_paths": ["x", "y"]},
    )

    skipped = handle_ja_file_context_status(
        file_ctx=file_ctx,
        fname="a.wav",
        lines=["a.wav=ka,0,0,0,0,0"],
        final_lines=out,
        alias_suffix="",
        log_fn=logs.append,
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
        debug_reason_logging=True,
    )
    assert skipped is True
    assert len(out) == 1
    assert any("TextGrid 없음" in msg for msg in logs)
    assert unset and unset[0][0][0] == "textgrid_missing"


def test_handle_kr_loop_prep_status_empty_intervals():
    out = []
    unset = []
    skipped = handle_kr_loop_prep_status(
        loop_prep=SimpleNamespace(status="empty_intervals"),
        fname="a.wav",
        lines=["a.wav=ka,0,0,0,0,0"],
        final_lines=out,
        alias_suffix="",
        log_fn=lambda _msg: None,
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert skipped is True
    assert len(out) == 1
    assert unset and unset[0][0][0] == "mapping_failed_empty_intervals"


def test_resolve_mapping_failure_reason_prefers_spn_heavy():
    reason = resolve_mapping_failure_reason(
        low_quality_reasons=["spn_heavy", "insufficient_phones"],
        low_phone_quality=True,
        has_word_intervals=True,
        has_phone_intervals=True,
    )
    assert reason == "mapping_failed_spn_heavy"


def test_resolve_mapping_failure_reason_marks_empty_intervals():
    reason = resolve_mapping_failure_reason(
        low_quality_reasons=[],
        low_phone_quality=False,
        has_word_intervals=True,
        has_phone_intervals=False,
    )
    assert reason == "mapping_failed_empty_intervals"


def test_handle_mapping_failure_fallback_records_and_preserves():
    out = []
    logs = []
    unset = []
    lines = ["a.wav=ka,0,0,0,0,0"]
    handle_mapping_failure_fallback(
        fail_reason="mapping_failed",
        fname="a.wav",
        lines=lines,
        final_lines=out,
        alias_suffix="C4",
        log_fn=logs.append,
        failure_log_message="warn",
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        meta={"mapping_confidence": 0.5},
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert logs == ["warn"]
    assert unset and unset[0][0][0] == "mapping_failed"
    assert unset[0][0][1] == "a.wav"
    assert unset[0][1]["meta"]["mapping_confidence"] == 0.5
    assert out == ["a.wav=ka_C4,0,0,0,0,0"]


def test_handle_mapping_abstain_fallback_records_and_preserves():
    out = []
    logs = []
    unset = []
    lines = ["a.wav=ka,0,0,0,0,0"]
    handle_mapping_abstain_fallback(
        fname="a.wav",
        lines=lines,
        final_lines=out,
        alias_suffix="C4",
        log_fn=logs.append,
        abstain_log_message="abstain",
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        meta={"mapping_tier": "low"},
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert logs == ["abstain"]
    assert unset and unset[0][0][0] == "mapping_v2_abstain"
    assert unset[0][0][1] == "a.wav"
    assert unset[0][1]["meta"]["mapping_tier"] == "low"
    assert out == ["a.wav=ka_C4,0,0,0,0,0"]


def test_resolve_mapping_line_alias_returns_alias():
    out = []
    unset = []
    alias = resolve_mapping_line_alias(
        line="a.wav=ka,0,0,0,0,0",
        fname="a.wav",
        real_wav_name="a.wav",
        final_lines=out,
        alias_suffix="C4",
        record_unset_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert alias == "ka"
    assert out == []
    assert unset == []


def test_resolve_mapping_line_alias_handles_malformed_line():
    out = []
    unset = []
    alias = resolve_mapping_line_alias(
        line="invalid_line",
        fname="a.wav",
        real_wav_name="a.wav",
        final_lines=out,
        alias_suffix="C4",
        record_unset_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert alias is None
    assert unset and unset[0][0] == ("malformed_line", "a.wav", "invalid_line")
    assert out == ["invalid_line"]


def test_resolve_mapping_line_alias_handles_empty_alias():
    out = []
    unset = []
    alias = resolve_mapping_line_alias(
        line="old.wav=,0,0,0,0,0",
        fname="old.wav",
        real_wav_name="new.wav",
        final_lines=out,
        alias_suffix="C4",
        record_unset_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
    )
    assert alias is None
    assert unset and unset[0][0] == ("empty_alias", "old.wav", "old.wav=,0,0,0,0,0")
    assert out == ["new.wav=_C4,0,0,0,0,0"]
