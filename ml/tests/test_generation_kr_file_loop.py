from types import SimpleNamespace

from core.generation.kr_file_loop import (
    handle_kr_file_context_status_with_guard,
    handle_kr_loop_prep_status_with_guard,
    rollback_auto_placeholder_fallback,
    update_wav_name_map_from_file_context,
)


def _suffix(line: str, suffix: str) -> str:
    if not suffix or "=" not in line:
        return line
    left, right = line.split("=", 1)
    alias, rest = right.split(",", 1)
    return f"{left}={alias}_{suffix},{rest}"


def test_update_wav_name_map_from_file_context_tracks_real_name_alias():
    wav_name_map = {}
    output = update_wav_name_map_from_file_context(
        wav_name_map=wav_name_map,
        fname="alias.wav",
        file_ctx=SimpleNamespace(output_wav_name="real.wav", real_wav_name="source.wav"),
    )

    assert output == "real.wav"
    assert wav_name_map == {
        "alias.wav": "real.wav",
        "source.wav": "real.wav",
    }


def test_rollback_auto_placeholder_fallback_removes_preserved_lines_for_fatal_reason():
    final_lines = ["keep.wav=a,1,2,3,4,5", "bad.wav=ka_C4,10,20,-30,15,5"]
    errors = []
    blocked = []
    logs = []

    changed = rollback_auto_placeholder_fallback(
        final_lines=final_lines,
        errors=errors,
        placeholder_block_errors=blocked,
        prev_final_len=1,
        fname="bad.wav",
        lines=["bad.wav=ka,10,20,-30,15,5"],
        reason="textgrid_missing",
        use_template=False,
        log_fn=logs.append,
    )

    assert changed is True
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]
    assert errors == blocked == logs
    assert "TextGrid/tier/interval/mapping" in errors[0]


def test_handle_kr_file_context_status_with_guard_preserves_then_rolls_back_blocked_line():
    final_lines = ["keep.wav=a,1,2,3,4,5"]
    errors = []
    blocked = []
    logs = []
    unset = []
    file_ctx = SimpleNamespace(status="textgrid_missing")

    handled = handle_kr_file_context_status_with_guard(
        file_ctx=file_ctx,
        fname="bad.wav",
        lines=["bad.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="C4",
        log_fn=logs.append,
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
        prev_final_len=1,
        use_template=False,
        errors=errors,
        placeholder_block_errors=blocked,
    )

    assert handled is True
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]
    assert unset and unset[0][0][0] == "textgrid_missing"
    assert errors == blocked
    assert any("TextGrid를 찾지 못해" in line for line in logs)
    assert any("저장을 중단" in line for line in logs)


def test_handle_kr_file_context_status_with_guard_ignores_ready_context():
    final_lines = ["keep.wav=a,1,2,3,4,5"]
    handled = handle_kr_file_context_status_with_guard(
        file_ctx=SimpleNamespace(status=""),
        fname="ok.wav",
        lines=["ok.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="",
        log_fn=lambda _msg: None,
        record_unset_lines_fn=lambda *args, **kwargs: None,
        apply_suffix_to_oto_line_fn=_suffix,
        prev_final_len=1,
        use_template=False,
        errors=[],
        placeholder_block_errors=[],
    )

    assert handled is False
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]


def test_handle_kr_loop_prep_status_with_guard_rolls_back_empty_intervals():
    final_lines = ["keep.wav=a,1,2,3,4,5"]
    errors = []
    blocked = []
    logs = []
    unset = []

    handled = handle_kr_loop_prep_status_with_guard(
        loop_prep=SimpleNamespace(status="empty_intervals"),
        fname="bad.wav",
        lines=["bad.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="C4",
        log_fn=logs.append,
        record_unset_lines_fn=lambda *args, **kwargs: unset.append((args, kwargs)),
        apply_suffix_to_oto_line_fn=_suffix,
        prev_final_len=1,
        use_template=False,
        errors=errors,
        placeholder_block_errors=blocked,
    )

    assert handled is True
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]
    assert unset and unset[0][0][0] == "mapping_failed_empty_intervals"
    assert errors == blocked
    assert any("유효한 음소 구간" in line for line in logs)
    assert any("저장을 중단" in line for line in logs)


def test_handle_kr_loop_prep_status_with_guard_ignores_ready_status():
    final_lines = ["keep.wav=a,1,2,3,4,5"]

    handled = handle_kr_loop_prep_status_with_guard(
        loop_prep=SimpleNamespace(status=""),
        fname="ok.wav",
        lines=["ok.wav=ka,10,20,-30,15,5"],
        final_lines=final_lines,
        alias_suffix="",
        log_fn=lambda _msg: None,
        record_unset_lines_fn=lambda *args, **kwargs: None,
        apply_suffix_to_oto_line_fn=_suffix,
        prev_final_len=1,
        use_template=False,
        errors=[],
        placeholder_block_errors=[],
    )

    assert handled is False
    assert final_lines == ["keep.wav=a,1,2,3,4,5"]
