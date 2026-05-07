from core.generation.kr_finish_guards import maybe_block_kr_generation_output


def test_maybe_block_kr_generation_output_blocks_placeholder_errors():
    logs = []
    errors = []
    report = {}
    finalized = []
    unset_logged = []

    blocked = maybe_block_kr_generation_output(
        placeholder_block_errors=[{"file": "a.wav"}],
        final_lines=[],
        auto_gen_format="cvvc",
        runtime_report=report,
        errors=errors,
        log_fn=logs.append,
        finish_context="finish",
        log_unset_summary_fn=lambda: unset_logged.append(True),
        build_offset_zero_collapse_error_fn=lambda *_args, **_kwargs: "",
        finalize_generator_finish_fn=finalized.append,
    )

    assert blocked is True
    assert finalized == ["finish"]
    assert unset_logged == [True]
    assert errors == [
        "[ERROR] 자동 설정을 완료할 수 없는 파일이 있어 OTO 저장을 중단했습니다 "
        "(blocked_files=1, format=cvvc)."
    ]
    assert report == {
        "auto_placeholder_block_errors": [{"file": "a.wav"}],
        "auto_generation_block_errors": [{"file": "a.wav"}],
    }
    assert logs == errors


def test_maybe_block_kr_generation_output_blocks_zero_collapse():
    logs = []
    errors = []
    report = {}
    finalized = []
    unset_logged = []

    blocked = maybe_block_kr_generation_output(
        placeholder_block_errors=[],
        final_lines=["a.wav=a,0,0,0,0,0"],
        auto_gen_format="cv",
        runtime_report=report,
        errors=errors,
        log_fn=logs.append,
        finish_context={"finish": True},
        log_unset_summary_fn=lambda: unset_logged.append(True),
        build_offset_zero_collapse_error_fn=lambda *_args, **_kwargs: "zero collapse",
        finalize_generator_finish_fn=finalized.append,
    )

    assert blocked is True
    assert finalized == [{"finish": True}]
    assert unset_logged == [True]
    assert errors == ["zero collapse"]
    assert report == {"offset_zero_collapse_error": "zero collapse"}
    assert logs == ["zero collapse"]


def test_maybe_block_kr_generation_output_allows_clean_output():
    errors = []
    blocked = maybe_block_kr_generation_output(
        placeholder_block_errors=[],
        final_lines=["a.wav=a,1,2,3,4,5"],
        auto_gen_format="cv",
        runtime_report={},
        errors=errors,
        log_fn=lambda _msg: None,
        finish_context=None,
        log_unset_summary_fn=lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
        build_offset_zero_collapse_error_fn=lambda *_args, **_kwargs: "",
        finalize_generator_finish_fn=lambda _context: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    assert blocked is False
    assert errors == []
