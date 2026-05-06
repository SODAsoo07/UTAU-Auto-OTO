from core.generation.placeholder_guards import (
    build_auto_placeholder_block_error,
    build_offset_zero_collapse_error,
    is_zero_placeholder_line,
    lines_are_zero_placeholders,
)


def test_is_zero_placeholder_line_detects_all_zero_params():
    assert is_zero_placeholder_line("a.wav=ka,0,0,0,0,0") is True
    assert is_zero_placeholder_line("a.wav=ka,0,0,0.01,0,0") is False
    assert is_zero_placeholder_line("malformed") is False


def test_lines_are_zero_placeholders_requires_all_parsed_rows_zero():
    assert lines_are_zero_placeholders(["a.wav=ka,0,0,0,0,0", "a.wav=a,0,0,0,0,0"]) is True
    assert lines_are_zero_placeholders(["a.wav=ka,0,0,0,0,0", "a.wav=a,1,0,0,0,0"]) is False
    assert lines_are_zero_placeholders(["not-an-oto-row"]) is False


def test_auto_placeholder_block_error_blocks_fatal_reason_without_zero_rows():
    err = build_auto_placeholder_block_error(
        fname="a.wav",
        lines=["a.wav=ka,12,34,-80,20,10"],
        reason="textgrid_missing",
        use_template=False,
    )
    assert "TextGrid/tier/interval/mapping" in err
    assert "reason=textgrid_missing" in err


def test_auto_placeholder_block_error_blocks_zero_template_rows():
    err = build_auto_placeholder_block_error(
        fname="a.wav",
        lines=["a.wav=ka,0,0,0,0,0"],
        reason="review_required",
        use_template=True,
    )
    assert "0값 템플릿 OTO" in err
    assert "reason=review_required" in err


def test_auto_placeholder_block_error_allows_nonfatal_nonzero_rows():
    err = build_auto_placeholder_block_error(
        fname="a.wav",
        lines=["a.wav=ka,12,34,-80,20,10"],
        reason="review_required",
        use_template=True,
    )
    assert err == ""


def test_offset_zero_collapse_error_requires_large_zero_offset_ratio():
    mostly_zero = [f"a.wav=a{i},0,10,-100,5,2" for i in range(9)]
    mostly_zero.append("a.wav=a9,12,10,-100,5,2")
    err = build_offset_zero_collapse_error(mostly_zero, auto_gen_format="cvvc")
    assert "zero_offset_rows=9/10" in err
    assert "format=cvvc" in err

    mixed = [f"a.wav=a{i},0,10,-100,5,2" for i in range(4)]
    mixed.extend(f"a.wav=b{i},12,10,-100,5,2" for i in range(6))
    assert build_offset_zero_collapse_error(mixed, auto_gen_format="cvvc") == ""
