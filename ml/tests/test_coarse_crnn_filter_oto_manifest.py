from __future__ import annotations

from ml.scripts.coarse_crnn.filter_oto_manifest import filter_rows, summarize_filter


def test_filter_rows_can_drop_blank_alias_for_one_voicebank_format():
    rows = [
        _row("target", "cvvc", "a"),
        _row("target", "cvvc", ""),
        _row("target", "cvc", "b"),
        _row("other", "cvvc", "c"),
    ]

    kept, removed = filter_rows(rows, voicebank_id="target", format_type="cvvc", drop_blank_alias=True)
    summary = summarize_filter(rows, kept, removed, manifest="in.jsonl", out="out.jsonl")

    assert [row["alias"] for row in kept] == ["a"]
    assert removed["blank_alias"] == 1
    assert removed["format_type"] == 1
    assert removed["voicebank_id"] == 1
    assert summary["rows_out"] == 1
    assert summary["by_context_out"] == {"cvvc|cv|multi": 1}


def _row(voicebank: str, fmt: str, alias: str) -> dict:
    return {
        "voicebank_id": voicebank,
        "format_type": fmt,
        "alias": alias,
        "alias_type": "cv",
        "transition_type": "multi",
    }
