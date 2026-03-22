from core.generation.mapping_trace import (
    build_ja_mapping_candidate_trace,
    build_top_mapping_candidate_rows,
    maybe_append_cv_mapping_trace,
)


def test_build_top_mapping_candidate_rows_sorts_and_limits():
    rows = build_top_mapping_candidate_rows(
        [
            {"name": "a", "score": 0.3, "objective": 1.0, "order_preserving": False, "lock_order": False},
            {"name": "b", "score": 0.5, "objective": 3.0, "order_preserving": True, "lock_order": True},
            {"name": "c", "score": 0.4, "objective": 2.0, "order_preserving": True, "lock_order": False},
        ],
        limit=2,
    )
    assert [row["name"] for row in rows] == ["b", "c"]
    assert rows[0]["lock_order"] is True
    assert rows[1]["order_preserving"] is True


def test_build_ja_mapping_candidate_trace_payload():
    payload = build_ja_mapping_candidate_trace(
        fname="sample.wav",
        format_type="cvvc",
        mapping_reason_code="alias_recover",
        mapping_tier="mid",
        mapping_confidence=0.63,
        mapping_margin=8.0,
        filename_order_locked=True,
        forced_words_mapping=False,
        alignment_weight=0.55,
        blank_conf_mean=0.12,
        anchor_lock_lite=True,
        selected_candidate_name="filename_token",
        filename_syllables=["ka", "ra"],
        cv_targets=["ka", "ra"],
        candidate_rows=[{"name": "filename_token"}],
    )
    assert payload["event"] == "ja_mapping_candidate"
    assert payload["file"] == "sample.wav"
    assert payload["format_type"] == "cvvc"
    assert payload["selected_candidate"] == "filename_token"
    assert payload["filename_order_locked"] is True
    assert payload["candidate_rows"][0]["name"] == "filename_token"


def test_maybe_append_cv_mapping_trace_appends_when_trace_enabled():
    traces = []
    out = maybe_append_cv_mapping_trace(
        fname="sample.wav",
        alias="ka",
        alias_type="cv",
        format_type="cvvc",
        target_tok="ka",
        expected_idx=0,
        mapped_idx=1,
        syllables_info=[{"tok": "ka"}, {"tok": "ra"}],
        mapping_tier="mid",
        mapping_reason_code="alias_recover",
        mapping_confidence=0.63,
        filename_order_locked=True,
        syllable_confidence_by_idx=[0.72, 0.81],
        should_trace_mapping_decision_fn=lambda **_kwargs: True,
        append_mapping_trace_fn=lambda rec: traces.append(rec),
        build_trace_record_fn=lambda **kwargs: dict(kwargs),
        normalize_syllable_token_fn=lambda s: str(s or "").strip().lower(),
        syllable_info_token_fn=lambda row: str((row or {}).get("tok", "")),
    )
    assert out is True
    assert len(traces) == 1
    assert traces[0]["expected_tok"] == "ka"
    assert traces[0]["mapped_tok"] == "ra"
    assert abs(float(traces[0]["local_conf"]) - 0.72) < 1e-9


def test_maybe_append_cv_mapping_trace_skips_when_trace_disabled():
    traces = []
    out = maybe_append_cv_mapping_trace(
        fname="sample.wav",
        alias="ka",
        alias_type="cv",
        format_type="cvvc",
        target_tok="ka",
        expected_idx=0,
        mapped_idx=0,
        syllables_info=[{"tok": "ka"}],
        mapping_tier="high",
        mapping_reason_code="filename_token",
        mapping_confidence=0.9,
        filename_order_locked=False,
        syllable_confidence_by_idx=[],
        should_trace_mapping_decision_fn=lambda **_kwargs: False,
        append_mapping_trace_fn=lambda rec: traces.append(rec),
        build_trace_record_fn=lambda **kwargs: dict(kwargs),
        normalize_syllable_token_fn=lambda s: str(s or "").strip().lower(),
        syllable_info_token_fn=lambda row: str((row or {}).get("tok", "")),
    )
    assert out is False
    assert traces == []
