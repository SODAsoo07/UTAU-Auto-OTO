from types import SimpleNamespace

from core.generation.kr_syllable_source import (
    build_kr_word_syllables,
    collect_blank_confidences,
    log_kr_syllable_source_choice,
    prepare_kr_syllable_source_context,
)


def test_build_kr_word_syllables_uses_interval_lookup_and_roman_kernel():
    word = SimpleNamespace(mark="ka", minTime=0.1, maxTime=0.4)
    phones = [SimpleNamespace(mark="k"), SimpleNamespace(mark="a")]

    result = build_kr_word_syllables(
        word_intervals=[word],
        phone_intervals=phones,
        force_words_phone_fill=False,
        build_interval_lookup_fn=lambda value: ("lookup", value),
        intervals_within_bounds_fn=lambda lookup, start, end: lookup[1] if start < 0.2 and end > 0.3 else [],
        synthesize_word_phones_fn=lambda *_args: ["unexpected"],
        decompose_hangul_to_roman_fn=lambda ch: [ch],
        kr_cv_kernel_fn=lambda text: text.upper(),
    )

    assert result == [
        {
            "word": "ka",
            "roman": "ka",
            "roman_cv": "KA",
            "start_time": 0.1,
            "end_time": 0.4,
            "phones": phones,
        }
    ]


def test_build_kr_word_syllables_can_fill_missing_word_phones():
    word = SimpleNamespace(mark="a", minTime=0.0, maxTime=0.3)

    result = build_kr_word_syllables(
        word_intervals=[word],
        phone_intervals=[],
        force_words_phone_fill=True,
        build_interval_lookup_fn=lambda value: value,
        intervals_within_bounds_fn=lambda *_args: [],
        synthesize_word_phones_fn=lambda mark, start, end, _decompose: [(mark, start, end)],
        decompose_hangul_to_roman_fn=lambda ch: [ch],
        kr_cv_kernel_fn=lambda text: text,
    )

    assert result[0]["phones"] == [("a", 0.0, 0.3)]


def test_collect_blank_confidences_returns_values_and_mean():
    values, mean = collect_blank_confidences([
        {"blank_confidence": 0.25},
        {"blank_confidence": 0.75},
    ])

    assert values == [0.25, 0.75]
    assert mean == 0.5


def test_log_kr_syllable_source_choice_emits_order_lock_reason():
    logs = []

    log_kr_syllable_source_choice(
        mapping_reason_code="order_locked_glide_mismatch",
        fname="take.wav",
        textgrid_trust_tier="mid",
        textgrid_trust_score=0.62,
        low_quality_reasons=[],
        base_score=3.0,
        alt_score=5.0,
        words_glide_mismatch_ratio=0.4,
        has_alias_based=True,
        has_targets_for_build=True,
        log_fn=logs.append,
    )

    assert "order_locked_glide_mismatch" in logs[0]
    assert "glide_mismatch=0.40" in logs[0]


def test_prepare_kr_syllable_source_context_selects_and_annotates_source():
    logs = []

    def annotate(syllables, _mel_ctx):
        for idx, syllable in enumerate(syllables or []):
            syllable["blank_confidence"] = 0.2 + (idx * 0.2)

    def select_source(**kwargs):
        return {
            "syllables_info": kwargs["alias_syllables"],
            "used_words_based": False,
            "used_alias_based": True,
            "mapping_reason_code": "alias_based_cvvc",
            "base_score": 2.0,
            "alt_score": 8.0,
            "words_glide_mismatch_ratio": 0.0,
        }

    context = prepare_kr_syllable_source_context(
        phone_intervals=["p0"],
        word_intervals=[SimpleNamespace(mark="ka", minTime=0.0, maxTime=0.2)],
        targets_for_build=["ka"],
        file_format="cvvc",
        prefer_filename_sequence=False,
        low_phone_quality=False,
        force_words_phone_fill=False,
        mel_ctx_for_file={"blank": True},
        fname="take.wav",
        textgrid_trust_tier="high",
        textgrid_trust_score=0.9,
        low_quality_reasons=[],
        log_fn=logs.append,
        build_interval_lookup_fn=lambda value: value,
        intervals_within_bounds_fn=lambda *_args: ["wphone"],
        synthesize_word_phones_fn=lambda *_args: ["unexpected"],
        decompose_hangul_to_roman_fn=lambda ch: [ch],
        kr_cv_kernel_fn=lambda text: text,
        build_syllables_from_phone_nuclei_fn=lambda _phones, _targets: [{"roman_cv": "ka", "phones": ["aphone"]}],
        annotate_blank_confidence_fn=annotate,
        select_syllable_source_fn=select_source,
        score_mapping_fn=lambda *_args, **_kwargs: 0.0,
        should_prefer_alias_fn=lambda *_args, **_kwargs: False,
        compute_glide_mismatch_fn=lambda *_args, **_kwargs: 0.0,
        is_order_locked_format_fn=lambda _fmt: False,
    )

    assert context["used_alias_based"] is True
    assert context["syllables_info"] == [{"roman_cv": "ka", "phones": ["aphone"], "blank_confidence": 0.2}]
    assert context["syllable_blank_confidences"] == [0.2]
    assert context["blank_conf_mean"] == 0.2
    assert logs and "CVVC" in logs[0]
