from core.generation.mapping_reason_logs import (
    build_ja_mapping_reason_log,
    build_kr_mapping_reason_log,
)


def test_build_ja_mapping_reason_log_filename_token():
    text = build_ja_mapping_reason_log(
        selected_candidate={"name": "filename_token"},
        mapping_reason_code="filename_token",
        filename_syllable_count=3,
        cv_target_count=3,
        phone_interval_count=8,
        word_interval_count=3,
        spn_ratio=0.2,
        base_score=12.0,
        alt_score=9.0,
    )
    assert text is not None
    assert "(3" in text


def test_build_ja_mapping_reason_log_alias_guard():
    text = build_ja_mapping_reason_log(
        selected_candidate={"name": "alias_phone_fallback"},
        mapping_reason_code="alias_cvvc_youon_mismatch",
        filename_syllable_count=2,
        cv_target_count=2,
        phone_interval_count=6,
        word_interval_count=2,
        spn_ratio=0.1,
        base_score=10.0,
        alt_score=20.0,
    )
    assert text is not None
    assert "CVVC" in text
    assert "reason=alias_cvvc_youon_mismatch" in text


def test_build_kr_mapping_reason_log_order_locked():
    text = build_kr_mapping_reason_log(
        mapping_reason_code="order_locked_glide_mismatch",
        base_score=14.0,
        alt_score=22.0,
        words_glide_mismatch_ratio=0.42,
        low_quality_reasons=[],
        alias_syllables_present=True,
        targets_for_build=["ga"],
        textgrid_trust_tier="mid",
        textgrid_trust_score=0.55,
    )
    assert text is not None
    assert "reason=order_locked_glide_mismatch" in text
    assert "glide_mismatch=0.42" in text


def test_build_kr_mapping_reason_log_default_words_keep():
    text = build_kr_mapping_reason_log(
        mapping_reason_code="words_keep",
        base_score=11.0,
        alt_score=10.0,
        words_glide_mismatch_ratio=0.0,
        low_quality_reasons=[],
        alias_syllables_present=True,
        targets_for_build=["ga"],
        textgrid_trust_tier="high",
        textgrid_trust_score=0.8,
    )
    assert text is not None
    assert "base=11.0" in text
    assert "corrected=10.0" in text
