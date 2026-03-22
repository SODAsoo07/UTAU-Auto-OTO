from core.error_codes import format_error_with_recovery, get_error_entry
from core.pipeline_status import (
    ALIGN_DICT_MISSING,
    ALIGN_EXEC_MISSING,
    ALIGN_MODEL_MISSING,
    ALIGN_NOT_READY,
    ALIGN_OUTPUT_EMPTY,
    ALIGN_RUN_FAILED,
    ALIGN_SKIPPED,
    ALIGN_USING_EXISTING,
    EXCEPTION,
    GENERATOR_ERROR,
    ML_APPLIED,
    ML_BUNDLE_INVALID,
    ML_DISABLED_ENV,
    ML_INFER_FAILED,
    ML_INPUT_MISSING,
    ML_MODEL_MISSING,
    ML_NO_FEATURES,
    ML_POLICY_OFF,
    ML_ROUTE_UNAVAILABLE,
    TEXTGRID_MISSING,
    UNSUPPORTED_LANGUAGE,
    VOICEBANK_MISSING,
)


def test_new_preflight_collision_codes_are_registered():
    entry = get_error_entry("PRE_OUTPUT_COLLISION")
    assert entry is not None
    assert entry.stage == "preflight"
    assert "same output oto path" in entry.cause


def test_ml_runtime_codes_are_registered():
    entry = get_error_entry("ML_INFER_FAILED")
    assert entry is not None
    assert entry.stage == "ml"
    assert "inference failed" in entry.cause


def test_pipeline_status_codes_are_registered_in_error_catalog():
    tracked_codes = [
        ALIGN_EXEC_MISSING,
        ALIGN_MODEL_MISSING,
        ALIGN_DICT_MISSING,
        ALIGN_NOT_READY,
        ALIGN_OUTPUT_EMPTY,
        ALIGN_RUN_FAILED,
        ALIGN_USING_EXISTING,
        ALIGN_SKIPPED,
        ML_POLICY_OFF,
        ML_DISABLED_ENV,
        ML_INPUT_MISSING,
        ML_MODEL_MISSING,
        ML_BUNDLE_INVALID,
        ML_ROUTE_UNAVAILABLE,
        ML_NO_FEATURES,
        ML_INFER_FAILED,
        ML_APPLIED,
        GENERATOR_ERROR,
        UNSUPPORTED_LANGUAGE,
        VOICEBANK_MISSING,
        TEXTGRID_MISSING,
        EXCEPTION,
    ]
    missing = [code for code in tracked_codes if get_error_entry(code) is None]
    assert missing == []


def test_format_error_with_recovery_includes_registered_recovery():
    text = format_error_with_recovery("PRE_REPLACE_COLLISION", "case collision")
    assert text.startswith("[PRE_REPLACE_COLLISION] case collision | ")
    assert "split targets" in text


def test_format_error_with_recovery_unknown_code_fallback():
    text = format_error_with_recovery("UNKNOWN_X", "raw message")
    assert text == "[UNKNOWN_X] raw message"
