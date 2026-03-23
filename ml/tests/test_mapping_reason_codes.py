from core.generation.mapping_reason_codes import (
    COMMON_REASON_FILENAME_TOKEN,
    JA_REASON_FILENAME_LINEAR_LOW_CONF,
    KR_REASON_WORDS_KEEP,
    get_known_mapping_reason_codes,
    is_known_mapping_reason_code,
    normalize_mapping_reason_code,
    resolve_mapping_reason_schema,
)


def test_known_mapping_reason_codes_include_language_defaults():
    kr_codes = get_known_mapping_reason_codes("kr")
    ja_codes = get_known_mapping_reason_codes("ja")
    assert KR_REASON_WORDS_KEEP in kr_codes
    assert JA_REASON_FILENAME_LINEAR_LOW_CONF in ja_codes
    assert COMMON_REASON_FILENAME_TOKEN in kr_codes
    assert COMMON_REASON_FILENAME_TOKEN in ja_codes


def test_normalize_mapping_reason_code_preserves_unknown():
    code = normalize_mapping_reason_code("Custom_New_Reason", language="ja", fallback_code=COMMON_REASON_FILENAME_TOKEN)
    assert code == "custom_new_reason"


def test_normalize_mapping_reason_code_uses_fallback_when_empty():
    code = normalize_mapping_reason_code("", language="kr", fallback_code=KR_REASON_WORDS_KEEP)
    assert code == KR_REASON_WORDS_KEEP


def test_resolve_mapping_reason_schema_for_known_reason():
    schema = resolve_mapping_reason_schema(KR_REASON_WORDS_KEEP, language="kr")
    assert schema["known"] is True
    assert schema["stage"] == "source_select"
    assert schema["cause"] != "unknown"
    assert schema["recovery"] != "inspect_runtime_log"


def test_resolve_mapping_reason_schema_for_unknown_reason():
    schema = resolve_mapping_reason_schema("some_unknown_reason", language="ja")
    assert schema["known"] is False
    assert schema["stage"] == "unknown"
    assert schema["recovery"] == "inspect_runtime_log"


def test_is_known_mapping_reason_code():
    assert is_known_mapping_reason_code(KR_REASON_WORDS_KEEP, language="kr") is True
    assert is_known_mapping_reason_code("not_real_reason", language="kr") is False
