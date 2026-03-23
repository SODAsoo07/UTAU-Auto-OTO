from core.preflight_common import (
    build_preflight_issue_payload,
    collect_runtime_preflight_issues,
    collect_batch_preflight_issue_records,
    collect_batch_preflight_issues,
    issue_records_to_display_strings,
    validate_batch_case_settings_records,
    validate_runtime_preflight,
)


def test_runtime_preflight_missing_wav_dir(tmp_path):
    result = validate_runtime_preflight(
        language="korean",
        wav_dir=str(tmp_path / "missing"),
        out_path="",
        aligner="mfa",
        textgrid_dir=str(tmp_path / "textgrids"),
        require_output=False,
    )
    codes = {entry.get("code") for entry in result.get("errors", [])}
    warning_codes = {entry.get("code") for entry in result.get("warnings", [])}
    assert "PRE_WAV_DIR_MISSING" in codes
    assert "PRE_OUTPUT_PATH_MISSING" in warning_codes
    wav_entry = next(
        entry for entry in result.get("errors", []) if entry.get("code") == "PRE_WAV_DIR_MISSING"
    )
    assert wav_entry.get("stage") == "preflight"
    assert "WAV/voicebank directory" in str(wav_entry.get("cause") or "")
    assert str(wav_entry.get("recovery") or "").strip()


def test_collect_runtime_preflight_issues_contains_records_and_strings(tmp_path):
    result = collect_runtime_preflight_issues(
        language="korean",
        wav_dir=str(tmp_path / "missing"),
        out_path="",
        aligner="mfa",
        textgrid_dir=str(tmp_path / "textgrids"),
        require_output=False,
    )
    assert isinstance(result.get("errors"), list)
    assert isinstance(result.get("warnings"), list)
    assert isinstance(result.get("error_records"), list)
    assert isinstance(result.get("warning_records"), list)
    assert result["error_records"][0]["code"] == "PRE_WAV_DIR_MISSING"
    assert result["errors"][0].startswith("[PRE_WAV_DIR_MISSING]")


def test_runtime_preflight_no_mfa_requires_textgrid(tmp_path):
    wav_dir = tmp_path / "vb"
    wav_dir.mkdir(parents=True, exist_ok=True)
    result = validate_runtime_preflight(
        language="japanese",
        wav_dir=str(wav_dir),
        out_path=str(tmp_path / "out.ini"),
        aligner="none",
        textgrid_dir=str(tmp_path / "missing_textgrids"),
        require_output=True,
    )
    codes = {entry.get("code") for entry in result.get("errors", [])}
    assert "PRE_TEXTGRID_REQUIRED" in codes


def test_validate_batch_case_settings_records(tmp_path):
    case_info = {
        "enabled": True,
        "name": "korean_case",
        "language": "korean",
        "voicebank_dir": str(tmp_path / "missing_vb"),
        "textgrid_dir": str(tmp_path / "missing_tg"),
        "align_if_missing": False,
        "tpl_path": "",
        "custom_phonemes_path": "",
        "output_oto": "",
        "replace_oto_ini": False,
        "replace_target_oto": "",
    }
    records = validate_batch_case_settings_records(case_info)
    codes = {entry.get("code") for entry in records.get("errors", [])}
    assert "PRE_WAV_DIR_MISSING" in codes
    assert "PRE_TEXTGRID_REQUIRED" in codes
    assert "PRE_OUTPUT_PATH_MISSING" in codes


def test_collect_batch_preflight_issue_records_has_structured_errors_and_warnings(tmp_path):
    vb1 = tmp_path / "vb1"
    vb2 = tmp_path / "vb2"
    vb1.mkdir(parents=True, exist_ok=True)
    vb2.mkdir(parents=True, exist_ok=True)
    shared_out = str(tmp_path / "out" / "shared.ini")
    shared_replace = str(tmp_path / "replace" / "oto.ini")
    case_infos = [
        {
            "enabled": True,
            "name": "duplicate_case",
            "language": "korean",
            "voicebank_dir": str(vb1),
            "textgrid_dir": str(tmp_path / "missing_tg_1"),
            "align_if_missing": True,
            "tpl_path": "",
            "custom_phonemes_path": "",
            "output_oto": shared_out,
            "replace_oto_ini": True,
            "replace_target_oto": shared_replace,
        },
        {
            "enabled": True,
            "name": "other_case",
            "language": "japanese",
            "voicebank_dir": str(vb2),
            "textgrid_dir": str(tmp_path / "missing_tg_2"),
            "align_if_missing": True,
            "tpl_path": "",
            "custom_phonemes_path": "",
            "output_oto": shared_out,
            "replace_oto_ini": True,
            "replace_target_oto": shared_replace,
        },
        {
            "enabled": True,
            "name": "duplicate_case",
            "language": "korean",
            "voicebank_dir": str(vb1),
            "textgrid_dir": str(tmp_path / "missing_tg_3"),
            "align_if_missing": True,
            "tpl_path": "",
            "custom_phonemes_path": "",
            "output_oto": str(tmp_path / "out" / "unique.ini"),
            "replace_oto_ini": False,
            "replace_target_oto": "",
        },
    ]
    records = collect_batch_preflight_issue_records(case_infos)
    error_codes = {entry.get("code") for entry in records.get("errors", [])}
    warning_codes = {entry.get("code") for entry in records.get("warnings", [])}
    assert "PRE_DUPLICATE_CASE" in error_codes
    assert "PRE_OUTPUT_COLLISION" in error_codes
    assert "PRE_REPLACE_COLLISION" in error_codes
    assert "PRE_TEXTGRID_REQUIRED" in warning_codes


def test_collect_batch_preflight_issues_returns_display_strings(tmp_path):
    vb = tmp_path / "vb"
    vb.mkdir(parents=True, exist_ok=True)
    case_infos = [
        {
            "enabled": True,
            "name": "unsupported_case",
            "language": "unknown",
            "voicebank_dir": str(vb),
            "textgrid_dir": str(tmp_path / "missing_tg"),
            "align_if_missing": False,
            "tpl_path": "",
            "custom_phonemes_path": "",
            "output_oto": str(tmp_path / "out.ini"),
            "replace_oto_ini": False,
            "replace_target_oto": "",
        }
    ]
    issues = collect_batch_preflight_issues(case_infos)
    assert issues.get("errors")
    assert isinstance(issues["errors"][0], str)
    assert issues["errors"][0].startswith("[UNSUPPORTED_LANGUAGE]")


def test_issue_records_to_display_strings_filters_empty_entries():
    out = issue_records_to_display_strings(
        [
            {"display": "[A] one"},
            {"display": ""},
            {},
            {"display": "[B] two"},
        ]
    )
    assert out == ["[A] one", "[B] two"]


def test_build_preflight_issue_payload_keeps_records_and_displays():
    records = {
        "errors": [{"display": "[E] e1", "code": "E1"}],
        "warnings": [{"display": "[W] w1", "code": "W1"}],
    }
    out = build_preflight_issue_payload(records)
    assert out["errors"] == ["[E] e1"]
    assert out["warnings"] == ["[W] w1"]
    assert out["error_records"][0]["code"] == "E1"
    assert out["warning_records"][0]["code"] == "W1"
