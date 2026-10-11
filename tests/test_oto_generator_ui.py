from __future__ import annotations

from pathlib import Path

import pytest

from universal_auto_oto.oto_generator_ui import (
    CREDIT_LINES,
    build_generation_options,
    build_product_generation_options,
    format_result_text,
    summarize_generation_result,
)


def test_build_generation_options_matches_generator_contract(tmp_path: Path) -> None:
    options = build_generation_options(
        bank=tmp_path / "bank",
        output=tmp_path / "output",
        base_oto="base.ini",
        recursive=True,
        output_encoding="cp932",
        review_threshold="0.7",
        allow_heuristic_clean=True,
        max_files="3",
        force=True,
        format_hint="cvvc",
        language_hint="korean",
        backend="routed_hsmm",
    )
    assert options.bank == tmp_path / "bank"
    assert options.output == tmp_path / "output"
    assert options.base_oto == Path("base.ini")
    assert options.recursive is True
    assert options.output_encoding == "cp932"
    assert options.review_threshold == 0.7
    assert options.max_files == 3
    assert options.backend == "routed_hsmm"
    assert options.allow_in_place_output is False


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_build_generation_options_rejects_invalid_threshold(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        build_generation_options(bank=tmp_path / "bank", output=tmp_path / "output", review_threshold=value)


def test_build_generation_options_rejects_invalid_max_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_generation_options(bank=tmp_path / "bank", output=tmp_path / "output", max_files="0")


def test_summary_contains_real_oto_hash(tmp_path: Path) -> None:
    oto = tmp_path / "oto.ini"
    oto.write_text("a.wav=a,0,10,-100,50,20\n", encoding="utf-8")
    summary = summarize_generation_result({"wav_count": 1, "generated_rows": 1}, output=tmp_path)
    assert summary["oto_exists"] is True
    assert summary["oto_sha256"]


def test_product_ui_uses_routed_backend_and_hides_unsafe_clean_controls(tmp_path: Path) -> None:
    options = build_product_generation_options(
        bank=tmp_path / "bank",
        output=tmp_path / "output",
        recursive=True,
        language_choice="한국어",
        format_choice="CVVC",
    )
    assert options.backend == "routed_hsmm"
    assert options.language_hint == "korean"
    assert options.format_hint == "cvvc"
    assert options.recursive is True
    assert options.allow_heuristic_clean is False
    assert options.max_files is None


def test_product_ui_defaults_to_the_last_real_bank_profile(tmp_path: Path) -> None:
    options = build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "output")
    assert options.language_hint == "korean"
    assert options.format_hint == "cvvc"


def test_product_ui_overwrite_toggle_enables_in_place_force(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    options = build_product_generation_options(
        bank=bank,
        output=bank,
        overwrite_existing_oto=True,
    )
    assert options.allow_in_place_output is True
    assert options.force is True


def test_product_ui_keeps_legacy_credits_and_explains_review_rows() -> None:
    assert CREDIT_LINES[0] == "카메비 Kamebit"
    assert CREDIT_LINES[-1] == "집냥 Blue"
    assert len(CREDIT_LINES) == 18
    text = format_result_text(
        "OTO 생성",
        {
            "wav_count": 73,
            "generated_rows": 532,
            "clean_rows": 0,
            "review_rows": 532,
            "review_notice": "review 행은 자동 검수 보류이며 오류 확정이 아닙니다.",
            "reason_counts": {"uncalibrated_confidence": 532},
        },
    )
    assert "검수 보류: 532" in text
    assert "오류 확정이 아닙니다" in text
    assert "uncalibrated_confidence: 532" in text


def test_product_ui_applies_the_recommended_generation_settings(tmp_path: Path) -> None:
    from universal_auto_oto.generator import DEVELOPMENT_ALIGNMENT_PROFILES
    from universal_auto_oto.oto_generator_ui import (
        RECOMMENDED_DEVELOPMENT_PROFILE, RECOMMENDED_RELEASE_PRE_POLICY, RECOMMENDED_VC_END_POLICY,
    )
    options = build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "output")
    assert options.development_alignment_profile == RECOMMENDED_DEVELOPMENT_PROFILE == "cvvc_slot_rhythm_v16"
    assert RECOMMENDED_DEVELOPMENT_PROFILE in DEVELOPMENT_ALIGNMENT_PROFILES
    assert options.vc_end_policy == RECOMMENDED_VC_END_POLICY == "consonant"
    assert options.release_pre_policy == RECOMMENDED_RELEASE_PRE_POLICY == "vowel_end"
    # The general builder keeps the plain defaults for other callers.
    plain = build_generation_options(bank=tmp_path / "bank", output=tmp_path / "output", backend="routed_hsmm")
    assert (plain.development_alignment_profile, plain.vc_end_policy, plain.release_pre_policy) == (None, "template", "template")


def test_generation_summary_reports_the_settings_used(tmp_path: Path) -> None:
    report = {"provenance": {"development_alignment_profile": "cvvc_slot_rhythm_v16",
                             "vc_end_policy": "consonant", "release_pre_policy": "vowel_end"}}
    summary = summarize_generation_result(report, output=tmp_path)
    assert summary["generation_settings"] == report["provenance"]


def test_product_ui_leaves_unbindable_rows_out_and_logs_them(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf
    from universal_auto_oto.generator import generate_voicebank
    from universal_auto_oto.oto_generator_ui import PRODUCT_UNMAPPED_ALIAS_POLICY, excluded_rows_log_lines

    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "ka.wav", np.sin(2 * np.pi * 220 * np.arange(12000) / 16000) * .15, 16000)
    (bank / "oto.ini").write_text("ka.wav=ka,0,0,0,0,0\nka.wav=zzz,0,0,0,0,0\n", encoding="utf-8")
    options = build_product_generation_options(bank=bank, output=tmp_path / "out", base_oto=bank / "oto.ini",
                                               format_choice="CV",
                                               diagnostic_files={"write_unmapped_aliases_csv": True})
    assert options.unmapped_alias_policy == PRODUCT_UNMAPPED_ALIAS_POLICY == "exclude_row"
    summary = summarize_generation_result(generate_voicebank(options), output=tmp_path / "out")
    assert summary["generated_rows"] == 1 and summary["excluded_alias_rows"] == 1
    assert summary["excluded_alias_samples"] == [{"wav": "ka.wav", "alias": "zzz", "reason": "unmapped_alias"}]
    lines = excluded_rows_log_lines(summary)
    assert lines[0].startswith("[주의] 생성되지 않은 행 1개")
    assert str(summary["excluded_alias_rows_csv"]) in lines[1] and lines[2] == "  - ka.wav = zzz"
    assert "생성되지 않은 행 1개" in format_result_text("OTO 생성", summary)


def test_excluded_row_log_lines_summarise_the_rest() -> None:
    from universal_auto_oto.oto_generator_ui import excluded_rows_log_lines

    assert excluded_rows_log_lines({"excluded_alias_rows": 0}) == []
    payload = {"excluded_alias_rows": 12, "excluded_alias_rows_csv": "out/unmapped_aliases.csv",
               "excluded_alias_samples": [{"wav": "a.wav", "alias": "x"}, {"wav": "b.wav", "alias": "y"}]}
    lines = excluded_rows_log_lines(payload)
    assert lines[-1] == "  … 외 10개" and len(lines) == 5


def _one_wav_bank(tmp_path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "ka.wav", np.sin(2 * np.pi * 220 * np.arange(12000) / 16000) * .15, 16000)
    (bank / "oto.ini").write_text("ka.wav=ka,0,0,0,0,0\nka.wav=zzz,0,0,0,0,0\n", encoding="utf-8")
    return bank


def test_product_ui_writes_no_diagnostic_files_by_default(tmp_path: Path) -> None:
    from universal_auto_oto.generator import DIAGNOSTIC_FILE_OPTIONS, generate_voicebank
    from universal_auto_oto.oto_generator_ui import excluded_rows_log_lines

    options = build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "output")
    assert not any(getattr(options, name) for name in DIAGNOSTIC_FILE_OPTIONS)
    # Other callers (CLI, experiment scripts) keep every diagnostic file.
    plain = build_generation_options(bank=tmp_path / "bank", output=tmp_path / "output", backend="routed_hsmm")
    assert all(getattr(plain, name) for name in DIAGNOSTIC_FILE_OPTIONS)

    bank = _one_wav_bank(tmp_path)
    out = tmp_path / "out"
    options = build_product_generation_options(bank=bank, output=out, base_oto=bank / "oto.ini", format_choice="CV")
    summary = summarize_generation_result(generate_voicebank(options), output=out)
    assert summary["generated_rows"] == 1
    for name in ("generation_report.json", "row_report.jsonl", "timelines", "project_manifest.json",
                 "unmapped_aliases.csv", "symbolic_alias_type_labels.jsonl"):
        assert not (out / name).exists(), name
    assert (out / "oto.ini").is_file()
    assert summary["report_path"] is None and summary["row_report_path"] is None
    # The log still names the excluded rows without the CSV.
    assert summary["excluded_alias_samples"] == [{"wav": "ka.wav", "alias": "zzz", "reason": "unmapped_alias"}]
    lines = excluded_rows_log_lines(summary)
    assert "unmapped_aliases.csv 저장을 켜세요" in lines[1] and lines[2] == "  - ka.wav = zzz"


def test_product_ui_diagnostic_toggles_do_not_change_oto(tmp_path: Path) -> None:
    from universal_auto_oto.generator import DIAGNOSTIC_FILE_OPTIONS, generate_voicebank

    bank = _one_wav_bank(tmp_path)
    off = build_product_generation_options(bank=bank, output=tmp_path / "off", base_oto=bank / "oto.ini",
                                           format_choice="CV")
    on = build_product_generation_options(bank=bank, output=tmp_path / "on", base_oto=bank / "oto.ini",
                                          format_choice="CV",
                                          diagnostic_files={name: True for name in DIAGNOSTIC_FILE_OPTIONS})
    generate_voicebank(off)
    generate_voicebank(on)
    for name in ("generation_report.json", "row_report.jsonl", "project_manifest.json", "unmapped_aliases.csv"):
        assert (tmp_path / "on" / name).is_file(), name
    assert any((tmp_path / "on" / "timelines").iterdir())
    assert (tmp_path / "off" / "oto.ini").read_bytes() == (tmp_path / "on" / "oto.ini").read_bytes()


def test_product_ui_rejects_unknown_diagnostic_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "output",
                                         diagnostic_files={"write_everything": True})
def test_product_ui_auto_format_uses_detection_and_generates_aliases_without_oto(tmp_path: Path) -> None:
    from universal_auto_oto.oto_generator_ui import AUTO_FORMAT_CHOICE

    options = build_product_generation_options(
        bank=tmp_path / "bank", output=tmp_path / "out", format_choice=AUTO_FORMAT_CHOICE, detected_format="vcv"
    )
    assert options.format_hint == "vcv"
    assert options.generate_missing_aliases is True
    with pytest.raises(ValueError, match="포맷을 직접"):
        build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "out", format_choice=AUTO_FORMAT_CHOICE)
    with_oto = build_product_generation_options(
        bank=tmp_path / "bank", output=tmp_path / "out", base_oto=tmp_path / "oto.ini", format_choice="CV"
    )
    assert with_oto.generate_missing_aliases is False


def test_product_ui_bpm_and_advanced_settings(tmp_path: Path) -> None:
    from universal_auto_oto.oto_generator_ui import RELEASE_PRE_CHOICES, VC_END_CHOICES, build_oto_style, parse_bpm

    assert parse_bpm("") is None
    assert parse_bpm(" 128 ") == 128.0
    with pytest.raises(ValueError):
        parse_bpm("fast")
    with pytest.raises(ValueError):
        parse_bpm("300")
    style = build_oto_style({"cv_overlap_ratio": "0.5", "vc_pre_shift_ms": ""})
    assert style.cv_overlap_ratio == 0.5 and style.vc_pre_shift_ms == 0.0
    with pytest.raises(ValueError):
        build_oto_style({"cv_overlap_ratio": "abc"})
    options = build_product_generation_options(
        bank=tmp_path / "bank", output=tmp_path / "out", bpm="120", estimate_bpm=True, oto_style=style,
        vc_end_choice=list(VC_END_CHOICES)[1], release_pre_choice=list(RELEASE_PRE_CHOICES)[1],
    )
    assert options.bpm == 120.0 and options.estimate_bpm is True
    assert options.oto_style == style
    assert options.vc_end_policy == "template" and options.release_pre_policy == "template"
    default = build_product_generation_options(bank=tmp_path / "bank", output=tmp_path / "out", oto_style=build_oto_style({}))
    assert default.oto_style is None


def test_tempo_and_alias_log_lines() -> None:
    from universal_auto_oto.oto_generator_ui import alias_generation_log_lines, tempo_log_lines

    assert tempo_log_lines({"recording_tempo": {"source": "user_bpm", "user_bpm": 120.0, "syllable_period_ms": 500.0}})
    refused = tempo_log_lines({"recording_tempo": {"source": "off", "estimate_enabled": True,
                                                   "estimate": {"reason": "박자가 일정한 파일이 2개뿐"}}})
    assert "추정 안 함" in refused[0]
    lines = alias_generation_log_lines({"generated_alias_wavs": 3, "alias_generation_skipped_count": 1,
                                        "alias_generation_skipped": [{"wav": "x.wav", "detail": "x"}]})
    assert len(lines) == 3


def test_progress_fraction_and_text():
    from universal_auto_oto.oto_generator_ui import progress_fraction, progress_text

    assert progress_fraction("align:1/3", 0, 10) == 0.0
    assert abs(progress_fraction("align:3/3", 10, 10) - 0.92) < 1e-9
    assert abs(progress_fraction("render", 5, 10) - 0.96) < 1e-9 and progress_fraction("render", 10, 10) == 1.0
    assert progress_text("align:2/3", 312, 758, 2.0, 0.4) == "정렬 2/3단계 312/758"  # too early for an estimate
    assert progress_text("align:2/3", 312, 758, 120.0, 0.4) == "정렬 2/3단계 312/758 · 남은 시간 약 3분"
    assert progress_text("render", 10, 758, 400.0, 0.95, "[2/5] ") == "[2/5] OTO 작성 10/758 · 남은 시간 약 21초"


def test_preview_oto_path_prefers_last_generated_then_output_then_bank(tmp_path: Path) -> None:
    from universal_auto_oto.oto_generator_ui import preview_oto_path

    bank, output = tmp_path / "bank", tmp_path / "out"
    bank.mkdir()
    output.mkdir()
    assert preview_oto_path(last_oto=None, output=output, bank=bank) is None
    (bank / "oto.ini").write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
    assert preview_oto_path(last_oto=None, output=output, bank=bank) == bank / "oto.ini"
    (output / "oto.ini").write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
    assert preview_oto_path(last_oto=None, output=output, bank=bank) == output / "oto.ini"
    generated = tmp_path / "gen.ini"
    generated.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
    assert preview_oto_path(last_oto=str(generated), output=output, bank=bank) == generated
