from __future__ import annotations

from pathlib import Path

import pytest

from universal_auto_oto.bank_detection import (
    check_language,
    classify_alias,
    detect_format,
    detect_format_from_aliases,
    detect_format_from_folder_name,
    strip_alias_suffix,
)


def test_strip_alias_suffix_removes_pitch_suffix_and_numbers() -> None:
    stripped, suffix = strip_alias_suffix(["a kC4", "kaC4", "- kaC4", "a -C4", "ka2C4"])
    assert suffix == "C4"
    assert stripped == ["a k", "ka", "- ka", "a -", "ka"]


@pytest.mark.parametrize(
    ("alias", "kind"),
    [
        ("ka", "cv"),
        ("か", "cv"),
        ("a", "v"),
        ("ang", "vc_syllable"),
        ("a k", "vc"),
        ("a ky", "vc"),
        ("a か", "vcv"),
        ("a ka", "vcv"),
        ("a i", "vv"),
        ("- ka", "head"),
        ("a -", "tail"),
        ("a R", "tail"),
        ("a k a", "vcv"),
        ("g a", "cv"),
        ("br1", "other"),
    ],
)
def test_classify_alias(alias: str, kind: str) -> None:
    assert classify_alias(alias) == kind


def test_detect_vcv_from_aliases() -> None:
    aliases = ["- か", "a か", "a き", "i く", "u け", "e こ", "o か", "a さ", "a -", "a し"]
    assert detect_format_from_aliases(aliases, language="japanese").format == "vcv"


def test_detect_cvvc_from_aliases() -> None:
    aliases = ["- ka", "ka", "a k", "ki", "i k", "ku", "u k", "ke", "e k", "a -", "ko", "o k"]
    assert detect_format_from_aliases(aliases, language="korean").format == "cvvc"


def test_detect_korean_cvc_needs_coda_syllables() -> None:
    aliases = ["ba", "a b", "ang", "da", "a d", "ot", "ga", "a g", "ol", "ja", "a j", "eun"]
    assert detect_format_from_aliases(aliases, language="korean").format == "cvc"
    assert detect_format_from_aliases(aliases, language="japanese").format == "cvvc"


def test_detect_cv_from_single_syllable_files(tmp_path: Path) -> None:
    names = [f"{kana}.wav" for kana in "あいうえおかきくけこ"]
    detection = detect_format(tmp_path / "bank", names)
    assert detection.format == "cv"
    assert detection.source == "filenames"


def test_list_files_without_oto_use_folder_name_or_ask(tmp_path: Path) -> None:
    names = ["_ああいあうえあ.wav", "_かかきかくかけ.wav", "_ささしさすさせ.wav", "_たたちたつたて.wav", "_ななになぬなね.wav"]
    assert detect_format(tmp_path / "Bank (JP VCV)", names).format == "vcv"
    unknown = detect_format(tmp_path / "Bank", names)
    assert unknown.format is None
    assert unknown.confidence == "none"


def test_folder_name_camel_case_boundary(tmp_path: Path) -> None:
    assert detect_format_from_folder_name(tmp_path / "DiKORvcv_8mora").format == "vcv"
    assert detect_format_from_folder_name(tmp_path / "Ceylon_KR_VCV").format == "vcv"
    assert detect_format_from_folder_name(tmp_path / "MyBank-KR_CV-VC").format == "cvvc"
    assert detect_format_from_folder_name(tmp_path / "Vocal bank") is None


def test_mixed_oto_follows_folder_name(tmp_path: Path) -> None:
    # A VCV bank that also carries CV and VC rows.
    aliases = ["a か", "i か", "u か", "か", "き", "く", "け", "a k", "i k", "u k", "e k", "o か"]
    detection = detect_format(tmp_path / "bank_jp_vcv", [], base_aliases=aliases, language="japanese")
    assert detection.format == "vcv"
    assert detection.confidence == "medium"


def test_language_warning_for_kana_in_korean_bank() -> None:
    names = ["_ああいあうえあ.wav", "_かかきかくかけ.wav", "_ささしさすさせ.wav"]
    assert check_language("korean", names).warning is not None
    assert check_language("japanese", names).warning is None


def test_language_warning_for_korean_romanization_in_japanese_bank() -> None:
    names = ["_geu'geo'ge'ga.wav", "_eu'eo'a.wav", "_ng'ga'gga.wav", "_sseu'sseo.wav", "_jja'jjeo.wav"]
    assert check_language("japanese", names).warning is not None
    assert check_language("korean", names).warning is None
