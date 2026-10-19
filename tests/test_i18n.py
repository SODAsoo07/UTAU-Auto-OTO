from __future__ import annotations

import ast
import string
from pathlib import Path

import pytest

from universal_auto_oto import i18n
from universal_auto_oto.oto_generator_ui import (
    CONFIDENCE_LABELS, DETECTION_SOURCE_LABELS, DIAGNOSTIC_FILE_LABELS, FORMAT_CHOICES, LANGUAGE_CHOICES,
    RELEASE_PRE_CHOICES, STYLE_FIELDS, UNMAPPED_ROW_CHOICES, VC_END_CHOICES, format_result_text, progress_text,
)
from universal_auto_oto.oto_preview import UNRATED_NOTE

PACKAGE = Path(i18n.__file__).parent
TRANSLATED_MODULES = ("oto_generator_ui.py", "oto_preview.py", "bank_detection.py", "batch.py", "tempo.py",
                      "generator.py")


def _literal_keys() -> set[str]:
    """Every string literal passed as the first argument of t(...) / translate(...)."""

    keys = set()
    for name in TRANSLATED_MODULES:
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"t", "translate"}
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
    return keys


def _indirect_keys() -> set[str]:
    """Korean keys kept in constants and translated where they are shown."""

    keys = {*LANGUAGE_CHOICES, *FORMAT_CHOICES, *VC_END_CHOICES, *RELEASE_PRE_CHOICES, *UNMAPPED_ROW_CHOICES, *CONFIDENCE_LABELS.values(),
            *DETECTION_SOURCE_LABELS.values(), *DIAGNOSTIC_FILE_LABELS.values(), UNRATED_NOTE,
            "다크 모드", "라이트 모드", "생성", "결과", "로그", "크레딧", "입력과 출력", "생성 프로필", "고급 설정",
            "보이스뱅크 폴더", "기존 oto.ini (선택)", "출력 폴더", "언어", "포맷", "인코딩", "VC 끝 위치", "'-' 선행발성", "생성하지 못한 기존 oto 행",
            "프리셋 불러오기", "프리셋 저장", "기본값으로", "입력 검사", "출력 검증", "출력 폴더 열기", "OTO 미리보기",
            "행 재생 (Space)", "전체 재생", "정지", "행에 맞춤 (F)", "전체 보기", "확대 +", "축소 -", "블랭크", "고정",
            "기본은 별도 폴더 저장입니다. 직접 저장은 아래 덮어쓰기 토글을 켠 경우에만 허용됩니다.",
            "언어는 직접 고르세요. 포맷은 기존 oto.ini 별칭·파일 이름·폴더 이름으로 자동 감지하거나 직접 고를 수 있습니다.",
            "비워 둔 칸은 자동(템플릿) 값을 씁니다. 음소 위치는 그대로 두고 OTO 행 모양만 바꿉니다."}
    for _, label, hint in STYLE_FIELDS:
        keys.add(label)
        if any("가" <= char <= "힣" for char in hint):
            keys.add(hint)
    return keys - {"CV", "CVC", "CVVC", "VCV"}


def _fields(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


@pytest.mark.parametrize("language", ["en", "ja"])
def test_every_ui_string_has_a_translation_with_the_same_fields(language: str) -> None:
    table = i18n._TABLES[language]
    keys = _literal_keys() | _indirect_keys()
    missing = sorted(key for key in keys if key not in table)
    assert not missing, missing
    mismatched = sorted(key for key in keys if _fields(key) != _fields(table[key]))
    assert not mismatched, mismatched
    unused = sorted(set(table) - keys)
    assert not unused, unused


def test_korean_is_the_identity_and_missing_keys_fall_back() -> None:
    assert i18n.translate("생성 행: {count}", "ko", count=3) == "생성 행: 3"
    assert i18n.translate("생성 행: {count}", "en", count=3) == "Generated rows: 3"
    assert i18n.translate("없는 문구 {x}", "ja", x=1) == "없는 문구 1"
    assert i18n.machine_translation_notice("ko") == ""
    assert i18n.machine_translation_notice("en") and i18n.machine_translation_notice("ja")


def test_ui_language_is_saved_and_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(i18n.LANGUAGE_ENV, raising=False)
    path = tmp_path / "AutoOTO" / "settings.json"
    try:
        assert i18n.load_ui_language(path) == "ko"
        i18n.save_ui_language("日本語", path)
        assert i18n.load_ui_language(path) == "ja"
        monkeypatch.setenv(i18n.LANGUAGE_ENV, "en")
        assert i18n.load_ui_language(path) == "en"
    finally:
        i18n.set_language("ko")


def test_result_text_follows_the_active_language() -> None:
    payload = {"wav_count": 2, "generated_rows": 5, "clean_rows": 3, "review_rows": 2}
    try:
        i18n.set_language("en")
        text = format_result_text("Generate OTO", payload)
        assert "Generated rows: 5" in text and "Held for review: 2" in text
        assert progress_text("render", 1, 4, 0.0, 0.0) == "Writing OTO 1/4"
        i18n.set_language("ja")
        assert "生成行: 5" in format_result_text("OTO生成", payload)
    finally:
        i18n.set_language("ko")
    assert "생성 행: 5" in format_result_text("OTO 생성", payload)
