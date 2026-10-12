from __future__ import annotations

from universal_auto_oto.alias_generation import generate_aliases, generate_bank_alias_rows
from universal_auto_oto.voicebank_project_adapter import parse_filename_phone_slots


def test_japanese_vcv_kana_aliases() -> None:
    aliases = generate_aliases("_ああいあうえあ.wav", format_hint="vcv", language_hint="japanese")
    assert aliases == ["- あ", "a あ", "a い", "i あ", "a う", "u え", "e あ", "a -"]


def test_japanese_cvvc_uses_kana_cv_and_romaji_vc() -> None:
    aliases = generate_aliases("_かきゃ.wav", format_hint="cvvc", language_hint="japanese")
    assert aliases == ["- か", "か", "a ky", "きゃ", "a -"]


def test_japanese_romaji_file_names_give_kana_aliases() -> None:
    aliases = generate_aliases("a-ba-bi.wav", format_hint="cv", language_hint="japanese")
    assert aliases == ["あ", "ば", "び"]


def test_japanese_romaji_uses_the_usual_kana_spelling() -> None:
    aliases = generate_aliases("cha-si-ti-tu-wi.wav", format_hint="cv", language_hint="japanese")
    assert aliases == ["ちゃ", "すぃ", "てぃ", "とぅ", "うぃ"]


def test_korean_cvvc_codas_and_pause_heads() -> None:
    aliases = generate_aliases("_an'ban'_'gin.wav", format_hint="cvvc", language_hint="korean")
    assert aliases == ["- a", "a", "a n", "n b", "ba", "a n", "- gi", "gi", "i n", "n -"]


def test_korean_cvc_writes_codas_as_one_syllable() -> None:
    aliases = generate_aliases("ban_gin.wav", format_hint="cvc", language_hint="korean")
    assert "an" in aliases and "in" in aliases
    assert "a n" not in aliases


def test_cv_has_no_transition_aliases() -> None:
    assert generate_aliases("ka_ki.wav", format_hint="cv", language_hint="korean") == ["ka", "ki"]


def test_bank_rows_keep_each_alias_once_and_list_unreadable_files() -> None:
    rows, skipped = generate_bank_alias_rows(
        ["ka_ki.wav", "ka_ku.wav", "???.wav"], format_hint="cvvc", language_hint="korean"
    )
    assert [row.alias for row in rows["ka_ki.wav"]] == ["- ka", "ka", "a k", "ki", "i -"]
    assert [row.alias for row in rows["ka_ku.wav"]] == ["ku", "u -"]
    assert [item["wav"] for item in skipped] == ["???.wav"]


def test_japanese_romaji_moraic_n_and_glued_breath_parse() -> None:
    assert parse_filename_phone_slots("ga-N-ga.wav", format_hint="cvvc", language_hint="japanese") == (
        ("g", "a"), ("n",), ("g", "a"))
    assert parse_filename_phone_slots("aR.wav", format_hint="cv", language_hint="japanese") == (("a",), ("exh",))


def test_korean_vcv_run_together_names_are_segmented_without_an_oto():
    from universal_auto_oto.alias_generation import generate_aliases
    from universal_auto_oto.voicebank_project_adapter import _segment_korean_vcv_compact

    assert generate_aliases("_gagageogagegagogaK.wav", format_hint="vcv", language_hint="korean") == [
        "- ga", "a ga", "a geo", "eo ga", "a ge", "e ga", "a go", "o ga", "a k"]
    # The base syllable repeats (bbeo, not bbe+o); vowel-only lists keep single vowels.
    assert _segment_korean_vcv_compact("_bbeobbeobbebbeobbobbeobbubbeo.wav")[:3] == (
        ("bb", "eo"), ("bb", "eo"), ("bb", "e"))
    assert _segment_korean_vcv_compact("_aaeoaeaoaNGH.wav") == (
        ("a",), ("a",), ("eo",), ("a",), ("e",), ("a",), ("o",), ("a", "ng", "h"))
    assert _segment_korean_vcv_compact("L'baL'beo.wav") == (("l",), ("b", "a"), ("l",), ("b", "eo"))
    assert generate_aliases("_aaeoaeaoaNGH.wav", format_hint="vcv", language_hint="korean")[-2:] == ["a ng", "ng h"]
