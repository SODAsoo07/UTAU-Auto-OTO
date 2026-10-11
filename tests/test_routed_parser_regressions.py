from pathlib import Path

import pytest

from universal_auto_oto.domain import SourceOtoRow
from universal_auto_oto.voicebank_project_adapter import (
    VoicebankReviewRequired, _flatten_slots, _inferred_phone_definition,
    _planned_aliases, parse_filename_phone_slots,
)


def plans(filename, aliases, *, language="korean", format_type="cvvc"):
    raw = parse_filename_phone_slots(filename, format_hint=format_type, language_hint=language)
    phones, slots = _flatten_slots(raw)
    rows = [SourceOtoRow(filename, alias, i) for i, alias in enumerate(aliases)]
    return phones, _planned_aliases(rows, slots, format_type, filename, language_hint=language)


def test_japanese_n_mora_remains_a_nasal_and_keeps_occurrences():
    phones, aliases = plans("_んあん.wav", ["- ん", "ん あ", "あ ん"], language="japanese", format_type="vcv")
    assert phones == ("n", "a", "n")
    assert _inferred_phone_definition("n").features["manner"] == "nasal"
    assert aliases[0].phone_indices == (0,)
    assert aliases[1].phone_indices == (0, 1)
    assert aliases[2].phone_indices == (1, 2)


@pytest.mark.parametrize("phone", ["n", "m", "ng", "l"])
def test_independent_consonants_bind_head_and_both_transition_directions(phone):
    phones, aliases = plans(f"_{phone}'a'{phone}'e'{phone}.wav",
        [f"- {phone}C4", f"{phone} aC4", f"a {phone}C4", f"{phone} eC4", f"e {phone}C4"])
    assert phones == (phone, "a", phone, "e", phone)
    assert [a.phone_indices for a in aliases] == [(0,), (0, 1), (1, 2), (2, 3), (3, 4)]
    assert [a.role for a in aliases] == ["special", "cv", "vc", "cv", "vc"]


def test_korean_vc_inside_cv_bank_owns_its_coda():
    phones, aliases = plans("ak.wav", ["a k"], format_type="cv")
    assert phones == ("a", "k")
    assert aliases[0].phone_indices == (0, 1)
    assert aliases[0].template == "builtin:vc"
    with pytest.raises(VoicebankReviewRequired):
        parse_filename_phone_slots("bak.wav", format_hint="cv", language_hint="korean")


def test_uppercase_consonants_and_consecutive_n_are_not_discarded():
    assert parse_filename_phone_slots("ga-ga-gga-na-N.wav",format_hint="cvc",language_hint="korean")[-1] == ("n",)
    assert parse_filename_phone_slots("_n'n'weo.wav",format_hint="cvvc",language_hint="korean") == (("n",),("n","weo"))
    assert parse_filename_phone_slots("_n'eu'n'yu'n'yeo.wav",format_hint="cvvc",language_hint="korean") == (("n","eu"),("n","yu"),("n","yeo"))


def test_ending_R_uses_preceding_phone_and_explicit_exhalation():
    phones, aliases = plans("_a'R.wav", ["a RC4"])
    assert phones == ("a", "exh")
    assert _inferred_phone_definition("exh").features["manner"] == "breath"
    assert aliases[0].alias == "a RC4"
    assert aliases[0].phone_indices == (0, 1)
    assert aliases[0].template == "builtin:ending_exhalation"
    with pytest.raises(VoicebankReviewRequired):
        parse_filename_phone_slots("_a'r.wav",format_hint="cvvc",language_hint="korean")


def test_japanese_breath_markers_and_blank_breath_alias():
    assert parse_filename_phone_slots("あ息い息.wav",format_hint="vcv",language_hint="japanese") == (("a",),("exh",),("i",),("exh",))
    phones, aliases = plans("息1.wav", ["息"], language="japanese",format_type="vcv")
    assert phones == ("exh",)
    assert aliases[0].template == "builtin:breath"
    phones, aliases = plans("br1.wav", [""])
    assert aliases[0].alias == ""
    assert aliases[0].template == "builtin:breath"


def test_blank_alias_report_identity_preserves_empty_alias_and_rejects_missing():
    from universal_auto_oto.alias_type_symbolic import _row_key
    assert _row_key({"wav":"br1.wav","alias":"","source_index":0},0) == "br1.wav||0"
    assert _row_key({"wav":"br1.wav","alias":"","source_index":1},1) != "br1.wav||0"
    with pytest.raises(ValueError):
        _row_key({"wav":"br1.wav","source_index":0},0)


def test_sample_file_never_gets_an_invented_phone_sequence():
    with pytest.raises(VoicebankReviewRequired):
        parse_filename_phone_slots("sample.wav",format_hint="cv",language_hint="korean")


@pytest.mark.parametrize("filename", ["kaki.wav", "_kaka.wav", "saki-ta.wav"])
def test_compact_cv_pairs_are_not_read_as_a_coda(filename):
    with pytest.raises(VoicebankReviewRequired, match="unparsed_filename_token"):
        parse_filename_phone_slots(filename, format_hint="cvvc", language_hint="japanese")


def test_real_codas_and_separated_cv_pairs_still_parse():
    assert parse_filename_phone_slots("kan.wav", format_hint="cvvc", language_hint="japanese") == (("k", "a", "n"),)
    assert parse_filename_phone_slots("ka-ki.wav", format_hint="cvvc", language_hint="japanese") == (("k", "a"), ("k", "i"))
    assert parse_filename_phone_slots("_saNG'saN.wav", format_hint="cvvc", language_hint="korean") == (("s", "a", "ng"), ("s", "a", "n"))


def bound(filename, aliases, *, language, format_type):
    phones, plans_ = plans(filename, aliases, language=language, format_type=format_type)
    return phones, {plan.alias: (plan.phone_indices, plan.template) for plan in plans_}


def test_japanese_yoon_binds_the_whole_consonant_glide_vowel_cluster():
    phones, got = bound("_あきゃしゃ.wav", ["きゃG4", "a kyG4", "a きゃ", "- あ"], language="japanese", format_type="cvvc")
    assert phones == ("a", "k", "y", "a", "sh", "y", "a")
    assert got["きゃG4"] == ((1, 2, 3), "builtin:cv_cluster")
    assert got["a kyG4"] == ((0, 1, 2), "builtin:vc_semivowel")
    assert got["a きゃ"] == ((0, 1, 2, 3), "builtin:vcv_cluster")
    _, got = bound("きぇ.wav", ["きぇD4", "- きぇD4"], language="japanese", format_type="cv")
    assert got["きぇD4"][0] == got["- きぇD4"][0] == (0, 1, 2)


def test_japanese_wo_and_ky_before_i_are_fallbacks_only():
    _, got = bound("_うういおあおい.wav", ["i をC4"], language="japanese", format_type="vcv")
    assert got["i をC4"][0] == (2, 3)
    _, got = bound("_いきし.wav", ["i kyG4"], language="japanese", format_type="cvvc")
    assert got["i kyG4"][0] == (0, 1)


def test_pause_token_makes_the_next_syllable_head_eligible():
    _, got = bound("_ba'ba'bya'-'bya'bwa.wav", ["- byaC4", "- baC4"], language="korean", format_type="cvvc")
    assert got["- byaC4"] == ((6, 7), "builtin:cv_head") and got["- baC4"][0] == (0, 1)
    # A lone '-' is only a separator; the mid-file head still binds by the
    # first-occurrence fallback (user policy 2026-10-08).
    from universal_auto_oto.voicebank_project_adapter import _pause_token_slot_indices
    assert _pause_token_slot_indices("ka-ki.wav", 2) == frozenset()
    _, got = bound("ka-ki.wav", ["- ki"], language="japanese", format_type="cv")
    assert got["- ki"] == ((2, 3), "builtin:cv_head")


def test_korean_cvc_vc_is_spelled_as_one_coda_syllable():
    phones, got = bound("ba-ba-bba-ma-M.wav", ["apC5P", "amC5P", "a bC5P"], language="korean", format_type="cvc")
    assert phones[3:5] == ("a", "bb") and phones[7:9] == ("a", "m")
    assert got["apC5P"] == ((3, 4), "builtin:vc_hard") and got["amC5P"] == ((7, 8), "builtin:vc_soft_voiced")
    assert got["a bC5P"] == ((1, 2), "builtin:vc")
    _, got = bound("a-sa-ssa-a-NG-nga.wav", ["angC5P"], language="korean", format_type="cvc")
    assert got["angC5P"][0] == (5, 6)


def test_kana_middle_dot_is_a_fry_or_glottal_slot():
    assert parse_filename_phone_slots("_あ・あ.wav", format_hint="cvvc", language_hint="japanese") == (("a",), ("vf",), ("a",))
    phones, got = bound("_あ・あ.wav", ["・ あG4", "a ・G4"], language="japanese", format_type="cvvc")
    assert phones == ("a", "vf", "a") and got["・ あG4"][0] == (1, 2) and got["a ・G4"][0] == (0, 1)


def test_trailing_asterisk_tags_a_fry_take_and_breath_samples_bind_by_name():
    _, got = bound("_a'a'e'a'i'a'o.wav", ["a a*", "- a*", "a e*"], language="korean", format_type="vcv")
    assert got["a a*"][0] == (0, 1) and got["- a*"][0] == (0,) and got["a e*"][0] == (1, 2)
    _, got = bound("息2.wav", ["br2"], language="japanese", format_type="vcv")
    assert got["br2"][0] == (0,)


def test_japanese_bank_korean_support_aliases_need_a_unique_respelling():
    _, got = bound("_くくきこかこき.wav", ["eo ka"], language="japanese", format_type="vcv")
    assert got["eo ka"][0] == (7, 8, 9)                       # こ→か
    _, got = bound("_ふぁふぁふぃふぁふふぇふぁ.wav", ["- hwa"], language="japanese", format_type="vcv")
    assert got["- hwa"][0] == (0, 1)
    _, got = bound("_そすんさんす.wav", ["ng sa"], language="japanese", format_type="vcv")
    assert got["ng sa"][0] == (4, 5, 6)
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        plans("_うぇうぇううぉうぇうぉうぉ.wav", ["o eo"], language="japanese", format_type="vcv")


def test_korean_cvvc_glide_nucleus_vv_and_bank_respellings():
    _, got = bound("_ya'ya'ye'ya'i'ya'yu.wav", ["a yaC4", "a yeC4"], language="korean", format_type="cvvc")
    assert got["a yaC4"] == ((0, 1), "builtin:vv") and got["a yeC4"][0] == (1, 2)
    _, got = bound("_kkwa'-'kkwe'-'kkwi'-'kkweo.wav", ["- kweC4"], language="korean", format_type="cvvc")
    assert got["- kweC4"][0] == (2, 3)
    _, got = bound("_m'm'weo.wav", ["m woC4"], language="korean", format_type="cvvc")
    assert got["m woC4"][0] == (1, 2)


def test_mid_file_head_alias_binds_the_first_occurrence_with_a_head_template():
    _, got = bound("_a'a'e'a'i'a'o.wav", ["- e*", "- a*"], language="korean", format_type="vcv")
    assert got["- e*"] == ((2,), "builtin:v_head") and got["- a*"][0] == (0,)
    _, got = bound("_いいゆやいぇいいぇ.wav", ["- いぇC4"], language="japanese", format_type="vcv")
    assert got["- いぇC4"] == ((6, 7), "builtin:cv_head")


def test_korean_cvvc_dash_ending_binds_the_last_vowel():
    phones, got = bound("_a'a'i'a'u'a'e'a.wav", ["a -", "i -"], language="korean", format_type="cvvc")
    assert got["a -"][0] == (max(i for i, p in enumerate(phones) if p == "a"),)
    assert got["i -"][0] == (phones.index("i"),)       # a single occurrence stays where it is
    # Other formats keep the first event for '-'.
    phones, got = bound("_a'a'i'a'u'a'e'a.wav", ["a -"], language="korean", format_type="vcv")
    assert got["a -"][0][0] == phones.index("a")


def test_banks_without_cw_spell_the_consonant_before_wi_as_cy():
    from universal_auto_oto.voicebank_project_adapter import _bank_writes_wi_onset_as_y

    name = "_gi'gi'gwi'-'geu'geu'gi.wav"
    aliases = ["i g", "i gy", "eu g", "eu gy"]
    raw = parse_filename_phone_slots(name, format_hint="cvvc", language_hint="korean")
    phones, slots = _flatten_slots(raw)
    assert phones == ("g", "i", "g", "i", "g", "wi", "g", "eu", "g", "eu", "g", "i")
    rows = [SourceOtoRow(name, alias, i) for i, alias in enumerate(aliases)]

    def got(**kw):
        return {p.alias: p.phone_indices for p in _planned_aliases(rows, slots, "cvvc", name, language_hint="korean", **kw)}

    # Cy bank: "i gy" is i -> gwi; "i g" stays i -> gi; "eu gy" has no wi to prefer.
    assert got(wi_as_y_glide=True) == {"i g": (1, 2), "i gy": (3, 4, 5), "eu g": (7, 8), "eu gy": (9, 10, 11)}
    assert got()["i gy"] == (1, 2, 3)
    assert _bank_writes_wi_onset_as_y([rows])
    # A bank that spells Cw anywhere (e.g. "i gw") keeps wi as a w glide.
    assert not _bank_writes_wi_onset_as_y([rows + [SourceOtoRow(name, "i gw", 9)]])
    assert not _bank_writes_wi_onset_as_y([[SourceOtoRow(name, "i g", 0)]])


def test_korean_cvvc_lowercase_vc_names_the_next_onset_not_the_uppercase_coda():
    phones, got = bound("_miNG'miN'miM'miL'mi'meu'mo'meu.wav", ["i m", "i M", "i n", "i N"],
                        language="korean", format_type="cvvc")
    assert phones[7:9] == ("i", "m") and phones[13:15] == ("i", "m")
    assert got["i m"][0] == (13, 14)        # mi -> meu onset (user review 2026-10-11: the coda took the next vowel)
    assert got["i M"][0] == (7, 8)          # miM coda keeps the uppercase alias
    assert got["i N"][0] == (4, 5)
    assert got["i n"][0] == (4, 5)          # no onset n in this file: falls back to the coda as before
