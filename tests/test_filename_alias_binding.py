"""Filename-first bindings must not depend on row order, count or OTO timing."""
import itertools
import json

import pytest

from universal_auto_oto.domain import SourceOtoRow
from universal_auto_oto.voicebank_project_adapter import (
    VoicebankReviewRequired, _flatten_slots, _planned_aliases, parse_filename_phone_slots,
    build_routed_voicebank_project,
    _alias_exhalation_slots,
)


def bind(filename, aliases, *, language="japanese", format_type="cv"):
    _, slots = _flatten_slots(parse_filename_phone_slots(filename, format_hint=format_type, language_hint=language))
    rows = [SourceOtoRow(filename, a, i) for i, a in enumerate(aliases)]
    if language == "korean" and format_type == "cvvc":
        _, slots, _ = _alias_exhalation_slots(slots, rows)
    return _planned_aliases(rows,
                            slots, format_type, filename, language_hint=language)


def test_sparse_mixed_aliases_share_occurrences_in_any_row_order():
    expected = {"ki": ((3, 4), "cv"), "a k1": ((0, 1), "vc"),
                "a ka": ((0, 1, 2), "vcv"), "か": ((1, 2), "cv")}
    for order in itertools.permutations(expected):
        plans = bind("a-ka-ki-ku-ke-ko.wav", order)
        assert {p.alias: (p.phone_indices, p.role) for p in plans} == expected
    # A one-row OTO must still refer to the complete recording topology.
    assert bind("a-ka-ki-ku-ke-ko.wav", ["ki"])[0].phone_indices == (3, 4)


def test_explicit_occurrences_and_synonyms_do_not_consume_slots():
    plans = bind("ka-ka-ka.wav", ["ka3", "か#2", "ka2C4", "ka2_C4", "ka@2", "-ka"])
    assert [p.phone_indices for p in plans] == [(4, 5), (2, 3), (2, 3), (2, 3), (2, 3), (0, 1)]
    assert plans[-1].role == "cv_head"
    # Explicit phone-pair and transition ordinals must include every occurrence.
    assert bind("ka-ka-ka.wav", ["k a3"])[0].phone_indices == (4, 5)
    assert bind("ka-ka-ka.wav", ["a k2"])[0].phone_indices == (3, 4)


@pytest.mark.parametrize("alias,reason", [("ka4", "unmapped_alias"),
                                         ("ka0", "unmapped_alias"),
                                         ("zu", "unmapped_alias")])
def test_missing_aliases_and_out_of_range_ordinals_do_not_use_row_position(alias, reason):
    with pytest.raises(VoicebankReviewRequired) as error:
        bind("ka-ka-ka.wav", [alias])
    assert error.value.reason == reason
    detail = json.loads(error.value.detail)
    assert detail["alias"] == alias


def test_no_phantom_trailing_consonant_and_mid_file_head_uses_first_occurrence():
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        bind("n_ba_bi_bu.wav", ["u bE4"], language="korean", format_type="cvvc")
    # User policy 2026-10-08 (supersedes 09-21): a head may name a mid-file syllable.
    head = bind("n_ba_bi_bu.wav", ["- biE4"], language="korean", format_type="cvvc")[0]
    assert head.template == "builtin:cv_head" and head.role == "cv_head"


def test_named_breath_variant_is_not_an_occurrence_ordinal():
    assert bind("_br2.wav", ["br2"], language="korean", format_type="cvvc")[0].phone_indices == (0,)
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        bind("_br2.wav", ["br3"], language="korean", format_type="cvvc")


@pytest.mark.parametrize("base", ["a", "L", "NG"])
def test_fry_prefix_and_suffix_bind_distinct_adjacent_events(base):
    plans = bind(f"_{base}(vf).wav", [f"*{base}", f"{base} *"], language="korean", format_type="cvvc")
    assert [p.phone_indices for p in plans] == [(0, 1), (1, 2)]
    assert plans[1].template == "builtin:ending_exhalation"
    assert plans[0].template == ("builtin:cv" if base == "a" else "builtin:fry_to_sustain")


def test_glottal_marker_only_in_tagged_recordings():
    plans = bind("_eu'a'i'eu(cl).wav", ["eu '", "'a", "a '", "'eu"], language="korean", format_type="cvvc")
    assert [p.phone_indices for p in plans] == [(1, 2), (2, 3), (3, 4), (0, 1)]
    assert bind("_eu'a'i'eu(cl).wav", ["'eu2"], language="korean", format_type="cvvc")[0].phone_indices == (6, 7)
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        bind("_eu'a.wav", ["'a"], language="korean", format_type="cvvc")


def test_semivowel_alias_includes_onset_and_vowel_transition():
    plans = bind("_beu'bwa'bwi'bo.wav", ["eu bw", "a bw", "i bw", "bwa"], language="korean", format_type="cvvc")
    assert [p.phone_indices for p in plans] == [(1, 2, 3), (3, 4, 5), (5, 6, 7), (2, 3)]
    assert all(p.template == "builtin:vc_semivowel" for p in plans[:3])
    assert bind("_eu'wa'wi.wav", ["a w"], language="korean", format_type="cvvc")[0].phone_indices == (1, 2)


def test_uppercase_coda_is_not_silence_and_h_is_exhalation():
    assert parse_filename_phone_slots("_siL.wav", format_hint="cvvc", language_hint="korean") == (("s", "i", "l"),)
    assert parse_filename_phone_slots("_sil.wav", format_hint="cvvc", language_hint="korean") == (("sil",),)
    assert parse_filename_phone_slots("_euNH.wav", format_hint="cvvc", language_hint="korean") == (("eu", "n", "exh"),)
    assert bind("_euNH.wav", ["N H"], language="korean", format_type="cvvc")[0].phone_indices == (1, 2)


def test_sonorant_release_and_korean_liquid_spelling():
    assert bind("_euN.wav", ["N R", "N -"], language="korean", format_type="cvvc")[0].phone_indices == (1, 2)
    assert bind("_reuL'raL.wav", ["la"], language="korean", format_type="cvvc")[0].phone_indices == (3, 4)


@pytest.mark.parametrize("vowel,glide", [("eu", ""), ("i", "y"), ("u", "w")])
def test_confirmed_korean_r_l_synonyms_keep_same_occurrence_and_template(vowel, glide):
    filename = f"_r{vowel}NG'r{vowel}N'r{vowel}M'r{vowel}L'r{vowel}.wav"
    aliases = [f"{coda} {liquid}{glide}" for coda in ("NG", "N", "M", "L") for liquid in ("r", "l")]
    plans = bind(filename, aliases, language="korean", format_type="cvvc")
    for r, l in zip(plans[::2], plans[1::2]):
        assert r.phone_indices == l.phone_indices
        assert r.template == l.template
    assert bind(filename, aliases[::-1], language="korean", format_type="cvvc") == plans[::-1]
    # Explicit numbering also has the same meaning for either spelling.
    r, l = bind("reu_reu_reu.wav", ["eu r#2", "eu l#2"], language="korean", format_type="cvvc")
    assert r.phone_indices == l.phone_indices == (3, 4)


def test_coda_case_is_distinct_and_explicit_r_includes_exhalation(tmp_path):
    plans = bind("a-kka.wav", ["a k", "a K"], language="korean", format_type="cvvc")
    assert [p.phone_indices for p in plans] == [(0, 1), (0, 1)]
    assert [p.template for p in plans] == ["builtin:vc", "builtin:vc_hard"]
    plans = bind("aN.wav", ["a n", "a N"], language="korean", format_type="cvvc")
    assert [p.template for p in plans] == ["builtin:vc", "builtin:vc_soft_voiced"]
    # Ordinals address vowel occurrences even if only the requested occurrence
    # supplies breath evidence. '-' excludes breath in the same recording.
    plans = bind("a-i-a.wav", ["a R2", "a -2"], language="korean", format_type="cvvc")
    assert [p.phone_indices for p in plans] == [(2, 3), (2,)]
    assert plans[0].template == "builtin:ending_exhalation"
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        bind("a-na.wav", ["a N"], language="korean", format_type="cvvc")
    wav = tmp_path / "a-i-a.wav"
    wav.write_bytes(b"builder hash-only input")
    built = build_routed_voicebank_project(bank=tmp_path, wavs=[wav],
        base_rows_by_wav={wav.name: [SourceOtoRow(wav.name, "a R", 0)]},
        format_hint="cvvc", language_hint="korean", recursive=False, output_encoding="utf-8")
    assert built.project.recordings[0].phones == ("a", "i", "a", "exh")
    assert built.project.recordings[0].metadata["alias_inferred_exhalation_indices"] == [3]
    assert built.manifest["recordings"][0]["alias_inferred_exhalation_indices"] == [3]


def test_vcv_context_selects_repeated_cv_without_consuming_it():
    plans = bind("a-ka-i-ka.wav", ["i ka", "a ka", "ka2", "a k"], format_type="vcv")
    assert [p.phone_indices for p in plans] == [(3, 4, 5), (0, 1, 2), (4, 5), (0, 1)]
    assert bind("a-ka-i-ka.wav", ["ka"], format_type="vcv")[0].phone_indices == (1, 2)


def test_vv_and_release_candidates_remain_occurrence_aware():
    plans = bind("a-i-a.wav", ["i a", "a i", "a R2"])
    assert [p.phone_indices for p in plans] == [(1, 2), (0, 1), (2,)]
    assert bind("a-i-a.wav", ["a R"])[0].phone_indices == (2,)


def test_explicit_ordinal_precedes_edge_meaning_and_default_first():
    expected = {"a": (0,), "- a": (0,), "a R": (2,), "a -": (0,),
                "a R1": (0,), "a -#1": (0,), "a R@2": (2,)}
    for order in (list(expected), list(reversed(expected))):
        assert {p.alias: p.phone_indices for p in bind("a-i-a.wav", order)} == expected
    # A final matching vowel may precede another vowel or an explicit breath.
    assert bind("a-i-a-u.wav", ["a R"])[0].phone_indices == (2,)
    assert bind("a-i-a_R.wav", ["a R"], language="korean", format_type="cvvc")[0].phone_indices == (2, 3)
    assert bind("_euN_euN.wav", ["N R", "N R1"], language="korean", format_type="cvvc")[0].phone_indices == (4, 5)


def test_mora_pair_does_not_hide_a_later_matching_onset_pair():
    assert bind("んあな.wav", ["n a"], format_type="vcv")[0].phone_indices == (0, 1)
    plans = bind("んあな.wav", ["n a1", "n a2"], format_type="vcv")
    assert [(p.phone_indices, p.role) for p in plans] == [((0, 1), "vv"), ((2, 3), "cv")]


def test_korean_compact_pair_contract_survives_shuffle_and_sparse_rows():
    expected = {"dda": (0, 1), "at": (1, 2), "ddi": (4, 5), "it": (5, 6)}
    for order in itertools.permutations(expected):
        plans = bind("ddadda_ddiddi.wav", order, language="korean", format_type="cvvc")
        assert {p.alias: p.phone_indices for p in plans} == expected
    assert bind("ddadda_ddiddi.wav", ["ddi"], language="korean", format_type="cvvc")[0].phone_indices == (4, 5)
    assert bind("dda_dda_ddi_ddi.wav", ["dda"], language="korean", format_type="cvvc")[0].phone_indices == (0, 1)


def test_routed_manifest_records_bindings_and_preserves_generated_kana(tmp_path):
    # Project construction reads hashes, not samples; no acoustic test fixture needed.
    wav = tmp_path / "あかき.wav"
    wav.write_bytes(b"hash-only builder input")
    kwargs = dict(bank=tmp_path, wavs=[wav], format_hint="cv", language_hint="japanese",
                  recursive=False, output_encoding="utf-8")
    generated = build_routed_voicebank_project(base_rows_by_wav={}, **kwargs)
    assert [p.alias for p in generated.project.recordings[0].aliases] == ["あ", "か", "き"]
    built = build_routed_voicebank_project(base_rows_by_wav={wav.name: [SourceOtoRow(wav.name, "き", 0)]}, **kwargs)
    record = built.manifest["recordings"][0]
    assert record["alias_binding_policy"] == "filename_semantics_v4_coda_exhalation"
    assert record["aliases"][0]["phones"] == [3, 4]
    assert record["aliases"][0]["phone_occurrence_ids"] == ["k#0002", "i#0001"]


def test_generation_aligns_once_and_reordering_base_rows_preserves_oto(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from universal_auto_oto import project_runtime
    from universal_auto_oto.generator import GenerationOptions, generate_voicebank
    from universal_auto_oto.cli import validate_oto

    bank = tmp_path / "bank"
    bank.mkdir()
    rate = 16000
    vowel = .25 * np.sin(2 * np.pi * 220 * np.arange(6400) / rate)
    silence = np.zeros(1280)
    sf.write(bank / "a-ka-ki.wav", np.concatenate([silence, vowel, silence, vowel, silence, vowel, silence]), rate)
    original = project_runtime.align_recording
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1].wav)
        return original(*args, **kwargs)

    monkeypatch.setattr(project_runtime, "align_recording", counted)
    rows_by_run = []
    for run, aliases in enumerate((("ki", "ka", "a ka", "a k1"), ("a k1", "a ka", "ka", "ki"))):
        base = tmp_path / f"source_{run}.ini"
        base.write_text("\n".join(f"a-ka-ki.wav={a},{900+run},1,-2,3,4" for a in aliases), encoding="utf-8")
        output = tmp_path / f"output_{run}"
        generate_voicebank(GenerationOptions(bank=bank, output=output, base_oto=base,
                                            backend="routed_hsmm", format_hint="cv", language_hint="japanese"))
        assert validate_oto(output / "oto.ini", bank)["valid"]
        values = [line.split("=", 1)[1].split(",", 1) for line in (output / "oto.ini").read_text().splitlines()]
        rows_by_run.append(dict(values))
        reports = [json.loads(line) for line in (output / "row_report.jsonl").read_text().splitlines()]
        assert len({r["timeline_source_sha256"] for r in reports}) == 1
    assert calls == ["a-ka-ki.wav", "a-ka-ki.wav"]
    assert rows_by_run[0] == rows_by_run[1]


def test_korean_vcv_diphthong_left_side_binds_by_nucleus():
    # bya'bya'bye: the second and third VCV rows are written "a bya", "a bye".
    plans = bind("_bya'bya'bye.wav", ["- bya", "a bya", "a bye"], language="korean", format_type="vcv")
    assert [(p.alias, p.phone_indices, p.role) for p in plans] == [
        ("- bya", (0, 1), "cv_head"), ("a bya", (1, 2, 3), "vcv"), ("a bye", (3, 4, 5), "vcv")]


def test_korean_vcv_tense_onset_respelling_binds_same_occurrence():
    plans = bind("_bba'bbeo.wav", ["a ppeo", "a bbeo"], language="korean", format_type="vcv")
    assert plans[0].phone_indices == plans[1].phone_indices == (1, 2, 3)


def test_korean_vcv_compact_names_are_segmented_by_the_bank_aliases():
    from universal_auto_oto.voicebank_project_adapter import _segment_korean_vcv_compact

    aliases = ["- ga", "a ga", "a geo", "eo ga", "a ge", "e ga", "a go", "o ga", "a k"]
    assert _segment_korean_vcv_compact("_gagageogagegagogaK.wav", aliases) == (
        ("g", "a"), ("g", "a"), ("g", "eo"), ("g", "a"), ("g", "e"), ("g", "a"), ("g", "o"), ("g", "a", "k"))
    # A diphthong is one unit when the aliases name it (ddui, not ddu+i), and a
    # mid-file sonorant coda becomes its own syllabic slot.
    assert _segment_korean_vcv_compact("_dduidduiddwa.wav", ["- ddui", "i ddui", "i ddwa"]) == (
        ("dd", "ui"), ("dd", "ui"), ("dd", "wa"))
    assert _segment_korean_vcv_compact("bbeubbweoN'bbeo.wav", ["eu bbweo", "n bbeo"]) == (
        ("bb", "eu"), ("bb", "weo"), ("n",), ("bb", "eo"))
    assert _segment_korean_vcv_compact("_xyz.wav", ["- ga"]) == ()


def test_korean_vcv_coda_and_release_rows_bind_to_the_file_end():
    from universal_auto_oto.voicebank_project_adapter import _segment_korean_vcv_compact

    for name, aliases, expected in [
        ("_nananeonaN.wav", ["- na", "a na", "a neo", "eo na", "a n"], (7, 8)),
        ("_ggoggogguggo-.wav", ["- ggo", "o ggo", "o ggu", "u ggo", "o -"], (7,)),
    ]:
        _, slots = _flatten_slots(_segment_korean_vcv_compact(name, aliases))
        rows = [SourceOtoRow(name, a, i) for i, a in enumerate(aliases)]
        plans = _planned_aliases(rows, slots, "vcv", name, language_hint="korean")
        assert plans[-1].phone_indices == expected


def test_korean_cvc_glued_codas_liquid_spelling_breath_and_held_vowels():
    plans = {p.alias: p.phone_indices for p in bind("ang'ing'ang.wav", ["ang", "ngi", "nga"], language="korean",
                                                  format_type="cvc")}
    assert plans == {"ang": (0, 1), "ngi": (1, 2), "nga": (3, 4)}
    plans = {p.alias: p.phone_indices for p in bind("ral'rwal.wav", ["la", "al", "lwa"], language="korean",
                                                  format_type="cvc")}
    assert plans == {"la": (0, 1), "al": (1, 2), "lwa": (3, 4)}
    assert parse_filename_phone_slots("eunR_eumR.wav", format_hint="cvc", language_hint="korean") == (
        ("eu", "n", "exh"), ("eu", "m", "exh"))
    assert bind("eunR_eumR.wav", ["m R"], language="korean", format_type="cvc")[0].phone_indices == (4, 5)
    assert parse_filename_phone_slots("aLong.wav", format_hint="cvc", language_hint="korean") == (("a",), ("a",))
    assert bind("aLong.wav", ["a a"], language="korean", format_type="cvc")[0].phone_indices == (0, 1)
