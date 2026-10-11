from universal_auto_oto.domain import SourceOtoRow
from universal_auto_oto.planning import infer_slot_count, normalize_wav_key, plan_aliases


def test_filename_separator_normalization_matches_base_oto():
    assert normalize_wav_key("ga'gi'gu.wav") == normalize_wav_key("ga_gi_gu.wav")


def test_filename_normalization_preserves_japanese_characters():
    assert normalize_wav_key("_あ・い.wav") == "あい"
    assert normalize_wav_key("か.wav") != normalize_wav_key("き.wav")


def test_two_variants_per_transition_map_to_seven_slots():
    aliases = ["ga", "a g", "gi", "i g", "gu", "u g", "ge", "e g", "go", "o g", "geu", "eu g", "geo", "eo g"]
    rows = [SourceOtoRow("ga'gi'gu'ge'go'geu'geo'ga.wav", alias, index) for index, alias in enumerate(aliases)]
    plan = plan_aliases(rows, slot_count=8)
    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert [item.role for item in plan[:4]] == ["cv", "vc", "cv", "vc"]


def test_three_variants_distinguish_cvc_and_vc():
    rows = [
        SourceOtoRow("nan'nin.wav", "na", 0),
        SourceOtoRow("nan'nin.wav", "an", 1),
        SourceOtoRow("nan'nin.wav", "a n", 2),
    ]
    plan = plan_aliases(rows, slot_count=2)
    assert [item.role for item in plan] == ["cv", "cvc", "vc"]


def test_cv_leading_guide_slot_is_not_assigned_an_alias():
    rows = [SourceOtoRow("a-ba-bi-bu.wav", alias, index) for index, alias in enumerate(("ba", "bi", "bu"))]
    plan = plan_aliases(rows, slot_count=4, format_hint="cv")
    assert [item.slot_index for item in plan] == [1, 2, 3]


def test_cv_repeated_leading_guide_uses_surface_order_without_oto_timing():
    wav_name = "a-a-i-u.wav"
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(("あ", "い", "う"))]

    plan = plan_aliases(rows, slot_count=4, format_hint="cv", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 2, 3]
    assert [item.role for item in plan] == ["cv", "cv", "cv"]


def test_cv_distinct_leading_guide_keeps_first_alias_on_second_slot():
    wav_name = "a-ba-bi-bu.wav"
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(("ば", "び", "ぶ"))]

    plan = plan_aliases(rows, slot_count=4, format_hint="cv", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [1, 2, 3]


def test_vcv_rows_map_to_transitions_and_tail_release():
    aliases = ("- あ", "a あ", "a い", "i あ", "a -")
    rows = [SourceOtoRow("_ああいあ.wav", alias, index) for index, alias in enumerate(aliases)]
    assert infer_slot_count("_ああいあ.wav", rows, "vcv") == 4
    plan = plan_aliases(rows, slot_count=4, format_hint="vcv")
    assert [item.slot_index for item in plan] == [0, 0, 1, 2, 3]
    assert [item.role for item in plan] == ["cv_head", "vcv", "vcv", "vcv", "release"]


def test_grouped_cv_and_cv_head_blocks_map_to_same_slots():
    aliases = ("a4", "i4", "u4", "- a4", "- i4", "- u4")
    rows = [SourceOtoRow("a_i_u.wav", alias, index) for index, alias in enumerate(aliases)]
    plan = plan_aliases(rows, slot_count=3, format_hint="cvc")
    assert [item.slot_index for item in plan] == [0, 1, 2, 0, 1, 2]
    assert [item.role for item in plan] == ["cv", "cv", "cv", "cv_head", "cv_head", "cv_head"]


def test_interleaved_cv_vc_then_cv_head_block_maps_by_slot():
    aliases = ("ga4", "a g4", "gi4", "i g4", "gu4", "u g4", "- ga4", "- gi4", "- gu4")
    rows = [SourceOtoRow("gaga_gigi_gugu.wav", alias, index) for index, alias in enumerate(aliases)]
    plan = plan_aliases(rows, slot_count=3, format_hint="cvc")
    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 2, 0, 1, 2]
    assert [item.role for item in plan] == ["cv", "cvc_vc", "cv", "cvc_vc", "cv", "cvc_vc", "cv_head", "cv_head", "cv_head"]


def test_cv_head_block_then_interleaved_cv_compact_cvc_maps_by_slot():
    aliases = ("- gga4", "- ggi4", "- ggu4", "gga4", "ak4", "ggi4", "ik4", "ggu4", "uk4")
    rows = [SourceOtoRow("ggagga_ggiggi_gguggu.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvc")

    assert [item.slot_index for item in plan] == [0, 1, 2, 0, 0, 1, 1, 2, 2]
    assert [item.role for item in plan] == ["cv_head", "cv_head", "cv_head", "cv", "cvc", "cv", "cvc", "cv", "cvc"]


def test_interleaved_cv_compact_cvc_then_cv_head_block_maps_by_slot():
    aliases = ("la4", "al4", "li4", "il4", "lu4", "ul4", "- la4", "- li4", "- lu4")
    rows = [SourceOtoRow("lala_lili_lulu.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvc")

    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 2, 0, 1, 2]
    assert [item.role for item in plan] == ["cv", "cvc", "cv", "cvc", "cv", "cvc", "cv_head", "cv_head", "cv_head"]


def test_interleaved_cvc_then_trailing_vowel_aliases_keep_timeline_slots():
    aliases = ("ha", "a h", "hi", "i h", "hu", "u h", "a", "i", "u")
    rows = [SourceOtoRow("ha'hi'hu'a.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=4, format_hint="cvc")

    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 2, 0, 1, 2]
    assert [item.role for item in plan] == ["cv", "vc", "cv", "vc", "cv", "vc", "cv", "cv", "cv"]


def test_cvvc_role_blocks_ignore_source_oto_written_order():
    aliases = ["a k_S", "i k_S", "u k_S", "e k_S", "o k_S", "n k_S", "- k_S", "き_S", "く_S", "け_S", "こ_S", "か_S"]
    rows = [SourceOtoRow("_ka-ki-ku-ke-ko-ka-N-ka.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=8, format_hint="cvvc")

    assert [item.slot_index for item in plan[:6]] == [0, 1, 2, 3, 4, 5]
    assert [item.role for item in plan[:6]] == ["vc"] * 6
    assert plan[6].role == "cv_head"
    assert plan[6].slot_index == 1
    assert [item.slot_index for item in plan[7:]] == [1, 2, 3, 4, 5]
    assert [item.role for item in plan[7:]] == ["cv"] * 5


def test_cvvc_vowel_transitions_are_vv_rows():
    aliases = ["- a", "a a", "a e", "e a", "a i", "i a", "a o"]
    rows = [SourceOtoRow("_a'a'e'a'i'a'o.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=7, format_hint="cvvc")

    assert [item.role for item in plan] == ["cv_head", "vv", "vv", "vv", "vv", "vv", "vv"]
    assert [item.slot_index for item in plan] == [0, 1, 2, 3, 4, 5, 6]


def test_lowercase_r_transition_is_vc_not_release():
    rows = [
        SourceOtoRow("_ra-ri-ru.wav", "a r_S", 0),
        SourceOtoRow("_ra-ri-ru.wav", "i r_S", 1),
        SourceOtoRow("_ra-ri-ru.wav", "a R", 2),
    ]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc")

    assert [item.role for item in plan] == ["vc", "vc", "release"]
    assert [item.slot_index for item in plan] == [0, 1, 2]


def test_cvvc_numeric_uppercase_release_maps_to_the_terminal_slot():
    rows = [SourceOtoRow("ya_ye_yeo.wav", "eo R32G3", 0)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc", wav_name="ya_ye_yeo.wav")

    assert [(item.slot_index, item.role) for item in plan] == [(2, "release")]


def test_cvvc_tail_to_head_vc_block_reverses_only_when_filename_surfaces_confirm_it():
    wav_name = "ya_ye_yeo_yo_yu_eui_wa_we_wi_weo_weo.wav"
    aliases = ("eo wG3", "i wG3", "e wG3", "a wG3", "eu yG3", "u yG3", "o yG3", "eo yG3", "e yG3", "a yG3")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=11, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == list(reversed(range(10)))
    assert [item.role for item in plan] == ["vc"] * 10


def test_cvvc_numbered_repeated_aliases_bind_to_the_named_occurrence_not_source_order():
    wav_name = "ya_ye_yeo_yo_yu_eui_wa_we_wi_weo_weo.wav"
    rows = [
        SourceOtoRow(wav_name, "weo2G3", 0),
        SourceOtoRow(wav_name, "weoG3", 1),
    ]

    plan = plan_aliases(rows, slot_count=11, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [10, 9]


def test_cvvc_leading_n_coda_does_not_create_a_phantom_planning_slot():
    wav_name = "n_ba_bi_bu.wav"
    rows = [
        SourceOtoRow(wav_name, "baE4", 0),
        SourceOtoRow(wav_name, "biE4", 1),
    ]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc", wav_name=wav_name)

    assert [(item.slot_index, item.role) for item in plan] == [(0, "cv"), (1, "cv")]


def test_release_rows_match_their_unique_filename_vowel_slots():
    wav_name = "aH_iH_uH_eH_oH_euH_eoH.wav"
    aliases = ("a R", "i R", "u R", "e R", "o R", "eu R", "eo R")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=7, format_hint="cvvc", wav_name=wav_name)

    assert [item.role for item in plan] == ["release"] * 7
    assert [item.slot_index for item in plan] == list(range(7))


def test_ambiguous_release_vowel_keeps_safe_final_slot_fallback():
    wav_name = "aH_aH_iH.wav"
    rows = [SourceOtoRow(wav_name, "a R", 0)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc", wav_name=wav_name)

    assert plan[0].role == "release"
    assert plan[0].slot_index == 2


def test_release_rows_match_unique_coda_owner_before_h_marker():
    wav_name = "nH_mH_ngH_lH.wav"
    aliases = ("n R", "m R", "ng R", "l R")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=4, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 1, 2, 3]
    assert [item.role for item in plan] == ["release"] * 4


def test_explicit_final_release_filename_token_keeps_release_aliases_on_final_slot():
    wav_name = "_NG'R.wav"
    aliases = ("ng R", "NG R")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=2, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [1, 1]
    assert [item.role for item in plan] == ["release", "release"]


def test_cvvc_kana_filename_infers_mora_slots_without_separators():
    rows = [SourceOtoRow("_あかさたがざだ.wav", "a k", 0)]

    assert infer_slot_count("_あかさたがざだ.wav", rows, "cvvc") == 7
    assert infer_slot_count("_うきゅしゅちゅぎゅじゅてゅ.wav", rows, "cvvc") == 7


def test_numbered_compact_cv_reclist_infers_up_to_four_slots():
    aliases = ("am", "a m", "ap", "a p", "ab", "a b")
    rows = [SourceOtoRow("am97.wav", alias, index) for index, alias in enumerate(aliases)]

    assert infer_slot_count("am97.wav", rows, "cv") == 4
    assert infer_slot_count("a36.wav", rows[:3], "cv") == 3


def test_cv_then_vc_blocks_map_independently_of_written_order():
    aliases = ("gaV", "giV", "guV", "a gV", "i gV", "u gV", "がV", "ぎV")
    rows = [SourceOtoRow("gaga_gigi_gugu.wav", alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvc")

    assert [item.slot_index for item in plan] == [0, 1, 2, 0, 1, 2, 0, 2]
    assert [item.role for item in plan] == ["cv", "cv", "cv", "cvc_vc", "cvc_vc", "cvc_vc", "cv", "cv"]


def test_filename_semantics_maps_sparse_romanized_vc_occurrences():
    aliases = ("i kA3", "i sA3", "i tA3")
    wav_name = "_i-ka-i-sa-i-ta.wav"
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=6, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 2, 4]
    assert [item.role for item in plan] == ["vc", "vc", "vc"]


def test_filename_semantics_maps_apostrophe_repeated_cvvc_groups():
    wav_name = "_a'a'_'ya'ya'_'wa'wa.wav"
    aliases = ("- a", "- ya", "- wa", "a a", "a ya", "a wa")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=6, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 2, 4, 0, 2, 4]
    assert [item.role for item in plan] == ["cv_head", "cv_head", "cv_head", "vv", "vv", "vv"]


def test_filename_semantics_maps_roman_cv_block_and_kana_alternates():
    wav_name = "bya_bye_byeo_byo_byu_beui_bwa_bwe_bwi_bweo.wav"
    aliases = (
        "byaV", "byeV", "byeoV", "byoV", "byuV", "beuiV", "bwaV", "bweV", "bwiV", "bweoV",
        "\u3073\u3083V", "\u3073\u3047V", "\u3073\u3087V", "\u3073\u3085V",
        "\u3076\u3041V", "\u3076\u3047V", "\u3076\u3043V", "\u3076\u3049V",
    )
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=10, format_hint="cvc", wav_name=wav_name)

    assert [item.slot_index for item in plan[:10]] == list(range(10))
    assert [item.slot_index for item in plan[10:]] == [0, 1, 3, 4, 6, 7, 8, 9]
    assert all(item.role == "cv" for item in plan)


def test_filename_semantics_maps_repeated_cvc_transitions_to_current_slot():
    wav_name = "baba_bibi_bubu_bebe_bobo_beubeu_beobeo.wav"
    aliases = ("baV", "biV", "buV", "beV", "boV", "beuV", "beoV", "a bV", "i bV", "u bV", "e bV", "o bV", "eu bV", "eo bV")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=7, format_hint="cvc", wav_name=wav_name)

    assert [item.slot_index for item in plan[:7]] == list(range(7))
    assert [item.slot_index for item in plan[7:]] == list(range(7))
    assert [item.role for item in plan[7:]] == ["cvc_vc"] * 7


def test_verified_compact_cvvc_coda_pairs_stay_with_their_cv_slots():
    wav_name = "bbabba_bbibbi_bbubbu_bbebbe_bbobbo_bbeubbeu_bbeobbeo.wav"
    aliases = ("bba", "ap", "bbi", "ip", "bbu", "up", "bbe", "ep", "bbo", "op", "bbeu", "eup", "bbeo", "eop")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=7, format_hint="cvvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert [item.role for item in plan] == ["cv", "vc"] * 6 + ["cv", "cv"]


def test_compact_cvc_coda_pair_correction_stays_disabled_without_no_worse_evidence():
    wav_name = "lala_lili_lulu.wav"
    aliases = ("la", "al", "li", "il", "lu", "ul")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvc", wav_name=wav_name)

    assert [item.slot_index for item in plan] == [0, 0, 1, 1, 2, 1]
    assert [item.role for item in plan] == ["cv", "cvc", "cv", "cv", "cv", "vc"]


def test_unverified_interleaved_rows_keep_existing_fallback():
    wav_name = "la_li_lu.wav"
    aliases = ("la", "alt", "li", "alt", "lu", "alt")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc", wav_name=wav_name)

    assert [item.role for item in plan] != ["cv", "vc"] * 3


def test_interleaved_vowel_to_vowel_rows_are_not_reclassified_as_codas():
    wav_name = "a_i_u.wav"
    aliases = ("a", "a e", "i", "i a", "u", "u o")
    rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    plan = plan_aliases(rows, slot_count=3, format_hint="cvvc", wav_name=wav_name)

    assert any(item.role == "vv" for item in plan)
