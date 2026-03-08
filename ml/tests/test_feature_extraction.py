import os
import sys
import json
import tempfile
import unittest
import wave
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import (
    FEATURE_NAMES,
    _normalize_alias_for_match,
    canonicalize_feature_row,
    dataset_fieldnames,
    extract_feature_rows,
    get_feature_schema,
    parse_oto_rows,
)
from core.ja_oto_generator import (
    _adaptive_ja_overlap,
    _collect_phone_tier_quality,
    _clamp_ja_bridge_overlap,
    _compute_vcv_params_from_virtual_split,
    _convert_ja_internal_cutoff_to_oto_field,
    _find_ja_cv_vowel_match_index,
    _find_ja_exact_target_index,
    _ja_is_n_bridge_alias,
    _limit_pre_anchor_shift,
    _prefer_vcv_candidate_index,
    _refine_ja_vc_with_adjacent_cv,
    _sanitize_ja_internal_params_for_wav_duration,
    _recenter_ja_params_around_pre,
    sanitize_ja_oto_for_wav_duration,
    _select_ja_cv_syllable_index,
    _select_vcv_syllable_index,
    detect_ja_alias_format,
    normalize_key as normalize_ja_key,
)
from core.ja_lab_generator import parse_ja_filename, repair_japanese_mojibake_text
from core.ja_oto_mapping import _extract_ja_cv_targets_from_lines
from core.ja_oto_postprocess import guard_ja_cv_cutoff_to_next_onset
from core.lab_generator import _parse_filename, _split_kr_lab_content_tokens
from core.oto_generator import _apply_soft_mel_offset_cutoff_guard
from core.oto_generator import _recenter_kr_params_around_pre
from core.oto_generator import _compute_vc_from_adjacent_cv
from core.oto_generator import _compute_kr_cvvc_vc_timing_direct
from core.oto_generator import _compute_kr_cvvc_vv_timing_direct
from core.oto_generator import _collect_kr_phone_tier_quality
from core.oto_generator import _build_kr_cvvc_occurrence_map
from core.oto_generator import _build_kr_cvvc_vv_occurrence_map
from core.oto_generator import _build_kr_syllables_from_phone_nuclei
from core.oto_generator import _extract_kr_cv_targets_from_filename
from core.oto_generator import _refine_kr_bridge_with_adjacent_cv
from core.oto_generator import _guard_kr_vc_cutoff_to_next_segment
from core.oto_generator import _kr_alias_contains_coda
from core.oto_generator import _resolve_kr_cvvc_occurrence_index
from core.oto_generator import _resolve_kr_cvvc_vv_index
from core.oto_generator import _select_kr_cv_onset_slice
from core.oto_generator import _should_allow_kr_exact_vowel_fix
from core.oto_generator import _should_prefer_alias_based_syllables
from core.oto_generator import _synthesize_kr_word_phones
from core.oto_generator import _uses_kr_vc_context
from core.oto_generator import _cv_match_score
from core.oto_generator import classify_alias, detect_alias_format
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.prefix_map_utils import find_prefix_map_path, strip_prefix_map_affixes
from scripts.analyze_oto_diff import analyze as analyze_oto_diff


class FeatureExtractionTests(unittest.TestCase):
    def test_schema_contains_feature_names(self):
        schema = get_feature_schema()
        self.assertEqual(schema["feature_names"], FEATURE_NAMES)

    def test_dataset_fieldnames_include_label_source_and_weight(self):
        fields = dataset_fieldnames()
        self.assertIn("label_source", fields)
        self.assertIn("sample_weight", fields)

    def test_canonicalize_feature_row_defaults(self):
        row = canonicalize_feature_row({"language": "korean", "base_offset": 123.4})
        self.assertEqual(row["language"], "korean")
        self.assertAlmostEqual(row["base_offset"], 123.4)
        self.assertIn("alias_type", row)
        self.assertIn("energy_mean", row)

    def test_extract_feature_rows_without_tg_or_wav(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n")
            rows = extract_feature_rows("korean", oto_path, tg_dir=td, wav_dir=td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["wav"], "test.wav")
            self.assertEqual(rows[0]["language"], "korean")

    def test_alias_match_normalization_strips_pitch_suffix(self):
        self.assertEqual(_normalize_alias_for_match("a k_C4", language="japanese"), "a k")
        self.assertEqual(_normalize_alias_for_match("- ka-D4", language="japanese"), "- ka")
        self.assertEqual(_normalize_alias_for_match("ri F4", language="japanese"), "ri")

    def test_alias_match_normalization_strips_style_suffix(self):
        self.assertEqual(_normalize_alias_for_match("ga_soft", language="japanese"), "ga")
        self.assertEqual(_normalize_alias_for_match("a か連", language="japanese"), "a ka")
        self.assertEqual(_normalize_alias_for_match("lyeo부드러움", language="korean"), "lyeo")

    def test_japanese_format_prefers_cvvc_or_cv(self):
        self.assertEqual(detect_ja_alias_format(["- か", "か", "き"]), "cv")
        self.assertEqual(detect_ja_alias_format(["a k", "ka", "a i"]), "cvvc")
        self.assertEqual(detect_ja_alias_format([f"a {'か'}", f"i {'き'}"]), "vcv")

    def test_parse_ja_filename_keeps_small_vowel_combo_as_single_mora(self):
        self.assertEqual(
            parse_ja_filename("すぃすぃすすぃせすぃそ.wav"),
            ["si", "si", "su", "si", "se", "si", "so"],
        )

    def test_parse_ja_filename_keeps_yoon_combo_as_single_mora(self):
        self.assertEqual(
            parse_ja_filename("じぇじぇじょじぇんじぇじゃ.wav"),
            ["je", "je", "jo", "je", "n", "je", "ja"],
        )

    def test_japanese_mojibake_filename_is_repaired_before_parsing(self):
        bad = "\u3042\u3044".encode("cp932").decode("cp949")
        self.assertEqual(repair_japanese_mojibake_text(bad), "あい")
        self.assertEqual(parse_ja_filename(bad + ".wav"), ["a", "i"])

    def test_ja_normalize_key_repairs_cp949_decoded_shift_jis_filename(self):
        bad = "\u304b\u304d".encode("cp932").decode("cp949")
        self.assertEqual(normalize_ja_key(bad + ".wav"), normalize_ja_key("かき.wav"))
        self.assertEqual(normalize_wav_key("_" + bad + ".wav"), normalize_wav_key("_かき.wav"))

    def test_japanese_alias_canonicalization_unifies_romaji_and_kana(self):
        self.assertEqual(canonicalize_alias_for_matching("japanese", "か"), "ka")
        self.assertEqual(canonicalize_alias_for_matching("japanese", "カ"), "ka")
        self.assertEqual(canonicalize_alias_for_matching("japanese", "a か"), "a ka")
        self.assertEqual(canonicalize_alias_for_matching("japanese", "a ka_C4"), "a ka")

    def test_ja_normalize_key_keeps_kana(self):
        self.assertEqual(normalize_ja_key("ききくきけきこ.wav"), "ききくきけきこ")
        self.assertEqual(normalize_ja_key("_ききく-きけ_きこ.TextGrid"), "ききくきけきこ")

    def test_ja_normalize_key_unifies_katakana_hiragana(self):
        self.assertEqual(normalize_ja_key("ヴぃヴぃヴ.wav"), normalize_ja_key("ゔぃゔぃゔ.wav"))

    def test_ja_phone_quality_flags_spn_heavy_and_insufficient_phones(self):
        phones = [
            SimpleNamespace(mark="spn"),
            SimpleNamespace(mark="spn"),
            SimpleNamespace(mark="a"),
            SimpleNamespace(mark="sil"),
        ]
        q = _collect_phone_tier_quality(phones, expected_syllables=5, min_vowel_phone_ratio=0.5)
        self.assertGreaterEqual(q["spn_ratio_in_phone_tier"], 0.5)
        self.assertIn("insufficient_phones", q["low_confidence_reasons"])

    def test_ja_phone_quality_counts_known_vowels(self):
        phones = [
            SimpleNamespace(mark="k"),
            SimpleNamespace(mark="a"),
            SimpleNamespace(mark="i"),
            SimpleNamespace(mark="u"),
            SimpleNamespace(mark="sil"),
        ]
        q = _collect_phone_tier_quality(phones, expected_syllables=3, min_vowel_phone_ratio=0.5)
        self.assertEqual(q["known_vowel_phone_count"], 3)
        self.assertGreaterEqual(q["phones_vs_expected_syllables_ratio"], 1.0)

    def test_analyze_oto_diff_handles_partial_key_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "base.ini")
            auto_path = os.path.join(td, "auto.ini")
            report_path = os.path.join(td, "report.json")
            with open(base_path, "w", encoding="utf-8") as f:
                f.write("a.wav=- か,100,200,-300,120,40\n")
                f.write("a.wav=a か,300,220,-360,130,45\n")
            with open(auto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=- か,110,210,-320,125,43\n")
                f.write("b.wav=a き,320,210,-340,120,40\n")
            report = analyze_oto_diff(base_path, auto_path, language="japanese", topn=5)
            self.assertEqual(report["counts"]["shared_rows"], 1)
            self.assertEqual(report["counts"]["base_only"], 1)
            self.assertEqual(report["counts"]["auto_only"], 1)
            with open(report_path, "w", encoding="utf-8") as rf:
                json.dump(report, rf, ensure_ascii=False, indent=2)
            self.assertTrue(os.path.exists(report_path))

    def test_ja_cv_targets_preserve_consecutive_duplicates_for_vcv(self):
        lines = [
            "x.wav=- ま,0,0,0,0,0",
            "x.wav=a ま,0,0,0,0,0",
            "x.wav=a み,0,0,0,0,0",
            "x.wav=i ま,0,0,0,0,0",
            "x.wav=a む,0,0,0,0,0",
            "x.wav=u ま,0,0,0,0,0",
            "x.wav=a め,0,0,0,0,0",
        ]
        self.assertEqual(
            _extract_ja_cv_targets_from_lines(lines),
            ["ma", "ma", "mi", "ma", "mu", "ma", "me"],
        )

    def test_parse_korean_filename_ignores_apostrophe_as_phoneme(self):
        self.assertEqual(
            _parse_filename("_yu'ye'yu'yo'yu'yeo'yu'i'yu.wav", convert_to_hangul=False),
            ["yu", "ye", "yu", "yo", "yu", "yeo", "yu", "i", "yu"],
        )

    def test_split_korean_lab_content_reconstructs_broken_roman_char_stream(self):
        self.assertEqual(
            _split_kr_lab_content_tokens("y u ' y e ' y u ' y o ' y u ' y e o ' y u ' i ' y u"),
            ["yu", "ye", "yu", "yo", "yu", "yeo", "yu", "i", "yu"],
        )

    def test_parse_korean_filename_merges_coda_tail_marker(self):
        self.assertEqual(_parse_filename("_l'R.wav", convert_to_hangul=False), ["lR"])
        self.assertEqual(_parse_filename("_ng'H.wav", convert_to_hangul=False), ["ngH"])

    def test_kr_phone_quality_flags_insufficient_vowels(self):
        phones = [
            SimpleNamespace(mark="spn"),
            SimpleNamespace(mark="k"),
            SimpleNamespace(mark="spn"),
            SimpleNamespace(mark="sil"),
        ]
        q = _collect_kr_phone_tier_quality(phones, expected_syllables=4, min_vowel_phone_ratio=0.5)
        self.assertIn("insufficient_phones", q["low_confidence_reasons"])
        self.assertIn("insufficient_vowel_phones", q["low_confidence_reasons"])
        self.assertGreater(q["spn_ratio_in_phone_tier"], 0.5)

    def test_kr_synthesize_word_phones_generates_minimum_nucleus(self):
        def _dec(ch):
            return [ch]
        phones = _synthesize_kr_word_phones("가", 0.0, 0.2, _dec)
        self.assertGreaterEqual(len(phones), 1)
        # 마지막 phone은 모음핵 계열이 되도록 설계됨
        self.assertGreater(float(phones[-1].maxTime), float(phones[-1].minTime))

    def test_korean_cvvc_prefers_alias_based_syllables_even_when_words_not_far_behind(self):
        self.assertTrue(_should_prefer_alias_based_syllables("cvvc", True, 72.0, 68.0))

    def test_korean_cvvc_prefers_alias_based_syllables_unless_alias_score_collapses(self):
        self.assertTrue(_should_prefer_alias_based_syllables("cvvc", True, 80.0, 44.0))
        self.assertFalse(_should_prefer_alias_based_syllables("cvvc", True, 80.0, 22.0))

    def test_korean_non_cvvc_keeps_conservative_words_preference(self):
        self.assertFalse(_should_prefer_alias_based_syllables("cvc", True, 72.0, 68.0))

    def test_korean_batchim_bridge_alias_is_not_misclassified_as_vcv(self):
        self.assertEqual(classify_alias("l b"), "vc")
        self.assertEqual(classify_alias("ng b"), "vc")

    def test_korean_cvvc_format_detects_head_cv_vc_combo_without_vv(self):
        self.assertEqual(
            detect_alias_format(["- bya", "ba", "bya", "a b"]),
            "cvvc",
        )

    def test_korean_vv_only_format_is_detected_separately(self):
        self.assertEqual(
            detect_alias_format(["a eui", "u eui", "e eui"]),
            "vv_only",
        )
        self.assertEqual(
            detect_alias_format(["ng eui"]),
            "vv_only",
        )

    def test_korean_cv_only_format_is_detected_as_cv(self):
        self.assertEqual(
            detect_alias_format(["- ga", "ga", "gya"]),
            "cv",
        )

    def test_parse_oto_rows_ignores_hiragana_aliases_for_korean(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=あ,0,10,-20,5,2\n")
                f.write("a.wav=ga,0,10,-20,5,2\n")
            rows = parse_oto_rows(oto_path, language="korean")
            self.assertEqual([row["alias"] for row in rows], ["ga"])

    def test_prefix_map_in_parent_folder_is_found_for_subfolder_oto(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "Voice")
            sub = os.path.join(root, "C4")
            os.makedirs(sub, exist_ok=True)
            prefix_map = os.path.join(root, "prefix.map")
            with open(prefix_map, "w", encoding="utf-8") as f:
                f.write("C4\t\t_C4\n")
            oto_path = os.path.join(sub, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a_C4.wav=ga_C4,0,10,-20,5,2\n")
            found = find_prefix_map_path(oto_path)
            self.assertEqual(os.path.abspath(found), os.path.abspath(prefix_map))
            rows = parse_oto_rows(
                oto_path,
                language="korean",
                prefix_map_path=found,
                prefix_context_paths=[oto_path, sub],
            )
            self.assertEqual(rows[0]["alias"], "ga")

    def test_prefix_map_prefix_and_suffix_are_stripped_for_training_alias(self):
        with tempfile.TemporaryDirectory() as td:
            prefix_map = os.path.join(td, "prefix.map")
            with open(prefix_map, "w", encoding="utf-8") as f:
                f.write("C4\tPRE_\t_SUF\n")
            stripped = strip_prefix_map_affixes(
                "PRE_ga_SUF",
                prefix_map_path=prefix_map,
                wav_name="a_C4.wav",
                context_paths=[os.path.join(td, "C4")],
            )
            self.assertEqual(stripped, "ga")

    def test_korean_cv_match_score_penalizes_simple_vs_glide_vowel_confusion(self):
        self.assertLess(_cv_match_score("do", "dyo"), 10)
        self.assertLess(_cv_match_score("dwa", "dya"), 10)
        self.assertEqual(_cv_match_score("do", "do"), 100)

    def test_korean_cvvc_filename_targets_keep_recorded_order(self):
        self.assertEqual(
            _extract_kr_cv_targets_from_filename("_do'do'_'dyo'dyo'_'ong'do.wav"),
            ["do", "do", "dyo", "dyo", "ong", "do"],
        )

    def test_korean_cvvc_occurrence_mapping_uses_filename_order(self):
        syllables_info = [
            {"roman": "do", "roman_cv": "do", "word": "do"},
            {"roman": "do", "roman_cv": "do", "word": "do"},
            {"roman": "dyo", "roman_cv": "dyo", "word": "dyo"},
            {"roman": "dyo", "roman_cv": "dyo", "word": "dyo"},
            {"roman": "ong", "roman_cv": "ong", "word": "ong"},
            {"roman": "do", "roman_cv": "do", "word": "do"},
        ]
        occ_map = _build_kr_cvvc_occurrence_map(syllables_info)
        state = {}
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("- do", "cv_head", occ_map, state), 0)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("do", "cv", occ_map, state), 0)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("- dyo", "cv_head", occ_map, state), 2)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("dyo", "cv", occ_map, state), 2)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("do", "cv", occ_map, state), 1)

    def test_korean_cvvc_forced_occurrence_blocks_exact_vowel_fix(self):
        self.assertFalse(_should_allow_kr_exact_vowel_fix("cvvc", 3))
        self.assertFalse(_should_allow_kr_exact_vowel_fix("cvvc", 0))

    def test_korean_nonforced_occurrence_allows_exact_vowel_fix(self):
        self.assertTrue(_should_allow_kr_exact_vowel_fix("cvvc", None))
        self.assertTrue(_should_allow_kr_exact_vowel_fix("cvc", 2))

    def test_korean_cv_head_severe_mismatch_allows_exact_vowel_fix(self):
        self.assertTrue(
            _should_allow_kr_exact_vowel_fix(
                "cvvc",
                1,
                alias_type="cv_head",
                severe_vowel_mismatch=True,
            )
        )
        self.assertFalse(
            _should_allow_kr_exact_vowel_fix(
                "cvvc",
                1,
                alias_type="cv",
                severe_vowel_mismatch=True,
            )
        )

    def test_korean_anchor_target_detects_coda_for_vcv_and_cv_head(self):
        self.assertTrue(_kr_alias_contains_coda("a gak", "vcv"))
        self.assertFalse(_kr_alias_contains_coda("a ga", "vcv"))
        self.assertTrue(_kr_alias_contains_coda("- gak", "cv_head"))
        self.assertFalse(_kr_alias_contains_coda("- ga", "cv_head"))

    def test_korean_cvvc_vv_occurrence_mapping_uses_pair_order(self):
        syllables_info = [
            {"roman": "eo", "roman_cv": "eo", "word": "eo"},
            {"roman": "a", "roman_cv": "a", "word": "a"},
            {"roman": "eo", "roman_cv": "eo", "word": "eo"},
            {"roman": "i", "roman_cv": "i", "word": "i"},
            {"roman": "eo", "roman_cv": "eo", "word": "eo"},
            {"roman": "yeo", "roman_cv": "yeo", "word": "yeo"},
        ]
        occ_map = _build_kr_cvvc_vv_occurrence_map(syllables_info)
        state = {}
        self.assertEqual(_resolve_kr_cvvc_vv_index("eo a", occ_map, state), 1)
        self.assertEqual(_resolve_kr_cvvc_vv_index("a eo", occ_map, state), 2)
        self.assertEqual(_resolve_kr_cvvc_vv_index("eo i", occ_map, state), 3)
        self.assertEqual(_resolve_kr_cvvc_vv_index("i eo", occ_map, state), 4)
        self.assertEqual(_resolve_kr_cvvc_vv_index("eo yeo", occ_map, state), 5)

    def test_korean_vv_does_not_use_vc_context(self):
        self.assertFalse(_uses_kr_vc_context("vv"))
        self.assertTrue(_uses_kr_vc_context("vc"))

    def test_korean_phone_nuclei_alignment_prefers_glide_vowel_targets(self):
        ph = [
            SimpleNamespace(mark="d", minTime=0.00, maxTime=0.03),
            SimpleNamespace(mark="o", minTime=0.03, maxTime=0.10),
            SimpleNamespace(mark="d", minTime=0.10, maxTime=0.13),
            SimpleNamespace(mark="j", minTime=0.13, maxTime=0.16),
            SimpleNamespace(mark="o", minTime=0.16, maxTime=0.23),
            SimpleNamespace(mark="ng", minTime=0.23, maxTime=0.28),
            SimpleNamespace(mark="d", minTime=0.28, maxTime=0.31),
            SimpleNamespace(mark="o", minTime=0.31, maxTime=0.38),
        ]
        infos = _build_kr_syllables_from_phone_nuclei(ph, ["do", "dyo", "do"])
        self.assertEqual([row["roman_cv"] for row in infos], ["do", "dyo", "do"])

    def test_korean_phone_nuclei_alignment_prefers_yeo_over_plain_eo(self):
        ph = [
            SimpleNamespace(mark="i", minTime=0.00, maxTime=0.07),
            SimpleNamespace(mark="eo", minTime=0.07, maxTime=0.14),
            SimpleNamespace(mark="j", minTime=0.14, maxTime=0.17),
            SimpleNamespace(mark="eo", minTime=0.17, maxTime=0.25),
        ]
        infos = _build_kr_syllables_from_phone_nuclei(ph, ["i", "yeo"])
        self.assertEqual([row["roman_cv"] for row in infos], ["i", "yeo"])

    def test_korean_phone_nuclei_fallback_splits_when_nuclei_are_too_few(self):
        ph = [
            SimpleNamespace(mark="a", minTime=0.00, maxTime=0.08),
            SimpleNamespace(mark="j", minTime=0.08, maxTime=0.10),
            SimpleNamespace(mark="i", minTime=0.10, maxTime=0.18),
        ]
        infos = _build_kr_syllables_from_phone_nuclei(ph, ["a", "i", "a", "u"])
        self.assertEqual([row["roman_cv"] for row in infos], ["a", "i", "a", "u"])
        self.assertEqual(len(infos), 4)
        self.assertTrue(all(row["phones"] for row in infos))

    def test_korean_cvvc_occurrence_maps_accept_filename_target_lists(self):
        cv_occ = _build_kr_cvvc_occurrence_map(["a", "i", "a", "u", "a", "e", "a", "r"])
        vv_occ = _build_kr_cvvc_vv_occurrence_map(["a", "i", "a", "u", "a", "e", "a", "r"])
        self.assertEqual(cv_occ["a"], [0, 2, 4, 6])
        self.assertEqual(vv_occ["a i"], [1])
        self.assertEqual(vv_occ["i a"], [2])

    def test_japanese_cv_selector_prefers_real_se_over_previous_si(self):
        syllables = [{"word": s} for s in ["si", "si", "su", "si", "se", "si", "so"]]
        self.assertEqual(_select_ja_cv_syllable_index("せ", 3, syllables, alias_type="cv"), 4)

    def test_japanese_vcv_selector_prefers_real_je_combo(self):
        syllables = [{"word": s} for s in ["je", "je", "jo", "je", "n", "je", "ja"]]
        self.assertEqual(_select_vcv_syllable_index("a じぇ", 2, syllables), 3)

    def test_japanese_vcv_selector_does_not_advance_without_exact_target(self):
        syllables = [{"word": s} for s in ["te", "tye", "to"]]
        # expected=0 에서 다음 칸이 유사 발음이어도 exact target("te")가 아니면 전진하지 않는다.
        self.assertEqual(_select_vcv_syllable_index("a て", 0, syllables), 0)

    def test_japanese_vcv_selector_does_not_remap_backward(self):
        syllables = [{"word": s} for s in ["ja", "jo", "ja"]]
        self.assertEqual(_select_vcv_syllable_index("a じょ", 2, syllables), 2)

    def test_japanese_vcv_selector_does_not_overjump_more_than_one(self):
        syllables = [{"word": s} for s in ["ja", "ka", "ku", "kyo", "sa"]]
        self.assertEqual(_select_vcv_syllable_index("a きょ", 1, syllables), 1)

    def test_japanese_exact_target_resync_prefers_backward_match(self):
        syllables = [{"word": s} for s in ["ma", "mi", "mu", "me", "mo"]]
        # expected(3=me)가 앞서간 상황에서 target=mu를 정확히 찾으면 역방향 복구한다.
        self.assertEqual(_find_ja_exact_target_index("mu", 3, syllables, search_back=3, search_fwd=1), 2)

    def test_japanese_exact_target_resync_allows_forward_two_when_exact(self):
        syllables = [{"word": s} for s in ["zi", "zi", "zu", "zi", "ze", "zi", "zo"]]
        # expected(0)이 뒤처진 상황에서 target=zu exact가 +2에만 있으면 그 위치로 복구한다.
        self.assertEqual(_find_ja_exact_target_index("zu", 0, syllables, search_back=3, search_fwd=2), 2)

    def test_japanese_vowel_mismatch_recovery_for_ri_vs_ra(self):
        syllables = [{"word": s} for s in ["ra", "ri", "ru", "re"]]
        self.assertEqual(_find_ja_cv_vowel_match_index("ri", 0, syllables, search_back=2, search_fwd=3), 1)

    def test_prefer_vcv_candidate_index_allows_one_step_exact_fix(self):
        syllables = [{"word": s} for s in ["ra", "ri", "ru"]]
        # expected=ra, mapped=ri, target=ri -> 1칸 보정 허용
        self.assertEqual(_prefer_vcv_candidate_index(0, 1, "ri", syllables, max_delta=1), 1)

    def test_prefer_vcv_candidate_index_blocks_ambiguous_repeat_shift(self):
        syllables = [{"word": s} for s in ["ma", "ma", "mi"]]
        # expected/mapped 모두 target=ma 가능한 반복 구간 -> 순서 안정 우선(expected 유지)
        self.assertEqual(_prefer_vcv_candidate_index(0, 1, "ma", syllables, max_delta=1), 0)

    def test_refine_ja_vc_with_adjacent_cv_clamps_cutoff_before_next_vowel(self):
        prev_cv = {
            "vowel_end_abs": 1200.0,
            "pre": 112.0,
        }
        next_cv = {
            "onset_abs": 1260.0,
            "vowel_start_abs": 1310.0,
            "pre": 96.0,
        }
        o, c, ct, p, ov = _refine_ja_vc_with_adjacent_cv(
            1060.0, 190.0, -340.0, 150.0, 70.0,
            c_char="k",
            prev_cv_anchor=prev_cv,
            next_cv_anchor=next_cv,
            prev_v_end_abs=1200.0,
            next_c_start_abs=1260.0,
            next_c_end_abs=1310.0,
        )
        cut_abs = o + abs(ct)
        self.assertLessEqual(cut_abs, 1308.0)  # next vowel(1310) 직전
        self.assertGreaterEqual(c, p + 8.0)

    def test_refine_ja_vc_with_adjacent_cv_keeps_overlap_near_prev_vowel_tail(self):
        prev_cv = {
            "vowel_end_abs": 980.0,
            "pre": 90.0,
        }
        next_cv = {
            "onset_abs": 1035.0,
            "vowel_start_abs": 1088.0,
            "pre": 84.0,
        }
        o, c, ct, p, ov = _refine_ja_vc_with_adjacent_cv(
            860.0, 160.0, -280.0, 128.0, 52.0,
            c_char="m",
            prev_cv_anchor=prev_cv,
            next_cv_anchor=next_cv,
            prev_v_end_abs=980.0,
            next_c_start_abs=1035.0,
            next_c_end_abs=1088.0,
        )
        ovl_abs = o + ov
        self.assertGreaterEqual(ovl_abs, 960.0)
        self.assertLessEqual(ovl_abs, 1010.0)

    def test_limit_pre_anchor_shift_leaves_small_delta_unchanged(self):
        vals = _limit_pre_anchor_shift(
            860.0,
            160.0,
            -280.0,
            128.0,
            52.0,
            pre_abs_before=980.0,
            max_shift_ms=26.0,
        )
        o, c, ct, p, ov, pre_abs_after, clamped = vals
        self.assertFalse(clamped)
        self.assertAlmostEqual(pre_abs_after, o + p, delta=1e-6)
        self.assertAlmostEqual(pre_abs_after, 988.0, delta=1e-6)
        self.assertGreaterEqual(c, p)

    def test_limit_pre_anchor_shift_caps_large_delta(self):
        vals = _limit_pre_anchor_shift(
            860.0,
            160.0,
            -280.0,
            128.0,
            52.0,
            pre_abs_before=930.0,
            max_shift_ms=22.0,
        )
        o, c, ct, p, ov, pre_abs_after, clamped = vals
        self.assertTrue(clamped)
        self.assertAlmostEqual(pre_abs_after - 930.0, 22.0, delta=1e-6)
        self.assertAlmostEqual(pre_abs_after, o + p, delta=1e-6)
        self.assertGreaterEqual(c, p)
        self.assertGreaterEqual(abs(ct), c)

    def test_guard_ja_cv_cutoff_to_next_onset_uses_stricter_margin_for_cvvc(self):
        def _validate(offset, consonant, cutoff, pre, ovl):
            return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

        syllables = [
            {"phones": [SimpleNamespace(mark="a", minTime=0.10, maxTime=0.22)]},
            {"phones": [SimpleNamespace(mark="k", minTime=0.50, maxTime=0.62)]},
        ]
        out_vcv = guard_ja_cv_cutoff_to_next_onset(
            100.0, 190.0, -420.0, 130.0, 0, syllables, _validate, alias_type="cv", format_type="vcv"
        )
        out_cvvc = guard_ja_cv_cutoff_to_next_onset(
            100.0, 190.0, -420.0, 130.0, 0, syllables, _validate, alias_type="cv", format_type="cvvc"
        )
        cut_abs_vcv = abs(out_vcv[2])
        cut_abs_cvvc = abs(out_cvvc[2])
        self.assertLessEqual(cut_abs_vcv, 382.0 + 1e-6)
        self.assertLessEqual(cut_abs_cvvc, 378.0 + 1e-6)
        self.assertLessEqual(cut_abs_cvvc, cut_abs_vcv - 3.0)

    def test_japanese_n_bridge_alias_detection(self):
        self.assertTrue(_ja_is_n_bridge_alias("n じょ", "vcv"))
        self.assertTrue(_ja_is_n_bridge_alias("n j", "vc"))
        self.assertFalse(_ja_is_n_bridge_alias("a じょ", "vcv"))

    def test_japanese_n_bridge_vcv_anchors_closer_to_consonant_boundary(self):
        normal = _compute_vcv_params_from_virtual_split("a じょ", 1000.0, 1120.0, 1180.0, 1360.0)
        nasal = _compute_vcv_params_from_virtual_split("n じょ", 1000.0, 1120.0, 1180.0, 1360.0)
        # n-bridge는 경계를 현재 자음 onset 쪽으로 더 강하게 붙여 선행발음/오프셋 흔들림을 줄여야 한다.
        self.assertGreater(nasal[0], normal[0])
        self.assertLess(abs((nasal[0] + nasal[3]) - 1180.0), abs((normal[0] + normal[3]) - 1180.0))

    def test_japanese_bridge_overlap_moves_close_to_pre(self):
        vc_hard = _clamp_ja_bridge_overlap(120.0, 20.0, "k", mode="vc")
        vc_soft = _clamp_ja_bridge_overlap(120.0, 20.0, "m", mode="vc")
        vv = _clamp_ja_bridge_overlap(120.0, 20.0, "", mode="vv")
        self.assertGreater(vc_hard, 95.0)
        self.assertGreater(vc_soft, vc_hard)
        self.assertGreater(vv, vc_soft)

    def test_adaptive_japanese_overlap_prefers_vv_then_sonorant_then_hard_vc(self):
        vc_hard = _adaptive_ja_overlap(120.0, "k", mode="vc")
        vc_sonorant = _adaptive_ja_overlap(120.0, "m", mode="vc")
        vv = _adaptive_ja_overlap(120.0, "", mode="vv")
        self.assertGreater(vc_sonorant, vc_hard)
        self.assertGreater(vv, vc_sonorant)

    def test_recenter_japanese_params_around_pre_reduces_bridge_gap(self):
        _o, c, ct, p, ov = _recenter_ja_params_around_pre(
            0.0, 220.0, -360.0, 120.0, 20.0, alias_type="vc", alias_text="a k"
        )
        self.assertLessEqual(p - ov, 20.0)
        self.assertGreaterEqual(c - p, 20.0)
        self.assertGreaterEqual(abs(ct) - c, 12.0)

    def test_sanitize_japanese_oto_prevents_cutoff_before_offset(self):
        o, c, ct, p, ov = sanitize_ja_oto_for_wav_duration(
            1800.0, 240.0, -520.0, 160.0, 80.0, 2000.0, alias_type="vcv"
        )
        active_end_abs = 2000.0 - abs(ct)
        self.assertGreater(active_end_abs, o)
        self.assertGreaterEqual(active_end_abs - o, c)
        self.assertGreaterEqual(c, p + 4.0)
        self.assertLessEqual(ov, p)

    def test_sanitize_japanese_oto_keeps_bridge_compact_when_room_is_small(self):
        o, c, ct, p, ov = sanitize_ja_oto_for_wav_duration(
            950.0, 180.0, -200.0, 140.0, 60.0, 1080.0, alias_type="vc"
        )
        active_end_abs = 1080.0 - abs(ct)
        self.assertGreater(active_end_abs, o)
        self.assertGreaterEqual(active_end_abs - o, c)
        self.assertGreaterEqual(c, p)
        self.assertLessEqual(ov, p)

    def test_sanitize_japanese_internal_params_preserves_internal_active_length(self):
        o, c, ct, p, ov = _sanitize_ja_internal_params_for_wav_duration(
            1500.0, 180.0, -220.0, 120.0, 60.0, 2000.0, alias_type="vcv"
        )
        self.assertAlmostEqual(abs(ct), 220.0, delta=20.0)
        self.assertLessEqual(c, abs(ct))
        self.assertLessEqual(p, c)
        self.assertLessEqual(ov, p)

    def test_convert_japanese_internal_cutoff_to_oto_field(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "test.wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(1000)
                wf.writeframes(b"\x00\x00" * 2000)

            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=n じょ,1500.00,180.00,-220.00,120.00,60.00\n")

            changed = _convert_ja_internal_cutoff_to_oto_field(oto_path, td)
            self.assertGreaterEqual(changed, 0)

            with open(oto_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
            parts = line.split("=", 1)[1].split(",")
            offset = float(parts[1])
            cons = float(parts[2])
            cutoff = float(parts[3])
            pre = float(parts[4])
            ovl = float(parts[5])
            active_len = 2000.0 - offset - abs(cutoff)
            self.assertAlmostEqual(abs(cutoff), 220.0, delta=25.0)
            self.assertGreaterEqual(active_len, 8.0)
            self.assertLessEqual(cons, active_len)
            self.assertLessEqual(pre, cons)
            self.assertLessEqual(ovl, pre)

    def test_recenter_korean_params_around_pre_reduces_bridge_gap(self):
        _o, c, ct, p, ov = _recenter_kr_params_around_pre(
            0.0, 220.0, -360.0, 120.0, 20.0, alias_type="vc", alias_text="a T"
        )
        self.assertGreaterEqual(p - ov, 70.0)
        self.assertLessEqual(p - ov, 130.0)
        self.assertGreaterEqual(c - p, 24.0)
        self.assertGreaterEqual(abs(ct) - c, 8.0)

    def test_korean_vc_from_adjacent_cv_keeps_stop_coda_compact(self):
        prev_cv = {
            "pre": 92.0,
            "vowel_end_abs": 820.0,
            "vowel_len": 180.0,
            "cons_gap": 84.0,
        }
        next_cv = {
            "pre": 88.0,
            "onset_abs": 900.0,
            "pre_abs": 900.0,
            "cons_abs": 990.0,
            "cons_gap": 82.0,
        }
        off, cons, cut, pre, ovl = _compute_vc_from_adjacent_cv(prev_cv, next_cv, "vc", True)
        self.assertLessEqual(pre, 110.0)
        self.assertGreaterEqual(pre - ovl, 6.0)
        self.assertLessEqual(pre - ovl, 24.0)
        self.assertLessEqual(cons - pre, 60.0)
        self.assertLessEqual(abs(cut) - cons, 24.0)

    def test_korean_vc_from_adjacent_cv_keeps_sonorant_overlap_near_pre(self):
        prev_cv = {
            "pre": 108.0,
            "vowel_end_abs": 1020.0,
            "vowel_len": 210.0,
            "cons_gap": 96.0,
        }
        next_cv = {
            "pre": 102.0,
            "onset_abs": 1105.0,
            "pre_abs": 1105.0,
            "cons_abs": 1200.0,
            "cons_gap": 90.0,
        }
        off, cons, cut, pre, ovl = _compute_vc_from_adjacent_cv(prev_cv, next_cv, "vc", False)
        self.assertLessEqual(pre, 132.0)
        self.assertLessEqual(pre - ovl, 18.0)
        self.assertLessEqual(cons - pre, 76.0)
        self.assertLessEqual(abs(cut) - cons, 42.0)

    def test_korean_refine_bridge_vc_uses_adjacent_cv_anchor(self):
        prev_cv = {
            "pre": 102.0,
            "vowel_end_abs": 860.0,
            "vowel_len": 190.0,
            "cons_gap": 86.0,
        }
        next_cv = {
            "pre": 96.0,
            "onset_abs": 940.0,
            "pre_abs": 940.0,
            "cons_abs": 1034.0,
            "cons_gap": 88.0,
        }
        off, cons, cut, pre, ovl = _refine_kr_bridge_with_adjacent_cv(
            770.0,
            222.0,
            -410.0,
            160.0,
            52.0,
            alias_type="vc",
            alias_text="a t",
            prev_cv=prev_cv,
            next_cv=next_cv,
        )
        pre_abs = off + pre
        self.assertLessEqual(abs(pre_abs - 940.0), 24.0)
        self.assertLessEqual(pre - ovl, 24.0)
        self.assertLessEqual(abs(cut), 184.0)

    def test_korean_refine_bridge_vv_keeps_pre_ovl_gap_stable(self):
        prev_cv = {
            "pre": 104.0,
            "vowel_end_abs": 1220.0,
            "vowel_len": 220.0,
            "cons_gap": 92.0,
        }
        next_cv = {
            "pre": 98.0,
            "onset_abs": 1285.0,
            "pre_abs": 1285.0,
            "cons_abs": 1380.0,
            "cons_gap": 90.0,
        }
        off, cons, cut, pre, ovl = _refine_kr_bridge_with_adjacent_cv(
            1130.0,
            250.0,
            -390.0,
            150.0,
            28.0,
            alias_type="vv",
            alias_text="a i",
            prev_cv=prev_cv,
            next_cv=next_cv,
        )
        self.assertGreaterEqual(pre - ovl, 4.0)
        self.assertLessEqual(pre - ovl, 14.0)
        self.assertGreaterEqual(cons - pre, 56.0)
        self.assertGreater(abs(cut), cons)

    def test_korean_cvvc_vc_direct_timing_starts_near_previous_vowel_tail(self):
        off, cons, cut, pre, ovl = _compute_kr_cvvc_vc_timing_direct(
            "a b", "vc", 1000.0, 1180.0, 1240.0, 1310.0
        )
        self.assertGreaterEqual(off, 1060.0)
        self.assertLessEqual(off, 1130.0)
        self.assertGreaterEqual(pre, 120.0)
        self.assertLessEqual(pre, 240.0)
        self.assertGreaterEqual(pre - ovl, 75.0)
        self.assertLessEqual(pre - ovl, 130.0)
        self.assertGreaterEqual(cons - pre, 18.0)
        self.assertLessEqual(cons - pre, 90.0)

    def test_korean_cvvc_vc_direct_timing_gives_liquid_more_body_than_stop(self):
        _off_s, cons_s, _cut_s, pre_s, ovl_s = _compute_kr_cvvc_vc_timing_direct(
            "a d", "vc", 1000.0, 1180.0, 1240.0, 1310.0
        )
        _off_l, cons_l, _cut_l, pre_l, ovl_l = _compute_kr_cvvc_vc_timing_direct(
            "a l", "vc", 1000.0, 1180.0, 1240.0, 1310.0
        )
        self.assertGreaterEqual(pre_l, pre_s)
        self.assertGreater(cons_l - pre_l, cons_s - pre_s)
        self.assertGreater(pre_l - ovl_l, pre_s - ovl_s)

    def test_korean_vc_cutoff_guard_keeps_cutoff_inside_next_consonant(self):
        syllables = [
            {"phones": [SimpleNamespace(mark="a", minTime=1.00, maxTime=1.16)]},
            {
                "phones": [
                    SimpleNamespace(mark="n", minTime=1.22, maxTime=1.28),
                    SimpleNamespace(mark="a", minTime=1.28, maxTime=1.40),
                ]
            },
        ]
        off, cons, cut, pre, ovl = _guard_kr_vc_cutoff_to_next_segment(
            1000.0, 260.0, -420.0, 180.0, 70.0, 0, syllables, alias_text="a n"
        )
        self.assertLessEqual(abs(cut), 270.0)
        self.assertLessEqual(abs(cut), 280.0)
        self.assertLessEqual(cons, abs(cut) - 10.0)

    def test_korean_cvvc_vv_direct_timing_stays_near_previous_vowel_tail(self):
        syllables_info = [
            {
                "phones": [
                    SimpleNamespace(mark="i", minTime=0.00, maxTime=0.10),
                ]
            },
            {
                "phones": [
                    SimpleNamespace(mark="eo", minTime=0.12, maxTime=0.26),
                ]
            },
        ]
        vals = _compute_kr_cvvc_vv_timing_direct(1, syllables_info, 120.0, 260.0)
        self.assertIsNotNone(vals)
        off, cons, cut, pre, ovl = vals
        self.assertGreaterEqual(off, 24.0)
        self.assertLessEqual(off, 100.0)
        self.assertGreater(pre, 18.0)
        self.assertLessEqual(ovl, pre)
        self.assertGreater(abs(cut), cons)

    def test_korean_cv_onset_slice_skips_previous_coda_and_keeps_onset_glide_cluster(self):
        phones = [
            SimpleNamespace(minTime=1.000, maxTime=1.080, mark="l"),
            SimpleNamespace(minTime=1.080, maxTime=1.150, mark="b"),
            SimpleNamespace(minTime=1.150, maxTime=1.210, mark="y"),
            SimpleNamespace(minTime=1.210, maxTime=1.420, mark="a"),
        ]
        onset_idx, c_start, c_end, n_start, n_end = _select_kr_cv_onset_slice(phones)
        self.assertEqual(onset_idx, 1)
        self.assertAlmostEqual(c_start, 1080.0, delta=1.0)
        self.assertAlmostEqual(c_end, 1210.0, delta=1.0)
        self.assertAlmostEqual(n_start, 1210.0, delta=1.0)
        self.assertAlmostEqual(n_end, 1420.0, delta=1.0)

    def test_korean_closure_alias_is_classified_as_vc_and_file_stays_cvvc(self):
        self.assertEqual(classify_alias("a pcl"), "vc")
        fmt = detect_alias_format(["- ppa", "- ppya", "- ppwa", "ppa", "ppya", "ppwa", "a pp", "a pcl"])
        self.assertEqual(fmt, "cvvc")

    def test_korean_cvvc_cv_skips_soft_mel_offset_guard(self):
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy unavailable")
        mel_ctx = {
            "times_ms": np.arange(0.0, 300.0, 10.0),
            "energy": np.concatenate([np.zeros(10), np.ones(20)]),
            "db_db": np.concatenate([np.full(10, -60.0), np.full(20, -20.0)]),
            "f0_voicing": np.zeros(30),
            "db_silence_th": -42.0,
        }
        off, cons, cut, pre, ovl, soft_off, soft_cut = _apply_soft_mel_offset_cutoff_guard(
            0.0, 160.0, -220.0, 80.0, 24.0, "cv", mel_ctx=mel_ctx, alias_text="pya", file_format="cvvc"
        )
        self.assertAlmostEqual(off, 0.0, delta=1e-6)
        self.assertAlmostEqual(soft_off, 0.0, delta=1e-6)
        self.assertLessEqual(abs(soft_cut), 220.0)


if __name__ == "__main__":
    unittest.main()
