import os
import sys
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
    extract_feature_rows,
    get_feature_schema,
)
from core.ja_oto_generator import (
    _adaptive_ja_overlap,
    _clamp_ja_bridge_overlap,
    _compute_vcv_params_from_virtual_split,
    _convert_ja_internal_cutoff_to_oto_field,
    _ja_is_n_bridge_alias,
    _sanitize_ja_internal_params_for_wav_duration,
    _recenter_ja_params_around_pre,
    sanitize_ja_oto_for_wav_duration,
    _select_ja_cv_syllable_index,
    _select_vcv_syllable_index,
    detect_ja_alias_format,
)
from core.ja_lab_generator import parse_ja_filename
from core.lab_generator import _parse_filename, _split_kr_lab_content_tokens
from core.oto_generator import _apply_soft_mel_offset_cutoff_guard
from core.oto_generator import _recenter_kr_params_around_pre
from core.oto_generator import _compute_vc_from_adjacent_cv
from core.oto_generator import _compute_kr_cvvc_vc_timing_direct
from core.oto_generator import _compute_kr_cvvc_vv_timing_direct
from core.oto_generator import _build_kr_cvvc_occurrence_map
from core.oto_generator import _build_kr_cvvc_vv_occurrence_map
from core.oto_generator import _build_kr_syllables_from_phone_nuclei
from core.oto_generator import _extract_kr_cv_targets_from_filename
from core.oto_generator import _guard_kr_vc_cutoff_to_next_segment
from core.oto_generator import _resolve_kr_cvvc_occurrence_index
from core.oto_generator import _resolve_kr_cvvc_vv_index
from core.oto_generator import _select_kr_cv_onset_slice
from core.oto_generator import _should_prefer_alias_based_syllables
from core.oto_generator import _uses_kr_vc_context
from core.oto_generator import _cv_match_score
from core.oto_generator import classify_alias, detect_alias_format


class FeatureExtractionTests(unittest.TestCase):
    def test_schema_contains_feature_names(self):
        schema = get_feature_schema()
        self.assertEqual(schema["feature_names"], FEATURE_NAMES)

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
        self.assertEqual(_normalize_alias_for_match("a か連", language="japanese"), "a か")
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
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("do", "cv", occ_map, state), 1)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("- dyo", "cv_head", occ_map, state), 2)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("dyo", "cv", occ_map, state), 3)
        self.assertEqual(_resolve_kr_cvvc_occurrence_index("do", "cv", occ_map, state), 5)

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

    def test_japanese_vcv_selector_does_not_remap_backward(self):
        syllables = [{"word": s} for s in ["ja", "jo", "ja"]]
        self.assertEqual(_select_vcv_syllable_index("a じょ", 2, syllables), 2)

    def test_japanese_vcv_selector_does_not_overjump_more_than_one(self):
        syllables = [{"word": s} for s in ["ja", "ka", "ku", "kyo", "sa"]]
        self.assertEqual(_select_vcv_syllable_index("a きょ", 1, syllables), 1)

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
