import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.alignment_ingest import build_ja_alignment_ingest, build_kr_alignment_ingest


class AlignmentIngestTests(unittest.TestCase):
    def test_build_ja_alignment_ingest_normalizes_snapshot(self):
        file_ctx = SimpleNamespace(
            real_wav_name="a.wav",
            tg_path="a.TextGrid",
            mel_ctx_for_file={"mel": 1},
            wav_duration_ms=1200.0,
        )
        loop_prep = SimpleNamespace(
            ph_intervals=["p1"],
            wd_intervals=["w1"],
            filename_syllables=["ka"],
            cv_targets=["ka"],
            detected_format="cvvc",
            format_type="cvvc",
            ja_style_profile={"mode": "x"},
            forced_words_mapping=True,
            conf_th=0.61,
            phone_quality={"known_vowel_phone_count": 3},
            low_quality_reasons=["spn_heavy"],
            low_phone_quality=True,
            textgrid_trust_score=0.42,
            textgrid_trust_tier="low",
            prefer_filename_sequence=True,
            timeline_start_ms=10.0,
            timeline_end_ms=900.0,
            effective_end_ms=880.0,
            boundary_points_ms=[10.0, 20.0],
            phone_spans_ms=[(10.0, 20.0)],
        )

        snap = build_ja_alignment_ingest(file_ctx, loop_prep)
        self.assertEqual(snap.language, "japanese")
        self.assertEqual(snap.phones, ["p1"])
        self.assertEqual(snap.words, ["w1"])
        self.assertEqual(snap.extra["format_type"], "cvvc")
        self.assertTrue(snap.extra["forced_words_mapping"])
        self.assertAlmostEqual(snap.textgrid_trust_score, 0.42)
        self.assertEqual(snap.timeline_meta["effective_end_ms"], 880.0)

    def test_build_kr_alignment_ingest_carries_phone_variants_and_timeline(self):
        file_ctx = SimpleNamespace(
            real_wav_name="b.wav",
            tg_path="b.TextGrid",
            mel_ctx_for_file=None,
            wav_duration_ms=1500.0,
        )
        loop_prep = SimpleNamespace(
            ph_intervals=["p_main"],
            ph_intervals_all=["p_all"],
            wd_intervals=["w1", "w2"],
            wav_duration_ms=1510.0,
            file_format="cvvc",
            file_mapping_conf_th=0.58,
            filename_cv_targets=["go", "gyo"],
            targets_for_build=["go", "gyo"],
            force_words_phone_fill=False,
            phone_quality={"spn_ratio_in_phone_tier": 0.1},
            low_quality_reasons=[],
            low_phone_quality=False,
            textgrid_trust_score=0.77,
            textgrid_trust_tier="high",
            prefer_filename_sequence=False,
            timeline_start_ms=0.0,
            timeline_end_ms=1480.0,
        )

        snap = build_kr_alignment_ingest(file_ctx, loop_prep)
        self.assertEqual(snap.language, "korean")
        self.assertEqual(snap.phones, ["p_main"])
        self.assertEqual(snap.phones_all, ["p_all"])
        self.assertEqual(snap.extra["file_format"], "cvvc")
        self.assertAlmostEqual(snap.extra["file_mapping_conf_th"], 0.58)
        self.assertAlmostEqual(snap.textgrid_trust_score, 0.77)
        self.assertEqual(snap.timeline_meta["timeline_end_ms"], 1480.0)


if __name__ == "__main__":
    unittest.main()
