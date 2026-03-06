import os
import sys
import tempfile
import unittest

import textgrid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.whisperx_runner import (
    _fallback_tokens_from_filename,
    _token_spans_to_phone_spans,
    _write_textgrid,
    get_default_whisperx_align_model,
)


class WhisperXRunnerTests(unittest.TestCase):
    def test_default_align_model_resolution(self):
        self.assertEqual(
            get_default_whisperx_align_model("korean", "low_load"),
            "Kkonjeong/wav2vec2-base-korean",
        )
        self.assertEqual(
            get_default_whisperx_align_model("japanese", "balanced"),
            "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        )
        self.assertEqual(
            get_default_whisperx_align_model("unknown", "balanced"),
            "kresnik/wav2vec2-large-xlsr-korean",
        )

    def test_phone_projection_uses_dictionary(self):
        token_spans = [
            {"token": "가", "start": 0.0, "end": 0.6},
            {"token": "나", "start": 0.6, "end": 1.2},
        ]
        dictionary = {
            "가": ["k", "a"],
            "나": ["n", "a"],
        }
        phones = _token_spans_to_phone_spans(token_spans, dictionary)
        self.assertEqual(len(phones), 4)
        self.assertEqual(phones[0]["mark"], "k")
        self.assertEqual(phones[1]["mark"], "a")
        self.assertAlmostEqual(float(phones[0]["start"]), 0.0, places=4)
        self.assertAlmostEqual(float(phones[-1]["end"]), 1.2, places=4)

    def test_filename_fallback_for_hangul(self):
        tokens = _fallback_tokens_from_filename("가나다")
        self.assertEqual(tokens, ["가", "나", "다"])

    def test_write_textgrid_contains_words_and_phones_tier(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a.TextGrid")
            token_spans = [{"token": "가", "start": 0.0, "end": 1.0}]
            phone_spans = [
                {"mark": "k", "start": 0.0, "end": 0.4},
                {"mark": "a", "start": 0.4, "end": 1.0},
            ]
            _write_textgrid(path, 1.0, token_spans, phone_spans)

            tg = textgrid.TextGrid.fromFile(path)
            tier_names = [getattr(t, "name", "") for t in tg]
            self.assertIn("words", tier_names)
            self.assertIn("phones", tier_names)


if __name__ == "__main__":
    unittest.main()
