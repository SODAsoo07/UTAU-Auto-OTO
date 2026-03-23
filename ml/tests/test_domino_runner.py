import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.domino_runner import check_domino_ready
from core.ja_domino_transcript import build_domino_phone_plan, syllable_to_domino_phones
from core.pipeline_status import ALIGN_MODEL_MISSING, ALIGN_NOT_READY


class DominoTranscriptTests(unittest.TestCase):
    def test_syllable_to_domino_phones_basic(self):
        self.assertEqual(syllable_to_domino_phones("kya"), ["ky", "a"])
        self.assertEqual(syllable_to_domino_phones("tsu"), ["ts", "u"])
        self.assertEqual(syllable_to_domino_phones("n"), ["N"])
        self.assertEqual(syllable_to_domino_phones("q"), ["cl"])

    def test_build_domino_phone_plan_inserts_pau(self):
        phones, spans = build_domino_phone_plan(["ka", "ki"])
        self.assertGreaterEqual(len(phones), 4)
        self.assertEqual(phones[0], "pau")
        self.assertEqual(phones[-1], "pau")
        self.assertEqual(spans[0][0], "ka")
        self.assertEqual(spans[1][0], "ki")


class DominoReadyTests(unittest.TestCase):
    def test_check_domino_ready_non_japanese(self):
        report = check_domino_ready(language="korean", wav_folder="")
        self.assertEqual(str(report.get("code", "")), ALIGN_NOT_READY)

    def test_check_domino_ready_missing_model(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get("UTOA_DOMINO_ONNX_PATH")
            if "UTOA_DOMINO_ONNX_PATH" in os.environ:
                del os.environ["UTOA_DOMINO_ONNX_PATH"]
            try:
                report = check_domino_ready(language="japanese", wav_folder=td)
            finally:
                if old is not None:
                    os.environ["UTOA_DOMINO_ONNX_PATH"] = old
            self.assertEqual(str(report.get("code", "")), ALIGN_MODEL_MISSING)


if __name__ == "__main__":
    unittest.main()
