import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.analyze_oto_diff import analyze


def _write_oto(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class AnalyzeOtoDiffTests(unittest.TestCase):
    def test_analyze_reports_cv_like_jump_quality(self):
        with tempfile.TemporaryDirectory() as td:
            base_oto = os.path.join(td, "base.ini")
            auto_oto = os.path.join(td, "auto.ini")
            _write_oto(
                base_oto,
                [
                    "a.wav=ka,0,0,0,0,0",
                    "a.wav=- ka,100,0,0,0,0",
                    "a.wav=a k,200,0,0,0,0",
                ],
            )
            _write_oto(
                auto_oto,
                [
                    "a.wav=ka,0,0,0,0,0",
                    "a.wav=- ka,220,0,0,0,0",
                    "a.wav=a k,210,0,0,0,0",
                ],
            )

            report = analyze(base_oto, auto_oto, language="japanese", topn=5)

            self.assertGreater(report["quality"]["cv_like_order_jump_ratio"], 0.0)
            self.assertGreater(report["quality"]["weighted_mae_offset"], report["mae"]["mae_offset"])
            self.assertGreaterEqual(report["quality"]["p90_offset"], report["mae"]["mae_offset"])


if __name__ == "__main__":
    unittest.main()
