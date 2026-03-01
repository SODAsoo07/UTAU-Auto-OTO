import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_refiner import apply_oto_ml_to_oto_file


class MlFallbackTests(unittest.TestCase):
    def test_refiner_skips_when_model_missing(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            original = "test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n"
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write(original)
            os.environ["UTOA_DISABLE_OTO_ML"] = "1"
            try:
                changed = apply_oto_ml_to_oto_file("korean", oto_path, tg_dir=td, wav_dir=td)
            finally:
                os.environ.pop("UTOA_DISABLE_OTO_ML", None)
            self.assertEqual(changed, 0)
            with open(oto_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original)


if __name__ == "__main__":
    unittest.main()
