import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_batch_prepare import _discover_work_items, prepare_staged_auto_pairs
from core.oto_ml_collection import _discover_oto_files


class StagingPrepareTests(unittest.TestCase):
    def test_empty_top_level_oto_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "dataset", "japanese", "vcv", "Voice")
            sub = os.path.join(root, "C4")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(root, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(sub, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(sub, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            otos = _discover_oto_files(root)
            self.assertEqual(len(otos), 1)
            self.assertTrue(otos[0].endswith(os.path.join("C4", "oto.ini")))

            items = _discover_work_items(os.path.join(td, "dataset"))
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0].manual_oto.endswith(os.path.join("C4", "oto.ini")))

    def test_prepare_reuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            work = os.path.join(td, "dataset", "korean", "cvc", "Voice")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "oto_auto_ml.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")
            with open(os.path.join(work, "a.TextGrid"), "w", encoding="utf-8") as f:
                f.write("dummy")

            result = prepare_staged_auto_pairs(os.path.join(td, "dataset"), dry_run=False)
            self.assertEqual(result["summary"]["prepared"], 1)
            self.assertEqual(result["items"][0].status, "prepared_existing")
            self.assertEqual(result["items"][0].tg_dir, work)


if __name__ == "__main__":
    unittest.main()
