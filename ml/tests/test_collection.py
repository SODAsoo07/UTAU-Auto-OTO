import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_collection import (
    build_datasets_from_candidates,
    discover_training_candidates_from_dataset_root,
    load_training_roots,
)


class CollectionTests(unittest.TestCase):
    def test_load_training_roots_ignores_notes(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "roots.yaml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(
                    "japanese:\n"
                    "  cvvc:\n"
                    "    - C:\\\\jp\n"
                    "korean:\n"
                    "  cvc:\n"
                    "    - C:\\\\kr\n"
                    "notes:\n"
                    "  strip_pitch_suffix_examples:\n"
                    "    - _C4\n"
                )
            data = load_training_roots(cfg)
            self.assertIn("japanese", data)
            self.assertIn("korean", data)
            self.assertNotIn("notes", data)

    def test_discover_training_candidates_from_dataset_root(self):
        with tempfile.TemporaryDirectory() as td:
            work = os.path.join(td, "dataset", "japanese", "cvvc", "VoiceA", "C4")
            tg = os.path.join(work, "textgrids_auto")
            os.makedirs(tg, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "oto_auto_ml.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")
            with open(os.path.join(tg, "a.TextGrid"), "w", encoding="utf-8") as f:
                f.write("dummy")

            candidates = discover_training_candidates_from_dataset_root(os.path.join(td, "dataset"))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].status, "ready")
            self.assertEqual(candidates[0].voicebank_id, "VoiceA")

    def test_build_datasets_from_candidates_appends_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            c1 = mock.Mock()
            c1.status = "ready"
            c1.language = "japanese"
            c1.format_type = "cvvc"
            c1.auto_oto = "auto1.ini"
            c1.manual_oto = "man1.ini"
            c1.tg_dir = "tg1"
            c1.wav_dir = "wav1"
            c1.custom_phonemes = ""
            c1.voicebank_id = "VB1"
            c1.reason = ""
            c1.saved_rows = 0
            c1.matched_rows = 0
            c1.skipped_rows = 0
            c1.out_csv = ""
            c2 = mock.Mock()
            c2.status = "ready"
            c2.language = "japanese"
            c2.format_type = "cvvc"
            c2.auto_oto = "auto2.ini"
            c2.manual_oto = "man2.ini"
            c2.tg_dir = "tg2"
            c2.wav_dir = "wav2"
            c2.custom_phonemes = ""
            c2.voicebank_id = "VB2"
            c2.reason = ""
            c2.saved_rows = 0
            c2.matched_rows = 0
            c2.skipped_rows = 0
            c2.out_csv = ""

            with mock.patch("core.oto_ml_collection.build_and_save_oto_ml_dataset") as fn:
                fn.side_effect = [
                    {"saved_rows": 2, "matched_rows": 2, "skipped_rows": 0, "out_csv": os.path.join(td, "datasets", "japanese", "dataset_japanese_cvvc.csv")},
                    {"saved_rows": 3, "matched_rows": 3, "skipped_rows": 0, "out_csv": os.path.join(td, "datasets", "japanese", "dataset_japanese_cvvc.csv")},
                ]
                result = build_datasets_from_candidates([c1, c2], workspace_root=td)

            self.assertEqual(fn.call_count, 2)
            self.assertFalse(fn.call_args_list[0].kwargs["append"])
            self.assertTrue(fn.call_args_list[1].kwargs["append"])
            self.assertEqual(result["summary"]["saved_rows"], 5)


if __name__ == "__main__":
    unittest.main()
