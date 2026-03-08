import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import _build_tg_index, build_training_rows, extract_feature_rows


class FeatureCacheTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("UTOA_OTO_ML_CACHE_DIR", None)
        os.environ.pop("UTOA_OTO_ML_ROW_CACHE_DIR", None)

    def test_extract_feature_rows_uses_disk_cache(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            tg_dir = os.path.join(td, "tg")
            wav_dir = os.path.join(td, "wav")
            os.makedirs(tg_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(tg_dir, "a.TextGrid"), "w", encoding="utf-8") as f:
                f.write("dummy")
            with open(os.path.join(wav_dir, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            cached_rows = [{"alias": "ga", "wav": "a.wav"}]
            with mock.patch("core.oto_ml_features._load_feature_cache", return_value=cached_rows), mock.patch("core.oto_ml_features.parse_oto_rows") as parse_rows:
                rows = extract_feature_rows("japanese", oto_path, tg_dir, wav_dir)

            self.assertEqual(rows, cached_rows)
            parse_rows.assert_not_called()

    def test_build_training_rows_uses_disk_cache(self):
        with tempfile.TemporaryDirectory() as td:
            auto_oto = os.path.join(td, "oto_auto.ini")
            manual_oto = os.path.join(td, "oto.ini")
            tg_dir = os.path.join(td, "tg")
            wav_dir = os.path.join(td, "wav")
            os.makedirs(tg_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)
            for path in (auto_oto, manual_oto):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("a.wav=ga,0,1,-2,0,0\n")
            payload = {
                "feature_version": "v1",
                "rows": [{"alias": "ga", "delta_offset": 1.0}],
                "stats": {"auto_rows": 1, "manual_rows": 1, "matched_rows": 1, "skipped_rows": 0},
            }
            with mock.patch("core.oto_ml_features._load_training_row_cache", return_value=(payload["rows"], payload["stats"])), mock.patch("core.oto_ml_features.extract_feature_rows") as extract_rows, mock.patch("core.oto_ml_features.parse_oto_rows") as parse_rows:
                rows, stats = build_training_rows("japanese", auto_oto, manual_oto, tg_dir, wav_dir)

            self.assertEqual(rows, payload["rows"])
            self.assertEqual(stats, payload["stats"])
            extract_rows.assert_not_called()
            parse_rows.assert_not_called()

    def test_build_tg_index_walks_nested_textgrid_paths(self):
        with tempfile.TemporaryDirectory() as td:
            sub = os.path.join(td, "C4")
            os.makedirs(sub, exist_ok=True)
            tg_path = os.path.join(sub, "a.TextGrid")
            with open(tg_path, "w", encoding="utf-8") as f:
                f.write("dummy")

            index = _build_tg_index(td)
            self.assertEqual(index.get("a"), tg_path)


if __name__ == "__main__":
    unittest.main()
