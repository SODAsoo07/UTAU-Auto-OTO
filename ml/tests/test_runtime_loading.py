import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_runtime import load_oto_model_bundle


class RuntimeLoadingTests(unittest.TestCase):
    def test_unknown_backend_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"backend": "unknown", "feature_version": "v1"}, f)
            bundle = load_oto_model_bundle(td)
            self.assertIsNone(bundle)


if __name__ == "__main__":
    unittest.main()
