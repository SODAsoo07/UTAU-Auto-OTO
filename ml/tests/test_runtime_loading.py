import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_runtime import OtoModelBundle, load_oto_model_bundle, predict_oto_deltas


class RuntimeLoadingTests(unittest.TestCase):
    def test_unknown_backend_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"backend": "unknown", "feature_version": "v1"}, f)
            bundle = load_oto_model_bundle(td)
            self.assertIsNone(bundle)

    def test_load_coupled_backend_dispatches_loader(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"backend": "coupled_nn_v1", "feature_version": "v7"}, f)
            with mock.patch("core.oto_ml_coupled.load_coupled_bundle", return_value={"ok": True}) as mocked:
                bundle = load_oto_model_bundle(td)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.backend, "coupled_nn_v1")
            self.assertEqual(bundle.payload, {"ok": True})
            mocked.assert_called_once()

    def test_load_ensemble_backend_dispatches_loader(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"backend": "ensemble_v1", "feature_version": "v11"}, f)
            with mock.patch("core.oto_ml_ensemble.load_ensemble_bundle", return_value={"ok": True}) as mocked:
                bundle = load_oto_model_bundle(td)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.backend, "ensemble_v1")
            self.assertEqual(bundle.payload, {"ok": True})
            mocked.assert_called_once()

    def test_predict_oto_deltas_keeps_coupled_confidence(self):
        bundle = OtoModelBundle(
            backend="coupled_nn_v1",
            model_dir="C:/tmp/model",
            meta={},
            feature_schema={"feature_names": []},
            payload={"model": "dummy"},
        )
        with mock.patch(
            "core.oto_ml_coupled.predict_coupled_deltas",
            return_value=({"delta_offset": 1.0, "delta_cons": 0.0, "delta_cutoff": 0.0, "delta_pre": 0.0, "delta_ovl": 0.0}, 0.77),
        ):
            out = predict_oto_deltas(bundle, {"language": "korean"})
        self.assertEqual(out.backend, "coupled_nn_v1")
        self.assertAlmostEqual(out.confidence, 0.77)

    def test_predict_oto_deltas_keeps_ensemble_route(self):
        bundle = OtoModelBundle(
            backend="ensemble_v1",
            model_dir="C:/tmp/ensemble",
            meta={},
            feature_schema={"feature_names": []},
            payload={"ensemble": "dummy"},
        )
        with mock.patch(
            "core.oto_ml_ensemble.predict_ensemble_deltas",
            return_value={
                "deltas": {"delta_offset": 1.0, "delta_cons": 0.0, "delta_cutoff": 0.0, "delta_pre": 0.0, "delta_ovl": 0.0},
                "confidence": 0.61,
                "route": "ensemble_v1:meta",
                "route_backend": "ensemble_v1",
            },
        ):
            out = predict_oto_deltas(bundle, {"language": "korean"})
        self.assertEqual(out.backend, "ensemble_v1")
        self.assertEqual(out.route, "ensemble_v1:meta")
        self.assertEqual(out.route_backend, "ensemble_v1")
        self.assertAlmostEqual(out.confidence, 0.61)


if __name__ == "__main__":
    unittest.main()
