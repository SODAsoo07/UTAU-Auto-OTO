import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_ensemble import ENSEMBLE_BACKEND, predict_gated_ensemble_deltas


class OtoMlEnsembleTests(unittest.TestCase):
    def test_predict_gated_ensemble_routes_to_lightgbm_on_low_confidence(self):
        with mock.patch(
            "core.oto_ml_ensemble.predict_lightgbm_deltas",
            return_value={
                "delta_offset": 1.0,
                "delta_cons": 2.0,
                "delta_cutoff": 3.0,
                "delta_pre": 4.0,
                "delta_ovl": 5.0,
            },
        ), mock.patch(
            "core.oto_ml_ensemble.predict_coupled_deltas",
            return_value=(
                {
                    "delta_offset": 10.0,
                    "delta_cons": 20.0,
                    "delta_cutoff": 30.0,
                    "delta_pre": 40.0,
                    "delta_ovl": 50.0,
                },
                0.25,
            ),
        ):
            out = predict_gated_ensemble_deltas(
                lightgbm_bundle={"payload": {}, "meta": {}, "schema": {}},
                coupled_bundle={"payload": {}, "meta": {}, "schema": {}, "backend": "coupled_nn_v2_rawmel"},
                feature_row={"blank_span_confidence": 0.8, "mapping_confidence": 0.4, "alias_type": "cv"},
                min_coupled_confidence=0.55,
            )
        self.assertEqual(out["route_backend"], "lightgbm")
        self.assertEqual(out["deltas"]["delta_offset"], 1.0)

    def test_predict_gated_ensemble_blends_mid_confidence(self):
        with mock.patch(
            "core.oto_ml_ensemble.predict_lightgbm_deltas",
            return_value={
                "delta_offset": 0.0,
                "delta_cons": 0.0,
                "delta_cutoff": 0.0,
                "delta_pre": 0.0,
                "delta_ovl": 0.0,
            },
        ), mock.patch(
            "core.oto_ml_ensemble.predict_coupled_deltas",
            return_value=(
                {
                    "delta_offset": 10.0,
                    "delta_cons": 10.0,
                    "delta_cutoff": 10.0,
                    "delta_pre": 10.0,
                    "delta_ovl": 10.0,
                },
                0.68,
            ),
        ):
            out = predict_gated_ensemble_deltas(
                lightgbm_bundle={"payload": {}, "meta": {}, "schema": {}},
                coupled_bundle={"payload": {}, "meta": {}, "schema": {}, "backend": "coupled_nn_v2_rawmel"},
                feature_row={"blank_span_confidence": 0.45, "mapping_confidence": 0.7, "alias_type": "vc"},
                min_coupled_confidence=0.55,
            )
        self.assertEqual(out["route_backend"], ENSEMBLE_BACKEND)
        self.assertGreater(out["deltas"]["delta_offset"], 0.0)
        self.assertLess(out["deltas"]["delta_offset"], 10.0)


if __name__ == "__main__":
    unittest.main()
