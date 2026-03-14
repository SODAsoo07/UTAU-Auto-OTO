import os
import unittest
from unittest import mock

from core.oto_ml_refiner import apply_oto_ml_delta
from core.oto_row_abstain import decide_cv_row_abstain


class MappingPolicyGuardsTests(unittest.TestCase):
    def test_abstain_uses_alias_specific_margin_override(self):
        out = decide_cv_row_abstain(
            alias_type="cv_head",
            format_type="cvvc",
            candidate_idx=0,
            candidate_count=1,
            candidate_active=True,
            confidence_margin=9.0,
            min_confidence_margin=8.0,
            margin_formats={"cvvc"},
            min_confidence_margin_by_alias_type={"cv_head": 10.0},
        )
        self.assertTrue(out["should_skip"])
        self.assertEqual(out["reason"], "row_low_margin_candidate")

    def test_abstain_uses_alias_specific_row_conf_override(self):
        out = decide_cv_row_abstain(
            alias_type="vcv",
            format_type="cvvc",
            candidate_idx=0,
            candidate_count=1,
            candidate_active=True,
            row_confidence=0.70,
            min_row_confidence=0.65,
            min_row_confidence_by_alias_type={"vcv": 0.75},
            margin_alias_types={"vcv"},
        )
        self.assertTrue(out["should_skip"])
        self.assertEqual(out["reason"], "row_low_confidence")

    def test_high_conf_rows_can_skip_ml_inference(self):
        row_context = {
            "mapping_confidence": 0.95,
            "mapping_margin": 15.0,
            "blank_span_confidence": 0.05,
            "syllable_blank_confidence": 0.05,
            "base_offset": 10.0,
            "base_cons": 20.0,
            "base_cutoff_abs": 30.0,
            "base_pre": 40.0,
            "base_ovl": 50.0,
        }

        env = {
            "UTOA_ML_SKIP_HIGH_CONF_RULE_ROWS": "1",
            "UTOA_ML_SKIP_HIGH_CONF_MIN": "0.90",
            "UTOA_ML_SKIP_HIGH_CONF_MIN_MARGIN": "12.0",
            "UTOA_ML_SKIP_HIGH_CONF_MAX_BLANK": "0.18",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("core.oto_ml_refiner._get_validate_func", return_value=lambda *a, **k: True):
                with mock.patch(
                    "core.oto_ml_refiner._finalize_ml_params",
                    side_effect=lambda language, row_context, params, *_args, **_kwargs: params,
                ):
                    with mock.patch("core.oto_ml_refiner.predict_oto_deltas") as pred_mock:
                        out = apply_oto_ml_delta("korean", row_context, bundle=mock.Mock(backend="lightgbm"))
        pred_mock.assert_not_called()
        self.assertEqual(out, (10.0, 20.0, -20.0, 40.0, 50.0))


if __name__ == "__main__":
    unittest.main()
