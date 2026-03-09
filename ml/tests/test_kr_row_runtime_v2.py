import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_row_runtime_v2 import (
    maybe_build_kr_realized_cv_anchor_record,
    prepare_kr_cv_head_anchor_context,
)


class KrRowRuntimeV2Tests(unittest.TestCase):
    def test_prepare_kr_cv_head_anchor_context_uses_prepared_bounds(self):
        context = prepare_kr_cv_head_anchor_context(
            [{"phones": []}],
            0,
            fallback_anchor_abs=200.0,
            prepare_cv_bounds_fn=lambda syllables, idx: (idx, [], 100.0, 130.0, 140.0, 250.0),
        )
        self.assertEqual(context["anchor_abs"], 130.0)
        self.assertEqual(context["onset_abs"], 100.0)
        self.assertEqual(context["vowel_start_abs"], 140.0)
        self.assertEqual(context["vowel_end_abs"], 250.0)

    def test_prepare_kr_cv_head_anchor_context_falls_back_on_error(self):
        context = prepare_kr_cv_head_anchor_context(
            [],
            0,
            fallback_anchor_abs=210.0,
            prepare_cv_bounds_fn=lambda syllables, idx: (_ for _ in ()).throw(ValueError("boom")),
        )
        self.assertEqual(context["anchor_abs"], 210.0)
        self.assertIsNone(context["onset_abs"])

    def test_maybe_build_kr_realized_cv_anchor_record_returns_none_without_anchor(self):
        record = maybe_build_kr_realized_cv_anchor_record(
            0,
            offset=1.0,
            consonant=2.0,
            cutoff=-3.0,
            pre=1.0,
            ovl=0.5,
            onset_abs=None,
            vowel_start_abs=10.0,
            vowel_end_abs=20.0,
            c_end_abs=9.0,
            build_anchor_fn=lambda *args, **kwargs: {"ok": True},
        )
        self.assertIsNone(record)

    def test_maybe_build_kr_realized_cv_anchor_record_wraps_builder_result(self):
        record = maybe_build_kr_realized_cv_anchor_record(
            3,
            offset=10.0,
            consonant=50.0,
            cutoff=-90.0,
            pre=30.0,
            ovl=20.0,
            onset_abs=100.0,
            vowel_start_abs=130.0,
            vowel_end_abs=240.0,
            c_end_abs=120.0,
            build_anchor_fn=lambda *args, **kwargs: {"payload": kwargs["onset_abs"]},
        )
        self.assertEqual(record, (3, {"payload": 100.0}))


if __name__ == "__main__":
    unittest.main()
