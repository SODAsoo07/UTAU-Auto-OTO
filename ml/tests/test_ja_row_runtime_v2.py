import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_row_runtime_v2 import (
    build_ja_alias_output_rows,
    build_ja_cv_guard_messages,
    maybe_build_ja_realized_cv_anchor_record,
)


class JaRowRuntimeV2Tests(unittest.TestCase):
    def test_maybe_build_ja_realized_cv_anchor_record_wraps_builder(self):
        record = maybe_build_ja_realized_cv_anchor_record(
            2,
            offset=10.0,
            consonant=50.0,
            cutoff=-90.0,
            pre=30.0,
            ovl=20.0,
            onset_abs=100.0,
            vowel_start_abs=130.0,
            c_end_abs=120.0,
            vowel_end_abs=240.0,
            build_anchor_fn=lambda *args, **kwargs: {"onset": kwargs["onset_abs"]},
        )
        self.assertEqual(record, (2, {"onset": 100.0}))

    def test_build_ja_cv_guard_messages_emits_all_applicable(self):
        messages = build_ja_cv_guard_messages(
            "a.wav",
            "ka",
            cutoff_reduced=1.5,
            offset_reduced=2.0,
            cutoff_extended=3.0,
        )
        self.assertEqual(len(messages), 3)
        self.assertIn("CV_HEAD 오프셋 과선행 보정", messages[0])
        self.assertIn("CV 컷오프 과연장 보정", messages[2])

    def test_build_ja_alias_output_rows_expands_openutau_aliases(self):
        rows = build_ja_alias_output_rows(
            "a.wav",
            "ka",
            10.0,
            50.0,
            -90.0,
            30.0,
            20.0,
            generate_openutau=True,
            generate_openutau_aliases_fn=lambda alias: [alias, f"- {alias}"],
            alias_out_fn=lambda alias: alias.upper(),
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("a.wav=KA,10.00"))
        self.assertIn("- KA", rows[1])


if __name__ == "__main__":
    unittest.main()
