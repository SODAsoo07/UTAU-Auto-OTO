import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_row_finalize_v2 import finalize_ja_row


class JaRowFinalizeV2Tests(unittest.TestCase):
    def test_finalize_ja_row_commits_rows_anchor_and_messages(self):
        lines = []
        logs = []
        anchor_store = {}

        finalize_ja_row(
            final_lines=lines,
            row_builder_fn=lambda *args, **kwargs: ["a.wav=KA,1,2,3,4,5"],
            real_wav_name="a.wav",
            alias="ka",
            offset=1.0,
            consonant=2.0,
            cutoff=-3.0,
            pre=4.0,
            ovl=5.0,
            generate_openutau=False,
            generate_openutau_aliases_fn=lambda alias: [alias],
            alias_out_fn=lambda alias: alias.upper(),
            anchor_store=anchor_store,
            anchor_record=(2, {"ok": True}),
            log_fn=logs.append,
            messages=["m1"],
        )

        self.assertEqual(lines, ["a.wav=KA,1,2,3,4,5"])
        self.assertEqual(anchor_store[2], {"ok": True})
        self.assertEqual(logs, ["m1"])


if __name__ == "__main__":
    unittest.main()
