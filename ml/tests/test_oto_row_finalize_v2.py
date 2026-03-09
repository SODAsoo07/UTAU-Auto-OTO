import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_row_finalize_v2 import apply_anchor_record, commit_row_artifacts


class OtoRowFinalizeV2Tests(unittest.TestCase):
    def test_apply_anchor_record_updates_store(self):
        store = {}
        changed = apply_anchor_record(store, (3, {"a": 1}))
        self.assertTrue(changed)
        self.assertEqual(store[3]["a"], 1)

    def test_commit_row_artifacts_logs_messages_and_appends_rows(self):
        store = {}
        lines = []
        logs = []
        commit_row_artifacts(
            lines,
            ["a.wav=ka,0,0,0,0,0"],
            anchor_store=store,
            anchor_record=(1, {"ok": True}),
            log_fn=logs.append,
            messages=["m1", "", None, "m2"],
        )
        self.assertEqual(lines, ["a.wav=ka,0,0,0,0,0"])
        self.assertEqual(store[1], {"ok": True})
        self.assertEqual(logs, ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
