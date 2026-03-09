import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_diagnostics import MappingTraceCollector, SkippedEntryCollector


class OtoDiagnosticsTests(unittest.TestCase):
    def test_skipped_entry_collector_groups_and_parses_alias(self):
        collector = SkippedEntryCollector()
        collector.record("row_inactive_candidate", "a.wav", "a.wav=go,0,0,0,0,0", meta={"diag_hint": "idx=1"})
        collector.record("row_inactive_candidate", "b.wav", "b.wav=gyo,0,0,0,0,0")
        collector.record("row_invalid_candidate_index", "c.wav", "c.wav=,0,0,0,0,0")

        self.assertEqual(collector.total(), 3)
        self.assertEqual(collector.grouped_counts()["row_inactive_candidate"], 2)
        self.assertEqual(collector.entries[0]["alias"], "go")

    def test_mapping_trace_collector_respects_enabled_flag(self):
        disabled = MappingTraceCollector(enabled=False)
        disabled.append({"event": "x"})
        self.assertEqual(disabled.count(), 0)

        enabled = MappingTraceCollector(enabled=True)
        enabled.append({"event": "x"})
        enabled.append(None)
        self.assertEqual(enabled.count(), 1)
        self.assertEqual(enabled.records[0]["event"], "x")


if __name__ == "__main__":
    unittest.main()
