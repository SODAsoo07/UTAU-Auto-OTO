import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_diagnostics import MappingTraceCollector, SkippedEntryCollector
from core.oto_diagnostics_adapter_v2 import GeneratorDiagnosticsAdapter


class OtoDiagnosticsAdapterV2Tests(unittest.TestCase):
    def test_adapter_records_unset_and_trace(self):
        skipped = SkippedEntryCollector()
        trace = MappingTraceCollector(enabled=True)
        logs = []
        adapter = GeneratorDiagnosticsAdapter(
            skipped_collector=skipped,
            log_fn=logs.append,
            trace_collector=trace,
        )

        adapter.record_unset("reason_a", "a.wav", "a.wav=ka,0,0,0,0,0")
        adapter.record_unset_lines("reason_b", "b.wav", ["b.wav=ki,0,0,0,0,0"])
        adapter.append_mapping_trace({"event": "trace"})
        adapter.log_unset_summary()

        self.assertEqual(skipped.total(), 2)
        self.assertEqual(trace.count(), 1)
        self.assertTrue(any("reason_a" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
