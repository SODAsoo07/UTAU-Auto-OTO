from __future__ import annotations


class GeneratorDiagnosticsAdapter:
    def __init__(self, *, skipped_collector, log_fn, trace_collector=None):
        self._skipped = skipped_collector
        self._log_fn = log_fn
        self._trace = trace_collector

    def record_unset(self, reason, fname, line, meta=None):
        self._skipped.record(reason, fname, line, meta=meta)

    def record_unset_lines(self, reason, fname, src_lines, meta=None):
        self._skipped.record_lines(reason, fname, src_lines, meta=meta)

    def log_unset_summary(self):
        self._skipped.log_summary(self._log_fn)

    def append_mapping_trace(self, record):
        if self._trace is None:
            return
        self._trace.append(record)


__all__ = ["GeneratorDiagnosticsAdapter"]
