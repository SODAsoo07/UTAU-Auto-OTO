from __future__ import annotations


class SkippedEntryCollector:
    def __init__(self):
        self.entries = []

    def record(self, reason, fname, line, meta=None):
        raw = (line or "").rstrip("\n")
        alias = ""
        if "=" in raw:
            rhs = raw.split("=", 1)[1]
            alias = rhs.split(",", 1)[0].strip()
        self.entries.append(
            {
                "reason": str(reason or ""),
                "file": str(fname or ""),
                "alias": alias,
                "line": raw,
                "meta": dict(meta or {}),
            }
        )

    def record_lines(self, reason, fname, src_lines, meta=None):
        for raw in (src_lines or []):
            self.record(reason, fname, raw, meta=meta)

    def total(self):
        return len(self.entries)

    def grouped_counts(self):
        out = {}
        for item in self.entries:
            reason = str(item.get("reason", ""))
            out[reason] = out.get(reason, 0) + 1
        return out

    def log_summary(self, log_fn, *, max_examples=5):
        total_unset = self.total()
        if total_unset <= 0:
            log_fn("[Auto-OTO] 자동 설정 제외 항목: 0")
            return
        log_fn(f"[Auto-OTO] 자동 설정 제외 항목: {total_unset}건")
        for reason, count in sorted(self.grouped_counts().items(), key=lambda x: (-x[1], x[0])):
            log_fn(f"  - {reason}: {count}")
            shown = 0
            for item in self.entries:
                if item.get("reason") != reason:
                    continue
                alias_txt = item["alias"] if item["alias"] else "<empty>"
                meta = item.get("meta") or {}
                extra = f" | {meta.get('diag_hint')}" if meta.get("diag_hint") else ""
                log_fn(f"    예시: {item['file']} | alias={alias_txt}{extra}")
                shown += 1
                if shown >= int(max_examples):
                    break


class MappingTraceCollector:
    def __init__(self, *, enabled=True):
        self.enabled = bool(enabled)
        self.records = []

    def append(self, record):
        if not self.enabled:
            return
        if not isinstance(record, dict):
            return
        self.records.append(record)

    def count(self):
        return len(self.records)


__all__ = [
    "MappingTraceCollector",
    "SkippedEntryCollector",
]
