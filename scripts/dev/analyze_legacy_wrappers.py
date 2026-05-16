from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "scripts" / "logs" / "legacy_wrapper_usage_report.json"


def _load_build_usage_report():
    target = REPO_ROOT / "scripts" / "dev" / "audit_script_usage.py"
    spec = importlib.util.spec_from_file_location("_audit_script_usage", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load audit module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "build_usage_report", None)
    if fn is None:
        raise RuntimeError("build_usage_report not found in audit_script_usage.py")
    return fn


def _bucket_ref(path: str) -> str:
    rel = str(path or "").replace("\\", "/").strip().lower()
    if rel.startswith(".github/"):
        return "ci"
    if rel.startswith("plans/") or rel.startswith("docs/") or "readme" in rel or "obsidian" in rel:
        return "docs"
    if rel.startswith("scripts/") or rel.startswith("tests/scripts/"):
        return "script"
    if rel.startswith("ml/tests/") or rel.endswith("_test.py") or "/test_" in rel:
        return "test"
    return "code"


def _summarize_legacy_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        if str(row.get("status", "")) != "legacy":
            continue
        external_refs = list(row.get("external_refs", []) or [])
        internal_refs = list(row.get("internal_refs", []) or [])
        bucket_counts: Dict[str, int] = {}
        for ref in external_refs:
            bucket = _bucket_ref(str(ref))
            bucket_counts[bucket] = int(bucket_counts.get(bucket, 0)) + 1

        ignorable_internal_prefixes = (
            "scripts/readme.md",
            "scripts/scripts_manifest.json",
        )
        non_ignorable_internal_refs = [
            r
            for r in internal_refs
            if not str(r or "").strip().lower().replace("\\", "/").startswith(ignorable_internal_prefixes)
        ]
        can_promote_candidate_delete = (
            len(external_refs) == 0 and len(non_ignorable_internal_refs) == 0
        )
        recommendation = (
            "candidate_delete"
            if can_promote_candidate_delete
            else "keep_legacy"
        )
        out.append(
            {
                "path": str(row.get("path", "")),
                "owner": str(row.get("owner", "")),
                "note": str(row.get("note", "")),
                "external_ref_count": len(external_refs),
                "internal_ref_count": len(internal_refs),
                "external_ref_buckets": bucket_counts,
                "external_refs": external_refs,
                "internal_refs": internal_refs,
                "non_ignorable_internal_refs": non_ignorable_internal_refs,
                "recommendation": recommendation,
                "reason": (
                    "no external refs and only README/manifest internal refs"
                    if recommendation == "candidate_delete"
                    else "still referenced in docs/ci/code; keep wrapper"
                ),
            }
        )
    return out


def build_legacy_wrapper_report() -> Dict[str, object]:
    build_usage_report = _load_build_usage_report()
    usage = build_usage_report()
    rows = list(usage.get("scripts", []) or [])
    legacy_rows = _summarize_legacy_rows(rows)
    summary = {
        "legacy_count": len(legacy_rows),
        "candidate_delete_recommendation_count": sum(
            1 for row in legacy_rows if row.get("recommendation") == "candidate_delete"
        ),
        "keep_legacy_count": sum(
            1 for row in legacy_rows if row.get("recommendation") == "keep_legacy"
        ),
    }
    return {
        "schema": "legacy_wrapper_usage_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": str(REPO_ROOT),
        "summary": summary,
        "legacy_wrappers": legacy_rows,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze legacy wrapper usage and suggest candidate_delete promotion readiness."
    )
    parser.add_argument("--json", action="store_true", help="Print report JSON to stdout.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help="Output JSON path (default: scripts/logs/legacy_wrapper_usage_report.json).",
    )
    args = parser.parse_args(argv)

    report = build_legacy_wrapper_report()
    out_path = Path(str(args.output)).expanduser()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[LEGACY-AUDIT] report={out_path}")
        summary = report.get("summary", {})
        print(
            "[LEGACY-AUDIT] legacy={0} keep={1} candidate_delete={2}".format(
                summary.get("legacy_count", 0),
                summary.get("keep_legacy_count", 0),
                summary.get("candidate_delete_recommendation_count", 0),
            )
        )
        for row in report.get("legacy_wrappers", []) or []:
            print(
                "  - {0} | ext={1} int={2} | rec={3}".format(
                    row.get("path", ""),
                    row.get("external_ref_count", 0),
                    row.get("internal_ref_count", 0),
                    row.get("recommendation", ""),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
