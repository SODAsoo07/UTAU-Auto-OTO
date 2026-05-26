from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "scripts" / "logs" / "candidate_delete_dry_run.json"


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


def _extract_wrapper_target(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(r'["\']([^"\']+)["\']\s*\)\s*$', text, flags=re.MULTILINE)
    if not m:
        return ""
    return str(m.group(1)).strip()


def build_dry_run_report() -> Dict[str, object]:
    build_usage_report = _load_build_usage_report()
    usage = build_usage_report()
    rows: List[Dict[str, object]] = list(usage.get("scripts", []) or [])
    candidates = [r for r in rows if str(r.get("status", "")) == "candidate_delete"]
    payload: List[Dict[str, object]] = []
    for row in candidates:
        rel = str(row.get("path", ""))
        abs_path = REPO_ROOT / rel
        exists = abs_path.is_file()
        external_refs = list(row.get("external_refs", []) or [])
        internal_refs = list(row.get("internal_refs", []) or [])
        target_hint = _extract_wrapper_target(abs_path) if exists else ""
        safe_remove = exists and len(external_refs) == 0
        payload.append(
            {
                "path": rel,
                "exists": exists,
                "external_ref_count": len(external_refs),
                "internal_ref_count": len(internal_refs),
                "external_refs": external_refs,
                "internal_refs": internal_refs,
                "wrapper_target_hint": target_hint,
                "safe_remove_now": safe_remove,
                "planned_action": ("delete_file" if safe_remove else "keep_for_now"),
            }
        )
    return {
        "schema": "candidate_delete_dry_run_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": str(REPO_ROOT),
        "candidate_count": len(payload),
        "delete_now_count": sum(1 for r in payload if bool(r.get("safe_remove_now"))),
        "candidates": payload,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run deletion planner for candidate_delete scripts.")
    parser.add_argument("--json", action="store_true", help="Print JSON result to stdout.")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output JSON path.")
    args = parser.parse_args(argv)

    report = build_dry_run_report()
    out_path = Path(str(args.output))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[DRY-RUN] report={out_path}")
        print(
            f"[DRY-RUN] candidates={report.get('candidate_count', 0)} "
            f"delete_now={report.get('delete_now_count', 0)}"
        )
        for row in report.get("candidates", []) or []:
            print(
                "  - {0} | ext={1} int={2} | action={3}".format(
                    row.get("path", ""),
                    row.get("external_ref_count", 0),
                    row.get("internal_ref_count", 0),
                    row.get("planned_action", ""),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
