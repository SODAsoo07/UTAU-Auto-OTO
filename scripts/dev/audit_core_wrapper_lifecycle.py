from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "core"
DEFAULT_OUT = REPO_ROOT / "scripts" / "logs" / "core_wrapper_lifecycle_report.json"

TEXT_EXTS = {
    ".py",
    ".ps1",
    ".bat",
    ".cmd",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".spec",
}
MAX_SCAN_FILE_SIZE = 2 * 1024 * 1024

WRAPPER_PREFIX = '"""Compatibility wrapper: moved to `core.'
TARGET_RE = re.compile(r'_import_module\("([^"]+)"\)')

IGNORABLE_INTERNAL_PREFIXES = (
    "core/readme.md",
    "core/__init__.py",
)
RUNTIME_REF_PREFIXES = (
    "core/alignment/",
    "core/generation/",
    "core/coarse_crnn/",
    "core/oto_ml/",
    "core/cvn/",
    "core/runtime/",
    "core/timing/",
    "core/common/",
    "ui/",
    "scripts/runtime/",
    "scripts/build/",
)
RUNTIME_REF_EXACT = {
    "main.py",
    "build.py",
    "setup_mfa.bat",
}


def _norm_rel(path_text: str) -> str:
    return str(path_text or "").strip().replace("\\", "/")


def _run_git_ls_files() -> List[str]:
    cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.check_output(cmd, cwd=str(REPO_ROOT))
    rows = [p for p in out.decode("utf-8", errors="surrogateescape").split("\x00") if p]
    result: List[str] = []
    for row in rows:
        rel = _norm_rel(row)
        if not rel:
            continue
        if not (REPO_ROOT / rel).exists():
            continue
        result.append(rel)
    return result


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _list_text_scan_targets(tracked_files: List[str]) -> List[str]:
    out: List[str] = []
    for rel in tracked_files:
        rel_norm = _norm_rel(rel)
        p = REPO_ROOT / rel_norm
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            if p.stat().st_size > MAX_SCAN_FILE_SIZE:
                continue
        except OSError:
            continue
        out.append(rel_norm)
    return sorted(set(out))


def _list_core_wrappers() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for file_path in sorted(CORE_DIR.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        text = _read_text_safe(file_path).lstrip()
        if not text.startswith(WRAPPER_PREFIX):
            continue
        module = file_path.stem
        m = TARGET_RE.search(text)
        target = str(m.group(1)).strip() if m else ""
        out.append((module, target))
    return out


def _is_runtime_ref(rel: str) -> bool:
    r = _norm_rel(rel).lower()
    if r in RUNTIME_REF_EXACT:
        return True
    return any(r.startswith(prefix) for prefix in RUNTIME_REF_PREFIXES)


def _is_doc_ref(rel: str) -> bool:
    r = _norm_rel(rel).lower()
    return r.endswith(".md") or r.startswith("docs/") or r.startswith("plans/") or r.startswith(".github/")


def _bucket_ref(path: str) -> str:
    rel = _norm_rel(path).lower()
    if _is_doc_ref(rel):
        return "docs"
    if rel.startswith("tests/") or "/test_" in rel or rel.endswith("_test.py"):
        return "test"
    if rel.startswith("scripts/"):
        return "script"
    if rel.startswith("core/"):
        return "core"
    return "code"


def _collect_refs(module: str, scan_targets: List[str]) -> List[str]:
    rel_path = f"core/{module}.py"
    path_tokens = {
        rel_path,
        rel_path.replace("/", "\\"),
        f"core.{module}",
        f"from core import {module}",
        f"import core.{module}",
    }
    refs: List[str] = []
    for file_rel in scan_targets:
        text = _read_text_safe(REPO_ROOT / file_rel)
        if not text:
            continue
        if any(token in text for token in path_tokens):
            refs.append(file_rel)
    # Exclude self reference row.
    return sorted(set(r for r in refs if _norm_rel(r) != rel_path))


def build_core_wrapper_lifecycle_report() -> Dict[str, object]:
    tracked = _run_git_ls_files()
    scan_targets = _list_text_scan_targets(tracked)
    wrappers = _list_core_wrappers()

    rows: List[Dict[str, object]] = []
    for module, target in wrappers:
        path = f"core/{module}.py"
        refs = _collect_refs(module, scan_targets)
        internal_refs = [r for r in refs if _norm_rel(r).startswith("core/")]
        external_refs = [r for r in refs if not _norm_rel(r).startswith("core/")]
        non_ignorable_internal = [
            r
            for r in internal_refs
            if not _norm_rel(r).lower().startswith(IGNORABLE_INTERNAL_PREFIXES)
        ]
        non_doc_external = [r for r in external_refs if not _is_doc_ref(r)]
        runtime_refs = [r for r in refs if _is_runtime_ref(r)]

        if runtime_refs:
            status = "active"
            reason = "runtime/build path references this wrapper"
        elif non_doc_external or non_ignorable_internal:
            status = "legacy"
            reason = "still referenced outside docs/manifest-only scope"
        else:
            status = "candidate_delete"
            reason = "only docs/manifest-level references remain"

        buckets: Dict[str, int] = {}
        for ref in refs:
            b = _bucket_ref(ref)
            buckets[b] = int(buckets.get(b, 0)) + 1

        rows.append(
            {
                "path": path,
                "module": module,
                "target_module": target,
                "status": status,
                "reason": reason,
                "ref_count": len(refs),
                "internal_ref_count": len(internal_refs),
                "external_ref_count": len(external_refs),
                "non_ignorable_internal_refs": non_ignorable_internal,
                "non_doc_external_refs": non_doc_external,
                "runtime_refs": runtime_refs,
                "ref_buckets": buckets,
                "refs": refs,
            }
        )

    rows = sorted(rows, key=lambda r: str(r.get("path", "")))
    summary = {
        "wrapper_count": len(rows),
        "active_count": sum(1 for r in rows if r.get("status") == "active"),
        "legacy_count": sum(1 for r in rows if r.get("status") == "legacy"),
        "candidate_delete_count": sum(1 for r in rows if r.get("status") == "candidate_delete"),
    }
    return {
        "schema": "core_wrapper_lifecycle_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": str(REPO_ROOT),
        "summary": summary,
        "wrappers": rows,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit core root-wrapper lifecycle status (active/legacy/candidate_delete)."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help="Output JSON path (default: scripts/logs/core_wrapper_lifecycle_report.json).",
    )
    parser.add_argument(
        "--enforce-clean",
        action="store_true",
        help="Return non-zero when legacy/candidate_delete wrappers are present.",
    )
    parser.add_argument(
        "--max-legacy",
        type=int,
        default=0,
        help="Allowed legacy wrapper count when --enforce-clean is set (default: 0).",
    )
    parser.add_argument(
        "--max-candidate-delete",
        type=int,
        default=0,
        help="Allowed candidate_delete wrapper count when --enforce-clean is set (default: 0).",
    )
    args = parser.parse_args(argv)

    report = build_core_wrapper_lifecycle_report()
    out_path = Path(str(args.output)).expanduser()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report.get("summary", {})
        print(f"[CORE-WRAPPER-AUDIT] report={out_path}")
        print(
            "[CORE-WRAPPER-AUDIT] wrappers={0} active={1} legacy={2} candidate_delete={3}".format(
                summary.get("wrapper_count", 0),
                summary.get("active_count", 0),
                summary.get("legacy_count", 0),
                summary.get("candidate_delete_count", 0),
            )
        )
    if args.enforce_clean:
        summary = report.get("summary", {})
        legacy_count = int(summary.get("legacy_count", 0) or 0)
        candidate_delete_count = int(summary.get("candidate_delete_count", 0) or 0)
        if legacy_count > int(args.max_legacy) or candidate_delete_count > int(args.max_candidate_delete):
            print(
                "[CORE-WRAPPER-AUDIT] enforce_clean_failed "
                f"(legacy={legacy_count}, candidate_delete={candidate_delete_count}, "
                f"max_legacy={int(args.max_legacy)}, max_candidate_delete={int(args.max_candidate_delete)})"
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
