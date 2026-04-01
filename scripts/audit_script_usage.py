from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MANIFEST_PATH = SCRIPTS_DIR / "scripts_manifest.json"
SCRIPT_EXTS = {".py", ".ps1", ".bat"}
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
    ".iss",
    ".spec",
}
MAX_SCAN_FILE_SIZE = 2 * 1024 * 1024


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
            # Ignore deleted-but-not-staged paths in working tree audit mode.
            continue
        result.append(rel)
    return result


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _load_manifest() -> Dict[str, Dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        payload = json.loads(_read_text_safe(MANIFEST_PATH) or "{}")
    except Exception:
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for entry in list(payload.get("entries", []) or []):
        rel = _norm_rel(str(entry.get("path", "") or ""))
        if not rel:
            continue
        result[rel] = {
            "status": str(entry.get("status", "unclassified") or "unclassified").strip().lower(),
            "owner": str(entry.get("owner", "") or "").strip(),
            "note": str(entry.get("note", "") or "").strip(),
        }
    return result


def _list_script_files(tracked_files: List[str]) -> List[str]:
    out: List[str] = []
    for rel in tracked_files:
        rel_norm = _norm_rel(rel)
        if not rel_norm.startswith("scripts/"):
            continue
        if rel_norm.startswith("scripts/logs/") or "/__pycache__/" in rel_norm:
            continue
        suffix = Path(rel_norm).suffix.lower()
        if suffix in SCRIPT_EXTS:
            out.append(rel_norm)
    return sorted(set(out))


def _list_text_scan_targets(tracked_files: List[str]) -> List[str]:
    out: List[str] = []
    for rel in tracked_files:
        rel_norm = _norm_rel(rel)
        p = REPO_ROOT / rel_norm
        suffix = p.suffix.lower()
        if suffix not in TEXT_EXTS:
            continue
        try:
            if p.stat().st_size > MAX_SCAN_FILE_SIZE:
                continue
        except OSError:
            continue
        out.append(rel_norm)
    return sorted(set(out))


def _classify_refs(script_rel: str, refs: List[str]) -> Tuple[List[str], List[str]]:
    external = []
    internal = []
    for ref in refs:
        ref_norm = _norm_rel(ref)
        if ref_norm == script_rel:
            continue
        if ref_norm.startswith("scripts/"):
            internal.append(ref_norm)
        else:
            external.append(ref_norm)
    return sorted(set(external)), sorted(set(internal))


def build_usage_report() -> Dict[str, object]:
    tracked = _run_git_ls_files()
    scripts = _list_script_files(tracked)
    manifest = _load_manifest()
    scan_targets = _list_text_scan_targets(tracked)

    rows: List[Dict[str, object]] = []
    missing_manifest = sorted([s for s in scripts if s not in manifest])
    missing_script_files = sorted([m for m in manifest if m not in scripts])

    for script_rel in scripts:
        script_name = Path(script_rel).name
        path_tokens = {script_rel, script_rel.replace("/", "\\"), script_name}
        refs: List[str] = []
        for file_rel in scan_targets:
            text = _read_text_safe(REPO_ROOT / file_rel)
            if not text:
                continue
            if any(token and token in text for token in path_tokens):
                refs.append(file_rel)
        external_refs, internal_refs = _classify_refs(script_rel, refs)

        meta = manifest.get(script_rel, {})
        status = str(meta.get("status", "unclassified") or "unclassified").strip().lower()
        protected = status in {"protected", "manual"}
        delete_trackable = status in {"candidate_delete", "unclassified"}
        safe_delete_candidate = (
            (not protected)
            and delete_trackable
            and len(external_refs) == 0
            and len(internal_refs) == 0
        )

        rows.append(
            {
                "path": script_rel,
                "status": status,
                "owner": str(meta.get("owner", "") or "").strip(),
                "external_refs": external_refs,
                "internal_refs": internal_refs,
                "external_ref_count": len(external_refs),
                "internal_ref_count": len(internal_refs),
                "safe_delete_candidate": bool(safe_delete_candidate),
                "note": str(meta.get("note", "") or "").strip(),
            }
        )

    safe_delete_paths = sorted([r["path"] for r in rows if bool(r.get("safe_delete_candidate"))])
    report = {
        "repo_root": str(REPO_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "tracked_script_count": len(scripts),
        "manifest_entry_count": len(manifest),
        "missing_manifest_entries": missing_manifest,
        "stale_manifest_entries": missing_script_files,
        "safe_delete_candidates": safe_delete_paths,
        "scripts": rows,
    }
    return report


def _print_text_summary(report: Dict[str, object]) -> None:
    print(f"[AUDIT] scripts={report.get('tracked_script_count', 0)}")
    print(f"[AUDIT] manifest_entries={report.get('manifest_entry_count', 0)}")
    missing = list(report.get("missing_manifest_entries", []) or [])
    stale = list(report.get("stale_manifest_entries", []) or [])
    if missing:
        print(f"[AUDIT] missing_manifest_entries={len(missing)}")
        for item in missing:
            print(f"  - {item}")
    if stale:
        print(f"[AUDIT] stale_manifest_entries={len(stale)}")
        for item in stale:
            print(f"  - {item}")
    safe = list(report.get("safe_delete_candidates", []) or [])
    print(f"[AUDIT] safe_delete_candidates={len(safe)}")
    for item in safe:
        print(f"  - {item}")


def _print_json_console(report: Dict[str, object]) -> None:
    try:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        # Fallback for legacy Windows code pages.
        print(json.dumps(report, ensure_ascii=True, indent=2))


def _check_delete_target(report: Dict[str, object], target: str) -> int:
    target_norm = _norm_rel(target)
    rows = list(report.get("scripts", []) or [])
    row = next((r for r in rows if _norm_rel(str(r.get("path", ""))) == target_norm), None)
    if row is None:
        print(f"[CHECK-DELETE] target_not_found={target_norm}")
        return 3
    if bool(row.get("safe_delete_candidate", False)):
        print(f"[CHECK-DELETE] safe=1 target={target_norm}")
        return 0
    print(f"[CHECK-DELETE] safe=0 target={target_norm}")
    print(
        f"[CHECK-DELETE] refs external={row.get('external_ref_count', 0)} "
        f"internal={row.get('internal_ref_count', 0)} status={row.get('status', 'unknown')}"
    )
    return 2


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit scripts usage to support safe script cleanup/refactoring."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    parser.add_argument("--output", default="", help="Optional path to write JSON report.")
    parser.add_argument(
        "--check-delete",
        default="",
        help="Check if a script is safe delete candidate. Returns non-zero when unsafe.",
    )
    args = parser.parse_args(argv)

    report = build_usage_report()
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.check_delete:
        rc = _check_delete_target(report, args.check_delete)
        if args.json:
            _print_json_console(report)
        else:
            _print_text_summary(report)
        return rc

    if args.json:
        _print_json_console(report)
    else:
        _print_text_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
