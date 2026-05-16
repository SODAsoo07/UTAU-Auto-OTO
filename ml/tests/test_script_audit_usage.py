import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_script_usage.py"


def _run_audit_json():
    proc = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    return payload


def test_script_audit_has_no_manifest_drift():
    payload = _run_audit_json()
    assert payload.get("missing_manifest_entries", []) == []
    assert payload.get("stale_manifest_entries", []) == []


def test_script_audit_status_values_are_known():
    payload = _run_audit_json()
    allowed = {"protected", "manual", "active", "legacy", "candidate_delete", "unclassified"}
    for row in payload.get("scripts", []):
        assert str(row.get("status", "")) in allowed


def test_candidate_delete_rows_are_strictly_unreferenced():
    payload = _run_audit_json()
    for row in payload.get("scripts", []):
        if str(row.get("status", "")) != "candidate_delete":
            continue
        assert int(row.get("external_ref_count", 0)) == 0
        assert int(row.get("internal_ref_count", 0)) == 0
        assert bool(row.get("safe_delete_candidate", False)) is True


def test_manual_scripts_live_under_tests_scripts():
    payload = _run_audit_json()
    for row in payload.get("scripts", []):
        if str(row.get("status", "")) != "manual":
            continue
        path = str(row.get("path", ""))
        assert path.startswith("tests/scripts/"), path
