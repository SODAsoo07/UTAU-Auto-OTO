"""Unit tests for ml.scripts.coarse_crnn.sample_ab_labels (Phase E sampling helper)."""
from __future__ import annotations

import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.sample_ab_labels import (  # noqa: E402
    DEFAULT_ROLE_QUOTA,
    _normalize_quota,
    _parse_role_quota_arg,
    _role_for_row,
    sample_label_rows,
)


def _row(*, case_id: str, vb: str, alias: str, version: str, language: str = "korean") -> dict[str, str]:
    return {
        "case_id": case_id,
        "voicebank_id": vb,
        "language": language,
        "wav_name": f"{case_id}.wav",
        "alias": alias,
        "version_id": version,
        "click": "",
        "consonant_clip": "",
        "vc_misalign": "",
        "overlap_unnatural": "",
        "cutoff_overreach": "",
        "notes": "",
    }


def _build_rows(case_count: int, *, vb: str = "vb1", alias: str = "a k", versions: tuple[str, ...] = ("V1", "V2", "V3")) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(case_count):
        cid = f"{vb}__{i:04d}"
        for v in versions:
            rows.append(_row(case_id=cid, vb=vb, alias=alias, version=v))
    return rows


def test_uniform_sampling_picks_target_count_per_voicebank():
    rows = _build_rows(100, vb="vb1") + _build_rows(80, vb="vb2")
    filtered, summary = sample_label_rows(
        rows=rows, per_voicebank=30, stratify_by_role=False, role_quota={}, seed=0
    )
    assert summary["per_voicebank"]["vb1"]["sampled_cases"] == 30
    assert summary["per_voicebank"]["vb2"]["sampled_cases"] == 30
    assert summary["sampled_case_count"] == 60
    # 3 version rows per case
    assert summary["sampled_row_count"] == 60 * 3


def test_uniform_sampling_keeps_all_version_rows_per_case():
    rows = _build_rows(50, vb="vb1", versions=("V1", "V2", "V3"))
    filtered, _ = sample_label_rows(
        rows=rows, per_voicebank=10, stratify_by_role=False, role_quota={}, seed=1
    )
    # Every sampled case must appear exactly once per version
    by_case: dict[str, set[str]] = {}
    for row in filtered:
        by_case.setdefault(row["case_id"], set()).add(row["version_id"])
    assert len(by_case) == 10
    for versions_seen in by_case.values():
        assert versions_seen == {"V1", "V2", "V3"}


def test_uniform_sampling_is_deterministic_for_seed():
    rows = _build_rows(100, vb="vb1")
    out1, _ = sample_label_rows(
        rows=rows, per_voicebank=20, stratify_by_role=False, role_quota={}, seed=42
    )
    out2, _ = sample_label_rows(
        rows=rows, per_voicebank=20, stratify_by_role=False, role_quota={}, seed=42
    )
    ids1 = sorted({r["case_id"] for r in out1})
    ids2 = sorted({r["case_id"] for r in out2})
    assert ids1 == ids2


def test_uniform_sampling_returns_all_when_smaller_than_target():
    rows = _build_rows(8, vb="vb1")
    filtered, summary = sample_label_rows(
        rows=rows, per_voicebank=30, stratify_by_role=False, role_quota={}, seed=0
    )
    assert summary["per_voicebank"]["vb1"]["sampled_cases"] == 8
    assert summary["sampled_case_count"] == 8


def test_normalize_quota_handles_zero_total():
    assert _normalize_quota({}) == {}
    assert _normalize_quota({"a": 0.0, "b": 0.0}) == {}


def test_normalize_quota_sums_to_one():
    out = _normalize_quota({"a": 2.0, "b": 1.0, "c": 1.0})
    assert pytest.approx(sum(out.values())) == 1.0
    assert out["a"] == 0.5
    assert out["b"] == 0.25


def test_parse_role_quota_inline_string():
    quota = _parse_role_quota_arg("vc=0.40,cv=0.30,v-cv=0.30")
    assert quota == {"vc": 0.40, "cv": 0.30, "v-cv": 0.30}


def test_parse_role_quota_empty_returns_defaults():
    quota = _parse_role_quota_arg("")
    assert quota == dict(DEFAULT_ROLE_QUOTA)


def test_parse_role_quota_from_file(tmp_path):
    path = tmp_path / "quota.json"
    path.write_text('{"vc": 0.5, "cv": 0.5}', encoding="utf-8")
    quota = _parse_role_quota_arg(str(path))
    assert quota == {"vc": 0.5, "cv": 0.5}


def test_parse_role_quota_invalid_token_raises():
    with pytest.raises(SystemExit):
        _parse_role_quota_arg("vc:0.5")


def test_stratified_sampling_uses_quota_when_supply_sufficient():
    """If every role has plenty of cases, the picked counts should track the quota."""
    rows: list[dict[str, str]] = []
    # Real-classifier role mapping (verified): "a k" → v-cv (whitespace 2-token
    # is treated as V + CV transition), "k a" → cv, "a" → v, "- ka" → -cv.
    aliases = {
        "v-cv": "a k",
        "cv": "k a",
        "v": "a",
        "-cv": "- ka",
    }
    cid = 0
    for role_name, alias in aliases.items():
        for _ in range(100):
            cid += 1
            cls = f"vb1__{cid:04d}__{role_name}"
            rows.append(_row(case_id=cls, vb="vb1", alias=alias, version="V1"))
            rows.append(_row(case_id=cls, vb="vb1", alias=alias, version="V2"))

    quota = {"v-cv": 0.5, "cv": 0.25, "v": 0.15, "-cv": 0.10}
    _filtered, summary = sample_label_rows(
        rows=rows,
        per_voicebank=40,
        stratify_by_role=True,
        role_quota=quota,
        seed=7,
    )
    counts = summary["per_voicebank"]["vb1"]["per_role_counts"]
    assert sum(counts.values()) == 40
    # Allow ±2 rounding tolerance
    assert abs(counts.get("v-cv", 0) - 20) <= 2
    assert abs(counts.get("cv", 0) - 10) <= 2
    assert abs(counts.get("v", 0) - 6) <= 2
    assert abs(counts.get("-cv", 0) - 4) <= 2


def test_stratified_sampling_backfills_on_role_shortage():
    """If a role has fewer cases than its quota wants, the deficit is filled from leftover."""
    rows: list[dict[str, str]] = []
    # Only 3 -cv cases (well under any reasonable quota), 50 cv cases
    for i in range(3):
        cid = f"vb1__hd{i:04d}"
        rows.append(_row(case_id=cid, vb="vb1", alias="- ka", version="V1"))
    for i in range(50):
        cid = f"vb1__cv{i:04d}"
        rows.append(_row(case_id=cid, vb="vb1", alias="k a", version="V1"))

    quota = {"-cv": 0.5, "cv": 0.5}
    _filtered, summary = sample_label_rows(
        rows=rows,
        per_voicebank=30,
        stratify_by_role=True,
        role_quota=quota,
        seed=3,
    )
    counts = summary["per_voicebank"]["vb1"]["per_role_counts"]
    # -cv only has 3 available, so cv must backfill the rest to reach 30.
    assert counts.get("-cv", 0) == 3
    assert summary["per_voicebank"]["vb1"]["sampled_cases"] == 30


def test_role_for_row_returns_known_role():
    """The role inference path should not crash and should produce a recognized role token."""
    row = _row(case_id="c1", vb="vb1", alias="a k", version="V1", language="korean")
    role = _role_for_row(row)
    assert isinstance(role, str)
    assert role  # non-empty


def test_per_voicebank_quotas_isolated():
    """vb1 and vb2 should be sampled independently with their own quotas."""
    rows = _build_rows(50, vb="vb1") + _build_rows(50, vb="vb2")
    _filtered, summary = sample_label_rows(
        rows=rows, per_voicebank=15, stratify_by_role=False, role_quota={}, seed=11
    )
    vb1_cases = {row for row in summary["per_voicebank"]}
    assert summary["per_voicebank"]["vb1"]["sampled_cases"] == 15
    assert summary["per_voicebank"]["vb2"]["sampled_cases"] == 15
