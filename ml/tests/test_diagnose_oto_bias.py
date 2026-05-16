"""Unit tests for ml.scripts.coarse_crnn.diagnose_oto_bias."""
from __future__ import annotations

import json
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.diagnose_oto_bias import (  # noqa: E402
    ANCHOR_KEYS,
    _compute_stats,
    aggregate_per_role,
    compute_abs_anchors,
    compute_deltas,
    diagnose,
    render_markdown,
    role_for_alias,
    worst_rows,
)


def _write_oto(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# compute_abs_anchors
# ---------------------------------------------------------------------------

def test_compute_abs_anchors_basic():
    parsed = {"wav": "x.wav", "alias": "a", "offset": 100.0, "cons": 60.0, "cutoff": -200.0, "pre": 40.0, "ovl": 20.0}
    abs_pos = compute_abs_anchors(parsed)
    assert abs_pos["offset_abs"] == 100.0
    assert abs_pos["pre_abs"] == 140.0       # 100 + 40
    assert abs_pos["ovl_abs"] == 120.0       # 100 + min(40, 20) = 100 + 20
    assert abs_pos["cons_abs"] == 160.0      # 100 + max(40, 60) = 100 + 60
    assert abs_pos["cutoff_abs"] == 300.0    # 100 + |-200|


def test_compute_abs_anchors_negative_cutoff_uses_abs():
    parsed = {"wav": "x", "alias": "a", "offset": 50.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0}
    abs_pos = compute_abs_anchors(parsed)
    assert abs_pos["cutoff_abs"] == 150.0


def test_compute_abs_anchors_positive_cutoff_offset_relative():
    parsed = {"wav": "x", "alias": "a", "offset": 50.0, "cons": 0.0, "cutoff": 100.0, "pre": 0.0, "ovl": 0.0}
    abs_pos = compute_abs_anchors(parsed)
    # Positive cutoff is treated as offset-relative by the same convention.
    assert abs_pos["cutoff_abs"] == 150.0


def test_compute_abs_anchors_clamps_negative_offset():
    parsed = {"wav": "x", "alias": "a", "offset": -10.0, "cons": 5.0, "cutoff": -100.0, "pre": 5.0, "ovl": 5.0}
    abs_pos = compute_abs_anchors(parsed)
    assert abs_pos["offset_abs"] == 0.0


# ---------------------------------------------------------------------------
# role_for_alias
# ---------------------------------------------------------------------------

def test_role_for_alias_returns_known_role():
    assert role_for_alias("a k", "korean") in {"v-cv", "vc", "other"}
    assert role_for_alias("- ka", "korean") == "-cv"


# ---------------------------------------------------------------------------
# compute_deltas
# ---------------------------------------------------------------------------

def test_compute_deltas_matches_by_wav_alias():
    ref = [
        {"wav": "a.wav", "alias": "a k", "offset": 100.0, "cons": 30.0, "cutoff": -200.0, "pre": 50.0, "ovl": 10.0},
        {"wav": "a.wav", "alias": "k a", "offset": 200.0, "cons": 30.0, "cutoff": -200.0, "pre": 40.0, "ovl": 10.0},
    ]
    comp = [
        {"wav": "a.wav", "alias": "a k", "offset": 110.0, "cons": 30.0, "cutoff": -200.0, "pre": 50.0, "ovl": 10.0},
        {"wav": "a.wav", "alias": "k a", "offset": 200.0, "cons": 30.0, "cutoff": -200.0, "pre": 60.0, "ovl": 10.0},
    ]
    records, counts = compute_deltas(reference_rows=ref, compared_rows=comp, language="korean")
    assert counts["matched"] == 2
    assert counts["unmatched_in_reference"] == 0
    # First row: offset_delta = 110 - 100 = +10
    assert records[0]["offset_abs"] == pytest.approx(10.0)
    # Second row: pre_delta absolute. ref pre_abs = 240, comp pre_abs = 260 → +20
    assert records[1]["pre_abs"] == pytest.approx(20.0)


def test_compute_deltas_unmatched_increments_counter():
    ref = [{"wav": "a.wav", "alias": "a k", "offset": 100.0, "cons": 30.0, "cutoff": -200.0, "pre": 50.0, "ovl": 10.0}]
    comp = [
        {"wav": "a.wav", "alias": "a k", "offset": 100.0, "cons": 30.0, "cutoff": -200.0, "pre": 50.0, "ovl": 10.0},
        {"wav": "b.wav", "alias": "missing", "offset": 0.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0},
    ]
    records, counts = compute_deltas(reference_rows=ref, compared_rows=comp, language="korean")
    assert counts["matched"] == 1
    assert counts["unmatched_in_reference"] == 1


def test_compute_deltas_handles_repeated_alias_by_occurrence():
    ref = [
        {"wav": "x.wav", "alias": "a", "offset": 100.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0},
        {"wav": "x.wav", "alias": "a", "offset": 200.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0},
    ]
    comp = [
        {"wav": "x.wav", "alias": "a", "offset": 110.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0},
        {"wav": "x.wav", "alias": "a", "offset": 210.0, "cons": 0.0, "cutoff": -100.0, "pre": 0.0, "ovl": 0.0},
    ]
    records, _ = compute_deltas(reference_rows=ref, compared_rows=comp, language="korean")
    assert [r["offset_abs"] for r in records] == [10.0, 10.0]


# ---------------------------------------------------------------------------
# _compute_stats
# ---------------------------------------------------------------------------

def test_compute_stats_basic_values():
    stats = _compute_stats([0.0, 10.0, -10.0, 20.0])
    assert stats["count"] == 4
    assert pytest.approx(stats["mean"]) == 5.0
    assert pytest.approx(stats["mae"]) == 10.0


def test_compute_stats_pct_late_early_within():
    # values: -20, -3, 0, 4, 8, 25 → 2 late (>5), 1 early (<-5), 3 within ±5
    stats = _compute_stats([-20.0, -3.0, 0.0, 4.0, 8.0, 25.0])
    assert stats["pct_late_5ms"] == pytest.approx(2 / 6 * 100.0)
    assert stats["pct_early_5ms"] == pytest.approx(1 / 6 * 100.0)
    assert stats["pct_within_5ms"] == pytest.approx(3 / 6 * 100.0)
    assert stats["pct_within_10ms"] == pytest.approx(4 / 6 * 100.0)
    assert stats["pct_within_20ms"] == pytest.approx(5 / 6 * 100.0)


def test_compute_stats_empty():
    stats = _compute_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["mae"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_per_role
# ---------------------------------------------------------------------------

def test_aggregate_per_role_groups_and_includes_all():
    records = [
        {"wav": "x", "alias": "a", "role": "vc", "offset_abs": 5.0, "pre_abs": 0.0, "ovl_abs": 0.0, "cons_abs": 0.0, "cutoff_abs": 0.0, "ref_offset": 0, "comp_offset": 0},
        {"wav": "x", "alias": "b", "role": "vc", "offset_abs": 15.0, "pre_abs": 0.0, "ovl_abs": 0.0, "cons_abs": 0.0, "cutoff_abs": 0.0, "ref_offset": 0, "comp_offset": 0},
        {"wav": "x", "alias": "c", "role": "cv", "offset_abs": -10.0, "pre_abs": 0.0, "ovl_abs": 0.0, "cons_abs": 0.0, "cutoff_abs": 0.0, "ref_offset": 0, "comp_offset": 0},
    ]
    agg = aggregate_per_role(records)
    assert set(agg.keys()) == {"vc", "cv", "ALL"}
    assert agg["vc"]["offset_abs"]["count"] == 2
    assert agg["vc"]["offset_abs"]["mean"] == pytest.approx(10.0)
    assert agg["cv"]["offset_abs"]["mean"] == pytest.approx(-10.0)
    assert agg["ALL"]["offset_abs"]["count"] == 3


# ---------------------------------------------------------------------------
# worst_rows
# ---------------------------------------------------------------------------

def test_worst_rows_sorts_by_abs_value_desc():
    records = [
        {"offset_abs": -25.0, "alias": "big_early"},
        {"offset_abs": 5.0, "alias": "small"},
        {"offset_abs": 30.0, "alias": "big_late"},
    ]
    out = worst_rows(records, anchor_key="offset_abs", n=2)
    assert [r["alias"] for r in out] == ["big_late", "big_early"]


# ---------------------------------------------------------------------------
# diagnose / render_markdown end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_run(tmp_path):
    run_dir = tmp_path / "run"
    # reference (manual) and compared (autooto) OTOs for one voicebank
    _write_oto(
        str(run_dir / "manual" / "vb1" / "oto.ini"),
        [
            "a.wav=a k,100,30,-200,50,10",
            "a.wav=k a,200,30,-200,40,10",
        ],
    )
    _write_oto(
        str(run_dir / "autooto_F1" / "vb1" / "oto.ini"),
        [
            "a.wav=a k,120,30,-200,40,10",  # offset +20, pre absolute = 100+50=150 vs 120+40=160 → +10
            "a.wav=k a,210,30,-200,40,10",  # offset +10, pre absolute unchanged
        ],
    )
    return str(run_dir)


def test_diagnose_end_to_end(tiny_run):
    manifest = {
        "run_id": "tiny",
        "voicebanks": [{"id": "vb1", "language": "korean"}],
    }
    diagnosis = diagnose(
        manifest=manifest,
        run_dir=tiny_run,
        reference_version="manual",
        compared_versions=["autooto_F1"],
    )
    results = diagnosis["voicebank_results"]
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "ok"
    assert r["matched"] == 2
    # ALL group stats
    all_stats = r["stats_by_role"]["ALL"]["offset_abs"]
    assert pytest.approx(all_stats["mean"]) == 15.0  # average of 20 and 10
    # Markdown render does not crash
    md = render_markdown(diagnosis)
    assert "OTO Bias Diagnostic" in md
    assert "vb1" in md
    assert "autooto_F1" in md


def test_diagnose_reports_missing_reference(tmp_path):
    manifest = {"run_id": "x", "voicebanks": [{"id": "vbz", "language": "korean"}]}
    diagnosis = diagnose(
        manifest=manifest,
        run_dir=str(tmp_path / "empty"),
        reference_version="manual",
        compared_versions=["autooto_F1"],
    )
    assert diagnosis["voicebank_results"][0]["status"] == "reference_missing"


def test_diagnose_reports_missing_compared(tmp_path):
    run_dir = tmp_path / "run"
    _write_oto(str(run_dir / "manual" / "vb1" / "oto.ini"), ["a.wav=a k,100,30,-200,50,10"])
    # autooto_F1 is missing
    manifest = {"run_id": "x", "voicebanks": [{"id": "vb1", "language": "korean"}]}
    diagnosis = diagnose(
        manifest=manifest,
        run_dir=str(run_dir),
        reference_version="manual",
        compared_versions=["autooto_F1"],
    )
    assert diagnosis["voicebank_results"][0]["status"] == "compared_missing"


def test_render_markdown_includes_interpretation_when_bias_large(tiny_run):
    manifest = {"run_id": "tiny", "voicebanks": [{"id": "vb1", "language": "korean"}]}
    diagnosis = diagnose(
        manifest=manifest,
        run_dir=tiny_run,
        reference_version="manual",
        compared_versions=["autooto_F1"],
    )
    md = render_markdown(diagnosis)
    # offset bias here is ~+15 ms, which crosses the 5 ms threshold and should
    # trigger the auto-interpretation block.
    assert "자동 해석" in md
    assert "offset bias" in md
