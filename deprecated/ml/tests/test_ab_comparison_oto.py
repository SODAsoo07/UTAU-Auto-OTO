"""Unit tests for ml.scripts.coarse_crnn.build_ab_comparison_oto (Phase E)."""
from __future__ import annotations

import json
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.build_ab_comparison_oto import (  # noqa: E402
    _build_lookup,
    _discover_versions,
    _format_oto_line,
    _lookup_occurrence,
    _read_oto_rows,
    _resolve_run_dir,
    build_combined_oto,
    build_comparison_for_voicebank,
    build_per_version_oto,
)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_oto(path: str, lines: list[str]) -> None:
    _write(path, "\n".join(lines) + "\n")


# Shared fixtures: a source OTO with 3 aliases, three versions each with
# distinguishable parameter values.

SOURCE_LINES = [
    "a.wav=a k,100.000,30.000,-200.000,50.000,10.000",
    "a.wav=a m,120.000,35.000,-210.000,55.000,12.000",
    "b.wav=k a,80.000,40.000,-180.000,30.000,8.000",
]


def _make_version_lines(*, offset_jitter: float) -> list[str]:
    return [
        f"a.wav=a k,{100.0 + offset_jitter:.3f},30.000,-200.000,50.000,10.000",
        f"a.wav=a m,{120.0 + offset_jitter:.3f},35.000,-210.000,55.000,12.000",
        f"b.wav=k a,{80.0 + offset_jitter:.3f},40.000,-180.000,30.000,8.000",
    ]


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run"
    for version, jitter in (("manual", 0.0), ("autooto_baseline", 1.0), ("autooto_F1", -1.0)):
        _write_oto(str(root / version / "vb1" / "oto.ini"), _make_version_lines(offset_jitter=jitter))
    return str(root)


@pytest.fixture
def source_oto_path(tmp_path):
    path = tmp_path / "source.ini"
    _write(str(path), "\n".join(SOURCE_LINES) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Path & lookup helpers
# ---------------------------------------------------------------------------

def test_discover_versions_sorts_alphabetically(run_dir):
    versions = _discover_versions(run_dir, "vb1")
    assert versions == ["autooto_F1", "autooto_baseline", "manual"]


def test_resolve_run_dir_default():
    manifest = {"run_id": "RUNZ", "voicebanks": []}
    out = _resolve_run_dir(manifest=manifest, explicit_run_dir="")
    assert out.endswith(os.path.join("ml", "eval", "ab_listening", "runs", "RUNZ"))


def test_format_oto_line_uses_three_decimal_places():
    line = _format_oto_line(
        wav="x.wav",
        alias="a k",
        params={"offset": 100.5, "cons": 30.25, "cutoff": -200.125, "pre": 50.0, "ovl": 10.75},
    )
    assert line == "x.wav=a k,100.500,30.250,-200.125,50.000,10.750"


def test_build_lookup_and_occurrence(tmp_path):
    rows = _read_oto_rows(str(tmp_path / "missing.ini"))
    assert rows == []
    path = str(tmp_path / "oto.ini")
    _write_oto(path, [
        "a.wav=a k,100,30,-200,50,10",
        "a.wav=a m,120,35,-210,55,12",
        "a.wav=a k,150,33,-205,51,11",
    ])
    rows = _read_oto_rows(path)
    assert len(rows) == 3
    lookup = _build_lookup(rows)
    occ0 = _lookup_occurrence(lookup, wav="a.wav", alias="a k", occurrence=0)
    occ1 = _lookup_occurrence(lookup, wav="a.wav", alias="a k", occurrence=1)
    assert occ0 and float(occ0["offset"]) == 100.0
    assert occ1 and float(occ1["offset"]) == 150.0
    assert _lookup_occurrence(lookup, wav="a.wav", alias="a k", occurrence=2) is None


# ---------------------------------------------------------------------------
# build_per_version_oto
# ---------------------------------------------------------------------------

def test_per_version_oto_contains_every_source_alias(source_oto_path, run_dir, tmp_path):
    source_rows = _read_oto_rows(source_oto_path)
    version_rows = _read_oto_rows(os.path.join(run_dir, "autooto_F1", "vb1", "oto.ini"))
    out_path = str(tmp_path / "out_F1.ini")
    info = build_per_version_oto(source_rows=source_rows, version_rows=version_rows, out_path=out_path)
    assert info["row_count"] == len(source_rows)
    assert info["missing_in_version"] == 0
    out_lines = [line for line in _read(out_path).splitlines() if line.strip()]
    assert len(out_lines) == len(source_rows)
    # First entry should have F1's offset (which is jitter -1.0 from source 100 → 99.000)
    assert out_lines[0].startswith("a.wav=a k,99.000,")


def test_per_version_oto_preserves_source_alias_text(tmp_path):
    """Even if the version OTO renamed the alias, the output should keep source text."""
    source_path = str(tmp_path / "source.ini")
    _write_oto(source_path, ["a.wav=a k,100,30,-200,50,10"])
    version_path = str(tmp_path / "version.ini")
    _write_oto(version_path, ["a.wav=a_k_renamed,150,30,-200,50,10"])
    out_path = str(tmp_path / "out.ini")
    info = build_per_version_oto(
        source_rows=_read_oto_rows(source_path),
        version_rows=_read_oto_rows(version_path),
        out_path=out_path,
    )
    # Version's alias does not match source, so it's a miss → source params fallback
    assert info["missing_in_version"] == 1
    line = _read(out_path).splitlines()[0]
    # Source alias text retained
    assert "=a k," in line
    assert "a_k_renamed" not in line


def test_per_version_oto_falls_back_to_source_params_when_version_missing(tmp_path):
    source_path = str(tmp_path / "source.ini")
    _write_oto(source_path, [
        "a.wav=a k,100,30,-200,50,10",
        "b.wav=k a,80,40,-180,30,8",
    ])
    version_path = str(tmp_path / "version.ini")
    # Only first alias present in version
    _write_oto(version_path, ["a.wav=a k,150,30,-200,50,10"])
    out_path = str(tmp_path / "out.ini")
    info = build_per_version_oto(
        source_rows=_read_oto_rows(source_path),
        version_rows=_read_oto_rows(version_path),
        out_path=out_path,
    )
    assert info["missing_in_version"] == 1
    lines = _read(out_path).splitlines()
    assert len(lines) == 2
    # First line: version params (offset 150)
    assert lines[0].startswith("a.wav=a k,150.000,")
    # Second line: fallback to source params (offset 80)
    assert lines[1].startswith("b.wav=k a,80.000,")


# ---------------------------------------------------------------------------
# build_combined_oto
# ---------------------------------------------------------------------------

def test_combined_oto_emits_unsuffixed_block(source_oto_path, run_dir, tmp_path):
    source_rows = _read_oto_rows(source_oto_path)
    version_rows = {
        v: _read_oto_rows(os.path.join(run_dir, v, "vb1", "oto.ini"))
        for v in ("manual", "autooto_baseline", "autooto_F1")
    }
    out_path = str(tmp_path / "combined.ini")
    info = build_combined_oto(
        source_rows=source_rows,
        version_rows_by_id=version_rows,
        out_path=out_path,
        include_unsuffixed=True,
    )
    text = _read(out_path)
    # Source block exists
    assert "# --- source (unsuffixed) ---" in text
    # Unsuffixed entries with source params (offset 100 for a k)
    assert "a.wav=a k,100.000," in text
    # Version blocks exist
    for v in ("manual", "autooto_baseline", "autooto_F1"):
        assert f"# --- {v} ---" in text
        assert f"a k__{v}," in text
    # Expected row count: 4 header comments + 3 unsuffixed + 3 versions × 3 aliases = 4 + 3 + 9 = 16
    assert info["row_count"] == 16


def test_combined_oto_no_unsuffixed_flag_skips_source_block(source_oto_path, run_dir, tmp_path):
    source_rows = _read_oto_rows(source_oto_path)
    version_rows = {
        v: _read_oto_rows(os.path.join(run_dir, v, "vb1", "oto.ini"))
        for v in ("manual",)
    }
    out_path = str(tmp_path / "combined.ini")
    info = build_combined_oto(
        source_rows=source_rows,
        version_rows_by_id=version_rows,
        out_path=out_path,
        include_unsuffixed=False,
    )
    text = _read(out_path)
    assert "# --- source (unsuffixed) ---" not in text
    assert "# --- manual ---" in text
    # 1 header + 3 suffixed rows
    assert info["row_count"] == 4


def test_combined_suffix_uses_source_alias_text(tmp_path):
    """Even when version OTO renamed an alias, combined uses source text + version suffix."""
    source_path = str(tmp_path / "source.ini")
    _write_oto(source_path, ["a.wav=a k,100,30,-200,50,10"])
    version_path = str(tmp_path / "version.ini")
    _write_oto(version_path, ["a.wav=a_k_renamed,150,30,-200,50,10"])
    out_path = str(tmp_path / "combined.ini")
    build_combined_oto(
        source_rows=_read_oto_rows(source_path),
        version_rows_by_id={"v1": _read_oto_rows(version_path)},
        out_path=out_path,
        include_unsuffixed=True,
    )
    text = _read(out_path)
    # Suffixed alias uses source text 'a k', not 'a_k_renamed'
    assert "a k__v1," in text
    assert "a_k_renamed" not in text


def test_combined_alias_separator_override(source_oto_path, run_dir, tmp_path):
    source_rows = _read_oto_rows(source_oto_path)
    version_rows = {"manual": _read_oto_rows(os.path.join(run_dir, "manual", "vb1", "oto.ini"))}
    out_path = str(tmp_path / "combined.ini")
    build_combined_oto(
        source_rows=source_rows,
        version_rows_by_id=version_rows,
        out_path=out_path,
        alias_separator="::",
    )
    text = _read(out_path)
    assert "a k::manual," in text
    assert "a k__manual" not in text


def test_combined_uses_version_params_per_block(source_oto_path, run_dir, tmp_path):
    """In the F1 block, parameters must be F1's (jitter -1.0), not source's."""
    source_rows = _read_oto_rows(source_oto_path)
    version_rows = {
        "autooto_F1": _read_oto_rows(os.path.join(run_dir, "autooto_F1", "vb1", "oto.ini")),
    }
    out_path = str(tmp_path / "combined.ini")
    build_combined_oto(
        source_rows=source_rows,
        version_rows_by_id=version_rows,
        out_path=out_path,
        include_unsuffixed=False,
    )
    text = _read(out_path)
    # F1 should produce offset 99.000 for a k (100 + -1.0)
    assert "a k__autooto_F1,99.000," in text


# ---------------------------------------------------------------------------
# build_comparison_for_voicebank end-to-end
# ---------------------------------------------------------------------------

def test_build_comparison_writes_all_outputs(source_oto_path, run_dir, tmp_path):
    out = str(tmp_path / "out")
    result = build_comparison_for_voicebank(
        vb_id="vb1",
        source_oto_path=source_oto_path,
        run_dir=run_dir,
        out_dir=out,
    )
    assert result["status"] == "ok"
    assert sorted(result["versions"]) == ["autooto_F1", "autooto_baseline", "manual"]
    # Per-version files exist with all 3 source aliases each
    for v in result["versions"]:
        path = os.path.join(out, "vb1", f"{v}.oto.ini")
        assert os.path.isfile(path)
        lines = [line for line in _read(path).splitlines() if line.strip()]
        assert len(lines) == 3
    # Combined file exists with the expected row layout
    combined = os.path.join(out, "vb1", "combined.oto.ini")
    assert os.path.isfile(combined)
    text = _read(combined)
    # All four blocks present
    for header in ("source (unsuffixed)", "manual", "autooto_baseline", "autooto_F1"):
        assert header in text


def test_build_comparison_with_no_versions_returns_status(tmp_path):
    source = str(tmp_path / "source.ini")
    _write_oto(source, ["a.wav=a k,100,30,-200,50,10"])
    empty_run = str(tmp_path / "empty_run")
    os.makedirs(empty_run, exist_ok=True)
    result = build_comparison_for_voicebank(
        vb_id="vb1", source_oto_path=source, run_dir=empty_run, out_dir=str(tmp_path / "out")
    )
    assert result["status"] == "no_versions_found"


def test_build_comparison_with_empty_source_returns_status(tmp_path):
    result = build_comparison_for_voicebank(
        vb_id="vb1",
        source_oto_path=str(tmp_path / "does_not_exist.ini"),
        run_dir=str(tmp_path / "run"),
        out_dir=str(tmp_path / "out"),
    )
    assert result["status"] == "source_oto_empty_or_missing"


def test_build_comparison_respects_version_filter(source_oto_path, run_dir, tmp_path):
    out = str(tmp_path / "out")
    result = build_comparison_for_voicebank(
        vb_id="vb1",
        source_oto_path=source_oto_path,
        run_dir=run_dir,
        out_dir=out,
        versions=["manual", "autooto_F1"],
    )
    assert sorted(result["versions"]) == ["autooto_F1", "manual"]
    # Combined should NOT mention autooto_baseline
    combined_text = _read(os.path.join(out, "vb1", "combined.oto.ini"))
    assert "autooto_baseline" not in combined_text
    # Per-version files for excluded version should not exist
    assert not os.path.isfile(os.path.join(out, "vb1", "autooto_baseline.oto.ini"))
