"""Unit tests for ml.scripts.coarse_crnn.build_listening_html (Phase E)."""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
import yaml


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.build_listening_html import (  # noqa: E402
    DEFECT_KEYS,
    _fmt_timestamp,
    _relative_path,
    _resolve_render_path,
    build_case_records,
    render_html,
)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_csv(path: str, rows: list[dict]) -> None:
    headers = ["case_id", "voicebank_id", "language", "wav_name", "alias", "version_id", *DEFECT_KEYS, "notes"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _write_ustx(path: str, *, bpm: int = 120, resolution: int = 480, notes: list[dict]) -> None:
    data = {
        "bpm": int(bpm),
        "resolution": int(resolution),
        "voice_parts": [
            {"position": 0, "notes": notes}
        ],
    }
    _write_text(path, yaml.safe_dump(data, allow_unicode=True))


@pytest.fixture
def env(tmp_path):
    """Manifest + sample CSV + USTx + run dir scaffold."""
    run_id = "test_run"
    run_dir = tmp_path / "runs" / run_id
    out_html = tmp_path / "reports" / f"{run_id}.labeling.html"

    ustx_path = tmp_path / "vb1.ustx"
    _write_ustx(
        str(ustx_path),
        notes=[
            {"position": 0, "duration": 480, "lyric": "a k"},      # 0–500ms
            {"position": 480, "duration": 480, "lyric": "k a"},    # 500–1000ms
            {"position": 960, "duration": 480, "lyric": "a k"},    # 1000–1500ms (2nd occurrence)
        ],
    )

    manifest = {
        "run_id": run_id,
        "voicebanks": [
            {"id": "vb1", "language": "korean", "ustx": str(ustx_path), "wav_dir": str(tmp_path / "vb1")},
        ],
        "versions": [
            {"id": "manual", "kind": "external"},
            {"id": "autooto_F1", "kind": "autooto"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_text(str(manifest_path), json.dumps(manifest))

    sample_path = tmp_path / "sample.csv"
    _write_csv(str(sample_path), [
        {"case_id": "vb1__0001__a_k", "voicebank_id": "vb1", "language": "korean", "wav_name": "x.wav", "alias": "a k", "version_id": "manual"},
        {"case_id": "vb1__0001__a_k", "voicebank_id": "vb1", "language": "korean", "wav_name": "x.wav", "alias": "a k", "version_id": "autooto_F1"},
        {"case_id": "vb1__0002__k_a", "voicebank_id": "vb1", "language": "korean", "wav_name": "x.wav", "alias": "k a", "version_id": "manual"},
        {"case_id": "vb1__0002__k_a", "voicebank_id": "vb1", "language": "korean", "wav_name": "x.wav", "alias": "k a", "version_id": "autooto_F1"},
    ])
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "sample_path": str(sample_path),
        "run_dir": str(run_dir),
        "out_html": str(out_html),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_fmt_timestamp_minutes_seconds():
    assert _fmt_timestamp(0) == "0:00.00"
    assert _fmt_timestamp(1500) == "0:01.50"
    assert _fmt_timestamp(65500) == "1:05.50"


def test_resolve_render_path_uses_convention():
    path = _resolve_render_path(run_dir="/r/run1", version_id="manual", vb_id="vb1")
    # Normalised path components
    assert path.endswith(os.path.join("manual", "vb1", "render.wav"))


def test_relative_path_uses_forward_slashes(tmp_path):
    out_html = str(tmp_path / "reports" / "run.html")
    wav = str(tmp_path / "runs" / "run1" / "manual" / "vb1" / "render.wav")
    rel = _relative_path(from_file=out_html, to_file=wav)
    assert "\\" not in rel
    assert rel.endswith("manual/vb1/render.wav")


# ---------------------------------------------------------------------------
# build_case_records
# ---------------------------------------------------------------------------

def test_build_case_records_emits_one_record_per_unique_case(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    payload = result["payload"]
    case_ids = [c["case_id"] for c in payload["cases"]]
    # Sample CSV has 4 rows but only 2 unique case_ids
    assert case_ids == ["vb1__0001__a_k", "vb1__0002__k_a"]


def test_build_case_records_matches_alias_to_ustx_notes(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    cases_by_id = {c["case_id"]: c for c in result["payload"]["cases"]}
    case_ak = cases_by_id["vb1__0001__a_k"]
    # USTx has "a k" twice → match_count == 2
    assert case_ak["match_count"] == 2
    assert len(case_ak["matches"]) == 2
    # First match should be near 0 ms (minus pre-pad)
    assert case_ak["matches"][0]["raw_start_ms"] == 0.0
    assert case_ak["matches"][0]["raw_end_ms"] == pytest.approx(500.0)


def test_build_case_records_applies_pre_post_padding(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
        case_offset_pad_ms=80.0,
        case_post_pad_ms=120.0,
    )
    case = next(c for c in result["payload"]["cases"] if c["case_id"] == "vb1__0001__a_k")
    m = case["matches"][0]
    # raw_start 0 → padded start clamps to 0 (can't go negative)
    assert m["start_ms"] == 0.0
    # raw_end 500 → padded end 500 + 120 = 620
    assert m["end_ms"] == pytest.approx(620.0)


def test_build_case_records_emits_audio_sources_for_every_version_vb(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    sources = result["payload"]["audio_sources"]
    # 1 voicebank × 2 versions = 2 sources
    assert set(sources.keys()) == {"vb1|manual", "vb1|autooto_F1"}
    for src in sources.values():
        assert src.endswith("render.wav")
        assert "\\" not in src


def test_build_case_records_flags_missing_audio(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    status = result["payload"]["audio_status"]
    # No render wavs were created → both should be missing
    assert all(not s["exists"] for s in status.values())


def test_build_case_records_warns_on_missing_ustx(env):
    manifest = dict(env["manifest"])
    manifest["voicebanks"] = [
        {"id": "vb1", "language": "korean", "ustx": "/does/not/exist.ustx", "wav_dir": "/x"},
    ]
    result = build_case_records(
        manifest=manifest,
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    assert any("ustx path missing" in w for w in result["summary"]["ustx_warnings"])
    # Cases still emit even without USTx; match_count is 0
    assert all(c["match_count"] == 0 for c in result["payload"]["cases"])


def test_build_case_records_caps_match_list(env):
    # USTx with 7 occurrences of the same alias
    ustx_path = os.path.join(os.path.dirname(env["manifest_path"]), "many.ustx")
    notes = [{"position": 480 * i, "duration": 480, "lyric": "a k"} for i in range(7)]
    _write_ustx(ustx_path, notes=notes)
    manifest = dict(env["manifest"])
    manifest["voicebanks"] = [{"id": "vb1", "language": "korean", "ustx": ustx_path, "wav_dir": "/x"}]
    result = build_case_records(
        manifest=manifest,
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
        max_matches_per_case=3,
    )
    case = next(c for c in result["payload"]["cases"] if c["case_id"] == "vb1__0001__a_k")
    assert case["match_count"] == 7
    assert len(case["matches"]) == 3  # capped


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

def test_render_html_embeds_payload_json(env):
    result = build_case_records(
        manifest=env["manifest"],
        sample_rows=_load_sample_rows(env["sample_path"]),
        run_dir=env["run_dir"],
        out_html_path=env["out_html"],
    )
    html_text = render_html(result["payload"])
    # JSON payload must be parseable from the embedded <script> tag
    start = html_text.index('<script id="payload"')
    end = html_text.index("</script>", start)
    json_block = html_text[start:end]
    # Strip the script start tag
    start_tag_end = json_block.index(">") + 1
    json_block = json_block[start_tag_end:]
    # Restore the </script> escape we applied in render_html
    json_block = json_block.replace("<\\/", "</")
    data = json.loads(json_block)
    assert data["run_id"] == "test_run"
    assert "audio_sources" in data
    assert isinstance(data["cases"], list)


def test_render_html_escapes_script_close_tag_in_payload():
    """If a value contained '</script>' verbatim it would break the page."""
    payload = {
        "run_id": "evil",
        "versions": ["v1"],
        "voicebanks": [],
        "audio_sources": {},
        "audio_status": {},
        "cases": [],
        "defect_keys": list(DEFECT_KEYS),
        "defect_labels_ko": {},
        "defect_shortcuts": {},
        "ustx_warnings": ["this contains </script> in text"],
    }
    out = render_html(payload)
    # The payload's script block must not contain a raw closing tag inside it.
    payload_start = out.index('<script id="payload"')
    payload_end = out.index("</script>", payload_start)
    payload_block = out[payload_start:payload_end]
    assert "</script>" not in payload_block, "raw </script> would terminate the payload prematurely"
    # The escaped form is what we expect to see instead.
    assert "<\\/script>" in payload_block


def test_render_html_contains_required_chrome():
    payload = {
        "run_id": "test",
        "versions": ["v1"],
        "voicebanks": [{"id": "vb1", "language": "korean", "ustx_total_ms": 1000.0}],
        "audio_sources": {"vb1|v1": "x.wav"},
        "audio_status": {"vb1|v1": {"exists": True, "abs_path": "/x.wav"}},
        "cases": [],
        "defect_keys": list(DEFECT_KEYS),
        "defect_labels_ko": {k: k for k in DEFECT_KEYS},
        "defect_shortcuts": {k: str(i+1) for i, k in enumerate(DEFECT_KEYS)},
        "ustx_warnings": [],
    }
    html_text = render_html(payload)
    # Static UI affordances must appear
    assert 'id="save-csv"' in html_text
    assert 'id="reset"' in html_text
    assert 'id="filter-vb"' in html_text
    assert 'STORAGE_KEY' in html_text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_sample_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
