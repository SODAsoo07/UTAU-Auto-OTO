"""Unit tests for ml.scripts.coarse_crnn.install_voicebank_oto (Phase E)."""
from __future__ import annotations

import json
import os
import sys
import time

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.install_voicebank_oto import (  # noqa: E402
    BACKUP_DIR_NAME,
    BACKUP_PREFIX,
    _find_latest_backup,
    _read_manifest,
    _resolve_comparison_root,
    _resolve_source_oto,
    install_for_voicebank,
    restore_for_voicebank,
)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Manifest / path helpers
# ---------------------------------------------------------------------------

def test_read_manifest_round_trip(tmp_path):
    path = tmp_path / "m.json"
    payload = {"run_id": "r1", "voicebanks": [{"id": "vb1", "wav_dir": "C:/x"}], "versions": []}
    _write(str(path), json.dumps(payload))
    out = _read_manifest(str(path))
    assert out["run_id"] == "r1"
    assert out["voicebanks"][0]["id"] == "vb1"


def test_read_manifest_rejects_missing_keys(tmp_path):
    path = tmp_path / "m.json"
    _write(str(path), json.dumps({"voicebanks": []}))
    with pytest.raises(SystemExit):
        _read_manifest(str(path))


def test_resolve_comparison_root_default():
    manifest = {"run_id": "RUNX", "voicebanks": []}
    root = _resolve_comparison_root(manifest=manifest, explicit_root="")
    assert root.endswith(os.path.join("ml", "eval", "ab_listening", "runs", "RUNX", "comparison"))


def test_resolve_comparison_root_explicit(tmp_path):
    manifest = {"run_id": "RUNX", "voicebanks": []}
    root = _resolve_comparison_root(manifest=manifest, explicit_root=str(tmp_path / "custom"))
    assert root == str((tmp_path / "custom").resolve()) or root == os.path.abspath(str(tmp_path / "custom"))


def test_resolve_source_oto_path_shape(tmp_path):
    root = str(tmp_path / "cmp")
    p = _resolve_source_oto(comparison_root=root, vb_id="vb1", variant="combined")
    assert p.endswith(os.path.join("cmp", "vb1", "combined.oto.ini"))


# ---------------------------------------------------------------------------
# install_for_voicebank
# ---------------------------------------------------------------------------

@pytest.fixture
def voicebank(tmp_path):
    """A wav_dir with an existing oto.ini that we will overwrite."""
    wav_dir = tmp_path / "vb1"
    wav_dir.mkdir()
    _write(str(wav_dir / "oto.ini"), "wav=alias_original,0,0,0,0,0\n")
    return str(wav_dir)


@pytest.fixture
def comparison_root(tmp_path):
    root = tmp_path / "cmp"
    _write(str(root / "vb1" / "combined.oto.ini"), "wav=alias_new__manual,1,1,1,1,1\n")
    _write(str(root / "vb1" / "manual.oto.ini"), "wav=alias_new,2,2,2,2,2\n")
    return str(root)


def test_install_writes_target_and_backs_up_existing(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=voicebank,
        source_oto=source,
        timestamp="20260515_120000",
    )
    assert result["status"] == "installed"
    # Target now contains the comparison content
    assert "alias_new__manual" in _read(os.path.join(voicebank, "oto.ini"))
    # Backup exists with original content
    backup = result["backup_path"]
    assert backup and os.path.isfile(backup)
    assert "alias_original" in _read(backup)
    assert BACKUP_DIR_NAME in backup
    assert BACKUP_PREFIX in os.path.basename(backup)


def test_install_no_existing_oto_skips_backup(tmp_path, comparison_root):
    wav_dir = tmp_path / "vb_fresh"
    wav_dir.mkdir()  # no existing oto.ini
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=str(wav_dir),
        source_oto=source,
        timestamp="20260515_120000",
    )
    assert result["status"] == "installed"
    assert result["backup_path"] is None
    assert os.path.isfile(os.path.join(str(wav_dir), "oto.ini"))


def test_install_dry_run_does_not_touch_files(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=voicebank,
        source_oto=source,
        timestamp="20260515_120000",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    # Target still has the original content
    assert "alias_original" in _read(os.path.join(voicebank, "oto.ini"))
    # No backup directory created
    assert not os.path.isdir(os.path.join(voicebank, BACKUP_DIR_NAME))


def test_install_no_backup_flag_skips_backup_but_overwrites(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=voicebank,
        source_oto=source,
        timestamp="20260515_120000",
        skip_backup=True,
    )
    assert result["status"] == "installed"
    assert result["backup_path"] is None
    # Target updated
    assert "alias_new__manual" in _read(os.path.join(voicebank, "oto.ini"))
    # Backup dir was not created
    assert not os.path.isdir(os.path.join(voicebank, BACKUP_DIR_NAME))


def test_install_missing_source_oto_returns_error(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="nonexistent")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=voicebank,
        source_oto=source,
        timestamp="20260515_120000",
    )
    assert result["status"] == "source_oto_missing"
    # Target unchanged
    assert "alias_original" in _read(os.path.join(voicebank, "oto.ini"))


def test_install_missing_wav_dir_returns_error(tmp_path, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    result = install_for_voicebank(
        vb_id="vb1",
        wav_dir=str(tmp_path / "does_not_exist"),
        source_oto=source,
        timestamp="20260515_120000",
    )
    assert result["status"] == "wav_dir_missing"


def test_install_variant_combined_vs_manual_picks_right_file(voicebank, comparison_root):
    # combined variant
    s1 = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    install_for_voicebank(vb_id="vb1", wav_dir=voicebank, source_oto=s1, timestamp="t1")
    assert "alias_new__manual" in _read(os.path.join(voicebank, "oto.ini"))
    # manual variant
    s2 = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="manual")
    install_for_voicebank(vb_id="vb1", wav_dir=voicebank, source_oto=s2, timestamp="t2")
    text = _read(os.path.join(voicebank, "oto.ini"))
    assert "alias_new," in text
    assert "alias_new__manual" not in text


# ---------------------------------------------------------------------------
# restore_for_voicebank
# ---------------------------------------------------------------------------

def test_restore_returns_no_backup_when_dir_absent(tmp_path):
    wav_dir = tmp_path / "vb"
    wav_dir.mkdir()
    result = restore_for_voicebank(vb_id="vb1", wav_dir=str(wav_dir))
    assert result["status"] == "no_backup_found"


def test_restore_uses_latest_backup(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    # First install creates one backup
    install_for_voicebank(vb_id="vb1", wav_dir=voicebank, source_oto=source, timestamp="20260515_120000")
    # Ensure filesystem timestamp ordering for the next backup
    time.sleep(0.01)
    # Second install creates a second backup (with the prior install's content)
    install_for_voicebank(vb_id="vb1", wav_dir=voicebank, source_oto=source, timestamp="20260515_130000")
    latest = _find_latest_backup(wav_dir=voicebank)
    assert latest and "130000" in os.path.basename(latest)
    result = restore_for_voicebank(vb_id="vb1", wav_dir=voicebank)
    assert result["status"] == "restored"
    # After restore, target equals what was in the latest backup (which is the combined content from earlier install)
    assert _read(os.path.join(voicebank, "oto.ini")) == _read(latest)


def test_restore_dry_run_does_not_touch_target(voicebank, comparison_root):
    source = _resolve_source_oto(comparison_root=comparison_root, vb_id="vb1", variant="combined")
    install_for_voicebank(vb_id="vb1", wav_dir=voicebank, source_oto=source, timestamp="20260515_120000")
    # After install, target has combined content
    assert "alias_new__manual" in _read(os.path.join(voicebank, "oto.ini"))
    result = restore_for_voicebank(vb_id="vb1", wav_dir=voicebank, dry_run=True)
    assert result["status"] == "dry_run"
    # Target unchanged by dry-run
    assert "alias_new__manual" in _read(os.path.join(voicebank, "oto.ini"))
