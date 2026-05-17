"""Unit tests for ml.scripts.coarse_crnn.auto_oto.

Focus on the path-resolution / preset / backup logic. The actual generation
call is end-to-end and exercised elsewhere; we don't re-run the model here.
"""
from __future__ import annotations

import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.auto_oto import (  # noqa: E402
    BACKUP_SUFFIX_PREFIX,
    SOURCE_OTO_CANDIDATES,
    VARIANT_PRESETS,
    apply_env_preset,
    backup_existing_oto,
    resolve_language,
    resolve_model_path,
    resolve_output_path,
    resolve_source_oto,
    restore_env,
)


# ---------------------------------------------------------------------------
# resolve_source_oto
# ---------------------------------------------------------------------------

def test_resolve_source_oto_uses_explicit_path(tmp_path):
    src = tmp_path / "custom.ini"
    src.write_text("a.wav=x,0,0,0,0,0\n", encoding="utf-8")
    out = resolve_source_oto(str(tmp_path), str(src))
    assert out == str(src.resolve()) or out == os.path.abspath(str(src))


def test_resolve_source_oto_rejects_missing_explicit(tmp_path):
    with pytest.raises(SystemExit):
        resolve_source_oto(str(tmp_path), str(tmp_path / "nope.ini"))


def test_resolve_source_oto_finds_canonical_name(tmp_path):
    (tmp_path / "source_oto.ini").write_text("a.wav=x,0,0,0,0,0\n", encoding="utf-8")
    out = resolve_source_oto(str(tmp_path), "")
    assert out.endswith("source_oto.ini")


def test_resolve_source_oto_tries_alternative_names(tmp_path):
    # reclist.ini is among SOURCE_OTO_CANDIDATES — script should find it
    assert "reclist.ini" in SOURCE_OTO_CANDIDATES
    (tmp_path / "reclist.ini").write_text("a.wav=x,0,0,0,0,0\n", encoding="utf-8")
    out = resolve_source_oto(str(tmp_path), "")
    assert out.endswith("reclist.ini")


def test_resolve_source_oto_raises_when_nothing_found(tmp_path):
    with pytest.raises(SystemExit):
        resolve_source_oto(str(tmp_path), "")


# ---------------------------------------------------------------------------
# resolve_language
# ---------------------------------------------------------------------------

def test_resolve_language_explicit_overrides_path(tmp_path):
    # Path component "korean" would normally infer Korean — explicit overrides.
    ko_path = str(tmp_path / "korean" / "vb")
    os.makedirs(ko_path, exist_ok=True)
    assert resolve_language(ko_path, "japanese") == "japanese"


def test_resolve_language_auto_uses_path_inference(tmp_path):
    # infer_language_from_path matches when a path COMPONENT is "korean"/"ko"/"kor".
    ko_path = str(tmp_path / "korean" / "vb")
    os.makedirs(ko_path, exist_ok=True)
    assert resolve_language(ko_path, "auto") == "korean"


def test_resolve_language_auto_raises_when_path_ambiguous(tmp_path):
    # Generic path name → infer_language returns "" → raise
    plain = str(tmp_path / "voicebank_xx")
    os.makedirs(plain, exist_ok=True)
    with pytest.raises(SystemExit):
        resolve_language(plain, "auto")


def test_resolve_language_rejects_unknown_value(tmp_path):
    with pytest.raises(SystemExit):
        resolve_language(str(tmp_path), "klingon")


# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------

def test_resolve_model_path_uses_explicit(tmp_path):
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"")
    out = resolve_model_path(str(fake))
    assert out == str(fake.resolve()) or out == os.path.abspath(str(fake))


def test_resolve_model_path_rejects_missing_explicit(tmp_path):
    with pytest.raises(SystemExit):
        resolve_model_path(str(tmp_path / "missing.pt"))


# ---------------------------------------------------------------------------
# resolve_output_path
# ---------------------------------------------------------------------------

def test_resolve_output_path_defaults_to_voicebank_oto_ini(tmp_path):
    out = resolve_output_path(str(tmp_path), "")
    assert out.endswith(os.sep + "oto.ini")


def test_resolve_output_path_uses_explicit_override(tmp_path):
    target = tmp_path / "elsewhere" / "out.ini"
    out = resolve_output_path(str(tmp_path), str(target))
    assert out.endswith(os.path.join("elsewhere", "out.ini"))


# ---------------------------------------------------------------------------
# backup_existing_oto
# ---------------------------------------------------------------------------

def test_backup_returns_none_when_no_existing(tmp_path):
    out = backup_existing_oto(str(tmp_path / "oto.ini"), "20260515_120000")
    assert out is None


def test_backup_copies_existing_with_timestamp_prefix(tmp_path):
    target = tmp_path / "oto.ini"
    target.write_text("a.wav=x,0,0,0,0,0\n", encoding="utf-8")
    backup = backup_existing_oto(str(target), "20260515_120000")
    assert backup is not None
    assert os.path.isfile(backup)
    assert BACKUP_SUFFIX_PREFIX in os.path.basename(backup)
    assert "20260515_120000" in os.path.basename(backup)


def test_backup_preserves_original_content(tmp_path):
    target = tmp_path / "oto.ini"
    content = "a.wav=x,1,2,-3,4,5\n"
    target.write_text(content, encoding="utf-8")
    backup = backup_existing_oto(str(target), "20260515_120000")
    with open(backup, "r", encoding="utf-8") as handle:
        assert handle.read() == content


# ---------------------------------------------------------------------------
# apply_env_preset / restore_env
# ---------------------------------------------------------------------------

def test_apply_env_preset_sets_and_restore_clears(monkeypatch):
    monkeypatch.delenv("UTOA_AUTO_OTO_TEST_VAR", raising=False)
    restore = apply_env_preset({"UTOA_AUTO_OTO_TEST_VAR": "hello"})
    assert os.environ.get("UTOA_AUTO_OTO_TEST_VAR") == "hello"
    restore_env(restore)
    assert "UTOA_AUTO_OTO_TEST_VAR" not in os.environ


def test_apply_env_preset_skips_existing_env(monkeypatch):
    """Pre-existing env vars take precedence over preset values — this lets
    ``UTOA_BOUNDARY_SHIFT_VC_MS=200 python auto_oto.py --variant F3 ...``
    actually override the F3 preset instead of being silently clobbered."""
    monkeypatch.setenv("UTOA_AUTO_OTO_TEST_VAR", "before")
    restore = apply_env_preset({"UTOA_AUTO_OTO_TEST_VAR": "during"})
    # Pre-existing value preserved
    assert os.environ["UTOA_AUTO_OTO_TEST_VAR"] == "before"
    # Restore is a no-op for keys we didn't touch
    restore_env(restore)
    assert os.environ["UTOA_AUTO_OTO_TEST_VAR"] == "before"


def test_apply_env_preset_empty_is_noop():
    restore = apply_env_preset({})
    assert restore == {}
    restore_env(restore)  # should not raise


# ---------------------------------------------------------------------------
# Variant presets sanity
# ---------------------------------------------------------------------------

def test_variant_presets_have_expected_keys():
    assert set(VARIANT_PRESETS.keys()) == {"baseline", "F1", "F2", "F3"}


def test_variant_F3_enables_filename_anchor_timeline():
    env = VARIANT_PRESETS["F3"]
    # F3 = F2 + filename timeline
    assert env.get("UTOA_BOUNDARY_ANCHOR_TIMELINE") == "filename"
    assert env.get("UTOA_BOUNDARY_CONS_MIN_MS") == "40"  # inherited from F2
    assert env.get("UTOA_BOUNDARY_LWC_ENABLE") == "on"  # inherited from F1
    # VC-focused constraints to avoid VC rows collapsing toward CV onset.
    assert env.get("UTOA_BOUNDARY_VC_TARGET_RATIO") == "0.58"
    assert env.get("UTOA_BOUNDARY_VC_CLAMP_RIGHT_RATIO") == "0.80"
    assert env.get("UTOA_BOUNDARY_TRANSITION_PRE_CONS_MIN_GAP_MS") == "24"
    # Row-shift rollback disabled by design — source-OTO values must not
    # influence inference output (2026-05-17 contract).
    assert env.get("UTOA_BOUNDARY_ROW_SHIFT_ROLLBACK_ENABLE") == "off"


def test_variant_baseline_has_no_env_overrides():
    assert VARIANT_PRESETS["baseline"] == {}


def test_variant_F1_enables_lwc_and_residual():
    env = VARIANT_PRESETS["F1"]
    assert env.get("UTOA_BOUNDARY_LWC_ENABLE") == "on"
    assert env.get("UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE") == "on"
    assert env.get("UTOA_BOUNDARY_RESIDUAL_TARGETS") == "delta_overlap,delta_cutoff"


def test_variant_F2_extends_cons_and_tail():
    env = VARIANT_PRESETS["F2"]
    # F2 inherits F1's keys
    assert env.get("UTOA_BOUNDARY_LWC_ENABLE") == "on"
    assert env.get("UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE") == "on"
    # Plus the cap extensions
    assert env.get("UTOA_BOUNDARY_CONS_MIN_MS") == "40"
    assert env.get("UTOA_BOUNDARY_CONS_MAX_MS") == "140"
    assert env.get("UTOA_BOUNDARY_TAIL_MIN_MS") == "60"
    assert env.get("UTOA_BOUNDARY_TAIL_MAX_MS") == "220"
    assert env.get("UTOA_BOUNDARY_MAX_SPAN_MS") == "600"
