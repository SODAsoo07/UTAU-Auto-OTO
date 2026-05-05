"""
Unit tests for core.runtime_config — MLRuntimeConfig dataclass and singleton.

Tests cover:
- Default field values (no env contamination)
- from_env() reading of all UTOA_* variables
- Flag parsing edge cases (1/0/true/false/on/off/yes/no)
- Bounded-float clamping behaviour
- Helper methods: model_dir_for_language, blank_floor_for_format,
  cv_ovl_cap_ratio_for_language
- Singleton lifecycle: get_ml_config / set_ml_config / reset_ml_config
"""

from __future__ import annotations

import os
import pytest

from core.runtime_config import (
    MLRuntimeConfig,
    get_ml_config,
    reset_ml_config,
    set_ml_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_env(monkeypatch, *names):
    """Remove all UTOA_* env vars so tests start from a clean slate."""
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _all_utoa_vars():
    return [k for k in os.environ if k.startswith("UTOA_")]


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """Reset the singleton before and after every test."""
    reset_ml_config()
    yield
    reset_ml_config()


@pytest.fixture()
def clean(monkeypatch):
    """Remove every UTOA_* variable from the environment."""
    for k in list(os.environ):
        if k.startswith("UTOA_"):
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_bool_defaults(self):
        cfg = MLRuntimeConfig()
        assert cfg.ml_disabled is False
        assert cfg.ensemble_enabled is True
        assert cfg.gated_ensemble_enabled is False
        assert cfg.legacy_fallback_enabled is False
        assert cfg.two_stage_model_enabled is True
        assert cfg.auto_select_vbest is False
        assert cfg.same_language_borrow_only is True
        assert cfg.batch_inference_enabled is True
        assert cfg.bundle_cache_enabled is True
        assert cfg.selector_enabled is False
        assert cfg.selector_runtime_guard_disabled is False
        assert cfg.delta_disabled is False
        assert cfg.delta_forced is False
        assert cfg.selector_disabled is False
        assert cfg.selector_forced is False
        assert cfg.coupled_strict_constraint is True
        assert cfg.kr_cv_keep_base_location is True
        assert cfg.head_zero_offset_guard is True
        assert cfg.skip_high_conf_rule_rows is True

    def test_float_defaults(self):
        cfg = MLRuntimeConfig()
        assert cfg.kr_cvvc_blank_floor == pytest.approx(0.64)
        assert cfg.kr_cvc_blank_floor == pytest.approx(0.62)
        assert cfg.kr_cv_blank_floor == pytest.approx(0.60)
        assert cfg.anchor_lock_min_conf == pytest.approx(0.45)
        assert cfg.coupled_min_conf_model_offset == pytest.approx(0.0)
        assert cfg.selector_top1_min_gain == pytest.approx(0.0)
        assert cfg.skip_high_conf_min_margin == pytest.approx(12.0)

    def test_int_defaults(self):
        cfg = MLRuntimeConfig()
        assert cfg.batch_inference_size == 256
        assert cfg.progress_every == 80

    def test_optional_none_defaults(self):
        cfg = MLRuntimeConfig()
        assert cfg.coupled_min_conf is None
        assert cfg.coupled_min_conf_cv_family is None
        assert cfg.coupled_min_conf_bridge_family is None
        assert cfg.kr_cv_ovl_cap_ratio is None
        assert cfg.ja_cv_ovl_cap_ratio is None
        assert cfg.skip_high_conf_min is None
        assert cfg.skip_high_conf_max_blank is None
        assert cfg.selector_margin_override is None

    def test_string_defaults(self):
        cfg = MLRuntimeConfig()
        assert cfg.route == "legacy"
        assert cfg.kr_model_dir == ""
        assert cfg.ja_model_dir == ""
        assert cfg.workspace_root == ""
        assert cfg.export_root == ""
        assert cfg.coupled_device == ""


# ---------------------------------------------------------------------------
# from_env() — flag parsing
# ---------------------------------------------------------------------------

class TestFromEnvFlags:
    def test_ml_disabled_set_by_1(self, clean):
        clean.setenv("UTOA_DISABLE_OTO_ML", "1")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ml_disabled is True

    def test_ml_disabled_set_by_true(self, clean):
        clean.setenv("UTOA_DISABLE_OTO_ML", "true")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ml_disabled is True

    def test_ml_disabled_false_by_0(self, clean):
        clean.setenv("UTOA_DISABLE_OTO_ML", "0")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ml_disabled is False

    def test_ml_disabled_false_by_false(self, clean):
        clean.setenv("UTOA_DISABLE_OTO_ML", "false")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ml_disabled is False

    @pytest.mark.parametrize("val", ["yes", "on", "YES", "ON", "True", "TRUE"])
    def test_truthy_variants(self, clean, val):
        clean.setenv("UTOA_DISABLE_OTO_ML", val)
        assert MLRuntimeConfig.from_env().ml_disabled is True

    @pytest.mark.parametrize("val", ["no", "off", "NO", "OFF", "False", "n"])
    def test_falsy_variants(self, clean, val):
        clean.setenv("UTOA_DISABLE_OTO_ML", val)
        assert MLRuntimeConfig.from_env().ml_disabled is False

    def test_ensemble_disabled_by_0(self, clean):
        clean.setenv("UTOA_ML_ENSEMBLE_ENABLE", "0")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ensemble_enabled is False

    def test_delta_forced(self, clean):
        clean.setenv("UTOA_FORCE_OTO_DELTA", "1")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.delta_forced is True
        assert cfg.delta_disabled is False

    def test_delta_disabled(self, clean):
        clean.setenv("UTOA_DISABLE_OTO_DELTA", "1")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.delta_disabled is True


# ---------------------------------------------------------------------------
# from_env() — path strings, int, float
# ---------------------------------------------------------------------------

class TestFromEnvValues:
    def test_kr_model_dir(self, clean):
        clean.setenv("UTOA_KR_OTO_ML_DIR", "/my/kr/models")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.kr_model_dir == "/my/kr/models"

    def test_ja_model_dir(self, clean):
        clean.setenv("UTOA_JA_OTO_ML_DIR", "/my/ja/models")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.ja_model_dir == "/my/ja/models"

    def test_workspace_root(self, clean):
        clean.setenv("UTOA_OTO_ML_WORKSPACE_ROOT", "/ws")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.workspace_root == "/ws"

    def test_batch_size_parsed(self, clean):
        clean.setenv("UTOA_ML_BATCH_INFERENCE_SIZE", "128")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.batch_inference_size == 128

    def test_batch_size_floor_at_32(self, clean):
        clean.setenv("UTOA_ML_BATCH_INFERENCE_SIZE", "10")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.batch_inference_size == 32

    def test_progress_every_floor_at_20(self, clean):
        clean.setenv("UTOA_ML_PROGRESS_EVERY", "5")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.progress_every == 20

    def test_anchor_lock_min_conf_clamped(self, clean):
        clean.setenv("UTOA_ML_ANCHOR_LOCK_MIN_CONF", "1.5")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.anchor_lock_min_conf == pytest.approx(1.0)

    def test_anchor_lock_min_conf_low_clamped(self, clean):
        clean.setenv("UTOA_ML_ANCHOR_LOCK_MIN_CONF", "-0.3")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.anchor_lock_min_conf == pytest.approx(0.0)

    def test_coupled_min_conf_set(self, clean):
        clean.setenv("UTOA_ML_COUPLED_MIN_CONF", "0.72")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.coupled_min_conf == pytest.approx(0.72)

    def test_coupled_min_conf_empty_stays_none(self, clean):
        # Env var absent → None
        cfg = MLRuntimeConfig.from_env()
        assert cfg.coupled_min_conf is None

    def test_selector_margin_override(self, clean):
        clean.setenv("UTOA_SELECTOR_MIN_MARGIN", "18.5")
        cfg = MLRuntimeConfig.from_env()
        assert cfg.selector_margin_override == pytest.approx(18.5)


# ---------------------------------------------------------------------------
# from_env() — route normalisation
# ---------------------------------------------------------------------------

class TestRouteNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("legacy", "legacy"),
        ("LEGACY", "legacy"),
        ("e2e_hybrid", "e2e_hybrid"),
        ("e2e", "e2e_hybrid"),
        ("hybrid", "e2e_hybrid"),
        ("E2E_HYBRID", "e2e_hybrid"),
        ("unknown_value", "legacy"),
        ("", "legacy"),
    ])
    def test_route_values(self, clean, raw, expected):
        if raw:
            clean.setenv("UTOA_ML_ROUTE", raw)
        cfg = MLRuntimeConfig.from_env()
        assert cfg.route == expected


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelperMethods:
    def test_model_dir_for_korean(self):
        cfg = MLRuntimeConfig(kr_model_dir="/kr/path", ja_model_dir="/ja/path")
        assert cfg.model_dir_for_language("korean") == "/kr/path"

    def test_model_dir_for_japanese(self):
        cfg = MLRuntimeConfig(kr_model_dir="/kr/path", ja_model_dir="/ja/path")
        assert cfg.model_dir_for_language("japanese") == "/ja/path"

    def test_model_dir_for_unknown_returns_kr(self):
        # Unknown language falls back to kr_model_dir
        cfg = MLRuntimeConfig(kr_model_dir="/kr/path", ja_model_dir="/ja/path")
        assert cfg.model_dir_for_language("chinese") == "/kr/path"

    def test_model_dir_empty_when_unset(self):
        cfg = MLRuntimeConfig()
        assert cfg.model_dir_for_language("korean") == ""

    def test_blank_floor_cvvc(self):
        cfg = MLRuntimeConfig(kr_cvvc_blank_floor=0.64)
        assert cfg.blank_floor_for_format("cvvc") == pytest.approx(0.64)

    def test_blank_floor_cvc(self):
        cfg = MLRuntimeConfig(kr_cvc_blank_floor=0.62)
        assert cfg.blank_floor_for_format("cvc") == pytest.approx(0.62)

    def test_blank_floor_cv(self):
        cfg = MLRuntimeConfig(kr_cv_blank_floor=0.60)
        assert cfg.blank_floor_for_format("cv") == pytest.approx(0.60)

    def test_blank_floor_unknown_returns_none(self):
        cfg = MLRuntimeConfig()
        assert cfg.blank_floor_for_format("vcv") is None
        assert cfg.blank_floor_for_format("") is None

    def test_ovl_cap_ratio_korean(self):
        cfg = MLRuntimeConfig(kr_cv_ovl_cap_ratio=0.75)
        assert cfg.cv_ovl_cap_ratio_for_language("korean") == pytest.approx(0.75)

    def test_ovl_cap_ratio_japanese(self):
        cfg = MLRuntimeConfig(ja_cv_ovl_cap_ratio=0.80)
        assert cfg.cv_ovl_cap_ratio_for_language("japanese") == pytest.approx(0.80)

    def test_ovl_cap_ratio_unset_is_none(self):
        cfg = MLRuntimeConfig()
        assert cfg.cv_ovl_cap_ratio_for_language("korean") is None
        assert cfg.cv_ovl_cap_ratio_for_language("japanese") is None

    def test_ovl_cap_ratio_unknown_language_is_none(self):
        cfg = MLRuntimeConfig(kr_cv_ovl_cap_ratio=0.75)
        assert cfg.cv_ovl_cap_ratio_for_language("chinese") is None


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_ml_config()
        b = get_ml_config()
        assert a is b

    def test_set_replaces_instance(self):
        custom = MLRuntimeConfig(ml_disabled=True)
        set_ml_config(custom)
        assert get_ml_config() is custom
        assert get_ml_config().ml_disabled is True

    def test_reset_triggers_reinit(self, clean):
        first = get_ml_config()
        reset_ml_config()
        second = get_ml_config()
        assert first is not second

    def test_reset_then_env_change_is_picked_up(self, clean):
        # Ensure the singleton re-reads the env after reset
        cfg_before = get_ml_config()
        assert cfg_before.ml_disabled is False

        reset_ml_config()
        clean.setenv("UTOA_DISABLE_OTO_ML", "1")
        cfg_after = get_ml_config()
        assert cfg_after.ml_disabled is True

    def test_set_config_does_not_affect_separate_instance(self):
        custom = MLRuntimeConfig(batch_inference_size=64)
        set_ml_config(custom)
        assert get_ml_config().batch_inference_size == 64
        # Direct construction is independent
        direct = MLRuntimeConfig()
        assert direct.batch_inference_size == 256
