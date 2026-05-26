"""
Unit tests for core.model_resolver — ModelResolver class and singleton.

Tests cover:
- Root path helpers: model_root, workspace_root, export_root, installed_root
- Env-override vs default path logic
- collect_candidates: deduplication, ordering, two-stage/vbest filtering
- resolve_backend / resolve_lightgbm / resolve_ensemble: return None for
  non-existent directories (no real disk required for path logic tests)
- is_disabled: UTOA_ML_RESOLVER_DISABLE env var
- Singleton lifecycle: get_resolver / reset_resolver
- Config injection (custom MLRuntimeConfig)
- read_backend: parses model_meta.json; handles missing file gracefully
"""

from __future__ import annotations

import json
import os
import pytest

from core.model_resolver import ModelResolver, _looks_like_two_stage, get_resolver, reset_resolver
from core.runtime_config import MLRuntimeConfig, reset_ml_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_singleton():
    reset_resolver()
    reset_ml_config()
    yield
    reset_resolver()
    reset_ml_config()


@pytest.fixture()
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("UTOA_"):
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _make_resolver(**kwargs) -> ModelResolver:
    """Build a ModelResolver with a custom MLRuntimeConfig."""
    return ModelResolver(MLRuntimeConfig(**kwargs))


# ---------------------------------------------------------------------------
# Root path helpers
# ---------------------------------------------------------------------------

class TestRootPaths:
    def test_base_dir_points_to_workspace_root(self):
        assert os.path.isdir(os.path.join(ModelResolver._BASE_DIR, "core"))
        assert os.path.isfile(os.path.join(ModelResolver._BASE_DIR, "core", "runtime", "model_resolver.py"))

    def test_model_root_uses_kr_override(self):
        r = _make_resolver(kr_model_dir="/custom/kr")
        assert r.model_root("korean") == "/custom/kr"

    def test_model_root_uses_ja_override(self):
        r = _make_resolver(ja_model_dir="/custom/ja")
        assert r.model_root("japanese") == "/custom/ja"

    def test_model_root_default_contains_lang(self):
        r = _make_resolver()
        root = r.model_root("korean")
        assert "korean" in root
        assert "oto_ml" in root

    def test_model_root_default_japanese(self):
        r = _make_resolver()
        root = r.model_root("japanese")
        assert "japanese" in root

    def test_installed_root_contains_lang(self):
        r = _make_resolver()
        path = r.installed_root("korean")
        assert "korean" in path
        assert "models_installed" in path

    def test_structured_export_root_contains_lang(self):
        r = _make_resolver()
        path = r.structured_export_root("korean")
        assert "korean" in path

    def test_export_root_uses_env_override(self):
        r = _make_resolver(export_root="/my/export")
        assert r.export_root() == "/my/export"

    def test_export_root_default_exists(self):
        r = _make_resolver()
        path = r.export_root()
        # Just verify it's a non-empty string pointing somewhere under BASE_DIR
        assert path
        assert ModelResolver._BASE_DIR in path

    def test_workspace_root_uses_env_override(self):
        r = _make_resolver(workspace_root="/my/workspace")
        path = r.workspace_root("korean")
        assert path.startswith("/my/workspace")
        assert "korean" in path

    def test_workspace_root_default_contains_lang(self):
        r = _make_resolver()
        path = r.workspace_root("korean")
        assert "korean" in path

    def test_has_explicit_override_true_for_korean(self):
        r = _make_resolver(kr_model_dir="/kr")
        assert r.has_explicit_override("korean") is True

    def test_has_explicit_override_false_when_empty(self):
        r = _make_resolver()
        assert r.has_explicit_override("korean") is False


# ---------------------------------------------------------------------------
# read_backend
# ---------------------------------------------------------------------------

class TestReadBackend:
    def test_missing_file_returns_empty(self, tmp_path):
        assert ModelResolver.read_backend(str(tmp_path)) == ""

    def test_reads_backend_field(self, tmp_path):
        meta = {"backend": "lightgbm", "version": "1.0"}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == "lightgbm"

    def test_reads_coupled_backend(self, tmp_path):
        meta = {"backend": "coupled_nn_v2_rawmel"}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == "coupled_nn_v2_rawmel"

    def test_empty_backend_field_returns_empty(self, tmp_path):
        meta = {"backend": ""}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == ""

    def test_missing_backend_key_returns_empty(self, tmp_path):
        meta = {"version": "1.0"}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == ""

    def test_malformed_json_returns_empty(self, tmp_path):
        (tmp_path / "model_meta.json").write_text("NOT JSON", encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == ""

    def test_backend_normalised_to_lowercase(self, tmp_path):
        meta = {"backend": "LightGBM"}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert ModelResolver.read_backend(str(tmp_path)) == "lightgbm"


# ---------------------------------------------------------------------------
# collect_candidates
# ---------------------------------------------------------------------------

class TestCollectCandidates:
    def test_returns_list(self):
        r = _make_resolver()
        candidates = r.collect_candidates("korean", "cvvc")
        assert isinstance(candidates, list)

    def test_no_duplicates(self):
        r = _make_resolver()
        candidates = r.collect_candidates("korean", "cvvc")
        abs_paths = [os.path.abspath(p) for p in candidates]
        assert len(abs_paths) == len(set(abs_paths))

    def test_explicit_override_limits_version_search(self):
        # Explicit override restricts the version-dir search to the override root.
        # Legacy export paths are always appended as fallback — that's by design.
        r = _make_resolver(kr_model_dir="/only/this")
        candidates = r.collect_candidates("korean", "cvvc")
        # workspace_root and installed_root should NOT appear since we have an override
        assert not any(("ml_workspace" in c or "models_installed" in c) for c in candidates)
        # The override root itself must be in the search (even if no model exists there)
        assert any(c.startswith("/only/this") for c in candidates)

    def test_vbest_excluded_by_default(self, tmp_path):
        # Create a vbest subdirectory with model_meta.json
        vbest_dir = tmp_path / "cvvc" / "vbest_run1"
        vbest_dir.mkdir(parents=True)
        (vbest_dir / "model_meta.json").write_text('{"backend":"lightgbm"}', encoding="utf-8")
        r = _make_resolver(kr_model_dir=str(tmp_path), auto_select_vbest=False)
        candidates = r.collect_candidates("korean", "cvvc")
        # The vbest_run1 directory itself must not be a candidate.
        # Use the absolute path to avoid substring false positives from the tmp_path name.
        vbest_abs = os.path.abspath(str(vbest_dir))
        assert not any(os.path.abspath(c) == vbest_abs for c in candidates)

    def test_vbest_included_when_enabled(self, tmp_path):
        vbest = tmp_path / "cvvc" / "vbest_run1"
        vbest.mkdir(parents=True)
        (vbest / "model_meta.json").write_text('{"backend":"lightgbm"}', encoding="utf-8")
        r = _make_resolver(kr_model_dir=str(tmp_path), auto_select_vbest=True)
        candidates = r.collect_candidates("korean", "cvvc")
        assert any("vbest" in c for c in candidates)

    def test_two_stage_excluded_by_default(self, tmp_path):
        v2 = tmp_path / "cvvc" / "v2_two_stage"
        v2.mkdir(parents=True)
        (v2 / "model_meta.json").write_text('{"backend":"lightgbm"}', encoding="utf-8")
        r = _make_resolver(kr_model_dir=str(tmp_path), two_stage_model_enabled=False)
        candidates = r.collect_candidates("korean", "cvvc")
        assert not any("v2" in c for c in candidates)

    def test_family_suffix_included_in_paths(self):
        r = _make_resolver()
        candidates_no_family = r.collect_candidates("korean", "cvvc")
        candidates_with_family = r.collect_candidates("korean", "cvvc", alias_family="cv")
        # With alias_family, there should be additional candidate paths
        assert len(candidates_with_family) >= len(candidates_no_family)

    def test_cvc_fallback_to_cv_format(self):
        r = _make_resolver()
        candidates = r.collect_candidates("korean", "cvc")
        # CVC format should include CV fallback paths
        assert any("cv" in c for c in candidates)


# ---------------------------------------------------------------------------
# resolve_* — return None when nothing exists on disk
# ---------------------------------------------------------------------------

class TestResolveReturnsNoneWhenMissing:
    def test_resolve_lightgbm_none_when_no_models(self, tmp_path):
        r = _make_resolver(kr_model_dir=str(tmp_path))
        assert r.resolve_lightgbm("korean", "cvvc") is None

    def test_resolve_ensemble_none_when_no_models(self, tmp_path):
        r = _make_resolver(kr_model_dir=str(tmp_path))
        assert r.resolve_ensemble("korean", "cvvc") is None

    def test_resolve_coupled_none_when_no_models(self, tmp_path):
        r = _make_resolver(kr_model_dir=str(tmp_path))
        assert r.resolve_coupled("korean", "cvvc") is None

    def test_resolve_none_when_no_models(self, tmp_path):
        r = _make_resolver(kr_model_dir=str(tmp_path))
        assert r.resolve("korean", "cvvc") is None


# ---------------------------------------------------------------------------
# resolve_* — return correct path when model exists on disk
# ---------------------------------------------------------------------------

class TestResolveFindsModels:
    def _make_model_dir(self, tmp_path, backend: str, subpath: str = "cvvc/v1") -> str:
        model_dir = tmp_path / subpath
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_meta.json").write_text(
            json.dumps({"backend": backend}), encoding="utf-8"
        )
        return str(tmp_path)

    def test_resolve_lightgbm_finds_dir(self, tmp_path):
        root = self._make_model_dir(tmp_path, "lightgbm")
        r = _make_resolver(kr_model_dir=root)
        result = r.resolve_lightgbm("korean", "cvvc")
        assert result is not None
        assert os.path.isfile(os.path.join(result, "model_meta.json"))

    def test_resolve_ensemble_finds_dir(self, tmp_path):
        root = self._make_model_dir(tmp_path, "ensemble_v1")
        r = _make_resolver(kr_model_dir=root, ensemble_enabled=True)
        result = r.resolve_ensemble("korean", "cvvc")
        assert result is not None

    def test_resolve_prefers_ensemble_over_lightgbm(self, tmp_path):
        # Both exist — ensemble should win when enabled
        lgbm = tmp_path / "cvvc" / "v1"
        lgbm.mkdir(parents=True)
        (lgbm / "model_meta.json").write_text('{"backend":"lightgbm"}', encoding="utf-8")

        ens = tmp_path / "cvvc" / "v1_ensemble"
        ens.mkdir(parents=True)
        (ens / "model_meta.json").write_text('{"backend":"ensemble_v1"}', encoding="utf-8")

        r = _make_resolver(kr_model_dir=str(tmp_path), ensemble_enabled=True)
        result = r.resolve("korean", "cvvc")
        assert result is not None
        assert ModelResolver.read_backend(result) == "ensemble_v1"

    def test_resolve_lightgbm_when_ensemble_disabled(self, tmp_path):
        lgbm = tmp_path / "cvvc" / "v1"
        lgbm.mkdir(parents=True)
        (lgbm / "model_meta.json").write_text('{"backend":"lightgbm"}', encoding="utf-8")

        ens = tmp_path / "cvvc" / "v1_ensemble"
        ens.mkdir(parents=True)
        (ens / "model_meta.json").write_text('{"backend":"ensemble_v1"}', encoding="utf-8")

        r = _make_resolver(kr_model_dir=str(tmp_path), ensemble_enabled=False)
        result = r.resolve("korean", "cvvc")
        assert result is not None
        assert ModelResolver.read_backend(result) == "lightgbm"

    def test_resolve_coupled_v2_over_v1(self, tmp_path):
        v1 = tmp_path / "cvvc" / "v1_coupled"
        v1.mkdir(parents=True)
        (v1 / "model_meta.json").write_text('{"backend":"coupled_nn_v1"}', encoding="utf-8")

        v2 = tmp_path / "cvvc" / "v2"
        v2.mkdir(parents=True)
        (v2 / "model_meta.json").write_text(
            '{"backend":"coupled_nn_v2_rawmel"}', encoding="utf-8"
        )

        r = _make_resolver(kr_model_dir=str(tmp_path), ensemble_enabled=False, two_stage_model_enabled=True)
        # resolve_coupled should prefer v2_rawmel
        result = r.resolve_coupled("korean", "cvvc")
        assert result is not None
        assert ModelResolver.read_backend(result) == "coupled_nn_v2_rawmel"

    def test_unlabelled_model_counts_as_lightgbm(self, tmp_path):
        # Model with no backend field → treated as lightgbm (backward compat)
        model_dir = tmp_path / "cvvc" / "v1"
        model_dir.mkdir(parents=True)
        (model_dir / "model_meta.json").write_text('{"version":"1.0"}', encoding="utf-8")
        r = _make_resolver(kr_model_dir=str(tmp_path))
        result = r.resolve_lightgbm("korean", "cvvc")
        assert result is not None


# ---------------------------------------------------------------------------
# is_disabled (rollback escape-hatch)
# ---------------------------------------------------------------------------

class TestIsDisabled:
    def test_not_disabled_by_default(self, clean):
        r = _make_resolver()
        assert r.is_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_disabled_by_truthy_env(self, clean, val):
        clean.setenv("UTOA_ML_RESOLVER_DISABLE", val)
        r = _make_resolver()
        assert r.is_disabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off"])
    def test_not_disabled_by_falsy_env(self, clean, val):
        clean.setenv("UTOA_ML_RESOLVER_DISABLE", val)
        r = _make_resolver()
        assert r.is_disabled() is False


# ---------------------------------------------------------------------------
# _looks_like_two_stage helper
# ---------------------------------------------------------------------------

class TestLooksLikeTwoStage:
    @pytest.mark.parametrize("name,expected", [
        ("v2", True),
        ("v2_coupled", True),
        ("v2_two_stage", True),
        ("v1_two_stage", True),
        ("twostage_v1", True),
        ("v1", False),
        ("v1_ensemble", False),
        ("vbest_run1", False),
        ("", False),
    ])
    def test_classification(self, name, expected):
        assert _looks_like_two_stage(name) is expected


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_resolver_returns_same_instance(self):
        a = get_resolver()
        b = get_resolver()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_resolver()
        reset_resolver()
        b = get_resolver()
        assert a is not b

    def test_pass_config_returns_one_shot_instance(self):
        cfg = MLRuntimeConfig(kr_model_dir="/one_shot")
        r = get_resolver(config=cfg)
        assert r.cfg.kr_model_dir == "/one_shot"
        # Singleton should NOT be replaced
        singleton = get_resolver()
        assert singleton.cfg.kr_model_dir != "/one_shot"

    def test_lazy_cfg_uses_singleton_config(self):
        from core.runtime_config import set_ml_config
        custom = MLRuntimeConfig(kr_model_dir="/injected")
        set_ml_config(custom)
        r = ModelResolver()  # no config passed → lazy fetch
        assert r.cfg.kr_model_dir == "/injected"

    def test_explicit_config_not_replaced_by_singleton(self):
        from core.runtime_config import set_ml_config
        set_ml_config(MLRuntimeConfig(kr_model_dir="/global"))
        r = ModelResolver(MLRuntimeConfig(kr_model_dir="/local"))
        assert r.cfg.kr_model_dir == "/local"
