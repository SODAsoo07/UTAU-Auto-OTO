"""
ModelResolver — single entry-point for OTO-ML model directory resolution.

Replaces the five scattered ``_resolve_*`` functions in oto_ml_refiner.py with
a class that holds all path-search logic and delegates configuration to
``MLRuntimeConfig``.

Usage
-----
    from core.model_resolver import get_resolver

    resolver = get_resolver()                         # uses process-wide config
    model_dir = resolver.resolve("korean", "cvvc")
    lgbm_dir  = resolver.resolve_lightgbm("korean", "cvvc")

For testing, pass a custom config:
    from core.runtime_config import MLRuntimeConfig
    from core.model_resolver import ModelResolver
    resolver = ModelResolver(MLRuntimeConfig(kr_model_dir="/tmp/test_models"))
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from core.format_type_utils import normalize_format_type
from core.oto_ml_policy import normalize_alias_family
from core.runtime_config import MLRuntimeConfig, get_ml_config


class ModelResolver:
    """Resolves ML model directories for a given language / format / alias-family.

    All search logic that was previously spread across::

        _collect_model_dir_candidates()
        _resolve_backend_model_dir()
        _resolve_lightgbm_model_dir()
        _resolve_ensemble_model_dir()
        _resolve_coupled_model_dir()
        _resolve_model_dir()

    …is now centralised here.
    """

    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def __init__(self, config: Optional[MLRuntimeConfig] = None) -> None:
        self._cfg = config  # None → lazy-fetch from process singleton on first use

    @property
    def cfg(self) -> MLRuntimeConfig:
        if self._cfg is None:
            self._cfg = get_ml_config()
        return self._cfg

    # ------------------------------------------------------------------
    # Root path helpers
    # ------------------------------------------------------------------

    def model_root(self, language: str) -> str:
        """Return the primary model-root for *language* (may be env-overridden)."""
        lang = str(language).strip().lower()
        override = self.cfg.model_dir_for_language(lang)
        if override:
            return override
        return os.path.join(self._BASE_DIR, "assets", "models", "oto_ml", lang)

    def workspace_root(self, language: str) -> str:
        lang = str(language).strip().lower()
        ws = self.cfg.workspace_root
        if ws:
            return os.path.join(ws, lang)
        local = os.path.join(self._BASE_DIR, "ml_workspace", "models")
        if os.path.isdir(local):
            return os.path.join(local, lang)
        return os.path.join(self._BASE_DIR, "logs", "ml_workspace", "models", lang)

    def export_root(self) -> str:
        configured = self.cfg.export_root
        if configured:
            return configured
        noml = os.path.join(self._BASE_DIR, "ML_models_noml_auto")
        if os.path.isdir(noml):
            return noml
        return os.path.join(self._BASE_DIR, "ML_models")

    def installed_root(self, language: str) -> str:
        lang = str(language).strip().lower()
        return os.path.join(self._BASE_DIR, "models_installed", "oto_ml", lang)

    def structured_export_root(self, language: str) -> str:
        return os.path.join(self.export_root(), str(language).strip().lower())

    def has_explicit_override(self, language: str) -> bool:
        return bool(self.cfg.model_dir_for_language(language))

    # ------------------------------------------------------------------
    # Bundle-meta helpers
    # ------------------------------------------------------------------

    @staticmethod
    def read_backend(model_dir: str) -> str:
        """Return the backend string recorded in *model_dir*/model_meta.json, or ''."""
        try:
            meta_path = os.path.join(model_dir, "model_meta.json")
            if not os.path.isfile(meta_path):
                return ""
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            return str(meta.get("backend", "") or "").strip().lower()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def collect_candidates(
        self,
        language: str,
        format_type: str,
        alias_family: str = "",
    ) -> List[str]:
        """Return an ordered list of model-directory candidates for the given context."""
        fmt = normalize_format_type(language, format_type) or "general"
        family = normalize_alias_family(alias_family)
        lang = str(language or "").strip().lower()
        allow_two_stage = self.cfg.two_stage_model_enabled
        allow_vbest = self.cfg.auto_select_vbest
        candidates: List[str] = []

        def _append_version_dirs(base_path: str) -> None:
            if not base_path:
                return
            candidates.extend([
                os.path.join(base_path, "v1"),
                os.path.join(base_path, "v1_ensemble"),
                os.path.join(base_path, "v1_coupled"),
            ])
            if not os.path.isdir(base_path):
                return
            try:
                for name in os.listdir(base_path):
                    if not str(name).lower().startswith("v"):
                        continue
                    if not allow_vbest and str(name).strip().lower().startswith("vbest"):
                        continue
                    if not allow_two_stage and _looks_like_two_stage(name):
                        continue
                    path = os.path.join(base_path, name)
                    if os.path.isfile(os.path.join(path, "model_meta.json")):
                        candidates.append(path)
            except Exception:
                return

        if self.has_explicit_override(language):
            root_list = (self.model_root(language),)
        else:
            root_list = (
                self.workspace_root(language),
                self.installed_root(language),
                self.structured_export_root(language),
                self.model_root(language),
            )

        for root in root_list:
            if os.path.isfile(os.path.join(root, "model_meta.json")) and (
                allow_two_stage or not _looks_like_two_stage(os.path.basename(root))
            ):
                candidates.append(root)
            else:
                if family:
                    _append_version_dirs(os.path.join(root, fmt, "families", family))
                    _append_version_dirs(os.path.join(root, f"{fmt}_{family}"))
                _append_version_dirs(os.path.join(root, fmt))
                if fmt == "cvc":
                    if family:
                        _append_version_dirs(os.path.join(root, "cv", "families", family))
                        _append_version_dirs(os.path.join(root, f"cv_{family}"))
                    _append_version_dirs(os.path.join(root, "cv"))
                if fmt != "general":
                    if family:
                        _append_version_dirs(os.path.join(root, "general", "families", family))
                        _append_version_dirs(os.path.join(root, f"general_{family}"))
                    _append_version_dirs(os.path.join(root, "general"))

        # Legacy ML_models export paths
        exp = self.export_root()
        legacy: List[str] = []
        if family:
            legacy += [
                os.path.join(exp, f"{lang}_{fmt}_{family}_v1"),
                os.path.join(exp, f"{lang}_{fmt}_{family}"),
            ]
        legacy += [
            os.path.join(exp, f"{lang}_{fmt}_v1"),
            os.path.join(exp, f"{lang}_{fmt}"),
            os.path.join(exp, f"{lang}_{fmt}_profile_run_new"),
            os.path.join(exp, f"{lang}_{fmt}_profile_run_v4"),
            os.path.join(exp, f"{lang}_{fmt}_profile_run_v3"),
            os.path.join(exp, f"{lang}_{fmt}_profile_run"),
        ]
        if fmt == "cvc":
            if family:
                legacy += [
                    os.path.join(exp, f"{lang}_cv_{family}_v1"),
                    os.path.join(exp, f"{lang}_cv_{family}"),
                ]
            legacy += [
                os.path.join(exp, f"{lang}_cv_v1"),
                os.path.join(exp, f"{lang}_cv"),
            ]
        candidates.extend(legacy)

        # Optional cross-language fallback (default OFF via same_language_borrow_only)
        if not self.cfg.same_language_borrow_only:
            for alt_lang in ("korean", "japanese"):
                if alt_lang == lang:
                    continue
                alt_root = self.model_root(alt_lang)
                for cand in (
                    os.path.join(alt_root, fmt, "families", family, "v1") if family else "",
                    os.path.join(alt_root, f"{fmt}_{family}", "v1") if family else "",
                    os.path.join(alt_root, fmt, "v1"),
                    os.path.join(alt_root, "general", "v1"),
                ):
                    if cand and os.path.isfile(os.path.join(cand, "model_meta.json")):
                        candidates.append(cand)

        # Deduplicate while preserving order
        seen: set = set()
        ordered: List[str] = []
        for p in candidates:
            if not p:
                continue
            key = os.path.abspath(p)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(p)
        return ordered

    # ------------------------------------------------------------------
    # Backend-specific resolution
    # ------------------------------------------------------------------

    def resolve_backend(
        self,
        language: str,
        format_type: str,
        alias_family: str = "",
        backend: str = "",
    ) -> Optional[str]:
        """Return the first candidate directory matching *backend*, or None."""
        backend_norm = str(backend or "").strip().lower()
        for cand in self.collect_candidates(language, format_type, alias_family=alias_family):
            if not os.path.isfile(os.path.join(cand, "model_meta.json")):
                continue
            if not backend_norm:
                return cand
            detected = self.read_backend(cand)
            if detected == backend_norm:
                return cand
            # Treat unlabelled models as lightgbm for backward compat
            if backend_norm == "lightgbm" and not detected:
                return cand
        return None

    def resolve_lightgbm(self, language: str, format_type: str, alias_family: str = "") -> Optional[str]:
        return self.resolve_backend(language, format_type, alias_family=alias_family, backend="lightgbm")

    def resolve_ensemble(self, language: str, format_type: str, alias_family: str = "") -> Optional[str]:
        return self.resolve_backend(language, format_type, alias_family=alias_family, backend="ensemble_v1")

    def resolve_coupled(self, language: str, format_type: str, alias_family: str = "") -> Optional[str]:
        """Prefer coupled_nn_v2_rawmel; fall back to coupled_nn_v1."""
        v2 = self.resolve_backend(
            language, format_type, alias_family=alias_family, backend="coupled_nn_v2_rawmel"
        )
        if v2:
            return v2
        return self.resolve_backend(
            language, format_type, alias_family=alias_family, backend="coupled_nn_v1"
        )

    def resolve(self, language: str, format_type: str, alias_family: str = "") -> Optional[str]:
        """Return the best available model dir: ensemble > lightgbm > coupled."""
        if self.cfg.ensemble_enabled:
            ensemble = self.resolve_ensemble(language, format_type, alias_family=alias_family)
            if ensemble:
                return ensemble
        lgbm = self.resolve_lightgbm(language, format_type, alias_family=alias_family)
        if lgbm:
            return lgbm
        return self.resolve_coupled(language, format_type, alias_family=alias_family)

    # ------------------------------------------------------------------
    # Rollback escape-hatch
    # ------------------------------------------------------------------

    def is_disabled(self) -> bool:
        """Return True when env ``UTOA_ML_RESOLVER_DISABLE=1`` is set.

        This allows instant rollback to the old inline resolution path without
        a code change: set the variable and the caller falls back to the legacy
        branch.
        """
        raw = str(os.environ.get("UTOA_ML_RESOLVER_DISABLE", "")).strip().lower()
        return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _looks_like_two_stage(name: str) -> bool:
    token = str(name or "").strip().lower()
    return token.startswith("v2") or "two_stage" in token or "twostage" in token


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_RESOLVER: Optional[ModelResolver] = None


def get_resolver(config: Optional[MLRuntimeConfig] = None) -> ModelResolver:
    """Return the process-wide ModelResolver (lazy-init).

    Pass *config* to override for the lifetime of this call (creates a new
    one-shot instance, does not replace the singleton).
    """
    global _RESOLVER
    if config is not None:
        return ModelResolver(config)
    if _RESOLVER is None:
        _RESOLVER = ModelResolver()
    return _RESOLVER


def reset_resolver() -> None:
    """Force re-creation of the singleton on next ``get_resolver()`` call."""
    global _RESOLVER
    _RESOLVER = None


__all__ = [
    "ModelResolver",
    "get_resolver",
    "reset_resolver",
]
