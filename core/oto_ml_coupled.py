"""
Coupled mel+OTO backend (coupled_nn_v1).

.. deprecated::
    이 모듈은 하위 호환을 위한 re-export shim입니다.
    실제 구현은 ``core.oto_ml.coupled`` 패키지를 참조하세요.

Runtime builds exclude ``core.oto_ml.coupled.training`` via ``EXCLUDED_TRAINING_MODULES``
in build.py.  The training symbols below are imported lazily so that the shim
does not raise ``ImportError`` in a stripped runtime environment.
"""

from __future__ import annotations

# ── Re-export from new module locations ──────────────────────────────────────

# Constants & model utilities
from core.oto_ml.coupled.model import (  # noqa: F401
    COUPLED_BACKEND,
    COUPLED_BACKEND_RAWMEL,
    COUPLED_MODEL_FILE,
    COUPLED_MODEL_ONNX_FILE,
    COUPLED_MODEL_ONNX_META_FILE,
    PATCH_FEATURES,
    _build_model,
    _build_model_rawmel,
    _env_int,
    _import_torch,
    _require_numpy,
    _require_training_stack,
    _resolve_device,
    _row_feature_vector,
    _row_patch_vector,
    _stable_hash_to_unit,
)

# Inference (always available at runtime)
from core.oto_ml.coupled.inference import (  # noqa: F401
    load_coupled_bundle,
    predict_coupled_deltas,
)

# Training & dataset — conditionally imported.
# These symbols are only needed by ml/scripts/* (training pipeline).
# In a stripped runtime build (without core.oto_ml.coupled.training), importing
# them raises ImportError which is caught here so the shim stays importable.
try:
    from core.oto_ml.coupled.training import (  # noqa: F401
        _prepare_training_frame,
        build_and_save_coupled_dataset,
        evaluate_coupled_bundle,
        train_coupled_bundle,
        train_coupled_bundle_rawmel,
    )
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False
    _prepare_training_frame = None          # type: ignore[assignment]
    build_and_save_coupled_dataset = None   # type: ignore[assignment]
    evaluate_coupled_bundle = None          # type: ignore[assignment]
    train_coupled_bundle = None             # type: ignore[assignment]
    train_coupled_bundle_rawmel = None      # type: ignore[assignment]

# Pairing
from core.oto_ml.pairing.vc_cv_pairing import (  # noqa: F401
    _batch_pair_positions,
    _build_vc_cv_pair_map,
)

# Re-export schema symbols that were previously accessed via this module
from core.oto_ml.features.schema import (  # noqa: F401
    AUX_TARGET_NAMES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_NAMES,
    canonicalize_feature_row,
    get_feature_schema,
    write_feature_schema,
)


__all__ = [
    "COUPLED_BACKEND",
    "COUPLED_BACKEND_RAWMEL",
    "COUPLED_MODEL_FILE",
    "COUPLED_MODEL_ONNX_FILE",
    "COUPLED_MODEL_ONNX_META_FILE",
    "PATCH_FEATURES",
    "build_and_save_coupled_dataset",
    "evaluate_coupled_bundle",
    "load_coupled_bundle",
    "predict_coupled_deltas",
    "train_coupled_bundle",
    "train_coupled_bundle_rawmel",
]
