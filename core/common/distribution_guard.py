from __future__ import annotations

import os
import sys


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def is_distribution_build() -> bool:
    # Explicit override for local debugging.
    if _env_flag("UTOA_FORCE_DISTRIBUTION_BUILD", default=False):
        return True
    if _env_flag("UTOA_FORCE_DEV_BUILD", default=False):
        return False
    return bool(getattr(sys, "frozen", False))


def is_training_paths_enabled() -> bool:
    # Distribution build에서는 기본적으로 훈련 경로를 비활성화한다.
    if is_distribution_build():
        return _env_flag("UTOA_ENABLE_TRAINING_IN_DISTRIBUTION", default=False)
    return True

