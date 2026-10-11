"""Per-run cache of WAV analyses that do not depend on the decode.

The sonorant identity passes re-align the same recordings with new templates; the acoustic
evidence, band balance, cepstra and voicing of a WAV are the same every time. Inside an
``active()`` block (one generation run, or one alignment worker process) they are computed
once per WAV. Cached arrays are made read-only so any in-place change raises instead of
silently altering a later pass. Outside a block every call computes as before.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

_cache: dict[tuple, Any] | None = None
_depth = 0


def enable() -> None:
    global _cache, _depth
    _depth += 1
    if _cache is None:
        _cache = {}


def disable() -> None:
    global _cache, _depth
    _depth = max(0, _depth - 1)
    if _depth == 0:
        _cache = None


@contextmanager
def active() -> Iterator[None]:
    enable()
    try:
        yield
    finally:
        disable()


def is_active() -> bool:
    return _cache is not None


def _file_key(path: Path) -> tuple:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


def cached(kind: str, path: Path, params: tuple, compute: Callable[[], Any]) -> Any:
    """compute() once per (kind, file, params) while active; values are frozen read-only."""
    if _cache is None:
        return compute()
    key = (kind, *_file_key(path), params)
    if key not in _cache:
        _cache[key] = _freeze(compute())
    return _cache[key]


def state(kind: str, path: Path, factory: Callable[[], Any]) -> Any:
    """A mutable per-file object owned by the caller (e.g. incrementally filled frame values)."""
    if _cache is None:
        return factory()
    key = (kind, *_file_key(path))
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze(item)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze(item)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _freeze(getattr(value, name))
    return value


__all__ = ["active", "cached", "disable", "enable", "is_active", "state"]
