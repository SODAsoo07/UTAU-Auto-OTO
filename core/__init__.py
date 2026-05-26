"""AutoOTO core package.

Export policy:
- Keep category packages (`alignment`, `generation`, `model_context`, `oto_ml`,
  `runtime`, `timing`, `common`) as the primary import surface.
- Keep root module wrappers for backward compatibility while implementations
  live under category folders.
"""

from importlib import import_module as _import_module

__all__ = [
    "alignment",
    "generation",
    "model_context",
    "oto_ml",
    "runtime",
    "timing",
    "common",
]


def __getattr__(name: str):
    if name in __all__:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
