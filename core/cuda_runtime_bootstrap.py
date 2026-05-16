"""Compatibility wrapper: moved to `core.runtime.cuda_runtime_bootstrap`."""

from importlib import import_module as _import_module

_impl = _import_module("core.runtime.cuda_runtime_bootstrap")
for _name, _value in _impl.__dict__.items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _import_module, _impl, _name, _value
