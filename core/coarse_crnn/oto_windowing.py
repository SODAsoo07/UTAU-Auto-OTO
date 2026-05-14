"""Deprecated direct-parameter OTO CRNN compatibility wrapper."""

from __future__ import annotations

from importlib import import_module as _import_module
import sys as _sys

_deprecated = _import_module("core.coarse_crnn.deprecated.direct_param.oto_windowing")
_sys.modules[__name__] = _deprecated
