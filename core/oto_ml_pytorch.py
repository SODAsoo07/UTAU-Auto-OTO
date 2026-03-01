"""
PyTorch backend stub for future OTO ML correction.
"""

from __future__ import annotations


def load_pytorch_bundle(model_dir, meta=None, schema=None):
    raise NotImplementedError("PyTorch OTO ML backend is not implemented yet.")


def predict_pytorch_deltas(payload, feature_row, meta=None, schema=None):
    raise NotImplementedError("PyTorch OTO ML backend is not implemented yet.")
