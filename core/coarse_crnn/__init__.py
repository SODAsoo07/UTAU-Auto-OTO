from __future__ import annotations

from core.coarse_crnn.labels import COARSE_LABELS, LABEL_TO_ID, coarse_for_phone


def predict_oto(*args, **kwargs):
    from core.coarse_crnn.oto_inference import predict_oto as _predict_oto

    return _predict_oto(*args, **kwargs)


def check_coarse_crnn_ready(*args, **kwargs):
    from core.coarse_crnn.workflow import check_coarse_crnn_ready as _check_ready

    return _check_ready(*args, **kwargs)


def run_coarse_crnn_align(*args, **kwargs):
    from core.coarse_crnn.workflow import run_coarse_crnn_align as _run_align

    return _run_align(*args, **kwargs)

__all__ = [
    "COARSE_LABELS",
    "LABEL_TO_ID",
    "check_coarse_crnn_ready",
    "coarse_for_phone",
    "predict_oto",
    "run_coarse_crnn_align",
]
