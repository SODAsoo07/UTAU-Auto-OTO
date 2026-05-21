import numpy as np

from core.cvn_training import (
    IGNORE_LABEL,
    OtoRow,
    _predict_linear,
    _weighted_softmax_train,
    label_frames_from_oto_rows,
)


def test_label_frames_cv_uses_preutterance_boundary():
    times_ms = np.arange(0, 201, 10, dtype=np.float32)
    rows = [
        OtoRow(
            wav="a.wav",
            alias="ka",
            offset_ms=50.0,
            cons_ms=80.0,
            cutoff=-100.0,  # cutoff_abs = 50 + 100 = 150
            pre_ms=30.0,  # split = 50 + 30 = 80 (CV rule)
            ovl_ms=0.0,
        )
    ]
    labels = label_frames_from_oto_rows(
        times_ms=times_ms,
        wav_duration_ms=400.0,
        oto_rows=rows,
        language="korean",
    )

    # 0~40ms: ignore
    assert np.all(labels[times_ms < 50.0] == IGNORE_LABEL)
    # 50~80ms: C
    assert np.all(labels[(times_ms >= 50.0) & (times_ms <= 80.0)] == 0)
    # 90~150ms: V
    assert np.all(labels[(times_ms > 80.0) & (times_ms <= 150.0)] == 1)
    # 160ms~: ignore
    assert np.all(labels[times_ms > 150.0] == IGNORE_LABEL)


def test_label_frames_skips_vc_family_rows():
    times_ms = np.arange(0, 151, 10, dtype=np.float32)
    rows = [
        OtoRow(
            wav="a.wav",
            alias="a k",  # vc
            offset_ms=20.0,
            cons_ms=40.0,
            cutoff=-80.0,
            pre_ms=25.0,
            ovl_ms=0.0,
        )
    ]
    stats = {}
    labels = label_frames_from_oto_rows(
        times_ms=times_ms,
        wav_duration_ms=300.0,
        oto_rows=rows,
        language="korean",
        label_stats=stats,
    )

    # VC family rows are skipped for supervision, so all frames are ignored.
    assert np.all(labels == IGNORE_LABEL)
    assert int(stats.get("skip_non_cv_family", 0)) >= 1


def test_weighted_softmax_train_fits_separable_data():
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=[0.05, 0.10, 0.90, 0.80, 0.00, 0.0], scale=0.02, size=(120, 6))
    x1 = rng.normal(loc=[0.12, 0.08, 0.30, 0.20, 0.50, 1.0], scale=0.02, size=(120, 6))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros((120,), dtype=np.int64),
            np.ones((120,), dtype=np.int64),
        ]
    )

    w, b, _ = _weighted_softmax_train(x, y, epochs=220, lr=0.12, l2=1e-5)
    pred = _predict_linear(x, w, b)
    acc = float(np.mean(pred == y))
    assert acc > 0.95
