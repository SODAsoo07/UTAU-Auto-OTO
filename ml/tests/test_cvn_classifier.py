import numpy as np

from core.cvn_classifier import predict_cvn
from core.cvn_types import FrameFeatures


def _demo_features() -> FrameFeatures:
    n = 9
    return FrameFeatures(
        sample_rate=16000,
        frame_ms=25.0,
        hop_ms=10.0,
        times_ms=(np.arange(n, dtype=np.float32) * 10.0),
        log_mel=np.zeros((n, 80), dtype=np.float32),
        f0_hz=np.array([0.0, 0.0, 0.0, 220.0, 230.0, 210.0, 0.0, 0.0, 0.0], dtype=np.float32),
        voiced_mask=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool),
        rms=np.array([0.001, 0.001, 0.001, 0.12, 0.11, 0.13, 0.06, 0.07, 0.05], dtype=np.float32),
        zcr=np.array([0.01, 0.01, 0.01, 0.08, 0.09, 0.10, 0.42, 0.45, 0.40], dtype=np.float32),
        spectral_centroid=np.array([300, 300, 300, 700, 680, 720, 3500, 3600, 3400], dtype=np.float32),
        spectral_flatness=np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.82, 0.84, 0.80], dtype=np.float32),
    )


def test_predict_cvn_rule_labels_expected_regions():
    out = predict_cvn(_demo_features(), backend="rule", smooth_window=1, min_run_frames=1)
    assert out.labels.tolist()[3:6] == ["V", "V", "V"]
    assert out.labels.tolist()[6:] == ["C", "C", "C"]
    assert out.probs.shape == (9, 2)
    assert np.allclose(np.sum(out.probs, axis=1), 1.0, atol=1e-6)


def test_predict_cvn_even_smoothing_window_is_supported():
    out = predict_cvn(_demo_features(), backend="rule", smooth_window=2, min_run_frames=1)
    assert out.probs.shape == (9, 2)
    assert out.labels.shape == (9,)


def test_predict_cvn_logreg_backend_uses_linear_npz(tmp_path):
    model_path = tmp_path / "cvn_linear.npz"
    np.savez(
        str(model_path),
        weights=np.zeros((6, 2), dtype=np.float32),
        bias=np.array([0.0, 5.0], dtype=np.float32),
    )
    out = predict_cvn(
        _demo_features(),
        backend="logreg",
        model_path=str(model_path),
        smooth_window=1,
        min_run_frames=1,
    )
    assert set(out.labels.tolist()) == {"V"}


def test_predict_cvn_logreg_uses_model_c_threshold(tmp_path):
    model_path = tmp_path / "cvn_linear_threshold.npz"
    np.savez(
        str(model_path),
        weights=np.zeros((6, 2), dtype=np.float32),
        bias=np.array([0.0, 0.0], dtype=np.float32),
        c_threshold=np.array(0.01, dtype=np.float32),
    )
    out = predict_cvn(
        _demo_features(),
        backend="logreg",
        model_path=str(model_path),
        smooth_window=1,
        min_run_frames=1,
    )
    assert set(out.labels.tolist()) == {"C"}


def test_predict_cvn_logreg_falls_back_to_rule_when_model_invalid(tmp_path):
    bad_model = tmp_path / "bad_model.npz"
    np.savez(str(bad_model), wrong=np.array([1.0], dtype=np.float32))

    out = predict_cvn(
        _demo_features(),
        backend="logreg",
        model_path=str(bad_model),
        smooth_window=1,
        min_run_frames=1,
    )
    assert out.labels.tolist()[3:6] == ["V", "V", "V"]
    assert out.labels.tolist()[6:] == ["C", "C", "C"]
