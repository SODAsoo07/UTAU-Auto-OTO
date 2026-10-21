import numpy as np
import soundfile as sf

from universal_auto_oto.nasal_frame import frame_features, model_available, nasal_probability


def test_nasal_model_ships_and_scores_every_frame(tmp_path):
    # separation itself is measured on the banks' hand OTOs (held-out AUC 0.924), not on synthetic tones
    assert model_available()
    rate = 44100
    t = np.arange(int(1.2 * rate)) / rate
    wav = tmp_path / "sample.wav"
    sf.write(wav, 0.2 * np.sin(2 * np.pi * 220 * t), rate)
    times = np.array([0.0, 300.0, 600.0, 1190.0])
    assert frame_features(wav, times).shape == (4, 62)
    p = nasal_probability(wav, times)
    assert p.shape == (4,)
    assert np.all(np.isfinite(p) & (p >= 0.0) & (p <= 1.0))
