from pathlib import Path

import numpy as np

from universal_auto_oto.audio import _select_boundaries_monotonic
from universal_auto_oto.generator import _select_decoder_mode


def test_monotonic_boundary_decoder_selects_ordered_acoustic_peaks():
    times = np.arange(0.0, 3005.0, 5.0)
    scores = np.zeros_like(times)
    for peak_ms, height in ((800.0, 1.0), (2050.0, 0.9)):
        scores += height * np.exp(-0.5 * ((times - peak_ms) / 25.0) ** 2)

    result = _select_boundaries_monotonic(times, scores, 0.0, 3000.0, 3)

    assert result is not None
    boundaries, _confidences = result
    assert np.all(np.diff(boundaries) > 0)
    assert abs(boundaries[1] - 800.0) <= 30.0
    assert abs(boundaries[2] - 2050.0) <= 30.0


def test_decoder_gate_keeps_japanese_and_continuous_korean_cvc_local():
    assert _select_decoder_mode(Path("_ka-ki-ku.wav"), language_hint="japanese", format_hint="cvvc") == "local"
    assert _select_decoder_mode(Path("da'di'du.wav"), language_hint="korean", format_hint="cvc") == "local"
    assert _select_decoder_mode(Path("dada_didi_dudu.wav"), language_hint="korean", format_hint="cvc") == "monotonic"
    assert _select_decoder_mode(Path("_a'a'e.wav"), language_hint="korean", format_hint="cvvc") == "monotonic"
