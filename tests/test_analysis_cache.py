import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto import analysis_cache
from universal_auto_oto.voicing import FrameVoicing, frame_voicing


def _wav(path, seed=0):
    rate = 16000
    rng = np.random.default_rng(seed)
    t = np.arange(rate) / rate
    audio = 0.3 * np.sin(2 * np.pi * 180 * t)
    audio[6000:9000] = 0.01 * rng.standard_normal(3000)
    sf.write(path, audio, rate)
    return path


def test_cache_only_while_active_and_values_are_read_only(tmp_path):
    wav = _wav(tmp_path / "a.wav")
    calls = []

    def compute():
        calls.append(1)
        return np.arange(3.0)

    analysis_cache.cached("x", wav, (), compute)
    analysis_cache.cached("x", wav, (), compute)
    assert len(calls) == 2  # inactive: every call computes
    with analysis_cache.active():
        first = analysis_cache.cached("x", wav, (), compute)
        assert analysis_cache.cached("x", wav, (), compute) is first and len(calls) == 3
        with pytest.raises(ValueError):
            first[0] = 1.0  # a cached array cannot be changed in place
        analysis_cache.cached("x", wav, ("other",), compute)
        assert len(calls) == 4  # different parameters, separate entry
    assert not analysis_cache.is_active()


def test_incremental_voicing_equals_one_whole_file_pass(tmp_path):
    wav = _wav(tmp_path / "b.wav", seed=3)
    whole = FrameVoicing(wav).voiced()
    with analysis_cache.active():
        shared = frame_voicing(wav)
        shared.voiced(np.arange(10, 40))
        shared.voiced(np.arange(30, 120))
        assert frame_voicing(wav) is shared
        assert np.array_equal(shared.voiced(), whole)
