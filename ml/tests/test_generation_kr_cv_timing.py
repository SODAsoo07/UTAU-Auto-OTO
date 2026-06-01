from types import SimpleNamespace

import pytest

from core.generation.kr.kr_oto_cv import (
    _is_silence_like_phone_mark,
    _select_kr_cv_onset_slice,
)


def _phone(mark, start_ms, end_ms):
    return SimpleNamespace(mark=mark, minTime=start_ms / 1000.0, maxTime=end_ms / 1000.0)


def test_korean_liquid_phone_marks_are_not_silence():
    assert not _is_silence_like_phone_mark("r")
    assert not _is_silence_like_phone_mark("l")
    assert _is_silence_like_phone_mark("sil")
    assert _is_silence_like_phone_mark("br")


@pytest.mark.parametrize("mark", ["r", "l"])
def test_korean_liquid_cv_onset_slice_keeps_consonant_interval(mark):
    phones = [_phone(mark, 100.0, 180.0), _phone("a", 180.0, 440.0)]

    onset_idx, c_start, c_end, n_start, n_end, glide_dur_ms = _select_kr_cv_onset_slice(phones)

    assert onset_idx == 0
    assert c_start == pytest.approx(100.0)
    assert c_end == pytest.approx(180.0)
    assert n_start == pytest.approx(180.0)
    assert n_end == pytest.approx(440.0)
    assert glide_dur_ms == pytest.approx(0.0)
