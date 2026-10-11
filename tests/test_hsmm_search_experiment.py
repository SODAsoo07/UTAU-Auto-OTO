import pytest

from universal_auto_oto.hsmm_decoder import HSMMStateSpec, decode_segmental_hsmm


@pytest.mark.parametrize('padding', [0, 200, 1300])
def test_activity_limit_can_reach_late_onset_and_preserves_padding(padding):
    times = list(range(0, 2200+padding, 10))
    onset = 1500+padding
    scores = {'v': [1. if onset <= t < onset+100 else -10. for t in times]}
    kwargs = dict(allow_leading_gap=True, beam_width_per_state=0)
    states = [HSMMStateSpec('v', 'vowel', 100, 100)]
    limited = decode_segmental_hsmm(states, times, scores, max_leading_gap_ms=900, **kwargs)
    activity = decode_segmental_hsmm(states, times, scores, max_leading_gap_ms=onset+120, **kwargs)
    assert limited.ok and activity.ok
    assert limited.states[0].start_ms < onset
    assert activity.states[0].start_ms == onset
    assert activity.states[0].end_ms == onset+100


def test_explicit_initial_state_stays_at_origin_without_leading_gap():
    times = list(range(0, 2000, 10))
    result = decode_segmental_hsmm([HSMMStateSpec('sil', 'silence', 100, 100)], times,
        {'sil': [-10.] * 150 + [1.] * 50}, allow_leading_gap=False,
        max_leading_gap_ms=1800, beam_width_per_state=0)
    assert result.ok and result.states[0].start_ms == 0
