import pytest

from universal_auto_oto.hsmm_decoder import (
    HSMMStateSpec, decode_segmental_hsmm, decode_segmental_hsmm_coarse_to_refine,
)


@pytest.mark.parametrize("beam", [0, 1, 64])
def test_activity_tail_allows_silence_but_penalizes_omitted_activity(beam):
    times = list(range(0, 200, 10))
    state = HSMMStateSpec("v", "vowel", 40, 40)
    scores = {"v": [1.] * 8 + [.8] * 12}
    kwargs = dict(allow_leading_gap=True, beam_width_per_state=beam,
                  trailing_gap_penalty_per_ms=.01)
    constant = decode_segmental_hsmm([state], times, scores, **kwargs)
    silence = decode_segmental_hsmm([state], times, scores,
        trailing_gap_frame_weights=[1.] * 8 + [0.] * 12, **kwargs)
    zero = decode_segmental_hsmm([state], times, scores,
        trailing_gap_frame_weights=[0.] * 20, **kwargs)
    assert constant.ok and silence.ok and zero.ok
    assert constant.states[-1].end_ms == 200
    assert silence.states[-1].end_ms == 80
    assert zero.states[-1].end_ms == 40
    assert silence.meta["trailing_gap_ms"] == 120
    assert silence.meta["trailing_gap_cost"] == 0


@pytest.mark.parametrize("decoder", [decode_segmental_hsmm, decode_segmental_hsmm_coarse_to_refine])
def test_all_one_weights_preserve_constant_objective(decoder):
    times = list(range(0, 800, 10))
    states = [HSMMStateSpec("v", "vowel", 80, 120)]
    scores = {"v": [1.] * 32 + [-2.] * 48}
    kwargs = dict(allow_leading_gap=True, trailing_gap_penalty_per_ms=.003)
    before = decoder(states, times, scores, **kwargs)
    after = decoder(states, times, scores, trailing_gap_frame_weights=[1.] * 80, **kwargs)
    assert before.ok and after.ok
    assert before.states == after.states
    assert before.score == pytest.approx(after.score)
    assert before.meta["trailing_gap_cost"] == pytest.approx(after.meta["trailing_gap_cost"])


def test_coarse_and_refinement_passes_use_activity_tail():
    result = decode_segmental_hsmm_coarse_to_refine(
        [HSMMStateSpec("v", "vowel", 80, 80)], list(range(0, 800, 10)),
        {"v": [1.] * 80}, allow_leading_gap=True, trailing_gap_penalty_per_ms=.01,
        trailing_gap_frame_weights=[1.] * 32 + [0.] * 48, coarse_stride=4,
    )
    assert result.ok and result.meta["coarse_to_refine"]
    assert result.states[-1].end_ms == 320
    assert result.meta["trailing_gap_weighted"]
    assert result.meta["trailing_gap_cost"] == 0


def test_fractional_activity_cost_and_full_fallback():
    # Thirty short states cannot fit on the eleven-frame coarse grid.
    states = [HSMMStateSpec(str(i), "vowel", 10, 10) for i in range(30)]
    result = decode_segmental_hsmm_coarse_to_refine(
        states, list(range(0, 400, 10)), {s.state_id: [1.] * 40 for s in states},
        trailing_gap_penalty_per_ms=.01, trailing_gap_frame_weights=[.25] * 40,
        coarse_stride=4, fallback_to_full=True,
    )
    assert result.ok and not result.meta["coarse_to_refine"]
    assert result.meta["coarse_to_refine_reason"].startswith("coarse_failed:")
    assert result.states[-1].end_ms == 300
    assert result.meta["trailing_gap_cost"] == pytest.approx(.25)
    assert result.score == pytest.approx(29.75)


@pytest.mark.parametrize("decoder", [decode_segmental_hsmm, decode_segmental_hsmm_coarse_to_refine])
@pytest.mark.parametrize("weights", [[1], [float("nan")] * 20, [float("inf")] * 20,
                                     [-.1] * 20, [1.1] * 20, ["bad"] * 20])
def test_invalid_activity_weights_fail_closed(decoder, weights):
    result = decoder([HSMMStateSpec("v", "vowel", 40, 40)], list(range(0, 200, 10)),
                     {"v": [1.] * 20}, trailing_gap_frame_weights=weights)
    assert not result.ok and result.reason == "invalid_trailing_gap_weights"
