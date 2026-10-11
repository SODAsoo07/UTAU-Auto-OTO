from universal_auto_oto.hsmm_decoder import (
    HSMMStateFrameWindow,
    HSMMStateSpec,
    decode_segmental_hsmm,
    decode_segmental_hsmm_coarse_to_refine,
)


def _score_range(times: list[float], start: float, end: float) -> list[float]:
    return [2.0 if start <= time < end else -2.0 for time in times]


def test_segmental_hsmm_recovers_known_sequence_intervals() -> None:
    times = [float(index * 10) for index in range(40)]
    states = [
        HSMMStateSpec("slot0.c", "consonant", 50.0, 90.0, mode_duration_ms=70.0, duration_sigma_ms=10.0),
        HSMMStateSpec("slot0.v", "vowel", 100.0, 160.0, mode_duration_ms=130.0, duration_sigma_ms=15.0),
        HSMMStateSpec("slot1.c", "consonant", 50.0, 90.0, mode_duration_ms=70.0, duration_sigma_ms=10.0),
        HSMMStateSpec("slot1.v", "vowel", 100.0, 160.0, mode_duration_ms=130.0, duration_sigma_ms=15.0),
    ]
    result = decode_segmental_hsmm(
        states,
        times,
        {
            "slot0.c": _score_range(times, 0.0, 70.0),
            "slot0.v": _score_range(times, 70.0, 200.0),
            "slot1.c": _score_range(times, 200.0, 270.0),
            "slot1.v": _score_range(times, 270.0, 400.0),
        },
        beam_width_per_state=32,
        timeout_ms=1000,
    )
    assert result.ok
    assert [(state.start_ms, state.end_ms) for state in result.states] == [
        (0.0, 70.0),
        (70.0, 200.0),
        (200.0, 270.0),
        (270.0, 400.0),
    ]


def test_segmental_hsmm_honors_state_windows() -> None:
    times = [float(index * 10) for index in range(50)]
    states = [HSMMStateSpec("slot0.v", "vowel", 80.0, 130.0, mode_duration_ms=100.0, duration_sigma_ms=12.0)]
    scores = {"slot0.v": [2.0 if 300.0 <= t < 400.0 else 1.7 if 80.0 <= t < 180.0 else -2.0 for t in times]}
    unrestricted = decode_segmental_hsmm(states, times, scores, beam_width_per_state=32, timeout_ms=1000, allow_leading_gap=True)
    restricted = decode_segmental_hsmm(
        states,
        times,
        scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        state_frame_windows={"slot0.v": HSMMStateFrameWindow(6, 12, 16, 22)},
    )
    assert unrestricted.ok and unrestricted.states[0].start_ms == 300.0
    assert restricted.ok
    assert 60.0 <= restricted.states[0].start_ms <= 120.0
    assert 160.0 <= restricted.states[0].end_ms <= 220.0


def test_segmental_hsmm_applies_instance_specific_transition_start_score() -> None:
    times = [float(index * 10) for index in range(30)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 100.0, 200.0),
        HSMMStateSpec("slot1.v", "vowel", 100.0, 200.0),
    ]
    result = decode_segmental_hsmm(
        states,
        times,
        {state.state_id: [0.0] * len(times) for state in states},
        allow_trailing_gap=False,
        transition_start_scores={"slot1.v": [0.2 if index == 15 else 0.0 for index in range(len(times))]},
    )

    assert result.ok
    assert [(state.start_frame, state.end_frame) for state in result.states] == [(0, 15), (15, 30)]
    assert result.meta["transition_start_score_state_count"] == 1


def test_segmental_hsmm_trailing_gap_penalty_prevents_early_path_collapse() -> None:
    times = [float(index * 10) for index in range(20)]
    state = HSMMStateSpec("slot0.v", "vowel", 40.0, 40.0, mode_duration_ms=40.0, duration_sigma_ms=10.0)
    scores = {state.state_id: [1.0] * len(times)}

    unpenalized = decode_segmental_hsmm(
        [state], times, scores, allow_leading_gap=True, allow_trailing_gap=True
    )
    penalized = decode_segmental_hsmm(
        [state],
        times,
        scores,
        allow_leading_gap=True,
        allow_trailing_gap=True,
        trailing_gap_penalty_per_ms=0.003,
    )

    assert unpenalized.ok and penalized.ok
    assert unpenalized.meta["trailing_gap_ms"] > penalized.meta["trailing_gap_ms"]
    assert penalized.meta["trailing_gap_ms"] == 0.0


def test_segmental_hsmm_coarse_to_refine_handles_internal_gap() -> None:
    times = [float(index * 10) for index in range(80)]
    states = [
        HSMMStateSpec("slot0.c", "consonant", 40.0, 90.0, mode_duration_ms=60.0, duration_sigma_ms=12.0),
        HSMMStateSpec("slot0.v", "vowel", 90.0, 170.0, mode_duration_ms=130.0, duration_sigma_ms=18.0),
        HSMMStateSpec("slot1.c", "consonant", 40.0, 90.0, mode_duration_ms=60.0, duration_sigma_ms=12.0),
        HSMMStateSpec("slot1.v", "vowel", 90.0, 170.0, mode_duration_ms=130.0, duration_sigma_ms=18.0),
    ]
    result = decode_segmental_hsmm_coarse_to_refine(
        states,
        times,
        {
            "slot0.c": _score_range(times, 50.0, 110.0),
            "slot0.v": _score_range(times, 110.0, 240.0),
            "slot1.c": _score_range(times, 390.0, 450.0),
            "slot1.v": _score_range(times, 450.0, 580.0),
        },
        coarse_stride=4,
        refine_window_ms=70.0,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        allow_internal_gaps=True,
        max_internal_gap_ms=220.0,
    )
    assert result.ok
    assert result.meta["coarse_to_refine"] is True
    assert [(state.start_ms, state.end_ms) for state in result.states] == [
        (50.0, 110.0),
        (110.0, 240.0),
        (390.0, 450.0),
        (450.0, 580.0),
    ]


def test_segmental_hsmm_leading_and_internal_gap_controls() -> None:
    times = [float(index * 10) for index in range(80)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
        HSMMStateSpec("slot1.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
    ]
    scores = {
        "slot0.v": _score_range(times, 80.0, 200.0),
        "slot1.v": _score_range(times, 520.0, 640.0),
    }
    result = decode_segmental_hsmm(
        states,
        times,
        scores,
        beam_width_per_state=32,
        timeout_ms=1000,
        allow_leading_gap=True,
        max_leading_gap_ms=120.0,
        allow_internal_gaps=True,
        max_internal_gap_ms=400.0,
        internal_gap_penalty_per_ms=0.00005,
    )
    assert result.ok
    assert [(state.start_ms, state.end_ms) for state in result.states] == [(80.0, 200.0), (520.0, 640.0)]
    assert result.meta["leading_gap_ms"] == 80.0
    assert result.meta["internal_gap_ms"] == 320.0
    assert result.meta["internal_gap_count"] == 1
    assert result.meta["max_internal_gap_observed_ms"] == 320.0
    assert result.meta["internal_gaps"] == [
        {
            "after_state_index": 0,
            "before_state_index": 1,
            "after_state_id": "slot0.v",
            "before_state_id": "slot1.v",
            "start_frame": 20,
            "end_frame": 52,
            "start_ms": 200.0,
            "end_ms": 520.0,
            "duration_ms": 320.0,
        }
    ]


def test_internal_gap_frame_weights_charge_only_the_weighted_frames() -> None:
    times = [float(index * 10) for index in range(80)]
    states = [
        HSMMStateSpec("slot0.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
        HSMMStateSpec("slot1.v", "vowel", 90.0, 140.0, mode_duration_ms=120.0, duration_sigma_ms=15.0),
    ]
    scores = {"slot0.v": _score_range(times, 80.0, 200.0), "slot1.v": _score_range(times, 520.0, 640.0)}
    common = dict(beam_width_per_state=32, timeout_ms=1000, allow_leading_gap=True, max_leading_gap_ms=120.0,
                  allow_internal_gaps=True, max_internal_gap_ms=400.0, internal_gap_penalty_per_ms=0.01)
    flat = decode_segmental_hsmm(states, times, scores, **common)
    ones = decode_segmental_hsmm(states, times, scores, internal_gap_frame_weights=[1.0] * len(times), **common)
    silent = [0.0 if 200.0 <= t < 520.0 else 1.0 for t in times]
    pause = decode_segmental_hsmm(states, times, scores, internal_gap_frame_weights=silent, **common)
    assert flat.ok and ones.ok and pause.ok
    assert [(s.start_ms, s.end_ms) for s in pause.states] == [(80.0, 200.0), (520.0, 640.0)]
    assert abs(ones.score - flat.score) < 1e-9
    assert abs((pause.score - flat.score) - 0.01 * 320.0) < 1e-9     # the silent 320 ms gap is free
    coarse = decode_segmental_hsmm_coarse_to_refine(states, times, scores, coarse_stride=4,
                                                     internal_gap_frame_weights=silent, **common)
    assert coarse.ok and [(s.start_ms, s.end_ms) for s in coarse.states] == [(80.0, 200.0), (520.0, 640.0)]
    bad = decode_segmental_hsmm(states, times, scores, internal_gap_frame_weights=[2.0] * len(times), **common)
    assert not bad.ok and bad.reason == "invalid_internal_gap_weights"


def test_unpruned_coarse_pass_keeps_the_path_a_narrow_beam_drops() -> None:
    # The only good path holds state a past its best-looking short endings; an 8-wide coarse
    # beam keeps only those short endings and loses every path that covers the file.
    times = [float(index * 10) for index in range(300)]
    states = [
        HSMMStateSpec("a", "vowel", 50.0, 2500.0, mode_duration_ms=300.0, duration_sigma_ms=2000.0),
        HSMMStateSpec("b", "vowel", 400.0, 700.0, mode_duration_ms=550.0, duration_sigma_ms=100.0),
    ]
    scores = {"a": _score_range(times, 0.0, 2000.0), "b": _score_range(times, 2400.0, 3000.0)}
    common = dict(coarse_stride=4, refine_window_ms=60.0, beam_width_per_state=8, timeout_ms=5000,
                  allow_trailing_gap=False)
    pruned = decode_segmental_hsmm_coarse_to_refine(states, times, scores, **common)
    unpruned = decode_segmental_hsmm_coarse_to_refine(states, times, scores, coarse_beam_width_per_state=0, **common)
    full = decode_segmental_hsmm(states, times, scores, beam_width_per_state=0, timeout_ms=60000,
                                 allow_trailing_gap=False)
    assert not pruned.ok
    assert unpruned.ok and unpruned.meta["coarse_beam_width_per_state"] == 0
    assert [(s.start_ms, s.end_ms) for s in unpruned.states] == [(s.start_ms, s.end_ms) for s in full.states]
    assert unpruned.score == full.score


def test_exact_search_matches_unpruned_decoder_on_random_problems() -> None:
    import random

    from universal_auto_oto.hsmm_decoder import decode_segmental_hsmm_exact

    rng = random.Random(7)
    for case in range(40):
        frames = rng.randint(25, 70)
        times = [float(index * 5) for index in range(frames)]
        states = [
            HSMMStateSpec(f"s{index}", rng.choice(["vowel", "consonant"]), float(rng.choice([5, 10, 15])),
                          float(rng.choice([40, 60, 90])), mode_duration_ms=float(rng.choice([0, 20, 30])),
                          duration_sigma_ms=float(rng.choice([10, 20])), over_duration_penalty_scale=rng.choice([1.0, 0.3]))
            for index in range(rng.randint(2, 6))
        ]
        scores = {spec.state_id: [rng.uniform(-2.0, 2.0) for _ in times] for spec in states}
        options = dict(
            allow_leading_gap=bool(case % 2),
            max_leading_gap_ms=float(rng.choice([0, 30])),
            leading_gap_penalty_per_ms=float(rng.choice([0.0, 0.01])),
            allow_trailing_gap=case % 5 != 0,
            trailing_gap_penalty_per_ms=0.003,
            trailing_gap_frame_weights=[rng.random() for _ in times] if case % 3 == 0 else None,
            allow_internal_gaps=case % 4 != 0,
            internal_gap_penalty_per_ms=float(rng.choice([0.0002, 0.01])),
            max_internal_gap_ms=float(rng.choice([10, 30])),
            internal_gap_frame_weights=[rng.random() for _ in times] if case % 2 == 0 else None,
            transition_start_scores={states[-1].state_id: [rng.uniform(-1, 1) for _ in times]},
            state_frame_windows={states[0].state_id: HSMMStateFrameWindow(0, 4, 1, None)} if case % 6 == 0 else None,
        )
        reference = decode_segmental_hsmm(states, times, scores, beam_width_per_state=0, timeout_ms=60000, **options)
        exact = decode_segmental_hsmm_exact(states, times, scores, **options)
        assert exact.ok == reference.ok, case
        if not reference.ok:
            continue
        assert abs(exact.score - reference.score) < 1e-9, case
        assert [(s.start_frame, s.end_frame) for s in exact.states] == [
            (s.start_frame, s.end_frame) for s in reference.states], case
        assert exact.meta["internal_gaps"] == reference.meta["internal_gaps"], case
        assert exact.meta["exact_search"] is True


def test_exact_search_score_is_never_below_the_pruned_search() -> None:
    import random

    from universal_auto_oto.hsmm_decoder import decode_segmental_hsmm_exact

    rng = random.Random(11)
    times = [float(index * 5) for index in range(600)]
    states = [HSMMStateSpec(f"s{index}", "vowel", 20.0, 300.0, mode_duration_ms=120.0, duration_sigma_ms=60.0)
              for index in range(12)]
    scores = {spec.state_id: [rng.uniform(-1.0, 1.0) for _ in times] for spec in states}
    pruned = decode_segmental_hsmm_coarse_to_refine(states, times, scores, beam_width_per_state=8, timeout_ms=60000)
    exact = decode_segmental_hsmm_exact(states, times, scores)
    assert pruned.ok and exact.ok
    assert exact.score >= pruned.score - 1e-9
