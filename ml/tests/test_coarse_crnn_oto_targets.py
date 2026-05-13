from __future__ import annotations

from core.coarse_crnn.oto_targets import _apply_row_context, anchors_to_oto_params, extract_alias_features, oto_params_to_anchors, resolve_cutoff_abs_ms
from core.coarse_crnn.oto_windowing import crop_oto_target_window, should_use_vcv_target_window, target_window_start_frame


def test_oto_params_roundtrip_negative_cutoff_coordinate_system():
    anchors = oto_params_to_anchors(
        offset=100.0,
        consonant=220.0,
        cutoff=-420.0,
        preutterance=80.0,
        overlap=35.0,
        duration_ms=900.0,
    )

    assert anchors.offset == 100.0
    assert anchors.overlap == 135.0
    assert anchors.preutterance == 180.0
    assert anchors.consonant == 320.0
    assert anchors.cutoff == 520.0

    params = anchors_to_oto_params(anchors, duration_ms=900.0)

    assert round(params["offset"], 3) == 100.0
    assert round(params["consonant"], 3) == 220.0
    assert round(params["cutoff"], 3) == -420.0
    assert round(params["preutterance"], 3) == 80.0
    assert round(params["overlap"], 3) == 35.0


def test_positive_cutoff_is_offset_relative_and_clamped():
    assert resolve_cutoff_abs_ms(offset_ms=100.0, cutoff=250.0, duration_ms=500.0) == 350.0
    assert resolve_cutoff_abs_ms(offset_ms=100.0, cutoff=900.0, duration_ms=500.0) == 499.0


def test_extract_alias_features_marks_vc_transition():
    features = extract_alias_features("a k", language="japanese")

    assert features["transition_type"] == "vc"
    assert features["alias_is_vc"] == 1.0
    assert features["alias_starts_vowel"] == 1.0
    assert features["alias_ends_vowel"] == 0.0


def test_extract_alias_features_emits_role_and_diphthong_flags():
    head = extract_alias_features("- ka", language="japanese")
    diphthong = extract_alias_features("야", language="korean")
    pure = extract_alias_features("아", language="korean")
    special = extract_alias_features("tt", language="korean", is_special=True)

    assert head["alias_role"] == "-cv"
    assert diphthong["is_diphthong"] == 1.0
    assert pure["is_diphthong"] == 0.0
    assert special["alias_role"] == "special"
    assert special["is_special"] == 1.0


def test_apply_row_context_adds_neighbor_aliases():
    rows = [
        {"oto_path": "x/oto.ini", "wav": "a.wav", "line_index": 0, "alias": "a", "alias_type": "vowel", "transition_type": "vowel", "alias_role": "v", "alias_starts_vowel": 1.0, "alias_ends_vowel": 1.0},
        {"oto_path": "x/oto.ini", "wav": "a.wav", "line_index": 1, "alias": "a k", "alias_type": "vc", "transition_type": "vc", "alias_role": "vc", "alias_starts_vowel": 1.0, "alias_ends_vowel": 0.0},
        {"oto_path": "x/oto.ini", "wav": "a.wav", "line_index": 2, "alias": "ka", "alias_type": "cv", "transition_type": "cv", "alias_role": "cv", "alias_starts_vowel": 0.0, "alias_ends_vowel": 1.0},
    ]

    _apply_row_context(rows)

    assert rows[1]["prev_alias"] == "a"
    assert rows[1]["next_alias"] == "ka"
    assert rows[1]["prev_alias_role"] == "v"
    assert rows[1]["next_alias_role"] == "cv"
    assert rows[1]["is_head_row"] == 0.0
    assert rows[1]["is_tail_row"] == 0.0


def test_vcv_target_window_is_format_specific_not_vowel_transition_specific():
    assert should_use_vcv_target_window("vcv", enabled=True)
    assert not should_use_vcv_target_window("cvvc", enabled=True)
    assert should_use_vcv_target_window("cvvc", enabled=True, formats=("vcv", "cvvc"), alias_type="vc")
    assert not should_use_vcv_target_window("cvvc", enabled=True, formats=("vcv", "cvvc"), alias_type="cv")
    assert not should_use_vcv_target_window("vcv", enabled=False)


def test_target_window_frames_can_be_format_specific():
    from types import SimpleNamespace

    from core.coarse_crnn.oto_windowing import target_window_frames_for

    cfg = SimpleNamespace(target_window_frame_overrides=("vcv=240", "cvvc=360"))

    assert target_window_frames_for(cfg, "vcv", 200) == 240
    assert target_window_frames_for(cfg, "cvvc", 200) == 360
    assert target_window_frames_for(cfg, "cvc", 200) == 200


def test_vcv_target_window_uses_row_prior_and_shifts_anchors():
    import numpy as np

    features = np.zeros((100, 4), dtype=np.float32)
    anchors = np.asarray([420.0, 440.0, 460.0, 500.0, 590.0], dtype=np.float32)

    cropped, shifted, duration_ms, start_frame = crop_oto_target_window(
        features,
        anchors,
        hop_sec=0.01,
        duration_ms=1000.0,
        row_index_in_wav=2,
        file_row_count=5,
        window_frames=20,
    )

    assert start_frame == target_window_start_frame(
        frame_count=100,
        row_index_in_wav=2,
        file_row_count=5,
        window_frames=20,
    )
    assert cropped.shape == (20, 4)
    assert duration_ms == 200.0
    assert shifted is not None
    assert round(float(shifted[2]), 3) == round(460.0 - start_frame * 10.0, 3)


def test_relative_oto_target_clips_right_boundary_outliers():
    from core.coarse_crnn.oto_param_priors import alias_timing_prior, normalize_relative_oto_target

    target = normalize_relative_oto_target(
        [20.0, 55.0, 90.0, 650.0, 1450.0],
        duration_ms=1600.0,
        alias_type="vcv",
    )
    prior = alias_timing_prior("vcv")

    assert round(target[0] * 1600.0, 3) == 20.0
    assert round(target[2] * 1600.0, 3) == 70.0
    assert round(target[3] * 1600.0, 3) == prior["max_cons_gap"]
    assert round(target[4] * 1600.0, 3) == prior["max_tail"]


def test_relative_oto_decode_blends_right_boundary_with_alias_prior():
    from core.coarse_crnn.oto_param_priors import alias_timing_prior, decode_relative_oto_params

    prior = alias_timing_prior("vc")
    params = decode_relative_oto_params(
        [0.02, 0.02, 0.05, 0.80, 0.90],
        duration_ms=1000.0,
        alias_type="vc",
        prior_blend=0.50,
    )

    expected_cons = 50.0 + ((prior["max_cons_gap"] * 0.5) + (prior["cons_gap"] * 0.5))
    expected_cutoff_abs = expected_cons + ((prior["max_tail"] * 0.5) + (prior["tail"] * 0.5))
    assert round(params["preutterance"], 3) == 50.0
    assert round(params["consonant"], 3) == round(expected_cons, 3)
    assert round(params["cutoff"], 3) == round(-expected_cutoff_abs, 3)


def test_active_relative_oto_target_normalizes_only_offset_origin():
    from core.coarse_crnn.oto_param_priors import decode_relative_oto_params, normalize_relative_oto_target

    target = normalize_relative_oto_target(
        [520.0, 548.0, 580.0, 650.0, 750.0],
        duration_ms=2000.0,
        alias_type="cv",
        active_origin_ms=400.0,
        active_span_ms=800.0,
    )
    assert round(target[0] * 800.0 + 400.0, 3) == 520.0
    assert round(target[2] * 2000.0, 3) == 60.0

    params = decode_relative_oto_params(
        target,
        duration_ms=2000.0,
        alias_type="cv",
        prior_blend=0.0,
        active_origin_ms=400.0,
        active_span_ms=800.0,
    )
    assert round(params["offset"], 3) == 520.0
    assert round(params["preutterance"], 3) == 60.0


def test_context_timing_prior_separates_cvvc_vc_from_vcv_vcv():
    from core.coarse_crnn.oto_param_priors import oto_timing_prior

    cvvc_vc = oto_timing_prior(format_type="cvvc", alias_type="vc", transition_type="vc")
    vcv_vcv = oto_timing_prior(format_type="vcv", alias_type="vcv", transition_type="vcv")

    assert cvvc_vc["max_tail"] < vcv_vcv["max_tail"]
    assert cvvc_vc["tail"] < vcv_vcv["tail"]
    assert cvvc_vc["max_cons_gap"] < vcv_vcv["max_cons_gap"]


def test_relative_oto_target_uses_format_alias_transition_context():
    from core.coarse_crnn.oto_param_priors import normalize_relative_oto_target, oto_timing_prior

    anchors = [20.0, 32.0, 58.0, 220.0, 420.0]
    target = normalize_relative_oto_target(
        anchors,
        duration_ms=1000.0,
        format_type="cvvc",
        alias_type="vc",
        transition_type="vc",
    )
    prior = oto_timing_prior(format_type="cvvc", alias_type="vc", transition_type="vc")

    assert round(target[3] * 1000.0, 3) == prior["max_cons_gap"]
    assert round(target[4] * 1000.0, 3) == prior["max_tail"]


def test_classify_blank_alias_empty_returns_blank():
    from core.coarse_crnn.oto_targets import _classify_blank_alias

    assert _classify_blank_alias("") == "blank"
    assert _classify_blank_alias("   ") == "blank"
    assert _classify_blank_alias(None) == "blank"
    assert _classify_blank_alias("-") == "blank"
    assert _classify_blank_alias("- ") == "blank"


def test_classify_blank_alias_silence_tokens_match_silence():
    from core.coarse_crnn.oto_targets import _classify_blank_alias

    # Single tokens
    assert _classify_blank_alias("R") == "silence"
    assert _classify_blank_alias("br") == "silence"
    assert _classify_blank_alias("pau") == "silence"
    assert _classify_blank_alias("sil") == "silence"
    assert _classify_blank_alias("rest") == "silence"
    # Hyphen-prefixed
    assert _classify_blank_alias("- R") == "silence"
    assert _classify_blank_alias("-R") == "silence"
    assert _classify_blank_alias("- pau") == "silence"
    # Combined silence tokens
    assert _classify_blank_alias("R sil") == "silence"
    assert _classify_blank_alias("br pau") == "silence"


def test_classify_blank_alias_normal_alias_returns_empty():
    from core.coarse_crnn.oto_targets import _classify_blank_alias

    # Real CV / VCV / CVVC aliases must NOT be classified as blank/silence.
    assert _classify_blank_alias("ka") == ""
    assert _classify_blank_alias("- ka") == ""
    assert _classify_blank_alias("a k") == ""
    assert _classify_blank_alias("か") == ""
    assert _classify_blank_alias("a ka") == ""
    # Mixed silence + real phone is NOT silence-only.
    assert _classify_blank_alias("ka R") == ""


def test_compute_row_order_violation_score_zero_for_perfectly_aligned_row():
    """If target_offset matches row_ratio*duration exactly, score is 0."""
    from core.coarse_crnn.oto_targets import _compute_row_order_violation_score

    s = _compute_row_order_violation_score(
        target_offset_ms=500.0, row_ratio=0.5, duration_ms=1000.0
    )
    assert abs(s) < 1e-9


def test_compute_row_order_violation_score_normalized_by_duration():
    """Same absolute delta produces smaller score on longer audio."""
    from core.coarse_crnn.oto_targets import _compute_row_order_violation_score

    short = _compute_row_order_violation_score(
        target_offset_ms=0.0, row_ratio=0.5, duration_ms=1000.0
    )
    long = _compute_row_order_violation_score(
        target_offset_ms=0.0, row_ratio=0.5, duration_ms=4000.0
    )
    # row_ratio*duration = 500 vs 2000, delta = 500 vs 2000.
    # Score = delta/duration = 0.5 vs 0.5 (same proportion!).
    assert abs(short - long) < 1e-9
    # But unequal absolute deltas distinguish them:
    a = _compute_row_order_violation_score(
        target_offset_ms=100.0, row_ratio=0.5, duration_ms=1000.0
    )
    b = _compute_row_order_violation_score(
        target_offset_ms=400.0, row_ratio=0.5, duration_ms=1000.0
    )
    # delta a = 400 (offset 100 vs ratio*dur 500) → 0.4
    # delta b = 100 (offset 400 vs ratio*dur 500) → 0.1
    assert a > b


def test_compute_row_order_violation_score_eps_protects_short_audio():
    """duration < eps_ms must not blow the score up to infinity."""
    from core.coarse_crnn.oto_targets import _compute_row_order_violation_score

    # Very short duration; delta = 50 ms, eps = 200 ms → score = 50/200 = 0.25.
    s = _compute_row_order_violation_score(
        target_offset_ms=50.0, row_ratio=0.0, duration_ms=10.0, eps_ms=200.0
    )
    assert abs(s - 0.25) < 1e-9


def test_compute_row_order_violation_score_zero_duration_returns_zero():
    """Defensive: 0 or negative duration returns 0 instead of dividing."""
    from core.coarse_crnn.oto_targets import _compute_row_order_violation_score

    assert _compute_row_order_violation_score(
        target_offset_ms=100.0, row_ratio=0.5, duration_ms=0.0
    ) == 0.0
    assert _compute_row_order_violation_score(
        target_offset_ms=100.0, row_ratio=0.5, duration_ms=-50.0
    ) == 0.0


def test_apply_row_context_attaches_violation_score():
    """The manifest pipeline must populate row_order_violation_score per row."""
    from core.coarse_crnn.oto_targets import _apply_row_context

    rows = [
        {
            "oto_path": "p1",
            "wav": "a.wav",
            "line_index": 0,
            "target_offset_ms": 100.0,
            "duration_ms": 1000.0,
        },
        {
            "oto_path": "p1",
            "wav": "a.wav",
            "line_index": 1,
            # Row 1 should sit at row_ratio=1.0 (count=2, idx=1) -> expected 1000.
            # Real offset 200 ms -> very high violation.
            "target_offset_ms": 200.0,
            "duration_ms": 1000.0,
        },
    ]
    _apply_row_context(rows)
    assert "row_order_violation_score" in rows[0]
    assert "row_order_violation_score" in rows[1]
    # Row 0: row_ratio=0, expected=0, actual=100 → score = 100/1000 = 0.1.
    assert abs(rows[0]["row_order_violation_score"] - 0.1) < 1e-9
    # Row 1: row_ratio=1, expected=1000, actual=200 → score = 800/1000 = 0.8.
    assert abs(rows[1]["row_order_violation_score"] - 0.8) < 1e-9
