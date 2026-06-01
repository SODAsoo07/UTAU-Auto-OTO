from __future__ import annotations

from core.phoneme_boundary.types import BoundaryFrameScores


def _score_map(*, label: str, peak_ms: float, peak_score: float = 0.9, current_ms: float = 0.0) -> BoundaryFrameScores:
    times = [float(i * 10) for i in range(41)]
    values = [0.05 for _ in times]
    for idx, time_ms in enumerate(times):
        if abs(time_ms - float(current_ms)) <= 0.1:
            values[idx] = max(values[idx], 0.2)
        if abs(time_ms - float(peak_ms)) <= 0.1:
            values[idx] = max(values[idx], float(peak_score))
    scores = {
        "phone_start": [0.0 for _ in times],
        "phone_end": [0.0 for _ in times],
        "consonant_onset": [0.0 for _ in times],
        "vowel_onset": [0.0 for _ in times],
        "vowel_nucleus": [0.0 for _ in times],
        "vowel_end": [0.0 for _ in times],
        "silence_boundary": [0.0 for _ in times],
    }
    scores[label] = values
    return BoundaryFrameScores(wav_path="sample.wav", times_ms=times, scores=scores, quality_scores=[0.9 for _ in times])


def _score_map_multi(peaks: dict[str, float], *, current_scores: dict[str, tuple[float, float]] | None = None) -> BoundaryFrameScores:
    times = [float(i * 10) for i in range(61)]
    scores = {
        "phone_start": [0.0 for _ in times],
        "phone_end": [0.0 for _ in times],
        "consonant_onset": [0.0 for _ in times],
        "vowel_onset": [0.0 for _ in times],
        "vowel_nucleus": [0.0 for _ in times],
        "vowel_end": [0.0 for _ in times],
        "silence_boundary": [0.0 for _ in times],
    }
    for label, peak_ms in peaks.items():
        values = [0.05 for _ in times]
        for idx, time_ms in enumerate(times):
            if abs(time_ms - float(peak_ms)) <= 0.1:
                values[idx] = 0.9
        scores[label] = values
    for label, item in (current_scores or {}).items():
        current_ms, value = item
        for idx, time_ms in enumerate(times):
            if abs(time_ms - float(current_ms)) <= 0.1:
                scores[label][idx] = max(scores[label][idx], float(value))
    return BoundaryFrameScores(wav_path="sample.wav", times_ms=times, scores=scores, quality_scores=[0.9 for _ in times])


def test_boundary_oto_candidates_adjust_cv_offset_to_vowel_onset_peak():
    from scripts.evaluate.apply_boundary_oto_candidates import (
        CandidateConfig,
        DEFAULT_ROLE_LABELS,
        apply_boundary_candidates,
        read_oto_lines,
    )

    rows = read_oto_lines_from_text("sample.wav=- k,100,150,-250,40,20\n")
    score_map = _score_map(label="vowel_onset", current_ms=140.0, peak_ms=160.0)
    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": score_map},
        CandidateConfig(
            role_labels={**DEFAULT_ROLE_LABELS, "cv_head": "vowel_onset"},
            min_score_margin=0.05,
            skip_edge_peaks=False,
        ),
    )

    assert out_lines == ["sample.wav=- k,120,150,-250,40,20"]
    assert report["summary"]["adjusted_rows"] == 1
    assert report["summary"]["adjusted_by_family_label"]["cv_head"]["vowel_onset"] == 1
    assert report["decisions"][0]["shift_ms"] == 20.0


def test_boundary_oto_candidates_default_skips_cv_head_until_gold_gate_proves_safe():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, DEFAULT_ROLE_LABELS, apply_boundary_candidates

    rows = read_oto_lines_from_text("sample.wav=- k,100,150,-250,40,20\n")
    score_map = _score_map(label="vowel_onset", current_ms=140.0, peak_ms=160.0)
    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": score_map},
        CandidateConfig(role_labels=DEFAULT_ROLE_LABELS, min_score_margin=0.05, skip_edge_peaks=False),
    )

    assert out_lines == ["sample.wav=- k,100,150,-250,40,20"]
    assert report["summary"]["adjusted_rows"] == 0
    assert report["summary"]["skipped_by_reason"]["role_not_enabled"] == 1


def test_boundary_oto_candidates_adjust_vc_offset_to_vowel_end_peak():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, DEFAULT_ROLE_LABELS, apply_boundary_candidates

    rows = read_oto_lines_from_text("sample.wav=a r,200,180,-300,80,30\n")
    score_map = _score_map(label="vowel_end", current_ms=280.0, peak_ms=250.0)
    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": score_map},
        CandidateConfig(role_labels=DEFAULT_ROLE_LABELS, min_score_margin=0.05, skip_edge_peaks=False),
    )

    assert out_lines == ["sample.wav=a r,170,180,-300,80,30"]
    assert report["summary"]["adjusted_rows"] == 1
    assert report["summary"]["adjusted_by_family_label"]["vc"]["vowel_end"] == 1
    assert report["decisions"][0]["shift_ms"] == -30.0


def test_boundary_oto_candidates_skip_disabled_roles_and_weak_margins():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, DEFAULT_ROLE_LABELS, apply_boundary_candidates

    rows = read_oto_lines_from_text(
        "\n".join(
            [
                "sample.wav=V -,100,150,-250,40,20",
                "sample.wav=ka,100,150,-250,40,20",
            ]
        )
        + "\n"
    )
    weak_score_map = _score_map(label="vowel_onset", current_ms=140.0, peak_ms=160.0, peak_score=0.22)
    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": weak_score_map},
        CandidateConfig(role_labels=DEFAULT_ROLE_LABELS, min_score_margin=0.1, skip_edge_peaks=False),
    )

    assert out_lines[0] == "sample.wav=V -,100,150,-250,40,20"
    assert out_lines[1] == "sample.wav=ka,100,150,-250,40,20"
    assert report["summary"]["adjusted_rows"] == 0
    assert report["summary"]["skipped_by_reason"]["role_not_enabled"] == 1
    assert report["summary"]["skipped_by_reason"]["score_margin_too_small"] == 1


def test_boundary_oto_candidates_gate_large_and_negative_offset_shifts():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, DEFAULT_ROLE_LABELS, apply_boundary_candidates

    rows = read_oto_lines_from_text(
        "\n".join(
            [
                "sample.wav=ka,220,150,-250,40,20",
                "sample.wav=ta,10,150,-250,40,20",
            ]
        )
        + "\n"
    )
    score_map = _score_map(label="vowel_onset", current_ms=260.0, peak_ms=150.0)
    score_map2 = _score_map(label="vowel_onset", current_ms=50.0, peak_ms=20.0)
    out_lines, report = apply_boundary_candidates(
        rows,
        {
            "sample.wav": score_map,
        },
        CandidateConfig(role_labels=DEFAULT_ROLE_LABELS, max_shift_ms=80.0, min_new_offset_ms=0.0, skip_edge_peaks=False),
    )

    assert out_lines[0] == "sample.wav=ka,220,150,-250,40,20"
    assert report["decisions"][0]["reason"] == "shift_too_large"

    out_lines2, report2 = apply_boundary_candidates(
        read_oto_lines_from_text("other.wav=ta,10,150,-250,40,20\n"),
        {"other.wav": score_map2},
        CandidateConfig(role_labels=DEFAULT_ROLE_LABELS, min_new_offset_ms=0.0, skip_edge_peaks=False),
    )
    assert out_lines2 == ["other.wav=ta,10,150,-250,40,20"]
    assert report2["decisions"][0]["reason"] == "new_offset_below_min"


def test_boundary_oto_candidates_support_role_specific_shift_limits():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, apply_boundary_candidates

    rows = read_oto_lines_from_text(
        "\n".join(
            [
                "sample.wav=ka,100,150,-250,40,20",
                "sample.wav=a r,200,180,-300,80,30",
            ]
        )
        + "\n"
    )
    score_map = _score_map_multi(
        {"vowel_onset": 200.0, "vowel_end": 340.0},
        current_scores={"vowel_onset": (140.0, 0.2), "vowel_end": (280.0, 0.2)},
    )

    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": score_map},
        CandidateConfig(
            role_labels={"cv": "vowel_onset", "vc": "vowel_end"},
            max_shift_ms=80.0,
            role_max_shift_ms={"cv": 60.0, "vc": 50.0},
            skip_edge_peaks=False,
        ),
    )

    assert out_lines[0] == "sample.wav=ka,160,150,-250,40,20"
    assert out_lines[1] == "sample.wav=a r,200,180,-300,80,30"
    assert report["decisions"][0]["action"] == "adjust"
    assert report["decisions"][1]["reason"] == "shift_too_large"


def test_boundary_oto_candidates_neighbor_guard_blocks_vc_crossing_next_cv():
    from scripts.evaluate.apply_boundary_oto_candidates import CandidateConfig, apply_boundary_candidates

    rows = read_oto_lines_from_text(
        "\n".join(
            [
                "sample.wav=ka,80,150,-250,40,20",
                "sample.wav=a r,200,180,-300,80,30",
                "sample.wav=ra,300,150,-250,40,20",
            ]
        )
        + "\n"
    )

    crossing_score = _score_map(label="vowel_end", current_ms=280.0, peak_ms=340.0)
    out_lines, report = apply_boundary_candidates(
        rows,
        {"sample.wav": crossing_score},
        CandidateConfig(
            role_labels={"vc": "vowel_end"},
            max_shift_ms=80.0,
            neighbor_guard_enabled=True,
            skip_edge_peaks=False,
        ),
    )

    assert out_lines[1] == "sample.wav=a r,200,180,-300,80,30"
    assert report["decisions"][1]["reason"] == "neighbor_guard_crosses_next"

    safe_score = _score_map(label="vowel_end", current_ms=280.0, peak_ms=310.0)
    safe_lines, safe_report = apply_boundary_candidates(
        rows,
        {"sample.wav": safe_score},
        CandidateConfig(
            role_labels={"vc": "vowel_end"},
            max_shift_ms=80.0,
            neighbor_guard_enabled=True,
            skip_edge_peaks=False,
        ),
    )
    assert safe_lines[1] == "sample.wav=a r,230,180,-300,80,30"
    assert safe_report["decisions"][1]["action"] == "adjust"


def test_no_mfa_boundary_candidate_cv_vc_guarded_preset(monkeypatch):
    from core.generation.common.no_mfa_oto_builder import _boundary_candidate_config_from_env

    monkeypatch.setenv("UTOA_NO_MFA_BOUNDARY_CANDIDATE_PRESET", "cv_vc_guarded")

    config = _boundary_candidate_config_from_env()

    assert config.role_labels == {"cv": "vowel_onset", "vc": "vowel_end"}
    assert config.role_max_shift_ms["cv"] == 60.0
    assert config.role_max_shift_ms["vc"] == 50.0
    assert config.role_min_score_margin["vc"] == 0.05
    assert config.neighbor_guard_enabled is True


def test_no_mfa_boundary_candidate_cv_vc_safe_preset_keeps_vc_no_guard(monkeypatch):
    from core.generation.common.no_mfa_oto_builder import _boundary_candidate_config_from_env

    monkeypatch.setenv("UTOA_NO_MFA_BOUNDARY_CANDIDATE_PRESET", "cv_vc_safe")

    config = _boundary_candidate_config_from_env()

    assert config.role_labels == {"cv": "vowel_onset", "vc": "vowel_end"}
    assert config.role_max_shift_ms["cv"] == 60.0
    assert config.role_max_shift_ms["vc"] == 50.0
    assert config.role_min_score_margin["vc"] == 0.08
    assert config.neighbor_guard_enabled is False


def test_no_mfa_boundary_candidate_cv_vc_p90_safe_preset_limits_cv_shift(monkeypatch):
    from core.generation.common.no_mfa_oto_builder import _boundary_candidate_config_from_env

    monkeypatch.setenv("UTOA_NO_MFA_BOUNDARY_CANDIDATE_PRESET", "cv_vc_p90_safe")

    config = _boundary_candidate_config_from_env()

    assert config.role_labels == {"cv": "vowel_onset", "vc": "vowel_end"}
    assert config.role_max_shift_ms["cv"] == 40.0
    assert config.role_max_shift_ms["vc"] == 50.0
    assert config.role_min_score_margin["vc"] == 0.08
    assert config.neighbor_guard_enabled is False


def test_boundary_oto_candidate_anchor_abs_report_tracks_offset_plus_pre(tmp_path):
    from scripts.evaluate.apply_boundary_oto_candidates import _anchor_abs_delta, _anchor_abs_report

    gold = tmp_path / "gold.ini"
    before = tmp_path / "before.ini"
    after = tmp_path / "after.ini"
    gold.write_text("sample.wav=- k,100,150,-250,40,20\n", encoding="utf-8")
    before.write_text("sample.wav=- k,120,150,-250,60,20\n", encoding="utf-8")
    after.write_text("sample.wav=- k,110,150,-250,40,20\n", encoding="utf-8")

    before_report = _anchor_abs_report(str(gold), str(before))
    after_report = _anchor_abs_report(str(gold), str(after))
    delta = _anchor_abs_delta(before_report, after_report)

    assert before_report["pre_abs"]["MAE_ms"] == 40.0
    assert after_report["pre_abs"]["MAE_ms"] == 10.0
    assert delta["pre_abs"]["MAE_ms_delta"] == -30.0
    assert delta["by_alias_family"]["cv_head"]["pre_abs"]["MAE_ms_delta"] == -30.0


def read_oto_lines_from_text(text: str):
    from scripts.evaluate.apply_boundary_oto_candidates import read_oto_lines

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "oto.ini"
        path.write_text(text, encoding="utf-8")
        return read_oto_lines(str(path))
