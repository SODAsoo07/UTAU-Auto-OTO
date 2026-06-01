from __future__ import annotations

import json


def test_boundary_oto_candidate_matrix_loads_and_validates_cases(tmp_path):
    from scripts.evaluate.run_boundary_oto_candidate_matrix import load_cases, validate_cases

    (tmp_path / "bank").mkdir()
    (tmp_path / "gold.ini").write_text("sample.wav=ka,0,0,0,0,0\n", encoding="utf-8")
    (tmp_path / "pred.ini").write_text("sample.wav=ka,0,0,0,0,0\n", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "gold_cvvc",
                        "gold": "gold.ini",
                        "pred": "pred.ini",
                        "wav_dir": "bank",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(str(cases_path))

    assert cases[0].case_id == "gold_cvvc"
    assert cases[0].language == "japanese"
    assert cases[0].format_type == "cvvc"
    assert validate_cases(cases, repo_root=tmp_path) == []


def test_boundary_oto_candidate_matrix_default_configs_are_conservative():
    from scripts.evaluate.run_boundary_oto_candidate_matrix import default_configs

    safety = default_configs("safety_sweep")
    default_only = default_configs("default_only")

    assert [item.name for item in default_only] == ["cv_vc_max60"]
    assert [item.name for item in safety] == [
        "cv_vc_max60",
        "cv_only_max60",
        "vc_only_max60",
        "cv_vc_max80",
        "cv_vc_max60_margin05",
    ]
    assert safety[0].args == ()
    assert safety[1].args == ("--role-labels", "cv:vowel_onset")


def test_boundary_oto_candidate_matrix_builds_single_case_command(tmp_path):
    from scripts.evaluate.run_boundary_oto_candidate_matrix import MatrixCase, MatrixConfig, build_apply_command

    case = MatrixCase(case_id="Case A", gold="gold.ini", pred="pred.ini", wav_dir="wav")
    config = MatrixConfig("cv only", ("--role-labels", "cv:vowel_onset"))

    cmd, report_path, adjusted_path = build_apply_command(
        python_exe="python",
        case=case,
        config=config,
        model="model.pt",
        out_dir=tmp_path,
        device="cpu",
        cutoff_end_mode="utau",
        disable_acoustic_heuristics=True,
        heuristic_weight=0.25,
    )

    assert cmd[:2] == ["python", "scripts/evaluate/apply_boundary_oto_candidates.py"]
    assert "--disable-acoustic-heuristics" in cmd
    assert cmd[cmd.index("--heuristic-weight") + 1] == "0.25"
    assert cmd[-2:] == ["--role-labels", "cv:vowel_onset"]
    assert report_path == tmp_path / "cv_only" / "Case_A" / "report.json"
    assert adjusted_path == tmp_path / "cv_only" / "Case_A" / "adjusted.ini"


def test_boundary_oto_candidate_matrix_summarizes_selection_hints():
    from scripts.evaluate.run_boundary_oto_candidate_matrix import MatrixCase, MatrixConfig, summarize_matrix

    cases = [
        MatrixCase(case_id="A", gold="gold.ini", pred="pred.ini", wav_dir="wav"),
        MatrixCase(case_id="B", gold="gold.ini", pred="pred.ini", wav_dir="wav"),
    ]
    configs = [
        MatrixConfig("safe", ()),
        MatrixConfig("tail_risk", ("--max-shift-ms", "80")),
    ]
    report_map = {
        ("safe", "A"): _fake_report(mae_delta=-2.0, p90_delta=-1.0, cv_mae_delta=-1.0),
        ("safe", "B"): _fake_report(mae_delta=0.0, p90_delta=0.0, cv_mae_delta=0.0),
        ("tail_risk", "A"): _fake_report(mae_delta=-3.0, p90_delta=4.0, cv_mae_delta=-2.0),
        ("tail_risk", "B"): _fake_report(mae_delta=-1.0, p90_delta=-1.0, cv_mae_delta=-1.0),
    }

    summary = summarize_matrix(
        cases=cases,
        configs=configs,
        report_map=report_map,
        cases_path="cases.json",
        out_dir="out",
        model="model.pt",
    )

    assert summary["configs"][0]["mean_mae_delta"] == -1.0
    assert summary["configs"][0]["worst_p90_delta"] == 0.0
    assert summary["configs"][0]["by_family"]["cv"]["nonworse_cases"] == 2
    assert summary["selection_hints"]["safe_overall"] == ["safe", "tail_risk"]
    assert summary["selection_hints"]["safe_p90"] == ["safe"]
    assert summary["selection_hints"]["safe_family_overall"]["cv"] == ["safe", "tail_risk"]
    assert summary["selection_hints"]["safe_family_p90"]["cv"] == ["safe"]
    assert summary["selection_hints"]["ranked_p90_aware"][0]["config"] == "safe"
    assert summary["selection_hints"]["ranked_p90_aware"][0]["p90_aware_score"] < summary["selection_hints"]["ranked_p90_aware"][1]["p90_aware_score"]


def _fake_report(*, mae_delta: float, p90_delta: float, cv_mae_delta: float) -> dict:
    return {
        "summary": {
            "adjusted_rows": 3,
            "adjusted_by_family_label": {"cv": {"vowel_onset": 3}},
        },
        "score_maps": {"requested_wavs": 2, "loaded_wavs": 2},
        "diff": {
            "after": {"rows_matched": 10},
            "anchor_abs": {
                "delta": {
                    "pre_abs": {
                        "MAE_ms_delta": mae_delta,
                        "abs_median_ms_delta": 0.0,
                        "abs_p90_ms_delta": p90_delta,
                    },
                    "by_alias_family": {
                        "cv": {
                            "pre_abs": {
                                "MAE_ms_delta": cv_mae_delta,
                                "abs_p90_ms_delta": p90_delta,
                            }
                        }
                    },
                }
            },
        },
    }
