from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path


def _write_silence_wav(path, *, duration_ms: float = 1000.0, rate: int = 16000) -> None:
    frames = int(round(rate * float(duration_ms) / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_diff_oto_compatible_cutoff_end_handles_mixed_positive_tail_and_negative_relative(tmp_path):
    from scripts.dev.diff_oto import build_report

    _write_silence_wav(tmp_path / "a.wav", duration_ms=1000.0)
    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text("a.wav=ka,100,180,600,120,40\n", encoding="utf-8")
    pred.write_text("a.wav=ka,150,180,-250,120,40\n", encoding="utf-8")

    report, _rows = build_report(
        gold_path=str(gold),
        pred_path=str(pred),
        wav_dir=str(tmp_path),
        cutoff_end_mode="compatible",
    )

    assert report["by_param"]["cutoff"]["MAE_ms"] == 850.0
    assert report["by_param"]["cutoff_end_abs"]["MAE_ms"] == 0.0
    assert report["cutoff_end_mode_pairs"] == {
        "positive_right_blank_from_end->negative_relative_to_offset": 1,
    }


def test_diff_oto_utau_cutoff_end_handles_negative_right_blank(tmp_path):
    from scripts.dev.diff_oto import build_report

    _write_silence_wav(tmp_path / "a.wav", duration_ms=1000.0)
    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text("a.wav=ka,100,180,-250,120,40\n", encoding="utf-8")
    pred.write_text("a.wav=ka,100,180,650,120,40\n", encoding="utf-8")

    report, _rows = build_report(
        gold_path=str(gold),
        pred_path=str(pred),
        wav_dir=str(tmp_path),
        cutoff_end_mode="utau",
    )

    assert report["by_param"]["cutoff"]["MAE_ms"] == 900.0
    assert report["by_param"]["cutoff_end_abs"]["MAE_ms"] == 0.0
    assert report["cutoff_end_mode_pairs"] == {
        "negative_right_blank_from_end->positive_relative_to_offset": 1,
    }


def test_diff_oto_matches_duplicate_wav_alias_by_occurrence(tmp_path):
    from scripts.dev.diff_oto import build_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "repeat.wav=a ka,100,180,-300,120,40",
                "repeat.wav=a ka,900,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "repeat.wav=a ka,110,180,-300,120,40",
                "repeat.wav=a ka,910,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report, rows = build_report(
        gold_path=str(gold),
        pred_path=str(pred),
    )

    assert report["rows_gold"] == 2
    assert report["rows_pred"] == 2
    assert report["rows_matched"] == 2
    assert report["duplicate_key_groups_gold"] == 1
    assert report["duplicate_key_groups_pred"] == 1
    assert report["by_param"]["offset"]["MAE_ms"] == 10.0
    assert [row["occurrence"] for row in rows] == [0, 1]


def test_diff_oto_keeps_rows_with_blank_overlap_field(tmp_path):
    from scripts.dev.diff_oto import build_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text("ka.wav=\u304b,3387.76,129.71,-282.99,48.07,\n", encoding="utf-8")
    pred.write_text("ka.wav=\u304b,3397.76,129.71,-282.99,48.07,0\n", encoding="utf-8")

    report, rows = build_report(
        gold_path=str(gold),
        pred_path=str(pred),
    )

    assert report["rows_gold"] == 1
    assert report["rows_pred"] == 1
    assert report["rows_matched"] == 1
    assert report["by_param"]["offset"]["MAE_ms"] == 10.0
    assert rows[0]["gold"]["overlap"] == 0.0


def test_diff_oto_out_file_is_valid_json_with_per_row_preview(tmp_path):
    _write_silence_wav(tmp_path / "a.wav", duration_ms=1000.0)
    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    out = tmp_path / "report.json"
    gold.write_text("a.wav=ka,100,180,-300,120,40\n", encoding="utf-8")
    pred.write_text("a.wav=ka,110,180,-300,120,40\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/dev/diff_oto.py",
            "--gold",
            str(gold),
            "--pred",
            str(pred),
            "--wav-dir",
            str(tmp_path),
            "--cutoff-end-mode",
            "compatible",
            "--out",
            str(out),
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["rows_matched"] == 1
    assert payload["per_row_total"] == 1
    assert payload["per_row_truncated_count"] == 1
    assert payload["per_row_truncated"][0]["delta"]["offset"] == 10.0


def test_diff_oto_reports_alias_family_buckets(tmp_path):
    from core.generation.common.oto_alias_family import alias_family, alias_token_is_vowel
    from scripts.dev.diff_oto import build_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "head.wav=- s,100,120,-300,80,40",
                "vv.wav=a \u3042,200,120,-300,80,40",
                "vc.wav=a k,300,120,-300,80,40",
                "tail.wav=V -,900,120,-80,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "head.wav=- s,130,120,-300,80,40",
                "vv.wav=a \u3042,220,120,-300,80,40",
                "vc.wav=a k,330,120,-300,80,40",
                "tail.wav=V -,910,120,-80,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report, rows = build_report(gold_path=str(gold), pred_path=str(pred))

    assert alias_family("- \u3042") == "cv_head"
    assert alias_family("V -") == "terminal_v_dash"
    assert alias_family("\u3042A3") == "v"
    assert alias_family("\u3044BA3") == "v"
    assert alias_family("a \u3042A3") == "vv"
    assert alias_family("e \u3048BA3") == "vv"
    assert alias_family("i bA3") == "vc"
    assert alias_family("a tsBA3") == "vc"
    assert alias_family("o g_S") == "vc"
    assert alias_family("brC4S") == "policy_breath"
    assert alias_family("briC4S") == "cv"
    assert not alias_token_is_vowel("brC4S")
    assert report["by_alias_family"]["cv_head"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert report["by_alias_family"]["vv"]["by_param"]["offset"]["MAE_ms"] == 20.0
    assert report["by_alias_family"]["vc"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert report["by_alias_family"]["terminal_v_dash"]["by_param"]["offset"]["MAE_ms"] == 10.0
    assert [row["alias_family"] for row in rows] == ["cv_head", "terminal_v_dash", "vc", "vv"]


def test_stratified_oto_residuals_reports_vc_role_buckets(tmp_path):
    from scripts.evaluate.stratified_oto_residuals import build_stratified_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "r.wav=e r,1000,180,-300,120,40",
                "k.wav=a k,1000,180,-300,120,40",
                "gy.wav=o gy,1000,180,-300,120,40",
                "tail.wav=a -6,1000,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "r.wav=e r,620,180,-300,120,40",
                "k.wav=a k,1040,180,-300,120,40",
                "gy.wav=o gy,1280,180,-300,120,40",
                "tail.wav=a -6,1120,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stratified_report(gold_path=str(gold), pred_path=str(pred), cutoff_end_mode="raw")

    by_class = report["by_dimension"]["vc_right_class"]
    assert by_class["liquid"]["by_param"]["offset"]["MAE_ms"] == 380.0
    assert by_class["obstruent"]["by_param"]["offset"]["MAE_ms"] == 40.0
    assert by_class["yoon"]["by_param"]["offset"]["MAE_ms"] == 280.0
    assert by_class["terminal"]["by_param"]["offset"]["MAE_ms"] == 120.0
    assert report["by_dimension"]["vc_right_token"]["r"]["by_param"]["offset"]["early_count"] == 1
    assert report["by_dimension"]["vc_pair"]["o->gy"]["by_param"]["offset"]["late_count"] == 1
    assert any(
        bucket["dimension"] == "vc_pair" and bucket["bucket"] == "e->r"
        for bucket in report["priority_buckets"]
    )


def test_stratified_oto_residuals_strips_attached_suffix_for_vc_buckets(tmp_path):
    from scripts.evaluate.stratified_oto_residuals import build_stratified_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "f.wav=i fBA3,1000,180,-300,120,40",
                "n.wav=a nBA3,1000,180,-300,120,40",
                "ts.wav=a tsBA3,1000,180,-300,120,40",
                "ny.wav=n nyA3,1000,180,-300,120,40",
                "soft.wav=o g_S,1000,180,-300,120,40",
                "tail.wav=a -6,1000,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "f.wav=i fBA3,1020,180,-300,120,40",
                "n.wav=a nBA3,1030,180,-300,120,40",
                "ts.wav=a tsBA3,1040,180,-300,120,40",
                "ny.wav=n nyA3,1050,180,-300,120,40",
                "soft.wav=o g_S,1060,180,-300,120,40",
                "tail.wav=a -6,1070,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stratified_report(gold_path=str(gold), pred_path=str(pred), cutoff_end_mode="raw")

    by_token = report["by_dimension"]["vc_right_token"]
    by_pair = report["by_dimension"]["vc_pair"]
    assert by_token["f"]["by_param"]["offset"]["MAE_ms"] == 20.0
    assert by_token["n"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert by_token["ts"]["by_param"]["offset"]["MAE_ms"] == 40.0
    assert by_token["ny"]["by_param"]["offset"]["MAE_ms"] == 50.0
    assert by_token["g"]["by_param"]["offset"]["MAE_ms"] == 60.0
    assert by_token["-"]["by_param"]["offset"]["MAE_ms"] == 70.0
    assert "fba" not in by_token
    assert "nba" not in by_token
    assert "g_s" not in by_token
    assert by_pair["i->f"]["by_param"]["offset"]["MAE_ms"] == 20.0
    assert by_pair["a->n"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert by_pair["o->g"]["by_param"]["offset"]["MAE_ms"] == 60.0


def test_stratified_oto_residuals_cli_writes_json_and_csv(tmp_path):
    _write_silence_wav(tmp_path / "a.wav", duration_ms=1000.0)
    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    out = tmp_path / "stratified.json"
    csv_out = tmp_path / "stratified.csv"
    gold.write_text("a.wav=a r,100,180,-300,120,40\n", encoding="utf-8")
    pred.write_text("a.wav=a r,130,180,-300,120,40\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate/stratified_oto_residuals.py",
            "--gold",
            str(gold),
            "--pred",
            str(pred),
            "--wav-dir",
            str(tmp_path),
            "--cutoff-end-mode",
            "compatible",
            "--out",
            str(out),
            "--csv-out",
            str(csv_out),
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    csv_text = csv_out.read_text(encoding="utf-8")

    assert result.returncode == 0
    assert payload["by_dimension"]["vc_right_token"]["r"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert "vc_right_class,liquid,offset" in csv_text


def test_stratified_oto_residuals_can_separate_low_trust_gold_rows(tmp_path):
    from scripts.evaluate.stratified_oto_residuals import build_stratified_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "a.wav=a r,100,180,-300,120,40",
                "b.wav=ka,0,0,-300,0,0",
                "c.wav=ke,100,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "a.wav=a r,130,180,-300,120,40",
                "b.wav=ka,1250,180,-300,120,40",
                "c.wav=ke,1600,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stratified_report(
        gold_path=str(gold),
        pred_path=str(pred),
        wav_dir=str(tmp_path),
        cutoff_end_mode="raw",
        language="korean",
        trusted_only=True,
    )

    quality = report["quality_summary"]
    assert quality["rows_total"] == 3
    assert quality["trusted_rows"] == 1
    assert quality["excluded_rows"] == 2
    assert quality["by_exclusion_reason"] == {
        "korean_catastrophic_offset_residual": 1,
        "manual_zero_anchor_profile": 1,
    }
    assert quality["trusted_by_param"]["offset"]["MAE_ms"] == 30.0
    assert report["by_dimension"]["alias_family"]["vc"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert "cv" not in report["by_dimension"]["alias_family"]


def test_stratified_oto_residuals_reports_assisted_use_tiers(tmp_path):
    from scripts.evaluate.stratified_oto_residuals import build_stratified_report

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "ready.wav=a,100,180,-300,120,40",
                "review.wav=ka,100,180,-300,120,40",
                "manual.wav=ke,100,180,-300,120,40",
                "untrusted.wav=ko,0,0,-300,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "ready.wav=a,140,210,-360,150,55",
                "review.wav=ka,210,180,-380,120,40",
                "manual.wav=ke,310,180,-560,120,40",
                "untrusted.wav=ko,20,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stratified_report(
        gold_path=str(gold),
        pred_path=str(pred),
        cutoff_end_mode="raw",
        language="korean",
    )

    assisted = report["quality_summary"]["assisted_use"]
    assert assisted["by_tier"] == {
        "assisted_ready": 1,
        "manual_required": 2,
        "review_recommended": 1,
    }
    assert assisted["assisted_ready_rows"] == 1
    assert assisted["review_recommended_rows"] == 1
    assert assisted["manual_required_rows"] == 2
    assert assisted["assisted_ready_ratio"] == 0.25
    by_alias = {row["alias"]: row["quality"]["assisted_use"] for row in report["tagged_rows"]}
    assert by_alias["a"]["tier"] == "assisted_ready"
    assert by_alias["ka"]["tier"] == "review_recommended"
    assert by_alias["ke"]["tier"] == "manual_required"
    assert by_alias["ko"]["reasons"] == ["untrusted.manual_zero_anchor_profile"]


def test_build_oto_residual_dataset_rows_and_group_split(tmp_path):
    from scripts.evaluate.build_oto_residual_dataset import build_residual_rows, split_rows

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "a.wav=a r,100,180,-300,120,40",
                "b.wav=a n,200,180,-300,120,40",
                "c.wav=ka,300,180,-300,120,40",
                "d.wav=e r,400,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "a.wav=a r,130,180,-300,120,40",
                "b.wav=a n,160,180,-300,120,40",
                "c.wav=ka,315,180,-300,120,40",
                "d.wav=e r,450,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, summary = build_residual_rows(
        [
            {
                "case_id": "toy_cvvc",
                "language": "japanese",
                "format_type": "cvvc",
                "gold": str(gold),
                "pred": str(pred),
                "wav_dir": str(tmp_path),
            }
        ],
        cutoff_end_mode="raw",
    )
    by_alias = {row["alias"]: row for row in rows}

    assert summary["rows_total"] == 4
    assert by_alias["a r"]["delta_offset"] == -30.0
    assert by_alias["a r"]["delta_cutoff"] == -30.0
    assert by_alias["a r"]["vc_right_class"] == "liquid"
    assert by_alias["a r"]["alias_group"] == "vc:liquid"
    assert by_alias["a n"]["vc_right_class"] == "nasal"
    assert by_alias["a r"]["sample_weight"] > 1.0

    train, heldout, split_summary = split_rows(rows, heldout_ratio=0.5, seed=7)
    train_groups = {row["group_id"] for row in train}
    heldout_groups = {row["group_id"] for row in heldout}

    assert len(train) + len(heldout) == len(rows)
    assert train_groups
    assert heldout_groups
    assert train_groups.isdisjoint(heldout_groups)
    assert split_summary["groups_total"] == 4


def test_build_oto_residual_dataset_skips_zero_anchor_manual_rows_by_default(tmp_path):
    from scripts.evaluate.build_oto_residual_dataset import build_residual_rows

    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    gold.write_text(
        "\n".join(
            [
                "a.wav=ka,100,180,-300,120,40",
                "b.wav=ke,0,0,-300,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "a.wav=ka,110,180,-300,120,40",
                "b.wav=ke,620,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, summary = build_residual_rows(
        [
            {
                "case_id": "toy_ko_cvc",
                "language": "korean",
                "format_type": "cvc",
                "gold": str(gold),
                "pred": str(pred),
                "wav_dir": str(tmp_path),
            }
        ],
        cutoff_end_mode="raw",
    )
    by_alias = {row["alias"]: row for row in rows}

    assert by_alias["ka"]["train_keep_default"] == 1
    assert by_alias["ka"]["sample_weight"] > 0.0
    assert by_alias["ke"]["train_keep_default"] == 0
    assert by_alias["ke"]["train_skip_reason"] == "manual_zero_anchor_profile"
    assert by_alias["ke"]["sample_weight"] == 0.0
    assert summary["train_keep_default_rows"] == 1
    assert summary["train_skip_default_rows"] == 1
    assert summary["by_train_skip_reason"] == {"manual_zero_anchor_profile": 1}


def test_build_oto_residual_dataset_cli_writes_jsonl_and_lightgbm_csv(tmp_path):
    _write_silence_wav(tmp_path / "a.wav", duration_ms=1000.0)
    _write_silence_wav(tmp_path / "b.wav", duration_ms=1000.0)
    gold = tmp_path / "gold.ini"
    pred = tmp_path / "pred.ini"
    out_dir = tmp_path / "residual_out"
    gold.write_text(
        "\n".join(
            [
                "a.wav=a r,100,180,-300,120,40",
                "b.wav=ka,200,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "\n".join(
            [
                "a.wav=a r,130,180,-300,120,40",
                "b.wav=ka,190,180,-300,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate/build_oto_residual_dataset.py",
            "--case-id",
            "toy_cli",
            "--language",
            "japanese",
            "--format-type",
            "cvvc",
            "--gold",
            str(gold),
            "--pred",
            str(pred),
            "--wav-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--heldout-ratio",
            "0.5",
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    jsonl_lines = (out_dir / "residual_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    csv_text = (out_dir / "residual_dataset.csv").read_text(encoding="utf-8")

    assert result.returncode == 0
    assert summary["rows_total"] == 2
    assert summary["split"]["heldout_rows"] == 1
    assert len(jsonl_lines) == 2
    assert "delta_offset" in csv_text
    assert "base_offset" in csv_text


def test_evaluate_oto_lightgbm_residual_summarizes_external_predictions():
    from scripts.evaluate.evaluate_oto_lightgbm_residual import summarize_predictions

    rows = [
        {"case_id": "a", "alias_family": "vc", "vc_right_class": "liquid", "delta_offset": "100", "delta_cons": "40"},
        {"case_id": "a", "alias_family": "vc", "vc_right_class": "liquid", "delta_offset": "-80", "delta_cons": "-20"},
        {"case_id": "a", "alias_family": "cv", "vc_right_class": "", "delta_offset": "20", "delta_cons": "0"},
    ]
    preds = [
        {"delta_offset": 70.0, "delta_cons": -30.0},
        {"delta_offset": -40.0, "delta_cons": 40.0},
        {"delta_offset": 5.0, "delta_cons": 30.0},
    ]

    report = summarize_predictions(
        rows,
        preds,
        language="japanese",
        group_fields=["case_id", "alias_family", "vc_right_class"],
        min_group_n=2,
    )

    assert report["targets"]["delta_offset"]["baseline_mae_ms"] == 66.667
    assert report["targets"]["delta_offset"]["model_mae_ms"] == 28.333
    assert report["targets"]["delta_cons"]["model_mae_ms"] > report["targets"]["delta_cons"]["baseline_mae_ms"]
    assert report["failed_targets"] == ["delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"]
    assert report["recommended_use"] == "do_not_enable"
    assert report["by_group"]["alias_family"]["vc"]["n"] == 2


def test_evaluate_oto_lightgbm_residual_reports_runtime_safety_target_exclusions():
    from scripts.evaluate.evaluate_oto_lightgbm_residual import (
        apply_runtime_safety_target_exclusions,
        summarize_predictions,
    )

    def row(alias_family: str) -> dict[str, str]:
        return {
            "case_id": "case",
            "alias_family": alias_family,
            "vc_right_class": "",
            "delta_offset": "0",
            "delta_cons": "0",
            "delta_cutoff": "0",
            "delta_pre": "0",
            "delta_ovl": "0",
        }

    rows = [row("cv_head"), row("spaced"), row("v"), row("vc")]
    preds = [
        {"delta_offset": 10.0, "delta_cons": 20.0, "delta_cutoff": 30.0, "delta_pre": 40.0, "delta_ovl": 50.0},
        {"delta_offset": 10.0, "delta_cons": 20.0, "delta_cutoff": 30.0, "delta_pre": 40.0, "delta_ovl": 50.0},
        {"delta_offset": 10.0, "delta_cons": 20.0, "delta_cutoff": 30.0, "delta_pre": 40.0, "delta_ovl": 50.0},
        {"delta_offset": 10.0, "delta_cons": 20.0, "delta_cutoff": 30.0, "delta_pre": 40.0, "delta_ovl": 50.0},
    ]

    gated, safety_report = apply_runtime_safety_target_exclusions(rows, preds)
    report = summarize_predictions(rows, preds, language="japanese")

    assert gated[0]["delta_cutoff"] == 0.0
    assert gated[0]["delta_offset"] == 10.0
    assert gated[1]["delta_cons"] == 0.0
    assert gated[2]["delta_offset"] == 0.0
    assert gated[2]["delta_cutoff"] == 0.0
    assert gated[3]["delta_cutoff"] == 30.0
    assert safety_report["excluded_values_changed"] == 4
    assert report["runtime_safety_target_exclusions"]["excluded_values_changed"] == 4
    assert "alias_family:cv_head:delta_cutoff" in report["runtime_safety_target_exclusions"]["excluded_row_targets"]
