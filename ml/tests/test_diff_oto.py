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
    from scripts.dev.diff_oto import alias_family, build_report

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
    assert alias_family("a \u3042A3") == "vv"
    assert alias_family("i bA3") == "vc"
    assert report["by_alias_family"]["cv_head"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert report["by_alias_family"]["vv"]["by_param"]["offset"]["MAE_ms"] == 20.0
    assert report["by_alias_family"]["vc"]["by_param"]["offset"]["MAE_ms"] == 30.0
    assert report["by_alias_family"]["terminal_v_dash"]["by_param"]["offset"]["MAE_ms"] == 10.0
    assert [row["alias_family"] for row in rows] == ["cv_head", "terminal_v_dash", "vc", "vv"]
