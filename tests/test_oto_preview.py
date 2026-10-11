from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.oto_preview import (
    PreviewAudio,
    WaveformViewport,
    load_preview_audio,
    load_row_statuses,
    occurrence_keys,
    oto_markers,
    parse_oto_rows,
    ppm_bytes,
    row_viewport,
    spectrogram_rgb,
)

OTO = (
    "a.wav=- ka,100,80,-300,60,20\n"
    "a.wav=a ki,500,90,-280,70,25\n"
    "b.wav=- ka,50,60,200,40,10\n"
    "b.wav=- ka,900,60,-200,40,10\n"
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_oto_rows_reads_parameters_and_line_numbers(tmp_path: Path) -> None:
    rows = parse_oto_rows(_write(tmp_path / "oto.ini", "# comment\n" + OTO + "c.wav=bad,x\n"))
    assert [row.line_number for row in rows] == [2, 3, 4, 5, 6]
    assert rows[0].alias == "- ka" and rows[0].offset == 100 and rows[0].cutoff == -300
    assert rows[0].preutterance == 60 and rows[0].overlap == 20
    assert rows[-1].offset == 0.0 and rows[-1].overlap == 0.0  # malformed numbers do not stop the preview


def test_markers_handle_negative_and_positive_cutoff(tmp_path: Path) -> None:
    rows = parse_oto_rows(_write(tmp_path / "oto.ini", OTO))
    negative = oto_markers(rows[0], 2000.0)
    assert (negative.offset, negative.fixed_end, negative.cutoff_end) == (100, 180, 400)
    assert (negative.preutterance, negative.overlap) == (160, 120)
    assert oto_markers(rows[2], 2000.0).cutoff_end == 1800  # positive cutoff counts from the WAV end


def test_row_viewport_covers_the_row_with_context(tmp_path: Path) -> None:
    rows = parse_oto_rows(_write(tmp_path / "oto.ini", OTO))
    viewport = row_viewport(rows[1], 5000.0)
    markers = oto_markers(rows[1], 5000.0)
    assert viewport.start_ms <= markers.offset and viewport.end_ms >= markers.cutoff_end
    assert viewport.span_ms < 5000.0


def test_occurrence_keys_count_duplicate_aliases(tmp_path: Path) -> None:
    rows = parse_oto_rows(_write(tmp_path / "oto.ini", OTO))
    assert occurrence_keys(rows)[2:] == [("b.wav", "- ka", 0), ("b.wav", "- ka", 1)]


def test_statuses_from_split_files_and_row_report(tmp_path: Path) -> None:
    oto = _write(tmp_path / "oto.ini", OTO)
    lines = OTO.splitlines(keepends=True)
    _write(tmp_path / "oto.clean.ini", lines[0] + lines[2])
    _write(tmp_path / "oto.review.ini", lines[1] + lines[3])
    report = [
        {"wav": "a.wav", "alias": "- ka", "split": "clean", "confidence": 0.95, "reasons": ["row_risk_low"],
         "output_oto": str(oto)},
        {"wav": "a.wav", "alias": "a ki", "split": "review", "confidence": 0.3,
         "reasons": ["hsmm_low_local_margin", "row_risk_high"], "output_oto": str(oto)},
        {"wav": "b.wav", "alias": "- ka", "split": "clean", "confidence": 0.9, "reasons": [], "output_oto": str(oto)},
        {"wav": "b.wav", "alias": "- ka", "split": "review", "confidence": 0.6, "reasons": ["row_risk_high"],
         "output_oto": str(oto)},
    ]
    _write(tmp_path / "row_report.jsonl", "\n".join(json.dumps(item) for item in report))
    statuses = load_row_statuses(oto, parse_oto_rows(oto))
    assert [status.split for status in statuses] == ["clean", "review", "clean", "review"]
    assert statuses[0].risk == pytest.approx(0.05)
    assert statuses[1].risk == pytest.approx(0.7) and statuses[1].color_key == "high_risk"
    assert statuses[2].risk is None and statuses[2].confidence == pytest.approx(0.9)  # not scored by row_risk
    assert statuses[3].color_key == "review"  # risk 0.4 stays below the red threshold

    review = parse_oto_rows(tmp_path / "oto.review.ini")
    review_statuses = load_row_statuses(tmp_path / "oto.review.ini", review)
    assert [status.split for status in review_statuses] == ["review", "review"]
    assert review_statuses[0].risk == pytest.approx(0.7) and review_statuses[1].risk == pytest.approx(0.4)


def test_statuses_ignore_stale_split_files(tmp_path: Path) -> None:
    oto = _write(tmp_path / "oto.ini", OTO)
    _write(tmp_path / "oto.clean.ini", OTO.splitlines(keepends=True)[0])
    _write(tmp_path / "oto.review.ini", "x.wav=- ka,1,1,-1,1,1\n")
    assert {status.split for status in load_row_statuses(oto, parse_oto_rows(oto))} == {"unknown"}


def test_spectrogram_image_shape_and_ppm(tmp_path: Path) -> None:
    rate = 22050
    t = np.arange(rate) / rate
    sf.write(str(tmp_path / "tone.wav"), 0.5 * np.sin(2 * np.pi * 1000 * t), rate)
    audio = load_preview_audio(tmp_path / "tone.wav")
    assert isinstance(audio, PreviewAudio) and audio.duration_ms == pytest.approx(1000.0)
    rgb = spectrogram_rgb(audio, WaveformViewport(audio.duration_ms), 200, 100)
    assert rgb.shape == (100, 200, 3) and rgb.dtype == np.uint8
    brightness = rgb.astype(int).sum(axis=(1, 2))
    tone_row = int(round((1 - 1000 / 8000) * 100))  # 1 kHz of an 8 kHz-high image, low at the bottom
    assert abs(int(np.argmax(brightness)) - tone_row) <= 3
    data = ppm_bytes(rgb)
    assert data.startswith(b"P6 200 100 255\n") and len(data) == len(b"P6 200 100 255\n") + 200 * 100 * 3


def test_status_summary_counts_and_missing_split_note() -> None:
    from universal_auto_oto.oto_preview import RowStatus, status_summary

    text = status_summary([RowStatus("clean"), RowStatus("review", 0.7), RowStatus("review", 0.2)])
    assert text.startswith("3행  clean 1  review 2") and "1)" in text
    assert "위험도 표시 없음" in status_summary([RowStatus("unknown")])


def test_preview_window_reloads_in_place(tmp_path: Path) -> None:
    tk = pytest.importorskip("tkinter")
    from universal_auto_oto.oto_preview import launch_oto_preview

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    rate = 8000
    for name in ("a.wav", "b.wav"):
        sf.write(str(tmp_path / name), 0.1 * np.random.default_rng(0).standard_normal(2 * rate), rate)
    oto = _write(tmp_path / "oto.ini", OTO)
    try:
        window = launch_oto_preview(oto, [tmp_path], master=root)
        tree = next(child for child in _descendants(window) if child.winfo_class() == "Treeview")
        assert window.cget("background").lower() == "#1f232b"
        tree.selection_set("2")
        window.update()
        window.load_oto(oto, [tmp_path])  # same file regenerated: keep the row (line 3)
        window.update()
        assert tree.selection() == ("2",)
        other = _write(tmp_path / "other.ini", "b.wav=x,10,10,-100,5,2\n")
        window.load_oto(other, [tmp_path])
        window.update()
        assert tree.get_children() == ("0",) and tree.selection() == ("0",)
        assert "other.ini" in window.title()
    finally:
        root.destroy()


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def test_all_review_without_risk_model_is_unrated(tmp_path: Path) -> None:
    from universal_auto_oto.oto_preview import status_summary

    oto = _write(tmp_path / "oto.ini", OTO)
    _write(tmp_path / "oto.clean.ini", "\n")
    _write(tmp_path / "oto.review.ini", OTO)
    statuses = load_row_statuses(oto, parse_oto_rows(oto))
    assert {status.split for status in statuses} == {"unrated"}
    assert {status.label for status in statuses} == {"미판정"}
    assert "모두 미판정" in status_summary(statuses)
    review = tmp_path / "oto.review.ini"  # opening the review file itself says the same
    assert {status.split for status in load_row_statuses(review, parse_oto_rows(review))} == {"unrated"}


def test_row_report_marks_uncalibrated_rows_unrated(tmp_path: Path) -> None:
    oto = _write(tmp_path / "oto.ini", OTO)
    lines = OTO.splitlines(keepends=True)
    _write(tmp_path / "oto.clean.ini", lines[0])
    _write(tmp_path / "oto.review.ini", "".join(lines[1:]))
    report = [
        {"wav": "a.wav", "alias": "- ka", "split": "clean", "confidence": 0.9, "reasons": ["row_risk_low"]},
        {"wav": "a.wav", "alias": "a ki", "split": "review", "confidence": 0.5,
         "reasons": ["uncalibrated_confidence"]},
        {"wav": "b.wav", "alias": "- ka", "split": "review", "confidence": 0.4, "reasons": ["row_risk_high"]},
        {"wav": "b.wav", "alias": "- ka", "split": "review", "confidence": 0.5, "reasons": ["low_confidence"]},
    ]
    _write(tmp_path / "row_report.jsonl", "\n".join(json.dumps(item) for item in report))
    splits = [status.split for status in load_row_statuses(oto, parse_oto_rows(oto))]
    assert splits == ["clean", "unrated", "review", "review"]
