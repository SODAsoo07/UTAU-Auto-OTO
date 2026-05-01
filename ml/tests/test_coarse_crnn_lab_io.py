from __future__ import annotations

from core.coarse_crnn.lab_io import frame_targets_from_segments, parse_timed_lab
from core.coarse_crnn.labels import LABEL_TO_ID


def test_parse_sinsy_100ns_lab_to_seconds(tmp_path):
    path = tmp_path / "sample.lab"
    path.write_text("0 1000000 SP\n1000000 2000000 a\n", encoding="utf-8")

    rows = parse_timed_lab(str(path))

    assert len(rows) == 2
    assert abs(rows[0].end - 0.1) < 1e-6
    assert rows[1].label == "a"


def test_frame_targets_from_segments_maps_to_coarse(tmp_path):
    path = tmp_path / "sample.lab"
    path.write_text("0 1000000 SP\n1000000 2000000 a\n", encoding="utf-8")
    rows = parse_timed_lab(str(path))

    targets = frame_targets_from_segments(
        rows,
        frame_count=20,
        hop_sec=0.01,
        duration_sec=0.2,
        language="japanese",
    )

    assert targets[0] == LABEL_TO_ID["SIL"]
    assert targets[-1] == LABEL_TO_ID["V"]
