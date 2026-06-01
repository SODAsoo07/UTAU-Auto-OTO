from __future__ import annotations

import pytest

from core.generation.ja import ja_oto_file_consistency as ja_vc_consistency
from core.generation.ja.ja_oto_file_consistency import apply_ja_vc_neighbor_to_oto_file
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def _identity_validate(offset, consonant, cutoff, preutterance, overlap, *, alias_type=""):
    return offset, consonant, cutoff, preutterance, overlap


def _row_by_alias(path, alias: str) -> dict[str, object]:
    for line in read_text_with_fallback(str(path)).splitlines():
        row = parse_oto_line(line)
        if row and row.get("alias") == alias:
            return row
    raise AssertionError(f"alias not found: {alias}")


def test_japanese_vc_neighbor_uses_filename_row_plan_when_timing_is_crossed(tmp_path, monkeypatch):
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_ORDER", "row_plan")
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_BLEND", "0.35")
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_MAX_SHIFT", "45")
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_LEAD_MS", "6")
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_TAIL_MS", "8")
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_MIN_LEN", "35")

    oto_path = tmp_path / "oto.ini"
    oto_path.write_text(
        "\n".join(
            [
                "ka_ki.wav=ka,100.00,180.00,-150.00,70.00,25.00",
                "ka_ki.wav=a k,410.00,120.00,-120.00,80.00,50.00",
                "ka_ki.wav=ki,300.00,180.00,-400.00,70.00,25.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = apply_ja_vc_neighbor_to_oto_file(str(oto_path), validate_fn=_identity_validate)
    vc = _row_by_alias(oto_path, "a k")

    assert stats["row_plan_ordered_wavs"] == 1
    assert stats["total_changed"] == 1
    assert float(vc["offset"]) == pytest.approx(276.0, abs=0.01)
    assert float(vc["offset"]) < 300.0


def test_japanese_vc_neighbor_can_keep_legacy_time_order(tmp_path, monkeypatch):
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_ORDER", "time")

    oto_path = tmp_path / "oto.ini"
    oto_path.write_text(
        "\n".join(
            [
                "ka_ki.wav=ka,100.00,180.00,-150.00,70.00,25.00",
                "ka_ki.wav=a k,410.00,120.00,-120.00,80.00,50.00",
                "ka_ki.wav=ki,300.00,180.00,-400.00,70.00,25.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = apply_ja_vc_neighbor_to_oto_file(str(oto_path), validate_fn=_identity_validate)
    vc = _row_by_alias(oto_path, "a k")

    assert stats["time_ordered_wavs"] == 1
    assert stats["total_changed"] == 0
    assert float(vc["offset"]) == pytest.approx(410.0, abs=0.01)


def test_japanese_vc_neighbor_rejects_implausible_row_plan_alias_match(monkeypatch):
    monkeypatch.setenv("UTOA_JA_VC_NEIGHBOR_ORDER", "row_plan")
    rows = [
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "a s", "offset": 885.0, "cons": 120.0, "cutoff": -146.0, "pre": 80.0, "ovl": 45.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "i s", "offset": 1736.0, "cons": 210.0, "cutoff": -236.0, "pre": 170.0, "ovl": 135.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "u s", "offset": 2285.0, "cons": 210.0, "cutoff": -236.0, "pre": 170.0, "ovl": 135.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "e s", "offset": 2825.0, "cons": 210.0, "cutoff": -236.0, "pre": 170.0, "ovl": 135.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "o s", "offset": 3676.0, "cons": 135.0, "cutoff": -161.0, "pre": 95.0, "ovl": 60.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "n s", "offset": 825.0, "cons": 185.0, "cutoff": -205.0, "pre": 135.0, "ovl": 50.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "si", "offset": 985.0, "cons": 190.0, "cutoff": -358.0, "pre": 150.0, "ovl": 115.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "su", "offset": 1420.0, "cons": 190.0, "cutoff": -400.0, "pre": 150.0, "ovl": 115.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "se", "offset": 2030.0, "cons": 190.0, "cutoff": -400.0, "pre": 150.0, "ovl": 115.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "so", "offset": 2560.0, "cons": 190.0, "cutoff": -380.0, "pre": 150.0, "ovl": 115.0},
        {"wav": "sa-si-su-se-so-sa-N-sa.wav", "alias": "sa", "offset": 3310.0, "cons": 190.0, "cutoff": -400.0, "pre": 150.0, "ovl": 115.0},
    ]

    ordered, source = ja_vc_consistency._ordered_rows_for_vc_neighbor("sa-si-su-se-so-sa-N-sa.wav", rows)

    assert source == "time_fallback"
    assert [row["alias"] for row in ordered[:3]] == ["n s", "a s", "si"]
