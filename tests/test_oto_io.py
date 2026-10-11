from pathlib import Path

from universal_auto_oto.oto_io import (
    oto_cutoff_end_abs,
    parse_oto,
    read_text_auto,
    resolve_output_encoding,
    validate_oto_line_values,
    write_lines_atomic,
)


def test_cp932_oto_round_trip(tmp_path: Path):
    path = tmp_path / "oto.ini"
    lines = ["あ.wav=あ,10.000,100.000,-20.000,60.000,20.000"]
    write_lines_atomic(path, lines, encoding="cp932")
    text, encoding, _newline = read_text_auto(path)
    rows, parsed_encoding, _ = parse_oto(path)
    assert text.strip() == lines[0]
    assert encoding == "cp932"
    assert parsed_encoding == "cp932"
    assert rows[0].alias == "あ"


def test_read_cp949_oto(tmp_path: Path):
    path = tmp_path / "oto.ini"
    path.write_bytes("가.wav=가,10,100,-200,60,20\r\n".encode("cp949"))
    text, encoding, newline = read_text_auto(path)
    assert text.startswith("가.wav=가")
    assert encoding == "cp949"
    assert newline == "\r\n"


def test_cp932_kana_alias_block_wins_over_cp949_mojibake(tmp_path: Path):
    path = tmp_path / "oto.ini"
    kana_aliases = "ばびぶべぼぱぴぷぺぽだぢづでどがぎぐげご息"
    lines = [f"sample.wav={alias},10,100,-200,60,20" for alias in kana_aliases]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))

    text, encoding, _newline = read_text_auto(path)

    assert encoding == "cp932"
    assert "ば" in text
    assert "궽" not in text


def test_output_encoding_preserves_korean_legacy_codec():
    assert resolve_output_encoding("preserve", "cp949") == "cp949"
    assert resolve_output_encoding("preserve", "euc-kr") == "euc_kr"


def test_atomic_writer_keeps_backup(tmp_path: Path):
    path = tmp_path / "oto.ini"
    write_lines_atomic(path, ["a.wav=a,0,10,-10,5,2"], encoding="utf-8")
    backup = write_lines_atomic(path, ["a.wav=a,1,10,-10,5,2"], encoding="utf-8")
    assert backup is not None and backup.is_file()
    assert path.read_text(encoding="utf-8").startswith("a.wav=a,1")


def test_negative_cutoff_is_span_from_offset():
    assert oto_cutoff_end_abs(duration_ms=5000, offset=960, cutoff=-422) == 1382
    valid, reasons = validate_oto_line_values(
        duration_ms=5000,
        offset=960,
        consonant=158,
        cutoff=-422,
        preutterance=70,
        overlap=21,
    )
    assert valid
    assert reasons == ()
