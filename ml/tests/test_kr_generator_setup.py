from core.kr_generator_setup import build_kr_textgrid_preparation, build_wav_index
from core.oto_normalization import normalize_wav_key


def test_build_wav_index_recurses_and_normalizes(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    wav_path = nested / "Ga_01.wav"
    wav_path.write_bytes(b"")

    index = build_wav_index(str(tmp_path), normalize_key_fn=normalize_wav_key)

    assert index[normalize_wav_key("Ga_01.wav")] == str(wav_path)


def test_kr_textgrid_preparation_resolves_exact_and_template_stats(tmp_path):
    tg_dir = tmp_path / "TextGrid"
    wav_dir = tmp_path
    tg_dir.mkdir()
    (tg_dir / "ga.TextGrid").write_text("", encoding="utf-8")
    (wav_dir / "ga.wav").write_bytes(b"")
    wav_index = build_wav_index(str(wav_dir), normalize_key_fn=normalize_wav_key)
    logs = []

    prep = build_kr_textgrid_preparation(
        tg_folder=str(tg_dir),
        wav_root_for_signal=str(wav_dir),
        wav_index_for_signal=wav_index,
        normalize_key_fn=normalize_wav_key,
        find_wav_path_fn=lambda wav_name, wav_root, index: index.get(normalize_wav_key(wav_name), ""),
        log_fn=logs.append,
    )

    assert len(prep.tg_entries) == 1
    assert prep.resolve_tg_info("ga.wav")["output_name"] == "ga.wav"
    assert prep.template_match_stats(["ga.wav=ga,0,0,0,0,0"]) == (1, 1, 1.0)
    assert logs == []


def test_kr_textgrid_preparation_reports_ambiguous_norm(tmp_path):
    tg_dir = tmp_path / "TextGrid"
    tg_dir.mkdir()
    (tg_dir / "ga.TextGrid").write_text("", encoding="utf-8")
    (tg_dir / "g a.TextGrid").write_text("", encoding="utf-8")
    logs = []

    prep = build_kr_textgrid_preparation(
        tg_folder=str(tg_dir),
        wav_root_for_signal=str(tmp_path),
        wav_index_for_signal={},
        normalize_key_fn=lambda _name: "same",
        find_wav_path_fn=lambda _wav_name, _wav_root, _index: "",
        log_fn=logs.append,
    )

    assert prep.resolve_tg_info("different.wav") is None
    assert any("ambiguous" in line for line in logs)
