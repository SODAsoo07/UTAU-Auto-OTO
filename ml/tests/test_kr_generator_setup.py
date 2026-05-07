from core.kr_generator_setup import build_kr_textgrid_preparation, build_wav_index, prepare_kr_generation_setup
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


def test_prepare_kr_generation_setup_uses_matching_template(tmp_path):
    tg_dir = tmp_path / "TextGrid"
    tg_dir.mkdir()
    (tg_dir / "ga.TextGrid").write_text("", encoding="utf-8")
    (tmp_path / "ga.wav").write_bytes(b"")
    (tmp_path / "base.oto.ini").write_text("", encoding="utf-8")
    wav_index = build_wav_index(str(tmp_path), normalize_key_fn=normalize_wav_key)
    logs = []

    setup = prepare_kr_generation_setup(
        tg_folder=str(tg_dir),
        tpl_path=str(tmp_path / "base.oto.ini"),
        auto_gen_format="cvvc",
        custom_phonemes_path="",
        alias_suffix="C4",
        wav_root_for_signal=str(tmp_path),
        wav_index_for_signal=wav_index,
        normalize_key_fn=normalize_wav_key,
        find_wav_path_fn=lambda wav_name, _wav_root, index: index.get(normalize_wav_key(wav_name), ""),
        log_fn=logs.append,
        load_template_lines_fn=lambda _path: (["ga.wav=ga_C4,0,0,0,0,0"], "utf-8", "", ""),
        strip_template_alias_suffixes_fn=lambda lines, alias_suffix: (
            [line.replace("_" + alias_suffix, "") for line in lines],
            alias_suffix,
            1,
        ),
    )

    assert setup.use_template is True
    assert setup.template_lines == ["ga.wav=ga,0,0,0,0,0"]
    assert setup.file_groups == {"ga.wav": ["ga.wav=ga,0,0,0,0,0"]}
    assert any("접미사" in line for line in logs)


def test_prepare_kr_generation_setup_falls_back_on_low_template_match(tmp_path):
    tg_dir = tmp_path / "TextGrid"
    tg_dir.mkdir()
    (tg_dir / "ga.TextGrid").write_text("", encoding="utf-8")
    (tmp_path / "base.oto.ini").write_text("", encoding="utf-8")
    logs = []

    setup = prepare_kr_generation_setup(
        tg_folder=str(tg_dir),
        tpl_path=str(tmp_path / "base.oto.ini"),
        auto_gen_format="cvvc",
        custom_phonemes_path="",
        alias_suffix="",
        wav_root_for_signal=str(tmp_path),
        wav_index_for_signal={},
        normalize_key_fn=normalize_wav_key,
        find_wav_path_fn=lambda _wav_name, _wav_root, _index: "",
        log_fn=logs.append,
        load_template_lines_fn=lambda _path: (["missing.wav=ga,0,0,0,0,0"], "utf-8", "", ""),
        strip_template_alias_suffixes_fn=lambda lines, alias_suffix: (list(lines), alias_suffix, 0),
    )

    assert setup.use_template is False
    assert setup.file_groups == {}
    assert any("매칭률 낮음" in line for line in logs)
