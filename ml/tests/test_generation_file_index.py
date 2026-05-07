from core.generation.file_index import (
    build_wav_index,
    has_direct_wav_files,
    iter_textgrid_files,
    list_wav_names,
)
from core.ja_generator_setup import build_ja_textgrid_preparation
from core.oto_normalization import normalize_wav_key


def test_textgrid_iterator_can_preserve_nonrecursive_behavior(tmp_path):
    top = tmp_path / "top.TextGrid"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "nested.TextGrid"
    top.write_text("", encoding="utf-8")
    nested.write_text("", encoding="utf-8")

    direct = list(iter_textgrid_files(str(tmp_path), recursive=False))
    recursive = list(iter_textgrid_files(str(tmp_path), recursive=True))

    assert [name for _dirpath, name in direct] == ["top.TextGrid"]
    assert sorted(name for _dirpath, name in recursive) == ["nested.TextGrid", "top.TextGrid"]


def test_wav_index_respects_recursive_flag(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (tmp_path / "root.wav").write_bytes(b"")
    (nested_dir / "nested.wav").write_bytes(b"")

    direct = build_wav_index(str(tmp_path), normalize_key_fn=normalize_wav_key, recursive=False)
    recursive = build_wav_index(str(tmp_path), normalize_key_fn=normalize_wav_key, recursive=True)

    assert normalize_wav_key("root.wav") in direct
    assert normalize_wav_key("nested.wav") not in direct
    assert normalize_wav_key("nested.wav") in recursive


def test_direct_wav_helpers_ignore_nested_files(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested.wav").write_bytes(b"")
    assert has_direct_wav_files(str(tmp_path)) is False
    assert list_wav_names(str(tmp_path), recursive=False) == set()

    (tmp_path / "root.wav").write_bytes(b"")
    assert has_direct_wav_files(str(tmp_path)) is True
    assert list_wav_names(str(tmp_path), recursive=False) == {"root.wav"}


def test_ja_textgrid_preparation_keeps_top_level_textgrid_scope(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (tmp_path / "root.TextGrid").write_text("", encoding="utf-8")
    (nested_dir / "nested.TextGrid").write_text("", encoding="utf-8")

    prep = build_ja_textgrid_preparation(str(tmp_path), normalize_key_fn=normalize_wav_key)

    assert [entry["real_name"] for entry in prep.tg_entries] == ["root.wav"]
