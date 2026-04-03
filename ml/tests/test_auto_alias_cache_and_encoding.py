import importlib
import sys
import types

from core.auto_alias_reference import _ALIAS_REFERENCE_CACHE, collect_auto_alias_reference
from core.oto_encoding import detect_text_file_encoding, is_shift_jis_family_encoding


def test_auto_alias_reference_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.delenv("UTOA_AUTO_ALIAS_DATA_DIRS", raising=False)
    _ALIAS_REFERENCE_CACHE.clear()

    app_dir = tmp_path / "app"
    ref_dir = app_dir / "data" / "oto_aliases" / "ja_ref"
    ref_dir.mkdir(parents=True)
    oto_path = ref_dir / "oto.ini"
    oto_path.write_text(
        "\n".join(
            [
                "a.wav=あ,0,10,-40,5,2",
                "i.wav=い,0,10,-40,5,2",
                "u.wav=う,0,10,-40,5,2",
                "e.wav=え,0,10,-40,5,2",
                "o.wav=お,0,10,-40,5,2",
                "n.wav=ん,0,10,-40,5,2",
            ]
        ),
        encoding="utf-8",
    )

    first = collect_auto_alias_reference(
        language="japanese",
        format_type="cv",
        app_dir=str(app_dir),
    )
    assert first["matched_files"] >= 1
    assert len(first["aliases"]) >= 6

    import core.auto_alias_reference as auto_alias_reference

    def _must_not_scan(*_args, **_kwargs):
        raise AssertionError("reference scan should not run on cache hit")

    monkeypatch.setattr(auto_alias_reference, "_iter_oto_ini_candidates", _must_not_scan)
    second = collect_auto_alias_reference(
        language="japanese",
        format_type="cv",
        app_dir=str(app_dir),
    )
    assert second["aliases"] == first["aliases"]


def test_ja_finalize_preserves_shift_jis_output(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    wav_path.write_bytes(b"dummy")

    oto_path = tmp_path / "oto.ini"
    oto_path.write_bytes("a.wav=あ,0.00,5.00,2.00,1.00,0.50\n".encode("cp932"))

    import core.ja_oto_mapping as ja_mapping
    import core.oto_generator as oto_generator
    from core.ja_oto_finalize import _convert_ja_internal_cutoff_to_oto_field

    monkeypatch.setattr(ja_mapping, "classify_ja_alias", lambda _alias: "cv")
    monkeypatch.setattr(
        oto_generator,
        "_find_wav_path_for_name",
        lambda _wav_name, _wav_dir, _wav_index: str(wav_path),
    )
    monkeypatch.setattr(oto_generator, "_wav_duration_ms", lambda _path: 400.0)

    changed = _convert_ja_internal_cutoff_to_oto_field(str(oto_path), str(wav_dir))
    assert changed >= 1
    assert is_shift_jis_family_encoding(detect_text_file_encoding(str(oto_path)))
    assert "a.wav=あ," in oto_path.read_bytes().decode("cp932")


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _FakeEntry:
    def __init__(self, text: str):
        self.text = text
        self.state = "normal"

    def get(self):
        return self.text

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]

    def delete(self, _start, _end=None):
        self.text = ""

    def insert(self, _index, value):
        self.text = str(value)


def _load_app_runtime_mixin(monkeypatch):
    if "customtkinter" not in sys.modules:
        monkeypatch.setitem(sys.modules, "customtkinter", types.SimpleNamespace())
    module = importlib.import_module("ui.app_mixins")
    return module.AppRuntimeMixin


def _build_reset_dummy(app_runtime_mixin_cls, answers):
    class _Dummy(app_runtime_mixin_cls):
        def __init__(self):
            self._answers = list(answers)
            self.wav_entry = _FakeEntry("C:/wav")
            self.tpl_entry = _FakeEntry("C:/base.ini")
            self.out_entry = _FakeEntry("C:/oto.ini")
            self.openutau_var = _FakeVar(True)
            self._saved = False
            self.logs = []

        def _ask_yes_no_dialog_sync(self, _title, _message, default=False):
            if self._answers:
                return bool(self._answers.pop(0))
            return bool(default)

        def _save_config(self):
            self._saved = True

        def _append_log(self, message):
            self.logs.append(str(message))

    return _Dummy()


def test_reset_all_settings_defaults_can_keep_paths(monkeypatch):
    app_runtime_mixin_cls = _load_app_runtime_mixin(monkeypatch)
    dummy = _build_reset_dummy(app_runtime_mixin_cls, [True, True])

    dummy._reset_all_settings_defaults()

    assert dummy._saved is True
    assert dummy.openutau_var.get() is False
    assert dummy.wav_entry.get() == "C:/wav"
    assert dummy.tpl_entry.get() == "C:/base.ini"
    assert dummy.out_entry.get() == "C:/oto.ini"


def test_reset_all_settings_defaults_can_clear_paths(monkeypatch):
    app_runtime_mixin_cls = _load_app_runtime_mixin(monkeypatch)
    dummy = _build_reset_dummy(app_runtime_mixin_cls, [True, False])

    dummy._reset_all_settings_defaults()

    assert dummy._saved is True
    assert dummy.openutau_var.get() is False
    assert dummy.wav_entry.get() == ""
    assert dummy.tpl_entry.get() == ""
    assert dummy.out_entry.get() == ""
