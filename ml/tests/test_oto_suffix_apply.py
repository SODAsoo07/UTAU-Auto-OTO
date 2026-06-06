from core.oto_file_utils import (
    apply_alias_suffix_to_oto_file,
    apply_alias_suffix_to_oto_text,
    detect_oto_text_encoding,
    reencode_oto_file,
)
from ui.oto_actions_mixin import OtoActionsMixin


def test_apply_alias_suffix_to_oto_text_appends_and_preserves():
    text = (
        "a.wav=- ka,0,1,2,3,4\r\n"
        "a.wav=a k,0,1,2,3,4\r\n"
        "# comment line\r\n"
        "b.wav=ki,0,1,2,3,4\r\n"
    )
    new_text, changed = apply_alias_suffix_to_oto_text(text, "C4")
    assert changed == 3
    assert "a.wav=- ka_C4,0,1,2,3,4" in new_text
    assert "a.wav=a k_C4,0,1,2,3,4" in new_text
    assert "b.wav=ki_C4,0,1,2,3,4" in new_text
    assert "# comment line" in new_text  # non-alias lines untouched
    assert "\r\n" in new_text  # newline style preserved


def test_apply_alias_suffix_to_oto_text_is_idempotent():
    text = "a.wav=- ka,0,1,2,3,4\n"
    once, _ = apply_alias_suffix_to_oto_text(text, "C4")
    twice, changed = apply_alias_suffix_to_oto_text(once, "C4")
    assert changed == 0
    assert once == twice


def test_apply_alias_suffix_to_oto_text_normalizes_leading_underscore():
    text = "a.wav=ka,0,1,2,3,4\n"
    new_text, changed = apply_alias_suffix_to_oto_text(text, "_F4")
    assert changed == 1
    assert "a.wav=ka_F4,0,1,2,3,4" in new_text


def test_apply_alias_suffix_to_oto_file_in_place(tmp_path):
    oto = tmp_path / "oto.ini"
    oto.write_text("a.wav=- ka,0,1,2,3,4\na.wav=a k,0,1,2,3,4\n", encoding="utf-8")
    changed, total = apply_alias_suffix_to_oto_file(str(oto), "C4")
    assert (changed, total) == (2, 2)
    body = oto.read_text(encoding="utf-8")
    assert "- ka_C4" in body and "a k_C4" in body
    # Re-running does not double-append.
    changed2, _ = apply_alias_suffix_to_oto_file(str(oto), "C4")
    assert changed2 == 0
    assert "C4_C4" not in oto.read_text(encoding="utf-8")


def test_apply_alias_suffix_to_oto_file_missing_or_empty(tmp_path):
    assert apply_alias_suffix_to_oto_file(str(tmp_path / "nope.ini"), "C4") == (0, 0)
    oto = tmp_path / "oto.ini"
    oto.write_text("a.wav=ka,0,1,2,3,4\n", encoding="utf-8")
    assert apply_alias_suffix_to_oto_file(str(oto), "") == (0, 0)


def test_detect_oto_text_encoding_shift_jis_vs_utf8(tmp_path):
    sjis = tmp_path / "base.ini"
    sjis.write_bytes("_あ.wav=- あ,0,1,2,3,4\n".encode("cp932"))
    assert detect_oto_text_encoding(str(sjis)) == "cp932"

    utf = tmp_path / "u.ini"
    utf.write_bytes("_あ.wav=- あ,0,1,2,3,4\n".encode("utf-8"))
    assert detect_oto_text_encoding(str(utf)) == "utf-8"

    bom = tmp_path / "bom.ini"
    bom.write_bytes(b"\xef\xbb\xbf" + "a.wav=- a,0,1,2,3,4\n".encode("utf-8"))
    assert detect_oto_text_encoding(str(bom)) == "utf-8-sig"


def test_reencode_oto_file_to_shift_jis(tmp_path):
    gen = tmp_path / "oto.ini"
    gen.write_bytes("_あ.wav=- あ,0,1,2,3,4\n_か.wav=- か,0,1,2,3,4\n".encode("utf-8"))
    assert detect_oto_text_encoding(str(gen)) == "utf-8"

    changed, used = reencode_oto_file(str(gen), "shift_jis")
    assert changed is True
    assert used == "cp932"
    # Round-trips cleanly as cp932 and is no longer valid as strict utf-8.
    raw = gen.read_bytes()
    assert "- あ" in raw.decode("cp932")
    assert detect_oto_text_encoding(str(gen)) == "cp932"

    # Idempotent: already cp932 -> no rewrite.
    changed2, _ = reencode_oto_file(str(gen), "cp932")
    assert changed2 is False


def test_reencode_oto_file_preserves_when_unencodable(tmp_path):
    # Hangul cannot be represented in cp932; the file must be left intact.
    gen = tmp_path / "oto.ini"
    original = "_가.wav=- 가,0,1,2,3,4\n".encode("utf-8")
    gen.write_bytes(original)
    changed, _ = reencode_oto_file(str(gen), "cp932")
    assert changed is False
    assert gen.read_bytes() == original


def test_apply_alias_suffix_preserves_shift_jis(tmp_path):
    oto = tmp_path / "oto.ini"
    oto.write_bytes("_か.wav=- か,0,1,2,3,4\n".encode("cp932"))
    changed, _ = apply_alias_suffix_to_oto_file(str(oto), "C4")
    assert changed == 1
    # Suffix applied while keeping Shift-JIS encoding.
    assert detect_oto_text_encoding(str(oto)) == "cp932"
    assert "- か_C4" in oto.read_bytes().decode("cp932")


class _Entry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _SuffixDummy(OtoActionsMixin):
    def __init__(self, out_path):
        self.wav_entry = _Entry("")
        self.out_entry = _Entry(out_path)
        self.alias_suffix_var = _Var("C4")
        self.recursive_voicebank_scan_var = _Var(False)
        self.logs = []
        self.status = ""
        self.saved = 0

    def _run_in_thread(self, func):
        func()

    def _set_running(self, _running):
        pass

    def _set_status(self, status):
        self.status = str(status)

    def _append_log(self, message):
        self.logs.append(str(message))

    def _save_config(self):
        self.saved += 1

    def _handle_error(self, _label, exc):  # pragma: no cover - surfaces failures
        raise exc


def test_apply_suffix_handler_applies_to_generated_oto(tmp_path):
    oto = tmp_path / "oto.ini"
    oto.write_text("a.wav=- ka,0,1,2,3,4\n", encoding="utf-8")

    dummy = _SuffixDummy(str(oto))
    dummy._apply_suffix_to_generated_oto()

    assert "- ka_C4" in oto.read_text(encoding="utf-8")
    assert dummy.saved == 1
    assert "완료" in dummy.status

    # Idempotent: re-running the handler does not double-append.
    dummy._apply_suffix_to_generated_oto()
    assert "C4_C4" not in oto.read_text(encoding="utf-8")


def test_apply_suffix_handler_requires_suffix(tmp_path):
    oto = tmp_path / "oto.ini"
    oto.write_text("a.wav=- ka,0,1,2,3,4\n", encoding="utf-8")
    dummy = _SuffixDummy(str(oto))
    dummy.alias_suffix_var = _Var("   ")
    dummy._apply_suffix_to_generated_oto()
    assert "오류" in dummy.status
    assert "- ka," in oto.read_text(encoding="utf-8")  # unchanged
