import os
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.alignment_pipeline import run_alignment_with_fallback
from core.pipeline_status import OK, normalize_aligner_name, resolve_aligner_chain


def _touch_textgrid(out_dir: str, name: str = "sample.TextGrid") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
        handle.write('File type = "ooTextFile"\n')


def test_normalize_aligner_name_accepts_sequence_aliases():
    for raw in ("sequence", "seq", "dedicated", "sequence-label", "전용(시퀀스)"):
        assert normalize_aligner_name(raw, default="mfa") == "sequence"


def test_resolve_aligner_chain_keeps_sequence_then_mfa():
    assert resolve_aligner_chain("sequence", "mfa") == ["sequence", "mfa"]


def test_sequence_primary_falls_back_to_mfa_when_sequence_not_ready(tmp_path):
    output_dir = os.path.join(str(tmp_path), "textgrids")

    def _fake_mfa_align(_mfa_path, _wav_folder, _dict_path, out_dir, **_kwargs):
        _touch_textgrid(out_dir)
        return True, ""

    with mock.patch(
        "core.alignment.alignment_pipeline.check_sequence_aligner_ready",
        return_value={"code": "ALIGN_NOT_READY", "message": "sequence unavailable"},
    ) as mocked_seq_ready, mock.patch(
        "core.alignment.alignment_pipeline.check_mfa_ready",
        return_value={"code": OK, "message": "ready", "mfa_path": "fake_mfa"},
    ) as mocked_mfa_ready, mock.patch(
        "core.alignment.alignment_pipeline.run_mfa_align",
        side_effect=_fake_mfa_align,
    ) as mocked_mfa_align:
        result = run_alignment_with_fallback(
            language="korean",
            wav_folder=str(tmp_path),
            dictionary_path=os.path.join(str(tmp_path), "dictionary.txt"),
            output_folder=output_dir,
            primary_aligner="sequence",
            fallback_aligner="",
            mfa_path="",
            mfa_align_profile="default",
            callback=None,
        )

    mocked_seq_ready.assert_called_once()
    mocked_mfa_ready.assert_called_once()
    mocked_mfa_align.assert_called_once()
    assert bool(result.get("ok", False)) is True
    assert str(result.get("used_engine", "")) == "mfa"
    assert bool(result.get("fallback_used", False)) is True
    assert "engine:sequence->mfa" in str(result.get("fallback_path", ""))


def test_sequence_primary_success_without_mfa(tmp_path):
    output_dir = os.path.join(str(tmp_path), "textgrids")

    def _fake_sequence_align(_sequence_path, _wav_folder, _dict_path, out_dir, **_kwargs):
        _touch_textgrid(out_dir)
        return True, "ok"

    with mock.patch(
        "core.alignment.alignment_pipeline.check_sequence_aligner_ready",
        return_value={"code": OK, "message": "ready"},
    ) as mocked_seq_ready, mock.patch(
        "core.alignment.alignment_pipeline.run_sequence_align",
        side_effect=_fake_sequence_align,
    ) as mocked_seq_align, mock.patch(
        "core.alignment.alignment_pipeline.check_mfa_ready"
    ) as mocked_mfa_ready:
        result = run_alignment_with_fallback(
            language="korean",
            wav_folder=str(tmp_path),
            dictionary_path=os.path.join(str(tmp_path), "dictionary.txt"),
            output_folder=output_dir,
            primary_aligner="sequence",
            fallback_aligner="",
            mfa_path="",
            mfa_align_profile="default",
            callback=None,
        )

    mocked_seq_ready.assert_called_once()
    mocked_seq_align.assert_called_once()
    mocked_mfa_ready.assert_not_called()
    assert bool(result.get("ok", False)) is True
    assert str(result.get("used_engine", "")) == "sequence"
    assert float(result.get("confidence", 0.0)) >= 0.0


def test_sequence_low_confidence_triggers_mfa_fallback(tmp_path):
    output_dir = os.path.join(str(tmp_path), "textgrids")

    def _fake_sequence_align(_sequence_path, _wav_folder, _dict_path, out_dir, **_kwargs):
        _touch_textgrid(out_dir)
        return True, "ok"

    def _fake_mfa_align(_mfa_path, _wav_folder, _dict_path, out_dir, **_kwargs):
        _touch_textgrid(out_dir)
        return True, ""

    with mock.patch(
        "core.alignment.alignment_pipeline.check_sequence_aligner_ready",
        return_value={"code": OK, "message": "ready"},
    ), mock.patch(
        "core.alignment.alignment_pipeline.run_sequence_align",
        side_effect=_fake_sequence_align,
    ), mock.patch(
        "core.alignment.alignment_pipeline.get_last_sequence_align_meta",
        return_value={"confidence": 0.10, "warnings": ["mis_mapping_rate_high:0.4"], "fallback_hint": "mfa"},
    ), mock.patch(
        "core.alignment.alignment_pipeline.check_mfa_ready",
        return_value={"code": OK, "message": "ready", "mfa_path": "fake_mfa"},
    ), mock.patch(
        "core.alignment.alignment_pipeline.run_mfa_align",
        side_effect=_fake_mfa_align,
    ):
        result = run_alignment_with_fallback(
            language="korean",
            wav_folder=str(tmp_path),
            dictionary_path=os.path.join(str(tmp_path), "dictionary.txt"),
            output_folder=output_dir,
            primary_aligner="sequence",
            fallback_aligner="mfa",
            mfa_path="",
            mfa_align_profile="default",
            callback=None,
        )

    assert bool(result.get("ok", False)) is True
    assert str(result.get("used_engine", "")) == "mfa"
    assert bool(result.get("fallback_used", False)) is True
