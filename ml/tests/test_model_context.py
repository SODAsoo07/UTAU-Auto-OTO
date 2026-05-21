from __future__ import annotations

import json
import subprocess
import sys


def test_oto_params_keep_raw_and_repaired_separate():
    from core.model_context.oto_params import source_params_to_context

    ctx = source_params_to_context(
        {"offset": 100.0, "consonant": 40.0, "cutoff": 20.0, "preutterance": 80.0, "overlap": 95.0},
        duration_ms=300.0,
    )
    assert dict(ctx.raw_params)["overlap"] == 95.0
    assert dict(ctx.repaired_params)["overlap"] <= dict(ctx.repaired_params)["preutterance"]
    assert ctx.anchor_mapping()["overlap_abs"] <= ctx.anchor_mapping()["pre_abs"]
    assert "anchor_order_repaired" in ctx.repair_warnings


def test_oto_params_round_trip_and_cutoff_signs():
    from core.model_context.oto_params import absolute_anchors_to_oto_params, source_params_to_context

    negative = source_params_to_context(
        {"offset": 100.0, "consonant": 180.0, "cutoff": -500.0, "preutterance": 120.0, "overlap": 40.0},
        duration_ms=1000.0,
    )
    assert negative.anchor_mapping()["cutoff_abs"] == 600.0

    positive = source_params_to_context(
        {"offset": 100.0, "consonant": 180.0, "cutoff": 500.0, "preutterance": 120.0, "overlap": 40.0},
        duration_ms=1000.0,
    )
    params = absolute_anchors_to_oto_params(positive.to_absolute_anchors(), duration_ms=1000.0)
    round_trip = source_params_to_context(params, duration_ms=1000.0)
    assert round_trip.anchor_mapping()["cutoff_abs"] == positive.anchor_mapping()["cutoff_abs"]
    assert tuple(params) == ("offset", "consonant", "cutoff", "preutterance", "overlap")


def test_filename_tokenization_golden_cases():
    from core.model_context.filename import canonicalize_order_tokens, filename_order_tokens

    assert filename_order_tokens("a-ka_01.wav") == ["a", "ka", "01"]
    assert filename_order_tokens("ko-a.wav") == ["ko", "a"]
    assert canonicalize_order_tokens(["shi", "a", "sil", "nn"]) == ["sh", "a", "n"]


def test_filename_lab_match_split_statuses():
    from core.model_context.filename import classify_filename_lab_match

    matched = classify_filename_lab_match("ka", ["k", "a"])
    assert matched.match_status in {"match", "ordered_match"}
    assert matched.match_ratio >= 0.99

    mismatch = classify_filename_lab_match("ka", ["m", "o"])
    assert mismatch.match_status == "mismatch"

    boundary_only = classify_filename_lab_match("", ["sil", "pau"])
    assert boundary_only.match_status == "boundary_only"


def test_model_context_imports_without_heavy_or_forbidden_imports():
    code = (
        "import json, sys\n"
        "import core.model_context.audio\n"
        "bad = [name for name in sys.modules if name in {'torch', 'transformers'} or name == 'ml' or name.startswith('ml.')]\n"
        "print(json.dumps(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.strip()) == []


def test_boundary_decode_adapter_accepts_model_context():
    from core.model_context.builder import contexts_to_boundary_decode_result
    from core.model_context.types import AliasRowContext, FilenameTokenContext, ModelSampleContext, OtoParamContext, WavIdentity

    ctx = ModelSampleContext(
        wav=WavIdentity(wav_path="a.wav", wav_name="a.wav", wav_stem="a", language="japanese", format_type="cv"),
        filename=FilenameTokenContext(raw="a.wav", tokens=("a",), canonical_tokens=("a",), token_count=1, slot_count=1),
        alias=AliasRowContext(alias="a", alias_role="cv", alias_type="cv"),
        oto=OtoParamContext(
            raw_params={"offset": 100.0, "consonant": 160.0, "cutoff": -500.0, "preutterance": 140.0, "overlap": 40.0},
            repaired_params={"offset": 100.0, "consonant": 160.0, "cutoff": -500.0, "preutterance": 140.0, "overlap": 40.0},
            anchors_ms=(100.0, 140.0, 240.0, 260.0, 600.0),
            duration_ms=1000.0,
        ),
    )
    result = contexts_to_boundary_decode_result([ctx])
    assert result.duration_ms == 1000.0
    assert len(result.rows) == 1
    assert result.rows[0].anchors.pre_abs == 240.0


def test_load_row_specs_from_source_oto_ignores_timing_by_default(tmp_path):
    from core.model_context.oto_rows import load_row_specs_from_source_oto

    wav = tmp_path / "a.wav"
    wav.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    src = tmp_path / "source.ini"
    src.write_text("a.wav=ka,1200,900,-700,520,440\n", encoding="utf-8")
    grouped = load_row_specs_from_source_oto(
        source_oto_path=str(src),
        wav_dir=str(tmp_path),
        language="japanese",
        format_type="CV",
    )
    rows = grouped[str(wav)]
    assert len(rows) == 1
    params = dict(rows[0].source_params)
    assert params["offset"] == 0.0
    assert params["consonant"] == 0.0
    assert params["cutoff"] == 0.0
    assert params["preutterance"] == 0.0
    assert params["overlap"] == 0.0
