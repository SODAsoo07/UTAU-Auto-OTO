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
    from core.model_context.filename import canonicalize_order_tokens, filename_order_tokens, parse_filename_context

    assert filename_order_tokens("a-ka_01.wav") == ["a", "ka", "01"]
    assert filename_order_tokens("ko-a.wav") == ["ko", "a"]
    assert canonicalize_order_tokens(["shi", "a", "sil", "nn"]) == ["sh", "a", "n"]
    assert parse_filename_context("\u304b-\u304d-\u304f.wav", language="japanese").canonical_tokens == ("ka", "ki", "ku")


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


def test_manual_oto_anchor_uses_utau_negative_cutoff_and_vc_role():
    from core.model_context.manual_oto_anchor import (
        classify_manual_oto_alias_role,
        manual_oto_alias_order_terms,
        manual_oto_params_to_anchor_record,
    )

    assert classify_manual_oto_alias_role("a k", language="japanese", format_type="cvvc") == "vc"
    assert classify_manual_oto_alias_role("a ka", language="japanese", format_type="vcv") == "vcv"
    assert classify_manual_oto_alias_role("a i", language="japanese", format_type="cvvc") == "vv"
    assert manual_oto_alias_order_terms("a k", language="japanese", format_type="cvvc", alias_role="vc")[0] == "a"
    assert manual_oto_alias_order_terms("n my_E4", language="japanese", format_type="cvvc", alias_role="vc")[0] == "n"
    assert manual_oto_alias_order_terms("a tsBA3", language="japanese", format_type="cvvc", alias_role="vc")[:2] == ["a", "ts"]
    assert manual_oto_alias_order_terms("n nyA3", language="japanese", format_type="cvvc", alias_role="vc")[:2] == ["n", "ny"]
    assert classify_manual_oto_alias_role("n nyA3", language="japanese", format_type="cvvc") == "vc"
    assert classify_manual_oto_alias_role("a RBA3", language="japanese", format_type="cvvc") == "vc"
    assert classify_manual_oto_alias_role("brC4S", language="korean", format_type="cvc") == "br"
    assert manual_oto_alias_order_terms("\u3042 \u304b", language="japanese", format_type="vcv", alias_role="vcv")[0] == "ka"

    record = manual_oto_params_to_anchor_record(
        {"offset": 100.0, "consonant": 180.0, "cutoff": -250.0, "preutterance": 120.0, "overlap": 40.0},
        duration_ms=1000.0,
    )
    assert record.anchors_abs_ms["offset"] == 100.0
    assert record.anchors_abs_ms["overlap"] == 140.0
    assert record.anchors_abs_ms["preutterance"] == 220.0
    assert record.anchors_abs_ms["fixed_end"] == 280.0
    assert record.anchors_abs_ms["cutoff"] == 750.0
    assert record.usable_as_target


def test_manual_oto_anchor_manifest_builder_smoke(tmp_path):
    from ml.scripts.mfa_free_oto.build_manual_oto_anchor_manifest import build_manual_oto_anchor_manifest

    bank = tmp_path / "japanese" / "cvvc" / "bank" / "C4"
    bank.mkdir(parents=True)
    wav = bank / "a_ka.wav"
    wav.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    (bank / "oto.ini").write_text(
        "a_ka.wav=a k,100,180,-10,120,40\n"
        "a_ka.wav=ka,300,220,-5,130,55\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    summary = build_manual_oto_anchor_manifest(dataset_root=tmp_path, out_path=out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert summary["stats"]["rows_written"] == 2
    assert rows[0]["label_source"] == "manual_oto_anchor"
    assert rows[0]["alias_role"] == "vc"
    assert rows[1]["alias_role"] == "cv"
    assert rows[0]["slot_index"] == 0
    assert rows[0]["slot_count"] == 2


def test_manual_oto_anchor_split_filters_unusable_and_invalid(tmp_path):
    from ml.scripts.mfa_free_oto.split_manual_oto_anchor_manifest import split_manual_oto_anchor_manifest

    manifest = tmp_path / "manual.jsonl"
    base = {
        "label_source": "manual_oto_anchor",
        "language": "japanese",
        "format_type": "cvvc",
        "voicebank_id": "bank-a",
        "wav_path": "a.wav",
        "alias": "a k",
    }
    rows = [
        {**base, "alias_role": "vc", "usable_as_target": True, "label_quality": "trusted_manual"},
        {**base, "alias_role": "invalid", "usable_as_target": True, "label_quality": "trusted_manual"},
        {**base, "alias_role": "cv", "usable_as_target": False, "label_quality": "unusable_manual"},
        {**base, "voicebank_id": "bank-b", "alias_role": "cv", "usable_as_target": True, "label_quality": "flagged_manual"},
    ]
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    summary = split_manual_oto_anchor_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "splits",
        seed=1,
        val_ratio=0.0,
        test_ratio=0.0,
        include_flagged=False,
    )
    assert summary["filtered_rows"] == 1
    filtered = (tmp_path / "splits" / "manual_oto_anchor_filtered.jsonl").read_text(encoding="utf-8")
    assert '"alias_role": "vc"' in filtered


def test_manual_oto_candidate_features_have_stable_shape():
    import numpy as np

    from core.mfa_free_oto.manual_oto_candidates import (
        ManualOtoCandidateTracks,
        manual_oto_candidate_feature_names,
        manual_oto_candidate_features,
    )

    times = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    tracks = ManualOtoCandidateTracks(
        times_ms=times,
        candidate_indices={"preutterance": (0, 1, 2)},
        anchor_scores={"preutterance": np.asarray([0.1, 0.9, 0.3], dtype=np.float32)},
        tracks={
            "transition": np.asarray([0.2, 0.8, 0.1], dtype=np.float32),
            "nucleus": np.asarray([0.1, 0.7, 0.5], dtype=np.float32),
            "rms": np.asarray([0.1, 0.6, 0.6], dtype=np.float32),
            "activity_edge": np.asarray([0.0, 0.5, 0.1], dtype=np.float32),
        },
        duration_ms=100.0,
        encoder="unit",
    )
    row = {
        "language": "japanese",
        "format_type": "cvvc",
        "alias_role": "vc",
        "duration_ms": 100.0,
        "slot_pos_norm": 0.5,
        "slot_index": 1,
        "slot_count": 3,
    }
    values = manual_oto_candidate_features(row, tracks, anchor="preutterance", candidate_index=1)
    assert values.shape == (len(manual_oto_candidate_feature_names()),)
    assert np.isfinite(values).all()
