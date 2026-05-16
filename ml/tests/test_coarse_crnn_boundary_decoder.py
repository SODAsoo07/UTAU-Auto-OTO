from __future__ import annotations

import wave

from core.coarse_crnn.boundary_targets import build_boundary_target_map, load_row_specs_from_source_oto
from core.coarse_crnn import boundary_targets as bt
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, BoundaryCandidate, OtoRowSpec
from core.coarse_crnn.wav_decoder import decode_wav_rows


def _spec(role: str, *, slot_index: int, line_index: int) -> OtoRowSpec:
    return OtoRowSpec(
        wav_name="ga_gi_gu.wav",
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        alias=f"{role}_{line_index}",
        role=role,
        slot_index=slot_index,
        slot_count=3,
        prev_alias="",
        next_alias="",
        language="korean",
        format_type="cvvc",
        line_index=line_index,
        duration_ms=1500.0,
        source_params={},
    )


def _write_wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def test_load_row_specs_ignores_source_param_values(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a.wav=a,123,456,-789,321,111\n", encoding="utf-8")
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="japanese",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 1
    assert specs[0].source_params == {}


def test_boundary_target_map_contains_vc_labels():
    spec = _spec("vc", slot_index=0, line_index=0)
    anchors = AbsoluteOtoAnchors(
        offset_abs=140.0,
        overlap_abs=165.0,
        pre_abs=190.0,
        consonant_abs=230.0,
        cutoff_abs=320.0,
    )
    _times, target = build_boundary_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert target.shape == (64, 8)
    # vc row should drive vowel_end / transition / next_onset labels.
    assert float(target[:, 4].max()) > 0.5
    assert float(target[:, 5].max()) > 0.5
    assert float(target[:, 6].max()) > 0.5


def test_wav_decoder_keeps_vc_inside_anchor_pair():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.93, source="model"),
        BoundaryCandidate(time_ms=420.0, kind="transition_peak", score=0.90, source="model"),
        BoundaryCandidate(time_ms=455.0, kind="next_onset", score=0.82, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    left = decoded.anchor_timeline_ms[0]
    right = decoded.anchor_timeline_ms[1]
    vc_row = decoded.rows[1]
    assert left - 28.0 <= vc_row.selected_time_ms <= right + 42.0


def test_wav_decoder_vc_prefers_late_transition_to_reduce_front_v():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=320.0, kind="transition_peak", score=0.92, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    vc_row = decoded.rows[1]
    # VC target is intentionally right-leaning to avoid over-long preceding vowel.
    assert float(vc_row.selected_time_ms) >= 480.0


def test_wav_decoder_vc_penalizes_early_vowel_end_even_when_score_is_higher():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        # Early vowel_end has a higher confidence score, but should not win for VC.
        BoundaryCandidate(time_ms=360.0, kind="vowel_end", score=0.99, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.90, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    vc_row = decoded.rows[1]
    assert float(vc_row.selected_time_ms) >= 500.0


def test_wav_decoder_right_guard_limits_cutoff_for_dense_cv():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
    ]
    cands = [
        BoundaryCandidate(time_ms=210.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=280.0, kind="syllable_onset", score=0.96, source="model"),
        BoundaryCandidate(time_ms=214.0, kind="vowel_start", score=0.90, source="model"),
        BoundaryCandidate(time_ms=284.0, kind="vowel_start", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\dense_cv.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 2
    right_anchor = float(decoded.anchor_timeline_ms[1])
    row0 = decoded.rows[0]
    # cutoff must stay before the next slot boundary (guarded by right-anchor margin).
    assert float(row0.anchors.cutoff_abs) <= right_anchor - 4.0


def test_wav_decoder_ignores_source_params():
    rows = [
        _spec("vc", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
    ]
    rows_with_bad_source = [
        OtoRowSpec(**{**rows[0].__dict__, "source_params": {"offset": 999.0, "preutterance": 888.0, "consonant": 777.0, "cutoff": -5.0, "overlap": 666.0}}),
        rows[1],
    ]
    cands = [
        BoundaryCandidate(time_ms=220.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=340.0, kind="syllable_onset", score=0.96, source="model"),
        BoundaryCandidate(time_ms=280.0, kind="transition_peak", score=0.92, source="model"),
        BoundaryCandidate(time_ms=300.0, kind="next_onset", score=0.90, source="model"),
    ]
    decoded_plain = decode_wav_rows(
        wav_path=r"C:\tmp\ignore_source_plain.wav",
        duration_ms=1000.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=100.0,
        active_end_ms=900.0,
    )
    decoded_bad = decode_wav_rows(
        wav_path=r"C:\tmp\ignore_source_bad.wav",
        duration_ms=1000.0,
        row_specs=rows_with_bad_source,
        candidates=cands,
        active_start_ms=100.0,
        active_end_ms=900.0,
    )
    assert len(decoded_plain.rows) == len(decoded_bad.rows) == 2
    assert float(decoded_plain.rows[0].selected_time_ms) == float(decoded_bad.rows[0].selected_time_ms)
    assert float(decoded_plain.rows[0].anchors.pre_abs) == float(decoded_bad.rows[0].anchors.pre_abs)
    assert float(decoded_plain.rows[0].anchors.cutoff_abs) == float(decoded_bad.rows[0].anchors.cutoff_abs)


def test_wav_decoder_active_end_pad_extends_anchor_timeline(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ACTIVE_END_PAD_MS", "200")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
        _spec("cv", slot_index=2, line_index=2),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\pad.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=[],
        active_start_ms=100.0,
        active_end_ms=700.0,
    )
    assert len(decoded.anchor_timeline_ms) == 3
    assert decoded.anchor_timeline_ms[-1] > 700.0


def test_wav_decoder_anchor_shift_moves_timeline_forward(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_SHIFT_MS", "120")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
        _spec("cv", slot_index=2, line_index=2),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\shift.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=[],
        active_start_ms=100.0,
        active_end_ms=700.0,
    )
    assert len(decoded.anchor_timeline_ms) == 3
    assert decoded.anchor_timeline_ms[0] >= 220.0


def test_predictor_dispatch_prefers_boundary_engine(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    called = {"boundary": 0, "legacy": 0}

    def _fake_boundary(**kwargs):
        called["boundary"] += 1
        return 1, 1, []

    def _fake_legacy(**kwargs):
        called["legacy"] += 1
        return 0, 1, ["legacy called"]

    monkeypatch.setattr(mod, "generate_oto_with_boundary_decoder", _fake_boundary)
    monkeypatch.setattr(mod._legacy, "generate_oto_with_crnn_predictor", _fake_legacy)

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path=".",
        source_oto_path=".",
        language="korean",
        engine="boundary_decoder",
    )
    assert (processed, total, errors) == (1, 1, [])
    assert called["boundary"] == 1
    assert called["legacy"] == 0


def test_slot_count_fallback_uses_anchor_roles_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["cv", "vc", "cv", "vc", "other"],
        row_count=5,
    )
    assert slot_count >= 2


def test_slot_count_fallback_handles_vv_chain_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["v", "vv", "vv", "vv", "vv", "vv", "vv"],
        row_count=7,
    )
    assert slot_count >= 6


def test_slot_count_fallback_handles_vcv_other_mix_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["other", "v-cv", "other", "v-cv", "other", "v-cv", "other", "v-cv", "other"],
        row_count=9,
    )
    assert slot_count >= 5


def test_project_row_index_to_slot_spreads_sparse_rows():
    projected = [
        bt._project_row_index_to_slot(row_index=i, row_count=4, slot_count=8)
        for i in range(4)
    ]
    assert projected == [0, 2, 5, 7]


def test_project_row_index_to_slot_is_monotonic():
    projected = [
        bt._project_row_index_to_slot(row_index=i, row_count=10, slot_count=8)
        for i in range(10)
    ]
    assert projected[0] == 0
    assert projected[-1] == 7
    assert projected == sorted(projected)


def test_low_anchor_mode_projects_slot_indices_monotonic(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_path = wav_dir / "_chain.wav"
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "\n".join(
            [
                "_chain.wav=あ,0,100,-200,80,40",
                "_chain.wav=a あ,0,100,-200,80,40",
                "_chain.wav=a い,0,100,-200,80,40",
                "_chain.wav=i あ,0,100,-200,80,40",
                "_chain.wav=a う,0,100,-200,80,40",
                "_chain.wav=u え,0,100,-200,80,40",
                "_chain.wav=e あ,0,100,-200,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="japanese",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 7
    assert int(specs[0].slot_count) > 1
    unique_slots = sorted({int(spec.slot_index) for spec in specs})
    assert len(unique_slots) > 2


def test_low_anchor_mode_shifts_korean_coda_bridge_rows_left_when_filename_tokens_exist(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_name = "_ssung'ssun'ssum'ssul'ssu'sseu'ssa'sseu.wav"
    wav_path = wav_dir / wav_name
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "\n".join(
            [
                f"{wav_name}=- ssw,0,100,-200,80,40",
                f"{wav_name}=NG ssw,0,100,-200,80,40",
                f"{wav_name}=N ssw,0,100,-200,80,40",
                f"{wav_name}=M ssw,0,100,-200,80,40",
                f"{wav_name}=L ssw,0,100,-200,80,40",
                f"{wav_name}=u ss,0,100,-200,80,40",
                f"{wav_name}=a ss,0,100,-200,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="korean",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 7
    # base projection for row_count=7, slot_count=8 is [0,1,2,4,5,6,7].
    # KO coda-bridge transition rows (NG/N/M/L + onset) should shift left by
    # one slot even when filename tokens exist, while plain VC rows stay put.
    assert [int(spec.slot_index) for spec in specs] == [0, 0, 1, 3, 4, 6, 7]


def test_japanese_vowel_space_alias_is_not_other():
    alias_type = bt._infer_alias_type("a あ", language="japanese")
    transition_type = bt._infer_transition_type("a あ", language="japanese")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "japanese",
            "a あ",
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert role in {"vv", "v-cv"}


def test_korean_v_plus_c_space_alias_is_vc():
    alias = "a g"
    alias_type = bt._infer_alias_type(alias, language="korean")
    transition_type = bt._infer_transition_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert alias_type == "vc"
    assert transition_type == "vc"
    assert role == "vc"


def test_korean_coda_bridge_alias_is_vcv_bridge():
    alias = "NG g"
    alias_type = bt._infer_alias_type(alias, language="korean")
    transition_type = bt._infer_transition_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert alias_type == "vcv"
    assert transition_type == "cv"
    assert role == "v-cv"


def test_korean_v_plus_glide_consonant_alias_is_vc():
    alias = "a gy"
    alias_type = bt._infer_alias_type(alias, language="korean")
    transition_type = bt._infer_transition_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert alias_type == "vc"
    assert transition_type == "vc"
    assert role == "vc"
