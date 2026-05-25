from __future__ import annotations

import json
import math
import struct
import sys
import wave

import numpy as np
import pytest

from core.mfa_free_oto.decode import decode_monotonic_events
from core.mfa_free_oto.acoustic_nucleus import (
    AcousticNucleusConfig,
    relabel_vowel_nuclei_from_batch,
    select_acoustic_nucleus,
)
from core.mfa_free_oto.features import extract_features
from core.mfa_free_oto.features import FeatureBatch
from core.mfa_free_oto.htk_lab import (
    build_gold_manifest_from_htk_lab_dirs,
    classify_htk_phone,
    parse_htk_lab,
    row_from_htk_lab,
)
from core.mfa_free_oto.manifest import (
    ManifestValidationError,
    build_goldset_scaffold,
    load_manifest_jsonl,
    validate_manifest_row,
    write_manifest_jsonl,
)
from core.mfa_free_oto.manifest_audit import audit_manifest_path, infer_filename_phone_sequence
from core.mfa_free_oto.metrics import boundary_error_metrics, frame_classification_metrics
from core.mfa_free_oto.oto_adapter import (
    OtoAdapterConfig,
    OtoAnchor,
    _alias_type_for_row,
    adapt_template_row,
    assign_template_row_anchors,
    bootstrap_row,
    expected_slots_for_template_rows,
    parse_template_oto_line,
)
from core.mfa_free_oto.review_overlay import render_review_html
from core.mfa_free_oto import runtime_inference as runtime_inference_module
from core.mfa_free_oto.slot_viterbi import (
    SlotAssignment,
    SlotViterbiResult,
    assign_slots_viterbi,
    expected_cv_slots_from_phones,
    slot_assignments_to_decoded_events,
)
from core.mfa_free_oto.targets import rasterize_targets
from core.mfa_free_oto.types import DecodedEvent, EVENT_LABELS, FRAME_LABELS, FramePosterior
from core.mfa_free_oto.workflow import generate_no_mfa_oto_with_model_context


def test_goldset_scaffold_marks_rows_for_manual_labelling(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "001_ka.wav"
    _write_tone_wav(wav_path)
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("wav_name\taliases\texpected_phones\n001_ka.wav\tka|a k\tk a\n", encoding="utf-8")

    rows, summary = build_goldset_scaffold(
        wav_dir,
        language="korean",
        format_type="CV",
        aliases_path=aliases,
    )
    assert summary.rows == 1
    assert summary.needs_manual_labels == 1
    assert rows[0]["label_source"] == "manual_gold"
    assert rows[0]["needs_manual_labels"] is True
    assert rows[0]["aliases"] == ["ka", "a k"]
    assert rows[0]["expected_phones"] == ["k", "a"]

    manifest = tmp_path / "manifest.jsonl"
    assert write_manifest_jsonl(manifest, rows) == 1
    loaded = load_manifest_jsonl(manifest, require_manual=True, require_labels=False)
    assert loaded[0]["wav_name"] == "001_ka.wav"


def test_manifest_rejects_oto_or_mfa_pseudo_labels():
    row = _labelled_row("source_oto")
    with pytest.raises(ManifestValidationError):
        validate_manifest_row(row, require_manual=True, require_labels=True)
    row = _labelled_row("mfa_textgrid")
    with pytest.raises(ManifestValidationError):
        validate_manifest_row(row, require_manual=True, require_labels=True)


def test_targets_and_metrics_cover_frame_and_boundary_gates():
    row = _labelled_row("manual_gold")
    times = np.arange(0.0, 240.0, 10.0, dtype=np.float32)
    targets = rasterize_targets(row, times)
    assert targets.frame_class.shape == (24,)
    assert targets.event_targets.shape == (24, len(EVENT_LABELS))
    assert FRAME_LABELS[int(targets.frame_class[2])] == "silence"
    assert FRAME_LABELS[int(targets.frame_class[7])] == "consonant"
    assert FRAME_LABELS[int(targets.frame_class[14])] == "vowel"
    assert float(targets.event_targets[:, EVENT_LABELS.index("cv_boundary")].max()) > 0.9

    frame_metrics = frame_classification_metrics(targets.frame_class, targets.frame_class)
    assert frame_metrics["macro_f1"] == pytest.approx(1.0)
    boundary_metrics = boundary_error_metrics([100.0, 250.0], [112.0, 245.0])
    assert boundary_metrics["median_error_ms"] == pytest.approx(8.5)
    assert boundary_metrics["p90_error_ms"] == pytest.approx(11.3)


def test_decoder_uses_monotonic_event_order():
    times = [float(idx * 10) for idx in range(31)]
    event_scores = {label: [0.0 for _ in times] for label in EVENT_LABELS}
    for label, center in {"cv_boundary": 90.0, "vowel_nucleus": 150.0}.items():
        for idx, time_ms in enumerate(times):
            event_scores[label][idx] = math.exp(-0.5 * ((time_ms - center) / 10.0) ** 2)
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.25 for _ in times] for label in FRAME_LABELS},
        event_scores=event_scores,
    )
    decoded = decode_monotonic_events(
        posterior,
        expected_events=["cv_boundary", "vowel_nucleus"],
        min_score=0.05,
    )
    assert [event.label for event in decoded] == ["cv_boundary", "vowel_nucleus"]
    assert decoded[0].selected_time_ms < decoded[1].selected_time_ms
    assert decoded[0].selected_time_ms == pytest.approx(90.0)


def test_slot_viterbi_assigns_filename_slots_in_order():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.02 for _ in times] for label in FRAME_LABELS}
    for idx, time_ms in enumerate(times):
        class_probs["consonant"][idx] = 0.85 if 50.0 <= time_ms <= 90.0 or 170.0 <= time_ms <= 210.0 else 0.05
        class_probs["vowel"][idx] = 0.90 if 100.0 <= time_ms <= 155.0 or 220.0 <= time_ms <= 285.0 else 0.05
    for label, centers in {"cv_boundary": [100.0, 220.0], "vowel_nucleus": [145.0, 270.0]}.items():
        for center in centers:
            for idx, time_ms in enumerate(times):
                event_scores[label][idx] = max(
                    event_scores[label][idx],
                    math.exp(-0.5 * ((time_ms - center) / 9.0) ** 2),
                )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores={
            "transition_likelihood": [0.9 if time_ms in {100.0, 220.0} else 0.2 for time_ms in times],
            "flux_likelihood": [0.9 if time_ms in {100.0, 220.0} else 0.2 for time_ms in times],
            "voicing": [0.9 if time_ms in {150.0, 270.0} else 0.2 for time_ms in times],
            "silence_likelihood": [0.05 for _ in times],
        },
    )
    slots = expected_cv_slots_from_phones(["k", "a", "k", "i"])
    assert [(slot.phone, slot.role) for slot in slots] == [
        ("a", "cv_boundary"),
        ("a", "vowel_nucleus"),
        ("i", "cv_boundary"),
        ("i", "vowel_nucleus"),
    ]
    result = assign_slots_viterbi(posterior, expected_phones=["k", "a", "k", "i"], min_event_score=0.05)
    assert result.ok
    assert [assignment.phone for assignment in result.assignments] == ["a", "a", "i", "i"]
    assert [assignment.role for assignment in result.assignments] == [
        "cv_boundary",
        "vowel_nucleus",
        "cv_boundary",
        "vowel_nucleus",
    ]
    # Times carry sub-frame parabolic refinement, so allow up to half a frame of drift.
    assert [assignment.selected_time_ms for assignment in result.assignments] == pytest.approx(
        [100.0, 150.0, 220.0, 270.0], abs=6.0
    )
    decoded = slot_assignments_to_decoded_events(result)
    assert [event.label for event in decoded] == ["cv_boundary", "vowel_nucleus", "cv_boundary", "vowel_nucleus"]


def test_acoustic_aux_features_preserve_absolute_gate_tracks(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path, duration_s=0.35)
    acoustic = extract_features(wav_path, encoder="acoustic")
    aux = extract_features(wav_path, encoder="acoustic-aux")
    assert acoustic.features.shape[1] == 29
    assert aux.features.shape[1] > acoustic.features.shape[1]
    assert aux.times_ms.shape[0] == aux.features.shape[0]
    assert {"rms", "spectral_flux", "voicing", "silence_likelihood", "transition_likelihood"}.issubset(aux.acoustic_scores)
    assert float(np.max(aux.acoustic_scores["rms"])) > 0.0
    assert float(np.max(aux.acoustic_scores["voicing"])) >= 0.0


def test_acoustic_world_v1_features_available_when_dependencies_installed(tmp_path):
    wav_path = tmp_path / "tone_world.wav"
    _write_tone_wav(wav_path, duration_s=0.42)
    try:
        world = extract_features(wav_path, encoder="acoustic_world_v1")
    except RuntimeError as exc:
        pytest.skip(f"world stack unavailable in current env: {exc}")
    assert world.features.shape[0] == world.times_ms.shape[0]
    assert world.features.shape[1] > 0
    assert "world_voicing" in world.acoustic_scores
    assert "world_nucleus" in world.acoustic_scores
    assert "transition_likelihood" in world.acoustic_scores
    assert float(np.max(np.asarray(world.acoustic_scores["world_voicing"], dtype=np.float32))) >= 0.0


def test_acoustic_nucleus_relabel_preserves_lab_midpoint_diagnostic():
    times = np.asarray([0.0, 40.0, 80.0, 120.0, 160.0, 200.0], dtype=np.float32)
    batch = FeatureBatch(
        times_ms=times,
        features=np.zeros((len(times), 4), dtype=np.float32),
        sample_rate=16000,
        duration_ms=220.0,
        encoder="test",
        acoustic_scores={
            "world_nucleus": np.asarray([0.0, 0.1, 0.4, 1.0, 0.5, 0.0], dtype=np.float32),
            "world_voicing": np.asarray([0.0, 0.2, 0.6, 1.0, 0.6, 0.0], dtype=np.float32),
            "world_periodicity": np.asarray([0.0, 0.1, 0.5, 1.0, 0.7, 0.0], dtype=np.float32),
            "world_spectral_stability": np.asarray([0.0, 0.2, 0.5, 0.9, 0.6, 0.0], dtype=np.float32),
            "rms": np.asarray([0.0, 0.2, 0.6, 0.9, 0.6, 0.0], dtype=np.float32),
            "silence_likelihood": np.asarray([1.0, 0.4, 0.1, 0.0, 0.1, 0.8], dtype=np.float32),
        },
    )
    row = {
        "row_id": "a-ka",
        "wav_path": "a-ka.wav",
        "label_source": "manual_gold",
        "frame_labels": [
            {"label": "vowel", "start_ms": 40.0, "end_ms": 200.0, "phone": "a"},
        ],
        "events": [
            {
                "label": "vowel_nucleus",
                "time_ms": 120.0,
                "phone": "a",
                "source": "htk_vowel_segment_midpoint_pseudo_gold",
            }
        ],
    }
    selected = select_acoustic_nucleus(
        batch,
        start_ms=40.0,
        end_ms=200.0,
        config=AcousticNucleusConfig(edge_margin_ms=0.0),
    )
    assert selected["time_ms"] == pytest.approx(120.0)

    shifted_scores = {
        key: np.asarray([0.0, 0.1, 1.0, 0.3, 0.2, 0.0], dtype=np.float32)
        for key in (
            "world_nucleus",
            "world_voicing",
            "world_periodicity",
            "world_spectral_stability",
            "rms",
        )
    }
    shifted_scores["silence_likelihood"] = np.asarray([1.0, 0.4, 0.0, 0.1, 0.2, 0.8], dtype=np.float32)
    shifted = FeatureBatch(
        times_ms=times,
        features=batch.features,
        sample_rate=batch.sample_rate,
        duration_ms=batch.duration_ms,
        encoder=batch.encoder,
        acoustic_scores=shifted_scores,
    )
    relabelled, summary = relabel_vowel_nuclei_from_batch(
        row,
        shifted,
        config=AcousticNucleusConfig(edge_margin_ms=0.0),
    )
    nucleus = [event for event in relabelled["events"] if event["label"] == "vowel_nucleus"][0]
    assert nucleus["source"] == "acoustic_recomputed"
    assert nucleus["time_ms"] == pytest.approx(80.0)
    assert nucleus["lab_midpoint_time_ms"] == pytest.approx(120.0)
    assert nucleus["lab_midpoint_shift_ms"] == pytest.approx(-40.0)
    assert any(event["label"] == "vowel_nucleus_lab_midpoint" for event in relabelled["auxiliary_events"])
    assert summary["relabelled_events"] == 1


def test_manifest_audit_flags_slot_eligibility_and_clean_rows(tmp_path):
    wav_path = tmp_path / "a-ba.wav"
    _write_tone_wav(wav_path, duration_s=0.45)
    row = {
        "row_id": "a-ba",
        "wav_name": "a-ba.wav",
        "wav_path": str(wav_path),
        "duration_ms": 450.0,
        "label_source": "manual_gold",
        "expected_phones": ["a", "b", "a", "exh"],
        "frame_labels": [
            {"label": "silence", "start_ms": 0.0, "end_ms": 60.0, "phone": "AP"},
            {"label": "vowel", "start_ms": 60.0, "end_ms": 180.0, "phone": "a"},
            {"label": "consonant", "start_ms": 180.0, "end_ms": 240.0, "phone": "b"},
            {"label": "vowel", "start_ms": 240.0, "end_ms": 380.0, "phone": "a"},
            {"label": "other", "start_ms": 380.0, "end_ms": 430.0, "phone": "exh"},
        ],
        "events": [
            {"label": "cv_boundary", "time_ms": 240.0, "phone": "a"},
            {"label": "vowel_nucleus", "time_ms": 310.0, "phone": "a"},
        ],
    }
    manifest = tmp_path / "gold.jsonl"
    clean = tmp_path / "gold.clean.jsonl"
    write_manifest_jsonl(manifest, [row])

    assert infer_filename_phone_sequence("a-ba.wav") == ["a", "b", "a"]
    report = audit_manifest_path(manifest, clean_out=clean)
    assert report["rows"] == 1
    assert report["clean_rows"] == 1
    assert report["slot_metric_eligible_rows"] == 1
    assert report["row_reports"][0]["filename_match_ratio"] == pytest.approx(1.0)
    assert load_manifest_jsonl(clean, require_labels=True)[0]["row_id"] == "a-ba"


def test_filename_parser_handles_japanese_kana_sequence():
    assert infer_filename_phone_sequence("_ああいあうえあ.wav") == ["a", "a", "i", "a", "u", "e", "a"]
    assert infer_filename_phone_sequence("あか.wav") == ["a", "k", "a"]
    assert infer_filename_phone_sequence("_ヴぁヴぃヴヴぇヴぉヴぁんヴぁ.wav") == [
        "v",
        "a",
        "v",
        "i",
        "v",
        "u",
        "v",
        "e",
        "v",
        "o",
        "v",
        "a",
        "n",
        "v",
        "a",
    ]
    assert infer_filename_phone_sequence("_きゃききゅきぇきょきゃんきゃ.wav") == [
        "k",
        "y",
        "a",
        "k",
        "i",
        "k",
        "y",
        "u",
        "k",
        "y",
        "e",
        "k",
        "y",
        "o",
        "k",
        "y",
        "a",
        "n",
        "k",
        "y",
        "a",
    ]
    assert infer_filename_phone_sequence("_すぃさすせそさんさ.wav") == [
        "s",
        "i",
        "s",
        "a",
        "s",
        "u",
        "s",
        "e",
        "s",
        "o",
        "s",
        "a",
        "n",
        "s",
        "a",
    ]
    assert infer_filename_phone_sequence("_てゅてゅ_でゅでゅ.wav") == [
        "t",
        "y",
        "u",
        "t",
        "y",
        "u",
        "d",
        "y",
        "u",
        "d",
        "y",
        "u",
    ]
    assert _alias_type_for_row("ヴぁ", "auto") == "cv"
    assert _alias_type_for_row("すぃ", "auto") == "cv"
    assert _alias_type_for_row("てゅ", "auto") == "cv"
    assert _alias_type_for_row("a v", "auto") == "vc"
    assert _alias_type_for_row("a ky", "auto") == "vc"
    assert _alias_type_for_row("a ny", "auto") == "vc"
    assert _alias_type_for_row("u を", "auto") == "vv"


def test_template_rows_define_oto_slots_without_extra_filename_vowels():
    rows = [
        parse_template_oto_line("_き.wav=き,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=i k,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=か,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=a k,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=く,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=u k,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=け,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=e k,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=こ,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=o k,0,0,0,0,0"),
        parse_template_oto_line("_き.wav=n k,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows(
        [row for row in rows if row is not None],
        ["k", "i", "k", "a", "k", "u", "k", "e", "k", "o", "k", "a", "n", "k", "a"],
    )
    assert [(slot.phone_index, slot.event_label) for slot in slots] == [
        (1, "cv_boundary"),
        (2, "phone_change"),
        (3, "cv_boundary"),
        (4, "phone_change"),
        (5, "cv_boundary"),
        (6, "phone_change"),
        (7, "cv_boundary"),
        (8, "phone_change"),
        (9, "cv_boundary"),
        (10, "phone_change"),
        (11, "cv_boundary"),
        (13, "phone_change"),
        (14, "cv_boundary"),
    ]


def test_template_vowel_rows_define_vowel_nucleus_slots_in_row_order():
    rows = [
        parse_template_oto_line("_a.wav=あ,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a あ,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a い,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows([row for row in rows if row is not None], ["a", "a", "i"])
    assert [(slot.phone_index, slot.event_label) for slot in slots] == [
        (1, "vowel_nucleus"),
        (2, "vowel_nucleus"),
    ]


def test_template_rows_keep_out_of_order_alternate_alias_target():
    rows = [
        parse_template_oto_line("_u.wav=う,0,0,0,0,0"),
        parse_template_oto_line("_u.wav=u い,0,0,0,0,0"),
        parse_template_oto_line("_u.wav=o い,0,0,0,0,0"),
        parse_template_oto_line("_u.wav=u を,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows([row for row in rows if row is not None], ["u", "u", "i", "o", "i"])
    assert [(slot.phone_index, slot.event_label) for slot in slots] == [
        (0, "vowel_nucleus"),
        (2, "vowel_nucleus"),
        (4, "vowel_nucleus"),
        (3, "vowel_nucleus"),
    ]


def test_template_yoon_vc_alias_targets_first_consonant_of_cluster():
    rows = [
        parse_template_oto_line("_kya.wav=きゃ,0,0,0,0,0"),
        parse_template_oto_line("_kya.wav=a ky,0,0,0,0,0"),
        parse_template_oto_line("_kya.wav=きゅ,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows(
        [row for row in rows if row is not None],
        ["k", "y", "a", "k", "i", "k", "y", "u"],
    )
    assert [(slot.phone_index, slot.event_label) for slot in slots] == [
        (2, "cv_boundary"),
        (3, "phone_change"),
        (4, "cv_boundary"),
        (7, "cv_boundary"),
    ]


def test_template_alias_consonant_equivalents_cover_bank_spellings():
    rows = [
        parse_template_oto_line("_z.wav=ずぃ,0,0,0,0,0"),
        parse_template_oto_line("_z.wav=i j,0,0,0,0,0"),
        parse_template_oto_line("_h.wav=ふ,0,0,0,0,0"),
        parse_template_oto_line("_h.wav=u h,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows(
        [row for row in rows if row is not None],
        ["z", "i", "z", "a", "f", "u", "h", "e"],
    )
    assert [(slot.phone_index, slot.event_label) for slot in slots] == [
        (1, "cv_boundary"),
        (2, "phone_change"),
        (3, "cv_boundary"),
        (5, "cv_boundary"),
        (6, "phone_change"),
        (7, "cv_boundary"),
    ]


def test_template_anchor_assignment_covers_vcv_vowel_rows():
    times = [float(idx * 10) for idx in range(61)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.85 if 90.0 <= time_ms <= 500.0 else 0.05
        for label, centers in {
            "phone_change": [120.0, 300.0, 470.0],
            "vowel_nucleus": [150.0, 330.0, 500.0],
            "cv_boundary": [115.0, 295.0, 465.0],
        }.items():
            event_scores[label][idx] = max(
                event_scores[label][idx],
                max(math.exp(-0.5 * ((time_ms - center) / 9.0) ** 2) for center in centers),
            )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    rows = [
        parse_template_oto_line("v.wav=あ,0,0,0,0,0"),
        parse_template_oto_line("v.wav=a あ,0,0,0,0,0"),
        parse_template_oto_line("v.wav=a い,0,0,0,0,0"),
    ]
    assert all(row is not None for row in rows)
    anchors = assign_template_row_anchors(posterior, [], [row for row in rows if row is not None])
    assert len(anchors) == 3
    assert all(anchor is not None for anchor in anchors)
    assert [anchor.anchor_abs_ms for anchor in anchors if anchor is not None] == sorted(
        anchor.anchor_abs_ms for anchor in anchors if anchor is not None
    )
    assert {anchor.role for anchor in anchors if anchor is not None}.issubset({"v", "vv", "vcv"})


def test_template_anchor_assignment_can_ignore_bad_source_timing_for_slots():
    times = [float(idx * 20) for idx in range(151)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    decoded = []
    for slot_idx, center in enumerate([420.0, 840.0, 1260.0]):
        frame_index = min(range(len(times)), key=lambda idx: abs(times[idx] - center))
        event_scores["cv_boundary"][frame_index] = 0.95
        class_probs["vowel"][frame_index] = 0.85
        decoded.append(
            {
                "label": "cv_boundary",
                "selected_time_ms": center,
                "score": 0.95,
                "frame_index": frame_index,
                "expected_phone": ["a", "i", "u"][slot_idx],
                "expected_phone_index": [1, 3, 5][slot_idx],
                "slot_index": slot_idx,
            }
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    rows = [
        parse_template_oto_line("v.wav=ka,2200,160,-520,120,85"),
        parse_template_oto_line("v.wav=ki,2300,160,-520,120,85"),
        parse_template_oto_line("v.wav=ku,2400,160,-520,120,85"),
    ]
    assert all(row is not None for row in rows)
    anchors = assign_template_row_anchors(
        posterior,
        decoded,
        [row for row in rows if row is not None],
        use_source_timing_prior=False,
        expected_phones=["k", "a", "k", "i", "k", "u"],
    )
    assert [anchor.anchor_abs_ms for anchor in anchors if anchor is not None] == pytest.approx([420.0, 840.0, 1260.0])
    assert all("slot_decoded_event" in anchor.warnings for anchor in anchors if anchor is not None)


def test_template_anchor_assignment_records_local_refine_delta():
    times = [float(idx * 10) for idx in range(21)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.90 if 50.0 <= time_ms <= 90.0 else 0.05
        class_probs["consonant"][idx] = 0.45 if 30.0 <= time_ms <= 60.0 else 0.05
    event_scores["cv_boundary"][5] = 0.20
    event_scores["cv_boundary"][6] = 0.96
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    row = parse_template_oto_line("v.wav=ka,0,0,0,0,0")
    assert row is not None

    anchors = assign_template_row_anchors(
        posterior,
        [
            {
                "label": "cv_boundary",
                "selected_time_ms": 50.0,
                "score": 0.92,
                "frame_index": 5,
                "expected_phone": "a",
                "expected_phone_index": 1,
                "slot_index": 0,
            }
        ],
        [row],
        use_source_timing_prior=False,
        expected_phones=["k", "a"],
    )

    anchor = anchors[0]
    assert anchor is not None
    assert anchor.anchor_abs_ms == pytest.approx(60.0)
    assert anchor.refined_from_abs_ms == pytest.approx(50.0)
    assert anchor.local_refine_delta_ms == pytest.approx(10.0)
    assert anchor.local_refine_margin is not None
    assert "local_refined_anchor:50.0->60.0" in anchor.warnings
    assert "local_refine_delta_ms:10.0" in anchor.warnings
    assert "local_refine_changed_anchor" not in anchor.warnings


def test_cvvc_vc_and_next_cv_use_separate_filename_slots():
    times = [float(idx * 20) for idx in range(81)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    expected = ["v", "a", "v", "i", "v", "u"]
    decoded = []
    slot_defs = [
        (1, "a", "cv_boundary", 200.0),
        (2, "v", "phone_change", 380.0),
        (3, "i", "cv_boundary", 500.0),
        (4, "v", "phone_change", 680.0),
        (5, "u", "cv_boundary", 800.0),
    ]
    for slot_index, (phone_index, phone, label, center) in enumerate(slot_defs):
        frame_index = min(range(len(times)), key=lambda idx: abs(times[idx] - center))
        event_scores[label][frame_index] = 0.95
        class_probs["vowel"][frame_index] = 0.85
        decoded.append(
            {
                "label": label,
                "selected_time_ms": center,
                "score": 0.95,
                "frame_index": frame_index,
                "expected_phone": phone,
                "expected_phone_index": phone_index,
                "slot_index": slot_index,
            }
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    rows = [
        parse_template_oto_line("v.wav=ヴぁ,0,0,0,0,0"),
        parse_template_oto_line("v.wav=a v,0,0,0,0,0"),
        parse_template_oto_line("v.wav=ヴぃ,0,0,0,0,0"),
        parse_template_oto_line("v.wav=i v,0,0,0,0,0"),
        parse_template_oto_line("v.wav=ヴ,0,0,0,0,0"),
    ]
    anchors = assign_template_row_anchors(
        posterior,
        decoded,
        [row for row in rows if row is not None],
        use_source_timing_prior=False,
        expected_phones=expected,
    )
    assert [anchor.anchor_abs_ms for anchor in anchors if anchor is not None] == pytest.approx(
        [200.0, 380.0, 500.0, 680.0, 800.0]
    )
    assert [anchor.expected_phone_index for anchor in anchors if anchor is not None] == [1, 2, 3, 4, 5]
    assert [anchor.role for anchor in anchors if anchor is not None] == ["cv", "vc", "cv", "vc", "cv"]


def test_template_preserve_allows_large_filename_slot_reanchor():
    row = parse_template_oto_line("v.wav=ka,2200,160,-520,120,85")
    assert row is not None
    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=420.0, score=0.95, warnings=("slot_decoded_event",)),
        file_duration_ms=3000.0,
        config=OtoAdapterConfig(language="japanese", format_type="CV", alias_type="auto", max_anchor_shift_ms=160.0),
    )
    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(420.0, abs=35.0)
    assert adapted.source_timing is None
    assert "source_timing_discarded" in adapted.warnings


def test_vc_bootstrap_cuts_before_next_cv_vowel_body():
    adapted = bootstrap_row(
        "v.wav",
        "a v",
        OtoAnchor(anchor_abs_ms=1365.0, score=0.95, vowel_end_abs_ms=1796.0),
        file_duration_ms=4600.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )
    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(1365.0)
    assert absolute["cutoff_abs"] < 1435.0
    assert absolute["cutoff_abs"] < 1796.0
    assert adapted.timing.cutoff == pytest.approx(-(adapted.timing.consonant + 26.0))


def test_zero_template_uses_bootstrap_fallback_instead_of_preserving_zeros():
    row = parse_template_oto_line("a-ka.wav=a ka,0,0,0,0,0")
    assert row is not None
    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=240.0, score=0.0, warnings=("synthetic_anchor:no_candidate",)),
        file_duration_ms=700.0,
        config=OtoAdapterConfig(language="japanese", format_type="VCV", alias_type="auto"),
    )
    absolute = adapted.to_json_dict()["absolute"]
    assert adapted.mode == "template-bootstrap"
    assert absolute["preutterance_abs"] == pytest.approx(240.0)
    assert adapted.timing.consonant > adapted.timing.preutterance
    assert adapted.timing.overlap <= adapted.timing.preutterance
    assert adapted.timing.cutoff < 0.0
    assert "zero_template_bootstrap" in adapted.warnings
    assert any(warning.startswith("low_anchor_score:") for warning in adapted.warnings)


def test_vcv_bootstrap_clamps_impossible_previous_vowel_end():
    adapted = bootstrap_row(
        "vcv.wav",
        "i myo",
        OtoAnchor(
            anchor_abs_ms=925.0,
            score=0.42,
            previous_vowel_end_abs_ms=2170.0,
            vowel_end_abs_ms=1500.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=3000.0,
        config=OtoAdapterConfig(language="korean", format_type="vcv", alias_type="auto"),
    )

    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["offset_abs"] < absolute["preutterance_abs"]
    assert adapted.timing.preutterance > 0.0
    assert adapted.timing.consonant > adapted.timing.preutterance
    assert any(warning.startswith("vcv_previous_vowel_end_clamped:") for warning in adapted.warnings)


def test_japanese_vcv_bootstrap_uses_role_specific_transition_profile():
    adapted = bootstrap_row(
        "vcv.wav",
        "a nya",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            previous_vowel_end_abs_ms=480.0,
            vowel_end_abs_ms=920.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=1200.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="vcv"),
    )

    assert adapted.timing.preutterance == pytest.approx(150.0)
    assert adapted.timing.overlap == pytest.approx(50.0)
    assert adapted.timing.consonant == pytest.approx(250.0)


def test_korean_vcv_bootstrap_uses_long_transition_profile():
    adapted = bootstrap_row(
        "vcv.wav",
        "a gyeo",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            previous_vowel_end_abs_ms=480.0,
            vowel_end_abs_ms=980.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="korean", format_type="vcv", alias_type="vcv"),
    )

    assert adapted.timing.preutterance == pytest.approx(300.0)
    assert adapted.timing.overlap == pytest.approx(80.0)
    assert adapted.timing.consonant == pytest.approx(450.0)


def test_korean_vcv_bootstrap_uses_vv_transition_profile():
    adapted = bootstrap_row(
        "vcv.wav",
        "yu yeo",
        OtoAnchor(
            anchor_abs_ms=720.0,
            score=0.72,
            previous_vowel_end_abs_ms=520.0,
            vowel_end_abs_ms=1120.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=1500.0,
        config=OtoAdapterConfig(language="korean", format_type="vcv", alias_type="vv"),
    )

    assert adapted.timing.preutterance == pytest.approx(360.0)
    assert adapted.timing.overlap == pytest.approx(100.0)
    assert adapted.timing.consonant == pytest.approx(505.0)


def test_vcv_cv_head_bootstrap_uses_initial_row_profile():
    adapted = bootstrap_row(
        "vcv.wav",
        "- ka",
        OtoAnchor(anchor_abs_ms=320.0, score=0.72, vowel_end_abs_ms=620.0),
        file_duration_ms=900.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="cv_head"),
    )

    assert adapted.timing.preutterance == pytest.approx(82.0)
    assert adapted.timing.overlap == pytest.approx(29.0)
    assert adapted.timing.consonant == pytest.approx(160.0)


def test_japanese_vcv_bootstrap_clamps_excessive_cutoff_tail():
    adapted = bootstrap_row(
        "vcv.wav",
        "a nya",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            previous_vowel_end_abs_ms=480.0,
            vowel_end_abs_ms=4300.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=4500.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="auto"),
    )

    assert abs(adapted.timing.cutoff) == pytest.approx(650.0)
    assert adapted.timing.consonant < abs(adapted.timing.cutoff)
    assert any(warning.startswith("bootstrap_cutoff_tail_clamped:") for warning in adapted.warnings)


def test_non_vcv_bootstrap_keeps_long_cutoff_tail():
    adapted = bootstrap_row(
        "cv.wav",
        "nya",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            vowel_end_abs_ms=4300.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=4500.0,
        config=OtoAdapterConfig(language="japanese", format_type="cv", alias_type="auto"),
    )

    assert abs(adapted.timing.cutoff) > 3000.0
    assert not any(warning.startswith("bootstrap_cutoff_tail_clamped:") for warning in adapted.warnings)


def test_slot_viterbi_reports_no_monotonic_path_for_shifted_candidates():
    times = [float(idx * 10) for idx in range(31)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.9
        event_scores["cv_boundary"][idx] = math.exp(-0.5 * ((time_ms - 200.0) / 8.0) ** 2)
        event_scores["vowel_nucleus"][idx] = math.exp(-0.5 * ((time_ms - 100.0) / 8.0) ** 2)
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    result = assign_slots_viterbi(
        posterior,
        expected_phones=["k", "a"],
        min_event_score=0.05,
        same_phone_min_gap_ms=20.0,
    )
    assert not result.ok
    assert "hard_no_monotonic_path" in result.warnings


def test_review_overlay_contains_manual_and_prediction_tracks(tmp_path):
    wav_path = tmp_path / "001_ka.wav"
    _write_tone_wav(wav_path)
    row = _labelled_row("manual_gold")
    row["wav_path"] = str(wav_path)
    times = [float(idx * 10) for idx in range(20)]
    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=times,
        class_probs={label: [0.25 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.1 for _ in times] for label in EVENT_LABELS},
    )
    html = render_review_html(row, posterior=posterior)
    assert "Manual frame labels" in html
    assert "Predicted posteriors" in html
    assert "cv_boundary" in html
    assert "spectrogram" in html.lower()


def test_oto_adapter_reanchors_template_row_to_predicted_anchor():
    row = parse_template_oto_line("a-ka.wav=a ka,100,90,-260,60,25")
    assert row is not None
    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=190.0, score=0.92),
        file_duration_ms=480.0,
        config=OtoAdapterConfig(language="japanese", format_type="CV", alias_type="cv"),
    )
    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(190.0, abs=35.0)
    assert adapted.timing.overlap <= adapted.timing.preutterance
    assert adapted.timing.consonant > adapted.timing.preutterance
    assert adapted.format_line().startswith("a-ka.wav=a ka,")


def test_oto_adapter_bootstraps_cv_row_from_anchor():
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        OtoAnchor(anchor_abs_ms=210.0, score=0.84, vowel_end_abs_ms=340.0),
        file_duration_ms=520.0,
        config=OtoAdapterConfig(mode="bootstrap", alias_type="cv"),
    )
    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(210.0)
    assert adapted.timing.offset == pytest.approx(90.0)
    assert adapted.timing.consonant > adapted.timing.preutterance
    assert adapted.timing.cutoff < 0.0


def test_oto_adapter_records_large_timing_clamp_warning():
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        OtoAnchor(anchor_abs_ms=210.0, score=0.84, vowel_end_abs_ms=340.0),
        file_duration_ms=520.0,
        config=OtoAdapterConfig(mode="bootstrap", alias_type="cv", cons_gap_ms=-100.0),
    )

    assert adapted.timing.consonant == pytest.approx(adapted.timing.preutterance + 8.0)
    assert "timing_clamped.consonant" in adapted.warnings
    assert "timing_clamp_large_delta" in adapted.warnings
    assert any(warning.startswith("timing_clamp_delta_ms:") for warning in adapted.warnings)


def test_review_overlay_renders_generated_oto_params(tmp_path):
    wav_path = tmp_path / "001_ka.wav"
    _write_tone_wav(wav_path)
    row = _labelled_row("manual_gold")
    row["wav_path"] = str(wav_path)
    html = render_review_html(
        row,
        generated_oto_rows=[
            {
                "alias": "ka",
                "absolute": {
                    "offset_abs": 20.0,
                    "overlap_abs": 70.0,
                    "preutterance_abs": 100.0,
                    "consonant_abs": 135.0,
                    "cutoff_abs": 260.0,
                },
            }
        ],
    )
    assert "Generated OTO params" in html
    assert "preutterance" in html


def test_model_forward_shape_when_torch_available():
    torch = pytest.importorskip("torch")
    from core.mfa_free_oto.model import MfaFreeFrameModelConfig, build_frame_model

    model = build_frame_model(MfaFreeFrameModelConfig(input_dim=8, hidden_dim=16, layers=1))
    out = model(torch.zeros((2, 12, 8), dtype=torch.float32))
    assert tuple(out["frame_logits"].shape) == (2, 12, len(FRAME_LABELS))
    assert tuple(out["event_logits"].shape) == (2, 12, len(EVENT_LABELS))


def test_htk_lab_manifest_uses_lab_as_manual_gold(tmp_path):
    lab_root = tmp_path / "lab"
    lab_root.mkdir()
    wav_path = lab_root / "a-ka.wav"
    lab_path = lab_root / "a-ka.lab"
    _write_tone_wav(wav_path)
    lab_path.write_text(
        "\n".join(
            [
                "0 5000000 AP",
                "5000000 6000000 vf",
                "6000000 9000000 a",
                "9000000 10000000 k",
                "10000000 14000000 a",
                "14000000 15000000 exh",
                "15000000 16000000 SP",
            ]
        ),
        encoding="utf-8",
    )
    segments, dropped = parse_htk_lab(lab_path)
    assert len(dropped) == 1
    assert classify_htk_phone("AP") == "silence"
    assert classify_htk_phone("exh") == "other"
    assert classify_htk_phone("ɯ") == "vowel"
    assert [segment.frame_label for segment in segments] == [
        "silence",
        "vowel",
        "consonant",
        "vowel",
        "other",
        "silence",
    ]

    row = row_from_htk_lab(lab_path, language="japanese", format_type="CV")
    assert row["label_source"] == "manual_gold"
    assert row["label_format"] == "htk_lab"
    assert row["expected_phones"] == ["a", "k", "a", "exh"]
    assert any(event["label"] == "cv_boundary" and event["time_ms"] == pytest.approx(1000.0) for event in row["events"])
    assert row["dropped_lab_segments"][0]["phone"] == "vf"

    out = tmp_path / "gold.jsonl"
    rows, summary = build_gold_manifest_from_htk_lab_dirs(
        [lab_root],
        out_path=out,
        language="japanese",
        format_type="CV",
    )
    assert summary.rows == 1
    assert summary.dropped_segments == 1
    assert rows[0]["wav_name"] == "a-ka.wav"
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["label_source"] == "manual_gold"


def test_model_context_runtime_workflow_generates_oto_and_quality_meta(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / "a.wav"
    _write_tone_wav(wav_path, duration_s=0.40)
    source_oto = tmp_path / "source.ini"
    source_oto.write_text("a.wav=ka,0,0,0,0,0\n", encoding="utf-8")
    out_oto = tmp_path / "out.ini"

    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=[0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        class_probs={label: [0.2] * 7 for label in FRAME_LABELS},
        event_scores={label: [0.0] * 7 for label in EVENT_LABELS},
    )
    prediction = type("RuntimePredictionStub", (), {})()
    prediction.posterior = posterior
    prediction.decoded_events = (
        DecodedEvent(label="cv_boundary", selected_time_ms=120.0, score=0.88, frame_index=2),
    )
    prediction.slot_result = SlotViterbiResult(
        assignments=(
            SlotAssignment(
                slot_index=0,
                phone_index=1,
                phone="a",
                role="cv_boundary",
                event_label="cv_boundary",
                selected_time_ms=120.0,
                score=0.88,
                frame_index=2,
                expected_time_ms=120.0,
            ),
        ),
        path_score=0.88,
        average_score=0.88,
        warnings=(),
    )

    monkeypatch.setattr("core.mfa_free_oto.workflow.predict_wav", lambda *args, **kwargs: prediction)

    report = generate_no_mfa_oto_with_model_context(
        wav_dir=str(wav_dir),
        out_path=str(out_oto),
        source_oto_path=str(source_oto),
        checkpoint_path=str(tmp_path / "dummy.pt"),
        language="japanese",
        format_type="CV",
    )
    assert report.processed == 1
    assert report.total == 1
    assert not report.errors
    assert report.confidence > 0.5
    assert out_oto.is_file()
    assert "a.wav=" in out_oto.read_text(encoding="utf-8")
    assert report.rows
    evidence = report.rows[0].boundary_evidence
    assert 0.0 <= evidence.cv_confidence <= 1.0
    assert 0.0 <= evidence.nucleus_confidence <= 1.0


def test_model_context_runtime_workflow_ignores_source_timing_values(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / "a.wav"
    _write_tone_wav(wav_path, duration_s=0.40)
    source_oto_a = tmp_path / "source_a.ini"
    source_oto_b = tmp_path / "source_b.ini"
    source_oto_a.write_text("a.wav=ka,0,0,0,0,0\n", encoding="utf-8")
    source_oto_b.write_text("a.wav=ka,1234,987,-654,321,210\n", encoding="utf-8")
    out_a = tmp_path / "out_a.ini"
    out_b = tmp_path / "out_b.ini"

    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=[0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        class_probs={label: [0.2] * 7 for label in FRAME_LABELS},
        event_scores={label: [0.0] * 7 for label in EVENT_LABELS},
        acoustic_scores={
            "transition_likelihood": [0.2, 0.2, 0.9, 0.3, 0.2, 0.2, 0.2],
            "voicing": [0.1, 0.2, 0.5, 0.9, 0.7, 0.3, 0.2],
            "nucleus_likelihood": [0.1, 0.2, 0.4, 0.95, 0.8, 0.3, 0.2],
            "silence_likelihood": [0.8, 0.5, 0.2, 0.1, 0.2, 0.4, 0.7],
        },
    )
    prediction = type("RuntimePredictionStub", (), {})()
    prediction.posterior = posterior
    prediction.decoded_events = (
        DecodedEvent(label="cv_boundary", selected_time_ms=120.0, score=0.88, frame_index=2),
        DecodedEvent(label="vowel_nucleus", selected_time_ms=160.0, score=0.91, frame_index=3),
    )
    prediction.slot_result = None
    monkeypatch.setattr("core.mfa_free_oto.workflow.predict_wav", lambda *args, **kwargs: prediction)

    report_a = generate_no_mfa_oto_with_model_context(
        wav_dir=str(wav_dir),
        out_path=str(out_a),
        source_oto_path=str(source_oto_a),
        checkpoint_path="",
        language="japanese",
        format_type="CV",
    )
    report_b = generate_no_mfa_oto_with_model_context(
        wav_dir=str(wav_dir),
        out_path=str(out_b),
        source_oto_path=str(source_oto_b),
        checkpoint_path="",
        language="japanese",
        format_type="CV",
    )
    assert report_a.processed == 1 and report_b.processed == 1
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")
    assert report_a.rows and report_b.rows
    assert report_a.rows[0].oto_params["preutterance"] >= 0.0
    be = report_a.rows[0].boundary_evidence
    if be.c_onset is not None and be.cv_boundary is not None:
        assert be.c_onset <= be.cv_boundary
    if be.cv_boundary is not None and be.nucleus is not None:
        assert be.cv_boundary <= be.nucleus
    assert 0.0 <= be.cv_confidence <= 1.0
    assert 0.0 <= be.nucleus_confidence <= 1.0


def test_mfa_free_preview_bootstrap_uses_filename_row_plan(tmp_path, monkeypatch):
    from dataclasses import replace

    from ml.scripts.mfa_free_oto import predict_oto_preview

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "ka_ki.wav"
    _write_tone_wav(wav_path, duration_s=0.45)
    times = [float(idx * 40) for idx in range(12)]
    event_scores = {label: [0.0 for _ in times] for label in EVENT_LABELS}
    event_scores["cv_boundary"][2] = 0.95
    event_scores["vowel_nucleus"][4] = 0.90
    event_scores["phone_change"][7] = 0.85
    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=times,
        class_probs={
            "silence": [0.02 for _ in times],
            "consonant": [0.85 if idx in {1, 2, 7} else 0.10 for idx, _ in enumerate(times)],
            "vowel": [0.85 if idx in {3, 4, 5, 8, 9} else 0.10 for idx, _ in enumerate(times)],
            "other": [0.03 for _ in times],
        },
        event_scores=event_scores,
        acoustic_scores={},
        metadata={
            "rule_based": True,
            "rule_fallback_reason": "checkpoint_missing_or_unset",
        },
    )
    prediction = type("PreviewPredictionStub", (), {})()
    prediction.posterior = posterior
    prediction.slot_result = None
    prediction.decoded_events = [
        {
            "label": "cv_boundary",
            "selected_time_ms": 80.0,
            "score": 0.95,
            "frame_index": 2,
            "expected_phone": "a",
            "expected_phone_index": 1,
            "slot_index": 0,
        },
        {
            "label": "vowel_nucleus",
            "selected_time_ms": 160.0,
            "score": 0.90,
            "frame_index": 4,
            "expected_phone": "a",
            "expected_phone_index": 1,
            "slot_index": 1,
        },
        {
            "label": "phone_change",
            "selected_time_ms": 280.0,
            "score": 0.85,
            "frame_index": 7,
            "expected_phone": "k",
            "expected_phone_index": 2,
            "slot_index": 2,
        },
    ]
    prediction.to_json_dict = lambda: {"stub": True}
    predict_kwargs = {}

    def fake_predict_wav(*args, **kwargs):
        predict_kwargs.update(kwargs)
        return prediction

    monkeypatch.setattr(predict_oto_preview, "predict_wav", fake_predict_wav)
    original_adapt_template_row = predict_oto_preview.adapt_template_row

    def adapt_with_warning(*args, **kwargs):
        row = original_adapt_template_row(*args, **kwargs)
        return replace(
            row,
            warnings=tuple(
                dict.fromkeys(
                    (
                        *row.warnings,
                        "timing_clamped.offset",
                        "timing_clamp_delta_ms:42.0",
                        "timing_clamp_large_delta",
                    )
                )
            ),
        )

    monkeypatch.setattr(predict_oto_preview, "adapt_template_row", adapt_with_warning)
    out_oto = tmp_path / "preview.ini"
    out_json = tmp_path / "anchors.json"
    review_dir = tmp_path / "review"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_oto_preview",
            "--wav-dir",
            str(wav_dir),
            "--out-oto",
            str(out_oto),
            "--out-json",
            str(out_json),
            "--review-out-dir",
            str(review_dir),
            "--review-name",
            "preview_review",
            "--format-type",
            "CVVC",
        ],
    )

    assert predict_oto_preview.main() == 0
    assert predict_kwargs["checkpoint_path"] == ""
    lines = out_oto.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3
    assert any(line.startswith("ka_ki.wav=ka,") for line in lines)
    assert any(line.startswith("ka_ki.wav=a k,") for line in lines)
    assert any(line.startswith("ka_ki.wav=ki,") for line in lines)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["rows"][0]["row_plan"]
    summary = json.loads((review_dir / "preview_review.summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["total"] == len(lines)
    assert summary["reason_counts"]["attention.timing_clamp_large_delta"] >= 1
    assert summary["reason_counts"]["attention.rule_based_inference"] == len(lines)
    assert summary["metric_positive_row_counts"]["timing_clamp_large_delta_count"] >= 1
    assert summary["metric_positive_row_counts"]["rule_based_inference_count"] == len(lines)
    assert summary["metric_positive_row_counts"]["rule_based_checkpoint_missing_count"] == len(lines)
    session = json.loads((review_dir / "preview_review.review_session.json").read_text(encoding="utf-8"))
    assert session["metadata"]["created_by"] == "predict_oto_preview"
    assert session["metadata"]["pipeline"] == "ui_preview"
    assert session["metadata"]["checkpoint"] == ""
    assert session["metadata"]["source_timing_trusted"] is False
    assert session["lineage"]["validation_jsonl_sha256"] == summary["validation_jsonl_sha256"]
    assert session["apply_policy"]["requires_manual_merge"] is True
    assert (review_dir / "preview_review.fix_required.ini").is_file()
    assert (review_dir / "preview_review.attention_only.ini").is_file()
    assert (review_dir / "preview_review.clean.ini").is_file()
    assert (review_dir / "preview_review.review_all.ini").is_file()
    assert (review_dir / "preview_review.validation.jsonl").is_file()
    assert (review_dir / "preview_review.review_session.json").is_file()


def test_mfa_free_preview_split_summary_log_is_readable():
    from ui.pipeline_actions_mixin import PipelineActionsMixin

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.logs = []

        def _append_log(self, message):
            self.logs.append(str(message))

    dummy = Dummy()
    payload = {
        "wav_count": 1,
        "oto_rows": 3,
        "review_split_counts": {
            "total": 3,
            "fix_required": 1,
            "attention_only": 1,
            "clean": 1,
        },
        "review_split_output_paths": {
            "fix_required": "review.fix_required.ini",
            "attention_only": "review.attention_only.ini",
            "clean": "review.clean.ini",
            "review_all": "review.review_all.ini",
            "validation_jsonl": "review.validation.jsonl",
            "summary": "review.summary.json",
            "review_session": "review.review_session.json",
        },
    }

    parsed = dummy._parse_mfa_free_preview_summary(
        [
            "noise before json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "noise after json",
        ]
    )
    dummy._log_mfa_free_preview_split_summary(parsed)

    assert any(
        "review split counts: fix_required=1, attention_only=1, clean=1, total=3" in log
        for log in dummy.logs
    )
    assert any("fix_required: review.fix_required.ini" in log for log in dummy.logs)
    assert any("validation_jsonl: review.validation.jsonl" in log for log in dummy.logs)
    assert any("review_session: review.review_session.json" in log for log in dummy.logs)


def test_mfa_free_ui_preview_allows_missing_checkpoint_rule_based(tmp_path):
    from ui.pipeline_actions_mixin import PipelineActionsMixin

    class FakeProcess:
        returncode = 0

        def wait(self):
            return 0

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.logs = []
            self.commands = []
            self.writable_data_dir = str(tmp_path)
            self.app_data_dir = ""
            self.app_dir = str(tmp_path)

        def _resolve_mfa_free_oto_checkpoint(self):
            return ""

        def _append_log(self, message):
            self.logs.append(str(message))

        def _popen_subprocess_hidden(self, cmd, **kwargs):
            self.commands.append(list(cmd))
            return FakeProcess()

        def _iter_decoded_stdout_lines(self, process):
            return iter(
                [
                    json.dumps(
                        {
                            "wav_count": 1,
                            "oto_rows": 2,
                            "review_split_counts": {
                                "total": 2,
                                "fix_required": 0,
                                "attention_only": 1,
                                "clean": 1,
                            },
                            "review_split_output_paths": {
                                "review_session": str(tmp_path / "review_session.json"),
                            },
                        },
                        ensure_ascii=False,
                    )
                ]
            )

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    out_path = tmp_path / "preview.ini"
    dummy = Dummy()

    generated, total, errors = dummy._run_mfa_free_oto_preview_generation(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        language="japanese",
        format_type="CVVC",
    )

    assert errors == []
    assert generated == 2
    assert total == 2
    assert dummy.commands
    command = dummy.commands[0]
    assert "--checkpoint" not in command
    assert "--review-out-dir" in command
    assert any("checkpoint: none; using rule-based acoustic fallback" in log for log in dummy.logs)
    assert any("review_session:" in log for log in dummy.logs)


def test_runtime_rejects_checkpoint_metadata_mismatch_and_falls_back_rule_based(tmp_path, monkeypatch):
    wav_path = tmp_path / "a.wav"
    _write_tone_wav(wav_path, duration_s=0.30)
    checkpoint_path = tmp_path / "mismatch.pt"
    checkpoint_path.write_bytes(b"stub")

    fake_checkpoint = {
        "format_version": "mfa_free_oto_frame_model_v1",
        "acoustic_feature_set": "world_v1",
        "frame_labels": list(FRAME_LABELS),
        "event_labels": list(EVENT_LABELS),
        "acoustic_config": {"frame_ms": 25.0, "hop_ms": 10.0},
        "encoder": "acoustic_world_v1",
    }
    called = {"model_infer": False}

    def _fake_load(*_args, **_kwargs):
        return fake_checkpoint, object(), "cpu"

    def _fake_model_infer(*_args, **_kwargs):
        called["model_infer"] = True
        raise AssertionError("model inference should be skipped when metadata mismatches")

    monkeypatch.setattr(runtime_inference_module, "_load_runtime_checkpoint", _fake_load)
    monkeypatch.setattr(runtime_inference_module, "_predict_posterior_with_loaded_model", _fake_model_infer)

    result = runtime_inference_module.predict_wav(
        wav_path,
        checkpoint_path=checkpoint_path,
        encoder="acoustic_world_v1",
        use_slot_viterbi=False,
    )
    assert called["model_infer"] is False
    assert bool(result.posterior.metadata.get("rule_based")) is True
    reason = str(result.posterior.metadata.get("rule_fallback_reason") or "")
    assert "checkpoint_inference_failed" in reason
    assert "checkpoint_format_mismatch" in reason


def test_read_wav_mono_supports_24bit_pcm(tmp_path):
    from core.mfa_free_oto.features import read_wav_mono

    wav_path = tmp_path / "pcm24.wav"
    values = [-8388608, -1024, 0, 1024, 8388607]
    payload = bytearray()
    for value in values:
        raw = int(value).to_bytes(4, byteorder="little", signed=True)
        payload.extend(raw[:3])
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(3)
        handle.setframerate(16000)
        handle.writeframes(bytes(payload))

    samples, sample_rate = read_wav_mono(wav_path)
    assert sample_rate == 16000
    assert samples.shape == (len(values),)
    assert samples[0] == pytest.approx(-1.0)
    assert samples[-1] == pytest.approx(8388607 / 8388608.0)


def test_read_wav_mono_supports_ieee_float_wav(tmp_path):
    from core.mfa_free_oto.features import read_wav_mono

    wav_path = tmp_path / "float32.wav"
    values = np.asarray([-1.0, -0.25, 0.25, 1.0], dtype="<f4")
    fmt = struct.pack("<HHIIHH", 3, 1, 22050, 22050 * 4, 4, 32)
    data = values.tobytes()
    payload = (
        b"RIFF"
        + (4 + (8 + len(fmt)) + (8 + len(data))).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + len(fmt).to_bytes(4, "little")
        + fmt
        + b"data"
        + len(data).to_bytes(4, "little")
        + data
    )
    wav_path.write_bytes(payload)

    samples, sample_rate = read_wav_mono(wav_path)
    assert sample_rate == 22050
    assert samples.tolist() == pytest.approx([-1.0, -0.25, 0.25, 1.0])


def test_manual_oto_monotonic_decoder_prefers_ordered_path():
    from core.mfa_free_oto.manual_oto_decoder import CandidateOption, decode_monotonic_candidate_indices

    rows = [
        [
            CandidateOption(candidate_index=0, time_ms=100.0, score=0.60, order_norm=0.0, slot_pos_norm=0.0),
            CandidateOption(candidate_index=9, time_ms=900.0, score=0.95, order_norm=1.0, slot_pos_norm=0.0),
        ],
        [
            CandidateOption(candidate_index=1, time_ms=200.0, score=0.90, order_norm=0.0, slot_pos_norm=0.5),
        ],
        [
            CandidateOption(candidate_index=2, time_ms=300.0, score=0.92, order_norm=0.0, slot_pos_norm=1.0),
        ],
    ]
    selected = decode_monotonic_candidate_indices(rows, order_penalty=0.0)
    assert selected == [0, 1, 2]


def test_manual_oto_joint_anchor_lattice_enforces_anchor_order():
    from core.mfa_free_oto.manual_oto_decoder import (
        CandidateOption,
        build_joint_anchor_options,
        decode_joint_anchor_lattice,
    )

    row0 = build_joint_anchor_options(
        {
            "offset": [
                CandidateOption(candidate_index=10, time_ms=180.0, score=0.99),
                CandidateOption(candidate_index=1, time_ms=100.0, score=0.80),
            ],
            "overlap": [CandidateOption(candidate_index=2, time_ms=120.0, score=0.80)],
            "preutterance": [CandidateOption(candidate_index=3, time_ms=150.0, score=0.80)],
            "fixed_end": [CandidateOption(candidate_index=4, time_ms=210.0, score=0.80)],
        },
        top_per_anchor=2,
    )
    assert row0
    assert row0[0].anchor_indices["offset"] == 1
    assert row0[0].anchor_times_ms["offset"] <= row0[0].anchor_times_ms["overlap"]

    row1 = build_joint_anchor_options(
        {
            "offset": [CandidateOption(candidate_index=5, time_ms=300.0, score=0.80)],
            "overlap": [CandidateOption(candidate_index=6, time_ms=320.0, score=0.80)],
            "preutterance": [CandidateOption(candidate_index=7, time_ms=350.0, score=0.80)],
            "fixed_end": [CandidateOption(candidate_index=8, time_ms=410.0, score=0.80)],
        }
    )
    decoded = decode_joint_anchor_lattice([row0, row1])
    assert decoded[0] is not None
    assert decoded[1] is not None
    assert decoded[0].center_time_ms <= decoded[1].center_time_ms


def test_manual_oto_joint_lattice_prefers_filename_time_order():
    from core.mfa_free_oto.manual_oto_decoder import JointAnchorOption, decode_joint_anchor_lattice

    rows = [
        [
            JointAnchorOption({"preutterance": 10}, {"preutterance": 800.0}, score=2.0, time_order_norm=0.80, slot_pos_norm=0.20),
            JointAnchorOption({"preutterance": 1}, {"preutterance": 200.0}, score=1.7, time_order_norm=0.20, slot_pos_norm=0.20),
        ],
        [
            JointAnchorOption({"preutterance": 2}, {"preutterance": 250.0}, score=2.0, time_order_norm=0.25, slot_pos_norm=0.80),
            JointAnchorOption({"preutterance": 11}, {"preutterance": 820.0}, score=1.7, time_order_norm=0.82, slot_pos_norm=0.80),
        ],
    ]
    decoded = decode_joint_anchor_lattice(rows, time_backward_penalty=12.0, slot_order_penalty=2.0, slot_transition_penalty=2.0)
    assert decoded[0] is not None
    assert decoded[1] is not None
    assert decoded[0].anchor_indices["preutterance"] == 1
    assert decoded[1].anchor_indices["preutterance"] == 11


def test_manual_oto_filename_ordered_rows_use_filename_tokens():
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _filename_ordered_rows

    rows = [
        {"wav_name": "ka-ki-ku.wav", "alias": "ku", "slot_index": 0, "slot_count": 3, "language": "japanese"},
        {"wav_name": "ka-ki-ku.wav", "alias": "ka", "slot_index": 1, "slot_count": 3, "language": "japanese"},
        {"wav_name": "ka-ki-ku.wav", "alias": "ki", "slot_index": 2, "slot_count": 3, "language": "japanese"},
    ]
    ordered = _filename_ordered_rows(rows, require_primary_transition_token=True)
    assert [row["alias"] for row, _pos in ordered] == ["ka", "ki", "ku"]


def test_manual_oto_filename_order_ignores_transition_fallback_only_match():
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _filename_ordered_rows

    rows = [
        {
            "wav_name": "_n-ma-mi-mu-me-mo-mu.wav",
            "filename_canonical_tokens": ["n", "ma", "mi", "mu", "me", "mo", "mu"],
            "alias": "n m_E4",
            "alias_role": "vc",
            "format_type": "cvvc",
            "slot_index": 0,
            "slot_count": 13,
            "language": "japanese",
        },
        {
            "wav_name": "_n-ma-mi-mu-me-mo-mu.wav",
            "filename_canonical_tokens": ["n", "ma", "mi", "mu", "me", "mo", "mu"],
            "alias": "u m_E4",
            "alias_role": "vc",
            "format_type": "cvvc",
            "slot_index": 5,
            "slot_count": 13,
            "language": "japanese",
        },
        {
            "wav_name": "_n-ma-mi-mu-me-mo-mu.wav",
            "filename_canonical_tokens": ["n", "ma", "mi", "mu", "me", "mo", "mu"],
            "alias": "n my_E4",
            "alias_role": "vc",
            "format_type": "cvvc",
            "slot_index": 10,
            "slot_count": 13,
            "language": "japanese",
        },
        {
            "wav_name": "_n-ma-mi-mu-me-mo-mu.wav",
            "filename_canonical_tokens": ["n", "ma", "mi", "mu", "me", "mo", "mu"],
            "alias": "i my_E4",
            "alias_role": "vc",
            "format_type": "cvvc",
            "slot_index": 11,
            "slot_count": 13,
            "language": "japanese",
        },
    ]
    ordered = _filename_ordered_rows(rows, require_primary_transition_token=True)
    aliases = [row["alias"] for row, _pos in ordered]
    assert aliases.index("n my_E4") < aliases.index("u m_E4")
    assert aliases.index("i my_E4") < aliases.index("u m_E4")


def test_manual_oto_row_slot_order_preserves_manifest_slot_index():
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _row_slot_ordered_rows

    rows = [
        {"alias": "late", "slot_index": 2, "slot_count": 3},
        {"alias": "early", "slot_index": 0, "slot_count": 3},
        {"alias": "middle", "slot_index": 1, "slot_count": 3},
    ]
    ordered = _row_slot_ordered_rows(rows)
    assert [row["alias"] for row, _pos in ordered] == ["early", "middle", "late"]
    assert [round(pos, 3) for _row, pos in ordered] == [0.167, 0.5, 0.833]


def test_manual_oto_anchor_vowel_island_slots_include_cv_leading_context_only():
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import (
        _row_vowel_island_slot_index,
        _vowel_island_slot_count,
    )

    cv_rows = [
        {
            "wav_name": "a-rya-ryu-ryo.wav",
            "filename_canonical_tokens": ["a", "rya", "ryu", "ryo"],
            "alias": "rya",
            "alias_role": "cv",
            "format_type": "cv",
            "slot_index": 0,
            "slot_count": 3,
            "language": "japanese",
        },
        {
            "wav_name": "a-rya-ryu-ryo.wav",
            "filename_canonical_tokens": ["a", "rya", "ryu", "ryo"],
            "alias": "ryu",
            "alias_role": "cv",
            "format_type": "cv",
            "slot_index": 1,
            "slot_count": 3,
            "language": "japanese",
        },
    ]
    assert _vowel_island_slot_count(cv_rows) == 4
    assert _row_vowel_island_slot_index(cv_rows[0], 4) == 1
    assert _row_vowel_island_slot_index(cv_rows[1], 4) == 2

    vcv_rows = [
        {
            "wav_name": "_pwi'pwi'o'pwi'u'pwi'pweo.wav",
            "filename_canonical_tokens": ["pwi", "pwi", "o", "pwi", "u", "pwi", "pweo"],
            "alias": "- pwi",
            "alias_role": "cv",
            "format_type": "vcv",
            "slot_index": 0,
            "slot_count": 5,
            "language": "korean",
        }
    ]
    assert _vowel_island_slot_count(vcv_rows) == 5
    assert _row_vowel_island_slot_index(vcv_rows[0], 5) == 0


def test_korean_cvvc_reclist_convention_distinguishes_tail_and_following_onset():
    from core.mfa_free_oto.reclist_convention import (
        STYLE_FOLLOWING_ONSET_VC,
        STYLE_TAIL_VC,
        analyze_reclist_convention,
        classify_korean_cvvc_alias_role,
    )

    tail_aliases = ["- ry", "NG ry", "NG ly", "N ry", "N ly", "M ry", "M ly", "L ry", "L ly", "i r", "o r"]
    tail_roles = [classify_korean_cvvc_alias_role(alias, language="korean", format_type="cvvc") for alias in tail_aliases]
    tail = analyze_reclist_convention(tail_aliases, roles=tail_roles, language="korean", format_type="cvvc")
    assert tail.vc_pre_style == STYLE_TAIL_VC
    assert tail.confidence >= 0.65

    following_aliases = ["ga", "na", "da", "ra", "ma", "ba", "a g", "a n", "a d", "a r", "a m", "a b"]
    following_roles = [
        classify_korean_cvvc_alias_role(alias, language="korean", format_type="cvvc")
        for alias in following_aliases
    ]
    following = analyze_reclist_convention(
        following_aliases,
        roles=following_roles,
        language="korean",
        format_type="cvvc",
    )
    assert following_roles[:6] == ["cv", "cv", "cv", "cv", "cv", "cv"]
    assert following_roles[6:] == ["vc", "vc", "vc", "vc", "vc", "vc"]
    assert following.vc_pre_style == STYLE_FOLLOWING_ONSET_VC
    assert following.confidence >= 0.85


def test_manual_oto_anchor_vcv_initial_cv_preutterance_policy_prefers_safe_island_start():
    import numpy as np

    from core.mfa_free_oto.manual_oto_candidates import ManualOtoCandidateTracks
    from core.mfa_free_oto.vowel_island import VowelIsland
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _vcv_initial_cv_preutterance_policy

    row = {
        "alias": "- pwi",
        "alias_role": "cv",
        "format_type": "vcv",
        "duration_ms": 1000.0,
        "slot_index": 0,
        "slot_count": 3,
    }
    island = VowelIsland(
        start_ms=120.0,
        nucleus_ms=180.0,
        end_ms=420.0,
        confidence=0.95,
        left_valley_ms=110.0,
        right_valley_ms=440.0,
        start_index=12,
        nucleus_index=18,
        end_index=42,
    )
    tracks = ManualOtoCandidateTracks(
        times_ms=np.asarray([100.0, 140.0, 500.0], dtype=np.float32),
        candidate_indices={"preutterance": (1, 2)},
        anchor_scores={"preutterance": np.asarray([0.1, 0.99, 1.0], dtype=np.float32)},
        tracks={},
        duration_ms=1000.0,
        encoder="test",
    )

    pred, source, reason = _vcv_initial_cv_preutterance_policy(
        row,
        tracks,
        island=island,
        fallback_pred_ms=500.0,
        fallback_safe_prob=0.99,
        duration_ms=1000.0,
        slot_count=3,
        slot_index=0,
    )

    assert pred == 140.0
    assert source.startswith("vcv_initial_cv_island_start")
    assert reason == ""


def test_manual_oto_anchor_vcv_sonorant_preutterance_policy_is_conservative():
    import numpy as np

    from core.mfa_free_oto.manual_oto_candidates import ManualOtoCandidateTracks
    from core.mfa_free_oto.vowel_island import VowelIsland
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _vcv_sonorant_preutterance_policy

    tracks = ManualOtoCandidateTracks(
        times_ms=np.asarray([420.0, 630.0, 690.0], dtype=np.float32),
        candidate_indices={"preutterance": (0, 1, 2)},
        anchor_scores={"preutterance": np.asarray([0.99, 1.0, 1.0], dtype=np.float32)},
        tracks={},
        duration_ms=3000.0,
        encoder="test",
    )
    row = {
        "alias": "n bweo",
        "alias_role": "vcv",
        "format_type": "vcv",
        "duration_ms": 3000.0,
        "slot_index": 0,
        "slot_count": 3,
    }
    island = VowelIsland(
        start_ms=180.0,
        nucleus_ms=700.0,
        end_ms=1430.0,
        confidence=0.95,
        left_valley_ms=180.0,
        right_valley_ms=1430.0,
        start_index=18,
        nucleus_index=70,
        end_index=143,
    )

    pred, source, reason = _vcv_sonorant_preutterance_policy(
        row,
        tracks,
        island=island,
        fallback_pred_ms=630.0,
        fallback_safe_prob=0.95,
        duration_ms=3000.0,
        slot_count=3,
        slot_index=0,
    )

    assert pred == 690.0
    assert source.startswith("vcv_sonorant_nasal_nucleus")
    assert reason == ""

    low_conf_inside = VowelIsland(
        start_ms=80.0,
        nucleus_ms=940.0,
        end_ms=1070.0,
        confidence=0.62,
        left_valley_ms=40.0,
        right_valley_ms=1110.0,
        start_index=8,
        nucleus_index=94,
        end_index=107,
    )
    pred, source, reason = _vcv_sonorant_preutterance_policy(
        {"alias": "l ma", "alias_role": "vcv", "format_type": "vcv", "duration_ms": 1600.0},
        tracks,
        island=low_conf_inside,
        fallback_pred_ms=550.0,
        fallback_safe_prob=0.99,
        duration_ms=3000.0,
        slot_count=3,
        slot_index=0,
    )

    assert pred is None
    assert source == ""
    assert reason == "vcv_sonorant_transition_review"


def test_manual_oto_anchor_slot_exact_island_start_preutterance_policy_requires_agreement():
    import numpy as np

    from core.mfa_free_oto.manual_oto_candidates import ManualOtoCandidateTracks
    from core.mfa_free_oto.vowel_island import VowelIsland
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import (
        _slot_exact_island_start_preutterance_policy,
    )

    row = {
        "alias": "ka",
        "alias_role": "cv",
        "format_type": "cv",
        "duration_ms": 1800.0,
        "slot_index": 0,
        "slot_count": 3,
    }
    island = VowelIsland(
        start_ms=260.0,
        nucleus_ms=340.0,
        end_ms=520.0,
        confidence=0.92,
        left_valley_ms=230.0,
        right_valley_ms=550.0,
        start_index=26,
        nucleus_index=34,
        end_index=52,
    )
    tracks = ManualOtoCandidateTracks(
        times_ms=np.asarray([250.0, 285.0, 980.0], dtype=np.float32),
        candidate_indices={"preutterance": (0, 1, 2)},
        anchor_scores={"preutterance": np.asarray([0.98, 0.96, 1.0], dtype=np.float32)},
        tracks={},
        duration_ms=1800.0,
        encoder="test",
    )

    pred, source, reason = _slot_exact_island_start_preutterance_policy(
        row,
        tracks,
        island=island,
        fallback_pred_ms=285.0,
        fallback_safe_prob=0.86,
        duration_ms=1800.0,
        slot_count=3,
        slot_index=0,
    )

    assert pred == 250.0
    assert source.startswith("slot_exact_island_start_preutterance")
    assert reason == ""

    pred, source, reason = _slot_exact_island_start_preutterance_policy(
        row,
        tracks,
        island=island,
        fallback_pred_ms=980.0,
        fallback_safe_prob=0.86,
        duration_ms=1800.0,
        slot_count=3,
        slot_index=0,
    )

    assert pred is None
    assert source == ""
    assert reason == "slot_exact_island_start_fallback_not_exact:one_step"


def test_manual_oto_alias_family_buckets_cover_common_suffixes():
    from core.mfa_free_oto.manual_oto_candidates import manual_oto_alias_family

    assert manual_oto_alias_family("ka") == "plain"
    assert manual_oto_alias_family("ka C4") == "pitch_suffix"
    assert manual_oto_alias_family("goF4P") == "power_suffix"
    assert manual_oto_alias_family("ka weak") == "weak_suffix"
    assert manual_oto_alias_family("a k") == "vowel_transition"
    assert manual_oto_alias_family("n a") == "leading_n"
    assert manual_oto_alias_family("br1") == "breath_silence"


def test_manual_oto_family_anchor_prior_prefers_transition_side():
    from ml.scripts.mfa_free_oto.train_manual_oto_anchor_scorer import _family_anchor_prior_score

    row = {
        "alias": "a k",
        "alias_role": "vc",
        "slot_count": 8,
    }
    earlier = _family_anchor_prior_score(
        row,
        "offset",
        candidate_time_norm=0.45,
        slot_pos_norm=0.50,
        weight=1.0,
    )
    much_later = _family_anchor_prior_score(
        row,
        "offset",
        candidate_time_norm=0.72,
        slot_pos_norm=0.50,
        weight=1.0,
    )
    assert earlier > much_later


def test_manual_anchor_preview_quality_keeps_local_gate_advisory():
    from core.mfa_free_oto.manual_anchor_runtime import _classify_preview_row_quality

    quality = _classify_preview_row_quality(
        {"alias": "ka", "alias_role": "cv", "duration_ms": 500.0},
        absolute={
            "offset_abs": 60.0,
            "overlap_abs": 90.0,
            "preutterance_abs": 130.0,
            "consonant_abs": 260.0,
            "cutoff_abs": 430.0,
        },
        warnings=["preutterance:local_gate_low:0.123"],
        local_gate_warnings=["preutterance:local_gate_low:0.123"],
        sources={"preutterance": "cv_landmark_vowel_onset"},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "confidence": 0.8, "warnings": []},
        duration_ms=500.0,
    )
    assert quality["status"] == "safe"
    assert quality["reasons"] == []
    assert "preutterance:local_gate_low:0.123" in quality["advisory"]
    assert quality["local_gate_policy"] == "advisory"


def test_manual_anchor_preview_quality_rejects_korean_cvvc_cv_by_default():
    from core.mfa_free_oto.manual_anchor_runtime import _classify_preview_row_quality

    quality = _classify_preview_row_quality(
        {"alias": "ka", "alias_role": "cv", "language": "korean", "format_type": "cvvc", "duration_ms": 500.0},
        absolute={
            "offset_abs": 60.0,
            "overlap_abs": 90.0,
            "preutterance_abs": 130.0,
            "consonant_abs": 260.0,
            "cutoff_abs": 430.0,
        },
        warnings=[],
        local_gate_warnings=[],
        sources={"preutterance": "cv_landmark_vowel_onset"},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "confidence": 0.99, "warnings": []},
        duration_ms=500.0,
    )
    assert quality["status"] == "needs_review"
    assert "korean_cvvc_cv_requires_review" in quality["reasons"]


def test_manual_anchor_preview_quality_rejects_japanese_vcv_until_verified():
    from core.mfa_free_oto.manual_anchor_runtime import _classify_preview_row_quality

    quality = _classify_preview_row_quality(
        {"alias": "a ka", "alias_role": "vcv", "language": "japanese", "format_type": "vcv", "duration_ms": 700.0},
        absolute={
            "offset_abs": 100.0,
            "overlap_abs": 180.0,
            "preutterance_abs": 210.0,
            "consonant_abs": 320.0,
            "cutoff_abs": 520.0,
        },
        warnings=[],
        local_gate_warnings=[],
        sources={"preutterance": "cv_landmark_vowel_onset"},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "confidence": 0.99, "warnings": []},
        duration_ms=700.0,
    )

    assert quality["status"] == "needs_review"
    assert "japanese_vcv_requires_review" in quality["reasons"]


def test_korean_cvvc_cv_timing_class_separates_initial_vowel_and_consonant():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _korean_cvvc_cv_cutoff_tail_gap_ms,
        _korean_cvvc_cv_landmark_preutterance_ms,
        _korean_cvvc_cv_relative_gaps,
        _korean_cvvc_cv_timing_class,
    )

    base = {"alias_role": "cv", "language": "korean", "format_type": "cvvc", "wav_name": "_x.wav"}
    initial_vowel = {**base, "alias": "- a"}
    initial_consonant = {**base, "alias": "- h"}
    apostrophe = {**base, "alias": "'a"}
    vf = {**base, "alias": "*L", "wav_name": "_L(vf).wav"}

    assert _korean_cvvc_cv_timing_class(initial_vowel) == "initial_vowel"
    assert _korean_cvvc_cv_landmark_preutterance_ms(
        initial_vowel,
        {"vowel_onset_ms": 200.0},
        fallback_ms=180.0,
    ) == pytest.approx(200.0)
    assert _korean_cvvc_cv_relative_gaps(initial_vowel) == {"offset_gap": 42.0, "overlap_gap": 21.0, "fixed_gap": 184.0}
    assert _korean_cvvc_cv_cutoff_tail_gap_ms(initial_vowel) == pytest.approx(390.0)

    assert _korean_cvvc_cv_timing_class(initial_consonant) == "initial_consonant"
    assert _korean_cvvc_cv_landmark_preutterance_ms(
        initial_consonant,
        {"vowel_onset_ms": 200.0},
        fallback_ms=180.0,
    ) == pytest.approx(138.0)
    assert _korean_cvvc_cv_relative_gaps(initial_consonant) == {"offset_gap": 36.0, "overlap_gap": 18.0, "fixed_gap": 18.0}
    assert _korean_cvvc_cv_cutoff_tail_gap_ms(initial_consonant) == pytest.approx(95.0)

    assert _korean_cvvc_cv_timing_class(apostrophe) == "apostrophe"
    assert _korean_cvvc_cv_relative_gaps(apostrophe) == {"offset_gap": 77.0, "overlap_gap": 38.0, "fixed_gap": 59.0}
    assert _korean_cvvc_cv_cutoff_tail_gap_ms(apostrophe) == pytest.approx(190.0)

    assert _korean_cvvc_cv_timing_class(vf) == "vf"
    assert _korean_cvvc_cv_relative_gaps(vf) == {"offset_gap": 153.0, "overlap_gap": 76.0, "fixed_gap": 370.0}
    assert _korean_cvvc_cv_cutoff_tail_gap_ms(vf) == pytest.approx(1530.0)


def test_manual_anchor_korean_cvvc_filename_tokens_split_underscore_tail_sets():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _filename_slot_tokens_for_row,
        _korean_cvvc_row_syllable_index,
    )

    base = {
        "language": "korean",
        "format_type": "cvvc",
        "wav_name": "_euN_euM_euNG_euL.wav",
        "slot_index": 0,
        "slot_count": 8,
        "source_order": 0,
    }

    tokens = _filename_slot_tokens_for_row({**base, "alias": "L R", "alias_role": "vc"})

    assert tokens == ["eun", "eum", "eung", "eul"]
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "L R", "alias_role": "vc"},
        tokens,
        len(tokens),
    ) == 3
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "NG R", "alias_role": "vc"},
        tokens,
        len(tokens),
    ) == 2
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "L -", "alias_role": "v"},
        tokens,
        len(tokens),
    ) == 3


def test_manual_anchor_korean_cvvc_tail_marker_h_alias_maps_to_matching_coda_slot():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _filename_slot_tokens_for_row,
        _korean_cvvc_row_syllable_index,
    )

    row = {
        "language": "korean",
        "format_type": "cvvc",
        "wav_name": "_aH_iH_uH_eH.wav",
        "slot_index": 0,
        "slot_count": 8,
        "source_order": 0,
        "alias": "e H",
        "alias_role": "vc",
    }
    tokens = _filename_slot_tokens_for_row(row)

    assert tokens == ["ah", "ih", "uh", "eh"]
    assert _korean_cvvc_row_syllable_index(row, tokens, len(tokens)) == 3


def test_manual_anchor_korean_cvvc_raw_uppercase_coda_profiles_are_case_sensitive():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _default_utau_cutoff_tail_gap_ms,
        _korean_cvvc_relative_gaps,
        _korean_cvvc_transition_profile,
    )

    base = {"alias_role": "vc", "language": "korean", "format_type": "cvvc", "wav_name": "_x.wav"}

    nasal = _korean_cvvc_transition_profile({**base, "alias": "a M"})
    assert nasal is not None
    assert nasal["raw_right"] == "M"
    assert nasal["pre_ratio"] == pytest.approx(0.75)
    assert _korean_cvvc_relative_gaps({**base, "alias": "a M"}) == {
        "offset_gap": 70.0,
        "overlap_gap": 35.0,
        "fixed_gap": 63.0,
    }
    assert _default_utau_cutoff_tail_gap_ms({**base, "alias": "a M"}) == pytest.approx(221.0)

    stop = _korean_cvvc_transition_profile({**base, "alias": "a P"})
    assert stop is not None
    assert stop["raw_right"] == "P"
    assert stop["pre_ratio"] == pytest.approx(0.85)
    assert _korean_cvvc_relative_gaps({**base, "alias": "a P"})["fixed_gap"] == pytest.approx(31.0)
    assert _default_utau_cutoff_tail_gap_ms({**base, "alias": "a P"}) == pytest.approx(224.0)

    assert _korean_cvvc_transition_profile({**base, "alias": "a m"}) is None
    assert _korean_cvvc_transition_profile({**base, "alias": "a p"}) is None


def test_manual_anchor_breath_silence_guard_overrides_late_island_cv_predictions():
    from core.mfa_free_oto.manual_anchor_runtime import _apply_manual_anchor_timing_guards

    predictions = {
        "offset": 2000.0,
        "overlap": 2025.0,
        "preutterance": 2050.0,
        "fixed_end": 2910.0,
        "cutoff": 2914.0,
    }

    warnings, _ = _apply_manual_anchor_timing_guards(
        {"alias": "br3", "alias_role": "cv", "language": "korean", "format_type": "cvvc"},
        predictions,
        island_context={},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "vowel_onset_ms": 2050.0, "confidence": 0.99},
        duration_ms=3799.3650793650795,
        overlap_context={},
    )

    assert "mfa_style_breath_silence_guard" in warnings
    assert predictions["preutterance"] == pytest.approx(760.0)
    assert predictions["offset"] == pytest.approx(710.0)
    assert predictions["overlap"] == pytest.approx(735.0)
    assert predictions["fixed_end"] == pytest.approx(3799.3650793650795 * 0.54)
    assert predictions["cutoff"] == pytest.approx(2914.0)


def test_manual_anchor_korean_cvvc_v_profiles_split_vf_tail_and_coda_endings():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _default_island_preutterance_ms,
        _default_utau_cutoff_tail_gap_ms,
        _korean_cvvc_relative_gaps,
    )
    from core.mfa_free_oto.vowel_island import VowelIsland

    island = VowelIsland(1000.0, 1300.0, 1800.0, 0.9, 980.0, 1820.0, 100, 130, 180)
    base = {"alias_role": "v", "language": "korean", "format_type": "cvvc", "duration_ms": 3000.0}

    vf_tail = {**base, "alias": "a *", "wav_name": "_a(vf).wav"}
    assert _default_island_preutterance_ms(vf_tail, island) == pytest.approx(2220.0)
    assert _korean_cvvc_relative_gaps(vf_tail) == {
        "offset_gap": 118.0,
        "overlap_gap": 59.0,
        "fixed_gap": 326.0,
    }
    assert _default_utau_cutoff_tail_gap_ms(vf_tail) == pytest.approx(610.0)

    vf_u_tail = {**base, "alias": "u *", "wav_name": "_u(vf).wav"}
    assert _default_island_preutterance_ms(vf_u_tail, island) == pytest.approx(1800.0)
    assert _korean_cvvc_relative_gaps(vf_u_tail) == {
        "offset_gap": 118.0,
        "overlap_gap": 59.0,
        "fixed_gap": 326.0,
    }
    assert _default_utau_cutoff_tail_gap_ms(vf_u_tail) == pytest.approx(610.0)

    vf_head = {**base, "alias": "*a", "wav_name": "_a(vf).wav"}
    assert _korean_cvvc_relative_gaps(vf_head) == {"offset_gap": 167.0, "overlap_gap": 84.0, "fixed_gap": 409.0}
    assert _default_utau_cutoff_tail_gap_ms(vf_head) == pytest.approx(1045.0)

    nasal_tail = {**base, "alias": "NG -", "wav_name": "_euN_euM_euNG_euL.wav"}
    assert _default_island_preutterance_ms(nasal_tail, island) == pytest.approx(1800.0)
    assert _korean_cvvc_relative_gaps(nasal_tail) == {
        "offset_gap": 50.0,
        "overlap_gap": 25.0,
        "fixed_gap": 12.0,
    }
    assert _default_utau_cutoff_tail_gap_ms(nasal_tail) == pytest.approx(70.0)

    liquid_tail = {**base, "alias": "L -", "wav_name": "_euN_euM_euNG_euL.wav"}
    assert _korean_cvvc_relative_gaps(liquid_tail) == {
        "offset_gap": 80.0,
        "overlap_gap": 40.0,
        "fixed_gap": 360.0,
    }
    assert _default_utau_cutoff_tail_gap_ms(liquid_tail) == pytest.approx(500.0)


def test_manual_anchor_korean_cvvc_closing_apostrophe_v_uses_previous_vowel_slot():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _default_island_preutterance_ms,
        _default_utau_cutoff_tail_gap_ms,
        _filename_slot_tokens_for_row,
        _korean_cvvc_relative_gaps,
        _korean_cvvc_row_syllable_index,
        _korean_cvvc_v_alias_is_tail,
    )
    from core.mfa_free_oto.vowel_island import VowelIsland

    base = {
        "language": "korean",
        "format_type": "cvvc",
        "wav_name": "_eu'a'i'o'e'u'eo'eu(cl).wav",
        "slot_count": 14,
        "alias_role": "v",
    }
    tokens = _filename_slot_tokens_for_row({**base, "alias": "a '", "source_order": 2, "slot_index": 2})

    assert tokens == ["eu", "a", "i", "o", "e", "u", "eo", "eu"]
    assert _korean_cvvc_v_alias_is_tail({**base, "alias": "a '", "source_order": 2, "slot_index": 2}) is True
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "eu '", "source_order": 0, "slot_index": 0},
        tokens,
        len(tokens),
    ) == 0
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "a '", "source_order": 2, "slot_index": 2},
        tokens,
        len(tokens),
    ) == 1
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "i '", "source_order": 4, "slot_index": 4},
        tokens,
        len(tokens),
    ) == 2
    assert _korean_cvvc_row_syllable_index(
        {**base, "alias": "eo '", "source_order": 12, "slot_index": 12},
        tokens,
        len(tokens),
    ) == 6

    island = VowelIsland(1520.0, 1610.0, 1750.0, 0.9, 1480.0, 1790.0, 152, 161, 175)
    row = {**base, "alias": "a '", "source_order": 2, "slot_index": 2, "duration_ms": 5526.35}
    assert _default_island_preutterance_ms(row, island) == pytest.approx(1750.0)
    assert _korean_cvvc_relative_gaps(row) == {
        "offset_gap": 77.0,
        "overlap_gap": 38.0,
        "fixed_gap": 110.0,
    }
    assert _default_utau_cutoff_tail_gap_ms(row) == pytest.approx(220.0)


def test_manual_anchor_korean_cvvc_vv_profiles_correct_tail_biased_valleys():
    from core.mfa_free_oto.manual_anchor_runtime import _korean_cvvc_transition_preutterance_ms
    from core.mfa_free_oto.vowel_island import VowelIsland

    base = {"alias_role": "vv", "language": "korean", "format_type": "cvvc"}

    i_eo = _korean_cvvc_transition_preutterance_ms(
        {**base, "alias": "i eo"},
        VowelIsland(3170.0, 3660.0, 3880.0, 0.9, 3170.0, 3880.0, 317, 366, 388),
        {"confidence": 0.69, "transition_valley_ms": 2940.0, "right_vowel_onset_ms": 3170.0},
        fallback_ms=3170.0,
    )
    assert i_eo is not None
    assert i_eo[0] == pytest.approx(3464.0)
    assert i_eo[1] == "korean_cvvc_vv_profile_right_eo"

    e_eo = _korean_cvvc_transition_preutterance_ms(
        {**base, "alias": "e eo"},
        VowelIsland(3810.0, 4010.0, 4310.0, 0.9, 3810.0, 4310.0, 381, 401, 431),
        {"confidence": 0.62, "transition_valley_ms": 3890.0, "right_vowel_onset_ms": 3810.0},
        fallback_ms=3810.0,
    )
    assert e_eo is not None
    assert e_eo[0] == pytest.approx(3890.0)
    assert e_eo[1] == "korean_cvvc_vv_profile_right_eo_valley"

    u_e = _korean_cvvc_transition_preutterance_ms(
        {**base, "alias": "u e"},
        VowelIsland(1790.0, 2010.0, 2480.0, 0.9, 1790.0, 2480.0, 179, 201, 248),
        {"confidence": 0.44, "transition_valley_ms": 1780.0, "right_vowel_onset_ms": 1790.0},
        fallback_ms=1790.0,
    )
    assert u_e is not None
    assert u_e[0] == pytest.approx(1974.8)
    assert u_e[1] == "korean_cvvc_vv_profile_u_e"

    a_a = _korean_cvvc_transition_preutterance_ms(
        {**base, "alias": "a a"},
        VowelIsland(1320.0, 1470.0, 2020.0, 0.9, 1320.0, 2020.0, 132, 147, 202),
        {"confidence": 0.63, "transition_valley_ms": 1300.0, "right_vowel_onset_ms": 1320.0},
        fallback_ms=1320.0,
    )
    assert a_a is not None
    assert a_a[0] == pytest.approx(1656.0)
    assert a_a[1] == "korean_cvvc_vv_profile_open_repeat"

    a_i = _korean_cvvc_transition_preutterance_ms(
        {**base, "alias": "a i"},
        VowelIsland(2030.0, 2140.0, 2400.0, 0.9, 2030.0, 2400.0, 203, 214, 240),
        {"confidence": 0.67, "transition_valley_ms": 1730.0, "right_vowel_onset_ms": 2030.0},
        fallback_ms=2030.0,
    )
    assert a_i is None


def test_manual_anchor_mfa_style_adaptive_overlap_uses_consonant_class():
    from core.mfa_free_oto.manual_anchor_runtime import _manual_anchor_adaptive_overlap_gap

    base = {"alias_role": "cv", "language": "korean", "format_type": "cvvc"}
    hard_gap = _manual_anchor_adaptive_overlap_gap({**base, "alias": "ga"}, 100.0)
    sonorant_gap = _manual_anchor_adaptive_overlap_gap({**base, "alias": "ma"}, 100.0)
    tense_gap = _manual_anchor_adaptive_overlap_gap({**base, "alias": "kka"}, 100.0)

    assert sonorant_gap < hard_gap
    assert tense_gap > hard_gap


def test_manual_anchor_mfa_style_order_validation_normalizes_bad_predictions():
    from core.mfa_free_oto.manual_anchor_runtime import _manual_anchor_validate_prediction_order

    predictions = {
        "offset": 90.0,
        "overlap": 30.0,
        "preutterance": 80.0,
        "fixed_end": 82.0,
        "cutoff": 81.0,
    }
    warnings = _manual_anchor_validate_prediction_order(
        {"alias": "- h", "alias_role": "cv", "language": "korean", "format_type": "cvvc"},
        predictions,
        duration_ms=120.0,
    )

    assert warnings == ["mfa_style_order_validation:overlap,preutterance,fixed_end,cutoff"]
    assert predictions["offset"] <= predictions["overlap"] <= predictions["preutterance"]
    assert predictions["preutterance"] < predictions["fixed_end"] < predictions["cutoff"]
    assert predictions["cutoff"] <= 120.0


def test_manual_anchor_mfa_style_vc_cutoff_guard_uses_next_island():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _apply_manual_anchor_timing_guards,
        _manual_anchor_island_context,
    )
    from core.mfa_free_oto.vowel_island import SlotIslandAssignment, VowelIsland

    islands = (
        VowelIsland(100.0, 150.0, 220.0, 0.9, 90.0, 230.0, 10, 15, 22),
        VowelIsland(260.0, 310.0, 380.0, 0.9, 250.0, 390.0, 26, 31, 38),
    )
    context = _manual_anchor_island_context(
        SlotIslandAssignment(slot_index=0, island_index=0, score=1.0, margin=0.5),
        islands,
    )
    predictions = {"offset": 110.0, "overlap": 145.0, "preutterance": 180.0, "fixed_end": 196.0, "cutoff": 180.0}

    warnings, _ = _apply_manual_anchor_timing_guards(
        {"alias": "a g", "alias_role": "vc", "language": "korean", "format_type": "cvvc"},
        predictions,
        island_context=context,
        acoustic_landmarks=None,
        duration_ms=500.0,
        overlap_context={},
    )

    assert "mfa_style_vc_cutoff_guard" in warnings
    assert predictions["cutoff"] == pytest.approx(264.0)


def test_manual_anchor_mfa_style_vv_cutoff_guard_uses_current_vowel_end():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _apply_manual_anchor_timing_guards,
        _manual_anchor_island_context,
    )
    from core.mfa_free_oto.vowel_island import SlotIslandAssignment, VowelIsland

    islands = (
        VowelIsland(100.0, 150.0, 220.0, 0.9, 90.0, 230.0, 10, 15, 22),
        VowelIsland(260.0, 310.0, 380.0, 0.9, 250.0, 390.0, 26, 31, 38),
    )
    context = _manual_anchor_island_context(
        SlotIslandAssignment(slot_index=0, island_index=0, score=1.0, margin=0.5),
        islands,
    )
    predictions = {"offset": 120.0, "overlap": 160.0, "preutterance": 200.0, "fixed_end": 216.0, "cutoff": 220.0}

    warnings, _ = _apply_manual_anchor_timing_guards(
        {"alias": "a i", "alias_role": "vv", "language": "korean", "format_type": "cvvc"},
        predictions,
        island_context=context,
        acoustic_landmarks=None,
        duration_ms=500.0,
        overlap_context={},
    )

    assert "mfa_style_vv_cutoff_guard" in warnings
    assert predictions["cutoff"] == pytest.approx(234.0)


def test_manual_anchor_mfa_style_cv_boundary_guard_clamps_outlier_preutterance():
    from core.mfa_free_oto.manual_anchor_runtime import _manual_anchor_cv_boundary_guard

    predictions = {"offset": 260.0, "overlap": 280.0, "preutterance": 300.0, "fixed_end": 320.0, "cutoff": 480.0}
    warning = _manual_anchor_cv_boundary_guard(
        {"alias": "- h", "alias_role": "cv", "language": "korean", "format_type": "cvvc"},
        predictions,
        acoustic_landmarks={"vowel_onset_ms": 200.0, "confidence": 0.8},
        duration_ms=500.0,
    )

    assert warning == "mfa_style_cv_boundary_guard"
    assert predictions["preutterance"] == pytest.approx(226.0)
    assert predictions["offset"] <= predictions["preutterance"] - 8.0


def test_manual_anchor_japanese_vcv_uses_current_vowel_start_as_island_preutterance():
    from core.mfa_free_oto.manual_anchor_runtime import _default_island_preutterance_ms
    from core.mfa_free_oto.vowel_island import VowelIsland

    island = VowelIsland(320.0, 390.0, 540.0, 0.9, 300.0, 560.0, 32, 39, 54)
    row = {"alias": "a ka", "alias_role": "vcv", "language": "japanese", "format_type": "vcv"}

    assert _default_island_preutterance_ms(row, island) == pytest.approx(320.0)


def test_manual_anchor_japanese_vcv_kana_filename_uses_filename_slot_count():
    from core.mfa_free_oto.manual_anchor_runtime import _row_vowel_island_slot_index, _vowel_island_slot_count

    base = {
        "wav_name": "\u3042\u3042\u3044\u3042\u3046\u3048\u3042.wav",
        "language": "japanese",
        "format_type": "vcv",
        "slot_count": 3,
    }
    rows = [
        {**base, "alias": "- a", "alias_role": "cv", "slot_index": 0},
        {**base, "alias": "a i", "alias_role": "vcv", "slot_index": 1},
        {**base, "alias": "a u", "alias_role": "vcv", "slot_index": 2},
    ]

    assert _vowel_island_slot_count(rows) == 7
    assert _row_vowel_island_slot_index(rows[2], 7) == 4


def test_manual_anchor_japanese_vcv_romaji_filename_uses_previous_vowel_for_duplicate_targets():
    from core.mfa_free_oto.manual_anchor_runtime import _row_vowel_island_slot_index, _vowel_island_slot_count
    from core.model_context.filename import filename_syllable_order_tokens

    base = {
        "wav_name": "tututitotatoti.wav",
        "language": "japanese",
        "format_type": "vcv",
        "slot_count": 4,
        "filename_tokens": filename_syllable_order_tokens("tututitotatoti.wav", language="japanese"),
    }
    rows = [
        {**base, "alias": "u ti", "alias_role": "vcv", "slot_index": 0},
        {**base, "alias": "o ta", "alias_role": "vcv", "slot_index": 1},
        {**base, "alias": "a to", "alias_role": "vcv", "slot_index": 2},
        {**base, "alias": "o ti", "alias_role": "vcv", "slot_index": 3},
    ]

    assert base["filename_tokens"] == ["tu", "tu", "ti", "to", "ta", "to", "ti"]
    assert _vowel_island_slot_count(rows) == 7
    assert _row_vowel_island_slot_index(rows[0], 7) == 2
    assert _row_vowel_island_slot_index(rows[3], 7) == 6


def test_manual_anchor_japanese_vcv_landmark_preutterance_refines_from_vowel_onset():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _cv_landmark_row,
        _landmark_preutterance_ms,
    )
    from core.mfa_free_oto.vowel_island import VowelIsland

    island = VowelIsland(320.0, 390.0, 540.0, 0.9, 300.0, 560.0, 32, 39, 54)
    row = {"alias": "a ka", "alias_role": "vcv", "language": "japanese", "format_type": "vcv"}

    assert _cv_landmark_row(row) is True
    assert _landmark_preutterance_ms(
        row,
        {"source": "acoustic_cv_landmark_v1", "vowel_onset_ms": 322.0, "confidence": 0.72},
        island,
        fallback_ms=540.0,
    ) == pytest.approx(322.0)


def test_manual_anchor_japanese_vv_landmark_blends_transition_and_right_onset():
    from core.mfa_free_oto.manual_anchor_runtime import _landmark_preutterance_ms
    from core.mfa_free_oto.vowel_island import VowelIsland

    island = VowelIsland(1280.0, 1420.0, 1540.0, 0.9, 1200.0, 1560.0, 128, 142, 154)
    row = {"alias": "i u", "alias_role": "vv", "language": "japanese", "format_type": "vcv"}

    assert _landmark_preutterance_ms(
        row,
        {
            "source": "acoustic_vv_landmark_v1",
            "transition_valley_ms": 1200.0,
            "right_vowel_onset_ms": 1280.0,
            "confidence": 0.58,
        },
        island,
        fallback_ms=1500.0,
    ) == pytest.approx(1340.0)


def test_manual_anchor_japanese_vv_predict_row_prefers_landmark_over_slot_exact(monkeypatch):
    import numpy as np

    from core.mfa_free_oto import manual_anchor_runtime as runtime
    from core.mfa_free_oto.manual_oto_candidates import ManualOtoCandidateTracks
    from core.mfa_free_oto.vowel_island import SlotIslandAssignment, VowelIsland, VowelIslandDecode

    row = {
        "wav_name": "iu.wav",
        "alias": "i u",
        "alias_role": "vv",
        "language": "japanese",
        "format_type": "vcv",
        "duration_ms": 2200.0,
        "slot_index": 1,
        "slot_count": 2,
    }
    tracks = ManualOtoCandidateTracks(
        times_ms=np.asarray([0.0, 1340.0, 1500.0, 2200.0], dtype=np.float32),
        candidate_indices={},
        anchor_scores={},
        tracks={},
        duration_ms=2200.0,
        encoder="test",
    )
    current = VowelIsland(1280.0, 1420.0, 1540.0, 0.9, 1200.0, 1560.0, 128, 142, 154)
    decode = VowelIslandDecode(
        islands=(
            VowelIsland(520.0, 700.0, 940.0, 0.9, 480.0, 960.0, 52, 70, 94),
            current,
        ),
        assignments=(
            SlotIslandAssignment(slot_index=0, island_index=0, score=1.0, margin=0.5),
            SlotIslandAssignment(slot_index=1, island_index=1, score=1.0, margin=0.5),
        ),
        score=1.0,
        margin=0.5,
    )
    scorer = runtime.ManualAnchorScorer(
        path="",
        anchors=("preutterance",),
        encoder="test",
        models={},
        failure_gate_models={},
        local_failure_models={},
        relative_anchor_priors={},
        island_anchor_priors={},
    )
    monkeypatch.setattr(
        runtime,
        "_role_acoustic_landmarks",
        lambda *args, **kwargs: {
            "source": "acoustic_vv_landmark_v1",
            "preutterance_source": "japanese_vv_landmark_blend",
            "transition_valley_ms": 1200.0,
            "right_vowel_onset_ms": 1280.0,
            "confidence": 0.58,
            "warnings": [],
        },
    )

    result = runtime._predict_row(
        row,
        tracks,
        scorer,
        anchors=("preutterance",),
        constrained={(id(row), "preutterance"): {"pred_ms": 1500.0, "slot_pos_norm": 0.75}},
        filename_slot={},
        joint={},
        island_decode=decode,
        slot_count=2,
    )

    assert result["record"]["predictions_abs_ms"]["preutterance"] == pytest.approx(1340.0)
    assert result["record"]["sources"]["preutterance"] == "japanese_vv_landmark_blend"


def test_manual_anchor_posterior_preserves_timebase_metadata():
    import numpy as np

    from core.mfa_free_oto.manual_anchor_runtime import _posterior_from_tracks
    from core.mfa_free_oto.manual_oto_candidates import ManualOtoCandidateTracks

    tracks = ManualOtoCandidateTracks(
        times_ms=np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
        candidate_indices={},
        anchor_scores={},
        tracks={},
        duration_ms=30.0,
        encoder="acoustic",
        timebase_metadata={
            "source_sample_rate": 44100,
            "analysis_sample_rate": 16000,
            "window_size_ms": 25.0,
            "frame_shift_ms": 10.0,
            "resample_method": "linear",
        },
    )

    posterior = _posterior_from_tracks("sample.wav", tracks)

    assert posterior.metadata["source"] == "manual_oto_anchor_runtime"
    assert posterior.metadata["source_sample_rate"] == 44100
    assert posterior.metadata["analysis_sample_rate"] == 16000
    assert posterior.metadata["resample_method"] == "linear"


def test_manual_anchor_japanese_adaptive_overlap_uses_vcv_onset_class():
    from core.mfa_free_oto.manual_anchor_runtime import _manual_anchor_adaptive_overlap_gap

    base = {"alias_role": "vcv", "language": "japanese", "format_type": "vcv"}
    hard_gap = _manual_anchor_adaptive_overlap_gap({**base, "alias": "a ka"}, 120.0)
    sonorant_gap = _manual_anchor_adaptive_overlap_gap({**base, "alias": "a ma"}, 120.0)

    assert hard_gap > sonorant_gap
    assert hard_gap < 20.0


def test_manual_anchor_japanese_vcv_boundary_guard_clamps_outlier_preutterance():
    from core.mfa_free_oto.manual_anchor_runtime import _manual_anchor_cv_boundary_guard

    predictions = {"offset": 470.0, "overlap": 500.0, "preutterance": 520.0, "fixed_end": 590.0, "cutoff": 760.0}
    warning = _manual_anchor_cv_boundary_guard(
        {"alias": "a ka", "alias_role": "vcv", "language": "japanese", "format_type": "vcv"},
        predictions,
        acoustic_landmarks={"vowel_onset_ms": 320.0, "confidence": 0.8},
        duration_ms=900.0,
    )

    assert warning == "mfa_style_ja_cv_boundary_guard"
    assert predictions["preutterance"] == pytest.approx(370.0)
    assert predictions["offset"] <= predictions["preutterance"] - 18.0


def test_manual_anchor_japanese_vc_cutoff_guard_only_raises_too_early_cutoff():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _apply_manual_anchor_timing_guards,
        _manual_anchor_island_context,
    )
    from core.mfa_free_oto.vowel_island import SlotIslandAssignment, VowelIsland

    islands = (
        VowelIsland(120.0, 180.0, 260.0, 0.9, 100.0, 280.0, 12, 18, 26),
        VowelIsland(310.0, 370.0, 460.0, 0.9, 290.0, 480.0, 31, 37, 46),
    )
    context = _manual_anchor_island_context(
        SlotIslandAssignment(slot_index=0, island_index=0, score=1.0, margin=0.5),
        islands,
    )
    predictions = {"offset": 180.0, "overlap": 220.0, "preutterance": 250.0, "fixed_end": 270.0, "cutoff": 280.0}

    warnings, _ = _apply_manual_anchor_timing_guards(
        {"alias": "a k", "alias_role": "vc", "language": "japanese", "format_type": "cvvc"},
        predictions,
        island_context=context,
        acoustic_landmarks=None,
        duration_ms=700.0,
        overlap_context={},
    )

    assert "mfa_style_vc_cutoff_guard" in warnings
    assert predictions["cutoff"] == pytest.approx(308.0)


def test_manual_anchor_preview_quality_rejects_structural_warnings():
    from core.mfa_free_oto.manual_anchor_runtime import _classify_preview_row_quality

    quality = _classify_preview_row_quality(
        {"alias": "ka", "alias_role": "cv", "duration_ms": 500.0},
        absolute={
            "offset_abs": 60.0,
            "overlap_abs": 90.0,
            "preutterance_abs": 130.0,
            "consonant_abs": 260.0,
            "cutoff_abs": 430.0,
        },
        warnings=["island_count_mismatch", "preutterance_slot_shift:+1"],
        local_gate_warnings=[],
        sources={"preutterance": "cv_landmark_vowel_onset"},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "confidence": 0.8, "warnings": []},
        duration_ms=500.0,
    )
    assert quality["status"] == "needs_review"
    assert "island_count_mismatch" in quality["reasons"]
    assert "preutterance_slot_shift" in quality["reasons"]


def test_manual_anchor_preview_sidecar_paths_keep_oto_ini_stem(tmp_path):
    from core.mfa_free_oto.manual_anchor_runtime import _preview_sidecar_paths

    paths = _preview_sidecar_paths(tmp_path / "preview.oto.ini")
    assert paths["safe_oto"].name == "preview.oto.safe.ini"
    assert paths["needs_review_oto"].name == "preview.oto.needs_review.ini"
    assert paths["needs_review_csv"].name == "preview.oto.needs_review.csv"


def test_manual_anchor_preview_sidecars_split_safe_and_review_rows(tmp_path):
    from core.mfa_free_oto.manual_anchor_runtime import _write_preview_quality_sidecars

    out = _write_preview_quality_sidecars(
        tmp_path / "preview.oto.ini",
        [
            {
                "wav": "a.wav",
                "generated_oto_rows": [
                    {
                        "wav": "a.wav",
                        "alias": "ka",
                        "alias_role": "cv",
                        "line": "a.wav=ka,0,100,-100,50,25",
                        "warnings": ["preutterance:local_gate_low:0.500"],
                        "sources": {"preutterance": "cv_landmark_vowel_onset"},
                        "quality": {"status": "safe", "reasons": [], "advisory": ["preutterance:local_gate_low:0.500"]},
                    },
                    {
                        "wav": "a.wav",
                        "alias": "a k",
                        "alias_role": "vc",
                        "line": "a.wav=a k,100,80,-80,45,20",
                        "warnings": ["island_count_mismatch"],
                        "sources": {"preutterance": "korean_cvvc_vowel_island"},
                        "quality": {"status": "needs_review", "reasons": ["island_count_mismatch"], "advisory": []},
                    },
                ],
            }
        ],
    )
    assert (tmp_path / "preview.oto.safe.ini").read_text(encoding="utf-8").strip() == "a.wav=ka,0,100,-100,50,25"
    assert (tmp_path / "preview.oto.needs_review.ini").read_text(encoding="utf-8").strip() == "a.wav=a k,100,80,-80,45,20"
    review_csv = (tmp_path / "preview.oto.needs_review.csv").read_text(encoding="utf-8")
    assert "island_count_mismatch" in review_csv
    assert out["summary"]["status_counts"] == {"safe": 1, "needs_review": 1}


def test_manual_anchor_preview_repaired_island_count_is_advisory():
    from core.mfa_free_oto.manual_anchor_runtime import (
        _classify_preview_row_quality,
        _mark_island_count_repaired,
    )
    from core.mfa_free_oto.vowel_island import SlotIslandAssignment, VowelIslandDecode

    repaired = _mark_island_count_repaired(
        VowelIslandDecode(
            islands=(),
            assignments=(SlotIslandAssignment(slot_index=0, island_index=0, score=1.0, margin=0.5),),
            score=1.0,
            margin=0.5,
        )
    )
    assert "island_count_repaired" in repaired.warnings
    assert "island_count_mismatch" not in repaired.warnings
    quality = _classify_preview_row_quality(
        {"alias": "ka", "alias_role": "cv", "duration_ms": 500.0},
        absolute={
            "offset_abs": 60.0,
            "overlap_abs": 90.0,
            "preutterance_abs": 130.0,
            "consonant_abs": 260.0,
            "cutoff_abs": 430.0,
        },
        warnings=list(repaired.warnings),
        local_gate_warnings=[],
        sources={"preutterance": "cv_landmark_vowel_onset"},
        acoustic_landmarks={"source": "acoustic_cv_landmark_v1", "confidence": 0.8, "warnings": []},
        duration_ms=500.0,
    )
    assert quality["status"] == "safe"
    assert quality["advisory"] == ["island_count_repaired"]


def test_read_text_with_fallback_prefers_cp932_over_latin1_when_japanese_oto_has_one_bad_byte(tmp_path):
    from core.generation.common.oto_file_utils import read_text_with_fallback

    source = "\u3042\u3042\u3044\u3042\u3046\u3048\u3042.wav=- a,518,148,3449,57,22\n"
    oto = tmp_path / "oto.ini"
    oto.write_bytes(source.encode("cp932") + b"\x81\x00\n")

    text = read_text_with_fallback(str(oto))

    assert "\u3042\u3042\u3044\u3042\u3046\u3048\u3042.wav=- a" in text
    assert "\x82\xa0" not in text


def _labelled_row(source: str) -> dict:
    return {
        "row_id": "001_ka",
        "wav_path": "001_ka.wav",
        "duration_ms": 240.0,
        "label_source": source,
        "expected_phones": ["k", "a"],
        "frame_labels": [
            {"label": "silence", "start_ms": 0.0, "end_ms": 50.0},
            {"label": "consonant", "start_ms": 50.0, "end_ms": 100.0},
            {"label": "vowel", "start_ms": 100.0, "end_ms": 220.0},
        ],
        "events": [
            {"label": "cv_boundary", "time_ms": 100.0, "phone": "a"},
            {"label": "vowel_nucleus", "time_ms": 160.0, "phone": "a"},
            {"label": "phone_change", "time_ms": 100.0, "phone": "a"},
        ],
    }


def _write_tone_wav(path, *, sample_rate: int = 16000, duration_s: float = 0.25) -> None:
    samples = []
    for idx in range(int(sample_rate * duration_s)):
        value = int(12000 * math.sin(2.0 * math.pi * 220.0 * idx / sample_rate))
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())
