from __future__ import annotations

import json
import math
import wave

import numpy as np
import pytest

from core.mfa_free_oto.decode import decode_monotonic_events
from core.mfa_free_oto.features import extract_features
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
