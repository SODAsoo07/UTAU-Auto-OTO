from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

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
    AdaptedOtoRow,
    OtoAdapterConfig,
    OtoAnchor,
    OtoTemplateRow,
    OtoTiming,
    _assign_alias_target_indices,
    _alias_phone_sequence,
    _alias_type_for_row,
    _refine_anchor_sequence_locally,
    adapt_template_row,
    anchors_from_prediction,
    assign_template_row_anchors,
    bootstrap_row,
    expected_slots_for_template_rows,
    parse_template_oto_line,
    repair_cvvc_row_sequence,
    repair_cvvc_vc_row_sequence,
)
from core.mfa_free_oto.row_plan import build_filename_slots, filename_phone_sequence_from_slots
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
    assert _alias_type_for_row("a n", "auto") == "vc"
    assert _alias_type_for_row("a ky", "auto") == "vc"
    assert _alias_type_for_row("a ny", "auto") == "vc"
    assert _alias_type_for_row("n i", "auto") == "vcv"
    assert _alias_type_for_row("ny a", "auto") == "vcv"
    assert _alias_type_for_row("kw i", "auto") == "vcv"
    assert _alias_type_for_row("w a", "auto") == "vcv"
    assert _alias_type_for_row("u を", "auto") == "vv"


def test_japanese_romaji_cvvc_slots_keep_y_w_onsets():
    slots = build_filename_slots("wa-wi-we-wo-wa-u-wa-n-wa.wav", language="japanese", format_type="cvvc")

    assert filename_phone_sequence_from_slots(slots) == (
        "w",
        "a",
        "w",
        "i",
        "w",
        "e",
        "w",
        "o",
        "w",
        "a",
        "u",
        "w",
        "a",
        "n",
        "w",
        "a",
    )
    assert [(slot.token, slot.onset, slot.vowel, slot.onset_phones) for slot in slots[:4]] == [
        ("wa", "w", "a", ("w",)),
        ("wi", "w", "i", ("w",)),
        ("we", "w", "e", ("w",)),
        ("wo", "w", "o", ("w",)),
    ]


def test_japanese_romaji_cvvc_slots_keep_yoon_onset_clusters():
    slots = build_filename_slots("rya-ryu-rye-ryo.wav", language="japanese", format_type="cvvc")

    assert filename_phone_sequence_from_slots(slots) == (
        "r",
        "y",
        "a",
        "r",
        "y",
        "u",
        "r",
        "y",
        "e",
        "r",
        "y",
        "o",
    )
    assert [(slot.token, slot.onset, slot.vowel, slot.onset_phones) for slot in slots] == [
        ("rya", "ry", "a", ("r", "y")),
        ("ryu", "ry", "u", ("r", "y")),
        ("rye", "ry", "e", ("r", "y")),
        ("ryo", "ry", "o", ("r", "y")),
    ]


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


def test_template_terminal_dash_row_does_not_reject_hsmm_sequence():
    rows = [
        parse_template_oto_line("_a.wav=- a,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a a,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a -,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows([row for row in rows if row is not None], ["a", "a"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (1, "cv_head", "cv_boundary"),
        (1, "vv", "vowel_nucleus"),
    ]


def test_cvvc_vv_alias_targeting_vowel_after_onset_uses_cv_boundary_slot():
    rows = [
        parse_template_oto_line("wewo.wav=e o,0,0,0,0,0"),
        parse_template_oto_line("wewo.wav=w o,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows(
        [row for row in rows if row is not None],
        ["w", "e", "w", "o"],
    )

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (3, "vv", "cv_boundary"),
    ]


def test_japanese_n_onset_aliases_do_not_treat_n_as_vowel_transition():
    na_rows = [
        OtoTemplateRow("na.wav", "- n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("na.wav", "\u306a", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("na.wav", "n a", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("na.wav", "a n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("na.wav", "\u306b", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("na.wav", "n i", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    na_slots = expected_slots_for_template_rows(na_rows, ["n", "a", "n", "i"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in na_slots] == [
        (0, "cv_head", "phone_change"),
        (1, "cv", "cv_boundary"),
        (2, "vc", "phone_change"),
        (3, "implicit_cv", "cv_boundary"),
    ]

    ny_rows = [
        OtoTemplateRow("nya.wav", "- ny", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("nya.wav", "\u306b\u3083", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("nya.wav", "ny a", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    ny_slots = expected_slots_for_template_rows(ny_rows, ["n", "y", "a"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in ny_slots] == [
        (0, "cv_head", "phone_change"),
        (2, "cv", "cv_boundary"),
    ]

    terminal_n_rows = [
        OtoTemplateRow("on.wav", "o n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("on.wav", "n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    terminal_n_slots = expected_slots_for_template_rows(terminal_n_rows, ["o", "n"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in terminal_n_slots] == [
        (1, "vv", "vowel_nucleus"),
    ]


def test_cvvc_template_rows_share_auxiliary_cv_targets_in_order():
    rows = [
        parse_template_oto_line("a-ka.wav=- a,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=あ,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=a,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=a k,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=か,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=k a,0,0,0,0,0"),
        parse_template_oto_line("a-ka.wav=a -,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows([row for row in rows if row is not None], ["a", "k", "a"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (0, "cv_head", "cv_boundary"),
        (0, "v", "vowel_nucleus"),
        (1, "vc", "phone_change"),
        (2, "implicit_cv", "cv_boundary"),
    ]


def test_cvvc_alias_targets_keep_kw_source_rows_on_split_kana_slots():
    rows = [
        OtoTemplateRow("kwa.wav", "- kw", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "\u304f\u3041", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "k a", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "a kw", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "\u304f\u3043", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "kw i", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "i kw", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "\u304f", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "k u", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "u kw", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "\u304f\u3047", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("kwa.wav", "kw e", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(
        rows,
        ["k", "w", "a", "k", "w", "i", "k", "u", "k", "w", "e"],
    ) == [0, 2, 2, 3, 5, 5, 6, 7, 7, 8, 10, 10]

    slots = expected_slots_for_template_rows(rows, ["k", "w", "a", "k", "w", "i", "k", "u", "k", "w", "e"])
    assert (slots[0].phone_index, slots[0].event_label) == (0, "phone_change")


def test_cvvc_alias_targets_allow_vc_then_kana_cv_backtrack_for_palatal_series():
    rows = [
        OtoTemplateRow("ja.wav", "- j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "a j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "i j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "u j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "e j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "o j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "n j", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "\u3058", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "\u3058\u3085", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "\u3058\u3047", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "\u3058\u3087", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ja.wav", "\u3058\u3083", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["j", "a", "j", "i", "j", "u", "j", "e", "j", "o", "j", "a", "n", "j", "i"]

    assert _assign_alias_target_indices(rows, phones) == [2, 2, 4, 6, 8, 10, 13, 3, 5, 7, 9, 11]


def test_cvvc_following_cv_block_targets_pre_tail_repeat_before_terminal_n():
    rows = [
        OtoTemplateRow("ya.wav", "- y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "a y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "u y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "e y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "o y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "i y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "n y", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "\u3086", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "\u3044\u3047", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "\u3088", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ya.wav", "\u3084", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["y", "a", "y", "u", "y", "e", "y", "o", "y", "a", "i", "y", "a", "n", "y", "a"]

    assert _assign_alias_target_indices(rows, phones) == [2, 2, 4, 6, 8, 11, 14, 3, 5, 7, 9]


def test_cvvc_following_cv_block_handles_standalone_wo_before_terminal_n():
    rows = [
        OtoTemplateRow("wa.wav", "- w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "a w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "i w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "e w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "o w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "u w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "n w", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "\u3046\u3043", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "\u3046\u3047", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "\u3092", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("wa.wav", "\u308f", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["w", "a", "w", "i", "w", "e", "w", "o", "w", "a", "u", "w", "a", "n", "w", "a"]

    assert _assign_alias_target_indices(rows, phones) == [2, 2, 4, 6, 8, 11, 14, 3, 5, 7, 9]


def test_cvvc_single_tail_cv_block_prefers_late_repeat_target():
    rows = [
        OtoTemplateRow("ha.wav", "- h", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ha.wav", "- \u306f", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ha.wav", "a h", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ha.wav", "i h", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ha.wav", "\u306f", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["h", "a", "h", "a", "i", "h", "a"]) == [0, 1, 2, 5, 6]


def test_japanese_cvvc_following_cv_block_cutoff_cap_preserves_offset_and_companion_timing():
    transition = AdaptedOtoRow(
        wav="case.wav",
        alias="a k",
        timing=OtoTiming(offset=300.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304b",
        timing=OtoTiming(offset=820.0, consonant=190.0, cutoff=-1200.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="k a",
        timing=OtoTiming(offset=820.0, consonant=190.0, cutoff=-1200.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    short_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304f",
        timing=OtoTiming(offset=1320.0, consonant=190.0, cutoff=-360.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _transition, repaired_kana, repaired_romaji, repaired_short = repair_cvvc_row_sequence(
        [transition, kana_cv, romaji_cv, short_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_kana.timing.offset == pytest.approx(820.0)
    assert repaired_kana.timing.preutterance == pytest.approx(150.0)
    assert repaired_kana.timing.overlap == pytest.approx(115.0)
    assert repaired_kana.timing.consonant == pytest.approx(190.0)
    assert repaired_kana.timing.cutoff == pytest.approx(-400.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert repaired_short.timing.cutoff == pytest.approx(-360.0)
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_kana.applied_rules
    assert any(
        warning.startswith("cvvc_following_cv_block_cutoff_capped:1200.0->400.0")
        for warning in repaired_kana.warnings
    )


def test_japanese_cvvc_following_cv_block_cutoff_cap_respects_fixed_region():
    transition = AdaptedOtoRow(
        wav="case.wav",
        alias="a w",
        timing=OtoTiming(offset=300.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    fixed_long_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3092",
        timing=OtoTiming(offset=820.0, consonant=410.0, cutoff=-1200.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _transition, repaired = repair_cvvc_row_sequence(
        [transition, fixed_long_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(820.0)
    assert repaired.timing.consonant == pytest.approx(410.0)
    assert repaired.timing.cutoff == pytest.approx(-418.0)
    assert "cvvc_following_cv_block_cutoff_cap" in repaired.applied_rules


def test_japanese_cvvc_cv_sequence_cutoff_caps_before_next_cv():
    first_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u308a\u3085",
        timing=OtoTiming(offset=990.0, consonant=190.0, cutoff=-2390.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u308a\u3047",
        timing=OtoTiming(offset=1780.0, consonant=190.0, cutoff=-1600.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_first, _next_cv = repair_cvvc_row_sequence(
        [first_cv, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_first.timing.offset == pytest.approx(990.0)
    assert repaired_first.timing.cutoff == pytest.approx(-870.0)
    assert "cvvc_cv_sequence_next_cv_cutoff_cap" in repaired_first.applied_rules
    assert any(
        warning.startswith("cvvc_cv_sequence_next_cv_cutoff_capped:2390.0->870.0")
        for warning in repaired_first.warnings
    )


def test_japanese_cvvc_cv_sequence_cutoff_keeps_near_next_cv_tail():
    first_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304d",
        timing=OtoTiming(offset=950.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3057",
        timing=OtoTiming(offset=1580.0, consonant=190.0, cutoff=-450.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_first, _next_cv = repair_cvvc_row_sequence(
        [first_cv, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_first.timing == first_cv.timing
    assert "cvvc_cv_sequence_next_cv_cutoff_cap" not in repaired_first.applied_rules


def test_japanese_cvvc_cv_next_transition_cutoff_caps_before_next_vc():
    cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u306c",
        timing=OtoTiming(offset=1340.0, consonant=190.0, cutoff=-2700.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vc = AdaptedOtoRow(
        wav="case.wav",
        alias="u n",
        timing=OtoTiming(offset=1761.74, consonant=198.29, cutoff=-215.77, preutterance=149.26, overlap=50.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u306d",
        timing=OtoTiming(offset=2190.0, consonant=190.0, cutoff=-1850.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_cv, _vc, _next_cv = repair_cvvc_row_sequence(
        [cv, vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_cv.timing.offset == pytest.approx(1340.0)
    assert repaired_cv.timing.cutoff == pytest.approx(-(1761.74 + 80.0 - 1340.0))
    assert repaired_cv.timing.preutterance == pytest.approx(70.0)
    assert repaired_cv.timing.overlap == pytest.approx(25.0)
    assert "cvvc_cv_next_transition_cutoff_cap" in repaired_cv.applied_rules
    assert any(
        warning.startswith("cvvc_cv_next_transition_cutoff_capped:2700.0->501.7")
        for warning in repaired_cv.warnings
    )


def test_japanese_cvvc_cv_next_transition_cutoff_allows_pitch_suffixed_cv_margin():
    cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304dA3",
        timing=OtoTiming(offset=950.0, consonant=190.0, cutoff=-540.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i kA3",
        timing=OtoTiming(offset=1180.0, consonant=198.0, cutoff=-216.0, preutterance=149.0, overlap=50.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_cv, _vc = repair_cvvc_row_sequence(
        [cv, vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_cv.timing == cv.timing
    assert "cvvc_cv_next_transition_cutoff_cap" not in repaired_cv.applied_rules


def test_japanese_cvvc_cv_next_transition_cutoff_caps_severe_pitch_suffixed_cv_tail():
    cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304dA3",
        timing=OtoTiming(offset=950.0, consonant=190.0, cutoff=-900.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i kA3",
        timing=OtoTiming(offset=1180.0, consonant=198.0, cutoff=-216.0, preutterance=149.0, overlap=50.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_cv, _vc = repair_cvvc_row_sequence(
        [cv, vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_cv.timing.offset == pytest.approx(950.0)
    assert repaired_cv.timing.cutoff == pytest.approx(-(1180.0 + 120.0 - 950.0))
    assert repaired_cv.timing.preutterance == pytest.approx(70.0)
    assert repaired_cv.timing.overlap == pytest.approx(25.0)
    assert "cvvc_cv_next_transition_cutoff_cap" in repaired_cv.applied_rules
    assert any(
        warning.startswith("cvvc_cv_next_transition_cutoff_capped:900.0->350.0")
        for warning in repaired_cv.warnings
    )


def test_cvvc_initial_cv_head_shares_first_transition_when_first_cv_row_is_omitted():
    rows = [
        OtoTemplateRow("sa.wav", "- s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "a s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "i s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "\u3059\u3043", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["s", "a", "s", "i", "s", "u"]) == [2, 2, 4, 3]


def test_cvvc_alias_targets_expand_sh_ch_ts_for_filename_phone_sequences():
    sh_rows = [
        OtoTemplateRow("sha.wav", "a sh", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sha.wav", "\u3057", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sha.wav", "\u3057\u3085", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    ch_rows = [
        OtoTemplateRow("cha.wav", "a ch", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("cha.wav", "\u3061", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("cha.wav", "\u3061\u3085", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    ts_rows = [
        OtoTemplateRow("tsa.wav", "a ts", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("tsa.wav", "\u3064", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("tsa.wav", "\u3064\u3045", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(sh_rows, ["s", "h", "a", "s", "h", "i", "s", "h", "u"]) == [3, 5, 8]
    assert _assign_alias_target_indices(ch_rows, ["c", "h", "a", "c", "h", "i", "c", "h", "u"]) == [3, 5, 8]
    assert _assign_alias_target_indices(ts_rows, ["t", "s", "a", "t", "s", "u"]) == [3, 5, 5]


def test_standalone_wo_alias_in_w_series_uses_cv_boundary_role():
    rows = [OtoTemplateRow("wa.wav", "\u3092", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))]

    slots = expected_slots_for_template_rows(rows, ["w", "o"])

    assert _assign_alias_target_indices(rows, ["w", "o"]) == [1]
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [(1, "cv", "cv_boundary")]


def test_template_preserve_uses_assigned_cv_role_for_standalone_wo_alias():
    row = OtoTemplateRow(
        "wa.wav",
        "\u3092",
        OtoTiming(offset=2399.09, consonant=184.13, cutoff=-386.39, preutterance=91.61, overlap=25.0),
    )
    anchor = OtoAnchor(
        anchor_abs_ms=3000.0,
        score=0.8,
        role="cv",
        expected_phone_index=7,
        vowel_start_abs_ms=970.0,
        vowel_end_abs_ms=5260.0,
        vowel_nucleus_abs_ms=3090.0,
    )

    adapted = adapt_template_row(
        row,
        anchor,
        file_duration_ms=5310.0,
        config=OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert adapted.timing.offset > 2200.0
    assert adapted.timing.preutterance < 260.0
    assert adapted.timing.consonant < 360.0


def test_template_anchor_assignment_allows_first_target_at_zero_ms():
    posterior = FramePosterior(
        wav_path="ts.wav",
        times_ms=[0.0, 10.0, 20.0],
        class_probs={
            "silence": [0.0, 0.0, 0.0],
            "consonant": [1.0, 0.0, 0.0],
            "vowel": [0.0, 0.0, 0.0],
            "other": [0.0, 0.0, 0.0],
        },
        event_scores={
            "cv_boundary": [1.0, 0.0, 0.0],
            "phone_change": [1.0, 0.0, 0.0],
            "vowel_nucleus": [0.0, 0.0, 0.0],
        },
    )
    rows = [OtoTemplateRow("ts.wav", "- ts", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))]

    anchors = assign_template_row_anchors(
        posterior,
        [
            {
                "label": "phone_change",
                "selected_time_ms": 0.0,
                "time_ms": 0.0,
                "score": 1.0,
                "expected_phone": "ts",
                "expected_phone_index": 0,
                "slot_index": 0,
                "frame_index": 0,
                "source": "filename_hsmm",
            }
        ],
        rows,
        expected_phones=["ts", "a"],
    )

    assert anchors[0] is not None
    assert anchors[0].anchor_abs_ms == 0.0
    assert anchors[0].expected_phone_index == 0


def test_cvvc_template_backtrack_anchor_is_not_monotonic_repaired():
    from core.mfa_free_oto.workflow import _repair_anchor_monotonicity

    anchors = [
        OtoAnchor(anchor_abs_ms=980.0, score=0.8, expected_phone_index=2),
        OtoAnchor(anchor_abs_ms=4370.0, score=0.8, expected_phone_index=13),
        OtoAnchor(
            anchor_abs_ms=1222.0,
            score=0.8,
            expected_phone_index=3,
            warnings=("alias_target_backtrack",),
        ),
        OtoAnchor(anchor_abs_ms=2070.0, score=0.8, expected_phone_index=5),
    ]

    repaired = _repair_anchor_monotonicity(anchors)

    assert [anchor.anchor_abs_ms for anchor in repaired] == [980.0, 4370.0, 1222.0, 2070.0]
    assert not any(
        warning.startswith("monotonic_repaired")
        for anchor in repaired
        for warning in anchor.warnings
    )


def test_local_refine_does_not_warn_when_shared_auxiliary_target_blocks_window():
    posterior = FramePosterior(
        wav_path="case.wav",
        times_ms=[80.0, 90.0, 100.0, 110.0, 120.0],
        class_probs={
            "silence": [0.0, 0.0, 0.0, 0.0, 0.0],
            "consonant": [0.2, 0.3, 0.5, 0.9, 0.4],
            "vowel": [0.4, 0.5, 0.6, 0.7, 0.5],
            "other": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        event_scores={
            "cv_boundary": [0.1, 0.2, 0.3, 1.0, 0.2],
            "phone_change": [0.1, 0.2, 0.3, 1.0, 0.2],
            "vowel_nucleus": [0.1, 0.2, 0.3, 0.4, 0.2],
        },
    )
    anchors = [
        OtoAnchor(anchor_abs_ms=100.0, score=0.8, role="cv", expected_phone_index=1),
        OtoAnchor(anchor_abs_ms=100.0, score=0.8, role="vcv", expected_phone_index=1),
    ]

    refined = _refine_anchor_sequence_locally(posterior, anchors)

    assert "local_refine_rejected_slot_boundary" not in refined[0].warnings


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


def test_cvvc_spaced_cv_alias_prefers_cv_boundary_over_vowel_nucleus():
    times = [float(idx * 20) for idx in range(51)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    cv_frame = times.index(500.0)
    nucleus_frame = times.index(680.0)
    event_scores["cv_boundary"][cv_frame] = 0.95
    event_scores["vowel_nucleus"][nucleus_frame] = 0.95
    class_probs["vowel"][cv_frame] = 0.70
    class_probs["vowel"][nucleus_frame] = 0.90
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    row = parse_template_oto_line("v.wav=kw i,0,0,0,0,0")
    assert row is not None
    anchors = assign_template_row_anchors(
        posterior,
        [
            {
                "label": "cv_boundary",
                "selected_time_ms": 500.0,
                "score": 0.82,
                "frame_index": cv_frame,
                "expected_phone": "i",
                "expected_phone_index": 2,
                "slot_index": 0,
            },
            {
                "label": "vowel_nucleus",
                "selected_time_ms": 680.0,
                "score": 0.82,
                "frame_index": nucleus_frame,
                "expected_phone": "i",
                "expected_phone_index": 2,
                "slot_index": 0,
            },
        ],
        [row],
        use_source_timing_prior=False,
        expected_phones=["k", "w", "i"],
    )
    assert anchors[0] is not None
    assert anchors[0].role == "vcv"
    assert anchors[0].source_event_label == "cv_boundary"
    assert anchors[0].anchor_abs_ms == pytest.approx(500.0)


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


def test_cvvc_vc_bootstrap_uses_previous_vowel_context_for_offset():
    adapted = bootstrap_row(
        "v.wav",
        "a k",
        OtoAnchor(
            anchor_abs_ms=520.0,
            score=0.82,
            previous_vowel_end_abs_ms=460.0,
            vowel_end_abs_ms=760.0,
        ),
        file_duration_ms=1000.0,
        config=OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(420.0)
    assert adapted.timing.preutterance == pytest.approx(100.0)
    assert adapted.timing.cutoff == pytest.approx(-(adapted.timing.consonant + 26.0))
    assert "cvvc_vc_left_context_pre:80.0->100.0" in adapted.warnings


def test_japanese_cvvc_vc_sequence_repair_uses_previous_cv_when_context_was_capped():
    previous = AdaptedOtoRow(
        wav="case.wav",
        alias="k i",
        timing=OtoTiming(offset=330.0, consonant=160.0, cutoff=-200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    capped_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i ky",
        timing=OtoTiming(offset=735.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("cvvc_vc_left_context_pre_capped:365.0->170.0",),
    )

    _unchanged, repaired = repair_cvvc_vc_row_sequence(
        [previous, capped_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(490.0)
    assert repaired.timing.preutterance == pytest.approx(110.0)
    assert repaired.timing.overlap == pytest.approx(100.0)
    assert repaired.timing.consonant == pytest.approx(120.0)
    assert repaired.timing.cutoff == pytest.approx(-146.0)
    assert "cvvc_vc_sequence_repair" in repaired.applied_rules
    assert any(warning.startswith("cvvc_vc_sequence_repaired:735.0->490.0") for warning in repaired.warnings)


def test_japanese_cvvc_vc_sequence_repair_keeps_uncapped_w_rows():
    previous = AdaptedOtoRow(
        wav="case.wav",
        alias="w i",
        timing=OtoTiming(offset=520.0, consonant=150.0, cutoff=-180.0, preutterance=110.0, overlap=80.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    uncapped_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i w",
        timing=OtoTiming(offset=620.0, consonant=108.0, cutoff=-136.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("cvvc_vc_previous_vowel_end_ignored:1430.0",),
    )

    _previous, kept = repair_cvvc_vc_row_sequence(
        [previous, uncapped_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert kept.timing == uncapped_vc.timing
    assert "cvvc_vc_sequence_repair" not in kept.applied_rules


def test_japanese_cvvc_following_glide_repair_uses_repaired_vc_and_next_onset():
    repaired_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a w",
        timing=OtoTiming(offset=300.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_vc_sequence_repair",),
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3046\u3043",
        timing=OtoTiming(offset=820.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="w i",
        timing=OtoTiming(offset=820.0, consonant=160.0, cutoff=-560.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    next_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i w",
        timing=OtoTiming(offset=1355.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1480.0, score=0.72),
        mode="bootstrap",
    )

    rows = repair_cvvc_row_sequence(
        [repaired_vc, kana_cv, romaji_vcv, next_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )
    repaired_kana = rows[1]
    repaired_romaji = rows[2]

    assert repaired_kana.timing.offset == pytest.approx(475.0)
    assert repaired_kana.timing.preutterance == pytest.approx(110.0)
    assert repaired_kana.timing.overlap == pytest.approx(100.0)
    assert repaired_kana.timing.consonant == pytest.approx(210.0)
    assert repaired_kana.timing.cutoff == pytest.approx(-400.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert "cvvc_following_glide_repair" in repaired_kana.applied_rules
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_kana.applied_rules
    assert "cvvc_following_glide_repair" in repaired_romaji.applied_rules
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_romaji.applied_rules


def test_japanese_cvvc_following_yoon_repair_uses_repaired_vc_pre_abs():
    repaired_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i ky",
        timing=OtoTiming(offset=490.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_vc_sequence_repair",),
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304d\u3085",
        timing=OtoTiming(offset=860.0, consonant=190.0, cutoff=-510.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="ky u",
        timing=OtoTiming(offset=850.0, consonant=160.0, cutoff=-470.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    rows = repair_cvvc_row_sequence(
        [repaired_vc, kana_cv, romaji_vcv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )
    repaired_kana = rows[1]
    repaired_romaji = rows[2]

    assert repaired_kana.timing.offset == pytest.approx(600.0)
    assert repaired_kana.timing.preutterance == pytest.approx(70.0)
    assert repaired_kana.timing.overlap == pytest.approx(60.0)
    assert repaired_kana.timing.consonant == pytest.approx(170.0)
    assert repaired_kana.timing.cutoff == pytest.approx(-400.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert "cvvc_following_yoon_repair" in repaired_kana.applied_rules
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_kana.applied_rules
    assert "cvvc_following_yoon_repair" in repaired_romaji.applied_rules
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_romaji.applied_rules


def test_japanese_cvvc_post_yoon_vc_repair_uses_previous_yoon_cv_offset():
    repaired_yoon = AdaptedOtoRow(
        wav="case.wav",
        alias="ky u",
        timing=OtoTiming(offset=600.0, consonant=170.0, cutoff=-770.0, preutterance=70.0, overlap=60.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_following_yoon_repair",),
    )
    late_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="u ky",
        timing=OtoTiming(offset=1275.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    rows = repair_cvvc_row_sequence(
        [repaired_yoon, late_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )
    repaired_vc = rows[1]

    assert repaired_vc.timing.offset == pytest.approx(925.0)
    assert repaired_vc.timing.preutterance == pytest.approx(110.0)
    assert repaired_vc.timing.overlap == pytest.approx(100.0)
    assert repaired_vc.timing.consonant == pytest.approx(120.0)
    assert repaired_vc.timing.cutoff == pytest.approx(-146.0)
    assert "cvvc_post_yoon_vc_repair" in repaired_vc.applied_rules
    assert any(warning.startswith("cvvc_post_yoon_vc_repaired:1275.0->925.0") for warning in repaired_vc.warnings)


def test_japanese_cvvc_following_yoon_repair_reruns_after_post_yoon_vc_repair():
    repaired_yoon = AdaptedOtoRow(
        wav="case.wav",
        alias="gy e",
        timing=OtoTiming(offset=1160.0, consonant=170.0, cutoff=-470.0, preutterance=70.0, overlap=60.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_following_yoon_repair",),
    )
    late_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="e gy",
        timing=OtoTiming(offset=1540.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_vc_sequence_repair",),
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304e\u3087",
        timing=OtoTiming(offset=1730.0, consonant=190.0, cutoff=-250.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="gy o",
        timing=OtoTiming(offset=1720.0, consonant=160.0, cutoff=-210.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _previous, repaired_vc, repaired_kana, repaired_romaji = repair_cvvc_row_sequence(
        [repaired_yoon, late_vc, kana_cv, romaji_vcv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_vc.timing.offset == pytest.approx(1435.0)
    assert "cvvc_post_yoon_vc_repair" in repaired_vc.applied_rules
    assert repaired_kana.timing.offset == pytest.approx(1545.0)
    assert repaired_kana.timing.preutterance == pytest.approx(70.0)
    assert repaired_kana.timing.overlap == pytest.approx(60.0)
    assert repaired_kana.timing.consonant == pytest.approx(170.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert "cvvc_following_yoon_repair" in repaired_kana.applied_rules


def test_japanese_cvvc_post_yoon_vc_repair_handles_late_e_yoon_chain():
    repaired_yoon = AdaptedOtoRow(
        wav="case.wav",
        alias="ky u",
        timing=OtoTiming(offset=600.0, consonant=170.0, cutoff=-770.0, preutterance=70.0, overlap=60.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_following_yoon_repair",),
    )
    post_yoon_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="u ky",
        timing=OtoTiming(offset=1275.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_yoon_e = AdaptedOtoRow(
        wav="case.wav",
        alias="ky e",
        timing=OtoTiming(offset=1250.0, consonant=160.0, cutoff=-190.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_e_yoon_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="e ky",
        timing=OtoTiming(offset=1400.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304d\u3087",
        timing=OtoTiming(offset=1510.0, consonant=170.0, cutoff=-330.0, preutterance=70.0, overlap=60.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="ky o",
        timing=OtoTiming(offset=1510.0, consonant=170.0, cutoff=-330.0, preutterance=70.0, overlap=60.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _yoon_u, repaired_u_ky, repaired_ky_e, repaired_e_ky, repaired_kana, repaired_romaji = repair_cvvc_row_sequence(
        [repaired_yoon, post_yoon_vc, late_yoon_e, late_e_yoon_vc, kana_cv, romaji_vcv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_u_ky.timing.offset == pytest.approx(925.0)
    assert repaired_ky_e.timing.offset == pytest.approx(1035.0)
    assert repaired_e_ky.timing.offset == pytest.approx(1310.0)
    assert repaired_kana.timing.offset == pytest.approx(1420.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert "cvvc_following_yoon_repair" in repaired_ky_e.applied_rules
    assert "cvvc_post_yoon_vc_repair" in repaired_e_ky.applied_rules


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


def test_korean_vcv_bootstrap_applies_hsmm_anchor_lead_only_to_hsmm_events():
    hsmm = bootstrap_row(
        "vcv.wav",
        "a gyeo",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            vowel_end_abs_ms=980.0,
            warnings=("slot_decoded_event", "event_source:filename_hsmm"),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="korean", format_type="vcv", alias_type="vcv"),
    )
    runtime = bootstrap_row(
        "vcv.wav",
        "a gyeo",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            vowel_end_abs_ms=980.0,
            warnings=("slot_decoded_event",),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="korean", format_type="vcv", alias_type="vcv"),
    )

    assert hsmm.timing.offset == pytest.approx(80.0)
    assert runtime.timing.offset == pytest.approx(320.0)
    assert hsmm.timing.cutoff == pytest.approx(runtime.timing.cutoff)
    assert "hsmm_anchor_lead:620.0->380.0" in hsmm.warnings


def test_anchors_from_prediction_preserves_event_source_for_hsmm_policy():
    posterior = FramePosterior(
        wav_path="case.wav",
        times_ms=[0.0, 10.0, 20.0],
        class_probs={
            "silence": [0.0, 0.0, 0.0],
            "consonant": [0.0, 0.2, 0.0],
            "vowel": [0.2, 0.8, 0.8],
            "other": [0.0, 0.0, 0.0],
        },
        event_scores={
            "cv_boundary": [0.1, 0.9, 0.1],
            "vowel_nucleus": [0.1, 0.2, 0.9],
            "phone_change": [0.1, 0.8, 0.1],
        },
    )

    anchors = anchors_from_prediction(
        posterior,
        (
            {
                "label": "vv_boundary",
                "selected_time_ms": 10.0,
                "score": 0.8,
                "expected_phone": "a",
                "expected_phone_index": 0,
                "slot_index": 0,
                "source": "filename_hsmm",
            },
        ),
    )

    assert anchors
    assert "event_source:filename_hsmm" in anchors[0].warnings


def test_assign_template_row_anchors_preserves_source_event_label_after_role_rewrite():
    posterior = FramePosterior(
        wav_path="case.wav",
        times_ms=[0.0, 50.0, 100.0, 150.0],
        class_probs={
            "silence": [0.0, 0.0, 0.0, 0.0],
            "consonant": [0.0, 0.5, 0.2, 0.0],
            "vowel": [0.2, 0.6, 0.9, 0.8],
            "other": [0.0, 0.0, 0.0, 0.0],
        },
        event_scores={
            "cv_boundary": [0.1, 0.3, 0.9, 0.2],
            "vowel_nucleus": [0.1, 0.2, 0.4, 0.8],
            "phone_change": [0.1, 0.7, 0.2, 0.1],
        },
    )
    rows = [
        OtoTemplateRow(
            wav="case.wav",
            alias="a ta",
            timing=OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0),
        )
    ]

    anchors = assign_template_row_anchors(
        posterior,
        (
            {
                "label": "cv_boundary",
                "selected_time_ms": 100.0,
                "score": 0.9,
                "expected_phone": "a",
                "expected_phone_index": 2,
                "slot_index": 0,
                "source": "filename_hsmm",
            },
        ),
        rows,
        expected_phones=("a", "t", "a"),
    )

    assert anchors[0] is not None
    assert anchors[0].role == "vcv"
    assert anchors[0].source_event_label == "cv_boundary"
    assert "event_source:filename_hsmm" in anchors[0].warnings


def test_japanese_vcv_hsmm_lead_uses_source_event_label():
    boundary = bootstrap_row(
        "vcv.wav",
        "a ti",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            vowel_end_abs_ms=980.0,
            source_event_label="cv_boundary",
            warnings=("slot_decoded_event", "event_source:filename_hsmm"),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="vcv"),
    )
    nucleus = bootstrap_row(
        "vcv.wav",
        "a ti",
        OtoAnchor(
            anchor_abs_ms=620.0,
            score=0.72,
            vowel_end_abs_ms=980.0,
            source_event_label="vowel_nucleus",
            warnings=("slot_decoded_event", "event_source:filename_hsmm"),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="vcv"),
    )

    assert boundary.timing.offset == pytest.approx(345.0)
    assert nucleus.timing.offset == pytest.approx(170.0)
    assert "hsmm_anchor_lead:620.0->495.0" in boundary.warnings
    assert "hsmm_anchor_lead:620.0->320.0" in nucleus.warnings


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


def test_japanese_cvvc_initial_consonant_cv_head_caps_cutoff_before_vowel_body():
    adapted = bootstrap_row(
        "cvvc.wav",
        "- ky",
        OtoAnchor(
            anchor_abs_ms=150.0,
            score=0.72,
            vowel_start_abs_ms=140.0,
            vowel_end_abs_ms=800.0,
            source_event_label="phone_change",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=1500.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert "hsmm_anchor_lead:150.0->60.0" in adapted.warnings
    assert adapted.timing.offset + abs(adapted.timing.cutoff) == pytest.approx(195.0)
    assert adapted.timing.consonant < abs(adapted.timing.cutoff)
    assert any(warning.startswith("cvvc_initial_cv_cutoff_clamped:") for warning in adapted.warnings)


def test_japanese_cvvc_initial_vowel_cv_head_caps_excessive_cutoff_tail():
    adapted = bootstrap_row(
        "cvvc.wav",
        "- a",
        OtoAnchor(
            anchor_abs_ms=230.0,
            score=0.72,
            vowel_start_abs_ms=0.0,
            vowel_end_abs_ms=1400.0,
            source_event_label="vv_boundary",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=1800.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(110.0)
    assert abs(adapted.timing.cutoff) == pytest.approx(560.0)
    assert adapted.timing.consonant < abs(adapted.timing.cutoff)
    assert any(warning.startswith("cvvc_initial_vowel_cutoff_clamped:") for warning in adapted.warnings)


def test_japanese_cvvc_initial_vowel_cv_head_cutoff_uses_next_vc_offset():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- a",
        timing=OtoTiming(offset=110.0, consonant=160.0, cutoff=-420.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vowel = AdaptedOtoRow(
        wav="case.wav",
        alias="a",
        timing=OtoTiming(offset=0.0, consonant=498.4, cutoff=-530.0, preutterance=379.6, overlap=192.4),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    next_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a k",
        timing=OtoTiming(offset=375.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _vowel, _next_vc = repair_cvvc_row_sequence(
        [initial, vowel, next_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.cutoff == pytest.approx(-265.0)
    assert repaired.timing.offset == pytest.approx(110.0)
    assert "cvvc_initial_vowel_cv_head_cutoff_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_vowel_cv_head_cutoff_repaired:530.0->375.0")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_uses_following_cv():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- z",
        timing=OtoTiming(offset=750.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a z",
        timing=OtoTiming(offset=905.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    second_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i z",
        timing=OtoTiming(offset=1420.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u305a\u3041",
        timing=OtoTiming(offset=1242.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _first_vc, _second_vc, _following_cv = repair_cvvc_row_sequence(
        [initial, first_vc, second_vc, following_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(1242.0)
    assert repaired.timing.preutterance == pytest.approx(35.0)
    assert repaired.timing.overlap == pytest.approx(10.0)
    assert repaired.timing.consonant == pytest.approx(65.0)
    assert repaired.timing.cutoff == pytest.approx(-95.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_following_cv_reference:")
        for warning in repaired.warnings
    )
    assert not any(
        warning.startswith("cvvc_initial_consonant_cv_head_shift_capped:")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_still_caps_large_shift():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- z",
        timing=OtoTiming(offset=750.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a z",
        timing=OtoTiming(offset=905.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u305a\u3041",
        timing=OtoTiming(offset=1500.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _first_vc, _following_cv = repair_cvvc_row_sequence(
        [initial, first_vc, following_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(1350.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_shift_capped:750.0->600.0")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_uses_immediate_vowel_head():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- h",
        timing=OtoTiming(offset=60.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_head = AdaptedOtoRow(
        wav="case.wav",
        alias="- \u306f",
        timing=OtoTiming(offset=660.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a h",
        timing=OtoTiming(offset=925.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _following_head, _first_vc = repair_cvvc_row_sequence(
        [initial, following_head, first_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(660.0)
    assert repaired.timing.preutterance == pytest.approx(35.0)
    assert repaired.timing.overlap == pytest.approx(10.0)
    assert repaired.timing.consonant == pytest.approx(65.0)
    assert repaired.timing.cutoff == pytest.approx(-95.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_following_cv_reference:- \u306f")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_keeps_small_following_cv_gap():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- sh",
        timing=OtoTiming(offset=1340.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a sh",
        timing=OtoTiming(offset=1390.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3057",
        timing=OtoTiming(offset=1415.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _first_vc, _following_cv = repair_cvvc_row_sequence(
        [initial, first_vc, following_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing == initial.timing
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" not in repaired.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_uses_terminal_reference_grid():
    initial = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="- a",
        timing=OtoTiming(offset=660.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a a",
        timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a i",
        timing=OtoTiming(offset=940.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a2 = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="i a",
        timing=OtoTiming(offset=1300.0, consonant=160.0, cutoff=-900.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a -",
        timing=OtoTiming(offset=2500.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_head, repaired_a, repaired_i, repaired_a2, _terminal = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, vv_a2, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing.offset == pytest.approx(950.0)
    assert repaired_head.timing.cutoff == pytest.approx(-360.0)
    assert repaired_a.timing.offset == pytest.approx(1070.0)
    assert repaired_i.timing.offset == pytest.approx(1570.0)
    assert repaired_a2.timing.offset == pytest.approx(2070.0)
    for repaired in (repaired_a, repaired_i, repaired_a2):
        assert repaired.timing.preutterance == pytest.approx(300.0)
        assert repaired.timing.overlap == pytest.approx(100.0)
        assert repaired.timing.consonant == pytest.approx(420.0)
        assert repaired.timing.cutoff == pytest.approx(-600.0)
        assert "cvvc_pure_vowel_sequence_row_repair" in repaired.applied_rules
    assert "cvvc_pure_vowel_sequence_head_repair" in repaired_head.applied_rules
    assert "cvvc_pure_vowel_sequence_head_cutoff_repair" in repaired_head.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_keeps_existing_head_cutoff_when_not_short():
    initial = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="- a",
        timing=OtoTiming(offset=660.0, consonant=160.0, cutoff=-320.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a a",
        timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a i",
        timing=OtoTiming(offset=940.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="i -",
        timing=OtoTiming(offset=2000.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_head, *_rest = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing.cutoff == pytest.approx(-320.0)
    assert "cvvc_pure_vowel_sequence_head_cutoff_repair" not in repaired_head.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_includes_moraic_n_transition_rows():
    initial = AdaptedOtoRow(
        wav="_n-n-a-n.wav",
        alias="- n",
        timing=OtoTiming(offset=620.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    moraic_n = AdaptedOtoRow(
        wav="_n-n-a-n.wav",
        alias="n n",
        timing=OtoTiming(offset=780.0, consonant=185.0, cutoff=-211.0, preutterance=145.0, overlap=110.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    n_to_a = AdaptedOtoRow(
        wav="_n-n-a-n.wav",
        alias="n a",
        timing=OtoTiming(offset=1260.0, consonant=160.0, cutoff=-900.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    a_to_n = AdaptedOtoRow(
        wav="_n-n-a-n.wav",
        alias="a n",
        timing=OtoTiming(offset=1885.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_n-n-a-n.wav",
        alias="n -",
        timing=OtoTiming(offset=2500.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    _head, repaired_n, repaired_na, repaired_an, _terminal = repair_cvvc_row_sequence(
        [initial, moraic_n, n_to_a, a_to_n, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_n.timing.offset == pytest.approx(1070.0)
    assert repaired_na.timing.offset == pytest.approx(1570.0)
    assert repaired_an.timing.offset == pytest.approx(2070.0)
    for repaired in (repaired_n, repaired_na, repaired_an):
        assert repaired.timing.preutterance == pytest.approx(300.0)
        assert "cvvc_pure_vowel_sequence_row_repair" in repaired.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_requires_terminal_anchor_rule():
    initial = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="- a",
        timing=OtoTiming(offset=660.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a a",
        timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a i",
        timing=OtoTiming(offset=940.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="i -",
        timing=OtoTiming(offset=2500.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired_head, repaired_a, repaired_i, _terminal = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing == initial.timing
    assert repaired_a.timing == vv_a.timing
    assert repaired_i.timing == vv_i.timing


def test_japanese_cvvc_pure_vowel_onset_sequence_uses_reliable_vowel_start_grid():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=1160.0,
        vowel_end_abs_ms=4900.0,
    )
    initial = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="- a",
        timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="a a",
        timing=OtoTiming(offset=600.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
        warnings=("nucleus_span_too_short",),
    )
    standalone_a = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="a",
        timing=OtoTiming(offset=834.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.84),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="a i",
        timing=OtoTiming(offset=1240.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    trailing_r = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="i RA3",
        timing=OtoTiming(offset=3045.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )

    repaired_head, repaired_aa, repaired_a, repaired_ai, repaired_r = repair_cvvc_row_sequence(
        [initial, vv_a, standalone_a, vv_i, trailing_r],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired_head.timing.offset == pytest.approx(1040.0)
    assert repaired_head.timing.cutoff == pytest.approx(-840.0)
    assert repaired_aa.timing.offset == pytest.approx(1410.0)
    assert repaired_a.timing.offset == pytest.approx(1660.0)
    assert repaired_ai.timing.offset == pytest.approx(1910.0)
    assert repaired_r.timing.offset == pytest.approx(4800.0)
    assert repaired_aa.timing.preutterance == pytest.approx(250.0)
    assert repaired_aa.timing.overlap == pytest.approx(83.0)
    assert repaired_a.timing.preutterance == pytest.approx(0.0)
    assert repaired_a.timing.overlap == pytest.approx(0.0)
    assert "cvvc_pure_vowel_onset_head_repair" in repaired_head.applied_rules
    assert "cvvc_pure_vowel_onset_head_cutoff_repair" in repaired_head.applied_rules
    assert "cvvc_pure_vowel_onset_transition_repair" in repaired_aa.applied_rules
    assert "cvvc_pure_vowel_onset_v_repair" in repaired_a.applied_rules
    assert "cvvc_pure_vowel_onset_trailing_r_repair" in repaired_r.applied_rules


def test_japanese_cvvc_terminal_release_r_uses_late_file_tail_without_lowercase_r():
    lowercase_r = AdaptedOtoRow(
        wav="_a-r.wav",
        alias="a rA3",
        timing=OtoTiming(offset=1320.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    release_r = AdaptedOtoRow(
        wav="_a-r.wav",
        alias="a RA3",
        timing=OtoTiming(offset=4100.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    kept_lowercase_r, repaired_release_r = repair_cvvc_row_sequence(
        [lowercase_r, release_r],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5874.467,
    )

    assert kept_lowercase_r.timing.offset == pytest.approx(1320.0)
    assert kept_lowercase_r.applied_rules == ()
    assert repaired_release_r.timing.offset == pytest.approx(4754.467)
    assert repaired_release_r.timing.consonant == pytest.approx(420.0)
    assert repaired_release_r.timing.cutoff == pytest.approx(-600.0)
    assert repaired_release_r.timing.preutterance == pytest.approx(250.0)
    assert repaired_release_r.timing.overlap == pytest.approx(83.0)
    assert "cvvc_terminal_release_r_tail_repair" in repaired_release_r.applied_rules


def test_japanese_cvvc_standalone_release_r_uses_midpoint_offset_only():
    release_r = AdaptedOtoRow(
        wav="aR.wav",
        alias="a R",
        timing=OtoTiming(offset=1138.82, consonant=340.0, cutoff=-557.18, preutterance=139.57, overlap=50.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [release_r],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3799.365,
    )

    assert repaired.timing.offset == pytest.approx(1899.6825)
    assert repaired.timing.consonant == pytest.approx(340.0)
    assert repaired.timing.cutoff == pytest.approx(-557.18)
    assert repaired.timing.preutterance == pytest.approx(139.57)
    assert repaired.timing.overlap == pytest.approx(50.0)
    assert "cvvc_standalone_release_r_midpoint_repair" in repaired.applied_rules


def test_japanese_cvvc_pure_vowel_onset_sequence_repairs_headless_transition_block():
    def anchor(target_index: int) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=1400.0 + (target_index * 200.0),
            score=0.4,
            role="vv",
            vowel_start_abs_ms=1230.0,
            vowel_end_abs_ms=2550.0,
            expected_phone_index=target_index,
        )

    first = AdaptedOtoRow(
        wav="_o-u-n-a.wav",
        alias="u n",
        timing=OtoTiming(offset=1285.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(2),
        mode="bootstrap",
    )
    second = AdaptedOtoRow(
        wav="_o-u-n-a.wav",
        alias="n a",
        timing=OtoTiming(offset=1430.0, consonant=160.0, cutoff=-1070.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=anchor(3),
        mode="bootstrap",
    )
    third = AdaptedOtoRow(
        wav="_o-u-n-a.wav",
        alias="a n",
        timing=OtoTiming(offset=1785.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(4),
        mode="bootstrap",
    )

    repaired_first, repaired_second, repaired_third = repair_cvvc_row_sequence(
        [first, second, third],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired_first.timing.offset == pytest.approx(1980.0)
    assert repaired_second.timing.offset == pytest.approx(2480.0)
    assert repaired_third.timing.offset == pytest.approx(2980.0)
    assert repaired_first.timing.preutterance == pytest.approx(250.0)
    assert repaired_first.timing.overlap == pytest.approx(83.0)
    assert "cvvc_pure_vowel_onset_transition_repair" in repaired_first.applied_rules


def test_japanese_cvvc_headless_vc_onset_sequence_repairs_early_consonant_transitions():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1450.0,
        score=0.4,
        role="vc",
        vowel_start_abs_ms=1140.0,
        vowel_end_abs_ms=4200.0,
    )
    first = AdaptedOtoRow(
        wav="_i-g-i-z.wav",
        alias="i gA3",
        timing=OtoTiming(offset=825.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    second = AdaptedOtoRow(
        wav="_i-g-i-z.wav",
        alias="i zA3",
        timing=OtoTiming(offset=1576.4, consonant=160.0, cutoff=-1070.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    third = AdaptedOtoRow(
        wav="_i-g-i-z.wav",
        alias="i dA3",
        timing=OtoTiming(offset=2442.4, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )

    repaired_first, repaired_second, repaired_third = repair_cvvc_row_sequence(
        [first, second, third],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired_first.timing.offset == pytest.approx(1390.0)
    assert repaired_second.timing.offset == pytest.approx(2390.0)
    assert repaired_third.timing.offset == pytest.approx(3390.0)
    assert repaired_first.timing.consonant == pytest.approx(300.0)
    assert repaired_first.timing.cutoff == pytest.approx(-330.0)
    assert repaired_first.timing.preutterance == pytest.approx(250.0)
    assert repaired_first.timing.overlap == pytest.approx(83.0)
    assert "cvvc_headless_vc_onset_repair" in repaired_first.applied_rules
    assert "cvvc_pure_vowel_onset_transition_repair" not in repaired_first.applied_rules


def test_japanese_cvvc_headless_vc_onset_sequence_preserves_special_breath_aliases():
    anchor = OtoAnchor(
        anchor_abs_ms=1500.0,
        score=0.9,
        role="vc",
        vowel_start_abs_ms=1200.0,
        vowel_end_abs_ms=3000.0,
        next_vowel_abs_ms=2400.0,
    )
    source_first = OtoTiming(offset=1000.0, consonant=111.0, cutoff=-222.0, preutterance=88.0, overlap=44.0)
    source_second = OtoTiming(offset=1500.0, consonant=112.0, cutoff=-223.0, preutterance=89.0, overlap=45.0)
    first = AdaptedOtoRow(
        wav="_a_breath.wav",
        alias="a \u5438",
        timing=source_first,
        source_timing=source_first,
        anchor=anchor,
        mode="template-special",
        warnings=("special_alias_source_timing_preserved",),
        applied_rules=("special_alias_source_timing_preserve",),
    )
    second = AdaptedOtoRow(
        wav="_a_breath.wav",
        alias="e \u5438",
        timing=source_second,
        source_timing=source_second,
        anchor=anchor,
        mode="template-special",
        warnings=("special_alias_source_timing_preserved",),
        applied_rules=("special_alias_source_timing_preserve",),
    )

    repaired_first, repaired_second = repair_cvvc_row_sequence(
        [first, second],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired_first.timing == source_first
    assert repaired_second.timing == source_second
    assert "special_alias_source_timing_preserve" in repaired_first.applied_rules
    assert "cvvc_headless_vc_onset_repair" not in repaired_first.applied_rules
    assert "cvvc_headless_vc_onset_repair" not in repaired_second.applied_rules


def test_japanese_cvvc_headless_vc_onset_sequence_prefers_first_following_vowel_reference():
    def anchor(
        *,
        anchor_abs_ms: float,
        vowel_start_abs_ms: float,
        vowel_end_abs_ms: float,
        next_vowel_abs_ms: float,
    ) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=anchor_abs_ms,
            score=0.4,
            role="vc",
            vowel_start_abs_ms=vowel_start_abs_ms,
            vowel_end_abs_ms=vowel_end_abs_ms,
            next_vowel_abs_ms=next_vowel_abs_ms,
        )

    first = AdaptedOtoRow(
        wav="_n-ga-n-za.wav",
        alias="n gA3",
        timing=OtoTiming(offset=555.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(
            anchor_abs_ms=770.0,
            vowel_start_abs_ms=770.0,
            vowel_end_abs_ms=770.0,
            next_vowel_abs_ms=1236.5,
        ),
        mode="bootstrap",
    )
    second = AdaptedOtoRow(
        wav="_n-ga-n-za.wav",
        alias="n zA3",
        timing=OtoTiming(offset=1385.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(
            anchor_abs_ms=1600.0,
            vowel_start_abs_ms=1600.0,
            vowel_end_abs_ms=1600.0,
            next_vowel_abs_ms=2077.5,
        ),
        mode="bootstrap",
    )
    third = AdaptedOtoRow(
        wav="_n-ga-n-za.wav",
        alias="n dA3",
        timing=OtoTiming(offset=2575.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(
            anchor_abs_ms=2790.0,
            vowel_start_abs_ms=2630.0,
            vowel_end_abs_ms=3550.0,
            next_vowel_abs_ms=3236.5,
        ),
        mode="bootstrap",
    )

    repaired_first, repaired_second, repaired_third = repair_cvvc_row_sequence(
        [first, second, third],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired_first.timing.offset == pytest.approx(1336.5)
    assert repaired_second.timing.offset == pytest.approx(2336.5)
    assert repaired_third.timing.offset == pytest.approx(3336.5)
    assert any("cvvc_headless_vc_onset_reference:1086.5" in item for item in repaired_first.warnings)


def test_japanese_cvvc_regular_pair_onset_sequence_repairs_early_vc_cv_prefix():
    def anchor(next_vowel_abs_ms: float) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=1180.0,
            score=0.5,
            role="vc",
            vowel_start_abs_ms=1180.0,
            vowel_end_abs_ms=1590.0,
            next_vowel_abs_ms=next_vowel_abs_ms,
        )

    rows = [
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a k",
            timing=OtoTiming(offset=575.0, consonant=254.0, cutoff=-272.0, preutterance=205.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="ka",
            timing=OtoTiming(offset=980.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a s",
            timing=OtoTiming(offset=1485.0, consonant=253.0, cutoff=-271.0, preutterance=204.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="sa",
            timing=OtoTiming(offset=1650.0, consonant=190.0, cutoff=-450.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1338.5, 1678.5, 1838.5, 2178.5])
    assert all("cvvc_regular_pair_onset_repair" in row.applied_rules for row in repaired)
    assert repaired[0].timing.cutoff == pytest.approx(-262.0)
    assert any("cvvc_regular_pair_onset_reference:1308.5" in item for item in repaired[0].warnings)


def test_japanese_cvvc_regular_pair_onset_sequence_handles_release_r_tail_gap():
    def anchor(next_vowel_abs_ms: float) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=1180.0,
            score=0.5,
            role="vc",
            vowel_start_abs_ms=1180.0,
            vowel_end_abs_ms=1590.0,
            next_vowel_abs_ms=next_vowel_abs_ms,
        )

    rows = [
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="i kyA3",
            timing=OtoTiming(offset=873.75, consonant=249.0, cutoff=-267.0, preutterance=200.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="\u304dA3",
            timing=OtoTiming(offset=950.0, consonant=190.0, cutoff=-400.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="i shA3",
            timing=OtoTiming(offset=1435.0, consonant=252.0, cutoff=-269.0, preutterance=203.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="\u3057A3",
            timing=OtoTiming(offset=1580.0, consonant=190.0, cutoff=-450.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="i dyA3",
            timing=OtoTiming(offset=4125.0, consonant=256.0, cutoff=-274.0, preutterance=207.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="i RA3",
            timing=OtoTiming(offset=4754.0, consonant=420.0, cutoff=-600.0, preutterance=250.0, overlap=83.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_i-ki-shi-di.wav",
            alias="\u3067\u3043A3",
            timing=OtoTiming(offset=4360.0, consonant=190.0, cutoff=-620.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=anchor(1267.5),
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx(
        [1297.5, 1637.5, 1797.5, 2137.5, 4125.0, 4754.0, 4360.0]
    )
    assert all("cvvc_regular_pair_onset_repair" in repaired[index].applied_rules for index in range(4))
    assert "cvvc_regular_pair_onset_repair" not in repaired[5].applied_rules
    assert "cvvc_regular_pair_onset_repair" not in repaired[6].applied_rules


def test_japanese_cvvc_regular_pair_onset_sequence_skips_partial_prefix_without_pitch_suffix():
    def anchor(next_vowel_abs_ms: float) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=1180.0,
            score=0.5,
            role="vc",
            vowel_start_abs_ms=1180.0,
            vowel_end_abs_ms=1590.0,
            next_vowel_abs_ms=next_vowel_abs_ms,
        )

    rows = [
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a k",
            timing=OtoTiming(offset=575.0, consonant=254.0, cutoff=-272.0, preutterance=205.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="ka",
            timing=OtoTiming(offset=980.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a s",
            timing=OtoTiming(offset=1485.0, consonant=253.0, cutoff=-271.0, preutterance=204.0, overlap=140.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="sa",
            timing=OtoTiming(offset=1650.0, consonant=190.0, cutoff=-450.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a R",
            timing=OtoTiming(offset=2420.0, consonant=420.0, cutoff=-600.0, preutterance=250.0, overlap=83.0),
            source_timing=None,
            anchor=anchor(1308.5),
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in repaired[:4]] == pytest.approx([575.0, 980.0, 1485.0, 1650.0])
    assert all("cvvc_regular_pair_onset_repair" not in row.applied_rules for row in repaired[:4])


def test_japanese_cvvc_grouped_vc_block_repairs_first_vc_from_first_cv():
    rows = [
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="a b",
            timing=OtoTiming(offset=850.0, consonant=234.0, cutoff=-252.0, preutterance=185.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="i b",
            timing=OtoTiming(offset=1737.0, consonant=233.0, cutoff=-250.0, preutterance=184.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="bi",
            timing=OtoTiming(offset=1423.0, consonant=190.0, cutoff=-388.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="bu",
            timing=OtoTiming(offset=1950.0, consonant=190.0, cutoff=-400.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.offset == pytest.approx(1213.0)
    assert repaired[0].timing.cutoff == pytest.approx(-252.0)
    assert repaired[1].timing.offset == pytest.approx(1737.0)
    assert "cvvc_grouped_vc_first_offset_repair" in repaired[0].applied_rules
    assert any("cvvc_grouped_vc_first_offset_reference:bi" in item for item in repaired[0].warnings)


def test_japanese_cvvc_grouped_vc_block_keeps_near_first_cv():
    rows = [
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="a b",
            timing=OtoTiming(offset=1060.0, consonant=234.0, cutoff=-252.0, preutterance=185.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="i b",
            timing=OtoTiming(offset=1737.0, consonant=233.0, cutoff=-250.0, preutterance=184.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="bi",
            timing=OtoTiming(offset=1423.0, consonant=190.0, cutoff=-388.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba-bi-bu.wav",
            alias="bu",
            timing=OtoTiming(offset=1950.0, consonant=190.0, cutoff=-400.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.offset == pytest.approx(1060.0)
    assert "cvvc_grouped_vc_first_offset_repair" not in repaired[0].applied_rules


def test_japanese_cvvc_grouped_vc_block_allows_unknown_cv_phone_when_vc_block_matches():
    rows = [
        AdaptedOtoRow(
            wav="ga-gi-gu.wav",
            alias="a g",
            timing=OtoTiming(offset=780.0, consonant=234.0, cutoff=-252.0, preutterance=185.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga-gi-gu.wav",
            alias="i g",
            timing=OtoTiming(offset=1740.0, consonant=233.0, cutoff=-250.0, preutterance=184.0, overlap=140.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga-gi-gu.wav",
            alias="??",
            timing=OtoTiming(offset=1420.0, consonant=190.0, cutoff=-388.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga-gi-gu.wav",
            alias="???",
            timing=OtoTiming(offset=1950.0, consonant=190.0, cutoff=-400.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.offset == pytest.approx(1210.0)
    assert "cvvc_grouped_vc_first_offset_repair" in repaired[0].applied_rules


def test_japanese_cvvc_vc_cutoff_order_extends_invalid_tail():
    row = AdaptedOtoRow(
        wav="ka.wav",
        alias="a k",
        timing=OtoTiming(offset=1338.5, consonant=248.6, cutoff=-218.0, preutterance=199.6, overlap=140.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.cutoff == pytest.approx(-293.6)
    assert repaired[0].timing.consonant == pytest.approx(248.6)
    assert repaired[0].timing.offset == pytest.approx(1338.5)
    assert "cvvc_vc_cutoff_order_repair" in repaired[0].applied_rules


def test_japanese_cvvc_vc_cutoff_order_keeps_valid_tail():
    row = AdaptedOtoRow(
        wav="ka.wav",
        alias="a k",
        timing=OtoTiming(offset=1338.5, consonant=248.6, cutoff=-320.0, preutterance=199.6, overlap=140.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.cutoff == pytest.approx(-320.0)
    assert "cvvc_vc_cutoff_order_repair" not in repaired[0].applied_rules


def test_japanese_cvvc_initial_v_regular_pair_onset_sequence_repairs_headless_v_prefix():
    rows = [
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a",
            timing=OtoTiming(offset=360.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=420.0,
                score=0.25,
                role="v",
                next_vowel_abs_ms=920.0,
                expected_phone_index=0,
                warnings=("nucleus_span_too_short", "local_refine_low_margin"),
            ),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a k",
            timing=OtoTiming(offset=1290.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1210.0, score=0.5, role="vc", next_vowel_abs_ms=1680.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="ka",
            timing=OtoTiming(offset=1350.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1550.0, score=0.5, role="cv", next_vowel_abs_ms=1680.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a s",
            timing=OtoTiming(offset=1930.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2060.0, score=0.5, role="vc", next_vowel_abs_ms=2410.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="sa",
            timing=OtoTiming(offset=2060.0, consonant=190.0, cutoff=-530.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2260.0, score=0.5, role="cv", next_vowel_abs_ms=2410.0),
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1560.0, 1710.0, 2050.0, 2210.0, 2550.0])
    assert "cvvc_pure_vowel_onset_v_repair" in repaired[0].applied_rules
    assert all("cvvc_regular_pair_onset_repair" in row.applied_rules for row in repaired[1:])
    assert repaired[0].timing.preutterance == pytest.approx(0.0)
    assert repaired[0].timing.consonant == pytest.approx(125.0)
    assert any("cvvc_regular_pair_onset_reference:1680.0" in item for item in repaired[1].warnings)


def test_japanese_cvvc_initial_v_regular_pair_onset_sequence_requires_large_initial_v_shift():
    rows = [
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a",
            timing=OtoTiming(offset=1350.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=420.0, score=0.25, role="v", expected_phone_index=0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a k",
            timing=OtoTiming(offset=1290.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1210.0, score=0.5, role="vc", next_vowel_abs_ms=1680.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="ka",
            timing=OtoTiming(offset=1350.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1550.0, score=0.5, role="cv", next_vowel_abs_ms=1680.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="a s",
            timing=OtoTiming(offset=1930.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2060.0, score=0.5, role="vc", next_vowel_abs_ms=2410.0),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-ka-sa.wav",
            alias="sa",
            timing=OtoTiming(offset=2060.0, consonant=190.0, cutoff=-530.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2260.0, score=0.5, role="cv", next_vowel_abs_ms=2410.0),
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1350.0, 1290.0, 1350.0, 1930.0, 2060.0])
    assert all("cvvc_regular_pair_onset_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_initial_vowel_cv_head_onset_uses_vowel_start_when_hsmm_is_late():
    rows = [
        AdaptedOtoRow(
            wav="_a-n-a.wav",
            alias="- \u3042",
            timing=OtoTiming(offset=1490.0, consonant=160.0, cutoff=-190.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1610.0,
                score=0.45,
                role="cv_head",
                vowel_start_abs_ms=820.0,
                vowel_nucleus_abs_ms=1770.0,
                vowel_end_abs_ms=4060.0,
                expected_phone_index=0,
                warnings=("local_refine_low_margin",),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav="_a-n-a.wav",
            alias="a \u3093",
            timing=OtoTiming(offset=1680.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1760.0, score=0.45, role="vc", expected_phone_index=1),
            mode="bootstrap",
        ),
    ]

    repaired_head, _next_vc = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired_head.timing.offset == pytest.approx(740.0)
    assert repaired_head.timing.preutterance == pytest.approx(80.0)
    assert repaired_head.timing.overlap == pytest.approx(40.0)
    assert repaired_head.timing.consonant == pytest.approx(220.0)
    assert repaired_head.timing.cutoff == pytest.approx(-320.0)
    assert "cvvc_initial_vowel_cv_head_onset_repair" in repaired_head.applied_rules
    assert any(
        warning.startswith("cvvc_initial_vowel_cv_head_onset_repaired:1490.0->740.0")
        for warning in repaired_head.warnings
    )


def test_japanese_cvvc_initial_vowel_cv_head_onset_uses_next_vc_vowel_start_when_hsmm_is_early():
    rows = [
        AdaptedOtoRow(
            wav="_i-n-i.wav",
            alias="- \u3044",
            timing=OtoTiming(offset=140.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=260.0,
                score=0.23,
                role="cv_head",
                vowel_start_abs_ms=260.0,
                vowel_nucleus_abs_ms=260.0,
                vowel_end_abs_ms=260.0,
                expected_phone_index=0,
                warnings=("nucleus_span_too_short", "low_boundary_confidence:0.083"),
            ),
            mode="bootstrap",
            warnings=("nucleus_span_too_short", "low_boundary_confidence:0.083"),
        ),
        AdaptedOtoRow(
            wav="_i-n-i.wav",
            alias="i \u3093",
            timing=OtoTiming(offset=1061.5, consonant=120.0, cutoff=-146.0, preutterance=170.0, overlap=45.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=960.0,
                score=0.45,
                role="vc",
                vowel_start_abs_ms=810.0,
                expected_phone_index=1,
            ),
            mode="bootstrap",
        ),
    ]

    repaired_head, _next_vc = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired_head.timing.offset == pytest.approx(730.0)
    assert repaired_head.timing.preutterance == pytest.approx(80.0)
    assert repaired_head.timing.overlap == pytest.approx(40.0)
    assert repaired_head.timing.consonant == pytest.approx(220.0)
    assert repaired_head.timing.cutoff == pytest.approx(-320.0)
    assert "cvvc_initial_vowel_cv_head_onset_repair" in repaired_head.applied_rules
    assert any(
        warning.startswith("cvvc_initial_vowel_cv_head_onset_pre_abs:260.0->810.0")
        for warning in repaired_head.warnings
    )


def test_japanese_cvvc_initial_vowel_cv_head_onset_requires_first_phone_index():
    rows = [
        AdaptedOtoRow(
            wav="_a-a-i.wav",
            alias="- \u3042",
            timing=OtoTiming(offset=1490.0, consonant=160.0, cutoff=-190.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1610.0,
                score=0.45,
                role="cv_head",
                vowel_start_abs_ms=820.0,
                vowel_nucleus_abs_ms=820.0,
                vowel_end_abs_ms=820.0,
                expected_phone_index=1,
            ),
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_a-a-i.wav",
            alias="a \u3042",
            timing=OtoTiming(offset=1680.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1760.0, score=0.45, role="vc", expected_phone_index=1),
            mode="bootstrap",
        ),
    ]

    repaired_head, _next_vc = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired_head.timing.offset == pytest.approx(1490.0)
    assert repaired_head.timing.preutterance == pytest.approx(120.0)
    assert "cvvc_initial_vowel_cv_head_onset_repair" not in repaired_head.applied_rules


def test_japanese_cvvc_terminal_release_r_copies_matching_terminal_timing():
    terminal_r = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a R",
        timing=OtoTiming(offset=2930.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3010.0, score=0.45, role="vc"),
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3630.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_r, _terminal = repair_cvvc_row_sequence(
        [terminal_r, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4687.0,
    )

    assert repaired_r.timing.offset == pytest.approx(3630.0)
    assert repaired_r.timing.preutterance == pytest.approx(200.0)
    assert repaired_r.timing.overlap == pytest.approx(100.0)
    assert repaired_r.timing.consonant == pytest.approx(320.0)
    assert repaired_r.timing.cutoff == pytest.approx(-480.0)
    assert "cvvc_terminal_release_r_terminal_copy_repair" in repaired_r.applied_rules


def test_japanese_cvvc_terminal_release_r_terminal_copy_requires_large_forward_shift():
    terminal_r = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a R",
        timing=OtoTiming(offset=3500.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3580.0, score=0.45, role="vc"),
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3630.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_r, _terminal = repair_cvvc_row_sequence(
        [terminal_r, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4687.0,
    )

    assert repaired_r.timing.offset == pytest.approx(3500.0)
    assert "cvvc_terminal_release_r_terminal_copy_repair" not in repaired_r.applied_rules


def test_japanese_cvvc_vc_next_vowel_local_repair_moves_late_confused_vc_offset():
    row = AdaptedOtoRow(
        wav="ga-gi-gu.wav",
        alias="i g",
        timing=OtoTiming(offset=1450.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1650.0,
            score=0.45,
            role="vc",
            next_vowel_abs_ms=2142.0,
            warnings=("local_refine_low_margin",),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired.timing.offset == pytest.approx(1746.0)
    assert repaired.timing.preutterance == pytest.approx(170.0)
    assert repaired.timing.cutoff == pytest.approx(-236.0)
    assert "cvvc_vc_next_vowel_local_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_vc_next_vowel_local_repaired:1450.0->1746.0")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_vc_next_vowel_local_repair_requires_local_refine_warning():
    row = AdaptedOtoRow(
        wav="ga-gi-gu.wav",
        alias="i g",
        timing=OtoTiming(offset=1450.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1650.0,
            score=0.45,
            role="vc",
            next_vowel_abs_ms=2142.0,
        ),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired.timing.offset == pytest.approx(1450.0)
    assert "cvvc_vc_next_vowel_local_repair" not in repaired.applied_rules


def test_japanese_cvvc_cv_rejected_next_vowel_repair_moves_early_cv_offset():
    row = AdaptedOtoRow(
        wav="wa-wi-we.wav",
        alias="\u3046\u3043",
        timing=OtoTiming(offset=1016.5, consonant=140.0, cutoff=-220.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1216.5,
            score=0.45,
            role="cv",
            vowel_nucleus_abs_ms=1216.5,
            next_vowel_abs_ms=2043.0,
            warnings=("local_refine_rejected_slot_boundary",),
        ),
        mode="bootstrap",
        warnings=("local_refine_rejected_slot_boundary",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired.timing.offset == pytest.approx(1415.775)
    assert repaired.timing.preutterance == pytest.approx(70.0)
    assert repaired.timing.cutoff == pytest.approx(-220.0)
    assert "cvvc_cv_rejected_next_vowel_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_cv_rejected_next_vowel_repaired:1016.5->1415.8")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_cv_rejected_next_vowel_repair_requires_rejected_boundary_warning():
    row = AdaptedOtoRow(
        wav="wa-wi-we.wav",
        alias="\u3046\u3043",
        timing=OtoTiming(offset=1016.5, consonant=140.0, cutoff=-220.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1216.5,
            score=0.45,
            role="cv",
            vowel_nucleus_abs_ms=1216.5,
            next_vowel_abs_ms=2043.0,
            warnings=("local_refine_low_margin",),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired.timing.offset == pytest.approx(1016.5)
    assert "cvvc_cv_rejected_next_vowel_repair" not in repaired.applied_rules


def test_japanese_cvvc_terminal_standalone_vowel_uses_terminal_reference():
    terminal = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3600.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )
    early_vowel = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="\u3042",
        timing=OtoTiming(offset=320.0, consonant=2500.0, cutoff=-2600.0, preutterance=1200.0, overlap=600.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _terminal, repaired = repair_cvvc_row_sequence(
        [terminal, early_vowel],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(3700.0)
    assert repaired.timing.consonant == pytest.approx(125.0)
    assert repaired.timing.cutoff == pytest.approx(-260.0)
    assert repaired.timing.preutterance == pytest.approx(30.0)
    assert repaired.timing.overlap == pytest.approx(20.0)
    assert "cvvc_terminal_standalone_v_repair" in repaired.applied_rules


def test_japanese_cvvc_alias_pitch_suffix_is_ignored_for_phone_matching():
    assert _alias_phone_sequence("i bA3") == ["i", "b"]
    assert _alias_phone_sequence("n nyA3") == ["n", "n", "y"]
    assert _alias_phone_sequence("u vA3") == ["u", "v"]
    assert _alias_phone_sequence("\u3042A3") == ["a"]
    assert _alias_type_for_row("i bA3", "auto") == "vc"
    assert _alias_type_for_row("u vA3", "auto") == "vc"
    assert _alias_type_for_row("\u3042A3", "auto") == "v"


def test_japanese_cvvc_initial_spaced_cv_offset_uses_matching_kana_cv_offset():
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3064\u3041",
        timing=OtoTiming(offset=16.5, consonant=190.0, cutoff=-443.5, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    early_spaced_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="ts a",
        timing=OtoTiming(offset=0.5, consonant=160.0, cutoff=-409.5, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("vcv_previous_vowel_end_clamped:460.0->0.5",),
    )

    _kana, repaired = repair_cvvc_row_sequence(
        [kana_cv, early_spaced_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(16.5)
    assert repaired.timing.preutterance == pytest.approx(120.0)
    assert repaired.timing.consonant == pytest.approx(160.0)
    assert repaired.timing.cutoff == pytest.approx(-393.5)
    assert "cvvc_initial_spaced_cv_offset_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_spaced_cv_offset_repaired:0.5->16.5")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_pure_vowel_transition_repairs_short_nucleus_following_v_rows():
    previous_v = AdaptedOtoRow(
        wav="case.wav",
        alias="a",
        timing=OtoTiming(offset=0.0, consonant=498.4, cutoff=-530.0, preutterance=379.6, overlap=192.4),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_vv = AdaptedOtoRow(
        wav="case.wav",
        alias="a i",
        timing=OtoTiming(offset=560.0, consonant=160.0, cutoff=-250.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=720.0, previous_vowel_end_abs_ms=480.0),
        mode="bootstrap",
        warnings=("low_boundary_confidence:0.266", "local_refine_low_margin"),
    )
    late_kana_v = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3044",
        timing=OtoTiming(offset=774.0, consonant=80.0, cutoff=-136.0, preutterance=56.0, overlap=35.84),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("nucleus_span_too_short", "local_refine_low_margin"),
    )
    late_romaji_v = AdaptedOtoRow(
        wav="case.wav",
        alias="i",
        timing=OtoTiming(offset=816.0, consonant=80.0, cutoff=-114.0, preutterance=56.0, overlap=35.84),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("nucleus_span_too_short",),
    )

    _previous, repaired_vv, repaired_kana_v, repaired_romaji_v = repair_cvvc_row_sequence(
        [previous_v, late_vv, late_kana_v, late_romaji_v],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_vv.timing.offset == pytest.approx(395.0)
    assert repaired_vv.timing.preutterance == pytest.approx(110.0)
    assert repaired_vv.timing.overlap == pytest.approx(100.0)
    assert repaired_vv.timing.consonant == pytest.approx(210.0)
    assert repaired_vv.timing.cutoff == pytest.approx(-535.0)
    assert repaired_kana_v.timing.offset == pytest.approx(505.0)
    assert repaired_kana_v.timing.preutterance == pytest.approx(90.0)
    assert repaired_kana_v.timing.overlap == pytest.approx(45.0)
    assert repaired_kana_v.timing.consonant == pytest.approx(100.0)
    assert repaired_kana_v.timing.cutoff == pytest.approx(-425.0)
    assert repaired_romaji_v.timing == repaired_kana_v.timing
    assert "cvvc_pure_vowel_transition_repair" in repaired_vv.applied_rules
    assert "cvvc_pure_vowel_v_repair" in repaired_kana_v.applied_rules


def test_japanese_cvvc_ry_yoon_sequence_keeps_sonorant_context():
    previous_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="ry e",
        timing=OtoTiming(offset=1240.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="e ry",
        timing=OtoTiming(offset=1475.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("cvvc_vc_left_context_pre_capped:235.0->170.0",),
    )
    kana_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u308a\u3087",
        timing=OtoTiming(offset=1550.0, consonant=190.0, cutoff=-260.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    spaced_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="ry o",
        timing=OtoTiming(offset=1540.0, consonant=160.0, cutoff=-220.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    post_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="o ry",
        timing=OtoTiming(offset=1770.0, consonant=205.0, cutoff=-231.0, preutterance=165.0, overlap=130.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _previous, repaired_vc, repaired_kana, repaired_spaced, repaired_post = repair_cvvc_row_sequence(
        [previous_vcv, late_vc, kana_cv, spaced_cv, post_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_vc.timing.offset == pytest.approx(1240.0)
    assert repaired_vc.timing.preutterance == pytest.approx(110.0)
    assert repaired_vc.timing.overlap == pytest.approx(100.0)
    assert "cvvc_sonorant_yoon_vc_repair" in repaired_vc.applied_rules
    assert repaired_kana.timing.offset == pytest.approx(1350.0)
    assert repaired_kana.timing.preutterance == pytest.approx(50.0)
    assert repaired_kana.timing.overlap == pytest.approx(40.0)
    assert repaired_spaced.timing == repaired_kana.timing
    assert repaired_post.timing.offset == pytest.approx(1675.0)
    assert repaired_post.timing.cutoff == pytest.approx(-326.0)
    assert "cvvc_post_yoon_vc_repair" in repaired_post.applied_rules


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


def test_cvvc_cv_bootstrap_keeps_leading_consonant_in_offset():
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        OtoAnchor(anchor_abs_ms=210.0, score=0.84, vowel_end_abs_ms=340.0),
        file_duration_ms=520.0,
        config=OtoAdapterConfig(mode="bootstrap", language="japanese", format_type="CVVC", alias_type="cv"),
    )

    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(210.0)
    assert adapted.timing.offset == pytest.approx(60.0)
    assert adapted.timing.preutterance == pytest.approx(150.0)
    assert "cvvc_cv_left_context_pre:120.0->150.0" in adapted.warnings


def test_cvvc_hsmm_anchor_lead_pulls_cv_offset_left():
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        OtoAnchor(
            anchor_abs_ms=300.0,
            score=0.9,
            role="cv",
            source_event_label="cv_boundary",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=800.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.preutterance == pytest.approx(150.0)
    assert adapted.timing.offset == pytest.approx(100.0)
    assert "hsmm_anchor_lead:300.0->250.0" in adapted.warnings


def test_cvvc_hsmm_anchor_lead_skips_terminal_silence_alias():
    adapted = bootstrap_row(
        "a.wav",
        "a -",
        OtoAnchor(
            anchor_abs_ms=300.0,
            score=0.9,
            role="vcv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=800.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(682.0)
    assert adapted.timing.consonant == pytest.approx(110.0)
    assert adapted.timing.preutterance == pytest.approx(110.0)
    assert adapted.timing.cutoff == pytest.approx(-118.0)
    assert "terminal_silence_end_anchored" in adapted.warnings
    assert not any(warning.startswith("hsmm_anchor_lead:") for warning in adapted.warnings)


def test_japanese_cvvc_terminal_silence_uses_vowel_end_anchor():
    adapted = bootstrap_row(
        "a.wav",
        "a -",
        OtoAnchor(
            anchor_abs_ms=3000.0,
            score=0.9,
            role="vcv",
            vowel_end_abs_ms=4400.0,
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=5900.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(3970.0)
    assert adapted.timing.preutterance == pytest.approx(200.0)
    assert adapted.timing.overlap == pytest.approx(100.0)
    assert adapted.timing.consonant == pytest.approx(320.0)
    assert adapted.timing.cutoff == pytest.approx(-480.0)
    assert "terminal_silence_vowel_end_anchored:4400.0" in adapted.warnings
    assert "japanese_cvvc_terminal_silence_vowel_end_anchor" in adapted.applied_rules


def test_special_breath_alias_preserves_zero_template_timing():
    row = OtoTemplateRow("breath.wav", "br2", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))

    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=530.0, score=0.9, role="vcv"),
        file_duration_ms=900.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.as_tuple() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0))
    assert adapted.mode == "template-special"
    assert "special_alias_zero_template_preserved" in adapted.warnings


def test_special_breath_alias_preserves_nonzero_source_timing():
    source_timing = OtoTiming(2191.38, 185.713, -186.713, 167.309, 15.058)
    row = OtoTemplateRow("br1.wav", "br1", source_timing)

    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=1090.0, score=0.9, role="cv"),
        file_duration_ms=3718.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing == source_timing
    assert adapted.source_timing == source_timing
    assert adapted.mode == "template-special"
    assert "special_alias_source_timing_preserved" in adapted.warnings
    assert "special_alias_source_timing_preserve" in adapted.applied_rules


def test_special_separator_alias_preserves_nonzero_source_timing():
    source_timing = OtoTiming(1129.25, 136.05, -252.94, 22.68, 0.0)
    row = OtoTemplateRow("_n_sep.wav", "\u30fb \u3042", source_timing)

    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=1700.0, score=0.9, role="vcv"),
        file_duration_ms=4200.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing == source_timing
    assert adapted.source_timing == source_timing
    assert adapted.mode == "template-special"
    assert "special_alias_source_timing_preserved" in adapted.warnings
    assert "special_alias_source_timing_preserve" in adapted.applied_rules


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


def test_no_mfa_hsmm_runtime_can_apply_lightgbm_layer(tmp_path, monkeypatch):
    from core.generation.common import no_mfa_oto_builder as builder
    from core.mfa_free_oto.workflow import NoMfaWorkflowReport

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_oto = tmp_path / "out.ini"
    calls = {}

    def _fake_generate_no_mfa_oto_with_model_context(**kwargs):
        Path(kwargs["out_path"]).write_text("a.wav=ka,10,100,-300,80,40\n", encoding="utf-8")
        calls["generate"] = dict(kwargs)
        return NoMfaWorkflowReport(
            processed=1,
            total=1,
            errors=(),
            warnings=(),
            confidence=0.75,
            fallback_hint="",
            mode="mfa_free_ssl_slot",
            guard_failed=False,
            metrics={"anchor_ratio": 1.0},
        )

    def _fake_apply_no_mfa_ml_correction(**kwargs):
        calls["lightgbm"] = dict(kwargs)
        return 2

    monkeypatch.setenv("UTOA_MFA_FREE_HSMM_LIGHTGBM_ENABLE", "1")
    monkeypatch.setenv("UTOA_NO_MFA_FORMAT_TYPE", "cvvc")
    monkeypatch.setattr(
        "core.mfa_free_oto.workflow.generate_no_mfa_oto_with_model_context",
        _fake_generate_no_mfa_oto_with_model_context,
    )
    monkeypatch.setattr(builder, "_apply_no_mfa_ml_correction", _fake_apply_no_mfa_ml_correction)

    processed, total, errors = builder.generate_no_mfa_auto_oto(
        wav_dir=str(wav_dir),
        out_path=str(out_oto),
        source_oto_path="",
        language="japanese",
        generation_mode="mfa_free_ssl_slot",
    )

    assert (processed, total, errors) == (1, 1, [])
    assert calls["generate"]["format_type"] == "cvvc"
    assert calls["lightgbm"]["oto_path"] == str(out_oto)
    assert calls["lightgbm"]["format_type"] == "cvvc"
    meta = builder.get_last_no_mfa_runtime_meta()
    assert meta["metrics"]["lightgbm_postprocess_requested"] == 1.0
    assert meta["metrics"]["lightgbm_postprocess_changed"] == 2.0


def test_oto_ml_features_imports_without_textgrid_dependency():
    from core.oto_ml import oto_ml_features

    phones, words = oto_ml_features._load_textgrid_tiers("")

    assert phones == []
    assert words == []


def test_japanese_oto_validator_imports_without_textgrid_dependency():
    from core.ja_oto_generator import validate_oto_params

    offset, consonant, cutoff, pre, overlap = validate_oto_params(10.0, 100.0, -180.0, 80.0, 40.0, "cv")

    assert offset == pytest.approx(10.0)
    assert overlap <= pre <= consonant
    assert abs(cutoff) > consonant


def test_no_mfa_lightgbm_safety_filter_preserves_terminal_and_caps_overlap(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter
    from core.generation.common.oto_alias_family import alias_family

    assert alias_family("br1") == "policy_breath"

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "a.wav=V -,4000,320,-90,200,100",
                "br1.wav=br1,2191.38,185.713,-186.713,167.309,15.058",
                "a.wav=a k,1000,120,-160,80,45",
                "a.wav=a \u3042,1400,160,-300,120,85",
                "a.wav=\u304b,1800,180,-300,120,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "a.wav=V -,3700,340,-180,210,170",
                "br1.wav=br1,260,220,-234,80.09,17.56",
                "a.wav=a k,980,220,-240,170,140",
                "a.wav=a \u3042,1420,250,-260,180,130",
                "a.wav=\u304b,1810,190,-290,130,90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
        max_overlap_increase_ms=20.0,
    )

    lines = post.read_text(encoding="utf-8").splitlines()
    assert report["changed_rows"] == 5
    assert report["restored_terminal_rows"] == 1
    assert report["restored_breath_rows"] == 1
    assert report["capped_cv_pre_overlap_rows"] == 1
    assert report["capped_overlap_rows"] == 2
    assert lines[0] == "a.wav=V -,4000,320,-90,200,100"
    assert lines[1] == "br1.wav=br1,2191.38,185.713,-186.713,167.309,15.058"
    assert lines[2] == "a.wav=a k,980,220,-240,170,65"
    assert lines[3] == "a.wav=a \u3042,1420,250,-260,180,105"
    assert lines[4] == "a.wav=\u304b,1810,190,-290,70,25"


def test_no_mfa_lightgbm_safety_filter_defaults_to_tight_overlap_cap(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("a.wav=a k,1000,120,-160,80,45\n", encoding="utf-8")
    post.write_text("a.wav=a k,980,220,-240,170,90\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["max_overlap_increase_ms"] == 5.0
    assert report["capped_overlap_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "a.wav=a k,980,220,-240,170,50"


def test_no_mfa_lightgbm_safety_filter_caps_cv_cutoff_before_next_transition(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    text = "\n".join(
        [
            "case.wav=\u306c,1340,190,-2700,70,25",
            "case.wav=u n,1761.74,198.29,-215.77,149.26,50",
            "case.wav=\u304dA3,950,190,-900,70,25",
            "case.wav=i kA3,1180,198,-216,149,50",
        ]
    )
    pre.write_text(text + "\n", encoding="utf-8")
    post.write_text(text + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    lines = post.read_text(encoding="utf-8").splitlines()
    assert report["capped_cv_next_transition_cutoff_rows"] == 2
    assert lines[0] == "case.wav=\u306c,1340,190,-501.74,70,25"
    assert lines[2] == "case.wav=\u304dA3,950,190,-350,70,25"


def test_no_mfa_lightgbm_safety_filter_repairs_vc_cutoff_order_after_regular_pair_restore(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-k-a-s.wav=a k,1338.5,210,-218,170,135",
                "_a-k-a-s.wav=ka,1678.5,190,-198,150,115",
                "_a-k-a-s.wav=a s,1838.5,210,-218,170,135",
                "_a-k-a-s.wav=sa,2178.5,190,-198,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-k-a-s.wav=a k,1338.5,248.59,-218,199.56,140",
                "_a-k-a-s.wav=ka,1678.5,190,-198,150,115",
                "_a-k-a-s.wav=a s,1838.5,248.6,-218,199.57,140",
                "_a-k-a-s.wav=sa,2178.5,190,-198,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    lines = post.read_text(encoding="utf-8").splitlines()
    assert report["restored_regular_pair_onset_sequence_rows"] == 2
    assert report["repaired_vc_cutoff_order_rows"] == 2
    assert lines[0] == "_a-k-a-s.wav=a k,1338.5,248.59,-293.59,199.56,140"
    assert lines[2] == "_a-k-a-s.wav=a s,1838.5,248.6,-293.6,199.57,140"


def test_no_mfa_lightgbm_safety_filter_restores_large_cv_fixed_cutoff_expansion(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("wa.wav=\u3092,2850,190,-400,150,115\n", encoding="utf-8")
    post.write_text("wa.wav=\u3092,2590,410,-940,77.66,41.24\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_cv_fixed_cutoff_rows"] == 1
    assert report["capped_cv_pre_overlap_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "wa.wav=\u3092,2590,190,-400,70,25"


def test_no_mfa_lightgbm_safety_filter_preserves_terminal_standalone_vowel(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-n-a.wav=a -,3600,320,-480,200,100",
                "_a-n-a.wav=\u3042,3700,125,-260,30,20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-n-a.wav=a -,3600,320,-480,200,100",
                "_a-n-a.wav=\u3042,640,1450,-2580,1160,540",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_terminal_standalone_vowel_rows"] == 1
    assert report["retimed_terminal_standalone_vowel_companion_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_a-n-a.wav=a -,3780,320,-480,200,100",
        "_a-n-a.wav=\u3042,3700,125,-260,30,20",
    ]


def test_no_mfa_lightgbm_safety_filter_retims_terminal_v_dash_from_standalone_vowel(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-n-a.wav=a -,3600,320,-480,200,100",
                "_a-n-a.wav=\u3042,3700,125,-260,30,20",
                "_i-n-i.wav=i -,3890,320,-480,200,100",
                "_i-n-i.wav=\u3044,3700,125,-260,30,20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(pre.read_text(encoding="utf-8"), encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["retimed_terminal_standalone_vowel_companion_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_a-n-a.wav=a -,3780,320,-480,200,100",
        "_a-n-a.wav=\u3042,3700,125,-260,30,20",
        "_i-n-i.wav=i -,3890,320,-480,200,100",
        "_i-n-i.wav=\u3044,3700,125,-260,30,20",
    ]


def test_no_mfa_lightgbm_safety_filter_regularizes_romaji_following_cv_block_offsets(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    lines = [
        "ma-mi-mu-me-mo-ma-N-ma.wav=a m,1180,220,-240,170,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=i m,1700,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=u m,2265,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=e m,3080,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=o m,3809,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=n m,4450,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u307f,1509,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3080,1870,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3081,2730,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3082,3280,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u307e,3980,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=a m,1180,220,-240,170,50",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=i m,1580,230,-250,180,50",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u307f,1509,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3080,2300,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3081,3100,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3082,3900,190,-400,70,25",
        "pitch.wav=a m,1180,220,-240,170,50",
        "pitch.wav=i m,1580,230,-250,180,50",
        "pitch.wav=\u307fA3,1509,190,-400,70,25",
        "pitch.wav=\u3080A3,2300,190,-400,70,25",
        "pitch.wav=\u3081A3,3100,190,-400,70,25",
        "pitch.wav=\u3082A3,3900,190,-400,70,25",
    ]
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["regularized_following_cv_block_offset_rows"] == 4
    assert post.read_text(encoding="utf-8").splitlines() == [
        "ma-mi-mu-me-mo-ma-N-ma.wav=a m,1180,220,-240,170,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=i m,1700,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=u m,2265,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=e m,3080,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=o m,3809,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=n m,4450,230,-250,180,50",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u307f,1509,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3080,2009,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3081,2509,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u3082,3009,190,-400,70,25",
        "ma-mi-mu-me-mo-ma-N-ma.wav=\u307e,3509,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=a m,1180,220,-240,170,50",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=i m,1580,230,-250,180,50",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u307f,1509,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3080,2300,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3081,3100,190,-400,70,25",
        "_ma-mi-mu-me-mo-ma-N-ma.wav=\u3082,3900,190,-400,70,25",
        "pitch.wav=a m,1180,220,-240,170,50",
        "pitch.wav=i m,1580,230,-250,180,50",
        "pitch.wav=\u307fA3,1509,190,-400,70,25",
        "pitch.wav=\u3080A3,2300,190,-400,70,25",
        "pitch.wav=\u3081A3,3100,190,-400,70,25",
        "pitch.wav=\u3082A3,3900,190,-400,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_retimes_romaji_vc_before_following_cv(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    lines = [
        "ka-ki-ku-ke-ko-ka-N-ka.wav=a k,756.06,220,-240,170,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=i k,1386.84,230,-250,180,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=u k,2250,230,-250,180,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u304d,1353,190,-400,70,25",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u304f,1870,190,-400,70,25",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u3051,2370,190,-400,70,25",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=a k,756.06,220,-240,170,50",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=i k,1386.84,230,-250,180,50",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=\u304d,1353,190,-400,70,25",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=\u304f,1870,190,-400,70,25",
        "pitch.wav=a k,756.06,220,-240,170,50",
        "pitch.wav=i k,1386.84,230,-250,180,50",
        "pitch.wav=\u304dA3,1353,190,-400,70,25",
        "pitch.wav=\u304fA3,1870,190,-400,70,25",
    ]
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["retimed_romaji_vc_following_cv_rows"] == 2
    assert post.read_text(encoding="utf-8").splitlines() == [
        "ka-ki-ku-ke-ko-ka-N-ka.wav=a k,1153,220,-240,170,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=i k,1670,230,-250,180,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=u k,2250,230,-250,180,50",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u304d,1353,190,-400,70,25",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u304f,1870,190,-400,70,25",
        "ka-ki-ku-ke-ko-ka-N-ka.wav=\u3051,2370,190,-400,70,25",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=a k,756.06,220,-240,170,50",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=i k,1386.84,230,-250,180,50",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=\u304d,1353,190,-400,70,25",
        "_ka-ki-ku-ke-ko-ka-N-ka.wav=\u304f,1870,190,-400,70,25",
        "pitch.wav=a k,756.06,220,-240,170,50",
        "pitch.wav=i k,1386.84,230,-250,180,50",
        "pitch.wav=\u304dA3,1353,190,-400,70,25",
        "pitch.wav=\u304fA3,1870,190,-400,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_restores_middle_dot_transitions(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre_lines = [
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=a \u30fb,1342.3,160,-240,120,85",
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=i \u30fb,1727.7,160,-240,120,85",
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=o \u30fbA3,2920.5,160,-240,120,85",
        "nondot.wav=a k,1000,160,-240,120,85",
    ]
    post_lines = [
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=a \u30fb,1253.9,260,-340,145,90",
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=i \u30fb,1523.7,260,-340,145,90",
        "_\u3093\u30fb\u3042\u30fb\u3044\u30fb\u3046\u30fb\u3048\u30fb\u304a\u30fb\u3093.wav=o \u30fbA3,2846.1,260,-340,145,90",
        "nondot.wav=a k,900,260,-340,145,90",
    ]
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(pre_lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(post_lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_middle_dot_transition_rows"] == 2
    assert post.read_text(encoding="utf-8").splitlines() == [
        pre_lines[0],
        pre_lines[1],
        post_lines[2],
        post_lines[3],
    ]


def test_no_mfa_lightgbm_safety_filter_shifts_pitch_headless_initial_slot_offsets(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    lines = [
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i bA3,901.4,256.2,-273.7,207.2,140",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=\u3070A3,1020,190,-400,70,25",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i pA3,1688,253.5,-271,204.5,140",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i rA3,2731.93,256.4,-273.9,207.4,140",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a yA3,1302,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3084A3,1642,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a wA3,1802,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u308fA3,2142,190,-420,70,25",
    ]
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["shifted_pitch_headless_initial_slot_rows"] == 4
    assert report["shifted_pitch_katakana_n_grid_rows"] == 0
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i bA3,1401.4,256.2,-273.7,207.2,140",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=\u3070A3,1520,190,-400,70,25",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i pA3,2188,253.5,-271,204.5,140",
        "_\u3044\u3070\u3044\u3071\u3044\u3089.wav=i rA3,3231.93,256.4,-273.9,207.4,140",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a yA3,1302,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3084A3,1642,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a wA3,1802,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u308fA3,2142,190,-420,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_shifts_pitch_katakana_n_grid_offsets(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    lines = [
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=u \u30f3A3,2262,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u3042A3,2762,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=a \u30f3A3,3262,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u30f3A3,3762,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u3046A3,4262,420,-600,250,83",
        "_\u30f3\u3044\u30f3\u3048\u30f3\u304a\u30f3.wav=- \u30f3A3,970,160,-560,90,45",
        "_\u30f3\u3044\u30f3\u3048\u30f3\u304a\u30f3.wav=N \u3044A3,1340,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N vA3,1336.5,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N hA3,2336.5,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N hyA3,3336.5,420,-600,250,83",
    ]
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["shifted_pitch_headless_initial_slot_rows"] == 0
    assert report["shifted_pitch_katakana_n_grid_rows"] == 5
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=u \u30f3A3,1882,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u3042A3,2382,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=a \u30f3A3,2882,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u30f3A3,3382,420,-600,250,83",
        "_\u304a\u3046\u30f3\u3042\u30f3\u30f3\u3046.wav=N \u3046A3,3882,420,-600,250,83",
        "_\u30f3\u3044\u30f3\u3048\u30f3\u304a\u30f3.wav=- \u30f3A3,970,160,-560,90,45",
        "_\u30f3\u3044\u30f3\u3048\u30f3\u304a\u30f3.wav=N \u3044A3,1340,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N vA3,1336.5,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N hA3,2336.5,420,-600,250,83",
        "_\u30f3\u30f4\u30f3\u306f\u30f3\u3072.wav=N hyA3,3336.5,420,-600,250,83",
    ]


def test_no_mfa_lightgbm_safety_filter_shifts_pitch_missing_cv_slot_offsets(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    compressed_lines = [
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a yA3,1307.5,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3084A3,1647.5,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a wA3,1807.5,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u308fA3,2147.5,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a tsA3,2485.3,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3064\u3041A3,2510,190,-430,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a fA3,2934,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3075\u3041A3,2980,190,-430,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a vA3,3416.7,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u30f4\u3041A3,3690,190,-430,70,25",
    ]
    wide_lines = [
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u yA3,1307.5,210,-218,170,135",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3086A3,1647.5,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u wA3,1807.5,210,-218,170,135",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3046\u3045A3,2157.5,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u tsA3,3375,230,-255,188,95",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3064A3,3600,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u fA3,3820,230,-255,188,95",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3075A3,4050,190,-430,70,25",
    ]
    lines = compressed_lines + wide_lines
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["shifted_pitch_missing_cv_slot_rows"] == 6
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a yA3,1307.5,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3084A3,1647.5,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a wA3,1807.5,210,-218,170,135",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u308fA3,2147.5,190,-420,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a tsA3,2985.3,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3064\u3041A3,3010,190,-430,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a fA3,3434,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u3075\u3041A3,3480,190,-430,70,25",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=a vA3,3916.7,230,-255,188,95",
        "_\u3042\u3084\u308f\u3061\u3083\u3064\u3041\u3075\u3041\u30f4\u3041.wav=\u30f4\u3041A3,4190,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u yA3,1307.5,210,-218,170,135",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3086A3,1647.5,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u wA3,1807.5,210,-218,170,135",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3046\u3045A3,2157.5,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u tsA3,3375,230,-255,188,95",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3064A3,3600,190,-430,70,25",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=u fA3,3820,230,-255,188,95",
        "_\u3046\u3086\u3046\u3045\u3061\u3085\u3064\u3075\u30f4.wav=\u3075A3,4050,190,-430,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_shifts_collapsed_underscore_kana_slot1_offsets(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    compressed_lines = [
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=- r,610,160,-168,70,25",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=a r,655,190,-220,150,50",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=\u308a,700,190,-360,70,25",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=i r,1351,190,-220,150,50",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=\u308b,1510,190,-360,70,25",
    ]
    late_head_lines = [
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=- s,1080,160,-168,70,25",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=a s,1059.71,190,-220,150,50",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=\u3059\u3043,1080,190,-360,70,25",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=i s,1514.42,190,-220,150,50",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=\u3059,1680,190,-360,70,25",
    ]
    lines = compressed_lines + late_head_lines
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["shifted_underscore_kana_slot1_rows"] == 2
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=- r,610,160,-168,70,25",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=a r,1155,190,-220,150,50",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=\u308a,1200,190,-360,70,25",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=i r,1351,190,-220,150,50",
        "_\u3089\u308a\u308b\u308c\u308d\u3089\u3093\u3089.wav=\u308b,1510,190,-360,70,25",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=- s,1080,160,-168,70,25",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=a s,1059.71,190,-220,150,50",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=\u3059\u3043,1080,190,-360,70,25",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=i s,1514.42,190,-220,150,50",
        "_\u3055\u3059\u3043\u3059\u305b\u305d\u3055\u3093\u3055.wav=\u3059,1680,190,-360,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_shifts_delayed_underscore_kana_slots(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    delayed_lines = [
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=- f,580,160,-168,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=a f,1105.24,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3043,1170,190,-229.8,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=i f,1319.8,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075,1500,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=u f,2140.95,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3047,2320,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=e f,2709.3,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3049,2870,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=o f,3078.31,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3041,3200,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=n f,3526.6,190,-220,150,50",
    ]
    no_head_lines = [
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=a ts,619.6,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064\u3043,670,190,-360,70,25",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=i ts,1301.27,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064,1360,190,-360,70,25",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=u ts,1746.69,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064\u3047,1930,190,-360,70,25",
    ]
    lines = delayed_lines + no_head_lines
    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["shifted_underscore_kana_delayed_slot_rows"] == 6
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=- f,580,160,-168,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=a f,1105.24,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3043,1170,190,-229.8,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=i f,1319.8,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075,1500,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=u f,1640.95,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3047,1820,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=e f,2209.3,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3049,2370,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=o f,2578.31,190,-220,150,50",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3041,2700,190,-360,70,25",
        "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=n f,3526.6,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=a ts,619.6,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064\u3043,670,190,-360,70,25",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=i ts,1301.27,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064,1360,190,-360,70,25",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=u ts,1746.69,190,-220,150,50",
        "_\u3064\u3041\u3064\u3043\u3064\u3064\u3047\u3064\u3049\u3064\u3041\u3093\u3064\u3041.wav=\u3064\u3047,1930,190,-360,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_restores_initial_standalone_vowel_profile(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_i-wi-chi.wav=\u3044A3,1557.5,125,-360,0,0",
                "_i-wi-chi.wav=i wA3,1707.5,210,-218,170,135",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_i-wi-chi.wav=\u3044A3,1297.5,345,-534.28,18.06,3.2",
                "_i-wi-chi.wav=i wA3,1707.5,210,-218,170,135",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_initial_standalone_vowel_rows"] == 1
    assert report["changed_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines()[0] == "_i-wi-chi.wav=\u3044A3,1557.5,125,-360,0,0"


def test_no_mfa_lightgbm_safety_filter_caps_vowel_cv_head_pre_overlap(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    lines = [
        "_a-a-i.wav=- \u3042A3,1040,160,-430,120,85",
        "_n-n.wav=- n,610,160,-168,120,85",
        "_ka-ki.wav=- k,600,160,-168,120,85",
    ]
    pre.write_text("\n".join(lines) + "\n", encoding="utf-8")
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["capped_cv_head_vowel_pre_overlap_rows"] == 2
    assert report["changed_rows"] == 2
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_a-a-i.wav=- \u3042A3,1040,160,-430,90,45",
        "_n-n.wav=- n,610,160,-168,90,45",
        "_ka-ki.wav=- k,600,160,-168,120,85",
    ]


def test_no_mfa_lightgbm_safety_filter_restores_underscore_vc_positive_offset_shift(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_ma-mi-mu.wav=o m,2805,253,-261,204,140",
                "ma-mi-mu.wav=o m,2805,253,-261,204,140",
                "_mi-mu.wav=i mA3,1495,253,-261,204,140",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_ma-mi-mu.wav=o m,2918,120,-210,120,80",
                "ma-mi-mu.wav=o m,2918,120,-210,120,80",
                "_mi-mu.wav=i mA3,1608,120,-210,120,80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_underscore_vc_positive_offset_shift_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_ma-mi-mu.wav=o m,2805,120,-210,120,80",
        "ma-mi-mu.wav=o m,2918,120,-210,120,80",
        "_mi-mu.wav=i mA3,1608,120,-210,120,80",
    ]


def test_no_mfa_lightgbm_safety_filter_preserves_pure_vowel_sequence_grid(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-i-n-a.wav=- a,950,160,-170,120,85",
                "_a-i-n-a.wav=a i,1070,420,-600,300,100",
                "_a-i-n-a.wav=i \u3093,1570,420,-600,300,100",
                "_a-i-n-a.wav=n \u3042,2070,420,-600,300,100",
                "_a-i-n-a.wav=\u3042 -,2500,320,-480,200,100",
                "ka.wav=a k,1000,120,-160,80,45",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-i-n-a.wav=- a,920,220,-260,150,110",
                "_a-i-n-a.wav=a i,1085,260,-300,180,130",
                "_a-i-n-a.wav=i \u3093,1585,260,-300,180,130",
                "_a-i-n-a.wav=n \u3042,2085,260,-300,180,130",
                "_a-i-n-a.wav=\u3042 -,2400,350,-520,210,130",
                "ka.wav=a k,980,220,-240,170,140",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
        max_overlap_increase_ms=20.0,
    )

    lines = post.read_text(encoding="utf-8").splitlines()
    assert report["restored_pure_vowel_sequence_rows"] == 4
    assert report["restored_terminal_rows"] == 1
    assert report["capped_overlap_rows"] == 1
    assert report["capped_cv_head_vowel_pre_overlap_rows"] == 1
    assert lines[:5] == [
        "_a-i-n-a.wav=- a,950,160,-170,90,45",
        "_a-i-n-a.wav=a i,1070,420,-600,300,100",
        "_a-i-n-a.wav=i \u3093,1570,420,-600,300,100",
        "_a-i-n-a.wav=n \u3042,2070,420,-600,300,100",
        "_a-i-n-a.wav=\u3042 -,2500,320,-480,200,100",
    ]
    assert lines[5] == "ka.wav=a k,980,220,-240,170,65"


def test_no_mfa_lightgbm_safety_filter_preserves_pure_vowel_onset_grid(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-a-i.wav=- a,1040,160,-170,120,85",
                "_a-a-i.wav=a a,1410,420,-600,250,83",
                "_a-a-i.wav=a,1660,125,-360,0,0",
                "_a-a-i.wav=a i,1910,420,-600,250,83",
                "_a-a-i.wav=i RA3,4800,420,-600,250,83",
                "ka.wav=a k,1000,120,-160,80,45",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-a-i.wav=- a,780,210,-260,150,110",
                "_a-a-i.wav=a a,1190,260,-300,180,130",
                "_a-a-i.wav=a,900,760,-1200,510,250",
                "_a-a-i.wav=a i,1680,260,-300,180,130",
                "_a-a-i.wav=i RA3,3300,260,-300,180,130",
                "ka.wav=a k,980,220,-240,170,140",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
        max_overlap_increase_ms=20.0,
    )

    lines = post.read_text(encoding="utf-8").splitlines()
    assert report["restored_pure_vowel_onset_sequence_rows"] == 5
    assert report["capped_overlap_rows"] == 1
    assert report["capped_cv_head_vowel_pre_overlap_rows"] == 1
    assert lines[:5] == [
        "_a-a-i.wav=- a,1040,160,-170,90,45",
        "_a-a-i.wav=a a,1410,420,-600,250,83",
        "_a-a-i.wav=a,1660,125,-360,0,0",
        "_a-a-i.wav=a i,1910,420,-600,250,83",
        "_a-a-i.wav=i RA3,4800,420,-600,250,83",
    ]
    assert lines[5] == "ka.wav=a k,980,220,-240,170,65"


def test_no_mfa_lightgbm_safety_filter_preserves_headless_pure_vowel_onset_grid(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_o-u-n-a.wav=u n,1980,420,-600,250,83",
                "_o-u-n-a.wav=n a,2480,420,-600,250,83",
                "_o-u-n-a.wav=a n,2980,420,-600,250,83",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_o-u-n-a.wav=u n,1240,300,-380,180,110",
                "_o-u-n-a.wav=n a,1690,300,-380,180,110",
                "_o-u-n-a.wav=a n,2020,300,-380,180,110",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_pure_vowel_onset_sequence_rows"] == 3
    assert post.read_text(encoding="utf-8").splitlines() == pre.read_text(encoding="utf-8").splitlines()


def test_no_mfa_lightgbm_safety_filter_preserves_headless_vc_onset_grid(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_i-g-i-z.wav=i gA3,1390,300,-330,250,83",
                "_i-g-i-z.wav=i zA3,2390,300,-330,250,83",
                "_i-g-i-z.wav=i dA3,3390,300,-330,250,83",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_i-g-i-z.wav=i gA3,1060,260,-290,180,120",
                "_i-g-i-z.wav=i zA3,1730,260,-290,180,120",
                "_i-g-i-z.wav=i dA3,2520,260,-290,180,120",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_headless_vc_onset_sequence_rows"] == 3
    assert report["restored_pure_vowel_onset_sequence_rows"] == 0
    assert post.read_text(encoding="utf-8").splitlines() == pre.read_text(encoding="utf-8").splitlines()


def test_no_mfa_lightgbm_safety_filter_preserves_regular_pair_onset_grid(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-ka-sa.wav=a k,1338.5,254,-262,205,140",
                "_a-ka-sa.wav=ka,1678.5,190,-198,150,115",
                "_a-ka-sa.wav=a s,1838.5,253,-261,204,140",
                "_a-ka-sa.wav=sa,2178.5,190,-198,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-ka-sa.wav=a k,780,260,-300,180,160",
                "_a-ka-sa.wav=ka,1020,220,-500,180,150",
                "_a-ka-sa.wav=a s,1540,260,-300,180,160",
                "_a-ka-sa.wav=sa,1700,220,-500,180,150",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_regular_pair_onset_sequence_rows"] == 4
    assert report["repaired_vc_cutoff_order_rows"] == 2
    assert report["capped_overlap_rows"] == 4
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_a-ka-sa.wav=a k,1338.5,260,-305,180,145",
        "_a-ka-sa.wav=ka,1678.5,220,-198,180,120",
        "_a-ka-sa.wav=a s,1838.5,260,-305,180,145",
        "_a-ka-sa.wav=sa,2178.5,220,-198,180,120",
    ]


def test_no_mfa_lightgbm_safety_filter_caps_pitch_regular_pair_cv_pre_overlap(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-ka-sa.wav=a k,1338.5,254,-262,205,140",
                "_a-ka-sa.wav=kaA3,1678.5,190,-198,150,115",
                "_a-ka-sa.wav=a s,1838.5,253,-261,204,140",
                "_a-ka-sa.wav=saA3,2178.5,190,-580,150,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-ka-sa.wav=a k,780,260,-300,180,140",
                "_a-ka-sa.wav=kaA3,1020,220,-500,180,115",
                "_a-ka-sa.wav=a s,1540,260,-300,180,140",
                "_a-ka-sa.wav=saA3,1700,220,-500,180,115",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_regular_pair_onset_sequence_rows"] == 4
    assert report["repaired_vc_cutoff_order_rows"] == 2
    assert report["capped_cv_pre_overlap_rows"] == 2
    assert report["capped_pitch_regular_pair_cv_pre_overlap_rows"] == 2
    assert report["clamped_pitch_regular_pair_cv_cutoff_rows"] == 2
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_a-ka-sa.wav=a k,1338.5,260,-305,180,140",
        "_a-ka-sa.wav=kaA3,1678.5,220,-330,70,25",
        "_a-ka-sa.wav=a s,1838.5,260,-305,180,140",
        "_a-ka-sa.wav=saA3,2178.5,220,-420,70,25",
    ]


def test_no_mfa_lightgbm_safety_filter_preserves_terminal_release_r_repairs(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_i-ki-ri.wav=i RA3,4754.467,420,-600,250,83",
                "aR.wav=a R,1899.683,340,-557.18,139.57,50",
                "_i-ri.wav=i rA3,4700,420,-600,250,83",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_i-ki-ri.wav=i RA3,4100,260,-300,180,140",
                "aR.wav=a R,1138.82,340,-557.18,139.57,50",
                "_i-ri.wav=i rA3,3300,260,-300,180,140",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_terminal_release_r_rows"] == 1
    assert report["restored_standalone_release_r_offset_rows"] == 1
    assert report["capped_overlap_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines() == [
        "_i-ki-ri.wav=i RA3,4754.467,420,-600,250,83",
        "aR.wav=a R,1899.683,340,-557.18,139.57,50",
        "_i-ri.wav=i rA3,3300,260,-300,180,88",
    ]


def test_no_mfa_lightgbm_safety_filter_caps_non_open_standalone_release_r_cutoff(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("iR.wav=i R,1554.286,120,-146,80,45\n", encoding="utf-8")
    post.write_text("iR.wav=i R,1000,340,-686,149,56\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_standalone_release_r_offset_rows"] == 1
    assert report["capped_standalone_release_r_cutoff_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "iR.wav=i R,1554.286,340,-420,149,56"


def test_no_mfa_lightgbm_safety_filter_preserves_terminal_release_r_terminal_copy(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text(
        "\n".join(
            [
                "_a-n-a.wav=a R,3630,320,-480,200,100",
                "_a-n-a.wav=a -,3630,320,-480,200,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    post.write_text(
        "\n".join(
            [
                "_a-n-a.wav=a R,2968.81,189.21,-206.7,140.18,50",
                "_a-n-a.wav=a -,3630,320,-480,200,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_terminal_release_r_terminal_copy_rows"] == 1
    assert post.read_text(encoding="utf-8").splitlines() == pre.read_text(encoding="utf-8").splitlines()


def test_validation_splitter_treats_trusted_breath_source_timing_as_clean():
    from core.generation.common.validation_splitter import validate_oto_lines

    records = validate_oto_lines(
        ["br1.wav=br1,2191.38,185.713,-186.713,167.309,15.058"],
        row_diagnostics={
            0: {
                "row_plan": {
                    "role_family": "policy_breath",
                    "source_timing_trusted": True,
                    "source_row_index": 12,
                },
                "hsmm_min_selected_vs_best_local_margin": -0.75,
                "local_refine_low_margin_count": 1,
            }
        },
    )

    assert len(records) == 1
    assert records[0].split == "clean"
    assert records[0].role_family == "policy_breath"
    assert records[0].source_timing_trusted is True


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


def test_hsmm_workflow_with_source_oto_preserves_base_alias_rows(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.wav", "i.wav"):
        _write_tone_wav(wav_dir / name, duration_s=0.40)
    source_oto = tmp_path / "source.ini"
    source_oto.write_text(
        "\n".join(
            [
                "a.wav=base_a_first,999,888,-777,666,555",
                "i.wav=base_i,1234,987,-654,321,210",
                "a.wav=base_a_second,1111,222,-333,444,55",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_oto = tmp_path / "out.ini"

    posterior = FramePosterior(
        wav_path="stub.wav",
        times_ms=[0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        class_probs={
            "silence": [0.3, 0.2, 0.1, 0.05, 0.1, 0.2, 0.3],
            "consonant": [0.1, 0.2, 0.5, 0.2, 0.1, 0.1, 0.1],
            "vowel": [0.1, 0.3, 0.7, 0.9, 0.7, 0.3, 0.1],
            "other": [0.1] * 7,
        },
        event_scores={label: [0.0] * 7 for label in EVENT_LABELS},
        acoustic_scores={
            "transition_likelihood": [0.1, 0.2, 0.9, 0.3, 0.2, 0.1, 0.1],
            "voicing": [0.1, 0.2, 0.7, 0.9, 0.7, 0.2, 0.1],
            "nucleus_likelihood": [0.1, 0.2, 0.6, 0.95, 0.6, 0.2, 0.1],
            "silence_likelihood": [0.6, 0.4, 0.1, 0.05, 0.1, 0.4, 0.7],
        },
    )
    prediction = SimpleNamespace(
        posterior=posterior,
        decoded_events=(),
        slot_result=None,
    )
    monkeypatch.setattr("core.mfa_free_oto.workflow.predict_wav", lambda *args, **kwargs: prediction)

    decode_calls = []

    def fake_decode_filename_slots_with_hsmm(_posterior, slots, **_kwargs):
        decode_calls.append(tuple(slot.wav for slot in slots))
        first = slots[0]
        result = SimpleNamespace(
            ok=True,
            reason="",
            score=0.91,
            timeout=False,
            pruned_endpoint_count=0,
            meta={},
            states=(),
        )
        return SimpleNamespace(
            ok=True,
            result=result,
            events=(
                {
                    "label": "cv_boundary",
                    "selected_time_ms": 120.0,
                    "score": 0.91,
                    "expected_phone": first.vowel_phone,
                    "expected_phone_index": first.vowel_phone_index,
                    "slot_index": 0,
                    "frame_index": 2,
                    "source": "filename_hsmm",
                },
            ),
            states=(),
            frame_scores={},
            state_interval_priors={},
            diagnostics={"state_count": len(slots)},
        )

    monkeypatch.setattr(
        "core.mfa_free_oto.workflow.decode_filename_slots_with_hsmm",
        fake_decode_filename_slots_with_hsmm,
    )

    report = generate_no_mfa_oto_with_model_context(
        wav_dir=str(wav_dir),
        out_path=str(out_oto),
        source_oto_path=str(source_oto),
        checkpoint_path="",
        language="japanese",
        format_type="VCV",
        use_hsmm_decoder=True,
    )

    assert report.processed == 3
    assert not report.errors
    assert decode_calls
    lines = out_oto.read_text(encoding="utf-8").splitlines()
    assert [line.split("=", 1)[1].split(",", 1)[0] for line in lines] == [
        "base_a_first",
        "base_i",
        "base_a_second",
    ]
    assert "999,888,-777,666,555" not in out_oto.read_text(encoding="utf-8")
    assert any(record["selected_event_source"] == "filename_hsmm" for record in report.timeline_debug)
    assert "base_oto_alias_list_used" in report.warnings


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


def test_hsmm_oto_ui_preview_uses_review_generate_hsmm(tmp_path):
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

        def _append_log(self, message):
            self.logs.append(str(message))

        def _popen_subprocess_hidden(self, cmd, **kwargs):
            self.commands.append(list(cmd))
            return FakeProcess()

        def _iter_decoded_stdout_lines(self, process):
            generated_path = tmp_path / "generated.ini"
            generated_path.write_text("a.wav=a,0,0,0,0,0\nb.wav=i,0,0,0,0,0\n", encoding="utf-8")
            return iter(
                [
                    json.dumps(
                        {
                            "processed": 2,
                            "total": 2,
                            "generated_oto_path": str(generated_path),
                            "split_counts": {
                                "total": 2,
                                "fix_required": 0,
                                "attention_only": 1,
                                "clean": 1,
                            },
                            "split_output_paths": {
                                "review_session": str(tmp_path / "review_session.json"),
                            },
                        },
                        ensure_ascii=False,
                    )
                ]
            )

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "oto.ini").write_text("a.wav=local should not auto-use,0,0,0,0,0\n", encoding="utf-8")
    out_path = tmp_path / "preview.ini"
    dummy = Dummy()

    generated, total, errors = dummy._run_hsmm_oto_preview_generation(
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
    assert command[:4] == [sys.executable or "python", "-m", "scripts.dev.mfa_free_oto_review", "generate"]
    assert "--use-hsmm-decoder" in command
    assert "--template-oto" not in command
    assert "--checkpoint" not in command
    assert out_path.read_text(encoding="utf-8").count("\n") == 2
    assert any("no base oto" in log for log in dummy.logs)
    assert any("review_session:" in log for log in dummy.logs)


def test_hsmm_oto_ui_preview_can_request_lightgbm_postprocess(tmp_path):
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

        def _append_log(self, message):
            self.logs.append(str(message))

        def _popen_subprocess_hidden(self, cmd, **kwargs):
            self.commands.append(list(cmd))
            return FakeProcess()

        def _iter_decoded_stdout_lines(self, process):
            generated_path = tmp_path / "generated.ini"
            pre_lightgbm_path = tmp_path / "pre_lightgbm.ini"
            generated_path.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
            pre_lightgbm_path.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
            return iter(
                [
                    json.dumps(
                        {
                            "processed": 1,
                            "total": 1,
                            "generated_oto_path": str(generated_path),
                            "lightgbm_postprocess": {
                                "enabled": True,
                                "status": "applied",
                                "changed": 1,
                                "pre_lightgbm_path": str(pre_lightgbm_path),
                            },
                            "split_counts": {"total": 1, "fix_required": 0, "attention_only": 0, "clean": 1},
                            "split_output_paths": {},
                        },
                        ensure_ascii=False,
                    )
                ]
            )

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_path = tmp_path / "preview.ini"
    dummy = Dummy()

    generated, total, errors = dummy._run_hsmm_oto_preview_generation(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        language="japanese",
        format_type="CVVC",
        apply_lightgbm=True,
        lightgbm_policy="on",
        lightgbm_model_dir=str(model_dir),
    )

    assert errors == []
    assert generated == 1
    assert total == 1
    command = dummy.commands[0]
    assert "--apply-lightgbm" in command
    policy_index = command.index("--lightgbm-policy")
    assert command[policy_index + 1] == "on"
    model_index = command.index("--lightgbm-model-dir")
    assert command[model_index + 1] == str(model_dir.resolve())
    assert any("LightGBM postprocess requested" in log for log in dummy.logs)
    assert any("LightGBM postprocess: status=applied, changed=1" in log for log in dummy.logs)
    assert any("pre-LightGBM oto:" in log for log in dummy.logs)


def test_hsmm_oto_ui_preview_passes_base_oto_as_template(tmp_path):
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

        def _append_log(self, message):
            self.logs.append(str(message))

        def _popen_subprocess_hidden(self, cmd, **kwargs):
            self.commands.append(list(cmd))
            return FakeProcess()

        def _iter_decoded_stdout_lines(self, process):
            generated_path = tmp_path / "generated.ini"
            generated_path.write_text("a.wav=base alias,0,0,0,0,0\n", encoding="utf-8")
            return iter(
                [
                    json.dumps(
                        {
                            "processed": 1,
                            "total": 1,
                            "generated_oto_path": str(generated_path),
                            "split_counts": {"total": 1, "fix_required": 0, "attention_only": 0, "clean": 1},
                            "split_output_paths": {},
                        },
                        ensure_ascii=False,
                    )
                ]
            )

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    source_oto = tmp_path / "baseoto.ini"
    source_oto.write_text("a.wav=base alias,999,888,-777,666,555\n", encoding="utf-8")
    out_path = tmp_path / "preview.ini"
    dummy = Dummy()

    generated, total, errors = dummy._run_hsmm_oto_preview_generation(
        wav_dir=str(wav_dir),
        out_path=str(out_path),
        source_oto_path=str(source_oto),
        language="japanese",
        format_type="CVVC",
    )

    assert errors == []
    assert generated == 1
    assert total == 1
    command = dummy.commands[0]
    template_index = command.index("--template-oto")
    assert command[template_index + 1] == str(source_oto.resolve())
    assert "--use-hsmm-decoder" in command
    assert any("base oto alias list:" in log and str(source_oto.resolve()) in log for log in dummy.logs)
    assert any("base oto timing ignored" in log for log in dummy.logs)


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


def test_parse_oto_line_accepts_blank_overlap_as_zero():
    from core.generation.common.oto_file_utils import parse_oto_line

    parsed = parse_oto_line("ka.wav=\u304b,3387.76,129.71,-282.99,48.07,")

    assert parsed is not None
    assert parsed["alias"] == "\u304b"
    assert parsed["offset"] == pytest.approx(3387.76)
    assert parsed["ovl"] == 0.0


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
