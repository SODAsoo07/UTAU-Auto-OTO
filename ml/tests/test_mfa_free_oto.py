from __future__ import annotations

import json
import math
import struct
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.mfa_free_oto.decode import decode_monotonic_events
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
    _alias_targets_from_template_rows_or_dp,
    _assign_alias_target_indices,
    _alias_phone_sequence,
    _alias_type_for_row,
    _cvvc_initial_cv_head_phone_sequence,
    _cvvc_initial_cv_head_vowel_token,
    _is_cvvc_terminal_release_r_alias,
    _is_nonphonetic_special_alias,
    _ja_cvvc_repair_dispatch,
    _refine_anchor_sequence_locally,
    adapt_template_row,
    anchors_from_prediction,
    assign_template_row_anchors,
    bootstrap_row,
    expected_slots_for_template_rows,
    parse_template_oto_line,
    repair_cvvc_row_sequence,
    repair_cvvc_vc_row_sequence,
    timeline_expected_slots_for_template_rows,
)
from core.mfa_free_oto.row_plan import build_filename_slots, filename_phone_sequence_from_slots
from core.mfa_free_oto.review_overlay import render_review_html
from core.mfa_free_oto import runtime_inference as runtime_inference_module
from core.mfa_free_oto.slot_viterbi import (
    ExpectedSlot,
    SlotAssignment,
    SlotViterbiResult,
    _expected_time_window,
    _slot_candidates,
    assign_slots_viterbi,
    expected_cv_slots_from_phones,
    slot_assignments_to_decoded_events,
)
from core.mfa_free_oto.targets import infer_event_slot_indices, rasterize_targets
from core.mfa_free_oto.types import DecodedEvent, EVENT_LABELS, FRAME_LABELS, FramePosterior
from core.mfa_free_oto.workflow import (
    _hsmm_runtime_replacement_rejection_reason,
    _template_group_in_filename_order,
    generate_no_mfa_oto_with_model_context,
)


def _enable_ja_cvvc_reference_repairs(monkeypatch):
    monkeypatch.setenv("UTOA_ENABLE_JA_CVVC_REFERENCE_REPAIRS", "1")


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


def test_slot_event_targets_follow_viterbi_event_order():
    events = [
        {"label": "cv_boundary", "time_ms": 60.0},
        {"label": "vowel_nucleus", "time_ms": 100.0},
        {"label": "phone_change", "time_ms": 180.0},
    ]
    assert infer_event_slot_indices(events) == [0, 1, 2]
    targets = rasterize_targets(
        {"events": events, "frame_labels": []},
        np.arange(0.0, 240.0, 10.0, dtype=np.float32),
        slot_event_bins=4,
    )
    assert targets.slot_event_targets.shape == (24, 4, len(EVENT_LABELS))
    assert targets.slot_event_mask.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert int(np.argmax(targets.slot_event_targets[:, 0, EVENT_LABELS.index("cv_boundary")])) == 6
    assert int(np.argmax(targets.slot_event_targets[:, 1, EVENT_LABELS.index("vowel_nucleus")])) == 10


def test_slot_viterbi_prefers_slot_specific_event_tracks():
    times = [float(idx * 10) for idx in range(31)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    event_scores["cv_boundary"][22] = 0.99
    event_scores["slot_event:0:cv_boundary"] = [0.01 for _ in times]
    event_scores["slot_event:1:cv_boundary"] = [0.01 for _ in times]
    event_scores["slot_event:0:cv_boundary"][7] = 0.99
    event_scores["slot_event:1:cv_boundary"][22] = 0.99
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.25 for _ in times] for label in FRAME_LABELS},
        event_scores=event_scores,
        acoustic_scores={},
    )
    slots = [
        ExpectedSlot(slot_index=0, phone_index=1, phone="a", role="cv", event_label="cv_boundary"),
        ExpectedSlot(slot_index=1, phone_index=3, phone="i", role="cv", event_label="cv_boundary"),
    ]
    result = assign_slots_viterbi(
        posterior,
        expected_slots=slots,
        expected_time_weight=0.0,
        min_event_score=0.01,
    )
    assert [item.selected_time_ms for item in result.assignments] == pytest.approx([70.0, 220.0])


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


def test_japanese_slot_active_window_requires_opt_in(monkeypatch):
    times = np.arange(0.0, 1010.0, 10.0, dtype=np.float32)
    silence = [0.92 if time_ms < 200.0 or time_ms > 800.0 else 0.08 for time_ms in times]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times.tolist(),
        class_probs={},
        event_scores={},
        acoustic_scores={"silence_likelihood": silence},
    )

    monkeypatch.delenv("UTOA_NO_MFA_JA_SLOT_ACTIVE_WINDOW", raising=False)
    assert _expected_time_window(posterior, times, 1000.0, language="japanese") == pytest.approx((0.0, 1000.0))

    monkeypatch.setenv("UTOA_NO_MFA_JA_SLOT_ACTIVE_WINDOW", "1")
    active_start, active_end = _expected_time_window(posterior, times, 1000.0, language="japanese")

    assert active_start > 0.0
    assert active_end < 1000.0


def test_slot_viterbi_sonorant_phone_change_prefers_voiced_onset_over_late_flux():
    times = [float(idx * 10) for idx in range(31)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.05 for _ in times],
        "flux_likelihood": [0.05 for _ in times],
        "sonorant_onset_likelihood": [0.05 for _ in times],
        "voicing": [0.90 for _ in times],
        "silence_likelihood": [0.05 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.86 if 110.0 <= time_ms <= 230.0 else 0.18
        class_probs["consonant"][idx] = 0.28 if 70.0 <= time_ms <= 115.0 else 0.08
        event_scores["phone_change"][idx] = max(
            0.01,
            0.04 * math.exp(-0.5 * ((time_ms - 90.0) / 8.0) ** 2),
            0.24 * math.exp(-0.5 * ((time_ms - 180.0) / 8.0) ** 2),
        )
        acoustic_scores["sonorant_onset_likelihood"][idx] = max(
            acoustic_scores["sonorant_onset_likelihood"][idx],
            0.95 * math.exp(-0.5 * ((time_ms - 90.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [ExpectedSlot(slot_index=0, phone_index=0, phone="n", role="vc", event_label="phone_change")]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    assert result.assignments[0].selected_time_ms == pytest.approx(90.0, abs=6.0)


def test_slot_viterbi_consecutive_cv_slots_do_not_collapse_to_one_peak_cluster():
    times = [float(idx * 10) for idx in range(211)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    for center in [360.0, 760.0, 1160.0, 1560.0]:
        for idx, time_ms in enumerate(times):
            event_scores["cv_boundary"][idx] = max(
                event_scores["cv_boundary"][idx],
                0.58 * math.exp(-0.5 * ((time_ms - center) / 10.0) ** 2),
            )
    for center in [1700.0, 1730.0, 1760.0, 1790.0]:
        for idx, time_ms in enumerate(times):
            event_scores["cv_boundary"][idx] = max(
                event_scores["cv_boundary"][idx],
                0.92 * math.exp(-0.5 * ((time_ms - center) / 8.0) ** 2),
            )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
    )
    slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx * 3 + 2, phone=phone, role="cv", event_label="cv_boundary")
        for idx, phone in enumerate(["u", "e", "o", "a"])
    ]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    selected = [assignment.selected_time_ms for assignment in result.assignments]
    assert min(b - a for a, b in zip(selected, selected[1:])) >= 180.0
    assert selected[0] < 1000.0


def test_slot_viterbi_template_cv_role_uses_boundary_acoustic_prior():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.04 for _ in times],
        "flux_likelihood": [0.04 for _ in times],
        "sonorant_onset_likelihood": [0.04 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["consonant"][idx] = 0.75 if 105.0 <= time_ms <= 140.0 else 0.05
        class_probs["vowel"][idx] = 0.78 if 135.0 <= time_ms <= 230.0 else 0.05
        event_scores["cv_boundary"][idx] = max(
            event_scores["cv_boundary"][idx],
            0.14 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
            0.24 * math.exp(-0.5 * ((time_ms - 260.0) / 8.0) ** 2),
        )
        acoustic_scores["transition_likelihood"][idx] = max(
            acoustic_scores["transition_likelihood"][idx],
            0.95 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
        acoustic_scores["flux_likelihood"][idx] = max(
            acoustic_scores["flux_likelihood"][idx],
            0.90 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
        acoustic_scores["sonorant_onset_likelihood"][idx] = max(
            acoustic_scores["sonorant_onset_likelihood"][idx],
            0.72 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [ExpectedSlot(slot_index=0, phone_index=1, phone="a", role="cv", event_label="cv_boundary")]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    assert result.assignments[0].selected_time_ms == pytest.approx(140.0, abs=6.0)


def test_slot_viterbi_template_v_role_uses_nucleus_acoustic_prior():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "voicing": [0.05 for _ in times],
        "nucleus_likelihood": [0.04 for _ in times],
        "world_nucleus": [0.04 for _ in times],
        "world_periodicity": [0.04 for _ in times],
        "world_spectral_stability": [0.04 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.82 if 110.0 <= time_ms <= 210.0 else 0.05
        event_scores["vowel_nucleus"][idx] = max(
            event_scores["vowel_nucleus"][idx],
            0.12 * math.exp(-0.5 * ((time_ms - 150.0) / 8.0) ** 2),
            0.28 * math.exp(-0.5 * ((time_ms - 280.0) / 8.0) ** 2),
        )
        for key, peak in (
            ("voicing", 0.96),
            ("nucleus_likelihood", 0.92),
            ("world_nucleus", 0.94),
            ("world_periodicity", 0.88),
            ("world_spectral_stability", 0.86),
        ):
            acoustic_scores[key][idx] = max(
                acoustic_scores[key][idx],
                peak * math.exp(-0.5 * ((time_ms - 150.0) / 8.0) ** 2),
            )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [ExpectedSlot(slot_index=0, phone_index=0, phone="a", role="v", event_label="vowel_nucleus")]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    assert result.assignments[0].selected_time_ms == pytest.approx(150.0, abs=6.0)


def test_korean_vowel_slots_use_expected_grid_when_nucleus_peaks_collapse(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_KR_CV_EXPECTED_HARD_WINDOW_MS", "80")
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "voicing": [0.05 for _ in times],
        "nucleus_likelihood": [0.04 for _ in times],
        "world_nucleus": [0.04 for _ in times],
        "world_periodicity": [0.04 for _ in times],
        "world_spectral_stability": [0.04 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.84 if 250.0 <= time_ms <= 330.0 else 0.08
        event_scores["vowel_nucleus"][idx] = max(
            event_scores["vowel_nucleus"][idx],
            0.36 * math.exp(-0.5 * ((time_ms - 300.0) / 8.0) ** 2),
        )
        for key in ("voicing", "nucleus_likelihood", "world_nucleus", "world_periodicity", "world_spectral_stability"):
            acoustic_scores[key][idx] = max(
                acoustic_scores[key][idx],
                0.90 * math.exp(-0.5 * ((time_ms - 300.0) / 8.0) ** 2),
            )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone=phone, role="v", event_label="vowel_nucleus")
        for idx, phone in enumerate(("a", "e", "i", "o"))
    ]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03, language="korean")

    assert result.ok
    assert [item.selected_time_ms for item in result.assignments] == pytest.approx([0.0, 120.0, 300.0, 380.0], abs=10.0)


def test_korean_ng_cluster_alias_targets_do_not_collapse_to_n_fallback():
    rows = [
        OtoTemplateRow("eunr_eumr_eungr_eulr.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("n R", "m R", "ng R", "l R")
    ]
    expected = ["eu", "n", "r", "eu", "m", "r", "eu", "ng", "r", "eu", "l", "r"]

    assert _alias_phone_sequence("ng R") == ["ng", "r"]
    assert _assign_alias_target_indices(rows, expected) == [2, 5, 8, 11]
    slots = timeline_expected_slots_for_template_rows(rows, expected, language="korean")

    assert [slot.phone_index for slot in slots] == [2, 3, 5, 8, 11]
    assert all(slot.phone_index != 1 for slot in slots)


def test_korean_no_space_cvc_coda_targets_following_onset_slot():
    rows = [
        OtoTemplateRow("bba'bbi'bbu.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("bba", "ap", "bbi", "ip", "bbu")
    ]
    expected = ["b", "b", "a", "b", "b", "i", "b", "b", "u"]

    assert _assign_alias_target_indices(rows, expected, language="korean") == [2, 3, 5, 6, 8]

    slots = expected_slots_for_template_rows(rows, expected, language="korean")
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (2, "cv", "cv_boundary"),
        (3, "vc", "phone_change"),
        (5, "implicit_cv", "cv_boundary"),
        (6, "vc", "phone_change"),
        (8, "implicit_cv", "cv_boundary"),
    ]


def test_korean_compact_glide_vowels_use_cv_boundary_slots():
    rows = [
        OtoTemplateRow("wa'we'wi'weo.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("wa", "we", "wi", "weo")
    ]
    expected = ["wa", "we", "wi", "weo"]

    slots = expected_slots_for_template_rows(rows, expected, language="korean")

    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in slots] == [
        (0, "wa", "cv", "cv_boundary"),
        (1, "we", "cv", "cv_boundary"),
        (2, "wi", "cv", "cv_boundary"),
        (3, "weo", "cv", "cv_boundary"),
    ]


def test_korean_yoon_timeline_keeps_unaliased_leading_vowel_context():
    rows = [
        OtoTemplateRow("ya'pya'pye'pyeo'pyo'pyu'peui.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("pya", "pye", "pyeo", "pyo", "pyu", "peui")
    ]
    expected = ["ya", "p", "ya", "p", "ye", "p", "yeo", "p", "yo", "p", "yu", "p", "eui"]

    row_slots = expected_slots_for_template_rows(rows, expected, language="korean")
    timeline_slots = timeline_expected_slots_for_template_rows(rows, expected, language="korean")

    assert [slot.phone_index for slot in row_slots[:2]] == [2, 4]
    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in timeline_slots[:2]] == [
        (0, "ya", "v", "vowel_nucleus"),
        (2, "ya", "cv", "cv_boundary"),
    ]
    assert [slot.slot_index for slot in timeline_slots] == list(range(len(timeline_slots)))


def test_korean_liquid_yoon_timeline_keeps_vc_leading_context():
    rows = [
        OtoTemplateRow("al'rwal'rwel'rwil'rweol.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("lwa", "lwe", "lwi", "lweo")
    ]
    expected = ["a", "l", "r", "wa", "l", "r", "we", "l", "r", "wi", "l", "r", "weo", "l"]

    timeline_slots = timeline_expected_slots_for_template_rows(rows, expected, language="korean")

    assert [(slot.phone_index, slot.phone, slot.role, slot.event_label) for slot in timeline_slots[:3]] == [
        (0, "a", "v", "vowel_nucleus"),
        (1, "l", "vc", "phone_change"),
        (3, "wa", "cv", "cv_boundary"),
    ]
    assert [slot.slot_index for slot in timeline_slots] == list(range(len(timeline_slots)))


def test_korean_dense_cv_skip_gap_repairs_following_anchors():
    rows = [
        OtoTemplateRow("jja'jji'jju'jje'jjo'jjeu'jjeo'jja.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("jja", "jji", "jju", "jje", "jjo", "jjeu", "jjeo")
    ]
    expected = [
        "j",
        "j",
        "a",
        "j",
        "j",
        "i",
        "j",
        "j",
        "u",
        "j",
        "j",
        "e",
        "j",
        "j",
        "o",
        "j",
        "j",
        "eu",
        "j",
        "j",
        "eo",
        "j",
        "j",
        "a",
    ]
    times = [float(idx * 10) for idx in range(521)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = [
        {
            "label": "cv_boundary",
            "selected_time_ms": time_ms,
            "score": 0.9,
            "frame_index": int(round(time_ms / 10.0)),
            "expected_phone": phone,
            "expected_phone_index": phone_index,
            "slot_index": slot_index,
        }
        for slot_index, (phone_index, phone, time_ms) in enumerate(
            [
                (2, "a", 1010.0),
                (5, "i", 1540.0),
                (8, "u", 2510.0),
                (11, "e", 3020.0),
                (14, "o", 3540.0),
                (17, "eu", 4040.0),
                (20, "eo", 4530.0),
            ]
        )
    ]

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert [round(anchor.anchor_abs_ms, 1) for anchor in anchors if anchor is not None] == [
        1010.0,
        1540.0,
        2050.0,
        2560.0,
        3080.0,
        3580.0,
        4070.0,
    ]
    assert any("korean_cv_skip_gap_repaired:" in warning for warning in anchors[2].warnings)


def test_korean_sonorant_cv_leading_skip_gap_repairs_backtracked_cvc_row():
    rows = [
        OtoTemplateRow("ang'ing'ung'eng'ong'eung'eong'ang.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in (
            "ang",
            "nga",
            "ngi",
            "ing",
            "ngu",
            "ung",
            "nge",
            "eng",
            "ngo",
            "ong",
            "ngeu",
            "eung",
            "ngeo",
            "eong",
        )
    ]
    expected = [
        "a",
        "ng",
        "i",
        "ng",
        "u",
        "ng",
        "e",
        "ng",
        "o",
        "ng",
        "eu",
        "ng",
        "eo",
        "ng",
        "a",
        "ng",
    ]
    times = [float(idx * 10) for idx in range(481)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = []
    for slot_index, (phone_index, phone, label, time_ms) in enumerate(
        [
            (1, "ng", "phone_change", 1000.0),
            (2, "i", "cv_boundary", 1002.0),
            (3, "ng", "phone_change", 1490.0),
            (4, "u", "cv_boundary", 1910.0),
            (5, "ng", "phone_change", 1950.0),
            (6, "e", "cv_boundary", 2520.0),
            (7, "ng", "phone_change", 2560.0),
            (8, "o", "cv_boundary", 2960.0),
            (9, "ng", "phone_change", 2970.0),
            (10, "eu", "cv_boundary", 3450.0),
            (11, "ng", "phone_change", 3510.0),
            (12, "eo", "cv_boundary", 3970.0),
            (13, "ng", "phone_change", 4410.0),
            (14, "a", "cv_boundary", 4430.0),
        ]
    ):
        decoded_events.append(
            {
                "label": label,
                "selected_time_ms": time_ms,
                "score": 0.9,
                "frame_index": int(round(time_ms / 10.0)),
                "expected_phone": phone,
                "expected_phone_index": phone_index,
                "slot_index": slot_index,
            }
        )

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert anchors[2].anchor_abs_ms == pytest.approx(1405.0, abs=0.1)
    assert any("korean_cv_leading_skip_gap_repaired:" in warning for warning in anchors[2].warnings)
    assert anchors[4].anchor_abs_ms == pytest.approx(1910.0, abs=0.1)


def test_korean_interleaved_cvc_compressed_cv_repairs_delayed_cv_only():
    rows = [
        OtoTemplateRow("dda'ddi'ddu'dde'ddo'ddeu'ddeo'dda.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("dda", "at", "ddi", "it", "ddu", "ut", "dde", "et", "ddo", "ot", "ddeu", "eut", "ddeo", "eot")
    ]
    expected = [
        "d",
        "d",
        "a",
        "d",
        "d",
        "i",
        "d",
        "d",
        "u",
        "d",
        "d",
        "e",
        "d",
        "d",
        "o",
        "d",
        "d",
        "eu",
        "d",
        "d",
        "eo",
        "d",
        "d",
        "a",
    ]
    times = [float(idx * 10) for idx in range(471)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = []
    for slot_index, (phone_index, phone, label, time_ms) in enumerate(
        [
            (2, "a", "cv_boundary", 1000.0),
            (3, "d", "phone_change", 1480.0),
            (5, "i", "cv_boundary", 1490.0),
            (6, "d", "phone_change", 1980.0),
            (8, "u", "cv_boundary", 2000.0),
            (9, "d", "phone_change", 2440.0),
            (11, "e", "cv_boundary", 2450.0),
            (12, "d", "phone_change", 2980.0),
            (14, "o", "cv_boundary", 3000.0),
            (15, "d", "phone_change", 3500.0),
            (17, "eu", "cv_boundary", 4000.0),
            (18, "d", "phone_change", 4018.0),
            (20, "eo", "cv_boundary", 4510.0),
            (21, "d", "phone_change", 4540.0),
        ]
    ):
        decoded_events.append(
            {
                "label": label,
                "selected_time_ms": time_ms,
                "score": 0.9,
                "frame_index": int(round(time_ms / 10.0)),
                "expected_phone": phone,
                "expected_phone_index": phone_index,
                "slot_index": slot_index,
            }
        )

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert anchors[10].anchor_abs_ms == pytest.approx(3680.0, abs=0.1)
    assert anchors[11].anchor_abs_ms == pytest.approx(4018.0, abs=0.1)
    assert anchors[12].anchor_abs_ms == pytest.approx(4198.0, abs=0.1)
    assert any("korean_interleaved_cvc_compressed_cv_repaired:" in warning for warning in anchors[10].warnings)
    assert any("korean_interleaved_cvc_compressed_cv_repaired:" in warning for warning in anchors[12].warnings)


def test_korean_interleaved_cvc_late_suffix_block_repairs_tail_block():
    rows = [
        OtoTemplateRow("bba'bbi'bbu'bbe'bbo'bbeu'bbeo'bba.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("bba", "ap", "bbi", "ip", "bbu", "up", "bbe", "ep", "bbo", "op", "bbeu", "eup", "bbeo", "eop")
    ]
    expected = [
        "b",
        "b",
        "a",
        "b",
        "b",
        "i",
        "b",
        "b",
        "u",
        "b",
        "b",
        "e",
        "b",
        "b",
        "o",
        "b",
        "b",
        "eu",
        "b",
        "b",
        "eo",
        "b",
        "b",
        "a",
    ]
    times = [float(idx * 10) for idx in range(517)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = []
    for slot_index, (phone_index, phone, label, time_ms) in enumerate(
        [
            (2, "a", "cv_boundary", 1250.0),
            (3, "b", "phone_change", 1490.0),
            (5, "i", "cv_boundary", 1830.0),
            (6, "b", "phone_change", 2020.0),
            (8, "u", "cv_boundary", 2030.0),
            (9, "b", "phone_change", 2510.0),
            (11, "e", "cv_boundary", 2580.0),
            (12, "b", "phone_change", 3000.0),
            (14, "o", "cv_boundary", 3130.0),
            (15, "b", "phone_change", 3500.0),
            (17, "eu", "cv_boundary", 4010.0),
            (18, "b", "phone_change", 4280.0),
            (20, "eo", "cv_boundary", 4490.0),
            (21, "b", "phone_change", 4530.0),
        ]
    ):
        decoded_events.append(
            {
                "label": label,
                "selected_time_ms": time_ms,
                "score": 0.9,
                "frame_index": int(round(time_ms / 10.0)),
                "expected_phone": phone,
                "expected_phone_index": phone_index,
                "slot_index": slot_index,
            }
        )

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert anchors[9].anchor_abs_ms == pytest.approx(3500.0, abs=0.1)
    assert anchors[10].anchor_abs_ms == pytest.approx(3680.0, abs=0.1)
    assert anchors[11].anchor_abs_ms == pytest.approx(3950.0, abs=0.1)
    assert anchors[12].anchor_abs_ms == pytest.approx(4160.0, abs=0.1)
    assert anchors[13].anchor_abs_ms == pytest.approx(4200.0, abs=0.1)
    assert any("korean_interleaved_cvc_late_suffix_block_repaired:" in warning for warning in anchors[10].warnings)
    assert any("korean_interleaved_cvc_late_suffix_block_repaired:" in warning for warning in anchors[13].warnings)


def test_korean_compound_vowel_skip_gap_repairs_only_large_gap():
    rows = [
        OtoTemplateRow("wa'we'wi'weo.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("wa", "we", "wi", "weo")
    ]
    expected = ["wa", "we", "wi", "weo"]
    times = [float(idx * 10) for idx in range(381)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = [
        {
            "label": "cv_boundary",
            "selected_time_ms": time_ms,
            "score": 0.9,
            "frame_index": int(round(time_ms / 10.0)),
            "expected_phone": phone,
            "expected_phone_index": phone_index,
            "slot_index": slot_index,
        }
        for slot_index, (phone_index, phone, time_ms) in enumerate(
            [
                (0, "wa", 1060.0),
                (1, "we", 1500.0),
                (2, "wi", 2530.0),
                (3, "weo", 2770.0),
            ]
        )
    ]

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert [round(anchor.anchor_abs_ms, 1) for anchor in anchors if anchor is not None] == [
        1060.0,
        1500.0,
        1940.0,
        2380.0,
    ]
    assert any("korean_compound_vowel_skip_gap_repaired:" in warning for warning in anchors[2].warnings)


def test_korean_compound_vowel_skip_gap_ignores_moderate_yoon_gap():
    rows = [
        OtoTemplateRow("ya'ye'yeo'yo'yu'eui.wav", alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in ("ya", "ye", "yeo", "yo", "yu", "eui")
    ]
    expected = ["ya", "ye", "yeo", "yo", "yu", "eui"]
    times = [float(idx * 10) for idx in range(421)]
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs={label: [0.05 for _ in times] for label in FRAME_LABELS},
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
    )
    decoded_events = [
        {
            "label": "cv_boundary",
            "selected_time_ms": time_ms,
            "score": 0.9,
            "frame_index": int(round(time_ms / 10.0)),
            "expected_phone": phone,
            "expected_phone_index": phone_index,
            "slot_index": slot_index,
        }
        for slot_index, (phone_index, phone, time_ms) in enumerate(
            [
                (0, "ya", 1010.0),
                (1, "ye", 1330.0),
                (2, "yeo", 2040.0),
                (3, "yo", 2500.0),
                (4, "yu", 2980.0),
                (5, "eui", 3470.0),
            ]
        )
    ]

    anchors = assign_template_row_anchors(
        posterior,
        decoded_events,
        rows,
        expected_phones=expected,
        language="korean",
        use_source_timing_prior=False,
    )

    assert [round(anchor.anchor_abs_ms, 1) for anchor in anchors if anchor is not None] == [
        1010.0,
        1330.0,
        2040.0,
        2500.0,
        2980.0,
        3470.0,
    ]
    assert not any(
        "korean_compound_vowel_skip_gap_repaired:" in warning
        for anchor in anchors
        for warning in anchor.warnings
    )


def test_korean_sonorant_vc_slots_use_phone_specific_expected_phase(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_KR_VC_EXPECTED_HARD_WINDOW_MS", "90")
    times = [float(idx * 10) for idx in range(321)]

    def posterior_for(phone: str, *, distractor_center: float) -> FramePosterior:
        event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
        class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
        acoustic_scores = {
            "transition_likelihood": [0.04 for _ in times],
            "flux_likelihood": [0.04 for _ in times],
            "sonorant_onset_likelihood": [0.04 for _ in times],
            "spectral_shape_delta_likelihood": [0.04 for _ in times],
            "voicing": [0.82 for _ in times],
            "silence_likelihood": [0.08 for _ in times],
        }
        direction = -0.45 if phone == "n" else 0.55
        for slot_idx in range(9):
            center = max(0.0, slot_idx * 400.0 + (400.0 * direction))
            for frame_idx, time_ms in enumerate(times):
                peak = math.exp(-0.5 * ((time_ms - center) / 8.0) ** 2)
                event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], 0.36 * peak)
                acoustic_scores["sonorant_onset_likelihood"][frame_idx] = max(
                    acoustic_scores["sonorant_onset_likelihood"][frame_idx],
                    0.72 * peak,
                )
                acoustic_scores["spectral_shape_delta_likelihood"][frame_idx] = max(
                    acoustic_scores["spectral_shape_delta_likelihood"][frame_idx],
                    0.68 * peak,
                )
        for frame_idx, time_ms in enumerate(times):
            late_peak = math.exp(-0.5 * ((time_ms - distractor_center) / 8.0) ** 2)
            event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], 0.96 * late_peak)
            acoustic_scores["sonorant_onset_likelihood"][frame_idx] = max(
                acoustic_scores["sonorant_onset_likelihood"][frame_idx],
                0.95 * late_peak,
            )
        return FramePosterior(
            wav_path=f"{phone}.wav",
            times_ms=times,
            class_probs=class_probs,
            event_scores=event_scores,
            acoustic_scores=acoustic_scores,
        )

    nasal_slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone="n", role="vc", event_label="phone_change")
        for idx in range(9)
    ]
    nasal = assign_slots_viterbi(
        posterior_for("n", distractor_center=1460.0),
        expected_slots=nasal_slots,
        min_event_score=0.03,
        language="korean",
    )

    assert nasal.ok
    slot_period = 3200.0 / 9.0
    assert nasal.assignments[3].expected_time_ms == pytest.approx(1200.0 - slot_period * 0.45, abs=0.1)
    assert nasal.assignments[3].selected_time_ms == pytest.approx(1020.0, abs=12.0)

    liquid_slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone="r", role="vc", event_label="phone_change")
        for idx in range(9)
    ]
    liquid = assign_slots_viterbi(
        posterior_for("r", distractor_center=1010.0),
        expected_slots=liquid_slots,
        min_event_score=0.03,
        language="korean",
    )

    assert liquid.ok
    assert liquid.assignments[3].expected_time_ms == pytest.approx(1200.0 + slot_period * 0.55, abs=0.1)
    assert liquid.assignments[3].selected_time_ms == pytest.approx(1420.0, abs=12.0)


def test_korean_obstruent_vc_slots_use_phone_specific_expected_phase(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_KR_OBSTRUENT_VC_EXPECTED_HARD_WINDOW_MS", "90")
    times = [float(idx * 10) for idx in range(321)]
    slot_period = 3200.0 / 9.0

    def posterior_for(phone: str, *, ratio: float, distractor_center: float) -> FramePosterior:
        event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
        class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
        acoustic_scores = {
            "transition_likelihood": [0.04 for _ in times],
            "flux_likelihood": [0.04 for _ in times],
            "voicing": [0.20 for _ in times],
            "silence_likelihood": [0.08 for _ in times],
        }
        for slot_idx in range(9):
            center = max(0.0, slot_idx * 400.0 + slot_period * ratio)
            for frame_idx, time_ms in enumerate(times):
                peak = math.exp(-0.5 * ((time_ms - center) / 8.0) ** 2)
                event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], 0.34 * peak)
                acoustic_scores["transition_likelihood"][frame_idx] = max(
                    acoustic_scores["transition_likelihood"][frame_idx],
                    0.74 * peak,
                )
                acoustic_scores["flux_likelihood"][frame_idx] = max(
                    acoustic_scores["flux_likelihood"][frame_idx],
                    0.70 * peak,
                )
        for frame_idx, time_ms in enumerate(times):
            distractor_peak = math.exp(-0.5 * ((time_ms - distractor_center) / 8.0) ** 2)
            event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], 0.96 * distractor_peak)
            acoustic_scores["transition_likelihood"][frame_idx] = max(
                acoustic_scores["transition_likelihood"][frame_idx],
                0.95 * distractor_peak,
            )
        return FramePosterior(
            wav_path=f"{phone}.wav",
            times_ms=times,
            class_probs=class_probs,
            event_scores=event_scores,
            acoustic_scores=acoustic_scores,
        )

    d_slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone="d", role="vc", event_label="phone_change")
        for idx in range(9)
    ]
    d_result = assign_slots_viterbi(
        posterior_for("d", ratio=0.74, distractor_center=1320.0),
        expected_slots=d_slots,
        min_event_score=0.03,
        language="korean",
    )

    assert d_result.ok
    assert d_result.assignments[3].expected_time_ms == pytest.approx(1200.0 + slot_period * 0.74, abs=0.1)
    assert d_result.assignments[3].selected_time_ms == pytest.approx(1460.0, abs=12.0)

    p_slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone="p", role="vc", event_label="phone_change")
        for idx in range(9)
    ]
    p_result = assign_slots_viterbi(
        posterior_for("p", ratio=-0.15, distractor_center=1320.0),
        expected_slots=p_slots,
        min_event_score=0.03,
        language="korean",
    )

    assert p_result.ok
    assert p_result.assignments[3].expected_time_ms == pytest.approx(1200.0 - slot_period * 0.15, abs=0.1)
    assert p_result.assignments[3].selected_time_ms == pytest.approx(1150.0, abs=12.0)


def test_slot_viterbi_overlap_segment_does_not_gap_against_same_slot(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_KR_OBSTRUENT_VC_EXPECTED_HARD_WINDOW_MS", "90")
    times = [float(idx * 10) for idx in range(451)]
    duration_ms = float(times[-1])
    slot_count = 11
    slot_period = duration_ms / float(slot_count)
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.04 for _ in times],
        "flux_likelihood": [0.04 for _ in times],
        "voicing": [0.10 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
    }
    for slot_idx in range(slot_count):
        center = max(0.0, (duration_ms * (slot_idx / float(slot_count - 1))) - (slot_period * 0.15))
        for frame_idx, time_ms in enumerate(times):
            peak = math.exp(-0.5 * ((time_ms - center) / 8.0) ** 2)
            event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], 0.82 * peak)
            acoustic_scores["transition_likelihood"][frame_idx] = max(
                acoustic_scores["transition_likelihood"][frame_idx],
                0.78 * peak,
            )
            acoustic_scores["flux_likelihood"][frame_idx] = max(
                acoustic_scores["flux_likelihood"][frame_idx],
                0.72 * peak,
            )
            class_probs["consonant"][frame_idx] = max(class_probs["consonant"][frame_idx], 0.76 * peak)
    posterior = FramePosterior(
        wav_path="dense_p_vc.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [
        ExpectedSlot(slot_index=idx, phone_index=idx, phone="p", role="vc", event_label="phone_change")
        for idx in range(slot_count)
    ]

    result = assign_slots_viterbi(
        posterior,
        expected_slots=slots,
        min_event_score=0.03,
        language="korean",
        long_row_segment_size=10,
        segment_overlap_slots=1,
    )

    assert result.ok
    assert not any(warning.startswith("hard_missing_candidates") for warning in result.warnings)
    assert len(result.assignments) == slot_count


def test_korean_cvc_uppercase_release_cluster_uses_compact_manual_style_profile():
    rows = [
        AdaptedOtoRow(
            wav="eunR_eumR_eungR_eulR.wav",
            alias=alias,
            timing=OtoTiming(anchor - pre, consonant, cutoff, pre, overlap),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=anchor,
                score=0.7,
                role=role,
                expected_phone="r",
                vowel_start_abs_ms=vowel_start,
                expected_phone_index=target,
                slot_index=index,
            ),
            mode="template-bootstrap",
        )
        for index, (alias, anchor, pre, overlap, consonant, cutoff, role, target, vowel_start) in enumerate(
            (
                ("n R", 930.0, 80.0, 45.0, 120.0, -154.0, "vc", 2, 920.0),
                ("m R", 2230.0, 120.0, 85.0, 160.0, -250.0, "vcv", 5, 1980.0),
                ("ng R", 3000.0, 120.0, 85.0, 160.0, -490.0, "vcv", 8, 2980.0),
                ("l R", 4010.0, 120.0, 85.0, 160.0, -460.0, "vcv", 11, 3990.0),
            )
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert [row.timing.offset + row.timing.preutterance for row in repaired] == pytest.approx(
        [1270.0, 2270.0, 3270.0, 4270.0],
        abs=0.01,
    )
    assert [row.timing.preutterance for row in repaired] == pytest.approx([55.0, 56.0, 45.0, 58.0], abs=0.01)
    assert [row.timing.overlap for row in repaired] == pytest.approx([28.0, 26.0, 22.0, 30.0], abs=0.01)
    assert all("korean_cvc_release_cluster_repair" in row.applied_rules for row in repaired)

    lower_case_regular_vc = [
        AdaptedOtoRow(
            wav="ra.wav",
            alias="a r",
            timing=OtoTiming(1000.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1080.0, role="vc", expected_phone="r", vowel_start_abs_ms=900.0),
            mode="template-bootstrap",
        )
    ]
    assert repair_cvvc_row_sequence(
        lower_case_regular_vc,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=2000.0,
    ) == lower_case_regular_vc


def test_korean_cvc_obstruent_vc_late_rows_use_previous_cv_profile():
    rows = [
        AdaptedOtoRow(
            wav="ssa_ssi.wav",
            alias="ssa",
            timing=OtoTiming(870.0, 180.0, -360.0, 120.0, 80.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=990.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa_ssi.wav",
            alias="a ss",
            timing=OtoTiming(1320.0, 180.0, -320.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1440.0, role="vc", expected_phone="ss"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa_ssi.wav",
            alias="ssi",
            timing=OtoTiming(1370.0, 180.0, -360.0, 120.0, 80.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1490.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa_ssi.wav",
            alias="i t",
            timing=OtoTiming(1940.0, 180.0, -320.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2060.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="other.wav",
            alias="a ch",
            timing=OtoTiming(2300.0, 180.0, -320.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2420.0, role="vc", expected_phone="ch"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert repaired[1].timing == OtoTiming(1145.0, 170.0, -250.0, 130.0, 55.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1275.0)
    assert "korean_cvc_obstruent_vc_previous_cv_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(1690.0, 155.0, -215.0, 125.0, 48.0)
    assert "korean_cvc_obstruent_vc_previous_cv_repair" in repaired[3].applied_rules
    assert repaired[4] == rows[4]


def test_korean_cvc_spaced_t_vc_after_profile_resyncs_mid_vowels_only():
    rows = [
        AdaptedOtoRow(
            wav="ta.wav",
            alias="ti",
            timing=OtoTiming(1440.0, 160.0, -490.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="i t",
            timing=OtoTiming(1600.0, 155.0, -215.0, 125.0, 48.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1725.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="te",
            timing=OtoTiming(2440.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2500.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="e t",
            timing=OtoTiming(2600.0, 155.0, -215.0, 125.0, 48.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2725.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="to",
            timing=OtoTiming(2940.0, 160.0, -480.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="o t",
            timing=OtoTiming(3100.0, 155.0, -215.0, 125.0, 48.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3225.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="teu",
            timing=OtoTiming(3440.0, 160.0, -500.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3500.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="eu t",
            timing=OtoTiming(3600.0, 155.0, -215.0, 125.0, 48.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3725.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1760.0, 155.0, -215.0, 125.0, 48.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1885.0)
    assert "korean_cvc_spaced_t_vc_after_profile_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(2760.0, 155.0, -215.0, 125.0, 48.0)
    assert "korean_cvc_spaced_t_vc_after_profile_repair" in repaired[3].applied_rules
    assert "korean_cvc_spaced_t_vc_after_profile_repair" not in repaired[5].applied_rules
    assert repaired[5].timing == rows[5].timing
    assert repaired[7].timing == OtoTiming(3760.0, 155.0, -215.0, 125.0, 48.0)
    assert "korean_cvc_spaced_t_vc_after_profile_repair" in repaired[7].applied_rules


def test_korean_cvc_internal_obstruent_vc_uses_neighbor_midpoint():
    rows = [
        AdaptedOtoRow(
            wav="bba.wav",
            alias="bbi",
            timing=OtoTiming(1710.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1830.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bba.wav",
            alias="ip",
            timing=OtoTiming(1940.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2020.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bba.wav",
            alias="bbu",
            timing=OtoTiming(1910.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2030.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ddo",
            timing=OtoTiming(2880.0, 160.0, -450.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ot",
            timing=OtoTiming(3420.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3500.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ddeu",
            timing=OtoTiming(3550.0, 160.0, -450.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3670.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="ke",
            timing=OtoTiming(2390.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2510.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="e k",
            timing=OtoTiming(2910.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2990.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="ko",
            timing=OtoTiming(2940.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3060.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="pe",
            timing=OtoTiming(2370.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2490.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="e p",
            timing=OtoTiming(2550.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2630.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="po",
            timing=OtoTiming(2580.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2700.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1810.0, 110.0, -150.0, 70.0, 38.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1880.0)
    assert "korean_cvc_internal_obstruent_vc_midpoint_repair" in repaired[1].applied_rules
    assert repaired[4].timing == OtoTiming(3215.0, 145.0, -195.0, 115.0, 45.0)
    assert repaired[4].anchor is not None
    assert repaired[4].anchor.anchor_abs_ms == pytest.approx(3330.0)
    assert "korean_cvc_internal_obstruent_vc_midpoint_repair" in repaired[4].applied_rules
    assert repaired[7].timing == OtoTiming(2665.0, 150.0, -195.0, 115.0, 60.0)
    assert repaired[7].anchor is not None
    assert repaired[7].anchor.anchor_abs_ms == pytest.approx(2780.0)
    assert "korean_cvc_internal_obstruent_vc_midpoint_repair" in repaired[7].applied_rules
    assert repaired[10] == rows[10]


def test_korean_cvc_default_cv_midpoint_repairs_late_true_cv_rows():
    rows = [
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bi",
            timing=OtoTiming(1440.0, 160.0, -550.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="i b",
            timing=OtoTiming(1840.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1920.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bu",
            timing=OtoTiming(2030.0, 160.0, -400.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="u b",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="be",
            timing=OtoTiming(2450.0, 160.0, -430.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2510.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ng.wav",
            alias="ngu",
            timing=OtoTiming(1790.0, 160.0, -480.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1850.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ng.wav",
            alias="nge",
            timing=OtoTiming(2400.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2520.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ng.wav",
            alias="ngo",
            timing=OtoTiming(2820.0, 160.0, -480.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2880.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired[2].timing == OtoTiming(1945.0, 160.0, -250.0, 60.0, 25.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(2005.0)
    assert "korean_cvc_default_cv_midpoint_repair" in repaired[2].applied_rules
    assert "korean_cvc_plain_obstruent_cv_cutoff_profile_repair" in repaired[2].applied_rules
    assert repaired[6] == rows[6]


def test_korean_cvc_default_cv_midpoint_ignores_no_space_vc_neighbors():
    rows = [
        AdaptedOtoRow(
            wav="dde.wav",
            alias="ddu",
            timing=OtoTiming(1930.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1990.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dde.wav",
            alias="ut",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dde.wav",
            alias="dde",
            timing=OtoTiming(2520.0, 160.0, -320.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2640.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dde.wav",
            alias="et",
            timing=OtoTiming(2700.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2780.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dde.wav",
            alias="ddo",
            timing=OtoTiming(2930.0, 160.0, -430.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2990.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired[2].timing == OtoTiming(2430.0, 160.0, -190.0, 60.0, 25.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(2490.0)
    assert "korean_cvc_default_cv_midpoint_repair" in repaired[2].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[2].applied_rules
    assert repaired[1].alias == "ut"
    assert repaired[3].alias == "et"


def test_korean_cvc_sonorant_cv_profile_repairs_later_true_cv_rows():
    rows = [
        AdaptedOtoRow(
            wav="mam.wav",
            alias="ma",
            timing=OtoTiming(1000.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="a m",
            timing=OtoTiming(1240.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1320.0, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="mi",
            timing=OtoTiming(1500.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1620.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="i m",
            timing=OtoTiming(1740.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1820.0, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="mu",
            timing=OtoTiming(2000.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2120.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="u m",
            timing=OtoTiming(2240.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2320.0, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="me",
            timing=OtoTiming(2500.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2620.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired[0] == rows[0]
    assert repaired[2] == rows[2]
    assert repaired[4].timing == OtoTiming(2060.0, 160.0, -600.0, 60.0, 25.0)
    assert repaired[4].anchor is not None
    assert repaired[4].anchor.anchor_abs_ms == pytest.approx(2120.0)
    assert "korean_cvc_sonorant_cv_profile_repair" in repaired[4].applied_rules
    assert repaired[6].timing == OtoTiming(2560.0, 160.0, -600.0, 60.0, 25.0)
    assert repaired[6].anchor is not None
    assert repaired[6].anchor.anchor_abs_ms == pytest.approx(2620.0)
    assert "korean_cvc_sonorant_cv_profile_repair" in repaired[6].applied_rules


def test_korean_cvc_sonorant_cv_profile_skips_l_and_early_yoon_rows():
    rows = [
        AdaptedOtoRow(
            wav="mya.wav",
            alias="mya",
            timing=OtoTiming(1000.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mya.wav",
            alias="mye",
            timing=OtoTiming(1500.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1620.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="la.wav",
            alias="la",
            timing=OtoTiming(1000.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="la.wav",
            alias="li",
            timing=OtoTiming(1500.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1620.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="la.wav",
            alias="lu",
            timing=OtoTiming(2000.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2120.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired == rows


def test_korean_cvc_liquid_yoon_cv_profile_repairs_midpoint_late_row_only():
    rows = [
        AdaptedOtoRow(
            wav="ly.wav",
            alias="lya",
            timing=OtoTiming(1440.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ly.wav",
            alias="lye",
            timing=OtoTiming(2030.0, 160.0, -2370.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ly.wav",
            alias="lyeo",
            timing=OtoTiming(2470.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2530.0, role="cv", expected_phone="yeo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="mya",
            timing=OtoTiming(1440.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="mye",
            timing=OtoTiming(2030.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="myeo",
            timing=OtoTiming(2470.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2530.0, role="cv", expected_phone="yeo"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1895.0, 160.0, -240.0, 60.0, 25.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1955.0)
    assert "korean_cvc_liquid_yoon_cv_profile_repair" in repaired[1].applied_rules
    assert "korean_cvc_liquid_yoon_cv_profile_repair" not in repaired[4].applied_rules
    assert repaired[4].timing == rows[4].timing


def test_korean_cvc_midrow_vc_neighbor_repairs_late_and_early_slots():
    rows = [
        AdaptedOtoRow(
            wav="ga.wav",
            alias="ge",
            timing=OtoTiming(2370.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2490.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="e g",
            timing=OtoTiming(2920.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3000.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="go",
            timing=OtoTiming(2910.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3030.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="su",
            timing=OtoTiming(1810.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1930.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="u s",
            timing=OtoTiming(2360.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2440.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="se",
            timing=OtoTiming(2330.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2450.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="pa",
            timing=OtoTiming(860.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=980.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="a p",
            timing=OtoTiming(1070.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1150.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="pi",
            timing=OtoTiming(1370.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1490.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="da",
            timing=OtoTiming(850.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=970.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="a d",
            timing=OtoTiming(1390.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1470.0, role="vc", expected_phone="d"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="di",
            timing=OtoTiming(1370.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1490.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="id",
            timing=OtoTiming(1770.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1850.0, role="vc", expected_phone="d"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="du",
            timing=OtoTiming(1860.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1980.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="u d",
            timing=OtoTiming(2370.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2450.0, role="vc", expected_phone="d"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="da.wav",
            alias="de",
            timing=OtoTiming(2360.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bi",
            timing=OtoTiming(1380.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="i b",
            timing=OtoTiming(1890.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1970.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bu",
            timing=OtoTiming(2030.0, 160.0, -410.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(2730.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(2810.0)
    assert "korean_cvc_midrow_vc_neighbor_repair" in repaired[1].applied_rules
    assert repaired[4].timing == OtoTiming(2235.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[7].timing == OtoTiming(1200.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[10] == rows[10]
    assert repaired[12] == rows[12]
    assert repaired[14].timing == OtoTiming(2230.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[17].timing == OtoTiming(1840.0, 120.0, -154.0, 80.0, 45.0)


def test_korean_cvc_initial_cv_onset_repairs_late_first_rows():
    rows = [
        AdaptedOtoRow(
            wav="rwa.wav",
            alias="rwa",
            timing=OtoTiming(1150.0, 160.0, -570.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1270.0, role="cv", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="rwa.wav",
            alias="rwe",
            timing=OtoTiming(1480.0, 160.0, -430.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1600.0, role="cv", expected_phone="we"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ngya.wav",
            alias="ngya",
            timing=OtoTiming(1390.0, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1510.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ngya.wav",
            alias="ngye",
            timing=OtoTiming(1850.0, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1970.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bya.wav",
            alias="bya",
            timing=OtoTiming(930.0, 160.0, -490.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1050.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bya.wav",
            alias="bye",
            timing=OtoTiming(1420.0, 160.0, -430.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="middle.wav",
            alias="pa",
            timing=OtoTiming(860.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=980.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="middle.wav",
            alias="sya",
            timing=OtoTiming(1070.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1190.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert repaired[0].timing == OtoTiming(940.0, 180.0, -570.0, 70.0, 35.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(1010.0)
    assert "korean_cvc_initial_cv_onset_repair" in repaired[0].applied_rules
    assert repaired[2] == rows[2]
    assert repaired[4].timing == OtoTiming(930.0, 160.0, -320.0, 120.0, 85.0)
    assert "korean_cvc_head_glide_cv_cutoff_profile_repair" in repaired[4].applied_rules
    assert repaired[7].timing == OtoTiming(1070.0, 160.0, -320.0, 120.0, 85.0)
    assert "korean_cvc_head_glide_cv_cutoff_profile_repair" in repaired[7].applied_rules


def test_korean_cvc_initial_local_gap_repairs_compressed_second_cv():
    rows = [
        AdaptedOtoRow(
            wav="ga.wav",
            alias="ga",
            timing=OtoTiming(940.0, 180.0, -490.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="a g",
            timing=OtoTiming(1140.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1220.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="gi",
            timing=OtoTiming(1180.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1300.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="i g",
            timing=OtoTiming(1740.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1820.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ka",
            timing=OtoTiming(940.0, 180.0, -270.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="a k",
            timing=OtoTiming(1220.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1300.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ki",
            timing=OtoTiming(1420.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="i k",
            timing=OtoTiming(1700.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1780.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert repaired[1].timing == OtoTiming(1340.0, 120.0, -150.0, 110.0, 60.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1450.0)
    assert "korean_cvc_initial_local_gap_repair" in repaired[1].applied_rules
    assert repaired[2].timing == OtoTiming(1490.0, 150.0, -340.0, 70.0, 35.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(1560.0)
    assert "korean_cvc_initial_local_gap_repair" in repaired[2].applied_rules
    assert repaired[4] == rows[4]
    assert repaired[5] == rows[5]
    assert repaired[6] == rows[6]
    assert repaired[7] == rows[7]


def test_korean_cvc_initial_block_collapse_repairs_j_ss_and_n_starts():
    rows = [
        AdaptedOtoRow(
            wav="ja.wav",
            alias="ja",
            timing=OtoTiming(840.0, 180.0, -490.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=910.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ja.wav",
            alias="a j",
            timing=OtoTiming(918.5, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=998.5, role="vc", expected_phone="j"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ja.wav",
            alias="ji",
            timing=OtoTiming(1230.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1350.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ja.wav",
            alias="i j",
            timing=OtoTiming(1770.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1850.0, role="vc", expected_phone="j"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa.wav",
            alias="ssa",
            timing=OtoTiming(940.0, 180.0, -490.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa.wav",
            alias="a ss",
            timing=OtoTiming(1130.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1210.0, role="vc", expected_phone="ss"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa.wav",
            alias="ssi",
            timing=OtoTiming(1130.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1250.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ssa.wav",
            alias="i ss",
            timing=OtoTiming(1405.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1485.0, role="vc", expected_phone="ss"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="na",
            timing=OtoTiming(860.0, 180.0, -490.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=930.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="an",
            timing=OtoTiming(927.5, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1007.5, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="a n",
            timing=OtoTiming(927.5, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1007.5, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="ni",
            timing=OtoTiming(1410.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1530.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="sa",
            timing=OtoTiming(940.0, 180.0, -270.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="a s",
            timing=OtoTiming(1220.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1300.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="si",
            timing=OtoTiming(1420.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="i s",
            timing=OtoTiming(1700.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1780.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert repaired[1].timing == OtoTiming(1290.0, 120.0, -180.0, 80.0, 45.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1370.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[1].applied_rules
    assert repaired[2].timing == OtoTiming(1410.0, 160.0, -360.0, 120.0, 85.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(1530.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[2].applied_rules
    assert repaired[3] == rows[3]

    assert repaired[5].timing == OtoTiming(1280.0, 170.0, -250.0, 130.0, 55.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(1410.0)
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[5].applied_rules
    assert repaired[6].timing == OtoTiming(1370.0, 160.0, -300.0, 120.0, 85.0)
    assert repaired[6].anchor is not None
    assert repaired[6].anchor.anchor_abs_ms == pytest.approx(1490.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[6].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[6].applied_rules
    assert "korean_cvc_residual_ss_doubled_head_short_cutoff_profile_repair" in repaired[6].applied_rules
    assert repaired[7].timing == OtoTiming(1645.0, 170.0, -250.0, 130.0, 55.0)
    assert repaired[7].anchor is not None
    assert repaired[7].anchor.anchor_abs_ms == pytest.approx(1775.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[7].applied_rules

    assert repaired[9].timing == OtoTiming(1080.0, 115.0, -360.0, 80.0, 45.0)
    assert repaired[9].anchor is not None
    assert repaired[9].anchor.anchor_abs_ms == pytest.approx(1160.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[9].applied_rules
    assert "korean_cvc_residual_compact_an_cutoff_profile_repair" in repaired[9].applied_rules
    assert repaired[10].timing == OtoTiming(1080.0, 115.0, -260.0, 80.0, 45.0)
    assert repaired[10].anchor is not None
    assert repaired[10].anchor.anchor_abs_ms == pytest.approx(1160.0)
    assert "korean_cvc_initial_block_collapse_repair" in repaired[10].applied_rules
    assert "korean_cvc_residual_spaced_n_vc_cutoff_profile_repair" in repaired[10].applied_rules
    assert repaired[11] == rows[11]

    assert repaired[12] == rows[12]
    assert repaired[13] == rows[13]
    assert repaired[14] == rows[14]
    assert repaired[15] == rows[15]


def test_korean_cvc_compressed_cv_vc_cv_repairs_late_plain_cv_pairs():
    rows = [
        AdaptedOtoRow(
            wav="be.wav",
            alias="u b",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="be.wav",
            alias="be",
            timing=OtoTiming(2670.0, 160.0, -620.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2790.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="be.wav",
            alias="e b",
            timing=OtoTiming(2740.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2820.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="be.wav",
            alias="bo",
            timing=OtoTiming(2850.0, 160.0, -590.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2970.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sse.wav",
            alias="u ss",
            timing=OtoTiming(2175.0, 170.0, -250.0, 130.0, 55.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2305.0, role="vc", expected_phone="ss"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sse.wav",
            alias="sse",
            timing=OtoTiming(2530.0, 160.0, -470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2650.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sse.wav",
            alias="e ss",
            timing=OtoTiming(2805.0, 170.0, -250.0, 130.0, 55.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2935.0, role="vc", expected_phone="ss"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sse.wav",
            alias="sso",
            timing=OtoTiming(2860.0, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2980.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="te.wav",
            alias="u t",
            timing=OtoTiming(2250.0, 125.0, -215.0, 125.0, 48.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2375.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="te.wav",
            alias="te",
            timing=OtoTiming(2570.0, 160.0, -520.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2690.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="te.wav",
            alias="e t",
            timing=OtoTiming(2918.576, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2998.576, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="te.wav",
            alias="to",
            timing=OtoTiming(2900.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3020.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="a g",
            timing=OtoTiming(1240.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1320.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="gi",
            timing=OtoTiming(1490.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1610.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="i g",
            timing=OtoTiming(1740.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1820.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="gu",
            timing=OtoTiming(1860.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1980.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired[1].timing == OtoTiming(2450.0, 160.0, -210.0, 120.0, 85.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(2570.0)
    assert "korean_cvc_compressed_cv_vc_cv_repair" in repaired[1].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[1].applied_rules
    assert repaired[2] == rows[2]

    assert repaired[5].timing == OtoTiming(2360.0, 160.0, -240.0, 120.0, 85.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(2480.0)
    assert "korean_cvc_compressed_cv_vc_cv_repair" in repaired[5].applied_rules
    assert "korean_cvc_residual_doubled_head_cv_cutoff_profile_repair" in repaired[5].applied_rules
    assert repaired[6].timing == OtoTiming(2575.0, 170.0, -250.0, 130.0, 55.0)
    assert repaired[6].anchor is not None
    assert repaired[6].anchor.anchor_abs_ms == pytest.approx(2705.0)
    assert "korean_cvc_compressed_cv_vc_cv_repair" in repaired[6].applied_rules

    assert repaired[9].timing == OtoTiming(2400.0, 160.0, -220.0, 120.0, 85.0)
    assert repaired[9].anchor is not None
    assert repaired[9].anchor.anchor_abs_ms == pytest.approx(2520.0)
    assert "korean_cvc_compressed_cv_vc_cv_repair" in repaired[9].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[9].applied_rules
    assert repaired[10].timing == OtoTiming(2700.0, 155.0, -215.0, 125.0, 48.0)
    assert repaired[10].anchor is not None
    assert repaired[10].anchor.anchor_abs_ms == pytest.approx(2825.0)
    assert "korean_cvc_compressed_cv_vc_cv_repair" in repaired[10].applied_rules

    assert repaired[13] == rows[13]
    assert repaired[14] == rows[14]
    assert repaired[15] == rows[15]


def test_korean_cvc_spaced_obstruent_cadence_repairs_late_and_early_vc_rows():
    def row(wav: str, alias: str, offset: float, *, pre: float = 80.0, overlap: float = 45.0) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset, 120.0, -154.0, pre, overlap),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=offset + pre, role="vc", expected_phone=alias.split()[-1]),
            mode="template-bootstrap",
        )

    def cv_row(wav: str, alias: str, offset: float) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=offset + 120.0, role="cv", expected_phone=alias[-1]),
            mode="template-bootstrap",
        )

    rows = [
        cv_row("b.wav", "ba", 840.0),
        row("b.wav", "a b", 1390.0),
        cv_row("b.wav", "bi", 1380.0),
        cv_row("d-early.wav", "da", 940.0),
        row("d-early.wav", "a d", 1200.0),
        cv_row("d-early.wav", "di", 1370.0),
        cv_row("d-late.wav", "du", 1850.0),
        row("d-late.wav", "u d", 2410.0),
        cv_row("d-late.wav", "de", 2460.0),
        cv_row("g.wav", "geo", 4103.309),
        row("g.wav", "eo g", 4170.0),
        cv_row("h.wav", "heu", 3370.0),
        row("h.wav", "eu h", 3880.0),
        cv_row("h.wav", "heo", 3920.0),
        cv_row("k.wav", "ku", 1880.0),
        row("k.wav", "u k", 2090.0),
        cv_row("k.wav", "ke", 2390.0),
        cv_row("ss.wav", "ssi", 1370.0),
        row("ss.wav", "i ss", 1645.0),
        cv_row("ss.wav", "ssu", 1900.0),
        cv_row("safe.wav", "ka", 870.0),
        row("safe.wav", "a k", 1150.0),
        cv_row("safe.wav", "ki", 1380.0),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1280.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1360.0)
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[1].applied_rules

    assert repaired[4].timing == OtoTiming(1320.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[7].timing == OtoTiming(2270.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[4].applied_rules
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[7].applied_rules

    assert repaired[10].timing == OtoTiming(4323.309, 120.0, -154.0, 80.0, 45.0)
    assert repaired[12].timing == OtoTiming(3790.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[10].applied_rules
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[12].applied_rules

    assert repaired[15].timing == OtoTiming(2150.0, 150.0, -195.0, 115.0, 60.0)
    assert repaired[18].timing == OtoTiming(1710.0, 170.0, -250.0, 130.0, 55.0)
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[15].applied_rules
    assert "korean_cvc_spaced_obstruent_cadence_repair" in repaired[18].applied_rules

    assert repaired[21] == rows[21]


def test_korean_cvc_late_plain_cv_suffix_uses_previous_cv_anchor():
    rows = [
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bo",
            timing=OtoTiming(2850.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2970.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="o b",
            timing=OtoTiming(3430.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3510.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="beu",
            timing=OtoTiming(3730.0, 160.0, -510.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3850.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="eu b",
            timing=OtoTiming(3910.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3990.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="beo",
            timing=OtoTiming(4240.0, 160.0, -580.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4360.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="eo b",
            timing=OtoTiming(4410.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4490.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ko",
            timing=OtoTiming(2850.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2970.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="o k",
            timing=OtoTiming(3340.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3420.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="keu",
            timing=OtoTiming(3530.0, 160.0, -510.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3650.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="eu k",
            timing=OtoTiming(3760.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3840.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sonorant.wav",
            alias="lya",
            timing=OtoTiming(1440.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1560.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sonorant.wav",
            alias="al",
            timing=OtoTiming(1900.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1980.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sonorant.wav",
            alias="lye",
            timing=OtoTiming(2030.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sonorant.wav",
            alias="el",
            timing=OtoTiming(2210.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2290.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[2].timing == OtoTiming(3470.0, 160.0, -300.0, 70.0, 35.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(3540.0)
    assert "korean_cvc_late_plain_cv_suffix_repair" in repaired[2].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[2].applied_rules
    assert repaired[4].timing == OtoTiming(4040.0, 160.0, -250.0, 70.0, 35.0)
    assert repaired[4].anchor is not None
    assert repaired[4].anchor.anchor_abs_ms == pytest.approx(4110.0)
    assert "korean_cvc_late_plain_cv_suffix_repair" in repaired[4].applied_rules
    assert "korean_cvc_plain_obstruent_cv_cutoff_profile_repair" in repaired[4].applied_rules
    assert repaired[8].timing == OtoTiming(3590.0, 160.0, -250.0, 60.0, 25.0)
    assert repaired[8].anchor is not None
    assert repaired[8].anchor.anchor_abs_ms == pytest.approx(3650.0)
    assert "korean_cvc_late_plain_cv_suffix_repair" not in repaired[8].applied_rules
    assert "korean_cvc_alternating_cv_profile_repair" in repaired[8].applied_rules
    assert "korean_cvc_plain_obstruent_cv_cutoff_profile_repair" in repaired[8].applied_rules
    assert repaired[12] == rows[12]


def test_korean_cvc_initial_nasal_ng_pair_repairs_compressed_start():
    rows = [
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ang",
            timing=OtoTiming(920.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1000.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="nga",
            timing=OtoTiming(4307.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4427.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ngi",
            timing=OtoTiming(1279.5, 160.0, -3580.5, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1399.5, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ing",
            timing=OtoTiming(1654.8, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1734.8, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="normal.wav",
            alias="ang",
            timing=OtoTiming(1160.0, 90.0, -180.0, 85.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1245.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="normal.wav",
            alias="ngi",
            timing=OtoTiming(1420.0, 135.0, -205.0, 50.0, 20.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1470.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="normal.wav",
            alias="ing",
            timing=OtoTiming(1718.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1798.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[0].timing == OtoTiming(1160.0, 93.0, -180.0, 85.0, 35.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(1245.0)
    assert "korean_cvc_initial_nasal_ng_pair_repair" in repaired[0].applied_rules
    assert repaired[1] == rows[1]
    assert repaired[2].timing == OtoTiming(1420.0, 135.0, -205.0, 50.0, 20.0)
    assert repaired[2].anchor is not None
    assert repaired[2].anchor.anchor_abs_ms == pytest.approx(1470.0)
    assert "korean_cvc_initial_nasal_ng_pair_repair" in repaired[2].applied_rules
    assert repaired[4].timing == OtoTiming(1160.0, 93.0, -180.0, 85.0, 35.0)
    assert "oto_parameter_order_repair" in repaired[4].applied_rules
    assert repaired[5] == rows[5]


def test_korean_cvc_pure_v_glide_sequence_repairs_compressed_rows():
    rows = [
        AdaptedOtoRow(
            wav="ya.wav",
            alias="ya",
            timing=OtoTiming(890.0, 160.0, -980.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="v", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ya.wav",
            alias="ye",
            timing=OtoTiming(1210.0, 160.0, -810.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1330.0, role="v", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ya.wav",
            alias="yeo",
            timing=OtoTiming(1920.0, 160.0, -510.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2040.0, role="v", expected_phone="yeo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="wa",
            timing=OtoTiming(940.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1060.0, role="v", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="we",
            timing=OtoTiming(1380.0, 160.0, -460.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="v", expected_phone="we"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="wi",
            timing=OtoTiming(1830.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1950.0, role="v", expected_phone="wi"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="weo",
            timing=OtoTiming(2230.0, 160.0, -1470.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2350.0, role="v", expected_phone="weo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ya",
            timing=OtoTiming(890.0, 160.0, -980.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="v", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ye",
            timing=OtoTiming(1410.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1530.0, role="v", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="yeo",
            timing=OtoTiming(1920.0, 160.0, -510.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2040.0, role="v", expected_phone="yeo"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1420.0, 180.0, -285.0, 60.0, 30.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1480.0)
    assert "korean_cvc_pure_v_glide_sequence_repair" in repaired[1].applied_rules
    assert repaired[6].timing == OtoTiming(2400.0, 216.0, -287.0, 45.0, 18.0)
    assert repaired[6].anchor is not None
    assert repaired[6].anchor.anchor_abs_ms == pytest.approx(2445.0)
    assert "korean_cvc_pure_v_glide_sequence_repair" in repaired[6].applied_rules
    assert repaired[8] == rows[8]


def test_korean_cvc_standalone_vowel_start_profile_repairs_rejected_boundary_row():
    rows = [
        AdaptedOtoRow(
            wav="ha.wav",
            alias="i",
            timing=OtoTiming(1409.469, 422.462, -560.531, 319.076, 161.724),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1728.545,
                role="v",
                expected_phone="i",
                vowel_start_abs_ms=1510.0,
                vowel_end_abs_ms=1920.0,
                vowel_nucleus_abs_ms=1728.545,
            ),
            warnings=("local_refine_rejected_slot_boundary",),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ha.wav",
            alias="u",
            timing=OtoTiming(1961.6, 263.6, -488.4, 58.4, 29.6),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=2050.0,
                role="v",
                expected_phone="u",
                vowel_start_abs_ms=1980.0,
                vowel_end_abs_ms=2400.0,
                vowel_nucleus_abs_ms=2020.0,
            ),
            warnings=("local_refine_low_margin",),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[0].timing == OtoTiming(1524.0, 150.0, -300.0, 25.0, 25.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(1549.0)
    assert "korean_cvc_standalone_vowel_start_profile_repair" in repaired[0].applied_rules
    assert repaired[1].timing == OtoTiming(1961.6, 263.6, -320.0, 58.4, 29.6)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(2050.0)
    assert "korean_cvc_residual_standalone_v_cutoff_profile_repair" in repaired[1].applied_rules


def test_korean_cvc_liquid_vc_late_rows_use_previous_cv_offset():
    rows = [
        AdaptedOtoRow(
            wav="ral.wav",
            alias="la",
            timing=OtoTiming(870.0, 160.0, -3890.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=990.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="al",
            timing=OtoTiming(1050.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1130.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="li",
            timing=OtoTiming(1380.0, 160.0, -3380.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="il",
            timing=OtoTiming(1890.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1970.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="lu",
            timing=OtoTiming(1980.0, 160.0, -2780.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2100.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="ul",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="other.wav",
            alias="ul",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="ra",
            timing=OtoTiming(1000.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="a r",
            timing=OtoTiming(1370.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1450.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1] == rows[1]
    assert repaired[3].timing == OtoTiming(1665.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[3].anchor is not None
    assert repaired[3].anchor.anchor_abs_ms == pytest.approx(1745.0)
    assert "korean_cvc_liquid_vc_previous_cv_repair" in repaired[3].applied_rules
    assert "korean_cvc_compact_l_vc_after_profile_repair" in repaired[3].applied_rules
    assert repaired[5].timing == OtoTiming(2050.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(2130.0)
    assert "korean_cvc_liquid_vc_previous_cv_repair" in repaired[5].applied_rules
    assert "korean_cvc_no_space_vc_cadence_repair" in repaired[5].applied_rules
    assert repaired[6] == rows[6]
    assert repaired[7] == rows[7]
    assert repaired[8] == rows[8]


def test_korean_cvc_r_liquid_vc_rows_use_next_cv_cadence():
    rows = [
        AdaptedOtoRow(
            wav="ra.wav",
            alias="ra",
            timing=OtoTiming(950.0, 160.0, -490.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1070.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="a r",
            timing=OtoTiming(1370.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1450.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="ri",
            timing=OtoTiming(1340.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1460.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="i r",
            timing=OtoTiming(1850.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1930.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="ru",
            timing=OtoTiming(1830.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1950.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="u r",
            timing=OtoTiming(2350.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2430.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="re",
            timing=OtoTiming(2350.0, 160.0, -580.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2470.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="e r",
            timing=OtoTiming(2841.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2921.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="ro",
            timing=OtoTiming(2810.0, 160.0, -670.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2930.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="o r",
            timing=OtoTiming(3380.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3460.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="reu",
            timing=OtoTiming(3350.0, 160.0, -640.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3470.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="eu r",
            timing=OtoTiming(3900.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3980.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="reo",
            timing=OtoTiming(3910.0, 160.0, -570.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4030.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ra.wav",
            alias="eo r",
            timing=OtoTiming(4400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4480.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ra",
            timing=OtoTiming(950.0, 160.0, -490.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1070.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="a r",
            timing=OtoTiming(1270.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1350.0, role="vc", expected_phone="r"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="ri",
            timing=OtoTiming(1340.0, 160.0, -600.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1460.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1270.0, 145.0, -160.0, 135.0, 50.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1405.0)
    assert "korean_cvc_r_liquid_vc_next_cv_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(1780.0, 128.0, -140.0, 120.0, 60.0)
    assert repaired[5].timing == OtoTiming(2290.0, 126.0, -140.0, 118.0, 40.0)
    assert repaired[7].timing == OtoTiming(2790.0, 116.0, -135.0, 108.0, 60.0)
    assert repaired[9].timing == OtoTiming(3275.0, 163.0, -185.0, 155.0, 67.0)
    assert repaired[11].timing == OtoTiming(3865.0, 96.0, -110.0, 88.0, 20.0)
    assert repaired[13].timing == OtoTiming(4310.0, 150.0, -165.0, 142.0, 36.0)
    assert repaired[15] == rows[15]


def test_korean_cvc_terminal_cv_before_spaced_vc_repairs_late_geo_only():
    rows = [
        AdaptedOtoRow(
            wav="ga.wav",
            alias="geu",
            timing=OtoTiming(3560.0, 160.0, -380.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3680.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="geo",
            timing=OtoTiming(4103.309, 160.0, -626.691, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4223.309, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ga.wav",
            alias="eo g",
            timing=OtoTiming(4323.309, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4403.309, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="seo",
            timing=OtoTiming(4030.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4150.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="eo s",
            timing=OtoTiming(4220.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4300.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggeo",
            timing=OtoTiming(3920.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4040.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="eok",
            timing=OtoTiming(4170.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4250.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing.offset == pytest.approx(3933.309)
    assert repaired[1].timing.consonant == pytest.approx(160.0)
    assert repaired[1].timing.cutoff == pytest.approx(-380.0)
    assert repaired[1].timing.preutterance == pytest.approx(40.0)
    assert repaired[1].timing.overlap == pytest.approx(20.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(3973.309)
    assert "korean_cvc_terminal_cv_before_spaced_vc_repair" in repaired[1].applied_rules
    assert repaired[3] == rows[3]
    assert repaired[5].timing == OtoTiming(3920.0, 160.0, -240.0, 120.0, 85.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(4040.0)
    assert "korean_cvc_residual_doubled_head_cv_cutoff_profile_repair" in repaired[5].applied_rules


def test_korean_cvc_internal_i_k_midpoint_repairs_late_ik_only():
    rows = [
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggi",
            timing=OtoTiming(1380.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ik",
            timing=OtoTiming(1860.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1940.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggu",
            timing=OtoTiming(2000.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2120.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gu.wav",
            alias="ggu",
            timing=OtoTiming(2000.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2120.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gu.wav",
            alias="uk",
            timing=OtoTiming(2280.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2360.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gu.wav",
            alias="gge",
            timing=OtoTiming(2430.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2550.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bi.wav",
            alias="bbi",
            timing=OtoTiming(1610.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1730.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bi.wav",
            alias="ip",
            timing=OtoTiming(1810.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1890.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bi.wav",
            alias="bbu",
            timing=OtoTiming(1910.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2030.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1720.0, 142.0, -170.0, 118.0, 52.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1838.0)
    assert "korean_cvc_internal_i_k_midpoint_repair" in repaired[1].applied_rules
    assert "korean_cvc_internal_i_k_midpoint_repair" not in repaired[4].applied_rules
    assert repaired[7] == rows[7]


def test_korean_cvc_compact_k_vc_after_profile_resyncs_ik_and_uk_only():
    rows = [
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggi",
            timing=OtoTiming(1440.0, 160.0, -480.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ik",
            timing=OtoTiming(1610.0, 135.0, -175.0, 105.0, 50.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1715.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggu",
            timing=OtoTiming(1900.0, 160.0, -520.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1960.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="uk",
            timing=OtoTiming(2135.0, 135.0, -175.0, 105.0, 50.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2240.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="gge",
            timing=OtoTiming(2430.0, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2550.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="ki",
            timing=OtoTiming(1440.0, 160.0, -470.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="i k",
            timing=OtoTiming(1630.0, 150.0, -195.0, 115.0, 60.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1745.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="ku",
            timing=OtoTiming(1940.0, 160.0, -470.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2000.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1720.0, 142.0, -170.0, 118.0, 52.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1838.0)
    assert "korean_cvc_compact_k_vc_after_profile_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(2245.0, 129.0, -158.0, 80.0, 49.0)
    assert repaired[3].anchor is not None
    assert repaired[3].anchor.anchor_abs_ms == pytest.approx(2325.0)
    assert "korean_cvc_compact_k_vc_after_profile_repair" in repaired[3].applied_rules
    assert "korean_cvc_compact_k_vc_after_profile_repair" not in repaired[6].applied_rules


def test_korean_cvc_nasal_ng_vc_rows_use_neighbor_cv_cadence():
    rows = [
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ngi",
            timing=OtoTiming(1000.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ing",
            timing=OtoTiming(1130.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1210.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ngu",
            timing=OtoTiming(1500.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1620.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ung",
            timing=OtoTiming(1580.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1660.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="nge",
            timing=OtoTiming(2020.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2140.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="eng",
            timing=OtoTiming(2300.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2380.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ngeu",
            timing=OtoTiming(2880.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="eung",
            timing=OtoTiming(2980.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3060.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="ngeo",
            timing=OtoTiming(3380.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3500.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang.wav",
            alias="eong",
            timing=OtoTiming(3900.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3980.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="spaced.wav",
            alias="nga",
            timing=OtoTiming(1000.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="spaced.wav",
            alias="a ng",
            timing=OtoTiming(1400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1480.0, role="vc", expected_phone="ng"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1370.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1450.0)
    assert "korean_cvc_nasal_ng_vc_neighbor_cv_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(1880.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_nasal_ng_vc_neighbor_cv_repair" in repaired[3].applied_rules
    assert repaired[5].timing == OtoTiming(2460.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_nasal_ng_vc_neighbor_cv_repair" in repaired[5].applied_rules
    assert repaired[7].timing == OtoTiming(3060.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_nasal_ng_vc_neighbor_cv_repair" in repaired[7].applied_rules
    assert repaired[9].timing == OtoTiming(3560.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_nasal_ng_vc_neighbor_cv_repair" in repaired[9].applied_rules
    assert repaired[10] == rows[10]
    assert repaired[11] == rows[11]


def test_korean_cvc_nasal_vc_pair_cadence_repairs_duplicate_n_m_rows():
    rows = [
        AdaptedOtoRow(
            wav="mam.wav",
            alias="ma",
            timing=OtoTiming(870.0, 160.0, -3890.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=990.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="am",
            timing=OtoTiming(979.4, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1059.4, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="a m",
            timing=OtoTiming(979.4, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1059.4, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="mi",
            timing=OtoTiming(1350.0, 160.0, -3340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1470.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mam.wav",
            alias="im",
            timing=OtoTiming(1451.3, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1531.3, role="vc", expected_phone="m"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="na",
            timing=OtoTiming(860.0, 160.0, -3890.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=980.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="an",
            timing=OtoTiming(1080.0, 115.0, -175.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1160.0, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="a n",
            timing=OtoTiming(1080.0, 115.0, -175.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1160.0, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="ni",
            timing=OtoTiming(1410.0, 160.0, -3340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1530.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="in",
            timing=OtoTiming(1477.8, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1557.8, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="i n",
            timing=OtoTiming(1477.8, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1557.8, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="nu",
            timing=OtoTiming(1840.0, 160.0, -2910.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1960.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="un",
            timing=OtoTiming(1961.8, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2041.8, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="neu",
            timing=OtoTiming(3350.0, 160.0, -1400.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3470.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="eun",
            timing=OtoTiming(3689.7, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3769.7, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="eu n",
            timing=OtoTiming(3689.7, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3769.7, role="vc", expected_phone="n"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="nan.wav",
            alias="neo",
            timing=OtoTiming(3870.0, 160.0, -880.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3990.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1070.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[2].timing == OtoTiming(1070.0, 120.0, -280.0, 80.0, 45.0)
    assert "korean_cvc_nasal_vc_pair_cadence_repair" in repaired[1].applied_rules
    assert "korean_cvc_nasal_vc_pair_cadence_repair" in repaired[2].applied_rules
    assert "korean_cvc_residual_spaced_m_vc_cutoff_profile_repair" in repaired[2].applied_rules
    assert repaired[4] == rows[4]
    assert repaired[9].timing == OtoTiming(1600.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[10].timing == OtoTiming(1600.0, 120.0, -260.0, 80.0, 45.0)
    assert "korean_cvc_nasal_vc_pair_cadence_repair" in repaired[9].applied_rules
    assert "korean_cvc_residual_spaced_n_vc_cutoff_profile_repair" in repaired[10].applied_rules
    assert repaired[12] == rows[12]
    assert repaired[14].timing == OtoTiming(3530.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[15].timing == OtoTiming(3530.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_residual_spaced_n_vc_cutoff_profile_repair" not in repaired[15].applied_rules


def test_korean_cvc_terminal_obstruent_vc_uses_terminal_profile():
    rows = [
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ddeo",
            timing=OtoTiming(4060.0, 160.0, -250.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="eot",
            timing=OtoTiming(4460.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4540.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="ggeo",
            timing=OtoTiming(3920.0, 160.0, -390.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4040.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gga.wav",
            alias="eok",
            timing=OtoTiming(4380.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4460.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bba.wav",
            alias="bbeo",
            timing=OtoTiming(4060.0, 160.0, -280.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bba.wav",
            alias="eop",
            timing=OtoTiming(4122.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4202.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="spaced.wav",
            alias="bbeo",
            timing=OtoTiming(4060.0, 160.0, -280.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="spaced.wav",
            alias="eo p",
            timing=OtoTiming(4122.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4202.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(4210.0, 185.0, -245.0, 150.0, 65.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(4360.0)
    assert "korean_cvc_terminal_obstruent_vc_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(4170.0, 105.0, -160.0, 70.0, 38.0)
    assert repaired[3].anchor is not None
    assert repaired[3].anchor.anchor_abs_ms == pytest.approx(4240.0)
    assert "korean_cvc_terminal_obstruent_vc_repair" in repaired[3].applied_rules
    assert repaired[5].timing == OtoTiming(4240.0, 95.0, -150.0, 70.0, 38.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(4310.0)
    assert "korean_cvc_terminal_obstruent_vc_repair" in repaired[5].applied_rules
    assert repaired[6] == rows[6]
    assert repaired[7] == rows[7]


def test_korean_cvc_terminal_spaced_obstruent_vc_uses_terminal_profile():
    rows = [
        AdaptedOtoRow(
            wav="pa.wav",
            alias="peo",
            timing=OtoTiming(3880.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4000.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="pa.wav",
            alias="eo p",
            timing=OtoTiming(4390.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4470.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="keo",
            timing=OtoTiming(3930.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4050.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ka.wav",
            alias="eo k",
            timing=OtoTiming(4410.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4490.0, role="vc", expected_phone="k"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="compact.wav",
            alias="bbeo",
            timing=OtoTiming(4060.0, 160.0, -280.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="compact.wav",
            alias="eop",
            timing=OtoTiming(4240.0, 95.0, -150.0, 70.0, 38.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4310.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="peo",
            timing=OtoTiming(3880.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4000.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="eo p",
            timing=OtoTiming(4200.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4280.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(4140.0, 150.0, -215.0, 120.0, 90.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(4260.0)
    assert "korean_cvc_terminal_spaced_obstruent_vc_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(4220.0, 125.0, -175.0, 82.0, 30.0)
    assert repaired[3].anchor is not None
    assert repaired[3].anchor.anchor_abs_ms == pytest.approx(4302.0)
    assert "korean_cvc_terminal_spaced_obstruent_vc_repair" in repaired[3].applied_rules
    assert "korean_cvc_terminal_spaced_obstruent_vc_repair" not in repaired[5].applied_rules
    assert repaired[7] == rows[7]


def test_korean_cvc_terminal_spaced_s_vc_uses_terminal_profile():
    rows = [
        AdaptedOtoRow(
            wav="sa.wav",
            alias="seo",
            timing=OtoTiming(3970.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4090.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sa.wav",
            alias="eo s",
            timing=OtoTiming(4420.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4500.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="seo",
            timing=OtoTiming(3970.0, 160.0, -360.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4090.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="safe.wav",
            alias="eo s",
            timing=OtoTiming(4260.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4340.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(4295.0, 165.0, -190.0, 150.0, 50.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(4445.0)
    assert "korean_cvc_terminal_spaced_obstruent_vc_repair" in repaired[1].applied_rules
    assert repaired[3] == rows[3]


def test_korean_cvc_cv_cutoff_caps_only_when_both_cutoff_modes_intrude_next_row():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ra.wav",
                alias="ro",
                timing=OtoTiming(2874.365, 160.0, -1185.635, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2934.365, role="cv", expected_phone="o"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ra.wav",
                alias="o r",
                timing=OtoTiming(3275.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3355.0, role="vc", expected_phone="r"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5087.7,
    )

    assert repaired[0].timing.offset == pytest.approx(2874.365)
    assert repaired[0].timing.consonant == pytest.approx(160.0)
    assert repaired[0].timing.cutoff == pytest.approx(-320.635)
    assert repaired[0].timing.preutterance == pytest.approx(60.0)
    assert repaired[0].timing.overlap == pytest.approx(25.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(2934.365)
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[0].applied_rules

    rel_only = [
        AdaptedOtoRow(
            wav="wa.wav",
            alias="wa",
            timing=OtoTiming(910.0, 160.0, -2760.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1030.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="we",
            timing=OtoTiming(1380.0, 160.0, -620.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
    ]
    skipped_rel_only = repair_cvvc_row_sequence(
        rel_only,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3978.3,
    )

    assert skipped_rel_only[0].timing == rel_only[0].timing
    assert "korean_cvc_cv_cutoff_before_next_row_repair" not in skipped_rel_only[0].applied_rules

    too_short = [
        AdaptedOtoRow(
            wav="short.wav",
            alias="ba",
            timing=OtoTiming(1000.0, 160.0, -1000.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="short.wav",
            alias="bi",
            timing=OtoTiming(1200.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1320.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
    ]
    skipped_too_short = repair_cvvc_row_sequence(
        too_short,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=2500.0,
    )

    assert skipped_too_short[0].timing == too_short[0].timing
    assert "korean_cvc_cv_cutoff_before_next_row_repair" not in skipped_too_short[0].applied_rules


def test_korean_cvc_glide_cv_cutoff_caps_before_next_row():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ya.wav",
                alias="ya",
                timing=OtoTiming(890.0, 160.0, -1130.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="ya"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ya.wav",
                alias="ye",
                timing=OtoTiming(1420.0, 160.0, -600.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="ye"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4106.3,
    )

    assert repaired[0].timing == OtoTiming(890.0, 160.0, -330.0, 120.0, 85.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(1010.0)
    assert "korean_cvc_glide_cv_cutoff_before_next_row_repair" in repaired[0].applied_rules

    yoon_tail = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="jya.wav",
                alias="jyu",
                timing=OtoTiming(2940.0, 160.0, -590.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="u"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="jya.wav",
                alias="jeui",
                timing=OtoTiming(3480.0, 160.0, -300.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3540.0, role="cv", expected_phone="eui"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4127.7,
    )

    assert yoon_tail[0].timing == OtoTiming(2940.0, 160.0, -340.0, 60.0, 25.0)
    assert "korean_cvc_glide_cv_cutoff_before_next_row_repair" in yoon_tail[0].applied_rules

    rel_only = [
        AdaptedOtoRow(
            wav="wa.wav",
            alias="wa",
            timing=OtoTiming(910.0, 160.0, -2760.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1030.0, role="cv", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="wa.wav",
            alias="we",
            timing=OtoTiming(1380.0, 160.0, -620.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="we"),
            mode="template-bootstrap",
        ),
    ]
    skipped_rel_only = repair_cvvc_row_sequence(
        rel_only,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3978.3,
    )

    assert skipped_rel_only[0].timing == rel_only[0].timing
    assert "korean_cvc_glide_cv_cutoff_before_next_row_repair" not in skipped_rel_only[0].applied_rules

    too_short = [
        AdaptedOtoRow(
            wav="short-yoon.wav",
            alias="sya",
            timing=OtoTiming(1000.0, 160.0, -1000.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="short-yoon.wav",
            alias="sye",
            timing=OtoTiming(1200.0, 160.0, -500.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1320.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
    ]
    skipped_too_short = repair_cvvc_row_sequence(
        too_short,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=2500.0,
    )

    assert skipped_too_short[0].timing == too_short[0].timing
    assert "korean_cvc_glide_cv_cutoff_before_next_row_repair" not in skipped_too_short[0].applied_rules


def test_korean_cvc_doubled_cv_cutoff_uses_compact_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="dde.wav",
                alias="dde",
                timing=OtoTiming(2390.0, 160.0, -520.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2450.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="dde.wav",
                alias="et",
                timing=OtoTiming(2605.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2685.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4998.0,
    )

    assert repaired[0].timing == OtoTiming(2390.0, 160.0, -220.0, 60.0, 25.0)
    assert repaired[0].anchor is not None
    assert repaired[0].anchor.anchor_abs_ms == pytest.approx(2450.0)
    assert "korean_cvc_doubled_cv_cutoff_profile_repair" in repaired[0].applied_rules

    high_pre = [
        AdaptedOtoRow(
            wav="bbu.wav",
            alias="bbu",
            timing=OtoTiming(1000.0, 160.0, -520.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bbu.wav",
            alias="up",
            timing=OtoTiming(1215.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1295.0, role="vc", expected_phone="p"),
            mode="template-bootstrap",
        ),
    ]
    skipped_high_pre = repair_cvvc_row_sequence(
        high_pre,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert skipped_high_pre[0].timing == high_pre[0].timing
    assert "korean_cvc_doubled_cv_cutoff_profile_repair" not in skipped_high_pre[0].applied_rules

    short_cutoff = [
        AdaptedOtoRow(
            wav="sse.wav",
            alias="sse",
            timing=OtoTiming(1000.0, 160.0, -300.0, 60.0, 25.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1060.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="sse.wav",
            alias="es",
            timing=OtoTiming(1210.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1290.0, role="vc", expected_phone="s"),
            mode="template-bootstrap",
        ),
    ]
    skipped_short_cutoff = repair_cvvc_row_sequence(
        short_cutoff,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=2500.0,
    )

    assert skipped_short_cutoff[0].timing == short_cutoff[0].timing
    assert "korean_cvc_doubled_cv_cutoff_profile_repair" not in skipped_short_cutoff[0].applied_rules

    late_plain_doubled = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="jje.wav",
                alias="jje",
                timing=OtoTiming(2470.0, 160.0, -470.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2530.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="jje.wav",
                alias="et",
                timing=OtoTiming(3000.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3080.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5237.0,
    )

    assert late_plain_doubled[0].timing == OtoTiming(2470.0, 160.0, -220.0, 60.0, 25.0)
    assert "korean_cvc_doubled_cv_cutoff_profile_repair" in late_plain_doubled[0].applied_rules


def test_korean_cvc_plain_obstruent_cv_cutoff_uses_compact_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ta-to.wav",
                alias="ta",
                timing=OtoTiming(900.0, 160.0, -360.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1020.0, role="cv", expected_phone="a"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-to.wav",
                alias="a t",
                timing=OtoTiming(1300.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1380.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-to.wav",
                alias="to",
                timing=OtoTiming(1500.0, 160.0, -480.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1560.0, role="cv", expected_phone="o"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-to.wav",
                alias="o t",
                timing=OtoTiming(1660.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1740.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3500.0,
    )

    assert repaired[2].timing == OtoTiming(1500.0, 160.0, -250.0, 60.0, 25.0)
    assert "korean_cvc_plain_obstruent_cv_cutoff_profile_repair" in repaired[2].applied_rules

    high_pre = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ta-high.wav",
                alias="ta",
                timing=OtoTiming(900.0, 160.0, -360.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1020.0, role="cv", expected_phone="a"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-high.wav",
                alias="a t",
                timing=OtoTiming(1300.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1380.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-high.wav",
                alias="to",
                timing=OtoTiming(1500.0, 160.0, -480.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1620.0, role="cv", expected_phone="o"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ta-high.wav",
                alias="o t",
                timing=OtoTiming(2300.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2380.0, role="vc", expected_phone="t"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert "korean_cvc_plain_obstruent_cv_cutoff_profile_repair" not in high_pre[2].applied_rules


def test_korean_cvc_residual_glide_cv_cutoff_uses_compact_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ba-bwi.wav",
                alias="ba",
                timing=OtoTiming(900.0, 160.0, -360.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1020.0, role="cv", expected_phone="a"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ba-bwi.wav",
                alias="a b",
                timing=OtoTiming(1300.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1380.0, role="vc", expected_phone="b"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ba-bwi.wav",
                alias="bwi",
                timing=OtoTiming(1500.0, 160.0, -530.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1560.0, role="cv", expected_phone="wi"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ba-bwi.wav",
                alias="i b",
                timing=OtoTiming(2030.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2110.0, role="vc", expected_phone="b"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3600.0,
    )

    assert repaired[2].timing == OtoTiming(1500.0, 160.0, -260.0, 60.0, 25.0)
    assert "korean_cvc_residual_glide_cv_cutoff_profile_repair" in repaired[2].applied_rules

    early_row = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="early-bwi.wav",
                alias="bwi",
                timing=OtoTiming(1000.0, 160.0, -530.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1060.0, role="cv", expected_phone="wi"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="early-bwi.wav",
                alias="i b",
                timing=OtoTiming(1530.0, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1610.0, role="vc", expected_phone="b"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert early_row[0].timing == OtoTiming(1000.0, 160.0, -530.0, 60.0, 25.0)
    assert "korean_cvc_residual_glide_cv_cutoff_profile_repair" not in early_row[0].applied_rules


def test_korean_cvc_head_glide_cv_cutoff_uses_head_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="dya.wav",
                alias="dye",
                timing=OtoTiming(1380.0, 160.0, -590.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="dya.wav",
                alias="dyeo",
                timing=OtoTiming(1950.0, 160.0, -400.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2010.0, role="cv", expected_phone="eo"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3800.0,
    )

    assert repaired[0].timing == OtoTiming(1380.0, 160.0, -320.0, 120.0, 85.0)
    assert "korean_cvc_head_glide_cv_cutoff_profile_repair" in repaired[0].applied_rules

    compact_pre = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="compact-dye.wav",
                alias="dye",
                timing=OtoTiming(1380.0, 160.0, -590.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1440.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=3000.0,
    )

    assert compact_pre[0].timing == OtoTiming(1380.0, 160.0, -590.0, 60.0, 25.0)
    assert "korean_cvc_head_glide_cv_cutoff_profile_repair" not in compact_pre[0].applied_rules


def test_korean_cvc_sonorant_liquid_cv_huge_cutoff_uses_compact_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="ng.wav",
                alias="ngu",
                timing=OtoTiming(1850.0, 160.0, -3070.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1910.0, role="cv", expected_phone="u"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ng.wav",
                alias="ngo",
                timing=OtoTiming(2900.0, 160.0, -2020.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2960.0, role="cv", expected_phone="o"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4920.0,
    )

    assert repaired[0].timing == OtoTiming(1850.0, 160.0, -240.0, 60.0, 25.0)
    assert repaired[1].timing == OtoTiming(2900.0, 160.0, -240.0, 60.0, 25.0)
    assert "korean_cvc_sonorant_liquid_cv_huge_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_sonorant_liquid_cv_huge_cutoff_profile_repair" in repaired[1].applied_rules

    nasal_m = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="m.wav",
                alias="mye",
                timing=OtoTiming(1490.0, 160.0, -2310.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1610.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert nasal_m[0].timing == OtoTiming(1490.0, 160.0, -300.0, 120.0, 85.0)
    assert "korean_cvc_sonorant_liquid_cv_huge_cutoff_profile_repair" not in nasal_m[0].applied_rules
    assert "korean_cvc_mn_glide_cv_huge_cutoff_profile_repair" in nasal_m[0].applied_rules


def test_korean_cvc_mn_glide_huge_cutoff_uses_tail_safe_profile():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="mya.wav",
                alias="mya",
                timing=OtoTiming(1000.0, 160.0, -2800.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1120.0, role="cv", expected_phone="a"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="nyu.wav",
                alias="nyu",
                timing=OtoTiming(3400.0, 160.0, -950.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3460.0, role="cv", expected_phone="yu"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="mweo.wav",
                alias="mweo",
                timing=OtoTiming(3900.0, 160.0, -560.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3960.0, role="cv", expected_phone="weo"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5200.0,
    )

    assert repaired[0].timing == OtoTiming(1000.0, 160.0, -300.0, 120.0, 85.0)
    assert repaired[1].timing == OtoTiming(3400.0, 160.0, -300.0, 60.0, 25.0)
    assert repaired[2].timing == OtoTiming(3900.0, 160.0, -370.0, 60.0, 25.0)
    assert "korean_cvc_mn_glide_cv_huge_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_mn_glide_cv_huge_cutoff_profile_repair" in repaired[1].applied_rules
    assert "korean_cvc_tail_weo_cv_cutoff_profile_repair" in repaired[2].applied_rules
    assert "korean_cvc_residual_exact_cv_cutoff_profile_repair" in repaired[2].applied_rules


def test_korean_cvc_tail_weo_cutoff_uses_subtype_profile_with_exclusions():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="weo.wav",
                alias="dweo",
                timing=OtoTiming(2460.0, 160.0, -520.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2520.0, role="cv", expected_phone="weo"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="hweo.wav",
                alias="hweo",
                timing=OtoTiming(2940.0, 160.0, -540.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="weo"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ggweo.wav",
                alias="ggweo",
                timing=OtoTiming(3420.0, 160.0, -510.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3480.0, role="cv", expected_phone="weo"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4600.0,
    )

    assert repaired[0].timing == OtoTiming(2460.0, 160.0, -260.0, 60.0, 25.0)
    assert repaired[1].timing == OtoTiming(2940.0, 160.0, -540.0, 60.0, 25.0)
    assert repaired[2].timing == OtoTiming(3420.0, 160.0, -510.0, 60.0, 25.0)
    assert "korean_cvc_tail_weo_cv_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_tail_weo_cv_cutoff_profile_repair" not in repaired[1].applied_rules
    assert "korean_cvc_tail_weo_cv_cutoff_profile_repair" not in repaired[2].applied_rules


def test_korean_cvc_tail_wi_yoon_and_eui_cutoff_profiles():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="tail.wav",
                alias="rwa",
                timing=OtoTiming(900.0, 160.0, -220.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1020.0, role="cv", expected_phone="wa"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="tail.wav",
                alias="rwe",
                timing=OtoTiming(1420.0, 160.0, -220.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="we"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="tail.wav",
                alias="rwi",
                timing=OtoTiming(1950.0, 160.0, -670.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2010.0, role="cv", expected_phone="wi"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="gyu.wav",
                alias="gyu",
                timing=OtoTiming(2970.0, 160.0, -490.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3030.0, role="cv", expected_phone="u"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="meui.wav",
                alias="meui",
                timing=OtoTiming(3480.0, 160.0, -400.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=3540.0, role="cv", expected_phone="eui"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="eui.wav",
                alias="eui",
                timing=OtoTiming(3970.0, 160.0, -780.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=4090.0, role="cv", expected_phone="eui"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5200.0,
    )

    assert repaired[2].timing == OtoTiming(1950.0, 160.0, -300.0, 60.0, 25.0)
    assert repaired[3].timing == OtoTiming(2970.0, 160.0, -300.0, 60.0, 25.0)
    assert repaired[4].timing == OtoTiming(3480.0, 160.0, -260.0, 60.0, 25.0)
    assert repaired[5].timing == OtoTiming(3970.0, 160.0, -560.0, 120.0, 85.0)
    assert "korean_cvc_tail_wi_cv_cutoff_profile_repair" in repaired[2].applied_rules
    assert "korean_cvc_tail_yoon_cv_cutoff_profile_repair" in repaired[3].applied_rules
    assert "korean_cvc_tail_eui_compact_cv_cutoff_profile_repair" in repaired[4].applied_rules
    assert "korean_cvc_standalone_eui_head_cutoff_profile_repair" in repaired[5].applied_rules


def test_korean_cvc_residual_doubled_plain_cutoff_profiles():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="jji.wav",
                alias="jji",
                timing=OtoTiming(1420.0, 160.0, -450.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1540.0, role="cv", expected_phone="i"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="jjeo.wav",
                alias="jjeo",
                timing=OtoTiming(3980.0, 160.0, -440.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=4040.0, role="cv", expected_phone="eo"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ssi.wav",
                alias="ssi",
                timing=OtoTiming(1370.0, 160.0, -170.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1490.0, role="cv", expected_phone="i"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5600.0,
    )

    assert repaired[0].timing == OtoTiming(1420.0, 160.0, -240.0, 120.0, 85.0)
    assert repaired[1].timing == OtoTiming(3980.0, 160.0, -240.0, 60.0, 25.0)
    assert repaired[2].timing == OtoTiming(1370.0, 160.0, -300.0, 120.0, 85.0)
    assert "korean_cvc_residual_doubled_head_cv_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_residual_doubled_compact_cv_cutoff_profile_repair" in repaired[1].applied_rules
    assert "korean_cvc_residual_ss_doubled_head_short_cutoff_profile_repair" in repaired[2].applied_rules


def test_korean_cvc_residual_plain_and_sonorant_cutoff_profiles():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="plain.wav",
                alias="cheo",
                timing=OtoTiming(3900.0, 160.0, -440.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=4020.0, role="cv", expected_phone="eo"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="plain.wav",
                alias="ge",
                timing=OtoTiming(2430.0, 160.0, -220.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2490.0, role="cv", expected_phone="e"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="plain.wav",
                alias="gweo",
                timing=OtoTiming(2890.0, 160.0, -260.0, 60.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2950.0, role="cv", expected_phone="weo"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="ng.wav",
                alias="nga",
                timing=OtoTiming(4307.0, 160.0, -553.0, 120.0, 85.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=4427.0, role="cv", expected_phone="a"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5600.0,
    )

    assert repaired[0].timing == OtoTiming(3900.0, 160.0, -340.0, 120.0, 85.0)
    assert repaired[1].timing == OtoTiming(2430.0, 160.0, -380.0, 60.0, 25.0)
    assert repaired[2].timing == OtoTiming(2890.0, 160.0, -260.0, 60.0, 25.0)
    assert repaired[3].timing == OtoTiming(4307.0, 160.0, -170.0, 120.0, 85.0)
    assert "korean_cvc_residual_plain_head_cv_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_residual_g_compact_cv_cutoff_profile_repair" in repaired[1].applied_rules
    assert "korean_cvc_residual_g_compact_cv_cutoff_profile_repair" not in repaired[2].applied_rules
    assert "korean_cvc_residual_sonorant_head_cv_cutoff_profile_repair" in repaired[3].applied_rules


def test_korean_cvc_residual_vowel_and_nasal_vc_cutoff_profiles():
    repaired = repair_cvvc_row_sequence(
        [
            AdaptedOtoRow(
                wav="vowel.wav",
                alias="e",
                timing=OtoTiming(2423.2, 124.8, -516.8, 116.8, 59.2),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=2540.0, role="v", expected_phone="e"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="vowel.wav",
                alias="i",
                timing=OtoTiming(1480.0, 80.0, -300.0, 25.0, 25.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1505.0, role="v", expected_phone="i"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="m-vc.wav",
                alias="a m",
                timing=OtoTiming(1057.1, 120.0, -154.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1137.1, role="vc", expected_phone="m"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="n-vc.wav",
                alias="a n",
                timing=OtoTiming(1080.0, 120.0, -175.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1160.0, role="vc", expected_phone="n"),
                mode="template-bootstrap",
            ),
            AdaptedOtoRow(
                wav="an.wav",
                alias="an",
                timing=OtoTiming(1080.0, 120.0, -175.0, 80.0, 45.0),
                source_timing=None,
                anchor=OtoAnchor(anchor_abs_ms=1160.0, role="vc", expected_phone="n"),
                mode="template-bootstrap",
            ),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5600.0,
    )

    assert repaired[0].timing == OtoTiming(2423.2, 124.8, -320.0, 116.8, 59.2)
    assert repaired[1].timing == OtoTiming(1480.0, 80.0, -300.0, 25.0, 25.0)
    assert repaired[2].timing == OtoTiming(1057.1, 120.0, -280.0, 80.0, 45.0)
    assert repaired[3].timing == OtoTiming(1080.0, 120.0, -260.0, 80.0, 45.0)
    assert repaired[4].timing == OtoTiming(1080.0, 120.0, -360.0, 80.0, 45.0)
    assert "korean_cvc_residual_standalone_v_cutoff_profile_repair" in repaired[0].applied_rules
    assert "korean_cvc_residual_standalone_v_cutoff_profile_repair" not in repaired[1].applied_rules
    assert "korean_cvc_residual_spaced_m_vc_cutoff_profile_repair" in repaired[2].applied_rules
    assert "korean_cvc_residual_spaced_n_vc_cutoff_profile_repair" in repaired[3].applied_rules
    assert "korean_cvc_residual_compact_an_cutoff_profile_repair" in repaired[4].applied_rules


def test_korean_cvc_residual_exact_cv_cutoff_profiles():
    def row(alias: str, cutoff: float, pre: float, overlap: float) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=f"{alias}.wav",
            alias=alias,
            timing=OtoTiming(1200.0, 160.0, cutoff, pre, overlap),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1200.0 + pre, role="cv", expected_phone=alias),
            mode="template-bootstrap",
        )

    repaired = repair_cvvc_row_sequence(
        [
            row("beu", -200.0, 70.0, 35.0),
            row("peui", -340.0, 60.0, 25.0),
            row("lwe", -320.0, 120.0, 85.0),
            row("mweo", -260.0, 60.0, 25.0),
            row("ddye", -420.0, 120.0, 85.0),
            row("ddyu", -300.0, 60.0, 25.0),
            row("kye", -420.0, 120.0, 85.0),
            row("ngya", -380.0, 120.0, 85.0),
            row("pya", -320.0, 120.0, 85.0),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5600.0,
    )

    assert repaired[0].timing.cutoff == pytest.approx(-350.0)
    assert repaired[1].timing.cutoff == pytest.approx(-200.0)
    assert repaired[2].timing.cutoff == pytest.approx(-200.0)
    assert repaired[3].timing.cutoff == pytest.approx(-370.0)
    assert repaired[4].timing.cutoff == pytest.approx(-215.0)
    assert repaired[5].timing.cutoff == pytest.approx(-190.0)
    assert repaired[6].timing.cutoff == pytest.approx(-310.0)
    assert repaired[7].timing.cutoff == pytest.approx(-270.0)
    assert repaired[8].timing.cutoff == pytest.approx(-320.0)
    for index in range(8):
        assert "korean_cvc_residual_exact_cv_cutoff_profile_repair" in repaired[index].applied_rules
    assert "korean_cvc_residual_exact_cv_cutoff_profile_repair" not in repaired[8].applied_rules


def test_korean_cvc_residual_exact_pre_fixed_profiles():
    def row(alias: str, offset: float, consonant: float, cutoff: float, pre: float, overlap: float) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=f"{alias}.wav",
            alias=alias,
            timing=OtoTiming(offset, consonant, cutoff, pre, overlap),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=offset + pre, role="cv", expected_phone=alias),
            warnings=("source_timing_discarded",),
            mode="template-bootstrap",
        )

    repaired = repair_cvvc_row_sequence(
        [
            row("ba", 840.0, 160.0, -360.0, 120.0, 85.0),
            row("bbya", 1430.0, 160.0, -300.0, 120.0, 85.0),
            row("e", 2423.2, 305.8, -320.0, 116.8, 59.2),
            row("o", 2944.0, 245.0, -320.0, 56.0, 34.4),
            row("e g", 2730.0, 120.0, -154.0, 80.0, 45.0),
            row("ryu", 2995.0, 160.0, -240.0, 60.0, 25.0),
            row("eui", 3350.0, 160.0, -560.0, 120.0, 85.0),
            row("pya", 1430.0, 160.0, -300.0, 120.0, 85.0),
        ],
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5600.0,
    )

    assert repaired[0].timing.as_tuple() == pytest.approx(OtoTiming(935.0, 120.0, -265.0, 25.0, 12.0).as_tuple())
    assert repaired[1].timing.as_tuple() == pytest.approx(OtoTiming(1430.0, 180.0, -300.0, 25.0, 12.0).as_tuple())
    assert repaired[2].timing.as_tuple() == pytest.approx(OtoTiming(2505.0, 150.0, -238.2, 35.0, 35.0).as_tuple())
    assert repaired[3].timing.as_tuple() == pytest.approx(OtoTiming(2944.0, 150.0, -320.0, 56.0, 34.4).as_tuple())
    assert repaired[4].timing.as_tuple() == pytest.approx(OtoTiming(2730.0, 200.0, -208.0, 170.0, 135.0).as_tuple())
    assert repaired[5].timing.as_tuple() == pytest.approx(OtoTiming(2995.0, 320.0, -328.0, 60.0, 25.0).as_tuple())
    assert repaired[6].timing.as_tuple() == pytest.approx(OtoTiming(3430.0, 320.0, -560.0, 40.0, 15.0).as_tuple())
    assert repaired[7].timing.as_tuple() == pytest.approx(OtoTiming(1430.0, 160.0, -300.0, 120.0, 85.0).as_tuple())
    for index in range(7):
        assert "korean_cvc_residual_exact_profile_repair" in repaired[index].applied_rules
    assert "korean_cvc_residual_exact_profile_repair" not in repaired[7].applied_rules


def test_korean_cvc_pitch_suffix_grid_profile_calibrates_sato_style_rows():
    rows = [
        AdaptedOtoRow(
            wav="sample.wav",
            alias=alias,
            timing=OtoTiming(
                1000.0 + index * 400.0,
                160.0,
                -300.0,
                120.0,
                85.0,
            ),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1120.0 + index * 400.0),
            mode="template-bootstrap",
            warnings=("source_timing_discarded",),
        )
        for index, alias in enumerate(["- baC4S", "a bC4S", "baC4S", "apC4S", "kaC4S"])
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert repaired[3].timing.offset == pytest.approx(1857.748)
    assert repaired[3].timing.preutterance == pytest.approx(61.8)
    assert repaired[3].timing.overlap == pytest.approx(18.7)
    assert repaired[3].timing.consonant == pytest.approx(186.2)
    assert repaired[3].timing.cutoff == pytest.approx(-332.5)
    assert repaired[3].anchor is not None
    assert repaired[3].anchor.anchor_abs_ms == pytest.approx(1919.548)
    assert "korean_cvc_pitch_suffix_grid_profile_repair" in repaired[3].applied_rules
    assert repaired[4].timing.offset == pytest.approx(2143.376)
    assert repaired[4].timing.preutterance == pytest.approx(151.4)
    assert repaired[4].timing.overlap == pytest.approx(0.0)
    assert any(
        warning == "korean_cvc_pitch_suffix_grid_profile_subtype:n5_p4_k"
        for warning in repaired[4].warnings
    )


def test_korean_cvc_pitch_suffix_grid_profile_requires_style_suffix():
    rows = [
        AdaptedOtoRow(
            wav="sample.wav",
            alias=alias,
            timing=OtoTiming(
                1000.0 + index * 400.0,
                160.0,
                -300.0,
                120.0,
                85.0,
            ),
            source_timing=None,
            anchor=None,
            mode="template-bootstrap",
            warnings=("source_timing_discarded",),
        )
        for index, alias in enumerate(["fooC4", "barC4", "bazC4", "quxC4", "quuxC4"])
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=4000.0,
    )

    assert all("korean_cvc_pitch_suffix_grid_profile_repair" not in row.applied_rules for row in repaired)


def test_korean_cvc_pitch_suffix_grid_profile_calibrates_repeated_tail_subtypes():
    aliases = [
        "- jaC4S",
        "a jC4S",
        "jaC4S",
        "a jjC4S",
        "jjaC4S",
        "a chC4S",
        "angC4S",
        "a kC4S",
        "a RC4S",
    ]
    rows = [
        AdaptedOtoRow(
            wav="sample.wav",
            alias=alias,
            timing=OtoTiming(
                900.0 + index * 300.0,
                160.0,
                -300.0,
                120.0,
                85.0,
            ),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1020.0 + index * 300.0),
            mode="template-bootstrap",
            warnings=("source_timing_discarded",),
        )
        for index, alias in enumerate(aliases)
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[5].timing.offset == pytest.approx(2040.77)
    assert repaired[5].timing.preutterance == pytest.approx(98.6)
    assert repaired[6].timing.offset == pytest.approx(2716.67)
    assert repaired[6].timing.consonant == pytest.approx(208.6)
    assert repaired[7].timing.offset == pytest.approx(2478.28)
    assert repaired[7].timing.overlap == pytest.approx(56.1)
    assert any(
        warning == "korean_cvc_pitch_suffix_grid_profile_subtype:n9_p5_ch_vc"
        for warning in repaired[5].warnings
    )
    assert any(
        warning == "korean_cvc_pitch_suffix_grid_profile_subtype:n9_p6_compact_ng"
        for warning in repaired[6].warnings
    )
    assert any(
        warning == "korean_cvc_pitch_suffix_grid_profile_subtype:n9_p7_k_vc"
        for warning in repaired[7].warnings
    )


def test_korean_cvc_compact_cv_sequence_gap_repairs_yoon_rows():
    rows = [
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jya",
            timing=OtoTiming(920.0, 160.0, -570.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1040.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jye",
            timing=OtoTiming(1180.0, 160.0, -310.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1300.0, role="cv", expected_phone="e"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jyeo",
            timing=OtoTiming(1880.0, 160.0, -620.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2000.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jyo",
            timing=OtoTiming(2420.0, 160.0, -540.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2540.0, role="cv", expected_phone="o"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jyu",
            timing=OtoTiming(2880.0, 160.0, -590.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3000.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="jya.wav",
            alias="jeui",
            timing=OtoTiming(3160.0, 160.0, -310.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3280.0, role="cv", expected_phone="eui"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gwa.wav",
            alias="gwa",
            timing=OtoTiming(870.0, 160.0, -620.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=990.0, role="cv", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gwa.wav",
            alias="gwe",
            timing=OtoTiming(1230.0, 160.0, -260.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1350.0, role="cv", expected_phone="we"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gwa.wav",
            alias="gwi",
            timing=OtoTiming(1920.0, 160.0, -530.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2040.0, role="cv", expected_phone="wi"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="gwa.wav",
            alias="gweo",
            timing=OtoTiming(2430.0, 160.0, -430.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2550.0, role="cv", expected_phone="weo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mixed.wav",
            alias="ga",
            timing=OtoTiming(1030.0, 160.0, -490.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1150.0, role="cv", expected_phone="a"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mixed.wav",
            alias="a g",
            timing=OtoTiming(1140.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1220.0, role="vc", expected_phone="g"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="mixed.wav",
            alias="gi",
            timing=OtoTiming(1180.0, 160.0, -340.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1300.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing.offset == pytest.approx(1400.0)
    assert repaired[1].timing.preutterance == pytest.approx(120.0)
    assert repaired[1].timing.overlap == pytest.approx(85.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1520.0)
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[1].applied_rules
    assert "korean_cvc_compact_cv_sequence_profile_repair" not in repaired[1].applied_rules
    assert repaired[5].timing.offset == pytest.approx(3480.0)
    assert repaired[5].timing.preutterance == pytest.approx(60.0)
    assert repaired[5].timing.overlap == pytest.approx(25.0)
    assert repaired[5].anchor is not None
    assert repaired[5].anchor.anchor_abs_ms == pytest.approx(3540.0)
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[5].applied_rules
    assert "korean_cvc_compact_cv_sequence_profile_repair" in repaired[5].applied_rules
    assert repaired[7].timing.offset == pytest.approx(1395.0)
    assert repaired[7].timing.preutterance == pytest.approx(120.0)
    assert repaired[7].timing.overlap == pytest.approx(85.0)
    assert repaired[7].anchor is not None
    assert repaired[7].anchor.anchor_abs_ms == pytest.approx(1515.0)
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[7].applied_rules
    assert "korean_cvc_compact_cv_sequence_profile_repair" not in repaired[7].applied_rules
    assert repaired[10].timing == OtoTiming(940.0, 180.0, -490.0, 70.0, 35.0)
    assert "korean_cvc_initial_cv_onset_repair" in repaired[10].applied_rules
    assert repaired[11] == rows[11]
    assert repaired[12] == rows[12]


def test_korean_cvc_compact_cv_sequence_gap_repairs_unaliased_head_and_reverse_gap():
    rows = [
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="lya",
            timing=OtoTiming(1590.0, 160.0, -400.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1710.0, role="cv", expected_phone="ya"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="lye",
            timing=OtoTiming(2030.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="ye"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="lyeo",
            timing=OtoTiming(2410.0, 160.0, -570.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2530.0, role="cv", expected_phone="yeo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="lyo",
            timing=OtoTiming(2870.0, 160.0, -640.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2990.0, role="cv", expected_phone="yo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="lyu",
            timing=OtoTiming(3370.0, 160.0, -700.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3490.0, role="cv", expected_phone="yu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="al'ryal'ryel'ryeol'lyol'ryul'reuil.wav",
            alias="leui",
            timing=OtoTiming(3950.0, 160.0, -290.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4070.0, role="cv", expected_phone="eui"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bwa'bwe'bwi'bweo.wav",
            alias="bwa",
            timing=OtoTiming(940.0, 180.0, -290.0, 70.0, 35.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1010.0, role="cv", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bwa'bwe'bwi'bweo.wav",
            alias="bwe",
            timing=OtoTiming(1620.0, 160.0, -220.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1740.0, role="cv", expected_phone="we"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bwa'bwe'bwi'bweo.wav",
            alias="bwi",
            timing=OtoTiming(1890.0, 160.0, -420.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2010.0, role="cv", expected_phone="wi"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="bwa'bwe'bwi'bweo.wav",
            alias="bweo",
            timing=OtoTiming(2420.0, 160.0, -450.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2540.0, role="cv", expected_phone="weo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang'wang'weng'wing'weong.wav",
            alias="ngwa",
            timing=OtoTiming(1390.0, 160.0, -2130.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1510.0, role="cv", expected_phone="wa"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang'wang'weng'wing'weong.wav",
            alias="ngwe",
            timing=OtoTiming(1810.0, 160.0, -1710.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1930.0, role="cv", expected_phone="we"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang'wang'weng'wing'weong.wav",
            alias="ngwi",
            timing=OtoTiming(2310.0, 160.0, -1210.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2430.0, role="cv", expected_phone="wi"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ang'wang'weng'wing'weong.wav",
            alias="ngweo",
            timing=OtoTiming(2890.0, 160.0, -630.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3010.0, role="cv", expected_phone="weo"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[0].timing.offset == pytest.approx(1410.0)
    assert repaired[1].timing.offset == pytest.approx(1910.0)
    assert repaired[1].timing.preutterance == pytest.approx(120.0)
    assert repaired[1].timing.overlap == pytest.approx(85.0)
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[0].applied_rules
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[1].applied_rules
    assert "korean_cvc_compact_cv_sequence_profile_repair" not in repaired[1].applied_rules
    assert repaired[7].timing.offset == pytest.approx(1390.0)
    assert repaired[7].timing.preutterance == pytest.approx(120.0)
    assert repaired[7].timing.overlap == pytest.approx(85.0)
    assert "korean_cvc_compact_cv_sequence_gap_repair" in repaired[7].applied_rules
    assert "korean_cvc_compact_cv_sequence_profile_repair" not in repaired[7].applied_rules
    assert repaired[8].timing == OtoTiming(1950.0, 160.0, -300.0, 60.0, 25.0)
    assert repaired[8].anchor is not None
    assert repaired[8].anchor.anchor_abs_ms == pytest.approx(2010.0)
    assert "korean_cvc_compact_cv_sequence_profile_repair" in repaired[8].applied_rules
    assert "korean_cvc_tail_wi_cv_cutoff_profile_repair" in repaired[8].applied_rules
    assert repaired[10].timing == OtoTiming(1390.0, 160.0, -340.0, 120.0, 85.0)
    assert repaired[10].anchor is not None
    assert repaired[10].anchor.anchor_abs_ms == pytest.approx(1510.0)
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[10].applied_rules
    assert repaired[11].timing == OtoTiming(1810.0, 160.0, -320.0, 120.0, 85.0)
    assert repaired[11].anchor is not None
    assert repaired[11].anchor.anchor_abs_ms == pytest.approx(1930.0)
    assert "korean_cvc_head_glide_cv_cutoff_profile_repair" in repaired[11].applied_rules
    assert repaired[12].timing == OtoTiming(2370.0, 160.0, -300.0, 60.0, 25.0)
    assert repaired[12].anchor is not None
    assert repaired[12].anchor.anchor_abs_ms == pytest.approx(2430.0)
    assert "korean_cvc_compact_cv_sequence_profile_repair" in repaired[12].applied_rules
    assert "korean_cvc_tail_wi_cv_cutoff_profile_repair" in repaired[12].applied_rules


def test_korean_cvc_alternating_cv_profile_repairs_long_cutoff_obstruent_rows_only():
    rows = [
        AdaptedOtoRow(
            wav="ba.wav",
            alias="a b",
            timing=OtoTiming(1280.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1360.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bi",
            timing=OtoTiming(1380.0, 160.0, -550.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1500.0, role="cv", expected_phone="i"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="i b",
            timing=OtoTiming(1840.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1920.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="bu",
            timing=OtoTiming(2030.0, 160.0, -400.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2150.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ba.wav",
            alias="u b",
            timing=OtoTiming(2400.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2480.0, role="vc", expected_phone="b"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="il",
            timing=OtoTiming(1600.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=1680.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="lu",
            timing=OtoTiming(1980.0, 160.0, -1010.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2100.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="ul",
            timing=OtoTiming(2150.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2230.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(1440.0, 160.0, -320.0, 60.0, 25.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(1500.0)
    assert "korean_cvc_alternating_cv_profile_repair" in repaired[1].applied_rules
    assert "korean_cvc_cv_cutoff_before_next_row_repair" in repaired[1].applied_rules
    assert repaired[3] == rows[3]
    assert repaired[6].timing == OtoTiming(1980.0, 160.0, -240.0, 120.0, 85.0)
    assert "korean_cvc_sonorant_liquid_cv_huge_cutoff_profile_repair" in repaired[6].applied_rules


def test_korean_cvc_no_space_vc_cadence_repairs_late_l_and_eu_t_only():
    rows = [
        AdaptedOtoRow(
            wav="ral.wav",
            alias="lu",
            timing=OtoTiming(1980.0, 160.0, -1010.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2100.0, role="cv", expected_phone="u"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="ul",
            timing=OtoTiming(2150.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=2230.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="leu",
            timing=OtoTiming(3450.0, 160.0, -580.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3570.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="eul",
            timing=OtoTiming(3620.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3700.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="leo",
            timing=OtoTiming(4000.0, 160.0, -480.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4120.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ral.wav",
            alias="eol",
            timing=OtoTiming(4170.0, 120.0, -154.0, 80.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4250.0, role="vc", expected_phone="l"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ddeu",
            timing=OtoTiming(3550.0, 160.0, -280.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3670.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="eut",
            timing=OtoTiming(3805.0, 145.0, -195.0, 115.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3920.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="dda.wav",
            alias="ddeo",
            timing=OtoTiming(4060.0, 160.0, -240.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="teu",
            timing=OtoTiming(3550.0, 160.0, -280.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3670.0, role="cv", expected_phone="eu"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="eu t",
            timing=OtoTiming(3805.0, 145.0, -195.0, 115.0, 45.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=3920.0, role="vc", expected_phone="t"),
            mode="template-bootstrap",
        ),
        AdaptedOtoRow(
            wav="ta.wav",
            alias="teo",
            timing=OtoTiming(4060.0, 160.0, -240.0, 120.0, 85.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4180.0, role="cv", expected_phone="eo"),
            mode="template-bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
        file_duration_ms=5000.0,
    )

    assert repaired[1].timing == OtoTiming(2050.0, 120.0, -154.0, 80.0, 45.0)
    assert repaired[1].anchor is not None
    assert repaired[1].anchor.anchor_abs_ms == pytest.approx(2130.0)
    assert "korean_cvc_no_space_vc_cadence_repair" in repaired[1].applied_rules
    assert repaired[3].timing == OtoTiming(3540.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_no_space_vc_cadence_repair" in repaired[3].applied_rules
    assert repaired[5].timing == OtoTiming(4065.0, 120.0, -154.0, 80.0, 45.0)
    assert "korean_cvc_no_space_vc_cadence_repair" in repaired[5].applied_rules
    assert repaired[7].timing == OtoTiming(3700.0, 145.0, -195.0, 115.0, 45.0)
    assert "korean_cvc_no_space_vc_cadence_repair" in repaired[7].applied_rules
    assert repaired[10] == rows[10]


def test_slot_viterbi_sonorant_phone_change_uses_spectral_shape_delta_when_event_is_late():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.05 for _ in times],
        "flux_likelihood": [0.05 for _ in times],
        "sonorant_onset_likelihood": [0.04 for _ in times],
        "spectral_shape_delta_likelihood": [0.03 for _ in times],
        "voicing": [0.86 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.74 if 100.0 <= time_ms <= 260.0 else 0.20
        class_probs["consonant"][idx] = 0.24 if 120.0 <= time_ms <= 160.0 else 0.08
        event_scores["phone_change"][idx] = max(
            event_scores["phone_change"][idx],
            0.28 * math.exp(-0.5 * ((time_ms - 250.0) / 8.0) ** 2),
        )
        acoustic_scores["spectral_shape_delta_likelihood"][idx] = max(
            acoustic_scores["spectral_shape_delta_likelihood"][idx],
            0.96 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [ExpectedSlot(slot_index=0, phone_index=1, phone="n", role="vc", event_label="phone_change")]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    assert result.assignments[0].selected_time_ms == pytest.approx(140.0, abs=6.0)


def test_slot_viterbi_vv_uses_vowel_boundary_when_nucleus_event_is_late():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.04 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.04 for _ in times],
        "vowel_boundary_likelihood": [0.03 for _ in times],
        "spectral_shape_delta_likelihood": [0.03 for _ in times],
        "voicing": [0.84 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
        "nucleus_likelihood": [0.08 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.84 if 80.0 <= time_ms <= 300.0 else 0.12
        event_scores["vowel_nucleus"][idx] = max(
            event_scores["vowel_nucleus"][idx],
            0.56 * math.exp(-0.5 * ((time_ms - 260.0) / 8.0) ** 2),
        )
        acoustic_scores["vowel_boundary_likelihood"][idx] = max(
            acoustic_scores["vowel_boundary_likelihood"][idx],
            0.96 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slots = [ExpectedSlot(slot_index=0, phone_index=1, phone="i", role="vv", event_label="vowel_nucleus")]

    result = assign_slots_viterbi(posterior, expected_slots=slots, min_event_score=0.03)

    assert result.ok
    assert result.assignments[0].selected_time_ms == pytest.approx(140.0, abs=6.0)


def test_japanese_slot_duration_prior_opt_in_penalizes_one_step_sonorant_distractor(monkeypatch):
    times = [float(idx * 10) for idx in range(301)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.03 for _ in times],
        "sonorant_onset_likelihood": [0.03 for _ in times],
        "spectral_shape_delta_likelihood": [0.03 for _ in times],
        "voicing": [0.20 for _ in times],
        "silence_likelihood": [0.02 for _ in times],
    }

    def add_peak(center_ms: float, amp: float) -> None:
        for frame_idx, time_ms in enumerate(times):
            peak = math.exp(-0.5 * ((time_ms - center_ms) / 7.0) ** 2)
            event_scores["phone_change"][frame_idx] = max(event_scores["phone_change"][frame_idx], amp * peak)
            acoustic_scores["sonorant_onset_likelihood"][frame_idx] = max(
                acoustic_scores["sonorant_onset_likelihood"][frame_idx],
                amp * peak,
            )
            acoustic_scores["transition_likelihood"][frame_idx] = max(
                acoustic_scores["transition_likelihood"][frame_idx],
                0.90 * amp * peak,
            )
            acoustic_scores["spectral_shape_delta_likelihood"][frame_idx] = max(
                acoustic_scores["spectral_shape_delta_likelihood"][frame_idx],
                0.80 * amp * peak,
            )

    add_peak(1000.0, 0.75)
    add_peak(1500.0, 0.82)
    posterior = FramePosterior(
        wav_path="ja_sonorant.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    slot = ExpectedSlot(slot_index=1, phone_index=1, phone="n", role="vc", event_label="phone_change")

    def scores_by_time() -> dict[float, float]:
        candidates = _slot_candidates(
            posterior,
            slot,
            min_event_score=0.03,
            top_k=8,
            expected_time_ms=1000.0,
            expected_time_weight=0.20,
            slot_period_ms=500.0,
            slot_count=6,
            language="japanese",
            min_time_ms=None,
            max_time_ms=None,
            window_ms=1200.0,
            expected_time_hard_window_ms=None,
            expected_time_fallback_enabled=False,
            nucleus_min_peak_distance_ms=26.0,
            dense_row=False,
        )
        return {round(candidate.time_ms, 1): float(candidate.score) for candidate in candidates}

    monkeypatch.setenv("UTOA_NO_MFA_JA_SLOT_DURATION_PRIOR_WEIGHT", "0")
    no_prior_scores = scores_by_time()
    assert no_prior_scores[1500.0] > no_prior_scores[1000.0]

    monkeypatch.setenv("UTOA_NO_MFA_JA_SLOT_DURATION_PRIOR_WEIGHT", "0.12")
    opt_in_prior_scores = scores_by_time()
    assert opt_in_prior_scores[1000.0] > opt_in_prior_scores[1500.0]


def test_acoustic_aux_sonorant_onset_sees_spectral_shape_change_without_volume_rise(tmp_path):
    wav_path = tmp_path / "same_level_shape_change.wav"
    sample_rate = 16000
    duration_s = 0.45
    switch = int(sample_rate * 0.22)
    samples: list[int] = []
    for idx in range(int(sample_rate * duration_s)):
        freq = 440.0 if idx < switch else 180.0
        value = int(9000 * math.sin(2.0 * math.pi * freq * idx / sample_rate))
        samples.append(value)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())

    batch = extract_features(wav_path, encoder="acoustic-aux")
    times = np.asarray(batch.times_ms, dtype=np.float32)
    sonorant = np.asarray(batch.acoustic_scores["sonorant_onset_likelihood"], dtype=np.float32)
    shape_delta = np.asarray(batch.acoustic_scores["spectral_shape_delta_likelihood"], dtype=np.float32)
    search = np.where((times >= 120.0) & (times <= 300.0))[0]
    peak_time = float(times[search[int(np.argmax(sonorant[search]))]])

    assert float(np.max(shape_delta[search])) > 0.35
    assert peak_time == pytest.approx(220.0, abs=45.0)


def test_acoustic_aux_vowel_boundary_sees_same_level_vowel_change(tmp_path):
    wav_path = tmp_path / "same_level_vowel_change.wav"
    sample_rate = 16000
    duration_s = 0.45
    switch = int(sample_rate * 0.22)
    samples: list[int] = []
    for idx in range(int(sample_rate * duration_s)):
        freq = 360.0 if idx < switch else 620.0
        value = int(9000 * math.sin(2.0 * math.pi * freq * idx / sample_rate))
        samples.append(value)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())

    batch = extract_features(wav_path, encoder="acoustic-aux")
    times = np.asarray(batch.times_ms, dtype=np.float32)
    boundary = np.asarray(batch.acoustic_scores["vowel_boundary_likelihood"], dtype=np.float32)
    search = np.where((times >= 120.0) & (times <= 300.0))[0]
    peak_time = float(times[search[int(np.argmax(boundary[search]))]])

    assert float(np.max(boundary[search])) > 0.30
    assert peak_time == pytest.approx(220.0, abs=45.0)


def test_acoustic_aux_features_preserve_absolute_gate_tracks(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path, duration_s=0.35)
    acoustic = extract_features(wav_path, encoder="acoustic")
    aux = extract_features(wav_path, encoder="acoustic-aux")
    assert acoustic.features.shape[1] == 29
    assert aux.features.shape[1] > acoustic.features.shape[1]
    assert aux.times_ms.shape[0] == aux.features.shape[0]
    assert {"rms", "spectral_flux", "voicing", "silence_likelihood", "transition_likelihood", "vowel_boundary_likelihood"}.issubset(aux.acoustic_scores)
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
        (0, "vowel_nucleus"),
        (1, "vowel_nucleus"),
        (2, "vowel_nucleus"),
    ]


def test_initial_bare_vowel_alias_does_not_shift_to_duplicate_vv_target():
    rows = [
        OtoTemplateRow("_a.wav", "\u3042", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_a.wav", "a \u3042", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_a.wav", "a \u3044", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["a", "a", "i"]) == [0, 1, 2]

    slots = expected_slots_for_template_rows(rows, ["a", "a", "i"])
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (0, "v", "vowel_nucleus"),
        (1, "vv", "vowel_nucleus"),
        (2, "vv", "vowel_nucleus"),
    ]


def test_template_terminal_dash_row_does_not_reject_hsmm_sequence():
    rows = [
        parse_template_oto_line("_a.wav=- a,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a a,0,0,0,0,0"),
        parse_template_oto_line("_a.wav=a -,0,0,0,0,0"),
    ]
    slots = expected_slots_for_template_rows([row for row in rows if row is not None], ["a", "a"])

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (0, "cv_head", "cv_boundary"),
        (1, "vv", "vowel_nucleus"),
    ]


def test_initial_vowel_cv_head_does_not_shift_to_following_duplicate_vv_target():
    rows = [
        OtoTemplateRow("_a.wav", "- \u3042", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_a.wav", "a \u3042", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_a.wav", "a \u3044", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_a.wav", "a -", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["a", "a", "i"]) == [0, 1, 2, 1]

    slots = expected_slots_for_template_rows(rows, ["a", "a", "i"])
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (0, "cv_head", "cv_boundary"),
        (1, "vv", "vowel_nucleus"),
        (2, "vv", "vowel_nucleus"),
    ]


def test_hold_release_suffix_aliases_are_not_assigned_to_hsmm_phone_slots():
    rows = [
        OtoTemplateRow("_aa_hh.wav", "\u3042L", OtoTiming(10.0, 20.0, -30.0, 15.0, 5.0)),
        OtoTemplateRow("_aa_hh.wav", "a H", OtoTiming(40.0, 50.0, -60.0, 25.0, 10.0)),
    ]

    assert _alias_phone_sequence("\u3042L") == ["a"]
    assert _is_nonphonetic_special_alias("\u3042L")
    assert _is_nonphonetic_special_alias("a H")
    assert not _is_nonphonetic_special_alias("a h")
    assert _assign_alias_target_indices(rows, ["a", "a"]) == [None, None]
    assert expected_slots_for_template_rows(rows, ["a", "a"]) == []


def test_japanese_cvvc_terminal_release_h_uses_posterior_energy_end():
    frame_count = 550
    times = np.arange(frame_count, dtype=np.float32) * 10.0
    rms = np.full(frame_count, 0.001, dtype=np.float32)
    rms[90:431] = 0.10
    posterior = FramePosterior(
        wav_path="_aahh.wav",
        times_ms=times,
        class_probs={},
        event_scores={},
        acoustic_scores={"rms": rms},
    )
    row = AdaptedOtoRow(
        wav="_aahh.wav",
        alias="a H_D4",
        timing=OtoTiming(1155.0, 210.0, -236.0, 170.0, 135.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1325.0, score=0.8),
        mode="template-bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5500.0,
        posterior=posterior,
    )

    assert repaired.timing.offset + repaired.timing.preutterance == pytest.approx(4300.0)
    assert repaired.anchor is not None
    assert repaired.anchor.anchor_abs_ms == pytest.approx(4300.0)
    assert "cvvc_terminal_release_h_energy_repair" in repaired.applied_rules


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


def test_japanese_moraic_n_transition_rows_use_vowel_slots_before_spaced_n_aliases():
    rows = [
        OtoTemplateRow("n-i-n-e-n.wav", "- n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("n-i-n-e-n.wav", "n i", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("n-i-n-e-n.wav", "i n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("n-i-n-e-n.wav", "n e", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("n-i-n-e-n.wav", "e n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("n-i-n-e-n.wav", "n -", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["n", "i", "n", "e", "n"]) == [0, 1, 2, 3, 4, 4]

    slots = expected_slots_for_template_rows(rows, ["n", "i", "n", "e", "n"])
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (0, "v", "vowel_nucleus"),
        (1, "vcv", "cv_boundary"),
        (2, "vv", "vowel_nucleus"),
        (3, "vcv", "cv_boundary"),
        (4, "vv", "vowel_nucleus"),
    ]


def test_japanese_moraic_n_anchor_refine_pulls_late_nucleus_to_sonorant_onset():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.05 for _ in times],
        "flux_likelihood": [0.04 for _ in times],
        "sonorant_onset_likelihood": [0.03 for _ in times],
        "spectral_shape_delta_likelihood": [0.03 for _ in times],
        "voicing": [0.88 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
        "nucleus_likelihood": [0.05 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.72 if 80.0 <= time_ms <= 260.0 else 0.18
        event_scores["vowel_nucleus"][idx] = max(
            event_scores["vowel_nucleus"][idx],
            0.82 * math.exp(-0.5 * ((time_ms - 220.0) / 8.0) ** 2),
        )
        acoustic_scores["sonorant_onset_likelihood"][idx] = max(
            acoustic_scores["sonorant_onset_likelihood"][idx],
            0.90 * math.exp(-0.5 * ((time_ms - 130.0) / 8.0) ** 2),
        )
        acoustic_scores["spectral_shape_delta_likelihood"][idx] = max(
            acoustic_scores["spectral_shape_delta_likelihood"][idx],
            0.92 * math.exp(-0.5 * ((time_ms - 130.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    rows = [
        OtoTemplateRow("a-n-a.wav", "a n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("a-n-a.wav", "n a", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    decoded = [
        {
            "label": "vowel_nucleus",
            "selected_time_ms": 220.0,
            "score": 0.82,
            "expected_phone": "n",
            "expected_phone_index": 1,
            "frame_index": 22,
            "source": "filename_hsmm",
        },
        {
            "label": "cv_boundary",
            "selected_time_ms": 310.0,
            "score": 0.70,
            "expected_phone": "a",
            "expected_phone_index": 2,
            "frame_index": 31,
            "source": "filename_hsmm",
        },
    ]

    anchors = assign_template_row_anchors(
        posterior,
        decoded,
        rows,
        expected_phones=["a", "n", "a"],
        use_source_timing_prior=False,
    )

    assert anchors[0] is not None
    assert anchors[0].anchor_abs_ms == pytest.approx(130.0, abs=6.0)
    assert "sonorant_onset_local_refine" in anchors[0].warnings


def test_japanese_vv_anchor_refine_uses_vowel_boundary_not_late_nucleus():
    times = [float(idx * 10) for idx in range(41)]
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    class_probs = {label: [0.05 for _ in times] for label in FRAME_LABELS}
    acoustic_scores = {
        "transition_likelihood": [0.04 for _ in times],
        "vowel_boundary_likelihood": [0.03 for _ in times],
        "spectral_shape_delta_likelihood": [0.03 for _ in times],
        "voicing": [0.88 for _ in times],
        "silence_likelihood": [0.08 for _ in times],
        "nucleus_likelihood": [0.06 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.76 if 70.0 <= time_ms <= 310.0 else 0.18
        event_scores["vowel_nucleus"][idx] = max(
            event_scores["vowel_nucleus"][idx],
            0.86 * math.exp(-0.5 * ((time_ms - 250.0) / 8.0) ** 2),
        )
        acoustic_scores["vowel_boundary_likelihood"][idx] = max(
            acoustic_scores["vowel_boundary_likelihood"][idx],
            0.94 * math.exp(-0.5 * ((time_ms - 140.0) / 8.0) ** 2),
        )
    posterior = FramePosterior(
        wav_path="dummy.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    rows = [OtoTemplateRow("a-i.wav", "a i", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))]
    decoded = [
        {
            "label": "vowel_nucleus",
            "selected_time_ms": 250.0,
            "score": 0.86,
            "expected_phone": "i",
            "expected_phone_index": 1,
            "frame_index": 25,
            "source": "filename_hsmm",
        }
    ]

    anchors = assign_template_row_anchors(
        posterior,
        decoded,
        rows,
        expected_phones=["a", "i"],
        use_source_timing_prior=False,
    )

    assert anchors[0] is not None
    assert anchors[0].anchor_abs_ms == pytest.approx(140.0, abs=6.0)
    assert "vowel_boundary_local_refine" in anchors[0].warnings


def test_japanese_cvvc_moraic_n_suffix_alias_targets_n_slot():
    rows = [
        OtoTemplateRow("kan.wav", "a \u3093ng", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("pan.wav", "a \u3093m", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("tan.wav", "a \u3093n", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["k", "a", "n", "k", "a"]

    assert [_alias_phone_sequence(row.alias) for row in rows] == [["a", "n"], ["a", "n"], ["a", "n"]]
    assert _assign_alias_target_indices(rows, phones) == [2, 2, 2]

    slots = expected_slots_for_template_rows(rows[:1], phones)
    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (2, "vv", "vowel_nucleus"),
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


def test_cvvc_alias_targets_ignore_separated_pitch_suffix_tokens():
    rows = [
        OtoTemplateRow("ka.wav", "a k_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ka.wav", "i k_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ka.wav", "ka_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ka.wav", "ki_A4", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["k", "a", "k", "i", "k", "a"]

    assert _alias_type_for_row("a k_S", "auto") == "vc"
    assert _alias_phone_sequence("ki_A4") == ["k", "i"]
    assert _assign_alias_target_indices(rows, phones) == [2, 4, 1, 3]


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

    assert _assign_alias_target_indices(rows, phones) == [0, 2, 4, 6, 8, 10, 13, 3, 5, 7, 9, 11]


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


def test_japanese_cvvc_timeline_slots_can_include_omitted_initial_cv_boundary(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_JA_CVVC_LEADING_CONTEXT_SLOTS", "1")
    rows = [
        OtoTemplateRow("ga.wav", "a g", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ga.wav", "i g", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ga.wav", "\u304e", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ga.wav", "\u3050", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    phones = ["g", "a", "g", "i", "g", "u"]

    slots = timeline_expected_slots_for_template_rows(rows, phones, language="japanese")

    assert [(slot.phone_index, slot.role, slot.event_label) for slot in slots] == [
        (1, "implicit_cv", "cv_boundary"),
        (2, "vc", "phone_change"),
        (3, "implicit_cv", "cv_boundary"),
        (4, "vc", "phone_change"),
        (5, "implicit_cv", "cv_boundary"),
    ]


def test_japanese_cvvc_hsmm_replacement_rejects_aligned_runtime_first_gap(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_JA_CVVC_HSMM_RUNTIME_GAP_GUARD", "1")
    slots = [
        ExpectedSlot(slot_index=0, phone_index=2, phone="g", role="vc", event_label="phone_change"),
        ExpectedSlot(slot_index=1, phone_index=3, phone="a", role="implicit_cv", event_label="cv_boundary"),
    ]
    runtime_events = [
        {"label": "phone_change", "expected_phone_index": 2, "selected_time_ms": 1010.0},
        {"label": "cv_boundary", "expected_phone_index": 3, "selected_time_ms": 1526.0},
    ]

    reason = _hsmm_runtime_replacement_rejection_reason(
        slots,
        runtime_events,
        language="japanese",
        format_type="cvvc",
    )

    assert reason.startswith("runtime_first_vc_cv_gap_aligned:")


def test_japanese_cvvc_hsmm_replacement_allows_collapsed_runtime_first_gap(monkeypatch):
    monkeypatch.setenv("UTOA_NO_MFA_JA_CVVC_HSMM_RUNTIME_GAP_GUARD", "1")
    slots = [
        ExpectedSlot(slot_index=0, phone_index=2, phone="g", role="vc", event_label="phone_change"),
        ExpectedSlot(slot_index=1, phone_index=3, phone="a", role="implicit_cv", event_label="cv_boundary"),
    ]
    runtime_events = [
        {"label": "phone_change", "expected_phone_index": 2, "selected_time_ms": 990.0},
        {"label": "cv_boundary", "expected_phone_index": 3, "selected_time_ms": 1108.0},
    ]

    assert (
        _hsmm_runtime_replacement_rejection_reason(
            slots,
            runtime_events,
            language="japanese",
            format_type="cvvc",
        )
        == ""
    )


def test_japanese_cvvc_hsmm_replacement_allows_soft_suffix_template_rows():
    slots = [
        ExpectedSlot(slot_index=0, phone_index=2, phone="g", role="vc", event_label="phone_change"),
        ExpectedSlot(slot_index=1, phone_index=3, phone="a", role="implicit_cv", event_label="cv_boundary"),
    ]
    runtime_events = [
        {"label": "phone_change", "expected_phone_index": 2, "selected_time_ms": 990.0},
        {"label": "cv_boundary", "expected_phone_index": 3, "selected_time_ms": 1108.0},
    ]
    template_rows = [
        OtoTemplateRow("_ga.wav", "a g_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert (
        _hsmm_runtime_replacement_rejection_reason(
            slots,
            runtime_events,
            language="japanese",
            format_type="cvvc",
            template_rows=template_rows,
        )
        == ""
    )


def test_cvvc_soft_suffix_cv_head_targets_first_following_vc_onset():
    rows = [
        OtoTemplateRow("ga.wav", "- g_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ga.wav", "a g_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("ga.wav", "gi_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["g", "a", "g", "i"]) == [2, 2, 3]

    pure_vowel_rows = [
        OtoTemplateRow("_n-n-a-n.wav", "- n_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("_n-n-a-n.wav", "a n_S", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    assert _assign_alias_target_indices(pure_vowel_rows, ["n", "n", "a", "n"]) == [0, 3]


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


def test_japanese_cvvc_following_cv_block_cutoff_cap_preserves_offset_and_companion_timing(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
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
    assert repaired_romaji.timing.preutterance == pytest.approx(150.0)
    assert repaired_romaji.timing.overlap == pytest.approx(115.0)
    assert repaired_short.timing.cutoff == pytest.approx(-360.0)
    assert "cvvc_following_cv_block_cutoff_cap" in repaired_kana.applied_rules
    assert "cvvc_cv_role_profile_repair" not in repaired_kana.applied_rules
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


def test_japanese_cvvc_cv_sequence_cutoff_keeps_near_next_cv_tail(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
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

    assert repaired_first.timing.offset == pytest.approx(first_cv.timing.offset)
    assert repaired_first.timing.consonant == pytest.approx(first_cv.timing.consonant)
    assert repaired_first.timing.cutoff == pytest.approx(first_cv.timing.cutoff)
    assert repaired_first.timing.preutterance == pytest.approx(70.0)
    assert repaired_first.timing.overlap == pytest.approx(25.0)
    assert "cvvc_cv_sequence_next_cv_cutoff_cap" not in repaired_first.applied_rules
    assert "cvvc_cv_role_profile_repair" in repaired_first.applied_rules


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


def test_cvvc_initial_cv_head_keeps_own_onset_when_first_cv_row_is_omitted():
    rows = [
        OtoTemplateRow("sa.wav", "- s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "a s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "i s", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
        OtoTemplateRow("sa.wav", "\u3059\u3043", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0)),
    ]

    assert _assign_alias_target_indices(rows, ["s", "a", "s", "i", "s", "u"]) == [0, 2, 4, 3]


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


def test_template_anchor_assignment_rejects_leading_noise_for_later_sonorant_target():
    times = [float(idx * 10) for idx in range(61)]
    class_probs = {label: [0.02 for _ in times] for label in FRAME_LABELS}
    event_scores = {label: [0.01 for _ in times] for label in EVENT_LABELS}
    acoustic_scores = {
        "silence_likelihood": [0.04 for _ in times],
        "voicing": [0.55 for _ in times],
        "transition_likelihood": [0.02 for _ in times],
        "nucleus_likelihood": [0.02 for _ in times],
    }
    for idx, time_ms in enumerate(times):
        class_probs["vowel"][idx] = 0.72 if 250.0 <= time_ms <= 460.0 else 0.05
        class_probs["consonant"][idx] = 0.32 if 210.0 <= time_ms <= 280.0 else 0.05
    event_scores["phone_change"][0] = 0.98
    acoustic_scores["silence_likelihood"][0] = 0.92
    acoustic_scores["voicing"][0] = 0.05
    posterior = FramePosterior(
        wav_path="ko-liquid.wav",
        times_ms=times,
        class_probs=class_probs,
        event_scores=event_scores,
        acoustic_scores=acoustic_scores,
    )
    rows = [OtoTemplateRow("ko-liquid.wav", "a r", OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))]

    anchors = assign_template_row_anchors(
        posterior,
        [
            {
                "label": "phone_change",
                "selected_time_ms": 0.0,
                "time_ms": 0.0,
                "score": 0.98,
                "expected_phone": "r",
                "expected_phone_index": 1,
                "slot_index": 0,
                "frame_index": 0,
                "source": "filename_hsmm",
            }
        ],
        rows,
        expected_phones=["a", "r", "a"],
        use_source_timing_prior=False,
    )

    assert anchors[0] is not None
    assert anchors[0].anchor_abs_ms > 100.0
    assert "leading_sonorant_noise_candidate_rejected" in anchors[0].warnings
    assert "synthetic_anchor:no_candidate" in anchors[0].warnings


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


def test_cvvc_template_timeline_slots_sort_cv_first_rows_by_phone_order():
    lines = [
        "_cvfirst.wav=- \u304b,0,0,0,0,0",
        "_cvfirst.wav=\u304d,0,0,0,0,0",
        "_cvfirst.wav=\u304f,0,0,0,0,0",
        "_cvfirst.wav=\u3051,0,0,0,0,0",
        "_cvfirst.wav=\u3053,0,0,0,0,0",
        "_cvfirst.wav=\u304b,0,0,0,0,0",
        "_cvfirst.wav=a k,0,0,0,0,0",
        "_cvfirst.wav=i k,0,0,0,0,0",
        "_cvfirst.wav=u k,0,0,0,0,0",
        "_cvfirst.wav=e k,0,0,0,0,0",
        "_cvfirst.wav=o k,0,0,0,0,0",
        "_cvfirst.wav=n k,0,0,0,0,0",
    ]
    rows = []
    for line in lines:
        row = parse_template_oto_line(line)
        assert row is not None
        rows.append(row)
    phones = ["k", "a", "k", "i", "k", "u", "k", "e", "k", "o", "k", "a", "n", "k", "a"]

    row_slots = expected_slots_for_template_rows(rows, phones)
    assert [slot.phone_index for slot in row_slots[:6]] == [1, 3, 5, 7, 9, 11]

    timeline_slots = timeline_expected_slots_for_template_rows(rows, phones)
    assert [(slot.phone_index, slot.event_label) for slot in timeline_slots] == [
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
    assert [slot.slot_index for slot in timeline_slots] == list(range(len(timeline_slots)))


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
    # The cutoff must not extend into the next CV's vowel body (1796.0),
    # though the exact value depends on the VC role profile (pre/cons_gap).
    assert absolute["cutoff_abs"] < 1796.0


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

    assert adapted.timing.offset == pytest.approx(320.0)
    assert adapted.timing.preutterance == pytest.approx(200.0)
    assert adapted.timing.overlap == pytest.approx(50.0)
    assert adapted.timing.consonant == pytest.approx(230.0)
    assert adapted.timing.cutoff == pytest.approx(-(adapted.timing.consonant + 26.0))
    assert not any(warning.startswith("cvvc_vc_left_context_pre:") for warning in adapted.warnings)


def test_cvvc_hsmm_vc_bootstrap_keeps_boundary_with_usable_parameter_profile():
    adapted = bootstrap_row(
        "v.wav",
        "a k",
        OtoAnchor(
            anchor_abs_ms=520.0,
            score=0.82,
            role="vc",
            source_event_label="phone_change",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=1000.0,
        config=OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    absolute = adapted.to_json_dict()["absolute"]
    assert absolute["preutterance_abs"] == pytest.approx(475.0)
    assert adapted.timing.preutterance == pytest.approx(200.0)
    assert adapted.timing.overlap == pytest.approx(50.0)
    assert adapted.timing.consonant == pytest.approx(230.0)
    assert "hsmm_anchor_lead:520.0->475.0" in adapted.warnings


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


def test_japanese_cvvc_following_yoon_repair_can_use_unrepaired_yoon_vc_anchor():
    anchor_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="u gy",
        timing=OtoTiming(offset=1910.0, consonant=125.0, cutoff=-151.0, preutterance=85.0, overlap=50.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u304e\u3047",
        timing=OtoTiming(offset=2360.0, consonant=190.0, cutoff=-530.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    romaji_vcv = AdaptedOtoRow(
        wav="case.wav",
        alias="gy e",
        timing=OtoTiming(offset=2360.0, consonant=190.0, cutoff=-530.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    _previous, repaired_kana, repaired_romaji = repair_cvvc_row_sequence(
        [anchor_vc, late_cv, romaji_vcv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_kana.timing.offset == pytest.approx(1995.0)
    assert repaired_kana.timing.preutterance == pytest.approx(70.0)
    assert repaired_kana.timing.overlap == pytest.approx(60.0)
    assert repaired_kana.timing.consonant == pytest.approx(170.0)
    assert repaired_romaji.timing == repaired_kana.timing
    assert "cvvc_following_yoon_repair" in repaired_kana.applied_rules
    assert "cvvc_following_yoon_repair" in repaired_romaji.applied_rules


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


def test_cv_bootstrap_offset_includes_leading_consonant_onset():
    # Regression: when the consonant runs longer than the fixed CV pre target the
    # offset used to land after the consonant onset and clip its attack. With a
    # known consonant onset the offset must reach back to include it.
    anchor = OtoAnchor(
        anchor_abs_ms=395.0,        # vowel onset (preutterance target)
        score=0.8,
        role="cv_boundary",
        consonant_onset_abs_ms=300.0,  # 95 ms consonant -> longer than the pre target
        vowel_end_abs_ms=900.0,
        warnings=("slot_decoded_event",),
    )
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        anchor,
        file_duration_ms=1200.0,
        config=OtoAdapterConfig(language="japanese", format_type="cv", alias_type="cv", pre_target_ms=60.0),
    )

    # Offset (minus a small lead) must sit at or before the consonant onset.
    assert adapted.timing.offset <= 300.0
    assert adapted.timing.offset == pytest.approx(288.0, abs=1.0)
    assert adapted.timing.preutterance == pytest.approx(395.0 - adapted.timing.offset, abs=0.5)
    assert any(warning.startswith("consonant_onset_included:") for warning in adapted.warnings)


def test_cv_bootstrap_offset_unchanged_when_consonant_already_inside():
    # A short consonant already covered by the pre target must not move the offset.
    anchor = OtoAnchor(
        anchor_abs_ms=395.0,
        score=0.8,
        role="cv_boundary",
        consonant_onset_abs_ms=360.0,  # 35 ms consonant, well inside the window
        vowel_end_abs_ms=900.0,
        warnings=("slot_decoded_event",),
    )
    adapted = bootstrap_row(
        "ka.wav",
        "ka",
        anchor,
        file_duration_ms=1200.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="cv"),
    )

    assert not any(warning.startswith("consonant_onset_included:") for warning in adapted.warnings)


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

    # Profile recalibrated 2026-07-14 to the DiKORVCV reclist template medians
    # (pre 176 / ovl_gap 64 / cons_gap 115) after the anchor-lead removal.
    assert adapted.timing.preutterance == pytest.approx(176.0)
    assert adapted.timing.overlap == pytest.approx(112.0)
    assert adapted.timing.consonant == pytest.approx(291.0)


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

    assert adapted.timing.preutterance == pytest.approx(117.0)
    assert adapted.timing.overlap == pytest.approx(94.0)
    assert adapted.timing.consonant == pytest.approx(220.0)


def test_korean_vcv_bootstrap_applies_no_hsmm_anchor_lead():
    # Korean VCV anchor leads were recalibrated to 0 (2026-07-14): decoded
    # anchors already sit at the vowel onset, so any lead double-compensates
    # and drags preutterance into the onset consonant/breath.
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

    assert hsmm.timing.offset == pytest.approx(444.0)
    assert runtime.timing.offset == pytest.approx(444.0)
    assert hsmm.timing.cutoff == pytest.approx(runtime.timing.cutoff)
    assert not any(str(w).startswith("hsmm_anchor_lead:") for w in hsmm.warnings)


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


def test_japanese_cv_nucleus_slot_refines_to_acoustic_cv_boundary():
    posterior = FramePosterior(
        wav_path="case.wav",
        times_ms=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        class_probs={
            "silence": [0.0] * 7,
            "consonant": [0.1, 0.2, 0.7, 0.3, 0.1, 0.0, 0.0],
            "vowel": [0.0, 0.1, 0.2, 0.8, 0.9, 0.9, 0.8],
            "other": [0.0] * 7,
        },
        event_scores={
            "cv_boundary": [0.0, 0.1, 0.3, 0.95, 0.4, 0.1, 0.0],
            "vowel_nucleus": [0.0, 0.0, 0.1, 0.3, 0.6, 0.95, 0.4],
            "phone_change": [0.0, 0.1, 0.4, 0.5, 0.2, 0.1, 0.0],
        },
        acoustic_scores={
            "transition_likelihood": [0.0, 0.1, 0.4, 0.9, 0.3, 0.1, 0.0],
            "nucleus_likelihood": [0.0, 0.0, 0.1, 0.3, 0.6, 0.95, 0.4],
            "vowel_boundary_likelihood": [0.0, 0.1, 0.3, 0.9, 0.4, 0.1, 0.0],
            "voicing": [0.0, 0.1, 0.2, 0.8, 0.9, 0.9, 0.8],
        },
    )
    rows = [
        OtoTemplateRow(
            wav="case.wav",
            alias="ka",
            timing=OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0),
        )
    ]

    anchors = assign_template_row_anchors(
        posterior,
        (
            {
                "label": "vowel_nucleus",
                "selected_time_ms": 500.0,
                "score": 0.95,
                "expected_phone": "a",
                "expected_phone_index": 1,
                "slot_index": 0,
                "source": "filename_hsmm",
            },
        ),
        rows,
        expected_phones=("k", "a"),
        language="japanese",
    )

    assert anchors[0] is not None
    assert anchors[0].anchor_abs_ms == pytest.approx(300.0)
    assert anchors[0].source_event_label == "cv_boundary"
    assert any(
        warning == "cv_boundary_from_nucleus_refine:500.0->300.0"
        for warning in anchors[0].warnings
    )


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


def test_japanese_vcv_local_refined_hsmm_anchor_does_not_apply_second_lead():
    adapted = bootstrap_row(
        "vcv.wav",
        "a ti",
        OtoAnchor(
            anchor_abs_ms=500.0,
            score=0.78,
            refined_from_abs_ms=620.0,
            local_refine_delta_ms=-120.0,
            vowel_end_abs_ms=980.0,
            source_event_label="vowel_nucleus",
            warnings=(
                "slot_decoded_event",
                "event_source:filename_hsmm",
                "local_refined_anchor:620.0->500.0",
                "vowel_boundary_local_refine",
            ),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="japanese", format_type="vcv", alias_type="vcv"),
    )

    assert adapted.timing.offset == pytest.approx(350.0)
    assert "hsmm_anchor_lead_skipped_after_local_refine" in adapted.warnings
    assert not any(warning.startswith("hsmm_anchor_lead:") for warning in adapted.warnings)


def test_japanese_cvvc_vv_local_refined_hsmm_anchor_does_not_apply_second_lead():
    adapted = bootstrap_row(
        "cvvc.wav",
        "a i",
        OtoAnchor(
            anchor_abs_ms=710.0,
            score=0.78,
            refined_from_abs_ms=790.0,
            local_refine_delta_ms=-80.0,
            vowel_end_abs_ms=940.0,
            source_event_label="vv_boundary",
            warnings=(
                "slot_decoded_event",
                "event_source:filename_hsmm",
                "local_refined_anchor:790.0->710.0",
                "vowel_boundary_local_refine",
            ),
        ),
        file_duration_ms=1400.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="vv"),
    )

    assert adapted.timing.offset + adapted.timing.preutterance == pytest.approx(710.0)
    assert "hsmm_anchor_lead_skipped_after_local_refine" in adapted.warnings
    assert not any(warning.startswith("hsmm_anchor_lead:") for warning in adapted.warnings)


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

    # The cv_head CVVC profile uses a short pre (~30ms), so offset ≈ anchor - 30.
    assert adapted.timing.offset == pytest.approx(200.0, abs=5.0)
    assert adapted.timing.consonant < abs(adapted.timing.cutoff)
    # Cutoff is still clamped to prevent excessive tail overshoot.
    assert abs(adapted.timing.cutoff) > adapted.timing.consonant


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


def test_japanese_cvvc_initial_vowel_cv_head_repairs_late_nasal_target():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- e",
        timing=OtoTiming(offset=610.0, consonant=160.0, cutoff=-348.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=730.0,
            score=0.72,
            vowel_start_abs_ms=430.0,
            expected_phone_index=1,
            expected_phone="n",
            source_event_label="phone_change",
        ),
        mode="bootstrap",
    )
    next_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="e n",
        timing=OtoTiming(offset=690.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=770.0,
            score=0.72,
            vowel_start_abs_ms=430.0,
            expected_phone_index=1,
            expected_phone="n",
            source_event_label="phone_change",
        ),
        mode="bootstrap",
    )

    repaired, _next_vc = repair_cvvc_row_sequence(
        [initial, next_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(350.0)
    assert repaired.timing.preutterance == pytest.approx(80.0)
    assert repaired.timing.consonant == pytest.approx(220.0)
    assert "cvvc_initial_vowel_cv_head_onset_repair" in repaired.applied_rules


def test_japanese_cvvc_initial_vowel_first_v_seeds_pure_vowel_jump_cascade():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- i1",
        timing=OtoTiming(offset=400.0, consonant=220.0, cutoff=-320.0, preutterance=80.0, overlap=40.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_first_v = AdaptedOtoRow(
        wav="case.wav",
        alias="i2",
        timing=OtoTiming(offset=1190.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )
    late_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="i n",
        timing=OtoTiming(offset=1240.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_n = AdaptedOtoRow(
        wav="case.wav",
        alias="n2",
        timing=OtoTiming(offset=1267.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )

    _initial, repaired_first_v, repaired_vc, repaired_n = repair_cvvc_row_sequence(
        [initial, late_first_v, late_vc, late_n],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_first_v.timing.offset == pytest.approx(510.0)
    assert repaired_first_v.timing.preutterance == pytest.approx(90.0)
    assert repaired_first_v.timing.overlap == pytest.approx(45.0)
    assert repaired_first_v.timing.consonant == pytest.approx(100.0)
    assert repaired_vc.timing.offset == pytest.approx(760.0)
    assert repaired_vc.timing.preutterance == pytest.approx(110.0)
    assert repaired_n.timing.offset == pytest.approx(870.0)
    assert "cvvc_initial_vowel_first_v_repair" in repaired_first_v.applied_rules
    assert "cvvc_pure_vowel_jump_repair" in repaired_vc.applied_rules
    assert "cvvc_pure_vowel_jump_v_repair" in repaired_n.applied_rules


def test_japanese_cvvc_initial_vowel_first_v_keeps_close_row():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- i1",
        timing=OtoTiming(offset=400.0, consonant=220.0, cutoff=-320.0, preutterance=80.0, overlap=40.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    close_first_v = AdaptedOtoRow(
        wav="case.wav",
        alias="i2",
        timing=OtoTiming(offset=500.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )
    transition = AdaptedOtoRow(
        wav="case.wav",
        alias="i n",
        timing=OtoTiming(offset=760.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_n = AdaptedOtoRow(
        wav="case.wav",
        alias="n2",
        timing=OtoTiming(offset=870.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )

    _initial, repaired_first_v, repaired_transition, repaired_n = repair_cvvc_row_sequence(
        [initial, close_first_v, transition, following_n],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_first_v.timing == close_first_v.timing
    assert repaired_transition.timing == transition.timing
    assert repaired_n.timing == following_n.timing
    assert "cvvc_initial_vowel_first_v_repair" not in repaired_first_v.applied_rules


def test_japanese_cvvc_initial_consonant_cv_head_uses_following_cv_lead():
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

    assert repaired.timing.offset == pytest.approx(882.0)
    assert repaired.timing.preutterance == pytest.approx(35.0)
    assert repaired.timing.overlap == pytest.approx(10.0)
    assert repaired.timing.consonant == pytest.approx(65.0)
    assert repaired.timing.cutoff == pytest.approx(-95.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_following_cv_reference:")
        for warning in repaired.warnings
    )
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_following_cv_lead:")
        for warning in repaired.warnings
    )
    assert not any(
        warning.startswith("cvvc_initial_consonant_cv_head_shift_capped:")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_keeps_lead_on_large_shift():
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

    assert repaired.timing.offset == pytest.approx(1140.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert not any(warning.startswith("cvvc_initial_consonant_cv_head_shift_capped:") for warning in repaired.warnings)


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

    assert repaired.timing.offset == pytest.approx(300.0)
    assert repaired.timing.preutterance == pytest.approx(35.0)
    assert repaired.timing.overlap == pytest.approx(10.0)
    assert repaired.timing.consonant == pytest.approx(65.0)
    assert repaired.timing.cutoff == pytest.approx(-95.0)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_following_cv_reference:- \u306f")
        for warning in repaired.warnings
    )


def test_japanese_cvvc_initial_consonant_cv_head_keeps_plausible_numeric_suffix_offset():
    initial = AdaptedOtoRow(
        wav="case.wav",
        alias="- h1",
        timing=OtoTiming(offset=360.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    following_cv = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3075\u3041",
        timing=OtoTiming(offset=980.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )

    repaired, _following_cv = repair_cvvc_row_sequence(
        [initial, following_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing == initial.timing
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" not in repaired.applied_rules


def test_japanese_cvvc_initial_consonant_cv_head_keeps_small_following_cv_gap(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
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

    assert repaired.timing.offset == pytest.approx(initial.timing.offset)
    assert repaired.timing.consonant == pytest.approx(174.84)
    assert repaired.timing.preutterance == pytest.approx(108.0)
    assert repaired.timing.overlap == pytest.approx(74.52)
    assert "cvvc_initial_consonant_cv_head_following_cv_repair" not in repaired.applied_rules
    assert "cvvc_cv_head_role_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_initial_consonant_cv_head_profile_repairs_anchor_row_without_moving_offset():
    row = AdaptedOtoRow(
        wav="case.wav",
        alias="- m",
        timing=OtoTiming(offset=932.0, consonant=160.0, cutoff=-168.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1052.0,
            score=0.8,
            role="cv_head",
            source_event_label="phone_change",
        ),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired.timing.offset == pytest.approx(932.0)
    assert repaired.timing.preutterance == pytest.approx(35.0)
    assert repaired.timing.overlap == pytest.approx(10.0)
    assert repaired.timing.consonant == pytest.approx(65.0)
    assert repaired.timing.cutoff == pytest.approx(-95.0)
    assert "cvvc_initial_consonant_cv_head_profile_repair" in repaired.applied_rules
    assert any(
        warning.startswith("cvvc_initial_consonant_cv_head_profile_repaired:")
        for warning in repaired.warnings
    )


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


def test_japanese_cvvc_pure_vowel_sequence_skips_grid_when_it_crosses_hsmm_anchors():
    initial = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="- a",
        timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=280.0, score=0.55, role="cv_head"),
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a a",
        timing=OtoTiming(offset=610.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=770.0, score=0.55, role="vv"),
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a i",
        timing=OtoTiming(offset=1180.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1340.0, score=0.55, role="vv"),
        mode="bootstrap",
    )
    vv_a2 = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="i a",
        timing=OtoTiming(offset=1650.0, consonant=160.0, cutoff=-900.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1810.0, score=0.55, role="vv"),
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3980.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3060.0, score=0.55, role="vcv"),
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_head, repaired_a, repaired_i, repaired_a2, _terminal = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, vv_a2, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing.offset == pytest.approx(160.0)
    assert repaired_a.timing.offset == pytest.approx(610.0)
    assert repaired_i.timing.offset == pytest.approx(1180.0)
    assert repaired_a2.timing.offset == pytest.approx(1650.0)
    for repaired in (repaired_head, repaired_a, repaired_i, repaired_a2):
        assert "cvvc_pure_vowel_sequence_row_repair" not in repaired.applied_rules
        assert "cvvc_pure_vowel_sequence_head_repair" not in repaired.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_overrides_low_confidence_anchors():
    initial = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="- a",
        timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=280.0,
            score=0.55,
            role="cv_head",
            local_refine_margin=0.01,
            warnings=("local_refine_low_margin",),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a a",
        timing=OtoTiming(offset=610.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=770.0,
            score=0.55,
            role="vv",
            local_refine_margin=0.01,
            warnings=("local_refine_low_margin",),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a i",
        timing=OtoTiming(offset=1180.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1340.0,
            score=0.55,
            role="vv",
            boundary_confidence=0.25,
            warnings=("low_boundary_confidence:0.250",),
        ),
        mode="bootstrap",
        warnings=("low_boundary_confidence:0.250",),
    )
    vv_a2 = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="i a",
        timing=OtoTiming(offset=1650.0, consonant=160.0, cutoff=-900.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1810.0,
            score=0.55,
            role="vv",
            warnings=("synthetic_anchor:no_candidate",),
        ),
        mode="bootstrap",
        warnings=("synthetic_anchor:no_candidate",),
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3980.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3060.0, score=0.55, role="vcv"),
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_head, repaired_a, repaired_i, repaired_a2, _terminal = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, vv_a2, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing.offset == pytest.approx(2430.0)
    assert repaired_a.timing.offset == pytest.approx(2550.0)
    assert repaired_i.timing.offset == pytest.approx(3050.0)
    assert repaired_a2.timing.offset == pytest.approx(3550.0)
    for repaired in (repaired_a, repaired_i, repaired_a2):
        assert repaired.timing.preutterance == pytest.approx(300.0)
        assert repaired.timing.overlap == pytest.approx(100.0)
        assert "cvvc_pure_vowel_sequence_row_repair" in repaired.applied_rules
    assert "cvvc_pure_vowel_sequence_head_repair" in repaired_head.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_uses_filename_hsmm_grid_when_terminal_grid_is_late():
    initial = AdaptedOtoRow(
        wav="_o-u-n-a-n-n-u.wav",
        alias="- o",
        timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=280.0,
            score=0.35,
            role="cv_head",
            source_event_label="vv_boundary",
            warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    vv_u = AdaptedOtoRow(
        wav="_o-u-n-a-n-n-u.wav",
        alias="o u",
        timing=OtoTiming(offset=590.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=900.0,
            score=0.35,
            role="vv",
            source_event_label="vv_boundary",
            warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    u_n = AdaptedOtoRow(
        wav="_o-u-n-a-n-n-u.wav",
        alias="u n",
        timing=OtoTiming(offset=820.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1120.0,
            score=0.35,
            role="vv",
            source_event_label="vv_boundary",
            warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    n_a = AdaptedOtoRow(
        wav="_o-u-n-a-n-n-u.wav",
        alias="n a",
        timing=OtoTiming(offset=1260.0, consonant=160.0, cutoff=-900.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=1560.0,
            score=0.35,
            role="vcv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
        ),
        mode="bootstrap",
        warnings=("local_refine_low_margin",),
    )
    terminal = AdaptedOtoRow(
        wav="_o-u-n-a-n-n-u.wav",
        alias="a -",
        timing=OtoTiming(offset=3950.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2612.0, score=0.5, role="vcv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_head, repaired_u, repaired_n, repaired_a, _terminal = repair_cvvc_row_sequence(
        [initial, vv_u, u_n, n_a, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_head.timing.offset == pytest.approx(160.0)
    assert repaired_head.timing.cutoff == pytest.approx(-360.0)
    assert repaired_u.timing.offset == pytest.approx(600.0)
    assert repaired_n.timing.offset == pytest.approx(820.0)
    assert repaired_a.timing.offset == pytest.approx(1260.0)
    for repaired in (repaired_u, repaired_n, repaired_a):
        assert repaired.timing.preutterance == pytest.approx(300.0)
        assert repaired.timing.overlap == pytest.approx(100.0)
        assert repaired.timing.consonant == pytest.approx(420.0)
        assert repaired.timing.cutoff == pytest.approx(-600.0)
        assert "cvvc_pure_vowel_sequence_row_repair" in repaired.applied_rules
        assert any("filename_hsmm" in warning for warning in repaired.warnings)
    assert "cvvc_pure_vowel_sequence_head_cutoff_repair" in repaired_head.applied_rules


def test_japanese_cvvc_pure_vowel_sequence_terminal_cadence_overrides_early_filename_hsmm():
    wav = "_a-a-i-a-u-a-e.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias="- \u3042",
            timing=OtoTiming(offset=140.0, consonant=160.0, cutoff=-360.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=260.0,
                score=0.35,
                role="cv_head",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="a \u3042",
            timing=OtoTiming(offset=590.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=890.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="a \u3044",
            timing=OtoTiming(offset=1100.0, consonant=160.0, cutoff=-3500.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1400.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="i \u3042",
            timing=OtoTiming(offset=1160.0, consonant=160.0, cutoff=-2900.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1460.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="a \u3046",
            timing=OtoTiming(offset=1680.0, consonant=160.0, cutoff=-2500.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1980.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="u \u3042",
            timing=OtoTiming(offset=2150.0, consonant=160.0, cutoff=-2100.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=2450.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="a \u3048",
            timing=OtoTiming(offset=2440.0, consonant=160.0, cutoff=-1800.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=2740.0,
                score=0.35,
                role="vv",
                source_event_label="vv_boundary",
                warnings=("event_source:filename_hsmm", "local_refine_low_margin"),
            ),
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="e -",
            timing=OtoTiming(offset=4310.0, consonant=210.0, cutoff=-1500.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=OtoAnchor(anchor_abs_ms=4420.0, score=0.55, role="vcv", source_event_label="vowel_nucleus"),
            mode="bootstrap",
            applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.offset == pytest.approx(990.0)
    assert "cvvc_pure_vowel_sequence_head_repair" in repaired[0].applied_rules
    expected_offsets = [1110.0, 1610.0, 2110.0, 2610.0, 3110.0, 3610.0]
    for row, expected in zip(repaired[1:-1], expected_offsets):
        assert row.timing.offset == pytest.approx(expected)
        assert row.timing.preutterance == pytest.approx(300.0)
        assert row.timing.overlap == pytest.approx(100.0)
        assert row.timing.consonant == pytest.approx(420.0)
        assert row.timing.cutoff == pytest.approx(-600.0)
        assert "cvvc_pure_vowel_sequence_row_repair" in row.applied_rules
        assert "cvvc_pure_vowel_sequence_terminal_reference:terminal_cadence" in row.warnings


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


def test_japanese_cvvc_pure_vowel_sequence_orders_grid_by_expected_phone_index():
    def row(alias: str, target: int) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav="_n-n-a-n.wav",
            alias=alias,
            timing=OtoTiming(offset=800.0, consonant=160.0, cutoff=-400.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=1000.0,
                score=0.0,
                role="vv",
                expected_phone_index=target,
            ),
            mode="bootstrap",
        )

    initial = row("- n", 0)
    terminal = replace(
        row("n -", 3),
        timing=OtoTiming(offset=2500.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    _head, repaired_an, repaired_nn, _terminal, repaired_na = repair_cvvc_row_sequence(
        [initial, row("a n", 3), row("n n", 1), terminal, row("n a", 2)],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert [repaired_an.timing.offset, repaired_nn.timing.offset, repaired_na.timing.offset] == pytest.approx(
        [2070.0, 1070.0, 1570.0]
    )


def test_japanese_cvvc_pure_vowel_sequence_requires_terminal_anchor_rule(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
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

    assert repaired_head.timing.offset == pytest.approx(initial.timing.offset)
    assert repaired_head.timing.consonant == pytest.approx(174.84)
    assert repaired_head.timing.preutterance == pytest.approx(108.0)
    assert repaired_head.timing.overlap == pytest.approx(74.52)
    assert "cvvc_cv_head_role_profile_repair" in repaired_head.applied_rules
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
    assert repaired_aa.timing.offset == pytest.approx(1290.0)
    assert repaired_a.timing.offset == pytest.approx(1540.0)
    assert repaired_ai.timing.offset == pytest.approx(1790.0)
    assert repaired_r.timing.offset == pytest.approx(4800.0)
    assert repaired_aa.timing.preutterance == pytest.approx(300.0)
    assert repaired_aa.timing.overlap == pytest.approx(100.0)
    assert repaired_a.timing.preutterance == pytest.approx(0.0)
    assert repaired_a.timing.overlap == pytest.approx(0.0)
    assert "cvvc_pure_vowel_onset_head_repair" in repaired_head.applied_rules
    assert "cvvc_pure_vowel_onset_head_cutoff_repair" in repaired_head.applied_rules
    assert "cvvc_pure_vowel_onset_transition_repair" in repaired_aa.applied_rules
    assert "cvvc_pure_vowel_onset_v_repair" in repaired_a.applied_rules
    assert "cvvc_pure_vowel_onset_trailing_r_repair" in repaired_r.applied_rules


def test_japanese_cvvc_pure_vowel_onset_sequence_shortens_pitch_suffix_terminal_tail():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=1160.0,
        vowel_end_abs_ms=4900.0,
    )
    initial = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="- a_S",
        timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vv_a = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="a a_S",
        timing=OtoTiming(offset=600.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    vv_i = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="a i_S",
        timing=OtoTiming(offset=1240.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-a-i.wav",
        alias="i -_S",
        timing=OtoTiming(offset=3045.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=reliable_anchor,
        mode="bootstrap",
    )

    repaired_head, repaired_aa, repaired_ai, repaired_terminal = repair_cvvc_row_sequence(
        [initial, vv_a, vv_i, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert repaired_head.timing.offset == pytest.approx(1040.0)
    assert repaired_aa.timing.offset == pytest.approx(1290.0)
    assert repaired_ai.timing.offset == pytest.approx(1790.0)
    assert repaired_terminal.timing.offset == pytest.approx(2135.0)
    assert repaired_terminal.timing.consonant == pytest.approx(260.0)
    assert repaired_terminal.timing.cutoff == pytest.approx(-430.0)
    assert repaired_terminal.timing.preutterance == pytest.approx(180.0)
    assert repaired_terminal.timing.overlap == pytest.approx(100.0)
    assert "cvvc_pure_vowel_onset_pitch_terminal_repair" in repaired_terminal.applied_rules


def test_japanese_cvvc_repair_dispatch_detects_targeted_family_shapes():
    cfg = OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto")

    def row(wav: str, alias: str) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset=100.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )

    unprefixed_pitch = _ja_cvvc_repair_dispatch(
        [
            row("gya-gyu-gye-gyo-gya.wav", "gyaF4S"),
            row("gya-gyu-gye-gyo-gya.wav", "gyuF4S"),
            row("gya-gyu-gye-gyo-gya.wav", "gyeF4S"),
            row("gya-gyu-gye-gyo-gya.wav", "gyoF4S"),
        ],
        cfg,
    )
    assert unprefixed_pitch.unprefixed_wav is True
    assert unprefixed_pitch.has_unprefixed_pitch_suffix is True
    assert unprefixed_pitch.has_unprefixed_pitch_yoon_first_gap is True
    assert unprefixed_pitch.has_headless_pitch_suffix is False

    headless_soft = _ja_cvvc_repair_dispatch(
        [
            row("_gya-gyu-gye-gyo-gya.wav", "gya_S"),
            row("_gya-gyu-gye-gyo-gya.wav", "gyu_S"),
            row("_gya-gyu-gye-gyo-gya.wav", "gye_S"),
            row("_gya-gyu-gye-gyo-gya.wav", "gyo_S"),
        ],
        cfg,
    )
    assert headless_soft.underscore_wav is True
    assert headless_soft.has_headless_pitch_suffix is True
    assert headless_soft.has_headless_soft_yoon_only is True
    assert headless_soft.has_unprefixed_pitch_suffix is False


def test_japanese_cvvc_repair_dispatch_detects_vc_block_and_n_glide_only():
    cfg = OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto")

    def row(wav: str, alias: str) -> AdaptedOtoRow:
        return AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset=100.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )

    grouped_vc = _ja_cvvc_repair_dispatch(
        [
            row("ba-bi-bu.wav", "a b"),
            row("ba-bi-bu.wav", "i b"),
            row("ba-bi-bu.wav", "ba"),
            row("ba-bi-bu.wav", "bi"),
        ],
        cfg,
    )
    assert grouped_vc.starts_with_vc_block is True
    assert grouped_vc.has_tail_moraic_n_glide_vc is False
    assert grouped_vc.has_unprefixed_pitch_suffix is False

    n_glide = _ja_cvvc_repair_dispatch(
        [
            row("_na-nya.wav", "na"),
            row("_na-nya.wav", "a y"),
            row("_na-nya.wav", "n y"),
        ],
        cfg,
    )
    assert n_glide.starts_with_vc_block is False
    assert n_glide.has_tail_moraic_n_glide_vc is True

    obstruent_head = _ja_cvvc_repair_dispatch(
        [
            row("sa-si-su-se-so.wav", "- sa"),
            row("sa-si-su-se-so.wav", "a s"),
            row("sa-si-su-se-so.wav", "si"),
        ],
        cfg,
    )
    assert obstruent_head.unprefixed_obstruent_head is True


def test_japanese_cvvc_unprefixed_pitch_suffix_terminal_uses_late_tail_grid():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=620.0,
        vowel_end_abs_ms=3720.0,
    )
    rows = [
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="- aF4S",
            timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a aF4S",
            timing=OtoTiming(offset=610.0, consonant=160.0, cutoff=-3030.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a iF4S",
            timing=OtoTiming(offset=1180.0, consonant=160.0, cutoff=-2460.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="i aF4S",
            timing=OtoTiming(offset=1700.0, consonant=160.0, cutoff=-1940.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a uF4S",
            timing=OtoTiming(offset=1840.0, consonant=160.0, cutoff=-1800.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="u aF4S",
            timing=OtoTiming(offset=2460.0, consonant=160.0, cutoff=-1180.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a eF4S",
            timing=OtoTiming(offset=2680.0, consonant=160.0, cutoff=-960.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="e -F4S",
            timing=OtoTiming(offset=2880.0, consonant=160.0, cutoff=-760.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
    ]

    *_, repaired_terminal = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3800.0,
    )

    assert repaired_terminal.timing.offset == pytest.approx(3330.0)
    assert repaired_terminal.timing.consonant == pytest.approx(260.0)
    assert repaired_terminal.timing.cutoff == pytest.approx(-430.0)
    assert repaired_terminal.timing.preutterance == pytest.approx(180.0)
    assert repaired_terminal.timing.overlap == pytest.approx(100.0)
    assert "cvvc_pure_vowel_onset_pitch_terminal_repair" in repaired_terminal.applied_rules


def test_japanese_cvvc_unprefixed_pitch_suffix_terminal_keeps_already_late_tail():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=620.0,
        vowel_end_abs_ms=3720.0,
    )
    rows = [
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="- aF4S",
            timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a iF4S",
            timing=OtoTiming(offset=2100.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="i eF4S",
            timing=OtoTiming(offset=2680.0, consonant=160.0, cutoff=-960.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="e -F4S",
            timing=OtoTiming(offset=3200.0, consonant=160.0, cutoff=-760.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
    ]

    *_, repaired_terminal = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3800.0,
    )

    assert repaired_terminal.timing.offset == pytest.approx(rows[-1].timing.offset)
    assert repaired_terminal.timing.consonant == pytest.approx(rows[-1].timing.consonant)
    assert repaired_terminal.timing.preutterance == pytest.approx(rows[-1].timing.preutterance)
    assert repaired_terminal.timing.overlap == pytest.approx(rows[-1].timing.overlap)
    assert "cvvc_pure_vowel_onset_pitch_terminal_repair" not in repaired_terminal.applied_rules


def test_japanese_cvvc_unprefixed_pitch_terminal_keeps_plain_terminal_anchor():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=620.0,
        vowel_end_abs_ms=3720.0,
    )
    rows = [
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="- a",
            timing=OtoTiming(offset=160.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="a i",
            timing=OtoTiming(offset=2100.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="i e",
            timing=OtoTiming(offset=2680.0, consonant=160.0, cutoff=-960.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="a-a-i-a-u-a-e.wav",
            alias="e -",
            timing=OtoTiming(offset=3300.0, consonant=210.0, cutoff=-520.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
            applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
        ),
    ]

    *_, repaired_terminal = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3800.0,
    )

    assert repaired_terminal.timing == rows[-1].timing
    assert repaired_terminal.applied_rules == rows[-1].applied_rules


def test_japanese_cvvc_pure_vowel_onset_u_head_delays_first_transition_only():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=990.0,
        vowel_end_abs_ms=4900.0,
    )
    rows = [
        AdaptedOtoRow(
            wav="_u-u-e-u.wav",
            alias="- u_S",
            timing=OtoTiming(offset=640.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_u-u-e-u.wav",
            alias="u u_S",
            timing=OtoTiming(offset=600.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_u-u-e-u.wav",
            alias="u e_S",
            timing=OtoTiming(offset=1240.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_u-u-e-u.wav",
            alias="e u_S",
            timing=OtoTiming(offset=1740.0, consonant=160.0, cutoff=-1200.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_u-u-e-u.wav",
            alias="u -_S",
            timing=OtoTiming(offset=3045.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in repaired[1:4]] == pytest.approx([1260.0, 1620.0, 2120.0])
    assert "cvvc_pure_vowel_onset_head_phone_first_transition_repair" in repaired[1].applied_rules
    assert "cvvc_pure_vowel_onset_transition_repair" in repaired[2].applied_rules


def test_japanese_cvvc_pure_vowel_onset_moraic_n_alternation_uses_compact_cadence():
    reliable_anchor = OtoAnchor(
        anchor_abs_ms=1400.0,
        score=0.4,
        role="vv",
        vowel_start_abs_ms=1040.0,
        vowel_end_abs_ms=4900.0,
    )
    aliases = ["- n_S", "n n_S", "n a_S", "a n_S", "n i_S", "i n_S", "n u_S", "u -_S"]
    rows = [
        AdaptedOtoRow(
            wav="_n-n-a-n-i-n-u.wav",
            alias=alias,
            timing=OtoTiming(offset=600.0 + (index * 120.0), consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=reliable_anchor,
            mode="bootstrap",
        )
        for index, alias in enumerate(aliases)
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in repaired[1:7]] == pytest.approx(
        [1170.0, 1550.0, 2170.0, 2550.0, 3290.0, 3670.0]
    )
    assert repaired[-1].timing.offset == pytest.approx(4015.0)
    for row in repaired[1:7]:
        assert "cvvc_pure_vowel_onset_n_alternation_repair" in row.applied_rules


def test_japanese_cvvc_tail_moraic_n_following_glide_repairs_backtracked_transition():
    aliases_offsets = [
        ("- yF4S", 390.0),
        ("a yF4S", 485.0),
        ("u yF4S", 765.0),
        ("e yF4S", 1325.0),
        ("o yF4S", 2085.0),
        ("i yF4S", 2905.0),
        ("n yF4S", 2410.0),
        ("\u3086F4S", 826.5),
        ("\u3044\u3047F4S", 705.0),
        ("\u3088F4S", 1200.0),
        ("\u3084F4S", 1710.0),
    ]
    rows = [
        AdaptedOtoRow(
            wav="ya-yu-ye-yo-ya-i-ya-N-ya.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=185.0 if alias.startswith("n ") else 120.0,
                cutoff=-205.0 if alias.startswith("n ") else -146.0,
                preutterance=135.0 if alias.startswith("n ") else 80.0,
                overlap=50.0 if alias.startswith("n ") else 45.0,
            ),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in aliases_offsets
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4281.22,
    )

    repaired_n_y = repaired[6]
    assert repaired_n_y.timing.offset == pytest.approx(3655.0)
    assert repaired_n_y.timing.preutterance == pytest.approx(170.0)
    assert repaired_n_y.timing.consonant == pytest.approx(220.0)
    assert repaired_n_y.timing.cutoff == pytest.approx(-260.0)
    assert "cvvc_tail_moraic_n_following_glide_repair" in repaired_n_y.applied_rules


def test_japanese_cvvc_tail_moraic_n_following_glide_keeps_aligned_soft_transition():
    rows = [
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in [
            ("a w_S", 1203.365),
            ("i w_S", 1703.365),
            ("e w_S", 2203.365),
            ("o w_S", 2703.365),
            ("u w_S", 3678.365),
            ("n w_S", 4678.365),
        ]
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([row.timing.offset for row in rows])
    assert all("cvvc_tail_moraic_n_following_glide_repair" not in row.applied_rules for row in repaired)


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
    assert repaired_release_r.timing.preutterance == pytest.approx(300.0)
    assert repaired_release_r.timing.overlap == pytest.approx(100.0)
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

    assert repaired_first.timing.offset == pytest.approx(1860.0)
    assert repaired_second.timing.offset == pytest.approx(2360.0)
    assert repaired_third.timing.offset == pytest.approx(2860.0)
    assert repaired_first.timing.preutterance == pytest.approx(300.0)
    assert repaired_first.timing.overlap == pytest.approx(100.0)
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


def test_japanese_cvvc_headless_vc_onset_sequence_skips_short_block_when_grid_passes_anchors():
    def anchor(anchor_abs_ms: float, next_vowel_abs_ms: float) -> OtoAnchor:
        return OtoAnchor(
            anchor_abs_ms=anchor_abs_ms,
            score=0.6,
            role="vc",
            vowel_start_abs_ms=anchor_abs_ms,
            vowel_end_abs_ms=anchor_abs_ms + 220.0,
            next_vowel_abs_ms=next_vowel_abs_ms,
        )

    first = AdaptedOtoRow(
        wav="_ho-ha-n-ha.wav",
        alias="o h",
        timing=OtoTiming(offset=540.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(910.0, 1400.0),
        mode="bootstrap",
    )
    second = AdaptedOtoRow(
        wav="_ho-ha-n-ha.wav",
        alias="n h",
        timing=OtoTiming(offset=2210.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=anchor(2340.0, 2580.0),
        mode="bootstrap",
    )

    repaired_first, repaired_second = repair_cvvc_row_sequence(
        [first, second],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert repaired_first.timing == first.timing
    assert repaired_second.timing == second.timing
    assert "cvvc_headless_vc_onset_repair" not in repaired_first.applied_rules
    assert "cvvc_headless_vc_onset_repair" not in repaired_second.applied_rules


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


def test_japanese_cvvc_cv_role_profile_repair_runs_by_default(monkeypatch):
    monkeypatch.delenv("UTOA_ENABLE_JA_CVVC_REFERENCE_REPAIRS", raising=False)
    row = AdaptedOtoRow(
        wav="_ka-ki-ku.wav",
        alias="ku",
        timing=OtoTiming(offset=1780.0, consonant=190.0, cutoff=-480.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1980.0, score=0.6, role="cv_boundary", source_event_label="cv_boundary"),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing.offset == pytest.approx(1860.0)
    assert repaired.timing.consonant == pytest.approx(150.0)
    assert repaired.timing.cutoff == pytest.approx(-360.0)
    assert repaired.timing.preutterance == pytest.approx(70.0)
    assert repaired.timing.overlap == pytest.approx(25.0)
    assert "cvvc_cv_role_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_cv_role_profile_calibrates_profile_preserving_anchor(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
    row = AdaptedOtoRow(
        wav="_ka-ki-ku.wav",
        alias="ku",
        timing=OtoTiming(offset=1780.0, consonant=190.0, cutoff=-480.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1980.0, score=0.6, role="cv_boundary", source_event_label="cv_boundary"),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing.offset == pytest.approx(1860.0)
    assert repaired.timing.consonant == pytest.approx(150.0)
    assert repaired.timing.cutoff == pytest.approx(-360.0)
    assert repaired.timing.preutterance == pytest.approx(70.0)
    assert repaired.timing.overlap == pytest.approx(25.0)
    assert "cvvc_cv_role_profile_repair" in repaired.applied_rules
    assert any(
        "cvvc_cv_role_profile_repaired:offset=1780.0->1860.0,consonant=190.0->150.0,"
        "cutoff=-480.0->-360.0,pre=150.0->70.0,overlap=115.0->25.0" in item
        for item in repaired.warnings
    )


def test_japanese_cvvc_cv_role_profile_keeps_existing_narrow_cv_profile():
    row = AdaptedOtoRow(
        wav="_rya-ryo.wav",
        alias="ryo",
        timing=OtoTiming(offset=1380.0, consonant=150.0, cutoff=-1990.0, preutterance=50.0, overlap=40.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3120.0, score=0.6, role="cv"),
        mode="bootstrap",
        applied_rules=("cvvc_following_yoon_repair",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert repaired.timing == row.timing
    assert "cvvc_cv_role_profile_repair" not in repaired.applied_rules


def test_japanese_cvvc_cv_head_role_profile_calibrates_default_profile_without_moving_offset(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
    row = AdaptedOtoRow(
        wav="_fi_fu_fe_fo.wav",
        alias="- fe",
        timing=OtoTiming(offset=1400.0, consonant=160.0, cutoff=-190.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1520.0, score=0.6, role="cv_head"),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3400.0,
    )

    assert repaired.timing.offset == pytest.approx(1400.0)
    assert repaired.timing.consonant == pytest.approx(174.84)
    assert repaired.timing.cutoff == pytest.approx(-240.0)
    assert repaired.timing.preutterance == pytest.approx(108.0)
    assert repaired.timing.overlap == pytest.approx(74.52)
    assert "cvvc_cv_head_role_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_cv_head_role_profile_keeps_existing_narrow_head_profile():
    row = AdaptedOtoRow(
        wav="_fi_fu_fe_fo.wav",
        alias="- fo",
        timing=OtoTiming(offset=3060.0, consonant=174.84, cutoff=-274.36, preutterance=108.0, overlap=74.52),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3168.0, score=0.6, role="cv_head"),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3600.0,
    )

    assert repaired.timing == row.timing
    assert "cvvc_cv_head_role_profile_repair" not in repaired.applied_rules


def test_japanese_cvvc_headed_cv_vowel_nucleus_gets_extra_lead_without_profile_change():
    head = AdaptedOtoRow(
        wav="_gya-gi-gyu.wav",
        alias="- gya",
        timing=OtoTiming(offset=460.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_cv = AdaptedOtoRow(
        wav="_gya-gi-gyu.wav",
        alias="gye",
        timing=OtoTiming(offset=2360.0, consonant=190.0, cutoff=-530.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2560.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    _head, repaired = repair_cvvc_row_sequence(
        [head, late_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing.offset == pytest.approx(1880.0)
    assert repaired.timing.consonant == pytest.approx(late_cv.timing.consonant)
    assert repaired.timing.cutoff == pytest.approx(late_cv.timing.cutoff)
    assert repaired.timing.preutterance == pytest.approx(late_cv.timing.preutterance)
    assert repaired.timing.overlap == pytest.approx(late_cv.timing.overlap)
    assert "cvvc_headed_cv_vowel_nucleus_repair" in repaired.applied_rules


def test_japanese_cvvc_headed_cv_vowel_nucleus_normalizes_oversized_bootstrap_profile():
    head = AdaptedOtoRow(
        wav="_gya-gi-gyu.wav",
        alias="- gya",
        timing=OtoTiming(offset=460.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_cv = AdaptedOtoRow(
        wav="_gya-gi-gyu.wav",
        alias="gye",
        timing=OtoTiming(offset=2360.0, consonant=240.0, cutoff=-530.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2560.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    _head, repaired = repair_cvvc_row_sequence(
        [head, late_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing.offset + repaired.timing.preutterance == pytest.approx(2030.0)
    assert repaired.timing.preutterance == pytest.approx(70.0)
    assert repaired.timing.overlap == pytest.approx(25.0)
    assert repaired.timing.consonant == pytest.approx(150.0)
    assert repaired.timing.cutoff == pytest.approx(-360.0)
    assert "cvvc_headed_cv_vowel_nucleus_repair" in repaired.applied_rules
    assert "cvvc_cv_role_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_cv_role_profile_prefers_anchor_role_over_vowel_surface_alias():
    row = AdaptedOtoRow(
        wav="wa-wi-we-wo.wav",
        alias="a",
        timing=OtoTiming(offset=2360.0, consonant=240.0, cutoff=-530.0, preutterance=150.0, overlap=115.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2560.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing == OtoTiming(
        offset=2440.0,
        consonant=150.0,
        cutoff=-360.0,
        preutterance=70.0,
        overlap=25.0,
    )
    assert "cvvc_cv_role_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_headed_cv_vowel_nucleus_stays_after_previous_cv_pre():
    head = AdaptedOtoRow(
        wav="_ka-ki-ku-ke.wav",
        alias="- ka",
        timing=OtoTiming(offset=420.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    previous_cv = AdaptedOtoRow(
        wav="_ka-ki-ku-ke.wav",
        alias="ku",
        timing=OtoTiming(offset=1780.0, consonant=190.0, cutoff=-425.0, preutterance=150.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1980.0, score=0.5, role="cv", source_event_label="cv_boundary"),
        mode="bootstrap",
    )
    late_vc = AdaptedOtoRow(
        wav="_ka-ki-ku-ke.wav",
        alias="u k",
        timing=OtoTiming(offset=2300.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2370.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ka-ki-ku-ke.wav",
        alias="ke",
        timing=OtoTiming(offset=2360.0, consonant=190.0, cutoff=-565.0, preutterance=150.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2560.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    _head, kept_previous, repaired_vc, repaired_cv = repair_cvvc_row_sequence(
        [head, previous_cv, late_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    previous_pre_abs = kept_previous.timing.offset + kept_previous.timing.preutterance
    assert repaired_cv.timing.offset == pytest.approx(previous_pre_abs + 65.0)
    assert repaired_vc.timing.offset == pytest.approx(1920.0)
    assert repaired_vc.timing.offset < repaired_cv.timing.offset
    assert repaired_vc.timing.offset + repaired_vc.timing.preutterance >= previous_pre_abs + 40.0
    assert "cvvc_headed_cv_vowel_nucleus_repair" in repaired_cv.applied_rules
    assert "cvvc_headed_cv_previous_vowel_order_guard_repair" in repaired_cv.applied_rules
    assert "cvvc_headed_regular_vc_from_next_cv_repair" in repaired_vc.applied_rules


def test_japanese_cvvc_headed_cv_vowel_nucleus_order_guard_keeps_cv_after_previous_vc_pre():
    head = AdaptedOtoRow(
        wav="_na-ni-nu.wav",
        alias="- na",
        timing=OtoTiming(offset=400.0, consonant=174.84, cutoff=-240.0, preutterance=108.0, overlap=74.52),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    previous_vc = AdaptedOtoRow(
        wav="_na-ni-nu.wav",
        alias="i q",
        timing=OtoTiming(offset=1405.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1530.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    late_cv = AdaptedOtoRow(
        wav="_na-ni-nu.wav",
        alias="nu",
        timing=OtoTiming(offset=1820.0, consonant=190.0, cutoff=-565.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2020.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    _head, _previous_vc, repaired = repair_cvvc_row_sequence(
        [head, previous_vc, late_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired.timing.offset == pytest.approx(1820.0)
    assert "cvvc_headed_cv_vowel_nucleus_repair" in repaired.applied_rules
    assert "cvvc_headed_cv_vowel_nucleus_order_guard_repair" in repaired.applied_rules


def test_japanese_cvvc_headed_cv_vowel_nucleus_keeps_unheaded_rows():
    row = AdaptedOtoRow(
        wav="gya-gi-gyu.wav",
        alias="gye",
        timing=OtoTiming(offset=2360.0, consonant=190.0, cutoff=-530.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2560.0,
            score=0.6,
            role="cv",
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )

    (kept,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert kept.timing == row.timing
    assert "cvvc_headed_cv_vowel_nucleus_repair" not in kept.applied_rules


def test_japanese_cvvc_headed_regular_vc_backfills_from_next_cv():
    head = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="- ma",
        timing=OtoTiming(offset=380.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    late_vc = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="i m",
        timing=OtoTiming(offset=1815.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1940.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="mu",
        timing=OtoTiming(offset=1690.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2130.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, repaired_vc, _next_cv = repair_cvvc_row_sequence(
        [head, late_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert repaired_vc.timing.offset == pytest.approx(1530.0)
    assert repaired_vc.timing.consonant == pytest.approx(185.0)
    assert repaired_vc.timing.cutoff == pytest.approx(-205.0)
    assert repaired_vc.timing.preutterance == pytest.approx(135.0)
    assert repaired_vc.timing.overlap == pytest.approx(50.0)
    assert "cvvc_headed_regular_vc_from_next_cv_repair" in repaired_vc.applied_rules


def test_japanese_cvvc_headed_regular_vc_preserves_direct_hsmm_anchor():
    head = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="- ma",
        timing=OtoTiming(offset=380.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    direct_vc = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="i m",
        timing=OtoTiming(offset=1815.0, consonant=230.0, cutoff=-256.0, preutterance=200.0, overlap=50.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=2060.0,
            score=0.7,
            role="vc",
            source_event_label="phone_change",
            warnings=("event_source:filename_hsmm",),
        ),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="mu",
        timing=OtoTiming(offset=1690.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1760.0, score=0.7, role="cv", source_event_label="cv_boundary"),
        mode="bootstrap",
    )

    _head, kept_vc, _next_cv = repair_cvvc_row_sequence(
        [head, direct_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert kept_vc.timing == direct_vc.timing
    assert "cvvc_headed_regular_vc_from_next_cv_repair" not in kept_vc.applied_rules
    assert "cvvc_internal_vc_slot_bound_repair" not in kept_vc.applied_rules


def test_japanese_cvvc_headed_regular_vc_respects_previous_cv_safe_window():
    head = AdaptedOtoRow(
        wav="_ta-te-tu-to.wav",
        alias="- ta",
        timing=OtoTiming(offset=420.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    previous_cv = AdaptedOtoRow(
        wav="_ta-te-tu-to.wav",
        alias="te",
        timing=OtoTiming(offset=2270.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2440.0, score=0.5, role="cv", source_event_label="cv_boundary"),
        mode="bootstrap",
    )
    late_vc = AdaptedOtoRow(
        wav="_ta-te-tu-to.wav",
        alias="e t",
        timing=OtoTiming(offset=2865.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2910.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ta-te-tu-to.wav",
        alias="to",
        timing=OtoTiming(offset=2440.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2560.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, _previous_cv, repaired_vc, _next_cv = repair_cvvc_row_sequence(
        [head, previous_cv, late_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3400.0,
    )

    assert repaired_vc.timing.offset == pytest.approx(2410.0)
    assert repaired_vc.timing.offset - previous_cv.timing.offset >= 140.0
    assert "cvvc_headed_regular_vc_from_next_cv_repair" in repaired_vc.applied_rules
    assert any(
        "cvvc_headed_regular_vc_prev_safe_window:2280.0->2410.0" in warning
        for warning in repaired_vc.warnings
    )


def test_japanese_cvvc_headed_regular_vc_keeps_safe_pre_before_next_cv():
    head = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="- ma",
        timing=OtoTiming(offset=380.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    safe_vc = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="i m",
        timing=OtoTiming(offset=1400.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1525.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="mu",
        timing=OtoTiming(offset=1600.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2040.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, kept_vc, _next_cv = repair_cvvc_row_sequence(
        [head, safe_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert kept_vc.timing == safe_vc.timing
    assert "cvvc_headed_regular_vc_from_next_cv_repair" not in kept_vc.applied_rules


def test_japanese_cvvc_headed_regular_vc_keeps_vc_already_before_next_cv():
    head = AdaptedOtoRow(
        wav="_ha-hi-hu.wav",
        alias="- h",
        timing=OtoTiming(offset=830.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="_ha-hi-hu.wav",
        alias="a h",
        timing=OtoTiming(offset=915.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1040.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ha-hi-hu.wav",
        alias="hi",
        timing=OtoTiming(offset=950.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1180.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, kept_vc, _next_cv = repair_cvvc_row_sequence(
        [head, first_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert kept_vc.timing == first_vc.timing
    assert "cvvc_headed_regular_vc_from_next_cv_repair" not in kept_vc.applied_rules


def test_japanese_cvvc_headed_regular_vc_skips_huge_terminal_backshift():
    head = AdaptedOtoRow(
        wav="_sha-shi-shu-she-sho-sha-n-shi.wav",
        alias="- sh",
        timing=OtoTiming(offset=830.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal_vc = AdaptedOtoRow(
        wav="_sha-shi-shu-she-sho-sha-n-shi.wav",
        alias="n sh",
        timing=OtoTiming(offset=4340.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=4430.0, score=0.5, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    stale_next_cv = AdaptedOtoRow(
        wav="_sha-shi-shu-she-sho-sha-n-shi.wav",
        alias="sha",
        timing=OtoTiming(offset=3550.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=3800.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, kept_vc, _next_cv = repair_cvvc_row_sequence(
        [head, terminal_vc, stale_next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5200.0,
    )

    assert kept_vc.timing == terminal_vc.timing
    assert "cvvc_headed_regular_vc_from_next_cv_repair" not in kept_vc.applied_rules


def test_japanese_cvvc_internal_vc_slot_bound_moves_first_vc_from_head():
    head = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="- ma",
        timing=OtoTiming(offset=400.0, consonant=160.0, cutoff=-260.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    early_vc = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="a m",
        timing=OtoTiming(offset=430.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=430.0, score=0.45, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="mi",
        timing=OtoTiming(offset=900.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1040.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, repaired_vc, _next_cv = repair_cvvc_row_sequence(
        [head, early_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=1800.0,
    )

    assert repaired_vc.timing.offset == pytest.approx(740.0)
    assert repaired_vc.timing.consonant == pytest.approx(early_vc.timing.consonant)
    assert "cvvc_internal_vc_slot_bound_repair" in repaired_vc.applied_rules
    assert any("cvvc_internal_vc_slot_bound_previous:- ma@400.0" in warning for warning in repaired_vc.warnings)


def test_japanese_cvvc_internal_vc_slot_bound_keeps_safe_internal_vc():
    head = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="- ma",
        timing=OtoTiming(offset=400.0, consonant=160.0, cutoff=-260.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    safe_vc = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="a m",
        timing=OtoTiming(offset=740.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=740.0, score=0.45, role="vc", source_event_label="phone_change"),
        mode="bootstrap",
    )
    next_cv = AdaptedOtoRow(
        wav="_ma-mi-mu.wav",
        alias="mi",
        timing=OtoTiming(offset=900.0, consonant=190.0, cutoff=-425.0, preutterance=70.0, overlap=25.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1040.0, score=0.5, role="cv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )

    _head, kept_vc, _next_cv = repair_cvvc_row_sequence(
        [head, safe_vc, next_cv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=1800.0,
    )

    assert kept_vc.timing == safe_vc.timing
    assert "cvvc_internal_vc_slot_bound_repair" not in kept_vc.applied_rules


def test_japanese_cvvc_obstruent_grid_repairs_late_cascade():
    rows = [
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="- s",
            timing=OtoTiming(offset=670.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="si",
            timing=OtoTiming(offset=290.0, consonant=190.0, cutoff=-430.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="i s",
            timing=OtoTiming(offset=540.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="sa",
            timing=OtoTiming(offset=700.0, consonant=190.0, cutoff=-370.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="a s",
            timing=OtoTiming(offset=1235.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=80.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="su",
            timing=OtoTiming(offset=1380.0, consonant=190.0, cutoff=-370.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="u s",
            timing=OtoTiming(offset=1520.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_si-sa-su-se.wav",
            alias="se",
            timing=OtoTiming(offset=1680.0, consonant=190.0, cutoff=-410.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _cv1, kept_vc, _cv2, repaired_vc, repaired_cv, repaired_vc2, repaired_cv2 = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert kept_vc.timing.offset == pytest.approx(540.0)
    assert "cvvc_obstruent_grid_late_cascade_repair" not in kept_vc.applied_rules
    assert repaired_vc.timing.offset == pytest.approx(890.0)
    assert repaired_cv.timing.offset == pytest.approx(1050.0)
    assert repaired_vc2.timing.offset == pytest.approx(1270.0)
    assert repaired_cv2.timing.offset == pytest.approx(1430.0)
    assert "cvvc_obstruent_grid_late_cascade_repair" in repaired_vc.applied_rules
    assert "cvvc_obstruent_grid_late_cascade_repair" in repaired_cv.applied_rules
    assert any("cvvc_obstruent_grid_first_step:410.0" in warning for warning in repaired_vc.warnings)


def test_japanese_cvvc_obstruent_grid_skips_sonorant_and_moraic_n_contexts():
    rows = [
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="- w",
            timing=OtoTiming(offset=640.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="wa",
            timing=OtoTiming(offset=290.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="a w",
            timing=OtoTiming(offset=1235.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="wi",
            timing=OtoTiming(offset=700.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="n s",
            timing=OtoTiming(offset=1800.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-n-sa.wav",
            alias="sa",
            timing=OtoTiming(offset=1900.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _cv1, repaired_glide, _cv2, repaired_n_vc, repaired_n_cv = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3200.0,
    )

    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_glide.applied_rules
    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_n_vc.applied_rules
    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_n_cv.applied_rules


def test_japanese_cvvc_obstruent_grid_skips_excessive_backshift():
    rows = [
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="- k",
            timing=OtoTiming(offset=600.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="ka",
            timing=OtoTiming(offset=300.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="a k",
            timing=OtoTiming(offset=2500.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="ki",
            timing=OtoTiming(offset=700.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="i k",
            timing=OtoTiming(offset=2600.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku.wav",
            alias="ku",
            timing=OtoTiming(offset=2700.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _ka, repaired_ak, _ki, repaired_ik, repaired_ku = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3600.0,
    )

    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_ak.applied_rules
    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_ik.applied_rules
    assert "cvvc_obstruent_grid_late_cascade_repair" not in repaired_ku.applied_rules


def test_japanese_cvvc_tail_moraic_n_repairs_late_cascade():
    rows = [
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="- k",
            timing=OtoTiming(offset=350.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="\u304b1",
            timing=OtoTiming(offset=2260.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="a n3",
            timing=OtoTiming(offset=2780.0, consonant=160.0, cutoff=-430.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="\u30937",
            timing=OtoTiming(offset=3254.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.8),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="n k",
            timing=OtoTiming(offset=3300.0, consonant=195.0, cutoff=-221.0, preutterance=155.0, overlap=120.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="\u304b2",
            timing=OtoTiming(offset=3460.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_kikakukekokakanka.wav",
            alias="a -3",
            timing=OtoTiming(offset=3650.0, consonant=210.0, cutoff=-337.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _anchor, repaired_an, repaired_n, repaired_nk, repaired_ka, repaired_terminal = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert repaired_an.timing.offset == pytest.approx(2490.0)
    assert repaired_n.timing.offset == pytest.approx(2600.0)
    assert repaired_nk.timing.offset == pytest.approx(2810.0)
    assert repaired_ka.timing.offset == pytest.approx(2960.0)
    assert repaired_terminal.timing.offset == pytest.approx(3150.0)
    assert "cvvc_tail_moraic_n_late_cascade_repair" in repaired_an.applied_rules
    assert "cvvc_tail_moraic_n_late_cascade_repair" in repaired_nk.applied_rules
    assert any("cvvc_tail_moraic_n_role:n_vc" in warning for warning in repaired_nk.warnings)


def test_japanese_cvvc_tail_yoon_repairs_late_vc_cascade():
    rows = [
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="- gy",
            timing=OtoTiming(offset=350.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u3087",
            timing=OtoTiming(offset=1710.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="o gy",
            timing=OtoTiming(offset=2620.0, consonant=160.0, cutoff=-410.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u30831",
            timing=OtoTiming(offset=2780.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="a n6",
            timing=OtoTiming(offset=2760.0, consonant=160.0, cutoff=-430.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u309310",
            timing=OtoTiming(offset=3130.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.8),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="n gy",
            timing=OtoTiming(offset=3290.0, consonant=195.0, cutoff=-221.0, preutterance=155.0, overlap=120.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u30832",
            timing=OtoTiming(offset=3520.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="a -6",
            timing=OtoTiming(offset=3650.0, consonant=210.0, cutoff=-337.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    (
        _head,
        _anchor,
        repaired_o_gy,
        repaired_gya,
        repaired_an,
        repaired_n,
        repaired_n_gy,
        repaired_final,
        repaired_terminal,
    ) = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert repaired_o_gy.timing.offset == pytest.approx(2060.0)
    assert repaired_gya.timing.offset == pytest.approx(2210.0)
    assert repaired_an.timing.offset == pytest.approx(2470.0)
    assert repaired_n.timing.offset == pytest.approx(2580.0)
    assert repaired_n_gy.timing.offset == pytest.approx(2770.0)
    assert repaired_final.timing.offset == pytest.approx(2910.0)
    assert repaired_terminal.timing.offset == pytest.approx(3110.0)
    assert "cvvc_tail_yoon_late_cascade_repair" in repaired_o_gy.applied_rules
    assert "cvvc_tail_yoon_late_cascade_repair" in repaired_n_gy.applied_rules
    assert any("cvvc_tail_yoon_role:n_vc" in warning for warning in repaired_n_gy.warnings)


def test_japanese_cvvc_tail_yoon_requires_late_yoon_vc_gate():
    rows = [
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="- gy",
            timing=OtoTiming(offset=350.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u3087",
            timing=OtoTiming(offset=1710.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="o gy",
            timing=OtoTiming(offset=2240.0, consonant=160.0, cutoff=-410.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u30831",
            timing=OtoTiming(offset=2380.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="a n6",
            timing=OtoTiming(offset=2760.0, consonant=160.0, cutoff=-430.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u309310",
            timing=OtoTiming(offset=3130.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.8),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="n gy",
            timing=OtoTiming(offset=3290.0, consonant=195.0, cutoff=-221.0, preutterance=155.0, overlap=120.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_gyagigyugegyogyangya.wav",
            alias="\u304e\u30832",
            timing=OtoTiming(offset=3520.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _anchor, kept_o_gy, *_rest = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert kept_o_gy.timing.offset == pytest.approx(2240.0)
    assert "cvvc_tail_yoon_late_cascade_repair" not in kept_o_gy.applied_rules


def test_japanese_cvvc_initial_u_to_w_sequence_repairs_split_vc_drift():
    rows = [
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="- u",
            timing=OtoTiming(offset=430.0, consonant=220.0, cutoff=-320.0, preutterance=80.0, overlap=40.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u30462",
            timing=OtoTiming(offset=154.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.84),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="u w",
            timing=OtoTiming(offset=115.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u308f",
            timing=OtoTiming(offset=275.0, consonant=190.0, cutoff=-400.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="a w",
            timing=OtoTiming(offset=450.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u3046\u3043",
            timing=OtoTiming(offset=610.0, consonant=190.0, cutoff=-425.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="i w",
            timing=OtoTiming(offset=1435.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u3046\u3047",
            timing=OtoTiming(offset=1580.0, consonant=190.0, cutoff=-525.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="e w",
            timing=OtoTiming(offset=2025.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u3092",
            timing=OtoTiming(offset=2080.0, consonant=190.0, cutoff=-910.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="o w",
            timing=OtoTiming(offset=2280.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_uwawiwewowanwa.wav",
            alias="\u308f1",
            timing=OtoTiming(offset=2440.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    (
        _head,
        repaired_u,
        repaired_u_w,
        repaired_wa,
        repaired_a_w,
        repaired_wi,
        repaired_i_w,
        repaired_we,
        repaired_e_w,
        repaired_wo,
        repaired_o_w,
        repaired_trailing_wa,
    ) = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3700.0,
    )

    assert repaired_u.timing.offset == pytest.approx(540.0)
    assert repaired_u_w.timing.offset == pytest.approx(556.0)
    assert repaired_wa.timing.offset == pytest.approx(757.0)
    assert repaired_a_w.timing.offset == pytest.approx(992.0)
    assert repaired_wi.timing.offset == pytest.approx(1132.0)
    assert repaired_i_w.timing.offset == pytest.approx(1322.0)
    assert repaired_we.timing.offset == pytest.approx(1477.0)
    assert repaired_e_w.timing.offset == pytest.approx(1682.0)
    assert repaired_wo.timing.offset == pytest.approx(1828.0)
    assert repaired_o_w.timing.offset == pytest.approx(2027.0)
    assert repaired_trailing_wa.timing.offset == pytest.approx(2175.0)
    assert "cvvc_initial_w_glide_sequence_repair" in repaired_u_w.applied_rules
    assert "cvvc_initial_w_glide_sequence_repair" in repaired_e_w.applied_rules


def test_japanese_cvvc_initial_w_head_sequence_repairs_split_vc_drift():
    rows = [
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="- \u308f",
            timing=OtoTiming(offset=430.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="a w",
            timing=OtoTiming(offset=510.0, consonant=120.0, cutoff=-146.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="\u3046\u3043",
            timing=OtoTiming(offset=685.0, consonant=210.0, cutoff=-400.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="i w",
            timing=OtoTiming(offset=1170.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="\u3046\u3045",
            timing=OtoTiming(offset=1330.0, consonant=190.0, cutoff=-545.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="u w",
            timing=OtoTiming(offset=1950.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="\u3046\u3047",
            timing=OtoTiming(offset=2110.0, consonant=190.0, cutoff=-245.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="e w",
            timing=OtoTiming(offset=2540.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="\u3046\u3049",
            timing=OtoTiming(offset=2700.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="o w",
            timing=OtoTiming(offset=3090.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wawiwuwewowanwa.wav",
            alias="\u308f",
            timing=OtoTiming(offset=3250.0, consonant=190.0, cutoff=-405.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    (
        _head,
        repaired_a_w,
        repaired_wi,
        kept_i_w,
        kept_wu,
        repaired_u_w,
        repaired_we,
        repaired_e_w,
        repaired_wo,
        repaired_o_w,
        repaired_trailing_wa,
    ) = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3700.0,
    )

    assert repaired_a_w.timing.offset == pytest.approx(785.0)
    assert repaired_wi.timing.offset == pytest.approx(910.0)
    assert kept_i_w.timing.offset == pytest.approx(1170.0)
    assert kept_wu.timing.offset == pytest.approx(1330.0)
    assert repaired_u_w.timing.offset == pytest.approx(1485.0)
    assert repaired_we.timing.offset == pytest.approx(1616.0)
    assert repaired_e_w.timing.offset == pytest.approx(1870.0)
    assert repaired_wo.timing.offset == pytest.approx(2022.0)
    assert repaired_o_w.timing.offset == pytest.approx(2235.0)
    assert repaired_trailing_wa.timing.offset == pytest.approx(2381.0)
    assert "cvvc_initial_w_glide_sequence_repair" in repaired_a_w.applied_rules
    assert "cvvc_initial_w_glide_sequence_repair" in repaired_e_w.applied_rules


def test_japanese_cvvc_tail_moraic_n_requires_late_vowel_n_gate():
    rows = [
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="- sh",
            timing=OtoTiming(offset=320.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="\u3057\u30831",
            timing=OtoTiming(offset=1935.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="a n10",
            timing=OtoTiming(offset=2282.0, consonant=160.0, cutoff=-410.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="\u309314",
            timing=OtoTiming(offset=2650.0, consonant=100.0, cutoff=-106.0, preutterance=90.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="n sh",
            timing=OtoTiming(offset=2660.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="\u3057\u30832",
            timing=OtoTiming(offset=2820.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_shashishusheshoshansha.wav",
            alias="a -10",
            timing=OtoTiming(offset=3130.0, consonant=210.0, cutoff=-337.0, preutterance=110.0, overlap=100.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _anchor, kept_an, kept_n, kept_nsh, kept_sha, kept_terminal = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert kept_an.timing.offset == pytest.approx(2282.0)
    assert kept_n.timing.offset == pytest.approx(2650.0)
    assert kept_nsh.timing.offset == pytest.approx(2660.0)
    assert kept_sha.timing.offset == pytest.approx(2820.0)
    assert kept_terminal.timing.offset == pytest.approx(3130.0)
    assert "cvvc_tail_moraic_n_late_cascade_repair" not in kept_nsh.applied_rules


def test_japanese_cvvc_tail_moraic_n_requires_late_transition_gate():
    rows = [
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="- h",
            timing=OtoTiming(offset=360.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="\u306f2",
            timing=OtoTiming(offset=1985.0, consonant=190.0, cutoff=-198.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="a n26",
            timing=OtoTiming(offset=2412.0, consonant=160.0, cutoff=-410.0, preutterance=120.0, overlap=85.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="\u309328",
            timing=OtoTiming(offset=2844.0, consonant=80.0, cutoff=-106.0, preutterance=56.0, overlap=35.8),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="n h",
            timing=OtoTiming(offset=2690.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_fafifufefanfua.wav",
            alias="\u306f3",
            timing=OtoTiming(offset=2850.0, consonant=190.0, cutoff=-200.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    _head, _anchor, kept_an, kept_n, kept_nh, kept_ha = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert kept_an.timing.offset == pytest.approx(2412.0)
    assert kept_n.timing.offset == pytest.approx(2844.0)
    assert kept_nh.timing.offset == pytest.approx(2690.0)
    assert kept_ha.timing.offset == pytest.approx(2850.0)
    assert "cvvc_tail_moraic_n_late_cascade_repair" not in kept_nh.applied_rules


def test_japanese_cvvc_tail_moraic_n_wrap_repairs_grouped_vc_target():
    wav = "sa-si-su-se-so-sa-N-sa.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias="- s",
            timing=OtoTiming(offset=800.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="a s",
            timing=OtoTiming(offset=885.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="i s",
            timing=OtoTiming(offset=1736.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="u s",
            timing=OtoTiming(offset=2285.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="e s",
            timing=OtoTiming(offset=2825.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="o s",
            timing=OtoTiming(offset=3676.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="n s",
            timing=OtoTiming(offset=825.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u3059\u3043",
            timing=OtoTiming(offset=985.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u3059",
            timing=OtoTiming(offset=1420.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u305b",
            timing=OtoTiming(offset=2030.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u305d",
            timing=OtoTiming(offset=2560.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u3055",
            timing=OtoTiming(offset=3310.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4800.0,
    )
    repaired_ns = next(row for row in repaired if row.alias == "n s")

    assert repaired_ns.timing.offset == pytest.approx(4211.0)
    assert "cvvc_tail_moraic_n_wrap_repair" in repaired_ns.applied_rules
    assert "cvvc_unprefixed_obstruent_extra_vc_grid_repair" in repaired_ns.applied_rules
    assert any(warning.startswith("cvvc_tail_moraic_n_wrap:825.0->4010.0") for warning in repaired_ns.warnings)
    assert "cvvc_tail_moraic_n_wrap_reference:\u3055" in repaired_ns.warnings


def test_japanese_cvvc_tail_moraic_n_wrap_keeps_near_reference_vc():
    wav = "sa-N-sa.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias="n s",
            timing=OtoTiming(offset=2690.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u3055",
            timing=OtoTiming(offset=2850.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    kept_ns, _cv = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=3990.0,
    )

    assert kept_ns.timing.offset == pytest.approx(2690.0)
    assert "cvvc_tail_moraic_n_wrap_repair" not in kept_ns.applied_rules


def test_japanese_cvvc_tail_moraic_n_wrap_allows_sonorant_right_context():
    wav = "wa-wi-we-wo-wa-u-wa-N-wa.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias="n w",
            timing=OtoTiming(offset=825.0, consonant=185.0, cutoff=-205.0, preutterance=135.0, overlap=50.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav=wav,
            alias="\u308f",
            timing=OtoTiming(offset=3310.0, consonant=190.0, cutoff=-455.0, preutterance=150.0, overlap=115.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired_nw, _cv = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4800.0,
    )

    assert repaired_nw.timing.offset == pytest.approx(4010.0)
    assert "cvvc_tail_moraic_n_wrap_repair" in repaired_nw.applied_rules


def test_japanese_cvvc_unrepaired_vv_long_cutoff_uses_short_profile(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
    vv = AdaptedOtoRow(
        wav="_aaiauea.wav",
        alias="a i",
        timing=OtoTiming(offset=1190.0, consonant=160.0, cutoff=-3060.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1310.0, score=0.5, role="vv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )
    vcv = AdaptedOtoRow(
        wav="_aaiauea.wav",
        alias="n a",
        timing=OtoTiming(offset=1340.0, consonant=160.0, cutoff=-2350.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1460.0, score=0.5, role="vcv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )
    nasal_vv = AdaptedOtoRow(
        wav="_aaiauea.wav",
        alias="i ん",
        timing=OtoTiming(offset=1110.0, consonant=160.0, cutoff=-3420.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1230.0, score=0.5, role="vv", source_event_label="vv_boundary"),
        mode="bootstrap",
    )

    repaired_vv, repaired_vcv, repaired_nasal = repair_cvvc_row_sequence(
        [vv, vcv, nasal_vv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5400.0,
    )

    expected_offsets = {"a i": 1190.0, "n a": 1340.0, "i ん": 1110.0}
    for repaired in (repaired_vv, repaired_vcv, repaired_nasal):
        assert repaired.timing.offset == pytest.approx(expected_offsets[repaired.alias])
        assert repaired.timing.consonant == pytest.approx(181.0)
        assert repaired.timing.cutoff == pytest.approx(-255.0)
        assert repaired.timing.preutterance == pytest.approx(118.0)
        assert repaired.timing.overlap == pytest.approx(90.0)
        assert "cvvc_unrepaired_vv_long_cutoff_profile_repair" in repaired.applied_rules


def test_japanese_cvvc_unrepaired_vv_long_cutoff_keeps_headed_or_grid_rows():
    head = AdaptedOtoRow(
        wav="_aaiauea.wav",
        alias="- a",
        timing=OtoTiming(offset=380.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    headed_vv = AdaptedOtoRow(
        wav="_aaiauea.wav",
        alias="a i",
        timing=OtoTiming(offset=1190.0, consonant=160.0, cutoff=-3060.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1310.0, score=0.5, role="vv", source_event_label="vowel_nucleus"),
        mode="bootstrap",
    )
    grid_vv = AdaptedOtoRow(
        wav="aaiauea.wav",
        alias="i a",
        timing=OtoTiming(offset=1550.0, consonant=420.0, cutoff=-600.0, preutterance=300.0, overlap=100.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=1360.0, score=0.5, role="vv", source_event_label="vv_boundary"),
        mode="bootstrap",
        applied_rules=("cvvc_pure_vowel_sequence_row_repair",),
    )

    _head, kept_headed, kept_grid = repair_cvvc_row_sequence(
        [head, headed_vv, grid_vv],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5400.0,
    )

    assert kept_headed.timing == headed_vv.timing
    assert kept_grid.timing == grid_vv.timing
    assert "cvvc_unrepaired_vv_long_cutoff_profile_repair" not in kept_headed.applied_rules
    assert "cvvc_unrepaired_vv_long_cutoff_profile_repair" not in kept_grid.applied_rules


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

    hsmm_rows = [
        replace(row, warnings=(*row.warnings, "event_source:filename_hsmm"))
        for row in rows
    ]
    kept = repair_cvvc_row_sequence(
        hsmm_rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6000.0,
    )

    assert [row.timing.offset for row in kept] == [row.timing.offset for row in rows]
    assert all("cvvc_regular_pair_onset_repair" not in row.applied_rules for row in kept)


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


def test_japanese_cvvc_grouped_initial_vc_sync_repairs_second_vc_after_final_cv_grid():
    aliases_offsets = [
        ("a b", 845.0),
        ("i b", 1455.0),
        ("u b", 2375.0),
        ("e b", 2895.0),
        ("o b", 3385.0),
        ("n b", 4395.0),
        ("bi", 1433.0),
        ("bu", 2030.0),
        ("be", 2600.0),
        ("bo", 3120.0),
        ("ba", 3580.0),
    ]
    rows = [
        AdaptedOtoRow(
            wav="ba-bi-bu-be-bo-ba-N-ba.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=210.0 if " " in alias else 150.0,
                cutoff=-236.0 if " " in alias else -360.0,
                preutterance=170.0 if " " in alias else 70.0,
                overlap=135.0 if " " in alias else 25.0,
            ),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in aliases_offsets
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired[0].timing.offset == pytest.approx(1223.0)
    assert repaired[1].timing.offset == pytest.approx(1820.0)
    assert repaired[2].timing.offset == pytest.approx(2375.0)
    assert "cvvc_grouped_vc_first_offset_repair" in repaired[0].applied_rules
    assert "cvvc_grouped_initial_vc_final_cv_sync_repair" in repaired[1].applied_rules


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


def test_japanese_cvvc_grouped_vc_cv_one_step_shift_regrids_collapsed_standard_block():
    aliases_offsets = [
        ("a g", 910.0),
        ("i g", 1060.0),
        ("u g", 1820.0),
        ("e g", 2840.0),
        ("o g", 4370.0),
        ("n g", 4496.0),
        ("ga", 1038.0),
        ("gi", 1480.0),
        ("gu", 2440.0),
        ("ge", 3370.0),
        ("go", 4510.0),
    ]
    rows = [
        AdaptedOtoRow(
            wav="ga-gi-gu-ge-go-ga-n-ga.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=210.0 if " " in alias else 150.0,
                cutoff=-236.0 if " " in alias else -360.0,
                preutterance=170.0 if " " in alias else 70.0,
                overlap=135.0 if " " in alias else 25.0,
            ),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            warnings=("local_refine_low_margin",),
        )
        for alias, offset in aliases_offsets
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert [row.timing.offset for row in repaired[:6]] == pytest.approx(
        [1270.0, 1770.0, 2270.0, 2770.0, 3270.0, 4230.0]
    )
    assert [row.timing.offset for row in repaired[6:]] == pytest.approx(
        [1480.0, 1980.0, 2480.0, 2980.0, 3480.0]
    )
    assert "cvvc_grouped_vc_cv_one_step_shift_repair" in repaired[0].applied_rules
    assert any("cvvc_grouped_vc_cv_one_step_reference:gi" in item for item in repaired[0].warnings)


def test_japanese_cvvc_grouped_vc_cv_one_step_shift_keeps_aligned_soft_block():
    aliases_offsets = [
        ("a g_S", 1231.557),
        ("i g_S", 1731.557),
        ("u g_S", 2231.557),
        ("e g_S", 2731.557),
        ("o g_S", 3231.557),
        ("n g_S", 4206.557),
        ("ga_S", 1456.557),
        ("gi_S", 1956.557),
        ("gu_S", 2456.557),
        ("ge_S", 2956.557),
        ("go_S", 3456.557),
    ]
    rows = [
        AdaptedOtoRow(
            wav="_ga-gi-gu-ge-go-ga-n-ga.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=210.0 if " " in alias else 150.0,
                cutoff=-236.0 if " " in alias else -360.0,
                preutterance=170.0 if " " in alias else 70.0,
                overlap=135.0 if " " in alias else 25.0,
            ),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in aliases_offsets
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([offset for _alias, offset in aliases_offsets])
    assert all("cvvc_grouped_vc_cv_one_step_shift_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_grouped_vc_cv_one_step_shift_keeps_normal_vc_cadence_block():
    aliases_offsets = [
        ("a bF4S", 755.0),
        ("i bF4S", 1175.0),
        ("u bF4S", 1595.0),
        ("e bF4S", 2185.0),
        ("o bF4S", 2655.0),
        ("n bF4S", 3335.0),
        ("biF4S", 876.5),
        ("buF4S", 1410.0),
        ("beF4S", 1840.0),
        ("boF4S", 2600.0),
        ("baF4S", 2900.0),
    ]
    rows = [
        AdaptedOtoRow(
            wav="ba-bi-bu-be-bo-ba-N-ba.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=210.0 if " " in alias else 150.0,
                cutoff=-236.0 if " " in alias else -360.0,
                preutterance=170.0 if " " in alias else 70.0,
                overlap=135.0 if " " in alias else 25.0,
            ),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in aliases_offsets
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert [row.timing.offset for row in repaired[:6]] == pytest.approx(
        [offset for _alias, offset in aliases_offsets[:6]]
    )
    assert [row.timing.offset for row in repaired[6:]] == pytest.approx([990.0, 1435.0, 1880.0, 2325.0, 2770.0])
    assert all("cvvc_grouped_vc_cv_one_step_shift_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_cv_block_regrids_compressed_cv_only():
    rows = [
        AdaptedOtoRow(
            wav="_nya-nyu-nye-nyo-nya.wav",
            alias="nyu_S",
            timing=OtoTiming(offset=930.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_nya-nyu-nye-nyo-nya.wav",
            alias="nye_S",
            timing=OtoTiming(offset=1180.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_nya-nyu-nye-nyo-nya.wav",
            alias="nyo_S",
            timing=OtoTiming(offset=1420.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_nya-nyu-nye-nyo-nya.wav",
            alias="nya_S",
            timing=OtoTiming(offset=1870.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5400.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1430.0, 1930.0, 2430.0, 2930.0])
    assert all("cvvc_headless_pitch_suffix_cv_grid_repair" in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_grouped_vc_cv_block_regrids_from_cv_block():
    rows = [
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="a d_S",
            timing=OtoTiming(offset=1020.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="i d_S",
            timing=OtoTiming(offset=1400.0, consonant=130.0, cutoff=-156.0, preutterance=90.0, overlap=55.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="u d_S",
            timing=OtoTiming(offset=2300.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="e d_S",
            timing=OtoTiming(offset=3310.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="o d_S",
            timing=OtoTiming(offset=4370.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="di_S",
            timing=OtoTiming(offset=1059.716, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="du_S",
            timing=OtoTiming(offset=1920.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="de_S",
            timing=OtoTiming(offset=2900.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="do_S",
            timing=OtoTiming(offset=3570.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_da-di-du-de-do-da-N-da.wav",
            alias="da_S",
            timing=OtoTiming(offset=4400.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1489.858, 1989.858, 2489.858, 2989.858, 3489.858]
    expected_vc = [1264.858, 1764.858, 2264.858, 2764.858, 3264.858]
    assert [row.timing.offset for row in repaired[:5]] == pytest.approx(expected_vc)
    assert [row.timing.offset for row in repaired[5:]] == pytest.approx(expected_cv)
    assert all("cvvc_headless_pitch_suffix_vc_grid_repair" in row.applied_rules for row in repaired[:5])
    assert all("cvvc_headless_pitch_suffix_cv_grid_repair" in row.applied_rules for row in repaired[5:])


def test_japanese_cvvc_headless_pitch_suffix_block_allows_initial_cv_head_dash():
    rows = [
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="- w_S",
            timing=OtoTiming(offset=882.761, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="a w_S",
            timing=OtoTiming(offset=930.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="i w_S",
            timing=OtoTiming(offset=1766.54, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="e w_S",
            timing=OtoTiming(offset=1910.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="o w_S",
            timing=OtoTiming(offset=2860.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="u w_S",
            timing=OtoTiming(offset=4820.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="n w_S",
            timing=OtoTiming(offset=4979.731, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u3046\u3043_S",
            timing=OtoTiming(offset=966.839, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u3046\u3047_S",
            timing=OtoTiming(offset=1890.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u3092_S",
            timing=OtoTiming(offset=2430.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u308f_S",
            timing=OtoTiming(offset=2930.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1428.4195, 1928.4195, 2428.4195, 2928.4195]
    expected_vc = [1203.4195, 1703.4195, 2203.4195, 2703.4195]
    expected_extra_vc = [3678.4195, 4678.4195]
    assert repaired[0].timing.offset == pytest.approx(882.761)
    assert [row.timing.offset for row in repaired[1:5]] == pytest.approx(expected_vc)
    assert [row.timing.offset for row in repaired[7:]] == pytest.approx(expected_cv)
    assert [row.timing.offset for row in repaired[5:7]] == pytest.approx(expected_extra_vc)
    assert all("cvvc_headless_pitch_suffix_extra_vc_grid_repair" in row.applied_rules for row in repaired[5:7])
    assert "cvvc_headless_pitch_suffix_cv_grid_repair" in repaired[7].applied_rules


def test_japanese_cvvc_headless_pitch_suffix_cv_block_regrids_large_internal_gap():
    rows = [
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias="\u304d_S",
            timing=OtoTiming(offset=946.824, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias="\u304f_S",
            timing=OtoTiming(offset=1430.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias="\u3051_S",
            timing=OtoTiming(offset=2420.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias="\u3053_S",
            timing=OtoTiming(offset=3000.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias="\u304b_S",
            timing=OtoTiming(offset=4380.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx(
        [1446.824, 1946.824, 2446.824, 2946.824, 3446.824]
    )
    assert all("cvvc_headless_pitch_suffix_cv_grid_repair" in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_keeps_direct_hsmm_cv_block():
    offsets = [946.824, 1430.0, 2420.0, 3000.0, 4380.0]
    aliases = ["\u304d_S", "\u304f_S", "\u3051_S", "\u3053_S", "\u304b_S"]
    rows = [
        AdaptedOtoRow(
            wav="_ka-ki-ku-ke-ko-ka-N-ka.wav",
            alias=alias,
            timing=OtoTiming(
                offset=offset,
                consonant=150.0,
                cutoff=-360.0,
                preutterance=70.0,
                overlap=25.0,
            ),
            source_timing=None,
            anchor=OtoAnchor(
                anchor_abs_ms=offset + 70.0,
                score=0.7,
                role="cv",
                source_event_label="cv_boundary",
                warnings=("event_source:filename_hsmm",),
            ),
            mode="bootstrap",
        )
        for alias, offset in zip(aliases, offsets)
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx(offsets)
    assert all("cvvc_headless_pitch_suffix_cv_grid_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_cv_block_keeps_aligned_first_cv_on_tail_gap():
    rows = [
        AdaptedOtoRow(
            wav="_tsa-tsi-tsu-tse-tso-tsa-N-tsu.wav",
            alias="\u3064\u3043_S",
            timing=OtoTiming(offset=1425.878, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_tsa-tsi-tsu-tse-tso-tsa-N-tsu.wav",
            alias="\u3064_S",
            timing=OtoTiming(offset=1890.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_tsa-tsi-tsu-tse-tso-tsa-N-tsu.wav",
            alias="\u3064\u3047_S",
            timing=OtoTiming(offset=2460.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_tsa-tsi-tsu-tse-tso-tsa-N-tsu.wav",
            alias="\u3064\u3049_S",
            timing=OtoTiming(offset=3280.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_tsa-tsi-tsu-tse-tso-tsa-N-tsu.wav",
            alias="\u3064\u3041_S",
            timing=OtoTiming(offset=4220.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx(
        [1425.878, 1925.878, 2425.878, 2925.878, 3425.878]
    )
    assert all("cvvc_headless_pitch_suffix_cv_grid_repair" in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_s_block_uses_earlier_grid_base():
    rows = [
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="- s_S",
            timing=OtoTiming(offset=963.1, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="a s_S",
            timing=OtoTiming(offset=1003.1, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="i s_S",
            timing=OtoTiming(offset=1380.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="u s_S",
            timing=OtoTiming(offset=1750.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="e s_S",
            timing=OtoTiming(offset=2320.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="o s_S",
            timing=OtoTiming(offset=3320.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="\u3059\u3043_S",
            timing=OtoTiming(offset=1105.533, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="\u3059_S",
            timing=OtoTiming(offset=1470.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="\u305b_S",
            timing=OtoTiming(offset=1920.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="\u305d_S",
            timing=OtoTiming(offset=2910.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_sa-si-su-se-so-sa-N-sa.wav",
            alias="\u3055_S",
            timing=OtoTiming(offset=4300.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1450.533, 1950.533, 2450.533, 2950.533, 3450.533]
    expected_vc = [1225.533, 1725.533, 2225.533, 2725.533, 3225.533]
    assert repaired[0].timing.offset == pytest.approx(1425.533)
    assert [row.timing.offset for row in repaired[1:6]] == pytest.approx(expected_vc)
    assert [row.timing.offset for row in repaired[6:]] == pytest.approx(expected_cv)


def test_japanese_cvvc_headless_pitch_suffix_r_block_uses_later_grid_base_without_moving_head():
    rows = [
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="- r_S",
            timing=OtoTiming(offset=962.158, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="a r_S",
            timing=OtoTiming(offset=1002.2, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="i r_S",
            timing=OtoTiming(offset=1450.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="u r_S",
            timing=OtoTiming(offset=1930.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="e r_S",
            timing=OtoTiming(offset=2910.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="o r_S",
            timing=OtoTiming(offset=3450.0, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="\u308a_S",
            timing=OtoTiming(offset=1383.671, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="\u308b_S",
            timing=OtoTiming(offset=1480.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="\u308c_S",
            timing=OtoTiming(offset=2420.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="\u308d_S",
            timing=OtoTiming(offset=2950.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ra-ri-ru-re-ro-ra-N-ra.wav",
            alias="\u3089_S",
            timing=OtoTiming(offset=4200.0, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1508.671, 2008.671, 2508.671, 3008.671, 3508.671]
    expected_vc = [1283.671, 1783.671, 2283.671, 2783.671, 3283.671]
    assert repaired[0].timing.offset == pytest.approx(962.158)
    assert [row.timing.offset for row in repaired[1:6]] == pytest.approx(expected_vc)
    assert [row.timing.offset for row in repaired[6:]] == pytest.approx(expected_cv)


def test_japanese_cvvc_headless_pitch_suffix_m_block_uses_compressed_sonorant_step():
    def row(alias: str, offset: float, role_profile: str = "vc") -> AdaptedOtoRow:
        if role_profile == "head":
            timing = OtoTiming(offset=offset, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0)
        elif role_profile == "cv":
            timing = OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0)
        else:
            timing = OtoTiming(offset=offset, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0)
        return AdaptedOtoRow(
            wav="_ma-mi-mu-me-mo-ma-N-ma.wav",
            alias=alias,
            timing=timing,
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )

    rows = [
        row("- m_S", 932.793, "head"),
        row("a m_S", 972.8),
        row("i m_S", 1410.0),
        row("u m_S", 1930.0),
        row("e m_S", 2910.0),
        row("o m_S", 3350.0),
        row("n m_S", 4398.9),
        row("\u307f_S", 1023.945, "cv"),
        row("\u3080_S", 1480.0, "cv"),
        row("\u3081_S", 2390.0, "cv"),
        row("\u3082_S", 3260.0, "cv"),
        row("\u307e_S", 4350.0, "cv"),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1523.945, 1988.945, 2453.945, 2918.945, 3383.945]
    expected_vc = [1298.945, 1763.945, 2228.945, 2693.945, 3158.945]
    assert repaired[0].timing.offset == pytest.approx(1498.945)
    assert [row.timing.offset for row in repaired[1:6]] == pytest.approx(expected_vc)
    assert repaired[6].timing.offset == pytest.approx(4208.945)
    assert [row.timing.offset for row in repaired[7:]] == pytest.approx(expected_cv)


def test_japanese_cvvc_headless_pitch_suffix_f_block_uses_earlier_base_and_vc_profile():
    def row(alias: str, offset: float, role_profile: str = "vc") -> AdaptedOtoRow:
        if role_profile == "head":
            timing = OtoTiming(offset=930.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0)
        elif role_profile == "cv":
            timing = OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0)
        else:
            timing = OtoTiming(offset=offset, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0)
        return AdaptedOtoRow(
            wav="_fa-fi-fu-fe-fo-fa-N-fu.wav",
            alias=alias,
            timing=timing,
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )

    rows = [
        row("- f_S", 930.0, "head"),
        row("a f_S", 970.0),
        row("i f_S", 1400.0),
        row("u f_S", 1930.0),
        row("e f_S", 2910.0),
        row("o f_S", 3350.0),
        row("\u3075\u3043_S", 1007.7, "cv"),
        row("\u3075_S", 1480.0, "cv"),
        row("\u3075\u3047_S", 2390.0, "cv"),
        row("\u3075\u3049_S", 3260.0, "cv"),
        row("\u3075\u3041_S", 4350.0, "cv"),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_cv = [1427.7, 1927.7, 2427.7, 2927.7, 3427.7]
    expected_vc = [1202.7, 1702.7, 2202.7, 2702.7, 3202.7]
    assert repaired[0].timing.offset == pytest.approx(1402.7)
    assert [row.timing.offset for row in repaired[1:6]] == pytest.approx(expected_vc)
    assert [row.timing.preutterance for row in repaired[1:6]] == pytest.approx([200.0] * 5)
    assert [row.timing.overlap for row in repaired[1:6]] == pytest.approx([50.0] * 5)
    assert [row.timing.consonant for row in repaired[1:6]] == pytest.approx([230.0] * 5)
    assert [row.timing.cutoff for row in repaired[1:6]] == pytest.approx([-270.0] * 5)
    assert all("cvvc_headless_pitch_suffix_vc_profile_repair" in row.applied_rules for row in repaired[1:6])
    assert [row.timing.offset for row in repaired[6:]] == pytest.approx(expected_cv)


def test_japanese_cvvc_unprefixed_obstruent_grid_uses_internal_vc_base():
    def row(alias: str, offset: float, role_profile: str = "vc") -> AdaptedOtoRow:
        if role_profile == "head":
            timing = OtoTiming(offset=offset, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0)
        elif role_profile == "cv":
            timing = OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0)
        else:
            timing = OtoTiming(offset=offset, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0)
        return AdaptedOtoRow(
            wav="fa-fi-fu-fe-fo-fa-N-fu.wav",
            alias=alias,
            timing=timing,
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )

    rows = [
        row("- f", 800.0, "head"),
        row("a f", 885.0),
        row("i f", 1721.5),
        row("u f", 2694.8),
        row("e f", 3275.0),
        row("o f", 3495.0),
        row("n f", 4180.0),
        row("\u3075\u3041", 1242.0, "cv"),
        row("\u3075", 1410.0, "cv"),
        row("\u3075\u3047", 2320.0, "cv"),
        row("\u3075\u3049", 2920.0, "cv"),
        row("\u3075\u3043", 3480.0, "cv"),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    expected_base = 1446.5
    assert repaired[0].timing.offset == pytest.approx(expected_base - 25.0)
    assert [row.timing.offset for row in repaired[1:6]] == pytest.approx(
        [1221.5, 1721.5, 2221.5, 2721.5, 3221.5]
    )
    assert repaired[6].timing.offset == pytest.approx(4196.5)
    assert [row.timing.offset for row in repaired[7:]] == pytest.approx(
        [1446.5, 1946.5, 2446.5, 2946.5, 3446.5]
    )
    assert "cvvc_unprefixed_obstruent_cv_head_grid_repair" in repaired[0].applied_rules
    assert all("cvvc_unprefixed_obstruent_vc_grid_repair" in row.applied_rules for row in repaired[1:6])
    assert all("cvvc_unprefixed_obstruent_vc_profile_repair" in row.applied_rules for row in repaired[1:6])
    assert all("cvvc_unprefixed_obstruent_cv_grid_repair" in row.applied_rules for row in repaired[7:])


def test_japanese_cvvc_unprefixed_obstruent_grid_ignores_pitch_suffixed_aliases():
    rows = [
        AdaptedOtoRow(
            wav="fa-fi-fu-fe-fo-fa-N-fu.wav",
            alias="- fF4S",
            timing=OtoTiming(offset=800.0, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        *[
            AdaptedOtoRow(
                wav="fa-fi-fu-fe-fo-fa-N-fu.wav",
                alias=alias,
                timing=OtoTiming(offset=offset, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
                source_timing=None,
                anchor=None,
                mode="bootstrap",
            )
            for alias, offset in (
                ("a fF4S", 885.0),
                ("i fF4S", 1721.5),
                ("u fF4S", 2694.8),
                ("e fF4S", 3275.0),
                ("o fF4S", 3495.0),
            )
        ],
        *[
            AdaptedOtoRow(
                wav="fa-fi-fu-fe-fo-fa-N-fu.wav",
                alias=alias,
                timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
                source_timing=None,
                anchor=None,
                mode="bootstrap",
            )
            for alias, offset in (
                ("\u3075\u3041F4S", 1242.0),
                ("\u3075F4S", 1410.0),
                ("\u3075\u3047F4S", 2320.0),
                ("\u3075\u3049F4S", 2920.0),
            )
        ],
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert repaired[0].timing.offset == pytest.approx(800.0)
    assert all(
        "cvvc_unprefixed_obstruent_cv_head_grid_repair" not in row.applied_rules
        for row in repaired
    )


def test_japanese_cvvc_unprefixed_pitch_suffix_cadence_repairs_compressed_cv_block():
    wav = "za-zi-zu-ze-zo-za-N-za.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=160.0, cutoff=-260.0, preutterance=120.0, overlap=80.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("- zF4S", 370.0),
            ("a zF4S", 455.0),
            ("i zF4S", 1158.0),
            ("u zF4S", 1575.0),
            ("e zF4S", 2045.0),
            ("o zF4S", 2375.0),
            ("n zF4S", 2790.0),
            ("ziF4S", 842.0),
            ("zuF4S", 770.0),
            ("zeF4S", 1330.0),
            ("zoF4S", 1690.0),
            ("zaF4S", 2090.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4300.0,
    )

    assert [row.timing.offset for row in repaired[7:12]] == pytest.approx(
        [973.0, 1390.0, 1807.0, 2224.0, 2641.0],
        abs=0.01,
    )
    assert repaired[1].timing.offset == pytest.approx(847.0, abs=0.01)
    assert repaired[3].timing.offset == pytest.approx(1575.0, abs=0.01)
    assert "cvvc_unprefixed_pitch_suffix_cv_cadence_repair" in repaired[8].applied_rules
    assert "cvvc_unprefixed_pitch_suffix_first_vc_cadence_repair" in repaired[1].applied_rules


def test_japanese_cvvc_unprefixed_pitch_yoon_first_gap_repairs_late_tail():
    wav = "gya-gyu-gye-gyo-gya.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("\u304e\u3085F4S", 970.0),
            ("\u304e\u3047F4S", 1710.0),
            ("\u304e\u3087F4S", 2120.0),
            ("\u304e\u3083F4S", 2510.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([970.0, 1370.0, 1770.0, 2170.0])
    assert "cvvc_unprefixed_pitch_yoon_first_gap_repair" not in repaired[0].applied_rules
    assert all("cvvc_unprefixed_pitch_yoon_first_gap_repair" in row.applied_rules for row in repaired[1:])


def test_japanese_cvvc_unprefixed_pitch_yoon_first_gap_keeps_normal_cadence():
    wav = "kya-kyu-kye-kyo-kya.wav"
    rows = [
        AdaptedOtoRow(
            wav=wav,
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("\u304d\u3083F4S", 890.0),
            ("\u304d\u3085F4S", 1390.0),
            ("\u304d\u3087F4S", 1770.0),
            ("\u304d\u3083F4S", 2300.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([890.0, 1390.0, 1770.0, 2300.0])
    assert all("cvvc_unprefixed_pitch_yoon_first_gap_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_final_parameter_order_repair_extends_short_cutoff():
    row = AdaptedOtoRow(
        wav="_ki_ku_ke_ko.wav",
        alias="- \u304d",
        timing=OtoTiming(
            offset=512.0,
            consonant=174.84,
            cutoff=-170.0,
            preutterance=108.0,
            overlap=70.2,
        ),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("preexisting_warning",),
        applied_rules=("preexisting_rule",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5404.4,
    )

    assert repaired.timing.consonant == pytest.approx(174.84)
    assert abs(repaired.timing.cutoff) >= repaired.timing.consonant + 8.0 - 1e-6
    assert "oto_parameter_order_repair" in repaired.applied_rules
    assert any("oto_parameter_order_repaired" in warning for warning in repaired.warnings)


def test_japanese_cvvc_final_parameter_order_repair_shrinks_near_tail_source_timing():
    row = AdaptedOtoRow(
        wav="_aa_hh.wav",
        alias="a H",
        timing=OtoTiming(
            offset=5368.0,
            consonant=714.0,
            cutoff=-722.0,
            preutterance=240.0,
            overlap=80.0,
        ),
        source_timing=None,
        anchor=None,
        mode="template-special",
        warnings=("special_alias_source_timing_preserved",),
        applied_rules=("special_alias_source_timing_preserve",),
    )

    (repaired,) = repair_cvvc_row_sequence(
        [row],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=5408.0,
    )

    assert repaired.timing.cutoff == pytest.approx(-39.0)
    assert repaired.timing.consonant == pytest.approx(31.0)
    assert repaired.timing.preutterance == pytest.approx(31.0)
    assert repaired.timing.overlap == pytest.approx(31.0)
    assert repaired.timing.offset + repaired.timing.consonant < 5407.0
    assert "oto_parameter_order_repair" in repaired.applied_rules


def test_japanese_cvvc_headless_soft_yoon_only_grid_shifts_early_block():
    rows = [
        AdaptedOtoRow(
            wav="_gya-gyu-gye-gyo-gya.wav",
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("\u304e\u3083_S", 940.0),
            ("\u304e\u3085_S", 1450.0),
            ("\u304e\u3087_S", 1900.0),
            ("\u304e\u3084_S", 2380.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1440.0, 1950.0, 2400.0, 2880.0])
    assert all("cvvc_headless_soft_yoon_only_grid_repair" in row.applied_rules for row in repaired)
    assert any("cvvc_headless_soft_yoon_only_onset:gy" in item for item in repaired[0].warnings)


def test_japanese_cvvc_headless_soft_yoon_only_grid_keeps_aligned_block():
    rows = [
        AdaptedOtoRow(
            wav="_kya-kyu-kye-kyo-kya.wav",
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("\u304d\u3083_S", 1430.0),
            ("\u304d\u3085_S", 1910.0),
            ("\u304d\u3087_S", 2420.0),
            ("\u304d\u3084_S", 2910.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([1430.0, 1910.0, 2420.0, 2910.0])
    assert all("cvvc_headless_soft_yoon_only_grid_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_soft_yoon_only_grid_ignores_pitch_suffix_bank_alias():
    rows = [
        AdaptedOtoRow(
            wav="_gya-gyu-gye-gyo-gya.wav",
            alias=alias,
            timing=OtoTiming(offset=offset, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        )
        for alias, offset in (
            ("\u304e\u3083A3", 940.0),
            ("\u304e\u3085A3", 1450.0),
            ("\u304e\u3087A3", 1900.0),
            ("\u304e\u3084A3", 2380.0),
        )
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=4200.0,
    )

    assert [row.timing.offset for row in repaired] == pytest.approx([940.0, 1450.0, 1900.0, 2380.0])
    assert all("cvvc_headless_soft_yoon_only_grid_repair" not in row.applied_rules for row in repaired)


def test_japanese_cvvc_headless_pitch_suffix_cv_head_follows_delayed_grid_cv():
    rows = [
        AdaptedOtoRow(
            wav="_ma-mi-mu-me-mo-ma-N-ma.wav",
            alias="- m_S",
            timing=OtoTiming(offset=932.793, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_ma-mi-mu-me-mo-ma-N-ma.wav",
            alias="a m_S",
            timing=OtoTiming(offset=1298.945, consonant=120.0, cutoff=-146.0, preutterance=80.0, overlap=45.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            applied_rules=("cvvc_headless_pitch_suffix_vc_grid_repair",),
        ),
        AdaptedOtoRow(
            wav="_ma-mi-mu-me-mo-ma-N-ma.wav",
            alias="\u307f_S",
            timing=OtoTiming(offset=1523.945, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            applied_rules=("cvvc_headless_pitch_suffix_cv_grid_repair",),
        ),
        AdaptedOtoRow(
            wav="_ma-mi-mu-me-mo-ma-N-ma.wav",
            alias="\u3080_S",
            timing=OtoTiming(offset=2023.945, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            applied_rules=("cvvc_headless_pitch_suffix_cv_grid_repair",),
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert repaired[0].timing.offset == pytest.approx(1498.945)
    assert "cvvc_headless_pitch_suffix_cv_head_grid_repair" in repaired[0].applied_rules
    assert any("cvvc_headless_pitch_suffix_cv_head_grid_reference" in item for item in repaired[0].warnings)


def test_japanese_cvvc_headless_pitch_suffix_cv_head_keeps_glide_head():
    rows = [
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="- w_S",
            timing=OtoTiming(offset=882.761, consonant=65.0, cutoff=-95.0, preutterance=35.0, overlap=10.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u3046\u3043_S",
            timing=OtoTiming(offset=1428.420, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            applied_rules=("cvvc_headless_pitch_suffix_cv_grid_repair",),
        ),
        AdaptedOtoRow(
            wav="_wa-wi-we-wo-wa-u-wa-N-wa.wav",
            alias="\u3046\u3047_S",
            timing=OtoTiming(offset=1928.420, consonant=150.0, cutoff=-360.0, preutterance=70.0, overlap=25.0),
            source_timing=None,
            anchor=None,
            mode="bootstrap",
            applied_rules=("cvvc_headless_pitch_suffix_cv_grid_repair",),
        ),
    ]

    repaired = repair_cvvc_row_sequence(
        rows,
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
        file_duration_ms=6200.0,
    )

    assert repaired[0].timing.offset == pytest.approx(882.761)
    assert "cvvc_headless_pitch_suffix_cv_head_grid_repair" not in repaired[0].applied_rules


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

    assert [row.timing.offset for row in repaired] == pytest.approx([1560.0, 1890.0, 2050.0, 2210.0, 2550.0])
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


def test_japanese_cvvc_initial_vowel_cv_head_onset_requires_first_phone_index(monkeypatch):
    _enable_ja_cvvc_reference_repairs(monkeypatch)
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
    assert repaired_head.timing.preutterance == pytest.approx(108.0)
    assert "cvvc_initial_vowel_cv_head_onset_repair" not in repaired_head.applied_rules
    assert "cvvc_cv_head_role_profile_repair" in repaired_head.applied_rules


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


def test_japanese_cvvc_vc_next_vowel_local_repair_skips_sonorant_coda():
    row = AdaptedOtoRow(
        wav="_ni-na-nu-ne-no-nan-na.wav",
        alias="o n",
        timing=OtoTiming(offset=3115.0, consonant=210.0, cutoff=-236.0, preutterance=80.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(
            anchor_abs_ms=3240.0,
            score=0.45,
            role="vc",
            next_vowel_abs_ms=3702.0,
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

    assert repaired.timing.offset == pytest.approx(3115.0)
    assert repaired.timing.preutterance == pytest.approx(80.0)
    assert "cvvc_vc_next_vowel_local_repair" not in repaired.applied_rules
    assert not any(
        warning.startswith("cvvc_vc_next_vowel_local_repaired")
        for warning in repaired.warnings
    )


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

    assert repaired.timing.offset == pytest.approx(3360.0)
    assert repaired.timing.consonant == pytest.approx(125.0)
    assert repaired.timing.cutoff == pytest.approx(-260.0)
    assert repaired.timing.preutterance == pytest.approx(30.0)
    assert repaired.timing.overlap == pytest.approx(20.0)
    assert "cvvc_terminal_standalone_v_repair" in repaired.applied_rules


def test_japanese_cvvc_terminal_standalone_vowel_does_not_move_inner_repeated_vowels():
    early_vowel = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="\u3042",
        timing=OtoTiming(offset=500.0, consonant=100.0, cutoff=-240.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    vc = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a n",
        timing=OtoTiming(offset=720.0, consonant=120.0, cutoff=-160.0, preutterance=90.0, overlap=55.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    final_vowel = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="\u30424",
        timing=OtoTiming(offset=2944.0, consonant=100.0, cutoff=-240.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    terminal = AdaptedOtoRow(
        wav="_a-n-a.wav",
        alias="a -",
        timing=OtoTiming(offset=3094.0, consonant=320.0, cutoff=-480.0, preutterance=200.0, overlap=100.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("japanese_cvvc_terminal_silence_vowel_end_anchor",),
    )

    repaired_early, _vc, repaired_final, _terminal = repair_cvvc_row_sequence(
        [early_vowel, vc, final_vowel, terminal],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_early.timing == early_vowel.timing
    assert "cvvc_terminal_standalone_v_repair" not in repaired_early.applied_rules
    assert repaired_final.timing.offset == pytest.approx(2854.0)
    assert "cvvc_terminal_standalone_v_repair" in repaired_final.applied_rules


def test_japanese_cvvc_alias_pitch_suffix_is_ignored_for_phone_matching():
    assert _alias_phone_sequence("i bA3") == ["i", "b"]
    assert _alias_phone_sequence("n nyA3") == ["n", "n", "y"]
    assert _alias_phone_sequence("u vA3") == ["u", "v"]
    assert _alias_phone_sequence("\u3042A3") == ["a"]
    assert _alias_phone_sequence("\u3044BA3") == ["i"]
    assert _alias_phone_sequence("a RBA3") == ["a", "r"]
    assert _is_cvvc_terminal_release_r_alias("a RBA3")
    assert _cvvc_initial_cv_head_vowel_token("- \u3044BA3") == "i"
    assert _alias_phone_sequence("baC4S") == ["b", "a"]
    assert _alias_phone_sequence("a ssC4S") == ["a", "s", "s"]
    assert _alias_phone_sequence("- baC4S") == ["b", "a"]
    assert _alias_phone_sequence("tsBA3") == ["ts"]
    assert _alias_phone_sequence("a tsBA3") == ["a", "ts"]
    assert _alias_phone_sequence("br1") == []
    assert _alias_phone_sequence("A3") == []
    assert _alias_type_for_row("i bA3", "auto") == "vc"
    assert _alias_type_for_row("u vA3", "auto") == "vc"
    assert _alias_type_for_row("a ssC4S", "auto") == "vc"
    assert _alias_type_for_row("\u3042A3", "auto") == "v"


def test_japanese_cvvc_nonmonotonic_vowel_targets_float_ambiguous_rows():
    rows = [
        OtoTemplateRow("_aaiouea.wav", "- \u3044BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "a \u3044BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "\u3048BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "i \u3042BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "a \u3046BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "u \u3048BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "e \u3042BA3", OtoTiming(0, 0, 0, 0, 0), ""),
        OtoTemplateRow("_aaiouea.wav", "a \u3042BA3", OtoTiming(0, 0, 0, 0, 0), ""),
    ]

    targets = _assign_alias_target_indices(rows, ["a", "a", "i", "a", "u", "e", "a"], language="japanese")

    assert targets == [2, 2, None, 3, 4, 5, 6, None]


def test_japanese_cvvc_initial_cv_head_duplicate_suffix_is_local_only():
    assert _alias_phone_sequence("- i1") == []
    assert _alias_phone_sequence("a k2") == ["a"]
    assert _cvvc_initial_cv_head_vowel_token("- i1") == "i"
    assert _cvvc_initial_cv_head_phone_sequence("- k1") == ["k"]
    assert _cvvc_initial_cv_head_phone_sequence("- g1") == ["g"]


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


def test_japanese_cvvc_pure_vowel_jump_repairs_late_transition_cascade():
    previous_v = AdaptedOtoRow(
        wav="case.wav",
        alias="e",
        timing=OtoTiming(offset=1280.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )
    late_vv = AdaptedOtoRow(
        wav="case.wav",
        alias="e e",
        timing=OtoTiming(offset=2460.0, consonant=160.0, cutoff=-760.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2580.0, role="vv", expected_phone_index=3),
        mode="bootstrap",
    )
    late_v = AdaptedOtoRow(
        wav="case.wav",
        alias="\u3048",
        timing=OtoTiming(offset=2490.0, consonant=100.0, cutoff=-360.0, preutterance=90.0, overlap=45.0),
        source_timing=None,
        anchor=OtoAnchor(anchor_abs_ms=2580.0, role="v", expected_phone_index=3),
        mode="bootstrap",
        applied_rules=("cvvc_standalone_vowel_compact_profile",),
    )

    repaired_previous, repaired_vv, repaired_v = repair_cvvc_row_sequence(
        [previous_v, late_vv, late_v],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_previous.timing == previous_v.timing
    assert repaired_vv.timing.offset == pytest.approx(1530.0)
    assert repaired_vv.timing.preutterance == pytest.approx(110.0)
    assert repaired_vv.timing.overlap == pytest.approx(100.0)
    assert repaired_vv.timing.consonant == pytest.approx(210.0)
    assert repaired_vv.timing.cutoff == pytest.approx(-600.0)
    assert repaired_v.timing.offset == pytest.approx(1640.0)
    assert repaired_v.timing.preutterance == pytest.approx(90.0)
    assert "cvvc_pure_vowel_jump_repair" in repaired_vv.applied_rules
    assert "cvvc_pure_vowel_jump_v_repair" in repaired_v.applied_rules


def test_japanese_cvvc_standalone_vowel_uses_compact_profile_for_stale_span():
    adapted = bootstrap_row(
        "_a.wav",
        "\u30421",
        OtoAnchor(
            anchor_abs_ms=1560.0,
            score=0.72,
            role="v",
            vowel_nucleus_abs_ms=650.0,
            vowel_start_abs_ms=500.0,
            vowel_end_abs_ms=3170.0,
            expected_phone_index=2,
            source_event_label="cv_boundary",
        ),
        file_duration_ms=3960.0,
        config=OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(1470.0)
    assert adapted.timing.preutterance == pytest.approx(90.0)
    assert adapted.timing.overlap == pytest.approx(45.0)
    assert adapted.timing.consonant == pytest.approx(100.0)
    assert adapted.timing.cutoff == pytest.approx(-360.0)
    assert "cvvc_standalone_vowel_compact_profile" in adapted.applied_rules
    assert "cvvc_standalone_vowel_stale_nucleus:650.0->1560.0" in adapted.warnings


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


def test_japanese_cvvc_sonorant_yoon_vc_does_not_snap_to_initial_head():
    head = AdaptedOtoRow(
        wav="case.wav",
        alias="- \u308a\u3083",
        timing=OtoTiming(offset=400.0, consonant=160.0, cutoff=-170.0, preutterance=120.0, overlap=85.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
    )
    first_vc = AdaptedOtoRow(
        wav="case.wav",
        alias="a ry",
        timing=OtoTiming(offset=895.0, consonant=210.0, cutoff=-236.0, preutterance=170.0, overlap=135.0),
        source_timing=None,
        anchor=None,
        mode="bootstrap",
        warnings=("cvvc_vc_left_context_pre_capped:235.0->170.0",),
    )

    _head, repaired_vc = repair_cvvc_row_sequence(
        [head, first_vc],
        OtoAdapterConfig(language="japanese", format_type="CVVC", alias_type="auto"),
    )

    assert repaired_vc.timing.offset == pytest.approx(895.0)
    assert "cvvc_sonorant_yoon_vc_repair" not in repaired_vc.applied_rules


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


def test_cvvc_hsmm_cv_boundary_keeps_detected_preutterance_position():
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
    assert adapted.timing.offset == pytest.approx(150.0)
    assert adapted.timing.offset + adapted.timing.preutterance == pytest.approx(300.0)
    assert not any(warning.startswith("hsmm_anchor_lead:") for warning in adapted.warnings)


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

    assert adapted.timing.offset == pytest.approx(4310.0)
    assert adapted.timing.preutterance == pytest.approx(110.0)
    assert adapted.timing.overlap == pytest.approx(100.0)
    assert adapted.timing.consonant == pytest.approx(210.0)
    assert adapted.timing.cutoff == pytest.approx(-1590.0)
    assert "terminal_silence_vowel_end_anchored:4400.0" in adapted.warnings
    assert "japanese_cvvc_terminal_silence_vowel_end_anchor" in adapted.applied_rules


def test_japanese_cvvc_terminal_silence_accepts_duplicate_suffix():
    adapted = bootstrap_row(
        "a.wav",
        "a -8",
        OtoAnchor(
            anchor_abs_ms=3000.0,
            score=0.9,
            role="vc",
            vowel_end_abs_ms=3200.0,
            source_event_label="vowel_nucleus",
            warnings=("event_source:filename_hsmm",),
        ),
        file_duration_ms=3980.0,
        config=OtoAdapterConfig(language="japanese", format_type="cvvc", alias_type="auto"),
    )

    assert adapted.timing.offset == pytest.approx(3110.0)
    assert adapted.timing.preutterance == pytest.approx(110.0)
    assert adapted.timing.overlap == pytest.approx(100.0)
    assert adapted.timing.consonant == pytest.approx(210.0)
    assert adapted.timing.cutoff == pytest.approx(-870.0)
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


def test_pitch_suffixed_breath_alias_preserves_nonzero_source_timing():
    from core.generation.common.validation_diagnostics import adapter_warning_validation_metrics

    source_timing = OtoTiming(755.33, 510.43, -510.43, 195.01, 94.1)
    row = OtoTemplateRow("br.wav", "brC4S", source_timing)

    adapted = adapt_template_row(
        row,
        OtoAnchor(anchor_abs_ms=1600.0, score=0.9, role="cv"),
        file_duration_ms=3718.0,
        config=OtoAdapterConfig(language="korean", format_type="cvc", alias_type="auto"),
    )

    assert _is_nonphonetic_special_alias("brC4S")
    assert _is_nonphonetic_special_alias("brC4")
    assert not _is_nonphonetic_special_alias("briC4S")
    assert adapted.timing == source_timing
    assert adapted.source_timing == source_timing
    assert adapted.mode == "template-special"
    assert "special_alias_source_timing_preserved" in adapted.warnings
    assert "special_alias_source_timing_preserve" in adapted.applied_rules
    assert adapter_warning_validation_metrics(adapted.to_json_dict())[
        "special_alias_source_timing_preserve_count"
    ] == pytest.approx(1.0)


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


def test_model_forward_includes_optional_slot_event_head():
    torch = pytest.importorskip("torch")
    from core.mfa_free_oto.model import MfaFreeFrameModelConfig, build_frame_model

    model = build_frame_model(
        MfaFreeFrameModelConfig(
            input_dim=8,
            hidden_dim=16,
            layers=1,
            slot_event_bins=6,
            slot_event_position_features=True,
            slot_event_query_conditioned=True,
        )
    )
    out = model(torch.zeros((2, 12, 8), dtype=torch.float32), slot_count=4)
    assert tuple(out["slot_event_logits"].shape) == (2, 12, 6, len(EVENT_LABELS))


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
    assert report["restored_vv_overlap_rows"] == 1
    assert report["capped_overlap_rows"] == 1
    assert lines[0] == "a.wav=V -,4000,320,-90,200,100"
    assert lines[1] == "br1.wav=br1,2191.38,185.713,-186.713,167.309,15.058"
    assert lines[2] == "a.wav=a k,980,220,-240,170,65"
    assert lines[3] == "a.wav=a \u3042,1420,250,-260,180,85"
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


def test_no_mfa_lightgbm_safety_filter_does_not_create_delayed_underscore_slot_shift(tmp_path):
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

    assert report["shifted_underscore_kana_delayed_slot_rows"] == 0
    assert post.read_text(encoding="utf-8").splitlines() == lines


def test_no_mfa_lightgbm_safety_filter_restores_ml_late_underscore_kana_slots(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre_lines = [
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
    post_lines = list(pre_lines)
    post_lines[5] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=u f,2640.95,190,-220,150,50"
    post_lines[6] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3047,2820,190,-360,70,25"
    post_lines[7] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=e f,3209.3,190,-220,150,50"
    post_lines[8] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3049,3370,190,-360,70,25"
    post_lines[9] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=o f,3578.31,190,-220,150,50"
    post_lines[10] = "_\u3075\u3041\u3075\u3043\u3075\u3075\u3047\u3075\u3049\u3075\u3041\u3093\u3075\u3041.wav=\u3075\u3041,3700,190,-360,70,25"
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

    assert report["shifted_underscore_kana_delayed_slot_rows"] == 6
    repaired_lines = post.read_text(encoding="utf-8").splitlines()
    for index in range(5, 11):
        pre_offset = float(pre_lines[index].split(",", 2)[1])
        repaired_offset = float(repaired_lines[index].split(",", 2)[1])
        assert repaired_offset == pytest.approx(pre_offset)


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


def test_no_mfa_lightgbm_safety_filter_restores_standalone_vowel_offset_cutoff_only(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("v.wav=a,100,130,-300,20,10\n", encoding="utf-8")
    post.write_text("v.wav=a,210,150,-680,30,12\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_initial_standalone_vowel_rows"] == 0
    assert report["restored_standalone_vowel_offset_cutoff_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "v.wav=a,100,150,-300,30,12"


def test_no_mfa_lightgbm_safety_filter_restores_cv_head_cutoff_only_then_caps_profile(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("head.wav=- \u3042,1040,160,-430,120,85\n", encoding="utf-8")
    post.write_text("head.wav=- \u3042,1080,180,-890,130,95\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_cv_head_cutoff_rows"] == 1
    assert report["capped_cv_head_vowel_pre_overlap_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "head.wav=- \u3042,1080,180,-430,90,45"


def test_no_mfa_lightgbm_safety_filter_repairs_post_lightgbm_cv_head_parameter_order(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("_ki.wav=- \u304d,510,160,-170,120,85\n", encoding="utf-8")
    post.write_text("_ki.wav=- \u304d,512,174.84,-170,108,70.2\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["repaired_parameter_order_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "_ki.wav=- \u304d,512,174.84,-182.84,108,70.2"


def test_no_mfa_lightgbm_safety_filter_restores_spaced_consonant_only(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter
    from core.generation.common.oto_alias_family import alias_family

    assert alias_family("ka a") == "spaced"

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("row.wav=ka a,1000,210,-300,170,80\n", encoding="utf-8")
    post.write_text("row.wav=ka a,980,315,-330,180,85\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_spaced_consonant_rows"] == 1
    assert post.read_text(encoding="utf-8").strip() == "row.wav=ka a,980,210,-330,180,85"


def test_no_mfa_lightgbm_safety_filter_restores_vv_overlap_only(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter
    from core.generation.common.oto_alias_family import alias_family

    assert alias_family("a \u3042") == "vv"

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("vv.wav=a \u3042,100,420,-600,300,100\n", encoding="utf-8")
    post.write_text("vv.wav=a \u3042,120,450,-650,280,25\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_vv_overlap_rows"] == 1
    assert report["capped_overlap_rows"] == 0
    assert post.read_text(encoding="utf-8").strip() == "vv.wav=a \u3042,120,450,-650,280,100"


def test_no_mfa_lightgbm_safety_filter_restores_terminal_vc_overlap_only(tmp_path):
    from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter
    from core.generation.common.oto_alias_family import alias_family

    assert alias_family("a -6") == "vc"

    pre = tmp_path / "pre.ini"
    post = tmp_path / "post.ini"
    pre.write_text("tail.wav=a -6,1000,220,-240,170,30\n", encoding="utf-8")
    post.write_text("tail.wav=a -6,1010,230,-250,180,80\n", encoding="utf-8")

    report = apply_no_mfa_lightgbm_safety_filter(
        pre_oto_path=str(pre),
        post_oto_path=str(post),
        language="japanese",
        format_type="cvvc",
    )

    assert report["restored_terminal_vc_overlap_rows"] == 1
    assert report["capped_overlap_rows"] == 0
    assert post.read_text(encoding="utf-8").strip() == "tail.wav=a -6,1010,230,-250,180,30"


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


def test_source_oto_written_order_is_not_used_for_cvvc_template_order():
    rows = [
        parse_template_oto_line("_まみむめもまんま.wav=a m,0,0,0,0,0"),
        parse_template_oto_line("_まみむめもまんま.wav=み,0,0,0,0,0"),
        parse_template_oto_line("_まみむめもまんま.wav=ま,0,0,0,0,0"),
        parse_template_oto_line("_まみむめもまんま.wav=- ま,0,0,0,0,0"),
        parse_template_oto_line("_まみむめもまんま.wav=i m,0,0,0,0,0"),
    ]

    ordered = _template_group_in_filename_order(
        "_まみむめもまんま.wav",
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == ["- ま", "a m", "み", "i m", "ま"]


def test_japanese_cvvc_template_order_matches_l_aliases_to_filename_r_slots():
    wav = "_l\u3089l\u308al\u308bl\u308cl\u308dl\u3089\u3093l\u3089.wav"
    aliases = [
        "- l\u3089",
        "l\u3089",
        "l\u308a",
        "l\u308b",
        "l\u308c",
        "l\u308d",
        "a l",
        "e l",
        "i l",
        "n l",
        "o l",
        "u l",
    ]
    rows = [
        OtoTemplateRow(wav, alias, OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0))
        for alias in aliases
    ]

    ordered = _template_group_in_filename_order(
        wav,
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == [
        "- l\u3089",
        "a l",
        "l\u308a",
        "i l",
        "l\u308b",
        "u l",
        "l\u308c",
        "e l",
        "l\u308d",
        "o l",
        "l\u3089",
        "n l",
    ]
    phones = filename_phone_sequence_from_slots(
        build_filename_slots(wav, language="japanese", format_type="cvvc")
    )
    assert _assign_alias_target_indices(ordered, phones, language="japanese") == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
    ]


def test_cvvc_template_plain_cv_skips_initial_duplicate_slot():
    rows = [
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=pa,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=n p,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=o p,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=po,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=a p,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=pi,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=i p,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=pu,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=u p,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=pe,0,0,0,0,0"),
        parse_template_oto_line("pa-pi-pu-pe-po-pa-N-pa.wav=e p,0,0,0,0,0"),
    ]

    ordered = _template_group_in_filename_order(
        "pa-pi-pu-pe-po-pa-N-pa.wav",
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == [
        "a p",
        "pi",
        "i p",
        "pu",
        "u p",
        "pe",
        "e p",
        "po",
        "o p",
        "pa",
        "n p",
    ]


def test_cvvc_template_consonant_only_cv_head_matches_initial_slot():
    rows = [
        parse_template_oto_line("ma-mi-mu-me-mo-ma-N-ma.wav=a m,0,0,0,0,0"),
        parse_template_oto_line("ma-mi-mu-me-mo-ma-N-ma.wav=mi,0,0,0,0,0"),
        parse_template_oto_line("ma-mi-mu-me-mo-ma-N-ma.wav=ma,0,0,0,0,0"),
        parse_template_oto_line("ma-mi-mu-me-mo-ma-N-ma.wav=- m,0,0,0,0,0"),
        parse_template_oto_line("ma-mi-mu-me-mo-ma-N-ma.wav=i m,0,0,0,0,0"),
    ]

    ordered = _template_group_in_filename_order(
        "ma-mi-mu-me-mo-ma-N-ma.wav",
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == ["- m", "a m", "mi", "i m", "ma"]


def test_cvvc_template_repeated_glide_aliases_follow_source_target_occurrence():
    wav = "wa-wi-we-wo-wa-u-wa-N-wa.wav"
    aliases = [
        "- w",
        "a w",
        "i w",
        "e w",
        "o w",
        "u w",
        "n w",
        "\u3046\u3043",
        "\u3046\u3047",
        "\u3092",
        "\u308f",
    ]
    rows = [
        OtoTemplateRow(
            wav,
            alias,
            OtoTiming(0.0, 0.0, 0.0, 0.0, 0.0),
            source_row_index=index,
        )
        for index, alias in enumerate(aliases)
    ]

    ordered = _template_group_in_filename_order(
        wav,
        rows,
        language="japanese",
        format_type="cvvc",
    )
    phones = filename_phone_sequence_from_slots(
        build_filename_slots(wav, language="japanese", format_type="cvvc")
    )

    assert [row.alias for row in ordered] == [
        "- w",
        "a w",
        "\u3046\u3043",
        "i w",
        "\u3046\u3047",
        "e w",
        "\u3092",
        "o w",
        "\u308f",
        "u w",
        "n w",
    ]
    assert _assign_alias_target_indices(ordered, phones, language="japanese") == [
        2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        14,
    ]


def test_source_oto_filename_order_matches_yoon_vc_surface_aliases():
    rows = [
        parse_template_oto_line("_きゃききゅ.wav=きゅ,0,0,0,0,0"),
        parse_template_oto_line("_きゃききゅ.wav=i ky,0,0,0,0,0"),
        parse_template_oto_line("_きゃききゅ.wav=a ky,0,0,0,0,0"),
        parse_template_oto_line("_きゃききゅ.wav=- きゃ,0,0,0,0,0"),
    ]

    ordered = _template_group_in_filename_order(
        "_きゃききゅ.wav",
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == ["- きゃ", "a ky", "i ky", "きゅ"]


def test_source_oto_filename_order_matches_split_cv_head_rows():
    rows = [
        parse_template_oto_line("_うぃ_うぅ_うぇ_うぉ.wav=- うぇ,0,0,0,0,0"),
        parse_template_oto_line("_うぃ_うぅ_うぇ_うぉ.wav=- うぃ,0,0,0,0,0"),
        parse_template_oto_line("_うぃ_うぅ_うぇ_うぉ.wav=- うぉ,0,0,0,0,0"),
        parse_template_oto_line("_うぃ_うぅ_うぇ_うぉ.wav=- うぅ,0,0,0,0,0"),
    ]

    ordered = _template_group_in_filename_order(
        "_うぃ_うぅ_うぇ_うぉ.wav",
        rows,
        language="japanese",
        format_type="cvvc",
    )

    assert [row.alias for row in ordered] == ["- うぃ", "- うぅ", "- うぇ", "- うぉ"]


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
        "base_a_second",
        "base_i",
    ]
    assert "999,888,-777,666,555" not in out_oto.read_text(encoding="utf-8")
    assert any(record["selected_event_source"] == "filename_hsmm" for record in report.timeline_debug)
    assert "base_oto_alias_list_used" in report.warnings
    assert "base_oto_alias_order_ignored" in report.warnings


def test_hsmm_workflow_matches_apostrophe_template_to_underscore_wav(tmp_path, monkeypatch):
    # Base OTO references reclist apostrophes (ka'ki.wav) but the recordings were
    # saved with underscores (ka_ki.wav). The rows must NOT be dropped; the output
    # must reference the real on-disk filename.
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    _write_tone_wav(wav_dir / "ka_ki.wav", duration_s=0.40)
    source_oto = tmp_path / "source.ini"
    source_oto.write_text("ka'ki.wav=ka,999,888,-777,666,555\n", encoding="utf-8")
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
            "voicing": [0.1, 0.2, 0.7, 0.9, 0.7, 0.2, 0.1],
            "nucleus_likelihood": [0.1, 0.2, 0.6, 0.95, 0.6, 0.2, 0.1],
        },
    )
    monkeypatch.setattr(
        "core.mfa_free_oto.workflow.predict_wav",
        lambda *a, **k: SimpleNamespace(posterior=posterior, decoded_events=(), slot_result=None),
    )

    report = generate_no_mfa_oto_with_model_context(
        wav_dir=str(wav_dir),
        out_path=str(out_oto),
        source_oto_path=str(source_oto),
        checkpoint_path="",
        language="korean",
        format_type="CVC",
        use_hsmm_decoder=True,
    )

    assert report.processed >= 1
    body = out_oto.read_text(encoding="utf-8")
    # Row is produced and references the actual underscore filename, not dropped.
    assert "ka_ki.wav=" in body
    assert "ka'ki.wav=" not in body
    assert any(str(w).startswith("wav_name_normalized:") for w in report.warnings)


def test_mfa_free_preview_template_preserve_applies_cvvc_profile_repair(tmp_path, monkeypatch):
    from ml.scripts.mfa_free_oto import predict_oto_preview

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "ki_ku.wav"
    _write_tone_wav(wav_path, duration_s=0.45)
    source_oto = wav_dir / "oto.ini"
    source_oto.write_text("ki_ku.wav=ki,0,0,0,0,0\n", encoding="utf-8")

    times = [0.0, 40.0, 80.0]
    posterior = FramePosterior(
        wav_path=str(wav_path),
        times_ms=times,
        class_probs={
            "silence": [0.05, 0.05, 0.05],
            "consonant": [0.80, 0.70, 0.20],
            "vowel": [0.10, 0.20, 0.75],
            "other": [0.05, 0.05, 0.05],
        },
        event_scores={label: [0.0 for _ in times] for label in EVENT_LABELS},
        acoustic_scores={},
        metadata={},
    )
    prediction = SimpleNamespace(
        posterior=posterior,
        slot_result=None,
        decoded_events=[],
        to_json_dict=lambda: {"stub": True},
    )

    monkeypatch.setattr(predict_oto_preview, "predict_wav", lambda *args, **kwargs: prediction)
    monkeypatch.setattr(
        predict_oto_preview,
        "assign_template_row_anchors",
        lambda posterior, event_source, template_rows, **kwargs: [
            OtoAnchor(anchor_abs_ms=250.0, score=0.8, role="cv_boundary", source_event_label="cv_boundary")
            for _row in template_rows
        ],
    )

    def fake_adapt_template_row(template_row, anchor, *, file_duration_ms, config):
        return AdaptedOtoRow(
            wav=template_row.wav,
            alias=template_row.alias,
            timing=OtoTiming(
                offset=100.0,
                consonant=190.0,
                cutoff=-900.0,
                preutterance=150.0,
                overlap=115.0,
            ),
            source_timing=template_row.timing,
            anchor=anchor,
            mode=config.mode,
        )

    monkeypatch.setattr(predict_oto_preview, "adapt_template_row", fake_adapt_template_row)
    out_oto = tmp_path / "preview.ini"
    out_json = tmp_path / "anchors.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_oto_preview",
            "--wav-dir",
            str(wav_dir),
            "--source-oto",
            str(source_oto),
            "--out-oto",
            str(out_oto),
            "--out-json",
            str(out_json),
            "--language",
            "japanese",
            "--format-type",
            "CVVC",
        ],
    )

    assert predict_oto_preview.main() == 0
    assert out_oto.read_text(encoding="utf-8").strip() == (
        "ki_ku.wav=ki,180.000,150.000,-350.000,70.000,25.000"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    row = payload["rows"][0]["generated_oto_rows"][0]
    assert "cvvc_cv_role_profile_repair" in row["applied_rules"]
    assert any(warning.startswith("cvvc_cv_role_profile_repaired:") for warning in row["warnings"])


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


def test_hsmm_row_provenance_flattens_timeline_rows():
    from core.mfa_free_oto.review_generation import _row_provenance_records_from_timeline_records

    records = [
        {
            "wav": "_ma-mi.wav",
            "selected_event_source": "filename_hsmm",
            "hsmm": {"ok": True, "reason": "ok", "score": 0.75},
            "adapted_rows": [
                {
                    "row_index": 1,
                    "wav": "_ma-mi.wav",
                    "alias": "a m",
                    "mode": "template-preserve",
                    "timing": {"offset": 740.0},
                    "absolute": {"preutterance_abs": 820.0},
                    "anchor": {
                        "anchor_abs_ms": 900.0,
                        "score": 0.5,
                        "role": "vc",
                        "source_event_label": "phone_change",
                        "expected_phone": "m",
                        "expected_phone_index": 1,
                        "slot_index": 1,
                        "warnings": ["event_source:filename_hsmm"],
                    },
                    "row_plan": {
                        "role_family": "vc",
                        "slot_index": 1,
                        "left_slot_index": 0,
                        "right_slot_index": 1,
                        "expected_phone_index": 1,
                    },
                    "applied_rules": ["cvvc_internal_vc_slot_bound_repair"],
                    "warnings": ["cvvc_internal_vc_slot_bound:430.0->740.0"],
                }
            ],
        }
    ]

    provenance = _row_provenance_records_from_timeline_records(records)

    assert len(provenance) == 1
    row = provenance[0]
    assert row["wav"] == "_ma-mi.wav"
    assert row["alias"] == "a m"
    assert row["role"] == "vc"
    assert row["selected_event_source"] == "filename_hsmm"
    assert row["slot"] == {
        "slot_index": 1,
        "left_slot_index": 0,
        "right_slot_index": 1,
        "expected_phone_index": 1,
    }
    assert row["anchor"]["source_event_label"] == "phone_change"
    assert row["hsmm"]["ok"] is True
    assert row["applied_rules"] == ["cvvc_internal_vc_slot_bound_repair"]


def test_hsmm_row_provenance_diagnosis_tags_vc_head_attachment(tmp_path):
    from scripts.dev.mfa_free_oto_review import _build_row_provenance_diagnosis

    rows = [
        {
            "wav": "ma_mi.wav",
            "alias": "- ま",
            "role": "cv_head",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 0, "left_slot_index": 0, "right_slot_index": 0, "expected_phone_index": 0},
            "absolute": {"offset_abs": 200.0},
            "applied_rules": [],
            "warnings": [],
        },
        {
            "wav": "ma_mi.wav",
            "alias": "a m",
            "role": "vc",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 1, "left_slot_index": 0, "right_slot_index": 1, "expected_phone_index": 1},
            "absolute": {"offset_abs": 210.0},
            "applied_rules": ["cvvc_headed_regular_vc_from_next_cv_repair"],
            "warnings": ["local_refine_delta_ms:-90.0", "local_refine_low_margin"],
        },
        {
            "wav": "ma_mi.wav",
            "alias": "ま",
            "role": "cv",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 1, "left_slot_index": 1, "right_slot_index": 1, "expected_phone_index": 1},
            "absolute": {"offset_abs": 820.0},
            "applied_rules": [],
            "warnings": [],
        },
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    report = _build_row_provenance_diagnosis(path, max_rows=10)

    assert report["ok"] is True
    assert report["role_counts"]["vc"] == 1
    assert report["tag_counts"]["vc_tied_to_prev_cv"] == 1
    assert report["tag_counts"]["vc_before_prev_safe_window"] == 1
    assert report["tag_counts"]["vc_sonorant_context"] == 1
    assert report["tag_counts"]["rule_headed_regular_vc_from_next_cv"] == 1
    suspect = report["suspect_rows"][0]
    assert suspect["alias"] == "a m"
    assert suspect["context"]["delta_from_prev_context_ms"] == 10.0
    assert suspect["context"]["delta_to_next_context_ms"] == 610.0


def test_hsmm_row_provenance_diagnosis_tags_cv_head_context_drift(tmp_path):
    from scripts.dev.mfa_free_oto_review import _build_row_provenance_diagnosis

    rows = [
        {
            "wav": "ma_mi.wav",
            "alias": "- m",
            "role": "cv_head",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 0, "left_slot_index": 0, "right_slot_index": 0, "expected_phone_index": 0},
            "absolute": {"offset_abs": 40.0},
            "applied_rules": [],
            "warnings": [],
        },
        {
            "wav": "ma_mi.wav",
            "alias": "mi",
            "role": "cv",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 1, "left_slot_index": 1, "right_slot_index": 1, "expected_phone_index": 1},
            "absolute": {"offset_abs": 1000.0},
            "applied_rules": [],
            "warnings": [],
        },
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    report = _build_row_provenance_diagnosis(path, max_rows=10)

    assert report["tag_counts"]["cv_head_at_file_head"] == 1
    assert report["tag_counts"]["cv_head_far_before_next_context"] == 1
    assert report["tag_counts_by_role"]["cv_head"]["cv_head_far_before_next_context"] == 1
    suspect = report["suspect_rows"][0]
    assert suspect["alias"] == "- m"
    assert suspect["context"]["delta_to_next_context_ms"] == 960.0


def test_hsmm_row_provenance_diagnosis_tags_glide_and_yoon_vc(tmp_path):
    from scripts.dev.mfa_free_oto_review import _build_row_provenance_diagnosis

    rows = [
        {
            "wav": "ya_kya.wav",
            "alias": "- y",
            "role": "cv_head",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 0, "left_slot_index": 0, "right_slot_index": 0, "expected_phone_index": 0},
            "absolute": {"offset_abs": 200.0},
            "applied_rules": [],
            "warnings": [],
        },
        {
            "wav": "ya_kya.wav",
            "alias": "a y",
            "role": "vc",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 1, "left_slot_index": 0, "right_slot_index": 1, "expected_phone_index": 1},
            "absolute": {"offset_abs": 350.0},
            "applied_rules": [],
            "warnings": ["local_refine_low_margin"],
        },
        {
            "wav": "ya_kya.wav",
            "alias": "ya",
            "role": "cv",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 1, "left_slot_index": 1, "right_slot_index": 1, "expected_phone_index": 1},
            "absolute": {"offset_abs": 500.0},
            "applied_rules": [],
            "warnings": [],
        },
        {
            "wav": "ya_kya.wav",
            "alias": "a ky",
            "role": "vc",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 2, "left_slot_index": 1, "right_slot_index": 2, "expected_phone_index": 2},
            "absolute": {"offset_abs": 800.0},
            "applied_rules": [],
            "warnings": [],
        },
        {
            "wav": "ya_kya.wav",
            "alias": "kya",
            "role": "cv",
            "selected_event_source": "filename_hsmm",
            "slot": {"slot_index": 2, "left_slot_index": 2, "right_slot_index": 2, "expected_phone_index": 2},
            "absolute": {"offset_abs": 1000.0},
            "applied_rules": [],
            "warnings": [],
        },
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    report = _build_row_provenance_diagnosis(path, max_rows=0)

    assert report["tag_counts"]["vc_glide_context"] == 1
    assert report["tag_counts"]["vc_yoon_context"] == 1
    assert report["tag_counts_by_role"]["vc"]["vc_glide_context"] == 1
    assert report["tag_counts_by_role"]["vc"]["vc_yoon_context"] == 1


def test_hsmm_oto_ui_preview_uses_review_generate_hsmm(tmp_path, monkeypatch):
    from ui.pipeline_actions_mixin import PipelineActionsMixin

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.logs = []
            self.writable_data_dir = str(tmp_path)
            self.app_data_dir = ""
            self.app_dir = str(tmp_path)

        def _append_log(self, message):
            self.logs.append(str(message))

    calls = []

    def _fake_generate_hsmm_oto_review(**kwargs):
        calls.append(dict(kwargs))
        generated_path = tmp_path / "generated.ini"
        generated_path.write_text("a.wav=a,0,0,0,0,0\nb.wav=i,0,0,0,0,0\n", encoding="utf-8")
        if kwargs.get("callback"):
            kwargs["callback"]("fake decoder progress")
        return {
            "ok": True,
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
        }

    monkeypatch.setattr("ui.pipeline_actions_mixin.generate_hsmm_oto_review", _fake_generate_hsmm_oto_review)

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
    assert calls
    call = calls[0]
    assert call["template_oto"] == ""
    assert call["encoder"] == "acoustic_world_v1"
    assert call["alias_type"] == "auto"
    assert out_path.read_text(encoding="utf-8").count("\n") == 2
    assert any("no base oto" in log for log in dummy.logs)
    assert any("fake decoder progress" in log for log in dummy.logs)
    assert any("review_session:" in log for log in dummy.logs)


def test_hsmm_oto_ui_preview_can_request_lightgbm_postprocess(tmp_path, monkeypatch):
    from ui.pipeline_actions_mixin import PipelineActionsMixin

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.logs = []
            self.writable_data_dir = str(tmp_path)
            self.app_data_dir = ""
            self.app_dir = str(tmp_path)

        def _append_log(self, message):
            self.logs.append(str(message))

    calls = []

    def _fake_generate_hsmm_oto_review(**kwargs):
        calls.append(dict(kwargs))
        generated_path = tmp_path / "generated.ini"
        pre_lightgbm_path = tmp_path / "pre_lightgbm.ini"
        generated_path.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
        pre_lightgbm_path.write_text("a.wav=a,0,0,0,0,0\n", encoding="utf-8")
        return {
            "ok": True,
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
        }

    monkeypatch.setattr("ui.pipeline_actions_mixin.generate_hsmm_oto_review", _fake_generate_hsmm_oto_review)

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
    call = calls[0]
    assert call["apply_lightgbm"] is True
    assert call["lightgbm_policy"] == "on"
    assert call["lightgbm_model_dir"] == str(model_dir.resolve())
    assert any("LightGBM postprocess requested" in log for log in dummy.logs)
    assert any("LightGBM postprocess: status=applied, changed=1" in log for log in dummy.logs)
    assert any("pre-LightGBM oto:" in log for log in dummy.logs)


def test_hsmm_oto_ui_preview_passes_base_oto_as_template(tmp_path, monkeypatch):
    from ui.pipeline_actions_mixin import PipelineActionsMixin

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.logs = []
            self.writable_data_dir = str(tmp_path)
            self.app_data_dir = ""
            self.app_dir = str(tmp_path)

        def _append_log(self, message):
            self.logs.append(str(message))

    calls = []

    def _fake_generate_hsmm_oto_review(**kwargs):
        calls.append(dict(kwargs))
        generated_path = tmp_path / "generated.ini"
        generated_path.write_text("a.wav=base alias,0,0,0,0,0\n", encoding="utf-8")
        return {
            "ok": True,
            "processed": 1,
            "total": 1,
            "generated_oto_path": str(generated_path),
            "split_counts": {"total": 1, "fix_required": 0, "attention_only": 0, "clean": 1},
            "split_output_paths": {},
        }

    monkeypatch.setattr("ui.pipeline_actions_mixin.generate_hsmm_oto_review", _fake_generate_hsmm_oto_review)

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
    assert calls[0]["template_oto"] == str(source_oto.resolve())
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


def test_pre_assigned_phone_indices_bypass_dp():
    """When OtoTemplateRow carries expected_phone_indices from RowPlanRecord,
    _alias_targets_from_template_rows_or_dp must use them directly instead of
    re-deriving via the DP alias parser."""
    rows_with_indices = [
        OtoTemplateRow("ka'ki'ku.wav", "ka", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(0, 1)),
        OtoTemplateRow("ka'ki'ku.wav", "a ki", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(1, 2)),
        OtoTemplateRow("ka'ki'ku.wav", "ki", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(2, 3)),
        OtoTemplateRow("ka'ki'ku.wav", "i ku", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(3, 4)),
        OtoTemplateRow("ka'ki'ku.wav", "ku", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(4, 5)),
    ]
    phones = ["k", "a", "k", "i", "k", "u"]
    targets = _alias_targets_from_template_rows_or_dp(rows_with_indices, phones)
    assert targets == [1, 2, 3, 4, 5]

    rows_without = [
        OtoTemplateRow("ka'ki'ku.wav", "ka", OtoTiming(0, 0, 0, 0, 0)),
        OtoTemplateRow("ka'ki'ku.wav", "a ki", OtoTiming(0, 0, 0, 0, 0)),
        OtoTemplateRow("ka'ki'ku.wav", "ki", OtoTiming(0, 0, 0, 0, 0)),
        OtoTemplateRow("ka'ki'ku.wav", "i ku", OtoTiming(0, 0, 0, 0, 0)),
        OtoTemplateRow("ka'ki'ku.wav", "ku", OtoTiming(0, 0, 0, 0, 0)),
    ]
    dp_targets = _alias_targets_from_template_rows_or_dp(rows_without, phones)
    assert dp_targets == _assign_alias_target_indices(rows_without, phones)


def test_pre_assigned_indices_partial_falls_back_to_dp():
    """If only some rows have indices, fall back to DP for all."""
    rows = [
        OtoTemplateRow("test.wav", "ka", OtoTiming(0, 0, 0, 0, 0), expected_phone_indices=(0, 1)),
        OtoTemplateRow("test.wav", "a i", OtoTiming(0, 0, 0, 0, 0)),
    ]
    phones = ["k", "a", "i"]
    targets = _alias_targets_from_template_rows_or_dp(rows, phones)
    assert targets == _assign_alias_target_indices(rows, phones)


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
