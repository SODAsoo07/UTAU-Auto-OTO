from __future__ import annotations

import numpy as np
import pytest

from core.coarse_crnn.oto_training import (
    OtoTrainConfig,
    _build_train_sampling_weights,
    _context_array_from_row_with_active,
    _filter_compatible_init_state,
    _iter_device_batches,
)
from core.coarse_crnn.training import _boundary_targets_from_frame_targets


def test_boundary_targets_mark_local_frame_transitions():
    targets = np.asarray([1, 1, 2, 2, 2, 3, -100, 3], dtype=np.int64)

    boundaries = _boundary_targets_from_frame_targets(targets)

    assert boundaries.tolist() == [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0]


def test_cvvc_vc_multi_sampling_boost_only_targets_multi_vc_rows():
    rows = [
        {
            "voicebank_id": "vb",
            "format_type": "cvvc",
            "alias_role": "vc",
            "alias_type": "vc",
            "transition_type": "multi",
        },
        {
            "voicebank_id": "vb",
            "format_type": "cvvc",
            "alias_role": "vc",
            "alias_type": "vc",
            "transition_type": "vc",
        },
        {
            "voicebank_id": "vb",
            "format_type": "cvvc",
            "alias_role": "vv",
            "alias_type": "vv",
            "transition_type": "multi",
        },
    ]
    cfg = OtoTrainConfig(
        voicebank_balance_power=0.0,
        language_balance_power=0.0,
        role_balance_power=0.0,
        format_balance_power=0.0,
        cvvc_vc_sampling_boost=1.0,
        cvvc_vv_sampling_boost=1.0,
        cvvc_vc_multi_sampling_boost=3.0,
    )

    weights = _build_train_sampling_weights(rows, cfg, hard_case_boosts=np.zeros((0,), dtype=np.float32))

    assert weights.tolist() == [3.0, 1.0, 1.0]


def test_language_balance_and_slice_sampling_boosts_are_explicit():
    rows = [
        {
            "voicebank_id": "vb_ko",
            "language": "korean",
            "format_type": "cvvc",
            "alias_role": "vc",
            "alias_type": "vc",
            "transition_type": "multi",
        },
        {
            "voicebank_id": "vb_ko",
            "language": "korean",
            "format_type": "cvvc",
            "alias_role": "cv",
            "alias_type": "cv",
            "transition_type": "multi",
        },
        {
            "voicebank_id": "vb_ja",
            "language": "japanese",
            "format_type": "cvvc",
            "alias_role": "vc",
            "alias_type": "vc",
            "transition_type": "multi",
        },
    ]
    cfg = OtoTrainConfig(
        voicebank_balance_power=0.0,
        language_balance_power=1.0,
        role_balance_power=0.0,
        format_balance_power=0.0,
        cvvc_vc_sampling_boost=1.0,
        cvvc_vv_sampling_boost=1.0,
        cvvc_vc_multi_sampling_boost=1.0,
        language_format_role_sampling_boosts=("korean/cvvc/vc=4.0",),
    )

    weights = _build_train_sampling_weights(rows, cfg, hard_case_boosts=np.zeros((0,), dtype=np.float32))

    assert weights.tolist() == [2.0, 0.5, 1.0]


def test_compatible_init_state_skips_missing_and_shape_mismatch():
    import torch

    source = {
        "shared.weight": torch.ones(2, 3),
        "old_only.weight": torch.ones(1),
        "changed.weight": torch.ones(4, 3),
    }
    target = {
        "shared.weight": torch.zeros(2, 3),
        "changed.weight": torch.zeros(2, 3),
        "new_only.weight": torch.zeros(1),
    }

    compatible, summary = _filter_compatible_init_state(source, target)

    assert set(compatible) == {"shared.weight"}
    assert summary["strategy"] == "compatible"
    assert summary["loaded"] == 1
    assert summary["skipped_missing_count"] == 1
    assert summary["skipped_shape_count"] == 1


def test_iter_device_batches_cpu_path_keeps_batch_count():
    import torch

    loader = [
        (torch.zeros((1, 2), dtype=torch.float32), torch.ones((1,), dtype=torch.float32)),
        (torch.ones((1, 2), dtype=torch.float32), torch.zeros((1,), dtype=torch.float32)),
    ]
    batches = list(_iter_device_batches(loader, torch.device("cpu"), torch=torch, prefetch_batches=2))
    assert len(batches) == 2
    assert all(batch[0].device.type == "cpu" for batch in batches)


def test_active_audio_context_extends_numeric_context_to_18_dims():
    row = {
        "file_row_count": 4,
        "row_ratio_in_wav": 0.25,
        "alias_phone_count": 2,
    }

    context = _context_array_from_row_with_active(
        row,
        active_context=(0.2, 0.9, 0.7),
        expected_dim=18,
    )

    assert context.shape == (18,)
    assert context[-3:].tolist() == pytest.approx([0.2, 0.9, 0.7])


def test_row_order_violation_multiplier_disabled_returns_one():
    """alpha=0 (default) keeps the legacy behaviour for every row."""
    from core.coarse_crnn.oto_training import _row_order_violation_multiplier

    cfg = OtoTrainConfig(row_order_violation_alpha=0.0)
    # High violation row should still get multiplier 1.0 when alpha is 0.
    row = {"row_order_violation_score": 2.5}
    assert abs(_row_order_violation_multiplier(row, cfg) - 1.0) < 1e-9


def test_row_order_violation_multiplier_scales_high_score_down():
    """alpha=0.5 + clipped score 1.0 should give multiplier 1/(1+0.5*1)=0.667."""
    from core.coarse_crnn.oto_training import _row_order_violation_multiplier

    cfg = OtoTrainConfig(row_order_violation_alpha=0.5)
    row = {"row_order_violation_score": 1.0}
    expected = 1.0 / (1.0 + 0.5 * 1.0)
    assert abs(_row_order_violation_multiplier(row, cfg) - expected) < 1e-9


def test_row_order_violation_multiplier_clips_extreme_score():
    """Score above 3 must be clipped to keep weight from collapsing to 0."""
    from core.coarse_crnn.oto_training import _row_order_violation_multiplier

    cfg = OtoTrainConfig(row_order_violation_alpha=1.0)
    # Without clip, score=10 would give 1/11 ≈ 0.09. With clip(0,3) → 1/(1+3) = 0.25.
    row = {"row_order_violation_score": 10.0}
    expected_clipped = 1.0 / (1.0 + 1.0 * 3.0)
    assert abs(_row_order_violation_multiplier(row, cfg) - expected_clipped) < 1e-9


def test_row_order_violation_multiplier_missing_column_falls_back_to_one():
    """Older manifests without the score column should not break training."""
    from core.coarse_crnn.oto_training import _row_order_violation_multiplier

    cfg = OtoTrainConfig(row_order_violation_alpha=0.5)
    # No row_order_violation_score key → 1.0 fallback.
    assert abs(_row_order_violation_multiplier({}, cfg) - 1.0) < 1e-9
    # Non-numeric also falls back.
    assert abs(_row_order_violation_multiplier({"row_order_violation_score": "?"}, cfg) - 1.0) < 1e-9


def test_row_order_violation_multiplier_zero_score_returns_one():
    """A row whose order is perfectly aligned (score=0) gets unchanged weight."""
    from core.coarse_crnn.oto_training import _row_order_violation_multiplier

    cfg = OtoTrainConfig(row_order_violation_alpha=0.5)
    row = {"row_order_violation_score": 0.0}
    assert abs(_row_order_violation_multiplier(row, cfg) - 1.0) < 1e-9
