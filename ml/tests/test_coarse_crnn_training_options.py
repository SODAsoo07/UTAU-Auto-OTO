from __future__ import annotations

import numpy as np

from core.coarse_crnn.oto_training import OtoTrainConfig, _build_train_sampling_weights, _filter_compatible_init_state
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
