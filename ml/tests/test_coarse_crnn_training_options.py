from __future__ import annotations

import numpy as np

from core.coarse_crnn.training import _boundary_targets_from_frame_targets


def test_boundary_targets_mark_local_frame_transitions():
    targets = np.asarray([1, 1, 2, 2, 2, 3, -100, 3], dtype=np.int64)

    boundaries = _boundary_targets_from_frame_targets(targets)

    assert boundaries.tolist() == [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0]
