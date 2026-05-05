from __future__ import annotations

import numpy as np

from core.coarse_crnn.decode import viterbi_align_phones
from core.coarse_crnn.labels import LABEL_TO_ID, coarse_for_phone


def test_coarse_label_mapping_keeps_singing_sp_as_silence():
    assert coarse_for_phone("SP") == "SIL"
    assert coarse_for_phone("a") == "V"
    assert coarse_for_phone("sh") == "C_FR"
    assert coarse_for_phone("N") == "SP"


def test_viterbi_align_phones_monotonic():
    probs = np.zeros((12, len(LABEL_TO_ID)), dtype=np.float32) + 0.01
    probs[0:2, LABEL_TO_ID["SIL"]] = 0.95
    probs[2:5, LABEL_TO_ID["C_OBS"]] = 0.95
    probs[5:10, LABEL_TO_ID["V"]] = 0.95
    probs[10:12, LABEL_TO_ID["SIL"]] = 0.95
    probs = probs / probs.sum(axis=1, keepdims=True)

    rows = viterbi_align_phones(
        probs,
        ["k", "a"],
        hop_sec=0.01,
        language="japanese",
        duration_sec=0.12,
    )

    assert [row.phone for row in rows] == ["k", "a"]
    assert rows[0].start <= rows[0].end <= rows[1].start <= rows[1].end
    assert rows[0].coarse == "C_OBS"
    assert rows[1].coarse == "V"
