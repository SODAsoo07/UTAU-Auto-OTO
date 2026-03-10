import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_mapping_scoring_v2 import resolve_cv_syllable_index


class KrMappingScoringV2Tests(unittest.TestCase):
    def test_resolve_cv_syllable_index_keeps_expected_when_forward_gain_is_small(self):
        current_w_idx, cv_seq_idx, meta = resolve_cv_syllable_index(
            "go",
            ["go", "gyo"],
            0,
            0,
            mapping_confidence=0.7,
            return_meta=True,
        )
        self.assertEqual(current_w_idx, 0)
        self.assertEqual(cv_seq_idx, 1)
        self.assertEqual(meta["chosen_idx"], 0)

    def test_resolve_cv_syllable_index_allows_high_conf_strong_gain_jump(self):
        current_w_idx, cv_seq_idx, meta = resolve_cv_syllable_index(
            "ryeo",
            ["ra", "reo", "ryeo"],
            0,
            0,
            mapping_confidence=0.95,
            max_jump_default=1,
            max_jump_high_conf=2,
            high_conf_threshold=0.82,
            return_meta=True,
        )
        self.assertEqual(current_w_idx, 2)
        self.assertEqual(cv_seq_idx, 3)
        self.assertEqual(meta["max_forward_jump"], 2)
        self.assertEqual(meta["jump_blocked"], 0)

    def test_resolve_cv_syllable_index_blocks_backward_move_without_large_gain(self):
        current_w_idx, cv_seq_idx, meta = resolve_cv_syllable_index(
            "ga",
            ["ga", "go"],
            1,
            1,
            mapping_confidence=0.8,
            return_meta=True,
        )
        self.assertEqual(current_w_idx, 1)
        self.assertEqual(cv_seq_idx, 2)
        self.assertEqual(meta["chosen_idx"], 0)


if __name__ == "__main__":
    unittest.main()
