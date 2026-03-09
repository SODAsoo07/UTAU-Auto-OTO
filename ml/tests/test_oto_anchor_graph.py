import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_anchor_graph import build_adjacent_anchor_graph, resolve_bridge_anchor_pair


class OtoAnchorGraphTests(unittest.TestCase):
    def test_build_adjacent_anchor_graph_skips_nonforward_duplicate(self):
        graph = build_adjacent_anchor_graph([1, 1, 3, 4])
        self.assertEqual(
            graph["pairs"],
            [
                {"prev_idx": 1, "next_idx": 3, "valid": True},
                {"prev_idx": 1, "next_idx": 3, "valid": True},
                {"prev_idx": 3, "next_idx": 4, "valid": True},
            ],
        )

    def test_resolve_bridge_anchor_pair_prefers_planned_realized_anchor(self):
        graph = build_adjacent_anchor_graph([0, 2, 4])
        realized = {0: {"name": "r0"}, 2: {"name": "r2"}}
        estimated = {0: {"name": "e0"}, 2: {"name": "e2"}}

        pair = resolve_bridge_anchor_pair(
            graph,
            0,
            realized_anchor_by_idx=realized,
            estimated_anchor_by_idx=estimated,
            local_prev_idx=1,
            local_next_idx=2,
        )

        self.assertEqual(pair["source"], "planned")
        self.assertEqual(pair["prev_idx"], 0)
        self.assertEqual(pair["next_idx"], 2)
        self.assertEqual(pair["prev_anchor"], {"name": "r0"})
        self.assertEqual(pair["next_anchor"], {"name": "r2"})

    def test_resolve_bridge_anchor_pair_falls_back_to_local_when_planned_missing_anchor(self):
        graph = build_adjacent_anchor_graph([0, 2])
        estimated = {1: {"name": "e1"}, 2: {"name": "e2"}}

        pair = resolve_bridge_anchor_pair(
            graph,
            0,
            realized_anchor_by_idx={},
            estimated_anchor_by_idx=estimated,
            local_prev_idx=1,
            local_next_idx=2,
        )

        self.assertEqual(pair["source"], "local")
        self.assertEqual(pair["prev_idx"], 1)
        self.assertEqual(pair["next_idx"], 2)
        self.assertEqual(pair["prev_anchor"], {"name": "e1"})
        self.assertEqual(pair["next_anchor"], {"name": "e2"})


if __name__ == "__main__":
    unittest.main()
