import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_runtime_policy import resolve_runtime_mapping_policy


class OtoRuntimePolicyTests(unittest.TestCase):
    def test_runtime_policy_forces_sequence_lock_on_low_tier_plan(self):
        ingest = SimpleNamespace(
            textgrid_trust_score=0.62,
            textgrid_trust_tier="mid",
            prefer_filename_sequence=False,
        )
        policy = resolve_runtime_mapping_policy(
            ingest_snapshot=ingest,
            plan_policy={"tier": "low", "confidence_cap": 0.8, "fallback_mode": "sequence_lock", "allow_generation": True},
            mapping_confidence=0.74,
            conf_threshold=0.60,
            format_type="cvvc",
            score_a=60.0,
            score_b=61.0,
            sequence_lock_formats={"cvvc", "cv"},
            abstain_formats={"cvvc", "vcv", "cv"},
        )
        self.assertEqual(policy["mapping_tier"], "low")
        self.assertTrue(policy["force_sequence_lock"])
        self.assertTrue(policy["is_low_conf"])

    def test_runtime_policy_abstains_when_plan_disallows_generation(self):
        ingest = SimpleNamespace(
            textgrid_trust_score=0.20,
            textgrid_trust_tier="low",
            prefer_filename_sequence=True,
        )
        policy = resolve_runtime_mapping_policy(
            ingest_snapshot=ingest,
            plan_policy={"tier": "low", "confidence_cap": 0.24, "allow_generation": False},
            mapping_confidence=0.50,
            conf_threshold=0.58,
            format_type="vcv",
            score_a=30.0,
            score_b=32.0,
            sequence_lock_formats={"cvvc", "cvc"},
            abstain_formats={"cvvc", "vcv", "cvc", "cv"},
        )
        self.assertTrue(policy["should_abstain"])
        self.assertLessEqual(policy["mapping_confidence"], 0.24)

    def test_runtime_policy_keeps_high_tier_when_trust_and_plan_are_good(self):
        ingest = SimpleNamespace(
            textgrid_trust_score=0.92,
            textgrid_trust_tier="high",
            prefer_filename_sequence=False,
        )
        policy = resolve_runtime_mapping_policy(
            ingest_snapshot=ingest,
            plan_policy={"tier": "high", "confidence_cap": 0.95, "allow_generation": True},
            mapping_confidence=0.88,
            conf_threshold=0.70,
            format_type="cv",
            score_a=80.0,
            score_b=78.0,
            sequence_lock_formats={"cvvc", "cv"},
            abstain_formats={"cvvc", "vcv", "cv"},
        )
        self.assertEqual(policy["mapping_tier"], "high")
        self.assertFalse(policy["force_sequence_lock"])
        self.assertFalse(policy["should_abstain"])
        self.assertFalse(policy["is_low_conf"])

    def test_runtime_policy_marks_low_margin_as_low_conf(self):
        ingest = SimpleNamespace(
            textgrid_trust_score=0.86,
            textgrid_trust_tier="high",
            prefer_filename_sequence=False,
        )
        policy = resolve_runtime_mapping_policy(
            ingest_snapshot=ingest,
            plan_policy={"tier": "high", "confidence_cap": 0.95, "allow_generation": True, "margin": 3.0},
            mapping_confidence=0.89,
            mapping_margin=3.0,
            conf_threshold=0.70,
            low_margin_floor=6.0,
            format_type="cvvc",
            score_a=81.0,
            score_b=78.0,
            sequence_lock_formats={"cvvc", "cv"},
            abstain_formats={"cvvc", "vcv", "cv"},
        )
        self.assertTrue(policy["low_margin"])
        self.assertTrue(policy["is_low_conf"])
        self.assertEqual(policy["mapping_tier"], "mid")

    def test_runtime_policy_strict_mode_increases_row_floors(self):
        ingest = SimpleNamespace(
            textgrid_trust_score=0.70,
            textgrid_trust_tier="mid",
            prefer_filename_sequence=False,
        )
        policy = resolve_runtime_mapping_policy(
            ingest_snapshot=ingest,
            plan_policy={"tier": "mid", "confidence_cap": 0.8, "allow_generation": True},
            mapping_confidence=0.66,
            conf_threshold=0.58,
            format_type="cvvc",
            score_a=62.0,
            score_b=61.0,
            sequence_lock_formats={"cvvc", "cv"},
            abstain_formats={"cvvc", "vcv", "cv"},
            strict_formats={"cvvc"},
        )
        self.assertTrue(policy["strict_mode"])
        self.assertGreater(policy["row_conf_floor"], policy["file_conf_floor"])
        self.assertGreaterEqual(policy["row_margin_floor"], 8.0)


if __name__ == "__main__":
    unittest.main()
