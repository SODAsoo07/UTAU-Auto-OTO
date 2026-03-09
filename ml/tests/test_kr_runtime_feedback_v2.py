import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_runtime_feedback_v2 import (
    build_kr_bridge_adjust_message,
    build_kr_cv_head_guard_messages,
)


class KrRuntimeFeedbackV2Tests(unittest.TestCase):
    def test_build_kr_cv_head_guard_messages_emits_expected_messages(self):
        messages = build_kr_cv_head_guard_messages(
            "a.wav",
            "ga",
            offset_reduced=2.5,
            cutoff_extended=3.5,
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("CV_HEAD 오프셋 과선행 보정", messages[0])
        self.assertIn("CV_HEAD 모음 길이 보정", messages[1])

    def test_build_kr_cv_head_guard_messages_skips_small_adjustments(self):
        messages = build_kr_cv_head_guard_messages(
            "a.wav",
            "ga",
            offset_reduced=0.8,
            cutoff_extended=0.9,
        )
        self.assertEqual(messages, [])

    def test_build_kr_bridge_adjust_message_only_for_vc_vv_above_threshold(self):
        self.assertIsNone(build_kr_bridge_adjust_message("a.wav", "ga", "cv", 12.0))
        self.assertIsNone(build_kr_bridge_adjust_message("a.wav", "a g", "vc", 5.0))
        message = build_kr_bridge_adjust_message("a.wav", "a g", "vc", 8.0)
        self.assertIsNotNone(message)
        self.assertIn("KR 브리지 앵커 미세보정", message)


if __name__ == "__main__":
    unittest.main()
