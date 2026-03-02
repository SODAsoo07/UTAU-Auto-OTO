import json
import tempfile
import unittest
from pathlib import Path

from Auto_OTO.core.log_events import append_structured_log, classify_log_message


class LogEventsTests(unittest.TestCase):
    def test_prepare_is_visible_progress(self):
        record = classify_log_message("[Prepare] (1/10) korean/cvvc sample 처리 시작")
        self.assertTrue(record["ui_visible"])
        self.assertEqual("prepare", record["category"])

    def test_ml_model_load_hidden(self):
        record = classify_log_message("[OTO-ML] 모델 로드 (pytorch): C:\\tmp\\model")
        self.assertFalse(record["ui_visible"])
        self.assertEqual("ml_model_load", record["category"])

    def test_error_visible(self):
        record = classify_log_message("❌ MFA 실패: test")
        self.assertTrue(record["ui_visible"])
        self.assertEqual("error", record["category"])

    def test_append_structured_log_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            append_structured_log(
                str(path),
                {
                    "message": "✅ 완료",
                    "severity": "success",
                    "category": "success",
                    "ui_visible": True,
                },
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            payload = json.loads(lines[0])
            self.assertEqual("✅ 완료", payload["message"])
            self.assertEqual("success", payload["severity"])


if __name__ == "__main__":
    unittest.main()
