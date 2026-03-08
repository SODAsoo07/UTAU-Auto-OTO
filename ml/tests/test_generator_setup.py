import os
import sys
import tempfile
import unittest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_generator_setup import (
    build_ja_textgrid_preparation,
    build_ja_trace_preparation,
    prepare_ja_generation_setup,
)
from core.kr_generator_setup import prepare_kr_profile_setup


class GeneratorSetupTests(unittest.TestCase):
    def test_prepare_kr_profile_setup_falls_back_to_preset(self):
        logs = []
        old_env = os.environ.get("UTOA_KR_ANCHOR_PROFILE_PATH")
        try:
            result = prepare_kr_profile_setup(
                fallback_format="cvc",
                auto_gen_format="cvc",
                kr_anchor_profile_path="anchor_profile.json",
                log_fn=logs.append,
                default_cache_path_fn=lambda fmt: f"cache_{fmt}.json",
                load_profile_fn=lambda path: None,
                resolve_reference_dirs_fn=lambda: [],
                build_profile_fn=lambda ref_dirs: None,
                save_profile_fn=lambda path, profile: True,
                get_preset_fn=lambda fmt: {"buckets": {"cv": {}}, "preset": fmt},
            )
        finally:
            if old_env is None:
                os.environ.pop("UTOA_KR_ANCHOR_PROFILE_PATH", None)
            else:
                os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = old_env
        self.assertEqual(result.profile_path, "cache_cvc.json")
        self.assertEqual(result.profile["preset"], "cvc")
        self.assertTrue(any("외부 앵커 프로파일 사용" in line for line in logs))
        self.assertTrue(any("추상화 프리셋 로드" in line for line in logs))

    def test_build_ja_trace_preparation_uses_expected_filenames(self):
        prep = build_ja_trace_preparation("C:/tmp/project", now=datetime(2026, 3, 8, 12, 34, 56))
        self.assertTrue(prep.anchor_log_path.endswith("timing_anchor_ja_20260308_123456.jsonl"))
        self.assertTrue(prep.mapping_trace_path.endswith("ja_mapping_trace_20260308_123456.jsonl"))

    def test_build_ja_textgrid_preparation_resolves_normalized_entry(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "A.TextGrid"), "w", encoding="utf-8").close()
            prep = build_ja_textgrid_preparation(td, normalize_key_fn=lambda name: os.path.splitext(name)[0].lower())
            info = prep.resolve_tg_info("a.wav")
            self.assertIsNotNone(info)
            self.assertEqual(info["real_name"], "A.wav")

    def test_prepare_ja_generation_setup_disables_template_on_low_match(self):
        logs = []
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "a.TextGrid"), "w", encoding="utf-8").close()
            tpl_path = os.path.join(td, "base_oto.ini")
            open(tpl_path, "w", encoding="utf-8").close()
            result = prepare_ja_generation_setup(
                tg_folder=td,
                tpl_path=tpl_path,
                auto_gen_format="cvvc",
                custom_phonemes_path="",
                log_fn=logs.append,
                normalize_key_fn=lambda name: os.path.splitext(name)[0].lower(),
                load_template_lines_fn=lambda path: (["b.wav=ka,0,0,0,0,0"], "utf-8", "", ""),
            )
        self.assertFalse(result.use_template)
        self.assertEqual(result.file_groups, {})
        self.assertTrue(any("매칭률 낮음" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
