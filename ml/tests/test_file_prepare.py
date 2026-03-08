import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.file_prepare import load_named_tiers, prepare_file_context


class FilePrepareTests(unittest.TestCase):
    def test_prepare_file_context_returns_missing_status(self):
        ctx = prepare_file_context(
            fname="a.wav",
            lines=[],
            resolve_tg_info_fn=lambda fname: None,
            missing_diagnostics_fn=lambda fname: {"lookup_norm": "a", "norm_candidates": []},
            wav_root_for_signal=".",
            wav_index_for_signal={},
            mel_cache_for_signal={},
            find_wav_path_fn=lambda *args: "",
            read_wav_fn=lambda path: ([], 0),
            mel_envelope_fn=lambda audio, sr: None,
        )
        self.assertEqual(ctx.status, "textgrid_missing")
        self.assertEqual(ctx.diagnostics["lookup_norm"], "a")

    def test_prepare_file_context_reads_audio_once_and_caches_mel(self):
        calls = []
        cache = {}
        ctx = prepare_file_context(
            fname="a.wav",
            lines=[],
            resolve_tg_info_fn=lambda fname: {"path": "a.TextGrid", "real_name": "a.wav"},
            wav_root_for_signal=".",
            wav_index_for_signal={"a": "a.wav"},
            mel_cache_for_signal=cache,
            find_wav_path_fn=lambda *args: "a.wav",
            read_wav_fn=lambda path: ([0, 1, 2, 3], 2),
            mel_envelope_fn=lambda audio, sr: calls.append((list(audio), sr)) or {"ok": True},
            wav_duration_fn=lambda path: 999.0,
        )
        self.assertEqual(ctx.status, "ok")
        self.assertAlmostEqual(ctx.wav_duration_ms, 2000.0)
        self.assertEqual(len(calls), 1)
        self.assertIn("a.wav", cache)

        ctx2 = prepare_file_context(
            fname="a.wav",
            lines=[],
            resolve_tg_info_fn=lambda fname: {"path": "a.TextGrid", "real_name": "a.wav"},
            wav_root_for_signal=".",
            wav_index_for_signal={"a": "a.wav"},
            mel_cache_for_signal=cache,
            find_wav_path_fn=lambda *args: "a.wav",
            read_wav_fn=lambda path: (_ for _ in ()).throw(RuntimeError("should not read again")),
            mel_envelope_fn=lambda audio, sr: {"bad": True},
            wav_duration_fn=lambda path: 321.0,
        )
        self.assertEqual(ctx2.mel_ctx_for_file, {"ok": True})
        self.assertAlmostEqual(ctx2.wav_duration_ms, 321.0)

    def test_load_named_tiers_sets_tier_missing_status(self):
        ctx = prepare_file_context(
            fname="a.wav",
            lines=[],
            resolve_tg_info_fn=lambda fname: {"path": "a.TextGrid", "real_name": "a.wav"},
            wav_root_for_signal=".",
            wav_index_for_signal={},
            mel_cache_for_signal={},
            find_wav_path_fn=lambda *args: "",
            read_wav_fn=lambda path: ([], 0),
            mel_envelope_fn=lambda audio, sr: None,
        )
        grid = [SimpleNamespace(name="words")]
        ctx = load_named_tiers(ctx, load_textgrid_fn=lambda path: grid, tier_predicate=lambda tier: True)
        self.assertEqual(ctx.status, "tier_missing")
        self.assertIsNotNone(ctx.word_tier)


if __name__ == "__main__":
    unittest.main()
