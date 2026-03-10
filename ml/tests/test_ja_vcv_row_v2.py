import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_vcv_row_v2 import run_ja_vcv_row


class JaVcvRowV2Tests(unittest.TestCase):
    def test_run_ja_vcv_row_executes_finalize_and_updates_indices(self):
        lines = []
        logs = []
        result = run_ja_vcv_row(
            syllables_info=[
                {"phones": [SimpleNamespace(minTime=0.10, maxTime=0.14), SimpleNamespace(minTime=0.15, maxTime=0.28)]}
            ],
            current_w_idx=0,
            stable_vcv_seq_idx=0,
            cv_seq_idx=0,
            mapped_idx=0,
            alias="a ka",
            base_shape={},
            mel_ctx_for_file=None,
            onset_hint_local="k",
            post_ctx=SimpleNamespace(
                file_format="vcv",
                post_adjust=lambda offset, consonant, cutoff, pre, ovl, **kwargs: (offset, consonant, cutoff, pre, ovl),
            ),
            real_wav_name="a.wav",
            final_lines=lines,
            generate_openutau=False,
            fname="a.wav",
            log_fn=logs.append,
            prepare_cv_bounds_fn=lambda phones, **kwargs: (100.0, 140.0, 150.0, 280.0),
            compute_vcv_params_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0),
            soft_guard_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.5, 2.5),
            enforce_vcv_cv_entry_guard_fn=lambda *args, **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            extract_target_syllable_fn=lambda alias: alias.split()[-1],
            split_syllable_fn=lambda tok: (tok[:-1], tok[-1:]),
            is_n_bridge_fn=lambda alias, mode: False,
            apply_anchor_lock_fn=lambda **kwargs: (12.0, 52.0, -92.0, 32.0, 22.0),
            finalize_row_fn=lambda **kwargs: lines.extend(["row"]),
            row_builder_fn=lambda *args, **kwargs: [],
            generate_openutau_aliases_fn=lambda alias: [alias],
            alias_out_fn=lambda alias: alias,
        )
        self.assertEqual(result, (0, 0, 0, 0))
        self.assertEqual(lines, ["row"])
        self.assertEqual(len(logs), 1)

    def test_run_ja_vcv_row_handles_missing_file_format_attr(self):
        lines = []
        result = run_ja_vcv_row(
            syllables_info=[
                {"phones": [SimpleNamespace(minTime=0.10, maxTime=0.14), SimpleNamespace(minTime=0.15, maxTime=0.28)]}
            ],
            current_w_idx=0,
            stable_vcv_seq_idx=0,
            cv_seq_idx=0,
            mapped_idx=0,
            alias="a ka",
            base_shape={},
            mel_ctx_for_file=None,
            onset_hint_local="k",
            post_ctx=SimpleNamespace(
                post_adjust=lambda offset, consonant, cutoff, pre, ovl, **kwargs: (offset, consonant, cutoff, pre, ovl),
            ),
            real_wav_name="a.wav",
            final_lines=lines,
            generate_openutau=False,
            fname="a.wav",
            log_fn=lambda msg: None,
            prepare_cv_bounds_fn=lambda phones, **kwargs: (100.0, 140.0, 150.0, 280.0),
            compute_vcv_params_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0),
            soft_guard_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.5, 2.5),
            enforce_vcv_cv_entry_guard_fn=lambda *args, **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            extract_target_syllable_fn=lambda alias: alias.split()[-1],
            split_syllable_fn=lambda tok: (tok[:-1], tok[-1:]),
            is_n_bridge_fn=lambda alias, mode: False,
            apply_anchor_lock_fn=lambda **kwargs: (12.0, 52.0, -92.0, 32.0, 22.0),
            finalize_row_fn=lambda **kwargs: lines.extend(["row"]),
            row_builder_fn=lambda *args, **kwargs: [],
            generate_openutau_aliases_fn=lambda alias: [alias],
            alias_out_fn=lambda alias: alias,
        )
        self.assertEqual(result, (0, 1, 0, 0))
        self.assertEqual(lines, ["row"])


if __name__ == "__main__":
    unittest.main()
