import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_cv_head_row_v2 import run_ja_cv_head_row


class JaCvHeadRowV2Tests(unittest.TestCase):
    def test_run_ja_cv_head_row_executes_finalize(self):
        lines = []
        run_ja_cv_head_row(
            curr_phones=[SimpleNamespace(mark="k", minTime=0.1, maxTime=0.15), SimpleNamespace(mark="a", minTime=0.16, maxTime=0.30)],
            current_w_idx=0,
            alias="- ka",
            format_type="cvvc",
            base_shape={},
            mel_ctx_for_file=None,
            post_ctx=SimpleNamespace(
                cv_head_min_cutoff=lambda *args, **kwargs: None,
                post_adjust=lambda offset, consonant, cutoff, pre, ovl, **kwargs: (offset, consonant, cutoff, pre, ovl),
                guard_cv_head_offset_to_onset=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 4.0),
                ensure_cv_head_min_vowel_coverage=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 5.0),
                guard_cv_cutoff_to_next_onset=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 6.0),
            ),
            real_wav_name="a.wav",
            final_lines=lines,
            generate_openutau=False,
            fname="a.wav",
            log_fn=lambda msg: None,
            validate_fn=lambda *args: args,
            extract_cv_bounds_fn=lambda phones, **kwargs: (100.0, 130.0, 140.0, 260.0),
            cv_offset_and_pre_fn=lambda *args, **kwargs: (10.0, 30.0),
            adaptive_overlap_fn=lambda *args, **kwargs: 20.0,
            soft_guard_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.5, 2.5),
            apply_base_shape_blend_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0),
            apply_anchor_lock_fn=lambda **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            target_vowel_from_alias_fn=lambda *args, **kwargs: "a",
            pick_vowel_phone_fn=lambda phones, target_vowel=None: (phones[-1], 1),
            build_anchor_record_fn=lambda idx, **kwargs: (idx, {"ok": True}),
            finalize_row_fn=lambda **kwargs: lines.extend(["row"]),
            row_builder_fn=lambda *args, **kwargs: [],
            generate_openutau_aliases_fn=lambda alias: [alias],
            alias_out_fn=lambda alias: alias,
            build_guard_messages_fn=lambda *args, **kwargs: ["m1"],
            anchor_store={},
        )
        self.assertEqual(lines, ["row"])


if __name__ == "__main__":
    unittest.main()
