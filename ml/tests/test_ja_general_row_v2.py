import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_general_row_v2 import run_ja_general_row


class JaGeneralRowV2Tests(unittest.TestCase):
    def test_run_ja_general_row_executes_finalize(self):
        lines = []
        run_ja_general_row(
            final_lines=lines,
            real_wav_name="a.wav",
            alias="ka",
            alias_type="cv",
            format_type="cvvc",
            offset=10.0,
            consonant=50.0,
            cutoff=-90.0,
            pre=30.0,
            ovl=20.0,
            c_end=120.0,
            n_start=130.0,
            n_end=240.0,
            c_start=100.0,
            current_w_idx=0,
            generate_openutau=False,
            fname="a.wav",
            log_fn=lambda msg: None,
            base_shape={},
            mel_ctx_for_file=None,
            onset_hint_local="k",
            mapping_tier="mid",
            c_char="k",
            vc_prev_anchor=None,
            vc_next_anchor=None,
            post_ctx=SimpleNamespace(
                post_adjust=lambda offset, consonant, cutoff, pre, ovl, **kwargs: (offset, consonant, cutoff, pre, ovl),
                guard_cv_cutoff_to_next_onset=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 6.0),
            ),
            validate_fn=lambda *args: args,
            soft_guard_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.0, 2.0),
            base_shape_blend_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0),
            refine_ja_vc_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0),
            onset_class_fn=lambda c: "plain",
            is_n_bridge_fn=lambda alias, mode: False,
            limit_pre_anchor_shift_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 30.0, False),
            apply_anchor_lock_fn=lambda **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            anchor_lock_enabled_fn=lambda language, fmt: True,
            enforce_cv_pre_anchor_guard_fn=lambda *args, **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
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
