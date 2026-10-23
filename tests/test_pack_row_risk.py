from __future__ import annotations

import math

import numpy as np

from universal_auto_oto import pack_row_risk
from universal_auto_oto.oto_preview import RowStatus


def test_review_ranks_send_the_riskiest_share_to_review():
    risks = [0.01 * k for k in range(20)]
    ranks = pack_row_risk.review_ranks(risks)
    assert ranks.count("inspect_first") == 2
    assert ranks.count("review") == 4
    assert ranks[19] == ranks[18] == "inspect_first"
    assert all(rank == "clean" for rank in ranks[:14])


def test_predict_walks_binned_trees():
    profile = {
        "features": ["a", "b"],
        "edges": [[0.5], [1.0, 2.0]],
        "base": 0.0,
        "learning_rate": 1.0,
        "trees": [["split", 0, 0, ["leaf", -1.0], ["split", 1, 1, ["leaf", 0.5], ["leaf", 2.0]]]],
    }
    risks = pack_row_risk.predict(profile, [{"a": 0.0, "b": 9.0}, {"a": 1.0, "b": 1.5}, {"a": 1.0, "b": 3.0}])
    assert np.allclose(risks, [1 / (1 + math.e), 1 / (1 + math.exp(-0.5)), 1 / (1 + math.exp(-2.0))])


def test_risk_hints_name_the_cues():
    hints = pack_row_risk.risk_hints({"tempo_z": 4.0, "dur_absmax": 1.2, "ac_span_quiet": 0.1, "fallback": 1.0})
    assert hints == ["pack_risk_off_tempo", "pack_risk_phone_length", "pack_risk_landmark_fallback"]


def test_english_profile_is_packaged():
    profile = pack_row_risk.load_profile("en_cz_vccv")
    assert profile is not None and profile["trees"]
    assert len(profile["edges"]) == len(profile["features"])


def test_preview_draws_first_ranked_pack_rows_red():
    first = RowStatus("review", 0.3, 0.7, ("pack_row_risk_first", "pack_risk_off_tempo"))
    later = RowStatus("review", 0.3, 0.7, ("pack_row_risk_high",))
    assert first.color_key == "high_risk"
    assert later.color_key == "review"


def test_template_like_reference_flags_unadjusted_templates(tmp_path):
    from universal_auto_oto.style_fitting import template_like_reference

    template = tmp_path / "template"
    timed = tmp_path / "timed"
    template.mkdir()
    timed.mkdir()
    (template / "oto.ini").write_text("".join(f"f{k}.wav=a{k},{1000 + 300 * (k % 3)},120,-300,150,60\n" for k in range(60)),
                                      encoding="utf-8")
    (timed / "oto.ini").write_text("".join(f"f{k}.wav=a{k},{903.7 + 4.13 * k:.2f},120,-300,150,60\n" for k in range(60)),
                                   encoding="utf-8")
    flagged = template_like_reference([template])
    assert flagged is not None and flagged["distinct_offsets"] == 3 and flagged["rows"] == 60
    assert template_like_reference([timed]) is None
