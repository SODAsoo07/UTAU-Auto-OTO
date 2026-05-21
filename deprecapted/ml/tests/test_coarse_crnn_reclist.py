from __future__ import annotations

from core.coarse_crnn.deprecated.coarse_alignment.reclist import estimate_reclist_phone_bounds


def test_reclist_bounds_use_filename_when_phone_shape_matches():
    priors = estimate_reclist_phone_bounds(
        r"C:\vb\k_a.wav",
        ["k", "a"],
        language="japanese",
        active_range_sec=(0.10, 0.40),
    )

    assert priors is not None
    assert len(priors) == 2
    assert round(priors[0][0], 3) == 0.100
    assert round(priors[-1][1], 3) == 0.400


def test_reclist_bounds_skip_unrelated_filename():
    priors = estimate_reclist_phone_bounds(
        r"C:\vb\a_i_u_e_o.wav",
        ["k", "a"],
        language="japanese",
        active_range_sec=(0.10, 0.40),
    )

    assert priors is None
