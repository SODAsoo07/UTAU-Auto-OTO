from types import SimpleNamespace

from core.generation.kr_row_context import prepare_kr_general_row_bounds


def test_prepare_kr_general_row_bounds_uses_cv_bounds_and_advances_current():
    phones = [SimpleNamespace(mark="k")]

    out = prepare_kr_general_row_bounds(
        is_vc=False,
        current_w_idx=1,
        selected_w_idx=3,
        bridge_pair={"next_idx": 4},
        syllables_info=[{}, {}, {}, {}],
        prepare_cv_bounds_fn=lambda _syllables, idx: (idx, phones, 10, 20, 30, 40),
        prepare_vc_bounds_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert out == {
        "current_w_idx": 3,
        "selected_w_idx": 3,
        "bridge_next_idx": None,
        "curr_phones": phones,
        "c_start": 10.0,
        "c_end": 20.0,
        "n_start": 30.0,
        "n_end": 40.0,
    }


def test_prepare_kr_general_row_bounds_uses_bridge_pair_for_vc_context():
    calls = []
    phones = [SimpleNamespace(mark="ng")]

    def _prepare_vc(_syllables, current_idx, *, next_w_idx=None):
        calls.append((current_idx, next_w_idx))
        return current_idx, phones, 100, 160, 180, 260

    out = prepare_kr_general_row_bounds(
        is_vc=True,
        current_w_idx=2,
        selected_w_idx=99,
        bridge_pair={"prev_idx": 1, "next_idx": 3},
        syllables_info=[{}, {}, {}, {}],
        prepare_cv_bounds_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
        prepare_vc_bounds_fn=_prepare_vc,
    )

    assert calls == [(1, 3)]
    assert out["current_w_idx"] == 1
    assert out["selected_w_idx"] == 2
    assert out["bridge_next_idx"] == 3
    assert out["curr_phones"] == phones
    assert out["c_start"] == 100.0
    assert out["n_end"] == 260.0
