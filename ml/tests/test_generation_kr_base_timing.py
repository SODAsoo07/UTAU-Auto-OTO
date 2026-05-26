from types import SimpleNamespace

from core.generation.kr_base_timing import compute_kr_general_base_timing


def test_compute_kr_general_base_timing_dispatches_vc_context():
    calls = []

    def compute_vc(*args, **kwargs):
        calls.append((args, kwargs))
        return 1, 2, -3, 4, 5, True

    out = compute_kr_general_base_timing(
        is_vc=True,
        alias="k a",
        alias_type="vc",
        file_format="cvvc",
        curr_phones=[SimpleNamespace(mark="k")],
        current_w_idx=2,
        selected_w_idx=2,
        bridge_next_idx=3,
        bridge_pair={"prev_anchor": {"a": 1}, "next_anchor": {"b": 2}},
        syllables_info=[],
        c_start=10,
        c_end=20,
        n_start=20,
        n_end=90,
        cv_anchor_by_idx={},
        is_diph=False,
        kr_cv_v_ref_baseline=70,
        kr_cv_timing_mode="vc_context",
        mel_ctx_for_file=None,
        fname="a.wav",
        compute_vc_timing_fn=compute_vc,
    )

    assert out == {
        "offset": 1.0,
        "consonant": 2.0,
        "cutoff": -3.0,
        "pre": 4.0,
        "ovl": 5.0,
        "n_start": 20.0,
        "n_end": 90.0,
        "cv_vowel_len": 70.0,
        "is_vc_plosive_coda": True,
    }
    assert calls[0][1] == {
        "next_w_idx": 3,
        "prev_cv_anchor": {"a": 1},
        "next_cv_anchor": {"b": 2},
    }


def test_compute_kr_general_base_timing_dispatches_diphthong_cv():
    captured = {}

    def compute_cv(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 10, 20, -30, 40, 50

    out = compute_kr_general_base_timing(
        is_vc=False,
        alias="kwa",
        alias_type="cv",
        file_format="cvvc",
        curr_phones=[SimpleNamespace(mark="x"), SimpleNamespace(mark="g")],
        current_w_idx=0,
        selected_w_idx=0,
        bridge_next_idx=None,
        bridge_pair={},
        syllables_info=[],
        c_start=10,
        c_end=30,
        n_start=30,
        n_end=120,
        cv_anchor_by_idx={},
        is_diph=True,
        kr_cv_v_ref_baseline=90,
        kr_cv_timing_mode="vc_context",
        mel_ctx_for_file=None,
        fname="a.wav",
        compute_cv_timing_fn=compute_cv,
        select_cv_onset_slice_fn=lambda _phones: (1, 0, 0, 0, 0, 7.5),
        extract_alias_onset_fn=lambda _alias: "kw",
    )

    assert out["offset"] == 10.0
    assert captured["args"] == (10, 30, 90.0, "g", "kw", True, False)
    assert captured["kwargs"] == {
        "glide_dur_ms": 7.5,
        "v_ref_baseline": 90,
        "cv_mode": "vc_context",
    }


def test_compute_kr_general_base_timing_refines_vv_vowel_span_before_direct_timing():
    logs = []
    captured = {}

    def direct_vv(selected_w_idx, syllables_info, n_start, n_end):
        captured["direct"] = (selected_w_idx, syllables_info, n_start, n_end)
        return 11, 22, -33, 44, 55

    out = compute_kr_general_base_timing(
        is_vc=False,
        alias="aeo",
        alias_type="cv",
        file_format="cvvc",
        curr_phones=[SimpleNamespace(mark="")],
        current_w_idx=0,
        selected_w_idx=0,
        bridge_next_idx=None,
        bridge_pair={},
        syllables_info=[{"phones": []}],
        c_start=0,
        c_end=20,
        n_start=30,
        n_end=120,
        cv_anchor_by_idx={},
        is_diph=False,
        kr_cv_v_ref_baseline=90,
        kr_cv_timing_mode="standalone",
        mel_ctx_for_file={"mel": True},
        fname="a.wav",
        estimate_mel_vowel_activity_span_fn=lambda *_args, **_kwargs: (42.0, 98.0),
        log_vv_refine_fn=lambda *args: logs.append(args),
        select_cv_onset_slice_fn=lambda _phones: (0, 0, 0, 0, 0, 0),
        looks_like_vv_alias_fn=lambda _alias: True,
        compute_cvvc_vv_timing_direct_fn=direct_vv,
    )

    assert out["offset"] == 11.0
    assert out["n_start"] == 42.0
    assert out["n_end"] == 98.0
    assert captured["direct"] == (0, [{"phones": []}], 42.0, 98.0)
    assert logs == [("a.wav", "aeo", 30.0, 120.0, 42.0, 98.0)]
