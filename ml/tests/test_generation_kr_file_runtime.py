from core.generation.kr_file_runtime import (
    analyze_kr_file_alias_families,
    prepare_kr_row_iteration_context,
    resolve_kr_cv_timing_mode,
)


def _classify(alias: str) -> str:
    return {
        "- a": "vc",
        "ka": "cv",
        "a": "mono",
        "a i": "vv",
    }.get(alias, "")


def test_analyze_kr_file_alias_families_detects_vc_and_cv_family():
    result = analyze_kr_file_alias_families(
        ["x.wav=- a,0,0,0,0,0", "x.wav=ka,0,0,0,0,0", "broken"],
        classify_alias_fn=_classify,
    )

    assert result == {
        "file_has_explicit_vc_alias": True,
        "file_has_cv_family_alias": True,
    }


def test_resolve_kr_cv_timing_mode_uses_vcv_context_for_vcv_mix():
    result = resolve_kr_cv_timing_mode(
        file_format="vcv",
        file_has_explicit_vc_alias=True,
        file_has_cv_family_alias=False,
    )

    assert result["format_norm"] == "vcv"
    assert result["file_has_mixed_cv_vc"] is True
    assert result["kr_cv_timing_mode"] == "vcv_context"


def test_prepare_kr_row_iteration_context_builds_maps_and_anchors():
    logs = []
    syllables = [
        {"roman_cv": "ka", "roman": "k a"},
        {"roman": "a"},
    ]

    context = prepare_kr_row_iteration_context(
        lines=["x.wav=- a,0,0,0,0,0", "x.wav=ka,0,0,0,0,0"],
        file_format="cvvc",
        syllables_info=syllables,
        ph_intervals=["p0"],
        filename_cv_targets=["ka", "a"],
        disable_cvvc_order_lock=False,
        debug_logging=True,
        fname="x.wav",
        log_fn=logs.append,
        is_order_locked_format_fn=lambda fmt: fmt == "cvvc",
        build_occurrence_map_fn=lambda source: {"source": list(source)},
        build_vv_occurrence_map_fn=lambda source: {"vv_source": list(source)},
        classify_alias_fn=_classify,
        compute_vowel_baseline_fn=lambda _syllables: 88.0,
        estimate_cv_anchor_fn=lambda syllable, ph_intervals, cv_mode: (
            syllable.get("roman_cv") or syllable.get("roman"),
            tuple(ph_intervals),
            cv_mode,
        ),
    )

    assert context["romaji_syllables"] == ["ka", "a"]
    assert context["kr_order_locked_format"] is True
    assert context["kr_cvvc_occurrence_map"] == {"source": ["ka", "a"]}
    assert context["kr_cvvc_vv_occurrence_map"] == {"vv_source": ["ka", "a"]}
    assert context["kr_cv_timing_mode"] == "vc_context"
    assert context["kr_cv_v_ref_baseline"] == 88.0
    assert syllables[0]["v_ref_baseline_ms"] == 88.0
    assert context["cv_anchor_by_idx"][0] == ("ka", ("p0",), "vc_context")
    assert logs and "KR CV timing mode=vc_context" in logs[0]


def test_prepare_kr_row_iteration_context_can_disable_order_lock():
    logs = []
    context = prepare_kr_row_iteration_context(
        lines=["x.wav=ka,0,0,0,0,0"],
        file_format="cvvc",
        syllables_info=[{"roman_cv": "ka"}],
        ph_intervals=[],
        filename_cv_targets=["ka"],
        disable_cvvc_order_lock=True,
        debug_logging=True,
        fname="x.wav",
        log_fn=logs.append,
        is_order_locked_format_fn=lambda _fmt: True,
        build_occurrence_map_fn=lambda _source: {"unexpected": True},
        build_vv_occurrence_map_fn=lambda source: {"vv_source": list(source)},
        classify_alias_fn=_classify,
        compute_vowel_baseline_fn=lambda _syllables: 70.0,
        estimate_cv_anchor_fn=lambda *_args, **_kwargs: "anchor",
    )

    assert context["kr_order_locked_format"] is False
    assert context["kr_cvvc_occurrence_map"] is None
    assert logs[0].endswith("UTOA_KR_DISABLE_CVVC_ORDER_LOCK=1)")
