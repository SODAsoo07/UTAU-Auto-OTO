from types import SimpleNamespace

from core.generation.kr_missing_vowels import (
    append_kr_missing_single_vowel_aliases,
    collect_template_aliases,
    detect_missing_single_vowel_alias,
    resolve_single_vowel_span,
)


def test_detect_missing_single_vowel_alias_prefers_filename_suffix():
    assert detect_missing_single_vowel_alias({"real_name": "sample-a-long.TextGrid", "norm_key": ""}) == "a"
    assert detect_missing_single_vowel_alias({"real_name": "sample.TextGrid", "norm_key": "eo"}) == "eo"
    assert detect_missing_single_vowel_alias({"real_name": "sample.TextGrid", "norm_key": "bad"}) == ""


def test_collect_template_aliases_reads_oto_alias_field():
    aliases = collect_template_aliases(
        {
            "a.wav": ["a.wav=a,1,2,3,4,5", "broken"],
            "b.wav": ["b.wav= EO ,1,2,3,4,5"],
        }
    )

    assert aliases == {"a", "eo"}


def test_resolve_single_vowel_span_uses_cache_before_loading():
    loaded = False

    def load_textgrid(_path):
        nonlocal loaded
        loaded = True
        return []

    span = resolve_single_vowel_span(
        {"path": "tg1"},
        single_vowel_span_by_tg_path={"tg1": (12.0, 80.0)},
        norm_tg_path_key_fn=lambda value: value,
        load_textgrid_fn=load_textgrid,
        coerce_interval_tier_fn=lambda tier: tier,
        is_interval_like_tier_fn=lambda _tier: True,
    )

    assert span == (12.0, 80.0)
    assert loaded is False


def test_resolve_single_vowel_span_loads_single_non_sil_phone():
    class FakeTier(list):
        pass

    tier = FakeTier([
        SimpleNamespace(mark="sil", minTime=0.0, maxTime=0.1),
        SimpleNamespace(mark="a", minTime=0.1, maxTime=0.5),
    ])
    setattr(tier, "name", "phones")

    span = resolve_single_vowel_span(
        {"path": "tg1"},
        single_vowel_span_by_tg_path={},
        norm_tg_path_key_fn=lambda value: value,
        load_textgrid_fn=lambda _path: [tier],
        coerce_interval_tier_fn=lambda loaded_tier: loaded_tier,
        is_interval_like_tier_fn=lambda _tier: True,
    )

    assert span == (100.0, 500.0)


def test_append_kr_missing_single_vowel_aliases_appends_validated_row():
    final_lines = []
    appended = []
    logs = []

    def append_rows(final_lines_arg, wav_name, alias, offset, consonant, cutoff, pre, ovl, **kwargs):
        appended.append((wav_name, alias, offset, consonant, cutoff, pre, ovl, kwargs))
        final_lines_arg.append(f"{wav_name}={alias},{offset},{consonant},{cutoff},{pre},{ovl}")

    added = append_kr_missing_single_vowel_aliases(
        final_lines=final_lines,
        file_groups={"base.wav": ["base.wav=ka,1,2,3,4,5"]},
        tg_entries=[
            {
                "real_name": "take-a-long.TextGrid",
                "output_name": "take-a.wav",
                "norm_key": "",
                "path": "tg1",
            }
        ],
        single_vowel_span_by_tg_path={"tg1": (20.0, 120.0)},
        norm_tg_path_key_fn=lambda value: value,
        load_textgrid_fn=lambda _path: [],
        coerce_interval_tier_fn=lambda tier: tier,
        is_interval_like_tier_fn=lambda _tier: True,
        compute_noninitial_vowel_timing_fn=lambda start, end: (start, end - start, -end, start + 1, 5.0),
        validate_fn=lambda offset, consonant, cutoff, pre, ovl: (
            offset + 1,
            consonant + 1,
            cutoff - 1,
            pre + 1,
            ovl + 1,
        ),
        append_alias_rows_fn=append_rows,
        generate_openutau=True,
        alias_suffix="_C4",
        log_added_fn=lambda tg_info, alias: logs.append((tg_info["real_name"], alias)),
    )

    assert added == 1
    assert final_lines == ["take-a.wav=a,21.0,101.0,-121.0,22.0,6.0"]
    assert appended[0][7]["generate_openutau"] is True
    assert appended[0][7]["alias_suffix"] == "_C4"
    assert appended[0][7]["alias_type"] == "mono"
    assert logs == [("take-a-long.TextGrid", "a")]


def test_append_kr_missing_single_vowel_aliases_skips_existing_template_alias():
    final_lines = []
    added = append_kr_missing_single_vowel_aliases(
        final_lines=final_lines,
        file_groups={"base.wav": ["base.wav=a,1,2,3,4,5"]},
        tg_entries=[{"real_name": "take-a.TextGrid", "norm_key": "", "path": "tg1"}],
        single_vowel_span_by_tg_path={"tg1": (20.0, 120.0)},
        norm_tg_path_key_fn=lambda value: value,
        load_textgrid_fn=lambda _path: [],
        coerce_interval_tier_fn=lambda tier: tier,
        is_interval_like_tier_fn=lambda _tier: True,
        compute_noninitial_vowel_timing_fn=lambda _start, _end: (0, 0, 0, 0, 0),
        validate_fn=lambda *values: values,
        append_alias_rows_fn=lambda *_args, **_kwargs: final_lines.append("unexpected"),
        generate_openutau=False,
        alias_suffix="",
        log_added_fn=lambda _tg_info, _alias: final_lines.append("unexpected-log"),
    )

    assert added == 0
    assert final_lines == []
