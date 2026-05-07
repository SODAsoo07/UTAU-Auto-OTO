from core.generation.kr_pipeline_finish import finalize_kr_generation_pipeline


def _base_kwargs(**overrides):
    calls = overrides.pop("calls", [])
    kwargs = {
        "processed": 2,
        "total": 3,
        "errors": [],
        "gen_missing_vowels": True,
        "final_lines": [],
        "file_groups": {"a": []},
        "tg_entries": [],
        "single_vowel_span_by_tg_path": {},
        "norm_tg_path_key_fn": lambda value: value,
        "load_textgrid_fn": lambda _path: None,
        "coerce_interval_tier_fn": lambda tier: tier,
        "is_interval_like_tier_fn": lambda _tier: True,
        "compute_noninitial_vowel_timing_fn": lambda *_args, **_kwargs: None,
        "validate_fn": lambda *args: args,
        "append_alias_rows_fn": lambda *_args, **_kwargs: None,
        "generate_openutau": False,
        "alias_suffix": "",
        "log_fn": lambda msg: calls.append(("log", msg)),
        "placeholder_block_errors": [],
        "auto_gen_format": "cvvc",
        "runtime_report": {},
        "anchor_stats": {},
        "anchor_log_path": None,
        "anchor_log_dir": None,
        "cleanup_timing_jsonl": False,
        "log_unset_summary_fn": lambda: calls.append(("unset_summary", None)),
        "out_path": "oto.ini",
        "tg_folder": "tg",
        "kr_profile": {},
        "custom_map": {},
        "custom_phonemes_path": "",
        "enable_ml_correction": False,
        "ml_policy": {},
        "normalize_key_fn": lambda value: value,
        "wav_name_map": {},
        "apply_output_wav_name_map_fn": lambda *_args, **_kwargs: 0,
        "review_queue_export": True,
        "review_queue_path": "review.jsonl",
        "row_apply_decisions": [{"alias": "ka"}],
        "row_apply_mode_counts": {"full_apply": 1},
        "append_missing_vowels_fn": lambda **_kwargs: calls.append(("missing", None)),
        "finish_context_factory": lambda **kwargs: {"finish": kwargs},
        "maybe_block_output_fn": lambda **_kwargs: False,
        "persist_output_fn": lambda **_kwargs: calls.append(("persist", _kwargs["out_path"])),
        "export_review_queue_fn": lambda **_kwargs: calls.append(("review", _kwargs["review_queue_path"])),
        "update_review_queue_report_fn": lambda report, **_kwargs: report.update({"review_updated": True}),
        "finalize_finish_fn": lambda context: calls.append(("finalize", context)),
    }
    kwargs.update(overrides)
    return kwargs, calls


def test_finalize_kr_generation_pipeline_runs_success_flow():
    kwargs, calls = _base_kwargs()

    processed, total, errors = finalize_kr_generation_pipeline(**kwargs)

    assert (processed, total, errors) == (2, 3, [])
    assert [name for name, _value in calls] == [
        "log",
        "missing",
        "persist",
        "review",
        "finalize",
        "unset_summary",
    ]
    assert kwargs["runtime_report"]["review_updated"] is True


def test_finalize_kr_generation_pipeline_stops_when_output_blocked():
    kwargs, calls = _base_kwargs(
        gen_missing_vowels=False,
        maybe_block_output_fn=lambda **_kwargs: True,
    )

    processed, total, errors = finalize_kr_generation_pipeline(**kwargs)

    assert (processed, total, errors) == (2, 3, [])
    assert calls == []
