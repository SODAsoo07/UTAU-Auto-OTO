from core.generation.kr_row_selection import prepare_kr_general_cv_selection_context


def _noop(*_args, **_kwargs):
    return None


def _base_kwargs(**overrides):
    kwargs = {
        "alias": "ka",
        "alias_type": "cv",
        "fname": "a.wav",
        "file_format": "cvvc",
        "target_clean": "ka",
        "current_w_idx": 0,
        "cv_seq_idx": 1,
        "bridge_pair": {},
        "row_mapping_confidence": 0.75,
        "row_jump_default": 1,
        "row_jump_high_conf": 2,
        "file_mapping_conf_th": 0.62,
        "file_mapping_low_conf": False,
        "romaji_syllables": ["ma", "ta", "ka"],
        "kr_planned_cv_indices": {},
        "kr_cvvc_occurrence_map": {},
        "kr_cvvc_occurrence_state": {},
        "kr_cvvc_vv_occurrence_map": {},
        "kr_cvvc_vv_occurrence_state": {},
        "syllable_blank_confidences": [0.1, 0.2, 0.3],
        "syllables_info": [
            {"active": True},
            {"active": True},
            {"active": True},
        ],
        "mapping_margin": 0.8,
        "row_margin_floor": 0.25,
        "row_conf_floor": 0.62,
        "row_blank_floor": 0.64,
        "debug_logging": True,
        "log_fn": lambda _msg: None,
        "resolve_vv_index_fn": lambda *_args: None,
        "resolve_cvvc_occurrence_index_fn": lambda *_args, **_kwargs: None,
        "resolve_planned_cv_index_fn": lambda *_args, **_kwargs: None,
        "select_general_cv_index_fn": lambda **_kw: {
            "expected_cv_idx": 1,
            "selected_w_idx": 1,
            "cv_seq_idx": 2,
            "row_mapping_confidence": 0.75,
            "resolve_meta": {},
            "row_jump_blocked": 0,
            "forced_selected_idx": None,
        },
        "remap_forced_cv_index_fn": _noop,
        "cv_match_score_fn": _noop,
        "split_syllable_parts_fn": _noop,
        "apply_row_confidence_penalty_fn": _noop,
        "resolve_cv_syllable_index_fn": _noop,
        "should_allow_exact_vowel_fix_fn": _noop,
        "find_cv_vowel_match_index_fn": _noop,
        "clamp_cv_index_to_order_fn": _noop,
        "blank_conf_at_fn": lambda values, idx: values[idx],
        "is_cv_syllable_active_fn": lambda row, **_kw: bool(row.get("active")),
        "decide_cv_row_abstain_fn": lambda **_kw: {"should_skip": False, "reason": ""},
    }
    kwargs.update(overrides)
    return kwargs


def test_prepare_kr_general_cv_selection_logs_planned_cv_remap():
    logs = []
    select_calls = []
    abstain_calls = []

    def select_general(**kwargs):
        select_calls.append(kwargs)
        return {
            "expected_cv_idx": 1,
            "selected_w_idx": 2,
            "cv_seq_idx": 3,
            "row_mapping_confidence": 0.71,
            "resolve_meta": {"planned": 1},
            "row_jump_blocked": 0,
            "forced_selected_idx": 2,
        }

    out = prepare_kr_general_cv_selection_context(
        **_base_kwargs(
            log_fn=logs.append,
            resolve_planned_cv_index_fn=lambda *_args, **_kwargs: 2,
            select_general_cv_index_fn=select_general,
            decide_cv_row_abstain_fn=lambda **kwargs: abstain_calls.append(kwargs) or {"should_skip": False},
        )
    )

    assert "KR CV 앵커 계획 보정" in logs[0]
    assert select_calls[0]["planned_cv_idx"] == 2
    assert out["selected_w_idx"] == 2
    assert out["cv_seq_idx"] == 3
    assert out["row_mapping_confidence"] == 0.71
    assert out["row_blank_conf"] == 0.3
    assert out["row_abstain"] == {"should_skip": False}
    assert abstain_calls[0]["candidate_active"] is True


def test_prepare_kr_general_cv_selection_prepares_vv_forced_and_planned_indices():
    select_calls = []
    vv_state = {}

    def select_general(**kwargs):
        select_calls.append(kwargs)
        return {
            "expected_cv_idx": 1,
            "selected_w_idx": 3,
            "cv_seq_idx": 4,
            "row_mapping_confidence": 0.6,
            "resolve_meta": {"vv": 1},
            "row_jump_blocked": 1,
            "forced_selected_idx": 4,
        }

    out = prepare_kr_general_cv_selection_context(
        **_base_kwargs(
            alias="a i",
            alias_type="vv",
            bridge_pair={"next_idx": 3},
            kr_cvvc_vv_occurrence_state=vv_state,
            row_blank_floor=None,
            syllable_blank_confidences=[0.1, 0.2, 0.3, 0.4],
            syllables_info=[
                {"active": True},
                {"active": True},
                {"active": True},
                {"active": True},
            ],
            resolve_vv_index_fn=lambda _alias, _map, state: state.setdefault("used", 2),
            select_general_cv_index_fn=select_general,
        )
    )

    assert select_calls[0]["forced_vv_idx"] == 2
    assert select_calls[0]["planned_vv_idx"] == 3
    assert out["forced_vv_idx"] == 2
    assert out["planned_vv_idx"] == 3
    assert out["row_jump_blocked"] == 1
    assert out["row_blank_floor_safe"] == 0.62
