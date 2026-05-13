from core.coarse_crnn.slot_graph import apply_slot_context, slot_context_from_role, slot_type_for_role


def test_slot_type_maps_vc_to_boundary_not_anchor():
    assert slot_type_for_role("cv") == "anchor"
    assert slot_type_for_role("v") == "anchor"
    assert slot_type_for_role("vc") == "boundary"
    assert slot_type_for_role("vv") == "boundary"
    assert slot_type_for_role("v-cv") == "boundary"
    assert slot_type_for_role("endbr") == "final"


def test_apply_slot_context_assigns_boundary_ordinals_separate_from_cv_rows():
    rows = [
        {"wav": "ka_ki_ku.wav", "alias": "ka", "alias_role": "cv"},
        {"wav": "ka_ki_ku.wav", "alias": "ki", "alias_role": "cv"},
        {"wav": "ka_ki_ku.wav", "alias": "ku", "alias_role": "cv"},
        {"wav": "ka_ki_ku.wav", "alias": "a k", "alias_role": "vc"},
        {"wav": "ka_ki_ku.wav", "alias": "i k", "alias_role": "vc"},
    ]

    apply_slot_context(rows)

    assert [row["slot_type"] for row in rows] == ["anchor", "anchor", "anchor", "boundary", "boundary"]
    assert [row["slot_index"] for row in rows[:3]] == [0, 1, 2]
    assert [row["slot_index"] for row in rows[3:]] == [0, 1]
    assert rows[3]["slot_ratio_in_wav"] == 0.0
    assert rows[4]["slot_ratio_in_wav"] == 1.0


def test_slot_context_from_role_marks_vc_as_boundary():
    context = slot_context_from_role("vc", row_index_in_wav=3, file_row_count=8)

    assert len(context) == 6
    assert context[1] == 0.0
    assert context[2] == 1.0
    assert context[3] == 0.0
    assert context[4] == 0.5
