from __future__ import annotations

from ml.scripts.coarse_crnn.train_oto import _cap_rows_stratified


def test_cap_rows_stratified_keeps_later_language_slices():
    rows = [
        {"id": "ja_cvvc_vc_1", "language": "japanese", "format_type": "cvvc", "alias_role": "vc"},
        {"id": "ja_cvvc_vc_2", "language": "japanese", "format_type": "cvvc", "alias_role": "vc"},
        {"id": "ja_cvvc_vv_1", "language": "japanese", "format_type": "cvvc", "alias_role": "vv"},
        {"id": "ko_cvvc_vc_1", "language": "korean", "format_type": "cvvc", "alias_role": "vc"},
        {"id": "ko_cvc_v-cv_1", "language": "korean", "format_type": "cvc", "alias_role": "v-cv"},
        {"id": "ko_cvc_v-cv_2", "language": "korean", "format_type": "cvc", "alias_role": "v-cv"},
    ]

    capped = _cap_rows_stratified(rows, 4)

    assert {row["language"] for row in capped} == {"japanese", "korean"}
    assert {row["format_type"] for row in capped} == {"cvvc", "cvc"}
    assert {row["alias_role"] for row in capped} == {"vc", "vv", "v-cv"}


def test_cap_rows_stratified_is_noop_when_limit_is_not_active():
    rows = [{"id": "a"}, {"id": "b"}]

    assert _cap_rows_stratified(rows, 0) == rows
    assert _cap_rows_stratified(rows, 3) == rows
