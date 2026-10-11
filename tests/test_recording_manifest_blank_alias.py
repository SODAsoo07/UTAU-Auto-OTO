import pytest

from universal_auto_oto.voicebank_project_adapter import _parse_manifest_aliases, VoicebankReviewRequired


def test_explicit_empty_oto_alias_keeps_its_phone_occurrence():
    row = {"alias": "", "phones": [0], "template": "builtin:breath", "role": "breath"}
    plan, = _parse_manifest_aliases({"aliases": [row]}, 1, "br1.wav")
    assert plan.alias == ""
    assert plan.phone_indices == (0,)
    assert plan.template == "builtin:breath"


@pytest.mark.parametrize("alias", [None, 123])
def test_missing_or_nontext_alias_is_still_rejected(alias):
    row = {"phones": [0], "template": "builtin:breath"}
    if alias is not None:
        row["alias"] = alias
    with pytest.raises(VoicebankReviewRequired, match="invalid_manifest_alias"):
        _parse_manifest_aliases({"aliases": [row]}, 1, "br1.wav")
