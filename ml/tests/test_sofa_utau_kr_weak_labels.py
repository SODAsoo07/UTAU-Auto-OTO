import unittest


raise unittest.SkipTest("legacy dependency removed: sofa_utau_kr")


class WeakLabelAliasTests(unittest.TestCase):
    def test_enforce_sp_rules_collapses_duplicates(self):
        tokens = ["SP", "SP", "a", "SP", "SP", "k", "SP"]
        self.assertEqual(enforce_sp_rules(tokens), ["SP", "a", "SP", "k", "SP"])

    def test_normalize_alias_to_ph_seq_with_strip_suffix(self):
        ph = normalize_alias_to_ph_seq(
            alias="- a_C4",
            phoneme_map={},
            sp_tokens={"SP", "-", "R"},
            strip_suffixes=["_C4"],
        )
        self.assertEqual(ph, "SP a SP")

    def test_normalize_alias_to_ph_seq_with_custom_map(self):
        ph = normalize_alias_to_ph_seq(
            alias="VV",
            phoneme_map={"VV": ["a", "i"]},
            sp_tokens={"SP", "-", "R"},
            strip_suffixes=[],
        )
        self.assertEqual(ph, "SP a i SP")


if __name__ == "__main__":
    unittest.main()

