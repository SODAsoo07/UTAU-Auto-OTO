import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core.lab_generator as lab_generator


class KoreanMfaDictionaryGuardTests(unittest.TestCase):
    def test_normalize_mfa_sensitive_marker_ipa(self):
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("R", "sil"), "h")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("r", "sil"), "ɾ")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("br", "sil"), "h")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("bre", "sil"), "h")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("H", "sil"), "h")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("x", "sil"), "sil")
        self.assertEqual(lab_generator._normalize_mfa_sensitive_marker_ipa("R", "h"), "h")

    def test_generate_dictionary_rewrites_sil_for_sensitive_markers(self):
        original_map = dict(lab_generator.KO_ROMAJI_IPA_TABLE)
        try:
            # Simulate external mapping that wrongly routes non-silence tokens to sil.
            lab_generator.KO_ROMAJI_IPA_TABLE["R"] = "sil"
            lab_generator.KO_ROMAJI_IPA_TABLE["br"] = "sil"

            with tempfile.TemporaryDirectory() as tmpdir:
                r_lab = os.path.join(tmpdir, "R.lab")
                br_lab = os.path.join(tmpdir, "br.lab")
                with open(r_lab, "w", encoding="utf-8") as f:
                    f.write("R")
                with open(br_lab, "w", encoding="utf-8") as f:
                    f.write("br")

                dict_path = os.path.join(tmpdir, "dictionary.txt")
                count_files, entry_count, errors = lab_generator.generate_dictionary(tmpdir, dict_path)

                self.assertEqual(errors, [])
                self.assertEqual(count_files, 2)
                self.assertGreaterEqual(entry_count, 2)

                entries = {}
                with open(dict_path, "r", encoding="utf-8") as f:
                    for line in f:
                        key, val = line.strip().split("\t", 1)
                        entries[key] = val

                self.assertEqual(entries.get("R"), "h")
                self.assertEqual(entries.get("br"), "h")
        finally:
            lab_generator.KO_ROMAJI_IPA_TABLE.clear()
            lab_generator.KO_ROMAJI_IPA_TABLE.update(original_map)

    def test_generate_dictionary_skips_multi_token_sentence_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lab_path = os.path.join(tmpdir, "ga'gi'gu.lab")
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write("가 기 구")

            dict_path = os.path.join(tmpdir, "dictionary.txt")
            count_files, entry_count, errors = lab_generator.generate_dictionary(tmpdir, dict_path)

            self.assertEqual(errors, [])
            self.assertEqual(count_files, 1)
            self.assertGreaterEqual(entry_count, 3)

            with open(dict_path, "r", encoding="utf-8") as f:
                keys = [line.split("\t", 1)[0] for line in f if "\t" in line]

            self.assertIn("가", keys)
            self.assertIn("기", keys)
            self.assertIn("구", keys)
            self.assertNotIn("가 기 구", keys)


if __name__ == "__main__":
    unittest.main()
