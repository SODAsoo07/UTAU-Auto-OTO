import unittest


raise unittest.SkipTest("legacy dependency removed: sofa_utau_kr")


class FullLabelDurationTests(unittest.TestCase):
    def test_normalize_durations_matches_wav_length(self):
        result = normalize_durations_to_wav_length(
            durations=[0.1, 0.2, 0.3],
            wav_length_sec=1.0,
            round_decimals=5,
        )
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_normalize_durations_raises_on_invalid_input(self):
        with self.assertRaises(ValueError):
            normalize_durations_to_wav_length(
                durations=[0.0, 0.0],
                wav_length_sec=1.0,
                round_decimals=5,
            )


if __name__ == "__main__":
    unittest.main()

