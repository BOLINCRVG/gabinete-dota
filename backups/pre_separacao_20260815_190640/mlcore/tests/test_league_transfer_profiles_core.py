"""Tests for mlcore/league_transfer_profiles_core.py — pure weight helpers."""
import unittest

from mlcore.league_transfer_profiles_core import _playoffs_weight, _year_weight


class TestYearWeight(unittest.TestCase):

    def test_2026_returns_1(self):
        self.assertAlmostEqual(_year_weight(2026), 1.00)

    def test_2025_returns_0_70(self):
        self.assertAlmostEqual(_year_weight(2025), 0.70)

    def test_2024_returns_0_45(self):
        self.assertAlmostEqual(_year_weight(2024), 0.45)

    def test_older_year_returns_0_35(self):
        self.assertAlmostEqual(_year_weight(2020), 0.35)

    def test_2023_returns_0_35(self):
        self.assertAlmostEqual(_year_weight(2023), 0.35)

    def test_none_returns_0_35(self):
        self.assertAlmostEqual(_year_weight(None), 0.35)

    def test_string_year_2025(self):
        self.assertAlmostEqual(_year_weight("2025"), 0.70)

    def test_float_year_2026(self):
        self.assertAlmostEqual(_year_weight(2026.0), 1.00)

    def test_invalid_string_returns_0_35(self):
        self.assertAlmostEqual(_year_weight("abc"), 0.35)

    def test_returns_float(self):
        self.assertIsInstance(_year_weight(2025), float)

    def test_future_year_returns_1(self):
        self.assertAlmostEqual(_year_weight(2030), 1.00)


class TestPlayoffsWeight(unittest.TestCase):

    def test_playoffs_string_returns_1_12(self):
        self.assertAlmostEqual(_playoffs_weight("playoffs"), 1.12)

    def test_true_string_returns_1_12(self):
        self.assertAlmostEqual(_playoffs_weight("true"), 1.12)

    def test_one_string_returns_1_12(self):
        self.assertAlmostEqual(_playoffs_weight("1"), 1.12)

    def test_sim_returns_1_12(self):
        self.assertAlmostEqual(_playoffs_weight("sim"), 1.12)

    def test_yes_returns_1_12(self):
        self.assertAlmostEqual(_playoffs_weight("yes"), 1.12)

    def test_empty_returns_1(self):
        self.assertAlmostEqual(_playoffs_weight(""), 1.0)

    def test_false_string_returns_1(self):
        self.assertAlmostEqual(_playoffs_weight("false"), 1.0)

    def test_none_returns_1(self):
        self.assertAlmostEqual(_playoffs_weight(None), 1.0)

    def test_zero_string_returns_1(self):
        self.assertAlmostEqual(_playoffs_weight("0"), 1.0)

    def test_case_insensitive_playoffs(self):
        self.assertAlmostEqual(_playoffs_weight("Playoffs"), 1.12)

    def test_case_insensitive_true(self):
        self.assertAlmostEqual(_playoffs_weight("True"), 1.12)

    def test_returns_float(self):
        self.assertIsInstance(_playoffs_weight("playoffs"), float)


if __name__ == "__main__":
    unittest.main()
