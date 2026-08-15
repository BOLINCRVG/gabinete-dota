"""Tests for pure helpers in mlcore/league_transfer_loader_core.py and
mlcore/league_transfer_profiles_core.py."""
import unittest

import pandas as pd

from mlcore.league_transfer_loader_core import _norm_pos, _shrink_prob, _weighted_mean, _weighted_rate
from mlcore.league_transfer_profiles_core import _playoffs_weight, _year_weight


class TestShrinkProb(unittest.TestCase):

    def test_zero_n_returns_prior(self):
        result = _shrink_prob(0.7, 0.0, prior=0.5, k=10.0)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_large_n_approaches_p(self):
        result = _shrink_prob(0.8, 10000.0, prior=0.5, k=10.0)
        self.assertAlmostEqual(result, 0.8, places=2)

    def test_midpoint(self):
        result = _shrink_prob(1.0, 10.0, prior=0.0, k=10.0)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_result_between_prior_and_p(self):
        result = _shrink_prob(0.8, 5.0, prior=0.5, k=10.0)
        self.assertGreater(result, 0.5)
        self.assertLess(result, 0.8)


class TestNormPos(unittest.TestCase):

    def test_top(self):
        self.assertEqual(_norm_pos("top"), "Top")

    def test_jungle_aliases(self):
        self.assertEqual(_norm_pos("jng"), "Jungle")
        self.assertEqual(_norm_pos("jg"), "Jungle")

    def test_mid(self):
        self.assertEqual(_norm_pos("mid"), "Mid")

    def test_bot_aliases(self):
        self.assertEqual(_norm_pos("bot"), "Bot")
        self.assertEqual(_norm_pos("adc"), "Bot")

    def test_support_aliases(self):
        self.assertEqual(_norm_pos("sup"), "Support")
        self.assertEqual(_norm_pos("support"), "Support")

    def test_empty_returns_empty(self):
        self.assertEqual(_norm_pos(""), "")

    def test_unknown_returns_empty(self):
        self.assertEqual(_norm_pos("unknown"), "")

    def test_case_insensitive(self):
        self.assertEqual(_norm_pos("TOP"), "Top")
        self.assertEqual(_norm_pos("MID"), "Mid")


class TestWeightedMean(unittest.TestCase):

    def test_equal_weights(self):
        v = pd.Series([1.0, 2.0, 3.0])
        w = pd.Series([1.0, 1.0, 1.0])
        result = _weighted_mean(v, w)
        self.assertAlmostEqual(result, 2.0, places=6)

    def test_weighted_result(self):
        v = pd.Series([0.0, 1.0])
        w = pd.Series([1.0, 3.0])
        result = _weighted_mean(v, w)
        self.assertAlmostEqual(result, 0.75, places=6)

    def test_empty_returns_default(self):
        v = pd.Series([], dtype=float)
        w = pd.Series([], dtype=float)
        result = _weighted_mean(v, w, default=99.0)
        self.assertAlmostEqual(result, 99.0, places=6)

    def test_zero_weights_returns_default(self):
        v = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        result = _weighted_mean(v, w, default=-1.0)
        self.assertAlmostEqual(result, -1.0, places=6)


class TestWeightedRate(unittest.TestCase):

    def test_all_true(self):
        mask = pd.Series([True, True, True])
        weights = pd.Series([1.0, 1.0, 1.0])
        result = _weighted_rate(mask, weights)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_half_true(self):
        mask = pd.Series([True, False])
        weights = pd.Series([1.0, 1.0])
        result = _weighted_rate(mask, weights)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_zero_weights_returns_default(self):
        mask = pd.Series([True, True])
        weights = pd.Series([0.0, 0.0])
        result = _weighted_rate(mask, weights, default=0.3)
        self.assertAlmostEqual(result, 0.3, places=6)


class TestYearWeight(unittest.TestCase):

    def test_2026_is_one(self):
        self.assertAlmostEqual(_year_weight(2026), 1.0, places=6)

    def test_2025_is_lower(self):
        self.assertLess(_year_weight(2025), 1.0)
        self.assertGreater(_year_weight(2025), 0.0)

    def test_2024_lower_than_2025(self):
        self.assertLess(_year_weight(2024), _year_weight(2025))

    def test_old_year_uses_fallback(self):
        self.assertAlmostEqual(_year_weight(2020), 0.35, places=6)

    def test_invalid_returns_default(self):
        self.assertAlmostEqual(_year_weight("bad"), 0.35, places=6)

    def test_future_year_is_one(self):
        self.assertAlmostEqual(_year_weight(2030), 1.0, places=6)


class TestPlayoffsWeight(unittest.TestCase):

    def test_true_string_gives_boost(self):
        result = _playoffs_weight("true")
        self.assertGreater(result, 1.0)

    def test_one_gives_boost(self):
        result = _playoffs_weight("1")
        self.assertGreater(result, 1.0)

    def test_yes_gives_boost(self):
        result = _playoffs_weight("yes")
        self.assertGreater(result, 1.0)

    def test_false_gives_one(self):
        self.assertAlmostEqual(_playoffs_weight("false"), 1.0, places=6)

    def test_empty_gives_one(self):
        self.assertAlmostEqual(_playoffs_weight(""), 1.0, places=6)

    def test_none_gives_one(self):
        self.assertAlmostEqual(_playoffs_weight(None), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
