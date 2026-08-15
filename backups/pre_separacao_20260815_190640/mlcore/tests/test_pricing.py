import math
import unittest

from mlcore.bo import p_series_from_pmap
from mlcore.utils import (
    clip,
    clip_safe,
    decimal_odds_to_prob,
    logit,
    logloss,
    prob_to_decimal_odds,
    sigmoid,
)


class TestProbDecimalOddsRoundtrip(unittest.TestCase):

    def test_roundtrip_for_various_probs(self):
        for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
            odd = prob_to_decimal_odds(p)
            self.assertGreater(odd, 1.0)
            p2 = decimal_odds_to_prob(odd)
            self.assertAlmostEqual(p, p2, places=9)

    def test_prob_to_odd_half_gives_two(self):
        self.assertAlmostEqual(prob_to_decimal_odds(0.5), 2.0, places=6)

    def test_decimal_to_prob_two_gives_half(self):
        self.assertAlmostEqual(decimal_odds_to_prob(2.0), 0.5, places=6)

    def test_odds_always_above_one(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.assertGreater(prob_to_decimal_odds(p), 1.0)


class TestSeriesMath(unittest.TestCase):

    def test_symmetry_bo3_bo5(self):
        for bo in [3, 5]:
            for p in [0.2, 0.35, 0.5, 0.65, 0.8]:
                pa = p_series_from_pmap(p, bo)
                pb = p_series_from_pmap(1.0 - p, bo)
                self.assertAlmostEqual(pa + pb, 1.0, places=10)

    def test_monotone_by_bo_when_favorite(self):
        p = 0.6
        self.assertGreater(p_series_from_pmap(p, 3), p_series_from_pmap(p, 1))
        self.assertGreater(p_series_from_pmap(p, 5), p_series_from_pmap(p, 3))

    def test_half_prob_always_half(self):
        for bo in [1, 3, 5]:
            self.assertAlmostEqual(p_series_from_pmap(0.5, bo), 0.5, places=10)

    def test_invalid_bo_raises(self):
        with self.assertRaises(ValueError):
            p_series_from_pmap(0.57, 2)

    def test_result_in_range(self):
        self.assertTrue(0.0 < p_series_from_pmap(0.57, 7) < 1.0)


class TestClip(unittest.TestCase):

    def test_within_range_unchanged(self):
        self.assertAlmostEqual(clip(0.5, 0.0, 1.0), 0.5)

    def test_below_lo_clipped(self):
        self.assertAlmostEqual(clip(-1.0, 0.0, 1.0), 0.0)

    def test_above_hi_clipped(self):
        self.assertAlmostEqual(clip(2.0, 0.0, 1.0), 1.0)

    def test_clip_safe_invalid_returns_clipped(self):
        result = clip_safe("abc", 0.0, 1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestSigmoidLogit(unittest.TestCase):

    def test_sigmoid_zero_gives_half(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=8)

    def test_logit_half_gives_zero(self):
        self.assertAlmostEqual(logit(0.5), 0.0, places=8)

    def test_sigmoid_logit_roundtrip(self):
        p = 0.73
        self.assertAlmostEqual(sigmoid(logit(p)), p, places=8)

    def test_sigmoid_large_positive_near_one(self):
        self.assertGreater(sigmoid(35.0), 0.99)


class TestLogloss(unittest.TestCase):

    def test_perfect_predictions_low_loss(self):
        y = [1, 0, 1, 0]
        p = [0.99, 0.01, 0.99, 0.01]
        result = logloss(y, p)
        self.assertLess(result, 0.05)

    def test_random_predictions_near_log2(self):
        y = [1, 0, 1, 0]
        p = [0.5, 0.5, 0.5, 0.5]
        result = logloss(y, p)
        self.assertAlmostEqual(result, math.log(2), places=5)

    def test_returns_float(self):
        self.assertIsInstance(logloss([1, 0], [0.8, 0.2]), float)

    def test_result_positive(self):
        result = logloss([1, 0], [0.7, 0.3])
        self.assertGreater(result, 0.0)


if __name__ == "__main__":
    unittest.main()
