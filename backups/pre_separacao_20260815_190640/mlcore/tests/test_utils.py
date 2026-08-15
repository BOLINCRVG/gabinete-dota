"""Tests for mlcore/utils.py — math and conversion helpers."""
import math
import unittest

import numpy as np

from mlcore.utils import (
    brier,
    clip,
    clip_prob,
    clip_safe,
    decimal_odds_to_prob,
    isfinite_num,
    logit,
    logloss,
    prob_to_decimal_odds,
    safe_float,
    safe_float_0,
    safe_float_or_none,
    sigmoid,
    wmean,
)


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat(unittest.TestCase):

    def test_valid_int(self):
        self.assertAlmostEqual(safe_float(3), 3.0)

    def test_valid_string(self):
        self.assertAlmostEqual(safe_float("2.5"), 2.5)

    def test_nan_returns_default(self):
        result = safe_float(float("nan"), 0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_inf_returns_default(self):
        result = safe_float(float("inf"), -1.0)
        self.assertAlmostEqual(result, -1.0)

    def test_invalid_string_returns_default(self):
        result = safe_float("abc", 99.0)
        self.assertAlmostEqual(result, 99.0)

    def test_none_returns_default(self):
        result = safe_float(None, 7.0)
        self.assertAlmostEqual(result, 7.0)

    def test_default_is_nan_when_omitted(self):
        result = safe_float("bad")
        self.assertTrue(math.isnan(result))

    def test_negative_value(self):
        self.assertAlmostEqual(safe_float(-3.14), -3.14)

    def test_zero(self):
        self.assertAlmostEqual(safe_float(0), 0.0)


# ---------------------------------------------------------------------------
# safe_float_0
# ---------------------------------------------------------------------------

class TestSafeFloat0(unittest.TestCase):

    def test_valid_returns_value(self):
        self.assertAlmostEqual(safe_float_0(1.5), 1.5)

    def test_invalid_returns_zero(self):
        self.assertAlmostEqual(safe_float_0("abc"), 0.0)

    def test_none_returns_zero(self):
        self.assertAlmostEqual(safe_float_0(None), 0.0)

    def test_nan_returns_zero(self):
        self.assertAlmostEqual(safe_float_0(float("nan")), 0.0)

    def test_numeric_string(self):
        self.assertAlmostEqual(safe_float_0("3.7"), 3.7)


# ---------------------------------------------------------------------------
# safe_float_or_none
# ---------------------------------------------------------------------------

class TestSafeFloatOrNone(unittest.TestCase):

    def test_valid_returns_float(self):
        result = safe_float_or_none(2.5)
        self.assertAlmostEqual(result, 2.5)

    def test_nan_returns_none(self):
        self.assertIsNone(safe_float_or_none(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(safe_float_or_none(float("inf")))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(safe_float_or_none("xyz"))

    def test_none_returns_none(self):
        self.assertIsNone(safe_float_or_none(None))

    def test_zero_returns_zero(self):
        self.assertAlmostEqual(safe_float_or_none(0), 0.0)


# ---------------------------------------------------------------------------
# isfinite_num
# ---------------------------------------------------------------------------

class TestIsfiniteNum(unittest.TestCase):

    def test_finite_int(self):
        self.assertTrue(isfinite_num(3))

    def test_finite_float(self):
        self.assertTrue(isfinite_num(3.14))

    def test_nan(self):
        self.assertFalse(isfinite_num(float("nan")))

    def test_inf(self):
        self.assertFalse(isfinite_num(float("inf")))

    def test_neg_inf(self):
        self.assertFalse(isfinite_num(float("-inf")))

    def test_string_number(self):
        self.assertTrue(isfinite_num("2.0"))

    def test_invalid_string(self):
        self.assertFalse(isfinite_num("abc"))

    def test_none(self):
        self.assertFalse(isfinite_num(None))


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------

class TestClip(unittest.TestCase):

    def test_within_range(self):
        self.assertAlmostEqual(clip(0.5, 0.0, 1.0), 0.5)

    def test_below_lo(self):
        self.assertAlmostEqual(clip(-1.0, 0.0, 1.0), 0.0)

    def test_above_hi(self):
        self.assertAlmostEqual(clip(2.0, 0.0, 1.0), 1.0)

    def test_exactly_lo(self):
        self.assertAlmostEqual(clip(0.0, 0.0, 1.0), 0.0)

    def test_exactly_hi(self):
        self.assertAlmostEqual(clip(1.0, 0.0, 1.0), 1.0)

    def test_negative_range(self):
        self.assertAlmostEqual(clip(-5.0, -10.0, -1.0), -5.0)


# ---------------------------------------------------------------------------
# clip_safe
# ---------------------------------------------------------------------------

class TestClipSafe(unittest.TestCase):

    def test_valid_float(self):
        self.assertAlmostEqual(clip_safe(0.5, 0.0, 1.0), 0.5)

    def test_invalid_maps_to_lo(self):
        self.assertAlmostEqual(clip_safe("abc", 0.0, 1.0), 0.0)

    def test_none_maps_to_lo(self):
        self.assertAlmostEqual(clip_safe(None, 0.0, 1.0), 0.0)

    def test_nan_maps_to_lo(self):
        self.assertAlmostEqual(clip_safe(float("nan"), 0.1, 0.9), 0.1)

    def test_above_hi_clipped(self):
        self.assertAlmostEqual(clip_safe(5.0, 0.0, 1.0), 1.0)


# ---------------------------------------------------------------------------
# clip_prob
# ---------------------------------------------------------------------------

class TestClipProb(unittest.TestCase):

    def test_half_stays(self):
        self.assertAlmostEqual(clip_prob(0.5), 0.5)

    def test_zero_clipped_to_eps(self):
        result = clip_prob(0.0)
        self.assertGreater(result, 0.0)
        self.assertAlmostEqual(result, 1e-6)

    def test_one_clipped_below_one(self):
        result = clip_prob(1.0)
        self.assertLess(result, 1.0)
        self.assertAlmostEqual(result, 1.0 - 1e-6)

    def test_custom_eps(self):
        result = clip_prob(0.0, eps=0.01)
        self.assertAlmostEqual(result, 0.01)


# ---------------------------------------------------------------------------
# wmean
# ---------------------------------------------------------------------------

class TestWmean(unittest.TestCase):

    def test_equal_weights(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(wmean(v, w), 2.0)

    def test_unequal_weights(self):
        v = np.array([0.0, 1.0])
        w = np.array([1.0, 3.0])
        self.assertAlmostEqual(wmean(v, w), 0.75)

    def test_all_nan_values_returns_nan(self):
        v = np.array([float("nan"), float("nan")])
        w = np.array([1.0, 1.0])
        self.assertTrue(math.isnan(wmean(v, w)))

    def test_zero_weights_skipped(self):
        v = np.array([10.0, 2.0])
        w = np.array([0.0, 1.0])
        self.assertAlmostEqual(wmean(v, w), 2.0)

    def test_empty_arrays(self):
        v = np.array([])
        w = np.array([])
        self.assertTrue(math.isnan(wmean(v, w)))

    def test_single_element(self):
        v = np.array([5.0])
        w = np.array([2.0])
        self.assertAlmostEqual(wmean(v, w), 5.0)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid(unittest.TestCase):

    def test_zero_gives_half(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_positive_gives_gt_half(self):
        self.assertGreater(sigmoid(1.0), 0.5)

    def test_negative_gives_lt_half(self):
        self.assertLess(sigmoid(-1.0), 0.5)

    def test_large_positive_approaches_one(self):
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=5)

    def test_large_negative_approaches_zero(self):
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=5)

    def test_symmetry(self):
        self.assertAlmostEqual(sigmoid(2.0) + sigmoid(-2.0), 1.0, places=10)

    def test_output_in_unit_interval(self):
        for z in (-10.0, -1.0, 0.0, 1.0, 10.0):
            s = sigmoid(z)
            self.assertGreater(s, 0.0)
            self.assertLess(s, 1.0)


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------

class TestLogit(unittest.TestCase):

    def test_half_gives_zero(self):
        self.assertAlmostEqual(logit(0.5), 0.0, places=10)

    def test_gt_half_positive(self):
        self.assertGreater(logit(0.75), 0.0)

    def test_lt_half_negative(self):
        self.assertLess(logit(0.25), 0.0)

    def test_inverse_of_sigmoid(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertAlmostEqual(sigmoid(logit(p)), p, places=10)

    def test_extreme_p_clipped(self):
        # Should not raise even for p=0 or p=1
        result_zero = logit(0.0)
        result_one = logit(1.0)
        self.assertIsInstance(result_zero, float)
        self.assertIsInstance(result_one, float)


# ---------------------------------------------------------------------------
# prob_to_decimal_odds / decimal_odds_to_prob
# ---------------------------------------------------------------------------

class TestOddsConversions(unittest.TestCase):

    def test_prob_half_gives_odds_two(self):
        self.assertAlmostEqual(prob_to_decimal_odds(0.5), 2.0, places=5)

    def test_prob_one_gives_odds_one(self):
        result = prob_to_decimal_odds(1.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_prob_to_odds_always_finite(self):
        for p in (0.1, 0.5, 0.9):
            self.assertTrue(math.isfinite(prob_to_decimal_odds(p)))

    def test_decimal_odds_two_gives_half(self):
        self.assertAlmostEqual(decimal_odds_to_prob(2.0), 0.5, places=10)

    def test_decimal_odds_le_one_gives_nan(self):
        self.assertTrue(math.isnan(decimal_odds_to_prob(1.0)))
        self.assertTrue(math.isnan(decimal_odds_to_prob(0.5)))

    def test_decimal_odds_inf_gives_nan(self):
        self.assertTrue(math.isnan(decimal_odds_to_prob(float("inf"))))

    def test_roundtrip(self):
        for p in (0.3, 0.5, 0.7):
            self.assertAlmostEqual(decimal_odds_to_prob(prob_to_decimal_odds(p)), p, places=8)


# ---------------------------------------------------------------------------
# logloss
# ---------------------------------------------------------------------------

class TestLogloss(unittest.TestCase):

    def test_perfect_predictions_near_zero(self):
        result = logloss([1, 0], [0.9999, 0.0001])
        self.assertLess(result, 0.001)

    def test_random_predictions_near_ln2(self):
        result = logloss([1, 0], [0.5, 0.5])
        self.assertAlmostEqual(result, math.log(2), places=5)

    def test_empty_returns_zero(self):
        result = logloss([], [])
        self.assertAlmostEqual(result, 0.0)

    def test_single_correct(self):
        result = logloss([1], [0.9])
        self.assertAlmostEqual(result, -math.log(0.9), places=8)

    def test_single_wrong(self):
        result = logloss([0], [0.9])
        self.assertAlmostEqual(result, -math.log(0.1), places=8)

    def test_result_nonnegative(self):
        result = logloss([1, 0, 1], [0.8, 0.2, 0.6])
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------

class TestBrier(unittest.TestCase):

    def test_perfect_predictions_zero(self):
        result = brier([1, 0], [1.0, 0.0])
        self.assertAlmostEqual(result, 0.0)

    def test_uniform_gives_quarter(self):
        result = brier([1, 0], [0.5, 0.5])
        self.assertAlmostEqual(result, 0.25)

    def test_wrong_predictions_high(self):
        result = brier([1, 0], [0.0, 1.0])
        self.assertAlmostEqual(result, 1.0)

    def test_empty_returns_zero(self):
        result = brier([], [])
        self.assertAlmostEqual(result, 0.0)

    def test_result_nonnegative(self):
        result = brier([1, 0, 1], [0.8, 0.2, 0.6])
        self.assertGreaterEqual(result, 0.0)

    def test_single_sample(self):
        result = brier([1], [0.7])
        self.assertAlmostEqual(result, (0.7 - 1.0) ** 2)


if __name__ == "__main__":
    unittest.main()
