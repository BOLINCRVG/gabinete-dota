from __future__ import annotations

import math
import unittest

from app_core.prob_math_core import clip_prob, isfinite_num, safe_float, safe_logit, safe_sigmoid


class TestSafeFloat(unittest.TestCase):

    def test_valid_string(self):
        self.assertEqual(safe_float("3.5"), 3.5)

    def test_nan_string_uses_default(self):
        self.assertEqual(safe_float("nan", default=1.25), 1.25)

    def test_invalid_string_uses_default(self):
        result = safe_float("abc")
        self.assertTrue(math.isnan(result))

    def test_integer_input(self):
        self.assertEqual(safe_float(7), 7.0)

    def test_returns_float_type(self):
        self.assertIsInstance(safe_float("2.0"), float)

    def test_none_uses_default(self):
        result = safe_float(None)
        self.assertTrue(math.isnan(result))


class TestIsFiniteNum(unittest.TestCase):

    def test_nan_string_not_finite(self):
        self.assertFalse(isfinite_num("nan"))

    def test_valid_float_finite(self):
        self.assertTrue(isfinite_num(1.0))

    def test_inf_not_finite(self):
        self.assertFalse(isfinite_num(float("inf")))

    def test_zero_is_finite(self):
        self.assertTrue(isfinite_num(0.0))

    def test_none_not_finite(self):
        self.assertFalse(isfinite_num(None))

    def test_returns_bool(self):
        self.assertIsInstance(isfinite_num(1.0), bool)


class TestClipProb(unittest.TestCase):

    def test_zero_clipped_above_zero(self):
        self.assertGreater(clip_prob(0.0), 0.0)

    def test_one_clipped_below_one(self):
        self.assertLess(clip_prob(1.0), 1.0)

    def test_half_unchanged(self):
        self.assertAlmostEqual(clip_prob(0.5), 0.5, places=10)

    def test_result_in_valid_range(self):
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = clip_prob(p)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_returns_float(self):
        self.assertIsInstance(clip_prob(0.5), float)


class TestSafeLogit(unittest.TestCase):

    def test_half_gives_zero(self):
        self.assertAlmostEqual(safe_logit(0.5), 0.0, places=8)

    def test_above_half_positive(self):
        self.assertGreater(safe_logit(0.75), 0.0)

    def test_below_half_negative(self):
        self.assertLess(safe_logit(0.25), 0.0)

    def test_returns_finite(self):
        self.assertTrue(math.isfinite(safe_logit(0.99)))

    def test_returns_float(self):
        self.assertIsInstance(safe_logit(0.6), float)


class TestSafeSigmoid(unittest.TestCase):

    def test_zero_gives_half(self):
        self.assertAlmostEqual(safe_sigmoid(0.0), 0.5, places=8)

    def test_positive_above_half(self):
        self.assertGreater(safe_sigmoid(1.0), 0.5)

    def test_negative_below_half(self):
        self.assertLess(safe_sigmoid(-1.0), 0.5)

    def test_logit_sigmoid_roundtrip(self):
        p = 0.73
        self.assertAlmostEqual(safe_sigmoid(safe_logit(p)), p, places=8)

    def test_returns_float(self):
        self.assertIsInstance(safe_sigmoid(0.0), float)

    def test_large_positive_near_one(self):
        self.assertGreater(safe_sigmoid(35.0), 0.99)

    def test_large_negative_near_zero(self):
        self.assertLess(safe_sigmoid(-35.0), 0.01)


if __name__ == "__main__":
    unittest.main()
