"""Tests for mlcore/calibration_diagnostics.py — pure helper functions."""
import math
import unittest

from mlcore.calibration_diagnostics import (
    _ece,
    _safe_logit,
    _safe_sigmoid,
    _seg_key,
)


class TestEce(unittest.TestCase):

    def test_perfect_predictions_low_ece(self):
        y = [1, 1, 0, 0]
        p = [0.95, 0.95, 0.05, 0.05]
        result = _ece(y, p, bins=10)
        self.assertLess(result, 0.1)

    def test_constant_half_prob_ece(self):
        y = [1, 0, 1, 0]
        p = [0.5, 0.5, 0.5, 0.5]
        result = _ece(y, p, bins=10)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_empty_inputs_return_nan(self):
        self.assertTrue(math.isnan(_ece([], [], bins=10)))

    def test_length_mismatch_returns_nan(self):
        self.assertTrue(math.isnan(_ece([1, 0], [0.5], bins=10)))

    def test_returns_float(self):
        self.assertIsInstance(_ece([1, 0], [0.9, 0.1], bins=5), float)

    def test_result_between_zero_and_one(self):
        y = [1, 0, 1, 0, 1]
        p = [0.8, 0.2, 0.7, 0.3, 0.6]
        result = _ece(y, p, bins=5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestSafeLogit(unittest.TestCase):

    def test_half_returns_zero(self):
        self.assertAlmostEqual(_safe_logit(0.5), 0.0, places=5)

    def test_above_half_positive(self):
        self.assertGreater(_safe_logit(0.7), 0.0)

    def test_below_half_negative(self):
        self.assertLess(_safe_logit(0.3), 0.0)

    def test_zero_clamped_not_inf(self):
        result = _safe_logit(0.0)
        self.assertTrue(math.isfinite(result))

    def test_one_clamped_not_inf(self):
        result = _safe_logit(1.0)
        self.assertTrue(math.isfinite(result))

    def test_returns_float(self):
        self.assertIsInstance(_safe_logit(0.5), float)


class TestSafeSigmoid(unittest.TestCase):

    def test_zero_returns_half(self):
        self.assertAlmostEqual(_safe_sigmoid(0.0), 0.5, places=5)

    def test_positive_above_half(self):
        self.assertGreater(_safe_sigmoid(2.0), 0.5)

    def test_negative_below_half(self):
        self.assertLess(_safe_sigmoid(-2.0), 0.5)

    def test_large_positive_clamped_below_one(self):
        result = _safe_sigmoid(1000.0)
        self.assertLess(result, 1.0)
        self.assertGreater(result, 0.99)

    def test_large_negative_clamped_above_zero(self):
        result = _safe_sigmoid(-1000.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.01)

    def test_returns_float(self):
        self.assertIsInstance(_safe_sigmoid(0.0), float)


class TestSegKey(unittest.TestCase):

    def test_builds_pipe_separated_key(self):
        key = _seg_key("LCK", "Spring", "Blue", "3", "Playoff")
        self.assertEqual(key, "LCK|Spring|blue|3|playoff")

    def test_lowercases_side_and_phase(self):
        key = _seg_key("LPL", "Summer", "RED", "5", "REGULAR")
        self.assertIn("red", key)
        self.assertIn("regular", key)

    def test_empty_fields(self):
        key = _seg_key("", "", "", "", "")
        self.assertEqual(key, "||||")

    def test_returns_string(self):
        self.assertIsInstance(_seg_key("LCK", "Spring", "Blue", "3", "Reg"), str)

    def test_strips_whitespace(self):
        key = _seg_key(" LCK ", " Spring ", " Blue ", " 3 ", " Reg ")
        self.assertEqual(key, "LCK|Spring|blue|3|reg")


if __name__ == "__main__":
    unittest.main()
