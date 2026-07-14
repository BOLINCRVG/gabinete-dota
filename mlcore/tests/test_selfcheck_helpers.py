"""Tests for pure helper functions in mlcore/selfcheck*.py modules."""
import math
import unittest

from mlcore.selfcheck import _p_from_diff_points
from mlcore.selfcheck_lines import _fmt as _fmt_lines, _fmt_odds
from mlcore.selfcheck_stats import _fmt as _fmt_stats


class TestPFromDiffPoints(unittest.TestCase):

    def test_zero_diff_returns_half(self):
        result = _p_from_diff_points(0.0, 400.0)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_positive_diff_above_half(self):
        result = _p_from_diff_points(200.0, 400.0)
        self.assertGreater(result, 0.5)

    def test_negative_diff_below_half(self):
        result = _p_from_diff_points(-200.0, 400.0)
        self.assertLess(result, 0.5)

    def test_returns_float(self):
        result = _p_from_diff_points(100.0, 400.0)
        self.assertIsInstance(result, float)

    def test_result_strictly_between_0_and_1(self):
        result = _p_from_diff_points(100.0, 400.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_symmetric(self):
        p = _p_from_diff_points(100.0, 400.0)
        q = _p_from_diff_points(-100.0, 400.0)
        self.assertAlmostEqual(p + q, 1.0, places=5)

    def test_large_diff_near_1(self):
        result = _p_from_diff_points(2000.0, 400.0)
        self.assertGreater(result, 0.99)

    def test_result_is_finite(self):
        result = _p_from_diff_points(0.0, 400.0)
        self.assertTrue(math.isfinite(result))


class TestFmtLines(unittest.TestCase):
    """_fmt from selfcheck_lines.py."""

    def test_finite_value_formatted(self):
        result = _fmt_lines(1.23456, 3)
        self.assertEqual(result, "1.235")

    def test_nan_returns_dash(self):
        result = _fmt_lines(float("nan"), 3)
        self.assertEqual(result, "-")

    def test_inf_returns_dash(self):
        result = _fmt_lines(float("inf"), 3)
        self.assertEqual(result, "-")

    def test_zero_decimals(self):
        result = _fmt_lines(1.7, 0)
        self.assertEqual(result, "2")

    def test_returns_string(self):
        self.assertIsInstance(_fmt_lines(1.0, 2), str)


class TestFmtOdds(unittest.TestCase):

    def test_valid_odd_formatted(self):
        result = _fmt_odds(1.85)
        self.assertEqual(result, "1.85")

    def test_nan_returns_dash(self):
        result = _fmt_odds(float("nan"))
        self.assertEqual(result, "-")

    def test_inf_returns_dash(self):
        result = _fmt_odds(float("inf"))
        self.assertEqual(result, "-")

    def test_returns_string(self):
        self.assertIsInstance(_fmt_odds(2.0), str)

    def test_two_decimal_places(self):
        result = _fmt_odds(1.9)
        self.assertEqual(result, "1.90")


class TestFmtStats(unittest.TestCase):
    """_fmt from selfcheck_stats.py."""

    def test_regular_value(self):
        result = _fmt_stats(1.5)
        self.assertEqual(result, "1.50")

    def test_nan_returns_dash(self):
        result = _fmt_stats(float("nan"))
        self.assertEqual(result, "-")

    def test_returns_string(self):
        self.assertIsInstance(_fmt_stats(0.0), str)

    def test_two_decimals(self):
        result = _fmt_stats(3.14159)
        self.assertEqual(result, "3.14")


if __name__ == "__main__":
    unittest.main()
