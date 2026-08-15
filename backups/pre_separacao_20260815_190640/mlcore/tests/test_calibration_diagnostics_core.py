import math
import unittest

from mlcore.calibration_diagnostics import (
    _apply_microseg_calibration,
    _ece,
    _safe_logit,
    _safe_sigmoid,
    _seg_key,
)


class CalibrationDiagnosticsCoreTest(unittest.TestCase):
    def test_microseg_adjusts_probability_with_strong_local_bias(self):
        table = {
            "LPL|S1|blue|1|regular": {"n": 120.0, "bias": -0.08},
            "*|*|*|*|*": {"n": 500.0, "bias": 0.00},
        }
        p0 = 0.70
        p1 = _apply_microseg_calibration(
            p0,
            league="LPL",
            split="S1",
            side="blue",
            bo="1",
            phase="regular",
            table=table,
            min_n=30,
            strength=1.0,
            cap_pp=10.0,
            hierarchical=False,
        )
        self.assertLess(float(p1), float(p0))

    def test_microseg_falls_back_to_global_hierarchical(self):
        table = {
            "*|*|*|*|*": {"n": 400.0, "bias": 0.05},
        }
        p0 = 0.45
        p1 = _apply_microseg_calibration(
            p0,
            league="LCK",
            split="S1",
            side="blue",
            bo="1",
            phase="regular",
            table=table,
            min_n=30,
            strength=0.7,
            cap_pp=8.0,
            hierarchical=True,
            hierarchy_decay=0.8,
        )
        self.assertGreater(float(p1), float(p0))


class TestEce(unittest.TestCase):

    def test_perfect_calibration_near_zero(self):
        y = [1, 0, 1, 0]
        p = [0.875, 0.125, 0.875, 0.125]
        result = _ece(y, p, bins=2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_empty_lists_return_nan(self):
        self.assertTrue(math.isnan(_ece([], [], bins=10)))

    def test_mismatched_lengths_return_nan(self):
        self.assertTrue(math.isnan(_ece([1, 0], [0.5], bins=10)))

    def test_returns_float(self):
        self.assertIsInstance(_ece([1, 0], [0.7, 0.3]), float)

    def test_result_between_0_and_1(self):
        y = [1, 1, 0, 0, 1]
        p = [0.8, 0.7, 0.3, 0.2, 0.6]
        result = _ece(y, p)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestSafeLogit(unittest.TestCase):

    def test_half_gives_zero(self):
        self.assertAlmostEqual(_safe_logit(0.5), 0.0, places=5)

    def test_high_prob_positive(self):
        self.assertGreater(_safe_logit(0.75), 0.0)

    def test_low_prob_negative(self):
        self.assertLess(_safe_logit(0.25), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(_safe_logit(0.6), float)

    def test_result_is_finite(self):
        self.assertTrue(math.isfinite(_safe_logit(0.5)))

    def test_clipped_zero_not_inf(self):
        self.assertTrue(math.isfinite(_safe_logit(0.0)))


class TestSafeSigmoid(unittest.TestCase):

    def test_zero_gives_half(self):
        self.assertAlmostEqual(_safe_sigmoid(0.0), 0.5, places=5)

    def test_positive_above_half(self):
        self.assertGreater(_safe_sigmoid(1.0), 0.5)

    def test_negative_below_half(self):
        self.assertLess(_safe_sigmoid(-1.0), 0.5)

    def test_returns_float(self):
        self.assertIsInstance(_safe_sigmoid(0.0), float)

    def test_large_positive_near_one(self):
        self.assertGreater(_safe_sigmoid(35.0), 0.99)

    def test_large_negative_near_zero(self):
        self.assertLess(_safe_sigmoid(-35.0), 0.01)

    def test_symmetric_around_half(self):
        a = _safe_sigmoid(2.0)
        b = _safe_sigmoid(-2.0)
        self.assertAlmostEqual(a + b, 1.0, places=5)


class TestSegKey(unittest.TestCase):

    def test_builds_pipe_delimited_key(self):
        key = _seg_key("LCK", "Spring", "blue", "1", "regular")
        parts = key.split("|")
        self.assertEqual(len(parts), 5)

    def test_league_in_key(self):
        key = _seg_key("LCK", "S1", "blue", "1", "regular")
        self.assertIn("LCK", key)

    def test_side_lowercased(self):
        key = _seg_key("LCK", "S1", "BLUE", "1", "regular")
        self.assertIn("blue", key)

    def test_phase_lowercased(self):
        key = _seg_key("LCK", "S1", "blue", "1", "REGULAR")
        self.assertIn("regular", key)

    def test_wildcard_key(self):
        key = _seg_key("*", "*", "*", "*", "*")
        self.assertEqual(key, "*|*|*|*|*")

    def test_returns_string(self):
        self.assertIsInstance(_seg_key("LCK", "S1", "blue", "1", "regular"), str)


if __name__ == "__main__":
    unittest.main()
