"""Tests for mlcore/calibration.py — Platt calibrator."""
import math
import unittest

from mlcore.calibration import PlattCalibrator, fit_platt_from_raw_probs


# ---------------------------------------------------------------------------
# PlattCalibrator.predict_proba
# ---------------------------------------------------------------------------

class TestPlattCalibratorPredict(unittest.TestCase):

    def test_identity_a1_b0_preserves_prob(self):
        cal = PlattCalibrator(a=1.0, b=0.0)
        for p in (0.3, 0.5, 0.7):
            self.assertAlmostEqual(cal.predict_proba(p), p, places=8)

    def test_positive_b_shifts_up(self):
        cal_base = PlattCalibrator(a=1.0, b=0.0)
        cal_up = PlattCalibrator(a=1.0, b=0.5)
        self.assertGreater(cal_up.predict_proba(0.5), cal_base.predict_proba(0.5))

    def test_negative_b_shifts_down(self):
        cal_base = PlattCalibrator(a=1.0, b=0.0)
        cal_down = PlattCalibrator(a=1.0, b=-0.5)
        self.assertLess(cal_down.predict_proba(0.5), cal_base.predict_proba(0.5))

    def test_larger_a_sharpens_toward_extremes(self):
        cal_sharp = PlattCalibrator(a=2.0, b=0.0)
        cal_flat = PlattCalibrator(a=0.5, b=0.0)
        # For p > 0.5, larger a → higher output
        self.assertGreater(cal_sharp.predict_proba(0.7), cal_flat.predict_proba(0.7))

    def test_output_in_unit_interval(self):
        cal = PlattCalibrator(a=1.5, b=0.2)
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            result = cal.predict_proba(p)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_half_with_zero_b_gives_half(self):
        cal = PlattCalibrator(a=2.0, b=0.0)
        self.assertAlmostEqual(cal.predict_proba(0.5), 0.5, places=8)


# ---------------------------------------------------------------------------
# PlattCalibrator.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestPlattCalibratorSerialize(unittest.TestCase):

    def test_to_dict_has_a_and_b(self):
        cal = PlattCalibrator(a=1.2, b=-0.3)
        d = cal.to_dict()
        self.assertIn("a", d)
        self.assertIn("b", d)

    def test_to_dict_values(self):
        cal = PlattCalibrator(a=1.2, b=-0.3)
        d = cal.to_dict()
        self.assertAlmostEqual(d["a"], 1.2)
        self.assertAlmostEqual(d["b"], -0.3)

    def test_from_dict_roundtrip(self):
        original = PlattCalibrator(a=1.5, b=0.2)
        restored = PlattCalibrator.from_dict(original.to_dict())
        self.assertAlmostEqual(original.a, restored.a)
        self.assertAlmostEqual(original.b, restored.b)

    def test_from_dict_defaults(self):
        cal = PlattCalibrator.from_dict({})
        self.assertAlmostEqual(cal.a, 1.0)
        self.assertAlmostEqual(cal.b, 0.0)

    def test_predict_preserved_after_roundtrip(self):
        original = PlattCalibrator(a=1.3, b=-0.1)
        restored = PlattCalibrator.from_dict(original.to_dict())
        for p in (0.4, 0.6, 0.8):
            self.assertAlmostEqual(original.predict_proba(p), restored.predict_proba(p), places=10)


# ---------------------------------------------------------------------------
# fit_platt_from_raw_probs
# ---------------------------------------------------------------------------

class TestFitPlattFromRawProbs(unittest.TestCase):

    def test_empty_input_returns_identity(self):
        cal = fit_platt_from_raw_probs([], [])
        self.assertAlmostEqual(cal.a, 1.0)
        self.assertAlmostEqual(cal.b, 0.0)

    def test_all_same_label_returns_identity(self):
        cal = fit_platt_from_raw_probs([1, 1, 1], [0.7, 0.6, 0.8])
        self.assertAlmostEqual(cal.a, 1.0)
        self.assertAlmostEqual(cal.b, 0.0)

    def test_returns_platt_calibrator(self):
        cal = fit_platt_from_raw_probs([1, 0, 1, 0], [0.7, 0.3, 0.8, 0.2])
        self.assertIsInstance(cal, PlattCalibrator)

    def test_overconfident_model_shrinks(self):
        # All predict 0.9, but actual win rate is only 45%: calibrator must push output down.
        # Same logit for all points → no separation, optimizer converges to match mean rate.
        y = [1] * 45 + [0] * 55
        p = [0.9] * 100
        cal = fit_platt_from_raw_probs(y, p)
        calibrated = cal.predict_proba(0.9)
        self.assertLess(calibrated, 0.9)

    def test_underconfident_model_expands(self):
        # All predict 0.5, but actual win rate is 80%: calibrator must push output up.
        y = [1] * 80 + [0] * 20
        p = [0.5] * 100
        cal = fit_platt_from_raw_probs(y, p)
        calibrated = cal.predict_proba(0.5)
        self.assertGreater(calibrated, 0.5)

    def test_calibrated_output_is_finite(self):
        # Mixed rates at same p → no perfect separation → finite, non-extreme a.
        y = [1, 0, 1, 0, 1, 0] * 5
        p = [0.6] * 30
        cal = fit_platt_from_raw_probs(y, p)
        for prob in (0.3, 0.5, 0.7):
            self.assertTrue(math.isfinite(cal.predict_proba(prob)))

    def test_output_in_unit_interval(self):
        # Use fitted calibrator from non-separable data (uniform p, mixed labels).
        y = [1, 0] * 25
        p = [0.6] * 50
        cal = fit_platt_from_raw_probs(y, p)
        for prob in (0.2, 0.5, 0.8):
            result = cal.predict_proba(prob)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)


if __name__ == "__main__":
    unittest.main()
