"""Tests for mlcore/fit_series_core.py — series probability helper."""
import math
import unittest

from mlcore.fit_series_core import _series_prob_with_gamma


class TestSeriesProbWithGamma(unittest.TestCase):

    def test_returns_float(self):
        result = _series_prob_with_gamma(0.6, 3, 1.0)
        self.assertIsInstance(result, float)

    def test_result_strictly_between_0_and_1(self):
        result = _series_prob_with_gamma(0.6, 3, 1.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_fifty_fifty_stays_near_half(self):
        result = _series_prob_with_gamma(0.5, 3, 1.0)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_higher_p_map_gives_higher_series_prob(self):
        r1 = _series_prob_with_gamma(0.6, 3, 1.0)
        r2 = _series_prob_with_gamma(0.4, 3, 1.0)
        self.assertGreater(r1, r2)

    def test_gamma_gt_1_amplifies_advantage(self):
        r_base = _series_prob_with_gamma(0.6, 3, 1.0)
        r_amp = _series_prob_with_gamma(0.6, 3, 2.0)
        self.assertGreater(r_amp, r_base)

    def test_gamma_lt_1_shrinks_advantage(self):
        r_base = _series_prob_with_gamma(0.6, 3, 1.0)
        r_shrunk = _series_prob_with_gamma(0.6, 3, 0.5)
        self.assertLess(r_shrunk, r_base)

    def test_bo5_higher_series_prob_than_bo3_for_favorite(self):
        r3 = _series_prob_with_gamma(0.6, 3, 1.0)
        r5 = _series_prob_with_gamma(0.6, 5, 1.0)
        self.assertGreater(r5, r3)

    def test_result_is_finite(self):
        result = _series_prob_with_gamma(0.7, 5, 1.0)
        self.assertTrue(math.isfinite(result))

    def test_extreme_p_map_clipped(self):
        result = _series_prob_with_gamma(0.999, 3, 1.0)
        self.assertLess(result, 1.0)
        self.assertGreater(result, 0.9)

    def test_low_p_map_clipped(self):
        result = _series_prob_with_gamma(0.001, 3, 1.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.1)

    def test_very_low_gamma_clamped_to_min(self):
        # gamma is clamped to max(0.25, ...) so very low values stay bounded
        result = _series_prob_with_gamma(0.6, 3, 0.01)
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_symmetry_at_half(self):
        # p(0.5) should stay ≈0.5 regardless of gamma
        result = _series_prob_with_gamma(0.5, 5, 2.0)
        self.assertAlmostEqual(result, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
