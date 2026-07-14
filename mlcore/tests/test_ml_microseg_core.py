"""Tests for app_core/ml_microseg_core.py — microseg helpers."""
import unittest

from app_core.ml_microseg_core import _estimate_shrink_k, _extract_fitted_k


# ---------------------------------------------------------------------------
# _estimate_shrink_k
# ---------------------------------------------------------------------------

class TestEstimateShrinkK(unittest.TestCase):

    def _agg(self, mus, ns):
        """Build agg_specific dict from lists of biases and sample sizes."""
        out = {}
        for i, (mu, n) in enumerate(zip(mus, ns)):
            out[f"seg_{i}"] = {"n": float(n), "sum_err": float(mu * n)}
        return out

    def test_fewer_than_min_segments_returns_kmax(self):
        # only 3 segments, min_segments=8 → k_max
        agg = self._agg([0.05, -0.03, 0.02], [50, 50, 50])
        result = _estimate_shrink_k(agg, k_max=300.0)
        self.assertAlmostEqual(result, 300.0, places=5)

    def test_segments_below_min_n_excluded(self):
        # 10 segments but all n<15 → excluded → k_max
        agg = self._agg([0.1] * 10, [5] * 10)
        result = _estimate_shrink_k(agg, min_n_per_segment=15.0, k_max=300.0)
        self.assertAlmostEqual(result, 300.0, places=5)

    def test_identical_biases_returns_kmax(self):
        # all same bias → sigma_true^2 = 0 → k_max
        agg = self._agg([0.05] * 10, [100] * 10)
        result = _estimate_shrink_k(agg, k_max=300.0)
        self.assertAlmostEqual(result, 300.0, places=5)

    def test_high_variance_biases_returns_k_min(self):
        # very high variance → sigma_true^2 large → ratio small → clamped to k_min
        agg = self._agg([-0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4], [500] * 10)
        result = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0)
        self.assertGreaterEqual(result, 25.0)
        self.assertLessEqual(result, 300.0)

    def test_result_within_bounds(self):
        agg = self._agg([0.01 * i for i in range(10)], [100] * 10)
        result = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0)
        self.assertGreaterEqual(result, 25.0)
        self.assertLessEqual(result, 300.0)

    def test_empty_agg_returns_kmax(self):
        result = _estimate_shrink_k({}, k_max=300.0)
        self.assertAlmostEqual(result, 300.0, places=5)

    def test_returns_float(self):
        agg = self._agg([0.01 * i for i in range(10)], [100] * 10)
        result = _estimate_shrink_k(agg)
        self.assertIsInstance(result, float)


# ---------------------------------------------------------------------------
# _extract_fitted_k
# ---------------------------------------------------------------------------

class TestExtractFittedK(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(_extract_fitted_k(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(_extract_fitted_k({}))

    def test_non_dict_returns_none(self):
        self.assertIsNone(_extract_fitted_k([1, 2, 3]))

    def test_returns_first_positive_k(self):
        table = {"seg_a": {"shrink_k": 90.0, "bias": 0.01}}
        result = _extract_fitted_k(table)
        self.assertAlmostEqual(result, 90.0, places=5)

    def test_skips_zero_k(self):
        table = {
            "seg_a": {"shrink_k": 0.0},
            "seg_b": {"shrink_k": 120.0},
        }
        result = _extract_fitted_k(table)
        self.assertAlmostEqual(result, 120.0, places=5)

    def test_skips_negative_k(self):
        table = {
            "seg_a": {"shrink_k": -5.0},
            "seg_b": {"shrink_k": 75.0},
        }
        result = _extract_fitted_k(table)
        self.assertAlmostEqual(result, 75.0, places=5)

    def test_all_invalid_k_returns_none(self):
        table = {"seg_a": {"shrink_k": 0.0}, "seg_b": {"shrink_k": -1.0}}
        self.assertIsNone(_extract_fitted_k(table))

    def test_non_dict_values_skipped(self):
        table = {"seg_a": "not_a_dict", "seg_b": {"shrink_k": 50.0}}
        result = _extract_fitted_k(table)
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_returns_float(self):
        table = {"seg_a": {"shrink_k": 150.0}}
        result = _extract_fitted_k(table)
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
