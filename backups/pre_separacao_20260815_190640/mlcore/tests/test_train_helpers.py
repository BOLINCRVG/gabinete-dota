import unittest

from mlcore.train_offline import _p_from_diff_points, _reliability_ece, _safe_int_year


class TestSafeIntYear(unittest.TestCase):

    def test_valid_string_year(self):
        self.assertEqual(_safe_int_year("2026"), 2026)

    def test_valid_int_year(self):
        self.assertEqual(_safe_int_year(2025), 2025)

    def test_zero_returns_none(self):
        self.assertIsNone(_safe_int_year(0))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(_safe_int_year("abc"))

    def test_none_returns_none(self):
        self.assertIsNone(_safe_int_year(None))

    def test_negative_returns_none(self):
        self.assertIsNone(_safe_int_year(-1))

    def test_float_string_truncates(self):
        result = _safe_int_year("2024.9")
        self.assertIsNone(result)

    def test_large_year_valid(self):
        self.assertEqual(_safe_int_year(9999), 9999)

    def test_returns_int_type(self):
        result = _safe_int_year("2025")
        self.assertIsInstance(result, int)


class TestReliabilityEce(unittest.TestCase):

    def test_basic_well_calibrated(self):
        y = [1, 1, 0, 0]
        p = [0.9, 0.8, 0.2, 0.1]
        out = _reliability_ece(y, p, n_bins=5)
        self.assertIn("ece", out)
        self.assertIn("bins", out)
        self.assertGreaterEqual(out["ece"], 0.0)

    def test_empty_returns_none_ece(self):
        out = _reliability_ece([], [], n_bins=10)
        self.assertIsNone(out["ece"])
        self.assertEqual(out["bins"], [])

    def test_mismatched_lengths_returns_none(self):
        out = _reliability_ece([1, 0], [0.5], n_bins=10)
        self.assertIsNone(out["ece"])

    def test_ece_between_0_and_1(self):
        y = [1, 1, 0, 0, 1]
        p = [0.8, 0.7, 0.3, 0.2, 0.6]
        out = _reliability_ece(y, p, n_bins=5)
        self.assertGreaterEqual(out["ece"], 0.0)
        self.assertLessEqual(out["ece"], 1.0)

    def test_bins_list_returned(self):
        y = [1, 0]
        p = [0.9, 0.1]
        out = _reliability_ece(y, p, n_bins=5)
        self.assertIsInstance(out["bins"], list)

    def test_returns_float_ece(self):
        y = [1, 0]
        p = [0.8, 0.2]
        out = _reliability_ece(y, p, n_bins=5)
        if out["ece"] is not None:
            self.assertIsInstance(out["ece"], float)


class TestPFromDiffPoints(unittest.TestCase):

    def test_zero_diff_gives_half(self):
        result = _p_from_diff_points(0.0)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_positive_diff_above_half(self):
        result = _p_from_diff_points(400.0)
        self.assertGreater(result, 0.5)

    def test_negative_diff_below_half(self):
        result = _p_from_diff_points(-400.0)
        self.assertLess(result, 0.5)

    def test_symmetric_around_half(self):
        p_pos = _p_from_diff_points(200.0)
        p_neg = _p_from_diff_points(-200.0)
        self.assertAlmostEqual(p_pos + p_neg, 1.0, places=5)

    def test_returns_float(self):
        self.assertIsInstance(_p_from_diff_points(0.0), float)

    def test_result_between_0_and_1(self):
        for diff in [-800.0, -400.0, 0.0, 400.0, 800.0]:
            result = _p_from_diff_points(diff)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_custom_scale(self):
        p_400 = _p_from_diff_points(400.0, scale_points=400.0)
        p_800 = _p_from_diff_points(400.0, scale_points=800.0)
        self.assertGreater(p_400, p_800)

    def test_zero_scale_falls_back(self):
        result = _p_from_diff_points(0.0, scale_points=0.0)
        self.assertAlmostEqual(result, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
