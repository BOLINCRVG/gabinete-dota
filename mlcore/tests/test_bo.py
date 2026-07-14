"""Tests for mlcore/bo.py — BO series win probability."""
import math
import unittest

from mlcore.bo import p_series_from_pmap, prob_win_series


class TestPSeriesFromPmap(unittest.TestCase):

    # BO1: trivially returns p
    def test_bo1_returns_p_unchanged(self):
        self.assertAlmostEqual(p_series_from_pmap(0.65, bo=1), 0.65, places=8)

    def test_bo1_half(self):
        self.assertAlmostEqual(p_series_from_pmap(0.5, bo=1), 0.5, places=8)

    # BO3: P(win 2 of 3) = 3p²(1-p) + p³ = p²(3-2p)
    def test_bo3_half_gives_half(self):
        result = p_series_from_pmap(0.5, bo=3)
        self.assertAlmostEqual(result, 0.5, places=8)

    def test_bo3_favorite_wins_more_often(self):
        result = p_series_from_pmap(0.7, bo=3)
        # p=0.7: 3*0.49*0.3 + 0.343 = 0.441 + 0.343 = 0.784
        self.assertAlmostEqual(result, 0.784, places=5)

    def test_bo3_greater_than_pmap_for_p_gt_half(self):
        p = 0.6
        self.assertGreater(p_series_from_pmap(p, bo=3), p)

    def test_bo3_less_than_pmap_for_p_lt_half(self):
        p = 0.4
        self.assertLess(p_series_from_pmap(p, bo=3), p)

    # BO5
    def test_bo5_half_gives_half(self):
        result = p_series_from_pmap(0.5, bo=5)
        self.assertAlmostEqual(result, 0.5, places=8)

    def test_bo5_greater_than_bo3_for_p_gt_half(self):
        p = 0.65
        self.assertGreater(p_series_from_pmap(p, bo=5), p_series_from_pmap(p, bo=3))

    def test_bo5_less_than_bo3_for_p_lt_half(self):
        p = 0.35
        self.assertLess(p_series_from_pmap(p, bo=5), p_series_from_pmap(p, bo=3))

    # Symmetry: p(p) + p(1-p) = 1 for any valid BO
    def test_bo3_symmetry(self):
        p = 0.65
        self.assertAlmostEqual(p_series_from_pmap(p, bo=3) + p_series_from_pmap(1.0 - p, bo=3), 1.0, places=8)

    def test_bo5_symmetry(self):
        p = 0.72
        self.assertAlmostEqual(p_series_from_pmap(p, bo=5) + p_series_from_pmap(1.0 - p, bo=5), 1.0, places=8)

    # Monotonicity
    def test_monotone_in_p(self):
        results = [p_series_from_pmap(p, bo=3) for p in (0.3, 0.4, 0.5, 0.6, 0.7)]
        self.assertEqual(results, sorted(results))

    # Result in (0, 1)
    def test_result_in_unit_interval(self):
        for bo in (1, 3, 5):
            for p in (0.1, 0.5, 0.9):
                r = p_series_from_pmap(p, bo=bo)
                self.assertGreater(r, 0.0)
                self.assertLess(r, 1.0)

    # Error cases
    def test_even_bo_raises(self):
        with self.assertRaises(ValueError):
            p_series_from_pmap(0.5, bo=2)

    def test_zero_bo_raises(self):
        with self.assertRaises(ValueError):
            p_series_from_pmap(0.5, bo=0)

    def test_negative_bo_raises(self):
        with self.assertRaises(ValueError):
            p_series_from_pmap(0.5, bo=-1)

    # prob_win_series alias
    def test_alias_matches(self):
        for bo in (1, 3, 5):
            for p in (0.4, 0.5, 0.7):
                self.assertAlmostEqual(prob_win_series(p, bo), p_series_from_pmap(p, bo), places=12)


if __name__ == "__main__":
    unittest.main()
