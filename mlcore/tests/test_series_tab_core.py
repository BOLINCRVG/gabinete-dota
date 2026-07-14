import unittest

from app_core.series_tab_core import score_probs


class TestScoreProbs(unittest.TestCase):

    def test_bo3_sums_to_one(self):
        probs = score_probs(0.6, 3)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_bo3_a_2_0(self):
        probs = score_probs(0.6, 3)
        self.assertAlmostEqual(probs["A_2-0"], 0.36, places=6)

    def test_bo3_symmetric_at_half(self):
        probs = score_probs(0.5, 3)
        self.assertAlmostEqual(probs["A_2-0"], probs["B_2-0"], places=6)
        self.assertAlmostEqual(probs["A_2-1"], probs["B_2-1"], places=6)

    def test_bo3_p_one_all_a_wins(self):
        probs = score_probs(1.0, 3)
        self.assertAlmostEqual(probs["A_2-0"], 1.0, places=6)
        self.assertAlmostEqual(probs["A_2-1"], 0.0, places=6)

    def test_bo3_p_zero_all_b_wins(self):
        probs = score_probs(0.0, 3)
        self.assertAlmostEqual(probs["B_2-0"], 1.0, places=6)

    def test_bo3_has_four_keys(self):
        probs = score_probs(0.6, 3)
        self.assertEqual(len(probs), 4)

    def test_bo5_sums_to_one(self):
        probs = score_probs(0.5, 5)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_bo5_a_3_0(self):
        probs = score_probs(0.5, 5)
        self.assertAlmostEqual(probs["A_3-0"], 0.125, places=6)

    def test_bo5_symmetric_at_half(self):
        probs = score_probs(0.5, 5)
        self.assertAlmostEqual(probs["A_3-0"], probs["B_3-0"], places=6)
        self.assertAlmostEqual(probs["A_3-1"], probs["B_3-1"], places=6)
        self.assertAlmostEqual(probs["A_3-2"], probs["B_3-2"], places=6)

    def test_bo5_has_six_keys(self):
        probs = score_probs(0.7, 5)
        self.assertEqual(len(probs), 6)

    def test_unknown_bo_returns_empty(self):
        probs = score_probs(0.6, 1)
        self.assertEqual(probs, {})

    def test_returns_dict(self):
        self.assertIsInstance(score_probs(0.5, 3), dict)


if __name__ == "__main__":
    unittest.main()
