import math
import unittest

from app_core.winner_pricing_core import compute_winner_pricing_bundle
from mlcore.bo import prob_win_series


def _odd_from_prob(p: float):
    pp = float(p)
    if not math.isfinite(pp) or pp <= 0.0 or pp >= 1.0:
        return None
    return 1.0 / pp


class WinnerPricingCoreTest(unittest.TestCase):
    def test_bundle_uses_given_series_prob(self):
        out = compute_winner_pricing_bundle(
            p_team_a_map=0.60,
            p_team_a_series=0.70,
            bo_for_series=5,
            prob_win_series_fn=prob_win_series,
            odd_from_prob_fn=_odd_from_prob,
        )
        self.assertAlmostEqual(float(out["pA_map"]), 0.60, places=12)
        self.assertAlmostEqual(float(out["pB_map"]), 0.40, places=12)
        self.assertAlmostEqual(float(out["pA_series"]), 0.70, places=12)
        self.assertAlmostEqual(float(out["pB_series"]), 0.30, places=12)
        self.assertAlmostEqual(float(out["oddA_map"]), 1.0 / 0.60, places=12)
        self.assertAlmostEqual(float(out["oddB_map"]), 1.0 / 0.40, places=12)
        self.assertAlmostEqual(float(out["oddA_series"]), 1.0 / 0.70, places=12)
        self.assertAlmostEqual(float(out["oddB_series"]), 1.0 / 0.30, places=12)
        self.assertTrue(math.isfinite(float(out["pA_bo3"])))
        self.assertTrue(math.isfinite(float(out["pA_bo5"])))

    def test_bundle_falls_back_series_from_map(self):
        out = compute_winner_pricing_bundle(
            p_team_a_map=0.57,
            p_team_a_series=float("nan"),
            bo_for_series=5,
            prob_win_series_fn=prob_win_series,
            odd_from_prob_fn=_odd_from_prob,
        )
        expected = float(prob_win_series(0.57, 5))
        self.assertAlmostEqual(float(out["pA_series"]), expected, places=12)
        self.assertAlmostEqual(float(out["pB_series"]), 1.0 - expected, places=12)
        self.assertAlmostEqual(float(out["oddA_series"]), 1.0 / expected, places=12)

    def test_invalid_map_keeps_outputs_non_crashing(self):
        out = compute_winner_pricing_bundle(
            p_team_a_map="bad",
            p_team_a_series=float("nan"),
            bo_for_series=3,
            prob_win_series_fn=prob_win_series,
            odd_from_prob_fn=_odd_from_prob,
        )
        self.assertTrue(math.isnan(float(out["pA_map"])))
        self.assertTrue(math.isnan(float(out["pA_series"])))
        self.assertIsNone(out["oddA_map"])
        self.assertIsNone(out["oddA_series"])
        self.assertTrue(math.isnan(float(out["pA_bo3"])))
        self.assertTrue(math.isnan(float(out["pA_bo5"])))


if __name__ == "__main__":
    unittest.main()
