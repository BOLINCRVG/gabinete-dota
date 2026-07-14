"""Tests for app_core/odds_core.py — pure pricing helpers."""
import math
import unittest

from app_core.odds_core import (
    edge_pp_from_odds,
    laplace_odds,
    laplace_pick_odd,
    odd_mix,
    odd_to_prob,
    prob_to_odd,
)


# ---------------------------------------------------------------------------
# laplace_odds
# ---------------------------------------------------------------------------

class TestLaplaceOdds(unittest.TestCase):

    def test_basic_formula(self):
        # (n+2)/(k+1) = (10+2)/(5+1) = 12/6 = 2.0
        self.assertAlmostEqual(laplace_odds(5, 10), 2.0)

    def test_zero_n_returns_none(self):
        self.assertIsNone(laplace_odds(0, 0))

    def test_negative_n_returns_none(self):
        self.assertIsNone(laplace_odds(0, -1))

    def test_floor_odd_applied(self):
        # (n+2)/(k+1) = (2+2)/(10+1) ≈ 0.36, below default floor 1.10
        result = laplace_odds(10, 2)
        self.assertGreaterEqual(result, 1.10)

    def test_custom_floor(self):
        result = laplace_odds(10, 2, floor_odd=1.50)
        self.assertGreaterEqual(result, 1.50)

    def test_all_wins_gives_floor_odds(self):
        # k=n=100 → raw=(102/101)≈1.009 < floor 1.10 → clamped to floor
        result = laplace_odds(100, 100)
        self.assertAlmostEqual(result, 1.10, places=4)

    def test_zero_wins_gives_high_odds(self):
        # k=0 → (n+2)/1 = n+2
        result = laplace_odds(0, 10)
        self.assertAlmostEqual(result, 12.0)


# ---------------------------------------------------------------------------
# odd_to_prob / prob_to_odd
# ---------------------------------------------------------------------------

class TestOddToProb(unittest.TestCase):

    def test_odd_2_gives_half(self):
        p = odd_to_prob(2.0)
        self.assertAlmostEqual(p, 0.5, places=4)

    def test_odd_1_returns_none(self):
        # decimal_odds_to_prob treats odds <= 1.0 as invalid → None
        self.assertIsNone(odd_to_prob(1.0))

    def test_high_odd_gives_low_prob(self):
        p = odd_to_prob(10.0)
        self.assertLess(p, 0.15)

    def test_invalid_returns_none(self):
        self.assertIsNone(odd_to_prob("abc"))
        self.assertIsNone(odd_to_prob(None))

    def test_result_in_unit_interval(self):
        for odd in (1.5, 2.0, 3.0, 5.0, 10.0):
            p = odd_to_prob(odd)
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestProbToOdd(unittest.TestCase):

    def test_half_gives_2(self):
        odd = prob_to_odd(0.5)
        self.assertAlmostEqual(odd, 2.0, places=4)

    def test_high_prob_gives_low_odd(self):
        odd = prob_to_odd(0.90)
        self.assertLess(odd, 1.25)

    def test_low_prob_gives_high_odd(self):
        odd = prob_to_odd(0.10)
        self.assertGreater(odd, 8.0)

    def test_nan_returns_none(self):
        self.assertIsNone(prob_to_odd(float("nan")))

    def test_invalid_returns_none(self):
        self.assertIsNone(prob_to_odd("abc"))

    def test_round_trip(self):
        p = 0.65
        odd = prob_to_odd(p)
        p_back = odd_to_prob(odd)
        self.assertAlmostEqual(p_back, p, places=4)


# ---------------------------------------------------------------------------
# odd_mix — weighted probability blend
# ---------------------------------------------------------------------------

class TestOddMix(unittest.TestCase):

    def test_equal_weights_returns_midpoint_odd(self):
        # p_model=0.5 (odd=2.0), p_hist=0.5 (odd=2.0) → mix=0.5 → odd=2.0
        result = odd_mix(2.0, 2.0, None, w_model=1.0, w_hist=1.0, w_stats=0.0)
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_model_only(self):
        result = odd_mix(2.0, None, None, w_model=1.0, w_hist=0.0, w_stats=0.0)
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_hist_only(self):
        result = odd_mix(None, 2.0, None, w_model=0.0, w_hist=1.0, w_stats=0.0)
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_no_valid_inputs_returns_none(self):
        result = odd_mix(None, None, None, w_model=1.0, w_hist=1.0, w_stats=1.0)
        self.assertIsNone(result)

    def test_higher_model_weight_pulls_toward_model(self):
        # model odd=1.5 (p≈0.667), hist odd=3.0 (p≈0.333)
        # heavy model weight → result closer to 1.5
        res_model = odd_mix(1.5, 3.0, None, w_model=3.0, w_hist=1.0, w_stats=0.0)
        res_hist = odd_mix(1.5, 3.0, None, w_model=1.0, w_hist=3.0, w_stats=0.0)
        # p_model>p_hist → lower odd for model-heavy mix
        self.assertLess(res_model, res_hist)

    def test_stats_pct_as_third_source(self):
        # stats_pct=60 (60% → p=0.60)
        result = odd_mix(None, None, 60.0, w_model=0.0, w_hist=0.0, w_stats=1.0)
        p = odd_to_prob(result)
        self.assertAlmostEqual(p, 0.60, places=3)

    def test_model_boost_increases_model_weight(self):
        # model_boost=2.0 → double weight for model component
        res_no_boost = odd_mix(1.5, 3.0, None, w_model=1.0, w_hist=1.0, w_stats=0.0, model_boost=1.0)
        res_with_boost = odd_mix(1.5, 3.0, None, w_model=1.0, w_hist=1.0, w_stats=0.0, model_boost=2.0)
        # model is lower odds (higher p) → boost makes result lower odds
        self.assertLessEqual(res_with_boost, res_no_boost)

    def test_result_is_finite(self):
        result = odd_mix(2.0, 2.5, 55.0, w_model=0.4, w_hist=0.4, w_stats=0.2)
        self.assertIsNotNone(result)
        self.assertTrue(math.isfinite(result))


# ---------------------------------------------------------------------------
# edge_pp_from_odds
# ---------------------------------------------------------------------------

class TestEdgePpFromOdds(unittest.TestCase):

    def test_same_odds_zero_edge(self):
        edge = edge_pp_from_odds(2.0, 2.0, 2.0)
        self.assertAlmostEqual(edge, 0.0, places=3)

    def test_mix_higher_prob_positive_edge(self):
        # mix odd=1.5 (p≈0.667), hist odd=2.0 (p=0.5) → edge ≈ 16.7pp
        edge = edge_pp_from_odds(1.5, 2.0, None)
        self.assertGreater(edge, 0.0)

    def test_mix_lower_prob_negative_edge(self):
        # mix odd=3.0 (p≈0.333), hist odd=2.0 (p=0.5) → edge ≈ -16.7pp
        edge = edge_pp_from_odds(3.0, 2.0, None)
        self.assertLess(edge, 0.0)

    def test_no_mix_returns_none(self):
        self.assertIsNone(edge_pp_from_odds(None, 2.0, 2.0))

    def test_no_baseline_returns_none(self):
        # both hist and model are None
        self.assertIsNone(edge_pp_from_odds(2.0, None, None))

    def test_falls_back_to_model_when_hist_missing(self):
        # hist is None, model present → uses model as baseline
        edge = edge_pp_from_odds(1.5, None, 2.0)
        self.assertGreater(edge, 0.0)

    def test_result_in_reasonable_range(self):
        edge = edge_pp_from_odds(2.0, 2.5, 2.5)
        self.assertTrue(math.isfinite(edge))
        self.assertGreater(edge, -100.0)
        self.assertLess(edge, 100.0)


# ---------------------------------------------------------------------------
# laplace_pick_odd
# ---------------------------------------------------------------------------

class TestLaplacePickOdd(unittest.TestCase):

    def _odds_dict(self):
        return {
            "Liga_15_over": 1.85,
            "Liga_Ano_over": 2.10,
            "H2H_15_over": 1.90,
            "H2H_Ano_over": 2.20,
        }

    def test_liga_15(self):
        result = laplace_pick_odd(self._odds_dict(), "over", "Liga 15")
        self.assertAlmostEqual(result, 1.85, places=4)

    def test_liga_ano(self):
        result = laplace_pick_odd(self._odds_dict(), "over", "Liga Ano")
        self.assertAlmostEqual(result, 2.10, places=4)

    def test_h2h_15(self):
        result = laplace_pick_odd(self._odds_dict(), "over", "H2H 15")
        self.assertAlmostEqual(result, 1.90, places=4)

    def test_media_returns_avg_prob(self):
        result = laplace_pick_odd(self._odds_dict(), "over", "Média 15")
        p1 = odd_to_prob(1.85)
        p2 = odd_to_prob(1.90)
        expected = prob_to_odd(0.5 * (p1 + p2))
        self.assertAlmostEqual(result, expected, places=3)

    def test_missing_key_returns_none(self):
        result = laplace_pick_odd({}, "over", "Liga 15")
        self.assertIsNone(result)

    def test_default_pick_falls_back_to_liga_15(self):
        result = laplace_pick_odd(self._odds_dict(), "over", "Unknown Pick")
        # Unrecognized prefix → falls back to _get("Liga", "15")
        self.assertAlmostEqual(result, 1.85, places=4)


if __name__ == "__main__":
    unittest.main()
