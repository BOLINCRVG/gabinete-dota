import math
import unittest

import pandas as pd

from app_core.odds_core import (
    edge_pp_from_odds,
    format_scope_odds_clean,
    laplace_odds,
    laplace_pick_odd,
    odd_mix,
    odd_to_prob,
    prob_to_odd,
    recency_slices,
)


class TestLaplaceOdds(unittest.TestCase):

    def test_floor_applied(self):
        self.assertEqual(laplace_odds(10, 10, floor_odd=1.10), 1.10)

    def test_zero_n_returns_none(self):
        self.assertIsNone(laplace_odds(0, 0, floor_odd=1.10))

    def test_zero_k_returns_formula(self):
        odd = laplace_odds(0, 10, floor_odd=1.10)
        self.assertAlmostEqual(odd, (10.0 + 2.0) / (0.0 + 1.0), places=6)

    def test_high_k_n_clamped_to_floor(self):
        odd = laplace_odds(99, 100, floor_odd=1.10)
        self.assertAlmostEqual(odd, 1.10, places=6)

    def test_returns_float(self):
        self.assertIsInstance(laplace_odds(5, 10), float)


class TestOddToProb(unittest.TestCase):

    def test_two_gives_half(self):
        p = odd_to_prob(2.0)
        self.assertAlmostEqual(p, 0.5, places=3)

    def test_invalid_returns_none(self):
        self.assertIsNone(odd_to_prob("abc"))

    def test_nan_returns_none(self):
        result = odd_to_prob(float("nan"))
        self.assertIsNone(result)


class TestProbToOdd(unittest.TestCase):

    def test_half_gives_two(self):
        odd = prob_to_odd(0.5)
        self.assertAlmostEqual(odd, 2.0, places=3)

    def test_nan_returns_none(self):
        self.assertIsNone(prob_to_odd(float("nan")))

    def test_returns_float_or_none(self):
        result = prob_to_odd(0.6)
        self.assertIsInstance(result, float)


class TestOddMix(unittest.TestCase):

    def test_all_sources_equal_returns_same(self):
        odd = odd_mix(2.0, 2.0, 50.0, w_model=1.0, w_hist=1.0, w_stats=1.0)
        self.assertAlmostEqual(odd, 2.0, places=3)

    def test_no_sources_returns_none(self):
        result = odd_mix(None, None, None, w_model=1.0, w_hist=1.0, w_stats=1.0)
        self.assertIsNone(result)

    def test_returns_float_or_none(self):
        result = odd_mix(2.0, 2.5, None, w_model=1.0, w_hist=1.0, w_stats=0.0)
        self.assertIsInstance(result, float)


class TestEdgePpFromOdds(unittest.TestCase):

    def test_no_diff_returns_zero(self):
        edge = edge_pp_from_odds(2.0, 2.0, 2.0)
        self.assertAlmostEqual(edge, 0.0, places=6)

    def test_none_mix_returns_none(self):
        self.assertIsNone(edge_pp_from_odds(None, 2.0, 2.0))

    def test_no_baseline_returns_none(self):
        self.assertIsNone(edge_pp_from_odds(2.0, None, None))

    def test_returns_float_or_none(self):
        result = edge_pp_from_odds(2.0, 2.5, 2.0)
        self.assertIsInstance(result, float)


class TestLaplacePickOdd(unittest.TestCase):

    def test_liga_15_key(self):
        odds = {"Liga_15_over": 1.80}
        result = laplace_pick_odd(odds, "over", "Liga 15")
        self.assertAlmostEqual(result, 1.80, places=6)

    def test_h2h_key(self):
        odds = {"H2H_10_under": 2.10}
        result = laplace_pick_odd(odds, "under", "H2H 10")
        self.assertAlmostEqual(result, 2.10, places=6)

    def test_missing_key_returns_none(self):
        result = laplace_pick_odd({}, "over", "Liga 15")
        self.assertIsNone(result)


class TestAppOddsCore(unittest.TestCase):
    def test_laplace_odds_floor(self):
        self.assertEqual(laplace_odds(10, 10, floor_odd=1.10), 1.10)
        self.assertIsNone(laplace_odds(0, 0, floor_odd=1.10))

    def test_recency_slices_shapes_and_dedup(self):
        df_a = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01"},
                {"gameid": "g2", "date": "2026-01-02"},
                {"gameid": "g3", "date": "2026-01-03"},
            ]
        )
        df_b = pd.DataFrame(
            [
                {"gameid": "g3", "date": "2026-01-03"},  # duplicado entre times
                {"gameid": "g4", "date": "2026-01-04"},
            ]
        )
        df_h2h = pd.DataFrame(
            [
                {"gameid": "h1", "date": "2026-01-05"},
                {"gameid": "h2", "date": "2026-01-06"},
            ]
        )
        s = recency_slices(df_a=df_a, df_b=df_b, df_h2h=df_h2h, id_col="gameid", date_col="date")
        self.assertEqual(int(s["league_year"].shape[0]), 4)  # g1,g2,g3,g4
        self.assertEqual(int(s["h2h_year"].shape[0]), 2)
        self.assertTrue(set(["league_5", "league_10", "league_15", "h2h_5", "h2h_10", "h2h_15"]).issubset(set(s.keys())))

    def test_recency_slices_can_backfill_unique_targets(self):
        df_a = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01"},
                {"gameid": "g2", "date": "2026-01-02"},
                {"gameid": "g3", "date": "2026-01-03"},
                {"gameid": "g4", "date": "2026-01-04"},
                {"gameid": "g5", "date": "2026-01-05"},
            ]
        )
        df_b = pd.DataFrame(
            [
                {"gameid": "g0", "date": "2025-12-31"},
                {"gameid": "g5", "date": "2026-01-05"},
                {"gameid": "g6", "date": "2026-01-06"},
                {"gameid": "g7", "date": "2026-01-07"},
                {"gameid": "g8", "date": "2026-01-08"},
                {"gameid": "g9", "date": "2026-01-09"},
            ]
        )
        df_h2h = pd.DataFrame([{"gameid": "g5", "date": "2026-01-05"}])
        s = recency_slices(
            df_a=df_a,
            df_b=df_b,
            df_h2h=df_h2h,
            id_col="gameid",
            date_col="date",
            dedupe_league=True,
            backfill_unique_targets=True,
        )
        self.assertEqual(int(s["league_5"].shape[0]), 10)
        self.assertEqual(int(s["league_year"].shape[0]), 10)
        self.assertEqual(int(s["h2h_year"].shape[0]), 1)


class TestFormatScopeOddsClean(unittest.TestCase):
    """Guards the reported bugs: no 'pick=' jargon, and H2H never shows Liga's odd."""

    def test_shows_scope_odds_within_limit(self):
        od = {"Liga_Ano_Over": 1.55, "Liga_15_Over": 1.70, "Liga_10_Over": 2.40}
        txt = format_scope_odds_clean(od, "Liga", "Over", 2.0)
        self.assertIn("méd", txt)
        self.assertIn("Ano 1.55", txt)
        self.assertIn("15 1.70", txt)
        self.assertNotIn("10", txt)          # 2.40 > 2.0 → excluded
        self.assertNotIn("pick=", txt)       # no jargon

    def test_h2h_does_not_leak_liga_odd(self):
        # The reported bug: H2H line was showing Liga's pick odd. With scope
        # formatting, H2H with no odds returns the empty marker, never Liga's.
        od = {"Liga_Ano_Over": 1.20, "Liga_15_Over": 1.20,
              "H2H_Ano_Over": None, "H2H_15_Over": None}
        liga_txt = format_scope_odds_clean(od, "Liga", "Over", 2.0)
        h2h_txt = format_scope_odds_clean(od, "H2H", "Over", 2.0)
        self.assertIn("1.20", liga_txt)
        self.assertNotIn("1.20", h2h_txt)    # Liga's 1.20 must NOT appear on H2H
        self.assertIn("limite", h2h_txt)     # "nenhum recorte ≤ limite"

    def test_consistent_with_badges_when_over_limit(self):
        # If everything is over the limit, both the badges (passed=[]) and this
        # text agree there's nothing — no phantom odd shown.
        od = {"Liga_Ano_Over": 1.20, "Liga_15_Over": 1.25}
        txt = format_scope_odds_clean(od, "Liga", "Over", 1.10)  # limit cuts all
        self.assertIn("limite", txt)
        self.assertNotIn("1.20", txt)


if __name__ == "__main__":
    unittest.main()
