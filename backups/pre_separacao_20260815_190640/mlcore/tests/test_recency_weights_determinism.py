"""Regression test for the recency_weights determinism bug.

Bug (2026-05-30): when ref_date=None, recency_weights used
pd.Timestamp.utcnow() as the reference, making weights drift across
reruns. Two consecutive calls with the same data returned slightly
different weights → slightly different MatchupTotals → cache miss in
the simulation → different odds shown to the user when toggling
between "Ano todo" / "Últimos 5" / "Ano todo".

Fix: when ref_date=None, derive it from dates.max() (deterministic).
"""
from __future__ import annotations

import time
import unittest

import numpy as np
import pandas as pd

from mlcore.stats_teams import recency_weights


class TestRecencyWeightsDeterminism(unittest.TestCase):

    def _sample_dates(self) -> pd.Series:
        return pd.to_datetime(pd.Series([
            "2024-01-15", "2024-06-01", "2024-12-31",
            "2025-03-15", "2025-09-01", "2025-12-25",
            "2026-02-10", "2026-05-15",
        ]))

    def test_no_ref_date_two_calls_identical(self):
        """Bug regression: two calls with ref_date=None must produce equal weights."""
        dates = self._sample_dates()
        w1 = recency_weights(dates, half_life_days=180.0, ref_date=None)
        # Force a small wall-clock delay to ensure utcnow() would differ.
        time.sleep(0.05)
        w2 = recency_weights(dates, half_life_days=180.0, ref_date=None)
        np.testing.assert_array_equal(w1, w2)

    def test_explicit_ref_date_still_works(self):
        dates = self._sample_dates()
        ref = pd.Timestamp("2026-05-15")
        w = recency_weights(dates, half_life_days=180.0, ref_date=ref)
        self.assertEqual(len(w), len(dates))
        # Last game (same as ref) has weight = 1
        self.assertAlmostEqual(w[-1], 1.0, places=6)

    def test_default_uses_max_date(self):
        """Default ref_date should be the max date in the input."""
        dates = pd.to_datetime(pd.Series(["2026-01-01", "2026-05-15"]))
        w_implicit = recency_weights(dates, half_life_days=180.0, ref_date=None)
        w_explicit = recency_weights(dates, half_life_days=180.0,
                                       ref_date=pd.Timestamp("2026-05-15"))
        np.testing.assert_array_almost_equal(w_implicit, w_explicit, decimal=10)

    def test_empty_dates_handled(self):
        dates = pd.Series([], dtype="object")
        w = recency_weights(dates, half_life_days=180.0, ref_date=None)
        self.assertEqual(len(w), 0)

    def test_all_nan_dates_handled(self):
        """If all dates are NaT, fall back to today midnight (deterministic in same UTC day)."""
        dates = pd.Series([pd.NaT, pd.NaT])
        # Should not raise
        w = recency_weights(dates, half_life_days=180.0, ref_date=None)
        self.assertEqual(len(w), 2)
        self.assertTrue(all(np.isfinite(w)))

    def test_old_games_get_smaller_weight(self):
        dates = pd.to_datetime(pd.Series(["2024-01-01", "2026-05-15"]))
        w = recency_weights(dates, half_life_days=180.0, ref_date=None)
        # Old game should have much smaller weight
        self.assertLess(w[0], w[1])
        self.assertLess(w[0], 0.1)  # 2+ years old, half-life 180d → very small


class TestBuildProfileDeterminism(unittest.TestCase):
    """End-to-end: build_profile should be deterministic across calls."""

    def test_build_profile_deterministic(self):
        from mlcore.lines_engine import Filters, build_profile
        # Build a small team_games DataFrame
        df = pd.DataFrame({
            "team": ["T1"] * 6,
            "opponent": ["X"] * 6,
            "year": [2026] * 6,
            "split": ["Spring"] * 6,
            "playoffs": [0] * 6,
            "league": ["LCK"] * 6,
            "date": pd.to_datetime([
                "2026-01-01", "2026-02-01", "2026-03-01",
                "2026-04-01", "2026-04-15", "2026-05-01",
            ]),
            "side": ["Blue", "Red", "Blue", "Red", "Blue", "Red"],
            "kills_for": [12, 10, 15, 8, 14, 11],
            "kills_against": [10, 12, 9, 13, 11, 10],
            "towers_for": [9, 8, 10, 7, 9, 9],
            "towers_against": [8, 9, 7, 10, 8, 8],
            "dragons_for": [3, 2, 4, 2, 3, 3],
            "dragons_against": [2, 3, 2, 4, 2, 3],
            "elementaldrakes_for": [3, 2, 4, 2, 3, 3],
            "elementaldrakes_against": [2, 3, 2, 4, 2, 3],
            "barons_for": [1, 0, 1, 0, 1, 1],
            "barons_against": [0, 1, 0, 1, 0, 1],
            "inhibitors_for": [1, 0, 2, 0, 1, 1],
            "inhibitors_against": [0, 1, 0, 2, 0, 0],
            "game_time_min": [30, 35, 28, 33, 31, 29],
            "result": [1, 0, 1, 0, 1, 0],
        })
        filters = Filters(year=2026, league=None, split="Spring", playoffs=False)
        p1 = build_profile(df, team="T1", filters=filters,
                            league_mode="all", map_mode="avg",
                            half_life_days=180.0, max_games=None)
        time.sleep(0.05)
        p2 = build_profile(df, team="T1", filters=filters,
                            league_mode="all", map_mode="avg",
                            half_life_days=180.0, max_games=None)
        # All numeric fields must match exactly
        self.assertEqual(p1.kills_for, p2.kills_for)
        self.assertEqual(p1.kills_against, p2.kills_against)
        self.assertEqual(p1.towers_for, p2.towers_for)
        self.assertEqual(p1.time_min, p2.time_min)


if __name__ == "__main__":
    unittest.main()
