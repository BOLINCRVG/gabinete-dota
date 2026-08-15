"""Tests for the contextual opponent profile (v3)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market_v3_analysis_core import (
    build_contextual_opponent_profile,
    classify_team_archetype,
    opposite_archetype_set,
)


def _league_with_archetypes() -> pd.DataFrame:
    """5 teams: 1 very_over (avg ~35), 1 over (~31), 1 neutral (~27),
    1 under (~23), 1 very_under (~19). Each plays the others ~6 times."""
    rows = []
    profiles = [
        ("VeryOver", 35.0),
        ("Over",     31.0),
        ("Neutral",  27.0),
        ("Under",    23.0),
        ("VeryUnder", 19.0),
    ]
    for ti, (team, base) in enumerate(profiles):
        for opp_i, (opp, _opp_base) in enumerate(profiles):
            if opp == team:
                continue
            for game_i in range(6):
                # Slight noise around base
                rows.append({
                    "team": team, "opponent": opp,
                    "league": "TST", "year": 2026, "split": "Spring",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=ti * 30 + opp_i * 6 + game_i),
                    "kills_for": base + (game_i - 3.0),
                })
    return pd.DataFrame(rows)


class TestArchetypeClassification(unittest.TestCase):

    def test_very_over_team_detected(self):
        league = _league_with_archetypes()
        team_df = league[league["team"] == "VeryOver"]
        arch = classify_team_archetype(team_name="VeryOver", team_df=team_df,
                                         league_scope_df=league, metric_col="kills_for")
        self.assertEqual(arch.archetype, "very_over")
        self.assertGreater(arch.diff_pct, 0.15)

    def test_neutral_team_detected(self):
        league = _league_with_archetypes()
        team_df = league[league["team"] == "Neutral"]
        arch = classify_team_archetype(team_name="Neutral", team_df=team_df,
                                         league_scope_df=league, metric_col="kills_for")
        self.assertEqual(arch.archetype, "neutral")

    def test_very_under_team_detected(self):
        league = _league_with_archetypes()
        team_df = league[league["team"] == "VeryUnder"]
        arch = classify_team_archetype(team_name="VeryUnder", team_df=team_df,
                                         league_scope_df=league, metric_col="kills_for")
        self.assertEqual(arch.archetype, "very_under")
        self.assertLess(arch.diff_pct, -0.15)


class TestOppositeArchetypeSet(unittest.TestCase):

    def test_over_returns_unders(self):
        self.assertEqual(set(opposite_archetype_set("over")), {"under", "very_under"})

    def test_very_under_returns_overs(self):
        self.assertEqual(set(opposite_archetype_set("very_under")), {"over", "very_over"})

    def test_neutral_returns_both_extremes(self):
        result = set(opposite_archetype_set("neutral"))
        self.assertEqual(result, {"over", "very_over", "under", "very_under"})


class TestContextualProfile(unittest.TestCase):

    def test_focal_team_vs_opposite_archetype(self):
        league = _league_with_archetypes()
        team_df = league[league["team"] == "VeryOver"]
        prof = build_contextual_opponent_profile(
            focal_team="VeryOver", focal_team_df=team_df,
            league_scope_df=league,
            target_archetypes=("under", "very_under"),
            metric_col="kills_for", line=30.0,
        )
        # Should include opponents from Under and VeryUnder
        self.assertGreater(prof.n_opponents, 0)
        self.assertIn("Under", prof.opponents)
        self.assertIn("VeryUnder", prof.opponents)
        # VeryOver vs unders: VeryOver still averages ~35, way above 30 → mostly OVER
        if prof.games_played > 0:
            self.assertGreaterEqual(prof.over_pct + prof.under_pct, 99.9)
            self.assertGreater(prof.over_pct, 50.0)

    def test_no_matching_opponents(self):
        league = _league_with_archetypes()
        team_df = league[league["team"] == "Neutral"]
        # Ask for an archetype that doesn't exist (made up)
        prof = build_contextual_opponent_profile(
            focal_team="Neutral", focal_team_df=team_df,
            league_scope_df=league,
            target_archetypes=("doesnotexist",),
            metric_col="kills_for", line=27.0,
        )
        self.assertEqual(prof.n_opponents, 0)
        self.assertEqual(prof.games_played, 0)


if __name__ == "__main__":
    unittest.main()
