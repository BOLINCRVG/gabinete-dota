"""Tests for market_v3_analysis_core — pure-data analytics for the v3 panel."""
from __future__ import annotations

import math
import unittest

import pandas as pd

from app_core.market_v3_analysis_core import (
    HISTORIC_WINDOWS,
    build_alerts,
    build_opponent_profile,
    build_recent_games,
    compute_h2h_stats,
    compute_map_stats,
    compute_window_stats,
)


def _league_fixture(n_teams: int = 4, n_games: int = 30) -> pd.DataFrame:
    """Each team plays the others in round-robin; kills correlate with team idx."""
    teams = [f"Team{i}" for i in range(n_teams)]
    rows = []
    for ti, team in enumerate(teams):
        for gi in range(n_games):
            opp_idx = (ti + 1 + (gi % (n_teams - 1))) % n_teams
            opp = teams[opp_idx]
            map_n = (gi % 3) + 1   # cycle 1/2/3
            base = 20.0 + 1.5 * ti + 2.0 * map_n + (1.0 if gi >= 20 else 0.0)
            rows.append({
                "team": team, "opponent": opp,
                "league": "TST", "year": 2026, "split": "Spring",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=gi + ti),
                "game": map_n,
                "kills_for": base,
            })
    return pd.DataFrame(rows)


class TestWindowStats(unittest.TestCase):

    def test_basic_windows_returned(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(team_df, metric_col="kills_for", line=27.5)
        self.assertEqual(len(stats), len(HISTORIC_WINDOWS))
        labels = [s.label for s in stats]
        self.assertEqual(labels, [w[0] for w in HISTORIC_WINDOWS])

    def test_hit_counts_consistent(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(team_df, metric_col="kills_for", line=23.0)
        for ws in stats:
            if ws.n > 0:
                self.assertEqual(ws.hits_over + ws.hits_under, ws.n)
                self.assertAlmostEqual(ws.over_pct + ws.under_pct, 100.0, places=4)

    def test_empty_team_handled(self):
        stats = compute_window_stats(pd.DataFrame(), metric_col="kills_for", line=20.0)
        for ws in stats:
            self.assertEqual(ws.n, 0)


class TestMapStats(unittest.TestCase):

    def test_map_breakdown_present(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        maps = compute_map_stats(team_df, metric_col="kills_for", line=25.0, overall_mean=27.0)
        # Only maps 1/2/3 in fixture
        self.assertEqual(sorted(m.map_n for m in maps), [1, 2, 3])
        for m in maps:
            self.assertGreater(m.games, 0)
            self.assertTrue(math.isfinite(m.mean))

    def test_note_shows_deviation(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        maps = compute_map_stats(team_df, metric_col="kills_for", line=25.0, overall_mean=27.0)
        any_note = any(("vs geral" in m.note) or ("vs média" in m.note)
                        for m in maps if m.note)
        self.assertTrue(any_note)


class TestH2H(unittest.TestCase):

    def test_h2h_filters_correctly(self):
        league = _league_fixture()
        team_a_df = league[league["team"] == "Team0"]
        h2h = compute_h2h_stats(team_a_df, "Team1", metric_col="kills_for", line=25.0)
        self.assertGreaterEqual(h2h.n, 1)
        self.assertEqual(h2h.label, "H2H")


class TestOpponentProfile(unittest.TestCase):

    def test_profile_has_one_row_per_team(self):
        league = _league_fixture()
        profile = build_opponent_profile(league, "Team0",
                                          metric_col="kills_for", line=25.0)
        self.assertEqual(len(profile), 4)
        # Sorted by mean descending
        means = [r.mean_vs_league for r in profile]
        self.assertEqual(means, sorted(means, reverse=True))

    def test_strength_tags_present(self):
        league = _league_fixture()
        profile = build_opponent_profile(league, "Team0",
                                          metric_col="kills_for", line=25.0)
        tags = {r.strength_tag for r in profile}
        self.assertTrue(tags.issubset({"very_under", "under", "avg", "over", "very_over"}))


class TestRecentGames(unittest.TestCase):

    def test_last_n_returned(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        recent = build_recent_games(team_df, league, metric_col="kills_for", line=25.0, last_n=5)
        self.assertEqual(len(recent), 5)
        for g in recent:
            self.assertIn(g.opponent_strength,
                          {"very_under", "under", "avg", "over", "very_over"})

    def test_hit_field_consistent_with_value(self):
        league = _league_fixture()
        team_df = league[league["team"] == "Team0"]
        recent = build_recent_games(team_df, league, metric_col="kills_for", line=25.0, last_n=8)
        for g in recent:
            self.assertEqual(g.hit, g.value > 25.0)


class TestAlerts(unittest.TestCase):

    def test_strong_over_signal_emits_alert(self):
        # Build a team with 5/5 over the line
        from app_core.market_v3_analysis_core import WindowStats, MapStats, RecentGameRow
        ws = WindowStats(
            label="Últimos 5", n=5, hits_over=5, hits_under=0,
            over_pct=100.0, under_pct=0.0, mean=32.0, trend="up", trend_value=0.5,
        )
        h2h = WindowStats(label="H2H", n=0, hits_over=0, hits_under=0,
                          over_pct=float("nan"), under_pct=float("nan"),
                          mean=float("nan"), trend="na", trend_value=float("nan"))
        alerts = build_alerts(team_name="Gen.G", window_stats=[ws], map_stats=[],
                              h2h=h2h, recent=[], line=27.5)
        self.assertTrue(any("sinal forte" in a["text"] for a in alerts))

    def test_streak_alert(self):
        from app_core.market_v3_analysis_core import RecentGameRow, WindowStats
        recent = [
            RecentGameRow(date="2026-05-15", opponent="X", game_map=1, value=30.0,
                          delta_from_line=2.5, hit=True, opponent_strength="avg")
            for _ in range(4)
        ]
        h2h = WindowStats(label="H2H", n=0, hits_over=0, hits_under=0,
                          over_pct=float("nan"), under_pct=float("nan"),
                          mean=float("nan"), trend="na", trend_value=float("nan"))
        alerts = build_alerts(team_name="T1", window_stats=[], map_stats=[],
                              h2h=h2h, recent=recent, line=27.5)
        self.assertTrue(any("seguidos" in a["text"] for a in alerts))


if __name__ == "__main__":
    unittest.main()
