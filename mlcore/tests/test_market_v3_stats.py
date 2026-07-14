"""Tests for build_kills_stats_table (v3 stats refeito)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market_v3_analysis_core import (
    StatsRow,
    build_kills_stats_table,
    _league_rank_for_team,
)


def _make_team_df(team: str, n: int, kf: float, ka: float, win_rate: float = 0.5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "team": team,
            "opponent": f"Opp{i % 3}",
            "win": 1 if (i / n) < win_rate else 0,
            "game_time_min": 32.0 + (i % 5),
            "kills_for": kf + (i % 4 - 2.0),
            "kills_against": ka + (i % 4 - 2.0),
            "mv2_total_kills": kf + ka + (i % 4 - 2.0) * 2,
            "firstblood": 1 if i % 2 == 0 else 0,
            "firsttower": 1 if i % 3 == 0 else 0,
            "firstdragon": 1 if i % 4 == 0 else 0,
        })
    return pd.DataFrame(rows)


class TestKillsStatsTable(unittest.TestCase):

    def test_returns_expected_rows(self):
        a = _make_team_df("A", n=20, kf=15.0, ka=12.0, win_rate=0.7)
        b = _make_team_df("B", n=18, kf=13.0, ka=14.0, win_rate=0.45)
        league = pd.concat([a, b], ignore_index=True)
        rows = build_kills_stats_table(
            team_a="A", team_b="B",
            df_a=a, df_b=b, league_scope_df=league,
            metric_col="mv2_total_kills", line=27.5,
        )
        labels = [r.label for r in rows]
        self.assertIn("Jogos no recorte", labels)
        self.assertIn("Win Rate", labels)
        self.assertIn("Kills feitas (/jogo)", labels)
        self.assertIn("Combined kills", labels)
        self.assertIn("First Blood %", labels)
        self.assertIn("Over 27.5 (combined)", labels)
        self.assertIn("Under 27.5 (combined)", labels)

    def test_games_count_correct(self):
        a = _make_team_df("A", n=12, kf=15.0, ka=12.0)
        b = _make_team_df("B", n=8, kf=13.0, ka=14.0)
        league = pd.concat([a, b], ignore_index=True)
        rows = build_kills_stats_table(
            team_a="A", team_b="B",
            df_a=a, df_b=b, league_scope_df=league,
            metric_col="mv2_total_kills", line=25.0,
        )
        games_row = next(r for r in rows if r.label == "Jogos no recorte")
        self.assertEqual(games_row.team_a_value, "12")
        self.assertEqual(games_row.team_b_value, "8")

    def test_over_under_sum_to_100(self):
        a = _make_team_df("A", n=20, kf=15.0, ka=12.0)
        b = _make_team_df("B", n=15, kf=13.0, ka=14.0)
        league = pd.concat([a, b], ignore_index=True)
        rows = build_kills_stats_table(
            team_a="A", team_b="B",
            df_a=a, df_b=b, league_scope_df=league,
            metric_col="mv2_total_kills", line=27.0,
        )
        over_row = next(r for r in rows if r.label.startswith("Over"))
        under_row = next(r for r in rows if r.label.startswith("Under"))
        # Parse "XX%" strings
        def _parse_pct(s: str) -> float:
            try:
                return float(s.rstrip("%"))
            except ValueError:
                return float("nan")
        ov_a = _parse_pct(over_row.team_a_value)
        un_a = _parse_pct(under_row.team_a_value)
        if ov_a == ov_a:  # not NaN
            self.assertAlmostEqual(ov_a + un_a, 100.0, places=1)

    def test_tone_higher_is_good(self):
        # Team A has 70% win rate, Team B has 30%, league avg 50%.
        # Win rate tone for A should be "good", for B "warn".
        a = _make_team_df("A", n=20, kf=15.0, ka=12.0, win_rate=0.7)
        b = _make_team_df("B", n=20, kf=13.0, ka=14.0, win_rate=0.3)
        league = pd.concat([a, b], ignore_index=True)
        rows = build_kills_stats_table(
            team_a="A", team_b="B",
            df_a=a, df_b=b, league_scope_df=league,
            metric_col="mv2_total_kills", line=27.0,
        )
        wr_row = next(r for r in rows if r.label == "Win Rate")
        self.assertEqual(wr_row.tone_a, "good")
        self.assertEqual(wr_row.tone_b, "warn")


class TestLeagueRankForTeam(unittest.TestCase):
    """Ranking badge: ↓N most-under / ↑N most-over, ties shared, top-5 only."""

    def _league(self):
        # 6 teams, distinct combined-kills means: A=20 ... F=45 (2 rows each)
        rows = []
        for team, v in [("A", 20), ("B", 25), ("C", 30), ("D", 35), ("E", 40), ("F", 45)]:
            rows += [{"team": team, "mv2_total_kills": v}, {"team": team, "mv2_total_kills": v}]
        return pd.DataFrame(rows)

    def test_lowest_is_rank1_low(self):
        low, high = _league_rank_for_team(self._league(), "mv2_total_kills", "A")
        self.assertEqual(low, "(1)")          # most under
        self.assertEqual(high, "")            # 6th highest → outside top 5

    def test_highest_is_rank1_high(self):
        low, high = _league_rank_for_team(self._league(), "mv2_total_kills", "F")
        self.assertEqual(high, "(1)")         # most over
        self.assertEqual(low, "")             # 6th lowest → outside top 5

    def test_middle_team(self):
        low, high = _league_rank_for_team(self._league(), "mv2_total_kills", "C")
        self.assertEqual(low, "(3)")          # 3rd lowest
        self.assertEqual(high, "(4)")         # 4th highest

    def test_ties_share_position(self):
        df = pd.DataFrame([
            {"team": "A", "mv2_total_kills": 20},
            {"team": "B", "mv2_total_kills": 20},
            {"team": "C", "mv2_total_kills": 30},
        ])
        la, _ = _league_rank_for_team(df, "mv2_total_kills", "A")
        lb, _ = _league_rank_for_team(df, "mv2_total_kills", "B")
        self.assertEqual(la, "(1)")
        self.assertEqual(lb, "(1)")           # tie → both rank 1

    def test_team_absent_returns_blank(self):
        self.assertEqual(_league_rank_for_team(self._league(), "mv2_total_kills", "ZZ"), ("", ""))

    def test_missing_column_returns_blank(self):
        self.assertEqual(_league_rank_for_team(self._league(), "nope", "A"), ("", ""))


class TestStatsRankIntegration(unittest.TestCase):
    """Ranking sits on the % rows: ↑N (most over) on Over, ↓N (most under) on Under."""

    def test_rank_on_over_under_rows_not_combined(self):
        a = _make_team_df("A", n=10, kf=18.0, ka=18.0)   # high combined
        b = _make_team_df("B", n=10, kf=8.0, ka=8.0)     # low combined
        league = pd.concat([a, b, _make_team_df("C", 10, 12, 12)], ignore_index=True)
        rows = build_kills_stats_table(
            team_a="A", team_b="B", df_a=a, df_b=b, league_scope_df=league,
            metric_col="mv2_total_kills", line=27.5,
        )
        combined = next(r for r in rows if r.label.startswith("Combined"))
        over = next(r for r in rows if r.label.startswith("Over"))
        under = next(r for r in rows if r.label.startswith("Under"))
        # Combined no longer carries the rank.
        self.assertEqual(combined.rank_a, "")
        self.assertEqual(combined.rank_b, "")
        # Over row: A is most-over (↑1), Under row: B is most-under (↓1).
        self.assertIn("↑", over.rank_a)
        self.assertIn("1", over.rank_a)
        self.assertIn("↓", under.rank_b)
        self.assertIn("1", under.rank_b)
        # Over row shows only ↑ (no ↓); Under shows only ↓ (no ↑).
        self.assertNotIn("↓", over.rank_a)
        self.assertNotIn("↑", under.rank_b)


if __name__ == "__main__":
    unittest.main()
