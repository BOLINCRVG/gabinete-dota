"""Tests for active_teams_in_recent_window + filter_to_active_teams.

Bug we're guarding against: roster-deleted teams (Karmine Corp Blue,
Los Ratones, etc.) leaking into the contextual opponent profile because
the league scope was using season-wide data."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market_v3_analysis_core import (
    active_teams_in_recent_window,
    filter_to_active_teams,
)


def _mk_row(team: str, opponent: str, date: str) -> dict:
    return {"team": team, "opponent": opponent, "date": date, "metric": 25.0}


class TestActiveTeamsRecentWindow(unittest.TestCase):

    def test_basic_recent_set(self):
        df = pd.DataFrame([
            _mk_row("T1", "Gen.G", "2026-05-26"),
            _mk_row("Gen.G", "T1", "2026-05-26"),
            _mk_row("DeadTeam", "T1", "2026-03-01"),
        ])
        active = active_teams_in_recent_window(df, days=21)
        self.assertEqual(active, {"T1", "Gen.G"})
        self.assertNotIn("DeadTeam", active)

    def test_uses_max_date_not_wall_clock(self):
        # Reference date must come from the data, not real time.
        df = pd.DataFrame([
            _mk_row("A", "B", "2020-01-10"),
            _mk_row("B", "A", "2020-01-10"),
            _mk_row("C", "A", "2019-12-01"),  # 40 days before max
        ])
        active = active_teams_in_recent_window(df, days=21)
        self.assertEqual(active, {"A", "B"})

    def test_empty_df_returns_empty_set(self):
        self.assertEqual(active_teams_in_recent_window(pd.DataFrame()), set())

    def test_missing_columns_returns_empty(self):
        self.assertEqual(
            active_teams_in_recent_window(pd.DataFrame([{"foo": 1}])),
            set(),
        )

    def test_all_dates_invalid_returns_empty(self):
        df = pd.DataFrame([{"team": "A", "date": "notadate"}])
        self.assertEqual(active_teams_in_recent_window(df), set())

    def test_window_size_respected(self):
        df = pd.DataFrame([
            _mk_row("A", "B", "2026-05-26"),
            _mk_row("OldB", "A", "2026-05-15"),  # 11 days before max
        ])
        # 7-day window → OldB out
        self.assertEqual(active_teams_in_recent_window(df, days=7), {"A"})
        # 14-day window → OldB in
        self.assertEqual(active_teams_in_recent_window(df, days=14), {"A", "OldB"})


class TestFilterToActiveTeams(unittest.TestCase):

    def test_drops_dead_team_rows(self):
        df = pd.DataFrame([
            _mk_row("T1", "Gen.G", "2026-05-26"),
            _mk_row("T1", "DeadTeam", "2026-04-01"),
            _mk_row("Gen.G", "T1", "2026-05-26"),
            _mk_row("DeadTeam", "T1", "2026-04-01"),
        ])
        out = filter_to_active_teams(df, {"T1", "Gen.G"})
        # Both team and opponent must be active
        self.assertEqual(len(out), 2)
        self.assertNotIn("DeadTeam", set(out["team"]) | set(out["opponent"]))

    def test_empty_active_set_is_noop(self):
        df = pd.DataFrame([_mk_row("T1", "G2", "2026-05-26")])
        out = filter_to_active_teams(df, set())
        self.assertEqual(len(out), 1)

    def test_empty_df_returns_empty(self):
        out = filter_to_active_teams(pd.DataFrame(), {"T1"})
        self.assertTrue(out is None or out.empty)

    def test_missing_opponent_col_only_filters_team(self):
        df = pd.DataFrame([
            {"team": "T1", "date": "2026-05-26"},
            {"team": "DeadTeam", "date": "2026-04-01"},
        ])
        out = filter_to_active_teams(df, {"T1"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["team"], "T1")


if __name__ == "__main__":
    unittest.main()
