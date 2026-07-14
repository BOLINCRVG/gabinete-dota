"""Tests for compute_statscamps_autofill — synthetic data only (injection-proof)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.statscamps_tab_core import compute_statscamps_autofill


def _g(team: str, opp: str, league: str, year: int, split: str, date: str) -> dict:
    return {"team": team, "opponent": opp, "league": league,
            "year": year, "split": split, "date": date}


class TestStatscampsAutofill(unittest.TestCase):

    def test_same_league_prefills(self):
        df = pd.DataFrame([
            _g("T1", "GenG", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
            _g("GenG", "T1", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
        ])
        out = compute_statscamps_autofill(df, "T1", "GenG")
        self.assertEqual(out, {"league": "LCK", "year": "2026", "split": "Rounds 1-2"})

    def test_different_leagues_returns_empty(self):
        df = pd.DataFrame([
            _g("T1", "X", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
            _g("G2", "Y", "LEC", 2026, "Spring", "2026-05-21"),
        ])
        out = compute_statscamps_autofill(df, "T1", "G2")
        self.assertEqual(out, {})

    def test_uses_most_recent_game_for_year_split(self):
        df = pd.DataFrame([
            _g("T1", "GenG", "LCK", 2026, "Cup", "2026-01-10"),
            _g("GenG", "T1", "LCK", 2026, "Rounds 1-2", "2026-05-20"),  # latest
        ])
        out = compute_statscamps_autofill(df, "T1", "GenG")
        self.assertEqual(out["split"], "Rounds 1-2")  # from latest game

    def test_uses_latest_league_per_team(self):
        # T1 moved: old LPL games, recent LCK. Should use LCK (most recent).
        df = pd.DataFrame([
            _g("T1", "Z", "LPL", 2025, "Spring", "2025-03-01"),
            _g("T1", "GenG", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
            _g("GenG", "T1", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
        ])
        out = compute_statscamps_autofill(df, "T1", "GenG")
        self.assertEqual(out.get("league"), "LCK")

    def test_mojibake_team_name_matches(self):
        # session has 'LØS', CSV stored as replacement-char garble.
        df = pd.DataFrame([
            _g("L�S", "RED Canids", "CBLOL", 2026, "Split 1", "2026-05-24"),
            _g("RED Canids", "L�S", "CBLOL", 2026, "Split 1", "2026-05-24"),
        ])
        out = compute_statscamps_autofill(df, "LØS", "RED Canids")
        self.assertEqual(out.get("league"), "CBLOL")

    def test_empty_df_returns_empty(self):
        self.assertEqual(compute_statscamps_autofill(pd.DataFrame(), "A", "B"), {})

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame([{"foo": 1}])
        self.assertEqual(compute_statscamps_autofill(df, "A", "B"), {})

    def test_blank_team_returns_empty(self):
        df = pd.DataFrame([_g("T1", "X", "LCK", 2026, "Cup", "2026-01-10")])
        self.assertEqual(compute_statscamps_autofill(df, "", "T1"), {})

    def test_team_not_in_data_returns_empty(self):
        df = pd.DataFrame([_g("T1", "GenG", "LCK", 2026, "Cup", "2026-01-10")])
        out = compute_statscamps_autofill(df, "T1", "NonExistent")
        self.assertEqual(out, {})

    def test_invalid_dates_skipped(self):
        df = pd.DataFrame([
            _g("T1", "GenG", "LCK", 2026, "Rounds 1-2", "notadate"),
            _g("GenG", "T1", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
            _g("T1", "GenG", "LCK", 2026, "Rounds 1-2", "2026-05-20"),
        ])
        out = compute_statscamps_autofill(df, "T1", "GenG")
        self.assertEqual(out.get("league"), "LCK")


if __name__ == "__main__":
    unittest.main()
