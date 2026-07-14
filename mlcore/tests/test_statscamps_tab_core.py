"""Tests for app_core/statscamps_tab_core.py — pure helpers (autofill + lookup)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.statscamps_tab_core import (
    _sc_team_match_key,
    _sc_team_row,
    compute_statscamps_autofill,
)


class TestTeamMatchKey(unittest.TestCase):

    def test_casing_and_spacing(self):
        self.assertEqual(_sc_team_match_key("  Gen.G "), _sc_team_match_key("gen.g"))

    def test_mojibake_stripped(self):
        self.assertEqual(_sc_team_match_key("L�S"), "ls")

    def test_none_empty(self):
        self.assertEqual(_sc_team_match_key(None), "")


def _games():
    return pd.DataFrame([
        {"team": "Gen.G", "opponent": "T1", "league": "LCK", "year": 2026, "split": "Spring", "date": "2026-02-01"},
        {"team": "Gen.G", "opponent": "DK", "league": "LCK", "year": 2026, "split": "Summer", "date": "2026-07-10"},
        {"team": "T1", "opponent": "Gen.G", "league": "LCK", "year": 2026, "split": "Summer", "date": "2026-07-09"},
        {"team": "LGD", "opponent": "TT", "league": "LPL", "year": 2026, "split": "Summer", "date": "2026-07-11"},
    ])


class TestAutofill(unittest.TestCase):

    def test_same_league_fills_recent(self):
        out = compute_statscamps_autofill(_games(), "Gen.G", "T1")
        self.assertEqual(out.get("league"), "LCK")
        self.assertEqual(out.get("year"), "2026")
        # mais recente entre os dois na liga = Gen.G Summer (2026-07-10)
        self.assertEqual(out.get("split"), "Summer")

    def test_different_leagues_returns_blank(self):
        out = compute_statscamps_autofill(_games(), "Gen.G", "LGD")
        self.assertEqual(out, {})

    def test_unknown_team_blank(self):
        self.assertEqual(compute_statscamps_autofill(_games(), "Gen.G", "Cloud9"), {})

    def test_empty_df_blank(self):
        self.assertEqual(compute_statscamps_autofill(pd.DataFrame(), "A", "B"), {})

    def test_missing_columns_blank(self):
        df = pd.DataFrame([{"team": "A", "foo": 1}])
        self.assertEqual(compute_statscamps_autofill(df, "A", "B"), {})

    def test_mojibake_team_name_matches(self):
        out = compute_statscamps_autofill(_games(), "gen.g", "t1")
        self.assertEqual(out.get("league"), "LCK")


class TestTeamRow(unittest.TestCase):

    def _raw(self):
        return pd.DataFrame([
            {"Team": "Gen.G", "Win Rate": 70.0, "Kills": 15.0},
            {"Team": "T1", "Win Rate": 65.0, "Kills": 14.0},
        ])

    def test_found_tolerant(self):
        row = _sc_team_row(self._raw(), "  gen.g ")
        self.assertIsNotNone(row)
        self.assertEqual(row["Win Rate"], 70.0)

    def test_not_found(self):
        self.assertIsNone(_sc_team_row(self._raw(), "Cloud9"))

    def test_empty_or_no_team_col(self):
        self.assertIsNone(_sc_team_row(pd.DataFrame(), "A"))
        self.assertIsNone(_sc_team_row(pd.DataFrame([{"X": 1}]), "A"))


if __name__ == "__main__":
    unittest.main()
