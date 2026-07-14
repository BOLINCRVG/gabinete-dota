"""Tests for the Resumo H2H fix.

Bug: H2H recortes showed all ❌ even when the two teams had a strong head-to-head
history, because the H2H slice came from the narrow recorte (split/playoffs) and
hid meetings from other phases. The fix builds H2H from the full season with
normalized name matching. Here we lock in (a) the name normalization and (b) the
odds path: with H2H games present, the recortes pass within the max odd.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.resumo_tab_core import _norm_team_key
from app_core.resumo_core import counts_ou, _merge_h2h_odds, passed_recortes
from app_core.resumo_runtime_core import select_h2h_games
from app_core.odds_core import laplace_odds


class TestNormTeamKey(unittest.TestCase):

    def test_casing_and_spacing(self):
        self.assertEqual(_norm_team_key("  ThunderTalk Gaming "), _norm_team_key("thundertalk gaming"))

    def test_mojibake_dropped(self):
        # Replacement char and stray '?' are stripped so 'L?S'/'L�S' match 'LS'.
        self.assertEqual(_norm_team_key("L�S"), "ls")

    def test_empty(self):
        self.assertEqual(_norm_team_key(None), "")


class TestH2HOddsPath(unittest.TestCase):
    """With H2H games present, the H2H recortes must pass within the max odd."""

    def _h2h_df(self, values):
        return pd.DataFrame([{"total_nashors": v} for v in values])

    def test_three_hits_under_passes(self):
        # Baron 1.5: 3 H2H games all UNDER (0/1 barons) → strong Under.
        df = self._h2h_df([1, 0, 1])
        over, under, n = counts_ou(df, "total_nashors", 1.5)
        self.assertEqual((over, under, n), (0, 3, 3))
        odds = _merge_h2h_odds(
            {}, df_year=df, df_15=df, df_10=df, df_5=df,
            side_builder=lambda d: (
                laplace_odds(*(lambda o, u, nn: (o, nn))(*counts_ou(d, "total_nashors", 1.5))),
                laplace_odds(*(lambda o, u, nn: (u, nn))(*counts_ou(d, "total_nashors", 1.5))),
            ),
        )
        # Under odd = laplace(3,3) = 5/4 = 1.25; within a 5.00 cap all recortes pass.
        passed = passed_recortes(odds, "H2H", "Under", max_odd=5.0)
        self.assertEqual(set(passed), {"Ano", "15", "10", "5"})

    def test_empty_h2h_yields_no_recortes(self):
        # No H2H games → laplace returns None → no recorte passes (correct ❌).
        df = self._h2h_df([])
        odds = _merge_h2h_odds(
            {}, df_year=df, df_15=df, df_10=df, df_5=df,
            side_builder=lambda d: (None, None),
        )
        self.assertEqual(passed_recortes(odds, "H2H", "Under", max_odd=5.0), [])


class TestSelectH2HGamesNotLimitedBySplit(unittest.TestCase):
    """H2H must span the whole season: a Split 1 meeting still counts in Split 2."""

    def _games(self):
        return pd.DataFrame([
            {"team": "A", "opponent": "B", "split": "Split 1", "playoffs": 0, "year": 2026, "date": "2026-02-01"},
            {"team": "B", "opponent": "A", "split": "Split 1", "playoffs": 0, "year": 2026, "date": "2026-02-01"},
            {"team": "A", "opponent": "B", "split": "Split 2", "playoffs": 1, "year": 2026, "date": "2026-07-10"},
            {"team": "A", "opponent": "C", "split": "Split 2", "playoffs": 0, "year": 2026, "date": "2026-07-12"},
        ])

    def test_includes_meetings_from_other_splits(self):
        df_a, df_b = select_h2h_games(
            self._games(), team_a="A", team_b="B", year=2026, norm_team_fn=_norm_team_key,
        )
        # A-vs-B: one from Split 1 + one from Split 2 playoffs = 2 (split never filters).
        self.assertEqual(len(df_a), 2)
        self.assertEqual(len(df_b), 1)
        # The unrelated A-vs-C game is excluded.
        self.assertTrue((df_a["opponent"].astype(str) == "B").all())

    def test_year_filter_scopes_when_provided(self):
        games = self._games()
        games.loc[len(games)] = {"team": "A", "opponent": "B", "split": "Split 1",
                                 "playoffs": 0, "year": 2025, "date": "2025-03-01"}
        df_a_2026, _ = select_h2h_games(games, team_a="A", team_b="B", year=2026, norm_team_fn=_norm_team_key)
        df_a_all, _ = select_h2h_games(games, team_a="A", team_b="B", year=None, norm_team_fn=_norm_team_key)
        self.assertEqual(len(df_a_2026), 2)   # only 2026 meetings
        self.assertEqual(len(df_a_all), 3)    # all years

    def test_empty_or_missing_columns(self):
        self.assertEqual(select_h2h_games(None, team_a="A", team_b="B", norm_team_fn=_norm_team_key)[0].shape[0], 0)
        bad = pd.DataFrame([{"foo": 1}])
        a, b = select_h2h_games(bad, team_a="A", team_b="B", norm_team_fn=_norm_team_key)
        self.assertTrue(a.empty and b.empty)


if __name__ == "__main__":
    unittest.main()
