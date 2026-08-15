import unittest

import pandas as pd

from app_core.matchup_filters_core import resolve_prev_season_with_fallback
from mlcore.stats_teams import Filters


def _make_games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _filter_fn(df: pd.DataFrame, *, team: str, filters: Filters, league_mode: str, map_mode: str, max_games) -> pd.DataFrame:
    """Minimal filter: by team and optionally by split."""
    if df.empty:
        return pd.DataFrame()
    result = df[df["team"].astype(str) == str(team)].copy()
    if filters.split:
        result = result[result["split"].astype(str) == str(filters.split)].copy()
    return result


SAMPLE_GAMES = _make_games([
    {"team": "T1", "split": "Spring", "league": "LCK", "year": 2024},
    {"team": "T1", "split": "Spring", "league": "LCK", "year": 2024},
    {"team": "GEN", "split": "Spring", "league": "LCK", "year": 2024},
    {"team": "GEN", "split": "Summer", "league": "LCK", "year": 2024},
])


class TestResolvePrevSeasonWithFallback(unittest.TestCase):

    def _call(self, prev_team_games, *, team_a="T1", team_b="GEN", split="Spring", playoffs=None, year=2024):
        filters = Filters(year=None, league="LCK", split=split, playoffs=playoffs)
        return resolve_prev_season_with_fallback(
            prev_team_games,
            filters=filters,
            team_a=team_a,
            team_b=team_b,
            prev_season_year=year,
            league_mode="all",
            map_mode="avg",
            filter_team_games_fn=_filter_fn,
        )

    def test_none_input_returns_empty(self):
        active, dfA, dfB, note = self._call(None)
        self.assertIsNone(active)
        self.assertTrue(dfA.empty)
        self.assertTrue(dfB.empty)
        self.assertEqual(note, "")

    def test_empty_dataframe_returns_empty(self):
        active, dfA, dfB, _note = self._call(pd.DataFrame())
        self.assertIsNone(active)
        self.assertTrue(dfA.empty)
        self.assertTrue(dfB.empty)

    def test_blank_team_a_returns_empty(self):
        active, _dfA, _dfB, _note = self._call(SAMPLE_GAMES, team_a="   ", team_b="GEN")
        self.assertIsNone(active)

    def test_blank_team_b_returns_empty(self):
        active, _dfA, _dfB, _note = self._call(SAMPLE_GAMES, team_a="T1", team_b="")
        self.assertIsNone(active)

    def test_successful_filter_both_teams_found(self):
        # Both T1 and GEN have Spring games
        active, dfA, dfB, note = self._call(SAMPLE_GAMES, split="Spring")
        self.assertIsNotNone(active)
        self.assertFalse(dfA.empty)
        self.assertFalse(dfB.empty)
        self.assertEqual(note, "")
        self.assertEqual(active.split, "Spring")

    def test_fallback_when_one_team_missing_in_split(self):
        # GEN has no Summer games vs T1 in Spring → only GEN has Summer
        # Use split=Summer: T1 has no Summer rows → triggers fallback
        active, dfA, dfB, note = self._call(SAMPLE_GAMES, split="Summer")
        # After fallback (split=None): T1 has Spring only, GEN has Spring+Summer
        self.assertIsNotNone(active)
        self.assertIsNone(active.split)
        self.assertFalse(dfA.empty)
        self.assertFalse(dfB.empty)
        self.assertIn("fallback", note)

    def test_no_note_when_no_fallback_needed(self):
        _active, _dfA, _dfB, note = self._call(SAMPLE_GAMES, split="Spring")
        self.assertEqual(note, "")

    def test_year_propagated_to_filters(self):
        active, _dfA, _dfB, _note = self._call(SAMPLE_GAMES, split="Spring", year=2024)
        self.assertIsNotNone(active)
        self.assertEqual(active.year, 2024)

    def test_zero_year_maps_to_none(self):
        active, _dfA, _dfB, _note = self._call(SAMPLE_GAMES, split="Spring", year=0)
        self.assertIsNotNone(active)
        self.assertIsNone(active.year)

    def test_returns_tuple_of_four(self):
        result = self._call(SAMPLE_GAMES, split="Spring")
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
