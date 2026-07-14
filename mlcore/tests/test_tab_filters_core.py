import unittest

import pandas as pd

from app_core.tab_filters_core import (
    collect_year_split_league_options,
    filter_by_league,
    filter_by_year,
    league_options,
    split_options,
)


class TestCollectYearSplitLeagueOptions(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({
            "year": [2025, 2024, 2025],
            "split": ["Spring", "Winter", "Spring"],
            "league": ["LCK", "LEC", "LPL"],
        })

    def test_years_sorted(self):
        years, _, _ = collect_year_split_league_options(self._df())
        self.assertEqual(years, [2024, 2025])

    def test_splits_sorted(self):
        _, splits, _ = collect_year_split_league_options(self._df())
        self.assertEqual(splits, ["Spring", "Winter"])

    def test_leagues_sorted(self):
        _, _, leagues = collect_year_split_league_options(self._df())
        self.assertEqual(leagues, ["LCK", "LEC", "LPL"])

    def test_empty_df_returns_empty_lists(self):
        years, splits, leagues = collect_year_split_league_options(pd.DataFrame())
        self.assertEqual(years, [])
        self.assertEqual(splits, [])
        self.assertEqual(leagues, [])

    def test_none_input_returns_empty(self):
        years, _splits, _leagues = collect_year_split_league_options(None)
        self.assertEqual(years, [])

    def test_years_are_ints(self):
        years, _, _ = collect_year_split_league_options(self._df())
        self.assertTrue(all(isinstance(y, int) for y in years))

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2]})
        years, splits, leagues = collect_year_split_league_options(df)
        self.assertEqual(years, [])
        self.assertEqual(splits, [])
        self.assertEqual(leagues, [])


class TestSplitOptionsAndLeagueOptions(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({
            "year": [2025],
            "split": ["Summer"],
            "league": ["LCK"],
        })

    def test_split_options_returns_list(self):
        self.assertIsInstance(split_options(self._df()), list)

    def test_league_options_returns_list(self):
        self.assertIsInstance(league_options(self._df()), list)

    def test_split_options_content(self):
        self.assertIn("Summer", split_options(self._df()))

    def test_league_options_content(self):
        self.assertIn("LCK", league_options(self._df()))


class TestFilterByLeague(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({"year": [2025, 2024], "league": ["LCK", "LPL"], "x": [1, 2]})

    def test_filters_to_matching_league(self):
        self.assertEqual(len(filter_by_league(self._df(), "LCK")), 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(len(filter_by_league(self._df(), "LEC")), 0)

    def test_empty_league_returns_original(self):
        df = self._df()
        result = filter_by_league(df, "")
        self.assertEqual(len(result), len(df))

    def test_none_league_returns_original(self):
        df = self._df()
        result = filter_by_league(df, None)
        self.assertEqual(len(result), len(df))


class TestFilterByYear(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({"year": [2025, 2024], "league": ["LCK", "LPL"], "x": [1, 2]})

    def test_filters_to_matching_year(self):
        self.assertEqual(len(filter_by_year(self._df(), 2024)), 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(len(filter_by_year(self._df(), 2020)), 0)

    def test_zero_year_returns_original(self):
        df = self._df()
        result = filter_by_year(df, 0)
        self.assertEqual(len(result), len(df))


if __name__ == "__main__":
    unittest.main()
