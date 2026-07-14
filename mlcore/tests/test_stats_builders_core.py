"""Tests for app_core/stats_builders_core.py — table builders and metric helpers."""
import math
import unittest

import pandas as pd

from app_core.stats_builders_core import (
    apply_scope_filters,
    build_championship_stats_table,
    build_stats_table_view,
    compute_team_metric_pct,
    enrich_resumo_rows_with_stats_detail,
    format_matchup_stats_detail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope_df(**kwargs) -> pd.DataFrame:
    """Minimal team-game DataFrame for scope tests."""
    base = {
        "team": "T1",
        "side": "blue",
        "win": 1,
        "total_kills": 25,
        "total_towers": 11,
        "total_dragons": 5,
        "total_nashors": 2,
        "total_inhibitors": 2,
        "game_time_min": 30.0,
        "kills_for": 15,
        "deaths_for": 5,
        "towers_for": 8,
        "gold_for": 60000,
        "firstblood": 1,
        "firstherald": 1,
        "firsttower": 1,
        "firstdragon": 1,
        "firstbaron": 1,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def _multi_team_df() -> pd.DataFrame:
    rows = [
        {"team": "T1", "side": "blue", "win": 1, "total_kills": 28, "kills_for": 15},
        {"team": "T1", "side": "red",  "win": 0, "total_kills": 22, "kills_for": 10},
        {"team": "G2", "side": "blue", "win": 1, "total_kills": 24, "kills_for": 13},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# apply_scope_filters
# ---------------------------------------------------------------------------

class TestApplyScopeFilters(unittest.TestCase):

    def test_none_returns_empty(self):
        result = apply_scope_filters(None)
        self.assertTrue(result.empty)

    def test_empty_df_returns_empty(self):
        result = apply_scope_filters(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_all_filter_keeps_all(self):
        df = pd.concat([_scope_df(side="blue"), _scope_df(side="red")], ignore_index=True)
        result = apply_scope_filters(df, side_filter="All")
        self.assertEqual(len(result), 2)

    def test_blue_filter(self):
        df = pd.concat([_scope_df(side="blue"), _scope_df(side="red")], ignore_index=True)
        result = apply_scope_filters(df, side_filter="blue")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["side"].iloc[0], "blue")

    def test_red_filter(self):
        df = pd.concat([_scope_df(side="blue"), _scope_df(side="red")], ignore_index=True)
        result = apply_scope_filters(df, side_filter="red")
        self.assertEqual(len(result), 1)

    def test_wins_filter(self):
        df = pd.concat([_scope_df(win=1), _scope_df(win=0)], ignore_index=True)
        result = apply_scope_filters(df, win_loss_filter="wins")
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result["win"].iloc[0]), 1.0)

    def test_losses_filter(self):
        df = pd.concat([_scope_df(win=1), _scope_df(win=0)], ignore_index=True)
        result = apply_scope_filters(df, win_loss_filter="losses")
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result["win"].iloc[0]), 0.0)

    def test_combined_side_and_win(self):
        rows = pd.concat([
            _scope_df(side="blue", win=1),
            _scope_df(side="blue", win=0),
            _scope_df(side="red", win=1),
        ], ignore_index=True)
        result = apply_scope_filters(rows, side_filter="blue", win_loss_filter="wins")
        self.assertEqual(len(result), 1)

    def test_does_not_mutate_input(self):
        df = _scope_df()
        original_len = len(df)
        apply_scope_filters(df, side_filter="red")
        self.assertEqual(len(df), original_len)


# ---------------------------------------------------------------------------
# compute_team_metric_pct
# ---------------------------------------------------------------------------

class TestComputeTeamMetricPct(unittest.TestCase):

    def _df(self, kills=(25, 30, 20)):
        rows = [{"team": "T1", "total_kills": k} for k in kills]
        return pd.DataFrame(rows)

    def test_none_df_returns_none(self):
        result = compute_team_metric_pct(None, team="T1", metric_key="kills", side_key="over", line_val=24.5)
        self.assertIsNone(result)

    def test_empty_df_returns_none(self):
        result = compute_team_metric_pct(pd.DataFrame(), team="T1", metric_key="kills", side_key="over", line_val=24.5)
        self.assertIsNone(result)

    def test_team_not_found_returns_none(self):
        df = self._df()
        result = compute_team_metric_pct(df, team="G2", metric_key="kills", side_key="over", line_val=24.5)
        self.assertIsNone(result)

    def test_over_line_all_above(self):
        # all 3 games: kills > 19.5 → 100%
        df = self._df(kills=(25, 30, 20))
        result = compute_team_metric_pct(df, team="T1", metric_key="kills", side_key="over", line_val=19.5)
        self.assertAlmostEqual(result, 100.0, places=2)

    def test_over_line_none_above(self):
        # all 3 games: kills > 35 → 0%
        df = self._df(kills=(25, 30, 20))
        result = compute_team_metric_pct(df, team="T1", metric_key="kills", side_key="over", line_val=35.0)
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_over_line_partial(self):
        # 2 of 3 games > 24.5 → 66.67%
        df = self._df(kills=(25, 30, 20))
        result = compute_team_metric_pct(df, team="T1", metric_key="kills", side_key="over", line_val=24.5)
        self.assertAlmostEqual(result, 200.0 / 3.0, places=1)

    def test_under_side(self):
        # kills <= 24.5: only 20 qualifies → 33.33%
        df = self._df(kills=(25, 30, 20))
        result = compute_team_metric_pct(df, team="T1", metric_key="kills", side_key="under", line_val=24.5)
        self.assertAlmostEqual(result, 100.0 / 3.0, places=1)

    def test_unknown_metric_returns_none(self):
        df = self._df()
        result = compute_team_metric_pct(df, team="T1", metric_key="unknown_metric", side_key="over", line_val=24.5)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# build_championship_stats_table
# ---------------------------------------------------------------------------

class TestBuildChampionshipStatsTable(unittest.TestCase):

    def test_none_returns_empty(self):
        result = build_championship_stats_table(None)
        self.assertTrue(result.empty)

    def test_empty_df_returns_empty(self):
        result = build_championship_stats_table(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_single_team_row(self):
        df = _scope_df()
        result = build_championship_stats_table(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Team"].iloc[0], "T1")

    def test_multiple_teams_separate_rows(self):
        df = _multi_team_df()
        result = build_championship_stats_table(df)
        self.assertEqual(len(result), 2)
        self.assertIn("T1", result["Team"].values)
        self.assertIn("G2", result["Team"].values)

    def test_games_count_correct(self):
        # T1 has 2 rows in _multi_team_df
        df = _multi_team_df()
        result = build_championship_stats_table(df)
        t1_row = result[result["Team"] == "T1"].iloc[0]
        self.assertEqual(t1_row["# Games"], 2)

    def test_threshold_columns_added(self):
        df = _scope_df()
        thresholds = {"kills": [24.5, 29.5]}
        result = build_championship_stats_table(df, threshold_lines_by_metric=thresholds)
        # Should have columns like "% Kills > 24.5" and "% Kills > 29.5"
        kill_cols = [c for c in result.columns if "Kills" in str(c) and "%" in str(c)]
        self.assertGreaterEqual(len(kill_cols), 2)

    def test_required_base_columns_present(self):
        df = _scope_df()
        result = build_championship_stats_table(df)
        for col in ("Team", "# Games", "Win Rate"):
            self.assertIn(col, result.columns, f"Missing column: {col}")


# ---------------------------------------------------------------------------
# build_stats_table_view
# ---------------------------------------------------------------------------

class TestBuildStatsTableView(unittest.TestCase):

    def _raw(self):
        return pd.DataFrame([
            {"Team": "T1", "# Games": 10, "Win Rate": 60.0, "Game Time": 30.5, "FB%": 55.0},
            {"Team": "G2", "# Games": 8,  "Win Rate": 50.0, "Game Time": 31.2, "FB%": 48.0},
        ])

    def test_none_returns_empty(self):
        result = build_stats_table_view(None, min_to_mmss_fn=lambda x: "-")
        self.assertTrue(result.empty)

    def test_empty_returns_empty(self):
        result = build_stats_table_view(pd.DataFrame(), min_to_mmss_fn=lambda x: "-")
        self.assertTrue(result.empty)

    def test_sorted_by_team_by_default(self):
        result = build_stats_table_view(self._raw(), min_to_mmss_fn=lambda x: str(x))
        self.assertEqual(result["Team"].iloc[0], "G2")

    def test_priority_teams_come_first(self):
        result = build_stats_table_view(
            self._raw(),
            min_to_mmss_fn=lambda x: str(x),
            priority_teams=["T1"],
        )
        self.assertEqual(result["Team"].iloc[0], "T1")

    def test_game_time_formatted(self):
        def mmss(x):
            return "30:30"
        result = build_stats_table_view(self._raw(), min_to_mmss_fn=mmss)
        self.assertEqual(result["Game Time"].iloc[0], "30:30")

    def test_games_count_as_int_string(self):
        result = build_stats_table_view(self._raw(), min_to_mmss_fn=lambda x: "-")
        self.assertTrue(all(result["# Games"].astype(str).str.isdigit()))


# ---------------------------------------------------------------------------
# format_matchup_stats_detail
# ---------------------------------------------------------------------------

class TestFormatMatchupStatsDetail(unittest.TestCase):

    def test_both_present(self):
        result = format_matchup_stats_detail(team_a="T1", team_b="G2", pct_a=65.4, pct_b=48.2)
        self.assertIn("T1", result)
        self.assertIn("G2", result)
        self.assertIn("65%", result)
        self.assertIn("48%", result)

    def test_none_pct_shows_dash(self):
        result = format_matchup_stats_detail(team_a="T1", team_b="G2", pct_a=None, pct_b=50.0)
        self.assertIn("-", result)
        self.assertIn("50%", result)

    def test_format_pipe_separated(self):
        result = format_matchup_stats_detail(team_a="T1", team_b="G2", pct_a=60.0, pct_b=40.0)
        self.assertIn("|", result)

    def test_rounding(self):
        result = format_matchup_stats_detail(team_a="T1", team_b="G2", pct_a=66.6, pct_b=33.3)
        self.assertIn("67%", result)
        self.assertIn("33%", result)


# ---------------------------------------------------------------------------
# enrich_resumo_rows_with_stats_detail
# ---------------------------------------------------------------------------

class TestEnrichResumoRowsWithStatsDetail(unittest.TestCase):

    def _scope(self):
        return pd.DataFrame([
            {"team": "T1", "total_kills": 28},
            {"team": "T1", "total_kills": 22},
            {"team": "G2", "total_kills": 25},
            {"team": "G2", "total_kills": 20},
        ])

    def test_none_rows_returns_empty(self):
        result = enrich_resumo_rows_with_stats_detail(
            None, df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        self.assertEqual(result, [])

    def test_adds_stats_detail_key(self):
        rows = [{"_metric": "kills", "_line": 24.5, "Lado": "over"}]
        result = enrich_resumo_rows_with_stats_detail(
            rows, df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        self.assertEqual(len(result), 1)
        self.assertIn("Stats Detail", result[0])

    def test_stats_detail_contains_team_names(self):
        rows = [{"_metric": "kills", "_line": 24.5, "Lado": "over"}]
        result = enrich_resumo_rows_with_stats_detail(
            rows, df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        detail = result[0]["Stats Detail"]
        self.assertIn("T1", detail)
        self.assertIn("G2", detail)

    def test_missing_metric_no_stats_detail(self):
        rows = [{"_metric": "", "_line": None, "Lado": "over"}]
        result = enrich_resumo_rows_with_stats_detail(
            rows, df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        self.assertNotIn("Stats Detail", result[0])

    def test_non_dict_rows_skipped(self):
        result = enrich_resumo_rows_with_stats_detail(
            ["not", "a", "dict"],
            df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        self.assertEqual(result, [])

    def test_under_side(self):
        rows = [{"_metric": "kills", "_line": 24.5, "Lado": "abaixo"}]
        result = enrich_resumo_rows_with_stats_detail(
            rows, df_scope=self._scope(), team_a="T1", team_b="G2"
        )
        self.assertIn("Stats Detail", result[0])


if __name__ == "__main__":
    unittest.main()
