"""Tests for app_core/live_bet_combo_core.py — combo analysis helpers."""
import unittest

from app_core.live_bet_combo_core import (
    _combo_alignment_bonus,
    _combo_metric_mean,
    _market_call_from_combo_means,
    _weighted_combo_edge,
)
import pandas as pd


class TestMarketCallFromComboMeans(unittest.TestCase):

    def test_kills_both_high_over(self):
        result = _market_call_from_combo_means(kind="kills", mean_a=32.0, mean_b=31.0, scope_mean=28.0)
        self.assertEqual(result, "Favorece over")

    def test_kills_both_low_under(self):
        result = _market_call_from_combo_means(kind="kills", mean_a=24.0, mean_b=23.0, scope_mean=28.0)
        self.assertEqual(result, "Favorece under")

    def test_kills_neutral(self):
        result = _market_call_from_combo_means(kind="kills", mean_a=28.0, mean_b=28.0, scope_mean=28.0)
        self.assertEqual(result, "Neutro")

    def test_time_encurta(self):
        result = _market_call_from_combo_means(kind="time", mean_a=28.0, mean_b=27.0, scope_mean=30.0)
        self.assertEqual(result, "Encurta")

    def test_time_alonga(self):
        result = _market_call_from_combo_means(kind="time", mean_a=32.0, mean_b=31.0, scope_mean=30.0)
        self.assertEqual(result, "Alonga")

    def test_none_scope_returns_sem_base(self):
        result = _market_call_from_combo_means(kind="kills", mean_a=30.0, mean_b=29.0, scope_mean=None)
        self.assertEqual(result, "Sem base")

    def test_unknown_kind_returns_neutro(self):
        result = _market_call_from_combo_means(kind="unknown", mean_a=30.0, mean_b=29.0, scope_mean=28.0)
        self.assertEqual(result, "Neutro")


class TestWeightedComboEdge(unittest.TestCase):

    def test_all_50_pct_gives_zero_edge(self):
        df = pd.DataFrame([{"_score_wr": 0.5, "_games": 10}])
        self.assertAlmostEqual(_weighted_combo_edge(df), 0.0, places=6)

    def test_high_wr_gives_positive_edge(self):
        df = pd.DataFrame([{"_score_wr": 0.75, "_games": 10}])
        result = _weighted_combo_edge(df)
        self.assertGreater(result, 0.0)

    def test_empty_df_gives_zero(self):
        self.assertAlmostEqual(_weighted_combo_edge(pd.DataFrame()), 0.0)

    def test_none_returns_zero(self):
        self.assertAlmostEqual(_weighted_combo_edge(None), 0.0)


class TestComboMetricMean(unittest.TestCase):

    def test_basic_mean(self):
        df = pd.DataFrame([{"Avg Kills": "28.5"}, {"Avg Kills": "30.5"}])
        result = _combo_metric_mean(df, "Avg Kills")
        self.assertAlmostEqual(result, 29.5, places=3)

    def test_missing_column_returns_none(self):
        df = pd.DataFrame([{"other": "1"}])
        self.assertIsNone(_combo_metric_mean(df, "Avg Kills"))

    def test_empty_df_returns_none(self):
        self.assertIsNone(_combo_metric_mean(pd.DataFrame(), "Avg Kills"))

    def test_only_uses_top_2_rows(self):
        df = pd.DataFrame([
            {"Avg Kills": "10.0"},
            {"Avg Kills": "20.0"},
            {"Avg Kills": "999.0"},
        ])
        result = _combo_metric_mean(df, "Avg Kills")
        self.assertAlmostEqual(result, 15.0, places=3)


class TestComboAlignmentBonus(unittest.TestCase):

    def test_over_alignment_adds_bonus(self):
        support_rows = [("Kills", "Mais kills", 2.5)]
        market_calls = {"Kills": "Favorece over"}
        bonus = _combo_alignment_bonus(support_rows, market_calls=market_calls, comp_compare={})
        self.assertGreater(bonus, 0.0)

    def test_under_alignment_adds_bonus(self):
        support_rows = [("Tempo", "Menos longo", 1.5)]
        market_calls = {"Tempo": "Encurta"}
        bonus = _combo_alignment_bonus(support_rows, market_calls=market_calls, comp_compare={})
        self.assertGreater(bonus, 0.0)

    def test_no_match_no_bonus(self):
        support_rows = [("Kills", "Mais kills", 2.0)]
        market_calls = {"Kills": "Favorece under"}
        bonus = _combo_alignment_bonus(support_rows, market_calls=market_calls, comp_compare={})
        self.assertAlmostEqual(bonus, 0.0)

    def test_empty_support_rows_zero(self):
        bonus = _combo_alignment_bonus([], market_calls={}, comp_compare={})
        self.assertAlmostEqual(bonus, 0.0)


if __name__ == "__main__":
    unittest.main()
