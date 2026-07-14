"""Tests for app_core/live_bet_stats_core.py — pure statistical helpers."""
import math
import unittest

import pandas as pd

from app_core.live_bet_stats_core import (
    _laplace_win_pct,
    _league_alert_label,
    _mean_per_min,
    _safe_ratio,
    _scale_0_100,
    _series_mean_safe,
    _wr_dual_html,
    _wr_side_span,
)


class TestLaplaceWinPct(unittest.TestCase):

    def test_all_wins(self):
        df = pd.DataFrame({"result": [1, 1, 1]})
        pct = _laplace_win_pct(df)
        self.assertIsNotNone(pct)
        self.assertGreater(pct, 50.0)

    def test_all_losses(self):
        df = pd.DataFrame({"result": [0, 0, 0]})
        pct = _laplace_win_pct(df)
        self.assertIsNotNone(pct)
        self.assertLess(pct, 50.0)

    def test_empty_df_returns_none(self):
        self.assertIsNone(_laplace_win_pct(pd.DataFrame()))

    def test_none_returns_none(self):
        self.assertIsNone(_laplace_win_pct(None))

    def test_missing_result_col_returns_none(self):
        df = pd.DataFrame({"other": [1, 2]})
        self.assertIsNone(_laplace_win_pct(df))

    def test_50_percent_equal_wins_losses(self):
        df = pd.DataFrame({"result": [1, 0, 1, 0]})
        pct = _laplace_win_pct(df)
        self.assertAlmostEqual(pct, 50.0, delta=5.0)

    def test_returns_float(self):
        df = pd.DataFrame({"result": [1, 0]})
        self.assertIsInstance(_laplace_win_pct(df), float)


class TestLeagueAlertLabel(unittest.TestCase):

    def test_large_positive_delta(self):
        result = _league_alert_label(current_wr=80.0, current_sample=5, other_wr=60.0, other_sample=10)
        self.assertEqual(result, "Campeonato puxa pra cima")

    def test_large_negative_delta(self):
        result = _league_alert_label(current_wr=40.0, current_sample=5, other_wr=60.0, other_sample=10)
        self.assertEqual(result, "Campeonato puxa pra baixo")

    def test_small_delta_returns_empty(self):
        result = _league_alert_label(current_wr=55.0, current_sample=5, other_wr=50.0, other_sample=10)
        self.assertEqual(result, "")

    def test_none_inputs_returns_empty(self):
        self.assertEqual(_league_alert_label(current_wr=None, current_sample=5, other_wr=60.0, other_sample=10), "")

    def test_low_sample_returns_empty(self):
        self.assertEqual(_league_alert_label(current_wr=90.0, current_sample=1, other_wr=50.0, other_sample=10), "")

    def test_mid_delta_returns_sensivel(self):
        result = _league_alert_label(current_wr=60.0, current_sample=5, other_wr=50.0, other_sample=10)
        self.assertEqual(result, "Sensivel ao campeonato")


class TestMeanPerMin(unittest.TestCase):

    def test_basic_calculation(self):
        df = pd.DataFrame({"kills": [10.0], "gamelength": [1800.0]})
        result = _mean_per_min(df, "kills")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 10.0 / 30.0, places=3)

    def test_missing_column_returns_none(self):
        df = pd.DataFrame({"kills": [10.0]})
        self.assertIsNone(_mean_per_min(df, "kills"))

    def test_empty_df_returns_none(self):
        self.assertIsNone(_mean_per_min(pd.DataFrame(), "kills"))

    def test_zero_time_handled(self):
        df = pd.DataFrame({"kills": [5.0], "gamelength": [0.0]})
        result = _mean_per_min(df, "kills")
        self.assertIsNone(result)


class TestScaleAndRatio(unittest.TestCase):

    def test_scale_100(self):
        self.assertAlmostEqual(_scale_0_100(10.0, target=10.0), 100.0)

    def test_scale_50(self):
        self.assertAlmostEqual(_scale_0_100(5.0, target=10.0), 50.0)

    def test_scale_clamped_at_100(self):
        self.assertAlmostEqual(_scale_0_100(20.0, target=10.0), 100.0)

    def test_scale_zero_target(self):
        self.assertAlmostEqual(_scale_0_100(5.0, target=0.0), 0.0)

    def test_safe_ratio_basic(self):
        self.assertAlmostEqual(_safe_ratio(6.0, 3.0), 2.0)

    def test_safe_ratio_zero_denom(self):
        self.assertIsNone(_safe_ratio(6.0, 0.0))

    def test_safe_ratio_none_inputs(self):
        self.assertIsNone(_safe_ratio(None, 3.0))


class TestSeriesMeanSafe(unittest.TestCase):

    def test_basic_mean(self):
        df = pd.DataFrame({"kills": [10.0, 20.0, 30.0]})
        self.assertAlmostEqual(_series_mean_safe(df, "kills"), 20.0)

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        self.assertIsNone(_series_mean_safe(df, "kills"))

    def test_empty_df(self):
        self.assertIsNone(_series_mean_safe(pd.DataFrame(), "kills"))


class TestWrHtml(unittest.TestCase):

    def test_side_span_with_value(self):
        result = _wr_side_span("B", "65.0", 10, "#6ec1ff")
        self.assertIn("65.0", result)
        self.assertIn("#6ec1ff", result)

    def test_side_span_none_shows_dash(self):
        result = _wr_side_span("B", None, 0, "#6ec1ff")
        self.assertIn("opacity", result)

    def test_dual_html_contains_both_sides(self):
        result = _wr_dual_html(blue_value="60.0", blue_games=8, red_value="55.0", red_games=6)
        self.assertIn("6ec1ff", result)
        self.assertIn("ff7a86", result)

    def test_returns_string(self):
        self.assertIsInstance(_wr_side_span("B", "50.0", 5, "blue"), str)


if __name__ == "__main__":
    unittest.main()
