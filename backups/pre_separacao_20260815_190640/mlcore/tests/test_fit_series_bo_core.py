"""Tests for fit_series_bo_core — walk-forward OOS gamma fitting."""

import math
import unittest
from typing import List

from mlcore.fit_series_bo_core import (
    _build_gamma_rows,
    _eval_gamma,
    _series_prob,
    _split_rows_temporal,
    fit_series_gamma_oos,
)
from mlcore.fit_samples_core import Sample


# ---------------------------------------------------------------------------
# Sample factory
# ---------------------------------------------------------------------------

def _sample(
    blue="T1", red="T2", y=1, game_no=1, league="LCK",
    split="Summer", phase="regular", date_day="2024-06-01",
    p_team=0.55, p_players=0.55, n_team=30.0, n_players=25.0,
) -> Sample:
    return Sample(
        month="2024-06",
        gameid=f"{blue}_{red}_{date_day}_{game_no}",
        league=league,
        patch="14.10",
        split=split,
        macro="KR",
        side="blue",
        bo_bucket="3",
        phase=phase,
        y=float(y),
        p_team=float(p_team),
        p_players=float(p_players),
        n_team=float(n_team),
        n_players=float(n_players),
        form_n_a=10.0,
        form_n_b=10.0,
        form_wins_a=6.0,
        form_wins_b=5.0,
        coverage_ratio=0.85,
        freshness_avg=0.75,
        transfer_signal=0.0,
        lineup_continuity=0.9,
        roster_changed=False,
        same_league=True,
        blue_team=blue,
        red_team=red,
        game_no=int(game_no),
        date_day=str(date_day),
        lane_top=0.0,
        lane_jungle=0.0,
        lane_mid=0.0,
        lane_bot=0.0,
        lane_support=0.0,
    )


def _make_bo3_series(blue, red, winner, date_day, league="LCK"):
    """Create 3 map samples forming a BO3 series that goes to game 3 (2-1).

    max game_no must be 3 for the row builder to classify this as BO3.
    A 2-0 series (max game_no=2) would be excluded by the BO identification logic.
    """
    if winner == blue:
        outcomes = [1, 0, 1]   # blue wins G1, red wins G2, blue wins G3
    else:
        outcomes = [0, 1, 0]   # red wins G1, blue wins G2, red wins G3
    return [
        _sample(blue=blue, red=red, y=outcomes[0], game_no=1, date_day=date_day, league=league),
        _sample(blue=blue, red=red, y=outcomes[1], game_no=2, date_day=date_day, league=league),
        _sample(blue=blue, red=red, y=outcomes[2], game_no=3, date_day=date_day, league=league),
    ]


def _make_series_pool(n: int, *, base_date: str = "2024-01-01") -> List[Sample]:
    """Create n BO3 series alternating winner."""
    from datetime import datetime, timedelta
    samples: List[Sample] = []
    start = datetime.strptime(base_date, "%Y-%m-%d")
    teams = [("T1", "T2"), ("T3", "T4"), ("T5", "T6"), ("T7", "T8")]
    for i in range(n):
        pair = teams[i % len(teams)]
        blue, red = pair
        winner = blue if i % 3 != 0 else red
        date = (start + timedelta(days=i * 4)).strftime("%Y-%m-%d")
        samples.extend(_make_bo3_series(blue, red, winner, date))
    return samples


class TestSeriesProb(unittest.TestCase):

    def test_gamma_1_equals_standard_combinatorial(self):
        from mlcore.bo import prob_win_series
        from mlcore.utils import clip_prob
        p = 0.62
        expected = float(clip_prob(prob_win_series(p, 3)))
        got = _series_prob(p, 3, 1.0)
        self.assertAlmostEqual(got, expected, places=6)

    def test_gamma_gt_1_amplifies_edge(self):
        p = 0.62
        p1 = _series_prob(p, 3, 1.0)
        p2 = _series_prob(p, 3, 1.5)
        self.assertGreater(p2, p1)

    def test_gamma_lt_1_shrinks_edge(self):
        p = 0.62
        p1 = _series_prob(p, 3, 1.0)
        p2 = _series_prob(p, 3, 0.7)
        self.assertLess(p2, p1)

    def test_50pct_unaffected_by_gamma(self):
        p = _series_prob(0.50, 3, 1.5)
        self.assertAlmostEqual(p, 0.50, places=4)


class TestEvalGamma(unittest.TestCase):

    def _rows(self, n=30):
        return [{"bo": 3, "y": int(i % 2), "p_map": 0.60, "date_day": f"2024-0{(i%9)+1}-01"} for i in range(n)]

    def test_returns_logloss_and_n(self):
        rows = self._rows(30)
        out = _eval_gamma(rows, bo=3, gamma=1.0)
        self.assertEqual(out["n"], 30)
        self.assertIn("logloss", out)
        self.assertTrue(math.isfinite(out["logloss"]))

    def test_no_rows_returns_error(self):
        out = _eval_gamma([], bo=3, gamma=1.0)
        self.assertIn("error", out)

    def test_wrong_bo_returns_error(self):
        rows = self._rows(20)
        out = _eval_gamma(rows, bo=5, gamma=1.0)
        self.assertIn("error", out)


class TestSplitRowsTemporal(unittest.TestCase):

    def _rows(self, dates):
        return [{"bo": 3, "y": 1, "p_map": 0.60, "date_day": d} for d in dates]

    def test_split_is_chronological(self):
        dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        rows = self._rows(dates)
        train, test = _split_rows_temporal(rows, oos_frac=0.25)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 1)
        self.assertEqual(test[0]["date_day"], "2024-04-01")

    def test_train_is_earlier_than_test(self):
        dates = [f"2024-{i:02d}-01" for i in range(1, 13)]
        rows = self._rows(dates)
        train, test = _split_rows_temporal(rows, oos_frac=0.25)
        max_train = max(r["date_day"] for r in train)
        min_test = min(r["date_day"] for r in test)
        self.assertLessEqual(max_train, min_test)


class TestBuildGammaRows(unittest.TestCase):

    def test_builds_one_row_per_series(self):
        samples = _make_series_pool(8)
        rows = _build_gamma_rows(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        # 8 series → 8 rows
        self.assertEqual(len(rows), 8)

    def test_rows_contain_date_day(self):
        samples = _make_series_pool(4)
        rows = _build_gamma_rows(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        for r in rows:
            self.assertIn("date_day", r)
            self.assertIsInstance(r["date_day"], str)

    def test_rows_contain_required_keys(self):
        samples = _make_series_pool(4)
        rows = _build_gamma_rows(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        for r in rows:
            for key in ("bo", "y", "p_map", "date_day", "league"):
                self.assertIn(key, r)

    def test_empty_samples_returns_empty(self):
        rows = _build_gamma_rows([], team_scale=1.0, players_scale=1.0, form_cfg={})
        self.assertEqual(rows, [])

    def test_p_map_in_valid_range(self):
        samples = _make_series_pool(8)
        rows = _build_gamma_rows(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        for r in rows:
            self.assertGreater(float(r["p_map"]), 0.0)
            self.assertLess(float(r["p_map"]), 1.0)


class TestFitSeriesGammaOos(unittest.TestCase):

    def test_no_samples_gives_shadow_only(self):
        out = fit_series_gamma_oos([], team_scale=1.0, players_scale=1.0, form_cfg={})
        self.assertEqual(out["gate"]["decision"], "shadow_only")
        self.assertIn("no_series_rows", out["gate"]["reasons"])

    def test_too_few_samples_gives_shadow_only(self):
        samples = _make_series_pool(5)  # << min_train_n=40
        out = fit_series_gamma_oos(samples, team_scale=1.0, players_scale=1.0, form_cfg={},
                                   min_train_n=40)
        self.assertEqual(out["gate"]["decision"], "shadow_only")

    def test_output_has_gate_key(self):
        samples = _make_series_pool(10)
        out = fit_series_gamma_oos(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        self.assertIn("gate", out)
        self.assertIn("decision", out["gate"])
        self.assertIn("reasons", out["gate"])

    def test_output_has_n_rows_total(self):
        samples = _make_series_pool(8)
        out = fit_series_gamma_oos(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        self.assertIn("n_rows_total", out)
        self.assertGreaterEqual(out["n_rows_total"], 0)

    def test_gate_is_promote_or_shadow_only(self):
        samples = _make_series_pool(60)
        out = fit_series_gamma_oos(samples, team_scale=1.0, players_scale=1.0, form_cfg={})
        self.assertIn(out["gate"]["decision"], ("promote", "shadow_only"))

    def test_sufficient_data_produces_bo3_entry(self):
        samples = _make_series_pool(80)
        out = fit_series_gamma_oos(
            samples, team_scale=1.0, players_scale=1.0, form_cfg={},
            min_train_n=30, min_test_n=10,
        )
        self.assertIn("bo3", out)

    def test_bo3_entry_has_gamma(self):
        samples = _make_series_pool(80)
        out = fit_series_gamma_oos(
            samples, team_scale=1.0, players_scale=1.0, form_cfg={},
            min_train_n=30, min_test_n=10,
        )
        if "bo3" in out and "gamma" in out["bo3"]:
            g = float(out["bo3"]["gamma"])
            self.assertGreater(g, 0.0)
            self.assertLess(g, 10.0)

    def test_custom_gammas_respected(self):
        """With only gamma=1.0 in the grid, fitting must return gamma=1.0."""
        samples = _make_series_pool(80)
        out = fit_series_gamma_oos(
            samples, team_scale=1.0, players_scale=1.0, form_cfg={},
            gammas=(1.0,), min_train_n=30, min_test_n=10,
        )
        if "bo3" in out and "gamma" in out["bo3"]:
            self.assertAlmostEqual(float(out["bo3"]["gamma"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
