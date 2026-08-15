"""Tests for app_core.resumo_score_core — temporal analysis and scoring."""
from __future__ import annotations

import unittest
from datetime import date, timedelta

import pandas as pd

from app_core.resumo_score_core import (
    CandidateScore,
    TemporalProfile,
    WindowHitRate,
    compute_temporal_profile,
    generate_veredito,
    rank_candidates,
    score_candidate,
    _trend_from_windows,
)


def _mk_df(values: list[float], col_for: str = "kills_for",
            col_against: str = "kills_against",
            split_50: bool = False) -> pd.DataFrame:
    """Build a minimal team_games df.

    split_50=True means first half is low (~10), second half is high (~30)
    to simulate a 'recovering' pattern.
    """
    base_date = date(2026, 1, 1)
    rows = []
    for i, v in enumerate(values):
        rows.append({
            "team": "A",
            "date": str(base_date + timedelta(days=i)),
            "win": 1 if i % 2 == 0 else 0,
            "game_time_min": 32.0,
            col_for:      v / 2,
            col_against:  v / 2,
        })
    return pd.DataFrame(rows)


class TestTrendFromWindows(unittest.TestCase):

    def test_recovering(self):
        """Bad overall (50%), good recent (80%) → recovering."""
        windows = [
            WindowHitRate("Ano todo", 40, 20, 50.0),   # 50%
            WindowHitRate("L15",      15, 10, 66.7),
            WindowHitRate("L10",      10, 7,  70.0),
            WindowHitRate("L5",       5,  4,  80.0),   # 80%
        ]
        trend, score, label, _ = _trend_from_windows(windows)
        self.assertEqual(trend, "recovering")
        self.assertGreater(score, 0)
        self.assertIn("forma recente", label.lower())

    def test_declining(self):
        """Was good (80%), now bad (30%) → declining."""
        windows = [
            WindowHitRate("Ano todo", 40, 32, 80.0),
            WindowHitRate("L15",      15, 8,  53.0),
            WindowHitRate("L10",      10, 4,  40.0),
            WindowHitRate("L5",       5,  1,  20.0),   # 20%
        ]
        trend, score, _label, _ = _trend_from_windows(windows)
        self.assertEqual(trend, "declining")
        self.assertLess(score, 0)

    def test_flat(self):
        windows = [
            WindowHitRate("Ano todo", 40, 24, 60.0),
            WindowHitRate("L15",      15, 9,  60.0),
            WindowHitRate("L10",      10, 6,  60.0),
            WindowHitRate("L5",       5,  3,  60.0),
        ]
        trend, score, _, _ = _trend_from_windows(windows)
        self.assertEqual(trend, "flat")
        self.assertAlmostEqual(score, 0.0)

    def test_improving(self):
        windows = [
            WindowHitRate("Ano todo", 40, 22, 55.0),
            WindowHitRate("L15",      15, 9,  60.0),
            WindowHitRate("L10",      10, 7,  70.0),
            WindowHitRate("L5",       5,  4,  80.0),
        ]
        trend, score, _, _ = _trend_from_windows(windows)
        self.assertIn(trend, ("recovering", "improving"))
        self.assertGreater(score, 0)


class TestComputeTemporalProfile(unittest.TestCase):

    def test_recovering_under_kills(self):
        """First 20 games: 5 kills avg (line=10 → miss). Last 10: 15 avg → hit."""
        # Under kill 10 means we need val < 10.
        # First 20: val=12 (misses), last 10: val=6 (hits).
        vals = [12.0] * 20 + [6.0] * 10
        df = _mk_df(vals)
        tp = compute_temporal_profile(df, metric_key="kills", line=10.0,
                                       side_key="Under", team_name="A")
        # L5 should be 100% (all hits).
        l5 = next((w for w in tp.windows if w.label == "L5"), None)
        self.assertIsNotNone(l5)
        self.assertEqual(l5.pct, 100.0)
        # Overall should be lower.
        year = next((w for w in tp.windows if w.label == "Ano todo"), None)
        self.assertLess(year.pct, l5.pct)

    def test_empty_df_returns_insufficient(self):
        tp = compute_temporal_profile(pd.DataFrame(), metric_key="kills", line=27.5,
                                       side_key="Under", team_name="A")
        self.assertEqual(tp.trend, "insufficient")

    def test_missing_column_returns_insufficient(self):
        df = pd.DataFrame([{"team": "A", "date": "2026-01-01", "win": 1}])
        tp = compute_temporal_profile(df, metric_key="kills", line=27.5,
                                       side_key="Under", team_name="A")
        self.assertEqual(tp.trend, "insufficient")

    def test_time_metric_uses_single_col(self):
        """game_time_min doesn't need to sum for+against."""
        df = pd.DataFrame([
            {"team": "A", "date": f"2026-01-0{i+1}", "win": 1, "game_time_min": 28.0 + i}
            for i in range(10)
        ])
        tp = compute_temporal_profile(df, metric_key="time", line=30.0,
                                       side_key="Under", team_name="A")
        # Values 28-30.5: first 3 (28,29,30) are Under 30 → hit; last 7 are miss.
        self.assertGreater(tp.windows[0].n, 0)


class TestScoreCandidate(unittest.TestCase):

    def _row(self, metric="kills", lado="Abaixo", linha="27.5",
              n_liga=14, n_h2h=4, conf=70):
        return {
            "Mercado": f"{metric} ({linha})",
            "Lado": lado,
            "Linha": linha,
            "Amostra Liga": n_liga,
            "Amostra H2H": n_h2h,
            "Confiança": conf,
            "Odd (Hist)": 1.78,
            "Odd (Modelo)": 1.75,
            "Odd (Mix)": 1.76,
            "_metric": metric,
            "_line": float(linha),
            "_gap_pp": 5.0,
        }

    def _df(self, n=20, val=25.0):
        return _mk_df([val] * n)

    def test_returns_candidate_score(self):
        row = self._row()
        s = score_candidate(row, oc_map={}, df_a=self._df(), df_b=self._df(),
                             team_a="A", team_b="B")
        self.assertIsInstance(s, CandidateScore)
        self.assertGreater(s.total, 0)
        self.assertLessEqual(s.total, 100)

    def test_higher_edge_scores_better(self):
        row = self._row()
        oc_map_good = {"kills (27.5)|Abaixo|27.5": 2.00}  # good edge vs 1.78 hist
        oc_map_bad  = {"kills (27.5)|Abaixo|27.5": 1.30}  # negative edge
        df = self._df()
        s_good = score_candidate(row, oc_map=oc_map_good, df_a=df, df_b=df, team_a="A", team_b="B")
        s_bad  = score_candidate(row, oc_map=oc_map_bad,  df_a=df, df_b=df, team_a="A", team_b="B")
        self.assertGreater(s_good.total, s_bad.total)

    def test_recovering_trend_gets_bonus(self):
        row = self._row(linha="10.0", conf=50)
        vals_recovering = [12.0] * 20 + [6.0] * 10   # starts bad, gets good
        vals_declining   = [6.0] * 20 + [12.0] * 10  # starts good, gets bad
        df_rec = _mk_df(vals_recovering)
        df_dec = _mk_df(vals_declining)
        s_rec = score_candidate(row, oc_map={}, df_a=df_rec, df_b=df_rec, team_a="A", team_b="B")
        s_dec = score_candidate(row, oc_map={}, df_a=df_dec, df_b=df_dec, team_a="A", team_b="B")
        self.assertGreater(s_rec.trend_bonus, s_dec.trend_bonus,
                           "Recovering trend should outscored declining")

    def test_bad_market_over_torres_penalized(self):
        row_bad  = self._row(metric="towers", lado="Acima")  # Over Torres: bad ROI
        row_good = self._row(metric="towers", lado="Abaixo") # Under Torres: good ROI
        df = self._df()
        s_bad  = score_candidate(row_bad,  oc_map={}, df_a=df, df_b=df, team_a="A", team_b="B")
        s_good = score_candidate(row_good, oc_map={}, df_a=df, df_b=df, team_a="A", team_b="B")
        self.assertLess(s_bad.anti_loss_penalty, 0, "Over Torres must have negative penalty")
        self.assertGreater(s_good.total, s_bad.total)


class TestRankCandidates(unittest.TestCase):

    def test_returns_top_n_sorted(self):
        rows = [
            {"Mercado": "Kills (27.5)", "Lado": "Abaixo", "Linha": "27.5",
             "_metric": "kills", "_line": 27.5, "Amostra Liga": 14, "Amostra H2H": 4,
             "Confiança": 70, "Odd (Hist)": 1.78, "Odd (Modelo)": 1.75, "Odd (Mix)": 1.76, "_gap_pp": 4.0},
            {"Mercado": "Torres (11.5)", "Lado": "Acima", "Linha": "11.5",
             "_metric": "towers", "_line": 11.5, "Amostra Liga": 10, "Amostra H2H": 2,
             "Confiança": 55, "Odd (Hist)": 1.92, "Odd (Modelo)": 2.0, "Odd (Mix)": 1.95, "_gap_pp": 20.0},
        ]
        df = _mk_df([25.0] * 20)
        ranked = rank_candidates(rows, oc_map={}, df_a=df, df_b=df, team_a="A", team_b="B", top_n=5)
        self.assertGreater(len(ranked), 0)
        # Kills Under should outscore Over Torres (bad ROI market).
        kills_score  = next((r.total for r in ranked if "kills" in r.row.get("_metric", "")), None)
        towers_score = next((r.total for r in ranked if "towers" in r.row.get("_metric", "")), None)
        if kills_score is not None and towers_score is not None:
            self.assertGreater(kills_score, towers_score)

    def _rows_varied_odds(self):
        return [
            {"Mercado": "Kills (27.5)", "Lado": "Abaixo", "Linha": "27.5",
             "_metric": "kills", "_line": 27.5, "Amostra Liga": 14, "Amostra H2H": 4,
             "Confiança": 70, "Odd (Hist)": 1.50, "Odd (Modelo)": 1.5, "Odd (Mix)": 1.5, "_gap_pp": 4.0},
            {"Mercado": "Torres (11.5)", "Lado": "Abaixo", "Linha": "11.5",
             "_metric": "towers", "_line": 11.5, "Amostra Liga": 10, "Amostra H2H": 2,
             "Confiança": 60, "Odd (Hist)": 2.40, "Odd (Modelo)": 2.4, "Odd (Mix)": 2.4, "_gap_pp": 8.0},
            {"Mercado": "Dragões (4.5)", "Lado": "Abaixo", "Linha": "4.5",
             "_metric": "dragons", "_line": 4.5, "Amostra Liga": 12, "Amostra H2H": 3,
             "Confiança": 65, "Odd (Hist)": 1.90, "Odd (Modelo)": 1.9, "Odd (Mix)": 1.9, "_gap_pp": 6.0},
        ]

    def test_sort_by_odd_ascending(self):
        df = _mk_df([25.0] * 20)
        ranked = rank_candidates(self._rows_varied_odds(), oc_map={}, df_a=df, df_b=df,
                                 team_a="A", team_b="B", top_n=5, sort_by="odd")
        odds = [r.odd_hist for r in ranked]
        self.assertEqual(odds, sorted(odds), "Odd sort must be ascending (smallest first)")
        self.assertEqual(ranked[0].odd_hist, 1.50)

    def test_sort_by_score_is_default(self):
        df = _mk_df([25.0] * 20)
        ranked = rank_candidates(self._rows_varied_odds(), oc_map={}, df_a=df, df_b=df,
                                 team_a="A", team_b="B", top_n=5)
        totals = [r.total for r in ranked]
        self.assertEqual(totals, sorted(totals, reverse=True), "Score sort must be descending")


class TestGenerateVeredito(unittest.TestCase):

    def _make_scored(self, total: int, label: str, pros=None, cons=None) -> CandidateScore:
        empty_w = [WindowHitRate(lbl, 0, 0, None) for lbl in ("Ano todo", "L15", "L10", "L5")]
        tp = TemporalProfile(team="A", windows=empty_w, trend="flat",
                              trend_label="—", trend_score=0.0, summary_line="—")
        row = {"Mercado": label, "Lado": "Abaixo", "Linha": "27.5",
               "_metric": "kills", "_line": 27.5}
        return CandidateScore(
            row=row, temporal_a=tp, temporal_b=tp, total=total,
            conf_pts=0, edge_pts=0, roi_pts=0, trend_bonus=0, anti_loss_penalty=0,
            edge_hist=None, odd_hist=1.78, odd_casa=None, stake_units=None,
            pros=pros or [], cons=cons or [],
        )

    def test_winner_is_highest_score(self):
        s1 = self._make_scored(80, "Kills Under 27.5")
        s2 = self._make_scored(65, "Torres Under 11.5")
        v = generate_veredito([s1, s2], team_a="A", team_b="B")
        self.assertIn("Kills", v["winner_label"])
        self.assertEqual(v["winner_score"], 80)

    def test_empty_ranked_returns_fallback(self):
        v = generate_veredito([], team_a="A", team_b="B")
        self.assertEqual(v["winner_label"], "—")

    def test_close_race_surfaces_tradeoff(self):
        s1 = self._make_scored(75, "Kills Under 27.5")
        s2 = self._make_scored(73, "Torres Under 11.5")
        v = generate_veredito([s1, s2], team_a="A", team_b="B")
        self.assertIsNotNone(v["tradeoff"])


if __name__ == "__main__":
    unittest.main()
