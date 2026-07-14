"""Tests do blend recency-ponderado das janelas (modo "Geral")."""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from app_core.window_blend_core import (
    GERAL_WEIGHTS,
    GERAL_WINDOWS,
    blend_matchup_totals,
    geral_window_weight,
)


@dataclass(frozen=True)
class _T:
    mean: float
    sd: float
    dist: str


class TestWeights(unittest.TestCase):

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(GERAL_WEIGHTS.values()), 1.0, places=9)

    def test_recency_tilt(self):
        # L5 pesa mais que L10, que pesa mais que ano todo.
        self.assertGreater(GERAL_WEIGHTS[5], GERAL_WEIGHTS[10])
        self.assertGreater(GERAL_WEIGHTS[10], GERAL_WEIGHTS[None])

    def test_geral_window_weight(self):
        self.assertEqual(geral_window_weight(5), GERAL_WEIGHTS[5])
        self.assertEqual(geral_window_weight(None), GERAL_WEIGHTS[None])
        self.assertEqual(geral_window_weight(999), 0.0)  # janela fora do Geral

    def test_windows_shape(self):
        labels = [lbl for lbl, _ in GERAL_WINDOWS]
        mgs = [mg for _, mg in GERAL_WINDOWS]
        self.assertEqual(labels, ["Ano todo", "L10", "L5"])
        self.assertEqual(mgs, [None, 10, 5])


class TestBlend(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(blend_matchup_totals([], totals_cls=_T), {})
        self.assertEqual(blend_matchup_totals([(0.0, {"kills": _T(20, 5, "normal")})], totals_cls=_T), {})

    def test_weighted_mean_normal(self):
        # 0.2*Ano(24) + 0.3*L10(26) + 0.5*L5(28) = 4.8+7.8+14 = 26.6
        per = [
            (0.2, {"kills": _T(24.0, 9.0, "normal")}),
            (0.3, {"kills": _T(26.0, 9.0, "normal")}),
            (0.5, {"kills": _T(28.0, 9.0, "normal")}),
        ]
        out = blend_matchup_totals(per, totals_cls=_T)
        self.assertAlmostEqual(out["kills"].mean, 26.6, places=6)
        self.assertEqual(out["kills"].dist, "normal")
        # sd = sqrt(variância ponderada) = 9.0 (todas iguais)
        self.assertAlmostEqual(out["kills"].sd, 9.0, places=6)

    def test_poisson_sd_derived_from_mean(self):
        per = [
            (0.5, {"dragons": _T(4.0, 2.0, "poisson")}),
            (0.5, {"dragons": _T(5.0, 2.2, "poisson")}),
        ]
        out = blend_matchup_totals(per, totals_cls=_T)
        self.assertAlmostEqual(out["dragons"].mean, 4.5, places=6)
        self.assertAlmostEqual(out["dragons"].sd, math.sqrt(4.5), places=6)
        self.assertEqual(out["dragons"].dist, "poisson")

    def test_weights_renormalized_per_metric(self):
        # 'towers' só existe em duas janelas -> pesos renormalizados entre elas.
        per = [
            (0.2, {"kills": _T(24.0, 5.0, "normal")}),
            (0.3, {"kills": _T(26.0, 5.0, "normal"), "towers": _T(11.0, 4.0, "normal")}),
            (0.5, {"kills": _T(28.0, 5.0, "normal"), "towers": _T(10.0, 4.0, "normal")}),
        ]
        out = blend_matchup_totals(per, totals_cls=_T)
        # towers: (0.3*11 + 0.5*10)/0.8 = (3.3+5.0)/0.8 = 10.375
        self.assertAlmostEqual(out["towers"].mean, 10.375, places=6)

    def test_skips_nonfinite_mean(self):
        per = [
            (0.5, {"kills": _T(float("nan"), 5.0, "normal")}),
            (0.5, {"kills": _T(26.0, 5.0, "normal")}),
        ]
        out = blend_matchup_totals(per, totals_cls=_T)
        # só a janela finita conta -> média 26
        self.assertAlmostEqual(out["kills"].mean, 26.0, places=6)

    def test_metric_order_preserved(self):
        per = [
            (1.0, {"kills": _T(24.0, 5.0, "normal"), "towers": _T(11.0, 4.0, "normal"), "dragons": _T(4.0, 2.0, "poisson")}),
        ]
        out = blend_matchup_totals(per, totals_cls=_T)
        self.assertEqual(list(out.keys()), ["kills", "towers", "dragons"])

    def test_real_matchup_totals_cls(self):
        # Integra com a classe real (sem passar totals_cls).
        from mlcore.lines_engine import MatchupTotals
        per = [
            (0.5, {"kills": MatchupTotals(mean=24.0, sd=9.0, dist="normal")}),
            (0.5, {"kills": MatchupTotals(mean=26.0, sd=9.0, dist="normal")}),
        ]
        out = blend_matchup_totals(per)
        self.assertIsInstance(out["kills"], MatchupTotals)
        self.assertAlmostEqual(out["kills"].mean, 25.0, places=6)


if __name__ == "__main__":
    unittest.main()
