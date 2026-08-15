# -*- coding: utf-8 -*-
"""Testes do entre-mapas (app_core/live_between_maps_core.py).

Pilar 2 da força-tarefa (23/07): a matemática do momentum (p_next_map) e a
probabilidade de fechar a série via DP (p_series_rest) — validadas OOS e sem
teste. Valores conferidos à mão a partir de MOMENTUM_D=0.17."""
import math
import unittest

from app_core.live_between_maps_core import (
    MOMENTUM_D,
    _fair_odd,
    _sig,
    p_next_map,
    p_series_rest,
)


class PNextMapTests(unittest.TestCase):

    def test_placar_empatado_nao_ajusta(self):
        self.assertAlmostEqual(p_next_map(0.5, 0, 0), 0.5)
        self.assertAlmostEqual(p_next_map(0.7, 2, 2), 0.7)
        self.assertAlmostEqual(p_next_map(0.42, 5, 5), 0.42)

    def test_lider_ganha_shift_positivo(self):
        # A na frente: +0.17 no logit de 0.5 -> sig(0.17)
        self.assertAlmostEqual(p_next_map(0.5, 1, 0), _sig(0.17), places=6)
        self.assertGreater(p_next_map(0.5, 1, 0), 0.5)

    def test_atras_ganha_shift_negativo_simetrico(self):
        self.assertAlmostEqual(p_next_map(0.5, 0, 1), _sig(-0.17), places=6)
        # simétrico em torno de 0.5 quando p_base=0.5
        self.assertAlmostEqual(p_next_map(0.5, 1, 0) + p_next_map(0.5, 0, 1), 1.0, places=6)

    def test_d_e_017(self):
        self.assertAlmostEqual(MOMENTUM_D, 0.17)


class PSeriesRestTests(unittest.TestCase):

    def test_series_simetrica_do_00_com_pbase_meio(self):
        # BO3, 0-0, p=0.5 -> exatamente 0.5 (momentum cancela no agregado)
        self.assertAlmostEqual(p_series_rest(0.5, 0, 0, 3), 0.5, places=6)

    def test_liderar_aumenta_prob_de_fechar(self):
        base = p_series_rest(0.5, 0, 0, 3)
        liderando = p_series_rest(0.5, 1, 0, 3)
        self.assertGreater(liderando, base)
        # valor exato conferido à mão: rec(1,0) com d=0.17
        pm = _sig(0.17)
        esperado = pm * 1.0 + (1.0 - pm) * 0.5  # rec(1,1)=0.5
        self.assertAlmostEqual(liderando, esperado, places=6)

    def test_estados_ja_decididos(self):
        self.assertEqual(p_series_rest(0.5, 2, 0, 3), 1.0)
        self.assertEqual(p_series_rest(0.5, 0, 2, 3), 0.0)

    def test_monotonica_no_pbase(self):
        vals = [p_series_rest(p, 0, 0, 5) for p in (0.3, 0.5, 0.7)]
        self.assertLess(vals[0], vals[1])
        self.assertLess(vals[1], vals[2])

    def test_bo5_precisa_de_3(self):
        # 2-0 em BO5 ainda não fecha (need=3); deve estar entre 0 e 1
        p = p_series_rest(0.5, 2, 0, 5)
        self.assertGreater(p, 0.5)
        self.assertLess(p, 1.0)


class FairOddTests(unittest.TestCase):

    def test_inverso_da_prob(self):
        self.assertAlmostEqual(_fair_odd(0.5), 2.0)
        self.assertAlmostEqual(_fair_odd(0.25), 4.0)

    def test_clipa_extremos_sem_estourar(self):
        self.assertTrue(math.isfinite(_fair_odd(0.0)))
        self.assertTrue(math.isfinite(_fair_odd(1.0)))


if __name__ == "__main__":
    unittest.main()
