# -*- coding: utf-8 -*-
"""Suas casas x Pinnacle (app_core/suas_casas_core.py)."""
from __future__ import annotations

import unittest

from app_core import suas_casas_core as sc

JOGO = {"time1": "T1", "time2": "Movistar Koi", "inicio_utc": "2026-10-28T20:00:00.000Z",
        "casas": {"pinnacle": {"181": [1.20, 5.00], "1847": [1.30, 3.50]},
                  "bet365": {"181": [1.25, 4.50]},
                  "estrelabet": {"181": [1.10, 6.00]}, "estrelabet.bet.br": {"181": [1.22, 4.80]},
                  "1xbet": {"181": [1.30, 4.00]}}}


class TestSuasCasas(unittest.TestCase):
    def test_achar_orienta_pelo_time_1_da_tela(self):
        g, inv = sc.achar([JOGO], "Movistar Koi", "T1")
        self.assertIs(g, JOGO)
        self.assertTrue(inv)
        g, inv = sc.achar([JOGO], "T1", "Movistar KOI")
        self.assertFalse(inv)
        self.assertEqual(sc.achar([JOGO], "T1", "Gen.G"), (None, False))

    def test_achar_usa_o_resolvedor(self):
        g, _ = sc.achar([JOGO], "T1", "MKOI", resolver=lambda n: {"Movistar Koi": "MKOI"}.get(n, n))
        self.assertIs(g, JOGO)

    def test_linhas_valor_contra_a_pinnacle_do_mesmo_retrato(self):
        lins = sc.linhas(JOGO, False, "lol", p_serie=0.70, p_mapa=0.65)
        casas = {(l["mercado"], l["casa"]) for l in lins}
        self.assertEqual(casas, {("Série", "Pinnacle"), ("Série", "bet365"), ("Série", "Estrela Bet"),
                                 ("Mapa 1", "Pinnacle")})       # 1xbet nao e casa dele
        b365 = next(l for l in lins if l["casa"] == "bet365")
        p_pin = (1 / 1.2) / (1 / 1.2 + 1 / 5.0)
        self.assertAlmostEqual(b365["valor_pin_a"], round((p_pin * 1.25 - 1) * 100, 1))
        self.assertGreater(b365["valor_pin_a"], 0)
        self.assertAlmostEqual(b365["valor_mod_a"], round((0.70 * 1.25 - 1) * 100, 1))
        estrela = next(l for l in lins if l["casa"] == "Estrela Bet")
        self.assertEqual(estrela["slug"], "estrelabet.bet.br")   # a brasileira primeiro

    def test_linhas_invertidas_e_sem_modelo(self):
        lins = sc.linhas(JOGO, True, "lol")
        b365 = next(l for l in lins if l["casa"] == "bet365")
        self.assertEqual((b365["odd_a"], b365["odd_b"]), (4.50, 1.25))
        self.assertIsNone(b365["valor_mod_a"])

    def test_html_marca_verde_e_estrela(self):
        lins = sc.linhas(JOGO, False, "lol", p_serie=0.85, p_mapa=0.65)
        h = sc.html("T1", "Movistar Koi", lins, "2026-10-27T22:36:26+00:00")
        self.assertIn('class="verde"', h)
        self.assertIn("★", h)
        self.assertIn("27/10 19:36", h)
        self.assertIn("a régua", h)                     # a linha da propria Pinnacle
        self.assertEqual(sc.html("a", "b", []), "")
        sem_pin = [dict(l, valor_pin_a=None, valor_pin_b=None) for l in lins if l["casa"] == "bet365"]
        self.assertIn("sem Pinnacle", sc.html("T1", "Movistar Koi", sem_pin))

    def test_mercados_por_esporte(self):
        self.assertEqual(sc.mercados("dota"), {"161": 0, "1647": 1, "1649": 2, "1651": 3})


if __name__ == "__main__":
    unittest.main()
