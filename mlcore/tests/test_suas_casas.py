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

    def test_so_as_casas_escolhidas_e_a_pinnacle(self):
        lins = sc.linhas(JOGO, False, "lol", casas_dele=["Stake", "Polymarket"])
        self.assertEqual({l["casa"] for l in lins}, {"Pinnacle"})     # o JOGO nao tem Stake nem Polymarket
        lins = sc.linhas(JOGO, False, "lol", casas_dele=["bet365"])
        self.assertEqual({l["casa"] for l in lins}, {"Pinnacle", "bet365"})
        ls = sc.linhas_de_mapa(JOGO_LINHAS, False, "lol", casas_dele=["Stake"])
        self.assertEqual({l["slug"] for l in ls}, {"stake"})

    def test_cotacao_furada_e_descartada(self):
        jogo = {"casas": {}, "linhas": {"polymarket": [["kills", "total", 1, 27.5, 1.852, 1.042, None]],
                                         "stake": [["kills", "total", 1, 27.5, 1.78, 1.78, None]]}}
        ls = sc.linhas_de_mapa(jogo, False, "lol")
        self.assertEqual({l["slug"] for l in ls}, {"stake"})      # 1,85 x 1,04 soma 150%: fora
        self.assertTrue(sc.par_ok(1.852, 2.041))
        self.assertFalse(sc.par_ok(1.852, 1.042))

    def test_mercados_por_esporte(self):
        self.assertEqual(sc.mercados("dota"), {"161": 0, "1647": 1, "1649": 2, "1651": 3})


JOGO_LINHAS = {"time1": "T1", "time2": "Movistar Koi",
               "casas": {"pinnacle": {"1847": [1.30, 3.50]}},
               "linhas": {
                   # bet365 publicou o handicap com o sinal trocado (o T1, favorito, +8,5)
                   "bet365": [["kills", "spread", 1, 8.5, 1.80, 1.80, None],
                              ["kills", "total", 1, 26.5, 1.80, 1.80, None]],
                   "bet365-nj": [["kills", "total", 1, 26.5, 2.50, 1.50, None]],   # espelho: fora
                   "stake": [["kills", "spread", 1, -7.5, 1.62, 1.98, None],
                             ["kills", "total", 1, 26.5, 1.78, 1.85, None],
                             ["kills", "total", 1, 28.5, 2.20, 1.50, None],
                             ["kills", "team_total", 1, 16.5, 1.80, 1.90, 2]],
                   "superbet": [["kills", "total", 1, 26.5, 1.87, 1.82, None]]}}


class TestMelhorLinha(unittest.TestCase):
    def test_so_a_versao_que_ele_usa_e_o_handicap_desvirado(self):
        ls = sc.linhas_de_mapa(JOGO_LINHAS, False, "lol")
        self.assertNotIn("bet365-nj", {l["slug"] for l in ls})
        b365 = {(l["lado"], l["pontos"]) for l in ls if l["slug"] == "bet365" and l["tipo"] == "spread"}
        self.assertEqual(b365, {("home", -8.5), ("away", 8.5)})   # o T1 (favorito) DA kills
        stake = {(l["lado"], l["pontos"]) for l in ls if l["slug"] == "stake" and l["tipo"] == "spread"}
        self.assertEqual(stake, {("home", -7.5), ("away", 7.5)})   # ja vinha certo: intocado

    def test_orientacao_invertida(self):
        ls = sc.linhas_de_mapa(JOGO_LINHAS, True, "lol")
        tt = [l for l in ls if l["tipo"] == "team_total"]
        self.assertEqual({l["time"] for l in tt}, {"home"})       # time 2 da OddsPapi = Time 1 da tela
        sp = {(l["lado"], l["pontos"]) for l in ls if l["slug"] == "stake" and l["tipo"] == "spread"}
        self.assertEqual(sp, {("away", -7.5), ("home", 7.5)})

    def test_melhor_odd_e_melhor_linha(self):
        ls = sc.linhas_de_mapa(JOGO_LINHAS, False, "lol")
        chances = {26.5: 0.62, 28.5: 0.40}

        def chance(l):
            if l["tipo"] != "total":
                return None
            p = chances[l["pontos"]]
            return p if l["lado"] == "over" else 1 - p
        rs = sc.melhores_linhas(ls, chance)
        over265 = next(r for r in rs if r["tipo"] == "total" and r["lado"] == "over" and r["pontos"] == 26.5)
        self.assertEqual((over265["casa"], over265["odd"]), ("Superbet", 1.87))
        self.assertEqual(dict(over265["outras"]), {"bet365": 1.80, "Stake": 1.78})
        self.assertAlmostEqual(over265["valor_mod"], round((0.62 * 1.87 - 1) * 100, 1))
        overs = [r for r in rs if r["tipo"] == "total" and r["lado"] == "over"]
        self.assertEqual([r["pontos"] for r in overs if r["melhor_linha"]], [26.5])
        # o Menos de 28,5 (60%, justa 1,67) passa na regra do resumo; o Menos de 26,5
        # (38%, justa 2,63) nao concorre a melhor linha, por maior que fosse o valor
        unders = {r["pontos"]: r for r in rs if r["tipo"] == "total" and r["lado"] == "under"}
        self.assertTrue(unders[28.5]["melhor_linha"])
        self.assertFalse(unders[26.5]["melhor_linha"])
        self.assertTrue(sc.provavel(unders[28.5]))
        self.assertFalse(sc.provavel(unders[26.5]))
        self.assertIsNone(over265["p_pinnacle"])            # a Pinnacle nao tem esta linha

    def test_rotulos_em_portugues(self):
        r = {"tipo": "spread", "lado": "home", "pontos": -8.5, "unidade": "kills"}
        self.assertEqual(sc.rotulo_linha(r, "T1", "KOI"), ("Handicap de kills", "T1 -8,5"))
        r = {"tipo": "team_total", "lado": "under", "pontos": 16.5, "unidade": "kills", "time": "away"}
        self.assertEqual(sc.rotulo_linha(r, "T1", "KOI"), ("Kills de KOI", "Menos de 16,5"))
        r = {"tipo": "total", "lado": "over", "pontos": 10.5, "unidade": "turrets"}
        self.assertEqual(sc.rotulo_linha(r, "T1", "KOI"), ("Total de torres", "Mais de 10,5"))


if __name__ == "__main__":
    unittest.main()
