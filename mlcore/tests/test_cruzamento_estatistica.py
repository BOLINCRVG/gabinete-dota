# -*- coding: utf-8 -*-
"""Cruzamento modelo x estatistica do historico (24/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.cruzamento_estatistica_core import acertos, cruzar, veredito


def _hist(time, jogos):
    """jogos: lista de (kills_for, kills_against, towers_for, towers_against, win)."""
    return pd.DataFrame([{"team": time, "date": f"2026-10-{i + 1:02d}", "kills_for": kf,
                          "kills_against": ka, "towers_for": tf, "towers_against": ta, "win": w}
                         for i, (kf, ka, tf, ta, w) in enumerate(jogos)])


LOS = _hist("LOS", [(10, 20, 3, 9, 0)] * 6 + [(15, 12, 8, 4, 1)] * 4)
T1 = _hist("T1", [(20, 8, 9, 2, 1)] * 8 + [(9, 14, 3, 8, 0)] * 2)
TG = pd.concat([LOS, T1], ignore_index=True)


class TestAcertos(unittest.TestCase):

    def test_total_de_torres_under(self):
        # torres por jogo: LOS 12 (x6) e 12 (x4); T1 11 (x8) e 11 (x2) -> under 12,5 bate sempre
        r = acertos({"tipo": "total", "unidade": "Turrets", "lado": "under", "pontos": 12.5}, LOS, T1)
        self.assertEqual(r, {"acertos": 20, "jogos": 20})
        # under 11,5: LOS (12) nunca, T1 (11) sempre -> 10 de 20
        r = acertos({"tipo": "total", "unidade": "Turrets", "lado": "under", "pontos": 11.5}, LOS, T1)
        self.assertEqual(r["acertos"], 10)

    def test_vencedor_usa_vitorias_e_derrotas_do_adversario(self):
        r = acertos({"tipo": "moneyline", "unidade": "", "lado": "home"}, LOS, T1)
        self.assertEqual(r, {"acertos": 4 + 2, "jogos": 20})

    def test_handicap_de_kills(self):
        # LOS +9,5: saldo LOS -10 (x6, perde) e +3 (x4, bate) -> 4;
        # T1 visto do lado da LOS: -12 (x8, perde) e +5 (x2, bate) -> 2
        r = acertos({"tipo": "spread", "unidade": "Kills", "lado": "home", "pontos": 9.5}, LOS, T1)
        self.assertEqual(r["jogos"], 20)
        self.assertEqual(r["acertos"], 4 + 2)

    def test_mercado_sem_estatistica(self):
        self.assertIsNone(acertos({"tipo": "moneyline", "unidade": "First Blood", "lado": "home"}, LOS, T1))


class TestVeredito(unittest.TestCase):

    def test_cortes(self):
        self.assertEqual(veredito({"acertos": 14, "jogos": 20}), "concorda")
        self.assertEqual(veredito({"acertos": 6, "jogos": 20}), "discorda")
        self.assertEqual(veredito({"acertos": 10, "jogos": 20}), "neutra")
        self.assertEqual(veredito({"acertos": 3, "jogos": 4}), "pouco dado")
        self.assertEqual(veredito(None), "sem estatistica")


class TestCruzar(unittest.TestCase):

    def test_resolve_o_nome_da_casa_e_junta_o_veredito(self):
        linha = {"casa": "LOS ", "fora": "T1", "tipo": "total", "unidade": "Turrets",
                 "lado": "over", "pontos": 12.5}
        r = cruzar([linha], TG, lambda n: n.strip())[0]
        self.assertEqual(r["estat_veredito"], "discorda")
        self.assertEqual(r["estat"]["jogos"], 20)

    def test_sem_historico_nao_inventa(self):
        r = cruzar([{"casa": "X", "fora": "Y", "tipo": "total", "unidade": "Kills",
                     "lado": "over", "pontos": 25.5}], TG, None)[0]
        self.assertIsNone(r["estat"])


if __name__ == "__main__":
    unittest.main()
