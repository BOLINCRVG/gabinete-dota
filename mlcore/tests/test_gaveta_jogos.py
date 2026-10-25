# -*- coding: utf-8 -*-
"""Gaveta de jogos da aba Modelo x Mercado (24/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest
from datetime import date

from app_core.gaveta_jogos_core import (
    e_o_confronto_aberto, jogos_da_gaveta, por_dia, rotulo_do_dia, rotulo_do_jogo)
from app_core.modelo_mercado_core import linhas_do_jogo

ODDS = {"jogos": [
    {"id": 1, "casa": "Top Esports", "fora": "Anyone's Legend", "liga": "World Championship", "bo": 1,
     "inicio_utc": "2026-10-24T17:00:00Z"},
    {"id": 2, "casa": "Liquid", "fora": "Bilibili", "liga": "World Championship", "bo": 1,
     "inicio_utc": "2026-10-25T00:00:00Z"},
]}
POLY = {"partidas": [
    {"time_a": "Top Esports", "time_b": "Anyone's Legend", "inicio_utc": "2026-10-24 17:00:00+00", "bo": 1},
    {"time_a": "Vitality Bees", "time_b": "Mental Rush", "inicio_utc": "2026-10-27 16:00:00+00",
     "bo": 5, "torneio": "LGC", "evento": "rb-mr"},
]}


class TestGaveta(unittest.TestCase):

    def test_junta_casas_sem_repetir_e_marca_as_fontes(self):
        js = jogos_da_gaveta(ODDS, POLY)
        self.assertEqual(len(js), 3)
        tes = js[0]
        self.assertEqual(tes["fontes"], ["Pinnacle", "Polymarket"])
        self.assertEqual(js[-1]["id"], "poly:rb-mr")

    def test_mesmo_jogo_com_nome_diferente_nas_casas(self):
        odds = {"jogos": [{"id": 9, "casa": "Invictus", "fora": "Secret Whales", "bo": 1,
                           "inicio_utc": "2026-10-24T21:00:00Z"}]}
        poly = {"partidas": [{"time_a": "Invictus Gaming", "time_b": "Team Secret Whales",
                              "inicio_utc": "2026-10-24 21:00:00+00", "bo": 1}]}
        js = jogos_da_gaveta(odds, poly)
        self.assertEqual(len(js), 1)
        self.assertEqual(js[0]["fontes"], ["Pinnacle", "Polymarket"])

    def test_horario_da_agenda_oficial_vence(self):
        # Pinnacle 17:15Z (+15 min), Polymarket com 1 h a mais, agenda 17:00Z = 14:00 BRT
        odds = {"jogos": [{"id": 3, "casa": "T1", "fora": "CTBC Flying Oyster", "bo": 1,
                           "inicio_utc": "2026-10-26T17:15:00Z"}]}
        poly = {"partidas": [{"time_a": "T1", "time_b": "CTBC Flying Oyster",
                              "inicio_utc": "2026-10-26 18:00:00+00", "bo": 1}]}
        agenda = {"jogos": [{"teamA_raw": "CTBC Flying Oyster", "teamB_raw": "T1",
                             "inicio_utc": "2026-10-26T17:00:00Z"}]}
        js = jogos_da_gaveta(odds, poly, agenda)
        self.assertEqual(len(js), 1)
        self.assertEqual(js[0]["quando"].strftime("%d/%m %H:%M"), "26/10 14:00")

    def test_dia_pelo_horario_de_brasilia(self):
        # 25/10 00:00 UTC = 24/10 21:00 em Brasilia -> mesmo dia do TES x AL
        dias = por_dia(jogos_da_gaveta(ODDS, POLY), hoje=date(2026, 10, 24))
        self.assertEqual(list(dias)[0], "Hoje 24/10")
        self.assertEqual(len(dias["Hoje 24/10"]), 2)
        self.assertIn("ter 27/10", dias)

    def test_rotulos(self):
        self.assertEqual(rotulo_do_dia(date(2026, 10, 25), date(2026, 10, 24)), "Amanha 25/10")
        g = jogos_da_gaveta(ODDS, {})[0]
        # Pinnacle registra 15 min depois do inicio: 17:00 UTC - 15 min = 13:45 em Brasilia
        self.assertEqual(rotulo_do_jogo(g), "13:45   Top Esports  x  Anyone's Legend")

    def test_reconhece_o_confronto_aberto_em_qualquer_ordem(self):
        g = jogos_da_gaveta(ODDS, {})[1]
        res = {"Liquid": "Team Liquid", "Bilibili": "Bilibili Gaming"}.get
        self.assertTrue(e_o_confronto_aberto(g, {"teamA": "Bilibili Gaming", "teamB": "Team Liquid"}, res))
        self.assertFalse(e_o_confronto_aberto(g, {"teamA": "T1", "teamB": "Team Liquid"}, res))


class TestLinhasDoJogo(unittest.TestCase):

    def test_a_linha_mais_equilibrada_de_cada_mercado_com_os_dois_lados(self):
        def l(p, lado, odd):
            return {"jogo": 1, "nome_pt": "Total - Kills - mapa 1", "tipo": "total",
                    "pontos": p, "lado": lado, "odd_casa": odd, "odd_modelo": 1.9}
        completo = [l(25.5, "over", 1.65), l(25.5, "under", 2.18),
                    l(26.5, "over", 1.81), l(26.5, "under", 1.97),
                    l(27.5, "over", 1.98), l(27.5, "under", 1.78)]
        r = linhas_do_jogo(completo, 1)
        self.assertEqual([(x["pontos"], x["lado"]) for x in r], [(26.5, "over"), (26.5, "under")])

    def test_sem_modelo_fica_de_fora_e_vencedor_bo1_uma_vez(self):
        def v(nome, lado, odd, mod=1.8):
            return {"jogo": 1, "nome_pt": nome, "tipo": "moneyline", "lado": lado,
                    "odd_casa": odd, "odd_modelo": mod}
        completo = [v("Vencedor - mapa 1", "home", 1.74), v("Vencedor - mapa 1", "away", 2.15),
                    v("Vencedor - serie", "home", 1.74), v("Vencedor - serie", "away", 2.15),
                    v("Primeiro sangue", "home", 1.83, None)]
        nomes = [x["nome_pt"] for x in linhas_do_jogo(completo, 1)]
        self.assertEqual(nomes, ["Vencedor - serie", "Vencedor - serie"])


if __name__ == "__main__":
    unittest.main()
