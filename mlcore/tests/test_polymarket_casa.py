# -*- coding: utf-8 -*-
"""A Polymarket lida direto dela vira a casa "polymarket" das telas (15/11/2026).
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app_core import polymarket_casa_core as pcc
from app_core import suas_casas_core as sc
from tools.coletar_polymarket import ler_partida, outros_mercados

AGORA = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)


def _m(tipo, pergunta, lados, linha=None, bid=0.48, ask=0.52, **k):
    return dict({"sportsMarketType": tipo, "question": pergunta, "outcomes": json.dumps(lados), "line": linha,
                 "outcomePrices": json.dumps(["0.5", "0.5"]), "bestBid": bid, "bestAsk": ask, "liquidity": "5000",
                 "gameStartTime": "2026-11-16 18:00:00+00", "closed": False}, **k)


EVENTO = {"title": "Dota 2: Alfa vs Beta (BO3) - Liga X", "slug": "alfa-beta", "markets": [
    _m("moneyline", "Dota 2: Alfa vs Beta (BO3) - Liga X", ["Alfa", "Beta"], bid=0.60, ask=0.64),
    _m("child_moneyline", "Dota 2: Alfa vs Beta - Game 1 Winner", ["Alfa", "Beta"], bid=0.56, ask=0.60),
    _m("totals", "Games Total: O/U 2.5", ["Over", "Under"], 2.5, bid=0.44, ask=0.48),
    _m("map_handicap", "Game Handicap: Beta (-1.5) vs Alfa (+1.5)", ["Beta", "Alfa"], -1.5, bid=0.18, ask=0.22),
    _m("kill_over_under_game", "Total Kills Over/Under 45.5 in Game 2?", ["Over", "Under"], 45.5),
    _m("kill_over_under_game", "Total Kills Over/Under 50.5 in Game 1?", ["Over", "Under"], 50.5, bid=0.10, ask=0.90),
    _m("dota2_rampage", "Game 1: Any Player Rampage?", ["Yes", "No"]),
    _m("totals", "Games Total: O/U 2.5", ["Over", "Under"], 2.5, closed=True),
]}


def _odd(p):
    return round(1 / p, 3)


class TestColetor(unittest.TestCase):
    def test_titulo_do_dota_e_do_lol(self):
        p = ler_partida(EVENTO, AGORA)
        self.assertEqual((p["time_a"], p["time_b"], p["bo"], p["torneio"]), ("Alfa", "Beta", 3, "Liga X"))
        self.assertAlmostEqual(p["p_a"], 0.62)
        lol = dict(EVENTO, title="LoL: Alfa vs Beta (BO5) - Worlds")
        self.assertEqual(ler_partida(lol, AGORA)["bo"], 5)

    def test_so_guarda_o_que_o_modelo_precifica(self):
        ms = outros_mercados(EVENTO)
        self.assertEqual(sorted({m["tipo"] for m in ms}),
                         ["child_moneyline", "kill_over_under_game", "map_handicap", "totals"])
        self.assertEqual(len(ms), 5)                      # o fechado e o rampage ficam de fora
        k = next(m for m in ms if m["tipo"] == "kill_over_under_game" and m["linha"] == 45.5)
        self.assertEqual(k["mapa"], 2)


class TestComoCasa(unittest.TestCase):
    def setUp(self):
        self.p = ler_partida(EVENTO, AGORA)

    def test_odd_de_compra(self):
        self.assertEqual(pcc.odds_de_compra(0.62, 0.04), (_odd(0.64), _odd(0.40)))
        self.assertIsNone(pcc.odds_de_compra(0.62, None))          # sem livro
        self.assertIsNone(pcc.odds_de_compra(0.50, 0.80))          # livro vazio

    def test_formato_das_casas(self):
        mls, lins = pcc.como_casa(self.p, "dota")
        self.assertEqual(mls["161"], [_odd(0.64), _odd(0.40)])
        self.assertEqual(mls["1647"], [_odd(0.60), _odd(0.44)])
        self.assertIn(["maps", "total", 0, 2.5, _odd(0.48), _odd(0.56), None], lins)
        # "Beta -1,5" a 0,20: no formato das casas e o Alfa (time 1) com +1,5
        self.assertIn(["maps", "spread", 0, 1.5, _odd(0.82), _odd(0.22), None], lins)
        self.assertIn(["kills", "total", 2, 45.5, _odd(0.52), _odd(0.52), None], lins)
        self.assertFalse([l for l in lins if l[3] == 50.5])        # spread de 0,80: fora

    def test_junta_no_jogo_da_oddspapi_e_vira_quando_os_times_estao_trocados(self):
        doc = {"lido_em": "2026-11-06T11:00:00+00:00", "jogos": [
            {"time1": "Beta", "time2": "Alfa", "inicio_utc": "2026-11-16T18:30:00.000Z",
             "casas": {"stake": {"161": [2.4, 1.5]}, "polymarket": {"161": [9.0, 9.0]}}, "linhas": {}}]}
        j = pcc.juntar(doc, {"coletado_em": "2026-11-15T11:35:00+00:00", "partidas": [self.p]}, "dota")
        self.assertEqual(len(j["jogos"]), 1)
        g = j["jogos"][0]
        self.assertEqual(g["casas"]["polymarket"]["161"], [_odd(0.40), _odd(0.64)])
        self.assertEqual(g["casas"]["stake"]["161"], [2.4, 1.5])
        self.assertIn(["maps", "spread", 0, -1.5, _odd(0.22), _odd(0.82), None], g["linhas"]["polymarket"])
        self.assertEqual(doc["jogos"][0]["casas"]["polymarket"]["161"], [9.0, 9.0])     # nao mexe no original

    def test_jogo_que_a_oddspapi_nao_tem_entra_novo(self):
        j = pcc.juntar({"jogos": []}, {"coletado_em": "x", "partidas": [self.p]}, "dota")
        g = j["jogos"][0]
        self.assertTrue(g["so_polymarket"])
        self.assertEqual((g["time1"], g["time2"], g["inicio_utc"]), ("Alfa", "Beta", "2026-11-16T18:00:00.000Z"))

    def test_enxerta_quando_o_nome_difere_nas_duas_fontes(self):
        opapi = {"time1": "Beta Esports", "time2": "Alfa", "inicio_utc": "2026-11-16T18:30:00.000Z",
                 "casas": {"stake": {"161": [2.4, 1.5]}}, "linhas": {}}
        novo = pcc.juntar({"jogos": [opapi]}, {"coletado_em": "x", "partidas": [self.p]}, "dota")["jogos"]
        self.assertEqual(len(novo), 2)
        g, inv = sc.achar(novo, "Alfa", "Beta", lambda n: "Beta" if n == "Beta Esports" else n)
        self.assertTrue(inv)                                   # a base e o jogo da OddsPapi (Beta Esports x Alfa)
        self.assertEqual(g["casas"]["stake"]["161"], [2.4, 1.5])
        self.assertEqual(g["casas"]["polymarket"]["161"], [_odd(0.40), _odd(0.64)])


class TestNaTela(unittest.TestCase):
    def test_carregar_poe_a_polymarket_e_tira_jogo_que_ja_comecou(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "dota_paths.py").write_text("", encoding="utf-8")
            velho = dict(ler_partida(EVENTO, AGORA), time_a="Gama", time_b="Delta", inicio_utc="2026-11-15 10:00:00+00")
            (raiz / "polymarket_dota.json").write_text(json.dumps(
                {"coletado_em": "2026-11-15T11:35:00+00:00", "partidas": [ler_partida(EVENTO, AGORA), velho]}),
                encoding="utf-8")
            doc = sc.carregar(raiz, AGORA)
            self.assertEqual([(g["time1"], g["time2"]) for g in doc["jogos"]], [("Alfa", "Beta")])
            self.assertNotIn("motivo", doc)
            g, inv = sc.achar(doc["jogos"], "Beta", "Alfa")
            ls = sc.linhas(g, inv, "dota", p_serie=0.45, casas_dele=["Polymarket"])
            serie = next(l for l in ls if l["periodo"] == 0)
            self.assertEqual((serie["casa"], serie["odd_a"]), ("Polymarket", _odd(0.40)))
            hc = sc.linhas_handicap_mapas(g, inv, "dota", casas_dele=["Polymarket"])
            self.assertEqual([(l["handicap_a"], l["odd_a"]) for l in hc], [(-1.5, _odd(0.22))])

    def test_sem_arquivo_da_polymarket_fica_como_era(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sc.carregar(Path(tmp))["jogos"], [])

    def test_aviso_de_leitura(self):
        lin = {"mercado": "Série", "periodo": 0, "casa": "Polymarket", "slug": "polymarket", "odd_a": 1.5, "odd_b": 2.5,
               "valor_pin_a": None, "valor_pin_b": None, "valor_mod_a": None, "valor_mod_b": None}
        h = sc.html("A", "B", [lin], "2026-11-06T11:00:00+00:00", "2026-11-15T11:35:00+00:00")
        self.assertIn("Polymarket</b> é lida direto dela", h)


if __name__ == "__main__":
    unittest.main()
