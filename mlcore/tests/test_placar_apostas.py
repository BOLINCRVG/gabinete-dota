# -*- coding: utf-8 -*-
"""Placar das apostas indicadas (15/11/2026). Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from app_core import placar_apostas_core as pac

AGORA = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)

JOGO = {"time1": "Team Spirit", "time2": "LGD Gaming", "inicio_utc": "2026-11-16T19:00:00.000Z", "torneio": "X",
        "polymarket_direta": True, "polymarket_evento": "dota2-spirit-lgd", "polymarket_times": ["Team Spirit", "LGD Gaming"],
        "casas": {"polymarket": {"161": [1.30, 4.60], "1647": [1.40, 3.00]}, "stake": {"161": [1.20, 4.30]}},
        "linhas": {"polymarket": [["maps", "total", 0, 2.5, 2.50, 1.60, None], ["maps", "spread", 0, -1.5, 2.00, 1.95, None],
                                  ["kills", "total", 1, 54.5, 1.90, 1.90, None]]}}


def _l(forma, per, lado, odd, pontos=None, metrica="ml", **k):
    return dict({"jogo": 7, "forma": forma, "periodo": per, "lado": lado, "odd": odd, "pontos": pontos,
                 "metrica": metrica}, **k)


PIN = {"jogos": [{"id": 7, "casa": "LGD", "fora": "Spirit", "bo": 3, "inicio_utc": "2026-11-16T19:00:00Z"},
                 {"id": 8, "casa": "Aurora", "fora": "OG", "bo": 3, "inicio_utc": "2026-11-16T19:00:00Z"}],
       "linhas": [_l("vence", 0, "home", 4.24), _l("vence", 0, "away", 1.2288),
                  _l("handicap", 0, "home", 1.9615, 1.5), _l("handicap", 0, "away", 1.8929, -1.5),
                  _l("total", 0, "Over", 2.27, 2.5), _l("total", 0, "Under", 1.6024, 2.5),
                  _l("total", 1, "over", 1.8065, 54.5, "kills"), _l("total", 1, "under", 1.9709, 54.5, "kills")]}


class TestCasarComAPinnacle(unittest.TestCase):
    def test_mesmo_time(self):
        self.assertTrue(pac.mesmo_time("LGD", "LGD Gaming"))
        self.assertTrue(pac.mesmo_time("Spirit", "Team Spirit"))
        self.assertTrue(pac.mesmo_time("Gen.G", "Gen.g Esports"))
        self.assertFalse(pac.mesmo_time("T1", "TSM"))
        self.assertFalse(pac.mesmo_time("OG", "LGD"))

    def test_acha_o_jogo_invertido(self):
        pj, inv = pac.jogo_da_pinnacle(PIN, JOGO)
        self.assertEqual((pj["id"], inv), (7, True))
        outro_dia = dict(JOGO, inicio_utc="2026-11-20T19:00:00.000Z")
        self.assertIsNone(pac.jogo_da_pinnacle(PIN, outro_dia)[0])

    def test_reguas_na_orientacao_do_jogo(self):
        rg = pac.reguas(PIN, PIN["jogos"][0], True)
        p_spirit = (1 / 1.2288) / (1 / 1.2288 + 1 / 4.24)
        self.assertAlmostEqual(rg[("ml", 0)], p_spirit)
        self.assertIn(("hc", -1.5), rg)                    # Spirit (time 1) -1,5
        self.assertAlmostEqual(rg[("hc", -1.5)], (1 / 1.8929) / (1 / 1.8929 + 1 / 1.9615))
        self.assertIn(("tm", 2.5), rg)
        self.assertIn(("tk", 1, 54.5), rg)


class TestCandidatos(unittest.TestCase):
    def test_so_entra_valor_entre_1_e_20_e_a_melhor_casa(self):
        cs = pac.candidatos(JOGO, PIN, "dota")
        nomes = {c["nome"]: c for c in cs}
        # Spirit: justa ~1,29; a Polymarket paga 1,30 (valor ~0,8%: fora) e a Stake 1,20 (negativo)
        self.assertNotIn("Team Spirit vence a série", nomes)
        # LGD: justa ~4,45; Polymarket 4,60 (+3%) ganha da Stake 4,30
        lgd = nomes["LGD Gaming vence a série"]
        self.assertEqual((lgd["casa"], lgd["odd"], lgd["regua"]), ("Polymarket", 4.6, "pinnacle"))
        self.assertTrue(1.0 <= lgd["valor_pct"] <= 20.0)
        self.assertFalse(lgd["estrela"])
        # mais de 2,5 mapas: justa ~2,42; paga 2,50
        self.assertIn("Mais de 2,5 mapas", nomes)
        self.assertIn("Team Spirit -1,5 mapas", nomes)
        self.assertEqual(nomes["Team Spirit -1,5 mapas"]["linha"], -1.5)
        self.assertTrue(all(c["evento"] == "dota2-spirit-lgd" for c in cs))

    def test_sem_jogo_na_pinnacle_nao_indica(self):
        self.assertEqual(pac.candidatos(dict(JOGO, time1="Alfa", polymarket_times=["Alfa", "LGD Gaming"]), PIN, "dota"), [])

    def test_novas_nao_repete_nem_grava_jogo_comecado(self):
        cs = pac.candidatos(JOGO, PIN, "dota")
        n1 = pac.novas(cs, [], AGORA, "dota")
        self.assertEqual(len(n1), len(cs))
        self.assertEqual(pac.novas(cs, n1, AGORA, "dota"), [])
        depois = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
        self.assertEqual(pac.novas(cs, [], depois, "dota"), [])


def _m(tipo, pergunta, lados, precos, linha=None, closed=True):
    return {"sportsMarketType": tipo, "question": pergunta, "outcomes": json.dumps(lados),
            "outcomePrices": json.dumps(precos), "line": linha, "closed": closed}


EVENTO = {"markets": [
    _m("moneyline", "Dota 2: LGD Gaming vs Team Spirit (BO3)", ["LGD Gaming", "Team Spirit"], ["0", "1"]),
    _m("child_moneyline", "... - Game 1 Winner", ["LGD Gaming", "Team Spirit"], ["1", "0"]),
    _m("child_moneyline", "... - Game 3 Winner", ["LGD Gaming", "Team Spirit"], ["0.5", "0.5"]),
    _m("totals", "Games Total: O/U 2.5", ["Over", "Under"], ["0", "1"], 2.5),
    _m("map_handicap", "Game Handicap: Team Spirit (-1.5) vs LGD Gaming (+1.5)", ["Team Spirit", "LGD Gaming"], ["1", "0"], -1.5),
    _m("kill_over_under_game", "Total Kills Over/Under 54.5 in Game 1?", ["Over", "Under"], ["0.4", "0.6"], 54.5, closed=False),
]}


def _a(mercado, lado, mapa=0, linha=None, odd=2.0):
    return {"times_poly": ["Team Spirit", "LGD Gaming"], "mercado": mercado, "lado": lado, "mapa": mapa,
            "linha": linha, "odd": odd}


class TestFechar(unittest.TestCase):
    def test_cada_mercado(self):
        self.assertEqual(pac.resultado(_a("ml", "1"), EVENTO), "ganhou")
        self.assertEqual(pac.resultado(_a("ml", "2"), EVENTO), "perdeu")
        self.assertEqual(pac.resultado(_a("ml", "1", mapa=1), EVENTO), "perdeu")
        self.assertEqual(pac.resultado(_a("ml", "1", mapa=3), EVENTO), "anulada")     # mapa 3 nao foi jogado
        self.assertEqual(pac.resultado(_a("tm", "2", linha=2.5), EVENTO), "ganhou")
        self.assertEqual(pac.resultado(_a("hc", "1", linha=-1.5), EVENTO), "ganhou")
        self.assertEqual(pac.resultado(_a("hc", "2", linha=1.5), EVENTO), "perdeu")
        self.assertIsNone(pac.resultado(_a("tk", "1", mapa=1, linha=54.5), EVENTO))     # ainda aberto
        self.assertIsNone(pac.resultado(_a("tm", "1", linha=3.5), EVENTO))              # linha que nao existe

    def test_lucro_e_resumo(self):
        aps = [dict(_a("ml", "1", odd=1.8), resultado="ganhou", lucro=pac.lucro({"odd": 1.8}, "ganhou")),
               dict(_a("ml", "2"), resultado="perdeu", lucro=pac.lucro({"odd": 2.0}, "perdeu")),
               dict(_a("ml", "1", mapa=3), resultado="anulada", lucro=0.0), _a("tm", "1", linha=2.5)]
        r = pac.resumo(aps)
        self.assertEqual((r["n"], r["abertas"], r["fechadas"], r["ganhou"], r["perdeu"], r["anuladas"]), (4, 1, 2, 1, 1, 1))
        self.assertAlmostEqual(r["lucro"], -0.2)
        self.assertAlmostEqual(r["roi_pct"], -10.0)
        self.assertIn("amostra pequena", pac.texto_do_resumo(r))
        self.assertIn("nenhuma gravada", pac.texto_do_resumo(pac.resumo([])))


class TestRobo(unittest.TestCase):
    def test_fechar_usa_o_evento_e_desiste_depois_de_10_dias(self):
        from tools import placar_apostas as rb
        aps = [dict(_a("ml", "1"), evento="e1", inicio_utc="2026-11-14T10:00:00Z"),
               dict(_a("tm", "1", linha=3.5), evento="e1", inicio_utc="2026-11-01T10:00:00Z"),
               dict(_a("ml", "1"), evento="e2", inicio_utc="2026-11-20T10:00:00Z")]
        pedidos = []

        def buscar(slug):
            pedidos.append(slug)
            return EVENTO
        self.assertEqual(rb.fechar(aps, AGORA, buscar), 2)
        self.assertEqual([a.get("resultado") for a in aps], ["ganhou", "sem_resultado", None])
        self.assertEqual(aps[0]["lucro"], 1.0)
        self.assertEqual(pedidos, ["e1"])                   # um pedido por evento; jogo futuro nao pede


if __name__ == "__main__":
    unittest.main()
