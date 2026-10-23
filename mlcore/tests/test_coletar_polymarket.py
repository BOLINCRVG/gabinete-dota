# -*- coding: utf-8 -*-
"""Leitura de partida da Polymarket (23/10/2026), com o formato medido pela sonda."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

try:
    from tools.coletar_polymarket import ler_partida
except ImportError as _sem_ferramenta:
    import unittest as _u
    raise _u.SkipTest("tools.coletar_polymarket nao existe neste repositorio: " + str(_sem_ferramenta))

ANTES = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def _ev(titulo, bid=0.73, ask=0.74, inicio="2026-10-23 23:00:00+00", tipo="moneyline"):
    return {"title": titulo, "slug": "x", "markets": [{
        "sportsMarketType": tipo, "outcomes": '["LOS", "T1"]', "outcomePrices": '["0.2", "0.8"]',
        "bestBid": bid, "bestAsk": ask, "gameStartTime": inicio, "liquidity": "1000", "volume": "50"}]}


class TestLerPartida(unittest.TestCase):

    def test_le_times_bo_torneio_e_meio_do_livro(self):
        p = ler_partida(_ev("LoL: LOS vs T1 (BO1) - Worlds Group Stage", 0.10, 0.12), ANTES)
        self.assertEqual((p["time_a"], p["time_b"], p["bo"]), ("LOS", "T1", 1))
        self.assertEqual(p["torneio"], "Worlds Group Stage")
        self.assertAlmostEqual(p["p_a"], 0.11)
        self.assertAlmostEqual(p["p_a"] + p["p_b"], 1.0)
        self.assertAlmostEqual(p["spread"], 0.02)

    def test_sem_livro_usa_o_ultimo_preco(self):
        p = ler_partida(_ev("LoL: LOS vs T1 (BO3) - X", None, None), ANTES)
        self.assertEqual((p["p_a"], p["fonte_preco"], p["bo"]), (0.2, "ultimo_preco", 3))

    def test_jogo_que_ja_comecou_fica_de_fora(self):
        self.assertIsNone(ler_partida(_ev("LoL: LOS vs T1 (BO1) - X", inicio="2026-10-23 10:00:00+00"), ANTES))

    def test_o_que_nao_e_partida_fica_de_fora(self):
        self.assertIsNone(ler_partida(_ev("Worlds 2026: Winner"), ANTES))
        self.assertIsNone(ler_partida(_ev("LoL: A vs B (BO1) - X", tipo="child_moneyline"), ANTES))


if __name__ == "__main__":
    unittest.main()
