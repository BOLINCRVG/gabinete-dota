# -*- coding: utf-8 -*-
"""Nota por jogador no Dota (27/10/2026): mistura, escalacoes e a etapa do motor."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from app_core.nota_jogadores_core import p_com_jogadores  # noqa: E402
import escalacoes_dota as escd  # noqa: E402

DADOS = {"parametros": {"w_jogador": 0.5, "temperatura": 1.1},
         "times": {"A": {"forca": 100.0, "elenco": [1, 2, 3, 4, 5]}, "B": {"forca": 0.0, "elenco": []}}}


class TestNotaJogadores(unittest.TestCase):
    def test_mistura_puxa_para_os_jogadores(self):
        r = p_com_jogadores("A", "B", 0.5, DADOS)
        self.assertAlmostEqual(r["p_jogador"], 1 / (1 + 10 ** (-100 / 400)))
        self.assertGreater(r["p_final"], 0.5)          # modelo 50%, jogadores ~64%
        self.assertLess(r["p_final"], r["p_jogador"])

    def test_time_fora_do_arquivo_nao_mexe(self):
        self.assertIsNone(p_com_jogadores("A", "Z", 0.6, DADOS))

    def test_simetria(self):
        a = p_com_jogadores("A", "B", 0.7, DADOS)["p_final"]
        b = p_com_jogadores("B", "A", 0.3, DADOS)["p_final"]
        self.assertAlmostEqual(a + b, 1.0, places=9)

    def test_escalacao_so_com_cinco(self):
        d = {"radiant_team_id": 10, "dire_team_id": 20,
             "players": [{"account_id": i, "isRadiant": True} for i in range(1, 6)]
                        + [{"account_id": i, "isRadiant": False} for i in range(6, 10)]}
        self.assertEqual(escd.de_payload(d), {10: frozenset({1, 2, 3, 4, 5})})

    def test_escalacao_pelo_player_slot_da_opendota(self):
        d = {"radiant_team_id": 10, "dire_team_id": 20,
             "players": [{"account_id": i, "player_slot": s} for i, s in
                         zip(range(1, 11), [0, 1, 2, 3, 4, 128, 129, 130, 131, 132])]}
        self.assertEqual(set(escd.de_payload(d)), {10, 20})


if __name__ == "__main__":
    unittest.main()
