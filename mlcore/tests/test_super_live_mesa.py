# -*- coding: utf-8 -*-
"""Mesa do SUPER-LIVE (24/11/2026): rolando agora, abrir/fechar jogos. Identico nos repos."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app_core import super_live_mesa_core as M

AGORA = datetime(2026, 11, 24, 14, 0, tzinfo=timezone.utc)
BASE = {"Team Synapse": "Team Synapse", "Synapse": "Team Synapse", "Angry Birds": "Angry Birds",
        "Team Spirit": "Team Spirit", "PVISION": "PVISION"}


class TestRolandoAgora(unittest.TestCase):

    def test_janela_repetido_e_motivo(self):
        agenda = {"jogos": [
            {"teamA_raw": "Angry Birds", "teamB_raw": "Team Synapse", "inicio_utc": "2026-11-24T13:00:00Z", "bo": 3,
             "league": "EPL"},
            {"teamA_raw": "Team Spirit", "teamB_raw": "PVISION", "inicio_utc": "2026-11-24T20:00:00Z"},   # depois
            {"teamA_raw": "Time Novo", "teamB_raw": "Team Spirit", "inicio_utc": "2026-11-24T14:10:00Z"},  # comeca ja
            {"teamA_raw": "Team Spirit", "teamB_raw": "PVISION", "inicio_utc": "2026-11-24T08:00:00Z"}]}  # antes
        vivo = [{"azul": "Synapse", "vermelho": "Angry Birds", "liga": "EPL"}]           # o mesmo jogo
        js = M.rolando_agora(agenda, vivo, lambda n: BASE.get(n, ""), AGORA)
        self.assertEqual(len(js), 2)
        self.assertEqual((js[0]["a"], js[0]["b"]), ("Angry Birds", "Team Synapse"))
        self.assertIn("ao vivo", js[0]["fonte"])
        novo = [j for j in js if j["a_bruto"] == "Time Novo"][0]
        self.assertIn("Time Novo", novo["motivo"])          # nada some calado


class TestMesa(unittest.TestCase):

    def test_abrir_sem_repetir_limite_e_fechar(self):
        ss = {}
        a = M.abrir(ss, "dota", "Team Spirit", "PVISION")
        self.assertEqual(M.abrir(ss, "dota", "PVISION", "Team Spirit"), a)   # o mesmo jogo nao abre 2x
        for _ in range(M.MAX_JOGOS - 1):
            self.assertIsNotNone(M.abrir(ss, "dota"))
        self.assertIsNone(M.abrir(ss, "dota"))                                # 10 no maximo
        ss[f"slm_dota_{a}_t1"] = "x"
        M.fechar(ss, "dota", a)
        self.assertEqual(len(ss["slm_dota"]), M.MAX_JOGOS - 1)
        self.assertNotIn(f"slm_dota_{a}_t1", ss)                              # fechar limpa o que foi digitado
        self.assertEqual(ss["slm_dota"][0]["rotulo"], "Jogo 2")


if __name__ == "__main__":
    unittest.main()
