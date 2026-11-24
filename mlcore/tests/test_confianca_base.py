# -*- coding: utf-8 -*-
"""Faixa de confianca da base + status da coleta (24/11/2026). Identico nos repos."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app_core import confianca_base_core as C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import status_coleta as S  # noqa: E402

AGORA = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
ULT = {"quando": datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc), "a": "Team Spirit", "b": "PVISION", "liga": "X"}


class TestVeredito(unittest.TestCase):

    def test_coleta_ok_recente(self):
        n, t = C.veredito("dota", ULT, {"resultado": "success", "ultimo_sucesso": "2026-11-24T10:00:00+00:00"}, AGORA)
        self.assertEqual(n, "ok")
        self.assertIn("24/11 06:00", t)            # horario de Brasilia
        self.assertIn("Team Spirit x PVISION", t)

    def test_falha_aparece_com_motivo(self):
        n, t = C.veredito("dota", ULT, {"resultado": "failure", "quando": "2026-11-23T23:26:00+00:00",
                                        "motivo": "contrato reprovou", "ultimo_sucesso": "2026-11-19T20:06:00+00:00"}, AGORA)
        self.assertEqual(n, "erro")
        self.assertIn("contrato reprovou", t)
        self.assertIn("19/11 17:06", t)

    def test_sucesso_velho_vira_atencao_e_sem_status_tambem(self):
        n, _ = C.veredito("dota", ULT, {"resultado": "success", "ultimo_sucesso": "2026-11-22T10:00:00+00:00"}, AGORA)
        self.assertEqual(n, "atencao")
        self.assertEqual(C.veredito("dota", ULT, None, AGORA)[0], "atencao")


class TestComparar(unittest.TestCase):

    def test_acha_liga_e_time_que_perderam_jogos(self):
        def linhas(jogos):
            return pd.DataFrame([{"gameid": g, "position": "team", "league": lg, "teamname": tm}
                                 for g, lg, a, b in jogos for tm in (a, b)])
        velho = linhas([("1", "EPL", "A", "B"), ("2", "EPL", "A", "C"), ("3", "PGL", "D", "E")])
        novo = linhas([("3", "PGL", "D", "E")])
        r = S.comparar(novo, velho)
        self.assertEqual((r["jogos_antes"], r["jogos_agora"], r["jogos_que_sumiram"]), (3, 1, 2))
        self.assertEqual(r["perdas_por_liga"], {"EPL": -2})
        self.assertEqual(r["perdas_por_time"]["A"], -2)


if __name__ == "__main__":
    unittest.main()
