# -*- coding: utf-8 -*-
"""Recorte inicial da Analise de Linha (26/10/2026): o campeonato do confronto.
Identico nos 2 repos."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.line_analysis_tab_core import SEM_CAMP, recorte_do_confronto


def _g(team, opp, liga, data, ano=2026, split="Split 2"):
    return {"team": team, "opponent": opp, "league": liga, "date": data, "year": ano, "split": split}


def k(x):
    return str(x).strip().lower()


class TestRecorte(unittest.TestCase):

    def test_ja_se_enfrentaram_vale_o_campeonato_mais_recente_entre_eles(self):
        tg = pd.DataFrame([_g("LOS", "T1", "MSI", "2026-07-01", split=None),
                           _g("T1", "LOS", "MSI", "2026-07-01", split=None),
                           _g("LOS", "pain", "CBLOL", "2026-09-01"),
                           _g("T1", "Gen.G", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg, "LOS", "T1", k), ("MSI", "2026", ""))

    def test_ligas_diferentes_sem_confronto_fica_sem_campeonato(self):
        tg = pd.DataFrame([_g("LOS", "pain", "CBLOL", "2026-09-01"),
                           _g("T1", "Gen.G", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg, "LOS", "T1", k)[0], SEM_CAMP)

    def test_mesma_liga_continua_a_liga(self):
        tg = pd.DataFrame([_g("T1", "KT", "LCK", "2026-09-01"), _g("Gen.G", "KT", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg, "T1", "Gen.G", k), ("LCK", "2026", "Split 2"))


if __name__ == "__main__":
    unittest.main()
