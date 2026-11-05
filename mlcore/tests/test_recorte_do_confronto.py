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

    def test_ja_se_enfrentaram_ha_pouco_vale_o_campeonato_entre_eles(self):
        tg = pd.DataFrame([_g("LOS", "T1", "MSI", "2026-07-01", split=None),
                           _g("T1", "LOS", "MSI", "2026-07-01", split=None),
                           _g("LOS", "pain", "CBLOL", "2026-07-20"),
                           _g("T1", "Gen.G", "LCK", "2026-07-21")])
        self.assertEqual(recorte_do_confronto(tg, "LOS", "T1", k), ("MSI", "2026", ""))

    def test_confronto_antigo_nao_prende_o_recorte(self):
        # 05/11/2026: DK x BLG no Worlds caia no EWC de julho (o ultimo confronto entre eles)
        tg = pd.DataFrame([_g("DK", "BLG", "EWC", "2026-07-17", split=None),
                           _g("BLG", "DK", "EWC", "2026-07-17", split=None),
                           _g("DK", "Gen.G", "WLDs", "2026-10-30", split=None),
                           _g("BLG", "HLE", "WLDs", "2026-10-31", split=None)])
        self.assertEqual(recorte_do_confronto(tg, "DK", "BLG", k), ("WLDs", "2026", ""))
        tg2 = pd.DataFrame([_g("LOS", "T1", "MSI", "2026-07-01", split=None),
                            _g("T1", "LOS", "MSI", "2026-07-01", split=None),
                            _g("LOS", "pain", "CBLOL", "2026-09-01"),
                            _g("T1", "Gen.G", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg2, "LOS", "T1", k)[0], SEM_CAMP)

    def test_ligas_diferentes_sem_confronto_fica_sem_campeonato(self):
        tg = pd.DataFrame([_g("LOS", "pain", "CBLOL", "2026-09-01"),
                           _g("T1", "Gen.G", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg, "LOS", "T1", k)[0], SEM_CAMP)

    def test_mesma_liga_continua_a_liga(self):
        tg = pd.DataFrame([_g("T1", "KT", "LCK", "2026-09-01"), _g("Gen.G", "KT", "LCK", "2026-09-02")])
        self.assertEqual(recorte_do_confronto(tg, "T1", "Gen.G", k), ("LCK", "2026", "Split 2"))


if __name__ == "__main__":
    unittest.main()
