# -*- coding: utf-8 -*-
"""O metodo dele e a forma do dia no Dota (15/11/2026). Arquivo identico nos dois repositorios."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core import series_tab_core as stc
from app_core import sinais_linha_core as sl


class TestTabelaDoDota(unittest.TestCase):
    def test_formato(self):
        self.assertEqual(sorted(sl.METRICAS_DOTA), ["barons", "inhibitors", "kills", "time", "towers"])
        for met, cfg in sl.METRICAS_DOTA.items():
            self.assertEqual(len(cfg["pesos"]), 7, met)
            self.assertTrue(0.0 <= cfg["ajuste"] <= 1.0, met)
            for lado in ("over", "under"):
                self.assertEqual(sorted(cfg["bateu"][lado]), [0, 1, 2, 3, 4], met)
        self.assertTrue(sl.METRICAS_DOTA["barons"]["instavel"] and sl.METRICAS_DOTA["inhibitors"]["instavel"])

    def test_cada_esporte_usa_a_sua(self):
        if sl.E_DOTA:
            self.assertEqual(sl.METRICAS["kills"]["ajuste"], 0.15)
            self.assertNotIn("dragons", sl.METRICAS)
            self.assertNotIn("dragons", sl.COLUNA_PARA_METRICA.values())
        else:
            self.assertIn("dragons", sl.METRICAS)
            self.assertEqual(sl.METRICAS["kills"]["ajuste"], 0.46)

    def test_motor_com_a_tabela_do_dota(self):
        # dois times, 12 jogos cada, totais de kills conhecidos: a fair sai e os passos contam
        linhas = []
        for i in range(12):
            for time, adv, kf, ka in (("A", "X", 30 + i % 3, 28), ("B", "Y", 31, 27 + i % 2)):
                linhas.append({"gameid": f"g{time}{i}", "date": f"2026-10-{i + 1:02d}", "team": time, "opponent": adv,
                               "league": "L", "split": "S", "map_number": 1, "kills_for": kf, "kills_against": ka})
        tg = pd.DataFrame(linhas)
        antigo = sl.METRICAS
        try:
            sl.METRICAS = dict(sl.METRICAS_DOTA)
            sn = sl.sinais(tg.assign(tend_kills=0.0), "A", "B", 50.5, metrica="kills")
        finally:
            sl.METRICAS = antigo
        self.assertIsNotNone(sn)
        self.assertGreater(sn["p_over"], 0.6)                 # todos os jogos passaram de 50,5
        self.assertEqual(sl.passam(sn, "over"), [True, True, True, True])


class TestFormaDoDia(unittest.TestCase):
    def test_kappa_por_formato(self):
        self.assertEqual(stc.kappa_do_bo(3), 3.0)
        self.assertEqual(stc.kappa_do_bo(5), None if stc.E_DOTA else 3.0)
        self.assertIsNone(stc.kappa_do_bo("x"))

    def test_bo3_tem_mais_varrida_que_independente(self):
        ind, forma = stc.score_probs(0.65, 3, kappa=None), stc.score_probs(0.65, 3)
        self.assertGreater(forma["A_2-0"], ind["A_2-0"])
        self.assertAlmostEqual(sum(forma.values()), 1.0, places=9)

    def test_bo5_do_dota_segue_independente(self):
        if stc.E_DOTA:
            self.assertAlmostEqual(stc.score_probs(0.65, 5)["A_3-0"], 0.65 ** 3)


if __name__ == "__main__":
    unittest.main()
