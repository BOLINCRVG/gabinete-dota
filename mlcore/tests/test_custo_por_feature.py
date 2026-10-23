# -*- coding: utf-8 -*-
"""Custo por camada (23/10/2026): roda na suite, entao o robo mede todo dia e
falha se alguma camada passar do teto. Arquivo identico nos dois repositorios."""
from __future__ import annotations

import unittest
from pathlib import Path

from tools.custo_por_feature import TETO_MS, medir

RAIZ = Path(__file__).resolve().parents[2]


@unittest.skipUnless((RAIZ / "ml_artifact.json").exists() and any((RAIZ / "CSV_DIARIO").glob("*.csv")),
                     "sem modelo ou sem CSV")
class TestCustoPorCamada(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = medir(RAIZ)

    def test_mede_ao_menos_o_time_e_a_fusao(self):
        self.assertIn("time", self.res["ms_por_previsao"])
        self.assertIn("etapa_fusao", self.res["ms_por_previsao"])
        self.assertIn("ml_artifact.json", self.res["mb_por_bloco"])

    def test_nenhuma_camada_acima_do_teto(self):
        self.assertEqual(self.res["acima_do_teto"], [],
                         f"teto {TETO_MS} ms por previsao: {self.res['ms_por_previsao']}")


if __name__ == "__main__":
    unittest.main()
