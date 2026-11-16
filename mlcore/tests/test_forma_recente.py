# -*- coding: utf-8 -*-
"""Forma recente no preco do mapa (16/11/2026). Arquivo identico nos dois repositorios."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from app_core import forma_recente_core as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import forma_recente as T  # noqa: E402

DOC = {"peso": 0.1, "times": {"A": {"forma": 3.0, "resultados": "VVVVV"},
                              "B": {"forma": -1.0, "resultados": "DDDVV"}}}


class TestFormaRecente(unittest.TestCase):
    def test_ajuste_no_logit(self):
        r = F.ajustar(0.587, "A", "B", DOC)
        z = math.log(0.587 / 0.413) + 0.1 * 4.0
        self.assertAlmostEqual(r["p"], 1 / (1 + math.exp(-z)))
        self.assertEqual(r["motivo"], "")

    def test_simetrico(self):
        a = F.ajustar(0.6, "A", "B", DOC)["p"]
        b = F.ajustar(0.4, "B", "A", DOC)["p"]
        self.assertAlmostEqual(a, 1 - b)

    def test_time_sem_forma_nao_mexe_e_explica(self):
        r = F.ajustar(0.6, "A", "Novo", DOC)
        self.assertEqual(r["p"], 0.6)
        self.assertIn("Novo", r["motivo"])
        self.assertEqual(F.ajustar(0.6, "A", "B", None)["p"], 0.6)

    def test_calcular_usa_so_os_ultimos_5(self):
        jogos = [("2026-01-%02d" % i, "A", "B", 1, 0.5) for i in range(1, 9)]
        t = T.calcular(jogos)
        self.assertEqual(t["A"]["mapas"], 5)
        self.assertAlmostEqual(t["A"]["forma"], 2.5)
        self.assertAlmostEqual(t["B"]["forma"], -2.5)
        self.assertEqual(t["B"]["resultados"], "DDDDD")


if __name__ == "__main__":
    unittest.main()
