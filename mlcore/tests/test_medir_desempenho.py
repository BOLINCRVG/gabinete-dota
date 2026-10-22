# -*- coding: utf-8 -*-
"""tools/medir_desempenho.py (22/10/2026): a regra de regressao de tempo.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import unittest

from tools.medir_desempenho import regrediu


def _hist(v, n=10):
    return [{"carregar_s": v, "prever_s": v / 2, "suite_s": 100.0} for _ in range(n)]


class TestRegressaoDeTempo(unittest.TestCase):

    def test_estavel_nao_acusa(self):
        self.assertEqual(regrediu({"carregar_s": 2.1, "prever_s": 1.0, "suite_s": 110.0}, _hist(2.0)), [])

    def test_dobrou_acima_do_piso_acusa(self):
        p = regrediu({"carregar_s": 5.0, "prever_s": 1.0, "suite_s": 100.0}, _hist(2.0))
        self.assertEqual(len(p), 1)
        self.assertIn("carregar_s", p[0])

    def test_dobrou_mas_e_ruido_de_milissegundo_nao_acusa(self):
        self.assertEqual(regrediu({"carregar_s": 0.09, "prever_s": 0.05, "suite_s": 100.0}, _hist(0.03)), [])

    def test_historico_curto_nao_decide(self):
        self.assertEqual(regrediu({"carregar_s": 50.0}, _hist(2.0, n=3)), [])


if __name__ == "__main__":
    unittest.main()
