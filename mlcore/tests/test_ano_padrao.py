# -*- coding: utf-8 -*-
"""Ano padrao da analise de linhas (23/10/2026): o do calendario, nao o maior.
O Oracle's Elixir marca pre-temporada como o ano seguinte. Identico nos 2 repos."""
from __future__ import annotations

import unittest
from datetime import date

from app_core.line_analysis_tab_core import ano_padrao


class TestAnoPadrao(unittest.TestCase):

    def test_ano_corrente_vence_o_ano_seguinte_de_pre_temporada(self):
        self.assertEqual(ano_padrao(["2025", "2026", "2027"], date(2026, 10, 23)), "2026")

    def test_sem_o_ano_corrente_fica_o_mais_recente(self):
        self.assertEqual(ano_padrao(["2025", "2026"], date(2027, 1, 5)), "2026")
        self.assertEqual(ano_padrao(["2027"], date(2026, 10, 23)), "2027")


if __name__ == "__main__":
    unittest.main()
