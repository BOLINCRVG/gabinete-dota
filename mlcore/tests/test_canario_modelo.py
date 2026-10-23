# -*- coding: utf-8 -*-
"""Canario (23/10/2026): a decisao barra o modelo que mexe demais e deixa passar
o retreino normal. Arquivo identico nos dois repositorios."""
from __future__ import annotations

import unittest

from tools.canario_modelo import TETO_FATIA_GRANDE, TETO_GRANDE, TETO_MEDIANA, decidir


class TestDecisaoDoCanario(unittest.TestCase):

    def test_retreino_normal_passa(self):
        # o pior dia medido no historico: mediana ~0,013 e ~2% dos jogos > 0,15
        dps = [0.013] * 98 + [0.30, 0.25]
        self.assertTrue(decidir(dps)["ok"])

    def test_modelo_que_vira_tudo_e_barrado(self):
        self.assertFalse(decidir([TETO_MEDIANA + 0.02] * 100)["ok"])

    def test_muitos_jogos_virando_e_barrado(self):
        n_grandes = int(100 * TETO_FATIA_GRANDE) + 5
        dps = [0.0] * (100 - n_grandes) + [TETO_GRANDE + 0.1] * n_grandes
        self.assertFalse(decidir(dps)["ok"])

    def test_sem_jogos_nao_barra(self):
        self.assertTrue(decidir([])["ok"])


if __name__ == "__main__":
    unittest.main()
