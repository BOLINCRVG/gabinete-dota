# -*- coding: utf-8 -*-
"""Desenho da aba Modelo x Mercado no tema da Linha justa (23/10/2026)."""
from __future__ import annotations

import unittest

from app_core.mercado_vencedor_core import ROTULO_SELO, comparar_vencedor
from app_core.modelo_mercado_core import aposta
from app_core.modelo_mercado_html_core import _hora_br, html_destaques, html_vencedor


class TestHtml(unittest.TestCase):

    def test_cartao_tem_odds_da_casa_e_do_modelo(self):
        h = html_destaques([{"casa": "LOS", "fora": "T1", "liga": "Worlds", "lado": "home",
                             "tipo": "spread", "pontos": 14.5, "odd_casa": 1.28, "odd_modelo": 1.07,
                             "vantagem_pct": 19.3, "nome_pt": "Handicap",
                             "inicio_utc": "2026-10-23T23:00:00Z"}], aposta)
        self.assertIn("1.28", h)
        self.assertIn("1.07", h)
        self.assertIn("+19,3%", h)
        self.assertIn("23/10 20:00", h)

    def test_hora_de_brasilia(self):
        self.assertEqual(_hora_br("2026-10-24T00:00:00Z"), "23/10 21:00")

    def test_vencedor_tres_colunas_em_odds(self):
        r = comparar_vencedor(0.85, pin_odd_a=1.33, pin_odd_b=3.38,
                              poly={"p_a": 0.735, "spread": 0.01, "liquidez": 3e5})
        h = html_vencedor(r, time_a="Team Liquid", time_b="CTBC", bo=1,
                          rotulos=ROTULO_SELO, fontes_faltando=[])
        for x in ("c-mod", "c-pin", "c-poly", "1.18", "1.33", "verde"):
            self.assertIn(x, h)

    def test_sem_destaques_diz(self):
        self.assertIn("Nenhuma aposta", html_destaques([], aposta))


if __name__ == "__main__":
    unittest.main()
