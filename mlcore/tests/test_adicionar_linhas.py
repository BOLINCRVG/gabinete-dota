# -*- coding: utf-8 -*-
"""Callback do "Adicionar linha(s)" da Analise de Linha (26/10/2026).

Ele: "quando eu coloco o valor 30.5 aqui, ele nao adiciona". Clicado com o app
ainda processando, o "if submitted" perdia o clique; o callback nao perde.
Reproduzido num Chrome de verdade (1 linha -> 1 antes; 2 com o callback).
"""
import unittest

from app_core.line_analysis_kills_core import adicionar_linhas


class TestAdicionarLinhas(unittest.TestCase):
    def test_soma_a_linha_e_ordena(self):
        ss = {"L": [29.5], "T": "30.5"}
        adicionar_linhas(ss, "L", "T", False)
        self.assertEqual(ss["L"], [29.5, 30.5])

    def test_varias_de_uma_vez_e_virgula_decimal(self):
        ss = {"L": [29.5], "T": "27,5, 31.5"}
        adicionar_linhas(ss, "L", "T", False)
        self.assertEqual(ss["L"], [27.5, 29.5, 31.5])

    def test_repetida_nao_duplica(self):
        ss = {"L": [29.5], "T": "29.5"}
        adicionar_linhas(ss, "L", "T", False)
        self.assertEqual(ss["L"], [29.5])

    def test_caixa_vazia_nao_mexe(self):
        ss = {"L": [29.5], "T": "  "}
        adicionar_linhas(ss, "L", "T", False)
        self.assertEqual(ss["L"], [29.5])

    def test_tempo_aceita_minutos_e_segundos(self):
        ss = {"L": [31.5], "T": "33:30"}
        adicionar_linhas(ss, "L", "T", True)
        self.assertEqual(ss["L"], [31.5, 33.5])


if __name__ == "__main__":
    unittest.main()
