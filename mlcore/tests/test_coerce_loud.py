# -*- coding: utf-8 -*-
"""coerce_loud: o preenchimento de dado ilegível para de ser silencioso.

Varredura de 16/08/2026: o par `to_numeric(errors="coerce").fillna(N)` aparece 99
vezes no projeto e é silencioso por construção — o que não dá pra ler vira NaN, e
o NaN vira um número plausível. Na coluna de probabilidade que alimenta o ajuste
do calibrador (fit_samples_core), isso significa que "a coluna mudou de nome" se
transforma em "todo mundo com 0.5" e o calibrador é ajustado em cima de lixo.

O default neutro segue certo para um caso isolado. O que faltava era perceber
quando ele deixou de ser isolado.
"""
from __future__ import annotations

import unittest

import pandas as pd

from mlcore.utils import coerce_loud


class TestCoerceLoud(unittest.TestCase):
    def setUp(self) -> None:
        self.avisos: list[str] = []

    def test_converte_igual_ao_padrao_antigo(self) -> None:
        s = coerce_loud(pd.Series(["0.7", "0.2", None]), nome="p", default=0.5,
                        avisar=self.avisos.append)
        self.assertEqual(list(s), [0.7, 0.2, 0.5])

    def test_caso_isolado_nao_avisa(self) -> None:
        """1 em 10 é ruído normal de dado — avisar aqui viraria barulho ignorado."""
        vals = ["0.5"] * 9 + ["lixo"]
        coerce_loud(pd.Series(vals), nome="p", default=0.5, avisar=self.avisos.append)
        self.assertEqual(self.avisos, [])

    def test_substituicao_em_massa_grita(self) -> None:
        """O cenário que motivou: a coluna inteira ilegível."""
        coerce_loud(pd.Series(["lixo"] * 10), nome="p (OOF)", default=0.5,
                    avisar=self.avisos.append)
        self.assertEqual(len(self.avisos), 1)
        self.assertIn("p (OOF)", self.avisos[0])
        self.assertIn("100%", self.avisos[0])

    def test_o_aviso_diz_quanto_e_o_que(self) -> None:
        vals = ["0.5"] * 5 + ["lixo"] * 5
        coerce_loud(pd.Series(vals), nome="coluna_x", default=0.5, avisar=self.avisos.append)
        self.assertIn("5/10", self.avisos[0])
        self.assertIn("coluna_x", self.avisos[0])

    def test_serie_vazia_nao_quebra(self) -> None:
        s = coerce_loud(pd.Series([], dtype=object), nome="p", default=0.5,
                        avisar=self.avisos.append)
        self.assertEqual(len(s), 0)
        self.assertEqual(self.avisos, [])

    def test_o_dado_continua_passando(self) -> None:
        """Grita, mas NÃO derruba: quebrar o treino seria pior que o silêncio."""
        s = coerce_loud(pd.Series(["lixo"] * 4), nome="p", default=0.5,
                        avisar=self.avisos.append)
        self.assertEqual(list(s), [0.5] * 4)


if __name__ == "__main__":
    unittest.main()
