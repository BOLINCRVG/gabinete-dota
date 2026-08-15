# -*- coding: utf-8 -*-
"""Testes dos helpers de avaliação/holdout do fit de forma+meta (12% em 19/07).

O mais importante aqui é `_stratified_time_holdout`: ele garante que o
treino usa SÓ o passado e o teste SÓ o futuro. Se esse split vazar, a nota
do modelo vira ficção (é exatamente o diagnóstico do "77% de acurácia" do
amigo dele — leakage). Por isso vale teste dedicado.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from mlcore.fit_form_meta_core import (  # noqa: E402
    _evaluate,
    _logloss_safe,
    _stratified_time_holdout,
)


class TestHoldoutTemporal(unittest.TestCase):
    """Treino = passado, teste = futuro. Sem interseção, sem buraco."""

    def _amostras(self, n: int) -> list:
        return [object() for _ in range(n)]  # o split só usa len()

    def test_treino_vem_todo_antes_do_teste(self):
        tr, te = _stratified_time_holdout(self._amostras(200), holdout_frac=0.2)
        self.assertTrue(tr and te)
        self.assertLess(max(tr), min(te), "treino invadiu o futuro = LEAKAGE")

    def test_sem_interseccao_e_sem_buraco(self):
        n = 500
        tr, te = _stratified_time_holdout(self._amostras(n), holdout_frac=0.25)
        self.assertEqual(set(tr) & set(te), set(), "amostra em treino E teste")
        self.assertEqual(sorted(tr + te), list(range(n)), "índices sumiram/duplicaram")

    def test_amostra_pequena_nao_gera_teste(self):
        """<30 amostras: não dá para separar teste com sentido — devolve tudo
        no treino (melhor não medir do que medir mentira)."""
        tr, te = _stratified_time_holdout(self._amostras(25))
        self.assertEqual(len(tr), 25)
        self.assertEqual(te, [])

    def test_teste_tem_tamanho_minimo(self):
        tr, te = _stratified_time_holdout(self._amostras(40), holdout_frac=0.01)
        self.assertGreaterEqual(len(te), 10, "teste minúsculo não mede nada")

    def test_treino_tem_tamanho_minimo(self):
        tr, te = _stratified_time_holdout(self._amostras(40), holdout_frac=0.99)
        self.assertGreaterEqual(len(tr), 20, "treino minúsculo não aprende nada")

    def test_fracao_maior_gera_teste_maior(self):
        _, te_p = _stratified_time_holdout(self._amostras(1000), holdout_frac=0.1)
        _, te_g = _stratified_time_holdout(self._amostras(1000), holdout_frac=0.3)
        self.assertLess(len(te_p), len(te_g))

    def test_lista_vazia_nao_quebra(self):
        tr, te = _stratified_time_holdout([], holdout_frac=0.2)
        self.assertEqual((tr, te), ([], []))


class TestLoglossSafe(unittest.TestCase):
    def test_predicao_perfeita_da_logloss_baixa(self):
        v = _logloss_safe([1, 0, 1, 0], [0.99, 0.01, 0.99, 0.01])
        self.assertLess(v, 0.05)

    def test_predicao_pessima_da_logloss_alta(self):
        boa = _logloss_safe([1, 0], [0.9, 0.1])
        ruim = _logloss_safe([1, 0], [0.1, 0.9])
        self.assertGreater(ruim, boa)

    def test_moeda_fica_perto_de_ln2(self):
        v = _logloss_safe([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(v, math.log(2), places=6)

    def test_entrada_invalida_vira_nan_em_vez_de_estourar(self):
        """Contrato do 'safe': nunca levanta — quem chama decide o que fazer."""
        v = _logloss_safe([1, 0], [])          # tamanhos incompatíveis
        self.assertTrue(math.isnan(v) or math.isfinite(v))


class TestEvaluate(unittest.TestCase):
    def test_retorna_logloss_brier_e_n(self):
        out = _evaluate(np.array([0.9, 0.2, 0.8]), np.array([1, 0, 1]))
        self.assertEqual(out["n"], 3)
        self.assertTrue(math.isfinite(out["logloss"]))
        self.assertTrue(math.isfinite(out["brier"]))

    def test_probabilidade_extrema_e_clipada_sem_estourar(self):
        """p=0 com y=1 daria logloss infinito; o clip protege."""
        out = _evaluate(np.array([0.0, 1.0]), np.array([1, 0]))
        self.assertTrue(math.isfinite(out["logloss"]), "clip de probabilidade falhou")

    def test_predicao_melhor_tem_brier_menor(self):
        boa = _evaluate(np.array([0.9, 0.1]), np.array([1, 0]))
        ruim = _evaluate(np.array([0.6, 0.4]), np.array([1, 0]))
        self.assertLess(boa["brier"], ruim["brier"])

    def test_vazio_nao_quebra(self):
        out = _evaluate(np.array([]), np.array([]))
        self.assertEqual(out["n"], 0)


if __name__ == "__main__":
    unittest.main()
