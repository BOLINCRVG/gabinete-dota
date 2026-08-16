# -*- coding: utf-8 -*-
"""Régua do Dota: as âncoras da nota NÃO podem divergir da régua do LoL.

A fórmula está duplicada nos dois repositórios de propósito (eles são separados),
então o risco real é uma mudar e a outra não — e as notas deixarem de ser
comparáveis em silêncio. Estes testes travam os pontos de ancoragem.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dota_scorecard import ELITE_LOGLOSS, RANDOM_LOGLOSS, grade_from_logloss  # noqa: E402


class TestReguaDota(unittest.TestCase):
    def test_ancoras_batem_com_a_regua_do_lol(self) -> None:
        self.assertAlmostEqual(RANDOM_LOGLOSS, 0.6931, places=4)
        self.assertAlmostEqual(ELITE_LOGLOSS, 0.55, places=4)
        self.assertAlmostEqual(grade_from_logloss(0.6931), 5.0, places=2)
        self.assertAlmostEqual(grade_from_logloss(0.55), 9.5, places=2)

    def test_pior_que_moeda_fica_abaixo_de_5(self) -> None:
        """O TI 2026 estava em 0.817 — a régua tem que expor isso, não maquiar."""
        self.assertLess(grade_from_logloss(0.817), 5.0)
        self.assertGreaterEqual(grade_from_logloss(0.817), 0.0)

    def test_travada_em_zero_e_dez(self) -> None:
        self.assertEqual(grade_from_logloss(5.0), 0.0)
        self.assertEqual(grade_from_logloss(0.01), 10.0)

    def test_nan_nao_vira_nota(self) -> None:
        self.assertTrue(grade_from_logloss(float("nan")) != grade_from_logloss(float("nan")))

    def test_monotonica(self) -> None:
        """Menos logloss tem que dar nota MAIOR, sempre."""
        vals = [0.50, 0.55, 0.60, 0.6931, 0.75, 0.90]
        notas = [grade_from_logloss(v) for v in vals]
        self.assertEqual(notas, sorted(notas, reverse=True))


if __name__ == "__main__":
    unittest.main()
