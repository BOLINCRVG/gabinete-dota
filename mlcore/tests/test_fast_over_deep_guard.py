# -*- coding: utf-8 -*-
"""Trava fit-rápido-sobre-profundo (2026-07-21): causa raiz do incidente que
se repetia — a manutenção automática rodava um fit RÁPIDO por cima do fit
PROFUNDO promovido, rebaixando os params (26515 -> 2500 amostras, 17 ligas ->
0 no players_context).

Antes a blindagem só fazia BACKUP e deixava o overwrite acontecer. Agora BLOQUEIA.
Estes testes travam a decisão pura para nunca regredir.
"""
from __future__ import annotations

import unittest

from app_core.csv_maintenance_runner_core import (
    params_json_is_deep_fit,
    should_block_fast_over_deep,
)


class ParamsJsonIsDeepFitTests(unittest.TestCase):
    def test_com_settings_by_league_e_profundo(self):
        data = {"players_context_fit": {"settings_by_league": {"LCK": {}, "LPL": {}}}}
        deep, n = params_json_is_deep_fit(data)
        self.assertTrue(deep)
        self.assertEqual(n, 2)

    def test_sem_segmentacao_nao_e_profundo(self):
        deep, n = params_json_is_deep_fit({"players_context_fit": {"settings_by_league": {}}})
        self.assertFalse(deep)
        self.assertEqual(n, 0)

    def test_sem_players_context_nao_e_profundo(self):
        deep, n = params_json_is_deep_fit({"samples_total": 2500})
        self.assertFalse(deep)
        self.assertEqual(n, 0)

    def test_entrada_invalida_nao_quebra(self):
        for bad in (None, "lixo", 42, []):
            deep, n = params_json_is_deep_fit(bad)
            self.assertFalse(deep)
            self.assertEqual(n, 0)


class ShouldBlockFastOverDeepTests(unittest.TestCase):
    def test_o_incidente_e_bloqueado(self):
        # fast por cima de profundo -> BLOQUEIA (o caso real de 21/07 12:42)
        self.assertTrue(should_block_fast_over_deep("fast", params_is_deep=True))

    def test_full_sobre_profundo_passa(self):
        # um refit real (full) NUNCA e bloqueado
        self.assertFalse(should_block_fast_over_deep("full", params_is_deep=True))

    def test_fast_sem_profundo_passa(self):
        # primeiro fit / params ainda raso -> fast pode rodar
        self.assertFalse(should_block_fast_over_deep("fast", params_is_deep=False))

    def test_case_e_espacos_tolerados(self):
        self.assertTrue(should_block_fast_over_deep("  FAST  ", params_is_deep=True))

    def test_valor_vazio_nao_bloqueia(self):
        self.assertFalse(should_block_fast_over_deep("", params_is_deep=True))
        self.assertFalse(should_block_fast_over_deep(None, params_is_deep=True))


if __name__ == "__main__":
    unittest.main()
