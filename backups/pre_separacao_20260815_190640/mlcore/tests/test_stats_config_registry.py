"""Registro central de mercados (stats_config) — trava a equivalência.

Consolidamos SC_METRIC_COL e RESUMO_DEFAULT_LINES pra DERIVAREM de
STATS_MARKET_CONFIG em vez de duplicar. Este teste prova que os valores
derivados são EXATAMENTE os que estavam hardcoded antes (zero mudança de
comportamento) — se alguém editar o registro e quebrar um mapa, falha aqui.
"""
from __future__ import annotations

import unittest

from app_core.stats_config import (
    combined_col_map,
    default_lines_map,
    market_key_by_label,
)

# Valores históricos (hardcoded antes da consolidação) — a referência imutável.
_EXPECTED_COMBINED = {
    "kills": "total_kills",
    "towers": "total_towers",
    "dragons": "total_dragons",
    "barons": "total_nashors",
    "inhibitors": "total_inhibitors",
    "time": "game_time_min",
}
_EXPECTED_DEFAULT_LINES = {
    "kills": [27.5],
    "towers": [11.5],
    "dragons": [4.5],
    "barons": [1.5],
    "inhibitors": [1.5],
    "time": [32.5],
}


class TestRegistryEquivalence(unittest.TestCase):

    def test_combined_col_map_matches_historic(self):
        self.assertEqual(combined_col_map(), _EXPECTED_COMBINED)

    def test_default_lines_map_matches_historic(self):
        self.assertEqual(default_lines_map(), _EXPECTED_DEFAULT_LINES)

    def test_resumo_runtime_derives_from_registry(self):
        from app_core.resumo_runtime_core import RESUMO_DEFAULT_LINES, SC_METRIC_COL
        self.assertEqual(SC_METRIC_COL, _EXPECTED_COMBINED)
        self.assertEqual(RESUMO_DEFAULT_LINES, _EXPECTED_DEFAULT_LINES)

    def test_line_analysis_metric_col_derives_from_registry(self):
        from app_core.line_analysis_tab_core import _MARKET_KEYS, _METRIC_COL
        # _MARKET_KEYS = {label -> metric_key}; _METRIC_COL = {metric_key -> combined_col}
        self.assertEqual(_MARKET_KEYS, market_key_by_label())
        for mk, col in _EXPECTED_COMBINED.items():
            self.assertEqual(_METRIC_COL.get(mk), col)

    def test_all_metric_keys_have_default_line(self):
        self.assertEqual(set(combined_col_map()), set(default_lines_map()))


if __name__ == "__main__":
    unittest.main()
