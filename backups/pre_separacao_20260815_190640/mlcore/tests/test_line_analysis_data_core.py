# -*- coding: utf-8 -*-
"""Testes da camada de dados da Análise de Linha (fatiamento 18/07).

Cobre _augment_with_total_columns (colunas mv2_total_*/spread) e valida que
os nomes seguem re-exportados por line_analysis_tab_core (compat interna +
radar). Ganho colateral: cobertura de uma função antes sem teste próprio.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from app_core.line_analysis_data_core import _augment_with_total_columns, _V3_MARKETS  # noqa: E402


class TestLineAnalysisDataCore(unittest.TestCase):
    def test_augment_totals_and_spreads(self):
        df = pd.DataFrame({"kills_for": [10, 12], "kills_against": [8, 15],
                           "towers_for": [5, 6], "towers_against": [3, 7]})
        out = _augment_with_total_columns(df)
        self.assertEqual(list(out["mv2_total_kills"]), [18, 27])
        self.assertEqual(list(out["mv2_total_towers"]), [8, 13])
        self.assertEqual(list(out["mv2_spread_kills"]), [2.0, -3.0])
        # não muta o df original
        self.assertNotIn("mv2_total_kills", df.columns)

    def test_augment_skips_missing_columns_and_empty(self):
        self.assertTrue(_augment_with_total_columns(pd.DataFrame()).empty)
        df = pd.DataFrame({"kills_for": [10]})  # sem _against
        out = _augment_with_total_columns(df)
        self.assertNotIn("mv2_total_kills", out.columns)  # guard evita coluna parcial

    def test_augment_keeps_nan_not_zero(self):
        # NaN em _against deve manter NaN (não virar 0 falso — bug 2026-07-11)
        df = pd.DataFrame({"kills_for": [10.0], "kills_against": [float("nan")]})
        out = _augment_with_total_columns(df)
        self.assertTrue(pd.isna(out["mv2_total_kills"].iloc[0]))

    def test_v3_markets_reexported_and_shaped(self):
        # re-export por line_analysis_tab_core (compat) aponta o MESMO objeto
        from app_core.line_analysis_tab_core import _V3_MARKETS as reexp
        self.assertIs(reexp, _V3_MARKETS)
        labels = [m["label"] for m in _V3_MARKETS]
        self.assertEqual(labels[0], "Vencedor")
        self.assertIn("ML + Totais", labels)

    def test_augment_reexported_by_tab_core(self):
        from app_core.line_analysis_tab_core import _augment_with_total_columns as reexp
        self.assertIs(reexp, _augment_with_total_columns)

    def test_safe_pct_over_under_and_nan(self):
        from app_core.line_analysis_data_core import _safe_pct
        self.assertEqual(_safe_pct(pd.Series([1, 2, 3, 4]), 2.5, "Over"), 50.0)
        self.assertEqual(_safe_pct(pd.Series([1, 2, 3, 4]), 2.5, "Under"), 50.0)
        self.assertTrue(pd.isna(_safe_pct(pd.Series([], dtype=float), 2.5, "Over")))

    def test_get_metric_col_maps_from_registry(self):
        from app_core.line_analysis_data_core import _get_metric_col
        self.assertEqual(_get_metric_col("Kills"), "total_kills")
        self.assertIsNone(_get_metric_col("MercadoInexistente"))
        # re-export por tab_core aponta o mesmo objeto
        from app_core.line_analysis_tab_core import _get_metric_col as reexp
        self.assertIs(reexp, _get_metric_col)


if __name__ == "__main__":
    unittest.main()
