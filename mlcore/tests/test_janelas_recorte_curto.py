# -*- coding: utf-8 -*-
"""Janelas 5/10/20 com recorte curto (28/10/2026: LYON com 5 jogos no Mundial repetia 5/5/5)."""
import unittest

import pandas as pd

from app_core.market_v3_analysis_core import INTERNATIONAL_LEAGUES, compute_window_stats


def _df(n, liga, base_dia=1, tot=30):
    return pd.DataFrame({"date": pd.date_range(f"2026-0{base_dia}-01", periods=n, freq="D"),
                         "league": liga, "mv2_total_kills": [tot] * n})


class TestJanelas(unittest.TestCase):
    def test_recorte_curto_completa_com_o_ano(self):
        mundial = _df(5, "WLDs", base_dia=9, tot=20)
        ano = pd.concat([_df(30, "LCS", base_dia=1, tot=35), mundial])
        ws = {w.label: w for w in compute_window_stats(mundial, metric_col="mv2_total_kills", line=27.5,
                                                       full_year_df=ano)}
        self.assertEqual(ws["Últimos 5"].n, 5)
        self.assertEqual(ws["Últimos 10"].n, 10)
        self.assertEqual(ws["Últimos 10"].hits_over, 5)       # 5 da LCS (35) + 5 do Mundial (20)
        self.assertEqual(ws["Últimos 20"].n, 20)

    def test_mundial_e_internacional(self):
        self.assertIn("WLDS", INTERNATIONAL_LEAGUES)


if __name__ == "__main__":
    unittest.main()
