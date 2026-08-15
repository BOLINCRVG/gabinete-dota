from __future__ import annotations

import unittest

import pandas as pd

from app_core.display_tables_core import df_contract_view, display_ml_totals_table, display_ou_table


class TestDisplayTablesCore(unittest.TestCase):
    def test_df_contract_view_adds_missing(self) -> None:
        df = pd.DataFrame([{"a": 1}])
        warns: list[str] = []
        out = df_contract_view(
            df,
            ["a", "b"],
            context="ctx",
            fill_value="-",
            warning_fn=lambda s: warns.append(str(s)),
        )
        self.assertEqual(list(out.columns), ["a", "b"])
        self.assertEqual(str(out.iloc[0]["b"]), "-")
        self.assertTrue(len(warns) == 1)

    def test_display_ou_table(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "line": 25.5,
                    "p_over": 0.52,
                    "odd_over": 1.92,
                    "p_under": 0.48,
                    "odd_under": 2.08,
                }
            ]
        )
        out = display_ou_table(
            df,
            df_contract_view_fn=lambda *args, **kwargs: df_contract_view(*args, **kwargs),
            format_pct_fn=lambda p: f"{100.0*float(p):.1f}%",
            format_num_fn=lambda x, n: f"{float(x):.{int(n)}f}",
        )
        self.assertEqual(list(out.columns), ["Linha", "Acima (%)", "Odd Acima", "Abaixo (%)", "Odd Abaixo"])
        self.assertEqual(str(out.iloc[0]["Odd Acima"]), "1.92")

    def test_display_ml_totals_table(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "line": 25.5,
                    "p_ML+OVER(map/map)": 0.21,
                    "odd_ML+OVER(map/map)": 4.76,
                    "p_ML+UNDER(map/map)": 0.19,
                    "odd_ML+UNDER(map/map)": 5.26,
                }
            ]
        )
        out = display_ml_totals_table(
            df,
            df_contract_view_fn=lambda *args, **kwargs: df_contract_view(*args, **kwargs),
            format_pct_fn=lambda p: f"{100.0*float(p):.1f}%",
            format_num_fn=lambda x, n: f"{float(x):.{int(n)}f}",
            team_label="TA",
        )
        self.assertEqual(list(out.columns), ["Time", "Linha", "ML & Acima (%)", "Odd ML & Acima", "ML & Abaixo (%)", "Odd ML & Abaixo"])
        self.assertEqual(str(out.iloc[0]["Time"]), "TA")


if __name__ == "__main__":
    unittest.main()

