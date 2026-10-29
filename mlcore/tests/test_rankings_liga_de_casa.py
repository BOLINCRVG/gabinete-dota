"""Ranking: liga de casa e o filtro de minimo de jogos (29/10/2026)."""
import unittest

import pandas as pd

from app_core import rankings_tab_core as rk


class TestRanking(unittest.TestCase):
    def test_liga_de_casa_e_a_mais_jogada(self):
        tg = pd.DataFrame({
            "team": ["KT Rolster"] * 30 + ["KT Rolster"] * 4 + ["Gen.G"] * 10,
            "league": ["LCK"] * 30 + ["DCGI"] * 4 + ["LCK"] * 10,
            "date": pd.date_range("2026-01-01", periods=44, freq="D"),
        })
        casa = rk.liga_de_casa(tg)
        self.assertEqual(casa["ktrolster"], "LCK")
        df = pd.DataFrame({"team": ["KT Rolster", "Gen.G", "Sem Jogo"], "league": ["DCGI", "LCK", "X"]})
        out = rk._com_liga_de_casa(df, casa)
        self.assertEqual(list(out["league"]), ["LCK", "LCK", "X"])

    def test_chave_normalizada(self):
        self.assertEqual(rk._norm_team_key("Top Esports"), "topesports")


if __name__ == "__main__":
    unittest.main()
