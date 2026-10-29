"""A logica das apostas do Resumo pelo metodo dele (29/10/2026)."""
import unittest

import pandas as pd

from app_core import resumo_metodo_core as rm


class TestResumoMetodo(unittest.TestCase):
    def test_veredito_e_faixa(self):
        r = {"_p_metodo": 0.63, "_passos": 3}
        self.assertEqual(rm.veredito_do_metodo(r, 5.0), "ENTRA")
        self.assertEqual(rm.faixa(r, 5.0), 0)
        self.assertEqual(rm.faixa(r, None), 1)            # falta a odd
        self.assertEqual(rm.faixa(r, -4.0), 2)            # odd abaixo da fair
        self.assertEqual(rm.faixa({"_p_metodo": 0.6, "_passos": 2}, 9.0), 1)
        self.assertEqual(rm.faixa({"_p_metodo": 0.6, "_passos": 1}, 9.0), 3)
        self.assertEqual(rm.faixa({"Odd (Hist)": 1.5}, None), 1)   # sem o metodo

    def test_pontos(self):
        pt = rm.pontos({"_p_metodo": 0.63, "_passos": 3}, 5.0)
        self.assertEqual(pt["total"], 78.0)
        self.assertEqual(rm.pontos({"_p_metodo": 0.7, "_passos": 1}, -40.0)["valor"], -15.0)

    def test_teto_na_fair_do_metodo(self):
        self.assertTrue(rm.dentro_do_teto({"_p_metodo": 0.6, "Odd (Mix)": 1.67}, 1.72))
        self.assertFalse(rm.dentro_do_teto({"_p_metodo": 0.55, "Odd (Mix)": 1.82, "Odd (Hist)": 1.5}, 1.72))
        self.assertTrue(rm.dentro_do_teto({"Odd (Hist)": 1.9, "Odd (Modelo)": 1.6}, 1.72))   # regra antiga

    def test_ml_vira_serie_com_odd_da_casa(self):
        try:
            import app_core.sinais_linha_core  # noqa: F401
        except ImportError:
            self.skipTest("so o LoL tem o motor do metodo")
        rows = [{"Mercado": "ML (Mapa)", "Lado": "A", "_metric": "00_ml_A"},
                {"Mercado": "ML (Mapa)", "Lado": "B", "_metric": "00_ml_B"},
                {"Mercado": "x", "_metric": "zz_mltot_A_kills"}]
        odds = lambda met, L: {"A": (1.40, "Stake"), "B": (3.1, "Polymarket")} if met == "ml" else {}
        out = rm.aplicar_metodo(rows, tg=pd.DataFrame(), team_a="A", team_b="B", p_serie_a=0.75, bo=3, odds=odds)
        self.assertEqual(len(out), 2)                       # a combinada antiga sai
        self.assertEqual(out[0]["Mercado"], "ML (Série MD3)")
        self.assertAlmostEqual(out[0]["_p_metodo"], 0.75)
        self.assertEqual(out[0]["Odd (Mix)"], 1.33)
        self.assertEqual((out[1]["_casa_odd"], out[1]["_casa_nome"]), (3.1, "Polymarket"))


if __name__ == "__main__":
    unittest.main()
