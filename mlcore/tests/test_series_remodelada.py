"""A aba Series remodelada (29/10/2026): tudo coerente com o ML da serie."""
import unittest

from app_core import series_tab_core as s


class TestSeriesRemodelada(unittest.TestCase):
    def test_tudo_soma_e_bate_com_a_serie(self):
        for bo in (3, 5):
            m = s.mercados_da_serie(0.75, bo, "A", "B")
            venc = {r["aposta"]: r["p"] for r in m if r["secao"] == "Vencedor da série"}
            self.assertAlmostEqual(venc["A"], 0.75, places=6)
            placar = [r["p"] for r in m if r["secao"] == "Placar exato"]
            self.assertAlmostEqual(sum(placar), 1.0, places=6)
            hc = {r["aposta"]: r["p"] for r in m if r["secao"] == "Handicap de mapas"}
            self.assertAlmostEqual(hc["A -1,5"] + hc["B +1,5"], 1.0, places=9)
            self.assertLess(hc["A -1,5"], 0.75)                  # -1,5 sempre mais dificil que vencer
            self.assertGreater(hc["A +1,5"], 0.75)

    def test_placar_bo3_bate_com_o_handicap(self):
        m = {r["chave"]: r["p"] for r in s.mercados_da_serie(0.6, 3, "A", "B")}
        self.assertAlmostEqual(m["hc_A_-1.5"], m["placar_A_2-0"], places=9)
        self.assertAlmostEqual(m["mapas_over_2.5"], m["placar_A_2-1"] + m["placar_B_2-1"], places=9)

    def test_totais_da_serie(self):
        nm = {2: 0.7, 3: 0.3}
        self.assertEqual(s.linhas_auto_serie(nm, 28.0, 7.5, 3.0), [58.5, 61.5, 64.5])
        p = s.p_mais_serie(nm, 28.0, 7.5, 61.5)
        self.assertTrue(0.4 < p < 0.6)
        self.assertGreater(s.p_mais_serie(nm, 28.0, 7.5, 55.5), p)
        self.assertIsNone(s.p_mais_serie(nm, 28.0, 0.0, 55.5))

    def test_odds_das_casas_sem_jogo(self):
        self.assertEqual(s.odds_das_casas_serie(None, False, "lol", ["Stake"]), {})


if __name__ == "__main__":
    unittest.main()
