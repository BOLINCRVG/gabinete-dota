import unittest

from app_core.series_tab_core import score_probs


class TestScoreProbs(unittest.TestCase):

    def test_bo3_sums_to_one(self):
        probs = score_probs(0.6, 3, kappa=None)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_bo3_a_2_0(self):
        probs = score_probs(0.6, 3, kappa=None)
        self.assertAlmostEqual(probs["A_2-0"], 0.36, places=6)

    def test_bo3_symmetric_at_half(self):
        probs = score_probs(0.5, 3, kappa=None)
        self.assertAlmostEqual(probs["A_2-0"], probs["B_2-0"], places=6)
        self.assertAlmostEqual(probs["A_2-1"], probs["B_2-1"], places=6)

    def test_bo3_p_one_all_a_wins(self):
        probs = score_probs(1.0, 3, kappa=None)
        self.assertAlmostEqual(probs["A_2-0"], 1.0, places=6)
        self.assertAlmostEqual(probs["A_2-1"], 0.0, places=6)

    def test_bo3_p_zero_all_b_wins(self):
        probs = score_probs(0.0, 3, kappa=None)
        self.assertAlmostEqual(probs["B_2-0"], 1.0, places=6)

    def test_bo3_has_four_keys(self):
        probs = score_probs(0.6, 3, kappa=None)
        self.assertEqual(len(probs), 4)

    def test_bo5_sums_to_one(self):
        probs = score_probs(0.5, 5, kappa=None)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_bo5_a_3_0(self):
        probs = score_probs(0.5, 5, kappa=None)
        self.assertAlmostEqual(probs["A_3-0"], 0.125, places=6)

    def test_bo5_symmetric_at_half(self):
        probs = score_probs(0.5, 5, kappa=None)
        self.assertAlmostEqual(probs["A_3-0"], probs["B_3-0"], places=6)
        self.assertAlmostEqual(probs["A_3-1"], probs["B_3-1"], places=6)
        self.assertAlmostEqual(probs["A_3-2"], probs["B_3-2"], places=6)

    def test_bo5_has_six_keys(self):
        probs = score_probs(0.7, 5, kappa=None)
        self.assertEqual(len(probs), 6)

    def test_unknown_bo_returns_empty(self):
        probs = score_probs(0.6, 1, kappa=None)
        self.assertEqual(probs, {})

    def test_returns_dict(self):
        self.assertIsInstance(score_probs(0.5, 3, kappa=None), dict)



class TestFormaDoDia(unittest.TestCase):
    """+-1,5 (29/10/2026): mapas com "forma do dia" (Beta, kappa=3) -- mais 2-0 que p*p."""

    def test_mais_varridas_que_mapas_independentes(self):
        from app_core.series_tab_core import KAPPA_SERIE
        if KAPPA_SERIE is None:
            self.skipTest("no Dota a forma do dia ainda nao foi medida")
        ind, beta = score_probs(0.65, 3, kappa=None), score_probs(0.65, 3)
        self.assertGreater(beta["A_2-0"], ind["A_2-0"])
        self.assertGreater(beta["B_2-0"], ind["B_2-0"])
        self.assertAlmostEqual(beta["A_2-0"], 0.65 * (0.65 * 3 + 1) / 4, places=9)

    def test_somam_um_e_simetria(self):
        for bo in (3, 5):
            for p in (0.3, 0.5, 0.8):
                self.assertAlmostEqual(sum(score_probs(p, bo, kappa=3.0).values()), 1.0, places=9)
        b = score_probs(0.5, 5, kappa=3.0)
        self.assertAlmostEqual(b["A_3-0"], b["B_3-0"], places=9)


    def test_menos_1_5_coerente_com_o_ml_da_serie(self):
        # ML da serie 1,10 (91%): o -1,5 tem de sair perto de 1,27, nao 1,83
        from app_core.series_tab_core import KAPPA_SERIE, chance_do_mapa_coerente
        if KAPPA_SERIE is None:
            self.skipTest("so LoL")
        m = chance_do_mapa_coerente(1 / 1.10, 3)
        p = score_probs(m, 3)
        self.assertAlmostEqual(p["A_2-0"] + p["A_2-1"], 1 / 1.10, places=4)
        self.assertLess(1 / p["A_2-0"], 1.35)


if __name__ == "__main__":
    unittest.main()
