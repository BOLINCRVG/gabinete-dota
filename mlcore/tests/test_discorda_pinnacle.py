# -*- coding: utf-8 -*-
"""Aviso de discordancia com a Pinnacle (25/11/2026)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app_core import discorda_pinnacle_core as dp  # noqa: E402


class TestDiscorda(unittest.TestCase):
    def test_concorda_sem_aviso(self):
        self.assertEqual(dp.aviso(0.60, 0.55, time_a="A", time_b="B"), "")

    def test_discorda_no_mapa(self):
        t = dp.aviso(0.70, 0.55, time_a="A", time_b="B")
        self.assertIn("15 pontos", t)
        self.assertIn("A (o favorito)", t)

    def test_azarao(self):
        t = dp.aviso(0.52, 0.35, time_a="A", time_b="B")
        self.assertIn("A (o azarão)", t)

    def test_serie_volta_para_o_mapa(self):
        # serie 64,8% x 50% e ~10 pontos no mapa; 70% x 50% de serie passa disso
        self.assertAlmostEqual(dp.mapa_equivalente(dp.p_serie(0.6, 3), 3), 0.6, places=6)
        self.assertEqual(dp.aviso(dp.p_serie(0.58, 3), 0.5, time_a="A", time_b="B", bo=3), "")
        self.assertNotEqual(dp.aviso(dp.p_serie(0.62, 3), 0.5, time_a="A", time_b="B", bo=3), "")

    def test_sem_margem(self):
        self.assertAlmostEqual(dp.sem_margem(1.9, 1.9), 0.5)
        self.assertIsNone(dp.sem_margem(None, 2.0))
        self.assertEqual(dp.aviso(0.7, None, time_a="A", time_b="B"), "")


if __name__ == "__main__":
    unittest.main()
