# -*- coding: utf-8 -*-
"""Vencedor: modelo x Pinnacle x Polymarket (23/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app_core.mercado_vencedor_core import (
    achar_na_polymarket, carregar_polymarket, comparar_vencedor, pinnacle_sem_margem, selo)

POLY = [{"time_a": "LOS", "time_b": "T1", "p_a": 0.125, "p_b": 0.875, "spread": 0.01,
         "liquidez": 336309.0, "bo": 1}]


class TestAcharNaPolymarket(unittest.TestCase):

    def test_acha_com_letra_especial_e_na_ordem_da_tela(self):
        p = achar_na_polymarket(POLY, "LØS", "T1")
        self.assertEqual(p["p_a"], 0.125)

    def test_confronto_invertido_vira_para_o_time_1(self):
        p = achar_na_polymarket(POLY, "T1", "LØS")
        self.assertEqual(p["p_a"], 0.875)
        self.assertEqual(p["time_a"], "T1")

    def test_jogo_que_nao_esta_la(self):
        self.assertIsNone(achar_na_polymarket(POLY, "G2 Esports", "T1"))


class TestComparar(unittest.TestCase):

    def test_pinnacle_sem_margem_soma_um(self):
        p = pinnacle_sem_margem(4.0, 1.25)
        self.assertAlmostEqual(p + (1 - p), 1.0)
        self.assertAlmostEqual(p, (1 / 4) / (1 / 4 + 1 / 1.25))

    def test_los_t1_modelo_ve_valor_na_los_contra_as_duas(self):
        # modelo 23,8% na LOS; Pinnacle paga 6,0; Polymarket 12,5%
        r = comparar_vencedor(0.238, pin_odd_a=6.0, pin_odd_b=1.12, poly=POLY[0])
        los = r["linhas"][0]
        self.assertGreater(los["vant_pinnacle"], 0)
        self.assertGreater(los["vant_polymarket"], 0)
        self.assertTrue(r["casas_discordam"] is False or r["casas_discordam"] is True)

    def test_vantagem_polymarket_usa_o_preco_de_compra(self):
        r = comparar_vencedor(0.5, poly={"p_a": 0.4, "spread": 0.04, "liquidez": 1e6})
        self.assertAlmostEqual(r["linhas"][0]["preco_compra_poly"], 0.42)
        self.assertAlmostEqual(r["linhas"][0]["vant_polymarket"], round(100 * (0.5 / 0.42 - 1), 1))

    def test_sem_modelo_nao_inventa(self):
        self.assertEqual(comparar_vencedor(None)["linhas"], [])


class TestSelo(unittest.TestCase):

    def test_forte_quando_casas_e_modelo_apontam_o_mesmo_lado(self):
        self.assertEqual(selo(5.0, 8.0, True, True), "forte")

    def test_entre_casas_sem_o_modelo(self):
        self.assertEqual(selo(-3.0, 8.0, True, False), "entre casas")

    def test_polymarket_fraca_nao_conta(self):
        self.assertEqual(selo(5.0, 8.0, False, True), "modelo")

    def test_vantagem_grande_demais_e_suspeita(self):
        self.assertEqual(selo(35.0, 2.0, True, True), "suspeito")
        self.assertEqual(selo(1.0, 30.0, True, True), "suspeito")

    def test_sem_valor_sem_selo(self):
        self.assertEqual(selo(-3.0, -1.0, True, False), "")


class TestEntreCasas(unittest.TestCase):

    def test_los_polymarket_barata_contra_a_pinnacle(self):
        # Pinnacle 6.0 x 1.12 (LOS ~16% sem margem), Polymarket compra a 13%
        r = comparar_vencedor(0.238, pin_odd_a=6.0, pin_odd_b=1.12, poly=POLY[0])
        los = r["linhas"][0]
        self.assertAlmostEqual(los["vant_entre_casas"], round(100 * (los["p_pinnacle"] / 0.13 - 1), 1), places=0)
        self.assertGreater(los["vant_entre_casas"], 0)
        self.assertLess(r["linhas"][1]["vant_entre_casas"], 0)


class TestCarregar(unittest.TestCase):

    def test_sem_arquivo_diz_por_que(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("motivo", carregar_polymarket(Path(d)))

    def test_no_dota_diz_que_nao_coleta(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "dota_paths.py").write_text("", encoding="utf-8")
            self.assertIn("Dota", carregar_polymarket(Path(d))["motivo"])

    def test_le_partidas(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "polymarket_lol.json").write_text(json.dumps(
                {"coletado_em": "2026-10-23T21:00:00+00:00", "partidas": POLY}), encoding="utf-8")
            r = carregar_polymarket(Path(d))
            self.assertEqual(len(r["partidas"]), 1)


if __name__ == "__main__":
    unittest.main()
