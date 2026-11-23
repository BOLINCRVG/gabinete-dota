# -*- coding: utf-8 -*-
"""Draft colado do SUPER-LIVE (23/11/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest
from datetime import datetime

from app_core import camada_draft_core as CD
from app_core import draft_colado_core as DC

HEROIS = {"1": "Anti-Mage", "2": "Axe", "13": "Puck", "41": "Faceless Void", "53": "Nature's Prophet",
          "74": "Invoker", "86": "Rubick", "96": "Centaur Warrunner", "100": "Tusk", "129": "Mars",
          "14": "Pudge", "26": "Lion"}
CAT = DC.catalogo_dota(HEROIS)
T = datetime(2026, 11, 20)


class TestLerDraft(unittest.TestCase):

    def test_duas_linhas_com_prefixo_e_apelidos(self):
        lados, nao, aviso = DC.ler_draft("Radiant: FV, Puck, Mars, Tusk, Rubick\n"
                                         "Dire: AM, invoker, NP, Axe, centaur", CAT)
        self.assertEqual(aviso, "")
        self.assertEqual(lados, (["41", "13", "129", "100", "86"], ["1", "74", "53", "2", "96"]))

    def test_dez_numa_linha_so(self):
        lados, _, _ = DC.ler_draft("Faceless Void / Puck / Mars / Tusk / Rubick / Anti-Mage / Invoker / "
                                   "Nature's Prophet / Axe / Centaur Warrunner", CAT)
        self.assertEqual(lados[1][0], "1")

    def test_nada_some_calado(self):
        lados, nao, aviso = DC.ler_draft("FV, Puck, Mars, Tusk, Rubikk\nAM, Invoker, NP, Axe, Cent", CAT)
        self.assertIsNone(lados)
        self.assertEqual(nao, ["Rubikk"])
        self.assertIn("Rubikk", aviso)
        self.assertIn("4 + 5", DC.ler_draft("FV, Puck, Mars, Tusk\nAM, Invoker, NP, Axe, Cent", CAT)[2])
        self.assertIn("repetido", DC.ler_draft("FV, Puck, Mars, Tusk, Axe\nAM, Invoker, NP, Axe, Cent", CAT)[2])


def _estado():
    e = CD.EstadoDraft()
    # conta 10 joga muito de FV e ganha; conta 11 joga Puck
    for _ in range(12):
        e.registrar([("10", "41"), ("11", "13"), ("12", "129"), ("13", "100"), ("14", "86")], 1, T)
        e.registrar([("20", "1"), ("21", "74"), ("22", "53"), ("23", "2"), ("24", "96")], 0, T)
    return e


class TestAtribuirEExplicar(unittest.TestCase):

    def test_atribui_pelo_historico(self):
        e = _estado()
        quem = DC.atribuir(["14", "13", "12", "11", "10"], ["41", "13", "129", "100", "86"], e)
        self.assertEqual(quem, ["10", "11", "12", "13", "14"])
        self.assertEqual(DC.atribuir(["10", "11"], ["41", "13"], e), [None, None])

    def test_motivos_somam_exato(self):
        e = _estado()
        coef = {"intercepto": 0.1, "p_pre": 1.0, "campeao": 10.0, "conforto": 3.0, "pratica": 0.2}
        az = list(zip(["10", "11", "12", "13", "14"], ["41", "13", "129", "100", "86"]))
        vm = list(zip(["20", "21", "22", "23", "24"], ["1", "74", "53", "2", "96"]))
        r = DC.explicar(e, coef, az, vm, T, 0.55)
        soma = r["pp_lado_azul"] + sum(x["pp"] if x["lado"] == "azul" else -x["pp"] for x in r["partes"])
        self.assertAlmostEqual(soma, 100 * (r["p_pos"] - r["p_pre"]), places=9)
        self.assertGreater(r["p_pos"], 0.55)              # o azul ganhou tudo com esses herois
        self.assertTrue(all(x["pp"] > 0 for x in r["partes"] if x["lado"] == "azul"))
        sa = sum(x["pp"] for x in r["partes"] if x["lado"] == "azul")
        sv = sum(x["pp"] for x in r["partes"] if x["lado"] == "vermelho")
        self.assertAlmostEqual(sa, -sv, places=9)       # duelo: um lado +X, o outro -X

    def test_jogador_fora_da_base_tira_a_parte_do_jogador(self):
        e = _estado()
        coef = {"intercepto": 0.0, "p_pre": 1.0, "campeao": 10.0, "conforto": 3.0, "pratica": 0.2}
        az = list(zip(["10", "11", "12", "13", "99"], ["41", "13", "129", "100", "86"]))
        vm = list(zip(["20", "21", "22", "23", "24"], ["1", "74", "53", "2", "96"]))
        r = DC.explicar(e, coef, az, vm, T, 0.5)
        self.assertEqual(r["faltam"], ["99"])
        self.assertTrue(all(x["pp_jogador"] == 0 for x in r["partes"]))


if __name__ == "__main__":
    unittest.main()
