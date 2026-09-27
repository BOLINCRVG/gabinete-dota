# -*- coding: utf-8 -*-
"""Testes do aviso de recorte por time.

O defeito que estes testes travam foi relatado pelo dono em 13/09/2026: a tela
dizia "Times de ligas diferentes" para Sentinels x FlyQuest, que sao os DOIS
da LCS. A condicao nao comparava liga nenhuma — disparava quando um dos times
ficava sem jogos no recorte, que e outra coisa.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market_v3_analysis_core import descrever_recorte_por_time


def _rec(liga: str, split: str = "Summer") -> dict:
    return {"league": liga, "split": split,
            "team_df": pd.DataFrame([{"x": 1}]), "league_scope": pd.DataFrame()}


class TestMesmaLiga(unittest.TestCase):
    """O caso que o dono viu e reportou."""

    def test_o_caso_relatado_Sentinels_x_FlyQuest(self):
        d = descrever_recorte_por_time(_rec("LCS"), _rec("LCS"),
                                       "Sentinels", "FlyQuest")
        self.assertTrue(d["mesma_liga"])
        self.assertNotIn("ligas diferentes", d["texto"])
        self.assertIn("LCS", d["texto"])

    def test_diz_o_MOTIVO_de_verdade(self):
        """Nao basta parar de mentir: tem de explicar o que houve."""
        d = descrever_recorte_por_time(_rec("LCS"), _rec("LCS"), "A", "B")
        t = d["texto"].lower()
        self.assertIn("recorte", t)
        self.assertTrue("filtro" in t or "vazio" in t,
                        "o aviso precisa dizer que foi o filtro que esvaziou")

    def test_maiuscula_e_minuscula_nao_inventam_diferenca(self):
        d = descrever_recorte_por_time(_rec("lcs"), _rec("LCS"), "A", "B")
        self.assertTrue(d["mesma_liga"])

    def test_espaco_sobrando_nao_inventa_diferenca(self):
        d = descrever_recorte_por_time(_rec("  LCS "), _rec("LCS"), "A", "B")
        self.assertTrue(d["mesma_liga"])


class TestLigasDiferentes(unittest.TestCase):
    """Quando sao mesmo diferentes, o aviso antigo continua valendo."""

    def test_ligas_diferentes_continuam_sendo_anunciadas(self):
        d = descrever_recorte_por_time(_rec("LCS"), _rec("LCK"), "A", "B")
        self.assertFalse(d["mesma_liga"])
        self.assertIn("ligas diferentes", d["texto"])
        self.assertIn("LCS", d["texto"])
        self.assertIn("LCK", d["texto"])

    def test_os_dois_splits_aparecem(self):
        d = descrever_recorte_por_time(_rec("LCS", "Summer"), _rec("LCK", "Spring"),
                                       "A", "B")
        self.assertIn("Summer", d["texto"])
        self.assertIn("Spring", d["texto"])


class TestSemSaber(unittest.TestCase):
    """PRINCIPIO IV: nao saber nao pode virar afirmacao."""

    def test_liga_vazia_NAO_afirma_que_e_igual(self):
        d = descrever_recorte_por_time(_rec(""), _rec(""), "A", "B")
        self.assertFalse(d["mesma_liga"])

    def test_liga_vazia_NAO_afirma_que_e_diferente(self):
        d = descrever_recorte_por_time(_rec(""), _rec("LCS"), "A", "B")
        self.assertNotIn("ligas diferentes", d["texto"])

    def test_registro_ausente_nao_estoura(self):
        d = descrever_recorte_por_time({}, None, "A", "B")
        self.assertFalse(d["mesma_liga"])
        self.assertTrue(d["texto"])

    def test_split_ausente_vira_travessao_e_nao_None(self):
        d = descrever_recorte_por_time(_rec("LCS", ""), _rec("LCK", ""), "A", "B")
        self.assertNotIn("None", d["texto"])


if __name__ == "__main__":
    unittest.main()
