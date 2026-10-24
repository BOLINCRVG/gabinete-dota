# -*- coding: utf-8 -*-
"""A linha inicial da Analise de Linha e a do jogo (24/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.linha_sugerida_core import linha_da_pinnacle, linha_inicial, media_das_medias

ODDS = {"jogos": [{"id": 7, "casa": "Liquid", "fora": "Bilibili Gaming"}],
        "linhas": [
            {"jogo": 7, "tipo": "total", "unidade": "Kills", "periodo": 1, "alternativa": True, "pontos": 24.5},
            {"jogo": 7, "tipo": "total", "unidade": "Kills", "periodo": 1, "alternativa": False, "pontos": 26.5},
            {"jogo": 7, "tipo": "total", "unidade": "Turrets", "periodo": 1, "alternativa": False, "pontos": 12.5},
        ]}
APELIDOS = {"liquid": "Team Liquid"}


class TestLinha(unittest.TestCase):

    def test_principal_da_pinnacle_com_apelido_e_ordem_invertida(self):
        self.assertEqual(linha_da_pinnacle(ODDS, "Bilibili Gaming", "Team Liquid", "kills", 1, APELIDOS), 26.5)
        self.assertEqual(linha_da_pinnacle(ODDS, "Team Liquid", "Bilibili Gaming", "turrets", 1, APELIDOS), 12.5)
        self.assertIsNone(linha_da_pinnacle(ODDS, "T1", "Gen.G", "kills", 1, APELIDOS))

    def test_media_das_medias_no_meio_ponto(self):
        # 23,5 e 24,5 de media -> 24,0 -> 24,5
        self.assertEqual(media_das_medias(pd.Series([23, 24]), pd.Series([24, 25])), 24.5)
        self.assertEqual(media_das_medias(pd.Series([20.2]), pd.Series([21.0])), 20.5)

    def test_ordem_pinnacle_media_padrao(self):
        sa, sb = pd.Series([23, 24]), pd.Series([24, 25])
        l, o = linha_inicial("kills", "X", "Y", sa, sb, 27.5, odds=ODDS)
        self.assertEqual((l, "media" in o), (24.5, True))
        l, o = linha_inicial("kills", "X", "Y", None, None, 27.5, odds=ODDS)
        self.assertEqual((l, "padrao" in o), (27.5, True))


if __name__ == "__main__":
    unittest.main()
