# -*- coding: utf-8 -*-
"""A linha inicial da Analise de Linha e a do jogo (24/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.linha_sugerida_core import linha_da_pinnacle, linha_inicial, media_das_medias

def _l(u, p, lado, odd, alt=False):
    return {"jogo": 7, "tipo": "total", "unidade": u, "periodo": 1, "alternativa": alt,
            "pontos": p, "lado": lado, "odd": odd}


# os numeros reais de TES x AL, 24/10/2026
ODDS = {"jogos": [{"id": 7, "casa": "Liquid", "fora": "Bilibili Gaming"}],
        "linhas": [
            _l("Kills", 25.5, "over", 1.6452, True), _l("Kills", 25.5, "under", 2.18, True),
            _l("Kills", 26.5, "over", 1.8065), _l("Kills", 26.5, "under", 1.9709),
            _l("Kills", 27.5, "over", 1.9804, True), _l("Kills", 27.5, "under", 1.7812, True),
            _l("Turrets", 12.5, "over", 2.59), _l("Turrets", 12.5, "under", 1.4484),
        ]}
APELIDOS = {"liquid": "Team Liquid"}


class TestLinha(unittest.TestCase):

    def test_principal_da_pinnacle_com_apelido_e_ordem_invertida(self):
        self.assertEqual(linha_da_pinnacle(ODDS, "Bilibili Gaming", "Team Liquid", "kills", 1, APELIDOS), 26.5)
        self.assertEqual(linha_da_pinnacle(ODDS, "Team Liquid", "Bilibili Gaming", "turrets", 1, APELIDOS), 11.5)
        self.assertIsNone(linha_da_pinnacle(ODDS, "T1", "Gen.G", "kills", 1, APELIDOS))

    def test_linha_torta_anda_um_passo_para_o_equilibrio(self):
        # torres: so 12,5, under 1,45 x over 2,59 -> 11,5
        from app_core.linha_sugerida_core import escada_da_pinnacle, linha_equilibrada
        l, porque = linha_equilibrada(escada_da_pinnacle(ODDS, "Team Liquid", "Bilibili Gaming",
                                                         "turrets", 1, APELIDOS))
        self.assertEqual(l, 11.5)
        self.assertIn("torta", porque)

    def test_entre_varias_fica_a_mais_equilibrada(self):
        from app_core.linha_sugerida_core import escada_da_pinnacle, linha_equilibrada
        l, _ = linha_equilibrada(escada_da_pinnacle(ODDS, "Team Liquid", "Bilibili Gaming",
                                                    "kills", 1, APELIDOS))
        self.assertEqual(l, 26.5)

    def test_contagem_pequena_nao_da_passo_grande(self):
        from app_core.linha_sugerida_core import linha_equilibrada
        escada = {1.5: {"over": 2.62, "under": 1.44}}
        self.assertEqual(linha_equilibrada(escada)[0], 1.5)               # sem media, linha < 3
        self.assertEqual(linha_equilibrada(escada, media=1.3)[0], 1.5)    # media perto do 1,5
        self.assertEqual(linha_equilibrada({12.5: {"over": 2.59, "under": 1.45}}, media=11.2)[0], 11.5)
        self.assertEqual(linha_equilibrada({12.5: {"over": 2.59, "under": 1.45}}, media=12.6)[0], 12.5)

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
