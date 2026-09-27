# -*- coding: utf-8 -*-
"""Liga nova nao pode sumir do CSV calada.

O DEFEITO (achado 27/09/2026): o cache de tiers (league_tiers.csv) so era
renovado quando o ARQUIVO tinha mais de 7 dias. No robo da nuvem o checkout
do git recria o arquivo a cada rodada, entao ele sempre parecia novo e ficou
parado em 16/08. Toda liga criada depois virou "sem_tier" e saiu do universo
profissional: 246 partidas perdidas (PGL Wallachia S9 com 113, qualificatorias
da BLAST Slam...) e o CSV diario do Dota congelado em 10/09 -- com o robo
rodando "verde" todo dia.

A REGRA: liga que aparece nas partidas e nao esta no cache prova que o cache
esta velho, sem depender de data de arquivo.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import eval_whr_dota as ew  # noqa: E402


class TestLigaNovaRenovaOCache(unittest.TestCase):

    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.cache = Path(self.pasta.name) / "league_tiers.csv"
        # Arquivo recem-criado, como depois do checkout do robo.
        pd.DataFrame({"league_id": [1], "tier": ["professional"]}).to_csv(
            self.cache, index=False)
        self.api = pd.DataFrame({"league_id": [1, 2],
                                 "tier": ["professional", "professional"]})

    def tearDown(self):
        self.pasta.cleanup()

    def _carregar(self, ligas):
        with mock.patch.object(ew, "TIERS_CACHE", self.cache), \
             mock.patch.object(ew, "_fetch_tiers_df", return_value=self.api) as busca:
            tiers = ew.load_tiers(ligas_usadas=ligas)
        return tiers, busca.call_count

    def test_liga_fora_do_cache_renova_mesmo_com_arquivo_novo(self):
        tiers, chamadas = self._carregar([1, 2])
        self.assertEqual(chamadas, 1)
        self.assertEqual(tiers.get(2), "professional")
        # e grava o cache renovado, para o robo commitar
        self.assertIn(2, set(pd.read_csv(self.cache)["league_id"]))

    def test_todas_as_ligas_conhecidas_nao_vai_a_api(self):
        _, chamadas = self._carregar([1])
        self.assertEqual(chamadas, 0)

    def test_liga_zero_sem_id_nao_conta_como_nova(self):
        _, chamadas = self._carregar([0, 1])
        self.assertEqual(chamadas, 0)

    def test_api_fora_do_ar_fica_com_o_cache(self):
        with mock.patch.object(ew, "TIERS_CACHE", self.cache), \
             mock.patch.object(ew, "_fetch_tiers_df", side_effect=OSError("fora")):
            tiers = ew.load_tiers(ligas_usadas=[1, 2])
        self.assertEqual(tiers, {1: "professional"})

    def test_as_duas_pontas_passam_as_ligas_das_partidas(self):
        """Quem so chama load_tiers() sem ligas volta ao defeito."""
        for arq in ("eval_whr_dota.py", "opendota_upgrade_objectives.py"):
            fonte = (RAIZ / arq).read_text(encoding="utf-8")
            self.assertIn("load_tiers(ligas_usadas=", fonte, arq)
            self.assertNotIn("load_tiers()", fonte, arq)


if __name__ == "__main__":
    unittest.main()
