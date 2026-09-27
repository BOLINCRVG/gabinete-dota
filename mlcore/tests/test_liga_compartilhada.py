# -*- coding: utf-8 -*-
"""A liga do confronto no painel ao vivo — bug silencioso achado em 16/08/2026.

O código era:
    scope_league = scope_meta.get("league","") if "scope_meta" in locals() else ""

`scope_meta` NUNCA existiu nesse módulo (vive em market_stats_ui e
opponent_profile_ui). O guarda por `locals()` era sempre falso, então o painel de
estado ao vivo recebeu liga VAZIA desde sempre. Não quebrava, não avisava — só
não fazia. Mesma família dos outros silêncios do dia.

O critério de "qual é a liga do jogo" é o do resolve_shared_league_hint: só existe
liga da partida quando os DOIS times são dela.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.live_bet_tab_core import _liga_compartilhada


def _tg(linhas: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(linhas, columns=["teamname", "league"])


class TestLigaCompartilhada(unittest.TestCase):
    def test_mesma_liga_devolve_a_liga(self) -> None:
        tg = _tg([("T1", "LCK"), ("T1", "LCK"), ("GEN", "LCK"), ("GEN", "LCK")])
        self.assertEqual(_liga_compartilhada(tg, "T1", "GEN"), "LCK")

    def test_ligas_diferentes_devolve_vazio(self) -> None:
        """Confronto internacional: 'a liga do jogo' não existe."""
        tg = _tg([("T1", "LCK"), ("G2", "LEC")])
        self.assertEqual(_liga_compartilhada(tg, "T1", "G2"), "")

    def test_usa_a_liga_de_maior_volume_de_cada_time(self) -> None:
        """Um jogo em torneio internacional não muda a casa do time."""
        tg = _tg([("T1", "LCK"), ("T1", "LCK"), ("T1", "WLDs"),
                  ("GEN", "LCK"), ("GEN", "WLDs")])
        self.assertEqual(_liga_compartilhada(tg, "T1", "GEN"), "LCK")

    def test_time_ausente_devolve_vazio(self) -> None:
        tg = _tg([("T1", "LCK")])
        self.assertEqual(_liga_compartilhada(tg, "T1", "NAO_EXISTE"), "")

    def test_entradas_degeneradas_nao_quebram(self) -> None:
        self.assertEqual(_liga_compartilhada(None, "A", "B"), "")
        self.assertEqual(_liga_compartilhada(pd.DataFrame(), "A", "B"), "")
        self.assertEqual(_liga_compartilhada(_tg([]), "A", "B"), "")
        # DataFrame sem as colunas necessárias
        self.assertEqual(_liga_compartilhada(pd.DataFrame({"x": [1]}), "A", "B"), "")

    def test_liga_em_branco_nao_vira_liga(self) -> None:
        tg = _tg([("T1", "  "), ("GEN", "  ")])
        self.assertEqual(_liga_compartilhada(tg, "T1", "GEN"), "")


if __name__ == "__main__":
    unittest.main()
