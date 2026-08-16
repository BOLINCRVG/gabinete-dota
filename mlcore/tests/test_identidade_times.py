# -*- coding: utf-8 -*-
"""Identidade de time = team_id, nao o nome (achado 16/08/2026).

Os dois lados que estavam quebrados no universo do modelo:
  85 team_id com mais de um nome  -> historia partida (BetBoom aparecia com 78
                                     jogos em vez de 520)
  155 nomes reusados por varios ids -> times somados num rating so (Shinigami
                                     Gaming juntava 19 times diferentes)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_whr_dota import canonical_team_names  # noqa: E402


def _m(linhas: list[tuple[int, str, int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(linhas, columns=["tid", "teamname", "oid", "opp_teamname", "ord_key"])


class TestIdentidadeTimes(unittest.TestCase):
    def test_rebrand_vira_um_nome_so_e_o_mais_recente_vence(self) -> None:
        """Caso BetBoom: 4 nomes, 1 organizacao."""
        canon = canonical_team_names(_m([
            (1, "BB", 9, "X", 100.0),
            (1, "BB Team", 9, "X", 200.0),
            (1, "BetBoom Team", 9, "X", 300.0),
            (1, "BoomBoys", 9, "X", 400.0),
        ]))
        self.assertEqual(canon[1], "BoomBoys")

    def test_nome_generico_de_varios_ids_e_separado(self) -> None:
        """Caso Shinigami: mesmo nome, times diferentes — nao podem virar um."""
        canon = canonical_team_names(_m([
            (10, "Shinigami Gaming", 99, "A", 1.0),
            (10, "Shinigami Gaming", 99, "A", 2.0),
            (11, "Shinigami Gaming", 99, "A", 3.0),
        ]))
        self.assertNotEqual(canon[10], canon[11])
        # o de mais jogos fica com o nome limpo
        self.assertEqual(canon[10], "Shinigami Gaming")
        self.assertIn("11", canon[11])

    def test_caixa_e_homoglifo_nao_atrapalham_quando_o_id_e_o_mesmo(self) -> None:
        """Business Club/business club e EYE Gaming/Eуе Gaming (cirilico)."""
        canon = canonical_team_names(_m([
            (7, "business club", 9, "X", 10.0),
            (7, "Business Club", 9, "X", 20.0),
        ]))
        self.assertEqual(canon[7], "Business Club")
        self.assertEqual(len({canon[7]}), 1)

    def test_id_invalido_e_nome_vazio_sao_ignorados(self) -> None:
        canon = canonical_team_names(_m([
            (0, "sem id", 9, "X", 1.0),
            (-1, "negativo", 9, "X", 2.0),
            (5, "   ", 9, "X", 3.0),
        ]))
        self.assertNotIn(0, canon)
        self.assertNotIn(-1, canon)
        self.assertNotIn(5, canon)

    def test_o_lado_adversario_tambem_conta(self) -> None:
        """O time pode aparecer so como opponent — a identidade tem que pegar."""
        canon = canonical_team_names(_m([
            (9, "X", 3, "Kalmychata", 10.0),
            (9, "X", 3, "Rune Eaters", 20.0),
        ]))
        self.assertEqual(canon[3], "Rune Eaters")


if __name__ == "__main__":
    unittest.main()
