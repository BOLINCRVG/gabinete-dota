# -*- coding: utf-8 -*-
"""Testes da derivação de split regional do EWC (app_core/ewc_region_split.py).

Pilar 2 da força-tarefa (23/07): lógica pura importante (região-casa por time a
partir das outras ligas; preenche o split vazio do EWC) e sem teste. Alimenta o
seletor de split que ele quer."""
import unittest

import pandas as pd

from app_core.ewc_region_split import _team_home_region_map, annotate_ewc_region_split


def _df(rows):
    return pd.DataFrame(rows)


class HomeRegionMapTests(unittest.TestCase):

    def test_regiao_da_liga_mais_frequente(self):
        df = _df([
            {"team": "A", "league": "LCK"}, {"team": "A", "league": "LCK"},
            {"team": "A", "league": "EWC"},                       # internacional: ignorado
            {"team": "B", "league": "LPL"}, {"team": "B", "league": "LPL"},
            {"team": "B", "league": "LCK"},                       # minoria -> perde
            {"team": "C", "league": "EWC"},                       # só internacional -> ausente
        ])
        m = _team_home_region_map(df)
        self.assertEqual(m.get("A"), "KR")
        self.assertEqual(m.get("B"), "CN")   # LPL(2) > LCK(1)
        self.assertNotIn("C", m)

    def test_liga_desconhecida_nao_entra(self):
        df = _df([{"team": "X", "league": "FOO_LIGA"}])
        self.assertEqual(_team_home_region_map(df), {})

    def test_df_vazio_ou_sem_colunas(self):
        self.assertEqual(_team_home_region_map(pd.DataFrame()), {})
        self.assertEqual(_team_home_region_map(_df([{"team": "A"}])), {})


class AnnotateEwcTests(unittest.TestCase):

    def _base(self):
        return _df([
            {"team": "A", "league": "LCK", "split": "Summer"},
            {"team": "A", "league": "LCK", "split": "Summer"},
            {"team": "B", "league": "LPL", "split": "Spring"},
            {"team": "B", "league": "LPL", "split": "Spring"},
            {"team": "A", "league": "EWC", "split": ""},      # -> KR
            {"team": "B", "league": "EWC", "split": ""},      # -> CN
            {"team": "NEW", "league": "EWC", "split": ""},    # sem casa -> Other
            {"team": "A", "league": "EWC", "split": "Manual"},  # idempotente
        ])

    def test_preenche_split_do_ewc_com_regiao_casa(self):
        out = annotate_ewc_region_split(self._base())
        ewc = out[out["league"] == "EWC"]
        by_team_empty = {(r.team): r.split for r in ewc.itertuples() if r.team in ("A", "B", "NEW")}
        # A aparece 2x no EWC (uma vazia->KR, uma Manual); pega as linhas por índice
        self.assertIn("KR", set(ewc[ewc["team"] == "A"]["split"]))
        self.assertEqual(set(ewc[ewc["team"] == "B"]["split"]), {"CN"})
        self.assertEqual(set(ewc[ewc["team"] == "NEW"]["split"]), {"Other"})

    def test_idempotente_nao_sobrescreve_split_existente(self):
        out = annotate_ewc_region_split(self._base())
        manual = out[(out["league"] == "EWC") & (out["team"] == "A") & (out["split"] == "Manual")]
        self.assertEqual(len(manual), 1)  # a linha "Manual" sobreviveu

    def test_sem_linhas_ewc_nao_muda(self):
        df = _df([{"team": "A", "league": "LCK", "split": "Summer"}])
        out = annotate_ewc_region_split(df.copy())
        self.assertEqual(list(out["split"]), ["Summer"])

    def test_sem_colunas_nao_quebra(self):
        df = _df([{"foo": 1}])
        # não deve levantar
        annotate_ewc_region_split(df)

    def test_cria_coluna_split_se_faltar(self):
        df = _df([
            {"team": "A", "league": "LCK"}, {"team": "A", "league": "LCK"},
            {"team": "A", "league": "EWC"},
        ])
        out = annotate_ewc_region_split(df)
        self.assertIn("split", out.columns)
        self.assertEqual(set(out[out["league"] == "EWC"]["split"]), {"KR"})


if __name__ == "__main__":
    unittest.main()
