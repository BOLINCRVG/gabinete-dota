# -*- coding: utf-8 -*-
"""Uniao de team_id pelo ELENCO (27/10/2026).

Regra dele: o historico de um mesmo time NUNCA pode ficar partido. Caso real:
PARIVISION usou os ids 9572001 e 9824702 ao mesmo tempo, com os mesmos 5
jogadores, e a trava "ids simultaneos nao se unem" partia o time.
Excecao segura: elenco 5/5 + nunca se enfrentaram + nunca comecaram partidas a
menos de 45 min. E a uniao tem de chegar ao ID (o motor guarda nota por tid).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_whr_dota as ew  # noqa: E402

DIA = 86400.0
P = [11, 12, 13, 14, 15]          # os mesmos cinco nos dois ids


def _cenario(extras=(), elencos=None):
    """ids 1 e 2 ativos JUNTOS por 60 dias (dias pares x impares), adversarios 100+."""
    linhas = []
    for d in range(60):
        tid = 1 if d % 2 == 0 else 2
        linhas.append((tid, 100 + d, d * DIA))
    linhas += list(extras)
    m = pd.DataFrame(linhas, columns=["tid", "oid", "ord_key"])
    el = elencos or {1: P, 2: P}
    comp = {str(i): [[int(k * DIA), accs] for k in range(20)] for i, accs in el.items()}
    return m, comp


def _rep(m, comp):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / ew.ELENCOS_FILE).write_text(json.dumps(comp), encoding="utf-8")
        return ew.roster_identity_map(m, raw_path=Path(tmp) / "raw_matches.jsonl")


class TestUniaoPorElenco(unittest.TestCase):
    def test_mesmo_time_em_dois_ids_ao_mesmo_tempo_vira_um(self):
        rep = _rep(*_cenario())
        self.assertEqual(rep[1], rep[2])

    def test_se_ja_se_enfrentaram_sao_rivais_e_nao_unem(self):
        rep = _rep(*_cenario(extras=[(1, 2, 30.5 * DIA)]))
        self.assertNotEqual(rep[1], rep[2])

    def test_partidas_comecando_juntas_sao_times_diferentes(self):
        # o id 2 comeca um jogo 10 min depois de um jogo do id 1
        rep = _rep(*_cenario(extras=[(2, 999, 10 * DIA + 600)]))
        self.assertNotEqual(rep[1], rep[2])

    def test_quatro_de_cinco_nao_basta(self):
        rep = _rep(*_cenario(elencos={1: P, 2: P[:4] + [99]}))
        self.assertNotEqual(rep[1], rep[2])


class TestIdDoGrupo(unittest.TestCase):
    def test_tid_vira_o_do_grupo_e_o_cru_fica_guardado(self):
        m = pd.DataFrame({"match_id": [1, 2], "tid": [1, 2], "oid": [100, 101],
                          "teamname": ["A velho", "A novo"], "opp_teamname": ["X", "Y"],
                          "ord_key": [0.0, DIA]})
        velho = (ew.roster_identity_map, ew.canonical_team_names, ew.ids_ativos)
        ew.roster_identity_map = lambda m, api_rosters=None: {1: 1, 2: 1, 100: 100, 101: 101}
        ew.canonical_team_names = lambda m, rep, api_names=None: {1: "A", 2: "A", 100: "X", 101: "Y"}
        os.environ["DOTA_IDENTITY_API"] = "0"
        ew._IDENT_CACHE.clear()
        try:
            out = ew._aplicar_identidade(m)
        finally:
            ew.roster_identity_map, ew.canonical_team_names, ew.ids_ativos = velho
            ew._IDENT_CACHE.clear()
        self.assertEqual(list(out.tid), [1, 1])
        self.assertEqual(list(out.tid_orig), [1, 2])
        self.assertEqual(list(out.teamname), ["A", "A"])


if __name__ == "__main__":
    unittest.main()
