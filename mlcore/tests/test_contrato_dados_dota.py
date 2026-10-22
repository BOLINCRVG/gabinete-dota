# -*- coding: utf-8 -*-
"""Contrato de dados do Dota (22/10/2026): pega o CSV quebrado, aprova o certo."""
from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

try:
    from tools.contrato_dados_dota import conferir
except ImportError as _sem_ferramenta:
    import unittest as _u
    raise _u.SkipTest("tools.contrato_dados_dota nao existe neste repositorio: " + str(_sem_ferramenta))

AGORA = datetime(2026, 10, 22)


def _jogo(gid, vencedor="Blue", data="2026-10-01", dur=2400, time="A"):
    return [
        {"gameid": gid, "date": data, "league": "L", "side": "Blue", "position": "team",
         "teamname": time, "result": int(vencedor == "Blue"), "gamelength": dur},
        {"gameid": gid, "date": data, "league": "L", "side": "Red", "position": "team",
         "teamname": "B", "result": int(vencedor == "Red"), "gamelength": dur},
    ]


def _df(*jogos):
    return pd.DataFrame([linha for j in jogos for linha in j])


class TestContrato(unittest.TestCase):

    def test_csv_certo_aprova(self):
        r = conferir(_df(_jogo(1), _jogo(2, "Red")), agora=AGORA)
        self.assertTrue(r["ok"], r["falhas"])
        self.assertEqual(r["jogos"], 2)

    def test_dois_vencedores_reprova(self):
        j = _jogo(1); j[1]["result"] = 1
        self.assertFalse(conferir(_df(j), agora=AGORA)["ok"])

    def test_jogo_com_tres_linhas_reprova(self):
        j = _jogo(1); j.append(dict(j[0]))
        self.assertFalse(conferir(_df(j), agora=AGORA)["ok"])

    def test_data_no_futuro_reprova(self):
        self.assertFalse(conferir(_df(_jogo(1, data="2027-01-01")), agora=AGORA)["ok"])

    def test_duracao_absurda_reprova(self):
        self.assertFalse(conferir(_df(_jogo(1, dur=30)), agora=AGORA)["ok"])

    def test_time_em_branco_reprova(self):
        self.assertFalse(conferir(_df(_jogo(1, time=None)), agora=AGORA)["ok"])

    def test_coluna_faltando_reprova(self):
        self.assertFalse(conferir(_df(_jogo(1)).drop(columns=["side"]), agora=AGORA)["ok"])

    def test_csv_que_encolheu_reprova(self):
        r = conferir(_df(_jogo(1), _jogo(2)), jogos_antes=10, agora=AGORA)
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
