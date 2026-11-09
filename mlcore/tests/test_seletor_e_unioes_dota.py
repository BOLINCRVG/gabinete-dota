# -*- coding: utf-8 -*-
"""Homonimo sem historia sai do seletor; uniao auditada junta o time partido (08/11/2026)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_whr_dota as ew  # noqa: E402
from app_core.seletor_times_dota_core import times_do_seletor  # noqa: E402


def _tg(linhas):
    fora = []
    for nome, n, fim in linhas:
        for d in pd.date_range(end=fim, periods=n, freq="D"):
            fora.append({"team": nome, "date": d})
    return pd.DataFrame(fora)


class TestSeletor(unittest.TestCase):
    def test_homonimo_sem_historia_some_e_o_time_real_fica(self):
        tg = _tg([("HULIGANI", 88, "2026-11-08"), ("HULIGANI #10225339", 1, "2026-08-19"),
                  ("MOUZ", 500, "2026-10-07"), ("MOUZ #9949250", 11, "2026-03-14"),
                  ("Inner Circle x Insanity", 176, "2026-09-04"),
                  ("Inner Circle x Insanity #10251974", 28, "2026-11-07"),
                  ("Sem Homonimo", 1, "2026-01-01")])
        self.assertEqual(times_do_seletor(sorted(tg.team.unique()), tg),
                         ["HULIGANI", "Inner Circle x Insanity", "Inner Circle x Insanity #10251974",
                          "MOUZ", "Sem Homonimo"])

    def test_sem_dados_nao_tira_ninguem(self):
        self.assertEqual(times_do_seletor(["A #1", "A"], pd.DataFrame()), ["A #1", "A"])


class TestUniaoAuditada(unittest.TestCase):
    def _mapa(self, unioes):
        m = pd.DataFrame([{"tid": 1, "oid": 100, "ord_key": 1000.0 * i} for i in range(1, 11)]
                         + [{"tid": 2, "oid": 101, "ord_key": 1000.0 * i + 500} for i in range(1, 4)]
                         + [{"tid": 3, "oid": 102, "ord_key": 9.0}])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ew.UNIOES_FILE).write_text(json.dumps({"unioes": unioes}), encoding="utf-8")
            return ew.roster_identity_map(m, raw_path=Path(tmp) / "raw_matches.jsonl")

    def test_lista_une_e_o_de_mais_jogos_representa(self):
        rep = self._mapa([{"nome": "X", "ids": [2, 1]}])
        self.assertEqual((rep[1], rep[2]), (1, 1))
        self.assertEqual(rep[3], 3)

    def test_sem_lista_nao_une(self):
        rep = self._mapa([])
        self.assertNotEqual(rep[1], rep[2])

    def test_a_lista_real_nao_tem_id_repetido(self):
        dados = json.loads((Path(ew.__file__).with_name(ew.UNIOES_FILE)).read_text(encoding="utf-8"))
        ids = [i for u in dados["unioes"] for i in u["ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(u["ids"]) >= 2 and u["prova"] for u in dados["unioes"]))


if __name__ == "__main__":
    unittest.main()
