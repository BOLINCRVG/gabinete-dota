# -*- coding: utf-8 -*-
"""Identidade do time pelo elenco e vigia de troca de ID (26/11/2026)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app_core import identidade_time_core as it  # noqa: E402

D = pd.Timestamp


def _j(*xs):
    return [(D(d), frozenset(e)) for d, e in xs]


class TestIdentidade(unittest.TestCase):
    def test_elenco_estavel(self):
        m = it.mudanca(_j(("2026-08-01", "abcde"), ("2026-10-01", "abcde"), ("2026-11-20", "abcde")))
        self.assertTrue(m["base_ok"])
        self.assertEqual(m["novos"], {})

    def test_jogador_novo_com_data(self):
        m = it.mudanca(_j(("2026-09-01", "abcde"), ("2026-11-10", "abcdx"), ("2026-11-20", "abcdx")))
        self.assertEqual(list(m["novos"]), ["x"])
        self.assertEqual(m["novos"]["x"], D("2026-11-10"))

    def test_confere(self):
        self.assertEqual(it.confere(list("abcdz"), frozenset("abcde")), (4, "ok"))
        self.assertEqual(it.confere(list("abcyz"), frozenset("abcde")), (3, "mexido"))
        self.assertEqual(it.confere(list("axyzw"), frozenset("abcde")), (1, "outro"))

    def test_troca_de_id(self):
        h = {"Velho": _j(("2026-09-01", "abcde"), ("2026-10-01", "abcde")),
             "Novo": _j(("2026-10-05", "abcde"), ("2026-11-20", "abcde")),
             "Outro": _j(("2026-10-05", "vwxyz"), ("2026-11-20", "vwxyz"))}
        ps = it.suspeitos_de_troca(h)
        self.assertEqual([(p["velho"], p["novo"], p["comum"]) for p in ps], [("Velho", "Novo", 5)])

    def test_nome_reaproveitado_por_outro_elenco_nao_une(self):
        h = {"4iki #1": _j(("2026-08-01", "abcde"), ("2026-09-06", "abcde")),
             "4iki": _j(("2026-10-31", "avwxy"), ("2026-11-20", "avwxy"))}
        self.assertEqual(it.suspeitos_de_troca(h), [])


class TestVigia(unittest.TestCase):
    def test_nome_base(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import vigia_trocas_id_dota as v
        self.assertEqual(v._nome_base("Scarlet Lotus #10313314"), "scarlet lotus")
        self.assertEqual(v._nome_base("CDUB"), "cdub")


if __name__ == "__main__":
    unittest.main()


class TestCompactar(unittest.TestCase):
    def test_linhas_com_id_comum_viram_uma(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import vigia_trocas_id_dota as v
        out = v.compactar([{"nome": "A", "ids": [1, 2], "prova": "p1"}, {"nome": "A2", "ids": [2, 3], "prova": "p2"},
                           {"nome": "B", "ids": [7, 8], "prova": "p3"}])
        self.assertEqual(sorted(tuple(u["ids"]) for u in out), [(1, 2, 3), (7, 8)])
