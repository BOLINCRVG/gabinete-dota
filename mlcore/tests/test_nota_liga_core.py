# -*- coding: utf-8 -*-
"""Nota por liga: lookup e rótulo de calibração (a leitura que ele usa pra apostar)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app_core.nota_liga_core import _calib_label, nota_for_league


class TestNotaLiga(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        ev = self.root / "evidence"
        ev.mkdir()
        (ev / "nota_por_liga.json").write_text(json.dumps({
            "season": 2026, "nota_geral": 8.63, "min_n": 80,
            "ligas": [
                {"liga": "LPL", "tier": "T1", "n": 569, "nota": 6.6, "logloss": 0.6423,
                 "acuracia": 0.624, "ece": 0.019},
                {"liga": "CBLOL", "tier": "T2", "n": 211, "nota": 6.9, "logloss": 0.6327,
                 "acuracia": 0.664, "ece": 0.114},
            ],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lookup_case_insensitive_e_ausente(self) -> None:
        self.assertEqual(nota_for_league(self.root, "lpl")["nota"], 6.6)
        self.assertEqual(nota_for_league(self.root, "LPL")["n"], 569)
        self.assertIsNone(nota_for_league(self.root, "LIGA_QUE_NAO_EXISTE"))
        self.assertIsNone(nota_for_league(self.root, ""))

    def test_sem_arquivo_nao_quebra(self) -> None:
        with tempfile.TemporaryDirectory() as vazio:
            self.assertIsNone(nota_for_league(Path(vazio), "LPL"))

    def test_rotulo_de_calibracao_separa_confiavel_de_desconfie(self) -> None:
        self.assertEqual(_calib_label(0.019), "honesta")   # LPL, o melhor caso real
        self.assertEqual(_calib_label(0.06), "honesta")    # limite inclusivo
        self.assertEqual(_calib_label(0.09), "ok")
        self.assertEqual(_calib_label(0.114), "DESCONFIE")  # CBLOL, o pior caso real


if __name__ == "__main__":
    unittest.main()
