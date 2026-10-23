# -*- coding: utf-8 -*-
"""Divida D4 (23/10/2026): a cadencia do frescor e dado do repositorio.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from tools.frescor import CADENCIA, cadencia_do_repo
except ImportError:
    from frescor import CADENCIA, cadencia_do_repo


class TestCadenciaDoRepo(unittest.TestCase):

    def test_sem_arquivo_vale_a_tabela(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cadencia_do_repo(Path(d)), CADENCIA)

    def test_repo_declara_por_cima(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "evidence").mkdir()
            (Path(d) / "evidence" / "_frescor_cadencia.json").write_text(
                json.dumps({"cadencia_dias": {"livro_caixa.json": 1}}), encoding="utf-8")
            tab = cadencia_do_repo(Path(d))
            self.assertEqual(tab["livro_caixa.json"][1], 1.0)
            self.assertEqual(tab["livro_caixa.json"][0], CADENCIA["livro_caixa.json"][0])
            self.assertEqual(tab["fusion_scorecard.json"], CADENCIA["fusion_scorecard.json"])


if __name__ == "__main__":
    unittest.main()
