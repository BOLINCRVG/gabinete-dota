# -*- coding: utf-8 -*-
"""O vigia numerico (22/10/2026): roda na suite, entao o robo vigia todo dia.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.vigia_numerico import ruins, varrer

RAIZ = Path(__file__).resolve().parents[2]


class TestVigiaPegaOProblema(unittest.TestCase):

    def test_acha_nan_e_infinito_aninhados(self):
        obj = {"a": 1.0, "b": [0.5, float("nan")], "c": {"d": float("inf")}}
        achados = ruins(obj)
        self.assertEqual(len(achados), 2)
        self.assertTrue(any("$.b[1]" in x for x in achados))

    def test_numeros_normais_passam(self):
        self.assertEqual(ruins({"a": [1, 2.5, -3e9], "b": "texto", "c": None}), [])

    def test_le_nan_escrito_no_arquivo(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "ml_artifact.json").write_text('{"x": NaN}', encoding="utf-8")
            self.assertIn("ml_artifact.json", varrer(Path(d))["achados"])


class TestOsModelosDesteRepositorio(unittest.TestCase):

    def test_nenhum_nan_ou_infinito_no_modelo_publicado(self):
        res = varrer(RAIZ)
        self.assertGreater(res["lidos"], 0, "nenhum arquivo de modelo encontrado")
        self.assertEqual(res["achados"], {}, json.dumps(res["achados"], ensure_ascii=False)[:800])


if __name__ == "__main__":
    unittest.main()
