# -*- coding: utf-8 -*-
"""tools/suite_na_esteira.py e o item "pipeline com teste de integracao" (22/10/2026).

A regua so da o ponto com a suite VERDE e RECENTE gravada pelo robo -- casar
texto no YAML nao prova que o passo roda. Arquivo identico nos dois repos.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from tools.suite_na_esteira import ler_resultado

try:
    from tools.regua_28 import v_suite_na_esteira
except ImportError:  # no Dota a regua mora na raiz
    from regua_28 import v_suite_na_esteira


class TestLeitura(unittest.TestCase):

    def test_verde(self):
        r = ler_resultado("....\nRan 5798 tests in 178.793s\n\nOK (skipped=6)\n")
        self.assertTrue(r["ok"]); self.assertEqual(r["testes"], 5798); self.assertEqual(r["pulados"], 6)

    def test_vermelho(self):
        r = ler_resultado("Ran 4697 tests in 220.1s\n\nFAILED (failures=2, errors=6, skipped=3)\n")
        self.assertFalse(r["ok"]); self.assertEqual((r["falhas"], r["erros"]), (2, 6))

    def test_sem_resumo_nao_e_verde(self):
        self.assertFalse(ler_resultado("ImportError: algo quebrou antes de rodar")["ok"])


class TestReguaExigeEvidenciaRecente(unittest.TestCase):

    def _raiz(self, d, conteudo):
        wf = Path(d) / ".github" / "workflows"; wf.mkdir(parents=True)
        (wf / "suite.yml").write_text("run: python tools/suite_na_esteira.py\n", encoding="utf-8")
        if conteudo is not None:
            ev = Path(d) / "evidence"; ev.mkdir()
            (ev / "suite_esteira.json").write_text(json.dumps(conteudo), encoding="utf-8")
        return Path(d)

    def test_verde_e_recente_ganha_ponto(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = self._raiz(d, {"ok": True, "testes": 10, "gerado_em": datetime.now().isoformat()})
            self.assertEqual(v_suite_na_esteira(raiz)[0], 1.0)

    def test_workflow_sem_resultado_nao_ganha(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(v_suite_na_esteira(self._raiz(d, None))[0], 0.0)

    def test_vermelho_ou_velho_nao_ganha(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = self._raiz(d, {"ok": False, "gerado_em": datetime.now().isoformat()})
            self.assertEqual(v_suite_na_esteira(raiz)[0], 0.0)
        with tempfile.TemporaryDirectory() as d:
            velho = (datetime.now() - timedelta(days=5)).isoformat()
            raiz = self._raiz(d, {"ok": True, "gerado_em": velho})
            self.assertEqual(v_suite_na_esteira(raiz)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
