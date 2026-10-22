# -*- coding: utf-8 -*-
"""ENSAIO DE REVERSAO (22/10/2026): publica v1, publica v2, reverte, confere v1.

Roda na suite, que o robo roda todo dia -- e isso que faz o "da para reverter"
valer ponto inteiro (ensaio automatico e repetido, nao ferramenta que so existe).
Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.reverter_modelo import reverter


def _git(raiz, *a):
    subprocess.run(["git", *a], cwd=str(raiz), check=True, capture_output=True)


@unittest.skipUnless(shutil.which("git"), "precisa do git")
class TestEnsaioDeReversao(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        _git(self.raiz, "init", "-q")
        _git(self.raiz, "config", "user.email", "t@t")
        _git(self.raiz, "config", "user.name", "t")
        (self.raiz / "live_models").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _publicar(self, versao):
        (self.raiz / "ml_artifact.json").write_text('{"v": %d}' % versao, encoding="utf-8")
        (self.raiz / "live_models" / "m.txt").write_text("modelo %d" % versao, encoding="utf-8")
        _git(self.raiz, "add", "-A")
        _git(self.raiz, "commit", "-q", "-m", "v%d" % versao)

    def test_sem_aplicar_nao_mexe_em_nada(self):
        self._publicar(1); self._publicar(2)
        res = reverter(self.raiz)
        self.assertFalse(res["aplicado"])
        self.assertEqual((self.raiz / "ml_artifact.json").read_text(encoding="utf-8"), '{"v": 2}')

    def test_reverte_modelo_e_modelos_ao_vivo_para_a_versao_anterior(self):
        self._publicar(1); self._publicar(2)
        res = reverter(self.raiz, aplicar=True)
        self.assertEqual((self.raiz / "ml_artifact.json").read_text(encoding="utf-8"), '{"v": 1}')
        self.assertEqual((self.raiz / "live_models" / "m.txt").read_text(encoding="utf-8"), "modelo 1")
        self.assertIn("ml_artifact.json", res["arquivos"])

    def test_sem_versao_anterior_recusa(self):
        self._publicar(1)
        with self.assertRaises(SystemExit):
            reverter(self.raiz, aplicar=True)


if __name__ == "__main__":
    unittest.main()
