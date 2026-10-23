# -*- coding: utf-8 -*-
"""O livro-caixa nao cai com caminho absoluto no codigo (23/10/2026).

No Linux do robo, um "/root/cfg.json" num teste dava PermissionError e
derrubava a ferramenta inteira, calada pelo continue-on-error, por 2 meses.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import tools.livro_caixa as lc
except ImportError:
    import livro_caixa as lc


class TestCaminhoAbsoluto(unittest.TestCase):

    def test_caminho_absoluto_e_ignorado_sem_tocar_no_disco(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = Path(d)
            (raiz / "app.py").write_text('CFG = "/root/cfg.json"\n', encoding="utf-8")
            vistos = []
            original = Path.exists

            def espiao(self, *a, **k):
                vistos.append(str(self))
                return original(self, *a, **k)

            with mock.patch.object(Path, "exists", espiao):
                lc.arquivos_lidos_nao_versionados(raiz)
            self.assertFalse(any(v.replace("\\", "/").endswith("/root/cfg.json") for v in vistos))


if __name__ == "__main__":
    unittest.main()
