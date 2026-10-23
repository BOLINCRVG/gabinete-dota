# -*- coding: utf-8 -*-
"""O livro-caixa nao cai com caminho absoluto no codigo (23/10/2026).

No Linux do robo, um "/root/cfg.json" num teste dava PermissionError e
derrubava a ferramenta inteira, calada pelo continue-on-error, por 2 meses.
Aqui o disco se comporta como o do robo: olhar esse caminho levanta
PermissionError. A ferramenta tem que seguir em frente.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import importlib.util

# Carrega a copia DESTE repositorio pelo caminho: na suite do Dota, outro teste
# poe a pasta do LoL no sys.path, e `import tools.livro_caixa` pegava a de la.
_RAIZ = Path(__file__).resolve().parents[2]
_ARQ = next(p for p in (_RAIZ / "tools" / "livro_caixa.py", _RAIZ / "livro_caixa.py") if p.exists())
_spec = importlib.util.spec_from_file_location("livro_caixa_deste_repo", _ARQ)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


class TestCaminhoAbsoluto(unittest.TestCase):

    def test_caminho_proibido_nao_derruba_a_varredura(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = Path(d)
            (raiz / "app.py").write_text('CFG = "/root/cfg.json"\n', encoding="utf-8")
            original = Path.exists

            def disco_do_robo(self, *a, **k):
                if str(self).replace("\\", "/").endswith("root/cfg.json"):
                    raise PermissionError(13, "Permission denied", str(self))
                return original(self, *a, **k)

            with mock.patch.object(Path, "exists", disco_do_robo):
                lc.arquivos_lidos_nao_versionados(raiz)   # nao pode levantar


if __name__ == "__main__":
    unittest.main()
