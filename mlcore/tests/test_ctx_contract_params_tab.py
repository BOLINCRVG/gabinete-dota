# -*- coding: utf-8 -*-
"""Contrato do ctx das abas: tudo que a aba pede via ctx.get TEM que existir.

Bug real encontrado em 19/07: a aba Parâmetros pedia 73 nomes do ctx e **35
nunca chegaram** ao globals() do plays_app — 27 deles chamados como função,
ou seja, `None(...)` = TypeError. A aba estava quebrada no modo "Completo"
desde 02/06 e ninguém viu, porque o E2E só exercita o modo padrão (onde a
aba não é renderizada).

Este teste é ESTÁTICO (rápido, sem subir o app): casa os `ctx.get("X")` das
abas com os nomes que o plays_app expõe no seu namespace. Vale para qualquer
aba que use o padrão globals()->ctx.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# módulos de aba que consomem ctx montado com globals() do plays_app
MODULOS_DE_ABA = [
    "app_core/params_tab_core.py",
    "app_core/params_sections_core.py",
    "app_core/visao_tab_core.py",
    "app_core/rankings_tab_core.py",
    "app_core/radar_tab_core.py",
]


def _nomes_do_plays_app() -> str:
    return (ROOT / "plays_app.py").read_text(encoding="utf-8", errors="replace")


class TestCtxContract(unittest.TestCase):
    def test_todo_ctx_get_das_abas_existe_no_plays_app(self):
        app = _nomes_do_plays_app()
        faltando: dict[str, list[str]] = {}
        for rel in MODULOS_DE_ABA:
            p = ROOT / rel
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            pedidos = set(re.findall(r'ctx\.get\(\s*"(\w+)"', txt))
            ausentes = sorted(n for n in pedidos if not re.search(rf"\b{re.escape(n)}\b", app))
            if ausentes:
                faltando[rel] = ausentes
        self.assertEqual(
            faltando, {},
            "Nomes pedidos via ctx.get que NÃO existem no plays_app (viram None e "
            "quebram a aba quando chamados). Adicione o import correspondente no "
            f"plays_app: {faltando}",
        )

    def test_params_tab_pede_o_que_espera(self):
        """Sanidade do próprio teste: a aba Parâmetros realmente usa o padrão."""
        txt = (ROOT / "app_core" / "params_tab_core.py").read_text(encoding="utf-8", errors="replace")
        pedidos = set(re.findall(r'ctx\.get\(\s*"(\w+)"', txt))
        self.assertGreater(len(pedidos), 20, "esperava dezenas de ctx.get na aba Parâmetros")


if __name__ == "__main__":
    unittest.main()
