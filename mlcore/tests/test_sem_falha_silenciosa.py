# -*- coding: utf-8 -*-
"""Guarda contra a classe de bug que mais apareceu em 16/08/2026: falha CALADA.

Num único dia, quatro ferramentas quebraram por UnicodeEncodeError no console
cp1252 do Windows — train_offline (nome de time em chinês), o gate do TI,
product_scorecard (o símbolo de aviso) e data_scorecard (uma seta). O padrão é
sempre o mesmo e é o pior possível: o script FAZ o trabalho todo e morre ao
imprimir o resultado. Exit code sujo, evidência não salva, e quem chamou conclui
que falhou. A régua de Dados ficou semanas assim, e o único sintoma visível era
"nota velha" — que não aponta para causa nenhuma.

A varredura achou 55 arquivos no mesmo risco. Este teste existe para o número
não voltar a crescer: ferramenta nova que imprime acento tem que ajustar o
encoding. Roda em segundos (é análise estática, não executa nada).

Ver tools/find_silent_failures.py.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PASTAS = ["tools", "mlcore", "app_core"]
RECONFIG = re.compile(r"stdout\.reconfigure|PYTHONIOENCODING|codecs\.getwriter")

# Sem 'import sys' no topo o ajuste não tem onde entrar; ficam de fora por serem
# scripts auxiliares que não fazem parte de nenhuma esteira.
ISENTOS = {"fix_ui_diacritics.py", "probe_oddspapi.py"}


def _imprime_nao_ascii(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(ord(c) > 127 for c in node.value):
                return True
    return False


class TestSemFalhaSilenciosa(unittest.TestCase):
    def test_quem_imprime_acento_ajusta_o_encoding(self) -> None:
        faltando = []
        for pasta in PASTAS:
            for p in sorted((ROOT / pasta).rglob("*.py")):
                if "tests" in p.parts or "__pycache__" in p.parts or p.name in ISENTOS:
                    continue
                try:
                    src = p.read_text(encoding="utf-8")
                    tree = ast.parse(src)
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                eh_script = "__main__" in src or "\nprint(" in src
                if eh_script and not RECONFIG.search(src) and _imprime_nao_ascii(tree):
                    faltando.append(str(p.relative_to(ROOT)))
        self.assertEqual(
            faltando, [],
            "Estes arquivos imprimem caractere não-ASCII sem ajustar o encoding e "
            "vão morrer no console cp1252 DEPOIS de fazer o trabalho:\n  "
            + "\n  ".join(faltando)
            + "\nRode: python tools/find_silent_failures.py --fix-encoding",
        )

    def test_subprocesso_com_saida_capturada_nao_pode_ser_ignorado(self) -> None:
        """Foi o que escondeu a régua de Dados quebrada por semanas."""
        ruins = []
        for pasta in PASTAS:
            for p in sorted((ROOT / pasta).rglob("*.py")):
                if "tests" in p.parts or "__pycache__" in p.parts:
                    continue
                try:
                    tree = ast.parse(p.read_text(encoding="utf-8"))
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                        continue
                    f = node.value.func
                    nome = ""
                    if isinstance(f, ast.Attribute):
                        nome = f"{f.value.id}.{f.attr}" if isinstance(f.value, ast.Name) else f.attr
                    if nome in {"subprocess.run", "subprocess.call", "run"} and any(
                        k.arg in {"capture_output", "stdout", "stderr"} for k in node.value.keywords
                    ):
                        ruins.append(f"{p.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(
            ruins, [],
            "Subprocesso com saída capturada e resultado descartado — a falha vira "
            "silêncio:\n  " + "\n  ".join(ruins),
        )


if __name__ == "__main__":
    unittest.main()
