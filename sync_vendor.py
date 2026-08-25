# -*- coding: utf-8 -*-
"""Embute o motor do LoL (app_core/ + mlcore/) DENTRO do GabineteDOTA.

Requisito 2026-07-12: o usuario envia SO a pasta do Dota pro amigo — ela tem
que funcionar sozinha em qualquer PC. Este script copia os pacotes do repo
irmao para ca (a "vendored copy"). O dota_paths prefere a pasta irma quando
ela existe (desenvolvimento sempre fresco) e cai na copia embutida quando nao
(maquina do amigo).

RODAR DE NOVO sempre que o motor do LoL mudar e for hora de reenviar o rar
(ou junto do build_clone.py — os dois andam juntos).

Uso: python sync_vendor.py
"""
from __future__ import annotations

import sys

# Console cp1252 no Windows derruba print com acento — e este script e rodado
# A MAO no PC dele (o robo roda em Linux/UTF-8 e nunca ve o problema).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_sibling() -> Path:
    for cand in sorted(HERE.parent.glob("gabinetelolweb*")):
        if (cand / "app_core").exists():
            return cand
    raise SystemExit("repo do LoL nao encontrado ao lado — rode na maquina de desenvolvimento")


# NAO excluir *.pyc globalmente: mlcore/_recovered/*.pyc e bytecode
# recuperado carregado em runtime (fit_form_style_core) — sem ele o app cai.
IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache")


def main() -> int:
    src = _find_sibling()
    total = 0
    for pkg in ("app_core", "mlcore"):
        dst = HERE / pkg
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / pkg, dst, ignore=IGNORE)
        n = sum(1 for _ in dst.rglob("*.py"))
        total += n
        print(f"vendored: {pkg}/ ({n} arquivos .py)")
    print(f"OK — motor embutido ({total} arquivos). A pasta do Dota agora roda sozinha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
