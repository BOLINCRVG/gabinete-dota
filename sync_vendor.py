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


def _blobs_do_lol(src: Path) -> set:
    """Todo conteudo que ja existiu em app_core/ e mlcore/ no historico do LoL."""
    import subprocess
    r = subprocess.run(["git", "-C", str(src), "rev-list", "--all", "--objects", "--",
                        "app_core", "mlcore"], capture_output=True, text=True)
    return {linha.split(" ", 1)[0] for linha in r.stdout.splitlines() if linha.strip()}


def _hash_git(conteudo: bytes) -> str:
    import hashlib
    dado = conteudo.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(dado) + dado).hexdigest()


def mudancas_so_do_dota(src: Path) -> list:
    """Arquivos daqui que seriam SOBRESCRITOS e cujo conteudo NUNCA existiu no LoL.

    O ERRO DE 27/09/2026: esta ferramenta apagava app_core/ e mlcore/ e
    copiava o LoL por cima. A copia do Dota tinha consertos proprios em 15
    pecas (o pre-voo do retreino, o aviso de retreino que falha, a incerteza no
    rating...), e eles teriam sumido calados -- o pre-voo e exatamente o que
    impediu o Dota de ficar sem modelo ao vivo em 16/08. Tambem apagaria os
    testes que so existem aqui.
    """
    conhecidos = _blobs_do_lol(src)
    fora = []
    for pkg in ("app_core", "mlcore"):
        for f in (HERE / pkg).rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(HERE)
            lol = src / rel
            if not lol.exists():
                continue            # so do Dota: a copia por cima nao toca
            meu, dele = f.read_bytes(), lol.read_bytes()
            if meu.replace(b"\r\n", b"\n") == dele.replace(b"\r\n", b"\n"):
                continue
            if _hash_git(meu) not in conhecidos:
                fora.append(str(rel))
    return sorted(fora)


def main() -> int:
    src = _find_sibling()
    forcar = "--forcar" in sys.argv
    proprios = mudancas_so_do_dota(src)
    if proprios and not forcar:
        print("PAREI: estas pecas daqui tem mudancas que NUNCA existiram no LoL e seriam")
        print("apagadas pela copia. Leve-as para o LoL primeiro (juncao de tres vias), ou")
        print("rode com --forcar se tiver certeza de que podem sumir:")
        for f in proprios:
            print("   ", f)
        return 1
    total = 0
    for pkg in ("app_core", "mlcore"):
        dst = HERE / pkg
        # COPIA POR CIMA, sem apagar: o que so existe aqui (testes do Dota)
        # continua existindo.
        shutil.copytree(src / pkg, dst, ignore=IGNORE, dirs_exist_ok=True)
        n = sum(1 for _ in dst.rglob("*.py"))
        total += n
        print(f"vendored: {pkg}/ ({n} arquivos .py)")
    print(f"OK — motor embutido ({total} arquivos). A pasta do Dota agora roda sozinha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
