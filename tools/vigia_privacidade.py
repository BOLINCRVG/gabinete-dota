# -*- coding: utf-8 -*-
"""VIGIA DE PRIVACIDADE: o pipeline grava algum dado pessoal alem do minimo? (23/10/2026)

Item "controles de privacidade no pipeline" da regua das 28 (LoL 0,5, Dota 0).

O que o projeto PRECISA de pessoa: o identificador do jogador profissional
(playerid/playername no LoL, account_id no Dota), porque a camada de lineup
mede quem joga. Ambos sao publicos: saem do CSV do Oracle's Elixir e da API
do OpenDota, ligados a carreira profissional. O que ele NAO precisa e nunca
deve guardar: apelido e perfil da conta Steam, nome real, pais, avatar,
e-mail, data de nascimento.

O CONTROLE ja existia em parte (o opendota_sync.py grava no sidecar so uma
lista fechada de campos por jogador) mas nada garantia que continuasse
assim. Este vigia percorre os arquivos de dados versionados (e o sidecar local,
se existir) e falha se aparecer qualquer campo da lista proibida. Roda na
suite (mlcore/tests/test_vigia_privacidade.py), que o robo roda todo dia.
Arquivo identico nos dois repositorios.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
dados "The data pipeline has appropriate privacy controls".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/vigia_privacidade.py     (sai com 1 se achar campo proibido)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# nomes de campo (chave JSON ou coluna CSV), comparados em minusculas
PROIBIDOS = frozenset({
    "personaname", "realname", "real_name", "fullname", "full_name",
    "avatar", "avatarfull", "avatarmedium", "profileurl", "steamid",
    "loccountrycode", "country", "country_code", "nationality",
    "email", "e-mail", "birthdate", "birthday", "date_of_birth", "phone",
})
EXTENSOES = (".csv", ".json", ".jsonl")
LOCAIS = ("raw_matches.jsonl",)          # sidecar fora do git, se existir
LIMITE_BYTES_JSON = 60_000_000


def _chaves(obj, saida: set, limite=200_000):
    pilha = [obj]
    while pilha and len(saida) < limite:
        o = pilha.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                saida.add(str(k).lower())
                pilha.append(v)
        elif isinstance(o, list):
            pilha.extend(o)
    return saida


def campos_de(arq: Path) -> set:
    """Nomes de campo de um arquivo de dados. Pura (so le)."""
    if arq.suffix == ".csv":
        with arq.open(encoding="utf-8", errors="replace") as f:
            return {c.strip().strip('"').lower() for c in f.readline().split(",")}
    if arq.stat().st_size > LIMITE_BYTES_JSON:
        return set()
    if arq.suffix == ".jsonl":
        saida = set()
        with arq.open(encoding="utf-8", errors="replace") as f:
            for linha in f:
                try:
                    _chaves(json.loads(linha), saida)
                except ValueError:
                    continue
        return saida
    try:
        return _chaves(json.loads(arq.read_text(encoding="utf-8", errors="replace")), set())
    except ValueError:
        return set()


def arquivos_de_dados(raiz: Path) -> list[Path]:
    r = subprocess.run(["git", "ls-files"], cwd=str(raiz), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=120)
    nomes = r.stdout.splitlines() if r.returncode == 0 else []
    arqs = [raiz / n for n in nomes if n.endswith(EXTENSOES)
            and not n.startswith(("evidence/", "mlcore/tests/", ".github/"))]
    arqs += [raiz / n for n in LOCAIS]
    return [a for a in arqs if a.is_file()]


def varrer(raiz: Path) -> dict:
    achados, lidos = {}, 0
    for arq in arquivos_de_dados(raiz):
        lidos += 1
        ruins = sorted(campos_de(arq) & PROIBIDOS)
        if ruins:
            achados[str(arq.relative_to(raiz))] = ruins
    return {"lidos": lidos, "achados": achados}


def main() -> int:
    res = varrer(Path(__file__).resolve().parent.parent)
    for arq, campos in res["achados"].items():
        print(f"PROIBIDO em {arq}: {', '.join(campos)}")
    print(f"[vigia_privacidade] {res['lidos']} arquivos de dados lidos, "
          f"{len(res['achados'])} com dado pessoal proibido")
    return 1 if res["achados"] else 0


if __name__ == "__main__":
    sys.exit(main())
