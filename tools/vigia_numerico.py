# -*- coding: utf-8 -*-
"""VIGIA NUMERICO: algum numero do modelo virou NaN ou infinito? (22/10/2026)

Item "numericamente estavel" da regua das 28 (LoL 0, Dota 0,5). O treino tinha
UM isfinite numa conta especifica; nada vigiava o resultado. NaN em rating ou
coeficiente nao quebra nada na hora: ele vira probabilidade NaN na tela, ou
uma comparacao que da sempre False -- o modo de falha calado que este projeto
ja pagou varias vezes (divida D11).

Percorre TODOS os numeros dos arquivos do modelo (o json do Python aceita NaN e
Infinity, que nao sao JSON valido, e os devolve como float -- e exatamente o que
se procura). Roda na suite (mlcore/tests/test_vigia_numerico.py), que o robo roda
todo dia. Arquivo identico nos dois repositorios.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
monitoramento "The model is numerically stable" (NaNs ou infinitos durante o
treino como sinal de problema).
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/vigia_numerico.py      (sai com 1 se achar NaN/infinito)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ARQUIVOS = ("ml_artifact.json", "players_artifact.json", "players_pscore.json", "params_fitted.json")
PADROES_AO_VIVO = ("live_models/**/metadata.json", "live_models/**/platt.json")


def ruins(obj, caminho="$", saida=None, limite=20):
    """Lista os caminhos com NaN/infinito dentro de um objeto JSON. Pura."""
    saida = [] if saida is None else saida
    if len(saida) >= limite:
        return saida
    if isinstance(obj, float):
        if not math.isfinite(obj):
            saida.append(f"{caminho} = {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            ruins(v, f"{caminho}.{k}", saida, limite)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            ruins(v, f"{caminho}[{i}]", saida, limite)
    return saida


def varrer(raiz: Path) -> dict:
    arquivos = [raiz / a for a in ARQUIVOS if (raiz / a).exists()]
    for padrao in PADROES_AO_VIVO:
        arquivos += sorted(raiz.glob(padrao))
    achados, lidos = {}, 0
    for arq in arquivos:
        try:
            obj = json.loads(arq.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            achados[str(arq.relative_to(raiz))] = [f"ilegivel: {e}"]
            continue
        lidos += 1
        r = ruins(obj)
        if r:
            achados[str(arq.relative_to(raiz))] = r
    return {"lidos": lidos, "achados": achados}


def main() -> int:
    raiz = Path(__file__).resolve().parent.parent
    res = varrer(raiz)
    for arq, lista in res["achados"].items():
        print(f"PROBLEMA em {arq}:", *lista[:5], sep="\n  ")
    print(f"[vigia_numerico] {res['lidos']} arquivos lidos, "
          f"{len(res['achados'])} com NaN/infinito")
    return 1 if res["achados"] else 0


if __name__ == "__main__":
    sys.exit(main())
