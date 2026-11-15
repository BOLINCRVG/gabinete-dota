# -*- coding: utf-8 -*-
"""VIGIA (15/11/2026): avisa quando um dado do robo fica velho.

Em novembro a OddsPapi ficou 10 dias barrada (cota do mes) e a Polymarket 3 dias (a API passou a
exigir gzip) -- e ninguem viu, porque os coletores saem "sucesso" quando a rede falha (para nao
encher a caixa de e-mail a cada soluco). Este script roda uma vez por dia e FALHA (sai 1) quando
algum arquivo passou do prazo: o GitHub manda e-mail de workflow agendado que falhou.

Uso: python vigia.py arquivo.json:campo_da_data:horas[:opcional] ...
  "opcional" = so avisa no texto, nao derruba (ex.: casas, que param quando a cota do mes acaba).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]      # tools/ -> raiz do repo


def idade_h(valor, agora: datetime):
    """Pura: horas desde a data `valor` (ISO, com ou sem fuso); None se nao der para ler."""
    try:
        t = datetime.fromisoformat(str(valor).replace("Z", "+00:00").replace(" ", "T"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (agora - t).total_seconds() / 3600.0


def conferir(regras: list, agora: datetime, raiz: Path = RAIZ) -> list:
    """Pura (fora a leitura): [(arquivo, idade em h ou None, prazo, situacao)], situacao em
    "ok" | "velho" | "velho (opcional)" | "ilegivel"."""
    fora = []
    for r in regras:
        partes = r.split(":")
        arq, campo, prazo = partes[0], partes[1], float(partes[2])
        opcional = len(partes) > 3 and partes[3] == "opcional"
        try:
            h = idade_h(json.loads((raiz / arq).read_text(encoding="utf-8")).get(campo), agora)
        except (OSError, ValueError, AttributeError):
            h = None
        if h is None:
            sit = "ilegivel"
        elif h > prazo:
            sit = "velho (opcional)" if opcional else "velho"
        else:
            sit = "ok"
        fora.append((arq, h, prazo, sit))
    return fora


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    res = conferir(sys.argv[1:], datetime.now(timezone.utc))
    ruim = False
    for arq, h, prazo, sit in res:
        idade = "sem data" if h is None else (f"{h:.0f} h" if h < 72 else f"{h / 24:.0f} dias")
        print(f"[vigia] {arq}: {idade} (prazo {prazo:.0f} h) -> {sit}")
        if sit in ("velho", "ilegivel"):
            ruim = True
            print(f"::error::{arq} esta parado ha {idade}: o robo que o atualiza esta falhando em silencio.")
        elif sit == "velho (opcional)":
            print(f"::warning::{arq} esta parado ha {idade} (esperado quando a cota do mes acaba).")
    return 1 if ruim else 0


if __name__ == "__main__":
    sys.exit(main())
