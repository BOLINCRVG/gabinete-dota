# -*- coding: utf-8 -*-
"""Agrupa TORNEIOS do Dota em CIRCUITOS, para a forca de liga acumular evidencia.

O problema (medido 25/08/2026): no LoL "liga" e a LCK, a LPL — entidades que
existem ha anos. No Dota "liga" e um torneio que dura duas semanas: sao 183
torneios distintos, 73% deles com menos de 100 partidas, mediana de 47. Cada um
estima a propria forca do ZERO, com a amostra dele sozinho.

A literatura diz o contrario disso: a Riot avalia o Elo de LIGA numa janela de 3
anos e o de TIME em 2, justamente porque jogo entre ligas e raro e a estimativa
precisa acumular (Global Power Rankings). Aqui o efeito e o oposto — a
DreamLeague estima a propria forca 14 vezes, uma por edicao, com ~200 partidas
cada, quando poderia usar 2.847.

Este modulo normaliza o nome: "DreamLeague Season 28" e "DreamLeague Season 29"
viram "DreamLeague". So isso multiplica por 14 a evidencia do circuito.

Uso: from dota_circuitos import circuito_de
"""
from __future__ import annotations

import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

# Sufixos que marcam EDICAO e nao identidade do circuito.
_EDICAO = re.compile(
    r"\b(?:season|temporada|division|div|episode|ep|stage|part|series|edition)\s*"
    r"[\dIVXLC]+\b",
    re.I,
)
_ANO = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{2,4})?\b")
_ROMANO_FIM = re.compile(r"\s+[IVXLC]{1,5}\s*$")
_NUM_FIM = re.compile(r"\s*[#]?\d+\s*$")
_SUJEIRA = re.compile(r"\s*[-:–—]\s*$")


def circuito_de(nome: str) -> str:
    """Nome do circuito a partir do nome do torneio.

    Conservador de proposito: na duvida devolve o nome original. Agrupar demais
    juntaria torneios diferentes e a forca de liga viraria media de coisas que
    nao tem relacao — pior que o problema que estamos resolvendo.
    """
    s = str(nome or "").strip()
    if not s:
        return ""
    base = s
    for _ in range(3):                       # "S28 2026" precisa de mais de uma passada
        base = _EDICAO.sub("", base)
        base = _ANO.sub("", base)
        base = _SUJEIRA.sub("", base).strip()
    base = _ROMANO_FIM.sub("", base).strip()
    base = _NUM_FIM.sub("", base).strip()
    base = _SUJEIRA.sub("", base).strip()
    base = re.sub(r"\s{2,}", " ", base).strip(" -:–—")
    # nome que virou curto demais nao e circuito, e ruido: fica o original
    return base if len(base) >= 4 else s


def main() -> int:
    import os
    os.environ.setdefault("DOTA_IDENTITY_API", "0")
    from eval_whr_dota import is_pro_tier, load_matches

    m = load_matches()
    m = m[is_pro_tier(m["tier"])]
    c = m["league"].value_counts()
    por_circ: dict[str, list[str]] = {}
    for nome in c.index:
        por_circ.setdefault(circuito_de(nome), []).append(str(nome))
    juntos = {k: v for k, v in por_circ.items() if len(v) > 1}
    print(f"torneios: {len(c)} -> circuitos: {len(por_circ)}")
    print(f"circuitos com mais de uma edicao: {len(juntos)}\n")
    ordem = sorted(juntos.items(), key=lambda kv: -sum(int(c[x]) for x in kv[1]))
    print(f"{'circuito':<34}{'edicoes':>8}{'partidas':>10}")
    for k, v in ordem[:12]:
        print(f"{k[:33]:<34}{len(v):>8}{sum(int(c[x]) for x in v):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
