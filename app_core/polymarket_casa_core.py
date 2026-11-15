# -*- coding: utf-8 -*-
"""A POLYMARKET COMO CASA, lida direto dela (15/11/2026). Arquivo identico nos dois repos.

Ele aposta na Stake e na Polymarket. As odds das casas vinham so da OddsPapi (250 leituras por
mes, uma por dia -- e parada do dia 5 ao fim do mes quando a cota acaba). A Polymarket tem API
propria, gratis, lida de hora em hora (tools/coletar_polymarket.py -> polymarket_<esporte>.json).
Aqui esse arquivo vira a casa "polymarket" dos jogos que as telas ja leem
(suas_casas_core.carregar): vencedor da serie e de cada mapa, +-1,5 de mapas, total de mapas e
total de kills por mapa -- no MESMO formato da OddsPapi, entao toda tela que mostra "suas casas"
passa a mostrar a Polymarket fresca sem mudar nada nela.

A ODD e a de COMPRA: 1 / (meio do livro + spread / 2), o que ele paga de verdade. Mercado sem
livro (so "ultimo preco") ou com spread acima de SPREAD_MAXIMO nao entra: e preco que ninguem
vende. (Mesma regra do CS, serie_cs.SPREAD_MAXIMO, e coerente com suas_casas_core.SOMA_MAXIMA.)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app_core.mercado_vencedor_core import _chave, _num

SPREAD_MAXIMO = 0.12
JANELA_H = 36.0                     # o mesmo par em ate 36 h = o mesmo jogo
SPORT_ID = {"lol": "18", "dota": "16"}
_MAPA_MID = {1: "47", 2: "49", 3: "51"}


def esporte(raiz: Path) -> str:
    return "dota" if (Path(raiz) / "dota_paths.py").exists() else "lol"


def _quando(v: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00").replace(" ", "T"))
    except (TypeError, ValueError):
        try:
            t = datetime.fromisoformat(str(v).strip().replace(" ", "T") + ":00")     # "...+00" -> "...+00:00"
        except (TypeError, ValueError):
            return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _iso(v: Any) -> str:
    t = _quando(v)
    return str(v or "") if t is None else t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def odds_de_compra(p1: Any, spread: Any) -> Optional[Tuple[float, float]]:
    """(odd do 1o lado, odd do 2o lado) comprando no livro; None se nao ha livro que preste. Pura."""
    p, s = _num(p1), _num(spread)
    if p is None or s is None or not (0 < p < 1) or s < 0 or s > SPREAD_MAXIMO:
        return None
    c1, c2 = min(0.999, p + s / 2), min(0.999, 1 - p + s / 2)
    return round(1 / c1, 3), round(1 / c2, 3)


def como_casa(partida: Mapping[str, Any], esp: str) -> Tuple[Dict[str, List[float]], List[list]]:
    """(casas["polymarket"], linhas["polymarket"]) da partida, com time_a como time 1. Pura."""
    s = SPORT_ID[esp]
    mls: Dict[str, List[float]] = {}
    lins: List[list] = []
    o = odds_de_compra(partida.get("p_a"), partida.get("spread"))
    if o:
        mls[s + "1"] = [o[0], o[1]]
    ka = _chave(partida.get("time_a"))
    for m in partida.get("mercados") or []:
        o = odds_de_compra(m.get("p1"), m.get("spread"))
        if not o:
            continue
        tipo, mapa, linha = m.get("tipo"), int(m.get("mapa") or 0), _num(m.get("linha"))
        lados = [str(x) for x in (m.get("lados") or [])]
        if tipo == "child_moneyline" and mapa in _MAPA_MID and len(lados) == 2:
            mls[s + _MAPA_MID[mapa]] = [o[0], o[1]] if _chave(lados[0]) == ka else [o[1], o[0]]
        elif tipo == "totals" and linha is not None:
            lins.append(["maps", "total", 0, linha, o[0], o[1], None])
        elif tipo == "kill_over_under_game" and linha is not None and mapa:
            lins.append(["kills", "total", mapa, linha, o[0], o[1], None])
        elif tipo == "map_handicap" and linha is not None and len(lados) == 2:
            # a linha e a do PRIMEIRO lado; o formato das casas quer o handicap do time 1
            if _chave(lados[0]) == ka:
                lins.append(["maps", "spread", 0, linha, o[0], o[1], None])
            else:
                lins.append(["maps", "spread", 0, -linha, o[1], o[0], None])
    return mls, lins


def _virar(mls: Mapping[str, List[float]], lins: List[list]) -> Tuple[Dict[str, List[float]], List[list]]:
    """O mesmo, com os times trocados. Pura."""
    m2 = {k: [v[1], v[0]] for k, v in mls.items()}
    l2 = [[u, t, mp, -p, ob, oa, q] if t == "spread" else [u, t, mp, p, oa, ob, q] for u, t, mp, p, oa, ob, q in lins]
    return m2, l2


def _mesmo_jogo(a: Any, b: Any) -> bool:
    ta, tb = _quando(a), _quando(b)
    return ta is None or tb is None or abs((ta - tb).total_seconds()) <= JANELA_H * 3600


def juntar(doc: Mapping[str, Any], poly: Mapping[str, Any], esp: str) -> Dict[str, Any]:
    """O retrato das casas (OddsPapi) com a Polymarket direta por cima. Pura.

    Jogo que a OddsPapi ja tem (mesmo par, ate 36 h): a casa "polymarket" dele passa a ser a
    direta (mais fresca). Jogo que ela nao tem: entra novo, so com a Polymarket."""
    fora = dict(doc)
    jogos = [dict(j) for j in (doc.get("jogos") or [])]
    partidas = list(poly.get("partidas") or [])
    for p in partidas:
        mls, lins = como_casa(p, esp)
        if not mls and not lins:
            continue
        ka, kb = _chave(p.get("time_a")), _chave(p.get("time_b"))
        alvo = None
        for j in jogos:
            k1, k2 = _chave(j.get("time1")), _chave(j.get("time2"))
            if {k1, k2} == {ka, kb} and k1 != k2 and _mesmo_jogo(j.get("inicio_utc"), p.get("inicio_utc")):
                alvo = j
                break
        if alvo is None:
            alvo = {"fixtureId": "polymarket:" + str(p.get("evento") or ""), "time1": p.get("time_a"),
                    "time2": p.get("time_b"), "torneio": p.get("torneio"), "inicio_utc": _iso(p.get("inicio_utc")),
                    "casas": {}, "linhas": {}, "so_polymarket": True}
            jogos.append(alvo)
        nomes = [p.get("time_a"), p.get("time_b")]
        if _chave(alvo.get("time1")) != ka:
            mls, lins = _virar(mls, lins)
            nomes = nomes[::-1]
        alvo["casas"] = dict(alvo.get("casas") or {}, polymarket=mls)
        alvo["linhas"] = dict(alvo.get("linhas") or {}, polymarket=lins)
        alvo["polymarket_direta"] = True
        alvo["polymarket_evento"] = p.get("evento")          # por ele o placar das apostas fecha
        alvo["polymarket_times"] = nomes                     # como a Polymarket escreve o time 1 e o 2
    fora["jogos"] = jogos
    fora["polymarket_coletado_em"] = poly.get("coletado_em")
    return fora


def enxertar(g: Mapping[str, Any], inv: bool, outros: List[Tuple[Mapping[str, Any], bool]]):
    """(jogo, invertido): o jogo `g` escolhido pela tela com a Polymarket direta de um jogo-irmao
    enxertada -- o MESMO confronto (ate 36 h) com o nome escrito diferente nas duas fontes vira
    dois jogos em `juntar`; aqui, ja com o resolvedor de nomes da tela, eles voltam a ser um.
    `outros` = [(jogo, invertido)] de todos os que casaram com o confronto. Pura."""
    def _irmao(base, quer_so_poly):
        for h, inv_h in outros:
            if h is not base and bool(h.get("so_polymarket")) == quer_so_poly \
                    and _mesmo_jogo(base.get("inicio_utc"), h.get("inicio_utc")):
                return h, inv_h
        return None, False

    if g.get("so_polymarket"):                       # a tela escolheu o da Polymarket: a base e o da OddsPapi
        base, inv_base = _irmao(g, False)
        if base is None:
            return dict(g), inv
        doador, inv_doador = g, inv
    elif g.get("polymarket_direta"):
        return dict(g), inv
    else:
        base, inv_base = g, inv
        doador, inv_doador = _irmao(g, True)
        if doador is None:
            return dict(g), inv
    mls = dict((doador.get("casas") or {}).get("polymarket") or {})
    lins = list((doador.get("linhas") or {}).get("polymarket") or [])
    nomes = list(doador.get("polymarket_times") or [])
    if inv_doador != inv_base:
        mls, lins = _virar(mls, lins)
        nomes = nomes[::-1]
    novo = dict(base)
    novo["polymarket_evento"] = doador.get("polymarket_evento")
    novo["polymarket_times"] = nomes
    novo["casas"] = dict(base.get("casas") or {}, polymarket=mls)
    novo["linhas"] = dict(base.get("linhas") or {}, polymarket=lins)
    novo["polymarket_direta"] = True
    return novo, inv_base


def carregar_direta(raiz: Path) -> Dict[str, Any]:
    """polymarket_<esporte>.json, ou {} se o robo ainda nao gravou / esta ilegivel."""
    try:
        d = json.loads((Path(raiz) / f"polymarket_{esporte(raiz)}.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}
