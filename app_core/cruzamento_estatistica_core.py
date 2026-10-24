# -*- coding: utf-8 -*-
"""CRUZAMENTO: o modelo ve valor -- e a estatistica do historico concorda? (24/10/2026)

Pedido dele: "se a casa fala que as torres em LOS x T1 e bom para o under, mas
a analise estatistica nao fala, voce me avisar". O valor da lista vem do
MODELO contra a odd da Pinnacle. A analise estatistica e a da aba Analise de
Linha: quantas vezes aquela linha teria batido nos jogos recentes dos dois
times. Sao duas leituras independentes; quando discordam, ele quer saber.

A CONTA (a mesma da Analise de Linha, sem modelo nenhum):
  - janela: os ULTIMOS 10 jogos de cada time (mapas), qualquer liga;
  - total do jogo (kills, torres, dragoes, baroes, inibidores): o total
    (a favor + contra) de cada jogo de cada time, contra a linha;
  - total do time (kills): os kills do time + os kills SOFRIDOS pelo adversario;
  - handicap de kills: o saldo do time (+ a linha) e o saldo do adversario
    visto do outro lado;
  - vencedor: vitorias do time + derrotas do adversario.
  Os jogos dos dois times entram juntos: pct = acertos / jogos.

O VEREDITO: >= 60% concorda, <= 40% discorda, no meio neutra; menos de 8
jogos somados = pouco dado (nao afirma nada). Os cortes sao de leitura
(maioria clara de 6 em 10), nao calibrados: o que mede acerto e a regua.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

JANELA = 10
CONCORDA = 60.0
DISCORDA = 40.0
MINIMO = 8

UNIDADE_COLUNA = {
    "kills": "kills", "towers": "towers", "turrets": "towers", "dragons": "dragons",
    "barons": "barons", "inhibitors": "inhibitors",
}


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def ultimos(team_games: pd.DataFrame, time: str, n: int = JANELA) -> pd.DataFrame:
    if team_games is None or team_games.empty or not time:
        return pd.DataFrame()
    d = team_games[team_games["team"].astype(str) == str(time)]
    if d.empty:
        return d
    return d.assign(_d=pd.to_datetime(d["date"], errors="coerce")).sort_values("_d").tail(n)


def _metrica(row: Mapping[str, Any]) -> Optional[str]:
    u = str(row.get("unidade") or "").strip().lower()
    if u in UNIDADE_COLUNA:
        return UNIDADE_COLUNA[u]
    nome = str(row.get("nome_pt") or row.get("mercado") or "").lower()
    for chave, col in (("kill", "kills"), ("torre", "towers"), ("drag", "dragons"),
                       ("bar", "barons"), ("inib", "inhibitors")):
        if chave in nome:
            return col
    return None


def acertos(row: Mapping[str, Any], hist_casa: pd.DataFrame, hist_fora: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """{'acertos', 'jogos'} da aposta nos jogos recentes dos dois times. Pura.

    `hist_casa`/`hist_fora`: ultimos jogos de cada time (linhas do team_games,
    do ponto de vista DAQUELE time)."""
    tipo = str(row.get("tipo") or "")
    lado = str(row.get("lado") or "").strip().lower()
    ponto = _num(row.get("pontos"))
    col = _metrica(row)
    unidade = str(row.get("unidade") or "").strip()

    def _serie(h, expr):
        try:
            return pd.to_numeric(expr(h), errors="coerce").dropna()
        except (KeyError, TypeError, ValueError):
            return pd.Series(dtype=float)

    partes: List[pd.Series] = []
    if tipo == "total" and lado in ("over", "under") and ponto is not None and col:
        for h in (hist_casa, hist_fora):
            tot = _serie(h, lambda d: d[f"{col}_for"] + d[f"{col}_against"])
            partes.append((tot > ponto) if lado == "over" else (tot < ponto))
    elif tipo == "team_total" and lado in ("over", "under") and ponto is not None and col == "kills":
        time = str(row.get("time") or "").lower()
        do_time, adv = (hist_casa, hist_fora) if time == "home" else (hist_fora, hist_casa)
        feitos = _serie(do_time, lambda d: d["kills_for"])
        sofridos = _serie(adv, lambda d: d["kills_against"])
        for s in (feitos, sofridos):
            partes.append((s > ponto) if lado == "over" else (s < ponto))
    elif tipo == "spread" and lado in ("home", "away") and ponto is not None and col == "kills":
        do_time, adv = (hist_casa, hist_fora) if lado == "home" else (hist_fora, hist_casa)
        saldo = _serie(do_time, lambda d: d["kills_for"] - d["kills_against"])
        saldo_adv = _serie(adv, lambda d: d["kills_against"] - d["kills_for"])
        for s in (saldo, saldo_adv):
            partes.append((s + ponto) > 0)
    elif tipo == "moneyline" and lado in ("home", "away") and not unidade:
        do_time, adv = (hist_casa, hist_fora) if lado == "home" else (hist_fora, hist_casa)
        partes.append(_serie(do_time, lambda d: d["win"]) == 1)
        partes.append(_serie(adv, lambda d: d["win"]) == 0)
    else:
        return None
    jogos = int(sum(len(p) for p in partes))
    return {"acertos": int(sum(int(p.sum()) for p in partes)), "jogos": jogos}


def veredito(r: Optional[Mapping[str, Any]]) -> str:
    if not r:
        return "sem estatistica"
    if r["jogos"] < MINIMO:
        return "pouco dado"
    pct = 100.0 * r["acertos"] / r["jogos"]
    if pct >= CONCORDA:
        return "concorda"
    if pct <= DISCORDA:
        return "discorda"
    return "neutra"


def cruzar(linhas: List[Mapping[str, Any]], team_games: Optional[pd.DataFrame],
           resolver: Optional[Callable[[str], str]]) -> List[Dict[str, Any]]:
    """As linhas com 'estat' = {'acertos','jogos','pct','veredito'} juntado."""
    cache: Dict[str, pd.DataFrame] = {}

    def _hist(nome: str) -> pd.DataFrame:
        if nome not in cache:
            alvo = nome
            if resolver is not None:
                try:
                    alvo = resolver(nome) or nome
                except Exception:
                    alvo = nome
            cache[nome] = ultimos(team_games, alvo) if team_games is not None else pd.DataFrame()
        return cache[nome]

    fora = []
    for x in linhas:
        y = dict(x)
        r = None
        if team_games is not None:
            hc, hf = _hist(str(x.get("casa") or "")), _hist(str(x.get("fora") or ""))
            if not hc.empty or not hf.empty:
                r = acertos(x, hc, hf)
        v = veredito(r)
        y["estat"] = None if not r else {
            **r, "pct": None if not r["jogos"] else round(100.0 * r["acertos"] / r["jogos"], 1),
            "veredito": v}
        y["estat_veredito"] = v
        fora.append(y)
    return fora
