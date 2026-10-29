"""K/D/A DE JOGADOR PELO CONFRONTO (29/10/2026).

Ele: "se for uma analise estatistica puramente vai ficar ruim porque esses times nessa parte do
campeonato ganharam de todos os times nas suas regioes... e se for analisar so no worlds vai ser
ruim porque tem poucos jogos". A media do jogador carrega o dominio regional do TIME. A parte do
jogador que nao depende do adversario e a FATIA dele; a parte do confronto sai da chance de
vitoria NESTE jogo (o ML do app, que ja cruza regioes):

    kills   = fatia do jogador nas kills do time      x  total do confronto x q(p)
    deaths  = fatia do jogador nas mortes do time     x  total do confronto x q(1-p)
    assists = assistencias por kill do time           x  total do confronto x q(p)
    q(p)    = 0,507 + 0,347 (p - 0,5)   (a parte das kills do jogo que fica com o time)

Distribuicao: binomial negativa (tamanho 6) -- kills de jogador variam mais que Poisson.
Medido no cego (Oracle's Elixir, ajuste ate 2025-06, teste 2025-07 a 2026-10; tudo pre-jogo):
  NLL antigo (media do recorte) x novo -- todos 124.450 linhas, internacionais 7.718:
    kills    todos +0,028   internacionais +0,016 IC95 [+0,011; +0,021]
    deaths   todos +0,017   internacionais +0,022
    assists  todos +0,038   internacionais +0,024
  log-loss Mais/Menos na linha da media: kills 0,678 -> 0,665 (internacionais 0,682 -> 0,674).
Resto conhecido: no internacional o azarao faz um pouco mais que o previsto (vies +0,4 kills).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

Q0, Q1 = 0.507, 0.347
TAMANHO_NB = 6.0
JANELA = 20
PESO_PRIOR = 10.0          # "jogos de kills do time" de encolhimento para a media da rota
STOMP_SHARE_LIMIAR = 0.20  # o mesmo filtro de atropelo de sempre (escolha dele)


def q(p: float) -> float:
    return Q0 + Q1 * (float(p) - 0.5)


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def priors_por_rota(df_pl: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Media da rota: fatia de kills, fatia de mortes, assistencias por kill do time."""
    d = df_pl.copy()
    for c in ("kills", "deaths", "assists", "teamkills", "teamdeaths"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d.dropna(subset=["kills", "deaths", "assists", "teamkills", "teamdeaths"])
    fora: Dict[str, Dict[str, float]] = {}
    for pos, g in d.groupby(d["pos"].astype(str)):
        tk, td = g["teamkills"].sum(), g["teamdeaths"].sum()
        if tk > 0 and td > 0:
            fora[pos] = {"kills": g["kills"].sum() / tk, "deaths": g["deaths"].sum() / td,
                         "assists": g["assists"].sum() / tk}
    return fora


def taxas_do_jogador(df_pl: pd.DataFrame, jogador: str, pos: str,
                     priors: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """As fatias do jogador nos ultimos JANELA jogos (qualquer liga), encolhidas para a rota."""
    d = df_pl[df_pl["player"].astype(str) == str(jogador)]
    if d.empty:
        return None
    d = d.sort_values("date").tail(JANELA)
    num = {c: pd.to_numeric(d[c], errors="coerce") for c in ("kills", "deaths", "assists", "teamkills", "teamdeaths")}
    pr = priors.get(str(pos)) or {"kills": 0.2, "deaths": 0.2, "assists": 0.5}
    tk, td = float(num["teamkills"].sum()), float(num["teamdeaths"].sum())
    return {"kills": (float(num["kills"].sum()) + PESO_PRIOR * pr["kills"]) / (tk + PESO_PRIOR),
            "deaths": (float(num["deaths"].sum()) + PESO_PRIOR * pr["deaths"]) / (td + PESO_PRIOR),
            "assists": (float(num["assists"].sum()) + PESO_PRIOR * pr["assists"]) / (tk + PESO_PRIOR),
            "n": int(len(d)),
            "media": {c: float(num[c].mean()) for c in ("kills", "deaths", "assists")}}


def total_do_confronto(df_pl: pd.DataFrame, time_a: str, time_b: str) -> Optional[float]:
    """O ritmo do confronto: media dos dois times do total de kills do jogo (ultimos JANELA)."""
    g = df_pl.drop_duplicates(subset=["gameid", "team"]).copy()
    g["_tot"] = pd.to_numeric(g["teamkills"], errors="coerce") + pd.to_numeric(g["teamdeaths"], errors="coerce")
    vals = []
    for t in (time_a, time_b):
        x = g[g["team"].astype(str) == str(t)].sort_values("date")["_tot"].dropna().tail(JANELA)
        if len(x) < 3:
            return None
        vals.append(float(x.mean()))
    return sum(vals) / 2


def projecao(taxas: Dict[str, float], stat: str, total: float, p_time: float) -> Optional[float]:
    if stat == "kills":
        return taxas["kills"] * total * q(p_time)
    if stat == "deaths":
        return taxas["deaths"] * total * q(1.0 - p_time)
    if stat == "assists":
        return taxas["assists"] * total * q(p_time)
    return None


def prob_mais(lam: float, linha: float, r: float = TAMANHO_NB) -> float:
    """P(X > linha) com X ~ binomial negativa (media lam, tamanho r)."""
    lam = max(1e-6, float(lam))
    k_max = int(math.floor(float(linha)))
    if k_max < 0:
        return 1.0
    p = r / (r + lam)
    termo = p ** r                       # P(X=0)
    acc = termo
    for k in range(1, k_max + 1):
        termo *= (k - 1 + r) / k * (1 - p)
        acc += termo
    return float(min(1.0, max(0.0, 1.0 - acc)))


def linha_justa(lam: float) -> float:
    """A linha .5 em que Mais e Menos ficam mais perto de 50%."""
    melhor, dist = 0.5, 9.0
    for k in range(0, 60):
        L = k + 0.5
        d = abs(prob_mais(lam, L) - 0.5)
        if d < dist:
            melhor, dist = L, d
    return melhor


def linha_de_prop(*, jogador: str, rota: str, time: str, stat: str, linha: Optional[float],
                  taxas: Dict[str, float], total: float, p_time: float) -> Dict[str, Any]:
    """Uma linha da tabela de props (mesmas chaves do motor antigo, para o resumo e o filtro)."""
    lam = projecao(taxas, stat, total, p_time)
    if lam is None:
        return {"ok": False}
    lam_neutro = projecao(taxas, stat, total, 0.5)
    L = float(linha) if linha is not None else linha_justa(lam)
    po = prob_mais(lam, L)
    share = min(1.0, abs(lam - lam_neutro) / max(1e-9, lam))
    return {"ok": True, "player": str(jogador), "role": str(rota), "team": str(time), "stat": stat,
            "line": L, "proj": lam, "proj_sem_gap": lam_neutro, "p_over": po, "p_under": 1 - po,
            "odd_justa_over": (1 / po) if po > 0 else float("nan"),
            "odd_justa_under": (1 / (1 - po)) if po < 1 else float("nan"),
            "linha_justa": linha_justa(lam), "media": taxas.get("media", {}).get(stat),
            "stomp_share": share, "movido_por_atropelo": bool(share >= STOMP_SHARE_LIMIAR)}
