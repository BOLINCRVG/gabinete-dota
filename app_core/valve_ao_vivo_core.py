# -*- coding: utf-8 -*-
"""DRAFT AO VIVO PELA API OFICIAL DA VALVE (Dota, 24/11/2026). Aprovado por ele ("sim, para ambos").

A sonda de 24/11 (evidence/sonda_steam_dota.json) mostrou: GetLiveLeagueGames responde (25 jogos de liga ao
vivo); GetMatchDetails da erro 500 (quebrado na Valve ha anos) -- por isso a Valve serve para o DRAFT AO
VIVO, nao para os resultados. Daqui vem, por jogo: os dois times e o lado de cada um (Radiant/Dire, sem chute),
picks e bans, QUEM joga cada heroi (account_id: sem adivinhar pelo historico), patrimonio e placar.
POSICOES: a Valve nao diz a posicao (pos 1..5). Com o jogo andando (>= 8 min) a ordem sai do patrimonio
(maior = pos 1); antes disso, os herois de suporte (papel "Support" da OpenDota) vao para 4-5 -- e a tela
avisa que as posicoes sao estimadas (a captura da cyberscore continua dando a ordem exata).
Nunca guarda nome de conta Steam (so account_id -> nome profissional da nossa base).
A chave: STEAM_API_KEY nos Secrets do Streamlit (ou variavel de ambiente). Sem chave: a tela diz como cadastrar.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

URL = "https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={}"
MIN_PATRIMONIO_S = 8 * 60


def chave(st: Any = None) -> str:
    v = os.environ.get("STEAM_API_KEY", "").strip()
    if v or st is None:
        return v
    try:
        return str(st.secrets.get("STEAM_API_KEY", "") or "").strip()
    except Exception:  # noqa: BLE001 -- sem secrets.toml o Streamlit levanta
        return ""


def _get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "GabineteDota/0.1"}), timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _ids(lst: Any) -> List[str]:
    return [str(x.get("hero_id")) for x in (lst or []) if isinstance(x, Mapping) and x.get("hero_id")]


def ordenar_posicoes(jogadores: List[Dict[str, Any]], duracao: float, papeis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Pura. 5 jogadores de um lado -> ordem pos 1..5 estimada (ver o topo)."""
    if duracao >= MIN_PATRIMONIO_S and all(j.get("patrimonio") is not None for j in jogadores):
        return sorted(jogadores, key=lambda j: -float(j["patrimonio"]))
    sup = lambda j: "Support" in ((papeis.get(str(j["heroi"])) or {}).get("papeis") or [])  # noqa: E731
    return [j for j in jogadores if not sup(j)] + [j for j in jogadores if sup(j)]


def ler_jogos(dados: Mapping[str, Any], papeis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Pura. Resposta do GetLiveLeagueGames -> [{match_id, liga, radiant, dire, picks: (rad, dire), bans: (rad,
    dire), contas: {heroi: conta}, duracao, abates: (rad, dire), posicoes_estimadas}]. So jogos com os 10 picks."""
    out = []
    for g in ((dados or {}).get("result") or {}).get("games") or []:
        sb = g.get("scoreboard") or {}
        rad, dire = sb.get("radiant") or {}, sb.get("dire") or {}
        pr, pd_ = _ids(rad.get("picks")), _ids(dire.get("picks"))
        if len(pr) != 5 or len(pd_) != 5:
            continue
        dur = float(sb.get("duration") or 0)
        lados = {}
        for nome_lado, bloco in (("rad", rad), ("dire", dire)):
            js = [{"conta": str(p.get("account_id") or ""), "heroi": str(p.get("hero_id")),
                   "patrimonio": p.get("net_worth")} for p in (bloco.get("players") or []) if p.get("hero_id")]
            lados[nome_lado] = js
        contas = {j["heroi"]: j["conta"] for js in lados.values() for j in js if j["conta"]}
        ord_r = ordenar_posicoes(lados["rad"], dur, papeis) if len(lados["rad"]) == 5 else [{"heroi": h} for h in pr]
        ord_d = ordenar_posicoes(lados["dire"], dur, papeis) if len(lados["dire"]) == 5 else [{"heroi": h} for h in pd_]
        out.append({"match_id": str(g.get("match_id") or ""), "liga": g.get("league_id"),
                    "radiant": str((g.get("radiant_team") or {}).get("team_name") or ""),
                    "dire": str((g.get("dire_team") or {}).get("team_name") or ""),
                    "picks": ([j["heroi"] for j in ord_r], [j["heroi"] for j in ord_d]),
                    "bans": (_ids(rad.get("bans")), _ids(dire.get("bans"))),
                    "contas": contas, "duracao": dur,
                    "contas_lado": ([j["conta"] for j in lados["rad"] if j["conta"]],
                                    [j["conta"] for j in lados["dire"] if j["conta"]]),
                    "abates": (int(rad.get("score") or 0), int(dire.get("score") or 0)),
                    "posicoes_estimadas": True})
    return out


MIN_JOGADORES_DO_TIME = 3   # de 5: o lado e daquele time se 3+ contas estao no elenco dele na base


def time_pelos_jogadores(contas: List[str], elencos: Mapping[str, Any]) -> str:
    """Pura. Achado com a chave real (24/11/2026): em 4 de 6 jogos ao vivo a Valve manda o nome do time VAZIO.
    O lado vira o time da base que tem 3+ destas contas no elenco (um so; empate = nao sabe)."""
    cs = {str(c) for c in contas if c}
    melhores = sorted(((len(cs & {str(x) for x in (e.get("jogadores") or [])}), t) for t, e in elencos.items()),
                      reverse=True)
    if not melhores or melhores[0][0] < MIN_JOGADORES_DO_TIME:
        return ""
    if len(melhores) > 1 and melhores[1][0] == melhores[0][0]:
        return ""
    return melhores[0][1]


def achar(jogos: List[Dict[str, Any]], t1: str, t2: str, resolver: Callable[[str], str],
          elencos: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Pura. O jogo ao vivo dos dois times, ou None. Cada lado: pelo nome (resolvido para a base); se a Valve
    mandou o nome vazio ou desconhecido, pelos jogadores (elenco da base)."""
    alvo = {t1, t2}
    for j in jogos:
        lados = []
        for k, nome in ((0, j.get("radiant")), (1, j.get("dire"))):
            try:
                r = resolver(nome) if nome else ""
            except Exception:  # noqa: BLE001
                r = ""
            if not r and elencos:
                r = time_pelos_jogadores((j.get("contas_lado") or ([], []))[k], elencos)
            lados.append(r or "")
        if set(lados) == alvo:
            return {**j, "radiant_base": lados[0], "dire_base": lados[1]}
    return None


def buscar(chave_: str, papeis: Mapping[str, Any], get: Callable[[str], Dict[str, Any]] = _get) -> List[Dict[str, Any]]:
    return ler_jogos(get(URL.format(chave_)), papeis)
