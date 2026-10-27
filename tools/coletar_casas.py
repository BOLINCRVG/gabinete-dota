# -*- coding: utf-8 -*-
"""ODDS DE OUTRAS CASAS (OddsPapi) -- LoL ou Dota 2 (27/10/2026; mesmo molde do CS2).

Ele: "a pinnacle vai abrir quase sempre com a odd mais regulada... se
conseguissemos outras casas sinto que seria possivel ver a qualidade do modelo".
A Pinnacle vira a REGUA (preco justo); o valor esta nas casas moles. A OddsPapi
(plano gratis, chave dele no secret ODDS_API_KEY) junta 369 casas, inclusive as
dele: bet365, Superbet, Polymarket, Estrela Bet (e clones), Blaze -- e Betano/Stake.

Por corrida: 1 chamada de jogos (proximos 3 dias, com odds) + 1 do catalogo de
mercados + 1 de odds por jogo (no maximo MAX_JOGOS). Guarda:
  odds_casas_cs2.json            retrato: TODAS as casas, serie (171) e mapas 1-3
  odds_casas_cs2_historico.jsonl so as casas DELE, so o que mudou
A chave nunca e impressa. 401/403/429 = para, sai 75, nao insiste.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE = "https://api.oddspapi.io/v4/"
ESPORTES = {"lol": 18, "dota": 16}
ESPORTE = (sys.argv[1] if len(sys.argv) > 1 else "lol").lower()
SPORT_ID = ESPORTES[ESPORTE]
UA = "Gabinete/0.1 (modelo pessoal; contato: gptconta401@gmail.com)"
RAIZ = Path(__file__).resolve().parents[1]
ATUAL = RAIZ / f"odds_casas_{ESPORTE}.json"
HIST = RAIZ / f"odds_casas_{ESPORTE}_historico.jsonl"
MAX_JOGOS = 30
RC_BLOQUEIO = 75
# mercado -> (nome, periodo: 0 = serie, n = mapa n), outcomes (time1, time2). O catalogo da
# OddsPapi segue o mesmo padrao nos tres jogos: <esporte>1 = vencedor, <esporte>47/49/51 = mapas.
_s = str(SPORT_ID)
MERCADOS = {_s + "1": ("Série", 0, (_s + "1", _s + "2")),
            _s + "47": ("Mapa 1", 1, (_s + "47", _s + "48")),
            _s + "49": ("Mapa 2", 2, (_s + "49", _s + "50")),
            _s + "51": ("Mapa 3", 3, (_s + "51", _s + "52"))}
# as casas DELE (27/10/2026) -- prefixos do slug na OddsPapi; guardamos historico so delas
CASAS_DELE = ("pinnacle", "bet365", "superbet", "polymarket", "estrelabet", "blaze", "betano", "stake")


def e_casa_dele(slug: str) -> bool:
    return any(str(slug).startswith(p) for p in CASAS_DELE)


def chamar(caminho: str, chave: str, **params):
    url = BASE + caminho + "?" + urllib.parse.urlencode({**params, "apiKey": chave})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def precos(odds: dict) -> dict:
    """Pura: {casa: {mercado: [odd time1, odd time2]}} so dos MERCADOS, casa ativa e nao suspensa."""
    fora = {}
    for casa, b in (odds.get("bookmakerOdds") or {}).items():
        if not b.get("bookmakerIsActive", True) or b.get("suspended"):
            continue
        mk = b.get("markets") or {}
        for mid, (_, _, (o1, o2)) in MERCADOS.items():
            outs = (mk.get(mid) or {}).get("outcomes") or {}
            par = []
            for o in (o1, o2):
                pl = ((outs.get(o) or {}).get("players") or {}).get("0") or {}
                par.append(pl.get("price") if pl.get("active", True) else None)
            if all(isinstance(x, (int, float)) and x > 1 for x in par):
                fora.setdefault(casa, {})[mid] = par
    return fora


# LINHAS (28/10/2026, pedido dele: "ver onde tem a melhor odd e a melhor linha
# possivel... por exemplo para kills"). Do catalogo da OddsPapi (1 chamada por
# corrida), so os mercados de MAPA (p1..p5) de total, handicap e total por time --
# kills, torres, dragoes, barões, inibidores (Dota: kills, torres, roshans, barracks).
# Guardadas so para as casas DELE (o retrato ja e grande com o vencedor de todas).
_PERIODOS = {"p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5}


def catalogo_de_linhas(markets) -> dict:
    """Pura: {marketId: (unidade, tipo, mapa, pontos, (outcome A, outcome B), time)}.
    tipo total -> (Over, Under); spread -> (time 1, time 2), pontos = handicap do time 1;
    team_total -> (Over, Under) e qual = 1|2."""
    fora = {}
    for m in markets if isinstance(markets, list) else []:
        nome = str(m.get("marketName") or "")
        mapa = _PERIODOS.get(str(m.get("period") or ""))
        if m.get("playerProp") or not mapa or " - " not in nome or int(m.get("marketLength") or 0) != 2:
            continue
        unidade, resto = nome.split(" - ", 1)
        outs = {str(o.get("outcomeName")): str(o.get("outcomeId")) for o in m.get("outcomes") or []}
        qual = None
        if resto.startswith("Over Under Team 1"):
            tipo, qual = "team_total", 1
        elif resto.startswith("Over Under Team 2"):
            tipo, qual = "team_total", 2
        elif resto.startswith("Over Under"):
            tipo = "total"
        elif resto.startswith("Handicap"):
            tipo = "spread"
        else:
            continue
        par = (outs.get("Over"), outs.get("Under")) if tipo != "spread" else (outs.get("1"), outs.get("2"))
        if None in par or m.get("handicap") is None:
            continue
        fora[str(m["marketId"])] = (unidade.lower(), tipo, mapa, float(m["handicap"]), par, qual)
    return fora


def linhas_das_casas(odds: dict, catalogo: dict) -> dict:
    """Pura: {casa DELE: [[unidade, tipo, mapa, pontos, odd A, odd B, time], ...]}."""
    fora = {}
    for casa, b in (odds.get("bookmakerOdds") or {}).items():
        if not e_casa_dele(casa) or not b.get("bookmakerIsActive", True) or b.get("suspended"):
            continue
        for mid, mk in (b.get("markets") or {}).items():
            c = catalogo.get(str(mid))
            if not c:
                continue
            outs = (mk or {}).get("outcomes") or {}
            par = []
            for o in c[4]:
                pl = ((outs.get(o) or {}).get("players") or {}).get("0") or {}
                par.append(pl.get("price") if pl.get("active", True) else None)
            if all(isinstance(x, (int, float)) and x > 1 for x in par):
                fora.setdefault(casa, []).append([c[0], c[1], c[2], c[3], par[0], par[1], c[5]])
    return fora


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    chave = os.environ.get("ODDS_API_KEY", "").strip()
    if not chave:
        print(f"[casas {ESPORTE}] sem ODDS_API_KEY")
        return 1
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hoje = date.today()
    try:
        fx = chamar("fixtures", chave, sportId=SPORT_ID, hasOdds="true",
                    **{"from": hoje.isoformat(), "to": (hoje + timedelta(days=3)).isoformat()})
        time.sleep(1.0)
        catalogo = catalogo_de_linhas(chamar("markets", chave, sportId=SPORT_ID))
        jogos = []
        for f in (fx if isinstance(fx, list) else [])[:MAX_JOGOS]:
            time.sleep(1.0)
            od = chamar("odds", chave, fixtureId=f["fixtureId"])
            jogos.append({"fixtureId": f["fixtureId"], "time1": f.get("participant1Name"),
                          "time2": f.get("participant2Name"), "torneio": f.get("tournamentName"),
                          "inicio_utc": f.get("startTime"), "casas": precos(od),
                          "linhas": linhas_das_casas(od, catalogo)})
    except urllib.error.HTTPError as e:
        print(f"[casas {ESPORTE}] a API respondeu {e.code}: parando sem insistir")
        return RC_BLOQUEIO
    except (urllib.error.URLError, OSError) as e:
        print(f"[casas {ESPORTE}] rede: {type(e).__name__}")
        return RC_BLOQUEIO
    anterior = {}
    if ATUAL.exists():
        try:
            for j in json.loads(ATUAL.read_text(encoding="utf-8")).get("jogos", []):
                for casa, mk in j["casas"].items():
                    for mid, par in mk.items():
                        anterior[(j["fixtureId"], casa, mid)] = par
        except (OSError, ValueError, KeyError):
            anterior = {}
    ATUAL.write_text(json.dumps({"lido_em": agora, "jogos": jogos}, ensure_ascii=False), encoding="utf-8")
    mudou = 0
    with HIST.open("a", encoding="utf-8") as h:
        for j in jogos:
            for casa, mk in j["casas"].items():
                if not e_casa_dele(casa):
                    continue
                for mid, par in mk.items():
                    if anterior.get((j["fixtureId"], casa, mid)) != par:
                        h.write(json.dumps({"lido_em": agora, "fixtureId": j["fixtureId"], "time1": j["time1"],
                                            "time2": j["time2"], "inicio_utc": j["inicio_utc"], "casa": casa,
                                            "mercado": MERCADOS[mid][0], "odds": par}, ensure_ascii=False) + "\n")
                        mudou += 1
    n_casas = len({c for j in jogos for c in j["casas"]})
    n_lin = sum(len(v) for j in jogos for v in j["linhas"].values())
    print(f"[casas {ESPORTE}] {len(jogos)} jogos, {n_casas} casas com odd; {mudou} linhas das casas dele mudaram; "
          f"{n_lin} linhas de mapa (kills, torres...) das casas dele; catalogo {len(catalogo)} mercados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
