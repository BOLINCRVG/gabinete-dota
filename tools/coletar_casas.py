# -*- coding: utf-8 -*-
"""ODDS DE OUTRAS CASAS (OddsPapi) -- LoL ou Dota 2 (27/10/2026; mesmo molde do CS2).

Ele: "a pinnacle vai abrir quase sempre com a odd mais regulada... se
conseguissemos outras casas sinto que seria possivel ver a qualidade do modelo".
A Pinnacle vira a REGUA (preco justo); o valor esta nas casas moles. A OddsPapi
(plano gratis, chave dele no secret ODDS_API_KEY) junta 369 casas, inclusive as
dele: bet365, Superbet, Polymarket, Estrela Bet (e clones), Blaze -- e Betano/Stake.

Por corrida (uma por dia): a conta (gratis) + 1 chamada de jogos + as odds que couberem na cota
do mes (ver "A COTA" abaixo). Guarda:
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
        # +-1,5 DE MAPAS da serie (29/10/2026, ele: "pegar das casas o +1,5 e -1,5 dos times"):
        # "Maps Handicap", periodo "result" -> unidade "maps", mapa 0, pontos = handicap do time 1
        if nome == "Maps Handicap" and str(m.get("period")) == "result" and int(m.get("marketLength") or 0) == 2:
            outs = {str(o.get("outcomeName")): str(o.get("outcomeId")) for o in m.get("outcomes") or []}
            if outs.get("1") and outs.get("2") and m.get("handicap") is not None:
                fora[str(m["marketId"])] = ("maps", "spread", 0, float(m["handicap"]), (outs["1"], outs["2"]), None)
            continue
        # MERCADOS DA SERIE (29/10/2026, achados no resumo do catalogo): total de mapas, placar
        # exato e os "Kills - ..." do periodo "result" (kills da serie, de cada time, handicap).
        if str(m.get("period")) == "result":
            outs = {str(o.get("outcomeName")): str(o.get("outcomeId")) for o in m.get("outcomes") or []}
            if nome == "Total Maps Over Under" and outs.get("Over") and outs.get("Under") \
                    and m.get("handicap") is not None:
                fora[str(m["marketId"])] = ("maps", "total", 0, float(m["handicap"]),
                                            (outs["Over"], outs["Under"]), None)
                continue
            if nome == "Correct Map Score":
                # um par por diferenca de mapas: (time 1 a:b, time 2 a:b), pontos = a - b
                pares = []
                for o in outs:
                    a, _, b = o.partition(":")
                    if a.isdigit() and b.isdigit() and int(a) > int(b) and f"{b}:{a}" in outs:
                        pares.append(("maps", "placar", 0, float(int(a) - int(b)),
                                      (outs[o], outs[f"{b}:{a}"]), None))
                if pares:
                    fora[str(m["marketId"])] = pares
                continue
            mapa = 0
        if m.get("playerProp") or mapa is None or " - " not in nome or int(m.get("marketLength") or 0) != 2:
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
            cs = catalogo.get(str(mid))
            if not cs:
                continue
            outs = (mk or {}).get("outcomes") or {}
            for c in (cs if isinstance(cs, list) else [cs]):      # o placar exato tem varios pares
                par = []
                for o in c[4]:
                    pl = ((outs.get(o) or {}).get("players") or {}).get("0") or {}
                    par.append(pl.get("price") if pl.get("active", True) else None)
                if all(isinstance(x, (int, float)) and x > 1 for x in par):
                    fora.setdefault(casa, []).append([c[0], c[1], c[2], c[3], par[0], par[1], c[5]])
    return fora


def resumo_do_catalogo(markets, odds_por_jogo) -> list:
    """Pura: os mercados do catalogo que as casas DELE cotam, agrupados por (nome, periodo, de jogador):
    quantas linhas, exemplos de handicap, nomes dos outcomes, casas e chaves de jogador.
    (29/10/2026: para achar os nomes da duracao do mapa e das linhas de jogador.)"""
    por_id = {str(m.get("marketId")): m for m in (markets if isinstance(markets, list) else [])}
    grupos: dict = {}
    for od in odds_por_jogo:
        for casa, b in (od.get("bookmakerOdds") or {}).items():
            if not e_casa_dele(casa):
                continue
            for mid, mk in (b.get("markets") or {}).items():
                m = por_id.get(str(mid))
                if not m:
                    continue
                k = (str(m.get("marketName")), str(m.get("period")), bool(m.get("playerProp")))
                g = grupos.setdefault(k, {"nome": k[0], "periodo": k[1], "jogador": k[2], "linhas": 0,
                                          "handicaps": set(), "outcomes": set(), "casas": set(), "chaves": set()})
                g["linhas"] += 1
                if m.get("handicap") is not None:
                    g["handicaps"].add(float(m["handicap"]))
                g["casas"].add(str(casa).split(".")[0])
                nomes = {str(x.get("outcomeId")): str(x.get("outcomeName")) for x in m.get("outcomes") or []}
                for oid, o in ((mk or {}).get("outcomes") or {}).items():
                    g["outcomes"].add(nomes.get(str(oid), str(oid)))
                    for pk, pv in ((o or {}).get("players") or {}).items():
                        if len(g["chaves"]) < 4:
                            g["chaves"].add(f"{pk}:{(pv or {}).get('playerName', '')}")
    fora = []
    for g in grupos.values():
        fora.append({**g, "handicaps": sorted(g["handicaps"])[:12], "outcomes": sorted(g["outcomes"])[:6],
                     "casas": sorted(g["casas"]), "chaves": sorted(g["chaves"])})
    return sorted(fora, key=lambda g: (-g["linhas"], g["nome"]))


# ---- A COTA (15/11/2026) ---------------------------------------------------------------------
# O plano gratis da OddsPapi da 250 chamadas POR MES (zera no dia 1o) para a chave inteira, e os
# robos de LoL, Dota e CS gastavam ~90 por dia: a cota acabava no dia 5 e as casas ficavam o resto
# do mes paradas. Agora cada corrida pergunta a conta (/account nao gasta) e so gasta a sua parte:
#   por corrida = (o que resta - reserva) / dias que faltam no mes / esportes que dividem a chave
# 1 chamada de jogos + o resto em odds. O catalogo de mercados vem de um arquivo versionado.
N_ESPORTES = 3            # LoL, Dota e CS dividem a mesma chave
RESERVA = 12              # folga para corrida manual / erro
CASA_DO_LOTE = "stake"    # /odds-by-tournaments so aceita UMA casa; Pinnacle e Polymarket tem robo proprio
MAX_TORNEIOS_POR_LOTE = 3
CATALOGO_FIXO = RAIZ / f"catalogo_mercados_{ESPORTE}.json"


def cota(conta: dict):
    """Pura: (limite, usadas) da assinatura ativa; (None, None) se a resposta nao trouxer."""
    for a in (conta or {}).get("subscriptions") or []:
        if a.get("is_active") and a.get("request_limit") is not None:
            return int(a["request_limit"]), int(a.get("request_count") or 0)
    return None, None


def orcamento(limite, usadas, hoje: date, n_esportes: int = N_ESPORTES, reserva: int = RESERVA) -> int:
    """Pura: quantas chamadas ESTA corrida pode gastar (uma corrida por dia por esporte)."""
    if limite is None:
        return 2
    prox = date(hoje.year + (hoje.month == 12), hoje.month % 12 + 1, 1)
    dias = max(1, (prox - hoje).days)
    return max(0, int((limite - usadas - reserva) / dias / max(1, n_esportes)))


def plano_de_coleta(fixtures: list, chamadas: int):
    """Pura: como gastar `chamadas` em odds. ("jogos", [fixture...]) = 1 chamada por jogo, todas as
    casas; ("lotes", [[tournamentId...]...]) = 1 chamada por ate 3 torneios, so a CASA_DO_LOTE.
    Jogo a jogo quando cabem todos; senao os torneios com jogo mais cedo primeiro."""
    fx = sorted([f for f in fixtures if f.get("fixtureId")], key=lambda f: str(f.get("startTime") or ""))
    if chamadas <= 0 or not fx:
        return "nada", []
    if len(fx) <= chamadas:
        return "jogos", fx
    ordem = []
    for f in fx:
        t = f.get("tournamentId")
        if t is not None and t not in ordem:
            ordem.append(t)
    lotes = [ordem[i:i + MAX_TORNEIOS_POR_LOTE] for i in range(0, len(ordem), MAX_TORNEIOS_POR_LOTE)]
    if not lotes:
        return "jogos", fx[:chamadas]
    return "lotes", lotes[:chamadas]


def carregar_catalogo(caminho: Path = CATALOGO_FIXO) -> dict:
    """O catalogo de linhas versionado (gerado de uma leitura de /markets; nao gasta cota)."""
    try:
        cru = json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    tup = lambda c: (c[0], c[1], int(c[2]), float(c[3]), tuple(c[4]), c[5])
    return {k: ([tup(c) for c in v] if v and isinstance(v[0], list) else tup(v)) for k, v in cru.items()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    chave = os.environ.get("ODDS_API_KEY", "").strip()
    if not chave:
        print(f"[casas {ESPORTE}] sem ODDS_API_KEY")
        return 1
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hoje = date.today()
    try:
        limite, usadas = cota(chamar("account", chave))
    except (urllib.error.URLError, OSError, ValueError):
        limite, usadas = None, None
    pode = orcamento(limite, usadas, hoje)
    print(f"[casas {ESPORTE}] cota do mes: {usadas} de {limite} usadas; esta corrida pode gastar {pode}")
    if pode < 2:
        print(f"[casas {ESPORTE}] sem cota para esta corrida (zera no dia 1o): nada escrito")
        return RC_BLOQUEIO
    catalogo = carregar_catalogo()
    try:
        fx = chamar("fixtures", chave, sportId=SPORT_ID, hasOdds="true",
                    **{"from": hoje.isoformat(), "to": (hoje + timedelta(days=2)).isoformat()})
        fx = fx if isinstance(fx, list) else []
        por_id = {f["fixtureId"]: f for f in fx if f.get("fixtureId")}
        modo, alvo = plano_de_coleta(fx, pode - 1)
        lidos = []                                            # [(fixture, odds)]
        if modo == "jogos":
            for f in alvo:
                time.sleep(1.2)
                lidos.append((f, chamar("odds", chave, fixtureId=f["fixtureId"])))
        elif modo == "lotes":
            for lote in alvo:
                time.sleep(1.2)
                r = chamar("odds-by-tournaments", chave, tournamentIds=",".join(str(t) for t in lote),
                           bookmaker=CASA_DO_LOTE)
                for od in (r if isinstance(r, list) else [r] if isinstance(r, dict) else []):
                    f = por_id.get(od.get("fixtureId"))
                    if f:
                        lidos.append((f, od))
        print(f"[casas {ESPORTE}] {len(fx)} jogos na agenda da API; coleta por {modo}: {len(lidos)} jogos lidos")
        jogos = []
        for f, od in lidos:
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
    if not jogos:
        print(f"[casas {ESPORTE}] nenhum jogo lido: o retrato anterior fica")
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
          f"{n_lin} linhas de mapa (kills, torres...) das casas dele; catalogo versionado {len(catalogo)} mercados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
