# -*- coding: utf-8 -*-
"""COLETOR DA POLYMARKET: precos das partidas de LoL e de Dota 2 (23/10/2026; Dota e os
mercados alem do vencedor em 15/11/2026). Arquivo identico nos dois repositorios: o esporte
sai de onde ele roda (dota_paths.py na raiz = Dota).

Medido pelas sondas (tools/sonda_polymarket.py, 23/10; tools/sonda_polymarket_dota.py, 15/11):
  LoL  = esporte id 39, tag 65,     titulo "LoL: <A> vs <B> (BO<n>) - <torneio>"
  Dota = esporte id 38, tag 102366, titulo "Dota 2: <A> vs <B> (BO<n>) - <torneio>"
Cada partida e um EVENTO; os mercados dele, pelo campo sportsMarketType:
  moneyline             vencedor da serie          outcomes = [A, B]
  child_moneyline       "... - Game N Winner"      outcomes = [A, B]
  totals                "Games Total: O/U 2.5"     outcomes = [Over, Under], line = 2.5
  map_handicap          "Game Handicap: X (-1.5) vs Y (+1.5)"  outcomes = [X, Y], line = -1.5 (de X)
  kill_over_under_game  "Total Kills Over/Under 24.5 in Game 1?"  outcomes = [Over, Under], line
Os demais (baron, roshan, rampage...) nao tem conta no modelo e ficam de fora.

O PRECO. Polymarket e bolsa, nao casa: nao ha margem embutida, ha o spread entre a melhor
compra e a melhor venda. Guardamos o MEIO do livro ((bestBid + bestAsk) / 2) como
probabilidade do primeiro lado; sem livro, o outcomePrices. O spread e a liquidez vao junto:
quem le decide se o livro esta cheio o bastante (app_core/polymarket_casa_core.py).

Os dominios da Polymarket nao resolvem no Brasil (conferido 23/10): roda no robo do GitHub
(EUA), .github/workflows/polymarket.yml.

FONTES: API Gamma publica (https://gamma-api.polymarket.com), formato medido pelas sondas
(evidence/sonda_polymarket.json e o log da sonda do Dota de 15/11/2026). Desde 12/11/2026 ela
exige "Accept-Encoding: gzip" para paginas com limit > 50 (406 gzip_required).

Uso: python tools/coletar_polymarket.py     (sai 75 se a rede falhar)
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://gamma-api.polymarket.com"
TAG_LOL = 65
TAG_DOTA = 102366
RAIZ = Path(__file__).resolve().parent.parent
ESPORTE = "dota" if (RAIZ / "dota_paths.py").exists() else "lol"
TAG = TAG_DOTA if ESPORTE == "dota" else TAG_LOL
ATUAL = RAIZ / f"polymarket_{ESPORTE}.json"
HISTORICO = RAIZ / f"polymarket_{ESPORTE}_historico.jsonl"
_TITULO = re.compile(r"^(?:LoL|Dota 2):\s*(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)\s*\(BO(?P<bo>\d)\)\s*(?:-\s*(?P<torneio>.+))?$")
TIPOS_GUARDADOS = ("child_moneyline", "totals", "map_handicap", "kill_over_under_game")
_GAME = re.compile(r"\bGame\s+(\d)\b", re.I)


def _get(caminho: str, **params):
    url = f"{BASE}/{caminho}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 gabinete", "Accept": "application/json",
                                               "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(_corpo(r.read(), r.headers.get("Content-Encoding")).decode("utf-8"))


def _corpo(dados: bytes, codificacao) -> bytes:
    """Pura: o corpo da resposta, descomprimido se veio em gzip."""
    if "gzip" in str(codificacao or "").lower() or dados[:2] == b"\x1f\x8b":
        import gzip
        return gzip.decompress(dados)
    return dados


def _lista(v):
    if isinstance(v, list):
        return v
    try:
        x = json.loads(v or "[]")
        return x if isinstance(x, list) else []
    except (TypeError, ValueError):
        return []


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _ja_comecou(inicio, agora: datetime) -> bool:
    try:
        dt = datetime.fromisoformat(str(inicio).replace("Z", "+00:00").replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt <= agora
    except (TypeError, ValueError):
        return False


def _preco(x: dict):
    """(p do primeiro lado, fonte, spread) de um mercado, ou None. Pura."""
    precos = [_f(p) for p in _lista(x.get("outcomePrices"))]
    bid, ask = _f(x.get("bestBid")), _f(x.get("bestAsk"))
    if bid is not None and ask is not None and 0 < bid <= ask < 1:
        return (bid + ask) / 2, "meio_do_livro", round(ask - bid, 4)
    if len(precos) == 2 and precos[0] is not None and 0 < precos[0] < 1:
        return precos[0], "ultimo_preco", None
    return None


def ler_partida(ev: dict, agora: datetime | None = None) -> dict | None:
    """Evento da Gamma -> partida com preco do time A. Pura.

    So PRE-JOGO: depois do inicio o preco e ao vivo (ou ja resolvido, 0,0005
    num jogo acabado, visto na sonda) e nao se compara com o modelo pre-jogo."""
    m = _TITULO.match(str(ev.get("title") or "").strip())
    if not m:
        return None
    ml = next((x for x in (ev.get("markets") or []) if x.get("sportsMarketType") == "moneyline"), None)
    if not ml or ml.get("closed"):
        return None
    times = [str(t) for t in _lista(ml.get("outcomes"))]
    if len(times) != 2:
        return None
    if _ja_comecou(ml.get("gameStartTime") or ev.get("startDate"), agora or datetime.now(timezone.utc)):
        return None
    pr = _preco(ml)
    if pr is None:
        precos = [_f(p) for p in _lista(ml.get("outcomePrices"))]     # compat.: 0 ou 1 de ultimo preco
        if len(precos) == 2 and precos[0] is not None:
            pr = (precos[0], "ultimo_preco", None)
        else:
            return None
    p_a, fonte, spread = pr
    bid, ask = _f(ml.get("bestBid")), _f(ml.get("bestAsk"))
    return {
        "time_a": times[0], "time_b": times[1],
        "p_a": round(p_a, 4), "p_b": round(1 - p_a, 4),
        "fonte_preco": fonte,
        "spread": None if bid is None or ask is None else round(ask - bid, 4),
        "liquidez": _f(ml.get("liquidity")), "volume": _f(ml.get("volume")),
        "bo": int(m.group("bo")), "torneio": (m.group("torneio") or "").strip(),
        "inicio_utc": ml.get("gameStartTime") or ev.get("startDate"),
        "evento": ev.get("slug"),
        "mercados": outros_mercados(ev),
    }


def outros_mercados(ev: dict) -> list:
    """Pura: os mercados do evento alem do vencedor da serie que o modelo sabe precificar
    (TIPOS_GUARDADOS). Cada um: tipo, mapa (0 = serie), linha, os dois lados, o preco do
    PRIMEIRO lado (meio do livro; sem livro, o ultimo preco), spread e liquidez."""
    fora = []
    for x in ev.get("markets") or []:
        tipo = x.get("sportsMarketType")
        if tipo not in TIPOS_GUARDADOS or x.get("closed"):
            continue
        lados = [str(t) for t in _lista(x.get("outcomes"))]
        pr = _preco(x)
        if len(lados) != 2 or pr is None:
            continue
        g = _GAME.search(str(x.get("question") or ""))
        mapa = int(g.group(1)) if g and tipo in ("child_moneyline", "kill_over_under_game") else 0
        if tipo in ("child_moneyline", "kill_over_under_game") and not mapa:
            continue
        linha = _f(x.get("line"))
        if tipo != "child_moneyline" and linha is None:
            continue
        fora.append({"tipo": tipo, "mapa": mapa, "linha": linha, "lados": lados,
                     "p1": round(pr[0], 4), "fonte_preco": pr[1], "spread": pr[2],
                     "liquidez": _f(x.get("liquidity"))})
    return fora


def main() -> int:
    try:
        evs = []
        for pagina in range(4):                       # ate 400 eventos (ha mercados que nao sao jogo)
            lote = _get("events", tag_id=TAG, closed="false", limit=100, offset=100 * pagina)
            lote = lote if isinstance(lote, list) else []
            evs += lote
            if len(lote) < 100:
                break
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[polymarket] rede/API falhou: {type(e).__name__}: {e}")
        return 75
    agora = datetime.now(timezone.utc)
    quando = agora.isoformat(timespec="seconds")
    partidas = [p for p in (ler_partida(e, agora) for e in evs) if p]
    ATUAL.write_text(json.dumps({"coletado_em": quando, "fonte": f"{BASE}/events?tag_id={TAG}",
                                 "partidas": partidas}, ensure_ascii=False, indent=1), encoding="utf-8")
    with HISTORICO.open("a", encoding="utf-8") as f:
        for p in partidas:                            # o historico guarda so o vencedor (o arquivo cresce a cada hora)
            f.write(json.dumps({"coletado_em": quando, **{k: v for k, v in p.items() if k != "mercados"}},
                               ensure_ascii=False) + "\n")
    n_merc = sum(len(p["mercados"]) for p in partidas)
    print(f"[polymarket {ESPORTE}] {len(partidas)} partida(s) com preco de vencedor; {n_merc} outros mercados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
