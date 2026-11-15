# -*- coding: utf-8 -*-
"""SONDA: como a Polymarket lista o Dota 2? (15/11/2026)

Os dominios da Polymarket nao resolvem no Brasil: roda no robo do GitHub e imprime no log
a tag do esporte, o formato do titulo e os tipos de mercado -- so reconhecimento, antes do coletor.
"""
from __future__ import annotations

import gzip
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://gamma-api.polymarket.com"


def _get(caminho: str, **params):
    url = f"{BASE}/{caminho}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 gabinete-sonda", "Accept": "application/json",
                                               "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = r.read()
            if d[:2] == b"\x1f\x8b":
                d = gzip.decompress(d)
            return r.status, json.loads(d.decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode("utf-8", "replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    st, sports = _get("sports")
    print("sports", st)
    tags = []
    for s in sports if isinstance(sports, list) else []:
        if "dota" in json.dumps(s).lower():
            print("ESPORTE", json.dumps({k: s.get(k) for k in ("id", "sport", "name", "tags", "primaryTagId", "series")}))
            if s.get("primaryTagId"):
                tags.append(s["primaryTagId"])
            tags += [t.strip() for t in str(s.get("tags") or "").split(",") if t.strip()]
    vistos = set()
    for tag in tags:
        if tag in vistos or str(tag) in ("1", "64"):
            continue
        vistos.add(tag)
        st, evs = _get("events", tag_id=tag, closed="false", limit=50)
        evs = evs if isinstance(evs, list) else []
        print(f"TAG {tag}: status {st}, {len(evs)} eventos")
        tipos = {}
        for e in evs:
            for m in e.get("markets") or []:
                t = str(m.get("sportsMarketType") or "")
                tipos[t] = tipos.get(t, 0) + 1
        print("TIPOS", json.dumps(tipos))
        for e in evs[:12]:
            print("TITULO", repr(e.get("title")), "| mercados", len(e.get("markets") or []))
        cheios = sorted(evs, key=lambda e: -len(e.get("markets") or []))[:2]
        for e in cheios:
            print("EVENTO", repr(e.get("title")), e.get("startDate"))
            for m in e.get("markets") or []:
                print("  M", json.dumps({k: m.get(k) for k in (
                    "sportsMarketType", "question", "groupItemTitle", "line", "outcomes", "outcomePrices",
                    "bestBid", "bestAsk", "liquidity", "gameStartTime", "closed")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
