# -*- coding: utf-8 -*-
"""AGENDA DO DOTA pela Liquipedia (17/11/2026).

Ele: "ja tem literalmente VARIOS jogos que ainda nao estao no modelo" (European Pro League 41 de
amanha). Causa: o Dota nao tinha agenda -- a lista so mostrava o que a Pinnacle/Polymarket ja
precificavam (a PandaScore nunca teve chave). Agora: 1 leitura de Liquipedia:Matches (dota2) por
rodada do robo -> agenda_dota.json; a gaveta mostra todo jogo, com ou sem odd.

Regras da Liquipedia: User-Agent proprio, gzip, action=parse e chamada PESADA (1 a cada 30 s; aqui
e uma so por rodada). Dados CC-BY-SA (citados na tela).
Uso: python tools/agenda_dota.py [--html ARQUIVO]  (--html: le um parse salvo, sem rede)
"""
from __future__ import annotations

import argparse
import gzip
import html as _html
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1]
ARQ = RAIZ / "agenda_dota.json"
URL = "https://liquipedia.net/dota2/api.php?action=parse&page=Liquipedia:Matches&format=json&prop=text"
UA = "GabineteDota/0.1 (modelo pessoal de previsao; dados citados como Liquipedia CC-BY-SA)"

_BLOCO = re.compile(r'<div class="match-info">(.*?)(?=<div class="match-info">|\Z)', re.S)
_TS = re.compile(r'data-timestamp="(\d+)"')
_OPON = re.compile(r'match-info-header-opponent(?: match-info-header-opponent-left)?">(.*?)</div></div>', re.S)
_TITULO = re.compile(r'<a href="[^"]*" (?:class="new" )?title="([^"]+)"')
_BO = re.compile(r'\(?Bo(\d)\)?')
_TORNEIO = re.compile(r'match-info-tournament">.*?title="([^"#]+)(?:#[^"]*)?"', re.S)
_TORNEIO_NOME = re.compile(r'match-info-tournament">.*?<span>([^<]+)</span></a>', re.S)


def _nome(bloco: str) -> str:
    m = _TITULO.search(bloco)
    if not m:
        return ""
    t = _html.unescape(m.group(1))
    t = re.sub(r"\s*\(page does not exist\)$", "", t)
    return re.sub(r"\s*\((?:team|Dota 2 team)\)$", "", t).strip()  # pagina desambiguada: "LEGION (team)"


# FORMATO NOVO (achado 24/11/2026): a Liquipedia trocou o bloco "match-info" por cartoes
# "match-ticker-card". O leitor antigo achava os 100 jogos mas nenhum time -> a agenda do Dota ficou
# VAZIA de 20/11 a 24/11 sem ninguem saber. Agora: formato novo primeiro, o antigo como reserva, e
# main() avisa alto se a pagina tem jogos e a leitura deu zero.
_CARTAO = re.compile(r'<div class="match-ticker-card">(.*?)(?=<div class="match-ticker-card">|\Z)', re.S)
# "opponent" ou "opponent winner" (jogo com placar): o time vencedor tem classe a mais
_OPON_N = re.compile(r'match-ticker-card(?:&#95;|_){2}opponent(?: [a-z-]+)*">(.*?)(?=match-ticker-card(?:&#95;|_){2}'
                     r'opponent(?: [a-z-]+)*"|match-ticker-card-tournament|\Z)', re.S)
_TORNEIO_N = re.compile(r'match-ticker-card-tournament-name"><a href="[^"]*" title="([^"#]+)[^"]*">([^<]+)</a>')


def _ler_cartoes(html_txt: str) -> list:
    out, vistos = [], set()
    for b in _CARTAO.findall(html_txt):
        ts = _TS.search(b)
        ops = _OPON_N.findall(b)
        if not ts or len(ops) < 2:
            continue
        a, c = _nome(ops[0]), _nome(ops[1])
        if not a or not c or a.upper() == "TBD" or c.upper() == "TBD":
            continue
        chave = (a, c, int(ts.group(1)))
        if chave in vistos:
            continue
        vistos.add(chave)
        bo = _BO.search(b)
        tn = _TORNEIO_N.search(b)
        ini = datetime.fromtimestamp(int(ts.group(1)), tz=timezone.utc)
        out.append({"time_a": a, "time_b": c, "inicio_utc": ini.strftime("%Y-%m-%d %H:%M:%S+00"),
                    "bo": int(bo.group(1)) if bo else None,
                    "torneio": _html.unescape(tn.group(2)).strip() if tn else "",
                    "pagina": _html.unescape(tn.group(1)).strip() if tn else ""})
    return out


def ler(html_txt: str) -> list:
    """Pura: HTML do Liquipedia:Matches -> [{time_a, time_b, inicio_utc, bo, torneio, pagina}]."""
    novos = _ler_cartoes(html_txt)
    if novos:
        return novos
    out, vistos = [], set()
    for b in _BLOCO.findall(html_txt):
        ts = _TS.search(b)
        ops = _OPON.findall(b)
        if not ts or len(ops) < 2:
            continue
        a, c = _nome(ops[0]), _nome(ops[1])
        if not a or not c or a.upper() == "TBD" or c.upper() == "TBD":
            continue
        ini = datetime.fromtimestamp(int(ts.group(1)), tz=timezone.utc)
        bo = _BO.search(b)
        tp = _TORNEIO.search(b)
        tn = _TORNEIO_NOME.search(b)
        chave = (a, c, int(ts.group(1)))
        if chave in vistos:
            continue
        vistos.add(chave)
        out.append({"time_a": a, "time_b": c, "inicio_utc": ini.strftime("%Y-%m-%d %H:%M:%S+00"),
                    "bo": int(bo.group(1)) if bo else None,
                    "torneio": _html.unescape(tn.group(1)).strip() if tn else "",
                    "pagina": _html.unescape(tp.group(1)).strip() if tp else ""})
    return out


def baixar() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dados = r.read()
        if "gzip" in str(r.headers.get("Content-Encoding") or "").lower() or dados[:2] == b"\x1f\x8b":
            dados = gzip.decompress(dados)
    return json.loads(dados.decode("utf-8"))["parse"]["text"]["*"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None)
    a = ap.parse_args()
    try:
        if a.html:
            txt = Path(a.html).read_text(encoding="utf-8")
            if txt.lstrip().startswith("{"):
                txt = json.loads(txt)["parse"]["text"]["*"]
        else:
            txt = baixar()
    except Exception as e:  # noqa: BLE001
        print(f"[agenda_dota] Liquipedia nao respondeu ({type(e).__name__}: {e}); agenda anterior fica")
        return 0
    agora = datetime.now(timezone.utc)
    jogos = [j for j in ler(txt)
             if datetime.strptime(j["inicio_utc"][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
             >= agora - timedelta(hours=6)]
    n_pagina = txt.count('class="match-ticker-card"') + txt.count('class="match-info"')
    doc = {"coletado_em": agora.isoformat(timespec="seconds"), "fonte": "Liquipedia (CC-BY-SA): Liquipedia:Matches",
           "partidas": jogos, "jogos_na_pagina": n_pagina}
    if n_pagina >= 10 and not ler(txt):
        doc["aviso"] = (f"a pagina tem {n_pagina} jogos e a leitura achou 0 -- a Liquipedia mudou o formato; "
                        "a agenda NAO esta confiavel")
        print(f"::warning::[agenda_dota] {doc['aviso']}")
    ARQ.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[agenda_dota] {len(jogos)} partidas -> {ARQ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
