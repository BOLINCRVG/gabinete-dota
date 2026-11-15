# -*- coding: utf-8 -*-
"""ROBO DO PLACAR DAS APOSTAS (15/11/2026). Arquivo identico nos dois repositorios.

A cada corrida (.github/workflows/apostas.yml, de 3 em 3 horas):
  1. GRAVA as apostas com valor nas casas dele que ainda nao estavam no arquivo
     (app_core/placar_apostas_core.candidatos: regua = Pinnacle sem margem);
  2. FECHA as que ja comecaram, pelo resultado oficial do mercado na Polymarket.
Arquivo: apostas_<esporte>.jsonl, uma aposta por linha.

A Polymarket nao abre do Brasil: o passo 2 so funciona no robo (EUA). Rede fora = sai 75 depois
de gravar o passo 1 (o que ja foi visto nao se perde).

Uso: python tools/placar_apostas.py [--sem-rede]
"""
from __future__ import annotations

import json
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app_core import placar_apostas_core as pac  # noqa: E402
from app_core import suas_casas_core as sc  # noqa: E402

ESPORTE = sc.esporte(RAIZ)
ARQUIVO = RAIZ / f"apostas_{ESPORTE}.jsonl"
DESISTE_DIAS = 10                 # mercado que nao fechou em 10 dias: "sem_resultado"


def ler(arq: Path = ARQUIVO) -> list:
    if not arq.exists():
        return []
    fora = []
    for l in arq.read_text(encoding="utf-8").splitlines():
        try:
            fora.append(json.loads(l))
        except ValueError:
            continue
    return fora


def gravar(apostas: list, arq: Path = ARQUIVO) -> None:
    arq.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in apostas), encoding="utf-8")


def registrar(apostas: list, agora: datetime) -> int:
    doc = sc.carregar(RAIZ, agora)
    try:
        pin = json.loads((RAIZ / "odds_pinnacle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pin = {}
    cands = []
    for jogo in doc.get("jogos") or []:
        if jogo.get("polymarket_evento"):
            cands += pac.candidatos(jogo, pin, ESPORTE)
    nov = pac.novas(cands, apostas, agora, ESPORTE)
    apostas += nov
    return len(nov)


def fechar(apostas: list, agora: datetime, buscar) -> int:
    """`buscar(slug)` devolve o evento da Gamma (ou None). Devolve quantas fecharam."""
    n, cache = 0, {}
    for a in apostas:
        t = pac._quando(a.get("inicio_utc"))
        if a.get("resultado") or t is None or t > agora:
            continue
        slug = str(a.get("evento") or "")
        if slug not in cache:
            cache[slug] = buscar(slug)
        ev = cache[slug]
        res = pac.resultado(a, ev) if ev else None
        if res is None and agora - t > timedelta(days=DESISTE_DIAS):
            res = "sem_resultado"
        if res:
            a["resultado"], a["lucro"], a["fechado_em"] = res, pac.lucro(a, res), agora.isoformat(timespec="seconds")
            n += 1
    return n


def _buscar_na_gamma(slug: str):
    from tools.coletar_polymarket import _get
    evs = _get("events", slug=slug)
    return evs[0] if isinstance(evs, list) and evs else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    agora = datetime.now(timezone.utc)
    apostas = ler()
    n_novas = registrar(apostas, agora)
    gravar(apostas)
    rc, n_fech = 0, 0
    if "--sem-rede" not in sys.argv:
        try:
            n_fech = fechar(apostas, agora, _buscar_na_gamma)
            gravar(apostas)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"[placar {ESPORTE}] rede/API falhou ao fechar: {type(e).__name__}: {e}")
            rc = 75
    print(f"[placar {ESPORTE}] {n_novas} nova(s), {n_fech} fechada(s). " + pac.texto_do_resumo(pac.resumo(apostas)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
