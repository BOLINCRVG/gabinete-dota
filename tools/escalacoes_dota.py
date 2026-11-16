# -*- coding: utf-8 -*-
"""ESCALACAO DE CADA PARTIDA, versionada (27/10/2026).

A nota por jogador (nota_jogadores_dota.py) precisa saber quem jogou cada mapa.
O raw_matches.jsonl tem isso, mas nao vai para o git e, no robo, so tem as
partidas baixadas no dia -- no PC dele parou em 16/08 e set-out ficaram SEM
escalacao. Aqui fica o arquivo versionado escalacoes_dota.json:
    {match_id: {team_id_cru: [5 account_id]}}
que cresce por tres caminhos:
  --raw ARQ      soma as escalacoes de um raw_matches.jsonl (o do PC, o do dia no robo);
  --backfill N   busca na OpenDota (/matches/{id}) ate N partidas PRO que faltam,
                 da mais recente para a mais antiga, ~1 por segundo (cota gratis),
                 parando no primeiro erro de cota;
O account_id e publico e e o que a regua de privacidade permite (vigia_privacidade).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ARQ = RAIZ / "escalacoes_dota.json"
PAUSA_S = 1.1


def carregar(arq: Path = ARQ) -> dict:
    """{match_id(int): {team_id(int): frozenset(contas)}}."""
    try:
        d = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(m): {int(t): frozenset(int(a) for a in accs) for t, accs in times.items()} for m, times in d.items()}


def gravar(esc: dict, arq: Path = ARQ) -> None:
    out = {str(m): {str(t): sorted(int(a) for a in accs) for t, accs in times.items()}
           for m, times in sorted(esc.items())}
    arq.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


def de_payload(d: dict) -> dict:
    """Pura: {team_id: frozenset(5 contas)} de uma partida (OpenDota ou raw). So times com 5."""
    fora = {}
    jog = d.get("players") or []
    for tid_raw, is_rad in ((d.get("radiant_team_id"), True), (d.get("dire_team_id"), False)):
        try:
            tid = int(tid_raw or 0)
        except (TypeError, ValueError):
            continue
        if tid <= 0:
            continue
        rad = lambda p: p.get("isRadiant") if "isRadiant" in p else (p.get("player_slot") or 0) < 128
        accs = frozenset(int(p["account_id"]) for p in jog if p.get("account_id") and rad(p) is is_rad)
        if len(accs) == 5:
            fora[tid] = accs
    return fora


def somar_raw(esc: dict, raw: Path) -> int:
    novos = 0
    try:
        fh = open(raw, encoding="utf-8")
    except OSError:
        return 0
    with fh:
        for l in fh:
            try:
                d = json.loads(l)
                mid = int(d.get("match_id") or 0)
            except (ValueError, TypeError):
                continue
            e = de_payload(d)
            if mid > 0 and e and mid not in esc:
                esc[mid] = e
                novos += 1
    return novos


def faltando(esc: dict, desde: str = "2025-01-01") -> list:
    """Partidas PRO do data.csv, sem escalacao, da mais recente para a mais antiga."""
    from eval_whr_dota import PRO_TIERS
    d = pd.read_csv(RAIZ / "data.csv", usecols=["match_id", "start_time", "league_id"], low_memory=False)
    d = d.drop_duplicates("match_id")
    d = d[pd.to_datetime(d.start_time, unit="s", errors="coerce") >= pd.Timestamp(desde)]
    # tier pela foto versionada (league_tiers.csv), sem rede: o mesmo universo do produto
    tiers = pd.read_csv(RAIZ / "league_tiers.csv")
    pro = {int(i) for i, t in zip(tiers.league_id, tiers.tier) if str(t) in PRO_TIERS}
    try:                                    # + as ligas promovidas (ligas_dota.py, 16/11/2026)
        import sys as _sys
        _sys.path.insert(0, str(RAIZ))
        from ligas_dota import promovidas
        pro |= promovidas()
    except Exception:
        pass
    d = d[d.league_id.fillna(0).astype(int).isin(pro)]
    return [int(m) for m in d.sort_values("start_time", ascending=False).match_id if int(m) not in esc]


def backfill(esc: dict, n: int) -> tuple:
    from opendota_client import get_match
    feitos, falhas = 0, 0
    for mid in faltando(esc)[:n]:
        try:
            e = de_payload(get_match(mid) or {})
        except Exception as ex:  # cota/rede: para, nao insiste
            print(f"[escalacoes] parou na partida {mid}: {type(ex).__name__}")
            break
        if e:
            esc[mid] = e
            feitos += 1
        else:
            falhas += 1
        time.sleep(PAUSA_S)
    return feitos, falhas


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=[])
    ap.add_argument("--backfill", type=int, default=0)
    a = ap.parse_args()
    esc = carregar()
    antes = len(esc)
    for r in a.raw:
        print(f"[escalacoes] +{somar_raw(esc, Path(r))} do raw {Path(r).name}")
    if a.backfill:
        f, s = backfill(esc, a.backfill)
        print(f"[escalacoes] OpenDota: +{f} (sem escalacao completa: {s}); ainda faltam {len(faltando(esc))}")
    gravar(esc)
    print(f"[escalacoes] {antes} -> {len(esc)} partidas com escalacao")
    return 0


if __name__ == "__main__":
    sys.exit(main())
