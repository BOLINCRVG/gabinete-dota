# -*- coding: utf-8 -*-
"""MODELO x PINNACLE no Dota: vencedor de cada mapa, no fechamento (27/10/2026).

Pedido dele ("levar os modelos a um proximo nivel"): a pergunta deixa de ser
"o modelo acerta mais?" e passa a ser "ONDE o modelo ganha da casa?". O LoL ja
tinha esta regua (tools/modelo_contra_a_casa.py); o Dota nao tinha nada.

PONTO-NO-TEMPO
  - Pinnacle: a ULTIMA leitura antes do inicio da serie, margem tirada em
    proporcao (o mercado "Vencedor - mapa N" de odds_pinnacle_historico.jsonl);
  - modelo: a nota WHR do motor do app, calculada jogo a jogo -- cada mapa e
    previsto so com o que veio ANTES dele (prequencial), SEM o lado (a casa nao
    sabe o lado antes do jogo), com a identidade de times de load_matches.
Casamento: a dupla de times (nomes da casa -> times do CSV pelo resolvedor da
agenda) e o mapa N daquela dupla no dia do jogo.

Saida: evidence/regua_pinnacle_vencedor_dota.json -- geral, por liga e apostas
simuladas (1 u onde o modelo ve vantagem de 1% a 20% contra a odd REAL).
Uso: python tools/modelo_contra_a_casa_dota.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import eval_whr_dota as ew  # noqa: E402
from dota_engine import W2_STAR  # noqa: E402
from app_core.schedule_import_core import resolve_schedule_team  # noqa: E402

HIST = RAIZ / "odds_pinnacle_historico.jsonl"
PARTIDAS = RAIZ / "odds_pinnacle_partidas.jsonl"
SAIDA = RAIZ / "evidence" / "regua_pinnacle_vencedor_dota.json"


def fechamento(hist_path: Path, partidas: dict) -> dict:
    """{(jogo, mapa): {"home": odd, "away": odd}} na ultima leitura antes do inicio."""
    ult = {}
    with open(hist_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("tipo") != "moneyline" or r.get("alternativa") or not r.get("periodo"):
                continue
            p = partidas.get(r["jogo"])
            if not p or not p.get("inicio_utc"):
                continue
            lido = pd.Timestamp(r["lido_em"]).tz_convert("UTC")
            if lido >= pd.Timestamp(p["inicio_utc"]):
                continue
            k = (r["jogo"], int(r["periodo"]), r["lado"])
            if k not in ult or lido > ult[k][0]:
                ult[k] = (lido, float(r["odd"]))
    fora = defaultdict(dict)
    for (jogo, mapa, lado), (_, odd) in ult.items():
        fora[(jogo, mapa)][lado] = odd
    return {k: v for k, v in fora.items() if "home" in v and "away" in v}


def previsoes_por_mapa(m: pd.DataFrame) -> dict:
    """{match_id: p(time tid vence)} PREQUENCIAL, sem lado (o motor do app, jogo a jogo)."""
    fitter = ew.WHRFitter(w2_elo_per_day=W2_STAR)
    fora, cont = {}, 0
    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        fora[int(r.match_id)] = 1.0 / (1.0 + math.exp(-(fitter.rating_nat(a) - fitter.rating_nat(b))))
        fitter.add_game(a, b, int(r.win), int(r.day))
        fitter.refresh_pair(a, b, iters=2)
        cont += 1
        if cont % 500 == 0:
            fitter.full_sweeps(sweeps=1, iters=1)
    return fora


def apostas_simuladas(casos, minimo=1.0, maximo=20.0) -> dict:
    def _um(cs):
        ap, lucro = 0, 0.0
        for c in cs:
            for p, odd, ganhou in ((c["p_modelo"], c["odd_casa"], c["y"] == 1),
                                   (1 - c["p_modelo"], c["odd_fora"], c["y"] == 0)):
                v = (p * odd - 1) * 100
                if minimo <= v <= maximo:
                    ap += 1
                    lucro += (odd - 1) if ganhou else -1.0
        return {"apostas": ap, "lucro": round(lucro, 2), "roi": round(lucro / ap, 4) if ap else None}
    out = {"total": _um(casos), "por_liga": {}}
    for liga in sorted({c["liga"] for c in casos}):
        r = _um([c for c in casos if c["liga"] == liga])
        if r["apostas"]:
            out["por_liga"][liga] = r
    return out


def resumo(casos) -> dict:
    if not casos:
        return {"mapas": 0}
    ll = lambda p, y: -math.log(min(max(p if y else 1 - p, 1e-9), 1))
    lm = [ll(c["p_modelo"], c["y"]) for c in casos]
    lc = [ll(c["p_casa"], c["y"]) for c in casos]
    dif = [b - a for a, b in zip(lm, lc)]                 # > 0 = modelo melhor
    rnd = random.Random(7)
    n = len(dif)
    boots = sorted(sum(rnd.choice(dif) for _ in dif) / n for _ in range(2000))
    return {"mapas": n, "logloss_modelo": round(sum(lm) / n, 4), "logloss_casa": round(sum(lc) / n, 4),
            "vantagem_modelo": round(sum(dif) / n, 4), "ic95": [round(boots[50], 4), round(boots[1949], 4)],
            "conclusivo": boots[50] > 0 or boots[1949] < 0}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    partidas = {}
    for l in PARTIDAS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            p = json.loads(l)
            partidas[p["id"]] = p
    precos = fechamento(HIST, partidas)
    m = ew.load_matches().sort_values(["ord_key", "match_id"]).reset_index(drop=True)
    prev = previsoes_por_mapa(m)
    times = sorted(set(m.teamname) | set(m.opp_teamname))
    try:
        apelidos = json.loads((RAIZ / "user_team_aliases.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        apelidos = {}
    casos, sem_time, sem_jogo = [], 0, 0
    for (jogo, mapa), odds in sorted(precos.items()):
        p = partidas[jogo]
        a, ok_a = resolve_schedule_team(p["casa"], times, apelidos)
        b, ok_b = resolve_schedule_team(p["fora"], times, apelidos)
        if not (ok_a and ok_b):
            sem_time += 1
            continue
        ini = pd.Timestamp(p["inicio_utc"])
        j = m[(m.dt >= ini - pd.Timedelta(hours=3)) & (m.dt <= ini + pd.Timedelta(hours=12))
              & (((m.teamname == a) & (m.opp_teamname == b)) | ((m.teamname == b) & (m.opp_teamname == a)))]
        j = j.drop_duplicates("match_id").sort_values("ord_key")
        if len(j) < mapa:
            sem_jogo += 1
            continue
        r = j.iloc[mapa - 1]
        p_tid = prev.get(int(r.match_id))
        if p_tid is None:
            sem_jogo += 1
            continue
        a_e_tid = r.teamname == a
        p_a = p_tid if a_e_tid else 1 - p_tid
        y = int(r.win) if a_e_tid else 1 - int(r.win)
        s = 1 / odds["home"] + 1 / odds["away"]
        casos.append({"liga": p.get("liga") or "?", "jogo": jogo, "mapa": mapa, "casa": a, "fora": b,
                      "p_modelo": p_a, "p_casa": (1 / odds["home"]) / s, "y": y,
                      "odd_casa": odds["home"], "odd_fora": odds["away"]})
    por_liga = {}
    for liga in sorted({c["liga"] for c in casos}):
        cs = [c for c in casos if c["liga"] == liga]
        if len(cs) >= 5:
            por_liga[liga] = resumo(cs)
    out = {"gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"), "mercado": "Vencedor de cada mapa",
           "geral": resumo(casos), "por_mapa": {str(k): resumo([c for c in casos if c["mapa"] == k]) for k in (1, 2, 3)},
           "por_liga": por_liga, "apostas_simuladas": apostas_simuladas(casos),
           "fora_da_medicao": {"sem_time": sem_time, "sem_jogo": sem_jogo, "precos": len(precos)}}
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    g = out["geral"]
    print(f"[dota x casa] {g.get('mapas', 0)} mapas | modelo {g.get('logloss_modelo')} x casa {g.get('logloss_casa')} "
          f"| vantagem do modelo {g.get('vantagem_modelo')} IC95 {g.get('ic95')} | fora: {out['fora_da_medicao']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
