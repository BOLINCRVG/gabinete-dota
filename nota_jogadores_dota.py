# -*- coding: utf-8 -*-
"""NOTA POR JOGADOR no Dota (27/10/2026) -- o mesmo que foi o maior ganho no CS.

Medido contra a Pinnacle: o modelo errava nos times de elenco NOVO (LEGION,
HULIGANI, Yellow Submarine) -- a nota do time carregava a historia de um elenco
que nao existe mais. Aqui cada jogador tem uma nota (Elo) que leva de time em
time; a forca do time = media das notas do elenco (o da partida, ou o ultimo
visto daquele team_id). Sem elenco conhecido: o Elo do time.

Prova cega (01/05-16/08/2026, 1.451 partidas, escolha no ajuste 15/02-30/04):
WHR de producao calibrado 0,6345 -> mistura 0,6218 (+0,0127, IC95 [+0,0057; +0,0195]).

Uso:
  python nota_jogadores_dota.py              grava nota_jogadores_dota.json (o app le)
  python nota_jogadores_dota.py --avaliar    refaz a prova cega e escolhe w e c
"""
from __future__ import annotations

import bisect
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import eval_whr_dota as ew  # noqa: E402
import escalacoes_dota as escd  # noqa: E402

SAIDA = RAIZ / "nota_jogadores_dota.json"
K_TIME, KJ = 24, 32            # escolhidos no ajuste (iguais aos do CS)
# Escolhidos no AJUSTE por cima de um modelo JA CALIBRADO (a estrutura de producao:
# o ML do app ja sai calibrado); prova cega 27/10/2026: 0,6345 -> 0,6218 (+0,0127).
W_JOGADOR, TEMPERATURA = 0.5, 1.1
TUNE = (pd.Timestamp("2026-02-15", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC"))
BLIND = (pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-08-16", tz="UTC"))


def _lg(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sg(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def mistura(p_modelo: float, p_jogador: float, w: float = W_JOGADOR, c: float = TEMPERATURA) -> float:
    """O que o app usa: logit = c * ((1-w)*logit(modelo) + w*logit(jogador))."""
    return _sg(c * ((1 - w) * _lg(p_modelo) + w * _lg(p_jogador)))


def elencos_por_time(esc: dict) -> dict:
    """team_id cru -> [(instante?, contas)] -- sem instante no arquivo: usa a ordem do match_id."""
    por = defaultdict(list)
    for mid in sorted(esc):
        for tid, accs in esc[mid].items():
            por[tid].append((mid, accs))
    return por


def rodar(m: pd.DataFrame, esc: dict, k: float = K_TIME, kj: float = KJ):
    """Prequencial. Devolve ({match_id: (p_jogador, y, dt, tier)}, nota_time, nota_jogador, ultimo_elenco)."""
    por_time = elencos_por_time(esc)
    chaves = {t: [mid for mid, _ in v] for t, v in por_time.items()}
    r, pr = defaultdict(float), {}
    ultimo = {}                    # nome do time (identidade) -> (contas, quando)
    fora = {}
    for x in m.itertuples(index=False):
        a, b = int(x.tid), int(x.oid)
        mid = int(x.match_id)
        doj = esc.get(mid, {})

        def elenco(tid_cru):
            e = doj.get(int(tid_cru))
            if e:
                return e
            lst = chaves.get(int(tid_cru))
            if not lst:
                return None
            i = bisect.bisect_left(lst, mid) - 1          # so o que veio ANTES
            return por_time[int(tid_cru)][i][1] if i >= 0 else None
        ea, eb = elenco(getattr(x, "tid_orig", x.tid)), elenco(getattr(x, "oid_orig", x.oid))

        def forca(t, e):
            if not e:
                return r[t]
            for j in e:
                pr.setdefault(j, r[t])
            return sum(pr[j] for j in e) / len(e)
        p = 1 / (1 + 10 ** (-(forca(a, ea) - forca(b, eb)) / 400))
        y = int(x.win)
        fora[mid] = (p, y, x.dt, x.tier)
        r[a] += k * (y - p)
        r[b] -= k * (y - p)
        for e, s in ((ea, 1), (eb, -1)):
            for j in (e or ()):
                pr[j] += s * kj * (y - p)
        for nome, e in ((x.teamname, ea), (x.opp_teamname, eb)):
            if e:
                ultimo[str(nome)] = (sorted(e), x.dt)
    return fora, r, pr, ultimo


def gravar(m: pd.DataFrame, esc: dict) -> dict:
    _, r, pr, ultimo = rodar(m, esc)
    nome_tid = {}
    for x in m.itertuples(index=False):
        nome_tid[str(x.teamname)] = int(x.tid)
        nome_tid[str(x.opp_teamname)] = int(x.oid)
    times = {}
    for nome, tid in nome_tid.items():
        el = ultimo.get(nome)
        forca = (sum(pr.get(j, r[tid]) for j in el[0]) / len(el[0])) if el else r[tid]
        times[nome] = {"forca": round(forca, 2), "elo_time": round(r[tid], 2),
                       "elenco": el[0] if el else [], "elenco_visto_em": el[1].strftime("%Y-%m-%d") if el else None}
    out = {"gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "parametros": {"k_time": K_TIME, "kj": KJ, "w_jogador": W_JOGADOR, "temperatura": TEMPERATURA},
           "como_usar": "p_jogador = 1/(1+10^(-(forca_A-forca_B)/400)); final = mistura(p_modelo, p_jogador)",
           "times": times}
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


def avaliar(m: pd.DataFrame, esc: dict) -> None:

    fitter = ew.WHRFitter(w2_elo_per_day=60.0)       # w2 de PRODUCAO (train_offline)
    W, c = {}, 0
    for x in m.itertuples(index=False):
        a, b = str(x.tid), str(x.oid)
        W[int(x.match_id)] = _sg(fitter.rating_nat(a) - fitter.rating_nat(b))
        fitter.add_game(a, b, int(x.win), int(x.day))
        fitter.refresh_pair(a, b, iters=2)
        c += 1
        if c % 500 == 0:
            fitter.full_sweeps(sweeps=1, iters=1)
    fj, *_ = rodar(m, esc)
    ll = lambda p, y: -math.log(min(max(p if y else 1 - p, 1e-9), 1))
    jan = lambda j: [k for k, v in fj.items() if j[0] <= v[2] < j[1] and v[3] == "professional"]
    aj, cego = jan(TUNE), jan(BLIND)
    # o "modelo de producao" calibrado (temperatura do ajuste, como o Platt do ML)
    cs = [x / 20 for x in range(8, 41)]
    cw = min(cs, key=lambda t: sum(ll(_sg(t * _lg(W[k])), fj[k][1]) for k in aj))
    pm = lambda k: _sg(cw * _lg(W[k]))
    grade = [(w / 10, t) for w in range(0, 11) for t in cs]
    w, t = min(grade, key=lambda g: sum(ll(mistura(pm(k), fj[k][0], *g), fj[k][1]) for k in aj))
    A = [ll(pm(k), fj[k][1]) for k in cego]
    B = [ll(mistura(pm(k), fj[k][0], w, t), fj[k][1]) for k in cego]
    dif = [x - y for x, y in zip(A, B)]
    n = len(dif)
    rnd = random.Random(7)
    bo = sorted(sum(rnd.choice(dif) for _ in dif) / n for _ in range(2000))
    print(f"ajuste: modelo calibrado c={cw}; mistura w={w} c={t}")
    print(f"CEGO ({n} partidas): modelo {sum(A) / n:.4f} -> mistura {sum(B) / n:.4f} ganho {sum(dif) / n:+.4f} "
          f"IC95 [{bo[50]:+.4f}, {bo[1949]:+.4f}] {'PASSA' if sum(dif) / n > 0.002 and bo[50] > 0 else 'nao passa'}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    m = ew.load_matches().sort_values(["ord_key", "match_id"]).reset_index(drop=True)
    if "tid_orig" not in m.columns:
        m["tid_orig"], m["oid_orig"] = m["tid"], m["oid"]
    esc = escd.carregar()
    if "--avaliar" in sys.argv:
        avaliar(m, esc)
        return 0
    out = gravar(m, esc)
    com = sum(1 for v in out["times"].values() if v["elenco"])
    print(f"[nota jogadores] {len(out['times'])} times ({com} com elenco conhecido); w={W_JOGADOR} c={TEMPERATURA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
