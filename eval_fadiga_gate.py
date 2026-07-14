# -*- coding: utf-8 -*-
"""Gate do termo de fadiga-diferencial (|d24| >= 3 mapas).

Um replay so, coletando (p, y, d24) prequencial; offset do termo estimado NO
TUNE (fev15..abr30), veredito NO BLIND (mai+): logloss geral e no subset
afetado, com vs sem o termo. Regua: mesma de sempre (numero decide).
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402
from eval_whr_dota import (  # noqa: E402
    SideTracker, load_matches, MIN_GAMES_EVAL, TUNE_FROM, TUNE_TO, BLIND_FROM,
)

W2_STAR = 20.0
H24 = 24 * 3600.0
D_TRIG = 3  # dispara com diferencial >= 3 mapas/24h


def ll(y, p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main() -> int:
    m = load_matches()
    fitter = WHRFitter(w2_elo_per_day=W2_STAR)
    side = SideTracker()
    hist = defaultdict(list)
    t_from = pd.Timestamp(TUNE_FROM, tz="UTC")
    rows = []
    count = 0

    def maps24(team, ts):
        return sum(1 for t in hist[team][-12:] if 0 < ts - t <= H24)

    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        ts = float(r.ord_key)
        is_prof = r.tier == "professional"
        ga = len(fitter.games.get(a, ()))
        gb = len(fitter.games.get(b, ()))
        if is_prof and ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL and r.dt >= t_from:
            logit = fitter.rating_nat(a) - fitter.rating_nat(b) + side.offset_nat(int(r.day))
            d24 = maps24(a, ts) - maps24(b, ts)
            rows.append({"dt": r.dt, "logit": logit, "y": int(r.win), "d24": d24})
        fitter.add_game(a, b, int(r.win), int(r.day))
        fitter.refresh_pair(a, b, iters=2)
        if is_prof:
            side.add(int(r.day), int(r.win))
        hist[a].append(ts)
        hist[b].append(ts)
        count += 1
        if count % 500 == 0:
            fitter.full_sweeps(sweeps=1, iters=1)

    df = pd.DataFrame(rows)
    tune = df[(df["dt"] >= pd.Timestamp(TUNE_FROM, tz="UTC")) & (df["dt"] <= pd.Timestamp(TUNE_TO, tz="UTC"))]
    blind = df[df["dt"] >= pd.Timestamp(BLIND_FROM, tz="UTC")]

    # estima o offset SO no tune: residual medio do lado cansado -> logit
    trig_tune = tune[tune["d24"].abs() >= D_TRIG]
    res = []
    for r in trig_tune.itertuples(index=False):
        p = 1.0 / (1.0 + math.exp(-r.logit))
        sign = 1.0 if r.d24 >= D_TRIG else -1.0  # perspectiva do cansado
        res.append((r.y - p) * sign)
    mean_res = sum(res) / len(res) if res else 0.0
    k_nat = 4.0 * mean_res  # dp/dlogit ~ 0.25 perto de 0.5 -> logit ~ 4*delta_p
    print(f"TUNE: gatilhos={len(trig_tune)} residual_cansado={mean_res*100:+.1f}pp -> k={k_nat:+.3f} nat")

    def eval_blind(use_term: bool) -> tuple[float, float, int]:
        tot = n = 0.0
        tot_sub = n_sub = 0.0
        for r in blind.itertuples(index=False):
            adj = 0.0
            if use_term and abs(r.d24) >= D_TRIG:
                adj = k_nat if r.d24 >= D_TRIG else -k_nat
            p = 1.0 / (1.0 + math.exp(-(r.logit + adj)))
            v = ll(r.y, p)
            tot += v
            n += 1
            if abs(r.d24) >= D_TRIG:
                tot_sub += v
                n_sub += 1
        return tot / n, (tot_sub / n_sub if n_sub else float("nan")), int(n_sub)

    ll_base, ll_base_sub, n_sub = eval_blind(False)
    ll_term, ll_term_sub, _ = eval_blind(True)
    print(f"BLIND geral : sem termo {ll_base:.4f} | com termo {ll_term:.4f} | dif {ll_base-ll_term:+.4f}")
    print(f"BLIND subset (|d24|>={D_TRIG}, n={n_sub}): sem {ll_base_sub:.4f} | com {ll_term_sub:.4f} "
          f"| dif {ll_base_sub-ll_term_sub:+.4f}")
    verd = "ACCEPT" if (ll_base_sub - ll_term_sub) > 0.005 and (ll_base - ll_term) >= 0 else "REJECT/neutro"
    print(f"VEREDITO: {verd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
