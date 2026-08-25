# -*- coding: utf-8 -*-
"""Estudo de FADIGA/carga de agenda (hipotese do usuario, 2026-07-08):
"times jogam 2-3x no mesmo dia — cansaco deveria contar".

Traducao mensuravel, no residual prequencial (y - p) do WHR config*:
  - mapas jogados pelo time nas ultimas 12h/24h ANTES do jogo;
  - DIFERENCIAL de carga (meus mapas - mapas do oponente) — o que importa
    para pricing e a assimetria, nao o cansaco absoluto.
Se residual < 0 com carga maior -> fadiga real alem do rating -> termo gateado.
"""
from __future__ import annotations

import math
import sys

# Console cp1252 no Windows derruba qualquer print com acento/emoji — e o estouro
# vem no fim, depois do trabalho todo feito. Ver tools/find_silent_failures.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402
from eval_whr_dota import SideTracker, load_matches, MIN_GAMES_EVAL, TUNE_FROM  # noqa: E402

W2_STAR = 20.0
H12 = 12 * 3600.0
H24 = 24 * 3600.0


def main() -> int:
    m = load_matches()
    fitter = WHRFitter(w2_elo_per_day=W2_STAR)
    side = SideTracker()
    t_from = pd.Timestamp(TUNE_FROM, tz="UTC")
    hist: dict[str, list[float]] = defaultdict(list)  # team -> timestamps dos mapas
    rows = []
    count = 0

    def maps_within(team: str, ts: float, window: float) -> int:
        return sum(1 for t in hist[team][-12:] if 0 < ts - t <= window)

    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        ts = float(r.ord_key)
        is_prof = r.tier == "professional"
        ga = len(fitter.games.get(a, ()))
        gb = len(fitter.games.get(b, ()))
        if is_prof and ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL and r.dt >= t_from:
            logit = fitter.rating_nat(a) - fitter.rating_nat(b) + side.offset_nat(int(r.day))
            p = 1.0 / (1.0 + math.exp(-logit))
            y = int(r.win)
            for team, opp, p_t, y_t in ((a, b, p, y), (b, a, 1 - p, 1 - y)):
                m12, o12 = maps_within(team, ts, H12), maps_within(opp, ts, H12)
                m24, o24 = maps_within(team, ts, H24), maps_within(opp, ts, H24)
                rows.append({"res": y_t - p_t, "m12": m12, "d12": m12 - o12,
                             "m24": m24, "d24": m24 - o24})
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
    print(f"observacoes (perspectiva de time): {len(df)}")

    def report(col: str, buckets: list) -> None:
        print(f"\n=== residual por {col} ===")
        for lo, hi, label in buckets:
            sub = df[(df[col] >= lo) & (df[col] <= hi)]["res"]
            if len(sub) < 20:
                print(f"  {label:12} n={len(sub):<5} (amostra pequena)")
                continue
            mu = sub.mean()
            se = sub.std(ddof=1) / math.sqrt(len(sub))
            z = mu / se if se > 0 else 0.0
            print(f"  {label:12} n={len(sub):<5} residual={mu*100:+5.1f}pp  z={z:+.1f}")

    report("m12", [(0, 0, "0 mapas/12h"), (1, 1, "1 mapa"), (2, 2, "2 mapas"), (3, 99, "3+ mapas")])
    report("m24", [(0, 0, "0 mapas/24h"), (1, 2, "1-2 mapas"), (3, 4, "3-4 mapas"), (5, 99, "5+ mapas")])
    report("d12", [(-99, -2, "descansado-2"), (-1, -1, "descansado-1"), (0, 0, "igual"),
                   (1, 1, "cansado+1"), (2, 99, "cansado+2")])
    report("d24", [(-99, -3, "descansado-3"), (-2, -1, "descansado"), (0, 0, "igual"),
                   (1, 2, "cansado"), (3, 99, "cansado+3")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
