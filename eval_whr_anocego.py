# -*- coding: utf-8 -*-
"""Protocolo ANO-CEGO do Dota (base 44k pos-backfill 2025) — igual ao LoL:
TUNE em 2025 (abr-dez, apos burn-in jan-mar), BLIND = 2026 inteiro.

Re-tuna na base 3x maior: w2, side on/off, excluded on/off, fadiga (k re-
estimado SO em 2025). Saida: config* + NOTA ano-cega (regua 0.6931->5 / 0.55->9.5).
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
from eval_whr_dota import SideTracker, load_matches, MIN_GAMES_EVAL  # noqa: E402

TUNE_FROM, TUNE_TO = "2025-04-01", "2025-12-31"
BLIND_FROM = "2026-01-01"
H24 = 86400.0
RANDOM_LL = 0.6931471805599453


def grade(ll: float) -> float:
    return max(0.0, min(10.0, 5.0 + (RANDOM_LL - ll) / (RANDOM_LL - 0.55) * 4.5))


def replay(m, *, w2: float, use_excluded: bool, use_side: bool,
           fatigue_k: float = 0.0, collect_fatigue: bool = False):
    """Um replay prequencial; retorna metricas por janela (+dados de fadiga)."""
    fitter = WHRFitter(w2_elo_per_day=w2)
    side = SideTracker()
    hist = defaultdict(list)
    t_tf = pd.Timestamp(TUNE_FROM, tz="UTC")
    t_tt = pd.Timestamp(TUNE_TO, tz="UTC")
    t_bf = pd.Timestamp(BLIND_FROM, tz="UTC")
    acc = {"tune": [0.0, 0, 0], "blind": [0.0, 0, 0]}
    fat_rows = []
    count = 0
    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        ts = float(r.ord_key)
        is_prof = r.tier == "professional"
        ga = len(fitter.games.get(a, ()))
        gb = len(fitter.games.get(b, ()))
        seg = None
        if is_prof and ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL:
            if t_tf <= r.dt <= t_tt:
                seg = "tune"
            elif r.dt >= t_bf:
                seg = "blind"
        if seg:
            logit = fitter.rating_nat(a) - fitter.rating_nat(b) + side.offset_nat(int(r.day))
            d24 = (sum(1 for t in hist[a][-12:] if 0 < ts - t <= H24)
                   - sum(1 for t in hist[b][-12:] if 0 < ts - t <= H24))
            if fatigue_k and abs(d24) >= 3:
                logit += -fatigue_k if d24 >= 3 else fatigue_k
            p = 1.0 / (1.0 + math.exp(-logit))
            pc = min(max(p, 1e-6), 1 - 1e-6)
            y = int(r.win)
            acc[seg][0] += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
            acc[seg][1] += 1
            acc[seg][2] += 1 if ((p >= 0.5) == (y == 1)) else 0
            if collect_fatigue and seg == "tune" and abs(d24) >= 3:
                sign = 1.0 if d24 >= 3 else -1.0
                fat_rows.append((y - p) * sign)
        if not (r.tier == "excluded" and not use_excluded):
            fitter.add_game(a, b, int(r.win), int(r.day))
            fitter.refresh_pair(a, b, iters=2)
            count += 1
            if count % 800 == 0:
                fitter.full_sweeps(sweeps=1, iters=1)
        if is_prof:
            side.add(int(r.day), int(r.win))
        hist[a].append(ts)
        hist[b].append(ts)
    out = {}
    for seg, (ll, n, hits) in acc.items():
        out[seg] = {"ll": ll / n if n else float("nan"), "n": n,
                    "acc": hits / n if n else float("nan")}
    if collect_fatigue:
        out["fat_mean"] = (sum(fat_rows) / len(fat_rows)) if fat_rows else 0.0
        out["fat_n"] = len(fat_rows)
    return out


def main() -> int:
    m = load_matches()
    print(f"[ano-cego] partidas={len(m)} prof={int((m['tier']=='professional').sum())} "
          f"janela {m['dt'].min():%Y-%m-%d}..{m['dt'].max():%Y-%m-%d}", flush=True)

    print("\n[1] sweep w2 (TUNE 2025, excl ON, side ON):", flush=True)
    best_w2, best_ll = None, float("inf")
    for w2 in (10.0, 20.0, 40.0, 60.0):
        res = replay(m, w2=w2, use_excluded=True, use_side=True)
        print(f"  w2={w2:>4.0f} tune ll={res['tune']['ll']:.4f} acc={res['tune']['acc']*100:.1f}% "
              f"n={res['tune']['n']}", flush=True)
        if res["tune"]["ll"] < best_ll:
            best_ll, best_w2 = res["tune"]["ll"], w2

    print(f"  w2* = {best_w2:.0f}", flush=True)

    print("\n[2] excluded off (confere) + fadiga k (estimado no TUNE):", flush=True)
    res_noexcl = replay(m, w2=best_w2, use_excluded=False, use_side=True)
    print(f"  excl OFF tune ll={res_noexcl['tune']['ll']:.4f}", flush=True)
    use_excl = res_noexcl["tune"]["ll"] >= best_ll
    res_fat = replay(m, w2=best_w2, use_excluded=use_excl, use_side=True, collect_fatigue=True)
    k_nat = 4.0 * res_fat["fat_mean"]
    print(f"  fadiga tune: residual={res_fat['fat_mean']*100:+.1f}pp n={res_fat['fat_n']} "
          f"-> k={-k_nat:+.3f} aplicado contra o cansado", flush=True)

    print("\n[3] BLIND 2026 (config*):", flush=True)
    final = replay(m, w2=best_w2, use_excluded=use_excl, use_side=True, fatigue_k=-k_nat)
    b = final["blind"]
    print(f"  excluded={'ON' if use_excl else 'OFF'} side=ON fadiga k={-k_nat:+.3f}", flush=True)
    print(f"  BLIND: n={b['n']} logloss={b['ll']:.4f} acc={b['acc']*100:.1f}%", flush=True)
    print(f"  NOTA DOTA (ano-cego) = {grade(b['ll']):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
