# -*- coding: utf-8 -*-
"""Fase 2 (objetivos) + estudo de janelas (viés do campeonato apontado pelo usuário).

PARTE A — gate torres/roshans/barracks (base de objetivos 100% completa):
  B0 liga decaída | V0 média do confronto (shrink) | DC-lite | +ajuste de mapa
  (delta pareado prequencial por mercado). MAE no TUNE (fev15-abr30) e BLIND (mai+).

PARTE B — janelas: V0 de kills calculado com stats do TIME (todos os torneios,
  o atual) vs stats SÓ do campeonato do jogo (recorte estilo LoL, fallback
  global qdo n_eff<3). Se o recorte por camp NÃO ganhar, o default do Dota
  vira janela por time e o recorte fica opcional.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from eval_whr_dota import load_matches, TUNE_FROM, TUNE_TO, BLIND_FROM  # noqa: E402
from eval_totals_dota import Decayed, PairedDelta, shrink  # noqa: E402
from dota_engine import _row_has_objectives  # noqa: E402

MIN_GAMES = 5

MARKETS = {
    "torres": lambda r, is_r: (float(r.towers) + float(r.opp_towers)),
    "roshans": lambda r, is_r: (float(r.roshans) + float(r.opp_roshans)),
    "barracks": lambda r, is_r: (float(r.barracks) + float(r.opp_barracks)),
}


def main() -> int:
    m = load_matches()
    m = m[m["tier"] == "professional"].copy()
    m["pair_key"] = list(zip(m["dt"].dt.date, [tuple(sorted(t)) for t in zip(m["tid"], m["oid"])]))
    m = m.sort_values(["ord_key", "match_id"]).reset_index(drop=True)

    t_tune_from = pd.Timestamp(TUNE_FROM, tz="UTC")
    t_tune_to = pd.Timestamp(TUNE_TO, tz="UTC")
    t_blind = pd.Timestamp(BLIND_FROM, tz="UTC")

    # PARTE A: trackers por mercado de objetivo
    lg = {k: Decayed() for k in MARKETS}
    tm = {k: {} for k in MARKETS}
    m2d = {k: PairedDelta() for k in MARKETS}
    first_val = {k: {} for k in MARKETS}
    series_count: dict = {}
    ngames: dict = {}
    rows_a = []

    # PARTE B: kills global vs por-campeonato
    lg_k = Decayed()
    tk_global: dict = {}
    tk_camp: dict = {}   # (team, league) -> Decayed
    rows_b = []

    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        day = int(r.day)
        mapno = series_count.get(r.pair_key, 0) + 1
        has_obj = _row_has_objectives(r)
        kills = float(r.teamkills) + float(r.oppkills)
        ga, gb = ngames.get(a, 0), ngames.get(b, 0)
        eligible = ga >= MIN_GAMES and gb >= MIN_GAMES and r.dt >= t_tune_from

        # ---------- predições ----------
        if eligible and has_obj:
            row = {"dt": r.dt, "mapno": mapno}
            ok = True
            for mkt, fn in MARKETS.items():
                prior = lg[mkt].mean(day)
                if prior is None:
                    ok = False
                    break
                def _mean(team):
                    dec = tm[mkt].get(team)
                    if dec is None:
                        return prior
                    v = dec.mean(day)
                    return shrink(v, prior, dec.eff_n()) if v is not None else prior
                v0 = 0.5 * (_mean(a) + _mean(b))
                adj = m2d[mkt].value() if mapno >= 2 else 0.0
                row[f"{mkt}_real"] = fn(r, True)
                row[f"{mkt}_b0"] = prior
                row[f"{mkt}_v0"] = v0
                row[f"{mkt}_v0m2"] = v0 + adj
            if ok:
                rows_a.append(row)

        if eligible and math.isfinite(kills):
            lk = lg_k.mean(day)
            if lk is not None:
                def _mg(team):
                    dec = tk_global.get(team)
                    if dec is None:
                        return lk, 0.0
                    v = dec.mean(day)
                    return (shrink(v, lk, dec.eff_n()) if v is not None else lk), dec.eff_n()
                def _mc(team):
                    dec = tk_camp.get((team, str(r.league)))
                    if dec is None:
                        return None, 0.0
                    v = dec.mean(day)
                    return v, dec.eff_n()
                vg = 0.5 * (_mg(a)[0] + _mg(b)[0])
                # recorte por campeonato: usa média do camp se n_eff>=3, senão global
                parts = []
                for team in (a, b):
                    vc, ne = _mc(team)
                    if vc is not None and ne >= 3.0:
                        parts.append(shrink(vc, lk, ne))
                    else:
                        parts.append(_mg(team)[0])
                vcamp = 0.5 * (parts[0] + parts[1])
                rows_b.append({"dt": r.dt, "real": kills, "global": vg, "camp": vcamp})

        # ---------- updates ----------
        if has_obj:
            for mkt, fn in MARKETS.items():
                val = fn(r, True)
                if math.isfinite(val):
                    lg[mkt].add(day, val)
                    tm[mkt].setdefault(a, Decayed()).add(day, val)
                    tm[mkt].setdefault(b, Decayed()).add(day, val)
                    if mapno == 1:
                        first_val[mkt][r.pair_key] = val
                    elif r.pair_key in first_val[mkt]:
                        m2d[mkt].add(val - first_val[mkt][r.pair_key])
        if math.isfinite(kills):
            lg_k.add(day, kills)
            for team in (a, b):
                tk_global.setdefault(team, Decayed()).add(day, kills)
                tk_camp.setdefault((team, str(r.league)), Decayed()).add(day, kills)
        ngames[a] = ga + 1
        ngames[b] = gb + 1
        series_count[r.pair_key] = mapno

    # ---------- relatórios ----------
    da = pd.DataFrame(rows_a)
    print("=== PARTE A: objetivos (MAE; menor=melhor) ===")
    for label, sub in (("TUNE ", da[(da["dt"] >= t_tune_from) & (da["dt"] <= t_tune_to)]),
                       ("BLIND", da[da["dt"] >= t_blind])):
        print(f"-- {label} n={len(sub)} --")
        for mkt in MARKETS:
            line = f"  {mkt:9}"
            for var in ("b0", "v0", "v0m2"):
                mae = (sub[f"{mkt}_{var}"] - sub[f"{mkt}_real"]).abs().mean()
                line += f" {var}={mae:6.3f}"
            print(line)
    for mkt in MARKETS:
        print(f"  delta m2+ {mkt}: {m2d[mkt].value():+.2f} (n={m2d[mkt].n})")

    db = pd.DataFrame(rows_b)
    print("\n=== PARTE B: janelas kills — TIME(global) vs CAMPEONATO ===")
    for label, sub in (("TUNE ", db[(db["dt"] >= t_tune_from) & (db["dt"] <= t_tune_to)]),
                       ("BLIND", db[db["dt"] >= t_blind])):
        mg = (sub["global"] - sub["real"]).abs().mean()
        mc = (sub["camp"] - sub["real"]).abs().mean()
        print(f"-- {label} n={len(sub)}: global={mg:.3f} camp={mc:.3f} dif={mg-mc:+.3f} "
              f"({'CAMP melhor' if mc < mg else 'GLOBAL melhor'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
