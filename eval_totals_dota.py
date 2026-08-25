# -*- coding: utf-8 -*-
"""Fase 2 — gate de totais do Dota (kills e duracao) com dados do indice.

Replay prequencial (stats SO de partidas tier professional — os niveis diferem
muito do excluded: 52 vs 61 kills/jogo). Tune = 2026-02-15..04-30; BLIND = 05-01+.
Elegivel: ambos os times com >=5 jogos professional no historico.

Variantes de media prevista (kills totais do mapa):
  B0  media da liga (decaida, meia-vida 60d)
  V0  media do confronto: 0.5*(media_total_A + media_total_B), shrink p/ liga
  DC  ataque x defesa multiplicativo (Dixon-Coles-lite), blend 50/50 com V0
  +M2 variante vencedora + ajuste de mapa>=2 (delta pareado estimado
      prequencialmente — nada fitado no blind)

Gate: MAE no BLIND < baseline V0 (mesma regua de sempre: o numero decide).
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
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from eval_whr_dota import load_matches, TUNE_FROM, TUNE_TO, BLIND_FROM  # noqa: E402

HL_DAYS = 60.0
MIN_GAMES = 5
SHRINK_KNEE = 8.0


class Decayed:
    """Media decaida (meia-vida em dias) de um valor por time/liga."""

    def __init__(self, halflife_days: float = HL_DAYS):
        self.hl = float(halflife_days)
        self.w = 0.0
        self.wx = 0.0
        self.last_day: int | None = None

    def _decay(self, day: int) -> None:
        if self.last_day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.hl)
            self.w *= f
            self.wx *= f
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    def add(self, day: int, x: float) -> None:
        self._decay(day)
        self.w += 1.0
        self.wx += float(x)

    def mean(self, day: int) -> float | None:
        self._decay(day)
        return (self.wx / self.w) if self.w > 0 else None

    def eff_n(self) -> float:
        return self.w


def shrink(value: float, prior: float, eff_n: float, knee: float = SHRINK_KNEE) -> float:
    lam = eff_n / (eff_n + knee)
    return lam * value + (1 - lam) * prior


class PairedDelta:
    """Delta pareado m_k - m1 (k>=2) estimado prequencialmente."""

    def __init__(self):
        self.n = 0
        self.s = 0.0

    def add(self, diff: float) -> None:
        self.n += 1
        self.s += float(diff)

    def value(self) -> float:
        return (self.s / self.n) if self.n >= 30 else 0.0


def main() -> int:
    m = load_matches()
    m["total_kills"] = pd.to_numeric(m["teamkills"], errors="coerce") + pd.to_numeric(m["oppkills"], errors="coerce")
    m["dur_min"] = pd.to_numeric(m["duration"], errors="coerce") / 60.0
    m["pair_key"] = list(zip(m["dt"].dt.date, [tuple(sorted(t)) for t in zip(m["tid"], m["oid"])]))

    t_tune_from = pd.Timestamp(TUNE_FROM, tz="UTC")
    t_tune_to = pd.Timestamp(TUNE_TO, tz="UTC")
    t_blind = pd.Timestamp(BLIND_FROM, tz="UTC")

    league_k = Decayed()
    league_d = Decayed()
    tk: dict[str, Decayed] = {}   # kills totais do jogo, por time
    tk_for: dict[str, Decayed] = {}
    tk_against: dict[str, Decayed] = {}
    td: dict[str, Decayed] = {}
    ngames: dict[str, int] = {}
    series_count: dict = {}
    m2_delta_k = PairedDelta()
    m2_delta_d = PairedDelta()
    series_first_k: dict = {}
    series_first_d: dict = {}

    rows = []
    for r in m.itertuples(index=False):
        if r.tier != "professional":
            continue
        a, b = str(r.tid), str(r.oid)
        day = int(r.day)
        kills = float(r.total_kills)
        dur = float(r.dur_min)
        mapno = series_count.get(r.pair_key, 0) + 1

        ga, gb = ngames.get(a, 0), ngames.get(b, 0)
        if ga >= MIN_GAMES and gb >= MIN_GAMES and r.dt >= t_tune_from and math.isfinite(kills) and math.isfinite(dur):
            lk = league_k.mean(day)
            ld = league_d.mean(day)
            if lk is not None and ld is not None:
                # V0: media do confronto com shrink
                va = shrink(tk[a].mean(day), lk, tk[a].eff_n())
                vb = shrink(tk[b].mean(day), lk, tk[b].eff_n())
                v0 = 0.5 * (va + vb)
                # DC-lite: E[kills_A] = for_A * against_B / (lk/2)
                half = lk / 2.0
                fa = shrink(tk_for[a].mean(day), half, tk_for[a].eff_n())
                fb = shrink(tk_for[b].mean(day), half, tk_for[b].eff_n())
                aa = shrink(tk_against[a].mean(day), half, tk_against[a].eff_n())
                ab = shrink(tk_against[b].mean(day), half, tk_against[b].eff_n())
                dc_raw = (fa * ab / half) + (fb * aa / half)
                dc = 0.5 * v0 + 0.5 * dc_raw
                # duracao
                da = shrink(td[a].mean(day), ld, td[a].eff_n())
                db = shrink(td[b].mean(day), ld, td[b].eff_n())
                d0 = 0.5 * (da + db)
                adj_k = m2_delta_k.value() if mapno >= 2 else 0.0
                adj_d = m2_delta_d.value() if mapno >= 2 else 0.0
                rows.append({
                    "dt": r.dt, "mapno": mapno, "kills": kills, "dur": dur,
                    "k_b0": lk, "k_v0": v0, "k_dc": dc,
                    "k_v0_m2": v0 + adj_k, "k_dc_m2": dc + adj_k,
                    "d_b0": ld, "d_v0": d0, "d_v0_m2": d0 + adj_d,
                })

        # updates prequenciais (professional only)
        if math.isfinite(kills) and math.isfinite(dur):
            league_k.add(day, kills)
            league_d.add(day, dur)
            for team, kf, kag in ((a, float(r.teamkills), float(r.oppkills)),
                                  (b, float(r.oppkills), float(r.teamkills))):
                tk.setdefault(team, Decayed()).add(day, kills)
                tk_for.setdefault(team, Decayed()).add(day, kf)
                tk_against.setdefault(team, Decayed()).add(day, kag)
                td.setdefault(team, Decayed()).add(day, dur)
                ngames[team] = ngames.get(team, 0) + 1
            if mapno == 1:
                series_first_k[r.pair_key] = kills
                series_first_d[r.pair_key] = dur
            elif r.pair_key in series_first_k:
                m2_delta_k.add(kills - series_first_k[r.pair_key])
                m2_delta_d.add(dur - series_first_d[r.pair_key])
        series_count[r.pair_key] = mapno

    df = pd.DataFrame(rows)
    for label, sub in (("TUNE ", df[(df["dt"] >= t_tune_from) & (df["dt"] <= t_tune_to)]),
                       ("BLIND", df[df["dt"] >= t_blind])):
        print(f"\n=== {label} n={len(sub)} (mapas>=2: {(sub['mapno']>=2).sum()}) ===")
        print("KILLS  (MAE):")
        for col in ("k_b0", "k_v0", "k_dc", "k_v0_m2", "k_dc_m2"):
            mae = (sub[col] - sub["kills"]).abs().mean()
            mae_m2 = (sub.loc[sub["mapno"] >= 2, col] - sub.loc[sub["mapno"] >= 2, "kills"]).abs().mean()
            print(f"  {col:8} mae={mae:6.3f}   mae_mapa2+={mae_m2:6.3f}")
        print("DURACAO (MAE, min):")
        for col in ("d_b0", "d_v0", "d_v0_m2"):
            mae = (sub[col] - sub["dur"]).abs().mean()
            print(f"  {col:8} mae={mae:6.3f}")
    print(f"\ndelta m2 estimado ao final: kills={m2_delta_k.value():+.2f} (n={m2_delta_k.n}) "
          f"dur={m2_delta_d.value():+.2f}min (n={m2_delta_d.n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
