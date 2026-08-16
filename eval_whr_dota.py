# -*- coding: utf-8 -*-
"""Fase 1c — primeiro WHR do Dota (reusa mlcore/whr_core do LoL, que e agnostico).

Protocolo (adaptado a 1 semestre de dados — nao ha ano-cego):
- Replay prequencial: TODA predicao e feita ANTES de adicionar o jogo.
- TUNE  = partidas tier professional 2026-02-15..2026-04-30 (escolhe w2 e config).
- BLIND = partidas tier professional 2026-05-01..fim (so o veredito final).
- Eval exige ambos os times com >=5 jogos no fitter.
- Identidade de time = team_id do OpenDota (estavel a renames).

Perguntas que este eval responde com numero:
1. w2* do Dota (hipotese do usuario: momento importa mais -> w2* > 60 do LoL);
2. ligas "excluded" (Destiny/UDPL/...) ajudam como DADO DE TREINO? (56% das
   partidas professional tem time que tambem joga nelas — mas ha risco 322);
3. termo de side Radiant rolante (meia-vida 30d, prequencial) melhora?

Uso: python eval_whr_dota.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402

TIERS_CACHE = Path(__file__).with_name("league_tiers.csv")
TUNE_FROM, TUNE_TO = "2026-02-15", "2026-04-30"
BLIND_FROM = "2026-05-01"
MIN_GAMES_EVAL = 5
SIDE_HALFLIFE_DAYS = 30.0
SIDE_MIN_EFF = 50.0
SIDE_MAX_OFFSET = 0.4  # natural (logit); 0.4 ~ 60/40


# A OpenDota classifica liga em premium > professional > amateur. O TI e marcado
# como PREMIUM: ate 16/08/2026 o universo do produto era so "professional", e o
# torneio inteiro ficava FORA do modelo (TI 2025 144 partidas + TI 2026 109, este
# em andamento). Era o maior evento do ano invisivel pro modelo. Universo = os
# dois tiers de cima; amateur/excluded seguem fora.
PRO_TIERS = ("professional", "premium")

# O cache de tiers e uma foto: sem prazo de validade, liga NOVA fica "sem_tier" e
# some do universo calada (visto em 16/08: BLAST Slam VI China Qualifier e ESL
# challenger China caidos assim). Revalida por semana, com queda-pro-cache se a
# API falhar — dado velho e melhor que erro.
TIERS_CACHE_MAX_AGE_DAYS = 7.0


def is_pro_tier(s: pd.Series) -> pd.Series:
    """Mascara: a linha esta no universo profissional do produto?"""
    return s.astype(str).str.strip().str.lower().isin(PRO_TIERS)


def _fetch_tiers_df() -> pd.DataFrame:
    from opendota_client import get_leagues
    return pd.DataFrame(get_leagues()).rename(columns={"leagueid": "league_id"})[["league_id", "tier"]]


def load_tiers() -> dict[int, str]:
    df = None
    stale = True
    if TIERS_CACHE.exists():
        try:
            age_days = (time.time() - TIERS_CACHE.stat().st_mtime) / 86400.0
            stale = age_days > TIERS_CACHE_MAX_AGE_DAYS
        except OSError:
            stale = True
        try:
            df = pd.read_csv(TIERS_CACHE)
        except (OSError, ValueError):
            df = None
    if df is None or stale:
        try:
            fresh = _fetch_tiers_df()
            if not fresh.empty:
                fresh.to_csv(TIERS_CACHE, index=False)
                df = fresh
        except Exception:
            if df is None:
                raise
    out = {}
    for r in df.itertuples(index=False):
        try:
            out[int(r.league_id)] = str(r.tier)
        except Exception:
            continue
    return out


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).with_name("data.csv"), low_memory=False)
    m = df[df["side"] == "radiant"].drop_duplicates("match_id").copy()
    m["dt"] = pd.to_datetime(m["date"], errors="coerce", utc=True)
    m["st"] = pd.to_numeric(m["start_time"], errors="coerce")
    # ordena por start_time quando existe (ordem real do dia); senao pela data
    m["ord_key"] = m["st"].fillna(m["dt"].astype("int64") // 10**9)
    m["tid"] = pd.to_numeric(m["team_id"], errors="coerce").fillna(0).astype(int)
    m["oid"] = pd.to_numeric(m["opp_team_id"], errors="coerce").fillna(0).astype(int)
    m["win"] = pd.to_numeric(m["result"], errors="coerce").fillna(0).astype(int)
    m["league_id"] = pd.to_numeric(m["league_id"], errors="coerce").fillna(0).astype(int)
    m = m[(m["tid"] > 0) & (m["oid"] > 0) & m["dt"].notna()]
    tiers = load_tiers()
    m["tier"] = m["league_id"].map(tiers).fillna("sem_tier")
    m["day"] = m["dt"].dt.date.map(lambda d: d.toordinal())
    return m.sort_values(["ord_key", "match_id"]).reset_index(drop=True)


class SideTracker:
    """Taxa Radiant decaida (meia-vida em dias), atualizada prequencialmente."""

    def __init__(self, halflife_days: float = SIDE_HALFLIFE_DAYS):
        self.hl = float(halflife_days)
        self.sw = 0.0
        self.swy = 0.0
        self.last_day: int | None = None

    def _decay_to(self, day: int) -> None:
        if self.last_day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.hl)
            self.sw *= f
            self.swy *= f
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    def offset_nat(self, day: int) -> float:
        self._decay_to(day)
        if self.sw < SIDE_MIN_EFF:
            return 0.0
        p = min(max(self.swy / self.sw, 1e-6), 1 - 1e-6)
        off = math.log(p / (1 - p))
        return min(max(off, -SIDE_MAX_OFFSET), SIDE_MAX_OFFSET)

    def add(self, day: int, radiant_won: int) -> None:
        self._decay_to(day)
        self.sw += 1.0
        self.swy += float(radiant_won)


def run(m: pd.DataFrame, *, w2: float, use_excluded: bool, use_side: bool,
        score_from: str, score_to: str | None, full_every: int = 500) -> dict:
    fitter = WHRFitter(w2_elo_per_day=w2)
    side = SideTracker()
    t_from = pd.Timestamp(score_from, tz="UTC")
    t_to = pd.Timestamp(score_to, tz="UTC") if score_to else None
    ll = br = 0.0
    n = hits = 0
    count = 0
    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        is_prof = r.tier == "professional"
        in_window = (r.dt >= t_from) and (t_to is None or r.dt <= t_to)
        if is_prof and in_window:
            ga = len(fitter.games.get(a, ()))
            gb = len(fitter.games.get(b, ()))
            if ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL:
                logit = (fitter.rating_nat(a) - fitter.rating_nat(b))
                if use_side:
                    logit += side.offset_nat(int(r.day))
                p = 1.0 / (1.0 + math.exp(-logit))
                pc = min(max(p, 1e-6), 1 - 1e-6)
                y = int(r.win)
                ll += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
                br += (p - y) ** 2
                hits += 1 if ((p >= 0.5) == (y == 1)) else 0
                n += 1
        if r.tier == "excluded" and not use_excluded:
            continue
        fitter.add_game(a, b, int(r.win), int(r.day))
        fitter.refresh_pair(a, b, iters=2)
        if is_prof:
            side.add(int(r.day), int(r.win))
        count += 1
        if count % full_every == 0:
            fitter.full_sweeps(sweeps=1, iters=1)
    return {"logloss": ll / n if n else float("nan"),
            "brier": br / n if n else float("nan"),
            "acc": hits / n if n else float("nan"), "n": n}


def main() -> int:
    m = load_matches()
    n_prof = int((m["tier"] == "professional").sum())
    print(f"[dota-whr] partidas={len(m)} (professional={n_prof}) "
          f"janela {m['dt'].min():%Y-%m-%d}..{m['dt'].max():%Y-%m-%d}", flush=True)

    print("\n[1] varredura de w2 no TUNE (excluded ON, side OFF):", flush=True)
    best_w2, best_ll = None, float("inf")
    for w2 in (20.0, 60.0, 120.0, 250.0, 500.0):
        t0 = time.monotonic()
        res = run(m, w2=w2, use_excluded=True, use_side=False,
                  score_from=TUNE_FROM, score_to=TUNE_TO)
        print(f"  w2={w2:>5.0f} -> ll={res['logloss']:.4f} acc={res['acc']*100:.1f}% "
              f"n={res['n']} ({time.monotonic()-t0:.0f}s)", flush=True)
        if res["logloss"] < best_ll:
            best_ll, best_w2 = res["logloss"], w2
    print(f"  w2* = {best_w2:.0f} (LoL usa 60 — hipotese do usuario: Dota mais alto)", flush=True)

    print(f"\n[2] grade 2x2 no TUNE com w2*={best_w2:.0f}:", flush=True)
    grid = {}
    for use_excluded in (True, False):
        for use_side in (True, False):
            key = f"excl={'on ' if use_excluded else 'off'} side={'on ' if use_side else 'off'}"
            res = run(m, w2=float(best_w2), use_excluded=use_excluded, use_side=use_side,
                      score_from=TUNE_FROM, score_to=TUNE_TO)
            grid[(use_excluded, use_side)] = res
            print(f"  {key} -> ll={res['logloss']:.4f} acc={res['acc']*100:.1f}% n={res['n']}", flush=True)
    best_cfg = min(grid, key=lambda k: grid[k]["logloss"])
    print(f"  config* = excluded={'ON' if best_cfg[0] else 'OFF'} side={'ON' if best_cfg[1] else 'OFF'}", flush=True)

    print(f"\n[3] BLIND (>= {BLIND_FROM}), config* vs baseline (excl ON, side OFF, w2=60):", flush=True)
    blind_best = run(m, w2=float(best_w2), use_excluded=best_cfg[0], use_side=best_cfg[1],
                     score_from=BLIND_FROM, score_to=None)
    blind_base = run(m, w2=60.0, use_excluded=True, use_side=False,
                     score_from=BLIND_FROM, score_to=None)
    for label, res in (("config*", blind_best), ("baseline", blind_base)):
        print(f"  {label:9} ll={res['logloss']:.4f} brier={res['brier']:.4f} "
              f"acc={res['acc']*100:.1f}% n={res['n']}", flush=True)
    dif = blind_base["logloss"] - blind_best["logloss"]
    print(f"  dif ll (base - config*) = {dif:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
