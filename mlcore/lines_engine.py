from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .stats_teams import Filters, apply_filters, recency_weights
from .utils import prob_to_decimal_odds, wmean as _wmean

Metric = Literal[
    "kills",
    "towers",
    "dragons",
    "dragons_elemental",
    "barons",
    "inhibitors",
    "time",
]

Dist = Literal["normal", "poisson"]
MapMode = Literal["avg", "m1", "m2", "m3", "m4", "m5"]


@dataclass(frozen=True)
class TeamProfile:
    team: str
    n_games: int
    # means
    kills_for: float
    kills_against: float
    towers_for: float
    towers_against: float
    dragons_for: float
    dragons_against: float
    elementaldrakes_for: float
    elementaldrakes_against: float
    barons_for: float
    barons_against: float
    inhib_for: float
    inhib_against: float
    time_min: float
    # stds
    kills_for_sd: float
    kills_against_sd: float
    towers_for_sd: float
    towers_against_sd: float
    dragons_for_sd: float
    dragons_against_sd: float
    elementaldrakes_for_sd: float
    elementaldrakes_against_sd: float
    barons_for_sd: float
    barons_against_sd: float
    inhib_for_sd: float
    inhib_against_sd: float
    time_min_sd: float


def _wstd(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    mu = _wmean(x, w)
    if not math.isfinite(mu):
        return float("nan")
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    var = float(np.sum(w * (x - mu) ** 2) / s)
    return float(math.sqrt(max(var, 0.0)))


def parse_time_line(s: str) -> float:
    """Parse time line to minutes.

    Accepts:
      - "35:30" -> 35.5
      - "35,5" or "35.5" -> 35.5
      - "35" -> 35.0
    """
    s = str(s).strip()
    if not s:
        raise ValueError("empty time string")
    if ":" in s:
        mm, ss = s.split(":", 1)
        mm = float(mm.replace(",", "."))
        ss = float(ss.replace(",", "."))
        return mm + ss / 60.0
    return float(s.replace(",", "."))


def normalize_map_mode(map_mode: str | None) -> MapMode:
    """Normalize map_mode to internal format: avg or m1..m5.

    Accepts: "avg", "all", "maps"; "m1".."m5"; "map1".."map5"; digits 1..5.
    """
    if map_mode is None:
        return "avg"
    m = str(map_mode).strip().lower()
    if m in ("avg", "all", "maps", "overall"):
        return "avg"
    # allow "map1".."map5"
    if m.startswith("map") and m[3:].isdigit():
        m = f"m{m[3:]}"
    if m.isdigit():
        m = f"m{m}"
    if m not in {"m1", "m2", "m3", "m4", "m5"}:
        raise ValueError(f"invalid map_mode: {map_mode}")
    return m  # type: ignore


def filter_team_games(
    team_games: pd.DataFrame,
    *,
    team: str,
    filters: Filters,
    league_mode: Literal["all", "fixed", "auto"] = "all",
    map_mode: MapMode = "avg",
    max_games: int | None = None,
) -> pd.DataFrame:
    """Apply shared filters + optional league-mode + map filter.

    Requires team_games to have columns: team, date, league, map_number.
    If max_games is set, keeps only the most recent N games after filters.
    """
    map_mode = normalize_map_mode(map_mode)
    # shared filters
    base = Filters(year=filters.year, league=None, split=filters.split, playoffs=filters.playoffs)
    df = apply_filters(team_games, base)
    df = df[df["team"] == team].copy()

    if league_mode == "fixed" and filters.league:
        df = df[df["league"].fillna("") == str(filters.league)].copy()
    elif league_mode == "auto":
        # pick league for this team inside shared filters.
        # Prefer "major" leagues (ex.: LCK/LPL/LEC/LTA etc.) quando existirem,
        # para não cair em torneios de pré-temporada (ex.: KeSPA) só porque são mais recentes.
        leagues = df["league"].fillna("").astype(str).map(lambda x: x.strip())
        vc = leagues.value_counts()
        if not vc.empty:
            def _is_major(lg: str) -> bool:
                u = (lg or "").strip().upper()
                # principais + regiões grandes + academias relevantes
                return u.startswith(("LCK", "LPL", "LEC", "LCS", "LTA")) or u in {
                    "PCS", "VCS", "CBLOL", "LJL", "LLA", "TCL", "LCO", "EUM", "EMEA MASTERS"
                }

            major_idx = [lg for lg in vc.index.tolist() if _is_major(str(lg))]
            if len(major_idx) > 0:
                # mais comum dentre majors
                lg = str(major_idx[0]).strip()
            else:
                # fallback: mais comum no geral
                lg = str(vc.index[0]).strip()

            if lg:
                df = df[leagues == lg].copy()

    if map_mode != "avg":
        mnum = int(map_mode[1:])
        if "map_number" in df.columns:
            df = df[pd.to_numeric(df["map_number"], errors="coerce") == mnum].copy()

    if max_games is not None and int(max_games) > 0:
        df = df.sort_values("date", ascending=False).head(int(max_games)).copy()
    return df.sort_values("date")


def build_profile(
    team_games: pd.DataFrame,
    *,
    team: str,
    filters: Filters,
    league_mode: Literal["all", "fixed", "auto"] = "all",
    map_mode: MapMode = "avg",
    half_life_days: float | None = 180.0,
    max_games: int | None = None,
) -> TeamProfile:
    map_mode = normalize_map_mode(map_mode)
    df = filter_team_games(team_games, team=team, filters=filters, league_mode=league_mode, map_mode=map_mode, max_games=max_games)

    if df.empty:
        # return NaNs, caller can detect
        nan = float("nan")
        return TeamProfile(
            team=team,
            n_games=0,
            kills_for=nan,
            kills_against=nan,
            towers_for=nan,
            towers_against=nan,
            dragons_for=nan,
            dragons_against=nan,
            elementaldrakes_for=nan,
            elementaldrakes_against=nan,
            barons_for=nan,
            barons_against=nan,
            inhib_for=nan,
            inhib_against=nan,
            time_min=nan,
            kills_for_sd=nan,
            kills_against_sd=nan,
            towers_for_sd=nan,
            towers_against_sd=nan,
            dragons_for_sd=nan,
            dragons_against_sd=nan,
            elementaldrakes_for_sd=nan,
            elementaldrakes_against_sd=nan,
            barons_for_sd=nan,
            barons_against_sd=nan,
            inhib_for_sd=nan,
            inhib_against_sd=nan,
            time_min_sd=nan,
        )

    if half_life_days is None:
        w = np.ones(len(df), dtype=float)
    else:
        w = recency_weights(df["date"], half_life_days=float(half_life_days), ref_date=None)

    # Convert to numpy arrays
    kf = df["kills_for"].to_numpy(dtype=float)
    ka = df["kills_against"].to_numpy(dtype=float)
    tf = df["towers_for"].to_numpy(dtype=float)
    ta = df["towers_against"].to_numpy(dtype=float)
    dfc = df["dragons_for"].to_numpy(dtype=float)
    dac = df["dragons_against"].to_numpy(dtype=float)
    edf = pd.to_numeric(
        df.get("elementaldrakes_for", pd.Series(float("nan"), index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    eda = pd.to_numeric(
        df.get("elementaldrakes_against", pd.Series(float("nan"), index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    bf = df["barons_for"].to_numpy(dtype=float)
    ba = df["barons_against"].to_numpy(dtype=float)
    inf = df["inhibitors_for"].to_numpy(dtype=float)
    ina = df["inhibitors_against"].to_numpy(dtype=float)
    tm = df["game_time_min"].to_numpy(dtype=float)

    return TeamProfile(
        team=team,
        n_games=len(df),
        kills_for=_wmean(kf, w),
        kills_against=_wmean(ka, w),
        towers_for=_wmean(tf, w),
        towers_against=_wmean(ta, w),
        dragons_for=_wmean(dfc, w),
        dragons_against=_wmean(dac, w),
        elementaldrakes_for=_wmean(edf, w),
        elementaldrakes_against=_wmean(eda, w),
        barons_for=_wmean(bf, w),
        barons_against=_wmean(ba, w),
        inhib_for=_wmean(inf, w),
        inhib_against=_wmean(ina, w),
        time_min=_wmean(tm, w),
        kills_for_sd=_wstd(kf, w),
        kills_against_sd=_wstd(ka, w),
        towers_for_sd=_wstd(tf, w),
        towers_against_sd=_wstd(ta, w),
        dragons_for_sd=_wstd(dfc, w),
        dragons_against_sd=_wstd(dac, w),
        elementaldrakes_for_sd=_wstd(edf, w),
        elementaldrakes_against_sd=_wstd(eda, w),
        barons_for_sd=_wstd(bf, w),
        barons_against_sd=_wstd(ba, w),
        inhib_for_sd=_wstd(inf, w),
        inhib_against_sd=_wstd(ina, w),
        time_min_sd=_wstd(tm, w),
    )


@dataclass(frozen=True)
class MatchupTotals:
    mean: float
    sd: float
    dist: Dist


def _combine_mean(a_for: float, b_against: float) -> float:
    if not math.isfinite(a_for) or not math.isfinite(b_against):
        return float("nan")
    return 0.5 * (a_for + b_against)


def _combine_sd(a_for_sd: float, b_against_sd: float) -> float:
    if not math.isfinite(a_for_sd) or not math.isfinite(b_against_sd):
        return float("nan")
    # average variances, then sqrt
    return float(math.sqrt(max(0.5 * (a_for_sd * a_for_sd + b_against_sd * b_against_sd), 1e-9)))


def matchup_expected_totals(pA: TeamProfile, pB: TeamProfile) -> Dict[Metric, MatchupTotals]:
    """Predict expected *totals* for the matchup (A vs B) at map-level.

    Uses simple for/against mixing:
      E[A metric] = (A_for + B_against)/2
      E[B metric] = (B_for + A_against)/2
      Total = E[A] + E[B]

    Distributions:
      - kills, towers, time: normal
      - dragons, dragons_elemental, barons, inhibitors: poisson (mean = lambda)
    """
    out: Dict[Metric, MatchupTotals] = {}

    # KILLS
    a = _combine_mean(pA.kills_for, pB.kills_against)
    b = _combine_mean(pB.kills_for, pA.kills_against)
    mean = a + b
    sd = math.sqrt(_combine_sd(pA.kills_for_sd, pB.kills_against_sd) ** 2 + _combine_sd(pB.kills_for_sd, pA.kills_against_sd) ** 2)
    out["kills"] = MatchupTotals(mean=float(mean), sd=float(max(sd, 1.0)), dist="normal")

    # TOWERS
    a = _combine_mean(pA.towers_for, pB.towers_against)
    b = _combine_mean(pB.towers_for, pA.towers_against)
    mean = a + b
    sd = math.sqrt(_combine_sd(pA.towers_for_sd, pB.towers_against_sd) ** 2 + _combine_sd(pB.towers_for_sd, pA.towers_against_sd) ** 2)
    out["towers"] = MatchupTotals(mean=float(mean), sd=float(max(sd, 0.8)), dist="normal")

    # DRAGONS (Poisson)
    a = _combine_mean(pA.dragons_for, pB.dragons_against)
    b = _combine_mean(pB.dragons_for, pA.dragons_against)
    mean = max(a + b, 1e-6)
    out["dragons"] = MatchupTotals(mean=float(mean), sd=float(math.sqrt(mean)), dist="poisson")

    # ELEMENTAL DRAGONS (Poisson, excludes elder)
    a = _combine_mean(pA.elementaldrakes_for, pB.elementaldrakes_against)
    b = _combine_mean(pB.elementaldrakes_for, pA.elementaldrakes_against)
    if math.isfinite(a) and math.isfinite(b):
        mean = max(a + b, 1e-6)
        out["dragons_elemental"] = MatchupTotals(mean=float(mean), sd=float(math.sqrt(mean)), dist="poisson")

    # BARONS (Poisson)
    a = _combine_mean(pA.barons_for, pB.barons_against)
    b = _combine_mean(pB.barons_for, pA.barons_against)
    mean = max(a + b, 1e-6)
    out["barons"] = MatchupTotals(mean=float(mean), sd=float(math.sqrt(mean)), dist="poisson")

    # INHIBITORS (Poisson)
    a = _combine_mean(pA.inhib_for, pB.inhib_against)
    b = _combine_mean(pB.inhib_for, pA.inhib_against)
    mean = max(a + b, 1e-6)
    out["inhibitors"] = MatchupTotals(mean=float(mean), sd=float(math.sqrt(mean)), dist="poisson")

    # TIME (Normal)
    mean = 0.5 * (pA.time_min + pB.time_min) if math.isfinite(pA.time_min) and math.isfinite(pB.time_min) else float("nan")
    sd = 0.5 * math.sqrt(pA.time_min_sd ** 2 + pB.time_min_sd ** 2) if math.isfinite(pA.time_min_sd) and math.isfinite(pB.time_min_sd) else float("nan")
    out["time"] = MatchupTotals(mean=float(mean), sd=float(max(sd, 1.5)) if math.isfinite(sd) else float("nan"), dist="normal")

    return out


# ---------------------------------------------------------------------------
# Blend Dixon-Coles (alavanca O, gate ACCEPT 2026-07-06)
# tools/eval_totals_dixon_coles.py, fit hl 2025 / eval cego 2026 (5204 mapas):
# média DC (ataque×defesa multiplicativo, decaimento 60d) blendada 50/50 com a
# média V0 melhora o Brier de KILLS +0.0028, DRAGÕES +0.0012 e TEMPO +0.0005.
# Torres/barões/inibs PIORAM → ficam de fora. DC PURO nunca vence — só o blend.
# ---------------------------------------------------------------------------

DC_BLEND_HL_DAYS = 60.0
DC_BLEND_WEIGHT = 0.5
DC_BLEND_MIN_EFF = 8.0
_DC_BLEND_COLS = {"kills": ("kills_for", "kills_against"),
                  "dragons": ("dragons_for", "dragons_against"),
                  "time": ("game_time_min", None)}


def _dc_decayed(dates: pd.Series, values: pd.Series, as_of: pd.Timestamp,
                hl_days: float) -> tuple[float, float]:
    """(média decaída, n efetivo) com half-life em dias."""
    v = pd.to_numeric(values, errors="coerce")
    d = pd.to_datetime(dates, errors="coerce")
    ok = v.notna() & d.notna()
    if not bool(ok.any()):
        return float("nan"), 0.0
    age = (as_of - d[ok]).dt.total_seconds() / 86400.0
    w = 0.5 ** (age.clip(lower=0.0) / float(hl_days))
    sw = float(w.sum())
    if sw <= 1e-9:
        return float("nan"), 0.0
    return float((w * v[ok]).sum() / sw), sw


def apply_dc_blend(
    team_games: pd.DataFrame,
    team_a: str,
    team_b: str,
    totals: Dict[Metric, MatchupTotals],
    *,
    hl_days: float = DC_BLEND_HL_DAYS,
    blend: float = DC_BLEND_WEIGHT,
    min_eff: float = DC_BLEND_MIN_EFF,
) -> Dict[Metric, MatchupTotals]:
    """Blenda a média V0 com a média Dixon-Coles nos mercados validados.

    Falha graciosa: sem dado/amostra efetiva suficiente, mantém o V0 intacto.
    A distribuição não muda (alavanca K validou o sd) — só a média desloca.
    """
    try:
        if team_games is None or team_games.empty or "date" not in team_games.columns:
            return totals
        df = team_games
        if "game_time_min" not in df.columns and "gamelength" in df.columns:
            df = df.copy()
            df["game_time_min"] = pd.to_numeric(df["gamelength"], errors="coerce") / 60.0
        as_of = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(as_of):
            return totals
        rows_a = df[df["team"].astype(str) == str(team_a)]
        rows_b = df[df["team"].astype(str) == str(team_b)]
        if rows_a.empty or rows_b.empty:
            return totals

        out = dict(totals)
        for metric, (col_for, col_against) in _DC_BLEND_COLS.items():
            tot = out.get(metric)
            if tot is None or not math.isfinite(float(tot.mean)) or col_for not in df.columns:
                continue
            if col_against is None:
                m_a, n_a = _dc_decayed(rows_a["date"], rows_a[col_for], as_of, hl_days)
                m_b, n_b = _dc_decayed(rows_b["date"], rows_b[col_for], as_of, hl_days)
                if min(n_a, n_b) < float(min_eff) or not (math.isfinite(m_a) and math.isfinite(m_b)):
                    continue
                dc_mean = 0.5 * (m_a + m_b)
            else:
                if col_against not in df.columns:
                    continue
                lm, ln = _dc_decayed(df["date"], df[col_for], as_of, hl_days)
                if not math.isfinite(lm) or lm <= 1e-9 or ln < float(min_eff):
                    continue
                att_a, na1 = _dc_decayed(rows_a["date"], rows_a[col_for], as_of, hl_days)
                def_a, na2 = _dc_decayed(rows_a["date"], rows_a[col_against], as_of, hl_days)
                att_b, nb1 = _dc_decayed(rows_b["date"], rows_b[col_for], as_of, hl_days)
                def_b, nb2 = _dc_decayed(rows_b["date"], rows_b[col_against], as_of, hl_days)
                if min(na1, na2, nb1, nb2) < float(min_eff):
                    continue
                if not all(math.isfinite(x) for x in (att_a, def_a, att_b, def_b)):
                    continue
                dc_mean = (att_a * def_b + att_b * def_a) / lm
            if not math.isfinite(dc_mean) or dc_mean <= 0:
                continue
            new_mean = (1.0 - float(blend)) * float(tot.mean) + float(blend) * float(dc_mean)
            new_sd = float(math.sqrt(new_mean)) if tot.dist == "poisson" else float(tot.sd)
            out[metric] = MatchupTotals(mean=float(new_mean), sd=new_sd, dist=tot.dist)
        return out
    except Exception:
        return totals


def _norm_cdf(x: float, mu: float, sd: float) -> float:
    # standard normal CDF via erf
    if sd <= 0:
        return 0.0 if x < mu else 1.0
    z = (x - mu) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for Poisson(lam). Stable for small k (<=50-ish)."""
    if k < 0:
        return 0.0
    lam = max(float(lam), 0.0)
    # start with p0
    p = math.exp(-lam)
    s = p
    for i in range(1, k + 1):
        p *= lam / i
        s += p
    return min(max(s, 0.0), 1.0)


def prob_over_from_total(total: MatchupTotals, line: float) -> float:
    """Probability that integer total is OVER a .5 line.

    For normal we treat continuous and use P(X > line).
    For poisson we use integer logic (>= floor(line)+1).
    """
    if not math.isfinite(total.mean):
        return float("nan")

    if total.dist == "normal":
        # continuity correction for .5 lines is basically the same as threshold
        return float(1.0 - _norm_cdf(line, total.mean, max(total.sd, 1e-6)))

    # poisson
    thr = math.floor(line) + 1
    return float(1.0 - _poisson_cdf(thr - 1, total.mean))


def fair_odds_from_p(p: float) -> float:
    if not math.isfinite(p):
        return float("nan")
    return float(prob_to_decimal_odds(p))


def simulate_series_total(
    *,
    bo: int,
    p_blue_map: float,
    total: MatchupTotals,
    n_sims: int = 20000,
    seed: int = 1337,
) -> np.ndarray:
    """Simulate series total (sum across maps played).

    - Series stops when a team reaches wins_needed.
    - Each map total is sampled from the chosen distribution.

    Returns array of simulated series totals.
    """
    bo = int(bo)
    if bo not in (1, 3, 5):
        raise ValueError("bo must be 1, 3, or 5")
    wins_needed = bo // 2 + 1

    rng = np.random.default_rng(int(seed))

    def sample_one() -> float:
        if total.dist == "normal":
            return float(rng.normal(loc=total.mean, scale=max(total.sd, 1e-6)))
        # poisson
        return float(rng.poisson(lam=max(total.mean, 1e-6)))

    out = np.zeros(int(n_sims), dtype=float)
    for i in range(int(n_sims)):
        w_blue = 0
        w_red = 0
        s = 0.0
        while w_blue < wins_needed and w_red < wins_needed:
            # map result
            if rng.random() < p_blue_map:
                w_blue += 1
            else:
                w_red += 1
            s += sample_one()
        out[i] = s

    return out


def prob_over_series_from_sims(series_totals: np.ndarray, line: float) -> float:
    if series_totals.size == 0:
        return float("nan")
    return float(np.mean(series_totals > float(line)))


def try_load_p_map_from_artifact(artifact_path: str, team_blue: str, team_red: str) -> Optional[Tuple[float, float]]:
    """Try to load p_map from ml_artifact.json.

    Returns (p_raw, p_cal) or None if the artifact modules are unavailable.
    """
    try:
        from .artifact import MLArtifact
    except Exception:
        return None

    try:
        art = MLArtifact.load_json(artifact_path)
        p_raw = float(art.model.p_blue(team_blue, team_red))
        p_cal = float(art.calibrator.predict_proba(p_raw))
        return p_raw, p_cal
    except Exception:
        return None


# Backwards-compat alias (UI may import this name)
def total_over_prob(*args, **kwargs) -> float:
    """Backwards-compatible wrapper.

    Supported call styles:
      1) total_over_prob(total: MatchupTotals, line: float)
      2) total_over_prob(line: float, mean: float, sd: float, dist="normal")  # legacy
    """
    # keyword styles
    if kwargs:
        if "total" in kwargs and "line" in kwargs:
            return prob_over_from_total(total=kwargs["total"], line=float(kwargs["line"]))
        if "mean" in kwargs and "sd" in kwargs and "line" in kwargs:
            t = MatchupTotals(mean=float(kwargs["mean"]), sd=float(kwargs["sd"]), dist=kwargs.get("dist", "normal"))
            return prob_over_from_total(total=t, line=float(kwargs["line"]))

    # positional styles
    if len(args) == 2:
        total, line = args
        if isinstance(total, MatchupTotals):
            return prob_over_from_total(total=total, line=float(line))
        if isinstance(total, dict) and "mean" in total:
            t = MatchupTotals(
                mean=float(total["mean"]),
                sd=float(total.get("sd", 0.0)),
                dist=total.get("dist", "normal"),
            )
            return prob_over_from_total(total=t, line=float(line))

    if len(args) >= 3:
        # legacy: (line, mean, sd, dist?)
        line = float(args[0])
        mean = float(args[1])
        sd = float(args[2])
        dist = args[3] if len(args) >= 4 else "normal"
        t = MatchupTotals(mean=mean, sd=sd, dist=dist)
        return prob_over_from_total(total=t, line=line)

    raise TypeError("total_over_prob expects (total,line) or (line,mean,sd[,dist])")
