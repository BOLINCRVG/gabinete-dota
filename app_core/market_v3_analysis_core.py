"""Market v3 analysis core (pure-data, UI-agnostic).

Builds the data structures needed by the Kills (and future) sub-tab:
  - Historic windows (último 5, último 10, último 20, ano todo) per team
  - Per-map breakdown (M1..M5) with mean / hit rate / trend
  - H2H summary (team A vs team B specifically)
  - Opponent profile table (every league team's avg + hit rate vs others)
  - Recent games table (last N with contextualized opponent strength tags)
  - Trend arrows from simple linear regression
  - Alerts derived from above

This module deliberately does NOT call Streamlit. All return values are
plain dicts / lists / dataclasses ready for a renderer to format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Window definitions (label, last_n_games). None = all.
HISTORIC_WINDOWS: Tuple[Tuple[str, Optional[int]], ...] = (
    ("Últimos 5", 5),
    ("Últimos 10", 10),
    ("Últimos 20", 20),
    ("Ano todo", None),
)


@dataclass(frozen=True)
class WindowStats:
    label: str
    n: int
    hits_over: int
    hits_under: int
    over_pct: float
    under_pct: float
    mean: float
    trend: str  # "up" | "down" | "flat" | "na"
    trend_value: float  # slope per game (NaN if na)


@dataclass(frozen=True)
class MapStats:
    map_n: int  # 1, 2, 3, 4, 5
    games: int
    mean: float
    sd: float
    hits_over: int
    hits_under: int
    over_pct: float
    under_pct: float
    trend: str
    trend_value: float
    note: str  # short text e.g. "+4.2 acima da média"


@dataclass(frozen=True)
class RecentGameRow:
    date: str
    opponent: str
    game_map: int
    value: float
    delta_from_line: float  # value - line
    hit: bool
    opponent_strength: str  # "very_under", "under", "avg", "over", "very_over"


@dataclass(frozen=True)
class OpponentProfileRow:
    team: str
    games_vs_league: int
    mean_vs_league: float
    over_pct_vs_league: float
    games_in_matchup: int  # H2H games (vs the target team)
    mean_in_matchup: float
    strength_tag: str


@dataclass(frozen=True)
class TeamArchetype:
    """Where a team sits on the over/under spectrum vs the league."""
    team: str
    archetype: str  # "very_under" | "under" | "neutral" | "over" | "very_over"
    mean: float
    league_mean: float
    diff_pct: float  # signed: positive = team is over the league


@dataclass(frozen=True)
class ContextualProfileRow:
    """Stats of a focal team playing against opponents of a given archetype."""
    opponent_archetype: str  # the archetype being queried (e.g. "under")
    n_opponents: int  # how many league teams fit that archetype
    games_played: int  # how many games the focal team played vs those opponents
    mean: float  # focal team's metric mean in those games
    sd: float
    hits_over: int
    hits_under: int
    over_pct: float
    under_pct: float
    opponents: List[str]  # the actual opponent names in this archetype


def _safe_series(df: Optional[pd.DataFrame], col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _sort_chronologically(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "date" not in df.columns:
        return df
    return df.sort_values("date", ascending=True, kind="mergesort").reset_index(drop=True)


def _take_last_n(df: Optional[pd.DataFrame], n: Optional[int]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if n is None:
        return df
    return df.tail(int(n)).copy()


def _trend_from_series(s: pd.Series) -> Tuple[str, float]:
    """Linear regression slope over the series (per-game). Returns ("up"/"down"/"flat"/"na", slope)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return ("na", float("nan"))
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    if np.std(y) < 1e-9:
        return ("flat", 0.0)
    # Simple polyfit degree 1
    try:
        slope, _intercept = np.polyfit(x, y, deg=1)
    except (np.linalg.LinAlgError, ValueError):
        return ("na", float("nan"))
    slope = float(slope)
    # Threshold based on absolute slope (per-game). Heuristic.
    if abs(slope) < 0.30:
        return ("flat", slope)
    return ("up" if slope > 0 else "down", slope)


def compute_window_stats(
    team_df: pd.DataFrame,
    *,
    metric_col: str,
    line: float,
    windows: Tuple[Tuple[str, Optional[int]], ...] = HISTORIC_WINDOWS,
    full_year_df: Optional[pd.DataFrame] = None,
) -> List[WindowStats]:
    """One WindowStats per (label, n_last) tuple.

    full_year_df: when provided, the "all-year" window (n_last is None) is
    computed from THIS dataframe (the team's full season, no league/split
    filter) instead of team_df. The bounded windows (L5/L10/L20) still use
    team_df (the selected recorte). This makes "Ano todo" = everything the
    team played in the year, while recent windows respect the filter.
    """
    sorted_df = _sort_chronologically(team_df)
    sorted_full = _sort_chronologically(full_year_df) if full_year_df is not None else None
    out: List[WindowStats] = []
    for label, n in windows:
        if n is None and sorted_full is not None:
            sub = sorted_full  # all-year window uses the unfiltered season df
        else:
            sub = _take_last_n(sorted_df, n)
        s = _safe_series(sub, metric_col).dropna()
        n_games = len(s)
        if n_games == 0:
            out.append(WindowStats(
                label=label, n=0, hits_over=0, hits_under=0,
                over_pct=float("nan"), under_pct=float("nan"),
                mean=float("nan"), trend="na", trend_value=float("nan"),
            ))
            continue
        over_mask = s > float(line)
        under_mask = s <= float(line)
        hits_over = int(over_mask.sum())
        hits_under = int(under_mask.sum())
        trend, slope = _trend_from_series(s)
        out.append(WindowStats(
            label=label, n=n_games,
            hits_over=hits_over, hits_under=hits_under,
            over_pct=100.0 * hits_over / n_games,
            under_pct=100.0 * hits_under / n_games,
            mean=float(s.mean()),
            trend=trend, trend_value=slope,
        ))
    return out


def compute_map_stats(
    team_df: pd.DataFrame,
    *,
    metric_col: str,
    line: float,
    overall_mean: Optional[float] = None,
) -> List[MapStats]:
    """One MapStats per map number 1..5. Skips maps with zero games.

    The map column is named `map_number` in team_games (from build_team_games)
    or `game` in raw Oracle Elixir rows — accept either."""
    if team_df is None or team_df.empty:
        return []
    map_col = "map_number" if "map_number" in team_df.columns else ("game" if "game" in team_df.columns else None)
    if map_col is None:
        return []
    out: List[MapStats] = []
    om = float(overall_mean) if (overall_mean is not None and math.isfinite(overall_mean)) else None
    for map_n in (1, 2, 3, 4, 5):
        try:
            mask = pd.to_numeric(team_df[map_col], errors="coerce").fillna(0).astype(int) == int(map_n)
        except (TypeError, ValueError):
            continue
        sub = team_df.loc[mask].copy()
        s = _safe_series(sub, metric_col).dropna()
        n = len(s)
        if n == 0:
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=0)) if n > 1 else 0.0
        over_mask = s > float(line)
        under_mask = s <= float(line)
        hits_over = int(over_mask.sum())
        hits_under = int(under_mask.sum())
        over_pct = 100.0 * float(hits_over) / float(n)
        under_pct = 100.0 * float(hits_under) / float(n)
        trend, slope = _trend_from_series(s)
        if om is not None and math.isfinite(om):
            diff = m - om
            sign = "+" if diff >= 0 else "−"
            note = f"{sign}{abs(diff):.1f} vs geral"
        else:
            note = ""
        out.append(MapStats(
            map_n=int(map_n), games=n, mean=m, sd=sd,
            hits_over=hits_over, hits_under=hits_under,
            over_pct=over_pct, under_pct=under_pct,
            trend=trend, trend_value=slope, note=note,
        ))
    return out


def compute_h2h_stats(
    team_a_df: pd.DataFrame,
    team_b_name: str,
    *,
    metric_col: str,
    line: float,
    last_n: Optional[int] = None,
) -> WindowStats:
    """Window stats restricted to games where team A faced team B."""
    if (team_a_df is None or team_a_df.empty
            or "opponent" not in team_a_df.columns):
        return WindowStats(
            label="H2H", n=0, hits_over=0, hits_under=0,
            over_pct=float("nan"), under_pct=float("nan"),
            mean=float("nan"), trend="na", trend_value=float("nan"),
        )
    h2h = team_a_df[team_a_df["opponent"].astype(str) == str(team_b_name)].copy()
    return compute_window_stats(h2h, metric_col=metric_col, line=line,
                                  windows=(("H2H", last_n),))[0]


def _categorize_opponent_strength(
    opp_baseline: float, league_avg: float
) -> str:
    """Tag opponent as very_under / under / avg / over / very_over relative to league."""
    if not (math.isfinite(opp_baseline) and math.isfinite(league_avg)) or league_avg <= 0:
        return "avg"
    diff_pct = (opp_baseline - league_avg) / league_avg
    if diff_pct <= -0.15:
        return "very_under"
    if diff_pct <= -0.05:
        return "under"
    if diff_pct < 0.05:
        return "avg"
    if diff_pct < 0.15:
        return "over"
    return "very_over"


def build_opponent_profile(
    league_scope_df: pd.DataFrame,
    target_team: str,
    *,
    metric_col: str,
    line: float,
) -> List[OpponentProfileRow]:
    """One row per team in the league with its baseline + H2H vs target."""
    if league_scope_df is None or league_scope_df.empty or "team" not in league_scope_df.columns:
        return []
    # League-wide mean for strength tagging
    league_vals = _safe_series(league_scope_df, metric_col).dropna()
    league_avg = float(league_vals.mean()) if len(league_vals) > 0 else float("nan")
    rows: List[OpponentProfileRow] = []
    for team_name, grp in league_scope_df.groupby("team"):
        s = _safe_series(grp, metric_col).dropna()
        n = len(s)
        if n == 0:
            continue
        mean_v = float(s.mean())
        over_pct = 100.0 * float((s > float(line)).mean())
        # H2H rows: where THIS team's "opponent" was the target_team
        if "opponent" in grp.columns:
            h2h = grp[grp["opponent"].astype(str) == str(target_team)]
            h2h_s = _safe_series(h2h, metric_col).dropna()
            n_h2h = len(h2h_s)
            mean_h2h = float(h2h_s.mean()) if n_h2h > 0 else float("nan")
        else:
            n_h2h, mean_h2h = 0, float("nan")
        rows.append(OpponentProfileRow(
            team=str(team_name), games_vs_league=n,
            mean_vs_league=mean_v, over_pct_vs_league=over_pct,
            games_in_matchup=n_h2h, mean_in_matchup=mean_h2h,
            strength_tag=_categorize_opponent_strength(mean_v, league_avg),
        ))
    rows.sort(key=lambda r: r.mean_vs_league, reverse=True)
    return rows


def build_recent_games(
    team_df: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    *,
    metric_col: str,
    line: float,
    last_n: int = 10,
) -> List[RecentGameRow]:
    """Last N games for the team, tagged with opponent's strength."""
    sorted_df = _sort_chronologically(team_df)
    sub = _take_last_n(sorted_df, last_n)
    if sub is None or sub.empty:
        return []

    # Build league baselines per opponent
    opp_means: Dict[str, float] = {}
    if league_scope_df is not None and not league_scope_df.empty and "team" in league_scope_df.columns:
        for opp_name, grp in league_scope_df.groupby("team"):
            s = _safe_series(grp, metric_col).dropna()
            if len(s) > 0:
                opp_means[str(opp_name)] = float(s.mean())
    league_vals = _safe_series(league_scope_df, metric_col).dropna()
    league_avg = float(league_vals.mean()) if len(league_vals) > 0 else float("nan")

    out: List[RecentGameRow] = []
    for _, row in sub.iterrows():
        try:
            val = float(pd.to_numeric(row.get(metric_col), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        opp = str(row.get("opponent", "") or "")
        try:
            map_n = int(row.get("map_number", row.get("game", 0)) or 0)
        except (TypeError, ValueError):
            map_n = 0
        date_str = str(row.get("date", "") or "")[:10]
        opp_base = opp_means.get(opp, float("nan"))
        strength = _categorize_opponent_strength(opp_base, league_avg)
        delta = val - float(line)
        out.append(RecentGameRow(
            date=date_str, opponent=opp, game_map=map_n, value=val,
            delta_from_line=delta, hit=(val > float(line)),
            opponent_strength=strength,
        ))
    return out


def build_alerts(
    *,
    team_name: str,
    window_stats: List[WindowStats],
    map_stats: List[MapStats],
    h2h: WindowStats,
    recent: List[RecentGameRow],
    line: float,
    overall_mean: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Synthesize alerts from the computed stats.

    Each alert: {"tone": "good"|"warn"|"info"|"alert", "text": "..."}.
    """
    alerts: List[Dict[str, str]] = []
    # Streak detection in recent games
    if recent:
        streak_kind = recent[-1].hit
        streak_len = 0
        for g in reversed(recent):
            if g.hit == streak_kind:
                streak_len += 1
            else:
                break
        if streak_len >= 3:
            side = "Over" if streak_kind else "Under"
            alerts.append({
                "tone": "alert",
                "text": f"{team_name} fechou {side} em {streak_len} jogos seguidos.",
            })

    # Map deviations (if a map's mean is much higher/lower than overall)
    if overall_mean is not None and math.isfinite(overall_mean) and map_stats:
        for ms in map_stats:
            if ms.games < 3:
                continue
            delta = ms.mean - overall_mean
            if abs(delta) >= 3.0:
                direction = "acima" if delta > 0 else "abaixo"
                tone = "alert" if abs(delta) >= 5.0 else "warn"
                alerts.append({
                    "tone": tone,
                    "text": (f"M{ms.map_n} de {team_name} tem média {abs(delta):.1f} "
                             f"{direction} da geral — considere mercado por mapa."),
                })

    # H2H surprise
    if h2h.n >= 3 and overall_mean is not None and math.isfinite(overall_mean):
        delta = h2h.mean - overall_mean
        if abs(delta) >= 2.0:
            direction = "acima" if delta > 0 else "abaixo"
            tone = "alert" if abs(delta) >= 4.0 else "warn"
            alerts.append({
                "tone": tone,
                "text": (f"H2H: {team_name} historicamente {abs(delta):.1f} {direction} "
                         f"da média neste mercado."),
            })

    # Strong window signal
    for ws in window_stats:
        if ws.n >= 5 and math.isfinite(ws.over_pct):
            if ws.over_pct >= 80.0:
                alerts.append({
                    "tone": "good",
                    "text": (f"{ws.label}: {team_name} bateu Over {line:g} em "
                             f"{ws.hits_over}/{ws.n} ({ws.over_pct:.0f}%) — sinal forte."),
                })
            elif ws.under_pct >= 80.0:
                alerts.append({
                    "tone": "good",
                    "text": (f"{ws.label}: {team_name} ficou Under {line:g} em "
                             f"{ws.hits_under}/{ws.n} ({ws.under_pct:.0f}%) — sinal forte."),
                })

    return alerts


def classify_team_archetype(
    *,
    team_name: str,
    team_df: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    metric_col: str,
) -> TeamArchetype:
    """Compute team's archetype relative to the league average for the metric.

    Cutoffs (signed % above/below league mean):
      diff > +15%  -> very_over
      +5%  to +15% -> over
      -5%  to +5%  -> neutral
      -15% to -5%  -> under
      < -15%       -> very_under
    """
    team_vals = _safe_series(team_df, metric_col).dropna()
    league_vals = _safe_series(league_scope_df, metric_col).dropna()
    team_mean = float(team_vals.mean()) if len(team_vals) > 0 else float("nan")
    league_mean = float(league_vals.mean()) if len(league_vals) > 0 else float("nan")
    if not (math.isfinite(team_mean) and math.isfinite(league_mean)) or league_mean <= 0:
        return TeamArchetype(team=team_name, archetype="neutral",
                              mean=team_mean, league_mean=league_mean,
                              diff_pct=float("nan"))
    diff_pct = (team_mean - league_mean) / league_mean
    if diff_pct > 0.15:
        arch = "very_over"
    elif diff_pct > 0.05:
        arch = "over"
    elif diff_pct >= -0.05:
        arch = "neutral"
    elif diff_pct >= -0.15:
        arch = "under"
    else:
        arch = "very_under"
    return TeamArchetype(team=team_name, archetype=arch,
                          mean=team_mean, league_mean=league_mean,
                          diff_pct=diff_pct)


def active_teams_in_recent_window(
    df: pd.DataFrame, *, days: int = 21,
) -> set:
    """Return teams that played at least once in the last `days` days.

    Reference date = max date in the dataframe (deterministic, no wall-clock).
    Empty df / missing columns → empty set (caller can skip filtering).

    Used to filter out roster-deleted teams ("Karmine Corp Blue", "Los Ratones",
    etc.) that have historical games in the scope but are no longer competing
    in the current split.
    """
    if df is None or df.empty or "team" not in df.columns or "date" not in df.columns:
        return set()
    d = pd.to_datetime(df["date"], errors="coerce")
    max_d = d.max()
    if pd.isna(max_d):
        return set()
    cutoff = max_d - pd.Timedelta(days=int(days))
    recent_mask = d >= cutoff
    return set(df.loc[recent_mask, "team"].astype(str).unique())


def filter_to_active_teams(
    df: pd.DataFrame, active_teams: set,
) -> pd.DataFrame:
    """Drop rows whose team OR opponent is not in `active_teams`.

    No-op when the set is empty (preserves callers that didn't compute
    activity). Returns a copy so the original df stays unmutated.
    """
    if df is None or df.empty or not active_teams:
        return df
    if "team" not in df.columns:
        return df
    out = df[df["team"].astype(str).isin(active_teams)]
    if "opponent" in out.columns:
        out = out[out["opponent"].astype(str).isin(active_teams)]
    return out.copy()


# International/aggregated tournaments — excluded when resolving a team's
# "home league + most-recent split". They have few games and inflate averages.
INTERNATIONAL_LEAGUES: frozenset = frozenset({
    "EWC", "MSI", "WORLDS", "FST", "IEM", "FIRST STAND",
})


def resolve_team_home_scope(
    team_games: pd.DataFrame,
    team: str,
    *,
    year: Optional[int] = None,
    exclude_international: bool = True,
) -> dict:
    """Resolve a team's domestic league + most-recent split, and the league pool.

    For cross-league matchups: each team should be evaluated against opponents
    of its OWN domestic league at the most recent split — NOT a short
    international event (EWC/MSI/...) which inflates averages with few games.

    Returns:
      {
        "league": str,            # domestic league of the team
        "split": str,             # most-recent split in that league/year
        "team_df": DataFrame,     # the team's games in that league+split
        "league_scope": DataFrame # all teams' games in that league+split
      }
    Empty/sane fallbacks when data is missing.
    """
    empty = {"league": "", "split": "", "team_df": pd.DataFrame(), "league_scope": pd.DataFrame()}
    if team_games is None or team_games.empty:
        return empty
    if not {"team", "league", "date"}.issubset(team_games.columns):
        return empty

    df = team_games
    if year is not None and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)]

    t_rows = df[df["team"].astype(str) == str(team)].copy()
    if exclude_international and "league" in t_rows.columns:
        t_rows = t_rows[~t_rows["league"].astype(str).str.upper().isin(INTERNATIONAL_LEAGUES)]
    if t_rows.empty:
        return empty

    t_rows["_d"] = pd.to_datetime(t_rows["date"], errors="coerce")
    t_rows = t_rows.dropna(subset=["_d"]).sort_values("_d")
    if t_rows.empty:
        return empty

    last = t_rows.iloc[-1]
    league = str(last.get("league", "") or "")
    split = str(last.get("split", "") or "")

    pool = df[df["league"].astype(str) == league].copy()
    if split and "split" in pool.columns:
        pool = pool[pool["split"].astype(str) == split]
    team_df = pool[pool["team"].astype(str) == str(team)].copy()
    return {
        "league": league,
        "split": split,
        "team_df": team_df,
        "league_scope": pool.copy(),
    }


def _list_teams_in_archetype(
    league_scope_df: pd.DataFrame,
    metric_col: str,
    *,
    target_archetypes: tuple,
    exclude_teams: tuple = (),
) -> List[str]:
    """Return league team names whose archetype matches `target_archetypes`."""
    if (league_scope_df is None or league_scope_df.empty
            or "team" not in league_scope_df.columns):
        return []
    league_vals = _safe_series(league_scope_df, metric_col).dropna()
    league_mean = float(league_vals.mean()) if len(league_vals) > 0 else float("nan")
    if not math.isfinite(league_mean) or league_mean <= 0:
        return []
    out: List[str] = []
    for team_name, grp in league_scope_df.groupby("team"):
        if str(team_name) in exclude_teams:
            continue
        s = _safe_series(grp, metric_col).dropna()
        if len(s) < 3:
            continue
        m = float(s.mean())
        diff = (m - league_mean) / league_mean
        if diff > 0.15:
            arch = "very_over"
        elif diff > 0.05:
            arch = "over"
        elif diff >= -0.05:
            arch = "neutral"
        elif diff >= -0.15:
            arch = "under"
        else:
            arch = "very_under"
        if arch in target_archetypes:
            out.append(str(team_name))
    return out


def build_contextual_opponent_profile(
    *,
    focal_team: str,
    focal_team_df: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    target_archetypes: tuple,  # e.g. ("under", "very_under")
    metric_col: str,
    line: float,
) -> ContextualProfileRow:
    """Compute the focal team's stats when playing against opponents of the
    target archetype set.

    Useful for surfacing 'Gen.G vs times under' style insights."""
    opponents = _list_teams_in_archetype(
        league_scope_df, metric_col,
        target_archetypes=tuple(target_archetypes),
        exclude_teams=(focal_team,),
    )
    if not opponents or focal_team_df is None or focal_team_df.empty or "opponent" not in focal_team_df.columns:
        return ContextualProfileRow(
            opponent_archetype="/".join(target_archetypes),
            n_opponents=len(opponents),
            games_played=0, mean=float("nan"), sd=0.0,
            hits_over=0, hits_under=0,
            over_pct=float("nan"), under_pct=float("nan"),
            opponents=list(opponents),
        )
    sub = focal_team_df[focal_team_df["opponent"].astype(str).isin(opponents)].copy()
    s = _safe_series(sub, metric_col).dropna()
    n = len(s)
    if n == 0:
        return ContextualProfileRow(
            opponent_archetype="/".join(target_archetypes),
            n_opponents=len(opponents),
            games_played=0, mean=float("nan"), sd=0.0,
            hits_over=0, hits_under=0,
            over_pct=float("nan"), under_pct=float("nan"),
            opponents=list(opponents),
        )
    hits_over = int((s > float(line)).sum())
    hits_under = int((s <= float(line)).sum())
    return ContextualProfileRow(
        opponent_archetype="/".join(target_archetypes),
        n_opponents=len(opponents),
        games_played=n,
        mean=float(s.mean()),
        sd=float(s.std(ddof=0)) if n > 1 else 0.0,
        hits_over=hits_over, hits_under=hits_under,
        over_pct=100.0 * hits_over / n,
        under_pct=100.0 * hits_under / n,
        opponents=list(opponents),
    )


@dataclass(frozen=True)
class StatsRow:
    """One row of comparative stats (team A vs team B vs league reference)."""
    label: str
    team_a_value: str  # display-ready string
    team_b_value: str
    league_value: str
    tone_a: str  # "good" | "warn" | "neutral" — color hint
    tone_b: str
    is_pct: bool  # for downstream formatting decisions
    # Second league reference, only populated in cross-league matchups where each
    # team belongs to a different league (team A's league vs team B's league).
    # Empty string when both teams share one league (single Liga column).
    league_value_b: str = ""
    # League ranking (1 = lowest value) shown next to the value. Only set when the
    # team is in the top-5 highest OR bottom-5 lowest for that metric; else "".
    rank_a: str = ""   # e.g. "(1)" lowest, "(2)" ... — empty when outside top/bottom 5
    rank_b: str = ""


def _fmt_num(v: float, dec: int = 1) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(dec)}f}"


def _fmt_pct(v: float, dec: int = 0) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(dec)}f}%"


def _fmt_mmss(v: float) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    mins = int(v); secs = round((v - mins) * 60)
    return f"{mins}:{secs:02d}"


def _safe_mean(df: Optional[pd.DataFrame], col: str) -> float:
    s = _safe_series(df, col).dropna()
    return float(s.mean()) if len(s) > 0 else float("nan")


def _safe_pct_above(df: Optional[pd.DataFrame], col: str, line: float) -> float:
    s = _safe_series(df, col).dropna()
    if len(s) == 0:
        return float("nan")
    return 100.0 * float((s > float(line)).mean())


def _safe_rate_bool(df: Optional[pd.DataFrame], col: str) -> float:
    """For boolean-like columns (firstblood etc): % of games where col >= 0.5."""
    s = _safe_series(df, col).dropna()
    if len(s) == 0:
        return float("nan")
    return 100.0 * float((s >= 0.5).mean())


def _tone_compare(team_value: float, league_value: float, *, higher_is_good: bool = True) -> str:
    """Tag whether team_value is notably above/below the league reference."""
    if not (math.isfinite(team_value) and math.isfinite(league_value)) or league_value == 0:
        return "neutral"
    diff = team_value - league_value
    pct = diff / abs(league_value) if league_value != 0 else 0.0
    if abs(pct) < 0.03:
        return "neutral"
    if (pct > 0) == higher_is_good:
        return "good"
    return "warn"


def _league_rank_for_team(
    league_scope_df: pd.DataFrame, col: str, team: str,
    *, top_n: int = 5,
) -> tuple[str, str]:
    """Return (rank_low, rank_high) labels for a team's mean of `col` vs league.

    rank_low  = position when sorting ascending (1 = lowest value), shown only
                if within the bottom `top_n`. Useful for "most under" reads.
    rank_high = position sorting descending (1 = highest), shown only if within
                top `top_n`. Useful for "most over" reads.
    Ties share the same position (standard competition ranking: 1,2,2,4...).
    Empty string when the team is outside both top/bottom N or data missing.
    """
    blank = ("", "")
    if (league_scope_df is None or league_scope_df.empty
            or col not in league_scope_df.columns or "team" not in league_scope_df.columns):
        return blank
    means = (
        league_scope_df.assign(_v=pd.to_numeric(league_scope_df[col], errors="coerce"))
        .dropna(subset=["_v"]).groupby("team")["_v"].mean()
    )
    if team not in means.index or means.empty:
        return blank
    tv = float(means.loc[team])
    vals = means.values
    # Competition rank: how many teams are strictly lower (for ascending).
    n_lower = int((vals < tv - 1e-9).sum())
    n_higher = int((vals > tv + 1e-9).sum())
    rank_low = n_lower + 1     # 1 = lowest
    rank_high = n_higher + 1   # 1 = highest
    low_lbl = f"({rank_low})" if rank_low <= top_n else ""
    high_lbl = f"({rank_high})" if rank_high <= top_n else ""
    return (low_lbl, high_lbl)


def build_market_stats_table(
    *,
    team_a: str,
    team_b: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    metric_col: str,
    line: float,
    metric_label: str = "Kills",
    for_col: Optional[str] = "kills_for",
    against_col: Optional[str] = "kills_against",
    metric_decimals: int = 1,
    is_time_metric: bool = False,
    league_scope_a: Optional[pd.DataFrame] = None,
    league_scope_b: Optional[pd.DataFrame] = None,
    cross_league: bool = False,
) -> List[StatsRow]:
    """Build a comparative stats table for any market sub-tab.

    metric_col      → combined-total column used for hit-rate (e.g. mv2_total_kills, mv2_total_towers).
    metric_label    → display name ("Kills", "Torres", "Drakes", "Barões", "Inibidores", "Tempo").
    for_col / against_col → per-team columns ("kills_for", "kills_against"). For metrics with
                              no per-team split (e.g. game_time_min), pass None.
    metric_decimals → decimals for the metric values (0 for kills/towers, 1 for tempo).
    is_time_metric  → True for game_time_min (formats as mm:ss instead of plain number).
    league_scope_a/b → per-team league pools for RANKING (cross-league: each team
                        ranked in its own league). Default to league_scope_df.
    """
    rows: List[StatsRow] = []
    _scope_a = league_scope_a if league_scope_a is not None else league_scope_df
    _scope_b = league_scope_b if league_scope_b is not None else league_scope_df

    def _rank_one(rank_col: Optional[str], team: str, scope: pd.DataFrame,
                   *, side: str = "both") -> str:
        # side: "low" → only ↓N (most under), "high" → only ↑N (most over),
        # "both" → ↓N ↑N. The ranking is always by `rank_col` (combined metric),
        # i.e. how under/over the team is vs its league.
        if not rank_col:
            return ""
        low, high = _league_rank_for_team(scope, rank_col, team)
        parts = []
        if low and side in ("low", "both"):
            parts.append(f"↓{low.strip('()')}")   # 1 = menor da liga (mais under)
        if high and side in ("high", "both"):
            parts.append(f"↑{high.strip('()')}")  # 1 = maior da liga (mais over)
        return " ".join(parts)

    def _row_num(label: str, stat_fn: "Callable[[Optional[pd.DataFrame]], float]",
                  *, higher_is_good: bool = True, fmt: str = "num", dec: int = 1,
                  rank_col: Optional[str] = None, rank_side: str = "both") -> StatsRow:
        # Each team is compared to / shown against ITS OWN league reference:
        # in cross-league matchups Liga A = team A's league, Liga B = team B's.
        # When both teams share a league (the common case) league_b == league_a
        # and only one Liga column is rendered.
        fmt_fn = _fmt_pct if fmt == "pct" else (_fmt_mmss if fmt == "mmss" else _fmt_num)

        def _fmt_v(x: float) -> str:
            return fmt_fn(x) if fmt == "mmss" else fmt_fn(x, dec)

        value_a = stat_fn(df_a)
        value_b = stat_fn(df_b)
        league_a = stat_fn(_scope_a)
        league_b = stat_fn(_scope_b) if cross_league else league_a
        return StatsRow(
            label=label,
            team_a_value=_fmt_v(value_a),
            team_b_value=_fmt_v(value_b),
            league_value=_fmt_v(league_a),
            league_value_b=(_fmt_v(league_b) if cross_league else ""),
            tone_a=_tone_compare(value_a, league_a, higher_is_good=higher_is_good),
            tone_b=_tone_compare(value_b, league_b, higher_is_good=higher_is_good),
            is_pct=(fmt == "pct"),
            rank_a=_rank_one(rank_col, team_a, _scope_a, side=rank_side),
            rank_b=_rank_one(rank_col, team_b, _scope_b, side=rank_side),
        )

    # # Games (no tone)
    n_a = len(df_a) if df_a is not None else 0
    n_b = len(df_b) if df_b is not None else 0
    rows.append(StatsRow(
        label="Jogos no recorte",
        team_a_value=str(n_a),
        team_b_value=str(n_b),
        league_value="—",
        league_value_b=("—" if cross_league else ""),
        tone_a="neutral", tone_b="neutral", is_pct=False,
    ))
    # Win Rate
    rows.append(_row_num(
        "Win Rate",
        lambda d: 100.0 * _safe_mean(d, "win"),
        higher_is_good=True, fmt="pct", dec=0,
    ))
    # Game Time
    rows.append(_row_num(
        "Game Time (mm:ss)",
        lambda d: _safe_mean(d, "game_time_min"),
        higher_is_good=False, fmt="mmss",  # neutral comparison really; "good" not meaningful
    ))
    # Per-team metric (e.g. Kills feitas/sofridas). Skip for time-only metrics.
    if for_col and against_col:
        rows.append(_row_num(
            f"{metric_label} feitas (/jogo)",
            lambda d, c=for_col: _safe_mean(d, c),
            higher_is_good=True, fmt="num", dec=int(metric_decimals),
        ))
        rows.append(_row_num(
            f"{metric_label} sofridas (/jogo)",
            lambda d, c=against_col: _safe_mean(d, c),
            higher_is_good=False, fmt="num", dec=int(metric_decimals),
        ))
    # Combined metric (= the col used for over/under)
    combined_label = f"Combined {metric_label.lower()}" if not is_time_metric else f"{metric_label} (/jogo)"
    rows.append(_row_num(
        f"{combined_label}",
        lambda d: _safe_mean(d, metric_col),
        higher_is_good=True,
        fmt="mmss" if is_time_metric else "num",
        dec=int(metric_decimals),
    ))
    # First objective rates (related to kills tempo)
    for col, label in (("firstblood", "First Blood %"),
                        ("firsttower", "First Tower %"),
                        ("firstdragon", "First Dragon %")):
        rows.append(_row_num(
            label,
            lambda d, c=col: _safe_rate_bool(d, c),
            higher_is_good=True, fmt="pct", dec=0,
        ))
    # Hit rate at the chosen line (Over and Under)
    line_str = (_fmt_mmss(line) if is_time_metric else f"{float(line):g}")
    suffix = " (combined)" if (for_col and against_col and not is_time_metric) else ""
    # League ranking now sits next to the % rows (per user): ↑N (most over) on
    # the Over line, ↓N (most under) on the Under line. Ranking is by the
    # combined metric — i.e. how over/under the team is vs its league.
    rows.append(_row_num(
        f"Over {line_str}{suffix}",
        lambda d: _safe_pct_above(d, metric_col, line),
        higher_is_good=True, fmt="pct", dec=0,
        rank_col=metric_col, rank_side="high",
    ))

    def _under_pct(d: "Optional[pd.DataFrame]") -> float:
        ov = _safe_pct_above(d, metric_col, line)
        return (100.0 - ov) if math.isfinite(ov) else float("nan")

    rows.append(_row_num(
        f"Under {line_str}{suffix}",
        _under_pct,
        higher_is_good=True, fmt="pct", dec=0,
        rank_col=metric_col, rank_side="low",
    ))
    return rows


# Backwards-compat alias (existing callers).
build_kills_stats_table = build_market_stats_table


def opposite_archetype_set(archetype: str, *, side_key: str = "") -> tuple:
    """Return the opponent archetype set for the contextual profile.

    Preferred: call `opposite_archetype_from_line` (uses team mean vs the
    specific line). This function is kept for backward compat and for the
    few callers that don't have access to team data.

    side_key (Over/Under) overrides archetype when provided:
      "Under" → show Over opponents (hardest test for Under)
      "Over"  → show Under opponents
    Archetype-only fallback:
      over/very_over → under opponents
      under/very_under → over opponents
      neutral without side_key → both extremes
    """
    sk = str(side_key).strip().lower()
    if sk in ("under", "abaixo"):
        return ("over", "very_over")
    if sk in ("over", "acima"):
        return ("under", "very_under")
    # Archetype fallback (no line context).
    if archetype in ("very_over", "over"):
        return ("under", "very_under")
    if archetype in ("very_under", "under"):
        return ("over", "very_over")
    return ("over", "very_over", "under", "very_under")


def opposite_archetype_from_line(
    team_df: pd.DataFrame,
    metric_col: str,
    line: float,
) -> tuple:
    """Correct approach for all archetypes: compare team mean vs the specific line.

    This is strictly more accurate than archetype-based because:
    - A team can be "over" vs league average but still tend Under for a HIGH line.
    - The question answered is: "which opponents challenge THIS specific bet?"
      Under bet → hardest test = Over opponents (they push the metric up)
      Over  bet → hardest test = Under opponents (they suppress the metric)

    Falls back to both extremes when mean is unavailable.
    """
    if team_df is None or team_df.empty or metric_col not in team_df.columns:
        return ("over", "very_over", "under", "very_under")
    try:
        mean_val = float(pd.to_numeric(team_df[metric_col], errors="coerce").dropna().mean())
    except Exception:
        return ("over", "very_over", "under", "very_under")
    if not math.isfinite(mean_val):
        return ("over", "very_over", "under", "very_under")
    if mean_val <= float(line):
        # Team tends Under this line → show vs Over opponents (hardest test)
        return ("over", "very_over")
    else:
        # Team tends Over this line → show vs Under opponents (hardest test)
        return ("under", "very_under")


# Opponent-group picker: maps a user-facing choice to a set of archetypes.
OPPONENT_GROUP_CHOICES: tuple[str, ...] = (
    "Muito under",
    "Under",
    "Neutro",
    "Over",
    "Muito over",
    "Todos os extremos",
)

_GROUP_TO_ARCHETYPES: dict[str, tuple] = {
    "Muito under": ("very_under",),
    "Under": ("under", "very_under"),
    "Neutro": ("neutral",),
    "Over": ("over", "very_over"),
    "Muito over": ("very_over",),
    "Todos os extremos": ("over", "very_over", "under", "very_under"),
}

# Default group label per team archetype: shows the team vs opponents OF ITS
# OWN profile (neutral team → vs neutral opponents). The user can switch.
_ARCHETYPE_TO_GROUP_LABEL: dict[str, str] = {
    "very_under": "Muito under",
    "under": "Under",
    "neutral": "Neutro",
    "over": "Over",
    "very_over": "Muito over",
}


def default_group_label_for_archetype(archetype: str) -> str:
    """The selector's default choice for a team: its own archetype's group."""
    return _ARCHETYPE_TO_GROUP_LABEL.get(str(archetype or "").strip(), "Neutro")


def archetypes_for_group_choice(choice: str) -> tuple:
    """Resolve a user's opponent-group choice into target archetypes.

    Unknown choice falls back to neutral (safe default).
    """
    return _GROUP_TO_ARCHETYPES.get(str(choice or "").strip(), ("neutral",))


__all__ = [
    "HISTORIC_WINDOWS",
    "INTERNATIONAL_LEAGUES",
    "OPPONENT_GROUP_CHOICES",
    "ContextualProfileRow",
    "MapStats",
    "OpponentProfileRow",
    "RecentGameRow",
    "StatsRow",
    "TeamArchetype",
    "WindowStats",
    "archetypes_for_group_choice",
    "build_alerts",
    "build_contextual_opponent_profile",
    "build_kills_stats_table",
    "build_market_stats_table",
    "build_opponent_profile",
    "build_recent_games",
    "classify_team_archetype",
    "compute_h2h_stats",
    "compute_map_stats",
    "compute_window_stats",
    "default_group_label_for_archetype",
    "opposite_archetype_from_line",
    "opposite_archetype_set",
    "resolve_team_home_scope",
]
