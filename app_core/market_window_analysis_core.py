"""Pure-data core for the "Mercado v2" panel.

Given a team's games filtered to a league/season scope and the opponent's
games in the same scope, computes for each historical window:

  - games played
  - hit count above the line
  - hit rate as a percentage
  - average market value
  - context strength: average value of the OPPONENTS that the target team
    faced in those games (so callers can flag "this team beat the over line
    but mostly against very-under opponents")

This module is UI-agnostic and side-effect free — it returns plain dicts
that any renderer can format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


WINDOWS_DEFAULT: tuple[tuple[str, Optional[int]], ...] = (
    ("Ano todo", None),
    ("Últimos 10", 10),
    ("Últimos 5", 5),
)


@dataclass(frozen=True)
class WindowStats:
    label: str
    n: int
    hits: int
    hit_rate_pct: float  # 0–100, NaN if n=0
    mean_value: float    # NaN if n=0
    # Context: avg opponent baseline value in this market for the games in this window
    opp_avg_baseline: float
    # Semaphore tone: "green" | "yellow" | "red" | "neutral"
    tone: str


@dataclass(frozen=True)
class HighlightedSample:
    """A single past game worth surfacing in the alert section."""
    date: str       # ISO-like date string for display
    opponent: str
    value: float    # the team's market value in that game
    opp_baseline_value: float  # the opponent's avg market value (context)
    side: str       # "hit" or "miss" relative to the line
    note: str       # short text e.g. "vs Hanwha (over forte, 32 méd)"


def _safe_series(df: Optional[pd.DataFrame], col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _sort_chronologically(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if "date" not in df.columns:
        return df
    return df.sort_values("date", ascending=True, kind="mergesort").reset_index(drop=True)


def _take_last_n(df: Optional[pd.DataFrame], n: Optional[int]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if n is None:
        return df
    return df.tail(int(n)).copy()


def _tone_for_hit_rate(hit_rate_pct: float, *, direction: str = "Over") -> str:
    """Map a hit rate to a semaphore tone.

    Cutoffs (updated 2026-05-30):
      >= 65% → green     (strong same-direction signal)
      40–64% → yellow
      < 40%  → red
    direction is informational; the same cutoffs apply to "Over" or "Under"
    because the caller chooses what the rate represents.
    """
    if not math.isfinite(float(hit_rate_pct)):
        return "neutral"
    p = float(hit_rate_pct)
    if p >= 65.0:
        return "green"
    if p >= 40.0:
        return "yellow"
    return "red"


def _opp_avg_for_team_games(
    *,
    team_df_window: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    metric_col: str,
) -> float:
    """For each game in team_df_window, look up the opponent's average value
    across the league scope. Returns the mean of those opponent averages.

    Tells you "what kind of opponents this team faced in this window" — a
    high number means the team faced overall-over opponents (so an Over hit
    is less impressive); a low number means mostly under opponents (so an
    Over hit IS impressive).
    """
    if team_df_window is None or team_df_window.empty:
        return float("nan")
    if "opponent" not in team_df_window.columns:
        return float("nan")

    league = league_scope_df if league_scope_df is not None else pd.DataFrame()
    if league.empty or "team" not in league.columns:
        return float("nan")

    opp_means: Dict[str, float] = {}
    for opp_name, grp in league.groupby("team"):
        s = _safe_series(grp, metric_col).dropna()
        if len(s) == 0:
            continue
        opp_means[str(opp_name)] = float(s.mean())

    vals = []
    for opp in team_df_window["opponent"].astype(str).tolist():
        v = opp_means.get(opp)
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def compute_window_stats(
    *,
    team_df: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    metric_col: str,
    line: float,
    direction: str = "Over",
    windows: tuple[tuple[str, Optional[int]], ...] = WINDOWS_DEFAULT,
) -> List[WindowStats]:
    """Compute WindowStats for each (label, last_n) tuple.

    team_df         : team's games already filtered to the scope of interest
    league_scope_df : all team-games in the same scope, used for opp baselines
    metric_col      : numeric column to apply the line to (e.g., "kills_for")
    line            : numeric threshold
    direction       : "Over" → hit when value > line; "Under" → value <= line
    """
    sorted_team = _sort_chronologically(team_df)
    direction_norm = "Under" if str(direction).strip().lower().startswith("u") else "Over"
    out: List[WindowStats] = []
    for label, n in windows:
        sub = _take_last_n(sorted_team, n)
        s = _safe_series(sub, metric_col).dropna()
        n_games = len(s)
        if n_games == 0:
            out.append(WindowStats(
                label=str(label), n=0, hits=0,
                hit_rate_pct=float("nan"), mean_value=float("nan"),
                opp_avg_baseline=float("nan"), tone="neutral",
            ))
            continue
        if direction_norm == "Under":
            hits_mask = s <= float(line)
        else:
            hits_mask = s > float(line)
        hits = int(hits_mask.sum())
        hit_rate = 100.0 * float(hits) / float(n_games)
        mean_val = float(s.mean())
        opp_avg = _opp_avg_for_team_games(
            team_df_window=sub, league_scope_df=league_scope_df, metric_col=metric_col,
        )
        out.append(WindowStats(
            label=str(label), n=n_games, hits=hits,
            hit_rate_pct=hit_rate, mean_value=mean_val,
            opp_avg_baseline=opp_avg, tone=_tone_for_hit_rate(hit_rate, direction=direction_norm),
        ))
    return out


def _categorize_opponent(opp_baseline: float, league_avg: float, *, direction: str) -> str:
    """Tag the opponent's baseline as 'very_under' / 'under' / 'avg' / 'over' / 'very_over'.

    Categories relative to league mean. For an Over line, facing a 'very_under'
    opponent means the hit is impressive (opponent normally suppresses kills).
    """
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


def highlight_context_samples(
    *,
    team_df: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    metric_col: str,
    line: float,
    direction: str = "Over",
    last_n_to_inspect: int = 10,
    max_samples_per_side: int = 2,
) -> Dict[str, List[HighlightedSample]]:
    """Find a few notable past games to surface in the context alerts panel.

    Returns {"impressive_hits": [...], "weak_hits": [...], "surprising_misses": [...]}
    """
    out = {"impressive_hits": [], "weak_hits": [], "surprising_misses": []}
    sorted_team = _sort_chronologically(team_df)
    sub = _take_last_n(sorted_team, last_n_to_inspect)
    if sub is None or sub.empty or "opponent" not in sub.columns:
        return out
    league = league_scope_df if league_scope_df is not None else pd.DataFrame()
    if league.empty or "team" not in league.columns:
        return out

    opp_means: Dict[str, float] = {}
    for opp_name, grp in league.groupby("team"):
        s = _safe_series(grp, metric_col).dropna()
        if len(s) > 0:
            opp_means[str(opp_name)] = float(s.mean())
    league_vals = _safe_series(league, metric_col).dropna()
    league_avg = float(league_vals.mean()) if len(league_vals) > 0 else float("nan")

    direction_norm = "Under" if str(direction).strip().lower().startswith("u") else "Over"
    for _, row in sub.iterrows():
        try:
            val = float(pd.to_numeric(row.get(metric_col), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        opp = str(row.get("opponent") or "")
        opp_base = opp_means.get(opp, float("nan"))
        date_str = str(row.get("date", "") or "")[:10]
        category = _categorize_opponent(opp_base, league_avg, direction=direction_norm)
        if direction_norm == "Over":
            is_hit = val > float(line)
        else:
            is_hit = val <= float(line)
        side = "hit" if is_hit else "miss"

        if is_hit and category in ("very_under", "under"):
            note = (f"vs {opp} ({category.replace('_', ' ')}, "
                    f"baseline {opp_base:.1f})") if math.isfinite(opp_base) else f"vs {opp}"
            out["impressive_hits"].append(HighlightedSample(
                date=date_str, opponent=opp, value=val, opp_baseline_value=opp_base,
                side=side, note=note,
            ))
        elif is_hit and category in ("very_over", "over"):
            note = (f"vs {opp} ({category.replace('_', ' ')}, "
                    f"baseline {opp_base:.1f})") if math.isfinite(opp_base) else f"vs {opp}"
            out["weak_hits"].append(HighlightedSample(
                date=date_str, opponent=opp, value=val, opp_baseline_value=opp_base,
                side=side, note=note,
            ))
        elif (not is_hit) and category in ("very_over", "over"):
            note = (f"vs {opp} ({category.replace('_', ' ')}, "
                    f"baseline {opp_base:.1f})") if math.isfinite(opp_base) else f"vs {opp}"
            out["surprising_misses"].append(HighlightedSample(
                date=date_str, opponent=opp, value=val, opp_baseline_value=opp_base,
                side=side, note=note,
            ))

    for k in out:
        out[k] = out[k][:int(max_samples_per_side)]
    return out


def build_alert_lines(
    *,
    team_name: str,
    samples: Dict[str, List[HighlightedSample]],
    direction: str = "Over",
) -> List[Dict[str, str]]:
    """Convert highlighted samples into display-ready alert lines.

    Each alert: {"tone": "warn"|"good"|"info", "text": "..."}.
    """
    lines: List[Dict[str, str]] = []
    direction_norm = "Under" if str(direction).strip().lower().startswith("u") else "Over"
    for s in samples.get("impressive_hits") or []:
        verb = "bateu" if direction_norm == "Over" else "ficou abaixo"
        text = (f"{team_name} {verb} a linha {s.note} — sinal forte "
                f"({s.value:.0f} vs baseline {s.opp_baseline_value:.1f}).")
        lines.append({"tone": "good", "text": text})
    for s in samples.get("weak_hits") or []:
        verb = "bateu" if direction_norm == "Over" else "ficou abaixo"
        text = (f"{team_name} {verb} a linha {s.note} — atenção: "
                f"adversário historicamente {direction_norm.lower()} "
                f"({s.opp_baseline_value:.1f}).")
        lines.append({"tone": "warn", "text": text})
    for s in samples.get("surprising_misses") or []:
        verb_neg = "não bateu" if direction_norm == "Over" else "ficou acima"
        text = (f"{team_name} {verb_neg} a linha {s.note} — surpresa: "
                f"adversário costuma puxar {direction_norm.lower()}.")
        lines.append({"tone": "info", "text": text})
    return lines


__all__ = [
    "WINDOWS_DEFAULT",
    "HighlightedSample",
    "WindowStats",
    "build_alert_lines",
    "compute_window_stats",
    "highlight_context_samples",
]
