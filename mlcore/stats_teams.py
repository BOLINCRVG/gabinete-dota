from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

# Columns used from OracleElixir team rows.
# Includes base market stats plus optional quality/context fields used by live ML layers.
BASE_COLS = [
    "gameid",
    "date",
    "year",
    "split",
    "playoffs",
    "league",
    "game",
    "participantid",
    "side",
    "teamname",
    "gamelength",
    "result",
    "kills",
    "deaths",
    "assists",
    "totalgold",
    "towers",
    "dragons",
    "barons",
    "inhibitors",
    "firstblood",
    "firsttower",
    "firstdragon",
    "firstherald",
    "firstbaron",
    "firstinhibitor",
    "firstmidtower",
    "firsttothreetowers",
    "elementaldrakes",
    "heralds",
    # Dota (Gabinete Dota): mercado megacreeps; ausente no CSV do LoL -> o
    # filtro "if c in available" ignora sem custo.
    "megacreeps",
    "voidgrubs",
    "void_grubs",
    "atakhans",
    "atakhan",
    "turretplates",
    "golddiffat10",
    "golddiffat15",
    "golddiffat20",
    "xpdiffat10",
    "xpdiffat15",
    "xpdiffat20",
    "csdiffat10",
    "csdiffat15",
    "csdiffat20",
    "killsat15",
    "opp_killsat15",
    "killsat20",
    "opp_killsat20",
    "datacompleteness",
]

# Our initial "tabs" (requested):
# - Win Rate
# - Total Kills
# - Total Towers
# - Total Dragons
# - Total Nashors (Barons)
# - Total Inhibitors
# - Game Time


@dataclass(frozen=True)
class Filters:
    """Filtering options applied before aggregations."""

    year: Optional[int] = None
    league: Optional[str] = None
    split: Optional[str] = None
    playoffs: Optional[bool] = None


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Parse OracleElixir 'date' column.

    Common format: 'YYYY-MM-DD HH:MM:SS' (as in your screenshot).
    We accept any pandas-parseable timestamp.
    """
    dt = pd.to_datetime(series, errors="coerce")
    # If timezone-aware, convert to naive UTC for consistent diffs.
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt


def load_team_rows(csv_path: str | Path) -> pd.DataFrame:
    """Load only TEAM rows (participantid 100/200) and only needed columns."""
    csv_path = str(csv_path)

    # Discover available columns (cheap) then read only what exists.
    preview = pd.read_csv(csv_path, nrows=1, low_memory=False)
    available = set(preview.columns)
    usecols = [c for c in BASE_COLS if c in available]

    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)

    # Filter team rows
    if "participantid" not in df.columns:
        raise ValueError("CSV does not contain 'participantid'. This is required.")
    df = df[df["participantid"].isin([100, 200])].copy()

    # Normalize column presence with defaults
    for col in ["split", "league", "playoffs", "year", "game", "side"]:
        if col not in df.columns:
            df[col] = None

    # Normalize types
    df["date"] = _safe_to_datetime(df["date"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "playoffs" in df.columns:
        # OracleElixir uses 0/1; keep as boolean with NA allowed.
        df["playoffs"] = pd.to_numeric(df["playoffs"], errors="coerce").astype("Int64")

    # Keep last record per (gameid, teamname) if duplicates exist.
    df.sort_values(["gameid", "date"], inplace=True, kind="mergesort")
    df = df.drop_duplicates(subset=["gameid", "teamname"], keep="last")

    return df


def build_team_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Pair each team row with its opponent within the same gameid.

    Output: one row per (gameid, team), with opponent stats and "total" metrics.
    """
    df = team_rows.copy()

    # Ensure required minimum columns
    for col in ["gameid", "teamname", "date", "result"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Self-join on gameid to obtain opponent rows.
    opp = df.add_suffix("_opp")
    merged = df.merge(opp, left_on="gameid", right_on="gameid_opp", how="inner")
    merged = merged[merged["participantid"] != merged["participantid_opp"]].copy()

    # Choose columns we will expose.
    out = pd.DataFrame({
        "gameid": merged["gameid"],
        "date": merged["date"],
        "map_number": pd.to_numeric(merged.get("game"), errors="coerce"),
        "year": merged.get("year"),
        "league": merged.get("league"),
        "split": merged.get("split"),
        "playoffs": merged.get("playoffs"),
        "side": merged.get("side"),
        "team": merged["teamname"],
        "opponent": merged["teamname_opp"],
        "win": pd.to_numeric(merged["result"], errors="coerce"),
        "gamelength": pd.to_numeric(merged.get("gamelength"), errors="coerce"),
        "kills_for": pd.to_numeric(merged.get("kills"), errors="coerce"),
        "deaths_for": pd.to_numeric(merged.get("deaths"), errors="coerce"),
        "assists_for": pd.to_numeric(merged.get("assists"), errors="coerce"),
        "gold_for": pd.to_numeric(merged.get("totalgold"), errors="coerce"),
        "kills_against": pd.to_numeric(merged.get("kills_opp"), errors="coerce"),
        "deaths_against": pd.to_numeric(merged.get("deaths_opp"), errors="coerce"),
        "assists_against": pd.to_numeric(merged.get("assists_opp"), errors="coerce"),
        "gold_against": pd.to_numeric(merged.get("totalgold_opp"), errors="coerce"),
        "towers_for": pd.to_numeric(merged.get("towers"), errors="coerce"),
        "towers_against": pd.to_numeric(merged.get("towers_opp"), errors="coerce"),
        "dragons_for": pd.to_numeric(merged.get("dragons"), errors="coerce"),
        "dragons_against": pd.to_numeric(merged.get("dragons_opp"), errors="coerce"),
        "barons_for": pd.to_numeric(merged.get("barons"), errors="coerce"),
        "barons_against": pd.to_numeric(merged.get("barons_opp"), errors="coerce"),
        "inhibitors_for": pd.to_numeric(merged.get("inhibitors"), errors="coerce"),
        "inhibitors_against": pd.to_numeric(merged.get("inhibitors_opp"), errors="coerce"),
    })

    # Add first-objective booleans (optional columns)
    for col in [
        "firstblood",
        "firsttower",
        "firstdragon",
        "firstherald",
        "firstbaron",
        "firstinhibitor",
        "firstmidtower",
        "firsttothreetowers",
    ]:
        if col in merged.columns:
            out[col] = pd.to_numeric(merged[col], errors="coerce")
            out[f"{col}_for"] = pd.to_numeric(merged[col], errors="coerce")
            out[f"{col}_against"] = pd.to_numeric(merged.get(f"{col}_opp"), errors="coerce")

    def _find_present_col(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in merged.columns:
                return c
        return None

    def _add_numeric_pair(canon: str, candidates: list[str]) -> None:
        src = _find_present_col(candidates)
        if not src:
            return
        out[f"{canon}_for"] = pd.to_numeric(merged.get(src), errors="coerce")
        out[f"{canon}_against"] = pd.to_numeric(merged.get(f"{src}_opp"), errors="coerce")

    # Optional quality/context numeric pairs for ML runtime.
    _add_numeric_pair("elementaldrakes", ["elementaldrakes"])
    _add_numeric_pair("heralds", ["heralds"])
    _add_numeric_pair("voidgrubs", ["voidgrubs", "void_grubs"])
    _add_numeric_pair("atakhans", ["atakhans", "atakhan"])
    _add_numeric_pair("turretplates", ["turretplates"])
    _add_numeric_pair("golddiffat10", ["golddiffat10"])
    _add_numeric_pair("golddiffat15", ["golddiffat15"])
    _add_numeric_pair("golddiffat20", ["golddiffat20"])
    _add_numeric_pair("xpdiffat10", ["xpdiffat10"])
    _add_numeric_pair("xpdiffat15", ["xpdiffat15"])
    _add_numeric_pair("xpdiffat20", ["xpdiffat20"])
    _add_numeric_pair("csdiffat10", ["csdiffat10"])
    _add_numeric_pair("csdiffat15", ["csdiffat15"])
    _add_numeric_pair("csdiffat20", ["csdiffat20"])
    _add_numeric_pair("killsat15", ["killsat15"])
    _add_numeric_pair("killsat20", ["killsat20"])
    # Dota (Gabinete Dota): megacreeps causados no jogo (0/1). O CSV do LoL
    # nao tem a coluna -> no-op aqui.
    _add_numeric_pair("megacreeps", ["megacreeps"])

    dc_src = _find_present_col(["datacompleteness"])
    if dc_src:
        out["datacompleteness_for"] = (
            pd.Series(merged.get(dc_src))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        out["datacompleteness_against"] = (
            pd.Series(merged.get(f"{dc_src}_opp"))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # Derived metrics
    out["game_time_min"] = out["gamelength"] / 60.0
    out["total_kills"] = out["kills_for"] + out["kills_against"]
    if "gold_for" in out.columns and "gold_against" in out.columns:
        out["total_gold"] = out["gold_for"] + out["gold_against"]
    if "deaths_for" in out.columns and "deaths_against" in out.columns:
        out["total_deaths"] = out["deaths_for"] + out["deaths_against"]
    out["total_towers"] = out["towers_for"] + out["towers_against"]
    out["total_dragons"] = out["dragons_for"] + out["dragons_against"]
    if "elementaldrakes_for" in out.columns and "elementaldrakes_against" in out.columns:
        out["total_elementaldrakes"] = out["elementaldrakes_for"] + out["elementaldrakes_against"]
    out["total_nashors"] = out["barons_for"] + out["barons_against"]
    out["total_inhibitors"] = out["inhibitors_for"] + out["inhibitors_against"]
    if "megacreeps_for" in out.columns and "megacreeps_against" in out.columns:
        out["total_megacreeps"] = out["megacreeps_for"] + out["megacreeps_against"]
    if "killsat15_for" in out.columns and "killsat15_against" in out.columns:
        out["total_kills_at15"] = out["killsat15_for"] + out["killsat15_against"]
    if "killsat20_for" in out.columns and "killsat20_against" in out.columns:
        out["total_kills_at20"] = out["killsat20_for"] + out["killsat20_against"]
    if "total_kills_at15" in out.columns and "total_kills_at20" in out.columns:
        out["total_kills_15_20"] = out["total_kills_at20"] - out["total_kills_at15"]
    if "total_kills" in out.columns and "total_kills_at20" in out.columns:
        out["total_kills_20_plus"] = out["total_kills"] - out["total_kills_at20"]

    # Normalize playoffs to boolean where possible
    if "playoffs" in out.columns:
        po = out["playoffs"]
        out["playoffs"] = po.map(lambda x: bool(int(x)) if pd.notna(x) else None)

    return out


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df
    if f.year is not None and "year" in out.columns:
        out = out[out["year"].astype("Int64") == int(f.year)]
    if f.league and f.league.upper() != "ALL" and "league" in out.columns:
        out = out[out["league"].fillna("") == f.league]
    if f.split and f.split.upper() != "ALL" and "split" in out.columns:
        out = out[out["split"].fillna("") == f.split]
    if f.playoffs is not None and "playoffs" in out.columns:
        out = out[out["playoffs"] == bool(f.playoffs)]
    return out


def recency_weights(dates: pd.Series, half_life_days: float, ref_date: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Compute exponential-decay weights with a half-life.

    When `ref_date` is None, derives it from the input dates' max value to keep
    weights deterministic across reruns (avoids wall-clock drift). If the dates
    series has no valid timestamps, falls back to today (midnight) so weights
    still degrade meaningfully when data is empty.
    """
    d = pd.to_datetime(dates, errors="coerce")
    if ref_date is None:
        try:
            max_d = d.max()
        except Exception:
            max_d = None
        if max_d is not None and pd.notna(max_d):
            ref_date = pd.Timestamp(max_d).tz_localize(None) if getattr(max_d, "tzinfo", None) else pd.Timestamp(max_d)
        else:
            ref_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    age_days = (ref_date - d).dt.total_seconds() / (24 * 3600)
    age_days = age_days.fillna(0).to_numpy(dtype=float)

    # weight = 0.5 ** (age / half_life)
    # guard against negative ages (future dates): clip to 0
    age_days = np.maximum(age_days, 0.0)
    w = np.power(0.5, age_days / float(half_life_days))
    # avoid all-zero
    if not np.isfinite(w).any() or w.sum() == 0:
        w = np.ones_like(w)
    return w


def _wmean(values: pd.Series, weights: np.ndarray) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v)
    if m.sum() == 0:
        return float("nan")
    w = weights[: len(v)]
    w = w[m]
    v = v[m]
    if w.sum() == 0:
        return float(np.mean(v))
    return float(np.average(v, weights=w))


SummaryMode = Literal["all", "last"]


def summarize_team(
    team_games: pd.DataFrame,
    team: str,
    *,
    mode: SummaryMode,
    last_n: Optional[int] = None,
    half_life_days: float = 180.0,
    ref_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Compute requested stats for a team.

    mode='all' uses recency weights (half-life) for averages.
    mode='last' uses simple averages over the last N games.
    """
    df = team_games[team_games["team"] == team].copy()
    df = df.sort_values("date")

    if mode == "last":
        if not last_n or last_n <= 0:
            raise ValueError("last_n must be a positive integer when mode='last'")
        df = df.tail(int(last_n)).copy()
        weights = np.ones(len(df), dtype=float)
    else:
        weights = recency_weights(df["date"], half_life_days=half_life_days, ref_date=ref_date)

    out: Dict[str, Any] = {
        "team": team,
        "games": len(df),
        "win_rate": _wmean(df["win"], weights),
        "avg_total_kills": _wmean(df["total_kills"], weights),
        "avg_total_towers": _wmean(df["total_towers"], weights),
        "avg_total_dragons": _wmean(df["total_dragons"], weights),
        "avg_total_nashors": _wmean(df["total_nashors"], weights),
        "avg_total_inhibitors": _wmean(df["total_inhibitors"], weights),
        "avg_game_time_min": _wmean(df["game_time_min"], weights),
        # extra helpful breakdowns
        "avg_kills_for": _wmean(df["kills_for"], weights),
        "avg_kills_against": _wmean(df["kills_against"], weights),
    }

    # small recent W/L sequence for debugging
    last_results = df.tail(15)["win"].fillna(0).astype(int).to_list()
    out["recent_wl_15"] = "".join("W" if x == 1 else "L" for x in last_results)

    return out


def compare_two_teams(
    team_games: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    filters: Filters,
    half_life_days: float = 180.0,
) -> Dict[str, Dict[str, Any]]:
    """Return summaries for A and B in the four buckets: ALL, last5, last10, last15."""
    filtered = apply_filters(team_games, filters)

    def one(team: str) -> Dict[str, Dict[str, Any]]:
        return {
            "all": summarize_team(filtered, team, mode="all", half_life_days=half_life_days),
            "last5": summarize_team(filtered, team, mode="last", last_n=5, half_life_days=half_life_days),
            "last10": summarize_team(filtered, team, mode="last", last_n=10, half_life_days=half_life_days),
            "last15": summarize_team(filtered, team, mode="last", last_n=15, half_life_days=half_life_days),
        }

    return {"A": one(team_a), "B": one(team_b)}


def _pick_team_league(df: pd.DataFrame, team: str) -> Optional[str]:
    """Pick the most common league for a team inside a pre-filtered dataframe."""
    if "league" not in df.columns:
        return None
    tdf = df[df["team"] == team]
    if tdf.empty:
        return None
    vc = tdf["league"].fillna("").value_counts()
    if vc.empty:
        return None
    league = str(vc.index[0]).strip()
    return league or None


def compare_two_teams_league_mode(
    team_games: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    filters: Filters,
    league_mode: Literal["all", "fixed", "auto"] = "all",
    half_life_days: float = 180.0,
) -> Dict[str, Dict[str, Any]]:
    """Like compare_two_teams, but allows per-team league filtering.

    - league_mode='all': ignore filters.league (no league filtering)
    - league_mode='fixed': apply filters.league to BOTH teams (legacy behavior)
    - league_mode='auto': pick each team's most common league inside the base filtered set

    Notes:
      Split/playoffs/year filters are still shared.
    """
    # Base filters are shared, but league may become per-team
    base = Filters(year=filters.year, league=None, split=filters.split, playoffs=filters.playoffs)
    base_df = apply_filters(team_games, base)

    if league_mode == "fixed":
        # legacy: just apply all filters (including league) to both
        return compare_two_teams(team_games, team_a, team_b, filters=filters, half_life_days=half_life_days)

    def _subset_for(team: str) -> pd.DataFrame:
        df = base_df[base_df["team"] == team].copy()
        if league_mode == "auto":
            lg = _pick_team_league(base_df, team)
            if lg:
                df = df[df["league"].fillna("") == lg].copy()
        # league_mode == 'all' -> no league restriction
        return df.sort_values("date")

    def _summarize(df_team: pd.DataFrame, team: str) -> Dict[str, Dict[str, Any]]:
        # Use the same logic as summarize_team, but df is already filtered to that team.
        def _bucket_all() -> Dict[str, Any]:
            w = recency_weights(df_team["date"], half_life_days=half_life_days, ref_date=None)
            return {
                "team": team,
                "games": len(df_team),
                "win_rate": _wmean(df_team["win"], w),
                "avg_total_kills": _wmean(df_team["total_kills"], w),
                "avg_total_towers": _wmean(df_team["total_towers"], w),
                "avg_total_dragons": _wmean(df_team["total_dragons"], w),
                "avg_total_nashors": _wmean(df_team["total_nashors"], w),
                "avg_total_inhibitors": _wmean(df_team["total_inhibitors"], w),
                "avg_game_time_min": _wmean(df_team["game_time_min"], w),
                "avg_kills_for": _wmean(df_team["kills_for"], w),
                "avg_kills_against": _wmean(df_team["kills_against"], w),
                "recent_wl_15": "".join(
                    "W" if x == 1 else "L" for x in df_team.tail(15)["win"].fillna(0).astype(int).to_list()
                ),
            }

        def _bucket_last(n: int) -> Dict[str, Any]:
            tail = df_team.tail(int(n)).copy()
            w = np.ones(len(tail), dtype=float)
            return {
                "team": team,
                "games": len(tail),
                "win_rate": _wmean(tail["win"], w),
                "avg_total_kills": _wmean(tail["total_kills"], w),
                "avg_total_towers": _wmean(tail["total_towers"], w),
                "avg_total_dragons": _wmean(tail["total_dragons"], w),
                "avg_total_nashors": _wmean(tail["total_nashors"], w),
                "avg_total_inhibitors": _wmean(tail["total_inhibitors"], w),
                "avg_game_time_min": _wmean(tail["game_time_min"], w),
                "avg_kills_for": _wmean(tail["kills_for"], w),
                "avg_kills_against": _wmean(tail["kills_against"], w),
                "recent_wl_15": "".join(
                    "W" if x == 1 else "L" for x in tail.tail(15)["win"].fillna(0).astype(int).to_list()
                ),
            }

        return {
            "all": _bucket_all(),
            "last5": _bucket_last(5),
            "last10": _bucket_last(10),
            "last15": _bucket_last(15),
        }

    return {
        "A": _summarize(_subset_for(team_a), team_a),
        "B": _summarize(_subset_for(team_b), team_b),
    }
