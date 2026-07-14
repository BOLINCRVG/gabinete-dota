"""Derive a regional split for EWC games.

EWC (Esports World Cup) rows in the OracleElixir CSV come with `split` empty,
so the league appears as one undifferentiated block in the UI. Users want to
slice it by region (KR, EU, CN, NA, ...). We derive each team's home region
from the OTHER leagues it appears in inside the same dataset and stamp that
region as the EWC split.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd


_LEAGUE_TO_REGION: Mapping[str, str] = {
    # Majors
    "LCK": "KR", "LCKC": "KR",
    "LEC": "EU", "EM": "EU", "LFL": "EU", "PRM": "EU", "NLC": "EU",
    "LIT": "EU", "LVP": "EU", "EBL": "EU", "HLL": "EU", "HM": "EU",
    "LES": "EU", "ROL": "EU", "TCL": "EU",
    "LPL": "CN", "LDL": "CN",
    "LCS": "NA", "NACL": "NA",
    "PCS": "PCS",
    "VCS": "VCS",
    "LJL": "JP",
    "LLA": "LATAM", "LAS": "LATAM",
    "CBLOL": "BR", "LRS": "BR", "LRN": "BR",
    "LCP": "APAC",
    "CCWS": "EU",
    "LPLOL": "EU",
}

# Leagues that should NOT count as "home" for the region inference,
# because they're international/aggregated events.
_INTERNATIONAL_LEAGUES = frozenset({"EWC", "MSI", "WORLDS", "FST"})


def _team_home_region_map(df: pd.DataFrame) -> dict[str, str]:
    """Build {team_canon -> region} by counting non-international league rows."""
    if df is None or df.empty:
        return {}
    if "league" not in df.columns or "team" not in df.columns:
        return {}
    sub = df[~df["league"].isin(_INTERNATIONAL_LEAGUES)].copy()
    if sub.empty:
        return {}
    sub["__region"] = sub["league"].map(_LEAGUE_TO_REGION).fillna("")
    sub = sub[sub["__region"] != ""]
    if sub.empty:
        return {}
    # For each team, pick the most frequent region.
    counts = sub.groupby(["team", "__region"]).size().reset_index(name="n")
    counts = counts.sort_values(["team", "n"], ascending=[True, False])
    return {str(t): str(r) for t, r in counts.drop_duplicates("team")[["team", "__region"]].values}


def annotate_ewc_region_split(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `split` for EWC rows with the home region of each team.

    Idempotent: if a row already has a non-empty split, it's left untouched.
    Returns the same DataFrame (mutated in place) for chaining ergonomics.
    """
    if df is None or df.empty:
        return df
    if "league" not in df.columns or "team" not in df.columns:
        return df
    ewc_mask = df["league"].astype(str).str.upper() == "EWC"
    if not bool(ewc_mask.any()):
        return df
    if "split" not in df.columns:
        df["split"] = ""
    region_map = _team_home_region_map(df)
    if not region_map:
        return df

    def _resolve(row: pd.Series) -> str:
        cur = row.get("split")
        if isinstance(cur, str) and cur.strip():
            return cur
        if pd.notna(cur) and str(cur).strip():
            return str(cur)
        return region_map.get(str(row.get("team", "")), "Other")

    df.loc[ewc_mask, "split"] = df.loc[ewc_mask].apply(_resolve, axis=1)
    return df


__all__ = ["annotate_ewc_region_split"]
