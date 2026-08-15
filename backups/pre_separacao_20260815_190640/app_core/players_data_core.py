"""Core helpers for Players data loading/filtering/matching."""

from __future__ import annotations

from typing import Any, Callable
import difflib
import re

import pandas as pd


CHAMPION_ALIASES: dict[str, str] = {
    "monkeyking": "Wukong",
}


def load_player_rows(
    *,
    csv_path: str,
    canon_league_display_label_fn: Callable[[Any], str] | None = None,
    norm_pos_ml_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Load OracleElixir player rows (participantid 1..10) with normalized columns."""
    usecols = [
        "gameid",
        "date",
        "league",
        "year",
        "split",
        "playoffs",
        "game",
        "patch",
        "participantid",
        "side",
        "position",
        "playerid",
        "playername",
        "teamname",
        "champion",
        "result",
        "kills",
        "deaths",
        "assists",
        "gamelength",
        "teamkills",
        "teamdeaths",
        "ckpm",
        "total cs",
        "cspm",
        "damagetochampions",
        "visionscore",
    ]
    try:
        try:
            df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        except ValueError as exc:
            msg = str(exc)
            if "Usecols do not match columns" in msg and "playerid" in msg:
                usecols2 = [c for c in usecols if c != "playerid"]
                df = pd.read_csv(csv_path, usecols=usecols2, low_memory=False)
            else:
                raise
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)
        keep = [c for c in usecols if c in df.columns]
        df = df[keep].copy()

    pid = pd.to_numeric(df.get("participantid"), errors="coerce")
    df = df.loc[pid.between(1, 10)].copy()

    for c in ["league", "split", "position", "side", "playerid", "playername", "teamname", "champion"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    try:
        if "league" in df.columns and callable(canon_league_display_label_fn):
            df["league"] = df["league"].map(canon_league_display_label_fn)
    except Exception:
        pass

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    if "teamname" in df.columns:
        df["team"] = df["teamname"]
    if "playername" in df.columns:
        df["player"] = df["playername"]
    if "champion" in df.columns:
        df["champ"] = df["champion"]
    if "position" in df.columns:
        df["pos"] = df["position"]

    if "playerid" in df.columns:
        df["playerid"] = df["playerid"].fillna("").astype(str).str.strip()
    else:
        df["playerid"] = ""

    df["player_key"] = df["playerid"]
    df.loc[df["player_key"].astype(str).str.strip().eq(""), "player_key"] = df.get("player", "").astype(str)

    if "pos" in df.columns and callable(norm_pos_ml_fn):
        df["pos_ml"] = df["pos"].apply(norm_pos_ml_fn)
    else:
        df["pos_ml"] = ""

    return df


def apply_filters_players(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    out = df
    if filters is None:
        return out
    if getattr(filters, "year", None) is not None and "year" in out.columns:
        out = out[pd.to_numeric(out["year"], errors="coerce") == int(filters.year)]
    if getattr(filters, "league", None) is not None and "league" in out.columns:
        out = out[out["league"].astype(str) == str(filters.league)]
    if getattr(filters, "split", None) is not None and "split" in out.columns:
        out = out[out["split"].astype(str) == str(filters.split)]
    if getattr(filters, "playoffs", None) is not None and "playoffs" in out.columns:
        target = 1 if bool(filters.playoffs) else 0
        po = pd.to_numeric(out["playoffs"], errors="coerce").fillna(0).astype(int)
        out = out[(po > 0).astype(int) == target]
    return out


def champion_vocab(df_players: pd.DataFrame) -> list[str]:
    try:
        return sorted([c for c in df_players["champ"].fillna("").astype(str).unique().tolist() if c])
    except Exception:
        return []


def fuzzy_match_one(query: str, choices: list[str], cutoff: float = 0.65) -> tuple[str | None, list[str]]:
    q = str(query or "").strip()
    if not q:
        return None, []
    q_norm = re.sub(r"[^A-Za-z0-9]", "", q).lower()
    if not q_norm:
        return None, []
    alias_target = CHAMPION_ALIASES.get(q_norm)

    norm_map: dict[str, str] = {}
    for ch in choices:
        key = re.sub(r"[^A-Za-z0-9]", "", str(ch)).lower()
        if key and key not in norm_map:
            norm_map[key] = ch

    if alias_target:
        alias_key = re.sub(r"[^A-Za-z0-9]", "", str(alias_target)).lower()
        if alias_key in norm_map:
            return norm_map[alias_key], []

    if q_norm in norm_map:
        return norm_map[q_norm], []

    cand = difflib.get_close_matches(q_norm, list(norm_map.keys()), n=5, cutoff=cutoff)
    sug = [norm_map[x] for x in cand]
    return (sug[0] if sug else None), sug
