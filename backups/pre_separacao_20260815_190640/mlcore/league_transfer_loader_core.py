"""League-transfer builder — loader layer.

Constants, CSV loaders, and domestic-context/identity helpers used by
both the profile builders and the main orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from mlcore.identity_registry_core import (
    empty_identity_registry,
    normalize_identity_key,
    player_identity_candidates,
    resolve_team_identity_record,
    resolve_player_identity_record,
    team_identity_candidates,
)
from mlcore.tiers import canonical_league, is_international_league
from mlcore.utils import safe_float as _safe_float_util


TEAM_REQUIRED_COLUMNS = [
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "patch",
    "participantid",
    "position",
    "teamname",
    "result",
    "gamelength",
    "kills",
    "deaths",
    "assists",
    "ckpm",
    "totalgold",
    "earnedgold",
    "towers",
    "opp_towers",
    "dragons",
    "opp_dragons",
    "barons",
    "opp_barons",
    "inhibitors",
    "opp_inhibitors",
    "firstblood",
    "firstdragon",
    "firstherald",
    "firstbaron",
    "firsttower",
    "firsttothreetowers",
    "goldat15",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
]

PLAYER_REQUIRED_COLUMNS = [
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "patch",
    "participantid",
    "position",
    "playername",
    "playerid",
    "teamname",
    "result",
    "gamelength",
    "kills",
    "deaths",
    "assists",
    "damagetochampions",
    "dpm",
    "damageshare",
    "visionscore",
    "vspm",
    "wpm",
    "wcpm",
    "controlwardsbought",
    "cspm",
    "earnedgoldshare",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "firstblood",
    "firsttower",
]

NUMERIC_COLUMNS = [
    "year",
    "participantid",
    "result",
    "gamelength",
    "kills",
    "deaths",
    "assists",
    "ckpm",
    "totalgold",
    "earnedgold",
    "towers",
    "opp_towers",
    "dragons",
    "opp_dragons",
    "barons",
    "opp_barons",
    "inhibitors",
    "opp_inhibitors",
    "firstblood",
    "firstdragon",
    "firstherald",
    "firstbaron",
    "firsttower",
    "firsttothreetowers",
    "goldat15",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
]

PLAYER_NUMERIC_COLUMNS = [
    "year",
    "participantid",
    "result",
    "gamelength",
    "kills",
    "deaths",
    "assists",
    "damagetochampions",
    "dpm",
    "damageshare",
    "visionscore",
    "vspm",
    "wpm",
    "wcpm",
    "controlwardsbought",
    "cspm",
    "earnedgoldshare",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "firstblood",
    "firsttower",
]

ROLE_CANON = ("Top", "Jungle", "Mid", "Bot", "Support")


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _f(value: Any, default: float = 0.0) -> float:
    return _safe_float_util(value, default)


def _weighted_mean(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    mask = values.notna() & weights.notna()
    if not bool(mask.any()):
        return float(default)
    v = pd.to_numeric(values.loc[mask], errors="coerce").astype(float)
    w = pd.to_numeric(weights.loc[mask], errors="coerce").astype(float)
    denom = float(w.sum())
    if denom <= 0:
        return float(default)
    return float((v * w).sum() / denom)


def _weighted_rate(mask: pd.Series, weights: pd.Series, *, default: float = 0.5) -> float:
    mm = mask.fillna(False).astype(bool)
    ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    denom = float(ww.sum())
    if denom <= 0:
        return float(default)
    return float(ww.loc[mm].sum() / denom)


def _shrink_prob(p: float, n: float, *, prior: float, k: float) -> float:
    nn = max(0.0, float(n))
    ww = nn / (nn + max(1.0, float(k)))
    return float((ww * float(p)) + ((1.0 - ww) * float(prior)))


def _existing_usecols(path: Path) -> list[str]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    return [col for col in TEAM_REQUIRED_COLUMNS if col in cols]


def _existing_player_usecols(path: Path) -> list[str]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    return [col for col in PLAYER_REQUIRED_COLUMNS if col in cols]


def _norm_pos(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"top", "t"}:
        return "Top"
    if text in {"jng", "jg", "jungle", "j"}:
        return "Jungle"
    if text in {"mid", "m"}:
        return "Mid"
    if text in {"bot", "adc", "bottom", "b"}:
        return "Bot"
    if text in {"sup", "support", "s"}:
        return "Support"
    return ""


def load_oracle_team_rows(csv_paths: Iterable[str | Path], *, chunksize: int = 200_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_like in csv_paths:
        path = Path(path_like)
        if not path.exists():
            continue
        usecols = _existing_usecols(path)
        if not usecols:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize), low_memory=False):
            pos = chunk.get("position")
            pos_series = pos.astype(str).str.strip().str.lower() if isinstance(pos, pd.Series) else pd.Series("", index=chunk.index)
            pid_series = pd.to_numeric(chunk.get("participantid"), errors="coerce") if "participantid" in chunk.columns else pd.Series(np.nan, index=chunk.index)
            mask = pos_series.eq("team") | pid_series.isin([100, 200])
            if bool(mask.any()):
                frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=TEAM_REQUIRED_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("league", "split", "patch", "position", "teamname", "gameid"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "league" in df.columns:
        df["league"] = df["league"].map(lambda v: str(canonical_league(str(v or "")) or str(v or "")).strip())
    return df.reset_index(drop=True)


def load_oracle_player_rows(csv_paths: Iterable[str | Path], *, chunksize: int = 200_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_like in csv_paths:
        path = Path(path_like)
        if not path.exists():
            continue
        usecols = _existing_player_usecols(path)
        if not usecols:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize), low_memory=False):
            pid_series = pd.to_numeric(chunk.get("participantid"), errors="coerce")
            pos_series = chunk.get("position")
            pos_series = pos_series.astype(str).str.strip().str.lower() if isinstance(pos_series, pd.Series) else pd.Series("", index=chunk.index)
            mask = pid_series.between(1, 10, inclusive="both") & pos_series.isin({p.lower() for p in ROLE_CANON} | {"adc", "bottom", "sup"})
            if bool(mask.any()):
                frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=PLAYER_REQUIRED_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in PLAYER_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("league", "split", "patch", "position", "playername", "playerid", "teamname", "gameid"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "league" in df.columns:
        df["league"] = df["league"].map(lambda v: str(canonical_league(str(v or "")) or str(v or "")).strip())
    df["pos_can"] = df.get("position", "").map(_norm_pos)
    df = df[df["pos_can"].isin(ROLE_CANON)].copy()
    return df.reset_index(drop=True)


def infer_team_domestic_league_by_year(team_rows: pd.DataFrame) -> dict[str, dict[int, str]]:
    if team_rows is None or team_rows.empty:
        return {}
    df = team_rows.copy()
    df["teamname"] = df.get("teamname", "").astype(str).str.strip()
    df["league"] = df.get("league", "").astype(str).str.strip()
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df = df[(df["teamname"] != "") & (df["league"] != "") & df["year"].notna()].copy()
    df = df[~df["league"].map(is_international_league)].copy()
    if df.empty:
        return {}
    counts = (
        df.groupby(["teamname", "year", "league"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["teamname", "year", "n", "league"], ascending=[True, True, False, True])
    )
    out: dict[str, dict[int, str]] = {}
    for row in counts.itertuples(index=False):
        team = str(getattr(row, "teamname", "") or "").strip()
        year = int(row.year)
        league = str(getattr(row, "league", "") or "").strip()
        if team and year not in (out.get(team) or {}):
            out.setdefault(team, {})[year] = league
    return out


def infer_team_meta_latest(domestic_by_year: Mapping[str, Mapping[int, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for team, mapping in (domestic_by_year or {}).items():
        if not mapping:
            continue
        years = [int(y) for y in mapping.keys()]
        latest = max(years)
        out[str(team)] = {
            "league": str((mapping or {}).get(latest) or "OTHER"),
            "year": int(latest),
        }
    return out


def _resolve_domestic_league(
    *,
    team: str,
    year: Any,
    match_league: Any,
    domestic_by_year: Mapping[str, Mapping[int, str]],
    team_meta_latest: Mapping[str, Mapping[str, Any]],
) -> str:
    name = str(team or "").strip()
    lg_match = str(canonical_league(str(match_league or "")) or str(match_league or "")).strip()
    yy = None
    try:
        yy = int(float(year))
    except Exception:
        yy = None
    if name and yy is not None:
        val = str(((domestic_by_year.get(name) or {}).get(yy)) or "").strip()
        if val:
            return val
    if lg_match and not is_international_league(lg_match):
        return lg_match
    latest = team_meta_latest.get(name) or {}
    return str(latest.get("league") or "OTHER")


def _attach_domestic_context(team_rows: pd.DataFrame) -> pd.DataFrame:
    df = team_rows.copy()
    domestic_by_year = infer_team_domestic_league_by_year(df)
    team_meta_latest = infer_team_meta_latest(domestic_by_year)
    resolved: list[str] = []
    for row in df.itertuples(index=False):
        resolved.append(
            _resolve_domestic_league(
                team=str(getattr(row, "teamname", "") or ""),
                year=getattr(row, "year", None),
                match_league=getattr(row, "league", ""),
                domestic_by_year=domestic_by_year,
                team_meta_latest=team_meta_latest,
            )
        )
    df["domestic_league"] = resolved
    return df


def _attach_identity(team_rows: pd.DataFrame, identity_registry: Mapping[str, Any] | None) -> pd.DataFrame:
    df = team_rows.copy()
    registry = identity_registry if isinstance(identity_registry, Mapping) else empty_identity_registry()
    team_map: dict[str, dict[str, Any]] = {}
    for team in sorted(df.get("teamname", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()):
        record = resolve_team_identity_record(registry, str(team))
        entity_key = str(record.get("entity_key") or normalize_identity_key(str(team)))
        aliases = team_identity_candidates(registry, str(team))
        current_name = str(record.get("current_name") or team)
        canonical_name = str(record.get("canonical_name") or current_name or team)
        current_leagues = list(record.get("current_leagues") or [])
        team_map[str(team)] = {
            "entity_key": entity_key,
            "aliases": aliases if aliases else [str(team)],
            "current_name": current_name,
            "canonical_name": canonical_name,
            "current_leagues": current_leagues,
            "relation_type": str(record.get("relation_type") or "alias"),
            "identity_confidence": str(record.get("confidence") or "medium"),
            "kept_players": list(record.get("kept_players") or []),
            "added_players": list(record.get("added_players") or []),
            "removed_players": list(record.get("removed_players") or []),
        }
    df["team_entity"] = df["teamname"].map(lambda v: str((team_map.get(str(v)) or {}).get("entity_key") or normalize_identity_key(str(v or ""))))
    df["team_current_name"] = df["teamname"].map(lambda v: str((team_map.get(str(v)) or {}).get("current_name") or str(v or "")))
    df["team_canonical_name"] = df["teamname"].map(lambda v: str((team_map.get(str(v)) or {}).get("canonical_name") or str(v or "")))
    df.attrs["identity_team_map"] = team_map
    return df


def _attach_player_identity(player_rows: pd.DataFrame, identity_registry: Mapping[str, Any] | None) -> pd.DataFrame:
    df = player_rows.copy()
    registry = identity_registry if isinstance(identity_registry, Mapping) else empty_identity_registry()
    player_map: dict[str, dict[str, Any]] = {}
    for player in sorted(df.get("playername", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()):
        record = resolve_player_identity_record(registry, str(player))
        entity_key = str(record.get("entity_key") or normalize_identity_key(str(player)))
        aliases = player_identity_candidates(registry, str(player))
        current_name = str(record.get("current_name") or player)
        canonical_name = str(record.get("canonical_name") or current_name or player)
        player_map[str(player)] = {
            "entity_key": entity_key,
            "aliases": aliases if aliases else [str(player)],
            "current_name": current_name,
            "canonical_name": canonical_name,
        }
    df["player_entity"] = df["playername"].map(lambda v: str((player_map.get(str(v)) or {}).get("entity_key") or normalize_identity_key(str(v or ""))))
    df["player_current_name"] = df["playername"].map(lambda v: str((player_map.get(str(v)) or {}).get("current_name") or str(v or "")))
    df["player_canonical_name"] = df["playername"].map(lambda v: str((player_map.get(str(v)) or {}).get("canonical_name") or str(v or "")))
    df.attrs["identity_player_map"] = player_map
    return df


def _attach_domestic_context_player_rows(
    player_rows: pd.DataFrame,
    *,
    domestic_by_year: Mapping[str, Mapping[int, str]],
    team_meta_latest: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    df = player_rows.copy()
    if "pos_can" not in df.columns:
        df["pos_can"] = df.get("position", "").map(_norm_pos)
    resolved: list[str] = []
    for row in df.itertuples(index=False):
        resolved.append(
            _resolve_domestic_league(
                team=str(getattr(row, "teamname", "") or ""),
                year=getattr(row, "year", None),
                match_league=getattr(row, "league", ""),
                domestic_by_year=domestic_by_year,
                team_meta_latest=team_meta_latest,
            )
        )
    df["domestic_league"] = resolved
    return df
