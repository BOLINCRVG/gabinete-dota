"""League-transfer builder — orchestrator.

Thin entry-point that wires the loader and profile layers together.
All heavy logic lives in league_transfer_loader_core and
league_transfer_profiles_core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from mlcore.identity_registry_core import normalize_identity_key
from mlcore.league_transfer_loader_core import (
    TEAM_REQUIRED_COLUMNS,
    PLAYER_REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    PLAYER_NUMERIC_COLUMNS,
    ROLE_CANON,
    _now_iso,
    load_oracle_team_rows,
    load_oracle_player_rows,
    infer_team_domestic_league_by_year,
    infer_team_meta_latest,
    _attach_domestic_context,
    _attach_identity,
    _attach_player_identity,
    _attach_domestic_context_player_rows,
)
from mlcore.league_transfer_profiles_core import (
    _build_team_and_league_profiles,
    _build_player_profiles,
    _build_lineup_profiles,
    _build_interleague_pair_profiles,
)

__all__ = [
    "NUMERIC_COLUMNS",
    "PLAYER_NUMERIC_COLUMNS",
    "PLAYER_REQUIRED_COLUMNS",
    "ROLE_CANON",
    "TEAM_REQUIRED_COLUMNS",
    "build_league_transfer_features_from_csvs",
    "build_league_transfer_features_from_team_rows",
    "infer_team_domestic_league_by_year",
    "infer_team_meta_latest",
    "load_oracle_player_rows",
    "load_oracle_team_rows",
]


def build_league_transfer_features_from_team_rows(
    team_rows: pd.DataFrame,
    *,
    player_rows: pd.DataFrame | None = None,
    identity_registry: Mapping[str, Any] | None = None,
    source_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    if team_rows is None or team_rows.empty:
        return {
            "version": "league_transfer_features_v1",
            "generated_at": _now_iso(),
            "meta": {"source_files": list(source_labels or []), "n_team_rows": 0},
            "by_league": {},
            "by_team": {},
            "by_player": {},
            "by_lineup": {},
            "by_pair": {},
            "team_alias_index": {},
            "player_alias_index": {},
        }

    rows = _attach_domestic_context(team_rows)
    rows = _attach_identity(rows, identity_registry)
    team_profiles, league_profiles, domestic = _build_team_and_league_profiles(rows)
    domestic_by_year = infer_team_domestic_league_by_year(rows)
    team_meta_latest = infer_team_meta_latest(domestic_by_year)
    player_profiles: dict[str, Any] = {}
    lineup_profiles: dict[str, Any] = {}
    player_alias_index: dict[str, str] = {}
    player_rows_ready = pd.DataFrame()
    if isinstance(player_rows, pd.DataFrame) and not player_rows.empty:
        player_rows_ready = _attach_domestic_context_player_rows(
            player_rows,
            domestic_by_year=domestic_by_year,
            team_meta_latest=team_meta_latest,
        )
        player_rows_ready = _attach_identity(player_rows_ready, identity_registry)
        player_rows_ready = _attach_player_identity(player_rows_ready, identity_registry)
        player_profiles, player_alias_index, domestic_players = _build_player_profiles(
            player_rows_ready,
            league_profiles=league_profiles,
        )
        lineup_profiles = _build_lineup_profiles(
            domestic_players,
            by_player=player_profiles,
        )
    pair_profiles = _build_interleague_pair_profiles(
        rows,
        team_profiles=team_profiles,
        league_profiles=league_profiles,
    )
    identity_map = rows.attrs.get("identity_team_map") or {}
    by_team: dict[str, Any] = {}
    alias_index: dict[str, str] = {}
    for entity_key, row in team_profiles.items():
        entity_rows = rows[rows["team_entity"].astype(str) == str(entity_key)]
        aliases: list[str] = []
        seen: set[str] = set()
        latest_name = str(row.get("team_current_name") or "")
        canonical_name = str(row.get("team_canonical_name") or latest_name or entity_key)
        for raw_team in entity_rows.get("teamname", pd.Series(dtype=str)).dropna().astype(str).unique().tolist():
            meta = identity_map.get(str(raw_team)) or {}
            for alias in list(meta.get("aliases") or [raw_team]):
                text = str(alias or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    aliases.append(text)
        if canonical_name and canonical_name not in seen:
            aliases.append(canonical_name)
            seen.add(canonical_name)
        if latest_name and latest_name not in seen:
            aliases.append(latest_name)
            seen.add(latest_name)
        for alias in aliases:
            alias_index[normalize_identity_key(alias)] = str(entity_key)
        by_team[str(entity_key)] = {
            "entity_key": str(entity_key),
            "canonical_name": canonical_name,
            "current_name": latest_name or canonical_name,
            "aliases": aliases,
            "relation_type": str((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("relation_type") or "alias"),
            "identity_confidence": str((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("identity_confidence") or "medium"),
            "current_leagues": list((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("current_leagues") or []),
            "kept_players": list((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("kept_players") or []),
            "added_players": list((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("added_players") or []),
            "removed_players": list((identity_map.get(str(latest_name)) or identity_map.get(str(canonical_name)) or {}).get("removed_players") or []),
            **dict(row),
        }

    years = sorted({int(v) for v in pd.to_numeric(rows.get("year"), errors="coerce").dropna().astype(int).tolist()})
    return {
        "version": "league_transfer_features_v1",
        "generated_at": _now_iso(),
        "meta": {
            "source_files": [str(v) for v in list(source_labels or [])],
            "years": years,
            "n_team_rows": len(rows),
            "n_domestic_team_rows": len(domestic),
            "n_player_rows": len(player_rows_ready),
            "n_leagues": len(league_profiles),
            "n_teams": len(by_team),
            "n_players": len(player_profiles),
            "n_lineups": len(lineup_profiles),
        },
        "by_league": league_profiles,
        "by_team": by_team,
        "by_player": player_profiles,
        "by_lineup": lineup_profiles,
        "by_pair": pair_profiles,
        "team_alias_index": alias_index,
        "player_alias_index": player_alias_index,
    }


def build_league_transfer_features_from_csvs(
    csv_paths: Iterable[str | Path],
    *,
    identity_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in csv_paths if Path(path).exists()]
    team_rows = load_oracle_team_rows(paths)
    player_rows = load_oracle_player_rows(paths)
    return build_league_transfer_features_from_team_rows(
        team_rows,
        player_rows=player_rows,
        identity_registry=identity_registry,
        source_labels=[str(path) for path in paths],
    )
