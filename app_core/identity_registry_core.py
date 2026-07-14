"""Backward-compat re-export shim. Implementation lives in mlcore.identity_registry_core."""
from __future__ import annotations
from mlcore.identity_registry_core import (
    normalize_identity_key,
    empty_identity_registry,
    normalize_identity_registry,
    load_identity_registry,
    save_identity_registry,
    search_identity_records,
    resolve_team_identity_record,
    resolve_player_identity_record,
    team_identity_candidates,
    player_identity_candidates,
    team_identity_league_candidates,
    upsert_team_identity,
    upsert_player_identity,
)

__all__ = [
    "empty_identity_registry",
    "load_identity_registry",
    "normalize_identity_key",
    "normalize_identity_registry",
    "player_identity_candidates",
    "resolve_player_identity_record",
    "resolve_team_identity_record",
    "save_identity_registry",
    "search_identity_records",
    "team_identity_candidates",
    "team_identity_league_candidates",
    "upsert_player_identity",
    "upsert_team_identity",
]
