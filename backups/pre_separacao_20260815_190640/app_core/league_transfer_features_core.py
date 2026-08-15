"""Backward-compat re-export shim. Implementation lives in mlcore.league_transfer_features_core."""
from __future__ import annotations
from mlcore.league_transfer_features_core import (
    empty_league_transfer_features,
    normalize_league_transfer_features,
    load_league_transfer_features,
    lookup_league_transfer_features,
    lookup_pair_transfer_features,
    lookup_team_transfer_features,
    lookup_player_transfer_features,
    lookup_lineup_transfer_features,
    build_interleague_transfer_adjustment,
    build_lineup_transfer_summary,
)

__all__ = [
    "build_interleague_transfer_adjustment",
    "build_lineup_transfer_summary",
    "empty_league_transfer_features",
    "load_league_transfer_features",
    "lookup_league_transfer_features",
    "lookup_lineup_transfer_features",
    "lookup_pair_transfer_features",
    "lookup_player_transfer_features",
    "lookup_team_transfer_features",
    "normalize_league_transfer_features",
]
