"""Helpers for resolving matchup filter contexts, including previous-season fallback."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from mlcore.stats_teams import Filters


def resolve_prev_season_with_fallback(
    prev_team_games: pd.DataFrame | None,
    *,
    filters: Filters,
    team_a: str,
    team_b: str,
    prev_season_year: int,
    league_mode: str,
    map_mode: str,
    filter_team_games_fn: Callable[..., pd.DataFrame],
) -> tuple[Filters | None, pd.DataFrame, pd.DataFrame, str]:
    """Return (active_filters, dfA, dfB, note) for the previous-season context window.

    Falls back to the full year (split=None, playoffs=None) when the split/playoffs
    combination produces empty DataFrames for either team.
    Returns (None, empty, empty, "") when prev_team_games is unavailable.
    """
    empty = pd.DataFrame()
    if (
        not isinstance(prev_team_games, pd.DataFrame)
        or prev_team_games.empty
        or not str(team_a).strip()
        or not str(team_b).strip()
    ):
        return None, empty, empty, ""

    year = int(prev_season_year) if prev_season_year > 0 else None
    selected = Filters(
        year=year,
        league=filters.league,
        split=filters.split,
        playoffs=filters.playoffs,
    )
    active = selected

    dfA = filter_team_games_fn(
        prev_team_games, team=team_a, filters=active, league_mode=league_mode, map_mode=map_mode, max_games=None
    )
    dfB = filter_team_games_fn(
        prev_team_games, team=team_b, filters=active, league_mode=league_mode, map_mode=map_mode, max_games=None
    )

    note = ""
    if (dfA.empty or dfB.empty) and (filters.split is not None or filters.playoffs is not None):
        active = Filters(year=year, league=filters.league, split=None, playoffs=None)
        dfA = filter_team_games_fn(
            prev_team_games, team=team_a, filters=active, league_mode=league_mode, map_mode=map_mode, max_games=None
        )
        dfB = filter_team_games_fn(
            prev_team_games, team=team_b, filters=active, league_mode=league_mode, map_mode=map_mode, max_games=None
        )
        if not dfA.empty and not dfB.empty:
            note = "Sem amostra suficiente no S15 com os filtros atuais; usando o ano anterior inteiro como fallback."

    return active, dfA, dfB, note
