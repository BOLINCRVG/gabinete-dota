# -*- coding: utf-8 -*-
"""Páginas de navegação de topo (Live-Bet e Radar) do plays_app.

Fatia 3 do fatiamento do plays_app (F3, 2026-07-18): blocos movidos VERBATIM.
Cada página monta seu contexto e encerra o script (st.stop) — mesmo
comportamento do bloco original. As funções cacheadas do host (load de CSV,
players etc.) entram por parâmetro; o resto importa dos módulos-irmãos.
"""
from __future__ import annotations

from typing import Any, Callable, MutableMapping, Sequence

import streamlit as st

from mlcore.stats_teams import Filters, apply_filters

from app_core.live_bet_tab_core import render_live_bet_tab
from app_core.players_data_core import (
    apply_filters_players,
    fuzzy_match_one,
)
from app_core.radar_tab_core import render_radar_tab
from app_core.results_runtime_state_core import resolve_results_runtime_settings
from app_core.ui_cache_core import cached_champion_vocab


def render_live_bet_page(
    session_state: MutableMapping[str, Any],
    *,
    season_year: int,
    teams: Sequence[str],
    csv_sig: Any,
    artifact_path: Any,
    load_team_games_fn: Callable[[Any], Any],
    load_player_rows_fn: Callable[[Any], Any],
    state_get_str_fn: Callable[..., str],
    format_num_fn: Callable[..., str],
    min_to_mmss_fn: Callable[..., str],
) -> None:
    team_games = load_team_games_fn(csv_sig)
    years = sorted([int(y) for y in team_games["year"].dropna().unique().tolist()]) if team_games is not None and not team_games.empty else [int(season_year)]
    _results_runtime_live = resolve_results_runtime_settings(
        session_state,
        season_year=season_year,
        years=years,
    )
    _live_year_opt = _results_runtime_live["year_opt"]
    _live_split_opt = _results_runtime_live["split_opt"]
    _live_playoffs_opt = _results_runtime_live["playoffs_opt"]
    _live_fixed_league = _results_runtime_live["fixed_league"]
    _live_filters = Filters(
        year=None if _live_year_opt == "All" else int(_live_year_opt),
        league=None,
        split=None if _live_split_opt == "All" else str(_live_split_opt),
        playoffs=None if _live_playoffs_opt == "all" else (True if _live_playoffs_opt == "true" else False),
    )
    if _results_runtime_live["league_mode"] == "fixed" and _live_fixed_league and _live_fixed_league != "All":
        _live_filters = Filters(
            year=_live_filters.year,
            league=_live_fixed_league,
            split=_live_filters.split,
            playoffs=_live_filters.playoffs,
        )

    _team_a_live = str(session_state.get("teamA") or (teams[0] if teams else ""))
    _team_b_live = str(session_state.get("teamB") or "")
    if not _team_b_live or _team_b_live == _team_a_live:
        _team_b_live = next((t for t in teams if str(t) != _team_a_live), _team_a_live)

    render_live_bet_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "Filters": Filters,
            "active_filters": _live_filters,
            "apply_filters_fn": apply_filters,
            "teamA": _team_a_live,
            "teamB": _team_b_live,
            "csv_sig": csv_sig,
            "state_get_str_fn": state_get_str_fn,
            "load_player_rows_fn": load_player_rows_fn,
            "apply_filters_players_fn": apply_filters_players,
            "champion_vocab_fn": cached_champion_vocab,
            "fuzzy_match_one_fn": fuzzy_match_one,
            "format_num_fn": format_num_fn,
            "min_to_mmss_fn": min_to_mmss_fn,
            "artifact_path": str(artifact_path),
        },
    )
    st.stop()


def render_radar_page(
    *,
    csv_sig: Any,
    artifact_path: Any,
    load_team_games_fn: Callable[[Any], Any],
    load_player_rows_fn: Callable[[Any], Any],
    format_num_fn: Callable[..., str],
    min_to_mmss_fn: Callable[..., str],
) -> None:
    team_games = load_team_games_fn(csv_sig)
    render_radar_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "Filters": Filters,
            "apply_filters_fn": apply_filters,
            "csv_sig": csv_sig,
            "load_player_rows_fn": load_player_rows_fn,
            "format_num_fn": format_num_fn,
            "min_to_mmss_fn": min_to_mmss_fn,
            "artifact_path": str(artifact_path),
        },
    )
    st.stop()
