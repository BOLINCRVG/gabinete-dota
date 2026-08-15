"""Players panel feature-trace helpers extracted from plays_app."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict


def _dedupe(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def append_players_recent_stats_panel_trace(
    *,
    append_feature_trace_fn: Callable[..., Any],
    enabled: bool,
    teamA: Any,
    teamB: Any,
    team_filter: Any,
    lane_filter: Any,
    player_filter: Any,
    opp_filter: Any,
    rows_before: Any,
    rows_after: Any,
) -> bool:
    if not bool(enabled):
        return False
    append_feature_trace_fn(
        feature="players_recent_stats_panel_v1",
        event="shadow_render",
        enabled=True,
        payload={
            "matchup": {"teamA": str(teamA), "teamB": str(teamB)},
            "filters": {
                "team": str(team_filter),
                "lane": str(lane_filter),
                "player": str(player_filter),
                "opp": str(opp_filter),
            },
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
        },
        dedupe_signature=_dedupe(
            {
                "ta": str(teamA),
                "tb": str(teamB),
                "t": str(team_filter),
                "l": str(lane_filter),
                "p": str(player_filter),
                "o": str(opp_filter),
                "n0": int(rows_before),
                "n1": int(rows_after),
            }
        ),
    )
    return True


def append_players_fair_panel_trace(
    *,
    append_feature_trace_fn: Callable[..., Any],
    enabled: bool,
    teamA: Any,
    teamB: Any,
    team_filter: Any,
    lane_filter: Any,
    player_filter: Any,
    rows_after: Any,
    rows_total: Any,
) -> bool:
    if not bool(enabled):
        return False
    append_feature_trace_fn(
        feature="players_fair_panel_v1",
        event="shadow_render",
        enabled=True,
        payload={
            "matchup": {"teamA": str(teamA), "teamB": str(teamB)},
            "filters": {
                "team": str(team_filter),
                "lane": str(lane_filter),
                "player": str(player_filter),
            },
            "rows_after": int(rows_after),
            "rows_total": int(rows_total),
        },
        dedupe_signature=_dedupe(
            {
                "ta": str(teamA),
                "tb": str(teamB),
                "t": str(team_filter),
                "l": str(lane_filter),
                "p": str(player_filter),
                "n": int(rows_after),
                "nt": int(rows_total),
            }
        ),
    )
    return True


def append_players_markets_panel_trace(
    *,
    append_feature_trace_fn: Callable[..., Any],
    enabled: bool,
    teamA: Any,
    teamB: Any,
    team_filter: Any,
    lane_filter: Any,
    market_filter: Any,
    player_filter: Any,
    rows_after: Any,
    rows_total: Any,
    fair_rows_total: Any,
    min_h2h_games: Any,
) -> bool:
    if not bool(enabled):
        return False
    append_feature_trace_fn(
        feature="players_markets_panel_v1",
        event="shadow_render",
        enabled=True,
        payload={
            "matchup": {"teamA": str(teamA), "teamB": str(teamB)},
            "filters": {
                "team": str(team_filter),
                "lane": str(lane_filter),
                "market": str(market_filter),
                "player": str(player_filter),
            },
            "rows_after": int(rows_after),
            "rows_total": int(rows_total),
            "fair_rows_total": int(fair_rows_total),
            "min_h2h_games": int(min_h2h_games),
        },
        dedupe_signature=_dedupe(
            {
                "ta": str(teamA),
                "tb": str(teamB),
                "t": str(team_filter),
                "l": str(lane_filter),
                "m": str(market_filter),
                "p": str(player_filter),
                "n": int(rows_after),
                "nt": int(rows_total),
                "nh": int(min_h2h_games),
            }
        ),
    )
    return True


def append_players_panels_bundle_trace(
    *,
    append_feature_trace_fn: Callable[..., Any],
    enabled: bool,
    teamA: Any,
    teamB: Any,
    recent_rows_total: Any,
    market_rows_total: Any,
    fair_rows_total: Any,
) -> bool:
    if not bool(enabled):
        return False
    append_feature_trace_fn(
        feature="players_panels_bundle_v1",
        event="persist_after_players_render",
        enabled=True,
        payload={
            "matchup": {"teamA": str(teamA), "teamB": str(teamB)},
            "recent_rows_total": int(recent_rows_total),
            "market_rows_total": int(market_rows_total),
            "fair_rows_total": int(fair_rows_total),
        },
        dedupe_signature=_dedupe(
            {
                "ta": str(teamA),
                "tb": str(teamB),
                "r": int(recent_rows_total),
                "m": int(market_rows_total),
                "f": int(fair_rows_total),
            }
        ),
        persist_files=False,
    )
    return True
