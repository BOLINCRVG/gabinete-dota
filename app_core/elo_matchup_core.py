# -*- coding: utf-8 -*-
"""ELO-based matchup probability combining team and player ratings."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from app_core.lineup_state import POS_CANON_ML as _POS_CANON_ML, safe_key as _safe_key
from app_core.players_elo_core import (
    compute_team_elo_season as _core_compute_team_elo_season,
)
from mlcore.stats_teams import Filters, apply_filters
from mlcore.utils import clip_prob
from app_core.ml_artifact_pricing_core import p_from_diff_points as _core_p_from_diff_points


def _p_from_diff_points(diff_points: float, scale_points: float = 400.0) -> float:
    return float(_core_p_from_diff_points(diff_points, scale_points=scale_points, clip_prob_fn=clip_prob))


def elo_season_players_p_map(
    *,
    team_games: pd.DataFrame,
    csv_path: str,
    teamA: str,
    teamB: str,
    filters: Filters,
    team_half_life_days: float,
    player_half_life_days: float,
    k_team: float,
    k_player: float,
    shrink_m: float,
    w_players: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
    ctx: Mapping[str, Any],
    st: Any,
) -> Dict[str, Any]:
    _file_sig = ctx.get("_file_sig")
    _compute_player_elo_global_cached = ctx.get("_compute_player_elo_global_cached")
    _discover_player_csv_paths = ctx.get("core_discover_player_csv_paths")

    # team elo season
    tinfo = _core_compute_team_elo_season(
        team_games=team_games,
        filters=filters,
        apply_filters_fn=apply_filters,
        half_life_days=float(team_half_life_days),
        k_team=float(k_team),
        shrink_m=float(shrink_m),
        base_rating=float(base_rating),
        scale=float(scale),
    )
    tr = tinfo.get("rating") or {}
    team_elo_A = float(tr.get(str(teamA), base_rating))
    team_elo_B = float(tr.get(str(teamB), base_rating))

    # players elo global
    paths = _discover_player_csv_paths(csv_path)
    sigs = tuple([_file_sig(p) for p in paths if p])
    pinfo = _compute_player_elo_global_cached(
        sigs,
        half_life_days=float(player_half_life_days),
        k_player=float(k_player),
        shrink_m=float(shrink_m),
        base_rating=float(base_rating),
        scale=float(scale),
    )
    pr = pinfo.get("rating") or {}
    name2id = pinfo.get("name2id") or {}
    id2name = pinfo.get("id2name") or {}

    def _lineup_keys(side: str) -> Dict[str, str]:
        out = {}
        for pos in _POS_CANON_ML:
            k = f"lineup_ml_{side}_{_safe_key(pos)}"
            out[pos] = str(st.session_state.get(k, "") or "").strip()
        return out

    def _lookup(pk: str) -> float:
        if not pk:
            return float("nan")
        if pk in pr:
            return float(pr[pk])
        if pk in name2id and name2id[pk] in pr:
            return float(pr[name2id[pk]])
        if pk in id2name and id2name[pk] in pr:
            return float(pr[id2name[pk]])
        return float("nan")

    def _avg_lineup(lineup: Dict[str, str]) -> Dict[str, Any]:
        vals = []
        rows = []
        for pos, pk in lineup.items():
            r = _lookup(pk)
            rows.append({"pos": pos, "player_key": pk, "elo": r})
            if math.isfinite(r):
                vals.append(float(r))
        avg = float(np.mean(vals)) if vals else float("nan")
        return {"avg": avg, "rows": rows, "found": len(vals), "total": len(lineup)}

    a_info = _avg_lineup(_lineup_keys("A"))
    b_info = _avg_lineup(_lineup_keys("B"))

    avgA = float(a_info["avg"]) if math.isfinite(float(a_info["avg"])) else float(base_rating)
    avgB = float(b_info["avg"]) if math.isfinite(float(b_info["avg"])) else float(base_rating)

    diff_team = team_elo_A - team_elo_B
    diff_players = avgA - avgB
    diff_total = float(diff_team) + float(w_players) * float(diff_players)

    p_raw = _p_from_diff_points(diff_total, scale_points=float(scale))
    return {
        "p_raw": float(p_raw),
        "p_cal": float(p_raw),
        "diff_points": float(diff_total),
        "team_elo_A": float(team_elo_A),
        "team_elo_B": float(team_elo_B),
        "avg_player_elo_A": float(avgA),
        "avg_player_elo_B": float(avgB),
        "diff_team": float(diff_team),
        "diff_players": float(diff_players),
        "lineup_A": a_info,
        "lineup_B": b_info,
        "team_games_A": int((tinfo.get("games") or {}).get(str(teamA), 0)),
        "team_games_B": int((tinfo.get("games") or {}).get(str(teamB), 0)),
    }
