from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import math
import pandas as pd

from .artifact import MLArtifact
from .fit_ml_params import _build_lineup_cache, _evaluate_walkforward, _make_samples
from .io_oracle import load_oracle_elixir_games
from .players_layer import PlayersArtifact, load_match_player_rows


def _load_games(csv_paths: List[str]) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for p in csv_paths:
        d = load_oracle_elixir_games(str(p), keep_playoffs=None)
        if d is not None and not d.empty:
            chunks.append(d)
    if not chunks:
        return pd.DataFrame()
    g = pd.concat(chunks, ignore_index=True)
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).copy()
    g = g.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last").reset_index(drop=True)
    return g


def run_layer_validation(
    *,
    artifact_path: str,
    players_artifact_path: str,
    csv_paths: List[str],
    min_train_games: int = 1500,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    min_league_games: int = 400,
) -> Dict[str, Any]:
    g = _load_games(csv_paths)
    if g.empty:
        return {"error": "no_games"}

    art = MLArtifact.load_json(str(artifact_path))
    p_art = PlayersArtifact.load_json(str(players_artifact_path))
    w_lineup = float((((art.meta or {}).get("players_layer") or {}).get("w_lineup") or 0.85))

    gameids = set(g["gameid"].astype(str).tolist())
    pr = load_match_player_rows(csv_paths, keep_playoffs=None, gameids=gameids)
    lineup_cache = _build_lineup_cache(pr)
    samples = _make_samples(
        art=art,
        p_art=p_art,
        games=g,
        lineup_cache=lineup_cache,
        w_lineup=w_lineup,
    )
    if not samples:
        return {"error": "no_samples"}

    min_train = int(max(50, int(min_train_games)))
    m_team = _evaluate_walkforward(samples, team_scale=1.0, players_scale=0.0, min_train_games=min_train)
    m_players = _evaluate_walkforward(samples, team_scale=0.0, players_scale=1.0, min_train_games=min_train)
    m_fused = _evaluate_walkforward(
        samples,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        min_train_games=min_train,
    )

    models = {
        "team_only": m_team,
        "players_only": m_players,
        "fused_active": m_fused,
    }
    base_ll = m_team.get("logloss")
    if base_ll is not None and math.isfinite(float(base_ll)):
        for _key, row in models.items():
            ll = row.get("logloss")
            if ll is None:
                continue
            row["gain_vs_team_logloss"] = float(base_ll) - float(ll)

    leagues = sorted({str(s.league) for s in samples if str(s.league).strip()})
    by_league: Dict[str, Any] = {}
    for lg in leagues:
        rep_team = _evaluate_walkforward(
            samples,
            team_scale=1.0,
            players_scale=0.0,
            min_train_games=min_train,
            league_filter=lg,
        )
        n_lg = int(rep_team.get("n_total") or 0)
        if n_lg < int(min_league_games):
            continue
        rep_players = _evaluate_walkforward(
            samples,
            team_scale=0.0,
            players_scale=1.0,
            min_train_games=min_train,
            league_filter=lg,
        )
        rep_fused = _evaluate_walkforward(
            samples,
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            min_train_games=min_train,
            league_filter=lg,
        )
        ll_team = rep_team.get("logloss")
        ll_fused = rep_fused.get("logloss")
        if ll_team is not None and ll_fused is not None:
            rep_fused["gain_vs_team_logloss"] = float(ll_team) - float(ll_fused)
        by_league[str(lg)] = {
            "team_only": rep_team,
            "players_only": rep_players,
            "fused_active": rep_fused,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "w_lineup_artifact": float(w_lineup),
        "active_scales": {"team_scale": float(team_scale), "players_scale": float(players_scale)},
        "models": models,
        "by_league": by_league,
    }
