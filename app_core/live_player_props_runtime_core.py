"""Runtime helper for Phase 10 player-prop models (v1 = kills).

Lazy-loads the saved player_kills model and exposes a high-level helper
that returns expected kills + P(over X) for a given (player, champion,
position, context). Predictions are pre-game only.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from mlcore.live_ml import PlayerPropModel, PlayerPropSample, PLAYER_PROP_MARKETS


ACTIVE_PLAYER_PROPS: Tuple[str, ...] = ("player_kills",)


def is_player_props_enabled(session_state: Mapping[str, Any] | None) -> bool:
    """Share the same flag as the state model — if state is ON, props are too."""
    ss = session_state if isinstance(session_state, Mapping) else {}
    return bool(ss.get("ml_live_state_model_enabled", False))


def player_props_dir_from_state(session_state: Mapping[str, Any] | None) -> Path:
    ss = session_state if isinstance(session_state, Mapping) else {}
    base = ss.get("ml_live_state_models_dir", "live_models") or "live_models"
    return Path(str(base)) / "player_props"


@lru_cache(maxsize=8)
def _cached_load_prop_model(props_dir: str, market: str) -> Optional[PlayerPropModel]:
    p = Path(str(props_dir)) / str(market) / "pregame"
    if not p.exists():
        return None
    try:
        return PlayerPropModel.load(p)
    except Exception:
        return None


def clear_player_prop_cache() -> None:
    _cached_load_prop_model.cache_clear()


def available_player_props(props_dir: str | Path = "live_models/player_props") -> Tuple[str, ...]:
    base = Path(str(props_dir))
    out = []
    if not base.exists():
        return tuple()
    for market in PLAYER_PROP_MARKETS:
        if (base / market / "pregame" / "metadata.json").exists():
            out.append(market)
    return tuple(out)


_TIER_INT: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "ERL": 3, "MINOR": 4, "OTHER": 5,
}
_POS_INT: Dict[str, int] = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}


def _logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return math.log(p / (1.0 - p))


def _features_for_player(
    *,
    player_kills_mean: float,
    player_kills_std: float,
    champ_kills_mean_pos: float,
    position: str,
    league: str,
    bo: int,
    is_playoffs: bool,
    p_map_pre_team: float,
) -> np.ndarray:
    from mlcore.tiers import tier_for_league
    tier_int = _TIER_INT.get(tier_for_league(str(league or "")), 5)
    pos_id = _POS_INT.get(str(position or "").lower(), 5)
    return np.array([
        float(player_kills_mean),
        float(player_kills_std),
        float(champ_kills_mean_pos),
        float(pos_id),
        float(tier_int),
        float(int(bo or 1)),
        1.0 if bool(is_playoffs) else 0.0,
        _logit(float(p_map_pre_team)),
    ], dtype=float)


def predict_player_kills(
    *,
    session_state: Mapping[str, Any] | None,
    player: str,
    champion: str,
    position: str,
    player_kills_mean: float,
    player_kills_std: float = 2.0,
    champ_kills_mean_pos: Optional[float] = None,
    league: str = "",
    bo: int = 1,
    is_playoffs: bool = False,
    p_map_pre_team: float = 0.5,
    line: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict player kills mean + P(over line). Returns:
        {'mean', 'p_over', 'sigma', 'market', 'reason'}.

    If `champ_kills_mean_pos` is None, falls back to the player's own rolling
    mean (less accurate but works without a champion baseline). Pass the actual
    champion baseline when available.
    """
    out: Dict[str, Any] = {
        "mean": None, "p_over": None, "sigma": None,
        "market": "player_kills", "line": line, "reason": "ok",
    }
    if not is_player_props_enabled(session_state):
        out["reason"] = "disabled"
        return out
    props_dir = str(player_props_dir_from_state(session_state))
    model = _cached_load_prop_model(props_dir, "player_kills")
    if model is None:
        out["reason"] = "model_not_available"
        return out

    champ_baseline = float(champ_kills_mean_pos) if champ_kills_mean_pos is not None else float(player_kills_mean)
    feat = _features_for_player(
        player_kills_mean=float(player_kills_mean),
        player_kills_std=float(player_kills_std),
        champ_kills_mean_pos=float(champ_baseline),
        position=str(position),
        league=str(league or ""),
        bo=int(bo or 1),
        is_playoffs=bool(is_playoffs),
        p_map_pre_team=float(p_map_pre_team),
    )
    sample = PlayerPropSample(
        gameid="runtime", player=str(player), champion=str(champion),
        position=str(position), market="player_kills",
        features=feat, target=0.0,
    )
    try:
        out["mean"] = float(model.predict_mean(sample))
        out["sigma"] = float(model.residual_sigma)
        if line is not None:
            out["p_over"] = float(model.prob_over(sample, float(line)))
    except Exception as exc:
        out["reason"] = f"predict_error:{type(exc).__name__}"
        return out
    return out


__all__ = [
    "ACTIVE_PLAYER_PROPS",
    "available_player_props",
    "clear_player_prop_cache",
    "is_player_props_enabled",
    "predict_player_kills",
]
