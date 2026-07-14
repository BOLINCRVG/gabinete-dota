"""Runtime helper for the Phase 8 first-X market models.

Lazy-loads per-market pre-game classifiers and exposes a high-level helper
that returns P(team_a wins the first-X market) using only pre-game info:
team_a / team_b / league / BO / playoffs + the pre-game p_map_pre.

Markets:
  - first_blood
  - first_dragon
  - first_tower
  - first_herald
  - first_baron
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import math

import numpy as np

from mlcore.live_ml import FIRST_MARKETS, FirstMarketModel, FirstMarketSample


# All five markets pass the gate, so all are active by default.
ACTIVE_FIRST_MARKETS: Tuple[str, ...] = tuple(FIRST_MARKETS)


def is_first_markets_enabled(session_state: Mapping[str, Any] | None) -> bool:
    """Share the same flag as the state model — if state is ON, first markets are too."""
    ss = session_state if isinstance(session_state, Mapping) else {}
    return bool(ss.get("ml_live_state_model_enabled", False))


def first_markets_dir_from_state(session_state: Mapping[str, Any] | None) -> Path:
    ss = session_state if isinstance(session_state, Mapping) else {}
    base = ss.get("ml_live_state_models_dir", "live_models") or "live_models"
    return Path(str(base)) / "first_markets"


@lru_cache(maxsize=16)
def _cached_load_first_market_model(markets_dir: str, market: str) -> Optional[FirstMarketModel]:
    """Lazy load with LRU cache. Returns None if model dir absent or load fails."""
    p = Path(str(markets_dir)) / str(market) / "pregame"
    if not p.exists():
        return None
    try:
        return FirstMarketModel.load(p)
    except Exception:
        return None


def clear_first_market_model_cache() -> None:
    _cached_load_first_market_model.cache_clear()


def available_first_markets(markets_dir: str | Path = "live_models/first_markets") -> Tuple[str, ...]:
    """Return list of markets with a saved pre-game model."""
    base = Path(str(markets_dir))
    out = []
    if not base.exists():
        return tuple()
    for market in FIRST_MARKETS:
        if (base / market / "pregame" / "metadata.json").exists():
            out.append(market)
    return tuple(out)


_TIER_INT_LOCAL: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "ERL": 3, "MINOR": 4, "OTHER": 5,
}


def _features_for_pregame(
    *,
    p_map_pre: float,
    league: str,
    bo: int,
    is_playoffs: bool,
) -> np.ndarray:
    """Build the canonical pre-game feature vector for first-X markets."""
    from mlcore.tiers import tier_for_league, league_strength_score
    p = max(1e-9, min(1.0 - 1e-9, float(p_map_pre)))
    z = math.log(p / (1.0 - p))
    tier_int = _TIER_INT_LOCAL.get(tier_for_league(str(league or "")), 5)
    return np.array([
        z,
        float(tier_int),
        float(league_strength_score(str(league or ""))),
        float(int(bo or 1)),
        1.0 if bool(is_playoffs) else 0.0,
    ], dtype=float)


def predict_first_market(
    *,
    session_state: Mapping[str, Any] | None,
    market: str,
    p_map_pre: float = 0.5,
    league: str = "",
    bo: int = 1,
    is_playoffs: bool = False,
    team_a_is_blue: bool = True,
) -> Dict[str, Any]:
    """Return {'p_team_a': float, 'p_blue': float, 'reason': str}.

    `team_a_is_blue` indicates which side of the (team_a, team_b) the user is
    asking about. Defaults to True (most UIs treat team_a as the focal side).
    """
    out: Dict[str, Any] = {
        "p_team_a": None, "p_blue": None, "market": str(market),
        "team_a_is_blue": bool(team_a_is_blue), "reason": "ok",
    }
    if not is_first_markets_enabled(session_state):
        out["reason"] = "disabled"
        return out
    if str(market) not in ACTIVE_FIRST_MARKETS:
        out["reason"] = "market_not_active"
        return out

    markets_dir = str(first_markets_dir_from_state(session_state))
    model = _cached_load_first_market_model(markets_dir, str(market))
    if model is None:
        out["reason"] = "model_not_available"
        return out

    feat = _features_for_pregame(
        p_map_pre=float(p_map_pre), league=str(league or ""),
        bo=int(bo or 1), is_playoffs=bool(is_playoffs),
    )
    sample = FirstMarketSample(gameid="runtime", market=str(market), features=feat, target=0)
    try:
        p_blue = float(model.predict_blue(sample))
    except Exception as exc:
        out["reason"] = f"predict_error:{type(exc).__name__}"
        return out

    out["p_blue"] = p_blue
    out["p_team_a"] = p_blue if team_a_is_blue else float(1.0 - p_blue)
    return out


__all__ = [
    "ACTIVE_FIRST_MARKETS",
    "available_first_markets",
    "clear_first_market_model_cache",
    "first_markets_dir_from_state",
    "is_first_markets_enabled",
    "predict_first_market",
]
