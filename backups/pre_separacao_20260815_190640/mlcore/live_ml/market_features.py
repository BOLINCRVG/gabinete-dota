"""Feature extraction for live market regression models.

For each market (total_kills, total_towers, total_dragons, total_barons,
gamelength_min) we predict the FINAL value given the game state at minute t.

Features at minute t (per game, not per team — markets are game-level totals):
  - p_map_pre           # pre-game prediction (used as a weak prior for context)
  - z_pre               # logit of p_map_pre
  - gold_diff           # blue gold - red gold (signed; magnitude matters)
  - abs_gold_diff       # |gold_diff| — proxy for game "intensity"
  - xp_diff, cs_diff
  - kills_total, kills_diff
  - league_tier, bo, playoffs

The target is the final game total for that market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from mlcore.live_ml.features import SNAPSHOT_MINUTES
from mlcore.tiers import tier_for_league
from mlcore.utils import safe_float as _f


# Feature columns produced (order matters).
MARKET_FEATURE_COLS: Tuple[str, ...] = (
    "z_pre",
    "gold_diff",
    "abs_gold_diff",
    "xp_diff",
    "cs_diff",
    "kills_total",
    "kills_diff",
    "deaths_total",
    "assists_total",
    "league_tier",
    "bo",
    "playoffs",
)


# Supported markets and how to derive their target from the per-game team rows.
SUPPORTED_MARKETS: Tuple[str, ...] = (
    "total_kills",
    "total_towers",
    "total_dragons",
    "total_barons",
    "gamelength_min",
)


_TIER_INT: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "MINOR": 3, "ERL": 4, "OTHER": 5,
}


def _encode_tier(league: str) -> int:
    return _TIER_INT.get(tier_for_league(str(league or "")), 5)


def feature_names() -> Tuple[str, ...]:
    return MARKET_FEATURE_COLS


@dataclass(frozen=True)
class LiveMarketSample:
    """One game's feature vector + target for a specific market at minute t."""

    gameid: str
    minute: int
    market: str
    features: np.ndarray   # shape (len(MARKET_FEATURE_COLS),)
    target: float          # the final value of the market for this game


def _logit(p: float) -> float:
    import math
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return math.log(p / (1.0 - p))


def _resolve_market_target(blue_row: pd.Series, red_row: pd.Series, market: str) -> Optional[float]:
    """Compute the final value of `market` from the two team rows of one game."""
    if market == "total_kills":
        return float(_f(blue_row.get("teamkills"), 0.0)) + float(_f(red_row.get("teamkills"), 0.0))
    if market == "total_towers":
        return float(_f(blue_row.get("towers"), 0.0)) + float(_f(red_row.get("towers"), 0.0))
    if market == "total_dragons":
        return float(_f(blue_row.get("dragons"), 0.0)) + float(_f(red_row.get("dragons"), 0.0))
    if market == "total_barons":
        return float(_f(blue_row.get("barons"), 0.0)) + float(_f(red_row.get("barons"), 0.0))
    if market == "gamelength_min":
        # gamelength is in seconds in the Oracle CSV.
        gl = float(_f(blue_row.get("gamelength"), 0.0))
        return gl / 60.0 if gl > 0 else None
    return None


def build_market_features(
    team_rows: pd.DataFrame,
    *,
    minute: int,
    market: str,
    p_map_pre_by_gameid_blue: Mapping[str, float],
    require_complete: bool = True,
) -> List[LiveMarketSample]:
    """Build training samples for one (market, minute) pair.

    One sample per game. Features are computed from the BLUE side perspective
    (gold_diff = blue - red), because markets are symmetric per game.
    """
    t = int(minute)
    if t not in SNAPSHOT_MINUTES:
        raise ValueError(f"minute must be in {SNAPSHOT_MINUTES}")
    if market not in SUPPORTED_MARKETS:
        raise ValueError(f"market must be one of {SUPPORTED_MARKETS}")

    df = team_rows[team_rows["participantid"].isin([100, 200])].copy()
    needed = [f"goldat{t}", f"xpat{t}", f"csat{t}", f"killsat{t}",
              f"deathsat{t}", f"assistsat{t}",
              f"opp_goldat{t}", f"opp_xpat{t}", f"opp_csat{t}"]
    if require_complete:
        df = df.dropna(subset=needed)

    # Group by gameid so we get the blue/red pair
    out: List[LiveMarketSample] = []
    for gid, g in df.groupby("gameid", sort=False):
        gid_s = str(gid)
        if len(g) < 2:
            continue
        # Blue row = participantid 100
        blue_rows = g[g["participantid"] == 100]
        red_rows = g[g["participantid"] == 200]
        if blue_rows.empty or red_rows.empty:
            continue
        blue = blue_rows.iloc[0]
        red = red_rows.iloc[0]

        # Target
        target = _resolve_market_target(blue, red, market)
        if target is None or not (isinstance(target, (int, float)) and target > 0):
            continue

        # Pre-game prediction for blue
        p_pre = float(p_map_pre_by_gameid_blue.get(gid_s, 0.5))
        if not (0.0 < p_pre < 1.0):
            p_pre = 0.5

        # Features (blue's perspective)
        gold_d = float(_f(blue.get(f"goldat{t}"), 0.0)) - float(_f(blue.get(f"opp_goldat{t}"), 0.0))
        xp_d = float(_f(blue.get(f"xpat{t}"), 0.0)) - float(_f(blue.get(f"opp_xpat{t}"), 0.0))
        cs_d = float(_f(blue.get(f"csat{t}"), 0.0)) - float(_f(blue.get(f"opp_csat{t}"), 0.0))
        k_b = float(_f(blue.get(f"killsat{t}"), 0.0))
        k_r = float(_f(blue.get(f"opp_killsat{t}"), 0.0))
        d_b = float(_f(blue.get(f"deathsat{t}"), 0.0))
        d_r = float(_f(blue.get(f"opp_deathsat{t}"), 0.0))
        a_b = float(_f(blue.get(f"assistsat{t}"), 0.0))
        a_r = float(_f(blue.get(f"opp_assistsat{t}"), 0.0))
        league = str(blue.get("league", "") or "")
        bo_raw = blue.get("game", 1)
        bo = int(bo_raw) if str(bo_raw).strip().isdigit() else 1
        playoffs = 1 if str(blue.get("playoffs", "") or "").strip().lower() in {"1", "true", "playoffs"} else 0

        features = np.array([
            _logit(p_pre),
            gold_d, abs(gold_d),
            xp_d, cs_d,
            (k_b + k_r), (k_b - k_r),
            (d_b + d_r),
            (a_b + a_r),
            float(_encode_tier(league)),
            float(bo),
            float(playoffs),
        ], dtype=float)

        out.append(LiveMarketSample(
            gameid=gid_s, minute=t, market=market,
            features=features, target=float(target),
        ))

    return out
