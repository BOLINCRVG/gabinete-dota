"""Live ML: state -> win-probability models for in-game predictions.

Pre-game produces a single p_map_pre. Live ML refines this prediction at
specific game-state snapshots (minutes 10, 15, 20, 25) using the actual
game state observed up to that point:

  P(blue wins | game_state_at_minute_t) = LightGBM(features_at_t + p_map_pre)

The model is trained on historical games where we know the actual outcome
and the full snapshot data. One model per snapshot time.

Modules
-------
- features: build feature rows from Oracle's Elixir team-row data
- state_models: StateToWinModel (one per snapshot time)
- fit: training pipeline
"""
from __future__ import annotations

from mlcore.live_ml.features import (
    SNAPSHOT_MINUTES,
    LiveSnapshot,
    build_snapshot_features,
)
from mlcore.live_ml.state_models import StateToWinModel
from mlcore.live_ml.market_features import (
    LiveMarketSample,
    MARKET_FEATURE_COLS,
    SUPPORTED_MARKETS,
    build_market_features,
)
from mlcore.live_ml.market_models import LiveMarketModel
from mlcore.live_ml.first_markets import (
    FIRST_MARKETS,
    FIRST_FEATURE_COLS,
    FirstMarketSample,
    FirstMarketModel,
    FirstMarketConfig,
    build_first_market_features,
)
from mlcore.live_ml.player_props import (
    PLAYER_PROP_MARKETS,
    PLAYER_PROP_FEATURE_COLS,
    PlayerPropSample,
    PlayerPropModel,
    PlayerPropConfig,
    build_player_kill_features,
)

__all__ = [
    "FIRST_FEATURE_COLS",
    "FIRST_MARKETS",
    "MARKET_FEATURE_COLS",
    "PLAYER_PROP_FEATURE_COLS",
    "PLAYER_PROP_MARKETS",
    "SNAPSHOT_MINUTES",
    "SUPPORTED_MARKETS",
    "FirstMarketConfig",
    "FirstMarketModel",
    "FirstMarketSample",
    "LiveMarketModel",
    "LiveMarketSample",
    "LiveSnapshot",
    "PlayerPropConfig",
    "PlayerPropModel",
    "PlayerPropSample",
    "StateToWinModel",
    "build_first_market_features",
    "build_market_features",
    "build_player_kill_features",
    "build_snapshot_features",
]
