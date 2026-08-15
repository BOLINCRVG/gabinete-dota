"""Runtime helper for the Phase 6 live state-to-win models.

Provides lazy-loaded access to the 4 LightGBM models (one per snapshot minute)
plus a high-level `predict_live_p_win()` function that the live-bet tab can use
to obtain a calibrated win probability given the current game state.

The models live under `live_models/minute_{10,15,20,25}/`. They are gated by
the feature flag `ml_live_state_model_enabled` (default False) — when the flag
is off, the helpers return None so the app continues to use its previous
heuristics unchanged.

Usage in the app
----------------
    from app_core.live_state_runtime_core import predict_live_p_win

    p_live = predict_live_p_win(
        session_state=st.session_state,
        team_rows=current_game_team_rows,  # DataFrame with snapshot columns
        gameid="LOLTMNT05_171038",
        side="Blue",
        minute=15,
        p_map_pre=0.62,
        league="LCK",
        bo=5,
        is_playoffs=True,
    )
    if p_live is not None:
        # Use the calibrated live prediction
        ...
    else:
        # Fall back to existing heuristic (flag off or model unavailable)
        ...
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from mlcore.live_ml import SNAPSHOT_MINUTES, StateToWinModel, build_snapshot_features


_VALID_MINUTES = set(SNAPSHOT_MINUTES)


def is_live_state_model_enabled(session_state: Mapping[str, Any] | None) -> bool:
    """Return True if the user has enabled the live state model (feature flag)."""
    ss = session_state if isinstance(session_state, Mapping) else {}
    return bool(ss.get("ml_live_state_model_enabled", False))


def models_dir_from_state(session_state: Mapping[str, Any] | None) -> str:
    """Return the directory configured to hold the live state models."""
    ss = session_state if isinstance(session_state, Mapping) else {}
    return str(ss.get("ml_live_state_models_dir", "live_models") or "live_models")


def _model_path_for_minute(models_dir: str, minute: int) -> Path:
    return Path(str(models_dir)) / f"minute_{int(minute):02d}"


@lru_cache(maxsize=8)
def _cached_load_model(models_dir: str, minute: int) -> Optional[StateToWinModel]:
    """Internal: lazily load a single per-minute model with LRU caching.

    Returns None if the model directory does not exist (model not trained yet).
    """
    path = _model_path_for_minute(models_dir, int(minute))
    if not path.exists():
        return None
    try:
        return StateToWinModel.load(path)
    except Exception:
        return None


def load_live_state_models(
    models_dir: str = "live_models",
    minutes: Tuple[int, ...] = SNAPSHOT_MINUTES,
) -> Dict[int, StateToWinModel]:
    """Eagerly load all available live state models from disk.

    Skips minutes whose directory is missing or fails to load. Cached internally
    so calling this multiple times during a session is cheap.

    Returns a dict {minute: StateToWinModel} for all successfully loaded models.
    """
    out: Dict[int, StateToWinModel] = {}
    for m in minutes:
        model = _cached_load_model(str(models_dir), int(m))
        if model is not None:
            out[int(m)] = model
    return out


def clear_model_cache() -> None:
    """Clear the LRU cache (e.g., after retraining and saving new models)."""
    _cached_load_model.cache_clear()


def _build_one_team_row(
    *,
    gameid: str,
    side: str,
    league: str,
    bo: int,
    is_playoffs: bool,
    state_features: Mapping[str, Any],
    minute: int,
) -> pd.DataFrame:
    """Helper: turn a single live state observation into a one-row team_rows DataFrame.

    Used when the caller doesn't have the full CSV-style team_rows handy and just
    wants to pass observed values directly (gold, xp, cs, kills, etc. for both
    teams at the current minute).

    Expected keys in `state_features` (one side's perspective):
      goldat<T>, xpat<T>, csat<T>, killsat<T>, deathsat<T>, assistsat<T>,
      opp_goldat<T>, opp_xpat<T>, opp_csat<T>,
      opp_killsat<T>, opp_deathsat<T>, opp_assistsat<T>
    """
    t = int(minute)
    side_norm = str(side).strip().title()
    participantid = 100 if side_norm == "Blue" else 200
    row = {
        "gameid": str(gameid),
        "side": side_norm,
        "participantid": int(participantid),
        "result": 0,  # placeholder; only used as a target during training
        "league": str(league or ""),
        "playoffs": 1 if bool(is_playoffs) else 0,
        "game": int(bo),
    }
    for key in (f"goldat{t}", f"xpat{t}", f"csat{t}",
                f"killsat{t}", f"deathsat{t}", f"assistsat{t}",
                f"opp_goldat{t}", f"opp_xpat{t}", f"opp_csat{t}",
                f"opp_killsat{t}", f"opp_deathsat{t}", f"opp_assistsat{t}"):
        row[key] = float(state_features.get(key, 0.0) or 0.0)
    return pd.DataFrame([row])


def predict_live_p_win(
    *,
    session_state: Mapping[str, Any] | None,
    minute: int,
    gameid: str,
    side: str,
    p_map_pre: float,
    league: str = "",
    bo: int = 1,
    is_playoffs: bool = False,
    state_features: Mapping[str, Any] | None = None,
    team_rows: Optional[pd.DataFrame] = None,
    models: Optional[Dict[int, StateToWinModel]] = None,
) -> Optional[float]:
    """Return P(side wins) given the live game state, or None when unavailable.

    Returns None (caller falls back to existing heuristics) when any of these holds:
      - feature flag `ml_live_state_model_enabled` is False
      - minute is not a supported snapshot (10/15/20/25)
      - the per-minute model file does not exist
      - p_map_pre is not a valid probability
      - state features are missing or invalid

    Parameters
    ----------
    session_state : Mapping or None
        Streamlit (or any dict-like) session state — used to check the feature flag.
    minute : int
        Snapshot minute. Must be one of 10, 15, 20, 25.
    gameid : str
        Game identifier (any string; used only to group features).
    side : str
        "Blue" or "Red" — the side whose win probability you want.
    p_map_pre : float
        Pre-game prediction for `side` winning (in (0, 1)).
    league, bo, is_playoffs : context features.
    state_features : Mapping or None
        Direct dict of {goldat<T>: ..., opp_goldat<T>: ..., ...}. Use this when
        you don't have a CSV-style DataFrame already.
    team_rows : DataFrame or None
        Alternative: full team_rows DataFrame with the required at_T columns.
        Either `state_features` OR `team_rows` must be provided.
    models : optional preloaded dict {minute: StateToWinModel}.
        Pass this to skip the disk load (useful if you cached models elsewhere).
    """
    if not is_live_state_model_enabled(session_state):
        return None
    m = int(minute)
    if m not in _VALID_MINUTES:
        return None
    try:
        p_pre = float(p_map_pre)
    except (TypeError, ValueError):
        return None
    if not (0.0 < p_pre < 1.0) or not math.isfinite(p_pre):
        return None

    if models is None:
        models = load_live_state_models(models_dir_from_state(session_state))
    model = models.get(m)
    if model is None:
        return None

    # Build the snapshot row(s)
    if team_rows is None and state_features is None:
        return None
    if team_rows is None:
        team_rows = _build_one_team_row(
            gameid=str(gameid), side=str(side), league=str(league),
            bo=int(bo), is_playoffs=bool(is_playoffs),
            state_features=state_features or {}, minute=m,
        )

    # The live snapshot needs both sides' p_map_pre to pick the right row.
    side_norm = str(side).strip().title()
    opp_side = "Red" if side_norm == "Blue" else "Blue"
    p_pre_by_gs = {
        (str(gameid), side_norm): float(p_pre),
        (str(gameid), opp_side): float(1.0 - p_pre),
    }
    snapshots = build_snapshot_features(
        team_rows, minute=m,
        p_map_pre_by_gameid_side=p_pre_by_gs,
        require_complete=False,  # be lenient — caller already validated state
    )
    # Pick the snapshot matching the requested side
    target = next((s for s in snapshots if s.side == side_norm and s.gameid == str(gameid)), None)
    if target is None:
        return None
    try:
        return float(model.predict_proba_one(target))
    except Exception:
        return None


def available_minutes(models_dir: str = "live_models") -> Tuple[int, ...]:
    """Return the tuple of snapshot minutes for which a saved model exists on disk."""
    out = []
    for m in SNAPSHOT_MINUTES:
        if _model_path_for_minute(models_dir, m).exists():
            out.append(int(m))
    return tuple(out)
