"""Feature extraction for live state-to-win models.

For each game-state snapshot at minute T (10/15/20/25), we extract:

  - gold_diff_at_t (blue - red)
  - xp_diff_at_t
  - cs_diff_at_t
  - kills_diff_at_t (sum of player kills_at_t per team)
  - p_map_pre (the pre-game prediction; treated as a prior)
  - z_pre = logit(p_map_pre)
  - league_tier (encoded as int)
  - bo (int)
  - playoffs (0/1)
  - minute (constant per snapshot model)
  - side_blue (1 if predicting blue's win; we standardize to blue)

The model output is P(blue wins | snapshot at minute t). For "red wins"
predictions, the caller can swap sides and invert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mlcore.tiers import tier_for_league


SNAPSHOT_MINUTES: Tuple[int, ...] = (10, 15, 20, 25)


# Feature columns produced by build_snapshot_features (order matters for the model).
_FEATURE_COLS: Tuple[str, ...] = (
    "z_pre",            # logit of pre-game prediction
    "gold_diff",        # blue - red gold at minute t
    "xp_diff",
    "cs_diff",
    "kills_diff",       # team kills sum
    "deaths_diff",
    "assists_diff",
    "league_tier",
    "bo",
    "playoffs",
)


_TIER_INT: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "MINOR": 3, "ERL": 4, "OTHER": 5,
}


def _encode_tier(league: str) -> int:
    return _TIER_INT.get(tier_for_league(str(league or "")), 5)


def feature_names() -> Tuple[str, ...]:
    """Return the canonical ordered list of features."""
    return _FEATURE_COLS


@dataclass(frozen=True)
class LiveSnapshot:
    """Per-row, per-snapshot feature vector + label for training."""

    gameid: str
    minute: int
    side: str  # "Blue" or "Red"
    p_pre: float
    features: np.ndarray  # shape (len(_FEATURE_COLS),)
    y: int  # 1 if `side` won the game


def _logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    import math
    return math.log(p / (1.0 - p))


def build_snapshot_features(
    team_rows: pd.DataFrame,
    *,
    minute: int,
    p_map_pre_by_gameid_side: Dict[Tuple[str, str], float],
    require_complete: bool = True,
) -> List[LiveSnapshot]:
    """Build feature rows for a given snapshot minute.

    Parameters
    ----------
    team_rows : DataFrame
        Oracle's Elixir team-level rows (participantid 100 or 200), with the
        relevant `goldat<T>`, `xpat<T>`, `csat<T>`, `killsat<T>`, etc. columns.
    minute : int
        Snapshot minute (10, 15, 20, 25).
    p_map_pre_by_gameid_side : Dict[(gameid, side), float]
        Pre-game prediction (P that this side wins) for each (gameid, side) pair.
    require_complete : bool
        If True, drop rows where any of the at-T features is missing.
    """
    t = int(minute)
    if t not in SNAPSHOT_MINUTES:
        raise ValueError(f"minute must be one of {SNAPSHOT_MINUTES}; got {t}")

    needed = [f"goldat{t}", f"xpat{t}", f"csat{t}", f"killsat{t}",
              f"deathsat{t}", f"assistsat{t}",
              f"opp_goldat{t}", f"opp_xpat{t}", f"opp_csat{t}",
              f"opp_killsat{t}", f"opp_deathsat{t}", f"opp_assistsat{t}"]

    df = team_rows.copy()
    if require_complete:
        df = df.dropna(subset=needed)
    df = df[df["participantid"].isin([100, 200])].copy()

    out: List[LiveSnapshot] = []
    for row in df.itertuples(index=False):
        gid = str(getattr(row, "gameid", "") or "")
        side_raw = str(getattr(row, "side", "") or "").strip().title()
        if side_raw not in {"Blue", "Red"}:
            # Some CSVs use participantid 100 = Blue, 200 = Red
            side_raw = "Blue" if int(getattr(row, "participantid", 100)) == 100 else "Red"
        if not gid:
            continue
        p_pre = p_map_pre_by_gameid_side.get((gid, side_raw))
        if p_pre is None or not (0 < float(p_pre) < 1):
            continue
        result = int(getattr(row, "result", 0) or 0)
        # All "diff" features are (mine - opp), from this side's perspective.
        gold_d = float(getattr(row, f"goldat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_goldat{t}", 0.0) or 0.0)
        xp_d = float(getattr(row, f"xpat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_xpat{t}", 0.0) or 0.0)
        cs_d = float(getattr(row, f"csat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_csat{t}", 0.0) or 0.0)
        kills_d = float(getattr(row, f"killsat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_killsat{t}", 0.0) or 0.0)
        deaths_d = float(getattr(row, f"deathsat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_deathsat{t}", 0.0) or 0.0)
        assists_d = float(getattr(row, f"assistsat{t}", 0.0) or 0.0) - float(getattr(row, f"opp_assistsat{t}", 0.0) or 0.0)
        league = str(getattr(row, "league", "") or "")
        bo_raw = getattr(row, "game", 1)
        bo = int(bo_raw) if str(bo_raw).strip().isdigit() else 1
        playoffs = 1 if str(getattr(row, "playoffs", "") or "").strip().lower() in {"1", "true", "playoffs"} else 0

        features = np.array([
            _logit(float(p_pre)),
            gold_d, xp_d, cs_d,
            kills_d, deaths_d, assists_d,
            float(_encode_tier(league)),
            float(bo),
            float(playoffs),
        ], dtype=float)

        out.append(LiveSnapshot(
            gameid=gid, minute=t, side=side_raw, p_pre=float(p_pre),
            features=features, y=int(result),
        ))

    return out
