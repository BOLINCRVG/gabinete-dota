"""Winner market pricing helpers (map/series/bo3/bo5)."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional


def _as_finite_or_nan(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return float(x) if math.isfinite(x) else float("nan")


def _odd_safe(
    p: float,
    *,
    odd_from_prob_fn: Optional[Callable[[float], Optional[float]]],
) -> Optional[float]:
    if not callable(odd_from_prob_fn):
        return None
    if not math.isfinite(float(p)):
        return None
    try:
        o = odd_from_prob_fn(float(p))
        if o is None:
            return None
        ov = float(o)
        return ov if math.isfinite(ov) else None
    except Exception:
        return None


def _series_safe(
    p_map: float,
    bo: int,
    *,
    prob_win_series_fn: Optional[Callable[[float, int], float]],
) -> float:
    if not callable(prob_win_series_fn):
        return float("nan")
    if not math.isfinite(float(p_map)):
        return float("nan")
    try:
        v = float(prob_win_series_fn(float(p_map), int(bo)))
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def compute_winner_pricing_bundle(
    *,
    p_team_a_map: Any,
    p_team_a_series: Any = float("nan"),
    bo_for_series: int = 1,
    prob_win_series_fn: Optional[Callable[[float, int], float]] = None,
    odd_from_prob_fn: Optional[Callable[[float], Optional[float]]] = None,
) -> Dict[str, Any]:
    """Compute reusable winner pricing values for UI blocks.

    Returns probabilities for team A/B and fair decimal odds for:
    - map
    - selected series
    - bo3
    - bo5
    """
    p_a_map = _as_finite_or_nan(p_team_a_map)
    p_b_map = float(1.0 - p_a_map) if math.isfinite(p_a_map) else float("nan")

    p_a_series = _as_finite_or_nan(p_team_a_series)
    if not math.isfinite(p_a_series):
        p_a_series = _series_safe(
            p_a_map,
            int(bo_for_series),
            prob_win_series_fn=prob_win_series_fn,
        )
    p_b_series = float(1.0 - p_a_series) if math.isfinite(p_a_series) else float("nan")

    p_a_bo3 = _series_safe(p_a_map, 3, prob_win_series_fn=prob_win_series_fn)
    p_b_bo3 = float(1.0 - p_a_bo3) if math.isfinite(p_a_bo3) else float("nan")
    p_a_bo5 = _series_safe(p_a_map, 5, prob_win_series_fn=prob_win_series_fn)
    p_b_bo5 = float(1.0 - p_a_bo5) if math.isfinite(p_a_bo5) else float("nan")

    return {
        "pA_map": p_a_map,
        "pB_map": p_b_map,
        "pA_series": p_a_series,
        "pB_series": p_b_series,
        "pA_bo3": p_a_bo3,
        "pB_bo3": p_b_bo3,
        "pA_bo5": p_a_bo5,
        "pB_bo5": p_b_bo5,
        "oddA_map": _odd_safe(p_a_map, odd_from_prob_fn=odd_from_prob_fn),
        "oddB_map": _odd_safe(p_b_map, odd_from_prob_fn=odd_from_prob_fn),
        "oddA_series": _odd_safe(p_a_series, odd_from_prob_fn=odd_from_prob_fn),
        "oddB_series": _odd_safe(p_b_series, odd_from_prob_fn=odd_from_prob_fn),
        "oddA_bo3": _odd_safe(p_a_bo3, odd_from_prob_fn=odd_from_prob_fn),
        "oddB_bo3": _odd_safe(p_b_bo3, odd_from_prob_fn=odd_from_prob_fn),
        "oddA_bo5": _odd_safe(p_a_bo5, odd_from_prob_fn=odd_from_prob_fn),
        "oddB_bo5": _odd_safe(p_b_bo5, odd_from_prob_fn=odd_from_prob_fn),
    }
