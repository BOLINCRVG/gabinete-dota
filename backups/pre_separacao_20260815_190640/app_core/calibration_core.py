"""Unified market calibration/fallback helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def calibrate_market_odds(
    *,
    odd_model: Any,
    odd_hist: Any,
    stats_pct: Any,
    max_odd: float,
    require_model: bool,
    odd_mix_fn: Callable[[Any, Any, Any], Optional[float]],
) -> Dict[str, Any]:
    om = _as_float(odd_model)
    oh = _as_float(odd_hist)
    sc = _as_float(stats_pct)
    mx = _as_float(max_odd) or 999.0

    if bool(require_model) and (om is None or om > mx):
        return {
            "ok": False,
            "reason": "model_missing_or_above_max",
            "source": "none",
            "odd_model": om,
            "odd_hist": oh,
            "odd_mix": None,
        }

    odd_mix = None
    try:
        odd_mix = odd_mix_fn(om, oh, sc)
    except Exception:
        odd_mix = None
    odd_mix = _as_float(odd_mix)

    if odd_mix is not None and odd_mix <= mx:
        return {
            "ok": True,
            "reason": "mix",
            "source": "mix",
            "odd_model": om,
            "odd_hist": oh,
            "odd_mix": odd_mix,
        }

    # explicit fallback order: model -> hist
    if om is not None and om <= mx:
        return {
            "ok": True,
            "reason": "fallback_model",
            "source": "model",
            "odd_model": om,
            "odd_hist": oh,
            "odd_mix": om,
        }
    if oh is not None and oh <= mx:
        return {
            "ok": True,
            "reason": "fallback_hist",
            "source": "hist",
            "odd_model": om,
            "odd_hist": oh,
            "odd_mix": oh,
        }

    return {
        "ok": False,
        "reason": "no_valid_odd_under_max",
        "source": "none",
        "odd_model": om,
        "odd_hist": oh,
        "odd_mix": None,
    }
